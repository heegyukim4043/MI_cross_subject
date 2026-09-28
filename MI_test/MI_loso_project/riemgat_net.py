"""
RiemGATNet: Riemannian Covariance -> Dynamic Graph Attention -> Deep Conv Net
for 2-class Motor Imagery EEG classification.

Pipeline:
  EEG (B, C, T)
    |-- TemporalEncoder                : (B,C,T) -> (B,C,d_node)  [node features]
    `-- CovarianceLayer -> LogMapLayer : (B,C,T) -> (B,C,C)       [dynamic adjacency]

  DynamicGATLayer x n_gat_layers
    Attention logits = Q*K^T/sqrt(d) + scale_h * adj_z
    Output: (B, C, d_gat)

  DeepConvBlock  [1D CNN along channel/node dimension]
    (B, C, d_gat) -> (B, d_conv)

  MLP Classifier -> (B, n_classes)

Key idea:
  The log-mapped Riemannian covariance serves as a *dynamic* per-trial graph
  adjacency.  GAT learns head-specific attention that is BIASED by this
  functional connectivity, so the network can up-weight or down-weight
  the Riemannian prior selectively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Riemannian building blocks
# ---------------------------------------------------------------------------

class CovarianceLayer(nn.Module):
    """
    Regularised sample covariance from EEG epochs.
    Input : (B, C, T)
    Output: (B, C, C)  symmetric positive definite
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):                        # (B, C, T)
        B, C, T = x.shape
        x = x - x.mean(dim=-1, keepdim=True)
        cov = x @ x.transpose(-2, -1) / (T - 1) # (B, C, C)
        # Trace-normalised Tikhonov regularisation
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        eye   = torch.eye(C, dtype=x.dtype, device=x.device).unsqueeze(0)
        return cov + self.eps * (trace / C) * eye


class LogMapLayer(nn.Module):
    """
    Matrix logarithm: SPD -> tangent space at identity.
    Input / Output: (B, C, C)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        L, V = torch.linalg.eigh(x)
        L = torch.log(torch.clamp(L, min=self.eps))
        return V @ torch.diag_embed(L) @ V.transpose(-2, -1)


# ---------------------------------------------------------------------------
# Temporal Encoder  (shared CNN per channel)
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """
    Shared 1D CNN applied independently to each EEG channel.
    Extracts per-channel temporal features that serve as GAT node features.

    Input : (B, C, T)
    Output: (B, C, d_node)
    """
    def __init__(self, d_node: int = 64, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: broad temporal filter
            nn.Conv1d(1, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(4),

            # Layer 2: mid-range patterns
            nn.Conv1d(32, 64, kernel_size=10, padding=5),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(4),

            # Layer 3: fine features -> d_node
            nn.Conv1d(64, d_node, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_node),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),   # -> (B*C, d_node, 1)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):              # (B, C, T)
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)    # (B*C, 1, T)
        x = self.net(x).squeeze(-1)   # (B*C, d_node)
        x = self.drop(x)
        return x.reshape(B, C, -1)    # (B, C, d_node)


# ---------------------------------------------------------------------------
# Dynamic Graph Attention Layer
# ---------------------------------------------------------------------------

class DynamicGATLayer(nn.Module):
    """
    Multi-head self-attention on a fully-connected channel graph,
    biased by the Riemannian log-covariance adjacency.

    Attention logit for edge (i->j), head h:
        e_ij^h = (Q_i^h . K_j^h) / sqrt(d_h)  +  alpha_h * adj_z_ij

    where adj_z is the z-score normalised log-covariance and alpha_h is a
    learnable per-head scalar (initialised to 0 -> no prior at start).

    Input:
        x   : (B, C, d_in)   node features
        adj : (B, C, C)      log-mapped covariance (symmetric)
    Output:
        (B, C, d_out)
    """
    def __init__(
        self,
        d_in:    int,
        d_out:   int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.H   = n_heads
        self.d_h = d_out // n_heads

        self.W_q = nn.Linear(d_in,  d_out, bias=False)
        self.W_k = nn.Linear(d_in,  d_out, bias=False)
        self.W_v = nn.Linear(d_in,  d_out, bias=False)

        # Learnable per-head adjacency scale (init=0 -> no prior influence at t=0)
        self.adj_scale = nn.Parameter(torch.zeros(n_heads))

        self.out_proj = nn.Linear(d_out, d_out)
        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)
        self.res  = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, adj):         # x:(B,C,d_in), adj:(B,C,C)
        B, C, _ = x.shape
        H, d_h  = self.H, self.d_h

        # Project to Q, K, V
        Q = self.W_q(x).reshape(B, C, H, d_h).permute(0, 2, 1, 3)  # (B,H,C,d_h)
        K = self.W_k(x).reshape(B, C, H, d_h).permute(0, 2, 1, 3)
        V = self.W_v(x).reshape(B, C, H, d_h).permute(0, 2, 1, 3)

        # Scaled dot-product attention scores
        scores = (Q @ K.transpose(-2, -1)) / (d_h ** 0.5)  # (B,H,C,C)

        # Z-score normalise adjacency (per sample) to stabilise scale
        adj_flat = adj.reshape(B, -1)
        adj_z = (adj - adj_flat.mean(-1)[:, None, None]) / (
                  adj_flat.std(-1)[:, None, None] + 1e-8)   # (B,C,C)

        # Add learnable-scaled adjacency bias per head
        scale  = self.adj_scale.reshape(1, H, 1, 1)         # (1,H,1,1)
        scores = scores + scale * adj_z.unsqueeze(1)         # (B,H,C,C)

        attn   = F.softmax(scores, dim=-1)                   # (B,H,C,C)
        attn   = self.attn_drop(attn)

        out = attn @ V                                       # (B,H,C,d_h)
        out = out.permute(0, 2, 1, 3).reshape(B, C, H * d_h)# (B,C,d_out)
        out = self.out_proj(out)

        return self.norm(out + self.res(x))                  # residual + LN


# ---------------------------------------------------------------------------
# Deep Conv Block  (1D CNN along the spatial/channel dimension)
# ---------------------------------------------------------------------------

class DeepConvBlock(nn.Module):
    """
    1D CNN applied along the channel (node) dimension after GAT.
    Treats channels as a spatial sequence and extracts higher-order patterns.

    Input : (B, C, d_gat)  -> transpose -> (B, d_gat, C)
    Output: (B, d_out)
    """
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in,      d_in * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_in * 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(d_in * 2, d_in * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_in * 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(d_in * 2, d_out,    kernel_size=3, padding=1),
            nn.BatchNorm1d(d_out),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),       # global spatial pooling
        )

    def forward(self, x):               # (B, C, d_in)
        x = x.transpose(1, 2)           # (B, d_in, C)
        return self.net(x).squeeze(-1)  # (B, d_out)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class RiemGATNet(nn.Module):
    """
    RiemGATNet: Riemannian Covariance + Dynamic Graph Attention + Deep Conv
    for 2-class Motor Imagery EEG.

    Args:
        n_channels   : EEG channels (C)
        n_times      : time samples (T)  -- unused at runtime, kept for docs
        n_classes    : output classes (default 2)
        d_node       : temporal encoder output dimension
        d_gat        : GAT hidden dimension  (must be divisible by n_heads)
        n_gat_layers : number of stacked GAT layers
        n_heads      : attention heads per GAT layer
        d_conv       : DeepConvBlock output dimension
        dropout      : dropout rate for classifier / deep conv
    """
    def __init__(
        self,
        n_channels:   int,
        n_times:      int   = 0,   # not used structurally; for documentation
        n_classes:    int   = 2,
        d_node:       int   = 64,
        d_gat:        int   = 64,
        n_gat_layers: int   = 3,
        n_heads:      int   = 4,
        d_conv:       int   = 128,
        dropout:      float = 0.5,
    ):
        super().__init__()

        # Riemannian branch -> dynamic adjacency
        self.cov    = CovarianceLayer(eps=1e-5)
        self.logmap = LogMapLayer(eps=1e-6)

        # Temporal encoder -> node features
        self.temporal_enc = TemporalEncoder(d_node=d_node, dropout=0.25)
        self.node_proj    = nn.Linear(d_node, d_gat)

        # Stacked GAT layers
        self.gat_layers = nn.ModuleList([
            DynamicGATLayer(d_gat, d_gat, n_heads=n_heads, dropout=0.1)
            for _ in range(n_gat_layers)
        ])

        # Deep conv over channel (node) dimension
        self.deep_conv = DeepConvBlock(d_gat, d_conv, dropout)

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_conv, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):              # (B, C, T)
        # --- Riemannian adjacency ---
        cov = self.cov(x)              # (B, C, C)
        adj = self.logmap(cov)         # (B, C, C)  tangent space

        # --- Node features via temporal encoder ---
        h = self.temporal_enc(x)       # (B, C, d_node)
        h = self.node_proj(h)          # (B, C, d_gat)

        # --- GAT layers ---
        for gat in self.gat_layers:
            h = gat(h, adj)            # (B, C, d_gat)

        # --- Deep conv + global pool ---
        h = self.deep_conv(h)          # (B, d_conv)

        # --- Classify ---
        return self.classifier(h)      # (B, n_classes)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cho2017: 64 ch, 128 Hz x 2 s = 257 samples
    B, C, T = 8, 64, 257
    x = torch.randn(B, C, T, device=device)
    model = RiemGATNet(n_channels=C, n_times=T).to(device)
    out = model(x)
    print(f"[Cho2017]  input {tuple(x.shape)} -> output {tuple(out.shape)}")
    print(f"           params: {sum(p.numel() for p in model.parameters()):,}")

    # Lee2019: 67 ch, 100 Hz x 2 s = 201 samples
    B2, C2, T2 = 8, 67, 201
    x2 = torch.randn(B2, C2, T2, device=device)
    model2 = RiemGATNet(n_channels=C2, n_times=T2).to(device)
    out2 = model2(x2)
    print(f"\n[Lee2019]  input {tuple(x2.shape)} -> output {tuple(out2.shape)}")
    print(f"           params: {sum(p.numel() for p in model2.parameters()):,}")

    # Show adjacency influence (adj_scale per layer)
    print("\n[adj_scale per GAT layer (init=0)]")
    for i, gat in enumerate(model.gat_layers):
        print(f"  layer {i}: {gat.adj_scale.data.tolist()}")
