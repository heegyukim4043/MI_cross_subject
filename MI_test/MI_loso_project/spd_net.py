"""
SPDNet-based classifier for Motor Imagery EEG.

Replaces the CSP layer from CSP-Net with Riemannian SPD manifold layers:
  EEG (B, C, T)
    → CovarianceLayer   : (B, C, C)   regularised covariance
    → BiMapLayer        : (B, n, n)   learnable projection on SPD manifold
    → ReEigLayer        : (B, n, n)   eigenvalue rectification (keeps SPD)
    → LogMapLayer       : (B, n, n)   matrix log  → tangent space
    → VectorizeLayer    : (B, n*(n+1)//2)
    → MLP head          : (B, 2)

Reference:
  Huang & Van Gool, "A Riemannian Network for SPD Matrix Learning", AAAI 2017
  Brooks et al., "Riemannian Batch Normalization for SPD Neural Networks", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SPD Manifold Layers
# ─────────────────────────────────────────────────────────────────────────────

class CovarianceLayer(nn.Module):
    """
    Compute regularised sample covariance from EEG epochs.
    Input : (B, C, T)
    Output: (B, C, C)   symmetric positive definite
    """
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):                       # (B, C, T)
        B, C, T = x.shape
        # Zero-mean per trial
        x = x - x.mean(dim=-1, keepdim=True)
        # Sample covariance
        cov = torch.bmm(x, x.transpose(-2, -1)) / (T - 1)   # (B, C, C)
        # Stronger Tikhonov regularisation — trace-normalised
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        eye   = torch.eye(C, dtype=x.dtype, device=x.device).unsqueeze(0)
        cov   = cov + self.eps * (trace / C) * eye
        return cov


class BiMapLayer(nn.Module):
    """
    Learnable projection  W^T Σ W  on the SPD manifold.
    Input : (B, C_in, C_in)
    Output: (B, C_out, C_out)
    W is a (C_in × C_out) semi-orthogonal matrix (Stiefel manifold).
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(c_in, c_out))
        nn.init.orthogonal_(self.W)

    def forward(self, x):                       # (B, C_in, C_in)
        W = self.W                              # (C_in, C_out)
        # W^T X W  — broadcast over batch
        out = W.t().unsqueeze(0) @ x @ W.unsqueeze(0)       # (B, C_out, C_out)
        return out


class ReEigLayer(nn.Module):
    """
    Eigenvalue rectification: clamp eigenvalues to [eps, ∞) to keep SPD.
    Input / Output: (B, C, C)
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        L, V = torch.linalg.eigh(x)            # L: (B, C), V: (B, C, C)
        L = torch.clamp(L, min=self.eps)
        return V @ torch.diag_embed(L) @ V.transpose(-2, -1)


class LogMapLayer(nn.Module):
    """
    Matrix logarithm: maps SPD matrix to its tangent space at the identity.
    Input / Output: (B, C, C)
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        L, V = torch.linalg.eigh(x)
        L = torch.log(torch.clamp(L, min=self.eps))
        return V @ torch.diag_embed(L) @ V.transpose(-2, -1)


class VectorizeLayer(nn.Module):
    """
    Extract upper triangular (including diagonal) elements.
    Input : (B, C, C)
    Output: (B, C*(C+1)//2)
    """
    def forward(self, x):
        C = x.shape[-1]
        idx = torch.triu_indices(C, C, device=x.device)    # (2, C*(C+1)//2)
        return x[:, idx[0], idx[1]]


# ─────────────────────────────────────────────────────────────────────────────
# Full SPDNet model
# ─────────────────────────────────────────────────────────────────────────────

class SPDNet(nn.Module):
    """
    SPDNet classifier for 2-class Motor Imagery.

    Args:
        n_channels : number of EEG channels (C)
        n_filters  : BiMap output dimension  (default 32)
        n_classes  : number of output classes (default 2)
        dropout    : dropout probability
    """
    def __init__(
        self,
        n_channels: int,
        n_filters:  int = 32,
        n_classes:  int = 2,
        dropout:    float = 0.5,
    ):
        super().__init__()

        vec_dim = n_filters * (n_filters + 1) // 2   # 32 → 528

        self.cov     = CovarianceLayer(eps=1e-5)
        self.bimap   = BiMapLayer(n_channels, n_filters)
        self.reeig   = ReEigLayer(eps=1e-4)
        self.logmap  = LogMapLayer(eps=1e-7)
        self.vec     = VectorizeLayer()

        self.classifier = nn.Sequential(
            nn.Linear(vec_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):               # x: (B, C, T)
        x = self.cov(x)                 # (B, C, C)
        x = self.bimap(x)               # (B, n_filters, n_filters)
        x = self.reeig(x)               # (B, n_filters, n_filters)
        x = self.logmap(x)              # (B, n_filters, n_filters)
        x = self.vec(x)                 # (B, vec_dim)
        x = self.classifier(x)          # (B, n_classes)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Cho2017: 64 ch, 128 Hz × 2 s = 256 samples
    B, C, T = 8, 64, 256
    x = torch.randn(B, C, T)
    model = SPDNet(n_channels=C, n_filters=32)
    out = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")

    # Lee2019: 62 ch, 100 Hz × 2 s = 200 samples
    B2, C2, T2 = 8, 62, 200
    x2 = torch.randn(B2, C2, T2)
    model2 = SPDNet(n_channels=C2, n_filters=32)
    out2 = model2(x2)
    print(f"\nLee2019 input  : {x2.shape}")
    print(f"Lee2019 output : {out2.shape}")
