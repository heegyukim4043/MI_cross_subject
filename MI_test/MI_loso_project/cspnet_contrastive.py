"""
CSPNet-Contrastive: CSPNet encoder + Supervised Contrastive auxiliary head.

Architecture
------------
  EEG Input (B, C, T)
       │
  ┌────┴──────────────────┐
  │   CSPNet Encoder      │   ← shared (identical to CSPNet body)
  └────┬──────────────────┘
       │  z ∈ R^n_flat
       ├──────────────────────────────┐
       ▼                              ▼
  ┌──────────┐              ┌──────────────────────┐
  │ MI Head  │              │  Contrastive Head    │
  │ Linear   │              │  MLP → L2-norm → h  │
  │ CE loss  │              │  SupCon loss         │
  └──────────┘              └──────────────────────┘

Joint loss (stage 2)
--------------------
  L = L_cls + λ_con * L_supcon(h, y)

  L_cls   : CrossEntropy on MI labels
  L_supcon: Supervised Contrastive — same class pulled together,
            different class pushed apart, purely driven by MI labels.
            No external quality/domain labels needed.

Two-stage training
------------------
  Stage 1: encoder + MI head only → stable MI features
  Stage 2: MI + contrastive (no encoder freeze; contrastive is
           complementary to classification, not adversarial)

SupCon temperature
------------------
  Lower T → sharper distribution (harder negatives weighted more).
  Default 0.07 (SimCLR/SupCon paper). For EEG with high intra-class
  variance, 0.1–0.2 may be more stable.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cspnet import CSPLayer, compute_csp_filters


# ─────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Loss
# ─────────────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Parameters
    ----------
    temperature : float
        Logit scale.  Default 0.07.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features : (B, D) — must be L2-normalised before passing
        labels   : (B,)  — class indices
        """
        B = features.shape[0]
        device = features.device

        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        labels_col = labels.view(-1, 1)
        mask_pos = (labels_col == labels_col.T) & ~mask_self   # same-class, not self

        if mask_pos.float().sum() == 0:
            return features.sum() * 0.0   # no positives: zero loss, keep grad graph

        # Numerically stable log-softmax over all non-self pairs
        sim = sim.masked_fill(mask_self, -1e9)
        sim_max = sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim - sim_max)
        log_prob = (sim - sim_max) - torch.log(
            exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-8)
        )

        n_pos = mask_pos.float().sum(dim=1).clamp_min(1.0)
        loss = -(log_prob * mask_pos.float()).sum(dim=1) / n_pos
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# CSPNet-Contrastive model
# ─────────────────────────────────────────────────────────────────────────────

class CSPNetContrastive(nn.Module):
    """CSPNet encoder + MI classifier + supervised contrastive projection head.

    Parameters
    ----------
    n_channels, n_times  : EEG shape
    n_classes            : MI classes (default 2)
    n_csp                : CSP spatial filters (default 8)
    F1, F2               : temporal / separable conv channels
    kernel_length        : temporal conv kernel (None → auto)
    dropout              : dropout probability (default 0.25)
    trainable_csp        : CSP weights receive gradients (default True)
    proj_dim             : projection head output dimension (default 64)
    proj_hidden          : projection head hidden dimension (default 128)
    temperature          : SupCon temperature (default 0.07)
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        n_csp: int = 8,
        F1: int = 8,
        F2: int = 16,
        kernel_length: int = None,
        dropout: float = 0.25,
        trainable_csp: bool = True,
        proj_dim: int = 64,
        proj_hidden: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()

        n_csp = min(n_csp, n_channels)
        if kernel_length is None:
            kernel_length = max(16, n_times // 4)
        if kernel_length % 2 == 0:
            kernel_length += 1
        pad_t = kernel_length // 2

        mid = F1 * n_csp
        sep_kern = max(8, n_times // 16)
        if sep_kern % 2 == 0:
            sep_kern += 1
        sep_pad = sep_kern // 2

        # ── Shared encoder ────────────────────────────────────────────────────
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, pad_t), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.csp_layer = CSPLayer(n_channels, n_csp=n_csp, trainable=trainable_csp)
        self.bn2   = nn.BatchNorm2d(mid)
        self.act2  = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(dropout)

        self.sep_dw = nn.Conv2d(mid, mid, (1, sep_kern),
                                padding=(0, sep_pad), groups=mid, bias=False)
        self.sep_pw = nn.Conv2d(mid, F2, (1, 1), bias=False)
        self.bn3   = nn.BatchNorm2d(F2)
        self.act3  = nn.ELU()
        self.pool3 = nn.AvgPool2d((1, 8))
        self.drop3 = nn.Dropout(dropout)

        with torch.no_grad():
            probe = torch.zeros(1, 1, n_channels, n_times)
            n_flat = self._encode(probe).shape[1]

        # ── MI classifier ─────────────────────────────────────────────────────
        self.classifier = nn.Linear(n_flat, n_classes)

        # ── Contrastive projection head ───────────────────────────────────────
        self.proj_head = nn.Sequential(
            nn.Linear(n_flat, proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim),
        )
        self.supcon = SupConLoss(temperature=temperature)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, C, T) → z : (B, n_flat)"""
        x = self.temporal_conv(x)
        x = self.csp_layer(x)
        x = self.bn2(x); x = self.act2(x); x = self.pool2(x); x = self.drop2(x)
        x = self.sep_dw(x); x = self.sep_pw(x)
        x = self.bn3(x); x = self.act3(x); x = self.pool3(x); x = self.drop3(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor):
        """
        Training  : returns (logits, h_normalized)
        Inference : returns logits only
        """
        z = self._encode(x.unsqueeze(1))
        logits = self.classifier(z)
        if self.training:
            h = F.normalize(self.proj_head(z), dim=1)
            return logits, h
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# CSP filter initialization helper
# ─────────────────────────────────────────────────────────────────────────────

def fit_csp_layer_contrastive(
    model: CSPNetContrastive,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    n_csp = model.csp_layer.n_csp
    W = compute_csp_filters(X_train, y_train, n_filters=n_csp)
    model.csp_layer.init_from_numpy(W)
    print(f"    [CSPNetContrastive] CSP init (n_csp={n_csp}, "
          f"trainable={isinstance(model.csp_layer.W, nn.Parameter)})")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== CSPNetContrastive sanity check ===\n")

    for ds, C, T in [("Cho2017", 64, 257), ("Lee2019", 62, 201)]:
        model = CSPNetContrastive(n_channels=C, n_times=T)
        model.eval()
        x = torch.randn(8, C, T)
        with torch.no_grad():
            logits = model(x)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{ds}: logits={tuple(logits.shape)}  params={n_p:,}")

    print("\n=== Training forward + loss ===")
    C, T, B = 64, 257, 16
    model = CSPNetContrastive(n_channels=C, n_times=T, temperature=0.07)
    model.train()
    x = torch.randn(B, C, T)
    y = torch.tensor([0]*8 + [1]*8)

    logits, h = model(x)
    cls_loss = F.cross_entropy(logits, y)
    con_loss = model.supcon(h, y)
    total = cls_loss + 0.5 * con_loss
    total.backward()

    print(f"  logits={tuple(logits.shape)}  h={tuple(h.shape)}")
    print(f"  cls={cls_loss.item():.4f}  con={con_loss.item():.4f}  total={total.item():.4f}")
    print(f"  h norm (mean): {h.norm(dim=1).mean().item():.4f}  (should be 1.0)")
    print("  OK" if h.norm(dim=1).mean().item() > 0.99 else "  WARNING: h not normalized")
