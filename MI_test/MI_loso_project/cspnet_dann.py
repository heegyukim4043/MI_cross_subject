"""
CSPNet-DANN: CSPNet + joint Domain-Adversarial + Quality discriminator.

Architecture
------------
  EEG Input (B, C, T)
       │
  ┌────┴──────────────────┐
  │   CSPNet Encoder      │   ← shared representation (identical to CSPNet body)
  │   Block1 + Block2     │
  │   + Block3 features   │
  └────┬──────────────────┘
       │  z ∈ R^n_flat
       ├─────────────────┐────────────────────┐
       ▼                 ▼                    ▼
  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐
  │ MI Head  │    │ Quality Head │    │  Domain Head    │
  │ (cls)    │    │ (clean/noisy)│    │  (GRL + subj)  │
  │ CE loss  │    │  BCE loss    │    │   CE loss       │
  └──────────┘    └──────────────┘    └─────────────────┘

Joint training objective
------------------------
  L = L_cls  +  λ_q * L_quality  +  λ_d * L_domain

  L_cls    : cross-entropy on MI labels  (primary)
  L_quality: BCE between quality_head sigmoid output and pre-computed quality labels
             (trial quality ~ AE reconstruction quality on tangent space)
  L_domain : cross-entropy on subject ID, applied AFTER a gradient reversal layer
             → encoder learns to be subject-invariant (DANN-style)

Quality labels
--------------
  Computed externally via TangentSpaceExtractor + TangentAutoEncoder before
  the main training loop and passed per trial as a float in [0, 1].

  quality_label = 1 / (1 + recon_error)   (high quality = low recon error)

GRL lambda scheduling
---------------------
  λ(p) = 2 / (1 + exp(-10 * p)) - 1,  p = epoch / total_epochs  ∈ [0, 1]
  This ramps λ from 0 → 1 over training, giving the encoder time to learn
  classification before domain adversarial pressure builds up.

Adversarial selection interpretation
-------------------------------------
  The GRL domain head forces the encoder to forget subject identity.
  This naturally down-weights subject-specific trials in representation space
  (option 3 in the user's description).
  The quality head provides an explicit clean/noisy signal that the encoder
  learns to align features with (options 1 & 2 via the shared backbone).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cspnet import CSPLayer, compute_csp_filters
from discriminator_selective import GradientReversal


# ─────────────────────────────────────────────────────────────────────────────
# GRL lambda schedule
# ─────────────────────────────────────────────────────────────────────────────

def grl_lambda(progress: float, exponent: float = 10.0) -> float:
    """progress ∈ [0, 1] → λ ∈ [0, 1] via sigmoid ramp.

    Lower exponent (e.g. 5.0) gives a slower, less aggressive ramp.
    """
    return 2.0 / (1.0 + math.exp(-exponent * progress)) - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# CSPNet-DANN
# ─────────────────────────────────────────────────────────────────────────────

class CSPNetDANN(nn.Module):
    """
    CSPNet encoder + three prediction heads trained end-to-end.

    Parameters
    ----------
    n_channels      : EEG channels C
    n_times         : time samples T
    n_subjects      : number of training subjects (for domain head output size)
    n_classes       : MI classes (default 2)
    n_csp           : CSP spatial filters (default 8)
    F1              : temporal filters (default 8)
    F2              : separable conv output channels (default 16)
    kernel_length   : temporal conv kernel; None → max(16, n_times // 4)
    dropout         : dropout probability (default 0.25)
    trainable_csp   : whether CSP weights receive gradients (default True)
    lambda_grl      : initial GRL strength (updated externally; default 0.0)
    qual_hidden     : hidden dim in quality head (default 64)
    domain_hidden   : hidden dim in domain head; 0 keeps a single linear layer
    domain_dropout  : dropout probability inside domain head
    use_grl         : if False, bypass gradient reversal and train the domain
                      head as a standard auxiliary classifier
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_subjects: int,
        n_classes: int = 2,
        n_csp: int = 8,
        F1: int = 8,
        F2: int = 16,
        kernel_length: int = None,
        dropout: float = 0.25,
        trainable_csp: bool = True,
        lambda_grl: float = 0.0,
        qual_hidden: int = 64,
        domain_hidden: int = 0,
        domain_dropout: float = 0.0,
        use_grl: bool = True,
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

        # ── Shared encoder (identical blocks to CSPNet) ───────────────────────
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

        # ── Head 1: MI classifier ─────────────────────────────────────────────
        self.classifier = nn.Linear(n_flat, n_classes)

        # ── Head 2: Quality discriminator (clean / noisy) ────────────────────
        self.quality_head = nn.Sequential(
            nn.Linear(n_flat, qual_hidden),
            nn.ELU(),
            nn.Linear(qual_hidden, 1),
        )

        # ── Head 3: Domain (subject) discriminator via GRL ───────────────────
        self.use_grl = bool(use_grl)
        self.grl = GradientReversal(lambda_grl=lambda_grl)
        if domain_hidden and domain_hidden > 0:
            self.domain_head = nn.Sequential(
                nn.Linear(n_flat, domain_hidden),
                nn.ELU(),
                nn.Dropout(domain_dropout),
                nn.Linear(domain_hidden, n_subjects),
            )
        else:
            self.domain_head = nn.Linear(n_flat, n_subjects)

    # ── Shared feature extraction ─────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, C, T) → z : (B, n_flat)"""
        x = self.temporal_conv(x)
        x = self.csp_layer(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.sep_dw(x)
        x = self.sep_pw(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        return x.flatten(1)

    def classify_from_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def quality_from_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.quality_head(z).squeeze(-1)

    def domain_from_features(self, z: torch.Tensor) -> torch.Tensor:
        z_in = self.grl(z) if self.use_grl else z
        return self.domain_head(z_in)

    # ── Forward (training: all heads; inference: classifier only) ────────────

    def forward(self, x: torch.Tensor):
        """
        x : (B, C, T)

        Training  : returns (logits, quality_logit, domain_logits)
        Inference : returns logits only
        """
        z = self._encode(x.unsqueeze(1))
        logits = self.classify_from_features(z)

        if self.training:
            quality_logit = self.quality_from_features(z)       # (B,)
            domain_logits = self.domain_from_features(z)        # (B, n_subjects)
            return logits, quality_logit, domain_logits

        return logits

    def set_grl_lambda(self, lam: float) -> None:
        """Update GRL strength (called each epoch from training loop)."""
        self.grl.lambda_grl = float(lam)


# ─────────────────────────────────────────────────────────────────────────────
# Loss helper
# ─────────────────────────────────────────────────────────────────────────────

def dann_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    quality_logit: torch.Tensor,
    quality_label: torch.Tensor,
    domain_logits: torch.Tensor,
    subject_id: torch.Tensor,
    lambda_q: float = 1.0,
    lambda_d: float = 1.0,
    sample_weight=None,
):
    """
    Compute joint DANN loss.

    Parameters
    ----------
    logits         : (B, n_classes)
    y              : (B,) int — MI class labels
    quality_logit  : (B,) float — quality head raw output
    quality_label  : (B,) float ∈ [0,1] — pre-computed quality scores
    domain_logits  : (B, n_subjects)
    subject_id     : (B,) int — subject indices in training split
    lambda_q/d     : loss weights
    sample_weight  : optional (B,) per-trial weight tensor

    Returns
    -------
    total : scalar loss tensor
    parts : dict with 'cls', 'quality', 'domain' components
    """
    # Classification loss (weighted)
    cls_per_sample = F.cross_entropy(logits, y, reduction="none")
    if sample_weight is not None:
        w = sample_weight.to(logits.device).float()
        w = w / w.sum().clamp_min(1e-8) * len(w)  # normalise to mean=1
        cls_loss = (cls_per_sample * w).mean()
    else:
        cls_loss = cls_per_sample.mean()

    # Quality loss: BCE between sigmoid(quality_logit) and quality_label
    quality_loss = F.binary_cross_entropy_with_logits(
        quality_logit, quality_label.float()
    )

    # Domain adversarial loss (CE; GRL reverses gradient in backbone)
    domain_loss = F.cross_entropy(domain_logits, subject_id)

    total = cls_loss + lambda_q * quality_loss + lambda_d * domain_loss

    return total, {
        "cls": cls_loss.item(),
        "quality": quality_loss.item(),
        "domain": domain_loss.item(),
        "total": total.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSP filter initialization
# ─────────────────────────────────────────────────────────────────────────────

def fit_csp_layer_dann(
    model: CSPNetDANN,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """Initialize model.csp_layer.W from training data."""
    n_csp = model.csp_layer.n_csp
    W = compute_csp_filters(X_train, y_train, n_filters=n_csp)
    model.csp_layer.init_from_numpy(W)
    print(f"    [CSPNetDANN] CSP init "
          f"(n_csp={n_csp}, trainable={isinstance(model.csp_layer.W, nn.Parameter)})")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== CSPNetDANN sanity check ===\n")

    for ds, C, T, n_subj in [("Cho2017", 64, 257, 51), ("Lee2019", 62, 201, 53)]:
        model = CSPNetDANN(n_channels=C, n_times=T, n_subjects=n_subj)
        model.eval()
        x = torch.randn(8, C, T)
        with torch.no_grad():
            logits = model(x)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{ds}: logits={tuple(logits.shape)}  params={n_params:,}")

    print()
    print("=== Training forward + loss ===")
    C, T, n_subj, B = 64, 257, 51, 16
    model = CSPNetDANN(n_channels=C, n_times=T, n_subjects=n_subj, lambda_grl=0.5)
    model.train()
    model.set_grl_lambda(grl_lambda(0.5))

    x         = torch.randn(B, C, T)
    y         = torch.randint(0, 2, (B,))
    q_labels  = torch.rand(B)
    subj_ids  = torch.randint(0, n_subj, (B,))

    logits, q_logit, d_logits = model(x)
    print(f"  logits={tuple(logits.shape)}, "
          f"quality={tuple(q_logit.shape)}, "
          f"domain={tuple(d_logits.shape)}")

    total, parts = dann_loss(logits, y, q_logit, q_labels, d_logits, subj_ids,
                             lambda_q=0.5, lambda_d=1.0)
    total.backward()
    print(f"  loss={parts['total']:.4f}  "
          f"(cls={parts['cls']:.4f}, "
          f"quality={parts['quality']:.4f}, "
          f"domain={parts['domain']:.4f})")

    grad_csp = model.csp_layer.W.grad
    grad_dom = model.domain_head.weight.grad
    grad_cls = model.classifier.weight.grad
    print(f"  grad_csp={grad_csp.norm():.4f}  "
          f"grad_domain={grad_dom.norm():.4f}  "
          f"grad_cls={grad_cls.norm():.4f}")
    print("  Gradients OK" if all(
        g.norm() > 0 for g in [grad_csp, grad_dom, grad_cls]) else "  WARNING: zero grads")

    print()
    print("=== CSP init ===")
    model = CSPNetDANN(n_channels=64, n_times=257, n_subjects=51)
    X_fake = np.random.randn(80, 64, 257).astype(np.float32)
    y_fake = np.array([0]*40 + [1]*40)
    fit_csp_layer_dann(model, X_fake, y_fake)
    print("CSP init OK")
