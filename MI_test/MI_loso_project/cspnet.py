"""
CSP-Net: Differentiable Common Spatial Patterns network.

Based on: "CSP-Net: Common Spatial Pattern Empowered Neural Networks
for EEG-Based Motor Imagery Classification"
arXiv 2411.11879 / Knowledge-Based Systems (2024)
DOI: 10.1016/j.knosys.2024.112668

Architecture implemented: CSP-Net-2 (EEGNet backbone)
  - Temporal conv     : (B, 1, C, T) → (B, F1, C, T1)
  - CSP spatial layer : (B, F1, C, T1) → (B, F1*n_csp, 1, T1)
    [replaces depthwise spatial conv in standard EEGNet]
  - Separable conv    : → (B, F2, 1, T2)
  - Classifier        : flatten → Linear → n_classes

CSP layer initialization
  Generalized eigenvalue decomposition:  Σ_1 w = λ Σ_c w
  where Σ_1 = mean normalized covariance of class 0,
        Σ_c = Σ_1 + Σ_2 (composite).
  Eigenvectors with the n_csp/2 smallest and n_csp/2 largest
  eigenvalues are selected as spatial filters (most discriminative).

Usage in LOSO loop
  model = CSPNet(n_channels, n_times).to(device)
  fit_csp_layer(model, X_train_prenorm, y_train)   # before training
  # then normal Adam + epoch loop
"""

import numpy as np
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# CSP filter computation (NumPy, CPU, called once per LOSO fold)
# ──────────────────────────────────────────────────────────────────────────────

def _class_covariances(X: np.ndarray, y: np.ndarray):
    """
    Compute normalized mean covariance matrix per class.

    Parameters
    ----------
    X : (N, C, T) float32
    y : (N,) int with values in {0, 1}

    Returns
    -------
    Sigma_0, Sigma_1 : (C, C) float64 each
    """
    covs = []
    for cls in (0, 1):
        Xc = X[y == cls].astype(np.float64)    # (N_cls, C, T)
        Xc = Xc - Xc.mean(axis=2, keepdims=True)  # centre per trial
        C = Xc.shape[1]
        cov = np.zeros((C, C), dtype=np.float64)
        for trial in Xc:
            cc = trial @ trial.T                # (C, C)
            tr = np.trace(cc)
            if tr > 1e-12:
                cov += cc / tr
        cov /= max(len(Xc), 1)
        covs.append(cov)
    return covs[0], covs[1]


def compute_csp_filters(
    X: np.ndarray,
    y: np.ndarray,
    n_filters: int = 8,
) -> np.ndarray:
    """
    Compute CSP spatial filters via generalized eigenvalue decomposition.

    Solves  Σ_0 w = λ Σ_c w,  Σ_c = Σ_0 + Σ_1.
    Returns the n_filters/2 smallest-λ and n_filters/2 largest-λ
    eigenvectors, stacked into W of shape (n_filters, C).

    Falls back to random init (xavier) on numerical failure.
    """
    try:
        from scipy.linalg import eigh as sp_eigh
        Sigma_0, Sigma_1 = _class_covariances(X, y)
        Sigma_c = Sigma_0 + Sigma_1

        # Regularise to avoid singular composite covariance
        eps = 1e-6 * np.trace(Sigma_c) / Sigma_c.shape[0]
        Sigma_c += eps * np.eye(Sigma_c.shape[0])

        # eigh(a, b) solves: a @ v = λ * b @ v, eigenvalues ascending
        eigvals, eigvecs = sp_eigh(Sigma_0, Sigma_c)   # (C,), (C, C)
        C = eigvecs.shape[0]
        n_half = n_filters // 2
        n_rest = n_filters - n_half

        # Clamp indices to available channels
        lo_idx = np.arange(min(n_half, C))
        hi_idx = np.arange(max(0, C - n_rest), C)
        idx    = np.unique(np.concatenate([lo_idx, hi_idx]))[:n_filters]

        W = eigvecs[:, idx].T.astype(np.float32)  # (n_filters, C)
        # Pad with zeros if we got fewer than n_filters (C < n_filters)
        if W.shape[0] < n_filters:
            pad = np.zeros((n_filters - W.shape[0], C), dtype=np.float32)
            W = np.vstack([W, pad])
        return W

    except Exception as exc:
        print(f"    [CSP] filter init failed ({exc}), using random init")
        C = X.shape[1]
        rng = np.random.default_rng(42)
        W = rng.standard_normal((n_filters, C)).astype(np.float32)
        # Orthogonalise via SVD for a better starting point
        U, _, Vt = np.linalg.svd(W, full_matrices=False)
        return U if U.shape == (n_filters, C) else Vt


# ──────────────────────────────────────────────────────────────────────────────
# Learnable CSP spatial layer
# ──────────────────────────────────────────────────────────────────────────────

class CSPLayer(nn.Module):
    """
    Differentiable spatial filter layer based on CSP.

    Applies n_csp spatial filters to the C-channel axis.

    Input  : (B, in_filters, C, T)
    Output : (B, in_filters * n_csp, 1, T)

    The same spatial projection W ∈ R^(n_csp × C) is shared across
    all in_filters temporal feature maps (analogous to group-wise
    spatial projection, but tied across groups for parameter efficiency).
    """

    def __init__(
        self,
        n_channels: int,
        n_csp: int = 8,
        trainable: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_csp      = n_csp

        W_init = torch.empty(n_csp, n_channels)
        nn.init.xavier_uniform_(W_init)

        if trainable:
            self.W = nn.Parameter(W_init)
        else:
            self.register_buffer("W", W_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, F1, C, T)
        B, F1, C, T = x.shape
        # Reshape → (B*F1, C, T)
        x = x.reshape(B * F1, C, T)
        # Spatial projection: W (n_csp, C) × x (C, T) → (n_csp, T) per sample
        # einsum: 'nc, bct -> bnt'  (n=n_csp, c=C, b=B*F1, t=T)
        x = torch.einsum("nc,bct->bnt", self.W, x)  # (B*F1, n_csp, T)
        # Reshape → (B, F1*n_csp, 1, T)
        x = x.reshape(B, F1 * self.n_csp, 1, T)
        return x

    def init_from_numpy(self, W: np.ndarray):
        """Set weights from a precomputed (n_csp, C) NumPy array."""
        W_t = torch.from_numpy(W[: self.n_csp].astype(np.float32))
        with torch.no_grad():
            if isinstance(self.W, nn.Parameter):
                self.W.data.copy_(W_t)
            else:
                self.W.copy_(W_t)


# ──────────────────────────────────────────────────────────────────────────────
# Full CSP-Net-2 model
# ──────────────────────────────────────────────────────────────────────────────

class CSPNet(nn.Module):
    """
    CSP-Net-2 (EEGNet backbone with CSP spatial layer).

    Parameters
    ----------
    n_channels     : number of EEG channels C
    n_times        : number of time samples T
    n_classes      : number of output classes (default 2)
    n_csp          : number of CSP spatial filters (default 8)
    F1             : number of temporal filters (default 8)
    F2             : number of point-wise filters in separable conv (default 16)
    kernel_length  : temporal conv kernel length; None → n_times // 4
    dropout        : dropout probability (default 0.25, paper value)
    trainable_csp  : whether CSP weights are updated during backprop (default True)

    Input  : (B, C, T)  — same convention as SPDNet / MIN2Net in train_loso.py
    Output : (B, n_classes) logits
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
    ):
        super().__init__()

        # Clamp n_csp to a sensible maximum
        n_csp = min(n_csp, n_channels)

        if kernel_length is None:
            kernel_length = max(16, n_times // 4)
        # Ensure kernel_length is odd for symmetric 'same' padding
        if kernel_length % 2 == 0:
            kernel_length += 1
        pad_t = kernel_length // 2

        self.n_csp = n_csp
        self.F1    = F1

        # ── Block 1 : Temporal convolution ──────────────────────────────────
        # (B, 1, C, T) → (B, F1, C, T)  [same-length via padding]
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, pad_t), bias=False),
            nn.BatchNorm2d(F1),
        )

        # ── Block 2 : CSP spatial filter (replaces depthwise conv) ──────────
        # (B, F1, C, T) → (B, F1*n_csp, 1, T)
        self.csp_layer = CSPLayer(n_channels, n_csp=n_csp, trainable=trainable_csp)
        self.bn2       = nn.BatchNorm2d(F1 * n_csp)
        self.act2      = nn.ELU()
        self.pool2     = nn.AvgPool2d((1, 4))
        self.drop2     = nn.Dropout(dropout)

        # ── Block 3 : Separable conv (depthwise + pointwise) ─────────────────
        mid = F1 * n_csp
        sep_kern = max(8, n_times // 16)
        if sep_kern % 2 == 0:
            sep_kern += 1
        sep_pad = sep_kern // 2

        self.sep_dw = nn.Conv2d(mid, mid, (1, sep_kern),
                                padding=(0, sep_pad), groups=mid, bias=False)
        self.sep_pw = nn.Conv2d(mid, F2, (1, 1), bias=False)
        self.bn3    = nn.BatchNorm2d(F2)
        self.act3   = nn.ELU()
        self.pool3  = nn.AvgPool2d((1, 8))
        self.drop3  = nn.Dropout(dropout)

        # ── Classifier ───────────────────────────────────────────────────────
        with torch.no_grad():
            n_flat = self._forward_features(
                torch.zeros(1, 1, n_channels, n_times)
            ).shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    # ── Feature extraction (no classification head) ─────────────────────────

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, C, T)"""
        x = self.temporal_conv(x)   # (B, F1, C, T)
        x = self.csp_layer(x)       # (B, F1*n_csp, 1, T)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)           # (B, F1*n_csp, 1, T//4)
        x = self.drop2(x)
        x = self.sep_dw(x)
        x = self.sep_pw(x)          # (B, F2, 1, ...)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)           # (B, F2, 1, T//32)
        x = self.drop3(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T) — same input convention as SPDNet / MIN2Net.
        Returns logits (B, n_classes).
        """
        x = x.unsqueeze(1)                      # (B, 1, C, T)
        return self.classifier(self._forward_features(x))


# ──────────────────────────────────────────────────────────────────────────────
# Fit helper (call once per LOSO fold, before the epoch loop)
# ──────────────────────────────────────────────────────────────────────────────

def fit_csp_layer(
    model: CSPNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """
    Initialize model.csp_layer.W from training data using generalized
    eigenvalue decomposition (CSP).

    Parameters
    ----------
    model    : CSPNet instance (already on target device)
    X_train  : (N, C, T) float32, UN-normalized training epochs
    y_train  : (N,) int, binary labels {0, 1}

    Notes
    -----
    - Must be called BEFORE the training loop.
    - Uses pre-normalization data so class covariance structure is intact.
    - The fitted weights are immediately differentiable (gradient flows through
      them if trainable_csp=True).
    """
    n_csp = model.csp_layer.n_csp
    W = compute_csp_filters(X_train, y_train, n_filters=n_csp)
    model.csp_layer.init_from_numpy(W)
    print(f"    [CSPNet] CSP layer initialized "
          f"(n_csp={n_csp}, trainable={isinstance(model.csp_layer.W, nn.Parameter)})")
