"""
CSP-Net-2 RSNN: Recurrent Spiking Neural Network version of CSP-Net-2.

Architecture difference from CSPNetSNN
---------------------------------------
  CSPNetSNN  : feedforward LIF — each timestep is independent (no lateral feedback)
  CSPNetRSNN : Recurrent LIF  — spike output at t feeds back as input at t+1

Recurrent connection
--------------------
For LIF layer operating on (B, C, 1, T_eeg), recurrent connections are
channel-wise linear projections applied identically at every EEG time position:

  v[t+1] = v[t] / tau + x[t] + W_rec @ s[t]
  s[t+1] = H(v[t+1] - v_th)
  v[t+1] = v[t+1] - v_th * s[t+1]          (soft reset)

W_rec is initialised orthogonally with a small gain (0.1) to avoid instability
at the start of training.

Two recurrent layers are added:
  rlif2: after CSP spatial block   — lateral recurrence across F1*n_csp channels
  rlif3: after separable conv block — lateral recurrence across F2 channels
"""

import math
import numpy as np
import torch
import torch.nn as nn

from cspnet import CSPLayer, compute_csp_filters
from cspnet_snn import _ATanSurrogate, atan_spike


# ─────────────────────────────────────────────────────────────────────────────
# Recurrent LIF cell
# ─────────────────────────────────────────────────────────────────────────────

class RLIFCell(nn.Module):
    """
    Recurrent LIF neuron cell.

    State per call: (v, s_prev) — membrane potential and last spike tensor.

    Equations (soft reset):
        v[t+1] = v[t] / tau + x[t] + W_rec @ s[t]   (integrate + recurrent)
        s[t+1] = H(v[t+1] - v_th)                    (fire)
        v[t+1] = v[t+1] - v_th * s[t+1]              (soft reset)

    The recurrent projection W_rec maps (B, C, 1, T) → (B, C, 1, T) by applying
    an (C × C) linear map independently at every EEG time position t.
    Implemented as a bias-free 1×1 Conv2d for efficiency.

    Parameters
    ----------
    n_neurons       : number of channels C in the feature map
    tau             : membrane time constant (default 2.0)
    v_threshold     : firing threshold (default 1.0)
    surrogate_alpha : sharpness of ATan surrogate gradient (default 2.0)
    rec_gain        : orthogonal init gain for W_rec (default 0.1)
    """

    def __init__(
        self,
        n_neurons: int,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
        rec_gain: float = 0.1,
    ):
        super().__init__()
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.surrogate_alpha = float(surrogate_alpha)
        self.n_neurons = n_neurons

        # 1×1 conv implements channel-wise linear W_rec at each spatial/time position
        self.rec = nn.Conv2d(n_neurons, n_neurons, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.rec.weight.view(n_neurons, n_neurons),
                            gain=rec_gain)

    def forward(
        self,
        x: torch.Tensor,        # (B, C, 1, T) — input current
        v: torch.Tensor,        # (B, C, 1, T) — membrane potential
        s_prev: torch.Tensor,   # (B, C, 1, T) — previous spike
    ):
        """
        Returns
        -------
        spike : (B, C, 1, T) — binary spike tensor (float)
        v_new : (B, C, 1, T) — updated membrane potential
        """
        rec_input = self.rec(s_prev)                          # (B, C, 1, T)
        v_new = v / self.tau + x + rec_input
        spike = atan_spike(v_new - self.v_threshold, self.surrogate_alpha)
        v_new = v_new - self.v_threshold * spike              # soft reset
        return spike, v_new

    def extra_repr(self) -> str:
        return (f"n_neurons={self.n_neurons}, tau={self.tau}, "
                f"v_th={self.v_threshold}, surrogate_alpha={self.surrogate_alpha}")


# ─────────────────────────────────────────────────────────────────────────────
# CSP-Net-2 RSNN
# ─────────────────────────────────────────────────────────────────────────────

class CSPNetRSNN(nn.Module):
    """
    Recurrent Spiking CSP-Net-2.

    Identical to CSPNetSNN except LIFCell → RLIFCell at blocks 2 & 3.
    Each RLIFCell carries a learnable recurrent weight matrix W_rec that feeds
    the spike output from SNN timestep t back into the membrane update at t+1.

    Parameters
    ----------
    n_channels      : EEG channels C
    n_times         : time samples T
    n_classes       : output classes (default 2)
    n_csp           : CSP spatial filters (default 8)
    F1              : temporal filters (default 8)
    F2              : separable conv output channels (default 16)
    kernel_length   : temporal conv kernel; None → max(16, n_times // 4)
    dropout         : dropout probability (default 0.25)
    trainable_csp   : whether CSP weights are updated during training
    T_sim           : SNN simulation timesteps (default 8)
    tau             : LIF membrane time constant (default 2.0)
    v_threshold     : LIF firing threshold (default 1.0)
    encoding        : 'direct' | 'rate'
    surrogate_alpha : ATan surrogate sharpness (default 2.0)
    rec_gain        : orthogonal init gain for recurrent weights (default 0.1)
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
        T_sim: int = 8,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        encoding: str = "direct",
        surrogate_alpha: float = 2.0,
        rec_gain: float = 0.1,
    ):
        super().__init__()

        n_csp = min(n_csp, n_channels)
        if kernel_length is None:
            kernel_length = max(16, n_times // 4)
        if kernel_length % 2 == 0:
            kernel_length += 1
        pad_t = kernel_length // 2

        sep_kern = max(8, n_times // 16)
        if sep_kern % 2 == 0:
            sep_kern += 1
        sep_pad = sep_kern // 2

        mid = F1 * n_csp
        self.n_csp    = n_csp
        self.F1       = F1
        self.T_sim    = T_sim
        self.encoding = encoding

        # ── Block 1: Temporal conv (no activation) ───────────────────────────
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, pad_t), bias=False),
            nn.BatchNorm2d(F1),
        )

        # ── Block 2: CSP spatial + BN + RLIF + Pool + Dropout ────────────────
        self.csp_layer = CSPLayer(n_channels, n_csp=n_csp, trainable=trainable_csp)
        self.bn2   = nn.BatchNorm2d(mid)
        self.rlif2 = RLIFCell(mid, tau, v_threshold, surrogate_alpha, rec_gain)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(dropout)

        # ── Block 3: Separable conv + BN + RLIF + Pool + Dropout ─────────────
        self.sep_dw = nn.Conv2d(mid, mid, (1, sep_kern),
                                padding=(0, sep_pad), groups=mid, bias=False)
        self.sep_pw = nn.Conv2d(mid, F2, (1, 1), bias=False)
        self.bn3   = nn.BatchNorm2d(F2)
        self.rlif3 = RLIFCell(F2, tau, v_threshold, surrogate_alpha, rec_gain)
        self.pool3 = nn.AvgPool2d((1, 8))
        self.drop3 = nn.Dropout(dropout)

        # ── Classifier + precompute buffer shapes ────────────────────────────
        with torch.no_grad():
            probe = torch.zeros(1, 1, n_channels, n_times)
            _h1  = self.temporal_conv(probe)
            _h2  = self.csp_layer(_h1)
            _h2p = self.pool2(_h2)
            _h3  = self.sep_pw(self.sep_dw(_h2p))
            _h3p = self.pool3(_h3)
            n_flat = _h3p.flatten(1).shape[1]
            self._v2_shape = tuple(_h2.shape[1:])   # (mid, 1, T_eeg)
            self._v3_shape = tuple(_h3.shape[1:])   # (F2,  1, T_eeg//4)
        self.classifier = nn.Linear(n_flat, n_classes)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)
        Returns logits (B, n_classes).

        Simulation loop:
            For each SNN timestep t = 0 … T_sim-1:
              1. Encode input → rates (same for 'direct'; resampled for 'rate')
              2. Block1: temporal conv (shared across steps for 'direct')
              3. Block2: CSP → BN → RLIF(v2, s2) → Pool → Drop
              4. Block3: sep conv → BN → RLIF(v3, s3) → Pool → Drop
              5. Accumulate spikes
            Output = mean spike rate → classifier
        """
        B = x.shape[0]
        x = x.unsqueeze(1)                              # (B, 1, C, T)
        base_rates = torch.sigmoid(x)

        # Initialise membrane potentials and previous spikes (all zeros)
        v2 = torch.zeros(B, *self._v2_shape, device=x.device, dtype=x.dtype)
        v3 = torch.zeros(B, *self._v3_shape, device=x.device, dtype=x.dtype)
        s2 = torch.zeros_like(v2)
        s3 = torch.zeros_like(v3)

        spike_sum = torch.zeros(B, self.classifier.in_features,
                                device=x.device, dtype=x.dtype)

        # Precompute Block 1 for 'direct' mode (same every SNN step)
        if self.encoding != "rate" or not self.training:
            h1 = self.temporal_conv(base_rates)         # (B, F1, C, T)
        else:
            h1 = None

        for _ in range(self.T_sim):
            # Block 1 — temporal conv
            if h1 is None:
                s_in = torch.bernoulli(base_rates)
                h = self.temporal_conv(s_in)
            else:
                h = h1

            # Block 2 — CSP + BN → RLIF(v2, s2) → Pool → Dropout
            h = self.csp_layer(h)                       # (B, mid, 1, T_eeg)
            h = self.bn2(h)
            s2, v2 = self.rlif2(h, v2, s2)             # recurrent spike + update
            h = self.pool2(s2)                          # pool the spikes
            h = self.drop2(h)

            # Block 3 — sep conv + BN → RLIF(v3, s3) → Pool → Dropout
            h = self.sep_dw(h)
            h = self.sep_pw(h)                          # (B, F2, 1, T//4)
            h = self.bn3(h)
            s3, v3 = self.rlif3(h, v3, s3)
            h = self.pool3(s3)
            h = self.drop3(h)

            spike_sum += h.flatten(1)

        return self.classifier(spike_sum / self.T_sim)


# ─────────────────────────────────────────────────────────────────────────────
# CSP filter initialization
# ─────────────────────────────────────────────────────────────────────────────

def fit_csp_layer_rsnn(
    model: CSPNetRSNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """Initialize model.csp_layer.W from training data."""
    n_csp = model.csp_layer.n_csp
    W = compute_csp_filters(X_train, y_train, n_filters=n_csp)
    model.csp_layer.init_from_numpy(W)
    print(f"    [CSPNetRSNN] CSP layer initialized "
          f"(n_csp={n_csp}, "
          f"trainable={isinstance(model.csp_layer.W, nn.Parameter)}, "
          f"T_sim={model.T_sim})")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=== CSPNetRSNN sanity check ===\n")

    for ds, C, T in [("Cho2017", 64, 257), ("Lee2019", 62, 201)]:
        model = CSPNetRSNN(n_channels=C, n_times=T, T_sim=8, encoding="direct")
        model.eval()
        x = torch.randn(8, C, T)
        with torch.no_grad():
            logits = model(x)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rec_params = sum(p.numel() for name, p in model.named_parameters()
                         if "rec" in name)
        x1 = torch.randn(1, C, T)
        for _ in range(10): model(x1)
        t0 = time.perf_counter()
        for _ in range(100):
            with torch.no_grad(): model(x1)
        t_ms = (time.perf_counter() - t0) / 100 * 1000
        print(f"{ds:10s}  logits={tuple(logits.shape)}  "
              f"params={n_params:,} (rec={rec_params:,})  "
              f"infer={t_ms:.2f}ms/trial (CPU)")

    print()
    print("=== Gradient flow check ===")
    model = CSPNetRSNN(n_channels=64, n_times=257)
    model.train()
    x = torch.randn(4, 64, 257)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    grad_csp = model.csp_layer.W.grad
    grad_rec2 = model.rlif2.rec.weight.grad
    grad_rec3 = model.rlif3.rec.weight.grad
    grad_cls  = model.classifier.weight.grad
    print(f"  csp_layer.W grad norm    : {grad_csp.norm().item():.4f}")
    print(f"  rlif2.rec.weight grad    : {grad_rec2.norm().item():.4f}")
    print(f"  rlif3.rec.weight grad    : {grad_rec3.norm().item():.4f}")
    print(f"  classifier.weight grad   : {grad_cls.norm().item():.4f}")
    ok = all(g.norm() > 0 for g in [grad_csp, grad_rec2, grad_rec3, grad_cls])
    print("  Gradients flow through recurrent RLIF: OK" if ok else "  WARNING: zero gradients")

    print()
    print("=== CSP init ===")
    model = CSPNetRSNN(n_channels=64, n_times=257)
    X_fake = np.random.randn(80, 64, 257).astype(np.float32)
    y_fake = np.array([0]*40 + [1]*40)
    fit_csp_layer_rsnn(model, X_fake, y_fake)
    print("CSP init OK")
