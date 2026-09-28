"""
CSP-Net-2 SNN: Spiking Neural Network version of CSP-Net-2.

Conversion strategy
-------------------
  ANN (CSPNet)                SNN (CSPNetSNN)
  ─────────────────────────────────────────────
  ELU (act2, act3)     →     LIF neuron (ATan surrogate gradient)
  Conv / BN / Pool     →     unchanged  (linear ops work on spike tensors)
  CSP spatial layer    →     unchanged  (same linear projection)
  Dropout              →     unchanged
  Linear classifier    →     unchanged  (reads mean firing rate)

Input encoding
--------------
  'rate'   : S[t] ~ Bernoulli(sigmoid(x))  — stochastic Poisson spike train
  'direct' : S[t] = sigmoid(x)             — soft, deterministic (default)

Both options normalise the continuous EEG input into the [0, 1] range
required for spike-rate representation.

Simulation
----------
  The network is unrolled for T_sim timesteps (default 8).
  Each LIF layer maintains a membrane potential v across timesteps.
  The output is the accumulated spike count divided by T_sim, fed into
  the linear classifier — equivalent to reading the mean firing rate.

CSP initialization
------------------
  fit_csp_layer_snn(model, X_train_prenorm, y_train)
  Delegates to cspnet.fit_csp_layer — works identically because
  model.csp_layer is the same CSPLayer class.

Reference (SNN training)
------------------------
  Wu et al. "Spatio-temporal backpropagation for training high-performance
  spiking neural networks." Frontiers in Neuroscience, 2018.
  (ATan surrogate gradient for LIF neurons)
"""

import math
import numpy as np
import torch
import torch.nn as nn

from cspnet import CSPLayer, compute_csp_filters


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate gradient
# ─────────────────────────────────────────────────────────────────────────────

class _ATanSurrogate(torch.autograd.Function):
    """
    Heaviside in the forward pass; ATan surrogate in the backward pass.

    Surrogate derivative  (Fang et al., 2021):
        dH/dx  ≈  alpha / (2 * (1 + (pi * alpha / 2 * x)^2))
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).float()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        sg = ctx.alpha / 2.0 / (1.0 + (math.pi * ctx.alpha / 2.0 * x).pow(2))
        return grad * sg, None


def atan_spike(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Convenience wrapper for _ATanSurrogate."""
    return _ATanSurrogate.apply(x, alpha)


# ─────────────────────────────────────────────────────────────────────────────
# LIF neuron cell
# ─────────────────────────────────────────────────────────────────────────────

class LIFCell(nn.Module):
    """
    Stateless LIF neuron (state passed explicitly).

    Equations (soft reset):
        v[t+1] = v[t] / tau + x[t]               (integrate)
        s[t]   = H(v[t+1] - v_th)                 (fire)
        v[t+1] = v[t+1] - v_th * s[t]             (soft reset)

    Soft reset keeps residual potential above threshold, which improves
    gradient flow compared to hard reset (v = 0 after spike).

    Parameters
    ----------
    tau           : membrane time constant (default 2.0 → leak of 0.5 per step)
    v_threshold   : firing threshold (default 1.0)
    surrogate_alpha : sharpness of ATan surrogate gradient (default 2.0)
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.surrogate_alpha = float(surrogate_alpha)

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
    ):
        """
        Parameters
        ----------
        x : input current tensor, any shape
        v : membrane potential, same shape as x

        Returns
        -------
        spike : binary spike tensor (float), same shape as x
        v_new : updated membrane potential
        """
        v_new = v / self.tau + x
        spike = atan_spike(v_new - self.v_threshold, self.surrogate_alpha)
        v_new = v_new - self.v_threshold * spike      # soft reset
        return spike, v_new

    def extra_repr(self) -> str:
        return (f"tau={self.tau}, v_th={self.v_threshold}, "
                f"surrogate_alpha={self.surrogate_alpha}")


# ─────────────────────────────────────────────────────────────────────────────
# Input encoder
# ─────────────────────────────────────────────────────────────────────────────

class RateEncoder(nn.Module):
    """
    Encode a continuous EEG tensor into per-step spike tensors.

    Modes
    -----
    'direct' : s[t] = sigmoid(x)          — deterministic, same every step
    'rate'   : s[t] ~ Bernoulli(sigmoid(x)) — Poisson spike sampling
    'none'   : s[t] = x                   — raw values (for ablation)

    sigmoid maps EEG amplitudes (arbitrary range) to [0, 1] spike rates.
    """

    def __init__(self, mode: str = "direct"):
        super().__init__()
        if mode not in ("direct", "rate", "none"):
            raise ValueError(f"Unknown encoding mode '{mode}'. "
                             f"Choose 'direct', 'rate', or 'none'.")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : any shape (typically (B, 1, C, T))
        Returns encoded tensor of the same shape.
        For 'rate' mode, sampling is done per call (one timestep at a time).
        """
        if self.mode == "none":
            return x
        rates = torch.sigmoid(x)
        if self.mode == "rate" and self.training:
            return torch.bernoulli(rates)
        return rates          # 'direct' or eval-time rate encoding


# ─────────────────────────────────────────────────────────────────────────────
# CSP-Net-2 SNN
# ─────────────────────────────────────────────────────────────────────────────

class CSPNetSNN(nn.Module):
    """
    Spiking CSP-Net-2.

    The architecture mirrors CSPNet exactly except that the two ELU
    activations (after Block 2 and Block 3) are replaced by LIF neurons.

    Parameters
    ----------
    n_channels      : EEG channels C
    n_times         : time samples T
    n_classes       : output classes (default 2)
    n_csp           : CSP spatial filters (default 8)
    F1              : temporal filters (default 8)
    F2              : separable conv output channels (default 16)
    kernel_length   : temporal conv kernel; None → n_times // 4
    dropout         : dropout probability (default 0.25)
    trainable_csp   : whether CSP weights are updated during training
    T_sim           : SNN simulation timesteps (default 8)
    tau             : LIF membrane time constant (default 2.0)
    v_threshold     : LIF firing threshold (default 1.0)
    encoding        : input encoding mode — 'direct' | 'rate' | 'none'
    surrogate_alpha : ATan surrogate sharpness (default 2.0)

    Input  : (B, C, T)
    Output : (B, n_classes) — logits from mean firing rate over T_sim steps
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

        self.n_csp  = n_csp
        self.F1     = F1
        self.T_sim  = T_sim

        # ── Input encoder ────────────────────────────────────────────────────
        self.encoder = RateEncoder(mode=encoding)

        # ── Block 1: Temporal convolution ────────────────────────────────────
        # Same as ANN — no activation here; BN output feeds LIF2
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, pad_t), bias=False),
            nn.BatchNorm2d(F1),
        )

        # ── Block 2: CSP spatial + LIF ───────────────────────────────────────
        self.csp_layer = CSPLayer(n_channels, n_csp=n_csp, trainable=trainable_csp)
        self.bn2       = nn.BatchNorm2d(mid)
        self.lif2      = LIFCell(tau=tau, v_threshold=v_threshold,
                                 surrogate_alpha=surrogate_alpha)
        self.pool2     = nn.AvgPool2d((1, 4))
        self.drop2     = nn.Dropout(dropout)

        # ── Block 3: Separable conv + LIF ────────────────────────────────────
        self.sep_dw = nn.Conv2d(mid, mid, (1, sep_kern),
                                padding=(0, sep_pad), groups=mid, bias=False)
        self.sep_pw = nn.Conv2d(mid, F2, (1, 1), bias=False)
        self.bn3    = nn.BatchNorm2d(F2)
        self.lif3   = LIFCell(tau=tau, v_threshold=v_threshold,
                               surrogate_alpha=surrogate_alpha)
        self.pool3  = nn.AvgPool2d((1, 8))
        self.drop3  = nn.Dropout(dropout)

        # ── Classifier ───────────────────────────────────────────────────────
        with torch.no_grad():
            n_flat = self._probe_flat(n_channels, n_times)
        self.classifier = nn.Linear(n_flat, n_classes)

    # ── Shape probe (no LIF state needed) ────────────────────────────────────

    def _probe_flat(self, C: int, T: int) -> int:
        """Compute flatten size using a zero-pass through conv/pool layers."""
        x = torch.zeros(1, 1, C, T)
        x = self.temporal_conv(x)
        x = self.csp_layer(x)
        x = self.pool2(x)
        x = self.sep_dw(x)
        x = self.sep_pw(x)
        x = self.pool3(x)
        return x.flatten(1).shape[1]

    # ── Single-step feature forward ──────────────────────────────────────────

    def _step(
        self,
        s_in: torch.Tensor,
        v2: torch.Tensor,
        v3: torch.Tensor,
    ):
        """
        One SNN simulation step.

        Parameters
        ----------
        s_in : encoded input spike/rate tensor  (B, 1, C, T)
        v2   : membrane potential for LIF2
        v3   : membrane potential for LIF3

        Returns
        -------
        feat : flattened feature spike tensor  (B, n_flat)
        v2   : updated membrane potential for LIF2
        v3   : updated membrane potential for LIF3
        """
        # Block 1: temporal conv (no LIF — output used as input current to LIF2)
        h = self.temporal_conv(s_in)        # (B, F1, C, T)

        # Block 2: CSP spatial → BN → LIF → Pool → Dropout
        h = self.csp_layer(h)               # (B, F1*n_csp, 1, T)
        h = self.bn2(h)
        h, v2 = self.lif2(h, v2)            # spike + update membrane
        h = self.pool2(h)                   # (B, F1*n_csp, 1, T//4)
        h = self.drop2(h)

        # Block 3: separable conv → BN → LIF → Pool → Dropout
        h = self.sep_dw(h)
        h = self.sep_pw(h)                  # (B, F2, 1, ...)
        h = self.bn3(h)
        h, v3 = self.lif3(h, v3)
        h = self.pool3(h)                   # (B, F2, 1, T//32)
        h = self.drop3(h)

        return h.flatten(1), v2, v3

    # ── Full forward pass ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)
        Returns logits (B, n_classes).

        Simulation loop:
            1. Encode input → firing rates / spikes
            2. Run T_sim steps, accumulating output spikes
            3. Mean firing rate → linear classifier → logits
        """
        x = x.unsqueeze(1)                          # (B, 1, C, T)
        rates = self.encoder(x)                     # (B, 1, C, T) in [0,1]

        # Initialise membrane potentials to zero
        v2 = torch.zeros_like(
            self.temporal_conv(rates).view(
                rates.shape[0], self.F1 * self.n_csp,
                1, rates.shape[-1]
            )
        )
        # Correct v2 shape from CSP output
        with torch.no_grad():
            _h = self.csp_layer(self.temporal_conv(rates))
        v2 = torch.zeros_like(_h)
        v3 = None                                   # initialised lazily below

        spike_sum = None
        for _ in range(self.T_sim):
            # For 'rate' mode, each step samples independent spikes
            s_in = self.encoder(x) if self.encoder.mode == "rate" else rates

            feat, v2, _v3_new = self._step(s_in, v2, v3 if v3 is not None
                                           else torch.zeros(
                                               rates.shape[0], *feat_shape
                                           ) if False else torch.zeros(1))

            # Lazy-init v3 on first step
            if v3 is None:
                # Determine v3 shape from feat by probing block3
                with torch.no_grad():
                    _h2 = self.pool2(
                        torch.zeros_like(
                            self.csp_layer(self.temporal_conv(rates))
                        )
                    )
                    _h3 = self.sep_pw(self.sep_dw(_h2))
                v3 = torch.zeros_like(_h3)
                # Redo step with proper v3
                feat, v2, v3 = self._step(s_in, v2, v3)

            else:
                v3 = _v3_new

            spike_sum = feat if spike_sum is None else spike_sum + feat

        return self.classifier(spike_sum / self.T_sim)


# ─────────────────────────────────────────────────────────────────────────────
# Cleaner forward using explicit membrane potential management
# ─────────────────────────────────────────────────────────────────────────────

class CSPNetSNN(nn.Module):  # noqa: F811 — redefine with cleaner implementation
    """
    Spiking CSP-Net-2 (clean implementation).

    ELU activations in CSPNet replaced by LIF neurons.
    All other layers (Conv, BN, CSP, Pool, Dropout, Linear) are unchanged.

    SNN simulation
    --------------
    T_sim steps are run per forward call.
    Membrane potentials (v2, v3) are reset to zero at each call.
    Output = mean spike rate over T_sim steps → linear classifier.

    Input encoding
    --------------
    'direct' : rates = sigmoid(x);  same rates used every step
    'rate'   : rates = sigmoid(x);  s[t] ~ Bernoulli(rates) per step
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
        self.n_csp   = n_csp
        self.F1      = F1
        self.T_sim   = T_sim
        self.encoding = encoding

        # ── Block 1: Temporal conv (no activation) ───────────────────────────
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, pad_t), bias=False),
            nn.BatchNorm2d(F1),
        )

        # ── Block 2: CSP spatial + BN + LIF + Pool + Dropout ─────────────────
        self.csp_layer = CSPLayer(n_channels, n_csp=n_csp, trainable=trainable_csp)
        self.bn2  = nn.BatchNorm2d(mid)
        self.lif2 = LIFCell(tau, v_threshold, surrogate_alpha)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(dropout)

        # ── Block 3: Separable conv + BN + LIF + Pool + Dropout ──────────────
        self.sep_dw = nn.Conv2d(mid, mid, (1, sep_kern),
                                padding=(0, sep_pad), groups=mid, bias=False)
        self.sep_pw = nn.Conv2d(mid, F2, (1, 1), bias=False)
        self.bn3  = nn.BatchNorm2d(F2)
        self.lif3 = LIFCell(tau, v_threshold, surrogate_alpha)
        self.pool3 = nn.AvgPool2d((1, 8))
        self.drop3 = nn.Dropout(dropout)

        # ── Classifier + precompute buffer shapes ────────────────────────────
        with torch.no_grad():
            probe = torch.zeros(1, 1, n_channels, n_times)
            _h1 = self.temporal_conv(probe)
            _h2 = self.csp_layer(_h1)
            _h2p = self.pool2(_h2)
            _h3 = self.sep_pw(self.sep_dw(_h2p))
            _h3p = self.pool3(_h3)
            n_flat = _h3p.flatten(1).shape[1]
            # Store shapes for membrane potential allocation in forward()
            self._v2_shape = tuple(_h2.shape[1:])   # (mid, 1, T)
            self._v3_shape = tuple(_h3.shape[1:])   # (F2,  1, T//4)
        self.classifier = nn.Linear(n_flat, n_classes)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _encode(x: torch.Tensor, mode: str, training: bool) -> torch.Tensor:
        """Map EEG values to [0,1] firing rates, optionally sample spikes."""
        rates = torch.sigmoid(x)
        if mode == "rate" and training:
            return torch.bernoulli(rates)
        return rates

    def _zero_v(self, ref: torch.Tensor, *shape) -> torch.Tensor:
        return torch.zeros(*shape, device=ref.device, dtype=ref.dtype)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)
        Returns logits (B, n_classes).
        """
        B = x.shape[0]
        x = x.unsqueeze(1)                              # (B, 1, C, T)

        # Compute rates once (reused every step in 'direct' mode)
        base_rates = torch.sigmoid(x)

        # Initialise membrane potentials using precomputed shapes
        v2 = torch.zeros(B, *self._v2_shape, device=x.device, dtype=x.dtype)
        v3 = torch.zeros(B, *self._v3_shape, device=x.device, dtype=x.dtype)

        spike_sum = torch.zeros(B, self.classifier.in_features,
                                device=x.device, dtype=x.dtype)

        # In 'direct' mode the temporal conv output is identical every step
        # → compute once and share across timesteps (Block 1 reuse).
        if self.encoding != "rate" or not self.training:
            h1 = self.temporal_conv(base_rates)         # (B, F1, C, T)
        else:
            h1 = None                                   # computed per-step for 'rate'

        for _ in range(self.T_sim):
            # Block 1 — temporal conv (reuse h1 for 'direct'; recompute for 'rate')
            if h1 is None:
                s = torch.bernoulli(base_rates)         # independent spike sample
                h = self.temporal_conv(s)
            else:
                h = h1                                  # same current every step

            # Block 2 — CSP + BN → LIF → Pool → Dropout
            h = self.csp_layer(h)                       # (B, mid, 1, T)
            h = self.bn2(h)
            h, v2 = self.lif2(h, v2)                    # spike + update v2
            h = self.pool2(h)                           # (B, mid, 1, T//4)
            h = self.drop2(h)

            # Block 3 — sep conv + BN → LIF → Pool → Dropout
            h = self.sep_dw(h)
            h = self.sep_pw(h)                          # (B, F2, 1, ...)
            h = self.bn3(h)
            h, v3 = self.lif3(h, v3)                    # spike + update v3
            h = self.pool3(h)                           # (B, F2, 1, T//32)
            h = self.drop3(h)

            spike_sum += h.flatten(1)

        # Mean firing rate → logits
        return self.classifier(spike_sum / self.T_sim)


# ─────────────────────────────────────────────────────────────────────────────
# CSP filter initialization (same as CSPNet)
# ─────────────────────────────────────────────────────────────────────────────

def fit_csp_layer_snn(
    model: CSPNetSNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """
    Initialize model.csp_layer.W from training data (same as CSPNet).

    Parameters
    ----------
    model    : CSPNetSNN instance (already on target device)
    X_train  : (N, C, T) float32, UN-normalized training epochs
    y_train  : (N,) int, binary labels {0, 1}
    """
    n_csp = model.csp_layer.n_csp
    W = compute_csp_filters(X_train, y_train, n_filters=n_csp)
    model.csp_layer.init_from_numpy(W)
    print(f"    [CSPNetSNN] CSP layer initialized "
          f"(n_csp={n_csp}, "
          f"trainable={isinstance(model.csp_layer.W, nn.Parameter)}, "
          f"T_sim={model.T_sim})")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=== CSPNetSNN sanity check ===\n")

    for ds, C, T in [("Cho2017", 64, 257), ("Lee2019", 62, 200)]:
        for enc in ("direct", "rate"):
            model = CSPNetSNN(
                n_channels=C, n_times=T,
                T_sim=8, encoding=enc,
            )
            model.eval()
            x = torch.randn(8, C, T)

            with torch.no_grad():
                logits = model(x)

            n_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

            # Inference time (single trial, CPU)
            x1 = torch.randn(1, C, T)
            for _ in range(20): model(x1)
            t0 = time.perf_counter()
            for _ in range(200):
                with torch.no_grad(): model(x1)
            t_ms = (time.perf_counter() - t0) / 200 * 1000

            print(f"{ds:10s}  enc={enc:6s}  "
                  f"logits={tuple(logits.shape)}  "
                  f"params={n_params:,}  "
                  f"infer={t_ms:.2f}ms/trial (CPU)")

    print()
    print("=== CSP filter initialization ===")
    C, T = 64, 257
    model = CSPNetSNN(n_channels=C, n_times=T)
    X_fake = np.random.randn(80, C, T).astype(np.float32)
    y_fake = np.array([0]*40 + [1]*40)
    fit_csp_layer_snn(model, X_fake, y_fake)
    print("CSP init OK")

    print()
    print("=== Gradient flow check (surrogate gradient) ===")
    model = CSPNetSNN(n_channels=64, n_times=257)
    model.train()
    x = torch.randn(4, 64, 257, requires_grad=False)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    grad_csp = model.csp_layer.W.grad
    grad_cls = model.classifier.weight.grad
    print(f"  csp_layer.W grad norm  : {grad_csp.norm().item():.4f}")
    print(f"  classifier.weight grad : {grad_cls.norm().item():.4f}")
    print("  Gradients flow through LIF surrogate: OK" if grad_csp.norm() > 0
          else "  WARNING: zero gradients")
