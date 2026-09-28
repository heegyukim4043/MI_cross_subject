"""
EEG signal-level and Riemannian manifold augmentation for MI classification.

Signal-level (applied to raw epoch tensors before covariance):
  - TimeJitter       : random cyclic shift along time axis
  - ChannelAmplitude : per-channel random gain  (changes correlation structure)
  - NoiseInjection   : additive Gaussian noise
  - NOTE: FrequencyShift is intentionally excluded — it destroys mu/beta band
          structure that is essential for MI discrimination.

Manifold-level (applied to SPD covariance matrices):
  - RiemannianMixup  : geodesic interpolation between two SPD matrices
                       Σ(t) = A^½ (A^-½ B A^-½)^t A^½,  t ~ U(0, alpha)
                       Label interpolated as soft target.
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Signal-level augmentation
# ---------------------------------------------------------------------------

class EEGAugment(nn.Module):
    """
    Applies stochastic augmentation to a batch of EEG signals.

    Input : (B, C, T)  float32 tensor (already z-score normalised)
    Output: (B, C, T)  augmented tensor

    Each augmentation is applied independently with probability `p`.
    """

    def __init__(
        self,
        p: float = 0.5,
        jitter_ms: float = 50.0,          # max ± shift in milliseconds
        sfreq: float = 128.0,             # sampling frequency (Hz)
        amp_range: tuple = (0.8, 1.2),   # per-channel scale range
        noise_std: float = 0.05,          # Gaussian noise std (signal is z-scored)
    ):
        super().__init__()
        self.p         = p
        self.jitter_T  = max(1, int(jitter_ms * sfreq / 1000.0))  # samples
        self.amp_range = amp_range
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        if not self.training:
            return x

        B, C, T = x.shape

        # -- Time jitter (cyclic roll per sample) ----------------------------
        if self.jitter_T > 0 and torch.rand(1).item() < self.p:
            shifts = torch.randint(
                -self.jitter_T, self.jitter_T + 1, (B,)
            )
            rolled = []
            for i, s in enumerate(shifts.tolist()):
                rolled.append(torch.roll(x[i], int(s), dims=-1))
            x = torch.stack(rolled, dim=0)

        # -- Per-channel amplitude scaling -----------------------------------
        if torch.rand(1).item() < self.p:
            lo, hi = self.amp_range
            scale = x.new_empty(B, C, 1).uniform_(lo, hi)
            x = x * scale

        # -- Additive Gaussian noise -----------------------------------------
        if torch.rand(1).item() < self.p:
            x = x + torch.randn_like(x) * self.noise_std

        return x

    def extra_repr(self):
        return (f"p={self.p}, jitter_T={self.jitter_T}samp, "
                f"amp={self.amp_range}, noise_std={self.noise_std}")


# ---------------------------------------------------------------------------
# Riemannian Mixup  (manifold-level, batch-wise)
# ---------------------------------------------------------------------------

def _spd_geodesic_batch(A: torch.Tensor, B: torch.Tensor, t: float) -> torch.Tensor:
    """
    Batch geodesic interpolation on the SPD manifold.

    Computes  A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}  for each matrix pair.

    A, B : (batch, n, n)  symmetric positive-definite
    t    : scalar in [0, 1]
    Returns (batch, n, n)
    """
    # Cholesky:  A = L L^T
    L   = torch.linalg.cholesky(A)           # (B, n, n)
    L_t = L.transpose(-2, -1)

    # L^{-1} B L^{-T}  via solve: L X = B  => X = L^{-1} B
    # Then solve X^T L^T = ? => use triangular solve twice
    # Equivalent: M = L^{-1} B L^{-T}
    Linv_B = torch.linalg.solve_triangular(L, B, upper=False)       # L^{-1} B
    M      = torch.linalg.solve_triangular(L_t, Linv_B.transpose(-2,-1), upper=True).transpose(-2,-1)
    # M is symmetric positive definite

    # Symmetrize for numerical stability
    M = 0.5 * (M + M.transpose(-2, -1))

    # Eigen-decomposition of M
    eigvals, eigvecs = torch.linalg.eigh(M)          # (B, n), (B, n, n)
    eigvals = eigvals.clamp(min=1e-6)                # ensure positive
    eigvals_t = eigvals.pow(t)                       # element-wise ^t

    # M^t = V diag(λ^t) V^T
    M_t = eigvecs @ torch.diag_embed(eigvals_t) @ eigvecs.transpose(-2, -1)

    # Result = L M^t L^T
    result = L @ M_t @ L_t
    return 0.5 * (result + result.transpose(-2, -1))


class RiemannianMixup:
    """
    Mixup on the SPD manifold.

    Given a batch of covariance matrices (B, C, C) and labels,
    interpolates pairs along the geodesic with ratio t ~ U(0, alpha).

    Designed to be inserted AFTER CovarianceLayer in the training loop.

    Usage
    -----
    mixup = RiemannianMixup(alpha=0.4)
    Sigma_mix, y_a, y_b, t = mixup(Sigma, y)
    logits = rest_of_model(Sigma_mix)
    loss = t * criterion(logits, y_a) + (1-t) * criterion(logits, y_b)
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(self, Sigma: torch.Tensor, y: torch.Tensor):
        """
        Sigma : (B, C, C) SPD batch
        y     : (B,) int64 labels

        Returns
        -------
        Sigma_mix : (B, C, C)
        y_a, y_b  : (B,)  original and permuted labels
        t         : float  mixing ratio (closer to 0 = more original)
        """
        B = Sigma.shape[0]
        t = float(np.random.beta(self.alpha, self.alpha))
        t = max(t, 1 - t)                     # keep dominant class dominant

        perm = torch.randperm(B, device=Sigma.device)
        A = Sigma
        B_ = Sigma[perm]
        y_a = y
        y_b = y[perm]

        Sigma_mix = _spd_geodesic_batch(A, B_, 1 - t)  # t=0 -> A, t=1 -> B
        return Sigma_mix, y_a, y_b, t


# ---------------------------------------------------------------------------
# Mixup-aware cross-entropy loss
# ---------------------------------------------------------------------------

def mixup_loss(criterion, logits, y_a, y_b, t):
    """t * CE(y_a) + (1-t) * CE(y_b)"""
    return t * criterion(logits, y_a) + (1 - t) * criterion(logits, y_b)
