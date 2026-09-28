"""
Covariance Style Transfer Augmentation for EEG.

EA removes each subject's covariance style (whitening).
StyleAug re-applies a randomly sampled training subject's covariance
structure to the whitened training data during training.

EA  (test-time):  x  → R_s^{-1/2} · x          (remove style)
StyleAug (train): x_ea → R_j^{1/2} · x_ea       (add random style)

Combined effect:  x_s  → R_j^{1/2} R_s^{-1/2} x_s
                         ≡ making subject s look like subject j

Reference:
  Analogous to AdaIN / image style transfer, applied in EEG covariance space.
"""

import numpy as np
import torch
import torch.nn as nn


class CovarianceStyleAug(nn.Module):
    """
    Batch-level covariance style transfer for EA-whitened EEG.

    Parameters
    ----------
    subject_covs : dict  {subject_id: R (C, C) mean covariance from PRE-EA data}
    p            : float  probability of applying augmentation per batch
    alpha        : float  interpolation weight  [0,1]; 1.0 = full style transfer,
                          0.5 = geometric mean between identity and R_j
    """

    def __init__(self, subject_covs: dict, p: float = 0.5, alpha: float = 1.0):
        super().__init__()
        self.p     = p
        self.alpha = alpha
        self.subject_ids = list(subject_covs.keys())

        # Precompute Cholesky factors L_j = chol(R_j)^alpha
        # For alpha=1: L_j s.t. L_j L_j^T = R_j  → standard Cholesky
        # For alpha<1: use eigendecomposition for fractional power
        self._L = {}
        for s, R in subject_covs.items():
            R_sym = (R + R.T) / 2
            if alpha == 1.0:
                # Pure style transfer: L s.t. L L^T = R
                R_reg = R_sym + 1e-6 * np.eye(R_sym.shape[0])
                L = np.linalg.cholesky(R_reg).astype(np.float32)
            else:
                # Fractional power R^{alpha/2}: via eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(R_sym)
                eigvals = np.maximum(eigvals, 1e-8)
                L = (eigvecs * (eigvals ** (alpha / 2))[None, :]) @ eigvecs.T
                L = L.astype(np.float32)
            self._L[s] = torch.from_numpy(L)   # (C, C), stored on CPU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T) batch of EA-whitened EEG trials (training only).
        Returns augmented batch of same shape.
        """
        if not self.training or np.random.rand() > self.p:
            return x
        j  = self.subject_ids[np.random.randint(len(self.subject_ids))]
        Lj = self._L[j].to(x.device)           # (C, C)
        return torch.einsum("cd,bdt->bct", Lj, x)

    def set_subjects(self, subject_ids: list):
        """Restrict to a subset of subjects (per LOSO fold)."""
        self.subject_ids = [s for s in subject_ids if s in self._L]


def build_style_aug(X_raw: np.ndarray, subjects: np.ndarray,
                    train_subj_ids: np.ndarray,
                    p: float = 0.5, alpha: float = 1.0) -> "CovarianceStyleAug":
    """
    Compute per-subject mean covariances from raw (pre-EA) data and
    return a CovarianceStyleAug restricted to training subjects.

    X_raw   : (N, C, T) raw EEG (before EA)
    subjects: (N,) subject labels
    train_subj_ids : 1-D array of training subject IDs for this fold
    """
    subj_covs = {}
    for s in train_subj_ids:
        Xs = X_raw[subjects == s].astype(np.float64)   # (n_s, C, T)
        n, c, t = Xs.shape
        Xs2d = Xs.transpose(0, 2, 1).reshape(-1, c)    # (n*T, C)
        R = Xs2d.T @ Xs2d / (n * t)
        R /= (np.trace(R) + 1e-12)
        subj_covs[int(s)] = R.astype(np.float32)

    return CovarianceStyleAug(subj_covs, p=p, alpha=alpha)
