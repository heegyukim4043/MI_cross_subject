"""Euclidean Alignment (EA) for EEG cross-subject transfer.

EA whitens each subject independently with that subject's mean covariance.
The transform uses unlabeled EEG only. The default ``relative`` floor is scale-consistent:
results do not depend on the amplitude unit of the stored EEG. ``legacy`` (absolute floor)
is kept only to reproduce superseded results and violates this scale equivariance.

Reference: He et al., IEEE Transactions on Biomedical Engineering, 2019.
"""

import os

import numpy as np


EA_EPS_MODES = ("legacy", "absolute64", "relative", "rank_aware")


def euclidean_align(
    X: np.ndarray,
    eps: float = 1e-8,
    eps_mode=None,
    rank_rtol: float = 1e-12,
) -> np.ndarray:
    """Whiten one subject's ``(trial, channel, time)`` EEG array.

    ``legacy`` uses float32 covariance and an absolute eigenvalue floor.
    ``absolute64`` uses the same floor in float64. ``relative`` uses
    ``eps * trace(R) / C`` and is invariant to global amplitude scaling.
    ``rank_aware`` applies a truncated inverse with a relative rank threshold.
    As in the original project and reference implementation, trial covariance
    is not divided by the number of time samples.
    """
    if eps_mode is None:
        eps_mode = os.environ.get("MI_EA_FLOOR", "relative")
    if eps_mode not in EA_EPS_MODES:
        raise ValueError(f"Unknown eps_mode={eps_mode!r}; choose from {EA_EPS_MODES}")
    if X.ndim != 3 or len(X) == 0:
        raise ValueError(f"X must be a non-empty (N, C, T) array, got {X.shape}")

    _, n_channels, _ = X.shape
    if eps_mode == "legacy":
        covariance = np.mean([trial @ trial.T for trial in X], axis=0)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.maximum(eigvals, eps)
        whitener = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
        aligned = np.einsum("cd,ndt->nct", whitener, X)
        return aligned.astype(np.float32)

    dtype = np.float64
    X_cov = X.astype(dtype, copy=False)
    covariance = np.mean(
        np.einsum("nct,ndt->ncd", X_cov, X_cov),
        axis=0,
        dtype=dtype,
    )
    covariance = (covariance + covariance.T) * 0.5

    eigvals, eigvecs = np.linalg.eigh(covariance)
    if eps_mode == "absolute64":
        inverse_eigvals = np.maximum(eigvals, eps) ** -0.5
    elif eps_mode == "relative":
        floor = eps * float(np.trace(covariance)) / n_channels
        inverse_eigvals = np.maximum(eigvals, floor) ** -0.5
    else:
        floor = rank_rtol * float(max(eigvals[-1], 0.0))
        keep = eigvals > floor
        inverse_eigvals = np.zeros_like(eigvals)
        inverse_eigvals[keep] = eigvals[keep] ** -0.5

    whitener = (eigvecs * inverse_eigvals) @ eigvecs.T
    aligned = np.einsum("cd,ndt->nct", whitener, X, optimize=True)
    return aligned.astype(np.float32)


def apply_ea_loso(
    X: np.ndarray,
    subjects: np.ndarray,
    eps: float = 1e-8,
    eps_mode=None,
    rank_rtol: float = 1e-12,
) -> np.ndarray:
    """Apply EA independently to every subject in a dataset."""
    if len(X) != len(subjects):
        raise ValueError(f"X/subjects length mismatch: {len(X)} != {len(subjects)}")

    aligned = np.empty_like(X, dtype=np.float32)
    for subject in np.unique(subjects):
        mask = subjects == subject
        aligned[mask] = euclidean_align(
            X[mask], eps=eps, eps_mode=eps_mode, rank_rtol=rank_rtol
        )
    return aligned
