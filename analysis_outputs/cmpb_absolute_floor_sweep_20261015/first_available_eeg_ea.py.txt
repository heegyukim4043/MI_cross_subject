"""
Euclidean Alignment (EA) for EEG cross-subject transfer.

Aligns each subject's EEG to a common Euclidean space by whitening
with the subject's own mean covariance matrix.

Reference: He et al., "Transfer Learning for Brain-Computer Interfaces:
A Euclidean Space Data Alignment Approach", IEEE TBME 2019.

Usage in LOSO
-------------
Apply per-subject BEFORE pooling and normalization:

    for each subject s:
        X_s_aligned = euclidean_align(X_s)   # uses subject's own R
    X_train = concatenate aligned training subjects
    X_test  = euclidean_align(X_test)         # test subject's own R

This is label-free for the test subject (R only needs X, not y).
"""

import numpy as np


def euclidean_align(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Align a single subject's EEG trials via Euclidean Alignment.

    Parameters
    ----------
    X   : (N, C, T) float32 — raw EEG trials for one subject
    eps : regularisation added to eigenvalues for numerical stability

    Returns
    -------
    X_aligned : (N, C, T) float32 — whitened trials
    """
    N, C, T = X.shape

    # Mean covariance: R = (1/N) Σ x_i x_i^T  (not divided by T — matches paper)
    R = np.mean([x @ x.T for x in X], axis=0)  # (C, C)

    # Symmetric eigendecomposition: R = V Λ V^T
    eigvals, eigvecs = np.linalg.eigh(R)         # ascending order
    eigvals = np.maximum(eigvals, eps)            # numerical floor

    # R^{-1/2} = V Λ^{-1/2} V^T
    R_inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T  # (C, C)

    # Apply: x_aligned = R^{-1/2} x
    X_aligned = np.einsum("cd,ndt->nct", R_inv_sqrt, X)
    return X_aligned.astype(np.float32)


def apply_ea_loso(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """
    Apply EA to every subject independently, in-place style.

    Parameters
    ----------
    X        : (N, C, T) — full dataset
    subjects : (N,)      — subject ID per trial

    Returns
    -------
    X_ea : (N, C, T) — EA-aligned copy
    """
    X_ea = X.copy()
    for subj in np.unique(subjects):
        mask = subjects == subj
        X_ea[mask] = euclidean_align(X[mask])
    return X_ea
