"""
Trial suitability scoring for LOSO train pool selection.

All scoring operates on the already-preprocessed (N, C, T) array
(bandpass 8-30 Hz, epoched [0.5, 2.5] s) — no raw data or baseline needed.

Scoring methods
---------------
band_power   : log-variance of each trial across channels and time.
               Trials far from the median (too high = artifact, too low = flat)
               get low scores.  Works for any dataset.

laterality   : |log-var(C3) - log-var(C4)| / (log-var(C3) + log-var(C4))
               Stronger lateralized mu/beta suppression = higher score.
               Falls back to band_power when C3 or C4 is absent.

cov_quality  : 1 / condition_number(Σ).  Well-conditioned covariance matrices
               score highest — best suited for Riemannian / SPDNet models.

combined     : laterality * band_power_quality  (element-wise product,
               both normalised to [0, 1]).  Default.

Selection
---------
select_trials(X, y, scores, keep_ratio, balanced=True)
    keep_ratio = 1.0  →  no-op, returns X and y unchanged.
    balanced   = True →  keep the same ratio from each class (0/1) so the
                          class balance is preserved.

score_to_weights(scores, y, keep_ratio, balanced=True, min_weight=0.25)
    Converts trial scores to soft per-sample weights for weighted training.
    keep_ratio controls the score threshold that receives full emphasis.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _log_var(X: np.ndarray) -> np.ndarray:
    """Per-trial log-variance across all channels and time. Shape: (N,)"""
    var = X.var(axis=(1, 2)) + 1e-10          # (N,)
    return np.log(var)


def score_band_power(X: np.ndarray, **_) -> np.ndarray:
    """
    Quality score based on log-variance proximity to the median.
    Outliers (artifacts or flat signals) score low; typical trials score high.
    Returns scores in [0, 1].
    """
    lv    = _log_var(X)                        # (N,)
    med   = np.median(lv)
    mad   = np.median(np.abs(lv - med)) + 1e-8 # robust spread
    z     = np.abs(lv - med) / mad             # robust z-score
    score = np.exp(-0.5 * z)                   # Gaussian-shaped penalty
    return score / (score.max() + 1e-10)


def score_laterality(X: np.ndarray, ch_names: list, **_) -> np.ndarray:
    """
    Laterality index: |log-var(C3) - log-var(C4)| / (log-var(C3) + log-var(C4))
    Falls back to band_power when C3 or C4 is missing.
    Returns scores in [0, 1].
    """
    c3_idx = next((i for i, ch in enumerate(ch_names)
                   if ch.upper() == "C3"), None)
    c4_idx = next((i for i, ch in enumerate(ch_names)
                   if ch.upper() == "C4"), None)

    if c3_idx is None or c4_idx is None:
        return score_band_power(X)

    lv_c3 = _log_var(X[:, c3_idx:c3_idx+1, :])   # (N,)
    lv_c4 = _log_var(X[:, c4_idx:c4_idx+1, :])   # (N,)
    denom  = np.abs(lv_c3) + np.abs(lv_c4) + 1e-8
    lat    = np.abs(lv_c3 - lv_c4) / denom        # [0, 1] by construction

    # Also penalise artifact/flat trials via band_power quality
    bpq   = score_band_power(X)
    score = lat * bpq
    return score / (score.max() + 1e-10)


def score_cov_quality(X: np.ndarray, **_) -> np.ndarray:
    """
    1 / condition_number(Σ) where Σ is the Tikhonov-regularised covariance.
    Better-conditioned matrices score higher. Returns scores in [0, 1].
    """
    N, C, T = X.shape
    eps     = 1e-5
    scores  = np.empty(N, dtype=np.float32)

    for i in range(N):
        x   = X[i]                              # (C, T)
        x   = x - x.mean(axis=1, keepdims=True)
        cov = x @ x.T / (T - 1)
        cov = cov + eps * np.eye(C)             # Tikhonov
        try:
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.clip(eigvals, 1e-10, None)
            cond    = eigvals[-1] / eigvals[0]
            scores[i] = 1.0 / cond
        except np.linalg.LinAlgError:
            scores[i] = 0.0

    scores = scores / (scores.max() + 1e-10)
    return scores


def score_combined(X: np.ndarray, ch_names: list, **_) -> np.ndarray:
    """
    Geometric mean of laterality and cov_quality scores.
    """
    lat = score_laterality(X, ch_names)
    cov = score_cov_quality(X)
    score = np.sqrt(lat * cov + 1e-10)
    return score / (score.max() + 1e-10)


_METHODS = {
    "band_power" : score_band_power,
    "laterality" : score_laterality,
    "cov_quality": score_cov_quality,
    "combined"   : score_combined,
}


def score_trials(
    X: np.ndarray,
    ch_names: list,
    method: str = "combined",
) -> np.ndarray:
    """
    Score each trial in X.

    Parameters
    ----------
    X        : (N, C, T) float32
    ch_names : list of str, length C
    method   : one of 'band_power', 'laterality', 'cov_quality', 'combined'

    Returns
    -------
    scores : (N,) float32, range [0, 1]
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Choose from {list(_METHODS.keys())}")
    return _METHODS[method](X, ch_names=ch_names).astype(np.float32)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def selection_indices(
    y: np.ndarray,
    scores: np.ndarray,
    keep_ratio: float,
    balanced: bool = True,
) -> np.ndarray:
    """
    Return the sorted indices kept by top-k selection.

    This mirrors the selection logic used by select_trials() so that
    visualisation and filtering use exactly the same ranking rule.
    """
    if keep_ratio >= 1.0:
        return np.arange(len(y), dtype=np.int64)

    scores = np.asarray(scores, dtype=np.float32)
    y = np.asarray(y)

    if balanced:
        keep_idx = []
        for cls in np.unique(y):
            cls_mask = y == cls
            cls_idx = np.where(cls_mask)[0]
            cls_scores = scores[cls_idx]
            n_keep = max(1, int(np.ceil(len(cls_idx) * keep_ratio)))
            top_local = np.argsort(cls_scores)[::-1][:n_keep]
            keep_idx.extend(cls_idx[top_local].tolist())
        keep_idx = np.array(sorted(keep_idx), dtype=np.int64)
    else:
        n_keep = max(1, int(np.ceil(len(y) * keep_ratio)))
        keep_idx = np.argsort(scores)[::-1][:n_keep]
        keep_idx = np.sort(keep_idx).astype(np.int64)
    return keep_idx


def selection_mask(
    y: np.ndarray,
    scores: np.ndarray,
    keep_ratio: float,
    balanced: bool = True,
) -> np.ndarray:
    """Boolean mask of selected trials."""
    mask = np.zeros(len(y), dtype=bool)
    mask[selection_indices(y, scores, keep_ratio, balanced=balanced)] = True
    return mask


def selection_thresholds(
    y: np.ndarray,
    scores: np.ndarray,
    keep_ratio: float,
    balanced: bool = True,
) -> dict:
    """
    Threshold summary for the current keep_ratio.

    Returns:
      {
        "global": float,
        "per_class": {class_id: float, ...},
      }
    """
    scores = np.asarray(scores, dtype=np.float32)
    y = np.asarray(y)
    q = max(0.0, 1.0 - keep_ratio)

    out = {
        "global": float(np.quantile(scores, q)) if len(scores) else 0.0,
        "per_class": {},
    }
    if balanced:
        for cls in np.unique(y):
            cls_scores = scores[y == cls]
            out["per_class"][int(cls)] = float(np.quantile(cls_scores, q))
    return out

def select_trials(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    keep_ratio: float,
    balanced: bool = True,
) -> tuple:
    """
    Keep the top-keep_ratio fraction of training trials by score.

    Parameters
    ----------
    X, y      : training data, shape (N, C, T) and (N,)
    scores    : (N,) quality scores from score_trials()
    keep_ratio: float in (0, 1].  1.0 → return X, y unchanged.
    balanced  : if True, apply the same keep_ratio within each class
                so the class balance is preserved.

    Returns
    -------
    X_sel, y_sel : filtered arrays
    """
    if keep_ratio >= 1.0:
        return X, y

    keep_idx = selection_indices(y, scores, keep_ratio, balanced=balanced)

    return X[keep_idx], y[keep_idx]


def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] robustly."""
    scores = np.asarray(scores, dtype=np.float32)
    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max <= s_min:
        return np.ones_like(scores, dtype=np.float32)
    return (scores - s_min) / (s_max - s_min)


def score_to_weights(
    scores: np.ndarray,
    y: np.ndarray,
    keep_ratio: float,
    balanced: bool = True,
    min_weight: float = 0.25,
) -> np.ndarray:
    """
    Convert trial scores to soft weights in [min_weight, 1].

    The keep_ratio defines the score quantile that receives full emphasis.
    Scores below that threshold are down-weighted but not discarded.
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1].")
    if not (0.0 < min_weight <= 1.0):
        raise ValueError("min_weight must be in (0, 1].")
    if keep_ratio >= 1.0:
        return np.ones_like(scores, dtype=np.float32)

    scores = np.asarray(scores, dtype=np.float32)
    y = np.asarray(y)
    weights = np.empty_like(scores, dtype=np.float32)

    if balanced:
        class_ids = np.unique(y)
        for cls in class_ids:
            cls_idx = np.where(y == cls)[0]
            cls_scores = _normalise_scores(scores[cls_idx])
            q = max(0.0, 1.0 - keep_ratio)
            thresh = float(np.quantile(cls_scores, q))
            if thresh >= 1.0 - 1e-8:
                cls_w = np.ones_like(cls_scores, dtype=np.float32)
            else:
                cls_w = min_weight + (1.0 - min_weight) * np.clip(
                    cls_scores / max(thresh, 1e-6), 0.0, 1.0
                )
            weights[cls_idx] = cls_w.astype(np.float32)
    else:
        norm_scores = _normalise_scores(scores)
        q = max(0.0, 1.0 - keep_ratio)
        thresh = float(np.quantile(norm_scores, q))
        if thresh >= 1.0 - 1e-8:
            weights.fill(1.0)
        else:
            weights = min_weight + (1.0 - min_weight) * np.clip(
                norm_scores / max(thresh, 1e-6), 0.0, 1.0
            )

    return weights.astype(np.float32)
