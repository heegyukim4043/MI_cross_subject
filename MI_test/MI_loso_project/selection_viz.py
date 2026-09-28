"""
Selection-score visualisation utilities.

Creates:
1. ranked strip plot with selected trials highlighted
2. score distribution plot with threshold / quantile markers
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trial_selection import selection_mask, selection_thresholds


DEFAULT_QUANTILES = (0.2, 0.4, 0.6, 0.8, 1.0)


def _subsample_order(order: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(order) <= max_points:
        return order
    idx = np.linspace(0, len(order) - 1, num=max_points, dtype=int)
    return order[idx]


def _prepare_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_ranked_selection_plot(
    scores: np.ndarray,
    y: np.ndarray,
    keep_ratio: float,
    out_path: str,
    balanced: bool = True,
    title: str = "",
    max_points: int = 4000,
):
    """
    Save ranked score plot with selected trials highlighted.
    """
    scores = np.asarray(scores, dtype=np.float32)
    y = np.asarray(y)
    chosen = selection_mask(y, scores, keep_ratio, balanced=balanced)
    order = np.argsort(scores)[::-1]
    order = _subsample_order(order, max_points=max_points)

    rank = np.arange(1, len(order) + 1)
    score_ord = scores[order]
    chosen_ord = chosen[order]

    _prepare_parent(out_path)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(rank[~chosen_ord], score_ord[~chosen_ord], s=10, c="#bdbdbd",
               alpha=0.7, label="not selected")
    ax.scatter(rank[chosen_ord], score_ord[chosen_ord], s=12, c="#d62728",
               alpha=0.9, label="selected")
    ax.set_xlabel("Rank (high to low score)")
    ax.set_ylabel("Selection score")
    ax.set_title(title or f"Ranked trial scores (keep={int(keep_ratio * 100)}%)")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_score_distribution_plot(
    scores: np.ndarray,
    y: np.ndarray,
    keep_ratio: float,
    out_path: str,
    balanced: bool = True,
    title: str = "",
    quantiles = DEFAULT_QUANTILES,
):
    """
    Save score histogram with current threshold and keep-ratio quantile markers.
    """
    scores = np.asarray(scores, dtype=np.float32)
    y = np.asarray(y)
    thresh = selection_thresholds(y, scores, keep_ratio, balanced=balanced)

    _prepare_parent(out_path)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(scores, bins=min(50, max(20, len(scores) // 40)),
            color="#4c78a8", alpha=0.75, edgecolor="white")

    for ratio in quantiles:
        q = max(0.0, 1.0 - float(ratio))
        q_val = float(np.quantile(scores, q))
        is_current = abs(float(ratio) - float(keep_ratio)) < 1e-8
        ax.axvline(
            q_val,
            color="#d62728" if is_current else "#7f7f7f",
            linestyle="-" if is_current else "--",
            linewidth=2.0 if is_current else 1.0,
            alpha=0.95 if is_current else 0.65,
            label=f"keep {int(ratio * 100)}%" if ratio < 1.0 else "keep 100%",
        )

    if balanced and thresh["per_class"]:
        for cls, cls_thr in sorted(thresh["per_class"].items()):
            ax.axvline(cls_thr, color="#2ca02c", linestyle=":", linewidth=1.2,
                       alpha=0.8, label=f"class {cls} threshold")

    ax.set_xlabel("Selection score")
    ax.set_ylabel("Trial count")
    ax.set_title(title or f"Score distribution (keep={int(keep_ratio * 100)}%)")
    ax.grid(alpha=0.15, linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, lab in zip(handles, labels):
        if lab not in uniq:
            uniq[lab] = h
    ax.legend(list(uniq.values()), list(uniq.keys()), frameon=False, ncol=2, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_selection_plots(
    scores: np.ndarray,
    y: np.ndarray,
    keep_ratio: float,
    out_dir: str,
    stem: str,
    balanced: bool = True,
    title_prefix: str = "",
    max_points: int = 4000,
):
    """
    Save both ranked and distribution plots for one fold.
    """
    ranked_path = os.path.join(out_dir, f"{stem}_ranked.png")
    dist_path = os.path.join(out_dir, f"{stem}_distribution.png")
    title = title_prefix.strip()

    save_ranked_selection_plot(
        scores=scores,
        y=y,
        keep_ratio=keep_ratio,
        out_path=ranked_path,
        balanced=balanced,
        title=(title + " ranked scores").strip(),
        max_points=max_points,
    )
    save_score_distribution_plot(
        scores=scores,
        y=y,
        keep_ratio=keep_ratio,
        out_path=dist_path,
        balanced=balanced,
        title=(title + " score distribution").strip(),
    )
    return ranked_path, dist_path
