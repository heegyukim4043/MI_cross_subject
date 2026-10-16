"""Archived plot-only extraction. Set FIG to a fresh directory before calling draw_sweep.
No model fitting, bootstrap or Word editing; paths must be configured by the caller.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

FIG = None
TEAL, RED = "#167A75", "#A83D54"
LEVELS = ["0", "1e-14", "1e-13", "1e-12", "1e-11", "1e-10", "1e-9", "1e-8", "1e-7", "1e-6"]

def plot_style(ax, interval):
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(interval))
    ax.grid(axis="y", color="#CBD0D3", linewidth=.65, linestyle="--", alpha=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#555555")
    ax.tick_params(axis="x", length=0, pad=8)

def draw_sweep(rows):
    # All ten frozen grid points, categorical spacing; zero is never log-transformed.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.3), layout="constrained")
    x = np.arange(len(rows))
    labels = ["0"] + [r"$10^{%d}$" % int(s.split("e")[1]) for s in LEVELS[1:]]
    error_kw = {"ecolor": "#222222", "elinewidth": 1.1, "capthick": 1.1}
    for stem, label, color, hatch, offset in [
        ("S", "SubjectEA", TEAL, None, -.18),
        ("DS", "DatasetEA -> SubjectEA", RED, "//", .18),
    ]:
        mean = np.array([float(r[stem+"_pct"]) for r in rows])
        lo = np.array([float(r[stem+"_ci_low"]) for r in rows])
        hi = np.array([float(r[stem+"_ci_high"]) for r in rows])
        axes[0].bar(x+offset, mean, width=.32, yerr=[mean-lo, hi-mean], capsize=3,
                    color=color, hatch=hatch, edgecolor="white", linewidth=.7,
                    label=label, error_kw=error_kw)
    axes[0].axhline(63.62847840670991, color="#60676D", linestyle=":", linewidth=1.3,
                    label="Archived no alignment (63.63%)")
    axes[0].set(ylim=(0, 100), ylabel="Equal-dataset BAC (%)")
    axes[0].legend(loc="upper left", ncols=2, frameon=False, fontsize=9)
    axes[0].set_title("(a) Pipeline BACs across absolute floors", loc="left", fontsize=11, pad=12)
    plot_style(axes[0], 20)
    mean = np.array([float(r["increment_pp"]) for r in rows])
    lo = np.array([float(r["increment_ci_low"]) for r in rows])
    hi = np.array([float(r["increment_ci_high"]) for r in rows])
    axes[1].bar(x, mean, width=.55, yerr=[mean-lo, hi-mean], capsize=4,
                color="#7D858A", edgecolor="white", linewidth=.7, hatch="//", error_kw=error_kw)
    for xx, yy, hh in zip(x, mean, hi):
        axes[1].text(xx, hh+.18, f"+{yy:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].axhline(0, color="#555555", linewidth=.9)
    axes[1].set(ylim=(-.8, 8.1), ylabel="Increment (pp)",
                xlabel=r"Raw-reference absolute floor $\epsilon$ (declared V$^2$; categorical grid)")
    axes[1].set_title("(b) Incremental DatasetEA gains", loc="left", fontsize=11, pad=12)
    plot_style(axes[1], 2)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlim(-.6, len(rows)-.4)
    fig.suptitle("Development data: adopted Cho amplitude version", fontsize=10)
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG / ("Figure_2_bar."+fmt), dpi=300, facecolor="white")
    plt.close(fig)
