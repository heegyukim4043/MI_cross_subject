"""Render existing Results estimates/intervals; never edit manuscript files."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
CURRENT = HERE.parent.parent
ROOT = CURRENT.parent.parent
OUT = CURRENT / "figures"
SOURCE1 = ROOT / "analysis_outputs/cmpb_paired_policy_ci_20261007/results/bootstrap_summary.csv"
SOURCE2 = ROOT / "MI cross subject journal/figure/fig3_rq1_v3_source.csv"
TEAL, RED = "#167A75", "#A83D54"


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def unique(rows, **query):
    selected = [r for r in rows if all(r[k] == v for k, v in query.items())]
    if len(selected) != 1:
        raise ValueError((query, len(selected)))
    return selected[0]


def intervals(rows, mean, low, high):
    values = np.array([float(r[mean]) for r in rows])
    lower = np.array([float(r[low]) for r in rows])
    upper = np.array([float(r[high]) for r in rows])
    assert np.isfinite([values, lower, upper]).all()
    assert ((lower <= values) & (values <= upper)).all()
    return values, np.array([values - lower, upper - values]), upper


def style(ax, interval):
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(interval))
    ax.grid(axis="y", color="#CBD0D3", linewidth=0.65, linestyle="--", alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#555555")
    ax.tick_params(axis="x", length=0, pad=8)


def export(fig, name):
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = records(SOURCE1)
    fig, ax = plt.subplots(figsize=(7.4, 4.0), layout="constrained")
    plot_rows = []
    for metrics, label, color, hatch, offset in [
        (["absolute_S", "relative_S"], "SubjectEA", TEAL, None, -0.18),
        (["absolute_DS", "relative_DS"], "DatasetEA -> SubjectEA", RED, "//", 0.18),
    ]:
        chosen = [unique(rows, scope="equal_dataset", scale="1", metric=m) for m in metrics]
        assert all(r["n"] == "224" and r["unit"] == "percent" for r in chosen)
        values, errors, upper = intervals(chosen, "estimate", "ci95_low", "ci95_high")
        x = np.arange(2) + offset
        ax.bar(x, values, width=0.32, yerr=errors, capsize=4, label=label,
               color=color, hatch=hatch, edgecolor="white", linewidth=0.7,
               error_kw={"ecolor": "#222222", "elinewidth": 1.1, "capthick": 1.1})
        for xx, yy, hi in zip(x, values, upper):
            ax.text(xx, hi + 1.3, f"{yy:.2f}", ha="center", va="bottom", fontsize=10)
        plot_rows.extend(chosen)
    style(ax, 10)
    ax.set_ylim(0, 85)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1], ["Coherent absolute", "Relative"])
    ax.set_ylabel("Equal-dataset BAC (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=2,
              frameon=False, fontsize=10)
    export(fig, "Figure_1_bar")

    gap = records(SOURCE2)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), layout="constrained")
    gap_rows = []
    specs = [
        ("a", ["alignment (none -> relative): measured under LOSO",
               "alignment (none -> relative): measured under LODO"],
         ["LOSO", "LODO"], "(a) Alignment gains", "BAC gain (pp)", (0, 9), 2),
        ("b", ["domain (LODO -> LOSO) with none", "domain (LODO -> LOSO) with relative"],
         ["No alignment", "Relative EA"], "(b) LOSO - LODO gap", "BAC gap (pp)", (-0.35, 2.9), 0.5),
    ]
    for ax, (panel, keys, labels, title, ylabel, ylim, tick) in zip(axes, specs):
        chosen = [unique(gap, panel=panel, dataset="all", quantity=k) for k in keys]
        values, errors, upper = intervals(chosen, "mean_pp", "ci_low", "ci_high")
        for i, (value, hi, color, hatch) in enumerate(zip(values, upper, [TEAL, RED], [None, "//"])):
            ax.bar(i, value, width=0.52, yerr=errors[:, i:i+1], capsize=5,
                   color=color, edgecolor="white", hatch=hatch, linewidth=0.7,
                   error_kw={"ecolor": "#222222", "elinewidth": 1.1, "capthick": 1.1})
            ax.text(i, hi + (ylim[1] - ylim[0]) * 0.035, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=10)
        style(ax, tick)
        ax.axhline(0, color="#555555", linewidth=0.9)
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=14, fontsize=11)
        gap_rows.extend(chosen)
    export(fig, "Figure_2_bar")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    report = {
        "scope": "Separate figure exports only; manuscript and original figures unchanged",
        "figure_1": {"estimator": "Equal-weight mean of the four dataset means, 224 subjects",
                     "uncertainty": "Stored 95% subject-bootstrap percentile CI; 10000 draws, seed 20261007",
                     "rows": plot_rows},
        "figure_2": {"estimator": "Pooled subject means of paired contrasts, 224 subjects",
                     "uncertainty": "Archived 95% subject-bootstrap CI, reused without recomputation",
                     "rows": gap_rows},
        "sources_sha256": {str(p): sha(p) for p in (SOURCE1, SOURCE2)},
        "exports_sha256": {p.name: sha(p) for p in OUT.glob("Figure_*_bar.*")},
        "visual_review": "pending",
    }
    (HERE / "figure_bar_qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "FIGURE_NOTES.md").write_text(
        "# Separate Results bar figures\n\n"
        "Figure_1_bar: equal-dataset BAC means (4 datasets, 224 subjects), with existing 95% subject-bootstrap CIs. "
        "The y axis starts at zero. Individual mean CIs are not CIs for paired method differences; "
        "do not infer paired significance from their overlap.\n\n"
        "Figure_2_bar: pooled alignment gains and LOSO-minus-LODO gaps, with archived 95% subject-bootstrap CIs. "
        "The two panels deliberately have different y-axis ranges because they show different quantities. "
        "No new bootstrap, fitting or numerical inference.\n\n"
        "Bars show point estimates; black whiskers show 95% CIs; dashed horizontal grids aid comparison. "
        "Original figure files, Word, Markdown and manuscript PDFs are not changed.\n",
        encoding="utf-8")
    print(json.dumps({"outputs": list(report["exports_sha256"]), "qa": str(HERE / "figure_bar_qa.json")}))


if __name__ == "__main__":
    main()
