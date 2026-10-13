"""Display archived mean and subject-paired contrast intervals, without resampling."""
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from draw_results_bars import HERE, ROOT, OUT, SOURCE1, TEAL, RED, records, unique, intervals, style

ANALYSIS = ROOT / "analysis_outputs/cmpb_paired_policy_ci_20261007"
DRAW_FILE = ANALYSIS / "results/aggregate_bootstrap_draws.npz"
METRICS = ["absolute_S", "absolute_DS", "relative_S", "relative_DS",
           "absolute_increment", "relative_increment", "increment_difference"]


def export(fig, name):
    for fmt in ["pdf", "png"]:
        data = io.BytesIO()
        fig.savefig(data, format=fmt, dpi=300, facecolor="white")
        (OUT / f"{name}.{fmt}").write_bytes(data.getvalue())
    plt.close(fig)


def main():
    rows = records(SOURCE1)
    chosen = {m: unique(rows, scope="equal_dataset", scale="1", metric=m) for m in METRICS}
    assert all(r["n"] == "224" and r["seed"] == "20261007" and r["resamples"] == "10000"
               for r in chosen.values())
    # Check saved draws, not a new bootstrap. Pairing is encoded by the original estimator.
    with np.load(DRAW_FILE, allow_pickle=False) as draws:
        names = list(draws["metric_names"])
        si = list(draws["scales"]).index("1")
        assert draws["equal_dataset"].shape[0] == 10000
        samples = draws["equal_dataset"][:, si, :]
        np.testing.assert_allclose(
            samples[:, names.index("increment_difference")],
            samples[:, names.index("absolute_increment")] - samples[:, names.index("relative_increment")],
            rtol=0, atol=1e-12)
        for m, row in chosen.items():
            bounds = np.quantile(draws["equal_dataset"][:, si, names.index(m)], [.025, .975], method="linear")
            np.testing.assert_allclose(bounds, [float(row["ci95_low"]), float(row["ci95_high"])], atol=1e-12)
    means = {m: float(r["estimate"]) for m, r in chosen.items()}
    np.testing.assert_allclose(means["absolute_DS"] - means["absolute_S"], means["absolute_increment"], atol=1e-12)
    np.testing.assert_allclose(means["relative_DS"] - means["relative_S"], means["relative_increment"], atol=1e-12)
    np.testing.assert_allclose(means["absolute_increment"] - means["relative_increment"], means["increment_difference"], atol=1e-12)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5), layout="constrained",
                             gridspec_kw={"width_ratios": [1.12, 1]})
    left, right = axes
    for metrics, label, color, hatch, offset in [
        (["absolute_S", "relative_S"], "SubjectEA", TEAL, None, -0.18),
        (["absolute_DS", "relative_DS"], "DatasetEA -> SubjectEA", RED, "//", 0.18),
    ]:
        values, errors, upper = intervals([chosen[m] for m in metrics], "estimate", "ci95_low", "ci95_high")
        x = np.arange(2) + offset
        left.bar(x, values, width=.32, yerr=errors, capsize=4, label=label,
                 color=color, edgecolor="white", linewidth=.7, hatch=hatch,
                 error_kw={"ecolor": "#222222", "elinewidth": 1.1, "capthick": 1.1})
        for xx, value, hi in zip(x, values, upper):
            left.text(xx, hi + 1.5, f"{value:.2f}", ha="center", va="bottom", fontsize=9.5)
    style(left, 20)
    left.set(ylim=(0, 100), xlim=(-.55, 1.55), ylabel="Equal-dataset BAC (%)")
    left.set_xticks([0, 1], ["Coherent\nabsolute", "Relative"])
    left.set_title("(a) Four pipeline BACs", loc="left", fontsize=11, pad=12)
    left.legend(loc="upper left", fontsize=9, frameon=False)

    metrics = ["absolute_increment", "relative_increment", "increment_difference"]
    values, errors, upper = intervals([chosen[m] for m in metrics], "estimate", "ci95_low", "ci95_high")
    for i, (value, hi, color, hatch) in enumerate(zip(values, upper, ["#7D858A", "#7D858A", "#60676D"], ["//", None, "xx"])):
        right.bar(i, value, width=.55, yerr=errors[:, i:i+1], capsize=4,
                  color=color, edgecolor="white", linewidth=.7, hatch=hatch,
                  error_kw={"ecolor": "#222222", "elinewidth": 1.1, "capthick": 1.1})
        label = f"+{value:.2f}"
        if i == 2:
            label += f"\n[{float(chosen[metrics[i]]['ci95_low']):.2f}, {float(chosen[metrics[i]]['ci95_high']):.2f}]"
        right.text(i, hi + .2, label, ha="center", va="bottom", fontsize=9.5, linespacing=1.25)
    style(right, 1)
    right.axhline(0, color="#555555", linewidth=.9)
    right.axvline(1.5, color="#ADB3B7", linewidth=.8, linestyle=":")
    right.set(ylim=(-.7, 7.1), xlim=(-.55, 2.65), ylabel="Increment / contrast (pp)")
    right.set_xticks([0, 1, 2], ["Coherent\nabsolute", "Relative", "Paired\ncontrast $K$"])
    right.set_title("(b) Incremental DatasetEA gains", loc="left", fontsize=11, pad=12)
    right.set_xlabel(
        r"Increment = BAC(D$\rightarrow$S) $-$ BAC(S)" + "\n" +
        r"Contrast $K = \Delta_{\mathrm{absolute}} - \Delta_{\mathrm{relative}}$",
        fontsize=9, labelpad=9)
    export(fig, "Figure_1_bar")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    report = {
        "scope": "Figure 1 separate exports only; no manuscript edits; Figure 2 unchanged",
        "estimator": "Mean of four dataset means with weight 1/4, 224 subjects",
        "uncertainty": {
            "type": "95% percentile CI (2.5th/97.5th percentiles, linear interpolation)",
            "sampling_unit": "Subjects resampled with replacement within each fixed dataset",
            "pairing": "Same sampled subject indices across all four conditions; contrasts calculated per subject",
            "resamples": 10000, "seed": 20261007, "rng": "NumPy PCG64",
            "status": "Exploratory inference specified after observing point estimates",
            "limitations": "Marginal conditional intervals; no retraining, trial sampling, cohort/reference-estimation or new-dataset uncertainty",
            "draw_quantiles_match_all_seven_csv_intervals": True,
            "K_equals_paired_increment_difference_on_every_saved_draw": True,
            "panel_a": "Marginal condition-mean intervals",
            "panel_b": "Marginal intervals for each paired within-policy increment and directly paired K",
        },
        "rows": chosen,
        "sources_sha256": {str(p): sha(p) for p in [SOURCE1, DRAW_FILE, ANALYSIS / "PROTOCOL.md", ANALYSIS / "analyze.py"]},
        "exports_sha256": {p.name: sha(p) for p in [OUT / "Figure_1_bar.pdf", OUT / "Figure_1_bar.png"]},
        "visual_review": "pending",
    }
    (HERE / "figure1_two_panel_qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "FIGURE_1_CAPTION_DRAFT.txt").write_text(
        "Fig. 1. Baseline-dependent incremental gains from additional dataset-level alignment. "
        "(a) Balanced accuracy (BAC) of SubjectEA and DatasetEA->SubjectEA under coherent absolute and relative "
        "regularization policies. (b) Incremental DatasetEA gains under each policy and their paired difference "
        "K (absolute minus relative), not a fifth pipeline. Results equally weight the four fixed development "
        "dataset means (224 subjects). Black error bars indicate marginal 95% percentile bootstrap confidence "
        "intervals: condition-mean intervals in (a), paired within-policy increment intervals and the directly "
        "paired K interval in (b). Subjects were resampled with replacement within each dataset for 10,000 draws, "
        "retaining the same sampled indices across all conditions (seed 20261007); each draw equally weighted "
        "the four dataset means. K was +4.43 pp (95% CI, 3.34-5.52), calculated directly from paired differences "
        "rather than by subtracting individual interval endpoints. The intervals are exploratory, conditional "
        "on fitted models, fixed datasets and alignment references, and not simultaneous; they omit retraining, "
        "trial-sampling and reference-estimation uncertainty. DatasetEA->SubjectEA additionally uses "
        "target-cohort EEG. Panel scales differ.\n",
        encoding="utf-8")
    print(json.dumps({"figure": str(OUT / "Figure_1_bar.png"), "verified_intervals": len(chosen)}))


if __name__ == "__main__":
    main()
