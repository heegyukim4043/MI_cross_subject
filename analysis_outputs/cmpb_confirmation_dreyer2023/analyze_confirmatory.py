"""Apply the pre-registered tests to the completed Dreyer2023 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
MARGIN_PP = 1.0
BOOTSTRAPS = 50000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def one_sided_wilcoxon(diff: np.ndarray) -> float:
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)


def tost(diff_pp: np.ndarray, margin_pp: float = MARGIN_PP) -> tuple[float, float, float]:
    n = len(diff_pp)
    mean = float(diff_pp.mean())
    se = float(diff_pp.std(ddof=1) / np.sqrt(n))
    if se == 0:
        p_value = 0.0 if abs(mean) < margin_pp else 1.0
        return p_value, mean, mean
    p_low = 1 - stats.t.cdf((mean + margin_pp) / se, n - 1)
    p_high = stats.t.cdf((mean - margin_pp) / se, n - 1)
    half = stats.t.ppf(0.95, n - 1) * se
    return float(max(p_low, p_high)), mean - half, mean + half


def bootstrap_ci(diff_pp: np.ndarray, seed: int, confidence: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff_pp, size=(BOOTSTRAPS, len(diff_pp)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def main() -> int:
    args = parse_args()
    suffix = "_smoke" if args.smoke else ""
    floor = pd.read_csv(HERE / f"floor_diagnostics{suffix}.csv")
    loso = pd.read_csv(HERE / f"loso_subject_results{suffix}.csv")
    cross = pd.read_csv(HERE / f"cross_subject_results{suffix}.csv")
    sep = pd.read_csv(HERE / f"separability_splithalf{suffix}.csv")
    labels = pd.read_csv(HERE / f"label5_subject_results{suffix}.csv")

    lp = loso.pivot(index="subject", columns="floor", values="bac")
    cp = cross.pivot(index="subject", columns="condition", values="bac")
    h1_diff = (lp["relative"] - lp["legacy"]).to_numpy()
    h2_diff = (cp["relative_subject"] - cp["none"]).to_numpy()
    h3_diff_pp = (cp["relative_dataset_subject"] - cp["relative_subject"]).to_numpy() * 100
    h4_diff_pp = (lp["relative"] - cp["relative_subject"]).to_numpy() * 100

    h1_applicable = float(floor["below_1e-8"].median()) > 0
    p_raw = {
        "H2": one_sided_wilcoxon(h2_diff),
        "H3": tost(h3_diff_pp)[0],
    }
    if h1_applicable:
        p_raw["H1"] = one_sided_wilcoxon(h1_diff)
    rho, p_h5_two = stats.spearmanr(sep["own_sep_P"], sep["loso_bac_E"])
    p_raw["H5"] = float(p_h5_two / 2 if rho >= 0 else 1 - p_h5_two / 2)
    p_holm = holm_adjust(p_raw)

    h3_p, h3_t90_low, h3_t90_high = tost(h3_diff_pp)
    h4_low, h4_high = bootstrap_ci(h4_diff_pp, 2026092504)
    label_subject = labels.groupby("subject", as_index=False).agg(
        fewshot_bac=("bac", "mean"), baseline_loso_bac=("baseline_loso_bac", "first")
    )
    h6_diff_pp = (label_subject["fewshot_bac"] - label_subject["baseline_loso_bac"]).to_numpy() * 100
    h6_low, h6_high = bootstrap_ci(h6_diff_pp, 2026092506)

    results = [
        {
            "hypothesis": "H1",
            "n": len(h1_diff),
            "estimate": h1_diff.mean() * 100,
            "estimate_unit": "pp relative-minus-legacy",
            "raw_p": p_raw.get("H1", np.nan),
            "holm_p": p_holm.get("H1", np.nan),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "supported": bool(h1_applicable and p_holm.get("H1", 1) < 0.05 and h1_diff.mean() > 0),
            "note": f"median legacy floor count={floor['below_1e-8'].median():.1f}",
        },
        {
            "hypothesis": "H2",
            "n": len(h2_diff),
            "estimate": h2_diff.mean() * 100,
            "estimate_unit": "pp relative-SubjectEA-minus-none",
            "raw_p": p_raw["H2"],
            "holm_p": p_holm["H2"],
            "ci_low": np.nan,
            "ci_high": np.nan,
            "supported": bool(p_holm["H2"] < 0.05 and h2_diff.mean() > 0),
            "note": "one-sided paired Wilcoxon",
        },
        {
            "hypothesis": "H3",
            "n": len(h3_diff_pp),
            "estimate": h3_diff_pp.mean(),
            "estimate_unit": "pp DatasetEA->SubjectEA-minus-SubjectEA",
            "raw_p": h3_p,
            "holm_p": p_holm["H3"],
            "ci_low": h3_t90_low,
            "ci_high": h3_t90_high,
            "supported": bool(p_holm["H3"] < 0.05 and h3_t90_low > -MARGIN_PP and h3_t90_high < MARGIN_PP),
            "note": "paired TOST; 90% t CI; +/-1 pp margin",
        },
        {
            "hypothesis": "H4",
            "n": len(h4_diff_pp),
            "estimate": h4_diff_pp.mean(),
            "estimate_unit": "pp LOSO-minus-cross",
            "raw_p": np.nan,
            "holm_p": np.nan,
            "ci_low": h4_low,
            "ci_high": h4_high,
            "supported": bool(h4_high < 2.0),
            "note": "pre-registered paired-bootstrap 95% CI; Holm not applicable to the fixed CI rule",
        },
        {
            "hypothesis": "H5",
            "n": len(sep),
            "estimate": float(rho),
            "estimate_unit": "Spearman rho",
            "raw_p": p_raw["H5"],
            "holm_p": p_holm["H5"],
            "ci_low": np.nan,
            "ci_high": np.nan,
            "supported": bool(rho > 0 and p_holm["H5"] < 0.05),
            "note": "one-sided Spearman; calibration separability vs evaluation BAC",
        },
        {
            "hypothesis": "H6",
            "n": len(h6_diff_pp),
            "estimate": h6_diff_pp.mean(),
            "estimate_unit": "pp few-shot-minus-LOSO",
            "raw_p": np.nan,
            "holm_p": np.nan,
            "ci_low": h6_low,
            "ci_high": h6_high,
            "supported": bool(h6_high < 1.0),
            "note": "3-draw subject mean; pre-registered paired-bootstrap 95% CI",
        },
    ]
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(HERE / f"confirmatory_hypotheses{suffix}.csv", index=False)

    summaries = []
    for condition, group in loso.groupby("floor"):
        summaries.append({"scope": "LOSO", "condition": condition, "n": len(group), "mean_bac": group.bac.mean(), "sd_bac": group.bac.std(ddof=1)})
    for condition, group in cross.groupby("condition"):
        summaries.append({"scope": "cross", "condition": condition, "n": len(group), "mean_bac": group.bac.mean(), "sd_bac": group.bac.std(ddof=1)})
    pd.DataFrame(summaries).to_csv(HERE / f"performance_summary{suffix}.csv", index=False)

    report = [
        "# Dreyer2023 Confirmatory Report",
        "",
        "The hypotheses, directions, margins and decision rules were fixed before Dreyer2023 was loaded.",
        "H1, H2, H3 and H5 use Holm-adjusted p-values. H4 and H6 retain their explicitly pre-registered 95% bootstrap-bound rules.",
        "",
        "## Performance",
        "",
        "| Scope | Condition | n | BAC mean +/- SD |",
        "|---|---|---:|---:|",
    ]
    for row in summaries:
        report.append(f"| {row['scope']} | {row['condition']} | {row['n']} | {row['mean_bac']*100:.2f} +/- {row['sd_bac']*100:.2f}% |")
    report += [
        "",
        "## Hypotheses",
        "",
        "| H | n | Estimate | CI | Raw p | Holm p | Confirmed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        ci = "-" if pd.isna(row["ci_low"]) else f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        raw_p = "-" if pd.isna(row["raw_p"]) else f"{row['raw_p']:.4g}"
        holm_p = "-" if pd.isna(row["holm_p"]) else f"{row['holm_p']:.4g}"
        report.append(f"| {row['hypothesis']} | {row['n']} | {row['estimate']:.3f} {row['estimate_unit']} | {ci} | {raw_p} | {holm_p} | {'yes' if row['supported'] else 'no'} |")
    report += [
        "",
        "## Interpretation Boundary",
        "",
        "Confirmation on one sealed dataset strengthens external validity but does not establish universality across acquisition systems or preprocessing pipelines.",
        "H5 is predictive association, not a causal mechanism. H6 concerns the specified CSP-LDA natural-weight update and is not evidence that all supervised personalization is ineffective.",
    ]
    (HERE / f"CONFIRMATORY_REPORT{suffix}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(result_frame.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
