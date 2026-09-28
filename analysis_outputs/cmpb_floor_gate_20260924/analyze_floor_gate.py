"""Analyze the completed EA floor gate with paired subject-level statistics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


RNG_SEED = 2026
BOOTSTRAPS = 10_000


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = (total - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def paired_rank_biserial(diff: np.ndarray) -> float:
    nonzero = diff[np.abs(diff) > 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_summary(diff: np.ndarray, rng: np.random.Generator) -> dict:
    diff = np.asarray(diff, dtype=float)
    samples = rng.choice(diff, size=(BOOTSTRAPS, len(diff)), replace=True).mean(axis=1)
    if np.allclose(diff, 0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "n": len(diff),
        "mean_diff_pp": float(diff.mean()),
        "sd_diff_pp": float(diff.std(ddof=1)),
        "median_diff_pp": float(np.median(diff)),
        "ci95_low_pp": float(np.quantile(samples, 0.025)),
        "ci95_high_pp": float(np.quantile(samples, 0.975)),
        "wilcoxon_W": statistic,
        "p_raw": p_value,
        "rank_biserial": paired_rank_biserial(diff),
        "n_positive": int((diff > 0).sum()),
        "n_negative": int((diff < 0).sum()),
        "n_zero": int((diff == 0).sum()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cross_statistics(full: Path) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    aligned = pd.read_csv(full / "cross_subject_results.csv")
    baseline = pd.read_csv(full / "baseline_cross_subject_results.csv")
    aligned["method"] = aligned["floor_mode"] + ":" + aligned["alignment"]
    baseline["method"] = "none:none"
    combined = pd.concat(
        [
            aligned[["source", "target", "subject", "method", "bac"]],
            baseline[["source", "target", "subject", "method", "bac"]],
        ],
        ignore_index=True,
    )
    pivot = combined.pivot(index=["source", "target", "subject"], columns="method", values="bac") * 100
    comparisons = [
        ("floor correction on SubjectEA", "relative:subject", "legacy:subject"),
        ("legacy hierarchy", "legacy:dataset_subject", "legacy:subject"),
        ("corrected hierarchy residual", "relative:dataset_subject", "relative:subject"),
        ("legacy scalar rescue", "legacy:scale_dataset_subject", "legacy:subject"),
        ("corrected SubjectEA vs no alignment", "relative:subject", "none:none"),
        ("corrected S->D invariance", "relative:subject_dataset", "relative:subject"),
    ]
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for (source, target), frame in pivot.groupby(level=["source", "target"]):
        for label, method_a, method_b in comparisons:
            diff = (frame[method_a] - frame[method_b]).dropna().to_numpy()
            rows.append(
                {
                    "scope": "cross",
                    "source": source,
                    "target": target,
                    "comparison": label,
                    "method_a": method_a,
                    "method_b": method_b,
                    **paired_summary(diff, rng),
                }
            )
    adjusted = holm_adjust([row["p_raw"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_holm_cross_family"] = p_adjusted
    return rows, aligned, baseline


def loso_statistics(full: Path) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    aligned = pd.read_csv(full / "loso_subject_results.csv")
    baseline = pd.read_csv(full / "baseline_loso_subject_results.csv")
    aligned["method"] = aligned["floor_mode"] + ":subject"
    baseline["method"] = "none:none"
    combined = pd.concat(
        [aligned[["dataset", "subject", "method", "bac"]], baseline[["dataset", "subject", "method", "bac"]]],
        ignore_index=True,
    )
    pivot = combined.pivot(index=["dataset", "subject"], columns="method", values="bac") * 100
    comparisons = [
        ("legacy SubjectEA vs no alignment", "legacy:subject", "none:none"),
        ("corrected SubjectEA vs no alignment", "relative:subject", "none:none"),
        ("floor correction on SubjectEA", "relative:subject", "legacy:subject"),
    ]
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    for dataset, frame in pivot.groupby(level="dataset"):
        for label, method_a, method_b in comparisons:
            diff = (frame[method_a] - frame[method_b]).dropna().to_numpy()
            rows.append(
                {
                    "scope": "loso",
                    "dataset": dataset,
                    "comparison": label,
                    "method_a": method_a,
                    "method_b": method_b,
                    **paired_summary(diff, rng),
                }
            )
    adjusted = holm_adjust([row["p_raw"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_holm_loso_family"] = p_adjusted
    return rows, aligned, baseline


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def p_text(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def report(full: Path, out: Path, cross_rows: list[dict], loso_rows: list[dict], aligned_cross, baseline_cross, aligned_loso, baseline_loso):
    cross_summary = aligned_cross.groupby(["source", "target", "floor_mode", "alignment"])["bac"].agg(["mean", "std"])
    cross_base = baseline_cross.groupby(["source", "target"])["bac"].agg(["mean", "std"])
    loso_summary = aligned_loso.groupby(["dataset", "floor_mode"])["bac"].agg(["mean", "std"])
    loso_base = baseline_loso.groupby("dataset")["bac"].agg(["mean", "std"])
    diagnostics = pd.read_csv(full / "numerical_diagnostics.csv")
    legacy_subject = diagnostics[
        (diagnostics["floor_mode"] == "legacy") & (diagnostics["stage"] == "subject")
    ].drop_duplicates(["dataset", "subject"])
    cho_floor = legacy_subject[legacy_subject["dataset"] == "cho2017"]["floored_or_dropped"]
    lee_floor = legacy_subject[legacy_subject["dataset"] == "lee2019"]["floored_or_dropped"]
    scalar = diagnostics[diagnostics["stage"] == "scalar_dataset"].drop_duplicates("dataset").set_index("dataset")
    scale_ratio = float(scalar.loc["cho2017", "scalar_scale"] / scalar.loc["lee2019", "scalar_scale"])
    corrected = diagnostics[
        diagnostics["floor_mode"].isin(["relative", "rank_aware"])
        & diagnostics["stage"].str.contains("subject", na=False)
    ]

    lines = [
        "# CMPB EA Floor Gate Results",
        "",
        "## Status and interpretation boundary",
        "",
        "This is a retrospective numerical-diagnostic gate, not a preregistered confirmatory experiment. "
        "All accuracy comparisons are paired by target subject. Confidence intervals are 10,000-resample "
        "paired bootstrap intervals for the mean BAC difference. Wilcoxon tests are two-sided and Holm-adjusted "
        "within the 12 cross-dataset comparisons and the 3 LOSO comparisons. Rank-biserial correlation is paired.",
        "",
        "## Cross-dataset BAC",
        "",
        "| Direction | No align | Legacy S | Legacy D->S | Scalar D->S | Corrected S | Corrected D->S | Corrected S->D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, target in (("cho2017", "lee2019"), ("lee2019", "cho2017")):
        idx = (source, target)
        values = [
            cross_base.loc[idx, "mean"],
            cross_summary.loc[(*idx, "legacy", "subject"), "mean"],
            cross_summary.loc[(*idx, "legacy", "dataset_subject"), "mean"],
            cross_summary.loc[(*idx, "legacy", "scale_dataset_subject"), "mean"],
            cross_summary.loc[(*idx, "relative", "subject"), "mean"],
            cross_summary.loc[(*idx, "relative", "dataset_subject"), "mean"],
            cross_summary.loc[(*idx, "relative", "subject_dataset"), "mean"],
        ]
        lines.append(f"| {source}->{target} | " + " | ".join(pct(value) for value in values) + " |")

    lines += [
              "", "## Numerical mechanism", "",
              f"- Under legacy SubjectEA, Cho floored {int(cho_floor.min())}-{int(cho_floor.max())} of 48 eigenvalues "
              f"per subject (median {cho_floor.median():.0f}); Lee floored {int(lee_floor.min())}-{int(lee_floor.max())} "
              f"(median {lee_floor.median():.0f}).",
              f"- Dataset scalar covariance scales were {scalar.loc['cho2017', 'scalar_scale']:.6g} for Cho and "
              f"{scalar.loc['lee2019', 'scalar_scale']:.6g} for Lee, a {scale_ratio:.1f}-fold ratio.",
              "- After full DatasetEA or scalar dataset normalization, the subsequent legacy SubjectEA floored zero "
              "eigenvalues in both datasets.",
              f"- Relative and rank-aware policies floored/dropped at most "
              f"{int(corrected['floored_or_dropped'].max())} eigenvalues and retained all 48 directions for every subject.",
              "", "## Paired cross-dataset contrasts", "",
              "| Direction | Contrast | Delta BAC | 95% CI | Holm p | rank-biserial | +/-/0 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in cross_rows:
        lines.append(
            f"| {row['source']}->{row['target']} | {row['comparison']} | {row['mean_diff_pp']:+.2f}%p | "
            f"[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] | "
            f"{p_text(row['p_holm_cross_family'])} | {row['rank_biserial']:+.3f} | "
            f"{row['n_positive']}/{row['n_negative']}/{row['n_zero']} |"
        )

    lines += [
        "",
        "## Lee2019 LOSO BAC",
        "",
        f"- No alignment: {pct(loso_base.loc['lee2019', 'mean'])} +/- {pct(loso_base.loc['lee2019', 'std'])}.",
        f"- Legacy SubjectEA: {pct(loso_summary.loc[('lee2019', 'legacy'), 'mean'])} +/- "
        f"{pct(loso_summary.loc[('lee2019', 'legacy'), 'std'])}.",
        f"- Corrected relative SubjectEA: {pct(loso_summary.loc[('lee2019', 'relative'), 'mean'])} +/- "
        f"{pct(loso_summary.loc[('lee2019', 'relative'), 'std'])}.",
        "",
        "| Contrast | Delta BAC | 95% CI | Holm p | rank-biserial | +/-/0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in loso_rows:
        lines.append(
            f"| {row['comparison']} | {row['mean_diff_pp']:+.2f}%p | "
            f"[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] | "
            f"{p_text(row['p_holm_loso_family'])} | {row['rank_biserial']:+.3f} | "
            f"{row['n_positive']}/{row['n_negative']}/{row['n_zero']} |"
        )

    def cross_row(direction, comparison):
        source, target = direction
        return next(row for row in cross_rows if row["source"] == source and row["target"] == target
                    and row["comparison"] == comparison)

    old_1 = cross_row(("cho2017", "lee2019"), "legacy hierarchy")["mean_diff_pp"]
    new_1 = cross_row(("cho2017", "lee2019"), "corrected hierarchy residual")["mean_diff_pp"]
    old_2 = cross_row(("lee2019", "cho2017"), "legacy hierarchy")["mean_diff_pp"]
    new_2 = cross_row(("lee2019", "cho2017"), "corrected hierarchy residual")["mean_diff_pp"]
    lines += [
        "",
        "## Gate decision",
        "",
        f"- The apparent D->S gain over SubjectEA shrank from {old_1:.2f} to {new_1:.2f}%p for Cho->Lee "
        f"and from {old_2:.2f} to {new_2:.2f}%p for Lee->Cho after replacing the absolute floor.",
        "- Legacy scalar dataset normalization reproduced corrected SubjectEA almost exactly. This is the "
        "predicted signature of a scale-dependent absolute-floor artifact.",
        "- Corrected S->D and corrected SubjectEA produced identical subject-level BACs in both directions, "
        "consistent with the expected global-rotation invariance of CSP-LDA.",
        "- Corrected D->S retained a small residual gain. It should be framed as a secondary empirical effect, "
        "not evidence that the original large hierarchy gain survived.",
        "- Gate outcome: the current manuscript's central hierarchical-alignment magnitude is not robust. "
        "Proceed with the numerical-methods/reproducibility reframing unless matched CSPNet pilots show a "
        "substantial corrected D->S benefit.",
        "",
        "## Limitations",
        "",
        "- Current sfreq100 files have no retained session/run/trial IDs and no demonstrated ICA application.",
        "- EA uses all unlabeled target trials and is transductive; this is not a causal online protocol.",
        "- The gate uses one deterministic CSP-LDA implementation. Neural-model conclusions require the planned pilot.",
        "- P-values are exploratory because the floor issue and these contrasts were identified after inspecting prior results.",
        "",
        "## Procedural reference",
        "",
        "The analysis/reporting workflow used the current Scientific Agent Skills statistical-analysis guidance: "
        "Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills: A Library of Procedural "
        "Knowledge for Research Agents. arXiv:2609.00065 (2026). https://doi.org/10.48550/arXiv.2609.00065",
    ]
    (out / "FLOOR_GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, default=Path(__file__).resolve().parent / "full")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    full, out = args.full.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cross_rows, aligned_cross, baseline_cross = cross_statistics(full)
    loso_rows, aligned_loso, baseline_loso = loso_statistics(full)
    write_csv(out / "cross_paired_statistics.csv", cross_rows)
    write_csv(out / "loso_paired_statistics.csv", loso_rows)
    report(full, out, cross_rows, loso_rows, aligned_cross, baseline_cross, aligned_loso, baseline_loso)
    print(out / "FLOOR_GATE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
