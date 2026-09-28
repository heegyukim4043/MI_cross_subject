"""Summarize the frozen amplitude-unit intervention."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SCALE_ORDER = ["1", "1e-3", "1e-6"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def bootstrap_ci(values, seed=20260925, n_boot=10000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        stop = min(start + 1000, n_boot)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def load_prediction(path, key):
    with np.load(path) as data:
        return data[key].copy()


def markdown_table(frame):
    display = frame.reset_index()
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    source = HERE / ("smoke" if args.smoke else "results")
    destination = HERE / ("smoke_analysis" if args.smoke else "analysis")
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted((source / "cells").glob("*.csv"))
    if len(files) != 12:
        raise RuntimeError(f"Expected 12 completed cells, found {len(files)}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame["scale_label"] = frame["scale"].map(
        {1.0: "1", 1e-3: "1e-3", 1e-6: "1e-6"}
    )
    if frame["scale_label"].isna().any():
        raise RuntimeError("Unrecognized amplitude scale in result cells")
    frame.to_csv(destination / "unit_intervention_subject_results.csv", index=False, encoding="utf-8-sig")

    summary = (
        frame.groupby(["classifier", "floor", "alignment", "scale_label", "held_out"], as_index=False)
        .agg(mean_bac=("bac", "mean"), sd_bac=("bac", "std"), n_subjects=("subject", "size"))
    )
    summary.to_csv(destination / "unit_intervention_means.csv", index=False, encoding="utf-8-sig")

    ranking_rows = []
    for keys, group in frame.groupby(["classifier", "floor", "scale_label", "held_out"]):
        pivot = group.pivot(index="subject", columns="alignment", values="bac").dropna()
        delta = (pivot["dataset_subject"] - pivot["subject"]) * 100
        lo, hi = bootstrap_ci(delta, seed=20260925 + len(ranking_rows))
        ranking_rows.append(
            {
                "classifier": keys[0],
                "floor": keys[1],
                "scale_label": keys[2],
                "held_out": keys[3],
                "n_subjects": len(delta),
                "dataset_subject_minus_subject_pp": delta.mean(),
                "ci95_low_pp": lo,
                "ci95_high_pp": hi,
                "winner": "dataset_subject" if delta.mean() > 0 else "subject" if delta.mean() < 0 else "tie",
            }
        )
    ranking = pd.DataFrame(ranking_rows)
    ranking.to_csv(destination / "alignment_ranking_by_scale.csv", index=False, encoding="utf-8-sig")

    disagreement_rows = []
    for floor in ("legacy", "relative"):
        for alignment in ("subject", "dataset_subject"):
            base_path = source / "predictions" / f"floor-{floor}_align-{alignment}_scale-1.npz"
            for classifier in ("CSP-LDA", "TS-LR"):
                for held_out in ("cho2017", "lee2019", "bnci2014001", "physionetmi"):
                    key = f"{classifier}__{held_out}"
                    base = load_prediction(base_path, key)
                    for scale_label in ("1e-3", "1e-6"):
                        path = source / "predictions" / f"floor-{floor}_align-{alignment}_scale-{scale_label}.npz"
                        pred = load_prediction(path, key)
                        disagreement_rows.append(
                            {
                                "classifier": classifier,
                                "floor": floor,
                                "alignment": alignment,
                                "held_out": held_out,
                                "scale_label": scale_label,
                                "n_trials": len(base),
                                "n_disagree": int(np.sum(base != pred)),
                                "disagreement_rate": float(np.mean(base != pred)),
                            }
                        )
    disagreement = pd.DataFrame(disagreement_rows)
    disagreement.to_csv(destination / "prediction_disagreement.csv", index=False, encoding="utf-8-sig")

    base_winners = ranking[ranking.scale_label == "1"][["classifier", "floor", "held_out", "winner"]].rename(columns={"winner": "winner_scale_1"})
    flips = ranking.merge(base_winners, on=["classifier", "floor", "held_out"])
    flips["ranking_changed"] = flips.winner != flips.winner_scale_1
    flips.to_csv(destination / "ranking_changes.csv", index=False, encoding="utf-8-sig")

    rel = disagreement[disagreement.floor == "relative"]
    legacy = disagreement[disagreement.floor == "legacy"]
    diagnostic_files = sorted((source / "diagnostics").glob("*.csv"))
    diagnostics = pd.concat([pd.read_csv(path) for path in diagnostic_files], ignore_index=True)
    diagnostics["scale_label"] = diagnostics["scale"].map(
        {1.0: "1", 1e-3: "1e-3", 1e-6: "1e-6"}
    )
    initial = diagnostics[
        (diagnostics.floor_mode == "legacy")
        & (diagnostics.scale_label == "1e-6")
        & diagnostics.stage.isin(["subject", "dataset"])
    ]
    fully_floored = int((initial.floored_or_dropped == 21).sum())
    criterion_pass = (
        int((rel.n_disagree > 0).sum()) == 0
        and int((legacy.n_disagree > 0).sum()) > 0
        and int(flips[(flips.floor == "legacy") & (flips.scale_label != "1")].ranking_changed.sum()) > 0
    )
    report = [
        "# Amplitude-unit intervention results",
        "",
        "Smoke output." if args.smoke else "Full pre-specified 4-dataset LODO analysis.",
        "",
        "## Decision endpoints",
        "",
        f"- Relative-floor maximum prediction disagreement: `{rel.disagreement_rate.max():.8f}`.",
        f"- Legacy-floor maximum prediction disagreement: `{legacy.disagreement_rate.max():.8f}`.",
        f"- Relative cells with any prediction change: `{int((rel.n_disagree > 0).sum())}/{len(rel)}`.",
        f"- Legacy cells with any prediction change: `{int((legacy.n_disagree > 0).sum())}/{len(legacy)}`.",
        f"- Relative alignment-ranking changes: `{int(flips[(flips.floor == 'relative') & (flips.scale_label != '1')].ranking_changed.sum())}`.",
        f"- Legacy alignment-ranking changes: `{int(flips[(flips.floor == 'legacy') & (flips.scale_label != '1')].ranking_changed.sum())}`.",
        f"- Legacy mean prediction disagreement at `1e-3`: `{legacy[legacy.scale_label == '1e-3'].disagreement_rate.mean():.4f}`.",
        f"- Legacy mean prediction disagreement at `1e-6`: `{legacy[legacy.scale_label == '1e-6'].disagreement_rate.mean():.4f}`.",
        f"- Fully floored initial legacy references at `1e-6`: `{fully_floored}/{len(initial)}` (21/21 eigenvalues).",
        f"- **Go/no-go criterion 2: {'PASS' if criterion_pass else 'FAIL'}.**",
        "",
        "## Mean BAC (%)",
        "",
        markdown_table(summary.assign(mean_bac=summary.mean_bac * 100).pivot_table(
            index=["classifier", "floor", "alignment", "held_out"],
            columns="scale_label", values="mean_bac"
        ).reindex(columns=SCALE_ORDER).round(2)),
        "",
        "## DatasetEA-to-SubjectEA minus SubjectEA (percentage points)",
        "",
        markdown_table(ranking.pivot_table(
            index=["classifier", "floor", "held_out"],
            columns="scale_label", values="dataset_subject_minus_subject_pp"
        ).reindex(columns=SCALE_ORDER).round(3)),
        "",
        "## Interpretation guardrail",
        "",
        "Amplitude recoding is a numerical intervention, not an estimated component of EEG dataset shift. Any dependence on this intervention is attributed to the method-unit interaction.",
    ]
    (destination / "UNIT_INTERVENTION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
