"""Frozen 4-dataset LODO amplitude-unit intervention.

This script is intentionally self-contained in analysis_outputs and only imports
the already-audited loaders/alignment/classifier helpers. It does not modify the
shared training code or use the GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
ROOT = ANALYSIS.parent
for path in (
    ANALYSIS / "cmpb_floor_gate_20260924",
    ANALYSIS / "cmpb_budget_20260924",
    ANALYSIS / "cmpb_bdea_20260924",
    ROOT / "MI_test" / "MI_loso_project",
):
    sys.path.insert(0, str(path))

from mrfbcsp_loso import ManualCSP  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from run_bdea_bnci import load_npz  # noqa: E402
from run_budget_causal import norm_stats  # noqa: E402
from run_floor_gate import apply_alignment, fit_csp_lda, metrics, predict  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402


NAMES = ("cho2017", "lee2019", "bnci2014001", "physionetmi")
PATHS = {
    "cho2017": ROOT / "MI_test/preprocessed_sfreq100/cho2017.npz",
    "lee2019": ROOT / "MI_test/preprocessed_sfreq100/lee2019.npz",
    "bnci2014001": ANALYSIS / "cmpb_bdea_20260924/bnci2014001.npz",
    "physionetmi": ANALYSIS / "cmpb_bdea_20260924/physionetmi.npz",
}
FLOORS = ("legacy", "relative")
ALIGNMENTS = ("subject", "dataset_subject")
SCALES = (("1", 1.0), ("1e-3", 1e-3), ("1e-6", 1e-6))
N_TIMES = 200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(smoke: bool):
    loaded = {name: load_npz(PATHS[name]) for name in NAMES}
    shared = [
        channel
        for channel in loaded[NAMES[0]]["ch_names"]
        if all(channel in loaded[name]["ch_names"] for name in NAMES)
    ]
    if len(shared) != 21:
        raise RuntimeError(f"Expected 21 shared channels, found {len(shared)}: {shared}")

    output = {}
    for name, data in loaded.items():
        indices = [data["ch_names"].index(channel) for channel in shared]
        x = data["X"][:, indices, :N_TIMES].astype(np.float32)
        y = data["y"].astype(np.int64)
        subjects = data["subjects"].astype(np.int64)
        if smoke:
            keep_subjects = np.unique(subjects)[:3]
            keep = np.isin(subjects, keep_subjects)
            x, y, subjects = x[keep], y[keep], subjects[keep]
        output[name] = {"X": x, "y": y, "subjects": subjects}
    return output, shared


def unique_source_subjects(parts: list[dict]) -> np.ndarray:
    output = []
    offset = 0
    for part in parts:
        ids = part["subjects"]
        _, encoded = np.unique(ids, return_inverse=True)
        output.append(encoded.astype(np.int64) + offset)
        offset += int(encoded.max()) + 1
    return np.concatenate(output)


def subject_rows(context: dict, y, pred, subjects):
    rows = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        rows.append(
            {
                **context,
                "subject": int(subject),
                "n_trials": int(mask.sum()),
                **metrics(y[mask], pred[mask]),
            }
        )
    return rows


def evaluate_condition(data, scale_label, scale, floor, alignment, out_dir):
    key = f"floor-{floor}_align-{alignment}_scale-{scale_label}"
    result_path = out_dir / "cells" / f"{key}.csv"
    pred_path = out_dir / "predictions" / f"{key}.npz"
    diagnostic_path = out_dir / "diagnostics" / f"{key}.csv"
    if result_path.exists() and pred_path.exists() and diagnostic_path.exists():
        print(f"[skip] {key}", flush=True)
        return

    started = time.perf_counter()
    aligned = {}
    diagnostics = []
    for name, item in data.items():
        scaled = (item["X"] * np.float32(scale)).astype(np.float32)
        aligned[name], diag = apply_alignment(
            scaled, item["subjects"], floor, alignment, name
        )
        for row in diag:
            diagnostics.append(
                {"scale_label": scale_label, "scale": scale, "alignment": alignment, **row}
            )
        del scaled

    covariance = {
        name: Covariances("oas").fit_transform(item.astype(np.float64))
        for name, item in aligned.items()
    }
    rows = []
    predictions = {}
    for held_out in NAMES:
        source_names = [name for name in NAMES if name != held_out]
        source_parts = [data[name] for name in source_names]
        x_source = np.concatenate([aligned[name] for name in source_names])
        y_source = np.concatenate([data[name]["y"] for name in source_names])
        source_subjects = unique_source_subjects(source_parts)
        x_target = aligned[held_out]
        y_target = data[held_out]["y"]
        target_subjects = data[held_out]["subjects"]

        mu, sd = norm_stats(x_source)
        csp, scaler, lda = fit_csp_lda(
            ManualCSP,
            ((x_source - mu) / sd).astype(np.float32),
            y_source,
            8,
        )
        pred_csp = predict(
            csp, scaler, lda, ((x_target - mu) / sd).astype(np.float32)
        ).astype(np.int8)
        context = {
            "scale_label": scale_label,
            "scale": scale,
            "floor": floor,
            "alignment": alignment,
            "classifier": "CSP-LDA",
            "held_out": held_out,
            "sources": "+".join(source_names),
        }
        rows.extend(subject_rows(context, y_target, pred_csp, target_subjects))
        predictions[f"CSP-LDA__{held_out}"] = pred_csp

        c_source = np.concatenate([covariance[name] for name in source_names])
        model = make_pipeline(
            TangentSpace(metric="riemann"),
            LogisticRegression(C=1.0, max_iter=3000),
        ).fit(c_source, y_source)
        pred_ts = model.predict(covariance[held_out]).astype(np.int8)
        context["classifier"] = "TS-LR"
        rows.extend(subject_rows(context, y_target, pred_ts, target_subjects))
        predictions[f"TS-LR__{held_out}"] = pred_ts

        predictions[f"y__{held_out}"] = y_target.astype(np.int8)
        predictions[f"subjects__{held_out}"] = target_subjects.astype(np.int16)
        print(
            f"  [{key}] held={held_out} n={len(y_target)} done",
            flush=True,
        )
        del x_source, y_source, source_subjects, x_target, c_source
        gc.collect()

    atomic_csv(result_path, rows)
    atomic_csv(diagnostic_path, diagnostics)
    atomic_npz(pred_path, predictions)
    elapsed = time.perf_counter() - started
    print(f"[done] {key} {elapsed / 60:.1f} min", flush=True)


def main() -> int:
    args = parse_args()
    out_dir = HERE / ("smoke" if args.smoke else "results")
    if args.force and out_dir.exists():
        raise RuntimeError("--force is intentionally unsupported; remove individual cell files explicitly")
    data, shared = load_inputs(args.smoke)
    manifest = {
        "datasets": list(NAMES),
        "shared_channels": shared,
        "n_times": N_TIMES,
        "smoke": args.smoke,
        "threads": args.threads,
        "input_sha256": {name: file_sha256(PATHS[name]) for name in NAMES},
        "cells": {
            "floors": list(FLOORS),
            "alignments": list(ALIGNMENTS),
            "scales": dict(SCALES),
            "classifiers": ["CSP-LDA", "TS-LR"],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"4-dataset LODO unit intervention: {len(shared)} channels, "
        f"{sum(len(item['y']) for item in data.values())} trials",
        flush=True,
    )
    with threadpool_limits(limits=args.threads):
        for scale_label, scale in SCALES:
            for floor in FLOORS:
                for alignment in ALIGNMENTS:
                    evaluate_condition(
                        data, scale_label, scale, floor, alignment, out_dir
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
