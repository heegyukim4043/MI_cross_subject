"""Run matched no-alignment CSP-LDA baselines for the EA floor gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run_floor_gate import (
    channel_normalize,
    fit_csp_lda,
    load_inputs,
    metrics,
    predict,
    sha256,
    write_rows,
)


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, selected in groups.items():
        output.append(
            {
                **dict(zip(group_keys, key)),
                "n_subjects": len(selected),
                "mean_acc": float(np.mean([row["acc"] for row in selected])),
                "std_acc": float(np.std([row["acc"] for row in selected], ddof=0)),
                "mean_bac": float(np.mean([row["bac"] for row in selected])),
                "std_bac": float(np.std([row["bac"] for row in selected], ddof=0)),
                "mean_kappa": float(np.mean([row["kappa"] for row in selected])),
            }
        )
    return output


def run_cross(loaded, manual_csp, n_csp: int, out: Path) -> None:
    rows = []
    for source, target in (("cho2017", "lee2019"), ("lee2019", "cho2017")):
        print(f"[baseline cross] {source}->{target}", flush=True)
        source_data, target_data = loaded[source], loaded[target]
        x_source, x_target = channel_normalize(source_data["X"], target_data["X"])
        csp, scaler, classifier = fit_csp_lda(manual_csp, x_source, source_data["y"], n_csp)
        for subject in np.unique(target_data["subjects"]):
            mask = target_data["subjects"] == subject
            prediction = predict(csp, scaler, classifier, x_target[mask])
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "alignment": "none",
                    "subject": int(subject),
                    "n_test": int(mask.sum()),
                    **metrics(target_data["y"][mask], prediction),
                }
            )
        del x_source, x_target
    write_rows(out / "baseline_cross_subject_results.csv", rows)
    write_rows(out / "baseline_cross_summary.csv", summarize(rows, ["source", "target", "alignment"]))


def run_loso(loaded, manual_csp, n_csp: int, datasets: list[str], out: Path) -> None:
    rows = []
    for dataset in datasets:
        data = loaded[dataset]
        print(f"[baseline loso] {dataset}", flush=True)
        for subject in np.unique(data["subjects"]):
            started = time.perf_counter()
            test_mask = data["subjects"] == subject
            train_mask = ~test_mask
            x_train, x_test = channel_normalize(data["X"][train_mask], data["X"][test_mask])
            csp, scaler, classifier = fit_csp_lda(manual_csp, x_train, data["y"][train_mask], n_csp)
            prediction = predict(csp, scaler, classifier, x_test)
            rows.append(
                {
                    "dataset": dataset,
                    "alignment": "none",
                    "subject": int(subject),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    **metrics(data["y"][test_mask], prediction),
                    "elapsed_min": (time.perf_counter() - started) / 60,
                }
            )
            write_rows(out / "baseline_loso_subject_results.csv", rows)
    write_rows(out / "baseline_loso_summary.csv", summarize(rows, ["dataset", "alignment"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "full")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n-csp", type=int, default=8)
    parser.add_argument("--loso-datasets", nargs="+", default=("lee2019",))
    args = parser.parse_args()

    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    project_code = root / "MI_test" / "MI_loso_project"
    sys.path.insert(0, str(project_code))
    from mrfbcsp_loso import ManualCSP

    loaded, common_channels, input_paths = load_inputs(root)
    manifest = {
        "root": str(root),
        "common_channels": common_channels,
        "n_csp": args.n_csp,
        "threads": args.threads,
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in input_paths.items()},
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (out / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with threadpool_limits(limits=args.threads):
        run_cross(loaded, ManualCSP, args.n_csp, out)
        run_loso(loaded, ManualCSP, args.n_csp, list(args.loso_datasets), out)
    print(f"[done] outputs={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
