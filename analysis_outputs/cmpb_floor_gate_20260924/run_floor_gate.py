"""Run the CPU-only EA floor gate on the current sfreq100 MI inputs.

This script is deliberately isolated from the research training code. It uses
the same ManualCSP, feature scaling, LDA, channel normalization, and subject-
level macro averaging as the existing CSP-LDA pipeline while making the EA
precision/floor policy explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


FLOOR_MODES = ("legacy", "absolute64", "relative", "rank_aware")
ALIGNMENTS = ("subject", "dataset_subject", "subject_dataset", "scale_dataset_subject")
EPS = 1e-8
RANK_RTOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n-csp", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Use two target subjects per direction.")
    parser.add_argument("--skip-cross", action="store_true")
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument(
        "--loso-datasets",
        nargs="+",
        choices=("cho2017", "lee2019"),
        default=("lee2019",),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_covariance(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    xx = x.astype(dtype, copy=False)
    return np.mean(np.einsum("nct,ndt->ncd", xx, xx), axis=0, dtype=dtype)


def whitener_from_cov(covariance: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    covariance = (covariance + covariance.T) * 0.5
    values, vectors = np.linalg.eigh(covariance)
    if mode in ("legacy", "absolute64"):
        floor = EPS
        kept = np.ones_like(values, dtype=bool)
        adjusted = np.maximum(values, floor)
        inverse = adjusted ** -0.5
    elif mode == "relative":
        floor = EPS * float(np.trace(covariance)) / len(covariance)
        kept = np.ones_like(values, dtype=bool)
        adjusted = np.maximum(values, floor)
        inverse = adjusted ** -0.5
    elif mode == "rank_aware":
        floor = RANK_RTOL * float(max(values[-1], 0.0))
        kept = values > floor
        adjusted = values.copy()
        inverse = np.zeros_like(values)
        inverse[kept] = values[kept] ** -0.5
    else:
        raise ValueError(f"Unknown floor mode: {mode}")

    whitener = (vectors * inverse) @ vectors.T
    diagnostic = {
        "floor": float(floor),
        "eigen_min": float(values[0]),
        "eigen_max": float(values[-1]),
        "floored_or_dropped": int((values < floor).sum()),
        "rank_retained": int(kept.sum()),
        "condition_raw": float(values[-1] / max(values[0], np.finfo(float).tiny)),
    }
    return whitener, diagnostic


def align_batch(x: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    dtype = np.float32 if mode == "legacy" else np.float64
    covariance = mean_covariance(x, dtype=dtype)
    whitener, diagnostic = whitener_from_cov(covariance, mode)
    aligned = np.einsum("cd,ndt->nct", whitener, x, optimize=True)
    return aligned.astype(np.float32), diagnostic


def align_subjects(
    x: np.ndarray,
    subjects: np.ndarray,
    mode: str,
    dataset: str,
    stage: str,
) -> tuple[np.ndarray, list[dict]]:
    output = np.empty_like(x, dtype=np.float32)
    diagnostics: list[dict] = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        output[mask], diagnostic = align_batch(x[mask], mode)
        diagnostics.append(
            {
                "dataset": dataset,
                "stage": stage,
                "floor_mode": mode,
                "subject": int(subject),
                **diagnostic,
            }
        )
    return output, diagnostics


def scalar_dataset_normalize(x: np.ndarray) -> tuple[np.ndarray, float]:
    covariance = mean_covariance(x, dtype=np.float64)
    scale = float(np.sqrt(np.trace(covariance) / len(covariance)))
    return (x / scale).astype(np.float32), scale


def apply_alignment(
    x: np.ndarray,
    subjects: np.ndarray,
    mode: str,
    alignment: str,
    dataset: str,
) -> tuple[np.ndarray, list[dict]]:
    diagnostics: list[dict] = []
    if alignment == "subject":
        return align_subjects(x, subjects, mode, dataset, "subject")
    if alignment == "dataset_subject":
        x, diagnostic = align_batch(x, mode)
        diagnostics.append(
            {"dataset": dataset, "stage": "dataset", "floor_mode": mode, "subject": "all", **diagnostic}
        )
        x, subject_diagnostics = align_subjects(x, subjects, mode, dataset, "subject_after_dataset")
        return x, diagnostics + subject_diagnostics
    if alignment == "subject_dataset":
        x, subject_diagnostics = align_subjects(x, subjects, mode, dataset, "subject_before_dataset")
        x, diagnostic = align_batch(x, mode)
        diagnostics = subject_diagnostics + [
            {"dataset": dataset, "stage": "dataset_after_subject", "floor_mode": mode, "subject": "all", **diagnostic}
        ]
        return x, diagnostics
    if alignment == "scale_dataset_subject":
        x, scale = scalar_dataset_normalize(x)
        diagnostics.append(
            {
                "dataset": dataset,
                "stage": "scalar_dataset",
                "floor_mode": mode,
                "subject": "all",
                "scalar_scale": scale,
            }
        )
        x, subject_diagnostics = align_subjects(x, subjects, mode, dataset, "subject_after_scalar_dataset")
        return x, diagnostics + subject_diagnostics
    raise ValueError(f"Unknown alignment: {alignment}")


def channel_normalize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def fit_csp_lda(manual_csp, x_train: np.ndarray, y_train: np.ndarray, n_csp: int):
    csp = manual_csp(n_components=n_csp)
    csp.fit(x_train, y_train)
    features = csp.transform(x_train)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(features, y_train)
    return csp, scaler, classifier


def predict(csp, scaler, classifier, x: np.ndarray) -> np.ndarray:
    return classifier.predict(scaler.transform(csp.transform(x)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc": float(np.mean(y_true == y_pred)),
        "bac": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(root: Path):
    data_dir = root / "MI_test" / "preprocessed_sfreq100"
    paths = {name: data_dir / f"{name}.npz" for name in ("cho2017", "lee2019")}
    loaded = {}
    names_by_dataset = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=True) as data:
            loaded[name] = {
                "X": data["X"].astype(np.float32),
                "y": data["y"].astype(np.int64),
                "subjects": data["subjects"].astype(np.int64),
            }
            names_by_dataset[name] = [str(channel) for channel in data["ch_names"]]
    common = [name for name in names_by_dataset["cho2017"] if name in names_by_dataset["lee2019"]]
    if len(common) != 48:
        raise RuntimeError(f"Expected common48, found {len(common)} channels: {common}")
    for name in loaded:
        indices = [names_by_dataset[name].index(channel) for channel in common]
        loaded[name]["X"] = loaded[name]["X"][:, indices, :201]
    return loaded, common, paths


def subset_subjects(data: dict, n_subjects: int) -> dict:
    selected = np.unique(data["subjects"])[:n_subjects]
    mask = np.isin(data["subjects"], selected)
    return {key: value[mask] for key, value in data.items()}


def run_cross(loaded, manual_csp, n_csp: int, smoke: bool, out: Path):
    rows: list[dict] = []
    diagnostics: list[dict] = []
    summaries: list[dict] = []
    for source, target in (("cho2017", "lee2019"), ("lee2019", "cho2017")):
        source_data, target_data = loaded[source], loaded[target]
        if smoke:
            source_data = subset_subjects(source_data, 3)
            target_data = subset_subjects(target_data, 3)
        target_ids = np.unique(target_data["subjects"])
        if smoke:
            target_ids = target_ids[:2]
        for mode in FLOOR_MODES:
            for alignment in ALIGNMENTS:
                started = time.perf_counter()
                print(f"[cross] {source}->{target} floor={mode} alignment={alignment}", flush=True)
                x_source, diagnostic_source = apply_alignment(
                    source_data["X"], source_data["subjects"], mode, alignment, source
                )
                x_target, diagnostic_target = apply_alignment(
                    target_data["X"], target_data["subjects"], mode, alignment, target
                )
                diagnostics.extend(
                    [{"scope": "cross", "direction": f"{source}_to_{target}", "alignment": alignment, **d}
                     for d in diagnostic_source + diagnostic_target]
                )
                x_source, x_target = channel_normalize(x_source, x_target)
                csp, scaler, classifier = fit_csp_lda(manual_csp, x_source, source_data["y"], n_csp)
                for subject in target_ids:
                    mask = target_data["subjects"] == subject
                    prediction = predict(csp, scaler, classifier, x_target[mask])
                    row = {
                        "source": source,
                        "target": target,
                        "floor_mode": mode,
                        "alignment": alignment,
                        "subject": int(subject),
                        "n_test": int(mask.sum()),
                        **metrics(target_data["y"][mask], prediction),
                    }
                    rows.append(row)
                selected = [r for r in rows if r["source"] == source and r["target"] == target
                            and r["floor_mode"] == mode and r["alignment"] == alignment]
                summaries.append(
                    {
                        "source": source,
                        "target": target,
                        "floor_mode": mode,
                        "alignment": alignment,
                        "n_subjects": len(selected),
                        "mean_acc": float(np.mean([r["acc"] for r in selected])),
                        "std_acc": float(np.std([r["acc"] for r in selected], ddof=0)),
                        "mean_bac": float(np.mean([r["bac"] for r in selected])),
                        "std_bac": float(np.std([r["bac"] for r in selected], ddof=0)),
                        "mean_kappa": float(np.mean([r["kappa"] for r in selected])),
                        "elapsed_min": (time.perf_counter() - started) / 60,
                    }
                )
                write_rows(out / "cross_subject_results.csv", rows)
                write_rows(out / "cross_summary.csv", summaries)
                write_rows(out / "numerical_diagnostics.csv", diagnostics)
                print(
                    f"  BAC={summaries[-1]['mean_bac']*100:.2f}% "
                    f"elapsed={summaries[-1]['elapsed_min']:.2f} min",
                    flush=True,
                )
                del x_source, x_target
    return rows, summaries, diagnostics


def run_loso(loaded, manual_csp, n_csp: int, smoke: bool, datasets: list[str], out: Path):
    rows: list[dict] = []
    summaries: list[dict] = []
    for dataset in datasets:
        data = loaded[dataset]
        if smoke:
            data = subset_subjects(data, 3)
        subject_ids = np.unique(data["subjects"])
        if smoke:
            subject_ids = subject_ids[:2]
        for mode in FLOOR_MODES:
            print(f"[loso] dataset={dataset} floor={mode} alignment=subject", flush=True)
            started = time.perf_counter()
            aligned, _ = align_subjects(data["X"], data["subjects"], mode, dataset, "subject")
            for subject in subject_ids:
                fold_started = time.perf_counter()
                test_mask = data["subjects"] == subject
                train_mask = ~test_mask
                x_train, x_test = channel_normalize(aligned[train_mask], aligned[test_mask])
                csp, scaler, classifier = fit_csp_lda(manual_csp, x_train, data["y"][train_mask], n_csp)
                prediction = predict(csp, scaler, classifier, x_test)
                rows.append(
                    {
                        "dataset": dataset,
                        "floor_mode": mode,
                        "alignment": "subject",
                        "subject": int(subject),
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        **metrics(data["y"][test_mask], prediction),
                        "elapsed_min": (time.perf_counter() - fold_started) / 60,
                    }
                )
                write_rows(out / "loso_subject_results.csv", rows)
            selected = [r for r in rows if r["dataset"] == dataset and r["floor_mode"] == mode]
            summaries.append(
                {
                    "dataset": dataset,
                    "floor_mode": mode,
                    "alignment": "subject",
                    "n_subjects": len(selected),
                    "mean_acc": float(np.mean([r["acc"] for r in selected])),
                    "std_acc": float(np.std([r["acc"] for r in selected], ddof=0)),
                    "mean_bac": float(np.mean([r["bac"] for r in selected])),
                    "std_bac": float(np.std([r["bac"] for r in selected], ddof=0)),
                    "mean_kappa": float(np.mean([r["kappa"] for r in selected])),
                    "elapsed_min": (time.perf_counter() - started) / 60,
                }
            )
            write_rows(out / "loso_summary.csv", summaries)
            print(
                f"  BAC={summaries[-1]['mean_bac']*100:.2f}% "
                f"elapsed={summaries[-1]['elapsed_min']:.2f} min",
                flush=True,
            )
            del aligned
    return rows, summaries


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    project_code = root / "MI_test" / "MI_loso_project"
    sys.path.insert(0, str(project_code))
    from mrfbcsp_loso import ManualCSP

    loaded, common_channels, input_paths = load_inputs(root)
    manifest = {
        "root": str(root),
        "output": str(out),
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "threads": args.threads,
        "n_csp": args.n_csp,
        "smoke": args.smoke,
        "common_channels": common_channels,
        "floor_definitions": {
            "legacy": "float32 covariance/eigendecomposition; absolute eigenvalue floor 1e-8",
            "absolute64": "float64 covariance/eigendecomposition; absolute eigenvalue floor 1e-8",
            "relative": "float64; floor = 1e-8 * trace(R) / C",
            "rank_aware": "float64 truncated inverse; retain eigenvalues > 1e-12 * lambda_max",
        },
        "alignment_definitions": {
            "subject": "SubjectEA only",
            "dataset_subject": "full DatasetEA followed by SubjectEA",
            "subject_dataset": "SubjectEA followed by full DatasetEA",
            "scale_dataset_subject": "scalar dataset RMS normalization followed by SubjectEA",
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path), "shape": list(loaded[name]["X"].shape)}
            for name, path in input_paths.items()
        },
        "source_hashes": {
            "mrfbcsp_loso.py": sha256(project_code / "mrfbcsp_loso.py"),
            "eeg_ea.py": sha256(project_code / "eeg_ea.py"),
            "run_floor_gate.py": sha256(Path(__file__).resolve()),
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with threadpool_limits(limits=args.threads):
        if not args.skip_cross:
            run_cross(loaded, ManualCSP, args.n_csp, args.smoke, out)
        if not args.skip_loso:
            run_loso(loaded, ManualCSP, args.n_csp, args.smoke, list(args.loso_datasets), out)
    print(f"[done] outputs={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
