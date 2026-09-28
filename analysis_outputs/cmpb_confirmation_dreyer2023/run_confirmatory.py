"""Run the pre-registered Dreyer2023 confirmation with CSP-LDA.

This script is isolated from the shared training code. It reuses the audited
EA and CSP-LDA primitives, writes each completed unit immediately, and can be
restarted without recomputing completed subject/condition cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from pyriemann.utils.distance import distance_riemann
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = ROOT / "analysis_outputs"
for path in (
    ANALYSIS / "cmpb_floor_gate_20260924",
    ANALYSIS / "cmpb_budget_20260924",
    ANALYSIS / "cmpb_bdea_20260924",
    ROOT / "MI_test" / "MI_loso_project",
):
    sys.path.insert(0, str(path))

from run_floor_gate import (  # noqa: E402
    apply_alignment,
    fit_csp_lda,
    mean_covariance,
    metrics,
    predict,
)
from run_budget_causal import norm_stats  # noqa: E402
from run_bdea_bnci import load_npz  # noqa: E402
from mrfbcsp_loso import ManualCSP  # noqa: E402


DEV_PATHS = {
    "cho2017": ROOT / "MI_test" / "preprocessed_sfreq100" / "cho2017.npz",
    "lee2019": ROOT / "MI_test" / "preprocessed_sfreq100" / "lee2019.npz",
    "bnci2014001": ANALYSIS / "cmpb_bdea_20260924" / "bnci2014001.npz",
    "physionetmi": ANALYSIS / "cmpb_bdea_20260924" / "physionetmi.npz",
}
DREYER_PATH = HERE / "dreyer2023.npz"
DEV_NAMES = tuple(DEV_PATHS)
N_TIMES = 200
N_CSP = 8
N_DRAWS = 3
LEGACY_FLOOR = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Use four Dreyer and three source subjects per dataset.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subset_subjects(data: dict, n_subjects: int) -> dict:
    keep_ids = np.unique(data["subjects"])[:n_subjects]
    keep = np.isin(data["subjects"], keep_ids)
    return {key: value[keep] if key in ("X", "y", "subjects") else value for key, value in data.items()}


def load_all(smoke: bool) -> tuple[dict, list[str]]:
    raw = {name: load_npz(path) for name, path in DEV_PATHS.items()}
    with np.load(DREYER_PATH, allow_pickle=True) as data:
        raw["dreyer2023"] = {
            "X": data["X"].astype(np.float32),
            "y": data["y"].astype(np.int64),
            "subjects": data["subjects"].astype(np.int64),
            "ch_names": [str(channel) for channel in data["ch_names"]],
        }
    names = DEV_NAMES + ("dreyer2023",)
    shared = [channel for channel in raw[names[0]]["ch_names"] if all(channel in raw[name]["ch_names"] for name in names)]
    if len(shared) != 18:
        raise RuntimeError(f"Pre-registered shared-channel count is 18, found {len(shared)}: {shared}")
    prepared = {}
    for name, data in raw.items():
        indices = [data["ch_names"].index(channel) for channel in shared]
        prepared[name] = {
            "X": data["X"][:, indices, :N_TIMES].astype(np.float32),
            "y": data["y"].astype(np.int64),
            "subjects": data["subjects"].astype(np.int64),
        }
        if smoke:
            prepared[name] = subset_subjects(prepared[name], 4 if name == "dreyer2023" else 3)
    return prepared, shared


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def existing_rows(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() and path.stat().st_size else []


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    mean, std = norm_stats(x_train)
    csp, scaler, classifier = fit_csp_lda(
        ManualCSP,
        ((x_train - mean) / std).astype(np.float32),
        y_train,
        N_CSP,
    )
    return predict(csp, scaler, classifier, ((x_test - mean) / std).astype(np.float32))


def split_eval_pool(y: np.ndarray, subject: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(subject) + 7919)
    evaluation = np.sort(
        np.concatenate(
            [rng.choice(np.flatnonzero(y == label), int((y == label).sum()) // 2, replace=False) for label in (0, 1)]
        )
    )
    return evaluation, np.setdiff1d(np.arange(len(y)), evaluation)


def class_covariances(x: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    covariances = np.einsum("nct,ndt->ncd", x.astype(np.float64), x.astype(np.float64), optimize=True)
    return {label: covariances[y == label].mean(axis=0) for label in (0, 1)}


def run_floor_diagnostics(target: dict, suffix: str) -> list[dict]:
    path = HERE / f"floor_diagnostics{suffix}.csv"
    rows = []
    for subject in np.unique(target["subjects"]):
        x = target["X"][target["subjects"] == subject]
        covariance = mean_covariance(x, np.float32)
        eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) * 0.5)
        rows.append(
            {
                "subject": int(subject),
                "n_trials": len(x),
                "n_eigenvalues": len(eigenvalues),
                "below_1e-8": int((eigenvalues < LEGACY_FLOOR).sum()),
                "eigen_min": float(eigenvalues[0]),
                "eigen_median": float(np.median(eigenvalues)),
                "eigen_max": float(eigenvalues[-1]),
            }
        )
    write_csv(path, rows)
    print(f"[diagnostic] median floored eigenvalues={np.median([row['below_1e-8'] for row in rows]):.1f}", flush=True)
    return rows


def aligned_sets(data: dict, mode: str, alignment: str) -> dict:
    result = {}
    for name, values in data.items():
        if alignment == "none":
            x = values["X"]
        else:
            x = apply_alignment(values["X"], values["subjects"], mode, alignment, name)[0]
        result[name] = {**values, "X": x}
    return result


def run_cross(data: dict, suffix: str) -> list[dict]:
    output = HERE / f"cross_subject_results{suffix}.csv"
    rows = existing_rows(output)
    done = {(str(row["condition"]), int(row["subject"])) for row in rows}
    conditions = (
        ("none", "relative", "none"),
        ("relative_subject", "relative", "subject"),
        ("relative_dataset_subject", "relative", "dataset_subject"),
    )
    for condition, mode, alignment in conditions:
        target_subjects = np.unique(data["dreyer2023"]["subjects"])
        if all((condition, int(subject)) in done for subject in target_subjects):
            continue
        start = time.perf_counter()
        aligned = aligned_sets(data, mode, alignment)
        source_x = np.concatenate([aligned[name]["X"] for name in DEV_NAMES])
        source_y = np.concatenate([aligned[name]["y"] for name in DEV_NAMES])
        target = aligned["dreyer2023"]
        predictions = fit_predict(source_x, source_y, target["X"])
        for subject in target_subjects:
            if (condition, int(subject)) in done:
                continue
            mask = target["subjects"] == subject
            rows.append(
                {
                    "condition": condition,
                    "subject": int(subject),
                    "n_test": int(mask.sum()),
                    **metrics(target["y"][mask], predictions[mask]),
                }
            )
        write_csv(output, rows)
        selected = [row["bac"] for row in rows if row["condition"] == condition]
        print(
            f"[cross] {condition}: BAC={np.mean(selected)*100:.2f}% "
            f"({(time.perf_counter()-start)/60:.1f} min)",
            flush=True,
        )
        del aligned, source_x, predictions
    return rows


def run_loso_and_labels(data: dict, suffix: str) -> tuple[list[dict], list[dict], list[dict]]:
    loso_path = HERE / f"loso_subject_results{suffix}.csv"
    sep_path = HERE / f"separability_splithalf{suffix}.csv"
    label_path = HERE / f"label5_subject_results{suffix}.csv"
    loso_rows = existing_rows(loso_path)
    sep_rows = existing_rows(sep_path)
    label_rows = existing_rows(label_path)
    loso_done = {(str(row["floor"]), int(row["subject"])) for row in loso_rows}
    sep_done = {int(row["subject"]) for row in sep_rows}
    label_done = {(int(row["subject"]), int(row["draw"])) for row in label_rows}

    target = data["dreyer2023"]
    aligned = {
        mode: apply_alignment(target["X"], target["subjects"], mode, "subject", "dreyer2023")[0]
        for mode in ("legacy", "relative")
    }
    subjects = np.unique(target["subjects"])
    for index, subject in enumerate(subjects, start=1):
        mask = target["subjects"] == subject
        y_train, y_test = target["y"][~mask], target["y"][mask]
        evaluation, pool = split_eval_pool(y_test, int(subject))
        for mode in ("legacy", "relative"):
            key = (mode, int(subject))
            if key in loso_done:
                continue
            start = time.perf_counter()
            predictions = fit_predict(aligned[mode][~mask], y_train, aligned[mode][mask])
            row = {
                "floor": mode,
                "subject": int(subject),
                "n_train": int((~mask).sum()),
                "n_test": int(mask.sum()),
                "n_eval": int(len(evaluation)),
                **metrics(y_test, predictions),
                "eval_bac": metrics(y_test[evaluation], predictions[evaluation])["bac"],
                "elapsed_min": (time.perf_counter() - start) / 60,
            }
            loso_rows.append(row)
            loso_done.add(key)
            write_csv(loso_path, loso_rows)

        x_subject = aligned["relative"][mask]
        if int(subject) not in sep_done:
            class_means = class_covariances(x_subject[pool], y_test[pool])
            relative_row = next(
                row for row in loso_rows if row["floor"] == "relative" and int(row["subject"]) == int(subject)
            )
            sep_rows.append(
                {
                    "subject": int(subject),
                    "n_calibration": int(len(pool)),
                    "n_evaluation": int(len(evaluation)),
                    "own_sep_P": float(distance_riemann(class_means[0], class_means[1])),
                    "loso_bac_E": float(relative_row["eval_bac"]),
                }
            )
            sep_done.add(int(subject))
            write_csv(sep_path, sep_rows)

        rng = np.random.default_rng(int(subject) + 7919)
        # Advance the generator exactly as in split_eval_pool before drawing labels.
        for label in (0, 1):
            rng.choice(np.flatnonzero(y_test == label), int((y_test == label).sum()) // 2, replace=False)
        relative_row = next(
            row for row in loso_rows if row["floor"] == "relative" and int(row["subject"]) == int(subject)
        )
        for draw in range(N_DRAWS):
            label_indices = np.concatenate(
                [rng.choice(pool[y_test[pool] == label], 5, replace=False) for label in (0, 1)]
            )
            key = (int(subject), draw)
            if key in label_done:
                continue
            x_train = np.concatenate([aligned["relative"][~mask], x_subject[label_indices]])
            labels_train = np.concatenate([y_train, y_test[label_indices]])
            predictions = fit_predict(x_train, labels_train, x_subject[evaluation])
            label_rows.append(
                {
                    "subject": int(subject),
                    "draw": draw,
                    "k_per_class": 5,
                    "n_train_source": int((~mask).sum()),
                    "n_train_target": int(len(label_indices)),
                    "n_eval": int(len(evaluation)),
                    "baseline_loso_bac": float(relative_row["eval_bac"]),
                    **metrics(y_test[evaluation], predictions),
                }
            )
            label_done.add(key)
            write_csv(label_path, label_rows)
        print(f"[loso/labels] subject {index}/{len(subjects)} ({int(subject)}) complete", flush=True)
    return loso_rows, sep_rows, label_rows


def main() -> int:
    args = parse_args()
    suffix = "_smoke" if args.smoke else ""
    started = time.time()
    data, shared = load_all(args.smoke)
    manifest = {
        "created_unix": started,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "threads": args.threads,
        "smoke": args.smoke,
        "n_csp": N_CSP,
        "n_draws_h6": N_DRAWS,
        "shared_channels": shared,
        "inputs": {
            **{name: {"path": str(path), "sha256": sha256(path)} for name, path in DEV_PATHS.items()},
            "dreyer2023": {"path": str(DREYER_PATH), "sha256": sha256(DREYER_PATH)},
        },
        "implementation_notes": {
            "split": "class-stratified half split; seed=subject+7919, matching the existing development analysis",
            "h6_draws": "3 draws, matching the existing development label-curve analysis",
            "ea": "audited run_floor_gate.apply_alignment with floor mode passed explicitly",
        },
    }
    (HERE / f"run_manifest{suffix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[input] 5 datasets, {len(shared)} shared channels; target={len(np.unique(data['dreyer2023']['subjects']))} subjects", flush=True)
    with threadpool_limits(limits=args.threads):
        run_floor_diagnostics(data["dreyer2023"], suffix)
        run_cross(data, suffix)
        run_loso_and_labels(data, suffix)
    print(f"[done] elapsed={(time.time()-started)/3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
