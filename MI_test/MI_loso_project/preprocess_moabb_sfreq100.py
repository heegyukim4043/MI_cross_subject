from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import resample


TARGET_SFREQ = 100.0
TARGET_N_TIMES = 201


def dataset_factory(name: str):
    from moabb.datasets import Cho2017

    if name == "cho2017":
        return Cho2017()
    if name == "lee2019":
        try:
            from moabb.datasets import Lee2019_MI
        except ImportError as exc:
            raise ImportError("This MOABB version does not expose Lee2019_MI.") from exc
        # training phase of both sessions only (EEG_MI_test excluded): 200 trials per subject
        return Lee2019_MI(train_run=True, test_run=False, sessions=(1, 2))
    raise ValueError(f"Unknown dataset: {name}")


def make_paradigm():
    from moabb.paradigms import MotorImagery

    kwargs = {
        "n_classes": 2,
        "fmin": 8,
        "fmax": 30,
        "tmin": 0.5,
        "tmax": 2.5,
    }
    sig = inspect.signature(MotorImagery)
    if "events" in sig.parameters:
        kwargs["events"] = ["left_hand", "right_hand"]
    if "resample" in sig.parameters:
        kwargs["resample"] = TARGET_SFREQ
    return MotorImagery(**kwargs)


def encode_labels(labels) -> tuple[np.ndarray, list[str]]:
    labels = np.asarray(labels)
    normalized = np.asarray([str(v).lower() for v in labels])
    y = np.full(len(labels), -1, dtype=np.int64)
    for key in ("left_hand", "left hand", "left"):
        y[normalized == key] = 0
    for key in ("right_hand", "right hand", "right"):
        y[normalized == key] = 1
    if np.any(y < 0):
        uniq = sorted(pd.unique(normalized).tolist())
        if len(uniq) != 2:
            raise ValueError(f"Expected two labels, got {uniq}")
        mapping = {label: i for i, label in enumerate(uniq)}
        y = np.asarray([mapping[v] for v in normalized], dtype=np.int64)
        return y, uniq
    return y, ["left_hand", "right_hand"]


def encode_subjects(metadata: pd.DataFrame) -> np.ndarray:
    if "subject" not in metadata.columns:
        raise KeyError(f"MOABB metadata has no subject column: {metadata.columns.tolist()}")
    subjects = pd.to_numeric(metadata["subject"], errors="coerce")
    if subjects.isna().any():
        codes, _ = pd.factorize(metadata["subject"], sort=True)
        return (codes + 1).astype(np.int64)
    return subjects.astype(np.int64).to_numpy()


def force_target_shape(X: np.ndarray) -> np.ndarray:
    if X.shape[2] == TARGET_N_TIMES:
        return X
    return resample(X, TARGET_N_TIMES, axis=2)


def preprocess_one(name: str, out_dir: Path, subjects: Optional[List[int]] = None) -> Path:
    dataset = dataset_factory(name)
    selected_subjects = subjects or list(dataset.subject_list)
    paradigm = make_paradigm()

    print(f"[load] {name}: subjects={len(selected_subjects)}")
    epochs, labels, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=selected_subjects,
        return_epochs=True,
    )
    epochs.load_data()
    if abs(float(epochs.info["sfreq"]) - TARGET_SFREQ) > 1e-6:
        print(f"[resample] {name}: {epochs.info['sfreq']} Hz -> {TARGET_SFREQ} Hz")
        epochs.resample(TARGET_SFREQ, npad="auto")

    X = epochs.get_data(copy=True).astype(np.float32)
    X = force_target_shape(X).astype(np.float32)
    y, label_names = encode_labels(labels)
    subj = encode_subjects(metadata)
    ch_names = np.asarray(epochs.ch_names, dtype=object)

    if len(X) != len(y) or len(X) != len(subj):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, subjects={len(subj)}")
    if X.shape[2] != TARGET_N_TIMES:
        raise ValueError(f"Unexpected n_times after resample: {X.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=y.astype(np.int64),
        subjects=subj.astype(np.int64),
        ch_names=ch_names,
        sfreq=np.float32(TARGET_SFREQ),
        label_names=np.asarray(label_names, dtype=object),
    )
    print(
        f"[save] {out_path} X={X.shape} subjects={len(np.unique(subj))} "
        f"labels={dict(zip(*np.unique(y, return_counts=True)))}"
    )
    return out_path


def parse_subjects(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="MI_test/preprocessed_sfreq100")
    parser.add_argument("--datasets", nargs="+", choices=["cho2017", "lee2019"], default=["cho2017", "lee2019"])
    parser.add_argument("--subjects", default=None, help="Optional comma-separated subject list for smoke tests.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    subjects = parse_subjects(args.subjects)
    paths = [preprocess_one(name, out_dir, subjects=subjects) for name in args.datasets]
    print("[done]")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
