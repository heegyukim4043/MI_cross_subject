"""Preprocess PhysionetMI (EEGMMIDB) imagined left/right-hand MI with the same MOABB path as the
other inputs: 8-30 Hz band-pass, 0.5-2.5 s after the cue, 100 Hz. Keeps run labels and the
chronological order. Output: physionetmi.npz with X (volts), y (0 = left, 1 = right), subjects,
sessions, runs, ch_names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from moabb.datasets import PhysionetMI
from moabb.paradigms import LeftRightImagery

OUT = Path(__file__).resolve().parent / "physionetmi.npz"


def main() -> int:
    dataset = PhysionetMI(imagined=True, executed=False)
    paradigm = LeftRightImagery(fmin=8, fmax=30, tmin=0.5, tmax=2.5, resample=100)
    xs, ys, metas, names = [], [], [], None
    for s in dataset.subject_list:
        try:
            epochs, labels, meta = paradigm.get_data(dataset, subjects=[s], return_epochs=True)
        except Exception as err:  # a few subjects have irregular recordings
            print(f"  skip subject {s}: {err}", flush=True)
            continue
        if names is None:
            names = epochs.ch_names
        if epochs.ch_names != names:
            print(f"  skip subject {s}: channel mismatch", flush=True)
            continue
        xs.append(epochs.get_data(copy=True).astype(np.float32)); ys.append(np.asarray(labels)); metas.append(meta)
        if s % 10 == 0:
            print(f"  subject {s} done", flush=True)
    n_t = min(x.shape[2] for x in xs)
    x = np.concatenate([x[:, :, :n_t] for x in xs])
    labels = np.concatenate(ys)
    import pandas as pd
    meta = pd.concat(metas, ignore_index=True)
    y = np.array([0 if lab == "left_hand" else 1 for lab in labels], dtype=np.int64)
    run_codes = {r: i for i, r in enumerate(sorted(meta["run"].astype(str).unique()))}
    ses_codes = {r: i for i, r in enumerate(sorted(meta["session"].astype(str).unique()))}
    subjects = meta["subject"].to_numpy(dtype=np.int64)
    runs = meta["run"].astype(str).map(run_codes).to_numpy(dtype=np.int64)
    sessions = meta["session"].astype(str).map(ses_codes).to_numpy(dtype=np.int64)
    order = np.lexsort((np.arange(len(y)), runs, sessions, subjects))
    np.savez(OUT, X=x[order], y=y[order], subjects=subjects[order], sessions=sessions[order], runs=runs[order],
             ch_names=np.array(names), sfreq=np.array(100.0), run_names=np.array(sorted(run_codes)))
    counts = np.bincount(subjects)[np.unique(subjects)]
    print(f"saved {OUT}: X={x.shape}, subjects={len(np.unique(subjects))}, trials/subject min={counts.min()} "
          f"median={int(np.median(counts))} max={counts.max()}, runs={run_codes}, channels={names}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
