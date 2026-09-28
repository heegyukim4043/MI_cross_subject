"""Preprocess BNCI2014-001 (BCI Competition IV 2a) left/right-hand MI for the BD-EA replication.

Same MOABB path as the Cho2017/Lee2019 sfreq100 inputs: 8-30 Hz band-pass, 0.5-2.5 s after the
cue, resampled to 100 Hz (201 samples). Keeps session and run labels and the chronological order
(MOABB returns epochs ordered by subject, session, run, and time within a run).
Output: bnci2014001.npz with X (volts), y (0 = left, 1 = right), subjects, sessions, runs, ch_names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

OUT = Path(__file__).resolve().parent / "bnci2014001.npz"


def main() -> int:
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=8, fmax=30, tmin=0.5, tmax=2.5, resample=100)
    epochs, labels, meta = paradigm.get_data(dataset, subjects=dataset.subject_list, return_epochs=True)
    x = epochs.get_data(copy=True).astype(np.float32)
    y = np.array([0 if lab == "left_hand" else 1 for lab in labels], dtype=np.int64)
    session_codes = {s: i for i, s in enumerate(sorted(meta["session"].unique()))}
    run_codes = {r: i for i, r in enumerate(sorted(meta["run"].unique()))}
    subjects = meta["subject"].to_numpy(dtype=np.int64)
    sessions = meta["session"].map(session_codes).to_numpy(dtype=np.int64)
    runs = meta["run"].map(run_codes).to_numpy(dtype=np.int64)
    # enforce chronological order: subject, session, run, original within-run order
    order = np.lexsort((np.arange(len(y)), runs, sessions, subjects))
    np.savez(OUT, X=x[order], y=y[order], subjects=subjects[order], sessions=sessions[order], runs=runs[order],
             ch_names=np.array(epochs.ch_names), sfreq=np.array(epochs.info["sfreq"]),
             session_names=np.array(sorted(session_codes)), run_names=np.array(sorted(run_codes)))
    print(f"saved {OUT}: X={x.shape}, sfreq={epochs.info['sfreq']}, sessions={session_codes}, runs={run_codes}")
    for s in np.unique(subjects):
        m = subjects[order] == s
        print(f"  subject {s}: trials={m.sum()}, per session={np.bincount(sessions[order][m]).tolist()}, "
              f"label runs={int(np.sum(np.diff(y[order][m]) != 0) + 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
