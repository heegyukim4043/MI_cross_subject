"""Preprocess the sealed confirmation dataset Dreyer2023 exactly as pre-registered (CONFIRMATORY_PLAN.md).
Run with the isolated environment .venv_moabb_new (Python 3.11, MOABB >= 1.7).
LeftRightImagery(fmin=8, fmax=30, tmin=0.5, tmax=2.5, resample=100); all subjects and runs; stored in volts.
Output: dreyer2023.npz with X, y (0 left, 1 right), subjects, runs, ch_names, sfreq (same schema as the others).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from moabb.datasets import Dreyer2023
from moabb.paradigms import LeftRightImagery

OUT = Path(__file__).resolve().parent / "dreyer2023.npz"


def main():
    ds = Dreyer2023()
    par = LeftRightImagery(fmin=8, fmax=30, tmin=0.5, tmax=2.5, resample=100)
    xs, ys, metas, names, skipped = [], [], [], None, []
    import time
    for s in ds.subject_list:
        ep = None
        for attempt in range(8):  # OSF returns HTTP 429 when rate-limited: wait and retry
            try:
                ep, lab, meta = par.get_data(ds, subjects=[s], return_epochs=True)
                break
            except Exception as err:
                if "429" in repr(err) and attempt < 7:
                    wait = 60 * (attempt + 1)
                    print(f"  subject {s}: rate-limited, retry in {wait}s", flush=True); time.sleep(wait); continue
                skipped.append((s, repr(err)[:120])); print(f"  skip {s}: {err}", flush=True); break
        if ep is None:
            continue
        if names is None:
            names = ep.ch_names
        if ep.ch_names != names:
            skipped.append((s, "channel mismatch")); continue
        xs.append(ep.get_data(copy=True).astype(np.float32)); ys.append(np.asarray(lab)); metas.append(meta)
        if s % 10 == 0:
            print(f"  subject {s} done", flush=True)
    n_t = min(x.shape[2] for x in xs)
    x = np.concatenate([a[:, :, :n_t] for a in xs]); lab = np.concatenate(ys); meta = pd.concat(metas, ignore_index=True)
    y = np.array([0 if l == "left_hand" else 1 for l in lab], dtype=np.int64)
    runs = meta["run"].astype(str).map({r: i for i, r in enumerate(sorted(meta["run"].astype(str).unique()))}).to_numpy()
    subjects = meta["subject"].to_numpy(dtype=np.int64)
    order = np.lexsort((np.arange(len(y)), runs, subjects))
    np.savez(OUT, X=x[order], y=y[order], subjects=subjects[order], runs=runs[order], ch_names=np.array(names),
             sfreq=np.array(100.0), run_names=np.array(sorted(meta["run"].astype(str).unique())))
    counts = np.bincount(subjects)[np.unique(subjects)]
    print(f"saved {OUT}: X={x.shape}, subjects={len(np.unique(subjects))}, trials/subject min={counts.min()} "
          f"median={int(np.median(counts))} max={counts.max()}, channels={names}, skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
