"""Download and preprocess the sealed test set for method A (DESIGN.md): MOABB Stieger2021, session 1,
subjects S21-S62, left- vs right-hand trials. Run with .venv_moabb_new (Python 3.11, MOABB >= 1.7).
LeftRightImagery(fmin=8, fmax=30, tmin=0.5, tmax=2.5, resample=100), stored in volts, same as the others.
Session-1 .mat files are downloaded first by download_stieger_session1.py (MOABB's loader would fetch every
session); data_path is redirected to those local files. One npz per subject in parts/ (resumable);
the merged file is written once all subjects are processed. Only label-free summaries are printed.
"""

import time
from pathlib import Path

import moabb
import numpy as np
from moabb.datasets import Stieger2021
from moabb.paradigms import LeftRightImagery

HERE = Path(__file__).resolve().parent
PARTS = HERE / "sealed_stieger2021" / "parts"
OUT = HERE / "sealed_stieger2021" / "stieger2021_s21_s62_session1.npz"
SUBJECTS = list(range(21, 63))
LOCAL = Path(r"D:\mne_data\stieger2021_session1")


def local_data_path(self, subject, *args, **kwargs):
    f = LOCAL / f"S{subject}_Session_1.mat"
    if not f.exists():
        raise FileNotFoundError(f)
    return [str(f)]


Stieger2021.data_path = local_data_path
moabb.set_download_provider("upstream")  # otherwise MOABB 1.7 prefetches every session from NEMAR


def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    ds = Stieger2021(sessions=[1])
    par = LeftRightImagery(fmin=8, fmax=30, tmin=0.5, tmax=2.5, resample=100)
    skipped = []
    for s in SUBJECTS:
        part = PARTS / f"S{s:02d}.npz"
        if part.exists():
            continue
        for attempt in range(6):
            try:
                ep, lab, meta = par.get_data(ds, subjects=[s], return_epochs=True)
                break
            except Exception as err:  # network errors: back off and retry
                if attempt == 5:
                    skipped.append((s, repr(err)[:120])); print(f"  skip S{s}: {err}", flush=True); ep = None; break
                time.sleep(60 * (attempt + 1))
        if ep is None:
            continue
        y = np.array([0 if l == "left_hand" else 1 for l in lab], dtype=np.int64)
        np.savez(part, X=ep.get_data(copy=True).astype(np.float32), y=y, ch_names=np.array(ep.ch_names),
                 sessions=meta["session"].astype(str).to_numpy(), runs=meta["run"].astype(str).to_numpy())
        print(f"  S{s}: trials={len(y)} channels={len(ep.ch_names)}", flush=True)
    parts = sorted(PARTS.glob("S*.npz"))
    if len(parts) + len(skipped) < len(SUBJECTS):
        print("not all subjects processed yet; rerun to resume", flush=True); return
    data = [np.load(p, allow_pickle=True) for p in parts]
    names = list(data[0]["ch_names"])
    keep = [d for d in data if list(d["ch_names"]) == names]
    n_t = min(d["X"].shape[2] for d in keep)
    subj = np.concatenate([np.full(len(d["y"]), int(p.stem[1:])) for p, d in zip(parts, data) if list(d["ch_names"]) == names])
    np.savez(OUT, X=np.concatenate([d["X"][:, :, :n_t] for d in keep]), y=np.concatenate([d["y"] for d in keep]),
             subjects=subj, ch_names=np.array(names), sfreq=np.array(100.0))
    counts = np.bincount(subj)[np.unique(subj)]
    print(f"saved {OUT}: subjects={len(counts)}, trials/subject min={counts.min()} median={int(np.median(counts))} "
          f"max={counts.max()}, n_times={n_t}, channels={len(names)}, channel_mismatch={len(data) - len(keep)}, "
          f"skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
