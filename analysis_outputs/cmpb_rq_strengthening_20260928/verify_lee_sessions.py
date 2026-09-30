"""Verify, for every Lee2019 subject, that the stored NPZ rows 0-99 are session 1 and rows 100-199 session 2.

Each subject is re-extracted with the same MOABB path as the stored file (MotorImagery, 8-30 Hz, 0.5-2.5 s, 100 Hz,
training phase of both sessions) and the 200-sample epochs are resampled to 201 samples with scipy.signal.resample,
exactly as in preprocess_moabb_sfreq100.py. The fresh epochs are compared with the stored rows (relative max difference
and per-trial correlation, same order and with the two halves swapped), and the MOABB session metadata of the fresh
epochs is checked (first 100 = first session). Labels are not used.
The raw data (about 67 GB) are downloaded to LEE_DOWNLOAD_DIR (default: MNE_DATA or ~/mne_data). With
LEE_DELETE_RAW=1 each subject's newly downloaded raw files are deleted after its check (files cached before the run
are kept). Resumable: finished subjects are read from the CSV.
Output: lee_sessions/lee_session_check.csv
"""
import os
DL_DIR = os.environ.get("LEE_DOWNLOAD_DIR") or os.environ.get("MNE_DATA") or os.path.expanduser("~/mne_data")
os.environ["MNE_DATASETS_LEE2019-MI_PATH"] = DL_DIR  # read by MNE before the config file
os.environ["MNE_DATA"] = DL_DIR
DELETE_RAW = os.environ.get("LEE_DELETE_RAW") == "1"

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample
from moabb.datasets import Lee2019_MI
from moabb.paradigms import MotorImagery

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "lee_sessions" / "lee_session_check.csv"
DL = Path(DL_DIR)


def corr(a, b):
    a = a.reshape(len(a), -1); b = b.reshape(len(b), -1)
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def main():
    OUT.parent.mkdir(exist_ok=True)
    npz = np.load(ROOT / "MI_test/preprocessed_sfreq100/lee2019.npz", allow_pickle=True)
    X, S, ch = npz["X"], npz["subjects"], [str(c) for c in npz["ch_names"]]
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {r["subject"] for r in rows}
    ds = Lee2019_MI(train_run=True, test_run=False, sessions=(1, 2))
    par = MotorImagery(n_classes=2, fmin=8, fmax=30, tmin=0.5, tmax=2.5, resample=100.0)
    for s in np.unique(S):
        if int(s) in done:
            continue
        before = {p for p in DL.rglob("*.mat")} if DL.exists() else set()
        ep, _, meta = par.get_data(ds, subjects=[int(s)], return_epochs=True)
        x = ep.get_data(copy=True)[:, [ep.ch_names.index(c) for c in ch], :]
        x = resample(x, X.shape[2], axis=2)
        st = X[S == s]
        ses = meta["session"].astype(str).to_numpy(); first = sorted(set(ses))[0]
        half = len(st) // 2
        r = {"subject": int(s), "n_fresh": len(x), "n_stored": len(st),
             "rel_max_diff": float(np.abs(x - st).max() / np.abs(st).max()) if x.shape == st.shape else np.nan,
             "corr_min_same": float(corr(x, st).min()) if x.shape == st.shape else np.nan,
             "corr_median_swapped": float(np.median(corr(x, np.concatenate([st[half:], st[:half]])))) if x.shape == st.shape else np.nan,
             "first_half_session": "|".join(sorted(set(ses[:half]))), "second_half_session": "|".join(sorted(set(ses[half:]))),
             "first_half_is_first_session": bool((ses[:half] == first).all() and (ses[half:] != first).all())}
        r["verified"] = bool(r["first_half_is_first_session"] and r["corr_min_same"] > 0.999)
        rows.append(r); pd.DataFrame(rows).to_csv(OUT, index=False)
        print(r, flush=True)
        for p in (DL.rglob("*.mat") if DL.exists() and DELETE_RAW else []):
            if p not in before and "lee2019" in str(p).lower():
                p.unlink()
    d = pd.DataFrame(rows)
    print(f"subjects {len(d)}; verified {int(d.verified.sum())}; not verified {d.loc[~d.verified, 'subject'].tolist()}", flush=True)


if __name__ == "__main__":
    main()
