"""Band-wise (filter-bank) EA vs broadband EA with identical filter-bank CSP-LDA features.

Sub-bands inside the stored 8-30 Hz signal: 8-12, 12-16, 16-20, 20-24, 24-30 Hz (4th-order zero-phase
Butterworth, sosfiltfilt). Features: per-band CSP (4 filters, log-variance) concatenated -> scaler ->
shrinkage LDA. Only the alignment differs between conditions:
  none          no alignment
  ea_broadband  relative-floor SubjectEA on the 8-30 Hz signal, then band split
  ea_bandwise   band split, then relative-floor SubjectEA separately in every band
Settings: Cho2017<->Lee2019 (common48) and 3-dataset LODO (21 shared channels). Batch transductive EA.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent / "cmpb_floor_gate_20260924", HERE.parent / "cmpb_budget_20260924", HERE, ROOT / "MI_test" / "MI_loso_project"):
    sys.path.insert(0, str(p))
from run_floor_gate import align_subjects, metrics, write_rows  # noqa: E402
from run_budget_causal import norm_stats  # noqa: E402
from run_lodo import PATHS  # noqa: E402
from run_bdea_bnci import load_npz  # noqa: E402
from mrfbcsp_loso import ManualCSP  # noqa: E402

BANDS = ((8, 12), (12, 16), (16, 20), (20, 24), (24, 30))
FS = 100.0


def load(names, channels=None):
    data = {n: load_npz(PATHS[n]) for n in names}
    shared = channels or [c for c in data[names[0]]["ch_names"] if all(c in data[n]["ch_names"] for n in names)]
    n_t = min(d["X"].shape[2] for d in data.values())
    return {n: {"X": d["X"][:, [d["ch_names"].index(c) for c in shared], :n_t].astype(np.float64),
                "y": d["y"].astype(np.int64), "subjects": d["subjects"].astype(np.int64)} for n, d in data.items()}, shared


def bands(x):
    return [sosfiltfilt(butter(4, b, btype="bandpass", fs=FS, output="sos"), x, axis=-1).astype(np.float32) for b in BANDS]


def prepare(d, cond):
    if cond == "none":
        return bands(d["X"])
    if cond == "ea_broadband":
        return bands(align_subjects(d["X"].astype(np.float32), d["subjects"], "relative", "x", "subject")[0])
    return [align_subjects(xb, d["subjects"], "relative", "x", "subject")[0] for xb in bands(d["X"])]


def fb_fit_predict(src_b, y_src, tgt_b):
    feats_s, feats_t = [], []
    for xs, xt in zip(src_b, tgt_b):
        mu, sd = norm_stats(xs)
        xs_n, xt_n = ((xs - mu) / sd).astype(np.float32), ((xt - mu) / sd).astype(np.float32)
        csp = ManualCSP(n_components=4); csp.fit(xs_n, y_src)
        feats_s.append(csp.transform(xs_n)); feats_t.append(csp.transform(xt_n))
    sc = StandardScaler().fit(np.hstack(feats_s))
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(np.hstack(feats_s)), y_src)
    return clf.predict(sc.transform(np.hstack(feats_t)))


def run_setting(label, src, tgt, rows):
    for cond in ("none", "ea_broadband", "ea_bandwise"):
        t0 = time.perf_counter()
        pred = fb_fit_predict(prepare(src, cond), src["y"], prepare(tgt, cond))
        for s in np.unique(tgt["subjects"]):
            m = tgt["subjects"] == s
            rows.append({**label, "condition": cond, "subject": int(s), **metrics(tgt["y"][m], pred[m])})
        print(f"  {label} {cond} {(time.perf_counter()-t0)/60:.1f} min", flush=True)


def concat(parts):
    off, subj = 0, []
    for p in parts:
        subj.append(p["subjects"] + off); off = int(subj[-1].max()) + 1000
    return {"X": np.concatenate([p["X"] for p in parts]), "y": np.concatenate([p["y"] for p in parts]), "subjects": np.concatenate(subj)}


def main() -> int:
    rows = []
    with threadpool_limits(limits=6):
        pair, _ = load(["cho2017", "lee2019"])
        for s, t in (("cho2017", "lee2019"), ("lee2019", "cho2017")):
            run_setting({"setting": "pair48", "source": s, "target": t}, pair[s], pair[t], rows)
        names = ["cho2017", "lee2019", "bnci2014001"]
        three, shared = load(names)
        for held in names:
            run_setting({"setting": f"lodo3_{len(shared)}ch", "source": "+".join(n for n in names if n != held), "target": held},
                        concat([three[n] for n in names if n != held]), three[held], rows)
            write_rows(HERE / "fb_ea_subject_results.csv", rows)
    write_rows(HERE / "fb_ea_subject_results.csv", rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    print((df.groupby(["setting", "target", "condition"]).bac.mean().unstack() * 100).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
