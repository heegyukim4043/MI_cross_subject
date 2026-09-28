"""Quantitative decomposition of performance loss in cross-subject / cross-dataset MI decoding.

Common setting for every component: the 21 channels shared by Cho2017, Lee2019, BNCI2014-001 and
PhysionetMI; 100 Hz, 200 samples, 8-30 Hz; CSP(8)+shrinkage LDA; subject-level balanced accuracy.

Components per dataset (subject means):
  within   within-subject 5-fold stratified CV (subject-specific model; upper reference). Cho2017 trials
           are stored class-blocked, so its within-subject estimate may be inflated by slow drift that is
           confounded with class (reported as a caveat).
  loso_none / loso_legacy / loso_rel   leave-one-subject-out with no EA, absolute-floor EA, relative EA
  cross_rel                            4-dataset LODO with relative EA (from run_lodo.py output)
Losses: numerical (loso_rel - loso_legacy), marginal shift (loso_rel - loso_none), cross-dataset-specific
(loso_rel - cross_rel), subject-specific transfer loss (within - loso_rel), separability ceiling (1 - within).

Diagnostics (analysis only, use target labels): after relative EA, per target subject s
  cond_shift  mean over classes of the Riemannian distance between the subject's class-mean covariance and
              the class-mean covariance of all other subjects of the dataset (conditional shift)
  own_sep     Riemannian distance between the subject's two class-mean covariances (own separability)
Subject typing (threshold 0.70): decodable+transferable (within>=.70, loso>=.70), transfer-limited
(within>=.70, loso<.70), decodability-limited (within<.70).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
A = HERE.parent
ROOT = A.parent
for p in (A / "cmpb_floor_gate_20260924", A / "cmpb_budget_20260924", A / "cmpb_bdea_20260924", ROOT / "MI_test" / "MI_loso_project"):
    sys.path.insert(0, str(p))
from run_floor_gate import align_subjects, fit_csp_lda, metrics, predict, write_rows  # noqa: E402
from run_budget_causal import norm_stats  # noqa: E402
from run_bdea_bnci import load_npz  # noqa: E402
from run_lodo import PATHS  # noqa: E402
from mrfbcsp_loso import ManualCSP  # noqa: E402
from pyriemann.utils.distance import distance_riemann  # noqa: E402

import os
# DECOMP_SET selects the dataset set: "4ds" (default; 21 shared channels) or "3ds48" (Cho, Lee, PhysionetMI; 48 channels)
DECOMP_SET = os.environ.get("DECOMP_SET", "4ds")
NAMES = ["cho2017", "lee2019", "bnci2014001", "physionetmi"] if DECOMP_SET == "4ds" else ["cho2017", "lee2019", "physionetmi"]
SUFFIX = "" if DECOMP_SET == "4ds" else f"_{DECOMP_SET}"
LODO_FILE = ("lodo_cho2017_lee2019_bnci2014001_physionetmi_subject_results.csv" if DECOMP_SET == "4ds"
             else "lodo_cho2017_lee2019_physionetmi_subject_results.csv")
N_T = 200


def fit_eval(xtr, ytr, xte, yte):
    mu, sd = norm_stats(xtr)
    csp, sc, clf = fit_csp_lda(ManualCSP, ((xtr - mu) / sd).astype(np.float32), ytr, 8)
    return metrics(yte, predict(csp, sc, clf, ((xte - mu) / sd).astype(np.float32)))["bac"]


def within_subject(x, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    for tr, te in skf.split(x, y):
        mu, sd = norm_stats(x[tr])
        csp, sc, clf = fit_csp_lda(ManualCSP, ((x[tr] - mu) / sd).astype(np.float32), y[tr], 8)
        pred[te] = predict(csp, sc, clf, ((x[te] - mu) / sd).astype(np.float32))
    return metrics(y, pred)["bac"]


def class_means(x, y):
    c = np.einsum("nct,ndt->ncd", x.astype(np.float64), x.astype(np.float64))
    return {k: c[y == k].sum(axis=0) for k in (0, 1)}, {k: int((y == k).sum()) for k in (0, 1)}


def main() -> int:
    raw = {n: load_npz(PATHS[n]) for n in NAMES}
    shared = [c for c in raw[NAMES[0]]["ch_names"] if all(c in raw[n]["ch_names"] for n in NAMES)]
    rows = []
    with threadpool_limits(limits=8):
        for n in NAMES:
            d = raw[n]
            x = d["X"][:, [d["ch_names"].index(c) for c in shared], :N_T].astype(np.float32)
            y, S = d["y"].astype(np.int64), d["subjects"].astype(np.int64)
            xa = {"none": x, "legacy": align_subjects(x, S, "legacy", n, "subject")[0],
                  "relative": align_subjects(x, S, "relative", n, "subject")[0]}
            sums, counts = {}, {}
            for s in np.unique(S):
                sums[s], counts[s] = class_means(xa["relative"][S == s], y[S == s])
            tot = {k: sum(sums[s][k] for s in sums) for k in (0, 1)}
            ntot = {k: sum(counts[s][k] for s in counts) for k in (0, 1)}
            for s in np.unique(S):
                m = S == s
                r = {"dataset": n, "subject": int(s), "n_trials": int(m.sum()),
                     "within": within_subject(xa["relative"][m], y[m], int(s))}
                for mode in ("none", "legacy", "relative"):
                    r[f"loso_{mode}"] = fit_eval(xa[mode][~m], y[~m], xa[mode][m], y[m])
                own = {k: sums[s][k] / counts[s][k] for k in (0, 1)}
                rest = {k: (tot[k] - sums[s][k]) / (ntot[k] - counts[s][k]) for k in (0, 1)}
                r["cond_shift"] = float(np.mean([distance_riemann(own[k], rest[k]) for k in (0, 1)]))
                r["own_sep"] = float(distance_riemann(own[0], own[1]))
                r["rest_sep"] = float(distance_riemann(rest[0], rest[1]))
                rows.append(r)
            print(f"[decomp] {n}: {len(np.unique(S))} subjects done", flush=True)
    df = pd.DataFrame(rows)
    lodo = pd.read_csv(A / "cmpb_bdea_20260924" / LODO_FILE)
    lodo = lodo[(lodo.floor == "relative") & (lodo.alignment == "subject")][["held_out", "subject", "bac"]]
    df = df.merge(lodo.rename(columns={"held_out": "dataset", "bac": "cross_rel"}), on=["dataset", "subject"], how="left")
    write_rows(HERE / f"decomposition_subject_results{SUFFIX}.csv", df.to_dict("records"))
    print(f"[decomp] saved {len(df)} subjects; shared channels = {len(shared)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
