"""Unified label curve and split-half separability (CSP-LDA, same pipeline as the decomposition).

Per target subject (LOSO within its dataset; relative-floor SubjectEA computed from all of the subject's
trials, label-free): a fixed class-stratified split into a calibration half P and an evaluation half E.
Only E is scored; labels of P are used for calibration and for the separability diagnostics.

Label curve (k labeled trials per class drawn from P, 3 draws, k in {5,10,20,40,80} if available):
  loso            source subjects only (no target labels)
  src_own_nat     source + k-shot own trials, natural weight
  src_own_eq      source + k-shot own trials replicated to about equal total weight
  own_only        subject-specific CSP-LDA on the k-shot trials only
Split-half diagnostics (from P, related to accuracy on E, so not circular):
  own_sep_P    Riemannian distance between the subject's class-mean covariances in P
  cond_shift_P mean Riemannian distance of the subject's P class means to the other subjects' class means
Env DECOMP_SET selects the dataset set (see run_decomposition.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_decomposition import NAMES, SUFFIX, N_T, PATHS, align_subjects, fit_csp_lda, load_npz, ManualCSP, metrics, norm_stats, predict  # noqa: E402
from pyriemann.utils.distance import distance_riemann  # noqa: E402

KS = (5, 10, 20, 40, 80)
DRAWS = 3


def fit_pred(xtr, ytr, xte):
    mu, sd = norm_stats(xtr)
    csp, sc, clf = fit_csp_lda(ManualCSP, ((xtr - mu) / sd).astype(np.float32), ytr, 8)
    return predict(csp, sc, clf, ((xte - mu) / sd).astype(np.float32))


def cmeans(x, y):
    c = np.einsum("nct,ndt->ncd", x.astype(np.float64), x.astype(np.float64))
    return {k: c[y == k].mean(axis=0) for k in (0, 1)}


def one_subject(dataset, x, y, S, s, rest_means):
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(int(s) + 7919)
        m = S == s
        xs, ys, xt, yt = x[~m], y[~m], x[m], y[m]
        e = np.sort(np.concatenate([rng.choice(np.flatnonzero(yt == c), (yt == c).sum() // 2, replace=False) for c in (0, 1)]))
        p = np.setdiff1d(np.arange(len(yt)), e)
        base = {"dataset": dataset, "subject": int(s), "n_eval": len(e)}
        rows = [{**base, "condition": "loso", "k": 0, "draw": 0, "bac": metrics(yt[e], fit_pred(xs, ys, xt[e]))["bac"]}]
        own = cmeans(xt[p], yt[p])
        diag = {**base, "own_sep_P": float(distance_riemann(own[0], own[1])),
                "cond_shift_P": float(np.mean([distance_riemann(own[k], rest_means[k]) for k in (0, 1)])),
                "loso_bac_E": rows[0]["bac"]}
        for k in KS:
            if any((yt[p] == c).sum() < k for c in (0, 1)):
                continue
            for dr in range(DRAWS):
                lab = np.concatenate([rng.choice(p[yt[p] == c], k, replace=False) for c in (0, 1)])
                rep = max(1, int(round(len(ys) / len(lab))))
                for cond, (xtr, ytr) in {
                    "src_own_nat": (np.concatenate([xs, xt[lab]]), np.concatenate([ys, yt[lab]])),
                    "src_own_eq": (np.concatenate([xs, np.repeat(xt[lab], rep, 0)]), np.concatenate([ys, np.repeat(yt[lab], rep)])),
                    "own_only": (xt[lab], yt[lab]),
                }.items():
                    rows.append({**base, "condition": cond, "k": k, "draw": dr, "bac": metrics(yt[e], fit_pred(xtr, ytr, xt[e]))["bac"]})
        return rows, diag


def main():
    raw = {n: load_npz(PATHS[n]) for n in NAMES}
    shared = [c for c in raw[NAMES[0]]["ch_names"] if all(c in raw[n]["ch_names"] for n in NAMES)]
    rows, diags = [], []
    for n in NAMES:
        d = raw[n]
        x = d["X"][:, [d["ch_names"].index(c) for c in shared], :N_T].astype(np.float32)
        y, S = d["y"].astype(np.int64), d["subjects"].astype(np.int64)
        xa = align_subjects(x, S, "relative", n, "subject")[0]
        allm = {s: cmeans(xa[S == s], y[S == s]) for s in np.unique(S)}
        jobs = []
        for s in np.unique(S):
            others = [allm[o] for o in allm if o != s]
            rest = {k: np.mean([o[k] for o in others], axis=0) for k in (0, 1)}
            jobs.append(delayed(one_subject)(n, xa, y, S, s, rest))
        for r, dg in Parallel(n_jobs=12)(jobs):
            rows += r; diags.append(dg)
        print(f"[labels/sep] {n} done ({len(shared)} ch)", flush=True)
    pd.DataFrame(rows).to_csv(HERE / f"label_curve_subject_results{SUFFIX}.csv", index=False)
    pd.DataFrame(diags).to_csv(HERE / f"separability_splithalf{SUFFIX}.csv", index=False)
    print("[labels/sep] saved", flush=True)


if __name__ == "__main__":
    main()
