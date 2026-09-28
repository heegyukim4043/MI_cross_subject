"""Subject-informed ceiling for the decomposition.

Within-subject CV alone is data-limited (PhysionetMI has ~45 trials per subject), so it is not an upper
bound for LOSO. Here each target subject is evaluated by 5-fold CV where the training set is all other
subjects of the same dataset (relative-floor EA) PLUS the target's own labeled training folds, replicated so
that they carry about the same total weight as the source data (CSP-LDA has no sample weights).
ceiling - LOSO = value of the subject's own labels given the same source pool (subject-specific information).
Same setting as run_decomposition.py (21 shared channels, 200 samples, CSP(8)+shrinkage LDA).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_decomposition import NAMES, SUFFIX, N_T, PATHS, align_subjects, fit_csp_lda, load_npz, ManualCSP, metrics, norm_stats, predict  # noqa: E402


def one_subject(x, y, S, s):
    with threadpool_limits(limits=1):
        m = S == s
        xs, ys, xt, yt = x[~m], y[~m], x[m], y[m]
        pred = np.empty_like(yt)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=int(s)).split(xt, yt):
            rep = max(1, int(round(len(ys) / len(tr))))
            xtr = np.concatenate([xs, np.repeat(xt[tr], rep, axis=0)])
            ytr = np.concatenate([ys, np.repeat(yt[tr], rep)])
            mu, sd = norm_stats(xtr)
            csp, sc, clf = fit_csp_lda(ManualCSP, ((xtr - mu) / sd).astype(np.float32), ytr, 8)
            pred[te] = predict(csp, sc, clf, ((xt[te] - mu) / sd).astype(np.float32))
        return {"subject": int(s), "ceiling_src_plus_own": metrics(yt, pred)["bac"]}


def main():
    raw = {n: load_npz(PATHS[n]) for n in NAMES}
    shared = [c for c in raw[NAMES[0]]["ch_names"] if all(c in raw[n]["ch_names"] for n in NAMES)]
    rows = []
    for n in NAMES:
        d = raw[n]
        x = d["X"][:, [d["ch_names"].index(c) for c in shared], :N_T].astype(np.float32)
        y, S = d["y"].astype(np.int64), d["subjects"].astype(np.int64)
        xa = align_subjects(x, S, "relative", n, "subject")[0]
        out = Parallel(n_jobs=12)(delayed(one_subject)(xa, y, S, s) for s in np.unique(S))
        rows += [{"dataset": n, **r} for r in out]
        print(f"[ceiling] {n} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(HERE / f"ceiling_subject_results{SUFFIX}.csv", index=False)
    dec = pd.read_csv(HERE / f"decomposition_subject_results{SUFFIX}.csv").merge(res, on=["dataset", "subject"])
    dec.to_csv(HERE / f"decomposition_subject_results{SUFFIX}.csv", index=False)
    print(dec.groupby("dataset")[["within", "loso_relative", "ceiling_src_plus_own"]].mean().mul(100).round(2).to_string())


if __name__ == "__main__":
    main()
