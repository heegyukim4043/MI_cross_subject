"""Labeled-trial budget for recovering the class-conditional (rotation) shift after scale-correct re-centering.

Pipeline (21 shared channels, OAS covariances): per-subject Riemannian re-centering + stretch (label-free,
all target trials), TS-LR trained on the source subjects. Per target subject a fixed class-stratified
evaluation half E is scored; k labeled trials per class are drawn from the other half P (5 draws per k).
Conditions (all scored on E):
  unlabeled          no target labels (re-centering + stretch only)
  rot_k              rotate the target so its k-shot class means match the pooled source class means (RPA step 3)
  pool_k             retrain LR on source + the k-shot target trials (target weighted to 50% of the total weight)
  rot_pool           rotation from all labeled trials in P (practical upper reference)
Settings: Cho2017<->Lee2019 pairs and 4-dataset LODO (Cho, Lee, BNCI2014-001, PhysionetMI).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cmpb_floor_gate_20260924"))
from run_floor_gate import metrics, write_rows  # noqa: E402
from run_beyond_ea import center, concat, fit_models, load_set, stretch  # noqa: E402
from pyriemann.utils.mean import mean_riemann  # noqa: E402

K_SHOTS = (1, 2, 3, 5, 10, 20)
N_DRAWS = 5


def split_eval(y, rng):
    e = np.concatenate([rng.choice(np.flatnonzero(y == c), size=(y == c).sum() // 2, replace=False) for c in (0, 1)])
    e = np.sort(e)
    return e, np.setdiff1d(np.arange(len(y)), e)


def subject_rows(ctx, ts, lr, feats_src, y_src, src_means, x, y, subject, seed):
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(seed)
        e, pool = split_eval(y, rng)
        base = {**ctx, "subject": int(subject), "n_eval": len(e)}
        f_e = ts.transform(x[e])
        out = [{**base, "condition": "unlabeled", "k": 0, "draw": 0, **metrics(y[e], lr.predict(f_e))}]
        # rotation from all labeled trials of the calibration half P only (evaluation labels are never used)
        m_pool = np.stack([mean_riemann(x[pool][y[pool] == c]) for c in (0, 1)])
        q_pool = _rotation(m_pool, src_means)
        out.append({**base, "condition": "rot_pool", "k": int(min((y[pool] == c).sum() for c in (0, 1))), "draw": 0,
                    **metrics(y[e], lr.predict(ts.transform(np.einsum("ij,njk,lk->nil", q_pool, x[e], q_pool))))})
        for k in K_SHOTS:
            if any((y[pool] == c).sum() < k for c in (0, 1)):
                continue
            for dr in range(N_DRAWS):
                lab = np.concatenate([rng.choice(pool[y[pool] == c], size=k, replace=False) for c in (0, 1)])
                m_t = np.stack([mean_riemann(x[lab][y[lab] == c]) if k > 1 else x[lab][y[lab] == c][0] for c in (0, 1)])
                q = _rotation(m_t, src_means)
                xr = np.einsum("ij,njk,lk->nil", q, x[e], q)
                out.append({**base, "condition": "rot_k", "k": k, "draw": dr, **metrics(y[e], lr.predict(ts.transform(xr)))})
                w_t = 0.5 * len(y_src) / len(lab)
                clf = LogisticRegression(C=1.0, max_iter=3000).fit(
                    np.vstack([feats_src, ts.transform(x[lab])]), np.concatenate([y_src, y[lab]]),
                    sample_weight=np.concatenate([np.ones(len(y_src)), np.full(len(lab), w_t)]))
                out.append({**base, "condition": "pool_k", "k": k, "draw": dr, **metrics(y[e], clf.predict(f_e))})
        return out


def _rotation(m_t, src_means):
    from pyriemann.transfer._rotate import _get_rotation_matrix
    return _get_rotation_matrix(m_t, src_means, metric="riemann", maxiter=500)


def evaluate(ctx, src, tgt, rows, seed0):
    cs = stretch(center(src["C"], src["subjects"], "ra"), src["subjects"])
    ct = stretch(center(tgt["C"], tgt["subjects"], "ra"), tgt["subjects"])
    model = fit_models(cs, src["y"])["TS-LR"]
    ts, lr = model[0], model[1]
    feats_src = ts.transform(cs)
    src_means = np.stack([mean_riemann(cs[src["y"] == k]) for k in (0, 1)])
    subjects = np.unique(tgt["subjects"])
    jobs = (delayed(subject_rows)(ctx, ts, lr, feats_src, src["y"], src_means, ct[tgt["subjects"] == s],
                                  tgt["y"][tgt["subjects"] == s], s, seed0 + int(s)) for s in subjects)
    for out in Parallel(n_jobs=12)(jobs):
        rows.extend(out)


def main() -> int:
    smoke = "--smoke" in sys.argv
    rows = []
    with threadpool_limits(limits=8):
        names = ["cho2017", "lee2019", "bnci2014001", "physionetmi"]
        data, shared = load_set(names)
        if smoke:
            data = {k: {kk: vv[np.isin(v["subjects"], np.unique(v["subjects"])[:3])] for kk, vv in v.items()} for k, v in data.items()}
        for s, t in (("cho2017", "lee2019"), ("lee2019", "cho2017")):
            t0 = time.perf_counter()
            evaluate({"setting": f"pair{len(shared)}", "source": s, "target": t}, data[s], data[t], rows, 1000)
            write_rows(HERE / "label_budget_subject_results.csv", rows)
            print(f"[labels] pair {s}->{t} {(time.perf_counter()-t0)/60:.1f} min", flush=True)
        for held in names:
            t0 = time.perf_counter()
            evaluate({"setting": f"lodo4_{len(shared)}ch", "source": "+".join(n for n in names if n != held), "target": held},
                     concat([data[n] for n in names if n != held]), data[held], rows, 2000)
            write_rows(HERE / "label_budget_subject_results.csv", rows)
            print(f"[labels] lodo4 held-out {held} {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    per = df.groupby(["setting", "target", "condition", "k", "subject"]).bac.mean().reset_index()
    pd.set_option("display.width", 250)
    print((per.groupby(["setting", "target", "condition", "k"]).bac.mean().unstack(["condition", "k"]) * 100).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
