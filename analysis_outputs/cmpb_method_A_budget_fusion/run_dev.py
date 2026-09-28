"""Method A development run (DESIGN.md): leave-one-dataset-out over Cho2017, Lee2019, BNCI2014-001, PhysionetMI.

For every held-out subject, fixed class-stratified split into pool P and evaluation set E (only E is scored),
k labeled trials per class drawn from P (k in KS, DRAWS draws), then BAC on E for:
  src_only                         source model (k=0 reference)
  own_only                         CSP-LDA on the k-shot trials only
  w=<share>                        source + own trials replicated to carry total weight share w (M1/M3 grid;
                                   "nat" = natural weight, 0.5 = the src_own_eq baseline)
  kappa=<kappa>                    M2 decision-level shrinkage a_k = k / (k + kappa)
  m3                               M3: w chosen per subject by stratified 5-fold CV on the k-shot trials
Nested selection for M1/M2 and the selection rule are applied afterwards by analyze_dev.py.

Implementation: CSP-LDA is fitted from per-trial covariances and means, which gives exactly the model obtained by
replicating trials (class covariances are weighted means; log-variance features are exact). Channel z-scoring is
omitted because CSP log-variance features are invariant to per-channel scaling (the baselines are recomputed here
with the same code, so all conditions are internally consistent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
A = HERE.parent
for p in ("cmpb_bdea_20260924", "cmpb_floor_gate_20260924", "cmpb_budget_20260924"):
    sys.path.insert(0, str(A / p))
from run_bdea_bnci import load_npz  # noqa: E402
from run_floor_gate import align_subjects  # noqa: E402
from run_lodo import PATHS  # noqa: E402

NAMES = ["cho2017", "lee2019", "bnci2014001", "physionetmi"]
N_T, N_CSP = 200, 8
KS = (5, 10, 20, 30)
DRAWS = 3
W_GRID = ("nat", 0.05, 0.1, 0.2, 0.35, 0.5)
KAPPAS = (5, 10, 20, 40, 80, 160)
OUT = HERE / "dev_results.csv"


def covs(x):
    x = x.astype(np.float64)
    return np.einsum("nct,ndt->ncd", x, x) / x.shape[2], x.mean(axis=2)


class Weighted:
    """CSP(8) + scaler + shrinkage LDA fitted on (source + replicated own) trials, from covariances."""

    def __init__(self, c, m, y, rep):
        cls = {}
        for k in (0, 1):
            s = np.tensordot(rep[y == k], c[y == k], axes=1) / rep[y == k].sum()
            cls[k] = s / np.trace(s)
        ev, evec = np.linalg.eigh(cls[0] + cls[1])
        w = np.diag(1.0 / np.sqrt(ev + 1e-9)) @ evec.T
        ev2, evec2 = np.linalg.eigh(w @ cls[0] @ w.T)
        idx = np.argsort(ev2)
        self.f = evec2[:, np.concatenate([idx[: N_CSP // 2], idx[-N_CSP // 2:]])].T @ w
        feats = self.feats(c, m)
        r = np.repeat(np.arange(len(y)), rep.astype(int))
        self.sc = StandardScaler().fit(feats[r])
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(self.sc.transform(feats[r]), y[r])
        d = self.lda.decision_function(self.sc.transform(feats[r]))
        self.dscale = d.std() + 1e-9

    def feats(self, c, m):
        var = np.einsum("kc,ncd,kd->nk", self.f, c, self.f) - (m @ self.f.T) ** 2
        return np.log(np.maximum(var, 0) + 1e-9)

    def decision(self, c, m):
        return self.lda.decision_function(self.sc.transform(self.feats(c, m))) / self.dscale


def reps_for(n_src, k, w, n_own):
    if w == "nat":
        return 1.0
    return max(1.0, round(w / (1 - w) * n_src / n_own))


def fit_share(sc, sm, sy, oc, om, oy, w):
    rep = np.concatenate([np.ones(len(sy)), np.full(len(oy), reps_for(len(sy), 0, w, len(oy)))])
    return Weighted(np.concatenate([sc, oc]), np.concatenate([sm, om]), np.concatenate([sy, oy]), rep)


def bac(y, d):
    return float(balanced_accuracy_score(y, (d > 0).astype(int)))


def inner_select(sc, sm, sy, oc, om, oy):
    """M3: choose w by stratified 5-fold CV on the k-shot trials (ties -> smaller w; 'nat' is the smallest)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for w in W_GRID:
        pred, true = [], []
        for tr, te in skf.split(oc, oy):
            mdl = fit_share(sc, sm, sy, oc[tr], om[tr], oy[tr], w)
            pred.append(mdl.decision(oc[te], om[te])); true.append(oy[te])
        scores.append(bac(np.concatenate(true), np.concatenate(pred)))
    return W_GRID[int(np.argmax(scores))]


def one_subject(held, s, sc, sm, sy, src_model, x, y):
    rng = np.random.default_rng(1000 * NAMES.index(held) + int(s))
    e_idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum() // 2, replace=False) for c in (0, 1)]))
    p_idx = np.setdiff1d(np.arange(len(y)), e_idx)
    c, m = covs(x)
    ce, me, ye = c[e_idx], m[e_idx], y[e_idx]
    d_src = src_model.decision(ce, me)
    rows = [{"held_out": held, "subject": int(s), "k": 0, "draw": 0, "method": "src_only", "param": "", "bac": bac(ye, d_src)}]
    for k in KS:
        pool = {cl: p_idx[y[p_idx] == cl] for cl in (0, 1)}
        if min(len(v) for v in pool.values()) < k:
            continue
        for draw in range(DRAWS):
            sel = np.concatenate([rng.choice(pool[cl], k, replace=False) for cl in (0, 1)])
            oc, om, oy = c[sel], m[sel], y[sel]
            base = {"held_out": held, "subject": int(s), "k": k, "draw": draw, "param": ""}
            own = Weighted(oc, om, oy, np.ones(len(oy)))
            d_own = own.decision(ce, me)
            rows.append({**base, "method": "own_only", "bac": bac(ye, d_own)})
            for w in W_GRID:
                rows.append({**base, "method": f"w={w}", "bac": bac(ye, fit_share(sc, sm, sy, oc, om, oy, w).decision(ce, me))})
            for kap in KAPPAS:
                a = k / (k + kap)
                rows.append({**base, "method": f"kappa={kap}", "bac": bac(ye, (1 - a) * d_src + a * d_own)})
            w3 = inner_select(sc, sm, sy, oc, om, oy)
            rows.append({**base, "method": "m3", "param": str(w3),
                         "bac": bac(ye, fit_share(sc, sm, sy, oc, om, oy, w3).decision(ce, me))})
    return rows


def main():
    raw = {n: load_npz(PATHS[n]) for n in NAMES}
    shared = [ch for ch in raw[NAMES[0]]["ch_names"] if all(ch in raw[n]["ch_names"] for n in NAMES)]
    data = {}
    for n in NAMES:
        d = raw[n]
        x = d["X"][:, [d["ch_names"].index(ch) for ch in shared], :N_T].astype(np.float32)
        S = d["subjects"].astype(np.int64)
        data[n] = (align_subjects(x, S, "relative", n, "subject")[0], d["y"].astype(np.int64), S)
    print(f"[dev] shared channels = {len(shared)}", flush=True)
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev.held_out, prev.subject))
    for held in NAMES:
        src = [n for n in NAMES if n != held]
        sx = np.concatenate([data[n][0] for n in src]); sy = np.concatenate([data[n][1] for n in src])
        sc, sm = covs(sx); del sx
        src_model = Weighted(sc, sm, sy, np.ones(len(sy)))
        x, y, S = data[held]
        todo = [s for s in np.unique(S) if (held, int(s)) not in done]
        for i in range(0, len(todo), 12):
            batch = todo[i:i + 12]
            res = Parallel(n_jobs=6, backend="loky")(
                delayed(one_subject)(held, s, sc, sm, sy, src_model, x[S == s], y[S == s]) for s in batch)
            df = pd.DataFrame([r for rr in res for r in rr])
            df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"[dev] {held}: {min(i + 12, len(todo))}/{len(todo)} subjects", flush=True)
    print("[dev] done", flush=True)


if __name__ == "__main__":
    main()
