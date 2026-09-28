"""CPU prototype of Budget- and Drift-aware online Euclidean Alignment (BD-EA), CSP-LDA.

All target-side variants are label-free and causal: trial t is aligned with a reference built
from trials 1..t only (inclusive, as in Wu's EA_online). The source model (SubjectEA on source
subjects, relative floor) is trained once per fold and never retrained; only the target
reference changes.

Target stream: Lee2019 in stored order (session 1 trials 0-99, then session 2 trials 100-199;
the boundary is NOT given to the causal methods except the oracle). Cho2017 is class-blocked in
storage, so it is streamed in random permutations (cold-start evaluation only; no drift).

Variants (parameters fixed before running):
  batch_subject / batch_session      transductive references (upper references)
  online_plain                       cumulative mean covariance
  online_oracle_reset                cumulative, reset at the true session boundary (Lee only)
  shrink_{I,src,site}_k{K}           component 2: R_t = (1-a_t) Rhat_t + a_t * (tr(Rhat_t)/C) * P,
                                     a_t = K / (K + n_t); P trace-normalised prior shape:
                                     I (identity), src (source-training mean), site (other target
                                     subjects of the same dataset, evaluated subject excluded)
  ewma_{lam}                         exponential forgetting R_t = lam R_{t-1} + (1-lam) C_t
  drift_reset                        component 3: running reference since last reset; window
                                     W_t = mean of last w trials; reset when d_R(R, W_t) exceeds
                                     mean + z*sd of past distances since the last reset (after m
                                     trials); after a reset the reference restarts from W_t.
  bdea                               component 2 + 3: drift_reset whose post-reset estimate is
                                     shrunk toward the pre-reset reference shape (k = K), and whose
                                     initial estimate is shrunk toward the site/source prior.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BUDGET = HERE.parent / "cmpb_budget_20260924"
sys.path.insert(0, str(HERE.parent / "cmpb_floor_gate_20260924"))
sys.path.insert(0, str(BUDGET))
from run_floor_gate import metrics, write_rows, whitener_from_cov  # noqa: E402
from run_budget_causal import Model, load, subset  # noqa: E402

K_SHRINK = (5, 10, 20)
EWMA = (0.95, 0.98, 0.99)
DRIFT = {"w": 10, "z": 3.0, "m": 20}
K_BDEA = 10
WINDOWS = {"all": (1, None), "first10": (1, 10), "first20": (1, 20)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=HERE.parents[1])
    p.add_argument("--out", type=Path, default=HERE)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--n-perm", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def trial_covs(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    return np.einsum("nct,ndt->ncd", x64, x64)


def shape(r: np.ndarray) -> np.ndarray:
    """Trace-normalised covariance shape (trace = C)."""
    return r * (len(r) / np.trace(r))


def riem_dist(a: np.ndarray, b: np.ndarray) -> float:
    w = eigh(b, a, eigvals_only=True)
    return float(np.sqrt(np.sum(np.log(np.clip(w, 1e-300, None)) ** 2)))


def shrink(rhat: np.ndarray, n: int, prior: np.ndarray, k: int) -> np.ndarray:
    a = k / (k + n)
    return (1 - a) * rhat + a * (np.trace(rhat) / len(rhat)) * prior


def references(c: np.ndarray, variant: str, priors: dict, boundary: int | None):
    """Yield the causal reference covariance for each trial t (uses trials 0..t)."""
    n_trials, ch = len(c), c.shape[1]
    out = np.empty_like(c)
    if variant == "online_plain" or variant == "online_oracle_reset" or variant.startswith("shrink_"):
        s, n = np.zeros((ch, ch)), 0
        for t in range(n_trials):
            if variant == "online_oracle_reset" and boundary is not None and t == boundary:
                s, n = np.zeros((ch, ch)), 0
            s += c[t]; n += 1
            rhat = s / n
            if variant.startswith("shrink_"):
                _, pname, kname = variant.split("_")
                rhat = shrink(rhat, n, priors[pname], int(kname[1:]))
            out[t] = rhat
        return out, []
    if variant.startswith("ewma_"):
        lam = float(variant.split("_")[1])
        r = None
        for t in range(n_trials):
            r = c[t].copy() if r is None else lam * r + (1 - lam) * c[t]
            out[t] = r
        return out, []
    if variant in ("drift_reset", "bdea"):
        w, z, m = DRIFT["w"], DRIFT["z"], DRIFT["m"]
        s, n, dists, resets = np.zeros((ch, ch)), 0, [], []
        prior_shape = priors.get("site", priors["src"]) if variant == "bdea" else None
        for t in range(n_trials):
            s += c[t]; n += 1
            rhat = s / n
            if n >= m and t + 1 >= w:
                win = c[t - w + 1:t + 1].mean(axis=0)
                d = riem_dist(rhat, win)
                if len(dists) >= 5 and d > np.mean(dists) + z * np.std(dists):
                    resets.append(t)
                    if variant == "bdea":
                        prior_shape = shape(rhat)          # previous reference becomes the prior
                    s, n, dists = win * w, w, []
                    rhat = s / n
                else:
                    dists.append(d)
            if variant == "bdea":
                rhat = shrink(rhat, n, prior_shape, K_BDEA)
            out[t] = rhat
        return out, resets
    raise ValueError(variant)


def align_stream(x: np.ndarray, refs: np.ndarray) -> np.ndarray:
    z = np.empty_like(x, dtype=np.float32)
    for t in range(len(x)):
        z[t] = (whitener_from_cov(refs[t], "relative")[0] @ x[t].astype(np.float64)).astype(np.float32)
    return z


def batch_align(x: np.ndarray, idx_groups) -> np.ndarray:
    z = np.empty_like(x, dtype=np.float32)
    c = trial_covs(x)
    for g in idx_groups:
        w = whitener_from_cov(c[g].mean(axis=0), "relative")[0]
        z[g] = np.einsum("cd,ndt->nct", w, x[g].astype(np.float64)).astype(np.float32)
    return z


def score(ctx, y, pred, pos_session, extra, rows):
    for wname, (lo, hi) in WINDOWS.items():
        keep = np.ones(len(y), bool) if hi is None else (np.arange(1, len(y) + 1) <= hi)
        rows.append({**ctx, **extra, "window": wname, "n": int(keep.sum()), **metrics(y[keep], pred[keep])})
    if pos_session is not None:  # first 20 trials after the (hidden) session boundary
        keep = (pos_session[1] >= 1) & (pos_session[1] <= 20)
        rows.append({**ctx, **extra, "window": "s2_first20", "n": int(keep.sum()), **metrics(y[keep], pred[keep])})


def evaluate_subject(ctx, model, x, y, dataset, priors, rng, n_perm, rows, resets_log):
    chrono = dataset == "lee2019"
    orders = [("chronological", 0, np.arange(len(y)))] if chrono else \
             [("random_perm", p, rng.permutation(len(y))) for p in range(n_perm)]
    variants = ["online_plain"] + [f"shrink_{p}_k{k}" for p in priors for k in K_SHRINK] + \
               [f"ewma_{l}" for l in EWMA] + ["drift_reset", "bdea"] + (["online_oracle_reset"] if chrono else [])
    for oname, pid, order in orders:
        xo, yo = x[order], y[order]
        c = trial_covs(xo)
        boundary = len(yo) // 2 if chrono else None
        pos = None
        if chrono:
            idx = np.arange(len(yo))
            pos = (idx < boundary, np.where(idx >= boundary, idx - boundary + 1, 0))
        base = {**ctx, "order": oname, "perm": pid}
        groups_subject = [np.arange(len(yo))]
        groups_session = [np.arange(boundary), np.arange(boundary, len(yo))] if chrono else groups_subject
        for vname, groups in (("batch_subject", groups_subject), ("batch_session", groups_session)):
            score(base, yo, model.predict(batch_align(xo, groups)), pos, {"variant": vname}, rows)
        for v in variants:
            refs, resets = references(c, v, priors, boundary)
            score(base, yo, model.predict(align_stream(xo, refs)), pos, {"variant": v}, rows)
            if v in ("drift_reset", "bdea"):
                resets_log.append({**base, "variant": v, "n_resets": len(resets),
                                   "resets": " ".join(map(str, resets)),
                                   "first_reset_after_boundary": next((r for r in resets if boundary and r >= boundary), -1)})


def run(raw, manual_csp, out: Path, n_perm: int, smoke: bool):
    rng = np.random.default_rng(20260924)
    data = {k: subset(v, 3) if smoke else v for k, v in raw.items()}
    rows, resets = [], []
    folds = [("cross", "cho2017", "lee2019"), ("cross", "lee2019", "cho2017"),
             ("loso", "cho2017", "cho2017"), ("loso", "lee2019", "lee2019")]
    for scope, source, target in folds:
        t0 = time.perf_counter()
        tgt_all = data[target]
        if scope == "cross":
            model = Model(manual_csp, data[source]["X"], data[source]["y"], data[source]["subjects"], 8, aligned=True)
            src_cov = trial_covs(data[source]["X"]).mean(axis=0)
        tgt_covs = {s: trial_covs(tgt_all["X"][tgt_all["subjects"] == s]).mean(axis=0)
                    for s in np.unique(tgt_all["subjects"])}
        for s in np.unique(tgt_all["subjects"]):
            if scope == "loso":
                tr = tgt_all["subjects"] != s
                model = Model(manual_csp, tgt_all["X"][tr], tgt_all["y"][tr], tgt_all["subjects"][tr], 8, aligned=True)
                src_cov = trial_covs(tgt_all["X"][tr]).mean(axis=0)
            priors = {"I": np.eye(src_cov.shape[0]), "src": shape(src_cov)}
            if scope == "cross":  # site prior: other subjects of the target dataset (evaluated subject excluded)
                priors["site"] = shape(np.mean([shape(v) for k, v in tgt_covs.items() if k != s], axis=0))
            m = tgt_all["subjects"] == s
            evaluate_subject({"scope": scope, "source": source, "target": target, "subject": int(s)},
                             model, tgt_all["X"][m], tgt_all["y"][m], target, priors, rng, n_perm, rows, resets)
        write_rows(out / "bdea_subject_results.csv", rows)
        write_rows(out / "bdea_resets.csv", resets)
        print(f"[bdea] {scope} {source}->{target} done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


def main() -> int:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root.resolve() / "MI_test" / "MI_loso_project"))
    from mrfbcsp_loso import ManualCSP
    raw, _ = load(args.root.resolve(), "common")
    with threadpool_limits(limits=args.threads):
        run(raw, ManualCSP, out, 2 if args.smoke else args.n_perm, args.smoke)
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
