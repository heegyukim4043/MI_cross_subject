"""CPU-only CSP-LDA analyses for the CMPB reframing (relative-floor EA throughout).

Parts
-----
budget : unlabeled target-trial budget for SubjectEA. Per target subject, a fixed
         class-stratified evaluation set E (100 trials) and a calibration pool P (100 trials).
         For n in N_BUDGET, the target reference is estimated from n trials drawn at random
         from P (R_DRAWS draws) and applied to E. References: whitener from E only, from all
         200 trials (legacy transductive protocol), and a no-alignment model.
causal : online SubjectEA. Trial t is aligned with the running mean covariance of trials
         1..t (label-free, as in Wu's EA_online). Lee2019 uses the stored (chronological)
         order within each session; Cho2017 is class-blocked in storage, so random
         permutations are used and flagged as non-chronological.
ch27   : standard_mi channel sensitivity (27 channels present in both datasets) for the
         cross-dataset no-alignment / SubjectEA / DatasetEA->SubjectEA comparison.

Both cross-dataset directions and within-dataset LOSO (both datasets) are evaluated.
Source-side alignment always uses all source trials (labels are available there).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

GATE = Path(__file__).resolve().parents[1] / "cmpb_floor_gate_20260924"
sys.path.insert(0, str(GATE))
from run_floor_gate import (  # noqa: E402
    align_batch,
    align_subjects,
    fit_csp_lda,
    mean_covariance,
    metrics,
    predict,
    whitener_from_cov,
    write_rows,
)

N_BUDGET = (5, 10, 20, 40, 80, 100)
R_DRAWS = 10
BURN_INS = (1, 10, 20)
MODE = "relative"
STANDARD_MI = ["Fz", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2",
               "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "P7", "P3", "Pz", "P4",
               "P8", "Oz"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--n-csp", type=int, default=8)
    p.add_argument("--parts", nargs="+", default=["budget", "causal", "ch27"])
    p.add_argument("--smoke", action="store_true", help="3 subjects per dataset, 2 draws.")
    return p.parse_args()


def load(root: Path, channel_set: str):
    data_dir = root / "MI_test" / "preprocessed_sfreq100"
    raw, names = {}, {}
    for name in ("cho2017", "lee2019"):
        with np.load(data_dir / f"{name}.npz", allow_pickle=True) as d:
            raw[name] = {"X": d["X"].astype(np.float32), "y": d["y"].astype(np.int64),
                         "subjects": d["subjects"].astype(np.int64)}
            names[name] = [str(c) for c in d["ch_names"]]
    common = [c for c in names["cho2017"] if c in names["lee2019"]]
    if channel_set == "standard_mi":
        common = [c for c in STANDARD_MI if c in common]
    for name in raw:
        idx = [names[name].index(c) for c in common]
        raw[name]["X"] = raw[name]["X"][:, idx, :201]
    return raw, common


def subset(data: dict, n_subjects: int) -> dict:
    keep = np.isin(data["subjects"], np.unique(data["subjects"])[:n_subjects])
    return {k: v[keep] for k, v in data.items()}


def norm_stats(x: np.ndarray):
    return x.mean(axis=(0, 2), keepdims=True), x.std(axis=(0, 2), keepdims=True) + 1e-8


def whiten(x_ref: np.ndarray) -> np.ndarray:
    return whitener_from_cov(mean_covariance(x_ref, np.float64), MODE)[0]


def apply_w(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.einsum("cd,ndt->nct", w, x, optimize=True).astype(np.float32)


class Model:
    """CSP-LDA trained on (optionally SubjectEA-aligned) source data plus its channel z-score."""

    def __init__(self, manual_csp, x, y, subjects, n_csp, aligned: bool):
        if aligned:
            x, _ = align_subjects(x, subjects, MODE, "source", "subject")
        self.mean, self.std = norm_stats(x)
        self.csp, self.scaler, self.clf = fit_csp_lda(manual_csp, (x - self.mean) / self.std, y, n_csp)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return predict(self.csp, self.scaler, self.clf, ((x - self.mean) / self.std).astype(np.float32))


def split_eval_pool(y: np.ndarray, rng: np.random.Generator):
    """Class-stratified fixed evaluation set E (half of each class) and calibration pool P."""
    e_idx = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        e_idx.append(rng.choice(idx, size=len(idx) // 2, replace=False))
    e = np.sort(np.concatenate(e_idx))
    return e, np.setdiff1d(np.arange(len(y)), e)


def budget_rows(ctx: dict, model_ea: Model, model_none: Model, x, y, subject, n_draws, rng):
    rows = []
    e, pool = split_eval_pool(y, rng)
    base = {**ctx, "subject": int(subject), "n_eval": len(e), "n_pool": len(pool)}
    rows.append({**base, "condition": "no_alignment", "n_ref": 0, "draw": 0,
                 **metrics(y[e], model_none.predict(x[e]))})
    rows.append({**base, "condition": "transductive_all200", "n_ref": len(y), "draw": 0,
                 **metrics(y[e], model_ea.predict(apply_w(whiten(x), x[e])))})
    rows.append({**base, "condition": "transductive_eval_only", "n_ref": len(e), "draw": 0,
                 **metrics(y[e], model_ea.predict(apply_w(whiten(x[e]), x[e])))})
    for n in N_BUDGET:
        draws = 1 if n >= len(pool) else n_draws
        for r in range(draws):
            ref = pool if n >= len(pool) else rng.choice(pool, size=n, replace=False)
            rows.append({**base, "condition": "pool_budget", "n_ref": int(n), "draw": r,
                         **metrics(y[e], model_ea.predict(apply_w(whiten(x[ref]), x[e])))})
    return rows


def online_align(x: np.ndarray) -> np.ndarray:
    """Align trial t with the running mean covariance of trials 1..t (inclusive, label-free)."""
    out = np.empty_like(x, dtype=np.float32)
    acc = np.zeros((x.shape[1], x.shape[1]))
    for t in range(len(x)):
        xt = x[t].astype(np.float64)
        acc += xt @ xt.T
        w = whitener_from_cov(acc / (t + 1), MODE)[0]
        out[t] = (w @ xt).astype(np.float32)
    return out


def causal_rows(ctx: dict, model_ea: Model, x, y, subject, dataset, rng, n_perm):
    """Per-trial correctness under online vs batch references; aggregated by burn-in."""
    rows = []
    if dataset == "lee2019":
        orders = [("chronological", 0, np.arange(len(y)))]
        sessions = [np.arange(0, len(y) // 2), np.arange(len(y) // 2, len(y))]
    else:
        orders = [("random_perm", p, rng.permutation(len(y))) for p in range(n_perm)]
        sessions = None
    for order_name, perm_id, order in orders:
        xo, yo = x[order], y[order]
        segs = sessions if sessions is not None else [np.arange(len(yo))]
        preds = {k: np.empty(len(yo), dtype=np.int64) for k in
                 ("online_reset", "online_carry", "batch_session", "batch_subject")}
        preds["batch_subject"][:] = model_ea.predict(apply_w(whiten(xo), xo))
        preds["online_carry"][:] = model_ea.predict(online_align(xo))
        pos_in_session = np.empty(len(yo), dtype=np.int64)
        for seg in segs:
            preds["online_reset"][seg] = model_ea.predict(online_align(xo[seg]))
            preds["batch_session"][seg] = model_ea.predict(apply_w(whiten(xo[seg]), xo[seg]))
            pos_in_session[seg] = np.arange(1, len(seg) + 1)
        for b in BURN_INS:
            keep = pos_in_session >= b
            for cond, pr in preds.items():
                rows.append({**ctx, "subject": int(subject), "order": order_name, "perm": perm_id,
                             "burn_in": b, "condition": cond, "n_scored": int(keep.sum()),
                             **metrics(yo[keep], pr[keep])})
    return rows


def evaluate_targets(ctx, manual_csp, n_csp, src, tgt, target_ids, parts, rng, n_draws, n_perm,
                     dataset_name, budget_out, causal_out):
    model_ea = Model(manual_csp, src["X"], src["y"], src["subjects"], n_csp, aligned=True)
    model_none = Model(manual_csp, src["X"], src["y"], src["subjects"], n_csp, aligned=False) \
        if "budget" in parts else None
    for subject in target_ids:
        m = tgt["subjects"] == subject
        x, y = tgt["X"][m], tgt["y"][m]
        if "budget" in parts:
            budget_out.extend(budget_rows(ctx, model_ea, model_none, x, y, subject, n_draws, rng))
        if "causal" in parts:
            causal_out.extend(causal_rows(ctx, model_ea, x, y, subject, dataset_name, rng, n_perm))


def run_budget_causal(raw, manual_csp, n_csp, parts, out: Path, smoke: bool):
    rng = np.random.default_rng(20260924)
    n_draws, n_perm = (2, 2) if smoke else (R_DRAWS, 5)
    data = {k: subset(v, 3) if smoke else v for k, v in raw.items()}
    budget, causal = [], []
    for source, target in (("cho2017", "lee2019"), ("lee2019", "cho2017")):
        t0 = time.perf_counter()
        ctx = {"scope": "cross", "source": source, "target": target}
        evaluate_targets(ctx, manual_csp, n_csp, data[source], data[target],
                         np.unique(data[target]["subjects"]), parts, rng, n_draws, n_perm, target,
                         budget, causal)
        write_rows(out / "budget_subject_results.csv", budget)
        write_rows(out / "causal_subject_results.csv", causal)
        print(f"[cross] {source}->{target} done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    for dataset in ("cho2017", "lee2019"):
        d = data[dataset]
        t0 = time.perf_counter()
        for subject in np.unique(d["subjects"]):
            tr = d["subjects"] != subject
            src = {k: v[tr] for k, v in d.items()}
            tgt = {k: v[~tr] for k, v in d.items()}
            ctx = {"scope": "loso", "source": dataset, "target": dataset}
            evaluate_targets(ctx, manual_csp, n_csp, src, tgt, [subject], parts, rng, n_draws,
                             n_perm, dataset, budget, causal)
        write_rows(out / "budget_subject_results.csv", budget)
        write_rows(out / "causal_subject_results.csv", causal)
        print(f"[loso] {dataset} done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


def run_ch27(root, manual_csp, n_csp, out: Path, smoke: bool):
    raw, channels = load(root, "standard_mi")
    data = {k: subset(v, 3) if smoke else v for k, v in raw.items()}
    rows = []
    for source, target in (("cho2017", "lee2019"), ("lee2019", "cho2017")):
        for mode in ("legacy", "relative"):
            for alignment in ("none", "subject", "dataset_subject"):
                if alignment == "none" and mode == "legacy":
                    continue
                xs, xt = data[source]["X"], data[target]["X"]
                if alignment == "dataset_subject":
                    xs, _ = align_batch(xs, mode)
                    xt, _ = align_batch(xt, mode)
                if alignment != "none":
                    xs, _ = align_subjects(xs, data[source]["subjects"], mode, source, "subject")
                    xt, _ = align_subjects(xt, data[target]["subjects"], mode, target, "subject")
                mu, sd = norm_stats(xs)
                csp, sc, clf = fit_csp_lda(manual_csp, ((xs - mu) / sd).astype(np.float32),
                                           data[source]["y"], n_csp)
                xt = ((xt - mu) / sd).astype(np.float32)
                for subject in np.unique(data[target]["subjects"]):
                    m = data[target]["subjects"] == subject
                    rows.append({"source": source, "target": target, "n_channels": len(channels),
                                 "floor_mode": "none" if alignment == "none" else mode,
                                 "alignment": alignment, "subject": int(subject),
                                 **metrics(data[target]["y"][m], predict(csp, sc, clf, xt[m]))})
                write_rows(out / "ch27_subject_results.csv", rows)
                print(f"[ch27] {source}->{target} {mode} {alignment} done", flush=True)


def main() -> int:
    args = parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "MI_test" / "MI_loso_project"))
    from mrfbcsp_loso import ManualCSP
    with threadpool_limits(limits=args.threads):
        if {"budget", "causal"} & set(args.parts):
            raw, _ = load(root, "common")
            run_budget_causal(raw, ManualCSP, args.n_csp, args.parts, out, args.smoke)
        if "ch27" in args.parts:
            run_ch27(root, ManualCSP, args.n_csp, out, args.smoke)
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
