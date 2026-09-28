"""RQ4: target-access tiers in the common four-dataset LODO design (CSP-LDA, relative floor, 21 channels; exploratory).

For each held-out dataset, the source model is trained on the other three (SubjectEA on every source subject).
Per target subject (fixed class-stratified evaluation half E, calibration pool P = the other half):
  budget (reuses run_budget_causal.budget_rows): no alignment; SubjectEA reference from n unlabeled pool trials
         (n in 5, 10, 20, 40, 80, 100, capped at the pool size; 10 draws); E-only reference; all trials (transductive).
  causal: online SubjectEA (running mean covariance of trials 1..t) versus batch session/subject references,
         in stored chronological order within sessions for Lee2019 (two halves), BNCI2014-001 (two sessions) and
         PhysionetMI (one session, runs in order); Cho2017 is class-blocked in storage, so 5 random permutations.
The limited-labeled tier is taken from method A's development run (same LODO4 design, same subjects).
Outputs: tiers_budget_results.csv, tiers_causal_results.csv (resumable per held-out dataset).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
A = HERE.parent
for p in ("cmpb_bdea_20260924", "cmpb_floor_gate_20260924", "cmpb_budget_20260924"):
    sys.path.insert(0, str(A / p))
sys.path.insert(0, str(A.parent / "MI_test" / "MI_loso_project"))
import run_budget_causal as rb  # noqa: E402
from mrfbcsp_loso import ManualCSP  # noqa: E402
from run_bdea_bnci import load_npz  # noqa: E402
from run_floor_gate import metrics, write_rows  # noqa: E402
from run_lodo import PATHS  # noqa: E402

NAMES = ["cho2017", "lee2019", "bnci2014001", "physionetmi"]
BUDGET_OUT, CAUSAL_OUT = HERE / "tiers_budget_results.csv", HERE / "tiers_causal_results.csv"


def load():
    raw = {n: load_npz(PATHS[n]) for n in NAMES}
    shared = [c for c in raw[NAMES[0]]["ch_names"] if all(c in raw[n]["ch_names"] for n in NAMES)]
    data = {}
    for n in NAMES:
        d = raw[n]
        s = d["subjects"].astype(np.int64)
        if "sessions" in d:
            sess = d["sessions"].astype(np.int64)
        elif n == "lee2019":  # stored as two consecutive sessions per subject
            sess = np.zeros(len(s), dtype=np.int64)
            for sub in np.unique(s):
                idx = np.flatnonzero(s == sub); sess[idx[len(idx) // 2:]] = 1
        else:
            sess = np.zeros(len(s), dtype=np.int64)
        data[n] = {"X": d["X"][:, [d["ch_names"].index(c) for c in shared], :200].astype(np.float32),
                   "y": d["y"].astype(np.int64), "subjects": s, "sessions": sess}
    return data, shared


def causal_rows(ctx, model, x, y, sess, subject, chronological, rng, n_perm=5):
    rows = []
    orders = [("chronological", 0, np.arange(len(y)))] if chronological else \
        [("random_perm", p, rng.permutation(len(y))) for p in range(n_perm)]
    for name, pid, order in orders:
        xo, yo, so = x[order], y[order], sess[order]
        segs = [np.flatnonzero(so == k) for k in np.unique(so)] if chronological else [np.arange(len(yo))]
        preds = {k: np.empty(len(yo), dtype=np.int64) for k in ("online_reset", "batch_session", "batch_subject")}
        preds["batch_subject"][:] = model.predict(rb.apply_w(rb.whiten(xo), xo))
        pos = np.empty(len(yo), dtype=np.int64)
        for seg in segs:
            preds["online_reset"][seg] = model.predict(rb.online_align(xo[seg]))
            preds["batch_session"][seg] = model.predict(rb.apply_w(rb.whiten(xo[seg]), xo[seg]))
            pos[seg] = np.arange(1, len(seg) + 1)
        for b in rb.BURN_INS:
            keep = pos >= b
            for cond, pr in preds.items():
                rows.append({**ctx, "subject": int(subject), "order": name, "perm": pid, "burn_in": b,
                             "condition": cond, "n_scored": int(keep.sum()), **metrics(yo[keep], pr[keep])})
    return rows


def main():
    data, shared = load()
    print(f"[tiers] {len(shared)} channels", flush=True)
    done = set(pd.read_csv(BUDGET_OUT).target.unique()) if BUDGET_OUT.exists() else set()
    budget = pd.read_csv(BUDGET_OUT).to_dict("records") if BUDGET_OUT.exists() else []
    causal = pd.read_csv(CAUSAL_OUT).to_dict("records") if CAUSAL_OUT.exists() else []
    with threadpool_limits(limits=4):
        for held in NAMES:
            if held in done:
                print(f"[skip] {held}", flush=True); continue
            t0 = time.perf_counter()
            rng = np.random.default_rng(20260928 + NAMES.index(held))
            src = [n for n in NAMES if n != held]
            off, subj = 0, []
            for n in src:
                _, enc = np.unique(data[n]["subjects"], return_inverse=True); subj.append(enc + off); off += enc.max() + 1
            xs = np.concatenate([data[n]["X"] for n in src]); ys = np.concatenate([data[n]["y"] for n in src])
            model_ea = rb.Model(ManualCSP, xs, ys, np.concatenate(subj), 8, aligned=True)
            model_none = rb.Model(ManualCSP, xs, ys, np.concatenate(subj), 8, aligned=False)
            del xs
            t = data[held]
            ctx = {"scope": "lodo4", "source": "+".join(src), "target": held}
            for s in np.unique(t["subjects"]):
                m = t["subjects"] == s
                budget += rb.budget_rows(ctx, model_ea, model_none, t["X"][m], t["y"][m], s, rb.R_DRAWS, rng)
                causal += causal_rows(ctx, model_ea, t["X"][m], t["y"][m], t["sessions"][m], s, held != "cho2017", rng)
            write_rows(BUDGET_OUT, budget); write_rows(CAUSAL_OUT, causal)
            print(f"[tiers] {held} done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    print("[tiers] all done", flush=True)


if __name__ == "__main__":
    main()
