"""BNCI2014-001 follow-ups (CSP-LDA, relative-floor EA, LOSO over 9 subjects).

budget  : unlabeled target-trial budget, same design as the Cho/Lee analysis (fixed class-stratified
          evaluation half, reference from n in {5,10,20,40,80,all-pool} random pool trials, 10 draws).
causal  : online SubjectEA (running mean of trials 1..t) reset per session vs batch session/subject,
          chronological order, burn-in 1/10/20.
session : per-subject Riemannian distance between the two session reference covariances, related to the
          per-subject SessionEA gain (batch session minus batch subject BAC). Computed for BNCI (LOSO)
          and Lee2019 (LOSO, sessions = trials 0-99 / 100-199) for comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent / "cmpb_floor_gate_20260924", HERE.parent / "cmpb_budget_20260924", HERE, ROOT / "MI_test" / "MI_loso_project"):
    sys.path.insert(0, str(p))
from run_floor_gate import metrics, write_rows  # noqa: E402
from run_budget_causal import Model, apply_w, budget_rows, online_align, whiten  # noqa: E402
from run_bdea import riem_dist, trial_covs  # noqa: E402
from run_bdea_bnci import load_npz, pick  # noqa: E402
from mrfbcsp_loso import ManualCSP  # noqa: E402


def session_gain_rows(dataset, x, y, sess, model, subject, rows):
    c = trial_covs(x)
    refs = [c[sess == k].mean(axis=0) for k in np.unique(sess)]
    dist = riem_dist(refs[0], refs[1])
    whole = model.predict(apply_w(whiten(x), x))
    per = np.empty_like(y)
    for k in np.unique(sess):
        m = sess == k
        per[m] = model.predict(apply_w(whiten(x[m]), x[m]))
    rows.append({"dataset": dataset, "subject": int(subject), "session_distance": dist,
                 "bac_subject": metrics(y, whole)["bac"], "bac_session": metrics(y, per)["bac"]})


def causal_rows(x, y, sess, model, subject, rows):
    preds = {"batch_subject": model.predict(apply_w(whiten(x), x)), "online_reset": np.empty_like(y),
             "batch_session": np.empty_like(y)}
    pos = np.empty(len(y), dtype=int)
    for k in np.unique(sess):
        m = np.flatnonzero(sess == k)
        preds["online_reset"][m] = model.predict(online_align(x[m]))
        preds["batch_session"][m] = model.predict(apply_w(whiten(x[m]), x[m]))
        pos[m] = np.arange(1, len(m) + 1)
    for b in (1, 10, 20):
        keep = pos >= b
        for cond, p in preds.items():
            rows.append({"subject": int(subject), "burn_in": b, "condition": cond, **metrics(y[keep], p[keep])})


def main() -> int:
    rng = np.random.default_rng(20260924)
    bnci = load_npz(HERE / "bnci2014001.npz")
    b = pick(bnci, bnci["ch_names"], bnci["X"].shape[2])
    budget, causal, gain = [], [], []
    with threadpool_limits(limits=8):
        for s in np.unique(b["subjects"]):
            tr = b["subjects"] != s
            model = Model(ManualCSP, b["X"][tr], b["y"][tr], b["subjects"][tr], 8, aligned=True)
            none = Model(ManualCSP, b["X"][tr], b["y"][tr], b["subjects"][tr], 8, aligned=False)
            m = ~tr
            x, y, sess = b["X"][m], b["y"][m], b["sessions"][m]
            budget += budget_rows({"scope": "loso", "source": "bnci2014001", "target": "bnci2014001"}, model, none, x, y, s, 10, rng)
            causal_rows(x, y, sess, model, s, causal)
            session_gain_rows("bnci2014001", x, y, sess, model, s, gain)
        lee = load_npz(ROOT / "MI_test" / "preprocessed_sfreq100" / "lee2019.npz")
        with np.load(ROOT / "MI_test" / "preprocessed_sfreq100" / "cho2017.npz", allow_pickle=True) as c:
            common = [n for n in [str(v) for v in c["ch_names"]] if n in lee["ch_names"]]
        L = pick(lee, common, 201)
        for s in np.unique(L["subjects"]):
            tr = L["subjects"] != s
            model = Model(ManualCSP, L["X"][tr], L["y"][tr], L["subjects"][tr], 8, aligned=True)
            x, y = L["X"][~tr], L["y"][~tr]
            sess = (np.arange(len(y)) >= len(y) // 2).astype(int)
            session_gain_rows("lee2019", x, y, sess, model, s, gain)
    write_rows(HERE / "bnci_budget_results.csv", budget)
    write_rows(HERE / "bnci_causal_results.csv", causal)
    write_rows(HERE / "session_distance_gain.csv", gain)
    bud = pd.DataFrame(budget).groupby(["condition", "n_ref", "subject"]).bac.mean().groupby(["condition", "n_ref"]).mean() * 100
    print("BNCI budget (BAC %):\n", bud.round(2).to_string())
    cau = pd.DataFrame(causal).groupby(["burn_in", "condition"]).bac.mean().unstack() * 100
    print("BNCI causal (BAC %):\n", cau.round(2).to_string())
    g = pd.DataFrame(gain)
    g["gain_pp"] = (g.bac_session - g.bac_subject) * 100
    for ds, gg in g.groupby("dataset"):
        rho, p = spearmanr(gg.session_distance, gg.gain_pp)
        print(f"{ds}: n={len(gg)} session distance median={gg.session_distance.median():.3f} "
              f"(IQR {gg.session_distance.quantile(.25):.3f}-{gg.session_distance.quantile(.75):.3f}); "
              f"mean SessionEA gain={gg.gain_pp.mean():+.2f} pp; Spearman rho={rho:+.3f} (p={p:.3g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
