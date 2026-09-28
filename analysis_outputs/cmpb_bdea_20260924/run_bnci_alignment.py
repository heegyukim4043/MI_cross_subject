"""External check on BNCI2014-001: no alignment vs SubjectEA vs DatasetEA->SubjectEA, legacy vs relative floor.
Cross-dataset Lee2019/Cho2017 -> BNCI2014-001 on shared channels, plus BNCI LOSO. CSP-LDA, batch (transductive) EA."""
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
for p in (HERE.parent / "cmpb_floor_gate_20260924", HERE.parent / "cmpb_budget_20260924", HERE, ROOT / "MI_test" / "MI_loso_project"):
    sys.path.insert(0, str(p))
from run_floor_gate import apply_alignment, fit_csp_lda, metrics, predict, write_rows
from run_budget_causal import norm_stats
from run_bdea_bnci import load_npz, pick
from mrfbcsp_loso import ManualCSP
CONDS = [("none", "none"), ("legacy", "subject"), ("legacy", "dataset_subject"), ("relative", "subject"), ("relative", "dataset_subject")]
def align(d, name, mode, how):
    return d["X"] if how == "none" else apply_alignment(d["X"], d["subjects"], mode, how, name)[0]
def fit_eval(xs, ys, xt, yt, st, ctx, rows):
    mu, sd = norm_stats(xs); csp, sc, clf = fit_csp_lda(ManualCSP, ((xs - mu) / sd).astype(np.float32), ys, 8)
    xt = ((xt - mu) / sd).astype(np.float32)
    for s in np.unique(st):
        m = st == s; rows.append({**ctx, "subject": int(s), **metrics(yt[m], predict(csp, sc, clf, xt[m]))})
rows = []
bnci = load_npz(HERE / "bnci2014001.npz"); T = bnci["X"].shape[2]
with threadpool_limits(limits=8):
    for src_name in ("lee2019", "cho2017"):
        src = load_npz(ROOT / "MI_test" / "preprocessed_sfreq100" / f"{src_name}.npz")
        shared = [c for c in bnci["ch_names"] if c in src["ch_names"]]
        s, t = pick(src, shared, T), pick(bnci, shared, T)
        for mode, how in CONDS:
            fit_eval(align(s, src_name, mode, how), s["y"], align(t, "bnci", mode, how), t["y"], t["subjects"],
                     {"scope": "cross", "source": src_name, "floor": mode, "alignment": how}, rows)
    b = pick(bnci, bnci["ch_names"], T)
    for mode, how in [c for c in CONDS if c[1] != "dataset_subject"]:
        xa = align(b, "bnci", mode, how)
        for s in np.unique(b["subjects"]):
            tr = b["subjects"] != s
            fit_eval(xa[tr], b["y"][tr], xa[~tr], b["y"][~tr], b["subjects"][~tr],
                     {"scope": "loso", "source": "bnci2014001", "floor": mode, "alignment": how}, rows)
write_rows(HERE / "bnci_alignment_results.csv", rows); print("done")
