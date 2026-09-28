"""PhysionetMI LOSO (109 subjects, all 64 channels): none vs SubjectEA under legacy and relative floors,
plus the per-subject number of covariance eigenvalues below the legacy absolute floor (1e-8). CSP-LDA."""
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
for p in (HERE.parent / "cmpb_floor_gate_20260924", HERE.parent / "cmpb_budget_20260924", HERE, ROOT / "MI_test" / "MI_loso_project"):
    sys.path.insert(0, str(p))
from run_floor_gate import apply_alignment, fit_csp_lda, mean_covariance, metrics, predict, write_rows
from run_budget_causal import norm_stats
from run_bdea_bnci import load_npz
from mrfbcsp_loso import ManualCSP
d = load_npz(HERE / "physionetmi.npz"); X, y, S = d["X"].astype(np.float32), d["y"].astype(np.int64), d["subjects"].astype(np.int64)
floored = [int((np.linalg.eigvalsh(mean_covariance(X[S == s], np.float32)) < 1e-8).sum()) for s in np.unique(S)]
print(f"eigenvalues below 1e-8 per subject (of {X.shape[1]}): min={min(floored)} median={int(np.median(floored))} max={max(floored)}", flush=True)
rows = []
with threadpool_limits(limits=8):
    for mode, how in (("none", "none"), ("legacy", "subject"), ("relative", "subject")):
        xa = X if how == "none" else apply_alignment(X, S, mode, how, "physionetmi")[0]
        for s in np.unique(S):
            tr = S != s; mu, sd = norm_stats(xa[tr])
            csp, sc, clf = fit_csp_lda(ManualCSP, ((xa[tr] - mu) / sd).astype(np.float32), y[tr], 8)
            rows.append({"floor": mode, "alignment": how, "subject": int(s), "n_floored_legacy": floored[list(np.unique(S)).index(s)],
                         **metrics(y[~tr], predict(csp, sc, clf, ((xa[~tr] - mu) / sd).astype(np.float32)))})
        print(f"  {mode}:{how} BAC={np.mean([r['bac'] for r in rows if r['floor']==mode])*100:.2f}%", flush=True)
write_rows(HERE / "physionet_loso_results.csv", rows)
