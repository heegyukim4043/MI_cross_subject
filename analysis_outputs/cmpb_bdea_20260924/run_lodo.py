"""Leave-one-dataset-out (LODO) test of the alignment conclusions (CSP-LDA, batch transductive EA).

For each held-out dataset, CSP-LDA is trained on the union of the remaining datasets and evaluated
per subject on the held-out one, on the channels shared by all datasets in the run.
Conditions: none; SubjectEA and DatasetEA->SubjectEA under the legacy absolute floor and the
relative (scale-invariant) floor. DatasetEA is estimated separately for every dataset (each source
dataset and the held-out target dataset).
Usage: python run_lodo.py cho2017 lee2019 bnci2014001 [physionetmi]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent / "cmpb_floor_gate_20260924", HERE.parent / "cmpb_budget_20260924", HERE, ROOT / "MI_test" / "MI_loso_project"):
    sys.path.insert(0, str(p))
from run_floor_gate import apply_alignment, fit_csp_lda, metrics, predict, write_rows  # noqa: E402
from run_budget_causal import norm_stats  # noqa: E402
from run_bdea_bnci import load_npz  # noqa: E402
from mrfbcsp_loso import ManualCSP  # noqa: E402

PATHS = {"cho2017": ROOT / "MI_test/preprocessed_sfreq100/cho2017.npz",
         "lee2019": ROOT / "MI_test/preprocessed_sfreq100/lee2019.npz",
         "bnci2014001": HERE / "bnci2014001.npz",
         "physionetmi": HERE / "physionetmi.npz"}
CONDS = [("none", "none"), ("legacy", "subject"), ("legacy", "dataset_subject"),
         ("relative", "subject"), ("relative", "dataset_subject")]


def main() -> int:
    names = sys.argv[1:] or ["cho2017", "lee2019", "bnci2014001"]
    data = {n: load_npz(PATHS[n]) for n in names}
    shared = [c for c in data[names[0]]["ch_names"] if all(c in data[n]["ch_names"] for n in names)]
    n_t = min(d["X"].shape[2] for d in data.values())
    sub = {}
    for n, d in data.items():
        idx = [d["ch_names"].index(c) for c in shared]
        sub[n] = {"X": d["X"][:, idx, :n_t].astype(np.float32), "y": d["y"].astype(np.int64),
                  "subjects": d["subjects"].astype(np.int64)}
    print(f"LODO over {names}: {len(shared)} shared channels, {n_t} samples", flush=True)
    rows = []
    tag = "_".join(names)
    with threadpool_limits(limits=8):
        for mode, how in CONDS:
            aligned = {n: (d["X"] if how == "none" else apply_alignment(d["X"], d["subjects"], mode, how, n)[0])
                       for n, d in sub.items()}
            for held in names:
                t0 = time.perf_counter()
                src = [n for n in names if n != held]
                xs = np.concatenate([aligned[n] for n in src]); ys = np.concatenate([sub[n]["y"] for n in src])
                mu, sd = norm_stats(xs)
                csp, sc, clf = fit_csp_lda(ManualCSP, ((xs - mu) / sd).astype(np.float32), ys, 8)
                xt = ((aligned[held] - mu) / sd).astype(np.float32)
                for s in np.unique(sub[held]["subjects"]):
                    m = sub[held]["subjects"] == s
                    rows.append({"datasets": tag, "held_out": held, "sources": "+".join(src), "n_channels": len(shared),
                                 "floor": mode, "alignment": how, "subject": int(s),
                                 **metrics(sub[held]["y"][m], predict(csp, sc, clf, xt[m]))})
                print(f"  {mode}:{how} held-out {held} done ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    write_rows(HERE / f"lodo_{tag}_subject_results.csv", rows)
    import pandas as pd
    df = pd.DataFrame(rows); df["cond"] = df.floor + ":" + df.alignment
    print((df.groupby(["held_out", "cond"]).bac.mean().unstack() * 100).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
