"""Pilot: the same EEG stored in volts (x1) vs microvolts (x1e6); legacy vs relative floor; SubjectEA vs
DatasetEA->SubjectEA; CSP-LDA, LODO4, 21 channels. Exploratory check of the unit explanation."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cmpb_unit_intervention_20260925"))
import run_unit_intervention as ui
data, _ = ui.load_inputs(False)
rows = []
with ui.threadpool_limits(limits=6):
    for label, scale in (("V (x1)", 1.0), ("uV (x1e6)", 1e6)):
        for floor in ("legacy", "relative"):
            for align in ("subject", "dataset_subject"):
                al, floored = {}, []
                for n, d in data.items():
                    al[n], diag = ui.apply_alignment((d["X"] * np.float32(scale)).astype(np.float32), d["subjects"], floor, align, n)
                    floored += [r["floored_or_dropped"] for r in diag if r["stage"] in ("subject", "subject_after_dataset")]
                for h in ui.NAMES:
                    src = [n for n in ui.NAMES if n != h]
                    xs = np.concatenate([al[n] for n in src]); ys = np.concatenate([data[n]["y"] for n in src])
                    mu, sd = ui.norm_stats(xs)
                    csp, sc, lda = ui.fit_csp_lda(ui.ManualCSP, ((xs - mu) / sd).astype(np.float32), ys, 8)
                    p = ui.predict(csp, sc, lda, ((al[h] - mu) / sd).astype(np.float32))
                    rows += ui.subject_rows({"unit": label, "floor": floor, "alignment": align, "held_out": h,
                                             "median_floored_subject": float(np.median(floored))}, data[h]["y"], p, data[h]["subjects"])
                print(f"done {label} {floor} {align}", flush=True)
d = pd.DataFrame(rows); d["bac"] *= 100
d.to_csv(HERE / "pilot_uv_subject_results.csv", index=False)
t = d.groupby(["unit", "floor", "alignment"]).agg(bac=("bac", "mean"), floored=("median_floored_subject", "first")).unstack("alignment")
t[("gap", "D->S - S")] = t[("bac", "dataset_subject")] - t[("bac", "subject")]
pd.set_option("display.width", 200); print(t.round(2).to_string())
