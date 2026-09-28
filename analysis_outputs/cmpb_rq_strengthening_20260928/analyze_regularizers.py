"""Summarize the regularizer comparison: BAC per regularizer x scale x classifier, and scale invariance
(max absolute BAC change across scales per held-out dataset; subject-level changes)."""
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
d = pd.concat([pd.read_csv(p, dtype={"scale_label": str}) for p in sorted((HERE / "regularizers" / "cells").glob("*.csv"))])
d["bac"] *= 100
pooled = d.groupby(["classifier", "regularizer", "scale_label"]).bac.mean().unstack("scale_label")[["1", "1e-3", "1e-6"]]
pooled["max_abs_change"] = (pooled.max(axis=1) - pooled.min(axis=1))
subj = d.pivot_table(index=["classifier", "regularizer", "held_out", "subject"], columns="scale_label", values="bac")
subj["changed"] = (subj.max(axis=1) - subj.min(axis=1)) > 1e-9
ch = subj.groupby(["classifier", "regularizer"]).changed.mean().mul(100).rename("subjects_with_any_change_%")
out = pooled.join(ch).round(3)
by_ds = d[d.scale_label == "1"].groupby(["classifier", "regularizer", "held_out"]).bac.mean().unstack("held_out").round(2)
out.to_csv(HERE / "regularizers_summary.csv"); by_ds.to_csv(HERE / "regularizers_by_dataset_scale1.csv")
pd.set_option("display.width", 200)
print(out.to_string()); print(); print(by_ds.to_string())
