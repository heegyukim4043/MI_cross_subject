"""Neural (CSPNet, seed 2026) decomposition on the LODO4 21-channel set: within-dataset LOSO vs cross-dataset LODO4.

Inputs (MI_test/results):
  generic_gen_loso21_{ds}_{cond}_loso.csv       cond in none, legacy, relative (pod E, 21 shared channels)
  generic_gen_loso21_cho2017_legacy_chofix_loso.csv  (Cho absolute floor after the unit correction; local GPU)
  generic_gen_lodo4_{cond}_lodo.csv             relative (unchanged by the Cho unit correction)
  generic_gen_lodo4_{none,legacy}_chofix_lodo.csv   (unit-corrected)
Components per subject (pp): numerical artifact under LOSO (relative - legacy), EA-removable shift under LOSO
(relative - none), cross-dataset-specific loss (LOSO - LODO4) for none, legacy and relative.
Cho legacy LOSO uses only the corrected run; it is left out until that run is complete.
Reported per dataset, pooled and equal-weight (mean of dataset means; bootstrap within dataset).
Output: neural_loso21_components.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "MI_test" / "results"
ORDER = ["cho2017", "lee2019", "bnci2014001", "physionetmi"]
N = {"cho2017": 52, "lee2019": 54, "bnci2014001": 9, "physionetmi": 109}
RNG = np.random.default_rng(20260930)


def read(f, key):
    if not f.exists() or f.stat().st_size == 0:  # missing, or a running job that has not flushed yet
        return None
    d = pd.read_csv(f).drop_duplicates([key, "subject"])
    return d.set_index([key, "subject"]).bac * 100


def loso(ds, cond):
    tag = "legacy_chofix" if (ds == "cho2017" and cond == "legacy") else cond
    s = read(RES / f"generic_gen_loso21_{ds}_{tag}_loso.csv", "held_out")
    return s if s is not None and len(s) == N[ds] else None


def lodo(cond):
    tag = cond + ("_chofix" if cond in ("none", "legacy") else "")
    return read(RES / f"generic_gen_lodo4_{tag}_lodo.csv", "held_out")


def summarize(label, diffs):
    rows, groups = [], {n: v for n, v in diffs.items() if v is not None}
    for n, v in list(groups.items()) + ([("pooled", np.concatenate(list(groups.values())))] if groups else []):
        b = RNG.choice(v, (10000, len(v))).mean(1)
        rows.append({"component": label, "scope": n, "n": len(v), "mean_pp": v.mean(),
                     "ci95_low": np.quantile(b, .025), "ci95_high": np.quantile(b, .975),
                     "wilcoxon_p": stats.wilcoxon(v).pvalue if len(v) > 5 and np.any(v != 0) else np.nan})
    if len(groups) == len(ORDER):
        ew = np.mean([RNG.choice(v, (10000, len(v))).mean(1) for v in groups.values()], axis=0)
        rows.append({"component": label, "scope": "equal_weight", "n": sum(len(v) for v in groups.values()),
                     "mean_pp": np.mean([v.mean() for v in groups.values()]),
                     "ci95_low": np.quantile(ew, .025), "ci95_high": np.quantile(ew, .975)})
    return rows


def main():
    L = {c: lodo(c) for c in ("none", "legacy", "relative")}
    levels, rows = [], []
    for ds in ORDER:
        levels.append({"dataset": ds, **{f"loso_{c}": (loso(ds, c).mean() if loso(ds, c) is not None else np.nan)
                                        for c in ("none", "legacy", "relative")},
                       **{f"lodo4_{c}": L[c].loc[ds].mean() for c in L}})

    def diff(fa, fb):
        out = {}
        for ds in ORDER:
            a, b = fa(ds), fb(ds)
            out[ds] = None if a is None or b is None else (a.droplevel(0) - b.droplevel(0) if a.index.nlevels > 1 else a - b).to_numpy()
        return out

    def lo(c):
        return lambda ds: loso(ds, c)

    def ld(c):
        return lambda ds: L[c].loc[[ds]]

    rows += summarize("numerical artifact, LOSO (relative - legacy)", diff(lo("relative"), lo("legacy")))
    rows += summarize("EA-removable shift, LOSO (relative - none)", diff(lo("relative"), lo("none")))
    for c in ("none", "legacy", "relative"):
        rows += summarize(f"cross-dataset-specific, {c} (LOSO - LODO4)", diff(lo(c), ld(c)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "neural_loso21_components.csv", index=False)
    pd.DataFrame(levels).to_csv(HERE / "neural_loso21_levels.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(levels).round(2).to_string(index=False)); print()
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
