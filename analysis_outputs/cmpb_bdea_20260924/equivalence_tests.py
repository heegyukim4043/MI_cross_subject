"""Equivalence tests (TOST) for "DatasetEA before SubjectEA adds nothing once EA is scale-correct".

Contrast: corrected (relative-floor) D->S minus corrected S, paired by target subject, in pp.
Equivalence margin: +/-1 pp (primary; smallest effect of practical interest), with +/-0.5 and
+/-2 pp as sensitivity. The margin was set after the point estimates had been seen, so the tests
are reported as supporting, not confirmatory.
Decision rule: two one-sided paired t-tests at alpha = 0.05 (equivalently, the 90% CI inside the
margin); a 90% paired-bootstrap CI (20,000 resamples) is reported alongside.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
A = HERE.parent
MARGINS = (0.5, 1.0, 2.0)
rng = np.random.default_rng(20260924)


def tost(diff: np.ndarray, margin: float) -> float:
    n, m, se = len(diff), diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
    p_low = 1 - stats.t.cdf((m + margin) / se, n - 1)   # H0: mean <= -margin
    p_high = stats.t.cdf((m - margin) / se, n - 1)      # H0: mean >= +margin
    return float(max(p_low, p_high))


def summarise(label: str, a: pd.Series, b: pd.Series) -> dict:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = (j["a"] - j["b"]).to_numpy() * (100 if j["a"].max() <= 1.0 else 1.0)
    boot = rng.choice(d, size=(20000, len(d))).mean(axis=1)
    se = d.std(ddof=1) / np.sqrt(len(d))
    t90 = stats.t.ppf(0.95, len(d) - 1) * se
    row = {"setting": label, "n": len(d), "mean_diff_pp": d.mean(), "t90_low": d.mean() - t90, "t90_high": d.mean() + t90,
           "boot90_low": np.quantile(boot, 0.05), "boot90_high": np.quantile(boot, 0.95)}
    for m in MARGINS:
        row[f"p_tost_{m}"] = tost(d, m)
        row[f"equivalent_{m}"] = row[f"p_tost_{m}"] < 0.05
    return row


def main() -> int:
    rows = []
    g = pd.read_csv(A / "cmpb_floor_gate_20260924/full/cross_subject_results.csv")
    for (s, t), gg in g[g.floor_mode == "relative"].groupby(["source", "target"]):
        pv = gg.pivot_table(index="subject", columns="alignment", values="bac")
        rows.append(summarise(f"CSP-LDA common48 {s}->{t}", pv["dataset_subject"], pv["subject"]))
    h = pd.read_csv(A / "cmpb_budget_20260924/ch27_subject_results.csv")
    for (s, t), gg in h[h.floor_mode == "relative"].groupby(["source", "target"]):
        pv = gg.pivot_table(index="subject", columns="alignment", values="bac")
        rows.append(summarise(f"CSP-LDA standard_mi27 {s}->{t}", pv["dataset_subject"], pv["subject"]))
    R = A / "cmpb_cspnet_pilot_20260924/server_final/results/loso_results_cmpbpilot_{}_cross_{}_cspnet.csv"
    for d in ("cho2017_to_lee2019", "lee2019_to_cho2017"):
        ds = pd.read_csv(str(R).format("relative_dataset_subject", d)).set_index("subject")["bac"]
        s_ = pd.read_csv(str(R).format("relative_subject", d)).set_index("subject")["bac"]
        rows.append(summarise(f"CSPNet common48 {d.replace('_to_', '->')} (seed 2026)", ds, s_))
    b = pd.read_csv(HERE / "bnci_alignment_results.csv")
    for s, gg in b[(b.scope == "cross") & (b.floor == "relative")].groupby("source"):
        pv = gg.pivot_table(index="subject", columns="alignment", values="bac")
        rows.append(summarise(f"CSP-LDA shared-ch {s}->bnci2014001", pv["dataset_subject"], pv["subject"]))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "equivalence_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
