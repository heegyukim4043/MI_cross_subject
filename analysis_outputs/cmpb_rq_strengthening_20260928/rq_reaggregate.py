"""RQ strengthening, re-aggregation of existing subject-level results (no new model fitting except floor counts).

Inputs (CSP-LDA, 21 shared channels, 224 subjects):
  cmpb_decomposition_20260924/decomposition_subject_results.csv   LOSO none/legacy/relative, within, source+own ceiling
  cmpb_bdea_20260924/lodo_cho2017_lee2019_bnci2014001_physionetmi_subject_results.csv   LODO4 by floor x alignment
Outputs (this folder):
  rq1_components.csv       component estimates with bootstrap 95% CI and Wilcoxon p, per dataset and pooled
  rq1_shapley.csv          two-factor attribution (alignment x transfer domain): both orders and the Shapley average
  rq1_robustness.csv       mean, 10th percentile, coverage (BAC >= 70%) per condition; negative-transfer rates per contrast
  rq2_dose_response.csv    per-subject count of legacy-floored eigenvalues vs the numerical loss (Spearman)
  rq3_equivalence_lodo4.csv  DatasetEA->SubjectEA minus SubjectEA per held-out dataset, TOST +/-1 pp (supporting; margin post hoc)
  RQ_REAGGREGATE_REPORT.md
All analyses here are exploratory re-aggregations of development data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
A = HERE.parent
for p in ("cmpb_bdea_20260924", "cmpb_floor_gate_20260924", "cmpb_budget_20260924"):
    sys.path.insert(0, str(A / p))
ORDER = ["cho2017", "lee2019", "bnci2014001", "physionetmi"]
RNG = np.random.default_rng(20260928)


def boot_ci(d, level=0.95, n=20000):
    m = RNG.choice(d, size=(n, len(d))).mean(axis=1)
    a = (1 - level) / 2
    return np.quantile(m, a), np.quantile(m, 1 - a)


def wil(d):
    return stats.wilcoxon(d).pvalue if np.any(d != 0) and len(d) > 5 else np.nan


def load_table():
    dec = pd.read_csv(A / "cmpb_decomposition_20260924/decomposition_subject_results.csv")
    lodo = pd.read_csv(A / "cmpb_bdea_20260924/lodo_cho2017_lee2019_bnci2014001_physionetmi_subject_results.csv")
    lodo["cond"] = "lodo_" + lodo.floor + "_" + lodo.alignment
    wide = lodo.pivot_table(index=["held_out", "subject"], columns="cond", values="bac").reset_index()
    wide = wide.rename(columns={"held_out": "dataset"})
    t = dec.merge(wide, on=["dataset", "subject"], how="inner")
    assert len(t) == 224, len(t)
    assert np.allclose(t.cross_rel, t.lodo_relative_subject), "decomposition cross_rel must equal LODO relative SubjectEA"
    cols = ["within", "loso_none", "loso_legacy", "loso_relative", "ceiling_src_plus_own", "lodo_none_none",
            "lodo_legacy_subject", "lodo_relative_subject", "lodo_legacy_dataset_subject", "lodo_relative_dataset_subject"]
    t[cols] = t[cols] * 100
    return t


COMPONENTS = {
    "numerical artifact, LOSO (rel - legacy)": ("loso_relative", "loso_legacy"),
    "numerical artifact, LODO (rel - legacy)": ("lodo_relative_subject", "lodo_legacy_subject"),
    "EA-removable shift, LOSO (rel - none)": ("loso_relative", "loso_none"),
    "EA-removable shift, LODO (rel - none)": ("lodo_relative_subject", "lodo_none_none"),
    "cross-dataset-specific, no alignment (LOSO - LODO)": ("loso_none", "lodo_none_none"),
    "cross-dataset-specific, legacy EA (LOSO - LODO)": ("loso_legacy", "lodo_legacy_subject"),
    "cross-dataset-specific, relative EA (LOSO - LODO)": ("loso_relative", "lodo_relative_subject"),
    "own labels (source+own - LOSO rel)": ("ceiling_src_plus_own", "loso_relative"),
}


def rq1_components(t):
    rows = []
    for n in ORDER + ["all"]:
        g = t if n == "all" else t[t.dataset == n]
        for lab, (a, b) in COMPONENTS.items():
            d = (g[a] - g[b]).to_numpy()
            lo, hi = boot_ci(d)
            rows.append({"dataset": n, "n": len(d), "component": lab, "mean_pp": d.mean(), "ci_low": lo, "ci_high": hi, "wilcoxon_p": wil(d)})
    return pd.DataFrame(rows)


def rq1_shapley(t):
    """f(alignment, domain): alignment in {none, legacy, relative}; domain in {LOSO, LODO}."""
    f = {("none", "LOSO"): "loso_none", ("legacy", "LOSO"): "loso_legacy", ("relative", "LOSO"): "loso_relative",
         ("none", "LODO"): "lodo_none_none", ("legacy", "LODO"): "lodo_legacy_subject", ("relative", "LODO"): "lodo_relative_subject"}
    rows = []
    for n in ORDER + ["all"]:
        g = t if n == "all" else t[t.dataset == n]
        v = {k: g[c].to_numpy() for k, c in f.items()}
        for lo_align, label in (("none", "alignment (none -> relative)"), ("legacy", "floor correction (legacy -> relative)")):
            # total change from (lo_align, LODO) to (relative, LOSO)
            align_first = v[("relative", "LODO")] - v[(lo_align, "LODO")]          # alignment measured under LODO
            align_second = v[("relative", "LOSO")] - v[(lo_align, "LOSO")]         # alignment measured under LOSO
            dom_first = v[(lo_align, "LOSO")] - v[(lo_align, "LODO")]              # domain measured without the change
            dom_second = v[("relative", "LOSO")] - v[("relative", "LODO")]         # domain measured with the change
            sh_align = (align_first + align_second) / 2
            sh_dom = (dom_first + dom_second) / 2
            for name, d in ((f"{label}: measured under LODO", align_first), (f"{label}: measured under LOSO", align_second),
                            (f"{label}: Shapley", sh_align), (f"domain (LODO -> LOSO) with {lo_align}", dom_first),
                            ("domain (LODO -> LOSO) with relative", dom_second), (f"domain: Shapley vs {lo_align}", sh_dom),
                            (f"interaction ({label} x domain)", align_second - align_first)):
                lo, hi = boot_ci(d)
                rows.append({"dataset": n, "n": len(d), "quantity": name, "mean_pp": d.mean(), "ci_low": lo, "ci_high": hi, "wilcoxon_p": wil(d)})
    return pd.DataFrame(rows)


CONDS = ["loso_none", "loso_legacy", "loso_relative", "lodo_none_none", "lodo_legacy_subject", "lodo_relative_subject",
         "lodo_relative_dataset_subject", "within", "ceiling_src_plus_own"]
CONTRASTS = {
    "relative EA vs none (LOSO)": ("loso_relative", "loso_none"),
    "relative EA vs none (LODO)": ("lodo_relative_subject", "lodo_none_none"),
    "legacy EA vs none (LODO)": ("lodo_legacy_subject", "lodo_none_none"),
    "LODO vs LOSO (relative EA)": ("lodo_relative_subject", "loso_relative"),
    "DatasetEA->SubjectEA vs SubjectEA (LODO, relative)": ("lodo_relative_dataset_subject", "lodo_relative_subject"),
}


def rq1_robustness(t):
    rows = []
    for n in ORDER + ["all"]:
        g = t if n == "all" else t[t.dataset == n]
        for c in CONDS:
            x = g[c].to_numpy()
            rows.append({"dataset": n, "kind": "condition", "name": c, "mean": x.mean(), "p10": np.percentile(x, 10),
                         "coverage_ge70": (x >= 70).mean() * 100})
        for lab, (a, b) in CONTRASTS.items():
            d = (g[a] - g[b]).to_numpy()
            rows.append({"dataset": n, "kind": "contrast", "name": lab, "mean": d.mean(), "p10": np.percentile(d, 10),
                         "ntr_lt0": (d < 0).mean() * 100, "ntr_lt_minus5": (d < -5).mean() * 100})
    return pd.DataFrame(rows)


def floor_counts():
    from run_bdea_bnci import load_npz
    from run_floor_gate import align_subjects
    from run_lodo import PATHS
    raw = {n: load_npz(PATHS[n]) for n in ORDER}
    shared = [c for c in raw[ORDER[0]]["ch_names"] if all(c in raw[n]["ch_names"] for n in ORDER)]
    rows = []
    for n in ORDER:
        d = raw[n]
        x = d["X"][:, [d["ch_names"].index(c) for c in shared], :200].astype(np.float32)
        _, diag = align_subjects(x, d["subjects"].astype(np.int64), "legacy", n, "subject")
        rows += [{"dataset": n, "subject": r["subject"], "floored": r["floored_or_dropped"], "eigen_max": r["eigen_max"]} for r in diag]
    return pd.DataFrame(rows), len(shared)


def rq2_dose(t):
    fc, n_ch = floor_counts()
    m = t.merge(fc, on=["dataset", "subject"])
    m["num_loss_loso"] = m.loso_relative - m.loso_legacy
    m["num_loss_lodo"] = m.lodo_relative_subject - m.lodo_legacy_subject
    rows = []
    for n in ORDER + ["all"]:
        g = m if n == "all" else m[m.dataset == n]
        for y in ("num_loss_loso", "num_loss_lodo"):
            r = stats.spearmanr(g.floored, g[y]) if g.floored.nunique() > 1 else None
            rows.append({"dataset": n, "n": len(g), "outcome": y, "floored_median": g.floored.median(),
                         "floored_min": g.floored.min(), "floored_max": g.floored.max(), "n_channels": n_ch,
                         "rho": r.statistic if r else np.nan, "p": r.pvalue if r else np.nan,
                         "loss_if_floored0": g[g.floored == 0][y].mean() if (g.floored == 0).any() else np.nan,
                         "loss_if_floored_gt0": g[g.floored > 0][y].mean() if (g.floored > 0).any() else np.nan})
    m[["dataset", "subject", "floored", "num_loss_loso", "num_loss_lodo"]].to_csv(HERE / "rq2_dose_response_subjects.csv", index=False)
    return pd.DataFrame(rows)


def rq3_equivalence(t):
    rows = []
    for floor in ("relative", "legacy"):
        for n in ORDER + ["all"]:
            g = t if n == "all" else t[t.dataset == n]
            d = (g[f"lodo_{floor}_dataset_subject"] - g[f"lodo_{floor}_subject"]).to_numpy()
            se = d.std(ddof=1) / np.sqrt(len(d)); tq = stats.t.ppf(0.95, len(d) - 1)
            lo90, hi90 = d.mean() - tq * se, d.mean() + tq * se
            p_low = 1 - stats.t.cdf((d.mean() + 1) / se, len(d) - 1)
            p_high = stats.t.cdf((d.mean() - 1) / se, len(d) - 1)
            b_lo, b_hi = boot_ci(d, level=0.90)
            rows.append({"floor": floor, "held_out": n, "n": len(d), "mean_pp": d.mean(), "t90_low": lo90, "t90_high": hi90,
                         "boot90_low": b_lo, "boot90_high": b_hi, "p_tost_1pp": max(p_low, p_high),
                         "equivalent_1pp": bool(lo90 > -1 and hi90 < 1)})
    return pd.DataFrame(rows)


def main():
    t = load_table()
    comp, shap, rob, eq = rq1_components(t), rq1_shapley(t), rq1_robustness(t), rq3_equivalence(t)
    dose = rq2_dose(t)
    for name, df in (("rq1_components", comp), ("rq1_shapley", shap), ("rq1_robustness", rob),
                     ("rq2_dose_response", dose), ("rq3_equivalence_lodo4", eq)):
        df.to_csv(HERE / f"{name}.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    fmt = lambda df: df.round(3).to_string(index=False)
    lines = ["# RQ strengthening: re-aggregation of existing development results (exploratory)", "",
             "CSP-LDA, 21 shared channels, 224 subjects (Cho 52, Lee 54, BNCI 9, PhysionetMI 109). Bootstrap 95% CI over subjects.", "",
             "## RQ1 components (pooled)", "```", fmt(comp[comp.dataset == "all"]), "```", "",
             "## RQ1 two-factor attribution, pooled (order sensitivity; Shapley = mean of both orders)", "```", fmt(shap[shap.dataset == "all"]), "```", "",
             "## RQ1 robustness, pooled", "```", fmt(rob[rob.dataset == "all"]), "```", "",
             "## RQ2 dose-response (legacy floored eigenvalues vs numerical loss)", "```", fmt(dose), "```", "",
             "## RQ3 DatasetEA->SubjectEA minus SubjectEA, LODO4 (TOST +/-1 pp; margin post hoc, supporting only)", "```", fmt(eq), "```"]
    (HERE / "RQ_REAGGREGATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
