"""Draft Results figures (Figs. 3-6) from the analysis outputs (Cho2017 unit-corrected).

Fig. 3  RQ1 decomposition: (a) CSP-LDA components per dataset; (b) CSP-LDA vs CSPNet, subject-pooled.
Fig. 4  RQ2 scale: (a) BAC vs amplitude scale per regularizer (CSP-LDA, SubjectEA, LODO); (b) share of changed test
        predictions (absolute floor); (c) Cho2017 unit correction as a natural intervention (pooled LODO CSP-LDA).
Fig. 5  RQ3 forest plot of DatasetEA->SubjectEA minus SubjectEA (90% CI; +/-1 pp band).
Fig. 6  RQ4 target access: (a) unlabeled budget n; (b) labeled k (development, frozen rule) and Stieger sealed test.
Outputs: fig3_rq1_draft ... fig6_rq4_draft (.pdf, .png) next to this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
A = HERE.parent / "analysis_outputs"  # repository root / analysis_outputs
RQ = A / "cmpb_rq_strengthening_20260928"
NEU = A / "cmpb_neural_decomposition"
DS = ["cho2017", "lee2019", "bnci2014001", "physionetmi"]
DSL = {"cho2017": "Cho2017", "lee2019": "Lee2019", "bnci2014001": "BNCI2014-001", "physionetmi": "PhysionetMI",
       "all": "Pooled (224)"}
C = {"num": "#D55E00", "ea": "#009E73", "cd": "#E69F00", "lab": "#0072B2", "grey": "#666666", "rel": "#0072B2",
     "abs": "#D55E00", "none": "#999999"}
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=400 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def label(ax, s):
    ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def fig3():
    comp = pd.read_csv(RQ / "rq1_components.csv")
    keys = [("numerical artifact, LODO (rel - legacy)", "Δnum (LODO)", C["num"]),
            ("EA-removable shift, LODO (rel - none)", "ΔEA (LODO)", C["ea"]),
            ("cross-dataset-specific, relative EA (LOSO - LODO)", "Δcd (relative)", C["cd"]),
            ("own labels (source+own - LOSO rel)", "Δlab", C["lab"])]
    fig, axs = plt.subplots(1, 2, figsize=(180 / 25.4, 70 / 25.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    groups = DS + ["all"]
    w = 0.19
    for j, (k, lab, col) in enumerate(keys):
        d = comp[comp.component == k].set_index("dataset").loc[groups]
        x = np.arange(len(groups)) + (j - 1.5) * w
        ax.bar(x, d.mean_pp, w, color=col, label=lab)
        ax.errorbar(x, d.mean_pp, [d.mean_pp - d.ci_low, d.ci_high - d.mean_pp], fmt="none", ecolor="black", lw=0.6,
                    capsize=1.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(np.arange(len(groups)), [DSL[g] for g in groups])
    ax.set_ylabel("paired difference in BAC (pp)")
    ax.set_ylim(-5.5, 13)
    ax.legend(ncol=4, loc="upper left", fontsize=6.2)
    ax.set_title("CSP-LDA, per dataset (95% CI)", fontsize=7)
    label(ax, "a")

    ax = axs[1]
    classical = {r.component: r for r in comp[comp.dataset == "all"].itertuples()}
    ms = pd.read_csv(NEU / "neural_multiseed_summary_chofix.csv")
    lo = pd.read_csv(NEU / "neural_loso21_components.csv")

    def ms_row(contrast):
        r = ms[(ms.contrast == contrast) & (ms["scope"] == "pooled")].iloc[0]
        return r.mean_pp, r.ci95_low, r.ci95_high

    def lo_row(comp_name):
        r = lo[(lo.component == comp_name) & (lo["scope"] == "pooled")].iloc[0]
        return r.mean_pp, r.ci95_low, r.ci95_high

    items = [("Δnum\nLODO", (classical["numerical artifact, LODO (rel - legacy)"].mean_pp,
                             classical["numerical artifact, LODO (rel - legacy)"].ci_low,
                             classical["numerical artifact, LODO (rel - legacy)"].ci_high),
              ms_row("numerical artifact (relative - legacy)")),
             ("ΔEA\nLODO", (classical["EA-removable shift, LODO (rel - none)"].mean_pp,
                            classical["EA-removable shift, LODO (rel - none)"].ci_low,
                            classical["EA-removable shift, LODO (rel - none)"].ci_high),
              ms_row("EA gain (relative - none)")),
             ("Δnum\nLOSO", (classical["numerical artifact, LOSO (rel - legacy)"].mean_pp,
                             classical["numerical artifact, LOSO (rel - legacy)"].ci_low,
                             classical["numerical artifact, LOSO (rel - legacy)"].ci_high),
              lo_row("numerical artifact, LOSO (relative - legacy)")),
             ("Δcd\nrelative", (classical["cross-dataset-specific, relative EA (LOSO - LODO)"].mean_pp,
                                classical["cross-dataset-specific, relative EA (LOSO - LODO)"].ci_low,
                                classical["cross-dataset-specific, relative EA (LOSO - LODO)"].ci_high),
              lo_row("cross-dataset-specific, relative (LOSO - LODO4)"))]
    for i, (lab, cl, nn) in enumerate(items):
        for off, (m, l, h), col, name in ((-0.18, cl, "#444444", "CSP-LDA"), (0.18, nn, "#56B4E9", "CSPNet")):
            ax.bar(i + off, m, 0.34, color=col, label=name if i == 0 else None)
            ax.errorbar(i + off, m, [[m - l], [h - m]], fmt="none", ecolor="black", lw=0.6, capsize=1.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(items)), [i[0] for i in items])
    ax.set_title("Subject-pooled, classical vs neural", fontsize=7)
    ax.legend(loc="upper right", fontsize=6.2)
    label(ax, "b")
    fig.tight_layout()
    save(fig, "fig3_rq1_draft")


def fig4():
    reg = pd.read_csv(RQ / "regularizers_summary.csv")
    dis = pd.read_csv(A / "cmpb_unit_intervention_20260925" / "analysis" / "prediction_disagreement.csv",
                      encoding="utf-8-sig", dtype={"scale_label": str})
    fig, axs = plt.subplots(1, 3, figsize=(180 / 25.4, 62 / 25.4), gridspec_kw={"width_ratios": [1.2, 1, 0.9]})
    ax = axs[0]
    scales = ["1", "1e-3", "1e-6"]
    style = {"legacy": (C["abs"], "-", "absolute floor, float32"), "absolute64": (C["abs"], "--", "absolute floor, float64"),
             "relative": (C["rel"], "-", "relative floor"), "trace_loading": ("#56B4E9", "-", "trace loading"),
             "oas": ("#009E73", "-", "OAS"), "ledoit_wolf": ("#009E73", "--", "Ledoit–Wolf")}
    d = reg[reg.classifier == "CSP-LDA"].set_index("regularizer")
    for r, (col, ls, lab) in style.items():
        ax.plot(range(3), d.loc[r, scales].astype(float), ls, color=col, marker="o", ms=2.5, lw=0.9, label=lab)
    ax.set_xticks(range(3), ["×1 (V)", "×10$^{-3}$", "×10$^{-6}$"])
    ax.set_ylabel("BAC (%), CSP-LDA, SubjectEA, LODO")
    ax.set_xlabel("amplitude scale")
    ax.legend(fontsize=5.6, loc="center left", bbox_to_anchor=(0.0, 0.42))
    label(ax, "a")

    ax = axs[1]
    sub = dis[(dis.classifier == "CSP-LDA") & (dis.floor == "legacy") & (dis.alignment == "subject")]
    for i, (s, hatch) in enumerate((("1e-3", ""), ("1e-6", "////"))):
        v = sub[sub.scale_label == s].set_index("held_out").loc[DS, "disagreement_rate"] * 100
        ax.bar(np.arange(4) + (i - 0.5) * 0.36, v, 0.36, color=C["abs"], alpha=0.55 + 0.35 * i, hatch=hatch,
               label=f"scale {s.replace('1e-', '×10$^{-') + '}$'}")
    ax.set_xticks(range(4), [DSL[g] for g in DS], rotation=20)
    ax.set_ylabel("test predictions changed (%)")
    ax.set_title("absolute floor (relative: 0 changed)", fontsize=6.6)
    ax.legend(fontsize=6)
    label(ax, "b")

    ax = axs[2]
    pil = pd.read_csv(RQ / "pilot_cho_unitfix_subject_results.csv")
    pil = pil[(pil.design == "LODO") & (pil.classifier == "CSP-LDA")]
    conds = [("none", "none", "no\nalignment", C["none"]), ("legacy", "subject", "absolute\nfloor", C["abs"]),
             ("relative", "subject", "relative\nfloor", C["rel"])]
    for i, (fl, al, lab, col) in enumerate(conds):
        q = pil[(pil.floor == fl) & (pil.alignment == al)]
        diff = (q[q.version == "cho_corrected"].set_index(["held_out", "subject"]).bac
                - q[q.version == "original"].set_index(["held_out", "subject"]).bac)
        m = diff.mean()
        ax.bar(i, m, 0.6, color=col)
        ax.text(i, min(m, 0) - 0.08, f"{m:+.2f}", ha="center", va="top", fontsize=6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(-3.7, 0.3)
    ax.set_xticks(range(3), [c[2] for c in conds])
    ax.set_ylabel("change in pooled BAC (pp)")
    ax.set_title("Cho2017 unit correction (×0.03125)", fontsize=6.6)
    label(ax, "c")
    fig.tight_layout()
    save(fig, "fig4_rq2_draft")


def fig5():
    eq = pd.read_csv(RQ / "rq3_equivalence_lodo4.csv")
    ms = pd.read_csv(NEU / "neural_multiseed_summary_chofix.csv")
    rows = []
    for fl, name, col in (("relative", "CSP-LDA, relative", C["rel"]), ("legacy", "CSP-LDA, absolute", C["abs"])):
        for g in DS + ["all"]:
            r = eq[(eq.floor == fl) & (eq.held_out == g)].iloc[0]
            rows.append((name, g, r.mean_pp, r.t90_low, r.t90_high, col))
    for contrast, name, col in (("DatasetEA increment (dsea_relative - relative)", "CSPNet, relative (5 seeds)", "#56B4E9"),
                                ("legacy DatasetEA increment (dsea_legacy - legacy)", "CSPNet, absolute (seed 2026)", "#E69F00")):
        for g in DS + ["pooled"]:
            r = ms[(ms.contrast == contrast) & (ms["scope"] == g)].iloc[0]
            rows.append((name, "all" if g == "pooled" else g, r.mean_pp, r.ci90_low, r.ci90_high, col))
    rows.append(("Dreyer2023 H3 (pre-specified)", "dreyer", 0.093, -0.230, 0.417, "black"))
    fig, ax = plt.subplots(figsize=(120 / 25.4, 110 / 25.4))
    y, ticks, labels = 0, [], []
    prev = None
    for name, g, m, l, h, col in rows:
        if name != prev and prev is not None:
            y += 0.7
        prev = name
        ax.errorbar(m, -y, xerr=[[m - l], [h - m]], fmt="o" if g not in ("all", "dreyer") else "D", color=col, ms=3,
                    lw=0.9, capsize=1.5)
        ticks.append(-y); labels.append(f"{name}: {'Pooled' if g == 'all' else DSL.get(g, 'Dreyer2023 (87)')}")
        y += 1
    ax.axvspan(-1, 1, color="#DDDDDD", alpha=0.6, zorder=0)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(ticks, labels, fontsize=5.8)
    ax.set_xlabel("DatasetEA→SubjectEA − SubjectEA (pp), 90% CI; shaded ±1 pp")
    fig.tight_layout()
    save(fig, "fig5_rq3_draft")


def fig6():
    bud = pd.read_csv(RQ / "tiers_summary_budget.csv")
    lab = pd.read_csv(RQ / "tiers_summary_labeled.csv")
    fig, axs = plt.subplots(1, 2, figsize=(180 / 25.4, 65 / 25.4))
    ax = axs[0]
    ns = [5, 10, 20, 40]
    for g, col in zip(DS + ["all"], ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "black"]):
        r = bud[bud.target == g].iloc[0]
        ys = [r[f"unlabeled n={n}"] for n in ns]
        ax.plot(ns, ys, "o-", color=col, ms=2.5, lw=1.2 if g == "all" else 0.8, label=DSL[g])
        ax.plot([55], [r["batch transductive (all trials)"]], "s", color=col, ms=3)
        ax.plot([2.5], [r["source-only (no alignment)"]], "^", color=col, ms=3)
    ax.set_xscale("log")
    ax.set_xticks([2.5, 5, 10, 20, 40, 55], ["none", "5", "10", "20", "40", "all"])
    ax.set_xlabel("unlabeled target trials used for alignment")
    ax.set_ylabel("BAC (%), evaluation half")
    ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    label(ax, "a")

    ax = axs[1]
    ks = [5, 10, 20, 30]
    # per dataset only: the summary's "all" row mixes subsets (PhysionetMI has no k >= 20) and counts overlapping IDs
    for g, col in zip(DS, ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]):
        r = lab[lab.target == g].iloc[0]
        ys = []
        for k in ks:
            v = str(r[f"labeled k={k} (M1, n)"])
            ys.append(np.nan if v.strip() == "-" else float(v.split()[0]))
        ax.plot([0] + ks, [r["src_only (transductive SubjectEA)"]] + ys, "o-", color=col, ms=2.5, lw=0.8,
                label=DSL[g])
    sr = pd.read_csv(A / "cmpb_method_A_budget_fusion" / "sealed_results.csv")
    sr = sr[sr.method.isin(["src_only", "selected"])].groupby(["k", "subject"]).bac.mean().groupby("k").mean() * 100
    st = sr.to_dict()  # k = 0 is source-only; k > 0 the frozen rule (subject means over draws)
    ax.plot(list(st), list(st.values()), "D--", color="#D55E00", ms=3, lw=1.0, label="Stieger2021 sealed (41)")
    ax.set_xticks([0] + ks, ["0"] + [str(k) for k in ks])
    ax.set_xlabel("labeled target trials per class (k)")
    ax.set_ylabel("BAC (%)")
    ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    label(ax, "b")
    fig.tight_layout()
    save(fig, "fig6_rq4_draft")


if __name__ == "__main__":
    fig3(); fig4(); fig5(); fig6()
