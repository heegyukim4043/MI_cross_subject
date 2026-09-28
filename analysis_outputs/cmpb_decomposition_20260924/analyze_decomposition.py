"""Summaries, statistics and figure for the loss decomposition (run_decomposition.py output)."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260924)
ORDER = ["cho2017", "lee2019", "bnci2014001", "physionetmi"]


def ci(d):
    boot = RNG.choice(d, size=(20000, len(d))).mean(axis=1)
    return np.quantile(boot, 0.025), np.quantile(boot, 0.975)


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z (rank residual method)."""
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return spearmanr(res(rx), res(ry))


def main():
    df = pd.read_csv(HERE / "decomposition_subject_results.csv")
    for c in ("within", "loso_none", "loso_legacy", "loso_relative", "cross_rel", "ceiling_src_plus_own"):
        df[c] = df[c] * 100
    comps = {
        "numerical (rel - legacy EA, LOSO)": ("loso_relative", "loso_legacy"),
        "marginal shift removed by EA (rel EA - none, LOSO)": ("loso_relative", "loso_none"),
        "cross-dataset-specific (LOSO - LODO4)": ("loso_relative", "cross_rel"),
        "subject-specific transfer loss (within - LOSO)": ("within", "loso_relative"),
        "value of own labels (source+own - LOSO)": ("ceiling_src_plus_own", "loso_relative"),
    }
    rows = []
    for n in ORDER + ["all"]:
        g = df if n == "all" else df[df.dataset == n]
        base = {"dataset": n, "n": len(g), "within": g.within.mean(), "loso_none": g.loso_none.mean(),
                "loso_legacy": g.loso_legacy.mean(), "loso_relative": g.loso_relative.mean(), "cross_rel": g.cross_rel.mean(), "ceiling_src_plus_own": g.ceiling_src_plus_own.mean(),
                "separability_ceiling_loss (100 - within)": 100 - g.within.mean()}
        for lab, (a, b) in comps.items():
            d = (g[a] - g[b]).to_numpy(); lo, hi = ci(d)
            base[lab] = f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] p={(wilcoxon(d).pvalue if np.any(d != 0) else 1.0):.2g}" if len(d) > 5 else f"{d.mean():+.2f}"
        rows.append(base)
    tab = pd.DataFrame(rows); tab.to_csv(HERE / "decomposition_summary.csv", index=False)
    # subject typing
    def typ(r):
        if r.within < 70: return "decodability-limited (within<70)"
        return "transfer-limited (within>=70, LOSO<70)" if r.loso_relative < 70 else "decodable+transferable"
    df["type"] = df.apply(typ, axis=1)
    types = pd.crosstab(df.dataset, df.type, normalize="index").mul(100).round(1)
    types.loc["all"] = df.type.value_counts(normalize=True).mul(100).round(1)
    types.to_csv(HERE / "decomposition_subject_types.csv")
    # diagnostics
    df["transfer_loss"] = df.within - df.loso_relative
    diag = []
    for n in ORDER + ["all"]:
        g = df if n == "all" else df[df.dataset == n]
        r1 = spearmanr(g.cond_shift, g.transfer_loss); r2 = partial_spearman(g.cond_shift, g.transfer_loss, g.own_sep)
        r3 = spearmanr(g.own_sep, g.within); r4 = spearmanr(g.own_sep, g.loso_relative)
        diag.append({"dataset": n, "n": len(g),
                     "rho(cond_shift, transfer_loss)": f"{r1.statistic:+.3f} (p={r1.pvalue:.2g})",
                     "partial rho | own_sep": f"{r2.statistic:+.3f} (p={r2.pvalue:.2g})",
                     "rho(own_sep, within)": f"{r3.statistic:+.3f} (p={r3.pvalue:.2g})",
                     "rho(own_sep, LOSO)": f"{r4.statistic:+.3f} (p={r4.pvalue:.2g})"})
    diag = pd.DataFrame(diag); diag.to_csv(HERE / "decomposition_diagnostics.csv", index=False)
    df.to_csv(HERE / "decomposition_subject_results_typed.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60)
    print(tab.round(2).to_string(index=False)); print(); print(types.to_string()); print(); print(diag.to_string(index=False))
    # waterfall figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True)
    for ax, n in zip(axes, ORDER):
        g = df[df.dataset == n]
        steps = [("no EA\n(LOSO)", g.loso_none.mean()), ("legacy EA", g.loso_legacy.mean()), ("scale-correct\nEA", g.loso_relative.mean()),
                 ("cross-dataset\n(LODO)", g.cross_rel.mean()), ("within-subject\nonly", g.within.mean()),
                 ("source +\nown labels", g.ceiling_src_plus_own.mean())]
        ax.bar(range(len(steps)), [v for _, v in steps], color=["#9aa5b1", "#c2847a", "#3b7dd8", "#6aa84f", "#888888", "#222222"])
        for i, (_, v) in enumerate(steps):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
        ax.axhline(50, color="k", lw=0.6, ls=":"); ax.set_ylim(45, 90)
        ax.set_xticks(range(len(steps))); ax.set_xticklabels([s for s, _ in steps], fontsize=8)
        ax.set_title(f"{n} (n={len(g)})", fontsize=10)
    axes[0].set_ylabel("balanced accuracy (%)")
    fig.suptitle("Performance along the decomposition (21 shared channels, CSP-LDA)", fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "decomposition_waterfall.png", dpi=160)


if __name__ == "__main__":
    main()
