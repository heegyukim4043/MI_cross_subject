"""Summarize the LODO4 target-access tiers (exploratory). Budget n above the pool size equals the full pool."""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ORDER = ["cho2017", "lee2019", "bnci2014001", "physionetmi"]
rng = np.random.default_rng(20260928)
def ci(d):
    m = rng.choice(d, (10000, len(d))).mean(1); return np.quantile(m, [0.025, 0.975])
b = pd.read_csv(HERE / "tiers_budget_results.csv")
b["n_eff"] = np.where(b.condition == "pool_budget", np.minimum(b.n_ref, b.n_pool), b.n_ref)
b["tier"] = b.condition.map({"no_alignment": "source-only (no alignment)", "transductive_all200": "batch transductive (all trials)",
                             "transductive_eval_only": "evaluation trials only"})
b.loc[b.condition == "pool_budget", "tier"] = "unlabeled n=" + b.n_ref.astype(str)
subj = b.groupby(["target", "subject", "tier"]).bac.mean().mul(100).unstack("tier")
tiers = ["source-only (no alignment)"] + [f"unlabeled n={n}" for n in (5, 10, 20, 40)] + ["batch transductive (all trials)"]
rows = []
for t in ORDER + ["all"]:
    g = subj if t == "all" else subj.loc[t]
    r = {"target": t, "n": len(g)}
    for c in tiers:
        r[c] = g[c].mean()
    for n in (5, 10, 20, 40):
        d = (g[f"unlabeled n={n}"] - g["batch transductive (all trials)"]).to_numpy(); lo, hi = ci(d)
        r[f"n={n} - transductive (pp)"] = f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]"
    rows.append(r)
bt = pd.DataFrame(rows)
c = pd.read_csv(HERE / "tiers_causal_results.csv")
c = c[c.burn_in == 20].groupby(["target", "subject", "condition"]).bac.mean().mul(100).unstack("condition")
crow = []
for t in ORDER + ["all"]:
    g = c if t == "all" else c.loc[t]
    d1 = (g.online_reset - g.batch_session).to_numpy(); d2 = (g.online_reset - g.batch_subject).to_numpy()
    l1, h1 = ci(d1); l2, h2 = ci(d2)
    crow.append({"target": t, "n": len(g), "online": g.online_reset.mean(), "batch_session": g.batch_session.mean(),
                 "batch_subject": g.batch_subject.mean(), "online - batch_session": f"{d1.mean():+.2f} [{l1:+.2f}, {h1:+.2f}]",
                 "online - batch_subject": f"{d2.mean():+.2f} [{l2:+.2f}, {h2:+.2f}]",
                 "order": "random permutations" if t == "cho2017" else "stored chronological"})
ct = pd.DataFrame(crow)
dev = pd.read_csv(HERE.parent / "cmpb_method_A_budget_fusion" / "dev_results.csv").groupby(["held_out", "subject", "k", "method"]).bac.mean().mul(100).reset_index()
frozen = {5: "w=0.05", 10: "w=0.1", 20: "w=0.2", 30: "w=0.35"}
lab = []
for t in ORDER + ["all"]:
    g = dev if t == "all" else dev[dev.held_out == t]
    r = {"target": t, "src_only (transductive SubjectEA)": g[g.method == "src_only"].bac.mean()}
    for k, m in frozen.items():
        x = g[(g.k == k) & (g.method == m)]
        r[f"labeled k={k} (M1, n)"] = f"{x.bac.mean():.2f} ({x.subject.nunique()})" if len(x) else "-"
    lab.append(r)
lt = pd.DataFrame(lab)
for name, df in (("tiers_summary_budget", bt), ("tiers_summary_causal", ct), ("tiers_summary_labeled", lt)):
    df.to_csv(HERE / f"{name}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(bt.round(2).to_string(index=False)); print(); print(ct.round(2).to_string(index=False)); print(); print(lt.round(2).to_string(index=False))
