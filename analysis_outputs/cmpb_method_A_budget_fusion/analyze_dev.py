"""Apply the DESIGN.md selection rule to dev_results.csv and freeze the selected method in FROZEN.json.

M1 (w per k) and M2 (kappa) are selected with nested leave-one-dataset-out: the value used for a held-out dataset
is chosen on the other three. M3 has no outer hyperparameter. Candidate score = mean BAC over the 224 held-out
subjects (subject-weighted) and over k in KS. The frozen hyperparameters for the sealed test are then chosen once
on all four development datasets. Ties: M1 -> smaller w (W_GRID is ascending, max keeps the first);
M2 -> larger kappa (KAPPAS scanned in reverse), i.e. the more source-heavy solution in both cases.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KS = (5, 10, 20, 30)
W_GRID = ["nat", "0.05", "0.1", "0.2", "0.35", "0.5"]
KAPPAS = ["5", "10", "20", "40", "80", "160"]
BASELINES = {"src_only": "src_only", "src_own_nat": "w=nat", "src_own_eq": "w=0.5", "own_only": "own_only"}


def per_subject(df):
    """Mean over draws -> one BAC per (held_out, subject, k, method)."""
    return df.groupby(["held_out", "subject", "k", "method"], as_index=False)["bac"].mean()


def score(tab, method, k, datasets):
    t = tab[(tab.method == method) & (tab.k == k) & (tab.held_out.isin(datasets))]
    return t.bac.mean()


def main():
    dev = per_subject(pd.read_csv(HERE / "dev_results.csv"))
    names = sorted(dev.held_out.unique())
    src = dev[dev.method == "src_only"][["held_out", "subject", "bac"]].rename(columns={"bac": "src"})
    # candidate rows per subject and k, using nested selection
    out = []
    for h in names:
        other = [n for n in names if n != h]
        w_sel = {k: max(W_GRID, key=lambda w: score(dev, f"w={w}", k, other)) for k in KS}
        kap_sel = max(KAPPAS[::-1], key=lambda kp: np.mean([score(dev, f"kappa={kp}", k, other) for k in KS]))
        for k in KS:
            sub = dev[(dev.held_out == h) & (dev.k == k)]
            for name, meth in [("M1", f"w={w_sel[k]}"), ("M2", f"kappa={kap_sel}"), ("M3", "m3"),
                               ("src_own_nat", "w=nat"), ("src_own_eq", "w=0.5"), ("own_only", "own_only")]:
                t = sub[sub.method == meth][["held_out", "subject", "bac"]].assign(k=k, cand=name)
                out.append(t)
    tab = pd.concat(out).merge(src, on=["held_out", "subject"])
    tab = pd.concat([tab, src.assign(k=0, cand="src_only").rename(columns={"src": "bac"}).assign(src=lambda d: d.bac)])
    # k-averaged subject scores (src_only is constant over k)
    kav = tab[tab.k.isin(KS)].groupby(["cand", "held_out", "subject"], as_index=False).bac.mean()
    src_mean = src.src.mean()
    summary = kav.groupby("cand").bac.mean().to_dict(); summary["src_only"] = src_mean
    per_k = tab[tab.k.isin(KS)].groupby(["cand", "k"]).apply(lambda d: (d.bac - d.src).mean() * 100).unstack("k")
    no_harm = {c: per_k.loc[c, 5] >= -0.5 for c in ("M1", "M2", "M3")}
    best_base_name = max(BASELINES, key=lambda b: summary[b])
    eligible = [c for c in ("M1", "M2", "M3") if no_harm[c]]
    winner = max(eligible, key=lambda c: summary[c]) if eligible else None
    margin = (summary[winner] - summary[best_base_name]) * 100 if winner else None
    supported = bool(winner and margin >= 0.5)
    # frozen hyperparameters chosen once on all four development datasets
    frozen_params = {
        "M1": {str(k): max(W_GRID, key=lambda w: score(dev, f"w={w}", k, names)) for k in KS},
        "M2": {"kappa": max(KAPPAS[::-1], key=lambda kp: np.mean([score(dev, f"kappa={kp}", k, names) for k in KS]))},
        "M3": {},
    }
    code = {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest()
            for p in ("DESIGN.md", "run_dev.py", "analyze_dev.py", "dev_results.csv", "prep_stieger2021.py",
                      "download_stieger_session1.py")}
    frozen = {"selected": winner, "supported_on_dev": supported, "margin_vs_best_baseline_pp": margin,
              "best_baseline": best_base_name, "params": frozen_params.get(winner, {}) if winner else {},
              "all_params": frozen_params, "no_harm_k5": no_harm,
              "k_averaged_bac": {c: round(v * 100, 3) for c, v in summary.items()}, "sha256": code,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol": {"ks": list(KS), "draws": 3, "channels": 21, "w_grid": W_GRID, "kappas": KAPPAS,
                           "subject_seed": "1000 x dataset_index + subject (cho 0, lee 1, bnci 2, physionetmi 3, stieger2021 4)",
                           "draw_order": "P/E split, then k = 5, 10, 20, 30 x draw = 0, 1, 2",
                           "m3_inner_cv": "StratifiedKFold(5, shuffle=True, random_state=0)",
                           "tie_rule": "M1 smaller w, M2 larger kappa, M3 smaller w",
                           "a3_bootstrap": "subject-level percentile, 10000 resamples, seed 2026",
                           "sealed_set": "Stieger2021 session 1, S21-S62, >= 60 left/right trials"}}
    (HERE / "FROZEN.json").write_text(json.dumps(frozen, indent=2, default=str))
    print("k-averaged BAC (%):", {c: round(v * 100, 2) for c, v in sorted(summary.items(), key=lambda kv: -kv[1])})
    print("difference from src_only by k (pp):\n", per_k.round(2))
    print(f"best baseline: {best_base_name}; winner: {winner}; margin {margin}; supported on dev: {supported}")
    print("frozen:", json.dumps(frozen["params"]))


if __name__ == "__main__":
    main()
