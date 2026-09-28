"""Single sealed test of the frozen method A on Stieger2021 S21-S62, session 1 (DESIGN.md, FROZEN.json).

Refuses to run unless FROZEN.json exists and DESIGN.md, run_dev.py and analyze_dev.py still match the hashes
recorded at freezing. Source = all four development datasets (21 shared channels, relative SubjectEA);
target = each Stieger2021 subject with >= 60 left/right trials, relative SubjectEA (transductive, label-free).
Same P/E split and k-shot draws for every method (generator seed 1000 x 4 + subject). Conditions: src_only,
the frozen method, and the baselines src_own_nat, src_own_eq, own_only. Hypotheses A1-A3 and the exploratory
plateau are computed exactly as specified and written to sealed_hypotheses.csv and SEALED_REPORT.md.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_dev as rd  # noqa: E402

SEALED = HERE / "sealed_stieger2021" / "stieger2021_s21_s62_session1.npz"
STIEGER_INDEX = 4
MIN_TRIALS = 60


def check_frozen():
    frozen = json.loads((HERE / "FROZEN.json").read_text())
    for name in ("DESIGN.md", "run_dev.py", "analyze_dev.py"):
        now = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if now != frozen["sha256"][name]:
            raise SystemExit(f"{name} changed after freezing; refusing to run the sealed test")
    return frozen


def main():
    frozen = check_frozen()
    if frozen["selected"] != "M1":
        raise SystemExit("this script implements the frozen M1 rule only")
    w_k = {int(k): (v if v == "nat" else float(v)) for k, v in frozen["params"].items()}
    base_name = frozen["best_baseline"]
    raw = {n: rd.load_npz(rd.PATHS[n]) for n in rd.NAMES}
    shared = [c for c in raw[rd.NAMES[0]]["ch_names"] if all(c in raw[n]["ch_names"] for n in rd.NAMES)]
    src_x, src_y = [], []
    for n in rd.NAMES:
        d = raw[n]
        x = d["X"][:, [d["ch_names"].index(c) for c in shared], :rd.N_T].astype(np.float32)
        S = d["subjects"].astype(np.int64)
        src_x.append(rd.align_subjects(x, S, "relative", n, "subject")[0]); src_y.append(d["y"].astype(np.int64))
    sc, sm = rd.covs(np.concatenate(src_x)); sy = np.concatenate(src_y); del src_x
    src_model = rd.Weighted(sc, sm, sy, np.ones(len(sy)))

    t = np.load(SEALED, allow_pickle=True)
    names = [str(c) for c in t["ch_names"]]
    x = t["X"][:, [names.index(c) for c in shared], :rd.N_T].astype(np.float32)
    y, S = t["y"].astype(np.int64), t["subjects"].astype(np.int64)
    counts = pd.Series(S).value_counts()
    keep = sorted(int(s) for s in counts.index if counts[s] >= MIN_TRIALS)
    excluded = sorted(int(s) for s in counts.index if counts[s] < MIN_TRIALS)
    xa = rd.align_subjects(x, S, "relative", "stieger2021", "subject")[0]

    rows = []
    for s in keep:
        m = S == s
        xs, ys = xa[m], y[m]
        rng = np.random.default_rng(1000 * STIEGER_INDEX + s)
        e_idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(ys == c), (ys == c).sum() // 2, replace=False) for c in (0, 1)]))
        p_idx = np.setdiff1d(np.arange(len(ys)), e_idx)
        c, mu = rd.covs(xs)
        ce, me, ye = c[e_idx], mu[e_idx], ys[e_idx]
        rows.append({"subject": s, "k": 0, "draw": 0, "method": "src_only", "bac": rd.bac(ye, src_model.decision(ce, me))})
        for k in rd.KS:
            pool = {cl: p_idx[ys[p_idx] == cl] for cl in (0, 1)}
            if min(len(v) for v in pool.values()) < k:
                continue
            for draw in range(rd.DRAWS):
                sel = np.concatenate([rng.choice(pool[cl], k, replace=False) for cl in (0, 1)])
                oc, om, oy = c[sel], mu[sel], ys[sel]
                base = {"subject": s, "k": k, "draw": draw}
                own = rd.Weighted(oc, om, oy, np.ones(len(oy)))
                rows.append({**base, "method": "own_only", "bac": rd.bac(ye, own.decision(ce, me))})
                for label, w in (("selected", w_k[k]), ("src_own_nat", "nat"), ("src_own_eq", 0.5)):
                    mdl = rd.fit_share(sc, sm, sy, oc, om, oy, w)
                    rows.append({**base, "method": label, "bac": rd.bac(ye, mdl.decision(ce, me))})
        print(f"[sealed] S{s} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "sealed_results.csv", index=False)

    ps = res.groupby(["subject", "k", "method"], as_index=False).bac.mean()
    src = ps[ps.method == "src_only"].set_index("subject").bac
    piv = ps[ps.k > 0].pivot_table(index=["subject", "k"], columns="method", values="bac")
    piv["src"] = src.reindex(piv.index.get_level_values(0)).to_numpy()
    d_src = (piv["selected"] - piv["src"]) * 100
    d_base = (piv["selected"] - piv[base_name]) * 100
    a1 = d_src[d_src.index.get_level_values(1).isin([10, 20, 30])].groupby(level=0).mean()
    a2 = d_base.groupby(level=0).mean()
    p1 = wilcoxon(a1, alternative="greater").pvalue
    p2 = wilcoxon(a2, alternative="greater").pvalue
    holm = sorted([("A1", p1), ("A2", p2)], key=lambda t: t[1])
    adj = {holm[0][0]: min(1.0, 2 * holm[0][1])}
    adj[holm[1][0]] = min(1.0, max(adj[holm[0][0]], holm[1][1]))
    k5 = d_src[d_src.index.get_level_values(1) == 5].groupby(level=0).mean().to_numpy()
    boot = np.random.default_rng(2026).choice(k5, (10000, len(k5))).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    means = ps[ps.method == "selected"].groupby("k").bac.mean() * 100
    ks = sorted(means.index)
    plateau = next((k for k, k2 in zip(ks, ks[1:]) if means[k2] - means[k] < 0.5), None)
    hyp = pd.DataFrame([
        {"hypothesis": "A1", "n": len(a1), "estimate_pp": a1.mean(), "raw_p": p1, "holm_p": adj["A1"],
         "supported": bool(adj["A1"] < 0.05 and a1.mean() > 0), "note": "selected - src_only, k in {10,20,30}"},
        {"hypothesis": "A2", "n": len(a2), "estimate_pp": a2.mean(), "raw_p": p2, "holm_p": adj["A2"],
         "supported": bool(adj["A2"] < 0.05 and a2.mean() > 0), "note": f"selected - {base_name}, k in {{5,10,20,30}}"},
        {"hypothesis": "A3", "n": len(k5), "estimate_pp": k5.mean(), "ci_low": lo, "ci_high": hi,
         "supported": bool(lo > -1.0), "note": "selected - src_only at k=5; subject bootstrap 10000, seed 2026"},
    ])
    hyp.to_csv(HERE / "sealed_hypotheses.csv", index=False)
    lines = ["# Sealed test of method A on Stieger2021 (S21-S62, session 1)", "",
             f"Frozen rule: {frozen['selected']} with w_k = {frozen['params']} (frozen {frozen['frozen_at_utc']}).",
             f"Subjects analysed: {len(keep)}; excluded (< {MIN_TRIALS} trials): {excluded}.", "",
             "```", hyp.to_string(index=False), "```", "",
             "Mean BAC by k (%): " + ", ".join(f"k={k}: {means[k]:.2f}" for k in ks) + f"; src_only {src.mean() * 100:.2f}.",
             f"Empirical plateau on the tested grid (exploratory): {plateau}."]
    (HERE / "SEALED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
