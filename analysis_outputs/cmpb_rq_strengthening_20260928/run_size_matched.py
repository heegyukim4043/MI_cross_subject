"""Concerns 1-2 of the critical review (exploratory, development data, CSP-LDA, 21 channels, relative SubjectEA).

Part A (dataset-weighted inference): for each component, per-dataset means with bootstrap CIs, the equal-weight
average of the four dataset means (bootstrap resampling subjects within each dataset), its range, and the
subject-pooled mean for comparison. Input: rq1 table built by rq_reaggregate.load_table().

Part B (size-matched LOSO vs LODO): the cross-dataset-specific loss LOSO - LODO confounds dataset membership with
the amount of source data. For each held-out dataset H and each target subject, models are refitted with the same
covariance-based CSP-LDA (run_dev.Weighted; identical to ManualCSP features, z-scoring omitted for all conditions):
  loso_full            all other subjects of H
  loso_half            random half of the other subjects of H (5 draws per target)
  lodo_full            all subjects of the other three datasets
  lodo_match_full      random subjects of the other datasets with total trials matched to loso_full (5 draws)
  lodo_match_half      same, matched to loso_half (5 draws)
Target trials: all trials of the subject (as in the decomposition), relative SubjectEA (transductive, label-free).
Outputs: rq1_dataset_weighted.csv, size_matched_subject_results.csv, size_matched_summary.csv.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cmpb_method_A_budget_fusion"))
import rq_reaggregate as rq  # noqa: E402
import run_dev as rd  # noqa: E402

ORDER = rq.ORDER
DRAWS = 5
RNG = np.random.default_rng(20260929)


def part_a():
    t = rq.load_table()
    rows = []
    for lab, (a, b) in rq.COMPONENTS.items():
        per = {n: (t[t.dataset == n][a] - t[t.dataset == n][b]).to_numpy() for n in ORDER}
        means = {n: v.mean() for n, v in per.items()}
        boots = np.mean([RNG.choice(v, (10000, len(v))).mean(1) for v in per.values()], axis=0)
        pooled = np.concatenate(list(per.values()))
        r = {"component": lab, **{f"{n}": means[n] for n in ORDER},
             "equal_weight_mean": np.mean(list(means.values())),
             "ew_ci_low": np.quantile(boots, 0.025), "ew_ci_high": np.quantile(boots, 0.975),
             "range_low": min(means.values()), "range_high": max(means.values()),
             "n_datasets_positive_ci": sum(np.quantile(RNG.choice(v, (5000, len(v))).mean(1), 0.025) > 0 for v in per.values()),
             "subject_pooled_mean": pooled.mean()}
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "rq1_dataset_weighted.csv", index=False)
    return out


def load_aligned():
    raw = {n: rd.load_npz(rd.PATHS[n]) for n in ORDER}
    shared = [c for c in raw[ORDER[0]]["ch_names"] if all(c in raw[n]["ch_names"] for n in ORDER)]
    data = {}
    for n in ORDER:
        d = raw[n]
        x = d["X"][:, [d["ch_names"].index(c) for c in shared], :rd.N_T].astype(np.float32)
        S = d["subjects"].astype(np.int64)
        xa = rd.align_subjects(x, S, "relative", n, "subject")[0]
        c, m = rd.covs(xa)
        data[n] = {"c": c, "m": m, "y": d["y"].astype(np.int64), "S": S}
    return data


def fit(parts):
    c = np.concatenate([p[0] for p in parts]); m = np.concatenate([p[1] for p in parts]); y = np.concatenate([p[2] for p in parts])
    return rd.Weighted(c, m, y, np.ones(len(y))), len(y)


def pick_subjects_by_trials(pool, target_trials, rng):
    """pool: list of (dataset, subject, n_trials). Random order, add subjects until the trial count is reached."""
    order = rng.permutation(len(pool)); chosen, total = [], 0
    for i in order:
        if total >= target_trials:
            break
        chosen.append(pool[i]); total += pool[i][2]
    return chosen


def subject_part(data, name, s):
    d = data[name]
    return (d["c"][d["S"] == s], d["m"][d["S"] == s], d["y"][d["S"] == s])


def one_target(data, h, s, lodo_models, seed):
    rng = np.random.default_rng(seed)
    d = data[h]; others = [x for x in np.unique(d["S"]) if x != s]
    tc, tm, ty = subject_part(data, h, s)
    rows = []
    full, n_full = fit([subject_part(data, h, o) for o in others])
    rows.append({"condition": "loso_full", "draw": 0, "n_src_trials": n_full, "n_src_subjects": len(others),
                 "bac": rd.bac(ty, full.decision(tc, tm))})
    for r in range(DRAWS):
        half = rng.choice(others, max(1, len(others) // 2), replace=False)
        mdl, n = fit([subject_part(data, h, o) for o in half])
        rows.append({"condition": "loso_half", "draw": r, "n_src_trials": n, "n_src_subjects": len(half),
                     "bac": rd.bac(ty, mdl.decision(tc, tm))})
    for cond, models in lodo_models.items():
        for r, (mdl, n, k) in enumerate(models):
            rows.append({"condition": cond, "draw": r, "n_src_trials": n, "n_src_subjects": k, "bac": rd.bac(ty, mdl.decision(tc, tm))})
    return [{"held_out": h, "subject": int(s), **r} for r in rows]


def part_b():
    data = load_aligned()
    out = []
    for h in ORDER:
        t0 = time.perf_counter()
        src = [n for n in ORDER if n != h]
        pool = [(n, int(x), int((data[n]["S"] == x).sum())) for n in src for x in np.unique(data[n]["S"])]
        per_subj = int(np.median([(data[h]["S"] == x).sum() for x in np.unique(data[h]["S"])]))
        loso_full_trials = per_subj * (len(np.unique(data[h]["S"])) - 1)
        lodo = {"lodo_full": [fit([subject_part(data, n, x) for n, x, _ in pool]) + (len(pool),)]}
        for cond, target in (("lodo_match_full", loso_full_trials), ("lodo_match_half", loso_full_trials // 2)):
            lodo[cond] = []
            for r in range(DRAWS):
                chosen = pick_subjects_by_trials(pool, target, RNG)
                lodo[cond].append(fit([subject_part(data, n, x) for n, x, _ in chosen]) + (len(chosen),))
        res = Parallel(n_jobs=6, backend="loky")(
            delayed(one_target)(data, h, s, lodo, 1000 * ORDER.index(h) + int(s)) for s in np.unique(data[h]["S"]))
        out += [r for rr in res for r in rr]
        pd.DataFrame(out).to_csv(HERE / "size_matched_subject_results.csv", index=False)
        print(f"[size-matched] {h} done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(out)
    ps = df.groupby(["held_out", "subject", "condition"]).agg(bac=("bac", "mean"), n=("n_src_trials", "mean")).reset_index()
    piv = ps.pivot_table(index=["held_out", "subject"], columns="condition", values="bac") * 100
    size = ps.groupby(["held_out", "condition"]).n.mean().unstack()
    rows = []
    for h in ORDER + ["equal_weight"]:
        if h == "equal_weight":
            continue
        g = piv.loc[h]
        r = {"held_out": h, "n_subjects": len(g)}
        for c in ("loso_full", "loso_half", "lodo_full", "lodo_match_full", "lodo_match_half"):
            r[f"{c} BAC"] = g[c].mean(); r[f"{c} src trials"] = size.loc[h, c]
        for lab, a, b in (("unmatched (loso_full - lodo_full)", "loso_full", "lodo_full"),
                          ("matched full (loso_full - lodo_match_full)", "loso_full", "lodo_match_full"),
                          ("matched half (loso_half - lodo_match_half)", "loso_half", "lodo_match_half")):
            dd = (g[a] - g[b]).to_numpy(); bt = RNG.choice(dd, (10000, len(dd))).mean(1)
            r[lab] = f"{dd.mean():+.2f} [{np.quantile(bt, 0.025):+.2f}, {np.quantile(bt, 0.975):+.2f}]"
        rows.append(r)
    summ = pd.DataFrame(rows); summ.to_csv(HERE / "size_matched_summary.csv", index=False)
    return summ


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    a = part_a(); print(a.round(2).to_string(index=False), flush=True)
    b = part_b(); print(b.round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
