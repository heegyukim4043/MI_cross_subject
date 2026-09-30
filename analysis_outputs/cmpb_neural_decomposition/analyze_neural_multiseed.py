"""Neural (CSPNet) LODO4 multi-seed summary (development data; exploratory except where noted in the paper).

Inputs: MI_test/results/generic_gen_lodo4_{cond}[_s{seed}]_lodo.csv for cond in none, legacy, relative, dsea_relative
and seeds 2026 (no suffix), 1, 2, 3, 4; optional adabn_relative (seed 2026 and any finished seeds).
Per subject, BAC is first averaged over the seeds available for every condition in a contrast. Contrasts:
  EA gain            relative - none
  numerical artifact relative - legacy
  DatasetEA increment dsea_relative - relative   (90% CI and TOST +/-1 pp reported; margin post hoc)
  AdaBN              adabn_relative - relative   (seeds with AdaBN results only)
Reported per held-out dataset, subject-pooled and equal-weight (mean of dataset means; bootstrap within dataset),
plus the between-seed SD of the pooled contrast. Outputs: neural_multiseed_summary.csv, neural_multiseed_per_seed.csv.
--chofix: use the Cho2017 unit-corrected runs (*_chofix) for none, legacy and dsea_legacy (relative-floor runs are
unchanged by the correction) and write the outputs with a _chofix suffix; the original outputs are kept.
"""
import sys

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "MI_test" / "results"
ORDER = ["cho2017", "lee2019", "bnci2014001", "physionetmi"]
SEEDS = ["2026", "1", "2", "3", "4"]
RNG = np.random.default_rng(20260929)
CHOFIX = "--chofix" in sys.argv
SUFFIX = "_chofix" if CHOFIX else ""
SCALE_DEPENDENT = ("none", "legacy", "dsea_legacy")  # conditions that depend on Cho's absolute scale


def load(cond, seed):
    tag = cond + ("_chofix" if CHOFIX and cond in SCALE_DEPENDENT else "")
    f = RES / f"generic_gen_lodo4_{tag}{'' if seed == '2026' else '_s' + seed}_lodo.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d.drop_duplicates(["held_out", "subject"]).set_index(["held_out", "subject"]).bac * 100


def complete(s):
    return s is not None and len(s) == 224


def contrast_rows(label, a, b):
    seeds = [sd for sd in SEEDS if complete(load(a, sd)) and complete(load(b, sd))]
    if not seeds:
        return [], []
    diffs = pd.concat([(load(a, sd) - load(b, sd)).rename(sd) for sd in seeds], axis=1)
    per_seed = [{"contrast": label, "seed": sd, "pooled_pp": diffs[sd].mean(),
                 "equal_weight_pp": diffs[sd].groupby(level=0).mean().mean()} for sd in seeds]
    d = diffs.mean(axis=1)  # seed-averaged per subject
    rows = []
    groups = {n: d.loc[n].to_numpy() for n in ORDER}
    for n, v in list(groups.items()) + [("pooled", d.to_numpy())]:
        b = RNG.choice(v, (10000, len(v))).mean(1)
        rows.append({"contrast": label, "seeds": ",".join(seeds), "scope": n, "n": len(v), "mean_pp": v.mean(),
                     "ci95_low": np.quantile(b, .025), "ci95_high": np.quantile(b, .975),
                     "ci90_low": np.quantile(b, .05), "ci90_high": np.quantile(b, .95),
                     "wilcoxon_p": stats.wilcoxon(v).pvalue if len(v) > 5 and np.any(v != 0) else np.nan,
                     "improved_share": (v > 0).mean() * 100})
    ew = np.mean([RNG.choice(v, (10000, len(v))).mean(1) for v in groups.values()], axis=0)
    rows.append({"contrast": label, "seeds": ",".join(seeds), "scope": "equal_weight", "n": len(d),
                 "mean_pp": np.mean([v.mean() for v in groups.values()]),
                 "ci95_low": np.quantile(ew, .025), "ci95_high": np.quantile(ew, .975),
                 "ci90_low": np.quantile(ew, .05), "ci90_high": np.quantile(ew, .95)})
    ps = pd.DataFrame(per_seed)
    rows.append({"contrast": label, "seeds": ",".join(seeds), "scope": "between_seed_sd_of_pooled",
                 "n": len(seeds), "mean_pp": ps.pooled_pp.std(ddof=1) if len(seeds) > 1 else np.nan})
    return rows, per_seed


def main():
    rows, per_seed = [], []
    for label, a, b in (("EA gain (relative - none)", "relative", "none"),
                        ("numerical artifact (relative - legacy)", "relative", "legacy"),
                        ("DatasetEA increment (dsea_relative - relative)", "dsea_relative", "relative"),
                        ("AdaBN (adabn_relative - relative)", "adabn_relative", "relative"),
                        ("legacy DatasetEA increment (dsea_legacy - legacy)", "dsea_legacy", "legacy")):
        r, p = contrast_rows(label, a, b); rows += r; per_seed += p
    level = []
    for cond in ("none", "legacy", "relative", "dsea_relative", "adabn_relative", "dsea_legacy"):
        seeds = [sd for sd in SEEDS if complete(load(cond, sd))]
        if seeds:
            m = pd.concat([load(cond, sd) for sd in seeds], axis=1).mean(axis=1)
            level.append({"condition": cond, "seeds": ",".join(seeds), "pooled_bac": m.mean(),
                          **{n: m.loc[n].mean() for n in ORDER}})
    out = pd.DataFrame(rows); out.to_csv(HERE / f"neural_multiseed_summary{SUFFIX}.csv", index=False)
    pd.DataFrame(per_seed).to_csv(HERE / f"neural_multiseed_per_seed{SUFFIX}.csv", index=False)
    pd.DataFrame(level).to_csv(HERE / f"neural_multiseed_levels{SUFFIX}.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(pd.DataFrame(level).round(2).to_string(index=False)); print()
    print(out[["contrast", "seeds", "scope", "n", "mean_pp", "ci95_low", "ci95_high", "ci90_low", "ci90_high", "wilcoxon_p"]]
          .round(3).to_string(index=False)); print()
    print(pd.DataFrame(per_seed).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
