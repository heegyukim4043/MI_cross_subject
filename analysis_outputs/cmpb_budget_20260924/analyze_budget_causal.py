"""Summarize budget / causal / ch27 / Cho-LOSO results with paired subject-level statistics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(os.environ.get("BUDGET_DIR", Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cmpb_floor_gate_20260924"))
from analyze_floor_gate import holm_adjust, paired_summary  # noqa: E402

RNG = np.random.default_rng(20260924)


def md(df) -> str:
    return "```\n" + df.to_string() + "\n```"


def contrast(a: pd.Series, b: pd.Series, label: dict) -> dict:
    """a - b, paired on the shared subject index (BAC in percentage points)."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    diff = (joined["a"] - joined["b"]).to_numpy() * 100
    return {**label, "mean_a": joined["a"].mean() * 100, "mean_b": joined["b"].mean() * 100,
            **paired_summary(diff, RNG)}


def budget(out: list[str]) -> list[dict]:
    b = pd.read_csv(HERE / "budget_subject_results.csv")
    # average random draws within subject first
    per = b.groupby(["scope", "source", "target", "condition", "n_ref", "subject"]).bac.mean().reset_index()
    table = per.groupby(["scope", "source", "target", "condition", "n_ref"]).bac.agg(["mean", "std", "count"])
    out += ["## Unlabeled target-trial budget (BAC on fixed 100-trial evaluation set)", "",
            md((table * [100, 100, 1]).round(2)), ""]
    rows = []
    for (scope, src, tgt), g in per.groupby(["scope", "source", "target"]):
        pv = g.pivot_table(index="subject", columns=["condition", "n_ref"], values="bac")
        full = pv[("transductive_all200", 200)]
        none = pv[("no_alignment", 0)]
        for n in (5, 10, 20, 40, 80, 100):
            col = pv[("pool_budget", n)]
            lab = {"scope": scope, "direction": f"{src}->{tgt}", "n_ref": n}
            rows.append(contrast(col, full, {**lab, "contrast": "budget n minus transductive all-200"}))
            rows.append(contrast(col, none, {**lab, "contrast": "budget n minus no alignment"}))
    return rows


def causal(out: list[str]) -> list[dict]:
    c = pd.read_csv(HERE / "causal_subject_results.csv")
    per = c.groupby(["scope", "source", "target", "order", "burn_in", "condition", "subject"]).bac.mean().reset_index()
    table = per.groupby(["scope", "source", "target", "order", "burn_in", "condition"]).bac.mean().unstack()
    out += ["## Causal online SubjectEA vs batch references (BAC, trials at/after burn-in)", "",
            "Lee2019 order is chronological within session; Cho2017 uses random permutations (not causal in time).", "",
            md((table * 100).round(2)), ""]
    rows = []
    for (scope, src, tgt, order, b), g in per.groupby(["scope", "source", "target", "order", "burn_in"]):
        pv = g.pivot_table(index="subject", columns="condition", values="bac")
        lab = {"scope": scope, "direction": f"{src}->{tgt}", "order": order, "burn_in": b}
        rows.append(contrast(pv["online_reset"], pv["batch_session"], {**lab, "contrast": "online_reset minus batch_session"}))
        rows.append(contrast(pv["online_reset"], pv["batch_subject"], {**lab, "contrast": "online_reset minus batch_subject"}))
    return rows


def ch27(out: list[str]) -> list[dict]:
    h = pd.read_csv(HERE / "ch27_subject_results.csv")
    h["method"] = h["floor_mode"] + ":" + h["alignment"]
    out += ["## standard_mi 27-channel sensitivity (cross-dataset CSP-LDA BAC)", "",
            md((h.groupby(["source", "target", "method"]).bac.mean().unstack() * 100).round(2)), ""]
    rows = []
    for (src, tgt), g in h.groupby(["source", "target"]):
        pv = g.pivot_table(index="subject", columns="method", values="bac")
        lab = {"scope": "ch27", "direction": f"{src}->{tgt}"}
        rows.append(contrast(pv["relative:dataset_subject"], pv["relative:subject"], {**lab, "contrast": "corrected D->S minus S"}))
        rows.append(contrast(pv["legacy:dataset_subject"], pv["legacy:subject"], {**lab, "contrast": "legacy D->S minus S"}))
        rows.append(contrast(pv["relative:subject"], pv["none:none"], {**lab, "contrast": "corrected S minus none"}))
    return rows


def cho_loso(out: list[str]) -> list[dict]:
    path = HERE / "cho_loso_floor" / "loso_subject_results.csv"
    if not path.exists():
        return []
    l = pd.read_csv(path)
    out += ["## Cho2017 LOSO SubjectEA by floor policy (CSP-LDA BAC)", "",
            md((l.groupby("floor_mode").bac.agg(["mean", "std"]) * 100).round(2)), ""]
    pv = l.pivot_table(index="subject", columns="floor_mode", values="bac")
    return [contrast(pv["relative"], pv["legacy"], {"scope": "loso", "direction": "cho2017", "contrast": "relative minus legacy"})]


def main() -> int:
    out: list[str] = ["# Budget / causal / sensitivity results", "",
                      "Exploratory, CSP-LDA, relative-floor EA unless stated. Paired by subject; 95% CI from paired "
                      "bootstrap of the mean difference; Wilcoxon p Holm-adjusted within each family.", ""]
    for name, fn in (("budget", budget), ("causal", causal), ("ch27", ch27), ("cho_loso", cho_loso)):
        try:
            rows = fn(out)
        except FileNotFoundError as err:
            out += [f"_{name}: missing input ({err.filename})_", ""]
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["p_holm"] = holm_adjust(df["p_raw"].tolist())
        df.to_csv(HERE / f"{name}_contrasts.csv", index=False)
        out += [f"### {name} contrasts", "", md(df.round(4).set_index("contrast")), ""]
    (HERE / "BUDGET_CAUSAL_REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
