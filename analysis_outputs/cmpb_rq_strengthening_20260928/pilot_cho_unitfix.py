"""Pilot: effect of correcting the Cho2017 unit (released values interpreted as 31.25-nV BioSemi ADC steps, so MOABB's
volts are about 32x too large; correction factor 0.03125) on the LODO4 and Cho LOSO results.

Only Cho2017 is rescaled; the other datasets are unchanged. CSP-LDA and TS-LR (as in the unit intervention),
21 channels, conditions: no alignment, absolute (legacy) and relative floor, SubjectEA and DatasetEA->SubjectEA.
Expected from theory: relative results identical; no-alignment LODO and absolute-floor results may change.
Outputs: pilot_cho_unitfix_subject_results.csv and a summary table."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cmpb_unit_intervention_20260925"))
import run_unit_intervention as ui  # noqa: E402

FACTOR = 0.03125
CONDS = [("none", "none"), ("legacy", "subject"), ("legacy", "dataset_subject"), ("relative", "subject"), ("relative", "dataset_subject")]


def align(x, S, floor, alignment, name):
    if floor == "none":
        return x
    return ui.apply_alignment(x, S, floor, alignment, name)[0]


def main():
    data, _ = ui.load_inputs(False)
    # Start from the MOABB (uncorrected) scale whichever version is in place: the corrected file has a median |x| of
    # about 2e-6 V, the uncorrected one about 32 times more. Division by 0.03125 is exact in float32.
    if float(np.median(np.abs(data["cho2017"]["X"]))) < 1e-5:
        data["cho2017"] = {**data["cho2017"], "X": (data["cho2017"]["X"] / np.float32(FACTOR)).astype(np.float32)}
    rows = []
    with ui.threadpool_limits(limits=6):
        for version, f in (("original", 1.0), ("cho_corrected", FACTOR)):
            scaled = {n: {**d, "X": (d["X"] * np.float32(f if n == "cho2017" else 1.0)).astype(np.float32)} for n, d in data.items()}
            for floor, alignment in CONDS:
                al = {n: align(d["X"], d["subjects"], floor, alignment, n) for n, d in scaled.items()}
                cov = {n: ui.Covariances("oas").fit_transform(v.astype(np.float64)) for n, v in al.items()}
                for h in ui.NAMES:
                    src = [n for n in ui.NAMES if n != h]
                    xs = np.concatenate([al[n] for n in src]); ys = np.concatenate([data[n]["y"] for n in src])
                    mu, sd = ui.norm_stats(xs)
                    csp, sc, lda = ui.fit_csp_lda(ui.ManualCSP, ((xs - mu) / sd).astype(np.float32), ys, 8)
                    ctx = {"version": version, "floor": floor, "alignment": alignment, "design": "LODO", "held_out": h}
                    rows += ui.subject_rows({**ctx, "classifier": "CSP-LDA"}, data[h]["y"],
                                            ui.predict(csp, sc, lda, ((al[h] - mu) / sd).astype(np.float32)), data[h]["subjects"])
                    m = ui.make_pipeline(ui.TangentSpace(metric="riemann"), ui.LogisticRegression(C=1.0, max_iter=3000))
                    m.fit(np.concatenate([cov[n] for n in src]), ys)
                    rows += ui.subject_rows({**ctx, "classifier": "TS-LR"}, data[h]["y"], m.predict(cov[h]), data[h]["subjects"])
                # Cho LOSO (CSP-LDA) for no alignment and SubjectEA
                if alignment in ("none", "subject"):
                    x, y, S = al["cho2017"], data["cho2017"]["y"], data["cho2017"]["subjects"]
                    for s in np.unique(S):
                        tr = S != s
                        mu, sd = ui.norm_stats(x[tr])
                        csp, sc, lda = ui.fit_csp_lda(ui.ManualCSP, ((x[tr] - mu) / sd).astype(np.float32), y[tr], 8)
                        rows += ui.subject_rows({"version": version, "floor": floor, "alignment": alignment, "design": "LOSO",
                                                 "held_out": "cho2017", "classifier": "CSP-LDA"}, y[~tr],
                                                ui.predict(csp, sc, lda, ((x[~tr] - mu) / sd).astype(np.float32)), S[~tr])
                print(f"done {version} {floor} {alignment}", flush=True)
    d = pd.DataFrame(rows); d["bac"] *= 100
    d.to_csv(HERE / "pilot_cho_unitfix_subject_results.csv", index=False)
    t = d.groupby(["design", "classifier", "floor", "alignment", "held_out", "version"]).bac.mean().unstack("version")
    t["change"] = t["cho_corrected"] - t["original"]
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print(t.round(2).to_string())
    pooled = d[d.design == "LODO"].groupby(["classifier", "floor", "alignment", "version"]).bac.mean().unstack("version")
    pooled["change"] = pooled["cho_corrected"] - pooled["original"]
    print("\nLODO pooled over 224 subjects:\n", pooled.round(2).to_string())


if __name__ == "__main__":
    main()
