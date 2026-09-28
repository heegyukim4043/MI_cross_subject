"""RQ2 negative control: affine-invariant Riemannian pipelines under the same amplitude rescaling (exploratory).

Same data, 21 channels, LODO4 design and scales (x1, x1e-3, x1e-6) as the unit intervention. Trial covariances are
estimated with OAS (scale-equivariant). Conditions:
  mdm            MDM, no alignment
  tslr           tangent space (Riemannian mean of the source) + logistic regression, no alignment
  ra_mdm         Riemannian re-centering per subject (C -> M^-1/2 C M^-1/2, M = subject Riemannian mean), then MDM
  ra_tslr        Riemannian re-centering per subject, then TS-LR
Expected: predictions identical across scales up to floating-point error (no absolute regularization is involved).
Outputs: riemann_controls/subject_results.csv, riemann_controls/predictions_<scale>.npz, and a summary printed at the end.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cmpb_unit_intervention_20260925"))
import run_unit_intervention as ui  # noqa: E402
from pyriemann.classification import MDM  # noqa: E402
from pyriemann.utils.base import invsqrtm  # noqa: E402
from pyriemann.utils.mean import mean_riemann  # noqa: E402

OUT = HERE / "riemann_controls"


def recenter(covs, subjects):
    out = np.empty_like(covs)
    for s in np.unique(subjects):
        m = subjects == s
        w = invsqrtm(mean_riemann(covs[m]))
        out[m] = w @ covs[m] @ w
    return out


def main():
    OUT.mkdir(exist_ok=True)
    data, shared = ui.load_inputs(False)
    rows, t0 = [], time.perf_counter()
    with ui.threadpool_limits(limits=4):
        for label, scale in ui.SCALES:
            covs = {n: ui.Covariances("oas").fit_transform((d["X"].astype(np.float64) * scale)) for n, d in data.items()}
            rc = {n: recenter(covs[n], data[n]["subjects"]) for n in covs}
            preds = {}
            for held in ui.NAMES:
                src = [n for n in ui.NAMES if n != held]
                ys = np.concatenate([data[n]["y"] for n in src]); yt, st = data[held]["y"], data[held]["subjects"]
                for cond, feats in (("mdm", covs), ("tslr", covs), ("ra_mdm", rc), ("ra_tslr", rc)):
                    xs = np.concatenate([feats[n] for n in src])
                    if cond.endswith("mdm"):
                        model = MDM(metric="riemann").fit(xs, ys)
                    else:
                        model = ui.make_pipeline(ui.TangentSpace(metric="riemann"), ui.LogisticRegression(C=1.0, max_iter=3000)).fit(xs, ys)
                    p = model.predict(feats[held]).astype(np.int8)
                    preds[f"{cond}__{held}"] = p
                    rows += ui.subject_rows({"scale_label": label, "condition": cond, "held_out": held}, yt, p, st)
                print(f"[riemann] scale {label} held {held} done ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
            np.savez_compressed(OUT / f"predictions_{label}.npz", **preds)
            pd.DataFrame(rows).to_csv(OUT / "subject_results.csv", index=False)
    ref = np.load(OUT / "predictions_1.npz")
    summ = []
    for label, _ in ui.SCALES[1:]:
        other = np.load(OUT / f"predictions_{label}.npz")
        for k in ref.files:
            cond, held = k.split("__")
            summ.append({"scale": label, "condition": cond, "held_out": held, "n": len(ref[k]),
                         "changed_predictions": int((ref[k] != other[k]).sum())})
    s = pd.DataFrame(summ); s.to_csv(OUT / "prediction_changes.csv", index=False)
    bac = pd.DataFrame(rows).groupby(["condition", "scale_label"]).bac.mean().unstack() * 100
    print(s.groupby(["condition", "scale"]).changed_predictions.sum().unstack().to_string())
    print(bac.round(2).to_string())


if __name__ == "__main__":
    main()
