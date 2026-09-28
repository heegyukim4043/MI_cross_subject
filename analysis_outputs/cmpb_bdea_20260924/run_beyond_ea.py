"""Does anything beyond scale-correct re-centering reduce the remaining cross-dataset shift?

Riemannian pipeline on trial covariances (OAS), per-subject re-centering, then optional steps:
  ea_center        re-centre each subject by its arithmetic mean covariance (= scale-correct EA)
  ra_center        re-centre each subject by its Riemannian mean
  ra_stretch       + stretch every subject to unit Riemannian dispersion around I (label-free, RPA step 2)
  ra_rotP1         + rotate each target subject so its pseudo-label class means match the pooled source
                     class means (RPA step 3 with pseudo-labels from the source classifier), 1 iteration
  ra_rotP3         same, 3 self-training iterations (a simple conditional-alignment variant)
  ra_stretch_rotP1 stretch + pseudo-label rotation
  ra_stretch_rotO  stretch + rotation from TRUE target labels (oracle upper bound, not deployable)
Classifiers trained on source subjects only: tangent space at the source mean + logistic regression
(TS-LR) and MDM. Target statistics are batch/transductive per subject, as in the main analyses.
Settings: Cho2017<->Lee2019 and 3-dataset LODO, all on the 21 channels shared with BNCI2014-001 (Cho, Lee, BNCI; 21 shared channels).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent / "cmpb_floor_gate_20260924", HERE):
    sys.path.insert(0, str(p))
from run_floor_gate import metrics, write_rows  # noqa: E402
from run_bdea_bnci import load_npz  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.utils.base import expm, invsqrtm, logm, powm  # noqa: E402
from pyriemann.utils.mean import mean_riemann  # noqa: E402
from pyriemann.utils.distance import distance_riemann  # noqa: E402
from pyriemann.transfer._rotate import _get_rotation_matrix  # noqa: E402
from pyriemann.classification import MDM  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

PATHS = {"cho2017": ROOT / "MI_test/preprocessed_sfreq100/cho2017.npz",
         "lee2019": ROOT / "MI_test/preprocessed_sfreq100/lee2019.npz",
         "bnci2014001": HERE / "bnci2014001.npz",
         "physionetmi": HERE / "physionetmi.npz"}
CONDS = ("ea_center", "ra_center", "ra_stretch", "ra_rotP1", "ra_rotP3", "ra_stretch_rotP1", "ra_stretch_rotO")


def load_set(names, channels=None):
    data = {n: load_npz(PATHS[n]) for n in names}
    shared = channels or [c for c in data[names[0]]["ch_names"] if all(c in data[n]["ch_names"] for n in names)]
    n_t = min(d["X"].shape[2] for d in data.values())
    out = {}
    for n, d in data.items():
        idx = [d["ch_names"].index(c) for c in shared]
        out[n] = {"C": Covariances("oas").fit_transform(d["X"][:, idx, :n_t].astype(np.float64)),
                  "y": d["y"].astype(np.int64), "subjects": d["subjects"].astype(np.int64)}
    return out, shared


def center(c, subjects, how):
    out = np.empty_like(c)
    for s in np.unique(subjects):
        m = subjects == s
        ref = c[m].mean(axis=0) if how == "ea" else mean_riemann(c[m])
        w = invsqrtm(ref)
        out[m] = w @ c[m] @ w
    return out


def stretch(c, subjects):
    out = np.empty_like(c)
    for s in np.unique(subjects):
        m = subjects == s
        disp = np.mean([distance_riemann(x, np.eye(len(x))) ** 2 for x in c[m]])
        p = 1.0 / np.sqrt(disp)
        out[m] = np.stack([powm(x, p) for x in c[m]])
    return out


def rotate_to(c_t, labels_t, source_means):
    m_t = np.stack([mean_riemann(c_t[labels_t == k]) for k in (0, 1)])
    # the loss is converged by 200 iterations on these data (checked on 3 subjects); the default 10,000 cap only costs time
    q = _get_rotation_matrix(m_t, source_means, metric="riemann", maxiter=500)
    return np.einsum("ij,njk,lk->nil", q, c_t, q)


def fit_models(c, y):
    return {"TS-LR": make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=3000)).fit(c, y),
            "MDM": MDM(metric="riemann").fit(c, y)}


def _rotation_rows(ctx, base, model, src_means, x, y, s):
    """Rotation conditions for one target subject (TS-LR only; run in a worker process)."""
    with threadpool_limits(limits=1):
        out = []
        if base == "ra_center":
            xr = x
            for it in range(3):
                xr = rotate_to(xr, model.predict(xr), src_means)
                if it in (0, 2):
                    out.append({**ctx, "classifier": "TS-LR", "condition": f"ra_rotP{it + 1}", "subject": int(s),
                                **metrics(y, model.predict(xr))})
        else:
            xr = rotate_to(x, model.predict(x), src_means)
            out.append({**ctx, "classifier": "TS-LR", "condition": "ra_stretch_rotP1", "subject": int(s), **metrics(y, model.predict(xr))})
            xo = rotate_to(x, y, src_means)
            out.append({**ctx, "classifier": "TS-LR", "condition": "ra_stretch_rotO", "subject": int(s), **metrics(y, model.predict(xo))})
        return out


def evaluate(ctx, src, tgt, rows):
    """src/tgt: dict with C, y, subjects (already concatenated for multi-source).
    Re-centering / stretch baselines use TS-LR and MDM; rotation conditions use TS-LR only and run in
    parallel over target subjects."""
    from joblib import Parallel, delayed
    prepared = {how: (center(src["C"], src["subjects"], how.split("_")[0]), center(tgt["C"], tgt["subjects"], how.split("_")[0]))
                for how in ("ea_center", "ra_center")}
    prepared["ra_stretch"] = (stretch(prepared["ra_center"][0], src["subjects"]), stretch(prepared["ra_center"][1], tgt["subjects"]))
    subjects = np.unique(tgt["subjects"])
    for base in ("ea_center", "ra_center", "ra_stretch"):
        cs, ct = prepared[base]
        models = fit_models(cs, src["y"])
        for mname, model in models.items():
            for s in subjects:
                m = tgt["subjects"] == s
                rows.append({**ctx, "classifier": mname, "condition": base, "subject": int(s), **metrics(tgt["y"][m], model.predict(ct[m]))})
        if base == "ea_center":
            continue
        src_means = np.stack([mean_riemann(cs[src["y"] == k]) for k in (0, 1)])
        jobs = (delayed(_rotation_rows)(ctx, base, models["TS-LR"], src_means, ct[tgt["subjects"] == s], tgt["y"][tgt["subjects"] == s], s)
                for s in subjects)
        for out in Parallel(n_jobs=12)(jobs):
            rows.extend(out)


def concat(parts):
    off, subj = 0, []
    for p in parts:  # make subject ids unique across datasets
        subj.append(p["subjects"] + off); off = int(subj[-1].max()) + 1000
    return {"C": np.concatenate([p["C"] for p in parts]), "y": np.concatenate([p["y"] for p in parts]),
            "subjects": np.concatenate(subj)}


def main() -> int:
    smoke = "--smoke" in sys.argv
    rows = []
    with threadpool_limits(limits=8):
        # 48x48 RPA rotation takes ~55 s per call; all settings use the 21 channels shared with BNCI2014-001
        _, shared21 = load_set(["cho2017", "lee2019", "bnci2014001"])
        pair, _ = load_set(["cho2017", "lee2019"], channels=shared21)
        if smoke:
            pair = {k: {kk: vv[np.isin(v["subjects"], np.unique(v["subjects"])[:3])] for kk, vv in v.items()} for k, v in pair.items()}
        for s, t in (("cho2017", "lee2019"), ("lee2019", "cho2017")):
            t0 = time.perf_counter()
            evaluate({"setting": "pair21", "source": s, "target": t}, pair[s], pair[t], rows)
            write_rows(HERE / "beyond_ea_subject_results.csv", rows)
            print(f"[beyond] pair21 {s}->{t} {(time.perf_counter()-t0)/60:.1f} min", flush=True)
        if not smoke:
            names = ["cho2017", "lee2019", "bnci2014001"]
            three, shared = load_set(names)
            for held in names:
                t0 = time.perf_counter()
                evaluate({"setting": f"lodo3_{len(shared)}ch", "source": "+".join(n for n in names if n != held), "target": held},
                         concat([three[n] for n in names if n != held]), three[held], rows)
                write_rows(HERE / "beyond_ea_subject_results.csv", rows)
                print(f"[beyond] lodo3 held-out {held} {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print((df.groupby(["setting", "target", "classifier", "condition"]).bac.mean().unstack("condition")[list(CONDS)] * 100).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
