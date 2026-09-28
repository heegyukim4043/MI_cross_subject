"""RQ2 robustness: covariance regularization strategies for SubjectEA under amplitude rescaling (exploratory).

Same data, channels (21), LODO design, classifiers (CSP-LDA, TS-LR) and scales (x1, x1e-3, x1e-6) as the unit
intervention (`cmpb_unit_intervention_20260925/run_unit_intervention.py`, whose helpers are reused). Only the way
the subject reference covariance is regularized before its inverse square root changes:
  legacy          float32 sample mean covariance, eigenvalues floored at an absolute 1e-8 (original code)
  absolute64      float64, absolute floor 1e-8
  relative        float64, floor 1e-8 * trace / C (corrected code)
  trace_loading   (R + 1e-3 * trace(R) / C * I)^(-1/2)
  ledoit_wolf     mean of per-trial Ledoit-Wolf covariances, no floor
  oas             mean of per-trial OAS covariances, no floor
  rank_aware      truncated inverse, eigenvalues below 1e-12 * max are dropped
  unregularized   plain inverse square root of the sample mean covariance (float64)
One resumable cell per (regularizer, scale): results/cells/<key>.csv with subject-level metrics.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.covariance import ledoit_wolf, oas

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path.insert(0, str(A / "cmpb_unit_intervention_20260925"))
import run_unit_intervention as ui  # noqa: E402

REGS = ("legacy", "absolute64", "relative", "trace_loading", "ledoit_wolf", "oas", "rank_aware", "unregularized")
OUT = HERE / "regularizers"


def reference(x, reg):
    if reg == "legacy":
        xx = x.astype(np.float32)
        return np.mean(np.einsum("nct,ndt->ncd", xx, xx), axis=0)
    if reg in ("ledoit_wolf", "oas"):
        est = ledoit_wolf if reg == "ledoit_wolf" else oas
        return np.mean([est(trial.T.astype(np.float64), assume_centered=True)[0] for trial in x], axis=0)
    xx = x.astype(np.float64)
    return np.mean(np.einsum("nct,ndt->ncd", xx, xx), axis=0)


def whitener(r, reg):
    r = (r + r.T) * 0.5
    c = len(r)
    if reg == "trace_loading":
        r = r + 1e-3 * np.trace(r) / c * np.eye(c)
    ev, vec = np.linalg.eigh(r.astype(np.float64) if reg != "legacy" else r)
    if reg in ("legacy", "absolute64"):
        inv = np.maximum(ev, 1e-8) ** -0.5
    elif reg == "relative":
        inv = np.maximum(ev, 1e-8 * np.trace(r) / c) ** -0.5
    elif reg == "rank_aware":
        keep = ev > 1e-12 * max(ev[-1], 0.0)
        inv = np.zeros_like(ev); inv[keep] = ev[keep] ** -0.5
    else:  # trace_loading, ledoit_wolf, oas, unregularized
        if np.any(ev <= 0):
            raise FloatingPointError(f"{reg}: non-positive eigenvalue {ev.min():.3g}")
        inv = ev ** -0.5
    return (vec * inv) @ vec.T, int((ev < 1e-8).sum())


def align(x, subjects, reg):
    out = np.empty_like(x, dtype=np.float32)
    floored = []
    for s in np.unique(subjects):
        m = subjects == s
        w, nf = whitener(reference(x[m], reg), reg)
        out[m] = np.einsum("cd,ndt->nct", w, x[m], optimize=True).astype(np.float32)
        floored.append(nf)
    return out, floored


def run_cell(data, reg, scale_label, scale):
    key = f"reg-{reg}_scale-{scale_label}"
    path = OUT / "cells" / f"{key}.csv"
    if path.exists():
        print(f"[skip] {key}", flush=True); return
    t0 = time.perf_counter()
    aligned = {}
    for name, item in data.items():
        aligned[name], _ = align((item["X"] * np.float32(scale)).astype(np.float32), item["subjects"], reg)
    cov = {n: ui.Covariances("oas").fit_transform(v.astype(np.float64)) for n, v in aligned.items()}
    rows = []
    for held in ui.NAMES:
        src = [n for n in ui.NAMES if n != held]
        xs = np.concatenate([aligned[n] for n in src]); ys = np.concatenate([data[n]["y"] for n in src])
        xt, yt, st = aligned[held], data[held]["y"], data[held]["subjects"]
        mu, sd = ui.norm_stats(xs)
        csp, sc, lda = ui.fit_csp_lda(ui.ManualCSP, ((xs - mu) / sd).astype(np.float32), ys, 8)
        ctx = {"regularizer": reg, "scale_label": scale_label, "scale": scale, "held_out": held, "classifier": "CSP-LDA"}
        rows += ui.subject_rows(ctx, yt, ui.predict(csp, sc, lda, ((xt - mu) / sd).astype(np.float32)), st)
        model = ui.make_pipeline(ui.TangentSpace(metric="riemann"), ui.LogisticRegression(C=1.0, max_iter=3000))
        model.fit(np.concatenate([cov[n] for n in src]), ys)
        rows += ui.subject_rows({**ctx, "classifier": "TS-LR"}, yt, model.predict(cov[held]), st)
    ui.atomic_csv(path, rows)
    print(f"[done] {key} {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global OUT
    if args.smoke:
        OUT = HERE / "regularizers_smoke"
    data, shared = ui.load_inputs(args.smoke)
    print(f"regularizer comparison: {len(shared)} channels, {sum(len(d['y']) for d in data.values())} trials", flush=True)
    with ui.threadpool_limits(limits=args.threads):
        for scale_label, scale in ui.SCALES:
            for reg in REGS:
                try:
                    run_cell(data, reg, scale_label, scale)
                except FloatingPointError as err:
                    print(f"[fail] reg-{reg}_scale-{scale_label}: {err}", flush=True)


if __name__ == "__main__":
    main()
