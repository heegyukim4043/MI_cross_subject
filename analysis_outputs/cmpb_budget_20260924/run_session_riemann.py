"""CPU follow-ups (relative-floor EA, CSP-LDA unless stated).

session : SubjectEA vs SessionEA on the Lee2019 side (reference per session = trials 0-99 / 100-199,
          verified for subject 1 and implied by the deterministic loader order). Cho2017 has one
          session and always uses SubjectEA. Cross-dataset (Lee as target or as source) and Lee LOSO.
riemann : label-free Riemannian re-centering baselines. Trial covariances (OAS), each subject's
          matrices re-centred to identity by its own Riemannian mean (as pyRiemann TLCenter, no
          labels), then MDM or tangent-space logistic regression trained on source subjects.
          A no-re-centering variant is included. Cross-dataset both directions and LOSO both datasets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cmpb_floor_gate_20260924"))
sys.path.insert(0, str(HERE))
from run_floor_gate import align_subjects, fit_csp_lda, metrics, predict, write_rows  # noqa: E402
from run_budget_causal import apply_w, load, norm_stats, subset, whiten  # noqa: E402

MODE = "relative"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=HERE.parents[1])
    p.add_argument("--out", type=Path, default=HERE)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--parts", nargs="+", default=["session", "riemann"])
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def session_align(x: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Whiten each Lee2019 session (first/second half of each subject's trials) separately."""
    out = np.empty_like(x, dtype=np.float32)
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        for seg in np.array_split(idx, 2):
            out[seg] = apply_w(whiten(x[seg]), x[seg])
    return out


def aligned(data: dict, name: str, how: str) -> np.ndarray:
    if how == "session" and name == "lee2019":
        return session_align(data["X"], data["subjects"])
    return align_subjects(data["X"], data["subjects"], MODE, name, "subject")[0]


def csp_eval(manual_csp, xs, ys, xt, yt, st, ctx, rows):
    mu, sd = norm_stats(xs)
    csp, sc, clf = fit_csp_lda(manual_csp, ((xs - mu) / sd).astype(np.float32), ys, 8)
    xt = ((xt - mu) / sd).astype(np.float32)
    for s in np.unique(st):
        m = st == s
        rows.append({**ctx, "subject": int(s), **metrics(yt[m], predict(csp, sc, clf, xt[m]))})


def run_session(raw, manual_csp, out: Path):
    rows = []
    for source, target in (("cho2017", "lee2019"), ("lee2019", "cho2017")):
        for how in ("subject", "session"):
            src, tgt = raw[source], raw[target]
            ctx = {"scope": "cross", "source": source, "target": target, "lee_alignment": how}
            csp_eval(manual_csp, aligned(src, source, how), src["y"], aligned(tgt, target, how),
                     tgt["y"], tgt["subjects"], ctx, rows)
            write_rows(out / "session_subject_results.csv", rows)
            print(f"[session] {source}->{target} {how} done", flush=True)
    lee = raw["lee2019"]
    for how in ("subject", "session"):
        xa = aligned(lee, "lee2019", how)
        for s in np.unique(lee["subjects"]):
            tr = lee["subjects"] != s
            ctx = {"scope": "loso", "source": "lee2019", "target": "lee2019", "lee_alignment": how}
            csp_eval(manual_csp, xa[tr], lee["y"][tr], xa[~tr], lee["y"][~tr], lee["subjects"][~tr], ctx, rows)
        write_rows(out / "session_subject_results.csv", rows)
        print(f"[session] loso lee2019 {how} done", flush=True)


def covs(x: np.ndarray) -> np.ndarray:
    from pyriemann.estimation import Covariances
    return Covariances("oas").fit_transform(x.astype(np.float64))


def recenter(c: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann
    out = np.empty_like(c)
    for s in np.unique(subjects):
        m = subjects == s
        w = invsqrtm(mean_riemann(c[m]))
        out[m] = w @ c[m] @ w
    return out


def riemann_models():
    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return {"MDM": lambda: MDM(metric="riemann"),
            "TS-LR": lambda: make_pipeline(TangentSpace(metric="riemann"),
                                           LogisticRegression(C=1.0, max_iter=2000))}


def run_riemann(raw, out: Path):
    rows = []
    cov = {k: covs(v["X"]) for k, v in raw.items()}
    rec = {k: recenter(cov[k], raw[k]["subjects"]) for k in raw}
    models = riemann_models()
    for variant, mats in (("recentered", rec), ("raw", cov)):
        for source, target in (("cho2017", "lee2019"), ("lee2019", "cho2017")):
            for mname, make in models.items():
                t0 = time.perf_counter()
                clf = make().fit(mats[source], raw[source]["y"])
                st = raw[target]["subjects"]
                pred = clf.predict(mats[target])
                for s in np.unique(st):
                    m = st == s
                    rows.append({"scope": "cross", "source": source, "target": target, "variant": variant,
                                 "model": mname, "subject": int(s), **metrics(raw[target]["y"][m], pred[m])})
                write_rows(out / "riemann_subject_results.csv", rows)
                print(f"[riemann] {variant} {source}->{target} {mname} {(time.perf_counter()-t0)/60:.1f} min", flush=True)
        for dataset in ("cho2017", "lee2019"):
            subj, y = raw[dataset]["subjects"], raw[dataset]["y"]
            for mname, make in models.items():
                t0 = time.perf_counter()
                for s in np.unique(subj):
                    tr = subj != s
                    pred = make().fit(mats[dataset][tr], y[tr]).predict(mats[dataset][~tr])
                    rows.append({"scope": "loso", "source": dataset, "target": dataset, "variant": variant,
                                 "model": mname, "subject": int(s), **metrics(y[~tr], pred)})
                write_rows(out / "riemann_subject_results.csv", rows)
                print(f"[riemann] {variant} loso {dataset} {mname} {(time.perf_counter()-t0)/60:.1f} min", flush=True)


def main() -> int:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root.resolve() / "MI_test" / "MI_loso_project"))
    from mrfbcsp_loso import ManualCSP
    raw, _ = load(args.root.resolve(), "common")
    if args.smoke:
        raw = {k: subset(v, 3) for k, v in raw.items()}
    with threadpool_limits(limits=args.threads):
        if "session" in args.parts:
            run_session(raw, ManualCSP, out)
        if "riemann" in args.parts:
            run_riemann(raw, out)
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
