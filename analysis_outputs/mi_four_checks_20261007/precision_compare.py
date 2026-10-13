"""Precision-matched rerun using unchanged archived CSP-LDA definitions."""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits

NAMES = ('cho2017', 'lee2019', 'bnci2014001', 'physionetmi')
SCALES = (('1', 1.), ('1e-6', 1e-6))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


def helpers(root):
    ns = dict(np=np, StandardScaler=StandardScaler,
              LinearDiscriminantAnalysis=LinearDiscriminantAnalysis)
    definitions = {
        root / 'MI_test/MI_loso_project/mrfbcsp_loso.py': ['ManualCSP'],
        root / 'analysis_outputs/cmpb_floor_gate_20260924/run_floor_gate.py': ['fit_csp_lda'],
        root / 'analysis_outputs/cmpb_budget_20260924/run_budget_causal.py': ['norm_stats']}
    for p, names in definitions.items():
        tree = ast.parse(p.read_text(encoding='utf-8-sig'))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), ns)
    return ns, {str(p): sha(p) for p in definitions}


def batch_align(x, policy, epsilon):
    xx = x.astype(np.float64)
    r = np.mean(np.einsum('nct,ndt->ncd', xx, xx), axis=0, dtype=np.float64)
    r = (r + r.T) * .5
    ev, vec = np.linalg.eigh(r)
    if ev[0] <= 0 or not np.isfinite(ev).all():
        raise ValueError('Non-SPD reference; no silent repair')
    floor = 1e-8 * np.trace(r) / len(r) if policy == 'relative' else epsilon
    adjusted = np.maximum(ev, floor)
    w = (vec * adjusted ** -.5) @ vec.T
    z = np.einsum('cd,ndt->nct', w, x, optimize=True).astype(np.float32)
    diag = dict(floor=float(floor), clipped=int((ev < floor).sum()),
                whitening_AI=float(np.linalg.norm(np.log(ev / adjusted))),
                raw_condition=float(ev[-1] / ev[0]))
    return z, diag


def align(x, ids, policy, alignment, scale, contract='coherent'):
    z = (x * np.float32(scale)).astype(np.float32)
    first = 1e-8 * scale ** 2 if contract != 'fixed' else 1e-8
    second = first if contract == 'wrong_both' else 1e-8
    diags = []
    if alignment == 'dataset_subject':
        z, d = batch_align(z, policy, first)
        diags.append(dict(stage='dataset', subject=-1, **d))
    for s in np.unique(ids):
        m = ids == s
        z[m], d = batch_align(z[m], policy, second if alignment == 'dataset_subject' else first)
        diags.append(dict(stage='subject', subject=int(s), **d))
    return z, diags


def relative_error(a, b):
    numerator = denominator = 0.
    for i in range(0, len(a), 128):
        aa, bb = a[i:i+128].astype(float), b[i:i+128].astype(float)
        numerator += np.sum((aa - bb) ** 2)
        denominator += np.sum(bb ** 2)
    return float(np.sqrt(numerator / denominator))


def bac(y, p):
    assert set(np.unique(y)) == {0, 1}
    return float(np.mean([np.mean(p[y == c] == c) for c in (0, 1)]))


def run(root, out):
    out.mkdir(exist_ok=False, parents=True)
    t0 = time.perf_counter()
    ns, hashes = helpers(root)
    here = Path(__file__).resolve().parent
    for p in (Path(__file__), here / 'EXECUTION_PLAN.md'):
        hashes[str(p)] = sha(p)
    u = root / 'analysis_outputs/cmpb_unit_intervention_20260925/results'
    manifest = json.loads((u / 'manifest.json').read_text(encoding='utf8'))
    hashes[str(u / 'manifest.json')] = sha(u / 'manifest.json')
    channels = manifest['shared_channels']
    paths = {n: root / ('MI_test/preprocessed_sfreq100/' + n + '.npz') if n in NAMES[:2]
             else root / ('analysis_outputs/cmpb_bdea_20260924/' + n + '.npz') for n in NAMES}
    dump(out / 'lock.json', dict(hashes=hashes, expected_inputs=manifest['input_sha256'],
         policies=['coherent_absolute', 'relative'], scales=SCALES, fits=32,
         python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
         sklearn=sklearn.__version__, threads=4, covariance_dtype='float64', output_dtype='float32'))
    data = {}
    for n, path in paths.items():
        digest = sha(path)
        assert digest == manifest['input_sha256'][n], (n, 'input hash changed')
        with np.load(path, allow_pickle=True) as d:
            names = d['ch_names'].astype(str).tolist()
            assert float(np.asarray(d['sfreq']).ravel()[0]) == 100
            data[n] = dict(X=d['X'][:, [names.index(c) for c in channels], :200].astype(np.float32),
                           y=d['y'].astype(int), subjects=d['subjects'].astype(int))
        print('[loaded]', n, data[n]['X'].shape, flush=True)
    assert sum(len(np.unique(d['subjects'])) for d in data.values()) == 224
    rows, diagnostics, checks, historical = [], [], [], []
    fits = 0
    for policy in ('coherent_absolute', 'relative'):
        for alignment in ('subject', 'dataset_subject'):
            base_arrays = base_predictions = base_scores = None
            for label, scale in SCALES:
                arrays = {}
                for n, d in data.items():
                    arrays[n], dg = align(d['X'], d['subjects'], policy, alignment, scale)
                    diagnostics += [dict(policy=policy, alignment=alignment, scale=label, dataset=n, **v) for v in dg]
                if label == '1':
                    base_arrays = arrays
                pred_store, decision_store, save = {}, {}, {}
                for target in NAMES:
                    started = time.perf_counter()
                    source = [n for n in NAMES if n != target]
                    xs = np.concatenate([arrays[n] for n in source])
                    ys = np.concatenate([data[n]['y'] for n in source])
                    mu, sd = ns['norm_stats'](xs)
                    xs = ((xs - mu) / sd).astype(np.float32)
                    xt = ((arrays[target] - mu) / sd).astype(np.float32)
                    csp, scaler, lda = ns['fit_csp_lda'](ns['ManualCSP'], xs, ys, 8)
                    ft = scaler.transform(csp.transform(xt))
                    pred, decision = lda.predict(ft).astype(np.int8), lda.decision_function(ft)
                    y, ids = data[target]['y'], data[target]['subjects']
                    reference = pred if label == '1' else base_predictions[target]
                    ref_decision = decision if label == '1' else base_scores[target]
                    pred_store[target], decision_store[target] = pred, decision
                    for s in np.unique(ids):
                        m = ids == s
                        rows.append(dict(policy=policy, alignment=alignment, scale=label,
                                         dataset=target, subject=int(s), n_trials=int(m.sum()), bac=bac(y[m], pred[m])))
                    err = relative_error(arrays[target], base_arrays[target])
                    checks.append(dict(policy=policy, alignment=alignment, scale=label, dataset=target,
                        alignment_relative_error=err, alignment_pass=err <= 1e-5,
                        prediction_flips=int((pred != reference).sum()), n_predictions=len(pred),
                        max_decision_difference=float(np.max(np.abs(decision - ref_decision)))))
                    save.update({f'pred__{target}': pred, f'decision__{target}': decision,
                                 f'y__{target}': y, f'subjects__{target}': ids})
                    if label == '1':
                        old_policy = 'legacy' if policy == 'coherent_absolute' else 'relative'
                        old_path = u / f'predictions/floor-{old_policy}_align-{alignment}_scale-1.npz'
                        with np.load(old_path, allow_pickle=False) as old:
                            assert np.array_equal(y, old[f'y__{target}']) and np.array_equal(ids, old[f'subjects__{target}'])
                            historical.append(dict(policy=policy, alignment=alignment, dataset=target,
                              prediction_flips=int((pred != old[f'CSP-LDA__{target}']).sum())))
                    fits += 1
                    print('[fit]', fits, '/32', policy, alignment, label, target,
                          'seconds', round(time.perf_counter()-started, 2), flush=True)
                    dump(out / 'status.json', dict(state='running', fits=fits, total=32,
                         elapsed_seconds=time.perf_counter()-t0))
                    del xs, ys, xt, ft, csp, scaler, lda
                    gc.collect()
                np.savez_compressed(out / f'predictions_{policy}_{alignment}_{label}.npz', **save)
                if label == '1':
                    base_predictions, base_scores = pred_store, decision_store
                pd.DataFrame(rows).to_csv(out / 'subject_results.csv', index=False)
                pd.DataFrame(diagnostics).to_csv(out / 'diagnostics.csv', index=False)
                pd.DataFrame(checks).to_csv(out / 'unit_checks.csv', index=False)
                pd.DataFrame(historical).to_csv(out / 'historical_agreement.csv', index=False)
            del base_arrays, arrays, base_predictions, base_scores
            gc.collect()
    f = pd.DataFrame(rows)
    assert len(f) == 224 * 8
    results = []
    for (policy, scale), g in f.groupby(['policy', 'scale']):
        w = g.pivot(index=['dataset', 'subject'], columns='alignment', values='bac') * 100
        w['increment_pp'] = w.dataset_subject - w.subject
        dsmeans = w.groupby('dataset').mean()
        for scope in (*NAMES, 'equal_dataset', 'pooled_subject'):
            r = dsmeans.mean() if scope == 'equal_dataset' else w.mean() if scope == 'pooled_subject' else dsmeans.loc[scope]
            results.append(dict(policy=policy, scale=scale, scope=scope,
                                S_pct=r.subject, DS_pct=r.dataset_subject, increment_pp=r.increment_pp))
    pd.DataFrame(results).to_csv(out / 'comparison.csv', index=False)
    dump(out / 'status.json', dict(state='complete', fits=fits, total=32, subjects=224,
         elapsed_seconds=time.perf_counter()-t0, alignment_checks_pass=all(r['alignment_pass'] for r in checks),
         prediction_flips=sum(r['prediction_flips'] for r in checks),
         max_decision_difference=max(r['max_decision_difference'] for r in checks)))
    print(pd.DataFrame(results).query("scope == 'equal_dataset'").to_string(index=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    with threadpool_limits(limits=4):
        run(args.project, args.out)
