from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(__file__).resolve().parent
NAMES = ('cho2017', 'lee2019', 'bnci2014001', 'physionetmi')
ARCHIVE = ROOT / 'analysis_outputs/mi_four_checks_20261007'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_archive():
    spec = importlib.util.spec_from_file_location('archived_precision', ARCHIVE / 'precision_compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configurations(cfg):
    labels = [cfg['anchor_absolute_epsilon']] + [x for x in cfg['epsilon_labels'] if x != cfg['anchor_absolute_epsilon']]
    result = []
    for pipeline in ('subject', 'dataset_subject'):
        result.append(dict(id='relative_' + pipeline, path='relative', pipeline=pipeline,
                           policy='relative', epsilon_label='relative', raw_epsilon=1e-8, post_epsilon=1e-8))
    for label in labels:
        for pipeline in ('subject', 'dataset_subject'):
            result.append(dict(id='fixed_' + label + '_' + pipeline, path='fixed_post', pipeline=pipeline,
                               policy='absolute', epsilon_label=label, raw_epsilon=float(label), post_epsilon=cfg['primary_post_dataset_epsilon']))
    for label in cfg['epsilon_labels']:
        if label == cfg['anchor_absolute_epsilon']: continue
        result.append(dict(id='tied_' + label + '_dataset_subject', path='tied_post', pipeline='dataset_subject',
                           policy='absolute', epsilon_label=label, raw_epsilon=float(label), post_epsilon=float(label)))
    assert len(result) == 31 and len({x['id'] for x in result}) == 31
    return result


def batch(x, policy, epsilon):
    xx = x.astype(np.float64)
    r = np.mean(np.einsum('nct,ndt->ncd', xx, xx), axis=0, dtype=np.float64)
    r = (r + r.T) * .5
    ev, vec = np.linalg.eigh(r)
    if ev[0] <= 0 or not np.isfinite(ev).all(): raise ValueError('Non-SPD reference; no silent repair')
    floor = 1e-8 * np.trace(r) / len(r) if policy == 'relative' else epsilon
    adjusted = np.maximum(ev, floor)
    w = (vec * adjusted ** -.5) @ vec.T
    z = np.einsum('cd,ndt->nct', w, x, optimize=True).astype(np.float32)
    return z, dict(floor=float(floor), clipped=int((ev < floor).sum()), channels=len(r),
                   clip_fraction=float((ev < floor).mean()), whitening_AI=float(np.linalg.norm(np.log(ev / adjusted))),
                   raw_condition=float(ev[-1] / ev[0]), min_eigenvalue=float(ev[0]), max_eigenvalue=float(ev[-1]),
                   mean_eigenvalue=float(ev.mean()), floor_over_mean=float(floor / ev.mean()))


def align(x, subjects, policy, pipeline, raw_epsilon, post_epsilon):
    z = x.astype(np.float32, copy=True)
    diagnostics = []
    if pipeline == 'dataset_subject':
        z, d = batch(z, policy, raw_epsilon)
        diagnostics.append(dict(stage='dataset_raw', subject=-1, unit='V2', **d))
    for subject in np.unique(subjects):
        mask = subjects == subject
        epsilon = post_epsilon if pipeline == 'dataset_subject' else raw_epsilon
        z[mask], d = batch(z[mask], policy, epsilon)
        diagnostics.append(dict(stage='subject_post_D' if pipeline == 'dataset_subject' else 'subject_raw',
                                subject=int(subject), unit='dimensionless' if pipeline == 'dataset_subject' else 'V2', **d))
    return z, diagnostics


def prepare():
    assert not (HOME / 'LOCK.json').exists(), 'Lock already exists; do not overwrite'
    archived = load_archive()
    _, helper_hashes = archived.helpers(ROOT)
    old_lock = json.loads((ARCHIVE / 'precision_results/lock.json').read_text(encoding='utf-8'))
    for path, value in helper_hashes.items(): assert old_lock['hashes'][path] == value
    assert digest(ARCHIVE / 'precision_compare.py') == '42dd5eb1a5fb4cbf6420cf5aec3ff99cf380a5b07da014845dbaa09b6042eca2'
    cfg = json.loads((HOME / 'config.json').read_text(encoding='utf-8'))
    tests = subprocess.run([__import__('sys').executable, str(HOME / 'test_study.py')], capture_output=True, text=True)
    (HOME / 'synthetic_tests.log').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    assert tests.returncode == 0, tests.stdout + tests.stderr
    # Anonymous export uses the supplied historical extraction, not private Git history.
    original = (HOME / 'first_available_eeg_ea.py.txt').read_bytes()
    assert b'eps: float = 1e-8' in original and b'np.maximum(eigvals, eps)' in original
    cfg_hashes = {str(HOME / name): digest(HOME / name) for name in ('PROTOCOL.md', 'config.json', 'study.py', 'test_study.py', 'analyze.py')}
    manifest_path = ROOT / 'analysis_outputs/cmpb_unit_intervention_20260925/results/manifest.json'
    cfg_hashes[str(manifest_path)] = digest(manifest_path)
    lock = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), client_date='2026-10-15',
                config=cfg, conditions=configurations(cfg), code_hashes={**cfg_hashes, **helper_hashes,
                str(ARCHIVE / 'precision_compare.py'): digest(ARCHIVE / 'precision_compare.py')},
                input_hashes=old_lock['expected_inputs'], prior_lock_sha256=digest(ARCHIVE / 'precision_results/lock.json'),
                historical_code_sha256=hashlib.sha256(original).hexdigest(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, python=platform.python_version(), maximum_fits=124,
                note='exploratory; fixed before this sweep, after prior findings; no new-result-based grid selection')
    dump(HOME / 'LOCK.json', lock)
    print('Frozen31conditions/124maximumfits; synthetic tests passed; EEG not loaded.', flush=True)


def run():
    lock = json.loads((HOME / 'LOCK.json').read_text(encoding='utf-8'))
    for path, value in lock['code_hashes'].items(): assert digest(path) == value, ('Locked code changed', path)
    out = HOME / 'results'
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    archived = load_archive()
    ns, _ = archived.helpers(ROOT)
    manifest_path = ROOT / 'analysis_outputs/cmpb_unit_intervention_20260925/results/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['input_sha256'] == lock['input_hashes']
    paths = {n: ROOT / ('MI_test/preprocessed_sfreq100/' + n + '.npz') if n in NAMES[:2]
             else ROOT / ('analysis_outputs/cmpb_bdea_20260924/' + n + '.npz') for n in NAMES}
    data = {}
    for name, path in paths.items():
        assert digest(path) == lock['input_hashes'][name], ('Input changed', name)
        with np.load(path, allow_pickle=True) as d:
            names = d['ch_names'].astype(str).tolist()
            assert float(np.asarray(d['sfreq']).ravel()[0]) == 100
            data[name] = dict(X=d['X'][:, [names.index(c) for c in manifest['shared_channels']], :200].astype(np.float32),
                              y=d['y'].astype(int), subjects=d['subjects'].astype(int))
        print('[loaded]', name, data[name]['X'].shape, flush=True)
    assert sum(len(np.unique(d['subjects'])) for d in data.values()) == 224
    diagnostics, execution, anchors, cache = [], [], [], {}
    fits = 0
    for ci, condition in enumerate(lock['conditions'], 1):
        arrays = {}
        for name, d in data.items():
            arrays[name], dg = align(d['X'], d['subjects'], condition['policy'], condition['pipeline'], condition['raw_epsilon'], condition['post_epsilon'])
            diagnostics.extend(dict(condition=condition['id'], dataset=name, **v) for v in dg)
        key = tuple(hashlib.sha256(arrays[n].tobytes()).hexdigest() for n in NAMES)
        save = {}
        if key in cache:
            previous = cache[key]
            with np.load(out / ('predictions_' + previous + '.npz'), allow_pickle=False) as f:
                save = {k: f[k].copy() for k in f.files}
            execution.append(dict(condition=condition['id'], reused_from=previous, new_fits=0))
        else:
            for target in NAMES:
                source = [n for n in NAMES if n != target]
                xs = np.concatenate([arrays[n] for n in source])
                ys = np.concatenate([data[n]['y'] for n in source])
                mu, sd = ns['norm_stats'](xs)
                xs = ((xs - mu) / sd).astype(np.float32)
                xt = ((arrays[target] - mu) / sd).astype(np.float32)
                csp, scaler, lda = ns['fit_csp_lda'](ns['ManualCSP'], xs, ys, 8)
                features = scaler.transform(csp.transform(xt))
                save['pred__' + target] = lda.predict(features).astype(np.int8)
                save['decision__' + target] = lda.decision_function(features)
                save['subjects__' + target] = data[target]['subjects']
                fits += 1
                print('[fit]', fits, condition['id'], target, flush=True)
                del xs, ys, xt, features, csp, scaler, lda
                gc.collect()
            cache[key] = condition['id']
            execution.append(dict(condition=condition['id'], reused_from='', new_fits=4))
        if condition['path'] == 'relative' or (condition['path'] == 'fixed_post' and condition['epsilon_label'] == '1e-8'):
            old_policy = 'relative' if condition['path'] == 'relative' else 'coherent_absolute'
            old_path = ARCHIVE / 'precision_results' / ('predictions_' + old_policy + '_' + condition['pipeline'] + '_1.npz')
            with np.load(old_path, allow_pickle=False) as old:
                for target in NAMES:
                    assert np.array_equal(save['pred__' + target], old['pred__' + target]), ('Anchor prediction mismatch', condition['id'], target)
                    assert np.array_equal(data[target]['subjects'], old['subjects__' + target])
                    assert np.array_equal(data[target]['y'], old['y__' + target])
                    anchors.append(dict(condition=condition['id'], dataset=target, predictions_identical=True,
                                        max_decision_difference=float(np.max(np.abs(save['decision__' + target] - old['decision__' + target])))))
        np.savez_compressed(out / ('predictions_' + condition['id'] + '.npz'), **save)
        pd.DataFrame(diagnostics).to_csv(out / 'diagnostics.csv', index=False)
        pd.DataFrame(execution).to_csv(out / 'execution.csv', index=False)
        dump(out / 'anchor_checks.json', anchors)
        dump(out / 'status.json', dict(state='predicting', completed_conditions=ci, total_conditions=31, actual_fits=fits, elapsed_seconds=time.perf_counter()-started))
        print('[condition]', ci, '/31', condition['id'], 'reuse', execution[-1]['reused_from'], flush=True)
        del arrays
        gc.collect()
    # Aggregate labels/scores are not used until every condition's prediction is saved.
    labels = {}
    for name, d in data.items():
        labels['y__' + name] = d['y']
        labels['subjects__' + name] = d['subjects']
    np.savez_compressed(out / 'scoring_labels.npz', **labels)
    for name, path in paths.items(): assert digest(path) == lock['input_hashes'][name]
    dump(out / 'status.json', dict(state='predictions_complete', completed_conditions=31, total_conditions=31, actual_fits=fits,
                                  reused_conditions=sum(bool(x['reused_from']) for x in execution), elapsed_seconds=time.perf_counter()-started))
    print('All predictions saved; starting separately specified scoring.', flush=True)
    subprocess.run([__import__('sys').executable, str(HOME / 'analyze.py')], check=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'run'])
    args = ap.parse_args()
    with threadpool_limits(limits=4):
        prepare() if args.action == 'prepare' else run()
