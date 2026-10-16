from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
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
ARCHIVE = ROOT / 'analysis_outputs/mi_four_checks_20261007'
SWEEP = ROOT / 'analysis_outputs/cmpb_absolute_floor_sweep_20261015'
NAMES = ('cho2017', 'lee2019', 'bnci2014001', 'physionetmi')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cells(cfg):
    return [dict(id='_'.join((v, p, s)), version=v, policy=p, pipeline=s)
            for v in cfg['versions'] for p in cfg['policies'] for s in cfg['pipelines']]


def input_paths():
    paths = {n: ROOT / ('MI_test/preprocessed_sfreq100/'+n+'.npz') if n in NAMES[:2]
             else ROOT / ('analysis_outputs/cmpb_bdea_20260924/'+n+'.npz') for n in NAMES}
    paths['cho_original'] = ROOT/'MI_test/preprocessed_sfreq100/cho2017_moabb_uncorrected.npz'
    return paths


def check_metadata(a, b):
    assert a['X'].shape == b['X'].shape, 'Cho used-data shape differs'
    for key in ['y', 'subjects']:
        assert np.array_equal(a[key], b[key]), 'Cho '+key+' order differs'
    assert a['channels'] == b['channels'], 'Cho used channel order differs'
    assert a['sfreq'] == b['sfreq'] == 100, 'Cho sampling rate differs'


def norm_error(a, b):
    num = den = 0.
    for i in range(0, len(a), 128):
        aa, bb = a[i:i+128].astype(float), b[i:i+128].astype(float)
        num += np.sum((aa-bb)**2)
        den += np.sum(bb**2)
    if den == 0:
        raise ValueError('Zero comparison norm')
    return float(np.sqrt(num/den))


def prepare():
    assert not (HOME/'LOCK.json').exists(), 'Do not overwrite a frozen plan'
    cfg = json.loads((HOME/'config.json').read_text(encoding='utf-8'))
    archive = load_module(ARCHIVE/'precision_compare.py', 'precision_archive')
    _, helpers = archive.helpers(ROOT)
    old = json.loads((ARCHIVE/'precision_results/lock.json').read_text(encoding='utf-8'))
    for p, h in helpers.items():
        assert old['hashes'][p] == h
    assert sha(ARCHIVE/'precision_compare.py') == '42dd5eb1a5fb4cbf6420cf5aec3ff99cf380a5b07da014845dbaa09b6042eca2'
    old_manifest = ROOT/'analysis_outputs/cmpb_chofix_20260929/backup_pre_chofix/cmpb_unit_intervention_20260925/results/manifest.json'
    manifest = json.loads(old_manifest.read_text(encoding='utf-8'))
    expected = dict(old['expected_inputs'])
    expected['cho_original'] = manifest['input_sha256']['cho2017']
    assert set(expected) == set(input_paths())
    for name, p in input_paths().items():
        assert sha(p) == expected[name], ('Input hash mismatch', name)
    tests = subprocess.run([sys.executable, str(HOME/'test_study.py')], capture_output=True, text=True)
    (HOME/'synthetic_tests.log').write_text(tests.stdout+tests.stderr, encoding='utf-8')
    assert tests.returncode == 0, tests.stdout+tests.stderr
    frozen = {str(HOME/n): sha(HOME/n) for n in ['config.json','PROTOCOL.md','study.py','test_study.py','analyze.py','verify.py']}
    frozen.update(helpers)
    for p in [ARCHIVE/'precision_compare.py', SWEEP/'study.py', old_manifest,
              ROOT/'analysis_outputs/cmpb_unit_intervention_20260925/results/manifest.json']:
        frozen[str(p)] = sha(p)
    anchors = {}
    for policy in cfg['policies']:
        for pipe in cfg['pipelines']:
            name = 'coherent_absolute' if policy == 'absolute' else policy
            p = ARCHIVE/'precision_results'/('predictions_'+name+'_'+pipe+'_1.npz')
            anchors[str(p)] = sha(p)
    dump(HOME/'LOCK.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(), config=cfg,
         conditions=cells(cfg), code_hashes=frozen, input_hashes=expected, anchor_hashes=anchors,
         python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
         maximum_fits=32, synthetic_tests_passed=True,
         note='Exploratory version sensitivity; freeze precedes EEG array loading, not prior studies'))
    print('Locked 8cells/32fits. Input file hashes and synthetic tests pass. EEG arrays not loaded.', flush=True)


def load_input(p, channels):
    with np.load(p, allow_pickle=True) as f:
        names = f['ch_names'].astype(str).tolist()
        sfreq = float(np.asarray(f['sfreq']).ravel()[0])
        assert sfreq == 100 and len(set(names)) == len(names)
        return dict(X=f['X'][:, [names.index(c) for c in channels], :200].astype(np.float32),
                    y=f['y'].astype(int), subjects=f['subjects'].astype(int), channels=list(channels), sfreq=sfreq)


def run():
    lock = json.loads((HOME/'LOCK.json').read_text(encoding='utf-8'))
    cfg = lock['config']
    for p, h in {**lock['code_hashes'], **lock['anchor_hashes']}.items():
        assert sha(p) == h, ('Frozen file changed', p)
    paths = input_paths()
    for name, p in paths.items():
        assert sha(p) == lock['input_hashes'][name], ('Input changed', name)
    out = HOME/'results'
    out.mkdir(exist_ok=False)
    t0 = time.perf_counter()
    archive = load_module(ARCHIVE/'precision_compare.py', 'precision_archive')
    sweep = load_module(SWEEP/'study.py', 'floor_operator')
    ns, _ = archive.helpers(ROOT)
    manifest = json.loads((ROOT/'analysis_outputs/cmpb_unit_intervention_20260925/results/manifest.json').read_text(encoding='utf-8'))
    data = {n: load_input(paths[n], manifest['shared_channels']) for n in NAMES}
    original = load_input(paths['cho_original'], manifest['shared_channels'])
    check_metadata(data['cho2017'], original)
    expected = original['X']*np.float32(cfg['adopted_over_original'])
    error = norm_error(data['cho2017']['X'], expected)
    audit = dict(used_shape=list(expected.shape), n_subjects=len(np.unique(original['subjects'])),
                 labels_subjects_order_equal=True, channel_sample_contract_equal=True,
                 adopted_over_original=cfg['adopted_over_original'], amplitude_relative_error=error,
                 used_arrays_bitwise_scaled_equal=bool(np.array_equal(data['cho2017']['X'], expected)),
                 amplitude_check_pass=bool(error <= cfg['amplitude_audit_relative_tolerance']),
                 original_file='retained archive, not reconstructed', physical_scaling_validated=False)
    dump(out/'input_audit.json', audit)
    assert audit['amplitude_check_pass'], 'Cho data differ beyond declared amplitude choice'
    assert sum(len(np.unique(d['subjects'])) for d in data.values()) == 224
    del expected
    print('G0 input audit pass:', json.dumps(audit), flush=True)
    anchors, invariance, diagnostics = [], [], []
    original_inputs = dict(data)
    original_inputs['cho2017'] = original
    fits = 0
    for ci, cell in enumerate(lock['conditions'], 1):
        dset = data if cell['version'] == 'adopted' else original_inputs
        arrays = {}
        for name in NAMES:
            d = dset[name]
            arrays[name], dg = sweep.align(d['X'], d['subjects'], cell['policy'], cell['pipeline'],
                                           cfg['raw_absolute_floor'], cfg['post_D_absolute_floor'])
            diagnostics.extend(dict(condition=cell['id'], dataset=name, **v) for v in dg)
            if cell['version'] == 'original' and cell['policy'] == 'relative':
                previous, _ = sweep.align(data[name]['X'], data[name]['subjects'], cell['policy'], cell['pipeline'],
                                           cfg['raw_absolute_floor'], cfg['post_D_absolute_floor'])
                relerr = norm_error(arrays[name], previous)
                invariance.append(dict(pipeline=cell['pipeline'], dataset=name,
                    alignment_relative_error=relerr, alignment_pass=bool(relerr<=cfg['relative_alignment_tolerance']),
                    arrays_byte_equal=bool(np.array_equal(arrays[name], previous)),
                    max_alignment_difference=float(np.max(np.abs(arrays[name]-previous)))))
                del previous
        save = {}
        for target in NAMES:
            sources = [n for n in NAMES if n != target]
            xs = np.concatenate([arrays[n] for n in sources])
            ys = np.concatenate([dset[n]['y'] for n in sources])
            mu, sd = ns['norm_stats'](xs)
            xs = ((xs-mu)/sd).astype(np.float32)
            xt = ((arrays[target]-mu)/sd).astype(np.float32)
            csp, scaler, lda = ns['fit_csp_lda'](ns['ManualCSP'], xs, ys, 8)
            features = scaler.transform(csp.transform(xt))
            pred = lda.predict(features).astype(np.int8)
            decision = lda.decision_function(features)
            save.update({'pred__'+target: pred, 'decision__'+target: decision, 'subjects__'+target:dset[target]['subjects']})
            if cell['version'] == 'adopted':
                old_policy = 'coherent_absolute' if cell['policy']=='absolute' else 'relative'
                p = ARCHIVE/'precision_results'/('predictions_'+old_policy+'_'+cell['pipeline']+'_1.npz')
                with np.load(p, allow_pickle=False) as prior:
                    assert np.array_equal(pred, prior['pred__'+target]), ('Anchor mismatch', cell, target)
                    assert np.array_equal(dset[target]['y'], prior['y__'+target])
                    assert np.array_equal(dset[target]['subjects'], prior['subjects__'+target])
                    anchors.append(dict(condition=cell['id'], dataset=target, predictions_equal=True,
                        max_decision_difference=float(np.max(np.abs(decision-prior['decision__'+target])))))
            elif cell['policy'] == 'relative':
                with np.load(out/('predictions_adopted_relative_'+cell['pipeline']+'.npz'), allow_pickle=False) as prior:
                    current = next(r for r in invariance if r['pipeline']==cell['pipeline'] and r['dataset']==target)
                    current.update(prediction_flips=int((pred!=prior['pred__'+target]).sum()),
                        n_predictions=len(pred), max_decision_difference=float(np.max(np.abs(decision-prior['decision__'+target]))))
            fits += 1
            dump(out/'status.json', dict(state='predicting', actual_fits=fits, total_fits=32, completed_cells=ci-1,
                                        elapsed_seconds=time.perf_counter()-t0))
            print('[fit]', fits, '/32', cell['id'], target, flush=True)
            del xs, ys, xt, features, csp, scaler, lda
            gc.collect()
        np.savez_compressed(out/('predictions_'+cell['id']+'.npz'), **save)
        dump(out/'anchor_checks.json', anchors)
        dump(out/'relative_invariance.json', invariance)
        pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv', index=False)
        del arrays
        gc.collect()
    labels = {}
    for name, d in data.items():
        labels['y__'+name] = d['y']
        labels['subjects__'+name] = d['subjects']
    np.savez_compressed(out/'scoring_labels.npz', **labels)
    for name, p in paths.items():
        assert sha(p) == lock['input_hashes'][name]
    assert len(anchors) == 16 and len(invariance) == 8
    dump(out/'status.json', dict(state='predictions_complete', actual_fits=fits, total_fits=32,
         completed_cells=8, elapsed_seconds=time.perf_counter()-t0,
         relative_alignment_check_pass=all(r['alignment_pass'] for r in invariance),
         relative_prediction_flips=sum(r['prediction_flips'] for r in invariance)))
    print('All predictions saved; scoring follows without refits.', flush=True)
    subprocess.run([sys.executable, str(HOME/'analyze.py')], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        prepare() if args.action=='prepare' else run()
