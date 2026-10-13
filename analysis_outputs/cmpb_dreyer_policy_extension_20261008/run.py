"""Post-hoc external policy contrast; original inputs and helpers are read-only."""
import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SOURCES = ('cho2017', 'lee2019', 'bnci2014001', 'physionetmi')
CONDITIONS = (('coherent_absolute', 'subject'),
              ('coherent_absolute', 'dataset_subject'),
              ('relative', 'subject'), ('relative', 'dataset_subject'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_operator(root):
    path = root / 'analysis_outputs/mi_four_checks_20261007/precision_compare.py'
    spec = importlib.util.spec_from_file_location('archived_precision', path)
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)
    ns, hashes = op.helpers(root)
    return op, ns, {**hashes, str(path): sha(path)}


def bootstrap(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 4 or not np.isfinite(scores).all():
        raise ValueError('Expected finite subject-by-four score matrix')
    if not ((scores >= 0) & (scores <= 100)).all():
        raise ValueError('Scores must be percentage BAC')
    a_s, a_ds, r_s, r_ds = scores.T
    contrasts = np.column_stack(((a_ds-a_s)-(r_ds-r_s), a_ds-a_s,
                                 r_ds-r_s, a_ds-r_s, r_s-a_s))
    names = ['policy_increment_difference', 'absolute_increment',
             'relative_increment', 'absolute_DS_minus_relative_S',
             'relative_S_minus_absolute_S']
    indices = np.random.default_rng(20261008).integers(0, len(scores), (10000, len(scores)))
    draws = contrasts[indices].mean(axis=1)
    rows = []
    for i, name in enumerate(names):
        lo, hi = np.quantile(draws[:, i], [0.025, 0.975])
        rows.append(dict(contrast=name, role='primary' if i == 0 else 'secondary_descriptive',
                         n=len(scores), mean_pp=float(contrasts[:, i].mean()),
                         ci95_low=float(lo), ci95_high=float(hi)))
    return rows, draws, contrasts


def test(root):
    op, ns, hashes = load_operator(root)
    rng = np.random.default_rng(12)
    x = rng.normal(size=(24, 8, 100)).astype(np.float32) * np.float32(1e-6)
    ids = np.repeat([1, 2, 3], 8)
    tests = []
    for policy, alignment in CONDITIONS:
        a, dg = op.align(x, ids, policy, alignment, 1.)
        b, _ = op.align(x, ids, policy, alignment, 1e-3)
        assert op.relative_error(a, b) < 1e-5
        assert a.dtype == np.float32 and np.isfinite(a).all()
        tests.append(policy + '_' + alignment + '_coherent')
    a, _ = op.align(x, ids, 'relative', 'subject', 1.)
    for s in np.unique(ids):
        xx = a[ids == s].astype(float)
        r = np.mean(np.einsum('nct,ndt->ncd', xx, xx), axis=0)
        assert np.allclose(r, np.eye(8), atol=1e-6)
    tests.append('relative_unclipped_whitening')
    matrix = np.tile([60., 70., 70., 71.], (20, 1))
    rows, draws, _ = bootstrap(matrix)
    assert rows[0]['mean_pp'] == 9. and np.all(draws[:, 0] == 9.)
    tests.append('paired_bootstrap_constant_difference')
    varied = rng.uniform(40, 85, (20, 4))
    rows, draws, _ = bootstrap(varied)
    swapped = varied[:, [2, 3, 0, 1]]
    _, other, _ = bootstrap(swapped)
    assert np.allclose(draws[:, 0], -other[:, 0])
    tests.append('policy_swap_reverses_contrast')
    y = np.tile([0, 1], 12)
    mu, sd = ns['norm_stats'](a)
    csp, scaler, lda = ns['fit_csp_lda'](ns['ManualCSP'], ((a-mu)/sd).astype(np.float32), y, 8)
    pred = lda.predict(scaler.transform(csp.transform(((a-mu)/sd).astype(np.float32))))
    assert pred.shape == y.shape and set(pred).issubset({0, 1})
    tests.append('synthetic_fit_predict_smoke')
    dump(HERE / 'synthetic_qa.json', dict(tests=tests, n_tests=len(tests), passed=True,
         eeg_opened=False, helpers=hashes, runner_sha256=sha(__file__)))
    print('Synthetic tests passed:', len(tests), flush=True)


def make_lock(root):
    old_path = root / 'analysis_outputs/cmpb_confirmation_dreyer2023/run_manifest.json'
    development_path = root / 'analysis_outputs/mi_four_checks_20261007/precision_results/lock.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    dev = json.loads(development_path.read_text(encoding='utf-8'))
    op, ns, hashes = load_operator(root)
    for helper, digest in hashes.items():
        assert digest in dev['hashes'].values(), ('Helper changed since development', helper)
    paths = {n: root / ('MI_test/preprocessed_sfreq100/' + n + '.npz') if n in SOURCES[:2]
             else root / ('analysis_outputs/cmpb_bdea_20260924/' + n + '.npz') for n in SOURCES}
    paths['dreyer2023'] = root / 'analysis_outputs/cmpb_confirmation_dreyer2023/dreyer2023.npz'
    expected = {**dev['expected_inputs'], 'dreyer2023': old['inputs']['dreyer2023']['sha256']}
    actual = {n: sha(path) for n, path in paths.items()}
    assert actual == expected, ('Input hashes changed', actual, expected)
    hashes.update({str(p): sha(p) for p in (Path(__file__), HERE/'PROTOCOL.md',
                                          old_path, development_path)})
    qa = json.loads((HERE/'synthetic_qa.json').read_text(encoding='utf-8'))
    assert qa['passed'] and qa['runner_sha256'] == sha(__file__)
    lock = dict(created_utc=timestamp(), design_status='post_hoc_external_extension',
                inputs={n: dict(path=str(paths[n]), sha256=actual[n]) for n in paths},
                code_and_protocol=hashes, channels=old['shared_channels'],
                old_cho_sha256=old['inputs']['cho2017']['sha256'],
                reason_four_fits='Original Cho differs; rerun both policies in matched precision',
                conditions=CONDITIONS, threads=4, n_csp=8, n_times=200,
                n_bootstrap=10000, bootstrap_seed=20261008,
                environment=dict(python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__))
    assert lock['environment']['numpy'] == '1.26.4'
    assert lock['environment']['scipy'] == '1.13.1'
    assert lock['environment']['sklearn'] == '1.5.2'
    return lock, op, ns


def load_data(lock):
    data, rows = {}, []
    for name, entry in lock['inputs'].items():
        with np.load(entry['path'], allow_pickle=True) as f:
            ch = f['ch_names'].astype(str).tolist()
            assert len(set(ch)) == len(ch)
            assert float(np.asarray(f['sfreq']).ravel()[0]) == 100
            indices = [ch.index(c) for c in lock['channels']]
            x = f['X'][:, indices, :200].astype(np.float32)
            ids = f['subjects'].astype(int)
            assert x.shape == (len(ids), 18, 200)
            assert np.isfinite(x).all()
            d = dict(X=x, subjects=ids)
            if name != 'dreyer2023':
                d['y'] = f['y'].astype(int)
                assert len(d['y']) == len(ids) and set(np.unique(d['y'])) == {0, 1}
            data[name] = d
            for subject in np.unique(ids):
                xx = x[ids == subject].astype(float)
                r = np.einsum('nct,ndt->ncd', xx, xx).mean(axis=0)
                ev = np.linalg.eigvalsh((r+r.T)*.5)
                assert ev[0] > 0 and np.isfinite(ev).all()
                rows.append(dict(dataset=name, subject=int(subject), n_trials=len(xx),
                                 eig_min=float(ev[0]), eig_max=float(ev[-1]),
                                 absolute_clipped=int((ev < 1e-8).sum()),
                                 relative_clipped=int((ev < 1e-8*ev.mean()).sum())))
            print('[loaded]', name, x.shape, len(np.unique(ids)), flush=True)
            del xx, r, ev
    assert sum(len(np.unique(data[n]['subjects'])) for n in SOURCES) == 224
    assert len(np.unique(data['dreyer2023']['subjects'])) == 87
    assert 'y' not in data['dreyer2023']
    return data, rows


def progress(root, heading, message):
    with (root/'MI_test/progress.md').open('a', encoding='utf-8') as f:
        f.write('\n\n## ' + heading + ' (2026-10-08)\n\n' + message + '\n')


def run(root):
    started = time.perf_counter()
    out = HERE/'results'
    out.mkdir(exist_ok=False)
    try:
        lock, op, ns = make_lock(root)
        dump(out/'lock.json', lock)
        dump(out/'status.json', dict(state='auditing_inputs', utc=timestamp()))
        progress(root, 'Dreyer policy extension locked and started',
            '- Post-hoc four-condition CSP-LDA extension, not preregistration. '
            'Original Dreyer Cho hash differs from corrected development input; '
            'both policies will therefore be rerun on the same 18 channels, 224 source '
            'subjects and 87 held-out subjects. No prior results overwritten.\n'
            '- Protocol/code/input hashes fixed before fitting in '
            '`analysis_outputs/cmpb_dreyer_policy_extension_20261008/results/lock.json`; '
            'synthetic tests passed. Target labels will be read only after all four '
            'prediction files exist. Primary subject-paired contrast and 10,000-draw '
            'percentile CI fixed in PROTOCOL.md. Execution started; results pending.')
        data, audit = load_data(lock)
        pd.DataFrame(audit).to_csv(out/'input_audit.csv', index=False)
        progress(root, 'Dreyer policy extension input audit passed',
            '- Verified expected hashes and finite 18x200 arrays at 100 Hz; '
            '224 source and 87 target subjects. All subject references SPD. '
            'Target labels not read. Per-subject spectra/trial counts in '
            '`analysis_outputs/cmpb_dreyer_policy_extension_20261008/results/input_audit.csv`. '
            'Four condition fits now starting; not complete.')
        pred_paths, diagnostic_rows, timings = [], [], []
        for i, (policy, alignment) in enumerate(CONDITIONS, 1):
            t0 = time.perf_counter()
            arrays = {}
            for name, d in data.items():
                arrays[name], dg = op.align(d['X'], d['subjects'], policy, alignment, 1.)
                diagnostic_rows += [dict(policy=policy, alignment=alignment, dataset=name, **v) for v in dg]
            xs = np.concatenate([arrays[n] for n in SOURCES])
            ys = np.concatenate([data[n]['y'] for n in SOURCES])
            mu, sd = ns['norm_stats'](xs)
            xs = ((xs-mu)/sd).astype(np.float32)
            xt = ((arrays['dreyer2023']-mu)/sd).astype(np.float32)
            csp, scaler, lda = ns['fit_csp_lda'](ns['ManualCSP'], xs, ys, 8)
            features = scaler.transform(csp.transform(xt))
            pred = lda.predict(features).astype(np.int8)
            decision = lda.decision_function(features)
            assert np.isfinite(decision).all() and len(pred) == len(data['dreyer2023']['subjects'])
            path = out/('predictions_' + policy + '_' + alignment + '.npz')
            np.savez_compressed(path, pred=pred, decision=decision,
                subjects=data['dreyer2023']['subjects'], trial_index=np.arange(len(pred)))
            pred_paths.append(path)
            np.savez_compressed(out/('model_'+policy+'_'+alignment+'.npz'),
                channel_mean=mu, channel_sd=sd, csp_filters=csp.filters_,
                feature_mean=scaler.mean_, feature_scale=scaler.scale_,
                lda_coef=lda.coef_, lda_intercept=lda.intercept_, classes=lda.classes_)
            timings.append(dict(condition=path.stem, seconds=time.perf_counter()-t0))
            dump(out/'status.json', dict(state='fitting', completed_fits=i, total_fits=4,
                target_labels_read=False, elapsed_seconds=time.perf_counter()-started))
            pd.DataFrame(diagnostic_rows).to_csv(out/'diagnostics.csv', index=False)
            print('[fit]', i, '/4', policy, alignment, round(time.perf_counter()-t0, 2), 's', flush=True)
            del arrays, xs, ys, xt, features, csp, scaler, lda
            gc.collect()
        dump(out/'predictions_frozen.json', dict(utc=timestamp(),
             target_labels_read=False, hashes={p.name: sha(p) for p in pred_paths}))
        with np.load(lock['inputs']['dreyer2023']['path'], allow_pickle=True) as f:
            target_y = f['y'].astype(int)
        ids = data['dreyer2023']['subjects']
        assert len(target_y) == len(ids)
        subject_ids = np.unique(ids)
        scores = np.zeros((len(subject_ids), 4))
        rows = []
        for j, ((policy, alignment), path) in enumerate(zip(CONDITIONS, pred_paths)):
            with np.load(path) as f:
                assert np.array_equal(f['subjects'], ids)
                assert np.array_equal(f['trial_index'], np.arange(len(ids)))
                pred = f['pred']
            for k, subject in enumerate(subject_ids):
                m = ids == subject
                assert set(np.unique(target_y[m])) == {0, 1}
                scores[k, j] = 100 * balanced_accuracy_score(target_y[m], pred[m])
                rows.append(dict(subject=int(subject), policy=policy, alignment=alignment,
                                 n_trials=int(m.sum()), bac_pct=float(scores[k, j])))
        pd.DataFrame(rows).to_csv(out/'subject_results.csv', index=False)
        contrasts, draws, individual = bootstrap(scores)
        pd.DataFrame(contrasts).to_csv(out/'contrasts.csv', index=False)
        pd.DataFrame(dict(subject=subject_ids, **{r['contrast']: individual[:,j]
                          for j,r in enumerate(contrasts)})).to_csv(out/'subject_contrasts.csv', index=False)
        np.savez_compressed(out/'bootstrap_draws.npz', draws=draws, subject_ids=subject_ids,
                            scores_pct=scores, y=target_y, subjects=ids)
        summary = [dict(policy=p, alignment=a, n=len(subject_ids), mean_bac_pct=float(scores[:,j].mean()),
                        sd_bac_pct=float(scores[:,j].std(ddof=1))) for j,(p,a) in enumerate(CONDITIONS)]
        pd.DataFrame(summary).to_csv(out/'performance_summary.csv', index=False)
        dump(out/'fit_timings.json', timings)
        dump(out/'status.json', dict(state='complete', completed_fits=4, target_subjects=87,
            source_subjects=224, target_trials=len(ids), elapsed_seconds=time.perf_counter()-started,
            target_labels_read_after_predictions=True, bootstrap_draws=10000, completed_utc=timestamp()))
        dump(out/'output_hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
        print(pd.DataFrame(summary).to_string(index=False), flush=True)
        print(pd.DataFrame(contrasts).to_string(index=False), flush=True)
        progress(root, 'Dreyer policy extension fits and bootstrap complete',
            '- Four precision-matched CSP-LDA fits and 10,000 subject-paired bootstrap '
            'draws completed; all 87 Dreyer subjects scored, no target labels used '
            'before freezing predictions. New outputs only in '
            '`analysis_outputs/cmpb_dreyer_policy_extension_20261008/results/`. '
            'Review contrasts.csv/performance_summary.csv before manuscript integration. '
            'Post-hoc, fixed-source/cohort-conditional analysis; independent QA/report pending.')
    except Exception as exc:
        dump(out/'status.json', dict(state='failed', error=repr(exc), utc=timestamp()))
        progress(root, 'Dreyer policy extension execution failed',
                 '- Stopped without silently changing conditions: ' + repr(exc) + '. See new run status.json.')
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()
    with threadpool_limits(limits=4):
        if args.test:
            test(args.root)
        else:
            run(args.root)
