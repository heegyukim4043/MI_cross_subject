"""Independently verify saved predictions and report numerical boundary changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
out = root / 'precision_results'
rows = pd.read_csv(out / 'subject_results.csv', dtype={'scale': str})
assert len(rows) == 1792 and not rows.duplicated(['policy', 'alignment', 'scale', 'dataset', 'subject']).any()
verified = 0
flips = []
for policy in ('coherent_absolute', 'relative'):
    for alignment in ('subject', 'dataset_subject'):
        with np.load(out / f'predictions_{policy}_{alignment}_1.npz') as a, np.load(out / f'predictions_{policy}_{alignment}_1e-6.npz') as b:
            for name in ('cho2017', 'lee2019', 'bnci2014001', 'physionetmi'):
                y = a[f'y__{name}']
                ids = a[f'subjects__{name}']
                assert np.array_equal(y, b[f'y__{name}'])
                assert np.array_equal(ids, b[f'subjects__{name}'])
                for scale, data in [('1', a), ('1e-6', b)]:
                    p = data[f'pred__{name}']
                    assert np.isfinite(data[f'decision__{name}']).all()
                    for subject in np.unique(ids):
                        m = ids == subject
                        score = np.mean([np.mean(p[m & (y == c)] == c) for c in (0, 1)])
                        found = rows[(rows.policy == policy) & (rows.alignment == alignment) & (rows.scale == scale) & (rows.dataset == name) & (rows.subject == subject)]
                        assert len(found) == 1 and abs(found.iloc[0].bac - score) < 1e-12
                        verified += 1
                for i in np.flatnonzero(a[f'pred__{name}'] != b[f'pred__{name}']):
                    flips.append(dict(policy=policy, alignment=alignment, dataset=name, subject=int(ids[i]),
                                      row=int(i), y=int(y[i]), pred_1=int(a[f'pred__{name}'][i]),
                                      pred_scaled=int(b[f'pred__{name}'][i]),
                                      decision_1=float(a[f'decision__{name}'][i]),
                                      decision_scaled=float(b[f'decision__{name}'][i])))
diag = pd.read_csv(out / 'diagnostics.csv', dtype={'scale': str})
diag.groupby(['policy', 'alignment', 'scale', 'stage', 'dataset']).agg(
    clipped_min=('clipped', 'min'), clipped_median=('clipped', 'median'), clipped_max=('clipped', 'max'),
    AI_median=('whitening_AI', 'median')).to_csv(out / 'diagnostic_summary.csv')
checks = pd.read_csv(out / 'unit_checks.csv', dtype={'scale': str})
report = dict(subject_scores_verified=verified, prediction_flips=flips,
              array_tolerance=1e-5, array_checks_pass=bool(checks.alignment_pass.all()),
              decision_tolerance=1e-3, decision_checks_pass=bool((checks.max_decision_difference <= 1e-3).all()),
              exact_prediction_invariance=not bool(flips))
(out / 'independent_qa.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report, indent=2))
