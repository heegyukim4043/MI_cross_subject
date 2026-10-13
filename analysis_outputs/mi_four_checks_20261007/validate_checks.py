"""Frozen event-specific controls; not a classifier of harmful regularization."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from precision_compare import align, relative_error


def main():
    rows = []
    for seed in range(20):
        rng = np.random.default_rng(seed + 20261007)
        q, _ = np.linalg.qr(rng.normal(size=(21, 21)))
        variances = np.geomspace(1e-13, 1e-8, 21)
        x = np.einsum('cd,ndt->nct', q @ np.diag(np.sqrt(variances)),
                      rng.normal(size=(10, 21, 200))).astype(np.float32)
        ids = np.repeat([1, 2], 5)
        for method in ('subject', 'dataset_subject'):
            for policy in ('coherent_absolute', 'relative'):
                base, dg = align(x, ids, policy, method, 1.)
                z, _ = align(x, ids, policy, method, 1e-6)
                error = relative_error(z, base)
                rows.append(dict(seed=seed, method=method, policy=policy, event='coherent',
                                 error=error, expected_flag=False, observed_flag=error > 1e-5,
                                 inspect_regularization=any(d['clipped'] > 0 for d in dg)))
                assert error <= 1e-5
                if policy == 'coherent_absolute':
                    assert any(d['clipped'] > 0 for d in dg)
                    bad, _ = align(x, ids, policy, method, 1e-6, contract='fixed')
                    error = relative_error(bad, base)
                    rows.append(dict(seed=seed, method=method, policy=policy, event='fixed_epsilon',
                                     error=error, expected_flag=True, observed_flag=error > 1e-5,
                                     inspect_regularization=True))
                    assert error > 1e-5
        # At large scale, wrongly rescaling H's second epsilon induces a visible failure.
        base, _ = align(x, ids, 'coherent_absolute', 'dataset_subject', 1.)
        bad, _ = align(x, ids, 'coherent_absolute', 'dataset_subject', 1e6, contract='wrong_both')
        error = relative_error(bad, base)
        rows.append(dict(seed=seed, method='dataset_subject', policy='coherent_absolute',
                         event='wrong_second_stage', error=error, expected_flag=True,
                         observed_flag=error > 1e-5, inspect_regularization=True))
        assert error > 1e-5
    out = Path(__file__).resolve().parent
    pd.DataFrame(rows).to_csv(out / 'synthetic_checks.csv', index=False)
    summary = dict(fixtures=20, control_cases=len(rows),
                   failed_expectations=sum(r['expected_flag'] != r['observed_flag'] for r in rows),
                   scope='Deterministic synthetic coverage; not clinical sensitivity/specificity or performance prediction')
    (out / 'synthetic_qa.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
