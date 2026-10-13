"""Recompute BAC and paired intervals from frozen predictions, without refitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main(root, folder):
    out = folder/'results'
    lock = json.loads((out/'lock.json').read_text(encoding='utf-8'))
    frozen = json.loads((out/'predictions_frozen.json').read_text(encoding='utf-8'))
    status = json.loads((out/'status.json').read_text(encoding='utf-8'))
    assert status['state'] == 'complete' and status['completed_fits'] == 4
    for path, expected in lock['code_and_protocol'].items():
        assert sha(path) == expected, path
    for entry in lock['inputs'].values():
        assert sha(entry['path']) == entry['sha256']
    for name, expected in frozen['hashes'].items():
        assert sha(out/name) == expected
    for name, expected in json.loads((out/'output_hashes.json').read_text(encoding='utf-8')).items():
        assert sha(out/name) == expected
    with np.load(lock['inputs']['dreyer2023']['path']) as f:
        y, ids = f['y'].astype(int), f['subjects'].astype(int)
    subjects = np.unique(ids)
    assert len(subjects) == 87 and len(y) == 20792
    cols = []
    for policy, alignment in lock['conditions']:
        with np.load(out/('predictions_'+policy+'_'+alignment+'.npz')) as f:
            assert np.array_equal(f['subjects'], ids)
            assert np.array_equal(f['trial_index'], np.arange(len(y)))
            pred = f['pred']
            decision = f['decision']
        assert np.array_equal(pred, (decision > 0).astype(int))
        scores = []
        for s in subjects:
            m = ids == s
            t0 = np.count_nonzero(m & (y == 0))
            t1 = np.count_nonzero(m & (y == 1))
            assert t0 and t1
            correct0 = np.count_nonzero(m & (y == 0) & (pred == 0))
            correct1 = np.count_nonzero(m & (y == 1) & (pred == 1))
            scores.append(50 * (correct0/t0 + correct1/t1))
        cols.append(scores)
    matrix = np.asarray(cols).T
    saved_rows = pd.read_csv(out/'subject_results.csv')
    for j, (p,a) in enumerate(lock['conditions']):
        g = saved_rows[(saved_rows.policy == p) & (saved_rows.alignment == a)].sort_values('subject')
        assert np.array_equal(g.subject.values, subjects)
        assert np.allclose(g.bac_pct.values, matrix[:,j], rtol=0, atol=1e-12)
    names = ['policy_increment_difference', 'absolute_increment', 'relative_increment',
             'absolute_DS_minus_relative_S', 'relative_S_minus_absolute_S']
    weights = np.array([[-1,1,1,-1],[-1,1,0,0],[0,0,-1,1],[0,1,-1,0],[-1,0,1,0]])
    d = matrix @ weights.T
    rng = np.random.default_rng(20261008)
    # Independently resample one row-index vector per draw instead of using the runner's vectorized reduction.
    boot = np.array([d[rng.integers(len(subjects), size=len(subjects))].mean(axis=0)
                     for _ in range(10000)])
    with np.load(out/'bootstrap_draws.npz') as f:
        max_bootstrap_difference = float(np.max(np.abs(f['draws'] - boot)))
        assert np.allclose(f['draws'], boot, rtol=0, atol=1e-12)
        assert np.allclose(f['scores_pct'], matrix, rtol=0, atol=1e-12)
        assert np.array_equal(f['y'], y)
    contrasts = pd.read_csv(out/'contrasts.csv').set_index('contrast').loc[names]
    assert np.allclose(contrasts.mean_pp, d.mean(axis=0), rtol=0, atol=1e-12)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    assert np.allclose(contrasts.ci95_low, lo, rtol=0, atol=1e-12)
    assert np.allclose(contrasts.ci95_high, hi, rtol=0, atol=1e-12)
    counts = pd.Series(ids).value_counts()
    audit = pd.read_csv(out/'input_audit.csv')
    assert len(audit) == 311 and (audit.eig_min > 0).all()
    target_audit = audit[audit.dataset == 'dreyer2023']
    old_dir = root/'analysis_outputs/cmpb_confirmation_dreyer2023'
    old = pd.read_csv(old_dir/'cross_subject_results.csv')
    old_difference = {}
    for j, condition in ((2, 'relative_subject'), (3, 'relative_dataset_subject')):
        g = old[old.condition == condition].sort_values('subject')
        assert np.array_equal(g.subject.values, subjects)
        old_difference[condition] = float(np.max(np.abs(g.bac.values*100-matrix[:,j])))
    h1 = pd.read_csv(old_dir/'confirmatory_hypotheses.csv').set_index('hypothesis').loc['H1']
    loso = pd.read_csv(old_dir/'loso_subject_results.csv').pivot(index='subject', columns='floor', values='bac')
    assert np.isclose((loso.relative-loso.legacy).mean()*100, h1.estimate, atol=1e-12)
    qa = dict(passed=True, source_and_target_hashes_verified=True,
        code_and_protocol_hashes_verified=True, frozen_prediction_hashes_verified=True,
        independently_recomputed_bac=True, independently_recomputed_10000_draws=True,
        max_bootstrap_difference=max_bootstrap_difference, n_subjects=87, n_trials=len(y),
        trials_per_subject_min=int(counts.min()), trials_per_subject_max=int(counts.max()),
        positive_reference_count=len(audit),
        target_absolute_clipping_median=float(target_audit.absolute_clipped.median()),
        target_absolute_clipping_range=[int(target_audit.absolute_clipped.min()),int(target_audit.absolute_clipped.max())],
        target_relative_clipping_max=int(target_audit.relative_clipped.max()),
        historical_relative_max_bac_difference_pp=old_difference,
        old_h1_recomputed_pp=float(h1.estimate), old_h1_archived_holm_p=float(h1.holm_p),
        inference='post-hoc; fixed source models and target cohort; secondary CIs pointwise',
        runtime_seconds=float(status['elapsed_seconds']))
    (folder/'verification.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    table = []
    for j, (p,a) in enumerate(lock['conditions']):
        table.append(f"| {p} | {a} | {matrix[:,j].mean():.2f} | {matrix[:,j].std(ddof=1):.2f} |")
    ctable = [f"| {n} | {r.mean_pp:+.3f} | [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] |"
              for n,r in contrasts.iterrows()]
    r = contrasts.loc['policy_increment_difference']
    statement = ('The interval is positive: the directional policy-dependence pattern was observed '
                 'in this external dataset under the specified post-hoc protocol.' if r.ci95_low > 0
                 else 'The interval does not establish the predicted positive policy contrast; do not claim replication.')
    report = '\n'.join([
        '# Dreyer external policy contrast: post-hoc results', '',
        '## Design and execution', '',
        'Four new CSP-LDA fits; source=224 development subjects, target=87 Dreyer subjects '
        '(20,792 trials); 18 shared channels, 100 Hz, first 200 samples. '
        'Both policies use float64 EA covariance/eigendecomposition and float32 aligned '
        'arrays/downstream input. Same source-only normalization and eight-filter CSP-LDA.', '',
        'The original confirmation used a different Cho file hash. All four conditions '
        'were rerun instead of joining old relative results to new absolute fits. '
        'Relative subject scores nevertheless agree with the archived scores to '
        'floating-point tolerance; this was checked after the run, not assumed. '
        'Cho physical calibration remains unresolved.', '',
        'The first preflight stopped on a NumPy version mismatch before any fit. '
        'It is preserved under preflight_wrong_environment/. The successful run used '
        'the archived .venv versions (NumPy 1.26.4, SciPy 1.13.1, sklearn 1.5.2), '
        f'passed eight synthetic checks and completed in {status["elapsed_seconds"]:.1f} seconds.', '',
        '## Performance', '', '| Policy | Pipeline | Mean BAC (%) | Subject SD |',
        '|---|---|---:|---:|', *table, '',
        '## Paired contrasts', '', '| Contrast | Mean (pp) | 95% percentile CI |',
        '|---|---:|---:|', *ctable, '', statement, '',
        'The absolute two-stage and relative single-stage scores are close, but no '
        'equivalence test was specified for this contrast. Neither a zero-containing '
        'relative increment CI nor similar means proves no benefit in all populations.', '',
        '## Historical H1 (separate analysis)', '',
        f'Original preregistered Dreyer LOSO relative-minus-legacy: {h1.estimate:.3f} pp; '
        f'archived Holm p={h1.holm_p:.6g}. Verified against its subject CSV. '
        'This is not the new cross-dataset policy-by-stage contrast and has its original '
        'legacy precision/protocol. Do not merge significance or confirmation status.', '',
        '## Boundaries', '',
        '- Specification was fixed before these new fits, but after development and Dreyer '
        'results were known. Label this a post-hoc external-data extension, not sealed confirmation.',
        '- Bootstrap resamples the 87 paired subject rows 10,000 times (seed 20261008). '
        'Models, cohort references and recorded trials stay fixed; source-pool, '
        'shared-reference and new-site uncertainty are not represented.',
        '- S uses each target subject; D->S additionally uses the target cohort. '
        'Access is matched between policies, not between pipelines. No causal/online claim.',
        '- The held-out cohort is new relative to development, but the research team, '
        'implementation and source data are shared. This is not independent implementation replication.',
        '- No regularization dose-response, prevalence estimate, optimality or general '
        'claim against hierarchical alignment follows from two policies.', '',
        '## Manuscript-ready insertion (not yet inserted)', '',
        f'In a post-hoc external-data extension using four matched CSP-LDA fits, '
        f'DatasetEA increased Dreyer BAC by {contrasts.loc["absolute_increment"].mean_pp:.2f} pp '
        f'under coherent absolute regularization and {contrasts.loc["relative_increment"].mean_pp:.2f} pp '
        f'under relative regularization. The paired difference in increments was '
        f'{r.mean_pp:.2f} pp (95% CI {r.ci95_low:.2f} to {r.ci95_high:.2f}; '
        '87 subjects; conditional on fitted models and references). '
        f'Absolute two-stage BAC ({matrix[:,1].mean():.2f}%) was close to relative '
        f'single-stage BAC ({matrix[:,2].mean():.2f}%), without a pre-specified equivalence claim.', '',
        'Adding this result requires corresponding Methods and Supplementary updates. '
        'Existing r3 Word/MD, Page and Supplementary have not been edited by this analysis.', '',
        '## Reproduction', '',
        'PROTOCOL.md, results/lock.json, run.py; predictions_*.npz and model_*.npz; '
        'subject_results.csv, subject_contrasts.csv, performance_summary.csv, contrasts.csv; '
        'input_audit.csv, diagnostics.csv, bootstrap_draws.npz, verification.json. '
        'Model archives store normalizers/CSP filters/scaler/linear decision parameters; '
        'alignment must still be reconstructed from the locked inputs. '
        'Trial indices are positions in the hashed NPZ, not native acquisition IDs.', ''
    ])
    (folder/'REPORT.md').write_text(report, encoding='utf-8')
    final = f'''# Dreyer 정책 대비 사후 외부 확인

## 결론

Dreyer 87명에서도 추가 단계 이득의 정책 의존성이 같은 방향으로 관측됐다.
주 대비는 {r.mean_pp:+.2f} pp [95% CI {r.ci95_low:.2f}, {r.ci95_high:.2f}]이다.
새로운 사전 지정 확인이 아니라, 기존 결과를 본 뒤 설계를 고정한 사후 외부 확장이다.

| 정책 | SubjectEA BAC | DatasetEA→SubjectEA BAC | 추가 이득 |
|---|---:|---:|---:|
| Coherent absolute | {matrix[:,0].mean():.2f}% | {matrix[:,1].mean():.2f}% | {contrasts.loc['absolute_increment'].mean_pp:+.2f} pp |
| Relative | {matrix[:,2].mean():.2f}% | {matrix[:,3].mean():.2f}% | {contrasts.loc['relative_increment'].mean_pp:+.2f} pp |

- absolute 두 단계와 relative 단일 단계 차이: {contrasts.loc['absolute_DS_minus_relative_S'].mean_pp:+.3f} pp [{contrasts.loc['absolute_DS_minus_relative_S'].ci95_low:+.3f}, {contrasts.loc['absolute_DS_minus_relative_S'].ci95_high:+.3f}]. 정식 동등성 판정은 하지 않았다.
- 기존 사전 지정 LOSO H1: relative−legacy {h1.estimate:+.3f} pp, 보관된 Holm p={h1.holm_p:.6g}. 새로운 증분 대비와 구분한다.

## 실행·검수

기존 Dreyer 분석은 Cho 보정 전 입력을 사용했으므로 absolute 두 조건만 추가하지 않고
18채널에서 네 조건을 모두 재실행했다. Source는 개발 4개 데이터셋 224명,
target은 Dreyer 87명·20,792 trial이다. 동일 환경·연산 정밀도·분류기로 비교했다.
라벨은 네 조건 예측 저장 후 평가에만 사용했다.

초기 환경 검사는 패키지 버전 불일치로 학습 전에 중단됐다. 중단 기록은 보존했고,
기존 .venv에서 합성 8개 검사와 본 실행을 마쳤다. 총 실행 시간은 약 {status['elapsed_seconds']/60:.1f}분이다.
원본·실행 코드·예측 해시를 재확인했고, 예측에서 BAC와 10,000회 대응 bootstrap을
별도 검산하여 저장 결과와 일치함을 확인했다. 기존 relative 피험자별 BAC도 일치한다.

## 해석 범위와 후속 작업

개발 자료 밖에서도 같은 양상이라는 근거가 추가됐다. 다만 같은 구현과 source 자료를
공유하므로 독립 구현 재현은 아니다. CI는 고정된 모델·cohort 참조에 조건부이며,
S와 D→S 사이의 target 정보량까지 같게 통제한 비교도 아니다. 연속적 강도 효과나
분야 전체의 빈도는 입증하지 않는다. Relative 이득의 CI가 0을 포함한다는 사실만으로
이득이 없거나 모든 계층 정렬이 불필요하다고 결론 내리지 않는다.

원고 수치·그림·Word는 아직 수정하지 않았다. REPORT.md에 영문 삽입문을 준비했다.
반영 시 Methods에 네 조건의 사후 비교 및 bootstrap 조건을 함께 추가하고,
Supplementary에 예측·대비 출처를 연결해야 한다. Cho 단위 출처와 저자 선언은 여전히 남아 있다.
'''
    (folder/'REPORT_KO.md').write_text(final, encoding='utf-8')
    with (root/'MI_test/progress.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## Dreyer external policy contrast verified (2026-10-08)\n\n'
            f'- Four new matched fits: absolute S={matrix[:,0].mean():.3f}%, D->S={matrix[:,1].mean():.3f}%; '
            f'relative S={matrix[:,2].mean():.3f}%, D->S={matrix[:,3].mean():.3f}%. '
            f'Primary increment difference {r.mean_pp:+.3f}pp [{r.ci95_low:.3f}, {r.ci95_high:.3f}], '
            '95% subject-paired percentile bootstrap, 10,000 draws, seed20261008. '
            '87 target subjects/20,792 trials; all224 development subjects as source,18channels.\n'
            '- Input/code/frozen prediction/output hashes verified; independent BAC and '
            'loop-based bootstrap reproduction agree with saved values. All311 reference '
            'matrices SPD. Original relative per-subject BAC agrees within floating-point '
            'tolerance despite rerunning, consistent with scalar invariance, not a prior assumption.\n'
            f'- Original H1 independently read back as {h1.estimate:+.3f}pp; archived Holm '
            f'p={h1.holm_p:.6g}. Separate historical LOSO evidence, not the new interaction.\n'
            '- This is a post-hoc external-data extension, not a new preregistered/sealed '
            'confirmation or independent implementation replication. Conditional CI excludes '
            'source-training/cohort-reestimation/new-site uncertainty; no equivalence claim '
            'or dose-response. S versus D->S access differs; policies are matched within pipelines.\n'
            '- Report: `analysis_outputs/cmpb_dreyer_policy_extension_20261008/REPORT_KO.md` '
            '(English REPORT.md includes manuscript-ready insertion); evidence in results/ '
            'and verification.json. Initial wrong-environment preflight stopped before fitting '
            'and remains archived; successful environment matches development. No historical '
            'results overwritten. Manuscript/Word/Page/Supplementary not edited; next integrate '
            'new post-hoc methods/results and synchronize Supplementary, retain Cho provenance '
            'and author-declaration limitations.\n')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--folder', type=Path, required=True)
    args = ap.parse_args()
    main(args.root, args.folder)
