from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from study import HOME, NAMES, sha, input_paths, dump


def main():
    out=HOME/'results'
    lock=json.loads((HOME/'LOCK.json').read_text(encoding='utf-8'))
    status=json.loads((out/'status.json').read_text(encoding='utf-8'))
    assert status['state']=='complete',status
    for p,h in {**lock['code_hashes'],**lock['anchor_hashes']}.items():
        assert sha(p)==h,('File hash changed',p)
    for n,p in input_paths().items():
        assert sha(p)==lock['input_hashes'][n]
    scores=pd.read_csv(out/'subject_scores.csv')
    assert len(scores)==224*8 and not scores.duplicated(['id','dataset','subject']).any()
    with np.load(out/'scoring_labels.npz',allow_pickle=False) as y:
        checked=0
        for cell,g in scores.groupby('id',sort=False):
            with np.load(out/('predictions_'+cell+'.npz'),allow_pickle=False) as f:
                for r in g.itertuples():
                    mask=y['subjects__'+r.dataset]==r.subject
                    score=100*balanced_accuracy_score(y['y__'+r.dataset][mask],f['pred__'+r.dataset][mask])
                    assert abs(score-r.bac_pct)<1e-10
                    checked+=1
    s=pd.read_csv(out/'summary.csv')
    c=pd.read_csv(out/'contrasts.csv')
    changes=pd.read_csv(out/'version_changes.csv')
    point={}
    with np.load(out/'bootstrap_draws.npz',allow_pickle=False) as b:
        ids=b['condition_ids'].tolist()
        direct={}
        for name in NAMES:
            wide=scores[scores.dataset==name].pivot(index='subject',columns='id',values='bac_pct').loc[:,ids]
            point[name]=wide.to_numpy().mean(axis=0)
            direct[name]=wide.to_numpy()[b['sample_indices__'+name]].mean(axis=1)
            np.testing.assert_allclose(direct[name],b[name],atol=1e-11,rtol=0)
        direct['equal_dataset']=sum(direct[n] for n in NAMES)/4
        point['equal_dataset']=sum(point[n] for n in NAMES)/4
        count={n:scores[(scores.dataset==n)&(scores.id==ids[0])].subject.nunique() for n in NAMES}
        direct['pooled_subject']=sum(direct[n]*count[n]/224 for n in NAMES)
        point['pooled_subject']=sum(point[n]*count[n]/224 for n in NAMES)
        for scope in ['equal_dataset','pooled_subject']:
            np.testing.assert_allclose(direct[scope],b[scope],atol=1e-11,rtol=0)
        for r in s.itertuples():
            i,j=ids.index(r.version+'_'+r.policy+'_subject'),ids.index(r.version+'_'+r.policy+'_dataset_subject')
            np.testing.assert_allclose([r.S_pct,r.DS_pct,r.increment_pp],
                [point[r.scope][i],point[r.scope][j],point[r.scope][j]-point[r.scope][i]],atol=1e-10,rtol=0)
            for vals,lo,hi in [(direct[r.scope][:,i],r.S_ci_low,r.S_ci_high),
                               (direct[r.scope][:,j],r.DS_ci_low,r.DS_ci_high),
                               (direct[r.scope][:,j]-direct[r.scope][:,i],r.increment_ci_low,r.increment_ci_high)]:
                np.testing.assert_allclose(np.percentile(vals,[2.5,97.5]),[lo,hi],atol=1e-10,rtol=0)
        for r in c.itertuples():
            def version_delta(a,v):
                return (a[...,ids.index(v+'_absolute_dataset_subject')]-a[...,ids.index(v+'_absolute_subject')]
                        -a[...,ids.index(v+'_relative_dataset_subject')]+a[...,ids.index(v+'_relative_subject')])
            def value(a):
                if r.contrast=='K_original_minus_adopted':
                    return version_delta(a,'original')-version_delta(a,'adopted')
                return version_delta(a,r.contrast.split('_',1)[1])
            estimate=value(point[r.scope])
            draws=value(direct[r.scope])
            assert abs(estimate-r.estimate_pp)<1e-10
            np.testing.assert_allclose(np.percentile(draws,[2.5,97.5]),[r.ci_low,r.ci_high],atol=1e-10,rtol=0)
        for r in changes.itertuples():
            i,j=ids.index('adopted_'+r.policy+'_'+r.pipeline),ids.index('original_'+r.policy+'_'+r.pipeline)
            assert abs(point[r.scope][j]-point[r.scope][i]-r.original_minus_adopted_pp)<1e-10
            np.testing.assert_allclose(np.percentile(direct[r.scope][:,j]-direct[r.scope][:,i],[2.5,97.5]),
                                       [r.ci_low,r.ci_high],atol=1e-10,rtol=0)
    anchors=json.loads((out/'anchor_checks.json').read_text(encoding='utf-8'))
    invariance=json.loads((out/'relative_invariance.json').read_text(encoding='utf-8'))
    assert len(anchors)==16 and all(r['predictions_equal'] for r in anchors)
    assert len(invariance)==8 and all(r['alignment_pass'] for r in invariance)
    assert status['actual_fits']==32
    dump(out/'verification.json',dict(subject_BACs_recomputed_with_sklearn=checked,
       all_summary_rows_verified=len(s),all_K_rows_verified=len(c),all_version_change_rows_verified=len(changes),
       all_bootstrap_draws_recomputed_from_saved_indices=True,all_pointwise_CIs_verified=True,
       anchors_reproduced=len(anchors),relative_controls=len(invariance),source_and_input_hashes_unchanged=True,
       raw_EEG_pipeline_independently_retrained=False,physical_scaling_validated=False))
    print('Verified1792subjectscores, all paired draws/pointwise CIs,16anchors and8relative controls.',flush=True)


if __name__=='__main__': main()
