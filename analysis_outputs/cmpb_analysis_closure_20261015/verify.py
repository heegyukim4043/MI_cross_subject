from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from study import HOME,sha,dump


def main():
    out=HOME/'results'
    lock=json.loads((HOME/'LOCK.json').read_text(encoding='utf-8'))
    status=json.loads((out/'status.json').read_text(encoding='utf-8'))
    assert status['state']=='complete'
    for p,h in {**lock['code_hashes'],**lock['anchor_hashes']}.items():assert sha(p)==h
    for r in lock['inputs'].values():assert sha(r['path'])==r['sha256']
    frozen=json.loads((out/'predictions_frozen.json').read_text(encoding='utf-8'))
    assert not frozen['target_labels_read'] and len(frozen['hashes'])==8
    for p,h in frozen['hashes'].items():assert sha(out/p)==h
    score=pd.read_csv(out/'subject_scores.csv')
    assert len(score)==87*8 and not score.duplicated(['id','subject']).any()
    count=0
    with np.load(out/'scoring_labels.npz',allow_pickle=False) as labels:
        for cell,g in score.groupby('id',sort=False):
            with np.load(out/('predictions_'+cell+'.npz'),allow_pickle=False) as f:
                for r in g.itertuples():
                    m=labels['subjects']==r.subject
                    value=100*balanced_accuracy_score(labels['y'][m],f['pred'][m])
                    assert abs(value-r.bac_pct)<1e-10
                    count+=1
    with np.load(out/'bootstrap_draws.npz',allow_pickle=False) as b:
        ids=b['condition_ids'].tolist()
        wide=score.pivot(index='subject',columns='id',values='bac_pct').loc[b['subject_ids'],ids].to_numpy()
        np.testing.assert_array_equal(wide,b['scores_pct'])
        draws=wide[b['sample_indices']].mean(axis=1)
        np.testing.assert_allclose(draws,b['mean_draws'],atol=1e-11,rtol=0)
    point=wide.mean(axis=0)
    for r in pd.read_csv(out/'performance_summary.csv').itertuples():
        i=ids.index(r.id)
        assert abs(point[i]-r.bac_pct)<1e-10
        np.testing.assert_allclose(np.percentile(draws[:,i],[2.5,97.5]),[r.ci_low,r.ci_high],atol=1e-10,rtol=0)
    for r in pd.read_csv(out/'increments.csv').itertuples():
        s,d=ids.index(r.version+'_'+r.policy+'_subject'),ids.index(r.version+'_'+r.policy+'_dataset_subject')
        assert abs(point[d]-point[s]-r.increment_pp)<1e-10
        np.testing.assert_allclose(np.percentile(draws[:,d]-draws[:,s],[2.5,97.5]),[r.ci_low,r.ci_high],atol=1e-10,rtol=0)
    for r in pd.read_csv(out/'version_changes.csv').itertuples():
        a,o=ids.index('adopted_'+r.policy+'_'+r.pipeline),ids.index('original_'+r.policy+'_'+r.pipeline)
        assert abs(point[o]-point[a]-r.original_minus_adopted_pp)<1e-10
        np.testing.assert_allclose(np.percentile(draws[:,o]-draws[:,a],[2.5,97.5]),[r.ci_low,r.ci_high],atol=1e-10,rtol=0)
    def K(x,v):
        return (x[...,ids.index(v+'_absolute_dataset_subject')]-x[...,ids.index(v+'_absolute_subject')]
                -x[...,ids.index(v+'_relative_dataset_subject')]+x[...,ids.index(v+'_relative_subject')])
    estimates=np.array([K(point,'adopted'),K(point,'original'),K(point,'original')-K(point,'adopted')])
    kd=np.stack([K(draws,'adopted'),K(draws,'original'),K(draws,'original')-K(draws,'adopted')],axis=1)
    c=pd.read_csv(out/'contrasts.csv')
    np.testing.assert_allclose(c.estimate_pp,estimates,atol=1e-10,rtol=0)
    for j,r in enumerate(c.itertuples()):
        np.testing.assert_allclose(np.percentile(kd[:,j],[2.5,97.5]),[r.ci_low,r.ci_high],atol=1e-10,rtol=0)
    sd=kd.std(axis=0,ddof=1)
    valid=sd>1e-12
    critical=np.quantile(np.abs((kd[:,valid]-estimates[valid])/sd[valid]).max(axis=1),.95) if valid.any() else 0.
    np.testing.assert_allclose(c.simultaneous_low,estimates-critical*sd,atol=1e-10,rtol=0)
    np.testing.assert_allclose(c.simultaneous_high,estimates+critical*sd,atol=1e-10,rtol=0)
    anchors=json.loads((out/'anchor_checks.json').read_text(encoding='utf-8'))
    checks=json.loads((out/'relative_checks.json').read_text(encoding='utf-8'))
    assert len(anchors)==4 and all(r['predictions_equal'] for r in anchors)
    assert all(r.get('pass_tolerance',True) for r in checks)
    assert status['completed_fits']==8 and status['target_labels_read_after_prediction_freeze']
    dump(out/'verification.json',dict(subject_BACs_recomputed_with_sklearn=count,all_draws_recomputed_from_saved_indices=True,
        all_pointwise_and_simultaneous_CIs_verified=True,adopted_prediction_anchors=4,
        relative_output_controls=sum('pass_tolerance' in r for r in checks),
        relative_prediction_flips=sum(r.get('prediction_flips',0) for r in checks),
        prediction_freeze_hashes_pass=True,source_and_input_hashes_pass=True,
        full_EEG_training_reimplemented_independently=False,physical_calibration_validated=False))
    print('Verified696subjectBACs/all paired draws/CIs/fouranchors; frozen inputs/code intact.',flush=True)


if __name__=='__main__':main()
