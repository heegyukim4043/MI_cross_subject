from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

HOME=Path(__file__).resolve().parent
OUT=HOME/'results'
lock=json.loads((HOME/'LOCK.json').read_text(encoding='utf-8'))
status=json.loads((OUT/'status.json').read_text(encoding='utf-8'))
assert status['state']=='complete'
for path,value in lock['code_hashes'].items():
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    assert h.hexdigest()==value, ('Lock mismatch',path)
score=pd.read_csv(OUT/'subject_scores.csv')
summary=pd.read_csv(OUT/'summary.csv')
diags=pd.read_csv(OUT/'diagnostics.csv')
execution=pd.read_csv(OUT/'execution.csv').fillna('')
assert len(score)==31*224
assert not score.duplicated(['condition','dataset','subject']).any()
assert len(summary)==6*(10*2+1)
assert sum(execution.new_fits)==status['actual_fits']
assert diags.clipped.between(0,21).all()
assert (diags['floor']>=0).all()
assert (diags.min_eigenvalue>0).all()
with np.load(OUT/'scoring_labels.npz',allow_pickle=False) as y:
    checked=0
    for condition,group in score.groupby('condition',sort=False):
        with np.load(OUT/('predictions_'+condition+'.npz'),allow_pickle=False) as f:
            for row in group.itertuples():
                subjects=y['subjects__'+row.dataset]
                mask=subjects==row.subject
                actual=100*balanced_accuracy_score(y['y__'+row.dataset][mask],f['pred__'+row.dataset][mask])
                assert abs(actual-row.bac_pct)<1e-10
                checked+=1
for row in execution.itertuples():
    if row.reused_from:
        with np.load(OUT/('predictions_'+row.condition+'.npz'),allow_pickle=False) as a, np.load(OUT/('predictions_'+row.reused_from+'.npz'),allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
for row in summary.itertuples():
    s_id='relative_subject' if row.path=='relative' else 'fixed_'+row.epsilon_label+'_subject'
    ds_id='relative_dataset_subject' if row.path=='relative' else ('fixed_' if row.path=='fixed_post' or row.epsilon_label=='1e-8' else 'tied_')+row.epsilon_label+'_dataset_subject'
    values=[]
    for identifier in [s_id,ds_id]:
        g=score[score.condition==identifier]
        estimate=g.groupby('dataset').bac_pct.mean().mean() if row.scope=='equal_dataset' else g.bac_pct.mean() if row.scope=='pooled_subject' else g[g.dataset==row.scope].bac_pct.mean()
        values.append(estimate)
    np.testing.assert_allclose(values,[row.S_pct,row.DS_pct],atol=1e-10,rtol=0)
    assert abs(values[1]-values[0]-row.increment_pp)<1e-10
with np.load(OUT/'bootstrap_draws.npz',allow_pickle=False) as b:
    ids=b['condition_ids'].tolist()
    for row in summary.itertuples():
        s_id='relative_subject' if row.path=='relative' else 'fixed_'+row.epsilon_label+'_subject'
        ds_id='relative_dataset_subject' if row.path=='relative' else ('fixed_' if row.path=='fixed_post' or row.epsilon_label=='1e-8' else 'tied_')+row.epsilon_label+'_dataset_subject'
        d=b[row.scope][:,ids.index(ds_id)]-b[row.scope][:,ids.index(s_id)]
        expected=np.percentile(d,[2.5,97.5],method='linear')
        np.testing.assert_allclose(expected,[row.increment_ci_low,row.increment_ci_high],atol=1e-10,rtol=0)
qa=dict(subject_BACs_recomputed_with_sklearn=checked,summary_rows_verified=len(summary),
        all_increment_CIs_verified_from_saved_joint_draws=True,
        cached_predictions_byte_equivalent=True,locked_source_hashes_unchanged=True,
        subject_counts=score[score.condition==score.condition.iloc[0]].groupby('dataset').subject.nunique().to_dict(),
        manuscript_or_existing_results_modified=False,
        raw_EEG_model_pipeline_reexecuted_by_verifier=False,
        visual_review='pending')
(OUT/'verification.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
print(json.dumps(qa,indent=2))
