from __future__ import annotations
import json
import numpy as np
import pandas as pd
from study import HOME,dump


def ci(a):
    return np.quantile(a,[.025,.975],axis=0,method='linear')


def k_weights(ids,version):
    w=np.zeros(len(ids))
    for tail,sign in [('absolute_dataset_subject',1),('absolute_subject',-1),
                      ('relative_dataset_subject',-1),('relative_subject',1)]:
        w[ids.index(version+'_'+tail)]=sign
    return w


def main():
    lock=json.loads((HOME/'LOCK.json').read_text(encoding='utf-8'))
    cfg=lock['config']
    out=HOME/'results'
    status=json.loads((out/'status.json').read_text(encoding='utf-8'))
    assert status['state']=='predictions_complete'
    cells=lock['conditions']
    ids=[c['id'] for c in cells]
    with np.load(out/'scoring_labels.npz',allow_pickle=False) as f:
        y,subjects=f['y'],f['subjects']
    unique=np.unique(subjects)
    matrix=np.empty((len(unique),len(cells)))
    rows=[]
    for j,cell in enumerate(cells):
        with np.load(out/('predictions_'+cell['id']+'.npz'),allow_pickle=False) as f:
            assert np.array_equal(f['subjects'],subjects)
            assert np.array_equal(f['trial_index'],np.arange(len(y)))
            p=f['pred']
        for i,s in enumerate(unique):
            m=subjects==s
            assert set(np.unique(y[m]))=={0,1}
            score=100*np.mean([np.mean(p[m&(y==c)]==c) for c in [0,1]])
            matrix[i,j]=score
            rows.append(dict(**cell,subject=int(s),n_trials=int(m.sum()),bac_pct=score))
    pd.DataFrame(rows).to_csv(out/'subject_scores.csv',index=False)
    indices=np.random.default_rng(cfg['bootstrap_seed']).integers(0,len(unique),(cfg['bootstrap_draws'],len(unique)))
    draws=matrix[indices].mean(axis=1)
    point=matrix.mean(axis=0)
    np.savez_compressed(out/'bootstrap_draws.npz',condition_ids=np.array(ids),subject_ids=unique,
                        scores_pct=matrix,sample_indices=indices,mean_draws=draws)
    levels=[]
    for j,cell in enumerate(cells):
        lo,hi=ci(draws[:,j])
        levels.append(dict(**cell,n_subjects=len(unique),bac_pct=point[j],ci_low=lo,ci_high=hi))
    increments=[]
    changes=[]
    for v in cfg['versions']:
        for p in cfg['policies']:
            s,d=ids.index(v+'_'+p+'_subject'),ids.index(v+'_'+p+'_dataset_subject')
            lo,hi=ci(draws[:,d]-draws[:,s])
            increments.append(dict(version=v,policy=p,S_pct=point[s],DS_pct=point[d],
                increment_pp=point[d]-point[s],ci_low=lo,ci_high=hi))
    for p in cfg['policies']:
        for pipe in cfg['pipelines']:
            a,o=ids.index('adopted_'+p+'_'+pipe),ids.index('original_'+p+'_'+pipe)
            lo,hi=ci(draws[:,o]-draws[:,a])
            changes.append(dict(policy=p,pipeline=pipe,original_minus_adopted_pp=point[o]-point[a],ci_low=lo,ci_high=hi))
    wa,wo=k_weights(ids,'adopted'),k_weights(ids,'original')
    weights=np.stack([wa,wo,wo-wa],axis=1)
    estimates=point@weights
    kd=draws@weights
    sd=kd.std(axis=0,ddof=1)
    valid=sd>1e-12
    critical=float(np.quantile(np.abs((kd[:,valid]-estimates[valid])/sd[valid]).max(axis=1),.95)) if valid.any() else 0.
    contrasts=[]
    for j,name in enumerate(cfg['primary_family']):
        lo,hi=ci(kd[:,j])
        contrasts.append(dict(contrast=name,estimate_pp=estimates[j],ci_low=lo,ci_high=hi,
            simultaneous_low=estimates[j]-critical*sd[j],simultaneous_high=estimates[j]+critical*sd[j],
            simultaneous_family_size=3,simultaneous_critical_value=critical))
    pd.DataFrame(levels).to_csv(out/'performance_summary.csv',index=False)
    pd.DataFrame(increments).to_csv(out/'increments.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'version_changes.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    diag=pd.read_csv(out/'diagnostics.csv')
    diag.groupby(['condition','dataset','stage']).agg(references=('subject','size'),
        clipped_min=('clipped','min'),clipped_median=('clipped','median'),clipped_max=('clipped','max'),
        residual_median=('whitening_AI','median')).reset_index().to_csv(out/'diagnostic_summary.csv',index=False)
    status.update(state='complete' if status['relative_alignment_pass'] else 'relative_control_failed',
                  bootstrap_draws=cfg['bootstrap_draws'],bootstrap_seed=cfg['bootstrap_seed'],
                  unit='BAC percent; effects percentage points',version_selected=False,scope_execution_complete=True)
    dump(out/'status.json',status)
    print(pd.DataFrame(increments).to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


if __name__=='__main__':main()
