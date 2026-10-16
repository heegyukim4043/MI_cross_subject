from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from study import HOME, NAMES, dump


def interval(draws):
    return np.quantile(draws,[.025,.975],axis=0,method='linear')


def coefficients(condition_ids, version):
    w=np.zeros(len(condition_ids))
    for suffix,sign in [('absolute_dataset_subject',1),('absolute_subject',-1),
                        ('relative_dataset_subject',-1),('relative_subject',1)]:
        w[condition_ids.index(version+'_'+suffix)]=sign
    return w


def main():
    lock=json.loads((HOME/'LOCK.json').read_text(encoding='utf-8'))
    cfg=lock['config']
    out=HOME/'results'
    status=json.loads((out/'status.json').read_text(encoding='utf-8'))
    assert status['state']=='predictions_complete'
    ids=[r['id'] for r in lock['conditions']]
    with np.load(out/'scoring_labels.npz',allow_pickle=False) as f:
        labels={k:f[k] for k in f.files}
    keys=[(n,int(s)) for n in NAMES for s in np.unique(labels['subjects__'+n])]
    scores=np.empty((len(keys),len(ids)))
    rows=[]
    for j,cell in enumerate(lock['conditions']):
        with np.load(out/('predictions_'+cell['id']+'.npz'),allow_pickle=False) as f:
            for i,(name,subject) in enumerate(keys):
                y=labels['y__'+name]
                subjects=labels['subjects__'+name]
                assert np.array_equal(f['subjects__'+name],subjects)
                mask=subjects==subject
                assert set(np.unique(y[mask]))=={0,1}
                pred=f['pred__'+name]
                score=100*np.mean([np.mean(pred[mask&(y==c)]==c) for c in [0,1]])
                scores[i,j]=score
                rows.append(dict(**cell,dataset=name,subject=subject,n_trials=int(mask.sum()),bac_pct=score))
    pd.DataFrame(rows).to_csv(out/'subject_scores.csv',index=False)
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    point,draw,sampled={},{},{}
    for name in NAMES:
        index=np.array([i for i,(n,s) in enumerate(keys) if n==name])
        a=scores[index]
        point[name]=a.mean(axis=0)
        sampled[name]=rng.integers(0,len(index),size=(cfg['bootstrap_draws'],len(index)))
        draw[name]=a[sampled[name]].mean(axis=1)
    point['equal_dataset']=np.mean([point[n] for n in NAMES],axis=0)
    draw['equal_dataset']=np.mean([draw[n] for n in NAMES],axis=0)
    point['pooled_subject']=scores.mean(axis=0)
    weights={n:sum(k[0]==n for k in keys)/len(keys) for n in NAMES}
    draw['pooled_subject']=sum(weights[n]*draw[n] for n in NAMES)
    np.savez_compressed(out/'bootstrap_draws.npz',condition_ids=np.array(ids),**draw,
                        **{'sample_indices__'+n:sampled[n] for n in NAMES})
    summary=[]
    contrasts=[]
    changes=[]
    scopes=(*NAMES,'equal_dataset','pooled_subject')
    for scope in scopes:
        for version in cfg['versions']:
            for policy in cfg['policies']:
                s=ids.index(version+'_'+policy+'_subject')
                ds=ids.index(version+'_'+policy+'_dataset_subject')
                sl,sh=interval(draw[scope][:,s])
                dl,dh=interval(draw[scope][:,ds])
                il,ih=interval(draw[scope][:,ds]-draw[scope][:,s])
                summary.append(dict(scope=scope,version=version,policy=policy,
                    S_pct=point[scope][s],S_ci_low=sl,S_ci_high=sh,DS_pct=point[scope][ds],DS_ci_low=dl,DS_ci_high=dh,
                    increment_pp=point[scope][ds]-point[scope][s],increment_ci_low=il,increment_ci_high=ih))
        coefficients_by_version={v:coefficients(ids,v) for v in cfg['versions']}
        family={'K_adopted':coefficients_by_version['adopted'],
                'K_original':coefficients_by_version['original'],
                'K_original_minus_adopted':coefficients_by_version['original']-coefficients_by_version['adopted']}
        values=np.array([point[scope]@family[c] for c in cfg['primary_family']])
        family_draw=np.stack([draw[scope]@family[c] for c in cfg['primary_family']],axis=1)
        sd=family_draw.std(axis=0,ddof=1)
        valid=sd>1e-12
        critical=float(np.quantile(np.abs((family_draw[:,valid]-values[valid])/sd[valid]).max(axis=1),.95,method='linear'))
        for j,name in enumerate(cfg['primary_family']):
            lo,hi=interval(family_draw[:,j])
            r=dict(scope=scope,contrast=name,estimate_pp=values[j],ci_low=lo,ci_high=hi)
            if scope=='equal_dataset':
                r.update(simultaneous_low=values[j]-critical*sd[j],simultaneous_high=values[j]+critical*sd[j],
                         simultaneous_family_size=len(family),simultaneous_critical_value=critical)
            contrasts.append(r)
        for policy in cfg['policies']:
            for pipeline in cfg['pipelines']:
                a=ids.index('adopted_'+policy+'_'+pipeline)
                o=ids.index('original_'+policy+'_'+pipeline)
                lo,hi=interval(draw[scope][:,o]-draw[scope][:,a])
                changes.append(dict(scope=scope,policy=policy,pipeline=pipeline,
                    original_minus_adopted_pp=point[scope][o]-point[scope][a],ci_low=lo,ci_high=hi))
    summary=pd.DataFrame(summary)
    contrast_frame=pd.DataFrame(contrasts)
    summary.to_csv(out/'summary.csv',index=False)
    contrast_frame.to_csv(out/'contrasts.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'version_changes.csv',index=False)
    d=pd.read_csv(out/'diagnostics.csv')
    dg=d.groupby(['condition','dataset','stage']).agg(references=('subject','size'),
        clipped_median=('clipped','median'),clipped_min=('clipped','min'),clipped_max=('clipped','max'),
        floor_over_mean_median=('floor_over_mean','median'),residual_median=('whitening_AI','median')).reset_index()
    dg.to_csv(out/'diagnostic_summary.csv',index=False)
    status.update(state='complete' if status['relative_alignment_check_pass'] else 'completed_with_failed_relative_control',
        subjects=len(keys),bootstrap_draws=cfg['bootstrap_draws'],bootstrap_seed=cfg['bootstrap_seed'],
        result_unit='BAC percent; effects percentage points',equivalence_claim=False,amplitude_version_selected=False)
    dump(out/'status.json',status)
    primary=summary[summary.scope=='equal_dataset']
    pc=contrast_frame[contrast_frame.scope=='equal_dataset']
    print(primary.to_string(index=False),flush=True)
    print(pc.to_string(index=False),flush=True)
    lines=['# Cho amplitude-version comparison: numerical results','',
      'Exploratory matched pipeline comparison; original constants retained. Not physical unit validation.',
      'Four fixed development datasets/224subjects; dataset-equal weighting; all32fits rerun.',
      'Intervals are conditional paired subject-bootstrap95% percentile CIs; no retraining/new-site uncertainty.','',
      '| Cho version | Policy | S BAC% | DS BAC% | DS-S pp | 95%CI |','|---|---|---:|---:|---:|---|']
    for r in primary.itertuples():
        lines.append(f'| {r.version} | {r.policy} | {r.S_pct:.3f} | {r.DS_pct:.3f} | {r.increment_pp:+.3f} | [{r.increment_ci_low:+.3f}, {r.increment_ci_high:+.3f}] |')
    lines.extend(['','| Contrast | Estimate pp | Pointwise95%CI | Approx simultaneous95% band |',
                  '|---|---:|---|---|'])
    for r in pc.itertuples():
        lines.append(f'| {r.contrast} | {r.estimate_pp:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | [{r.simultaneous_low:+.3f}, {r.simultaneous_high:+.3f}] |')
    (HOME/'NUMERICAL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
