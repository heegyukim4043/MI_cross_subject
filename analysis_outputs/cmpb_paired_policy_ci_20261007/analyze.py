"""Subject-paired, dataset-stratified bootstrap using archived predictions only."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'analysis_outputs/mi_four_checks_20261007/precision_results'
DATASETS=['cho2017','lee2019','bnci2014001','physionetmi']
COUNTS=[52,54,9,109]
SCALES=['1','1e-6']
CONDITIONS=[('coherent_absolute','subject'),('coherent_absolute','dataset_subject'),('relative','subject'),('relative','dataset_subject')]
METRICS=['absolute_S','absolute_DS','relative_S','relative_DS','absolute_increment','relative_increment','increment_difference','absolute_DS_minus_relative_S']
COEF=np.column_stack([np.eye(4),[-1,1,0,0],[0,0,-1,1],[-1,1,1,-1],[0,1,-1,0]])
B=10000
SEED=20261007

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path, obj): path.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def fixture_test():
    values=np.array([[41.,50.,51.,55.],[61.,66.,67.,68.],[80.,90.,91.,92.]])
    ids=np.random.default_rng(9).integers(0,3,(20,3))
    vectorized=values[ids].mean(axis=1)@COEF
    literal=[]
    for idx in ids:
        a,b,c,d=values[idx].mean(axis=0)
        literal.append([a,b,c,d,b-a,d-c,(b-a)-(d-c),b-c])
    np.testing.assert_allclose(vectorized,literal,atol=1e-12)
    np.testing.assert_allclose((values@COEF)[ids].mean(axis=1),vectorized,atol=1e-12)
    np.testing.assert_allclose(np.zeros((3,4))@COEF,0)

def main():
    out=HERE/'results'
    out.mkdir(exist_ok=False)
    inputs=[SRC/f'predictions_{p}_{a}_{s}.npz' for s in SCALES for p,a in CONDITIONS]
    inputs += [SRC/'subject_results.csv',SRC/'comparison.csv',SRC/'diagnostics.csv',SRC/'lock.json',HERE/'PROTOCOL.md',Path(__file__)]
    lock={'started_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'resamples':B,'numpy':np.__version__,
          'rng':'PCG64','status':'exploratory_after_point_estimates','hashes':{str(p):sha(p) for p in inputs}}
    dump(out/'analysis_lock.json',lock)
    fixture_test()
    old=pd.read_csv(SRC/'subject_results.csv',dtype={'scale':str})
    assert not old.duplicated(['policy','alignment','scale','dataset','subject']).any()
    lookup=old.set_index(['policy','alignment','scale','dataset','subject'])
    blocks={name:np.empty((n,2,4)) for name,n in zip(DATASETS,COUNTS)}
    canonical={}
    scores=[]
    for si,scale in enumerate(SCALES):
        for ci,(policy,alignment) in enumerate(CONDITIONS):
            with np.load(SRC/f'predictions_{policy}_{alignment}_{scale}.npz',allow_pickle=False) as data:
                for name,n in zip(DATASETS,COUNTS):
                    y,ids,pred=[data[f'{key}__{name}'] for key in ['y','subjects','pred']]
                    assert y.shape==ids.shape==pred.shape and y.ndim==1
                    assert set(np.unique(y))=={0,1} and set(np.unique(pred))<={0,1}
                    if name not in canonical: canonical[name]=(y.copy(),ids.copy())
                    else:
                        assert np.array_equal(y,canonical[name][0]) and np.array_equal(ids,canonical[name][1])
                    subjects=np.unique(ids)
                    assert len(subjects)==n
                    for j,subject in enumerate(subjects):
                        m=ids==subject
                        assert set(np.unique(y[m]))=={0,1}
                        bac=100*np.mean([np.mean(pred[m][y[m]==label]==label) for label in [0,1]])
                        np.testing.assert_allclose(bac,100*balanced_accuracy_score(y[m],pred[m]),atol=1e-12)
                        archived=lookup.loc[(policy,alignment,scale,name,int(subject))]
                        np.testing.assert_allclose(bac,100*archived.bac,atol=1e-10)
                        assert int(archived.n_trials)==m.sum()
                        blocks[name][j,si,ci]=bac
                        scores.append(dict(dataset=name,subject=int(subject),scale=scale,policy=policy,alignment=alignment,n_trials=int(m.sum()),bac_pct=bac))
    assert len(scores)==len(old)==1792
    pd.DataFrame(scores).to_csv(out/'recomputed_subject_scores.csv',index=False)
    comparison=pd.read_csv(SRC/'comparison.csv',dtype={'scale':str})
    aggregates={name:arr.mean(axis=0) for name,arr in blocks.items()}
    stacked=np.stack(list(aggregates.values()))
    aggregates['equal_dataset']=stacked.mean(axis=0)
    aggregates['pooled_subject']=np.average(stacked,axis=0,weights=COUNTS)
    for row in comparison.itertuples():
        si=SCALES.index(row.scale)
        ci=0 if row.policy=='coherent_absolute' else 2
        values=aggregates[row.scope][si]
        np.testing.assert_allclose([values[ci],values[ci+1],values[ci+1]-values[ci]],[row.S_pct,row.DS_pct,row.increment_pp],atol=1e-10)
    rng=np.random.Generator(np.random.PCG64(SEED))
    boots={}
    descriptive=[]
    paired=[]
    for name in DATASETS:
        arr=blocks[name]
        derived=arr@COEF
        sampled=rng.integers(0,len(arr),(B,len(arr)))
        boots[name]=derived[sampled].mean(axis=1)
        for si,scale in enumerate(SCALES):
            for j,subject in enumerate(np.unique(canonical[name][1])):
                paired.append(dict(dataset=name,subject=int(subject),scale=scale,**dict(zip(METRICS,derived[j,si]))))
            for k,metric in enumerate(METRICS):
                v=derived[:,si,k]
                descriptive.append(dict(dataset=name,scale=scale,metric=metric,n=len(v),mean=np.mean(v),sd=np.std(v,ddof=1),median=np.median(v),min=np.min(v),max=np.max(v),positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum())))
    stack=np.stack([boots[n] for n in DATASETS])
    boots['equal_dataset']=stack.mean(axis=0)
    boots['pooled_subject']=np.average(stack,axis=0,weights=COUNTS)
    result=[]
    for scope in [*DATASETS,'equal_dataset','pooled_subject']:
        points=aggregates[scope]@COEF
        for si,scale in enumerate(SCALES):
            for k,metric in enumerate(METRICS):
                lo,hi=np.quantile(boots[scope][:,si,k],[.025,.975],method='linear')
                result.append(dict(scope=scope,scale=scale,metric=metric,n=COUNTS[DATASETS.index(scope)] if scope in DATASETS else 224,estimate=float(points[si,k]),ci95_low=float(lo),ci95_high=float(hi),unit='percent' if k<4 else 'pp',resamples=B,seed=SEED))
    pd.DataFrame(result).to_csv(out/'bootstrap_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_subject_contrasts.csv',index=False)
    pd.DataFrame(descriptive).to_csv(out/'descriptives.csv',index=False)
    np.savez_compressed(out/'aggregate_bootstrap_draws.npz',equal_dataset=boots['equal_dataset'],pooled_subject=boots['pooled_subject'],metric_names=np.asarray(METRICS),scales=np.asarray(SCALES))
    diag=pd.read_csv(SRC/'diagnostics.csv',dtype={'scale':str})
    d=diag.query("alignment == 'subject' and stage == 'subject' and scale == '1'")
    support=[]
    for name,n in zip(DATASETS,COUNTS):
        a=d.query('dataset == @name and policy == "coherent_absolute"').set_index('subject').sort_index()
        r=d.query('dataset == @name and policy == "relative"').set_index('subject').sort_index()
        assert len(a)==len(r)==n and a.index.equals(r.index)
        np.testing.assert_allclose(a.raw_condition,r.raw_condition,rtol=1e-12)
        mean_eigen=r.floor.to_numpy()/1e-8
        support.append(dict(dataset=name,n=n,declared_unit='V; Cho calibration provenance unresolved' if name=='cho2017' else 'V per saved preprocessing contract',mean_eigen_min=mean_eigen.min(),mean_eigen_median=np.median(mean_eigen),mean_eigen_max=mean_eigen.max(),absolute_epsilon=1e-8,epsilon_over_mean_eigen_median=np.median(1e-8/mean_eigen),absolute_clipped_min=int(a.clipped.min()),absolute_clipped_median=float(a.clipped.median()),absolute_clipped_max=int(a.clipped.max()),relative_clipped_max=int(r.clipped.max()),channels=21))
    pd.DataFrame(support).to_csv(out/'eigenvalue_scale_support.csv',index=False)
    qa={'subject_scores_recomputed':len(scores),'all_archive_bac_match':True,'all_comparison_points_match':True,'labels_subject_arrays_match':True,'fixture_tests_pass':True,'subjects_per_dataset':dict(zip(DATASETS,COUNTS)),'no_raw_eeg_opened':True,'no_training':True,'no_manuscript_edits':True,'completed_utc':datetime.now(timezone.utc).isoformat()}
    dump(out/'qa.json',qa)
    print(pd.DataFrame(result).query("scope in ['equal_dataset','pooled_subject'] and scale == '1' and metric in ['increment_difference','absolute_DS_minus_relative_S']").to_string(index=False))

if __name__=='__main__': main()
