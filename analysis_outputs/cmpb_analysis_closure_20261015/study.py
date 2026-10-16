from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
HOME=Path(__file__).resolve().parent
OLD=ROOT/'analysis_outputs/cmpb_dreyer_policy_extension_20261008'
DEV=ROOT/'analysis_outputs/cmpb_cho_amplitude_versions_20261015'
SOURCES=('cho2017','lee2019','bnci2014001','physionetmi')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def module(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def cells(cfg):
    return [dict(id='_'.join((v,p,s)),version=v,policy=p,pipeline=s)
            for v in cfg['versions'] for p in cfg['policies'] for s in cfg['pipelines']]


def old_prediction(policy,pipeline):
    p='coherent_absolute' if policy=='absolute' else policy
    return OLD/'results'/('predictions_'+p+'_'+pipeline+'.npz')


def prepare():
    assert not (HOME/'LOCK.json').exists(),'Never overwrite a lock'
    cfg=json.loads((HOME/'config.json').read_text(encoding='utf-8'))
    old=json.loads((OLD/'results/lock.json').read_text(encoding='utf-8'))
    dev=json.loads((DEV/'LOCK.json').read_text(encoding='utf-8'))
    for p,h in old['code_and_protocol'].items():assert sha(p)==h,('Archived code changed',p)
    external=module(OLD/'run.py','archived_external')
    _,_,helper_hashes=external.load_operator(ROOT)
    for p,h in helper_hashes.items():assert old['code_and_protocol'][p]==h
    inputs=dict(old['inputs'])
    inputs['cho_original']=dict(path=str(ROOT/'MI_test/preprocessed_sfreq100/cho2017_moabb_uncorrected.npz'),
                                sha256=dev['input_hashes']['cho_original'])
    assert inputs['cho_original']['sha256']==old['old_cho_sha256']
    for n,r in inputs.items():assert sha(r['path'])==r['sha256'],('Input mismatch',n)
    tests=subprocess.run([sys.executable,str(HOME/'test_study.py')],capture_output=True,text=True)
    (HOME/'synthetic_tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    assert tests.returncode==0,tests.stdout+tests.stderr
    frozen={str(HOME/n):sha(HOME/n) for n in ['config.json','PROTOCOL.md','study.py','test_study.py','analyze.py','verify.py']}
    frozen.update(old['code_and_protocol'])
    for p in [OLD/'results/lock.json',DEV/'LOCK.json',DEV/'study.py']:
        frozen[str(p)]=sha(p)
    anchors={str(old_prediction(p,s)):sha(old_prediction(p,s)) for p in cfg['policies'] for s in cfg['pipelines']}
    env=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    assert env==old['environment'],('Environment mismatch',env,old['environment'])
    dump(HOME/'LOCK.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),config=cfg,conditions=cells(cfg),
        inputs=inputs,channels=old['channels'],code_hashes=frozen,anchor_hashes=anchors,environment=env,
        fits=8,source_subjects=224,target_subjects=87,target_trials=20792,
        scope='last exploratory external amplitude-version follow-up; not confirmatory registration'))
    print('Locked8fits; synthetic tests and all input/anchor hashes pass. No EEG arrays loaded.',flush=True)


def load_target(path,channels):
    with np.load(path,allow_pickle=True) as f:
        names=f['ch_names'].astype(str).tolist()
        assert len(set(names))==len(names)
        assert float(np.asarray(f['sfreq']).ravel()[0])==100
        x=f['X'][:,[names.index(c) for c in channels],:200].astype(np.float32)
        subjects=f['subjects'].astype(int)
    assert x.shape==(len(subjects),18,200) and np.isfinite(x).all()
    return dict(X=x,subjects=subjects)


def run():
    lock=json.loads((HOME/'LOCK.json').read_text(encoding='utf-8'))
    cfg=lock['config']
    for p,h in {**lock['code_hashes'],**lock['anchor_hashes']}.items():assert sha(p)==h,('Frozen change',p)
    for n,r in lock['inputs'].items():assert sha(r['path'])==r['sha256'],('Input changed',n)
    out=HOME/'results'
    out.mkdir(exist_ok=False)
    started=time.perf_counter()
    ext=module(OLD/'run.py','archived_external')
    op,ns,_=ext.load_operator(ROOT)
    dev=module(DEV/'study.py','development_version_io')
    data={n:dev.load_input(Path(lock['inputs'][n]['path']),lock['channels']) for n in SOURCES}
    orig=dev.load_input(Path(lock['inputs']['cho_original']['path']),lock['channels'])
    dev.check_metadata(data['cho2017'],orig)
    error=dev.norm_error(data['cho2017']['X'],orig['X']*np.float32(cfg['amplitude_factor']))
    assert error<=cfg['amplitude_relation_tolerance']
    target=load_target(lock['inputs']['dreyer2023']['path'],lock['channels'])
    assert 'y' not in target
    assert len(np.unique(target['subjects']))==87 and len(target['subjects'])==20792
    assert sum(len(np.unique(d['subjects'])) for d in data.values())==224
    dump(out/'input_audit.json',dict(cho_used_shape=list(orig['X'].shape),cho_amplitude_relative_error=error,
        matched_labels_subjects_channels=True,cho_exact_binary_relation=bool(np.array_equal(data['cho2017']['X'],orig['X']*np.float32(.03125))),
        source_subjects=224,target_subjects=87,target_trials=len(target['subjects']),channels=lock['channels'],
        target_labels_read=False,physical_calibration_validated=False))
    print('G0 audit passed:18channels/224source/87target; Cho amplitude/order match; target labels unread.',flush=True)
    anchors,checks,diags,paths=[],[],[],[]
    original_data=dict(data,cho2017=orig)
    for i,cell in enumerate(lock['conditions'],1):
        current=data if cell['version']=='adopted' else original_data
        arrays={}
        policy='coherent_absolute' if cell['policy']=='absolute' else cell['policy']
        for name in (*SOURCES,'dreyer2023'):
            d=target if name=='dreyer2023' else current[name]
            arrays[name],dg=op.align(d['X'],d['subjects'],policy,cell['pipeline'],1.)
            diags.extend(dict(condition=cell['id'],dataset=name,**r) for r in dg)
            if cell['version']=='original' and cell['policy']=='relative':
                previous=target if name=='dreyer2023' else data[name]
                a,_=op.align(previous['X'],previous['subjects'],policy,cell['pipeline'],1.)
                e=dev.norm_error(arrays[name],a)
                checks.append(dict(pipeline=cell['pipeline'],dataset=name,relative_error=e,
                    byte_equal=bool(np.array_equal(arrays[name],a)),pass_tolerance=bool(e<=cfg['relative_alignment_tolerance'])))
                del a
        xs=np.concatenate([arrays[n] for n in SOURCES])
        ys=np.concatenate([current[n]['y'] for n in SOURCES])
        mu,sd=ns['norm_stats'](xs)
        xs=((xs-mu)/sd).astype(np.float32)
        xt=((arrays['dreyer2023']-mu)/sd).astype(np.float32)
        csp,scaler,lda=ns['fit_csp_lda'](ns['ManualCSP'],xs,ys,8)
        features=scaler.transform(csp.transform(xt))
        pred=lda.predict(features).astype(np.int8)
        decision=lda.decision_function(features)
        assert np.isfinite(decision).all()
        save=dict(pred=pred,decision=decision,subjects=target['subjects'],trial_index=np.arange(len(pred)))
        if cell['version']=='adopted':
            with np.load(old_prediction(cell['policy'],cell['pipeline']),allow_pickle=False) as old:
                for key in ['pred','subjects','trial_index']:assert np.array_equal(save[key],old[key]),('Anchor mismatch',cell,key)
                anchors.append(dict(condition=cell['id'],predictions_equal=True,
                    max_decision_difference=float(np.max(np.abs(decision-old['decision'])))))
        elif cell['policy']=='relative':
            with np.load(out/('predictions_adopted_relative_'+cell['pipeline']+'.npz'),allow_pickle=False) as old:
                checks.append(dict(pipeline=cell['pipeline'],dataset='prediction',
                    prediction_flips=int((pred!=old['pred']).sum()),max_decision_difference=float(np.max(np.abs(decision-old['decision'])))))
        p=out/('predictions_'+cell['id']+'.npz')
        np.savez_compressed(p,**save)
        paths.append(p)
        dump(out/'anchor_checks.json',anchors)
        dump(out/'relative_checks.json',checks)
        pd.DataFrame(diags).to_csv(out/'diagnostics.csv',index=False)
        dump(out/'status.json',dict(state='predicting',completed_fits=i,total_fits=8,target_labels_read=False,
                                  elapsed_seconds=time.perf_counter()-started))
        print('[fit]',i,'/8',cell['id'],flush=True)
        del arrays,xs,ys,xt,features,csp,scaler,lda
        gc.collect()
    dump(out/'predictions_frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),target_labels_read=False,
                                          hashes={p.name:sha(p) for p in paths}))
    with np.load(lock['inputs']['dreyer2023']['path'],allow_pickle=True) as f:y=f['y'].astype(int)
    assert len(y)==len(target['subjects'])
    np.savez_compressed(out/'scoring_labels.npz',y=y,subjects=target['subjects'])
    for n,r in lock['inputs'].items():assert sha(r['path'])==r['sha256'],('Input changed after run',n)
    dump(out/'status.json',dict(state='predictions_complete',completed_fits=8,total_fits=8,
        target_labels_read_after_prediction_freeze=True,target_subjects=87,source_subjects=224,
        elapsed_seconds=time.perf_counter()-started,relative_alignment_pass=all(r.get('pass_tolerance',True) for r in checks),
        relative_prediction_flips=sum(r.get('prediction_flips',0) for r in checks)))
    print('All8predictions frozen before target labels. Scoring next.',flush=True)
    subprocess.run([sys.executable,str(HOME/'analyze.py')],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    with threadpool_limits(limits=4):prepare() if args.action=='prepare' else run()
