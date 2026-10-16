"""Replay the unchanged locked verifier with exact round-trip CSV float parsing.

The default pandas fast converter changed92/696 saved decimal score values by
at most1.4210854715202004e-14. Round-trip parsing reconstructs the binary NPZ
matrix exactly. No analytical code, results, assertions or tolerances change.
This reader-only wrapper was added after observing the verification failure.
"""
import json
from functools import partial
from types import SimpleNamespace
import numpy as np
import pandas as pd
from study import HOME,module,dump


def main():
    out=HOME/'results'
    with np.load(out/'bootstrap_draws.npz',allow_pickle=False) as b:
        ids=b['condition_ids'].tolist()
        report={}
        for label,mode in [('default',None),('round_trip','round_trip')]:
            scores=pd.read_csv(out/'subject_scores.csv',float_precision=mode)
            wide=scores.pivot(index='subject',columns='id',values='bac_pct').loc[b['subject_ids'],ids].to_numpy()
            report[label]=dict(mismatched_elements=int((wide!=b['scores_pct']).sum()),
                               maximum_difference=float(np.max(np.abs(wide-b['scores_pct']))))
        assert report['round_trip']['mismatched_elements']==0
    verifier=module(HOME/'verify.py','unchanged_locked_verifier')
    verifier.pd=SimpleNamespace(read_csv=partial(pd.read_csv,float_precision='round_trip'))
    verifier.main()
    qa=json.loads((out/'verification.json').read_text(encoding='utf-8'))
    qa.update(csv_float_parser='round_trip',default_csv_parser_roundoff=report['default'],
              round_trip_binary_score_reconstruction=report['round_trip'],
              frozen_verifier_assertions_and_tolerances_unchanged=True,
              reader_wrapper_added_after_initial_QA=True)
    dump(out/'verification.json',qa)
    dump(out/'verification_parser_check.json',report)


if __name__=='__main__':main()
