from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import HOME, NAMES, dump


def ci(draws):
    return np.quantile(draws, [.025, .975], axis=0, method='linear')


def main():
    lock = json.loads((HOME / 'LOCK.json').read_text(encoding='utf-8'))
    cfg = lock['config']
    out = HOME / 'results'
    status = json.loads((out / 'status.json').read_text(encoding='utf-8'))
    assert status['state'] == 'predictions_complete'
    conditions = lock['conditions']
    cond_ids = [c['id'] for c in conditions]
    with np.load(out / 'scoring_labels.npz', allow_pickle=False) as f:
        labels = {k: f[k] for k in f.files}
    keys = [(name, int(s)) for name in NAMES for s in np.unique(labels['subjects__' + name])]
    matrix = np.empty((len(keys), len(conditions)))
    rows = []
    for j, condition in enumerate(conditions):
        with np.load(out / ('predictions_' + condition['id'] + '.npz'), allow_pickle=False) as f:
            for i, (name, subject) in enumerate(keys):
                y = labels['y__' + name]
                ids = labels['subjects__' + name]
                assert np.array_equal(f['subjects__' + name], ids)
                m = ids == subject
                p = f['pred__' + name]
                assert set(np.unique(y[m])) == {0, 1}
                score = 100 * np.mean([np.mean(p[m & (y == k)] == k) for k in (0, 1)])
                matrix[i, j] = score
                rows.append(dict(condition=condition['id'], dataset=name, subject=subject, n_trials=int(m.sum()), bac_pct=score))
    pd.DataFrame(rows).to_csv(out / 'subject_scores.csv', index=False)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    draws_n = cfg['bootstrap_draws']
    strata, draw_strata = {}, {}
    for name in NAMES:
        index = np.array([i for i, key in enumerate(keys) if key[0] == name])
        stratum = matrix[index]
        strata[name] = stratum.mean(axis=0)
        sampled = rng.integers(0, len(index), size=(draws_n, len(index)))
        draw_strata[name] = stratum[sampled].mean(axis=1)
    means = dict(strata)
    draws = dict(draw_strata)
    means['equal_dataset'] = np.mean([strata[n] for n in NAMES], axis=0)
    draws['equal_dataset'] = np.mean([draw_strata[n] for n in NAMES], axis=0)
    means['pooled_subject'] = matrix.mean(axis=0)
    weights = np.array([sum(k[0] == n for k in keys) / len(keys) for n in NAMES])
    draws['pooled_subject'] = sum(weights[i] * draw_strata[n] for i, n in enumerate(NAMES))
    np.savez_compressed(out / 'bootstrap_draws.npz', condition_ids=np.array(cond_ids), **draws)
    scopes = (*NAMES, 'equal_dataset', 'pooled_subject')
    summary = []
    primary_draws = []
    primary_estimates = []
    labels_order = cfg['epsilon_labels']
    zero_s = cond_ids.index('fixed_0_subject')
    zero_ds = cond_ids.index('fixed_0_dataset_subject')
    for scope in scopes:
        for path in ['fixed_post', 'tied_post', 'relative']:
            for label in (['relative'] if path == 'relative' else labels_order):
                s_id = 'relative_subject' if path == 'relative' else 'fixed_' + label + '_subject'
                ds_id = 'relative_dataset_subject' if path == 'relative' else ('fixed_' if path == 'fixed_post' or label == '1e-8' else 'tied_') + label + '_dataset_subject'
                s, ds = cond_ids.index(s_id), cond_ids.index(ds_id)
                s_ci, ds_ci = ci(draws[scope][:, s]), ci(draws[scope][:, ds])
                delta = means[scope][ds] - means[scope][s]
                delta_draw = draws[scope][:, ds] - draws[scope][:, s]
                delta_ci = ci(delta_draw)
                excess = delta - (means[scope][zero_ds] - means[scope][zero_s])
                excess_draw = delta_draw - (draws[scope][:, zero_ds] - draws[scope][:, zero_s])
                excess_ci = ci(excess_draw)
                summary.append(dict(scope=scope, path=path, epsilon_label=label,
                    S_pct=means[scope][s], S_ci_low=s_ci[0], S_ci_high=s_ci[1],
                    DS_pct=means[scope][ds], DS_ci_low=ds_ci[0], DS_ci_high=ds_ci[1],
                    increment_pp=delta, increment_ci_low=delta_ci[0], increment_ci_high=delta_ci[1],
                    increment_minus_zero_pp=excess, excess_ci_low=excess_ci[0], excess_ci_high=excess_ci[1]))
                if scope == 'equal_dataset' and path == 'fixed_post':
                    primary_draws.append(delta_draw)
                    primary_estimates.append(delta)
    primary_draws = np.stack(primary_draws, axis=1)
    primary_estimates = np.array(primary_estimates)
    all_draws = np.concatenate([primary_draws, primary_draws - primary_draws[:, [0]]], axis=1)
    all_estimates = np.concatenate([primary_estimates, primary_estimates - primary_estimates[0]])
    sd = all_draws.std(axis=0, ddof=1)
    nonconstant = sd > 1e-12
    standardized = np.abs((all_draws[:, nonconstant] - all_estimates[nonconstant]) / sd[nonconstant])
    critical = float(np.quantile(standardized.max(axis=1), .95, method='linear'))
    lo, hi = all_estimates - critical * sd, all_estimates + critical * sd
    frame = pd.DataFrame(summary)
    mask = (frame.scope == 'equal_dataset') & (frame.path == 'fixed_post')
    frame.loc[mask, 'increment_simultaneous_low'] = lo[:len(labels_order)]
    frame.loc[mask, 'increment_simultaneous_high'] = hi[:len(labels_order)]
    frame.loc[mask, 'excess_simultaneous_low'] = lo[len(labels_order):]
    frame.loc[mask, 'excess_simultaneous_high'] = hi[len(labels_order):]
    frame.to_csv(out / 'summary.csv', index=False)
    diagnostic = pd.read_csv(out / 'diagnostics.csv')
    diag_summary = diagnostic.groupby(['condition', 'dataset', 'stage']).agg(
        references=('subject','size'), clipped_median=('clipped','median'), clipped_min=('clipped','min'), clipped_max=('clipped','max'),
        residual_median=('whitening_AI','median'), residual_max=('whitening_AI','max'),
        floor_over_mean_median=('floor_over_mean','median')).reset_index()
    diag_summary.to_csv(out / 'diagnostic_summary.csv', index=False)
    anchors = json.loads((out / 'anchor_checks.json').read_text(encoding='utf-8'))
    assert len(anchors) == 16 and all(v['predictions_identical'] for v in anchors)
    anchor = frame[(frame.scope == 'equal_dataset') & (frame.path == 'fixed_post') & (frame.epsilon_label == '1e-8')].iloc[0]
    rel = frame[(frame.scope == 'equal_dataset') & (frame.path == 'relative')].iloc[0]
    assert abs((anchor.increment_pp - rel.increment_pp) - 4.429623344645) < 1e-6
    draw_plot(frame, diag_summary, labels_order, out)
    status.update(state='complete', bootstrap_draws=draws_n, bootstrap_seed=cfg['bootstrap_seed'], subjects=224,
                  anchor_prediction_checks=16, simultaneous_family_columns=20, simultaneous_critical_value=critical,
                  result_unit='percentage points; BAC levels percent', new_epsilon_selected=False)
    dump(out / 'status.json', status)
    table = frame[mask]
    print(table[['epsilon_label','S_pct','DS_pct','increment_pp','increment_ci_low','increment_ci_high']].to_string(index=False), flush=True)
    lines = ['# Absolute-floor sensitivity: numerical results', '',
             'Exploratory follow-up; protocol/config/code frozen before this sweep. No optimal epsilon selected.',
             'Primary path: raw epsilon varies; post-D dimensionless floor fixed at1e-8. All source/target alignment operators use their assigned policy.',
             'Adopted Cho amplitude version only; no manuscript edits or neural training.', '',
             '| Raw epsilon | S BAC | DS BAC | DS-S pp | Pointwise95%CI |', '|---|---:|---:|---:|---|']
    for row in table.itertuples():
        lines.append(f'| {row.epsilon_label} | {row.S_pct:.3f} | {row.DS_pct:.3f} | {row.increment_pp:+.3f} | [{row.increment_ci_low:+.3f}, {row.increment_ci_high:+.3f}] |')
    lines += ['', 'Primary increment/excess curves also have approximate95% simultaneous bands in summary.csv; displayed intervals above are pointwise.',
              'Intervals condition on fixed fitted models, observed subjects/trials and shared references; no retraining/new-site uncertainty.',
              'Independent interpretation and plot inspection follow; historical provenance is documented separately.']
    (HOME / 'NUMERICAL_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def draw_plot(frame, diag, labels, out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig, axes = plt.subplots(2,2,figsize=(12.5,8),layout='constrained')
    x = np.arange(len(labels))
    ticks = ['0\n(reference)'] + [r'$10^{%d}$' % int(np.log10(float(v))) for v in labels[1:]]
    primary = frame[(frame.scope=='equal_dataset') & (frame.path=='fixed_post')].set_index('epsilon_label').loc[labels]
    tied = frame[(frame.scope=='equal_dataset') & (frame.path=='tied_post')].set_index('epsilon_label').loc[labels]
    rel = frame[(frame.scope=='equal_dataset') & (frame.path=='relative')].iloc[0]
    def series(ax, y, color, label, marker='o'):
        ax.plot(x[1:],np.asarray(y)[1:],marker=marker,color=color,label=label)
        ax.plot(x[0],np.asarray(y)[0],marker=marker,color=color,linestyle='none')
    for field,color,label in [('S_pct','#007f86','SubjectEA'),('DS_pct','#bd4b36','DatasetEA -> SubjectEA')]:
        series(axes[0,0],primary[field],color,label)
        prefix='S' if field=='S_pct' else 'DS'
        axes[0,0].errorbar(x,primary[field],yerr=[primary[field]-primary[prefix+'_ci_low'],primary[prefix+'_ci_high']-primary[field]],fmt='none',ecolor=color,capsize=2)
    axes[0,0].set(title='(a) Decoding performance',ylabel='Balanced accuracy (%)',ylim=(0,100))
    axes[0,0].legend(loc='lower left',frameon=False)
    series(axes[0,1],primary.increment_pp,'#2a4f7a','Fixed post-D floor')
    axes[0,1].fill_between(x[1:],primary.increment_simultaneous_low.to_numpy(float)[1:],primary.increment_simultaneous_high.to_numpy(float)[1:],color='#2a4f7a',alpha=.16,label='Simultaneous 95% band')
    zero=primary.iloc[0]
    axes[0,1].errorbar([0],[zero.increment_pp],yerr=[[zero.increment_pp-zero.increment_simultaneous_low],[zero.increment_simultaneous_high-zero.increment_pp]],fmt='none',ecolor='#2a4f7a',capsize=3)
    axes[0,1].axhline(rel.increment_pp,color='#666666',linestyle='--',label='Relative-policy increment')
    axes[0,1].axhline(0,color='black',linewidth=.7)
    axes[0,1].set(title='(b) Incremental gain: primary path',ylabel='DS - S (pp)')
    axes[0,1].legend(frameon=False,fontsize=8)
    series(axes[1,0],primary.increment_pp,'#007f86','Post-D floor fixed at 1e-8')
    series(axes[1,0],tied.increment_pp,'#bd4b36','Post-D floor follows numerical grid','s')
    axes[1,0].axhline(0,color='black',linewidth=.7)
    axes[1,0].set(title='(c) Secondary post-stage sensitivity',ylabel='DS - S (pp)')
    axes[1,0].legend(frameon=False,fontsize=8)
    colors=['#0072b2','#d55e00','#009e73','#cc79a7']
    markers=['o','s','^','D']
    for name,color,marker in zip(NAMES,colors,markers):
        vals=[]
        for label in labels:
            row=diag[(diag.condition=='fixed_'+label+'_subject') & (diag.dataset==name) & (diag.stage=='subject_raw')]
            assert len(row)==1
            vals.append(float(row.iloc[0].clipped_median))
        series(axes[1,1],vals,color,name,marker)
    axes[1,1].set(title='(d) Subject reference clipping',ylabel='Median clipped eigenvalues / 21',ylim=(-.5,21.8),yticks=[0,7,14,21])
    axes[1,1].legend(frameon=False,fontsize=8)
    for ax in axes.flat:
        ax.set_xticks(x,ticks)
        ax.set_xlabel(r'Raw-stage absolute floor ($V^2$; declared)')
        ax.grid(axis='y',alpha=.3,linestyle='--')
        ax.axvline(.5,color='#aaaaaa',linewidth=.7)
        ax.axvline(labels.index('1e-8'),color='#555555',linestyle=':',linewidth=.8)
        ax.tick_params(axis='x',labelsize=8)
    fig.suptitle('Absolute-floor sensitivity: four-dataset equal-weight CSP-LDA\nZero is a separate reference; dotted line marks the historical floor',fontsize=12)
    fig.savefig(out/'floor_sensitivity.png',dpi=200)
    fig.savefig(out/'floor_sensitivity.pdf')
    plt.close(fig)


if __name__=='__main__': main()
