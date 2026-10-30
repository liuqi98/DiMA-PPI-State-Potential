"""Regenerate manuscript tables and figure from archived summary data."""
from pathlib import Path
import csv
import json
import statistics as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'evidence'
def rows(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))

accuracy = rows('state_potential_model_comparison.csv')
names = ['Pairwise', 'Reverse supervision', 'Soft consistency',
         'Hard antisymmetry', 'No state descriptors', 'DiMA-PPI']
out = [r'\begin{table}[t]', r'\centering',
       r'\caption{S1129 predictive accuracy: mean $\pm$ sample SD across five repeats or seeds, each averaged over five folds. RMSE is in kcal mol$^{-1}$.}',
       r'\label{tab:accuracy}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lccc}',
       r'\toprule Model & PCC $\uparrow$ & Spearman $\uparrow$ & RMSE $\downarrow$ \\', r'\midrule']
def accuracy_row(name, row):
    return name + ' & ' + ' & '.join(
        f'${float(row[m+"_mean"]):.3f}\\pm{float(row[m+"_sd"]):.3f}$'
        for m in ['pearson', 'spearman', 'rmse']) + r' \\'

comparators = {r['model']: r for r in rows('comparator_model_summary.csv')}
out.append(r'\multicolumn{4}{l}{\textit{Published-model comparison}} \\')
for key, name in [('DSSA_released', r'DSSA-PPI~\cite{dssappi}'),
                  ('DiMA', 'DiMA-PPI (ours)')]:
    out.append(accuracy_row(name, comparators[key]))
out += [r'\midrule', r'\multicolumn{4}{l}{\textit{Matched controls and ablation}} \\']
for name, row in zip(names[:-1], accuracy[:-1]):
    out.append(accuracy_row(name, row))
out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(HERE / 'table_accuracy.tex').write_text('\n'.join(out)+'\n')

audit = json.loads((DATA / 'real_mutation_path_summary.json').read_text())
variants = ['pairwise', 'soft_consistency', 'hard_antisymmetry', 'four_state_potential']
out = [r'\begin{table}[t]', r'\centering',
       r'\caption{PPI-Path92 mean absolute consistency residuals (kcal mol$^{-1}$), averaged across five seeds.}',
       r'\label{tab:consistency}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
       r'\begin{tabular}{lcccc}', r'\toprule Model & Reverse & Path & Triangle & Four-cycle \\', r'\midrule']
def fmt(x):
    if x >= 0.001:
        return f'${x:.3f}$'
    a, b = f'{x:.2e}'.split('e')
    return f'${a}\\!\\times\\!10^{{{int(b)}}}$'
for key, name in zip(variants, ['Pairwise', 'Soft', 'Hard anti.', 'DiMA-PPI']):
    selected = [r for r in audit['summaries'] if r['variant'] == key]
    assert len(selected) == 5
    values = [stats.mean(r[m]['mean'] for r in selected) for m in
              ['reverse_error', 'path_error', 'triangle_cycle_error', 'four_state_cycle_error']]
    out.append(name + ' & ' + ' & '.join(map(fmt, values)) + r' \\')
out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(HERE / 'table_consistency.tex').write_text('\n'.join(out)+'\n')

plt.rcParams.update({'font.family':'Liberation Sans', 'font.size':8,
                     'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False,
                     'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2), layout='constrained')
unseen = {(r['variant'],r['prediction_type']):r for r in rows('unseen_edge_summary.csv')}
x = np.arange(4)
for offset, kind, label, color, hatch in [(-.18,'native_unseen','Native edge','#187d89',None),
                                        (.18,'endpoint_composed','Endpoint composition','white','///')]:
    values = [float(unseen[(v,kind)]['rmse_mean']) for v in variants]
    sd = [float(unseen[(v,kind)]['rmse_sd']) for v in variants]
    axes[0].bar(x+offset, values, .34, yerr=sd, capsize=2, color=color,
                edgecolor='#187d89', hatch=hatch, label=label, error_kw={'linewidth':.8})
axes[0].set_xticks(x, ['Pairwise','Soft','Hard anti.','DiMA-PPI'])
axes[0].set_ylim(0, 2.75)
axes[0].set_ylabel(r'RMSE (kcal mol$^{-1}$)')
axes[0].set_title('(a) PPI-Path92 edge prediction', loc='left', fontsize=9)
axes[0].legend(frameon=False, fontsize=7, loc='upper right')

mc = [r for r in rows('mc_summary.csv') if r['arm']=='independent']
for metric, label, color, marker in [('path','Path','#187d89','o'),
                                    ('cycle4','Four-cycle','#b15b20','s')]:
    n = [int(r['N']) for r in mc]
    y = [float(r[metric+'_mean']) for r in mc]
    sd = [float(r[metric+'_sd']) for r in mc]
    axes[1].errorbar(n, y, yerr=sd, color=color, marker=marker, markersize=3,
                      linewidth=1.2, capsize=2, label=label)
axes[1].plot(n, float(mc[0]['path_mean'])/np.sqrt(n), '--', color='#64748b',
             linewidth=.9, label=r'$N^{-1/2}$ reference')
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_xticks(n, [str(i) for i in n])
axes[1].set_ylim(.035, 1.15)
axes[1].set_xlabel('Monte Carlo draws, N')
axes[1].set_ylabel(r'Residual (kcal mol$^{-1}$)')
axes[1].set_title('(b) PPI-Path92 state identity', loc='left', fontsize=9)
axes[1].legend(frameon=False, fontsize=7, loc='upper right')
axes[1].text(.03,.03,'Canonical reuse: zero at every N', transform=axes[1].transAxes,
              fontsize=7.5, color='#187d89')
for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#dce2e7', linewidth=.5)
fig.savefig(HERE / 'fig_results.pdf')
fig.savefig(HERE / 'fig_results.png', dpi=220)
plt.close(fig)
print('Generated two tables and a two-panel results figure from evidence files.')
