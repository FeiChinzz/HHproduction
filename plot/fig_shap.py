#!/usr/bin/env python3
"""fig_shap — SHAP-based feature importance figures for ML1 and ML2.
1×2 panel per network (3 TeV / 10 TeV side-by-side), every feature as
a horizontal bar, mean-|SHAP| value annotated as text at the end of
each bar.

Reads `feature_analysis/{stage}_{head}_shap.npz` (produced by
`feature_analysis/run_shap_fi_prod.py`)."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '/root/work/Papers')
import paper_style as ps; ps.set_style()
from paper_style import hep_tag

FA = '/root/work/Papers/feature_analysis'
OUT = '/root/work/Papers/figures'

# Stream colour map (groups for visual aid; not a strict requirement)
STREAM_COLOR = {
    'jet'   : '#1f77b4',     # jet token continuous + btag
    'H'     : '#d62728',     # Higgs token (width-7)
    'tda'   : '#9467bd',     # topological data analysis
    'global': '#2ca02c',     # other globals
    'll'    : '#7f7f7f',     # LL particle cloud
}
TDA_NAMES = {'H_0', 'H_1', 'S_0', 'S_1', 'LB_1'}
HIGGS_NAMES = {'H_log_pt', 'H_eta', 'H_sin_phi', 'H_cos_phi', 'H_log_m',
               'H_nbtag', 'H_dR_jj'}
JET_NAMES = {'jet_log_pt', 'jet_eta', 'jet_sinphi', 'jet_cosphi',
             'jet_log_m', 'jet_dphi_met'}


def stream_of(name):
    if name in TDA_NAMES:                          return 'tda'
    if name in HIGGS_NAMES:                        return 'H'
    if name in JET_NAMES or name.startswith('jet'): return 'jet'
    if name.startswith('ll') or name == 'll_cloud_total': return 'll'
    return 'global'


def load_npz(stage, head):
    """Load mean|SHAP| ranking and collapse the four per-particle LL features
    (ll_pT_frac, ll_dEta, ll_dPhi, ll_type) into a single `ll_cloud_total`
    entry whose value is the event-mean of Σ_{particle, feature} |SHAP|.

    Rationale: every other feature contributes one attribution per event,
    while LL features contribute 40 × 4 attributions per event that get
    diluted by the per-particle mean. Collapsing puts LL on the same
    "one bar = one event-stream" footing as the rest of the panel.
    """
    d = np.load(f'{FA}/{stage}_{head}_shap.npz', allow_pickle=True)
    records = [(str(n), float(v)) for n, v in zip(d['names'], d['shap'])
               if not str(n).startswith('ll_')]
    if 'sv_llc' in d.files:
        sv_llc = np.asarray(d['sv_llc'])              # (n_ev, 40, 4)
        ll_total = float(np.mean(np.sum(np.abs(sv_llc), axis=(1, 2))))
        records.append(('ll_cloud_total', ll_total))
    else:
        # Fallback for legacy npz without sv_llc: rough approximation
        # = sum of 4 LL feature mean|SHAP| × 40 particles
        ll_legacy = [v for n, v in zip(d['names'], d['shap'])
                     if str(n).startswith('ll_')]
        records.append(('ll_cloud_total', float(40.0 * sum(ll_legacy))))
    records.sort(key=lambda r: -r[1])
    return records


def draw_panel(ax, records, title):
    """records = list of (name, shap_value), pre-sorted ascending (small at bottom)."""
    n = len(records)
    names  = [r[0] for r in records]
    values = [r[1] for r in records]
    colors = [STREAM_COLOR[stream_of(nm)] for nm in names]
    ypos = np.arange(n)
    ax.barh(ypos, values, color=colors, edgecolor='none', height=0.78)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=7.5)
    # annotate numeric value at the end of each bar
    vmax = max(values) if values else 1.0
    pad  = 0.012 * vmax
    for y, v in zip(ypos, values):
        # left-pad bars whose values are too small to be visible: put text inside the chart
        ax.text(v + pad, y, f'{v:.4f}',
                ha='left', va='center', fontsize=6.8, color='0.15')
    ax.set_xlabel(r'mean $|$SHAP$|$')
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left', pad=4)
    ax.set_xlim(0, vmax * 1.30)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis='x', linestyle=':', alpha=0.45)


def make_figure(head, out_name):
    """Build the 1×2 figure (3 TeV | 10 TeV) for one network head."""
    recs_3  = load_npz('3tev',  head)
    recs_10 = load_npz('10tev', head)
    # use the 3T ordering on BOTH panels so the same feature lines up vertically
    order = sorted(recs_3, key=lambda r: r[1])               # ascending → top is highest
    name_to_3  = dict(recs_3)
    name_to_10 = dict(recs_10)
    feature_order = [r[0] for r in order]
    recs_3_sorted  = [(nm, name_to_3 [nm]) for nm in feature_order]
    recs_10_sorted = [(nm, name_to_10[nm]) for nm in feature_order]

    # 28 features → fairly tall figure
    n = len(feature_order)
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 0.22 * n + 0.6))
    draw_panel(axes[0], recs_3_sorted,  fr'(a) $\sqrt{{s}}=3$ TeV')
    draw_panel(axes[1], recs_10_sorted, fr'(b) $\sqrt{{s}}=10$ TeV')
    # right panel: feature labels are the same → hide ticks, save space
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    # HEP tag on the leftmost panel
    hep_tag(axes[0], [], loc='lower right', simulation=True)
    # mini-legend for stream colors (top-right of right panel)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=STREAM_COLOR[k])
                      for k in ['jet', 'H', 'global', 'tda', 'll']]
    legend_labels = ['jet token', 'Higgs token', 'global HL',
                     'TDA', 'LL particle']
    axes[1].legend(legend_handles, legend_labels, loc='lower right',
                    fontsize=7, frameon=True, framealpha=0.92,
                    handlelength=1.0, labelspacing=0.25)
    fig.tight_layout(pad=0.4)
    fig.savefig(f'{OUT}/{out_name}.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/{out_name}.png', dpi=200, bbox_inches='tight')
    print(f'saved {OUT}/{out_name}.{{pdf,png}}  ({n} features per panel)')
    plt.close(fig)


make_figure('ml1', 'fig_shap_ml1')
make_figure('ml2', 'fig_shap_ml2')
