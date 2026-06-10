#!/usr/bin/env python3
"""fig_shap_beeswarm — collaborator-style SHAP beeswarm.

Layout: 1×2 panel per network head, left = 3 TeV, right = 10 TeV.

Per row (one feature): each event contributes one signed-SHAP point on the
x-axis, jittered on the y-axis. Point colour encodes the feature value
(blue = low, red = high), normalised per feature to its own [5%, 95%]
quantile range so a few outliers do not wash out the colour scale.

Per-particle / per-jet streams are collapsed to one point per event:
  • jet_cont (n_ev, 4, 6)      → sum signed SHAP across the 4 jets    → 6 features
  • higgs_tok (n_ev, 2, 7)     → sum across 2 Higgs                   → 7 features
  • ll_cloud (n_ev, 40, 4)     → sum over 40 particles × 4 features   → 1 "ll_cloud_total" row
  • globals_non_tda / globals_tda → already 1 value per event         → 8 / 5 features

Features sorted top→bottom by mean|signed-SHAP| (most important on top),
using the 3 TeV ordering for both panels so feature rows line up across √s.

Inputs : feature_analysis/{3tev,10tev}_{head}_shap.npz  (raw sv_* + X_*).
Outputs: figures/fig_shap_{head}_beeswarm.{png,pdf}
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sys.path.insert(0, '/root/work/Papers')
import paper_style as ps; ps.set_style()
from paper_style import hep_tag

FA  = '/root/work/Papers/feature_analysis'
OUT = '/root/work/Papers/figures'

STREAM_COLOR = {'jet': '#1f77b4', 'H': '#d62728', 'tda': '#9467bd',
                'global': '#2ca02c', 'll': '#7f7f7f'}
TDA_NAMES    = {'H_0', 'H_1', 'S_0', 'S_1', 'LB_1'}
HIGGS_NAMES  = {'H_log_pt', 'H_eta', 'H_sin_phi', 'H_cos_phi', 'H_log_m',
                'H_nbtag', 'H_dR_jj'}
JET_NAMES    = {'jet_log_pt', 'jet_eta', 'jet_sinphi', 'jet_cosphi',
                'jet_log_m', 'jet_dphi_met'}


def stream_of(name):
    if name in TDA_NAMES:   return 'tda'
    if name in HIGGS_NAMES: return 'H'
    if name in JET_NAMES or name.startswith('jet'): return 'jet'
    if name.startswith('ll') or name == 'll_cloud_total': return 'll'
    return 'global'


def load_streams(stage, head):
    """Return per-feature (signed_SHAP_per_event, feature_value_per_event)
    after collapsing per-particle/per-jet dims. dict[name] = (sv, X)."""
    d = np.load(f'{FA}/{stage}_{head}_shap.npz', allow_pickle=True)
    out = {}

    # jet_cont (n_ev, 4, 6) → sum jets, mean feature value
    sv = d['sv_jc']; X = d['X_jc']
    names = [str(n) for n in d['names_jc']]
    sv_e = sv.sum(axis=1)               # (n_ev, 6)
    X_e  = X.mean(axis=1)               # (n_ev, 6)
    for i, nm in enumerate(names):
        out[nm] = (sv_e[:, i], X_e[:, i])

    # higgs_tok (n_ev, 2, 7) → sum Higgs, mean value
    sv = d['sv_ht']; X = d['X_ht']
    names = [str(n) for n in d['names_ht']]
    sv_e = sv.sum(axis=1); X_e = X.mean(axis=1)
    for i, nm in enumerate(names):
        out[nm] = (sv_e[:, i], X_e[:, i])

    # globals_non_tda (n_ev, n)
    sv = d['sv_gnt']; X = d['X_gnt']
    names = [str(n) for n in d['names_gnt']]
    for i, nm in enumerate(names):
        out[nm] = (sv[:, i], X[:, i])

    # globals_tda (n_ev, 5)
    sv = d['sv_gtda']; X = d['X_gtda']
    names = [str(n) for n in d['names_gtda']]
    for i, nm in enumerate(names):
        out[nm] = (sv[:, i], X[:, i])

    # ll_cloud → 1 row "ll_cloud_total"
    sv = d['sv_llc']; X = d['X_llc']
    sv_total = sv.sum(axis=(1, 2))      # signed sum over particles × features
    pT_idx = list(d['names_llc']).index('ll_pT_frac') \
             if 'll_pT_frac' in list(d['names_llc']) else 0
    X_total = np.abs(X[:, :, pT_idx]).sum(axis=1)   # total LL activity proxy
    out['ll_cloud_total'] = (sv_total, X_total)
    return out


def draw_beeswarm(ax, streams_3, streams_10_for_order, title, *,
                  rng_seed=0, jitter=0.32, point_size=4.5,
                  cmap_name='coolwarm'):
    """streams_3 = the dict to draw on this axis.
    streams_10_for_order: used only to compute the union ordering if needed.
    """
    feats = list(streams_3.keys())
    mean_abs = {nm: float(np.mean(np.abs(streams_3[nm][0]))) for nm in feats}
    # sort descending → most important at TOP (large y)
    feats_sorted = sorted(feats, key=lambda n: -mean_abs[n])
    n = len(feats_sorted)
    rng = np.random.default_rng(rng_seed)
    cmap = plt.get_cmap(cmap_name)

    for i, nm in enumerate(feats_sorted):
        sv, X = streams_3[nm]
        # robust normalisation: 5–95% quantile clipping → [0, 1] for the cmap
        q_lo, q_hi = np.quantile(X, [0.05, 0.95])
        if q_hi - q_lo < 1e-12:
            colour = np.full(len(X), 0.5)
        else:
            colour = np.clip((X - q_lo) / (q_hi - q_lo), 0, 1)
        y = (n - 1 - i) + rng.uniform(-jitter, jitter, size=len(sv))
        ax.scatter(sv, y, c=colour, cmap=cmap, vmin=0, vmax=1,
                   s=point_size, alpha=0.65, edgecolors='none', rasterized=True)
        # left-side colour-tag squares
        ax.add_patch(plt.Rectangle((-0.0, n - 1 - i - 0.35), 0, 0.7,
                                   color=STREAM_COLOR[stream_of(nm)]))

    ax.axvline(0, color='0.4', lw=0.7, ls='--')
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(feats_sorted[::-1], fontsize=7.5)
    ax.set_xlabel('SHAP value (signed, summed over particles/jets)')
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left', pad=4)
    ax.grid(axis='x', linestyle=':', alpha=0.45)
    ax.set_ylim(-0.6, n - 0.4)
    return feats_sorted


def make_figure(head, out_name):
    s3  = load_streams('3tev',  head)
    s10 = load_streams('10tev', head)

    # share row ordering (use 3T ranking)
    feats = list(s3.keys())
    mean_abs_3 = {nm: float(np.mean(np.abs(s3[nm][0]))) for nm in feats}
    feature_order = sorted(feats, key=lambda n: -mean_abs_3[n])  # most-imp first
    # make s10 use the same feature set (should be identical)

    n = len(feature_order)
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL * 1.2, 0.22 * n + 0.8),
                             gridspec_kw=dict(wspace=0.05))
    rng_seed = 0

    def _draw(ax, streams, title):
        cmap = plt.get_cmap('coolwarm')
        rng = np.random.default_rng(rng_seed)
        for i, nm in enumerate(feature_order):
            if nm not in streams:
                continue
            sv, X = streams[nm]
            q_lo, q_hi = np.quantile(X, [0.05, 0.95])
            if q_hi - q_lo < 1e-12:
                colour = np.full(len(X), 0.5)
            else:
                colour = np.clip((X - q_lo) / (q_hi - q_lo), 0, 1)
            y = (n - 1 - i) + rng.uniform(-0.32, 0.32, size=len(sv))
            ax.scatter(sv, y, c=colour, cmap=cmap, vmin=0, vmax=1,
                       s=4.5, alpha=0.65, edgecolors='none', rasterized=True)
        ax.axvline(0, color='0.4', lw=0.7, ls='--')
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(feature_order[::-1], fontsize=7.5)
        ax.set_xlabel('SHAP value (signed)')
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left', pad=4)
        ax.grid(axis='x', linestyle=':', alpha=0.45)
        ax.set_ylim(-0.6, n - 0.4)

    _draw(axes[0], s3,  fr'(a) $\sqrt{{s}}=3$ TeV')
    _draw(axes[1], s10, fr'(b) $\sqrt{{s}}=10$ TeV')
    axes[1].set_yticklabels([])

    # single colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.65])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1),
                                     cmap=plt.get_cmap('coolwarm')),
                      cax=cbar_ax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['low', '', 'high'], fontsize=7.5)
    cb.set_label('feature value', fontsize=8)

    hep_tag(axes[0], [], loc='lower right', simulation=True)
    fig.tight_layout(rect=[0, 0, 0.90, 1.0])
    fig.savefig(f'{OUT}/{out_name}.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/{out_name}.png', dpi=200, bbox_inches='tight')
    print(f'saved {OUT}/{out_name}.{{pdf,png}}  ({n} features per panel)')
    plt.close(fig)


make_figure('ml1', 'fig_shap_ml1_beeswarm')
make_figure('ml2', 'fig_shap_ml2_beeswarm')
