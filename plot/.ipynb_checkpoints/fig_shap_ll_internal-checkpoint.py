#!/usr/bin/env python3
"""fig_shap_ll_internal — beeswarm of the 4 LL particle-cloud features
(ll_pT_frac, ll_dEta, ll_dPhi, ll_type), one row per feature.

Each event contributes up to 40 particle-level points per LL feature.
Padded particles (pT_frac == 0) are filtered out so the distribution
reflects only real PFlow constituents.

Layout: 1×2 panels — left = 3 TeV, right = 10 TeV. Features sorted by
mean|SHAP| descending (using the 3 TeV ranking, shared with the right
panel so the rows line up). Point colour = particle-level feature value,
robust [5%, 95%] quantile-normalised.

Inputs : feature_analysis/{3tev,10tev}_{head}_shap.npz (raw sv_llc + X_llc).
Outputs: figures/fig_shap_{head}_ll_internal.{png,pdf}
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


def load_ll(stage, head):
    """Return dict[feat_name] = (sv_flat, X_flat) over non-padded particles."""
    d = np.load(f'{FA}/{stage}_{head}_shap.npz', allow_pickle=True)
    sv = d['sv_llc']                                # (n_ev, 40, 4)
    X  = d['X_llc']                                 # (n_ev, 40, 4)
    names = [str(n) for n in d['names_llc']]
    pT_idx = names.index('ll_pT_frac') if 'll_pT_frac' in names else 0
    keep = X[:, :, pT_idx] > 0.0                    # (n_ev, 40) — drop padded
    out = {}
    for i, nm in enumerate(names):
        out[nm] = (sv[:, :, i][keep], X[:, :, i][keep])
    return out


def make_figure(head, out_name):
    s3  = load_ll('3tev',  head)
    s10 = load_ll('10tev', head)

    # mean|SHAP| ranking on 3T → most important first, shared between panels
    feats = list(s3.keys())
    rank = {nm: float(np.mean(np.abs(s3[nm][0]))) for nm in feats}
    feature_order = sorted(feats, key=lambda n: -rank[n])
    n = len(feature_order)
    rng_seed = 0

    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL * 1.20, 0.6 * n + 1.6),
                             gridspec_kw=dict(wspace=0.05))

    def _draw(ax, streams, title):
        cmap = plt.get_cmap('coolwarm')
        rng = np.random.default_rng(rng_seed)
        for i, nm in enumerate(feature_order):
            sv, X = streams[nm]
            if len(sv) == 0:
                continue
            q_lo, q_hi = np.quantile(X, [0.05, 0.95])
            if q_hi - q_lo < 1e-12:
                colour = np.full(len(X), 0.5)
            else:
                colour = np.clip((X - q_lo) / (q_hi - q_lo), 0, 1)
            y = (n - 1 - i) + rng.uniform(-0.32, 0.32, size=len(sv))
            ax.scatter(sv, y, c=colour, cmap=cmap, vmin=0, vmax=1,
                       s=2.0, alpha=0.30, edgecolors='none', rasterized=True)
        ax.axvline(0, color='0.4', lw=0.7, ls='--')
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(feature_order[::-1], fontsize=9)
        ax.set_xlabel('SHAP value (per particle, signed)')
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left', pad=4)
        ax.grid(axis='x', linestyle=':', alpha=0.45)
        ax.set_ylim(-0.6, n - 0.4)

    _draw(axes[0], s3,  fr'(a) $\sqrt{{s}}=3$ TeV')
    _draw(axes[1], s10, fr'(b) $\sqrt{{s}}=10$ TeV')
    axes[1].set_yticklabels([])

    # right-side colour bar
    cbar_ax = fig.add_axes([0.92, 0.22, 0.012, 0.55])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1),
                                     cmap=plt.get_cmap('coolwarm')),
                      cax=cbar_ax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['low', '', 'high'], fontsize=8)
    cb.set_label('feature value (per particle)', fontsize=8)

    hep_tag(axes[0], [], loc='lower right', simulation=True)
    fig.tight_layout(rect=[0, 0, 0.90, 1.0])
    fig.savefig(f'{OUT}/{out_name}.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/{out_name}.png', dpi=200, bbox_inches='tight')
    print(f'saved {OUT}/{out_name}.{{pdf,png}}  ({n} LL features per panel)')
    plt.close(fig)


make_figure('ml1', 'fig_shap_ml1_ll_internal')
make_figure('ml2', 'fig_shap_ml2_ll_internal')
