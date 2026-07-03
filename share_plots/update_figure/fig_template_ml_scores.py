#!/usr/bin/env python3
"""Template: 1x2 classifier-score figure (paper format).
Left panel: D_HH score, signal vs background, log y-scale.
Right panel: D_kappa3 score for three kappa3 hypotheses, linear y-scale."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps; ps.set_style(); ps.compact_12()

# ══════════════════ EDIT HERE ═══════════════════════════════════════════
# DUMMY DATA — replace with your score arrays (and weights if needed).
rng = np.random.default_rng(2)
DHH_SIG = rng.beta(6.0, 1.3, 50000)      # D_HH score, signal (kappa3=1)
DHH_BG  = rng.beta(0.7, 4.0, 400000)     # D_HH score, background
DK3 = {0.4: rng.beta(3.4, 3.4, 50000),   # D_kappa3 score per kappa3
       1.0: rng.beta(4.0, 3.0, 50000),
       1.6: rng.beta(4.6, 2.7, 50000)}
TAG_EXTRA = 'boosted'
OUTPDF    = './template_ml_scores.pdf'
# ═════════════════════════════════════════════════════════════════════════

KCOL = {0.4: '#d62728', 1.0: 'k', 1.6: '#ff7f0e'}
BINS = np.linspace(0, 1, 41)


def normed(arr):
    h, _ = np.histogram(arr, bins=BINS)
    return h / max(h.sum(), 1)


fig, ax = plt.subplots(1, 2, figsize=ps.SIZE12)

# left: D_HH, log scale
h = normed(DHH_BG)
ax[0].fill_between(BINS, np.r_[h, h[-1]], step='post', color='steelblue',
                   alpha=0.55, linewidth=0, label='background')
h = normed(DHH_SIG)
ax[0].step(BINS, np.r_[h, h[-1]], where='post', color='k', lw=1.8,
           label=r'signal ($\kappa_3=1$)')
ax[0].set_xlim(0, 1); ax[0].set_yscale('log')
ax[0].set_ylim(1e-4, 30.0)
ax[0].set_xlabel(r'$\mathcal{D}_\mathrm{HH}$ score')
ax[0].set_ylabel('Normalized Events')
ax[0].legend(loc='upper left', framealpha=0.95)
ps.hep_tag(ax[0], [r'$\sqrt{s}=3$ TeV', TAG_EXTRA],
           loc='upper right', simulation=True, fontsize=7.5)

# right: D_kappa3, three hypotheses
ymax = 0.0
for kv, arr in DK3.items():
    h = normed(arr); ymax = max(ymax, h.max())
    lbl = rf'$\kappa_3 = {kv}$' + (' (SM)' if kv == 1.0 else '')
    ax[1].step(BINS, np.r_[h, h[-1]], where='post', color=KCOL[kv], lw=1.8,
               label=lbl)
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, ymax * 1.70)
ax[1].set_xlabel(r'$\mathcal{D}_{\kappa_3}$ score')
ax[1].set_ylabel('Normalized Events')
ax[1].legend(loc='upper left', framealpha=0.95)
ps.hep_tag(ax[1], [r'$\sqrt{s}=3$ TeV', TAG_EXTRA],
           loc='upper right', simulation=True, fontsize=7.5)

fig.tight_layout(w_pad=1.8)
fig.savefig(OUTPDF, bbox_inches='tight')
print(f'saved {OUTPDF}')
