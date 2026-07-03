#!/usr/bin/env python3
"""Template: 1x2 kinematic-distribution figure (paper format).
Left/right panels: filled background + step histograms for three kappa3
signal hypotheses, normalized to unit area."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps; ps.set_style(); ps.compact_12()

# ══════════════════ EDIT HERE ═══════════════════════════════════════════
# Per panel: variable arrays (event values) + optional weights.
# DUMMY DATA below — replace with your arrays (e.g. M_JJ, pT(J1), ...).
rng = np.random.default_rng(1)
PANELS = [
    dict(xlabel=r'$M_{JJ}$  [GeV]',
         bins=np.linspace(200, 1200, 41),
         bg =rng.gamma(4, 90,  200000) + 150,
         sig={0.4: rng.gamma(5, 60, 50000) + 220,
              1.0: rng.gamma(5, 70, 50000) + 220,
              1.6: rng.gamma(5, 82, 50000) + 220}),
    dict(xlabel=r'$p_T(J_1)$  [GeV]',
         bins=np.linspace(200, 800, 41),
         bg =rng.gamma(3, 60,  200000) + 200,
         sig={0.4: rng.gamma(4, 55, 50000) + 210,
              1.0: rng.gamma(4, 60, 50000) + 210,
              1.6: rng.gamma(4, 66, 50000) + 210}),
]
TAG_EXTRA = 'boosted'
OUTPDF    = './template_kin.pdf'
# ═════════════════════════════════════════════════════════════════════════

KCOL = {0.4: '#d62728', 1.0: 'k', 1.6: '#ff7f0e'}   # paper kappa3 colours

fig, axes = plt.subplots(1, 2, figsize=ps.SIZE12)
for ax, P in zip(axes, PANELS):
    b = P['bins']
    h, _ = np.histogram(P['bg'], bins=b); h = h / max(h.sum(), 1)
    ax.fill_between(b, np.r_[h, h[-1]], step='post', color='steelblue',
                    alpha=0.55, linewidth=0, label='Background')
    for kv, arr in P['sig'].items():
        h, _ = np.histogram(arr, bins=b); h = h / max(h.sum(), 1)
        ax.step(b, np.r_[h, h[-1]], where='post', color=KCOL[kv], lw=1.5,
                label=rf'$\kappa_3 = {kv}$')
    ax.set_xlim(b[0], b[-1]); ax.set_ylim(0, None)
    ax.set_xlabel(P['xlabel'])
    ax.set_ylabel('Normalized Events')
axes[1].legend(loc='upper right')
ps.hep_tag(axes[0], [r'$\sqrt{s} = 3$ TeV', TAG_EXTRA],
           loc='upper right', simulation=True, fontsize=7.5)

fig.tight_layout(w_pad=1.5)
fig.savefig(OUTPDF, bbox_inches='tight')
print(f'saved {OUTPDF}')
