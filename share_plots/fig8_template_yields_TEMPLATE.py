#!/usr/bin/env python3
"""Figure 8 template — 2D (ML1, ML2) template expected yields.

Layout: 2×2 panels per energy.
  top-left  : background
  top-right : SM signal (κ_3 = 1)
  bot-left  : signal κ_3 = 0.4
  bot-right : signal κ_3 = 1.6

All four panels share a single horizontal log colourbar at the top.
Each bin shows its expected count numerically.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps; ps.set_style()


# ─────────────────────────────────────────────────────────────────────
# FILL IN YOUR DATA HERE
#   Each of B, S_SM, S_04, S_16 is a 1-D array of length 100 (= 10 × 10
#   flattened) holding the expected event count per template bin at the
#   requested integrated luminosity.
#
#   sqrt_s_label : the centre-of-mass energy label for the colourbar
#   lumi_label   : luminosity label for the colourbar
# ─────────────────────────────────────────────────────────────────────
B    = np.zeros(100)
S_SM = np.zeros(100)
S_04 = np.zeros(100)
S_16 = np.zeros(100)

sqrt_s_label = '3'        # '3' or '10'
lumi_label   = '1'        # ab^-1


def fmt(v):
    if v >= 100:  return f'{v:.0f}'
    if v >= 10:   return f'{v:.1f}'
    if v >= 1:    return f'{v:.2f}'
    if v >= 0.01: return f'{v:.3f}'
    if v >  0:    return f'{v:.0e}'.replace('e-0', 'e-')
    return ''


all_pos = np.concatenate([B, S_SM, S_04, S_16])
all_pos = all_pos[all_pos > 0]
vmin = max(1e-3, all_pos.min()) if all_pos.size else 1e-3
vmax = all_pos.max()             if all_pos.size else 1.0
norm = LogNorm(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis.copy(); cmap.set_bad('0.85')

log_vmin, log_vmax = np.log10(vmin), np.log10(vmax)
def text_color(v):
    if v <= 0: return 'k'
    n = (np.log10(v) - log_vmin) / (log_vmax - log_vmin)
    return 'k' if n > 0.55 else 'white'

fig = plt.figure(figsize=(9.0, 9.0))
gs  = fig.add_gridspec(3, 2, height_ratios=[0.05, 1, 1],
                       hspace=0.35, wspace=0.25,
                       top=0.92, bottom=0.07, left=0.10, right=0.96)
cax  = fig.add_subplot(gs[0, :])
ax00 = fig.add_subplot(gs[1, 0]); ax01 = fig.add_subplot(gs[1, 1])
ax10 = fig.add_subplot(gs[2, 0]); ax11 = fig.add_subplot(gs[2, 1])

panels = [
    ('Background',                   B,    ax00),
    (r'Signal, $\kappa_3 = 1$ (SM)', S_SM, ax01),
    (r'Signal, $\kappa_3 = 0.4$',    S_04, ax10),
    (r'Signal, $\kappa_3 = 1.6$',    S_16, ax11),
]
im = None
for title, vals, ax in panels:
    H  = vals.reshape(10, 10)
    Hm = np.ma.masked_less_equal(H, 0)
    im = ax.imshow(Hm.T, origin='lower', cmap=cmap, norm=norm,
                   extent=[0, 10, 0, 10], aspect='equal')
    for i in range(10):
        for j in range(10):
            v = H[i, j]
            if v <= 0: continue
            ax.text(i + 0.5, j + 0.5, fmt(v),
                    ha='center', va='center', fontsize=5.8,
                    color=text_color(v), zorder=4)
    ax.set_xlabel('ML1 bin (signal vs bg)')
    ax.set_ylabel(r'ML2 bin ($\kappa_3$, quantile)')
    ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(0, 11, 2))
    ax.set_title(title, fontsize=10, pad=4)

cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
cb.set_label(
    rf'expected events per bin   ($\sqrt{{s}}={sqrt_s_label}$ TeV, '
    rf'$\mathcal{{L}}={lumi_label}\,\rm ab^{{-1}}$)',
    fontsize=10)

OUT = f'fig8_template_yields_{sqrt_s_label}tev.png'
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'saved {OUT}')
