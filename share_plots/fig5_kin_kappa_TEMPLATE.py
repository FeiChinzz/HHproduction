#!/usr/bin/env python3
"""Figure 5 template — κ_3-dependent kinematic shapes (signal only, 4 panels).

Layout: 2×2 panels. Each panel overlays three signal κ_3 slices
(κ_3 = 0.4 / 1.0 / 1.6) as step lines, with the total background as a
filled reference. All curves use Σ_bins = 1 normalisation.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps; ps.set_style()

COL_BG  = 'steelblue'
COL_K04 = '#d62728'   # red
COL_K1  = 'k'         # black (SM)
COL_K16 = '#ff7f0e'   # orange


# ─────────────────────────────────────────────────────────────────────
# FILL IN YOUR DATA HERE
#   For each of the 4 features, provide:
#     v_k04, v_k1, v_k16  — signal values at κ_3 = 0.4 / 1.0 / 1.6
#     v_bg                 — total background values
#   You can build these from a single (values, κ_3) array via masks if
#   you have a κ-scan sample.
# ─────────────────────────────────────────────────────────────────────
PANELS = [
    (r'$m_{HH}$  [GeV]',                 None, (100, 1500),  50),
    (r'$|\Delta\phi(j_1,\,\mathrm{MET})|$', None, (0.0, np.pi), 50),
    (r'$\mathrm{MET}$  [GeV]',           None, (0, 350),     50),
    (r'$\Delta R(H_1,\,H_2)$',           None, (0.0, 6.5),   50),
]

# Replace each `None` below with a tuple of four arrays
# (v_k04, v_k1, v_k16, v_bg) for that feature.
DATA = [None, None, None, None]


def hist_frac(vals, edges):
    h, _ = np.histogram(vals, bins=edges)
    n = len(vals)
    return h / float(n) if n else h


fig, axes = plt.subplots(2, 2, figsize=(ps.FULL, ps.FULL * 0.85))

for (xlbl, _, (lo, hi), nb), data, ax in zip(PANELS, DATA, axes.flat):
    edges = np.linspace(lo, hi, nb + 1)
    if data is None:                 # placeholder — skip until user fills in
        v_k04 = v_k1 = v_k16 = v_bg = np.zeros(0)
    else:
        v_k04, v_k1, v_k16, v_bg = data

    h_bg  = hist_frac(v_bg,  edges)
    h_k04 = hist_frac(v_k04, edges)
    h_k1  = hist_frac(v_k1,  edges)
    h_k16 = hist_frac(v_k16, edges)

    ax.fill_between(edges, np.r_[h_bg, h_bg[-1]], step='post',
                    color=COL_BG, alpha=0.55, linewidth=0,
                    label='Background', zorder=1)
    ax.step(edges, np.r_[h_k04, h_k04[-1]], where='post',
            color=COL_K04, lw=1.4, label=r'$\kappa_3 = 0.4$', zorder=2)
    ax.step(edges, np.r_[h_k1, h_k1[-1]],  where='post',
            color=COL_K1,  lw=1.4, label=r'$\kappa_3 = 1.0$', zorder=3)
    ax.step(edges, np.r_[h_k16, h_k16[-1]], where='post',
            color=COL_K16, lw=1.4, label=r'$\kappa_3 = 1.6$', zorder=2)

    ax.set_xlabel(xlbl); ax.set_ylabel('Normalized Events')
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)

ps.hep_tag(axes[0, 0], [r'$\sqrt{s} = 3$ TeV'], loc='upper right', simulation=True)
axes[0, 1].legend(loc='upper right')

fig.tight_layout()
OUT = 'fig5_kin_kappa.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'saved {OUT}')
