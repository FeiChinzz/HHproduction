#!/usr/bin/env python3
"""Figure 4 template — sig vs total-bg for 4 representative ML1 features.

Layout: 2×2 panels. Each panel:
  • background      : steel-blue stepfilled, alpha 0.55
  • signal κ₃=1     : black step

Both classes are independently normalised so Σ_bins = 1 per class within
the plotted x-range (events outside the range are correctly excluded).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps; ps.set_style()

COL_SIG = 'k'
COL_BG  = 'steelblue'


# ─────────────────────────────────────────────────────────────────────
# FILL IN YOUR DATA HERE
#   For each of the 4 features, provide arrays of "values" and a boolean
#   mask `target_signal` (True = signal κ=1, False = background).
#   Example loader skeleton:
#       with h5py.File('your_file.h5', 'r') as f:
#           target = f['hl/target_sigbg'][:].astype(bool)   # True = signal
#           f1 = f['hl/feature_a'][:]
#           f2 = f['hl/feature_b'][:]
#           f3 = f['hl/feature_c'][:]
#           f4 = f['hl/feature_d'][:]
# ─────────────────────────────────────────────────────────────────────
target_signal = np.zeros(0, dtype=bool)     # TODO: True/False per event

# (panel_xlabel, values_array, (lo, hi), n_bins)
PANELS = [
    (r'feature 1  $[\mathrm{unit}]$', np.zeros(0),  (0,   300),  50),
    (r'feature 2',                    np.zeros(0),  (3.5, 7.0),  50),
    (r'feature 3',                    np.zeros(0),  (5.0, 30.0), 50),
    (r'feature 4  $[\mathrm{unit}]$', np.zeros(0),  (20,  250),  50),
]


def class_weights(mask):
    n = int(mask.sum())
    return np.full(n, 1.0 / n) if n else np.zeros(0)


fig, axes = plt.subplots(2, 2, figsize=(ps.FULL, ps.FULL * 0.85))

for (xlbl, vals, (lo, hi), nb), ax in zip(PANELS, axes.flat):
    sig = vals[target_signal]; bg = vals[~target_signal]
    edges = np.linspace(lo, hi, nb + 1)
    h_bg,  _ = np.histogram(bg,  bins=edges, weights=class_weights(~target_signal))
    h_sig, _ = np.histogram(sig, bins=edges, weights=class_weights( target_signal))

    ax.fill_between(edges, np.r_[h_bg, h_bg[-1]], step='post',
                    color=COL_BG, alpha=0.55, linewidth=0,
                    label='Background', zorder=1)
    ax.step(edges, np.r_[h_sig, h_sig[-1]], where='post',
            color=COL_SIG, lw=1.4, label=r'Signal  $\kappa_3 = 1$', zorder=2)

    ax.set_xlabel(xlbl)
    ax.set_ylabel('Normalized Events')
    ax.set_xlim(lo, hi)
    top = max(h_bg.max(initial=0.0), h_sig.max(initial=0.0))
    ax.set_ylim(0, top * 1.30)

axes[0, 1].set_ylim(0, axes[0, 1].get_ylim()[1] * 1.30)   # legend headroom
ps.hep_tag(axes[0, 0], [r'$\sqrt{s} = 3$ TeV'], loc='upper right', simulation=True)
axes[0, 1].legend(loc='upper right')

fig.tight_layout()

# ── OUT — set your own output path ──
OUT = 'fig4_kin_sgbg.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'saved {OUT}')
