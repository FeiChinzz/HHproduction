#!/usr/bin/env python3
"""Figure 4 by-process template — per-process step histograms.

Layout: 2×2 panels. Signal (κ₃=1) overlaid as black thicker step;
each background process drawn as its own coloured step.

Per-class normalisation (Σ_bins = 1) so shapes are directly comparable.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps; ps.set_style()

# (process_id, legend_label, colour)  — adapt to your samples
PROCS = [
    (0, r'Signal  $\kappa_3 = 1$', 'k'),
    (1, 'process 1',               '#1f77b4'),
    (2, 'process 2',               '#ff7f0e'),
    (3, 'process 3',               '#2ca02c'),
    (4, 'process 4',               '#d62728'),
    (5, 'process 5',               '#9467bd'),
    (6, 'process 6',               '#8c564b'),
    (7, 'process 7',               '#e377c2'),
]


# ─────────────────────────────────────────────────────────────────────
# FILL IN YOUR DATA HERE
#   target_everytype : int array, per-event process id (matches PROCS)
#   PANELS[i][1]     : float array, the feature value per event
# ─────────────────────────────────────────────────────────────────────
target_everytype = np.zeros(0, dtype=int)

PANELS = [
    (r'feature 1  $[\mathrm{unit}]$', np.zeros(0),  (0,   300),  50),
    (r'feature 2',                    np.zeros(0),  (3.5, 7.0),  50),
    (r'feature 3',                    np.zeros(0),  (5.0, 30.0), 50),
    (r'feature 4  $[\mathrm{unit}]$', np.zeros(0),  (20,  250),  50),
]


def hist_frac(vals, edges):
    h, _ = np.histogram(vals, bins=edges)
    n = len(vals)
    return h / float(n) if n else h


fig, axes = plt.subplots(2, 2, figsize=(ps.FULL * 1.10, ps.FULL * 0.95))

for (xlbl, vals, (lo, hi), nb), ax in zip(PANELS, axes.flat):
    edges = np.linspace(lo, hi, nb + 1)
    for tid, lab, col in PROCS:
        mask = (target_everytype == tid)
        if not mask.any(): continue
        h = hist_frac(vals[mask], edges)
        lw = 1.6 if tid == 0 else 1.2
        ax.step(edges, np.r_[h, h[-1]], where='post',
                color=col, lw=lw, label=lab)
    ax.set_xlabel(xlbl); ax.set_ylabel('Normalized Events')
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)

axes[0, 1].set_ylim(0.0, 0.10)
axes[0, 1].legend(loc='upper right', fontsize=7, ncol=2,
                  framealpha=0.92, handlelength=1.6, labelspacing=0.25)
ps.hep_tag(axes[0, 0], [r'$\sqrt{s} = 3$ TeV'],
           loc='upper right', simulation=True)

fig.tight_layout()
OUT = 'fig4_kin_sgbg_byproc.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'saved {OUT}')
