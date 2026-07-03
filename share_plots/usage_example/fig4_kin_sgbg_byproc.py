#!/usr/bin/env python3
"""Companion to fig4_kin_sgbg.py — process-by-process breakdown of the
same 4 representative ML1 input features at √s = 3 TeV.

The original `fig4_kin_sgbg.py` collapses all 7 background processes
into one "Background" histogram (sig vs total-bg) — kept as is.  This
figure instead draws one step histogram per process (boosted-style),
each fraction-per-bin-normalised inside its own class so the *shapes*
are directly comparable.

Process map (from `target_everytype`, ordered by physics_constants.py):
  0 : signal (κ₃ = 1, SM)
  1 : hqqvv         (single-Higgs background with H→qq×V→ν̄ν)
  2 : wwvv
  3 : zzvv
  4 : ttvv
  5 : ww
  6 : zz
  7 : tt

m_H1 uses the SPANet pairing (recomputed from saved assignment), matching
the original figure's choice.
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '/root/work/Papers')
sys.path.insert(0, '/root/work/Papers/pipeline/src')
sys.path.insert(0, '/root/work/Papers/pipeline/src/lib')
os.environ.setdefault('PIPELINE_STAGE', '3tev')
import paper_style as ps; ps.set_style()
from paper_style import hep_tag
from lib.spanet_engine import recompute_hl_from_assignment

H5     = '/root/work/Papers/pipeline/data/3tev/sigbg_main.h5'
ASSIGN = '/root/work/Papers/pipeline/models/3tev/assign_sigbg_main.npy'
OUT    = '/root/work/Papers/figures/png/fig4_kin_sgbg_byproc'

# (target_everytype, legend label, colour)
PROCS = [
    (0, r'Signal  $\kappa_3 = 1$', 'k'),
    (1, 'hqqvv',                   '#1f77b4'),
    (2, 'wwvv',                    '#ff7f0e'),
    (3, 'zzvv',                    '#2ca02c'),
    (4, 'ttvv',                    '#d62728'),
    (5, 'ww',                      '#9467bd'),
    (6, 'zz',                      '#8c564b'),
    (7, 'tt',                      '#e377c2'),
]


with h5py.File(H5, 'r') as f:
    everytype = f['hl/target_everytype'][:].astype(np.int8)
    jets      = f['jets'][:]
    met       = f['hl/met'][:]
    met_phi   = f['hl/met_phi'][:]
    s0        = f['hl/S_0'][:]
    h0        = f['hl/H_0'][:]
    j1_pt     = f['jets'][:, 0, 0]
print(f'3 TeV sigbg_main: N={len(everytype):,}')
for tid, lab, _ in PROCS:
    print(f'  {tid} {lab:<20s}  n={int((everytype==tid).sum()):,}')

assign = np.load(ASSIGN).astype(np.int8)
assert len(assign) == len(everytype)
rec = recompute_hl_from_assignment(jets, assign, met, met_phi)
H1_m_span = np.asarray(rec['H1_m'], dtype=np.float32)


PANELS = [
    (r'$m_{H_1}$  [GeV]',                  H1_m_span, (0, 300),    50),
    (r'$S_0$',                              s0,        (3.5, 7.0),  50),
    (r'$H_0$',                              h0,        (5.0, 30.0), 50),
    (r'$p_{T,j_1}$  [GeV]',                 j1_pt,     (20.0, 250.0), 50),
]


def hist_frac(vals, edges):
    h, _ = np.histogram(vals, bins=edges)
    n = len(vals)
    return h / float(n) if n else h


fig, axes = plt.subplots(2, 2, figsize=(ps.FULL * 1.10, ps.FULL * 0.95))

for (xlbl, vals, (lo, hi), nb), ax in zip(PANELS, axes.flat):
    edges = np.linspace(lo, hi, nb + 1)
    for tid, lab, col in PROCS:
        mask = (everytype == tid)
        if not mask.any(): continue
        h = hist_frac(vals[mask], edges)
        lw = 1.6 if tid == 0 else 1.2
        ax.step(edges, np.r_[h, h[-1]], where='post',
                color=col, lw=lw, label=lab)
    ax.set_xlabel(xlbl); ax.set_ylabel('Normalized Events')
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)

# headroom in the top-right panel (S_0) so the legend doesn't sit on the curves
axes[0, 1].set_ylim(0.0, 0.10)
# legend in top-right panel; smaller font + 2 cols to keep the figure compact
axes[0, 1].legend(loc='upper right', fontsize=7, ncol=2,
                  framealpha=0.92, handlelength=1.6, labelspacing=0.25)
hep_tag(axes[0, 0], [r'$\sqrt{s} = 3$ TeV'], loc='upper right', simulation=True)

fig.tight_layout()
fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight')
print(f'saved {OUT}.png')
