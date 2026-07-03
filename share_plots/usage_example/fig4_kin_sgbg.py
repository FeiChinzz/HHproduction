#!/usr/bin/env python3
"""Figure 4 — representative ML1 input features at √s = 3 TeV (2×2 grid).
Signal (κ₃=1, SM) vs background (8-process inclusive). Fraction-per-bin
normalisation per class (sum of bin heights = 1 within the x-range when
all events are inside; events outside x-range are correctly missing).

m_H1 is shown with **SPANet pairing** (recomputed from the saved
assignment), not the χ²/XHH-pairing stored under hl/H1_m.
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
OUT    = '/root/work/Papers/figures/png/fig4_kin_sgbg'

COL_SIG = 'k'
COL_BG  = 'steelblue'


def class_weights(mask):
    """Per-event weights = 1 / (events in class) so the histogram sum
    over bins (across the FULL data range) is exactly 1. Events outside
    the plotted x-range are correctly missing from the integral."""
    n = int(mask.sum())
    return np.full(n, 1.0 / n) if n else np.zeros(0)


# load + SPANet-paired m_H1
with h5py.File(H5, 'r') as f:
    target = f['hl/target_sigbg'][:].astype(bool)
    jets   = f['jets'][:]
    met    = f['hl/met'][:]
    met_phi= f['hl/met_phi'][:]
    s0     = f['hl/S_0'][:]
    h0     = f['hl/H_0'][:]
    j1_pt  = f['jets'][:, 0, 0]
print(f'3 TeV sigbg_main: N={len(target):,}  '
      f'signal={int(target.sum()):,}  background={int((~target).sum()):,}')

assign = np.load(ASSIGN).astype(np.int8)
assert len(assign) == len(target)
rec = recompute_hl_from_assignment(jets, assign, met, met_phi)
H1_m_span = np.asarray(rec['H1_m'], dtype=np.float32)
print(f'  SPANet-paired m_H1: median = {np.median(H1_m_span):.2f}  '
      f'sig median = {np.median(H1_m_span[target]):.2f}')

# (xlabel, values, (lo, hi), n_bins)
PANELS = [
    (r'$m_{H_1}$  [GeV]',                  H1_m_span, (0, 300),    50),
    (r'$S_0$',                              s0,        (3.5, 7.0),  50),
    (r'$H_0$',                              h0,        (5.0, 30.0), 50),
    (r'$p_{T,j_1}$  [GeV]',                 j1_pt,     (20.0, 250.0), 50),
]


fig, axes = plt.subplots(2, 2, figsize=(ps.FULL, ps.FULL * 0.85))

for (xlbl, vals, (lo, hi), nb), ax in zip(PANELS, axes.flat):
    sig = vals[target]; bg = vals[~target]
    edges = np.linspace(lo, hi, nb + 1)

    h_bg,  _ = np.histogram(bg,  bins=edges, weights=class_weights(~target))
    h_sig, _ = np.histogram(sig, bins=edges, weights=class_weights(target))

    ax.fill_between(edges, np.r_[h_bg, h_bg[-1]], step='post',
                    color=COL_BG, alpha=0.55, linewidth=0,
                    label='Background', zorder=1)
    ax.step(edges, np.r_[h_sig, h_sig[-1]], where='post',
            color=COL_SIG, lw=1.4, label=r'Signal  $\kappa_3 = 1$', zorder=2)

    ax.set_xlabel(xlbl)
    ax.set_ylabel('Normalized Events')
    ax.set_xlim(lo, hi)
    # set y-top with extra headroom so legend in upper-right S_0 panel doesn't overlap
    top = max(h_bg.max(), h_sig.max())
    ax.set_ylim(0, top * 1.30)

# extra headroom for the top-right S_0 panel (which has the legend)
axes[0, 1].set_ylim(0, axes[0, 1].get_ylim()[1] * 1.30)

hep_tag(axes[0, 0], [r'$\sqrt{s} = 3$ TeV'], loc='upper right', simulation=True)
axes[0, 1].legend(loc='upper right')

fig.tight_layout()
fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(OUT + '.pdf', bbox_inches='tight')   # PDF disabled — use png/
print(f'saved {OUT}.png')
