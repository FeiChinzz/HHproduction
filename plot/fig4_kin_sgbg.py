#!/usr/bin/env python3
"""Figure 3 — representative ML1 input features at √s = 3 TeV (single panel
2×2 grid).  Signal (κ₃=1, SM) vs background (8-process inclusive bkg) on
area-normalised histograms.  Log scalings used at ML training (log(pT),
log(1+m)) are stripped here because the figure is for *visual* sanity-
checking of the discriminating power, not for showing the network's
ingestion scale."""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '/root/work/Papers')
import paper_style as ps; ps.set_style()
from paper_style import hep_tag

H5  = '/root/work/Papers/pipeline/data/3tev/sigbg_main.h5'
OUT = '/root/work/Papers/figures/fig4_kin_sgbg'

# Colours: signal = black step, background = blue filled step
COL_SIG = 'k'
COL_BG  = 'steelblue'

# (panel_label, xlabel, h5 accessor, x-range, n_bins)
PANELS = [
    ('(a)', r'$m_{H_1}$  [GeV]',
        lambda f: f['hl/H1_m'][:],
        (0, 300), 50),
    ('(b)', r'$S_0$',
        lambda f: f['hl/S_0'][:],
        (3.5, 7.0), 50),
    ('(c)', r'$H_0$',
        lambda f: f['hl/H_0'][:],
        (5.0, 30.0), 50),
    ('(d)', r'$p_{T,j_1}$  [GeV]',
        lambda f: f['jets'][:, 0, 0],          # leading-pT jet, raw
        (20.0, 250.0), 50),
]


with h5py.File(H5, 'r') as f:
    target = f['hl/target_sigbg'][:].astype(bool)
    print(f'3 TeV sigbg_main:  N={len(target):,}  '
          f'signal={int(target.sum()):,}  background={int((~target).sum()):,}')

    fig, axes = plt.subplots(2, 2, figsize=(ps.FULL, ps.FULL * 0.85))

    for (lbl, xlbl, getter, (lo, hi), nb), ax in zip(PANELS, axes.flat):
        vals = getter(f)
        sig = vals[target]
        bg  = vals[~target]
        edges = np.linspace(lo, hi, nb + 1)

        h_bg, _  = np.histogram(bg,  bins=edges, density=True)
        h_sig, _ = np.histogram(sig, bins=edges, density=True)

        # Background — filled step (reference); zorder lower so signal stays on top
        ax.fill_between(edges, np.r_[h_bg, h_bg[-1]], step='post',
                        color=COL_BG, alpha=0.55, linewidth=0,
                        label='Background', zorder=1)
        # Signal — black step
        ax.step(edges, np.r_[h_sig, h_sig[-1]], where='post',
                color=COL_SIG, lw=1.4, label=r'Signal  $\kappa_3 = 1$', zorder=2)

        ax.set_xlabel(xlbl)
        ax.set_ylabel('Normalised  [a.u.]')
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)

        # Panel label, bold, top-left inside the axes
        ax.text(0.04, 0.94, lbl, transform=ax.transAxes,
                ha='left', va='top', fontsize=11, fontweight='bold')

# HEP tag + legend in panel (a)
hep_tag(axes[0, 0], [r'$\sqrt{s} = 3$ TeV'], loc='upper right', simulation=True)
axes[0, 1].legend(loc='upper right')

fig.tight_layout()
fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT + '.pdf',             bbox_inches='tight')
print(f'saved {OUT}.png  +  {OUT}.pdf')
