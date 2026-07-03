#!/usr/bin/env python3
"""Figure 4 — representative ML2 input features at √s = 3 TeV (single panel
2×2 grid).  Three κ₃ signal slices (0.4, 1.0, 1.6) drawn from the
independent kappa_set2 MC, with the inclusive 8-process background overlaid
in blue filled step as a reference.  Area-normalised histograms."""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '/root/work/Papers')
import paper_style as ps; ps.set_style()
from paper_style import hep_tag

DDIR = '/root/work/Papers/pipeline/data/3tev'
OUT  = '/root/work/Papers/figures/png/fig5_kin_kappa'

# Colours
COL_K04 = 'crimson'        # κ = 0.4
COL_K1  = 'k'              # κ = 1.0  (SM)
COL_K16 = 'darkorange'     # κ = 1.6
COL_BG  = 'steelblue'      # background reference

# (panel_label, xlabel, h5 accessor on kappa_set2, h5 accessor on sigbg_main,
#  x-range, n_bins)
PANELS = [
    ('(a)', r'$m_{HH}$  [GeV]',
        lambda f: f['hl/mHH'][:], lambda f: f['hl/mHH'][:],
        (100, 1500), 50),
    ('(b)', r'$|\Delta\phi(j_1,\,\mathrm{MET})|$',
        lambda f: f['hl/dphi_jet0_met'][:], lambda f: f['hl/dphi_jet0_met'][:],
        (0.0, np.pi), 50),
    ('(c)', r'$\mathrm{MET}$  [GeV]',
        lambda f: f['hl/met'][:], lambda f: f['hl/met'][:],
        (0, 350), 50),
    ('(d)', r'$\Delta R(H_1,\,H_2)$',
        lambda f: f['hl/dR_H1H2'][:], lambda f: f['hl/dR_H1H2'][:],
        (0.0, 6.5), 50),
]


KAPPA_TOL = 0.005

f_kset = h5py.File(os.path.join(DDIR, 'kappa_set2.h5'),  'r')
f_sb   = h5py.File(os.path.join(DDIR, 'sigbg_main.h5'), 'r')

k3 = f_kset['hl/kappa3_value'][:]
m_k04 = np.abs(k3 - 0.4) < KAPPA_TOL
m_k1  = np.abs(k3 - 1.0) < KAPPA_TOL
m_k16 = np.abs(k3 - 1.6) < KAPPA_TOL
bg_mask = ~f_sb['hl/target_sigbg'][:].astype(bool)
print(f'3 TeV kappa_set2:  N={len(k3):,}  '
      f'(κ=0.4 {int(m_k04.sum()):,} / κ=1.0 {int(m_k1.sum()):,} / κ=1.6 {int(m_k16.sum()):,})')
print(f'3 TeV sigbg_main:  bg={int(bg_mask.sum()):,}')

fig, axes = plt.subplots(2, 2, figsize=(ps.FULL, ps.FULL * 0.85))

for (lbl, xlbl, getter_k, getter_sb, (lo, hi), nb), ax in zip(PANELS, axes.flat):
    vals_k  = getter_k(f_kset)
    vals_sb = getter_sb(f_sb)
    v_k04 = vals_k[m_k04]
    v_k1  = vals_k[m_k1]
    v_k16 = vals_k[m_k16]
    v_bg  = vals_sb[bg_mask]
    edges = np.linspace(lo, hi, nb + 1)

    def _h(x):
        # Normalized Events: sum of bin heights = 1 within the x-range.
        # Events outside [lo, hi] are correctly excluded from the integral
        # (so each curve's bins sum to ≤ 1; equals 1 when the full population
        # fits inside the panel).
        h, _ = np.histogram(x, bins=edges)
        n = len(x)
        return h / float(n) if n else h
    h_bg  = _h(v_bg)
    h_k04 = _h(v_k04)
    h_k1  = _h(v_k1)
    h_k16 = _h(v_k16)

    # Background — filled (reference)
    ax.fill_between(edges, np.r_[h_bg, h_bg[-1]], step='post',
                    color=COL_BG, alpha=0.55, linewidth=0,
                    label='Background', zorder=1)
    # κ slices — coloured steps
    ax.step(edges, np.r_[h_k04, h_k04[-1]], where='post',
            color=COL_K04, lw=1.4, label=r'$\kappa_3 = 0.4$', zorder=2)
    ax.step(edges, np.r_[h_k1,  h_k1[-1]],  where='post',
            color=COL_K1,  lw=1.4, label=r'$\kappa_3 = 1.0$', zorder=3)
    ax.step(edges, np.r_[h_k16, h_k16[-1]], where='post',
            color=COL_K16, lw=1.4, label=r'$\kappa_3 = 1.6$', zorder=2)

    ax.set_xlabel(xlbl)
    ax.set_ylabel('Normalized Events')
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)

f_kset.close(); f_sb.close()

hep_tag(axes[0, 0], [r'$\sqrt{s} = 3$ TeV'], loc='upper right', simulation=True)
axes[0, 1].legend(loc='upper right')

fig.tight_layout()
fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(OUT + '.pdf',             bbox_inches='tight')   # PDF disabled — use png/
print(f'saved {OUT}.png  +  {OUT}.pdf')
