#!/usr/bin/env python3
"""Template: 1x2 Case-1 (observability) -DeltalnL(kappa3), boosted channel.
Null = SM background only (no HH); test = background + HH(kappa3).
Full kappa3 range, scan points + poly4 display curve."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps; ps.set_style(); ps.compact_12()

# ══════════════════ EDIT HERE ═══════════════════════════════════════════
# Case-1 scans (null = no HH).  DUMMY values = your Colab boosted scans.
K_3TEV   = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8])
DLL_3TEV = np.array([12.951, 9.501, 6.851, 4.864, 4.082, 3.427, 2.791, 2.348,
                     1.721, 1.460, 1.404])          # REPLACE with your scan

K_10TEV   = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
DLL_10TEV = np.array([661.156, 512.630, 385.699, 296.263, 220.061, 176.351,
                      130.960, 109.664, 104.608])   # REPLACE with your scan
YLIM_3TEV  = (0, 15)
YLIM_10TEV = (0, 700)
TAG_EXTRA  = 'boosted'
OUTPDF     = './template_case1_dll.pdf'
# ═════════════════════════════════════════════════════════════════════════

PANELS = [
    ('3 TeV',  ps.C_3TEV,  K_3TEV,  DLL_3TEV,  YLIM_3TEV),
    ('10 TeV', ps.C_10TEV, K_10TEV, DLL_10TEV, YLIM_10TEV),
]

kfine = np.linspace(0.2, 1.8, 401)
fig, ax = plt.subplots(1, 2, figsize=ps.SIZE12)
for a, (name, c, k, dll, ylim) in zip(ax, PANELS):
    curve = np.poly1d(np.polyfit(k, dll, 4))(kfine)   # display smoothing
    a.plot(kfine, curve, '-', color=c, lw=1.3, zorder=4)
    a.plot(k, dll, marker='o', ls='none', mfc='white', mec=c,
           mew=0.9, ms=3.2, zorder=5)
    a.axvline(1.0, color='0.7', lw=0.7, alpha=0.5)
    a.set_xlim(0.2, 1.8); a.set_ylim(*ylim)
    a.set_xticks(np.arange(0.2, 1.81, 0.2))
    a.set_xlabel(r'$\kappa_3 = \lambda_3/\lambda_3^{\rm SM}$')
    a.set_ylabel(r'$-\Delta\ln L$')
    ps.hep_tag(a, [rf'$\sqrt{{s}}={name.split()[0]}$ TeV', TAG_EXTRA],
               loc='upper right', simulation=True, fontsize=7.5)
    i1 = int(np.argmin(np.abs(k - 1.0)))
    print(f'{name}: kappa3=1  -DeltalnL={dll[i1]:.2f}  Z={np.sqrt(2*dll[i1]):.2f} sigma')

fig.tight_layout(w_pad=2.0)
fig.savefig(OUTPDF, bbox_inches='tight')
print(f'saved {OUTPDF}')
