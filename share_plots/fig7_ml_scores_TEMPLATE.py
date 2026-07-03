#!/usr/bin/env python3
"""Figure 7 template — ML output score distributions.

Layout: 1×2 panels.
  Left  — ML1 (signal vs background) on the held-out test fold, log y
  Right — ML2 (κ_3 shape discriminant), three signal κ slices, linear y
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps; ps.set_style()


# ─────────────────────────────────────────────────────────────────────
# FILL IN YOUR DATA HERE
#   ML1 panel:
#     d1            — ML1 score per event           (float array, shape (N1,))
#     y1            — signal/bg label per event     (int array; 1 = signal, 0 = bg)
#     w1            — per-event weight              (float array)
#
#   ML2 panel:
#     d2            — ML2 score per event           (float array, shape (N2,))
#     k3            — κ_3 value per event           (float array)
#     w2            — per-event weight              (float array)
# ─────────────────────────────────────────────────────────────────────
d1 = np.zeros(0); y1 = np.zeros(0, dtype=int); w1 = np.zeros(0)
d2 = np.zeros(0); k3 = np.zeros(0);            w2 = np.zeros(0)


def class_weights(score, w, m):
    n = int(m.sum())
    if n == 0: return np.zeros(0)
    return w[m] / max(w[m].sum(), 1e-30)


fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0))
b = np.linspace(0, 1, 41)

# ── ML1 ── (background filled, signal step)
m_sig = (y1 == 1); m_bg = (y1 == 0)
ax[0].hist(d1[m_bg],  bins=b, weights=class_weights(d1, w1, m_bg),
           histtype='stepfilled', color='steelblue', alpha=0.45,
           ec='steelblue', lw=1.4, label='background')
ax[0].hist(d1[m_sig], bins=b, weights=class_weights(d1, w1, m_sig),
           histtype='step', color='k', lw=1.6,
           label=r'signal ($\kappa_3{=}1$)')
ax[0].set_xlabel('ML1 score (signal vs background)')
ax[0].set_ylabel('Normalized Events')
ax[0].set_xlim(0, 1); ax[0].set_yscale('log'); ax[0].set_ylim(1e-4, 5.0)
ax[0].legend(loc='upper right', fontsize=8, framealpha=0.95)
ps.hep_tag(ax[0], [r'$\sqrt{s} = 3$ TeV'], loc='upper left', simulation=True)
ax[0].texts[-1].set_fontsize(8)

# ── ML2 ── (three κ_3 slices)
def kmatch(arr, target, tol=0.005):
    return np.abs(arr - target) < tol

for kv, c in [(0.4, '#d62728'), (1.0, 'k'), (1.6, '#ff7f0e')]:
    m = kmatch(k3, kv)
    if not m.any(): continue
    nrm = w2[m] / max(w2[m].sum(), 1e-30)
    ax[1].hist(d2[m], bins=b, weights=nrm,
               histtype='step', color=c, lw=1.6,
               label=rf'$\kappa_3={kv:.1f}$' + (' (SM)' if kv == 1.0 else ''))
ax[1].set_xlabel(r'ML2 score ($\kappa_3$ discriminant)')
ax[1].set_ylabel('Normalized Events')
ax[1].set_xlim(0, 1)
# headroom for the legend
ymax = 0.0
for kv in (0.4, 1.0, 1.6):
    m = kmatch(k3, kv)
    if not m.any(): continue
    nrm = w2[m] / max(w2[m].sum(), 1e-30)
    h, _ = np.histogram(d2[m], bins=b, weights=nrm)
    ymax = max(ymax, float(h.max()))
if ymax > 0:
    ax[1].set_ylim(0, ymax * 1.45)
ax[1].legend(loc='upper left', fontsize=8, framealpha=0.95)
ps.hep_tag(ax[1], [r'$\sqrt{s} = 3$ TeV'], loc='upper right', simulation=True)
ax[1].texts[-1].set_fontsize(8)

fig.tight_layout(w_pad=1.8)
OUT = 'fig7_ml_scores.png'
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'saved {OUT}')
