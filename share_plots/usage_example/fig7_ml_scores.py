#!/usr/bin/env python3
"""Figure 7 — ML output scores at √s = 3 TeV.

Left  panel: ML1 (signal vs background) on the held-out test fold of
             sigbg_main. Physics-weighted (xsec×BR×LUMI/N_gen via
             lib.weights.sigbg_weights) and normalised so that within
             each *class* the histogram height = (weighted events in bin)
             / (weighted events of that class, full range).
Right panel: ML2 (κ-discriminant) on the full κ-scan, three κ values
             overlaid: κ=0.4, κ=1.0 (held out from training), κ=1.6.
             Normalised so each curve's heights = (events in bin) /
             (events of that κ, full range).

Both panels use "fraction-of-events per bin" (not density=True), so that
events outside the x-range are correctly *missing* from the integral
(sum of bin heights ≤ 1 if any events lie outside [xmin, xmax]).

Inputs:
  pipeline/models/3tev/ml1_scores.npz       (d_test, y_test, idx_test)
  pipeline/data/3tev/sigbg_main.h5          (sigbg targets + n_btag for weights)
  feature_analysis/3tev_ml2_scan_scores.npz (d_kp, k3 — full κ-scan inference)
"""
import os, sys
import numpy as np, h5py
import matplotlib.pyplot as plt
sys.path.insert(0, '/root/work/Papers')
sys.path.insert(0, '/root/work/Papers/pipeline/src')
sys.path.insert(0, '/root/work/Papers/pipeline/src/lib')
os.environ.setdefault('PIPELINE_STAGE', '3tev')
import paper_style as ps; ps.set_style()
from lib.weights import sigbg_weights
from lib.physics_constants import kappa_match

PIPE = '/root/work/Papers/pipeline'
FA   = '/root/work/Papers/feature_analysis'

# ── ML1 score on the test fold ──
m1 = np.load(f'{PIPE}/models/3tev/ml1_scores.npz')
d1 = m1['d_test'].astype(np.float32)
y1 = m1['y_test'].astype(np.int8)
idx_test = m1['idx_test']

with h5py.File(f'{PIPE}/data/3tev/sigbg_main.h5', 'r') as f:
    tgt_full = f['hl/target_sigbg'][:].astype(np.int8)
    evt_full = f['hl/target_everytype'][:].astype(np.int8)
    nbt_full = f['hl/n_btag_total'][:].astype(np.int8)
w_full = sigbg_weights(tgt_full, evt_full, nbt_full, apply_btag_cut=False)
w_te = w_full[idx_test]

# ── ML2 score across full κ scan ──
m2 = np.load(f'{FA}/3tev_ml2_scan_scores.npz')
d2 = m2['d_kp'].astype(np.float32)
k3 = m2['k3'].astype(np.float32)


def class_weights(d, w, mask):
    """Fraction-per-bin weights: every event of the class gets the same
    physics weight, divided by the total class weighted sum (across the
    full d range, including events outside the plot window). This ensures
    that the bin heights actually represent the FRACTION of class events
    in each score bin — events outside the x-range are missing from the
    integral, never artificially compressed inside."""
    total = float(w[mask].sum())
    if total <= 0:
        return np.zeros(int(mask.sum()))
    return w[mask] / total


def event_weights(mask):
    """Same idea but unweighted: every event = 1 / N_total."""
    n = int(mask.sum())
    if n == 0:
        return np.zeros(0)
    return np.full(n, 1.0 / n)


fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0))
b = np.linspace(0, 1, 41)

# ─── ML1 ───
m_sig = (y1 == 1); m_bg  = (y1 == 0)
# background drawn first (filled blue), signal on top (black step)
ax[0].hist(d1[m_bg],  bins=b, weights=class_weights(d1, w_te, m_bg),
           histtype='stepfilled', color='steelblue', alpha=0.45,
           ec='steelblue', lw=1.4, label='background')
ax[0].hist(d1[m_sig], bins=b, weights=class_weights(d1, w_te, m_sig),
           histtype='step', color='k', lw=1.6,
           label=r'signal ($\kappa_3{=}1$)')
ax[0].set_xlabel('ML1 score (signal vs background)')
ax[0].set_ylabel('Normalized Events')
ax[0].set_xlim(0, 1); ax[0].set_yscale('log'); ax[0].set_ylim(1e-4, 5.0)
ax[0].legend(loc='upper right', fontsize=8, framealpha=0.95)
ps.hep_tag(ax[0], [r'$\sqrt{s}=3$ TeV'], loc='upper left', simulation=True)
ax[0].texts[-1].set_fontsize(8)

# ─── ML2 ───
for kv, c in [(0.4, '#d62728'), (1.0, 'k'), (1.6, '#ff7f0e')]:
    m = kappa_match(k3, kv)
    if not m.any():
        continue
    ax[1].hist(d2[m], bins=b, weights=event_weights(m),
               histtype='step', color=c, lw=1.6,
               label=rf'$\kappa_3={kv:.1f}$' + (' (SM)' if kv == 1.0 else ''))
ax[1].set_xlabel(r'ML2 score ($\kappa_3$ discriminant)')
ax[1].set_ylabel('Normalized Events')
ax[1].set_xlim(0, 1)
# raise y so the legend in the upper-left doesn't overlap the histograms
ymax = 0.0
for kv in (0.4, 1.0, 1.6):
    m = kappa_match(k3, kv)
    if not m.any(): continue
    h, _ = np.histogram(d2[m], bins=b, weights=event_weights(m))
    ymax = max(ymax, float(h.max()))
ax[1].set_ylim(0, ymax * 1.45)
ax[1].legend(loc='upper left', fontsize=8, framealpha=0.95)
ps.hep_tag(ax[1], [r'$\sqrt{s}=3$ TeV'], loc='upper right', simulation=True)
ax[1].texts[-1].set_fontsize(8)

fig.tight_layout(w_pad=1.8)
fig.savefig('/root/work/Papers/figures/png/fig7_ml_scores.png', dpi=200)
# fig.savefig('/root/work/Papers/figures/png/fig7_ml_scores.pdf')   # PDF disabled — use png/
print('saved fig7_ml_scores.png')
