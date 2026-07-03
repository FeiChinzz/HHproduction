#!/usr/bin/env python3
"""SHAP beeswarm for the boosted D_HH, paper format.
Features on the x-axis sorted left-to-right by mean|SHAP| (most important
left), signed SHAP on the y-axis (symmetric range), colorbar on the right.
The whole LL stream is collapsed into one 'particle cloud' entry.
Input: the npz produced by compute_boosted_shap.py."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps; ps.set_style(); ps.compact_12()

# ══════════════════ EDIT HERE ═══════════════════════════════════════════
NPZ    = './boosted_shap.npz'                    # from compute_boosted_shap.py
OUTPDF = './fig_shap_boosted_3tev.pdf'
TITLE  = r'Boosted $\mathcal{D}_{\rm HH}$ at $\sqrt{s}=3$ TeV'
# ═════════════════════════════════════════════════════════════════════════

# npz feature name → physics symbol used in the paper
SYMBOL = {
    'm_J_leading':     r'$M(J_1)$',
    'm_J_subleading':  r'$M(J_2)$',
    'pt_J_leading':    r'$p_T(J_1)$',
    'pt_J_subleading': r'$p_T(J_2)$',
    'missET':          r'$E_T^{\rm miss}$',
    'X_HH':            r'$X_{HH}$',
    'M_JJ':            r'$M_{JJ}$',
    'pT_JJ':           r'$p_T^{JJ}$',
    'dEta_JJ':         r'$\Delta\eta_{JJ}$',
    'BTag_leading':    r'${\rm BTag}(J_1)$',
    'BTag_subleading': r'${\rm BTag}(J_2)$',
    'H_0': r'$H_0$', 'H_1': r'$H_1$', 'S_0': r'$S_0$', 'S_1': r'$S_1$',
    'LB_1': r'$LB_1$',
    'particle_cloud': 'particle cloud',
}

d = np.load(NPZ, allow_pickle=True)
streams = {}
for i, nm in enumerate([str(n) for n in d['names_HL']]):
    streams[nm] = (d['sv_HL'][:, i].astype(float), d['X_HL'][:, i].astype(float))
# particle cloud: signed SUM of all 2x20x4 attributions per event (SHAP is
# additive, so this is the net effect of the whole LL stream on the score;
# note that +/- contributions inside the stream cancel).  Colour: total
# constituent |E_T| (channel 0) as a single-unit activity proxy — the four
# LL channels have different units so no mixed average is taken.
streams['particle_cloud'] = (d['sv_LL'].astype(float).sum(axis=(1, 2, 3)),
                             np.abs(d['X_LL'][:, :, :, 0].astype(float)).sum(axis=(1, 2)))

order = sorted(streams, key=lambda nm: -float(np.mean(np.abs(streams[nm][0]))))
n = len(order)


def swarm_offsets(vals, row_height=0.40, nbins=100):
    vals = np.asarray(vals); N = len(vals)
    rng = np.max(vals) - np.min(vals)
    if N == 0 or rng < 1e-12:
        return np.zeros(N)
    quant = np.round(nbins * (vals - np.min(vals)) / rng).astype(int)
    inds = np.argsort(quant + np.random.RandomState(0).randn(N) * 1e-6)
    layer, last, ys = 0, -1, np.zeros(N)
    for k in inds:
        if quant[k] != last:
            layer = 0
        ys[k] = np.ceil(layer / 2) * ((-1) ** layer)
        layer += 1; last = quant[k]
    return ys * (0.9 * row_height / max(1.0, np.max(np.abs(ys))))


fig, ax = plt.subplots(1, 1, figsize=(6.3, 2.9))
cmap = plt.get_cmap('coolwarm')
for i, nm in enumerate(order):
    sv, X = streams[nm]
    q_lo, q_hi = np.quantile(X, [0.05, 0.95])
    colour = (np.full(len(X), 0.5) if q_hi - q_lo < 1e-12
              else np.clip((X - q_lo) / (q_hi - q_lo), 0, 1))
    ax.scatter(i + swarm_offsets(sv), sv, c=colour, cmap=cmap, vmin=0, vmax=1,
               s=6.0, alpha=0.75, edgecolors='none', rasterized=True)
ax.axhline(0, color='0.4', lw=0.7)
ax.set_xticks(np.arange(n))
ax.set_xticklabels([SYMBOL.get(nm, nm) for nm in order],
                   rotation=60, ha='right', rotation_mode='anchor', fontsize=7.5)
ax.set_ylabel('SHAP value')
ax.set_title(TITLE, fontsize=8.5, fontweight='bold', loc='left', pad=3)
ax.grid(axis='y', linestyle=':', alpha=0.35)
ax.set_xlim(-0.7, n - 0.3)
ymax = np.ceil(10 * max(np.max(np.abs(streams[nm][0])) for nm in order)) / 10
ax.set_ylim(-ymax, ymax)
ax.tick_params(axis='x', which='minor', bottom=False, top=False)

cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                  ax=ax, ticks=[0, 1], pad=0.012, fraction=0.028)
cb.ax.set_yticklabels(['low', 'high'], fontsize=7.5)
cb.set_label('feature value', fontsize=8.4)
cb.ax.tick_params(which='minor', right=False)

fig.tight_layout()
fig.savefig(OUTPDF, bbox_inches='tight')
print(f'saved {OUTPDF}   top-5: {order[:5]}')
