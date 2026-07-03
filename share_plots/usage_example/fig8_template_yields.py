#!/usr/bin/env python3
"""Figure 8 — expected event yield per bin on the 10×10 (ML1 uniform ×
ML2 quantile) template at √s = 3 / 10 TeV. Four panels per energy:
background (top-left), SM signal κ₃=1 (top-right), κ₃=0.4 (bottom-left),
κ₃=1.6 (bottom-right). Shared log colorbar above the grid (horizontal);
each bin shows its expected count numerically.

Output: one figure per energy.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, '/root/work/Papers')
sys.path.insert(0, '/root/work/Papers/pipeline/src')
sys.path.insert(0, '/root/work/Papers/pipeline/src/lib')
import paper_style as ps; ps.set_style()
from lib.morphing import evaluate as morph_eval

LUMI = {'3tev': 1.0, '10tev': 10.0}      # ab^-1
NPZ  = '/root/work/Papers/check/seed_ensemble/{stage}_seed42_dll_morphing.npz'


def fmt(v):
    if v >= 100:   return f'{v:.0f}'
    if v >= 10:    return f'{v:.1f}'
    if v >= 1:     return f'{v:.2f}'
    if v >= 0.01:  return f'{v:.3f}'
    if v >  0:     return f'{v:.0e}'.replace('e-0', 'e-')
    return ''


def make_figure(stage, lumi_label):
    os.environ['PIPELINE_STAGE'] = stage
    R = np.load(NPZ.format(stage=stage), allow_pickle=True)
    B    = np.asarray(R['B'], dtype=np.float64)
    fit  = dict(coef=np.asarray(R['morph_coef']))
    S_SM = np.clip(morph_eval(fit, 1.0), 0.0, None)
    S_04 = np.clip(morph_eval(fit, 0.4), 0.0, None)
    S_16 = np.clip(morph_eval(fit, 1.6), 0.0, None)

    all_pos = np.concatenate([B, S_SM, S_04, S_16])
    all_pos = all_pos[all_pos > 0]
    vmin = max(1e-3, all_pos.min())
    vmax = all_pos.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis.copy(); cmap.set_bad('0.85')

    log_vmin, log_vmax = np.log10(vmin), np.log10(vmax)
    def text_color(v):
        if v <= 0: return 'k'
        n = (np.log10(v) - log_vmin) / (log_vmax - log_vmin)
        return 'k' if n > 0.55 else 'white'

    # 2×2 grid with a top horizontal colorbar
    fig = plt.figure(figsize=(9.0, 9.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.05, 1, 1],
                          hspace=0.35, wspace=0.25,
                          top=0.92, bottom=0.07, left=0.10, right=0.96)
    cax = fig.add_subplot(gs[0, :])
    ax00 = fig.add_subplot(gs[1, 0])
    ax01 = fig.add_subplot(gs[1, 1])
    ax10 = fig.add_subplot(gs[2, 0])
    ax11 = fig.add_subplot(gs[2, 1])
    axes = [ax00, ax01, ax10, ax11]
    panels = [
        ('Background',                   B,    ax00),
        (r'Signal, $\kappa_3 = 1$ (SM)', S_SM, ax01),
        (r'Signal, $\kappa_3 = 0.4$',    S_04, ax10),
        (r'Signal, $\kappa_3 = 1.6$',    S_16, ax11),
    ]
    for title, vals, ax in panels:
        H  = vals.reshape(10, 10)
        Hm = np.ma.masked_less_equal(H, 0)
        im = ax.imshow(Hm.T, origin='lower', cmap=cmap, norm=norm,
                       extent=[0, 10, 0, 10], aspect='equal')
        for i in range(10):
            for j in range(10):
                v = H[i, j]
                if v <= 0:
                    continue
                ax.text(i + 0.5, j + 0.5, fmt(v),
                        ha='center', va='center', fontsize=5.8,
                        color=text_color(v), zorder=4)
        ax.set_xlabel('ML1 bin (signal vs bg)')
        ax.set_ylabel(r'ML2 bin ($\kappa_3$, quantile)')
        ax.set_xticks(range(0, 11, 2))
        ax.set_yticks(range(0, 11, 2))
        ax.set_title(title, fontsize=10, pad=4)

    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.set_label(
        rf'expected events per bin   ($\sqrt{{s}}={lumi_label.split()[0]}$ TeV, '
        rf'$\mathcal{{L}}={LUMI[stage]:.0f}\,\rm ab^{{-1}}$)',
        fontsize=10)

    out = f'/root/work/Papers/figures/png/fig8_template_yields_{stage}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'saved {out}')
    plt.close(fig)
    return dict(B=B, S_SM=S_SM, S_04=S_04, S_16=S_16, vmin=vmin, vmax=vmax)


for stage, label in [('3tev', '3 TeV'), ('10tev', '10 TeV')]:
    info = make_figure(stage, label)
    print(f'  [{stage}]', {k: float(v.sum()) if hasattr(v, 'sum') else v
                            for k, v in info.items()})
