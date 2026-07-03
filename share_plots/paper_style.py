"""Shared publication style for the Muon-Collider HH κ₃ paper figures (JHEP, single column).
sans-serif, vector PDF, in-plot HEP tag, caption carries the title. import * and call set_style()."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# JHEP single-column text width ≈ 6.3 in. Full = 6.3", half (subfigure) ≈ 3.15".
FULL = 6.3
HALF = 3.15

def set_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 10,
        'axes.labelsize': 12, 'axes.titlesize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9.5, 'legend.frameon': False,
        'axes.linewidth': 0.9,
        'lines.linewidth': 1.6, 'lines.markersize': 5,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.minor.size': 2.8, 'ytick.minor.size': 2.8,
        'figure.dpi': 130, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'legend.handlelength': 1.6, 'legend.labelspacing': 0.3, 'legend.borderpad': 0.3,
    })

# consistent colors
C_3TEV = '#1f77b4'; C_10TEV = '#d62728'
C_SIG = '#2166ac'; C_BG = '#b2182b'
KAPPA_CMAP = 'viridis'

def hep_tag(ax, lines, loc='upper left', simulation=False, pad=0.035):
    """In-plot HEP tag: bold 'Muon Collider' (+ italic 'Simulation') + extra lines (e.g. √s)."""
    x = pad if 'left' in loc else 1 - pad
    y = 1 - pad if 'upper' in loc else pad
    ha = 'left' if 'left' in loc else 'right'
    va = 'top' if 'upper' in loc else 'bottom'
    txt = r'$\bf{Muon\ Collider}$'
    if simulation:
        txt += '\n' + r'$\it{Simulation}$'
    for l in (lines if isinstance(lines, (list, tuple)) else [lines]):
        txt += '\n' + l
    ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=9.5, linespacing=1.4)

def panel_label(ax, label, loc=(0.5, -0.30)):
    ax.text(loc[0], loc[1], label, transform=ax.transAxes, ha='center', va='top', fontsize=11)
