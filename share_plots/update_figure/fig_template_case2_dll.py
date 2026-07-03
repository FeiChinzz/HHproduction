#!/usr/bin/env python3
"""Template: 1x2 Case-2 -DeltalnL(kappa3) figure with poly4 fit + 68%/95% CL bands.
Same conventions as the resolved figures: scan points shifted to 0 at
kappa3=1, fourth-order polynomial fit (optionally restricted to a kappa3
window), thresholds 0.5/1.92 referenced to the FITTED MINIMUM, connected
regions drawn as shaded bands."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps; ps.set_style(); ps.compact_12()

# ══════════════════ EDIT HERE ═══════════════════════════════════════════
# Per panel: kappa3 scan points, -DeltalnL values (any reference; they are
# re-anchored at kappa3=1 below), fit window (None = use all points).
K_3TEV   = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8])
DLL_3TEV = np.array([1.979, 0.975, 0.375, 0.080, 0.018, 0.000, 0.020, 0.064,
                     0.181, 0.256, 0.275])   # REPLACE with your 3 TeV scan
WIN_3TEV = None

K_10TEV   = np.array([0.2, 0.4, 0.6, 0.8, 0.84, 0.88, 0.92, 0.96, 1.0,
                      1.04, 1.08, 1.12, 1.16, 1.2, 1.4, 1.6, 1.8])
DLL_10TEV = np.array([120.54, 63.90, 24.78, 5.35, 3.33, 1.90, 0.98, 0.34, 0.0,
                      0.19, 0.58, 1.24, 2.19, 3.01, 11.16, 17.96, 18.62])
                                         # REPLACE with your boosted 10 TeV scan
WIN_10TEV = (0.8, 1.2)                   # suggested: fit only the refined grid

TAG_EXTRA = 'boosted'                    # region label inside the panel
OUTPDF    = './template_case2_dll.pdf'
# ═════════════════════════════════════════════════════════════════════════


def poly4_cl(k, dll, window=None, n_fine=5000):
    """Shift at kappa3=1, poly4 fit (inside `window`), CL bands referenced
    to the fitted minimum. Returns dict for plotting."""
    k = np.asarray(k, float); dll = np.asarray(dll, float)
    dll = dll - dll[np.argmin(np.abs(k - 1.0))]          # anchor at kappa3=1
    m = np.ones(len(k), bool) if window is None else \
        (k >= window[0] - 1e-9) & (k <= window[1] + 1e-9)
    coef = np.polyfit(k[m], dll[m], 4)
    kf = np.linspace(k[m].min(), k[m].max(), n_fine)
    df = np.polyval(coef, kf)
    sh = np.maximum(df - df.min(), 0.0)                  # fitted-min reference

    def connected(thresh):
        mask = sh < thresh
        i = int(sh.argmin()); lo = hi = i
        while lo > 0 and mask[lo - 1]: lo -= 1
        while hi < len(mask) - 1 and mask[hi + 1]: hi += 1
        return float(kf[lo]), float(kf[hi])

    lo68, hi68 = connected(0.50)
    lo95, hi95 = connected(1.92)
    # display curve re-anchored at kappa3=1 so it passes through the points
    disp = np.maximum(df - np.polyval(coef, 1.0), 0.0)
    return dict(k=k, dll=dll, kf=kf, disp=disp,
                lo68=lo68, hi68=hi68, lo95=lo95, hi95=hi95)


PANELS = [
    ('3 TeV',  ps.C_3TEV,  K_3TEV,  DLL_3TEV,  WIN_3TEV),
    ('10 TeV', ps.C_10TEV, K_10TEV, DLL_10TEV, WIN_10TEV),
]

fig, ax = plt.subplots(1, 2, figsize=ps.SIZE12)
for a, (name, c, k, dll, win) in zip(ax, PANELS):
    P = poly4_cl(k, dll, win)
    a.axvspan(P['lo95'], P['hi95'], color=c, alpha=0.10, zorder=1)   # 95% band
    a.axvspan(P['lo68'], P['hi68'], color=c, alpha=0.22, zorder=2)   # 68% band
    a.plot(P['kf'], P['disp'], '-', color=c, lw=1.4, zorder=4)
    a.plot(P['k'], P['dll'], marker='o', ls='none', mfc='white', mec=c,
           mew=0.9, ms=3.2, zorder=5)
    a.axhline(0.50, color='0.45', ls=':', lw=1.0)
    a.axhline(1.92, color='0.65', ls=':', lw=1.0)
    a.axvline(1.0, color='0.7', lw=0.7, alpha=0.5)
    a.set_xlim(0.2, 1.8); a.set_ylim(-0.1, 3.0)
    a.set_xticks(np.arange(0.2, 1.81, 0.2))
    a.set_xlabel(r'$\kappa_3 = \lambda_3/\lambda_3^{\rm SM}$')
    a.set_ylabel(r'$-\Delta\ln L$')
    ps.hep_tag(a, [rf'$\sqrt{{s}}={name.split()[0]}$ TeV', TAG_EXTRA],
               loc='upper right', simulation=True, fontsize=7.5)
    print(f'{name}: 68% [{P["lo68"]:.3f}, {P["hi68"]:.3f}]   '
          f'95% [{P["lo95"]:.3f}, {P["hi95"]:.3f}]   window={win}')

fig.tight_layout(w_pad=2.0)
fig.savefig(OUTPDF, bbox_inches='tight')
print(f'saved {OUTPDF}')
