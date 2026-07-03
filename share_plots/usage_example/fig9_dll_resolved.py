#!/usr/bin/env python3
"""Figure 9 — resolved κ₃ DLL: P1 single-seed raw + poly4 fit (headline).

Headline method ("Opt1" in the project notes):
  • Data points (open markers): single-seed raw scan DLL values, re-anchored
    to the scan's own κ=1 entry so that −ΔlnL(κ=1)=0 by construction
    (same single-seed sample as both n and μ at κ=1). P1 / Asimov.
  • Solid curve: 4th-order polynomial fit through these data points.
  • Shaded bands: 68% / 95% connected confidence intervals read from the
    poly4 curve (single-parameter convention, ΔlnL < 0.5 / 1.92).
No morphing, no bootstrap — the morphing-Asimov result is quoted only as
a cross-check in the body text (it agrees with the poly4 to ≲1σ_seed)."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
# physics_constants reads PIPELINE_STAGE at import; default '3tev' is safe
# because poly4_w68 doesn't touch any stage-dependent constants.
os.environ.setdefault('PIPELINE_STAGE', '3tev')
sys.path.insert(0, '/root/work/Papers')
sys.path.insert(0, '/root/work/Papers/pipeline/src')
sys.path.insert(0, '/root/work/Papers/pipeline/src/lib')
import paper_style as ps; ps.set_style()
# Single source of truth for w68 extraction (per lib/dll.py docstring +
# review_consistency_c §1.1). The 68% CL comes directly from poly4_w68;
# the 95% CL re-uses its poly_coef with the same connected-region
# algorithm, only the threshold changes (0.5 → 1.92).
from lib.dll import poly4_w68


def w_at_thresh(coef, k_lo_grid, k_hi_grid, thresh, n_fine=5000):
    """Connected region width around the polynomial minimum at a given
    threshold. Algorithm is identical to lib.dll.poly4_w68's internal
    connected-region loop, with the threshold parameterised so it can be
    re-used for the 95% CL on the same poly_coef without re-fitting."""
    kf = np.linspace(k_lo_grid, k_hi_grid, n_fine)
    df = np.polyval(coef, kf)
    df_shifted = np.maximum(df - df.min(), 0.0)
    mask = df_shifted < thresh
    if not mask.any():
        return float('nan'), float('nan'), float('nan')
    imin = int(df_shifted.argmin())
    lo, hi = imin, imin
    while lo > 0 and mask[lo - 1]: lo -= 1
    while hi < len(mask) - 1 and mask[hi + 1]: hi += 1
    return float(kf[hi] - kf[lo]), float(kf[lo]), float(kf[hi])


def compute_all(npz_path, fit_window=None):
    """Headline = P1 single-seed raw + poly4. The fit + 68% CL come from
    lib.dll.poly4_w68 (single source of truth); the 95% CL re-uses the
    same poly_coef.

    fit_window : (κ_lo, κ_hi) tuple — restrict the 4th-order fit to this
    κ range. The scatter is still drawn over the full available grid;
    only the polynomial is constrained to the well region. Useful when
    far-from-well points (κ=0.2, 1.8) bias κ̂ away from κ=1 — see the
    10 TeV case where the full-grid fit puts κ̂≈1.028 and creates a
    visible vertical offset between scatter and line. With a narrower
    window κ̂≈1, the scatter–line anchors realign automatically and
    the well shape is captured more faithfully (lower rmse)."""
    R = np.load(npz_path, allow_pickle=True)
    fit_grid_full = np.asarray(R['fit_grid'], dtype=np.float64)
    raw_vals_full = np.asarray(R['raw_vals'],  dtype=np.float64)

    # Re-anchor raw scatter at κ=1 (P1-form: P3 raw with κ=1 entry shifted
    # to 0 — paper convention "P1 single-seed raw").
    i_k1 = int(np.argmin(np.abs(fit_grid_full - 1.0)))
    raw_P1_full = raw_vals_full - raw_vals_full[i_k1]

    if fit_window is not None:
        lo, hi = fit_window
        mfit = (fit_grid_full >= lo - 1e-9) & (fit_grid_full <= hi + 1e-9)
        fit_grid = fit_grid_full[mfit]
        raw_P1   = raw_P1_full[mfit]
    else:
        fit_grid = fit_grid_full
        raw_P1   = raw_P1_full

    res  = poly4_w68(fit_grid, raw_P1)            # canonical 68% CL
    coef = np.asarray(res['poly_coef'])
    w95, lo95, hi95 = w_at_thresh(coef, fit_grid.min(), fit_grid.max(), 1.92)

    # Display curve: P1 convention — anchor the plotted line at κ=1 so it
    # shares the scatter's anchor (scatter is also P1: raw − raw[κ=1]).
    # Without this, line uses fit-minimum anchor (κ̂ ≠ 1 in general) and
    # the two curves drift vertically — visible in 10 TeV where κ̂≈1.028
    # shifts the line ≈+0.1 above the scatter at κ=1. The 68%/95% CL
    # widths still come from poly4_w68 (fit-min anchored) which is the
    # correct single-parameter ΔlnL convention; only the *display* line
    # is re-anchored to match the scatter's P1 definition.
    kfine = np.linspace(fit_grid.min(), fit_grid.max(), 4001)
    poly  = np.polyval(coef, kfine)
    sh    = np.maximum(poly - np.polyval(coef, 1.0), 0.0)

    return dict(
        kfine=kfine, sh_P1=sh,
        # full-grid scatter for plotting (window only affects the fit)
        fit_grid=fit_grid_full, raw_P1=raw_P1_full,
        w68=res['w68_connected'], lo68=res['k3_lo'], hi68=res['k3_hi'],
        w95=w95, lo95=lo95, hi95=hi95,
        kmin=res['k3_min'], r2=res['r2'], n_regions=res['n_regions'],
    )


STAGES = [
    # name, npz, color, xlim, ylim, xticks, fit_window
    ('3 TeV',  '/root/work/Papers/check/seed_ensemble/3tev_seed42_dll_morphing.npz',
       ps.C_3TEV,  (0.2, 1.8),  (-0.5, 3.0),
       np.arange(0.2, 1.81, 0.2), None),
    ('10 TeV', '/root/work/Papers/check/seed_ensemble/10tev_seed42_dll_morphing.npz',
       ps.C_10TEV, (0.2, 1.8),  (-0.5, 3.0),
       np.arange(0.2, 1.81, 0.2), (0.8, 1.2)),
]

# Wider canvas so "Muon Collider" tag does not overlap the curves
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.5))

for a, (name, npz_path, c, xlim, ylim, xticks, fwin) in zip(ax, STAGES):
    P = compute_all(npz_path, fit_window=fwin)
    # CL bands
    a.axvspan(P['lo95'], P['hi95'], color=c, alpha=0.10, zorder=1)
    a.axvspan(P['lo68'], P['hi68'], color=c, alpha=0.22, zorder=2)
    # Asimov main curve (P1) — passes through 0 at κ=1 by construction
    a.plot(P['kfine'], P['sh_P1'], '-', color=c, lw=2.0, zorder=4)
    # P1 single-seed raw scatter (no bootstrap, no error bars).
    # Re-anchored to κ=1 so DLL(κ=1)=0 by construction (same sample as μ at κ=1).
    mscat = (P['fit_grid'] >= xlim[0]) & (P['fit_grid'] <= xlim[1])
    a.plot(P['fit_grid'][mscat], P['raw_P1'][mscat],
           marker='o', ls='none', mfc='white', mec=c, mew=1.1, ms=5.0,
           zorder=5)
    # threshold reference lines
    a.axhline(0.50, color='0.45', ls=':', lw=1.0)
    a.axhline(1.92, color='0.65', ls=':', lw=1.0)
    a.axvline(1.0,  color='0.7',  lw=0.7, alpha=0.5)
    a.set_xlim(*xlim); a.set_ylim(*ylim)
    a.set_xticks(xticks)
    a.set_xlabel(r'$\kappa_3 = \lambda_3/\lambda_3^{\rm SM}$')
    a.set_ylabel(r'$-\Delta\ln L$')
    ps.hep_tag(a, [rf'$\sqrt{{s}}={name.split()[0]}$ TeV', 'resolved'],
               loc='upper right', simulation=True)

fig.tight_layout(w_pad=2.0)
fig.savefig('/root/work/Papers/figures/png/fig9_dll_resolved.png', dpi=200, bbox_inches='tight')
# fig.savefig('/root/work/Papers/figures/png/fig9_dll_resolved.pdf', bbox_inches='tight')   # PDF disabled — use png/
print('saved /root/work/Papers/figures/png/fig9_dll_resolved.{png,pdf}')

# Re-print the CL ranges for body-text reference (via lib.dll.poly4_w68)
print('\n=== resolved CL ranges (P1 single-seed raw + poly4, via lib.dll) ===')
for name, npz, _c, _xl, _yl, _xt, fwin in STAGES:
    P = compute_all(npz, fit_window=fwin)
    print(f'{name}: w68={P["w68"]:.4f} [{P["lo68"]:.3f}, {P["hi68"]:.3f}]   '
          f'w95={P["w95"]:.4f} [{P["lo95"]:.3f}, {P["hi95"]:.3f}]   '
          f'κ_min={P["kmin"]:.3f}  R²={P["r2"]:.4f}  '
          f'fit_window={fwin}')
