#!/usr/bin/env python3
"""Figure 10 — combined (resolved ⊕ boosted) κ₃ DLL at 3 & 10 TeV.

Headline = P1 single-seed raw + poly4, applied identically to every
channel. Each panel uses a single colour (blue at 3 TeV, red at 10 TeV);
the three channels are distinguished only by line style:
   boosted   — dotted (poly4 through collaborator's 11/17 boosted points)
   resolved  — dashed (poly4 through this work's single-seed raw scatter)
   combined  — solid  (poly4 through the per-κ sum resolved_raw + boosted)
All three input scatters are re-anchored to their own κ=1 entry so that
DLL(κ=1)=0 by construction (P1 / Asimov, n = μ at κ=1 same sample).
Numerical CL ranges are reported in the body text + summary table."""
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
# Single source of truth for w68 extraction (see lib/dll.py docstring +
# review_consistency_c §1.1). 95% CL re-uses the same poly_coef.
from lib.dll import poly4_w68


def w_at_thresh(coef, k_lo_grid, k_hi_grid, thresh, n_fine=5000):
    """Connected-region width at a given threshold around the polynomial
    minimum. Algorithm identical to poly4_w68's internal loop; only the
    threshold is parameterised so 95% CL re-uses the same poly_coef."""
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


def anchor_at_k1(grid, vals):
    """Subtract vals at κ=1 (the scan's own κ=1 entry) so DLL(κ=1)=0."""
    return vals - float(vals[int(np.argmin(np.abs(grid - 1.0)))])


def channel_cl(grid, raw_anchored, fit_window=None):
    """Returns dict with the canonical 68% + 95% CL extracted via
    lib.dll.poly4_w68 and the same poly_coef re-used for the 95% width.

    fit_window : optional (κ_lo, κ_hi) to restrict the 4th-order fit to
    the well region. Used at 10 TeV (0.8, 1.2) to keep the κ̂ near 1 and
    capture the well shape without far-κ leverage — matches fig9 of the
    resolved-only headline (where full-grid fit puts κ̂≈1.028 and creates
    a visible scatter–line anchor offset)."""
    if fit_window is not None:
        lo, hi = fit_window
        m = (grid >= lo - 1e-9) & (grid <= hi + 1e-9)
        grid_fit, vals_fit = grid[m], raw_anchored[m]
    else:
        grid_fit, vals_fit = grid, raw_anchored
    res  = poly4_w68(grid_fit, vals_fit)
    coef = np.asarray(res['poly_coef'])
    w95, lo95, hi95 = w_at_thresh(coef, grid_fit.min(), grid_fit.max(), 1.92)
    return dict(coef=coef,
                w68=res['w68_connected'], lo68=res['k3_lo'], hi68=res['k3_hi'],
                w95=w95, lo95=lo95, hi95=hi95,
                kmin=res['k3_min'], poly_min_raw=res['poly_min_raw'])


def resolved_raw(npz_path):
    R = np.load(npz_path, allow_pickle=True)
    return np.asarray(R['fit_grid'], dtype=np.float64), \
           np.asarray(R['raw_vals'], dtype=np.float64)


BOOSTED = {
 '3tev':  (np.array([0.2,0.4,0.6,0.8,0.9,1.0,1.1,1.2,1.4,1.6,1.8]),
           np.array([1.979129,0.974742,0.374819,0.079510,0.018070,0.0,
                     0.020456,0.063519,0.180815,0.256348,0.275134])),
 '10tev': (np.array([0.2,0.4,0.6,0.8,0.84,0.88,0.92,0.96,1.0,
                     1.04,1.08,1.12,1.16,1.2,1.4,1.6,1.8]),
           np.array([120.542288,63.899270,24.777416,5.348065,3.330502,1.899730,
                     0.979175,0.341904,0.0,0.187548,0.583891,1.240877,
                     2.193318,3.008681,11.155762,17.964321,18.624093])),
}

STAGES = [
    # key, name, color, npz, xlim, ylim, xticks, fit_window
    ('3tev',  '3 TeV',  ps.C_3TEV,  '/root/work/Papers/check/seed_ensemble/3tev_seed42_dll_morphing.npz',
       (0.2, 1.8),  (-1.5, 3.0),  np.arange(0.2, 1.81, 0.2), None),
    ('10tev', '10 TeV', ps.C_10TEV, '/root/work/Papers/check/seed_ensemble/10tev_seed42_dll_morphing.npz',
       (0.8, 1.2),  (-1.5, 3.0),  np.arange(0.8, 1.21, 0.1), (0.8, 1.2)),
]

fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.5))

CL_REPORT = {}        # collect numerical CL ranges for the body-text print

for a, (key, name, c, npz_path, xlim, ylim, xticks, fwin) in zip(ax, STAGES):
    # ── ingredients ──
    grid_res, raw_res = resolved_raw(npz_path)
    res_anchored      = anchor_at_k1(grid_res, raw_res)
    g_b, y_b          = BOOSTED[key]
    boost_anchored    = anchor_at_k1(g_b, y_b)

    # Combined channel = per-κ sum of the two raw scatters at common grid
    # (the boosted grid and the resolved fit_grid match exactly at 3 + 10 TeV)
    assert np.allclose(grid_res, g_b), \
        f'{key}: resolved fit_grid does not match boosted grid'
    com_anchored = res_anchored + boost_anchored

    # Canonical fit + 68/95 CL via lib.dll.poly4_w68 (one call per channel).
    # fwin restricts the fit at 10 TeV to (0.8, 1.2) so the well is captured
    # cleanly and the resolved number matches fig9 exactly. Boosted uses the
    # full grid (it is the collaborator's input, not refit by us).
    res = channel_cl(g_b, res_anchored, fit_window=fwin)
    bst = channel_cl(g_b, boost_anchored)
    com = channel_cl(g_b, com_anchored, fit_window=fwin)
    CL_REPORT[name] = dict(boosted=bst, resolved=res, combined=com)

    # Display curves — same Phase-A2 shift convention as poly4_w68 internally
    kfine = np.linspace(g_b.min(), g_b.max(), 4001)
    res_poly   = np.maximum(np.polyval(res['coef'], kfine) - res['poly_min_raw'], 0.0)
    boost_poly = np.maximum(np.polyval(bst['coef'], kfine) - bst['poly_min_raw'], 0.0)
    com_poly   = np.maximum(np.polyval(com['coef'], kfine) - com['poly_min_raw'], 0.0)
    res_s, boost_s, com_s = res_poly, boost_poly, com_poly

    # CL bands of the combined fit
    a.axvspan(com['lo95'], com['hi95'], color=c, alpha=0.10, zorder=1)
    a.axvspan(com['lo68'], com['hi68'], color=c, alpha=0.22, zorder=2)

    # Three poly4 lines, single colour per panel
    a.plot(kfine, boost_poly, ls=':',  color=c, lw=1.7, label='boosted')
    a.plot(kfine, res_poly,   ls='--', color=c, lw=1.7, label='resolved')
    a.plot(kfine, com_poly,   ls='-',  color=c, lw=2.4, label='combined')

    # (scatter overlays removed at user request — lines only)

    a.axhline(0.50, color='0.45', ls=':', lw=1.0)
    a.axhline(1.92, color='0.65', ls=':', lw=1.0)
    a.axvline(1.0,  color='0.7',  lw=0.7, alpha=0.5)
    a.set_xlim(*xlim); a.set_ylim(*ylim)
    a.set_xticks(xticks)
    a.set_xlabel(r'$\kappa_3 = \lambda_3/\lambda_3^{\rm SM}$')
    a.set_ylabel(r'$-\Delta\ln L$')
    a.legend(loc='lower left', fontsize=9, framealpha=0.92, handlelength=2.2)
    ps.hep_tag(a, [rf'$\sqrt{{s}}={name.split()[0]}$ TeV'],
               loc='lower right', simulation=True)

fig.tight_layout(w_pad=2.0)
fig.savefig('/root/work/Papers/figures/png/fig10_dll_combined.png', dpi=200, bbox_inches='tight')
# fig.savefig('/root/work/Papers/figures/png/fig10_dll_combined.pdf', bbox_inches='tight')   # PDF disabled — use png/
print('saved /root/work/Papers/figures/png/fig10_dll_combined.{png,pdf}')

# Print CL ranges per channel (P1 single-seed raw + poly4 via lib.dll.poly4_w68).
print('\n=== CL ranges per channel (P1 single-seed raw + poly4, via lib.dll) ===')
for name, R in CL_REPORT.items():
    for ch in ('boosted', 'resolved', 'combined'):
        ch_r = R[ch]
        print(f'{name:>6} {ch:>9}: 68% [{ch_r["lo68"]:.3f}, {ch_r["hi68"]:.3f}] '
              f'(w68={ch_r["w68"]:.4f})   '
              f'95% [{ch_r["lo95"]:.3f}, {ch_r["hi95"]:.3f}] '
              f'(w95={ch_r["w95"]:.4f})   '
              f'κ_min={ch_r["kmin"]:.3f}')
