# Shared figure formats for the boosted-region figures

This package contains the figure formats used for the resolved-region plots,
so the boosted figures can look identical in the paper. Everything runs with
`python3 <script>.py` after editing the `EDIT HERE` block at the top of each
script. Keep `paper_style.py` in the same folder as the scripts.

## Files

| file | what it is |
|---|---|
| `paper_style.py` | shared style: fonts, figure sizes (`SIZE1/SIZE12/SIZE22`), `compact_12()`, `hep_tag()` |
| `fig_template_kin.py` | template: 1x2 kinematic distributions (filled background + kappa3 hypotheses) |
| `fig_template_ml_scores.py` | template: 1x2 D_HH (log scale) / D_kappa3 score panels |
| `fig_template_case2_dll.py` | template: 1x2 Case-2 -DeltalnL(kappa3) with poly4 fit + 68%/95% CL shaded bands |
| `fig_template_case1_dll.py` | template: 1x2 Case-1 (observability, null = no HH) -DeltalnL scan |
| `template_*.pdf` | outputs of the three templates, for reference |
| `fig_result_combined_example.pdf` | a real resolved-side figure in this format |
| `compute_boosted_shap.py` | your model + your test HL/LL h5 → `boosted_shap.npz` |
| `fig_shap_boosted.py` | `boosted_shap.npz` → paper-format SHAP beeswarm PDF |
| `fig_shap_boosted_3tev_example.pdf` | example SHAP output (made with `model_mucollider_10.h5`; please regenerate with your latest model) |

## House rules (all figures)

- Vector **PDF**, not PNG.
- Do not set font sizes by hand: call `ps.set_style()` and, for 1x2 figures,
  `ps.compact_12()`. In-panel tags: `ps.hep_tag(..., fontsize=7.5)`.
- Figure width = LaTeX insertion width, so fonts render at true size:
  1x2 → `figsize=ps.SIZE12`, inserted with `width=\textwidth`;
  single panel → `ps.SIZE1`, inserted with `width=0.667\textwidth`.
- Colours: 3 TeV = `ps.C_3TEV` (blue), 10 TeV = `ps.C_10TEV` (red);
  kappa3 histograms: 0.4 = red, 1.0 = black, 1.6 = orange.

## DLL template notes

`fig_template_case2_dll.py` implements the convention used on the resolved side:
the scan is shifted to 0 at kappa3 = 1, fitted with a fourth-order
polynomial, and the 68%/95% intervals are the connected regions below
0.5/1.92 **referenced to the fitted minimum**. `WIN_10TEV = (0.8, 1.2)`
restricts the 10 TeV fit to the refined grid; without it the quartic dips
below zero at kappa3 = 1 (the scan is too steep outside the window). The
dummy arrays in the script are your Colab boosted scans, so the output
already reproduces your 3 TeV interval and shows the window in action.

## SHAP figure notes

1. Edit the paths at the top of `compute_boosted_shap.py` (your trained
   D_HH model, your `test_HLfeature.h5` / `test_LLfeature.h5`) and run it
   (needs the `shap` package). It assumes the inputs are still the 16 HL
   features plus the 2 x 20 x 4 low-level `particle_info` array from
   `generate_{HL,LL}features.ipynb`; tell me if that changed.
2. Edit the npz path/title in `fig_shap_boosted.py` and run it.

Conventions inside the figure:
- columns sorted left→right by the mean |SHAP| over the evaluated events
  (most important on the left);
- `particle cloud` = the signed SUM of all per-constituent attributions per
  event — SHAP is additive, so this is the net effect of the low-level
  stream on the score (internal +/− contributions cancel);
- its colour encodes one single-unit proxy, the total constituent |E_T|
  (channel 0), normalised to its own 5–95% quantile range; the colour never
  affects positions or the ranking;
- the y-range is symmetric, rounded up to the next 0.1.
