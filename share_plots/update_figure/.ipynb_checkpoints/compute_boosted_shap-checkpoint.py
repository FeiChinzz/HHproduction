#!/usr/bin/env python3
"""Compute SHAP for the boosted D_HH (Keras) and save one npz for plotting.
Inputs follow the generate_{HL,LL}features.ipynb conventions
(16 HL features + LL particle_info of 2 jets x 20 particles x 4 channels).
Run this first, then fig_shap_boosted.py on the produced npz."""
import os, sys
_LIB = os.environ.get('HHML_CONDA_LIB')          # conda env lib dir (optional)
if _LIB and _LIB not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np, h5py
from tensorflow import keras
import shap

# ══════════════════ EDIT HERE ═══════════════════════════════════════════
MODEL   = '/path/to/your/ML_model/model_mucollider_10.h5'   # trained D_HH
TEST_HL = '/path/to/your/test_HLfeature.h5'
TEST_LL = '/path/to/your/test_LLfeature.h5'
OUT     = './boosted_shap.npz'                              # output npz
N_BG    = 128     # SHAP background sample size
N_EVAL  = 1500    # events to explain
# ═════════════════════════════════════════════════════════════════════════

HL_NAMES = ['BTag_leading', 'BTag_subleading', 'H_0', 'H_1', 'LB_1',
            'M_JJ', 'S_0', 'S_1', 'X_HH', 'dEta_JJ',
            'm_J_leading', 'm_J_subleading', 'missET', 'pT_JJ',
            'pt_J_leading', 'pt_J_subleading']
LL_CHAN  = ['ll_ET_or_PT', 'll_eta', 'll_phi', 'll_type']

with h5py.File(TEST_HL, 'r') as f:
    cols = []
    for nm in HL_NAMES:
        if nm.startswith('BTag'):
            cols.append(f[nm][:] & 2)          # WP70 bit
        else:
            cols.append(f[nm][:])
    HL_all = np.stack(cols, axis=1).astype(np.float32)
with h5py.File(TEST_LL, 'r') as f:
    LL_all = np.asarray(f['particle_info'][:], dtype=np.float32)
print(f'HL {HL_all.shape}   LL {LL_all.shape}')

model = keras.models.load_model(MODEL, compile=False, safe_mode=False)

rng = np.random.RandomState(0)
N = len(HL_all)
idx_bg = rng.choice(N, size=min(N_BG, N), replace=False)
idx_ev = rng.choice(N, size=min(N_EVAL, N), replace=False)
n_ev = len(idx_ev)

explainer = shap.GradientExplainer(model, [HL_all[idx_bg], LL_all[idx_bg]])
shap_vals = explainer.shap_values([HL_all[idx_ev], LL_all[idx_ev]])

def _arr(s):
    a = np.asarray(s)
    return a[..., 0] if a.shape[-1] == 1 and a.ndim >= 2 else a

sv_HL = _arr(shap_vals[0])                       # (n_ev, 16)
sv_LL = _arr(shap_vals[1]).reshape(n_ev, 2, 20, 4)

np.savez(OUT,
         sv_HL=sv_HL.astype(np.float32),
         sv_LL=sv_LL.astype(np.float32),
         X_HL=HL_all[idx_ev].astype(np.float32),
         X_LL=LL_all[idx_ev].reshape(n_ev, 2, 20, 4).astype(np.float32),
         names_HL=np.array(HL_NAMES, dtype=object),
         names_LL=np.array(LL_CHAN, dtype=object),
         n_ev=n_ev, n_bg=len(idx_bg))
print(f'saved {OUT}')
