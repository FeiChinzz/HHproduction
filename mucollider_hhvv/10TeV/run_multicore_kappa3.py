import os
import gc
import time
import shutil
import warnings
import uproot
import numpy as np
import pandas as pd
import h5py
import ripser
from multiprocessing import Pool

# Suppress numpy deprecation warnings from ragged arrays
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# ==============================================================================
# 1. Configuration & Sample List Definition
# ==============================================================================
OUTPUT_H5 = '/data/mucollider/two_boosted/10TeV/scan_kappa3_hhmumu_small_HL_v2.h5'
TEMP_DIR = '/data/mucollider/two_boosted/10TeV/temp_scan_kappa3_parts'
JET_NAME = 'VLCjetR10N2'
NUM_WORKERS = 11

# Define kappa_3 parameter values corresponding to run_01 through run_11
KAPPA_VALUES = [0.8, 0.84, 0.88, 0.92, 0.96, 1, 1.04, 1.08, 1.12, 1.16, 1.2]

# Construct sample metadata dictionary list
SAMPLES = []
for idx, kappa_val in enumerate(KAPPA_VALUES, start=1):
    run_str = f"{idx:02d}"
    file_path = f"/data/mucollider/two_boosted/10TeV/scan_kappa3_hhmumu_small/Events/run_{run_str}_decayed_1/delphes_output_v2.root"
    SAMPLES.append({
        'name': f"run_{run_str}",
        'path': file_path,
        'kappa_3': float(kappa_val)
    })

# Selection cuts
PT_MIN = 200.0
LEAD_PT_MAX = 850.0
SUBLEAD_PT_MAX = 600.0
MASS_MIN = 65.0
MASS_MAX = 150.0
EETA_MAX = -np.log(np.tan(np.radians(10) / 2))  # ~ 2.436

EFF_EDGES = np.array([20, 38, 56, 74, 92, 110, 128, 146, 164, 182, 200])
EFF_VAL = np.array([81.8, 93.9, 95.6, 95.8, 97.4, 97.8, 97.3, 97.2, 98.5, 97.9]) / 100.0

# ==============================================================================
# 2. Physics Feature Calculation Helper Functions
# ==============================================================================
def jet_eff(pt):
    idx = np.clip(np.searchsorted(EFF_EDGES, pt, side='right') - 1, 0, len(EFF_VAL) - 1)
    return EFF_VAL[idx]

def btag(ev, j, J):
    return ev[f'{J}.BTag'][j]

def tau_ratios(ev, j, J):
    tau = ev[f'{J}.Tau[5]'][j]
    t21 = tau[1] / tau[0] if tau[0] != 0 else np.nan
    t32 = tau[2] / tau[1] if tau[1] != 0 else np.nan
    return t21, t32

def mass(p4):
    return np.sqrt(max(p4.t**2 - p4.x**2 - p4.y**2 - p4.z**2, 0))

def softdrop(ev, j, J):
    return mass(ev[f'{J}.SoftDroppedP4[5]'][j * 5]), ev[f'{J}.NSubJetsSoftDropped'][j]

def music_scale(ev, j, J, J0):
    if J == J0:
        return 1.0
    k = np.argmin((ev[f'{J0}.Eta'] - ev[f'{J}.Eta'][j])**2 + (ev[f'{J0}.Phi'] - ev[f'{J}.Phi'][j])**2)
    return ev[f'{J}.PT'][j] / ev[f'{J0}.PT'][k]

def event_level(ev):
    return ev['MissingET.MET'][0], ev['ScalarHT.HT'][0]

def ecf(pt, eta, phi):
    if len(pt) < 2:
        return np.nan, np.nan
    z = pt / pt.sum()
    dphi = np.abs(phi[:, None] - phi[None, :])
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    R = np.sqrt((eta[:, None] - eta[None, :])**2 + dphi**2)
    e2 = z @ R @ z / 2
    e3 = np.einsum('i,j,k,ij,ik,jk->', z, z, z, R, R, R, optimize=True) / 6
    Rmax = np.maximum(np.maximum(R[:, :, None], R[:, None, :]), R[None, :, :])
    with np.errstate(invalid='ignore', divide='ignore'):
        T = np.where(Rmax > 0, R[:, :, None] * R[:, None, :] * R[None, :, :] / Rmax, 0)
    e3_2 = np.einsum('i,j,k,ijk->', z, z, z, T) / 6
    return (e3 / (e2**3) if e2 != 0 else np.nan), (e3_2 / (e2**2) if e2 != 0 else np.nan)

def event_shapes(pt, eta, phi):
    pt, eta, phi = (np.asarray(x, dtype=np.float64) for x in (pt, eta, phi))
    p = np.c_[pt * np.cos(phi), pt * np.sin(phi)]
    sum_E = (pt * np.cosh(eta)).sum()
    lam = np.sort(np.linalg.eigvalsh((p.T / pt) @ p / pt.sum()))[::-1]
    sphericity_T = 2 * lam[1] / (lam[0] + lam[1])
    ang = np.r_[phi + np.pi / 2 - 1e-7, phi + np.pi / 2 + 1e-7]
    s = np.sign(np.cos(ang)[:, None] * p[:, 0] + np.sin(ang)[:, None] * p[:, 1])
    thrust_T = np.hypot(*(s @ p).T).max() / pt.sum()
    return sum_E, len(pt), sphericity_T, thrust_T

def dijet_kin(ev, J):
    masses = ev[f'{J}.Mass']
    m1, m2 = (masses[0], masses[1]) if masses[0] >= masses[1] else (masses[1], masses[0])
    X_HH = np.sqrt(((m1 - 124) / (0.1 * m1 + 1e-5))**2 + ((m2 - 115) / (0.1 * m2 + 1e-5))**2)
    
    p4 = [0.0, 0.0, 0.0, 0.0]
    for j in range(2):
        pt, eta, phi, m = ev[f'{J}.PT'][j], ev[f'{J}.Eta'][j], ev[f'{J}.Phi'][j], ev[f'{J}.Mass'][j]
        p4[1] += pt * np.cos(phi)
        p4[2] += pt * np.sin(phi)
        p4[3] += pt * np.sinh(eta)
        p4[0] += np.sqrt(m**2 + (pt * np.cos(phi))**2 + (pt * np.sin(phi))**2 + (pt * np.sinh(eta))**2)
    
    M_JJ = np.sqrt(max(0, p4[0]**2 - p4[1]**2 - p4[2]**2 - p4[3]**2))
    pT_JJ = np.sqrt(p4[1]**2 + p4[2]**2)
    dEta_JJ = np.abs(ev[f'{J}.Eta'][0] - ev[f'{J}.Eta'][1])
    return X_HH, M_JJ, pT_JJ, dEta_JJ

def cal_TDA(eflow, ev, J):
    eta_center = np.mean(ev[f'{J}.Eta'])
    jet_phi = ev[f'{J}.Phi']
    phi_center = np.arctan2(np.sin(jet_phi[0]) + np.sin(jet_phi[1]), np.cos(jet_phi[0]) + np.cos(jet_phi[1]))
    
    filtered_particles = np.array([(eta, phi) for pt, eta, phi in eflow.values() if pt > 1.0])
    if len(filtered_particles) == 0:
        P = np.empty((0, 2))
    else:
        part_eta = filtered_particles[:, 0]
        part_phi = filtered_particles[:, 1]
        d_eta = part_eta - eta_center
        d_phi = (part_phi - phi_center + np.pi) % (2.0 * np.pi) - np.pi
        P = np.column_stack([d_eta, d_phi])
    
    return ripser.ripser(P)['dgms']

def cal_TDA_var(diagrams):
    H_0 = np.sum(diagrams[0][:-1, 1] - diagrams[0][:-1, 0])
    H_1 = np.sum(diagrams[1][:, 1] - diagrams[1][:, 0]) if len(diagrams[1]) > 0 else 0.0
    
    l_0 = diagrams[0][:-1, 1] - diagrams[0][:-1, 0]
    L_0 = np.sum(l_0)
    s_0 = (l_0 / L_0) * np.log2(l_0 / L_0) if L_0 > 0 else 0.0
    S_0 = -np.sum(s_0)
    
    if len(diagrams[1]) > 0:
        l_1 = diagrams[1][:, 1] - diagrams[1][:, 0]
        L_1 = np.sum(l_1)
        s_1 = (l_1 / L_1) * np.log2(l_1 / L_1) if L_1 > 0 else 0.0
        S_1 = -np.sum(s_1)
        LB_1 = np.sum(diagrams[1][:, 0] * l_1)
    else:
        S_1 = 0.0
        LB_1 = 0.0
        
    return H_0, H_1, S_0, S_1, LB_1

# ==============================================================================
# 3. Worker Task for Single Process
# ==============================================================================
def worker_task(sample_cfg):
    fname = sample_cfg['path']
    s_name = sample_cfg['name']
    val_kappa_3 = sample_cfg['kappa_3']
    temp_h5 = os.path.join(TEMP_DIR, f"temp_{s_name}.h5")
    
    if not os.path.exists(fname):
        print(f"[Worker Error] File not found: {fname}")
        return None

    t0 = time.time()
    print(f"--> [Started] Process {s_name} (PID: {os.getpid()}, kappa_3: {val_kappa_3})")
    
    J = JET_NAME
    J0 = J.replace('_MUSIC', '')
    branches = [
        f'{J}.PT', f'{J}.Eta', f'{J}.Phi', f'{J}.Mass', f'{J}.BTag', f'{J}.Tau[5]', f'{J}.SoftDroppedP4[5]',
        f'{J}.NSubJetsSoftDropped', f'{J}.Constituents', f'{J0}.PT', f'{J0}.Eta', f'{J0}.Phi',
        'MissingET.MET', 'ScalarHT.HT',
        'EFlowTrack.fUniqueID', 'EFlowTrack.PT', 'EFlowTrack.Eta', 'EFlowTrack.Phi',
        'EFlowPhoton.fUniqueID', 'EFlowPhoton.ET', 'EFlowPhoton.Eta', 'EFlowPhoton.Phi',
        'EFlowNeutralHadron.fUniqueID', 'EFlowNeutralHadron.ET', 'EFlowNeutralHadron.Eta', 'EFlowNeutralHadron.Phi'
    ]
    
    events = uproot.open(fname)['Delphes'].arrays(branches)
    keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in events.keys()]
    events = [dict(zip(keys, row)) for row in zip(*events.values())]
    n_total = len(events)
    
    leading = {k: [] for k in ('weight', 'msd', 'm_J', 'pt_J', 'tau21', 'tau32', 'D2', 'N2', 'BTag', 'n_sd')}
    subleading = {k: [] for k in ('weight', 'msd', 'm_J', 'pt_J', 'tau21', 'tau32', 'D2', 'N2', 'BTag', 'n_sd')}
    evt = {k: [] for k in ('weight', 'sum_E', 'n_eflow', 'sphericity_T', 'thrust_T', 'missET', 'HT', 'X_HH', 'M_JJ', 'pT_JJ', 'dEta_JJ', 'H_0', 'H_1', 'S_0', 'S_1', 'LB_1')}
    
    for num, ev in enumerate(events):
        pts = ev[f'{J}.PT']
        etas = ev[f'{J}.Eta']
        masses = ev[f'{J}.Mass']
        
        # Pre-selection criteria
        if (
            len(pts) != 2
            or np.any(pts < PT_MIN)
            or np.any(np.abs(etas) > EETA_MAX)
            or (pts[0] > LEAD_PT_MAX)
            or (pts[1] > SUBLEAD_PT_MAX)
            or np.any(masses < MASS_MIN)
            or np.any(masses > MASS_MAX)
        ):
            continue
            
        w = np.prod(jet_eff(ev[f'{J0}.PT']))
        
        eflow = {}
        for b, p in [('EFlowTrack', 'PT'), ('EFlowPhoton', 'ET'), ('EFlowNeutralHadron', 'ET')]:
            for uid, pt, eta, phi in zip(ev[f'{b}.fUniqueID'], ev[f'{b}.{p}'], ev[f'{b}.Eta'], ev[f'{b}.Phi']):
                eflow[uid] = (pt, eta, phi)
                
        diagrams = cal_TDA(eflow, ev, J)
        tda_vars = cal_TDA_var(diagrams)
        evt_shapes = event_shapes(*np.array(list(eflow.values())).T)
        evt_level = event_level(ev)
        dijet = dijet_kin(ev, J)
        
        for k, v in zip(evt, (w,) + evt_shapes + evt_level + dijet + tda_vars):
            evt[k].append(v)
            
        for j, jet in enumerate(ev[f'{J}.Constituents']):
            const = np.array([eflow[r] for r in jet if r in eflow]).reshape(-1, 3)
            msd_val, nsub = softdrop(ev, j, J)
            s = music_scale(ev, j, J, J0)
            jet_feat = (w, msd_val * s, masses[j] * s, pts[j]) + tau_ratios(ev, j, J) + ecf(*const.T) + (btag(ev, j, J), nsub)
            
            if j == 0:
                for k, v in zip(leading, jet_feat):
                    leading[k].append(v)
            elif j == 1:
                for k, v in zip(subleading, jet_feat):
                    subleading[k].append(v)
                    
    # Construct combined DataFrame
    df_lead = pd.DataFrame(leading).drop(columns=['weight'], errors='ignore').add_suffix('_leading')
    df_sublead = pd.DataFrame(subleading).drop(columns=['weight'], errors='ignore').add_suffix('_subleading')
    df_evt = pd.DataFrame(evt).drop(columns=['weight'], errors='ignore')
    
    df = pd.concat([df_lead, df_sublead, df_evt], axis=1)
    
    # Assign target label
    df['target_kappa_3'] = val_kappa_3
    
    # Save to independent temporary HDF5 file
    with h5py.File(temp_h5, 'w') as f:
        for col in df.columns:
            f.create_dataset(col, data=df[col].to_numpy(), compression='gzip')
            
    print(f"<-- [Finished] Process {s_name}: Passed {len(df)}/{n_total} events | Elapsed: {time.time() - t0:.1f}s")
    
    del events, leading, subleading, evt, df, df_lead, df_sublead, df_evt
    gc.collect()
    return temp_h5

# ==============================================================================
# 4. Merge Temporary HDF5 Files
# ==============================================================================
def merge_all_temp_h5(temp_files, target_file):
    print("\n" + "="*70)
    print("Merging temporary HDF5 files into final output...")
    t0 = time.time()
    
    if os.path.exists(target_file):
        os.remove(target_file)
        
    with h5py.File(target_file, 'w') as out_f:
        first = True
        for temp_f in temp_files:
            if not temp_f or not os.path.exists(temp_f):
                continue
            with h5py.File(temp_f, 'r') as in_f:
                for key in in_f.keys():
                    data = in_f[key][:]
                    if first:
                        out_f.create_dataset(key, data=data, maxshape=(None,), chunks=True, compression='gzip')
                    else:
                        dset = out_f[key]
                        old_len = len(dset)
                        dset.resize(old_len + len(data), axis=0)
                        dset[old_len:] = data
            first = False
            
    print(f"Merge successfully completed! Elapsed: {time.time() - t0:.2f}s")
    print("="*70)

# ==============================================================================
# 5. Main Execution Entry
# ==============================================================================
if __name__ == '__main__':
    t_total_start = time.time()
    
    # Ensure temporary directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    out_dir = os.path.dirname(OUTPUT_H5)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"Starting multi-core extraction for {len(SAMPLES)} samples using {NUM_WORKERS} workers...")
    print(f"Target Output: {OUTPUT_H5}")
    
    # Execute with multiprocessing Pool
    with Pool(processes=NUM_WORKERS) as pool:
        temp_files = pool.map(worker_task, SAMPLES)
        
    # Merge temporary parts into final HDF5 file
    merge_all_temp_h5(temp_files, OUTPUT_H5)
    
    # Clean up temporary directory
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    print(f"\nAll processes completed successfully! Total elapsed time: {(time.time() - t_total_start)/60:.2f} minutes")
    print(f"Final dataset saved to: {OUTPUT_H5}")
