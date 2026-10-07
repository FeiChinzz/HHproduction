import os
import gc
import time
import shutil
import uproot
import numpy as np
import pandas as pd
import h5py
import ripser
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 1. 樣本與參數設定
# ==============================================================================
OUTPUT_H5 = '/data/mucollider/two_boosted/10TeV/data_HLfeature_v2.h5'
TEMP_DIR = '/data/mucollider/two_boosted/10TeV/temp_h5_parts'
JET_NAME = 'VLCjetR10N2'

# 設定要使用的 CPU 核心數（例如保留 2 核給系統，或手動指定如 NUM_WORKERS = 8）
NUM_WORKERS = 32

SAMPLES = [
    {'name': 'jjBG',    'path': '/data/mucollider/two_boosted/10TeV/jjBG_ptcut_500K/delphes_output_v2.root', 'sigbg': 0.0, 'everytype': 2.0},
    {'name': 'ttBG',    'path': '/data/mucollider/two_boosted/10TeV/ttBG_ptcut_500K/delphes_output_v2.root', 'sigbg': 0.0, 'everytype': 3.0},
    {'name': 'bbBG',    'path': '/data/mucollider/two_boosted/10TeV/bbBG_ptcut_500K/delphes_output_v2.root', 'sigbg': 0.0, 'everytype': 4.0},
    {'name': 'wwBG',    'path': '/data/mucollider/two_boosted/10TeV/wwBG_500K/delphes_output_v2.root',    'sigbg': 0.0, 'everytype': 5.0},
    {'name': 'wwvvBG',  'path': '/data/mucollider/two_boosted/10TeV/wwvvBG_500K/delphes_output_v2.root',  'sigbg': 0.0, 'everytype': 6.0},
    {'name': 'zzBG',    'path': '/data/mucollider/two_boosted/10TeV/zzBG_500K/delphes_output_v2.root',    'sigbg': 0.0, 'everytype': 7.0},
    {'name': 'zzvvBG',  'path': '/data/mucollider/two_boosted/10TeV/zzvvBG_500K/delphes_output_v2.root',  'sigbg': 0.0, 'everytype': 8.0},
    {'name': 'hzvvBG',  'path': '/data/mucollider/two_boosted/10TeV/hzvvBG_500K/delphes_output_v2.root',  'sigbg': 0.0, 'everytype': 9.0}
]

PT_MIN = 200.0
LEAD_PT_MAX = 850.0
SUBLEAD_PT_MAX = 600.0
MASS_MIN = 65.0
MASS_MAX = 150.0
EETA_MAX = -np.log(np.tan(np.radians(10) / 2))

EFF_EDGES = np.array([20, 38, 56, 74, 92, 110, 128, 146, 164, 182, 200])
EFF_VAL = np.array([81.8, 93.9, 95.6, 95.8, 97.4, 97.8, 97.3, 97.2, 98.5, 97.9]) / 100.0

# ==============================================================================
# 2. 物理計算輔助函式
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
# 3. 單核心 Worker 函式
# ==============================================================================
def worker_task(sample_cfg):
    fname = sample_cfg['path']
    s_name = sample_cfg['name']
    val_sigbg = sample_cfg['sigbg']
    val_everytype = sample_cfg['everytype']
    temp_h5 = os.path.join(TEMP_DIR, f"temp_{s_name}.h5")
    
    if not os.path.exists(fname):
        print(f"[Worker Error] 找不到檔案: {fname}")
        return None

    t0 = time.time()
    print(f"--> [開始] Process {s_name} (PID: {os.getpid()})")
    
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
                    
    df_lead = pd.DataFrame(leading).drop(columns=['weight'], errors='ignore').add_suffix('_leading')
    df_sublead = pd.DataFrame(subleading).drop(columns=['weight'], errors='ignore').add_suffix('_subleading')
    df_evt = pd.DataFrame(evt).drop(columns=['weight'], errors='ignore')
    
    df = pd.concat([df_lead, df_sublead, df_evt], axis=1)
    df['target_sigbg'] = val_sigbg
    df['target_everytype'] = val_everytype
    
    # 寫入各自獨立的暫存 H5 檔案
    with h5py.File(temp_h5, 'w') as f:
        for col in df.columns:
            f.create_dataset(col, data=df[col].to_numpy(), compression='gzip')
            
    print(f"<-- [完成] Process {s_name}: 通過 {len(df)}/{n_total} 筆 | 耗時: {time.time() - t0:.1f}s")
    
    del events, leading, subleading, evt, df, df_lead, df_sublead, df_evt
    gc.collect()
    return temp_h5

# ==============================================================================
# 4. 合併所有暫存 H5 檔案
# ==============================================================================
def merge_all_temp_h5(temp_files, target_file):
    print("\n" + "="*60)
    print("正在將所有 Worker 的暫存檔案合併為單一 HDF5 檔案...")
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
            
    print(f"合併完成！耗時: {time.time() - t0:.2f} 秒")
    print("="*60)

# ==============================================================================
# 5. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    t_total_start = time.time()
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print(f"啟動多核心批次運算，使用核心數 (Workers): {NUM_WORKERS}")
    
    # 使用進程池平行處理 10 個樣本
    with Pool(processes=NUM_WORKERS) as pool:
        temp_files = pool.map(worker_task, SAMPLES)
        
    # 合併暫存檔案
    merge_all_temp_h5(temp_files, OUTPUT_H5)
    
    # 刪除暫存資料夾
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    print(f"\n全部運算與儲存完畢！總耗時: {(time.time() - t_total_start)/60:.2f} 分鐘")
    print(f"最終輸出檔案: {OUTPUT_H5}")
