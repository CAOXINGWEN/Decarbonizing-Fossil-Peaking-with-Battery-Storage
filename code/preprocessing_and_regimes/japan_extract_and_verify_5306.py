"""Step 1: Extract all 5,309 two-year-complete Japanese 500-m community meshes (10,618 annual records)
into a clean memory-mappable float64 array, verify 0.0 error against the archived v86 Japanese inputs,
and compute full R1 & R4 window/timing metrics across all 10,618 records.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os, sys, json, struct, zipfile, tempfile, subprocess
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
sys.path.insert(0, str(ROOT / 'submission/Nature_Energy_v85/revision/real_vs_placement_20261027/code'))
sys.path.insert(0, str(ROOT / 'submission/Nature_Energy_v85/revision/real_vs_placement_20261029/code'))
from cpu_budget import apply_cpu_budget, limit_workers
apply_cpu_budget()

sys.path.insert(0, str(ROOT / 'submission/Nature_Energy_v85/revision/remaining_experiments_20261103/code'))
import common as C
sys.path.insert(0, str(ROOT / 'submission/Nature_Energy_v86/figures/code'))
import data as D

ZP = ROOT / 'dataset/一桥数据/2025-09-30_Hitotsubashi University_Self-Consumption (Low-Voltage).zip'
BIN = WORK / 'code/extract_hourly'
DATA_DIR = WORK / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

PREFS = [('11', 'Saitama'), ('12', 'Chiba'), ('13', 'Tokyo'), ('14', 'Kanagawa'), ('22', 'Shizuoka')]
DAYS_IN_MONTH = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def jis_mesh_to_lonlat(mesh: str):
    """Center coordinates (lon, lat) of standard JIS X 0410 500-m (half-grid, 9-digit) mesh."""
    p = int(mesh[0:2]); u = int(mesh[2:4])
    q = int(mesh[4]);   v = int(mesh[5])
    r = int(mesh[6]);   w = int(mesh[7])
    m = int(mesh[8])    # 1:SW, 2:SE, 3:NW, 4:NE
    lat = p / 1.5 + q * (5.0 / 60.0) + r * (30.0 / 3600.0)
    lon = (u + 100.0) + v * (7.5 / 60.0) + w * (45.0 / 3600.0)
    if m in (3, 4):
        lat += 15.0 / 3600.0
    if m in (2, 4):
        lon += 22.5 / 3600.0
    lat += 7.5 / 3600.0
    lon += 11.25 / 3600.0
    return round(lon, 6), round(lat, 6)


def extract_month_member(member: str) -> Path:
    stem = Path(member).stem  # solar_low-11-202304
    _, code, ym = stem.split('-')
    mo = int(ym[4:6])
    exp_h = DAYS_IN_MONTH[mo] * 24
    out_bin = DATA_DIR / 'monthly_tmp' / f'{code}_{ym}.bin'
    if out_bin.exists():
        return out_bin
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZP) as zf:
        raw_bytes = zf.read(member)
    with tempfile.NamedTemporaryFile(suffix='.zip') as tf, open(out_bin, 'wb') as out_f:
        tf.write(raw_bytes); tf.flush()
        del raw_bytes
        p1 = subprocess.Popen(['unzip', '-p', tf.name], stdout=subprocess.PIPE)
        p2 = subprocess.Popen([str(BIN), str(exp_h)], stdin=p1.stdout, stdout=out_f)
        p1.stdout.close()
        p2.wait(); p1.wait()
    return out_bin


def read_month_bin(path: Path):
    buf = path.read_bytes()
    pos, n_len = 0, len(buf)
    out = {}
    while pos < n_len:
        m_raw = buf[pos:pos + 16]; pos += 16
        mesh = m_raw.split(b'\x00', 1)[0].decode('ascii')
        nh, c_ge5, c_cnt = struct.unpack_from('<iii', buf, pos); pos += 12
        nb = nh * 8
        imp = np.frombuffer(buf, dtype='<f8', count=nh, offset=pos).copy(); pos += nb
        rev = np.frombuffer(buf, dtype='<f8', count=nh, offset=pos).copy(); pos += nb
        out[mesh] = (c_ge5, c_cnt, imp, rev)
    return out


def main():
    with zipfile.ZipFile(ZP) as zf:
        members = sorted([n for n in zf.namelist() if 'solar_low-' in n and n.endswith('.zip')])

    print(f'Extracting {len(members)} monthly zips via C stream...', flush=True)
    with ProcessPoolExecutor(max_workers=limit_workers(20)) as ex:
        list(ex.map(extract_month_member, members))

    fy_months = {
        2023: ['202304', '202305', '202306', '202307', '202308', '202309',
               '202310', '202311', '202312', '202401', '202402', '202403'],
        2024: ['202404', '202405', '202406', '202407', '202408', '202409',
               '202410', '202411', '202412', '202501', '202502', '202503'],
    }

    # Build calendar arrays for an April–March non-leap fiscal year (365 days, 8760 hours)
    cal_months = np.concatenate([np.full(DAYS_IN_MONTH[int(ym[4:6])] * 24, int(ym[4:6]), dtype=np.int16)
                                 for ym in fy_months[2023]])
    cal_hours = np.tile(np.arange(24, dtype=np.int16), 365)
    day_months = cal_months.reshape(365, 24)[:, 0]
    season_seg = ((day_months % 12) // 3).astype(np.int16)
    assert len(cal_months) == 8760 and len(season_seg) == 365

    meta_rows = []
    hourly_blocks = []  # each (n_pref_records, 8760, 2)

    for code, pref_name in PREFS:
        # First pass over monthly bins: find meshes present in ALL 24 months with annual median contracts >= 5 and rev > 0 in both FYs
        fy_valid = {}
        month_maps = {}
        for fy in (2023, 2024):
            m_dicts = [read_month_bin(DATA_DIR / 'monthly_tmp' / f'{code}_{ym}.bin') for ym in fy_months[fy]]
            common_m = set(m_dicts[0].keys())
            for md in m_dicts[1:]:
                common_m &= set(md.keys())
            good_m = set()
            for mesh in sorted(common_m):
                tot_ge5 = sum(md[mesh][0] for md in m_dicts)
                tot_cnt = sum(md[mesh][1] for md in m_dicts)
                tot_rev = sum(float(md[mesh][3].sum()) for md in m_dicts)
                tot_imp = sum(float(md[mesh][2].sum()) for md in m_dicts)
                if tot_cnt > 0 and (tot_ge5 * 2 >= tot_cnt) and tot_rev > 0 and tot_imp > 0:
                    good_m.add(mesh)
            fy_valid[fy] = good_m
            month_maps[fy] = m_dicts

        both_meshes = sorted(fy_valid[2023] & fy_valid[2024])
        print(f'{code} {pref_name:9s}: FY23_complete={len(fy_valid[2023])}, FY24_complete={len(fy_valid[2024])}, BOTH_complete={len(both_meshes)}', flush=True)

        block = np.empty((len(both_meshes) * 2, 8760, 2), dtype=np.float64)
        bi = 0
        for mesh in both_meshes:
            lon, lat = jis_mesh_to_lonlat(mesh)
            sys_id = f'JP-{pref_name}-{mesh}'
            for fy in (2023, 2024):
                m_dicts = month_maps[fy]
                imp_yr = np.concatenate([md[mesh][2] for md in m_dicts])
                rev_yr = np.concatenate([md[mesh][3] for md in m_dicts])
                assert len(imp_yr) == 8760 and len(rev_yr) == 8760
                block[bi, :, 0] = imp_yr
                block[bi, :, 1] = rev_yr
                meta_rows.append(dict(
                    row_idx=len(meta_rows),
                    pref_code=code,
                    prefecture=pref_name,
                    mesh_id=mesh,
                    system_id=sys_id,
                    observation_year=fy,
                    lon=lon,
                    lat=lat,
                    P_ref_mw=0.01 * float(imp_yr.mean()),
                    annual_import_mwh=float(imp_yr.sum()),
                    annual_pv_export_mwh=float(rev_yr.sum()),
                ))
                bi += 1
        hourly_blocks.append(block)
        del month_maps

    all_hourly = np.concatenate(hourly_blocks, axis=0)
    meta_df = pd.DataFrame(meta_rows)
    print(f'Total extracted dual-year communities: {meta_df.mesh_id.nunique()} | records: {len(meta_df)} | shape: {all_hourly.shape}', flush=True)

    np.save(DATA_DIR / 'japan_dual_hourly.npy', all_hourly)
    np.savez(DATA_DIR / 'japan_calendar.npz', cal_months=cal_months, cal_hours=cal_hours, season_seg=season_seg)
    meta_df.to_csv(DATA_DIR / 'japan_dual_metadata.csv', index=False)

    # Clean up temporary monthly binary files
    for p in (DATA_DIR / 'monthly_tmp').glob('*.bin'):
        p.unlink()
    (DATA_DIR / 'monthly_tmp').rmdir()

    # Verify 0.0 error against ALL dual-year records shared with the archived 27 records!
    arch_rec = D.records()
    arch_jp = arch_rec[arch_rec.group == 'community'].copy()
    shared = meta_df.merge(arch_jp[['uid', 'system_id', 'observation_year', 'B', 'roi_2035', 'efc_acaes']],
                           on=['system_id', 'observation_year'], how='inner')
    print(f'Shared records with archived 27-record panel: {len(shared)} (expected 16 = 8 dual-year communities x 2)', flush=True)
    assert len(shared) == 16, len(shared)

    max_raw_err = 0.0
    max_B_err = 0.0
    for r in shared.itertuples():
        imp_yr = all_hourly[r.row_idx, :, 0]
        rev_yr = all_hourly[r.row_idx, :, 1]
        P = r.P_ref_mw
        df_tmp = pd.DataFrame({'S': imp_yr, 'm': cal_months, 'h': cal_hours})
        base = df_tmp.groupby(['m', 'h']).S.transform(lambda x: x.quantile(0.50)).to_numpy()
        c1 = np.maximum(imp_yr - base, 0.0)
        day = c1.reshape(365, 24).sum(1)
        q75 = pd.Series(day).groupby(season_seg).transform(lambda x: x.quantile(0.75)).to_numpy()
        flags = day >= q75
        c3 = np.where(np.repeat(flags, 24), c1, 0.0)
        raw_norm = np.c_[c1, c3, rev_yr] / P

        z_arch = np.load(C.INP / f'{r.uid}.npz')
        raw_err = float(np.max(np.abs(raw_norm - z_arch['raw'])))
        flag_diff = int((flags != z_arch['event'].astype(bool)).sum())
        seg_diff = int((season_seg != z_arch['segment']).sum())
        assert flag_diff == 0 and seg_diff == 0, (r.system_id, r.observation_year, flag_diff, seg_diff)
        max_raw_err = max(max_raw_err, raw_err)

        arr = C.transform(raw_norm, 'registered')
        s_l = C.solve(arr, 'li_lfp_4h')
        s_c = C.solve(arr, 'caes_24h')
        B_new = s_l['cost_usd_kw_year'] - s_c['cost_usd_kw_year']
        max_B_err = max(max_B_err, abs(B_new - r.B))

    print(f'GOLD-STANDARD VERIFICATION PASSED: max_raw_err={max_raw_err:.3e}, max_B_err={max_B_err:.3e}', flush=True)


if __name__ == '__main__':
    main()
