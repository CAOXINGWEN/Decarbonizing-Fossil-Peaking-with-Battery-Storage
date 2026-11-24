"""Hourly allocation of the daily gas-peaking energy of the 26 provinces, using the official 2020 typical load curves
(NDRC/NEA, 发改运行〔2020〕1784号, digitised by digitize_ndrc.py).

Only the hourly shape of the provincial peaking task changes. Daily energy, high-demand days (C3 on the same days),
hourly curtailment, storage, costs (2035, 3.6 %), the central rule (no charging in task hours) and the solver are those
of remaining_experiments_20261103 (common.solve, COPT, source_output_rated). Variants
  flat24          daily energy spread evenly over 24 h (current main setting; must reproduce the delivered B)
  prop            weights proportional to the typical curve above its daily minimum
  capU            peaking capacity C = max(annual energy / U, max daily energy / 24) with U = 1,000, 1,400, 2,000 h;
                  each day runs E_d / C full-load hours, filled into the highest-load hours of the typical curve
  topK            daily energy spread evenly over the K highest-load hours, K = 4, 8, 12
Typical curves: workday curve for every day (the peaking table has month-day columns and no year); holiday curve as a
check for prop and cap1400. Hebei = mean of the normalised 冀北 and 河北南网 shapes, Inner Mongolia = 蒙西 and 蒙东.
Writes results/intraday_records.csv, results/intraday_summary.csv, results/peak_hours.csv.
"""
from pathlib import Path
import sys, json, multiprocessing as mp
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V86 = HERE.parents[1]
REV = V86.parent / 'Nature_Energy_v85/revision'
sys.path.insert(0, str(REV / 'remaining_experiments_20261103/code'))
sys.path.insert(0, str(REV / 'real_vs_placement_20261029/code'))
OUT = HERE / 'results'; OUT.mkdir(exist_ok=True)
GRID = {'Anhui': ['安徽'], 'Beijing': ['北京'], 'Fujian': ['福建'], 'Gansu': ['甘肃'], 'Guangdong': ['广东'], 'Guangxi': ['广西'],
        'Guizhou': ['贵州'], 'Hebei': ['冀北', '河北南网'], 'Henan': ['河南'], 'Hubei': ['湖北'], 'Hunan': ['湖南'],
        'Inner Mongolia': ['蒙西', '蒙东'], 'Jiangsu': ['江苏'], 'Jiangxi': ['江西'], 'Jilin': ['吉林'], 'Liaoning': ['辽宁'],
        'Ningxia': ['宁夏'], 'Qinghai': ['青海'], 'Shaanxi': ['陕西'], 'Shandong': ['山东'], 'Shanxi': ['山西'], 'Sichuan': ['四川'],
        'Tianjin': ['天津'], 'Xinjiang': ['新疆'], 'Yunnan': ['云南'], 'Zhejiang': ['浙江']}
VARIANTS = ['flat24', 'prop', 'cap1000', 'cap1400', 'cap2000', 'top4', 'top8', 'top12', 'prop_holiday', 'cap1400_holiday']


def shapes():
    s = pd.read_csv(HERE / 'ndrc2020_hourly_shapes.csv')
    out = {}
    for prov, grids in GRID.items():
        for dt in ('workday', 'holiday'):
            v = np.mean([s[(s.grid == g) & (s.day_type == dt)].sort_values('hour').rel.to_numpy() for g in grids], axis=0)
            assert v.shape == (24,)
            out[(prov, dt)] = (v - v.min()) / (v.max() - v.min())
    return out


def allocate(E, shape, variant):
    """Hourly task (365 x 24) from daily energy E (365,) and a relative typical curve (24,)."""
    order = np.argsort(-shape, kind='stable')
    T = np.zeros((365, 24))
    if variant == 'flat24':
        T[:] = E[:, None] / 24
    elif variant.startswith('prop'):
        w = shape / shape.sum(); T[:] = E[:, None] * w[None, :]
    elif variant.startswith('top'):
        k = int(variant[3:]); T[:, order[:k]] = E[:, None] / k
    elif variant.startswith('cap'):
        U = float(variant[3:7])
        C = max(E.sum() / U, E.max() / 24)
        for d in np.flatnonzero(E > 0):
            h = E[d] / C; n = int(np.floor(h + 1e-12))
            T[d, order[:n]] = C
            if n < 24 and h - n > 1e-12:
                T[d, order[n]] = (h - n) * C
    assert np.allclose(T.sum(1), E, rtol=1e-12, atol=1e-12)
    return T


def job(payload):
    uid, prov, variant = payload
    import common as C
    import report_metrics as RM
    raw, ens = C.inputs(uid)
    c1, c3, r = raw.T
    E = c1.reshape(365, 24).sum(1)
    ev = (c3.reshape(365, 24) > 1e-10).any(1)
    dt = 'holiday' if variant.endswith('holiday') else 'workday'
    T = allocate(E, SHAPES[(prov, dt)], variant.replace('_holiday', ''))
    new = np.c_[T.ravel(), (T * ev[:, None]).ravel(), r]
    arr = C.transform(new)
    sol = {cid: C.solve(arr, cid) for cid in ('li_lfp_4h', 'caes_24h')}
    lf, ca = sol['li_lfp_4h'], sol['caes_24h']
    B = lf['cost_usd_kw_year'] - ca['cost_usd_kw_year']
    row = dict(uid=uid, province=prov, variant=variant, B=B, roi_2035=float(RM.roi(np.array([B + RM.PREM36[2035]]), 2035)[0]),
               chargeable_hours=int((arr[:, 2] > 1e-9).sum()), task_hours=int((arr[:, 0] > 1e-10).sum()),
               peak_task_per_P=float(arr[:, 0].max()), efc_acaes=ca['tail_discharge_mwh_per_mw'] / 24,
               efc_lfp=lf['tail_discharge_mwh_per_mw'] / 4)
    for k in C.COMP:
        row[f'B_{k}'] = lf[f'{k}_usd_kw_year'] - ca[f'{k}_usd_kw_year']
    return row


def init():
    global SHAPES
    SHAPES = shapes()


if __name__ == '__main__':
    from cpu_budget import limit_workers
    SHAPES = shapes()
    rec = pd.read_csv(REV / 'remaining_experiments_20261103/inputs/records.csv', dtype={'uid': str})
    rec = rec[rec.group == 'provincial']
    rec['province'] = rec.system_id.str.split('-', n=1).str[1]
    assert set(rec.province) == set(GRID) and len(rec) == 26
    # peak hours of the typical curves
    rows = []
    for prov in GRID:
        for dt in ('workday', 'holiday'):
            v = SHAPES[(prov, dt)]; o = np.argsort(-v, kind='stable')
            rows.append(dict(province=prov, day_type=dt, top1=o[0], top4=' '.join(map(str, sorted(o[:4]))),
                             top8=' '.join(map(str, sorted(o[:8]))), **{f'h{h:02d}': v[h] for h in range(24)}))
    pd.DataFrame(rows).to_csv(OUT / 'peak_hours.csv', index=False)
    tasks = [(u, p, v) for u, p in zip(rec.uid, rec.province) for v in VARIANTS]
    with mp.get_context('spawn').Pool(limit_workers(52), initializer=init) as pool:
        res = pool.map(job, tasks, chunksize=1)
    d = pd.DataFrame(res)
    ref = pd.read_csv(REV / 'remaining_experiments_20261103/tables/annual_B_decomposition.csv', dtype={'uid': str})
    ref = ref[(ref.kind == 'mechanism') & (ref.variant == 'real')].set_index('uid').B
    f = d[d.variant == 'flat24'].set_index('uid').B
    err = float((f - ref.reindex(f.index)).abs().max())
    assert err < 1e-6, err
    d.to_csv(OUT / 'intraday_records.csv', index=False)
    s = d.groupby('variant').agg(n=('uid', 'size'), hybrid_cheaper=('B', lambda b: int((b >= 0).sum())), B_mean=('B', 'mean'),
                                 B_median=('B', 'median'), roi_median=('roi_2035', 'median'), roi_min=('roi_2035', 'min'),
                                 B_residual_energy=('B_residual_energy', 'mean'), B_residual_peak=('B_residual_peak', 'mean'),
                                 chargeable_hours=('chargeable_hours', 'median'), task_hours=('task_hours', 'median'),
                                 efc_acaes=('efc_acaes', 'median'), efc_lfp=('efc_lfp', 'median')).reindex(VARIANTS)
    s.to_csv(OUT / 'intraday_summary.csv')
    (OUT / 'qa.json').write_text(json.dumps(dict(flat24_max_abs_diff_vs_delivered=err, records=26, variants=VARIANTS), indent=1))
    pd.set_option('display.width', 250)
    print(s.round(2).to_string())
