"""Parallelized (16-worker) exact COPT robustness verifications for the Joule manuscript polish:
1. Japan 5-prefecture (n=10, hundred-MW scale) and Kanto-Shizuoka regional (n=2, GW scale) aggregation:
   historical dates, 1,000 seasonal MC rearrangements, 1-12 day event extensions, and 8-tech 58-config catalogue.
2. Service-rule & net-load robustness (`registered`, `all_task`, `netted_tail`, `netted_all`) under 1-12 day
   event extensions across Europe (n=132), China positive curtailment (n=178), and Japan prefectures (n=10) +
   stratified 265 community records.
3. Penetration-depth transition (`P_add = 1%, 2%, 5%, 10%, 20%, 30%, 50%` of mean reference load/gas) across
   Europe (n=132) to quantify Delta C_peak activation bridging 1% marginal slice and S5 capacity planning.
"""
from __future__ import annotations
from pathlib import Path
import multiprocessing as mp
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path('/data2/cxw/Decarbonizing-Fossil-Peaking-with-Battery-Storage')
V86 = ROOT / 'submission/Nature_Energy_v86'
OUT_DIR = V86 / 'revision/joule_robustness_checks_20261124'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(V86 / 'revision/japan_5309_audit_20261117/code'))
sys.path.insert(0, str(V86 / 'figures/code'))
import run_full_10612_suite as J
import data as D

BASE_CFG = J._CAT58['li_lfp_4h']
CAES_CFG = J._CAT58['caes_24h']
SCEN = J.BASE_SCENARIO_EUR
PREM36 = float(J.RM.PREM36[2035])


def _worker_init():
    J._init_worker()


def eval_pair(arr: np.ndarray, prev_order: np.ndarray | None = None):
    cidx = J.CollapsedIndexer(arr)
    mL = J.PersistentBranchLP(cidx, BASE_CFG, BASE_CFG, SCEN)
    mC = J.PersistentBranchLP(cidx, BASE_CFG, CAES_CFG, SCEN)
    try:
        if prev_order is not None:
            mL.rewire(prev_order)
            mC.rewire(prev_order)
        sL = mL.full_solution()
        sC = mC.full_solution()
        B = sL['cost_usd_kw_year'] - sC['cost_usd_kw_year']
        d_peak = sL['residual_peak_usd_kw_year'] - sC['residual_peak_usd_kw_year']
        d_re = sL['residual_energy_usd_kw_year'] - sC['residual_energy_usd_kw_year']
        efc_c = sC['tail_discharge_mwh_per_mw'] / 24.0
        efc_l = sL['tail_discharge_mwh_per_mw'] / 4.0
        roi = float(J.RM.roi(B + PREM36, 2035))
        return dict(B=B, d_peak=d_peak, d_re=d_re, efc_c=efc_c, efc_l=efc_l, roi=roi)
    finally:
        mL.dispose()
        mC.dispose()


def _solve_jp_agg_item(task):
    level, label, pref, fy, ridxs = task
    sub = np.asarray(J._HOURLY[ridxs], dtype=np.float64)
    imp = sub[:, :, 0].sum(axis=0)
    rev = sub[:, :, 1].sum(axis=0)
    mean_imp_mw = float(imp.mean())
    peak_imp_mw = float(imp.max())
    P = 0.01 * mean_imp_mw

    base_q = np.empty(J.N_HOURS, dtype=np.float64)
    for mo in range(1, 13):
        m = J._CAL_M == mo
        blk = imp[m].reshape(-1, 24)
        base_q[m] = np.tile(np.quantile(blk, 0.50, axis=0), blk.shape[0])
    c1 = np.maximum(imp - base_q, 0.0)
    day = c1.reshape(J.N_DAYS, 24).sum(1)
    q75 = np.empty(J.N_DAYS, dtype=np.float64)
    for s in range(4):
        ms = J._SEG == s
        q75[ms] = np.quantile(day[ms], 0.75)
    flags = day >= q75
    c3 = np.where(np.repeat(flags, 24), c1, 0.0)
    raw = np.c_[c1, c3, rev] / P
    arr = J._transform_rule(raw, 'registered')

    uk = f'JapanAgg:{label}'
    ens = J.ce.ChronologyEnsemble(pd.DataFrame({'segment': J._SEG, 'event': flags}, index=np.arange(J.N_DAYS)), uk)
    segs_fast = J._precompute_obs_segments(J._SEG, flags)

    cidx = J.CollapsedIndexer(arr)
    mL = J.PersistentBranchLP(cidx, BASE_CFG, BASE_CFG, SCEN)
    mC = J.PersistentBranchLP(cidx, BASE_CFG, CAES_CFG, SCEN)
    ext_rows = []
    try:
        sL = mL.full_solution()
        sC = mC.full_solution()
        B_real = sL['cost_usd_kw_year'] - sC['cost_usd_kw_year']
        roi_real = float(J.RM.roi(B_real + PREM36, 2035))
        lw_real = J._long_windows_fast(arr)
        ev_runs = J._runs(flags)
        mean_dur = float(flags.sum() / len(ev_runs))

        b_mc = np.empty(1000, dtype=np.float64)
        lw_mc = np.empty(1000, dtype=np.int32)
        roi_mc = np.empty(1000, dtype=np.float64)
        arr_days = arr.reshape(365, 24, 3)
        for seed in range(1000):
            perm = J._fast_obs_random_order(uk, segs_fast, seed)
            prev = J._prev_from_order(perm)
            mL.rewire(prev)
            mC.rewire(prev)
            bm = mL.solve_cost_usd() - mC.solve_cost_usd()
            b_mc[seed] = bm
            roi_mc[seed] = float(J.RM.roi(bm + PREM36, 2035))
            lw_mc[seed] = J._long_windows_fast(arr_days[perm].reshape(8760, 3))

        mL.rewire(J.INIT_PREV)
        mC.rewire(J.INIT_PREV)
        for tgt in range(1, 13):
            ord_t = np.asarray(ens.orders(tgt)['clustered'], dtype=np.int32)
            prev_t = J._prev_from_order(ord_t)
            mL.rewire(prev_t)
            mC.rewire(prev_t)
            sLt = mL.full_solution()
            sCt = mC.full_solution()
            Bt = sLt['cost_usd_kw_year'] - sCt['cost_usd_kw_year']
            fl_t = flags[ord_t]
            md_t = float(fl_t.sum() / len(J._runs(fl_t)))
            ext_rows.append(dict(
                level=level, label=label, target_days=tgt, mean_dur=md_t,
                B=Bt, dB_vs_real=Bt - B_real, efc_acaes=sCt['tail_discharge_mwh_per_mw'] / 24.0,
            ))

        mL.rewire(J.INIT_PREV)
        cost_L = sL['cost_usd_kw_year']
        best_cid, best_tid, best_dur, best_B = 'li_lfp_4h', 'li_lfp', 4.0, 0.0
        for tid, cfg_list in J._TECH_GROUPS.items():
            mT = J.PersistentBranchLP(cidx, BASE_CFG, cfg_list[0][1], SCEN)
            try:
                for cid, cfg in cfg_list:
                    mT.set_tail_cfg(cfg)
                    bv = 0.0 if cid == 'li_lfp_4h' else (cost_L - mT.solve_cost_usd())
                    if bv > best_B + 1e-9:
                        best_cid, best_tid, best_dur, best_B = cid, tid, float(cfg['duration_h']), bv
            finally:
                mT.dispose()

        arr_net = J._transform_rule(raw, 'netted_all')
        res_net = eval_pair(arr_net)
        ord12 = np.asarray(ens.orders(12)['clustered'], dtype=np.int32)
        res_net12 = eval_pair(arr_net, J._prev_from_order(ord12))
        ord1 = np.asarray(ens.orders(1)['clustered'], dtype=np.int32)
        res_net1 = eval_pair(arr_net, J._prev_from_order(ord1))

        row = dict(
            level=level, label=label, prefecture=pref, fy=fy, n_communities=len(ridxs),
            mean_imp_mw=mean_imp_mw, peak_imp_mw=peak_imp_mw, P_ref_mw=P,
            mean_dur_real=mean_dur, charge_hours=int((arr[:, 2] > 1e-9).sum()),
            long_win_real=lw_real, long_win_mc_med=float(np.median(lw_mc)),
            efc_acaes_real=sC['tail_discharge_mwh_per_mw'] / 24.0,
            efc_lfp_real=sL['tail_discharge_mwh_per_mw'] / 4.0,
            B_real=B_real, win_real=bool(B_real >= 0.0),
            win_prob_mc=float((b_mc >= 0.0).mean()), B_mc_med=float(np.median(b_mc)),
            roi_real=roi_real, roi_mc_med=float(np.median(roi_mc)),
            roi_pctile_mc=float(((roi_mc < roi_real - 1e-8).sum() + 0.5 * (np.abs(roi_mc - roi_real) <= 1e-8).sum()) / 1000.0),
            best_58_cid=best_cid,
            B_netted_all_real=res_net['B'], B_netted_all_1d=res_net1['B'], B_netted_all_12d=res_net12['B'],
            dB_netted_all_12d_vs_real=res_net12['B'] - res_net['B'],
            dB_netted_all_12d_vs_1d=res_net12['B'] - res_net1['B'],
        )
        return row, ext_rows
    finally:
        mL.dispose()
        mC.dispose()


def _solve_rule_record(item):
    group, uid, uk, raw, flags, seg, rules = item
    ens = J.ce.ChronologyEnsemble(pd.DataFrame({'segment': seg, 'event': flags}, index=np.arange(365)), uk)
    ords = {t: J._prev_from_order(np.asarray(ens.orders(t)['clustered'], dtype=np.int32)) for t in (1, 2, 3, 4, 12)}
    out = []
    for rule in rules:
        arr = J._transform_rule(raw, rule)
        cidx = J.CollapsedIndexer(arr)
        mL = J.PersistentBranchLP(cidx, BASE_CFG, BASE_CFG, SCEN)
        mC = J.PersistentBranchLP(cidx, BASE_CFG, CAES_CFG, SCEN)
        try:
            sL0 = mL.full_solution()
            sC0 = mC.full_solution()
            B0 = sL0['cost_usd_kw_year'] - sC0['cost_usd_kw_year']
            efc0 = sC0['tail_discharge_mwh_per_mw'] / 24.0
            out.append(dict(group=group, uid=uid, rule=rule, tgt=0, B=B0, dB_real=0.0, dB_1d=np.nan, efc_c=efc0))
            B1 = None
            for t, prev_t in ords.items():
                mL.rewire(prev_t)
                mC.rewire(prev_t)
                sLt = mL.full_solution()
                sCt = mC.full_solution()
                Bt = sLt['cost_usd_kw_year'] - sCt['cost_usd_kw_year']
                efct = sCt['tail_discharge_mwh_per_mw'] / 24.0
                if t == 1:
                    B1 = Bt
                out.append(dict(group=group, uid=uid, rule=rule, tgt=t, B=Bt, dB_real=Bt - B0, dB_1d=Bt - B1, efc_c=efct))
        finally:
            mL.dispose()
            mC.dispose()
    return out


def _solve_pen_record(item):
    uid, uk, raw, flags, seg = item
    ens = J.ce.ChronologyEnsemble(pd.DataFrame({'segment': seg, 'event': flags}, index=np.arange(365)), uk)
    prev1 = J._prev_from_order(np.asarray(ens.orders(1)['clustered'], dtype=np.int32))
    prev4 = J._prev_from_order(np.asarray(ens.orders(4)['clustered'], dtype=np.int32))
    out = []
    for s_pct in (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0):
        arr_s = J._transform_rule(raw / s_pct, 'registered')
        res = eval_pair(arr_s)
        arr_all = J._transform_rule(raw / s_pct, 'all_task')
        res_all = eval_pair(arr_all)
        res_all_1d = eval_pair(arr_all, prev1)
        res_all_4d = eval_pair(arr_all, prev4)
        out.append(dict(
            group='national', uid=uid, scale_pct=s_pct,
            B_reg=res['B'], d_peak_reg=res['d_peak'], d_re_reg=res['d_re'], efc_c_reg=res['efc_c'],
            B_all=res_all['B'], d_peak_all=res_all['d_peak'],
            B_all_1d=res_all_1d['B'], d_peak_all_1d=res_all_1d['d_peak'],
            B_all_4d=res_all_4d['B'], d_peak_all_4d=res_all_4d['d_peak'],
        ))
    return out


if __name__ == '__main__':
    t0 = time.monotonic()
    meta = J._META
    jp_tasks = []
    for (pref, fy), g in meta.groupby(['prefecture', 'observation_year'], sort=True):
        jp_tasks.append(('prefecture', f'{pref}_{fy}', str(pref), int(fy), g['row_idx'].to_numpy(dtype=int)))
    for fy, g in meta.groupby('observation_year', sort=True):
        jp_tasks.append(('region_kanto_shizuoka', f'KantoShizuoka_{fy}', 'ALL_5_PREF', int(fy), g['row_idx'].to_numpy(dtype=int)))

    recs = D.rd(D.INP / 'records.csv')
    eu = recs[recs.group == 'national'].copy()
    eu_items = []
    for r in eu.itertuples():
        z = np.load(D.INP / f'{r.uid}.npz')
        eu_items.append((r.uid, str(r.system_id), z['raw'].copy(), z['event'].astype(bool), z['segment'].astype(int)))

    cn_pos_uids = sorted(D.cn_positive_uids())
    cn_items = []
    for uid in cn_pos_uids:
        raw, flags, seg, uk = D.cn_raw_and_flags(uid)
        cn_items.append(('provincial_pos', uid, uk, raw, flags, seg, ('registered', 'all_task', 'netted_all')))

    # Load 265 stratified JP community records
    J._init_worker()
    jp_sample_items = []
    for ridx in range(0, len(meta), 40):
        r, uid, uk, imp, rev, P, c1, flags, raw, _ = J._build_record_arrays(ridx)
        jp_sample_items.append(('community_sample266', uid, uk, raw, flags, J._SEG, ('registered', 'netted_tail', 'netted_all')))

    rule_tasks = [('national', u, uk, raw, fl, sg, ('registered', 'netted_tail', 'all_task', 'netted_all')) for u, uk, raw, fl, sg in eu_items] + cn_items + jp_sample_items

    with mp.get_context('spawn').Pool(16, initializer=_worker_init) as pool:
        jp_res = pool.map(_solve_jp_agg_item, jp_tasks)
        print(f'[1] Japan 12 aggregated records (12,000 MC + 58-cat + 1-12d) done in {time.monotonic()-t0:.1f}s', flush=True)
        t1 = time.monotonic()
        rule_res = pool.map(_solve_rule_record, rule_tasks)
        print(f'[2] Service-rule extension ({len(rule_tasks)} records) done in {time.monotonic()-t1:.1f}s', flush=True)
        t2 = time.monotonic()
        pen_res = pool.map(_solve_pen_record, eu_items)
        print(f'[3] Europe penetration transition (132 x 7 depths) done in {time.monotonic()-t2:.1f}s', flush=True)

    df_agg = pd.DataFrame([r for r, _ in jp_res])
    df_ext = pd.DataFrame([x for _, ex in jp_res for x in ex])
    df_rule_ext = pd.DataFrame([x for sub in rule_res for x in sub])
    df_pen = pd.DataFrame([x for sub in pen_res for x in sub])

    df_agg.to_csv(OUT_DIR / 'japan_prefecture_regional_aggregation.csv', index=False)
    df_ext.to_csv(OUT_DIR / 'japan_prefecture_regional_extension_1_12d.csv', index=False)
    df_rule_ext.to_csv(OUT_DIR / 'service_rule_extension_robustness.csv', index=False)
    df_pen.to_csv(OUT_DIR / 'europe_penetration_depth_transition.csv', index=False)

    print('\n=== SUMMARY 1: JAPAN PREFECTURE (n=10) & REGIONAL (n=2) AGGREGATION ===', flush=True)
    for lvl, g in df_agg.groupby('level'):
        print(f'-- {lvl} (n={len(g)}) --')
        print(g[['label', 'n_communities', 'mean_imp_mw', 'peak_imp_mw', 'P_ref_mw', 'mean_dur_real', 'long_win_real', 'long_win_mc_med',
                 'efc_acaes_real', 'efc_lfp_real', 'B_real', 'win_real', 'win_prob_mc', 'roi_real', 'roi_mc_med', 'best_58_cid',
                 'B_netted_all_real', 'dB_netted_all_12d_vs_real', 'dB_netted_all_12d_vs_1d']].to_string(index=False))

    print('\n-- Japan Prefecture (n=10) 1..12d extension medians --')
    p_ext = df_ext[df_ext.level == 'prefecture'].groupby('target_days').agg(
        med_dur=('mean_dur', 'median'), med_B=('B', 'median'),
        med_dB=('dB_vs_real', 'median'), win_rate=('B', lambda s: (s >= 0).mean()),
        med_efc=('efc_acaes', 'median'),
    )
    print(p_ext.to_string())

    print('\n=== SUMMARY 2: SERVICE RULE & NETTED-ALL EXTENSION ROBUSTNESS ===')
    for (grp, rule), g in df_rule_ext.groupby(['group', 'rule']):
        g0 = g[g.tgt == 0].set_index('uid')
        g4 = g[g.tgt == 4].set_index('uid')
        g12 = g[g.tgt == 12].set_index('uid')
        g1 = g[g.tgt == 1].set_index('uid')
        dB_12_real = g12['B'] - g0['B']
        dB_12_1 = g12['B'] - g1['B']
        defc_12_real = g12['efc_c'] - g0['efc_c']
        print(f'{grp:20s} | {rule:12s} | real win={(g0["B"]>=0).mean()*100:5.1f}% (med B={g0["B"].median():6.2f}) '
              f'| 4d med dB_real={(g4["B"]-g0["B"]).median():6.2f} '
              f'| 12d med dB_real={dB_12_real.median():6.2f} (neg={(dB_12_real<0).mean()*100:5.1f}%) '
              f'| 12d med dB_vs_1d={dB_12_1.median():6.2f} (neg={(dB_12_1<0).mean()*100:5.1f}%) '
              f'| 12d med dEFC={defc_12_real.median():6.2f}')

    print('\n=== SUMMARY 3: EUROPE PENETRATION DEPTH TRANSITION (1% -> 50%) ===')
    ps = df_pen.groupby('scale_pct').agg(
        reg_win_pct=('B_reg', lambda s: (s >= 0).mean() * 100),
        reg_peak_gt05_pct=('d_peak_reg', lambda s: (s > 0.5).mean() * 100),
        reg_mean_dpeak=('d_peak_reg', 'mean'),
        all_win_pct=('B_all', lambda s: (s >= 0).mean() * 100),
        all_peak_gt05_pct=('d_peak_all', lambda s: (s > 0.5).mean() * 100),
        all_mean_dpeak=('d_peak_all', 'mean'),
        all_med_dB_4d_vs_1d=('B_all_4d', lambda s: np.median(s.values - df_pen.loc[s.index, 'B_all_1d'].values)),
        all_med_dpeak_4d_vs_1d=('d_peak_all_4d', lambda s: np.median(s.values - df_pen.loc[s.index, 'd_peak_all_1d'].values)),
    )
    print(ps.to_string(), flush=True)
