"""Precompute exact aggregated views for redesigned Main Figures 1, 4 and 5 across all 11,054 records.

Memory-bounded (< 1.5 GB RSS) using chunked memmap reads for Japan (10,612 x 8,760).
Outputs deterministic CSV tables in `figures/source_data/`.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D
import style as S

HOURS_OF_YEAR = np.tile(np.arange(24), 365)
EU_CAL = pd.date_range('2019-01-01', periods=8760, freq='h')
EU_MONTHS = EU_CAL.month.to_numpy()
CN_MONTHS_H = np.repeat(D._CN_MONTH, 24)


def compute_fig1b_and_1d():
    """Compute:
    1) figure1b_month_hour_regimes.csv: 12 x 24 diurnal-seasonal probabilities of peaking task,
       high-demand event task, and chargeable surplus across China (pos & all), Europe, and Japan.
    2) figure1d_window_exceedance.csv: Mean & median annual windows per record exceeding fill fraction
       tau = E_window / E_usable for 4 h LFP (usable 3.6) and 24 h A-CAES (usable 21.6).
    """
    taus = np.geomspace(0.05, 4.0, 65)
    # Include exact 1.00 threshold
    taus = np.unique(np.sort(np.r_[taus, 0.25, 0.5, 1.0, 2.0]))

    mh_rows = []
    win_rows = []

    def _accum_mh(group_tag, task_cnt, event_cnt, chg_cnt, n_rec, months_arr):
        for mo in range(1, 13):
            for hr in range(24):
                mh_rows.append(dict(
                    regime=group_tag,
                    n_records=int(n_rec),
                    month=int(mo),
                    hour=int(hr),
                    task_prob=float(task_cnt[mo - 1, hr] / n_rec),
                    event_prob=float(event_cnt[mo - 1, hr] / n_rec),
                    chargeable_prob=float(chg_cnt[mo - 1, hr] / n_rec),
                ))

    def _window_curves_from_arrays(group_tag, lfp_ratios_per_rec, caes_ratios_per_rec):
        n_rec = len(lfp_ratios_per_rec)
        lfp_mat = np.zeros((n_rec, len(taus)), dtype=np.float64)
        caes_mat = np.zeros((n_rec, len(taus)), dtype=np.float64)
        for i in range(n_rec):
            lr = lfp_ratios_per_rec[i]
            cr = caes_ratios_per_rec[i]
            if len(lr):
                lfp_mat[i] = (lr[:, None] >= taus[None, :] - 1e-9).sum(axis=0)
            if len(cr):
                caes_mat[i] = (cr[:, None] >= taus[None, :] - 1e-9).sum(axis=0)
        for ti, tau in enumerate(taus):
            for tech, mat in (('4h_LFP', lfp_mat), ('24h_ACAES', caes_mat)):
                q25, med, q75 = np.quantile(mat[:, ti], [0.25, 0.50, 0.75])
                win_rows.append(dict(
                    regime=group_tag,
                    tech=tech,
                    n_records=int(n_rec),
                    fill_multiple=float(tau),
                    mean_windows=float(mat[:, ti].mean()),
                    q25_windows=float(q25),
                    median_windows=float(med),
                    q75_windows=float(q75),
                ))

    # --- 1. China (310 all & 178 positive-supply) ---
    cn_df = D._cn_full_df()
    pos_uids = D.cn_positive_uids()
    t_all = np.zeros((12, 24)); e_all = np.zeros((12, 24)); c_all = np.zeros((12, 24))
    t_pos = np.zeros((12, 24)); e_pos = np.zeros((12, 24)); c_pos = np.zeros((12, 24))
    lfp_all, caes_all, lfp_pos, caes_pos = [], [], [], []

    for uid in cn_df.uid:
        raw, flags, _, _ = D.cn_raw_and_flags(uid)
        arr = D.transform(raw)
        tk = (arr[:, 0] > 1e-10).reshape(365, 24)
        ev = (arr[:, 1] > 1e-10).reshape(365, 24)
        ch = (arr[:, 2] > 1e-9).reshape(365, 24)
        for mo in range(1, 13):
            m = D._CN_MONTH == mo
            dm = float(m.sum())
            t_all[mo - 1] += tk[m].sum(0) / dm
            e_all[mo - 1] += ev[m].sum(0) / dm
            c_all[mo - 1] += ch[m].sum(0) / dm
            if uid in pos_uids:
                t_pos[mo - 1] += tk[m].sum(0) / dm
                e_pos[mo - 1] += ev[m].sum(0) / dm
                c_pos[mo - 1] += ch[m].sum(0) / dm

        is_call, run, _ = D.WD.segments(arr)
        call = arr[:, 1] > 1e-10
        chg_w = np.bincount(run, weights=(arr[:, 2] > 1e-9).astype(float))
        w_ok = (~is_call) & (chg_w > 0)
        el = (np.bincount(run, weights=np.where(call, 0., np.minimum(arr[:, 2], 1.0))) * 0.90)[w_ok] / 3.6
        ec = (np.bincount(run, weights=np.where(call, 0., np.minimum(arr[:, 2], 1.5))) * 0.55)[w_ok] / 21.6
        lfp_all.append(el); caes_all.append(ec)
        if uid in pos_uids:
            lfp_pos.append(el); caes_pos.append(ec)

    _accum_mh('provincial_all', t_all, e_all, c_all, len(cn_df), D._CN_MONTH)
    _accum_mh('provincial_pos', t_pos, e_pos, c_pos, len(pos_uids), D._CN_MONTH)
    _window_curves_from_arrays('provincial_all', lfp_all, caes_all)
    _window_curves_from_arrays('provincial_pos', lfp_pos, caes_pos)

    # --- 2. Europe (132 country-years) ---
    rec = D.records()
    eu_uids = rec[rec.group == 'national'].uid.tolist()
    t_eu = np.zeros((12, 24)); e_eu = np.zeros((12, 24)); c_eu = np.zeros((12, 24))
    lfp_eu, caes_eu = [], []
    for uid in eu_uids:
        arr, _ = D.hourly(uid)
        tk = (arr[:, 0] > 1e-10).reshape(365, 24)
        ev = (arr[:, 1] > 1e-10).reshape(365, 24)
        ch = (arr[:, 2] > 1e-9).reshape(365, 24)
        for mo in range(1, 13):
            m = D._CN_MONTH == mo
            dm = float(m.sum())
            t_eu[mo - 1] += tk[m].sum(0) / dm
            e_eu[mo - 1] += ev[m].sum(0) / dm
            c_eu[mo - 1] += ch[m].sum(0) / dm

        is_call, run, _ = D.WD.segments(arr)
        call = arr[:, 1] > 1e-10
        chg_w = np.bincount(run, weights=(arr[:, 2] > 1e-9).astype(float))
        w_ok = (~is_call) & (chg_w > 0)
        el = (np.bincount(run, weights=np.where(call, 0., np.minimum(arr[:, 2], 1.0))) * 0.90)[w_ok] / 3.6
        ec = (np.bincount(run, weights=np.where(call, 0., np.minimum(arr[:, 2], 1.5))) * 0.55)[w_ok] / 21.6
        lfp_eu.append(el); caes_eu.append(ec)

    _accum_mh('national', t_eu, e_eu, c_eu, len(eu_uids), D._CN_MONTH)
    _window_curves_from_arrays('national', lfp_eu, caes_eu)

    # --- 3. Japan (10,612 community-years, chunked memmap) ---
    jp_df = D._jp_full_df()
    h_mm, cal_m, seg = D._jp_mmap()
    day_months = cal_m.reshape(365, 24)[:, 0]
    t_jp = np.zeros((12, 24)); e_jp = np.zeros((12, 24)); c_jp = np.zeros((12, 24))
    lfp_jp, caes_jp = [], []

    n_jp = len(jp_df)
    month_masks_d = {mo: (day_months == mo) for mo in range(1, 13)}
    for s0 in range(0, n_jp, 500):
        s1 = min(n_jp, s0 + 500)
        imp_blk = np.asarray(h_mm[s0:s1, :, 0], dtype=np.float64)
        rev_blk = np.asarray(h_mm[s0:s1, :, 1], dtype=np.float64)
        P_vec = 0.01 * imp_blk.mean(axis=1, keepdims=True)

        imp_d = imp_blk.reshape(-1, 365, 24)
        rev_d = rev_blk.reshape(-1, 365, 24)
        task_d = np.empty_like(imp_d)
        for mo, mm in month_masks_d.items():
            q50 = np.quantile(imp_d[:, mm, :], 0.50, axis=1, keepdims=True)
            task_d[:, mm, :] = np.maximum(imp_d[:, mm, :] - q50, 0.0)

        day_sum = task_d.sum(axis=2)
        flags_b = np.zeros((s1 - s0, 365), dtype=bool)
        for s_id in range(4):
            ms = seg == s_id
            q75 = np.quantile(day_sum[:, ms], 0.75, axis=1, keepdims=True)
            flags_b[:, ms] = day_sum[:, ms] >= q75

        c3_d = np.where(flags_b[:, :, None], task_d, 0.0)
        chg_d = np.where(task_d > 1e-10, 0.0, rev_d) / P_vec[:, None]

        tk_b = task_d > 1e-10
        ev_b = c3_d > 1e-10
        ch_b = chg_d > 1e-9

        for mo, mm in month_masks_d.items():
            dm = float(mm.sum())
            t_jp[mo - 1] += tk_b[:, mm, :].sum(axis=(0, 1)) / dm
            e_jp[mo - 1] += ev_b[:, mm, :].sum(axis=(0, 1)) / dm
            c_jp[mo - 1] += ch_b[:, mm, :].sum(axis=(0, 1)) / dm

        for bi in range(s1 - s0):
            c1_f = (task_d[bi] / P_vec[bi, 0]).ravel()
            c3_f = (c3_d[bi] / P_vec[bi, 0]).ravel()
            r_f = chg_d[bi].ravel()
            arr = np.c_[c1_f, c3_f, r_f]
            is_call, run, _ = D.WD.segments(arr)
            call = c3_f > 1e-10
            chg_w = np.bincount(run, weights=(r_f > 1e-9).astype(float))
            w_ok = (~is_call) & (chg_w > 0)
            el = (np.bincount(run, weights=np.where(call, 0., np.minimum(r_f, 1.0))) * 0.90)[w_ok] / 3.6
            ec = (np.bincount(run, weights=np.where(call, 0., np.minimum(r_f, 1.5))) * 0.55)[w_ok] / 21.6
            lfp_jp.append(el); caes_jp.append(ec)

    _accum_mh('community', t_jp, e_jp, c_jp, n_jp, day_months)
    _window_curves_from_arrays('community', lfp_jp, caes_jp)

    mh_df = pd.DataFrame(mh_rows)
    win_df = pd.DataFrame(win_rows)
    S.SD.mkdir(parents=True, exist_ok=True)
    mh_df.to_csv(S.SD / 'figure1b_month_hour_regimes.csv', index=False)
    win_df.to_csv(S.SD / 'figure1d_window_exceedance.csv', index=False)

    # Sanity check exact medians at fill_multiple == 1.0 (using 0.90 single-layer LFP RTE and 0.55 A-CAES RTE)
    at1 = win_df[np.isclose(win_df.fill_multiple, 1.0)].set_index(['regime', 'tech'])['median_windows']
    print('Sanity @ 1.0x usable capacity:', at1.to_dict())
    assert np.isclose(at1[('provincial_pos', '24h_ACAES')], 17.0)
    assert np.isclose(at1[('national', '24h_ACAES')], 24.5)
    assert np.isclose(at1[('community', '24h_ACAES')], 25.0)
    assert np.isclose(at1[('national', '4h_LFP')], 39.0)
    assert np.isclose(at1[('community', '4h_LFP')], 62.0)


def compute_fig5d_carpet_and_5f_bounds():
    """Compute:
    1) figure5d_carpet_58x150.csv: 58 configurations x 50 percentile bins per regime (150 cols),
       sorted by each record's 24 h A-CAES B, summarizing all 641,132 LP optimizations.
    2) figure5f_within_tech_bounds.csv: Exact lower-bound, interior, and upper-bound optimal-duration
       record counts and shares across all 8 technologies and 3 regimes.
    """
    cat = D.catalogue_records()
    assert len(cat) == 641132

    # Sort UIDs within each regime by 24 h A-CAES B (tie-broken by uid)
    caes24 = cat[(cat.tech == 'caes') & (cat.duration_h == 24)].sort_values(['group', 'B', 'uid'])
    n_bins = 50
    uid_bin = {}
    for g in S.MAIN:
        sub = caes24[caes24.group == g]
        bins = np.floor(np.linspace(0, n_bins - 1e-9, len(sub))).astype(int)
        for u, b in zip(sub.uid, bins):
            uid_bin[u] = int(b)

    cat['pbin'] = cat['uid'].map(uid_bin)
    agg = cat.groupby(['tech', 'duration_h', 'group', 'pbin'], observed=True)['B'].mean().reset_index()
    agg.to_csv(S.SD / 'figure5d_carpet_58x150.csv', index=False)
    print('Saved figure5d_carpet_58x150.csv, rows:', len(agg))

    # Within-tech optimal duration per (uid, tech): tie-break toward shortest duration (standard convention in analyse_r5)
    cat_sorted = cat.sort_values(['uid', 'tech', 'B', 'duration_h'], ascending=[True, True, False, True])
    best = cat_sorted.drop_duplicates(['uid', 'tech'], keep='first')
    d_bounds = cat.groupby('tech')['duration_h'].agg(h_min='min', h_max='max').reset_index()
    best = best.merge(d_bounds, on='tech', how='left')
    best['loc_class'] = np.where(
        best.duration_h == best.h_min, 'lower',
        np.where(best.duration_h == best.h_max, 'upper', 'interior')
    )
    b_rows = []
    for tech in ['li_lfp', 'vrfb', 'fecr_flow', 'gravity', 'molten_salt_tes', 'caes', 'laes', 'h2_tank']:
        bt = best[best.tech == tech]
        hmin, hmax = float(bt.h_min.iloc[0]), float(bt.h_max.iloc[0])
        for g in [*S.MAIN, 'all_pooled']:
            sub = bt if g == 'all_pooled' else bt[bt.group == g]
            n = len(sub)
            n_lo = int((sub.loc_class == 'lower').sum())
            n_in = int((sub.loc_class == 'interior').sum())
            n_up = int((sub.loc_class == 'upper').sum())
            b_rows.append(dict(
                tech=tech, group=g, n=n, h_min=hmin, h_max=hmax,
                lower_count=n_lo, interior_count=n_in, upper_count=n_up,
                lower_pct=100.0 * n_lo / n, interior_pct=100.0 * n_in / n, upper_pct=100.0 * n_up / n,
            ))
    bdf = pd.DataFrame(b_rows)
    bdf.to_csv(S.SD / 'figure5f_within_tech_bounds.csv', index=False)
    print('Sanity Fig 5f pooled:\n', bdf[bdf.group == 'all_pooled'][['tech', 'lower_count', 'interior_count', 'upper_count', 'lower_pct', 'upper_pct']])
    assert int(bdf[(bdf.tech == 'caes') & (bdf.group == 'all_pooled')].upper_count.iloc[0]) == 10863
    assert int(bdf[(bdf.tech == 'h2_tank') & (bdf.group == 'all_pooled')].lower_count.iloc[0]) == 10962


if __name__ == '__main__':
    compute_fig1b_and_1d()
    compute_fig5d_carpet_and_5f_bounds()
