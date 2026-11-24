"""Inputs of the v86 main, extended-data and supplementary figures (11,054 main-text records + 107 external validation records).

Main-text contexts (11,054 complete 8,760-hour annual records):
  - Chinese provinces (n = 310): all 31 mainland provinces, 2015–2024 reconstructed hourly electricity load minus month x hour q50
    baseline, paired with proportional annual wind/solar curtailment (`../../revision/china_load_mc_2015_2024_20261118/results/q50_r4_r5_suite`)
  - European countries (n = 132): 15 countries, 2015–2024 (`Nature_Energy_v85/revision/remaining_experiments_20261103`)
  - Japanese communities (n = 10,612): all 5,306 dual-year complete 500-m communities across Saitama, Chiba, Tokyo,
    Kanagawa and Shizuoka in FY2023 and FY2024 (`../../revision/japan_5309_audit_20261117/results/full_10612`)
External validation contexts (107 records):
  - Europe historical weather years (n = 75, 1950/51–2024/25) and US operating regions (n = 32).
Legacy appendix context (26 records):
  - Chinese single-year gas-peaking provincial case (`../../analysis/provincial_prop/results`).
No optimisation is solved here.
"""
from pathlib import Path
from functools import lru_cache
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SUB = HERE.parents[2]
REV = SUB / 'Nature_Energy_v85/revision'
RC = REV / 'real_vs_placement_20261029/results/report_corrected'
RE = REV / 'remaining_experiments_20261103'
TAB = RE / 'tables'
INP = RE / 'inputs'
sys.path.insert(0, str(REV / 'real_vs_placement_20261029/code'))
sys.path.insert(0, str(SUB / 'Nature_Energy_v82/revision/issue6_window_saturation_20261019/scripts'))
import report_metrics as RM      # noqa: E402
import window_diagnostic as WD   # noqa: E402
V86 = HERE.parents[1]
PROV = V86 / 'analysis/provincial_prop/results'
CN_DIR = V86 / 'revision/china_load_mc_2015_2024_20261118'
CN_DATA = CN_DIR / 'inputs'
CN_SUITE = CN_DIR / 'results/q50_r4_r5_suite'
CN_SUITE_CLIP = CN_DIR / 'results/q50_r4_r5_suite_clipped'
EU_Q5_DIR = V86 / 'revision/europe_q5_mc_20261120'
JP_AUDIT = V86 / 'revision/japan_5309_audit_20261117'
JP_DATA = JP_AUDIT / 'data'
JP_F10 = JP_AUDIT / 'results/full_10612'
JP_MC_ALLQ = V86 / 'revision/japan_baseline_mc_10612_20261118/results'
sys.path.insert(0, str(V86 / 'analysis/provincial_intraday'))
import run_intraday as RI         # noqa: E402

GROUPS = ['provincial', 'national', 'community', 'hist', 'us']
# Example years of Fig. 1b: record of each main context closest to the group medians of chargeable hours and A-CAES EFC
# (for Chinese provinces, among the 178 province-years with positive curtailment budget)
EXAMPLES = {
    'provincial': 'afcbb34e84d5ec83',
    'national': '4fe14978e6589ab2',
    'community': '643a607c10783317',
}
MAIN = ['provincial', 'national', 'community']
RTE, RATIO, DUR = 0.55, 1.5, 24.0

_CN_DAYS = pd.date_range('2019-01-01', periods=365, freq='D')
_CN_MONTH = _CN_DAYS.month.to_numpy()
_CN_SEASON = (_CN_MONTH % 12) // 3


def rd(p, **kw):
    return pd.read_csv(p, dtype={'uid': str}, **kw)


def transform(raw):
    """Central rule: added storage serves C3, charges only in hours without task."""
    c1, c3, r = raw.T
    return np.c_[c1, c3, np.where(c1 > 1e-10, 0., r)]


@lru_cache(None)
def _province_of():
    """Legacy 26-province gas-peaking UID -> province map (used by Appendix S1 figure)."""
    m = rd(INP / 'records.csv')
    m = m[m.group == 'provincial']
    return dict(zip(m.uid, m.system_id.str.split('-', n=1).str[1]))


@lru_cache(None)
def _shapes():
    return RI.shapes()


@lru_cache(None)
def _cn_full_df():
    return rd(CN_SUITE / 'china_records.csv.gz').sort_values('row_idx').reset_index(drop=True)


@lru_cache(None)
def _cn_uid_map():
    df = _cn_full_df()
    return dict(zip(df.uid, df.row_idx.astype(int)))


@lru_cache(None)
def _cn_mmap():
    load = np.load(CN_DATA / 'china_2015_2024_load_mw.npy', mmap_mode='r')
    surplus = np.load(CN_DATA / 'china_yearly_surplus_mw.npy', mmap_mode='r')
    return load, surplus


def cn_positive_uids() -> set:
    df = _cn_full_df()
    return set(df.loc[df.raw_supply_mwh > 0, 'uid'])


def cn_raw_and_flags(uid: str):
    """Reconstruct exact (raw_norm, flags, seg, unit_key) for any of the 310 Chinese province-year records."""
    ridx = _cn_uid_map()[uid]
    load_mm, surp_mm = _cn_mmap()
    load = np.asarray(load_mm[ridx], dtype=np.float64)
    rev = np.asarray(surp_mm[ridx], dtype=np.float64)
    P = 0.01 * float(load.mean())
    blk = load.reshape(365, 24)
    task = np.empty_like(blk)
    for mo in range(1, 13):
        m = _CN_MONTH == mo
        task[m] = np.maximum(0.0, blk[m] - np.quantile(blk[m], 0.50, axis=0))
    day = task.sum(1)
    flags = np.zeros(365, dtype=bool)
    for s in range(4):
        ms = _CN_SEASON == s
        flags[ms] = day[ms] >= np.quantile(day[ms], 0.75)
    c1 = task.ravel()
    c3 = (task * flags[:, None]).ravel()
    raw = np.c_[c1, c3, rev] / P
    uk = _cn_full_df().iloc[ridx].unit_key
    return raw, flags, _CN_SEASON, str(uk)


@lru_cache(None)
def _jp_full_df():
    return rd(JP_F10 / 'japan_records.csv.gz').sort_values('row_idx').reset_index(drop=True)


@lru_cache(None)
def _jp_uid_map():
    df = _jp_full_df()
    return dict(zip(df.uid, df.row_idx.astype(int)))


@lru_cache(None)
def _jp_mmap():
    h = np.load(JP_DATA / 'japan_dual_hourly.npy', mmap_mode='r')
    cal = np.load(JP_DATA / 'japan_calendar.npz')
    return h, cal['cal_months'], cal['season_seg']


def jp_raw_and_flags(uid: str):
    """Reconstruct exact (raw_norm, flags, seg, unit_key) for any of the 10,612 Japanese records."""
    ridx = _jp_uid_map()[uid]
    h, cal_m, seg = _jp_mmap()
    imp = np.asarray(h[ridx, :, 0], dtype=np.float64)
    rev = np.asarray(h[ridx, :, 1], dtype=np.float64)
    P = 0.01 * float(imp.mean())
    base_q = np.empty(8760, dtype=np.float64)
    for mo in range(1, 13):
        m = cal_m == mo
        blk = imp[m].reshape(-1, 24)
        base_q[m] = np.tile(np.quantile(blk, 0.50, axis=0), blk.shape[0])
    c1 = np.maximum(imp - base_q, 0.0)
    day = c1.reshape(365, 24).sum(1)
    q75 = np.empty(365, dtype=np.float64)
    for s in range(4):
        ms = seg == s
        q75[ms] = np.quantile(day[ms], 0.75)
    flags = day >= q75
    c3 = np.where(np.repeat(flags, 24), c1, 0.0)
    raw = np.c_[c1, c3, rev] / P
    uk = _jp_full_df().iloc[ridx].unit_key
    return raw, flags, seg, str(uk)


def hourly(uid):
    if uid in _cn_uid_map():
        raw, ev, _, _ = cn_raw_and_flags(uid)
        return transform(raw), ev
    if uid in _jp_uid_map():
        raw, ev, _, _ = jp_raw_and_flags(uid)
        return transform(raw), ev
    z = np.load(INP / f'{uid}.npz')
    raw, ev = z['raw'].copy(), z['event'].astype(bool)
    prov = _province_of().get(uid)
    if prov is not None:
        daily = raw[:, 0].reshape(365, 24).sum(1)
        shape = _shapes()[(prov, 'workday')]
        task = daily[:, None] * (shape / shape.sum())[None, :]
        raw = np.c_[task.ravel(), (task * ev[:, None]).ravel(), raw[:, 2]]
    return transform(raw), ev


def runs(flags):
    out, n = [], 0
    for x in flags:
        if x:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def fill_windows(arr):
    """Charging-timing analysis of window_diagnostic.branch, also returning the hour at which each saturated window
    reaches the usable capacity."""
    is_call, run, hours = WD.segments(arr)
    h = WD.harvest(arr, RATIO, run)
    call = arr[:, 1] > 1e-10
    r = np.where(call, 0.0, np.minimum(arr[:, 2], RATIO))
    starts = np.r_[0, np.flatnonzero(np.diff(run)) + 1]
    E = WD.USABLE * DUR
    s = E
    for sweep in range(2):
        full_hours = []
        for k in range(len(h)):
            if is_call[k]:
                s -= min(s, float(hours[k]))
            elif h[k] > 1e-12:
                if s + RTE * h[k] >= E - 1e-12:
                    idx = starts[k] + np.arange(hours[k])
                    cum = s + RTE * np.cumsum(r[idx])
                    full_hours.append(int(idx[np.argmax(cum >= E - 1e-12)]))
                s = min(E, s + RTE * h[k])
    return full_hours


def equal_regime_spearman(df: pd.DataFrame, xcol: str, ycol: str, groups=MAIN):
    """Compute exact equal-regime-weighted rank correlation rho_eq, pooled rho_raw, and within-group Spearman rhos.

    Prevents the 10,612 Japanese community records from drowning out China (310) and Europe (132) in cross-scale summaries.
    """
    sub = df[df.group.isin(groups)].copy()
    sub['group'] = sub['group'].astype(str)
    counts = sub.groupby('group').size().to_dict()
    w = sub['group'].map({g: 1.0 / (len(groups) * counts[g]) for g in groups}).to_numpy(float)
    x = sub[xcol].to_numpy(float)
    y = sub[ycol].to_numpy(float)

    def _wrank(a, wt):
        order = np.argsort(a, kind='mergesort')
        asort, wsort = a[order], wt[order]
        uniq, idx_start, ucounts = np.unique(asort, return_index=True, return_counts=True)
        cdf = np.cumsum(wsort)
        r_out = np.empty_like(a, dtype=float)
        for s_i, c_i in zip(idx_start, ucounts):
            lo_w = 0.0 if s_i == 0 else cdf[s_i - 1]
            hi_w = cdf[s_i + c_i - 1]
            r_out[order[s_i:s_i + c_i]] = 0.5 * (lo_w + hi_w)
        return r_out

    rx, ry = _wrank(x, w), _wrank(y, w)
    mx, my = float(np.sum(w * rx)), float(np.sum(w * ry))
    cov = float(np.sum(w * (rx - mx) * (ry - my)))
    sdx = float(np.sqrt(np.sum(w * (rx - mx) ** 2)))
    sdy = float(np.sqrt(np.sum(w * (ry - my) ** 2)))
    rho_eq = cov / (sdx * sdy)
    rho_raw = float(stats.spearmanr(x, y, nan_policy='raise').statistic)
    by_g = {
        g: float(stats.spearmanr(sub[sub.group == g][xcol], sub[sub.group == g][ycol], nan_policy='raise').statistic)
        for g in groups
    }
    return rho_eq, rho_raw, by_g


@lru_cache(None)
def records():
    # Non-provincial, non-community records (national=132, hist=75, us=32 -> 239 records)
    meta = rd(INP / 'records.csv')
    meta = meta[~meta.group.isin(['provincial', 'community'])][['uid', 'group', 'system_id', 'observation_year', 'unit_key', 'P_ref_mw']]
    dec = rd(TAB / 'annual_B_decomposition.csv')
    dec = dec[(dec.kind == 'mechanism') & (dec.variant == 'real')]
    keep = ['uid', 'B', *[c for c in dec.columns if c.startswith('B_')], 'hybrid_window_full',
            'hybrid_tail_discharge_mwh_per_mw', 'lfp_tail_discharge_mwh_per_mw',
            'additional_delivery_mwh_per_mw', 'roi_2025', 'roi_2030', 'roi_2035']
    d = meta.merge(dec[keep], on='uid', validate='1:1')
    rows = []
    for uid in d.uid:
        arr, ev = hourly(uid)
        L = runs(ev)
        rows.append(dict(
            uid=uid, task_hours=int((arr[:, 0] > 1e-10).sum()), charge_hours=int((arr[:, 2] > 1e-9).sum()),
            event_days=int(ev.sum()), events=len(L), mean_dur=ev.sum() / len(L), longest=max(L),
        ))
    d = d.merge(pd.DataFrame(rows), on='uid', validate='1:1')
    d['efc_acaes'] = d.hybrid_tail_discharge_mwh_per_mw / 24.0
    d['efc_lfp'] = d.lfp_tail_discharge_mwh_per_mw / 4.0
    d['win36'] = RM.wins(d.reset_index(drop=True), 2035, 3.6).to_numpy()

    cols = list(d.columns)
    cn = _cn_full_df()[cols]
    jp = _jp_full_df()[cols]
    d_all = pd.concat([cn, d, jp], ignore_index=True)
    assert len(d_all) == 11161 and d_all.uid.is_unique
    d_all['group'] = pd.Categorical(d_all.group, GROUPS, ordered=True)
    assert np.array_equal(d_all.win36.to_numpy(), (d_all.B >= 0).to_numpy())
    return d_all.sort_values(['group', 'uid']).reset_index(drop=True)


@lru_cache(None)
def _cn_placements_df():
    cp = rd(CN_SUITE / 'china_placements_records.csv.gz')
    pa = rd(CN_SUITE / 'china_percentile_audit.csv').set_index('uid')
    cp['real_pct'] = cp['uid'].map(pa['real_pct_tolerance_1e_8']).to_numpy(float)
    return cp


@lru_cache(None)
def placements():
    """Real-date ROI and 5th/50th/95th percentiles and percentile rank of real date among 1,000 placements."""
    a = rd(RC / 'primary_all.csv.gz')
    a = a[~a.group.isin(['provincial', 'community'])]
    real = a[a.seed == -1].set_index('uid')
    rnd = a[a.seed >= 0]
    q = RM.record_quantiles(rnd, 'roi_2035')
    q.columns = ['mc_q05', 'mc_q50', 'mc_q95']
    win = RM.wins(rnd, 2035, 3.6).groupby(rnd.uid).mean().rename('mc_win36')
    rank = {}
    for uid, z in rnd.groupby('uid'):
        v = real.loc[uid, 'roi_2035']
        vals = z.roi_2035.to_numpy()
        rank[uid] = float(np.mean(vals < v) + 0.5 * np.mean(vals == v))
    out_other = q.join(win).join(pd.Series(rank, name='real_pct')).join(
        real[['roi_2035', 'group']].rename(columns={'roi_2035': 'roi_real'})
    )
    out_other.index.name = 'uid'
    out_other = out_other.reset_index()

    cn_pl = _cn_placements_df()[out_other.columns]
    jp_pl = rd(JP_F10 / 'japan_placements_records.csv.gz')[out_other.columns]
    out = pd.concat([cn_pl, out_other, jp_pl], ignore_index=True)
    assert out.mc_q05.notna().all() and out.roi_real.notna().all() and len(out) == 11161
    return out


@lru_cache(None)
def summary():
    s = pd.read_csv(RC / 'primary_summary.csv')
    s = s[(~s.group.isin(['provincial', 'community'])) & (s.q == 50)]
    sp = pd.read_csv(CN_SUITE / 'china_primary_summary.csv')[s.columns].copy()
    sp['real_pct_med'] = float(_cn_placements_df()['real_pct'].median())
    sj = pd.read_csv(JP_F10 / 'japan_primary_summary.csv')[s.columns]
    return pd.concat([sp, s, sj], ignore_index=True)


@lru_cache(None)
def summary_cn_positive():
    sp = pd.read_csv(CN_SUITE / 'china_primary_summary_positive_supply.csv').copy()
    pos_uids = cn_positive_uids()
    cp = _cn_placements_df()
    sp['real_pct_med'] = float(cp.loc[cp.uid.isin(pos_uids), 'real_pct'].median())
    return sp


@lru_cache(None)
def central12():
    """12-day central extension minus real dates across all 11,054 main records."""
    c = rd(TAB / 'central_1to12_decomposition.csv.gz')
    c = c[(pd.to_numeric(c.target) == 12) & (c.group == 'national')]

    def _ext_diff(path, grp):
        ex = rd(path)
        e0 = ex[ex.target == 0].set_index('uid')
        e12 = ex[ex.target == 12].set_index('uid').reindex(e0.index)
        return pd.DataFrame({
            'uid': e0.index,
            'group': grp,
            'delta_B': (e12.B - e0.B).to_numpy(),
            'delta_hybrid_window_full': (e12.hybrid_window_full - e0.hybrid_window_full).to_numpy(),
            'delta_hybrid_tail_discharge_mwh_per_mw': (e12.hybrid_tail_discharge_mwh_per_mw - e0.hybrid_tail_discharge_mwh_per_mw).to_numpy(),
            'delta_additional_delivery_mwh_per_mw': (e12.additional_delivery_mwh_per_mw - e0.additional_delivery_mwh_per_mw).to_numpy(),
        })

    cdf = _ext_diff(CN_SUITE / 'china_extension_0to12.csv.gz', 'provincial')
    jdf = _ext_diff(JP_F10 / 'japan_extension_0to12.csv.gz', 'community')
    cols = ['uid', 'group', 'delta_B', 'delta_hybrid_window_full',
            'delta_hybrid_tail_discharge_mwh_per_mw', 'delta_additional_delivery_mwh_per_mw']
    out = pd.concat([cdf[cols], c[cols], jdf[cols]], ignore_index=True)
    assert len(out) == 11054 and out.uid.is_unique
    return out


def catalogue_summary():
    return pd.read_csv(H2_REVISION / 'tables/catalogue_summary.csv')


def catalogue_best():
    return pd.read_csv(H2_REVISION / 'tables/catalogue_best_counts.csv')


def catalogue_records():
    return rd(H2_REVISION / 'tables/catalogue_records.csv.gz')


@lru_cache(None)
def placement_delta_long_windows():
    """Return DataFrame [uid, group, lw_real, lw_mc_med, dlw] across all 11,054 main records."""
    pw = rd(V86 / 'analysis/placement_windows.csv.gz')
    pw = pw[pw.group == 'national'][['uid', 'seed', 'long_windows', 'group']]
    r_o = pw[pw.seed < 0].set_index('uid')
    m_o = pw[pw.seed >= 0].groupby('uid').long_windows.median()
    eu_part = pd.DataFrame({
        'uid': r_o.index,
        'group': r_o.group.to_numpy(),
        'lw_real': r_o.long_windows.to_numpy(),
        'lw_mc_med': m_o.reindex(r_o.index).to_numpy(),
    })
    cn_part = _cn_placements_df()[['uid', 'group', 'lw_real', 'lw_mc_med']].copy()
    jp_part = rd(JP_F10 / 'japan_placements_records.csv.gz')[['uid', 'group', 'lw_real', 'lw_mc_med']].copy()
    out = pd.concat([cn_part, eu_part, jp_part], ignore_index=True)
    out['dlw'] = out.lw_real - out.lw_mc_med
    assert len(out) == 11054 and out.uid.is_unique
    return out


@lru_cache(None)
def placement_windows():
    """Per-(uid, seed) long_windows for EU example (France 2016)."""
    pw = rd(V86 / 'analysis/placement_windows.csv.gz')
    return pw[pw.group == 'national'][['uid', 'seed', 'long_windows', 'group']]


def examples():
    """Record of each main context closest to the medians of chargeable hours and A-CAES equivalent full cycles."""
    d = records()
    out = {}
    for g in MAIN:
        z = d[d.group == g]
        if g == 'provincial':
            z = z[z.charge_hours > 0]
        dist = np.hypot((z.charge_hours - z.charge_hours.median()) / z.charge_hours.std(),
                        (z.efc_acaes - z.efc_acaes.median()) / z.efc_acaes.std())
        out[g] = z.uid.to_numpy()[np.argmin(dist.to_numpy())]
    return out


def n12_mean_duration():
    r = rd(RC / 'reorder_all.csv.gz', low_memory=False)
    r = r[(r.scenario.astype(str) == '12') & (r.seed == -1) & (r.group == 'national') & (r.q == 50)]
    ce = rd(CN_SUITE / 'china_extension_0to12.csv.gz')
    c12 = ce[ce.target == 12][['uid', 'mean_dur']]
    je = rd(JP_F10 / 'japan_extension_0to12.csv.gz')
    j12 = je[je.target == 12][['uid', 'mean_dur']]
    out = pd.concat([c12, r[['uid', 'mean_dur']], j12], ignore_index=True)
    assert out.uid.is_unique and len(out) == 11054
    return out.set_index('uid').mean_dur


@lru_cache(None)
def extension():
    """Central event extension, target 1-12 days, plus real dates (target 0) across all 11,054 main records."""
    cols = ['uid', 'group', 'target', 'B', 'hybrid_tail_discharge_mwh_per_mw', 'lfp_tail_discharge_mwh_per_mw',
            'additional_delivery_mwh_per_mw', 'B_residual_energy', 'B_residual_peak']
    c = rd(TAB / 'central_1to12_decomposition.csv.gz')
    c['target'] = pd.to_numeric(c.target).astype(int)
    c = c[c.group == 'national'][cols]
    r = rd(RC / 'reorder_all.csv.gz', low_memory=False)
    r = r[(r.seed == -1) & (r.group == 'national') & (r.q == 50)]
    r = r[r.scenario.astype(str).isin([str(i) for i in range(1, 13)])]
    r = r.assign(target=r.scenario.astype(int))[['uid', 'target', 'mean_dur']]
    x_eu = c.merge(r, on=['uid', 'target'], how='left', validate='1:1')
    rec = records()
    rec_eu = rec[rec.group == 'national'].assign(target=0)
    x_eu = pd.concat([rec_eu[cols + ['mean_dur']].assign(group='national'), x_eu], ignore_index=True)

    ce = rd(CN_SUITE / 'china_extension_0to12.csv.gz')[cols + ['mean_dur']]
    je = rd(JP_F10 / 'japan_extension_0to12.csv.gz')[cols + ['mean_dur']]
    out = pd.concat([ce, x_eu, je], ignore_index=True)
    out['efc_acaes'] = out.hybrid_tail_discharge_mwh_per_mw / 24.0
    out['efc_lfp'] = out.lfp_tail_discharge_mwh_per_mw / 4.0
    assert len(out) == 11054 * 13 and out.mean_dur.notna().all()
    return out


@lru_cache(None)
def _factorial_raw():
    eu = rd(PROV / 'factorial_records.csv.gz')
    eu = eu[eu.regime == 'Europe']
    cf = rd(CN_SUITE / 'china_factorial.csv.gz')[eu.columns]
    jf = rd(JP_F10 / 'japan_factorial.csv.gz')[eu.columns]
    return pd.concat([cf, eu, jf], ignore_index=True)


@lru_cache(None)
def factorial(outcome='real'):
    """Surplus-energy x charge-power x charging-access factorial contrasts per record (11,054 records)."""
    d = _factorial_raw()
    w = d.pivot_table(index=['uid', 'regime', 'energy_multiplier', 'charge_power_multiplier', 'access'],
                      columns='order', values='B').reset_index()
    w['dB12'] = w.n12 - w.real
    w['group'] = w.regime.map({'Europe': 'national', 'China': 'provincial', 'Japan': 'community'})
    wide = w.pivot_table(index=['uid', 'group'],
                         columns=['energy_multiplier', 'charge_power_multiplier', 'access'], values=outcome)
    col = lambda e, p, a: wide[(e, p, a)]
    out = pd.DataFrame({
        'surplus_energy_4x': sum(col(4, p, a) - col(1, p, a) for p in (1, 4) for a in ('observed', 'expanded')) / 4,
        'charge_power_4x': sum(col(e, 4, a) - col(e, 1, a) for e in (1, 4) for a in ('observed', 'expanded')) / 4,
        'expanded_access': sum(col(e, p, 'expanded') - col(e, p, 'observed') for e in (1, 4) for p in (1, 4)) / 4,
        'power_x_access': sum(col(e, 4, 'expanded') - col(e, 1, 'expanded')
                              - col(e, 4, 'observed') + col(e, 1, 'observed') for e in (1, 4)) / 2,
    }).reset_index()
    assert len(out) == 11054
    return out


@lru_cache(None)
def windows():
    """Charging windows on real dates per main-context record (11,054 records)."""
    rec = records()
    eu = rec[rec.group == 'national']
    rows = []
    for uid in eu.uid:
        arr, _ = hourly(uid)
        is_call, run, _h = WD.segments(arr)
        call = arr[:, 1] > 1e-10
        ch = np.bincount(run, weights=(arr[:, 2] > 1e-9).astype(float))
        e_c = np.bincount(run, weights=np.where(call, 0., np.minimum(arr[:, 2], 1.5))) * 0.55
        e_l = np.bincount(run, weights=np.where(call, 0., np.minimum(arr[:, 2], 1.0))) * 0.9
        w = (~is_call) & (ch > 0)
        rows.append(dict(uid=uid, group='national', windows=int(w.sum()),
                         lfp_fill_windows=int((w & (e_l >= 3.6 - 1e-9)).sum()),
                         long_windows=int((w & (e_c >= 21.6 - 1e-9)).sum())))
    cn = _cn_full_df()[['uid', 'group', 'windows', 'lfp_fill_windows', 'long_windows']]
    jp = _jp_full_df()[['uid', 'group', 'windows', 'lfp_fill_windows', 'long_windows']]
    out = pd.concat([cn, pd.DataFrame(rows), jp], ignore_index=True)
    assert len(out) == 11054 and out.uid.is_unique
    return out


H2_REVISION = V86 / 'revision/hydrogen_tank_only_20261114'
sys.path.insert(0, str(H2_REVISION / 'code'))
import parameters as _h2_parameters  # noqa: E402
STORE = {
    'caes': (1.5, 0.55, [8, 10, 12, 14, 16, 18, 20, 22, 24]),
    'h2_tank': (3.0, _h2_parameters.h2_physics()['rte'],
                sorted(c['duration_h'] for c in _h2_parameters.catalogue().values() if c['tech_id'] == 'h2_tank')),
}


@lru_cache(None)
def fill_windows_by_duration():
    rec = records()
    eu = rec[rec.group == 'national']
    rows = []
    for uid in eu.uid:
        arr, _ = hourly(uid)
        is_call, run, _h = WD.segments(arr)
        call = arr[:, 1] > 1e-10
        ok = (~is_call) & (np.bincount(run, weights=(arr[:, 2] > 1e-9).astype(float)) > 0)
        for tech, (ratio, rte, durs) in STORE.items():
            e = np.bincount(run, weights=np.where(call, 0., np.minimum(arr[:, 2], ratio))) * rte
            rows += [dict(uid=uid, group='national', tech=tech, duration_h=d,
                          fill_windows=int((ok & (e >= WD.USABLE * d - 1e-9)).sum())) for d in durs]
    cn_fw = rd(CN_SUITE / 'china_fill_windows_by_duration.csv.gz')
    jp_fw = rd(JP_F10 / 'japan_fill_windows_by_duration.csv.gz')
    out = pd.concat([cn_fw, pd.DataFrame(rows), jp_fw], ignore_index=True)
    assert len(out) == 11054 * (len(STORE['caes'][2]) + len(STORE['h2_tank'][2]))
    return out


@lru_cache(None)
def factorial_cells():
    d = _factorial_raw()
    d = d[d.order == 'real'].copy()
    d['group'] = d.regime.map({'Europe': 'national', 'China': 'provincial', 'Japan': 'community'})
    out = d.groupby(['group', 'energy_multiplier', 'charge_power_multiplier', 'access']).B.agg(
        n='size', hybrid_cheaper=lambda b: int((b >= 0).sum())
    ).reset_index()
    assert len(out) == 24
    return out


@lru_cache(None)
def charge_power_windows():
    rec = records()
    eu = rec[rec.group == 'national']
    rows = []
    for uid in eu.uid:
        arr, _ = hourly(uid)
        is_call, run, hours = WD.segments(arr)
        call = arr[:, 1] > 1e-10
        ok = (~is_call) & (np.bincount(run, weights=(arr[:, 2] > 1e-9).astype(float)) > 0)
        r = np.where(call, 0., arr[:, 2])
        row = dict(uid=uid, group='national', one_hour_windows=int((ok & (hours == 1)).sum()))
        for m in (1, 4):
            row[f'lfp_fill_x{m}'] = int((ok & (np.bincount(run, weights=np.minimum(r, 1.0 * m)) * 0.9 >= 3.6 - 1e-9)).sum())
            row[f'long_x{m}'] = int((ok & (np.bincount(run, weights=np.minimum(r, 1.5 * m)) * 0.55 >= 21.6 - 1e-9)).sum())
        rows.append(row)
    cols_cp = ['uid', 'group', 'one_hour_windows', 'lfp_fill_x1', 'lfp_fill_x4', 'long_x1', 'long_x4']
    cn = _cn_full_df()[cols_cp]
    jp = _jp_full_df()[cols_cp]
    out = pd.concat([cn, pd.DataFrame(rows), jp], ignore_index=True)
    assert len(out) == 11054 and out.uid.is_unique
    return out


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    d = records()
    assert examples() == EXAMPLES, examples()
    for g, uid in EXAMPLES.items():
        arr, ev = hourly(uid)
        n = len(fill_windows(arr))
        assert n == WD.branch(arr, DUR, RTE, RATIO)['saturated'] == d.set_index('uid').loc[uid, 'hybrid_window_full']
        print(g, uid, d.set_index('uid').loc[uid, ['unit_key', 'charge_hours', 'efc_acaes', 'mean_dur', 'B', 'roi_2035']].to_dict())
