"""Recover compatible evidence for the full 11,054-record main sample + 107 external validation records."""
from pathlib import Path
import os, sys, json, hashlib
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import numpy as np
import pandas as pd
from scipy import optimize, stats

W = Path(__file__).resolve().parents[1]
V = W.parents[1]
SUB = V.parent
V85 = SUB / 'Nature_Energy_v85'
RE = V85 / 'revision/remaining_experiments_20261103'
RC = V85 / 'revision/real_vs_placement_20261029/results/report_corrected'
A = V85 / 'source_data/archived_v82'
sys.path.insert(0, str(V / 'figures/code'))
import data as D  # noqa: E402
sys.path.insert(0, str(RE / 'code'))
import common as C  # noqa: E402
sys.path.insert(0, str(SUB / 'Nature_Energy_v82/revision/advisor_key_tests_20261022/t6_window'))
from cyclic_diagnostic import cyclic_recursion  # noqa: E402
D.WD.recursion = cyclic_recursion

SD = W / 'tables'
provenance = {}


def save(name, d, source):
    p = SD / (name + '.csv')
    d.to_csv(p, index=False)
    provenance[name] = {'source': source, 'rows': len(d), 'columns': list(d), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def copy_table(path, name):
    save(name, pd.read_csv(path), str(path))


rec = D.records()
save('annual_records_current', rec, 'Current figure input adapter; all 11,161 records (11,054 main + 107 ext)')
main = rec[rec.group.isin(D.MAIN)]
ext = D.extension()
save('extension_current', ext, 'Current figure input adapter; 11,054 x 13')
placements = D.placements()
save('placements_current', placements, 'Current 1,000-placement record summaries')
print('Annual, extension and placement records collected', flush=True)

# Price and operating-rule sensitivities across all 11,054 main records
for stem in ('price', 'rule'):
    old = pd.read_csv(RE / f'tables/{stem}_all.csv', dtype={'uid': str})
    old = old[old.group == 'national'][['uid', 'group', 'variant', 'B']]
    cn = pd.read_csv(D.CN_SUITE / f'china_{stem}.csv.gz', dtype={'uid': str})[old.columns]
    jp = pd.read_csv(D.JP_F10 / f'japan_{stem}.csv.gz', dtype={'uid': str})[old.columns]
    df = pd.concat([cn, old, jp], ignore_index=True)
    assert len(df) == 11054 * 4 and not df.duplicated(['uid', 'variant']).any()
    central = df[df.variant.eq('x1' if stem == 'price' else 'registered')].set_index('uid').B
    assert np.allclose(central, main.set_index('uid').B.reindex(central.index), atol=1e-6)
    save(stem + '_current', df, 'Provincial (310) + national (132) + Japanese community (10,612) q50 results')
    sm = df.groupby(['group', 'variant']).B.agg(
        n='size', median='median', mean='mean', minimum='min', maximum='max', win=lambda z: int((z >= 0).sum())
    ).reset_index()
    save(stem + '_summary_current', sm, 'Per-record results above')

copy_table(RE / 'tables/reused_national_vre_sensitivity.csv', 'national_vre_current')

# Unified 7-baseline (q5..q50) real vs MC summary across all 3 regimes (+ China monthly peak-clipping)
eu_q10_50 = pd.read_csv(V / 'revision/china_load_baseline_audit_20261118/tables/archived_multi_q_2035.csv')
eu_q10_50 = eu_q10_50[eu_q10_50.group == 'national'].copy()
eu_q5 = pd.read_csv(D.EU_Q5_DIR / 'europe_q5_summary.csv')
eu_q5_35 = eu_q5[eu_q5.year == 2035].iloc[0]
jp_7q = pd.read_csv(D.JP_MC_ALLQ / 'japan_MC_baseline_summary.csv')

q_rows = []
cn_prop_sum = pd.read_csv(D.CN_DIR / 'results/yearly_surplus/years_2015_2024/china_baseline_summary.csv')
cn_prop_sum = cn_prop_sum[cn_prop_sum.cohort == 'all_31']
cn_clip_sum = pd.read_csv(D.CN_DIR / 'results/yearly_concentrated_all_q_20261120/years_2015_2024/china_baseline_summary.csv')
cn_clip_sum = cn_clip_sum[cn_clip_sum.cohort == 'all_31']
for label_s, df_s in [('China_proportional', cn_prop_sum), ('China_clipped', cn_clip_sum)]:
    for r in df_s.sort_values('q').itertuples():
        q_rows.append(dict(
            series=label_s, group='provincial', q=int(r.q), n=int(r.annual_records),
            real_wins=int(r.original_wins), real_share_pct=100.0 * float(r.original_fraction),
            mc_share_pct=100.0 * float(r.mc_fraction),
        ))
q_rows.append(dict(
    series='Europe', group='national', q=5, n=132,
    real_wins=int(eu_q5_35['real_win_3.6']), real_share_pct=100.0 * float(eu_q5_35['real_win_3.6']) / 132.0,
    mc_share_pct=100.0 * float(eu_q5_35['mc_win_3.6']),
))
for r in eu_q10_50.sort_values('q').itertuples():
    rw_eu = int(eu_q10_50.loc[eu_q10_50.q == r.q, 'real_win_3.6'].iloc[0])
    mw_eu = float(eu_q10_50.loc[eu_q10_50.q == r.q, 'mc_win_3.6'].iloc[0])
    q_rows.append(dict(
        series='Europe', group='national', q=int(r.q), n=132,
        real_wins=rw_eu, real_share_pct=100.0 * rw_eu / 132.0, mc_share_pct=100.0 * mw_eu,
    ))
for r in jp_7q.sort_values('q').itertuples():
    q_rows.append(dict(
        series='Japan', group='community', q=int(r.q), n=int(r.records),
        real_wins=int(r.observed_wins), real_share_pct=100.0 * float(r.observed_win_fraction),
        mc_share_pct=100.0 * float(r.mc_win_fraction),
    ))
mq_df = pd.DataFrame(q_rows).sort_values(['series', 'q']).reset_index(drop=True)
assert len(mq_df) == 4 * 7
save('multi_q_7baselines_current', mq_df, 'Complete 7-baseline (q5..q50) real vs 1,000-MC win shares across CN (prop & clip), EU, JP')
copy_table(V / 'revision/中日基线与蒙特卡洛实验报告_20261118/中国_两种充电方法_七个负荷基线_混合组合胜出比例.csv', 'china_charging_7baselines_current')

# Thresholds, historical calendar, conservation & window recursion (239 EU/hist/US + precomputed 310 CN + 10,612 JP)
tr, hist_days, wd, cons, threshold_ties = [], [], [], [], []
ext_eu = ext[ext.group == 'national'].set_index(['uid', 'target'])
for row in rec[~rec.group.isin(['provincial', 'community'])].itertuples():
    arr, ev = D.hourly(row.uid)
    z = np.load(D.INP / f'{row.uid}.npz')
    daily = arr[:, 0].reshape(365, 24).sum(1)
    if row.group == 'national':
        seg = z['segment']
        for q in (0.70, 0.75, 0.80):
            flag = np.zeros(365, bool)
            for s in np.unique(seg):
                k = seg == s
                threshold = np.quantile(daily[k], q)
                flag[k] = (daily[k] > threshold) & ~np.isclose(daily[k], threshold, rtol=1e-12, atol=1e-12)
            if q == 0.75:
                for j in np.flatnonzero(flag != ev):
                    threshold = np.quantile(daily[seg == seg[j]], q)
                    assert np.isclose(daily[j], threshold, rtol=1e-12, atol=1e-12), (row.uid, j, daily[j], threshold)
                    threshold_ties.append(dict(uid=row.uid, day=int(j), difference=float(daily[j] - threshold), stored_flag=bool(ev[j])))
                flag = ev.copy()
            rr = D.runs(flag)
            tr.append(dict(
                uid=row.uid, country=row.system_id, year=row.observation_year, q=q,
                high_days=int(flag.sum()), events=len(rr), longest=max(rr), mean_duration=float(np.mean(rr)),
                task_energy=daily.sum(), high_task_energy=daily[flag].sum(), high_task_share=daily[flag].sum() / daily.sum(),
            ))
        _, ens = C.inputs(row.uid)
        for target in (0, 1, 4, 12):
            order = np.arange(365) if target == 0 else np.asarray(ens.orders(target)['clustered'])
            b = C.ce.reorder_days(arr, order)
            cons.append(dict(
                uid=row.uid, group='national', target=target,
                absolute_error=float(np.max(np.abs(b.sum(0) - arr.sum(0)))),
                relative_error=float(np.max(np.abs(b.sum(0) - arr.sum(0)) / np.maximum(1, np.abs(arr.sum(0))))),
            ))
            ca = D.WD.branch(b, 24, 0.55, 1.5)
            lf = D.WD.branch(b, 4, 0.9, 1.0)
            actual = ext_eu.loc[(row.uid, target)]
            wd.append(dict(
                uid=row.uid, group='national', target=target,
                diagnostic_caes=ca['delivered'], diagnostic_lfp=lf['delivered'],
                actual_caes=actual.hybrid_tail_discharge_mwh_per_mw,
                actual_lfp=actual.lfp_tail_discharge_mwh_per_mw,
                B=actual.B, cyclic_caes=ca['cyclic_residual'], cyclic_lfp=lf['cyclic_residual'],
            ))
    if row.group == 'hist':
        hist_days.extend(dict(uid=row.uid, year=int(row.observation_year), day=j, event=int(e)) for j, e in enumerate(ev))

cons_df = pd.concat([
    pd.read_csv(D.CN_SUITE / 'china_chronology_conservation.csv.gz', dtype={'uid': str}),
    pd.DataFrame(cons),
    pd.read_csv(D.JP_F10 / 'japan_chronology_conservation.csv.gz', dtype={'uid': str}),
], ignore_index=True)
assert len(cons_df) == 11054 * 4

wd_df = pd.concat([
    pd.read_csv(D.CN_SUITE / 'china_window_recursion.csv.gz', dtype={'uid': str}),
    pd.DataFrame(wd),
    pd.read_csv(D.JP_F10 / 'japan_window_recursion.csv.gz', dtype={'uid': str}),
], ignore_index=True)
assert len(wd_df) == 11054 * 4

save('threshold_current', pd.DataFrame(tr), 'Recomputed from q50 national tasks; seasonal q70/q75/q80; all 132 records')
save('historical_calendar_current', pd.DataFrame(hist_days), 'Exact 75 current q50 annual input flags')
save('chronology_conservation_current', cons_df, 'Independent complete-day reorder checks, all 11,054 records, target 0/1/4/12')
save('window_recursion_current', wd_df, 'Exact cyclic reduced window recursion, all 11,054 records, target 0/1/4/12')

dd = wd_df[wd_df.target.isin([4, 12])].merge(wd_df[wd_df.target.eq(0)].drop(columns='target'), on=['uid', 'group'], suffixes=('', '_real'), validate='m:1')
dd['I_E'] = (dd.diagnostic_caes - dd.diagnostic_caes_real) - (dd.actual_lfp * 0.0 + dd.diagnostic_lfp - dd.diagnostic_lfp_real)
dd['actual_delta_delivery'] = (dd.actual_caes - dd.actual_caes_real) - (dd.actual_lfp - dd.actual_lfp_real)
dd['delta_B'] = dd.B - dd.B_real
dd['sign_match'] = np.sign(dd.I_E) == np.sign(dd.delta_B)
save('window_recursion_effects_current', dd, 'Paired with real dates; no optimization changes')
print('Thresholds, calendars and window diagnostics recomputed', flush=True)

# Alternative equal-delivery constant-power service valuation across all 11,161 records
base = pd.read_csv(RE / 'tables/annual_B_decomposition.csv', dtype={'uid': str})
base = base[(base.kind == 'mechanism') & (base.variant == 'real') & (~base.group.isin(['provincial', 'community']))]
cols = ['uid', 'group', 'B', 'B_residual_peak', 'lfp_total_discharge_mwh_per_mw', 'hybrid_total_discharge_mwh_per_mw']
vals_other = base[cols].reset_index(drop=True).copy()
assert len(vals_other) == 239
for idx, row in vals_other.iterrows():
    arr, _ = D.hourly(row.uid)
    load = arr[:, 0]
    for branch in ('lfp', 'hybrid'):
        delivered = row[f'{branch}_total_discharge_mwh_per_mw']
        fn = lambda f: np.minimum(load, f).sum() - delivered
        vals_other.loc[idx, f'F_{branch}'] = optimize.brentq(fn, 0, float(load.max()), xtol=1e-12)
vals_other['B_equivalent_service'] = vals_other.B - vals_other.B_residual_peak + 117.040888604 * (vals_other.F_hybrid - vals_other.F_lfp)

cn_srv = D._cn_full_df()[cols + ['F_lfp', 'F_hybrid', 'B_equivalent_service']].copy()
jp_srv = D._jp_full_df()[cols + ['F_lfp', 'F_hybrid', 'B_equivalent_service']].copy()
vals = pd.concat([cn_srv, vals_other, jp_srv], ignore_index=True)
assert len(vals) == 11161
save('equivalent_service_current', vals, 'Identical current dispatch; peak term replaced by equal-delivery constant-power valuation')

# Finance paths & 12-day loss vs M
financial = []
d12 = D.central12().set_index('uid')
for rate in [0.0, 0.01, 0.02, 0.036, 0.05, 0.06, 0.08, 0.10]:
    for h in [8, 16, 24]:
        prem = lambda y: D.RM.FC(f'caes_{h}h', y, rate) - D.RM.FC('li_lfp_4h', y, rate)
        financial.append(dict(rate=rate, duration_h=h, premium2025=prem(2025), premium2030=prem(2030), premium2035=prem(2035), M=prem(2025) - prem(2035)))
fin = pd.DataFrame(financial)
save('finance_paths_current', fin, 'Unchanged original A-CAES/LFP learning paths and component schedules')
fr = []
for f in fin[fin.duration_h.eq(24)].itertuples():
    for g in D.MAIN:
        ids = main[main.group == g].uid
        delta = d12.loc[ids].delta_B
        fr.append(dict(group=g, rate=f.rate, M=f.M, n=len(ids), loss_ge_M=int((-delta >= f.M).sum()) if f.M > 0 else np.nan, negative=int((delta < 0).sum())))
save('finance_extension_current', pd.DataFrame(fr), 'Current 12-day minus real-date effects; operating prices fixed')

h = rec[rec.group == 'hist'].sort_values('observation_year')
pairs = []
for i, a in enumerate(h.itertuples()):
    for b in list(h.itertuples())[i + 1:]:
        pairs.append(dict(year0=int(a.observation_year), year1=int(b.observation_year), gap=int(b.observation_year - a.observation_year), delta_B=b.B - a.B))
save('historical_pairs_current', pd.DataFrame(pairs), 'All 2,775 current q50 weather-year pairs; same cost year')

ed = A / 'extended_data'
for name in ['capacity_weights', 'zone_inventory', 'zone_matching', 'extended_data_figure2b_spatial_coverage',
             'extended_data_figure2d_matching_counts', 'extended_data_figure1a_country_holdout_fit',
             'extended_data_figure1d_hourly_error', 'demand_month_hour_surface', 'demand_annual_shape', 'extended_data_figure1f_yearly_rmse']:
    copy_table(ed / f'{name}.csv', name)
copy_table(A / 'system_cost_v82/supplementary/input_statistics/demand_loco_country_validation.csv', 'demand_spatial_validation')
for q in [10, 50]:
    copy_table(V85 / f'revision/europe_trend_q50_20261103/results/annual_event_metrics_q{q}.csv', f'historical_trend_q{q}')
copy_table(V85 / 'revision/europe_trend_q50_20261103/results/comparison.csv', 'historical_trend_comparison')
f = pd.read_csv(V85 / 'source_data/figures_main/figure4_fixed_technology.csv')
f = f[f.technology == 'caes_control']
save('fixed_caes_archived', f, 'Fixed A-CAES capacity mechanism; q10 national/community, flat provincial task; original published computation.')
for name in ['ceilings', 'nines_summary', 'event_extension_effect', 'feasibility_by_regime', 'extended_data_figure8_source_data']:
    copy_table(A / f'reliability_nines/{name}.csv', f'reliability_{name}')
for name in ['capacity_design_summary', 'capacity_technology_power', 'nearoptimal_intervals', 'nearoptimal_endpoints', 'capacity_scenario_metrics']:
    copy_table(D.H2_REVISION / f'tables/{name}.csv', f'{name}_current')
copy_table(D.H2_REVISION / 'tables_185_backup/hydrogen_dispatch.csv.gz' if (D.H2_REVISION / 'tables_185_backup/hydrogen_dispatch.csv.gz').exists() else D.H2_REVISION / 'tables/hydrogen_dispatch.csv.gz', 'hydrogen_dispatch_current')

summary = {}
summary['window_sign_match'] = {str(k): int(g.sign_match.sum()) for k, g in dd.groupby('target')}
summary['window_spearman'] = {str(k): float(stats.spearmanr(g.I_E, g.delta_B).statistic) for k, g in dd.groupby('target')}
summary['window_equal_regime_spearman'] = {str(k): float(D.equal_regime_spearman(g, 'I_E', 'delta_B')[0]) for k, g in dd.groupby('target')}
summary['service_valuation'] = []
for g, z in vals.groupby('group'):
    summary['service_valuation'].append(dict(
        group=g, n=len(z), rho=float(stats.spearmanr(z.B, z.B_equivalent_service).statistic),
        system_positive=int((z.B >= 0).sum()), service_positive=int((z.B_equivalent_service >= 0).sum()),
        max_absolute_difference=float((z.B - z.B_equivalent_service).abs().max()),
    ))
summary['historical_B_span'] = float(h.B.max() - h.B.min())
summary['historical_B_p05_p95'] = float(h.B.quantile(0.95) - h.B.quantile(0.05))
summary['chronology_max_relative_error'] = float(cons_df.relative_error.max())
summary['threshold_q75_reproduced'] = 132
summary['threshold_floating_ties'] = threshold_ties
(W / 'qa/restored_statistics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
(W / 'qa/restored_source_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
