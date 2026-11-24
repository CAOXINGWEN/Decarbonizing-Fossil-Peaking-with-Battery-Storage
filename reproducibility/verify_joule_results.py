"""Standalone numerical verification of headline results reported in:
Cao, X. & Zhang, H. 'The economic value of long-duration energy storage under stochastic balancing requirements' (Joule).

Run from the repository root:
    python reproducibility/verify_joule_results.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]


def main():
    checks = {}

    # 1. Sample coverage: 11,054 main-text records (310 China + 132 Europe + 10,612 Japan)
    f2 = pd.read_csv(REPO / 'source_data/figures/figure2_records.csv')
    counts = f2.groupby('group').size().to_dict()
    assert len(f2) == 11054, f'Expected 11,054 main records, got {len(f2)}'
    assert counts == {'community': 10612, 'national': 132, 'provincial': 310}, counts
    checks['sample_counts'] = {
        'total_main_records': int(len(f2)),
        'chinese_provinces': int(counts['provincial']),
        'european_countries': int(counts['national']),
        'japanese_communities': int(counts['community']),
    }

    # 2. Historical win rates at 2035 costs and 3.6% real cost of capital
    hist_wins = f2.groupby('group')['win36'].agg(['sum', 'mean']).to_dict('index')
    assert int(hist_wins['provincial']['sum']) == 67
    assert int(hist_wins['national']['sum']) == 24
    assert int(hist_wins['community']['sum']) == 1468
    checks['historical_2035_3p6_wins'] = {
        k: {'wins': int(v['sum']), 'share_pct': round(100.0 * float(v['mean']), 2)}
        for k, v in hist_wins.items()
    }

    # 3. Conditional date rearrangement win rates (11,054,000 optimizations = 1,000 per record)
    pl = pd.read_csv(REPO / 'source_data/tables/placements_current.csv')
    pl_main = pl[pl['group'].isin(['provincial', 'national', 'community'])]
    assert len(pl_main) == 11054
    win_frac = pl_main['mc_win36'] / (1000.0 if pl_main['mc_win36'].max() > 1.5 else 1.0)
    mc_means = {
        g: round(100.0 * float(win_frac.loc[sub.index].mean()), 2)
        for g, sub in pl_main.groupby('group')
    }
    assert round(mc_means['provincial']) == 22
    assert round(mc_means['national']) == 34
    assert round(mc_means['community']) == 56
    checks['rearrangement_mean_win_pct_3p6'] = mc_means

    # 4. Controlled 1–12 day event-duration extension: 3-day continuous call threshold turns median B < 0 across all 3 regimes
    ext = pd.read_csv(REPO / 'source_data/tables/extension_current.csv')
    ar_full = pd.read_csv(REPO / 'source_data/tables/annual_records_current.csv')
    cn_pos_uids = set(ar_full[(ar_full['group'] == 'provincial') & (ar_full['charge_hours'] > 0)]['uid'])
    assert len(cn_pos_uids) == 178, f'Expected 178 positive-curtailment CN records, got {len(cn_pos_uids)}'

    d3 = ext[ext['target'] == 3]
    med_b_d3 = {
        'provincial_pos178': round(float(d3[d3['uid'].isin(cn_pos_uids)]['B'].median()), 2),
        'national': round(float(d3[d3['group'] == 'national']['B'].median()), 2),
        'community': round(float(d3[d3['group'] == 'community']['B'].median()), 2),
    }
    assert all(v < 0 for v in med_b_d3.values()), med_b_d3
    checks['three_day_continuous_call_median_B_usd_kw_yr'] = med_b_d3

    # 5. 58-configuration catalog across 8 storage technologies (641,132 main-text evaluations)
    cat_df = pd.read_csv(REPO / 'source_data/technology_catalog_58/catalogue_records.csv.gz')
    cat_main = cat_df[(cat_df['group'].isin(['provincial', 'national', 'community'])) & (cat_df['variant'] == 'baseline')] if 'variant' in cat_df.columns else cat_df[cat_df['group'].isin(['provincial', 'national', 'community'])]
    if len(cat_main) == 0:
        cat_main = cat_df[cat_df['group'].isin(['provincial', 'national', 'community'])]
    n_configs = int(cat_main[['tech', 'duration_h']].drop_duplicates().shape[0])
    assert n_configs == 58, f'Expected 58 configurations, got {n_configs}'
    assert len(cat_main) == 11054 * 58 == 641132, f'Expected 641,132 main rows, got {len(cat_main)}'
    best_counts = pd.read_csv(REPO / 'source_data/technology_catalog_58/catalogue_best_counts.csv')
    checks['technology_catalog'] = {
        'configurations': n_configs,
        'main_evaluations': int(len(cat_main)),
        'total_archived_rows': int(len(cat_df)),
        'least_cost_families': best_counts.to_dict('records'),
    }

    # 6. Equal-energy constant-power service valuation Spearman correlations
    eq = pd.read_csv(REPO / 'source_data/tables/equivalent_service_current.csv')
    eq_rho = {}
    for g in ('provincial', 'national', 'community'):
        sub = eq[eq['group'] == g]
        rho, _ = spearmanr(sub['B'], sub['B_equivalent_service'])
        eq_rho[g] = round(float(abs(rho)), 2)
    checks['equal_energy_service_spearman_abs_rho'] = eq_rho

    out = {'passed': True, 'manuscript': 'Joule (v1.0-joule)', 'verified_metrics': checks}
    out_path = REPO / 'verification/joule_headline_verification.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
