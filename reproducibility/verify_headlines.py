"""Recompute manuscript headline statistics from the published USD Source Data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

repository = Path(__file__).resolve().parents[1]
src = repository / 'source_data'
cost = pd.read_csv(src / 'main_efc/figure5_configuration_cost_scales.csv').set_index('config_id')
benchmark = float(cost.loc['caes_24h', 'absolute_relative_cost_movement_usd_kw_year'])
natural = pd.read_csv(src / 'natural_years/pecd_v42_75_window_relative_values_2025.csv')
v = natural.sort_values('window_start_year').value_2025_usd_kw_year.to_numpy()
pairs = np.abs(v[:, None] - v[None, :])[np.triu_indices(len(v), 1)]
decades = v[10:] - v[:-10]
span = float(np.diff(np.quantile(v, [.05, .95]))[0])
focal = pd.read_csv(src / 'main_efc/figure3_24h_value_cycle_records.csv')
observations = {}
for regime, group in focal.groupby('regime'):
    delta = group.signed_sequencing_effect_usd_kw_year
    observations[regime] = {'n': len(group), 'median_absolute_efc_shift_usd_kw_year': float(delta.abs().median()),
                           'pairs_reaching_benchmark': int(delta.abs().ge(benchmark).sum()),
                           'cycling_decreases': int(group.cycle_effect.lt(0).sum())}
p = pd.read_csv(src / 'capacity_procurement/procurement_designs_v69.csv')
checks = {
    '75_operating_years': len(v) == 75,
    'central_span_8_63_USD': abs(span - 8.63232552381285) < 1e-8,
    '496_of_2775_pairs_reach_learning_scale': len(pairs) == 2775 and int((pairs >= benchmark).sum()) == 496,
    '6_of_65_adverse_decades_offset_learning': len(decades) == 65 and int((decades <= -benchmark).sum()) == 6,
    '185_paired_observations': len(focal) == 185,
    '185_cycling_decreases': sum(x['cycling_decreases'] for x in observations.values()) == 185,
    'regime_parity_counts': {k: x['pairs_reaching_benchmark'] for k, x in observations.items()} == {'Europe': 68, 'China': 26, 'Japan': 11},
    'regime_rounded_medians': all(abs(observations[k]['median_absolute_efc_shift_usd_kw_year'] - x) < .005 for k, x in {'Europe': 6.19, 'China': 67.24, 'Japan': 5.41}.items()),
    '540_procurement_designs': len(p) == 540,
    'all_procurement_optimal': p.solver_status.eq('optimal').all(),
    'uncapped_procurement_power': p.capacity_bound.eq('nonnegative_without_task_peak_upper_bound').all(),
    '2_4_6_hour_grids': set(p.planning_step_hours) == {2, 4, 6},
    'task_balance': p.maximum_task_balance_residual.abs().max() < 1e-7,
    'shared_recharge': p.maximum_shared_recharge_violation.abs().max() < 1e-7,
    'cyclic_soc': p.maximum_state_balance_residual_mwh.abs().max() < 1e-7,
}
report = {'passed': all(checks.values()), 'checks': {k: bool(v) for k, v in checks.items()},
          'benchmark_usd_kw_year': benchmark, 'central_5_95_span_usd_kw_year': span, 'controlled_results': observations,
          'validation_scope': 'Arithmetic from included Source Data and archived feasibility residuals. No optimization reruns.'}
(repository / 'verification').mkdir(exist_ok=True)
(repository / 'verification/headline_check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if not report['passed']:
    raise SystemExit(1)
