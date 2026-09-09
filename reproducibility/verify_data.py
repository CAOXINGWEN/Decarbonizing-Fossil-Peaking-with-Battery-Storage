"""Verify package-local ledger units, identities and Figure 5 scales."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

package=Path(__file__).resolve().parents[1]
src=package/'source_data'
p=pd.read_csv(src/'main_efc/paired_ledger_panel.csv',float_precision='round_trip')
meta=json.loads((src/'main_efc/paired_ledger_panel.units.json').read_text())
assert meta['reporting_currency']=='USD' and len(p)==132
assert all(c['unit']=='USD kW^-1 yr^-1' for c in meta['columns'].values())
errors={
 'valuation_difference':float(abs(p.efc_value-p.system_cost_value-p.accounting_difference).max()),
 'capacity_credit_difference':float(abs(p.incremental_efc_credit-p.avoided_residual_peak_credit-p.accounting_difference_from_capacity_terms).max()),
 'two_difference_definitions':float(abs(p.accounting_difference-p.accounting_difference_from_capacity_terms).max())}
assert max(errors.values())<meta['numerical_tolerance']
c=pd.read_csv(src/'main_efc/figure5_configuration_cost_scales.csv').set_index('config_id')
for y in [2025,2035]:
 cost=c[f'annual_fixed_cost_{y}_usd_kw_year'];gap=c[f'relative_fixed_cost_gap_{y}_usd_kw_year']
 np.testing.assert_allclose(cost-cost.loc['li_lfp_4h'],gap,rtol=1e-12,atol=1e-10)
movement=(c.relative_fixed_cost_gap_2035_usd_kw_year-c.relative_fixed_cost_gap_2025_usd_kw_year).abs()
np.testing.assert_allclose(movement,c.absolute_relative_cost_movement_usd_kw_year,rtol=1e-12,atol=1e-10)
r=pd.read_csv(src/'main_efc/all_duration_efc_records.csv.gz')
matched=.5*(r.dominant_config_id_clustered.map(movement)+r.dominant_config_id_dispersed.map(movement))
assert matched.notna().all()
np.testing.assert_allclose(matched,r.matched_cost_movement_usd_kw_year,rtol=1e-12,atol=1e-10)
assert np.array_equal(r.sequencing_shift_usd_kw_year>=matched,r.above_parity)
surface=pd.read_csv(src/'event_cluster_surface/v60_cluster_intensity_system_year.csv.gz')
for col in ['tech_id_clustered','tech_id_dispersed']:
 assert surface.loc[surface.duration_h==24,col].eq('caes').all()
 assert surface.loc[surface.duration_h>24,col].eq('h2_cavern').all()
benchmark=float(movement.loc['caes_24h'])
a=pd.read_csv(src/'event_cluster_surface/v60_cluster_intensity_surface.csv')
np.testing.assert_allclose(a.block_mean_effect_usd_kw_year/benchmark,a.signed_parity_ratio,rtol=1e-12,atol=1e-10)
report={'passed':True,'ledger_rows':len(p),'identity_max_absolute_errors_usd_kw_year':errors,'figure5_paired_records':len(r),'surface_records':len(surface),'fixed_benchmark_usd_kw_year':benchmark}
(package/'verification/portable_data_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
