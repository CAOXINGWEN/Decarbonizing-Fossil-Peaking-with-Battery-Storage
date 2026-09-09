"""Revalue frozen natural-year dispatch under the user's selected EFC endpoint."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RESULTS=ROOT/'Decarbonizing-Fossil-Peaking-with-Battery-Storage-master/paper-code/docs/musrcipt/revision_nature_energy_v62/weather_year_parity_v42/results'

def main():
    arm_path=RESULTS/'pecd_v42_75_window_arm_results.csv.gz'
    cost_path=RESULTS/'pecd_v42_75_window_relative_values_by_cost_year.csv'
    d=pd.read_csv(arm_path)
    nonservice_cost=d.system_cost_eur-d.energy_value_eur_mwh*d.unserved_eens_mwh-1000*d.firm_value_eur_kw_year*d.residual_peak_mw
    d['efc_value_rebuilt']=d.energy_value_eur_mwh*(d.base_discharge_mwh+d.tail_discharge_mwh)+1000*d.firm_value_eur_kw_year*d.portfolio_efc_mw-nonservice_cost
    a=d[d.config_id.eq('caes_24h')].set_index('window_start_year').sort_index()
    b=d[d.config_id.eq('li_lfp_4h')].set_index('window_start_year').sort_index()
    for col in ['fixed_common_power_mw','energy_value_eur_mwh','firm_value_eur_kw_year','charge_price_eur_mwh','base_task_mwh']:
        assert np.allclose(a[col],b[col],rtol=0,atol=1e-8),col
    p=a.fixed_common_power_mw
    obj=(b.system_cost_eur-a.system_cost_eur)/(1000*p)
    wedge=a.firm_value_eur_kw_year*((a.portfolio_efc_mw-b.portfolio_efc_mw)-(b.residual_peak_mw-a.residual_peak_mw))/p
    efc_direct=(a.efc_value_rebuilt-b.efc_value_rebuilt)/(1000*p)
    efc=obj+wedge
    assert np.allclose(efc,efc_direct,rtol=0,atol=1e-8)
    years=pd.DataFrame({'window_start_year':a.index,'system_cost_value_2025':obj.to_numpy(),'efc_value_2025':efc.to_numpy(),'valuation_difference':wedge.to_numpy()})
    cy=pd.read_csv(cost_path)
    cy=cy.merge(years,on='window_start_year',validate='m:1')
    assert np.allclose(cy.caes24_minus_lfp4_eur_kw_year,cy.system_cost_value_2025+cy.relative_value_shift_from_2025_eur_kw_year,rtol=0,atol=1e-8)
    cy['efc_value_by_cost_year']=cy.efc_value_2025+cy.relative_value_shift_from_2025_eur_kw_year
    threshold=float(cy.loc[cy.cost_year.eq(2035),'relative_value_shift_from_2025_eur_kw_year'].iloc[0])
    summary={}
    for name,vals,col in [('system_cost',obj,'caes24_minus_lfp4_eur_kw_year'),('efc',efc,'efc_value_by_cost_year')]:
        values=vals.to_numpy(); differences=values[None,:]-values[:,None]
        pairs=differences[np.triu_indices(len(values),1)]; decades=values[10:]-values[:-10]
        summary[name]={'central_5_95_span':float(np.quantile(values,.95)-np.quantile(values,.05)),
            'full_span':float(np.ptp(values)),'pair_exceedances':int((abs(pairs)>=threshold).sum()),'pair_count':len(pairs),
            'adverse_decade_exceedances':int((decades<=-threshold).sum()),'favourable_decade_exceedances':int((decades>=threshold).sum()),'decade_count':len(decades),
            'nonnegative_relative_value_counts':{str(int(y)):int((g[col]>=0).sum()) for y,g in cy.groupby('cost_year')}}
    old=pd.read_csv(OUT/'paired_ledger_panel.csv').accounting_difference
    output={'scope':'Revaluation of existing frozen dispatch only; manuscript not modified',
        'cost_scale':threshold,'natural_75_year_results':summary,
        'direct_vs_difference_identity_max_error':float(abs(efc-efc_direct).max()),
        'country_year_132_direction_check':{'efc_above_objective':int((old>1e-8).sum()),'efc_below_objective':int((old< -1e-8).sum())},
        'interpretation':'EFC crossings are service-valuation crossings on system-cost-optimal dispatch, not system-cost-optimal technology switches.',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [arm_path,cost_path]}}
    years.to_csv(OUT/'natural_75_years_efc_and_system_cost.csv',index=False)
    cy.to_csv(OUT/'natural_75_years_both_ledgers_by_cost_year.csv',index=False)
    (OUT/'efc_primary_natural_year_diagnostic.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
