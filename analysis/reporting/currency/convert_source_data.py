"""Convert monetary Source Data to USD without rounding stored observations."""
from config import *
import pandas as pd, numpy as np, json, shutil, hashlib, re

EXTRA_BY_NAME={
 'figure3_value_component_effects.csv.gz':['fixed_cost','variable_om','charging','energy_service','firm_service','total','component_sum'],
 'coefficient_comparison_both_ledgers_and_difference.csv':['coefficient','ci95_low','ci95_high','reported_coefficient','reported_ci95_low','reported_ci95_high'],
 'coefficient_summary.csv':['coefficient','bootstrap_ci95_low','bootstrap_ci95_high'],
 'pecd_v42_focal_cost_path_2025_2035.csv':['caes_24h','li_lfp_4h'],
 'natural_75_years_both_ledgers_by_cost_year.csv':['system_cost_value_2025','efc_value_2025','efc_value_by_cost_year','valuation_difference'],
 'paired_ledger_panel.csv':['efc_value','system_cost_value','accounting_difference','incremental_efc_credit','avoided_residual_peak_credit','accounting_difference_from_capacity_terms'],
 'figure5_controlled_cluster_dose.csv':['country_block_mean','country_year_median','ci95_low','ci95_high'],
 'duration_boundary_summary.csv':['block_mean_effect_eur_kw_year','boundary_ci95_low','boundary_ci95_high'],
 'recharge_efc_records.csv':['efc_score_clustered','efc_score_dispersed','chronology_effect'],
 'factorial_cell_summary.csv':['block_mean_effect','ci95_low','ci95_high'],
 'factorial_contrast_summary.csv':['block_mean_contrast','ci95_low','ci95_high'],
 'ed3_v43_accounting_residuals.csv':['maximum_absolute_residual'],
 'price_channel_caes24_summary.csv':['ci95_low','ci95_high'],
 's3b_caes24_summary.csv':['ci95_low','ci95_high'],
}
CI_NAMES={'v60_cluster_intensity_surface.csv','figure4_recharge_intervention_summary.csv','figure5_duration_response_summary.csv'}

def columns_to_scale(d,name):
    monetary=[c for c in d if re.search(r'(^|_)eur($|_)',c,re.I) and not c.startswith('reporting_fx_')]
    for c in EXTRA_BY_NAME.get(name,[]):
        if c in d and c not in monetary:monetary.append(c)
    if name in CI_NAMES:monetary += [c for c in ['ci95_low','ci95_high'] if c in d]
    if 'lp_cvar_epigraph_value' in d:monetary.append('lp_cvar_epigraph_value')
    return list(dict.fromkeys(monetary))

JSON_CHANGES=[]
def json_usd(obj,key='',name='',path=''):
    if isinstance(obj,dict):
        return {k.replace('_eur','_usd'):json_usd(v,k,name,path+'.'+k) for k,v in obj.items()}
    if isinstance(obj,list):return [json_usd(v,key,name,path+f'[{i}]') for i,v in enumerate(obj)]
    monetary=bool(re.search(r'(^|_)eur($|_)',key))
    if name=='cost_uncertainty_metrics.json':
        monetary |= path.startswith('.natural_pairwise_absolute_change.') or key in ['overlap_low','overlap_high','overlap_width']
    if name=='diagnostic_summary.json':
        monetary |= path.startswith('.controlled_coefficient_decomposition_per_10pp.') or key in ['efc_original_response_max_error','coefficient_identity_error']
    if name=='efc_primary_natural_year_diagnostic.json':
        monetary |= key in ['cost_scale','central_5_95_span','full_span','direct_vs_difference_identity_max_error']
    if isinstance(obj,(float,int)) and not isinstance(obj,bool) and monetary:
        JSON_CHANGES.append({'file':name,'path':path,'before':obj,'after':obj*USD_PER_EUR})
        return obj*USD_PER_EUR
    if isinstance(obj,str) and key in ['unit','units','reporting_currency']:return obj.replace('EUR','USD')
    return obj

def main():
    base=BASE/'source_data';out=SOURCE_DATA_DIR;audit=[]
    for p in sorted(base.rglob('*')):
        if not p.is_file():continue
        q=out/p.relative_to(base);q.parent.mkdir(parents=True,exist_ok=True)
        if p.name.endswith(('.csv','.csv.gz')):
            d=pd.read_csv(p);before=d.copy(deep=True);cols=columns_to_scale(d,p.name);rowscale={}
            for c in cols:
                d[c]=pd.to_numeric(d[c],errors='raise')*USD_PER_EUR
            if p.name=='figure2f_risk_accounting.csv':
                mask=d.metric.isin(['full_span','q95_q05_span','cost_improvement']);d.loc[mask,'value']*=USD_PER_EUR;rowscale['value']=mask.tolist()
            elif 'unit' in d and 'value' in d:
                mask=d.unit.astype(str).str.contains('EUR',regex=False);d.loc[mask,'value']*=USD_PER_EUR;d.loc[mask,'unit']=d.loc[mask,'unit'].str.replace('EUR','USD',regex=False);rowscale['value']=mask.tolist()
            renamed={c:c.replace('_eur','_usd') for c in cols if '_eur' in c}
            d=d.rename(columns=renamed)
            d.to_csv(q,index=False,float_format='%.17g')
            for c in cols:
                assert np.allclose(d[renamed.get(c,c)],pd.to_numeric(before[c])*USD_PER_EUR,rtol=2e-15,atol=0,equal_nan=True),(p,c)
            for c in before:
                if c in cols or c in rowscale or c=='unit':continue
                assert d[c].equals(before[c]),(p,c)
            audit.append(dict(path=str(p.relative_to(base)),rows=len(d),columns=renamed,
                              monetary_columns=cols,rowwise=rowscale,
                              original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),usd_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
        elif p.suffix=='.json':
            j=json.loads(p.read_text());q.write_text(json.dumps(json_usd(j,name=p.name),ensure_ascii=False,indent=2)+'\n')
        elif p.suffix not in {'.tex','.md'}:shutil.copy2(p,q)
    # Include supplementary-figure inputs missing from the older compact package.
    supplemental={
      'duration_boundary_summary.csv':REV/'robustness_evidence/duration_boundary_summary.csv',
      'factorial_cell_summary.csv':REV/'recharge_factorial/factorial_cell_summary.csv',
      'factorial_contrast_summary.csv':REV/'recharge_factorial/factorial_contrast_summary.csv',
      'cost_comparator_scenario_ensemble.csv':REV/'cost_uncertainty_v62/results/cost_comparator_scenario_ensemble.csv',
      'recharge_efc_records.csv':REV/'revision_issue1_v67/source_data/recharge_efc_records.csv',
      'price_channel_caes24_summary.csv':REV/'price_channel/empirical_summary_v0_1/price_channel_caes24_summary.csv',
      's3b_caes24_summary.csv':REV/'price_channel/empirical_reoptimized_summary_v0_1/s3b_caes24_summary.csv',
      'discount_duration_cost_scale.csv':REV/'figure_work/supplementary_figures_v62/source_data/discount_duration_cost_scale.csv',
    }
    for name,p in supplemental.items():
        if not p.exists():continue
        d=pd.read_csv(p);cols=columns_to_scale(d,name)
        for c in cols:d[c]*=USD_PER_EUR
        renamed={c:c.replace('_eur','_usd') for c in cols};d=d.rename(columns=renamed)
        q=out/'supplementary_analyses'/name;q.parent.mkdir(exist_ok=True);d.to_csv(q,index=False,float_format='%.17g')
        audit.append(dict(path='supplementary_analyses/'+name,rows=len(d),columns=renamed,monetary_columns=cols,original_path=str(p.relative_to(ROOT))))
    library=pd.read_csv(ROOT/'dataset/M2_LDES_duration_year_parameter_library_v0_2_currency_provenance.csv')
    # New reporting aliases use the established source-aware EUR column, so CNY
    # anchors never use the obsolete USD-model-equivalent/7.2 route.
    money=[c for c in library if re.search(r'_eur_',c) and not c.startswith('reporting_fx_')]
    for c in money:library[c.replace('_eur_','_usd_reporting_')]=library[c]*USD_PER_EUR
    library['reporting_currency_v71']='USD';library['reporting_fx_cny_per_usd_v71']=CNY_PER_USD
    library.to_csv(out/'cost_sources/technology_parameters_usd_v71.csv',index=False,float_format='%.17g')
    (WORK/'qa/source_data_conversion.json').write_text(json.dumps({'factor':USD_PER_EUR,'cny_per_usd':CNY_PER_USD,'tables':audit,'json_monetary_fields':JSON_CHANGES},indent=2)+'\n')
    print('Converted',len(audit),'tables and source-aware technology catalogue')

if __name__=='__main__':main()
