"""Build v67 EFC sources from immutable dispatch and original cost parameters."""
from pathlib import Path
import importlib.util, inspect, json, sys, shutil, hashlib, ast, types
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parents[1]
ROOT=WORK.parent
SRC=WORK/'source_data'
NAT=WORK/'natural_results'
VF=103.79921785880974
THRESHOLD=5.2960380996609935

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def natural():
    old=ROOT/'weather_year_parity_v42/results'
    for name in ['pecd_v42_75_window_arm_results.csv.gz','pecd_v42_75_window_event_geometry.csv','pecd_v42_75_window_prepared_summary.csv']:
        shutil.copy2(old/name,NAT/name)
    efc=pd.read_csv(SRC/'natural_75_years_both_ledgers_by_cost_year.csv')
    mapping=efc[efc.cost_year.eq(2025)].set_index('window_start_year').efc_value_2025
    original_source=(ROOT/'weather_year_parity_v42/compute_pecd_v42_75y_parity.py').read_text()
    tree=ast.parse(original_source)
    code='\n\n'.join(ast.get_source_segment(original_source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['cost_path','summarize'])
    original='''values["value_2025_eur_kw_year"] = (
        values.system_cost_eur_li_lfp_4h - values.system_cost_eur_caes_24h
    ) / (1000.0 * values.fixed_common_power_mw)'''
    assert code.count(original)==1
    code=code.replace(original,'values["value_2025_eur_kw_year"] = values.window_start_year.map(EFC_VALUES)')
    ns=dict(pd=pd,np=np,json=json,RESULTS=NAT,COST_FILE=ROOT/'cost_and_decomposition/cost_catalogue_2025_2035_long.csv',EFC_VALUES=mapping,
        write_json=lambda path,data:path.write_text(json.dumps(data,indent=2)),
        storage_model=types.SimpleNamespace(output_csv_gz=lambda frame,path:frame.to_csv(path,index=False)))
    exec(compile(code,str(__file__),'exec'),ns);ns['summarize']()
    summary=json.loads((NAT/'pecd_v42_75y_parity_summary.json').read_text())
    summary.update(value_ledger='EFC service valuation on unchanged system-cost-optimal dispatch',crossing_interpretation='service valuation crossover, not optimal technology switch')
    (NAT/'pecd_v42_75y_parity_summary.json').write_text(json.dumps(summary,indent=2))
    c=module('v67_cost_summary',ROOT/'cost_uncertainty_v62/run_v62_cost_uncertainty.py')
    ensemble=pd.read_csv(ROOT/'cost_uncertainty_v62/results/cost_comparator_scenario_ensemble.csv')
    pairs=pd.read_csv(NAT/'pecd_v42_75_window_all_pairs.csv.gz')
    cs,metrics=c.build_summary(ensemble,summary['natural_q95_q05_span_eur_kw_year'],pairs.absolute_weather_value_change_eur_kw_year.to_numpy())
    c.RESULTS=SRC;c.write_macros(cs,metrics,{'central_comparator_eur_kw_year':THRESHOLD})
    cs.to_csv(SRC/'cost_uncertainty_summary.csv',index=False)
    (SRC/'cost_uncertainty_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Natural EFC:',summary['natural_q95_q05_span_eur_kw_year'],summary['worst_matched_decade'],flush=True)

def arm_components(d,regime):
    d=d.copy()
    if regime=='Europe':
        d=d.rename(columns={'country':'system_id','year':'observation_year'})
        cost='system_cost_eur'; firm='residual_firm_cost_eur'; energy='residual_energy_cost_eur'; charge='charge_opportunity_cost_eur'
    else:
        d=d.rename(columns={'case_id':'system_id','case_year':'observation_year'})
        cost='S0_system_cost_eur';firm='residual_firm_cost_eur';energy='S0_residual_energy_cost_eur';charge='S0_charge_cost_eur'
    d['regime']=regime
    d['inference_block']=d.system_id if regime!='Japan' else d.system_id.str.split('-').str[1]
    d['fixed_cost']=-(d.base_fixed_cost_eur+d.tail_fixed_cost_eur)
    d['variable_om']=-(d.base_vom_cost_eur+d.tail_vom_cost_eur)
    d['charging']=-d[charge];d['energy_service']=-d[energy]
    d['firm_service']=1000*VF*d.portfolio_efc_mw
    d['efc_score']=d[['fixed_cost','variable_om','charging','energy_service','firm_service']].sum(axis=1)
    assert np.allclose(d.efc_score,-d[cost]+d[firm]+1000*VF*d.portfolio_efc_mw,rtol=1e-10,atol=1e-4)
    # The common annual unserved-energy constant cancels between technology arms.
    keys=['regime','system_id','observation_year','chronology']
    components=['fixed_cost','variable_om','charging','energy_service','firm_service','efc_score']
    comp=d[d.arm_type.eq('extra_lfp4_comparator')][keys+components].copy()
    out=d[d.arm_type.eq('ldes_candidate')].merge(comp,on=keys,suffixes=('','_lfp'),validate='m:1')
    for col in components:out[col]=(out[col]-out[col+'_lfp'])/(1000*out.tail_power_mw)
    return out

def build_controlled():
    h=module('v67_harmonized',ROOT/'figure_work/build_v58_harmonized_parity.py')
    b=module('v67_summary_helpers',ROOT/'figure_work/build_v58_figure_sources.py')
    movement=h.relative_cost_movement()
    europe=arm_components(pd.read_csv(ROOT/'portfolio_model/full_132cy_fixed_s20_central/portfolio_arm_results.csv.gz'),'Europe')
    cross=pd.read_csv(ROOT/'cross_scale_cases/summary_v0_2/cross_scale_arms.csv.gz')
    combined=pd.concat([europe]+[arm_components(cross[cross.application_case.eq(r.lower())],r) for r in ['China','Japan']],ignore_index=True)
    keys=['regime','system_id','observation_year','duration_h']
    selected=combined.sort_values(keys+['chronology','efc_score'],ascending=[True]*5+[False],kind='stable').groupby(keys+['chronology'],as_index=False).first()
    selected=selected.rename(columns={'efc_score':'benefit_eur_kw_year','config_id':'dominant_config_id','tech_id':'dominant_tech_id'})
    fields=['benefit_eur_kw_year','dominant_config_id','dominant_tech_id']
    records=selected.pivot(index=keys,columns='chronology',values=fields)
    records.columns=['_'.join(x) for x in records.columns];records=records.reset_index()
    records['signed_sequencing_effect_eur_kw_year']=records.benefit_eur_kw_year_clustered-records.benefit_eur_kw_year_dispersed
    records['sequencing_shift_eur_kw_year']=records.signed_sequencing_effect_eur_kw_year.abs()
    records['matched_cost_movement_eur_kw_year']=.5*(records.dominant_config_id_clustered.map(movement)+records.dominant_config_id_dispersed.map(movement))
    records['parity_ratio']=records.sequencing_shift_eur_kw_year/records.matched_cost_movement_eur_kw_year
    records['above_parity']=records.parity_ratio.ge(1)
    records['inference_block']=records.system_id
    j=records.regime.eq('Japan');records.loc[j,'inference_block']=records.loc[j,'system_id'].str.split('-').str[1]
    records.to_csv(SRC/'all_duration_efc_records.csv.gz',index=False)
    h.aggregate(records).to_csv(SRC/'figure5_system_duration_exposure.csv',index=False)
    duration=b.duration_summary(records);duration.to_csv(SRC/'figure5_duration_response_summary.csv',index=False)
    focal=records[records.duration_h.eq(24)].copy()
    # Focal 24 h comparison must retain the preselected CAES configuration.
    assert focal.dominant_config_id_clustered.eq('caes_24h').all() and focal.dominant_config_id_dispersed.eq('caes_24h').all()
    phys=pd.read_csv(ROOT/'figure_work/source_data_v58/figure3_24h_value_cycle_records.csv')
    focal=focal.merge(phys[['regime','system_id','observation_year','cycles_dispersed','cycles_clustered','cycle_effect']],on=['regime','system_id','observation_year'],validate='1:1')
    assert len(focal)==185
    focal.to_csv(SRC/'figure3_24h_value_cycle_records.csv',index=False);focal.to_csv(SRC/'figure3_24h_value_records.csv',index=False)
    ck=['regime','system_id','observation_year'];components=['fixed_cost','variable_om','charging','energy_service','firm_service']
    caes=combined[combined.config_id.eq('caes_24h')]
    c=caes[caes.chronology.eq('clustered')].set_index(ck);d=caes[caes.chronology.eq('dispersed')].set_index(ck)
    decomposition=(c[components]-d[components]).reset_index()
    decomposition['total']=(c.efc_score-d.efc_score).to_numpy()
    decomposition['component_sum']=decomposition[components].sum(axis=1)
    assert np.allclose(decomposition.total,decomposition.component_sum,atol=1e-8)
    decomposition.to_csv(SRC/'figure3_value_component_effects.csv.gz',index=False)
    for name in ['figure3_representative_dispatch_profiles.csv.gz','figure3_representative_dispatch_selection.csv']:
        shutil.copy2(ROOT/'figure_work/figure3_relayout_v62/source_data'/name,SRC/name)
    shutil.copy2(ROOT/'figure_work/source_data_v58/figure3_recovery_window_records.csv',SRC/'figure3_recovery_window_records.csv')
    summaries=[];rr=[]
    for regime in ['Europe','China','Japan']:
        for multiplier in [1,2,4]:
            if regime=='Europe':
                directory={1:'full_132cy_fixed_s20_central',2:'recharge_intervention_fixed_s20_r2',4:'recharge_intervention_fixed_s20_r4'}[multiplier]
                arms=pd.read_csv(ROOT/'portfolio_model'/directory/'portfolio_arm_results.csv.gz')
            else:arms=pd.read_csv(ROOT/'cross_scale_recharge_v58'/f'{regime.lower()}_r{multiplier}'/'cross_scale_results.csv.gz')
            a=arm_components(arms,regime);a=a[a.config_id.eq('caes_24h')]
            w=a.pivot(index=ck+['inference_block'],columns='chronology',values=['efc_score','tail_equivalent_full_cycles'])
            w.columns=['_'.join(x) for x in w.columns];w=w.reset_index()
            w['chronology_effect']=w.efc_score_clustered-w.efc_score_dispersed
            w['cycle_effect']=w.tail_equivalent_full_cycles_clustered-w.tail_equivalent_full_cycles_dispersed
            w['recharge_multiplier']=multiplier;rr.append(w)
            point,low,high=b.block_bootstrap(w,'chronology_effect','inference_block',seed=5803)
            cp,cl,ch=b.block_bootstrap(w,'cycle_effect','inference_block',seed=5803)
            summaries.append(dict(regime=regime,recharge_multiplier=multiplier,n_system_years=len(w),n_blocks=w.inference_block.nunique(),country_block_mean_effect_eur_kw_year=point,ci95_low=low,ci95_high=high,median_effect_eur_kw_year=w.chronology_effect.median(),country_block_mean_cycle_effect=cp,cycle_ci95_low=cl,cycle_ci95_high=ch,median_cycle_effect=w.cycle_effect.median()))
    recharge=pd.concat(rr,ignore_index=True);recharge.to_csv(SRC/'recharge_efc_records.csv',index=False)
    summary=pd.DataFrame(summaries)
    baseline=summary[summary.recharge_multiplier.eq(1)].set_index('regime').country_block_mean_effect_eur_kw_year
    summary['effect_retained_pct']=100*summary.country_block_mean_effect_eur_kw_year/summary.regime.map(baseline)
    summary.to_csv(SRC/'figure4_recharge_intervention_summary.csv',index=False)
    headlines=focal.groupby('regime').agg(n=('above_parity','size'),hits=('above_parity','sum'),absolute_median=('sequencing_shift_eur_kw_year','median'),signed_median=('signed_sequencing_effect_eur_kw_year','median'))
    assert headlines.hits.to_dict()=={'China':26,'Europe':68,'Japan':11},headlines
    headlines.to_csv(SRC/'efc_headlines.csv')
    print('EFC focal and all-duration records ready',headlines.to_dict(),flush=True)

def surface():
    m=module('v67_surface_postprocess',ROOT/'cluster_intensity_surface/postprocess_v60_cluster_intensity_surface.py')
    m.RUNS=ROOT/'cluster_intensity_surface/runs_efc';m.OUT=WORK/'surface'
    source=inspect.getsource(m.main)
    old='''pd.read_csv(
        ROOT
        / "figure_work/source_data_v58/harmonized_parity/three_regime_system_year_duration_records.csv.gz"
    )'''
    assert source.count(old)==1
    source=source.replace(old,'pd.read_csv(EFC_BASELINE)')
    source=source.replace('objective_aligned_surface_vs_harmonized_4d','efc_surface_vs_harmonized_4d')
    source=source.replace('exact solver-system-cost difference versus additional 4 h LFP','EFC service-value difference on unchanged system-cost-optimal dispatch')
    source=source.replace('operational diagnostic only','primary service valuation')
    m.EFC_BASELINE=SRC/'all_duration_efc_records.csv.gz'
    exec(compile(source,str(__file__),'exec'),m.__dict__);m.main()
    # Check frozen EFC surface consistency at the shared 4 d focal endpoint.
    surf=pd.read_csv(m.OUT/'v60_cluster_intensity_system_year.csv.gz')
    focal=pd.read_csv(SRC/'figure3_24h_value_records.csv')
    f=surf[surf.target_cluster_days.eq(4)&surf.duration_h.eq(24)].merge(focal,on=['regime','system_id','observation_year'],validate='1:1')
    err=float(abs(f.sequencing_effect_eur_kw_year-f.signed_sequencing_effect_eur_kw_year).max())
    assert len(surf)==31080 and len(f)==185 and err<.01,(len(surf),len(f),err)
    (WORK/'qa/surface_alignment.json').write_text(json.dumps({'rows':len(surf),'focal_4d_max_error':err},indent=2))

if __name__=='__main__':
    natural();build_controlled();surface()
