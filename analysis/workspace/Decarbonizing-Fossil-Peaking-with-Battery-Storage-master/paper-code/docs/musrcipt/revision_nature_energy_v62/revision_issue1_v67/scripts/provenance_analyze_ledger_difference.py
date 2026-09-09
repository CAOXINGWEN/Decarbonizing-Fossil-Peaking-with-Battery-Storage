"""Read-only response-ledger diagnosis with the original four model specifications."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
REV=ROOT/'Decarbonizing-Fossil-Peaking-with-Battery-Storage-master/paper-code/docs/musrcipt/revision_nature_energy_v62'
SOURCE=REV/'joint_choice/natural_caes24_s20_fixed_highs/natural_focal_arm_results.csv.gz'
OLD=REV/'recovery_window_incremental_v65/analyze_recovery_window_incremental.py'

def main():
    spec=importlib.util.spec_from_file_location('original_recovery_analysis',OLD)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    frame=pd.read_csv(REV/'recovery_window_incremental_v65/source_data/analysis_panel_132_country_years.csv')
    arms=pd.read_csv(SOURCE)
    arms['efc_value_eur']=(arms.energy_value_eur_mwh*arms.total_discharge_mwh
        +1000*arms.firm_value_eur_kw_year*arms.portfolio_efc_mw
        -arms.base_fixed_cost_eur-arms.tail_fixed_cost_eur-arms.base_vom_cost_eur
        -arms.tail_vom_cost_eur-arms.charge_price_eur_mwh*arms.total_charge_mwh)
    arms['system_cost_rebuilt_eur']=(arms.base_fixed_cost_eur+arms.tail_fixed_cost_eur
        +arms.base_vom_cost_eur+arms.tail_vom_cost_eur+arms.charge_price_eur_mwh*arms.total_charge_mwh
        +arms.energy_value_eur_mwh*arms.unserved_eens_mwh
        +1000*arms.firm_value_eur_kw_year*arms.residual_peak_mw)
    index=['country','year']; a=arms[arms.config_id.eq('caes_24h')].set_index(index)
    b=arms[arms.config_id.eq('li_lfp_4h')].set_index(index).reindex(a.index)
    for col in ['base_power_mw','firm_value_eur_kw_year','energy_value_eur_mwh','base_task_mwh','tail_task_mwh']:
        assert np.allclose(a[col],b[col],rtol=0,atol=1e-8),col
    metrics=a.index.to_frame(index=False)
    p=a.base_power_mw.to_numpy(); vf=a.firm_value_eur_kw_year.to_numpy()
    metrics['efc_value']=(a.efc_value_eur.to_numpy()-b.efc_value_eur.to_numpy())/(1000*p)
    metrics['system_cost_value']=(b.system_cost_eur.to_numpy()-a.system_cost_eur.to_numpy())/(1000*p)
    metrics['accounting_difference']=metrics.efc_value-metrics.system_cost_value
    metrics['incremental_efc_credit']=vf*(a.portfolio_efc_mw.to_numpy()-b.portfolio_efc_mw.to_numpy())/p
    metrics['avoided_residual_peak_credit']=vf*(b.residual_peak_mw.to_numpy()-a.residual_peak_mw.to_numpy())/p
    metrics['accounting_difference_from_capacity_terms']=metrics.incremental_efc_credit-metrics.avoided_residual_peak_credit
    frame=frame.merge(metrics,on=index,validate='1:1')
    frame.to_csv(OUT/'paired_ledger_panel.csv',index=False)
    result={'n':len(frame),'countries':frame.country.nunique(),'bootstrap_draws':mod.BOOTSTRAP_DRAWS,
        'identity_max_error_eur_kw_year':float(abs(frame.accounting_difference-frame.accounting_difference_from_capacity_terms).max()),
        'system_cost_rebuild_max_error_eur':float(abs(arms.system_cost_eur-arms.system_cost_rebuilt_eur).max()),
        'efc_original_response_max_error':float(abs(frame.efc_value-frame.natural_caes24_minus_lfp4_eur_kw_year).max()),
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'original_analysis_sha256':hashlib.sha256(OLD.read_bytes()).hexdigest()}
    comparisons=[]; coefficients=[]; all_predictions=[]
    for response in ['efc_value','system_cost_value','accounting_difference']:
        mod.OUTCOME=response
        predictions=mod.leave_one_country_out(frame)
        predictions['response']=response; all_predictions.append(predictions)
        countries=sorted(frame.country.unique())
        rng=np.random.default_rng(mod.SEED+991)
        draws=rng.choice(countries,size=(mod.BOOTSTRAP_DRAWS,len(countries)),replace=True)
        counts=np.column_stack([(draws==c).sum(axis=1) for c in countries])
        reference_boot_r2=None
        reference_loco_r2=None
        for model,predictors in mod.MODELS.items():
            beta,within_r2=mod.fit_within(frame,predictors)
            _,bootstrap_beta=mod.coefficient_bootstrap(frame,predictors,model)
            group=predictions[predictions.model_id.eq(model)]
            pred_summary=mod.summarize_prediction(group)
            sse=[]; sst=[]
            for c in countries:
                g=group[group.country.eq(c)]
                y=g.observed_country_centred_value_eur_kw_year.to_numpy()
                yh=g.predicted_country_centred_value_eur_kw_year.to_numpy()
                sse.append(np.sum((y-yh)**2)); sst.append(np.sum(y**2))
            boot_r2=1-(counts@np.array(sse))/(counts@np.array(sst))
            if model=='longest_event':
                reference_boot_r2=boot_r2.copy()
                reference_loco_r2=pred_summary['loco_r2']
            delta_low,delta_high=np.quantile(boot_r2-reference_boot_r2,[.025,.975])
            lo,hi=np.quantile(boot_r2,[.025,.975])
            comparisons.append({'response':response,'model':model,'within_r2':within_r2,**pred_summary,
                'loco_r2_ci95_low':lo,'loco_r2_ci95_high':hi,
                'delta_loco_r2_vs_longest':pred_summary['loco_r2']-reference_loco_r2,
                'delta_loco_r2_ci95_low':delta_low,'delta_loco_r2_ci95_high':delta_high})
            for j,predictor in enumerate(predictors):
                lo,hi=np.nanquantile(bootstrap_beta[:,j],[.025,.975])
                divisor=10 if predictor=='natural_zero_recovery_window_share' else 1
                coefficients.append({'response':response,'model':model,'predictor':predictor,
                    'coefficient':beta[j],'ci95_low':lo,'ci95_high':hi,
                    'reported_unit':'per 10 percentage points' if divisor==10 else 'per predictor unit',
                    'reported_coefficient':beta[j]/divisor,'reported_ci95_low':lo/divisor,'reported_ci95_high':hi/divisor,
                    'valid_bootstrap_draws':int(np.isfinite(bootstrap_beta[:,j]).sum())})
        print('Completed',response,flush=True)
    comp=pd.DataFrame(comparisons); coef=pd.DataFrame(coefficients)
    comp.to_csv(OUT/'model_comparison_both_ledgers_and_difference.csv',index=False)
    coef.to_csv(OUT/'coefficient_comparison_both_ledgers_and_difference.csv',index=False)
    pd.concat(all_predictions).to_csv(OUT/'loco_predictions.csv',index=False)
    c=coef[coef.model.eq('joint_controlled')&coef.predictor.eq('natural_zero_recovery_window_share')].set_index('response')
    result['controlled_coefficient_decomposition_per_10pp']=c[['reported_coefficient','reported_ci95_low','reported_ci95_high']].to_dict('index')
    result['coefficient_identity_error']=float(abs(c.loc['efc_value','reported_coefficient']-c.loc['system_cost_value','reported_coefficient']-c.loc['accounting_difference','reported_coefficient']))
    result['diagnostic_difference_fraction_of_efc_coefficient']=float(c.loc['accounting_difference','reported_coefficient']/c.loc['efc_value','reported_coefficient'])
    result['interpretation_limit']='The fraction is an algebraic decomposition of fitted slopes, not a causal attribution or fraction of explained variance.'
    (OUT/'diagnostic_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
