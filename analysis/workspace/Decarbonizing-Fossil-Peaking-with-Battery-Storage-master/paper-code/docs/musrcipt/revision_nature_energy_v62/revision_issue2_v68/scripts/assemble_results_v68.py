"""Audit 540 independent solves and rebuild all procurement tables and macros."""
from pathlib import Path
import ast
import contextlib
import io
import json
import shutil
import textwrap
import types

import numpy as np
import pandas as pd

from run_capacity_v68 import WORK, REV, CAP, module, sha, write_json

KEYS = ['country','planning_step_hours','ensemble','risk_weight','mode']
HELPER = WORK/'scripts/frozen_helpers'


def helper(relative):
    source = CAP/relative
    target = HELPER/source.name
    HELPER.mkdir(exist_ok=True)
    if target.exists():
        assert target.read_bytes() == source.read_bytes()
    else:
        shutil.copy2(source,target)
    return target


def old_designs():
    frames = []
    for step, relative in ((2,'resolution_convergence_v64/chronology_2h'),
                           (4,'chronology_procurement_validation/resolution4h_results'),
                           (6,'chronology_procurement_validation/results')):
        d = pd.read_csv(CAP/relative/'design_summary.csv')
        frames.append(d)
    for relative in ('resolution4h_all15','results'):
        d = pd.read_csv(CAP/relative/'design_summary.csv')
        d = d.loc[d['mode'].ne('diversified')].copy()
        d['ensemble'] = 'combined'
        frames.append(d)
    return pd.concat(frames,ignore_index=True)


def qa_for(summaries, conservation):
    energy = conservation.groupby(['country','year'])[
        ['task_energy_normalized_mwh','recharge_energy_normalized_mwh']].agg(lambda x:x.max()-x.min())
    fields = {
        'maximum_objective_identity_error_eur_per_normalized_mw':'objective_identity_error_eur_per_normalized_mw',
        'maximum_task_balance_residual':'maximum_task_balance_residual',
        'maximum_shared_recharge_violation':'maximum_shared_recharge_violation',
        'maximum_state_balance_residual_mwh':'maximum_state_balance_residual_mwh',
        'maximum_inequality_violation':'maximum_inequality_violation',
        'maximum_equality_residual':'maximum_equality_residual',
        'maximum_nonnegativity_violation':'maximum_nonnegativity_violation',
        'maximum_stationarity_residual':'maximum_stationarity_residual',
        'maximum_absolute_primal_dual_gap_eur':'primal_dual_objective_gap_eur',
    }
    q = {key:float(summaries[value].abs().max()) for key,value in fields.items()}
    q.update({
        'all_solver_statuses_optimal':bool(summaries.solver_status.eq('optimal').all()),
        'country_count':int(summaries.country.nunique()),'countries':sorted(summaries.country.unique().tolist()),
        'natural_country_year_count':len(conservation[['country','year']].drop_duplicates()),
        'design_count':len(summaries), 'planning_step_hours':int(summaries.planning_step_hours.iloc[0]),
        'maximum_chronology_task_energy_residual':float(energy.task_energy_normalized_mwh.max()),
        'maximum_chronology_recharge_energy_residual':float(energy.recharge_energy_normalized_mwh.max()),
        'capacity_domain':'P >= 0',
    })
    passed = q['all_solver_statuses_optimal']
    for key,value in q.items():
        if key.startswith('maximum_'):
            tol = 0.02 if ('objective' in key or 'dual_gap' in key) else 1e-6
            if 'chronology_' in key:
                tol = 1e-10
            passed = passed and value <= tol
    q['status'] = 'pass' if passed else 'fail'
    return q


def diversification(summaries):
    rows=[]
    for (country,risk),group in summaries.groupby(['country','risk_weight']):
        combined=group.loc[group['mode'].eq('diversified')].iloc[0]
        singles=group.loc[group['mode'].ne('diversified')]
        best=singles.sort_values(['risk_adjusted_cost_eur_per_normalized_mw','mode'],kind='stable').iloc[0]
        row={'country':country,'risk_weight':risk,'best_single_mode':best['mode'],
             'diversified_selected_technology_count':int(combined.selected_technology_count),
             'diversified_total_power_capacity_percent_mean_gas':100*float(combined.total_power_capacity_per_normalized_mw),
             'best_single_total_power_capacity_percent_mean_gas':100*float(best.total_power_capacity_per_normalized_mw)}
        for label,column in [('mean','mean_total_cost'),('worst','worst_total_cost'),
                             ('cvar90','cvar90_total_cost'),('risk_adjusted','risk_adjusted_cost')]:
            row[f'diversified_minus_best_single_{label}_cost_eur_per_normalized_mw']=float(
                combined[column+'_eur_per_normalized_mw']-best[column+'_eur_per_normalized_mw'])
        assert row['diversified_minus_best_single_risk_adjusted_cost_eur_per_normalized_mw'] <= .02
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_headline(summaries,capacities,conservation,comparisons):
    # Reuse the frozen analysis exactly, removing its obsolete bound diagnostic.
    path=helper('merge_v62_capacity_expansion_shards.py')
    original=path.read_text()
    start=original.index('    diversified_capacity =')
    end=original.index('    summaries.to_csv(',start)
    body=textwrap.dedent(original[start:end])
    old='''    "minimum_capacity_upper_bound_slack": float(
        summaries["maximum_capacity_upper_bound_slack"].min()
    ),
'''
    assert body.count(old)==1
    body=body.replace(old,'')
    ns=dict(pd=pd,np=np,Any=object,summaries=summaries,capacities=capacities,conservation=conservation,
            comparisons=comparisons,expected_scenarios=264,ACTIVE_TOLERANCE=1e-6,
            MATERIAL_POWER_CHANGE_PERCENTAGE_POINTS=1.)
    exec(compile(body,str(path),'exec'),ns)
    return ns['procurement_shifts'],ns['headline']


def main():
    manifest=json.loads((WORK/'scenario_cache/manifest.json').read_text())
    summaries=[]; capacities=[]; scenarios=[]
    for design in manifest['designs']:
        path=WORK/'designs'/(design['key']+'.json')
        data=json.loads(path.read_text())
        assert data['fingerprint']==manifest['fingerprint']
        common={'ensemble':design['ensemble'],'planning_step_hours':design['step']}
        summaries.append({**data['summary'],**common})
        capacities.extend({**row,**common} for row in data['capacities'])
        scenarios.extend({**row,**common} for row in data['scenario_metrics'])
    s=pd.DataFrame(summaries); c=pd.DataFrame(capacities); m=pd.DataFrame(scenarios)
    assert len(s)==540 and not s.duplicated(KEYS).any()
    assert len(s.loc[s['mode'].eq('diversified')])==360
    assert len(c)==1260 and len(c)==int(s.technology_count.sum())
    assert len(m)==7128 and len(m)==int(s.scenario_count.sum())
    assert not c.duplicated(KEYS+['config_id']).any()
    assert not m.duplicated(KEYS+['scenario_id']).any()
    counts=m.groupby(KEYS)['year'].nunique().rename('count')
    count_check=s.set_index(KEYS)['natural_year_count'].sort_index()-counts.sort_index()
    assert count_check.eq(0).all()
    old=old_designs()
    pair=s.merge(old,on=KEYS,validate='one_to_one',suffixes=('_v68','_v67'))
    pair['objective_change_eur']=pair.solver_objective_eur_per_normalized_mw_v68-pair.solver_objective_eur_per_normalized_mw_v67
    pair['total_power_change_pp']=100*(pair.total_power_capacity_per_normalized_mw_v68-pair.total_power_capacity_per_normalized_mw_v67)
    pair['energy_capacity_change']=pair.total_energy_capacity_mwh_per_normalized_mw_v68-pair.total_energy_capacity_mwh_per_normalized_mw_v67
    assert pair.objective_change_eur.max() <= .02
    source=WORK/'source_data'; source.mkdir(exist_ok=True)
    s.to_csv(source/'all_540_design_summary.csv',index=False)
    c.to_csv(source/'all_capacity_mix.csv',index=False)
    m.to_csv(source/'all_scenario_metrics.csv.gz',index=False)
    pair.to_csv(source/'all_540_before_after.csv',index=False)
    conservation=pd.read_csv(WORK/'scenario_cache/chronology_conservation.csv')
    chron=module('v68_chronology_analysis',helper('run_v62_chronology_procurement_counterfactual.py'))
    macros=module('v68_chronology_macros',helper('merge_v62_chronology_procurement_shards.py'))
    qas={}
    for step in (2,4,6):
        cons=conservation.loc[conservation.planning_step_hours.eq(step)].copy()
        grid_qa=qa_for(s.loc[s.planning_step_hours.eq(step)],cons)
        assert grid_qa['status']=='pass',grid_qa
        qas[f'grid_{step}h']=grid_qa
        for kind in ('chronology','pooled'):
            root=WORK/'results'/f'{kind}_{step}h'; root.mkdir(parents=True,exist_ok=True)
            select=lambda f: f.loc[f.planning_step_hours.eq(step) & (
                f['mode'].eq('diversified') if kind=='chronology' else f.ensemble.eq('combined'))].copy()
            ss,cc,mm=map(select,(s,c,m))
            co=cons if kind=='chronology' else cons.loc[cons.chronology.ne('observed')]
            for name,frame in [('design_summary.csv',ss),('capacity_mix.csv',cc),
                               ('scenario_metrics.csv.gz',mm),('chronology_conservation.csv',co)]:
                frame.to_csv(root/name,index=False)
            qa=qa_for(ss,co)
            assert qa['status']=='pass',qa
            qa['input_fingerprint']=manifest['fingerprint']
            if kind=='chronology':
                ensemble,direct,headline=chron.build_comparisons(ss,cc)
                ensemble.to_csv(root/'ensemble_comparison.csv',index=False)
                direct.to_csv(root/'direct_chronology_effect.csv',index=False)
                macros.write_tex_macros(root/'chronology_procurement_macros.tex',headline,cc)
                write_json(root/'headline_metrics.json',headline)
            elif step in (4,6):
                comparisons=diversification(ss)
                comparisons.to_csv(root/'diversification_comparison.csv',index=False)
                shifts,headline=pooled_headline(ss,cc,co,comparisons)
                shifts.to_csv(root/'procurement_design_shifts.csv',index=False)
                write_json(root/'headline_metrics.json',headline)
            qa['output_sha256']={p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.name!='qa.json'}
            write_json(root/'qa.json',qa);qas[f'{kind}_{step}h']=qa
    for steps,relative,name in [((4,6),'compare_v62_planning_resolution.py','comparison_4h_6h'),
                                ((2,4,6),'resolution_convergence_v64/compare_v64_planning_resolutions.py','comparison_2h_4h_6h')]:
        compare=module('v68_'+name,helper(relative))
        final_output=WORK/'results'/name
        output=WORK/'results'/(name+'_building')
        if output.exists():
            shutil.rmtree(output)
        args={'output':output,'force':False}
        for step in steps:
            args[f'pooled_{step}h']=WORK/'results'/f'pooled_{step}h'
            args[f'chronology_{step}h']=WORK/'results'/f'chronology_{step}h'
        compare.parse_args=lambda:types.SimpleNamespace(**args)
        with contextlib.redirect_stdout(io.StringIO()):
            assert compare.main()==0
        if len(steps)==3:
            compare.write_macros(output/'resolution_convergence_macros.tex',
                                 json.loads((output/'headline_metrics.json').read_text()),qas['grid_2h'])
        if final_output.exists():
            shutil.rmtree(final_output)
        output.rename(final_output)
    macro=module('v68_planning_macros',helper('write_v62_planning_resolution_macros.py'))
    macro.FOUR=WORK/'results/pooled_4h';macro.SIX=WORK/'results/pooled_6h'
    macro.FOUR_CHRON=WORK/'results/chronology_4h';macro.SIX_CHRON=WORK/'results/chronology_6h'
    macro.PAIRED=WORK/'results/comparison_4h_6h';macro.OUTPUT=macro.PAIRED/'planning_resolution_macros.tex'
    original_read=macro.read
    def read_with_grid_checks(path):
        if path==macro.FOUR/'qa.json':
            return qas['grid_4h']
        if path==macro.SIX/'qa.json':
            return qas['grid_6h']
        return original_read(path)
    macro.read=read_with_grid_checks
    macro.main()
    audit={'status':'pass','design_count':540,'diversified_design_count':360,'single_technology_count':180,
           'capacity_row_count':len(c),'scenario_row_count':len(m),
           'maximum_objective_increase_eur':float(pair.objective_change_eur.max()),
           'objective_improvement_over_0_02_eur_designs':int(pair.objective_change_eur.lt(-.02).sum()),
           'power_change_over_1pp_designs':int(pair.total_power_change_pp.abs().gt(1).sum()),
           'maximum_absolute_power_change_pp':float(pair.total_power_change_pp.abs().max()),
           'natural_year_counts_verified':True,'group_qa':qas}
    write_json(WORK/'qa/model_verification.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='group_qa'},indent=2))


if __name__=='__main__':
    main()
