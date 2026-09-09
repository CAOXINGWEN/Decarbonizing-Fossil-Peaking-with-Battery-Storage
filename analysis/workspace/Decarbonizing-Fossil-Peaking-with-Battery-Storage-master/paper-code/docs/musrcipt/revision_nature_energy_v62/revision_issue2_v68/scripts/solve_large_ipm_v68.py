"""Independently solve the two large residual LPs with the HiGHS interior-point method."""
import concurrent.futures
import json
import time

from run_capacity_v68 import WORK,HERE,module,sha,write_json,scenarios


def trial(design):
    base=module('v68_ipm_'+design['country'],HERE/'capacity_model_v68.py')
    original=base.linprog
    def ipm(*args,**kwargs):
        kwargs['method']='highs-ipm'
        kwargs['options']={**kwargs['options'],'ipm_optimality_tolerance':1e-10}
        return original(*args,**kwargs)
    base.linprog=ipm
    started=time.time()
    summary,capacities,metrics=base.solve_country_design_highs(
        design['country'],scenarios(design['country'],design['step'])[design['ensemble']],
        base.load_technologies(2035),base.load_value_scenario(20),
        design['risk'],design['mode'],design['step'],1)
    for name in ('maximum_task_balance_residual','maximum_shared_recharge_violation',
                 'maximum_state_balance_residual_mwh','maximum_inequality_violation',
                 'maximum_equality_residual','maximum_nonnegativity_violation','maximum_stationarity_residual'):
        assert abs(summary[name])<=1e-6,(name,summary[name])
    assert summary['objective_identity_error_eur_per_normalized_mw']<=.02
    assert abs(summary['primal_dual_objective_gap_eur'])<=.02
    manifest=json.loads((WORK/'scenario_cache/manifest.json').read_text())
    summary['solver_algorithm']='highs-ipm-with-crossover'
    result={'fingerprint':manifest['fingerprint'],'design':design,'summary':summary,
            'capacities':capacities,'scenario_metrics':metrics,'elapsed_seconds':time.time()-started,
            'execution_script_sha256':sha(HERE/'solve_large_ipm_v68.py')}
    write_json(WORK/'solver_trials'/(design['key']+'.json'),result)
    return {'key':design['key'],'seconds':round(result['elapsed_seconds'],1),
            'objective':summary['solver_objective_eur_per_normalized_mw'],
            'power':summary['total_power_capacity_per_normalized_mw']}


def main():
    (WORK/'solver_trials').mkdir(exist_ok=True)
    manifest=json.loads((WORK/'scenario_cache/manifest.json').read_text())
    selected=[d for d in manifest['designs'] if d['country'] in ('BE','DE') and d['step']==2
              and d['ensemble']=='combined' and d['risk']==.5 and d['mode']=='diversified']
    assert len(selected)==2
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(trial,selected):
            print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
