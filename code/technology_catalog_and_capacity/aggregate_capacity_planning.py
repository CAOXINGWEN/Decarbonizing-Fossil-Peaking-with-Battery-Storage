"""Aggregate the hydrogen-only capacity and near-optimal results."""
import json,sys,importlib.util,argparse
import numpy as np
import pandas as pd
import parameters as P
OLD=P.ROOT/'submission/Nature_Energy_v85/revision/remaining_experiments_20261103'

def capacity():
    sys.path.insert(0,str(OLD/'code'))
    spec=importlib.util.spec_from_file_location('old_capacity_aggregate',OLD/'code/aggregate.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.W=P.WORK;a.capacity()
    c=pd.read_csv(P.WORK/'tables/capacity_technology_selection.csv')
    h=c[c.config_id.eq('h2_tank_168h') & c.step.ne(1)]
    assert len(h)==360 and h.capacity_per_normalized_mw.abs().max()<1e-10
    h[h.selected].to_csv(P.WORK/'tables/capacity_selected_hydrogen_designs.csv',index=False)

def nearoptimal():
    rows=[]
    for f in sorted((P.WORK/'results/nearoptimal').glob('*/eps*.json')):
        x=json.loads(f.read_text());a=x['attempts'][-1]
        rows.append({**x['design'],**{k:x[k] for k in ['epsilon','technology','bound','capacity_bound','C_star','budget','numerical_slack','solver_status']},
            'result_json_path':str(f.relative_to(P.WORK)),
            'solution_origin':'exact verified reuse' if 'reuse_proof' in x else 'analytic certificate' if 'analytic_proof' in x else 'recomputed',
            'lp_method':a.get('lp_method',a.get('method')),'crossover':a.get('crossover',1),
            **x['audit'],**{'physical_'+k:v for k,v in x['physical'].items()}})
    d=pd.DataFrame(rows);assert len(d)==288,len(d)
    keys=['country','ensemble','risk','epsilon','technology']
    assert not d.duplicated(keys+['bound']).any()
    q=d.pivot(index=keys,columns='bound',values='capacity_bound').reset_index()
    assert len(q)==144 and np.all(q['min']<=q['max']+1e-6)
    refs=pd.read_csv(P.WORK/'tables/capacity_technology_power.csv')
    refs=refs[refs.step.eq(1)&refs['mode'].eq('diversified')].set_index(['country','ensemble','risk','config_id'])
    nesting=outside=0.
    for (cty,ens,risk,tech),g in q.groupby(['country','ensemble','risk','technology']):
        g=g.sort_values('epsilon');v=float(refs.loc[(cty,ens,risk,tech),'capacity_per_normalized_mw'])
        nesting=max(nesting,float(np.diff(g['min']).max()),float(-np.diff(g['max']).min()))
        outside=max(outside,float((g['min']-v).max()),float((v-g['max']).max()))
    checks=['maximum_equality_residual','maximum_inequality_violation','maximum_nonnegativity_violation',
            'maximum_stationarity_residual','maximum_reduced_cost_sign_violation','maximum_inequality_dual_sign_violation']
    qa={k:float(d[k].abs().max()) for k in checks}
    qa.update(status='PASS',endpoints=len(d),intervals=len(q),
        exact_reuse_endpoints=int(d.solution_origin.eq('exact verified reuse').sum()),
        maximum_interval_nesting_violation=nesting,maximum_reference_outside_interval=outside,
        max_primal_dual_gap=float(d.primal_dual_gap.abs().max()),
        maximum_budget_violation=float(d.budget_violation_eur_per_normalized_mw.max()),
        maximum_physical_budget_violation=float(np.maximum(d.physical_full_cost_eur_per_normalized_mw-d.budget,0).max()),
        all_solver_statuses_optimal=bool(d.solver_status.eq('OPTIMAL').all()))
    assert max(qa[k] for k in checks)<=1e-6 and qa['max_primal_dual_gap']<=1e-6,qa
    assert nesting<=1e-5 and outside<=1e-5,qa
    assert qa['all_solver_statuses_optimal'] and max(qa['maximum_budget_violation'],qa['maximum_physical_budget_violation'])<=.02,qa
    d.to_csv(P.WORK/'tables/nearoptimal_endpoints.csv',index=False)
    q.to_csv(P.WORK/'tables/nearoptimal_intervals.csv',index=False)
    q['positive_lower']=q['min']>1e-6
    q.groupby(['ensemble','risk','epsilon','technology']).agg(countries=('country','size'),
        positive_lower_countries=('positive_lower','sum'),smallest_lower=('min','min'),largest_lower=('min','max')).reset_index().to_csv(
        P.WORK/'tables/nearoptimal_positive_lower_summary.csv',index=False)
    (P.WORK/'qa/nearoptimal_verification.json').write_text(json.dumps(qa,indent=2)+'\n');print(json.dumps(qa))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['capacity','nearoptimal','all'],default='all');args=ap.parse_args()
    if args.part in ('capacity','all'):capacity()
    if args.part in ('nearoptimal','all'):nearoptimal()
