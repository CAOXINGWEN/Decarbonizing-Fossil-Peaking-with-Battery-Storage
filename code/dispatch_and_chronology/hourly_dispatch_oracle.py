"""Independent 8,760-step v82/v62 matrix with verified primal and dual certificates."""
import sys
from suite_common import M, ROOT, np
sys.path.insert(0,str(ROOT/"submission/Nature_Energy_v85/revision/remaining_experiments_20261103/code"))
import common
from scipy import sparse
from functools import lru_cache


@lru_cache(None)
def full_matrix(variant):
    """Retain every hourly state; substitute the fixed power and hourly unserved load analytically."""
    source=common.model
    base,tail=M._CAT58["li_lfp_4h"],M._CAT58[variant]
    eb,et=np.sqrt(base["rte"]),np.sqrt(tail["rte"])
    rb,rt=[(1-v["self_discharge_pct_day"]/100)**(1/24) for v in (base,tail)]
    ht=source.capacity_h(tail,"source_output_rated")
    ae,au,_=source.matrices(eb,et,rb,rt,tail["charge_power_ratio"],ht)
    n=8760
    zero=sparse.csr_matrix((n,1))
    states=sparse.hstack((ae[n:3*n,:6*n],sparse.vstack((zero,zero))))
    share=sparse.hstack((au[:n,:6*n],zero))
    dispatch=ae[:n,:6*n]
    nonnegative=sparse.hstack((dispatch,zero))
    peak=sparse.hstack((dispatch,np.ones((n,1))))
    return sparse.vstack((states,share,nonnegative,peak),format="csc")


def full_cost(arr, variant, unit_scale=1., settings=None, feasibility_tolerance=1e-10):
    source=common.model
    base,tail=M._CAT58["li_lfp_4h"],M._CAT58[variant]
    hp=source.capacity_h(base,"source_output_rated")
    ht=source.capacity_h(tail,"source_output_rated")
    A=full_matrix(variant)
    task,day_task,surplus=arr.T
    n=len(task)
    inf=M.cp.COPT.INFINITY
    lb=np.zeros(6*n+1); ub=np.full_like(lb,np.inf)
    ub[:n]=np.minimum(surplus,base["charge_power_ratio"])
    ub[n:2*n]=np.minimum(task,1.)
    ub[2*n:3*n]=.9*hp; ub[3*n:4*n]=np.minimum(surplus,tail["charge_power_ratio"])
    ub[4*n:5*n]=np.minimum(day_task,1.); ub[5*n:6*n]=.9*ht
    rl=np.r_[np.zeros(2*n),np.full(2*n,-np.inf),task]
    ru=np.r_[np.zeros(2*n),surplus,task,np.full(n,np.inf)]
    scenario=M.BASE_SCENARIO_EUR
    c=np.zeros_like(lb)
    c[:n]=c[3*n:4*n]=scenario["charge_price_eur_mwh"]
    ev=scenario["energy_value_eur_mwh"]
    c[n:2*n]=base["vom_eur_mwh"]-ev;c[4*n:5*n]=tail["vom_eur_mwh"]-ev
    c[-1]=1000*scenario["firm_value_eur_kw_year"]
    fixed=base["fixed_cost_eur_kw_year"]+tail["fixed_cost_eur_kw_year"]
    error=np.inf
    for method,presolve,scale in (settings or ((1,-1,-1),(2,0,0),(1,0,0),(2,0,1),(3,0,0))):
        lp=source.environment().createModel("hourly_reference")
        try:
            for k,v in (("Logging",0),("Threads",1),("FeasTol",feasibility_tolerance),("DualTol",feasibility_tolerance),
                    ("MatrixTol",1e-12),("DualPerturb",0),("LpMethod",method),("TimeLimit",180)):
                lp.setParam(k,v)
            if presolve>=0: lp.setParam("Presolve",presolve)
            if scale>=0: lp.setParam("Scaling",scale)
            lp.loadMatrix(c,A,np.maximum(rl*unit_scale,-inf),np.minimum(ru*unit_scale,inf),
                          lb,np.minimum(ub*unit_scale,inf))
            try: lp.solveLP()
            except M.cp.CoptError: continue
            if lp.status!=M.cp.COPT.OPTIMAL: continue
            x,_,dual,red=map(np.asarray,lp.getLpSolution())
            x=x/unit_scale
            obj=c@x
            dobj=np.where(np.isfinite(rl),rl,0.)@np.maximum(dual,0.)+np.where(np.isfinite(ru),ru,0.)@np.minimum(dual,0.)+np.where(np.isfinite(ub),ub,0.)@np.minimum(red,0.)
            activity=A@x
            residual=max(np.max(rl-activity),np.max(activity-ru),np.max(lb-x),np.max(x-ub),0.)
            error=max(residual,np.max(abs(c-A.T@dual-red)),abs(obj-dobj)/1000.)
            if error<1e-8:
                return ((obj+task.sum()*ev)/1000.+fixed)*M.FX
        finally:
            lp.dispose()
    raise ArithmeticError(f"Independent hourly certificate for {variant} failed: {error}")
