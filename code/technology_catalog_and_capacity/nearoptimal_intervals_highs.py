"""HiGHS endpoint solve with the same original-matrix independent audit."""
import json,time,pickle,hashlib,warnings
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import vstack,csc_matrix

def run(d,C,empirical,one=False):
    import run_nearoptimal as N
    tech=C.TECH[C.TECH.tech_id!='h2_tank'].copy().reset_index(drop=True)
    out=C.W/'results/nearoptimal'/d['key'];out.mkdir(exist_ok=True)
    targets=[(e,j,t,s) for e in (.01,.001,0.) for j,t in enumerate(('li_lfp_4h','caes_24h')) for s in ('min','max')
             if not (out/f'eps{e:g}_{t}_{s}.json').exists()]
    if not targets:return
    refpath=C.W/'results/capacity'/f"{d['key']}.json";ref=json.loads(refpath.read_text());star=ref['summary']['solver_objective_eur_per_normalized_mw']
    sc=pickle.load(open(C.CACHE/f"{d['country']}_1h.pkl",'rb'))[d['ensemble']]
    cap=N.N.Capture();C.BASE.linprog=cap
    try:C.BASE.solve_country_design_highs(d['country'],sc,tech,C.VALUE,d['risk'],'diversified',1,1)
    except N.N.MatrixCaptured:pass
    digest=hashlib.sha256()
    for a in (cap.A.data,cap.A.indices,cap.A.indptr,cap.b,cap.c):digest.update(a.tobytes())
    bs=.001;ub=vstack([cap.A[cap.n_eq:],csc_matrix((cap.c*bs).reshape(1,-1))],format='csc')
    eq=cap.A[:cap.n_eq];slack=max(1e-5,abs(star)*1e-9)
    for eps,idx,cid,side in targets:
        target=out/f'eps{eps:g}_{cid}_{side}.json'
        if target.exists():continue
        budget=star+eps*abs(star)+slack;sign=1. if side=='min' else -1.;scale=1000.
        objective=np.zeros(len(cap.c));objective[idx]=sign
        t=time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',OptimizeWarning)
            r=linprog(objective*scale,A_eq=eq,b_eq=cap.b[:cap.n_eq],A_ub=ub,b_ub=np.r_[cap.b[cap.n_eq:],budget*bs],
                bounds=(0,None),method='highs-ipm',options={'time_limit':240,'threads':2,'presolve':True,
                    'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-11})
        at={'method':3,'lp_method':'highs-ipm','solver':'SciPy HiGHS','seconds':time.time()-t,'status':int(r.status),'message':r.message}
        class Adapter:
            @property
            def objval(self):return r.fun
            def getLpSolution(self):return r.x,[],np.r_[r.eqlin.marginals,r.ineqlin.marginals],r.lower.marginals
        if r.success:
            try:
                x,audit=N.N.audit(N.CapacityObjectiveView(Adapter(),scale,bs),None,cap,objective,budget)
                audit.pop('capacity_h2_cavern_168h',None);audit['capacity_h2_tank_168h']=0.
                physical=empirical(x,tech,C.VALUE,sc,d['risk']);assert physical['full_cost_eur_per_normalized_mw']<=budget+.02
                C.write(target,{'design':d,'epsilon':eps,'technology':cid,'bound':side,'capacity_bound':float(x[idx]),
                    'C_star':star,'budget':budget,'numerical_slack':slack,'attempts':[at],'audit':audit,'physical':physical,
                    'objective_scale':scale,'budget_row_scale':bs,'solver_status':'OPTIMAL',
                    'provenance':{'reference_sha256':C.sha(refpath),'matrix_sha256':digest.hexdigest(),'solver':'SciPy 1.13.1 HiGHS IPM'}})
                print(json.dumps({'highs_endpoint':target.name,'seconds':at['seconds']}),flush=True)
            except AssertionError as exc:at['audit_error']=str(exc)
        if not target.exists():C.write(out/f'failed_highs_{target.name}',at);print(json.dumps({'failed_highs':target.name,**at}),flush=True)
        if one:break
