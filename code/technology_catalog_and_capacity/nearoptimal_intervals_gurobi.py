"""Independent original-unit endpoint certificate using Gurobi barrier."""
import os,time,pickle,json,hashlib
import numpy as np
os.environ.setdefault('GRB_LICENSE_FILE','/etc/gurobi/gurobi.lic')

def run(d,C,empirical,one=False):
    import gurobipy as gp
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
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    m=gp.Model('hydrogen_only_near',env=env)
    try:
        m.Params.OutputFlag=1;m.Params.LogFile=str(out/'gurobi_solver.log')
        m.Params.Threads=2;m.Params.Method=2;m.Params.Crossover=0;m.Params.BarConvTol=1e-11
        m.Params.FeasibilityTol=1e-8;m.Params.OptimalityTol=1e-9;m.Params.TimeLimit=300
        x=m.addMVar(len(cap.c),lb=0.);sense=np.full(len(cap.b),'<');sense[:cap.n_eq]='='
        con=m.addMConstr(cap.A,x,sense,cap.b);bs=.001
        budgetrow=m.addConstr(cap.c@x*bs<=star*bs);slack=max(1e-5,abs(star)*1e-9)
        class Adapter:
            @property
            def objval(self):return m.ObjVal
            def getLpSolution(self):return x.X,[],np.r_[con.Pi,budgetrow.Pi.item()],x.RC
        for eps,idx,cid,side in targets:
            target=out/f'eps{eps:g}_{cid}_{side}.json'
            if target.exists():continue
            budget=star+eps*abs(star)+slack;budgetrow.RHS=budget*bs;scale=1000.;sign=1. if side=='min' else -1.
            objective=np.zeros(len(cap.c));objective[idx]=sign
            m.setObjective(x[idx]*sign*scale,gp.GRB.MINIMIZE);t=time.time();m.optimize()
            attempt={'method':2,'lp_method':2,'solver':'Gurobi','crossover':0,'seconds':time.time()-t,'status':m.Status}
            if m.Status==gp.GRB.OPTIMAL:
                try:
                    solution,audit=N.N.audit(N.CapacityObjectiveView(Adapter(),scale,bs),None,cap,objective,budget)
                    audit.pop('capacity_h2_cavern_168h',None);audit['capacity_h2_tank_168h']=0.
                    physical=empirical(solution,tech,C.VALUE,sc,d['risk']);assert physical['full_cost_eur_per_normalized_mw']<=budget+.02
                    C.write(target,{'design':d,'epsilon':eps,'technology':cid,'bound':side,'capacity_bound':float(solution[idx]),
                        'C_star':star,'budget':budget,'numerical_slack':slack,'attempts':[attempt],'audit':audit,'physical':physical,
                        'objective_scale':scale,'budget_row_scale':bs,'solver_status':'OPTIMAL',
                        'provenance':{'reference_sha256':C.sha(refpath),'matrix_sha256':digest.hexdigest(),'solver_version':gp.gurobi.version()}})
                    print(json.dumps({'gurobi_endpoint':target.name,'seconds':attempt['seconds']}),flush=True)
                except AssertionError as exc:attempt['audit_error']=str(exc)
            if not target.exists():C.write(out/f'failed_gurobi_{target.name}',attempt)
            if one:break
    finally:m.dispose();env.dispose()
