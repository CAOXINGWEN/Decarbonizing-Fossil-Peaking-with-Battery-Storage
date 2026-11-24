"""Exact capacity endpoints by warm cost-LP solves and a mapped dual certificate.

For a maximum, impose p_j >= q and minimize the unchanged total cost. A
positive reduced cost g at this bound supplies the supporting-line update
q <- q + (budget - cost)/g. For a minimum use p_j <= q and g=-reduced_j.
At the cost boundary, the endpoint certificate is y/g, budget dual -1/g,
and r/g + sign*e_j. It is audited against the original hourly matrix.
"""
import json,pickle,time,hashlib
import numpy as np

def run(d,C,empirical,one=False):
    import coptpy as cp
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
    cfg=cp.EnvrConfig();cfg.set('nobanner','1');env=cp.Envr(cfg);m=env.createModel('cost_oracle')
    class Certificate:
        def __init__(self,x,y,r,objective):self.x=x;self.y=y;self.r=r;self.objval=float(objective@x)
        def getLpSolution(self):return self.x,[],self.y,self.r
    try:
        m.setLogFile(str(out/'parametric_cost_solver.log'))
        for k,v in {'Logging':1,'Threads':2,'LpMethod':2,'Crossover':1,'FeasTol':1e-8,'DualTol':1e-9,'TimeLimit':300}.items():m.setParam(k,v)
        n=len(cap.c);oscale=.001
        m.loadMatrix(cap.c*oscale,cap.A,np.r_[cap.b[:cap.n_eq],np.full(len(cap.b)-cap.n_eq,-cp.COPT.INFINITY)],cap.b,np.zeros(n),np.full(n,cp.COPT.INFINITY))
        variables=[m.getVar(0),m.getVar(1)];first=True
        for eps,idx,cid,side in targets:
            target=out/f'eps{eps:g}_{cid}_{side}.json'
            if target.exists():continue
            slack=max(1e-5,abs(star)*1e-9);budget=star+eps*abs(star)+slack
            m.setInfo(cp.COPT.Info.LB,variables,[0.,0.]);m.setInfo(cp.COPT.Info.UB,variables,[cp.COPT.INFINITY]*2)
            sign=1. if side=='min' else -1.
            reference=next(z['capacity_per_normalized_mw'] for z in ref['capacities'] if z['config_id']==cid)
            delta=max(1e-4,4*eps*budget/cap.c[idx]);q=max(0.,reference-sign*delta)
            objective=np.zeros(n);objective[idx]=sign;attempts=[];started=time.time();audit=None
            for iteration in range(30):
                m.setInfo(cp.COPT.Info.UB if side=='min' else cp.COPT.Info.LB,[variables[idx]],[q])
                m.setParam('LpMethod',2 if first else 1);m.setParam('TimeLimit',300 if first else 30)
                t=time.time();m.solve();first=False
                if m.status!=cp.COPT.OPTIMAL:
                    m.reset();m.setParam('LpMethod',2);m.setParam('TimeLimit',300);m.solve()
                at={'method':int(m.getParam('LpMethod')),'solver':'COPT cost-LP continuation','iteration':iteration,
                    'bound':q,'status':int(m.status),'seconds':time.time()-t};attempts.append(at)
                if m.status!=cp.COPT.OPTIMAL:break
                xx,_,yy,rr=m.getLpSolution();x=np.asarray(xx);y=np.asarray(yy)/oscale;r=np.asarray(rr)/oscale
                cost=float(cap.c@x);g=float(-sign*r[idx]);at.update(cost=cost,marginal_cost=g)
                if side=='min' and q==0. and cost<=budget+.000001:
                    cert=Certificate(x,np.zeros(len(cap.b)+1),objective.copy(),objective)
                elif g>1e-12:
                    reduced=r/g;reduced[idx]+=sign
                    cert=Certificate(x,np.r_[y/g,-1/g],reduced,objective)
                else:
                    if cost<=budget:
                        delta*=2;q=max(0.,reference-sign*delta);continue
                    break
                try:
                    _,audit=N.N.audit(cert,cp,cap,objective,budget)
                    audit.pop('capacity_h2_cavern_168h',None);audit['capacity_h2_tank_168h']=0.
                    physical=empirical(x,tech,C.VALUE,sc,d['risk'])
                    assert physical['full_cost_eur_per_normalized_mw']<=budget+.02
                    C.write(target,{'design':d,'epsilon':eps,'technology':cid,'bound':side,'capacity_bound':float(x[idx]),
                        'C_star':star,'budget':budget,'numerical_slack':slack,'attempts':attempts,'audit':audit,'physical':physical,
                        'solver_status':'OPTIMAL','provenance':{'reference_sha256':C.sha(refpath),'matrix_sha256':digest.hexdigest(),
                        'algorithm':'bound continuation on unchanged cost LP; certificate mapped to original endpoint LP'},
                        'total_seconds':time.time()-started})
                    print(json.dumps({'parametric_endpoint':target.name,'iterations':iteration+1,'seconds':time.time()-started}),flush=True);break
                except AssertionError as exc:at['endpoint_audit_error']=str(exc)
                if cost<budget:
                    delta*=2;q=max(0.,reference-sign*delta);continue
                if g<=1e-12:break
                q_new=max(0.,q-sign*(budget-cost)/g)
                if abs(q_new-q)<1e-13:
                    m.setParam('FeasTol',1e-9);m.setParam('DualTol',1e-9)
                q=q_new
            if not target.exists():C.write(out/f'failed_parametric_{target.name}',{'attempts':attempts})
            if one:break
    finally:m.dispose();env.close()
