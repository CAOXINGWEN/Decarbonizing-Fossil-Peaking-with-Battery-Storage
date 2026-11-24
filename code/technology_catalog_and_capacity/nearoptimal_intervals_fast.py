"""Solve retained-capacity endpoints with barrier, auditing original units.

An interior-point certificate is sufficient; a simplex crossover is optional.
The full cost budget and every hourly physical constraint are unchanged.
Uncertified endpoints are left for the original robust solver.
"""
import json, time, pickle, hashlib
import numpy as np

def run(d, C, empirical, one=False):
    import run_nearoptimal as N
    import coptpy as cp
    tech=C.TECH[C.TECH.tech_id!='h2_tank'].copy().reset_index(drop=True)
    out=C.W/'results/nearoptimal'/d['key'];out.mkdir(exist_ok=True)
    # The zero-tolerance face uses the equivalent dual-shifted budget in the
    # robust solver; a raw cost row is ill-conditioned at the exact optimum.
    targets=[(e,j,t,s) for e in (.01,.001) for j,t in enumerate(('li_lfp_4h','caes_24h'))
             for s in ('min','max') if not (out/f'eps{e:g}_{t}_{s}.json').exists()]
    if not targets:return
    refpath=C.W/'results/capacity'/f"{d['key']}.json"
    ref=json.loads(refpath.read_text());star=float(ref['summary']['solver_objective_eur_per_normalized_mw'])
    assert ref['summary']['maximum_stationarity_residual']<=1e-6
    assert abs(ref['summary']['primal_dual_objective_gap_eur'])<=.02
    # If the audited cost optimum already has x_j=0, every nonnegative-cost
    # tolerance has the exact minimum 0. The endpoint dual is y=0, r=e_j.
    # This uses a feasible witness, not a rounded empirical shortcut.
    capacities={r['config_id']:r['capacity_per_normalized_mw'] for r in ref['capacities']}
    summary=ref['summary'];left=[]
    for eps,idx,cid,side in targets:
        if side!='min' or capacities[cid]!=0.:
            left.append((eps,idx,cid,side));continue
        slack=max(1e-5,abs(star)*1e-9);budget=star+eps*abs(star)+slack
        audit={k:summary[k] for k in ('maximum_equality_residual','maximum_inequality_violation','maximum_nonnegativity_violation')}
        audit.update(maximum_stationarity_residual=0.,maximum_reduced_cost_sign_violation=0.,maximum_inequality_dual_sign_violation=0.,
            primal_dual_gap=0.,original_objective_eur_per_normalized_mw=star,budget_violation_eur_per_normalized_mw=0.,
            capacity_li_lfp_4h=capacities['li_lfp_4h'],capacity_caes_24h=capacities['caes_24h'],capacity_h2_tank_168h=0.)
        physical={'full_cost_eur_per_normalized_mw':summary['risk_adjusted_cost_eur_per_normalized_mw'],
            'fixed_cost_eur_per_normalized_mw':summary['annual_fixed_cost_eur_per_normalized_mw'],
            'mean_operating_cost_eur_per_normalized_mw':summary['mean_operating_cost_eur_per_normalized_mw'],
            'cvar90_operating_cost_eur_per_normalized_mw':summary['cvar90_operating_cost_eur_per_normalized_mw']}
        assert physical['full_cost_eur_per_normalized_mw']<=budget+.02
        C.write(out/f'eps{eps:g}_{cid}_{side}.json',{'design':d,'epsilon':eps,'technology':cid,'bound':side,
            'capacity_bound':0.,'C_star':star,'budget':budget,'numerical_slack':slack,'audit':audit,'physical':physical,
            'attempts':[{'method':0,'lp_method':0,'crossover':0,'certificate':'feasible zero-capacity witness and nonnegative lower bound'}],
            'solver_status':'OPTIMAL','analytic_proof':{'primal_witness_sha256':C.sha(refpath),
                'dual_certificate':'zero row multipliers; reduced cost one at minimized capacity, zero elsewhere',
                'feasibility_audit':'inherited from independently certified cost-optimal witness'}})
    targets=left
    if not targets:return
    sc=pickle.load(open(C.CACHE/f"{d['country']}_1h.pkl",'rb'))[d['ensemble']]
    cap=N.N.Capture();C.BASE.linprog=cap
    try:C.BASE.solve_country_design_highs(d['country'],sc,tech,C.VALUE,d['risk'],'diversified',1,1)
    except N.N.MatrixCaptured:pass
    digest=hashlib.sha256()
    for a in (cap.A.data,cap.A.indices,cap.A.indptr,cap.b,cap.c):digest.update(a.tobytes())
    provenance={'reference_sha256':C.sha(refpath),'matrix_sha256':digest.hexdigest(),
                'method':'barrier certificate without mandatory simplex crossover',
                'hydrogen_projection':'unchanged by removing strictly dominated hydrogen',
                'cost_reference':'independently audited hourly capacity LP'}
    if not (out/'baseline.json').exists():
        C.write(out/'baseline.json',{'design':d,'C_star':star,'capacity_reference':str(refpath),
                                    'reference_sha256':C.sha(refpath),'summary_audit':ref['summary']})
    cfg=cp.EnvrConfig();cfg.set('nobanner','1');env=cp.Envr(cfg);m=env.createModel('hydrogen_only_near')
    try:
        m.setLogFile(str(out/'barrier_solver.log'))
        for k,v in {'Logging':1,'Threads':2,'LpMethod':2,'Crossover':0,'Dualize':0,'BarIterLimit':1000,
                    'BarHomogeneous':1,'FeasTol':1e-7,'DualTol':1e-7,'TimeLimit':240}.items():m.setParam(k,v)
        n=len(cap.c);bs=.001
        m.loadMatrix(cap.c,cap.A,np.r_[cap.b[:cap.n_eq],np.full(len(cap.b)-cap.n_eq,-cp.COPT.INFINITY)],
                     cap.b,np.zeros(n),np.full(n,cp.COPT.INFINITY))
        slack=max(1e-5,abs(star)*1e-9)
        nz=np.flatnonzero(cap.c)
        row=m.addConstr(cp.quicksum(float(cap.c[i])*bs*m.getVar(int(i)) for i in nz)<=(star+slack)*bs)
        for eps,idx,cid,side in targets:
            target=out/f'eps{eps:g}_{cid}_{side}.json';budget=star+eps*abs(star)+slack
            m.setInfo(cp.COPT.Info.UB,[row],[budget*bs]);attempts=[]
            for scale in (1000.,1.):
                m.reset();sign=1. if side=='min' else -1.
                m.setObjective(sign*scale*m.getVar(idx),cp.COPT.MINIMIZE)
                objective=np.zeros(n);objective[idx]=sign
                started=time.time();m.solve()
                at={'lp_method':2,'crossover':0,'status':int(m.status),'seconds':time.time()-started,
                    'objective_scale':scale,'budget_row_scale':bs};attempts.append(at)
                if m.status!=cp.COPT.OPTIMAL:continue
                try:
                    x,audit=N.N.audit(N.CapacityObjectiveView(m,scale,bs),cp,cap,objective,budget)
                    audit.pop('capacity_h2_cavern_168h',None);audit['capacity_h2_tank_168h']=0.
                    phys=empirical(x,tech,C.VALUE,sc,d['risk'])
                    assert phys['full_cost_eur_per_normalized_mw']<=budget+.02
                    C.write(target,{'design':d,'epsilon':eps,'technology':cid,'bound':side,
                        'capacity_bound':float(x[idx]),'C_star':star,'budget':budget,'numerical_slack':slack,
                        'attempts':attempts,'audit':audit,'physical':phys,'objective_scale':scale,
                        'budget_row_scale':bs,'solver_status':'OPTIMAL','provenance':provenance})
                    print(json.dumps({'fast_endpoint':target.name,'seconds':at['seconds']}),flush=True)
                    break
                except AssertionError as exc:at['audit_error']=str(exc)
            if not target.exists():C.write(out/f'failed_fast_{target.name}',{'attempts':attempts})
            if one:break
    finally:m.dispose();env.close()
