"""Hydrogen-only update of capacity planning, with an exact dominance certificate."""
import os,sys,json,time,pickle,subprocess,argparse
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import parameters as P
LEGACY=P.ROOT/'submission/Nature_Energy_v85/revision/remaining_experiments_20261103'
sys.path.insert(0,str(LEGACY/'code'));import capacity_common as C
C.W=P.WORK;C.solver_threads=lambda:4
for sub in ('results/capacity','results/nearoptimal','logs','qa','tables'):(C.W/sub).mkdir(parents=True,exist_ok=True)
C.TECH['config_id']=C.TECH.config_id.astype(str).str.replace('h2_cavern_','h2_tank_',regex=False)
C.TECH['tech_id']=C.TECH.tech_id.astype(str).str.replace('h2_cavern','h2_tank',regex=False)
for i,row in C.TECH.iterrows():
    cid=row.config_id;cfg=P.catalogue()[cid]
    for key in ('rte','self_discharge_pct_day','charge_power_ratio'):C.TECH.loc[i,key]=cfg[key]
    C.TECH.loc[i,'fixed_cost_eur_kw_year']=P.annual_fixed_cost_usd(cid)/P.FX
C.BASE.CONFIG_IDS=tuple(C.TECH.config_id)
ORIGINAL_SOLVE=C.BASE.solve_country_design_highs

def certificate(sc,step):
    r=C.TECH[C.TECH.tech_id=='h2_tank'].iloc[0]
    b=max(C.VALUE['firm_value_eur_kw_year']+C.VALUE['energy_value_eur_mwh']/1000*min(
        float((s['task']>1e-12).sum()*step),r.rte*r.charge_power_ratio*float((s['usable_recharge']>1e-12).sum()*step)) for s in sc)
    margin=float(r.fixed_cost_eur_kw_year-b);assert margin>1.,margin
    return dict(max_operating_benefit_eur_kw_year=b,fixed_cost_eur_kw_year=float(r.fixed_cost_eur_kw_year),strict_margin_eur_kw_year=margin)

def reduced(country,sc,tech,value,risk,mode,step,threads):
    if mode!='diversified':return ORIGINAL_SOLVE(country,sc,tech,value,risk,mode,step,threads)
    cert=certificate(sc,step)
    s,c,m=ORIGINAL_SOLVE(country,sc,tech[tech.tech_id!='h2_tank'],value,risk,mode,step,threads)
    s.update(technology_count=3,hydrogen_dominance_certificate=cert,
             optimization_formulation='LFP and A-CAES LP, hydrogen excluded by strict dominance',
             kkt_scope='retained LP; hydrogen zero-capacity conclusion has an independent strict-dominance proof')
    c.append(dict(country=country,risk_weight=risk,mode=mode,config_id='h2_tank_168h',tech_id='h2_tank',duration_h=168.,
                  capacity_per_normalized_mw=0.,capacity_share=0.,energy_capacity_mwh_per_normalized_mw=0.,annual_fixed_cost_eur_per_normalized_mw=0.))
    return s,c,m

def designs():
    ds=json.loads((LEGACY/'qa/capacity_design.json').read_text())['designs']
    return [{k:(v.replace('h2_cavern','h2_tank') if isinstance(v,str) else v) for k,v in d.items()} for d in ds]

def prepare():
    P.export();ds=designs();reused=[];certs=[]
    for d in ds:
        sc=pickle.load(open(C.CACHE/f"{d['country']}_{d['step']}h.pkl",'rb'))[d['ensemble']]
        cert=certificate(sc,d['step']);certs.append({**d,**cert})
        oldkey=d['key'].replace('h2_tank','h2_cavern');old=json.loads((LEGACY/'results/capacity'/f'{oldkey}.json').read_text())
        hcap=sum(r['capacity_per_normalized_mw'] for r in old['capacities'] if r['config_id']=='h2_cavern_168h')
        if d['mode'] not in ('diversified','h2_tank_168h') or (d['mode']=='diversified' and abs(hcap)<1e-10):
            obj=json.loads(json.dumps(old).replace('h2_cavern','h2_tank'));obj['design']=d
            obj['reuse_proof']=dict(original_json_sha256=C.sha(LEGACY/'results/capacity'/f'{oldkey}.json'),
                reason='Original optimum uses zero hydrogen; retained technology costs and operations are identical; tank hydrogen is strictly dominated.',**cert)
            C.write(C.W/'results/capacity'/f"{d['key']}.json",obj);reused.append(d['key'])
    C.csv(C.W/'qa/capacity_hydrogen_dominance.csv',certs)
    manifest=json.loads((LEGACY/'qa/capacity_design.json').read_text());manifest.update(designs=ds,
        costs=C.TECH[['config_id','fixed_cost_eur_kw_year']].to_dict('records'),configuration='hydrogen_tank_only',
        parameter_sha256=C.sha(P.WORK/'parameters/parameters.json'),reused_verified_designs=reused)
    C.write(C.W/'qa/capacity_design.json',manifest)
    print(json.dumps({'exact_reuse':len(reused),'reoptimize':len(ds)-len(reused),'total':len(ds)}),flush=True)

def empirical(x,tech,value,sc,risk):
    n=len(sc);nt=len(tech);steps=len(sc[0]['task']);block=(3*nt+1)*steps+1;cost=[]
    for i in range(n):
        off=nt+1+n+i*block;ch=x[off:off+nt*steps];dis=x[off+nt*steps:off+2*nt*steps].reshape(nt,steps)
        res=x[off+3*nt*steps:off+(3*nt+1)*steps];peak=x[off+(3*nt+1)*steps]
        cost.append(float(value['charge_price_eur_mwh']*ch.sum()+np.sum(tech.vom_eur_mwh.to_numpy()*dis.sum(axis=1))+
                          value['energy_value_eur_mwh']*res.sum()+1000*value['firm_value_eur_kw_year']*peak))
    cost=np.array(cost);cvar=min(float(t+np.maximum(cost-t,0).sum()/(.1*n)) for t in cost)
    fixed=float(np.sum(x[:nt]*1000*tech.fixed_cost_eur_kw_year.to_numpy()))
    return dict(full_cost_eur_per_normalized_mw=fixed+(1-risk)*float(cost.mean())+risk*cvar,fixed_cost_eur_per_normalized_mw=fixed,
                mean_operating_cost_eur_per_normalized_mw=float(cost.mean()),cvar90_operating_cost_eur_per_normalized_mw=cvar)

def near(d):
    import fcntl
    folder=C.W/'results/nearoptimal'/d['key'];folder.mkdir(exist_ok=True)
    with open(folder/'revision_worker.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _near(d)

def _near(d):
    # Removing dominated hydrogen lowers cost without changing either retained
    # capacity, so the projection onto their near-optimal intervals is identical.
    import near_fast
    near_fast.run(d,C,empirical)
    if len(list((C.W/'results/nearoptimal'/d['key']).glob('eps*.json')))==12:return
    import run_nearoptimal as N
    N.TECH=C.TECH[C.TECH.tech_id!='h2_tank'].copy().reset_index(drop=True)
    N.N.empirical_cost=empirical;original_audit=N.N.audit
    def audit(*args,**kwargs):
        x,r=original_audit(*args,**kwargs);r.pop('capacity_h2_cavern_168h',None);r['capacity_h2_tank_168h']=0.;return x,r
    N.N.audit=audit;N.job(d)

def available():return int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))/2**20

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker');ap.add_argument('--workers',type=int,default=6);ap.add_argument('--near',action='store_true');a=ap.parse_args();ds=designs()
    if a.worker:
        d=next(v for v in ds if v['key']==a.worker)
        if a.near:near(d)
        else:C.BASE.solve_country_design_highs=reduced;print(json.dumps(C.solve_design(d)),flush=True)
        return
    if not a.near:prepare()
    if a.near:ds=[d for d in ds if d['step']==1]
    def done(d):
        if a.near:return len(list((C.W/'results/nearoptimal'/d['key']).glob('eps*.json')))==12
        return (C.W/'results/capacity'/f"{d['key']}.json").exists()
    pending=[d for d in ds if not done(d)];pending.sort(key=lambda d:(d['step'],d['ensemble']!='combined',d['key']))
    active={};failed=[];count=0
    while pending or active:
        while pending and len(active)<a.workers and available()>30:
            d=pending.pop(0);f=open(C.W/'logs'/f"{d['key']}{'_near' if a.near else ''}.worker.log",'a')
            cmd=[sys.executable,__file__,'--worker',d['key']]+(['--near'] if a.near else [])
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);active[proc.pid]=(proc,d,f);time.sleep(.2)
        time.sleep(2)
        for pid,(proc,d,f) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            f.close();del active[pid];count+=1
            if rc:failed.append(d['key'])
            print(json.dumps({'done':count,'remaining':len(pending),'key':d['key'],'exit':rc}),flush=True)
        C.write(C.W/'qa'/('near_progress.json' if a.near else 'capacity_progress.json'),dict(done=count,pending=len(pending),active=list(active),failures=failed))
    if failed:raise RuntimeError(failed)

if __name__=='__main__':main()
