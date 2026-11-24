"""Hydrogen-only dispatch under original power-side learning and new tanks."""
from pathlib import Path
import sys,os,json,time,multiprocessing as mp
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import parameters as P
sys.path.insert(0,str(P.V86/'figures/code'));import data as D
sys.path.insert(0,str(P.ROOT/'submission/Nature_Energy_v85/revision/remaining_experiments_20261103/code'));import common as C
OUT=P.WORK/'results/hydrogen';OUT.mkdir(exist_ok=True)

def solve(arr,cid):
    cat=P.catalogue()
    for method in (1,2):
        try:
            a=C.model.solve(*arr.T,cat['li_lfp_4h'],cat[cid],C.scenario(),convention='source_output_rated',method=method);break
        except (AssertionError,RuntimeError):
            if method==2:raise
    return {**a,'cost_usd_kw_year':P.FX*a['cost_eur_kw_year']}

def job(uid):
    p=OUT/f'{uid}.json'
    if p.exists():return uid,'cached'
    arr,_=D.hourly(uid);base=solve(arr,'li_lfp_4h');rows=[];t=time.monotonic()
    for cid,cfg in P.catalogue().items():
        if cfg['tech_id']!=P.H2_ID:continue
        a=solve(arr,cid);rows.append(dict(uid=uid,variant=cid,tech=P.H2_ID,duration_h=cfg['duration_h'],B=base['cost_usd_kw_year']-a['cost_usd_kw_year'],**a))
    obj=dict(uid=uid,lfp=base,hydrogen=rows,seconds=time.monotonic()-t)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(p);C.model.matrices.cache_clear()
    return uid,round(obj['seconds'],1)

def available():return int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))/2**20

if __name__=='__main__':
    P.export();r=D.rd(D.INP/'records.csv');uids=r[r.group.isin(D.MAIN)].uid.tolist()
    todo=[u for u in uids if not (OUT/f'{u}.json').exists()]
    print(json.dumps({'event':'hydrogen_only_start','records':len(todo),'configs':22}),flush=True)
    for i in range(0,len(todo),8):
        while available()<25:time.sleep(10)
        with mp.get_context('spawn').Pool(8) as pool:list(pool.imap_unordered(job,todo[i:i+8]))
        print(json.dumps({'done':min(i+8,len(todo)),'total':len(todo)}),flush=True)
