"""Exact recosting of fixed-power dispatch; no simulated observations added."""
import sys,json,os
if __name__=='__main__':os.environ['V86_LEGACY_DATA']='1'
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.interpolate import PchipInterpolator
from functools import lru_cache
import parameters as P
sys.path.insert(0,str(P.V86/'figures/code'))
import data as D
RM=D.RM
OUT=P.WORK/'tables';OUT.mkdir(exist_ok=True)

@lru_cache(None)
def curve(year):
    # New vessel costs can imply roots below -15%; search the whole negative domain.
    grid=np.linspace(-.95,0.,2001)
    yy=np.array([RM.premium(year,r) for r in grid]);i=int(yy.argmin())
    lo,hi=grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]
    res=minimize_scalar(lambda r:RM.premium(year,r),bounds=(lo,hi),method='bounded',options={'xatol':1e-13})
    low=float(res.x);rates=low+(.8-low)*np.linspace(0,1,24001)**2
    vals=np.array([RM.premium(year,r) for r in rates]);keep=np.r_[True,np.diff(vals)>0]
    assert np.all(np.diff(vals[keep])>0)
    return low,float(res.fun),rates[keep],vals[keep],PchipInterpolator(vals[keep],rates[keep],extrapolate=False)

def configure():
    RM.FC=P.annual_fixed_cost_usd;RM.cost_components=P.cost_components
    RM.PREM36={y:RM.premium(y,.036) for y in RM.YEARS};RM.curve=curve

def recost(df,delta):
    x=df.copy();x['B']=x.B-delta
    if 'B_fixed' in x:x['B_fixed']=x.B_fixed-delta
    for y in RM.YEARS:
        if f'roi_{y}' in x:x[f'roi_{y}']=RM.roi(x.B+RM.PREM36[2035],y)
    if 'win36' in x:x['win36']=x.B>=0
    return x

def save(x,name):x.to_csv(OUT/name,index=False)

def main():
    # Snapshot old values before changing the finance function references.
    rec=D.records();ext=D.extension()
    a=D.rd(D.RC/'primary_all.csv.gz')
    a=pd.concat([a[a.group!='provincial'],D.rd(D.PROV/'provincial_primary_all.csv.gz')[a.columns]],ignore_index=True)
    oldprem=RM.PREM36[2035]
    configure();delta=RM.PREM36[2035]-oldprem
    rec=recost(rec,delta);a=recost(a,delta);ext=recost(ext,delta)
    save(rec,'records.csv');save(a,'primary_all.csv.gz');save(ext,'extension.csv.gz')
    s=RM.summaries(a)
    # Rank B, which is defined for every record even where no return root exists.
    real=a[a.seed==-1].set_index('uid');rnd=a[a.seed>=0]
    rank={}
    for uid,z in rnd.groupby('uid'):
        v=real.loc[uid,'B'];vv=z.B.to_numpy();rank[uid]=float((vv<v).mean()+.5*(vv==v).mean())
    for g in s.group.unique():s.loc[s.group==g,'real_pct_med']=np.median([rank[u] for u in real[real.group==g].index])
    save(s,'summary.csv')
    q=RM.record_quantiles(rnd,'roi_2035');q.columns=['mc_q05','mc_q50','mc_q95']
    bq=RM.record_quantiles(rnd,'B');bq.columns=['B_mc_q05','B_mc_q50','B_mc_q95']
    prob=RM.wins(rnd,2035,3.6).groupby(rnd.uid).mean().rename('mc_win36')
    pp=q.join(bq).join(prob).join(pd.Series(rank,name='real_pct')).join(real[['roi_2035','B','group']].rename(columns={'roi_2035':'roi_real','B':'B_real'}))
    pp.index.name='uid';save(pp.reset_index(),'placements.csv')
    f=D.rd(D.PROV/'factorial_records.csv.gz');f=recost(f,delta);save(f,'factorial_records.csv.gz')
    i=D.rd(D.V86/'analysis/provincial_intraday/results/intraday_records.csv');save(recost(i,delta),'intraday_records.csv')
    # CAES capex uncertainty is a cost-only counterfactual, exactly reprice all records.
    sens=[]
    for ce in [200.,300.,400.]:
        change=P.cost_components('caes_24h',2035,.036,ce)['annual_fixed_cost_usd_kw_year']-P.annual_fixed_cost_usd('caes_24h')
        for g,z in rec[rec.group.isin(D.MAIN)].groupby('group',observed=True):
            b=z.B-change;sens.append(dict(energy_cost_internal_usd_kwh=ce,group=g,n=len(z),B_mean=b.mean(),B_median=b.median(),hybrid_cheaper=int((b>=0).sum())))
    save(pd.DataFrame(sens),'caes_cost_sensitivity.csv')
    qa={'fixed_cost_shift_usd_kw_year':delta,'records':len(rec),'random_rows':int((a.seed>=0).sum()),
        'main_records':int(rec.group.isin(D.MAIN).sum()),'B_component_max_error':float(abs(rec[[c for c in rec if c.startswith('B_')]].sum(axis=1)-rec.B).max()),
        'main_wins':rec[rec.group.isin(D.MAIN)].groupby('group',observed=True).win36.sum().to_dict(),
        'main_return_missing':int((~np.isfinite(rec[rec.group.isin(D.MAIN)].roi_2035)).sum())}
    assert qa['B_component_max_error']<1e-6
    (P.WORK/'qa/recost.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(qa,indent=2),flush=True)

if __name__=='__main__':main()
