import os
os.environ['V86_ORIGINAL_CATALOGUE']='1'
import sys,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import parameters as P
sys.path.insert(0,str(P.V86/'figures/code'));import data as D
T=P.WORK/'tables'
def rho(x,y):return float(spearmanr(x,y).statistic) if x.nunique()>1 and y.nunique()>1 else None
def main():
    legacy=D.catalogue_records();c=legacy[legacy.tech!='h2_cavern'].copy();rows=[]
    for f in sorted((P.WORK/'results/hydrogen').glob('*.json')):
        d=json.loads(f.read_text());assert len(d['hydrogen'])==22;rows.extend(d['hydrogen'])
    assert len(rows)==185*22
    h=pd.DataFrame(rows).merge(D.records()[['uid','group']],on='uid',validate='m:1')
    h.to_csv(T/'hydrogen_dispatch.csv.gz',index=False)
    c=pd.concat([c,h[c.columns]],ignore_index=True).sort_values(['group','uid','tech','duration_h'])
    c.to_csv(T/'catalogue_records.csv.gz',index=False)
    keys=['group','variant','tech','duration_h'];s=c.groupby(keys).B.agg(mean='mean',median='median',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index()
    s.to_csv(T/'catalogue_summary.csv',index=False)
    b=c.loc[c.groupby('uid').B.idxmax()];b.to_csv(T/'catalogue_best_records.csv',index=False)
    b.groupby(keys).size().rename('records').reset_index().to_csv(T/'catalogue_best_counts.csv',index=False)
    q=c[c.tech==P.H2_ID];best=q.loc[q.groupby('uid').B.idxmax()].merge(D.windows()[['uid','long_windows']],on='uid')
    best.to_csv(T/'hydrogen_best_records.csv',index=False)
    original=legacy[legacy.tech!='h2_cavern'].sort_values(['uid','variant']).reset_index(drop=True)
    unchanged=c[c.tech!=P.H2_ID].sort_values(['uid','variant']).reset_index(drop=True)
    pd.testing.assert_frame_equal(original,unchanged)
    qa=dict(configuration='hydrogen_tank_only',records=185,candidate_configs=58,hydrogen_lp_solves=len(h),
            unchanged_nonhydrogen_rows=len(unchanged),unchanged_nonhydrogen_max_error=0.,
            rho_h2_duration_vs_long_windows=rho(best.long_windows,best.duration_h),
            h2_best_duration_counts=best.duration_h.value_counts().to_dict(),
            all_h2_optimal=bool(h.solver_status.eq(1).all()),
            max_physical_residual=float(h.max_physical_residual.max()),
            max_stationarity_residual=float(h.max_stationarity_residual.max()),
            max_duality_gap=float(abs(h.duality_gap_eur_kw_year).max()),groups={})
    for g in D.MAIN:
        v=best[best.group==g];m=s[(s.tech==P.H2_ID)&(s.group==g)]
        qa['groups'][g]=dict(h2_optimal_median=float(v.duration_h.median()),h2_optimal_range=[float(v.duration_h.min()),float(v.duration_h.max())],
                            h2_best_group_mean_duration=float(m.loc[m['mean'].idxmax(),'duration_h']),
                            rho=rho(v.long_windows,v.duration_h),h2_mean_B={str(int(r.duration_h)):float(r.mean) for r in m.itertuples()},
                            winners=b[b.group==g].variant.value_counts().to_dict())
    (P.WORK/'qa/catalogue.json').write_text(json.dumps(qa,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in qa.items() if k!='groups'},indent=2));print('winners',b.groupby(['group','variant']).size().to_dict())
if __name__=='__main__':main()
