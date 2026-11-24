"""Eight-technology comparison using all audited annual-record results."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

W=Path(__file__).resolve().parents[1]
V=W.parents[1]
H=V/'revision/hydrogen_tank_only_20261114'
sys.path.insert(0,str(H/'code'))
import parameters as P

ORDER=['li_lfp','vrfb','fecr_flow','gravity','molten_salt_tes','caes','laes','h2_tank']
LABELS=['LFP','Vanadium flow','Fe-Cr flow','Gravity','Molten-salt thermal','A-CAES','LAES','Hydrogen']
ZH=['LFP 锂电池','全钒液流电池','铁铬液流电池','重力储能','熔盐储热','A-CAES','液态空气储能','氢储能']
GROUPS=['provincial','national','community']

def main():
    c=pd.read_csv(H/'tables/catalogue_records.csv.gz')
    assert len(c)==11054*58 and c.groupby('uid').size().eq(58).all()
    s=pd.read_csv(H/'tables/catalogue_summary.csv')
    catalogue=P.catalogue()
    costs=pd.DataFrame([P.cost_components(cid) for cid in catalogue])
    costmap=costs.set_index('config_id').annual_fixed_cost_usd_kw_year
    c['annual_fixed_cost']=c.variant.map(costmap)
    c['operating_saving_vs_lfp']=c.B+c.annual_fixed_cost-costmap.loc['li_lfp_4h']
    endpoints=[];best=[];metadata=[]
    for tech,label,zh in zip(ORDER,LABELS,ZH):
        q=c[c.tech.eq(tech)];hs=sorted(q.duration_h.unique());lo,hi=hs[0],hs[-1]
        ids=sorted(q.variant.unique())
        phy={k:sorted({float(catalogue[cid][k]) for cid in ids}) for k in ['rte','charge_power_ratio','self_discharge_pct_day','vom_eur_mwh']}
        metadata.append(dict(tech=tech,label=label,zh=zh,durations=hs,config_count=len(hs),physics=phy))
        for group in GROUPS:
            z=q[q.group.eq(group)];low=z[z.duration_h.eq(lo)].set_index('uid');high=z[z.duration_h.eq(hi)].set_index('uid').reindex(low.index)
            assert high.B.notna().all()
            fixed=high.annual_fixed_cost-low.annual_fixed_cost
            operating=high.operating_saving_vs_lfp-low.operating_saving_vs_lfp
            delta=high.B-low.B
            assert np.allclose(delta,operating-fixed,rtol=0,atol=1e-9)
            endpoints.append(dict(tech=tech,label=label,group=group,n=len(z)//len(hs),h_min=lo,h_max=hi,
                fixed_increment=float(fixed.mean()),operating_saving=float(operating.mean()),B_change=float(delta.mean()),
                B_min_duration=float(low.B.mean()),B_max_duration=float(high.B.mean()),
                records_positive_change=int(delta.gt(1e-6).sum())))
            zbest=z.loc[z.groupby('uid').B.idxmax()]
            means=z.groupby('duration_h').B.mean();bh=float(means.idxmax())
            best.append(dict(tech=tech,label=label,group=group,n=len(zbest),
                best_group_mean_duration=bh,best_group_mean_B=float(means.loc[bh]),
                record_best_median=float(zbest.duration_h.median()),record_best_min=float(zbest.duration_h.min()),
                record_best_max=float(zbest.duration_h.max()),lower_boundary_count=int(zbest.duration_h.eq(lo).sum()),
                upper_boundary_count=int(zbest.duration_h.eq(hi).sum()),
                records_ever_cheaper_than_lfp=int(zbest.B.gt(1e-6).sum())))
    e=pd.DataFrame(endpoints);b=pd.DataFrame(best)
    e.to_csv(W/'tables/r5_endpoint_decomposition.csv',index=False)
    b.to_csv(W/'tables/r5_technology_summary.csv',index=False)
    costs.to_csv(W/'tables/technology_costs_2035.csv',index=False)
    (W/'tables/technology_catalogue.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    qa=dict(records=11054,configurations=58,rows=len(c),technologies=8,exclusions=0,
            decomposition_max_error=float(np.max(np.abs(e.B_change-e.operating_saving+e.fixed_increment))),
            min_fixed_increment=float(e.fixed_increment.min()),min_operating_saving=float(e.operating_saving.min()),
            source='Audited hydrogen-only catalogue; no new dispatch or parameter changes.')
    (W/'qa/r5_analysis.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(e.to_string(index=False));print(b.to_string(index=False));print(json.dumps(qa))

if __name__=='__main__':main()
