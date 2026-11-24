"""Disaggregate annual curtailed energy on the same local year's wind and PV profiles."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from extract_cds_china import LOCAL
from prepare_generation_sources import WORK

OUT=WORK/"inputs/reconstruction"
SRC=WORK/"sources"


def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""): h.update(block)
    return h.hexdigest()


def local_profile(year,tech):
    files=[LOCAL/f"{year-1}_{tech}_{'12-12' if year==2015 else '01-12'}.npz",LOCAL/f"{year}_{tech}_01-12.npz"]
    data=[]
    for f in files:
        with np.load(f,allow_pickle=False) as z:
            days=pd.to_datetime(z["utc_hours"],unit="h",utc=True).tz_convert("Asia/Shanghai")
            data.append(pd.DataFrame(z["cf"],index=days,columns=z["province"].tolist()))
    df=pd.concat(data)
    df=df[df.index.year==year]
    expected=pd.date_range(f"{year}-01-01",f"{year+1}-01-01",freq="h",inclusive="left",tz="Asia/Shanghai")
    assert df.index.equals(expected) and np.isfinite(df.to_numpy()).all()
    return df,files


def clipped(available,budget):
    """Monthly unconstrained generation minus a constant export limit, preserving the budget."""
    if budget<=0: return np.zeros_like(available)
    v=np.sort(available)[::-1]
    level=(np.cumsum(v)-budget)/np.arange(1,len(v)+1)
    ix=np.flatnonzero(v>level)[-1]
    limit=max(0.,level[ix])
    return np.maximum(available-limit,0.)


def prepare(year):
    OUT.mkdir(parents=True,exist_ok=True)
    units=pd.read_csv(WORK/"inputs/records.csv").query("year==@year").sort_values("province_idx")
    order=units.province.tolist()
    annual=pd.read_csv(WORK/"inputs/annual_curtailment_ledger.csv").query("year==@year").set_index(["province","tech"])
    monthly=pd.read_csv(SRC/"ember_provincial_generation_monthly.csv").query("year==@year")
    mon=monthly.set_index(["province","tech","month"]).generation_twh
    combined={}
    checks=[]; dependencies=[]
    for tech in ["wind","pv"]:
        cf,files=local_profile(year,tech)
        dependencies.extend(files)
        assert cf.columns.tolist()==order
        cal=cf.index
        if not combined:
            combined={k:np.zeros((31,len(cal)),dtype=np.float64) for k in ["proportional","clipped"]}
        for i,province in enumerate(order):
            info=annual.loc[province,tech]
            budget=info.annual_curtailment_twh*1e6
            generation=0. if pd.isna(info.generation_twh) else info.generation_twh*1e6
            z=cf[province].to_numpy(dtype=float)
            weather_months=np.bincount(cal.month,weights=z,minlength=13)[1:]
            if year>2015:
                obs=mon.loc[province,tech].reindex(range(1,13)).to_numpy(dtype=float)*1e6
                assert np.isfinite(obs).all() and (obs>=0).all()
            else:
                obs=np.zeros(12)
            if obs.sum()>0:
                month_weights=obs/obs.sum()
                shape_source="contemporaneous_monthly_generation"
            else:
                month_weights=weather_months/weather_months.sum()
                shape_source="annual_generation_and_hourly_capacity_factor"
                assert year==2015 or budget==0 or generation==0
            E0=np.zeros(len(cal)); E1=np.zeros(len(cal))
            max_error=0.
            for m in range(1,13):
                select=cal.month==m
                target=budget*month_weights[m-1]
                shape=z[select]
                assert shape.sum()>0 or target==0.
                if not target: continue
                w=shape/shape.sum()
                potential=(generation+budget)*month_weights[m-1]*w
                E0[select]=target*w
                E1[select]=clipped(potential,target)
                assert np.all(E1[select]<=potential+1e-7)
                max_error=max(max_error,abs(E0[select].sum()-target),abs(E1[select].sum()-target))
            for e in (E0,E1):
                assert (e>=0).all() and np.all(e[z==0]==0) and abs(e.sum()-budget)<1e-5
            assert max_error<1e-5
            keep=~((cal.month==2)&(cal.day==29))
            combined["proportional"][i]+=E0
            combined["clipped"][i]+=E1
            checks.append(dict(year=year,province=province,tech=tech,source_hours=len(z),annual_budget_mwh=budget,
                full_proportional_mwh=E0.sum(),full_clipped_mwh=E1.sum(),analysis_proportional_mwh=E0[keep].sum(),
                analysis_clipped_mwh=E1[keep].sum(),max_monthly_error_mwh=max_error,full_generation_mwh=generation,
                observed_monthly_generation_sum_mwh=obs.sum() if year>2015 else np.nan,allocation_source=shape_source,
                positive_proportional_hours=int((E0[keep]>0).sum()),positive_clipped_hours=int((E1[keep]>0).sum())))
    files=[]
    for method,value in combined.items():
        f=OUT/f"{year}_{method}.npy"; part=OUT/f"{year}_{method}.partial.npy"
        np.save(part,np.ascontiguousarray(value[:,keep]));part.replace(f);files.append(f)
    checks_path=OUT/f"{year}_validation.csv"
    pd.DataFrame(checks).to_csv(checks_path,index=False)
    dependencies.extend([WORK/"inputs/annual_curtailment_ledger.csv",SRC/"ember_provincial_generation_monthly.csv",
                         SRC/"c3s_province_codes.csv",Path(__file__)])
    receipt=dict(year=year,timezone="Asia/Shanghai",source_utc_offset_hours=8,full_year_hours=len(cal),analysis_hours=8760,
        source_sha256={str(p.relative_to(WORK)):digest(p) for p in dependencies},
        outputs_sha256={p.name:digest(p) for p in files+[checks_path]})
    f=OUT/f"{year}_complete.json";temp=f.with_suffix(".partial.json")
    temp.write_text(json.dumps(receipt,indent=2)+"\n");temp.replace(f)
    print("reconstructed",year,"31 provinces x 2 resources",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(__doc__)
    cli.add_argument("year",type=int)
    prepare(cli.parse_args().year)
