"""Provincial load-based peaking. Solve original dates and seeded seasonal placements."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable]="1"
WORK=Path(__file__).resolve().parents[1]
REV=WORK.parent
sys.path.insert(0,str(REV/"japan_baseline_mc_10612_20261118/code"))
from run_mc import solve_order, write_json, file_digest, model, QUANTILES
from compact_dispatch import DayGraph, CompactLP
import numpy as np
import pandas as pd
from prepare_inputs import DATA

NSEEDS=1000
DAYS=pd.date_range("2019-01-01",periods=365,freq="D")
MONTH=DAYS.month.to_numpy()
SEASON=(MONTH%12)//3
META=None
LOAD=None
SUPPLY=None
SCENARIO=None
OUT=None


def setup(scenario):
    global META, LOAD, SUPPLY, SCENARIO, OUT
    META=pd.read_csv(DATA/"records.csv",dtype={"uid":str}).sort_values("row_idx").reset_index(drop=True)
    assert np.array_equal(META.row_idx,np.arange(310)) and META.uid.is_unique
    LOAD=np.load(DATA/"china_2015_2024_load_mw.npy",mmap_mode="r")
    SCENARIO=scenario
    source="china_2020_curtailment_mw.npy" if scenario=="fixed_2020" else "china_yearly_surplus_mw.npy"
    SUPPLY=np.load(DATA/source,mmap_mode="r")
    assert SUPPLY.shape==((31,8760) if scenario=="fixed_2020" else (310,8760))
    OUT=WORK/"results"/scenario
    (OUT/"cases").mkdir(parents=True,exist_ok=True)


def init_worker():
    # The common model's Japanese input mmap is unnecessary for China.
    os.environ["MALLOC_ARENA_MAX"]="2"
    config=model.cp.EnvrConfig()
    config.set("nobanner","1")
    model._ENV=model.cp.Envr(config)


def inputs(index,q):
    rec=META.iloc[index]
    load=LOAD[index]
    r=SUPPLY[int(rec.province_idx) if SCENARIO=="fixed_2020" else index]
    block=load.reshape(365,24)
    task=np.empty_like(block)
    for month in range(1,13):
        mask=MONTH==month
        task[mask]=np.maximum(0.,block[mask]-np.quantile(block[mask],q/100.,axis=0))
    daily=task.sum(axis=1)
    flags=np.zeros(365,dtype=bool)
    for season in range(4):
        mask=SEASON==season
        flags[mask]=daily[mask]>=np.quantile(daily[mask],.75)
    p=.01*load.mean()
    assert np.isclose(p,rec.reference_power_mw,rtol=1e-13,atol=1e-9)
    task=task.ravel()
    arr=model._transform_rule(np.column_stack([task,task*np.repeat(flags,24),r])/p)
    return rec,load,r,task,flags,arr


def date_metrics(index,q):
    rec,load,r,task,flags,arr=inputs(index,q)
    s=model.CollapsedIndexer(arr)
    base=model._CAT58["li_lfp_4h"]
    solutions=[]
    for tail in (base,model._CAT58["caes_24h"]):
        lp=model.PersistentBranchLP(s,base,tail,model.BASE_SCENARIO_EUR)
        try: solutions.append(lp.full_solution())
        finally: lp.dispose()
    lf,hy=solutions
    runs=model._runs(flags)
    ratio=float(load.mean()*.01/task.mean())
    row=dict(row_idx=index,uid=rec.uid,province=rec.province,year=int(rec.year),q=q,
             in_gas_case_26=bool(rec.in_gas_case_26),supply_scenario=SCENARIO,
             ref_power_mw=float(.01*load.mean()),load_mwh=float(load.sum()),task_mwh=float(task.sum()),
             task_share=float(task.sum()/load.sum()),power_over_mean_task=ratio,
             raw_supply_mwh=float(r.sum()),eligible_supply_mwh=float(arr[:,2].sum()*.01*load.mean()),
             task_hours=int((arr[:,0]>1e-10).sum()),charge_hours=int((arr[:,2]>1e-9).sum()),
             event_days=int(flags.sum()),events=len(runs),mean_event_days=float(np.mean(runs)),
             longest_event_days=max(runs),long_windows=model._long_windows_fast(arr),
             lfp_efc=lf["tail_discharge_mwh_per_mw"]/4.,acaes_efc=hy["tail_discharge_mwh_per_mw"]/24.,
             lfp_cost=lf["cost_usd_kw_year"],hybrid_cost=hy["cost_usd_kw_year"],
             real_B=lf["cost_usd_kw_year"]-hy["cost_usd_kw_year"])
    for component in ("fixed","variable_om","charging","residual_energy","residual_peak"):
        row["B_"+component]=lf[component+"_usd_kw_year"]-hy[component+"_usd_kw_year"]
    assert abs(sum(row["B_"+k] for k in ("fixed","variable_om","charging","residual_energy","residual_peak"))-row["real_B"])<1e-6
    return row,flags,arr


def placement_inputs(key,flags,arr,seeds):
    segs=model._precompute_obs_segments(SEASON,flags)
    for _,clusters,ordinary in segs:
        if clusters and len(ordinary)<len(clusters)+1:
            raise ValueError(f"Insufficient ordinary days in {key}: {len(ordinary)}, {len(clusters)}")
    previous=np.empty((len(seeds),365),dtype=np.int32)
    windows=np.empty(len(seeds),dtype=np.int16)
    by_day=arr.reshape(365,24,3)
    for i,seed in enumerate(seeds):
        order=model._fast_obs_random_order(key,segs,int(seed))
        if not np.array_equal(np.sort(order),np.arange(365)):
            raise ValueError(f"Day conservation failed for {key}, seed {seed}")
        previous[i]=model._prev_from_order(order)
        windows[i]=model._long_windows_fast(by_day[order].reshape(8760,3))
    return previous,windows


def randomized(index,flags,arr,seeds,real):
    rec=META.iloc[index]
    previous,windows=placement_inputs(rec.seed_key,flags,arr,seeds)
    graph=DayGraph(arr)
    base=model._CAT58["li_lfp_4h"]
    battery=CompactLP(graph,base,base,model)
    hybrid=CompactLP(graph,base,model._CAT58["caes_24h"],model)
    b=np.empty(len(seeds),dtype=np.float64)
    try:
        error=abs((battery.solve_cost_usd()-hybrid.solve_cost_usd())-real)
        if error>1e-6:
            raise ValueError(f"Compact vs reference original dates disagree at {rec.uid}: {error}")
        for i in solve_order(previous):
            battery.rewire(previous[i]); hybrid.rewire(previous[i])
            b[i]=battery.solve_cost_usd()-hybrid.solve_cost_usd()
    finally:
        battery.dispose()
        hybrid.dispose()
    assert np.isfinite(b).all()
    return b,windows,error


def case_path(index,q):
    return OUT/"cases"/f"{index:03d}_q{q:02d}.npz"


def run_case(case):
    index,q=case
    row,flags,arr=date_metrics(index,q)
    b,w,error=randomized(index,flags,arr,range(NSEEDS),row["real_B"])
    row["compact_native_cost_error"]=error
    path=case_path(index,q)
    temp=path.with_suffix(".partial.npz")
    np.savez_compressed(temp,B=b,long_windows=w,metrics=json.dumps(row,ensure_ascii=False,allow_nan=False))
    temp.replace(path)
    return index,q


def contract():
    supply_file="china_2020_curtailment_mw.npy" if SCENARIO=="fixed_2020" else "china_yearly_surplus_mw.npy"
    hashes={name:file_digest(DATA/name) for name in ["records.csv","china_2015_2024_load_mw.npy",supply_file]}
    sources=[Path(__file__),Path(model.__file__),REV/"japan_baseline_mc_10612_20261118/code/compact_dispatch.py"]
    return dict(scenario=SCENARIO,inputs_sha256=hashes,source_sha256={p.name:file_digest(p) for p in sources},
                seed_count=NSEEDS,q=list(QUANTILES),calendar="January-December, remove February 29",
                reference_power="0.01 * raw province-year mean load in the analysis calendar",
                fixed_schedule="same province, same local month, day, hour as 2020",cost_year=2035,real_rate=.036)


def run(years,workers):
    target=contract()
    p=OUT/"contract.json"
    if p.exists() and json.loads(p.read_text())!=target:
        raise RuntimeError("The run contract differs from existing cases")
    write_json(p,target)
    selected=META[META.year.isin(years)]
    cases=[(int(i),q) for i in selected.row_idx for q in QUANTILES]
    remaining=[c for c in cases if not case_path(*c).exists()]
    np.random.default_rng(20261118).shuffle(remaining)
    completed=len(cases)-len(remaining)
    start=time.monotonic()
    workers=min(model.limit_workers(workers),max(1,len(remaining)))
    def status(count):
        info=dict(years=years,records=len(selected),baseline_cases_done=count,baseline_cases=len(cases),
                  placements_done=count*NSEEDS,placements_target=len(cases)*NSEEDS,workers=workers,
                  elapsed_s=round(time.monotonic()-start,1),unix_time=time.time())
        write_json(OUT/"progress.json",info)
        print(json.dumps(info),flush=True)
    status(completed)
    last=start
    if remaining:
        with mp.get_context("fork").Pool(workers,initializer=init_worker,maxtasksperchild=20) as pool:
            for _ in pool.imap_unordered(run_case,remaining,chunksize=1):
                completed+=1
                if time.monotonic()-last>=20 or completed==len(cases):
                    status(completed); last=time.monotonic()
    assert completed==len(cases)


if __name__=="__main__":
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument("--supply",choices=["fixed_2020","yearly_surplus"],default="fixed_2020")
    ap.add_argument("--years",type=int,nargs="+",default=list(range(2015,2025)))
    ap.add_argument("--workers",type=int,default=80)
    args=ap.parse_args()
    setup(args.supply)
    with (OUT/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(sorted(set(args.years)),args.workers)
