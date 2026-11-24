"""Run and verify the two annual-statistics reconstructions with the registered portfolio model."""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import china_mc as C
from reconstruct_charging import OUT as RECON, digest
import summarize_china as summary

SCENARIOS={"proportional":"yearly_surplus","clipped":"yearly_concentrated"}
SEVEN=tuple(C.QUANTILES)


class OneYear:
    def __init__(self,year,source):
        self.start=(year-2015)*31
        self.data=np.load(source,mmap_mode="r")
        assert self.data.shape==(31,8760)

    def __getitem__(self,i):
        pos=int(i)-self.start
        if not 0<=pos<31: raise IndexError("Record is outside this charging year")
        return self.data[pos]


def initialize(method):
    C.setup("fixed_2020")
    C.SCENARIO=SCENARIOS[method]
    C.OUT=C.WORK/"results"/C.SCENARIO
    C.QUANTILES=SEVEN if method=="proportional" else (50,)


def run(year,method,workers):
    initialize(method)
    C.OUT=C.OUT/"year_chunks"/str(year)
    (C.OUT/"cases").mkdir(parents=True,exist_ok=True)
    source=RECON/f"{year}_{method}.npy"
    receipt=RECON/f"{year}_complete.json"
    assert digest(source)==json.loads(receipt.read_text())["outputs_sha256"][source.name]
    C.SUPPLY=OneYear(year,source)
    files=[Path(__file__),Path(C.__file__),Path(C.model.__file__),C.REV/"japan_baseline_mc_10612_20261118/code/compact_dispatch.py",
           C.REV/"japan_baseline_mc_10612_20261118/code/run_mc.py",C.DATA/"records.csv",C.DATA/"china_2015_2024_load_mw.npy"]
    contract=dict(year=year,resource=C.SCENARIO,estimated_hourly_placement=method,annual_calibration_sha256=digest(receipt),
                  input_resource_sha256=digest(source),seeds=list([0,999]),seeds_per_record=1000,quantiles=list(C.QUANTILES),
                  technical_cost_year=2035,real_financing_rate=.036,calendar="local January–December, remove February 29",
                  source_sha256={f.name:digest(f) for f in files})
    C.contract=lambda:contract
    with (C.OUT/"run.lock").open("a") as lock:
        C.fcntl.flock(lock,C.fcntl.LOCK_EX|C.fcntl.LOCK_NB)
        C.run([year],workers)
        C.write_json(C.OUT/"finished.json",{"year":year,"cases":31*len(C.QUANTILES),"placements":31000*len(C.QUANTILES)})


def publish(method):
    initialize(method)
    cases=C.OUT/"cases";cases.mkdir(exist_ok=True)
    all_supply=[];contracts={}
    for year in range(2015,2025):
        year_dir=C.OUT/"year_chunks"/str(year)
        completion=json.loads((year_dir/"finished.json").read_text())
        paths=sorted((year_dir/"cases").glob("*_q*.npz"))
        assert len(paths)==completion["cases"]==31*len(C.QUANTILES)
        for path in paths:
            target=cases/path.name
            if target.exists(): assert os.path.samefile(path,target) or digest(path)==digest(target)
            else: os.link(path,target)
        source=RECON/f"{year}_{method}.npy"
        rec=json.loads((year_dir/"contract.json").read_text())
        assert rec["input_resource_sha256"]==digest(source) and rec["estimated_hourly_placement"]==method
        all_supply.append(np.load(source))
        contracts[str(year)]=digest(year_dir/"contract.json")
    supply=np.concatenate(all_supply)
    assert supply.shape==(310,8760) and np.isfinite(supply).all() and (supply>=0).all()
    if method=="proportional":
        joined=C.DATA/"china_yearly_surplus_mw.npy"
        np.save(joined,supply)
    C.SUPPLY=supply
    C.write_json(C.OUT/"contract.json",dict(scenario=C.SCENARIO,placement=method,
         description="Year-specific estimated hourly curtailed wind and PV energy, calibrated to the province-year ledger",
         annual_contracts_sha256=contracts,quantiles=list(C.QUANTILES),seeds_per_record=1000))
    summary.publish(list(range(2015,2025)))


if __name__=="__main__":
    cli=argparse.ArgumentParser(__doc__)
    cli.add_argument("--year",type=int);cli.add_argument("--method",choices=list(SCENARIOS),required=True)
    cli.add_argument("--workers",type=int,default=80);cli.add_argument("--publish",action="store_true")
    args=cli.parse_args()
    if args.publish: publish(args.method)
    elif args.year in range(2015,2025): run(args.year,args.method,args.workers)
    else: cli.error("Expected a supported --year or --publish")
