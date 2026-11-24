"""China phase 1 using the registered q50 input constructor and physical LPs."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import json
import math
import pickle
import time

from suite_common import C, M, RM, WORK, REV, VERSION, parallel, sha, signatures, contract, write_json
import numpy as np
import pandas as pd
from scipy.optimize import brentq

DIRECTORIES = {"proportional": "q50_r4_r5_suite", "clipped": "q50_r4_r5_suite_clipped"}
ARCHIVES = {"proportional": "yearly_surplus", "clipped": "yearly_concentrated_all_q_20261120"}
OUT = None
QA = None
KEYS = ("max_physical_residual", "max_stationarity_residual", "duality_gap_eur_kw_year")


def setup(method):
    global OUT
    C.setup("yearly_surplus")
    C.SCENARIO = ARCHIVES[method]
    OUT = WORK/"results"/DIRECTORIES[method]
    (OUT/"checkpoints").mkdir(parents=True, exist_ok=True)
    resources = [C.DATA/"china_yearly_surplus_mw.npy"]
    if method == "clipped":
        resources = [C.DATA/"reconstruction"/f"{y}_clipped.npy" for y in range(2015,2025)]
        C.SUPPLY = np.concatenate([np.load(f, mmap_mode="r") for f in resources])
    params = REV/"hydrogen_tank_only_20261114/parameters/parameters.json"
    physics = json.loads(params.read_text())
    assert physics["catalogue"] == M._CAT58 and len(M._CAT58)==58 and len(M._TECH_GROUPS)==8
    assert len(M._H2_DURS)==22 and M._H2_DURS[0]==4 and M._H2_DURS[-1]==720
    assert round(M._H2_RTE,6)==.320787
    assert np.array_equal(C.META.year, np.repeat(np.arange(2015,2025),31))
    assert np.array_equal(C.META.province_idx, np.tile(np.arange(31),10))
    arc = WORK/"results"/ARCHIVES[method]/"years_2015_2024"
    receipt = json.loads((arc/"complete.json").read_text())
    for name in ("china_records.csv.gz","china_seed_matrices.h5"):
        assert sha(arc/name)==receipt["files_sha256"][name]
    evidence = signatures(resources + [C.DATA/"records.csv", C.DATA/"china_2015_2024_load_mw.npy", params,
        Path(__file__), Path(__file__).with_name("suite_common.py"), Path(C.__file__), Path(M.__file__),
        Path(M.HP.__file__),Path(RM.__file__),Path(M.ce.__file__),Path(M.WD.__file__),
        Path(__import__("cyclic_diagnostic").__file__),arc/"complete.json"])
    contract(OUT, dict(method=method, q=50, solver=VERSION, sha256=evidence))
    return OUT, arc


@contextmanager
def branch(idx, cfg):
    lp = M.PersistentBranchLP(idx,M._CAT58["li_lfp_4h"],cfg,M.BASE_SCENARIO_EUR)
    try:
        # COPT encodes open bounds by a finite sentinel. The dual certificate
        # uses only actual finite bounds, even when a multiplier is roundoff-sized.
        lp.rl[lp.rl <= -M.cp.COPT.INFINITY] = -np.inf
        lp.ru[lp.ru >= M.cp.COPT.INFINITY] = np.inf
        lp.cu[lp.cu >= M.cp.COPT.INFINITY] = np.inf
        yield lp
    finally:
        lp.dispose()


def solve(lp):
    # full_solution checks the original constraint matrix. Refresh it after calendar rewiring.
    try:
        sol = lp.full_solution(check_kkt=True)
    except AssertionError:
        sol = None
    if sol is None or max(sol[k] for k in KEYS) >= 1e-8:
        # Cold resolve the same matrix and current bounds. COPT's presolver can
        # eliminate tiny but positive provincial charging quantities.
        saved=lp.m
        inf=M.cp.COPT.INFINITY
        for method in (2,1,3):
            fresh=M._ENV.createModel("strict_certificate")
            try:
                for k,v in (("Logging",0),("Threads",1),("LpMethod",method),("Presolve",0),
                            ("Scaling",0),("DualPerturb",0),("FeasTol",1e-10),("DualTol",1e-10),("MatrixTol",1e-12)):
                    fresh.setParam(k,v)
                fresh.loadMatrix(lp.c,lp.A,np.maximum(lp.rl,-inf),np.minimum(lp.ru,inf),lp.cl,np.minimum(lp.cu,inf))
                lp.m=fresh
                try:
                    sol=lp.full_solution(check_kkt=True)
                except AssertionError:
                    continue
                if max(sol[k] for k in KEYS)<1e-8:
                    break
            finally:
                lp.m=saved
                fresh.dispose()
    assert sol and sol["solver_status"]==M.cp.COPT.OPTIMAL and max(sol[k] for k in KEYS)<1e-8, (lp.tail_cfg,sol)
    QA["solves"] += 1
    for k in KEYS:
        QA[k] = max(QA[k],sol[k])
    return sol


def rewire(lp, prev):
    lp.rewire(prev)
    lp.A = lp.m.getA().tocsc()


def expanded(arr, multiplier, access):
    out, _, _ = M._access_factorial(arr,multiplier,access)
    mask = arr[:,0] <= 1e-10
    total = math.fsum((arr[mask,2]*multiplier).tolist())
    if access == "expanded" and total:
        # Round the common level across neighboring binary64 values; preserve the budget
        # to the correctly rounded total even for large provincial power ratios.
        positions = np.flatnonzero(mask)
        level = float(total/len(positions))
        delta = np.longdouble(total)-np.longdouble(level)*len(positions)
        adj = np.nextafter(level, np.inf if delta>0 else 0.)
        n = min(len(positions), round(float(delta/(np.longdouble(adj)-level)))) if delta else 0
        out[positions,2]=level
        if n:
            chosen = np.floor((np.arange(n)+.5)*len(positions)/n).astype(int)
            out[positions[chosen],2]=adj
    observed = math.fsum(out[mask,2].tolist())
    assert abs(observed-total)<1e-12
    return out,total,abs(observed-total)


def run_one(index):
    global QA
    QA = dict(solves=0, **dict.fromkeys(KEYS,0.))
    path = OUT/"checkpoints"/f"{index:03d}.pkl"
    rec, load, supply, task, flags, arr = C.inputs(index,50)
    p_ref = .01*load.mean()
    raw = np.column_stack((task,task*np.repeat(flags,24),supply))/p_ref
    assert np.array_equal(arr,M._transform_rule(raw))
    ens = M.ce.ChronologyEnsemble(pd.DataFrame(dict(segment=C.SEASON,event=flags)),rec.seed_key)
    orders = [np.arange(365)] + [np.asarray(ens.orders(t)["clustered"]) for t in range(1,13)]
    for o in orders:
        assert np.array_equal(np.sort(o),np.arange(365))
    uid = rec.uid
    group = "provincial"
    base,caes = M._CAT58["li_lfp_4h"],M._CAT58["caes_24h"]
    is_call, runs, hours = M.WD.segments(arr)
    available = np.where(arr[:,1]>1e-10,0.,arr[:,2])
    ok = (~is_call) & (np.bincount(runs,weights=(arr[:,2]>1e-9))>0)
    def energy(ratio,rte):
        return np.bincount(runs,weights=np.minimum(available,ratio))*rte
    def fill(e, dur):
        return int((ok & (e>=.9*dur-1e-9)).sum())
    e_c,e_l,e_h=energy(1.5,.55),energy(1.,.9),energy(3.,M._H2_RTE)
    result = {"fw":[(uid,group,tech,d,fill(e,d)) for tech,nodes,e in
                     (("caes",M._CAES_DURS,e_c),("h2_tank",M._H2_DURS,e_h)) for d in nodes],
              "cat":[],"fac":[],"ext":[],"wr":[],"cons":[],"price":[],"rule":[]}
    h2qa=dict(solves=0,**dict.fromkeys(KEYS,0.))
    idx = M.CollapsedIndexer(arr)
    comps = ("fixed","variable_om","charging","residual_energy","residual_peak")
    sums = np.array([math.fsum(col) for col in arr.T])
    with branch(idx,base) as lf, branch(idx,caes) as hy:
        sl,sh=solve(lf),solve(hy)
        c_l = sl["cost_usd_kw_year"]
        b=c_l-sh["cost_usd_kw_year"]
        op=b+RM.PREM36[2035]
        terms={"B_"+k:sl[k+"_usd_kw_year"]-sh[k+"_usd_kw_year"] for k in comps}
        assert abs(sum(terms.values())-b)<1e-10
        def delivery(sol):
            target=sol["total_discharge_mwh_per_mw"]
            return brentq(lambda f: np.minimum(arr[:,0],f).sum()-target,0.,float(arr[:,0].max()),xtol=1e-12)
        f_l,f_h=delivery(sl),delivery(sh)
        durations=M._runs(flags)
        sat=M.WD.branch(arr,24,.55,1.5)["saturated"]
        result["rec"]=dict(uid=uid,row_idx=index,group=group,regime="China",province=rec.province,
            unit_key=rec.seed_key,system_id=rec.seed_key.rsplit(":",1)[0].split(":",1)[1],
            observation_year=int(rec.year),in_gas_case_26=bool(rec.in_gas_case_26),P_ref_mw=p_ref,
            raw_supply_mwh=supply.sum(),task_hours=int((arr[:,0]>1e-10).sum()),charge_hours=int((arr[:,2]>1e-9).sum()),
            event_days=int(flags.sum()),events=len(durations),mean_dur=np.mean(durations),longest=max(durations),
            hybrid_window_full=sat,windows=int(ok.sum()),lfp_fill_windows=fill(e_l,4),long_windows=fill(e_c,24),
            one_hour_windows=int((ok & (hours==1)).sum()),lfp_fill_x1=fill(e_l,4),lfp_fill_x4=fill(energy(4,.9),4),
            long_x1=fill(e_c,24),long_x4=fill(energy(6,.55),24),B=b,**terms,
            lfp_total_discharge_mwh_per_mw=sl["total_discharge_mwh_per_mw"],
            hybrid_total_discharge_mwh_per_mw=sh["total_discharge_mwh_per_mw"],
            lfp_tail_discharge_mwh_per_mw=sl["tail_discharge_mwh_per_mw"],
            hybrid_tail_discharge_mwh_per_mw=sh["tail_discharge_mwh_per_mw"],
            additional_delivery_mwh_per_mw=sh["total_discharge_mwh_per_mw"]-sl["total_discharge_mwh_per_mw"],
            efc_acaes=sh["tail_discharge_mwh_per_mw"]/24,efc_lfp=sl["tail_discharge_mwh_per_mw"]/4,
            **{f"roi_{y}":float(RM.roi(op,y)) for y in RM.YEARS},win36=b>0,
            F_lfp=f_l,F_hybrid=f_h,B_equivalent_service=b-terms["B_residual_peak"]+M.SERVICE_PRICE_USD_KW_YR*(f_h-f_l))
        for tech,cfgs in M._TECH_GROUPS.items():
            with branch(idx,cfgs[0][1]) as other:
                for j,(variant,cfg) in enumerate(cfgs):
                    if j: other.set_tail_cfg(cfg)
                    if variant=="li_lfp_4h":
                        val=0.
                    else:
                        sol=solve(other)
                        val=c_l-sol["cost_usd_kw_year"]
                        if tech=="h2_tank":
                            h2qa["solves"]+=1
                            for k in KEYS: h2qa[k]=max(h2qa[k],sol[k])
                    result["cat"].append((uid,group,variant,tech,cfg["duration_h"],val))
        result["price"].append((uid,group,"x1",b))
        for mult in (.5,.75,1.25):
            lf.set_energy_mult(mult)
            hy.set_energy_mult(mult)
            result["price"].append((uid,group,f"x{mult:g}",solve(lf)["cost_usd_kw_year"]-solve(hy)["cost_usd_kw_year"]))
        lf.set_energy_mult(1.)
        hy.set_energy_mult(1.)
        for t,o in enumerate(orders):
            a=M.ce.reorder_days(arr,o) if t else arr
            if t:
                prev=M._prev_from_order(o)
                rewire(lf,prev)
                rewire(hy,prev)
                l,h=solve(lf),solve(hy)
            else:
                l,h=sl,sh
            bt=l["cost_usd_kw_year"]-h["cost_usd_kw_year"]
            cw=M.WD.branch(a,24,.55,1.5)
            result["ext"].append((uid,group,t,bt,cw["saturated"],h["tail_discharge_mwh_per_mw"],
                l["tail_discharge_mwh_per_mw"],h["total_discharge_mwh_per_mw"]-l["total_discharge_mwh_per_mw"],
                l["residual_energy_usd_kw_year"]-h["residual_energy_usd_kw_year"],
                l["residual_peak_usd_kw_year"]-h["residual_peak_usd_kw_year"],np.mean(M._runs(flags[o]))))
            if t in (0,1,4,12):
                lw=M.WD.branch(a,4,.9,1.)
                result["wr"].append((uid,group,t,cw["delivered"],lw["delivered"],h["tail_discharge_mwh_per_mw"],
                    l["tail_discharge_mwh_per_mw"],bt,cw["cyclic_residual"],lw["cyclic_residual"]))
                diff=np.abs(np.array([math.fsum(col) for col in a.T])-sums)
                result["cons"].append((uid,group,t,diff.max(),(diff/np.maximum(1.,np.abs(sums))).max()))
    for multiplier in (1,4):
        for access in ("observed","expanded"):
            a,usable,err=expanded(arr,multiplier,access)
            ci=M.CollapsedIndexer(a)
            with branch(ci,base) as lf, branch(ci,caes) as hy:
                for order,t in (("real",0),("n12",12)):
                    if t:
                        prev=M._prev_from_order(orders[t])
                        rewire(lf,prev)
                        rewire(hy,prev)
                    for power in (1,4):
                        if ci.nq:
                            lf.set_charge_power_mult(power)
                            hy.set_charge_power_mult(power)
                        val=solve(lf)["cost_usd_kw_year"]-solve(hy)["cost_usd_kw_year"]
                        result["fac"].append((uid,"China",rec.province,order,multiplier,power,access,val,usable,err))
    result["rule"].append((uid,group,"registered",b))
    for rule in ("all_task","netted_tail","netted_all_task"):
        g=M.CollapsedIndexer(M._transform_rule(raw,rule))
        with branch(g,base) as lf, branch(g,caes) as hy:
            result["rule"].append((uid,group,rule,solve(lf)["cost_usd_kw_year"]-solve(hy)["cost_usd_kw_year"]))
    result["h2"]=h2qa
    result["qa"]=QA.copy()
    assert h2qa["solves"]==22
    tmp=path.with_suffix(".partial")
    with tmp.open("wb") as f: pickle.dump(result,f,protocol=4)
    tmp.replace(path)
    M._trim_heap()
    return index


def main():
    import argparse, fcntl
    cli=argparse.ArgumentParser()
    cli.add_argument("method",choices=DIRECTORIES)
    cli.add_argument("--workers",type=int,default=16)
    cli.add_argument("--indices",nargs="+",type=int)
    args=cli.parse_args()
    out,arc=setup(args.method)
    with (out/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for y in RM.YEARS: RM.curve(y)
        todo=[i for i in (args.indices or range(310)) if not (out/"checkpoints"/f"{i:03d}.pkl").exists()]
        np.random.default_rng(20261120).shuffle(todo)
        if todo:
            run=parallel(out,todo,run_one,args.workers)
            write_json(out/"checkpoints"/f"execution_{time.time_ns()}.json",run)


if __name__=="__main__":
    main()
