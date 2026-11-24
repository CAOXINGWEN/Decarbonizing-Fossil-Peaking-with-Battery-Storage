"""Check the q5 reference and every seed and publish the European experiment."""
from pathlib import Path
import json
import resource
import time
import run_europe as E
import h5py
from europe_inputs import OUT,INPUT,ROOT,BASE,legacy,C,M,RM,np,pd,VERSION
from suite_common import finance,signatures,sha,write_json
from hourly_oracle import full_cost,common


def main():
    start=time.monotonic()
    E.setup()
    rec=E.META.copy()
    b=np.empty((132,1000)); lw=np.empty((132,1000),dtype=np.int16); sat=np.empty_like(lw)
    real=[]; longs=[]; full=[]; certificates=[];cold=0;solves=0
    for i in range(132):
        with np.load(E.CHECK/f"{i:03d}.npz",allow_pickle=False) as z:
            assert str(z["uid"])==rec.uid.iat[i]
            b[i],lw[i],sat[i]=z["B"],z["long_windows"],z["saturated"]
            real.append(float(z["real_B"])); longs.append(int(z["lw_real"]));full.append(int(z["sat_real"]))
            certificates.append(z["max_residuals"])
            solves+=int(z["nsolves"]);cold+=int(z["cold"])
    assert solves==132*2002 and b.shape==(132,1000) and np.isfinite(b).all()
    assert np.all(lw>=0) and np.all(sat>=0) and np.max(certificates)<1e-8
    rec["B"]=real
    rec["long_windows"]=longs;rec["hybrid_window_full"]=full
    maxnative=float(abs(rec.B-rec.reference_B).max())
    assert (rec.B>0).sum()==7 and np.array_equal(rec.B>0,rec.reference_B>0)
    ranks=(b<rec.B.to_numpy()[:,None]).mean(1)+.5*(b==rec.B.to_numpy()[:,None]).mean(1)
    summary,place=finance(rec,b,sat,5,"national",ranks)
    place["lw_mc_med"]=np.median(lw,axis=1)
    details=rec.merge(place,on="uid",validate="one_to_one")
    details["regime"]="Europe"
    details["real_B"]=details.B
    details["roi_2035"]=details.roi_real
    details["mc_win_count"]=(b>0).sum(1)
    details["mc_winning_draws"]=details.mc_win_count
    details["placements"]=1000
    for j,q in enumerate((.05,.5,.95)):
        details[f"mc_B_q{int(q*100):02d}"]=np.quantile(b,q,axis=1)
        details[f"mc_roi_2035_q{int(q*100):02d}"]=place[["mc_q05","mc_q50","mc_q95"][j]].to_numpy()
    details["saturated_mc_med"]=np.median(sat,axis=1)
    draws=int((b>0).sum())
    assert np.isclose(summary.loc[summary.year==2035,"mc_win_3.6"].iat[0],draws/b.size)
    meta=rec.set_index("row_idx")
    # Closest financial margin plus nine distinct country-years spanning winning probabilities.
    near=np.unravel_index(np.abs(b).argmin(),b.shape)
    pool=np.argsort((b>0).mean(1)+np.arange(132)*1e-12)
    pool=pool[pool!=near[0]]
    targets=[(int(near[0]),int(near[1]))]+[(int(pool[j]),s) for j,s in zip(
        np.linspace(0,len(pool)-1,9).astype(int),(0,3,17,42,137,291,503,771,999))]
    checks=[]
    for idx,seed in targets:
        u=meta.loc[idx]
        # Independently load legacy q5 inputs, calendar and random-order generator.
        a,ens,reg,_=legacy.inputs(u.uid,5)
        assert np.array_equal(a,E.ARR[idx]) and np.array_equal(ens.daily.event,E.FLAGS[idx])
        o=np.asarray(legacy.O.obs_random_order(ens,seed))
        arr=legacy.ce.reorder_days(a,o)
        reference=full_cost(arr,"li_lfp_4h")-full_cost(arr,"caes_24h")
        error=abs(reference-b[idx,seed])
        assert error<1e-10 and (reference>0)==(b[idx,seed]>0),(u.uid,seed,error)
        assert M._long_windows_fast(arr)==lw[idx,seed] and M.WD.branch(arr,24,.55,1.5)["saturated"]==sat[idx,seed]
        checks.append(dict(uid=u.uid,seed=seed,B=b[idx,seed],hourly_B=reference,absolute_error=error))
    # Cross-baseline control of RNG compatibility against archived individual q20 draws.
    historical=[]
    for u in rec.itertuples():
        if len(historical)>=3: break
        path=legacy.path(u.uid,20,-1)
        if not path.exists(): continue
        stored=pd.read_csv(path)
        if "B" not in stored or stored.same_as_q10.all(): continue
        seed=7
        a,ens,reg,_=legacy.inputs(u.uid,20)
        o=legacy.O.obs_random_order(ens,seed)
        compact_order=M._fast_obs_random_order(ens.unit_key,E.segments(ens.daily.event.to_numpy(),ens.daily.segment.to_numpy()),seed)
        assert np.array_equal(o,compact_order)
        arr=legacy.ce.reorder_days(a,o)
        val=full_cost(arr,"li_lfp_4h")-full_cost(arr,"caes_24h")
        error=abs(val-stored.loc[stored.seed==seed,"B"].iat[0])
        assert error<1e-10
        historical.append(dict(uid=u.uid,q=20,seed=seed,absolute_error=error,archive_sha256=sha(path)))
    assert len(historical)==3
    file=OUT/"europe_q5_seed_matrix.h5"
    tmp=file.with_suffix(".partial.h5")
    with h5py.File(tmp,"w") as h:
        h.attrs.update(group="national",regime="Europe",q=5,technology_cost_year=2035,real_financing_rate=.036,
            B_definition="pure LFP cost minus hybrid cost in USD/(added kW-year)",long_windows="empty-start filling criterion",
            saturated_windows="cyclic-inventory saturation criterion")
        h.create_dataset("uid",data=np.asarray(rec.uid,dtype="S16"))
        h.create_dataset("row_idx",data=np.arange(132));h.create_dataset("seed",data=np.arange(1000,dtype=np.int16))
        for name,values in (("B",b),("long_windows",lw),("saturated_windows",sat)):
            h.create_dataset(name,data=values,compression="gzip",shuffle=True,fletcher32=True,chunks=(1,1000))
    tmp.replace(file)
    with h5py.File(file) as h:
        assert np.array_equal(h["B"][:],b) and np.array_equal(h["long_windows"][:],lw) and np.array_equal(h["saturated_windows"][:],sat)
        assert np.array_equal((h["B"][:]>0).sum(1),details.mc_win_count)
    details.to_csv(OUT/"europe_q5_records.csv.gz",index=False)
    summary.to_csv(OUT/"europe_q5_summary.csv",index=False)
    selftest={"random_placements":checks,"max_abs_err":max(row["absolute_error"] for row in checks),
        "legacy_q20_draw_controls":historical,"reference_source_sha256":signatures([common.model.__file__,
        common.model.legacy.__file__,Path(__import__("hourly_oracle").__file__)])}
    cert=np.max(certificates,axis=0).tolist()
    write_json(OUT/"validation.json",dict(archive_native_wins=7,current_native_wins=int((rec.B>0).sum()),native_max_abs_error=maxnative,
        max_physical_residual=cert[0],max_stationarity_residual=cert[1],max_duality_gap_eur_kw_year=cert[2],
        all_optimal=True,certified_lp_results=solves,successful_cold_resolves=cold,
        observation_keys_and_five_season_orders_verified=True,hdf5_reread=True,selftest=selftest))
    txt=f"""# 欧洲 132 个国家年 q5 基线与事件日期蒙特卡洛实验

欧洲样本包括 {rec.system_id.nunique()} 个国家的 132 个国家年。放电任务为逐小时燃气发电量超过同月同钟点第 5 百分位的正差。按原项目的五个连续季节段（1—2 月、3—5 月、6—8 月、9—11 月、12 月），取日任务电量严格大于段内第 75 百分位的日期识别事件。每个国家年使用相同的原始记录键与种子 0—999，保持各事件长度、次序以及普通日相对次序。充电资源、放电任务和标记以完整日块共同移动。

按 2035 年设备成本和 3.6% 实际资金成本，混合组合在原始日期胜出 7/132（{7/132:.2%}），在随机放置中胜出 {draws:,}/132,000（{draws/132000:.4%}）。混合组合由共同 4 h LFP 基础层和新增 24 h A-CAES 组成，对照为共同基础层再增配 4 h LFP。B 等于两组合的年化系统成本差，B 大于零计为混合胜出。

逐条原始成本差与 q5 归档结果的最大绝对差为 {maxnative:.3g} USD/(kW·年)。独立 8,760 小时模型复算了 10 个随机放置，含最接近成本持平的种子，最大成本差误差为 {selftest['max_abs_err']:.3g}。另与既有 q20 逐种子归档进行了三次对照。

[三个设备成本年及全贴现率汇总](europe_q5_summary.csv) · [逐国家年结果](europe_q5_records.csv.gz) · [逐种子成本差与窗口矩阵](europe_q5_seed_matrix.h5) · [验证记录](validation.json)
"""
    report=OUT/"欧洲132国家年_q5蒙特卡洛实验摘要.md"
    report.write_text(txt)
    runs=[json.loads(p.read_text()) for p in sorted(E.CHECK.glob("execution_*.json"))]
    peak=max([x["peak_rss_bytes"] for x in runs]+[resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024])
    assert peak<30e9
    outputs=[OUT/n for n in ("europe_q5_records.csv.gz","europe_q5_seed_matrix.h5","europe_q5_summary.csv","validation.json","run_contract.json")]+[report]
    write_json(OUT/"complete.json",dict(status="complete",solver=VERSION,q=5,records=132,seeds=1000,placements=132000,
        native_wins=7,mc_wins=draws,mc_win_fraction=draws/b.size,all_draws_finite=True,
        native_archive_max_abs_difference=maxnative,selftest=selftest,peak_rss_bytes=peak,
        execution_attempts=runs,dispatch_seconds=sum(v["elapsed_seconds"] for v in runs),publication_seconds=time.monotonic()-start,
        input_provenance=json.loads((INPUT/"inputs.json").read_text()),source_sha256=json.loads((OUT/"run_contract.json").read_text())["source_sha256"],
        reporting_code_sha256=signatures([Path(__file__)]),output_sha256={p.name:sha(p) for p in outputs}))
    print(OUT,draws,selftest["max_abs_err"],flush=True)


if __name__=="__main__":
    main()
