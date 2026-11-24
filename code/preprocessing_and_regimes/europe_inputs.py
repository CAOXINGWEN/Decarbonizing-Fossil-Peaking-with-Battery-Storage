"""Rebuild q5 national gas tasks through the archived multi-baseline implementation."""
from pathlib import Path
import os
import sys
OUT=Path(__file__).resolve().parents[1]
REV=OUT.parent
sys.path.insert(0,str(REV/"china_load_mc_2015_2024_20261118/results/q50_r4_r5_suite/code"))
from suite_common import C, M, RM, ROOT, VERSION, np, pd, signatures, sha, write_json
V85=ROOT/"submission/Nature_Energy_v85/revision"
BASE=V85/"real_vs_placement_20261029"
sys.path.insert(0,str(BASE/"code"))
import baseline_mc as legacy

INPUT=OUT/"inputs"
def prepare():
    INPUT.mkdir(exist_ok=True)
    receipt=INPUT/"inputs.json"
    units=[u for u in legacy.R.P.run.M["units"] if u["regime"]=="Europe"]
    aux=[legacy.R.P.run.OLD/"cache"/(u["uid"]+".pkl") for u in units]
    source=signatures([Path(__file__),Path(legacy.__file__),Path(legacy.T.__file__),Path(legacy.O.__file__),
        Path(legacy.R.__file__),Path(M.ce.__file__),legacy.T.NAT,BASE/"results/baseline_test.csv"]+aux)
    if receipt.exists():
        import json
        meta=json.loads(receipt.read_text())
        assert meta["source"]==source and all(sha(INPUT/k)==v for k,v in meta["outputs"].items())
        return
    assert len(units)==132
    archive=pd.read_csv(BASE/"results/baseline_test.csv",dtype={"uid":str},float_precision="round_trip")
    archive=archive.query("regime=='Europe' and q==5").set_index("uid")
    assert len(archive)==132 and (archive.B>0).sum()==7
    rows=[]
    matrix=np.empty((132,8760,3))
    flags=np.empty((132,365),dtype=bool)
    segments=np.empty((132,365),dtype=np.int8)
    for idx,u in enumerate(units):
        uid=u["uid"]
        arr,ens,reg,same=legacy.inputs(uid,5)
        seg=ens.daily.segment.to_numpy()
        assert ens.unit_key==u["unit_key"] and reg=="Europe"
        assert np.array_equal(seg,np.repeat([1,2,3,4,5],[59,92,92,91,31]))
        flags[idx]=ens.daily.event.to_numpy()
        segments[idx]=seg
        matrix[idx]=arr
        old=archive.loc[uid]
        assert np.isfinite(arr).all() and (arr>=0).all()
        assert int((arr[:,0]*u["P_ref_mw"]>1e-10).sum())==old.task_hours
        rows.append(dict(row_idx=idx,uid=uid,unit_key=ens.unit_key,system_id=u["system_id"],
          observation_year=int(u["observation_year"]),P_ref_mw=u["P_ref_mw"],reference_B=old.B,
          q=5,task_hours=int(old.task_hours),charge_hours=int(old.charge_hours),task_share=old.task_share,
          event_days=int(flags[idx].sum()),mean_dur=old.mean_dur,archive_saturated=int(old.caes_full)))
    for name,a in (("hourly",matrix),("flags",flags),("segments",segments)):
        np.save(INPUT/(name+".npy"),a)
    pd.DataFrame(rows).to_csv(INPUT/"records.csv",index=False)
    files=["hourly.npy","flags.npy","segments.npy","records.csv"]
    write_json(receipt,dict(source=source,outputs={n:sha(INPUT/n) for n in files},records=len(rows),
        segment_lengths=[59,92,92,91,31],q75_comparison="strict greater",solver=VERSION))
    print("Archived European q5 inputs rebuilt for 132 country-years",flush=True)


if __name__=="__main__":
    prepare()
