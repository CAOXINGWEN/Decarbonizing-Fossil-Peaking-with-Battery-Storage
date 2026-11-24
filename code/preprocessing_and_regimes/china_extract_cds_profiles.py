"""Select linked provincial columns from C3S originals and verify the UTC chronology."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from prepare_generation_sources import WORK

CACHE=Path("/tmp/ne86_cds_energy_profiles")
LOCAL=WORK/"inputs/c3s"
LINKS=WORK/"sources/c3s_province_codes.csv"


def extract(path,meta):
    target=LOCAL/(path.stem+".npz")
    if target.exists(): return
    year,kind,months=path.stem.split("_")
    tags={"pv":"_SPV_","wind":"_WON_"}
    keys=meta.adm1_code.tolist()
    parts=[]
    members=[]
    with ZipFile(path) as z:
        for f in sorted(z.namelist()):
            assert tags[kind] in f and "_Glob_ADM1_" in f and "_01h_" in f and f.endswith(".csv"),f
            with z.open(f) as stream:
                db=pd.read_csv(stream,comment="#",usecols=["Date"]+keys,parse_dates=["Date"]).set_index("Date")[keys]
            parts.append(db)
            members.append(f)
    df=pd.concat(parts).sort_index()
    start,end=map(int,months.split("-"))
    raw_start=pd.Timestamp(int(year),start,1)
    raw_end=pd.Timestamp(int(year),end,1)+pd.offsets.MonthBegin(1)
    assert df.index.equals(pd.date_range(raw_start,raw_end,freq="h",inclusive="left"))
    v=df.to_numpy(dtype=np.float32)
    assert np.isfinite(v).all() and (v>=0).all() and (v<=1.00001).all() and (v.sum(axis=0)>0).all()
    temp=target.with_suffix(".partial.npz")
    np.savez_compressed(temp,utc_hours=df.index.to_numpy(dtype="datetime64[h]").astype("int64"),
            cf=v,province=meta.province.to_numpy(dtype="U24"))
    temp.replace(target)
    receipt=dict(original_archive=path.name,archive_sha256=sha256(path.read_bytes()).hexdigest(),
            members=members,local_utc_min=str(df.index.min()),local_utc_max=str(df.index.max()),
            n_hours=len(df),n_provinces=len(keys),extracted_sha256=sha256(target.read_bytes()).hexdigest(),
            request=json.loads(path.with_suffix(".json").read_text()))
    target.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(target.name,len(df),flush=True)


if __name__=="__main__":
    LOCAL.mkdir(exist_ok=True)
    provinces=pd.read_csv(WORK/"inputs/records.csv").query("year==2015").sort_values("province_idx").province
    meta=pd.read_csv(LINKS).set_index("province").loc[provinces].reset_index()
    for file in sorted(CACHE.glob("*.zip")):
        if file.with_suffix(".json").exists(): extract(file,meta)
