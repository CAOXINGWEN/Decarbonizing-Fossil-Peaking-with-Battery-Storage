"""Match C3S ADM1 columns and retain the Ember provincial generation amounts used here."""
from hashlib import sha256
from pathlib import Path
import json
import re
import struct
import pandas as pd

WORK=Path(__file__).resolve().parents[1]
OUT=WORK/"sources"
ENERGY_URL="https://files.ember-energy.org/public-downloads/generation/outputs/release_chn_subnational_generation_"
ADM_URL="https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"


def prepare():
    data=Path("/tmp/ne_10m_admin_1_states_provinces.dbf").read_bytes()
    count,header,size=struct.unpack_from("<IHH",data,4)
    fields=[(data[i:i+11].split(bytes([0]))[0].decode(),data[i+16]) for i in range(32,header-1,32)]
    records=[]
    for row in range(count):
        i=header+row*size+1
        record={}
        for name,width in fields:
            if name in ("adm1_code","iso_3166_2","name_en","name_zh"):
                record[name]=data[i:i+width].decode("utf-8","replace").strip("\0 ")
            i+=width
        if record["adm1_code"].startswith("CHN-"):
            records.append(record)
    frame=pd.DataFrame(records).rename(columns={"name_en":"province","name_zh":"zh_name"})
    frame["zh_short"]=frame.zh_name.str.replace(r"(维吾尔自治区|壮族自治区|回族自治区|自治区|省|市)$","",regex=True)
    provinces=set(pd.read_csv(WORK/"inputs/records.csv").province)
    assert set(frame.province)==provinces and frame.adm1_code.nunique()==frame.zh_short.nunique()==31
    frame.sort_values("province").to_csv(OUT/"c3s_province_codes.csv",index=False)
    receipt={"admin1":{"url":ADM_URL,"dbf_sha256":sha256(data).hexdigest()},"ember":{}}
    for freq,filename in [("yearly","year"),("monthly","month")]:
        p=Path(f"/tmp/ne86_ember_gen_{filename}.csv")
        frame=pd.read_csv(p)
        frame=frame[frame["Electricity source"].isin(["Wind","Solar"]) & frame.Province.ne("China (total)")].copy()
        frame["province"]=frame.Province.replace({"Tibet (Xizang)":"Tibet"})
        frame["year"]=frame.Year if freq=="yearly" else pd.to_datetime(frame.Date).dt.year
        if freq=="monthly": frame["month"]=pd.to_datetime(frame.Date).dt.month
        frame["tech"]=frame["Electricity source"].map({"Wind":"wind","Solar":"pv"})
        frame["generation_twh"]=frame["Generation (TWh)"]
        columns=["province","tech","year"]+(["month"] if freq=="monthly" else [])
        frame=frame[frame.year.between(2015,2024)].copy()
        assert not frame.duplicated(columns).any() and set(frame.province)==provinces
        assert frame.generation_twh.ge(0).all()
        frame[columns+["generation_twh"]].sort_values(columns).to_csv(OUT/f"ember_provincial_generation_{freq}.csv",index=False)
        receipt["ember"][freq]=dict(url=ENERGY_URL+freq+".csv",original_sha256=sha256(p.read_bytes()).hexdigest(),retained_rows=len(frame))
    (OUT/"generation_source_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    prepare()
