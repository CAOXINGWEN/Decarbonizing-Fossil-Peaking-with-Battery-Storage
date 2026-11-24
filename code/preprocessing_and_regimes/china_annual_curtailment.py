"""Build an auditable 31 x 10 x 2 annual curtailment ledger, in TWh."""
from pathlib import Path
from zipfile import ZipFile
import json
import re
from bs4 import BeautifulSoup
from lxml import etree
import numpy as np
import pandas as pd
from prepare_inputs import WORK, STATIONS

SRC=WORK/"sources"
A=SRC/"annual_reports"
PROV=pd.read_csv(SRC/"c3s_province_codes.csv").set_index("zh_short").province.to_dict()
GEN=pd.read_csv(SRC/"ember_provincial_generation_yearly.csv").set_index(["year","province","tech"]).generation_twh
# Quantities printed on the national monitoring centre's regional maps, in 100 million kWh.
# East and west Inner Mongolia are added. Unlabelled map cells are resolved separately.
MAP={
 (2019,"pv"):{"新疆":10.7,"甘肃":5.1,"内蒙古":2.4+.1,"吉林":.5,"河北":2.3,"山西":.2,"山东":.1,
              "宁夏":4,"青海":12.2,"陕西":3.7,"西藏":4.1,"贵州":.1,"云南":.2},
 (2020,"wind"):{"新疆":49.7,"甘肃":16.8,"内蒙古":33.3+6.3,"黑龙江":.7,"吉林":3.2,"辽宁":1.9,
             "青海":4.1,"宁夏":4.4,"陕西":3.2,"山西":8.1,"河北":18.3,"山东":8.4,"河南":.2,
             "湖南":5.8,"贵州":.3,"云南":1.5},
 (2020,"pv"):{"新疆":7.2,"甘肃":2.9,"内蒙古":5.1+.2,"吉林":.6,"青海":14.4,"宁夏":3.5,"陕西":3.6,
             "山西":4.8,"河北":2.8,"山东":2.,"西藏":4.8,"贵州":.3,"云南":.3},
 (2021,"wind"):{"新疆":43.2,"甘肃":12.2,"内蒙古":50.6+10.3,"黑龙江":3.1,"吉林":4.2,"辽宁":4.7,
             "青海":15.6,"宁夏":7,"陕西":4.1,"山西":12,"河北":24.9,"山东":6.1,"河南":5.6,
             "湖南":1.5,"江西":.1,"贵州":.5,"云南":.2},
 (2021,"pv"):{"新疆":3.2,"甘肃":2.2,"内蒙古":5.6+.3,"黑龙江":.2,"吉林":.6,"辽宁":.2,"青海":33.4,
             "宁夏":4.7,"陕西":2.9,"西藏":4.3,"山西":1.8,"河北":5.1,"山东":2.7,"河南":.2,"贵州":.3,"云南":.1},
}
NATIONAL={(2015,"wind"):33.9,(2016,"wind"):49.7,(2017,"wind"):41.9,(2018,"wind"):27.7,(2019,"wind"):16.86,
          (2015,"pv"):4.9,(2016,"pv"):7.4,(2017,"pv"):7.3,(2018,"pv"):5.49,(2019,"pv"):4.57,
          (2020,"wind"):16.61,(2020,"pv"):5.26,(2021,"wind"):20.61,(2021,"pv"):6.78}


def docx_tables(file):
    with ZipFile(file) as z:
        tree=etree.fromstring(z.read("word/document.xml"))
    return [[["".join(c.xpath('.//*[local-name()="t"]/text()')).strip() for c in tr.xpath('./*[local-name()="tc"]')]
             for tr in t.xpath('./*[local-name()="tr"]')] for t in tree.xpath('//*[local-name()="tbl"]')]


def early_wind(y):
    soup=BeautifulSoup((A/f"wind_{y}.html").read_bytes(),"html.parser")
    if y in (2017,2018):
        title=next(t for t in soup.find_all(string=True) if t.strip()==f"{y}年风电并网运行统计数据")
        cells=["".join(x.get_text(strip=True).split()) for x in title.find_parent("p").find_next_siblings("p")]
        return {v:cells[i+3] for i,v in enumerate(cells[:-5]) if v in PROV}
    table=soup.find_all("table")[-1]
    rows=[["".join(c.get_text(strip=True).split()) for c in r.find_all(["td","th"],recursive=False)] for r in table.find_all("tr")]
    col=rows[0].index("弃风电量")
    return {r[0]:r[col] for r in rows[1:] if r[0] in PROV}


def rates():
    result={}
    for report,yrange in [(2021,[2020,2021]),(2022,[2021,2022])]:
        tbl=docx_tables(A/f"{report}_1.docx")
        for tech,table in zip(["wind","pv"],tbl[4:6]):
            for area,v1,v2 in table[1:]:
                for y,v in zip(yrange,[v1,v2]):
                    p=float(v.strip("%"))/100
                    key=(y,tech,area)
                    if key in result: assert result[key]==p
                    result[key]=p
    for r in pd.read_csv(SRC/"nea_2023_2024_regional_utilization.csv").itertuples():
        result[(r.year,"pv" if r.technology=="solar_pv" else "wind",r.region_as_published)]=r.annual_utilization_pct/100
    return result


def east_shares():
    meta=pd.DataFrame(f["properties"] for f in json.loads(STATIONS.read_bytes())["features"])
    m=meta[meta.State_Province.isin(["Inner Mongolia east","Inner Mongolia west"])].copy()
    m["start"]=pd.to_numeric(m.Start_year,errors="coerce")
    m["end"]=pd.to_numeric(m.Retired_year,errors="coerce")
    m["tech"]=np.where(m.ObjectId.astype(str).str.startswith("W"),"wind","pv")
    m["capacity_mw"]=pd.to_numeric(m.Capacity__MW_,errors="coerce")
    shares={}
    for y in range(2022,2025):
        valid=m[m.start.le(y)&(m.end.isna()|m.end.gt(y))&m.Status.isin(["operating","retired","mothballed"])]
        for tech,g in valid.groupby("tech"):
            cap=g.groupby("State_Province").capacity_mw.sum()
            shares[y,tech]=cap["Inner Mongolia east"]/cap.sum()
    return shares


def main():
    data={}
    def record(y,tech,zh,est,method,source,**extra):
        key=(y,PROV[zh],tech)
        assert key not in data and np.isfinite(est) and est>=0,key
        data[key]=dict(year=y,province=PROV[zh],tech=tech,annual_curtailment_twh=est,method=method,source=source,
                  generation_twh=GEN.get(key,np.nan),**extra)
    for y in range(2015,2020):
        vals=early_wind(y)
        assert set(vals)==set(PROV)
        for zh,raw in vals.items():
            if raw: method="reported_volume"
            elif y==2018: method="source_explicit_blank_means_zero"
            else: method="blank_in_full_wind_table_zero_at_report_precision"
            record(y,"wind",zh,float(raw or 0)/10,method,f"wind_{y}.html")
    early=docx_tables(A/"2015_1.docx")[3]
    raw={r[0]:float(r[1])/10 for r in early[1:] if r[0] in PROV}
    assert abs(sum(raw.values())-NATIONAL[2015,"pv"])<1e-12
    for zh in PROV:
        record(2015,"pv",zh,raw.get(zh,0),"reported_volume" if zh in raw else "unlisted_zero_by_report_total","2015_1.docx table 4")
    # Northwest plus the two Shaanxi utilities where separately reported.
    selected={2016:({"陕西":1.40,"甘肃":25.78,"青海":8.13,"宁夏":4.03,"新疆":31.08},"northwest_2016.html"),
              2017:({"陕西":7.53,"宁夏":4.84,"青海":7.50,"新疆":28.2,"甘肃":18.5},"nw_2017_regulator.html + solar_2017.html"),
              2018:({"陕西":4.16+1.16,"甘肃":10.40,"青海":6.51,"宁夏":4.15,"新疆":21.92},"nw_2018_pv_table.png")}
    ref=MAP[2019,"pv"]
    assert abs(sum(ref.values())/10-NATIONAL[2019,"pv"])<1e-12
    for zh in PROV:
        record(2019,"pv",zh,ref.get(zh,0)/10,"reported_volume" if zh in ref else "unlabelled_zero_by_map_total",
               "monitor_2019_pv_map.jpg")
    for y,(amounts,source) in selected.items():
        for zh,v in amounts.items():
            note="The NEA 2017 Xinjiang figure expressly excludes the Production and Construction Corps." if (y,zh)==(2017,"新疆") else ""
            record(y,"pv",zh,v/10,"reported_volume",source,scope_note=note)
        if y==2018:
            # 2019 NEA press release gives the 24.1% rate and 19.5 percentage point fall versus 2018.
            r=(24.1+19.5)/100
            record(y,"pv","西藏",GEN[y,"Tibet","pv"]*r/(1-r),"rate_from_reported_year_change",
                   "solar_2019.html and Ember",published_or_derived_curtailment_rate=r)
        residual=NATIONAL[y,"pv"]-sum(v["annual_curtailment_twh"] for k,v in data.items() if (k[0],k[2])==(y,"pv"))
        assert residual>0
        missing=[k for k in PROV if (y,PROV[k],"pv") not in data]
        # The 2019 national map identifies the first geographically exhaustive PV distribution.
        # Preserve those regional propensities but scale by the contemporaneous local generation.
        weights={k:(ref.get(k,0)/10 /GEN[2019,PROV[k],"pv"]*GEN[y,PROV[k],"pv"] if ref.get(k,0)>0 else 0.) for k in missing}
        for zh,v in weights.items():
            record(y,"pv",zh,residual*v/sum(weights.values()),"national_remainder_allocated",
                   f"national {y} PV bulletin; 2019 regional PV map; contemporaneous Ember generation",
                   remainder_pool_twh=residual,scope_note="A zero allocation here is modelled, not observed.")
    late=rates()
    split=east_shares()
    for y in range(2020,2025):
        for tech in ("wind","pv"):
            known=MAP.get((y,tech),{})
            for zh,prov in PROV.items():
                gen=GEN[y,prov,tech]
                name="solar" if tech=="pv" else tech
                if zh in known:
                    record(y,tech,zh,known[zh]/10,"reported_volume",f"monitor_{y}_{tech}_map",
                           scope_note="East and west summed where applicable.")
                    continue
                src=("2021_1.docx tables 5–6" if y<=2021 else "2022_1.docx tables 5–6" if y==2022 else "2024_1.doc tables 3–4")
                if zh=="内蒙古":
                    east,west=late[y,tech,"蒙东"],late[y,tech,"蒙西"]
                    share=split[y,tech]
                    cost=gen*(share*(1/east-1)+(1-share)*(1/west-1))
                    record(y,tech,zh,cost,"regional_rates_capacity_weighted","Ember; "+src,
                          inner_mongolia_east_capacity_share=share,min_east_west_estimate_twh=gen*(1/max(east,west)-1),
                          max_east_west_estimate_twh=gen*(1/min(east,west)-1))
                else:
                    u=late[y,tech,zh]
                    record(y,tech,zh,gen*(1/u-1),"published_full_utilization_rounded" if u==1. else "reported_rate_times_generation",
                           src+"; Ember",published_or_derived_curtailment_rate=1-u)
    frame=pd.DataFrame(data.values()).sort_values(["year","province","tech"])
    assert len(frame)==620 and not frame.duplicated(["year","province","tech"]).any()
    frame.to_csv(WORK/"inputs/annual_curtailment_ledger.csv",index=False)
    checks=[]
    for (y,t),d in frame.groupby(["year","tech"]):
        total=d.annual_curtailment_twh.sum();official=NATIONAL.get((y,t),np.nan)
        if y in (2015,2016,2017,2018,2019) and t=="pv": assert abs(total-official)<1e-9
        checks.append(dict(year=y,tech=t,provincial_total_twh=total,published_national_twh=official,
             difference_twh=total-official,remainder_twh=d.loc[d.method.eq("national_remainder_allocated"),"annual_curtailment_twh"].sum()))
    pd.DataFrame(checks).to_csv(WORK/"inputs/annual_calibration_checks.csv",index=False)
    print(frame.method.value_counts().to_string(),flush=True)


if __name__=="__main__":
    main()
