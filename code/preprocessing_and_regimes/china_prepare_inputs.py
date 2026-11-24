"""Read the authors' provincial load workbook and the archived 2020 charging supply."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook

WORK = Path(__file__).resolve().parents[1]
V86 = WORK.parents[1]
ROOT = V86.parents[1]
DATA = WORK / "inputs"
LOAD_BOOK = ROOT / "external_downloads/ndrc_typical_load_2020/Data output.xlsx"
STATIONS = ROOT / "dataset/省份修正合并.geojson"
SUPPLY_DIR = ROOT / "dataset/30日弃电"
OLD_INPUTS = ROOT / "submission/Nature_Energy_v85/revision/remaining_experiments_20261103/inputs"
MERGE = {"Hebei north":"Hebei", "Hebei south":"Hebei",
         "Inner Mongolia east":"Inner Mongolia", "Inner Mongolia west":"Inner Mongolia"}
YEARS = list(range(2015, 2025))


def digest(path, method="sha256"):
    h = hashlib.new(method)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def prepare_load():
    receipt = json.loads((WORK.parent/"china_load_baseline_audit_20261118/sources/local_figshare_receipt.json").read_text())
    assert digest(LOAD_BOOK, "md5") == receipt["file"]["supplied_md5"]
    book = load_workbook(LOAD_BOOK, read_only=True, data_only=True)
    loads, records, names = [], [], None
    for year in YEARS:
        it = iter(book[f"{year} load curve"].iter_rows(max_col=32, values_only=True))
        headings = next(it)
        assert headings[0] == "Unit: GWh"
        provinces = [v.split(" (")[0] for v in headings[1:]]
        if names is None:
            names = provinces
        assert provinces == names and len(set(names)) == 31
        # Author workbook has a formatted blank last row in 2023 and extra
        # formatting beyond the named provincial columns in 2024.
        frame = pd.DataFrame.from_records(it, columns=["hour"]+names).dropna(how="all").set_index("hour")
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        assert frame.index.equals(expected), year
        full = frame.to_numpy(dtype=np.float64)*1000.
        assert np.isfinite(full).all() and (full > 0).all()
        keep = ~((expected.month == 2) & (expected.day == 29))
        arr = np.ascontiguousarray(full[keep].T)
        assert arr.shape == (31, 8760)
        loads.append(arr)
        for i, province in enumerate(names):
            key = f"China:CN-{province}:{year}"
            records.append(dict(row_idx=len(records), year=year, province=province, province_idx=i,
                                uid=hashlib.sha256(key.encode()).hexdigest()[:16], seed_key=key,
                                source_hours=len(expected), analysis_hours=8760,
                                annual_load_mwh_full=float(full[:,i].sum()), annual_load_mwh=float(arr[i].sum()),
                                reference_power_mw=.01*arr[i].mean()))
        print("load", year, len(frame), flush=True)
    book.close()
    return names, pd.DataFrame(records), np.concatenate(loads)


def prepare_supply(provinces):
    geo = json.loads(STATIONS.read_bytes())
    crosswalk = {}
    for feature in geo["features"]:
        props = feature["properties"]
        sid = str(props["ObjectId"]).removesuffix(".0")
        province = MERGE.get(props["State_Province"],props["State_Province"])
        assert sid not in crosswalk and province in provinces
        crosswalk[sid] = province
    monthly, audit = [], []
    for month in range(1,13):
        path = SUPPLY_DIR/f"TotalCurt_Hourly_2020{month:02}.csv"
        raw = pd.read_csv(path, dtype={"ObjectId":str}, encoding="utf-8-sig", index_col="ObjectId")
        keys = raw.index.astype(str).str.strip().str.removesuffix(".0")
        assert keys.is_unique and set(keys) == set(crosswalk)
        dates = pd.to_datetime(raw.columns, format="%Y%m%d%H")
        expected = pd.date_range(f"2020-{month:02}-01", periods=pd.Period(f"2020-{month:02}").days_in_month*24, freq="h")
        assert dates.equals(expected)
        numeric = raw.to_numpy(dtype=np.float64, copy=False)
        assert np.isfinite(numeric).all() and numeric.min() >= 0
        raw.index = [crosswalk[sid] for sid in keys]
        grouped = raw.groupby(level=0,sort=False).sum().loc[provinces]
        # The original station panel has kWh over each one-hour interval.
        mw = grouped.to_numpy(dtype=np.float64)/1000.
        assert np.allclose(mw.sum(axis=0)*1000., numeric.sum(axis=0), rtol=1e-10, atol=1e-5)
        monthly.append(mw)
        audit.append(dict(month=month,filename=path.name,sha256=digest(path),stations=len(keys),
                          hours=len(dates),missing_values=0))
        print("supply", month, len(keys), flush=True)
    full = np.concatenate(monthly,axis=1)
    cal = pd.date_range("2020-01-01",periods=8784,freq="h")
    result = full[:,~((cal.month==2)&(cal.day==29))]
    assert result.shape == (len(provinces),8760)
    return np.ascontiguousarray(result),audit


def main():
    DATA.mkdir(exist_ok=True)
    provinces,meta,loads=prepare_load()
    supply,monthly=prepare_supply(provinces)
    old=pd.read_csv(OLD_INPUTS/"records.csv").query("group == 'provincial'")
    members={row.system_id.removeprefix("CN-"):row for row in old.itertuples(index=False)}
    assert len(members)==26
    meta["in_gas_case_26"]=meta.province.isin(members)
    error=0.
    for province,source in members.items():
        actual=supply[provinces.index(province)]
        with np.load(OLD_INPUTS/f"{source.uid}.npz") as z:
            reference=z["raw"][:,2]*source.P_ref_mw
        error=max(error,float(np.max(np.abs(reference-actual))))
        assert np.allclose(actual,reference,rtol=1e-11,atol=1e-6), province
    assert meta.uid.is_unique and loads.shape==(310,8760)
    np.save(DATA/"china_2015_2024_load_mw.npy",loads)
    np.save(DATA/"china_2020_curtailment_mw.npy",supply)
    meta.to_csv(DATA/"records.csv",index=False)
    pd.DataFrame(dict(province=provinces,annual_curtailment_mwh=supply.sum(axis=1),
                      positive_hours=(supply>0).sum(axis=1),
                      included_in_gas_case_26=[p in members for p in provinces])).to_csv(DATA/"supply_coverage.csv",index=False)
    info=dict(load_source=str(LOAD_BOOK.relative_to(ROOT)),load_source_md5=digest(LOAD_BOOK,"md5"),
              source_doi="10.6084/m9.figshare.29832701.v3",station_map_sha256=digest(STATIONS),
              load_years=YEARS,unique_provinces=len(provinces),annual_records=len(meta),
              source_2020_max_hourly_error_against_archived_26_mw=error,
              monthly_supply=monthly,
              hourly_data_sha256={n:digest(DATA/n) for n in ["china_2015_2024_load_mw.npy","china_2020_curtailment_mw.npy","records.csv"]})
    (DATA/"input_manifest.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n")
    print("verified",len(meta),"annual records,",len(provinces),"supply profiles; overlap max error MW",error,flush=True)


if __name__=="__main__":
    main()
