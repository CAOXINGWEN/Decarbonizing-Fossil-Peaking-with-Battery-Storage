#!/usr/bin/env python3
"""Build the Chinese provincial hourly task--recharge application panel.

Daily provincial gas-peaking energy is distributed uniformly within each day
because the observed task is daily.  Station-level hourly renewable
curtailment is aggregated to provinces through the archived geospatial
crosswalk.  The experiment preserves the 24-hour day as the atomic block and
therefore identifies the effect of inter-day event clustering without
inventing an unobserved intra-day gas profile.

The provincial gas-peaking series is already a separated peaking task.  It is
therefore passed directly to ``task_c1_mw`` without a q10 or other low-load
baseline subtraction.  The seasonal q75 threshold labels extreme peaking days
only and never changes the all-peaking task used by the common 4 h layer.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[5]
MASTER = WORKSPACE / "Decarbonizing-Fossil-Peaking-with-Battery-Storage-master"
DEMAND = MASTER / "data" / "Daily Natural Gas Peaking Power Generation by Province in China.xlsx"
CURTAILMENT = WORKSPACE / "dataset" / "30日弃电"
GEOJSON = WORKSPACE / "dataset" / "省份修正合并.geojson"
OUT = HERE / "china_province_panel"
MERGE = {
    "Hebei north": "Hebei", "Hebei south": "Hebei",
    "Inner Mongolia east": "Inner Mongolia", "Inner Mongolia west": "Inner Mongolia",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def station_crosswalk() -> dict[str, str]:
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    mapping = {}
    for feature in geo.get("features", []):
        props = feature.get("properties", {})
        object_id = str(props.get("ObjectId", "")).strip()
        province = MERGE.get(str(props.get("State_Province", "")).strip(), str(props.get("State_Province", "")).strip())
        if object_id and province:
            mapping[object_id] = province
    return mapping


def aggregate_curtailment(provinces: set[str]) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    timestamps = pd.date_range("2020-01-01 00:00", "2020-12-31 23:00", freq="h")
    timestamps = timestamps[~((timestamps.month == 2) & (timestamps.day == 29))]
    key_to_position = {ts.strftime("%Y%m%d%H"): i for i, ts in enumerate(timestamps)}
    values = {province: np.zeros(len(timestamps), dtype=float) for province in provinces}
    crosswalk = station_crosswalk()
    for path in sorted(CURTAILMENT.glob("TotalCurt_Hourly_*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            positions = np.asarray([key_to_position.get(value, -1) for value in header[1:]], dtype=int)
            keep = positions >= 0
            for row in reader:
                province = crosswalk.get(str(row[0]).strip().removesuffix(".0"))
                if province not in values:
                    continue
                numeric = pd.to_numeric(pd.Series(row[1:]), errors="coerce").fillna(0.0).to_numpy(float)
                values[province][positions[keep]] += numeric[keep]
        print(f"aggregated {path.name}", flush=True)
    return timestamps, values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    demand = pd.read_excel(DEMAND, index_col=0)
    demand.index = demand.index.astype(str)
    dates = pd.to_datetime([f"2021-{str(column)[:2]}-{str(column)[2:]}" for column in demand.columns])
    demand.columns = dates
    timestamps, recharge = aggregate_curtailment(set(demand.index))
    rows: list[pd.DataFrame] = []
    for province in demand.index:
        if province not in recharge or float(recharge[province].sum()) <= 0.0:
            continue
        daily_kwh = pd.to_numeric(demand.loc[province], errors="coerce").fillna(0.0).to_numpy(float)
        raw_task_mw = np.repeat(daily_kwh / 1000.0 / 24.0, 24)
        daily_mwh = raw_task_mw.reshape(365, 24).sum(axis=1)
        day_dates = pd.date_range("2021-01-01", periods=365, freq="D")
        segment = ((day_dates.month % 12) // 3).astype(int)
        threshold = np.zeros(365)
        for value in np.unique(segment):
            threshold[segment == value] = np.quantile(daily_mwh[segment == value], 0.75)
        event = daily_mwh >= threshold
        frame = pd.DataFrame(
            {
                "timestamp_local": pd.date_range("2021-01-01", periods=8760, freq="h"),
                "province": province,
                "year": 1,
                "source_hour_id": np.arange(8760),
                "source_day_id": np.arange(8760) // 24,
                "season_segment": np.repeat(segment, 24),
                "event_flag_q75": np.repeat(event, 24),
                "task_c1_mw": raw_task_mw,
                "task_c3_q75_mw": np.where(np.repeat(event, 24), raw_task_mw, 0.0),
                "r2_potential_recharge_mw": recharge[province] / 1000.0,
                "raw_task_mean_mw": raw_task_mw.mean(),
                "forced_discharge_power_mw": 0.01 * raw_task_mw.mean(),
            }
        )
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel.to_csv(OUT / "china_province_event_task_panel.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary = panel.groupby("province", as_index=False).agg(
        annual_task_mwh=("task_c1_mw", "sum"),
        annual_recharge_mwh=("r2_potential_recharge_mw", "sum"),
        event_hours=("event_flag_q75", "sum"),
    )
    summary.to_csv(OUT / "china_province_input_summary.csv", index=False)
    qa = {
        "demand": str(DEMAND), "demand_sha256": sha256(DEMAND),
        "curtailment_files": [str(path) for path in sorted(CURTAILMENT.glob("TotalCurt_Hourly_*.csv"))],
        "province_count": int(panel.province.nunique()), "rows": len(panel),
        "all_provinces_8760_hours": bool(panel.groupby("province").size().eq(8760).all()),
        "task_nonnegative": bool((panel.task_c1_mw >= 0).all()),
        "recharge_nonnegative": bool((panel.r2_potential_recharge_mw >= 0).all()),
        "task_definition": "observed_provincial_gas_peaking_direct_no_baseline_subtraction",
        "q10_baseline_applied": False,
        "q75_role": "seasonal_extreme_peaking_day_label_only",
        "task_c1_equals_raw_observed_peaking_after_unit_conversion": bool(
            np.allclose(panel.task_c1_mw, np.repeat(demand.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float) / 1000.0 / 24.0, 24, axis=1).reshape(-1))
        ),
        "annual_task_energy_matches_source_after_kwh_to_mwh": bool(
            np.isclose(panel.task_c1_mw.sum(), demand.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float).sum() / 1000.0)
        ),
    }
    (OUT / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
