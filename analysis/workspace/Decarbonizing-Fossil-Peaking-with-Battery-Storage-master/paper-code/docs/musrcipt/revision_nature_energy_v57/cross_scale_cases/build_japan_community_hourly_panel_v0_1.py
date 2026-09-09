#!/usr/bin/env python3
"""Build an hourly community PV/import panel from Hitotsubashi mesh data.

Ten 500-m meshes with complete numeric low-voltage records are selected in
each of five prefectures using the first delivery month.  The same frozen mesh
set is then read for two complete Japanese fiscal years.  Half-hourly grid
supply is the community import task input and reverse-flow energy is the
potential local battery recharge input.  February 29 is removed so every
community-year contains 365 intact 24-hour blocks.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[5]
ARCHIVE = WORKSPACE / "dataset" / "一桥数据" / "2025-09-30_Hitotsubashi University_Self-Consumption (Low-Voltage).zip"
OUT = HERE / "japan_community_panel"
N_MESHES = 10
PREFECTURES = {
    "11": "Saitama",
    "12": "Chiba",
    "13": "Tokyo",
    "14": "Kanagawa",
    "22": "Shizuoka",
}
FIELDS = ["メッシュ", "年月日時分", "自家消費量", "契約口数", "電力量_供給", "発電設備容量", "電力量_受電"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member_map(outer: ZipFile) -> dict[tuple[str, str], str]:
    result = {}
    for name in outer.namelist():
        if "solar_low-" not in name or not name.endswith(".zip"):
            continue
        stem = Path(name).stem
        _, prefecture, yearmonth = stem.split("-")
        result[(prefecture, yearmonth)] = name
    return result


def row_stream(outer: ZipFile, member: str):
    payload = outer.read(member)
    with ZipFile(io.BytesIO(payload)) as inner:
        csv_name = inner.namelist()[0]
        with inner.open(csv_name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            yield from csv.DictReader(text)


def numeric(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def choose_meshes(outer: ZipFile, member: str) -> list[str]:
    chosen: list[str] = []
    current = None
    rows: list[dict[str, str]] = []

    def evaluate(mesh: str | None, block: list[dict[str, str]]) -> None:
        if mesh is None or mesh == "00_all" or len(chosen) >= N_MESHES:
            return
        supply = np.asarray([numeric(row["電力量_供給"]) for row in block])
        reverse = np.asarray([numeric(row["電力量_受電"]) for row in block])
        contracts = np.asarray([numeric(row["契約口数"]) for row in block])
        complete = np.isfinite(supply) & np.isfinite(reverse) & np.isfinite(contracts)
        if complete.mean() >= 0.95 and np.nanmedian(contracts) >= 5 and np.nansum(reverse) > 0:
            chosen.append(mesh)

    for row in row_stream(outer, member):
        mesh = row["メッシュ"]
        if current is None:
            current = mesh
        if mesh != current:
            evaluate(current, rows)
            if len(chosen) >= N_MESHES:
                break
            current, rows = mesh, []
        rows.append(row)
    evaluate(current, rows)
    if len(chosen) != N_MESHES:
        raise RuntimeError(f"Only {len(chosen)} eligible meshes found in {member}")
    return chosen


def extract_member(outer: ZipFile, member: str, selected: set[str], prefecture: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    remaining = set(selected)
    current = None
    for row in row_stream(outer, member):
        mesh = row["メッシュ"]
        if current is not None and mesh != current and current in remaining:
            remaining.remove(current)
            if not remaining:
                break
        current = mesh
        if mesh not in selected:
            continue
        output.append(
            {
                "prefecture": prefecture,
                "mesh_id": mesh,
                "timestamp_jst": row["年月日時分"],
                "self_consumption_kwh_halfhour": numeric(row["自家消費量"]),
                "contracts": numeric(row["契約口数"]),
                "grid_import_kwh_halfhour": numeric(row["電力量_供給"]),
                "pv_capacity_kw": numeric(row["発電設備容量"]),
                "reverse_flow_kwh_halfhour": numeric(row["電力量_受電"]),
            }
        )
    return output


def build_model_panel(hourly: pd.DataFrame) -> pd.DataFrame:
    hourly["timestamp_jst"] = pd.to_datetime(hourly["timestamp_jst"])
    hourly = hourly[~((hourly.timestamp_jst.dt.month == 2) & (hourly.timestamp_jst.dt.day == 29))].copy()
    hourly["fiscal_year"] = np.where(hourly.timestamp_jst.dt.month >= 4, hourly.timestamp_jst.dt.year, hourly.timestamp_jst.dt.year - 1)
    hourly["month"] = hourly.timestamp_jst.dt.month
    hourly["hour"] = hourly.timestamp_jst.dt.hour
    keys = ["prefecture", "mesh_id", "fiscal_year"]
    floor = hourly.groupby(keys + ["month", "hour"])["grid_import_mw"].transform(lambda x: x.quantile(0.10))
    hourly["task_c1_mw"] = np.maximum(hourly["grid_import_mw"] - floor, 0.0)
    hourly["calendar_day"] = hourly.timestamp_jst.dt.floor("D")
    daily = hourly.groupby(keys + ["calendar_day"], as_index=False).agg(daily_task_mwh=("task_c1_mw", "sum"))
    daily["season_segment"] = ((daily.calendar_day.dt.month % 12) // 3).astype(int)
    daily["season_q75_mwh"] = daily.groupby(keys + ["season_segment"])["daily_task_mwh"].transform(lambda x: x.quantile(0.75))
    daily["event_flag_q75"] = daily.daily_task_mwh >= daily.season_q75_mwh
    hourly = hourly.merge(daily[keys + ["calendar_day", "season_segment", "event_flag_q75"]], on=keys + ["calendar_day"], how="left", validate="many_to_one")
    hourly["task_c3_q75_mw"] = np.where(hourly.event_flag_q75, hourly.task_c1_mw, 0.0)
    hourly["r2_potential_recharge_mw"] = hourly["reverse_flow_mw"]
    hourly = hourly.sort_values(keys + ["timestamp_jst"], kind="stable")
    hourly["source_hour_id"] = hourly.groupby(keys).cumcount()
    hourly["source_day_id"] = hourly.source_hour_id // 24
    hourly["raw_grid_import_mean_mw"] = hourly.groupby(keys)["grid_import_mw"].transform("mean")
    hourly["forced_discharge_power_mw"] = 0.01 * hourly.raw_grid_import_mean_mw
    counts = hourly.groupby(keys).size()
    valid = counts[counts.eq(8760)].index
    valid_set = set(valid.tolist())
    hourly = hourly[[tuple(row) in valid_set for row in hourly[keys].itertuples(index=False, name=None)]].copy()
    return hourly


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE) as outer:
        members = member_map(outer)
        selections = {
            code: choose_meshes(outer, members[(code, "202304")])
            for code in PREFECTURES
        }
        rows: list[dict[str, object]] = []
        for code, name in PREFECTURES.items():
            selected = set(selections[code])
            for yearmonth in sorted(value for pref, value in members if pref == code):
                rows.extend(extract_member(outer, members[(code, yearmonth)], selected, name))
                print(f"extracted {name} {yearmonth}", flush=True)
    halfhour = pd.DataFrame(rows)
    halfhour["timestamp_jst"] = pd.to_datetime(halfhour["timestamp_jst"])
    numeric_cols = [column for column in halfhour if column.endswith("_halfhour")]
    halfhour = halfhour.dropna(subset=numeric_cols)
    halfhour["hour_start_jst"] = halfhour.timestamp_jst.dt.floor("h")
    hourly = halfhour.groupby(["prefecture", "mesh_id", "hour_start_jst"], as_index=False).agg(
        grid_import_kwh=("grid_import_kwh_halfhour", "sum"),
        reverse_flow_kwh=("reverse_flow_kwh_halfhour", "sum"),
        self_consumption_kwh=("self_consumption_kwh_halfhour", "sum"),
        contracts=("contracts", "median"),
        pv_capacity_kw=("pv_capacity_kw", "median"),
    )
    hourly = hourly.rename(columns={"hour_start_jst": "timestamp_jst"})
    hourly["grid_import_mw"] = hourly.grid_import_kwh / 1000.0
    hourly["reverse_flow_mw"] = hourly.reverse_flow_kwh / 1000.0
    model = build_model_panel(hourly)
    hourly.to_csv(OUT / "japan_selected_mesh_hourly_observed.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    model.to_csv(OUT / "japan_community_event_task_panel.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    selection_rows = [
        {"prefecture_code": code, "prefecture": PREFECTURES[code], "mesh_id": mesh}
        for code, meshes in selections.items() for mesh in meshes
    ]
    pd.DataFrame(selection_rows).to_csv(OUT / "selected_meshes.csv", index=False)
    qa = {
        "archive": str(ARCHIVE), "archive_sha256": sha256(ARCHIVE),
        "prefectures": PREFECTURES, "meshes_per_prefecture": N_MESHES,
        "selected_meshes": sum(map(len, selections.values())),
        "model_rows": len(model),
        "community_years": int(model[["prefecture", "mesh_id", "fiscal_year"]].drop_duplicates().shape[0]),
        "all_community_years_8760_hours": bool(model.groupby(["prefecture", "mesh_id", "fiscal_year"]).size().eq(8760).all()),
        "task_nonnegative": bool((model.task_c1_mw >= 0).all()),
        "recharge_nonnegative": bool((model.r2_potential_recharge_mw >= 0).all()),
    }
    (OUT / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
