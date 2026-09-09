#!/usr/bin/env python3
"""Event-level LFP-only reliability test for paper 2.

This module fixes the 4h LFP capacity at the first-paper economic boundary
(main: OSR; robustness: CSR and PSR), constructs real q75 consecutive peaking
events, and evaluates whether the fixed LFP fleet can cover multi-day subdaily
peaking windows using real hourly curtailment for recharge.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_CODE_DIR = Path(__file__).resolve().parents[1]
OUTER_PROJECT_DIR = PAPER_CODE_DIR.parents[1]
M3_OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
M4_OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
RAW_DATA_DIR = OUTER_PROJECT_DIR / "dataset"

if str(PAPER_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_CODE_DIR))

from config import CONFIG  # noqa: E402
from ldes_params import calculate_crf, extract_tech_params, load_tech_library  # noqa: E402

MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_DIR / "output"
INTERMEDIATE_DIR = MODULE_DIR / "intermediate"
LOG_DIR = MODULE_DIR / "logs"
FIGURE_DIR = MODULE_DIR / "figures"

DEFAULT_PROVINCES = ["Beijing", "Jiangsu", "Fujian", "Zhejiang", "Sichuan"]
BOUNDARY_METRICS = ["osr", "csr", "psr"]
MAIN_BOUNDARY_METRIC = "osr"
PEAKING_WINDOWS_H = [8, 12, 16, 24]
LDES_DURATIONS_H = {
    "caes": [8, 10, 12, 16, 24],
    "laes": [8, 10, 12, 16, 24],
    "gravity": [8, 10, 12],
    "fecr_flow": [8, 10, 12],
    "molten_salt_tes": [8, 10, 12, 16],
    "vrfb": [4, 6, 8],
    "h2_cavern": [24, 48, 72, 168],
}
LDES_TECHS = list(LDES_DURATIONS_H.keys())

LFP_DURATION_H = 4.0
SOC_MIN = 0.05
SOC_MAX = 0.95
CHARGE_EFF = 0.95
DISCHARGE_EFF = 0.95
COMPLETION_TOL_KWH = 1e-6
HOUR_COL_RE = re.compile(r"^\d{10}$")
DISCOUNT_RATE = float(CONFIG["cost"]["discount_rate"])


@dataclass(frozen=True)
class DispatchResult:
    served_kwh: float
    unserved_kwh: float
    charged_from_curtailment_kwh: float
    min_soc_pct: float
    final_soc_pct: float
    soc_safety_floor_hours: int
    soc_safety_floor_days: int
    completed: bool


@dataclass(frozen=True)
class HybridDispatchResult:
    lfp_served_kwh: float
    ldes_served_kwh: float
    unserved_kwh: float
    lfp_charged_from_curtailment_kwh: float
    ldes_charged_from_curtailment_kwh: float
    lfp_min_soc_pct: float
    ldes_min_soc_pct: float
    lfp_final_soc_pct: float
    ldes_final_soc_pct: float
    lfp_safety_floor_hours: int
    ldes_safety_floor_hours: int
    lfp_safety_floor_days: int
    ldes_safety_floor_days: int
    max_residual_after_lfp_kwh: float
    completed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 4h LFP-only subdaily reliability tests for real q75 events."
    )
    parser.add_argument(
        "--event-quantile",
        type=float,
        default=0.75,
        help="Quantile used to define real daily extreme peaking demand events.",
    )
    parser.add_argument(
        "--min-event-days",
        type=int,
        default=3,
        help="Minimum consecutive event length for the real-event test set.",
    )
    parser.add_argument(
        "--peaking-window-start",
        type=int,
        default=8,
        help=(
            "Start hour for the synthetic subdaily peaking window. "
            "Default 8 gives 8/12/16h windows without crossing midnight."
        ),
    )
    parser.add_argument(
        "--provinces",
        nargs="+",
        default=DEFAULT_PROVINCES,
        help="Province names to include.",
    )
    parser.add_argument(
        "--force-hourly-cache",
        action="store_true",
        help="Rebuild the hourly provincial curtailment cache from raw files.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, INTERMEDIATE_DIR, LOG_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_province_name(name: str | float | None) -> str | None:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return None
    name = str(name).strip()
    mapping = {
        "Hebei north": "Hebei",
        "Hebei south": "Hebei",
        "Inner Mongolia east": "Inner Mongolia",
        "Inner Mongolia west": "Inner Mongolia",
    }
    return mapping.get(name, name)


def interpolate_capacity_for_satisfy_rate(curve: pd.DataFrame, target_pct: float) -> float:
    curve = curve.sort_values(["satisfy_rate_pct", "capacity_mwh"])
    curve = curve.drop_duplicates("satisfy_rate_pct", keep="first")
    x = curve["satisfy_rate_pct"].to_numpy(dtype=float)
    y = curve["capacity_mwh"].to_numpy(dtype=float)
    if len(x) == 0:
        raise ValueError("Empty LFP curve while interpolating boundary capacity.")
    if target_pct <= x[0]:
        return float(y[0])
    if target_pct >= x[-1]:
        return float(y[-1])
    return float(np.interp(target_pct, x, y))


def interpolate_annual_cost(curve: pd.DataFrame, capacity_mwh: float) -> float:
    curve = curve.sort_values("capacity_mwh").drop_duplicates("capacity_mwh", keep="first")
    x = curve["capacity_mwh"].to_numpy(dtype=float)
    y = curve["annual_cost_cny"].to_numpy(dtype=float)
    if len(x) == 0:
        raise ValueError("Empty LFP curve while interpolating annual cost.")
    if capacity_mwh <= x[0]:
        return float(y[0] * capacity_mwh / x[0])
    if capacity_mwh >= x[-1]:
        slope = y[-1] / x[-1]
        return float(slope * capacity_mwh)
    return float(np.interp(capacity_mwh, x, y))


def load_baseline_boundaries(provinces: list[str]) -> pd.DataFrame:
    summary_path = M4_OUTPUT_DIR / "m4_3r_summary_all_scenarios.csv"
    curve_path = M4_OUTPUT_DIR / "m4_curve_points_baseline.csv"

    summary = pd.read_csv(summary_path)
    summary = summary[
        (summary["tech_id"] == "li_lfp")
        & (summary["cost_scenario"] == "baseline")
        & (summary["duration_h"] == LFP_DURATION_H)
        & (summary["province_en"].isin(provinces))
    ].copy()

    curve_cols = [
        "tech_id",
        "dispatch_scenario",
        "duration_h",
        "cost_scenario",
        "province_en",
        "province_cn",
        "capacity_mwh",
        "satisfy_rate_pct",
        "annual_cost_cny",
        "unit_cost_cny_kwh",
        "gas_unit_cost_cny_kwh",
    ]
    curves = pd.read_csv(curve_path, usecols=lambda c: c in curve_cols)
    curves = curves[
        (curves["tech_id"] == "li_lfp")
        & (curves["duration_h"] == LFP_DURATION_H)
        & (curves["cost_scenario"] == "baseline")
        & (curves["province_en"].isin(provinces))
    ].copy()
    if "dispatch_scenario" in curves.columns:
        curves = curves[curves["dispatch_scenario"] == "baseline"].copy()

    rows: list[dict[str, float | str]] = []
    for _, srow in summary.sort_values("province_en").iterrows():
        province = srow["province_en"]
        curve = curves[curves["province_en"] == province]
        if curve.empty:
            raise ValueError(f"No baseline LFP curve for {province}.")
        province_cn = srow.get("province_cn", curve["province_cn"].iloc[0])
        for metric in BOUNDARY_METRICS:
            target_pct = float(srow[f"{metric}_pct"])
            capacity_mwh = interpolate_capacity_for_satisfy_rate(curve, target_pct)
            annual_cost_cny = interpolate_annual_cost(curve, capacity_mwh)
            rows.append(
                {
                    "province_en": province,
                    "province_cn": province_cn,
                    "tech_id": "li_lfp",
                    "duration_h": LFP_DURATION_H,
                    "boundary_metric": metric,
                    "boundary_role": "main" if metric == MAIN_BOUNDARY_METRIC else "robustness",
                    "target_satisfy_rate_pct": target_pct,
                    "baseline_LFP_boundary_mwh": capacity_mwh,
                    "baseline_LFP_boundary_annual_cost_cny": annual_cost_cny,
                    "annual_cost_per_mwh_year_cny": annual_cost_cny / capacity_mwh,
                    "max_satisfy_rate_pct": float(srow["max_satisfy_rate_pct"]),
                    "source_boundary_table": str(summary_path.relative_to(PAPER_CODE_DIR)),
                    "source_curve_table": str(curve_path.relative_to(PAPER_CODE_DIR)),
                    "boundary_interpretation": (
                        "First-paper decarbonization-asymptote LFP economic boundary; "
                        "OSR is the main boundary and CSR/PSR are robustness checks."
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "baseline_LFP_boundary.csv", index=False, encoding="utf-8-sig")
    return out


def contiguous_runs(mask: list[bool]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask + [False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            end = idx - 1
            runs.append((start, end, end - start + 1))
            start = None
    return runs


def build_real_events(
    provinces: list[str], event_quantile: float, min_event_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demand_path = M3_OUTPUT_DIR / "demand_yr0_real_only.csv"
    demand = pd.read_csv(demand_path)
    if "day_idx" not in demand.columns:
        demand.insert(0, "day_idx", np.arange(len(demand), dtype=int))

    event_rows: list[dict[str, float | int | str]] = []
    run_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for province in provinces:
        series = demand[province].astype(float).reset_index(drop=True)
        positive = series[series > 0]
        threshold = float(positive.quantile(event_quantile)) if len(positive) else 0.0
        runs = contiguous_runs((series > threshold).tolist())
        qualifying_idx = 0
        for run_idx, (start_pos, end_pos, length_days) in enumerate(runs, start=1):
            row = {
                "province_en": province,
                "run_id": f"{province}_q{int(event_quantile * 100)}_{run_idx:03d}",
                "threshold_quantile": event_quantile,
                "threshold_kwh": threshold,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "start_day_idx": int(demand.loc[start_pos, "day_idx"]),
                "end_day_idx": int(demand.loc[end_pos, "day_idx"]),
                "length_days": length_days,
                "total_event_demand_kwh": float(series.iloc[start_pos : end_pos + 1].sum()),
                "max_daily_demand_kwh": float(series.iloc[start_pos : end_pos + 1].max()),
                "mean_daily_demand_kwh": float(series.iloc[start_pos : end_pos + 1].mean()),
                "source_demand_table": str(demand_path.relative_to(PAPER_CODE_DIR)),
            }
            run_rows.append(row)
            if length_days >= min_event_days:
                qualifying_idx += 1
                event_row = dict(row)
                event_row["event_id"] = (
                    f"{province}_q{int(event_quantile * 100)}_ge{min_event_days}_"
                    f"{qualifying_idx:03d}"
                )
                event_rows.append(event_row)

        lengths = [length for _, _, length in runs]
        summary_rows.append(
            {
                "province_en": province,
                "threshold_quantile": event_quantile,
                "threshold_kwh": threshold,
                "run_count_ge1": len(lengths),
                "run_count_ge3": sum(length >= 3 for length in lengths),
                "run_count_ge5": sum(length >= 5 for length in lengths),
                "run_count_ge7": sum(length >= 7 for length in lengths),
                "run_count_ge10": sum(length >= 10 for length in lengths),
                "run_count_ge14": sum(length >= 14 for length in lengths),
                "max_run_length_days": max(lengths) if lengths else 0,
                "note": (
                    "Real q75 events are used for the >=3-day minimal loop; "
                    "5/7/10/14-day stress tests should be simulated when real runs are sparse."
                ),
            }
        )

    events = pd.DataFrame(event_rows)
    runs_df = pd.DataFrame(run_rows)
    event_summary = pd.DataFrame(summary_rows)
    events.to_csv(OUTPUT_DIR / "real_q75_events_ge3.csv", index=False, encoding="utf-8-sig")
    runs_df.to_csv(OUTPUT_DIR / "real_q75_all_runs.csv", index=False, encoding="utf-8-sig")
    event_summary.to_csv(
        OUTPUT_DIR / "real_q75_event_length_summary.csv", index=False, encoding="utf-8-sig"
    )
    return demand, events, event_summary


def load_objectid_province_map() -> dict[str, str]:
    geojson_path = RAW_DATA_DIR / "省份修正合并.geojson"
    with geojson_path.open("r", encoding="utf-8") as f:
        geo = json.load(f)
    mapping: dict[str, str] = {}
    for feature in geo.get("features", []):
        props = feature.get("properties", {})
        object_id = props.get("ObjectId")
        province = normalize_province_name(props.get("State_Province"))
        if object_id and province:
            mapping[str(object_id)] = province
    return mapping


def build_hourly_curtailment_cache(
    provinces: list[str], force_hourly_cache: bool = False
) -> pd.DataFrame:
    cache_path = INTERMEDIATE_DIR / "hourly_provincial_curtailment_focus.csv"
    if cache_path.exists() and not force_hourly_cache:
        hourly = pd.read_csv(cache_path)
        return hourly

    objectid_to_province = load_objectid_province_map()
    hourly_dir = RAW_DATA_DIR / "30日弃电"
    raw_files = sorted(hourly_dir.glob("TotalCurt_Hourly_*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No hourly curtailment files found in {hourly_dir}")

    timestamps = pd.date_range("2020-01-01 00:00", "2020-12-31 23:00", freq="h")
    timestamps = timestamps[~((timestamps.month == 2) & (timestamps.day == 29))]
    hour_keys = [ts.strftime("%Y%m%d%H") for ts in timestamps]
    key_to_idx = {key: idx for idx, key in enumerate(hour_keys)}
    values = {province: np.zeros(len(timestamps), dtype=float) for province in provinces}

    for raw_path in raw_files:
        month_df = pd.read_csv(raw_path, dtype={"ObjectId": str}, low_memory=False)
        if "ObjectId" not in month_df.columns:
            continue
        hour_cols = [col for col in month_df.columns if HOUR_COL_RE.match(str(col))]
        if not hour_cols:
            continue
        province_series = month_df["ObjectId"].astype(str).map(objectid_to_province)
        month_df = month_df.loc[province_series.isin(provinces), ["ObjectId", *hour_cols]].copy()
        if month_df.empty:
            continue
        month_df["_province"] = month_df["ObjectId"].astype(str).map(objectid_to_province)
        grouped = month_df.groupby("_province")[hour_cols].sum(numeric_only=True)
        for col in hour_cols:
            hour_idx = key_to_idx.get(str(col))
            if hour_idx is None:
                continue
            for province in provinces:
                if province in grouped.index:
                    values[province][hour_idx] += float(grouped.at[province, col])

    hourly = pd.DataFrame(
        {
            "hour_idx": np.arange(len(timestamps), dtype=int),
            "day_pos": np.arange(len(timestamps), dtype=int) // 24,
            "hour": [int(ts.hour) for ts in timestamps],
            "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        }
    )
    for province in provinces:
        hourly[province] = values[province]
    hourly.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return hourly


def peaking_hours(peaking_window_h: int, start_hour: int) -> set[int]:
    if peaking_window_h >= 24:
        return set(range(24))
    return {(start_hour + offset) % 24 for offset in range(peaking_window_h)}


def dispatch_lfp_event(
    demand_values_kwh: np.ndarray,
    event_start_pos: int,
    province: str,
    hourly_curtailment: pd.DataFrame,
    capacity_mwh: float,
    peaking_window_h: int,
    peaking_window_start: int,
) -> DispatchResult:
    capacity_kwh = capacity_mwh * 1000.0
    power_kw = capacity_kwh / LFP_DURATION_H
    e_min_kwh = SOC_MIN * capacity_kwh
    e_max_kwh = SOC_MAX * capacity_kwh
    soc_kwh = e_max_kwh
    min_soc_kwh = soc_kwh

    peak_hours = peaking_hours(peaking_window_h, peaking_window_start)
    served_kwh = 0.0
    unserved_kwh = 0.0
    charged_from_curtailment_kwh = 0.0
    floor_hours = 0
    floor_days: set[int] = set()

    hourly_idx = hourly_curtailment.set_index("hour_idx")

    for day_offset, daily_demand_kwh in enumerate(demand_values_kwh):
        day_pos = event_start_pos + day_offset
        hourly_peaking_demand_kwh = float(daily_demand_kwh) / float(peaking_window_h)
        for hour in range(24):
            hour_idx = day_pos * 24 + hour
            if hour in peak_hours:
                request_kwh = hourly_peaking_demand_kwh
                available_delivery_kwh = max(0.0, (soc_kwh - e_min_kwh) * DISCHARGE_EFF)
                deliver_kwh = min(request_kwh, available_delivery_kwh, power_kw)
                if deliver_kwh > 0:
                    soc_kwh -= deliver_kwh / DISCHARGE_EFF
                served_kwh += deliver_kwh
                unserved_kwh += request_kwh - deliver_kwh
                if soc_kwh <= e_min_kwh + COMPLETION_TOL_KWH:
                    floor_hours += 1
                    floor_days.add(day_pos)
            else:
                curtailment_kwh = 0.0
                if hour_idx in hourly_idx.index:
                    curtailment_kwh = float(hourly_idx.at[hour_idx, province])
                charge_input_kwh = min(
                    max(0.0, curtailment_kwh),
                    power_kw,
                    max(0.0, (e_max_kwh - soc_kwh) / CHARGE_EFF),
                )
                if charge_input_kwh > 0:
                    soc_kwh += charge_input_kwh * CHARGE_EFF
                    charged_from_curtailment_kwh += charge_input_kwh
            min_soc_kwh = min(min_soc_kwh, soc_kwh)

    min_soc_pct = 100.0 * min_soc_kwh / capacity_kwh if capacity_kwh > 0 else np.nan
    final_soc_pct = 100.0 * soc_kwh / capacity_kwh if capacity_kwh > 0 else np.nan
    return DispatchResult(
        served_kwh=served_kwh,
        unserved_kwh=unserved_kwh,
        charged_from_curtailment_kwh=charged_from_curtailment_kwh,
        min_soc_pct=min_soc_pct,
        final_soc_pct=final_soc_pct,
        soc_safety_floor_hours=floor_hours,
        soc_safety_floor_days=len(floor_days),
        completed=unserved_kwh <= COMPLETION_TOL_KWH,
    )


def required_capacity_for_completion(
    demand_values_kwh: np.ndarray,
    event_start_pos: int,
    province: str,
    hourly_curtailment: pd.DataFrame,
    baseline_capacity_mwh: float,
    peaking_window_h: int,
    peaking_window_start: int,
) -> tuple[float, DispatchResult]:
    base_result = dispatch_lfp_event(
        demand_values_kwh,
        event_start_pos,
        province,
        hourly_curtailment,
        baseline_capacity_mwh,
        peaking_window_h,
        peaking_window_start,
    )
    if base_result.completed:
        return baseline_capacity_mwh, base_result

    total_event_mwh = float(np.sum(demand_values_kwh)) / 1000.0
    max_hourly_demand_mwh = float(np.max(demand_values_kwh)) / float(peaking_window_h) / 1000.0
    energy_bound_mwh = total_event_mwh / ((SOC_MAX - SOC_MIN) * DISCHARGE_EFF)
    power_bound_mwh = max_hourly_demand_mwh * LFP_DURATION_H
    high_mwh = max(baseline_capacity_mwh * 1.05 + 1.0, energy_bound_mwh, power_bound_mwh) * 1.02

    high_result = dispatch_lfp_event(
        demand_values_kwh,
        event_start_pos,
        province,
        hourly_curtailment,
        high_mwh,
        peaking_window_h,
        peaking_window_start,
    )
    grow_count = 0
    while not high_result.completed and grow_count < 20:
        high_mwh *= 1.5
        high_result = dispatch_lfp_event(
            demand_values_kwh,
            event_start_pos,
            province,
            hourly_curtailment,
            high_mwh,
            peaking_window_h,
            peaking_window_start,
        )
        grow_count += 1
    if not high_result.completed:
        return np.nan, high_result

    low_mwh = baseline_capacity_mwh
    best_result = high_result
    for _ in range(32):
        mid_mwh = (low_mwh + high_mwh) / 2.0
        mid_result = dispatch_lfp_event(
            demand_values_kwh,
            event_start_pos,
            province,
            hourly_curtailment,
            mid_mwh,
            peaking_window_h,
            peaking_window_start,
        )
        if mid_result.completed:
            high_mwh = mid_mwh
            best_result = mid_result
        else:
            low_mwh = mid_mwh
    return high_mwh, best_result


def load_ldes_tech_params() -> list[dict[str, object]]:
    tech_lib = load_tech_library()
    params: list[dict[str, object]] = []
    for tech_id in LDES_TECHS:
        if tech_id not in tech_lib.index:
            continue
        row = tech_lib.loc[tech_id]
        for duration_h in LDES_DURATIONS_H[tech_id]:
            tp = extract_tech_params(row, tech_id, "baseline", duration_h)
            params.append(
                {
                    "tech_id": tech_id,
                    "technology_en": tp.technology_en,
                    "technology_cn": tp.technology_cn,
                    "duration_h": float(duration_h),
                    "params": tp,
                }
            )
    out = pd.DataFrame(
        [
            {
                "ldes_tech_id": item["tech_id"],
                "technology_en": item["technology_en"],
                "technology_cn": item["technology_cn"],
                "ldes_duration_h": item["duration_h"],
                "cost_scenario": "baseline",
                "rte": item["params"].rte,
                "charge_eff": item["params"].charge_eff,
                "discharge_eff": item["params"].discharge_eff,
                "charge_power_ratio": item["params"].charge_power_ratio,
                "self_discharge_rate_day": item["params"].self_discharge_rate_day,
                "calendar_life_years": item["params"].calendar_life_years,
                "power_cost_cny_kw": item["params"].power_cost_cny_kw,
                "energy_cost_cny_kwh": item["params"].energy_cost_cny_kwh,
                "fixed_om_cny_kw_yr": item["params"].fixed_om_cny_kw_yr,
            }
            for item in params
        ]
    )
    out.to_csv(OUTPUT_DIR / "ldes_baseline_tech_params.csv", index=False, encoding="utf-8-sig")
    return params


def annualized_storage_cost(tp, capacity_mwh: float) -> tuple[float, float]:
    if capacity_mwh <= 0:
        return 0.0, 0.0
    power_mw = capacity_mwh / float(tp.duration_h)
    discharge_power_kw = power_mw * 1000.0
    rated_power_kw = discharge_power_kw * max(float(tp.charge_power_ratio), 1.0)
    capacity_kwh = capacity_mwh * 1000.0
    capex_cny = (
        float(tp.power_cost_cny_kw) * rated_power_kw
        + float(tp.energy_cost_cny_kwh) * capacity_kwh
    )
    annual_capex_cny = capex_cny * calculate_crf(DISCOUNT_RATE, int(tp.calendar_life_years))
    annual_fixed_om_cny = float(tp.fixed_om_cny_kw_yr) * rated_power_kw
    return capex_cny, annual_capex_cny + annual_fixed_om_cny


def dispatch_lfp_ldes_event(
    demand_values_kwh: np.ndarray,
    event_start_pos: int,
    province: str,
    curtailment_values_kwh: np.ndarray,
    lfp_capacity_mwh: float,
    ldes_capacity_mwh: float,
    ldes_tp,
    peaking_window_h: int,
    peaking_window_start: int,
) -> HybridDispatchResult:
    lfp_capacity_kwh = lfp_capacity_mwh * 1000.0
    lfp_power_kw = lfp_capacity_kwh / LFP_DURATION_H if lfp_capacity_kwh > 0 else 0.0
    lfp_e_min_kwh = SOC_MIN * lfp_capacity_kwh
    lfp_e_max_kwh = SOC_MAX * lfp_capacity_kwh
    lfp_soc_kwh = lfp_e_max_kwh
    lfp_min_soc_kwh = lfp_soc_kwh

    ldes_capacity_kwh = ldes_capacity_mwh * 1000.0
    ldes_duration_h = float(ldes_tp.duration_h)
    ldes_discharge_power_kw = (
        ldes_capacity_kwh / ldes_duration_h if ldes_capacity_kwh > 0 else 0.0
    )
    ldes_charge_power_kw = ldes_discharge_power_kw * float(ldes_tp.charge_power_ratio)
    ldes_e_min_kwh = SOC_MIN * ldes_capacity_kwh
    ldes_e_max_kwh = SOC_MAX * ldes_capacity_kwh
    ldes_soc_kwh = ldes_e_max_kwh
    ldes_min_soc_kwh = ldes_soc_kwh
    ldes_hourly_self_discharge = max(
        0.0, 1.0 - float(ldes_tp.self_discharge_rate_day)
    ) ** (1.0 / 24.0)

    peak_hours = peaking_hours(peaking_window_h, peaking_window_start)
    lfp_served_kwh = 0.0
    ldes_served_kwh = 0.0
    unserved_kwh = 0.0
    lfp_charged_kwh = 0.0
    ldes_charged_kwh = 0.0
    lfp_floor_hours = 0
    ldes_floor_hours = 0
    lfp_floor_days: set[int] = set()
    ldes_floor_days: set[int] = set()
    max_residual_after_lfp_kwh = 0.0

    for day_offset, daily_demand_kwh in enumerate(demand_values_kwh):
        day_pos = event_start_pos + day_offset
        hourly_peaking_demand_kwh = float(daily_demand_kwh) / float(peaking_window_h)
        for hour in range(24):
            hour_idx = day_pos * 24 + hour
            if ldes_capacity_kwh > 0:
                ldes_soc_kwh = max(ldes_e_min_kwh, ldes_soc_kwh * ldes_hourly_self_discharge)

            if hour in peak_hours:
                request_kwh = hourly_peaking_demand_kwh

                lfp_available_kwh = max(0.0, (lfp_soc_kwh - lfp_e_min_kwh) * DISCHARGE_EFF)
                lfp_deliver_kwh = min(request_kwh, lfp_available_kwh, lfp_power_kw)
                if lfp_deliver_kwh > 0:
                    lfp_soc_kwh -= lfp_deliver_kwh / DISCHARGE_EFF
                residual_after_lfp_kwh = request_kwh - lfp_deliver_kwh
                max_residual_after_lfp_kwh = max(
                    max_residual_after_lfp_kwh, residual_after_lfp_kwh
                )

                ldes_available_kwh = max(
                    0.0, (ldes_soc_kwh - ldes_e_min_kwh) * float(ldes_tp.discharge_eff)
                )
                ldes_deliver_kwh = min(
                    residual_after_lfp_kwh, ldes_available_kwh, ldes_discharge_power_kw
                )
                if ldes_deliver_kwh > 0:
                    ldes_soc_kwh -= ldes_deliver_kwh / float(ldes_tp.discharge_eff)

                lfp_served_kwh += lfp_deliver_kwh
                ldes_served_kwh += ldes_deliver_kwh
                unserved_kwh += residual_after_lfp_kwh - ldes_deliver_kwh

                if lfp_soc_kwh <= lfp_e_min_kwh + COMPLETION_TOL_KWH:
                    lfp_floor_hours += 1
                    lfp_floor_days.add(day_pos)
                if ldes_capacity_kwh > 0 and ldes_soc_kwh <= ldes_e_min_kwh + COMPLETION_TOL_KWH:
                    ldes_floor_hours += 1
                    ldes_floor_days.add(day_pos)
            else:
                curtailment_kwh = (
                    float(curtailment_values_kwh[hour_idx])
                    if 0 <= hour_idx < len(curtailment_values_kwh)
                    else 0.0
                )
                remaining_curtailment_kwh = max(0.0, curtailment_kwh)

                lfp_charge_input_kwh = min(
                    remaining_curtailment_kwh,
                    lfp_power_kw,
                    max(0.0, (lfp_e_max_kwh - lfp_soc_kwh) / CHARGE_EFF),
                )
                if lfp_charge_input_kwh > 0:
                    lfp_soc_kwh += lfp_charge_input_kwh * CHARGE_EFF
                    lfp_charged_kwh += lfp_charge_input_kwh
                    remaining_curtailment_kwh -= lfp_charge_input_kwh

                ldes_charge_input_kwh = min(
                    remaining_curtailment_kwh,
                    ldes_charge_power_kw,
                    max(
                        0.0,
                        (ldes_e_max_kwh - ldes_soc_kwh) / float(ldes_tp.charge_eff)
                        if ldes_capacity_kwh > 0
                        else 0.0,
                    ),
                )
                if ldes_charge_input_kwh > 0:
                    ldes_soc_kwh += ldes_charge_input_kwh * float(ldes_tp.charge_eff)
                    ldes_charged_kwh += ldes_charge_input_kwh

            lfp_min_soc_kwh = min(lfp_min_soc_kwh, lfp_soc_kwh)
            ldes_min_soc_kwh = min(ldes_min_soc_kwh, ldes_soc_kwh)

    lfp_min_soc_pct = 100.0 * lfp_min_soc_kwh / lfp_capacity_kwh if lfp_capacity_kwh > 0 else np.nan
    ldes_min_soc_pct = (
        100.0 * ldes_min_soc_kwh / ldes_capacity_kwh if ldes_capacity_kwh > 0 else np.nan
    )
    lfp_final_soc_pct = 100.0 * lfp_soc_kwh / lfp_capacity_kwh if lfp_capacity_kwh > 0 else np.nan
    ldes_final_soc_pct = (
        100.0 * ldes_soc_kwh / ldes_capacity_kwh if ldes_capacity_kwh > 0 else np.nan
    )

    return HybridDispatchResult(
        lfp_served_kwh=lfp_served_kwh,
        ldes_served_kwh=ldes_served_kwh,
        unserved_kwh=unserved_kwh,
        lfp_charged_from_curtailment_kwh=lfp_charged_kwh,
        ldes_charged_from_curtailment_kwh=ldes_charged_kwh,
        lfp_min_soc_pct=lfp_min_soc_pct,
        ldes_min_soc_pct=ldes_min_soc_pct,
        lfp_final_soc_pct=lfp_final_soc_pct,
        ldes_final_soc_pct=ldes_final_soc_pct,
        lfp_safety_floor_hours=lfp_floor_hours,
        ldes_safety_floor_hours=ldes_floor_hours,
        lfp_safety_floor_days=len(lfp_floor_days),
        ldes_safety_floor_days=len(ldes_floor_days),
        max_residual_after_lfp_kwh=max_residual_after_lfp_kwh,
        completed=unserved_kwh <= COMPLETION_TOL_KWH,
    )


def required_ldes_capacity_for_completion(
    demand_values_kwh: np.ndarray,
    event_start_pos: int,
    province: str,
    curtailment_values_kwh: np.ndarray,
    lfp_capacity_mwh: float,
    ldes_tp,
    peaking_window_h: int,
    peaking_window_start: int,
) -> tuple[float, HybridDispatchResult]:
    zero_result = dispatch_lfp_ldes_event(
        demand_values_kwh,
        event_start_pos,
        province,
        curtailment_values_kwh,
        lfp_capacity_mwh,
        0.0,
        ldes_tp,
        peaking_window_h,
        peaking_window_start,
    )
    if zero_result.completed:
        return 0.0, zero_result

    residual_mwh = zero_result.unserved_kwh / 1000.0
    max_residual_hourly_mwh = zero_result.max_residual_after_lfp_kwh / 1000.0
    usable_fraction = (SOC_MAX - SOC_MIN) * float(ldes_tp.discharge_eff)
    energy_bound_mwh = residual_mwh / usable_fraction if usable_fraction > 0 else residual_mwh
    power_bound_mwh = max_residual_hourly_mwh * float(ldes_tp.duration_h)
    high_mwh = max(1.0, energy_bound_mwh, power_bound_mwh) * 1.02

    high_result = dispatch_lfp_ldes_event(
        demand_values_kwh,
        event_start_pos,
        province,
        curtailment_values_kwh,
        lfp_capacity_mwh,
        high_mwh,
        ldes_tp,
        peaking_window_h,
        peaking_window_start,
    )
    grow_count = 0
    while not high_result.completed and grow_count < 24:
        high_mwh *= 1.5
        high_result = dispatch_lfp_ldes_event(
            demand_values_kwh,
            event_start_pos,
            province,
            curtailment_values_kwh,
            lfp_capacity_mwh,
            high_mwh,
            ldes_tp,
            peaking_window_h,
            peaking_window_start,
        )
        grow_count += 1
    if not high_result.completed:
        return np.nan, high_result

    low_mwh = 0.0
    best_result = high_result
    for _ in range(32):
        mid_mwh = (low_mwh + high_mwh) / 2.0
        mid_result = dispatch_lfp_ldes_event(
            demand_values_kwh,
            event_start_pos,
            province,
            curtailment_values_kwh,
            lfp_capacity_mwh,
            mid_mwh,
            ldes_tp,
            peaking_window_h,
            peaking_window_start,
        )
        if mid_result.completed:
            high_mwh = mid_mwh
            best_result = mid_result
        else:
            low_mwh = mid_mwh
    return high_mwh, best_result


def run_event_reliability(
    demand: pd.DataFrame,
    events: pd.DataFrame,
    boundaries: pd.DataFrame,
    hourly_curtailment: pd.DataFrame,
    peaking_window_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        empty = pd.DataFrame()
        empty.to_csv(OUTPUT_DIR / "lfp_only_event_reliability.csv", index=False)
        empty.to_csv(OUTPUT_DIR / "lfp_only_event_reliability_summary.csv", index=False)
        return empty, empty

    result_rows: list[dict[str, float | int | str]] = []
    for _, event in events.iterrows():
        province = str(event["province_en"])
        start_pos = int(event["start_pos"])
        end_pos = int(event["end_pos"])
        demand_values = demand.loc[start_pos:end_pos, province].astype(float).to_numpy()
        event_total_kwh = float(np.sum(demand_values))
        province_boundaries = boundaries[boundaries["province_en"] == province]

        for _, boundary in province_boundaries.iterrows():
            baseline_capacity_mwh = float(boundary["baseline_LFP_boundary_mwh"])
            annual_cost_per_mwh = float(boundary["annual_cost_per_mwh_year_cny"])
            baseline_annual_cost = float(boundary["baseline_LFP_boundary_annual_cost_cny"])

            for window_h in PEAKING_WINDOWS_H:
                baseline_result = dispatch_lfp_event(
                    demand_values,
                    start_pos,
                    province,
                    hourly_curtailment,
                    baseline_capacity_mwh,
                    window_h,
                    peaking_window_start,
                )
                required_mwh, required_result = required_capacity_for_completion(
                    demand_values,
                    start_pos,
                    province,
                    hourly_curtailment,
                    baseline_capacity_mwh,
                    window_h,
                    peaking_window_start,
                )
                required_overbuild_mwh = required_mwh - baseline_capacity_mwh
                required_overbuild_annual_cost = required_overbuild_mwh * annual_cost_per_mwh
                required_annual_cost = required_mwh * annual_cost_per_mwh
                baseline_unit_cost = baseline_annual_cost / max(baseline_result.served_kwh, 1.0)
                reliability_unit_cost = required_annual_cost / max(event_total_kwh, 1.0)
                overbuild_unit_cost = required_overbuild_annual_cost / max(event_total_kwh, 1.0)

                result_rows.append(
                    {
                        "event_id": event["event_id"],
                        "province_en": province,
                        "boundary_metric": boundary["boundary_metric"],
                        "boundary_role": boundary["boundary_role"],
                        "peaking_window_h": window_h,
                        "peaking_window_start": peaking_window_start,
                        "event_length_days": int(event["length_days"]),
                        "start_day_idx": int(event["start_day_idx"]),
                        "end_day_idx": int(event["end_day_idx"]),
                        "total_event_demand_kwh": event_total_kwh,
                        "baseline_LFP_boundary_mwh": baseline_capacity_mwh,
                        "baseline_served_extreme_demand_kwh": baseline_result.served_kwh,
                        "unserved_extreme_demand_kwh": baseline_result.unserved_kwh,
                        "baseline_served_ratio_pct": (
                            100.0 * baseline_result.served_kwh / event_total_kwh
                            if event_total_kwh > 0
                            else np.nan
                        ),
                        "event_completed": int(baseline_result.completed),
                        "SOC_safety_floor_hours": baseline_result.soc_safety_floor_hours,
                        "SOC_safety_floor_days": baseline_result.soc_safety_floor_days,
                        "min_SOC_pct": baseline_result.min_soc_pct,
                        "final_SOC_pct": baseline_result.final_soc_pct,
                        "charged_from_real_curtailment_kwh": (
                            baseline_result.charged_from_curtailment_kwh
                        ),
                        "required_LFP_capacity_for_completion_mwh": required_mwh,
                        "required_LFP_overbuild_MWh": required_overbuild_mwh,
                        "required_LFP_overbuild_annual_cost_cny": required_overbuild_annual_cost,
                        "required_overbuild_ratio_to_boundary": (
                            required_overbuild_mwh / baseline_capacity_mwh
                            if baseline_capacity_mwh > 0
                            else np.nan
                        ),
                        "required_completion_min_SOC_pct": required_result.min_soc_pct,
                        "baseline_extreme_event_unit_service_cost_cny_per_kwh": baseline_unit_cost,
                        "extreme_event_unit_service_cost_cny_per_kwh": reliability_unit_cost,
                        "lfp_only_overbuild_unit_service_cost_cny_per_kwh": overbuild_unit_cost,
                        "required_reliability_annual_cost_cny": required_annual_cost,
                    }
                )

    results = pd.DataFrame(result_rows)
    summary = (
        results.groupby(["province_en", "boundary_metric", "boundary_role", "peaking_window_h"])
        .agg(
            event_count=("event_id", "nunique"),
            event_completion_rate=("event_completed", "mean"),
            total_unserved_extreme_demand_kwh=("unserved_extreme_demand_kwh", "sum"),
            mean_unserved_extreme_demand_kwh=("unserved_extreme_demand_kwh", "mean"),
            mean_baseline_served_ratio_pct=("baseline_served_ratio_pct", "mean"),
            total_SOC_safety_floor_days=("SOC_safety_floor_days", "sum"),
            mean_SOC_safety_floor_days=("SOC_safety_floor_days", "mean"),
            mean_required_LFP_overbuild_MWh=("required_LFP_overbuild_MWh", "mean"),
            max_required_LFP_overbuild_MWh=("required_LFP_overbuild_MWh", "max"),
            mean_required_LFP_overbuild_annual_cost_cny=(
                "required_LFP_overbuild_annual_cost_cny",
                "mean",
            ),
            mean_extreme_event_unit_service_cost_cny_per_kwh=(
                "extreme_event_unit_service_cost_cny_per_kwh",
                "mean",
            ),
            mean_lfp_only_overbuild_unit_service_cost_cny_per_kwh=(
                "lfp_only_overbuild_unit_service_cost_cny_per_kwh",
                "mean",
            ),
            baseline_LFP_boundary_mwh=("baseline_LFP_boundary_mwh", "first"),
        )
        .reset_index()
    )
    summary["event_completion_rate"] = summary["event_completion_rate"] * 100.0

    results.to_csv(OUTPUT_DIR / "lfp_only_event_reliability.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(
        OUTPUT_DIR / "lfp_only_event_reliability_summary.csv", index=False, encoding="utf-8-sig"
    )
    return results, summary


def run_lfp_ldes_residual_comparison(
    demand: pd.DataFrame,
    events: pd.DataFrame,
    lfp_results: pd.DataFrame,
    hourly_curtailment: pd.DataFrame,
    peaking_window_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame()
    if events.empty or lfp_results.empty:
        for filename in [
            "lfp_ldes_residual_event_reliability.csv",
            "lfp_ldes_residual_event_summary_by_tech.csv",
            "lfp_only_vs_lfp_ldes_best_event.csv",
            "lfp_only_vs_lfp_ldes_best_summary.csv",
            "lfp_ldes_best_tech_counts.csv",
        ]:
            empty.to_csv(OUTPUT_DIR / filename, index=False)
        return empty, empty, empty, empty, empty

    ldes_params = load_ldes_tech_params()
    events_by_id = events.set_index("event_id")
    hourly_sorted = hourly_curtailment.sort_values("hour_idx")
    curtailment_by_province = {
        province: hourly_sorted[province].astype(float).to_numpy()
        for province in sorted(lfp_results["province_en"].unique())
    }

    rows: list[dict[str, float | int | str]] = []
    for _, lfp_row in lfp_results.iterrows():
        event_id = str(lfp_row["event_id"])
        event = events_by_id.loc[event_id]
        province = str(lfp_row["province_en"])
        start_pos = int(event["start_pos"])
        end_pos = int(event["end_pos"])
        demand_values = demand.loc[start_pos:end_pos, province].astype(float).to_numpy()
        event_total_kwh = float(lfp_row["total_event_demand_kwh"])
        lfp_capacity_mwh = float(lfp_row["baseline_LFP_boundary_mwh"])
        peaking_window_h = int(lfp_row["peaking_window_h"])
        curtailment_values = curtailment_by_province[province]
        lfp_overbuild_cost = float(lfp_row["required_LFP_overbuild_annual_cost_cny"])
        lfp_unserved_kwh = float(lfp_row["unserved_extreme_demand_kwh"])

        for item in ldes_params:
            tp = item["params"]
            required_ldes_mwh, hybrid_result = required_ldes_capacity_for_completion(
                demand_values,
                start_pos,
                province,
                curtailment_values,
                lfp_capacity_mwh,
                tp,
                peaking_window_h,
                peaking_window_start,
            )
            if np.isfinite(required_ldes_mwh):
                ldes_capex_cny, ldes_annual_cost_cny = annualized_storage_cost(
                    tp, required_ldes_mwh
                )
                required_ldes_power_mw = required_ldes_mwh / float(tp.duration_h)
            else:
                ldes_capex_cny = np.nan
                ldes_annual_cost_cny = np.nan
                required_ldes_power_mw = np.nan

            residual_coverage_pct = (
                100.0 * hybrid_result.ldes_served_kwh / lfp_unserved_kwh
                if lfp_unserved_kwh > 0
                else 100.0
            )
            ldes_unit_cost = ldes_annual_cost_cny / max(event_total_kwh, 1.0)
            incremental_cost_saving = lfp_overbuild_cost - ldes_annual_cost_cny
            incremental_cost_saving_pct = (
                100.0 * incremental_cost_saving / lfp_overbuild_cost
                if lfp_overbuild_cost > 0
                else np.nan
            )

            rows.append(
                {
                    "event_id": event_id,
                    "province_en": province,
                    "boundary_metric": lfp_row["boundary_metric"],
                    "boundary_role": lfp_row["boundary_role"],
                    "peaking_window_h": peaking_window_h,
                    "peaking_window_start": peaking_window_start,
                    "event_length_days": int(lfp_row["event_length_days"]),
                    "start_day_idx": int(lfp_row["start_day_idx"]),
                    "end_day_idx": int(lfp_row["end_day_idx"]),
                    "total_event_demand_kwh": event_total_kwh,
                    "baseline_LFP_boundary_mwh": lfp_capacity_mwh,
                    "lfp_only_event_completed": int(lfp_row["event_completed"]),
                    "lfp_only_unserved_extreme_demand_kwh": lfp_unserved_kwh,
                    "lfp_only_required_overbuild_MWh": float(
                        lfp_row["required_LFP_overbuild_MWh"]
                    ),
                    "lfp_only_overbuild_annual_cost_cny": lfp_overbuild_cost,
                    "lfp_only_overbuild_unit_service_cost_cny_per_kwh": float(
                        lfp_row["lfp_only_overbuild_unit_service_cost_cny_per_kwh"]
                    ),
                    "ldes_tech_id": item["tech_id"],
                    "ldes_technology_en": item["technology_en"],
                    "ldes_technology_cn": item["technology_cn"],
                    "ldes_duration_h": float(tp.duration_h),
                    "ldes_rte": float(tp.rte),
                    "required_LDES_MWh": required_ldes_mwh,
                    "required_LDES_MW": required_ldes_power_mw,
                    "ldes_capex_cny": ldes_capex_cny,
                    "ldes_annual_cost_cny": ldes_annual_cost_cny,
                    "ldes_unit_service_cost_cny_per_kwh": ldes_unit_cost,
                    "ldes_to_lfp_overbuild_cost_ratio": (
                        ldes_annual_cost_cny / lfp_overbuild_cost
                        if lfp_overbuild_cost > 0
                        else np.nan
                    ),
                    "incremental_cost_saving_cny": incremental_cost_saving,
                    "incremental_cost_saving_pct": incremental_cost_saving_pct,
                    "lfp_ldes_event_completed": int(hybrid_result.completed),
                    "lfp_ldes_unserved_extreme_demand_kwh": hybrid_result.unserved_kwh,
                    "ldes_served_residual_demand_kwh": hybrid_result.ldes_served_kwh,
                    "ldes_residual_coverage_pct": residual_coverage_pct,
                    "lfp_served_with_ldes_kwh": hybrid_result.lfp_served_kwh,
                    "ldes_charged_from_real_curtailment_kwh": (
                        hybrid_result.ldes_charged_from_curtailment_kwh
                    ),
                    "ldes_min_SOC_pct": hybrid_result.ldes_min_soc_pct,
                    "ldes_final_SOC_pct": hybrid_result.ldes_final_soc_pct,
                    "ldes_SOC_safety_floor_hours": hybrid_result.ldes_safety_floor_hours,
                    "ldes_SOC_safety_floor_days": hybrid_result.ldes_safety_floor_days,
                    "lfp_SOC_safety_floor_days_with_ldes": hybrid_result.lfp_safety_floor_days,
                }
            )

    all_results = pd.DataFrame(rows)
    all_results.to_csv(
        OUTPUT_DIR / "lfp_ldes_residual_event_reliability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tech_summary = (
        all_results.groupby(
            [
                "province_en",
                "boundary_metric",
                "boundary_role",
                "peaking_window_h",
                "ldes_tech_id",
                "ldes_duration_h",
            ]
        )
        .agg(
            event_count=("event_id", "nunique"),
            lfp_only_event_completion_rate=("lfp_only_event_completed", "mean"),
            lfp_ldes_event_completion_rate=("lfp_ldes_event_completed", "mean"),
            total_lfp_only_unserved_extreme_demand_kwh=(
                "lfp_only_unserved_extreme_demand_kwh",
                "sum",
            ),
            total_lfp_ldes_unserved_extreme_demand_kwh=(
                "lfp_ldes_unserved_extreme_demand_kwh",
                "sum",
            ),
            mean_required_LFP_overbuild_MWh=("lfp_only_required_overbuild_MWh", "mean"),
            mean_lfp_only_overbuild_annual_cost_cny=(
                "lfp_only_overbuild_annual_cost_cny",
                "mean",
            ),
            mean_required_LDES_MWh=("required_LDES_MWh", "mean"),
            mean_required_LDES_MW=("required_LDES_MW", "mean"),
            mean_ldes_annual_cost_cny=("ldes_annual_cost_cny", "mean"),
            mean_ldes_to_lfp_overbuild_cost_ratio=(
                "ldes_to_lfp_overbuild_cost_ratio",
                "mean",
            ),
            mean_incremental_cost_saving_pct=("incremental_cost_saving_pct", "mean"),
            mean_lfp_only_overbuild_unit_service_cost_cny_per_kwh=(
                "lfp_only_overbuild_unit_service_cost_cny_per_kwh",
                "mean",
            ),
            mean_ldes_unit_service_cost_cny_per_kwh=(
                "ldes_unit_service_cost_cny_per_kwh",
                "mean",
            ),
        )
        .reset_index()
    )
    for col in ["lfp_only_event_completion_rate", "lfp_ldes_event_completion_rate"]:
        tech_summary[col] = tech_summary[col] * 100.0
    tech_summary.to_csv(
        OUTPUT_DIR / "lfp_ldes_residual_event_summary_by_tech.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best = (
        all_results.sort_values(
            [
                "event_id",
                "boundary_metric",
                "peaking_window_h",
                "ldes_annual_cost_cny",
                "required_LDES_MWh",
            ]
        )
        .drop_duplicates(["event_id", "boundary_metric", "peaking_window_h"], keep="first")
        .reset_index(drop=True)
    )
    best.to_csv(
        OUTPUT_DIR / "lfp_only_vs_lfp_ldes_best_event.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_summary = (
        best.groupby(["province_en", "boundary_metric", "boundary_role", "peaking_window_h"])
        .agg(
            event_count=("event_id", "nunique"),
            lfp_only_event_completion_rate=("lfp_only_event_completed", "mean"),
            lfp_ldes_event_completion_rate=("lfp_ldes_event_completed", "mean"),
            total_lfp_only_unserved_extreme_demand_kwh=(
                "lfp_only_unserved_extreme_demand_kwh",
                "sum",
            ),
            total_lfp_ldes_unserved_extreme_demand_kwh=(
                "lfp_ldes_unserved_extreme_demand_kwh",
                "sum",
            ),
            mean_required_LFP_overbuild_MWh=("lfp_only_required_overbuild_MWh", "mean"),
            max_required_LFP_overbuild_MWh=("lfp_only_required_overbuild_MWh", "max"),
            mean_lfp_only_overbuild_annual_cost_cny=(
                "lfp_only_overbuild_annual_cost_cny",
                "mean",
            ),
            mean_required_LDES_MWh=("required_LDES_MWh", "mean"),
            max_required_LDES_MWh=("required_LDES_MWh", "max"),
            mean_required_LDES_MW=("required_LDES_MW", "mean"),
            mean_ldes_annual_cost_cny=("ldes_annual_cost_cny", "mean"),
            mean_ldes_to_lfp_overbuild_cost_ratio=(
                "ldes_to_lfp_overbuild_cost_ratio",
                "mean",
            ),
            mean_incremental_cost_saving_cny=("incremental_cost_saving_cny", "mean"),
            mean_incremental_cost_saving_pct=("incremental_cost_saving_pct", "mean"),
            mean_lfp_only_overbuild_unit_service_cost_cny_per_kwh=(
                "lfp_only_overbuild_unit_service_cost_cny_per_kwh",
                "mean",
            ),
            mean_ldes_unit_service_cost_cny_per_kwh=(
                "ldes_unit_service_cost_cny_per_kwh",
                "mean",
            ),
        )
        .reset_index()
    )
    for col in ["lfp_only_event_completion_rate", "lfp_ldes_event_completion_rate"]:
        best_summary[col] = best_summary[col] * 100.0
    best_summary.to_csv(
        OUTPUT_DIR / "lfp_only_vs_lfp_ldes_best_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tech_counts = (
        best.groupby(
            [
                "province_en",
                "boundary_metric",
                "boundary_role",
                "peaking_window_h",
                "ldes_tech_id",
                "ldes_duration_h",
            ]
        )
        .size()
        .reset_index(name="best_event_count")
    )
    tech_counts.to_csv(
        OUTPUT_DIR / "lfp_ldes_best_tech_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return all_results, tech_summary, best, best_summary, tech_counts


def write_run_manifest(args: argparse.Namespace, rows: dict[str, int]) -> None:
    manifest = {
        "module": "m4_lfp_event_reliability",
        "purpose": (
            "Minimal loop for paper-2 event reliability: LFP-only baseline and "
            "baseline LFP + LDES residual demand comparison."
        ),
        "main_boundary_metric": MAIN_BOUNDARY_METRIC,
        "robustness_boundary_metrics": ["csr", "psr"],
        "lfp_duration_h": LFP_DURATION_H,
        "ldes_techs": LDES_TECHS,
        "ldes_durations_h": LDES_DURATIONS_H,
        "soc_min": SOC_MIN,
        "soc_max": SOC_MAX,
        "charge_eff": CHARGE_EFF,
        "discharge_eff": DISCHARGE_EFF,
        "event_quantile": args.event_quantile,
        "min_event_days": args.min_event_days,
        "peaking_windows_h": PEAKING_WINDOWS_H,
        "peaking_window_start": args.peaking_window_start,
        "provinces": args.provinces,
        "row_counts": rows,
        "note": (
            "The real-event loop uses observed consecutive q75 events. "
            "Longer 5/7/10/14-day stress cases are intentionally left for simulation."
        ),
    }
    with (OUTPUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    boundaries = load_baseline_boundaries(args.provinces)
    demand, events, event_summary = build_real_events(
        args.provinces, args.event_quantile, args.min_event_days
    )
    hourly_curtailment = build_hourly_curtailment_cache(
        args.provinces, force_hourly_cache=args.force_hourly_cache
    )
    results, summary = run_event_reliability(
        demand, events, boundaries, hourly_curtailment, args.peaking_window_start
    )
    ldes_results, ldes_summary, best_event, best_summary, tech_counts = (
        run_lfp_ldes_residual_comparison(
            demand, events, results, hourly_curtailment, args.peaking_window_start
        )
    )
    write_run_manifest(
        args,
        {
            "baseline_boundary_rows": len(boundaries),
            "real_event_rows": len(events),
            "event_length_summary_rows": len(event_summary),
            "lfp_only_result_rows": len(results),
            "lfp_only_summary_rows": len(summary),
            "lfp_ldes_result_rows": len(ldes_results),
            "lfp_ldes_summary_by_tech_rows": len(ldes_summary),
            "lfp_ldes_best_event_rows": len(best_event),
            "lfp_ldes_best_summary_rows": len(best_summary),
            "lfp_ldes_best_tech_count_rows": len(tech_counts),
            "hourly_curtailment_rows": len(hourly_curtailment),
        },
    )

    print("Wrote:")
    for path in [
        OUTPUT_DIR / "baseline_LFP_boundary.csv",
        OUTPUT_DIR / "real_q75_events_ge3.csv",
        OUTPUT_DIR / "real_q75_all_runs.csv",
        OUTPUT_DIR / "real_q75_event_length_summary.csv",
        OUTPUT_DIR / "lfp_only_event_reliability.csv",
        OUTPUT_DIR / "lfp_only_event_reliability_summary.csv",
        OUTPUT_DIR / "ldes_baseline_tech_params.csv",
        OUTPUT_DIR / "lfp_ldes_residual_event_reliability.csv",
        OUTPUT_DIR / "lfp_ldes_residual_event_summary_by_tech.csv",
        OUTPUT_DIR / "lfp_only_vs_lfp_ldes_best_event.csv",
        OUTPUT_DIR / "lfp_only_vs_lfp_ldes_best_summary.csv",
        OUTPUT_DIR / "lfp_ldes_best_tech_counts.csv",
        OUTPUT_DIR / "run_manifest.json",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
