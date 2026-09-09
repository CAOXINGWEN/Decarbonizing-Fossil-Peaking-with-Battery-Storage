#!/usr/bin/env python3
"""Replay 35 PECD weather years through the focal 4 h LFP--24 h A-CAES model.

The experiment is designed to put natural weather chronology on the same
scale as the manuscript's conserved chronology experiment.  For each country
and weather year, 2030 demand is netted against the 2030 onshore-wind,
offshore-wind and solar-PV mix.  VRE is scaled to annual demand so that the
positive residual is a weather-driven firm-power task and the negative
residual is a physically synchronized recharge opportunity.  Within each
country, annual task and recharge totals are then standardized to their
35-year medians.  The resulting experiment changes chronology and event shape
while holding system scale, annual task and annual recharge fixed.

The fixed-power portfolio equations are an algebraically reduced version of
the registered v54 LP: unserved energy is eliminated using
u = task - d_base - d_tail.  This leaves the same cyclic SOC, shared recharge,
tail eligibility, energy-value and residual-peak terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix


HERE = Path(__file__).resolve().parent
V54 = HERE.parent
MUSRIPT_ROOT = V54.parent


def find_workspace() -> Path:
    for candidate in HERE.parents:
        if (candidate / "dataset" / "external_hourly").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the workspace dataset directory")


WORKSPACE = find_workspace()
DATA_ROOT = (
    WORKSPACE
    / "dataset"
    / "external_hourly"
    / "europe_mid_duration_valley_upgrade_v0_1"
)
RAW = DATA_ROOT / "raw" / "pecd_2021_3_country_2030"
PREPARED = HERE / "prepared"
RESULTS = HERE / "results"

DEMAND_FILE = RAW / "PECD-country-demand_national_estimates-2030.parquet"
ONSHORE_FILE = RAW / "PECD-2021.3-country-Onshore-2030.parquet"
OFFSHORE_FILE = RAW / "PECD-2021.3-country-Offshore-2030.parquet"
SOLAR_FILE = RAW / "PECD-2021.3-country-LFSolarPV-2030.parquet"
CAPACITY_FILE = RAW / "capacities-national-estimates.csv"
SOURCE_MANIFEST = (
    DATA_ROOT / "artifacts" / "research" / "PECD_2021_3_ZENODO_SOURCE_MANIFEST.csv"
)

CATALOGUE_FILE = (
    MUSRIPT_ROOT
    / "revision_nature_energy_v40"
    / "currency_work"
    / "configuration_catalog_53_source_aware_eur.csv"
)
SCENARIO_FILE = (
    MUSRIPT_ROOT
    / "revision_nature_energy_v40"
    / "currency_work"
    / "value_sensitivity_gap_summary_source_aware_eur.csv"
)
PARITY_AUDIT = V54 / "sequencing_parity" / "v50_sequencing_parity_audit.json"
NATURAL_FE_COEFFICIENTS = V54 / "joint_choice" / "natural_fixed_effect_coefficients.csv"

YEARS = np.arange(1982, 2017, dtype=int)
COUNTRIES = ["AT", "BE", "BG", "CZ", "DE", "DK", "ES", "FR", "GB", "GR", "HR", "IT", "PT", "RO", "SI"]
SOURCE_COUNTRY = {country: ("UK" if country == "GB" else country) for country in COUNTRIES}
SOURCE_TO_MANUSCRIPT = {source: country for country, source in SOURCE_COUNTRY.items()}

HOURS = 8760
DAYS = 365
SOC_USABLE_FRACTION = 0.90
FORCED_POWER_SHARE = 0.01
FLOOR_QUANTILE = 0.10
EVENT_SHARE = 0.25
PRIMARY_ABS_TOL_EUR = 0.01

SEASON_SEGMENTS = {
    1: (1, 2),
    2: (3, 4, 5),
    3: (6, 7, 8),
    4: (9, 10, 11),
    5: (12,),
}
EXPECTED_SEGMENT_DAYS = {1: 59, 2: 92, 3: 92, 4: 91, 5: 31}

EXPECTED_MD5 = {
    DEMAND_FILE.name: "1460a0b93e92e265c7208606c270f0ca",
    ONSHORE_FILE.name: "7b1ecb37f4d38456c0c04f769e96ab54",
    OFFSHORE_FILE.name: "eb42d2ff92cbcaab5a5dd247f4589b71",
    SOLAR_FILE.name: "ee09150e658a27e9456cef441f2b373b",
    CAPACITY_FILE.name: "b7a8e2b23f3d7a47ecc50bb36ef4c4fb",
}

# The 60-year 24 h A-CAES value is the approved v54 cost anchor.
CAES24_FIXED_COST_EUR_KW_YEAR = 56.87352019939117


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def output_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def source_audit() -> dict[str, Any]:
    rows = []
    for path in [DEMAND_FILE, ONSHORE_FILE, OFFSHORE_FILE, SOLAR_FILE, CAPACITY_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)
        actual = md5(path)
        expected = EXPECTED_MD5[path.name]
        rows.append(
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "expected_md5": expected,
                "actual_md5": actual,
                "verified": actual == expected,
            }
        )
    if not all(row["verified"] for row in rows):
        raise RuntimeError("At least one PECD source file failed MD5 verification")
    manifest = pd.read_csv(SOURCE_MANIFEST)
    selected = manifest[manifest.file_key.isin(EXPECTED_MD5)].copy()
    return {
        "status": "pass",
        "record_id": 7224854,
        "version_doi": "10.5281/zenodo.7224854",
        "concept_doi": "10.5281/zenodo.5780184",
        "license": "CC BY 4.0",
        "target_scenario": "National Estimates 2030",
        "climate_years": YEARS.tolist(),
        "weather_year_count": len(YEARS),
        "countries": COUNTRIES,
        "files": rows,
        "manifest_rows_found": int(len(selected)),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
    }


def read_filtered_parquet(path: Path, value_column: str) -> pd.DataFrame:
    columns = ["country", "year", "month", "day", "hour", value_column]
    table = pq.read_table(path, columns=columns)
    country_mask = pc.is_in(
        table["country"], value_set=pa.array(sorted(SOURCE_TO_MANUSCRIPT), type=pa.string())
    )
    year_mask = pc.and_(pc.greater_equal(table["year"], 1982), pc.less_equal(table["year"], 2016))
    selected = table.filter(pc.and_(country_mask, year_mask)).to_pandas()
    selected["year"] = selected["year"].astype(int)
    selected[["month", "day", "hour"]] = selected[["month", "day", "hour"]].astype(int)
    selected["country"] = selected["country"].map(SOURCE_TO_MANUSCRIPT)
    return selected


def frame_to_arrays(frame: pd.DataFrame, value_column: str) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, np.ndarray]]:
    arrays: dict[str, dict[int, np.ndarray]] = {country: {} for country in COUNTRIES}
    months: dict[int, np.ndarray] = {}
    for (country, year), group in frame.groupby(["country", "year"], sort=True):
        group = group.sort_values(["month", "day", "hour"], kind="stable")
        if len(group) != HOURS or group[["month", "day", "hour"]].duplicated().any():
            raise RuntimeError(f"Invalid PECD hourly coverage for {country}-{year}")
        value = group[value_column].to_numpy(float)
        if not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite PECD values for {country}-{year}")
        arrays[str(country)][int(year)] = value
        month = group["month"].to_numpy(np.int8)
        if year not in months:
            months[year] = month
        elif not np.array_equal(months[year], month):
            raise RuntimeError(f"Inconsistent calendar coordinates in climate year {year}")
    return arrays, months


def load_hourly_inputs() -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[str, dict[int, np.ndarray]]], dict[int, np.ndarray]]:
    demand_frame = read_filtered_parquet(DEMAND_FILE, "dem_MW")
    demand, months = frame_to_arrays(demand_frame, "dem_MW")
    del demand_frame

    technology: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for name, path in [("onshore", ONSHORE_FILE), ("offshore", OFFSHORE_FILE), ("solar", SOLAR_FILE)]:
        frame = read_filtered_parquet(path, "cf")
        arrays, technology_months = frame_to_arrays(frame, "cf")
        technology[name] = arrays
        for year, month in technology_months.items():
            if not np.array_equal(months[year], month):
                raise RuntimeError(f"{name} calendar does not match demand in {year}")
        del frame

    for country in COUNTRIES:
        if set(demand[country]) != set(YEARS):
            raise RuntimeError(f"Demand does not cover all 35 years for {country}")
        for name in ["onshore", "solar"]:
            if set(technology[name][country]) != set(YEARS):
                raise RuntimeError(f"{name} does not cover all 35 years for {country}")
    return demand, technology, months


def load_capacities() -> pd.DataFrame:
    frame = pd.read_csv(CAPACITY_FILE)
    frame = frame[frame.scenario.eq("National Estimates 2030")].copy()
    frame["country"] = frame["country"].map(SOURCE_TO_MANUSCRIPT)
    frame = frame[frame.country.notna()]
    technologies = {
        "Wind Onshore": "onshore_capacity_mw",
        "Wind Offshore": "offshore_capacity_mw",
        "Solar (Photovoltaic)": "solar_capacity_mw",
    }
    frame = frame[frame.tech.isin(technologies)]
    frame["technology"] = frame.tech.map(technologies)
    unique = frame[["country", "technology", "total_country"]].drop_duplicates()
    conflicts = unique.groupby(["country", "technology"]).total_country.nunique()
    if (conflicts > 1).any():
        raise RuntimeError("Capacity table has conflicting repeated total_country values")
    wide = unique.pivot(index="country", columns="technology", values="total_country").fillna(0.0)
    wide = wide.reindex(COUNTRIES).reset_index()
    if wide.isna().any().any() or (wide.drop(columns="country") < 0.0).any().any():
        raise RuntimeError("Invalid 2030 VRE capacities")
    return wide


def season_ids(month: np.ndarray) -> np.ndarray:
    result = np.zeros(len(month), dtype=np.int8)
    for segment, members in SEASON_SEGMENTS.items():
        result[np.isin(month, members)] = segment
    if (result == 0).any():
        raise RuntimeError("Calendar contains an unmapped month")
    return result


def longest_run(flags: np.ndarray, segments: np.ndarray) -> int:
    best = 0
    run = 0
    previous = -1
    for flag, segment in zip(flags, segments):
        if segment != previous:
            run = 0
        run = run + 1 if flag else 0
        best = max(best, run)
        previous = int(segment)
    return int(best)


def derive_tasks(raw_task: np.ndarray, recharge: np.ndarray, month: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    hour_of_day = np.tile(np.arange(24, dtype=np.int8), DAYS)
    floor = np.zeros(HOURS, dtype=float)
    for current_month in range(1, 13):
        for hour in range(24):
            mask = (month == current_month) & (hour_of_day == hour)
            floor[mask] = np.quantile(raw_task[mask], FLOOR_QUANTILE, method="linear")
    base_task = np.maximum(raw_task - floor, 0.0)
    daily = base_task.reshape(DAYS, 24).sum(axis=1)
    daily_month = month.reshape(DAYS, 24)[:, 0]
    segments = season_ids(daily_month)
    event = np.zeros(DAYS, dtype=bool)
    for segment, expected_days in EXPECTED_SEGMENT_DAYS.items():
        positions = np.flatnonzero(segments == segment)
        if len(positions) != expected_days:
            raise RuntimeError(f"Segment {segment} has {len(positions)} days, expected {expected_days}")
        target = math.ceil(len(positions) * EVENT_SHARE)
        order = np.lexsort((positions, -daily[positions]))
        event[positions[order[:target]]] = True
    tail_task = base_task * np.repeat(event, 24)

    recovery = []
    for segment in sorted(SEASON_SEGMENTS):
        positions = np.flatnonzero(event & (segments == segment))
        if len(positions) > 1:
            recovery.extend((np.diff(positions) - 1) * 24.0)
    recovery_array = np.asarray(recovery, dtype=float)
    metrics = {
        "event_days": int(event.sum()),
        "natural_longest_event_run_days": longest_run(event, segments),
        "natural_zero_recovery_window_share": float(np.mean(recovery_array == 0.0)),
        "natural_median_recovery_window_h": float(np.median(recovery_array)),
        "annual_base_task_mwh": float(base_task.sum()),
        "annual_tail_task_mwh": float(tail_task.sum()),
        "annual_recharge_mwh": float(recharge.sum()),
        "recharge_to_base_task_ratio": float(recharge.sum() / max(base_task.sum(), 1.0e-12)),
    }
    return base_task, tail_task, event, metrics


def prepare() -> None:
    PREPARED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    audit = source_audit()
    write_json(RESULTS / "pecd_source_audit.json", audit)
    print("PECD source audit passed", flush=True)

    demand, technology, months = load_hourly_inputs()
    capacities = load_capacities().set_index("country")
    summary_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []

    for country in COUNTRIES:
        cap = capacities.loc[country]
        raw_by_year: dict[int, dict[str, Any]] = {}
        for year in YEARS:
            load = demand[country][int(year)]
            onshore = technology["onshore"][country][int(year)]
            solar = technology["solar"][country][int(year)]
            offshore = technology["offshore"][country].get(int(year), np.zeros(HOURS))
            if cap.offshore_capacity_mw > 0.0 and int(year) not in technology["offshore"][country]:
                raise RuntimeError(f"Missing offshore capacity factors for {country}-{year}")
            vre_unscaled = (
                cap.onshore_capacity_mw * onshore
                + cap.offshore_capacity_mw * offshore
                + cap.solar_capacity_mw * solar
            )
            if vre_unscaled.sum() <= 0.0:
                raise RuntimeError(f"Zero VRE generation for {country}-{year}")
            alpha = float(load.sum() / vre_unscaled.sum())
            residual = load - alpha * vre_unscaled
            residual[np.abs(residual) < 1.0e-10] = 0.0
            raw_task = np.maximum(residual, 0.0)
            raw_recharge = np.maximum(-residual, 0.0)
            if np.any((raw_task > 0.0) & (raw_recharge > 0.0)):
                raise RuntimeError("Task and recharge overlap")
            raw_by_year[int(year)] = {
                "raw_task": raw_task,
                "raw_recharge": raw_recharge,
                "load_mwh": float(load.sum()),
                "unscaled_vre_mwh": float(vre_unscaled.sum()),
                "vre_scaling_alpha": alpha,
            }

        target_task = float(np.median([value["raw_task"].sum() for value in raw_by_year.values()]))
        target_recharge = float(np.median([value["raw_recharge"].sum() for value in raw_by_year.values()]))
        fixed_power = FORCED_POWER_SHARE * target_task / HOURS
        if min(target_task, target_recharge, fixed_power) <= 0.0:
            raise RuntimeError(f"Invalid climatological scaling target for {country}")

        base_stack = []
        tail_stack = []
        recharge_stack = []
        event_stack = []
        for year in YEARS:
            raw = raw_by_year[int(year)]
            task_scale = target_task / raw["raw_task"].sum()
            recharge_scale = target_recharge / raw["raw_recharge"].sum()
            standardized_task = raw["raw_task"] * task_scale
            standardized_recharge = raw["raw_recharge"] * recharge_scale
            base_task, tail_task, event, metrics = derive_tasks(
                standardized_task, standardized_recharge, months[int(year)]
            )
            base_stack.append(base_task)
            tail_stack.append(tail_task)
            recharge_stack.append(standardized_recharge)
            event_stack.append(event.astype(np.int8))
            summary_rows.append(
                {
                    "country": country,
                    "weather_year": int(year),
                    "system_definition": "2030_mix_annual_vre_equals_demand",
                    "standardization": "country_35y_median_raw_task_and_recharge",
                    "demand_mwh": raw["load_mwh"],
                    "unscaled_vre_mwh": raw["unscaled_vre_mwh"],
                    "vre_scaling_alpha": raw["vre_scaling_alpha"],
                    "raw_task_mwh": float(raw["raw_task"].sum()),
                    "raw_recharge_mwh": float(raw["raw_recharge"].sum()),
                    "task_standardization_factor": task_scale,
                    "recharge_standardization_factor": recharge_scale,
                    "target_raw_task_mwh": target_task,
                    "target_raw_recharge_mwh": target_recharge,
                    "fixed_common_power_mw": fixed_power,
                    **metrics,
                }
            )
            geometry_rows.append(
                {
                    "country": country,
                    "weather_year": int(year),
                    "fixed_common_power_mw": fixed_power,
                    **metrics,
                }
            )
        np.savez_compressed(
            PREPARED / f"{country}_pecd_1982_2016_standardized.npz",
            years=YEARS,
            base_task_mw=np.asarray(base_stack),
            tail_task_mw=np.asarray(tail_stack),
            recharge_mw=np.asarray(recharge_stack),
            event_day=np.asarray(event_stack),
            fixed_common_power_mw=np.asarray([fixed_power]),
        )
        print(f"prepared {country}: 35 weather years", flush=True)

    summary = pd.DataFrame(summary_rows)
    geometry = pd.DataFrame(geometry_rows)
    summary.to_csv(RESULTS / "pecd_35y_prepared_input_summary.csv", index=False)
    geometry.to_csv(RESULTS / "pecd_35y_event_geometry.csv", index=False)
    capacities.reset_index().to_csv(RESULTS / "pecd_2030_vre_capacities_15_countries.csv", index=False)
    qa = {
        "status": "pass",
        "countries": int(summary.country.nunique()),
        "weather_years": int(summary.weather_year.nunique()),
        "country_weather_years": int(len(summary)),
        "expected_country_weather_years": len(COUNTRIES) * len(YEARS),
        "all_event_day_counts_equal_92": bool(geometry.event_days.eq(92).all()),
        "all_fixed_power_positive": bool(summary.fixed_common_power_mw.gt(0.0).all()),
        "maximum_standardized_raw_task_energy_error_mwh": float(
            summary.groupby("country").raw_task_mwh.transform("median").sub(summary.target_raw_task_mwh).abs().max()
        ),
        "prepared_files": {path.name: sha256(path) for path in sorted(PREPARED.glob("*.npz"))},
    }
    if len(summary) != len(COUNTRIES) * len(YEARS) or not qa["all_event_day_counts_equal_92"]:
        qa["status"] = "fail"
    write_json(RESULTS / "pecd_35y_preparation_qa.json", qa)
    if qa["status"] != "pass":
        raise RuntimeError("PECD preparation QA failed")


@dataclass(frozen=True)
class Technology:
    config_id: str
    duration_h: float
    rte: float
    self_discharge_pct_day: float
    charge_power_ratio: float
    fixed_cost_eur_kw_year: float
    vom_eur_mwh: float


@dataclass(frozen=True)
class Scenario:
    energy_value_eur_mwh: float
    firm_value_eur_kw_year: float
    charge_price_eur_mwh: float


def load_parameters() -> tuple[Technology, Technology, Scenario, float]:
    catalogue = pd.read_csv(CATALOGUE_FILE).set_index("config_id")

    def technology(config_id: str, fixed_override: float | None = None) -> Technology:
        row = catalogue.loc[config_id]
        fixed = float(row.annual_fixed_cost_eur_per_kw_year_source_aware)
        if fixed_override is not None:
            fixed = fixed_override
        return Technology(
            config_id=config_id,
            duration_h=float(row.duration_h),
            rte=float(row.rte),
            self_discharge_pct_day=float(row.self_discharge_pct_day),
            charge_power_ratio=float(row.charge_power_ratio),
            fixed_cost_eur_kw_year=fixed,
            vom_eur_mwh=float(row.variable_om_eur_mwh_source_aware),
        )

    scenario_frame = pd.read_csv(SCENARIO_FILE)
    row = scenario_frame[scenario_frame.scenario_id.eq(20)].iloc[0]
    scenario = Scenario(
        energy_value_eur_mwh=float(row.energy_value_eur_mwh),
        firm_value_eur_kw_year=float(row.firm_value_eur_kw_year),
        charge_price_eur_mwh=float(row.charge_price_eur_mwh),
    )
    threshold = float(json.loads(PARITY_AUDIT.read_text())["central_24h_cost_movement_eur_kw_year"])
    return technology("li_lfp_4h"), technology("caes_24h", CAES24_FIXED_COST_EUR_KW_YEAR), scenario, threshold


@dataclass
class ReducedModel:
    base: Technology
    tail: Technology
    scenario: Scenario
    a_eq: csr_matrix
    b_eq: np.ndarray
    a_ub: csr_matrix
    objective: np.ndarray


def build_reduced_model(base: Technology, tail: Technology, scenario: Scenario) -> ReducedModel:
    # Variable blocks: qB, dB, sB, qT, dT, sT, residual_peak.
    q_b, d_b, s_b, q_t, d_t, s_t = [index * HOURS for index in range(6)]
    residual_peak = 6 * HOURS
    n_variables = residual_peak + 1
    hours = np.arange(HOURS)
    previous = np.roll(hours, 1)

    rows = []
    cols = []
    values = []
    base_eta = math.sqrt(base.rte)
    tail_eta = math.sqrt(tail.rte)
    base_retention = (1.0 - base.self_discharge_pct_day / 100.0) ** (1.0 / 24.0)
    tail_retention = (1.0 - tail.self_discharge_pct_day / 100.0) ** (1.0 / 24.0)

    rows.extend([hours, hours, hours, hours])
    cols.extend([s_b + hours, s_b + previous, q_b + hours, d_b + hours])
    values.extend([
        np.ones(HOURS),
        np.full(HOURS, -base_retention),
        np.full(HOURS, -base_eta),
        np.full(HOURS, 1.0 / base_eta),
    ])
    tail_rows = HOURS + hours
    rows.extend([tail_rows, tail_rows, tail_rows, tail_rows])
    cols.extend([s_t + hours, s_t + previous, q_t + hours, d_t + hours])
    values.extend([
        np.ones(HOURS),
        np.full(HOURS, -tail_retention),
        np.full(HOURS, -tail_eta),
        np.full(HOURS, 1.0 / tail_eta),
    ])
    a_eq = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(2 * HOURS, n_variables),
    ).tocsr()

    # Shared recharge; total discharge <= task; residual_peak >= unserved.
    ub_rows = np.concatenate([hours, hours, HOURS + hours, HOURS + hours, 2 * HOURS + hours, 2 * HOURS + hours, 2 * HOURS + hours])
    ub_cols = np.concatenate([
        q_b + hours,
        q_t + hours,
        d_b + hours,
        d_t + hours,
        d_b + hours,
        d_t + hours,
        np.full(HOURS, residual_peak),
    ])
    ub_values = np.concatenate([
        np.ones(HOURS),
        np.ones(HOURS),
        np.ones(HOURS),
        np.ones(HOURS),
        -np.ones(HOURS),
        -np.ones(HOURS),
        -np.ones(HOURS),
    ])
    a_ub = coo_matrix((ub_values, (ub_rows, ub_cols)), shape=(3 * HOURS, n_variables)).tocsr()

    objective = np.zeros(n_variables)
    objective[q_b:d_b] = scenario.charge_price_eur_mwh
    objective[d_b:s_b] = base.vom_eur_mwh - scenario.energy_value_eur_mwh
    objective[q_t:d_t] = scenario.charge_price_eur_mwh
    objective[d_t:s_t] = tail.vom_eur_mwh - scenario.energy_value_eur_mwh
    objective[residual_peak] = 1000.0 * scenario.firm_value_eur_kw_year
    return ReducedModel(base, tail, scenario, a_eq, np.zeros(2 * HOURS), a_ub, objective)


def equivalent_firm_capacity(task: np.ndarray, target_unserved: float) -> float:
    if target_unserved >= task.sum() - 1.0e-9:
        return 0.0
    low = 0.0
    high = float(task.max())
    for _ in range(70):
        midpoint = 0.5 * (low + high)
        unserved = float(np.maximum(task - midpoint, 0.0).sum())
        if unserved > target_unserved:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def solve_arm(model: ReducedModel, task: np.ndarray, tail_task: np.ndarray, recharge: np.ndarray, power: float) -> dict[str, Any]:
    q_b, d_b, s_b, q_t, d_t, s_t = [index * HOURS for index in range(6)]
    residual_peak_index = 6 * HOURS
    n_variables = residual_peak_index + 1
    lower = np.zeros(n_variables)
    upper = np.full(n_variables, np.inf)
    upper[q_b:d_b] = np.minimum(recharge, model.base.charge_power_ratio * power)
    upper[d_b:s_b] = np.minimum(task, power)
    upper[s_b:q_t] = SOC_USABLE_FRACTION * model.base.duration_h * power
    upper[q_t:d_t] = np.minimum(recharge, model.tail.charge_power_ratio * power)
    upper[d_t:s_t] = np.minimum(tail_task, power)
    upper[s_t:residual_peak_index] = SOC_USABLE_FRACTION * model.tail.duration_h * power
    b_ub = np.concatenate([recharge, task, -task])
    started = time.perf_counter()
    result = linprog(
        model.objective,
        A_ub=model.a_ub,
        b_ub=b_ub,
        A_eq=model.a_eq,
        b_eq=model.b_eq,
        bounds=np.column_stack([lower, upper]),
        method="highs-ds",
        options={
            "dual_feasibility_tolerance": 1.0e-8,
            "primal_feasibility_tolerance": 1.0e-8,
        },
    )
    runtime = time.perf_counter() - started
    if not result.success:
        return {
            "config_id": model.tail.config_id,
            "solver_status": int(result.status),
            "solver_status_name": "failed",
            "solver_message": result.message,
            "solver_runtime_seconds": runtime,
        }
    x = result.x
    q_base, d_base, s_base = x[q_b:d_b], x[d_b:s_b], x[s_b:q_t]
    q_tail, d_tail, s_tail = x[q_t:d_t], x[d_t:s_t], x[s_t:residual_peak_index]
    residual_peak = float(x[residual_peak_index])
    unserved = task - d_base - d_tail
    fixed_base = 1000.0 * power * model.base.fixed_cost_eur_kw_year
    fixed_tail = 1000.0 * power * model.tail.fixed_cost_eur_kw_year
    variable_objective = float(result.fun)
    constant_residual_energy = model.scenario.energy_value_eur_mwh * task.sum()
    system_cost = fixed_base + fixed_tail + constant_residual_energy + variable_objective
    equality_error = float(np.max(np.abs(model.a_eq @ x)))
    inequality_violation = float(np.maximum(model.a_ub @ x - b_ub, 0.0).max())
    bound_violation = float(max(np.maximum(lower - x, 0.0).max(), np.maximum(x - upper, 0.0).max()))
    return {
        "config_id": model.tail.config_id,
        "solver_status": int(result.status),
        "solver_status_name": "optimal",
        "solver_message": result.message,
        "solver_runtime_seconds": runtime,
        "solver_iterations": int(result.nit),
        "fixed_common_power_mw": power,
        "base_task_mwh": float(task.sum()),
        "tail_task_mwh": float(tail_task.sum()),
        "recharge_mwh": float(recharge.sum()),
        "base_charge_mwh": float(q_base.sum()),
        "tail_charge_mwh": float(q_tail.sum()),
        "base_discharge_mwh": float(d_base.sum()),
        "tail_discharge_mwh": float(d_tail.sum()),
        "unserved_eens_mwh": float(unserved.sum()),
        "residual_peak_mw": residual_peak,
        "portfolio_efc_mw": equivalent_firm_capacity(task, float(unserved.sum())),
        "base_equivalent_full_cycles": float(d_base.sum() / (model.base.duration_h * power)),
        "tail_equivalent_full_cycles": float(d_tail.sum() / (model.tail.duration_h * power)),
        "base_fixed_cost_eur": fixed_base,
        "tail_fixed_cost_eur": fixed_tail,
        "system_cost_eur": system_cost,
        "maximum_equality_residual": equality_error,
        "maximum_inequality_violation": inequality_violation,
        "maximum_bound_violation": bound_violation,
        "minimum_unserved_mw": float(unserved.min()),
        "maximum_base_soc_mwh": float(s_base.max()),
        "maximum_tail_soc_mwh": float(s_tail.max()),
    }


def solve_country(country: str) -> tuple[str, list[dict[str, Any]]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    base, caes, scenario, _ = load_parameters()
    models = {
        base.config_id: build_reduced_model(base, base, scenario),
        caes.config_id: build_reduced_model(base, caes, scenario),
    }
    data = np.load(PREPARED / f"{country}_pecd_1982_2016_standardized.npz")
    years = data["years"].astype(int)
    power = float(data["fixed_common_power_mw"][0])
    rows: list[dict[str, Any]] = []
    for index, year in enumerate(years):
        task = data["base_task_mw"][index].astype(float, copy=False)
        tail_task = data["tail_task_mw"][index].astype(float, copy=False)
        recharge = data["recharge_mw"][index].astype(float, copy=False)
        for config_id in [base.config_id, caes.config_id]:
            answer = solve_arm(models[config_id], task, tail_task, recharge, power)
            rows.append(
                {
                    "country": country,
                    "weather_year": int(year),
                    "chronology": "natural_pecd",
                    "annual_scale": "country_35y_median_standardized",
                    "scenario_id": 20,
                    "energy_value_eur_mwh": scenario.energy_value_eur_mwh,
                    "firm_value_eur_kw_year": scenario.firm_value_eur_kw_year,
                    "charge_price_eur_mwh": scenario.charge_price_eur_mwh,
                    **answer,
                }
            )
        print(f"{country} solved {index + 1}/35", flush=True)
    return country, rows


def solve(countries: list[str], workers: int) -> None:
    missing = [country for country in countries if not (PREPARED / f"{country}_pecd_1982_2016_standardized.npz").exists()]
    if missing:
        raise RuntimeError(f"Prepared inputs are missing for {missing}; run --stage prepare first")
    RESULTS.mkdir(parents=True, exist_ok=True)
    completed: dict[str, pd.DataFrame] = {}
    pending = []
    for country in countries:
        checkpoint = RESULTS / f"pecd_35y_arm_results_{country}.csv.gz"
        if checkpoint.exists():
            frame = pd.read_csv(checkpoint)
            if len(frame) == 70 and frame.solver_status_name.eq("optimal").all():
                completed[country] = frame
                print(f"reused verified checkpoint for {country}", flush=True)
                continue
        pending.append(country)

    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            future_map = {executor.submit(solve_country, country): country for country in pending}
            for future in as_completed(future_map):
                country, rows = future.result()
                frame = pd.DataFrame(rows)
                output_csv_gz(frame, RESULTS / f"pecd_35y_arm_results_{country}.csv.gz")
                completed[country] = frame
                print(f"completed {country}: {len(frame)} arm-years", flush=True)

    output = pd.concat([completed[country] for country in countries], ignore_index=True)
    output = output.sort_values(["country", "weather_year", "config_id"]).reset_index(drop=True)
    output_path = RESULTS / "pecd_35y_arm_results.csv.gz"
    output_csv_gz(output, output_path)
    qa = {
        "status": "pass",
        "countries": int(output.country.nunique()),
        "weather_years": int(output.weather_year.nunique()),
        "rows": int(len(output)),
        "expected_rows": len(countries) * len(YEARS) * 2,
        "all_optimal": bool(output.solver_status_name.eq("optimal").all()),
        "maximum_equality_residual": float(output.maximum_equality_residual.max()),
        "maximum_inequality_violation": float(output.maximum_inequality_violation.max()),
        "maximum_bound_violation": float(output.maximum_bound_violation.max()),
        "minimum_unserved_mw": float(output.minimum_unserved_mw.min()),
        "total_solver_runtime_seconds": float(output.solver_runtime_seconds.sum()),
        "wall_parallel_workers": workers,
        "output_sha256": sha256(output_path),
    }
    if len(output) != qa["expected_rows"] or not qa["all_optimal"] or qa["maximum_equality_residual"] > 1.0e-5 or qa["maximum_inequality_violation"] > 1.0e-5 or qa["minimum_unserved_mw"] < -1.0e-5:
        qa["status"] = "fail"
    write_json(RESULTS / "pecd_35y_optimization_qa.json", qa)
    if qa["status"] != "pass":
        raise RuntimeError("PECD optimization QA failed")


def summarize() -> None:
    path = RESULTS / "pecd_35y_arm_results.csv.gz"
    if not path.exists():
        raise RuntimeError("Full arm results are absent; run --stage solve first")
    arms = pd.read_csv(path)
    geometry = pd.read_csv(RESULTS / "pecd_35y_event_geometry.csv")
    base, _, _, threshold = load_parameters()
    expected_keys = len(COUNTRIES) * len(YEARS)
    wide = arms.pivot(
        index=["country", "weather_year", "fixed_common_power_mw"],
        columns="config_id",
        values=["system_cost_eur", "tail_equivalent_full_cycles", "portfolio_efc_mw", "unserved_eens_mwh"],
    )
    wide.columns = [f"{metric}_{config}" for metric, config in wide.columns]
    values = wide.reset_index()
    values["caes24_minus_lfp4_eur_kw_year"] = (
        values["system_cost_eur_li_lfp_4h"] - values["system_cost_eur_caes_24h"]
    ) / (1000.0 * values.fixed_common_power_mw)
    values["caes24_minus_lfp4_variable_eur_kw_year"] = (
        values.caes24_minus_lfp4_eur_kw_year
        + CAES24_FIXED_COST_EUR_KW_YEAR
        - base.fixed_cost_eur_kw_year
    )
    values = values.merge(geometry, on=["country", "weather_year", "fixed_common_power_mw"], validate="one_to_one")
    if len(values) != expected_keys:
        raise RuntimeError(f"Expected {expected_keys} relative-value rows, found {len(values)}")

    values["country_median_value"] = values.groupby("country").caes24_minus_lfp4_eur_kw_year.transform("median")
    values["within_country_value_deviation"] = values.caes24_minus_lfp4_eur_kw_year - values.country_median_value
    values["absolute_within_country_value_deviation"] = values.within_country_value_deviation.abs()
    values.to_csv(RESULTS / "pecd_35y_relative_values.csv", index=False)

    fe = pd.read_csv(NATURAL_FE_COEFFICIENTS)
    fe = fe[fe.predictor.eq("natural_zero_recovery_window_share")]
    fe_slopes = fe.coefficient.abs().to_numpy(float)
    fe_slope_low = float(fe_slopes.min())
    fe_slope_high = float(fe_slopes.max())
    pairwise_rows = []
    country_rows = []
    for country, group in values.groupby("country", sort=True):
        group = group.sort_values("weather_year")
        value = group.caes24_minus_lfp4_eur_kw_year
        zero = group.natural_zero_recovery_window_share
        minimum = group.loc[value.idxmin()]
        maximum = group.loc[value.idxmax()]
        value_array = value.to_numpy(float)
        year_array = group.weather_year.to_numpy(int)
        pair_matrix = np.abs(value_array[:, None] - value_array[None, :])
        left, right = np.triu_indices(len(group), 1)
        pair_differences = pair_matrix[left, right]
        adjacent_differences = np.abs(np.diff(value_array))
        for i, j, difference in zip(left, right, pair_differences):
            pairwise_rows.append(
                {
                    "country": country,
                    "weather_year_a": int(year_array[i]),
                    "weather_year_b": int(year_array[j]),
                    "absolute_value_difference_eur_kw_year": float(difference),
                    "reaches_cost_parity": bool(difference >= threshold),
                }
            )
        zero_span = float(zero.max() - zero.min())
        country_rows.append(
            {
                "country": country,
                "weather_years": len(group),
                "value_mean_eur_kw_year": float(value.mean()),
                "value_median_eur_kw_year": float(value.median()),
                "value_min_eur_kw_year": float(value.min()),
                "value_min_year": int(minimum.weather_year),
                "value_max_eur_kw_year": float(value.max()),
                "value_max_year": int(maximum.weather_year),
                "natural_full_span_eur_kw_year": float(value.max() - value.min()),
                "natural_q95_q05_span_eur_kw_year": float(value.quantile(0.95) - value.quantile(0.05)),
                "maximum_absolute_deviation_from_median_eur_kw_year": float((value - value.median()).abs().max()),
                "weather_year_pairs": int(len(pair_differences)),
                "pairwise_difference_median_eur_kw_year": float(np.median(pair_differences)),
                "pairwise_difference_q95_eur_kw_year": float(np.quantile(pair_differences, 0.95)),
                "pairwise_pairs_reaching_cost_parity": int((pair_differences >= threshold).sum()),
                "pairwise_fraction_reaching_cost_parity": float(np.mean(pair_differences >= threshold)),
                "adjacent_weather_year_transitions": int(len(adjacent_differences)),
                "adjacent_transitions_reaching_cost_parity": int((adjacent_differences >= threshold).sum()),
                "adjacent_fraction_reaching_cost_parity": float(np.mean(adjacent_differences >= threshold)),
                "maximum_adjacent_change_eur_kw_year": float(adjacent_differences.max()),
                "weather_years_deviating_from_median_by_parity": int(((value - value.median()).abs() >= threshold).sum()),
                "zero_window_min_share": float(zero.min()),
                "zero_window_max_share": float(zero.max()),
                "zero_window_full_span_percentage_points": float(100.0 * zero_span),
                "fe_projected_displacement_low_eur_kw_year": float(fe_slope_low * zero_span),
                "fe_projected_displacement_high_eur_kw_year": float(fe_slope_high * zero_span),
                "fe_projection_reaches_cost_parity": bool(fe_slope_high * zero_span >= threshold),
                "full_span_reaches_cost_parity": bool(value.max() - value.min() >= threshold),
                "q95_q05_span_reaches_cost_parity": bool(value.quantile(0.95) - value.quantile(0.05) >= threshold),
                "single_year_deviation_reaches_cost_parity": bool((value - value.median()).abs().max() >= threshold),
            }
        )
    country_summary = pd.DataFrame(country_rows)
    country_summary.to_csv(RESULTS / "pecd_35y_country_parity_summary.csv", index=False)
    pairwise = pd.DataFrame(pairwise_rows)
    output_csv_gz(pairwise, RESULTS / "pecd_35y_within_country_year_pairs.csv.gz")

    climate_year = values.groupby("weather_year", as_index=False).agg(
        countries=("country", "nunique"),
        mean_caes24_minus_lfp4_eur_kw_year=("caes24_minus_lfp4_eur_kw_year", "mean"),
        median_caes24_minus_lfp4_eur_kw_year=("caes24_minus_lfp4_eur_kw_year", "median"),
        mean_within_country_value_deviation=("within_country_value_deviation", "mean"),
        mean_zero_recovery_window_share=("natural_zero_recovery_window_share", "mean"),
        mean_longest_event_run_days=("natural_longest_event_run_days", "mean"),
    )
    climate_year.to_csv(RESULTS / "pecd_35y_climate_year_summary.csv", index=False)

    common_effect = climate_year.mean_within_country_value_deviation
    strongest_negative = climate_year.loc[common_effect.idxmin()]
    strongest_positive = climate_year.loc[common_effect.idxmax()]
    zero_window_spans = country_summary.zero_window_full_span_percentage_points
    pair_differences = pairwise.absolute_value_difference_eur_kw_year
    total_adjacent = int(country_summary.adjacent_weather_year_transitions.sum())
    adjacent_reaching = int(country_summary.adjacent_transitions_reaching_cost_parity.sum())
    summary = {
        "status": "pass",
        "experiment": "PECD 2021.3 natural chronology with country-specific 35-year median task and recharge scale",
        "source_record": "10.5281/zenodo.7224854",
        "countries": len(COUNTRIES),
        "weather_years": len(YEARS),
        "country_weather_years": len(values),
        "caes24_fixed_cost_eur_kw_year": CAES24_FIXED_COST_EUR_KW_YEAR,
        "cost_parity_threshold_eur_kw_year": threshold,
        "countries_whose_full_natural_span_reaches_parity": int(country_summary.full_span_reaches_cost_parity.sum()),
        "countries_whose_q95_q05_span_reaches_parity": int(country_summary.q95_q05_span_reaches_cost_parity.sum()),
        "countries_with_single_year_deviation_reaching_parity": int(country_summary.single_year_deviation_reaches_cost_parity.sum()),
        "weather_year_observations_deviating_from_country_median_by_parity": int(country_summary.weather_years_deviating_from_median_by_parity.sum()),
        "weather_year_observations_total": int(len(values)),
        "median_country_full_natural_span_eur_kw_year": float(country_summary.natural_full_span_eur_kw_year.median()),
        "median_country_q95_q05_span_eur_kw_year": float(country_summary.natural_q95_q05_span_eur_kw_year.median()),
        "maximum_country_full_natural_span_eur_kw_year": float(country_summary.natural_full_span_eur_kw_year.max()),
        "maximum_country_q95_q05_span_eur_kw_year": float(country_summary.natural_q95_q05_span_eur_kw_year.max()),
        "median_zero_window_full_span_percentage_points": float(zero_window_spans.median()),
        "minimum_zero_window_full_span_percentage_points": float(zero_window_spans.min()),
        "maximum_zero_window_full_span_percentage_points": float(zero_window_spans.max()),
        "zero_window_fe_required_span_low_percentage_points": float(100.0 * threshold / fe_slope_high),
        "zero_window_fe_required_span_high_percentage_points": float(100.0 * threshold / fe_slope_low),
        "countries_whose_zero_window_fe_projection_reaches_parity": int(country_summary.fe_projection_reaches_cost_parity.sum()),
        "within_country_weather_year_pairs": int(len(pairwise)),
        "within_country_pairs_reaching_parity": int(pairwise.reaches_cost_parity.sum()),
        "within_country_pair_fraction_reaching_parity": float(pairwise.reaches_cost_parity.mean()),
        "within_country_pairwise_difference_median_eur_kw_year": float(pair_differences.median()),
        "within_country_pairwise_difference_q95_eur_kw_year": float(pair_differences.quantile(0.95)),
        "adjacent_weather_year_transitions": total_adjacent,
        "adjacent_transitions_reaching_parity": adjacent_reaching,
        "adjacent_transition_fraction_reaching_parity": float(adjacent_reaching / total_adjacent),
        "countries_with_at_least_one_adjacent_transition_reaching_parity": int(country_summary.adjacent_transitions_reaching_cost_parity.gt(0).sum()),
        "cross_country_mean_weather_year_effect_full_span_eur_kw_year": float(common_effect.max() - common_effect.min()),
        "cross_country_mean_weather_year_effect_q95_q05_span_eur_kw_year": float(common_effect.quantile(0.95) - common_effect.quantile(0.05)),
        "strongest_negative_common_weather_year": int(strongest_negative.weather_year),
        "strongest_negative_common_effect_eur_kw_year": float(strongest_negative.mean_within_country_value_deviation),
        "strongest_positive_common_weather_year": int(strongest_positive.weather_year),
        "strongest_positive_common_effect_eur_kw_year": float(strongest_positive.mean_within_country_value_deviation),
        "files": {
            "relative_values": sha256(RESULTS / "pecd_35y_relative_values.csv"),
            "country_summary": sha256(RESULTS / "pecd_35y_country_parity_summary.csv"),
            "climate_year_summary": sha256(RESULTS / "pecd_35y_climate_year_summary.csv"),
            "year_pairs": sha256(RESULTS / "pecd_35y_within_country_year_pairs.csv.gz"),
        },
    }
    write_json(RESULTS / "pecd_35y_parity_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def parse_countries(value: str | None) -> list[str]:
    if value is None:
        return COUNTRIES
    countries = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = sorted(set(countries) - set(COUNTRIES))
    if unknown:
        raise ValueError(f"Unknown countries: {unknown}")
    return countries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "solve", "summarize", "all"], default="all")
    parser.add_argument("--countries", help="Comma-separated manuscript country codes")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    countries = parse_countries(args.countries)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.stage in {"prepare", "all"}:
        prepare()
    if args.stage in {"solve", "all"}:
        solve(countries, args.workers)
    if args.stage in {"summarize", "all"}:
        if countries != COUNTRIES:
            raise ValueError("Full summarization requires all 15 countries")
        summarize()


if __name__ == "__main__":
    main()
