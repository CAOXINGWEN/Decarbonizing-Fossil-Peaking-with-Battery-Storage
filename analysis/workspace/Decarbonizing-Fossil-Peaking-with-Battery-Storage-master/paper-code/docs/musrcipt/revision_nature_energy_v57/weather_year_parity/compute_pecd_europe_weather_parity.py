#!/usr/bin/env python3
"""Evaluate natural sequencing parity for synchronized European weather years.

The PECD archive contains 35 climate years (1982--2016). Following the
summer-to-summer convention in Kittel et al., these data form 34 complete
July--June operating years. Two synchronized copperplate scopes are replayed:
the 33-country European scope used by Kittel et al. and the 15-country scope
shared with the observed gas-peaking panel in this manuscript.

For each operating year, fixed 2030 demand is netted against the fixed 2030
wind--solar capacity mix. Positive residual load defines the endogenous
firm-power task; negative residual load defines its physically synchronized
recharge opportunity. Annual task and recharge are standardized to the scope's
34-year medians, leaving chronology and event geometry as the weather-varying
inputs. A common 4 h LFP layer plus either a 4 h LFP or 24 h A-CAES tail is
then solved with the registered reduced linear program.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import compute_pecd_35y_parity as country_model


HERE = Path(__file__).resolve().parent
PREPARED = HERE / "prepared_europe"
RESULTS = HERE / "results_europe"

RAW = country_model.RAW
DEMAND_FILE = country_model.DEMAND_FILE
ONSHORE_FILE = country_model.ONSHORE_FILE
OFFSHORE_FILE = country_model.OFFSHORE_FILE
SOLAR_FILE = country_model.SOLAR_FILE
CAPACITY_FILE = country_model.CAPACITY_FILE
COST_FILE = HERE.parent / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
COUNTRY_VALUES_FILE = HERE / "results" / "pecd_35y_relative_values.csv"

CLIMATE_YEARS = np.arange(1982, 2017, dtype=int)
WINDOW_START_YEARS = np.arange(1982, 2016, dtype=int)
HOURS = 8760
DAYS = 365

EUROPE33 = [
    "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "EE", "ES",
    "FI", "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "ME",
    "MK", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK", "UK",
]
EUROPE15 = ["AT", "BE", "BG", "CZ", "DE", "DK", "ES", "FR", "GR", "HR", "IT", "PT", "RO", "SI", "UK"]
SCOPES = {
    "Europe33_copperplate": EUROPE33,
    "Europe15_copperplate": EUROPE15,
}

# Four contiguous meteorological seasons in a July--June operating year.
WINDOW_SEASONS = {
    1: (7, 8, 9),
    2: (10, 11, 12),
    3: (1, 2, 3),
    4: (4, 5, 6),
}
EXPECTED_SEASON_DAYS = {1: 92, 2: 92, 3: 90, 4: 91}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def capacity_table() -> pd.DataFrame:
    frame = pd.read_csv(CAPACITY_FILE)
    frame = frame[frame.scenario.eq("National Estimates 2030")].copy()
    technologies = {
        "Wind Onshore": "onshore",
        "Wind Offshore": "offshore",
        "Solar (Photovoltaic)": "solar",
    }
    frame = frame[frame.tech.isin(technologies)].copy()
    frame["technology"] = frame.tech.map(technologies)
    unique = frame[["country", "technology", "total_country"]].drop_duplicates()
    conflicts = unique.groupby(["country", "technology"]).total_country.nunique()
    if (conflicts > 1).any():
        raise RuntimeError("Conflicting PECD capacity values")
    wide = unique.pivot(index="country", columns="technology", values="total_country").fillna(0.0)
    wide = wide.reindex(EUROPE33).fillna(0.0)
    if (wide < 0.0).any().any():
        raise RuntimeError("Negative PECD capacity")
    return wide


def selected_table(path: Path, value_column: str) -> pd.DataFrame:
    columns = ["country", "year", "month", "day", "hour", value_column]
    table = pq.read_table(path, columns=columns)
    country_mask = pc.is_in(table["country"], value_set=pa.array(EUROPE33, type=pa.string()))
    year_mask = pc.and_(pc.greater_equal(table["year"], 1982), pc.less_equal(table["year"], 2016))
    frame = table.filter(pc.and_(country_mask, year_mask)).to_pandas()
    frame["year"] = frame.year.astype(np.int16)
    frame[["month", "day", "hour"]] = frame[["month", "day", "hour"]].astype(np.int8)
    if not np.isfinite(frame[value_column]).all():
        raise RuntimeError(f"Non-finite values in {path.name}")
    return frame


def aggregate_hourly(
    path: Path,
    value_column: str,
    weights: pd.Series | None = None,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, np.ndarray]]:
    frame = selected_table(path, value_column)
    if weights is None:
        frame["weighted_value"] = frame[value_column].astype(float)
    else:
        mapped = frame.country.map(weights).fillna(0.0).to_numpy(float)
        frame["weighted_value"] = frame[value_column].to_numpy(float) * mapped

    outputs: dict[str, dict[int, np.ndarray]] = {scope: {} for scope in SCOPES}
    months: dict[int, np.ndarray] = {}
    coordinates = ["year", "month", "day", "hour"]
    for scope, countries in SCOPES.items():
        current = frame[frame.country.isin(countries)]
        grouped = current.groupby(coordinates, sort=True, observed=True).weighted_value.sum()
        grouped = grouped.reset_index()
        for year, year_frame in grouped.groupby("year", sort=True):
            year_frame = year_frame.sort_values(["month", "day", "hour"], kind="stable")
            if len(year_frame) != HOURS:
                raise RuntimeError(f"{scope}-{year} has {len(year_frame)} hourly rows")
            outputs[scope][int(year)] = year_frame.weighted_value.to_numpy(float)
            month = year_frame.month.to_numpy(np.int8)
            if int(year) not in months:
                months[int(year)] = month
            elif not np.array_equal(months[int(year)], month):
                raise RuntimeError(f"Calendar mismatch in {year}")
    return outputs, months


def combine_windows(
    arrays: dict[str, dict[int, np.ndarray]],
    months: dict[int, np.ndarray],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {scope: {} for scope in SCOPES}
    window_months: dict[int, np.ndarray] = {}
    for start in WINDOW_START_YEARS:
        left_mask = months[int(start)] >= 7
        right_mask = months[int(start + 1)] <= 6
        month = np.concatenate([months[int(start)][left_mask], months[int(start + 1)][right_mask]])
        if len(month) != HOURS:
            raise RuntimeError(f"Window {start}/{str(start + 1)[-2:]} is not 8,760 h")
        window_months[int(start)] = month
        for scope in SCOPES:
            result[scope][int(start)] = np.concatenate(
                [arrays[scope][int(start)][left_mask], arrays[scope][int(start + 1)][right_mask]]
            )
    return result, window_months


def season_ids(month: np.ndarray) -> np.ndarray:
    result = np.zeros(len(month), dtype=np.int8)
    for season, members in WINDOW_SEASONS.items():
        result[np.isin(month, members)] = season
    if (result == 0).any():
        raise RuntimeError("Unmapped month in July--June window")
    return result


def longest_run(flags: np.ndarray) -> int:
    best = 0
    run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def derive_tasks(
    raw_task: np.ndarray,
    recharge: np.ndarray,
    month: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    hour_of_day = np.tile(np.arange(24, dtype=np.int8), DAYS)
    floor = np.zeros(HOURS, dtype=float)
    for current_month in range(1, 13):
        for hour in range(24):
            mask = (month == current_month) & (hour_of_day == hour)
            floor[mask] = np.quantile(raw_task[mask], country_model.FLOOR_QUANTILE, method="linear")
    base_task = np.maximum(raw_task - floor, 0.0)
    daily = base_task.reshape(DAYS, 24).sum(axis=1)
    daily_month = month.reshape(DAYS, 24)[:, 0]
    seasons = season_ids(daily_month)
    event = np.zeros(DAYS, dtype=bool)
    for season, expected_days in EXPECTED_SEASON_DAYS.items():
        positions = np.flatnonzero(seasons == season)
        if len(positions) != expected_days:
            raise RuntimeError(f"Season {season} has {len(positions)} days, expected {expected_days}")
        target = math.ceil(len(positions) * country_model.EVENT_SHARE)
        order = np.lexsort((positions, -daily[positions]))
        event[positions[order[:target]]] = True
    if int(event.sum()) != 92:
        raise RuntimeError(f"Expected 92 event days, found {event.sum()}")
    tail_task = base_task * np.repeat(event, 24)

    recovery_h: list[float] = []
    for season in sorted(WINDOW_SEASONS):
        positions = np.flatnonzero(event & (seasons == season))
        if len(positions) > 1:
            recovery_h.extend(((np.diff(positions) - 1) * 24.0).tolist())
    recovery = np.asarray(recovery_h, dtype=float)
    return base_task, tail_task, event, {
        "event_days": int(event.sum()),
        "natural_longest_event_run_days": longest_run(event),
        "natural_zero_recovery_window_share": float(np.mean(recovery == 0.0)),
        "natural_median_recovery_window_h": float(np.median(recovery)),
        "annual_base_task_mwh": float(base_task.sum()),
        "annual_tail_task_mwh": float(tail_task.sum()),
        "annual_recharge_mwh": float(recharge.sum()),
        "recharge_to_base_task_ratio": float(recharge.sum() / max(base_task.sum(), 1.0e-12)),
    }


def prepare() -> None:
    PREPARED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    capacities = capacity_table()

    demand_calendar, months = aggregate_hourly(DEMAND_FILE, "dem_MW")
    demand, window_months = combine_windows(demand_calendar, months)
    del demand_calendar

    generation_calendar: dict[str, dict[int, np.ndarray]] = {
        scope: {int(year): np.zeros(HOURS) for year in CLIMATE_YEARS} for scope in SCOPES
    }
    for technology, path in [
        ("onshore", ONSHORE_FILE),
        ("offshore", OFFSHORE_FILE),
        ("solar", SOLAR_FILE),
    ]:
        aggregate, technology_months = aggregate_hourly(path, "cf", capacities[technology])
        for year in CLIMATE_YEARS:
            if not np.array_equal(months[int(year)], technology_months[int(year)]):
                raise RuntimeError(f"{technology} calendar mismatch in {year}")
        for scope in SCOPES:
            for year in CLIMATE_YEARS:
                generation_calendar[scope][int(year)] += aggregate[scope][int(year)]
        del aggregate
    generation, generation_window_months = combine_windows(generation_calendar, months)
    for year in WINDOW_START_YEARS:
        if not np.array_equal(window_months[int(year)], generation_window_months[int(year)]):
            raise RuntimeError(f"Window calendar mismatch in {year}")

    summary_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        raw: dict[int, dict[str, Any]] = {}
        for start in WINDOW_START_YEARS:
            load = demand[scope][int(start)]
            vre_unscaled = generation[scope][int(start)]
            alpha = float(load.sum() / vre_unscaled.sum())
            residual = load - alpha * vre_unscaled
            residual[np.abs(residual) < 1.0e-10] = 0.0
            raw_task = np.maximum(residual, 0.0)
            raw_recharge = np.maximum(-residual, 0.0)
            if np.any((raw_task > 0.0) & (raw_recharge > 0.0)):
                raise RuntimeError(f"Overlapping task and recharge in {scope}-{start}")
            raw[int(start)] = {
                "task": raw_task,
                "recharge": raw_recharge,
                "demand_mwh": float(load.sum()),
                "vre_unscaled_mwh": float(vre_unscaled.sum()),
                "alpha": alpha,
            }

        target_task = float(np.median([row["task"].sum() for row in raw.values()]))
        target_recharge = float(np.median([row["recharge"].sum() for row in raw.values()]))
        fixed_power = country_model.FORCED_POWER_SHARE * target_task / HOURS
        if min(target_task, target_recharge, fixed_power) <= 0.0:
            raise RuntimeError(f"Invalid standardized scale for {scope}")

        base_stack = []
        tail_stack = []
        recharge_stack = []
        event_stack = []
        for start in WINDOW_START_YEARS:
            current = raw[int(start)]
            task_scale = target_task / current["task"].sum()
            recharge_scale = target_recharge / current["recharge"].sum()
            standardized_task = current["task"] * task_scale
            standardized_recharge = current["recharge"] * recharge_scale
            base_task, tail_task, event, metrics = derive_tasks(
                standardized_task,
                standardized_recharge,
                window_months[int(start)],
            )
            base_stack.append(base_task)
            tail_stack.append(tail_task)
            recharge_stack.append(standardized_recharge)
            event_stack.append(event.astype(np.int8))
            row = {
                "scope": scope,
                "countries": len(SCOPES[scope]),
                "window_start_year": int(start),
                "weather_window": f"{start}/{str(start + 1)[-2:]}",
                "system_definition": "synchronized_copperplate_2030_mix_annual_vre_equals_demand",
                "standardization": "scope_34_window_median_raw_task_and_recharge",
                "demand_mwh": current["demand_mwh"],
                "unscaled_vre_mwh": current["vre_unscaled_mwh"],
                "vre_scaling_alpha": current["alpha"],
                "raw_task_mwh": float(current["task"].sum()),
                "raw_recharge_mwh": float(current["recharge"].sum()),
                "task_standardization_factor": task_scale,
                "recharge_standardization_factor": recharge_scale,
                "target_raw_task_mwh": target_task,
                "target_raw_recharge_mwh": target_recharge,
                "fixed_common_power_mw": fixed_power,
                **metrics,
            }
            summary_rows.append(row)
            geometry_rows.append({
                "scope": scope,
                "window_start_year": int(start),
                "weather_window": row["weather_window"],
                "fixed_common_power_mw": fixed_power,
                **metrics,
            })
        np.savez_compressed(
            PREPARED / f"{scope}_pecd_1982_2016_jul_jun_standardized.npz",
            window_start_years=WINDOW_START_YEARS,
            base_task_mw=np.asarray(base_stack),
            tail_task_mw=np.asarray(tail_stack),
            recharge_mw=np.asarray(recharge_stack),
            event_day=np.asarray(event_stack),
            fixed_common_power_mw=np.asarray([fixed_power]),
        )
        print(f"prepared {scope}: 34 July--June windows", flush=True)

    summary = pd.DataFrame(summary_rows)
    geometry = pd.DataFrame(geometry_rows)
    summary.to_csv(RESULTS / "pecd_europe_34_window_prepared_summary.csv", index=False)
    geometry.to_csv(RESULTS / "pecd_europe_34_window_event_geometry.csv", index=False)
    capacities.reset_index().to_csv(RESULTS / "pecd_2030_vre_capacities_europe33.csv", index=False)
    qa = {
        "status": "pass",
        "source_climate_years": 35,
        "complete_july_june_windows": 34,
        "scopes": {scope: len(countries) for scope, countries in SCOPES.items()},
        "scope_weather_windows": int(len(summary)),
        "expected_scope_weather_windows": len(SCOPES) * len(WINDOW_START_YEARS),
        "all_event_day_counts_equal_92": bool(geometry.event_days.eq(92).all()),
        "all_fixed_power_positive": bool(summary.fixed_common_power_mw.gt(0.0).all()),
        "prepared_files": {
            path.name: country_model.sha256(path) for path in sorted(PREPARED.glob("*.npz"))
        },
    }
    if len(summary) != qa["expected_scope_weather_windows"] or not qa["all_event_day_counts_equal_92"]:
        qa["status"] = "fail"
    write_json(RESULTS / "pecd_europe_34_window_preparation_qa.json", qa)
    if qa["status"] != "pass":
        raise RuntimeError("European aggregate preparation QA failed")


def solve_scope(scope: str) -> tuple[str, list[dict[str, Any]]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    base, caes, scenario, _ = country_model.load_parameters()
    models = {
        base.config_id: country_model.build_reduced_model(base, base, scenario),
        caes.config_id: country_model.build_reduced_model(base, caes, scenario),
    }
    data = np.load(PREPARED / f"{scope}_pecd_1982_2016_jul_jun_standardized.npz")
    starts = data["window_start_years"].astype(int)
    power = float(data["fixed_common_power_mw"][0])
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        task = data["base_task_mw"][index].astype(float, copy=False)
        tail_task = data["tail_task_mw"][index].astype(float, copy=False)
        recharge = data["recharge_mw"][index].astype(float, copy=False)
        for config_id in [base.config_id, caes.config_id]:
            answer = country_model.solve_arm(models[config_id], task, tail_task, recharge, power)
            rows.append({
                "scope": scope,
                "window_start_year": int(start),
                "weather_window": f"{start}/{str(start + 1)[-2:]}",
                "chronology": "natural_pecd_july_june",
                "annual_scale": "scope_34_window_median_standardized",
                "scenario_id": 20,
                "energy_value_eur_mwh": scenario.energy_value_eur_mwh,
                "firm_value_eur_kw_year": scenario.firm_value_eur_kw_year,
                "charge_price_eur_mwh": scenario.charge_price_eur_mwh,
                **answer,
            })
        print(f"{scope} solved {index + 1}/34", flush=True)
    return scope, rows


def solve(workers: int) -> None:
    missing = [
        scope for scope in SCOPES
        if not (PREPARED / f"{scope}_pecd_1982_2016_jul_jun_standardized.npz").exists()
    ]
    if missing:
        raise RuntimeError(f"Prepared inputs missing for {missing}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    completed: dict[str, pd.DataFrame] = {}
    pending: list[str] = []
    for scope in SCOPES:
        checkpoint = RESULTS / f"pecd_europe_34_window_arm_results_{scope}.csv.gz"
        if checkpoint.exists():
            frame = pd.read_csv(checkpoint)
            if len(frame) == 68 and frame.solver_status_name.eq("optimal").all():
                completed[scope] = frame
                print(f"reused verified checkpoint for {scope}", flush=True)
                continue
        pending.append(scope)
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(solve_scope, scope): scope for scope in pending}
            for future in as_completed(futures):
                scope, rows = future.result()
                frame = pd.DataFrame(rows)
                country_model.output_csv_gz(
                    frame,
                    RESULTS / f"pecd_europe_34_window_arm_results_{scope}.csv.gz",
                )
                completed[scope] = frame
                print(f"completed {scope}: {len(frame)} arm-windows", flush=True)
    output = pd.concat([completed[scope] for scope in SCOPES], ignore_index=True)
    output = output.sort_values(["scope", "window_start_year", "config_id"]).reset_index(drop=True)
    output_path = RESULTS / "pecd_europe_34_window_arm_results.csv.gz"
    country_model.output_csv_gz(output, output_path)
    qa = {
        "status": "pass",
        "rows": int(len(output)),
        "expected_rows": len(SCOPES) * len(WINDOW_START_YEARS) * 2,
        "all_optimal": bool(output.solver_status_name.eq("optimal").all()),
        "maximum_equality_residual": float(output.maximum_equality_residual.max()),
        "maximum_inequality_violation": float(output.maximum_inequality_violation.max()),
        "maximum_bound_violation": float(output.maximum_bound_violation.max()),
        "minimum_unserved_mw": float(output.minimum_unserved_mw.min()),
        "total_solver_runtime_seconds": float(output.solver_runtime_seconds.sum()),
        "parallel_workers": min(workers, len(SCOPES)),
        "output_sha256": country_model.sha256(output_path),
    }
    if (
        len(output) != qa["expected_rows"]
        or not qa["all_optimal"]
        or qa["maximum_equality_residual"] > 1.0e-5
        or qa["maximum_inequality_violation"] > 1.0e-5
        or qa["minimum_unserved_mw"] < -1.0e-5
    ):
        qa["status"] = "fail"
    write_json(RESULTS / "pecd_europe_34_window_optimization_qa.json", qa)
    if qa["status"] != "pass":
        raise RuntimeError("European aggregate optimization QA failed")


def cost_path() -> pd.DataFrame:
    frame = pd.read_csv(COST_FILE)
    selected = frame[frame.config_id.isin(["li_lfp_4h", "caes_24h"])].copy()
    wide = selected.pivot(
        index="learning_year",
        columns="config_id",
        values="annual_fixed_cost_eur_source_aware_per_kw_discharge_year",
    ).reset_index()
    wide["lfp4_minus_caes24_fixed_eur_kw_year"] = wide.li_lfp_4h - wide.caes_24h
    anchor = float(wide.loc[wide.learning_year.eq(2025), "lfp4_minus_caes24_fixed_eur_kw_year"].iloc[0])
    wide["relative_value_shift_from_2025_eur_kw_year"] = wide.lfp4_minus_caes24_fixed_eur_kw_year - anchor
    return wide.sort_values("learning_year").reset_index(drop=True)


def interval_risk_rows(group: pd.DataFrame, unit_type: str, unit_id: str) -> list[dict[str, Any]]:
    group = group.sort_values("sequence_index")
    value = group.value_2025_eur_kw_year.to_numpy(float)
    sequence = group.sequence_index.to_numpy(int)
    labels = group.sequence_label.astype(str).to_numpy()
    path = cost_path().set_index("learning_year")
    intervals = [(2025, 2030, 5), (2030, 2035, 5), (2025, 2035, 10)]
    rows = []
    left, right = np.triu_indices(len(value), 1)
    pair_abs = np.abs(value[right] - value[left])
    for start_cost, end_cost, horizon in intervals:
        shift = float(
            path.loc[end_cost, "relative_value_shift_from_2025_eur_kw_year"]
            - path.loc[start_cost, "relative_value_shift_from_2025_eur_kw_year"]
        )
        matched = value[horizon:] - value[:-horizon]
        adverse = matched <= -shift + 1.0e-12
        beneficial = matched >= shift - 1.0e-12
        rows.append({
            "unit_type": unit_type,
            "unit_id": unit_id,
            "start_cost_year": start_cost,
            "end_cost_year": end_cost,
            "weather_horizon_years": horizon,
            "relative_cost_improvement_eur_kw_year": shift,
            "full_natural_span_eur_kw_year": float(value.max() - value.min()),
            "q95_q05_natural_span_eur_kw_year": float(np.quantile(value, 0.95) - np.quantile(value, 0.05)),
            "all_weather_pairs": int(len(pair_abs)),
            "all_pairs_with_absolute_shift_reaching_cost_improvement": int((pair_abs >= shift - 1.0e-12).sum()),
            "all_pair_fraction_absolute_shift_reaching_cost_improvement": float(np.mean(pair_abs >= shift - 1.0e-12)),
            "horizon_matched_transitions": int(len(matched)),
            "horizon_matched_adverse_transitions_erasing_learning": int(adverse.sum()),
            "horizon_matched_adverse_fraction_erasing_learning": float(adverse.mean()),
            "horizon_matched_beneficial_transitions_exceeding_learning": int(beneficial.sum()),
            "maximum_horizon_matched_deterioration_eur_kw_year": float(-matched.min()),
            "worst_start_sequence": labels[int(np.argmin(matched))],
            "worst_end_sequence": labels[int(np.argmin(matched)) + horizon],
            "worst_directional_change_eur_kw_year": float(matched.min()),
            "sequence_index_start": int(sequence[0]),
            "sequence_index_end": int(sequence[-1]),
        })
    return rows


def summarize_country_risk() -> tuple[pd.DataFrame, pd.DataFrame]:
    country = pd.read_csv(COUNTRY_VALUES_FILE)
    country = country.rename(columns={
        "weather_year": "sequence_index",
        "caes24_minus_lfp4_eur_kw_year": "value_2025_eur_kw_year",
    })
    country["sequence_label"] = country.sequence_index.astype(int).astype(str)
    all_rows = []
    for country_id, group in country.groupby("country", sort=True):
        all_rows.extend(interval_risk_rows(group, "country_calendar_year", str(country_id)))
    risk = pd.DataFrame(all_rows)
    pooled_rows = []
    for (start_cost, end_cost), group in risk.groupby(["start_cost_year", "end_cost_year"], sort=True):
        pooled_rows.append({
            "start_cost_year": int(start_cost),
            "end_cost_year": int(end_cost),
            "countries": int(group.unit_id.nunique()),
            "countries_with_at_least_one_horizon_matched_adverse_transition": int(
                group.horizon_matched_adverse_transitions_erasing_learning.gt(0).sum()
            ),
            "horizon_matched_transitions": int(group.horizon_matched_transitions.sum()),
            "horizon_matched_adverse_transitions_erasing_learning": int(
                group.horizon_matched_adverse_transitions_erasing_learning.sum()
            ),
            "horizon_matched_adverse_fraction_erasing_learning": float(
                group.horizon_matched_adverse_transitions_erasing_learning.sum()
                / group.horizon_matched_transitions.sum()
            ),
            "countries_whose_full_span_reaches_cost_improvement": int(
                (group.full_natural_span_eur_kw_year >= group.relative_cost_improvement_eur_kw_year - 1.0e-12).sum()
            ),
            "countries_whose_q95_q05_span_reaches_cost_improvement": int(
                (group.q95_q05_natural_span_eur_kw_year >= group.relative_cost_improvement_eur_kw_year - 1.0e-12).sum()
            ),
        })
    return risk, pd.DataFrame(pooled_rows)


def summarize() -> None:
    arm_path = RESULTS / "pecd_europe_34_window_arm_results.csv.gz"
    if not arm_path.exists():
        raise RuntimeError("Aggregate arm results missing; run solve first")
    arms = pd.read_csv(arm_path)
    geometry = pd.read_csv(RESULTS / "pecd_europe_34_window_event_geometry.csv")
    wide = arms.pivot(
        index=["scope", "window_start_year", "weather_window", "fixed_common_power_mw"],
        columns="config_id",
        values=["system_cost_eur", "tail_equivalent_full_cycles", "portfolio_efc_mw", "unserved_eens_mwh"],
    )
    wide.columns = [f"{metric}_{config}" for metric, config in wide.columns]
    values = wide.reset_index()
    values["value_2025_eur_kw_year"] = (
        values.system_cost_eur_li_lfp_4h - values.system_cost_eur_caes_24h
    ) / (1000.0 * values.fixed_common_power_mw)
    values = values.merge(
        geometry,
        on=["scope", "window_start_year", "weather_window", "fixed_common_power_mw"],
        validate="one_to_one",
    )
    values["sequence_index"] = values.window_start_year.astype(int)
    values["sequence_label"] = values.weather_window
    values.to_csv(RESULTS / "pecd_europe_34_window_relative_values_2025.csv", index=False)

    costs = cost_path()
    costs.to_csv(RESULTS / "pecd_focal_cost_path_2025_2035.csv", index=False)
    long_rows = []
    for _, row in costs.iterrows():
        current = values.copy()
        current["cost_year"] = int(row.learning_year)
        current["lfp4_fixed_cost_eur_kw_year"] = float(row.li_lfp_4h)
        current["caes24_fixed_cost_eur_kw_year"] = float(row.caes_24h)
        current["relative_value_shift_from_2025_eur_kw_year"] = float(
            row.relative_value_shift_from_2025_eur_kw_year
        )
        current["caes24_minus_lfp4_eur_kw_year"] = (
            current.value_2025_eur_kw_year + row.relative_value_shift_from_2025_eur_kw_year
        )
        current["caes24_outperforms_lfp4"] = current.caes24_minus_lfp4_eur_kw_year >= 0.0
        long_rows.append(current)
    cost_values = pd.concat(long_rows, ignore_index=True)
    cost_values.to_csv(RESULTS / "pecd_europe_34_window_relative_values_by_cost_year.csv", index=False)

    scope_risk_rows = []
    for scope, group in values.groupby("scope", sort=True):
        scope_risk_rows.extend(interval_risk_rows(group, "synchronized_copperplate", str(scope)))
    scope_risk = pd.DataFrame(scope_risk_rows)
    scope_risk.to_csv(RESULTS / "pecd_europe_34_window_directional_risk.csv", index=False)

    country_risk, country_pooled = summarize_country_risk()
    country_risk.to_csv(RESULTS / "pecd_country_35y_directional_risk.csv", index=False)
    country_pooled.to_csv(RESULTS / "pecd_country_35y_directional_risk_pooled.csv", index=False)

    tech_summary = cost_values.groupby(["scope", "cost_year"], as_index=False).agg(
        weather_windows=("weather_window", "size"),
        value_min_eur_kw_year=("caes24_minus_lfp4_eur_kw_year", "min"),
        value_median_eur_kw_year=("caes24_minus_lfp4_eur_kw_year", "median"),
        value_max_eur_kw_year=("caes24_minus_lfp4_eur_kw_year", "max"),
        weather_windows_caes24_outperforms_lfp4=("caes24_outperforms_lfp4", "sum"),
    )
    tech_summary.to_csv(RESULTS / "pecd_europe_34_window_cost_year_summary.csv", index=False)

    main_interval = scope_risk[
        scope_risk.start_cost_year.eq(2025) & scope_risk.end_cost_year.eq(2035)
    ].set_index("unit_id")
    country_main = country_pooled[
        country_pooled.start_cost_year.eq(2025) & country_pooled.end_cost_year.eq(2035)
    ].iloc[0]
    summary = {
        "status": "pass",
        "source": "PECD 2021.3 corrected release, 35 climate years (1982-2016)",
        "operating_year_convention": "34 complete July-June windows (1982/83-2015/16)",
        "cost_years": costs.learning_year.astype(int).tolist(),
        "relative_cost_improvement_2025_2030_eur_kw_year": float(
            costs.set_index("learning_year").loc[2030, "relative_value_shift_from_2025_eur_kw_year"]
        ),
        "relative_cost_improvement_2030_2035_eur_kw_year": float(
            costs.set_index("learning_year").loc[2035, "relative_value_shift_from_2025_eur_kw_year"]
            - costs.set_index("learning_year").loc[2030, "relative_value_shift_from_2025_eur_kw_year"]
        ),
        "relative_cost_improvement_2025_2035_eur_kw_year": float(
            costs.set_index("learning_year").loc[2035, "relative_value_shift_from_2025_eur_kw_year"]
        ),
        "europe33_2025_2035": main_interval.loc["Europe33_copperplate"].to_dict(),
        "europe15_copperplate_2025_2035": main_interval.loc["Europe15_copperplate"].to_dict(),
        "europe15_country_2025_2035": country_main.to_dict(),
        "all_aggregate_weather_cost_cells_caes24_outperforms_lfp4": int(
            cost_values.caes24_outperforms_lfp4.sum()
        ),
        "aggregate_weather_cost_cells": int(len(cost_values)),
        "files": {
            "relative_values": country_model.sha256(
                RESULTS / "pecd_europe_34_window_relative_values_by_cost_year.csv"
            ),
            "scope_directional_risk": country_model.sha256(
                RESULTS / "pecd_europe_34_window_directional_risk.csv"
            ),
            "country_directional_risk": country_model.sha256(
                RESULTS / "pecd_country_35y_directional_risk.csv"
            ),
        },
    }
    write_json(RESULTS / "pecd_europe_weather_parity_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "solve", "summarize", "all"], default="all")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.stage in {"prepare", "all"}:
        prepare()
    if args.stage in {"solve", "all"}:
        solve(args.workers)
    if args.stage in {"summarize", "all"}:
        summarize()


if __name__ == "__main__":
    main()
