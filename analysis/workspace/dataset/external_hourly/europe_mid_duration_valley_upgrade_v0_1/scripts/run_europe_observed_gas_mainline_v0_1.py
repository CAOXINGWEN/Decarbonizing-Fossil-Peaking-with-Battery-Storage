#!/usr/bin/env python3
"""European observed-gas storage-duration experiment.

The experiment uses the frozen hourly aggregate fossil-gas generation event
task and the R2 potential renewable-surplus recharge profile.  Intact 24-hour
task/recharge blocks are reordered together into dispersed and clustered
chronologies.  A deterministic cyclic-SOC dispatch forbids charging in any
hour with a positive gas task, thereby preventing same-hour charge/discharge
pass-through.

The primary screen fixes storage discharge power at 1% of mean observed gas
generation and evaluates the complete 53-configuration catalogue (4--720 h).
Physical outputs, EENS-equivalent firm capacity and direct annual system-cost
components are written for the natural, dispersed and clustered chronologies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd
from numba import njit


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_PANEL = (
    ROOT / "processed" / "observed_gas_event_task_definitions_v0_2.csv.gz"
)
DEFAULT_CATALOGUE = (
    ROOT
    / "model_adapter"
    / "energy_charts_2015_2024_representative_pilot"
    / "configuration_catalog_53.csv"
)
DEFAULT_OUTPUT = ROOT / "model_results" / "europe_observed_gas_mainline_v0_1"

TASK_COLUMN = "task_c3_q75_mw"
RECHARGE_COLUMN = "r2_potential_recharge_mw"
CLUSTER_DAYS = 4
SOC_MIN = 0.05
SOC_MAX = 0.95
FORCED_POWER_SHARE = 0.01
GAS_VARIABLE_CNY_MWH = 529.5970113779516
GAS_FIRM_CNY_KW_YEAR = 842.6943979480799
PRIMARY_CHARGE_PRICE_CNY_MWH = 50.0
VALUE_ENERGY_GRID = (300.0, GAS_VARIABLE_CNY_MWH, 800.0)
VALUE_FIRM_GRID = (0.0, 400.0, GAS_FIRM_CNY_KW_YEAR, 1200.0)
VALUE_CHARGE_GRID = (0.0, 50.0, 100.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            json_ready(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def interleave(extreme: list[int], normal: list[int]) -> list[int]:
    slots: list[list[int]] = [[] for _ in range(len(normal) + 1)]
    if extreme:
        positions = np.linspace(1, max(1, len(normal) - 1), len(extreme))
        for day, position in zip(extreme, np.rint(positions).astype(int)):
            slots[int(np.clip(position, 0, len(normal)))].append(day)
    order: list[int] = []
    for index, day in enumerate(normal):
        order.extend(slots[index])
        order.append(day)
    order.extend(slots[-1])
    return order


def interleave_clusters(
    extreme: list[int], normal: list[int], target_run: int
) -> list[int]:
    clusters = [
        extreme[index : index + target_run]
        for index in range(0, len(extreme), target_run)
    ]
    slots: list[list[int]] = [[] for _ in range(len(normal) + 1)]
    if clusters:
        positions = np.linspace(1, max(1, len(normal) - 1), len(clusters))
        for cluster, position in zip(clusters, np.rint(positions).astype(int)):
            slots[int(np.clip(position, 0, len(normal)))].extend(cluster)
    order: list[int] = []
    for index, day in enumerate(normal):
        order.extend(slots[index])
        order.append(day)
    order.extend(slots[-1])
    return order


def day_orders(frame: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, Any]]:
    daily = (
        frame.groupby("source_day_id", sort=True)
        .agg(
            segment=("season_segment", "first"),
            event=("event_flag_q75", "max"),
            task_mwh=(TASK_COLUMN, "sum"),
        )
        .reset_index()
    )
    if len(daily) != 365:
        raise RuntimeError("Each country-year must contain 365 complete days")
    orders: dict[str, list[int]] = {"natural": list(range(365))}
    dispersed: list[int] = []
    clustered: list[int] = []
    for segment in sorted(daily["segment"].unique()):
        subset = daily[daily["segment"].eq(segment)]
        extreme = [
            int(value)
            for value in subset.loc[subset["event"].astype(bool), "source_day_id"]
        ]
        normal = [
            int(value)
            for value in subset.loc[~subset["event"].astype(bool), "source_day_id"]
        ]
        dispersed.extend(interleave(extreme, normal))
        clustered.extend(interleave_clusters(extreme, normal, CLUSTER_DAYS))
    orders["dispersed"] = dispersed
    orders["clustered"] = clustered
    for label, order in orders.items():
        if len(order) != 365 or sorted(order) != list(range(365)):
            raise RuntimeError(f"Invalid intact-day order for {label}")

    def adjacency(order: list[int]) -> tuple[int, int]:
        event_days = set(
            int(value)
            for value in daily.loc[daily["event"].astype(bool), "source_day_id"]
        )
        adjacent = 0
        longest = 0
        run = 0
        segment_by_day = daily.set_index("source_day_id")["segment"].to_dict()
        previous_segment: int | None = None
        previous_event = False
        for day in order:
            segment = int(segment_by_day[day])
            flag = day in event_days
            if segment != previous_segment:
                run = 0
                previous_event = False
            if flag:
                run += 1
                longest = max(longest, run)
                if previous_event:
                    adjacent += 1
            else:
                run = 0
            previous_event = flag
            previous_segment = segment
        return adjacent, longest

    dispersed_adjacent, dispersed_longest = adjacency(dispersed)
    clustered_adjacent, clustered_longest = adjacency(clustered)
    return orders, {
        "event_days": int(daily["event"].sum()),
        "dispersed_adjacent_event_pairs": dispersed_adjacent,
        "clustered_adjacent_event_pairs": clustered_adjacent,
        "dispersed_longest_event_run_days": dispersed_longest,
        "clustered_longest_event_run_days": clustered_longest,
    }


def reorder_days(values: np.ndarray, order: list[int]) -> np.ndarray:
    return values.reshape(365, 24)[np.asarray(order, dtype=int)].reshape(-1)


@njit(cache=True)
def annual_dispatch(
    task: np.ndarray,
    recharge: np.ndarray,
    duration_h: float,
    rte: float,
    self_discharge_pct_day: float,
    charge_power_ratio: float,
    pdis_mw: float,
) -> tuple[float, float, float, float, int, int, float, float, float, int]:
    energy_mwh = duration_h * pdis_mw
    e_min = SOC_MIN * energy_mwh
    e_max = SOC_MAX * energy_mwh
    eta = np.sqrt(rte)
    pch_mw = charge_power_ratio * pdis_mw
    retention = (1.0 - self_discharge_pct_day / 100.0) ** (1.0 / 24.0)
    soc = 0.5 * energy_mwh
    iterations = 0

    for repetition in range(200):
        start = soc
        for hour in range(task.size):
            soc = max(e_min, soc * retention)
            if task[hour] > 1.0e-12:
                available_output = max(0.0, (soc - e_min) * eta)
                discharge = min(pdis_mw, task[hour], available_output)
                soc -= discharge / eta
            elif recharge[hour] > 1.0e-12:
                room_input = max(0.0, (e_max - soc) / eta)
                charge = min(pch_mw, recharge[hour], room_input)
                soc += charge * eta
        iterations = repetition + 1
        if abs(soc - start) <= max(1.0e-9, 1.0e-6 * energy_mwh):
            break

    cyclic_start = soc
    charge_total = 0.0
    discharge_total = 0.0
    unserved_total = 0.0
    spill_total = 0.0
    charge_saturated = 0
    discharge_saturated = 0
    soc_min_seen = soc
    soc_max_seen = soc
    overlap_hours = 0
    for hour in range(task.size):
        soc = max(e_min, soc * retention)
        if task[hour] > 1.0e-12:
            if recharge[hour] > 1.0e-12:
                overlap_hours += 1
            available_output = max(0.0, (soc - e_min) * eta)
            discharge = min(pdis_mw, task[hour], available_output)
            soc -= discharge / eta
            discharge_total += discharge
            unserved_total += task[hour] - discharge
            spill_total += recharge[hour]
            if discharge >= pdis_mw - 1.0e-9:
                discharge_saturated += 1
        elif recharge[hour] > 1.0e-12:
            room_input = max(0.0, (e_max - soc) / eta)
            charge = min(pch_mw, recharge[hour], room_input)
            soc += charge * eta
            charge_total += charge
            spill_total += recharge[hour] - charge
            if charge >= pch_mw - 1.0e-9:
                charge_saturated += 1
        soc_min_seen = min(soc_min_seen, soc)
        soc_max_seen = max(soc_max_seen, soc)
    return (
        charge_total,
        discharge_total,
        unserved_total,
        spill_total,
        charge_saturated,
        discharge_saturated,
        soc_min_seen / energy_mwh,
        soc_max_seen / energy_mwh,
        soc - cyclic_start,
        overlap_hours,
    )


def equivalent_firm_capacity(task: np.ndarray, target_eens: float) -> float:
    if target_eens >= float(task.sum()) - 1.0e-10:
        return 0.0
    low = 0.0
    high = float(task.max())
    for _ in range(70):
        midpoint = 0.5 * (low + high)
        eens = float(np.maximum(task - midpoint, 0.0).sum())
        if eens > target_eens:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def run_configuration(
    task: np.ndarray,
    recharge: np.ndarray,
    config: pd.Series,
    pdis_mw: float,
) -> dict[str, float | int]:
    (
        charge,
        discharge,
        unserved,
        spill,
        charge_sat,
        discharge_sat,
        soc_min,
        soc_max,
        cyclic_residual,
        overlap_hours,
    ) = annual_dispatch(
        task,
        recharge,
        float(config["duration_h"]),
        float(config["rte"]),
        float(config["self_discharge_pct_day"]),
        float(config["charge_power_ratio"]),
        pdis_mw,
    )
    task_total = float(task.sum())
    efc_mw = equivalent_firm_capacity(task, unserved)
    fixed_cost_per_kw = (
        float(config["fixed_cost_cny_per_kwh_year"])
        * float(config["duration_h"])
    )
    variable_value_per_kw = (
        discharge * GAS_VARIABLE_CNY_MWH / (pdis_mw * 1000.0)
    )
    storage_vom_per_kw = (
        discharge
        * float(config["variable_om_cny_mwh"])
        / (pdis_mw * 1000.0)
    )
    charge_cost_per_kw = (
        charge * PRIMARY_CHARGE_PRICE_CNY_MWH / (pdis_mw * 1000.0)
    )
    turnover = (
        variable_value_per_kw
        - fixed_cost_per_kw
        - storage_vom_per_kw
        - charge_cost_per_kw
    )
    capacity_value = efc_mw / pdis_mw * GAS_FIRM_CNY_KW_YEAR
    remaining = np.maximum(task, 0.0)
    baseline_peak = float(remaining.max())
    # Capacity avoided at the actual peak is reported separately from EFC.
    # It is not used in the joint value ledger.
    return {
        "pdis_mw": pdis_mw,
        "energy_capacity_mwh": float(config["duration_h"]) * pdis_mw,
        "task_mwh": task_total,
        "recharge_available_mwh": float(recharge.sum()),
        "charge_mwh": charge,
        "discharge_mwh": discharge,
        "unserved_mwh": unserved,
        "spill_mwh": spill,
        "served_share": discharge / task_total if task_total > 0.0 else 0.0,
        "equivalent_full_cycles": (
            discharge / (float(config["duration_h"]) * pdis_mw)
        ),
        "efc_mw": efc_mw,
        "efc_ratio": efc_mw / pdis_mw,
        "baseline_task_peak_mw": baseline_peak,
        "charge_saturation_hours": charge_sat,
        "discharge_saturation_hours": discharge_sat,
        "soc_min_fraction": soc_min,
        "soc_max_fraction": soc_max,
        "cyclic_soc_residual_mwh": cyclic_residual,
        "task_recharge_overlap_hours": overlap_hours,
        "annual_fixed_cost_cny_per_kw_year": fixed_cost_per_kw,
        "storage_vom_cny_per_kw_year": storage_vom_per_kw,
        "charge_opportunity_cost_cny_per_kw_year": charge_cost_per_kw,
        "energy_displacement_value_cny_per_kw_year": variable_value_per_kw,
        "turnover_net_benefit_cny_per_kw_year": turnover,
        "firm_capacity_value_cny_per_kw_year": capacity_value,
        "joint_net_benefit_cny_per_kw_year": turnover + capacity_value,
    }


def country_cluster_interval(
    frame: pd.DataFrame,
    value: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    country_means = frame.groupby("country", sort=True)[value].mean().to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(country_means), size=(draws, len(country_means))
    )
    samples = country_means[indices].mean(axis=1)
    return (
        float(country_means.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def summarize_paired(paired: pd.DataFrame, draws: int) -> pd.DataFrame:
    metrics = (
        "delta_equivalent_full_cycles",
        "retained_equivalent_full_cycles",
        "delta_served_share_pp",
        "delta_efc_ratio",
        "delta_joint_net_benefit_cny_per_kw_year",
    )
    rows: list[dict[str, Any]] = []
    for config_id, group in paired.groupby("config_id", sort=False):
        base = group.iloc[0]
        row: dict[str, Any] = {
            "config_id": config_id,
            "tech_id": base["tech_id"],
            "technology_en": base["technology_en"],
            "duration_h": float(base["duration_h"]),
            "country_years": len(group),
            "countries": group["country"].nunique(),
        }
        for index, metric in enumerate(metrics):
            mean, low, high = country_cluster_interval(
                group, metric, draws, 20260727 + index
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["duration_h", "tech_id"], kind="stable"
    )


def value_gap_scan(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    physical = results[
        results["chronology"].isin(["dispersed", "clustered"])
        & results["duration_h"].le(168.0)
    ].copy()
    rows: list[pd.DataFrame] = []
    scenarios: list[dict[str, Any]] = []
    scenario_id = 0
    for energy_value in VALUE_ENERGY_GRID:
        for firm_value in VALUE_FIRM_GRID:
            for charge_price in VALUE_CHARGE_GRID:
                scenario_id += 1
                current = physical[
                    [
                        "country",
                        "year",
                        "chronology",
                        "config_id",
                        "tech_id",
                        "duration_h",
                        "pdis_mw",
                        "charge_mwh",
                        "discharge_mwh",
                        "efc_mw",
                        "annual_fixed_cost_cny_per_kw_year",
                        "storage_vom_cny_per_kw_year",
                    ]
                ].copy()
                current["scenario_id"] = scenario_id
                current["energy_value_cny_mwh"] = energy_value
                current["firm_value_cny_kw_year"] = firm_value
                current["charge_price_cny_mwh"] = charge_price
                current["net_benefit_cny_per_kw_year"] = (
                    current["discharge_mwh"]
                    * energy_value
                    / (current["pdis_mw"] * 1000.0)
                    + current["efc_mw"]
                    / current["pdis_mw"]
                    * firm_value
                    - current["annual_fixed_cost_cny_per_kw_year"]
                    - current["storage_vom_cny_per_kw_year"]
                    - current["charge_mwh"]
                    * charge_price
                    / (current["pdis_mw"] * 1000.0)
                )
                envelope = (
                    current.sort_values(
                        [
                            "country",
                            "year",
                            "chronology",
                            "duration_h",
                            "net_benefit_cny_per_kw_year",
                        ],
                        ascending=[True, True, True, True, False],
                        kind="stable",
                    )
                    .drop_duplicates(
                        ["country", "year", "chronology", "duration_h"]
                    )
                )
                rows.append(envelope)
                gap_records: list[dict[str, Any]] = []
                for keys, group in envelope.groupby(
                    ["country", "year", "chronology"], sort=False
                ):
                    ordered = group.sort_values("duration_h", kind="stable")
                    durations = ordered["duration_h"].to_numpy(float)
                    positive = (
                        ordered["net_benefit_cny_per_kw_year"].to_numpy(float)
                        > 0.0
                    )
                    gaps: list[float] = []
                    for position in range(1, len(positive) - 1):
                        if positive[position]:
                            continue
                        left_positive = np.flatnonzero(positive[:position])
                        right_positive = np.flatnonzero(positive[position + 1 :])
                        if len(left_positive) and len(right_positive):
                            left = int(left_positive[-1])
                            right = int(position + 1 + right_positive[0])
                            gaps.append(float(durations[right] - durations[left]))
                    gap_records.append(
                        {
                            "country": keys[0],
                            "year": keys[1],
                            "chronology": keys[2],
                            "has_strict_gap": bool(gaps),
                            "gap_width_h": max(gaps, default=0.0),
                        }
                    )
                gaps = pd.DataFrame(gap_records)
                for chronology, group in gaps.groupby("chronology"):
                    scenarios.append(
                        {
                            "scenario_id": scenario_id,
                            "energy_value_cny_mwh": energy_value,
                            "firm_value_cny_kw_year": firm_value,
                            "charge_price_cny_mwh": charge_price,
                            "chronology": chronology,
                            "country_years_with_gap": int(
                                group["has_strict_gap"].sum()
                            ),
                            "country_years": len(group),
                            "mean_gap_width_h": float(
                                group["gap_width_h"].mean()
                            ),
                        }
                    )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(scenarios)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    usecols = [
        "timestamp_utc",
        "country",
        "year",
        "source_hour_id",
        "source_day_id",
        "season_segment",
        "event_flag_q75",
        "raw_gas_mean_mw",
        TASK_COLUMN,
        RECHARGE_COLUMN,
    ]
    panel = pd.read_csv(args.panel, usecols=usecols)
    panel = panel.sort_values(
        ["country", "year", "source_hour_id"], kind="stable"
    ).reset_index(drop=True)
    catalogue = pd.read_csv(args.catalogue)
    if len(catalogue) != 53:
        raise RuntimeError(f"Expected 53 configurations; found {len(catalogue)}")

    rows: list[dict[str, Any]] = []
    chronology_rows: list[dict[str, Any]] = []
    for (country, year), frame in panel.groupby(
        ["country", "year"], sort=True
    ):
        if len(frame) != 8760:
            raise RuntimeError(f"{country}-{year} has {len(frame)} hours")
        orders, order_stats = day_orders(frame)
        task_original = frame[TASK_COLUMN].to_numpy(float)
        recharge_original = frame[RECHARGE_COLUMN].to_numpy(float)
        pdis_mw = FORCED_POWER_SHARE * float(frame["raw_gas_mean_mw"].iloc[0])
        if pdis_mw <= 0.0:
            raise RuntimeError(f"{country}-{year} has non-positive forced power")
        for chronology, order in orders.items():
            task = reorder_days(task_original, order)
            recharge = reorder_days(recharge_original, order)
            chronology_rows.append(
                {
                    "country": country,
                    "year": int(year),
                    "chronology": chronology,
                    **order_stats,
                    "task_mwh": float(task.sum()),
                    "recharge_mwh": float(recharge.sum()),
                    "task_original_conservation_error_mwh": float(
                        task.sum() - task_original.sum()
                    ),
                    "recharge_original_conservation_error_mwh": float(
                        recharge.sum() - recharge_original.sum()
                    ),
                }
            )
            for config in catalogue.itertuples(index=False):
                config_series = pd.Series(config._asdict())
                metrics = run_configuration(
                    task, recharge, config_series, pdis_mw
                )
                rows.append(
                    {
                        "country": country,
                        "year": int(year),
                        "chronology": chronology,
                        "config_id": config.config_id,
                        "tech_id": config.tech_id,
                        "technology_en": config.technology_en,
                        "duration_h": float(config.duration_h),
                        "rte": float(config.rte),
                        "charge_power_ratio": float(config.charge_power_ratio),
                        "self_discharge_pct_day": float(
                            config.self_discharge_pct_day
                        ),
                        "released_library_configuration": bool(
                            config.released_library_configuration
                        ),
                        "direct_duration_anchor": bool(
                            config.direct_duration_anchor
                        ),
                        "structural_extrapolation": bool(
                            config.structural_extrapolation
                        ),
                        **metrics,
                    }
                )

    results = pd.DataFrame(rows)
    chronology = pd.DataFrame(chronology_rows)
    index_columns = ["country", "year", "config_id"]
    dispersed = results[results["chronology"].eq("dispersed")].set_index(
        index_columns
    )
    clustered = results[results["chronology"].eq("clustered")].set_index(
        index_columns
    )
    paired = clustered[
        [
            "tech_id",
            "technology_en",
            "duration_h",
            "equivalent_full_cycles",
            "served_share",
            "efc_ratio",
            "joint_net_benefit_cny_per_kw_year",
        ]
    ].copy()
    paired = paired.rename(
        columns={
            "equivalent_full_cycles": "clustered_equivalent_full_cycles",
            "served_share": "clustered_served_share",
            "efc_ratio": "clustered_efc_ratio",
            "joint_net_benefit_cny_per_kw_year": (
                "clustered_joint_net_benefit_cny_per_kw_year"
            ),
        }
    )
    paired["dispersed_equivalent_full_cycles"] = dispersed[
        "equivalent_full_cycles"
    ]
    paired["dispersed_served_share"] = dispersed["served_share"]
    paired["dispersed_efc_ratio"] = dispersed["efc_ratio"]
    paired["dispersed_joint_net_benefit_cny_per_kw_year"] = dispersed[
        "joint_net_benefit_cny_per_kw_year"
    ]
    paired["delta_equivalent_full_cycles"] = (
        paired["clustered_equivalent_full_cycles"]
        - paired["dispersed_equivalent_full_cycles"]
    )
    paired["retained_equivalent_full_cycles"] = np.divide(
        paired["clustered_equivalent_full_cycles"],
        paired["dispersed_equivalent_full_cycles"],
        out=np.full(len(paired), np.nan),
        where=paired["dispersed_equivalent_full_cycles"].to_numpy() > 0.0,
    )
    paired["delta_served_share_pp"] = 100.0 * (
        paired["clustered_served_share"] - paired["dispersed_served_share"]
    )
    paired["delta_efc_ratio"] = (
        paired["clustered_efc_ratio"] - paired["dispersed_efc_ratio"]
    )
    paired["delta_joint_net_benefit_cny_per_kw_year"] = (
        paired["clustered_joint_net_benefit_cny_per_kw_year"]
        - paired["dispersed_joint_net_benefit_cny_per_kw_year"]
    )
    paired = paired.reset_index()
    paired_summary = summarize_paired(paired, args.bootstrap_draws)
    value_envelope, value_gap = value_gap_scan(results)

    results.to_csv(
        args.output / "configuration_results.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    chronology.to_csv(args.output / "chronology_conservation.csv", index=False)
    paired.to_csv(args.output / "paired_configuration_changes.csv.gz", index=False)
    paired_summary.to_csv(
        args.output / "paired_configuration_summary.csv", index=False
    )
    value_envelope.to_csv(
        args.output / "value_sensitivity_envelope.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    value_gap.to_csv(args.output / "value_sensitivity_gap_summary.csv", index=False)

    key_ids = ["li_lfp_4h", "caes_24h", "h2_cavern_48h", "h2_cavern_168h"]
    key = paired_summary[paired_summary["config_id"].isin(key_ids)].copy()
    baseline_gap = value_gap[
        np.isclose(value_gap["energy_value_cny_mwh"], GAS_VARIABLE_CNY_MWH)
        & np.isclose(value_gap["firm_value_cny_kw_year"], GAS_FIRM_CNY_KW_YEAR)
        & np.isclose(
            value_gap["charge_price_cny_mwh"],
            PRIMARY_CHARGE_PRICE_CNY_MWH,
        )
    ]
    summary = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "countries": int(panel["country"].nunique()),
        "country_years": int(
            panel[["country", "year"]].drop_duplicates().shape[0]
        ),
        "hour_rows": len(panel),
        "configurations": len(catalogue),
        "chronologies": ["natural", "dispersed", "clustered"],
        "task": (
            "C3-q75 actual aggregate public-net fossil-gas generation "
            "above the month-hour q10 floor on selected event days"
        ),
        "recharge": (
            "R2 potential renewable-surplus counterfactual from the same "
            "country-year load, wind and solar profiles"
        ),
        "dispatch": (
            "deterministic cyclic-SOC, task-first; charging prohibited in "
            "hours with positive task"
        ),
        "primary_values": {
            "energy_cny_mwh": GAS_VARIABLE_CNY_MWH,
            "firm_capacity_cny_kw_year": GAS_FIRM_CNY_KW_YEAR,
            "charging_opportunity_cny_mwh": PRIMARY_CHARGE_PRICE_CNY_MWH,
        },
        "key_configuration_results": key.to_dict(orient="records"),
        "baseline_gap_summary": baseline_gap.to_dict(orient="records"),
        "maximum_absolute_task_conservation_error_mwh": float(
            chronology["task_original_conservation_error_mwh"].abs().max()
        ),
        "maximum_absolute_recharge_conservation_error_mwh": float(
            chronology["recharge_original_conservation_error_mwh"].abs().max()
        ),
        "maximum_absolute_cyclic_soc_residual_mwh": float(
            results["cyclic_soc_residual_mwh"].abs().max()
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output / "mainline_summary.json", summary)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256(args.panel),
        "catalogue": str(args.catalogue.resolve()),
        "catalogue_sha256": sha256(args.catalogue),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": numba.__version__,
        "output_files": {},
    }
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            manifest["output_files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    write_json(args.output / "run_manifest.json", manifest)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
