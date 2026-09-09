#!/usr/bin/env python3
"""Gurobi perfect-foresight check for the European chronology mechanism.

The runner can evaluate either a focused LFP4/A-CAES24 comparison or the full
53-configuration duration spectrum.  It asks whether the
clustered-minus-dispersed response of fixed storage assets, and when requested
the strict valley itself, survive when annual dispatch is optimized with
perfect foresight instead of a task-first cyclic heuristic.

The experiment uses the same frozen 132-country-year observed-gas task panel
and the same within-season intact-day permutations as the European mainline.
For every country-year, chronology and technology:

* discharge power is fixed at 1% of mean observed fossil-gas generation;
* charge/discharge power interfaces are either matched across technologies or
  retained at their frozen catalogue values;
* the usable SOC interval is 5--95% and terminal SOC equals initial SOC;
* charging is limited by the synchronous R2 renewable-surplus profile;
* charging is prohibited in positive-task hours, so simultaneous
  charge/discharge cannot occur;
* a full-year Gurobi LP maximizes task service (equivalently minimizes the
  task-EENS proxy), then minimizes charging among service-optimal solutions.

Focused runs are labelled as mechanism robustness checks.  A full-catalogue
run additionally recomputes the 4--168 h economic envelope and strict-valley
width, while remaining distinct from a capacity-expansion or network-wide
test.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from numba import njit


SCRIPT_VERSION = "v41.0.1"
EVIDENCE_STATUS = "mechanism_robustness_not_full_duration_valley"
MANUSCRIPT_EVIDENCE_ROLE = "mechanism_robustness"
TASK_COLUMN = "task_c3_q75_mw"
RECHARGE_COLUMN = "r2_potential_recharge_mw"
SOC_MIN = 0.05
SOC_MAX = 0.95
FORCED_POWER_SHARE = 0.01
COMMON_CHARGE_TO_DISCHARGE_RATIO = 1.0
ENERGY_VALUE_EUR_MWH = 98.5
FIRM_VALUE_EUR_KW_YEAR = 103.8
CHARGE_OPPORTUNITY_PRICE_EUR_MWH = 0.0
CLUSTER_DAYS = 4
SERVICE_ABSOLUTE_TOLERANCE_MWH = 1.0e-6
BALANCE_TOLERANCE_MWH = 1.0e-6


HERE = Path(__file__).resolve().parent


def repository_root() -> Path:
    for candidate in HERE.parents:
        if (candidate / "dataset").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root containing dataset/")


ROOT = repository_root()
DEFAULT_PANEL = (
    ROOT
    / "dataset"
    / "external_hourly"
    / "europe_mid_duration_valley_upgrade_v0_1"
    / "processed"
    / "observed_gas_event_task_definitions_v0_2.csv.gz"
)
DEFAULT_CATALOGUE = (
    HERE.parents[1]
    / "revision_nature_energy_v40"
    / "currency_work"
    / "configuration_catalog_53_source_aware_eur.csv"
)
DEFAULT_OUTPUT = HERE / "results_primary"


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            json_ready(payload),
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
        )
        .reset_index()
    )
    if len(daily) != 365:
        raise RuntimeError("Every country-year must contain 365 complete days")
    dispersed: list[int] = []
    clustered: list[int] = []
    for segment in sorted(daily["segment"].unique()):
        subset = daily.loc[daily["segment"].eq(segment)]
        extreme = [
            int(value)
            for value in subset.loc[
                subset["event"].astype(bool), "source_day_id"
            ]
        ]
        normal = [
            int(value)
            for value in subset.loc[
                ~subset["event"].astype(bool), "source_day_id"
            ]
        ]
        dispersed.extend(interleave(extreme, normal))
        clustered.extend(interleave_clusters(extreme, normal, CLUSTER_DAYS))
    orders = {"dispersed": dispersed, "clustered": clustered}
    for label, order in orders.items():
        if len(order) != 365 or sorted(order) != list(range(365)):
            raise RuntimeError(f"Invalid intact-day order for {label}")

    event_days = set(
        int(value)
        for value in daily.loc[
            daily["event"].astype(bool), "source_day_id"
        ]
    )
    segment_by_day = daily.set_index("source_day_id")["segment"].to_dict()

    def run_stats(order: list[int]) -> tuple[int, int]:
        adjacent = 0
        longest = 0
        run = 0
        prior_event = False
        prior_segment: int | None = None
        for day in order:
            segment = int(segment_by_day[day])
            event = day in event_days
            if segment != prior_segment:
                run = 0
                prior_event = False
            if event:
                run += 1
                longest = max(longest, run)
                adjacent += int(prior_event)
            else:
                run = 0
            prior_event = event
            prior_segment = segment
        return adjacent, longest

    dispersed_adjacent, dispersed_longest = run_stats(dispersed)
    clustered_adjacent, clustered_longest = run_stats(clustered)
    return orders, {
        "event_days": len(event_days),
        "dispersed_adjacent_event_pairs": dispersed_adjacent,
        "clustered_adjacent_event_pairs": clustered_adjacent,
        "dispersed_longest_event_run_days": dispersed_longest,
        "clustered_longest_event_run_days": clustered_longest,
    }


def reorder_days(values: np.ndarray, order: list[int]) -> np.ndarray:
    return values.reshape(365, 24)[np.asarray(order, dtype=int)].reshape(-1)


def equivalent_firm_supply(task: np.ndarray, target_unserved: float) -> float:
    task_total = float(task.sum())
    if target_unserved >= task_total - 1.0e-10:
        return 0.0
    low = 0.0
    high = float(task.max())
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        unserved = float(np.maximum(task - midpoint, 0.0).sum())
        if unserved > target_unserved:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def technology_metrics(
    *,
    country: str,
    year: int,
    chronology: str,
    method: str,
    config: pd.Series,
    pdis_mw: float,
    common_charge_ratio: float,
    task: np.ndarray,
    recharge: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    usable_soc: np.ndarray,
    solver_status: int,
    solver_runtime_s: float,
    solver_iterations: float,
    maximum_balance_residual_mwh: float,
    cyclic_residual_mwh: float,
    heuristic_iterations: int,
    energy_value_eur_mwh: float,
    firm_value_eur_kw_year: float,
    charge_opportunity_price_eur_mwh: float,
) -> dict[str, Any]:
    duration_h = float(config["duration_h"])
    rated_energy_mwh = duration_h * pdis_mw
    task_total = float(task.sum())
    charge_total = float(charge.sum())
    discharge_total = float(discharge.sum())
    unserved = task_total - discharge_total
    firm_mw = equivalent_firm_supply(task, unserved)
    fixed_cost = float(
        config["annual_fixed_cost_eur_per_kw_year_source_aware"]
    )
    variable_om = (
        discharge_total
        * float(config["variable_om_eur_mwh_source_aware"])
        / (pdis_mw * 1_000.0)
    )
    charge_cost = (
        charge_total
        * charge_opportunity_price_eur_mwh
        / (pdis_mw * 1_000.0)
    )
    energy_value = (
        discharge_total
        * energy_value_eur_mwh
        / (pdis_mw * 1_000.0)
    )
    firm_value = firm_mw / pdis_mw * firm_value_eur_kw_year
    net_benefit = (
        energy_value
        + firm_value
        - fixed_cost
        - variable_om
        - charge_cost
    )
    soc_fraction = SOC_MIN + usable_soc / rated_energy_mwh
    return {
        "country": country,
        "year": int(year),
        "chronology": chronology,
        "method": method,
        "config_id": str(config["config_id"]),
        "tech_id": str(config["tech_id"]),
        "technology_en": str(config["technology_en"]),
        "duration_h": duration_h,
        "rte": float(config["rte"]),
        "self_discharge_pct_day": float(config["self_discharge_pct_day"]),
        "catalogue_charge_power_ratio": float(config["charge_power_ratio"]),
        "applied_common_charge_power_ratio": common_charge_ratio,
        "pdis_mw": pdis_mw,
        "pch_mw": common_charge_ratio * pdis_mw,
        "rated_energy_mwh": rated_energy_mwh,
        "usable_energy_mwh": (SOC_MAX - SOC_MIN) * rated_energy_mwh,
        "task_mwh": task_total,
        "recharge_available_mwh": float(recharge.sum()),
        "charge_mwh": charge_total,
        "discharge_mwh": discharge_total,
        "task_eens_mwh": unserved,
        "task_service_share": (
            discharge_total / task_total if task_total > 0 else 0.0
        ),
        "equivalent_full_cycles_rated_energy": (
            discharge_total / rated_energy_mwh
        ),
        "equivalent_full_cycles_usable_soc_window": (
            discharge_total / ((SOC_MAX - SOC_MIN) * rated_energy_mwh)
        ),
        "task_firm_supply_mw": firm_mw,
        "task_firm_supply_ratio": firm_mw / pdis_mw,
        "minimum_soc_fraction": float(soc_fraction.min()),
        "maximum_soc_fraction": float(soc_fraction.max()),
        "annual_fixed_cost_eur_per_kw_year": fixed_cost,
        "storage_vom_eur_per_kw_year": variable_om,
        "charge_opportunity_cost_eur_per_kw_year": charge_cost,
        "energy_value_eur_per_kw_year": energy_value,
        "firm_value_eur_per_kw_year": firm_value,
        "joint_net_benefit_eur_per_kw_year": net_benefit,
        "solver_status": solver_status,
        "solver_runtime_s": solver_runtime_s,
        "solver_iterations": solver_iterations,
        "maximum_soc_balance_residual_mwh": maximum_balance_residual_mwh,
        "cyclic_soc_residual_mwh": cyclic_residual_mwh,
        "heuristic_fixed_point_iterations": heuristic_iterations,
        "evidence_status": EVIDENCE_STATUS,
        "manuscript_evidence_role": MANUSCRIPT_EVIDENCE_ROLE,
    }


@njit(cache=True)
def greedy_cyclic_dispatch(
    task: np.ndarray,
    recharge: np.ndarray,
    *,
    pdis_mw: float,
    duration_h: float,
    rte: float,
    self_discharge_pct_day: float,
    common_charge_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    rated_energy = duration_h * pdis_mw
    usable_capacity = (SOC_MAX - SOC_MIN) * rated_energy
    eta = math.sqrt(rte)
    retention = (
        1.0 - self_discharge_pct_day / 100.0
    ) ** (1.0 / 24.0)
    pch_mw = common_charge_ratio * pdis_mw
    charge_cap = np.minimum(recharge, pch_mw)
    charge_cap = np.where(task > 1.0e-12, 0.0, charge_cap)
    discharge_cap = np.minimum(task, pdis_mw)

    start = 0.5 * usable_capacity
    iterations = 0
    for repetition in range(2_000):
        usable = start
        for hour in range(task.size):
            usable *= retention
            discharge = min(
                discharge_cap[hour], max(0.0, usable * eta)
            )
            usable -= discharge / eta
            charge = min(
                charge_cap[hour],
                max(0.0, (usable_capacity - usable) / eta),
            )
            usable += charge * eta
        iterations = repetition + 1
        if abs(usable - start) <= max(
            1.0e-10, 1.0e-12 * usable_capacity
        ):
            start = usable
            break
        start = usable
    else:
        raise RuntimeError("Greedy cyclic dispatch did not converge")

    charge = np.zeros(task.size, dtype=float)
    discharge = np.zeros(task.size, dtype=float)
    usable_soc = np.zeros(task.size, dtype=float)
    previous = start
    residuals = np.zeros(task.size, dtype=float)
    for hour in range(task.size):
        decayed = previous * retention
        discharge[hour] = min(
            discharge_cap[hour], max(0.0, decayed * eta)
        )
        after_discharge = decayed - discharge[hour] / eta
        charge[hour] = min(
            charge_cap[hour],
            max(0.0, (usable_capacity - after_discharge) / eta),
        )
        usable_soc[hour] = after_discharge + charge[hour] * eta
        residuals[hour] = usable_soc[hour] - (
            retention * previous
            + eta * charge[hour]
            - discharge[hour] / eta
        )
        previous = usable_soc[hour]
    cyclic_residual = previous - start
    return (
        charge,
        discharge,
        usable_soc,
        iterations,
        float(np.max(np.abs(residuals))),
        float(cyclic_residual),
    )


def gurobi_perfect_foresight_dispatch(
    task: np.ndarray,
    recharge: np.ndarray,
    *,
    pdis_mw: float,
    duration_h: float,
    rte: float,
    self_discharge_pct_day: float,
    common_charge_ratio: float,
    threads: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    float,
    float,
    float,
    float,
    float,
]:
    rated_energy = duration_h * pdis_mw
    usable_capacity = (SOC_MAX - SOC_MIN) * rated_energy
    eta = math.sqrt(rte)
    retention = (
        1.0 - self_discharge_pct_day / 100.0
    ) ** (1.0 / 24.0)
    pch_mw = common_charge_ratio * pdis_mw
    charge_cap = np.minimum(recharge, pch_mw)
    charge_cap = np.where(task > 1.0e-12, 0.0, charge_cap)
    discharge_cap = np.minimum(task, pdis_mw)
    hours = task.size

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0
    model.Params.Threads = threads
    model.Params.Method = 1
    model.Params.Presolve = 2
    model.Params.FeasibilityTol = 1.0e-9
    model.Params.OptimalityTol = 1.0e-9
    model.Params.NumericFocus = 1

    charge = model.addMVar(
        hours, lb=0.0, ub=charge_cap, name="charge"
    )
    discharge = model.addMVar(
        hours, lb=0.0, ub=discharge_cap, name="discharge"
    )
    usable_soc = model.addMVar(
        hours, lb=0.0, ub=usable_capacity, name="usable_soc"
    )
    model.addConstr(
        usable_soc[0]
        == retention * usable_soc[hours - 1]
        + eta * charge[0]
        - discharge[0] / eta,
        name="cyclic_hour_0",
    )
    model.addConstr(
        usable_soc[1:]
        == retention * usable_soc[:-1]
        + eta * charge[1:]
        - discharge[1:] / eta,
        name="soc_balance",
    )

    model.setObjective(discharge.sum(), GRB.MAXIMIZE)
    started = time.perf_counter()
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Gurobi service maximization failed with status {model.Status}"
        )
    maximum_service = float(discharge.X.sum())
    model.addConstr(
        discharge.sum()
        >= maximum_service - SERVICE_ABSOLUTE_TOLERANCE_MWH,
        name="service_optimum_floor",
    )
    model.setObjective(charge.sum(), GRB.MINIMIZE)
    model.optimize()
    runtime = time.perf_counter() - started
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Gurobi minimum-charge tie break failed with status {model.Status}"
        )

    charge_x = np.asarray(charge.X, dtype=float)
    discharge_x = np.asarray(discharge.X, dtype=float)
    soc_x = np.asarray(usable_soc.X, dtype=float)
    previous = np.roll(soc_x, 1)
    residual = soc_x - (
        retention * previous + eta * charge_x - discharge_x / eta
    )
    service_loss = maximum_service - float(discharge_x.sum())
    return (
        charge_x,
        discharge_x,
        soc_x,
        int(model.Status),
        runtime,
        float(model.IterCount),
        float(np.max(np.abs(residual))),
        float(residual[0]),
        float(service_loss),
    )


def country_cluster_interval(
    frame: pd.DataFrame,
    value: str,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    country_means = (
        frame.groupby("country", sort=True)[value].mean().to_numpy(float)
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(country_means), size=(draws, len(country_means))
    )
    draws_values = country_means[indices].mean(axis=1)
    return (
        float(country_means.mean()),
        float(np.quantile(draws_values, 0.025)),
        float(np.quantile(draws_values, 0.975)),
    )


def paired_changes(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["country", "year", "config_id", "method"]
    metrics = [
        "charge_mwh",
        "discharge_mwh",
        "task_eens_mwh",
        "task_service_share",
        "equivalent_full_cycles_rated_energy",
        "task_firm_supply_ratio",
        "joint_net_benefit_eur_per_kw_year",
    ]
    dispersed = results.loc[
        results["chronology"].eq("dispersed"), keys + metrics
    ].set_index(keys)
    clustered = results.loc[
        results["chronology"].eq("clustered"), keys + metrics
    ].set_index(keys)
    if not dispersed.index.equals(clustered.index):
        raise RuntimeError("Dispersed/clustered result keys are inconsistent")
    output = clustered.copy()
    for metric in metrics:
        output[f"dispersed_{metric}"] = dispersed[metric]
        output[f"clustered_{metric}"] = clustered[metric]
        output[f"delta_{metric}"] = clustered[metric] - dispersed[metric]
    output["delta_task_service_share_pp"] = (
        100.0 * output["delta_task_service_share"]
    )
    output["retained_equivalent_full_cycles"] = np.divide(
        output["clustered_equivalent_full_cycles_rated_energy"],
        output["dispersed_equivalent_full_cycles_rated_energy"],
        out=np.full(len(output), np.nan),
        where=(
            output[
                "dispersed_equivalent_full_cycles_rated_energy"
            ].to_numpy(float)
            > 0.0
        ),
    )
    return output.reset_index()


def relative_caes_lfp(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["country", "year", "chronology", "method"]
    metrics = [
        "charge_mwh",
        "discharge_mwh",
        "task_eens_mwh",
        "task_service_share",
        "equivalent_full_cycles_rated_energy",
        "task_firm_supply_ratio",
        "joint_net_benefit_eur_per_kw_year",
    ]
    subset = results.loc[
        results["config_id"].isin(["li_lfp_4h", "caes_24h"]),
        keys + ["config_id"] + metrics,
    ]
    wide = subset.pivot(index=keys, columns="config_id", values=metrics)
    expected = {"li_lfp_4h", "caes_24h"}
    if set(wide.columns.get_level_values(1)) != expected:
        raise RuntimeError("LFP4/CAES24 relative comparison is incomplete")
    output = pd.DataFrame(index=wide.index)
    for metric in metrics:
        output[f"caes_minus_lfp_{metric}"] = (
            wide[(metric, "caes_24h")] - wide[(metric, "li_lfp_4h")]
        )
    output["caes_minus_lfp_task_service_share_pp"] = (
        100.0 * output["caes_minus_lfp_task_service_share"]
    )
    output = output.reset_index()

    pair_keys = ["country", "year", "method"]
    dispersed = output.loc[
        output["chronology"].eq("dispersed")
    ].set_index(pair_keys)
    clustered = output.loc[
        output["chronology"].eq("clustered")
    ].set_index(pair_keys)
    if not dispersed.index.equals(clustered.index):
        raise RuntimeError("Relative comparison chronology keys differ")
    paired = clustered.drop(columns=["chronology"]).copy()
    relative_metrics = [
        column
        for column in output.columns
        if column.startswith("caes_minus_lfp_")
    ]
    for metric in relative_metrics:
        paired[f"dispersed_{metric}"] = dispersed[metric]
        paired[f"clustered_{metric}"] = clustered[metric]
        paired[f"delta_{metric}"] = clustered[metric] - dispersed[metric]
    return output, paired.reset_index()


def summarize(
    paired: pd.DataFrame,
    relative_paired: pd.DataFrame,
    *,
    draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_metrics = [
        "delta_equivalent_full_cycles_rated_energy",
        "retained_equivalent_full_cycles",
        "delta_task_service_share_pp",
        "delta_task_firm_supply_ratio",
        "delta_joint_net_benefit_eur_per_kw_year",
    ]
    paired_rows: list[dict[str, Any]] = []
    for (config_id, method), group in paired.groupby(
        ["config_id", "method"], sort=True
    ):
        row: dict[str, Any] = {
            "config_id": config_id,
            "method": method,
            "country_years": len(group),
            "countries": group["country"].nunique(),
        }
        for index, metric in enumerate(paired_metrics):
            mean, low, high = country_cluster_interval(
                group,
                metric,
                draws=draws,
                seed=20260728 + index,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        paired_rows.append(row)

    relative_metrics = [
        "delta_caes_minus_lfp_equivalent_full_cycles_rated_energy",
        "delta_caes_minus_lfp_task_service_share_pp",
        "delta_caes_minus_lfp_task_firm_supply_ratio",
        "delta_caes_minus_lfp_joint_net_benefit_eur_per_kw_year",
    ]
    relative_rows: list[dict[str, Any]] = []
    for method, group in relative_paired.groupby("method", sort=True):
        row = {
            "method": method,
            "country_years": len(group),
            "countries": group["country"].nunique(),
        }
        for index, metric in enumerate(relative_metrics):
            mean, low, high = country_cluster_interval(
                group,
                metric,
                draws=draws,
                seed=20260828 + index,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        row[
            "country_years_with_negative_relative_net_benefit_shift"
        ] = int(
            (
                group[
                    "delta_caes_minus_lfp_joint_net_benefit_eur_per_kw_year"
                ]
                < 0.0
            ).sum()
        )
        relative_rows.append(row)
    return pd.DataFrame(paired_rows), pd.DataFrame(relative_rows)


def parse_config_ids(value: str) -> list[str]:
    output = [item.strip() for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("At least one configuration is required")
    return output


def strict_valley_outputs(
    results: pd.DataFrame,
    *,
    maximum_duration_h: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = results.loc[
        results["duration_h"].le(maximum_duration_h)
    ].copy()
    envelope_keys = [
        "country",
        "year",
        "chronology",
        "method",
        "duration_h",
    ]
    selected_index = eligible.groupby(
        envelope_keys, sort=False
    )["joint_net_benefit_eur_per_kw_year"].idxmax()
    envelope = (
        eligible.loc[selected_index]
        .sort_values(envelope_keys, kind="stable")
        .reset_index(drop=True)
    )

    records: list[dict[str, Any]] = []
    for keys, group in envelope.groupby(
        ["country", "year", "chronology", "method"], sort=False
    ):
        ordered = group.sort_values("duration_h", kind="stable")
        durations = ordered["duration_h"].to_numpy(float)
        positive = (
            ordered["joint_net_benefit_eur_per_kw_year"].to_numpy(float)
            > 0.0
        )
        gaps: list[tuple[float, float, float]] = []
        for position in range(1, len(positive) - 1):
            if positive[position]:
                continue
            left_positive = np.flatnonzero(positive[:position])
            right_positive = np.flatnonzero(positive[position + 1 :])
            if len(left_positive) and len(right_positive):
                left = int(left_positive[-1])
                right = int(position + 1 + right_positive[0])
                gaps.append(
                    (
                        float(durations[right] - durations[left]),
                        float(durations[left]),
                        float(durations[right]),
                    )
                )
        if gaps:
            width, left_h, right_h = max(
                gaps, key=lambda item: item[0]
            )
        else:
            width, left_h, right_h = 0.0, np.nan, np.nan
        records.append(
            {
                "country": keys[0],
                "year": int(keys[1]),
                "chronology": keys[2],
                "method": keys[3],
                "has_strict_valley": bool(gaps),
                "strict_valley_width_h": width,
                "left_positive_duration_h": left_h,
                "right_reentry_duration_h": right_h,
                "maximum_tested_duration_h": maximum_duration_h,
            }
        )
    valley_records = pd.DataFrame(records)
    valley_summary = (
        valley_records.groupby(
            ["chronology", "method"], as_index=False, sort=True
        )
        .agg(
            country_years=("has_strict_valley", "size"),
            strict_valley_country_years=("has_strict_valley", "sum"),
            mean_valley_width_h=("strict_valley_width_h", "mean"),
            median_valley_width_h=("strict_valley_width_h", "median"),
            mean_positive_valley_width_h=(
                "strict_valley_width_h",
                lambda values: float(
                    values.loc[values.gt(0.0)].mean()
                )
                if values.gt(0.0).any()
                else 0.0,
            ),
        )
    )
    return envelope, valley_records, valley_summary


def main() -> None:
    global EVIDENCE_STATUS
    global MANUSCRIPT_EVIDENCE_ROLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--configs", default="li_lfp_4h,caes_24h"
    )
    parser.add_argument(
        "--common-charge-ratio",
        type=float,
        default=COMMON_CHARGE_TO_DISCHARGE_RATIO,
    )
    parser.add_argument(
        "--interface-mode",
        choices=("common", "catalogue"),
        default="common",
        help=(
            "Use one common charge/discharge-power ratio or each "
            "configuration's frozen catalogue ratio."
        ),
    )
    parser.add_argument(
        "--energy-value",
        type=float,
        default=ENERGY_VALUE_EUR_MWH,
    )
    parser.add_argument(
        "--firm-value",
        type=float,
        default=FIRM_VALUE_EUR_KW_YEAR,
    )
    parser.add_argument(
        "--charge-price",
        type=float,
        default=CHARGE_OPPORTUNITY_PRICE_EUR_MWH,
    )
    parser.add_argument(
        "--valley-max-duration",
        type=float,
        default=168.0,
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument(
        "--country-year-limit",
        type=int,
        default=0,
        help="Positive values run the first N country-years for smoke testing.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    started = time.perf_counter()

    if args.common_charge_ratio <= 0:
        raise ValueError("Common charge-power ratio must be positive")
    if args.threads <= 0:
        raise ValueError("Gurobi thread count must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    usecols = [
        "country",
        "year",
        "source_hour_id",
        "source_day_id",
        "season_segment",
        "event_flag_q75",
        "raw_gas_mean_mw",
        "forced_discharge_power_mw",
        TASK_COLUMN,
        RECHARGE_COLUMN,
    ]
    panel = pd.read_csv(args.panel, usecols=usecols)
    panel = panel.sort_values(
        ["country", "year", "source_hour_id"], kind="stable"
    ).reset_index(drop=True)
    country_year_keys = (
        panel[["country", "year"]]
        .drop_duplicates()
        .sort_values(["country", "year"], kind="stable")
    )
    if args.country_year_limit > 0:
        country_year_keys = country_year_keys.head(args.country_year_limit)
        key_index = pd.MultiIndex.from_frame(country_year_keys)
        panel_index = pd.MultiIndex.from_frame(panel[["country", "year"]])
        panel = panel.loc[panel_index.isin(key_index)].copy()

    catalogue = pd.read_csv(args.catalogue)
    if args.configs.strip().lower() == "all":
        config_ids = catalogue["config_id"].astype(str).tolist()
    else:
        config_ids = parse_config_ids(args.configs)
    selected = catalogue.loc[catalogue["config_id"].isin(config_ids)].copy()
    if set(selected["config_id"]) != set(config_ids):
        missing = sorted(set(config_ids).difference(selected["config_id"]))
        raise RuntimeError(f"Missing configuration IDs: {missing}")
    selected["_order"] = selected["config_id"].map(config_ids.index)
    selected = selected.sort_values("_order").drop(columns="_order")
    full_catalogue_run = set(config_ids) == set(
        catalogue["config_id"].astype(str)
    )
    if full_catalogue_run:
        EVIDENCE_STATUS = (
            "full_duration_perfect_foresight_valley_robustness"
        )
        MANUSCRIPT_EVIDENCE_ROLE = (
            "full_duration_chronology_valley_robustness"
        )

    rows: list[dict[str, Any]] = []
    chronology_rows: list[dict[str, Any]] = []
    total_country_years = panel[["country", "year"]].drop_duplicates().shape[0]
    for cy_index, ((country, year), frame) in enumerate(
        panel.groupby(["country", "year"], sort=True), start=1
    ):
        if len(frame) != 8_760:
            raise RuntimeError(f"{country}-{year} has {len(frame)} rows")
        source_ids = frame["source_hour_id"].to_numpy(int)
        if not np.array_equal(source_ids, np.arange(8_760)):
            raise RuntimeError(f"{country}-{year} has a non-canonical hour axis")
        orders, order_stats = day_orders(frame)
        original_task = frame[TASK_COLUMN].to_numpy(float)
        original_recharge = frame[RECHARGE_COLUMN].to_numpy(float)
        pdis_mw = float(frame["forced_discharge_power_mw"].iloc[0])
        raw_mean = float(frame["raw_gas_mean_mw"].iloc[0])
        if not np.isclose(
            pdis_mw,
            FORCED_POWER_SHARE * raw_mean,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise RuntimeError(f"{country}-{year}: forced-power identity failed")

        for chronology, order in orders.items():
            task = reorder_days(original_task, order)
            recharge = reorder_days(original_recharge, order)
            chronology_rows.append(
                {
                    "country": country,
                    "year": int(year),
                    "chronology": chronology,
                    **order_stats,
                    "task_mwh": float(task.sum()),
                    "recharge_mwh": float(recharge.sum()),
                    "task_conservation_error_mwh": float(
                        task.sum() - original_task.sum()
                    ),
                    "recharge_conservation_error_mwh": float(
                        recharge.sum() - original_recharge.sum()
                    ),
                }
            )
            for config in selected.to_dict(orient="records"):
                config_series = pd.Series(config)
                applied_charge_ratio = (
                    args.common_charge_ratio
                    if args.interface_mode == "common"
                    else float(config_series["charge_power_ratio"])
                )
                common = {
                    "pdis_mw": pdis_mw,
                    "duration_h": float(config_series["duration_h"]),
                    "rte": float(config_series["rte"]),
                    "self_discharge_pct_day": float(
                        config_series["self_discharge_pct_day"]
                    ),
                    "common_charge_ratio": applied_charge_ratio,
                }
                (
                    h_charge,
                    h_discharge,
                    h_soc,
                    h_iterations,
                    h_residual,
                    h_cyclic,
                ) = greedy_cyclic_dispatch(
                    task, recharge, **common
                )
                rows.append(
                    technology_metrics(
                        country=country,
                        year=int(year),
                        chronology=chronology,
                        method="greedy_cyclic",
                        config=config_series,
                        task=task,
                        recharge=recharge,
                        charge=h_charge,
                        discharge=h_discharge,
                        usable_soc=h_soc,
                        solver_status=0,
                        solver_runtime_s=0.0,
                        solver_iterations=0.0,
                        maximum_balance_residual_mwh=h_residual,
                        cyclic_residual_mwh=h_cyclic,
                        heuristic_iterations=h_iterations,
                        pdis_mw=pdis_mw,
                        common_charge_ratio=applied_charge_ratio,
                        energy_value_eur_mwh=args.energy_value,
                        firm_value_eur_kw_year=args.firm_value,
                        charge_opportunity_price_eur_mwh=args.charge_price,
                    )
                )
                (
                    o_charge,
                    o_discharge,
                    o_soc,
                    o_status,
                    o_runtime,
                    o_iterations,
                    o_residual,
                    o_cyclic,
                    o_service_loss,
                ) = gurobi_perfect_foresight_dispatch(
                    task,
                    recharge,
                    threads=args.threads,
                    **common,
                )
                if o_service_loss > 1.01 * SERVICE_ABSOLUTE_TOLERANCE_MWH:
                    raise RuntimeError(
                        f"{country}-{year}/{chronology}/"
                        f"{config_series['config_id']}: tie-break lost "
                        f"{o_service_loss} MWh of service"
                    )
                rows.append(
                    technology_metrics(
                        country=country,
                        year=int(year),
                        chronology=chronology,
                        method="gurobi_perfect_foresight",
                        config=config_series,
                        task=task,
                        recharge=recharge,
                        charge=o_charge,
                        discharge=o_discharge,
                        usable_soc=o_soc,
                        solver_status=o_status,
                        solver_runtime_s=o_runtime,
                        solver_iterations=o_iterations,
                        maximum_balance_residual_mwh=o_residual,
                        cyclic_residual_mwh=o_cyclic,
                        heuristic_iterations=0,
                        pdis_mw=pdis_mw,
                        common_charge_ratio=applied_charge_ratio,
                        energy_value_eur_mwh=args.energy_value,
                        firm_value_eur_kw_year=args.firm_value,
                        charge_opportunity_price_eur_mwh=args.charge_price,
                    )
                )
        if (
            args.progress_every > 0
            and (
                cy_index % args.progress_every == 0
                or cy_index == total_country_years
            )
        ):
            print(
                f"[{cy_index}/{total_country_years}] "
                f"{country}-{year}; elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    results = pd.DataFrame(rows)
    chronology = pd.DataFrame(chronology_rows)
    paired = paired_changes(results)
    relative, relative_paired = relative_caes_lfp(results)
    paired_summary, relative_summary = summarize(
        paired,
        relative_paired,
        draws=args.bootstrap_draws,
    )
    (
        duration_envelope,
        strict_valley_records,
        strict_valley_summary,
    ) = strict_valley_outputs(
        results,
        maximum_duration_h=args.valley_max_duration,
    )

    optimization_uplift = (
        results.pivot(
            index=["country", "year", "chronology", "config_id"],
            columns="method",
            values=[
                "discharge_mwh",
                "task_service_share",
                "task_firm_supply_ratio",
                "joint_net_benefit_eur_per_kw_year",
            ],
        )
        .sort_index()
    )
    uplift_rows: dict[str, pd.Series] = {}
    for metric in optimization_uplift.columns.get_level_values(0).unique():
        uplift_rows[f"perfect_minus_greedy_{metric}"] = (
            optimization_uplift[
                (metric, "gurobi_perfect_foresight")
            ]
            - optimization_uplift[(metric, "greedy_cyclic")]
        )
    optimization_uplift_frame = pd.DataFrame(uplift_rows).reset_index()

    result_key_count = (
        total_country_years
        * 2
        * len(config_ids)
        * 2
    )
    perfect = results.loc[
        results["method"].eq("gurobi_perfect_foresight")
    ]
    greedy = results.loc[results["method"].eq("greedy_cyclic")]
    qa_gates = {
        "expected_result_rows": len(results) == result_key_count,
        "all_country_years_have_8760_hours": (
            panel.groupby(["country", "year"]).size().eq(8_760).all()
        ),
        "two_chronologies_present": set(results["chronology"])
        == {"dispersed", "clustered"},
        "two_dispatch_methods_present": set(results["method"])
        == {"greedy_cyclic", "gurobi_perfect_foresight"},
        "all_requested_configurations_present": set(results["config_id"])
        == set(config_ids),
        "charge_interface_matches_selected_contract": (
            results["applied_common_charge_power_ratio"]
            .eq(args.common_charge_ratio)
            .all()
            if args.interface_mode == "common"
            else np.allclose(
                results["applied_common_charge_power_ratio"],
                results["catalogue_charge_power_ratio"],
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "all_gurobi_cells_optimal": perfect["solver_status"].eq(GRB.OPTIMAL).all(),
        "perfect_foresight_never_serves_less_than_greedy": (
            optimization_uplift_frame[
                "perfect_minus_greedy_discharge_mwh"
            ].min()
            >= -2.0 * SERVICE_ABSOLUTE_TOLERANCE_MWH
        ),
        "all_task_conservation_pass": (
            chronology["task_conservation_error_mwh"].abs().max()
            <= BALANCE_TOLERANCE_MWH
        ),
        "all_recharge_conservation_pass": (
            chronology["recharge_conservation_error_mwh"].abs().max()
            <= BALANCE_TOLERANCE_MWH
        ),
        "all_soc_balances_pass": (
            results["maximum_soc_balance_residual_mwh"].abs().max()
            <= BALANCE_TOLERANCE_MWH
        ),
        "all_greedy_cyclic_boundaries_pass": (
            greedy["cyclic_soc_residual_mwh"].abs().max()
            <= BALANCE_TOLERANCE_MWH
        ),
        "all_metrics_finite": np.isfinite(
            results[
                [
                    "charge_mwh",
                    "discharge_mwh",
                    "task_eens_mwh",
                    "task_service_share",
                    "equivalent_full_cycles_rated_energy",
                    "task_firm_supply_ratio",
                    "joint_net_benefit_eur_per_kw_year",
                ]
            ].to_numpy(float)
        ).all(),
        "evidence_scope_matches_declared_status": results[
            "evidence_status"
        ].eq(EVIDENCE_STATUS).all(),
    }

    results.to_csv(
        args.output / "cell_results.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    chronology.to_csv(
        args.output / "chronology_conservation.csv", index=False
    )
    paired.to_csv(
        args.output / "paired_configuration_changes.csv", index=False
    )
    relative.to_csv(
        args.output / "caes_minus_lfp_by_chronology.csv", index=False
    )
    relative_paired.to_csv(
        args.output / "caes_minus_lfp_paired_changes.csv", index=False
    )
    paired_summary.to_csv(
        args.output / "paired_configuration_summary.csv", index=False
    )
    relative_summary.to_csv(
        args.output / "caes_minus_lfp_summary.csv", index=False
    )
    optimization_uplift_frame.to_csv(
        args.output / "perfect_foresight_uplift.csv", index=False
    )
    duration_envelope.to_csv(
        args.output / "duration_value_envelope.csv", index=False
    )
    strict_valley_records.to_csv(
        args.output / "strict_valley_records.csv", index=False
    )
    strict_valley_summary.to_csv(
        args.output / "strict_valley_summary.csv", index=False
    )

    qa = {
        "status": "pass" if all(qa_gates.values()) else "fail",
        "qa_pass": bool(all(qa_gates.values())),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "evidence_status": EVIDENCE_STATUS,
        "manuscript_evidence_admitted_as": MANUSCRIPT_EVIDENCE_ROLE,
        "coverage": {
            "countries": int(panel["country"].nunique()),
            "country_years": int(total_country_years),
            "country_hours": int(len(panel)),
            "configurations": len(config_ids),
            "chronologies": 2,
            "dispatch_methods": 2,
            "result_rows": len(results),
        },
        "physical_contract": {
            "forced_discharge_power_share_of_mean_gas": FORCED_POWER_SHARE,
            "charge_interface_mode": args.interface_mode,
            "common_charge_to_discharge_power_ratio": (
                args.common_charge_ratio
                if args.interface_mode == "common"
                else None
            ),
            "soc_min_fraction": SOC_MIN,
            "soc_max_fraction": SOC_MAX,
            "cyclic_soc": True,
            "charging_in_positive_task_hours": False,
            "simultaneous_charge_discharge": False,
            "objective": (
                "lexicographic maximum task service followed by minimum "
                "charging energy"
            ),
        },
        "economic_reporting_contract": {
            "energy_value_eur_mwh": args.energy_value,
            "task_firm_supply_value_eur_kw_year": (
                args.firm_value
            ),
            "charge_opportunity_price_eur_mwh": (
                args.charge_price
            ),
            "technology_costs": (
                "source-aware EUR 2035 catalogue used in manuscript v40"
            ),
        },
        "qa_gates": qa_gates,
        "maximum_absolute_task_conservation_error_mwh": float(
            chronology["task_conservation_error_mwh"].abs().max()
        ),
        "maximum_absolute_recharge_conservation_error_mwh": float(
            chronology["recharge_conservation_error_mwh"].abs().max()
        ),
        "maximum_absolute_soc_balance_residual_mwh": float(
            results["maximum_soc_balance_residual_mwh"].abs().max()
        ),
        "maximum_absolute_greedy_cyclic_residual_mwh": float(
            greedy["cyclic_soc_residual_mwh"].abs().max()
        ),
        "minimum_perfect_minus_greedy_discharge_mwh": float(
            optimization_uplift_frame[
                "perfect_minus_greedy_discharge_mwh"
            ].min()
        ),
        "total_gurobi_runtime_s": float(perfect["solver_runtime_s"].sum()),
        "maximum_single_gurobi_runtime_s": float(
            perfect["solver_runtime_s"].max()
        ),
        "wall_elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output / "QA.json", qa)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256(args.panel),
        "catalogue": str(args.catalogue.resolve()),
        "catalogue_sha256": sha256(args.catalogue),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "gurobi": ".".join(str(value) for value in gp.gurobi.version()),
        },
        "output_files": {},
    }
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            manifest["output_files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    write_json(args.output / "run_manifest.json", manifest)
    print(json.dumps(json_ready(qa), ensure_ascii=False, indent=2))
    if not qa["qa_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
