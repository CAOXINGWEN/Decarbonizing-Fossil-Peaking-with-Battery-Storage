#!/usr/bin/env python3
"""Confirmatory European perfect-foresight test of the mid-duration valley.

This runner keeps the frozen v40 task, recharge, intact-day chronology,
technology catalogue, forced power and scenario-31 valuation inputs.  For each
country-year, chronology and technology-duration configuration it compares:

1. a matched task-first greedy dispatch; and
2. a Gurobi continuous linear programme with annual perfect foresight.

Both dispatches use the same cyclic usable-SOC accounting.  The LP maximizes
annual peaking-task supply and then minimizes charging energy lexicographically.
No capacity, cost or value parameter is fitted to the resulting valley.

The runner writes one checkpoint shard per country-year and can resume safely.
It is intended to be run first on the fixed DE-2023/SI-2018 smoke sample and
then on the complete 132-country-year panel.
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
from typing import Any, Iterable

import gurobipy as gp
import numba
import numpy as np
import pandas as pd
import scipy
from gurobipy import GRB
from numba import njit
from scipy import sparse


HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent
REPOSITORY_ROOT = HERE.parents[3]
MANUSCRIPT_ROOT = (
    REPOSITORY_ROOT
    / "Decarbonizing-Fossil-Peaking-with-Battery-Storage-master"
    / "paper-code"
    / "docs"
    / "musrcipt"
)
V40_ROOT = MANUSCRIPT_ROOT / "revision_nature_energy_v40"
V41_ROOT = MANUSCRIPT_ROOT / "revision_nature_energy_v41"

DEFAULT_PANEL = (
    DATASET_ROOT
    / "processed"
    / "observed_gas_event_task_definitions_v0_2.csv.gz"
)
DEFAULT_CATALOGUE = (
    V40_ROOT
    / "currency_work"
    / "configuration_catalog_53_source_aware_eur.csv"
)
DEFAULT_SCENARIO_SUMMARY = (
    V40_ROOT
    / "currency_work"
    / "value_sensitivity_gap_summary_source_aware_eur.csv"
)
DEFAULT_FROZEN_V40_RESULTS = (
    V40_ROOT
    / "currency_work"
    / "configuration_results_source_aware_eur.csv.gz"
)
DEFAULT_OUTPUT = V41_ROOT / "perfect_foresight_valley_full_132cy"

TASK_COLUMN = "task_c3_q75_mw"
RECHARGE_COLUMN = "r2_potential_recharge_mw"
SCENARIO_ID = 31
CLUSTER_DAYS = 4
SOC_MIN = 0.05
SOC_MAX = 0.95
USABLE_SOC_FRACTION = SOC_MAX - SOC_MIN
FORCED_POWER_SHARE = 0.01
SIGN_THRESHOLD_EUR_PER_KW_YEAR = 0.0
ENVELOPE_MAX_DURATION_H = 168.0
NUMERICAL_ZERO = 1.0e-12
CYCLIC_TOLERANCE_NORMALIZED_MWH = 1.0e-8
LP_FEASIBILITY_TOLERANCE = 1.0e-8
LP_OPTIMALITY_TOLERANCE = 1.0e-8
PRIMARY_OBJECTIVE_ABS_TOLERANCE = 1.0e-8
EXPECTED_COUNTRY_YEARS = 132
EXPECTED_CONFIGURATIONS = 53
EXPECTED_CHRONOLOGIES = ("dispersed", "clustered")


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
    if isinstance(value, float):
        return value if np.isfinite(value) else None
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_country_years(value: str | None) -> list[tuple[str, int]] | None:
    if value is None or not value.strip():
        return None
    pairs: list[tuple[str, int]] = []
    for item in value.split(","):
        country, year = item.strip().split(":", maxsplit=1)
        pairs.append((country.strip().upper(), int(year)))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate country-years were requested")
    return pairs


def parse_csv_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one item is required")
    if len(result) != len(set(result)):
        raise ValueError("Duplicate items are not allowed")
    return result


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
    """Reproduce the frozen v40 intact-day dispersed/clustered construction."""
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
    orders = {"dispersed": dispersed, "clustered": clustered}
    for label, order in orders.items():
        if len(order) != 365 or sorted(order) != list(range(365)):
            raise RuntimeError(f"Invalid intact-day order for {label}")

    event_days = set(
        int(value)
        for value in daily.loc[daily["event"].astype(bool), "source_day_id"]
    )
    segment_by_day = daily.set_index("source_day_id")["segment"].to_dict()

    def adjacency(order: list[int]) -> tuple[int, int]:
        adjacent = 0
        longest = 0
        run = 0
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
def greedy_dispatch_usable_soc(
    task_norm: np.ndarray,
    recharge_norm: np.ndarray,
    duration_h: float,
    rte: float,
    self_discharge_pct_day: float,
    charge_power_ratio: float,
) -> tuple[
    float,
    float,
    float,
    float,
    int,
    int,
    float,
    float,
    float,
    int,
    int,
]:
    """Matched task-first greedy dispatch on cyclic usable SOC.

    All powers and energies are normalized by discharge power.  The state
    represents inventory above the frozen 5% minimum, so its bounds are
    0--0.9 times duration.  This removes the floor-clamping energy injection
    in the legacy v40 heuristic while keeping its usable energy window.
    """
    usable_capacity = USABLE_SOC_FRACTION * duration_h
    eta = np.sqrt(rte)
    retention = (1.0 - self_discharge_pct_day / 100.0) ** (1.0 / 24.0)
    state = 0.5 * usable_capacity
    iterations = 0

    for repetition in range(2000):
        start = state
        for hour in range(task_norm.size):
            state *= retention
            if task_norm[hour] > NUMERICAL_ZERO:
                discharge = min(1.0, task_norm[hour], state * eta)
                state -= discharge / eta
            elif recharge_norm[hour] > NUMERICAL_ZERO:
                charge = min(
                    charge_power_ratio,
                    recharge_norm[hour],
                    max(0.0, (usable_capacity - state) / eta),
                )
                state += charge * eta
        iterations = repetition + 1
        if abs(state - start) <= CYCLIC_TOLERANCE_NORMALIZED_MWH:
            break

    cyclic_start = state
    charge_total = 0.0
    discharge_total = 0.0
    charge_saturated = 0
    discharge_saturated = 0
    state_min = state
    state_max = state
    overlap_hours = 0
    for hour in range(task_norm.size):
        state *= retention
        if task_norm[hour] > NUMERICAL_ZERO:
            if recharge_norm[hour] > NUMERICAL_ZERO:
                overlap_hours += 1
            discharge = min(1.0, task_norm[hour], state * eta)
            state -= discharge / eta
            discharge_total += discharge
            if discharge >= 1.0 - 1.0e-9:
                discharge_saturated += 1
        elif recharge_norm[hour] > NUMERICAL_ZERO:
            charge = min(
                charge_power_ratio,
                recharge_norm[hour],
                max(0.0, (usable_capacity - state) / eta),
            )
            state += charge * eta
            charge_total += charge
            if charge >= charge_power_ratio - 1.0e-9:
                charge_saturated += 1
        state_min = min(state_min, state)
        state_max = max(state_max, state)
    return (
        charge_total,
        discharge_total,
        float(task_norm.sum()) - discharge_total,
        float(recharge_norm.sum()) - charge_total,
        charge_saturated,
        discharge_saturated,
        state_min,
        state_max,
        state - cyclic_start,
        overlap_hours,
        iterations,
    )


def build_soc_matrix(
    hours: int, eta: float, retention: float
) -> sparse.csr_matrix:
    """Build x_soc[t] = r*x_soc[t-1] + eta*q[t] - d[t]/eta."""
    rows = np.tile(np.arange(hours, dtype=np.int32), 4)
    previous = (np.arange(hours, dtype=np.int32) - 1) % hours
    columns = np.concatenate(
        [
            np.arange(hours, dtype=np.int32),
            hours + np.arange(hours, dtype=np.int32),
            2 * hours + np.arange(hours, dtype=np.int32),
            2 * hours + previous,
        ]
    )
    values = np.concatenate(
        [
            np.full(hours, -eta, dtype=float),
            np.full(hours, 1.0 / eta, dtype=float),
            np.ones(hours, dtype=float),
            np.full(hours, -retention, dtype=float),
        ]
    )
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(hours, 3 * hours)
    ).tocsr()


def solve_perfect_foresight_lp(
    task_norm: np.ndarray,
    recharge_norm: np.ndarray,
    duration_h: float,
    rte: float,
    self_discharge_pct_day: float,
    charge_power_ratio: float,
    env: gp.Env,
    threads: int,
    method: int,
) -> dict[str, Any]:
    """Solve one annual continuous LP with cyclic usable SOC."""
    hours = len(task_norm)
    eta = float(np.sqrt(rte))
    retention = float(
        (1.0 - self_discharge_pct_day / 100.0) ** (1.0 / 24.0)
    )
    usable_capacity = USABLE_SOC_FRACTION * duration_h
    charge_ub = np.where(
        task_norm > NUMERICAL_ZERO,
        0.0,
        np.minimum(recharge_norm, charge_power_ratio),
    )
    discharge_ub = np.minimum(task_norm, 1.0)
    lower = np.zeros(3 * hours, dtype=float)
    upper = np.concatenate(
        [
            charge_ub,
            discharge_ub,
            np.full(hours, usable_capacity, dtype=float),
        ]
    )
    matrix = build_soc_matrix(hours, eta, retention)

    model = gp.Model(env=env)
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.Method = method
    model.Params.MultiObjMethod = method
    model.Params.Presolve = 2
    model.Params.FeasibilityTol = LP_FEASIBILITY_TOLERANCE
    model.Params.OptimalityTol = LP_OPTIMALITY_TOLERANCE
    x = model.addMVar(3 * hours, lb=lower, ub=upper, name="dispatch")
    model.addMConstr(matrix, x, GRB.EQUAL, np.zeros(hours), name="soc")
    model.ModelSense = GRB.MINIMIZE
    model.setObjectiveN(
        -x[hours : 2 * hours].sum(),
        index=0,
        priority=2,
        weight=1.0,
        abstol=PRIMARY_OBJECTIVE_ABS_TOLERANCE,
        reltol=0.0,
        name="maximize_task_supply",
    )
    model.setObjectiveN(
        x[:hours].sum(),
        index=1,
        priority=1,
        weight=1.0,
        abstol=1.0e-8,
        reltol=0.0,
        name="minimize_charge_tiebreak",
    )
    model.optimize()
    status = int(model.Status)
    if status != GRB.OPTIMAL:
        result = {
            "solver_status": status,
            "solver_status_name": {
                GRB.INFEASIBLE: "infeasible",
                GRB.INF_OR_UNBD: "infeasible_or_unbounded",
                GRB.UNBOUNDED: "unbounded",
                GRB.NUMERIC: "numeric",
            }.get(status, f"status_{status}"),
            "solver_runtime_seconds": float(model.Runtime),
        }
        model.dispose()
        return result

    values = x.X
    charge = values[:hours]
    discharge = values[hours : 2 * hours]
    state = values[2 * hours :]
    previous_state = np.roll(state, 1)
    residual = state - retention * previous_state - eta * charge + discharge / eta
    charge_total = float(charge.sum())
    discharge_total = float(discharge.sum())
    charge_saturated = int(
        np.sum(
            (charge_power_ratio > NUMERICAL_ZERO)
            & (charge >= charge_power_ratio - 1.0e-7)
        )
    )
    discharge_saturated = int(np.sum(discharge >= 1.0 - 1.0e-7))
    overlap_hours = int(
        np.sum(
            (task_norm > NUMERICAL_ZERO)
            & (recharge_norm > NUMERICAL_ZERO)
        )
    )
    result = {
        "charge_norm_mwh_per_mw": charge_total,
        "discharge_norm_mwh_per_mw": discharge_total,
        "unserved_norm_mwh_per_mw": float(task_norm.sum()) - discharge_total,
        "spill_norm_mwh_per_mw": float(recharge_norm.sum()) - charge_total,
        "charge_saturation_hours": charge_saturated,
        "discharge_saturation_hours": discharge_saturated,
        "usable_soc_min_norm_mwh_per_mw": float(state.min()),
        "usable_soc_max_norm_mwh_per_mw": float(state.max()),
        "cyclic_soc_residual_norm_mwh_per_mw": float(
            state[0]
            - retention * state[-1]
            - eta * charge[0]
            + discharge[0] / eta
        ),
        "maximum_soc_balance_residual_norm_mwh_per_mw": float(
            np.max(np.abs(residual))
        ),
        "task_recharge_overlap_hours": overlap_hours,
        "solver_status": status,
        "solver_status_name": "optimal",
        "solver_runtime_seconds": float(model.Runtime),
        "solver_iterations": float(model.IterCount),
        "solver_barrier_iterations": int(model.BarIterCount),
        "solver_node_count": float(model.NodeCount),
        "solver_variables": int(model.NumVars),
        "solver_constraints": int(model.NumConstrs),
        "solver_binary_variables": int(model.NumBinVars),
        "solver_integer_variables": int(model.NumIntVars),
        "primary_task_supply_objective_norm_mwh_per_mw": discharge_total,
        "secondary_charge_objective_norm_mwh_per_mw": charge_total,
    }
    model.dispose()
    return result


def equivalent_firm_capacity(
    task_mw: np.ndarray, target_unserved_mwh: float
) -> float:
    total = float(task_mw.sum())
    if target_unserved_mwh >= total - 1.0e-10:
        return 0.0
    low = 0.0
    high = float(task_mw.max())
    for _ in range(70):
        midpoint = 0.5 * (low + high)
        unserved = float(np.maximum(task_mw - midpoint, 0.0).sum())
        if unserved > target_unserved_mwh:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def metrics_from_normalized(
    raw: dict[str, Any],
    task_mw: np.ndarray,
    recharge_mw: np.ndarray,
    pdis_mw: float,
    config: Any,
    scenario: dict[str, float],
    dispatch_mode: str,
) -> dict[str, Any]:
    if raw.get("solver_status_name", "optimal") != "optimal":
        return {
            "dispatch_mode": dispatch_mode,
            **raw,
        }
    charge_norm = float(raw["charge_norm_mwh_per_mw"])
    discharge_norm = float(raw["discharge_norm_mwh_per_mw"])
    unserved_norm = float(raw["unserved_norm_mwh_per_mw"])
    spill_norm = float(raw["spill_norm_mwh_per_mw"])
    charge_mwh = charge_norm * pdis_mw
    discharge_mwh = discharge_norm * pdis_mw
    unserved_mwh = unserved_norm * pdis_mw
    spill_mwh = spill_norm * pdis_mw
    duration_h = float(config.duration_h)
    task_total = float(task_mw.sum())
    efc_mw = equivalent_firm_capacity(task_mw, unserved_mwh)
    fixed_cost = float(
        config.annual_fixed_cost_eur_per_kw_year_source_aware
    )
    variable_om = float(config.variable_om_eur_mwh_source_aware)
    energy_component = (
        discharge_mwh
        * scenario["energy_value_eur_mwh"]
        / (pdis_mw * 1000.0)
    )
    firm_component = (
        efc_mw
        / pdis_mw
        * scenario["firm_value_eur_kw_year"]
    )
    vom_component = discharge_mwh * variable_om / (pdis_mw * 1000.0)
    charge_component = (
        charge_mwh
        * scenario["charge_price_eur_mwh"]
        / (pdis_mw * 1000.0)
    )
    usable_min = float(raw["usable_soc_min_norm_mwh_per_mw"])
    usable_max = float(raw["usable_soc_max_norm_mwh_per_mw"])
    return {
        "dispatch_mode": dispatch_mode,
        "pdis_mw": pdis_mw,
        "energy_capacity_mwh": duration_h * pdis_mw,
        "usable_energy_capacity_mwh": (
            USABLE_SOC_FRACTION * duration_h * pdis_mw
        ),
        "task_mwh": task_total,
        "recharge_available_mwh": float(recharge_mw.sum()),
        "charge_mwh": charge_mwh,
        "discharge_mwh": discharge_mwh,
        "unserved_mwh": unserved_mwh,
        "spill_mwh": spill_mwh,
        "served_share": discharge_mwh / task_total if task_total > 0.0 else 0.0,
        "equivalent_full_cycles": discharge_mwh / (duration_h * pdis_mw),
        "efc_mw": efc_mw,
        "efc_ratio": efc_mw / pdis_mw,
        "charge_saturation_hours": int(raw["charge_saturation_hours"]),
        "discharge_saturation_hours": int(raw["discharge_saturation_hours"]),
        "soc_min_fraction": SOC_MIN + usable_min / duration_h,
        "soc_max_fraction": SOC_MIN + usable_max / duration_h,
        "cyclic_soc_residual_mwh": (
            float(raw["cyclic_soc_residual_norm_mwh_per_mw"]) * pdis_mw
        ),
        "maximum_soc_balance_residual_mwh": (
            float(
                raw.get(
                    "maximum_soc_balance_residual_norm_mwh_per_mw",
                    abs(raw["cyclic_soc_residual_norm_mwh_per_mw"]),
                )
            )
            * pdis_mw
        ),
        "task_recharge_overlap_hours": int(
            raw["task_recharge_overlap_hours"]
        ),
        "scenario_id": SCENARIO_ID,
        "energy_value_eur_mwh": scenario["energy_value_eur_mwh"],
        "firm_value_eur_kw_year": scenario["firm_value_eur_kw_year"],
        "charge_price_eur_mwh": scenario["charge_price_eur_mwh"],
        "annual_fixed_cost_eur_per_kw_year_source_aware": fixed_cost,
        "storage_vom_eur_mwh_source_aware": variable_om,
        "energy_value_component_eur_per_kw_year": energy_component,
        "firm_value_component_eur_per_kw_year": firm_component,
        "storage_vom_component_eur_per_kw_year": vom_component,
        "charge_cost_component_eur_per_kw_year": charge_component,
        "scenario31_net_benefit_eur_per_kw_year": (
            energy_component
            + firm_component
            - fixed_cost
            - vom_component
            - charge_component
        ),
        **{
            key: value
            for key, value in raw.items()
            if key.startswith("solver_")
            or key.startswith("primary_")
            or key.startswith("secondary_")
            or key == "greedy_cyclic_iterations"
        },
    }


def matched_greedy_metrics(
    task_norm: np.ndarray,
    recharge_norm: np.ndarray,
    duration_h: float,
    rte: float,
    self_discharge_pct_day: float,
    charge_power_ratio: float,
) -> dict[str, Any]:
    values = greedy_dispatch_usable_soc(
        task_norm,
        recharge_norm,
        duration_h,
        rte,
        self_discharge_pct_day,
        charge_power_ratio,
    )
    return {
        "charge_norm_mwh_per_mw": float(values[0]),
        "discharge_norm_mwh_per_mw": float(values[1]),
        "unserved_norm_mwh_per_mw": float(values[2]),
        "spill_norm_mwh_per_mw": float(values[3]),
        "charge_saturation_hours": int(values[4]),
        "discharge_saturation_hours": int(values[5]),
        "usable_soc_min_norm_mwh_per_mw": float(values[6]),
        "usable_soc_max_norm_mwh_per_mw": float(values[7]),
        "cyclic_soc_residual_norm_mwh_per_mw": float(values[8]),
        "maximum_soc_balance_residual_norm_mwh_per_mw": abs(float(values[8])),
        "task_recharge_overlap_hours": int(values[9]),
        "greedy_cyclic_iterations": int(values[10]),
        "solver_status": GRB.OPTIMAL,
        "solver_status_name": "optimal",
    }


def base_result_record(
    country: str,
    year: int,
    chronology: str,
    config: Any,
) -> dict[str, Any]:
    return {
        "country": country,
        "year": int(year),
        "chronology": chronology,
        "config_id": str(config.config_id),
        "tech_id": str(config.tech_id),
        "technology_en": str(config.technology_en),
        "duration_h": float(config.duration_h),
        "rte": float(config.rte),
        "charge_power_ratio": float(config.charge_power_ratio),
        "self_discharge_pct_day": float(config.self_discharge_pct_day),
        "released_library_configuration": bool(
            config.released_library_configuration
        ),
        "direct_duration_anchor": bool(config.direct_duration_anchor),
        "structural_extrapolation": bool(config.structural_extrapolation),
        "technology_currency_route": str(config.technology_currency_route),
    }


def run_country_year(
    frame: pd.DataFrame,
    catalogue: pd.DataFrame,
    chronologies: list[str],
    scenario: dict[str, float],
    env: gp.Env,
    threads: int,
    method: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    country = str(frame["country"].iloc[0])
    year = int(frame["year"].iloc[0])
    if len(frame) != 8760:
        raise RuntimeError(f"{country}-{year} has {len(frame)} hours")
    orders, order_stats = day_orders(frame)
    task_original = frame[TASK_COLUMN].to_numpy(float)
    recharge_original = frame[RECHARGE_COLUMN].to_numpy(float)
    pdis_mw = FORCED_POWER_SHARE * float(frame["raw_gas_mean_mw"].iloc[0])
    if pdis_mw <= 0.0:
        raise RuntimeError(f"{country}-{year} has non-positive forced power")

    perfect_rows: list[dict[str, Any]] = []
    greedy_rows: list[dict[str, Any]] = []
    chronology_rows: list[dict[str, Any]] = []
    for chronology in chronologies:
        order = orders[chronology]
        task = reorder_days(task_original, order)
        recharge = reorder_days(recharge_original, order)
        task_norm = task / pdis_mw
        recharge_norm = recharge / pdis_mw
        chronology_rows.append(
            {
                "country": country,
                "year": year,
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
            base = base_result_record(country, year, chronology, config)
            greedy_raw = matched_greedy_metrics(
                task_norm,
                recharge_norm,
                float(config.duration_h),
                float(config.rte),
                float(config.self_discharge_pct_day),
                float(config.charge_power_ratio),
            )
            greedy_rows.append(
                {
                    **base,
                    **metrics_from_normalized(
                        greedy_raw,
                        task,
                        recharge,
                        pdis_mw,
                        config,
                        scenario,
                        "matched_greedy",
                    ),
                }
            )
            perfect_raw = solve_perfect_foresight_lp(
                task_norm,
                recharge_norm,
                float(config.duration_h),
                float(config.rte),
                float(config.self_discharge_pct_day),
                float(config.charge_power_ratio),
                env,
                threads,
                method,
            )
            perfect_rows.append(
                {
                    **base,
                    **metrics_from_normalized(
                        perfect_raw,
                        task,
                        recharge,
                        pdis_mw,
                        config,
                        scenario,
                        "perfect_foresight",
                    ),
                }
            )
    return (
        pd.DataFrame(perfect_rows),
        pd.DataFrame(greedy_rows),
        pd.DataFrame(chronology_rows),
    )


def prefix_metrics(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keys = ["country", "year", "chronology", "config_id"]
    return frame.rename(
        columns={
            column: f"{prefix}{column}"
            for column in frame.columns
            if column not in keys
        }
    )


def scenario31_for_frozen_v40(
    frozen: pd.DataFrame, scenario: dict[str, float]
) -> pd.DataFrame:
    frame = frozen.copy()
    frame["dispatch_mode"] = "frozen_v40_heuristic"
    frame["scenario_id"] = SCENARIO_ID
    frame["energy_value_eur_mwh"] = scenario["energy_value_eur_mwh"]
    frame["firm_value_eur_kw_year"] = scenario["firm_value_eur_kw_year"]
    frame["charge_price_eur_mwh"] = scenario["charge_price_eur_mwh"]
    frame["storage_vom_eur_mwh_source_aware"] = np.divide(
        frame["storage_vom_eur_per_kw_year_source_aware"]
        * frame["pdis_mw"]
        * 1000.0,
        frame["discharge_mwh"],
        out=np.zeros(len(frame), dtype=float),
        where=frame["discharge_mwh"].to_numpy(float) > 0.0,
    )
    frame["energy_value_component_eur_per_kw_year"] = (
        frame["discharge_mwh"]
        * scenario["energy_value_eur_mwh"]
        / (frame["pdis_mw"] * 1000.0)
    )
    frame["firm_value_component_eur_per_kw_year"] = (
        frame["efc_mw"]
        / frame["pdis_mw"]
        * scenario["firm_value_eur_kw_year"]
    )
    frame["storage_vom_component_eur_per_kw_year"] = (
        frame["storage_vom_eur_per_kw_year_source_aware"]
    )
    frame["charge_cost_component_eur_per_kw_year"] = (
        frame["charge_mwh"]
        * scenario["charge_price_eur_mwh"]
        / (frame["pdis_mw"] * 1000.0)
    )
    frame["scenario31_net_benefit_eur_per_kw_year"] = (
        frame["energy_value_component_eur_per_kw_year"]
        + frame["firm_value_component_eur_per_kw_year"]
        - frame["annual_fixed_cost_eur_per_kw_year_source_aware"]
        - frame["storage_vom_component_eur_per_kw_year"]
        - frame["charge_cost_component_eur_per_kw_year"]
    )
    return frame


def build_comparison(
    perfect: pd.DataFrame,
    greedy: pd.DataFrame,
    frozen: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["country", "year", "chronology", "config_id"]
    shared = [
        "tech_id",
        "technology_en",
        "duration_h",
        "pdis_mw",
        "charge_mwh",
        "discharge_mwh",
        "unserved_mwh",
        "served_share",
        "equivalent_full_cycles",
        "efc_mw",
        "efc_ratio",
        "soc_min_fraction",
        "soc_max_fraction",
        "cyclic_soc_residual_mwh",
        "scenario31_net_benefit_eur_per_kw_year",
    ]
    left = prefix_metrics(greedy[keys + shared], "matched_")
    right = prefix_metrics(perfect[keys + shared], "perfect_")
    comparison = left.merge(
        right, on=keys, how="inner", validate="one_to_one"
    )
    frozen_columns = keys + [
        column
        for column in shared
        if column in frozen.columns
    ]
    comparison = comparison.merge(
        prefix_metrics(frozen[frozen_columns], "frozen_v40_"),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    for identity in ["tech_id", "technology_en", "duration_h", "pdis_mw"]:
        matched = f"matched_{identity}"
        perfect_name = f"perfect_{identity}"
        if not np.all(
            comparison[matched].astype(str).to_numpy()
            == comparison[perfect_name].astype(str).to_numpy()
        ):
            raise RuntimeError(f"Matched/perfect identity mismatch: {identity}")
        comparison[identity] = comparison[matched]
        comparison = comparison.drop(columns=[matched, perfect_name])
    for metric in [
        "charge_mwh",
        "discharge_mwh",
        "unserved_mwh",
        "served_share",
        "equivalent_full_cycles",
        "efc_mw",
        "efc_ratio",
        "scenario31_net_benefit_eur_per_kw_year",
    ]:
        comparison[f"perfect_minus_matched_{metric}"] = (
            comparison[f"perfect_{metric}"] - comparison[f"matched_{metric}"]
        )
    comparison["perfect_service_ge_matched"] = (
        comparison["perfect_discharge_mwh"]
        + 1.0e-6
        >= comparison["matched_discharge_mwh"]
    )
    if "frozen_v40_discharge_mwh" in comparison:
        comparison[
            "matched_minus_frozen_v40_discharge_mwh"
        ] = (
            comparison["matched_discharge_mwh"]
            - comparison["frozen_v40_discharge_mwh"]
        )
        comparison[
            "matched_minus_frozen_v40_scenario31_net_benefit_eur_per_kw_year"
        ] = (
            comparison[
                "matched_scenario31_net_benefit_eur_per_kw_year"
            ]
            - comparison[
                "frozen_v40_scenario31_net_benefit_eur_per_kw_year"
            ]
        )
    return comparison


def build_envelope(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "country",
        "year",
        "chronology",
        "dispatch_mode",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
        "charge_mwh",
        "discharge_mwh",
        "efc_mw",
        "annual_fixed_cost_eur_per_kw_year_source_aware",
        "energy_value_component_eur_per_kw_year",
        "firm_value_component_eur_per_kw_year",
        "storage_vom_component_eur_per_kw_year",
        "charge_cost_component_eur_per_kw_year",
        "scenario31_net_benefit_eur_per_kw_year",
    ]
    combined = pd.concat(
        [frame[columns] for frame in frames],
        ignore_index=True,
    )
    combined = combined[
        combined["duration_h"].le(ENVELOPE_MAX_DURATION_H)
    ].copy()
    sort_columns = [
        "dispatch_mode",
        "country",
        "year",
        "chronology",
        "duration_h",
        "scenario31_net_benefit_eur_per_kw_year",
        "config_id",
    ]
    envelope = (
        combined.sort_values(
            sort_columns,
            ascending=[True, True, True, True, True, False, True],
            kind="stable",
        )
        .drop_duplicates(
            [
                "dispatch_mode",
                "country",
                "year",
                "chronology",
                "duration_h",
            ]
        )
        .sort_values(
            [
                "dispatch_mode",
                "country",
                "year",
                "chronology",
                "duration_h",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return envelope


def strict_valley_records(envelope: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-specified positive/non-positive/positive criterion."""
    records: list[dict[str, Any]] = []
    group_columns = ["dispatch_mode", "country", "year", "chronology"]
    for keys, group in envelope.groupby(group_columns, sort=False):
        ordered = group.sort_values("duration_h", kind="stable")
        durations = ordered["duration_h"].to_numpy(float)
        values = ordered[
            "scenario31_net_benefit_eur_per_kw_year"
        ].to_numpy(float)
        positive = values > SIGN_THRESHOLD_EUR_PER_KW_YEAR
        candidates: list[dict[str, Any]] = []
        position = 0
        while position < len(positive):
            if positive[position]:
                position += 1
                continue
            start = position
            while position + 1 < len(positive) and not positive[position + 1]:
                position += 1
            end = position
            if start > 0 and end < len(positive) - 1:
                candidates.append(
                    {
                        "gap_width_h": float(
                            durations[end + 1] - durations[start - 1]
                        ),
                        "left_positive_duration_h": float(durations[start - 1]),
                        "nonpositive_start_duration_h": float(durations[start]),
                        "nonpositive_end_duration_h": float(durations[end]),
                        "right_reentry_duration_h": float(durations[end + 1]),
                        "left_positive_net_eur_per_kw_year": float(
                            values[start - 1]
                        ),
                        "minimum_gap_net_eur_per_kw_year": float(
                            np.min(values[start : end + 1])
                        ),
                        "right_positive_net_eur_per_kw_year": float(
                            values[end + 1]
                        ),
                        "nonpositive_duration_nodes": int(end - start + 1),
                    }
                )
            position += 1
        if candidates:
            chosen = sorted(
                candidates,
                key=lambda item: (
                    -item["gap_width_h"],
                    item["left_positive_duration_h"],
                ),
            )[0]
        else:
            chosen = {
                "gap_width_h": 0.0,
                "left_positive_duration_h": np.nan,
                "nonpositive_start_duration_h": np.nan,
                "nonpositive_end_duration_h": np.nan,
                "right_reentry_duration_h": np.nan,
                "left_positive_net_eur_per_kw_year": np.nan,
                "minimum_gap_net_eur_per_kw_year": np.nan,
                "right_positive_net_eur_per_kw_year": np.nan,
                "nonpositive_duration_nodes": 0,
            }
        records.append(
            {
                "dispatch_mode": keys[0],
                "country": keys[1],
                "year": int(keys[2]),
                "chronology": keys[3],
                "scenario_id": SCENARIO_ID,
                "sign_threshold_eur_per_kw_year": (
                    SIGN_THRESHOLD_EUR_PER_KW_YEAR
                ),
                "has_strict_valley": bool(candidates),
                "candidate_nonpositive_intervals": len(candidates),
                **chosen,
            }
        )
    return pd.DataFrame(records)


def build_valley_pairing(
    valley: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = ["dispatch_mode", "country", "year"]
    dispersed = (
        valley[valley["chronology"].eq("dispersed")]
        .set_index(index)
        .add_prefix("dispersed_")
    )
    clustered = (
        valley[valley["chronology"].eq("clustered")]
        .set_index(index)
        .add_prefix("clustered_")
    )
    paired = dispersed.join(
        clustered, how="inner", validate="one_to_one"
    ).reset_index()
    paired["clustering_induced_valley_formation"] = (
        ~paired["dispersed_has_strict_valley"].astype(bool)
        & paired["clustered_has_strict_valley"].astype(bool)
    )
    paired["clustered_minus_dispersed_gap_width_h"] = (
        paired["clustered_gap_width_h"] - paired["dispersed_gap_width_h"]
    )
    summary = (
        paired.groupby("dispatch_mode", as_index=False)
        .agg(
            country_years=("country", "size"),
            dispersed_strict_valleys=("dispersed_has_strict_valley", "sum"),
            clustered_strict_valleys=("clustered_has_strict_valley", "sum"),
            clustering_induced_formations=(
                "clustering_induced_valley_formation",
                "sum",
            ),
            dispersed_mean_width_h=("dispersed_gap_width_h", "mean"),
            clustered_mean_width_h=("clustered_gap_width_h", "mean"),
            clustered_median_width_h=("clustered_gap_width_h", "median"),
            mean_width_change_h=(
                "clustered_minus_dispersed_gap_width_h",
                "mean",
            ),
        )
    )
    return paired, summary


def load_scenario(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    subset = frame[frame["scenario_id"].eq(SCENARIO_ID)]
    if set(subset["chronology"]) != set(EXPECTED_CHRONOLOGIES):
        raise RuntimeError("Scenario 31 must contain both frozen chronologies")
    columns = [
        "energy_value_eur_mwh",
        "firm_value_eur_kw_year",
        "charge_price_eur_mwh",
    ]
    unique = subset[columns].drop_duplicates()
    if len(unique) != 1:
        raise RuntimeError("Scenario 31 value parameters are not unique")
    return {
        column: float(unique.iloc[0][column])
        for column in columns
    }


def output_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


def concatenate_checkpoints(
    checkpoint_dir: Path,
    country_years: list[tuple[str, int]],
    suffix: str,
) -> pd.DataFrame:
    frames = []
    for country, year in country_years:
        path = checkpoint_dir / f"{country}_{year}_{suffix}.csv.gz"
        if not path.exists():
            raise RuntimeError(f"Missing checkpoint: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def select_panel(
    panel: pd.DataFrame,
    requested: list[tuple[str, int]] | None,
    max_country_years: int | None,
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    available = [
        (str(row.country), int(row.year))
        for row in (
            panel[["country", "year"]]
            .drop_duplicates()
            .sort_values(["country", "year"])
            .itertuples(index=False)
        )
    ]
    if requested is None:
        chosen = available
    else:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise RuntimeError(f"Requested country-years are absent: {missing}")
        chosen = requested
    if max_country_years is not None:
        if max_country_years <= 0:
            raise ValueError("--max-country-years must be positive")
        chosen = chosen[:max_country_years]
    chosen_set = set(chosen)
    selected = panel[
        [
            (str(country), int(year)) in chosen_set
            for country, year in zip(panel["country"], panel["year"])
        ]
    ].copy()
    return selected, chosen


def make_qa(
    perfect: pd.DataFrame,
    greedy: pd.DataFrame,
    frozen: pd.DataFrame,
    comparison: pd.DataFrame,
    chronology: pd.DataFrame,
    envelope: pd.DataFrame,
    valley: pd.DataFrame,
    valley_summary: pd.DataFrame,
    country_years: list[tuple[str, int]],
    catalogue: pd.DataFrame,
    chronologies: list[str],
    scenario: dict[str, float],
    elapsed_seconds: float,
) -> dict[str, Any]:
    expected_rows = (
        len(country_years) * len(catalogue) * len(chronologies)
    )
    full_sample = (
        len(country_years) == EXPECTED_COUNTRY_YEARS
        and len(catalogue) == EXPECTED_CONFIGURATIONS
        and tuple(chronologies) == EXPECTED_CHRONOLOGIES
    )
    optimal = perfect["solver_status_name"].eq("optimal")
    gates = {
        "expected_perfect_rows": len(perfect) == expected_rows,
        "expected_matched_greedy_rows": len(greedy) == expected_rows,
        "unique_perfect_keys": not perfect.duplicated(
            ["country", "year", "chronology", "config_id"]
        ).any(),
        "all_lp_models_optimal": bool(optimal.all()),
        "all_lp_models_continuous": bool(
            perfect.loc[optimal, "solver_binary_variables"].eq(0).all()
            and perfect.loc[optimal, "solver_integer_variables"].eq(0).all()
        ),
        "lp_soc_balance_within_tolerance": bool(
            perfect.loc[
                optimal, "maximum_soc_balance_residual_mwh"
            ].abs().max()
            <= 1.0e-5
        ),
        "greedy_cyclic_residual_within_tolerance": bool(
            greedy["cyclic_soc_residual_mwh"].abs().max() <= 1.0e-5
        ),
        "perfect_service_weakly_exceeds_matched_greedy": bool(
            comparison["perfect_service_ge_matched"].all()
        ),
        "task_conserved_under_reordering": bool(
            chronology["task_original_conservation_error_mwh"].abs().max()
            <= 1.0e-6
        ),
        "recharge_conserved_under_reordering": bool(
            chronology[
                "recharge_original_conservation_error_mwh"
            ].abs().max()
            <= 1.0e-6
        ),
        "scenario31_zero_charge_price": bool(
            np.isclose(scenario["charge_price_eur_mwh"], 0.0)
        ),
        "strict_valley_sign_threshold_is_zero": (
            SIGN_THRESHOLD_EUR_PER_KW_YEAR == 0.0
        ),
        "envelope_limited_to_prespecified_168h_domain": bool(
            envelope["duration_h"].max() <= ENVELOPE_MAX_DURATION_H
        ),
        "frozen_v40_comparator_complete": len(frozen) == expected_rows,
    }
    status = "pass" if all(gates.values()) else "fail"
    return {
        "status": status,
        "created_utc": utc_now(),
        "run_scope": {
            "country_years": len(country_years),
            "countries": len({country for country, _ in country_years}),
            "configurations": len(catalogue),
            "chronologies": chronologies,
            "expected_result_rows_per_dispatch": expected_rows,
            "full_confirmatory_sample": full_sample,
        },
        "frozen_design": {
            "task_column": TASK_COLUMN,
            "recharge_column": RECHARGE_COLUMN,
            "cluster_days": CLUSTER_DAYS,
            "forced_power_share_of_mean_gas": FORCED_POWER_SHARE,
            "soc_min": SOC_MIN,
            "soc_max": SOC_MAX,
            "envelope_max_duration_h": ENVELOPE_MAX_DURATION_H,
            "strict_positive_threshold_eur_per_kw_year": (
                SIGN_THRESHOLD_EUR_PER_KW_YEAR
            ),
            "scenario_id": SCENARIO_ID,
            **scenario,
        },
        "model_contract": {
            "solver": "Gurobi continuous LP",
            "state": "cyclic usable SOC above the frozen 5% minimum",
            "primary_objective": "maximize annual peaking-task supply",
            "secondary_objective": (
                "minimize annual charging energy without materially "
                "degrading primary task supply"
            ),
            "charging_in_positive_task_hours": "forbidden",
            "charge_and_discharge_simultaneity": (
                "impossible because their hourly upper-bound supports "
                "are disjoint"
            ),
            "capacity_optimization": False,
            "technology_choice_optimization": False,
            "parameter_fitting": False,
        },
        "gates": gates,
        "maximums": {
            "lp_soc_balance_residual_mwh": float(
                perfect.loc[
                    optimal, "maximum_soc_balance_residual_mwh"
                ].abs().max()
            ),
            "lp_cyclic_soc_residual_mwh": float(
                perfect.loc[
                    optimal, "cyclic_soc_residual_mwh"
                ].abs().max()
            ),
            "greedy_cyclic_soc_residual_mwh": float(
                greedy["cyclic_soc_residual_mwh"].abs().max()
            ),
            "task_conservation_error_mwh": float(
                chronology[
                    "task_original_conservation_error_mwh"
                ].abs().max()
            ),
            "recharge_conservation_error_mwh": float(
                chronology[
                    "recharge_original_conservation_error_mwh"
                ].abs().max()
            ),
            "perfect_minus_matched_discharge_mwh": float(
                comparison[
                    "perfect_minus_matched_discharge_mwh"
                ].max()
            ),
            "matched_minus_frozen_v40_abs_discharge_mwh": float(
                comparison[
                    "matched_minus_frozen_v40_discharge_mwh"
                ].abs().max()
            ),
        },
        "minimum_absolute_envelope_net_value_eur_per_kw_year": float(
            envelope[
                "scenario31_net_benefit_eur_per_kw_year"
            ].abs().min()
        ),
        "valley_summary": valley_summary.to_dict(orient="records"),
        "elapsed_seconds": elapsed_seconds,
        "evidence_status": (
            "full_confirmatory_run_complete_pending_independent_audit"
            if full_sample and status == "pass"
            else "smoke_or_partial_run_not_manuscript_evidence"
        ),
        "manuscript_evidence_admitted": False,
        "admission_note": (
            "The runner does not self-promote. Full-sample outputs require "
            "an independent audit before manuscript use."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument(
        "--scenario-summary", type=Path, default=DEFAULT_SCENARIO_SUMMARY
    )
    parser.add_argument(
        "--frozen-v40-results",
        type=Path,
        default=DEFAULT_FROZEN_V40_RESULTS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--country-years",
        type=str,
        default=None,
        help="Optional comma list such as DE:2023,SI:2018",
    )
    parser.add_argument("--max-country-years", type=int, default=None)
    parser.add_argument(
        "--chronologies",
        type=str,
        default="dispersed,clustered",
    )
    parser.add_argument(
        "--config-ids",
        type=str,
        default=None,
        help="Optional comma-separated subset for diagnostics only",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--method",
        type=int,
        default=1,
        choices=[-1, 0, 1, 2, 3, 4, 5],
        help="Gurobi LP method; default 1 is dual simplex",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    chronologies = parse_csv_list(args.chronologies)
    if any(item not in EXPECTED_CHRONOLOGIES for item in chronologies):
        raise ValueError(
            f"Chronologies must be drawn from {EXPECTED_CHRONOLOGIES}"
        )
    requested_country_years = parse_country_years(args.country_years)
    scenario = load_scenario(args.scenario_summary)

    usecols = [
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
    panel, country_years = select_panel(
        panel, requested_country_years, args.max_country_years
    )
    catalogue = pd.read_csv(args.catalogue)
    if args.config_ids:
        requested_configs = parse_csv_list(args.config_ids)
        missing = sorted(set(requested_configs) - set(catalogue["config_id"]))
        if missing:
            raise RuntimeError(f"Unknown configuration IDs: {missing}")
        catalogue = catalogue[
            catalogue["config_id"].isin(requested_configs)
        ].copy()
    catalogue = catalogue.sort_values(
        ["duration_h", "tech_id"], kind="stable"
    ).reset_index(drop=True)
    if len(catalogue) == 0:
        raise RuntimeError("No configurations selected")

    contract = {
        "runner_sha256": sha256(Path(__file__).resolve()),
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256(args.panel),
        "catalogue": str(args.catalogue.resolve()),
        "catalogue_sha256": sha256(args.catalogue),
        "scenario_summary": str(args.scenario_summary.resolve()),
        "scenario_summary_sha256": sha256(args.scenario_summary),
        "frozen_v40_results": str(args.frozen_v40_results.resolve()),
        "frozen_v40_results_sha256": sha256(args.frozen_v40_results),
        "country_years": country_years,
        "config_ids": catalogue["config_id"].tolist(),
        "chronologies": chronologies,
        "scenario": scenario,
        "threads": args.threads,
        "method": args.method,
        "soc_accounting": "usable_inventory_above_5pct_floor",
        "primary_objective_abs_tolerance": (
            PRIMARY_OBJECTIVE_ABS_TOLERANCE
        ),
    }
    contract_path = args.output / "run_contract.json"
    if args.resume and contract_path.exists():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != json_ready(contract):
            raise RuntimeError(
                "Resume contract differs from the existing run contract"
            )
    else:
        write_json(contract_path, contract)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    completed = 0
    for country, year in country_years:
        perfect_path = (
            checkpoint_dir / f"{country}_{year}_perfect.csv.gz"
        )
        greedy_path = checkpoint_dir / f"{country}_{year}_greedy.csv.gz"
        chronology_path = (
            checkpoint_dir / f"{country}_{year}_chronology.csv.gz"
        )
        if (
            args.resume
            and perfect_path.exists()
            and greedy_path.exists()
            and chronology_path.exists()
        ):
            completed += 1
            print(
                f"[resume {completed}/{len(country_years)}] "
                f"{country}-{year}",
                flush=True,
            )
            continue
        frame = panel[
            panel["country"].eq(country) & panel["year"].eq(year)
        ].copy()
        cy_started = time.perf_counter()
        perfect_cy, greedy_cy, chronology_cy = run_country_year(
            frame,
            catalogue,
            chronologies,
            scenario,
            env,
            args.threads,
            args.method,
        )
        output_csv_gz(perfect_cy, perfect_path)
        output_csv_gz(greedy_cy, greedy_path)
        output_csv_gz(chronology_cy, chronology_path)
        completed += 1
        elapsed = time.perf_counter() - cy_started
        print(
            f"[solve {completed}/{len(country_years)}] {country}-{year}: "
            f"{len(perfect_cy)} LPs in {elapsed:.2f} s",
            flush=True,
        )
    env.dispose()

    perfect = concatenate_checkpoints(
        checkpoint_dir, country_years, "perfect"
    )
    greedy = concatenate_checkpoints(
        checkpoint_dir, country_years, "greedy"
    )
    chronology = concatenate_checkpoints(
        checkpoint_dir, country_years, "chronology"
    )
    frozen = pd.read_csv(args.frozen_v40_results)
    chosen_country_years = pd.DataFrame(
        country_years, columns=["country", "year"]
    )
    frozen = frozen.merge(
        chosen_country_years,
        on=["country", "year"],
        how="inner",
        validate="many_to_one",
    )
    frozen = frozen[
        frozen["chronology"].isin(chronologies)
        & frozen["config_id"].isin(catalogue["config_id"])
    ].copy()
    frozen = scenario31_for_frozen_v40(frozen, scenario)

    comparison = build_comparison(perfect, greedy, frozen)
    envelope = build_envelope([frozen, greedy, perfect])
    valley = strict_valley_records(envelope)
    valley_paired, valley_summary = build_valley_pairing(valley)

    output_csv_gz(
        perfect,
        args.output / "configuration_results_perfect_foresight.csv.gz",
    )
    output_csv_gz(
        greedy,
        args.output / "configuration_results_matched_greedy.csv.gz",
    )
    output_csv_gz(
        comparison,
        args.output / "heuristic_vs_perfect_foresight.csv.gz",
    )
    chronology.to_csv(
        args.output / "chronology_conservation.csv", index=False
    )
    output_csv_gz(
        envelope,
        args.output / "scenario31_duration_envelope.csv.gz",
    )
    valley.to_csv(
        args.output / "strict_valley_records.csv", index=False
    )
    valley_paired.to_csv(
        args.output / "strict_valley_paired_formation.csv", index=False
    )
    valley_summary.to_csv(
        args.output / "strict_valley_summary.csv", index=False
    )

    elapsed_seconds = time.perf_counter() - started
    qa = make_qa(
        perfect,
        greedy,
        frozen,
        comparison,
        chronology,
        envelope,
        valley,
        valley_summary,
        country_years,
        catalogue,
        chronologies,
        scenario,
        elapsed_seconds,
    )
    write_json(args.output / "QA.json", qa)
    summary = {
        "status": qa["status"],
        "created_utc": utc_now(),
        "run_scope": qa["run_scope"],
        "scenario": scenario,
        "valley_summary": valley_summary.to_dict(orient="records"),
        "perfect_foresight_mean_service_gain_pp": float(
            100.0
            * comparison["perfect_minus_matched_served_share"].mean()
        ),
        "perfect_foresight_mean_net_benefit_gain_eur_per_kw_year": float(
            comparison[
                "perfect_minus_matched_scenario31_net_benefit_eur_per_kw_year"
            ].mean()
        ),
        "total_gurobi_runtime_seconds": float(
            perfect["solver_runtime_seconds"].sum()
        ),
        "wall_elapsed_seconds": elapsed_seconds,
        "evidence_status": qa["evidence_status"],
        "manuscript_evidence_admitted": False,
    }
    write_json(args.output / "summary.json", summary)

    manifest = {
        "created_utc": utc_now(),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            str(path.resolve()): sha256(path)
            for path in [
                args.panel,
                args.catalogue,
                args.scenario_summary,
                args.frozen_v40_results,
            ]
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "gurobipy": gp.__version__,
            "gurobi": gp.gurobi.version(),
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
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
