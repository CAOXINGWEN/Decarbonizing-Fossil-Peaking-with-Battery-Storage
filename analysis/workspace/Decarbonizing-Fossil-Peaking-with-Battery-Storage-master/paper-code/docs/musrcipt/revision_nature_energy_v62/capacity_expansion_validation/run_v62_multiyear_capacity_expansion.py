#!/usr/bin/env python3
"""Multi-year endogenous storage-capacity expansion for manuscript v62.

The experiment uses the registered European C1 peaking task, C3 event labels
and R2 recharge resource.  For each country, one normalized storage-capacity
vector is shared by every observed natural year and both conserved intact-day
chronologies.  Hourly operations remain scenario specific.  The model jointly
chooses 4 h LFP, 24 h CAES and 168 h cavern-hydrogen capacity, or restricts the
choice to one technology for the single-technology comparators.

The primary objective is annual fixed cost plus a convex combination of mean
scenario operating cost and its 90% conditional value at risk (CVaR).  A zero
risk weight is expected-cost procurement; 0.5 gives equal weight to mean and
upper-tail chronology-year cost.  Planning uses non-overlapping power blocks
while retaining the original hourly energy ledger through an explicit step
length.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
MUSRIPT_ROOT = REVISION_ROOT.parents[0]
WORKSPACE_ROOT = REVISION_ROOT.parents[4]
DATASET_ROOT = (
    WORKSPACE_ROOT
    / "dataset"
    / "external_hourly"
    / "europe_mid_duration_valley_upgrade_v0_1"
)
PANEL_PATH = (
    DATASET_ROOT
    / "processed"
    / "observed_gas_event_task_definitions_v0_2.csv.gz"
)
REGISTERED_RUNNER_PATH = (
    MUSRIPT_ROOT
    / "revision_nature_energy_v44"
    / "portfolio_model"
    / "run_v42_europe_portfolio_gurobi.py"
)
COST_CATALOGUE_PATH = (
    MUSRIPT_ROOT
    / "revision_nature_energy_v43"
    / "cost_and_decomposition"
    / "cost_catalogue_2025_2035_long.csv"
)
VALUE_SCENARIO_PATH = (
    MUSRIPT_ROOT
    / "revision_nature_energy_v40"
    / "currency_work"
    / "value_sensitivity_gap_summary_source_aware_eur.csv"
)

CONFIG_IDS = ("li_lfp_4h", "caes_24h", "h2_cavern_168h")
TASK_COLUMN = "task_c1_mw"
RECHARGE_COLUMN = "r2_potential_recharge_mw"
SOC_MIN = 0.05
SOC_MAX = 0.95
USABLE_SOC_FRACTION = SOC_MAX - SOC_MIN
NUMERICAL_ZERO = 1.0e-10
PRIMARY_ABS_TOLERANCE_EUR = 0.01
CVAR_ALPHA = 0.90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
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


def load_registered_runner():
    specification = importlib.util.spec_from_file_location(
        "v62_registered_chronology_utilities", REGISTERED_RUNNER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot import registered chronology utilities")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def load_value_scenario(scenario_id: int) -> dict[str, float]:
    frame = pd.read_csv(VALUE_SCENARIO_PATH)
    subset = frame.loc[frame["scenario_id"].eq(scenario_id)]
    columns = [
        "energy_value_eur_mwh",
        "firm_value_eur_kw_year",
        "charge_price_eur_mwh",
    ]
    unique = subset[columns].drop_duplicates()
    if len(unique) != 1:
        raise RuntimeError(f"Value scenario {scenario_id} is not unique")
    return {column: float(unique.iloc[0][column]) for column in columns}


def load_technologies(cost_year: int) -> pd.DataFrame:
    frame = pd.read_csv(COST_CATALOGUE_PATH)
    selected = frame.loc[
        frame["learning_year"].eq(cost_year)
        & frame["config_id"].isin(CONFIG_IDS)
    ].copy()
    if len(selected) != len(CONFIG_IDS) or selected["config_id"].duplicated().any():
        raise RuntimeError("The three capacity-expansion configurations are not unique")
    selected["config_id"] = pd.Categorical(
        selected["config_id"], categories=CONFIG_IDS, ordered=True
    )
    selected = selected.sort_values("config_id").reset_index(drop=True)
    selected["fixed_cost_eur_kw_year"] = selected[
        "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"
    ].astype(float)
    selected["vom_eur_mwh"] = selected[
        "variable_om_eur_source_aware_mwh"
    ].astype(float)
    return selected


def aggregate_blocks(values: np.ndarray, step_hours: int, rule: str) -> np.ndarray:
    if len(values) != 8760 or 8760 % step_hours:
        raise RuntimeError("Planning blocks must partition the 8,760-hour year")
    reshaped = np.asarray(values, dtype=float).reshape(-1, step_hours)
    if rule == "mean":
        return reshaped.mean(axis=1)
    if rule == "strict_recharge":
        return reshaped.min(axis=1)
    raise ValueError(rule)


def build_scenarios(
    frame: pd.DataFrame,
    runner,
    step_hours: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenarios: list[dict[str, Any]] = []
    conservation: list[dict[str, Any]] = []
    for year, annual in frame.groupby("year", sort=True):
        annual = annual.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
        if len(annual) != 8760:
            raise RuntimeError(f"Year {year} does not contain 8,760 rows")
        scale_mw = float(annual["raw_gas_mean_mw"].iloc[0])
        if scale_mw <= 0:
            raise RuntimeError("The normalization scale must be positive")
        orders, order_statistics = runner.day_orders(annual)
        task_original = annual[TASK_COLUMN].to_numpy(dtype=float) / scale_mw
        recharge_original = annual[RECHARGE_COLUMN].to_numpy(dtype=float) / scale_mw
        for chronology, order in orders.items():
            task_hourly = runner.reorder_days(task_original, order)
            recharge_hourly = runner.reorder_days(recharge_original, order)
            usable_recharge_hourly = np.where(
                task_hourly > NUMERICAL_ZERO, 0.0, recharge_hourly
            )
            task = aggregate_blocks(task_hourly, step_hours, "mean")
            # A planning block is chargeable only when every constituent hour
            # is outside the peaking task.  This removes sub-block simultaneous
            # charge/discharge opportunities.
            usable_recharge = aggregate_blocks(
                usable_recharge_hourly, step_hours, "strict_recharge"
            )
            scenarios.append(
                {
                    "scenario_id": f"{int(year)}_{chronology}",
                    "year": int(year),
                    "chronology": chronology,
                    "normalization_scale_mw": scale_mw,
                    "task": task,
                    "usable_recharge": usable_recharge,
                }
            )
            conservation.append(
                {
                    "year": int(year),
                    "chronology": chronology,
                    **order_statistics,
                    "task_energy_normalized_mwh": float(
                        task.sum() * step_hours
                    ),
                    "recharge_energy_normalized_mwh": float(
                        recharge_hourly.sum()
                    ),
                    "usable_recharge_energy_normalized_mwh": float(
                        usable_recharge.sum() * step_hours
                    ),
                }
            )
    return scenarios, conservation


def solve_country_design(
    country: str,
    scenarios: list[dict[str, Any]],
    technologies: pd.DataFrame,
    value: dict[str, float],
    risk_weight: float,
    mode: str,
    step_hours: int,
    threads: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "The optional Gurobi implementation requires gurobipy; "
            "the registered workflow uses solve_country_design_highs."
        ) from exc
    if not 0 <= risk_weight <= 1:
        raise ValueError("risk weight must lie in [0, 1]")
    if mode == "diversified":
        active = technologies.copy()
    else:
        active = technologies.loc[technologies["config_id"].astype(str).eq(mode)].copy()
        if len(active) != 1:
            raise RuntimeError(f"Unknown single-technology mode: {mode}")
    active = active.reset_index(drop=True)
    config_ids = active["config_id"].astype(str).tolist()
    technology_count = len(active)
    scenario_count = len(scenarios)
    step_count = len(scenarios[0]["task"])
    probability = 1.0 / scenario_count

    model = gp.Model(f"v62_capacity_expansion_{country}_{mode}_{risk_weight:g}")
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.Method = 2
    model.Params.Crossover = 0
    model.Params.Presolve = 2
    model.Params.FeasibilityTol = 1.0e-8
    model.Params.OptimalityTol = 1.0e-8

    maximum_task = max(float(np.max(item["task"])) for item in scenarios)
    capacity = model.addMVar(
        technology_count,
        lb=0.0,
        ub=maximum_task,
        name="capacity_per_normalized_mw",
    )
    worst_operating_cost = model.addVar(lb=0.0, name="worst_operating_cost")
    fixed_cost = gp.quicksum(
        capacity[index]
        * 1000.0
        * float(active.iloc[index]["fixed_cost_eur_kw_year"])
        for index in range(technology_count)
    )

    operating_cost_expressions: list[gp.LinExpr] = []
    scenario_variables: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        task = np.asarray(scenario["task"], dtype=float)
        recharge = np.asarray(scenario["usable_recharge"], dtype=float)
        charge = model.addMVar(
            (technology_count, step_count), lb=0.0,
            name=f"charge_{scenario_index}"
        )
        discharge = model.addMVar(
            (technology_count, step_count), lb=0.0,
            name=f"discharge_{scenario_index}"
        )
        state = model.addMVar(
            (technology_count, step_count), lb=0.0,
            name=f"state_{scenario_index}"
        )
        unserved = model.addMVar(
            step_count, lb=0.0, ub=task,
            name=f"unserved_{scenario_index}"
        )
        residual_peak = model.addVar(lb=0.0, name=f"residual_peak_{scenario_index}")
        model.addConstr(
            discharge.sum(axis=0) + unserved == task,
            name=f"task_balance_{scenario_index}",
        )
        model.addConstr(
            charge.sum(axis=0) <= recharge,
            name=f"shared_recharge_{scenario_index}",
        )
        model.addConstr(
            residual_peak >= unserved,
            name=f"residual_peak_{scenario_index}",
        )
        for technology_index, row in active.iterrows():
            duration_h = float(row["duration_h"])
            eta = float(np.sqrt(float(row["rte"])))
            retention = float(
                (1.0 - float(row["self_discharge_pct_day"]) / 100.0)
                ** (step_hours / 24.0)
            )
            model.addConstr(
                charge[technology_index, :]
                <= float(row["charge_power_ratio"]) * capacity[technology_index],
                name=f"charge_power_{scenario_index}_{technology_index}",
            )
            model.addConstr(
                discharge[technology_index, :] <= capacity[technology_index],
                name=f"discharge_power_{scenario_index}_{technology_index}",
            )
            model.addConstr(
                state[technology_index, :]
                <= USABLE_SOC_FRACTION * duration_h * capacity[technology_index],
                name=f"energy_capacity_{scenario_index}_{technology_index}",
            )
            model.addConstr(
                state[technology_index, 0]
                == retention * state[technology_index, -1]
                + eta * step_hours * charge[technology_index, 0]
                - step_hours * discharge[technology_index, 0] / eta,
                name=f"cyclic_state_{scenario_index}_{technology_index}",
            )
            model.addConstr(
                state[technology_index, 1:]
                == retention * state[technology_index, :-1]
                + eta * step_hours * charge[technology_index, 1:]
                - step_hours * discharge[technology_index, 1:] / eta,
                name=f"state_balance_{scenario_index}_{technology_index}",
            )
        variable_cost = gp.quicksum(
            float(active.iloc[index]["vom_eur_mwh"])
            * step_hours
            * discharge[index, :].sum()
            for index in range(technology_count)
        )
        charging_cost = (
            value["charge_price_eur_mwh"]
            * step_hours
            * charge.sum()
        )
        residual_energy_cost = (
            value["energy_value_eur_mwh"]
            * step_hours
            * unserved.sum()
        )
        residual_firm_cost = (
            value["firm_value_eur_kw_year"] * 1000.0 * residual_peak
        )
        operating_cost = (
            variable_cost
            + charging_cost
            + residual_energy_cost
            + residual_firm_cost
        )
        operating_cost_expressions.append(operating_cost)
        model.addConstr(
            worst_operating_cost >= operating_cost,
            name=f"worst_cost_{scenario_index}",
        )
        scenario_variables.append(
            {
                "charge": charge,
                "discharge": discharge,
                "state": state,
                "unserved": unserved,
                "residual_peak": residual_peak,
            }
        )

    mean_operating_cost = gp.quicksum(
        probability * expression for expression in operating_cost_expressions
    )
    risk_adjusted_cost = (
        fixed_cost
        + (1.0 - risk_weight) * mean_operating_cost
        + risk_weight * worst_operating_cost
    )
    model.ModelSense = GRB.MINIMIZE
    model.setObjectiveN(
        risk_adjusted_cost,
        index=0,
        priority=2,
        weight=1.0,
        abstol=PRIMARY_ABS_TOLERANCE_EUR,
        reltol=0.0,
        name="risk_adjusted_annual_system_cost",
    )
    model.setObjectiveN(
        capacity.sum(),
        index=1,
        priority=1,
        weight=1.0,
        name="minimum_capacity_tiebreak",
    )
    started = time.perf_counter()
    model.optimize()
    runtime_seconds = time.perf_counter() - started
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"{country}-{mode}-{risk_weight:g} status={model.Status}"
        )

    capacity_values = np.asarray(capacity.X, dtype=float)
    fixed_cost_value = float(sum(
        capacity_values[index]
        * 1000.0
        * float(active.iloc[index]["fixed_cost_eur_kw_year"])
        for index in range(technology_count)
    ))
    capacity_rows: list[dict[str, Any]] = []
    total_capacity = float(capacity_values.sum())
    for index, row in active.iterrows():
        capacity_rows.append(
            {
                "country": country,
                "risk_weight": risk_weight,
                "mode": mode,
                "config_id": str(row["config_id"]),
                "tech_id": str(row["tech_id"]),
                "duration_h": float(row["duration_h"]),
                "capacity_per_normalized_mw": float(capacity_values[index]),
                "capacity_share": (
                    float(capacity_values[index] / total_capacity)
                    if total_capacity > NUMERICAL_ZERO
                    else 0.0
                ),
                "energy_capacity_mwh_per_normalized_mw": float(
                    capacity_values[index] * float(row["duration_h"])
                ),
                "annual_fixed_cost_eur_per_normalized_mw": float(
                    capacity_values[index]
                    * 1000.0
                    * float(row["fixed_cost_eur_kw_year"])
                ),
            }
        )

    scenario_rows: list[dict[str, Any]] = []
    operating_values: list[float] = []
    maximum_balance_residual = 0.0
    maximum_shared_recharge_violation = 0.0
    maximum_state_residual = 0.0
    for index, scenario in enumerate(scenarios):
        variables = scenario_variables[index]
        charge_values = np.asarray(variables["charge"].X, dtype=float)
        discharge_values = np.asarray(variables["discharge"].X, dtype=float)
        state_values = np.asarray(variables["state"].X, dtype=float)
        unserved_values = np.asarray(variables["unserved"].X, dtype=float)
        residual_peak_value = float(variables["residual_peak"].X)
        task = np.asarray(scenario["task"], dtype=float)
        recharge = np.asarray(scenario["usable_recharge"], dtype=float)
        balance_residual = discharge_values.sum(axis=0) + unserved_values - task
        maximum_balance_residual = max(
            maximum_balance_residual, float(np.max(np.abs(balance_residual)))
        )
        maximum_shared_recharge_violation = max(
            maximum_shared_recharge_violation,
            float(np.max(np.maximum(charge_values.sum(axis=0) - recharge, 0.0))),
        )
        for technology_index, row in active.iterrows():
            eta = float(np.sqrt(float(row["rte"])))
            retention = float(
                (1.0 - float(row["self_discharge_pct_day"]) / 100.0)
                ** (step_hours / 24.0)
            )
            state_residual = (
                state_values[technology_index]
                - retention * np.roll(state_values[technology_index], 1)
                - eta * step_hours * charge_values[technology_index]
                + step_hours * discharge_values[technology_index] / eta
            )
            maximum_state_residual = max(
                maximum_state_residual,
                float(np.max(np.abs(state_residual))),
            )
        variable_cost_value = float(sum(
            float(active.iloc[technology_index]["vom_eur_mwh"])
            * step_hours
            * discharge_values[technology_index].sum()
            for technology_index in range(technology_count)
        ))
        charging_cost_value = float(
            value["charge_price_eur_mwh"] * step_hours * charge_values.sum()
        )
        residual_energy_cost_value = float(
            value["energy_value_eur_mwh"] * step_hours * unserved_values.sum()
        )
        residual_firm_cost_value = float(
            value["firm_value_eur_kw_year"] * 1000.0 * residual_peak_value
        )
        operating_cost_value = (
            variable_cost_value
            + charging_cost_value
            + residual_energy_cost_value
            + residual_firm_cost_value
        )
        operating_values.append(operating_cost_value)
        scenario_rows.append(
            {
                "country": country,
                "risk_weight": risk_weight,
                "mode": mode,
                "scenario_id": scenario["scenario_id"],
                "year": scenario["year"],
                "chronology": scenario["chronology"],
                "normalization_scale_mw": scenario["normalization_scale_mw"],
                "annual_discharge_mwh_per_normalized_mw": float(
                    discharge_values.sum() * step_hours
                ),
                "annual_charge_mwh_per_normalized_mw": float(
                    charge_values.sum() * step_hours
                ),
                "unserved_mwh_per_normalized_mw": float(
                    unserved_values.sum() * step_hours
                ),
                "residual_peak_per_normalized_mw": residual_peak_value,
                "variable_cost_eur_per_normalized_mw": variable_cost_value,
                "charging_cost_eur_per_normalized_mw": charging_cost_value,
                "residual_energy_cost_eur_per_normalized_mw": residual_energy_cost_value,
                "residual_firm_cost_eur_per_normalized_mw": residual_firm_cost_value,
                "operating_cost_eur_per_normalized_mw": operating_cost_value,
                "total_cost_eur_per_normalized_mw": (
                    fixed_cost_value + operating_cost_value
                ),
            }
        )

    mean_operating_value = float(np.mean(operating_values))
    worst_operating_value = float(np.max(operating_values))
    risk_adjusted_value = float(
        fixed_cost_value
        + (1.0 - risk_weight) * mean_operating_value
        + risk_weight * worst_operating_value
    )
    objective_error = abs(float(model.ObjNVal) - risk_adjusted_value)
    summary = {
        "country": country,
        "risk_weight": risk_weight,
        "mode": mode,
        "natural_year_count": int(scenario_count // 2),
        "scenario_count": scenario_count,
        "planning_step_hours": step_hours,
        "technology_count": technology_count,
        "selected_technology_count": int(
            np.count_nonzero(capacity_values > 1.0e-6)
        ),
        "total_power_capacity_per_normalized_mw": total_capacity,
        "total_energy_capacity_mwh_per_normalized_mw": float(sum(
            capacity_values[index] * float(active.iloc[index]["duration_h"])
            for index in range(technology_count)
        )),
        "annual_fixed_cost_eur_per_normalized_mw": fixed_cost_value,
        "mean_operating_cost_eur_per_normalized_mw": mean_operating_value,
        "worst_operating_cost_eur_per_normalized_mw": worst_operating_value,
        "mean_total_cost_eur_per_normalized_mw": (
            fixed_cost_value + mean_operating_value
        ),
        "worst_total_cost_eur_per_normalized_mw": (
            fixed_cost_value + worst_operating_value
        ),
        "risk_adjusted_cost_eur_per_normalized_mw": risk_adjusted_value,
        "solver_objective_eur_per_normalized_mw": float(model.ObjNVal),
        "objective_identity_error_eur_per_normalized_mw": objective_error,
        "maximum_capacity_upper_bound_slack": float(
            maximum_task - np.max(capacity_values)
        ),
        "maximum_task_balance_residual": maximum_balance_residual,
        "maximum_shared_recharge_violation": maximum_shared_recharge_violation,
        "maximum_state_balance_residual_mwh": maximum_state_residual,
        "runtime_seconds": runtime_seconds,
        "solver_status": "optimal",
    }
    model.dispose()
    return summary, capacity_rows, scenario_rows


def solve_country_design_highs(
    country: str,
    scenarios: list[dict[str, Any]],
    technologies: pd.DataFrame,
    value: dict[str, float],
    risk_weight: float,
    mode: str,
    step_hours: int,
    threads: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Solve the shared-capacity, scenario-specific dispatch LP in HiGHS."""
    del threads  # HiGHS thread control follows the process environment.
    if not 0 <= risk_weight <= 1:
        raise ValueError("risk weight must lie in [0, 1]")
    if mode == "diversified":
        active = technologies.copy()
    else:
        active = technologies.loc[
            technologies["config_id"].astype(str).eq(mode)
        ].copy()
        if len(active) != 1:
            raise RuntimeError(f"Unknown single-technology mode: {mode}")
    active = active.reset_index(drop=True)
    technology_count = len(active)
    scenario_count = len(scenarios)
    step_count = len(scenarios[0]["task"])
    scenario_block = 3 * technology_count * step_count + step_count + 1
    capacity_index = np.arange(technology_count, dtype=int)
    risk_threshold_index = technology_count
    excess_indices = np.arange(
        technology_count + 1,
        technology_count + 1 + scenario_count,
        dtype=int,
    )
    first_scenario_index = technology_count + 1 + scenario_count
    variable_count = first_scenario_index + scenario_count * scenario_block

    def indices(scenario_index: int) -> dict[str, Any]:
        offset = first_scenario_index + scenario_index * scenario_block
        charge = np.arange(
            offset, offset + technology_count * step_count, dtype=int
        ).reshape(technology_count, step_count)
        discharge = np.arange(
            offset + technology_count * step_count,
            offset + 2 * technology_count * step_count,
            dtype=int,
        ).reshape(technology_count, step_count)
        state = np.arange(
            offset + 2 * technology_count * step_count,
            offset + 3 * technology_count * step_count,
            dtype=int,
        ).reshape(technology_count, step_count)
        unserved = np.arange(
            offset + 3 * technology_count * step_count,
            offset + 3 * technology_count * step_count + step_count,
            dtype=int,
        )
        residual_peak = offset + 3 * technology_count * step_count + step_count
        return {
            "charge": charge,
            "discharge": discharge,
            "state": state,
            "unserved": unserved,
            "residual_peak": residual_peak,
        }

    scenario_indices = [indices(index) for index in range(scenario_count)]
    objective = np.zeros(variable_count, dtype=float)
    for technology_index, row in active.iterrows():
        objective[capacity_index[technology_index]] = (
            1000.0 * float(row["fixed_cost_eur_kw_year"])
        )
    objective[risk_threshold_index] = risk_weight
    objective[excess_indices] = (
        risk_weight / ((1.0 - CVAR_ALPHA) * scenario_count)
    )
    mean_weight = (1.0 - risk_weight) / scenario_count
    for scenario_index, item in enumerate(scenario_indices):
        objective[item["charge"].reshape(-1)] = (
            mean_weight * value["charge_price_eur_mwh"] * step_hours
        )
        for technology_index, row in active.iterrows():
            objective[item["discharge"][technology_index]] = (
                mean_weight * float(row["vom_eur_mwh"]) * step_hours
            )
        objective[item["unserved"]] = (
            mean_weight * value["energy_value_eur_mwh"] * step_hours
        )
        objective[item["residual_peak"]] = (
            mean_weight * value["firm_value_eur_kw_year"] * 1000.0
        )

    equality_rows: list[np.ndarray] = []
    equality_columns: list[np.ndarray] = []
    equality_values: list[np.ndarray] = []
    equality_rhs: list[np.ndarray] = []
    equality_row_offset = 0
    hours = np.arange(step_count, dtype=int)
    previous = np.roll(hours, 1)
    for scenario_index, scenario in enumerate(scenarios):
        item = scenario_indices[scenario_index]
        task_rows = equality_row_offset + hours
        for technology_index in range(technology_count):
            equality_rows.append(task_rows)
            equality_columns.append(item["discharge"][technology_index])
            equality_values.append(np.ones(step_count))
        equality_rows.append(task_rows)
        equality_columns.append(item["unserved"])
        equality_values.append(np.ones(step_count))
        equality_rhs.append(np.asarray(scenario["task"], dtype=float))
        equality_row_offset += step_count
        for technology_index, row in active.iterrows():
            state_rows = equality_row_offset + hours
            eta = float(np.sqrt(float(row["rte"])))
            retention = float(
                (1.0 - float(row["self_discharge_pct_day"]) / 100.0)
                ** (step_hours / 24.0)
            )
            equality_rows.extend([state_rows, state_rows, state_rows, state_rows])
            equality_columns.extend(
                [
                    item["state"][technology_index, hours],
                    item["state"][technology_index, previous],
                    item["charge"][technology_index, hours],
                    item["discharge"][technology_index, hours],
                ]
            )
            equality_values.extend(
                [
                    np.ones(step_count),
                    np.full(step_count, -retention),
                    np.full(step_count, -eta * step_hours),
                    np.full(step_count, step_hours / eta),
                ]
            )
            equality_rhs.append(np.zeros(step_count))
            equality_row_offset += step_count
    equality_matrix = coo_matrix(
        (
            np.concatenate(equality_values),
            (np.concatenate(equality_rows), np.concatenate(equality_columns)),
        ),
        shape=(equality_row_offset, variable_count),
    ).tocsr()
    equality_rhs_array = np.concatenate(equality_rhs)

    inequality_rows: list[np.ndarray] = []
    inequality_columns: list[np.ndarray] = []
    inequality_values: list[np.ndarray] = []
    inequality_rhs: list[np.ndarray] = []
    inequality_row_offset = 0
    for scenario_index, scenario in enumerate(scenarios):
        item = scenario_indices[scenario_index]
        shared_rows = inequality_row_offset + hours
        for technology_index in range(technology_count):
            inequality_rows.append(shared_rows)
            inequality_columns.append(item["charge"][technology_index])
            inequality_values.append(np.ones(step_count))
        inequality_rhs.append(np.asarray(scenario["usable_recharge"], dtype=float))
        inequality_row_offset += step_count

        peak_rows = inequality_row_offset + hours
        inequality_rows.extend([peak_rows, peak_rows])
        inequality_columns.extend(
            [item["unserved"], np.full(step_count, item["residual_peak"], dtype=int)]
        )
        inequality_values.extend([np.ones(step_count), -np.ones(step_count)])
        inequality_rhs.append(np.zeros(step_count))
        inequality_row_offset += step_count

        for technology_index, row in active.iterrows():
            charge_rows = inequality_row_offset + hours
            inequality_rows.extend([charge_rows, charge_rows])
            inequality_columns.extend(
                [
                    item["charge"][technology_index],
                    np.full(step_count, capacity_index[technology_index], dtype=int),
                ]
            )
            inequality_values.extend(
                [
                    np.ones(step_count),
                    np.full(step_count, -float(row["charge_power_ratio"])),
                ]
            )
            inequality_rhs.append(np.zeros(step_count))
            inequality_row_offset += step_count

            discharge_rows = inequality_row_offset + hours
            inequality_rows.extend([discharge_rows, discharge_rows])
            inequality_columns.extend(
                [
                    item["discharge"][technology_index],
                    np.full(step_count, capacity_index[technology_index], dtype=int),
                ]
            )
            inequality_values.extend([np.ones(step_count), -np.ones(step_count)])
            inequality_rhs.append(np.zeros(step_count))
            inequality_row_offset += step_count

            state_rows = inequality_row_offset + hours
            inequality_rows.extend([state_rows, state_rows])
            inequality_columns.extend(
                [
                    item["state"][technology_index],
                    np.full(step_count, capacity_index[technology_index], dtype=int),
                ]
            )
            inequality_values.extend(
                [
                    np.ones(step_count),
                    np.full(
                        step_count,
                        -USABLE_SOC_FRACTION * float(row["duration_h"]),
                    ),
                ]
            )
            inequality_rhs.append(np.zeros(step_count))
            inequality_row_offset += step_count

        worst_row = np.asarray([inequality_row_offset], dtype=int)
        for technology_index, row in active.iterrows():
            inequality_rows.append(
                np.full(step_count, inequality_row_offset, dtype=int)
            )
            inequality_columns.append(item["discharge"][technology_index])
            inequality_values.append(
                np.full(step_count, float(row["vom_eur_mwh"]) * step_hours)
            )
        inequality_rows.append(np.full(technology_count * step_count, inequality_row_offset, dtype=int))
        inequality_columns.append(item["charge"].reshape(-1))
        inequality_values.append(
            np.full(
                technology_count * step_count,
                value["charge_price_eur_mwh"] * step_hours,
            )
        )
        inequality_rows.append(np.full(step_count, inequality_row_offset, dtype=int))
        inequality_columns.append(item["unserved"])
        inequality_values.append(
            np.full(step_count, value["energy_value_eur_mwh"] * step_hours)
        )
        inequality_rows.append(worst_row)
        inequality_columns.append(np.asarray([item["residual_peak"]], dtype=int))
        inequality_values.append(
            np.asarray([value["firm_value_eur_kw_year"] * 1000.0])
        )
        inequality_rows.append(worst_row)
        inequality_columns.append(np.asarray([risk_threshold_index], dtype=int))
        inequality_values.append(np.asarray([-1.0]))
        inequality_rows.append(worst_row)
        inequality_columns.append(
            np.asarray([excess_indices[scenario_index]], dtype=int)
        )
        inequality_values.append(np.asarray([-1.0]))
        inequality_rhs.append(np.zeros(1))
        inequality_row_offset += 1
    inequality_matrix = coo_matrix(
        (
            np.concatenate(inequality_values),
            (np.concatenate(inequality_rows), np.concatenate(inequality_columns)),
        ),
        shape=(inequality_row_offset, variable_count),
    ).tocsr()
    inequality_rhs_array = np.concatenate(inequality_rhs)

    maximum_task = max(float(np.max(item["task"])) for item in scenarios)
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * variable_count
    for index in capacity_index:
        bounds[int(index)] = (0.0, maximum_task)
    started = time.perf_counter()
    result = linprog(
        objective,
        A_ub=inequality_matrix,
        b_ub=inequality_rhs_array,
        A_eq=equality_matrix,
        b_eq=equality_rhs_array,
        bounds=bounds,
        method="highs",
        options={
            "presolve": True,
            "dual_feasibility_tolerance": 1.0e-8,
            "primal_feasibility_tolerance": 1.0e-8,
        },
    )
    runtime_seconds = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(
            f"{country}-{mode}-{risk_weight:g}: {result.message}"
        )
    solution = np.asarray(result.x, dtype=float)
    capacity_values = solution[capacity_index]
    total_capacity = float(capacity_values.sum())
    fixed_cost_value = float(sum(
        capacity_values[index]
        * 1000.0
        * float(active.iloc[index]["fixed_cost_eur_kw_year"])
        for index in range(technology_count)
    ))
    capacity_rows: list[dict[str, Any]] = []
    for index, row in active.iterrows():
        capacity_rows.append(
            {
                "country": country,
                "risk_weight": risk_weight,
                "mode": mode,
                "config_id": str(row["config_id"]),
                "tech_id": str(row["tech_id"]),
                "duration_h": float(row["duration_h"]),
                "capacity_per_normalized_mw": float(capacity_values[index]),
                "capacity_share": (
                    float(capacity_values[index] / total_capacity)
                    if total_capacity > NUMERICAL_ZERO
                    else 0.0
                ),
                "energy_capacity_mwh_per_normalized_mw": float(
                    capacity_values[index] * float(row["duration_h"])
                ),
                "annual_fixed_cost_eur_per_normalized_mw": float(
                    capacity_values[index]
                    * 1000.0
                    * float(row["fixed_cost_eur_kw_year"])
                ),
            }
        )

    scenario_rows: list[dict[str, Any]] = []
    operating_values: list[float] = []
    maximum_balance_residual = 0.0
    maximum_shared_recharge_violation = 0.0
    maximum_state_residual = 0.0
    for scenario_index, scenario in enumerate(scenarios):
        item = scenario_indices[scenario_index]
        charge_values = solution[item["charge"]]
        discharge_values = solution[item["discharge"]]
        state_values = solution[item["state"]]
        unserved_values = solution[item["unserved"]]
        residual_peak_value = float(solution[item["residual_peak"]])
        task = np.asarray(scenario["task"], dtype=float)
        recharge = np.asarray(scenario["usable_recharge"], dtype=float)
        maximum_balance_residual = max(
            maximum_balance_residual,
            float(np.max(np.abs(discharge_values.sum(axis=0) + unserved_values - task))),
        )
        maximum_shared_recharge_violation = max(
            maximum_shared_recharge_violation,
            float(np.max(np.maximum(charge_values.sum(axis=0) - recharge, 0.0))),
        )
        for technology_index, row in active.iterrows():
            eta = float(np.sqrt(float(row["rte"])))
            retention = float(
                (1.0 - float(row["self_discharge_pct_day"]) / 100.0)
                ** (step_hours / 24.0)
            )
            state_residual = (
                state_values[technology_index]
                - retention * np.roll(state_values[technology_index], 1)
                - eta * step_hours * charge_values[technology_index]
                + step_hours * discharge_values[technology_index] / eta
            )
            maximum_state_residual = max(
                maximum_state_residual,
                float(np.max(np.abs(state_residual))),
            )
        variable_cost_value = float(sum(
            float(active.iloc[technology_index]["vom_eur_mwh"])
            * step_hours
            * discharge_values[technology_index].sum()
            for technology_index in range(technology_count)
        ))
        charging_cost_value = float(
            value["charge_price_eur_mwh"] * step_hours * charge_values.sum()
        )
        residual_energy_cost_value = float(
            value["energy_value_eur_mwh"] * step_hours * unserved_values.sum()
        )
        residual_firm_cost_value = float(
            value["firm_value_eur_kw_year"] * 1000.0 * residual_peak_value
        )
        operating_cost_value = (
            variable_cost_value
            + charging_cost_value
            + residual_energy_cost_value
            + residual_firm_cost_value
        )
        operating_values.append(operating_cost_value)
        scenario_rows.append(
            {
                "country": country,
                "risk_weight": risk_weight,
                "mode": mode,
                "scenario_id": scenario["scenario_id"],
                "year": scenario["year"],
                "chronology": scenario["chronology"],
                "normalization_scale_mw": scenario["normalization_scale_mw"],
                "annual_discharge_mwh_per_normalized_mw": float(
                    discharge_values.sum() * step_hours
                ),
                "annual_charge_mwh_per_normalized_mw": float(
                    charge_values.sum() * step_hours
                ),
                "unserved_mwh_per_normalized_mw": float(
                    unserved_values.sum() * step_hours
                ),
                "residual_peak_per_normalized_mw": residual_peak_value,
                "variable_cost_eur_per_normalized_mw": variable_cost_value,
                "charging_cost_eur_per_normalized_mw": charging_cost_value,
                "residual_energy_cost_eur_per_normalized_mw": residual_energy_cost_value,
                "residual_firm_cost_eur_per_normalized_mw": residual_firm_cost_value,
                "operating_cost_eur_per_normalized_mw": operating_cost_value,
                "total_cost_eur_per_normalized_mw": fixed_cost_value + operating_cost_value,
            }
        )

    mean_operating_value = float(np.mean(operating_values))
    worst_operating_value = float(np.max(operating_values))
    risk_threshold_value = float(solution[risk_threshold_index])
    excess_values = solution[excess_indices]
    cvar_operating_value = float(
        risk_threshold_value
        + excess_values.sum() / ((1.0 - CVAR_ALPHA) * scenario_count)
    )
    risk_adjusted_value = float(
        fixed_cost_value
        + (1.0 - risk_weight) * mean_operating_value
        + risk_weight * cvar_operating_value
    )
    objective_error = abs(float(result.fun) - risk_adjusted_value)
    summary = {
        "country": country,
        "risk_weight": risk_weight,
        "mode": mode,
        "natural_year_count": int(scenario_count // 2),
        "scenario_count": scenario_count,
        "planning_step_hours": step_hours,
        "technology_count": technology_count,
        "selected_technology_count": int(np.count_nonzero(capacity_values > 1.0e-6)),
        "total_power_capacity_per_normalized_mw": total_capacity,
        "total_energy_capacity_mwh_per_normalized_mw": float(sum(
            capacity_values[index] * float(active.iloc[index]["duration_h"])
            for index in range(technology_count)
        )),
        "annual_fixed_cost_eur_per_normalized_mw": fixed_cost_value,
        "mean_operating_cost_eur_per_normalized_mw": mean_operating_value,
        "worst_operating_cost_eur_per_normalized_mw": worst_operating_value,
        "cvar90_operating_cost_eur_per_normalized_mw": cvar_operating_value,
        "mean_total_cost_eur_per_normalized_mw": fixed_cost_value + mean_operating_value,
        "worst_total_cost_eur_per_normalized_mw": fixed_cost_value + worst_operating_value,
        "cvar90_total_cost_eur_per_normalized_mw": fixed_cost_value + cvar_operating_value,
        "risk_adjusted_cost_eur_per_normalized_mw": risk_adjusted_value,
        "solver_objective_eur_per_normalized_mw": float(result.fun),
        "objective_identity_error_eur_per_normalized_mw": objective_error,
        "maximum_capacity_upper_bound_slack": float(maximum_task - np.max(capacity_values)),
        "maximum_task_balance_residual": maximum_balance_residual,
        "maximum_shared_recharge_violation": maximum_shared_recharge_violation,
        "maximum_state_balance_residual_mwh": maximum_state_residual,
        "runtime_seconds": runtime_seconds,
        "solver_status": "optimal",
    }
    return summary, capacity_rows, scenario_rows


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in parse_list(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--countries", default="")
    parser.add_argument("--cost-year", type=int, default=2035)
    parser.add_argument("--value-scenario", type=int, default=20)
    parser.add_argument("--risk-weights", default="0,0.5")
    parser.add_argument("--planning-step-hours", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.planning_step_hours <= 0 or 8760 % args.planning_step_hours:
        raise ValueError("planning step must be a positive divisor of 8,760")
    risk_weights = parse_float_list(args.risk_weights)
    if any(weight < 0 or weight > 1 for weight in risk_weights):
        raise ValueError("risk weights must lie in [0, 1]")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise RuntimeError("Output directory is non-empty; pass --force to replace outputs")
    output.mkdir(parents=True, exist_ok=True)

    runner = load_registered_runner()
    usecols = [
        "country",
        "year",
        "timestamp_utc",
        "source_day_id",
        "season_segment",
        "event_flag_q75",
        "task_c3_q75_mw",
        "raw_gas_mean_mw",
        TASK_COLUMN,
        RECHARGE_COLUMN,
    ]
    panel = pd.read_csv(PANEL_PATH, usecols=usecols)
    available_countries = sorted(panel["country"].astype(str).unique())
    countries = parse_list(args.countries) or available_countries
    missing = sorted(set(countries) - set(available_countries))
    if missing:
        raise RuntimeError(f"Countries absent from the panel: {missing}")
    technologies = load_technologies(args.cost_year)
    value = load_value_scenario(args.value_scenario)

    summary_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    conservation_rows: list[dict[str, Any]] = []
    started_at = utc_now()
    modes = ["diversified", *CONFIG_IDS]
    for country in countries:
        country_frame = panel.loc[panel["country"].astype(str).eq(country)].copy()
        scenarios, conservation = build_scenarios(
            country_frame, runner, args.planning_step_hours
        )
        conservation_rows.extend(
            [{"country": country, **row} for row in conservation]
        )
        for risk_weight in risk_weights:
            for mode in modes:
                summary, capacities, scenario_metrics = solve_country_design_highs(
                    country=country,
                    scenarios=scenarios,
                    technologies=technologies,
                    value=value,
                    risk_weight=risk_weight,
                    mode=mode,
                    step_hours=args.planning_step_hours,
                    threads=args.threads,
                )
                summary_rows.append(summary)
                capacity_rows.extend(capacities)
                scenario_rows.extend(scenario_metrics)
                pd.DataFrame(summary_rows).to_csv(
                    output / "design_summary_checkpoint.csv", index=False
                )

    summaries = pd.DataFrame(summary_rows)
    capacities = pd.DataFrame(capacity_rows)
    scenario_metrics = pd.DataFrame(scenario_rows)
    conservation_frame = pd.DataFrame(conservation_rows)
    summaries.to_csv(output / "design_summary.csv", index=False)
    capacities.to_csv(output / "capacity_mix.csv", index=False)
    scenario_metrics.to_csv(output / "scenario_metrics.csv.gz", index=False)
    conservation_frame.to_csv(output / "chronology_conservation.csv", index=False)

    comparisons: list[dict[str, Any]] = []
    for (country, risk_weight), group in summaries.groupby(
        ["country", "risk_weight"], sort=True
    ):
        diversified = group.loc[group["mode"].eq("diversified")].iloc[0]
        singles = group.loc[~group["mode"].eq("diversified")].copy()
        best = singles.sort_values(
            ["risk_adjusted_cost_eur_per_normalized_mw", "mode"],
            kind="stable",
        ).iloc[0]
        comparisons.append(
            {
                "country": country,
                "risk_weight": risk_weight,
                "best_single_mode": best["mode"],
                "diversified_selected_technology_count": int(
                    diversified["selected_technology_count"]
                ),
                "diversified_total_power_capacity_percent_mean_gas": float(
                    diversified["total_power_capacity_per_normalized_mw"] * 100.0
                ),
                "best_single_total_power_capacity_percent_mean_gas": float(
                    best["total_power_capacity_per_normalized_mw"] * 100.0
                ),
                "diversified_minus_best_single_mean_cost_eur_per_normalized_mw": float(
                    diversified["mean_total_cost_eur_per_normalized_mw"]
                    - best["mean_total_cost_eur_per_normalized_mw"]
                ),
                "diversified_minus_best_single_worst_cost_eur_per_normalized_mw": float(
                    diversified["worst_total_cost_eur_per_normalized_mw"]
                    - best["worst_total_cost_eur_per_normalized_mw"]
                ),
                "diversified_minus_best_single_cvar90_cost_eur_per_normalized_mw": float(
                    diversified["cvar90_total_cost_eur_per_normalized_mw"]
                    - best["cvar90_total_cost_eur_per_normalized_mw"]
                ),
                "diversified_minus_best_single_risk_adjusted_cost_eur_per_normalized_mw": float(
                    diversified["risk_adjusted_cost_eur_per_normalized_mw"]
                    - best["risk_adjusted_cost_eur_per_normalized_mw"]
                ),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output / "diversification_comparison.csv", index=False)

    max_task_conservation = float(
        conservation_frame.groupby(["country", "year"])[
            "task_energy_normalized_mwh"
        ].agg(lambda values: float(values.max() - values.min())).max()
    )
    max_recharge_conservation = float(
        conservation_frame.groupby(["country", "year"])[
            "recharge_energy_normalized_mwh"
        ].agg(lambda values: float(values.max() - values.min())).max()
    )
    qa = {
        "status": "pass",
        "evidence_status": "multiyear_endogenous_capacity_expansion_validation",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "countries": countries,
        "country_count": len(countries),
        "natural_year_count": int(
            panel.loc[panel["country"].isin(countries), ["country", "year"]]
            .drop_duplicates().shape[0]
        ),
        "chronology_scenario_count": int(
            conservation_frame[["country", "year", "chronology"]]
            .drop_duplicates().shape[0]
        ),
        "risk_weights": risk_weights,
        "modes": modes,
        "planning_step_hours": args.planning_step_hours,
        "all_solver_statuses_optimal": bool(
            summaries["solver_status"].eq("optimal").all()
        ),
        "maximum_objective_identity_error_eur_per_normalized_mw": float(
            summaries["objective_identity_error_eur_per_normalized_mw"].max()
        ),
        "maximum_task_balance_residual": float(
            summaries["maximum_task_balance_residual"].max()
        ),
        "maximum_shared_recharge_violation": float(
            summaries["maximum_shared_recharge_violation"].max()
        ),
        "maximum_state_balance_residual_mwh": float(
            summaries["maximum_state_balance_residual_mwh"].max()
        ),
        "minimum_capacity_upper_bound_slack": float(
            summaries["maximum_capacity_upper_bound_slack"].min()
        ),
        "capacity_upper_bound_binding_design_count": int(
            summaries["maximum_capacity_upper_bound_slack"].le(1.0e-6).sum()
        ),
        "maximum_chronology_task_energy_residual": max_task_conservation,
        "maximum_chronology_recharge_energy_residual": max_recharge_conservation,
        "diversified_portfolio_count": int(
            comparison_frame["diversified_selected_technology_count"].gt(1).sum()
        ),
        "risk_aware_country_count": int(
            comparison_frame["risk_weight"].gt(0).sum()
        ),
        "risk_aware_cvar90_cost_improvement_country_count": int(
            comparison_frame.loc[
                comparison_frame["risk_weight"].gt(0),
                "diversified_minus_best_single_cvar90_cost_eur_per_normalized_mw",
            ].lt(-1.0e-6).sum()
        ),
        "input_sha256": {
            "panel": sha256(PANEL_PATH),
            "registered_chronology_runner": sha256(REGISTERED_RUNNER_PATH),
            "cost_catalogue": sha256(COST_CATALOGUE_PATH),
            "value_scenarios": sha256(VALUE_SCENARIO_PATH),
        },
        "output_sha256": {
            name: sha256(output / name)
            for name in (
                "design_summary.csv",
                "capacity_mix.csv",
                "scenario_metrics.csv.gz",
                "chronology_conservation.csv",
                "diversification_comparison.csv",
            )
        },
    }
    tolerances_pass = (
        qa["all_solver_statuses_optimal"]
        and qa["maximum_objective_identity_error_eur_per_normalized_mw"] <= 0.02
        and qa["maximum_task_balance_residual"] <= 1.0e-6
        and qa["maximum_shared_recharge_violation"] <= 1.0e-6
        and qa["maximum_state_balance_residual_mwh"] <= 1.0e-6
        # The upper bound equals the largest scenario task.  Capacity above
        # that value cannot serve additional task and is strictly dominated
        # when fixed costs are positive, so equality is an admissible
        # full-coverage optimum; QA checks for violations rather than slack.
        and qa["minimum_capacity_upper_bound_slack"] >= -1.0e-8
        and qa["maximum_chronology_task_energy_residual"] <= 1.0e-10
        and qa["maximum_chronology_recharge_energy_residual"] <= 1.0e-10
    )
    qa["status"] = "pass" if tolerances_pass else "fail"
    write_json(output / "qa.json", qa)
    (output / "design_summary_checkpoint.csv").unlink(missing_ok=True)
    print(json.dumps(json_ready(qa), ensure_ascii=False, indent=2))
    return 0 if tolerances_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
