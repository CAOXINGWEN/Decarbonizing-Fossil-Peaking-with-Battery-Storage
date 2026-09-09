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

HERE = Path(__file__).resolve().parents[2] / "capacity_expansion_validation"

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
            "threads": 1,
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
    maximum_inequality_violation = float(np.maximum(inequality_matrix @ solution - inequality_rhs_array, 0).max())
    maximum_equality_residual = float(np.abs(equality_matrix @ solution - equality_rhs_array).max())
    stationarity = objective - equality_matrix.T @ result.eqlin.marginals - inequality_matrix.T @ result.ineqlin.marginals - result.lower.marginals
    dual_objective = float(equality_rhs_array @ result.eqlin.marginals + inequality_rhs_array @ result.ineqlin.marginals)
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
    lp_cvar_epigraph_value = cvar_operating_value
    op = np.asarray(operating_values, dtype=float)
    cvar_operating_value = float(min(t + np.maximum(op - t, 0).sum() / ((1.0 - CVAR_ALPHA) * scenario_count) for t in op))
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
        "natural_year_count": len({int(item["year"]) for item in scenarios}),
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
        "task_peak_minus_maximum_capacity": float(maximum_task - np.max(capacity_values)),
        "capacity_bound": "nonnegative_without_task_peak_upper_bound",
        "maximum_inequality_violation": maximum_inequality_violation,
        "maximum_equality_residual": maximum_equality_residual,
        "maximum_nonnegativity_violation": float(np.maximum(-solution, 0).max()),
        "maximum_stationarity_residual": float(np.abs(stationarity).max()),
        "primal_dual_objective_gap_eur": float(result.fun - dual_objective),
        "lp_cvar_epigraph_value": lp_cvar_epigraph_value,
        "cvar_reporting_method": "empirical_equal_probability_scenario_costs",
        "solver_iterations": int(result.nit),
        "maximum_task_balance_residual": maximum_balance_residual,
        "maximum_shared_recharge_violation": maximum_shared_recharge_violation,
        "maximum_state_balance_residual_mwh": maximum_state_residual,
        "runtime_seconds": runtime_seconds,
        "solver_status": "optimal",
    }
    return summary, capacity_rows, scenario_rows
