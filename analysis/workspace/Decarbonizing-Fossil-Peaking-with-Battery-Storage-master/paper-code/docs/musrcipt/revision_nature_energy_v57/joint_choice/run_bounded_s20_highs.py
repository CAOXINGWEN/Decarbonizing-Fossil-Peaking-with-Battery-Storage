#!/usr/bin/env python3
"""Exact bounded-tail-power portfolio LP in SciPy HiGHS.

This runner reproduces the continuous Gurobi formulation registered for the
v42/v43 portfolio experiment, including its two-stage objective.  The tail
power, hourly dispatch and cyclic state of charge are optimized jointly.  It
is used because the local Gurobi token server is unavailable; a mandatory
smoke mode compares this formulation against the archived Gurobi S31 result.

Production runs use the central S20 valuation and one of the technology-
specific 2025, 2030 or 2035 cost catalogues.  Country-year chunks are written
independently so interrupted runs can be resumed without mixing model years.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Avoid hidden BLAS over-subscription when four country-year chunks run in
# parallel.  These must be set before NumPy/SciPy are imported.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack


HERE = Path(__file__).resolve().parent
V44 = HERE.parent
V43 = V44.parent / "revision_nature_energy_v43"
V40 = V44.parent / "revision_nature_energy_v40"
RUNNER_PATH = V44 / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"
LONG_CATALOGUE = V43 / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
SCENARIO_PATH = V40 / "currency_work" / "value_sensitivity_gap_summary_source_aware_eur.csv"
GUR_S31 = V43 / "portfolio_model" / "full_132cy_optimized" / "portfolio_arm_results.csv.gz"
PARTITIONS = V43 / "portfolio_model" / "full_132cy_fixed_s20_central_chunks"
DEFAULT_OUTPUT = HERE / "bounded_s20_highs"

HOURS = 8760
SOC_MIN = 0.05
SOC_MAX = 0.95
USABLE_SOC_FRACTION = SOC_MAX - SOC_MIN
NUMERICAL_ZERO = 1.0e-10
PRIMARY_ABS_TOL_EUR = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def equivalent_firm_capacity(task_mw: np.ndarray, target_unserved_mwh: float) -> float:
    """Flat firm capacity yielding the same EENS as the optimized portfolio."""
    total = float(task_mw.sum())
    if target_unserved_mwh >= total - 1.0e-9:
        return 0.0
    low, high = 0.0, float(task_mw.max())
    for _ in range(70):
        midpoint = 0.5 * (low + high)
        if float(np.maximum(task_mw - midpoint, 0.0).sum()) > target_unserved_mwh:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def load_runner():
    spec = importlib.util.spec_from_file_location("v44_registered_runner_highs", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import registered model utilities")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_scenario(scenario_id: int) -> dict[str, float]:
    frame = pd.read_csv(SCENARIO_PATH)
    subset = frame[frame["scenario_id"].eq(scenario_id)]
    columns = ["energy_value_eur_mwh", "firm_value_eur_kw_year", "charge_price_eur_mwh"]
    unique = subset[columns].drop_duplicates()
    if len(unique) != 1:
        raise RuntimeError(f"Scenario {scenario_id} is not unique")
    return {column: float(unique.iloc[0][column]) for column in columns}


def load_catalogue(cost_year: int) -> tuple[pd.Series, pd.DataFrame]:
    frame = pd.read_csv(LONG_CATALOGUE)
    frame = frame[frame["learning_year"].eq(cost_year)].copy()
    if len(frame) != 53 or frame["config_id"].duplicated().any():
        raise RuntimeError(f"Cost-year {cost_year} must contain 53 unique configurations")
    base = frame[frame["config_id"].eq("li_lfp_4h")]
    if len(base) != 1:
        raise RuntimeError("The 4 h LFP base configuration is not unique")
    candidates = frame[~frame["tech_id"].eq("li_lfp")].copy()
    candidates = candidates.sort_values(["duration_h", "tech_id"], kind="stable").reset_index(drop=True)
    if len(candidates) != 49:
        raise RuntimeError(f"Expected 49 non-LFP configurations, found {len(candidates)}")
    return base.iloc[0], candidates


def parameters(row: pd.Series) -> dict[str, float]:
    return {
        "duration_h": float(row["duration_h"]),
        "rte": float(row["rte"]),
        "self_discharge_pct_day": float(row["self_discharge_pct_day"]),
        "charge_power_ratio": float(row["charge_power_ratio"]),
        "fixed_cost_eur_kw_year": float(
            row["annual_fixed_cost_eur_source_aware_per_kw_discharge_year"]
        ),
        "vom_eur_mwh": float(row["variable_om_eur_source_aware_mwh"]),
    }


def identity(row: pd.Series) -> dict[str, Any]:
    parameter = parameters(row)
    return {
        "config_id": str(row["config_id"]),
        "tech_id": str(row["tech_id"]),
        "technology_en": str(row["technology_en"]),
        **parameter,
    }


def partition_country_years(chunk: int | None) -> list[tuple[str, int]]:
    if chunk is None:
        pairs: list[tuple[str, int]] = []
        for index in range(1, 5):
            pairs.extend(partition_country_years(index))
        return sorted(pairs)
    contract = json.loads((PARTITIONS / f"chunk{chunk}" / "run_contract.json").read_text())
    return [(str(country), int(year)) for country, year in contract["country_years"]]


def shard_country_years(index: int, count: int) -> list[tuple[str, int]]:
    if count < 1 or index < 1 or index > count:
        raise ValueError("Shard index must be between one and shard count")
    all_pairs = partition_country_years(None)
    shards = np.array_split(np.asarray(all_pairs, dtype=object), count)
    return [(str(country), int(year)) for country, year in shards[index - 1].tolist()]


def build_equalities(
    task: np.ndarray,
    base_eta: float,
    tail_eta: float,
    base_retention: float,
    tail_retention: float,
) -> tuple[csr_matrix, np.ndarray]:
    # Variables: qB,dB,sB,qT,dT,sT,u,P,r.
    q_b, d_b, s_b, q_t, d_t, s_t, u = [index * HOURS for index in range(7)]
    n_variables = 7 * HOURS + 2
    hour = np.arange(HOURS)
    previous = np.roll(hour, 1)
    row_base = HOURS + hour
    row_tail = 2 * HOURS + hour
    rows = np.concatenate([
        hour, hour, hour,
        row_base, row_base, row_base, row_base,
        row_tail, row_tail, row_tail, row_tail,
    ])
    columns = np.concatenate([
        d_b + hour, d_t + hour, u + hour,
        s_b + hour, s_b + previous, q_b + hour, d_b + hour,
        s_t + hour, s_t + previous, q_t + hour, d_t + hour,
    ])
    values = np.concatenate([
        np.ones(HOURS), np.ones(HOURS), np.ones(HOURS),
        np.ones(HOURS), -base_retention * np.ones(HOURS),
        -base_eta * np.ones(HOURS), np.full(HOURS, 1.0 / base_eta),
        np.ones(HOURS), -tail_retention * np.ones(HOURS),
        -tail_eta * np.ones(HOURS), np.full(HOURS, 1.0 / tail_eta),
    ])
    matrix = coo_matrix((values, (rows, columns)), shape=(3 * HOURS, n_variables)).tocsr()
    return matrix, np.concatenate([task, np.zeros(2 * HOURS)])


def build_inequalities(
    usable_recharge: np.ndarray,
    charge_power_ratio: float,
    duration_h: float,
) -> tuple[csr_matrix, np.ndarray]:
    q_b, q_t, d_t, s_t, u = 0, 3 * HOURS, 4 * HOURS, 5 * HOURS, 6 * HOURS
    p, r = 7 * HOURS, 7 * HOURS + 1
    n_variables = 7 * HOURS + 2
    hour = np.arange(HOURS)
    row_shared = hour
    row_peak = HOURS + hour
    row_q = 2 * HOURS + hour
    row_d = 3 * HOURS + hour
    row_s = 4 * HOURS + hour
    rows = np.concatenate([
        row_shared, row_shared,
        row_peak, row_peak,
        row_q, row_q,
        row_d, row_d,
        row_s, row_s,
    ])
    columns = np.concatenate([
        q_b + hour, q_t + hour,
        u + hour, np.full(HOURS, r),
        q_t + hour, np.full(HOURS, p),
        d_t + hour, np.full(HOURS, p),
        s_t + hour, np.full(HOURS, p),
    ])
    values = np.concatenate([
        np.ones(HOURS), np.ones(HOURS),
        np.ones(HOURS), -np.ones(HOURS),
        np.ones(HOURS), np.full(HOURS, -charge_power_ratio),
        np.ones(HOURS), -np.ones(HOURS),
        np.ones(HOURS), np.full(HOURS, -USABLE_SOC_FRACTION * duration_h),
    ])
    matrix = coo_matrix((values, (rows, columns)), shape=(5 * HOURS, n_variables)).tocsr()
    rhs = np.concatenate([usable_recharge, np.zeros(4 * HOURS)])
    return matrix, rhs


def solve_bounded_arm(
    task: np.ndarray,
    tail_task: np.ndarray,
    raw_recharge: np.ndarray,
    base_power: float,
    base: dict[str, float],
    tail: dict[str, float],
    scenario: dict[str, float],
) -> dict[str, Any]:
    if any(len(values) != HOURS for values in (task, tail_task, raw_recharge)):
        raise RuntimeError("Every annual input must have exactly 8,760 hours")
    usable_recharge = np.where(task > NUMERICAL_ZERO, 0.0, raw_recharge)
    base_eta = float(np.sqrt(base["rte"]))
    tail_eta = float(np.sqrt(tail["rte"]))
    base_retention = float((1.0 - base["self_discharge_pct_day"] / 100.0) ** (1.0 / 24.0))
    tail_retention = float((1.0 - tail["self_discharge_pct_day"] / 100.0) ** (1.0 / 24.0))
    a_eq, b_eq = build_equalities(task, base_eta, tail_eta, base_retention, tail_retention)
    a_ub, b_ub = build_inequalities(
        usable_recharge, tail["charge_power_ratio"], tail["duration_h"]
    )

    q_b, d_b, s_b, q_t, d_t, s_t, u = [index * HOURS for index in range(7)]
    p, r = 7 * HOURS, 7 * HOURS + 1
    lower = np.zeros(7 * HOURS + 2)
    upper = np.full(7 * HOURS + 2, np.inf)
    upper[q_b:d_b] = np.minimum(usable_recharge, base["charge_power_ratio"] * base_power)
    upper[d_b:s_b] = np.minimum(task, base_power)
    upper[s_b:q_t] = USABLE_SOC_FRACTION * base["duration_h"] * base_power
    upper[q_t:d_t] = usable_recharge
    upper[d_t:s_t] = tail_task
    upper[u:p] = task
    upper[p] = base_power
    bounds = np.column_stack([lower, upper])

    objective = np.zeros(7 * HOURS + 2)
    objective[q_b:d_b] = scenario["charge_price_eur_mwh"]
    objective[d_b:s_b] = base["vom_eur_mwh"]
    objective[q_t:d_t] = scenario["charge_price_eur_mwh"]
    objective[d_t:s_t] = tail["vom_eur_mwh"]
    objective[u:p] = scenario["energy_value_eur_mwh"]
    objective[p] = 1000.0 * tail["fixed_cost_eur_kw_year"]
    objective[r] = 1000.0 * scenario["firm_value_eur_kw_year"]
    options = {"dual_feasibility_tolerance": 1.0e-8, "primal_feasibility_tolerance": 1.0e-8}
    started = time.perf_counter()
    primary = linprog(
        objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
        bounds=bounds, method="highs-ds", options=options,
    )
    if not primary.success:
        raise RuntimeError(f"HiGHS primary solve failed: {primary.message}")
    charge_objective = np.zeros_like(objective)
    charge_objective[q_b:d_b] = 1.0
    charge_objective[q_t:d_t] = 1.0
    # Normalize the primary-objective cut.  Without this scaling, enforcing a
    # one-cent absolute lexicographic tolerance against a billion-euro annual
    # objective can trigger HiGHS status 15 even when the primary basis is
    # optimal.  Division by the largest objective coefficient preserves the
    # exact feasible set while keeping the cut numerically commensurate with
    # the physical constraints.
    objective_scale = max(float(np.max(np.abs(objective))), 1.0)
    primary_cut = csr_matrix((objective / objective_scale).reshape(1, -1))
    secondary = linprog(
        charge_objective,
        A_ub=vstack([a_ub, primary_cut], format="csr"),
        b_ub=np.append(b_ub, (primary.fun + PRIMARY_ABS_TOL_EUR) / objective_scale),
        A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs-ds", options=options,
    )
    if not secondary.success:
        secondary = linprog(
            charge_objective,
            A_ub=vstack([a_ub, primary_cut], format="csr"),
            b_ub=np.append(b_ub, (primary.fun + PRIMARY_ABS_TOL_EUR) / objective_scale),
            A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs-ipm", options=options,
        )
    secondary_success = bool(secondary.success)
    # The primary solution is already a valid optimum for the procurement
    # experiment.  The secondary objective only selects the least-charge
    # dispatch inside the registered one-cent primary tolerance.  Retain the
    # primary optimum if both HiGHS algorithms reject that numerically narrow
    # secondary slice, and expose the fallback explicitly for QA.
    selected = secondary if secondary_success else primary

    x = selected.x
    q_base, d_base, s_base = x[q_b:d_b], x[d_b:s_b], x[s_b:q_t]
    q_tail, d_tail, s_tail = x[q_t:d_t], x[d_t:s_t], x[s_t:u]
    unserved = x[u:p]
    tail_power = float(x[p])
    residual_peak = float(x[r])
    base_charge, tail_charge = float(q_base.sum()), float(q_tail.sum())
    base_discharge, tail_discharge = float(d_base.sum()), float(d_tail.sum())
    eens = float(unserved.sum())
    task_mwh = float(task.sum())
    efc = equivalent_firm_capacity(task, eens)
    base_fixed = 1000.0 * base_power * base["fixed_cost_eur_kw_year"]
    tail_fixed = 1000.0 * tail_power * tail["fixed_cost_eur_kw_year"]
    base_vom = base["vom_eur_mwh"] * base_discharge
    tail_vom = tail["vom_eur_mwh"] * tail_discharge
    charge_cost = scenario["charge_price_eur_mwh"] * (base_charge + tail_charge)
    residual_energy = scenario["energy_value_eur_mwh"] * eens
    residual_firm = 1000.0 * scenario["firm_value_eur_kw_year"] * residual_peak
    system_cost = base_fixed + tail_fixed + base_vom + tail_vom + charge_cost + residual_energy + residual_firm
    gas_only = scenario["energy_value_eur_mwh"] * task_mwh + 1000.0 * scenario["firm_value_eur_kw_year"] * float(task.max())
    tail_capacity = tail["duration_h"] * tail_power
    balance = d_base + d_tail + unserved - task
    base_soc = s_base - base_retention * np.roll(s_base, 1) - base_eta * q_base + d_base / base_eta
    tail_soc = s_tail - tail_retention * np.roll(s_tail, 1) - tail_eta * q_tail + d_tail / tail_eta
    shared = q_base + q_tail - usable_recharge
    tail_cycle = tail_discharge / tail_capacity if tail_capacity > NUMERICAL_ZERO else 0.0
    return {
        "solver_status": 0,
        "solver_status_name": "optimal",
        "solver_method": (
            "scipy_highs_two_stage" if secondary_success
            else "scipy_highs_primary_optimum_secondary_numerical_fallback"
        ),
        "secondary_tiebreak_success": secondary_success,
        "solver_runtime_seconds": time.perf_counter() - started,
        "solver_iterations": int(primary.nit + (secondary.nit if secondary_success else 0)),
        "base_power_mw": base_power,
        "base_duration_h": base["duration_h"],
        "base_energy_capacity_mwh": base["duration_h"] * base_power,
        "base_usable_energy_capacity_mwh": USABLE_SOC_FRACTION * base["duration_h"] * base_power,
        "tail_power_upper_mw": base_power,
        "tail_power_mode": "optimized",
        "tail_power_mw": tail_power,
        "tail_energy_capacity_mwh": tail_capacity,
        "tail_usable_energy_capacity_mwh": USABLE_SOC_FRACTION * tail_capacity,
        "base_task_mwh": task_mwh,
        "tail_task_mwh": float(tail_task.sum()),
        "raw_recharge_mwh": float(raw_recharge.sum()),
        "usable_recharge_mwh": float(usable_recharge.sum()),
        "task_recharge_overlap_mwh": float(raw_recharge[task > NUMERICAL_ZERO].sum()),
        "base_charge_mwh": base_charge,
        "tail_charge_mwh": tail_charge,
        "total_charge_mwh": base_charge + tail_charge,
        "base_discharge_mwh": base_discharge,
        "tail_discharge_mwh": tail_discharge,
        "total_discharge_mwh": base_discharge + tail_discharge,
        "unserved_eens_mwh": eens,
        "served_share": (base_discharge + tail_discharge) / task_mwh if task_mwh else 0.0,
        "portfolio_efc_mw": efc,
        "portfolio_efc_ratio_to_base_power": efc / base_power,
        "firm_peak_reduction_mw": float(task.max()) - residual_peak,
        "residual_peak_mw": residual_peak,
        "base_equivalent_full_cycles": base_discharge / (base["duration_h"] * base_power),
        "tail_equivalent_full_cycles": tail_cycle,
        "base_fixed_cost_eur": base_fixed,
        "tail_fixed_cost_eur": tail_fixed,
        "base_vom_cost_eur": base_vom,
        "tail_vom_cost_eur": tail_vom,
        "charge_opportunity_cost_eur": charge_cost,
        "residual_energy_cost_eur": residual_energy,
        "residual_firm_cost_eur": residual_firm,
        "storage_cost_eur": base_fixed + tail_fixed + base_vom + tail_vom + charge_cost,
        "system_cost_eur": system_cost,
        "gas_only_system_cost_eur": gas_only,
        "net_benefit_vs_gas_only_eur": gas_only - system_cost,
        "system_cost_eur_per_mwh_task": system_cost / task_mwh if task_mwh else np.nan,
        "base_soc_cyclic_residual_mwh": float(base_soc[0]),
        "tail_soc_cyclic_residual_mwh": float(tail_soc[0]),
        "base_max_soc_balance_residual_mwh": float(np.abs(base_soc).max()),
        "tail_max_soc_balance_residual_mwh": float(np.abs(tail_soc).max()),
        "base_soc_min_fraction": SOC_MIN + float(s_base.min()) / (base["duration_h"] * base_power),
        "base_soc_max_fraction": SOC_MIN + float(s_base.max()) / (base["duration_h"] * base_power),
        "tail_soc_min_fraction": SOC_MIN + float(s_tail.min()) / tail_capacity if tail_capacity > NUMERICAL_ZERO else np.nan,
        "tail_soc_max_fraction": SOC_MIN + float(s_tail.max()) / tail_capacity if tail_capacity > NUMERICAL_ZERO else np.nan,
        "maximum_task_balance_residual_mw": float(np.abs(balance).max()),
        "maximum_shared_recharge_violation_mw": float(max(0.0, shared.max())),
        "shared_recharge_binding_hours": int(np.sum((usable_recharge > NUMERICAL_ZERO) & (q_base + q_tail >= usable_recharge - 1.0e-7))),
        "base_charge_power_binding_hours": int(np.sum(q_base >= base["charge_power_ratio"] * base_power - 1.0e-7)),
        "tail_charge_power_binding_hours": int(np.sum((tail_power > NUMERICAL_ZERO) & (q_tail >= tail["charge_power_ratio"] * tail_power - 1.0e-7))),
        "base_discharge_power_binding_hours": int(np.sum(d_base >= base_power - 1.0e-7)),
        "tail_discharge_power_binding_hours": int(np.sum((tail_power > NUMERICAL_ZERO) & (d_tail >= tail_power - 1.0e-7))),
    }


def prepare_inputs(cost_year: int, scenario_id: int):
    runner = load_runner()
    panel = pd.read_csv(
        runner.DEFAULT_PANEL,
        usecols=[
            "country", "year", "source_hour_id", "source_day_id", "season_segment",
            "event_flag_q75", "raw_gas_mean_mw", "forced_discharge_power_mw",
            runner.BASE_TASK_COLUMN, runner.TAIL_TASK_COLUMN, runner.RECHARGE_COLUMN,
        ],
    ).sort_values(["country", "year", "source_hour_id"], kind="stable")
    base_row, candidates = load_catalogue(cost_year)
    return runner, panel, base_row, candidates, load_scenario(scenario_id)


def solve_country_year(
    runner, frame: pd.DataFrame, base_row: pd.Series, candidates: pd.DataFrame,
    scenario: dict[str, float], scenario_id: int, cost_year: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    orders, order_stats = runner.day_orders(frame)
    original_task = frame[runner.BASE_TASK_COLUMN].to_numpy(float)
    original_tail = frame[runner.TAIL_TASK_COLUMN].to_numpy(float)
    original_recharge = frame[runner.RECHARGE_COLUMN].to_numpy(float)
    base_power = float(frame["forced_discharge_power_mw"].iloc[0])
    base = parameters(base_row)
    arms = [("extra_lfp4_comparator", base_row)] + [
        ("ldes_candidate", row) for _, row in candidates.iterrows()
    ]
    rows: list[dict[str, Any]] = []
    chronology_rows: list[dict[str, Any]] = []
    for chronology in ("dispersed", "clustered"):
        order = orders[chronology]
        task = runner.reorder_days(original_task, order)
        tail_task = runner.reorder_days(original_tail, order)
        recharge = runner.reorder_days(original_recharge, order)
        chronology_rows.append({
            "country": str(frame["country"].iloc[0]), "year": int(frame["year"].iloc[0]),
            "chronology": chronology, **order_stats,
            "base_task_conservation_error_mwh": float(task.sum() - original_task.sum()),
            "tail_task_conservation_error_mwh": float(tail_task.sum() - original_tail.sum()),
            "recharge_conservation_error_mwh": float(recharge.sum() - original_recharge.sum()),
        })
        for arm_type, tail_row in arms:
            result = solve_bounded_arm(task, tail_task, recharge, base_power, base, parameters(tail_row), scenario)
            rows.append({
                "country": str(frame["country"].iloc[0]), "year": int(frame["year"].iloc[0]),
                "chronology": chronology, "arm_type": arm_type,
                **identity(tail_row), "learning_year": cost_year,
                "scenario_id": scenario_id, **scenario, **result,
            })
    return rows, chronology_rows


def smoke() -> None:
    runner, panel, base_row, candidates, scenario = prepare_inputs(2035, 31)
    frame = panel[panel["country"].eq("DE") & panel["year"].eq(2023)].copy()
    selected = candidates[candidates["config_id"].eq("caes_24h")]
    if len(selected) != 1:
        raise RuntimeError("CAES24 smoke candidate is not unique")
    rows, _ = solve_country_year(runner, frame, base_row, selected, scenario, 31, 2035)
    result = pd.DataFrame(rows)
    reference = pd.read_csv(GUR_S31)
    reference = reference[
        reference["country"].eq("DE") & reference["year"].eq(2023)
        & reference["config_id"].isin(["li_lfp_4h", "caes_24h"])
    ]
    checks = []
    metrics = [
        "tail_power_mw", "total_charge_mwh", "total_discharge_mwh",
        "unserved_eens_mwh", "portfolio_efc_mw", "residual_peak_mw", "system_cost_eur",
    ]
    for row in result.itertuples(index=False):
        ref = reference[
            reference["chronology"].eq(row.chronology)
            & reference["config_id"].eq(row.config_id)
        ].iloc[0]
        for metric in metrics:
            actual = float(getattr(row, metric)); expected = float(ref[metric])
            absolute = abs(actual - expected)
            checks.append({
                "chronology": row.chronology, "config_id": row.config_id, "metric": metric,
                "highs": actual, "gurobi": expected, "absolute_difference": absolute,
                "relative_difference": absolute / max(abs(expected), 1.0),
            })
    check = pd.DataFrame(checks)
    output = DEFAULT_OUTPUT / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    check.to_csv(output / "highs_vs_gurobi_smoke.csv", index=False)
    qa = {
        "status": "pass" if float(check["relative_difference"].max()) < 1.0e-5 else "fail",
        "maximum_relative_difference": float(check["relative_difference"].max()),
        "maximum_absolute_difference": float(check["absolute_difference"].max()),
        "comparison_rows": int(len(check)),
    }
    write_json(output / "smoke_qa.json", qa)
    print(check.to_string(index=False)); print(json.dumps(qa, indent=2))
    if qa["status"] != "pass":
        raise RuntimeError("Bounded HiGHS-to-Gurobi smoke comparison failed")


def production(
    cost_year: int,
    output: Path,
    chunk: int | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> None:
    smoke_qa = json.loads((DEFAULT_OUTPUT / "smoke" / "smoke_qa.json").read_text())
    if smoke_qa.get("status") != "pass":
        raise RuntimeError("A passing bounded-power smoke comparison is required")
    runner, panel, base_row, candidates, scenario = prepare_inputs(cost_year, 20)
    if shard_index is not None or shard_count is not None:
        if shard_index is None or shard_count is None:
            raise ValueError("Both shard index and shard count are required")
        pairs = shard_country_years(shard_index, shard_count)
        partition_label = f"shard{shard_index:02d}_of_{shard_count:02d}"
    else:
        if chunk is None:
            raise ValueError("A chunk or shard partition is required")
        pairs = partition_country_years(chunk)
        partition_label = f"chunk{chunk}"
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "model": "registered_v42_continuous_portfolio_lp_reproduced_in_scipy_highs_ds",
        "runner": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__).resolve()),
        "panel": str(runner.DEFAULT_PANEL), "panel_sha256": sha256(runner.DEFAULT_PANEL),
        "cost_catalogue": str(LONG_CATALOGUE), "cost_catalogue_sha256": sha256(LONG_CATALOGUE),
        "cost_year": cost_year, "scenario_id": 20, "scenario": scenario,
        "country_years": pairs, "partition_label": partition_label,
        "candidate_count": int(len(candidates)),
        "tail_power_mode": "optimized", "tail_power_upper_multiplier": 1.0,
        "chronologies": ["dispersed", "clustered"], "primary_objective_abs_tolerance_eur": PRIMARY_ABS_TOL_EUR,
    }
    write_json(output / "run_contract.json", contract)
    rows: list[dict[str, Any]] = []
    chronology_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (country, year) in enumerate(pairs, start=1):
        frame = panel[panel["country"].eq(country) & panel["year"].eq(year)].copy()
        if len(frame) != HOURS:
            raise RuntimeError(f"{country}-{year} has {len(frame)} hours")
        new_rows, new_chronology = solve_country_year(
            runner, frame, base_row, candidates, scenario, 20, cost_year
        )
        rows.extend(new_rows); chronology_rows.extend(new_chronology)
        checkpoint = pd.DataFrame(rows)
        checkpoint.to_csv(output / "portfolio_arm_results_unfinished.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        elapsed = time.perf_counter() - started
        print(f"cost {cost_year} {partition_label}: {index}/{len(pairs)} country-years, {len(rows)} LPs, {elapsed/60:.1f} min", flush=True)
    results = pd.DataFrame(rows)
    path = output / "portfolio_arm_results.csv.gz"
    results.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(chronology_rows).to_csv(output / "chronology_conservation.csv", index=False)
    qa = {
        "status": "pass",
        "cost_year": cost_year, "partition_label": partition_label,
        "country_year_count": len(pairs), "candidate_count": len(candidates),
        "expected_row_count": len(pairs) * 2 * (len(candidates) + 1),
        "row_count": len(results), "all_optimal": bool(results["solver_status_name"].eq("optimal").all()),
        "maximum_task_balance_residual_mw": float(results["maximum_task_balance_residual_mw"].max()),
        "maximum_shared_recharge_violation_mw": float(results["maximum_shared_recharge_violation_mw"].max()),
        "maximum_soc_balance_residual_mwh": float(max(results["base_max_soc_balance_residual_mwh"].max(), results["tail_max_soc_balance_residual_mwh"].max())),
        "minimum_tail_power_mw": float(results["tail_power_mw"].min()),
        "maximum_tail_power_minus_upper_mw": float((results["tail_power_mw"] - results["tail_power_upper_mw"]).max()),
        "runtime_minutes": (time.perf_counter() - started) / 60.0,
        "output_sha256": sha256(path),
    }
    if qa["row_count"] != qa["expected_row_count"] or not qa["all_optimal"]:
        qa["status"] = "fail"
    if qa["maximum_task_balance_residual_mw"] > 1.0e-6 or qa["maximum_soc_balance_residual_mwh"] > 1.0e-6:
        qa["status"] = "fail"
    write_json(output / "qa.json", qa)
    (output / "portfolio_arm_results_unfinished.csv.gz").unlink(missing_ok=True)
    print(json.dumps(qa, indent=2))
    if qa["status"] != "pass":
        raise RuntimeError("Production QA failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "production"], default="smoke")
    parser.add_argument("--cost-year", type=int, choices=[2025, 2030, 2035], default=2035)
    parser.add_argument("--chunk", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke()
    else:
        if args.shard_index is not None:
            label = f"shard{args.shard_index:02d}_of_{args.shard_count:02d}"
        else:
            label = f"chunk{args.chunk or 1}"
        destination = args.output or DEFAULT_OUTPUT / f"cost_{args.cost_year}" / label
        production(
            args.cost_year, destination, chunk=args.chunk or (1 if args.shard_index is None else None),
            shard_index=args.shard_index, shard_count=args.shard_count,
        )


if __name__ == "__main__":
    main()
