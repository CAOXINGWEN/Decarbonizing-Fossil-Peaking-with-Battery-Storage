#!/usr/bin/env python3
"""Isolated IPM-first rescue for cost-2030 ES-2017.

This script deliberately does not alter the production runner or its output
directory.  It imports the registered model, changes only the SciPy/HiGHS
algorithm order (IPM first, dual simplex only on failure), and solves the
same S20, 49-candidate, two-chronology experiment with the registered
one-cent primary-objective tolerance.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from scipy.optimize import linprog as scipy_linprog


HERE = Path(__file__).resolve().parent
JOINT = HERE.parents[1]
PRODUCTION_RUNNER = JOINT / "run_bounded_s20_highs.py"
PRODUCTION_SHARD = HERE.parent / "cost_2030" / "shard30_of_66"
PRIMARY_ABS_TOL_EUR = 0.01
TARGET = ("ES", 2017)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_production_runner():
    spec = importlib.util.spec_from_file_location("v44_bounded_production_for_ipm_rescue", PRODUCTION_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the registered bounded-power runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_country_year_output(cost_year: int, country: str, year: int) -> Path:
    root = HERE.parent / f"cost_{cost_year}"
    for path in sorted(root.glob("shard*_of_66/portfolio_arm_results.csv.gz")):
        identity = pd.read_csv(path, usecols=["country", "year"])
        if bool((identity["country"].eq(country) & identity["year"].eq(year)).any()):
            return path
    raise FileNotFoundError(f"No complete {cost_year} output for {country}-{year}")


def main() -> None:
    module = load_production_runner()
    if abs(float(module.PRIMARY_ABS_TOL_EUR) - PRIMARY_ABS_TOL_EUR) > 1.0e-12:
        raise RuntimeError("Production primary-objective tolerance is not one cent")

    traces: list[list[dict[str, Any]]] = []
    current_trace: list[dict[str, Any]] = []

    def ipm_first_linprog(*args, **kwargs):
        requested = str(kwargs.pop("method", "highs"))
        options = dict(kwargs.get("options") or {})
        started = time.perf_counter()
        ipm = scipy_linprog(*args, method="highs-ipm", **kwargs)
        current_trace.append({
            "requested_by_registered_runner": requested,
            "attempt": "highs-ipm",
            "success": bool(ipm.success),
            "status": int(ipm.status),
            "iterations": int(ipm.nit),
            "runtime_seconds": time.perf_counter() - started,
            "message": str(ipm.message),
            "primal_feasibility_tolerance": options.get("primal_feasibility_tolerance"),
            "dual_feasibility_tolerance": options.get("dual_feasibility_tolerance"),
        })
        if ipm.success:
            return ipm
        started = time.perf_counter()
        ds = scipy_linprog(*args, method="highs-ds", **kwargs)
        current_trace.append({
            "requested_by_registered_runner": requested,
            "attempt": "highs-ds_fallback",
            "success": bool(ds.success),
            "status": int(ds.status),
            "iterations": int(ds.nit),
            "runtime_seconds": time.perf_counter() - started,
            "message": str(ds.message),
            "primal_feasibility_tolerance": options.get("primal_feasibility_tolerance"),
            "dual_feasibility_tolerance": options.get("dual_feasibility_tolerance"),
        })
        return ds

    original_solve = module.solve_bounded_arm
    module.linprog = ipm_first_linprog

    def solve_with_trace(*args, **kwargs):
        nonlocal current_trace
        current_trace = []
        result = original_solve(*args, **kwargs)
        trace_copy = [dict(item) for item in current_trace]
        traces.append(trace_copy)
        actual = [item["attempt"] for item in trace_copy]
        result["solver_method"] = "rescue_ipm_first__" + "__".join(actual)
        result["rescue_primary_solver_actual"] = actual[0] if actual else "missing"
        result["rescue_solver_trace_json"] = json.dumps(trace_copy, separators=(",", ":"))
        return result

    module.solve_bounded_arm = solve_with_trace
    runner, panel, base_row, candidates, scenario = module.prepare_inputs(2030, 20)
    country, year = TARGET
    frame = panel[panel["country"].eq(country) & panel["year"].eq(year)].copy()
    if len(frame) != module.HOURS:
        raise RuntimeError(f"Expected 8,760 rows for {country}-{year}, found {len(frame)}")

    contract = {
        "status": "isolated_rescue",
        "purpose": "replace only missing ES-2017 rows from cost-2030 shard30_of_66",
        "production_runner": str(PRODUCTION_RUNNER),
        "production_runner_sha256": sha256(PRODUCTION_RUNNER),
        "model_change": "solver algorithm order only: HiGHS-IPM primary, HiGHS-DS fallback",
        "cost_year": 2030,
        "scenario_id": 20,
        "scenario": scenario,
        "country_years": [[country, year]],
        "candidate_count": int(len(candidates)),
        "chronologies": ["dispersed", "clustered"],
        "primary_objective_abs_tolerance_eur": PRIMARY_ABS_TOL_EUR,
        "tail_power_mode": "optimized",
        "tail_power_upper_multiplier": 1.0,
        "production_directory_untouched": str(PRODUCTION_SHARD),
    }
    write_json(HERE / "rescue_contract.json", contract)

    started = time.perf_counter()
    rows, chronology = module.solve_country_year(
        runner, frame, base_row, candidates, scenario, 20, 2030
    )
    results = pd.DataFrame(rows)
    result_path = HERE / "portfolio_arm_results.csv.gz"
    results.to_csv(result_path, index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(chronology).to_csv(HERE / "chronology_conservation.csv", index=False)

    component_sum = (
        results["base_fixed_cost_eur"] + results["tail_fixed_cost_eur"]
        + results["base_vom_cost_eur"] + results["tail_vom_cost_eur"]
        + results["charge_opportunity_cost_eur"] + results["residual_energy_cost_eur"]
        + results["residual_firm_cost_eur"]
    )
    actual_methods = [item["attempt"] for trace in traces for item in trace]
    primary_methods = [trace[0]["attempt"] if trace else "missing" for trace in traces]
    qa = {
        "status": "pass",
        "country_year_count": 1,
        "row_count": int(len(results)),
        "expected_row_count": 100,
        "candidate_count": int(len(candidates)),
        "all_optimal": bool(results["solver_status_name"].eq("optimal").all()),
        "all_primary_solves_ipm": bool(all(value == "highs-ipm" for value in primary_methods)),
        "solver_call_counts": {value: actual_methods.count(value) for value in sorted(set(actual_methods))},
        "maximum_task_balance_residual_mw": float(results["maximum_task_balance_residual_mw"].max()),
        "maximum_shared_recharge_violation_mw": float(results["maximum_shared_recharge_violation_mw"].max()),
        "maximum_soc_balance_residual_mwh": float(max(
            results["base_max_soc_balance_residual_mwh"].max(),
            results["tail_max_soc_balance_residual_mwh"].max(),
        )),
        "maximum_system_cost_ledger_residual_eur": float(np.abs(component_sum - results["system_cost_eur"]).max()),
        "minimum_tail_power_mw": float(results["tail_power_mw"].min()),
        "maximum_tail_power_minus_upper_mw": float((results["tail_power_mw"] - results["tail_power_upper_mw"]).max()),
        "secondary_tiebreak_success_count": int(results["secondary_tiebreak_success"].sum()),
        "primary_objective_abs_tolerance_eur": PRIMARY_ABS_TOL_EUR,
        "runtime_minutes": (time.perf_counter() - started) / 60.0,
        "output_sha256": sha256(result_path),
    }
    if (
        qa["row_count"] != qa["expected_row_count"]
        or not qa["all_optimal"]
        or not qa["all_primary_solves_ipm"]
        or qa["maximum_task_balance_residual_mw"] > 1.0e-6
        or qa["maximum_shared_recharge_violation_mw"] > 1.0e-6
        or qa["maximum_soc_balance_residual_mwh"] > 1.0e-6
        or qa["maximum_system_cost_ledger_residual_eur"] > 1.0e-6
        or qa["minimum_tail_power_mw"] < -1.0e-8
        or qa["maximum_tail_power_minus_upper_mw"] > 1.0e-8
    ):
        qa["status"] = "fail"
    write_json(HERE / "qa.json", qa)

    comparison_rows = []
    for cost_year in (2025, 2035):
        path = find_country_year_output(cost_year, country, year)
        other = pd.read_csv(path)
        other = other[other["country"].eq(country) & other["year"].eq(year)].copy()
        merged = results.merge(
            other,
            on=["country", "year", "chronology", "arm_type", "config_id", "tech_id", "duration_h"],
            suffixes=("_2030", f"_{cost_year}"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            comparison_rows.append({
                "comparison_cost_year": cost_year,
                "country": country,
                "year": year,
                "chronology": row.chronology,
                "config_id": row.config_id,
                "duration_h": row.duration_h,
                "base_task_difference_mwh": getattr(row, f"base_task_mwh_2030") - getattr(row, f"base_task_mwh_{cost_year}"),
                "raw_recharge_difference_mwh": getattr(row, f"raw_recharge_mwh_2030") - getattr(row, f"raw_recharge_mwh_{cost_year}"),
                "tail_power_mw_2030": getattr(row, "tail_power_mw_2030"),
                "tail_power_mw_other": getattr(row, f"tail_power_mw_{cost_year}"),
                "system_cost_eur_2030": getattr(row, "system_cost_eur_2030"),
                "system_cost_eur_other": getattr(row, f"system_cost_eur_{cost_year}"),
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(HERE / "cross_year_comparison.csv", index=False)

    checkpoint_path = PRODUCTION_SHARD / "portfolio_arm_results_unfinished.csv.gz"
    checkpoint = pd.read_csv(checkpoint_path)
    checkpoint_qa = {
        "path": str(checkpoint_path),
        "sha256": sha256(checkpoint_path),
        "row_count": int(len(checkpoint)),
        "country_years": checkpoint[["country", "year"]].drop_duplicates().values.tolist(),
        "same_columns_as_rescue": bool(set(checkpoint.columns).issubset(set(results.columns))),
        "identity_keys_unique": bool(not checkpoint.duplicated(["country", "year", "chronology", "config_id"]).any()),
        "all_optimal": bool(checkpoint["solver_status_name"].eq("optimal").all()),
    }
    cross_qa = {
        "status": "pass",
        "rescue": qa,
        "cross_year_rows": int(len(comparison)),
        "expected_cross_year_rows": 200,
        "maximum_absolute_base_task_difference_mwh": float(comparison["base_task_difference_mwh"].abs().max()),
        "maximum_absolute_raw_recharge_difference_mwh": float(comparison["raw_recharge_difference_mwh"].abs().max()),
        "es2016_production_checkpoint": checkpoint_qa,
    }
    if (
        qa["status"] != "pass"
        or cross_qa["cross_year_rows"] != cross_qa["expected_cross_year_rows"]
        or cross_qa["maximum_absolute_base_task_difference_mwh"] > 1.0e-8
        or cross_qa["maximum_absolute_raw_recharge_difference_mwh"] > 1.0e-8
        or checkpoint_qa["row_count"] != 100
        or checkpoint_qa["country_years"] != [["ES", 2016]]
        or not checkpoint_qa["same_columns_as_rescue"]
        or not checkpoint_qa["identity_keys_unique"]
        or not checkpoint_qa["all_optimal"]
    ):
        cross_qa["status"] = "fail"
    write_json(HERE / "cross_year_and_checkpoint_qa.json", cross_qa)
    print(json.dumps(cross_qa, indent=2, sort_keys=True))
    if cross_qa["status"] != "pass":
        raise RuntimeError("Isolated rescue QA failed")


if __name__ == "__main__":
    main()
