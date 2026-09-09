#!/usr/bin/env python3
"""V0.5 exact-domain formulation benchmark for v0.4 hard nodes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, vstack

import run_observed_gas_event_task_stage1_fixed_xch_grid_v0_4 as v04


core = v04.core
VERSION = "0.5.0"
MULTIPLIERS = (0.0, 1.0, 2.0)
FORMULATIONS = ("tight_fixed_bigm", "indicator_hull")
TIME_LIMIT_SECONDS = float(
    os.environ.get("STAGE1_V05_TIME_LIMIT_SECONDS", "300")
)
THREADS = int(os.environ.get("STAGE1_V05_THREADS", "16"))
DESIGN = (
    core.ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE1_FORMULATION_BENCHMARK_DESIGN_V0_5.md"
)
OUTPUT = (
    core.ROOT
    / "artifacts"
    / "research"
    / "stage1_formulation_benchmark_v0_5"
)
V04_RESULTS = (
    core.ROOT
    / "artifacts"
    / "research"
    / "stage1_fixed_xch_grid_v0_4"
    / "fixed_xch_grid_results.csv"
)
EVIDENCE_STATUS = "non_evidential_stage1_formulation_benchmark"


def append_linear_rows(
    model: Any,
    rows: list[list[tuple[int, float]]],
    lower: list[float],
    upper: list[float],
) -> None:
    row_index: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for r, terms in enumerate(rows):
        for column, value in terms:
            if value != 0.0:
                row_index.append(r)
                columns.append(int(column))
                data.append(float(value))
    extra = coo_matrix(
        (data, (row_index, columns)),
        shape=(len(rows), model.indices.nvar),
    ).tocsr()
    matrix = vstack(
        [csr_matrix(model.constraint.A), extra], format="csr"
    )
    model.constraint = core.LinearConstraint(
        matrix,
        np.concatenate(
            [np.asarray(model.constraint.lb, float), np.asarray(lower, float)]
        ),
        np.concatenate(
            [np.asarray(model.constraint.ub, float), np.asarray(upper, float)]
        ),
    )


def build_benchmark_model(
    task: np.ndarray,
    recharge: np.ndarray,
    config: Any,
    tangents: list[dict[str, float]],
    multiplier: float,
    formulation: str,
) -> tuple[Any, dict[str, Any]]:
    # Build the common linear base without any mutual-exclusion matrix rows.
    model = core.build_model(
        task,
        recharge,
        config,
        "sparse",
        "indicator",
        tangents,
        core.LFP_REPLACEABLE_FRACTION,
    )
    pdis = model.pdis_mw
    fixed_xch = multiplier * pdis
    lower = np.array(model.bounds.lb, copy=True)
    upper = np.array(model.bounds.ub, copy=True)
    lower[model.indices.xch] = fixed_xch
    upper[model.indices.xch] = fixed_xch
    model.bounds = core.Bounds(lower, upper)

    rows: list[list[tuple[int, float]]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []
    binary_columns = range(
        model.indices.binary_start,
        model.indices.binary_start + len(model.indices.binary_hours),
    )
    for hour, binary in zip(model.indices.binary_hours, binary_columns):
        charge = model.indices.starts["charge"] + hour
        discharge = model.indices.starts["discharge"] + hour
        max_charge = min(
            float(recharge[hour]),
            model.native_charge_power_mw + fixed_xch,
        )
        max_discharge = min(float(task[hour]), pdis)
        if max_charge <= 0.0 or max_discharge <= 0.0:
            raise RuntimeError("Sparse binary hour has a zero disjunct bound")
        if formulation == "tight_fixed_bigm":
            rows.extend(
                [
                    [(charge, 1.0), (binary, -max_charge)],
                    [(discharge, 1.0), (binary, max_discharge)],
                ]
            )
            row_lower.extend([-np.inf, -np.inf])
            row_upper.extend([0.0, max_discharge])
        elif formulation == "indicator_hull":
            rows.append(
                [
                    (charge, 1.0 / max_charge),
                    (discharge, 1.0 / max_discharge),
                ]
            )
            row_lower.append(-np.inf)
            row_upper.append(1.0)
        else:
            raise ValueError(formulation)
    append_linear_rows(model, rows, row_lower, row_upper)
    return model, {
        "valid_inequality_rows_added": len(rows),
        "indicator_constraints_added_at_solver": (
            len(model.indices.binary_hours)
            if formulation == "indicator_hull"
            else 0
        ),
        "integer_feasible_set_changed": False,
    }


def add_linear_constraints(gurobi_model: Any, variables: Any, model: Any) -> None:
    matrix = csr_matrix(model.constraint.A)
    lower = np.asarray(model.constraint.lb, dtype=float)
    upper = np.asarray(model.constraint.ub, dtype=float)
    equality = np.isfinite(lower) & np.isfinite(upper) & np.isclose(
        lower, upper, rtol=0.0, atol=1e-12
    )
    upper_only = np.isfinite(upper) & ~equality
    lower_only = np.isfinite(lower) & ~equality
    if equality.any():
        gurobi_model.addMConstr(
            matrix[equality], variables, "=", upper[equality], name="eq"
        )
    if upper_only.any():
        gurobi_model.addMConstr(
            matrix[upper_only],
            variables,
            "<",
            upper[upper_only],
            name="ub",
        )
    if lower_only.any():
        gurobi_model.addMConstr(
            matrix[lower_only],
            variables,
            ">",
            lower[lower_only],
            name="lb",
        )


def solve_exact_domain_milp(
    model: Any,
    formulation: str,
    log_path: Path,
    mip_start: np.ndarray,
) -> SimpleNamespace:
    import gurobipy as gp

    grb = gp.GRB
    optimization = gp.Model()
    optimization.Params.OutputFlag = 1
    optimization.Params.LogToConsole = 0
    optimization.Params.LogFile = str(log_path)
    optimization.Params.Presolve = 2
    optimization.Params.MIPGap = 1e-12
    optimization.Params.MIPGapAbs = (
        v04.ABSOLUTE_GAP_EUR_PER_FORCED_KW_YEAR_LIMIT
        * v04.FX_CNY_PER_EUR2024
    )
    optimization.Params.FeasibilityTol = 1e-9
    optimization.Params.IntFeasTol = 1e-9
    optimization.Params.OptimalityTol = 1e-9
    optimization.Params.TimeLimit = TIME_LIMIT_SECONDS
    optimization.Params.Threads = THREADS
    optimization.Params.MIPFocus = 3
    optimization.Params.Cuts = 2
    optimization.Params.Symmetry = 2
    optimization.Params.Seed = 20260727

    variable_types = np.full(
        model.indices.nvar, grb.CONTINUOUS, dtype="U1"
    )
    variable_types[model.integrality.astype(bool)] = grb.BINARY
    variables = optimization.addMVar(
        model.indices.nvar,
        lb=np.asarray(model.bounds.lb, dtype=float),
        ub=np.asarray(model.bounds.ub, dtype=float),
        vtype=variable_types,
        name="x",
    )
    add_linear_constraints(optimization, variables, model)
    variable_list = variables.tolist()
    binary_indices = np.arange(
        model.indices.binary_start,
        model.indices.binary_start + len(model.indices.binary_hours),
        dtype=int,
    )
    if formulation == "indicator_hull":
        for hour, binary_column in zip(
            model.indices.binary_hours, binary_indices
        ):
            binary = variable_list[int(binary_column)]
            charge = variable_list[
                model.indices.starts["charge"] + hour
            ]
            discharge = variable_list[
                model.indices.starts["discharge"] + hour
            ]
            optimization.addGenConstrIndicator(
                binary,
                0,
                charge == 0.0,
                name=f"no_charge_{hour}",
            )
            optimization.addGenConstrIndicator(
                binary,
                1,
                discharge == 0.0,
                name=f"no_discharge_{hour}",
            )

    # Start values and branch priorities guide the search but do not restrict
    # any variable domain.
    optimization.setAttr(
        "Start",
        [variable_list[int(index)] for index in binary_indices],
        np.asarray(mip_start, dtype=float).tolist(),
    )
    optimization.setAttr(
        "BranchPriority",
        [variable_list[int(index)] for index in binary_indices],
        [10] * len(binary_indices),
    )
    normalization = model.pdis_mw * 1_000.0
    optimization.setObjective(
        (model.c @ variables) / normalization, grb.MINIMIZE
    )
    started = time.perf_counter()
    optimization.optimize()
    wall = time.perf_counter() - started
    has_incumbent = optimization.SolCount > 0
    if has_incumbent:
        vector = np.asarray(variables.X, dtype=float)
        objective_total_cny = float(optimization.ObjVal) * normalization
    else:
        vector = np.full(model.indices.nvar, np.nan)
        objective_total_cny = float("nan")
    lower_total_cny = float(optimization.ObjBound) * normalization
    return SimpleNamespace(
        status=int(optimization.Status),
        incumbent_available=has_incumbent,
        x=vector,
        objective_total_cny_year=objective_total_cny,
        lower_bound_total_cny_year=lower_total_cny,
        solver_reported_gap=(
            float(optimization.MIPGap) if has_incumbent else float("nan")
        ),
        node_count=float(optimization.NodeCount),
        wall_seconds=wall,
        model_fingerprint=int(optimization.Fingerprint),
        solution_count=int(optimization.SolCount),
    )


def load_mip_start(
    multiplier: float, model: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    path = (
        V04_RESULTS.parent
        / "patterns"
        / f"xch_{multiplier:g}x_pdis_binary_pattern.csv"
    )
    table = pd.read_csv(path)
    expected_hours = np.asarray(model.indices.binary_hours, dtype=int)
    if not np.array_equal(table["model_hour"].to_numpy(int), expected_hours):
        raise RuntimeError("v0.4 MIP-start hours do not match v0.5 model")
    pattern = table["rounded_binary"].to_numpy(float)
    if not np.isin(pattern, [0.0, 1.0]).all():
        raise RuntimeError("v0.4 MIP start is not binary")
    return pattern, {
        "mip_start_source": str(path.resolve()),
        "mip_start_source_sha256": v04.sha256(path),
        "mip_start_pattern_sha256": v04.canonical_array_sha256(
            pattern, "<i8"
        ),
        "mip_start_variables_fixed": False,
    }


def save_pattern(
    formulation: str,
    multiplier: float,
    model: Any,
    raw: np.ndarray,
) -> dict[str, Any]:
    rounded = np.rint(raw).astype(np.int8)
    path = (
        OUTPUT
        / "patterns"
        / f"{formulation}_xch_{multiplier:g}x_binary.csv"
    )
    pd.DataFrame(
        {
            "binary_position": np.arange(len(raw)),
            "model_hour": np.asarray(model.indices.binary_hours, int),
            "raw_binary": raw,
            "rounded_binary": rounded,
        }
    ).to_csv(path, index=False, float_format="%.17g")
    return {
        "raw_binary_pattern_sha256": v04.canonical_array_sha256(
            raw, "<f8"
        ),
        "rounded_binary_pattern_sha256": v04.canonical_array_sha256(
            rounded, "<i8"
        ),
        "pattern_file": str(path.relative_to(OUTPUT)),
        "pattern_file_sha256": v04.sha256(path),
        "maximum_binary_fractionality": float(
            np.minimum(np.abs(raw), np.abs(1.0 - raw)).max(initial=0.0)
        ),
    }


def run_case(
    frame: pd.DataFrame,
    orders: dict[str, list[int]],
    config: Any,
    formulation: str,
    multiplier: float,
) -> dict[str, Any]:
    task_indices = np.asarray(
        core.hourly_order(orders[v04.CELL[0]]), int
    )
    recharge_indices = np.asarray(
        core.hourly_order(orders[v04.CELL[1]]), int
    )
    task = frame["task_c3_q75_mw"].to_numpy(float)[task_indices]
    recharge = frame["r2_potential_recharge_mw"].to_numpy(float)[
        recharge_indices
    ]
    core.build_model.raw_gas_mean_mw = float(
        frame["raw_gas_mean_mw"].iloc[0]
    )
    pdis = (
        core.build_model.raw_gas_mean_mw
        * core.FORCED_POWER_SHARE_OF_RAW_GAS_MEAN
    )
    energy = config.duration_h * pdis
    task_efc_ceiling = float(np.minimum(task, pdis).sum() / energy)
    tangents = [
        core.lfp_tangent(
            0.0, config, energy, core.LFP_REPLACEABLE_FRACTION
        ),
        core.lfp_tangent(
            task_efc_ceiling,
            config,
            energy,
            core.LFP_REPLACEABLE_FRACTION,
        ),
    ]
    model, formulation_meta = build_benchmark_model(
        task, recharge, config, tangents, multiplier, formulation
    )
    snapshot_before = v04.model_snapshot(model)
    mip_start, start_meta = load_mip_start(multiplier, model)
    log_path = (
        OUTPUT
        / "logs"
        / f"{formulation}_xch_{multiplier:g}x_milp.log"
    )
    result = solve_exact_domain_milp(
        model, formulation, log_path, mip_start
    )
    snapshot_after = v04.model_snapshot(model)
    invariant_checks = {
        key.replace("_sha256", "_unchanged"): (
            snapshot_before[key] == snapshot_after[key]
        )
        for key in (
            "constraint_matrix_sha256",
            "constraint_lower_sha256",
            "constraint_upper_sha256",
            "objective_sha256",
            "variable_bounds_sha256",
            "integrality_sha256",
        )
    }
    invariant_path = (
        OUTPUT
        / "invariants"
        / f"{formulation}_xch_{multiplier:g}x_invariants.json"
    )
    invariant_path.write_text(
        json.dumps(
            v04.json_ready(
                {
                    "formulation": formulation,
                    "xch_multiplier": multiplier,
                    "before": snapshot_before,
                    "after": snapshot_after,
                    "checks": invariant_checks,
                    **formulation_meta,
                    **start_meta,
                }
            ),
            indent=2,
        )
        + "\n"
    )
    if not result.incumbent_available:
        raise RuntimeError("MILP returned no incumbent")

    nmode = len(model.indices.binary_hours)
    binary_slice = slice(
        model.indices.binary_start,
        model.indices.binary_start + nmode,
    )
    raw_mode = np.asarray(result.x[binary_slice], float)
    pattern_meta = save_pattern(
        formulation, multiplier, model, raw_mode
    )
    fields = core.fields(model, result.x)
    simultaneous = np.minimum(fields["charge"], fields["discharge"])
    simultaneous_ratio = float(
        simultaneous.sum() / max(fields["discharge"].sum(), 1e-12)
    )
    task_error = task - fields["gas"] - fields["discharge"]
    recharge_error = (
        recharge - fields["recharge_used"] - fields["spill"]
    )
    charge_source_error = fields["charge"] - fields["recharge_used"]
    eta = np.sqrt(config.rte)
    retention = (
        max(0.0, 1.0 - config.self_discharge_pct_day / 100.0)
        ** (1.0 / 24.0)
    )
    soc_error = (
        fields["soc"]
        - retention * np.roll(fields["soc"], 1)
        - eta * fields["charge"]
        + fields["discharge"] / eta
        - (1.0 - retention) * core.SOC_MIN_FRAC * model.energy_mwh
    )

    denominator = pdis * 1_000.0
    upper_total_eur = (
        result.objective_total_cny_year / v04.FX_CNY_PER_EUR2024
    )
    lower_total_eur = (
        result.lower_bound_total_cny_year / v04.FX_CNY_PER_EUR2024
    )
    signed_total_eur = upper_total_eur - lower_total_eur
    signed_normalized = signed_total_eur / denominator
    relative = signed_total_eur / max(abs(upper_total_eur), 1.0)
    rated_efc = float(fields["discharge"].sum() / model.energy_mwh)
    true_capital, _, _ = core.lfp_capital_function(
        rated_efc,
        config,
        model.energy_mwh,
        core.LFP_REPLACEABLE_FRACTION,
    )
    epigraph = float(result.x[model.indices.capital])
    kink = config.cycle_life_cycles / config.calendar_life_years

    gates = {
        "signed_gap_nonnegative_pass": (
            signed_total_eur
            >= -v04.SIGNED_GAP_NEGATIVE_TOLERANCE_EUR_YEAR
        ),
        "relative_gap_pass": relative <= v04.RELATIVE_GAP_LIMIT,
        "absolute_gap_pass": (
            signed_normalized
            <= v04.ABSOLUTE_GAP_EUR_PER_FORCED_KW_YEAR_LIMIT
        ),
        "dual_gap_and_pass": (
            signed_total_eur
            >= -v04.SIGNED_GAP_NEGATIVE_TOLERANCE_EUR_YEAR
            and relative <= v04.RELATIVE_GAP_LIMIT
            and signed_normalized
            <= v04.ABSOLUTE_GAP_EUR_PER_FORCED_KW_YEAR_LIMIT
        ),
        "model_invariants_pass": all(invariant_checks.values()),
        "binary_fractionality_pass": (
            pattern_meta["maximum_binary_fractionality"] <= 1e-9
        ),
        "simultaneous_hourly_pass": (
            float(simultaneous.max(initial=0.0))
            <= core.SIMULTANEOUS_HOURLY_TOLERANCE_MW
        ),
        "simultaneous_ratio_pass": (
            simultaneous_ratio <= core.SIMULTANEOUS_RATIO_TOLERANCE
        ),
        "task_balance_pass": (
            float(np.abs(task_error).max(initial=0.0))
            <= core.ABSOLUTE_BALANCE_TOLERANCE
        ),
        "recharge_balance_pass": (
            float(np.abs(recharge_error).max(initial=0.0))
            <= core.ABSOLUTE_BALANCE_TOLERANCE
        ),
        "charge_source_pass": (
            float(np.abs(charge_source_error).max(initial=0.0))
            <= core.ABSOLUTE_BALANCE_TOLERANCE
        ),
        "soc_balance_pass": (
            float(np.abs(soc_error).max(initial=0.0))
            <= core.ABSOLUTE_BALANCE_TOLERANCE
        ),
        "finite_node_lfp_exact_pass": (
            rated_efc <= kink + 1e-9
            and abs(true_capital - epigraph)
            / max(abs(true_capital), 1.0)
            <= 1e-10
        ),
    }
    return {
        "country": v04.COUNTRY,
        "year": v04.YEAR,
        "config_id": v04.CONFIG_ID,
        "cell": v04.CELL,
        "formulation": formulation,
        "xch_multiplier": multiplier,
        "forced_discharge_power_mw": pdis,
        "normalization_forced_kw": denominator,
        "milp_status": result.status,
        "milp_solution_count": result.solution_count,
        "milp_node_count": result.node_count,
        "milp_wall_seconds": result.wall_seconds,
        "gurobi_model_fingerprint": result.model_fingerprint,
        "milp_incumbent_total_eur2024_year": upper_total_eur,
        "milp_global_lower_bound_total_eur2024_year": lower_total_eur,
        "milp_incumbent_eur2024_per_forced_kw_year": (
            upper_total_eur / denominator
        ),
        "milp_global_lower_bound_eur2024_per_forced_kw_year": (
            lower_total_eur / denominator
        ),
        "signed_bound_width_total_eur2024_year": signed_total_eur,
        "signed_bound_width_eur2024_per_forced_kw_year": (
            signed_normalized
        ),
        "certified_relative_bound_width": relative,
        "solver_reported_gap": result.solver_reported_gap,
        "rated_output_efc": rated_efc,
        "maximum_simultaneous_charge_discharge_mw": float(
            simultaneous.max(initial=0.0)
        ),
        "simultaneous_energy_ratio": simultaneous_ratio,
        "maximum_task_balance_error_mw": float(
            np.abs(task_error).max(initial=0.0)
        ),
        "maximum_recharge_balance_error_mw": float(
            np.abs(recharge_error).max(initial=0.0)
        ),
        "maximum_charge_source_error_mw": float(
            np.abs(charge_source_error).max(initial=0.0)
        ),
        "maximum_soc_balance_error_mwh": float(
            np.abs(soc_error).max(initial=0.0)
        ),
        **formulation_meta,
        **start_meta,
        **pattern_meta,
        "invariant_file": str(invariant_path.relative_to(OUTPUT)),
        "invariant_file_sha256": v04.sha256(invariant_path),
        **gates,
        "case_pass": bool(all(gates.values())),
        "incumbent_pattern_fixed": False,
        "optimal_domain_changed": False,
        "manuscript_evidence_admitted": False,
        "evidence_status": EVIDENCE_STATUS,
        "log_file": str(log_path.relative_to(OUTPUT)),
    }


def write_manifest(qa_pass: bool) -> None:
    files = [
        path
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "version": VERSION,
        "qa_pass": qa_pass,
        "files": {
            str(path.relative_to(OUTPUT)): v04.sha256(path)
            for path in files
        },
        "external_inputs": {
            str(path.resolve()): v04.sha256(path)
            for path in (
                v04.STAGE0,
                v04.STAGE0_QA,
                DESIGN,
                Path(v04.__file__).resolve(),
                Path(core.__file__).resolve(),
                core.CONFIGURATIONS,
                V04_RESULTS,
                Path(__file__).resolve(),
            )
        },
        "manuscript_evidence_admitted": False,
        "evidence_status": EVIDENCE_STATUS,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(v04.json_ready(manifest), indent=2) + "\n"
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {OUTPUT}")
    for directory in ("logs", "patterns", "invariants"):
        (OUTPUT / directory).mkdir(parents=True, exist_ok=True)
    failures_path = OUTPUT / "failure_trace.jsonl"
    failures_path.write_text("")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    setup_qa = v04.validate_stage0()
    frame = v04.load_country_year()
    orders, _ = core.make_day_orders(frame, v04.COUNTRY, v04.YEAR)
    config = core.load_configuration(v04.CONFIG_ID)
    for formulation in FORMULATIONS:
        for multiplier in MULTIPLIERS:
            try:
                row = run_case(
                    frame, orders, config, formulation, multiplier
                )
                rows.append(row)
                print(json.dumps(v04.json_ready(row), ensure_ascii=False))
            except Exception as exc:
                failure = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "formulation": formulation,
                    "xch_multiplier": multiplier,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "runner_sha256": v04.sha256(Path(__file__)),
                }
                failures.append(failure)
                with failures_path.open("a") as stream:
                    stream.write(
                        json.dumps(
                            v04.json_ready(failure), ensure_ascii=False
                        )
                        + "\n"
                    )
                rows.append(
                    {
                        "formulation": formulation,
                        "xch_multiplier": multiplier,
                        "case_pass": False,
                        "failure_message": str(exc),
                        "manuscript_evidence_admitted": False,
                        "evidence_status": EVIDENCE_STATUS,
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "formulation_benchmark_results.csv", index=False)
    expected_cases = len(FORMULATIONS) * len(MULTIPLIERS)
    all_cases_present = (
        len(results) == expected_cases
        and results[["formulation", "xch_multiplier"]]
        .drop_duplicates()
        .shape[0]
        == expected_cases
    )
    formulation_summary = (
        results.groupby("formulation", dropna=False)
        .agg(
            cases=("xch_multiplier", "count"),
            passes=("case_pass", "sum"),
            maximum_wall_seconds=("milp_wall_seconds", "max"),
            maximum_relative_width=(
                "certified_relative_bound_width",
                "max",
            ),
            maximum_absolute_width=(
                "signed_bound_width_eur2024_per_forced_kw_year",
                "max",
            ),
        )
        .reset_index()
        .to_dict(orient="records")
        if "milp_wall_seconds" in results
        else []
    )
    all_cases_pass = bool(
        all_cases_present
        and results["case_pass"].fillna(False).astype(bool).all()
        and not failures
    )
    qa = {
        "status": "pass" if all_cases_pass else "fail",
        "qa_pass": all_cases_pass,
        "version": VERSION,
        "scope": "DE-2023 LFP4 DD exact-domain formulation benchmark",
        "formulations": list(FORMULATIONS),
        "multipliers": list(MULTIPLIERS),
        "expected_cases": expected_cases,
        "all_cases_present": all_cases_present,
        "all_cases_pass": all_cases_pass,
        "failure_count": len(failures),
        "formulation_summary": formulation_summary,
        "relative_gap_limit": v04.RELATIVE_GAP_LIMIT,
        "absolute_gap_limit_eur2024_per_forced_kw_year": (
            v04.ABSOLUTE_GAP_EUR_PER_FORCED_KW_YEAR_LIMIT
        ),
        "gap_gate_logic": "signed nonnegative AND relative AND absolute",
        "incumbent_pattern_fixed": False,
        "optimal_domain_changed": False,
        "mip_start_used": True,
        "time_limit_seconds_per_case": TIME_LIMIT_SECONDS,
        "threads": THREADS,
        "stage0_sha256": v04.sha256(v04.STAGE0),
        "stage0_dataset_version": setup_qa.get("dataset_version"),
        "canonical_gas_field": v04.CANONICAL_GAS_FIELD,
        "runner_sha256": v04.sha256(Path(__file__)),
        "design_sha256": v04.sha256(DESIGN),
        "v04_results_sha256": v04.sha256(V04_RESULTS),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "manuscript_evidence_admitted": False,
        "evidence_status": EVIDENCE_STATUS,
    }
    (OUTPUT / "QA.json").write_text(
        json.dumps(v04.json_ready(qa), indent=2) + "\n"
    )
    write_manifest(all_cases_pass)
    raise SystemExit(0 if all_cases_pass else 2)


if __name__ == "__main__":
    main()
