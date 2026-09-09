#!/usr/bin/env python3
"""Hardened v0.4 fixed-xch numerical smoke on final Stage-0 v0.2.

This runner creates a new artifact and does not modify any v0.1--v0.3 runner
or result.  It is deliberately limited to DE-2023 / LFP 4 h / DD and the six
fixed extra-charge-power nodes 0--5 times forced discharge power.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import run_observed_gas_event_task_stage1_v0_3 as core


VERSION = "0.4.0"
COUNTRY = "DE"
YEAR = 2023
CONFIG_ID = "li_lfp_4h"
CELL = "DD"
FORMULATION = "perspective_bigm"
GRID_MULTIPLIERS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
MIP_TIME_LIMIT_SECONDS = float(
    os.environ.get("STAGE1_V04_MIP_TIME_LIMIT_SECONDS", "900")
)
LP_TIME_LIMIT_SECONDS = 60.0
FX_CNY_PER_EUR2024 = 7.787469922
RELATIVE_GAP_LIMIT = 1e-6
ABSOLUTE_GAP_EUR_PER_FORCED_KW_YEAR_LIMIT = 0.01
SIGNED_GAP_NEGATIVE_TOLERANCE_EUR_YEAR = 1e-6
BINARY_ROUNDING_TOLERANCE = 1e-9

STAGE0 = (
    core.ROOT
    / "processed"
    / "observed_gas_event_task_definitions_v0_2.csv.gz"
)
STAGE0_QA = (
    core.ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE0_V0_2_QA.json"
)
EXPECTED_STAGE0_SHA256 = (
    "78c890a4b8b08808858e0a387fd3031271afb8459b453331400e9c0dab5a1e53"
)
EXPECTED_SOURCE_PANEL_SHA256 = (
    "31cd2b0f4b8a2f38e37addf77296affe9629155bf7a97870e7e650aebaac26ce"
)
CANONICAL_GAS_FIELD = "observed_aggregate_fossil_gas_generation_mw"
DESIGN = (
    core.ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE1_FIXED_XCH_GRID_HARDENED_DESIGN_V0_4.md"
)
OUTPUT = (
    core.ROOT
    / "artifacts"
    / "research"
    / "stage1_fixed_xch_grid_v0_4"
)
EVIDENCE_STATUS = "non_evidential_stage1_hardened_numerical_smoke"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_sha256(array: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(canonical.shape), "dtype": canonical.dtype.str},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def csr_sha256(matrix: Any) -> str:
    canonical = csr_matrix(matrix, dtype=np.float64, copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(canonical.shape), "format": "csr"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for array, dtype in (
        (canonical.indptr, "<i8"),
        (canonical.indices, "<i8"),
        (canonical.data, "<f8"),
    ):
        normalized = np.ascontiguousarray(np.asarray(array, dtype=dtype))
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def bounds_sha256(lower: np.ndarray, upper: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (lower, upper):
        normalized = np.ascontiguousarray(
            np.asarray(array, dtype="<f8")
        )
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def model_snapshot(model: Any) -> dict[str, Any]:
    matrix = csr_matrix(model.constraint.A)
    return {
        "constraint_matrix_sha256": csr_sha256(matrix),
        "constraint_lower_sha256": canonical_array_sha256(
            model.constraint.lb, "<f8"
        ),
        "constraint_upper_sha256": canonical_array_sha256(
            model.constraint.ub, "<f8"
        ),
        "objective_sha256": canonical_array_sha256(model.c, "<f8"),
        "variable_bounds_sha256": bounds_sha256(
            model.bounds.lb, model.bounds.ub
        ),
        "integrality_sha256": canonical_array_sha256(
            model.integrality, "<i8"
        ),
        "matrix_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
        "matrix_nonzeros": int(matrix.nnz),
    }


def json_ready(value: Any) -> Any:
    return core.json_ready(value)


def validate_stage0() -> dict[str, Any]:
    if sha256(STAGE0) != EXPECTED_STAGE0_SHA256:
        raise RuntimeError(
            f"Stage-0 v0.2 SHA mismatch: {sha256(STAGE0)}"
        )
    qa = json.loads(STAGE0_QA.read_text())
    if qa.get("status") != "pass":
        raise RuntimeError("Stage-0 v0.2 QA is not pass")
    failed = [
        key
        for key, value in qa.get("checks", {}).items()
        if value is not True
    ]
    if failed:
        raise RuntimeError(f"Stage-0 v0.2 failed checks: {failed}")
    hashes = qa.get("hashes", {})
    if hashes.get("hourly_output_sha256") != EXPECTED_STAGE0_SHA256:
        raise RuntimeError("Stage-0 QA/output SHA cross-check failed")
    if (
        hashes.get("source_panel_sha256")
        != EXPECTED_SOURCE_PANEL_SHA256
    ):
        raise RuntimeError("Stage-0 source panel SHA cross-check failed")
    canonical = qa.get("canonical_field", {})
    if canonical.get("name") != CANONICAL_GAS_FIELD:
        raise RuntimeError("Stage-0 canonical field contract changed")
    return qa


def load_country_year() -> pd.DataFrame:
    required = [
        "timestamp_utc",
        "country",
        "year",
        "source_hour_id",
        "source_day_id",
        "source_hour_of_day",
        "season_segment",
        CANONICAL_GAS_FIELD,
        "raw_gas_mean_mw",
        "r2_potential_recharge_mw",
        "event_flag_q75",
        "task_c3_q75_mw",
    ]
    frame = pd.read_csv(STAGE0, usecols=required)
    frame = frame[
        frame["country"].eq(COUNTRY) & frame["year"].eq(YEAR)
    ].copy()
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="raise"
    )
    frame = frame.sort_values("source_hour_id").reset_index(drop=True)
    if len(frame) != 8760:
        raise RuntimeError("Expected exactly 8,760 DE-2023 rows")
    if not np.array_equal(
        frame["source_hour_id"].to_numpy(int), np.arange(8760)
    ):
        raise RuntimeError("source_hour_id is not canonical")
    if not np.array_equal(
        frame["source_day_id"].to_numpy(int), np.arange(8760) // 24
    ):
        raise RuntimeError("source_day_id is not canonical")
    if not np.array_equal(
        frame["source_hour_of_day"].to_numpy(int),
        np.arange(8760) % 24,
    ):
        raise RuntimeError("source_hour_of_day is not canonical")
    numeric = [
        CANONICAL_GAS_FIELD,
        "raw_gas_mean_mw",
        "r2_potential_recharge_mw",
        "task_c3_q75_mw",
    ]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise RuntimeError("Non-finite canonical Stage-0 values")
    if (frame[numeric] < 0.0).any().any():
        raise RuntimeError("Negative canonical Stage-0 values")
    observed = frame[CANONICAL_GAS_FIELD].to_numpy(float)
    task = frame["task_c3_q75_mw"].to_numpy(float)
    if np.any(task > observed + 1e-9):
        raise RuntimeError("C3 task exceeds canonical fossil-gas field")
    if frame["raw_gas_mean_mw"].nunique() != 1:
        raise RuntimeError("raw_gas_mean_mw is not constant")
    if abs(float(frame["raw_gas_mean_mw"].iloc[0]) - observed.mean()) > 1e-9:
        raise RuntimeError("raw_gas_mean_mw differs from canonical mean")
    flags = (
        frame["event_flag_q75"]
        if frame["event_flag_q75"].dtype == bool
        else frame["event_flag_q75"]
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
    if np.any((task > 1e-12) & ~np.asarray(flags, dtype=bool)):
        raise RuntimeError("Positive C3 task outside q75 event days")
    return frame


def configure_core_numerics() -> None:
    # The core objective is internally normalized in CNY per forced kW-year.
    # Set a stricter relative stop and the EUR absolute target converted back
    # to that internal unit. Final admission independently checks both gates.
    core.MIP_RELATIVE_GAP_TOLERANCE = 1e-12
    core.MIP_ABSOLUTE_GAP_CNY_PER_FORCED_KW_YEAR = (
        ABSOLUTE_GAP_EUR_PER_FORCED_KW_YEAR_LIMIT
        * FX_CNY_PER_EUR2024
    )


def save_pattern(
    path: Path,
    model: Any,
    raw_mode: np.ndarray,
    rounded_mode: np.ndarray,
    task_indices: np.ndarray,
    recharge_indices: np.ndarray,
) -> dict[str, Any]:
    hours = np.asarray(model.indices.binary_hours, dtype=int)
    table = pd.DataFrame(
        {
            "binary_position": np.arange(len(hours), dtype=int),
            "model_hour": hours,
            "task_source_hour_id": task_indices[hours],
            "recharge_source_hour_id": recharge_indices[hours],
            "raw_binary": raw_mode,
            "rounded_binary": rounded_mode.astype(np.int8),
        }
    )
    table.to_csv(path, index=False, float_format="%.17g")
    return {
        "pattern_file": str(path.relative_to(OUTPUT)),
        "pattern_file_sha256": sha256(path),
        "raw_binary_pattern_sha256": canonical_array_sha256(
            raw_mode, "<f8"
        ),
        "rounded_binary_pattern_sha256": canonical_array_sha256(
            rounded_mode, "<i8"
        ),
    }


def run_node(
    frame: pd.DataFrame,
    orders: dict[str, list[int]],
    config: Any,
    multiplier: float,
) -> dict[str, Any]:
    label = f"xch_{multiplier:g}x_pdis"
    mip_log = OUTPUT / "logs" / f"{label}_mip.log"
    lp_log = OUTPUT / "logs" / f"{label}_fixed_pattern_lp.log"
    pattern_path = OUTPUT / "patterns" / f"{label}_binary_pattern.csv"
    invariant_path = OUTPUT / "invariants" / f"{label}_model_hashes.json"

    task_indices = np.asarray(core.hourly_order(orders[CELL[0]]), dtype=int)
    recharge_indices = np.asarray(
        core.hourly_order(orders[CELL[1]]), dtype=int
    )
    task = frame["task_c3_q75_mw"].to_numpy(float)[task_indices]
    recharge = frame["r2_potential_recharge_mw"].to_numpy(float)[
        recharge_indices
    ]
    raw_mean = float(frame["raw_gas_mean_mw"].iloc[0])
    core.build_model.raw_gas_mean_mw = raw_mean
    pdis = raw_mean * core.FORCED_POWER_SHARE_OF_RAW_GAS_MEAN
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
    model = core.build_model(
        task,
        recharge,
        config,
        "sparse",
        FORMULATION,
        tangents,
        core.LFP_REPLACEABLE_FRACTION,
    )
    fixed_xch = multiplier * pdis
    lower = np.array(model.bounds.lb, copy=True)
    upper = np.array(model.bounds.ub, copy=True)
    lower[model.indices.xch] = fixed_xch
    upper[model.indices.xch] = fixed_xch
    model.bounds = core.Bounds(lower, upper)
    pre_snapshot = model_snapshot(model)
    pre_lower = np.array(model.bounds.lb, copy=True)
    pre_upper = np.array(model.bounds.ub, copy=True)
    pre_integrality = np.array(model.integrality, copy=True)

    os.environ["STAGE1_GUROBI_LOG"] = str(mip_log)
    os.environ["STAGE1_GUROBI_MIPFOCUS"] = "3"
    started = time.perf_counter()
    mip_result = core.solve_once_gurobi(
        model, MIP_TIME_LIMIT_SECONDS
    )
    mip_seconds = time.perf_counter() - started
    if not mip_result.incumbent_available:
        raise RuntimeError("MILP returned no incumbent")

    nmode = len(model.indices.binary_hours)
    binary_slice = slice(
        model.indices.binary_start,
        model.indices.binary_start + nmode,
    )
    raw_mode = np.asarray(mip_result.x[binary_slice], dtype=float)
    rounded_mode = np.rint(raw_mode).astype(np.int8)
    binary_distance = float(
        np.abs(raw_mode - rounded_mode).max(initial=0.0)
    )
    pattern_meta = save_pattern(
        pattern_path,
        model,
        raw_mode,
        rounded_mode,
        task_indices,
        recharge_indices,
    )
    if binary_distance > BINARY_ROUNDING_TOLERANCE:
        raise RuntimeError(
            f"Binary rounding distance {binary_distance} exceeds tolerance"
        )

    post_lower = np.array(pre_lower, copy=True)
    post_upper = np.array(pre_upper, copy=True)
    post_integrality = np.zeros_like(pre_integrality)
    post_lower[binary_slice] = rounded_mode
    post_upper[binary_slice] = rounded_mode
    model.bounds = core.Bounds(post_lower, post_upper)
    model.integrality = post_integrality
    post_snapshot = model_snapshot(model)

    changed_lower = np.flatnonzero(post_lower != pre_lower)
    changed_upper = np.flatnonzero(post_upper != pre_upper)
    changed_integrality = np.flatnonzero(
        post_integrality != pre_integrality
    )
    allowed = np.arange(
        model.indices.binary_start,
        model.indices.binary_start + nmode,
        dtype=int,
    )
    allowed_set = set(allowed.tolist())
    unexpected = sorted(
        (
            set(changed_lower.tolist())
            | set(changed_upper.tolist())
            | set(changed_integrality.tolist())
        )
        - allowed_set
    )
    invariant_checks = {
        "constraint_matrix_unchanged": (
            pre_snapshot["constraint_matrix_sha256"]
            == post_snapshot["constraint_matrix_sha256"]
        ),
        "constraint_lower_unchanged": (
            pre_snapshot["constraint_lower_sha256"]
            == post_snapshot["constraint_lower_sha256"]
        ),
        "constraint_upper_unchanged": (
            pre_snapshot["constraint_upper_sha256"]
            == post_snapshot["constraint_upper_sha256"]
        ),
        "objective_unchanged": (
            pre_snapshot["objective_sha256"]
            == post_snapshot["objective_sha256"]
        ),
        "only_binary_bounds_or_integrality_changed": not unexpected,
        "all_original_binary_integrality_removed": (
            set(changed_integrality.tolist()) == allowed_set
        ),
    }
    invariant_record = {
        "node": label,
        "before_fixed_pattern": pre_snapshot,
        "after_fixed_pattern": post_snapshot,
        "changed_lower_bound_indices": changed_lower.tolist(),
        "changed_upper_bound_indices": changed_upper.tolist(),
        "changed_integrality_indices": changed_integrality.tolist(),
        "allowed_binary_indices": allowed.tolist(),
        "unexpected_changed_indices": unexpected,
        "checks": invariant_checks,
    }
    invariant_path.write_text(
        json.dumps(json_ready(invariant_record), indent=2) + "\n"
    )
    if not all(invariant_checks.values()):
        raise RuntimeError(
            f"Fixed-pattern model invariant failure: {invariant_checks}"
        )

    os.environ["STAGE1_GUROBI_LOG"] = str(lp_log)
    started = time.perf_counter()
    lp_result = core.solve_once_gurobi(model, LP_TIME_LIMIT_SECONDS)
    lp_seconds = time.perf_counter() - started
    if not lp_result.incumbent_available or lp_result.raw_solver_status != 2:
        raise RuntimeError("Fixed-pattern LP did not solve to optimality")

    fields = core.fields(model, lp_result.x)
    mode = lp_result.x[binary_slice]
    product = lp_result.x[
        model.indices.product_start :
        model.indices.product_start + nmode
    ]
    xch = float(lp_result.x[model.indices.xch])
    kg = float(lp_result.x[model.indices.kg])
    denominator_forced_kw = pdis * 1_000.0

    milp_incumbent_total_eur = (
        float(mip_result.fun) / FX_CNY_PER_EUR2024
    )
    milp_lower_total_eur = (
        float(mip_result.objective_bound) / FX_CNY_PER_EUR2024
    )
    fixed_pattern_upper_total_eur = (
        float(lp_result.fun) / FX_CNY_PER_EUR2024
    )
    signed_gap_total_eur = (
        fixed_pattern_upper_total_eur - milp_lower_total_eur
    )
    signed_gap_normalized_eur = (
        signed_gap_total_eur / denominator_forced_kw
    )
    relative_gap = signed_gap_total_eur / max(
        abs(fixed_pattern_upper_total_eur), 1.0
    )
    signed_gap_pass = (
        signed_gap_total_eur
        >= -SIGNED_GAP_NEGATIVE_TOLERANCE_EUR_YEAR
    )
    relative_gap_pass = (
        signed_gap_pass and relative_gap <= RELATIVE_GAP_LIMIT
    )
    absolute_gap_pass = (
        signed_gap_pass
        and signed_gap_normalized_eur
        <= ABSOLUTE_GAP_EUR_PER_FORCED_KW_YEAR_LIMIT
    )

    def normalized(total: float) -> float:
        return total / denominator_forced_kw

    ledger = {
        "milp_incumbent_total_eur2024_year": milp_incumbent_total_eur,
        "milp_incumbent_eur2024_per_forced_kw_year": normalized(
            milp_incumbent_total_eur
        ),
        "milp_global_lower_bound_total_eur2024_year": (
            milp_lower_total_eur
        ),
        "milp_global_lower_bound_eur2024_per_forced_kw_year": (
            normalized(milp_lower_total_eur)
        ),
        "fixed_pattern_feasible_upper_bound_total_eur2024_year": (
            fixed_pattern_upper_total_eur
        ),
        "fixed_pattern_feasible_upper_bound_eur2024_per_forced_kw_year": (
            normalized(fixed_pattern_upper_total_eur)
        ),
        "signed_bound_width_total_eur2024_year": signed_gap_total_eur,
        "signed_bound_width_eur2024_per_forced_kw_year": (
            signed_gap_normalized_eur
        ),
    }
    ledger_residuals = {
        key.replace("_total_eur2024_year", "_ledger_residual_eur2024_year"):
        abs(
            value
            - ledger[
                key.replace(
                    "_total_eur2024_year",
                    "_eur2024_per_forced_kw_year",
                )
            ]
            * denominator_forced_kw
        )
        for key, value in ledger.items()
        if key.endswith("_total_eur2024_year")
    }
    ledger_tolerance = max(
        1e-6,
        1e-10
        * max(
            abs(fixed_pattern_upper_total_eur),
            abs(milp_lower_total_eur),
            1.0,
        ),
    )

    simultaneous = np.minimum(fields["charge"], fields["discharge"])
    simultaneous_ratio = float(
        simultaneous.sum() / max(fields["discharge"].sum(), 1e-12)
    )
    eta = math.sqrt(config.rte)
    retention = (
        max(0.0, 1.0 - config.self_discharge_pct_day / 100.0)
        ** (1.0 / 24.0)
    )
    soc_error = (
        fields["soc"]
        - retention * np.roll(fields["soc"], 1)
        - eta * fields["charge"]
        + fields["discharge"] / eta
        - (1.0 - retention)
        * core.SOC_MIN_FRAC
        * model.energy_mwh
    )
    task_error = task - fields["gas"] - fields["discharge"]
    recharge_error = (
        recharge - fields["recharge_used"] - fields["spill"]
    )
    charge_source_error = fields["charge"] - fields["recharge_used"]
    extra_upper = core.MAX_EXTRA_CHARGE_POWER_MULTIPLE * pdis
    envelope_violation = max(
        float(np.maximum(-product, 0.0).max(initial=0.0)),
        float(np.maximum(product - xch, 0.0).max(initial=0.0)),
        float(
            np.maximum(product - extra_upper * mode, 0.0).max(
                initial=0.0
            )
        ),
        float(
            np.maximum(
                xch + extra_upper * mode - extra_upper - product, 0.0
            ).max(initial=0.0)
        ),
    )
    rated_efc = float(fields["discharge"].sum() / model.energy_mwh)
    true_capital, _, replacement_life = core.lfp_capital_function(
        rated_efc,
        config,
        model.energy_mwh,
        core.LFP_REPLACEABLE_FRACTION,
    )
    epigraph_capital = float(lp_result.x[model.indices.capital])
    capital_relative_error = abs(
        true_capital - epigraph_capital
    ) / max(abs(true_capital), 1.0)
    cycle_life_kink_efc = (
        config.cycle_life_cycles / config.calendar_life_years
    )

    gates = {
        "milp_incumbent_available": mip_result.incumbent_available,
        "binary_rounding_pass": (
            binary_distance <= BINARY_ROUNDING_TOLERANCE
        ),
        "fixed_pattern_lp_optimal": lp_result.raw_solver_status == 2,
        "signed_gap_nonnegative_pass": signed_gap_pass,
        "relative_gap_le_1e_6_pass": relative_gap_pass,
        "absolute_gap_le_0_01_eur_per_forced_kw_year_pass": (
            absolute_gap_pass
        ),
        "dual_gap_gate_and_pass": (
            signed_gap_pass and relative_gap_pass and absolute_gap_pass
        ),
        "ledger_identity_pass": (
            max(ledger_residuals.values(), default=0.0)
            <= ledger_tolerance
        ),
        "matrix_objective_invariants_pass": all(
            invariant_checks.values()
        ),
        "maximum_simultaneous_pass": (
            float(simultaneous.max(initial=0.0))
            <= core.SIMULTANEOUS_HOURLY_TOLERANCE_MW
        ),
        "simultaneous_ratio_pass": (
            simultaneous_ratio <= core.SIMULTANEOUS_RATIO_TOLERANCE
        ),
        "binary_fractionality_pass": (
            float(
                np.minimum(np.abs(mode), np.abs(1.0 - mode)).max(
                    initial=0.0
                )
            )
            <= 1e-9
        ),
        "product_identity_pass": (
            float(np.abs(product - xch * mode).max(initial=0.0))
            <= 1e-9
        ),
        "envelope_pass": envelope_violation <= 1e-9,
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
        "finite_node_lfp_solution_point_exact_pass": (
            rated_efc <= cycle_life_kink_efc + 1e-9
            and capital_relative_error <= 1e-10
        ),
    }
    return {
        "country": COUNTRY,
        "year": YEAR,
        "config_id": CONFIG_ID,
        "cell": CELL,
        "formulation": FORMULATION,
        "xch_multiplier": multiplier,
        "forced_discharge_power_mw": pdis,
        "normalization_forced_kw": denominator_forced_kw,
        "fixed_extra_charge_power_mw": fixed_xch,
        "native_charge_power_mw": model.native_charge_power_mw,
        "total_charge_power_mw": model.native_charge_power_mw + xch,
        "gas_capacity_mw": kg,
        "mip_solver_status": mip_result.raw_solver_status,
        "mip_solver_reported_gap": float(mip_result.mip_gap),
        "mip_node_count": int(mip_result.mip_node_count),
        "mip_solve_seconds": mip_seconds,
        "fixed_pattern_lp_status": lp_result.raw_solver_status,
        "fixed_pattern_lp_solve_seconds": lp_seconds,
        "total_solve_seconds": mip_seconds + lp_seconds,
        **ledger,
        "certified_relative_bound_width": relative_gap,
        **ledger_residuals,
        "ledger_tolerance_eur2024_year": ledger_tolerance,
        "maximum_pre_fix_binary_distance": binary_distance,
        **pattern_meta,
        "model_invariant_file": str(invariant_path.relative_to(OUTPUT)),
        "model_invariant_file_sha256": sha256(invariant_path),
        "rated_output_efc": rated_efc,
        "cycle_life_kink_efc": cycle_life_kink_efc,
        "lfp_replacement_life_years": replacement_life,
        "lfp_true_capital_total_eur2024_year": (
            true_capital / FX_CNY_PER_EUR2024
        ),
        "lfp_epigraph_capital_total_eur2024_year": (
            epigraph_capital / FX_CNY_PER_EUR2024
        ),
        "lfp_solution_point_relative_error": capital_relative_error,
        "pwl_certificate_scope": (
            "finite_six_node_solution_points_only_not_full_domain"
        ),
        "binary_hours": nmode,
        "binary_charge_mode_hours": int((mode > 0.5).sum()),
        "binary_discharge_mode_hours": int((mode <= 0.5).sum()),
        "maximum_binary_fractionality": float(
            np.minimum(np.abs(mode), np.abs(1.0 - mode)).max(
                initial=0.0
            )
        ),
        "maximum_product_identity_error_mw": float(
            np.abs(product - xch * mode).max(initial=0.0)
        ),
        "maximum_mccormick_envelope_violation_mw": envelope_violation,
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
        "cyclic_soc_transition_residual_mwh": float(abs(soc_error[0])),
        **gates,
        "node_pass": bool(all(gates.values())),
        "mip_log": str(mip_log.relative_to(OUTPUT)),
        "fixed_pattern_lp_log": str(lp_log.relative_to(OUTPUT)),
        "manuscript_evidence_admitted": False,
        "evidence_status": EVIDENCE_STATUS,
    }


def environment_record() -> dict[str, Any]:
    try:
        import gurobipy as gp

        gurobi_version = ".".join(map(str, gp.gurobi.version()))
    except Exception as exc:  # pragma: no cover - provenance fallback
        gurobi_version = f"unavailable: {exc}"
    import scipy

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "gurobi": gurobi_version,
        "cpu_count": os.cpu_count(),
    }


def write_manifest(qa_pass: bool) -> None:
    files = [
        path
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    external = {
        str(path.resolve()): sha256(path)
        for path in (
            STAGE0,
            STAGE0_QA,
            DESIGN,
            Path(core.__file__).resolve(),
            core.CONFIGURATIONS,
            Path(__file__).resolve(),
        )
    }
    manifest = {
        "version": VERSION,
        "qa_pass": qa_pass,
        "files": {
            str(path.relative_to(OUTPUT)): sha256(path)
            for path in files
        },
        "external_inputs": external,
        "manuscript_evidence_admitted": False,
        "evidence_status": EVIDENCE_STATUS,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2) + "\n"
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(
            f"Output directory must be absent or empty: {OUTPUT}"
        )
    for directory in ("logs", "patterns", "invariants"):
        (OUTPUT / directory).mkdir(parents=True, exist_ok=True)
    failure_path = OUTPUT / "failure_trace.jsonl"
    failure_path.write_text("")
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    exit_code = 0
    try:
        stage0_qa = validate_stage0()
        configure_core_numerics()
        static_test = core.perspective_envelope_static_test()
        if not static_test["all_cases_pass"]:
            raise RuntimeError("Perspective static equivalence failed")
        frame = load_country_year()
        orders, _ = core.make_day_orders(frame, COUNTRY, YEAR)
        config = core.load_configuration(CONFIG_ID)

        for multiplier in GRID_MULTIPLIERS:
            try:
                row = run_node(frame, orders, config, multiplier)
                rows.append(row)
                print(json.dumps(json_ready(row), ensure_ascii=False))
            except Exception as exc:
                exit_code = 2
                failure = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "country": COUNTRY,
                    "year": YEAR,
                    "config_id": CONFIG_ID,
                    "cell": CELL,
                    "xch_multiplier": multiplier,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "stage0_sha256": sha256(STAGE0),
                    "runner_sha256": sha256(Path(__file__)),
                }
                failures.append(failure)
                with failure_path.open("a") as stream:
                    stream.write(
                        json.dumps(json_ready(failure), ensure_ascii=False)
                        + "\n"
                    )
                rows.append(
                    {
                        "country": COUNTRY,
                        "year": YEAR,
                        "config_id": CONFIG_ID,
                        "cell": CELL,
                        "xch_multiplier": multiplier,
                        "node_pass": False,
                        "failure_recorded": True,
                        "failure_message": str(exc),
                        "manuscript_evidence_admitted": False,
                        "evidence_status": EVIDENCE_STATUS,
                    }
                )

        results = pd.DataFrame(rows)
        results.to_csv(OUTPUT / "fixed_xch_grid_results.csv", index=False)
        six_nodes_present = (
            len(results) == len(GRID_MULTIPLIERS)
            and sorted(results["xch_multiplier"].astype(float).tolist())
            == list(GRID_MULTIPLIERS)
        )
        all_nodes_pass = bool(
            six_nodes_present
            and results["node_pass"].fillna(False).astype(bool).all()
        )
        qa_pass = all_nodes_pass and not failures
        qa = {
            "status": "pass" if qa_pass else "fail",
            "qa_pass": qa_pass,
            "version": VERSION,
            "scope": "DE-2023 LFP4 DD complete six-node finite-grid numerical smoke",
            "grid_multipliers": list(GRID_MULTIPLIERS),
            "six_nodes_present": six_nodes_present,
            "all_nodes_pass": all_nodes_pass,
            "failure_count": len(failures),
            "stage0_version": stage0_qa.get("dataset_version"),
            "stage0_sha256": sha256(STAGE0),
            "stage0_qa_sha256": sha256(STAGE0_QA),
            "canonical_gas_field": CANONICAL_GAS_FIELD,
            "deprecated_alias_read": False,
            "relative_gap_limit": RELATIVE_GAP_LIMIT,
            "absolute_gap_eur2024_per_forced_kw_year_limit": (
                ABSOLUTE_GAP_EUR_PER_FORCED_KW_YEAR_LIMIT
            ),
            "mip_time_limit_seconds_per_node": MIP_TIME_LIMIT_SECONDS,
            "fixed_pattern_lp_time_limit_seconds_per_node": (
                LP_TIME_LIMIT_SECONDS
            ),
            "gap_gate_logic": "signed nonnegative AND relative AND absolute",
            "maximum_certified_relative_bound_width": (
                float(
                    results[
                        "certified_relative_bound_width"
                    ].dropna().max()
                )
                if "certified_relative_bound_width" in results
                and results["certified_relative_bound_width"].notna().any()
                else None
            ),
            "maximum_signed_bound_width_eur2024_per_forced_kw_year": (
                float(
                    results[
                        "signed_bound_width_eur2024_per_forced_kw_year"
                    ]
                    .dropna()
                    .max()
                )
                if (
                    "signed_bound_width_eur2024_per_forced_kw_year"
                    in results
                    and results[
                        "signed_bound_width_eur2024_per_forced_kw_year"
                    ]
                    .notna()
                    .any()
                )
                else None
            ),
            "pwl_certificate_scope": (
                "finite_six_node_solution_points_only_not_full_domain"
            ),
            "fixed_pattern_lp_interpretation": (
                "continuous reoptimization on one incumbent integer slice; "
                "feasible MILP upper bound, not full MILP optimum"
            ),
            "matrix_objective_invariants_required": True,
            "raw_and_rounded_patterns_saved": True,
            "failure_trace_path": str(failure_path.relative_to(OUTPUT)),
            "failure_trace_sha256": sha256(failure_path),
            "design_path": str(DESIGN.resolve()),
            "design_sha256": sha256(DESIGN),
            "core_runner_path": str(Path(core.__file__).resolve()),
            "core_runner_sha256": sha256(Path(core.__file__)),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__)),
            "configuration_path": str(core.CONFIGURATIONS.resolve()),
            "configuration_sha256": sha256(core.CONFIGURATIONS),
            "environment": environment_record(),
            "perspective_static_equivalence": static_test,
            "cost_scope": (
                "inherited Stage-1 structural objective retained solely for "
                "numerical certification; no Stage-2 cost choice made"
            ),
            "manuscript_evidence_admitted": False,
            "evidence_status": EVIDENCE_STATUS,
        }
        (OUTPUT / "QA.json").write_text(
            json.dumps(json_ready(qa), indent=2) + "\n"
        )
        write_manifest(qa_pass)
        if not qa_pass:
            exit_code = 2
    except Exception as exc:
        exit_code = 2
        failure = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "runner_setup_or_summary",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "stage0_sha256": sha256(STAGE0) if STAGE0.exists() else None,
            "runner_sha256": sha256(Path(__file__)),
        }
        with failure_path.open("a") as stream:
            stream.write(
                json.dumps(json_ready(failure), ensure_ascii=False) + "\n"
            )
        (OUTPUT / "FATAL_ERROR.json").write_text(
            json.dumps(json_ready(failure), indent=2) + "\n"
        )
        write_manifest(False)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
