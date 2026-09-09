#!/usr/bin/env python3
"""V0.2 fixed-xch grid with binary-pattern LP polishing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_observed_gas_event_task_stage1_v0_3 as core


GRID_MULTIPLIERS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
TIME_LIMIT_SECONDS = 60.0
POLISH_TIME_LIMIT_SECONDS = 30.0
COUNTRY = "DE"
YEAR = 2023
CONFIG_ID = "li_lfp_4h"
CELL = "DD"
FORMULATION = "perspective_bigm"

DESIGN = (
    core.ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE1_FIXED_XCH_GRID_POLISHED_DESIGN_V0_2.md"
)
OUTPUT = (
    core.ROOT
    / "artifacts"
    / "research"
    / "stage1_fixed_xch_grid_v0_2"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    return core.json_ready(value)


def run_node(
    frame: pd.DataFrame,
    orders: dict[str, list[int]],
    config: core.Configuration,
    multiplier: float,
    mip_log_path: Path,
    polish_log_path: Path,
) -> dict[str, Any]:
    task_indices = core.hourly_order(orders[CELL[0]])
    recharge_indices = core.hourly_order(orders[CELL[1]])
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
            0.0,
            config,
            energy,
            core.LFP_REPLACEABLE_FRACTION,
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

    os.environ["STAGE1_GUROBI_LOG"] = str(mip_log_path)
    mip_started = time.perf_counter()
    mip_result = core.solve_once_gurobi(model, TIME_LIMIT_SECONDS)
    mip_elapsed = time.perf_counter() - mip_started
    if not mip_result.incumbent_available:
        return {
            "xch_multiplier": multiplier,
            "fixed_extra_charge_power_mw": fixed_xch,
            "mip_solver_status": mip_result.raw_solver_status,
            "mip_solver_message": mip_result.message,
            "mip_solve_seconds": mip_elapsed,
            "incumbent_available": False,
            "node_pass": False,
            "mip_log": str(mip_log_path.relative_to(OUTPUT)),
        }

    nmode = len(model.indices.binary_hours)
    raw_mode = mip_result.x[
        model.indices.binary_start :
        model.indices.binary_start + nmode
    ]
    rounded_mode = np.rint(raw_mode)
    pre_polish_binary_distance = float(
        abs(raw_mode - rounded_mode).max(initial=0.0)
    )
    if pre_polish_binary_distance > 1e-9:
        return {
            "xch_multiplier": multiplier,
            "fixed_extra_charge_power_mw": fixed_xch,
            "mip_solver_status": mip_result.raw_solver_status,
            "mip_solver_message": mip_result.message,
            "mip_solve_seconds": mip_elapsed,
            "mip_objective_incumbent_cny_year": float(mip_result.fun),
            "mip_objective_best_bound_cny_year": float(
                mip_result.objective_bound
            ),
            "mip_gap": float(mip_result.mip_gap),
            "pre_polish_maximum_binary_distance": (
                pre_polish_binary_distance
            ),
            "incumbent_available": True,
            "polish_eligible": False,
            "node_pass": False,
            "mip_log": str(mip_log_path.relative_to(OUTPUT)),
        }

    polished_lower = np.array(model.bounds.lb, copy=True)
    polished_upper = np.array(model.bounds.ub, copy=True)
    binary_slice = slice(
        model.indices.binary_start,
        model.indices.binary_start + nmode,
    )
    polished_lower[binary_slice] = rounded_mode
    polished_upper[binary_slice] = rounded_mode
    model.bounds = core.Bounds(polished_lower, polished_upper)
    model.integrality = np.zeros_like(model.integrality)
    os.environ["STAGE1_GUROBI_LOG"] = str(polish_log_path)
    polish_started = time.perf_counter()
    polished_result = core.solve_once_gurobi(
        model,
        POLISH_TIME_LIMIT_SECONDS,
    )
    polish_elapsed = time.perf_counter() - polish_started
    if not polished_result.incumbent_available:
        return {
            "xch_multiplier": multiplier,
            "fixed_extra_charge_power_mw": fixed_xch,
            "mip_solver_status": mip_result.raw_solver_status,
            "mip_solver_message": mip_result.message,
            "mip_solve_seconds": mip_elapsed,
            "mip_objective_incumbent_cny_year": float(mip_result.fun),
            "mip_objective_best_bound_cny_year": float(
                mip_result.objective_bound
            ),
            "mip_gap": float(mip_result.mip_gap),
            "pre_polish_maximum_binary_distance": (
                pre_polish_binary_distance
            ),
            "polish_eligible": True,
            "polished_lp_status": polished_result.raw_solver_status,
            "polish_solve_seconds": polish_elapsed,
            "incumbent_available": True,
            "node_pass": False,
            "mip_log": str(mip_log_path.relative_to(OUTPUT)),
            "polish_log": str(polish_log_path.relative_to(OUTPUT)),
        }

    result = polished_result
    fields = core.fields(model, result.x)
    kg = float(result.x[model.indices.kg])
    xch = float(result.x[model.indices.xch])
    mode = result.x[binary_slice]
    product = result.x[
        model.indices.product_start :
        model.indices.product_start + nmode
    ]
    certified_absolute_gap_cny_year = max(
        0.0,
        float(result.fun) - float(mip_result.objective_bound),
    )
    certified_relative_gap = (
        certified_absolute_gap_cny_year / max(abs(float(result.fun)), 1.0)
    )
    certified_absolute_gap_per_forced_kw_year = (
        certified_absolute_gap_cny_year / (pdis * 1_000.0)
    )
    elapsed = mip_elapsed + polish_elapsed
    extra_upper = core.MAX_EXTRA_CHARGE_POWER_MULTIPLE * pdis
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
    rated_efc = float(fields["discharge"].sum() / model.energy_mwh)
    true_capital, _, replacement_life = core.lfp_capital_function(
        rated_efc,
        config,
        model.energy_mwh,
        core.LFP_REPLACEABLE_FRACTION,
    )
    epigraph_capital = float(result.x[model.indices.capital])
    pwl_gap = max(0.0, true_capital - epigraph_capital) / max(
        true_capital,
        1.0,
    )
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
                xch + extra_upper * mode - extra_upper - product,
                0.0,
            ).max(initial=0.0)
        ),
    )
    physical_gates = {
        "mip_solver_status_optimal": mip_result.raw_solver_status == 2,
        "pre_polish_binary_distance_pass": (
            pre_polish_binary_distance <= 1e-9
        ),
        "polished_lp_status_optimal": result.raw_solver_status == 2,
        "certified_gap_pass": (
            certified_relative_gap <= core.MIP_RELATIVE_GAP_TOLERANCE
            or certified_absolute_gap_per_forced_kw_year
            <= core.MIP_ABSOLUTE_GAP_CNY_PER_FORCED_KW_YEAR
        ),
        "maximum_simultaneous_pass": (
            float(simultaneous.max(initial=0.0))
            <= core.SIMULTANEOUS_HOURLY_TOLERANCE_MW
        ),
        "simultaneous_ratio_pass": (
            simultaneous_ratio <= core.SIMULTANEOUS_RATIO_TOLERANCE
        ),
        "binary_fractionality_pass": (
            float(np.minimum(abs(mode), abs(1.0 - mode)).max(initial=0.0))
            <= 1e-9
        ),
        "product_identity_pass": (
            float(abs(product - xch * mode).max(initial=0.0)) <= 1e-9
        ),
        "envelope_pass": envelope_violation <= 1e-9,
        "task_balance_pass": (
            float(abs(task_error).max(initial=0.0))
            <= core.ABSOLUTE_BALANCE_TOLERANCE
        ),
        "recharge_balance_pass": (
            float(abs(recharge_error).max(initial=0.0))
            <= core.ABSOLUTE_BALANCE_TOLERANCE
        ),
        "charge_source_pass": (
            float(abs(charge_source_error).max(initial=0.0))
            <= core.ABSOLUTE_BALANCE_TOLERANCE
        ),
        "soc_balance_pass": (
            float(abs(soc_error).max(initial=0.0))
            <= core.ABSOLUTE_BALANCE_TOLERANCE
        ),
        "cyclic_soc_pass": (
            float(abs(soc_error[0]))
            <= core.ABSOLUTE_BALANCE_TOLERANCE
        ),
        "lfp_outer_approximation_pass": (
            pwl_gap <= core.PWL_RELATIVE_GAP_TOLERANCE
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
        "fixed_extra_charge_power_mw": fixed_xch,
        "native_charge_power_mw": model.native_charge_power_mw,
        "total_charge_power_mw": model.native_charge_power_mw + xch,
        "gas_capacity_mw": kg,
        "mip_solver_status": mip_result.raw_solver_status,
        "mip_solver_message": mip_result.message,
        "mip_solve_seconds": mip_elapsed,
        "mip_objective_incumbent_cny_year": float(mip_result.fun),
        "mip_objective_best_bound_cny_year": float(
            mip_result.objective_bound
        ),
        "mip_reported_gap": float(mip_result.mip_gap),
        "mip_node_count": int(mip_result.mip_node_count),
        "pre_polish_maximum_binary_distance": (
            pre_polish_binary_distance
        ),
        "rounded_binary_pattern_sha256": hashlib.sha256(
            rounded_mode.astype(np.int8).tobytes()
        ).hexdigest(),
        "polished_lp_status": result.raw_solver_status,
        "polished_lp_message": result.message,
        "polish_solve_seconds": polish_elapsed,
        "total_solve_seconds": elapsed,
        "polished_lp_objective_cny_year": float(result.fun),
        "certified_global_best_bound_cny_year": float(
            mip_result.objective_bound
        ),
        "certified_relative_gap": certified_relative_gap,
        "certified_absolute_gap_cny_per_forced_kw_year": (
            certified_absolute_gap_per_forced_kw_year
        ),
        "polish_objective_change_cny_year": float(
            result.fun - mip_result.fun
        ),
        "rated_output_efc": rated_efc,
        "usable_band_efc": rated_efc / (
            core.SOC_MAX_FRAC - core.SOC_MIN_FRAC
        ),
        "lfp_replacement_life_years": replacement_life,
        "lfp_true_capital_cny_year": true_capital,
        "lfp_epigraph_capital_cny_year": epigraph_capital,
        "lfp_outer_approximation_relative_gap": pwl_gap,
        "binary_hours": nmode,
        "binary_charge_mode_hours": int((mode > 0.5).sum()),
        "binary_discharge_mode_hours": int((mode <= 0.5).sum()),
        "maximum_binary_fractionality": float(
            np.minimum(abs(mode), abs(1.0 - mode)).max(initial=0.0)
        ),
        "maximum_product_identity_error_mw": float(
            abs(product - xch * mode).max(initial=0.0)
        ),
        "maximum_mccormick_envelope_violation": envelope_violation,
        "maximum_simultaneous_charge_discharge_mw": float(
            simultaneous.max(initial=0.0)
        ),
        "simultaneous_energy_ratio": simultaneous_ratio,
        "maximum_task_balance_error_mw": float(
            abs(task_error).max(initial=0.0)
        ),
        "maximum_recharge_balance_error_mw": float(
            abs(recharge_error).max(initial=0.0)
        ),
        "maximum_charge_source_error_mw": float(
            abs(charge_source_error).max(initial=0.0)
        ),
        "maximum_soc_balance_error_mwh": float(
            abs(soc_error).max(initial=0.0)
        ),
        "cyclic_soc_difference_mwh": float(abs(soc_error[0])),
        "incumbent_available": True,
        **physical_gates,
        "node_pass": bool(all(physical_gates.values())),
        "mip_log": str(mip_log_path.relative_to(OUTPUT)),
        "polish_log": str(polish_log_path.relative_to(OUTPUT)),
        "manuscript_evidence_admitted": False,
        "evidence_status": core.EVIDENCE_STATUS,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(
            f"Output directory must be absent or empty: {OUTPUT}"
        )
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    core.check_stage0_qa(core.STAGE0_QA, core.STAGE0)
    static_test = core.perspective_envelope_static_test()
    if not static_test["all_cases_pass"]:
        raise RuntimeError("Perspective static equivalence failed")
    frame = core.load_country_year(core.STAGE0, COUNTRY, YEAR)
    orders, _ = core.make_day_orders(frame, COUNTRY, YEAR)
    config = core.load_configuration(CONFIG_ID)

    rows: list[dict[str, Any]] = []
    for multiplier in GRID_MULTIPLIERS:
        mip_log_path = (
            log_dir / f"xch_{multiplier:g}x_pdis_mip.log"
        )
        polish_log_path = (
            log_dir / f"xch_{multiplier:g}x_pdis_polished_lp.log"
        )
        row = run_node(
            frame,
            orders,
            config,
            multiplier,
            mip_log_path,
            polish_log_path,
        )
        rows.append(row)
        print(json.dumps(json_ready(row), ensure_ascii=False))

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "fixed_xch_grid_results.csv", index=False)
    all_nodes_present = len(results) == len(GRID_MULTIPLIERS)
    all_nodes_pass = bool(
        all_nodes_present and results["node_pass"].astype(bool).all()
    )
    qa = {
        "status": "pass" if all_nodes_pass else "fail",
        "qa_pass": all_nodes_pass,
        "scope": (
            "single-cell fixed-xch numerical timing and physical diagnostic"
        ),
        "country": COUNTRY,
        "year": YEAR,
        "config_id": CONFIG_ID,
        "cell": CELL,
        "grid_multipliers": list(GRID_MULTIPLIERS),
        "time_limit_seconds_per_node": TIME_LIMIT_SECONDS,
        "all_nodes_present": all_nodes_present,
        "all_nodes_pass": all_nodes_pass,
        "maximum_mip_solve_seconds": (
            float(results["mip_solve_seconds"].max())
            if not results.empty
            else None
        ),
        "maximum_polish_solve_seconds": (
            float(results["polish_solve_seconds"].max())
            if not results.empty and "polish_solve_seconds" in results
            else None
        ),
        "maximum_total_solve_seconds": (
            float(results["total_solve_seconds"].max())
            if not results.empty and "total_solve_seconds" in results
            else None
        ),
        "maximum_mip_reported_gap": (
            float(results["mip_reported_gap"].max())
            if not results.empty and "mip_reported_gap" in results
            else None
        ),
        "maximum_certified_relative_gap_after_polish": (
            float(results["certified_relative_gap"].max())
            if not results.empty
            and "certified_relative_gap" in results
            else None
        ),
        "maximum_pre_polish_binary_distance": (
            float(
                results[
                    "pre_polish_maximum_binary_distance"
                ].max()
            )
            if not results.empty
            and "pre_polish_maximum_binary_distance" in results
            else None
        ),
        "maximum_simultaneous_charge_discharge_mw": (
            float(
                results[
                    "maximum_simultaneous_charge_discharge_mw"
                ].max()
            )
            if not results.empty
            and "maximum_simultaneous_charge_discharge_mw" in results
            else None
        ),
        "maximum_hourly_balance_error": (
            float(
                results[
                    [
                        "maximum_task_balance_error_mw",
                        "maximum_recharge_balance_error_mw",
                        "maximum_charge_source_error_mw",
                        "maximum_soc_balance_error_mwh",
                    ]
                ]
                .max()
                .max()
            )
            if all_nodes_present
            else None
        ),
        "stage0_sha256": sha256(core.STAGE0),
        "design_path": str(DESIGN.resolve()),
        "design_sha256": sha256(DESIGN),
        "core_runner_path": str(Path(core.__file__).resolve()),
        "core_runner_sha256": sha256(Path(core.__file__)),
        "grid_runner_path": str(Path(__file__).resolve()),
        "grid_runner_sha256": sha256(Path(__file__)),
        "perspective_static_equivalence": static_test,
        "gas_charging_allowed": False,
        "manuscript_evidence_admitted": False,
        "evidence_status": core.EVIDENCE_STATUS,
        "interpretation": (
            "Passing nodes establish only tractability and polished "
            "physical admissibility conditional on fixed xch. The polished "
            "LP retains each rounded incumbent binary pattern and does not "
            "identify an optimal charge-interface capacity."
        ),
    }
    (OUTPUT / "QA.json").write_text(
        json.dumps(json_ready(qa), indent=2, ensure_ascii=False) + "\n"
    )
    manifest_files = [
        path
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "files": {
            str(path.relative_to(OUTPUT)): sha256(path)
            for path in manifest_files
        },
        "qa_pass": all_nodes_pass,
        "manuscript_evidence_admitted": False,
        "evidence_status": core.EVIDENCE_STATUS,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, ensure_ascii=False)
        + "\n"
    )
    print(json.dumps(json_ready(qa), indent=2, ensure_ascii=False))
    if not all_nodes_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
