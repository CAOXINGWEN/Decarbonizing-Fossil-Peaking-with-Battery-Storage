#!/usr/bin/env python3
"""Executable entry point for the ERAA 2025 confirmatory-model skeleton.

Current modes
-------------
``smoke``
    Runs synchronized day-block conservation, a two-zone stochastic
    gas/hydro/import/storage LP, endogenous charging-price diagnostics, cyclic
    SOC, non-zero LFP degradation and the frozen valley decision rule on
    synthetic data.  It is an integrity test, never manuscript evidence.

``phase1-contract``
    Validates a standardized ERAA hourly table and constructs one paired
    chronology audit before an expensive phase-1 or network run is admitted.
    It does not silently approximate missing ERAA inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from eraa_confirmatory_model_core import (
    SolveSettings,
    StorageTechnology,
    apply_synchronized_day_order,
    chronology_conservation_audit,
    daily_scarcity_scores,
    evaluate_delta_w,
    evaluate_valley_for_chronology,
    event_statistics,
    make_day_order,
    solve_signed_value_variants,
    synthetic_hourly_panel,
    synthetic_transport_case,
    validate_standardized_hourly_contract,
    write_json,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_SPEC = (
    ROOT
    / "artifacts"
    / "research"
    / "ERAA_2025_CONFIRMATORY_MODEL_SPECIFICATION.md"
)
EXECUTION_CONTRACT = (
    ROOT
    / "artifacts"
    / "research"
    / "ERAA_2025_CONFIRMATORY_EXECUTION_CONTRACT.md"
)
RANDOM_SEED = 2026072601


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"output exists; choose a new directory or pass --force: {path}"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def storage_configurations() -> dict[str, StorageTechnology]:
    # These parameters are deliberately synthetic and exist only to test the
    # executable cost ledger.  A real run must load the audited 53-node
    # catalogue and its year-specific currency conversion.
    lfp_degradation = 100.0 / (6_000.0 * 0.90) * 1_000.0
    return {
        "LFP4": StorageTechnology(
            config_id="synthetic_LFP4",
            tech_id="li_lfp",
            duration_h=4.0,
            round_trip_efficiency=0.88,
            self_discharge_pct_day=0.05,
            soc_min_fraction=0.05,
            soc_max_fraction=0.95,
            reference_charge_to_discharge_power_ratio=1.0,
            maximum_charge_to_discharge_power_ratio=1.25,
            annual_discharge_power_cost_eur_kw=34.0,
            annual_charge_power_cost_eur_kw=24.0,
            annual_energy_cost_eur_kwh=18.0,
            variable_om_eur_mwh=1.5,
            degradation_cost_eur_mwh_discharged=lfp_degradation,
            main_evidence_domain=True,
            structural_extrapolation=False,
        ),
        "CAES24": StorageTechnology(
            config_id="synthetic_CAES24",
            tech_id="caes",
            duration_h=24.0,
            round_trip_efficiency=0.65,
            self_discharge_pct_day=0.15,
            soc_min_fraction=0.05,
            soc_max_fraction=0.95,
            reference_charge_to_discharge_power_ratio=1.0,
            maximum_charge_to_discharge_power_ratio=2.0,
            annual_discharge_power_cost_eur_kw=56.0,
            annual_charge_power_cost_eur_kw=42.0,
            annual_energy_cost_eur_kwh=6.5,
            variable_om_eur_mwh=3.0,
            degradation_cost_eur_mwh_discharged=0.0,
            main_evidence_domain=True,
            structural_extrapolation=False,
        ),
    }


def synthetic_valley_envelope(chronology: str) -> pd.DataFrame:
    durations = np.array([4, 8, 12, 24, 48, 72, 168, 240, 336], dtype=float)
    # Both curves have a connected 4-8 h turnover branch and a 12-24 h
    # negative interval.  The clustered synthetic curve moves stable
    # reliability-supported re-entry from 48 to 72 h.  These are unit-test
    # values, not empirical results.
    bt_lower = np.array([4.0, 2.0, -1.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    bj_upper = np.array([5.0, 2.0, -0.5, -0.3, 1.0, 2.0, 3.0, 4.0, 4.5])
    if chronology == "clustered":
        bj_upper[4] = -0.1
    bj_lower = np.array([2.0, 1.0, -1.2, -1.0, 0.5, 0.8, 1.0, 1.1, 1.2])
    ba_lower = np.array([-1.0, -0.4, -0.2, -0.1, 0.3, 0.6, 0.8, 0.9, 1.0])
    elcc_lower = np.array([-0.1, -0.1, 0.0, 0.0, 0.2, 0.5, 0.7, 0.8, 0.9])
    if chronology == "clustered":
        bj_lower[4] = -0.2
        ba_lower[4] = -0.1
        elcc_lower[4] = -0.1
    return pd.DataFrame(
        {
            "chronology": chronology,
            "duration_h": durations,
            "bt_lower": bt_lower,
            "bj_upper": bj_upper,
            "bj_lower": bj_lower,
            "ba_lower": ba_lower,
            "elcc_lower": elcc_lower,
            "main_evidence_domain": durations <= 168,
        }
    )


def run_smoke(output: Path) -> dict[str, Any]:
    original = synthetic_hourly_panel(days=14, seed=RANDOM_SEED)
    daily = daily_scarcity_scores(original)
    transformed: dict[str, pd.DataFrame] = {}
    chronology_audits: dict[str, dict[str, Any]] = {}
    event_rows: list[dict[str, Any]] = []
    for chronology in ("natural", "dispersed", "clustered"):
        order = make_day_order(
            daily,
            chronology,
            quantile=0.75,
            cluster_days=4,
            seed=RANDOM_SEED,
            randomized=False,
        )
        panel = apply_synchronized_day_order(original, order, chronology)
        transformed[chronology] = panel
        chronology_audits[chronology] = chronology_conservation_audit(
            original, panel
        )
        event_rows.append(
            {
                "chronology": chronology,
                **event_statistics(panel, quantile=0.75),
            }
        )

    settings = SolveSettings(candidate_total_discharge_power_mw=8.0)
    configs = storage_configurations()
    summary_rows: list[dict[str, Any]] = []
    first_stage_rows: list[pd.DataFrame] = []
    selected_dispatch: list[pd.DataFrame] = []
    selected_duals: list[pd.DataFrame] = []
    cases: dict[str, Any] = {}
    for chronology in ("dispersed", "clustered"):
        case = synthetic_transport_case(
            transformed[chronology],
            f"synthetic_{chronology}",
        )
        cases[chronology] = case
        for label, storage in configs.items():
            solutions = solve_signed_value_variants(
                case,
                storage,
                settings,
                keep_dispatch=True,
            )
            for variant, solution in solutions.items():
                summary_rows.append(
                    {
                        "chronology": chronology,
                        "storage_label": label,
                        **solution.summary,
                    }
                )
                first = solution.first_stage.copy()
                first.insert(0, "variant", variant)
                first.insert(0, "storage_label", label)
                first.insert(0, "chronology", chronology)
                first_stage_rows.append(first)
                if label == "CAES24" and variant == "joint":
                    dispatch = solution.dispatch.copy()
                    dispatch.insert(0, "chronology", chronology)
                    selected_dispatch.append(dispatch)
                    duals = solution.balance_duals.copy()
                    duals.insert(0, "chronology", chronology)
                    selected_duals.append(duals)

    summaries = pd.DataFrame(summary_rows)
    first_stage = pd.concat(first_stage_rows, ignore_index=True)
    events = pd.DataFrame(event_rows)
    summaries.to_csv(output / "synthetic_model_summary.csv", index=False)
    first_stage.to_csv(output / "synthetic_first_stage.csv", index=False)
    events.to_csv(output / "synthetic_event_statistics.csv", index=False)
    pd.concat(selected_dispatch, ignore_index=True).to_csv(
        output / "synthetic_caes24_joint_dispatch.csv.gz", index=False
    )
    pd.concat(selected_duals, ignore_index=True).to_csv(
        output / "synthetic_caes24_joint_balance_duals.csv.gz", index=False
    )

    dispersed_envelope = synthetic_valley_envelope("dispersed")
    clustered_envelope = synthetic_valley_envelope("clustered")
    envelope = pd.concat(
        [dispersed_envelope, clustered_envelope], ignore_index=True
    )
    envelope.to_csv(output / "synthetic_valley_envelope.csv", index=False)
    dispersed_boundary = evaluate_valley_for_chronology(dispersed_envelope)
    clustered_boundary = evaluate_valley_for_chronology(clustered_envelope)
    delta_w = evaluate_delta_w(
        dispersed_boundary,
        clustered_boundary,
        delta_w_ci_lower=0.05,
    )
    boundaries = {
        "dispersed": dispersed_boundary,
        "clustered": clustered_boundary,
        "paired": delta_w,
    }
    write_json(output / "synthetic_boundary_decision.json", boundaries)

    solver_residual_pass = bool(
        summaries["maximum_equality_residual"].le(1e-6).all()
        and summaries["maximum_inequality_residual"].le(1e-6).all()
        and summaries["maximum_soc_transition_residual_mwh"].le(1e-6).all()
        and summaries["maximum_hydro_stock_residual_mwh"].le(1e-6).all()
        and summaries["objective_recalculation_error_eur"].le(1e-5).all()
    )
    lfp = summaries.loc[summaries["tech_id"].eq("li_lfp")]
    qa = {
        "evidence_status": "synthetic_integrity_test_only",
        "chronology_audits": chronology_audits,
        "all_chronologies_conserved": all(
            audit["pass"] for audit in chronology_audits.values()
        ),
        "clustered_event_duration_exceeds_dispersed": bool(
            events.set_index("chronology").loc[
                "clustered", "mean_event_duration_d"
            ]
            > events.set_index("chronology").loc[
                "dispersed", "mean_event_duration_d"
            ]
        ),
        "solver_residual_pass": solver_residual_pass,
        "lfp_throughput_degradation_positive": bool(
            not lfp.empty and lfp["lfp_degradation_cost_eur_mwh"].gt(0).all()
        ),
        "endogenous_charging_duals_reported": bool(
            summaries[
                "charging_opportunity_cost_from_balance_duals_eur"
            ].notna().all()
        ),
        "simultaneous_charge_discharge_audit_pass": bool(
            summaries[
                "expected_simultaneous_charge_discharge_mwh"
            ].le(1e-6).all()
        ),
        "cyclic_soc_equations_in_solver": bool(
            summaries["maximum_soc_transition_residual_mwh"].le(1e-6).all()
        ),
        "gas_hydro_and_import_balance_in_solver": True,
        "outage_paths_fixed_by_calendar_across_pair": bool(
            np.array_equal(
                cases["dispersed"].thermal_available_fraction,
                cases["clustered"].thermal_available_fraction,
            )
        ),
        "network_state_fixed_by_calendar_across_pair": bool(
            np.array_equal(
                cases["dispersed"].network_capacity_mw,
                cases["clustered"].network_capacity_mw,
            )
        ),
        "synthetic_valley_rule_pass": bool(
            dispersed_boundary["confirmed"]
            and clustered_boundary["confirmed"]
            and delta_w["confirmed_widening"]
        ),
    }
    qa["pass"] = bool(
        qa["all_chronologies_conserved"]
        and qa["clustered_event_duration_exceeds_dispersed"]
        and qa["solver_residual_pass"]
        and qa["lfp_throughput_degradation_positive"]
        and qa["endogenous_charging_duals_reported"]
        and qa["simultaneous_charge_discharge_audit_pass"]
        and qa["cyclic_soc_equations_in_solver"]
        and qa["outage_paths_fixed_by_calendar_across_pair"]
        and qa["network_state_fixed_by_calendar_across_pair"]
        and qa["synthetic_valley_rule_pass"]
    )
    write_json(output / "QA.json", qa)
    return qa


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"unsupported standardized table format: {path}")


def run_phase1_contract(
    hourly_path: Path,
    output: Path,
    allow_subset: bool,
) -> dict[str, Any]:
    hourly = read_table(hourly_path)
    audit = validate_standardized_hourly_contract(hourly)
    counts = audit["weather_scenarios_per_target_year"]
    expected_years = {2028, 2030, 2033, 2035}
    observed_years = set(audit["target_years"])
    if not allow_subset:
        if observed_years != expected_years:
            raise ValueError(
                f"expected target years {sorted(expected_years)}, "
                f"observed {sorted(observed_years)}"
            )
        wrong_counts = {
            year: count
            for year, count in counts.items()
            if int(count) != 36
        }
        if wrong_counts:
            raise ValueError(
                f"each target year must have 36 weather scenarios: {wrong_counts}"
            )
    coverage = audit.pop("coverage")
    coverage.to_csv(output / "standardized_hourly_coverage.csv", index=False)

    # Audit one deterministic pair without pretending that this subset is a
    # model result.
    first_key = (
        hourly[["target_year", "weather_scenario"]]
        .drop_duplicates()
        .sort_values(["target_year", "weather_scenario"])
        .iloc[0]
    )
    subset = hourly.loc[
        hourly["target_year"].eq(first_key["target_year"])
        & hourly["weather_scenario"].eq(first_key["weather_scenario"])
    ].copy()
    daily = daily_scarcity_scores(subset)
    pair_audits: dict[str, Any] = {}
    event_rows: list[dict[str, Any]] = []
    for chronology in ("dispersed", "clustered"):
        order = make_day_order(
            daily,
            chronology,
            quantile=0.75,
            cluster_days=4,
            seed=RANDOM_SEED,
        )
        transformed = apply_synchronized_day_order(subset, order, chronology)
        pair_audits[chronology] = chronology_conservation_audit(
            subset, transformed
        )
        event_rows.append(
            {"chronology": chronology, **event_statistics(transformed)}
        )
    pd.DataFrame(event_rows).to_csv(
        output / "paired_chronology_event_audit.csv", index=False
    )
    result = {
        **audit,
        "phase": "phase1_input_contract_only",
        "allow_subset": allow_subset,
        "paired_chronology_audits": pair_audits,
        "pass": bool(
            audit["all_hour_axes_synchronized"]
            and all(value["pass"] for value in pair_audits.values())
        ),
    }
    write_json(output / "QA.json", result)
    return result


def manifest(mode: str, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_contract": "eraa_2025_confirmatory_v0_1",
        "mode": mode,
        "evidence_status": (
            "synthetic_integrity_test_only"
            if mode == "smoke"
            else "input_contract_only"
        ),
        "output": str(output),
        "random_seed": RANDOM_SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "model_specification": str(MODEL_SPEC),
        "model_specification_sha256": sha256(MODEL_SPEC),
        "execution_contract": str(EXECUTION_CONTRACT),
        "execution_contract_sha256": sha256(EXECUTION_CONTRACT),
        "core_script_sha256": sha256(HERE / "eraa_confirmatory_model_core.py"),
        "runner_script_sha256": sha256(Path(__file__).resolve()),
        "arguments": vars(args),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("smoke", "phase1-contract"),
        default="smoke",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hourly-standardized", type=Path)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow a target-year/weather subset for development only.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prepare_output(args.output, args.force)
    try:
        if args.mode == "smoke":
            qa = run_smoke(args.output)
        else:
            if args.hourly_standardized is None:
                raise ValueError(
                    "--hourly-standardized is required for phase1-contract"
                )
            qa = run_phase1_contract(
                args.hourly_standardized,
                args.output,
                args.allow_subset,
            )
        write_json(args.output / "run_manifest.json", manifest(args.mode, args.output, args))
    except Exception:
        # Leave a clearly incomplete directory; never make it look released.
        failure = {
            "pass": False,
            "mode": args.mode,
            "error": f"{type(sys.exc_info()[1]).__name__}: {sys.exc_info()[1]}",
        }
        write_json(args.output / "FAILED.json", failure)
        raise
    if not qa.get("pass", False):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
