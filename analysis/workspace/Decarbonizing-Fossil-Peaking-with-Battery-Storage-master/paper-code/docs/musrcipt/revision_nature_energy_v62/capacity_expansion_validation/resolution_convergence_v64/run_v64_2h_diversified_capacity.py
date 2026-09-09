#!/usr/bin/env python3
"""Run the v64 2 h convergence check for diversified storage procurement.

The script imports the frozen v62 capacity-expansion model and changes only
the planning-block length.  It solves the diversified technology menu under
the expected-cost and mean-plus-CVaR90 objectives for every requested country.
Single-technology comparators are not part of the resolution-convergence
question and are therefore not re-solved here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE_RUNNER_PATH = HERE.parent / "run_v62_multiyear_capacity_expansion.py"
RISK_WEIGHTS = (0.0, 0.5)
ACTIVE_TOLERANCE = 1.0e-6


def load_base_runner():
    specification = importlib.util.spec_from_file_location(
        "v64_resolution_convergence_base", BASE_RUNNER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot import the frozen capacity-expansion model")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def active_set(group: pd.DataFrame) -> str:
    active = group.loc[
        group["capacity_per_normalized_mw"].gt(ACTIVE_TOLERANCE), "config_id"
    ].astype(str)
    return "+".join(sorted(active)) if len(active) else "none"


def dominant_technology(group: pd.DataFrame) -> str:
    row = group.sort_values(
        ["capacity_per_normalized_mw", "config_id"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    if float(row["capacity_per_normalized_mw"]) <= ACTIVE_TOLERANCE:
        return "none"
    return str(row["config_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--countries", required=True)
    parser.add_argument("--planning-step-hours", type=int, default=2)
    parser.add_argument("--cost-year", type=int, default=2035)
    parser.add_argument("--value-scenario", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.planning_step_hours <= 0 or 8760 % args.planning_step_hours:
        raise ValueError("Planning step must be a positive divisor of 8,760")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise RuntimeError("Output directory is non-empty; pass --force to replace")
    output.mkdir(parents=True, exist_ok=True)

    base = load_base_runner()
    chronology_runner = base.load_registered_runner()
    usecols = [
        "country",
        "year",
        "timestamp_utc",
        "source_day_id",
        "season_segment",
        "event_flag_q75",
        "task_c3_q75_mw",
        "raw_gas_mean_mw",
        base.TASK_COLUMN,
        base.RECHARGE_COLUMN,
    ]
    panel = pd.read_csv(base.PANEL_PATH, usecols=usecols)
    available = sorted(panel["country"].astype(str).unique())
    countries = parse_list(args.countries)
    missing = sorted(set(countries) - set(available))
    if missing:
        raise RuntimeError(f"Countries absent from the panel: {missing}")
    technologies = base.load_technologies(args.cost_year)
    value = base.load_value_scenario(args.value_scenario)

    summary_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    conservation_rows: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    for country in countries:
        country_frame = panel.loc[panel["country"].astype(str).eq(country)].copy()
        scenarios, conservation = base.build_scenarios(
            country_frame, chronology_runner, args.planning_step_hours
        )
        conservation_rows.extend({"country": country, **row} for row in conservation)
        for risk_weight in RISK_WEIGHTS:
            summary, capacities, scenario_metrics = base.solve_country_design_highs(
                country=country,
                scenarios=scenarios,
                technologies=technologies,
                value=value,
                risk_weight=risk_weight,
                mode="diversified",
                step_hours=args.planning_step_hours,
                threads=1,
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
    conservation = pd.DataFrame(conservation_rows)
    summaries.to_csv(output / "design_summary.csv", index=False)
    capacities.to_csv(output / "capacity_mix.csv", index=False)
    scenario_metrics.to_csv(output / "scenario_metrics.csv.gz", index=False)
    conservation.to_csv(output / "chronology_conservation.csv", index=False)

    shift_rows: list[dict[str, Any]] = []
    for country, group in capacities.groupby("country", sort=True):
        expected = group.loc[group["risk_weight"].eq(0.0)].copy()
        risk = group.loc[group["risk_weight"].eq(0.5)].copy()
        expected_total = float(expected["capacity_per_normalized_mw"].sum())
        risk_total = float(risk["capacity_per_normalized_mw"].sum())
        shift_rows.append(
            {
                "country": country,
                "expected_active_set": active_set(expected),
                "risk_aware_active_set": active_set(risk),
                "expected_dominant_technology": dominant_technology(expected),
                "risk_aware_dominant_technology": dominant_technology(risk),
                "expected_total_power_percent_mean_gas": expected_total * 100.0,
                "risk_aware_total_power_percent_mean_gas": risk_total * 100.0,
                "risk_aware_minus_expected_power_percentage_points":
                    (risk_total - expected_total) * 100.0,
            }
        )
    shifts = pd.DataFrame(shift_rows)
    shifts.to_csv(output / "procurement_design_shifts.csv", index=False)

    energy_ranges = conservation.groupby(["country", "year"])[
        ["task_energy_normalized_mwh", "recharge_energy_normalized_mwh"]
    ].agg(lambda values: float(values.max() - values.min()))
    qa = {
        "status": "pass",
        "evidence_status": "v64_two_hour_diversified_resolution_convergence",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "countries": countries,
        "country_count": len(countries),
        "planning_step_hours": args.planning_step_hours,
        "risk_weights": list(RISK_WEIGHTS),
        "modes": ["diversified"],
        "all_solver_statuses_optimal": bool(summaries["solver_status"].eq("optimal").all()),
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
        "maximum_chronology_task_energy_residual": float(
            energy_ranges["task_energy_normalized_mwh"].max()
        ),
        "maximum_chronology_recharge_energy_residual": float(
            energy_ranges["recharge_energy_normalized_mwh"].max()
        ),
        "input_sha256": {
            "panel": sha256(base.PANEL_PATH),
            "registered_chronology_runner": sha256(base.REGISTERED_RUNNER_PATH),
            "cost_catalogue": sha256(base.COST_CATALOGUE_PATH),
            "value_scenarios": sha256(base.VALUE_SCENARIO_PATH),
            "frozen_model": sha256(BASE_RUNNER_PATH),
        },
    }
    qa["status"] = (
        "pass"
        if qa["all_solver_statuses_optimal"]
        and qa["maximum_objective_identity_error_eur_per_normalized_mw"] <= 0.02
        and qa["maximum_task_balance_residual"] <= 1.0e-6
        and qa["maximum_shared_recharge_violation"] <= 1.0e-6
        and qa["maximum_state_balance_residual_mwh"] <= 1.0e-6
        and qa["minimum_capacity_upper_bound_slack"] >= -1.0e-8
        and qa["maximum_chronology_task_energy_residual"] <= 1.0e-10
        and qa["maximum_chronology_recharge_energy_residual"] <= 1.0e-10
        else "fail"
    )
    output_names = (
        "design_summary.csv",
        "capacity_mix.csv",
        "scenario_metrics.csv.gz",
        "chronology_conservation.csv",
        "procurement_design_shifts.csv",
    )
    qa["output_sha256"] = {
        name: sha256(output / name) for name in output_names
    }
    write_json(output / "qa.json", qa)
    (output / "design_summary_checkpoint.csv").unlink(missing_ok=True)
    print(json.dumps(json_ready(qa), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
