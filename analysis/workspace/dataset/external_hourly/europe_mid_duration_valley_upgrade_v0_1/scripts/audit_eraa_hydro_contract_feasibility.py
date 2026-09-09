#!/usr/bin/env python3
"""Diagnose ERAA hydro-contract slack independently of the power system.

The full-network runner uses soft hydro inequalities to prevent a released
hydro inconsistency from making the entire dispatch model infeasible.  This
audit solves each hydro resource in isolation and reports the exact minimum
slack by constraint family.  It also compares the runner's historical
first/last-trajectory fallback with the official annual start/end levels
released in ``Other Data/Reservoir Levels.csv``.

The audit is diagnostic only.  It does not change the dispatch model and its
outputs are never admitted as manuscript evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("run_eraa_2035_full_network_gurobi.py")
DEFAULT_LEVELS = (
    ROOT
    / "raw"
    / "eraa_2025"
    / "extracted"
    / "Other Data"
    / "Reservoir Levels.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "research"
    / "eraa_hydro_contract_feasibility_by_resource.csv"
)
TECHNOLOGY_TO_CODE = {
    "Reservoir": "HRI",
    "Pondage": "HPI",
    "Open loop pumping": "HOL",
    "Closed loop pumping": "HCL",
}
SLACK_FAMILIES = (
    "minimum_power",
    "minimum_pumping_power",
    "maximum_generation_energy",
    "minimum_generation_energy",
    "maximum_reservoir_level",
    "minimum_reservoir_level",
)


def load_runner():
    spec = importlib.util.spec_from_file_location("eraa_network_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runner_arguments(runner, weather_scenario: str, chronology: str):
    saved = sys.argv
    try:
        sys.argv = [
            str(RUNNER),
            "--config-id",
            "caes_24h",
            "--weather-scenario",
            weather_scenario,
            "--chronology",
            chronology,
        ]
        return runner.parse_args()
    finally:
        sys.argv = saved


def official_endpoint_lookup(
    path: Path, weather_scenario: str
) -> dict[tuple[str, str], tuple[float, float]]:
    frame = pd.read_csv(path)
    required = {
        "MARKET_NODE",
        "TARGET_YEAR",
        "WEATHER_SCENARIO",
        "RESERVOIR_LEVEL_START",
        "RESERVOIR_LEVEL_END",
        "TECHNOLOGY",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    frame = frame.loc[frame["TECHNOLOGY"].isin(TECHNOLOGY_TO_CODE)].copy()
    frame["hydro_code"] = frame["TECHNOLOGY"].map(TECHNOLOGY_TO_CODE)
    frame["target_rank"] = np.select(
        [
            frame["TARGET_YEAR"].astype(str).eq("ERAA_2035"),
            frame["TARGET_YEAR"].astype(str).eq("All_years_ERAA"),
        ],
        [0, 1],
        default=99,
    )
    exact_weather = f"ERAA-{weather_scenario}"
    frame["weather_rank"] = np.select(
        [
            frame["WEATHER_SCENARIO"].astype(str).eq(exact_weather),
            frame["WEATHER_SCENARIO"]
            .astype(str)
            .eq("All (other) weather scenarios"),
        ],
        [0, 1],
        default=99,
    )
    frame = frame.loc[
        frame["target_rank"].lt(99) & frame["weather_rank"].lt(99)
    ].copy()
    frame = frame.sort_values(
        ["MARKET_NODE", "hydro_code", "target_rank", "weather_rank"]
    )
    frame = frame.drop_duplicates(["MARKET_NODE", "hydro_code"], keep="first")
    for column in ("RESERVOIR_LEVEL_START", "RESERVOIR_LEVEL_END"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].between(0, 1).all():
            raise ValueError(f"{path}: {column} lies outside [0,1]")
    return {
        (str(row.MARKET_NODE), str(row.hydro_code)): (
            float(row.RESERVOIR_LEVEL_START),
            float(row.RESERVOIR_LEVEL_END),
        )
        for row in frame.itertuples(index=False)
    }


def classify_slack(name: str) -> str:
    if "_minimum_power_" in name:
        return "minimum_power"
    if "_minimum_pumping_power_" in name:
        return "minimum_pumping_power"
    if "_max_energy_" in name:
        return "maximum_generation_energy"
    if "_min_energy_" in name:
        return "minimum_generation_energy"
    if "_max_level_slack" in name:
        return "maximum_reservoir_level"
    if "_min_level_slack" in name:
        return "minimum_reservoir_level"
    raise ValueError(f"unclassified hydro slack variable: {name}")


def solve_one(
    runner,
    resource_type: str,
    row: Any,
    start_fraction: float,
    end_fraction: float,
    threads: int,
) -> dict[str, Any]:
    import gurobipy as gp
    from gurobipy import GRB

    hours = len(np.asarray(row.inflow_mwh, dtype=float))
    energy = float(row.energy_capacity_mwh)
    model = gp.Model(
        f"hydro_audit_{resource_type}_{row.zone}_{row.hydro_code}"
    )
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.Method = 1
    model.Params.Presolve = 2
    model.Params.FeasibilityTol = 1e-8
    model.Params.OptimalityTol = 1e-8
    generation = model.addMVar(
        hours,
        lb=0.0,
        ub=np.asarray(row.discharge_ub_mw, dtype=float),
        name="generation",
    )
    soc = model.addMVar(
        hours,
        lb=0.0,
        ub=energy,
        name="soc",
    )
    spill = model.addMVar(hours, lb=0.0, name="spill")
    inflow = np.asarray(row.inflow_mwh, dtype=float)
    if resource_type == "reservoir":
        charge = None
        model.addConstr(
            soc[0] + generation[0] + spill[0]
            == start_fraction * energy + inflow[0]
        )
        model.addConstr(
            soc[1:] - soc[:-1] + generation[1:] + spill[1:] == inflow[1:]
        )
        charge_lb = None
    elif resource_type == "pumped_hydro":
        charge = model.addMVar(
            hours,
            lb=0.0,
            ub=np.asarray(row.charge_ub_mw, dtype=float),
            name="charge",
        )
        model.addConstr(
            soc[0]
            - float(row.charge_efficiency) * charge[0]
            + generation[0] / float(row.discharge_efficiency)
            + spill[0]
            == start_fraction * energy + inflow[0]
        )
        model.addConstr(
            soc[1:]
            - soc[:-1]
            - float(row.charge_efficiency) * charge[1:]
            + generation[1:] / float(row.discharge_efficiency)
            + spill[1:]
            == inflow[1:]
        )
        charge_lb = np.asarray(row.charge_lb_mw, dtype=float)
    else:
        raise ValueError(resource_type)
    model.addConstr(soc[-1] == end_fraction * energy)
    slacks, counts = runner.add_hydro_contract_constraints(
        model=model,
        generation=generation,
        discharge_lb_mw=np.asarray(row.discharge_lb_mw, dtype=float),
        period_constraints=row.period_constraints,
        penalty_eur_mwh=1.0,
        name="contract",
        soc=soc,
        energy_capacity_mwh=energy,
        charge=charge,
        charge_lb_mw=charge_lb,
        soft_nonlevel_source_discrepancy=True,
    )
    model.ModelSense = GRB.MINIMIZE
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return {
            "gurobi_status": int(model.Status),
            "minimum_total_slack_mwh": np.nan,
            **{f"{family}_slack_mwh": np.nan for family in SLACK_FAMILIES},
            **{f"{family}_constraint_count": counts[family]
               for family in SLACK_FAMILIES},
        }
    family_values = {family: 0.0 for family in SLACK_FAMILIES}
    for item in (
        slacks["source_contract_discrepancy"]
        + slacks["level_trajectory"]
    ):
        values = np.asarray(item.X, dtype=float)
        if values.ndim == 0:
            variables = [item]
            values = values.reshape(1)
        else:
            variables = item.tolist()
        for variable, value in zip(variables, values):
            family_values[classify_slack(variable.VarName)] += float(value)
    total = float(sum(family_values.values()))
    return {
        "gurobi_status": int(model.Status),
        "minimum_total_slack_mwh": total,
        **{
            f"{family}_slack_mwh": float(family_values[family])
            for family in SLACK_FAMILIES
        },
        **{
            f"{family}_constraint_count": int(counts[family])
            for family in SLACK_FAMILIES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-scenario", default="WS01")
    parser.add_argument(
        "--chronology", choices=("dispersed", "clustered"), default="dispersed"
    )
    parser.add_argument("--reservoir-level-file", type=Path, default=DEFAULT_LEVELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    runner = load_runner()
    model_args = runner_arguments(
        runner, args.weather_scenario, args.chronology
    )
    system, _ = runner.load_system(model_args)
    lookup = official_endpoint_lookup(
        args.reservoir_level_file.resolve(), args.weather_scenario
    )
    records: list[dict[str, Any]] = []
    for resource_type, frame in (
        ("reservoir", system.reservoir),
        ("pumped_hydro", system.pumped_hydro),
    ):
        for row in frame.itertuples(index=False):
            endpoint_cases = {
                "model_endpoints": (
                    float(row.initial_soc_fraction),
                    float(row.terminal_soc_fraction),
                )
            }
            official = lookup.get((str(row.zone), str(row.hydro_code)))
            if official is not None and not np.allclose(
                official,
                endpoint_cases["model_endpoints"],
                rtol=0,
                atol=1e-12,
            ):
                endpoint_cases["released_file_countercheck"] = official
            for endpoint_case, (start, end) in endpoint_cases.items():
                result = solve_one(
                    runner,
                    resource_type,
                    row,
                    start,
                    end,
                    args.threads,
                )
                records.append(
                    {
                        "weather_scenario": args.weather_scenario,
                        "chronology": args.chronology,
                        "resource_type": resource_type,
                        "zone": str(row.zone),
                        "hydro_code": str(row.hydro_code),
                        "endpoint_case": endpoint_case,
                        "initial_soc_fraction": float(start),
                        "terminal_soc_fraction": float(end),
                        "model_annual_endpoint_source": str(
                            row.annual_endpoint_source
                        ),
                        "energy_capacity_mwh": float(row.energy_capacity_mwh),
                        **result,
                    }
                )
    output = pd.DataFrame(records).sort_values(
        ["resource_type", "zone", "hydro_code", "endpoint_case"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.10g")
    summary = {
        "diagnostic_only": True,
        "weather_scenario": args.weather_scenario,
        "chronology": args.chronology,
        "resource_case_rows": int(len(output)),
        "resources": int(
            output[["resource_type", "zone", "hydro_code"]]
            .drop_duplicates()
            .shape[0]
        ),
        "resources_with_official_annual_levels": int(
            output.loc[
                output["model_annual_endpoint_source"].eq(
                    "official_annual_levels"
                ),
                ["resource_type", "zone", "hydro_code"],
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "minimum_slack_by_endpoint_case_mwh": {
            str(key): float(value)
            for key, value in output.groupby("endpoint_case")[
                "minimum_total_slack_mwh"
            ].sum().items()
        },
        "slack_by_family_and_endpoint_case_mwh": {
            str(case): {
                family: float(
                    group[f"{family}_slack_mwh"].sum()
                )
                for family in SLACK_FAMILIES
            }
            for case, group in output.groupby("endpoint_case")
        },
        "all_resource_cases_optimal": bool(output["gurobi_status"].eq(2).all()),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
