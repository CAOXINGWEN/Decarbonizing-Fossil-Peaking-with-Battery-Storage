#!/usr/bin/env python3
"""Solve the natural-order 24 h A-CAES versus 4 h LFP comparison.

The v43 runner is imported as the single source of the dispatch equations.
Only the chronology wrapper is changed: source hours remain in their observed
order.  This supplies a bounded external-validity outcome for the country
fixed-effect analysis without changing or overwriting any v43 artefact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V44 = HERE.parent
RUNNER_PATH = V44 / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"
OUTPUT = HERE / "natural_caes24_s20_fixed"


def load_runner():
    spec = importlib.util.spec_from_file_location("v44_portfolio_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the audited portfolio runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    runner = load_runner()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(
        runner.DEFAULT_PANEL,
        usecols=[
            "country",
            "year",
            "source_hour_id",
            "source_day_id",
            "season_segment",
            "event_flag_q75",
            "raw_gas_mean_mw",
            "forced_discharge_power_mw",
            runner.BASE_TASK_COLUMN,
            runner.TAIL_TASK_COLUMN,
            runner.RECHARGE_COLUMN,
        ],
    ).sort_values(["country", "year", "source_hour_id"], kind="stable")
    country_years = [
        (str(row.country), int(row.year))
        for row in panel[["country", "year"]].drop_duplicates().itertuples(index=False)
    ]
    if len(country_years) != 132:
        raise RuntimeError(f"Expected 132 country-years, found {len(country_years)}")

    catalogue = pd.read_csv(runner.DEFAULT_CATALOGUE)
    base, candidates = runner.select_catalogue(catalogue, ["caes_24h"])
    candidate = candidates.iloc[0]
    scenario = runner.load_scenario(runner.DEFAULT_SCENARIO, 20)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, (country, year) in enumerate(country_years, start=1):
        frame = panel[panel.country.eq(country) & panel.year.eq(year)]
        if len(frame) != 8760 or not np.array_equal(
            frame.source_hour_id.to_numpy(), np.arange(8760)
        ):
            raise RuntimeError(f"Natural chronology is incomplete for {country}-{year}")
        base_task = frame[runner.BASE_TASK_COLUMN].to_numpy(float)
        tail_task = frame[runner.TAIL_TASK_COLUMN].to_numpy(float)
        recharge = frame[runner.RECHARGE_COLUMN].to_numpy(float)
        power = float(frame.forced_discharge_power_mw.iloc[0])
        arms = [
            ("base_only", base, 0.0, "base_only_lfp4", "none", 0.0),
            ("extra_lfp4_comparator", base, power, "li_lfp_4h", "li_lfp", 4.0),
            ("ldes_candidate", candidate, power, "caes_24h", "caes", 24.0),
        ]
        for arm_type, config, upper, config_id, tech_id, duration in arms:
            summary, _ = runner.solve_portfolio_arm(
                base_task,
                tail_task,
                recharge,
                power,
                upper,
                "fixed",
                base,
                config,
                scenario,
                env,
                1,
                1,
                False,
            )
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "chronology": "natural",
                    "arm_type": arm_type,
                    "config_id": config_id,
                    "tech_id": tech_id,
                    "duration_h": duration,
                    "scenario_id": 20,
                    **scenario,
                    **summary,
                }
            )
        if index % 10 == 0 or index == len(country_years):
            print(f"natural S20 solved {index}/{len(country_years)}", flush=True)
    env.dispose()

    result, comparison = runner.add_reference_effects(pd.DataFrame(rows))
    output_path = OUTPUT / "portfolio_arm_results.csv.gz"
    comparison_path = OUTPUT / "ldes_vs_extra_lfp4_results.csv.gz"
    runner.output_csv_gz(result, output_path)
    runner.output_csv_gz(comparison, comparison_path)
    qa = {
        "status": "pass",
        "country_year_count": len(country_years),
        "arm_row_count": len(result),
        "all_solver_statuses_optimal": bool(result.solver_status_name.eq("optimal").all()),
        "maximum_task_balance_residual_mw": float(result.maximum_task_balance_residual_mw.abs().max()),
        "maximum_shared_recharge_violation_mw": float(result.maximum_shared_recharge_violation_mw.clip(lower=0).max()),
        "maximum_cyclic_soc_residual_mwh": float(
            max(result.base_soc_cyclic_residual_mwh.abs().max(), result.tail_soc_cyclic_residual_mwh.abs().max())
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "output_sha256": {
            output_path.name: sha256(output_path),
            comparison_path.name: sha256(comparison_path),
        },
    }
    if not qa["all_solver_statuses_optimal"] or len(result) != 396:
        qa["status"] = "fail"
    (OUTPUT / "qa_summary.json").write_text(json.dumps(qa, indent=2) + "\n")
    (OUTPUT / "run_contract.json").write_text(
        json.dumps(
            {
                "runner": str(RUNNER_PATH.resolve()),
                "runner_sha256": sha256(RUNNER_PATH),
                "wrapper": str(Path(__file__).resolve()),
                "wrapper_sha256": sha256(Path(__file__).resolve()),
                "panel": str(runner.DEFAULT_PANEL.resolve()),
                "panel_sha256": sha256(runner.DEFAULT_PANEL),
                "catalogue": str(runner.DEFAULT_CATALOGUE.resolve()),
                "catalogue_sha256": sha256(runner.DEFAULT_CATALOGUE),
                "scenario_id": 20,
                "scenario": scenario,
                "chronology": "unaltered source-hour order",
                "tail_power_mode": "fixed",
                "candidate_config_ids": ["caes_24h"],
                "country_years": country_years,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("GRB_LICENSE_FILE", "/etc/gurobi/gurobi.lic")
    main()
