#!/usr/bin/env python3
"""Finish the pathological ES-2017 shard with the equivalent HiGHS IPM path.

The registered LP, objective, constraints and one-cent lexicographic cut are
unchanged.  Only the HiGHS continuous-LP algorithm is switched from dual
simplex to IPM for ES-2017 after the dual-simplex shard remained inside one
country-year for more than fifteen minutes.  A separate IPM-to-archived-Gurobi
smoke comparison is required before the rescued rows are accepted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog as scipy_linprog

import run_bounded_s20_highs as bounded


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "bounded_s20_highs" / "cost_2030" / "shard30_of_66"
RESCUE_OUTPUT = (
    HERE / "bounded_s20_highs" / "cost_2030"
    / "rescue_es2017_ipm_temporary"
)
SMOKE_OUTPUT = HERE / "bounded_s20_highs" / "smoke" / "ipm_rescue"
METRICS = (
    "tail_power_mw", "total_charge_mwh", "total_discharge_mwh",
    "unserved_eens_mwh", "portfolio_efc_mw", "residual_peak_mw",
    "system_cost_eur",
)


class IpmDispatcher:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        kwargs["method"] = "highs-ipm"
        self.calls += 1
        started = time.perf_counter()
        result = scipy_linprog(*args, **kwargs)
        if self.calls % 10 == 0:
            print(
                f"HiGHS-IPM calls completed: {self.calls}; "
                f"latest={time.perf_counter() - started:.2f}s; status={result.status}",
                flush=True,
            )
        return result


def ipm_smoke(dispatcher: IpmDispatcher) -> dict:
    runner, panel, base_row, candidates, scenario = bounded.prepare_inputs(2035, 31)
    frame = panel[panel["country"].eq("DE") & panel["year"].eq(2023)].copy()
    selected = candidates[candidates["config_id"].eq("caes_24h")]
    rows, _ = bounded.solve_country_year(
        runner, frame, base_row, selected, scenario, 31, 2035
    )
    result = pd.DataFrame(rows)
    reference = pd.read_csv(bounded.GUR_S31)
    reference = reference[
        reference["country"].eq("DE") & reference["year"].eq(2023)
        & reference["config_id"].isin(["li_lfp_4h", "caes_24h"])
    ]
    checks = []
    for row in result.itertuples(index=False):
        ref = reference[
            reference["chronology"].eq(row.chronology)
            & reference["config_id"].eq(row.config_id)
        ].iloc[0]
        for metric in METRICS:
            actual = float(getattr(row, metric))
            expected = float(ref[metric])
            absolute = abs(actual - expected)
            checks.append({
                "chronology": row.chronology,
                "config_id": row.config_id,
                "metric": metric,
                "highs_ipm": actual,
                "archived_gurobi": expected,
                "absolute_difference": absolute,
                "relative_difference": absolute / max(abs(expected), 1.0),
            })
    check = pd.DataFrame(checks)
    SMOKE_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = SMOKE_OUTPUT / "highs_ipm_vs_archived_gurobi_smoke.csv"
    check.to_csv(path, index=False)
    qa = {
        "status": "pass" if float(check["relative_difference"].max()) < 1.0e-5 else "fail",
        "maximum_relative_difference": float(check["relative_difference"].max()),
        "maximum_absolute_difference": float(check["absolute_difference"].max()),
        "comparison_rows": int(len(check)),
        "role": "algorithm_equivalence_smoke_for_ES_2017_rescue_only",
    }
    bounded.write_json(SMOKE_OUTPUT / "smoke_qa.json", qa)
    if qa["status"] != "pass":
        raise RuntimeError("HiGHS-IPM rescue smoke comparison failed")
    print(json.dumps(qa, indent=2), flush=True)
    return qa


def chronology_rows(runner, frame: pd.DataFrame) -> list[dict]:
    orders, order_stats = runner.day_orders(frame)
    original_task = frame[runner.BASE_TASK_COLUMN].to_numpy(float)
    original_tail = frame[runner.TAIL_TASK_COLUMN].to_numpy(float)
    original_recharge = frame[runner.RECHARGE_COLUMN].to_numpy(float)
    rows = []
    for chronology in ("dispersed", "clustered"):
        order = orders[chronology]
        task = runner.reorder_days(original_task, order)
        tail = runner.reorder_days(original_tail, order)
        recharge = runner.reorder_days(original_recharge, order)
        rows.append({
            "country": str(frame["country"].iloc[0]),
            "year": int(frame["year"].iloc[0]),
            "chronology": chronology,
            **order_stats,
            "base_task_conservation_error_mwh": float(task.sum() - original_task.sum()),
            "tail_task_conservation_error_mwh": float(tail.sum() - original_tail.sum()),
            "recharge_conservation_error_mwh": float(recharge.sum() - original_recharge.sum()),
        })
    return rows


def main() -> None:
    unfinished = OUTPUT / "portfolio_arm_results_unfinished.csv.gz"
    if not unfinished.exists():
        raise RuntimeError("The completed ES-2016 checkpoint is missing")
    es2016 = pd.read_csv(unfinished)
    if set(zip(es2016["country"], es2016["year"])) != {("ES", 2016)} or len(es2016) != 100:
        raise RuntimeError("The checkpoint is not the expected 100-row ES-2016 block")

    dispatcher = IpmDispatcher()
    bounded.linprog = dispatcher
    smoke_qa = ipm_smoke(dispatcher)
    RESCUE_OUTPUT.mkdir(parents=True, exist_ok=True)

    runner, panel, base_row, candidates, scenario = bounded.prepare_inputs(2030, 20)
    es2017_frame = panel[
        panel["country"].eq("ES") & panel["year"].eq(2017)
    ].copy()
    started = time.perf_counter()
    new_rows, es2017_chronology = bounded.solve_country_year(
        runner, es2017_frame, base_row, candidates, scenario, 20, 2030
    )
    es2017 = pd.DataFrame(new_rows)
    es2017["solver_method"] = np.where(
        es2017["secondary_tiebreak_success"],
        "scipy_highs_ipm_two_stage_rescue",
        "scipy_highs_ipm_primary_optimum_secondary_numerical_fallback_rescue",
    )
    rescue_result_path = RESCUE_OUTPUT / "ES_2017_arm_results.csv.gz"
    es2017.to_csv(
        rescue_result_path, index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(es2017_chronology).to_csv(
        RESCUE_OUTPUT / "ES_2017_chronology_conservation.csv", index=False
    )
    rescue_contract = {
        "status": "completed_pending_explicit_merge",
        "country": "ES", "year": 2017, "cost_year": 2030,
        "scenario_id": 20, "candidate_count": 49,
        "solver": "scipy_highs_ipm_two_stage_same_LP_and_lexicographic_cut",
        "source_checkpoint": str(unfinished.resolve()),
        "source_checkpoint_role": "completed_ES_2016_dual_simplex_block",
        "rescue_script": str(Path(__file__).resolve()),
        "rescue_script_sha256": bounded.sha256(Path(__file__).resolve()),
        "ipm_smoke_qa": smoke_qa,
        "row_count": int(len(es2017)),
        "output_sha256": bounded.sha256(rescue_result_path),
    }
    bounded.write_json(RESCUE_OUTPUT / "rescue_contract.json", rescue_contract)
    results = pd.concat([es2016, es2017], ignore_index=True)
    if len(results) != 200 or results.duplicated(
        ["country", "year", "chronology", "arm_type", "config_id"]
    ).any():
        raise RuntimeError("Rescued shard does not contain 200 unique arm rows")

    es2016_frame = panel[
        panel["country"].eq("ES") & panel["year"].eq(2016)
    ].copy()
    conservation = pd.DataFrame(
        chronology_rows(runner, es2016_frame) + es2017_chronology
    )
    output_path = OUTPUT / "portfolio_arm_results.csv.gz"
    results.to_csv(
        output_path, index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    conservation.to_csv(OUTPUT / "chronology_conservation.csv", index=False)

    contract_path = OUTPUT / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["model"] = (
        "registered_v42_continuous_portfolio_lp__ES2016_highs_ds__"
        "ES2017_equivalent_highs_ipm_rescue"
    )
    contract["solver_assignments"] = {
        "ES-2016": "scipy_highs_dual_simplex_two_stage",
        "ES-2017": "scipy_highs_ipm_two_stage_same_lp_and_lexicographic_cut",
    }
    contract["rescue_reason"] = (
        "dual_simplex_ES_2017_country_year_exceeded_15_minutes_without_"
        "country_year_checkpoint; algorithm changed, model contract unchanged"
    )
    contract["rescue_script"] = str(Path(__file__).resolve())
    contract["rescue_script_sha256"] = bounded.sha256(Path(__file__).resolve())
    contract["rescue_ipm_smoke_qa"] = smoke_qa
    contract["explicit_merge_sources"] = {
        "ES-2016": str(unfinished.resolve()),
        "ES-2017": str(rescue_result_path.resolve()),
    }
    bounded.write_json(contract_path, contract)

    qa = {
        "status": "pass",
        "cost_year": 2030,
        "partition_label": "shard30_of_66",
        "country_year_count": 2,
        "candidate_count": 49,
        "expected_row_count": 200,
        "row_count": int(len(results)),
        "all_optimal": bool(results["solver_status_name"].eq("optimal").all()),
        "maximum_task_balance_residual_mw": float(results["maximum_task_balance_residual_mw"].max()),
        "maximum_shared_recharge_violation_mw": float(results["maximum_shared_recharge_violation_mw"].max()),
        "maximum_soc_balance_residual_mwh": float(max(
            results["base_max_soc_balance_residual_mwh"].max(),
            results["tail_max_soc_balance_residual_mwh"].max(),
        )),
        "minimum_tail_power_mw": float(results["tail_power_mw"].min()),
        "maximum_tail_power_minus_upper_mw": float(
            (results["tail_power_mw"] - results["tail_power_upper_mw"]).max()
        ),
        "runtime_minutes_rescued_ES_2017": (time.perf_counter() - started) / 60.0,
        "output_sha256": bounded.sha256(output_path),
        "solver_algorithms": sorted(results["solver_method"].unique().tolist()),
        "ipm_rescue_smoke_status": smoke_qa["status"],
        "rescue_temporary_output": str(RESCUE_OUTPUT.resolve()),
        "explicit_merge_source_rows": {"ES-2016": 100, "ES-2017": int(len(es2017))},
    }
    if (
        qa["row_count"] != qa["expected_row_count"]
        or not qa["all_optimal"]
        or qa["maximum_task_balance_residual_mw"] > 1.0e-6
        or qa["maximum_shared_recharge_violation_mw"] > 1.0e-6
        or qa["maximum_soc_balance_residual_mwh"] > 1.0e-6
        or qa["maximum_tail_power_minus_upper_mw"] > 1.0e-6
        or qa["minimum_tail_power_mw"] < -1.0e-6
        or qa["ipm_rescue_smoke_status"] != "pass"
    ):
        qa["status"] = "fail"
    bounded.write_json(OUTPUT / "qa.json", qa)
    if qa["status"] == "pass":
        unfinished.unlink()
    print(json.dumps(qa, indent=2), flush=True)
    if qa["status"] != "pass":
        raise RuntimeError("ES-2017 IPM rescue QA failed")


if __name__ == "__main__":
    main()
