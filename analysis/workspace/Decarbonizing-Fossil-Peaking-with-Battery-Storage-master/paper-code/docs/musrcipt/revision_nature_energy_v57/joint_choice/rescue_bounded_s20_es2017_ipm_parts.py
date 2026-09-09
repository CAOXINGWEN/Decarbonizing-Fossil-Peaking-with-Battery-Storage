#!/usr/bin/env python3
"""Short-batch IPM rescue and explicit merge for ES-2017 at 2030 cost.

The execution environment terminates a single long foreground job after
roughly 90 seconds.  This driver partitions only the 49 catalogue candidates;
each batch solves the identical annual LP and includes the common LFP arm.
The merge verifies duplicated LFP solutions before accepting one copy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_bounded_s20_highs as bounded
from rescue_bounded_s20_es2017_ipm import (
    IpmDispatcher, SMOKE_OUTPUT, chronology_rows, ipm_smoke,
)


HERE = Path(__file__).resolve().parent
FINAL_OUTPUT = HERE / "bounded_s20_highs" / "cost_2030" / "shard30_of_66"
PART_ROOT = (
    HERE / "bounded_s20_highs" / "cost_2030"
    / "rescue_es2017_ipm_temporary"
)
PART_COUNT = 10
KEYS = ["country", "year", "chronology", "arm_type", "config_id"]
COMPARE = [
    "tail_power_mw", "total_charge_mwh", "total_discharge_mwh",
    "unserved_eens_mwh", "portfolio_efc_mw", "residual_peak_mw",
    "system_cost_eur",
]


def load_smoke(dispatcher: IpmDispatcher) -> dict:
    path = SMOKE_OUTPUT / "smoke_qa.json"
    if path.exists():
        qa = json.loads(path.read_text())
        if qa.get("status") == "pass":
            return qa
    return ipm_smoke(dispatcher)


def run_part(index: int) -> None:
    dispatcher = IpmDispatcher()
    bounded.linprog = dispatcher
    smoke = load_smoke(dispatcher)
    runner, panel, base_row, candidates, scenario = bounded.prepare_inputs(2030, 20)
    candidate_indices = np.array_split(np.arange(len(candidates)), PART_COUNT)[index - 1]
    subset = candidates.iloc[candidate_indices].copy()
    frame = panel[panel["country"].eq("ES") & panel["year"].eq(2017)].copy()
    started = time.perf_counter()
    rows, chronology = bounded.solve_country_year(
        runner, frame, base_row, subset, scenario, 20, 2030
    )
    result = pd.DataFrame(rows)
    result["solver_method"] = np.where(
        result["secondary_tiebreak_success"],
        "scipy_highs_ipm_two_stage_rescue",
        "scipy_highs_ipm_primary_optimum_secondary_numerical_fallback_rescue",
    )
    output = PART_ROOT / f"part{index:02d}_of_{PART_COUNT:02d}"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "arm_results.csv.gz"
    result.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(chronology).to_csv(output / "chronology_conservation.csv", index=False)
    expected = 2 * (1 + len(subset))
    qa = {
        "status": "pass",
        "part_index": index, "part_count": PART_COUNT,
        "country": "ES", "year": 2017, "cost_year": 2030,
        "candidate_ids": subset["config_id"].astype(str).tolist(),
        "candidate_count": int(len(subset)),
        "includes_common_lfp_comparator": True,
        "row_count": int(len(result)), "expected_row_count": expected,
        "all_optimal": bool(result["solver_status_name"].eq("optimal").all()),
        "maximum_task_balance_residual_mw": float(result["maximum_task_balance_residual_mw"].max()),
        "maximum_shared_recharge_violation_mw": float(result["maximum_shared_recharge_violation_mw"].max()),
        "maximum_soc_balance_residual_mwh": float(max(
            result["base_max_soc_balance_residual_mwh"].max(),
            result["tail_max_soc_balance_residual_mwh"].max(),
        )),
        "runtime_seconds": time.perf_counter() - started,
        "solver": "scipy_highs_ipm_two_stage_same_LP_and_lexicographic_cut",
        "ipm_smoke_qa": smoke,
        "output_sha256": bounded.sha256(path),
    }
    if (
        qa["row_count"] != expected or not qa["all_optimal"]
        or qa["maximum_task_balance_residual_mw"] > 1.0e-6
        or qa["maximum_shared_recharge_violation_mw"] > 1.0e-6
        or qa["maximum_soc_balance_residual_mwh"] > 1.0e-6
        or smoke.get("status") != "pass"
    ):
        qa["status"] = "fail"
    bounded.write_json(output / "qa.json", qa)
    print(json.dumps(qa, indent=2), flush=True)
    if qa["status"] != "pass":
        raise RuntimeError(f"Rescue part {index} QA failed")


def merge_parts() -> None:
    unfinished = FINAL_OUTPUT / "portfolio_arm_results_unfinished.csv.gz"
    if not unfinished.exists():
        raise RuntimeError("The completed ES-2016 checkpoint is missing")
    es2016 = pd.read_csv(unfinished)
    if set(zip(es2016["country"], es2016["year"])) != {("ES", 2016)} or len(es2016) != 100:
        raise RuntimeError("The checkpoint is not the expected 100-row ES-2016 block")
    frames = []
    part_qas = []
    for index in range(1, PART_COUNT + 1):
        directory = PART_ROOT / f"part{index:02d}_of_{PART_COUNT:02d}"
        qa = json.loads((directory / "qa.json").read_text())
        if qa.get("status") != "pass":
            raise RuntimeError(f"Rescue part {index} QA failed")
        part_qas.append(qa)
        frames.append(pd.read_csv(directory / "arm_results.csv.gz"))
    stacked = pd.concat(frames, ignore_index=True)
    lfp = stacked[stacked["config_id"].eq("li_lfp_4h")]
    duplicate_spread = float(
        lfp.groupby(["country", "year", "chronology"])[COMPARE]
        .agg(lambda values: float(values.max() - values.min()))
        .to_numpy().max()
    )
    es2017 = stacked.drop_duplicates(KEYS, keep="first").copy()
    if len(es2017) != 100 or es2017["config_id"].nunique() != 50:
        raise RuntimeError("Explicit rescue merge did not recover 100 unique ES-2017 arms")
    if duplicate_spread > 1.0e-6:
        raise RuntimeError("Duplicated common-LFP batch solutions are inconsistent")
    results = pd.concat([es2016, es2017], ignore_index=True)
    if len(results) != 200 or results.duplicated(KEYS).any():
        raise RuntimeError("Final shard does not contain 200 unique arm rows")

    chronology_es2017 = pd.read_csv(
        PART_ROOT / f"part01_of_{PART_COUNT:02d}" / "chronology_conservation.csv"
    )
    runner, panel, _, _, _ = bounded.prepare_inputs(2030, 20)
    es2016_frame = panel[panel["country"].eq("ES") & panel["year"].eq(2016)].copy()
    conservation = pd.concat([
        pd.DataFrame(chronology_rows(runner, es2016_frame)), chronology_es2017
    ], ignore_index=True)

    result_path = FINAL_OUTPUT / "portfolio_arm_results.csv.gz"
    results.to_csv(result_path, index=False, compression={"method": "gzip", "mtime": 0})
    conservation.to_csv(FINAL_OUTPUT / "chronology_conservation.csv", index=False)
    contract_path = FINAL_OUTPUT / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    contract.update({
        "model": (
            "registered_v42_continuous_portfolio_lp__ES2016_highs_ds__"
            "ES2017_equivalent_highs_ipm_batched_rescue"
        ),
        "solver_assignments": {
            "ES-2016": "scipy_highs_dual_simplex_two_stage",
            "ES-2017": "scipy_highs_ipm_two_stage_same_LP_and_lexicographic_cut",
        },
        "rescue_reason": (
            "dual-simplex ES-2017 exceeded the execution window; only the "
            "continuous-LP algorithm and batch execution changed"
        ),
        "rescue_script": str(Path(__file__).resolve()),
        "rescue_script_sha256": bounded.sha256(Path(__file__).resolve()),
        "rescue_part_root": str(PART_ROOT.resolve()),
        "rescue_part_count": PART_COUNT,
        "rescue_ipm_smoke_qa": part_qas[0]["ipm_smoke_qa"],
        "explicit_merge_sources": {
            "ES-2016": str(unfinished.resolve()),
            "ES-2017_parts": [
                str((PART_ROOT / f"part{i:02d}_of_{PART_COUNT:02d}" / "arm_results.csv.gz").resolve())
                for i in range(1, PART_COUNT + 1)
            ],
        },
    })
    bounded.write_json(contract_path, contract)
    qa = {
        "status": "pass", "cost_year": 2030,
        "partition_label": "shard30_of_66", "country_year_count": 2,
        "candidate_count": 49, "expected_row_count": 200,
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
        "common_lfp_duplicate_maximum_spread": duplicate_spread,
        "solver_algorithms": sorted(results["solver_method"].unique().tolist()),
        "ipm_rescue_smoke_status": part_qas[0]["ipm_smoke_qa"]["status"],
        "rescue_parts_all_pass": all(item["status"] == "pass" for item in part_qas),
        "rescue_part_count": PART_COUNT,
        "output_sha256": bounded.sha256(result_path),
    }
    if (
        qa["row_count"] != qa["expected_row_count"] or not qa["all_optimal"]
        or qa["maximum_task_balance_residual_mw"] > 1.0e-6
        or qa["maximum_shared_recharge_violation_mw"] > 1.0e-6
        or qa["maximum_soc_balance_residual_mwh"] > 1.0e-6
        or qa["maximum_tail_power_minus_upper_mw"] > 1.0e-6
        or qa["minimum_tail_power_mw"] < -1.0e-6
        or duplicate_spread > 1.0e-6
        or not qa["rescue_parts_all_pass"]
        or qa["ipm_rescue_smoke_status"] != "pass"
    ):
        qa["status"] = "fail"
    bounded.write_json(FINAL_OUTPUT / "qa.json", qa)
    if qa["status"] == "pass":
        unfinished.unlink()
    print(json.dumps(qa, indent=2), flush=True)
    if qa["status"] != "pass":
        raise RuntimeError("Explicit ES-2017 rescue merge QA failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["part", "merge"], required=True)
    parser.add_argument("--part-index", type=int)
    args = parser.parse_args()
    if args.mode == "part":
        if args.part_index is None or not 1 <= args.part_index <= PART_COUNT:
            raise ValueError(f"--part-index must be 1..{PART_COUNT}")
        run_part(args.part_index)
    else:
        merge_parts()


if __name__ == "__main__":
    main()
