#!/usr/bin/env python3
"""Test recharge-quantity multipliers for the focal cross-regime comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PORTFOLIO = HERE / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"
PRICE_RUNNER = HERE / "price_channel" / "run_price_channel_revaluation_v0_1.py"
CASES = HERE / "cross_scale_cases"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("cross_scale_base", PORTFOLIO)
price = load_module("cross_scale_price", PRICE_RUNNER)


def load_case(case: str) -> tuple[pd.DataFrame, str]:
    if case == "china":
        path = CASES / "china_province_panel" / "china_province_event_task_panel.csv.gz"
        frame = pd.read_csv(path)
        frame["case_id"] = "CN-" + frame["province"].astype(str)
        frame["case_year"] = frame["year"].astype(int)
        frame["raw_mean_mw"] = frame["raw_task_mean_mw"]
    elif case == "japan":
        path = CASES / "japan_community_panel" / "japan_community_event_task_panel.csv.gz"
        frame = pd.read_csv(path)
        frame["case_id"] = "JP-" + frame["prefecture"].astype(str) + "-" + frame["mesh_id"].astype(str)
        frame["case_year"] = frame["fiscal_year"].astype(int)
        frame["raw_mean_mw"] = frame["raw_grid_import_mean_mw"]
    else:
        raise ValueError(case)
    return frame, str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["china", "japan"], required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--recharge-multiplier", type=float, choices=[1.0, 2.0, 4.0], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    panel, panel_path = load_case(args.case)
    pairs = sorted(panel[["case_id", "case_year"]].drop_duplicates().itertuples(index=False, name=None))
    shards = np.array_split(np.asarray(pairs, dtype=object), args.shard_count)
    selected = [(str(a), int(b)) for a, b in shards[args.shard_index - 1].tolist()]
    catalogue = pd.read_csv(base.DEFAULT_CATALOGUE)
    base_config, candidates = base.select_catalogue(catalogue, None)
    candidates = candidates[candidates.config_id.eq("caes_24h")].copy()
    if len(candidates) != 1:
        raise RuntimeError("The focal catalogue must contain exactly one caes_24h row")
    scenario = base.load_scenario(base.DEFAULT_SCENARIO, 20)
    rows = []
    started = time.perf_counter()
    for index, (case_id, year) in enumerate(selected, start=1):
        frame = panel[panel.case_id.eq(case_id) & panel.case_year.eq(year)].copy()
        adapter = frame.rename(columns={"case_id": "country", "case_year": "year"})
        orders, stats = base.day_orders(adapter)
        original_task = frame[base.BASE_TASK_COLUMN].to_numpy(float)
        original_tail = frame[base.TAIL_TASK_COLUMN].to_numpy(float)
        original_recharge = (
            frame[base.RECHARGE_COLUMN].to_numpy(float) * args.recharge_multiplier
        )
        power = float(frame["forced_discharge_power_mw"].iloc[0])
        for chronology, order in orders.items():
            task = base.reorder_days(original_task, order)
            tail_task = base.reorder_days(original_tail, order)
            recharge = base.reorder_days(original_recharge, order)
            arms = [("base_only", base_config, 0.0), ("extra_lfp4_comparator", base_config, power)]
            arms.extend(("ldes_candidate", row, power) for _, row in candidates.iterrows())
            for arm_type, config, tail_power in arms:
                summary, schedule = price.solve_fixed_arm_highs(
                    task, tail_task, recharge, power, tail_power,
                    base_config, config, scenario,
                )
                identity = base.config_identity(config)
                if arm_type == "base_only":
                    identity.update(config_id="base_only_lfp4", tech_id="none", duration_h=0.0)
                rows.append(
                    {
                        "application_case": args.case,
                        "case_id": case_id,
                        "case_year": year,
                        "recharge_multiplier": args.recharge_multiplier,
                        "chronology": chronology,
                        "arm_type": arm_type,
                        **identity,
                        **stats,
                        **summary,
                        **price.revalue(summary, schedule, price.price_vectors(task, recharge, price.event_run_index(adapter, order), scenario, None)),
                    }
                )
        pd.DataFrame(rows).to_csv(args.output / "cross_scale_results_checkpoint.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        print(f"{args.case} shard {args.shard_index}: {index}/{len(selected)} {case_id}-{year}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "cross_scale_results.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (args.output / "run_contract.json").write_text(
        json.dumps(
            {
                "application_case": args.case,
                "input_panel": panel_path,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "case_years": selected,
                "rows": len(result),
                "catalogue_nodes": 1,
                "focal_config_id": "caes_24h",
                "recharge_multiplier": args.recharge_multiplier,
                "chronologies": ["dispersed", "clustered"],
                "common_value_scenario": scenario,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
