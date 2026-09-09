#!/usr/bin/env python3
"""Run the preregistered 2 x 2 x 2 recharge-constraint stress test.

The three factors are deliberately separable:

* available usable recharge energy: 1x or 4x;
* charge-interface power for both storage layers: 1x or 4x; and
* recharge-hour accessibility: observed hours or every non-task hour.

The accessibility intervention preserves the chronology-specific total usable
recharge energy and redistributes it uniformly across hours in which the C1
task is zero.  It is an optimistic timing/accessibility stress test, not a
network-expansion model.  Discharge power, energy capacity, task, event days,
technology costs and the S0 valuation scenario remain fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PORTFOLIO = ROOT / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"
PRICE_RUNNER = ROOT / "price_channel" / "run_price_channel_revaluation_v0_1.py"
CASES = ROOT / "cross_scale_cases"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("v62_recharge_factorial_base", PORTFOLIO)
price = load_module("v62_recharge_factorial_price", PRICE_RUNNER)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_panel(regime: str) -> tuple[pd.DataFrame, Path]:
    if regime == "europe":
        path = Path(base.DEFAULT_PANEL)
        columns = [
            "country", "year", "source_hour_id", "source_day_id",
            "season_segment", "event_flag_q75", "raw_gas_mean_mw",
            "forced_discharge_power_mw", base.BASE_TASK_COLUMN,
            base.TAIL_TASK_COLUMN, base.RECHARGE_COLUMN,
        ]
        frame = pd.read_csv(path, usecols=columns)
        frame["case_id"] = frame["country"].astype(str)
        frame["case_year"] = frame["year"].astype(int)
    elif regime == "china":
        path = CASES / "china_province_panel" / "china_province_event_task_panel.csv.gz"
        frame = pd.read_csv(path)
        frame["case_id"] = "CN-" + frame["province"].astype(str)
        frame["case_year"] = frame["year"].astype(int)
        frame["country"] = frame["case_id"]
        frame["year"] = frame["case_year"]
    elif regime == "japan":
        path = CASES / "japan_community_panel" / "japan_community_event_task_panel.csv.gz"
        frame = pd.read_csv(path)
        frame["case_id"] = (
            "JP-" + frame["prefecture"].astype(str) + "-" + frame["mesh_id"].astype(str)
        )
        frame["case_year"] = frame["fiscal_year"].astype(int)
        frame["country"] = frame["case_id"]
        frame["year"] = frame["case_year"]
    else:
        raise ValueError(regime)
    frame = frame.sort_values(["case_id", "case_year", "source_hour_id"], kind="stable")
    return frame, path


def accessible_recharge(
    task: np.ndarray,
    recharge: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    open_mask = task <= base.NUMERICAL_ZERO
    usable_observed = np.where(open_mask, recharge, 0.0)
    observed_total = float(usable_observed.sum())
    observed_positive_hours = int(np.count_nonzero(usable_observed > base.NUMERICAL_ZERO))
    if mode == "observed":
        result = recharge.copy()
    elif mode == "expanded":
        open_hours = int(open_mask.sum())
        if open_hours == 0 and observed_total > base.NUMERICAL_ZERO:
            raise RuntimeError("Usable recharge cannot be redistributed without an open hour")
        result = np.zeros_like(recharge)
        if open_hours:
            result[open_mask] = observed_total / open_hours
    else:
        raise ValueError(mode)
    usable_result = np.where(open_mask, result, 0.0)
    conservation_error = float(usable_result.sum() - observed_total)
    tolerance = max(1.0e-8, 1.0e-12 * max(abs(observed_total), 1.0))
    if abs(conservation_error) > tolerance:
        raise RuntimeError(f"Usable-recharge conservation error: {conservation_error}")
    return result, {
        "open_non_task_hours": int(open_mask.sum()),
        "observed_positive_recharge_hours": observed_positive_hours,
        "intervention_positive_recharge_hours": int(
            np.count_nonzero(usable_result > base.NUMERICAL_ZERO)
        ),
        "usable_recharge_mwh": float(usable_result.sum()),
        "usable_recharge_conservation_error_mwh": conservation_error,
    }


def scaled_config(config: pd.Series, multiplier: float) -> pd.Series:
    result = config.copy()
    result["charge_power_ratio"] = float(result["charge_power_ratio"]) * multiplier
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["europe", "china", "japan"], required=True)
    parser.add_argument("--energy-multiplier", type=float, choices=[1.0, 4.0], required=True)
    parser.add_argument("--charge-power-multiplier", type=float, choices=[1.0, 4.0], required=True)
    parser.add_argument("--window-mode", choices=["observed", "expanded"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    panel, panel_path = load_panel(args.regime)
    pairs = sorted(
        panel[["case_id", "case_year"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if args.limit is not None:
        pairs = pairs[: args.limit]

    catalogue = pd.read_csv(base.DEFAULT_CATALOGUE)
    base_config_raw, candidates = base.select_catalogue(catalogue, None)
    caes_rows = candidates[candidates["config_id"].eq("caes_24h")]
    if len(caes_rows) != 1:
        raise RuntimeError("The focal catalogue must contain exactly one caes_24h row")
    caes_config_raw = caes_rows.iloc[0]
    base_config = scaled_config(base_config_raw, args.charge_power_multiplier)
    caes_config = scaled_config(caes_config_raw, args.charge_power_multiplier)
    scenario = base.load_scenario(base.DEFAULT_SCENARIO, 20)

    rows: list[dict[str, object]] = []
    chronology_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, (case_id, year) in enumerate(pairs, start=1):
        frame = panel[panel["case_id"].eq(case_id) & panel["case_year"].eq(year)].copy()
        if len(frame) != 8760:
            raise RuntimeError(f"{case_id}-{year} has {len(frame)} hours")
        adapter = frame.rename(columns={"case_id": "country", "case_year": "year"})
        orders, order_stats = base.day_orders(adapter)
        original_task = frame[base.BASE_TASK_COLUMN].to_numpy(float)
        original_tail = frame[base.TAIL_TASK_COLUMN].to_numpy(float)
        original_recharge = (
            frame[base.RECHARGE_COLUMN].to_numpy(float) * args.energy_multiplier
        )
        power = float(frame["forced_discharge_power_mw"].iloc[0])
        for chronology, order in orders.items():
            task = base.reorder_days(original_task, order)
            tail_task = base.reorder_days(original_tail, order)
            recharge_ordered = base.reorder_days(original_recharge, order)
            recharge, recharge_qa = accessible_recharge(
                task, recharge_ordered, args.window_mode
            )
            chronology_rows.append(
                {
                    "regime": args.regime,
                    "case_id": str(case_id),
                    "case_year": int(year),
                    "chronology": chronology,
                    "energy_multiplier": args.energy_multiplier,
                    "charge_power_multiplier": args.charge_power_multiplier,
                    "window_mode": args.window_mode,
                    **order_stats,
                    **recharge_qa,
                }
            )
            arms = [
                ("extra_lfp4_comparator", base_config),
                ("ldes_candidate", caes_config),
            ]
            s0_vectors = {
                "S0": (
                    np.full(len(task), scenario["energy_value_eur_mwh"], dtype=float),
                    np.full(len(task), scenario["charge_price_eur_mwh"], dtype=float),
                )
            }
            for arm_type, config in arms:
                summary, schedule = price.solve_fixed_arm_highs(
                    task,
                    tail_task,
                    recharge,
                    power,
                    power,
                    base_config,
                    config,
                    scenario,
                )
                rows.append(
                    {
                        "regime": args.regime,
                        "case_id": str(case_id),
                        "case_year": int(year),
                        "chronology": chronology,
                        "energy_multiplier": args.energy_multiplier,
                        "charge_power_multiplier": args.charge_power_multiplier,
                        "window_mode": args.window_mode,
                        "arm_type": arm_type,
                        **base.config_identity(config),
                        **summary,
                        **price.revalue(summary, schedule, s0_vectors),
                    }
                )
        pd.DataFrame(rows).to_csv(
            args.output / "arm_results_checkpoint.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        print(
            f"{args.regime} E{args.energy_multiplier:g} P{args.charge_power_multiplier:g} "
            f"{args.window_mode}: {index}/{len(pairs)} {case_id}-{year}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    chronology = pd.DataFrame(chronology_rows)
    result_path = args.output / "arm_results.csv.gz"
    chronology_path = args.output / "chronology_contract.csv.gz"
    result.to_csv(
        result_path, index=False, compression={"method": "gzip", "mtime": 0}
    )
    chronology.to_csv(
        chronology_path, index=False, compression={"method": "gzip", "mtime": 0}
    )
    maximum_conservation_error = float(
        chronology["usable_recharge_conservation_error_mwh"].abs().max()
    )
    contract = {
        "analysis": "v62_recharge_constraint_factorial",
        "regime": args.regime,
        "input_panel": str(panel_path.resolve()),
        "input_panel_sha256": sha256(panel_path),
        "catalogue": str(Path(base.DEFAULT_CATALOGUE).resolve()),
        "catalogue_sha256": sha256(Path(base.DEFAULT_CATALOGUE)),
        "scenario": str(Path(base.DEFAULT_SCENARIO).resolve()),
        "scenario_sha256": sha256(Path(base.DEFAULT_SCENARIO)),
        "script_sha256": sha256(Path(__file__)),
        "case_years": [[str(case), int(year)] for case, year in pairs],
        "n_system_years": len(pairs),
        "expected_arm_rows": len(pairs) * 2 * 2,
        "arm_rows": len(result),
        "chronology_rows": len(chronology),
        "energy_multiplier": args.energy_multiplier,
        "charge_power_multiplier": args.charge_power_multiplier,
        "window_mode": args.window_mode,
        "window_intervention": (
            "chronology-specific usable recharge redistributed uniformly across all non-task hours"
            if args.window_mode == "expanded"
            else "observed recharge hours retained"
        ),
        "usable_recharge_energy_preserved_within_window_intervention": True,
        "maximum_usable_recharge_conservation_error_mwh": maximum_conservation_error,
        "fixed_contract": [
            "C1 and C3 tasks", "event-day sets and chronology", "discharge power",
            "energy capacity", "technology cost", "S0 value scenario",
        ],
        "focal_comparison": "24 h A-CAES versus additional 4 h LFP",
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "result_sha256": sha256(result_path),
        "chronology_sha256": sha256(chronology_path),
    }
    (args.output / "run_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
