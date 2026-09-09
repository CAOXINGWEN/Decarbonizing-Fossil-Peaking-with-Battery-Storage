#!/usr/bin/env python3
"""Solve duration-by-cluster-intensity surfaces for three balancing regimes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_RUNNER = ROOT / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"
PRICE_RUNNER = ROOT / "price_channel" / "run_price_channel_revaluation_v0_1.py"
DEFAULT_OUTPUT = HERE / "runs" / "smoke"
DEFAULT_CLUSTER_DAYS = tuple(range(1, 13))
PARITY_THRESHOLD = 5.2960380996609935

# This is the registered mechanism-signature catalogue used by the existing
# 2/4/6 d European experiment. It covers 24-168 h and retains all three 24 h
# technologies before taking an objective-aligned system-cost envelope.
CANDIDATE_CONFIG_IDS = (
    "caes_24h",
    "h2_cavern_24h",
    "laes_24h",
    "h2_cavern_28h",
    "h2_cavern_32h",
    "h2_cavern_36h",
    "h2_cavern_40h",
    "h2_cavern_44h",
    "h2_cavern_48h",
    "h2_cavern_60h",
    "h2_cavern_72h",
    "h2_cavern_96h",
    "h2_cavern_100h",
    "h2_cavern_120h",
    "h2_cavern_144h",
    "h2_cavern_168h",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("v60_cluster_surface_base", BASE_RUNNER)
price = load_module("v60_cluster_surface_price", PRICE_RUNNER)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def parse_cluster_days(value: str) -> tuple[int, ...]:
    days = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not days or any(item <= 0 for item in days):
        raise ValueError("Cluster lengths must be positive integers")
    if len(days) != len(set(days)):
        raise ValueError("Cluster lengths must be unique")
    return tuple(sorted(days))


def load_panel(regime: str) -> tuple[pd.DataFrame, Path]:
    common = [
        "source_hour_id",
        "source_day_id",
        "season_segment",
        "event_flag_q75",
        base.BASE_TASK_COLUMN,
        base.TAIL_TASK_COLUMN,
        base.RECHARGE_COLUMN,
        "forced_discharge_power_mw",
    ]
    if regime == "Europe":
        path = Path(base.DEFAULT_PANEL)
        frame = pd.read_csv(path, usecols=["country", "year", *common])
        frame["system_id"] = frame["country"].astype(str)
        frame["observation_year"] = frame["year"].astype(int)
        frame["inference_block"] = frame["country"].astype(str)
    elif regime == "China":
        path = ROOT / "cross_scale_cases" / "china_province_panel" / "china_province_event_task_panel.csv.gz"
        frame = pd.read_csv(path, usecols=["province", "year", *common])
        frame["system_id"] = "CN-" + frame["province"].astype(str)
        frame["observation_year"] = frame["year"].astype(int)
        frame["inference_block"] = frame["province"].astype(str)
    elif regime == "Japan":
        path = ROOT / "cross_scale_cases" / "japan_community_panel" / "japan_community_event_task_panel.csv.gz"
        frame = pd.read_csv(path, usecols=["prefecture", "mesh_id", "fiscal_year", *common])
        frame["system_id"] = (
            "JP-" + frame["prefecture"].astype(str) + "-" + frame["mesh_id"].astype(str)
        )
        frame["observation_year"] = frame["fiscal_year"].astype(int)
        frame["inference_block"] = frame["prefecture"].astype(str)
    else:
        raise ValueError(regime)
    frame = frame.sort_values(
        ["system_id", "observation_year", "source_hour_id"], kind="stable"
    ).reset_index(drop=True)
    return frame, path


def day_orders(frame: pd.DataFrame, cluster_days: int) -> dict[str, list[int]]:
    daily = (
        frame.groupby("source_day_id", sort=True)
        .agg(segment=("season_segment", "first"), event=("event_flag_q75", "max"))
        .reset_index()
    )
    if len(daily) != 365:
        raise RuntimeError("Each system-year must contain 365 complete days")
    dispersed: list[int] = []
    clustered: list[int] = []
    for segment in sorted(daily["segment"].unique()):
        subset = daily.loc[daily["segment"].eq(segment)]
        event_days = subset.loc[subset["event"].astype(bool), "source_day_id"].astype(int).tolist()
        normal_days = subset.loc[~subset["event"].astype(bool), "source_day_id"].astype(int).tolist()
        dispersed.extend(base.interleave(event_days, normal_days))
        clustered.extend(base.interleave_clusters(event_days, normal_days, cluster_days))
    expected = list(range(365))
    for label, order in (("dispersed", dispersed), ("clustered", clustered)):
        if len(order) != 365 or sorted(order) != expected:
            raise RuntimeError(f"Invalid {label} intact-day order")
    return {"dispersed": dispersed, "clustered": clustered}


def longest_run(flags: np.ndarray) -> int:
    longest = 0
    run = 0
    for flag in flags.astype(bool):
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def chronology_metrics(frame: pd.DataFrame, order: list[int]) -> dict[str, float]:
    event_by_day = frame.groupby("source_day_id", sort=True)["event_flag_q75"].max().astype(bool)
    flags = np.asarray([bool(event_by_day.loc[day]) for day in order], dtype=bool)
    positions = np.flatnonzero(flags)
    if len(positions) < 2:
        raise RuntimeError("Each system-year must contain at least two event days")
    recovery_days = np.diff(positions) - 1
    return {
        "event_days": int(flags.sum()),
        "longest_event_run_days": longest_run(flags),
        "zero_recovery_window_share": float(np.mean(recovery_days == 0)),
        "median_recovery_hours": float(np.median(recovery_days) * 24.0),
        "mean_recovery_hours": float(np.mean(recovery_days) * 24.0),
    }


def solve_chronology(
    frame: pd.DataFrame,
    order: list[int],
    base_config: pd.Series,
    candidates: pd.DataFrame,
    scenario: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_task = frame[base.BASE_TASK_COLUMN].to_numpy(float)
    original_tail = frame[base.TAIL_TASK_COLUMN].to_numpy(float)
    original_recharge = frame[base.RECHARGE_COLUMN].to_numpy(float)
    task = base.reorder_days(original_task, order)
    tail_task = base.reorder_days(original_tail, order)
    recharge = base.reorder_days(original_recharge, order)
    power = float(frame["forced_discharge_power_mw"].iloc[0])
    if power <= 0:
        raise RuntimeError("Tail power must be positive")
    vectors = {
        "S0": (
            np.full(len(task), scenario["energy_value_eur_mwh"], dtype=float),
            np.full(len(task), scenario["charge_price_eur_mwh"], dtype=float),
        )
    }
    arm_defs: list[tuple[str, pd.Series]] = [("extra_lfp4_comparator", base_config)]
    arm_defs.extend(("ldes_candidate", row) for _, row in candidates.iterrows())
    arm_rows: list[dict[str, Any]] = []
    for arm_type, config in arm_defs:
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
        identity = base.config_identity(config)
        system_cost = float(price.revalue(summary, schedule, vectors)["S0_system_cost_eur"])
        total_charge = float(
            schedule["base_charge_mw"].sum() + schedule["tail_charge_mw"].sum()
        )
        total_discharge = float(task.sum() - float(summary["unserved_eens_mwh"]))
        arm_rows.append(
            {
                "arm_type": arm_type,
                **identity,
                "system_cost_eur": system_cost,
                "tail_power_mw": power,
                "tail_equivalent_full_cycles": float(summary["tail_equivalent_full_cycles"]),
                "total_charge_mwh": total_charge,
                "total_discharge_mwh": total_discharge,
                "unserved_eens_mwh": float(summary["unserved_eens_mwh"]),
                "portfolio_efc_mw": float(summary["portfolio_efc_mw"]),
                "lexicographic_secondary_success": bool(summary["lexicographic_secondary_success"]),
            }
        )
    arms = pd.DataFrame(arm_rows)
    comparator_cost = float(
        arms.loc[arms["arm_type"].eq("extra_lfp4_comparator"), "system_cost_eur"].iloc[0]
    )
    candidates_only = arms.loc[arms["arm_type"].eq("ldes_candidate")].copy()
    candidates_only["benefit_vs_extra_lfp4_eur_kw_year"] = (
        comparator_cost - candidates_only["system_cost_eur"]
    ) / (power * 1000.0)
    envelope = (
        candidates_only.sort_values(
            ["duration_h", "benefit_vs_extra_lfp4_eur_kw_year"],
            ascending=[True, False],
            kind="stable",
        )
        .groupby("duration_h", as_index=False, sort=True)
        .first()
    )
    return arms, envelope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["Europe", "China", "Japan"], required=True)
    parser.add_argument(
        "--cluster-days",
        default=",".join(map(str, DEFAULT_CLUSTER_DAYS)),
        help="Comma-separated target cluster lengths in days",
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cluster_days_values = parse_cluster_days(args.cluster_days)
    if not 1 <= args.shard_index <= args.shard_count:
        raise ValueError("Invalid shard index")
    args.output.mkdir(parents=True, exist_ok=True)
    panel, panel_path = load_panel(args.regime)
    units = sorted(
        panel[["system_id", "observation_year"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    shards = np.array_split(np.asarray(units, dtype=object), args.shard_count)
    selected = [(str(a), int(b)) for a, b in shards[args.shard_index - 1].tolist()]
    catalogue = pd.read_csv(base.DEFAULT_CATALOGUE)
    base_config, candidates = base.select_catalogue(catalogue, list(CANDIDATE_CONFIG_IDS))
    scenario = base.load_scenario(base.DEFAULT_SCENARIO, 20)
    duration_count = int(candidates["duration_h"].nunique())
    started = time.perf_counter()
    effect_frames: list[pd.DataFrame] = []
    arm_frames: list[pd.DataFrame] = []
    conservation_errors: list[float] = []
    for unit_index, (system_id, observation_year) in enumerate(selected, start=1):
        frame = panel.loc[
            panel["system_id"].eq(system_id)
            & panel["observation_year"].eq(observation_year)
        ].copy()
        if len(frame) != 8760:
            raise RuntimeError(f"{system_id}-{observation_year} has {len(frame)} hours")
        inference_block = str(frame["inference_block"].iloc[0])
        first_orders = day_orders(frame, cluster_days_values[0])
        dispersed_order = first_orders["dispersed"]
        dispersed_metrics = chronology_metrics(frame, dispersed_order)
        dispersed_arms, dispersed_envelope = solve_chronology(
            frame, dispersed_order, base_config, candidates, scenario
        )
        original_sums = np.asarray(
            [
                frame[base.BASE_TASK_COLUMN].sum(),
                frame[base.TAIL_TASK_COLUMN].sum(),
                frame[base.RECHARGE_COLUMN].sum(),
            ],
            dtype=float,
        )
        for cluster_days in cluster_days_values:
            orders = day_orders(frame, cluster_days)
            clustered_order = orders["clustered"]
            clustered_metrics = chronology_metrics(frame, clustered_order)
            if clustered_order == dispersed_order:
                clustered_arms = dispersed_arms.copy()
                clustered_envelope = dispersed_envelope.copy()
            else:
                clustered_arms, clustered_envelope = solve_chronology(
                    frame, clustered_order, base_config, candidates, scenario
                )
            for chronology, order in (("dispersed", dispersed_order), ("clustered", clustered_order)):
                reordered_sums = np.asarray(
                    [
                        base.reorder_days(frame[base.BASE_TASK_COLUMN].to_numpy(float), order).sum(),
                        base.reorder_days(frame[base.TAIL_TASK_COLUMN].to_numpy(float), order).sum(),
                        base.reorder_days(frame[base.RECHARGE_COLUMN].to_numpy(float), order).sum(),
                    ],
                    dtype=float,
                )
                conservation_errors.append(float(np.max(np.abs(reordered_sums - original_sums))))
            paired = dispersed_envelope[
                [
                    "duration_h",
                    "config_id",
                    "tech_id",
                    "benefit_vs_extra_lfp4_eur_kw_year",
                    "tail_equivalent_full_cycles",
                ]
            ].merge(
                clustered_envelope[
                    [
                        "duration_h",
                        "config_id",
                        "tech_id",
                        "benefit_vs_extra_lfp4_eur_kw_year",
                        "tail_equivalent_full_cycles",
                    ]
                ],
                on="duration_h",
                suffixes=("_dispersed", "_clustered"),
                validate="one_to_one",
            )
            paired["regime"] = args.regime
            paired["system_id"] = system_id
            paired["observation_year"] = observation_year
            paired["inference_block"] = inference_block
            paired["target_cluster_days"] = cluster_days
            for key, value in dispersed_metrics.items():
                paired[f"{key}_dispersed"] = value
            for key, value in clustered_metrics.items():
                paired[f"{key}_clustered"] = value
            paired["sequencing_effect_eur_kw_year"] = (
                paired["benefit_vs_extra_lfp4_eur_kw_year_clustered"]
                - paired["benefit_vs_extra_lfp4_eur_kw_year_dispersed"]
            )
            paired["absolute_sequencing_shift_eur_kw_year"] = paired[
                "sequencing_effect_eur_kw_year"
            ].abs()
            paired["reaches_cost_scale"] = paired[
                "absolute_sequencing_shift_eur_kw_year"
            ].ge(PARITY_THRESHOLD)
            effect_frames.append(paired)
            for chronology, arms in (("dispersed", dispersed_arms), ("clustered", clustered_arms)):
                labelled = arms.copy()
                labelled["regime"] = args.regime
                labelled["system_id"] = system_id
                labelled["observation_year"] = observation_year
                labelled["inference_block"] = inference_block
                labelled["target_cluster_days"] = cluster_days
                labelled["chronology"] = chronology
                arm_frames.append(labelled)
        effects = pd.concat(effect_frames, ignore_index=True)
        arms = pd.concat(arm_frames, ignore_index=True)
        output_csv_gz(effects, args.output / "system_duration_effects_checkpoint.csv.gz")
        output_csv_gz(arms, args.output / "arm_results_checkpoint.csv.gz")
        print(
            f"{args.regime} shard {args.shard_index}: "
            f"{unit_index}/{len(selected)} {system_id}-{observation_year}",
            flush=True,
        )
    effects = pd.concat(effect_frames, ignore_index=True)
    arms = pd.concat(arm_frames, ignore_index=True)
    output_csv_gz(effects, args.output / "system_duration_effects.csv.gz")
    output_csv_gz(arms, args.output / "arm_results.csv.gz")
    expected_effect_rows = len(selected) * len(cluster_days_values) * duration_count
    cluster1_effects = effects.loc[
        effects["target_cluster_days"].eq(1), "sequencing_effect_eur_kw_year"
    ]
    maximum_cluster1_effect = (
        float(cluster1_effects.abs().max()) if len(cluster1_effects) else None
    )
    qa = {
        "status": "pass",
        "regime": args.regime,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected_system_years": selected,
        "cluster_days": cluster_days_values,
        "candidate_config_ids": list(CANDIDATE_CONFIG_IDS),
        "duration_count": duration_count,
        "expected_effect_rows": expected_effect_rows,
        "actual_effect_rows": int(len(effects)),
        "maximum_absolute_conservation_error_mwh": float(max(conservation_errors, default=0.0)),
        "maximum_absolute_cluster1_effect_eur_kw_year": maximum_cluster1_effect,
        "secondary_solve_success_share": float(arms["lexicographic_secondary_success"].mean()),
        "input_panel": str(panel_path.resolve()),
        "input_panel_sha256": sha256(panel_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "scenario": scenario,
        "value_metric": "exact solver-system-cost difference versus additional 4 h LFP",
        "efc_role": "operational diagnostic only",
        "elapsed_seconds": time.perf_counter() - started,
    }
    if qa["actual_effect_rows"] != qa["expected_effect_rows"]:
        qa["status"] = "fail"
    if qa["maximum_absolute_conservation_error_mwh"] > 1.0e-6:
        qa["status"] = "fail"
    if maximum_cluster1_effect is not None and maximum_cluster1_effect > 1.0e-9:
        qa["status"] = "fail"
    (args.output / "qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if qa["status"] != "pass":
        raise RuntimeError(qa)


if __name__ == "__main__":
    main()
