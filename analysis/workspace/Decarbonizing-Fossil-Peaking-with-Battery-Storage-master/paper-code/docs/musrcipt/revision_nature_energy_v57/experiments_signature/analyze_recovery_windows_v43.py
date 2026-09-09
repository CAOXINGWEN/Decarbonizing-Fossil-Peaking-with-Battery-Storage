#!/usr/bin/env python3
"""Quantify recovery windows moved by the conserved chronology experiment.

An event-day call is a complete source day with ``event_flag_q75 == 1``.
For each pair of consecutive calls in a reconstructed chronology, the recovery
window is the set of complete, non-event days strictly between the two calls.
Thus adjacent event days have a zero-hour recovery window.  Recharge available
in a window is the sum of the preregistered R2 potential-recharge series over
those intervening days.  This definition matches the intact-24-h-day treatment
and does not use optimized storage dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_v43_cluster_signature_gurobi.py"
WORKSPACE_ROOT = HERE.parents[5]
DEFAULT_PANEL = (
    WORKSPACE_ROOT
    / "dataset"
    / "external_hourly"
    / "europe_mid_duration_valley_upgrade_v0_1"
    / "processed"
    / "observed_gas_event_task_definitions_v0_2.csv.gz"
)
DEFAULT_OUTPUT = HERE / "results_recovery_windows"
EVENT_COLUMN = "event_flag_q75"
RECHARGE_COLUMN = "r2_potential_recharge_mw"
BASE_TASK_COLUMN = "task_c1_mw"
TAIL_TASK_COLUMN = "task_c3_q75_mw"


def load_runner():
    spec = importlib.util.spec_from_file_location("v43_cluster_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner at {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_run(flags: np.ndarray) -> int:
    longest = 0
    run = 0
    for flag in flags.astype(bool):
        if flag:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return int(longest)


def country_block_bootstrap(
    paired: pd.DataFrame,
    column: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    countries = np.asarray(sorted(paired["country"].unique()))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    by_country = {
        country: paired.loc[paired["country"].eq(country), column].to_numpy(float)
        for country in countries
    }
    for _ in range(draws):
        selected = rng.choice(countries, size=len(countries), replace=True)
        sample = np.concatenate([by_country[country] for country in selected])
        values.append(float(np.mean(sample)))
    point = float(paired[column].mean())
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster-days", default="2,4,6")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=43017)
    args = parser.parse_args()
    cluster_days_values = [
        int(value.strip())
        for value in args.cluster_days.split(",")
        if value.strip()
    ]
    if not cluster_days_values or any(value <= 0 for value in cluster_days_values):
        raise ValueError("--cluster-days must contain positive integers")
    args.output.mkdir(parents=True, exist_ok=True)

    runner = load_runner()
    usecols = [
        "country",
        "year",
        "source_hour_id",
        "source_day_id",
        "season_segment",
        EVENT_COLUMN,
        RECHARGE_COLUMN,
        BASE_TASK_COLUMN,
        TAIL_TASK_COLUMN,
    ]
    panel = pd.read_csv(args.panel, usecols=usecols)
    panel = panel.sort_values(
        ["country", "year", "source_hour_id"], kind="stable"
    ).reset_index(drop=True)

    window_rows: list[dict[str, Any]] = []
    country_year_rows: list[dict[str, Any]] = []
    natural_rows: list[dict[str, Any]] = []
    conservation_rows: list[dict[str, Any]] = []
    for (country, year), frame in panel.groupby(["country", "year"], sort=True):
        if len(frame) != 8760:
            raise RuntimeError(f"{country}-{year} has {len(frame)} hours")
        daily = (
            frame.groupby("source_day_id", sort=True)
            .agg(
                event=(EVENT_COLUMN, "max"),
                segment=("season_segment", "first"),
                recharge_mwh=(RECHARGE_COLUMN, "sum"),
                base_task_mwh=(BASE_TASK_COLUMN, "sum"),
                tail_task_mwh=(TAIL_TASK_COLUMN, "sum"),
            )
            .reset_index()
        )
        natural_flags = daily["event"].to_numpy(bool)
        natural_positions = np.flatnonzero(natural_flags)
        natural_gaps = np.diff(natural_positions) - 1
        natural_rows.append(
            {
                "country": country,
                "year": int(year),
                "event_days": int(natural_flags.sum()),
                "natural_longest_event_run_days": longest_run(natural_flags),
                "natural_median_recovery_hours": (
                    float(np.median(natural_gaps * 24.0))
                    if len(natural_gaps)
                    else np.nan
                ),
            }
        )

        event_by_day = daily.set_index("source_day_id")["event"].astype(bool)
        recharge_by_day = daily.set_index("source_day_id")["recharge_mwh"]
        annual_recharge_mwh = float(daily["recharge_mwh"].sum())
        if annual_recharge_mwh <= 0.0:
            raise RuntimeError(f"{country}-{year} has non-positive annual recharge")
        base_by_day = daily.set_index("source_day_id")["base_task_mwh"]
        tail_by_day = daily.set_index("source_day_id")["tail_task_mwh"]
        for cluster_days in cluster_days_values:
            orders, stats = runner.day_orders(frame, cluster_days)
            for chronology, order in orders.items():
                flags = np.asarray(
                    [bool(event_by_day.loc[day]) for day in order], dtype=bool
                )
                recharge = np.asarray(
                    [float(recharge_by_day.loc[day]) for day in order]
                )
                base_task = np.asarray(
                    [float(base_by_day.loc[day]) for day in order]
                )
                tail_task = np.asarray(
                    [float(tail_by_day.loc[day]) for day in order]
                )
                event_positions = np.flatnonzero(flags)
                if len(event_positions) < 2:
                    raise RuntimeError(f"{country}-{year} has <2 event days")
                local_rows: list[dict[str, Any]] = []
                for call_index, (left, right) in enumerate(
                    zip(event_positions[:-1], event_positions[1:]), start=1
                ):
                    interior = slice(int(left + 1), int(right))
                    record = {
                        "country": country,
                        "year": int(year),
                        "target_cluster_days": int(cluster_days),
                        "chronology": chronology,
                        "call_pair_index": call_index,
                        "preceding_event_day_position": int(left),
                        "following_event_day_position": int(right),
                        "recovery_days": int(right - left - 1),
                        "recovery_hours": int((right - left - 1) * 24),
                        "available_recharge_mwh": float(recharge[interior].sum()),
                        "available_recharge_share_of_annual": float(
                            recharge[interior].sum() / annual_recharge_mwh
                        ),
                        "zero_recovery_window": bool(right == left + 1),
                    }
                    local_rows.append(record)
                    window_rows.append(record)
                local = pd.DataFrame(local_rows)
                country_year_rows.append(
                    {
                        "country": country,
                        "year": int(year),
                        "target_cluster_days": int(cluster_days),
                        "chronology": chronology,
                        "event_days": int(flags.sum()),
                        "longest_event_run_days": longest_run(flags),
                        "adjacent_event_pairs": int(
                            local["zero_recovery_window"].sum()
                        ),
                        "recovery_window_count": int(len(local)),
                        "median_recovery_hours": float(
                            local["recovery_hours"].median()
                        ),
                        "mean_recovery_hours": float(
                            local["recovery_hours"].mean()
                        ),
                        "q25_recovery_hours": float(
                            local["recovery_hours"].quantile(0.25)
                        ),
                        "q75_recovery_hours": float(
                            local["recovery_hours"].quantile(0.75)
                        ),
                        "zero_recovery_window_share": float(
                            local["zero_recovery_window"].mean()
                        ),
                        "median_available_recharge_mwh": float(
                            local["available_recharge_mwh"].median()
                        ),
                        "mean_available_recharge_mwh": float(
                            local["available_recharge_mwh"].mean()
                        ),
                        "median_available_recharge_share_of_annual": float(
                            local[
                                "available_recharge_share_of_annual"
                            ].median()
                        ),
                        "mean_available_recharge_share_of_annual": float(
                            local[
                                "available_recharge_share_of_annual"
                            ].mean()
                        ),
                        "median_positive_recovery_hours": float(
                            local.loc[
                                local["recovery_hours"].gt(0),
                                "recovery_hours",
                            ].median()
                        ),
                        "median_positive_window_recharge_mwh": float(
                            local.loc[
                                local["recovery_hours"].gt(0),
                                "available_recharge_mwh",
                            ].median()
                        ),
                    }
                )
                conservation_rows.append(
                    {
                        "country": country,
                        "year": int(year),
                        "target_cluster_days": int(cluster_days),
                        "chronology": chronology,
                        "base_task_error_mwh": float(
                            base_task.sum() - daily["base_task_mwh"].sum()
                        ),
                        "tail_task_error_mwh": float(
                            tail_task.sum() - daily["tail_task_mwh"].sum()
                        ),
                        "recharge_error_mwh": float(
                            recharge.sum() - daily["recharge_mwh"].sum()
                        ),
                        "reported_longest_event_run_days": int(
                            stats[
                                f"{chronology}_longest_event_run_days"
                            ]
                        ),
                        "recomputed_longest_event_run_days": longest_run(flags),
                    }
                )

    windows = pd.DataFrame(window_rows)
    country_year = pd.DataFrame(country_year_rows)
    natural = pd.DataFrame(natural_rows)
    conservation = pd.DataFrame(conservation_rows)
    pair_keys = ["country", "year", "target_cluster_days"]
    metrics = [
        "median_recovery_hours",
        "mean_recovery_hours",
        "zero_recovery_window_share",
        "median_available_recharge_mwh",
        "mean_available_recharge_mwh",
        "median_available_recharge_share_of_annual",
        "mean_available_recharge_share_of_annual",
        "longest_event_run_days",
    ]
    wide = country_year.pivot(
        index=pair_keys, columns="chronology", values=metrics
    )
    wide.columns = [f"{metric}_{chronology}" for metric, chronology in wide]
    paired = wide.reset_index()
    for metric in metrics:
        paired[f"clustered_minus_dispersed_{metric}"] = (
            paired[f"{metric}_clustered"]
            - paired[f"{metric}_dispersed"]
        )

    overall_rows: list[dict[str, Any]] = []
    for cluster_days, subset in paired.groupby("target_cluster_days"):
        for metric in [
            "median_recovery_hours",
            "mean_recovery_hours",
            "zero_recovery_window_share",
            "median_available_recharge_mwh",
            "mean_available_recharge_mwh",
            "median_available_recharge_share_of_annual",
            "mean_available_recharge_share_of_annual",
            "longest_event_run_days",
        ]:
            effect = f"clustered_minus_dispersed_{metric}"
            point, low, high = country_block_bootstrap(
                subset, effect, args.bootstrap_draws, args.seed + int(cluster_days)
            )
            overall_rows.append(
                {
                    "target_cluster_days": int(cluster_days),
                    "metric": metric,
                    "dispersed_country_year_median": float(
                        subset[f"{metric}_dispersed"].median()
                    ),
                    "clustered_country_year_median": float(
                        subset[f"{metric}_clustered"].median()
                    ),
                    "paired_mean_difference": point,
                    "country_block_bootstrap_ci_low": low,
                    "country_block_bootstrap_ci_high": high,
                    "country_year_count": int(len(subset)),
                    "country_count": int(subset["country"].nunique()),
                }
            )
    overall = pd.DataFrame(overall_rows)

    windows.to_csv(
        args.output / "recovery_windows.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    country_year.to_csv(args.output / "recovery_window_country_year.csv", index=False)
    paired.to_csv(args.output / "recovery_window_paired_effects.csv", index=False)
    overall.to_csv(args.output / "recovery_window_overall_summary.csv", index=False)
    natural.to_csv(args.output / "natural_event_geometry.csv", index=False)
    conservation.to_csv(args.output / "chronology_reconstruction_qa.csv", index=False)

    qa = {
        "status": "pass",
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256(args.panel),
        "runner": str(RUNNER.resolve()),
        "runner_sha256": sha256(RUNNER),
        "country_year_count": int(natural.shape[0]),
        "country_count": int(natural["country"].nunique()),
        "cluster_days": cluster_days_values,
        "expected_window_count": int(
            natural["event_days"].sub(1).sum()
            * 2
            * len(cluster_days_values)
        ),
        "actual_window_count": int(len(windows)),
        "maximum_absolute_base_task_conservation_error_mwh": float(
            conservation["base_task_error_mwh"].abs().max()
        ),
        "maximum_absolute_tail_task_conservation_error_mwh": float(
            conservation["tail_task_error_mwh"].abs().max()
        ),
        "maximum_absolute_recharge_conservation_error_mwh": float(
            conservation["recharge_error_mwh"].abs().max()
        ),
        "maximum_longest_run_reconstruction_difference_days": int(
            (
                conservation["reported_longest_event_run_days"]
                - conservation["recomputed_longest_event_run_days"]
            )
            .abs()
            .max()
        ),
    }
    if qa["expected_window_count"] != qa["actual_window_count"]:
        qa["status"] = "fail"
    if qa["maximum_longest_run_reconstruction_difference_days"] != 0:
        qa["status"] = "fail"
    (args.output / "qa_summary.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if qa["status"] != "pass":
        raise RuntimeError(f"Recovery-window QA failed: {qa}")


if __name__ == "__main__":
    main()
