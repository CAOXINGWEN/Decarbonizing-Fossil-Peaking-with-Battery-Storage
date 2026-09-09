#!/usr/bin/env python3
"""Solve sequencing parity across 75 PECD v4.2 July--June operating years."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V56 = HERE.parent
LEGACY_DIR = V56 / "weather_year_parity"
sys.path.insert(0, str(LEGACY_DIR))

import compute_pecd_35y_parity as storage_model  # noqa: E402
import compute_pecd_europe_weather_parity as legacy_weather  # noqa: E402


PROCESSED = HERE / "processed"
PREPARED = HERE / "prepared"
RESULTS = HERE / "results"
CHECKPOINTS = RESULTS / "checkpoints"
PANEL = PROCESSED / "pecd_v42_europe33_hourly_1950_2025.parquet"
COST_FILE = V56 / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"

WINDOW_START_YEARS = np.arange(1950, 2025, dtype=int)
HOURS = 8760
DAYS = 365
SCOPE = "Europe33_copperplate"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def window_label(start: int) -> str:
    return f"{start}/{str(start + 1)[-2:]}"


def combine_windows(frame: pd.DataFrame, value_column: str) -> dict[int, np.ndarray]:
    arrays = {}
    for year, group in frame.groupby("calendar_year", sort=True):
        group = group.sort_values("timestamp_utc")
        if len(group) != HOURS:
            raise RuntimeError(f"Calendar year {year} has {len(group)} rows")
        arrays[int(year)] = group[value_column].to_numpy(float)
    timestamp = pd.DatetimeIndex(frame.timestamp_utc)
    months = {
        int(year): timestamp[frame.calendar_year.to_numpy(int) == year].month.to_numpy(np.int8)
        for year in range(1950, 2026)
    }
    windows = {}
    for start in WINDOW_START_YEARS:
        left = months[int(start)] >= 7
        right = months[int(start + 1)] <= 6
        output = np.concatenate([arrays[int(start)][left], arrays[int(start + 1)][right]])
        if len(output) != HOURS:
            raise RuntimeError(f"Operating year {window_label(int(start))} has {len(output)} hours")
        windows[int(start)] = output
    return windows


def window_months(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    result = {}
    frame = frame.sort_values("timestamp_utc")
    for start in WINDOW_START_YEARS:
        mask = (
            ((frame.calendar_year == start) & (frame.timestamp_utc.dt.month >= 7))
            | ((frame.calendar_year == start + 1) & (frame.timestamp_utc.dt.month <= 6))
        )
        month = frame.loc[mask, "timestamp_utc"].dt.month.to_numpy(np.int8)
        if len(month) != HOURS:
            raise RuntimeError(f"Month vector for {window_label(int(start))} has {len(month)} hours")
        result[int(start)] = month
    return result


def prepare() -> None:
    if not PANEL.exists():
        raise FileNotFoundError(f"Hourly PECD v4.2 panel missing: {PANEL}")
    PREPARED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(PANEL)
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    demand = combine_windows(frame, "demand_mw")
    generation = combine_windows(frame, "vre_unscaled_mw")
    months = window_months(frame)

    raw: dict[int, dict[str, Any]] = {}
    for start in WINDOW_START_YEARS:
        load = demand[int(start)]
        vre = generation[int(start)]
        alpha = float(load.sum() / vre.sum())
        residual = load - alpha * vre
        residual[np.abs(residual) < 1.0e-10] = 0.0
        task = np.maximum(residual, 0.0)
        recharge = np.maximum(-residual, 0.0)
        if np.any((task > 0.0) & (recharge > 0.0)):
            raise RuntimeError(f"Task/recharge overlap in {window_label(int(start))}")
        raw[int(start)] = {
            "task": task,
            "recharge": recharge,
            "demand_mwh": float(load.sum()),
            "vre_unscaled_mwh": float(vre.sum()),
            "vre_scaling_alpha": alpha,
        }

    target_task = float(np.median([item["task"].sum() for item in raw.values()]))
    target_recharge = float(np.median([item["recharge"].sum() for item in raw.values()]))
    fixed_power = storage_model.FORCED_POWER_SHARE * target_task / HOURS
    if min(target_task, target_recharge, fixed_power) <= 0.0:
        raise RuntimeError("Invalid standardized task scale")

    base_stack = []
    tail_stack = []
    recharge_stack = []
    event_stack = []
    rows = []
    for index, start in enumerate(WINDOW_START_YEARS, start=1):
        current = raw[int(start)]
        task_scale = target_task / current["task"].sum()
        recharge_scale = target_recharge / current["recharge"].sum()
        standardized_task = current["task"] * task_scale
        standardized_recharge = current["recharge"] * recharge_scale
        base_task, tail_task, event, metrics = legacy_weather.derive_tasks(
            standardized_task,
            standardized_recharge,
            months[int(start)],
        )
        base_stack.append(base_task)
        tail_stack.append(tail_task)
        recharge_stack.append(standardized_recharge)
        event_stack.append(event.astype(np.int8))
        rows.append({
            "scope": SCOPE,
            "window_start_year": int(start),
            "weather_window": window_label(int(start)),
            "source": "PECD v4.2 ERA5 historical",
            "system_definition": "synchronized_Europe33_fixed_2030_mix_annual_vre_equals_demand",
            "demand_model": "actual_load_temperature_calendar_model_ERAA2025_2030_geographic_weights",
            "standardization": "75_window_median_raw_task_and_recharge",
            "demand_mwh": current["demand_mwh"],
            "unscaled_vre_mwh": current["vre_unscaled_mwh"],
            "vre_scaling_alpha": current["vre_scaling_alpha"],
            "raw_task_mwh": float(current["task"].sum()),
            "raw_recharge_mwh": float(current["recharge"].sum()),
            "task_standardization_factor": task_scale,
            "recharge_standardization_factor": recharge_scale,
            "target_raw_task_mwh": target_task,
            "target_raw_recharge_mwh": target_recharge,
            "fixed_common_power_mw": fixed_power,
            **metrics,
        })
        print(f"prepared {index}/75 {window_label(int(start))}", flush=True)

    prepared_path = PREPARED / "Europe33_pecd_v42_1950_2025_75_windows.npz"
    np.savez_compressed(
        prepared_path,
        window_start_years=WINDOW_START_YEARS,
        base_task_mw=np.asarray(base_stack),
        tail_task_mw=np.asarray(tail_stack),
        recharge_mw=np.asarray(recharge_stack),
        event_day=np.asarray(event_stack),
        fixed_common_power_mw=np.asarray([fixed_power]),
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "pecd_v42_75_window_prepared_summary.csv", index=False)
    geometry_columns = [
        "scope", "window_start_year", "weather_window", "fixed_common_power_mw",
        "event_days", "natural_longest_event_run_days", "natural_zero_recovery_window_share",
        "natural_median_recovery_window_h", "annual_base_task_mwh", "annual_tail_task_mwh",
        "annual_recharge_mwh", "recharge_to_base_task_ratio",
    ]
    summary[geometry_columns].to_csv(RESULTS / "pecd_v42_75_window_event_geometry.csv", index=False)
    qa = {
        "status": "pass",
        "source_calendar_years": 76,
        "complete_july_june_operating_years": 75,
        "first_operating_year": "1950/51",
        "last_operating_year": "2024/25",
        "all_event_day_counts_equal_92": bool(summary.event_days.eq(92).all()),
        "all_fixed_power_positive": bool(summary.fixed_common_power_mw.gt(0.0).all()),
        "prepared_sha256": storage_model.sha256(prepared_path),
    }
    if not qa["all_event_day_counts_equal_92"] or not qa["all_fixed_power_positive"]:
        qa["status"] = "fail"
    write_json(RESULTS / "pecd_v42_75_window_preparation_qa.json", qa)
    if qa["status"] != "pass":
        raise RuntimeError("PECD v4.2 preparation QA failed")


def solve_chunk(indices: list[int]) -> list[dict[str, Any]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    base, caes, scenario, _ = storage_model.load_parameters()
    models = {
        base.config_id: storage_model.build_reduced_model(base, base, scenario),
        caes.config_id: storage_model.build_reduced_model(base, caes, scenario),
    }
    data = np.load(PREPARED / "Europe33_pecd_v42_1950_2025_75_windows.npz")
    starts = data["window_start_years"].astype(int)
    power = float(data["fixed_common_power_mw"][0])
    rows = []
    for index in indices:
        start = int(starts[index])
        checkpoint = CHECKPOINTS / f"window_{start}.csv.gz"
        if checkpoint.exists():
            cached = pd.read_csv(checkpoint)
            if len(cached) == 2 and cached.solver_status_name.eq("optimal").all():
                rows.extend(cached.to_dict("records"))
                continue
        task = data["base_task_mw"][index].astype(float, copy=False)
        tail_task = data["tail_task_mw"][index].astype(float, copy=False)
        recharge = data["recharge_mw"][index].astype(float, copy=False)
        current = []
        for config_id in (base.config_id, caes.config_id):
            answer = storage_model.solve_arm(models[config_id], task, tail_task, recharge, power)
            current.append({
                "scope": SCOPE,
                "window_start_year": start,
                "weather_window": window_label(start),
                "chronology": "natural_pecd_v42_july_june",
                "annual_scale": "75_window_median_standardized",
                "scenario_id": 20,
                "energy_value_eur_mwh": scenario.energy_value_eur_mwh,
                "firm_value_eur_kw_year": scenario.firm_value_eur_kw_year,
                "charge_price_eur_mwh": scenario.charge_price_eur_mwh,
                **answer,
            })
        storage_model.output_csv_gz(pd.DataFrame(current), checkpoint)
        rows.extend(current)
        print(f"solved {window_label(start)}", flush=True)
    return rows


def solve(workers: int) -> None:
    prepared = PREPARED / "Europe33_pecd_v42_1950_2025_75_windows.npz"
    if not prepared.exists():
        raise FileNotFoundError("Prepared v4.2 input is missing")
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    chunks = [list(map(int, part)) for part in np.array_split(np.arange(75), min(workers, 75)) if len(part)]
    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as executor:
        futures = [executor.submit(solve_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            rows.extend(future.result())
    output = pd.DataFrame(rows).sort_values(["window_start_year", "config_id"]).reset_index(drop=True)
    output_path = RESULTS / "pecd_v42_75_window_arm_results.csv.gz"
    storage_model.output_csv_gz(output, output_path)
    qa = {
        "status": "pass",
        "rows": int(len(output)),
        "expected_rows": 150,
        "all_optimal": bool(output.solver_status_name.eq("optimal").all()),
        "maximum_equality_residual": float(output.maximum_equality_residual.max()),
        "maximum_inequality_violation": float(output.maximum_inequality_violation.max()),
        "maximum_bound_violation": float(output.maximum_bound_violation.max()),
        "minimum_unserved_mw": float(output.minimum_unserved_mw.min()),
        "total_solver_runtime_seconds": float(output.solver_runtime_seconds.sum()),
        "workers": min(workers, len(chunks)),
        "output_sha256": storage_model.sha256(output_path),
    }
    if (
        len(output) != 150
        or not qa["all_optimal"]
        or qa["maximum_equality_residual"] > 1.0e-5
        or qa["maximum_inequality_violation"] > 1.0e-5
        or qa["minimum_unserved_mw"] < -1.0e-5
    ):
        qa["status"] = "fail"
    write_json(RESULTS / "pecd_v42_75_window_optimization_qa.json", qa)
    if qa["status"] != "pass":
        raise RuntimeError("PECD v4.2 optimization QA failed")


def cost_path() -> pd.DataFrame:
    frame = pd.read_csv(COST_FILE)
    selected = frame[frame.config_id.isin(["li_lfp_4h", "caes_24h"])].copy()
    wide = selected.pivot(
        index="learning_year",
        columns="config_id",
        values="annual_fixed_cost_eur_source_aware_per_kw_discharge_year",
    ).reset_index()
    wide["lfp4_minus_caes24_fixed_eur_kw_year"] = wide.li_lfp_4h - wide.caes_24h
    anchor = float(wide.loc[wide.learning_year.eq(2025), "lfp4_minus_caes24_fixed_eur_kw_year"].iloc[0])
    wide["relative_value_shift_from_2025_eur_kw_year"] = wide.lfp4_minus_caes24_fixed_eur_kw_year - anchor
    return wide.sort_values("learning_year").reset_index(drop=True)


def summarize() -> None:
    arms = pd.read_csv(RESULTS / "pecd_v42_75_window_arm_results.csv.gz")
    geometry = pd.read_csv(RESULTS / "pecd_v42_75_window_event_geometry.csv")
    wide = arms.pivot(
        index=["scope", "window_start_year", "weather_window", "fixed_common_power_mw"],
        columns="config_id",
        values=["system_cost_eur", "tail_equivalent_full_cycles", "portfolio_efc_mw", "unserved_eens_mwh"],
    )
    wide.columns = [f"{metric}_{config}" for metric, config in wide.columns]
    values = wide.reset_index().sort_values("window_start_year")
    values["value_2025_eur_kw_year"] = (
        values.system_cost_eur_li_lfp_4h - values.system_cost_eur_caes_24h
    ) / (1000.0 * values.fixed_common_power_mw)
    values = values.merge(
        geometry,
        on=["scope", "window_start_year", "weather_window", "fixed_common_power_mw"],
        validate="one_to_one",
    )
    values.to_csv(RESULTS / "pecd_v42_75_window_relative_values_2025.csv", index=False)

    costs = cost_path()
    costs.to_csv(RESULTS / "pecd_v42_focal_cost_path_2025_2035.csv", index=False)
    long_rows = []
    for _, cost in costs.iterrows():
        current = values.copy()
        current["cost_year"] = int(cost.learning_year)
        current["lfp4_fixed_cost_eur_kw_year"] = float(cost.li_lfp_4h)
        current["caes24_fixed_cost_eur_kw_year"] = float(cost.caes_24h)
        current["relative_value_shift_from_2025_eur_kw_year"] = float(cost.relative_value_shift_from_2025_eur_kw_year)
        current["caes24_minus_lfp4_eur_kw_year"] = current.value_2025_eur_kw_year + float(
            cost.relative_value_shift_from_2025_eur_kw_year
        )
        current["caes24_outperforms_lfp4"] = current.caes24_minus_lfp4_eur_kw_year >= 0.0
        long_rows.append(current)
    by_cost = pd.concat(long_rows, ignore_index=True)
    by_cost.to_csv(RESULTS / "pecd_v42_75_window_relative_values_by_cost_year.csv", index=False)

    value = values.value_2025_eur_kw_year.to_numpy(float)
    labels = values.weather_window.astype(str).to_numpy()
    years = values.window_start_year.to_numpy(int)
    threshold = float(
        costs.loc[costs.learning_year.eq(2035), "relative_value_shift_from_2025_eur_kw_year"].iloc[0]
        - costs.loc[costs.learning_year.eq(2025), "relative_value_shift_from_2025_eur_kw_year"].iloc[0]
    )
    left, right = np.triu_indices(len(value), 1)
    pair_delta = value[right] - value[left]
    pairs = pd.DataFrame({
        "start_window": labels[left],
        "end_window": labels[right],
        "start_year": years[left],
        "end_year": years[right],
        "weather_value_change_eur_kw_year": pair_delta,
        "absolute_weather_value_change_eur_kw_year": np.abs(pair_delta),
        "reaches_2025_2035_cost_scale": np.abs(pair_delta) >= threshold - 1.0e-12,
    })
    storage_model.output_csv_gz(pairs, RESULTS / "pecd_v42_75_window_all_pairs.csv.gz")

    decade_delta = value[10:] - value[:-10]
    decade = pd.DataFrame({
        "start_window": labels[:-10],
        "end_window": labels[10:],
        "start_year": years[:-10],
        "end_year": years[10:],
        "weather_value_change_eur_kw_year": decade_delta,
        "erases_2025_2035_learning": decade_delta <= -threshold + 1.0e-12,
        "exceeds_learning_beneficially": decade_delta >= threshold - 1.0e-12,
        "start_value_at_2025_cost_eur_kw_year": value[:-10],
        "end_value_at_2035_cost_eur_kw_year": value[10:] + threshold,
        "net_2025_to_2035_value_change_eur_kw_year": decade_delta + threshold,
    })
    decade.to_csv(RESULTS / "pecd_v42_65_matched_decade_transitions.csv", index=False)

    cost_summary = by_cost.groupby("cost_year", as_index=False).agg(
        weather_windows=("weather_window", "size"),
        value_min_eur_kw_year=("caes24_minus_lfp4_eur_kw_year", "min"),
        value_median_eur_kw_year=("caes24_minus_lfp4_eur_kw_year", "median"),
        value_max_eur_kw_year=("caes24_minus_lfp4_eur_kw_year", "max"),
        weather_windows_caes24_outperforms_lfp4=("caes24_outperforms_lfp4", "sum"),
    )
    cost_summary.to_csv(RESULTS / "pecd_v42_75_window_cost_year_summary.csv", index=False)

    worst_index = int(np.argmin(decade_delta))
    best_index = int(np.argmax(decade_delta))
    summary = {
        "status": "pass",
        "source": "PECD v4.2 ERA5 historical, 1950-2025 calendar years",
        "complete_july_june_operating_years": 75,
        "all_unordered_weather_pairs": int(len(pairs)),
        "matched_ten_year_transitions": int(len(decade)),
        "cost_improvement_2025_2035_eur_kw_year": threshold,
        "natural_full_span_eur_kw_year": float(value.max() - value.min()),
        "natural_q95_q05_span_eur_kw_year": float(np.quantile(value, 0.95) - np.quantile(value, 0.05)),
        "minimum_value_window": labels[int(np.argmin(value))],
        "maximum_value_window": labels[int(np.argmax(value))],
        "all_pairs_reaching_cost_scale": int(pairs.reaches_2025_2035_cost_scale.sum()),
        "all_pair_fraction_reaching_cost_scale": float(pairs.reaches_2025_2035_cost_scale.mean()),
        "matched_decades_erasing_learning": int(decade.erases_2025_2035_learning.sum()),
        "matched_decade_fraction_erasing_learning": float(decade.erases_2025_2035_learning.mean()),
        "matched_decades_exceeding_learning_beneficially": int(decade.exceeds_learning_beneficially.sum()),
        "worst_matched_decade": {
            "start": labels[worst_index],
            "end": labels[worst_index + 10],
            "weather_change_eur_kw_year": float(decade_delta[worst_index]),
            "net_change_after_learning_eur_kw_year": float(decade_delta[worst_index] + threshold),
        },
        "best_matched_decade": {
            "start": labels[best_index],
            "end": labels[best_index + 10],
            "weather_change_eur_kw_year": float(decade_delta[best_index]),
            "net_change_after_learning_eur_kw_year": float(decade_delta[best_index] + threshold),
        },
        "economic_crossovers_by_cost_year": {
            str(int(row.cost_year)): int(row.weather_windows_caes24_outperforms_lfp4)
            for _, row in cost_summary.iterrows()
        },
    }
    write_json(RESULTS / "pecd_v42_75y_parity_summary.json", summary)

    correlation = float(np.corrcoef(
        values.natural_zero_recovery_window_share.to_numpy(float),
        values.value_2025_eur_kw_year.to_numpy(float),
    )[0, 1])
    geometry_ranges = {
        "longest_min": int(values.natural_longest_event_run_days.min()),
        "longest_max": int(values.natural_longest_event_run_days.max()),
        "zero_min": 100.0 * float(values.natural_zero_recovery_window_share.min()),
        "zero_max": 100.0 * float(values.natural_zero_recovery_window_share.max()),
    }
    cost_lookup = cost_summary.set_index("cost_year")

    def command(name: str, value: object) -> str:
        return rf"\newcommand{{\{name}}}{{{value}}}"

    macro_lines = [
        "% Generated by compute_pecd_v42_75y_parity.py; do not edit manually.",
        command("NatCalendarYears", "1950--2025"),
        command("NatFirstWindow", "1950/51"),
        command("NatLastWindow", "2024/25"),
        command("NatWindows", "75"),
        command("NatPairs", f"{len(pairs):,}"),
        command("NatDecades", f"{len(decade)}"),
        command("NatThreshold", f"{threshold:.3f}"),
        command("NatThresholdRounded", f"{threshold:.2f}"),
        command("NatFullSpan", f"{summary['natural_full_span_eur_kw_year']:.2f}"),
        command("NatCentralSpan", f"{summary['natural_q95_q05_span_eur_kw_year']:.2f}"),
        command("NatMinimumWindow", summary["minimum_value_window"]),
        command("NatMaximumWindow", summary["maximum_value_window"]),
        command("NatPairHits", f"{summary['all_pairs_reaching_cost_scale']:,}"),
        command("NatPairPercent", f"{100.0 * summary['all_pair_fraction_reaching_cost_scale']:.1f}"),
        command("NatDecadeHits", summary["matched_decades_erasing_learning"]),
        command("NatDecadePercent", f"{100.0 * summary['matched_decade_fraction_erasing_learning']:.1f}"),
        command("NatBeneficialDecadeHits", summary["matched_decades_exceeding_learning_beneficially"]),
        command("NatWorstStart", summary["worst_matched_decade"]["start"]),
        command("NatWorstEnd", summary["worst_matched_decade"]["end"]),
        command("NatWorstChange", f"{summary['worst_matched_decade']['weather_change_eur_kw_year']:.2f}"),
        command("NatWorstNet", f"{summary['worst_matched_decade']['net_change_after_learning_eur_kw_year']:.2f}"),
        command("NatBestStart", summary["best_matched_decade"]["start"]),
        command("NatBestEnd", summary["best_matched_decade"]["end"]),
        command("NatBestChange", f"{summary['best_matched_decade']['weather_change_eur_kw_year']:.2f}"),
        command("NatZeroWindowMin", f"{geometry_ranges['zero_min']:.1f}"),
        command("NatZeroWindowMax", f"{geometry_ranges['zero_max']:.1f}"),
        command("NatLongestRunMin", geometry_ranges["longest_min"]),
        command("NatLongestRunMax", geometry_ranges["longest_max"]),
        command("NatRecoveryCorrelation", f"{correlation:.2f}"),
    ]
    for year, suffix in [(2025, "TwentyFive"), (2030, "Thirty"), (2035, "ThirtyFive")]:
        row = cost_lookup.loc[year]
        macro_lines.extend([
            command(f"NatCross{suffix}", int(row.weather_windows_caes24_outperforms_lfp4)),
            command(f"NatValueMin{suffix}", f"{row.value_min_eur_kw_year:.2f}"),
            command(f"NatValueMedian{suffix}", f"{row.value_median_eur_kw_year:.2f}"),
            command(f"NatValueMax{suffix}", f"{row.value_max_eur_kw_year:.2f}"),
        ])
    (RESULTS / "pecd_v42_v56_results_macros.tex").write_text(
        "\n".join(macro_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "solve", "summarize", "all"], nargs="?", default="all")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.stage in {"prepare", "all"}:
        prepare()
    if args.stage in {"solve", "all"}:
        solve(max(1, args.workers))
    if args.stage in {"summarize", "all"}:
        summarize()


if __name__ == "__main__":
    main()
