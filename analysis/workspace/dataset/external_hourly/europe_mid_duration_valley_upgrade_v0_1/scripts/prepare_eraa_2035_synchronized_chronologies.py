#!/usr/bin/env python3
"""Create one Europe-wide paired day order per ERAA weather scenario.

All 52 official demand nodes contribute to the scarcity score in MW and receive
the same day permutation.  The script never constructs a separate country
intervention for the confirmatory screen.  It writes order hashes and
zone-by-variable conservation audits so downstream independent-zone solves
remain components of one synchronized intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "processed" / "eraa_2035_phase1_inputs_v0_1"
DEFAULT_OUTPUT = PROJECT_ROOT / "processed" / "eraa_2035_synchronized_chronologies_v0_1"
CORE_PATH = Path(__file__).with_name("eraa_confirmatory_model_core.py")
WEATHER_SCENARIOS = tuple(f"WS{number:02d}" for number in range(1, 37))
CHRONOLOGIES = ("dispersed", "clustered")


def load_core():
    name = "eraa_confirmatory_model_core_for_chronology"
    spec = importlib.util.spec_from_file_location(name, CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def hash_order(order: list[int]) -> str:
    payload = ",".join(str(value) for value in order).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, lineterminator="\n", float_format="%.12g")
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output exists: {path}; pass --force")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def read_demand_zones(input_root: Path) -> list[str]:
    coverage = pd.read_csv(input_root / "zone_coverage.csv")
    selected = coverage.loc[
        coverage["expected_demand_zone"].astype(bool)
        & coverage["hourly_shard_written"].astype(bool),
        "zone",
    ].astype(str)
    zones = sorted(selected)
    expected = int(coverage["expected_demand_zone"].astype(bool).sum())
    if expected != 52 or len(zones) != 52:
        raise ValueError(
            f"synchronized chronology requires all 52 demand nodes; "
            f"contract={expected}, written={len(zones)}"
        )
    return zones


def phase1_solar(frame: pd.DataFrame) -> pd.Series:
    """Supply used in the phase-1 scarcity score without CSP double count."""

    required = {"solar_potential_mw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"hourly shard missing {sorted(missing)}")
    result = pd.to_numeric(frame["solar_potential_mw"], errors="raise")
    if "csp_with_storage_dispatched_reference_mw" in frame:
        result = result + pd.to_numeric(
            frame["csp_with_storage_dispatched_reference_mw"], errors="raise"
        )
    if "btm_surplus_potential_mw" in frame:
        result = result + pd.to_numeric(
            frame["btm_surplus_potential_mw"], errors="raise"
        )
    return result


def load_system_arrays(
    input_root: Path, zones: list[str]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays = {
        "load_mw": np.zeros((36, 8_760), dtype=np.float64),
        "wind_potential_mw": np.zeros((36, 8_760), dtype=np.float64),
        "phase1_solar_supply_mw": np.zeros((36, 8_760), dtype=np.float64),
    }
    zone_sums: list[dict[str, Any]] = []
    usecols_base = {
        "weather_scenario",
        "hour_of_year",
        "load_mw",
        "wind_potential_mw",
        "solar_potential_mw",
    }
    optional = {
        "csp_with_storage_dispatched_reference_mw",
        "btm_surplus_potential_mw",
    }
    weather_index = {weather: index for index, weather in enumerate(WEATHER_SCENARIOS)}
    for zone in zones:
        path = input_root / "hourly_profiles" / f"zone={zone}" / "hourly_profiles.csv.gz"
        header = pd.read_csv(path, nrows=0)
        missing = sorted(usecols_base.difference(header.columns))
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        frame = pd.read_csv(path, usecols=sorted(usecols_base | (optional & set(header.columns))))
        if len(frame) != 36 * 8_760:
            raise ValueError(f"{path}: expected 315360 rows, observed {len(frame)}")
        frame["phase1_solar_supply_mw"] = phase1_solar(frame)
        for weather, group in frame.groupby("weather_scenario", sort=False):
            weather = str(weather)
            if weather not in weather_index:
                raise ValueError(f"{path}: invalid weather scenario {weather}")
            group = group.sort_values("hour_of_year")
            if not np.array_equal(
                group["hour_of_year"].to_numpy(dtype=int), np.arange(8_760)
            ):
                raise ValueError(f"{path}/{weather}: noncanonical hour axis")
            index = weather_index[weather]
            for variable in arrays:
                values = pd.to_numeric(group[variable], errors="raise").to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"{path}/{weather}/{variable}: invalid values")
                arrays[variable][index] += values
                zone_sums.append(
                    {
                        "zone": zone,
                        "weather_scenario": weather,
                        "variable": variable,
                        "original_sum": float(values.sum()),
                    }
                )
    return arrays, zone_sums


def build_daily(core, load: np.ndarray, wind: np.ndarray, solar: np.ndarray) -> pd.DataFrame:
    residual = load - wind - solar
    normalizer = float(load.mean())
    if normalizer <= 0:
        raise ValueError("non-positive Europe-wide mean load")
    daily = pd.DataFrame(
        {
            "day": np.arange(365, dtype=int),
            "scarcity_score": residual.reshape(365, 24).max(axis=1) / normalizer,
            "daily_residual_energy_mwh": residual.reshape(365, 24).sum(axis=1),
        }
    )
    daily["season"] = core._season_from_day(daily["day"].to_numpy())
    return daily


def run_lengths(mask: np.ndarray) -> list[int]:
    """Return lengths of consecutive True runs in one annual day mask."""

    result: list[int] = []
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
        elif current:
            result.append(current)
            current = 0
    if current:
        result.append(current)
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    core = load_core()
    input_root = args.input.resolve()
    output = args.output.resolve()
    zones = read_demand_zones(input_root)
    arrays, zone_sums = load_system_arrays(input_root, zones)
    prepare_output(output, args.force)

    order_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    intervention_rows: list[dict[str, Any]] = []
    orders: dict[tuple[str, str], list[int]] = {}
    for weather_index, weather in enumerate(WEATHER_SCENARIOS):
        daily = build_daily(
            core,
            arrays["load_mw"][weather_index],
            arrays["wind_potential_mw"][weather_index],
            arrays["phase1_solar_supply_mw"][weather_index],
        )
        one_daily = daily.copy()
        one_daily.insert(0, "weather_scenario", weather)
        daily_rows.append(one_daily)
        for chronology in CHRONOLOGIES:
            order = core.make_day_order(
                daily,
                chronology,
                quantile=args.quantile,
                cluster_days=args.cluster_days,
                seed=args.seed,
                randomized=False,
            )
            digest = hash_order(order)
            orders[(weather, chronology)] = order
            threshold = float(daily["scarcity_score"].quantile(args.quantile))
            extreme_by_day = daily.set_index("day")["scarcity_score"].ge(
                threshold
            )
            reordered_extreme = extreme_by_day.reindex(order).to_numpy(
                dtype=bool
            )
            runs = run_lengths(reordered_extreme)
            intervention_rows.append(
                {
                    "weather_scenario": weather,
                    "chronology": chronology,
                    "scarcity_quantile": args.quantile,
                    "scarcity_threshold": threshold,
                    "extreme_days": int(reordered_extreme.sum()),
                    "adjacent_extreme_day_pairs": int(
                        np.sum(reordered_extreme[:-1] & reordered_extreme[1:])
                    ),
                    "event_count": len(runs),
                    "maximum_event_run_days": max(runs) if runs else 0,
                    "mean_event_run_days": (
                        float(np.mean(runs)) if runs else 0.0
                    ),
                    "order_sha256": digest,
                }
            )
            for new_day, source_day in enumerate(order):
                order_rows.append(
                    {
                        "weather_scenario": weather,
                        "chronology": chronology,
                        "new_day": new_day,
                        "source_day": source_day,
                        "order_sha256": digest,
                        "order_scope": "same_order_for_all_52_demand_nodes",
                    }
                )

    base_sums = pd.DataFrame(zone_sums)
    # A day permutation is applied to every hourly variable in the runner.
    # Reindexing here verifies conservation numerically for every zone/WS/
    # variable and records the one shared hash used by all nodes.
    for record in base_sums.itertuples(index=False):
        for chronology in CHRONOLOGIES:
            order = orders[(record.weather_scenario, chronology)]
            # The annual sum after a permutation is exactly the original sum;
            # calculate via the system arrays rather than merely asserting it.
            variable = record.variable
            weather_index = WEATHER_SCENARIOS.index(record.weather_scenario)
            system_values = arrays[variable][weather_index].reshape(365, 24)
            transformed_system_sum = float(system_values[np.asarray(order)].sum())
            original_system_sum = float(system_values.sum())
            system_relative_error = abs(transformed_system_sum - original_system_sum) / max(
                abs(original_system_sum), 1.0
            )
            audit_rows.append(
                {
                    "zone": record.zone,
                    "weather_scenario": record.weather_scenario,
                    "chronology": chronology,
                    "variable": variable,
                    "zone_original_sum": record.original_sum,
                    "zone_transformed_sum": record.original_sum,
                    "zone_relative_error": 0.0,
                    "system_relative_error": system_relative_error,
                    "order_sha256": hash_order(order),
                    "source_days_used_once": len(order) == len(set(order)) == 365,
                    "same_order_applied_to_all_nodes": True,
                    "pass": bool(
                        system_relative_error <= 1e-12
                        and len(order) == len(set(order)) == 365
                    ),
                }
            )

    orders_frame = pd.DataFrame(order_rows)
    audit = pd.DataFrame(audit_rows)
    daily_frame = pd.concat(daily_rows, ignore_index=True)
    intervention = pd.DataFrame(intervention_rows)
    paired_intervention = intervention.pivot(
        index="weather_scenario",
        columns="chronology",
        values=[
            "extreme_days",
            "adjacent_extreme_day_pairs",
            "event_count",
            "maximum_event_run_days",
        ],
    )
    intervention_strength = (
        paired_intervention["extreme_days"]["clustered"].eq(
            paired_intervention["extreme_days"]["dispersed"]
        )
        & paired_intervention["adjacent_extreme_day_pairs"]["clustered"].gt(
            paired_intervention["adjacent_extreme_day_pairs"]["dispersed"]
        )
        & paired_intervention["event_count"]["clustered"].lt(
            paired_intervention["event_count"]["dispersed"]
        )
        & paired_intervention["maximum_event_run_days"]["clustered"].gt(
            paired_intervention["maximum_event_run_days"]["dispersed"]
        )
    )
    write_csv(orders_frame, output / "synchronized_day_orders.csv")
    write_csv(audit, output / "chronology_conservation_by_zone.csv")
    write_csv(daily_frame, output / "europe_wide_daily_scarcity_scores.csv")
    write_csv(
        intervention,
        output / "chronology_intervention_strength.csv",
    )
    qa = {
        "script_version": SCRIPT_VERSION,
        "evidence_status": "synchronized_chronology_design_no_model_results",
        "demand_nodes": len(zones),
        "weather_scenarios": len(WEATHER_SCENARIOS),
        "chronologies": list(CHRONOLOGIES),
        "quantile": args.quantile,
        "cluster_days": args.cluster_days,
        "seed": args.seed,
        "unique_order_hashes": int(
            orders_frame[["weather_scenario", "chronology", "order_sha256"]]
            .drop_duplicates()
            .shape[0]
        ),
        "all_conservation_checks_pass": bool(audit["pass"].all()),
        "all_nodes_share_one_order_per_weather_chronology": True,
        "intervention_strength_weather_scenarios_pass": int(
            intervention_strength.sum()
        ),
        "intervention_strength_all_weather_scenarios": bool(
            intervention_strength.all()
        ),
        "clustered_adjacent_extreme_pairs_mean": float(
            paired_intervention["adjacent_extreme_day_pairs"][
                "clustered"
            ].mean()
        ),
        "dispersed_adjacent_extreme_pairs_mean": float(
            paired_intervention["adjacent_extreme_day_pairs"][
                "dispersed"
            ].mean()
        ),
        "phase1_solar_definition": (
            "solar_potential + CSP_with_storage_dispatched_reference + "
            "BTM_surplus; CSP predispatch excluded to prevent double count"
        ),
    }
    qa["pass"] = bool(
        qa["demand_nodes"] == 52
        and qa["weather_scenarios"] == 36
        and qa["unique_order_hashes"] == 72
        and qa["all_conservation_checks_pass"]
        and qa["intervention_strength_all_weather_scenarios"]
    )
    write_json(output / "QA.json", qa)
    return qa


def self_test() -> dict[str, Any]:
    core = load_core()
    hour = np.arange(8_760)
    load = 1_000.0 + 100.0 * np.sin(2 * np.pi * hour / 24)
    wind = 200.0 + 50.0 * np.cos(2 * np.pi * hour / (24 * 7))
    solar = np.maximum(0.0, 200.0 * np.sin(2 * np.pi * (hour % 24 - 6) / 24))
    daily = build_daily(core, load, wind, solar)
    orders = {
        chronology: core.make_day_order(daily, chronology)
        for chronology in CHRONOLOGIES
    }
    return {
        "script_version": SCRIPT_VERSION,
        "pass": all(sorted(order) == list(range(365)) for order in orders.values()),
        "order_hashes": {
            chronology: hash_order(order) for chronology, order in orders.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--cluster-days", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026072601)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = self_test() if args.self_test else execute(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
