#!/usr/bin/env python3
"""Admit standardized ERAA 2035 shards and freeze the phase-1 task graph.

This utility is intentionally separate from the optimization solver.  It
verifies that every selected zone contains the same 36 weather scenarios and
8,760-hour axis, selects the preregistered critical durations, and writes
deterministic outer/scenario task registries.  It never labels a task registry
as a model result.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
TARGET_YEAR = 2035
WEATHER_SCENARIOS = tuple(f"WS{number:02d}" for number in range(1, 37))
CRITICAL_DURATIONS_H = (4.0, 8.0, 24.0, 48.0, 72.0, 168.0)
CHRONOLOGIES = ("dispersed", "clustered")
ESTIMANDS = ("joint", "turnover")
OUTAGE_PATHS = tuple(range(1, 16))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "processed" / "eraa_2035_phase1_inputs_v0_1"
)
DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "model_adapter"
    / "energy_charts_2015_2024_representative_pilot"
    / "configuration_catalog_53.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "eraa_2035_phase1_task_graph_v0_1"

# Physical CORE bidding-zone baseline.  LUG1 is retained as ERAA's separate
# Luxembourg demand node.  This is a development network case, not the final
# flow-based European confirmation.
CORE_COPPERPLATE_ZONES = (
    "AT00",
    "BE00",
    "CZ00",
    "DE00",
    "FR00",
    "HR00",
    "HU00",
    "LUG1",
    "NL00",
    "PL00",
    "RO00",
    "SI00",
    "SK00",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def task_id(fields: Iterable[object]) -> str:
    text = "__".join(str(value) for value in fields)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, lineterminator="\n")
    os.replace(temporary, path)


def prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"output exists: {path}; pass --force to replace it"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def parse_values(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        result.update(
            part.strip() for part in value.split(",") if part.strip()
        )
    return result


def audit_hourly_shard(path: Path, expected_zone: str) -> dict[str, Any]:
    required = {
        "target_year",
        "weather_scenario",
        "hour_of_year",
        "zone",
        "load_mw",
        "wind_potential_mw",
        "solar_potential_mw",
    }
    header = pd.read_csv(path, nrows=0)
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    row_count = 0
    scenarios: dict[str, set[int]] = {}
    zones: set[str] = set()
    target_years: set[int] = set()
    minima = {
        "load_mw": np.inf,
        "wind_potential_mw": np.inf,
        "solar_potential_mw": np.inf,
    }
    for chunk in pd.read_csv(
        path,
        usecols=sorted(required),
        chunksize=100_000,
    ):
        row_count += len(chunk)
        zones.update(chunk["zone"].astype(str).unique())
        target_years.update(
            pd.to_numeric(chunk["target_year"], errors="raise")
            .astype(int)
            .unique()
        )
        for column in minima:
            values = pd.to_numeric(chunk[column], errors="raise")
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: {column} has non-finite values")
            minima[column] = min(minima[column], float(values.min()))
        for weather, group in chunk.groupby(
            "weather_scenario", sort=False
        ):
            scenarios.setdefault(str(weather), set()).update(
                pd.to_numeric(
                    group["hour_of_year"], errors="raise"
                ).astype(int)
            )

    expected_hours = set(range(8_760))
    synchronized = (
        set(scenarios) == set(WEATHER_SCENARIOS)
        and all(hours == expected_hours for hours in scenarios.values())
    )
    return {
        "zone": expected_zone,
        "path": str(path),
        "rows": row_count,
        "target_years": "|".join(str(value) for value in sorted(target_years)),
        "observed_zones": "|".join(sorted(zones)),
        "weather_scenarios": len(scenarios),
        "minimum_hours": min((len(value) for value in scenarios.values()), default=0),
        "maximum_hours": max((len(value) for value in scenarios.values()), default=0),
        "synchronized_36x8760": synchronized,
        "nonnegative_core_profiles": all(
            value >= 0 for value in minima.values()
        ),
        "sha256": sha256_file(path),
        "pass": bool(
            row_count == 36 * 8_760
            and zones == {expected_zone}
            and target_years == {TARGET_YEAR}
            and synchronized
            and all(value >= 0 for value in minima.values())
        ),
    }


def select_zones(
    input_root: Path,
    network_case: str,
    requested: set[str] | None,
) -> list[str]:
    coverage = pd.read_csv(input_root / "zone_coverage.csv")
    written = set(
        coverage.loc[
            coverage["hourly_shard_written"].astype(bool), "zone"
        ].astype(str)
    )
    if requested is not None:
        zones = sorted(requested)
    elif network_case == "core13-copperplate":
        zones = list(CORE_COPPERPLATE_ZONES)
    else:
        zones = sorted(written)
    missing = sorted(set(zones).difference(written))
    if missing:
        raise ValueError(
            f"selected zones lack standardized hourly shards: {missing}"
        )
    return zones


def select_critical_configs(catalog_path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path)
    required = {
        "config_id",
        "tech_id",
        "duration_h",
        "structural_extrapolation",
    }
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise ValueError(f"{catalog_path}: missing columns {missing}")
    duration = pd.to_numeric(catalog["duration_h"], errors="raise")
    structural = catalog["structural_extrapolation"].astype(bool)
    selected = catalog.loc[
        duration.isin(CRITICAL_DURATIONS_H) & ~structural
    ].copy()
    observed = set(
        pd.to_numeric(selected["duration_h"], errors="raise").astype(float)
    )
    missing_durations = sorted(
        set(CRITICAL_DURATIONS_H).difference(observed)
    )
    if missing_durations:
        raise ValueError(
            f"configuration catalogue misses critical durations "
            f"{missing_durations}"
        )
    return selected.sort_values(
        ["duration_h", "tech_id", "config_id"]
    ).reset_index(drop=True)


def build_outer_tasks(
    configs: pd.DataFrame,
    network_case: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chronology in CHRONOLOGIES:
        fields = (
            TARGET_YEAR,
            network_case,
            chronology,
            "baseline",
            "baseline",
        )
        rows.append(
            {
                "outer_task_id": task_id(fields),
                "target_year": TARGET_YEAR,
                "network_case": network_case,
                "chronology": chronology,
                "config_id": "baseline",
                "tech_id": "none",
                "duration_h": 0.0,
                "estimand": "baseline",
                "reuse_baseline": False,
                "status": "pending",
            }
        )
        for config in configs.itertuples(index=False):
            for estimand in ESTIMANDS:
                fields = (
                    TARGET_YEAR,
                    network_case,
                    chronology,
                    config.config_id,
                    estimand,
                )
                rows.append(
                    {
                        "outer_task_id": task_id(fields),
                        "target_year": TARGET_YEAR,
                        "network_case": network_case,
                        "chronology": chronology,
                        "config_id": config.config_id,
                        "tech_id": config.tech_id,
                        "duration_h": float(config.duration_h),
                        "estimand": estimand,
                        "reuse_baseline": True,
                        "status": "pending",
                    }
                )
    return pd.DataFrame(rows)


def build_scenario_tasks(outer: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in outer.itertuples(index=False):
        for weather in WEATHER_SCENARIOS:
            for outage_path in OUTAGE_PATHS:
                fields = (
                    task.outer_task_id,
                    weather,
                    f"FOS{outage_path:02d}",
                )
                rows.append(
                    {
                        "scenario_task_id": task_id(fields),
                        "outer_task_id": task.outer_task_id,
                        "target_year": task.target_year,
                        "network_case": task.network_case,
                        "chronology": task.chronology,
                        "config_id": task.config_id,
                        "estimand": task.estimand,
                        "weather_scenario": weather,
                        "forced_outage_path": f"FOS{outage_path:02d}",
                        "status": "pending",
                    }
                )
    return pd.DataFrame(rows)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input.resolve()
    catalog_path = args.configuration_catalog.resolve()
    output = args.output.resolve()
    if not (input_root / "QA.json").exists():
        raise FileNotFoundError(input_root / "QA.json")
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)
    prepare_output(output, args.force)

    requested = parse_values(args.zones)
    zones = select_zones(input_root, args.network_case, requested)
    audit_rows: list[dict[str, Any]] = []
    for zone in zones:
        path = (
            input_root
            / "hourly_profiles"
            / f"zone={zone}"
            / "hourly_profiles.csv.gz"
        )
        audit_rows.append(audit_hourly_shard(path, zone))
    audit = pd.DataFrame(audit_rows).sort_values("zone").reset_index(drop=True)
    write_csv(audit, output / "selected_zone_input_audit.csv")
    if not audit["pass"].all():
        raise RuntimeError("one or more standardized input shards failed")

    configs = select_critical_configs(catalog_path)
    write_csv(configs, output / "critical_configuration_catalog.csv")
    outer = build_outer_tasks(configs, args.network_case)
    scenarios = build_scenario_tasks(outer)
    write_csv(outer, output / "outer_tasks.csv")
    write_csv(scenarios, output / "scenario_tasks.csv")

    metrics = {
        "dispatch_outputs": [
            "system_cost_eur",
            "EENS_mwh",
            "LOLE_h",
            "firm_capacity_addition_mw",
            "equivalent_full_cycles_output_basis",
            "renewable_spill_mwh",
            "gas_generation_mwh_by_ccgt_ocgt",
            "hydro_generation_and_stock",
            "physical_import_export_mwh",
            "charge_discharge_and_soc_recovery",
            "balance_dual_charging_opportunity_cost_eur_mwh",
        ],
        "reliability_search": {
            "metric": "ELCC_mw",
            "rule": (
                "increase synchronous load by bisection; both EENS and LOLE "
                "must remain no worse than the chronology-specific baseline"
            ),
            "tolerance": "1e-4 of selected-region mean load",
        },
        "boundary_adjudication": {
            "h_T": (
                "rightmost node of the connected short-duration branch with "
                "positive turnover-value lower bound, followed by at least "
                "two adjacent negative joint-value upper bounds"
            ),
            "h_I": (
                "first stable post-gap node whose joint, adequacy and ELCC "
                "lower bounds are positive and confirmed by the next tested "
                "node"
            ),
            "W": "ln(h_I / h_T)",
            "DeltaW": "W_clustered - W_dispersed",
            "phase1_limitation": (
                "The six critical durations screen mechanisms and solver "
                "closure. Final h_T/h_I/W requires the complete 50-config "
                "4-168 h catalogue, not interpolation between these nodes."
            ),
        },
    }
    write_json(output / "metric_and_boundary_contract.json", metrics)
    input_qa = json.loads((input_root / "QA.json").read_text(encoding="utf-8"))
    result = {
        "script_version": SCRIPT_VERSION,
        "evidence_status": "task_graph_only_no_model_results",
        "input_root": str(input_root),
        "input_release_eligible": bool(
            input_qa.get("release_eligible", False)
        ),
        "network_case": args.network_case,
        "zones": zones,
        "zone_count": len(zones),
        "critical_durations_h": list(CRITICAL_DURATIONS_H),
        "critical_configuration_count": len(configs),
        "outer_task_count": len(outer),
        "scenario_task_count": len(scenarios),
        "synchronized_weather_scenarios": 36,
        "forced_outage_paths": 15,
        "admission_pass": True,
        "network_interpretation": (
            "Copperplate is a preregistered phase-1 development baseline. "
            "It cannot confirm deployable European h_I or DeltaW; final "
            "evidence uses the connected flow-based/NTC network."
        ),
        "input_shard_hashes": dict(
            zip(audit["zone"], audit["sha256"])
        ),
        "configuration_catalog_sha256": sha256_file(catalog_path),
    }
    write_json(output / "run_manifest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--configuration-catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--network-case",
        choices=(
            "core13-copperplate",
            "all-zones-copperplate",
            "full-flowbased-ntc",
        ),
        default="core13-copperplate",
    )
    parser.add_argument(
        "--zones",
        nargs="*",
        help="explicit development-zone subset, space or comma separated",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = execute(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
