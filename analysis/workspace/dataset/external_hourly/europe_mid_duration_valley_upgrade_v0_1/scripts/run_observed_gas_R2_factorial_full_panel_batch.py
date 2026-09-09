#!/usr/bin/env python3
"""Run or resume the 132 country-year x 2 technology R2 factorial batch."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CELL_RUNNER = HERE / "run_observed_gas_R2_factorial_full_panel_cell.py"
CORE_RUNNER = (
    HERE / "run_observed_gas_independent_recharge_pilot.py"
)
FACTORIAL_HELPER = (
    HERE / "run_observed_gas_independent_recharge_factorial.py"
)
PANEL = (
    ROOT
    / "processed"
    / "energy_charts_observed_gas_task_panel_2015_2024.csv.gz"
)
CONFIGURATIONS = (
    ROOT.parents[2]
    / "Decarbonizing-Fossil-Peaking-with-Battery-Storage-master"
    / "paper-code"
    / "docs"
    / "musrcipt"
    / "revision_nature_energy_v36"
    / "system_value_closure"
    / "inputs"
    / "technology_duration_configurations_2035.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "model_adapter"
    / "observed_gas_R2_factorial_full132_v0_1"
)
CONFIGS = ("li_lfp_4h", "caes_24h")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def eligible_country_years() -> list[tuple[str, int]]:
    frame = pd.read_csv(
        PANEL,
        usecols=[
            "country",
            "year",
            "gas_task_eligible",
            "gas_task_material_share_ge_1pct",
        ],
    )
    selected = frame[
        as_bool(frame["gas_task_eligible"])
        & as_bool(frame["gas_task_material_share_ge_1pct"])
    ][["country", "year"]].drop_duplicates()
    selected = selected.sort_values(["country", "year"], kind="stable")
    pairs = [
        (str(row.country), int(row.year))
        for row in selected.itertuples(index=False)
    ]
    if len(pairs) != 132:
        raise RuntimeError(
            f"Frozen complete historical sample must have 132 country-years; "
            f"found {len(pairs)}"
        )
    return pairs


def task_key(country: str, year: int, config: str) -> str:
    return f"{country}_{year}_{config}"


def valid_completed(
    directory: Path,
    expected_hashes: dict[str, str],
) -> bool:
    qa_path = directory / "QA.json"
    manifest_path = directory / "run_manifest.json"
    if not qa_path.exists() or not manifest_path.exists():
        return False
    try:
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return False
    return bool(
        qa.get("status") == "pass"
        and all(bool(value) for value in qa.get("checks", {}).values())
        and qa.get("manuscript_evidence_admitted") is False
        and manifest.get("manuscript_evidence_admitted") is False
        and all(
            manifest.get(name) == value
            for name, value in expected_hashes.items()
        )
    )


def run_one(
    country: str,
    year: int,
    config: str,
    output_root: Path,
    expected_hashes: dict[str, str],
    force: bool,
) -> dict[str, Any]:
    key = task_key(country, year, config)
    directory = output_root / key
    log_directory = output_root / "_logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"{key}.log"
    if not force and valid_completed(directory, expected_hashes):
        qa = json.loads(
            (directory / "QA.json").read_text(encoding="utf-8")
        )
        return {
            "key": key,
            "country": country,
            "year": year,
            "config_id": config,
            "status": "skipped_complete",
            "returncode": 0,
            "wall_seconds": 0.0,
            "model_solve_seconds": qa.get("solve_seconds_total"),
            "output_directory": str(directory),
            "log": str(log_path),
        }

    command = [
        sys.executable,
        str(CELL_RUNNER),
        "--country",
        country,
        "--year",
        str(year),
        "--config-id",
        config,
        "--output",
        str(directory),
    ]
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    elapsed = time.perf_counter() - started
    qa: dict[str, Any] = {}
    qa_path = directory / "QA.json"
    if qa_path.exists():
        try:
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            qa = {}
    passed = bool(
        completed.returncode == 0
        and valid_completed(directory, expected_hashes)
    )
    return {
        "key": key,
        "country": country,
        "year": year,
        "config_id": config,
        "status": "pass" if passed else "fail",
        "returncode": int(completed.returncode),
        "wall_seconds": elapsed,
        "model_solve_seconds": qa.get("solve_seconds_total"),
        "output_directory": str(directory),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-country-years", type=int)
    parser.add_argument(
        "--country-year",
        action="append",
        help="Optional exact pair COUNTRY:YEAR; may be repeated.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("--workers must be between 1 and 4")

    all_pairs = eligible_country_years()
    if args.country_year:
        requested: list[tuple[str, int]] = []
        for item in args.country_year:
            country, raw_year = item.split(":", maxsplit=1)
            pair = (country, int(raw_year))
            if pair not in all_pairs:
                raise RuntimeError(f"Not in frozen sample: {pair}")
            requested.append(pair)
        selected_pairs = requested
    else:
        selected_pairs = all_pairs
    if args.limit_country_years is not None:
        selected_pairs = selected_pairs[: args.limit_country_years]

    tasks = [
        (country, year, config)
        for country, year in selected_pairs
        for config in CONFIGS
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    expected_hashes = {
        "script_sha256": sha256(CELL_RUNNER),
        "core_runner_sha256": sha256(CORE_RUNNER),
        "key_factorial_helper_sha256": sha256(FACTORIAL_HELPER),
        "panel_sha256": sha256(PANEL),
        "configuration_source_sha256": sha256(CONFIGURATIONS),
    }
    plan = {
        "panel": str(PANEL),
        "panel_sha256": sha256(PANEL),
        "cell_runner": str(CELL_RUNNER),
        "cell_runner_sha256": expected_hashes["script_sha256"],
        "core_runner_sha256": expected_hashes[
            "core_runner_sha256"
        ],
        "factorial_helper_sha256": expected_hashes[
            "key_factorial_helper_sha256"
        ],
        "configuration_source": str(CONFIGURATIONS),
        "configuration_source_sha256": expected_hashes[
            "configuration_source_sha256"
        ],
        "full_frozen_country_year_count": len(all_pairs),
        "selected_country_year_count": len(selected_pairs),
        "selected_task_count": len(tasks),
        "configurations": list(CONFIGS),
        "workers": args.workers,
        "force": args.force,
        "manuscript_evidence_admitted": False,
        "tasks": [
            {
                "country": country,
                "year": year,
                "config_id": config,
                "key": task_key(country, year, config),
            }
            for country, year, config in tasks
        ],
    }
    (args.output_root / "batch_plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(
                run_one,
                country,
                year,
                config,
                args.output_root,
                expected_hashes,
                args.force,
            ): (country, year, config)
            for country, year, config in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['status']:>16} {row['key']} "
                f"{row['wall_seconds']:.1f}s",
                flush=True,
            )
    rows_frame = pd.DataFrame(rows).sort_values(
        ["country", "year", "config_id"],
        kind="stable",
    )
    rows_frame.to_csv(
        args.output_root / "batch_execution_ledger.csv",
        index=False,
    )
    failed = rows_frame[rows_frame["status"].eq("fail")]
    summary = {
        "status": "pass" if failed.empty else "fail",
        "selected_country_year_count": len(selected_pairs),
        "selected_task_count": len(tasks),
        "passed_or_resumed_task_count": int(
            rows_frame["status"].isin(
                {"pass", "skipped_complete"}
            ).sum()
        ),
        "failed_task_count": int(len(failed)),
        "batch_wall_seconds": time.perf_counter() - started,
        "workers": args.workers,
        "manuscript_evidence_admitted": False,
        "failed_keys": failed["key"].tolist(),
    }
    (args.output_root / "batch_execution_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not failed.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
