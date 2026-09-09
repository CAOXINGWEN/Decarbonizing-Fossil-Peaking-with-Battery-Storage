#!/usr/bin/env python3
"""Run and merge three calendar-cost bounded-power procurement experiments."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V44 = HERE.parent
RUNNER = V44 / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"
PANEL = (
    V44.parents[4]
    / "dataset"
    / "external_hourly"
    / "europe_mid_duration_valley_upgrade_v0_1"
    / "processed"
    / "observed_gas_event_task_definitions_v0_2.csv.gz"
)
CATALOGUES = HERE / "catalogues"
ROOT = HERE / "optimized_procurement"
YEARS = (2025, 2030, 2035)
CHUNK_COUNT = 4
MAX_CONCURRENT_JOBS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def country_year_chunks() -> list[list[str]]:
    panel = pd.read_csv(PANEL, usecols=["country", "year"])
    pairs = (
        panel.drop_duplicates()
        .sort_values(["country", "year"])
        .apply(lambda row: f"{row.country}:{int(row.year)}", axis=1)
        .tolist()
    )
    if len(pairs) != 132:
        raise RuntimeError(f"Expected 132 country-years, obtained {len(pairs)}")
    return [list(values) for values in np.array_split(pairs, CHUNK_COUNT)]


def launch() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    chunks = country_year_chunks()
    env = os.environ.copy()
    env["GRB_LICENSE_FILE"] = "/etc/gurobi/gurobi.lic"
    pending: list[tuple[int, int, list[str], object, Path]] = []
    for year in YEARS:
        catalogue = CATALOGUES / f"configuration_catalog_53_source_aware_eur_{year}.csv"
        if not catalogue.exists():
            raise FileNotFoundError(catalogue)
        for number, pairs in enumerate(chunks, start=1):
            output = ROOT / str(year) / "chunks" / f"chunk{number}"
            output.mkdir(parents=True, exist_ok=True)
            qa = output / "qa_summary.json"
            if qa.exists() and json.loads(qa.read_text()).get("status") == "pass":
                print(f"skip completed {year} chunk{number}", flush=True)
                continue
            log_path = output / "runner.log"
            log = log_path.open("wb")
            command = [
                __import__("sys").executable,
                str(RUNNER),
                "--panel",
                str(PANEL),
                "--catalogue",
                str(catalogue),
                "--scenario-id",
                "20",
                "--output",
                str(output),
                "--country-years",
                ",".join(pairs),
                "--chronologies",
                "dispersed,clustered",
                "--tail-power-mode",
                "optimized",
                "--threads",
                "1",
                "--method",
                "1",
            ]
            pending.append((year, number, command, log, log_path))

    jobs: list[tuple[int, int, subprocess.Popen[bytes], object, Path]] = []
    while pending or jobs:
        while pending and len(jobs) < MAX_CONCURRENT_JOBS:
            year, number, command, log, log_path = pending.pop(0)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            jobs.append((year, number, process, log, log_path))
            print(f"launched {year} chunk{number} pid={process.pid}", flush=True)
        remaining: list[tuple[int, int, subprocess.Popen[bytes], object, Path]] = []
        for year, number, process, log, log_path in jobs:
            code = process.poll()
            if code is None:
                remaining.append((year, number, process, log, log_path))
                continue
            log.close()
            if code != 0:
                raise RuntimeError(f"{year} chunk{number} failed; inspect {log_path}")
            print(f"completed {year} chunk{number}", flush=True)
        jobs = remaining
        if pending or jobs:
            print(f"waiting for {len(jobs)} active chunks; {len(pending)} queued", flush=True)
            time.sleep(20)


def merge_year(year: int) -> None:
    chunk_root = ROOT / str(year) / "chunks"
    chunks = [chunk_root / f"chunk{number}" for number in range(1, CHUNK_COUNT + 1)]
    output = ROOT / str(year) / "merged"
    output.mkdir(parents=True, exist_ok=True)

    def load(filename: str) -> pd.DataFrame:
        paths = [chunk / filename for chunk in chunks]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(missing)
        return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    arm = load("portfolio_arm_results.csv.gz")
    comparison = load("ldes_vs_extra_lfp4_results.csv.gz")
    paired = load("clustered_vs_dispersed_paired_effects.csv")
    chronology = load("chronology_conservation.csv")
    checks = [
        (arm, ["country", "year", "chronology", "config_id"], 132 * 2 * 51),
        (comparison, ["country", "year", "chronology", "config_id"], 132 * 2 * 49),
        (paired, ["country", "year", "config_id"], 132 * 51),
        (chronology, ["country", "year", "chronology"], 132 * 2),
    ]
    for frame, keys, expected in checks:
        if len(frame) != expected or frame.duplicated(keys).any():
            raise RuntimeError(f"Merge gate failed for {keys}: {len(frame)} rows")
    if not arm["solver_status_name"].eq("optimal").all():
        raise RuntimeError(f"Non-optimal solve detected in {year}")

    paths = {
        "portfolio_arm_results.csv.gz": arm,
        "ldes_vs_extra_lfp4_results.csv.gz": comparison,
    }
    for name, frame in paths.items():
        csv_gz(frame.sort_values(["country", "year", "chronology", "config_id"]), output / name)
    paired.sort_values(["country", "year", "config_id"]).to_csv(
        output / "clustered_vs_dispersed_paired_effects.csv", index=False
    )
    chronology.sort_values(["country", "year", "chronology"]).to_csv(
        output / "chronology_conservation.csv", index=False
    )

    contracts = [json.loads((chunk / "run_contract.json").read_text()) for chunk in chunks]
    merged_contract = dict(contracts[0])
    merged_contract["model_cost_year"] = year
    merged_contract["country_years"] = (
        arm[["country", "year"]]
        .drop_duplicates()
        .sort_values(["country", "year"])
        .astype({"year": int})
        .values.tolist()
    )
    merged_contract["merged_from_chunks"] = [str(path.resolve()) for path in chunks]
    (output / "run_contract.json").write_text(
        json.dumps(merged_contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    residual = (
        arm["system_cost_eur"]
        - arm["residual_energy_cost_eur"]
        - arm["residual_firm_cost_eur"]
        - arm["storage_cost_eur"]
    ).abs()
    qa = {
        "status": "pass",
        "model_cost_year": year,
        "country_year_count": 132,
        "arm_row_count": len(arm),
        "comparison_row_count": len(comparison),
        "all_arm_statuses_optimal": True,
        "maximum_task_balance_residual_mw": float(arm["maximum_task_balance_residual_mw"].abs().max()),
        "maximum_shared_recharge_violation_mw": float(arm["maximum_shared_recharge_violation_mw"].clip(lower=0).max()),
        "maximum_cyclic_soc_residual_mwh": float(
            max(arm["base_soc_cyclic_residual_mwh"].abs().max(), arm["tail_soc_cyclic_residual_mwh"].abs().max())
        ),
        "maximum_accounting_identity_residual_eur": float(residual.max()),
        "output_sha256": {
            path.name: sha256(path)
            for path in output.iterdir()
            if path.is_file() and path.name not in {"qa_summary.json"}
        },
    }
    (output / "qa_summary.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False), flush=True)


def main() -> None:
    launch()
    for year in YEARS:
        merge_year(year)


if __name__ == "__main__":
    main()
