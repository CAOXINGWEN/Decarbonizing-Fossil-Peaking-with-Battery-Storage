#!/usr/bin/env python3
"""Run the 132 country-years in four disjoint S20 bounded-power chunks.

This is a v44-only production runner.  It reuses the frozen v43 country-year
partition and the audited portfolio solver, but writes every output beneath
``revision_nature_energy_v44/joint_choice``.  Four single-threaded Gurobi
processes are used so that the experiment is reproducible without changing
the optimization formulation.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
V44 = HERE.parent
V43 = V44.parent / "revision_nature_energy_v43"
RUNNER = V44 / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"
PARTITIONS = (
    V43 / "portfolio_model" / "full_132cy_fixed_s20_central_chunks"
)
OUTPUT_ROOT = HERE / "optimized_s20_chunks"


def country_year_argument(chunk: int) -> str:
    contract = json.loads(
        (PARTITIONS / f"chunk{chunk}" / "run_contract.json").read_text()
    )
    return ",".join(f"{country}:{year}" for country, year in contract["country_years"])


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("GRB_LICENSE_FILE", "/etc/gurobi/gurobi.lic")
    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []

    for chunk in range(1, 5):
        output = OUTPUT_ROOT / f"chunk{chunk}"
        output.mkdir(parents=True, exist_ok=True)
        log = (output / "run.log").open("wb")
        command = [
            __import__("sys").executable,
            str(RUNNER),
            "--country-years",
            country_year_argument(chunk),
            "--scenario-id",
            "20",
            "--tail-power-mode",
            "optimized",
            "--tail-power-upper-multiplier",
            "1.0",
            "--threads",
            "1",
            "--method",
            "1",
            "--output",
            str(output),
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        processes.append((chunk, process, log))
        print(f"started chunk{chunk}: pid={process.pid}", flush=True)

    failures: list[int] = []
    for chunk, process, log in processes:
        returncode = process.wait()
        log.close()
        print(f"finished chunk{chunk}: returncode={returncode}", flush=True)
        if returncode != 0:
            failures.append(chunk)
    if failures:
        raise SystemExit(f"S20 bounded-power chunks failed: {failures}")


if __name__ == "__main__":
    main()
