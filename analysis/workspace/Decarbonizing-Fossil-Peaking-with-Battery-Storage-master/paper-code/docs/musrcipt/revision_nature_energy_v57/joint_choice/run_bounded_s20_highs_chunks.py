#!/usr/bin/env python3
"""Launch four disjoint HiGHS bounded-power country-year chunks."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_bounded_s20_highs.py"
PYTHON = Path(__import__("sys").executable)
OUTPUT_ROOT = HERE / "bounded_s20_highs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-year", type=int, choices=[2025, 2030, 2035], required=True)
    parser.add_argument("--shards", type=int, default=24)
    args = parser.parse_args()
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    processes = []
    for chunk in range(1, args.shards + 1):
        label = f"shard{chunk:02d}_of_{args.shards:02d}"
        output = OUTPUT_ROOT / f"cost_{args.cost_year}" / label
        output.mkdir(parents=True, exist_ok=True)
        log = (output / "run.log").open("wb")
        command = [
            str(PYTHON), str(RUNNER), "--mode", "production",
            "--cost-year", str(args.cost_year),
            "--shard-index", str(chunk), "--shard-count", str(args.shards),
            "--output", str(output),
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        processes.append((chunk, process, log))
        print(f"started cost {args.cost_year} {label}: pid={process.pid}", flush=True)
    failed = []
    for chunk, process, log in processes:
        returncode = process.wait()
        log.close()
        print(f"finished cost {args.cost_year} shard {chunk}: returncode={returncode}", flush=True)
        if returncode:
            failed.append(chunk)
    if failed:
        raise SystemExit(f"bounded S20 HiGHS chunks failed: {failed}")


if __name__ == "__main__":
    main()
