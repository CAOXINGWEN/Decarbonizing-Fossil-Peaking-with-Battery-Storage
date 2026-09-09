#!/usr/bin/env python3
"""Launch restartable full-sample cluster-intensity surface shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_v60_cluster_intensity_surface.py"
CLUSTER_DAYS = ",".join(map(str, range(1, 13)))
SHARDS = {"Europe": 16, "China": 4, "Japan": 4}


def completed(output: Path) -> bool:
    qa_path = output / "qa.json"
    result_path = output / "system_duration_effects.csv.gz"
    if not qa_path.exists() or not result_path.exists():
        return False
    try:
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return qa.get("status") == "pass" and qa.get("cluster_days") == list(range(1, 13))


def run_task(task: tuple[str, int, int, Path]) -> tuple[str, int]:
    regime, shard_index, shard_count, runs_root = task
    slug = regime.lower()
    output = runs_root / f"{slug}_shard{shard_index:02d}"
    output.mkdir(parents=True, exist_ok=True)
    if completed(output):
        return f"SKIP {regime} {shard_index}/{shard_count}", 0
    command = [
        sys.executable,
        str(RUNNER),
        "--regime",
        regime,
        "--cluster-days",
        CLUSTER_DAYS,
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
        "--output",
        str(output),
    ]
    log_path = output / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    return f"DONE {regime} {shard_index}/{shard_count} exit={process.returncode}", process.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--runs-root", type=Path, default=HERE / "runs")
    args = parser.parse_args()
    args.runs_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (regime, shard_index, shard_count, args.runs_root)
        for regime, shard_count in SHARDS.items()
        for shard_index in range(1, shard_count + 1)
    ]
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            message, returncode = future.result()
            print(message, flush=True)
            if returncode != 0:
                failures.append(message)
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
