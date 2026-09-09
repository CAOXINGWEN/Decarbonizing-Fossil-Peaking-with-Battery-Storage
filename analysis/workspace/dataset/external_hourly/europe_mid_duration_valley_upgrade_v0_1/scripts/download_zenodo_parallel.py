#!/usr/bin/env python3
"""Download one public Zenodo file with parallel HTTP range requests.

The script writes independent range chunks, checks their byte counts, joins
them atomically, and verifies an expected MD5 checksum when supplied.  It is
intended for immutable public research data whose size and checksum have
already been obtained from the Zenodo records API.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def md5sum(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def fetch_chunk(
    url: str,
    chunk_path: Path,
    start: int,
    end: int,
    retries: int,
) -> tuple[Path, int]:
    expected = end - start + 1
    if chunk_path.exists() and chunk_path.stat().st_size == expected:
        return chunk_path, expected
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        chunk_path.unlink(missing_ok=True)
        command = [
            "curl",
            "--http1.1",
            "-fsSL",
            "--retry",
            "2",
            "--retry-all-errors",
            "--retry-connrefused",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "30",
            "--max-time",
            "600",
            "--range",
            f"{start}-{end}",
            url,
            "-o",
            str(chunk_path),
        ]
        try:
            subprocess.run(command, check=True)
            observed = chunk_path.stat().st_size
            if observed != expected:
                raise RuntimeError(
                    f"Range {start}-{end} returned {observed:,} bytes; "
                    f"expected {expected:,}."
                )
            return chunk_path, observed
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(30, 2 * attempt))
    raise RuntimeError(
        f"Range {start}-{end} failed after {retries} attempts: {last_error}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument(
        "--remote-start",
        type=int,
        default=0,
        help=(
            "zero-based first byte in the remote object; the assembled "
            "output still starts at local byte zero"
        ),
    )
    parser.add_argument("--md5")
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument(
        "--chunk-mib",
        type=float,
        help=(
            "Persistent HTTP-range chunk size in MiB. When omitted, the "
            "legacy behaviour creates one chunk per connection."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Maximum concurrent range requests; defaults to --connections.",
    )
    parser.add_argument("--retries", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.remote_start < 0:
        raise ValueError("--remote-start must be non-negative")
    if args.connections <= 0:
        raise ValueError("--connections must be positive")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.chunk_mib is not None:
        if args.chunk_mib <= 0:
            raise ValueError("--chunk-mib must be positive")
        chunk_size = max(1, int(args.chunk_mib * 1024 * 1024))
        chunk_count = (args.size + chunk_size - 1) // chunk_size
    else:
        chunk_count = min(args.connections, args.size)
        chunk_size = (args.size + chunk_count - 1) // chunk_count
    max_workers = args.workers or args.connections
    if max_workers <= 0:
        raise ValueError("--workers must be positive")
    max_workers = min(max_workers, chunk_count)

    temp_dir = output.with_name(f".{output.name}.ranges")
    jobs_spec: list[dict[str, int | str]] = []
    for index in range(chunk_count):
        local_start = index * chunk_size
        if local_start >= args.size:
            break
        local_end = min(
            args.size - 1,
            local_start + chunk_size - 1,
        )
        start = args.remote_start + local_start
        end = args.remote_start + local_end
        jobs_spec.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "name": f"{index:04d}.part",
            }
        )
    receipt = {
        "url": args.url,
        "size": args.size,
        "remote_start": args.remote_start,
        "jobs": jobs_spec,
    }
    receipt_path = temp_dir / "ranges.json"
    if receipt_path.exists():
        try:
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = None
        if previous != receipt:
            shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    jobs = [
        (
            int(item["start"]),
            int(item["end"]),
            temp_dir / str(item["name"]),
        )
        for item in jobs_spec
    ]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = [
            executor.submit(
                fetch_chunk,
                args.url,
                chunk_path,
                start,
                end,
                args.retries,
            )
            for start, end, chunk_path in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    assembled = temp_dir / "assembled.download"
    with assembled.open("wb") as target:
        for _, _, chunk_path in jobs:
            with chunk_path.open("rb") as source:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())

    if assembled.stat().st_size != args.size:
        raise RuntimeError(
            f"Joined file has {assembled.stat().st_size:,} bytes; "
            f"expected {args.size:,}."
        )
    observed_md5 = md5sum(assembled)
    if args.md5 and observed_md5.lower() != args.md5.lower():
        raise RuntimeError(
            f"MD5 mismatch: observed {observed_md5}; expected {args.md5}."
        )

    os.replace(assembled, output)
    shutil.rmtree(temp_dir)

    print(f"downloaded={output}")
    print(f"bytes={output.stat().st_size}")
    print(f"md5={md5sum(output)}")


if __name__ == "__main__":
    main()
