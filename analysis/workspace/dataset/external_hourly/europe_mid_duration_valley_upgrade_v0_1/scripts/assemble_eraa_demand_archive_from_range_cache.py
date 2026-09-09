#!/usr/bin/env python3
"""Assemble the official ERAA demand ZIP from verified range caches plus a tail.

Earlier selective extraction retained complete, byte-exact HTTP ranges from
the official Azure object.  The two retained caches jointly cover a contiguous
prefix of the archive.  This utility reuses that prefix, appends a separately
downloaded final range, and then verifies the complete ZIP before atomically
replacing the destination.  It never treats a sparse/preallocated partial file
as a completed archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "eraa_2025"
EXPECTED_SIZE = 442_714_786
DEFAULT_TAIL_START = 359_661_568


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_ranges(cache: Path) -> list[tuple[int, int, Path]]:
    metadata = json.loads((cache / "ranges.json").read_text(encoding="utf-8"))
    ranges: list[tuple[int, int, Path]] = []
    for job in metadata["jobs"]:
        start = int(job["start"])
        end = int(job["end"])
        path = cache / str(job["name"])
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            ranges.append((start, end, path))
    return ranges


def copy_prefix(
    handle,
    caches: list[Path],
    prefix_end_exclusive: int,
) -> list[dict[str, object]]:
    ranges = [
        item
        for cache in caches
        for item in valid_ranges(cache)
    ]
    position = 0
    ledger: list[dict[str, object]] = []
    while position < prefix_end_exclusive:
        candidates = [
            item
            for item in ranges
            if item[0] <= position <= item[1]
        ]
        if not candidates:
            raise RuntimeError(
                f"verified range caches have a gap at byte {position:,}"
            )
        start, end, path = max(candidates, key=lambda item: item[1])
        copy_end = min(end + 1, prefix_end_exclusive)
        offset = position - start
        count = copy_end - position
        with path.open("rb") as source:
            source.seek(offset)
            remaining = count
            while remaining:
                block = source.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise RuntimeError(f"truncated verified range {path}")
                handle.write(block)
                remaining -= len(block)
        ledger.append(
            {
                "source": str(path),
                "archive_start": position,
                "archive_end": copy_end - 1,
                "bytes": count,
            }
        )
        position = copy_end
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        action="append",
        type=Path,
        default=None,
        help="verified range-cache directory; may be repeated",
    )
    parser.add_argument(
        "--tail",
        type=Path,
        default=Path("/tmp/ERAA_2025_DemandData.tail"),
    )
    parser.add_argument(
        "--tail-start",
        type=int,
        default=DEFAULT_TAIL_START,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW / "ERAA_2025_DemandData.zip",
    )
    parser.add_argument(
        "--qa-output",
        type=Path,
        default=RAW / "ERAA_2025_DemandData.full_archive_qa.json",
    )
    args = parser.parse_args()

    caches = args.cache or [
        RAW / ".ERAA_2025_DemandData.zip.ranges",
        RAW / ".ERAA_2025_DemandData_edge.zip.ranges",
    ]
    for cache in caches:
        if not (cache / "ranges.json").exists():
            raise FileNotFoundError(cache)
    expected_tail_size = EXPECTED_SIZE - args.tail_start
    if (
        not args.tail.exists()
        or args.tail.stat().st_size != expected_tail_size
    ):
        raise RuntimeError(
            "tail is absent or has the wrong size: "
            f"expected {expected_tail_size:,} bytes"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=args.output.parent,
        prefix=f".{args.output.name}.assemble.",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        ledger = copy_prefix(temporary, caches, args.tail_start)
        with args.tail.open("rb") as tail:
            for block in iter(lambda: tail.read(8 * 1024 * 1024), b""):
                temporary.write(block)

    try:
        if temporary_path.stat().st_size != EXPECTED_SIZE:
            raise RuntimeError("assembled archive has the wrong byte size")
        with zipfile.ZipFile(temporary_path) as archive:
            bad_member = archive.testzip()
            member_count = len(archive.infolist())
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failed for {bad_member}")
        archive_sha256 = sha256(temporary_path)
        os.replace(temporary_path, args.output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    qa = {
        "official_archive": "ERAA_2025_DemandData.zip",
        "expected_size_bytes": EXPECTED_SIZE,
        "observed_size_bytes": args.output.stat().st_size,
        "archive_sha256": archive_sha256,
        "zip_integrity_pass": True,
        "zip_member_count": member_count,
        "tail_start": args.tail_start,
        "tail_size_bytes": expected_tail_size,
        "tail_sha256": sha256(args.tail),
        "range_caches": [str(path) for path in caches],
        "prefix_copy_ledger": ledger,
        "pass": True,
    }
    args.qa_output.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
