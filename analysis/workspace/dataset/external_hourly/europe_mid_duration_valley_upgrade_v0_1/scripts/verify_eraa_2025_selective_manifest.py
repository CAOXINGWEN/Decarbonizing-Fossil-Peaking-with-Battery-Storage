#!/usr/bin/env python3
"""Independently verify the selectively extracted ERAA 2025 member manifest.

The extractor verifies each member while writing it.  This script is a second,
read-only pass over the final CSV and every decompressed output.  It recomputes
byte size, CRC32 and SHA-256, checks the frozen member inventory, and emits a
machine-readable report.  Schema and node-join checks remain the responsibility
of ``build_eraa_2035_phase1_inputs.py --audit-only``.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[4]
DATASET_ROOT = SCRIPT.parents[1]
DEFAULT_MANIFEST = (
    DATASET_ROOT
    / "artifacts"
    / "research"
    / "ERAA_2025_SELECTIVE_2035_MEMBER_MANIFEST.csv"
)
DEFAULT_OUTPUT_ROOT = (
    DATASET_ROOT / "raw" / "eraa_2025" / "selective_2035"
)
DEFAULT_REPORT = (
    DATASET_ROOT
    / "artifacts"
    / "research"
    / "ERAA_2025_SELECTIVE_2035_INTEGRITY_QA.json"
)

ARCHIVE_SIZES = {
    "ERAA_2025_PECDWeather.zip": 2_144_730_060,
    "ERAA_2025_DemandData.zip": 442_714_786,
    "ERAA_2025_PECDRES.zip": 1_274_578_535,
}
BASE_GROUP_COUNTS = {
    "weather_mapping": 1,
    "demand_total": 52,
    "capacity_factor": 309,
    "hydro_constraint": 41,
    "hydro_inflow": 105,
}
EXTENDED_GROUP_COUNTS = {
    **BASE_GROUP_COUNTS,
    "btm_pv": 4,
    "btm_battery": 4,
}
GOOD_STATUSES = {"downloaded_verified", "verified_existing"}


def classify(member: str) -> str:
    if member == "PECD - weather/WeatherScenarios_Mapping.xlsx":
        return "weather_mapping"
    if (
        member.startswith("Demand data/Demand timeseries/")
        and member.endswith("_Demand_total_2035_National Trends.csv")
    ):
        return "demand_total"
    if (
        member.startswith(
            "Demand data/Behind the meter PV-Battery system additions/PV/"
        )
        and member.endswith("_PV_2035_National Trends.csv")
    ):
        return "btm_pv"
    if (
        member.startswith(
            "Demand data/Behind the meter PV-Battery system additions/"
            "Batteries/"
        )
        and member.endswith("_Batteries_2035_National Trends.csv")
    ):
        return "btm_battery"
    if (
        member.startswith("PECD - RES/Capacity Factors_250716/")
        and member.endswith("_2035.csv")
    ):
        return "capacity_factor"
    if (
        member.startswith("PECD - RES/Hydro Constraints_250716/")
        and member.endswith("_2035.csv")
    ):
        return "hydro_constraint"
    if (
        member.startswith("PECD - RES/Hydro Inflows_250704/")
        and member.endswith("_2035.csv")
    ):
        return "hydro_inflow"
    return "unexpected"


def resolve_output(text: str, output_root: Path) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"output path escapes selective root: {text}"
        ) from exc
    return resolved


def digest_file(path: Path) -> tuple[int, str, str]:
    size = 0
    crc = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            size += len(block)
            crc = binascii.crc32(block, crc)
            digest.update(block)
    return size, f"{crc & 0xFFFFFFFF:08x}", digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify(
    manifest: Path,
    output_root: Path,
    expected_mode: str,
) -> dict[str, Any]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    contains_btm = any(
        "Behind the meter PV-Battery system additions" in row["member_path"]
        for row in rows
    )
    if expected_mode == "auto":
        expected_mode = "extended516" if contains_btm else "base508"
    expected_groups = (
        EXTENDED_GROUP_COUNTS
        if expected_mode == "extended516"
        else BASE_GROUP_COUNTS
    )
    expected_total = sum(expected_groups.values())

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    group_counts = {name: 0 for name in expected_groups}
    archive_counts: dict[str, int] = {}
    keys: set[tuple[str, str]] = set()
    outputs: set[Path] = set()
    total_uncompressed_bytes = 0

    for index, row in enumerate(rows, start=2):
        archive = row.get("archive_name", "")
        member = row.get("member_path", "")
        key = (archive, member)
        if key in keys:
            errors.append(
                {"row": str(index), "error": f"duplicate member key {key}"}
            )
        keys.add(key)

        member_path = PurePosixPath(member)
        if member_path.is_absolute() or ".." in member_path.parts:
            errors.append(
                {"row": str(index), "error": f"unsafe member path {member}"}
            )

        group = classify(member)
        if group not in expected_groups:
            errors.append(
                {
                    "row": str(index),
                    "error": f"unexpected member group {group}: {member}",
                }
            )
        else:
            group_counts[group] += 1

        archive_counts[archive] = archive_counts.get(archive, 0) + 1
        expected_archive_size = ARCHIVE_SIZES.get(archive)
        try:
            observed_archive_size = int(row.get("archive_size", ""))
        except ValueError:
            observed_archive_size = -1
        if expected_archive_size != observed_archive_size:
            errors.append(
                {
                    "row": str(index),
                    "error": (
                        f"archive size mismatch for {archive}: "
                        f"{observed_archive_size} != {expected_archive_size}"
                    ),
                }
            )
        if row.get("status") not in GOOD_STATUSES:
            errors.append(
                {
                    "row": str(index),
                    "error": f"non-verified status {row.get('status')!r}",
                }
            )
        if row.get("error"):
            errors.append(
                {
                    "row": str(index),
                    "error": f"extractor error field is nonempty: {row['error']}",
                }
            )
        if len(row.get("compressed_sha256", "")) != 64:
            errors.append(
                {
                    "row": str(index),
                    "error": "compressed SHA-256 is absent or malformed",
                }
            )

        try:
            output = resolve_output(row.get("output_path", ""), output_root)
        except ValueError as exc:
            errors.append({"row": str(index), "error": str(exc)})
            continue
        if output in outputs:
            errors.append(
                {
                    "row": str(index),
                    "error": f"duplicate output path {output}",
                }
            )
        outputs.add(output)
        if not output.is_file():
            errors.append(
                {
                    "row": str(index),
                    "error": f"output is absent: {output}",
                }
            )
            continue
        observed_size, observed_crc, observed_sha = digest_file(output)
        total_uncompressed_bytes += observed_size
        try:
            expected_size = int(row.get("uncompressed_size", ""))
        except ValueError:
            expected_size = -1
        comparisons = {
            "size": (str(observed_size), str(expected_size)),
            "crc32": (
                observed_crc.lower(),
                row.get("member_crc32", "").lower(),
            ),
            "sha256": (
                observed_sha.lower(),
                row.get("sha256", "").lower(),
            ),
        }
        for field, (observed, expected) in comparisons.items():
            if observed != expected:
                errors.append(
                    {
                        "row": str(index),
                        "error": (
                            f"{field} mismatch for {member}: "
                            f"{observed} != {expected}"
                        ),
                    }
                )

    if len(rows) != expected_total:
        errors.append(
            {
                "row": "manifest",
                "error": f"row count {len(rows)} != {expected_total}",
            }
        )
    if group_counts != expected_groups:
        errors.append(
            {
                "row": "manifest",
                "error": (
                    f"group counts {group_counts} != {expected_groups}"
                ),
            }
        )
    if expected_mode == "base508" and contains_btm:
        warnings.append(
            {
                "row": "manifest",
                "warning": "BTM members exist but base508 mode was requested",
            }
        )

    return {
        "audit": "eraa_2025_selective_2035_integrity",
        "manifest": str(manifest),
        "output_root": str(output_root),
        "expected_mode": expected_mode,
        "expected_total": expected_total,
        "observed_rows": len(rows),
        "unique_member_keys": len(keys),
        "unique_output_paths": len(outputs),
        "group_counts": group_counts,
        "archive_counts": archive_counts,
        "total_uncompressed_bytes_rehashed": total_uncompressed_bytes,
        "errors": errors,
        "warnings": warnings,
        "status": "pass" if not errors else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--expected-mode",
        choices=("auto", "base508", "extended516"),
        default="auto",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = verify(
        args.manifest.resolve(),
        args.output_root.resolve(),
        args.expected_mode,
    )
    write_json_atomic(args.json_output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
