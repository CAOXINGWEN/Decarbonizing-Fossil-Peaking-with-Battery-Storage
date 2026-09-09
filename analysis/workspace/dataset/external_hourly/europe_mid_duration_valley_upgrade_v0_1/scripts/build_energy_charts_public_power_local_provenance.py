#!/usr/bin/env python3
"""Build an offline, current-file provenance inventory for Energy-Charts.

This program deliberately has no HTTP client and no network code path.  It
does not modify the historic download receipt or any raw source file.  The
result is evidence about files that are present *now*, not a reconstruction
of an HTTP transaction that was not recorded at download time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"
ENDPOINT = "public_power"
API_BASE = "https://api.energy-charts.info"
RAW_NAME_RE = re.compile(r"^public_power_([a-z]+)_([0-9]{4})\.json$")

EXPECTED_INPUT_SHA256 = {
    "historic_receipt": (
        "8172423556904ee0c2cb75bb8b6b0f0fe719f0fbce0ae7acfc6cede2b1c0c0b3"
    ),
    "source_manifest": (
        "112f0efa8bf8777522a7b16ad8fe340c2059e3d9b37538b1d73217fdfc2dd760"
    ),
    "country_year_qa": (
        "8cfc5d81e3aef56fd6d3b85b33ad53fa45a0861dd00fdbe07b9e97b513700092"
    ),
    "final_sample": (
        "cf1bbfae8cf1212ab3aeab2b70a5fc7cb9148c6f308310e385a7d7f043ff9ef8"
    ),
}

EXPECTED_COUNTS = {
    "raw_public_power_files": 170,
    "historic_public_power_receipts": 97,
    "missing_historic_public_power_receipts": 73,
    "final_sample_country_years": 132,
}

CSV_FIELDS = [
    "endpoint",
    "country",
    "country_api_code",
    "year",
    "final_132_included",
    "canonical_url",
    "local_path",
    "bytes",
    "sha256",
    "timestamp_count",
    "current_verified_utc",
    "filesystem_mtime_utc",
    "filesystem_mtime_ns",
    "filesystem_mtime_semantics",
    "historic_receipt_present",
    "historic_retrieved_utc",
    "historic_status",
    "historic_headers_json",
    "deprecated",
    "json_valid",
    "required_keys_present",
    "source_manifest_url_match",
    "source_manifest_path_match",
    "source_manifest_bytes_match",
    "source_manifest_sha256_match",
    "source_manifest_timestamp_count_match",
    "historic_receipt_url_match",
    "historic_receipt_path_match",
    "historic_receipt_bytes_match",
    "historic_receipt_sha256_match",
    "historic_receipt_timestamp_count_match",
]


def parse_args() -> argparse.Namespace:
    product_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline provenance manifest for current Energy-Charts "
            "public_power files without changing the historic receipt."
        )
    )
    parser.add_argument("--root", type=Path, default=product_root)
    parser.add_argument(
        "--historic-receipt",
        default="metadata/energy_charts_download_receipt.json",
    )
    parser.add_argument(
        "--source-manifest",
        default="metadata/energy_charts_panel_source_files.csv",
    )
    parser.add_argument(
        "--country-year-qa",
        default="artifacts/research/energy_charts_panel_country_year_qa.csv",
    )
    parser.add_argument(
        "--final-sample",
        default=(
            "artifacts/research/"
            "energy_charts_observed_gas_task_prespecified_sample.csv"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default="raw/energy_charts/public_power",
    )
    parser.add_argument(
        "--csv-output",
        default="metadata/energy_charts_public_power_local_provenance.csv",
    )
    parser.add_argument(
        "--json-output",
        default="metadata/energy_charts_public_power_local_provenance.json",
    )
    parser.add_argument(
        "--qa-output",
        default=(
            "artifacts/research/"
            "energy_charts_public_power_local_provenance_qa.json"
        ),
    )
    parser.add_argument(
        "--readme",
        default="metadata/ENERGY_CHARTS_PUBLIC_POWER_LOCAL_PROVENANCE_README.md",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def ns_to_utc(ns: int) -> str:
    seconds, nanoseconds = divmod(ns, 1_000_000_000)
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds))
    return f"{base}.{nanoseconds:09d}Z"


def canonical_url(api_code: str, year: int) -> str:
    return (
        f"{API_BASE}/{ENDPOINT}?country={api_code}"
        f"&start={year}-01-01&end={year}-12-31"
    )


def expected_local_path(api_code: str, year: int) -> str:
    return (
        "raw/energy_charts/public_power/"
        f"public_power_{api_code}_{year}.json"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"cannot parse Boolean value: {value!r}")


def normalized_manifest_path(path_text: str, product_root: Path) -> str:
    """Return the product-root-relative suffix stored by the source manifest."""
    normalized = Path(path_text).as_posix()
    marker = f"{product_root.name}/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def historic_headers(record: dict[str, Any]) -> Any:
    """Carry a recorded header object only; never synthesize one."""
    for field in ("headers", "http_headers", "response_headers"):
        if field in record:
            return record[field]
    return None


def stable_inventory_digest(records: Iterable[dict[str, Any]]) -> str:
    lines = [
        (
            f"{row['endpoint']}\t{row['country_api_code']}\t{row['year']}\t"
            f"{row['bytes']}\t{row['sha256']}\t{row['timestamp_count']}\n"
        )
        for row in sorted(
            records,
            key=lambda item: (
                item["endpoint"],
                item["country_api_code"],
                item["year"],
            ),
        )
    ]
    return sha256_text("".join(lines))


def rescan_raw_inventory(
    raw_by_key: dict[tuple[str, int], Path],
) -> tuple[str, bool]:
    """Hash every raw file again and detect changes during the second read."""
    records: list[dict[str, Any]] = []
    stable_within_rescan = True
    for api_code, year in sorted(raw_by_key):
        path = raw_by_key[(api_code, year)]
        before = path.stat()
        digest = sha256_file(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        after = path.stat()
        stable_within_rescan &= (
            before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
        )
        records.append(
            {
                "endpoint": ENDPOINT,
                "country_api_code": api_code,
                "year": year,
                "bytes": before.st_size,
                "sha256": digest,
                "timestamp_count": len(payload["unix_seconds"]),
            }
        )
    return stable_inventory_digest(records), stable_within_rescan


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = dict(record)
            headers = row.pop("historic_headers")
            row["historic_headers_json"] = (
                json.dumps(headers, ensure_ascii=False, sort_keys=True)
                if headers is not None
                else ""
            )
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    paths = {
        "historic_receipt": root / args.historic_receipt,
        "source_manifest": root / args.source_manifest,
        "country_year_qa": root / args.country_year_qa,
        "final_sample": root / args.final_sample,
        "raw_dir": root / args.raw_dir,
        "csv_output": root / args.csv_output,
        "json_output": root / args.json_output,
        "qa_output": root / args.qa_output,
        "readme": root / args.readme,
        "script": Path(__file__).resolve(),
    }

    for name in (
        "historic_receipt",
        "source_manifest",
        "country_year_qa",
        "final_sample",
        "raw_dir",
        "readme",
    ):
        if not paths[name].exists():
            raise FileNotFoundError(f"required input is missing: {paths[name]}")

    input_hashes = {
        name: sha256_file(paths[name])
        for name in (
            "historic_receipt",
            "source_manifest",
            "country_year_qa",
            "final_sample",
        )
    }
    input_hash_matches = {
        name: input_hashes[name] == expected
        for name, expected in EXPECTED_INPUT_SHA256.items()
    }
    if not all(input_hash_matches.values()):
        mismatch = {
            name: {
                "expected": EXPECTED_INPUT_SHA256[name],
                "observed": input_hashes[name],
            }
            for name, passed in input_hash_matches.items()
            if not passed
        }
        raise RuntimeError(
            "frozen input hash mismatch; re-audit before generating provenance: "
            + json.dumps(mismatch, sort_keys=True)
        )

    receipt_bytes_before = paths["historic_receipt"].read_bytes()
    receipt_sha_before = hashlib.sha256(receipt_bytes_before).hexdigest()
    receipt = json.loads(receipt_bytes_before)
    receipt_records = receipt.get("records")
    if not isinstance(receipt_records, list):
        raise ValueError("historic receipt does not contain a records list")

    historic_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in receipt_records:
        key = (
            str(record.get("endpoint")),
            str(record.get("country_api_code")),
            int(record.get("year")),
        )
        if key in historic_by_key:
            raise ValueError(f"duplicate historic receipt key: {key}")
        historic_by_key[key] = record
    historic_public = {
        key: value
        for key, value in historic_by_key.items()
        if key[0] == ENDPOINT
    }

    source_rows = read_csv(paths["source_manifest"])
    source_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in source_rows:
        key = (row["api_code"].strip().lower(), int(row["source_local_calendar_year"]))
        if key in source_by_key:
            raise ValueError(f"duplicate source-manifest key: {key}")
        source_by_key[key] = row

    qa_rows = read_csv(paths["country_year_qa"])
    qa_by_country_year: dict[tuple[str, int], dict[str, str]] = {}
    qa_by_api_year: dict[tuple[str, int], dict[str, str]] = {}
    for row in qa_rows:
        country_key = (row["country"].strip(), int(row["year"]))
        api_key = (row["api_code"].strip().lower(), int(row["year"]))
        if country_key in qa_by_country_year or api_key in qa_by_api_year:
            raise ValueError(f"duplicate country-year QA key: {country_key}/{api_key}")
        qa_by_country_year[country_key] = row
        qa_by_api_year[api_key] = row

    final_rows = read_csv(paths["final_sample"])
    selected_final_row_count = sum(
        parse_bool(row["main_prespecified_sample"]) for row in final_rows
    )
    final_country_years = {
        (row["country"].strip(), int(row["year"]))
        for row in final_rows
        if parse_bool(row["main_prespecified_sample"])
    }
    final_api_years: set[tuple[str, int]] = set()
    for country_year in final_country_years:
        if country_year not in qa_by_country_year:
            raise ValueError(f"final sample key absent from country-year QA: {country_year}")
        row = qa_by_country_year[country_year]
        final_api_years.add((row["api_code"].strip().lower(), int(row["year"])))

    raw_by_key: dict[tuple[str, int], Path] = {}
    for path in sorted(paths["raw_dir"].iterdir()):
        if not path.is_file():
            continue
        match = RAW_NAME_RE.fullmatch(path.name)
        if match is None:
            continue
        key = (match.group(1), int(match.group(2)))
        if key in raw_by_key:
            raise ValueError(f"duplicate raw-file key: {key}")
        raw_by_key[key] = path

    verification_time = utc_now()
    records: list[dict[str, Any]] = []
    raw_unchanged_during_verification = True
    for api_code, year in sorted(raw_by_key):
        raw_path = raw_by_key[(api_code, year)]
        stat_before = raw_path.stat()
        content_sha256 = sha256_file(raw_path)
        try:
            with raw_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            json_valid = True
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
            json_valid = False
        stat_after = raw_path.stat()
        raw_unchanged_during_verification &= (
            stat_before.st_size == stat_after.st_size
            and stat_before.st_mtime_ns == stat_after.st_mtime_ns
        )

        required_keys_present = (
            json_valid
            and all(
                key in payload
                for key in ("unix_seconds", "production_types", "deprecated")
            )
            and isinstance(payload.get("unix_seconds"), list)
            and len(payload.get("unix_seconds", [])) > 0
        )
        timestamp_count = (
            len(payload["unix_seconds"])
            if json_valid and isinstance(payload.get("unix_seconds"), list)
            else None
        )
        deprecated = payload.get("deprecated") if json_valid else None

        key = (api_code, year)
        source = source_by_key.get(key)
        qa = qa_by_api_year.get(key)
        historic = historic_public.get((ENDPOINT, api_code, year))
        local_path = expected_local_path(api_code, year)
        url = canonical_url(api_code, year)

        if source is None or qa is None:
            source_url_match = False
            source_path_match = False
            source_bytes_match = False
            source_sha_match = False
            source_timestamp_match = False
            country = None
        else:
            source_url_match = source["source_url"] == url
            source_path_match = (
                normalized_manifest_path(source["source_file"], root) == local_path
            )
            source_bytes_match = int(source["source_bytes"]) == stat_before.st_size
            source_sha_match = source["source_sha256"] == content_sha256
            source_timestamp_match = (
                int(source["raw_timestamps"]) == timestamp_count
            )
            country = qa["country"].strip()

        historic_present = historic is not None
        if historic_present:
            historic_url_match: bool | None = historic.get("url") == url
            historic_path_match: bool | None = (
                historic.get("local_path") == local_path
            )
            historic_bytes_match: bool | None = (
                historic.get("bytes") == stat_before.st_size
            )
            historic_sha_match: bool | None = (
                historic.get("sha256") == content_sha256
            )
            historic_timestamp_match: bool | None = (
                historic.get("n_timestamps") == timestamp_count
            )
            historic_retrieved_utc = historic.get("retrieved_utc")
            historic_status = historic.get("status")
            carried_headers = historic_headers(historic)
        else:
            historic_url_match = None
            historic_path_match = None
            historic_bytes_match = None
            historic_sha_match = None
            historic_timestamp_match = None
            historic_retrieved_utc = None
            historic_status = None
            carried_headers = None

        records.append(
            {
                "endpoint": ENDPOINT,
                "country": country,
                "country_api_code": api_code,
                "year": year,
                "final_132_included": key in final_api_years,
                "canonical_url": url,
                "local_path": local_path,
                "bytes": stat_before.st_size,
                "sha256": content_sha256,
                "timestamp_count": timestamp_count,
                "current_verified_utc": verification_time,
                "filesystem_mtime_utc": ns_to_utc(stat_before.st_mtime_ns),
                "filesystem_mtime_ns": stat_before.st_mtime_ns,
                "filesystem_mtime_semantics": (
                    "local filesystem metadata only; not a download or "
                    "retrieval time"
                ),
                "historic_receipt_present": historic_present,
                "historic_retrieved_utc": historic_retrieved_utc,
                "historic_status": historic_status,
                "historic_headers": carried_headers,
                "deprecated": deprecated,
                "json_valid": json_valid,
                "required_keys_present": required_keys_present,
                "source_manifest_url_match": source_url_match,
                "source_manifest_path_match": source_path_match,
                "source_manifest_bytes_match": source_bytes_match,
                "source_manifest_sha256_match": source_sha_match,
                "source_manifest_timestamp_count_match": source_timestamp_match,
                "historic_receipt_url_match": historic_url_match,
                "historic_receipt_path_match": historic_path_match,
                "historic_receipt_bytes_match": historic_bytes_match,
                "historic_receipt_sha256_match": historic_sha_match,
                "historic_receipt_timestamp_count_match": (
                    historic_timestamp_match
                ),
            }
        )

    receipt_sha_after_scan = sha256_file(paths["historic_receipt"])
    raw_inventory_digest = stable_inventory_digest(records)
    raw_inventory_digest_rescan, raw_stable_within_rescan = rescan_raw_inventory(
        raw_by_key
    )
    raw_inventory_stable_across_full_scan = (
        raw_inventory_digest == raw_inventory_digest_rescan
        and raw_unchanged_during_verification
        and raw_stable_within_rescan
    )
    historic_records = [row for row in records if row["historic_receipt_present"]]
    final_records = [row for row in records if row["final_132_included"]]

    all_source_checks = all(
        row[field]
        for row in records
        for field in (
            "source_manifest_url_match",
            "source_manifest_path_match",
            "source_manifest_bytes_match",
            "source_manifest_sha256_match",
            "source_manifest_timestamp_count_match",
        )
    )
    all_historic_checks = all(
        row[field]
        for row in historic_records
        for field in (
            "historic_receipt_url_match",
            "historic_receipt_path_match",
            "historic_receipt_bytes_match",
            "historic_receipt_sha256_match",
            "historic_receipt_timestamp_count_match",
        )
    )
    missing_history_null = all(
        row["historic_retrieved_utc"] is None
        and row["historic_status"] is None
        and row["historic_headers"] is None
        for row in records
        if not row["historic_receipt_present"]
    )
    final_keys_covered = { 
        (row["country_api_code"], row["year"]) for row in final_records
    } == final_api_years
    final_hash_consistent = all(
        row["source_manifest_sha256_match"] for row in final_records
    )
    final_file_valid = all(
        row["json_valid"] and row["required_keys_present"] for row in final_records
    )

    counts = {
        "raw_public_power_files": len(raw_by_key),
        "source_manifest_public_power_rows": len(source_by_key),
        "historic_public_power_receipts": len(historic_public),
        "missing_historic_public_power_receipts": (
            len(raw_by_key) - len(historic_records)
        ),
        "json_valid_files": sum(bool(row["json_valid"]) for row in records),
        "required_keys_present_files": sum(
            bool(row["required_keys_present"]) for row in records
        ),
        "source_manifest_sha256_matches": sum(
            bool(row["source_manifest_sha256_match"]) for row in records
        ),
        "historic_receipt_sha256_matches": sum(
            bool(row["historic_receipt_sha256_match"])
            for row in historic_records
        ),
        "final_sample_country_years": len(final_api_years),
        "final_sample_provenance_rows": len(final_records),
        "final_sample_sha256_matches": sum(
            bool(row["source_manifest_sha256_match"]) for row in final_records
        ),
    }

    gates = {
        "frozen_input_hashes_match": all(input_hash_matches.values()),
        "raw_public_power_file_count_is_170": (
            counts["raw_public_power_files"]
            == EXPECTED_COUNTS["raw_public_power_files"]
        ),
        "source_manifest_keys_equal_raw_keys": set(source_by_key) == set(raw_by_key),
        "country_year_qa_keys_equal_raw_keys": set(qa_by_api_year) == set(raw_by_key),
        "historic_public_power_receipt_count_is_97": (
            counts["historic_public_power_receipts"]
            == EXPECTED_COUNTS["historic_public_power_receipts"]
        ),
        "missing_historic_receipt_count_is_73": (
            counts["missing_historic_public_power_receipts"]
            == EXPECTED_COUNTS["missing_historic_public_power_receipts"]
        ),
        "all_raw_json_and_required_keys_valid": all(
            row["json_valid"] and row["required_keys_present"] for row in records
        ),
        "all_source_manifest_fields_match_current_files": all_source_checks,
        "all_historic_receipt_fields_match_current_files": all_historic_checks,
        "missing_history_fields_are_null": missing_history_null,
        "final_sample_key_count_is_132": (
            selected_final_row_count
            == EXPECTED_COUNTS["final_sample_country_years"]
            and len(final_api_years)
            == EXPECTED_COUNTS["final_sample_country_years"]
        ),
        "final_sample_provenance_coverage_is_132_of_132": (
            len(final_records)
            == EXPECTED_COUNTS["final_sample_country_years"]
            and final_keys_covered
        ),
        "final_sample_hash_consistency_is_132_of_132": final_hash_consistent,
        "final_sample_files_are_valid_132_of_132": final_file_valid,
        "raw_files_unchanged_during_verification": (
            raw_inventory_stable_across_full_scan
        ),
        "historic_receipt_unchanged_during_verification": (
            receipt_sha_before == receipt_sha_after_scan
        ),
    }

    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(
            "provenance gates failed before output generation: " + ", ".join(failed)
        )

    input_metadata = {
        name: {
            "path": os.path.relpath(paths[name], root),
            "sha256": input_hashes[name],
            "expected_sha256": EXPECTED_INPUT_SHA256[name],
            "hash_match": input_hash_matches[name],
        }
        for name in EXPECTED_INPUT_SHA256
    }
    input_metadata["raw_inventory"] = {
        "path": os.path.relpath(paths["raw_dir"], root),
        "record_count": len(records),
        "content_inventory_sha256": raw_inventory_digest,
        "content_inventory_sha256_rescan": raw_inventory_digest_rescan,
        "rescan_match": raw_inventory_stable_across_full_scan,
    }
    input_metadata["script"] = {
        "path": os.path.relpath(paths["script"], root),
        "sha256": sha256_file(paths["script"]),
    }

    json_payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": (
            "offline_current_file_provenance_not_historic_download_receipt"
        ),
        "current_verified_utc": verification_time,
        "network_accessed": False,
        "historic_receipt_modified": False,
        "scope": {
            "endpoint": ENDPOINT,
            "all_current_raw_files": True,
            "record_count": len(records),
            "final_sample_gate": "132 country-years",
        },
        "time_semantics": {
            "current_verified_utc": (
                "time at which this offline script verified the current file"
            ),
            "filesystem_mtime_utc": (
                "local filesystem metadata only; not a download or retrieval time"
            ),
            "historic_retrieved_utc": (
                "carried only from the unchanged historic receipt; null when "
                "no historic receipt exists"
            ),
        },
        "inputs": input_metadata,
        "counts": counts,
        "gates": {**gates, "all_gates_passed": all(gates.values())},
        "records": records,
    }

    write_csv(paths["csv_output"], records)
    write_json(paths["json_output"], json_payload)

    receipt_sha_after_outputs = sha256_file(paths["historic_receipt"])
    gates["historic_receipt_unchanged_after_outputs"] = (
        receipt_sha_before == receipt_sha_after_outputs
    )
    gates["all_gates_passed"] = all(gates.values())
    if not gates["all_gates_passed"]:
        raise RuntimeError("historic receipt changed while writing derived outputs")

    qa_payload = {
        "schema_version": SCHEMA_VERSION,
        "qa_type": "energy_charts_public_power_local_provenance",
        "current_verified_utc": verification_time,
        "network_accessed": False,
        "historic_receipt_modified": False,
        "historic_receipt_sha256_before": receipt_sha_before,
        "historic_receipt_sha256_after": receipt_sha_after_outputs,
        "raw_inventory_sha256": raw_inventory_digest,
        "raw_inventory_sha256_rescan": raw_inventory_digest_rescan,
        "inputs": input_metadata,
        "outputs": {
            "csv": {
                "path": os.path.relpath(paths["csv_output"], root),
                "bytes": paths["csv_output"].stat().st_size,
                "sha256": sha256_file(paths["csv_output"]),
            },
            "json": {
                "path": os.path.relpath(paths["json_output"], root),
                "bytes": paths["json_output"].stat().st_size,
                "sha256": sha256_file(paths["json_output"]),
            },
            "readme": {
                "path": os.path.relpath(paths["readme"], root),
                "bytes": paths["readme"].stat().st_size,
                "sha256": sha256_file(paths["readme"]),
            },
        },
        "counts": counts,
        "gates": gates,
        "failures": [name for name, passed in gates.items() if not passed],
        "interpretation": (
            "All checks describe current local objects. Missing historic receipt "
            "fields were left null; no download time or HTTP transaction was "
            "reconstructed."
        ),
    }
    write_json(paths["qa_output"], qa_payload)

    print(
        json.dumps(
            {
                "all_gates_passed": gates["all_gates_passed"],
                "records": len(records),
                "historic_receipts": len(historic_records),
                "missing_historic_receipts": len(records) - len(historic_records),
                "final_sample_coverage": len(final_records),
                "csv": str(paths["csv_output"]),
                "json": str(paths["json_output"]),
                "qa": str(paths["qa_output"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
