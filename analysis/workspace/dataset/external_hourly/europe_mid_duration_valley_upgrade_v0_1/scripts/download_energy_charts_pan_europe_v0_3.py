#!/usr/bin/env python3
"""Download an append-only pan-European Energy-Charts expansion.

This script never writes to the frozen v0.1/v0.2 raw archive. New country-year
responses are stored under ``raw/energy_charts_pan_europe_v0_3`` with an
incrementally committed receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_COUNTRIES = (
    "fi",
    "hu",
    "ie",
    "lt",
    "lu",
    "lv",
    "nl",
    "no",
    "pl",
    "se",
    "sk",
)
DEFAULT_YEARS = tuple(range(2015, 2025))
BASE_URL = "https://api.energy-charts.info/public_power"
USER_AGENT = (
    "LDES-Europe-mid-duration-valley/0.3 "
    "(academic reproducibility; CC BY 4.0 attribution)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"unix_seconds", "production_types", "deprecated"}
    if not required.issubset(payload):
        raise ValueError(f"missing required keys in {path}")
    timestamps = payload["unix_seconds"]
    if not isinstance(timestamps, list) or not timestamps:
        raise ValueError(f"empty timestamp array in {path}")
    names = [
        str(series.get("name", ""))
        for series in payload["production_types"]
    ]
    for series in payload["production_types"]:
        if len(series.get("data", [])) != len(timestamps):
            raise ValueError(
                f"series length mismatch in {path}: {series.get('name')}"
            )
    return {
        "timestamp_count": len(timestamps),
        "production_types": names,
        "has_fossil_gas": "Fossil gas" in names,
        "has_load": "Load" in names,
        "has_solar": "Solar" in names,
        "has_wind_onshore": "Wind onshore" in names,
        "has_wind_offshore": "Wind offshore" in names,
        "deprecated": bool(payload["deprecated"]),
    }


def read_receipt(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        (str(record["country_api_code"]), int(record["year"])): record
        for record in payload.get("records", [])
    }


def write_receipt(path: Path, records: dict[tuple[str, int], dict]) -> None:
    ordered = [
        records[key] for key in sorted(records, key=lambda key: (key[0], key[1]))
    ]
    payload = {
        "schema_version": "0.3.0",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": BASE_URL,
        "provider": "Fraunhofer ISE Energy-Charts",
        "license": "CC BY 4.0 unless stated otherwise",
        "attribution": "Energy-Charts.info",
        "release_contract": (
            "append-only expansion; frozen v0.1/v0.2 raw files are not modified"
        ),
        "records": ordered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--delay-seconds", type=float, default=3.0)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    raw = (
        args.root
        / "raw"
        / "energy_charts_pan_europe_v0_3"
        / "public_power"
    )
    receipt_path = (
        args.root
        / "metadata"
        / "energy_charts_pan_europe_download_receipt_v0_3.json"
    )
    raw.mkdir(parents=True, exist_ok=True)
    records = read_receipt(receipt_path)

    for country in sorted(set(str(code).lower() for code in args.countries)):
        for year in sorted(set(int(value) for value in args.years)):
            destination = raw / f"public_power_{country}_{year}.json"
            query = urllib.parse.urlencode(
                {
                    "country": country,
                    "start": f"{year}-01-01",
                    "end": f"{year}-12-31",
                }
            )
            url = f"{BASE_URL}?{query}"
            if destination.exists():
                try:
                    fields = validate(destination)
                    record = {
                        "country_api_code": country,
                        "year": year,
                        "status": "reused",
                        "url": url,
                        "local_path": str(destination.relative_to(args.root)),
                        "bytes": destination.stat().st_size,
                        "sha256": sha256(destination),
                        **fields,
                    }
                    records[(country, year)] = record
                    write_receipt(receipt_path, records)
                    print(country.upper(), year, "reused", flush=True)
                    continue
                except Exception:  # noqa: BLE001 - replace invalid partial
                    destination.unlink(missing_ok=True)

            part = destination.with_suffix(".json.part")
            part.unlink(missing_ok=True)
            error = ""
            status = "failed"
            fields: dict = {}
            attempts = 0
            for attempt in range(1, args.retries + 1):
                attempts = attempt
                try:
                    request = urllib.request.Request(
                        url, headers={"User-Agent": USER_AGENT}
                    )
                    with urllib.request.urlopen(request, timeout=180) as response:
                        part.write_bytes(response.read())
                    fields = validate(part)
                    part.replace(destination)
                    status = "downloaded"
                    error = ""
                    break
                except urllib.error.HTTPError as exc:
                    part.unlink(missing_ok=True)
                    error = f"HTTPError {exc.code}: {exc.reason}"
                    if exc.code == 429 and attempt < args.retries:
                        retry_after = exc.headers.get("Retry-After")
                        wait = (
                            float(retry_after)
                            if retry_after and retry_after.isdigit()
                            else args.delay_seconds * attempt
                        )
                        time.sleep(wait)
                        continue
                    break
                except Exception as exc:  # noqa: BLE001 - audited failure
                    part.unlink(missing_ok=True)
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.retries:
                        time.sleep(args.delay_seconds * attempt)
                        continue
                    break

            record = {
                "country_api_code": country,
                "year": year,
                "status": status,
                "url": url,
                "local_path": str(destination.relative_to(args.root)),
                "attempts": attempts,
                "error": error,
                "bytes": (
                    destination.stat().st_size if destination.exists() else 0
                ),
                "sha256": sha256(destination) if destination.exists() else None,
                **fields,
            }
            records[(country, year)] = record
            write_receipt(receipt_path, records)
            print(
                country.upper(),
                year,
                status,
                record.get("timestamp_count", 0),
                record.get("has_fossil_gas", False),
                flush=True,
            )
            time.sleep(max(0.0, args.delay_seconds))


if __name__ == "__main__":
    main()
