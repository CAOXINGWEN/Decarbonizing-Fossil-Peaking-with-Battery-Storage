#!/usr/bin/env python3
"""Download auditable country-year extracts from the Energy-Charts API.

The default country set combines the broad OPSD load/wind/solar candidate
sample with Belgium, Croatia and Portugal, which passed targeted
Energy-Charts field probes.  The download itself does not determine the final
analysis sample: completeness and variable-availability gates are applied by
``audit_energy_charts_coverage.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_COUNTRIES = (
    "at",
    "be",
    "bg",
    "ch",
    "cz",
    "de",
    "dk",
    "ee",
    "es",
    "fr",
    "uk",
    "gr",
    "hr",
    "it",
    "pt",
    "ro",
    "si",
)
DEFAULT_YEARS = tuple(range(2015, 2020))
ENDPOINT_KEYS = {
    "public_power": ("unix_seconds", "production_types", "deprecated"),
    "cbpf": ("unix_seconds", "countries", "deprecated"),
    "price": ("unix_seconds", "price", "unit", "deprecated"),
}
BASE_URL = "https://api.energy-charts.info"
USER_AGENT = "LDES-Europe-mid-duration-valley/0.1 (academic reproducibility)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_json(path: Path, endpoint: str) -> tuple[bool, dict | None]:
    if not path.exists() or path.stat().st_size == 0:
        return False, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False, None
    required = ENDPOINT_KEYS[endpoint]
    valid = all(key in payload for key in required) and bool(payload["unix_seconds"])
    return valid, payload if valid else None


def download(url: str, destination: Path, endpoint: str, retries: int) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    valid, payload = valid_json(destination, endpoint)
    if valid:
        return {
            "status": "reused",
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "n_timestamps": len(payload["unix_seconds"]),
            "deprecated": bool(payload["deprecated"]),
        }

    part = destination.with_suffix(destination.suffix + ".part")
    part.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=300) as response:
                part.write_bytes(response.read())
            valid, payload = valid_json(part, endpoint)
            if not valid:
                raise ValueError("response is not a valid Energy-Charts payload")
            part.replace(destination)
            return {
                "status": "downloaded",
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "n_timestamps": len(payload["unix_seconds"]),
                "deprecated": bool(payload["deprecated"]),
            }
        except (
            OSError,
            ValueError,
            urllib.error.URLError,
            http.client.IncompleteRead,
        ) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    raise RuntimeError(f"failed after {retries} attempts: {url}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="data-product root",
    )
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument(
        "--years", nargs="+", type=int, default=list(DEFAULT_YEARS)
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        choices=tuple(ENDPOINT_KEYS),
        default=("public_power",),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    countries = tuple(country.lower() for country in args.countries)
    years = tuple(sorted(set(args.years)))
    raw = args.root / "raw" / "energy_charts"
    jobs: list[tuple[str, str, int, str, Path]] = []
    for endpoint in args.endpoints:
        for country in countries:
            for year in years:
                url = (
                    f"{BASE_URL}/{endpoint}?country={country}"
                    f"&start={year}-01-01&end={year}-12-31"
                )
                destination = raw / endpoint / f"{endpoint}_{country}_{year}.json"
                jobs.append((endpoint, country, year, url, destination))

    def run_job(job: tuple[str, str, int, str, Path]) -> dict:
        endpoint, country, year, url, destination = job
        retrieved_utc = datetime.now(timezone.utc).isoformat()
        result = download(url, destination, endpoint, args.retries)
        return {
            "endpoint": endpoint,
            "country_api_code": country,
            "year": year,
            "url": url,
            "local_path": str(destination.relative_to(args.root)),
            "retrieved_utc": retrieved_utc,
            **result,
        }

    receipts: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            endpoint, country, year, url, destination = futures[future]
            try:
                receipt = future.result()
            except Exception as exc:
                receipt = {
                    "endpoint": endpoint,
                    "country_api_code": country,
                    "year": year,
                    "url": url,
                    "local_path": str(destination.relative_to(args.root)),
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "bytes": 0,
                    "sha256": None,
                    "n_timestamps": 0,
                    "deprecated": None,
                }
            receipts.append(receipt)
            print(
                f"{receipt['endpoint']:12s} "
                f"{receipt['country_api_code'].upper():>2s} "
                f"{receipt['year']}: {receipt['status']} "
                f"{receipt['n_timestamps']} timestamps",
                flush=True,
            )

    receipts.sort(
        key=lambda row: (
            row["endpoint"],
            row["country_api_code"],
            int(row["year"]),
        )
    )
    metadata = args.root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    path = metadata / "energy_charts_download_receipt.json"
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get(
                "records", []
            )
        except (json.JSONDecodeError, OSError):
            existing = []
    merged = {
        (
            row["endpoint"],
            row["country_api_code"],
            int(row["year"]),
        ): row
        for row in existing
    }
    for row in receipts:
        merged[
            (
                row["endpoint"],
                row["country_api_code"],
                int(row["year"]),
            )
        ] = row
    records = [
        merged[key]
        for key in sorted(merged, key=lambda item: (item[0], item[1], item[2]))
    ]
    path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "api_base": BASE_URL,
                "license": (
                    "CC BY 4.0 unless stated otherwise; attribution to "
                    "Energy-Charts.info required"
                ),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
