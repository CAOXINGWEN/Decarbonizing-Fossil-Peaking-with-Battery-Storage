#!/usr/bin/env python3
"""Probe Energy-Charts country coverage without modifying a frozen release."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_COUNTRIES = (
    "al",
    "at",
    "ba",
    "be",
    "bg",
    "ch",
    "cy",
    "cz",
    "de",
    "dk",
    "ee",
    "es",
    "fi",
    "fr",
    "gr",
    "hr",
    "hu",
    "ie",
    "it",
    "lt",
    "lu",
    "lv",
    "me",
    "mk",
    "mt",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "rs",
    "se",
    "si",
    "sk",
    "tr",
    "uk",
    "xk",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-01-02")
    parser.add_argument("--delay-seconds", type=float, default=2.5)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "metadata"
        / "energy_charts_pan_europe_country_probe_v0_3.json",
    )
    args = parser.parse_args()

    prior_records: dict[str, dict] = {}
    if args.output.exists():
        try:
            prior = json.loads(args.output.read_text(encoding="utf-8"))
            prior_records = {
                str(record["country_api_code"]): record
                for record in prior.get("records", [])
            }
        except (json.JSONDecodeError, OSError, KeyError):
            prior_records = {}
    completed = {
        country: record
        for country, record in prior_records.items()
        if record.get("status") == "ok"
    }
    records_by_country: dict[str, dict] = dict(prior_records)

    def checkpoint() -> None:
        records = [
            records_by_country[country]
            for country in sorted(records_by_country)
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "schema_version": "0.3.0",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "endpoint": (
                        "https://api.energy-charts.info/public_power"
                    ),
                    "start": args.start,
                    "end": args.end,
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for country in args.countries:
        if country in completed:
            print(
                country.upper(),
                "reused",
                completed[country].get("timestamp_count", 0),
                completed[country].get("has_fossil_gas", False),
                flush=True,
            )
            continue
        query = urllib.parse.urlencode(
            {"country": country, "start": args.start, "end": args.end}
        )
        url = f"https://api.energy-charts.info/public_power?{query}"
        record: dict = {
            "country_api_code": country,
            "url": url,
            "status": "failed",
        }
        for attempt in range(1, args.retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": (
                            "LDES-Europe-mid-duration-valley/0.3 "
                            "(academic coverage probe)"
                        )
                    },
                )
                with urllib.request.urlopen(request, timeout=20) as response:
                    payload = json.load(response)
                names = [
                    str(item.get("name", ""))
                    for item in payload.get("production_types", [])
                ]
                record.update(
                    {
                        "status": "ok",
                        "timestamp_count": len(
                            payload.get("unix_seconds", [])
                        ),
                        "deprecated": bool(
                            payload.get("deprecated", False)
                        ),
                        "production_types": names,
                        "has_fossil_gas": "Fossil gas" in names,
                        "has_load": "Load" in names,
                        "has_solar": "Solar" in names,
                        "has_wind_onshore": "Wind onshore" in names,
                        "has_wind_offshore": "Wind offshore" in names,
                        "attempts": attempt,
                    }
                )
                break
            except urllib.error.HTTPError as exc:
                record.update(
                    {
                        "http_status": exc.code,
                        "error": str(exc),
                        "attempts": attempt,
                    }
                )
                if exc.code != 429 or attempt == args.retries:
                    break
                retry_after = exc.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else args.delay_seconds * attempt
                )
                time.sleep(wait)
            except Exception as exc:  # noqa: BLE001 - preserve provider failure
                record.update(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "attempts": attempt,
                    }
                )
                break
        records_by_country[country] = record
        checkpoint()
        print(
            country.upper(),
            record["status"],
            record.get("timestamp_count", 0),
            record.get("has_fossil_gas", False),
            flush=True,
        )
        time.sleep(max(0.0, args.delay_seconds))

    checkpoint()


if __name__ == "__main__":
    main()
