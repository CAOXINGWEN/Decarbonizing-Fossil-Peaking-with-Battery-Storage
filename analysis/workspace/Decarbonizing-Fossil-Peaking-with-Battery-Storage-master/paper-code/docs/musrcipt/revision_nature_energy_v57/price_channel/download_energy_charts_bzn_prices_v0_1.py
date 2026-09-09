#!/usr/bin/env python3
"""Download licensed day-ahead prices from the Fraunhofer Energy-Charts API.

The legacy project price pull used the deprecated ``country`` query argument
and returned the DE-LU series for every requested country.  This downloader
uses the documented bidding-zone argument ``bzn`` and writes a separate,
versioned dataset.  Only zones whose API response declares a CC BY 4.0 licence
are retained for manuscript analysis.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[5]
DATA_ROOT = (
    WORKSPACE
    / "dataset"
    / "external_hourly"
    / "europe_mid_duration_valley_upgrade_v0_1"
)
OUT = DATA_ROOT / "raw" / "energy_charts" / "price_bzn_valid_v0_1"

ZONES = {
    "AT": ["AT"],
    "BE": ["BE"],
    "CZ": ["CZ"],
    "DE": ["DE-LU"],
    "DK": ["DK1", "DK2"],
    "FR": ["FR"],
    "SI": ["SI"],
}
YEARS = range(2019, 2025)
URL = "https://api.energy-charts.info/price"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(zone: str, year: int) -> tuple[pd.DataFrame, dict[str, object]]:
    params = {
        "bzn": zone,
        "start": f"{year}-01-01",
        "end": f"{year + 1}-01-01",
    }
    response = None
    for attempt in range(8):
        try:
            response = requests.get(URL, params=params, timeout=300)
        except requests.RequestException:
            time.sleep(min(90.0, 5.0 * 2**attempt))
            continue
        if response.status_code != 429:
            response.raise_for_status()
            break
        time.sleep(min(90.0, 5.0 * 2**attempt))
    if response is None or response.status_code == 429:
        raise RuntimeError(f"Rate limit persisted for {zone}-{year}")
    payload = response.json()
    licence = str(payload.get("license_info", ""))
    if "CC BY 4.0" not in licence:
        raise RuntimeError(f"{zone}-{year} is not declared CC BY 4.0: {licence}")
    timestamp = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamp,
            "price_eur_mwh": pd.to_numeric(payload["price"], errors="coerce"),
            "bidding_zone": zone,
            "year_requested": year,
        }
    )
    metadata = {
        "request_url": response.url,
        "bidding_zone": zone,
        "year": year,
        "license_info": licence,
        "unit": payload.get("unit"),
        "deprecated": payload.get("deprecated"),
        "rows": len(frame),
        "finite_prices": int(frame["price_eur_mwh"].notna().sum()),
        "timestamp_min_utc": str(frame["timestamp_utc"].min()),
        "timestamp_max_utc": str(frame["timestamp_utc"].max()),
    }
    return frame, metadata


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    for country, zones in ZONES.items():
        for zone in zones:
            for year in YEARS:
                path = OUT / f"price_{zone.lower()}_{year}.csv.gz"
                meta_path = OUT / f"price_{zone.lower()}_{year}.json"
                if path.exists() and meta_path.exists():
                    frame = pd.read_csv(path, parse_dates=["timestamp_utc"])
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                else:
                    frame, meta = download(zone, year)
                    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
                    meta_path.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    time.sleep(2.0)
                frame["country"] = country
                rows.append(frame)
                manifest.append({**meta, "country": country, "file": str(path), "sha256": sha256(path)})

    all_prices = pd.concat(rows, ignore_index=True)
    # Denmark has two bidding zones.  Their unweighted national mean is used
    # only for the country-level sensitivity and the two zonal series remain
    # in the raw table for audit.
    national = (
        all_prices.groupby(["country", "timestamp_utc"], as_index=False)
        .agg(price_eur_mwh=("price_eur_mwh", "mean"), n_zones=("bidding_zone", "nunique"))
    )
    all_prices.to_csv(
        OUT / "energy_charts_bidding_zone_prices_2019_2024.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    national.to_csv(
        OUT / "energy_charts_country_price_sensitivity_2019_2024.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "endpoint": URL,
                "query_argument": "bzn",
                "countries": ZONES,
                "years": list(YEARS),
                "price_rows": len(all_prices),
                "country_price_rows": len(national),
                "files": manifest,
                "legacy_country_argument_data_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
