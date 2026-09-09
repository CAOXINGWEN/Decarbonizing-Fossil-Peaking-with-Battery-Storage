#!/usr/bin/env python3
"""Audit day-ahead-price charging and post-event recovery windows.

Price availability defines a separate descriptive subsample and never changes
the 17-country, 155-country-year physical-profile sample. Natural q75 events
are defined from observed load minus observed wind and solar. Low price means
the country-year price distribution's lower quartile; negative-price hours are
reported separately.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_PATTERN = re.compile(r"price_([a-z]+)_(\d{4})\.json$")
API_TO_ISO = {"uk": "GB"}
MIN_COVERAGE = 0.99
MAX_GAP_H = 24
API_DOCUMENTATION = "https://api.energy-charts.info/"


def longest_missing_run(values: pd.Series) -> int:
    missing = values.isna().to_numpy()
    if not missing.any():
        return 0
    changes = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int((ends - starts).max(initial=0))


def read_price(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    unit = str(payload["unit"]).replace(" ", "")
    if unit.upper() not in {"EUR/MWH", "€/MWH"}:
        raise RuntimeError(f"Unexpected price unit {payload['unit']}: {path}")
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                payload["unix_seconds"], unit="s", utc=True
            ),
            "day_ahead_price_eur_mwh": pd.to_numeric(
                pd.Series(payload["price"]), errors="coerce"
            ).to_numpy(float),
        }
    ).set_index("timestamp_utc")
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).mean()
    return frame.resample("1h").mean()


def combine_price(paths: list[Path]) -> pd.DataFrame:
    frame = pd.concat([read_price(path) for path in sorted(paths)]).sort_index()
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).mean()
    return frame


def label_runs(extreme: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full(len(extreme), -1, dtype=int)
    duration = np.zeros(len(extreme), dtype=int)
    next_id = 0
    start = 0
    while start < len(extreme):
        if not extreme[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(extreme) and extreme[stop]:
            stop += 1
        ids[start:stop] = next_id
        duration[start:stop] = stop - start
        next_id += 1
        start = stop
    return ids, duration


def event_class(duration: int) -> str:
    if duration == 1:
        return "isolated_1d"
    if duration == 2:
        return "adjacent_2d"
    return "multiday_3plus"


def mean_or_nan(frame: pd.DataFrame, field: str) -> float:
    values = pd.to_numeric(frame[field], errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--coverage-threshold", type=float, default=MIN_COVERAGE)
    parser.add_argument("--max-gap-hours", type=int, default=MAX_GAP_H)
    args = parser.parse_args()

    receipt_path = (
        args.root / "metadata" / "energy_charts_download_receipt.json"
    )
    if not receipt_path.exists():
        raise RuntimeError(
            "Price source receipt is required before analysis."
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    price_receipts = [
        row
        for row in receipt.get("records", [])
        if row.get("endpoint") == "price"
        and row.get("status") in {"downloaded", "reused"}
    ]
    invalid_queries = [
        row
        for row in price_receipts
        if "bzn=" not in str(row.get("url", ""))
    ]
    if invalid_queries:
        raise RuntimeError(
            "QUARANTINED price source: Energy-Charts /price requires bzn=; "
            f"{len(invalid_queries)} accepted receipt rows do not record it. "
            "See metadata/energy_charts_price_query_integrity_qa.json."
        )

    panel = pd.read_csv(
        args.root
        / "processed"
        / "energy_charts_europe_hourly_eligible_2015_2024.csv.gz",
        usecols=[
            "timestamp_utc",
            "country",
            "year",
            "load_mw",
            "wind_mw",
            "solar_mw",
        ],
    )
    panel["timestamp_utc"] = pd.to_datetime(
        panel["timestamp_utc"], utc=True
    )
    price_dir = args.root / "raw" / "energy_charts" / "price"
    by_country: dict[str, list[Path]] = {}
    for path in sorted(price_dir.glob("price_*_*.json")):
        match = RAW_PATTERN.search(path.name)
        if not match:
            continue
        api_code = match.group(1)
        by_country.setdefault(
            API_TO_ISO.get(api_code, api_code.upper()), []
        ).append(path)
    price_cache = {
        country: combine_price(paths)
        for country, paths in sorted(by_country.items())
    }

    coverage_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for (country, year), physical in panel.groupby(
        ["country", "year"], sort=True
    ):
        physical = physical.sort_values("timestamp_utc")
        index = pd.DatetimeIndex(physical["timestamp_utc"])
        if country in price_cache:
            raw_price = price_cache[country].reindex(index)[
                "day_ahead_price_eur_mwh"
            ]
        else:
            raw_price = pd.Series(np.nan, index=index)
        coverage = float(raw_price.notna().mean())
        max_gap = longest_missing_run(raw_price)
        eligible = bool(
            coverage >= args.coverage_threshold
            and max_gap <= args.max_gap_hours
        )
        price = raw_price.copy()
        if eligible:
            price = price.interpolate(
                method="time",
                limit=args.max_gap_hours,
                limit_direction="both",
                limit_area="inside",
            )
            price = price.ffill(limit=args.max_gap_hours).bfill(
                limit=args.max_gap_hours
            )
            if price.isna().any():
                raise RuntimeError(
                    f"Eligible price group unresolved: {country}-{year}"
                )
        coverage_rows.append(
            {
                "country": country,
                "year": int(year),
                "physical_profile_eligible": True,
                "price_file_country_available": country in price_cache,
                "price_coverage": coverage,
                "price_max_missing_gap_h": max_gap,
                "price_eligible_country_year": eligible,
                "price_observed_hours": int(raw_price.notna().sum()),
                "price_imputed_hours": (
                    int(raw_price.isna().sum()) if eligible else 0
                ),
                "price_subsample_only": True,
            }
        )
        if not eligible:
            continue

        load = physical["load_mw"].to_numpy(float)
        vre = (
            physical["wind_mw"].to_numpy(float)
            + physical["solar_mw"].to_numpy(float)
        )
        actual_residual = np.maximum(load - vre, 0.0)
        hourly = pd.DataFrame(
            {
                "timestamp_utc": index,
                "actual_residual_mw": actual_residual,
                "day_ahead_price_eur_mwh": price.to_numpy(float),
            }
        )
        hourly["date_utc"] = hourly["timestamp_utc"].dt.floor("D")
        daily = hourly.groupby("date_utc", as_index=False).agg(
            actual_residual_mwh=("actual_residual_mw", "sum")
        )
        threshold = float(daily["actual_residual_mwh"].quantile(0.75))
        daily["extreme"] = daily["actual_residual_mwh"] >= threshold
        event_id, duration = label_runs(daily["extreme"].to_numpy(bool))
        daily["event_id"] = event_id
        daily["duration_d"] = duration
        low_price_threshold = float(price.quantile(0.25))

        for event_id_value, event in daily[daily["extreme"]].groupby(
            "event_id", sort=True
        ):
            duration_d = int(len(event))
            recovery_start = event["date_utc"].max() + pd.Timedelta(days=1)
            recovery = hourly[
                (hourly["timestamp_utc"] >= recovery_start)
                & (
                    hourly["timestamp_utc"]
                    < recovery_start + pd.Timedelta(hours=168)
                )
            ].copy()
            recovery["hours_after_event"] = np.arange(len(recovery))
            recovery["low_price"] = (
                recovery["day_ahead_price_eur_mwh"] <= low_price_threshold
            )
            recovery["negative_price"] = (
                recovery["day_ahead_price_eur_mwh"] < 0
            )
            low_positions = recovery.loc[
                recovery["low_price"], "hours_after_event"
            ]
            negative_positions = recovery.loc[
                recovery["negative_price"], "hours_after_event"
            ]
            row: dict[str, Any] = {
                "country": country,
                "year": int(year),
                "event_id_within_country_year": int(event_id_value),
                "event_start_utc": event["date_utc"].min(),
                "event_end_utc": event["date_utc"].max(),
                "event_duration_d": duration_d,
                "event_cluster_class": event_class(duration_d),
                "q75_actual_residual_threshold_mwh": threshold,
                "country_year_low_price_q25_eur_mwh": low_price_threshold,
                "annual_negative_price_hours": int((price < 0).sum()),
                "recovery_observed_hours": int(len(recovery)),
                "hours_to_first_low_price": (
                    float(low_positions.iloc[0])
                    if len(low_positions)
                    else np.nan
                ),
                "hours_to_first_negative_price": (
                    float(negative_positions.iloc[0])
                    if len(negative_positions)
                    else np.nan
                ),
            }
            for horizon in (24, 72, 168):
                window = recovery.iloc[:horizon]
                row[f"post_event_{horizon}h_mean_price_eur_mwh"] = (
                    float(window["day_ahead_price_eur_mwh"].mean())
                    if len(window)
                    else np.nan
                )
                row[f"post_event_{horizon}h_low_price_share"] = (
                    float(window["low_price"].mean())
                    if len(window)
                    else np.nan
                )
                row[f"post_event_{horizon}h_negative_price_share"] = (
                    float(window["negative_price"].mean())
                    if len(window)
                    else np.nan
                )
            event_rows.append(row)

        country_events = pd.DataFrame(
            [
                row
                for row in event_rows
                if row["country"] == country and row["year"] == int(year)
            ]
        )
        isolated = country_events[
            country_events["event_cluster_class"] == "isolated_1d"
        ]
        multiday = country_events[
            country_events["event_cluster_class"] == "multiday_3plus"
        ]
        comparison: dict[str, Any] = {
            "country": country,
            "year": int(year),
            "isolated_event_count": int(len(isolated)),
            "multiday_event_count": int(len(multiday)),
            "price_subsample_only": True,
        }
        comparison_fields = [
            "hours_to_first_low_price",
            "hours_to_first_negative_price",
        ]
        for horizon in (24, 72, 168):
            comparison_fields.extend(
                [
                    f"post_event_{horizon}h_mean_price_eur_mwh",
                    f"post_event_{horizon}h_low_price_share",
                    f"post_event_{horizon}h_negative_price_share",
                ]
            )
        for field in comparison_fields:
            isolated_mean = mean_or_nan(isolated, field)
            multiday_mean = mean_or_nan(multiday, field)
            comparison[f"isolated_mean_{field}"] = isolated_mean
            comparison[f"multiday_mean_{field}"] = multiday_mean
            comparison[f"multiday_minus_isolated_{field}"] = (
                multiday_mean - isolated_mean
            )
        contrast_rows.append(comparison)

    research = args.root / "artifacts" / "research"
    processed = args.root / "processed"
    coverage_frame = pd.DataFrame(coverage_rows).sort_values(
        ["country", "year"]
    )
    event_frame = pd.DataFrame(event_rows).sort_values(
        ["country", "year", "event_id_within_country_year"]
    )
    contrast_frame = pd.DataFrame(contrast_rows).sort_values(
        ["country", "year"]
    )
    coverage_path = research / "energy_charts_price_country_year_coverage.csv"
    event_path = (
        processed / "energy_charts_price_recovery_events_2015_2024.csv.gz"
    )
    contrast_path = research / "energy_charts_price_recovery_contrasts.csv"
    coverage_frame.to_csv(coverage_path, index=False)
    event_frame.to_csv(
        event_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.6f",
    )
    contrast_frame.to_csv(contrast_path, index=False)

    manifest = json.loads(
        (
            args.root / "metadata" / "energy_charts_panel_manifest.json"
        ).read_text(encoding="utf-8")
    )
    balanced = set(manifest["sample"]["balanced_eu_countries"])
    sample_rows: list[dict[str, Any]] = []
    fields = [
        column
        for column in contrast_frame.columns
        if column.startswith("multiday_minus_isolated_")
    ]
    for sample, countries in (
        ("price_expanded_available", None),
        ("price_balanced_eu_available", balanced),
    ):
        selected = (
            contrast_frame
            if countries is None
            else contrast_frame[contrast_frame["country"].isin(countries)]
        )
        for field in fields:
            values = pd.to_numeric(selected[field], errors="coerce").dropna()
            sample_rows.append(
                {
                    "sample": sample,
                    "metric": field,
                    "country_count": int(
                        selected.loc[values.index, "country"].nunique()
                    ),
                    "country_year_count": int(len(values)),
                    "country_year_median": (
                        float(values.median()) if len(values) else np.nan
                    ),
                    "positive_country_year_fraction": (
                        float((values > 0).mean()) if len(values) else np.nan
                    ),
                    "interpretation": (
                        "descriptive day-ahead-price subsample; no change to "
                        "physical-profile sample"
                    ),
                }
            )
    sample_path = research / "energy_charts_price_recovery_sample_summary.csv"
    pd.DataFrame(sample_rows).to_csv(sample_path, index=False)

    eligible = coverage_frame[coverage_frame["price_eligible_country_year"]]
    lines = [
        "# Energy-Charts day-ahead-price recovery-window audit",
        "",
        (
            f"Price passes the 99%/24 h gate in {len(eligible)}/"
            f"{len(coverage_frame)} physical country-years across "
            f"{eligible['country'].nunique()} countries. This is a separate "
            "descriptive subsample and does not remove any country-year from "
            "the physical mechanism screen."
        ),
        "",
        (
            "Natural events use each country-year's observed residual-demand "
            "q75 threshold. Low price is the country-year lower quartile; "
            "negative-price hours are reported separately. Recovery metrics "
            "cover the first 24, 72 and 168 hours following each event."
        ),
        "",
        (
            "Prices are market outcomes jointly determined by demand, "
            "generation, interconnection and bidding. These tables test "
            "external consistency of charging opportunities and do not "
            "identify a causal price channel."
        ),
    ]
    memo_path = research / "ENERGY_CHARTS_PRICE_RECOVERY_AUDIT.md"
    memo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "physical_country_years": int(len(coverage_frame)),
                "price_eligible_country_years": int(len(eligible)),
                "price_countries": int(eligible["country"].nunique()),
                "coverage": str(coverage_path),
                "events": str(event_path),
                "contrasts": str(contrast_path),
                "sample_summary": str(sample_path),
                "memo": str(memo_path),
                "api_documentation": API_DOCUMENTATION,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
