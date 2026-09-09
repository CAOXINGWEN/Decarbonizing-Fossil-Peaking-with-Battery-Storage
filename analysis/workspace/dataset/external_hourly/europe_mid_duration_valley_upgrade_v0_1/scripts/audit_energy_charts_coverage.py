#!/usr/bin/env python3
"""Audit country-year completeness in downloaded Energy-Charts extracts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


COUNTRY_METADATA = {
    "at": ("AT", "Austria", "Europe/Vienna", True),
    "be": ("BE", "Belgium", "Europe/Brussels", True),
    "bg": ("BG", "Bulgaria", "Europe/Sofia", True),
    "ch": ("CH", "Switzerland", "Europe/Zurich", False),
    "cz": ("CZ", "Czechia", "Europe/Prague", True),
    "de": ("DE", "Germany", "Europe/Berlin", True),
    "dk": ("DK", "Denmark", "Europe/Copenhagen", True),
    "ee": ("EE", "Estonia", "Europe/Tallinn", True),
    "es": ("ES", "Spain", "Europe/Madrid", True),
    "fr": ("FR", "France", "Europe/Paris", True),
    "uk": ("GB", "United Kingdom", "Europe/London", False),
    "gr": ("GR", "Greece", "Europe/Athens", True),
    "hr": ("HR", "Croatia", "Europe/Zagreb", True),
    "it": ("IT", "Italy", "Europe/Rome", True),
    "pt": ("PT", "Portugal", "Europe/Lisbon", True),
    "ro": ("RO", "Romania", "Europe/Bucharest", True),
    "si": ("SI", "Slovenia", "Europe/Ljubljana", True),
}
CORE_VARIABLES = (
    "load",
    "wind_onshore",
    "wind_offshore",
    "solar",
    "fossil_gas",
    "hydro_pumped_storage_consumption",
    "hydro_pumped_storage",
    "hydro_run_of_river",
    "hydro_water_reservoir",
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def max_missing_run(values: pd.Series) -> int:
    missing = values.isna().to_numpy()
    if not missing.any():
        return 0
    changes = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int((ends - starts).max(initial=0))


def load_hourly(path: Path, timezone_name: str, year: int) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    data: dict[str, np.ndarray | pd.DatetimeIndex] = {"timestamp_utc": index}
    for series in payload["production_types"]:
        values = pd.to_numeric(
            pd.Series(series["data"]), errors="coerce"
        ).to_numpy()
        if len(values) != len(index):
            raise ValueError(f"length mismatch in {path}: {series['name']}")
        data[slug(series["name"])] = values
    frame = pd.DataFrame(data).set_index("timestamp_utc").sort_index()
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).mean()
    frame = frame.resample("1h").mean()

    timezone = ZoneInfo(timezone_name)
    start = pd.Timestamp(f"{year}-01-01", tz=timezone).tz_convert("UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz=timezone).tz_convert("UTC")
    expected = pd.date_range(start, end, freq="1h", inclusive="left")
    return frame.reindex(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.99)
    parser.add_argument("--max-gap-hours", type=int, default=24)
    parser.add_argument("--min-years-expanded", type=int, default=3)
    args = parser.parse_args()

    source = args.root / "raw" / "energy_charts" / "public_power"
    audit = args.root / "artifacts" / "research"
    audit.mkdir(parents=True, exist_ok=True)
    country_year_rows: list[dict] = []
    variable_rows: list[dict] = []

    paths = sorted(source.glob("public_power_*_*.json"))
    if not paths:
        raise FileNotFoundError(f"no public_power files under {source}")
    pattern = re.compile(r"public_power_([a-z]+)_(\d{4})\.json$")

    for path in paths:
        match = pattern.search(path.name)
        if not match:
            continue
        api_code, year_text = match.groups()
        if api_code not in COUNTRY_METADATA:
            continue
        iso2, country_name, timezone_name, is_eu27 = COUNTRY_METADATA[api_code]
        year = int(year_text)
        frame = load_hourly(path, timezone_name, year)
        n_expected = len(frame)

        wind_columns = [
            column
            for column in ("wind_onshore", "wind_offshore")
            if column in frame
        ]
        vre_columns = wind_columns + (
            ["solar"] if "solar" in frame else []
        )
        if vre_columns:
            frame["vre_generation"] = frame[vre_columns].sum(
                axis=1, min_count=len(vre_columns)
            )
        else:
            frame["vre_generation"] = np.nan

        for variable in (*CORE_VARIABLES, "vre_generation"):
            present = variable in frame
            values = (
                frame[variable]
                if present
                else pd.Series(np.nan, index=frame.index)
            )
            variable_rows.append(
                {
                    "country": iso2,
                    "country_name": country_name,
                    "api_code": api_code,
                    "year": year,
                    "is_current_eu27": is_eu27,
                    "variable": variable,
                    "field_present": present,
                    "expected_hours": n_expected,
                    "valid_hours": int(values.notna().sum()),
                    "coverage_share": float(values.notna().mean()),
                    "max_consecutive_missing_hours": max_missing_run(values),
                    "annual_energy_twh": (
                        float(values.sum(min_count=1) / 1e6)
                        if values.notna().any()
                        else np.nan
                    ),
                }
            )

        load = (
            frame["load"]
            if "load" in frame
            else pd.Series(np.nan, index=frame.index)
        )
        vre = frame["vre_generation"]
        gas = (
            frame["fossil_gas"]
            if "fossil_gas" in frame
            else pd.Series(np.nan, index=frame.index)
        )
        load_ok = (
            load.notna().mean() >= args.coverage_threshold
            and max_missing_run(load) <= args.max_gap_hours
        )
        vre_ok = (
            bool(wind_columns)
            and "solar" in frame
            and vre.notna().mean() >= args.coverage_threshold
            and max_missing_run(vre) <= args.max_gap_hours
            and float(vre.sum(min_count=1)) > 0
        )
        gas_ok = (
            gas.notna().mean() >= args.coverage_threshold
            and max_missing_run(gas) <= args.max_gap_hours
        )
        country_year_rows.append(
            {
                "country": iso2,
                "country_name": country_name,
                "api_code": api_code,
                "year": year,
                "is_current_eu27": is_eu27,
                "expected_hours": n_expected,
                "load_coverage": float(load.notna().mean()),
                "load_max_missing_gap_h": max_missing_run(load),
                "vre_coverage": float(vre.notna().mean()),
                "vre_max_missing_gap_h": max_missing_run(vre),
                "gas_coverage": float(gas.notna().mean()),
                "gas_max_missing_gap_h": max_missing_run(gas),
                "physical_core_eligible": bool(load_ok and vre_ok),
                "gas_response_eligible": bool(load_ok and vre_ok and gas_ok),
            }
        )

    country_year = pd.DataFrame(country_year_rows).sort_values(
        ["country", "year"]
    )
    variables = pd.DataFrame(variable_rows).sort_values(
        ["country", "year", "variable"]
    )
    year_counts = (
        country_year.groupby(
            ["country", "country_name", "is_current_eu27"], observed=True
        )
        .agg(
            downloaded_years=("year", "nunique"),
            physical_core_years=("physical_core_eligible", "sum"),
            gas_response_years=("gas_response_eligible", "sum"),
        )
        .reset_index()
    )
    expected_years = int(country_year["year"].nunique())
    year_counts["primary_balanced_eu"] = (
        year_counts["is_current_eu27"]
        & (year_counts["physical_core_years"] == expected_years)
        & (year_counts["downloaded_years"] == expected_years)
    )
    year_counts["expanded_europe"] = (
        year_counts["physical_core_years"] >= args.min_years_expanded
    )
    year_counts["gas_response_sample"] = (
        year_counts["gas_response_years"] >= args.min_years_expanded
    )

    country_year.to_csv(
        audit / "energy_charts_country_year_coverage.csv", index=False
    )
    variables.to_csv(
        audit / "energy_charts_variable_coverage_long.csv", index=False
    )
    year_counts.to_csv(
        audit / "energy_charts_country_sample_summary.csv", index=False
    )

    summary_lines = [
        "# Energy-Charts 欧洲样本覆盖审计",
        "",
        (
            f"门槛为负荷与风光逐时覆盖率不低于 "
            f"{args.coverage_threshold:.1%}，最长连续缺口不超过 "
            f"{args.max_gap_hours} h。"
        ),
        "",
        "## 平衡 EU 主样本候选",
        "",
        ", ".join(
            year_counts.loc[
                year_counts["primary_balanced_eu"], "country"
            ].tolist()
        )
        or "无",
        "",
        "## 扩展欧洲样本候选",
        "",
        ", ".join(
            year_counts.loc[
                year_counts["expanded_europe"], "country"
            ].tolist()
        )
        or "无",
        "",
        "## 具有燃气响应字段的扩展候选",
        "",
        ", ".join(
            year_counts.loc[
                year_counts["gas_response_sample"], "country"
            ].tolist()
        )
        or "无",
        "",
        (
            "燃气响应字段只表示国家级化石燃气发电，不等同于实际燃气调峰。"
            "机组是否承担尖峰任务需使用机组级运行统计另行筛选。"
        ),
    ]
    (audit / "energy_charts_coverage_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print(year_counts.to_string(index=False))


if __name__ == "__main__":
    main()
