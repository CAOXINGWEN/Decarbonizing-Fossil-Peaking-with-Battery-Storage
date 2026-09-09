#!/usr/bin/env python3
"""Build the prespecified above-floor sensitivity for observed fossil gas.

The main historical task remains the raw, non-negative hourly Energy-Charts
``Fossil gas`` series.  This script creates a *sensitivity only* that removes
the country-year-specific 10th percentile within each month-by-hour-of-day
cell.  The positive remainder is intended to reduce the influence of
persistent CHP/baseload operation without pretending to identify OCGT or CCGT
units.  The main-sample eligibility rule is frozen before this transformation:
complete raw gas observations and annual raw gas energy of at least 1% of
system load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INPUT = (
    ROOT
    / "processed"
    / "energy_charts_observed_gas_task_panel_2015_2024.csv.gz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "processed"
    / "energy_charts_observed_gas_above_floor_task_panel_2015_2024.csv.gz"
)
DEFAULT_QA = (
    ROOT
    / "artifacts"
    / "research"
    / "energy_charts_observed_gas_task_sample_and_floor_qa.json"
)
DEFAULT_SAMPLE = (
    ROOT
    / "artifacts"
    / "research"
    / "energy_charts_observed_gas_task_prespecified_sample.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--floor-quantile", type=float, default=0.10)
    args = parser.parse_args()

    if not 0.0 <= args.floor_quantile <= 0.5:
        raise ValueError("--floor-quantile must be in [0, 0.5]")
    source = args.input.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    required = {
        "timestamp_utc",
        "country",
        "year",
        "system_load_mw",
        "observed_wind_mw",
        "observed_solar_mw",
        "observed_gas_peaking_task_mw",
        "gas_task_eligible",
        "gas_task_material_share_ge_1pct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Input fields missing: {missing}")
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="coerce"
    )
    frame["_month"] = frame["timestamp_utc"].dt.month
    frame["_hour"] = frame["timestamp_utc"].dt.hour
    frame["_raw_gas_mw"] = pd.to_numeric(
        frame["observed_gas_peaking_task_mw"], errors="coerce"
    )
    main_sample = as_bool(frame["gas_task_eligible"]) & as_bool(
        frame["gas_task_material_share_ge_1pct"]
    )
    frame["gas_task_main_prespecified_sample"] = main_sample

    keys = ["country", "year", "_month", "_hour"]
    frame["gas_month_hour_q10_floor_mw"] = frame.groupby(keys)[
        "_raw_gas_mw"
    ].transform(lambda values: values.quantile(args.floor_quantile))
    frame["observed_gas_above_floor_task_mw"] = np.maximum(
        frame["_raw_gas_mw"] - frame["gas_month_hour_q10_floor_mw"], 0.0
    )
    frame["load_mw"] = frame["observed_gas_above_floor_task_mw"]
    frame["wind_mw"] = frame["observed_wind_mw"]
    frame["solar_mw"] = frame["observed_solar_mw"]
    frame["gas_task_variant"] = "month_hour_q10_above_floor_sensitivity"
    frame["gas_task_sensitivity_contract"] = (
        "sensitivity only; main sample fixed on complete raw Fossil gas and "
        "raw annual gas share >=1%; run adapter with vre_energy_share=0"
    )

    annual = (
        frame.groupby(["country", "year"], as_index=False)
        .agg(
            hours=("timestamp_utc", "size"),
            unique_hours=("timestamp_utc", "nunique"),
            main_prespecified_sample=(
                "gas_task_main_prespecified_sample",
                "all",
            ),
            system_load_mwh=("system_load_mw", "sum"),
            raw_gas_mwh=("_raw_gas_mw", "sum"),
            above_floor_gas_mwh=("observed_gas_above_floor_task_mw", "sum"),
            raw_gas_missing_hours=("_raw_gas_mw", lambda x: int(x.isna().sum())),
            above_floor_missing_hours=(
                "observed_gas_above_floor_task_mw",
                lambda x: int(x.isna().sum()),
            ),
        )
        .sort_values(["country", "year"])
    )
    annual["raw_gas_share_of_system_load"] = (
        annual["raw_gas_mwh"] / annual["system_load_mwh"]
    )
    annual["above_floor_gas_share_of_system_load"] = (
        annual["above_floor_gas_mwh"] / annual["system_load_mwh"]
    )
    annual["above_floor_share_of_raw_gas"] = (
        annual["above_floor_gas_mwh"] / annual["raw_gas_mwh"]
    )
    annual["above_floor_material_share_ge_1pct"] = (
        annual["above_floor_gas_share_of_system_load"] >= 0.01
    )
    annual.to_csv(args.sample, index=False)

    selected = annual[annual["main_prespecified_sample"]].copy()
    expected_sample = int(main_sample.groupby([frame["country"], frame["year"]]).all().sum())
    checks = {
        "input_exists": source.exists(),
        "all_country_years_have_8760_rows": bool((annual["hours"] == 8760).all()),
        "all_country_years_have_unique_utc_hours": bool(
            (annual["unique_hours"] == annual["hours"]).all()
        ),
        "selected_raw_gas_complete": bool(
            (selected["raw_gas_missing_hours"] == 0).all()
        ),
        "selected_above_floor_complete": bool(
            (selected["above_floor_missing_hours"] == 0).all()
        ),
        "selected_raw_gas_share_ge_1pct": bool(
            (selected["raw_gas_share_of_system_load"] >= 0.01).all()
        ),
        "above_floor_nonnegative": bool(
            (
                frame.loc[
                    frame["gas_task_main_prespecified_sample"],
                    "observed_gas_above_floor_task_mw",
                ]
                >= 0
            ).all()
        ),
        "above_floor_not_above_raw": bool(
            (
                frame.loc[
                    frame["gas_task_main_prespecified_sample"],
                    "observed_gas_above_floor_task_mw",
                ]
                <= frame.loc[
                    frame["gas_task_main_prespecified_sample"], "_raw_gas_mw"
                ]
                + 1e-10
            ).all()
        ),
        "selected_sample_nonempty": bool(len(selected)),
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(args.output.resolve()),
        "floor_definition": (
            "country-year month-by-hour-of-day empirical 10th percentile"
        ),
        "floor_quantile": args.floor_quantile,
        "main_task_definition": (
            "raw non-negative hourly Energy-Charts public_power Fossil gas"
        ),
        "main_sample_rule": (
            "complete raw Fossil gas series AND annual raw gas energy >=1% "
            "of annual system load"
        ),
        "sensitivity_task_definition": (
            "positive part of raw Fossil gas minus month-hour q10 floor"
        ),
        "country_years_total": int(len(annual)),
        "main_prespecified_country_years": int(len(selected)),
        "main_prespecified_countries": int(selected["country"].nunique()),
        "expected_sample_from_hourly_flags": expected_sample,
        "sensitivity_country_years_above_floor_share_ge_1pct": int(
            selected["above_floor_material_share_ge_1pct"].sum()
        ),
        "selected_above_floor_share_of_raw_gas": {
            key: float(value)
            for key, value in selected["above_floor_share_of_raw_gas"]
            .describe()
            .to_dict()
            .items()
        },
        "interpretation": (
            "Aggregate Fossil gas mixes OCGT, CCGT, CHP and other operating "
            "roles. The floor sensitivity reduces persistent output but does "
            "not identify plant type or prove that the remainder is peaking."
        ),
    }
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, indent=2))

    frame = frame.drop(columns=["_month", "_hour", "_raw_gas_mw"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.qa.parent.mkdir(parents=True, exist_ok=True)
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, compression="gzip")
    qa["output_sha256"] = sha256(args.output)
    args.qa.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
