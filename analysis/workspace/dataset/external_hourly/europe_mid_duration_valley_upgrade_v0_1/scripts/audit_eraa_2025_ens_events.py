#!/usr/bin/env python3
"""Audit sparse hourly ENS events in the official ERAA 2025 outputs.

The ENTSO-E files contain only positive ENS observations.  This script restores
contiguous hourly events at bidding-zone and pan-European levels without
filling missing observations with anything other than zero ENS.  Weather
scenarios (WS) are the climate units; forced-outage scenarios (FOS) remain
nested within each WS.

This is an external reliability audit, not a storage-value calculation.  ENS
event duration and energy can show whether multi-day adequacy events exist,
but they do not by themselves establish an ELCC or an economic re-entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RAW = ROOT / "raw" / "eraa_2025" / "extracted"
DEFAULT_PROCESSED = ROOT / "processed"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "research"
THRESHOLDS_H = (4, 8, 12, 24, 48, 72, 96, 168)
# ERAA reports these national/all-island aggregates or aliases alongside
# their component model zones.  The separate hierarchy audit verifies exact
# hourly reconciliation in every target year.  They must be excluded from a
# European arithmetic sum to avoid double counting.
NON_ATOMIC_ZONE_CODES = frozenset(
    {"DK00", "ISEM", "IT00", "LU00", "NO00", "SE00"}
)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small dataframe without the optional ``tabulate`` package."""

    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(headers)
    ) + " |"
    rule = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| "
        + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def eventize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Collapse positive hourly ENS observations into contiguous events."""

    if frame.empty:
        return pd.DataFrame()
    data = frame.sort_values([*group_columns, "timestamp"]).copy()
    gap = (
        data.groupby(group_columns, observed=True)["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    data["_new_event"] = gap.isna() | gap.gt(1.0)
    data["event_id"] = (
        data.groupby(group_columns, observed=True)["_new_event"]
        .cumsum()
        .astype(int)
    )
    events = (
        data.groupby([*group_columns, "event_id"], observed=True)
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            duration_h=("timestamp", "nunique"),
            ens_mwh=("ens_mwh", "sum"),
            peak_ens_mw=("ens_mwh", "max"),
        )
        .reset_index()
    )
    events["span_h"] = (
        (events["end"] - events["start"]).dt.total_seconds() / 3600 + 1
    )
    events["contiguous_qa_pass"] = np.isclose(
        events["duration_h"], events["span_h"], atol=1e-9
    )
    return events


def parse_one(path: Path, target_year: int) -> tuple[pd.DataFrame, int]:
    data = pd.read_csv(path)
    required = {"WS", "FOS", "Date", "Bidding Zone", "ENS (MWh)"}
    absent = required.difference(data.columns)
    if absent:
        raise ValueError(f"{path} missing columns: {sorted(absent)}")
    data = data.rename(
        columns={
            "WS": "weather_scenario",
            "FOS": "forced_outage_scenario",
            "Bidding Zone": "bidding_zone",
            "ENS (MWh)": "ens_mwh",
        }
    )
    data["timestamp"] = pd.to_datetime(
        data.pop("Date"), dayfirst=True, errors="raise"
    )
    data["target_year"] = target_year
    if (data["ens_mwh"] < 0).any():
        raise ValueError(f"{path} contains negative ENS rows")
    zero_rows = int(data["ens_mwh"].eq(0).sum())
    data = data.loc[data["ens_mwh"].gt(0)].copy()
    duplicated = data.duplicated(
        [
            "weather_scenario",
            "forced_outage_scenario",
            "timestamp",
            "bidding_zone",
        ]
    )
    if duplicated.any():
        raise ValueError(f"{path} has {int(duplicated.sum())} duplicate keys")
    return (
        data[
            [
                "target_year",
                "weather_scenario",
                "forced_outage_scenario",
                "timestamp",
                "bidding_zone",
                "ens_mwh",
            ]
        ],
        zero_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2028, 2030, 2033, 2035]
    )
    args = parser.parse_args()
    args.processed.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)

    hourly_parts = []
    zero_rows_by_year: dict[str, int] = {}
    for year in args.years:
        path = args.raw / f"ENS_hourly_data_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        parsed, zero_rows = parse_one(path, year)
        hourly_parts.append(parsed)
        zero_rows_by_year[str(year)] = zero_rows
    hourly = pd.concat(hourly_parts, ignore_index=True)

    reported_zone_events = eventize(
        hourly,
        [
            "target_year",
            "weather_scenario",
            "forced_outage_scenario",
            "bidding_zone",
        ],
    )
    atomic_hourly = hourly.loc[
        ~hourly["bidding_zone"].isin(NON_ATOMIC_ZONE_CODES)
    ].copy()
    zone_events = eventize(
        atomic_hourly,
        [
            "target_year",
            "weather_scenario",
            "forced_outage_scenario",
            "bidding_zone",
        ],
    )
    pan_hourly = (
        atomic_hourly.groupby(
            [
                "target_year",
                "weather_scenario",
                "forced_outage_scenario",
                "timestamp",
            ],
            observed=True,
            as_index=False,
        )["ens_mwh"]
        .sum()
        .assign(system="pan_europe")
    )
    pan_events = eventize(
        pan_hourly,
        [
            "target_year",
            "weather_scenario",
            "forced_outage_scenario",
            "system",
        ],
    )

    zone_events.to_csv(
        args.processed / "eraa_2025_zone_ens_events.csv.gz",
        index=False,
        compression="gzip",
    )
    zone_events.to_csv(
        args.processed / "eraa_2025_atomic_zone_ens_events.csv.gz",
        index=False,
        compression="gzip",
    )
    reported_zone_events.to_csv(
        args.processed / "eraa_2025_reported_zone_ens_events.csv.gz",
        index=False,
        compression="gzip",
    )
    pan_events.to_csv(
        args.processed / "eraa_2025_pan_europe_ens_events.csv.gz",
        index=False,
        compression="gzip",
    )
    zone_summary = (
        zone_events.groupby(
            ["target_year", "bidding_zone"], observed=True
        )
        .agg(
            event_count=("event_id", "size"),
            total_ens_gwh=("ens_mwh", lambda values: values.sum() / 1e3),
            nested_simulations_with_ens=(
                "forced_outage_scenario",
                lambda values: int(
                    zone_events.loc[values.index][
                        ["weather_scenario", "forced_outage_scenario"]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
            ),
            weather_scenarios_with_ens=("weather_scenario", "nunique"),
            median_event_duration_h=("duration_h", "median"),
            p95_event_duration_h=(
                "duration_h",
                lambda values: values.quantile(0.95),
            ),
            maximum_event_duration_h=("duration_h", "max"),
        )
        .reset_index()
    )
    for threshold in THRESHOLDS_H:
        counts = (
            zone_events.loc[zone_events["duration_h"].ge(threshold)]
            .groupby(["target_year", "bidding_zone"], observed=True)
            .size()
            .rename(f"events_ge_{threshold}h")
            .reset_index()
        )
        zone_summary = zone_summary.merge(
            counts,
            on=["target_year", "bidding_zone"],
            how="left",
            validate="one_to_one",
        )
        zone_summary[f"events_ge_{threshold}h"] = (
            zone_summary[f"events_ge_{threshold}h"].fillna(0).astype(int)
        )
    zone_summary.to_csv(
        args.artifacts / "eraa_2025_ens_atomic_zone_summary.csv",
        index=False,
    )

    rows: list[dict] = []
    for year, group in pan_events.groupby("target_year", observed=True):
        weather_scenarios = int(
            hourly.loc[
                hourly["target_year"].eq(year), "weather_scenario"
            ].nunique()
        )
        forced_outage_scenarios = int(
            hourly.loc[
                hourly["target_year"].eq(year),
                "forced_outage_scenario",
            ].nunique()
        )
        total_simulations = weather_scenarios * forced_outage_scenarios
        simulation_keys = group[
            ["weather_scenario", "forced_outage_scenario"]
        ].drop_duplicates()
        ens_sum_gwh = float(group["ens_mwh"].sum() / 1e3)
        row: dict[str, int | float] = {
            "target_year": int(year),
            "weather_scenarios": weather_scenarios,
            "forced_outage_scenarios": forced_outage_scenarios,
            "total_nested_simulations": total_simulations,
            "simulations_with_ens": int(len(simulation_keys)),
            "simulations_with_ens_share": float(
                len(simulation_keys) / total_simulations
            ),
            "pan_europe_events": int(len(group)),
            "ens_sum_across_simulations_gwh": ens_sum_gwh,
            "mean_ens_gwh_per_nested_simulation": float(
                ens_sum_gwh / total_simulations
            ),
            "median_event_duration_h": float(group["duration_h"].median()),
            "p95_event_duration_h": float(
                group["duration_h"].quantile(0.95)
            ),
            "maximum_event_duration_h": int(group["duration_h"].max()),
        }
        for threshold in THRESHOLDS_H:
            row[f"events_ge_{threshold}h"] = int(
                group["duration_h"].ge(threshold).sum()
            )
            qualifying = group.loc[
                group["duration_h"].ge(threshold),
                ["weather_scenario", "forced_outage_scenario"],
            ].drop_duplicates()
            row[f"simulations_with_event_ge_{threshold}h"] = int(
                len(qualifying)
            )
            row[f"weather_scenarios_with_event_ge_{threshold}h"] = int(
                qualifying["weather_scenario"].nunique()
            )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("target_year")
    summary.to_csv(
        args.artifacts / "eraa_2025_ens_event_summary.csv", index=False
    )

    by_ws = (
        pan_events.groupby(
            ["target_year", "weather_scenario"], observed=True
        )
        .agg(
            forced_outage_scenarios_with_ens=(
                "forced_outage_scenario",
                "nunique",
            ),
            event_count=("event_id", "size"),
            total_ens_gwh=("ens_mwh", lambda values: values.sum() / 1e3),
            maximum_event_duration_h=("duration_h", "max"),
            p95_event_duration_h=(
                "duration_h",
                lambda values: values.quantile(0.95),
            ),
        )
        .reset_index()
    )
    by_ws.to_csv(
        args.artifacts / "eraa_2025_ens_by_weather_scenario.csv", index=False
    )

    qa = {
        "source_files": [
            str(args.raw / f"ENS_hourly_data_{year}.csv")
            for year in args.years
        ],
        "positive_hourly_rows": int(len(hourly)),
        "atomic_positive_hourly_rows": int(len(atomic_hourly)),
        "excluded_non_atomic_zone_codes": sorted(NON_ATOMIC_ZONE_CODES),
        "reported_zone_events": int(len(reported_zone_events)),
        "explicit_zero_rows_dropped_by_year": zero_rows_by_year,
        "weather_scenarios_by_year": {
            str(year): int(group["weather_scenario"].nunique())
            for year, group in hourly.groupby("target_year", observed=True)
        },
        "forced_outage_scenarios_by_year": {
            str(year): int(group["forced_outage_scenario"].nunique())
            for year, group in hourly.groupby("target_year", observed=True)
        },
        "zone_events": int(len(zone_events)),
        "pan_europe_events": int(len(pan_events)),
        "zone_event_contiguity_all_passed": bool(
            zone_events["contiguous_qa_pass"].all()
        ),
        "pan_event_contiguity_all_passed": bool(
            pan_events["contiguous_qa_pass"].all()
        ),
        "interpretation": (
            "Weather scenarios are climate units and forced-outage scenarios "
            "are nested replicates. European totals retain only atomic ERAA "
            "model zones; verified aggregate and alias codes are excluded. "
            "Event durations are external adequacy diagnostics, not storage "
            "ELCC or economic-value estimates."
        ),
    }
    (args.artifacts / "eraa_2025_ens_event_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ERAA 2025 逐时未供能事件审计",
        "",
        (
            "该审计使用 ENTSO-E 发布的稀疏正值 ENS 结果，将相邻小时恢复为"
            "原子竞价区与泛欧洲事件。泛欧洲总量剔除了已逐时核验的 6 个聚合"
            "或别名代码，避免重复计算。天气情景是气候统计单元，强迫停运情景"
            "嵌套于天气情景。"
        ),
        "",
        frame_to_markdown(summary),
        "",
        (
            "这些事件证明未来欧洲可靠性评估中是否存在多小时或多日供能缺口，"
            "但不能单独给出储能 ELCC、系统成本或中时长谷。它们用于选择可靠性"
            "压力情景并校验后续联网储能模型。"
        ),
        "",
    ]
    (args.artifacts / "ERAA_2025_ENS_EVENT_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
