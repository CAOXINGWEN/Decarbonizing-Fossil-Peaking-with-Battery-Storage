#!/usr/bin/env python3
"""Freeze the observed daily task, recharge and event calendar for Figure 2a."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[1]
WORKSPACE = HERE.parents[6]
SOURCE = REVISION / "figure_work" / "source_data_v60"
OUT = HERE / "source_data"

EUROPE = (
    WORKSPACE
    / "dataset"
    / "external_hourly"
    / "europe_mid_duration_valley_upgrade_v0_1"
    / "processed"
    / "observed_gas_event_task_definitions_v0_2.csv.gz"
)
CHINA = REVISION / "cross_scale_cases" / "china_province_panel" / "china_province_event_task_panel.csv.gz"
JAPAN = REVISION / "cross_scale_cases" / "japan_community_panel" / "japan_community_event_task_panel.csv.gz"


def daily_frame(path: Path, regime: str) -> pd.DataFrame:
    if regime == "Europe":
        identifiers = ["country", "year"]
        usecols = identifiers + [
            "source_day_id",
            "task_c1_mw",
            "task_c3_q75_mw",
            "r2_potential_recharge_mw",
            "event_flag_q75",
        ]
    elif regime == "China":
        identifiers = ["province", "year"]
        usecols = identifiers + [
            "source_day_id",
            "task_c1_mw",
            "task_c3_q75_mw",
            "r2_potential_recharge_mw",
            "event_flag_q75",
        ]
    else:
        identifiers = ["prefecture", "mesh_id", "fiscal_year"]
        usecols = identifiers + [
            "source_day_id",
            "task_c1_mw",
            "task_c3_q75_mw",
            "r2_potential_recharge_mw",
            "event_flag_q75",
        ]

    hourly = pd.read_csv(path, usecols=usecols)
    daily = (
        hourly.groupby(identifiers + ["source_day_id"], as_index=False)
        .agg(
            daily_base_task_mwh=("task_c1_mw", "sum"),
            daily_tail_task_mwh=("task_c3_q75_mw", "sum"),
            daily_recharge_mwh=("r2_potential_recharge_mw", "sum"),
            event_flag_q75=("event_flag_q75", "max"),
        )
        .sort_values(identifiers + ["source_day_id"])
    )
    daily["regime"] = regime
    if regime == "Europe":
        daily["system_id"] = daily["country"].astype(str)
        daily["observation_year"] = daily["year"].astype(int)
    elif regime == "China":
        daily["system_id"] = "CN-" + daily["province"].astype(str)
        daily["observation_year"] = daily["year"].astype(int)
    else:
        daily["system_id"] = (
            "JP-"
            + daily["prefecture"].astype(str)
            + "-"
            + daily["mesh_id"].astype("int64").astype(str)
        )
        daily["observation_year"] = daily["fiscal_year"].astype(int)
    return daily[
        [
            "regime",
            "system_id",
            "observation_year",
            "source_day_id",
            "daily_base_task_mwh",
            "daily_tail_task_mwh",
            "daily_recharge_mwh",
            "event_flag_q75",
        ]
    ]


def robust_positive_scale(values: pd.Series) -> float:
    positive = values.to_numpy(float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    return float(np.quantile(positive, 0.95)) if len(positive) else 1.0


def longest_run(flags: pd.Series) -> int:
    values = flags.astype(bool).to_numpy()
    longest = current = 0
    for active in values:
        current = current + 1 if active else 0
        longest = max(longest, current)
    return int(longest)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(SOURCE / "figure2_task_event_records.csv")
    daily = pd.concat(
        [
            daily_frame(EUROPE, "Europe"),
            daily_frame(CHINA, "China"),
            daily_frame(JAPAN, "Japan"),
        ],
        ignore_index=True,
    )
    keys = ["regime", "system_id", "observation_year"]
    metrics = records[
        keys + ["annual_task_percentile", "event_run_days", "event_task_share_pct"]
    ].copy()
    daily = daily.merge(metrics, on=keys, how="left", validate="many_to_one")
    if daily["annual_task_percentile"].isna().any():
        missing = daily.loc[daily["annual_task_percentile"].isna(), keys].drop_duplicates()
        raise ValueError(f"Unmatched system-years: {missing.to_dict('records')}")

    scales = (
        daily.groupby(keys, as_index=False)
        .agg(
            base_task_q95_mwh=("daily_base_task_mwh", robust_positive_scale),
            tail_task_q95_mwh=("daily_tail_task_mwh", robust_positive_scale),
            recharge_q95_mwh=("daily_recharge_mwh", robust_positive_scale),
        )
    )
    daily = daily.merge(scales, on=keys, how="left", validate="many_to_one")
    daily["base_task_display_norm"] = np.clip(
        daily["daily_base_task_mwh"] / daily["base_task_q95_mwh"], 0, 1
    )
    daily["tail_task_display_norm"] = np.clip(
        daily["daily_tail_task_mwh"] / daily["tail_task_q95_mwh"], 0, 1
    )
    daily["recharge_display_norm"] = np.clip(
        daily["daily_recharge_mwh"] / daily["recharge_q95_mwh"], 0, 1
    )

    counts = daily.groupby(keys)["source_day_id"].nunique()
    if not counts.eq(365).all():
        raise ValueError(f"Incomplete operating years: {counts[counts.ne(365)].to_dict()}")
    computed_runs = (
        daily.sort_values(keys + ["source_day_id"])
        .groupby(keys)["event_flag_q75"]
        .apply(longest_run)
    )
    declared_runs = daily.groupby(keys)["event_run_days"].first().astype(int)
    if not computed_runs.equals(declared_runs):
        mismatch = pd.concat([computed_runs.rename("computed"), declared_runs.rename("declared")], axis=1)
        raise ValueError(f"Event-run mismatch: {mismatch[mismatch.computed.ne(mismatch.declared)].to_dict('index')}")

    output_columns = keys + [
        "source_day_id",
        "daily_base_task_mwh",
        "daily_tail_task_mwh",
        "daily_recharge_mwh",
        "event_flag_q75",
        "base_task_display_norm",
        "tail_task_display_norm",
        "recharge_display_norm",
        "annual_task_percentile",
        "event_run_days",
        "event_task_share_pct",
    ]
    daily[output_columns].to_csv(OUT / "figure2a_regime_calendar_daily.csv.gz", index=False)

    qa = {
        "hourly_source_rows": {
            "Europe": 1_156_320,
            "China": 227_760,
            "Japan": 236_520,
        },
        "daily_rows": int(len(daily)),
        "expected_daily_rows": 185 * 365,
        "system_years": daily.groupby("regime")[keys].apply(lambda frame: frame.drop_duplicates().shape[0]).to_dict(),
        "all_system_years_complete_365_days": bool(counts.eq(365).all()),
        "event_run_matches_frozen_figure2_records": bool(computed_runs.equals(declared_runs)),
        "display_normalization": "within-system-year positive-day q95, clipped to [0,1]; raw MWh retained",
    }
    (OUT / "figure2a_regime_calendar_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
