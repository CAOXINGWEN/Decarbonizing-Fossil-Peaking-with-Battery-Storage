#!/usr/bin/env python3
"""Build current-ledger source data for Extended Data Figure 7."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
REVISION = HERE.parent.parent
SURFACE = REVISION / "cluster_intensity_surface/source_data/v60_cluster_intensity_system_year.csv.gz"
EVENTS = REVISION / "figure_work/source_data_v60/figure2_task_event_records.csv"
OUT = HERE / "source_data"

REGIMES = ["Europe", "China", "Japan"]
CLUSTERS = [2, 4, 6]


def first_sustained_positive(frame: pd.DataFrame) -> dict[str, float | str]:
    """Return the first post-trough duration with a sustained positive effect."""
    curve = (
        frame.groupby("duration_h", as_index=False)["sequencing_effect_eur_kw_year"]
        .max()
        .sort_values("duration_h")
    )
    durations = curve["duration_h"].to_numpy(float)
    effects = curve["sequencing_effect_eur_kw_year"].to_numpy(float)
    if np.all(effects > 0):
        status = "left_censored"
        crossing = float(durations.min())
    elif np.all(effects <= 0):
        status = "right_censored"
        crossing = np.nan
    else:
        status = "right_censored"
        crossing = np.nan
        trough = int(np.argmin(effects))
        for index in range(trough + 1, len(durations)):
            if effects[index] > 0 and np.all(effects[index:] > 0):
                status = "observed"
                crossing = float(durations[index])
                break
    return {
        "crossing_status": status,
        "crossing_duration_h": crossing,
        "plot_duration_h": (
            float(durations.max()) if status == "right_censored" else crossing
        ),
        "minimum_effect_eur_kw_year": float(effects.min()),
        "minimum_effect_duration_h": float(durations[int(np.argmin(effects))]),
        "maximum_tested_duration_h": float(durations.max()),
        "tested_duration_count": int(len(durations)),
    }


def build_records(surface: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    selected = surface.loc[
        surface["regime"].isin(REGIMES)
        & surface["target_cluster_days"].isin(CLUSTERS)
    ].copy()
    rows: list[dict[str, object]] = []
    keys = ["regime", "target_cluster_days", "system_id", "observation_year"]
    for key, frame in selected.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update(first_sustained_positive(frame))
        rows.append(row)
    records = pd.DataFrame(rows)
    records = records.merge(
        events[
            [
                "regime",
                "system_id",
                "observation_year",
                "event_run_days",
                "event_task_share_pct",
                "annual_task_percentile",
            ]
        ],
        on=["regime", "system_id", "observation_year"],
        how="left",
        validate="many_to_one",
    )
    records["system_year_id"] = (
        records["system_id"].astype(str)
        + "-"
        + records["observation_year"].astype(str)
    )
    return records


def build_status(records: pd.DataFrame) -> pd.DataFrame:
    status = (
        records.groupby(["regime", "target_cluster_days", "crossing_status"])
        .size()
        .rename("system_years")
        .reset_index()
    )
    totals = status.groupby(["regime", "target_cluster_days"])["system_years"].transform("sum")
    status["share_pct"] = 100 * status["system_years"] / totals
    return status


def build_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, cluster), frame in records.groupby(["regime", "target_cluster_days"]):
        observed = frame.loc[frame["crossing_status"].eq("observed"), "crossing_duration_h"]
        rows.append(
            {
                "regime": regime,
                "target_cluster_days": cluster,
                "system_years": int(len(frame)),
                "observed_crossings": int(len(observed)),
                "left_censored": int(frame["crossing_status"].eq("left_censored").sum()),
                "right_censored": int(frame["crossing_status"].eq("right_censored").sum()),
                "median_crossing_duration_h": float(observed.median()) if len(observed) else np.nan,
                "q25_crossing_duration_h": float(observed.quantile(0.25)) if len(observed) else np.nan,
                "q75_crossing_duration_h": float(observed.quantile(0.75)) if len(observed) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_correlations(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    four_day = records.loc[
        records["target_cluster_days"].eq(4)
        & records["crossing_status"].eq("observed")
    ]
    for regime, frame in four_day.groupby("regime"):
        rho, p_value = spearmanr(frame["event_run_days"], frame["crossing_duration_h"])
        rows.append(
            {
                "regime": regime,
                "observed_crossings": int(len(frame)),
                "spearman_rho": float(rho),
                "p_value_two_sided": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    surface = pd.read_csv(SURFACE)
    events = pd.read_csv(EVENTS)
    records = build_records(surface, events)
    status = build_status(records)
    summary = build_summary(records)
    correlations = build_correlations(records)

    expected = {"Europe": 132, "China": 26, "Japan": 27}
    counts = records.groupby(["regime", "target_cluster_days"]).size()
    for regime, count in expected.items():
        for cluster in CLUSTERS:
            assert counts.loc[(regime, cluster)] == count
    assert records["event_run_days"].notna().all()
    assert set(records["crossing_status"]) <= {
        "left_censored",
        "observed",
        "right_censored",
    }

    records.to_csv(OUT / "ed7_cross_duration_records.csv", index=False)
    status.to_csv(OUT / "ed7_crossing_status.csv", index=False)
    summary.to_csv(OUT / "ed7_crossing_summary.csv", index=False)
    correlations.to_csv(OUT / "ed7_four_day_correlations.csv", index=False)
    qa = {
        "ledger": "objective-aligned system-cost sequencing effect",
        "record_count": int(len(records)),
        "system_years": expected,
        "cluster_targets_days": CLUSTERS,
        "duration_nodes_h": sorted(surface["duration_h"].unique().astype(float).tolist()),
        "all_event_geometry_matched": bool(records["event_run_days"].notna().all()),
        "status_counts": status.to_dict(orient="records"),
        "four_day_correlations": correlations.to_dict(orient="records"),
    }
    (OUT / "ed7_source_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\n", correlations.to_string(index=False))


if __name__ == "__main__":
    main()
