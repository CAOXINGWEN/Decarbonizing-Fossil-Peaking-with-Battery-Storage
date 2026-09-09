#!/usr/bin/env python3
"""Aggregate full-sample duration-by-cluster-intensity results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs_objective_aligned"
OUT = HERE / "source_data"
PARITY_THRESHOLD = 5.2960380996609935
REGIME_ORDER = ("Europe", "China", "Japan")
EXPECTED_SYSTEM_YEARS = {"Europe": 132, "China": 26, "Japan": 27}
EXPECTED_BLOCKS = {"Europe": 15, "China": 26, "Japan": 5}
EXPECTED_CLUSTER_DAYS = tuple(range(1, 13))


def block_bootstrap(
    frame: pd.DataFrame,
    value: str,
    seed: int,
    draws: int = 10_000,
) -> tuple[float, float, float]:
    values = frame.groupby("inference_block", sort=True)[value].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(estimates, [0.025, 0.975]).tolist()


def first_sustained_positive(frame: pd.DataFrame) -> float:
    ordered = frame.sort_values("duration_h")
    durations = ordered["duration_h"].to_numpy(float)
    effects = ordered["block_mean_effect_eur_kw_year"].to_numpy(float)
    for index, duration in enumerate(durations):
        if effects[index] > 0 and np.all(effects[index:] > 0):
            return float(duration)
    return np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result_paths = sorted(RUNS.glob("*/system_duration_effects.csv.gz"))
    qa_paths = sorted(RUNS.glob("*/qa.json"))
    if not result_paths:
        raise RuntimeError("No completed surface runs found")
    shard_qa = [json.loads(path.read_text(encoding="utf-8")) for path in qa_paths]
    if len(shard_qa) != len(result_paths) or any(item["status"] != "pass" for item in shard_qa):
        raise RuntimeError("One or more shard QA records are missing or failed")
    records = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    records = records.sort_values(
        ["regime", "system_id", "observation_year", "target_cluster_days", "duration_h"]
    ).reset_index(drop=True)
    output_records = OUT / "v60_cluster_intensity_system_year.csv.gz"
    records.to_csv(
        output_records,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    summary_rows: list[dict[str, float | int | str]] = []
    for regime_index, regime in enumerate(REGIME_ORDER):
        regime_frame = records.loc[records["regime"].eq(regime)]
        for (cluster_days, duration_h), frame in regime_frame.groupby(
            ["target_cluster_days", "duration_h"], sort=True
        ):
            point, low, high = block_bootstrap(
                frame,
                "sequencing_effect_eur_kw_year",
                seed=600_000 + regime_index * 10_000 + int(cluster_days) * 100 + int(duration_h),
            )
            summary_rows.append(
                {
                    "regime": regime,
                    "target_cluster_days": int(cluster_days),
                    "duration_h": float(duration_h),
                    "block_mean_effect_eur_kw_year": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "system_year_median_effect_eur_kw_year": float(
                        frame["sequencing_effect_eur_kw_year"].median()
                    ),
                    "median_absolute_shift_eur_kw_year": float(
                        frame["absolute_sequencing_shift_eur_kw_year"].median()
                    ),
                    "parity_exposure_share": float(frame["reaches_cost_scale"].mean()),
                    "n_system_years": int(len(frame)),
                    "n_inference_blocks": int(frame["inference_block"].nunique()),
                }
            )
    surface = pd.DataFrame(summary_rows)
    surface["signed_parity_ratio"] = (
        surface["block_mean_effect_eur_kw_year"] / PARITY_THRESHOLD
    )
    surface.to_csv(OUT / "v60_cluster_intensity_surface.csv", index=False)

    intensity = (
        records.groupby(["regime", "target_cluster_days"], sort=True)
        .agg(
            median_zero_window_share=("zero_recovery_window_share_clustered", "median"),
            mean_zero_window_share=("zero_recovery_window_share_clustered", "mean"),
            median_recovery_hours=("median_recovery_hours_clustered", "median"),
            median_longest_run_days=("longest_event_run_days_clustered", "median"),
            n_system_years=("system_id", "size"),
        )
        .reset_index()
    )
    intensity["n_system_years"] = intensity["n_system_years"] // surface["duration_h"].nunique()
    intensity.to_csv(OUT / "v60_cluster_intensity_axis.csv", index=False)

    boundary_rows: list[dict[str, float | int | str]] = []
    for (regime, cluster_days), frame in surface.groupby(
        ["regime", "target_cluster_days"], sort=True
    ):
        frame = frame.sort_values("duration_h")
        trough = frame.loc[frame["block_mean_effect_eur_kw_year"].idxmin()]
        negative_parity = frame.loc[
            frame["block_mean_effect_eur_kw_year"].le(-PARITY_THRESHOLD)
        ]
        positive_parity = frame.loc[
            frame["block_mean_effect_eur_kw_year"].ge(PARITY_THRESHOLD)
        ]
        axis = intensity.loc[
            intensity["regime"].eq(regime)
            & intensity["target_cluster_days"].eq(cluster_days)
        ].iloc[0]
        boundary_rows.append(
            {
                "regime": regime,
                "target_cluster_days": int(cluster_days),
                "median_zero_window_share": float(axis["median_zero_window_share"]),
                "trough_duration_h": float(trough["duration_h"]),
                "trough_effect_eur_kw_year": float(trough["block_mean_effect_eur_kw_year"]),
                "first_sustained_positive_duration_h": first_sustained_positive(frame),
                "negative_parity_min_duration_h": (
                    float(negative_parity["duration_h"].min()) if len(negative_parity) else np.nan
                ),
                "negative_parity_max_duration_h": (
                    float(negative_parity["duration_h"].max()) if len(negative_parity) else np.nan
                ),
                "positive_parity_min_duration_h": (
                    float(positive_parity["duration_h"].min()) if len(positive_parity) else np.nan
                ),
                "duration_nodes_reaching_absolute_parity": int(
                    frame["block_mean_effect_eur_kw_year"].abs().ge(PARITY_THRESHOLD).sum()
                ),
            }
        )
    boundaries = pd.DataFrame(boundary_rows)
    boundaries.to_csv(OUT / "v60_cluster_intensity_boundaries.csv", index=False)

    comparison_rows: list[dict[str, float | str]] = []
    baseline = pd.read_csv(
        ROOT
        / "figure_work/source_data_v58/harmonized_parity/three_regime_system_year_duration_records.csv.gz"
    )
    current_four = records.loc[records["target_cluster_days"].eq(4)]
    matched = current_four.merge(
        baseline,
        on=["regime", "system_id", "observation_year", "duration_h"],
        how="inner",
        validate="one_to_one",
    )
    for regime, frame in matched.groupby("regime"):
        difference = (
            frame["sequencing_effect_eur_kw_year"]
            - frame["signed_sequencing_effect_eur_kw_year"]
        )
        comparison_rows.append(
            {
                "comparison": "objective_aligned_surface_vs_harmonized_4d",
                "regime": regime,
                "maximum_absolute_difference_eur_kw_year": float(difference.abs().max()),
                "rmse_eur_kw_year": float(np.sqrt(np.mean(np.square(difference)))),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "v60_cluster_intensity_reproduction.csv", index=False)

    expected_rows = sum(EXPECTED_SYSTEM_YEARS.values()) * len(EXPECTED_CLUSTER_DAYS) * int(
        surface["duration_h"].nunique()
    )
    monotonic = intensity.groupby("regime")["median_zero_window_share"].apply(
        lambda values: bool(np.all(np.diff(values.to_numpy(float)) >= -1.0e-12))
    )
    qa = {
        "status": "pass",
        "result_shards": len(result_paths),
        "expected_system_duration_rows": expected_rows,
        "actual_system_duration_rows": int(len(records)),
        "regime_system_years": {
            regime: int(
                records.loc[records["regime"].eq(regime), ["system_id", "observation_year"]]
                .drop_duplicates()
                .shape[0]
            )
            for regime in REGIME_ORDER
        },
        "regime_inference_blocks": {
            regime: int(records.loc[records["regime"].eq(regime), "inference_block"].nunique())
            for regime in REGIME_ORDER
        },
        "cluster_days": sorted(records["target_cluster_days"].unique().astype(int).tolist()),
        "durations_h": sorted(records["duration_h"].unique().astype(float).tolist()),
        "maximum_absolute_cluster1_effect_eur_kw_year": float(
            records.loc[records["target_cluster_days"].eq(1), "sequencing_effect_eur_kw_year"]
            .abs()
            .max()
        ),
        "intensity_monotonic_by_regime": monotonic.to_dict(),
        "all_effects_finite": bool(np.isfinite(records["sequencing_effect_eur_kw_year"]).all()),
        "maximum_registered_reproduction_difference_eur_kw_year": float(
            comparison["maximum_absolute_difference_eur_kw_year"].max()
        ),
        "parity_threshold_eur_kw_year": PARITY_THRESHOLD,
        "value_metric": "exact solver-system-cost difference versus additional 4 h LFP",
        "efc_role": "operational diagnostic only",
    }
    if qa["actual_system_duration_rows"] != qa["expected_system_duration_rows"]:
        qa["status"] = "fail"
    if qa["regime_system_years"] != EXPECTED_SYSTEM_YEARS:
        qa["status"] = "fail"
    if qa["regime_inference_blocks"] != EXPECTED_BLOCKS:
        qa["status"] = "fail"
    if qa["cluster_days"] != list(EXPECTED_CLUSTER_DAYS):
        qa["status"] = "fail"
    if qa["maximum_absolute_cluster1_effect_eur_kw_year"] > 1.0e-9:
        qa["status"] = "fail"
    if qa["maximum_registered_reproduction_difference_eur_kw_year"] > 1.0e-6:
        qa["status"] = "fail"
    if not all(qa["intensity_monotonic_by_regime"].values()) or not qa["all_effects_finite"]:
        qa["status"] = "fail"
    (OUT / "qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(surface.to_string(index=False))
    print("\n", boundaries.to_string(index=False))
    print("\n", comparison.to_string(index=False))
    print("\n", json.dumps(qa, indent=2))
    if qa["status"] != "pass":
        raise RuntimeError(qa)


if __name__ == "__main__":
    main()
