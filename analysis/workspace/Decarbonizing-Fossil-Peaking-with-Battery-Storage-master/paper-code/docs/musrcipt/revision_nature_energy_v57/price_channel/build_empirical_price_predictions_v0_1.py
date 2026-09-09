#!/usr/bin/env python3
"""Fit country-specific residual-load to day-ahead-price curves for S3a."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[5]
DATA = WORKSPACE / "dataset" / "external_hourly" / "europe_mid_duration_valley_upgrade_v0_1"
PRICE = DATA / "raw" / "energy_charts" / "price_bzn_valid_v0_1" / "energy_charts_country_price_sensitivity_2019_2024.csv.gz"
SYSTEM = DATA / "processed" / "energy_charts_europe_hourly_eligible_2015_2024.csv.gz"
MAIN = DATA / "processed" / "observed_gas_event_task_definitions_v0_2.csv.gz"
OUT = HERE / "empirical_price"


def normalize_within_year(frame: pd.DataFrame) -> pd.Series:
    q05 = frame.residual_load_mw.quantile(0.05)
    q95 = frame.residual_load_mw.quantile(0.95)
    if q95 <= q05:
        return pd.Series(np.zeros(len(frame)), index=frame.index)
    return ((frame.residual_load_mw - q05) / (q95 - q05)).clip(0.0, 1.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE, parse_dates=["timestamp_utc"])
    system = pd.read_csv(SYSTEM, usecols=["timestamp_utc", "country", "year", "load_mw", "wind_mw", "solar_mw"], parse_dates=["timestamp_utc"])
    system["residual_load_mw"] = (system.load_mw - system.wind_mw - system.solar_mw).clip(lower=0.0)
    system["residual_rank"] = system.groupby(["country", "year"], group_keys=False).apply(normalize_within_year)
    training = system.merge(prices[["country", "timestamp_utc", "price_eur_mwh"]], on=["country", "timestamp_utc"], how="inner", validate="one_to_one")
    training = training[training.year.ne(2022) & training.price_eur_mwh.notna()].copy()
    curve_rows, models = [], {}
    for country, group in training.groupby("country"):
        low, high = group.price_eur_mwh.quantile([0.01, 0.99])
        y = group.price_eur_mwh.clip(low, high).to_numpy(float)
        x = group.residual_rank.to_numpy(float)
        model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
        grid = np.linspace(0.0, 1.0, 21)
        prediction = model.predict(grid)
        models[country] = (model, float(low), float(high))
        curve_rows.extend(
            {"country": country, "residual_rank": rank, "predicted_price_eur_mwh": value, "price_p01": low, "price_p99": high, "training_hours": len(group)}
            for rank, value in zip(grid, prediction)
        )
    main = pd.read_csv(MAIN, usecols=["timestamp_utc", "country", "year", "source_hour_id"], parse_dates=["timestamp_utc"])
    features = system[["timestamp_utc", "country", "year", "residual_rank"]]
    main = main.merge(features, on=["timestamp_utc", "country", "year"], how="left", validate="one_to_one")
    output = []
    for country, group in main.groupby("country"):
        if country not in models:
            continue
        model, low, high = models[country]
        predicted = np.clip(model.predict(group.residual_rank.to_numpy(float)), low, high)
        part = group[["country", "year", "source_hour_id"]].copy()
        part["predicted_price_eur_mwh"] = predicted
        output.append(part)
    predictions = pd.concat(output, ignore_index=True)
    predictions.to_csv(OUT / "empirical_price_predictions_main_panel.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(curve_rows).to_csv(OUT / "residual_load_price_curves.csv", index=False)
    coverage = predictions.groupby(["country", "year"], as_index=False).agg(hours=("source_hour_id", "size"), mean_predicted_price=("predicted_price_eur_mwh", "mean"))
    coverage.to_csv(OUT / "empirical_price_prediction_coverage.csv", index=False)
    qa = {
        "training_excludes_year": 2022,
        "training_countries": sorted(models),
        "training_hours": len(training),
        "predicted_country_years": len(coverage),
        "all_predicted_country_years_8760": bool(coverage.hours.eq(8760).all()),
        "curve": "country-specific increasing isotonic piecewise function of within-country-year residual-load rank",
        "price_winsorization": "country-specific 1st--99th percentile",
        "dispatch_reoptimized": False,
    }
    (OUT / "qa.json").write_text(json.dumps(qa, indent=2) + "\n")


if __name__ == "__main__":
    main()
