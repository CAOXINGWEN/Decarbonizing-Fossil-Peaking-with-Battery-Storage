#!/usr/bin/env python3
"""Rebuild v43 main-figure bootstrap sources from country-year pairs.

This script preserves the manuscript estimand: observations are first averaged
within country across available years and countries are then weighted equally.
The uncertainty interval resamples countries with replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V43 = HERE.parent
DEFAULT_INPUT = (
    V43 / "source_data" / "v42_fixed_scale_mix_envelope_paired_effects.csv"
)
DEFAULT_OUTPUT = (
    V43 / "source_data" / "v43_figure5b_duration_envelope_bootstrap_10000.csv"
)
DEFAULT_AUDIT = (
    V43 / "source_data" / "v43_figure5b_duration_envelope_bootstrap_audit.json"
)
METRIC = (
    "clustered_minus_dispersed_"
    "best_mix_benefit_eur_per_kw_common_power_year"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_equal_country_mean(
    group: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    country_values = (
        group.groupby("country", sort=True)[METRIC].mean().dropna().to_numpy(float)
    )
    if country_values.size == 0:
        return np.nan, np.nan, np.nan, 0
    point = float(country_values.mean())
    indices = rng.integers(
        0,
        country_values.size,
        size=(draws, country_values.size),
    )
    estimates = country_values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high), int(country_values.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.draws <= 0:
        raise ValueError("--draws must be positive")

    source = pd.read_csv(args.input)
    required = {
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
        "duration_h",
        METRIC,
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise RuntimeError(f"Missing paired-result columns: {missing}")
    source = source[
        source["source_label"].eq("fixed_s20_central")
        & source["model_mode"].eq("fixed")
        & source["valuation_scenario_id"].eq(20)
    ].copy()
    if source.empty:
        raise RuntimeError("Central S20 fixed-scale paired results are absent")
    duplicate = source.duplicated(["country", "year", "duration_h"])
    if duplicate.any():
        raise RuntimeError("Duplicate country-year-duration paired effects")

    groups = list(source.groupby("duration_h", sort=True))
    children = np.random.SeedSequence(args.seed).spawn(len(groups))
    rows: list[dict[str, object]] = []
    for (duration_h, group), child in zip(groups, children):
        point, low, high, n_countries = bootstrap_equal_country_mean(
            group,
            args.draws,
            np.random.default_rng(child),
        )
        rows.append(
            {
                "analysis": "duration_envelope_paired_effect",
                "source_label": "fixed_s20_central",
                "model_mode": "fixed",
                "valuation_scenario_id": 20,
                "duration_h": float(duration_h),
                "metric": METRIC,
                "country_aggregation": "mean",
                "bootstrap_draws": args.draws,
                "n_country_years": int(
                    group[["country", "year"]].drop_duplicates().shape[0]
                ),
                "point_estimate": point,
                "ci_low": low,
                "ci_high": high,
                "n_countries": n_countries,
                "bootstrap_independent_block": "country",
                "within_country_aggregation": "mean_across_available_years",
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != source["duration_h"].nunique():
        raise RuntimeError("Incomplete duration bootstrap output")
    if not output["n_country_years"].eq(132).all():
        raise RuntimeError("Expected 132 country-years at every duration")
    if not output["n_countries"].eq(15).all():
        raise RuntimeError("Expected 15 independent country blocks")
    if not output["bootstrap_draws"].eq(args.draws).all():
        raise RuntimeError("Bootstrap draw metadata mismatch")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    audit = {
        "analysis_id": "v43_figure5b_duration_envelope_bootstrap",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "estimand": (
            "country-equal mean of clustered minus dispersed best incremental "
            "portfolio value at each LDES duration"
        ),
        "independent_block": "country",
        "within_country_aggregation": "mean_across_available_years",
        "bootstrap_draws": args.draws,
        "seed": args.seed,
        "n_country_years": int(
            source[["country", "year"]].drop_duplicates().shape[0]
        ),
        "n_countries": int(source["country"].nunique()),
        "n_durations": int(source["duration_h"].nunique()),
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256(args.input),
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
        },
    }
    args.audit.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
