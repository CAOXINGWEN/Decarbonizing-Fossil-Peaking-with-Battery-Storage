#!/usr/bin/env python3
"""Freeze the v43 24 h A-CAES focal statistics with 10,000 country blocks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V43 = HERE.parent
RUN = V43 / "portfolio_model" / "full_132cy_fixed_s20_central"
LONG_PATH = RUN / "ldes_vs_extra_lfp4_results.csv.gz"
ARM_PATH = RUN / "portfolio_arm_results.csv.gz"
PAIRED_PATH = RUN / "clustered_vs_dispersed_paired_effects.csv"
CHANNEL_PATH = (
    V43
    / "cost_and_decomposition"
    / "common_config_channel_decomposition_country_year.csv.gz"
)
OUTPUT = V43 / "source_data" / "v43_focal24_country_block_summary_10000.csv"
MANIFEST = V43 / "source_data" / "v43_focal24_country_block_manifest.json"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260729


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def country_values(
    frame: pd.DataFrame, value: str, operation: str = "mean"
) -> pd.Series:
    grouped = frame.groupby("country")[value]
    if operation == "mean":
        return grouped.mean().sort_index()
    if operation == "median":
        return grouped.median().sort_index()
    raise ValueError(operation)


def summarize_country_blocks(
    values: pd.Series, metric: str, unit: str
) -> dict[str, object]:
    array = values.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(array), size=(BOOTSTRAP_DRAWS, len(array))
    )
    draws = array[indices].mean(axis=1)
    return {
        "metric": metric,
        "unit": unit,
        "country_block_mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "n_countries": int(len(array)),
    }


def main() -> None:
    long = pd.read_csv(LONG_PATH)
    arms = pd.read_csv(ARM_PATH)
    paired = pd.read_csv(PAIRED_PATH)
    channel = pd.read_csv(CHANNEL_PATH)
    caes = long[
        long["arm_type"].eq("ldes_candidate")
        & long["config_id"].eq("caes_24h")
    ].copy()
    lfp = arms[arms["arm_type"].eq("extra_lfp4_comparator")].copy()
    caes_pair = paired[
        paired["arm_type"].eq("ldes_candidate")
        & paired["config_id"].eq("caes_24h")
    ].copy()
    channel24 = channel[channel["duration_h"].eq(24)].copy()
    if (
        caes[["country", "year"]].drop_duplicates().shape[0] != 132
        or lfp[["country", "year"]].drop_duplicates().shape[0] != 132
        or caes_pair[["country", "year"]].drop_duplicates().shape[0] != 132
        or channel24[["country", "year"]].drop_duplicates().shape[0] != 132
    ):
        raise RuntimeError("The focal statistics do not cover all 132 country-years")

    rows: list[dict[str, object]] = []
    for chronology in ["dispersed", "clustered"]:
        caes_chronology = caes[caes["chronology"].eq(chronology)]
        lfp_chronology = lfp[lfp["chronology"].eq(chronology)]
        for column, label, unit in [
            ("tail_equivalent_full_cycles", "caes_tail_cycles", "cycles yr-1"),
            ("tail_charge_mwh", "caes_tail_charge", "MWh yr-1"),
            ("tail_discharge_mwh", "caes_tail_discharge", "MWh yr-1"),
            (
                "shared_recharge_binding_hours",
                "caes_shared_recharge_binding",
                "h yr-1",
            ),
            (
                "tail_charge_power_binding_hours",
                "caes_charge_interface_binding",
                "h yr-1",
            ),
        ]:
            rows.append(
                summarize_country_blocks(
                    country_values(caes_chronology, column),
                    f"{label}_{chronology}",
                    unit,
                )
            )
        rows.append(
            summarize_country_blocks(
                country_values(
                    lfp_chronology, "tail_equivalent_full_cycles"
                ),
                f"lfp_tail_cycles_{chronology}",
                "cycles yr-1",
            )
        )

    for source, column, label, unit in [
        (
            caes_pair,
            "clustered_minus_dispersed_tail_equivalent_full_cycles",
            "caes_tail_cycle_effect_clustered_minus_dispersed",
            "cycles yr-1",
        ),
        (
            channel24,
            "delta_benefit_eur_kw_year",
            "mix_value_effect_clustered_minus_dispersed",
            "EUR kW-1 yr-1",
        ),
    ]:
        rows.append(
            summarize_country_blocks(
                country_values(source, column), label, unit
            )
        )

    retention_wide = caes.pivot_table(
        index=["country", "year"],
        columns="chronology",
        values=[
            "tail_charge_mwh",
            "tail_discharge_mwh",
            "tail_equivalent_full_cycles",
        ],
        aggfunc="first",
    )
    retention_wide.columns = [
        f"{metric}_{chronology}"
        for metric, chronology in retention_wide.columns
    ]
    retention_wide = retention_wide.reset_index()
    for numerator, denominator, label in [
        (
            "tail_charge_mwh_clustered",
            "tail_charge_mwh_dispersed",
            "caes_tail_charge_retention",
        ),
        (
            "tail_discharge_mwh_clustered",
            "tail_discharge_mwh_dispersed",
            "caes_tail_task_service_retention",
        ),
        (
            "tail_equivalent_full_cycles_clustered",
            "tail_equivalent_full_cycles_dispersed",
            "caes_tail_cycle_retention",
        ),
    ]:
        valid = retention_wide[
            retention_wide[denominator].abs().gt(1.0e-12)
        ].copy()
        valid["_retention_pct"] = (
            100.0 * valid[numerator] / valid[denominator]
        )
        rows.append(
            summarize_country_blocks(
                country_values(valid, "_retention_pct"),
                label,
                "%",
            )
        )

    output = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    manifest = {
        "analysis_id": "v43_focal24_country_block_summary_10000",
        "inputs": {
            str(LONG_PATH.resolve()): sha256(LONG_PATH),
            str(ARM_PATH.resolve()): sha256(ARM_PATH),
            str(PAIRED_PATH.resolve()): sha256(PAIRED_PATH),
            str(CHANNEL_PATH.resolve()): sha256(CHANNEL_PATH),
        },
        "output": str(OUTPUT.resolve()),
        "output_sha256": sha256(OUTPUT),
        "point_estimator": (
            "mean of country-specific means, giving each of 15 countries "
            "equal weight while retaining all years within country"
        ),
        "bootstrap": {
            "independent_block": "country",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "country_years": 132,
        "countries": 15,
        "configuration": "caes_24h",
        "scenario_id": 20,
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
