#!/usr/bin/env python3
"""Build configuration-resolved sequencing-parity exposure tables for v53."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "source_data"
SOURCE49 = HERE / "source_data_v49"
OUT = SOURCE / "v53_parity_exposure"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cost_movement() -> pd.Series:
    catalogue = pd.read_csv(ROOT / "cost_and_decomposition/cost_catalogue_2025_2035_long.csv")
    costs = catalogue.pivot(
        index="config_id",
        columns="learning_year",
        values="annual_fixed_cost_eur_source_aware_per_kw_discharge_year",
    )
    lfp4 = costs.loc["li_lfp_4h"]
    relative_2025 = lfp4.loc[2025] - costs[2025]
    relative_2035 = lfp4.loc[2035] - costs[2035]
    return (relative_2035 - relative_2025).abs().rename("cost_movement")


def pair_chronologies(
    data: pd.DataFrame,
    index: list[str],
    value_column: str,
    config_column: str,
    tech_column: str,
    regime: str,
) -> pd.DataFrame:
    paired = data.pivot(
        index=index,
        columns="chronology",
        values=[value_column, config_column, tech_column],
    ).reset_index()
    paired.columns = ["_".join(map(str, column)).strip("_") for column in paired.columns]
    movement = cost_movement()
    paired["sequencing_shift_eur_kw_year"] = (
        paired[f"{value_column}_clustered"] - paired[f"{value_column}_dispersed"]
    ).abs()
    paired["matched_cost_movement_eur_kw_year"] = 0.5 * (
        paired[f"{config_column}_clustered"].map(movement)
        + paired[f"{config_column}_dispersed"].map(movement)
    )
    if paired["matched_cost_movement_eur_kw_year"].isna().any():
        missing = paired.loc[
            paired["matched_cost_movement_eur_kw_year"].isna(),
            [f"{config_column}_dispersed", f"{config_column}_clustered"],
        ].drop_duplicates()
        raise ValueError(f"Missing cost paths for {regime}: {missing.to_dict('records')}")
    if (paired["matched_cost_movement_eur_kw_year"] <= 0).any():
        raise ValueError(f"Non-positive matched cost movement in {regime}")
    paired["parity_ratio"] = (
        paired["sequencing_shift_eur_kw_year"]
        / paired["matched_cost_movement_eur_kw_year"]
    )
    paired["above_parity"] = paired["parity_ratio"] >= 1.0
    paired["regime"] = regime
    return paired


def europe_records() -> pd.DataFrame:
    source = pd.read_csv(
        ROOT / "source_data/v42_fixed_scale_mix_envelope.csv",
        usecols=[
            "country", "year", "chronology", "duration_h", "dominant_config_id",
            "dominant_tech_id", "best_mix_benefit_eur_per_kw_common_power_year",
        ],
    )
    records = pair_chronologies(
        source,
        ["country", "year", "duration_h"],
        "best_mix_benefit_eur_per_kw_common_power_year",
        "dominant_config_id",
        "dominant_tech_id",
        "Europe",
    )
    records["system_id"] = records["country"]
    records["system_label"] = records["country"]
    records["observation_year"] = records["year"]
    return records


def cross_scale_records(regime: str, selection_label: str) -> pd.DataFrame:
    source = pd.read_csv(SOURCE49 / "figure5f_selected_cross_scale_envelopes.csv")
    source = source[
        source["application_case"].eq(regime.lower())
        & source["selection_label"].eq(selection_label)
    ].copy()
    records = pair_chronologies(
        source,
        ["case_id", "case_year", "duration_h"],
        "benefit_eur_kw_year",
        "dominant_config_id",
        "dominant_tech_id",
        regime,
    )
    records["system_id"] = records["case_id"]
    if regime == "China":
        records["system_label"] = records["case_id"].str.replace("CN-", "", regex=False)
    else:
        records["system_label"] = records["case_id"].str.replace("JP-", "", regex=False)
    records["observation_year"] = records["case_year"]
    return records


def modal(series: pd.Series) -> str:
    modes = series.mode()
    return str(modes.iloc[0]) if not modes.empty else ""


def aggregate(records: pd.DataFrame) -> pd.DataFrame:
    config_clustered = "dominant_config_id_clustered"
    tech_clustered = "dominant_tech_id_clustered"
    grouped = records.groupby(
        ["regime", "system_id", "system_label", "duration_h"], sort=True
    )
    result = grouped.agg(
        n_observation_years=("observation_year", "size"),
        parity_years=("above_parity", "sum"),
        parity_exposure_frequency=("above_parity", "mean"),
        median_parity_ratio=("parity_ratio", "median"),
        median_sequencing_shift_eur_kw_year=("sequencing_shift_eur_kw_year", "median"),
        median_matched_cost_movement_eur_kw_year=(
            "matched_cost_movement_eur_kw_year", "median"
        ),
        modal_technology=(tech_clustered, modal),
        modal_configuration=(config_clustered, modal),
    ).reset_index()
    result["parity_years"] = result["parity_years"].astype(int)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = pd.concat(
        [
            europe_records(),
            cross_scale_records("China", "China central 2030"),
            cross_scale_records("Japan", "Japan clustered-valley phase"),
        ],
        ignore_index=True,
        sort=False,
    )
    keep = [
        "regime", "system_id", "system_label", "observation_year", "duration_h",
        "sequencing_shift_eur_kw_year", "matched_cost_movement_eur_kw_year",
        "parity_ratio", "above_parity", "dominant_config_id_dispersed",
        "dominant_config_id_clustered", "dominant_tech_id_dispersed",
        "dominant_tech_id_clustered",
    ]
    records = records[keep].sort_values(
        ["regime", "system_id", "observation_year", "duration_h"]
    )
    aggregate_table = aggregate(records)
    records_path = OUT / "figure6cde_system_year_parity_records.csv"
    exposure_path = OUT / "figure6cde_system_duration_exposure.csv"
    records.to_csv(records_path, index=False)
    aggregate_table.to_csv(exposure_path, index=False)

    focus = records[np.isclose(records["duration_h"], 24.0)]
    summary = focus.groupby("regime", sort=False).agg(
        n_system_years=("above_parity", "size"),
        above_parity_count=("above_parity", "sum"),
        above_parity_share=("above_parity", "mean"),
        median_sequencing_shift_eur_kw_year=("sequencing_shift_eur_kw_year", "median"),
        median_matched_cost_movement_eur_kw_year=(
            "matched_cost_movement_eur_kw_year", "median"
        ),
        median_parity_ratio=("parity_ratio", "median"),
    ).reset_index()
    summary["above_parity_count"] = summary["above_parity_count"].astype(int)
    summary_path = OUT / "figure6b_24h_cross_regime_parity_summary.csv"
    summary.to_csv(summary_path, index=False)

    inputs = [
        ROOT / "source_data/v42_fixed_scale_mix_envelope.csv",
        SOURCE49 / "figure5f_selected_cross_scale_envelopes.csv",
        ROOT / "cost_and_decomposition/cost_catalogue_2025_2035_long.csv",
    ]
    manifest = {
        "definition": "absolute clustered-minus-dispersed value shift divided by the matched 2025-2035 relative fixed-cost movement versus 4 h LFP",
        "parity_threshold": 1.0,
        "rows": {"records": len(records), "exposure": len(aggregate_table)},
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "output_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [records_path, exposure_path, summary_path]
        },
    }
    (OUT / "v53_parity_exposure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
