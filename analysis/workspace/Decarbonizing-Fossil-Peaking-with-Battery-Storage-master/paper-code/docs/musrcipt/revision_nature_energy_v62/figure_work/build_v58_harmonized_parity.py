#!/usr/bin/env python3
"""Build objective-aligned sequencing-parity records for three regimes.

The primary value metric is the exact annual-system-cost difference used by
the optimization: C_sys(additional 4 h LFP) - C_sys(candidate).  EENS-
equivalent firm supply is retained elsewhere as an operational diagnostic,
but is not substituted for the residual-peak term in the primary ledger.
All regimes use the S0 price vector, 2035 technology costs and the same
2025--2035 relative fixed-cost movement against additional 4 h LFP.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "source_data_v58" / "harmonized_parity"
FIRM_VALUE = 103.7992178588


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_cost_movement() -> pd.Series:
    catalogue = pd.read_csv(
        ROOT / "cost_and_decomposition/cost_catalogue_2025_2035_long.csv"
    )
    costs = catalogue.pivot(
        index="config_id",
        columns="learning_year",
        values="annual_fixed_cost_eur_source_aware_per_kw_discharge_year",
    )
    lfp = costs.loc["li_lfp_4h"]
    relative_2025 = costs[2025] - lfp.loc[2025]
    relative_2035 = costs[2035] - lfp.loc[2035]
    return (relative_2035 - relative_2025).abs().rename("cost_movement")


def cross_scale_system_cost_envelope(arms: pd.DataFrame) -> pd.DataFrame:
    """Return the system-cost-aligned duration envelope for China and Japan."""
    keys = ["application_case", "case_id", "case_year", "chronology"]
    comparator = arms.loc[
        arms["arm_type"].eq("extra_lfp4_comparator"),
        keys + ["S0_system_cost_eur"],
    ].rename(columns={"S0_system_cost_eur": "comparator_system_cost_eur"})
    candidates = arms.loc[arms["arm_type"].eq("ldes_candidate")].copy()
    candidates = candidates.merge(comparator, on=keys, validate="many_to_one")
    candidates["benefit_eur_kw_year"] = (
        candidates["comparator_system_cost_eur"]
        - candidates["S0_system_cost_eur"]
    ) / (candidates["tail_power_mw"] * 1000.0)
    envelope = (
        candidates.sort_values(
            [*keys, "duration_h", "benefit_eur_kw_year"],
            ascending=[True, True, True, True, True, False],
            kind="stable",
        )
        .groupby([*keys, "duration_h"], as_index=False, sort=True)
        .first()
        .rename(
            columns={
                "config_id": "dominant_config_id",
                "tech_id": "dominant_tech_id",
            }
        )
    )
    return envelope[
        [
            *keys,
            "duration_h",
            "benefit_eur_kw_year",
            "dominant_config_id",
            "dominant_tech_id",
        ]
    ]


def pair_envelopes(
    data: pd.DataFrame,
    index: list[str],
    regime: str,
    id_column: str,
    year_column: str,
) -> pd.DataFrame:
    paired = data.pivot(
        index=index,
        columns="chronology",
        values=["benefit_eur_kw_year", "dominant_config_id", "dominant_tech_id"],
    ).reset_index()
    paired.columns = ["_".join(map(str, col)).strip("_") for col in paired.columns]

    movement = relative_cost_movement()
    paired["sequencing_shift_eur_kw_year"] = (
        paired["benefit_eur_kw_year_clustered"]
        - paired["benefit_eur_kw_year_dispersed"]
    ).abs()
    paired["signed_sequencing_effect_eur_kw_year"] = (
        paired["benefit_eur_kw_year_clustered"]
        - paired["benefit_eur_kw_year_dispersed"]
    )
    paired["matched_cost_movement_eur_kw_year"] = 0.5 * (
        paired["dominant_config_id_clustered"].map(movement)
        + paired["dominant_config_id_dispersed"].map(movement)
    )
    if paired["matched_cost_movement_eur_kw_year"].isna().any():
        missing = paired.loc[
            paired["matched_cost_movement_eur_kw_year"].isna(),
            ["dominant_config_id_dispersed", "dominant_config_id_clustered"],
        ].drop_duplicates()
        raise ValueError(f"Missing cost path for {regime}: {missing.to_dict('records')}")

    paired["parity_ratio"] = (
        paired["sequencing_shift_eur_kw_year"]
        / paired["matched_cost_movement_eur_kw_year"]
    )
    paired["above_parity"] = paired["parity_ratio"].ge(1.0)
    paired["regime"] = regime
    paired["system_id"] = paired[id_column].astype(str)
    paired["observation_year"] = paired[year_column]
    return paired


def modal(series: pd.Series) -> str:
    modes = series.mode()
    return str(modes.iloc[0]) if not modes.empty else ""


def aggregate(records: pd.DataFrame) -> pd.DataFrame:
    return (
        records.groupby(["regime", "system_id", "duration_h"], sort=True)
        .agg(
            n_observation_years=("observation_year", "size"),
            parity_years=("above_parity", "sum"),
            parity_exposure_frequency=("above_parity", "mean"),
            median_parity_ratio=("parity_ratio", "median"),
            median_sequencing_shift_eur_kw_year=(
                "sequencing_shift_eur_kw_year",
                "median",
            ),
            median_matched_cost_movement_eur_kw_year=(
                "matched_cost_movement_eur_kw_year",
                "median",
            ),
            modal_technology=("dominant_tech_id_clustered", modal),
            modal_configuration=("dominant_config_id_clustered", modal),
        )
        .reset_index()
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    europe_path = (
        ROOT / "price_channel/full_summary_v0_1/price_channel_duration_envelope.csv.gz"
    )
    europe = pd.read_csv(europe_path)
    europe = europe.loc[europe["scenario"].eq("S0")].copy()
    europe_records = pair_envelopes(
        europe,
        ["country", "year", "duration_h"],
        "Europe",
        "country",
        "year",
    )

    cross_path = ROOT / "cross_scale_cases/summary_v0_2/cross_scale_arms.csv.gz"
    cross = cross_scale_system_cost_envelope(pd.read_csv(cross_path))
    china_records = pair_envelopes(
        cross.loc[cross["application_case"].eq("china")],
        ["case_id", "case_year", "duration_h"],
        "China",
        "case_id",
        "case_year",
    )
    japan_records = pair_envelopes(
        cross.loc[cross["application_case"].eq("japan")],
        ["case_id", "case_year", "duration_h"],
        "Japan",
        "case_id",
        "case_year",
    )

    records = pd.concat(
        [europe_records, china_records, japan_records],
        ignore_index=True,
        sort=False,
    )
    keep = [
        "regime",
        "system_id",
        "observation_year",
        "duration_h",
        "benefit_eur_kw_year_dispersed",
        "benefit_eur_kw_year_clustered",
        "signed_sequencing_effect_eur_kw_year",
        "sequencing_shift_eur_kw_year",
        "matched_cost_movement_eur_kw_year",
        "parity_ratio",
        "above_parity",
        "dominant_config_id_dispersed",
        "dominant_config_id_clustered",
        "dominant_tech_id_dispersed",
        "dominant_tech_id_clustered",
    ]
    records = records[keep].sort_values(
        ["regime", "system_id", "observation_year", "duration_h"]
    )
    exposure = aggregate(records)
    focus = records.loc[np.isclose(records["duration_h"], 24.0)].copy()
    summary = (
        focus.groupby("regime", sort=False)
        .agg(
            n_system_years=("above_parity", "size"),
            above_parity_count=("above_parity", "sum"),
            above_parity_share=("above_parity", "mean"),
            median_sequencing_shift_eur_kw_year=(
                "sequencing_shift_eur_kw_year",
                "median",
            ),
            q25_sequencing_shift_eur_kw_year=(
                "sequencing_shift_eur_kw_year",
                lambda x: x.quantile(0.25),
            ),
            q75_sequencing_shift_eur_kw_year=(
                "sequencing_shift_eur_kw_year",
                lambda x: x.quantile(0.75),
            ),
            median_matched_cost_movement_eur_kw_year=(
                "matched_cost_movement_eur_kw_year",
                "median",
            ),
            median_parity_ratio=("parity_ratio", "median"),
        )
        .reset_index()
    )
    summary["above_parity_count"] = summary["above_parity_count"].astype(int)

    records_path = OUT / "three_regime_system_year_duration_records.csv.gz"
    exposure_path = OUT / "three_regime_system_duration_exposure.csv"
    summary_path = OUT / "three_regime_24h_summary.csv"
    records.to_csv(records_path, index=False, compression={"method": "gzip", "mtime": 0})
    exposure.to_csv(exposure_path, index=False)
    summary.to_csv(summary_path, index=False)

    catalogue_path = ROOT / "cost_and_decomposition/cost_catalogue_2025_2035_long.csv"
    manifest = {
        "definition": (
            "absolute clustered-minus-dispersed change in the duration envelope of "
            "C_sys(additional 4 h LFP)-C_sys(candidate), divided by the matched "
            "2025-2035 relative annual fixed-cost movement versus 4 h LFP"
        ),
        "common_ledger": {
            "scenario": "S0",
            "cost_year": 2035,
            "firm_value_eur_kw_year": FIRM_VALUE,
            "ldes_cost_multiplier": 1.0,
            "optimization_and_reporting_metric": "annual_system_cost_eur",
            "firm_term": "1000 * firm_value_eur_kw_year * residual_peak_mw",
            "efc_role": "operational_diagnostic_only",
        },
        "input_rows": {
            "europe_envelope": int(len(europe)),
            "china_envelope": int(cross["application_case"].eq("china").sum()),
            "japan_envelope": int(cross["application_case"].eq("japan").sum()),
        },
        "output_rows": {"records": int(len(records)), "exposure": int(len(exposure))},
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [europe_path, cross_path, catalogue_path]
        },
        "output_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [records_path, exposure_path, summary_path]
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
