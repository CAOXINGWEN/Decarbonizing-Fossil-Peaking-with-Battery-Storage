#!/usr/bin/env python3
"""Build the ACER--ERAA annual curtailment calibration contract.

The script has one-way temporal logic:

* ERAA model hours may be aggregated upward to country-year quantities.
* ACER country-year observations are never allocated downward to hours.

The admissible comparison is limited to annual scale and cross-country rank.
It is a plausibility calibration across periods, not a claim that the 2035
model reproduces the 2023 or 2024 electricity system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
BASE = Path(__file__).resolve().parents[1]
RESEARCH = BASE / "artifacts" / "research"
ACER_SOURCE = RESEARCH / "ACER_RES_CURTAILMENT_2023_2024.csv"
GENERATION_CAPACITY = (
    BASE
    / "raw"
    / "eraa_2025"
    / "extracted"
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "GenerationCapacities.csv"
)
DEFAULT_OUTPUT = RESEARCH

REFERENCE_ROLE = "reference_system_without_candidate_storage"
NATURAL_CHRONOLOGY = "natural"
POSITIVE_REPORTING = frozenset(
    {"reported_nonzero", "reported_nonzero_but_share_rounds_to_zero"}
)
MISSING_REPORTING = frozenset(
    {"not_present_in_acer_table", "explicitly_missing"}
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"unsupported table: {path}")


def build_observation_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(ACER_SOURCE)
    source = source.rename(
        columns={
            "year": "observed_year",
            "res_generation_gwh": "observed_res_generation_gwh",
            "res_curtailment_gwh": "observed_res_curtailment_gwh",
            "official_share_pct": "published_share_pct",
        }
    ).copy()
    numeric = (
        "observed_res_generation_gwh",
        "observed_res_curtailment_gwh",
        "published_share_pct",
    )
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["derived_share_pct"] = (
        100.0
        * source["observed_res_curtailment_gwh"]
        / source["observed_res_generation_gwh"]
    )
    source["observation_missing"] = source["reporting_status"].isin(
        MISSING_REPORTING
    )
    source["reported_zero"] = source["reporting_status"].eq(
        "reported_zero_in_acer_table"
    )
    source["annual_volume_scale_eligible"] = (
        ~source["observation_missing"]
        & source["observed_res_generation_gwh"].notna()
        & source["observed_res_curtailment_gwh"].notna()
    )
    source["annual_intensity_scale_eligible"] = np.where(
        source["observed_year"].eq(2023),
        source["derived_share_pct"].notna(),
        source["published_share_pct"].notna(),
    )
    source["primary_volume_rank_eligible"] = (
        source["reporting_status"].isin(POSITIVE_REPORTING)
        & source["observed_res_curtailment_gwh"].gt(0)
    )
    source["primary_intensity_rank_eligible"] = (
        source["reporting_status"].isin(POSITIVE_REPORTING)
        & np.where(
            source["observed_year"].eq(2023),
            source["derived_share_pct"].gt(0),
            source["published_share_pct"].gt(0),
        )
    )
    source["zero_inclusive_rank_sensitivity_eligible"] = (
        ~source["observation_missing"]
        & np.where(
            source["observed_year"].eq(2023),
            source["observed_res_curtailment_gwh"].notna(),
            source["published_share_pct"].notna(),
        )
    )
    source["observed_intensity_for_comparison_pct"] = np.where(
        source["observed_year"].eq(2023),
        source["derived_share_pct"],
        source["published_share_pct"],
    )
    source["observed_volume_rank_primary"] = np.nan
    source["observed_intensity_rank_primary"] = np.nan
    for year, group in source.groupby("observed_year"):
        volume_index = group.index[group["primary_volume_rank_eligible"]]
        source.loc[volume_index, "observed_volume_rank_primary"] = (
            source.loc[
                volume_index, "observed_res_curtailment_gwh"
            ].rank(method="average", ascending=False)
        )
        intensity_index = group.index[group["primary_intensity_rank_eligible"]]
        source.loc[
            intensity_index, "observed_intensity_rank_primary"
        ] = source.loc[
            intensity_index, "observed_intensity_for_comparison_pct"
        ].rank(
            method="average", ascending=False
        )
    source["temporal_resolution"] = "country-year"
    source["permitted_calibration_role"] = (
        "annual scale and country ranking only"
    )
    source["hourly_allocation_permitted"] = False
    source["missing_must_not_be_encoded_as_zero"] = source[
        "observation_missing"
    ]

    keep = [
        "observed_year",
        "country",
        "observed_res_generation_gwh",
        "observed_res_curtailment_gwh",
        "published_share_pct",
        "derived_share_pct",
        "observed_intensity_for_comparison_pct",
        "reporting_status",
        "observation_missing",
        "reported_zero",
        "annual_volume_scale_eligible",
        "annual_intensity_scale_eligible",
        "primary_volume_rank_eligible",
        "primary_intensity_rank_eligible",
        "zero_inclusive_rank_sensitivity_eligible",
        "observed_volume_rank_primary",
        "observed_intensity_rank_primary",
        "technology_scope",
        "cause_scope",
        "temporal_resolution",
        "permitted_calibration_role",
        "hourly_allocation_permitted",
        "missing_must_not_be_encoded_as_zero",
        "source_url",
        "note",
    ]
    reference = source[keep].sort_values(
        ["observed_year", "country"]
    ).reset_index(drop=True)

    summaries: list[dict[str, Any]] = []
    for year, group in reference.groupby("observed_year"):
        volume = group.loc[group["annual_volume_scale_eligible"]]
        intensity = group.loc[group["annual_intensity_scale_eligible"]]
        observed_generation = float(
            volume["observed_res_generation_gwh"].sum()
        )
        observed_curtailment = float(
            volume["observed_res_curtailment_gwh"].sum()
        )
        summaries.append(
            {
                "observed_year": int(year),
                "reported_country_rows": int((~group["observation_missing"]).sum()),
                "missing_country_rows": int(group["observation_missing"].sum()),
                "absolute_volume_country_rows": int(len(volume)),
                "positive_rank_country_rows": int(
                    group["primary_intensity_rank_eligible"].sum()
                ),
                "observed_res_generation_gwh": (
                    observed_generation if len(volume) else np.nan
                ),
                "observed_res_curtailment_gwh": (
                    observed_curtailment if len(volume) else np.nan
                ),
                "observed_aggregate_derived_share_pct": (
                    100.0 * observed_curtailment / observed_generation
                    if observed_generation > 0
                    else np.nan
                ),
                "observed_country_median_intensity_pct": float(
                    intensity[
                        "observed_intensity_for_comparison_pct"
                    ].median()
                )
                if len(intensity)
                else np.nan,
                "absolute_scale_available": bool(len(volume)),
                "intensity_scale_available": bool(len(intensity)),
            }
        )
    summary = pd.DataFrame(summaries)
    return reference, summary


def build_zone_country_crosswalk(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    capacity = pd.read_csv(GENERATION_CAPACITY)
    capacity = capacity.loc[
        capacity["data_version"].eq("ERAA 2025 final")
        & pd.to_numeric(capacity["Target year"], errors="coerce").eq(2035)
    ]
    zones = sorted(capacity["Market_Node"].dropna().astype(str).unique())
    obs_lookup = {
        (int(row.observed_year), str(row.country)): str(row.reporting_status)
        for row in observations.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for zone in zones:
        country = zone[:2]
        rows.append(
            {
                "eraa_zone": zone,
                "acer_country": country,
                "mapping_rule": (
                    "explicit frozen map generated from the first two ISO "
                    "characters of official ERAA atomic/offshore zone codes"
                ),
                "eraa_2035_final_capacity_zone_present": True,
                "acer_2023_reporting_status": obs_lookup.get(
                    (2023, country), "not_in_acer_extract"
                ),
                "acer_2024_reporting_status": obs_lookup.get(
                    (2024, country), "not_in_acer_extract"
                ),
                "absolute_scale_2023_eligible": bool(
                    (
                        observations["observed_year"].eq(2023)
                        & observations["country"].eq(country)
                        & observations["annual_volume_scale_eligible"]
                    ).any()
                ),
                "intensity_rank_2023_eligible": bool(
                    (
                        observations["observed_year"].eq(2023)
                        & observations["country"].eq(country)
                        & observations["primary_intensity_rank_eligible"]
                    ).any()
                ),
                "intensity_rank_2024_eligible": bool(
                    (
                        observations["observed_year"].eq(2024)
                        & observations["country"].eq(country)
                        & observations["primary_intensity_rank_eligible"]
                    ).any()
                ),
                "country_aggregation_only": True,
                "hourly_observation_join_permitted": False,
            }
        )
    return pd.DataFrame(rows)


def build_input_contract() -> pd.DataFrame:
    rows = [
        (
            "model_case_id",
            "string",
            True,
            "identifies one frozen reference-system run",
            "not an ACER field",
        ),
        (
            "target_year",
            "integer",
            True,
            "ERAA target year; expected primary value 2035",
            "must not be renamed observed_year",
        ),
        (
            "weather_scenario",
            "string",
            True,
            "synchronized pan-European weather trajectory",
            "not an independent country sample",
        ),
        (
            "scenario_weight",
            "float",
            True,
            "non-negative weights summing to one within case and target year",
            "constant across zones for the same scenario",
        ),
        (
            "chronology",
            "string",
            True,
            "must equal natural",
            "clustered/dispersed interventions are not calibration targets",
        ),
        (
            "calibration_role",
            "string",
            True,
            f"must equal {REFERENCE_ROLE}",
            "candidate-storage post-treatment spill is excluded",
        ),
        (
            "zone",
            "string",
            True,
            "official ERAA atomic or offshore model zone",
            "aggregated to country using the frozen crosswalk",
        ),
        (
            "annual_vre_spill_mwh",
            "float",
            True,
            "model-endogenous annual VRE spill",
            "never overwritten by ACER observations",
        ),
        (
            "annual_vre_potential_mwh",
            "float",
            True,
            "annual potential wind plus solar energy before spill",
            "required for intensity-scale and intensity-rank metrics",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "field",
            "type",
            "required",
            "definition",
            "constraint",
        ],
    )


def build_metrics_contract() -> pd.DataFrame:
    rows = [
        (
            "A23_TOTAL_VOLUME_RATIO",
            2023,
            "annual_scale",
            "sum model spill GWh / sum ACER curtailment GWh",
            "all common 2023 countries with reported volumes, including "
            "published zeros",
            "descriptive cross-period plausibility",
            "not a goodness-of-fit probability and not a future target",
        ),
        (
            "A23_AGGREGATE_INTENSITY_DIFF_PP",
            2023,
            "annual_scale",
            "model aggregate spill share minus ACER aggregate derived share",
            "all common 2023 countries with volume and generation",
            "descriptive cross-period plausibility",
            "does not identify hourly charging availability",
        ),
        (
            "A24_MEDIAN_INTENSITY_DIFF_PP",
            2024,
            "annual_scale",
            "median model spill share minus median published ACER share",
            "common 2024 countries with published non-missing shares",
            "share-scale plausibility only",
            "2024 absolute GWh cannot be reconstructed from this table",
        ),
        (
            "R23_VOLUME_SPEARMAN",
            2023,
            "country_ranking",
            "Spearman rank correlation of annual GWh",
            "common positive-volume reporters",
            "primary rank diagnostic",
            "country size and 2035 build-out remain confounders",
        ),
        (
            "R23_INTENSITY_SPEARMAN",
            2023,
            "country_ranking",
            "Spearman rank correlation of annual percentage",
            "common positive-volume reporters; ACER intensity is derived "
            "from published GWh",
            "primary rank diagnostic",
            "not a causal validation of network constraints",
        ),
        (
            "R24_INTENSITY_SPEARMAN",
            2024,
            "country_ranking",
            "Spearman rank correlation of annual percentage",
            "common countries with published positive shares",
            "primary rank diagnostic",
            "no absolute-volume comparison is permitted",
        ),
        (
            "R23_TOP5_VOLUME_OVERLAP",
            2023,
            "country_ranking",
            "intersection size / min(5, common positive reporters)",
            "common positive-volume reporters",
            "high-curtailment-country diagnostic",
            "not an hourly event-overlap metric",
        ),
        (
            "R24_TOP5_INTENSITY_OVERLAP",
            2024,
            "country_ranking",
            "intersection size / min(5, common positive reporters)",
            "common countries with published positive shares",
            "high-curtailment-country diagnostic",
            "not an hourly event-overlap metric",
        ),
        (
            "S23_ZERO_INCLUSIVE_INTENSITY_SPEARMAN",
            2023,
            "country_ranking_sensitivity",
            "Spearman rank correlation including published zero rows as ties",
            "all common reported 2023 rows; missing excluded",
            "sensitivity only",
            "a published zero is not proof of physical zero",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "metric_id",
            "observed_year",
            "comparison_dimension",
            "formula",
            "country_set",
            "permitted_role",
            "prohibited_interpretation",
        ],
    )


def _validate_scenario_weights(frame: pd.DataFrame) -> None:
    uniqueness = (
        frame.groupby(
            ["model_case_id", "target_year", "weather_scenario"]
        )["scenario_weight"]
        .nunique()
    )
    if uniqueness.gt(1).any():
        raise ValueError("scenario_weight differs across zones")
    weights = (
        frame.drop_duplicates(
            ["model_case_id", "target_year", "weather_scenario"]
        )
        .groupby(["model_case_id", "target_year"])["scenario_weight"]
        .sum()
    )
    if not np.allclose(weights.to_numpy(dtype=float), 1.0, atol=1e-10):
        raise ValueError(f"scenario weights do not sum to one: {weights}")


def standardize_model_annual(
    path: Path,
    input_kind: str,
) -> pd.DataFrame:
    frame = read_table(path)
    if input_kind == "hourly":
        required = {
            "model_case_id",
            "target_year",
            "weather_scenario",
            "scenario_weight",
            "chronology",
            "calibration_role",
            "zone",
            "hour_of_year",
            "vre_spill_mw",
            "wind_potential_mw",
            "solar_potential_mw",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"hourly model output missing {sorted(missing)}")
        if "interval_hours" not in frame:
            frame["interval_hours"] = 1.0
        key = [
            "model_case_id",
            "target_year",
            "weather_scenario",
            "zone",
            "hour_of_year",
        ]
        if frame.duplicated(key).any():
            raise ValueError("duplicate model zone-hour keys")
        frame["spill_energy_mwh"] = (
            pd.to_numeric(frame["vre_spill_mw"], errors="raise")
            * pd.to_numeric(frame["interval_hours"], errors="raise")
        )
        frame["potential_energy_mwh"] = (
            pd.to_numeric(frame["wind_potential_mw"], errors="raise")
            + pd.to_numeric(frame["solar_potential_mw"], errors="raise")
        ) * pd.to_numeric(frame["interval_hours"], errors="raise")
        group = [
            "model_case_id",
            "target_year",
            "weather_scenario",
            "scenario_weight",
            "chronology",
            "calibration_role",
            "zone",
        ]
        frame = (
            frame.groupby(group, as_index=False)[
                ["spill_energy_mwh", "potential_energy_mwh"]
            ]
            .sum()
            .rename(
                columns={
                    "spill_energy_mwh": "annual_vre_spill_mwh",
                    "potential_energy_mwh": "annual_vre_potential_mwh",
                }
            )
        )
    elif input_kind == "annual":
        required = set(build_input_contract()["field"])
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"annual model output missing {sorted(missing)}")
        frame = frame[list(build_input_contract()["field"])].copy()
    else:
        raise ValueError(f"unknown input kind: {input_kind}")

    for column in (
        "target_year",
        "scenario_weight",
        "annual_vre_spill_mwh",
        "annual_vre_potential_mwh",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["chronology"].astype(str).ne(NATURAL_CHRONOLOGY).any():
        values = sorted(frame["chronology"].astype(str).unique())
        raise ValueError(
            f"calibration accepts natural chronology only; observed {values}"
        )
    if frame["calibration_role"].astype(str).ne(REFERENCE_ROLE).any():
        values = sorted(frame["calibration_role"].astype(str).unique())
        raise ValueError(
            f"calibration_role must be {REFERENCE_ROLE}; observed {values}"
        )
    if (
        frame["scenario_weight"].lt(0).any()
        or frame["annual_vre_spill_mwh"].lt(0).any()
        or frame["annual_vre_potential_mwh"].lt(0).any()
    ):
        raise ValueError("negative weight, spill, or VRE potential")
    if (
        frame["annual_vre_spill_mwh"]
        .gt(frame["annual_vre_potential_mwh"] + 1e-8)
        .any()
    ):
        raise ValueError("annual spill exceeds annual VRE potential")
    _validate_scenario_weights(frame)
    return frame


def aggregate_model_to_country(
    annual: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    zone_map = crosswalk[["eraa_zone", "acer_country"]].rename(
        columns={"eraa_zone": "zone", "acer_country": "country"}
    )
    merged = annual.merge(zone_map, on="zone", how="left", validate="many_to_one")
    if merged["country"].isna().any():
        missing = sorted(merged.loc[merged["country"].isna(), "zone"].unique())
        raise ValueError(f"unmapped ERAA zones: {missing}")
    scenario_country = (
        merged.groupby(
            [
                "model_case_id",
                "target_year",
                "weather_scenario",
                "scenario_weight",
                "country",
            ],
            as_index=False,
        )[["annual_vre_spill_mwh", "annual_vre_potential_mwh"]]
        .sum()
    )
    scenario_country["weighted_spill_mwh"] = (
        scenario_country["annual_vre_spill_mwh"]
        * scenario_country["scenario_weight"]
    )
    scenario_country["weighted_potential_mwh"] = (
        scenario_country["annual_vre_potential_mwh"]
        * scenario_country["scenario_weight"]
    )
    country = (
        scenario_country.groupby(
            ["model_case_id", "target_year", "country"], as_index=False
        )[["weighted_spill_mwh", "weighted_potential_mwh"]]
        .sum()
        .rename(
            columns={
                "weighted_spill_mwh": "model_expected_vre_spill_mwh",
                "weighted_potential_mwh": (
                    "model_expected_vre_potential_mwh"
                ),
            }
        )
    )
    country["model_expected_vre_spill_gwh"] = (
        country["model_expected_vre_spill_mwh"] / 1_000.0
    )
    country["model_expected_spill_share_pct"] = (
        100.0
        * country["model_expected_vre_spill_mwh"]
        / country["model_expected_vre_potential_mwh"]
    )
    return country


def spearman(frame: pd.DataFrame, x: str, y: str) -> float:
    if len(frame) < 3:
        return np.nan
    return float(
        frame[x].rank(method="average").corr(
            frame[y].rank(method="average"), method="pearson"
        )
    )


def top_k_overlap(frame: pd.DataFrame, x: str, y: str, k: int = 5) -> float:
    if frame.empty:
        return np.nan
    size = min(k, len(frame))
    x_top = set(frame.nlargest(size, x)["country"])
    y_top = set(frame.nlargest(size, y)["country"])
    return len(x_top & y_top) / size


def evaluate_model(
    country_model: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for (case_id, target_year), model_group in country_model.groupby(
        ["model_case_id", "target_year"], sort=True
    ):
        for observed_year, observed_group in observations.groupby(
            "observed_year", sort=True
        ):
            joined = observed_group.merge(
                model_group,
                on="country",
                how="inner",
                validate="one_to_one",
            )
            comparisons.append(joined)

            def add(
                metric_id: str,
                value: float,
                model_value: float,
                observed_value: float,
                frame: pd.DataFrame,
                status: str,
            ) -> None:
                metrics.append(
                    {
                        "model_case_id": case_id,
                        "target_year": int(target_year),
                        "observed_year": int(observed_year),
                        "metric_id": metric_id,
                        "metric_value": value,
                        "model_value": model_value,
                        "observed_value": observed_value,
                        "n_common_countries": int(len(frame)),
                        "model_country_coverage_fraction": (
                            len(frame) / len(observed_group)
                            if len(observed_group)
                            else np.nan
                        ),
                        "interpretation_status": status,
                    }
                )

            if observed_year == 2023:
                scale = joined.loc[joined["annual_volume_scale_eligible"]]
                model_total = float(
                    scale["model_expected_vre_spill_gwh"].sum()
                )
                observed_total = float(
                    scale["observed_res_curtailment_gwh"].sum()
                )
                add(
                    "A23_TOTAL_VOLUME_RATIO",
                    model_total / observed_total
                    if observed_total > 0
                    else np.nan,
                    model_total,
                    observed_total,
                    scale,
                    "descriptive_cross_period_plausibility",
                )
                model_potential = float(
                    scale["model_expected_vre_potential_mwh"].sum()
                )
                observed_generation = float(
                    scale["observed_res_generation_gwh"].sum()
                )
                model_share = (
                    100_000.0
                    * float(scale["model_expected_vre_spill_gwh"].sum())
                    / model_potential
                    if model_potential > 0
                    else np.nan
                )
                observed_share = (
                    100.0
                    * float(scale["observed_res_curtailment_gwh"].sum())
                    / observed_generation
                    if observed_generation > 0
                    else np.nan
                )
                add(
                    "A23_AGGREGATE_INTENSITY_DIFF_PP",
                    model_share - observed_share,
                    model_share,
                    observed_share,
                    scale,
                    "descriptive_cross_period_plausibility",
                )
                volume_rank = joined.loc[
                    joined["primary_volume_rank_eligible"]
                ]
                add(
                    "R23_VOLUME_SPEARMAN",
                    spearman(
                        volume_rank,
                        "model_expected_vre_spill_gwh",
                        "observed_res_curtailment_gwh",
                    ),
                    np.nan,
                    np.nan,
                    volume_rank,
                    "primary_country_rank_diagnostic",
                )
                add(
                    "R23_TOP5_VOLUME_OVERLAP",
                    top_k_overlap(
                        volume_rank,
                        "model_expected_vre_spill_gwh",
                        "observed_res_curtailment_gwh",
                    ),
                    np.nan,
                    np.nan,
                    volume_rank,
                    "primary_country_rank_diagnostic",
                )
                intensity_rank = joined.loc[
                    joined["primary_intensity_rank_eligible"]
                ]
                add(
                    "R23_INTENSITY_SPEARMAN",
                    spearman(
                        intensity_rank,
                        "model_expected_spill_share_pct",
                        "observed_intensity_for_comparison_pct",
                    ),
                    np.nan,
                    np.nan,
                    intensity_rank,
                    "primary_country_rank_diagnostic",
                )
                zero_rank = joined.loc[
                    joined["zero_inclusive_rank_sensitivity_eligible"]
                ]
                add(
                    "S23_ZERO_INCLUSIVE_INTENSITY_SPEARMAN",
                    spearman(
                        zero_rank,
                        "model_expected_spill_share_pct",
                        "observed_intensity_for_comparison_pct",
                    ),
                    np.nan,
                    np.nan,
                    zero_rank,
                    "zero_inclusive_sensitivity_only",
                )
            elif observed_year == 2024:
                scale = joined.loc[
                    joined["annual_intensity_scale_eligible"]
                ]
                model_median = float(
                    scale["model_expected_spill_share_pct"].median()
                )
                observed_median = float(
                    scale["observed_intensity_for_comparison_pct"].median()
                )
                add(
                    "A24_MEDIAN_INTENSITY_DIFF_PP",
                    model_median - observed_median,
                    model_median,
                    observed_median,
                    scale,
                    "share_scale_only_no_absolute_volume",
                )
                intensity_rank = joined.loc[
                    joined["primary_intensity_rank_eligible"]
                ]
                add(
                    "R24_INTENSITY_SPEARMAN",
                    spearman(
                        intensity_rank,
                        "model_expected_spill_share_pct",
                        "observed_intensity_for_comparison_pct",
                    ),
                    np.nan,
                    np.nan,
                    intensity_rank,
                    "primary_country_rank_diagnostic",
                )
                add(
                    "R24_TOP5_INTENSITY_OVERLAP",
                    top_k_overlap(
                        intensity_rank,
                        "model_expected_spill_share_pct",
                        "observed_intensity_for_comparison_pct",
                    ),
                    np.nan,
                    np.nan,
                    intensity_rank,
                    "primary_country_rank_diagnostic",
                )
    comparison = (
        pd.concat(comparisons, ignore_index=True)
        if comparisons
        else pd.DataFrame()
    )
    return comparison, pd.DataFrame(metrics)


def synthetic_self_test(
    observations: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> dict[str, Any]:
    observed_2023 = observations.loc[
        observations["observed_year"].eq(2023)
        & observations["annual_volume_scale_eligible"]
    ].copy()
    first_zone = (
        crosswalk.sort_values("eraa_zone")
        .drop_duplicates("acer_country")
        .set_index("acer_country")["eraa_zone"]
        .to_dict()
    )
    rows = []
    for row in observed_2023.itertuples(index=False):
        if row.country not in first_zone:
            continue
        rows.append(
            {
                "model_case_id": "synthetic_identity_2023",
                "target_year": 2035,
                "weather_scenario": "synthetic_weather",
                "scenario_weight": 1.0,
                "chronology": NATURAL_CHRONOLOGY,
                "calibration_role": REFERENCE_ROLE,
                "zone": first_zone[row.country],
                "annual_vre_spill_mwh": (
                    float(row.observed_res_curtailment_gwh) * 1_000.0
                ),
                "annual_vre_potential_mwh": (
                    float(row.observed_res_generation_gwh) * 1_000.0
                ),
            }
        )
    synthetic = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        path = temporary_path / "synthetic_annual.csv"
        synthetic.to_csv(path, index=False)
        standardized = standardize_model_annual(path, "annual")

        # Independent model-only numbers test upward aggregation.  They are
        # deliberately unrelated to ACER observations so this test never
        # allocates an observed annual quantity to hours.
        hourly = pd.DataFrame(
            [
                {
                    "model_case_id": "synthetic_hourly_upward_only",
                    "target_year": 2035,
                    "weather_scenario": "synthetic_weather",
                    "scenario_weight": 1.0,
                    "chronology": NATURAL_CHRONOLOGY,
                    "calibration_role": REFERENCE_ROLE,
                    "zone": "AT00",
                    "hour_of_year": 0,
                    "vre_spill_mw": 10.0,
                    "wind_potential_mw": 60.0,
                    "solar_potential_mw": 40.0,
                },
                {
                    "model_case_id": "synthetic_hourly_upward_only",
                    "target_year": 2035,
                    "weather_scenario": "synthetic_weather",
                    "scenario_weight": 1.0,
                    "chronology": NATURAL_CHRONOLOGY,
                    "calibration_role": REFERENCE_ROLE,
                    "zone": "AT00",
                    "hour_of_year": 1,
                    "vre_spill_mw": 20.0,
                    "wind_potential_mw": 120.0,
                    "solar_potential_mw": 80.0,
                },
            ]
        )
        hourly_path = temporary_path / "synthetic_hourly.csv"
        hourly.to_csv(hourly_path, index=False)
        hourly_annual = standardize_model_annual(hourly_path, "hourly")

        invalid = synthetic.copy()
        invalid["chronology"] = "clustered"
        invalid_path = temporary_path / "invalid_chronology.csv"
        invalid.to_csv(invalid_path, index=False)
        invalid_chronology_rejected = False
        try:
            standardize_model_annual(invalid_path, "annual")
        except ValueError:
            invalid_chronology_rejected = True
    country = aggregate_model_to_country(standardized, crosswalk)
    comparison, metrics = evaluate_model(country, observations)
    lookup = metrics.set_index("metric_id")["metric_value"]
    return {
        "synthetic_2023_total_volume_ratio_is_one": bool(
            np.isclose(lookup["A23_TOTAL_VOLUME_RATIO"], 1.0)
        ),
        "synthetic_2023_aggregate_share_difference_is_zero": bool(
            np.isclose(lookup["A23_AGGREGATE_INTENSITY_DIFF_PP"], 0.0)
        ),
        "synthetic_2023_volume_rank_spearman_is_one": bool(
            np.isclose(lookup["R23_VOLUME_SPEARMAN"], 1.0)
        ),
        "no_hour_column_in_country_comparison": bool(
            not any(
                column in comparison.columns
                for column in ("hour", "hour_of_year", "timestamp")
            )
        ),
        "annual_observations_not_row_expanded": bool(
            len(
                comparison.loc[
                    comparison["observed_year"].eq(2023)
                ]
            )
            == len(rows)
        ),
        "no_2024_absolute_volume_metric": bool(
            not metrics.loc[
                metrics["observed_year"].eq(2024), "metric_id"
            ]
            .str.contains("TOTAL_VOLUME", regex=False)
            .any()
        ),
        "hourly_model_output_aggregates_upward_only": bool(
            len(hourly_annual) == 1
            and np.isclose(
                hourly_annual.iloc[0]["annual_vre_spill_mwh"], 30.0
            )
            and np.isclose(
                hourly_annual.iloc[0]["annual_vre_potential_mwh"], 300.0
            )
        ),
        "non_natural_chronology_rejected": invalid_chronology_rejected,
    }


def write_decision_memo(
    output: Path,
    observations: pd.DataFrame,
    summary: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> None:
    s23 = summary.set_index("observed_year").loc[2023]
    s24 = summary.set_index("observed_year").loc[2024]
    ie = observations.loc[observations["country"].eq("IE")]
    memo = f"""# ACER–ERAA 年度弃电校准契约

## 结论

ACER 表可用于约束 ERAA 模型的年度弃电规模与国家排序，不能决定弃电
发生在哪些小时。本契约只允许模型逐时结果向上汇总为国家年度量，任何
把 ACER 年度总量按模型余量、价格、天气或权重分配回小时的操作均被禁止。

2023 年表在 {int(s23["absolute_volume_country_rows"])} 个有年度 GWh 的国家
合计报告 {s23["observed_res_curtailment_gwh"]:.0f} GWh 可再生能源下调，
对应发布发电量口径下的汇总比率约
{s23["observed_aggregate_derived_share_pct"]:.2f}%。这一截面支持绝对规模、
强度与国家排序。2024 年表对 {int(s24["reported_country_rows"])} 个国家
只给出百分比，因此只支持强度尺度与排序，不得反推 GWh。

爱尔兰在两个截面均为缺失而不是零
（状态：{"; ".join(ie["reporting_status"].astype(str))}）。因此 IE00 不进入
ACER 国家排名；这与 IE00 退役交叉表审计中的剔除归因敏感性一致。

## 与 ERAA 输出的接口

输入可以是已经年度汇总的模型表，也可以是模型逐时 dispatch。后一种
情况下脚本只执行 `sum(hourly model spill)` 的向上汇总，绝不读取 ACER
数量参与小时运算。校准对象必须是自然日序、无候选储能的参考系统，并
先在同步天气情景内按冻结权重求期望，再与 ACER 横截面比较。

冻结的 ERAA 区域到国家映射包含 {len(crosswalk)} 个 2035-final 区域。
所有 offshore 和多竞价区节点先保留为原子区，再汇总到 ISO2 国家；没有
映射的区域会令脚本失败。

## 允许报告

- 2023 年共同国家的年度弃电 GWh 比率；
- 2023 年模型与观测汇总弃电强度之差；
- 2024 年国家弃电强度中位数之差；
- 正值报告国的 GWh 或强度 Spearman 排序；
- 前五高弃电国家的重合率；
- 把 2023 年发布零值作为并列下界的单独敏感性。

这些量是跨时期的结构合理性检查。2035 年风光规模、网络和需求与
2023–2024 年不同，因而不能把比率偏离 1 直接称为模型误差。

## 禁止事项

1. 不把年度 ACER GWh 分配到小时或用它生成 SOC 充电路径。
2. 不把未报告值编码为零。
3. 不把 actual generation、forecast error、redispatch 总量或 downward
   balancing 自动改称实际弃电。
4. 不用 clustered/dispersed 反事实或候选储能投产后的 spill 校准基准。
5. 不把天气场景、区域小时或国家行当作独立样本扩大样本量。
6. 不从 2024 百分比反推绝对 GWh。

模型结果到位后，可用同一脚本的 `--model-input` 和 `--input-kind` 生成
国家年度比较表与指标表。脚本不改模型核心或运行器。
"""
    (output / "ACER_ERAA_ANNUAL_CALIBRATION_DECISION.md").write_text(
        memo, encoding="utf-8"
    )


def run(
    output: Path,
    model_input: Path | None,
    input_kind: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    observations, summary = build_observation_reference()
    crosswalk = build_zone_country_crosswalk(observations)
    input_contract = build_input_contract()
    metrics_contract = build_metrics_contract()

    observations.to_csv(
        output / "acer_eraa_annual_observation_reference.csv", index=False
    )
    summary.to_csv(
        output / "acer_eraa_annual_observation_summary.csv", index=False
    )
    crosswalk.to_csv(
        output / "acer_eraa_zone_country_crosswalk.csv", index=False
    )
    input_contract.to_csv(
        output / "acer_eraa_annual_model_input_contract.csv", index=False
    )
    metrics_contract.to_csv(
        output / "acer_eraa_annual_calibration_metrics_contract.csv",
        index=False,
    )
    write_decision_memo(output, observations, summary, crosswalk)

    self_test = synthetic_self_test(observations, crosswalk)
    model_results_written = False
    model_input_sha256 = None
    if model_input is not None:
        annual = standardize_model_annual(model_input, input_kind)
        country = aggregate_model_to_country(annual, crosswalk)
        comparison, metrics = evaluate_model(country, observations)
        country.to_csv(
            output / "acer_eraa_model_country_annual_expected.csv",
            index=False,
        )
        comparison.to_csv(
            output / "acer_eraa_annual_country_comparison.csv", index=False
        )
        metrics.to_csv(
            output / "acer_eraa_annual_calibration_metrics.csv", index=False
        )
        model_results_written = True
        model_input_sha256 = sha256(model_input)

    summary_lookup = summary.set_index("observed_year")
    ie = observations.loc[observations["country"].eq("IE")]
    qa = {
        "script_version": SCRIPT_VERSION,
        "acer_source_sha256": sha256(ACER_SOURCE),
        "eraa_capacity_source_sha256": sha256(GENERATION_CAPACITY),
        "observed_years": sorted(
            int(value) for value in observations["observed_year"].unique()
        ),
        "acer_2023_total_curtailment_gwh": float(
            summary_lookup.loc[2023, "observed_res_curtailment_gwh"]
        ),
        "acer_2023_total_res_generation_gwh": float(
            summary_lookup.loc[2023, "observed_res_generation_gwh"]
        ),
        "acer_2023_aggregate_derived_share_pct": float(
            summary_lookup.loc[
                2023, "observed_aggregate_derived_share_pct"
            ]
        ),
        "acer_2024_absolute_volume_unavailable": bool(
            not summary_lookup.loc[2024, "absolute_scale_available"]
        ),
        "ireland_missing_not_zero_both_years": bool(
            len(ie) == 2
            and ie["observation_missing"].all()
            and ie["observed_res_curtailment_gwh"].isna().all()
        ),
        "annual_reference_has_no_hour_field": bool(
            not any(
                column in observations.columns
                for column in ("hour", "hour_of_year", "timestamp")
            )
        ),
        "all_hourly_allocation_flags_false": bool(
            not observations["hourly_allocation_permitted"].any()
            and not crosswalk["hourly_observation_join_permitted"].any()
        ),
        "metric_contract_limited_to_scale_and_ranking": bool(
            set(metrics_contract["comparison_dimension"]).issubset(
                {
                    "annual_scale",
                    "country_ranking",
                    "country_ranking_sensitivity",
                }
            )
        ),
        "eraa_zone_crosswalk_complete": bool(
            crosswalk["eraa_zone"].notna().all()
            and crosswalk["acer_country"].str.len().eq(2).all()
        ),
        "synthetic_self_test": self_test,
        "model_results_written": model_results_written,
        "model_input_sha256": model_input_sha256,
        "model_core_or_runner_modified": False,
    }
    qa["pass"] = bool(
        qa["observed_years"] == [2023, 2024]
        and np.isclose(qa["acer_2023_total_curtailment_gwh"], 12_764.0)
        and qa["acer_2024_absolute_volume_unavailable"]
        and qa["ireland_missing_not_zero_both_years"]
        and qa["annual_reference_has_no_hour_field"]
        and qa["all_hourly_allocation_flags_false"]
        and qa["metric_contract_limited_to_scale_and_ranking"]
        and qa["eraa_zone_crosswalk_complete"]
        and all(self_test.values())
        and not qa["model_core_or_runner_modified"]
    )
    (output / "acer_eraa_annual_calibration_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not qa["pass"]:
        raise AssertionError(json.dumps(qa, indent=2))
    return qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for annual calibration contract and optional results",
    )
    parser.add_argument(
        "--model-input",
        type=Path,
        default=None,
        help="Optional ERAA model annual or hourly output",
    )
    parser.add_argument(
        "--input-kind",
        choices=("annual", "hourly"),
        default="annual",
    )
    args = parser.parse_args()
    qa = run(args.output_dir, args.model_input, args.input_kind)
    print(json.dumps(qa, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
