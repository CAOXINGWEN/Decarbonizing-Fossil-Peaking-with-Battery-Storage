#!/usr/bin/env python3
"""Re-cost the frozen v39 European dispatch ledger in source-aware EUR.

The v39 hourly dispatch is independent of monetary assumptions.  This script
therefore preserves every physical output and recomputes only the economic
ledger, duration envelope and strict mid-duration-gap screen.

Requested technology-level currency routes
------------------------------------------
USD -> EUR:
    CAES, LAES, Fe-Cr flow, gravity, molten-salt TES and cavern hydrogen.
CNY -> EUR:
    LFP and VRFB.

All model-CNY values on the USD route are first returned to their historical
model-USD equivalent at 7.2 CNY USD-1 and then converted using the ECB 2025
annual average of 1.1300 USD EUR-1.  CNY-route values use the ECB 2025 annual
average of 8.1185043137255 CNY EUR-1.

Avoided-generation value, monetized firm capacity and charging opportunity
price are kept on the v39 CNY-assumption basis and converted CNY -> EUR.  They
are not technology-cost observations.  A separate report flag notes that the
firm-capacity value ultimately traces to a US ATB NGCC cost and deserves its
own provenance decision before final submission.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MANUSCRIPT_DIR = HERE.parents[1]
V39 = MANUSCRIPT_DIR / "revision_nature_energy_v39"
V39_RESULTS = V39 / "source_data" / "europe_mainline_results"
DATASET_ROOT = HERE.parents[5] / "dataset"
CATALOGUE = (
    DATASET_ROOT
    / "external_hourly"
    / "europe_mid_duration_valley_upgrade_v0_1"
    / "model_adapter"
    / "energy_charts_2015_2024_representative_pilot"
    / "configuration_catalog_53.csv"
)
ED1_PATHS_V36 = (
    MANUSCRIPT_DIR
    / "revision_nature_energy_v36"
    / "supplement"
    / "ed1_cost_paths_53"
    / "source_data"
    / "ed1_53_duration_year_cost_paths.csv"
)

MODEL_CNY_PER_USD = 7.2
ECB_2025_USD_PER_EUR = 1.1300
ECB_2025_CNY_PER_EUR = 8.1185043137255
EUR_PER_MODEL_USD = 1.0 / ECB_2025_USD_PER_EUR
EUR_PER_CNY = 1.0 / ECB_2025_CNY_PER_EUR

ECB_USD_SERIES = "EXR.A.USD.EUR.SP00.A"
ECB_CNY_SERIES = "EXR.A.CNY.EUR.SP00.A"
ECB_USD_URL = (
    "https://data.ecb.europa.eu/data/datasets/EXR/"
    "EXR.A.USD.EUR.SP00.A"
)
ECB_CNY_URL = (
    "https://data.ecb.europa.eu/data/datasets/EXR/"
    "EXR.A.CNY.EUR.SP00.A"
)

TECH_ROUTE = {
    "caes": "USD_to_EUR",
    "laes": "USD_to_EUR",
    "fecr_flow": "USD_to_EUR",
    "gravity": "USD_to_EUR",
    "molten_salt_tes": "USD_to_EUR",
    "h2_cavern": "USD_to_EUR",
    "li_lfp": "CNY_to_EUR",
    "vrfb": "CNY_to_EUR",
}

TECH_BASIS = {
    "caes": "USD components with author-selected learning path",
    "laes": "model-USD proxy from predominantly US evidence",
    "fecr_flow": "model-USD proxy rather than a raw technology quotation",
    "gravity": "model-USD proxy from predominantly US evidence",
    "molten_salt_tes": "model-USD proxy from US thermal-storage evidence",
    "h2_cavern": "model-USD proxy from US hydrogen-storage evidence",
    "li_lfp": "CNY-direct or hybrid parameter anchored to China prices",
    "vrfb": "CNY-direct parameter calibrated to China EPC tenders",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            json_ready(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def route_factor(route: pd.Series) -> np.ndarray:
    """Return EUR per model-CNY for the assigned route."""
    return np.where(
        route.eq("USD_to_EUR"),
        1.0 / (MODEL_CNY_PER_USD * ECB_2025_USD_PER_EUR),
        EUR_PER_CNY,
    )


def source_aware_catalogue(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    frame["technology_currency_route"] = frame["tech_id"].map(TECH_ROUTE)
    frame["technology_currency_basis"] = frame["tech_id"].map(TECH_BASIS)
    if frame["technology_currency_route"].isna().any():
        missing = sorted(
            frame.loc[
                frame["technology_currency_route"].isna(), "tech_id"
            ].unique()
        )
        raise RuntimeError(f"Unmapped technologies: {missing}")
    factor = route_factor(frame["technology_currency_route"])
    money = {
        "fixed_cost_cny_per_kwh_year": (
            "fixed_cost_eur_per_kwh_year_source_aware"
        ),
        "variable_om_cny_mwh": "variable_om_eur_mwh_source_aware",
        "annual_power_cost_cny_per_kw_year": (
            "annual_power_cost_eur_per_kw_year_source_aware"
        ),
        "power_cost_cny_kw": "power_cost_eur_kw_source_aware",
        "energy_cost_cny_kwh": "energy_cost_eur_kwh_source_aware",
    }
    for source_name, target_name in money.items():
        frame[target_name] = pd.to_numeric(
            frame[source_name], errors="raise"
        ) * factor
    frame["total_capex_cny_per_kwh"] = (
        frame["power_cost_cny_kw"] / frame["duration_h"]
        + frame["energy_cost_cny_kwh"]
    )
    frame["total_capex_eur_per_kwh_source_aware"] = (
        frame["power_cost_eur_kw_source_aware"] / frame["duration_h"]
        + frame["energy_cost_eur_kwh_source_aware"]
    )
    frame["annual_fixed_cost_cny_per_kw_year"] = (
        frame["fixed_cost_cny_per_kwh_year"] * frame["duration_h"]
    )
    frame["annual_fixed_cost_eur_per_kw_year_source_aware"] = (
        frame["fixed_cost_eur_per_kwh_year_source_aware"]
        * frame["duration_h"]
    )
    frame["model_fx_cny_per_usd"] = MODEL_CNY_PER_USD
    frame["reporting_fx_usd_per_eur"] = ECB_2025_USD_PER_EUR
    frame["reporting_fx_cny_per_eur"] = ECB_2025_CNY_PER_EUR
    frame["reporting_fx_reference_year"] = 2025
    return frame


def source_aware_ed1_paths(source: pd.DataFrame) -> pd.DataFrame:
    """Convert the 53-configuration, three-anchor-year ED1 cost ledger."""
    frame = source.copy()
    frame["technology_currency_route"] = frame["tech_id"].map(TECH_ROUTE)
    frame["technology_currency_basis"] = frame["tech_id"].map(TECH_BASIS)
    if frame["technology_currency_route"].isna().any():
        raise RuntimeError("ED1 cost paths contain an unmapped technology")
    factor = route_factor(frame["technology_currency_route"])
    money_columns = [
        "power_cost_cny_kw",
        "energy_cost_cny_kwh",
        "fixed_om_cny_kw_yr",
        "variable_om_cny_mwh",
        "total_capex_cny_kwh",
        "annualized_capex_cny_per_kwh_capacity_year",
        "fixed_om_cny_per_kwh_capacity_year",
        "replacement_pv_cny_per_kwh",
        "annual_replacement_reserve_cny_per_kwh_year",
        "annual_fixed_cost_cny_per_kwh_capacity_year",
        "annual_fixed_cost_cny_per_kw_discharge_year",
        "annualized_power_capex_cny_per_kw_year",
        "annualized_energy_capex_cny_per_kwh_year",
        "annual_replacement_reserve_cny_per_kw_year",
    ]
    for column in money_columns:
        target = column.replace("_cny_", "_eur_source_aware_")
        frame[target] = pd.to_numeric(frame[column], errors="raise") * factor
    frame["reporting_fx_reference_year"] = 2025
    frame["reporting_fx_usd_per_eur"] = ECB_2025_USD_PER_EUR
    frame["reporting_fx_cny_per_eur"] = ECB_2025_CNY_PER_EUR

    total_error = (
        frame["total_capex_eur_source_aware_kwh"]
        - frame["power_cost_eur_source_aware_kw"] / frame["duration_h"]
        - frame["energy_cost_eur_source_aware_kwh"]
    )
    fixed_error = (
        frame[
            "annual_fixed_cost_eur_source_aware_per_kwh_capacity_year"
        ]
        - frame[
            "annualized_capex_eur_source_aware_per_kwh_capacity_year"
        ]
        - frame["fixed_om_eur_source_aware_per_kwh_capacity_year"]
        - frame[
            "annual_replacement_reserve_eur_source_aware_per_kwh_year"
        ]
    )
    frame["total_capex_eur_source_aware_recalculation_error"] = total_error
    frame[
        "annual_fixed_cost_eur_source_aware_recalculation_error"
    ] = fixed_error
    if max(total_error.abs().max(), fixed_error.abs().max()) > 1.0e-10:
        raise RuntimeError("Source-aware ED1 cost identity failed")
    return frame


def recost_configuration_results(
    source: pd.DataFrame,
) -> pd.DataFrame:
    frame = source.copy()
    frame["technology_currency_route"] = frame["tech_id"].map(TECH_ROUTE)
    frame["technology_currency_basis"] = frame["tech_id"].map(TECH_BASIS)
    if frame["technology_currency_route"].isna().any():
        raise RuntimeError("At least one result row has an unmapped technology")
    tech_factor = route_factor(frame["technology_currency_route"])

    # Retain the uniform v39 conversion as an explicit comparator.
    frame["annual_fixed_cost_eur_per_kw_year_uniform_v39"] = (
        frame["annual_fixed_cost_cny_per_kw_year"] * EUR_PER_CNY
    )
    frame["storage_vom_eur_per_kw_year_uniform_v39"] = (
        frame["storage_vom_cny_per_kw_year"] * EUR_PER_CNY
    )
    frame["joint_net_benefit_eur_per_kw_year_uniform_v39"] = (
        frame["joint_net_benefit_cny_per_kw_year"] * EUR_PER_CNY
    )

    # Requested strict technology-level route.
    frame["annual_fixed_cost_eur_per_kw_year_source_aware"] = (
        frame["annual_fixed_cost_cny_per_kw_year"] * tech_factor
    )
    frame["storage_vom_eur_per_kw_year_source_aware"] = (
        frame["storage_vom_cny_per_kw_year"] * tech_factor
    )

    # Benefits and charging price preserve the v39 valuation assumptions.
    frame["charge_opportunity_cost_eur_per_kw_year"] = (
        frame["charge_opportunity_cost_cny_per_kw_year"] * EUR_PER_CNY
    )
    frame["energy_displacement_value_eur_per_kw_year"] = (
        frame["energy_displacement_value_cny_per_kw_year"] * EUR_PER_CNY
    )
    frame["firm_capacity_value_eur_per_kw_year"] = (
        frame["firm_capacity_value_cny_per_kw_year"] * EUR_PER_CNY
    )
    frame["turnover_net_benefit_eur_per_kw_year_source_aware"] = (
        frame["energy_displacement_value_eur_per_kw_year"]
        - frame["annual_fixed_cost_eur_per_kw_year_source_aware"]
        - frame["storage_vom_eur_per_kw_year_source_aware"]
        - frame["charge_opportunity_cost_eur_per_kw_year"]
    )
    frame["joint_net_benefit_eur_per_kw_year_source_aware"] = (
        frame["turnover_net_benefit_eur_per_kw_year_source_aware"]
        + frame["firm_capacity_value_eur_per_kw_year"]
    )

    # Sensitivity ledger: technology-specific route for fixed cost only,
    # while variable O&M remains on its curated CNY model basis.
    frame["turnover_net_benefit_eur_per_kw_year_fixed_only"] = (
        frame["energy_displacement_value_eur_per_kw_year"]
        - frame["annual_fixed_cost_eur_per_kw_year_source_aware"]
        - frame["storage_vom_eur_per_kw_year_uniform_v39"]
        - frame["charge_opportunity_cost_eur_per_kw_year"]
    )
    frame["joint_net_benefit_eur_per_kw_year_fixed_only"] = (
        frame["turnover_net_benefit_eur_per_kw_year_fixed_only"]
        + frame["firm_capacity_value_eur_per_kw_year"]
    )
    frame["source_aware_minus_uniform_eur_per_kw_year"] = (
        frame["joint_net_benefit_eur_per_kw_year_source_aware"]
        - frame["joint_net_benefit_eur_per_kw_year_uniform_v39"]
    )
    frame["model_fx_cny_per_usd"] = MODEL_CNY_PER_USD
    frame["reporting_fx_usd_per_eur"] = ECB_2025_USD_PER_EUR
    frame["reporting_fx_cny_per_eur"] = ECB_2025_CNY_PER_EUR
    frame["reporting_fx_reference_year"] = 2025
    frame["reporting_fx_usd_series_key"] = ECB_USD_SERIES
    frame["reporting_fx_cny_series_key"] = ECB_CNY_SERIES
    return frame


def build_paired(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["country", "year", "config_id"]
    columns = [
        "tech_id",
        "technology_en",
        "duration_h",
        "joint_net_benefit_eur_per_kw_year_uniform_v39",
        "joint_net_benefit_eur_per_kw_year_source_aware",
        "joint_net_benefit_eur_per_kw_year_fixed_only",
    ]
    dispersed = (
        results[results["chronology"].eq("dispersed")]
        .set_index(keys)[columns]
        .add_prefix("dispersed_")
    )
    clustered = (
        results[results["chronology"].eq("clustered")]
        .set_index(keys)[columns]
        .add_prefix("clustered_")
    )
    paired = dispersed.join(clustered, how="inner", validate="one_to_one")
    for name in [
        "tech_id",
        "technology_en",
        "duration_h",
    ]:
        left = f"dispersed_{name}"
        right = f"clustered_{name}"
        if not paired[left].equals(paired[right]):
            raise RuntimeError(f"Paired identity mismatch: {name}")
        paired[name] = paired[left]
        paired = paired.drop(columns=[left, right])
    for mode in ["uniform_v39", "source_aware", "fixed_only"]:
        paired[f"delta_joint_net_benefit_eur_per_kw_year_{mode}"] = (
            paired[
                f"clustered_joint_net_benefit_eur_per_kw_year_{mode}"
            ]
            - paired[
                f"dispersed_joint_net_benefit_eur_per_kw_year_{mode}"
            ]
        )
    paired["source_aware_minus_uniform_delta_eur_per_kw_year"] = (
        paired["delta_joint_net_benefit_eur_per_kw_year_source_aware"]
        - paired["delta_joint_net_benefit_eur_per_kw_year_uniform_v39"]
    )
    paired = paired.reset_index()
    summary = (
        paired.groupby(
            ["config_id", "tech_id", "technology_en", "duration_h"],
            as_index=False,
        )
        .agg(
            country_years=("country", "size"),
            uniform_delta_mean_eur_per_kw_year=(
                "delta_joint_net_benefit_eur_per_kw_year_uniform_v39",
                "mean",
            ),
            source_aware_delta_mean_eur_per_kw_year=(
                "delta_joint_net_benefit_eur_per_kw_year_source_aware",
                "mean",
            ),
            mean_delta_change_eur_per_kw_year=(
                "source_aware_minus_uniform_delta_eur_per_kw_year",
                "mean",
            ),
            max_abs_delta_change_eur_per_kw_year=(
                "source_aware_minus_uniform_delta_eur_per_kw_year",
                lambda values: float(np.max(np.abs(values))),
            ),
        )
        .sort_values(["duration_h", "tech_id"], kind="stable")
    )
    return paired, summary


def strict_gap_records(envelope: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in envelope.groupby(
        ["scenario_id", "country", "year", "chronology"], sort=False
    ):
        ordered = group.sort_values("duration_h", kind="stable")
        durations = ordered["duration_h"].to_numpy(float)
        positive = (
            ordered[
                "net_benefit_eur_per_kw_year_source_aware"
            ].to_numpy(float)
            > 0.0
        )
        gaps: list[tuple[float, float, float]] = []
        for position in range(1, len(positive) - 1):
            if positive[position]:
                continue
            left_positive = np.flatnonzero(positive[:position])
            right_positive = np.flatnonzero(positive[position + 1 :])
            if len(left_positive) and len(right_positive):
                left = int(left_positive[-1])
                right = int(position + 1 + right_positive[0])
                gaps.append(
                    (
                        float(durations[right] - durations[left]),
                        float(durations[left]),
                        float(durations[right]),
                    )
                )
        if gaps:
            width, left_h, right_h = max(gaps, key=lambda item: item[0])
        else:
            width, left_h, right_h = 0.0, np.nan, np.nan
        records.append(
            {
                "scenario_id": int(keys[0]),
                "country": keys[1],
                "year": int(keys[2]),
                "chronology": keys[3],
                "has_strict_gap": bool(gaps),
                "gap_width_h": width,
                "left_positive_duration_h": left_h,
                "right_reentry_duration_h": right_h,
            }
        )
    return pd.DataFrame(records)


def recost_value_scan(
    results: pd.DataFrame,
    old_envelope: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    physical = results[
        results["chronology"].isin(["dispersed", "clustered"])
        & results["duration_h"].le(168.0)
    ].copy()
    scenario_columns = [
        "scenario_id",
        "energy_value_cny_mwh",
        "firm_value_cny_kw_year",
        "charge_price_cny_mwh",
    ]
    scenarios = (
        old_envelope[scenario_columns]
        .drop_duplicates()
        .sort_values("scenario_id")
    )
    rows: list[pd.DataFrame] = []
    for scenario in scenarios.itertuples(index=False):
        current = physical[
            [
                "country",
                "year",
                "chronology",
                "config_id",
                "tech_id",
                "technology_en",
                "duration_h",
                "pdis_mw",
                "charge_mwh",
                "discharge_mwh",
                "efc_mw",
                "annual_fixed_cost_eur_per_kw_year_source_aware",
                "storage_vom_eur_per_kw_year_source_aware",
            ]
        ].copy()
        current["scenario_id"] = int(scenario.scenario_id)
        current["energy_value_eur_mwh"] = (
            float(scenario.energy_value_cny_mwh) * EUR_PER_CNY
        )
        current["firm_value_eur_kw_year"] = (
            float(scenario.firm_value_cny_kw_year) * EUR_PER_CNY
        )
        current["charge_price_eur_mwh"] = (
            float(scenario.charge_price_cny_mwh) * EUR_PER_CNY
        )
        current["energy_value_cny_mwh_v39_assumption"] = float(
            scenario.energy_value_cny_mwh
        )
        current["firm_value_cny_kw_year_v39_assumption"] = float(
            scenario.firm_value_cny_kw_year
        )
        current["charge_price_cny_mwh_v39_assumption"] = float(
            scenario.charge_price_cny_mwh
        )
        current["net_benefit_eur_per_kw_year_source_aware"] = (
            current["discharge_mwh"]
            * current["energy_value_eur_mwh"]
            / (current["pdis_mw"] * 1000.0)
            + current["efc_mw"]
            / current["pdis_mw"]
            * current["firm_value_eur_kw_year"]
            - current["annual_fixed_cost_eur_per_kw_year_source_aware"]
            - current["storage_vom_eur_per_kw_year_source_aware"]
            - current["charge_mwh"]
            * current["charge_price_eur_mwh"]
            / (current["pdis_mw"] * 1000.0)
        )
        envelope = (
            current.sort_values(
                [
                    "country",
                    "year",
                    "chronology",
                    "duration_h",
                    "net_benefit_eur_per_kw_year_source_aware",
                ],
                ascending=[True, True, True, True, False],
                kind="stable",
            )
            .drop_duplicates(
                ["country", "year", "chronology", "duration_h"]
            )
        )
        rows.append(envelope)
    envelope = pd.concat(rows, ignore_index=True)
    gap_records = strict_gap_records(envelope)
    gap_summary = (
        gap_records.groupby(
            ["scenario_id", "chronology"], as_index=False
        )
        .agg(
            country_years_with_gap=("has_strict_gap", "sum"),
            country_years=("has_strict_gap", "size"),
            mean_gap_width_h=("gap_width_h", "mean"),
            median_gap_width_h=("gap_width_h", "median"),
            mean_left_boundary_h=("left_positive_duration_h", "mean"),
            mean_right_reentry_h=("right_reentry_duration_h", "mean"),
        )
        .merge(
            scenarios.assign(
                energy_value_eur_mwh=lambda x: (
                    x["energy_value_cny_mwh"] * EUR_PER_CNY
                ),
                firm_value_eur_kw_year=lambda x: (
                    x["firm_value_cny_kw_year"] * EUR_PER_CNY
                ),
                charge_price_eur_mwh=lambda x: (
                    x["charge_price_cny_mwh"] * EUR_PER_CNY
                ),
            ),
            on="scenario_id",
            how="left",
            validate="many_to_one",
        )
    )

    old_choice = old_envelope[
        [
            "scenario_id",
            "country",
            "year",
            "chronology",
            "duration_h",
            "config_id",
            "net_benefit_cny_per_kw_year",
        ]
    ].rename(
        columns={
            "config_id": "uniform_v39_best_config_id",
            "net_benefit_cny_per_kw_year": (
                "uniform_v39_best_net_benefit_cny_per_kw_year"
            ),
        }
    )
    comparison = envelope.merge(
        old_choice,
        on=[
            "scenario_id",
            "country",
            "year",
            "chronology",
            "duration_h",
        ],
        how="left",
        validate="one_to_one",
    )
    comparison["best_configuration_changed"] = (
        comparison["config_id"]
        != comparison["uniform_v39_best_config_id"]
    )
    comparison["uniform_v39_positive"] = (
        comparison["uniform_v39_best_net_benefit_cny_per_kw_year"] > 0.0
    )
    comparison["source_aware_positive"] = (
        comparison["net_benefit_eur_per_kw_year_source_aware"] > 0.0
    )
    comparison["entry_sign_changed"] = (
        comparison["uniform_v39_positive"]
        != comparison["source_aware_positive"]
    )
    return envelope, gap_records, gap_summary, comparison


def main() -> None:
    catalogue_raw = pd.read_csv(CATALOGUE)
    ed1_paths_raw = pd.read_csv(ED1_PATHS_V36)
    results_raw = pd.read_csv(V39_RESULTS / "configuration_results.csv.gz")
    old_envelope = pd.read_csv(
        V39_RESULTS / "value_sensitivity_envelope.csv.gz"
    )
    old_gap = pd.read_csv(
        V39_RESULTS / "value_sensitivity_gap_summary.csv"
    )

    catalogue = source_aware_catalogue(catalogue_raw)
    ed1_paths = source_aware_ed1_paths(ed1_paths_raw)
    results = recost_configuration_results(results_raw)
    paired, paired_summary = build_paired(results)
    envelope, gap_records, gap_summary, choice_comparison = (
        recost_value_scan(results, old_envelope)
    )
    gap_comparison = gap_summary.merge(
        old_gap.rename(
            columns={
                "country_years_with_gap": (
                    "uniform_v39_country_years_with_gap"
                ),
                "country_years": "uniform_v39_country_years",
                "mean_gap_width_h": "uniform_v39_mean_gap_width_h",
            }
        ),
        on=["scenario_id", "chronology"],
        how="left",
        validate="one_to_one",
    )
    gap_comparison["gap_count_change"] = (
        gap_comparison["country_years_with_gap"]
        - gap_comparison["uniform_v39_country_years_with_gap"]
    )
    gap_comparison["mean_gap_width_change_h"] = (
        gap_comparison["mean_gap_width_h"]
        - gap_comparison["uniform_v39_mean_gap_width_h"]
    )

    outputs = {
        "configuration_catalog_53_source_aware_eur.csv": catalogue,
        "ed1_53_duration_year_cost_paths_source_aware_eur.csv": ed1_paths,
        "configuration_results_source_aware_eur.csv.gz": results,
        "paired_configuration_changes_source_aware_eur.csv.gz": paired,
        "paired_configuration_summary_source_aware_eur.csv": paired_summary,
        "value_sensitivity_envelope_source_aware_eur.csv.gz": envelope,
        "value_sensitivity_gap_records_source_aware_eur.csv.gz": gap_records,
        "value_sensitivity_gap_summary_source_aware_eur.csv": gap_summary,
        "value_sensitivity_gap_comparison_v39_v40.csv": gap_comparison,
        "value_sensitivity_choice_comparison_v39_v40.csv.gz": (
            choice_comparison
        ),
    }
    for filename, frame in outputs.items():
        compression = (
            {"method": "gzip", "mtime": 0}
            if filename.endswith(".gz")
            else None
        )
        frame.to_csv(HERE / filename, index=False, compression=compression)

    usd_result = results["technology_currency_route"].eq("USD_to_EUR")
    scenario31 = gap_comparison[
        gap_comparison["scenario_id"].eq(31)
    ].sort_values("chronology")
    catalogue_ed1 = catalogue.merge(
        ed1_paths[ed1_paths["learning_year"].eq(2035)][
            [
                "tech_id",
                "duration_h",
                "total_capex_eur_source_aware_kwh",
                (
                    "annual_fixed_cost_eur_source_aware_"
                    "per_kwh_capacity_year"
                ),
                "variable_om_eur_source_aware_mwh",
            ]
        ],
        on=["tech_id", "duration_h"],
        how="left",
        validate="one_to_one",
    )
    checks = {
        "result_rows_preserved": len(results) == len(results_raw),
        "configuration_count": int(catalogue["config_id"].nunique()),
        "technology_count": int(catalogue["tech_id"].nunique()),
        "ed1_cost_path_rows": int(len(ed1_paths)),
        "ed1_cost_path_configurations": int(
            ed1_paths[["tech_id", "duration_h"]].drop_duplicates().shape[0]
        ),
        "ed1_cost_path_anchor_years": sorted(
            int(value) for value in ed1_paths["learning_year"].unique()
        ),
        "currency_routes_complete": not results[
            "technology_currency_route"
        ].isna().any(),
        "physical_columns_byte_equivalent_in_memory": all(
            results[column].equals(results_raw[column])
            for column in [
                "charge_mwh",
                "discharge_mwh",
                "unserved_mwh",
                "spill_mwh",
                "served_share",
                "equivalent_full_cycles",
                "efc_mw",
                "efc_ratio",
                "cyclic_soc_residual_mwh",
            ]
        ),
        "maximum_abs_uniform_conversion_error": float(
            np.max(
                np.abs(
                    results[
                        "joint_net_benefit_eur_per_kw_year_uniform_v39"
                    ]
                    - results["joint_net_benefit_cny_per_kw_year"]
                    * EUR_PER_CNY
                )
            )
        ),
        "maximum_abs_source_aware_fixed_cost_formula_error": float(
            np.max(
                np.abs(
                    results[
                        "annual_fixed_cost_eur_per_kw_year_source_aware"
                    ]
                    - results["annual_fixed_cost_cny_per_kw_year"]
                    * route_factor(results["technology_currency_route"])
                )
            )
        ),
        "maximum_abs_ed1_total_capex_identity_error": float(
            ed1_paths[
                "total_capex_eur_source_aware_recalculation_error"
            ].abs().max()
        ),
        "maximum_abs_ed1_fixed_cost_identity_error": float(
            ed1_paths[
                "annual_fixed_cost_eur_source_aware_recalculation_error"
            ].abs().max()
        ),
        "maximum_abs_catalogue_vs_ed1_total_capex_error": float(
            (
                catalogue_ed1["total_capex_eur_per_kwh_source_aware"]
                - catalogue_ed1["total_capex_eur_source_aware_kwh"]
            )
            .abs()
            .max()
        ),
        "maximum_abs_catalogue_vs_ed1_fixed_cost_error": float(
            (
                catalogue_ed1[
                    "fixed_cost_eur_per_kwh_year_source_aware"
                ]
                - catalogue_ed1[
                    "annual_fixed_cost_eur_source_aware_"
                    "per_kwh_capacity_year"
                ]
            )
            .abs()
            .max()
        ),
        "usd_route_cost_reduction_pct_vs_uniform_cny_route": float(
            (
                ECB_2025_CNY_PER_EUR
                / (MODEL_CNY_PER_USD * ECB_2025_USD_PER_EUR)
                - 1.0
            )
            * 100.0
        ),
        "maximum_source_aware_net_benefit_change_eur_per_kw_year": float(
            results["source_aware_minus_uniform_eur_per_kw_year"].max()
        ),
        "minimum_source_aware_net_benefit_change_eur_per_kw_year": float(
            results["source_aware_minus_uniform_eur_per_kw_year"].min()
        ),
        "usd_route_result_rows": int(usd_result.sum()),
        "cny_route_result_rows": int((~usd_result).sum()),
        "best_configuration_switches_across_all_value_cells": int(
            choice_comparison["best_configuration_changed"].sum()
        ),
        "entry_sign_switches_across_all_value_cells": int(
            choice_comparison["entry_sign_changed"].sum()
        ),
        "gap_count_cells_changed": int(
            gap_comparison["gap_count_change"].ne(0).sum()
        ),
        "gap_width_cells_changed": int(
            ~np.isclose(
                gap_comparison["mean_gap_width_change_h"],
                0.0,
                atol=1.0e-12,
            ).sum()
        ),
        "scenario_31": scenario31.to_dict(orient="records"),
    }
    # Correct the boolean-count expression, which cannot be written cleanly
    # inside the JSON literal without obscuring its meaning.
    checks["gap_width_cells_changed"] = int(
        (
            ~np.isclose(
                gap_comparison["mean_gap_width_change_h"].to_numpy(float),
                0.0,
                atol=1.0e-12,
            )
        ).sum()
    )
    for name, status in {
        "result_rows_preserved": checks["result_rows_preserved"],
        "currency_routes_complete": checks["currency_routes_complete"],
        "physical_columns_byte_equivalent_in_memory": checks[
            "physical_columns_byte_equivalent_in_memory"
        ],
        "uniform_conversion_identity": (
            checks["maximum_abs_uniform_conversion_error"] < 1.0e-10
        ),
        "source_aware_fixed_cost_identity": (
            checks["maximum_abs_source_aware_fixed_cost_formula_error"]
            < 1.0e-10
        ),
        "ed1_total_capex_identity": (
            checks["maximum_abs_ed1_total_capex_identity_error"] < 1.0e-10
        ),
        "ed1_fixed_cost_identity": (
            checks["maximum_abs_ed1_fixed_cost_identity_error"] < 1.0e-10
        ),
        "catalogue_ed1_total_capex_identity": (
            checks["maximum_abs_catalogue_vs_ed1_total_capex_error"]
            < 1.0e-10
        ),
        "catalogue_ed1_fixed_cost_identity": (
            checks["maximum_abs_catalogue_vs_ed1_fixed_cost_error"]
            < 1.0e-10
        ),
    }.items():
        if not status:
            raise RuntimeError(f"Validation failed: {name}")

    summary = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Independent v40 source-aware EUR re-costing of frozen v39 "
            "European physical dispatch outputs"
        ),
        "technology_routes": TECH_ROUTE,
        "exchange_rates": {
            "historical_model_cny_per_usd": MODEL_CNY_PER_USD,
            "ecb_2025_usd_per_eur": ECB_2025_USD_PER_EUR,
            "ecb_2025_cny_per_eur": ECB_2025_CNY_PER_EUR,
            "ecb_usd_series": ECB_USD_SERIES,
            "ecb_cny_series": ECB_CNY_SERIES,
            "ecb_usd_url": ECB_USD_URL,
            "ecb_cny_url": ECB_CNY_URL,
        },
        "value_item_route": (
            "v39 CNY assumptions converted CNY_to_EUR; technology cost "
            "routes do not automatically determine avoided-generation, "
            "capacity-value or charging-price provenance"
        ),
        "checks": checks,
        "input_files": {
            str(CATALOGUE): sha256(CATALOGUE),
            str(ED1_PATHS_V36): sha256(ED1_PATHS_V36),
            str(V39_RESULTS / "configuration_results.csv.gz"): sha256(
                V39_RESULTS / "configuration_results.csv.gz"
            ),
            str(V39_RESULTS / "value_sensitivity_envelope.csv.gz"): sha256(
                V39_RESULTS / "value_sensitivity_envelope.csv.gz"
            ),
            str(
                V39_RESULTS / "value_sensitivity_gap_summary.csv"
            ): sha256(V39_RESULTS / "value_sensitivity_gap_summary.csv"),
        },
        "output_files": {},
    }
    for filename in sorted(outputs):
        path = HERE / filename
        summary["output_files"][filename] = {
            "rows": int(len(outputs[filename])),
            "sha256": sha256(path),
        }
    write_json(HERE / "source_aware_currency_integration_summary.json", summary)
    print(json.dumps(json_ready(summary["checks"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
