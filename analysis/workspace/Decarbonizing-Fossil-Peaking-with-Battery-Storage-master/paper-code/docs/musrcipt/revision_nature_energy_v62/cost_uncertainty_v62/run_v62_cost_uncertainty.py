#!/usr/bin/env python3
"""Source-bounded uncertainty propagation for the 24 h A-CAES--4 h LFP comparator.

The experiment uses a balanced factorial ensemble instead of assigning
occurrence probabilities to technology-cost projections. Coherent PNNL ESGC
Low/Point/High estimates are retained for each technology. The focal China LFP
anchor is scaled by the corresponding PNNL ratios, while the released LFP and
A-CAES learning paths define the 2035 bounds.
"""

from __future__ import annotations

from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parents[4]
DATASET = WORKSPACE / "dataset"
RESULTS = HERE / "results"

COST_CATALOGUE = ROOT / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
ESGC_DATABASE = DATASET / "ESGC_Cost_Performance_Database_v2024.xlsx"
LEARNING_CURVES = DATASET / "M2_LDES_learning_cost_curves_v0_7.csv"
NATURAL_PAIRS = ROOT / "natural_year_heatmap" / "source_data" / "natural_year_transition_pairs.csv"

DISCOUNT_RATES = (0.020, 0.036, 0.060, 0.080)
CENTRAL_DISCOUNT_RATE = 0.036
A_CAES_DURATION_H = 24.0
LFP_DURATION_H = 4.0
A_CAES_LIFE_YR = 60
LFP_LIFE_YR = (15, 16)
A_CAES_LEARNING_REDUCTION_2030 = {
    "faster": 0.25,
    "reference": None,
    "slower": 0.06,
}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def crf(rate: float, life_years: int) -> float:
    return rate * (1.0 + rate) ** life_years / ((1.0 + rate) ** life_years - 1.0)


def catalogue_rows() -> pd.DataFrame:
    table = pd.read_csv(COST_CATALOGUE)
    focal = table[
        table["config_id"].isin(("caes_24h", "li_lfp_4h"))
        & table["learning_year"].isin((2025, 2035))
    ].copy()
    if len(focal) != 4:
        raise AssertionError(f"Expected four focal catalogue rows, found {len(focal)}")
    if focal.duplicated(["config_id", "learning_year"]).any():
        raise AssertionError("Duplicate focal catalogue keys")
    return focal.set_index(["config_id", "learning_year"])


def esgc_value(table: pd.DataFrame, tech: str, estimate: str, parameter: str) -> float:
    duration = A_CAES_DURATION_H if tech == "CAES" else LFP_DURATION_H
    row = table[
        table["Technology"].eq(tech)
        & table["Year"].eq(2023)
        & table["Power_MW"].eq(100)
        & table["Duration_hr"].eq(duration)
        & table["Estimate_type"].eq(estimate)
        & table["Parameter"].eq(parameter)
    ]
    if len(row) != 1:
        raise AssertionError((tech, estimate, parameter, len(row)))
    return float(row.iloc[0]["Value"])


def build_inputs(
    catalogue: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, float],
    dict[str, float],
]:
    esgc = pd.read_excel(ESGC_DATABASE, sheet_name="Database")
    estimate_order = ("Low", "Point", "High")

    caes_point = catalogue.loc[("caes_24h", 2025)]
    usd_per_eur = esgc_value(
        esgc,
        "CAES",
        "Point",
        "CAES Capital (Turbine, Compressor, BOP, & EPC Management) ($/kW)",
    ) / float(caes_point["power_cost_eur_source_aware_kw"])

    caes_estimates: dict[str, dict[str, float]] = {}
    for estimate in estimate_order:
        caes_estimates[estimate] = {
            "power_capex_eur_kw": esgc_value(
                esgc,
                "CAES",
                estimate,
                "CAES Capital (Turbine, Compressor, BOP, & EPC Management) ($/kW)",
            )
            / usd_per_eur,
            "energy_capex_eur_kwh": esgc_value(
                esgc, "CAES", estimate, "Cavern Storage ($/kWh)"
            )
            / usd_per_eur,
            "fixed_om_eur_kw_yr": esgc_value(
                esgc, "CAES", estimate, "Fixed O&M ($/kW-year)"
            )
            / usd_per_eur,
        }

    lfp_point = catalogue.loc[("li_lfp_4h", 2025)]
    lfp_esgc_point_capex = esgc_value(
        esgc, "Lithium-ion LFP", "Point", "Total Installed Cost ($/kWh)"
    )
    lfp_esgc_point_fom = esgc_value(
        esgc, "Lithium-ion LFP", "Point", "Fixed O&M ($/kW-year)"
    )
    lfp_estimates: dict[str, dict[str, float]] = {}
    for estimate in estimate_order:
        capex_ratio = esgc_value(
            esgc, "Lithium-ion LFP", estimate, "Total Installed Cost ($/kWh)"
        ) / lfp_esgc_point_capex
        fom_ratio = esgc_value(
            esgc, "Lithium-ion LFP", estimate, "Fixed O&M ($/kW-year)"
        ) / lfp_esgc_point_fom
        lfp_estimates[estimate] = {
            "energy_capex_eur_kwh": float(
                lfp_point["total_capex_eur_source_aware_kwh"]
            )
            * capex_ratio,
            "fixed_om_eur_kw_yr": float(
                lfp_point["fixed_om_eur_source_aware_kw_yr"]
            )
            * fom_ratio,
        }

    caes_2035_factor = float(
        catalogue.loc[("caes_24h", 2035), "power_cost_eur_source_aware_kw"]
        / catalogue.loc[("caes_24h", 2025), "power_cost_eur_source_aware_kw"]
    )
    caes_learning = {
        "faster": (1.0 - A_CAES_LEARNING_REDUCTION_2030["faster"]) ** 2,
        "reference": caes_2035_factor,
        "slower": (1.0 - A_CAES_LEARNING_REDUCTION_2030["slower"]) ** 2,
    }

    curves = pd.read_csv(LEARNING_CURVES, encoding="utf-8-sig")
    lfp_learning: dict[str, float] = {
        "reference": float(
            catalogue.loc[("li_lfp_4h", 2035), "total_capex_eur_source_aware_kwh"]
            / catalogue.loc[("li_lfp_4h", 2025), "total_capex_eur_source_aware_kwh"]
        )
    }
    for label, case in (("conservative", "conservative_lfp"),):
        row = curves[
            curves["tech_id"].eq("li_lfp")
            & curves["curve_case"].eq(case)
            & curves["learning_year"].eq(2035)
        ]
        if len(row) != 1:
            raise AssertionError((label, case, len(row)))
        lfp_learning[label] = float(row.iloc[0]["energy_cost_factor_vs_2025"])

    records: list[dict[str, object]] = []
    for estimate, values in caes_estimates.items():
        records.extend(
            [
                {
                    "technology": "24 h A-CAES",
                    "factor": "coherent ESGC estimate",
                    "level": estimate,
                    "power_capex_eur_kw": values["power_capex_eur_kw"],
                    "energy_capex_eur_kwh": values["energy_capex_eur_kwh"],
                    "fixed_om_eur_kw_yr": values["fixed_om_eur_kw_yr"],
                    "calendar_life_years": A_CAES_LIFE_YR,
                    "learning_factor_2035_vs_2025": np.nan,
                }
            ]
        )
    for estimate, values in lfp_estimates.items():
        records.extend(
            [
                {
                    "technology": "4 h LFP",
                    "factor": "China anchor scaled by ESGC estimate ratio",
                    "level": estimate,
                    "power_capex_eur_kw": 0.0,
                    "energy_capex_eur_kwh": values["energy_capex_eur_kwh"],
                    "fixed_om_eur_kw_yr": values["fixed_om_eur_kw_yr"],
                    "calendar_life_years": "15|16",
                    "learning_factor_2035_vs_2025": np.nan,
                }
            ]
        )
    for label, factor in caes_learning.items():
        records.append(
            {
                "technology": "24 h A-CAES",
                "factor": "2035 CAPEX learning",
                "level": label,
                "power_capex_eur_kw": np.nan,
                "energy_capex_eur_kwh": np.nan,
                "fixed_om_eur_kw_yr": np.nan,
                "calendar_life_years": A_CAES_LIFE_YR,
                "learning_factor_2035_vs_2025": factor,
            }
        )
    for label, factor in lfp_learning.items():
        records.append(
            {
                "technology": "4 h LFP",
                "factor": "2035 CAPEX learning",
                "level": label,
                "power_capex_eur_kw": 0.0,
                "energy_capex_eur_kwh": np.nan,
                "fixed_om_eur_kw_yr": np.nan,
                "calendar_life_years": "15|16",
                "learning_factor_2035_vs_2025": factor,
            }
        )
    return pd.DataFrame(records), caes_estimates, lfp_estimates, caes_learning, lfp_learning


def annual_costs(
    rate: float,
    caes_estimate: dict[str, float],
    lfp_estimate: dict[str, float],
    caes_learning_factor: float,
    lfp_learning_factor: float,
    lfp_life_years: int,
) -> tuple[float, float, float, float]:
    caes_capital_2025 = (
        caes_estimate["power_capex_eur_kw"]
        + A_CAES_DURATION_H * caes_estimate["energy_capex_eur_kwh"]
    ) * crf(rate, A_CAES_LIFE_YR)
    caes_2025 = caes_capital_2025 + caes_estimate["fixed_om_eur_kw_yr"]
    caes_2035 = (
        caes_capital_2025 * caes_learning_factor
        + caes_estimate["fixed_om_eur_kw_yr"]
    )

    lfp_capital_2025 = (
        LFP_DURATION_H
        * lfp_estimate["energy_capex_eur_kwh"]
        * crf(rate, lfp_life_years)
    )
    lfp_2025 = lfp_capital_2025 + lfp_estimate["fixed_om_eur_kw_yr"]
    lfp_2035 = (
        lfp_capital_2025 * lfp_learning_factor
        + lfp_estimate["fixed_om_eur_kw_yr"]
    )
    return caes_2025, caes_2035, lfp_2025, lfp_2035


def build_ensemble(
    caes_estimates: dict[str, dict[str, float]],
    lfp_estimates: dict[str, dict[str, float]],
    caes_learning: dict[str, float],
    lfp_learning: dict[str, float],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    analysis_sets = {
        "cost_parameters_fixed_discount": (CENTRAL_DISCOUNT_RATE,),
        "cost_parameters_and_discount_rate": DISCOUNT_RATES,
    }
    for analysis_set, rates in analysis_sets.items():
        combinations = product(
            caes_estimates,
            lfp_estimates,
            caes_learning,
            lfp_learning,
            LFP_LIFE_YR,
            rates,
        )
        for index, (caes_case, lfp_case, caes_path, lfp_path, lfp_life, rate) in enumerate(
            combinations, start=1
        ):
            caes_2025, caes_2035, lfp_2025, lfp_2035 = annual_costs(
                rate,
                caes_estimates[caes_case],
                lfp_estimates[lfp_case],
                caes_learning[caes_path],
                lfp_learning[lfp_path],
                lfp_life,
            )
            signed = (caes_2025 - lfp_2025) - (caes_2035 - lfp_2035)
            records.append(
                {
                    "analysis_set": analysis_set,
                    "scenario_id": index,
                    "caes_2025_estimate": caes_case,
                    "lfp_2025_estimate": lfp_case,
                    "caes_learning_path": caes_path,
                    "lfp_learning_path": lfp_path,
                    "lfp_life_years": lfp_life,
                    "discount_rate": rate,
                    "caes_annual_fixed_cost_2025_eur_kw_year": caes_2025,
                    "caes_annual_fixed_cost_2035_eur_kw_year": caes_2035,
                    "lfp_annual_fixed_cost_2025_eur_kw_year": lfp_2025,
                    "lfp_annual_fixed_cost_2035_eur_kw_year": lfp_2035,
                    "signed_relative_cost_improvement_eur_kw_year": signed,
                    "absolute_relative_cost_movement_eur_kw_year": abs(signed),
                    "relative_cost_gap_narrows": signed > 0.0,
                }
            )
    return pd.DataFrame(records)


def natural_metrics() -> tuple[float, np.ndarray]:
    pairs = pd.read_csv(NATURAL_PAIRS)
    pair_values = pairs["absolute_weather_value_change_eur_kw_year"].to_numpy(float)
    first_year = int(min(pairs["start_year"].min(), pairs["end_year"].min()))
    anchored = pairs[pairs["start_year"].eq(first_year)].sort_values("end_year")
    annual_values = np.r_[0.0, anchored["weather_value_change_eur_kw_year"].to_numpy(float)]
    if len(annual_values) != 75:
        raise AssertionError(f"Expected 75 reconstructed natural years, found {len(annual_values)}")
    span = float(np.quantile(annual_values, 0.95) - np.quantile(annual_values, 0.05))
    return span, pair_values


def q(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def build_summary(ensemble: pd.DataFrame, natural_span: float, pair_values: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    records: list[dict[str, object]] = []
    for analysis_set, frame in ensemble.groupby("analysis_set", sort=False):
        values = frame["absolute_relative_cost_movement_eur_kw_year"].to_numpy(float)
        signed = frame["signed_relative_cost_improvement_eur_kw_year"].to_numpy(float)
        records.append(
            {
                "analysis_set": analysis_set,
                "scenario_cells": len(frame),
                "q05": q(values, 0.05),
                "q25": q(values, 0.25),
                "median": q(values, 0.50),
                "q75": q(values, 0.75),
                "q95": q(values, 0.95),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "cells_no_greater_than_natural_span": int(np.sum(values <= natural_span + 1e-12)),
                "share_no_greater_than_natural_span": float(np.mean(values <= natural_span + 1e-12)),
                "cells_with_narrowing_relative_gap": int(np.sum(signed > 0.0)),
                "share_with_narrowing_relative_gap": float(np.mean(signed > 0.0)),
            }
        )
    summary = pd.DataFrame(records)
    fixed = summary.loc[
        summary["analysis_set"].eq("cost_parameters_fixed_discount")
    ].iloc[0]
    natural_q05 = q(pair_values, 0.05)
    natural_q25 = q(pair_values, 0.25)
    natural_median = q(pair_values, 0.50)
    natural_q75 = q(pair_values, 0.75)
    natural_q95 = q(pair_values, 0.95)
    overlap_low = max(float(fixed["q05"]), natural_q05)
    overlap_high = min(float(fixed["q95"]), natural_q95)
    overlap_width = max(0.0, overlap_high - overlap_low)
    metrics: dict[str, object] = {
        "natural_operating_years": 75,
        "natural_year_pairs": int(len(pair_values)),
        "natural_q95_q05_annual_value_span_eur_kw_year": natural_span,
        "natural_pairwise_absolute_change": {
            "q05": natural_q05,
            "q25": natural_q25,
            "median": natural_median,
            "q75": natural_q75,
            "q95": natural_q95,
        },
        "fixed_discount_cost_interval_overlap": {
            "overlap_low": overlap_low,
            "overlap_high": overlap_high,
            "overlap_width": overlap_width,
            "fraction_of_natural_pairwise_central_90_interval": overlap_width
            / (natural_q95 - natural_q05),
            "fraction_of_cost_central_90_interval": overlap_width
            / (float(fixed["q95"]) - float(fixed["q05"])),
        },
    }
    return summary, metrics


def validate_central(ensemble: pd.DataFrame, catalogue: pd.DataFrame, natural_span: float) -> dict[str, float]:
    central = ensemble[
        ensemble["analysis_set"].eq("cost_parameters_fixed_discount")
        & ensemble["caes_2025_estimate"].eq("Point")
        & ensemble["lfp_2025_estimate"].eq("Point")
        & ensemble["caes_learning_path"].eq("reference")
        & ensemble["lfp_learning_path"].eq("reference")
        & ensemble["lfp_life_years"].eq(15)
    ]
    if len(central) != 1:
        raise AssertionError(f"Expected one central cell, found {len(central)}")
    row = central.iloc[0]
    expected = {
        "caes_2025": float(
            catalogue.loc[("caes_24h", 2025), "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"]
        ),
        "caes_2035": float(
            catalogue.loc[("caes_24h", 2035), "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"]
        ),
        "lfp_2025": float(
            catalogue.loc[("li_lfp_4h", 2025), "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"]
        ),
        "lfp_2035": float(
            catalogue.loc[("li_lfp_4h", 2035), "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"]
        ),
    }
    observed = {
        "caes_2025": float(row["caes_annual_fixed_cost_2025_eur_kw_year"]),
        "caes_2035": float(row["caes_annual_fixed_cost_2035_eur_kw_year"]),
        "lfp_2025": float(row["lfp_annual_fixed_cost_2025_eur_kw_year"]),
        "lfp_2035": float(row["lfp_annual_fixed_cost_2035_eur_kw_year"]),
    }
    errors = {key: observed[key] - expected[key] for key in expected}
    if max(abs(value) for value in errors.values()) > 2e-6:
        raise AssertionError({"central_replication_errors": errors})
    if not 5.42 < natural_span < 5.44:
        raise AssertionError(f"Unexpected natural span {natural_span}")
    return {
        "central_comparator_eur_kw_year": float(
            row["absolute_relative_cost_movement_eur_kw_year"]
        ),
        "maximum_abs_cost_replication_error": max(abs(value) for value in errors.values()),
    }


def write_macros(summary: pd.DataFrame, metrics: dict[str, object], validation: dict[str, float]) -> None:
    fixed = summary.loc[
        summary["analysis_set"].eq("cost_parameters_fixed_discount")
    ].iloc[0]
    joint = summary.loc[
        summary["analysis_set"].eq("cost_parameters_and_discount_rate")
    ].iloc[0]
    pair = metrics["natural_pairwise_absolute_change"]
    overlap = metrics["fixed_discount_cost_interval_overlap"]
    lines = [
        rf"\newcommand{{\CostCentralComparator}}{{{validation['central_comparator_eur_kw_year']:.3f}}}",
        rf"\newcommand{{\CostFixedCells}}{{{int(fixed['scenario_cells'])}}}",
        rf"\newcommand{{\CostFixedQFive}}{{{fixed['q05']:.2f}}}",
        rf"\newcommand{{\CostFixedQTwentyFive}}{{{fixed['q25']:.2f}}}",
        rf"\newcommand{{\CostFixedMedian}}{{{fixed['median']:.2f}}}",
        rf"\newcommand{{\CostFixedQSeventyFive}}{{{fixed['q75']:.2f}}}",
        rf"\newcommand{{\CostFixedQNinetyFive}}{{{fixed['q95']:.2f}}}",
        rf"\newcommand{{\CostFixedBelowSpanCount}}{{{int(fixed['cells_no_greater_than_natural_span'])}}}",
        rf"\newcommand{{\CostFixedBelowSpanPercent}}{{{100 * fixed['share_no_greater_than_natural_span']:.1f}}}",
        rf"\newcommand{{\CostFixedNarrowCount}}{{{int(fixed['cells_with_narrowing_relative_gap'])}}}",
        rf"\newcommand{{\CostFixedNarrowPercent}}{{{100 * fixed['share_with_narrowing_relative_gap']:.1f}}}",
        rf"\newcommand{{\CostJointCells}}{{{int(joint['scenario_cells'])}}}",
        rf"\newcommand{{\CostJointQFive}}{{{joint['q05']:.2f}}}",
        rf"\newcommand{{\CostJointMedian}}{{{joint['median']:.2f}}}",
        rf"\newcommand{{\CostJointQNinetyFive}}{{{joint['q95']:.2f}}}",
        rf"\newcommand{{\CostJointBelowSpanCount}}{{{int(joint['cells_no_greater_than_natural_span'])}}}",
        rf"\newcommand{{\CostJointBelowSpanPercent}}{{{100 * joint['share_no_greater_than_natural_span']:.1f}}}",
        rf"\newcommand{{\NaturalPairQFive}}{{{pair['q05']:.2f}}}",
        rf"\newcommand{{\NaturalPairMedian}}{{{pair['median']:.2f}}}",
        rf"\newcommand{{\NaturalPairQNinetyFive}}{{{pair['q95']:.2f}}}",
        rf"\newcommand{{\CostOverlapLow}}{{{overlap['overlap_low']:.2f}}}",
        rf"\newcommand{{\CostOverlapHigh}}{{{overlap['overlap_high']:.2f}}}",
        rf"\newcommand{{\CostOverlapNaturalPercent}}{{{100 * overlap['fraction_of_natural_pairwise_central_90_interval']:.1f}}}",
    ]
    (RESULTS / "cost_uncertainty_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary: pd.DataFrame, metrics: dict[str, object], validation: dict[str, float]) -> None:
    fixed = summary.loc[
        summary["analysis_set"].eq("cost_parameters_fixed_discount")
    ].iloc[0]
    joint = summary.loc[
        summary["analysis_set"].eq("cost_parameters_and_discount_rate")
    ].iloc[0]
    overlap = metrics["fixed_discount_cost_interval_overlap"]
    report = f"""# V62 cost-parameter uncertainty experiment

The central 24 h A-CAES versus 4 h LFP comparator is reproduced at
{validation['central_comparator_eur_kw_year']:.3f} EUR kW^-1 yr^-1.

At a fixed 3.6% real discount rate, the balanced source-bounded ensemble has
{int(fixed['scenario_cells'])} cells. The absolute 2025--2035 relative-cost
movement has median {fixed['median']:.2f}, interquartile range
{fixed['q25']:.2f}--{fixed['q75']:.2f}, and central 90% interval
{fixed['q05']:.2f}--{fixed['q95']:.2f} EUR kW^-1 yr^-1. The natural-year
5th--95th value span lies above {int(fixed['cells_no_greater_than_natural_span'])}
of {int(fixed['scenario_cells'])} cost cells
({100 * fixed['share_no_greater_than_natural_span']:.1f}%).

The central 90% interval of absolute natural-year pair changes overlaps the
fixed-rate cost interval from {overlap['overlap_low']:.2f} to
{overlap['overlap_high']:.2f} EUR kW^-1 yr^-1, covering
{100 * overlap['fraction_of_natural_pairwise_central_90_interval']:.1f}% of the
natural-pair interval.

Adding the four prespecified discount rates yields {int(joint['scenario_cells'])}
cells, a median of {joint['median']:.2f}, and a central 90% interval of
{joint['q05']:.2f}--{joint['q95']:.2f} EUR kW^-1 yr^-1. The natural span is no
smaller than {int(joint['cells_no_greater_than_natural_span'])} cells
({100 * joint['share_no_greater_than_natural_span']:.1f}%).
"""
    (HERE / "RESULTS.md").write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    catalogue = catalogue_rows()
    inputs, caes_estimates, lfp_estimates, caes_learning, lfp_learning = build_inputs(
        catalogue
    )
    ensemble = build_ensemble(
        caes_estimates, lfp_estimates, caes_learning, lfp_learning
    )
    natural_span, pair_values = natural_metrics()
    summary, metrics = build_summary(ensemble, natural_span, pair_values)
    validation = validate_central(ensemble, catalogue, natural_span)

    inputs.to_csv(RESULTS / "cost_parameter_inputs.csv", index=False)
    ensemble.to_csv(RESULTS / "cost_comparator_scenario_ensemble.csv", index=False)
    summary.to_csv(RESULTS / "cost_uncertainty_summary.csv", index=False)
    qa = {
        "version": "v62_source_bounded_cost_uncertainty_1.0",
        "interpretation": "balanced source-bounded scenario ensemble; cell shares are not occurrence probabilities",
        "inputs": {
            str(COST_CATALOGUE): digest(COST_CATALOGUE),
            str(ESGC_DATABASE): digest(ESGC_DATABASE),
            str(LEARNING_CURVES): digest(LEARNING_CURVES),
            str(NATURAL_PAIRS): digest(NATURAL_PAIRS),
        },
        "discount_rates": DISCOUNT_RATES,
        "caes_life_years": A_CAES_LIFE_YR,
        "lfp_life_years": LFP_LIFE_YR,
        "validation": validation,
        "metrics": metrics,
        "all_values_finite": bool(
            np.isfinite(
                ensemble.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
        ),
        "scenario_keys_unique": bool(
            ~ensemble.duplicated(["analysis_set", "scenario_id"]).any()
        ),
    }
    (RESULTS / "cost_uncertainty_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_macros(summary, metrics, validation)
    write_report(summary, metrics, validation)
    print(summary.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
