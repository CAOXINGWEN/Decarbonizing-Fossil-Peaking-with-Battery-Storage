#!/usr/bin/env python3
"""Compare strict-42 and exploratory-63 complete duration screens.

The script writes technology, duration, country-year and censoring summaries,
then creates the decision memo requested for the Europe-mainline upgrade.
Counts over configuration-country-year pairs and country-year directions are
explicitly labelled descriptive. The eventual confirmatory resampling unit is
the synchronized European climate year because weather is spatially
correlated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
UPGRADE = HERE.parent
MODEL_ROOT = UPGRADE / "model_adapter"
ARTIFACTS = UPGRADE / "artifacts" / "research"
DEFAULT_STRICT = MODEL_ROOT / "strict42_full53_aggregate"
DEFAULT_EXPLORATORY = MODEL_ROOT / "exploratory63_full53_aggregate"
DEFAULT_OUTPUT = MODEL_ROOT / "strict42_vs_exploratory63_full53"

METRICS = {
    "EFC": "delta_efc_clustered_minus_dispersed",
    "turnover_benefit_cny_per_kw_year": (
        "delta_turnover_benefit_clustered_minus_dispersed"
    ),
    "joint_benefit_cny_per_kw_year": (
        "delta_joint_benefit_clustered_minus_dispersed"
    ),
}
EU27 = {
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
}


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_screen(path: Path, sample: str) -> dict[str, pd.DataFrame]:
    physical = pd.read_csv(path / "unit_configuration_paired_changes.csv")
    configuration = pd.read_csv(path / "all_configuration_screen_metrics.csv")
    metadata = configuration[
        [
            "config_id",
            "tech_id",
            "technology_en",
            "duration_h",
            "released_library_configuration",
            "structural_extrapolation",
        ]
    ].drop_duplicates()
    physical = physical.drop(columns=["duration_h"], errors="ignore").merge(
        metadata, on="config_id", how="left", validate="many_to_one"
    )
    physical.insert(0, "sample", sample)
    units = pd.read_csv(path / "country_year_direction_summary.csv")
    units.insert(0, "sample", sample)
    boundaries = pd.read_csv(path / "all_boundary_diagnostics.csv")
    boundaries.insert(0, "sample", sample)
    envelopes = pd.read_csv(path / "all_duration_envelopes.csv")
    envelopes.insert(0, "sample", sample)
    qa = json.loads(
        (path / "AGGREGATE_QA_AND_SUMMARY.json").read_text(encoding="utf-8")
    )
    return {
        "physical": physical,
        "units": units,
        "boundaries": boundaries,
        "envelopes": envelopes,
        "qa": qa,
    }


def metric_summary(group: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "country_year_count": int(group["unit_id"].nunique()),
        "configuration_count": int(group["config_id"].nunique()),
        "configuration_country_year_pairs": int(len(group)),
    }
    for label, column in METRICS.items():
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        row.update(
            {
                f"{label}_negative_n": int((values < 0).sum()),
                f"{label}_negative_fraction": float((values < 0).mean()),
                f"{label}_mean": float(values.mean()),
                f"{label}_median": float(values.median()),
                f"{label}_q25": float(values.quantile(0.25)),
                f"{label}_q75": float(values.quantile(0.75)),
            }
        )
    return row


def grouped_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row.update(metric_summary(group))
        rows.append(row)
    return pd.DataFrame(rows)


def boundary_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    full = frame[frame["full_configuration_set"].map(as_bool)].copy()
    status_rows = []
    for (sample, chronology, field), group in pd.concat(
        [
            full.assign(status_field="h_T", status=full["h_T_censoring_status"]),
            full.assign(status_field="h_I", status=full["h_I_censoring_status"]),
        ]
    ).groupby(["sample", "chronology", "status_field"], sort=True):
        for status, count in group["status"].value_counts().items():
            status_rows.append(
                {
                    "sample": sample,
                    "chronology": chronology,
                    "boundary": field,
                    "status": status,
                    "count": int(count),
                }
            )
    summary = {}
    for sample, group in full.groupby("sample", sort=True):
        summary[sample] = {
            "country_year_count": int(group["unit_id"].nunique()),
            "chronology_unit_screens": int(len(group)),
            "point_complete_valley_n": int(
                group["point_screen_complete_valley"].map(as_bool).sum()
            ),
            "confirmatory_complete_valley_n": 0,
            "negative_middle_interval_n": int(
                group["negative_interval_start_h"].notna().sum()
            ),
            "h_I_right_censored_n": int(
                group["h_I_censoring_status"]
                .astype(str)
                .str.startswith("right_censored")
                .sum()
            ),
            "h_I_no_negative_middle_n": int(
                (
                    group["h_I_censoring_status"]
                    == "not_identified_no_two_node_negative_middle_interval"
                ).sum()
            ),
        }
    return pd.DataFrame(status_rows), summary


def unit_level_summary(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample, group in units.groupby("sample", sort=True):
        row: dict[str, Any] = {
            "sample": sample,
            "country_year_count": int(group["unit_id"].nunique()),
        }
        for label in ("EFC", "turnover_benefit", "joint_benefit"):
            means = group[f"{label}_mean"]
            row[f"{label}_country_year_mean_negative_n"] = int((means < 0).sum())
            row[f"{label}_country_year_mean_negative_fraction"] = float(
                (means < 0).mean()
            )
            row[f"{label}_median_of_country_year_means"] = float(means.median())
        rows.append(row)
    return pd.DataFrame(rows)


def unit_level_summary_from_physical(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sample, unit_id), group in frame.groupby(
        ["sample", "unit_id"], sort=True
    ):
        row: dict[str, Any] = {"sample": sample, "unit_id": unit_id}
        for label, column in METRICS.items():
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{label}_mean"] = float(values.mean())
            row[f"{label}_median"] = float(values.median())
            row[f"{label}_negative_fraction"] = float((values < 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def adjacent_negative_run(values: np.ndarray, minimum: int = 2) -> bool:
    run = 0
    for value in values:
        run = run + 1 if value < 0 else 0
        if run >= minimum:
            return True
    return False


def evidence_domain_boundaries(
    envelopes: pd.DataFrame, maximum_duration_h: float = 168.0
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evidence = envelopes[envelopes["duration_h"] <= maximum_duration_h].copy()
    rows = []
    for (sample, unit_id, chronology), group in evidence.groupby(
        ["sample", "unit_id", "chronology"], sort=True
    ):
        ordered = group.sort_values("duration_h")
        max_node = ordered[
            np.isclose(ordered["duration_h"], maximum_duration_h)
        ]
        if len(max_node) != 1:
            raise RuntimeError(
                f"Missing unique {maximum_duration_h:g} h envelope node for "
                f"{sample}-{unit_id}-{chronology}"
            )
        turnover_at_max = float(
            max_node[
                "turnover_signed_benefit_cny_per_forced_kw_year"
            ].iloc[0]
        )
        joint = ordered[
            "joint_signed_benefit_cny_per_forced_kw_year"
        ].to_numpy(float)
        rows.append(
            {
                "sample": sample,
                "unit_id": unit_id,
                "chronology": chronology,
                "evidence_maximum_duration_h": maximum_duration_h,
                "turnover_benefit_at_168h_cny_per_kw_year": turnover_at_max,
                "h_T_evidence_status": (
                    "right_censored_above_168h"
                    if turnover_at_max > 0
                    else "boundary_within_evidence_domain"
                ),
                "negative_joint_node_count_4_168h": int((joint < 0).sum()),
                "two_adjacent_negative_joint_nodes_4_168h": (
                    adjacent_negative_run(joint)
                ),
                "h_I_evidence_status": (
                    "not_identified_no_negative_middle_interval"
                    if not adjacent_negative_run(joint)
                    else "requires_post_gap_reentry_test"
                ),
                "W_evidence_status": "not_identified",
            }
        )
    detail = pd.DataFrame(rows)
    summary = {}
    for sample, group in detail.groupby("sample", sort=True):
        summary[sample] = {
            "country_year_count": int(group["unit_id"].nunique()),
            "chronology_unit_screens": int(len(group)),
            "h_T_right_censored_above_168h_n": int(
                (
                    group["h_T_evidence_status"]
                    == "right_censored_above_168h"
                ).sum()
            ),
            "negative_joint_nodes_4_168h": int(
                group["negative_joint_node_count_4_168h"].sum()
            ),
            "screens_with_two_adjacent_negative_nodes_4_168h": int(
                group["two_adjacent_negative_joint_nodes_4_168h"]
                .map(as_bool)
                .sum()
            ),
            "h_I_identified_n": int(
                (
                    group["h_I_evidence_status"]
                    != "not_identified_no_negative_middle_interval"
                ).sum()
            ),
            "W_identified_n": 0,
        }
    return detail, summary


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_memo(
    strict: dict[str, Any],
    exploratory: dict[str, Any],
    technology: pd.DataFrame,
    duration: pd.DataFrame,
    unit_summary: pd.DataFrame,
    boundaries: dict[str, Any],
    evidence_boundaries: dict[str, Any],
    output: Path,
) -> None:
    strict_qa = strict["qa"]
    exploratory_qa = exploratory["qa"]
    strict_country_count = int(
        strict["units"]["unit_id"].str.rsplit("_", n=1).str[0].nunique()
    )
    exploratory_country_count = int(
        exploratory["units"]["unit_id"].str.rsplit("_", n=1).str[0].nunique()
    )
    strict_evidence = strict["physical"][
        ~strict["physical"]["structural_extrapolation"].map(as_bool)
    ].copy()
    exploratory_evidence = exploratory["physical"][
        ~exploratory["physical"]["structural_extrapolation"].map(as_bool)
    ].copy()
    strict_pool = metric_summary(strict_evidence)
    exploratory_pool = metric_summary(exploratory_evidence)
    strict_unit_effects = unit_level_summary_from_physical(strict_evidence)
    exploratory_unit_effects = unit_level_summary_from_physical(
        exploratory_evidence
    )
    strict_unit_negative = {
        label: int((strict_unit_effects[f"{label}_mean"] < 0).sum())
        for label in METRICS
    }
    exploratory_unit_negative = {
        label: int((exploratory_unit_effects[f"{label}_mean"] < 0).sum())
        for label in METRICS
    }
    strict_boundary = boundaries["strict42"]
    exploratory_boundary = boundaries["exploratory63"]
    strict_eu = strict_evidence.copy()
    strict_eu["country"] = strict_eu["unit_id"].str.rsplit("_", n=1).str[0]
    strict_eu = strict_eu[strict_eu["country"].isin(EU27)]
    exploratory_eu = exploratory_evidence.copy()
    exploratory_eu["country"] = (
        exploratory_eu["unit_id"].str.rsplit("_", n=1).str[0]
    )
    exploratory_eu = exploratory_eu[exploratory_eu["country"].isin(EU27)]
    strict_eu_stats = metric_summary(strict_eu)
    exploratory_eu_stats = metric_summary(exploratory_eu)

    strict_tech = technology[technology["sample"] == "strict42"].copy()
    strict_tech = strict_tech.sort_values(
        "turnover_benefit_cny_per_kw_year_negative_fraction", ascending=False
    )
    tech_lines = []
    for row in strict_tech.itertuples(index=False):
        tech_lines.append(
            f"| {row.tech_id} | "
            f"{format_pct(row.EFC_negative_fraction)} | "
            f"{format_pct(row.turnover_benefit_cny_per_kw_year_negative_fraction)} | "
            f"{format_pct(row.joint_benefit_cny_per_kw_year_negative_fraction)} |"
        )

    lines = [
        "# EUROPE SCREEN DECISION MEMO",
        "",
        "## Decision",
        "",
        "The European upgrade materially strengthens the empirical support for "
        "the repeat-use mechanism but does not confirm a complete "
        "mid-duration valley under the tested cost and system-value structure. "
        "The manuscript can use Europe as the main multi-country, multi-weather-"
        "year evidence that temporal clustering compresses storage cycling and "
        "joint marginal value. It should retain the mid-duration entry gap as "
        "a conditional theoretical implication rather than an observed "
        "positive-negative-positive duration-value pattern.",
        "",
        "## Evidence base and estimand",
        "",
        f"- The strict screen contains {strict_qa['country_year_count']} "
        f"country-weather-years from {strict_country_count} countries; the "
        f"exploratory sensitivity contains "
        f"{exploratory_qa['country_year_count']} country-weather-years from "
        f"{exploratory_country_count} countries.",
        "- The current-EU27-only sensitivity excludes Switzerland and Great "
        f"Britain. It contains {strict_eu_stats['country_year_count']} strict "
        f"and {exploratory_eu_stats['country_year_count']} exploratory "
        "country-weather-years.",
        "- Every natural, dispersed and clustered case uses exactly the same "
        "365 paired 24-hour load-wind-solar blocks and annual energy. Only day "
        "order changes within seasons.",
        "- Wind and solar retain observed hourly shapes but are scaled so annual "
        "VRE energy equals annual load. This is a controlled high-VRE "
        "counterfactual, not an observed future European dispatch year.",
        "- The evidence-supported spectrum contains 50 configurations from "
        "eight technology families at 24 unique durations from 4 to 168 h. "
        "Cavern-hydrogen nodes at 240, 336 and 720 h are retained only as "
        "explicit structural extrapolations. The signed 1%-of-mean-load tranche values "
        "energy shifting, gas-capacity substitution, charging opportunity, "
        "spill, EENS and technology cost in one hourly LP.",
        "",
        "## What is robust",
        "",
        f"In the strict 4–168 h evidence spectrum, clustered ordering lowers "
        f"EFC in {strict_pool['EFC_negative_n']}/"
        f"{strict_pool['configuration_country_year_pairs']} "
        f"configuration-country-year comparisons "
        f"({format_pct(strict_pool['EFC_negative_fraction'])}), turnover "
        f"benefit in "
        f"{strict_pool['turnover_benefit_cny_per_kw_year_negative_n']}/"
        f"{strict_pool['configuration_country_year_pairs']} "
        f"({format_pct(strict_pool['turnover_benefit_cny_per_kw_year_negative_fraction'])}), "
        f"and joint benefit in "
        f"{strict_pool['joint_benefit_cny_per_kw_year_negative_n']}/"
        f"{strict_pool['configuration_country_year_pairs']} "
        f"({format_pct(strict_pool['joint_benefit_cny_per_kw_year_negative_fraction'])}). "
        f"The respective medians are {strict_pool['EFC_median']:.2f} EFC, "
        f"{strict_pool['turnover_benefit_cny_per_kw_year_median']:.2f} and "
        f"{strict_pool['joint_benefit_cny_per_kw_year_median']:.2f} "
        "CNY kW−1 yr−1.",
        "",
        f"Using the country-weather-year as the reporting unit, the mean joint "
        f"effect is "
        f"negative in "
        f"{strict_unit_negative['joint_benefit_cny_per_kw_year']}/"
        f"{len(strict_unit_effects)} strict country-years, the mean "
        f"turnover effect in "
        f"{strict_unit_negative['turnover_benefit_cny_per_kw_year']}/"
        f"{len(strict_unit_effects)}, and the mean EFC effect in "
        f"{strict_unit_negative['EFC']}/"
        f"{len(strict_unit_effects)}. Configuration-level and "
        "country-year counts are descriptive because technology nodes repeat "
        "within a system and European weather is spatially correlated. Formal "
        "inference must resample synchronized European climate years.",
        "",
        "| Technology | EFC negative | Turnover value negative | Joint value negative |",
        "|---|---:|---:|---:|",
        *tech_lines,
        "",
        "The exploratory sensitivity is directionally consistent. Its pooled "
        f"negative shares are "
        f"{format_pct(exploratory_pool['EFC_negative_fraction'])} for EFC, "
        f"{format_pct(exploratory_pool['turnover_benefit_cny_per_kw_year_negative_fraction'])} "
        "for turnover benefit and "
        f"{format_pct(exploratory_pool['joint_benefit_cny_per_kw_year_negative_fraction'])} "
        "for joint benefit. Its country-year mean joint effect is negative in "
        f"{exploratory_unit_negative['joint_benefit_cny_per_kw_year']}/"
        f"{len(exploratory_unit_effects)} units.",
        "",
        "Restricting the strict sample to current EU27 members also preserves "
        f"the result. Negative shares are "
        f"{format_pct(strict_eu_stats['EFC_negative_fraction'])} for EFC, "
        f"{format_pct(strict_eu_stats['turnover_benefit_cny_per_kw_year_negative_fraction'])} "
        "for turnover benefit and "
        f"{format_pct(strict_eu_stats['joint_benefit_cny_per_kw_year_negative_fraction'])} "
        "for joint benefit.",
        "",
        "## Why the complete valley is not confirmed",
        "",
        "Within the evidence-supported 4–168 h spectrum, turnover benefit "
        f"remains positive at 168 h in all "
        f"{evidence_boundaries['strict42']['chronology_unit_screens']} strict "
        "chronology-unit screens. The short-side boundary h_T is therefore "
        "right-censored above 168 h, not estimated at 168–240 h. There are "
        f"{evidence_boundaries['strict42']['negative_joint_nodes_4_168h']} "
        "negative joint-benefit envelope nodes in this evidence domain, so "
        "h_I, W and ΔW are not identified.",
        "",
        "The three structural cavern-hydrogen extensions do not rescue the "
        f"hypothesis. The 4–720 h sensitivity contains "
        f"{strict_boundary['point_complete_valley_n']}/"
        f"{strict_boundary['chronology_unit_screens']} point-estimate complete "
        "valleys; the exploratory screen contains "
        f"{exploratory_boundary['point_complete_valley_n']}/"
        f"{exploratory_boundary['chronology_unit_screens']}.",
        "",
        "The failure is structural rather than a missing statistical star. "
        "Across most country-years, the joint-benefit envelope remains positive "
        "through the short and intermediate durations and turns negative only "
        "at the far-right tail. Thus there are not two adjacent negative middle "
        "nodes followed by a positive long-duration branch. In the strict "
        f"tail, {strict_boundary['h_I_no_negative_middle_n']} screens have no "
        "qualifying negative middle interval. The remaining "
        f"{strict_boundary['h_I_right_censored_n']} reach a negative interval "
        "but do not re-enter by 720 h, leaving h_I right-censored. A monotonic "
        "tail decline is not a mid-duration valley, and structural extension "
        "nodes cannot define the primary evidence boundary.",
        "",
        "The current LP also yields a relatively flat firm-capacity component "
        "once a forced tranche can displace gas capacity. That component keeps "
        "many middle and long nodes positive but does not produce a distinct "
        "long-duration reliability re-entry after a negative gap. The screen "
        "therefore supports contraction of the repeat-use domain, not a "
        "separate long-inventory branch.",
        "",
        "## ERAA 2025 networked re-test",
        "",
        "A valid re-test should not tune parameters to manufacture a valley. It "
        "should replace the point copper-plate screen with the following "
        "pre-specified European adequacy experiment.",
        "",
        "1. Use synchronized PECD climate-year demand, onshore wind, offshore "
        "wind, solar and hydro inputs over their common multi-decadal period. "
        "The locally audited PECD files provide hourly country demand for "
        "1982–2016 and hourly wind/solar capacity factors extending through "
        "2019; the conservative common window is therefore 1982–2016.",
        "2. Represent ERAA bidding zones and cross-border constraints using NTC "
        "or flow-based domains. Move a pan-European daily weather vector, not "
        "independent national days, so spatial weather correlation is exactly "
        "preserved.",
        "3. Include generator capacities, fuel and carbon costs, planned and "
        "forced outages, hydro energy budgets, demand response, existing "
        "storage and interconnector availability. Gas generation becomes an "
        "endogenous response rather than a demand proxy.",
        "4. Co-optimize storage charging power, discharge power and energy "
        "capacity. Charge electricity at the model's hourly marginal "
        "opportunity cost and retain technology-specific duration domains and "
        "cost evidence.",
        "5. Compute LOLE, EENS and ELCC for every duration-envelope winner. "
        "Identify h_T only from a positive turnover branch and h_I only from a "
        "post-gap node with positive joint benefit and positive ELCC support.",
        "6. Re-select the technology envelope within every weather-year "
        "bootstrap draw. Confirm a valley only with two adjacent negative "
        "middle nodes, uncensored h_T and h_I, and a positive lower confidence "
        "bound for clustered-minus-dispersed ΔW.",
        "",
        "If that networked adequacy model still lacks long-duration re-entry, "
        "the full mid-duration-valley hypothesis should be rejected for the "
        "tested European system. The durable theoretical contribution would "
        "then be the empirically supported result that scarcity clustering "
        "endogenously contracts storage repeat-use value.",
        "",
        "## Files",
        "",
        "- Strict summary: `model_adapter/strict42_full53_aggregate/`.",
        "- Exploratory summary: `model_adapter/exploratory63_full53_aggregate/`.",
        "- Cross-sample tables: `model_adapter/strict42_vs_exploratory63_full53/`.",
        "- Primary evidence-domain tables use the `evidence_4_168h_` prefix; "
        "unprefixed 53-node tables include structural tail sensitivities.",
        "- Adapter: `scripts/run_europe_duration_spectrum_pilot.py`.",
    ]
    text = "\n".join(lines) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "EUROPE_SCREEN_DECISION_MEMO.md").write_text(
        text, encoding="utf-8"
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "EUROPE_SCREEN_DECISION_MEMO.md").write_text(
        text, encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", default=str(DEFAULT_STRICT))
    parser.add_argument("--exploratory", default=str(DEFAULT_EXPLORATORY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strict = load_screen(Path(args.strict), "strict42")
    exploratory = load_screen(Path(args.exploratory), "exploratory63")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    physical = pd.concat(
        [strict["physical"], exploratory["physical"]], ignore_index=True
    )
    evidence_physical = physical[
        ~physical["structural_extrapolation"].map(as_bool)
    ].copy()
    units = pd.concat([strict["units"], exploratory["units"]], ignore_index=True)
    boundaries = pd.concat(
        [strict["boundaries"], exploratory["boundaries"]], ignore_index=True
    )
    envelopes = pd.concat(
        [strict["envelopes"], exploratory["envelopes"]], ignore_index=True
    )
    technology = grouped_summary(physical, ["sample", "tech_id"])
    duration = grouped_summary(physical, ["sample", "duration_h"])
    configuration = grouped_summary(physical, ["sample", "config_id"])
    overall = grouped_summary(physical, ["sample"])
    evidence_technology = grouped_summary(
        evidence_physical, ["sample", "tech_id"]
    )
    evidence_duration = grouped_summary(
        evidence_physical, ["sample", "duration_h"]
    )
    evidence_configuration = grouped_summary(
        evidence_physical, ["sample", "config_id"]
    )
    evidence_overall = grouped_summary(evidence_physical, ["sample"])
    evidence_unit_effects = unit_level_summary_from_physical(evidence_physical)
    unit_summary = unit_level_summary(units)
    boundary_status, boundary_counts = boundary_summary(boundaries)
    evidence_boundary_detail, evidence_boundary_counts = (
        evidence_domain_boundaries(envelopes)
    )
    evidence_physical["country"] = (
        evidence_physical["unit_id"].str.rsplit("_", n=1).str[0]
    )
    eu_only = evidence_physical[
        evidence_physical["country"].isin(EU27)
    ].copy()
    eu_only_overall = grouped_summary(eu_only, ["sample"])
    eu_boundaries = boundaries.copy()
    eu_boundaries["country"] = (
        eu_boundaries["unit_id"].str.rsplit("_", n=1).str[0]
    )
    eu_boundary_status, eu_boundary_counts = boundary_summary(
        eu_boundaries[eu_boundaries["country"].isin(EU27)]
    )
    inventory = units[["sample", "unit_id"]].drop_duplicates().copy()
    split = inventory["unit_id"].str.rsplit("_", n=1, expand=True)
    inventory["country"] = split[0]
    inventory["year"] = pd.to_numeric(split[1], errors="raise").astype(int)
    inventory = inventory.sort_values(["sample", "country", "year"])
    country_coverage = (
        inventory.groupby(["sample", "country"], as_index=False)
        .agg(
            country_year_count=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            years=("year", lambda values: ";".join(map(str, sorted(values)))),
        )
        .sort_values(["sample", "country"])
    )
    technology.to_csv(output / "direction_by_technology.csv", index=False)
    duration.to_csv(output / "direction_by_duration.csv", index=False)
    configuration.to_csv(output / "direction_by_configuration.csv", index=False)
    overall.to_csv(output / "strict_vs_exploratory_overall.csv", index=False)
    evidence_technology.to_csv(
        output / "evidence_4_168h_direction_by_technology.csv", index=False
    )
    evidence_duration.to_csv(
        output / "evidence_4_168h_direction_by_duration.csv", index=False
    )
    evidence_configuration.to_csv(
        output / "evidence_4_168h_direction_by_configuration.csv", index=False
    )
    evidence_overall.to_csv(
        output / "evidence_4_168h_strict_vs_exploratory_overall.csv",
        index=False,
    )
    evidence_unit_effects.to_csv(
        output / "evidence_4_168h_country_year_effects.csv", index=False
    )
    eu_only_overall.to_csv(
        output / "current_eu27_only_overall.csv", index=False
    )
    unit_summary.to_csv(output / "country_year_level_comparison.csv", index=False)
    inventory.to_csv(output / "sample_country_year_inventory.csv", index=False)
    country_coverage.to_csv(
        output / "sample_country_coverage_counts.csv", index=False
    )
    boundary_status.to_csv(output / "boundary_censoring_counts.csv", index=False)
    eu_boundary_status.to_csv(
        output / "current_eu27_only_boundary_censoring_counts.csv",
        index=False,
    )
    (output / "boundary_summary.json").write_text(
        json.dumps(boundary_counts, indent=2), encoding="utf-8"
    )
    evidence_boundary_detail.to_csv(
        output / "evidence_4_168h_boundary_detail.csv", index=False
    )
    (output / "evidence_4_168h_boundary_summary.json").write_text(
        json.dumps(evidence_boundary_counts, indent=2), encoding="utf-8"
    )
    (output / "current_eu27_only_boundary_summary.json").write_text(
        json.dumps(eu_boundary_counts, indent=2), encoding="utf-8"
    )
    overlap_keys = ["unit_id", "config_id", "duration_h"]
    overlap = strict["physical"].merge(
        exploratory["physical"],
        on=overlap_keys,
        suffixes=("_strict", "_exploratory"),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    overlap_errors = {}
    for column in METRICS.values():
        overlap_errors[column] = float(
            (
                overlap[f"{column}_strict"]
                - overlap[f"{column}_exploratory"]
            )
            .abs()
            .max()
        )
    strict_units = set(strict["physical"]["unit_id"])
    exploratory_units = set(exploratory["physical"]["unit_id"])
    comparison_qa = {
        "strict_country_year_count": len(strict_units),
        "exploratory_country_year_count": len(exploratory_units),
        "strict_units_subset_of_exploratory": strict_units.issubset(
            exploratory_units
        ),
        "strict_overlap_rows": int(len(overlap)),
        "strict_overlap_all_matched": bool(
            (overlap["_merge"] == "both").all()
        ),
        "maximum_overlap_absolute_errors": overlap_errors,
        "strict_and_exploratory_directionally_consistent": bool(
            all(
                strict["qa"]["pooled_direction_distributions"][metric][
                    "median"
                ]
                < 0
                and exploratory["qa"]["pooled_direction_distributions"][
                    metric
                ]["median"]
                < 0
                for metric in ("EFC", "turnover_benefit", "joint_benefit")
            )
        ),
        "both_point_complete_valley_counts_zero": bool(
            boundary_counts["strict42"]["point_complete_valley_n"] == 0
            and boundary_counts["exploratory63"][
                "point_complete_valley_n"
            ]
            == 0
        ),
        "evidence_4_168h_all_hT_right_censored": bool(
            all(
                values["h_T_right_censored_above_168h_n"]
                == values["chronology_unit_screens"]
                for values in evidence_boundary_counts.values()
            )
        ),
        "evidence_4_168h_negative_joint_nodes_zero": bool(
            all(
                values["negative_joint_nodes_4_168h"] == 0
                for values in evidence_boundary_counts.values()
            )
        ),
    }
    comparison_qa["passed"] = bool(
        comparison_qa["strict_units_subset_of_exploratory"]
        and comparison_qa["strict_overlap_all_matched"]
        and max(overlap_errors.values()) <= 1e-9
        and comparison_qa[
            "strict_and_exploratory_directionally_consistent"
        ]
        and comparison_qa["both_point_complete_valley_counts_zero"]
        and comparison_qa["evidence_4_168h_all_hT_right_censored"]
        and comparison_qa["evidence_4_168h_negative_joint_nodes_zero"]
    )
    (output / "COMPARISON_QA.json").write_text(
        json.dumps(comparison_qa, indent=2), encoding="utf-8"
    )
    write_memo(
        strict,
        exploratory,
        evidence_technology,
        evidence_duration,
        unit_summary,
        boundary_counts,
        evidence_boundary_counts,
        output,
    )
    if not comparison_qa["passed"]:
        raise RuntimeError(f"Cross-sample comparison QA failed: {comparison_qa}")
    print(output)


if __name__ == "__main__":
    main()
