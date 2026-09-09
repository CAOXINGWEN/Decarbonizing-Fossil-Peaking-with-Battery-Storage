#!/usr/bin/env python3
"""Audit chronology-induced storage procurement flips for the v43 model.

The primary audit uses the exact fixed-scale revaluation at the central service
values and the technology-specific 2025, 2030 and 2035 fixed costs.  It treats
the additional 4 h LFP arm as a zero-benefit outside option and distinguishes
an economically selected configuration from the best (but possibly still
negative) LDES configuration.

The final section reads the already solved bounded-tail-power S31 robustness
case.  That case uses 2035 costs, vE=98.5403 EUR MWh-1, vF=103.7992 EUR
kW-1 yr-1 and zero charging price.  It is intentionally reported separately:
no complete bounded-power solve exists for the central S20 valuation, so the
S31 result must not be described as a three-year central-value estimate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V43 = HERE.parents[1]
CENTRAL_CELLS = (
    V43 / "cost_and_decomposition" / "year_specific_revaluation_cells.csv.gz"
)
BOUNDED_CELLS = (
    V43
    / "source_data"
    / "v42_all_cases_mix_efc_discrete_scenario_cells.csv.gz"
)

YEARS = (2025, 2030, 2035)
ZERO_TOLERANCE = 1.0e-6
VALLEY_MIN_H = 8.0
VALLEY_MAX_H = 168.0
EXPECTED_COUNTRY_YEARS = 132
EXPECTED_LDES_CONFIGS = 49
EXPECTED_VALLEY_DURATIONS = 22


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(value: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [value > ZERO_TOLERANCE, value < -ZERO_TOLERANCE],
            ["positive", "negative"],
            default="zero",
        ),
        index=value.index,
    )


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [
        "_".join(str(token) for token in column if str(token))
        if isinstance(column, tuple)
        else str(column)
        for column in result.columns
    ]
    return result


def choose_duration_envelope(cells: pd.DataFrame) -> pd.DataFrame:
    """Select the best technology at every 8--168 h duration node."""

    work = cells[
        cells["duration_h"].between(
            VALLEY_MIN_H, VALLEY_MAX_H, inclusive="both"
        )
    ].copy()
    keys = [
        "cost_year",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "duration_h",
    ]
    work = work.sort_values(
        keys
        + [
            "mix_benefit_revalued_eur_kw_year",
            "candidate_fixed_cost_eur_kw_year",
            "config_id",
        ],
        ascending=[True] * len(keys) + [False, True, True],
        kind="stable",
    )
    envelope = work.drop_duplicates(keys, keep="first").copy()
    counts = envelope.groupby(
        ["cost_year", "country_year_id", "chronology"]
    )["duration_h"].nunique()
    if set(counts) != {EXPECTED_VALLEY_DURATIONS}:
        raise RuntimeError("The duration envelope is not complete")
    return envelope


def duration_flip_records(envelope: pd.DataFrame) -> pd.DataFrame:
    index = ["cost_year", "country", "year", "country_year_id", "duration_h"]
    metrics = [
        "mix_benefit_revalued_eur_kw_year",
        "config_id",
        "tech_id",
        "candidate_fixed_cost_eur_kw_year",
    ]
    wide = flatten_columns(
        envelope[index + ["chronology"] + metrics]
        .pivot(index=index, columns="chronology", values=metrics)
        .reset_index()
    )
    benefit_d = wide["mix_benefit_revalued_eur_kw_year_dispersed"].astype(
        float
    )
    benefit_c = wide["mix_benefit_revalued_eur_kw_year_clustered"].astype(
        float
    )
    wide["sign_dispersed"] = classify(benefit_d)
    wide["sign_clustered"] = classify(benefit_c)
    wide["sign_transition"] = (
        wide["sign_dispersed"] + "_to_" + wide["sign_clustered"]
    )
    wide["ldes_entry_dispersed"] = benefit_d > ZERO_TOLERANCE
    wide["ldes_entry_clustered"] = benefit_c > ZERO_TOLERANCE
    wide["cluster_induced_exit"] = (
        wide["ldes_entry_dispersed"] & ~wide["ldes_entry_clustered"]
    )
    wide["cluster_induced_entry"] = (
        ~wide["ldes_entry_dispersed"] & wide["ldes_entry_clustered"]
    )
    wide["technology_identity_flip_at_duration"] = (
        wide["tech_id_dispersed"] != wide["tech_id_clustered"]
    )
    wide["configuration_identity_flip_at_duration"] = (
        wide["config_id_dispersed"] != wide["config_id_clustered"]
    )
    wide["cluster_effect_eur_kw_year"] = benefit_c - benefit_d
    wide["absolute_cluster_effect_eur_kw_year"] = wide[
        "cluster_effect_eur_kw_year"
    ].abs()
    # A common additive shift to all LDES-minus-LFP benefits changes the
    # chronology decision when it crosses the nearer of the two zero points.
    wide["decision_margin_to_zero_eur_kw_year"] = np.minimum(
        benefit_d.abs(), benefit_c.abs()
    )
    wide.loc[
        wide["ldes_entry_dispersed"] != wide["ldes_entry_clustered"],
        "decision_margin_to_zero_eur_kw_year",
    ] = 0.0
    return wide.sort_values(index, kind="stable")


def duration_flip_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cost_year, group in records.groupby("cost_year", sort=True):
        rows.append(
            {
                "cost_year": int(cost_year),
                "n_country_year_duration_nodes": int(len(group)),
                "n_countries": int(group["country"].nunique()),
                "positive_nodes_dispersed": int(
                    group["ldes_entry_dispersed"].sum()
                ),
                "positive_nodes_clustered": int(
                    group["ldes_entry_clustered"].sum()
                ),
                "cluster_induced_exit_count": int(
                    group["cluster_induced_exit"].sum()
                ),
                "cluster_induced_exit_rate": float(
                    group["cluster_induced_exit"].mean()
                ),
                "cluster_induced_entry_count": int(
                    group["cluster_induced_entry"].sum()
                ),
                "cluster_induced_entry_rate": float(
                    group["cluster_induced_entry"].mean()
                ),
                "countries_with_any_sign_flip": int(
                    group.groupby("country")[
                        ["cluster_induced_exit", "cluster_induced_entry"]
                    ].any().any(axis=1).sum()
                ),
                "technology_identity_flip_count_at_duration": int(
                    group["technology_identity_flip_at_duration"].sum()
                ),
                "configuration_identity_flip_count_at_duration": int(
                    group["configuration_identity_flip_at_duration"].sum()
                ),
                "median_abs_cluster_effect_eur_kw_year": float(
                    group["absolute_cluster_effect_eur_kw_year"].median()
                ),
                "median_unflipped_margin_eur_kw_year": float(
                    group.loc[
                        ~group["cluster_induced_exit"]
                        & ~group["cluster_induced_entry"],
                        "decision_margin_to_zero_eur_kw_year",
                    ].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def choose_portfolio(cells: pd.DataFrame, domain: str) -> pd.DataFrame:
    if domain == "8_to_168_h":
        work = cells[
            cells["duration_h"].between(
                VALLEY_MIN_H, VALLEY_MAX_H, inclusive="both"
            )
        ].copy()
    elif domain == "all_49_ldes_configurations":
        work = cells.copy()
    else:
        raise ValueError(domain)
    keys = [
        "cost_year",
        "country",
        "year",
        "country_year_id",
        "chronology",
    ]
    work = work.sort_values(
        keys
        + [
            "mix_benefit_revalued_eur_kw_year",
            "candidate_fixed_cost_eur_kw_year",
            "duration_h",
            "config_id",
        ],
        ascending=[True] * len(keys) + [False, True, True, True],
        kind="stable",
    )
    chosen = work.drop_duplicates(keys, keep="first").copy()
    chosen["decision_domain"] = domain
    return chosen


def portfolio_flip_records(chosen: pd.DataFrame) -> pd.DataFrame:
    index = [
        "decision_domain",
        "cost_year",
        "country",
        "year",
        "country_year_id",
    ]
    metrics = [
        "mix_benefit_revalued_eur_kw_year",
        "config_id",
        "tech_id",
        "duration_h",
        "candidate_fixed_cost_eur_kw_year",
    ]
    wide = flatten_columns(
        chosen[index + ["chronology"] + metrics]
        .pivot(index=index, columns="chronology", values=metrics)
        .reset_index()
    )
    benefit_d = wide["mix_benefit_revalued_eur_kw_year_dispersed"].astype(
        float
    )
    benefit_c = wide["mix_benefit_revalued_eur_kw_year_clustered"].astype(
        float
    )
    wide["ldes_entry_dispersed"] = benefit_d > ZERO_TOLERANCE
    wide["ldes_entry_clustered"] = benefit_c > ZERO_TOLERANCE
    wide["cluster_induced_exit"] = (
        wide["ldes_entry_dispersed"] & ~wide["ldes_entry_clustered"]
    )
    wide["cluster_induced_entry"] = (
        ~wide["ldes_entry_dispersed"] & wide["ldes_entry_clustered"]
    )
    wide["shadow_ldes_config_flip"] = (
        wide["config_id_dispersed"] != wide["config_id_clustered"]
    )
    wide["shadow_ldes_technology_flip"] = (
        wide["tech_id_dispersed"] != wide["tech_id_clustered"]
    )
    wide["shadow_ldes_duration_flip"] = (
        wide["duration_h_dispersed"].astype(float)
        != wide["duration_h_clustered"].astype(float)
    )

    for chronology, entry in [
        ("dispersed", wide["ldes_entry_dispersed"]),
        ("clustered", wide["ldes_entry_clustered"]),
    ]:
        wide[f"selected_arm_{chronology}"] = np.where(
            entry, "ldes_tail", "additional_lfp4"
        )
        wide[f"selected_config_{chronology}"] = np.where(
            entry, wide[f"config_id_{chronology}"], "additional_lfp4"
        )
        wide[f"selected_technology_{chronology}"] = np.where(
            entry, wide[f"tech_id_{chronology}"], "li_lfp"
        )
        wide[f"selected_duration_h_{chronology}"] = np.where(
            entry, wide[f"duration_h_{chronology}"].astype(float), 4.0
        )
    wide["selected_arm_flip"] = (
        wide["selected_arm_dispersed"] != wide["selected_arm_clustered"]
    )
    wide["selected_config_identity_flip"] = (
        wide["selected_config_dispersed"]
        != wide["selected_config_clustered"]
    )
    wide["selected_technology_identity_flip"] = (
        wide["selected_technology_dispersed"]
        != wide["selected_technology_clustered"]
    )
    wide["selected_duration_identity_flip"] = (
        wide["selected_duration_h_dispersed"]
        != wide["selected_duration_h_clustered"]
    )
    wide["cluster_effect_eur_kw_year"] = benefit_c - benefit_d
    wide["absolute_cluster_effect_eur_kw_year"] = wide[
        "cluster_effect_eur_kw_year"
    ].abs()
    wide["decision_margin_to_zero_eur_kw_year"] = np.minimum(
        benefit_d.abs(), benefit_c.abs()
    )
    wide.loc[
        wide["selected_arm_flip"], "decision_margin_to_zero_eur_kw_year"
    ] = 0.0
    return wide.sort_values(index, kind="stable")


def portfolio_flip_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain, cost_year), group in records.groupby(
        ["decision_domain", "cost_year"], sort=True
    ):
        unflipped = group[~group["selected_arm_flip"]]
        rows.append(
            {
                "decision_domain": domain,
                "cost_year": int(cost_year),
                "n_country_years": int(len(group)),
                "n_countries": int(group["country"].nunique()),
                "ldes_selected_dispersed_count": int(
                    group["ldes_entry_dispersed"].sum()
                ),
                "ldes_selected_clustered_count": int(
                    group["ldes_entry_clustered"].sum()
                ),
                "cluster_induced_exit_count": int(
                    group["cluster_induced_exit"].sum()
                ),
                "cluster_induced_exit_rate": float(
                    group["cluster_induced_exit"].mean()
                ),
                "cluster_induced_entry_count": int(
                    group["cluster_induced_entry"].sum()
                ),
                "cluster_induced_entry_rate": float(
                    group["cluster_induced_entry"].mean()
                ),
                "countries_with_any_selected_arm_flip": int(
                    group.groupby("country")["selected_arm_flip"].any().sum()
                ),
                "selected_config_identity_flip_count": int(
                    group["selected_config_identity_flip"].sum()
                ),
                "selected_technology_identity_flip_count": int(
                    group["selected_technology_identity_flip"].sum()
                ),
                "selected_duration_identity_flip_count": int(
                    group["selected_duration_identity_flip"].sum()
                ),
                "shadow_ldes_config_flip_count": int(
                    group["shadow_ldes_config_flip"].sum()
                ),
                "shadow_ldes_technology_flip_count": int(
                    group["shadow_ldes_technology_flip"].sum()
                ),
                "shadow_ldes_duration_flip_count": int(
                    group["shadow_ldes_duration_flip"].sum()
                ),
                "median_abs_cluster_effect_eur_kw_year": float(
                    group["absolute_cluster_effect_eur_kw_year"].median()
                ),
                "median_unflipped_margin_eur_kw_year": float(
                    unflipped["decision_margin_to_zero_eur_kw_year"].median()
                ),
                "minimum_unflipped_margin_eur_kw_year": float(
                    unflipped["decision_margin_to_zero_eur_kw_year"].min()
                ),
            }
        )
    return pd.DataFrame(rows)


def add_cost_path_comparison(
    records: pd.DataFrame, index: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare chronology effects with the 2025--2035 benefit-path span."""

    value_columns = [
        "mix_benefit_revalued_eur_kw_year_dispersed",
        "mix_benefit_revalued_eur_kw_year_clustered",
    ]
    wide = records.pivot(index=index, columns="cost_year", values=value_columns)
    wide = flatten_columns(wide.reset_index())
    for chronology in ("dispersed", "clustered"):
        columns = [
            f"mix_benefit_revalued_eur_kw_year_{chronology}_{year}"
            for year in YEARS
        ]
        wide[f"cost_path_span_{chronology}_eur_kw_year"] = (
            wide[columns].max(axis=1) - wide[columns].min(axis=1)
        )
    wide["mean_chronology_cost_path_span_eur_kw_year"] = 0.5 * (
        wide["cost_path_span_dispersed_eur_kw_year"]
        + wide["cost_path_span_clustered_eur_kw_year"]
    )
    long_rows: list[pd.DataFrame] = []
    for year in YEARS:
        part = wide[index].copy()
        part["cost_year"] = year
        part["benefit_dispersed_eur_kw_year"] = wide[
            f"mix_benefit_revalued_eur_kw_year_dispersed_{year}"
        ]
        part["benefit_clustered_eur_kw_year"] = wide[
            f"mix_benefit_revalued_eur_kw_year_clustered_{year}"
        ]
        part["cluster_effect_eur_kw_year"] = (
            part["benefit_clustered_eur_kw_year"]
            - part["benefit_dispersed_eur_kw_year"]
        )
        part["absolute_cluster_effect_eur_kw_year"] = part[
            "cluster_effect_eur_kw_year"
        ].abs()
        part["cost_path_span_dispersed_eur_kw_year"] = wide[
            "cost_path_span_dispersed_eur_kw_year"
        ]
        part["cost_path_span_clustered_eur_kw_year"] = wide[
            "cost_path_span_clustered_eur_kw_year"
        ]
        part["mean_chronology_cost_path_span_eur_kw_year"] = wide[
            "mean_chronology_cost_path_span_eur_kw_year"
        ]
        span = part["mean_chronology_cost_path_span_eur_kw_year"]
        part["cluster_effect_to_cost_path_span_ratio"] = np.where(
            span > ZERO_TOLERANCE,
            part["absolute_cluster_effect_eur_kw_year"] / span,
            np.nan,
        )
        part["decision_margin_to_zero_eur_kw_year"] = np.minimum(
            part["benefit_dispersed_eur_kw_year"].abs(),
            part["benefit_clustered_eur_kw_year"].abs(),
        )
        flip = (
            (part["benefit_dispersed_eur_kw_year"] > ZERO_TOLERANCE)
            != (part["benefit_clustered_eur_kw_year"] > ZERO_TOLERANCE)
        )
        part.loc[flip, "decision_margin_to_zero_eur_kw_year"] = 0.0
        part["cluster_effect_at_least_cost_path_span"] = (
            part["absolute_cluster_effect_eur_kw_year"] >= span
        )
        part["decision_margin_within_cost_path_span"] = (
            part["decision_margin_to_zero_eur_kw_year"] <= span
        )
        part["decision_margin_within_cluster_effect"] = (
            part["decision_margin_to_zero_eur_kw_year"]
            <= part["absolute_cluster_effect_eur_kw_year"]
        )
        long_rows.append(part)
    long = pd.concat(long_rows, ignore_index=True)
    group_columns = [column for column in index if column.startswith("decision_domain")]
    group_columns += ["cost_year"]
    rows: list[dict[str, object]] = []
    for key, group in long.groupby(group_columns, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_cases": int(len(group)),
                "median_abs_cluster_effect_eur_kw_year": float(
                    group["absolute_cluster_effect_eur_kw_year"].median()
                ),
                "median_cost_path_span_eur_kw_year": float(
                    group[
                        "mean_chronology_cost_path_span_eur_kw_year"
                    ].median()
                ),
                "median_effect_to_span_ratio": float(
                    group["cluster_effect_to_cost_path_span_ratio"].median()
                ),
                "effect_at_least_span_count": int(
                    group["cluster_effect_at_least_cost_path_span"].sum()
                ),
                "effect_at_least_span_rate": float(
                    group["cluster_effect_at_least_cost_path_span"].mean()
                ),
                "margin_within_span_count": int(
                    group["decision_margin_within_cost_path_span"].sum()
                ),
                "margin_within_span_rate": float(
                    group["decision_margin_within_cost_path_span"].mean()
                ),
                "margin_within_cluster_effect_count": int(
                    group["decision_margin_within_cluster_effect"].sum()
                ),
                "margin_within_cluster_effect_rate": float(
                    group["decision_margin_within_cluster_effect"].mean()
                ),
            }
        )
        rows.append(row)
    return long, pd.DataFrame(rows)


def bounded_s31_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "source_label",
        "model_mode",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "config_id",
        "tech_id",
        "duration_h",
        "tail_power_mw",
        "tail_power_upper_mw",
        "mix_benefit_eur_per_kw_common_power_year",
        "valuation_scenario_id",
        "valuation_energy_value_eur_mwh",
        "valuation_firm_value_eur_kw_year",
        "valuation_charge_price_eur_mwh",
    ]
    cells = pd.read_csv(BOUNDED_CELLS, usecols=columns)
    cells = cells[
        cells["source_label"].eq("optimized_s31_robustness")
        & cells["model_mode"].eq("optimized")
        & cells["valuation_scenario_id"].eq(31)
    ].copy()
    if cells["config_id"].nunique() != EXPECTED_LDES_CONFIGS:
        raise RuntimeError("Bounded S31 catalogue is incomplete")
    keys = ["country", "year", "country_year_id", "chronology"]
    chosen = cells.sort_values(
        keys + ["mix_benefit_eur_per_kw_common_power_year", "config_id"],
        ascending=[True] * len(keys) + [False, True],
        kind="stable",
    ).drop_duplicates(keys, keep="first")
    metrics = [
        "mix_benefit_eur_per_kw_common_power_year",
        "config_id",
        "tech_id",
        "duration_h",
        "tail_power_mw",
        "tail_power_upper_mw",
    ]
    wide = flatten_columns(
        chosen[["country", "year", "country_year_id", "chronology"] + metrics]
        .pivot(
            index=["country", "year", "country_year_id"],
            columns="chronology",
            values=metrics,
        )
        .reset_index()
    )
    entry_d = wide["tail_power_mw_dispersed"].astype(float) > ZERO_TOLERANCE
    entry_c = wide["tail_power_mw_clustered"].astype(float) > ZERO_TOLERANCE
    wide["tail_selected_dispersed"] = entry_d
    wide["tail_selected_clustered"] = entry_c
    wide["cluster_induced_tail_exit"] = entry_d & ~entry_c
    wide["cluster_induced_tail_entry"] = ~entry_d & entry_c
    wide["tail_entry_identity_flip"] = entry_d != entry_c
    # Config labels are not economically identified when optimal power is zero.
    wide["selected_config_dispersed"] = np.where(
        entry_d, wide["config_id_dispersed"], "no_tail"
    )
    wide["selected_config_clustered"] = np.where(
        entry_c, wide["config_id_clustered"], "no_tail"
    )
    wide["selected_config_identity_flip"] = (
        wide["selected_config_dispersed"]
        != wide["selected_config_clustered"]
    )
    summary = pd.DataFrame(
        [
            {
                "analysis_case": "bounded_power_s31_2035_cost",
                "energy_value_eur_mwh": float(
                    cells["valuation_energy_value_eur_mwh"].iloc[0]
                ),
                "firm_value_eur_kw_year": float(
                    cells["valuation_firm_value_eur_kw_year"].iloc[0]
                ),
                "charge_price_eur_mwh": float(
                    cells["valuation_charge_price_eur_mwh"].iloc[0]
                ),
                "n_country_years": int(len(wide)),
                "tail_selected_dispersed_count": int(entry_d.sum()),
                "tail_selected_clustered_count": int(entry_c.sum()),
                "cluster_induced_tail_exit_count": int(
                    wide["cluster_induced_tail_exit"].sum()
                ),
                "cluster_induced_tail_exit_rate": float(
                    wide["cluster_induced_tail_exit"].mean()
                ),
                "cluster_induced_tail_entry_count": int(
                    wide["cluster_induced_tail_entry"].sum()
                ),
                "countries_with_any_tail_exit": int(
                    wide.groupby("country")[
                        "cluster_induced_tail_exit"
                    ].any().sum()
                ),
                "selected_config_identity_flip_count": int(
                    wide["selected_config_identity_flip"].sum()
                ),
                "scope_note": (
                    "Exact S31 bounded-power solve with 2035 costs; not a "
                    "central-S20 or three-cost-year result"
                ),
            }
        ]
    )
    return wide, summary


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
        )
    headers = [str(column) for column in shown.columns]
    rows = [
        [str(value) for value in row]
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def write_readme(
    duration_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    portfolio_span_summary: pd.DataFrame,
    bounded_summary: pd.DataFrame,
    portfolio_records: pd.DataFrame,
) -> None:
    full = portfolio_summary[
        portfolio_summary["decision_domain"].eq(
            "all_49_ldes_configurations"
        )
    ]
    actual_flips = portfolio_records[
        portfolio_records["decision_domain"].eq(
            "all_49_ldes_configurations"
        )
        & portfolio_records["selected_arm_flip"]
    ][
        [
            "cost_year",
            "country_year_id",
            "selected_config_dispersed",
            "mix_benefit_revalued_eur_kw_year_dispersed",
            "selected_config_clustered",
            "mix_benefit_revalued_eur_kw_year_clustered",
        ]
    ]
    lines = [
        "# V44 decision-flip audit",
        "",
        "## What is tested",
        "",
        "The primary estimand is whether the dispersed-to-clustered chronology "
        "change alters the economically selected arm under the central service "
        "values and the technology-specific 2025, 2030 and 2035 costs. The "
        "additional 4 h LFP arm is the zero-benefit outside option. A negative "
        "best LDES benefit therefore means that the actual selection remains "
        "additional LFP, even if the identity of the best losing LDES "
        "configuration changes.",
        "",
        "## Main result",
        "",
        "Across the full 49-configuration catalogue, clustering changes the "
        "economic arm in 0 of 132 country-years with 2025 costs and in 1 of "
        "132 country-years with each of the 2030 and 2035 costs. Both later "
        "flips are the same AT-2024 case, where dispersed CAES 24 h is positive "
        "and clustering returns the choice to additional 4 h LFP. There is no "
        "cluster-induced LDES entry in any cost year. Thus the central fixed-"
        "scale decision result is localized rather than a broad procurement "
        "reordering.",
        "",
        markdown_table(full),
        "",
        "Actual arm flips:",
        "",
        markdown_table(actual_flips),
        "",
        "## Duration-node sign audit",
        "",
        "The 8--168 h technology envelope contains 2,904 country-year-duration "
        "nodes per cost year. Clustering induces 0, 1 and 3 positive-to-"
        "nonpositive transitions under the 2025, 2030 and 2035 costs, "
        "respectively. All occur in AT-2024 and all are CAES nodes. No "
        "negative-to-positive transition or technology-family switch occurs at "
        "a fixed duration.",
        "",
        markdown_table(duration_summary),
        "",
        "## Distance to the boundary and the cost path",
        "",
        "The cost-path comparison uses the exact 2025--2035 change in the "
        "released central cost trajectory. It is a learning-path span, not a "
        "probabilistic cost-uncertainty interval. At the all-catalogue portfolio "
        "level, the median absolute chronology effect is 5.48 EUR kW-1 yr-1 "
        "and the median 2025--2035 span is 6.03 EUR kW-1 yr-1. The chronology "
        "effect is at least as large as the path span in 60 of 132 cases "
        "(45.5%), but only 1--4 cases per cost year lie within one path span of "
        "the zero-benefit decision boundary. This separates a value effect of "
        "material scale from an actual procurement flip.",
        "",
        markdown_table(
            portfolio_span_summary[
                portfolio_span_summary["decision_domain"].eq(
                    "all_49_ldes_configurations"
                )
            ]
        ),
        "",
        "## Bounded-power robustness already available in v43",
        "",
        "The existing exact S31 solve uses 2035 costs, vE=98.54 EUR MWh-1, "
        "vF=103.80 EUR kW-1 yr-1 and zero charging price. Under this valuation, "
        "92 of 132 dispersed country-years spanning 11 of 15 countries select positive LDES tail power, "
        "whereas all 132 clustered country-years select zero tail power. The "
        "result is a direct 92/132 procurement exit, but it is not a result for "
        "the central S20 valuation and must remain labelled as a bounded-power "
        "sensitivity.",
        "",
        markdown_table(bounded_summary),
        "",
        "## Reproducibility",
        "",
        "Run `python run_decision_flip_audit.py` from this directory. The script "
        "validates the expected 132 country-years, 49 LDES configurations and "
        "22 duration nodes in the published 8--168 h valley domain, and writes "
        "a SHA-256 manifest for every output.",
        "",
    ]
    (HERE / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cells = pd.read_csv(CENTRAL_CELLS)
    cells = cells[
        cells["revaluation_case_id"].eq("central_actual_year_cost")
    ].copy()
    if set(cells["cost_year"].astype(int)) != set(YEARS):
        raise RuntimeError("Central cost-year panel is incomplete")
    if cells["country_year_id"].nunique() != EXPECTED_COUNTRY_YEARS:
        raise RuntimeError("Expected 132 country-years")
    if cells["config_id"].nunique() != EXPECTED_LDES_CONFIGS:
        raise RuntimeError("Expected 49 LDES configurations")
    if cells.duplicated(
        ["cost_year", "country_year_id", "chronology", "config_id"]
    ).any():
        raise RuntimeError("Central cell keys are not unique")

    duration_envelope = choose_duration_envelope(cells)
    duration_records = duration_flip_records(duration_envelope)
    duration_summary = duration_flip_summary(duration_records)

    choices = pd.concat(
        [
            choose_portfolio(cells, "all_49_ldes_configurations"),
            choose_portfolio(cells, "8_to_168_h"),
        ],
        ignore_index=True,
    )
    portfolio_records = portfolio_flip_records(choices)
    portfolio_summary = portfolio_flip_summary(portfolio_records)

    duration_span, duration_span_summary = add_cost_path_comparison(
        duration_records,
        ["country", "year", "country_year_id", "duration_h"],
    )
    portfolio_span, portfolio_span_summary = add_cost_path_comparison(
        portfolio_records,
        ["decision_domain", "country", "year", "country_year_id"],
    )
    bounded_records, bounded_summary = bounded_s31_audit()

    outputs: dict[str, pd.DataFrame] = {
        "duration_envelope_decision_flips.csv.gz": duration_records,
        "duration_envelope_decision_flip_summary.csv": duration_summary,
        "portfolio_choice_decision_flips.csv": portfolio_records,
        "portfolio_choice_decision_flip_summary.csv": portfolio_summary,
        "duration_cost_path_span_comparison.csv.gz": duration_span,
        "duration_cost_path_span_summary.csv": duration_span_summary,
        "portfolio_cost_path_span_comparison.csv": portfolio_span,
        "portfolio_cost_path_span_summary.csv": portfolio_span_summary,
        "bounded_power_s31_choice_flips.csv": bounded_records,
        "bounded_power_s31_choice_flip_summary.csv": bounded_summary,
    }
    for filename, frame in outputs.items():
        compression = "gzip" if filename.endswith(".gz") else None
        frame.to_csv(HERE / filename, index=False, compression=compression)

    write_readme(
        duration_summary,
        portfolio_summary,
        portfolio_span_summary,
        bounded_summary,
        portfolio_records,
    )
    output_files = [HERE / filename for filename in outputs] + [
        HERE / "README.md"
    ]
    manifest = {
        "analysis_id": "v44_decision_flip_audit",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "zero_tolerance_eur_kw_year": ZERO_TOLERANCE,
        "primary_scope": {
            "service_values": "central S20",
            "cost_years": list(YEARS),
            "country_years": EXPECTED_COUNTRY_YEARS,
            "ldes_configurations": EXPECTED_LDES_CONFIGS,
            "outside_option": "additional 4 h LFP",
        },
        "bounded_power_scope": (
            "S31, 2035 costs, vE=98.5403, vF=103.7992, cQ=0; not central S20"
        ),
        "inputs": {
            str(CENTRAL_CELLS): sha256(CENTRAL_CELLS),
            str(BOUNDED_CELLS): sha256(BOUNDED_CELLS),
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        },
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Hard gates for the manuscript-facing claims.
    full = portfolio_summary[
        portfolio_summary["decision_domain"].eq(
            "all_49_ldes_configurations"
        )
    ].set_index("cost_year")
    if full["cluster_induced_exit_count"].astype(int).to_dict() != {
        2025: 0,
        2030: 1,
        2035: 1,
    }:
        raise RuntimeError("Unexpected central portfolio flip counts")
    if duration_summary.set_index("cost_year")[
        "cluster_induced_exit_count"
    ].astype(int).to_dict() != {2025: 0, 2030: 1, 2035: 3}:
        raise RuntimeError("Unexpected duration-node flip counts")
    if int(bounded_summary["cluster_induced_tail_exit_count"].iloc[0]) != 92:
        raise RuntimeError("Unexpected bounded S31 tail-exit count")
    print("PASS: v44 decision-flip audit outputs written to", HERE)


if __name__ == "__main__":
    main()
