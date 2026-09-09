#!/usr/bin/env python3
"""V43 cost-year revaluation and paired value-channel decomposition.

This module deliberately works from the already solved fixed-power physical
trajectories.  Changing only annual fixed costs is therefore an exact
revaluation of the fixed-power model because fixed costs do not enter any
hourly feasibility or marginal-dispatch condition and technology VOM is
unchanged across the three cost years.  Changing service prices (notably the
firm-supply value) is reported separately as a frozen-trajectory sensitivity;
it is not described as a re-optimized dispatch result.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V43 = HERE.parent
V40 = V43.parent / "revision_nature_energy_v40"
COST_SOURCE = (
    V40
    / "currency_work"
    / "ed1_53_duration_year_cost_paths_source_aware_eur.csv"
)
CELL_SOURCE = V43 / "source_data" / "v42_fixed_scale_mix_cells.csv.gz"
FROZEN_SOURCE_DIR = V43 / "source_data"

YEARS = (2025, 2030, 2035)
VALLEY_MIN_H = 8.0
VALLEY_MAX_H = 168.0
ZERO_TOLERANCE = 1.0e-6
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260729

# Exact scenario values carried by the S20 physical trajectories.
CENTRAL_ENERGY_VALUE = 65.23332265557728
CENTRAL_FIRM_VALUE = 103.79921785880974
CENTRAL_CHARGE_PRICE = 6.15876989995162


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_id(tech_id: str, duration_h: float) -> str:
    duration = float(duration_h)
    token = f"{duration:g}"
    return f"{tech_id}_{token}h"


def build_cost_catalogues() -> pd.DataFrame:
    source = pd.read_csv(COST_SOURCE)
    required = {
        "learning_year",
        "tech_id",
        "duration_h",
        "technology_en",
        "technology_currency_route",
        "annual_fixed_cost_eur_source_aware_per_kw_discharge_year",
        "variable_om_eur_source_aware_mwh",
        "total_capex_eur_source_aware_kwh",
        "power_cost_eur_source_aware_kw",
        "energy_cost_eur_source_aware_kwh",
        "fixed_om_eur_source_aware_kw_yr",
        "rte",
        "self_discharge_pct_day",
        "charge_power_ratio",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise RuntimeError(f"Cost source is missing columns: {missing}")
    source = source[source["learning_year"].isin(YEARS)].copy()
    source["config_id"] = [
        config_id(tech_id, duration_h)
        for tech_id, duration_h in zip(
            source["tech_id"], source["duration_h"]
        )
    ]
    columns = [
        "learning_year",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
        "technology_currency_route",
        "total_capex_eur_source_aware_kwh",
        "power_cost_eur_source_aware_kw",
        "energy_cost_eur_source_aware_kwh",
        "fixed_om_eur_source_aware_kw_yr",
        "variable_om_eur_source_aware_mwh",
        "annual_fixed_cost_eur_source_aware_per_kw_discharge_year",
        "rte",
        "self_discharge_pct_day",
        "charge_power_ratio",
        "calendar_life_years",
        "cycle_life_cycles",
        "discount_rate",
        "primary_source",
        "source_key",
        "source_url_primary",
        "source_url_secondary",
    ]
    catalogue = source[columns].sort_values(
        ["learning_year", "duration_h", "tech_id"], kind="stable"
    )
    if catalogue.groupby("learning_year").size().to_dict() != {
        2025: 53,
        2030: 53,
        2035: 53,
    }:
        raise RuntimeError("Expected exactly 53 configurations per cost year")
    if catalogue.duplicated(["learning_year", "config_id"]).any():
        raise RuntimeError("Cost-year/configuration keys are not unique")
    catalogue.to_csv(HERE / "cost_catalogue_2025_2035_long.csv", index=False)
    for year in YEARS:
        catalogue[catalogue["learning_year"].eq(year)].to_csv(
            HERE / f"cost_catalogue_{year}_eur.csv", index=False
        )
    return catalogue


def find_valley(frame: pd.DataFrame) -> dict[str, object]:
    ordered = frame.sort_values("duration_h", kind="stable").reset_index(
        drop=True
    )
    values = ordered["best_benefit_eur_kw_year"].to_numpy(float)
    positive = values > ZERO_TOLERANCE
    negative = values < -ZERO_TOLERANCE
    candidates: list[dict[str, object]] = []
    index = 0
    while index < len(ordered):
        if not negative[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(ordered) and negative[index + 1]:
            index += 1
        end = index
        if (
            end - start + 1 >= 2
            and start > 0
            and end < len(ordered) - 1
            and positive[start - 1]
            and positive[end + 1]
        ):
            gap = ordered.iloc[start : end + 1]
            left = ordered.iloc[start - 1]
            right = ordered.iloc[end + 1]
            candidates.append(
                {
                    "strict_valley": True,
                    "left_boundary_h": float(left["duration_h"]),
                    "right_boundary_h": float(right["duration_h"]),
                    "valley_width_h": float(
                        right["duration_h"] - left["duration_h"]
                    ),
                    "negative_node_count": int(len(gap)),
                    "negative_durations_h": "|".join(
                        f"{value:g}" for value in gap["duration_h"]
                    ),
                    "minimum_gap_depth_eur_kw_year": float(
                        gap["best_benefit_eur_kw_year"].min()
                    ),
                }
            )
        index += 1
    # Retain v42's strict rule: an unresolved node anywhere on an otherwise
    # bounded envelope withholds a strict-valley classification.
    if candidates and np.any(np.abs(values) <= ZERO_TOLERANCE):
        candidates = []
    if candidates:
        candidates.sort(
            key=lambda item: (
                -float(item["valley_width_h"]),
                float(item["minimum_gap_depth_eur_kw_year"]),
                float(item["left_boundary_h"]),
            )
        )
        selected = candidates[0]
        selected["envelope_shape"] = "strict_valley"
        selected["strict_valley_count"] = len(candidates)
        return selected
    if len(values) and np.all(positive):
        shape = "full_ldes_entry"
    elif len(values) and np.all(negative):
        shape = "full_ldes_exit"
    elif (
        len(values)
        and positive[0]
        and not positive[-1]
        and np.sum(np.diff(positive.astype(int)) != 0) == 1
    ):
        shape = "positive_to_exit_frontier"
    elif (
        len(values)
        and not positive[0]
        and positive[-1]
        and np.sum(np.diff(positive.astype(int)) != 0) == 1
    ):
        shape = "entry_frontier"
    else:
        shape = "complex_nonvalley"
    return {
        "strict_valley": False,
        "envelope_shape": shape,
        "strict_valley_count": 0,
        "left_boundary_h": np.nan,
        "right_boundary_h": np.nan,
        "valley_width_h": 0.0,
        "negative_node_count": 0,
        "negative_durations_h": "",
        "minimum_gap_depth_eur_kw_year": np.nan,
    }


def revalue_cells(
    cells: pd.DataFrame,
    catalogue: pd.DataFrame,
    cost_year: int,
    case_id: str,
    energy_value: float,
    firm_value: float,
    charge_price: float,
    ldes_cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    cost = catalogue[catalogue["learning_year"].eq(cost_year)][
        [
            "config_id",
            "annual_fixed_cost_eur_source_aware_per_kw_discharge_year",
        ]
    ].rename(
        columns={
            "annual_fixed_cost_eur_source_aware_per_kw_discharge_year": (
                "candidate_fixed_cost_eur_kw_year"
            )
        }
    )
    work = cells.merge(cost, on="config_id", how="left", validate="many_to_one")
    if work["candidate_fixed_cost_eur_kw_year"].isna().any():
        missing = sorted(
            work.loc[
                work["candidate_fixed_cost_eur_kw_year"].isna(), "config_id"
            ].unique()
        )
        raise RuntimeError(f"Missing cost rows for candidate IDs: {missing}")
    lfp = catalogue[
        catalogue["learning_year"].eq(cost_year)
        & catalogue["config_id"].eq("li_lfp_4h")
    ]
    if len(lfp) != 1:
        raise RuntimeError(f"4 h LFP cost is non-unique in {cost_year}")
    lfp_fixed = float(
        lfp[
            "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"
        ].iloc[0]
    )
    work["cost_year"] = int(cost_year)
    work["revaluation_case_id"] = case_id
    work["revaluation_energy_value_eur_mwh"] = energy_value
    work["revaluation_firm_value_eur_kw_year"] = firm_value
    work["revaluation_charge_price_eur_mwh"] = charge_price
    work["ldes_fixed_cost_multiplier"] = ldes_cost_multiplier
    work["lfp4_fixed_cost_eur_kw_year"] = lfp_fixed
    work["candidate_fixed_cost_eur_kw_year_unscaled"] = work[
        "candidate_fixed_cost_eur_kw_year"
    ]
    work["candidate_fixed_cost_eur_kw_year"] *= ldes_cost_multiplier

    base_kw = work["base_power_mw"] * 1000.0
    candidate_tail_kw = work["tail_power_mw"] * 1000.0
    comparator_tail_kw = work["extra_lfp4_tail_power_mw"] * 1000.0
    work["candidate_portfolio_fixed_cost_eur_year"] = (
        lfp_fixed * base_kw
        + work["candidate_fixed_cost_eur_kw_year"] * candidate_tail_kw
    )
    work["comparator_portfolio_fixed_cost_eur_year"] = (
        lfp_fixed * base_kw + lfp_fixed * comparator_tail_kw
    )
    work["fixed_cost_contribution_revalued_eur_year"] = (
        work["comparator_portfolio_fixed_cost_eur_year"]
        - work["candidate_portfolio_fixed_cost_eur_year"]
    )
    work["cycle_return_contribution_eur_year"] = (
        energy_value * work["mix_discharge_gain_vs_extra_lfp4_mwh"]
        + charge_price
        * (
            work["direct_extra_lfp4_total_charge_mwh"]
            - work["total_charge_mwh"]
        )
        + work["variable_om_contribution_eur"]
    )
    work["coverage_return_contribution_eur_year"] = (
        firm_value * 1000.0 * work["mix_efc_gain_vs_extra_lfp4_mw"]
    )
    work["mix_benefit_revalued_eur_year"] = (
        work["fixed_cost_contribution_revalued_eur_year"]
        + work["cycle_return_contribution_eur_year"]
        + work["coverage_return_contribution_eur_year"]
    )
    work["mix_benefit_revalued_eur_kw_year"] = (
        work["mix_benefit_revalued_eur_year"] / work["common_base_power_kw"]
    )
    return work


def choose_envelope(cells: pd.DataFrame) -> pd.DataFrame:
    work = cells[
        cells["duration_h"].between(
            VALLEY_MIN_H, VALLEY_MAX_H, inclusive="both"
        )
    ].copy()
    keys = [
        "revaluation_case_id",
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
    return envelope.rename(
        columns={
            "config_id": "dominant_config_id",
            "tech_id": "dominant_tech_id",
            "mix_benefit_revalued_eur_kw_year": (
                "best_benefit_eur_kw_year"
            ),
        }
    )


def valley_records(envelope: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "revaluation_case_id",
        "cost_year",
        "country",
        "year",
        "country_year_id",
        "chronology",
    ]
    rows: list[dict[str, object]] = []
    for key, group in envelope.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key))
        row.update(find_valley(group))
        row["tested_duration_count"] = int(group["duration_h"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_valleys(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        records.groupby(
            ["revaluation_case_id", "cost_year", "chronology"],
            sort=True,
        )
        .agg(
            n_country_years=("country_year_id", "size"),
            strict_valley_count=("strict_valley", "sum"),
            strict_valley_rate=("strict_valley", "mean"),
            median_valley_width_h=(
                "valley_width_h",
                lambda x: float(np.median(x[x > 0])) if np.any(x > 0) else 0.0,
            ),
        )
        .reset_index()
    )
    pair = records.pivot(
        index=["revaluation_case_id", "cost_year", "country_year_id"],
        columns="chronology",
        values="strict_valley",
    ).reset_index()
    pair["paired_state"] = np.select(
        [
            ~pair["dispersed"] & pair["clustered"],
            pair["dispersed"] & pair["clustered"],
            pair["dispersed"] & ~pair["clustered"],
        ],
        ["cluster_induced", "maintained", "cluster_removed"],
        default="never",
    )
    paired_summary = (
        pair.groupby(["revaluation_case_id", "cost_year", "paired_state"])
        .size()
        .rename("n_country_years")
        .reset_index()
    )
    paired_summary["rate"] = paired_summary["n_country_years"] / 132.0
    return summary, paired_summary


def summarize_year_duration(envelope: pd.DataFrame) -> pd.DataFrame:
    """Freeze the duration-wise annual-cost envelopes used in Figure 6a."""

    return (
        envelope.groupby(
            [
                "revaluation_case_id",
                "cost_year",
                "chronology",
                "duration_h",
            ],
            sort=True,
        )["best_benefit_eur_kw_year"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            n_country_years="size",
        )
        .reset_index()
    )


def select_central_envelope(cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["country", "year", "country_year_id", "chronology", "duration_h"]
    work = cells.sort_values(
        keys
        + [
            "mix_benefit_eur_per_kw_common_power_year",
            "fixed_cost_eur_kw_year",
            "config_id",
        ],
        ascending=[True] * len(keys) + [False, True, True],
        kind="stable",
    )
    return work.drop_duplicates(keys, keep="first").copy()


def component_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    # Absolute arm changes are retained alongside the paired contribution.
    # They show whether a positive relative channel reflects an absolute gain
    # or a smaller loss than the additional-LFP comparator.
    work["candidate_absolute_cycle_return_eur_kw_year"] = (
        CENTRAL_ENERGY_VALUE * work["total_discharge_mwh"]
        - CENTRAL_CHARGE_PRICE * work["total_charge_mwh"]
        - work["base_vom_cost_eur"]
        - work["tail_vom_cost_eur"]
    ) / work["common_base_power_kw"]
    work["comparator_absolute_cycle_return_eur_kw_year"] = (
        CENTRAL_ENERGY_VALUE
        * work["direct_extra_lfp4_total_discharge_mwh"]
        - CENTRAL_CHARGE_PRICE
        * work["direct_extra_lfp4_total_charge_mwh"]
        - work["direct_extra_lfp4_base_vom_cost_eur"]
        - work["direct_extra_lfp4_tail_vom_cost_eur"]
    ) / work["common_base_power_kw"]
    work["candidate_absolute_coverage_return_eur_kw_year"] = (
        CENTRAL_FIRM_VALUE
        * 1000.0
        * work["portfolio_efc_mw"]
        / work["common_base_power_kw"]
    )
    work["comparator_absolute_coverage_return_eur_kw_year"] = (
        CENTRAL_FIRM_VALUE
        * 1000.0
        * work["direct_extra_lfp4_portfolio_efc_mw"]
        / work["common_base_power_kw"]
    )
    work["cycle_return_eur_kw_year"] = (
        work["energy_service_contribution_eur"]
        + work["charging_cost_contribution_eur"]
        + work["variable_om_contribution_eur"]
    ) / work["common_base_power_kw"]
    work["coverage_return_eur_kw_year"] = (
        work["firm_service_contribution_eur"]
        / work["common_base_power_kw"]
    )
    work["fixed_selection_eur_kw_year"] = (
        work["fixed_cost_contribution_eur"] / work["common_base_power_kw"]
    )
    work["benefit_eur_kw_year"] = (
        work["mix_benefit_eur_per_kw_common_power_year"]
    )
    metrics = [
        "benefit_eur_kw_year",
        "cycle_return_eur_kw_year",
        "coverage_return_eur_kw_year",
        "fixed_selection_eur_kw_year",
        "energy_service_contribution_eur_per_kw_common_power_year",
        "charging_cost_contribution_eur_per_kw_common_power_year",
        "variable_om_contribution_eur_per_kw_common_power_year",
        "tail_equivalent_full_cycles",
        "mix_efc_gain_vs_extra_lfp4_mw",
        "candidate_absolute_cycle_return_eur_kw_year",
        "comparator_absolute_cycle_return_eur_kw_year",
        "candidate_absolute_coverage_return_eur_kw_year",
        "comparator_absolute_coverage_return_eur_kw_year",
    ]
    index = [
        "country",
        "year",
        "country_year_id",
        "duration_h",
        "config_id",
        "tech_id",
    ]
    wide = work[index + ["chronology"] + metrics].pivot(
        index=index, columns="chronology", values=metrics
    )
    wide.columns = [
        f"{metric}_{chronology}" for metric, chronology in wide.columns
    ]
    wide = wide.reset_index()
    for metric in metrics:
        wide[f"delta_{metric}"] = (
            wide[f"{metric}_clustered"] - wide[f"{metric}_dispersed"]
        )
    wide["decomposition_closure_residual_eur_kw_year"] = (
        wide["delta_benefit_eur_kw_year"]
        - wide["delta_cycle_return_eur_kw_year"]
        - wide["delta_coverage_return_eur_kw_year"]
        - wide["delta_fixed_selection_eur_kw_year"]
    )
    return wide


def select_common_configuration(pairs: pd.DataFrame) -> pd.DataFrame:
    work = pairs.copy()
    work["mean_two_chronology_benefit_eur_kw_year"] = 0.5 * (
        work["benefit_eur_kw_year_clustered"]
        + work["benefit_eur_kw_year_dispersed"]
    )
    keys = ["country", "year", "country_year_id", "duration_h"]
    work = work.sort_values(
        keys
        + [
            "mean_two_chronology_benefit_eur_kw_year",
            "config_id",
        ],
        ascending=[True] * len(keys) + [False, True],
        kind="stable",
    )
    return work.drop_duplicates(keys, keep="first").copy()


def bootstrap_by_country(
    frame: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for duration, group in frame.groupby("duration_h", sort=True):
        country = (
            group.groupby("country", sort=True)[value_columns]
            .mean()
            .dropna()
        )
        array = country.to_numpy(float)
        n = len(array)
        draws = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
        boot = array[draws].mean(axis=1)
        means = array.mean(axis=0)
        medians = np.median(group[value_columns].to_numpy(float), axis=0)
        for position, column in enumerate(value_columns):
            rows.append(
                {
                    "duration_h": float(duration),
                    "metric": column,
                    "country_block_mean": float(means[position]),
                    "country_year_median": float(medians[position]),
                    "ci95_low": float(np.quantile(boot[:, position], 0.025)),
                    "ci95_high": float(
                        np.quantile(boot[:, position], 0.975)
                    ),
                    "n_countries": int(n),
                    "n_country_years": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def decompose(cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    config = component_pairs(cells)
    common = select_common_configuration(config)

    envelope = select_central_envelope(cells)
    chosen = envelope[
        [
            "country",
            "year",
            "country_year_id",
            "chronology",
            "duration_h",
            "config_id",
        ]
    ]
    selected = cells.merge(
        chosen,
        on=[
            "country",
            "year",
            "country_year_id",
            "chronology",
            "duration_h",
            "config_id",
        ],
        how="inner",
        validate="one_to_one",
    )
    envelope_config = component_pairs(selected)
    if len(envelope_config) != 132 * envelope["duration_h"].nunique():
        raise RuntimeError("Chronology-specific envelope does not form pairs")

    value_columns = [
        "delta_benefit_eur_kw_year",
        "delta_cycle_return_eur_kw_year",
        "delta_coverage_return_eur_kw_year",
        "delta_fixed_selection_eur_kw_year",
        "decomposition_closure_residual_eur_kw_year",
        "delta_tail_equivalent_full_cycles",
        "delta_mix_efc_gain_vs_extra_lfp4_mw",
        "delta_candidate_absolute_cycle_return_eur_kw_year",
        "delta_comparator_absolute_cycle_return_eur_kw_year",
        "delta_candidate_absolute_coverage_return_eur_kw_year",
        "delta_comparator_absolute_coverage_return_eur_kw_year",
    ]
    common_summary = bootstrap_by_country(common, value_columns)
    envelope_summary = bootstrap_by_country(envelope_config, value_columns)
    common_summary["selection_rule"] = (
        "one common configuration maximising mean benefit across chronologies"
    )
    envelope_summary["selection_rule"] = (
        "chronology-specific benefit envelope"
    )
    summary = pd.concat([common_summary, envelope_summary], ignore_index=True)

    focal = config[config["config_id"].eq("caes_24h")].copy()
    focal_summary = bootstrap_by_country(focal, value_columns)
    focal_summary["selection_rule"] = "fixed CAES 24 h configuration"
    summary = pd.concat([summary, focal_summary], ignore_index=True)
    return {
        "configuration": config,
        "common": common,
        "envelope": envelope_config,
        "summary": summary,
    }


def figure5c_wide(summary: pd.DataFrame) -> pd.DataFrame:
    """Return one plotting row per duration for the signed two-channel ledger."""
    metrics = {
        "delta_benefit_eur_kw_year": "total_effect",
        "delta_cycle_return_eur_kw_year": "cycling_channel",
        "delta_coverage_return_eur_kw_year": "coverage_channel",
        "delta_fixed_selection_eur_kw_year": "fixed_selection",
        "decomposition_closure_residual_eur_kw_year": "closure_residual",
    }
    selected = summary[
        summary["selection_rule"].eq(
            "chronology-specific benefit envelope"
        )
        & summary["metric"].isin(metrics)
    ].copy()
    selected["channel"] = selected["metric"].map(metrics)
    value_columns = [
        "country_block_mean",
        "ci95_low",
        "ci95_high",
        "country_year_median",
    ]
    wide = selected.pivot(
        index=["duration_h", "n_countries", "n_country_years"],
        columns="channel",
        values=value_columns,
    )
    wide.columns = [
        f"{channel}_{stat}" for stat, channel in wide.columns
    ]
    wide = wide.reset_index()
    ordered = [
        "duration_h",
        "n_countries",
        "n_country_years",
    ]
    for channel in [
        "total_effect",
        "cycling_channel",
        "coverage_channel",
        "fixed_selection",
        "closure_residual",
    ]:
        ordered.extend(
            [
                f"{channel}_country_block_mean",
                f"{channel}_ci95_low",
                f"{channel}_ci95_high",
                f"{channel}_country_year_median",
            ]
        )
    return wide[ordered].sort_values("duration_h", kind="stable")


def write_report(
    catalogue: pd.DataFrame,
    year_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    decomposition: dict[str, pd.DataFrame],
    central_residual: float,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        shown = frame.copy()
        for column in shown.select_dtypes(include=[np.number]).columns:
            shown[column] = shown[column].map(
                lambda value: ""
                if pd.isna(value)
                else f"{float(value):.4f}"
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

    lines = [
        "# V43 cost-year revaluation and value-channel decomposition",
        "",
        "## Scope and exactness",
        "",
        "- The three annual catalogues contain the source-aware, technology-specific "
        "2025, 2030 and 2035 costs for 53 released configurations. Both the "
        "candidate LDES tail and the additional 4 h LFP comparator are updated.",
        "- The common 4 h LFP base layer is also updated in both arms. It cancels "
        "algebraically in the candidate-minus-comparator estimand, whereas the "
        "additional 4 h LFP tail does not cancel.",
        "- Year-cost revaluation is exact for the solved fixed-power model: annual "
        "fixed costs are constant with respect to hourly dispatch, and VOM, RTE, "
        "self-discharge and charge-power ratios do not vary across the three cost "
        "years. No Gurobi rerun is needed for these three fixed-cost updates.",
        "- The vF=575 cases are frozen-trajectory valuation sensitivities. A "
        "different firm-supply value can change the optimal dispatch objective, "
        "so a fully re-optimized vF=575 claim would require a new Gurobi run.",
        "",
        "## Cost catalogue",
        "",
        f"- Rows: {len(catalogue)} (53 configurations × 3 years).",
        f"- Source: `{COST_SOURCE}`.",
        "",
        "## Strict-valley counts",
        "",
        markdown_table(year_summary),
        "",
        "## Paired valley states",
        "",
        markdown_table(paired_summary),
        "",
        "## Decomposition interpretation",
        "",
        "The signed cycling-return channel equals energy-substitution value plus "
        "the avoided charging-opportunity-cost contribution plus avoided variable "
        "O&M. The coverage-return channel equals the EFC-equivalent firm-supply "
        "gain multiplied by vF. Their sum plus the fixed/technology-selection "
        "term closes exactly to the candidate-minus-additional-LFP benefit.",
        "",
        "A negative equivalent-cycle change does not mechanically imply a "
        "negative *relative* cycling-return contribution. At long durations the "
        "candidate and additional-LFP comparator can both cycle less under "
        "clustering, while the comparator loses more; the relative cycling "
        "channel can therefore become positive. This distinction prevents "
        "absolute physical cycling loss from being conflated with a paired "
        "economic contribution.",
        "",
        f"- Maximum reconstruction residual for the original 2035 central cells: "
        f"{central_residual:.3e} EUR kW^-1 yr^-1.",
        f"- Maximum paired decomposition closure residual: "
        f"{decomposition['configuration']['decomposition_closure_residual_eur_kw_year'].abs().max():.3e} "
        "EUR kW^-1 yr^-1.",
        "",
        "## Selected duration results",
        "",
    ]
    summary = decomposition["summary"]
    selected = summary[
        summary["selection_rule"].eq("chronology-specific benefit envelope")
        & summary["duration_h"].isin([24.0, 32.0, 40.0, 96.0, 168.0])
        & summary["metric"].isin(
            [
                "delta_benefit_eur_kw_year",
                "delta_cycle_return_eur_kw_year",
                "delta_coverage_return_eur_kw_year",
                "delta_fixed_selection_eur_kw_year",
            ]
        )
    ]
    lines.append(markdown_table(selected))
    absolute = summary[
        summary["selection_rule"].eq("chronology-specific benefit envelope")
        & summary["duration_h"].isin([24.0, 96.0, 168.0])
        & summary["metric"].isin(
            [
                "delta_candidate_absolute_cycle_return_eur_kw_year",
                "delta_comparator_absolute_cycle_return_eur_kw_year",
                "delta_candidate_absolute_coverage_return_eur_kw_year",
                "delta_comparator_absolute_coverage_return_eur_kw_year",
            ]
        )
    ]
    lines += [
        "",
        "## Absolute arm changes behind the paired contribution",
        "",
        markdown_table(absolute),
    ]
    lines += [
        "",
        "These files are source data, not manuscript prose. Values should be "
        "rounded in the paper according to statistical precision and displayed "
        "with the country-block interval where an inferential claim is made.",
        "",
    ]
    (HERE / "METHODS_AND_KEY_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    catalogue = build_cost_catalogues()
    cells = pd.read_csv(CELL_SOURCE)
    if set(cells["model_mode"].unique()) != {"fixed"}:
        raise RuntimeError("Expected only fixed-power cells")
    if cells[["country", "year"]].drop_duplicates().shape[0] != 132:
        raise RuntimeError("Expected 132 country-years")

    cases: list[pd.DataFrame] = []
    for year in YEARS:
        cases.append(
            revalue_cells(
                cells,
                catalogue,
                year,
                "central_actual_year_cost",
                CENTRAL_ENERGY_VALUE,
                CENTRAL_FIRM_VALUE,
                CENTRAL_CHARGE_PRICE,
            )
        )
        cases.append(
            revalue_cells(
                cells,
                catalogue,
                year,
                "vf575_actual_year_cost_frozen_trajectory",
                CENTRAL_ENERGY_VALUE,
                575.0,
                CENTRAL_CHARGE_PRICE,
            )
        )
    cases.append(
        revalue_cells(
            cells,
            catalogue,
            2035,
            "figure6_central_m1_vf103p8",
            CENTRAL_ENERGY_VALUE,
            CENTRAL_FIRM_VALUE,
            CENTRAL_CHARGE_PRICE,
            1.0,
        )
    )
    cases.append(
        revalue_cells(
            cells,
            catalogue,
            2035,
            "figure6_conditional_peak_m0p7_vf575_frozen_trajectory",
            CENTRAL_ENERGY_VALUE,
            575.0,
            CENTRAL_CHARGE_PRICE,
            0.7,
        )
    )
    revalued = pd.concat(cases, ignore_index=True)

    # Validate that repricing changes the frozen v42 benefit only through the
    # annual fixed-cost contribution before any envelope selection.
    check = revalued[
        revalued["revaluation_case_id"].eq("central_actual_year_cost")
        & revalued["cost_year"].eq(2035)
    ]
    observed_change = (
        check["mix_benefit_revalued_eur_kw_year"]
        - check["mix_benefit_eur_per_kw_common_power_year"]
    )
    expected_change = (
        check["fixed_cost_contribution_revalued_eur_year"]
        / check["common_base_power_kw"]
        - check["fixed_cost_contribution_eur_per_kw_common_power_year"]
    )
    central_residual = float(
        (
            observed_change
            - expected_change
        )
        .abs()
        .max()
    )
    if central_residual > 2.0e-6:
        raise RuntimeError(
            f"2035 fixed-cost-only revaluation residual is {central_residual}"
        )

    keep = [
        "revaluation_case_id",
        "cost_year",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "config_id",
        "tech_id",
        "duration_h",
        "common_base_power_kw",
        "tail_power_mw",
        "extra_lfp4_tail_power_mw",
        "revaluation_energy_value_eur_mwh",
        "revaluation_firm_value_eur_kw_year",
        "revaluation_charge_price_eur_mwh",
        "ldes_fixed_cost_multiplier",
        "lfp4_fixed_cost_eur_kw_year",
        "candidate_fixed_cost_eur_kw_year_unscaled",
        "candidate_fixed_cost_eur_kw_year",
        "candidate_portfolio_fixed_cost_eur_year",
        "comparator_portfolio_fixed_cost_eur_year",
        "fixed_cost_contribution_revalued_eur_year",
        "cycle_return_contribution_eur_year",
        "coverage_return_contribution_eur_year",
        "mix_benefit_revalued_eur_year",
        "mix_benefit_revalued_eur_kw_year",
    ]
    revalued[keep].to_csv(
        HERE / "year_specific_revaluation_cells.csv.gz",
        index=False,
        compression="gzip",
    )
    envelope = choose_envelope(revalued)
    envelope.to_csv(
        HERE / "year_specific_envelope.csv.gz",
        index=False,
        compression="gzip",
    )
    records = valley_records(envelope)
    records.to_csv(HERE / "year_specific_valley_records.csv", index=False)
    summary, paired = summarize_valleys(records)
    year_duration = summarize_year_duration(envelope)
    summary.to_csv(HERE / "year_specific_valley_summary.csv", index=False)
    paired.to_csv(
        HERE / "year_specific_paired_valley_summary.csv", index=False
    )
    year_duration.to_csv(
        HERE / "year_specific_duration_summary.csv", index=False
    )

    decomposition = decompose(cells)
    decomposition["configuration"].to_csv(
        HERE / "configuration_channel_decomposition_country_year.csv.gz",
        index=False,
        compression="gzip",
    )
    decomposition["common"].to_csv(
        HERE / "common_config_channel_decomposition_country_year.csv.gz",
        index=False,
        compression="gzip",
    )
    decomposition["envelope"].to_csv(
        HERE / "envelope_channel_decomposition_country_year.csv.gz",
        index=False,
        compression="gzip",
    )
    decomposition["summary"].to_csv(
        HERE / "duration_channel_decomposition_summary.csv", index=False
    )
    fig5c = figure5c_wide(decomposition["summary"])
    fig5c.to_csv(HERE / "figure5c_channel_decomposition_wide.csv", index=False)

    write_report(
        catalogue, summary, paired, decomposition, central_residual
    )
    manifest = {
        "analysis_id": "v43_cost_year_and_channel_decomposition",
        "inputs": {
            str(COST_SOURCE): sha256(COST_SOURCE),
            str(CELL_SOURCE): sha256(CELL_SOURCE),
        },
        "cost_years": list(YEARS),
        "cost_rows": int(len(catalogue)),
        "country_years": int(
            cells[["country", "year"]].drop_duplicates().shape[0]
        ),
        "exactness": {
            "actual_year_costs": (
                "exact fixed-power revaluation; no Gurobi rerun required"
            ),
            "vF_575": (
                "frozen-trajectory sensitivity; Gurobi rerun required for "
                "fully re-optimized dispatch"
            ),
        },
        "central_2035_max_reconstruction_residual_eur_kw_year": (
            central_residual
        ),
        "paired_decomposition_max_closure_residual_eur_kw_year": float(
            decomposition["configuration"][
                "decomposition_closure_residual_eur_kw_year"
            ]
            .abs()
            .max()
        ),
        "strict_valley_definition": {
            "duration_domain_h": [VALLEY_MIN_H, VALLEY_MAX_H],
            "minimum_adjacent_negative_nodes": 2,
            "positive_nodes_required_on_both_sides": True,
            "zero_tolerance_eur_kw_year": ZERO_TOLERANCE,
        },
        "bootstrap": {
            "independent_block": "country",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
    }
    (HERE / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Freeze the compact manuscript-facing evidence under source_data.  The
    # larger country-year cell files remain in this analysis directory.
    frozen_frames = {
        "v43_cost_catalogue_2025_2035_source_aware_eur.csv": catalogue,
        "v43_year_specific_valley_summary.csv": summary,
        "v43_year_specific_paired_valley_summary.csv": paired,
        "v43_year_specific_duration_summary.csv": year_duration,
        "v43_duration_channel_decomposition_summary.csv": decomposition[
            "summary"
        ],
        "v43_figure5c_channel_decomposition_wide.csv": fig5c,
    }
    frozen_paths: list[Path] = []
    for filename, frame in frozen_frames.items():
        path = FROZEN_SOURCE_DIR / filename
        frame.to_csv(path, index=False)
        frozen_paths.append(path)
    records_path = FROZEN_SOURCE_DIR / "v43_year_specific_valley_records.csv"
    records.to_csv(records_path, index=False)
    frozen_paths.append(records_path)
    notes_path = FROZEN_SOURCE_DIR / "V43_COST_REVALUATION_SCOPE.md"
    notes_path.write_text(
        "\n".join(
            [
                "# V43 cost revaluation scope",
                "",
                "The 2025, 2030 and 2035 central results update both the "
                "technology-specific LDES tail cost and the 4 h LFP "
                "comparator cost. They are exact revaluations of the solved "
                "fixed-power model because annual fixed costs do not alter "
                "hourly feasibility or marginal dispatch and VOM is unchanged "
                "across cost years.",
                "",
                "The vF=575 results are frozen-trajectory valuation "
                "sensitivities, not re-optimized dispatch outcomes. Main-text "
                "evidence should use the three central cost-year results as "
                "the exact fixed-scale revaluation. A fully optimized vF=575 "
                "claim requires a new Gurobi run.",
                "",
                "Figure 5c should read "
                "`v43_figure5c_channel_decomposition_wide.csv`. Cycling and "
                "coverage channels are signed contributions; the fixed "
                "selection and closure-residual columns are shown explicitly.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    frozen_paths.append(notes_path)
    frozen_manifest = {
        "analysis_id": "v43_cost_and_decomposition_frozen_evidence",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in frozen_paths
        },
        "scope_note": (
            "Central 2025/2030/2035 values are exact fixed-power cost "
            "revaluations. vF=575 values are frozen-trajectory sensitivities."
        ),
    }
    manifest_path = (
        FROZEN_SOURCE_DIR
        / "v43_cost_and_decomposition_freeze_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(frozen_manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
