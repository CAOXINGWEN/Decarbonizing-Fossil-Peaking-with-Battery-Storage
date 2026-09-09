#!/usr/bin/env python3
"""Audit chronology-induced storage decision flips under central valuations.

This is a post-processing analysis of the frozen v43 fixed-power solutions.
It does not modify v43 and does not rerun dispatch.  The three cost years
update both the candidate-tail and additional-4 h-LFP fixed costs exactly,
using the technology-specific 2025, 2030 and 2035 catalogues.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COST_DIR = ROOT / "cost_and_decomposition"
CELLS_PATH = COST_DIR / "year_specific_revaluation_cells.csv.gz"
COST_PATH = COST_DIR / "cost_catalogue_2025_2035_long.csv"
RUN_MANIFEST_PATH = COST_DIR / "run_manifest.json"
FIXED_RUN_CONTRACT_PATH = (
    ROOT / "portfolio_model/full_132cy_fixed_s20_central/run_contract.json"
)
FIXED_QA_PATH = ROOT / "portfolio_model/full_132cy_fixed_s20_central/qa_summary.json"
OPTIMIZED_CENTRAL_DIR = ROOT / "portfolio_model/full_132cy_optimized_s20_central"

VALUE = "mix_benefit_revalued_eur_kw_year"
EPS = 1e-6
EXPECTED_YEARS = [2025, 2030, 2035]
EXPECTED_CHRONOLOGIES = ["dispersed", "clustered"]
EXPECTED_COUNTRY_YEARS = 132
EXPECTED_CONFIGS = 49
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260804


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state(value: float) -> str:
    if value > EPS:
        return "candidate_tail"
    if value < -EPS:
        return "additional_lfp4"
    return "tie"


def flip_direction(dispersed_state: str, clustered_state: str) -> str:
    if dispersed_state == clustered_state:
        return "no_flip"
    return f"{dispersed_state}_to_{clustered_state}"


def qstats(series: pd.Series) -> Dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan}
    return {
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def attach_config_spans(frame: pd.DataFrame, spans: pd.DataFrame, config_col: str, prefix: str) -> pd.DataFrame:
    rename = {
        "config_id": config_col,
        "candidate_fixed_cost_span_2025_2035_eur_kw_year": f"{prefix}candidate_fixed_cost_span_2025_2035_eur_kw_year",
        "relative_to_lfp_fixed_cost_span_2025_2035_eur_kw_year": f"{prefix}relative_to_lfp_fixed_cost_span_2025_2035_eur_kw_year",
        "relative_to_lfp_fixed_cost_change_2025_to_2035_eur_kw_year": f"{prefix}relative_to_lfp_fixed_cost_change_2025_to_2035_eur_kw_year",
        "absolute_relative_to_lfp_fixed_cost_movement_2025_to_2035_eur_kw_year": f"{prefix}absolute_relative_to_lfp_fixed_cost_movement_2025_to_2035_eur_kw_year",
    }
    use = spans[list(rename)].rename(columns=rename)
    return frame.merge(use, on=config_col, how="left", validate="many_to_one")


def build_cost_spans(costs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fixed = "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"
    required = {"learning_year", "config_id", "tech_id", "duration_h", fixed}
    missing = required - set(costs.columns)
    assert not missing, f"Missing catalogue columns: {sorted(missing)}"
    assert sorted(costs["learning_year"].unique().tolist()) == EXPECTED_YEARS
    assert costs.duplicated(["learning_year", "config_id"]).sum() == 0

    lfp = costs.loc[costs["config_id"].eq("li_lfp_4h"), ["learning_year", fixed]].set_index("learning_year")[fixed]
    assert lfp.index.tolist() == EXPECTED_YEARS

    rows = []
    for config_id, group in costs.groupby("config_id", sort=True):
        year_cost = group.set_index("learning_year")[fixed].reindex(EXPECTED_YEARS)
        assert year_cost.notna().all(), config_id
        relative = lfp - year_cost
        first = group.iloc[0]
        rows.append(
            {
                "config_id": config_id,
                "tech_id": first["tech_id"],
                "duration_h": float(first["duration_h"]),
                "fixed_cost_2025_eur_kw_year": float(year_cost.loc[2025]),
                "fixed_cost_2030_eur_kw_year": float(year_cost.loc[2030]),
                "fixed_cost_2035_eur_kw_year": float(year_cost.loc[2035]),
                "candidate_fixed_cost_span_2025_2035_eur_kw_year": float(year_cost.max() - year_cost.min()),
                "relative_fixed_advantage_vs_lfp_2025_eur_kw_year": float(relative.loc[2025]),
                "relative_fixed_advantage_vs_lfp_2030_eur_kw_year": float(relative.loc[2030]),
                "relative_fixed_advantage_vs_lfp_2035_eur_kw_year": float(relative.loc[2035]),
                "relative_to_lfp_fixed_cost_span_2025_2035_eur_kw_year": float(relative.max() - relative.min()),
                "relative_to_lfp_fixed_cost_change_2025_to_2035_eur_kw_year": float(relative.loc[2035] - relative.loc[2025]),
                "absolute_relative_to_lfp_fixed_cost_movement_2025_to_2035_eur_kw_year": float(abs(relative.loc[2035] - relative.loc[2025])),
            }
        )
    spans = pd.DataFrame(rows)
    cost_pivot = costs.pivot(index="config_id", columns="learning_year", values=fixed).sort_index()
    return spans, cost_pivot


def pair_cost_span(cost_pivot: pd.DataFrame, first: str, second: str) -> float:
    # The top-minus-runner benefit receives fixed-cost term C_runner-C_top.
    difference = cost_pivot.loc[second, EXPECTED_YEARS] - cost_pivot.loc[first, EXPECTED_YEARS]
    return float(difference.max() - difference.min())


def build_config_level(cells: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    identity = ["cost_year", "country", "year", "country_year_id", "config_id", "tech_id", "duration_h"]
    wide = cells.pivot(index=identity, columns="chronology", values=VALUE).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "dispersed": "benefit_dispersed_eur_kw_year",
            "clustered": "benefit_clustered_eur_kw_year",
        }
    )
    wide["chronology_effect_clustered_minus_dispersed_eur_kw_year"] = (
        wide["benefit_clustered_eur_kw_year"] - wide["benefit_dispersed_eur_kw_year"]
    )
    wide["state_dispersed"] = wide["benefit_dispersed_eur_kw_year"].map(state)
    wide["state_clustered"] = wide["benefit_clustered_eur_kw_year"].map(state)
    wide["sign_flip"] = wide["state_dispersed"].ne(wide["state_clustered"])
    wide["flip_direction"] = [
        flip_direction(d, c) for d, c in zip(wide["state_dispersed"], wide["state_clustered"])
    ]
    wide["distance_to_zero_dispersed_eur_kw_year"] = wide["benefit_dispersed_eur_kw_year"].abs()
    wide["distance_to_zero_clustered_eur_kw_year"] = wide["benefit_clustered_eur_kw_year"].abs()
    wide["nearest_distance_to_zero_eur_kw_year"] = wide[
        ["distance_to_zero_dispersed_eur_kw_year", "distance_to_zero_clustered_eur_kw_year"]
    ].min(axis=1)
    wide = attach_config_spans(wide, spans, "config_id", "")
    wide["nearest_boundary_within_relative_cost_path_span"] = (
        wide["nearest_distance_to_zero_eur_kw_year"]
        <= wide["relative_to_lfp_fixed_cost_span_2025_2035_eur_kw_year"] + EPS
    )
    return wide.sort_values(["cost_year", "country_year_id", "duration_h", "config_id"]).reset_index(drop=True)


def chronology_selected_table(
    cells: pd.DataFrame,
    keys: Iterable[str],
    rank_second: bool = False,
) -> pd.DataFrame:
    keys = list(keys)
    ordered = cells.sort_values(keys + [VALUE, "config_id"], ascending=[True] * len(keys) + [False, True]).copy()
    ordered["selection_rank"] = ordered.groupby(keys).cumcount() + 1
    limit = 2 if rank_second else 1
    return ordered.loc[ordered["selection_rank"].le(limit)].copy()


def build_duration_envelope(cells: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    keys = ["cost_year", "country", "year", "country_year_id", "chronology", "duration_h"]
    selected = chronology_selected_table(cells, keys)
    keep = keys + ["config_id", "tech_id", VALUE]
    dispersed = selected.loc[selected["chronology"].eq("dispersed"), keep].drop(columns="chronology")
    clustered = selected.loc[selected["chronology"].eq("clustered"), keep].drop(columns="chronology")
    ren_d = {
        "config_id": "best_config_dispersed",
        "tech_id": "best_tech_dispersed",
        VALUE: "best_benefit_dispersed_eur_kw_year",
    }
    ren_c = {
        "config_id": "best_config_clustered",
        "tech_id": "best_tech_clustered",
        VALUE: "best_benefit_clustered_eur_kw_year",
    }
    join_keys = ["cost_year", "country", "year", "country_year_id", "duration_h"]
    out = dispersed.rename(columns=ren_d).merge(
        clustered.rename(columns=ren_c), on=join_keys, validate="one_to_one"
    )
    out["chronology_effect_clustered_minus_dispersed_eur_kw_year"] = (
        out["best_benefit_clustered_eur_kw_year"] - out["best_benefit_dispersed_eur_kw_year"]
    )
    out["state_dispersed"] = out["best_benefit_dispersed_eur_kw_year"].map(state)
    out["state_clustered"] = out["best_benefit_clustered_eur_kw_year"].map(state)
    out["sign_flip"] = out["state_dispersed"].ne(out["state_clustered"])
    out["flip_direction"] = [
        flip_direction(d, c) for d, c in zip(out["state_dispersed"], out["state_clustered"])
    ]
    out["within_duration_config_identity_flip"] = out["best_config_dispersed"].ne(out["best_config_clustered"])
    out["within_duration_technology_identity_flip"] = out["best_tech_dispersed"].ne(out["best_tech_clustered"])
    out["distance_to_zero_dispersed_eur_kw_year"] = out["best_benefit_dispersed_eur_kw_year"].abs()
    out["distance_to_zero_clustered_eur_kw_year"] = out["best_benefit_clustered_eur_kw_year"].abs()
    out["nearest_distance_to_zero_eur_kw_year"] = out[
        ["distance_to_zero_dispersed_eur_kw_year", "distance_to_zero_clustered_eur_kw_year"]
    ].min(axis=1)
    out = attach_config_spans(out, spans, "best_config_dispersed", "dispersed_")
    out = attach_config_spans(out, spans, "best_config_clustered", "clustered_")
    out["maximum_selected_relative_cost_path_span_eur_kw_year"] = out[
        [
            "dispersed_relative_to_lfp_fixed_cost_span_2025_2035_eur_kw_year",
            "clustered_relative_to_lfp_fixed_cost_span_2025_2035_eur_kw_year",
        ]
    ].max(axis=1)
    out["nearest_boundary_within_relative_cost_path_span"] = (
        out["nearest_distance_to_zero_eur_kw_year"]
        <= out["maximum_selected_relative_cost_path_span_eur_kw_year"] + EPS
    )
    return out.sort_values(["cost_year", "country_year_id", "duration_h"]).reset_index(drop=True)


def build_catalogue_choice(
    cells: pd.DataFrame,
    spans: pd.DataFrame,
    cost_pivot: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["cost_year", "country", "year", "country_year_id", "chronology"]
    ranked = chronology_selected_table(cells, keys, rank_second=True)
    top = ranked.loc[ranked["selection_rank"].eq(1)].copy()
    runner = ranked.loc[ranked["selection_rank"].eq(2)].copy()
    merge_keys = keys
    top = top[merge_keys + ["config_id", "tech_id", "duration_h", VALUE]].rename(
        columns={
            "config_id": "top_config",
            "tech_id": "top_tech",
            "duration_h": "top_duration_h",
            VALUE: "top_benefit_eur_kw_year",
        }
    )
    runner = runner[merge_keys + ["config_id", "tech_id", "duration_h", VALUE]].rename(
        columns={
            "config_id": "runner_config",
            "tech_id": "runner_tech",
            "duration_h": "runner_duration_h",
            VALUE: "runner_benefit_eur_kw_year",
        }
    )
    selected = top.merge(runner, on=merge_keys, validate="one_to_one")
    selected["top_runner_gap_eur_kw_year"] = (
        selected["top_benefit_eur_kw_year"] - selected["runner_benefit_eur_kw_year"]
    )
    selected["top_runner_cost_path_span_eur_kw_year"] = [
        pair_cost_span(cost_pivot, first, second)
        for first, second in zip(selected["top_config"], selected["runner_config"])
    ]
    selected["top_runner_gap_within_pair_cost_path_span"] = (
        selected["top_runner_gap_eur_kw_year"]
        <= selected["top_runner_cost_path_span_eur_kw_year"] + EPS
    )
    selected = attach_config_spans(selected, spans, "top_config", "top_")
    selected["procurement_state"] = selected["top_benefit_eur_kw_year"].map(state)
    selected["purchased_identity"] = np.where(
        selected["procurement_state"].eq("candidate_tail"),
        selected["top_config"],
        np.where(selected["procurement_state"].eq("tie"), "tie", "additional_lfp4"),
    )

    id_cols = ["cost_year", "country", "year", "country_year_id"]
    measures = [
        "top_config",
        "top_tech",
        "top_duration_h",
        "top_benefit_eur_kw_year",
        "runner_config",
        "runner_tech",
        "runner_duration_h",
        "runner_benefit_eur_kw_year",
        "top_runner_gap_eur_kw_year",
        "top_runner_cost_path_span_eur_kw_year",
        "top_runner_gap_within_pair_cost_path_span",
        "top_candidate_fixed_cost_span_2025_2035_eur_kw_year",
        "top_relative_to_lfp_fixed_cost_span_2025_2035_eur_kw_year",
        "top_relative_to_lfp_fixed_cost_change_2025_to_2035_eur_kw_year",
        "top_absolute_relative_to_lfp_fixed_cost_movement_2025_to_2035_eur_kw_year",
        "procurement_state",
        "purchased_identity",
    ]
    dispersed = selected.loc[selected["chronology"].eq("dispersed"), id_cols + measures].rename(
        columns={c: f"{c}_dispersed" for c in measures}
    )
    clustered = selected.loc[selected["chronology"].eq("clustered"), id_cols + measures].rename(
        columns={c: f"{c}_clustered" for c in measures}
    )
    out = dispersed.merge(clustered, on=id_cols, validate="one_to_one")
    out["best_benefit_change_clustered_minus_dispersed_eur_kw_year"] = (
        out["top_benefit_eur_kw_year_clustered"] - out["top_benefit_eur_kw_year_dispersed"]
    )
    out["catalogue_config_identity_flip"] = out["top_config_dispersed"].ne(out["top_config_clustered"])
    out["catalogue_technology_identity_flip"] = out["top_tech_dispersed"].ne(out["top_tech_clustered"])
    out["catalogue_duration_identity_flip"] = out["top_duration_h_dispersed"].ne(out["top_duration_h_clustered"])
    out["procurement_identity_flip"] = out["procurement_state_dispersed"].ne(out["procurement_state_clustered"])
    out["procurement_flip_direction"] = [
        flip_direction(d, c)
        for d, c in zip(out["procurement_state_dispersed"], out["procurement_state_clustered"])
    ]
    out["distance_to_lfp_boundary_dispersed_eur_kw_year"] = out["top_benefit_eur_kw_year_dispersed"].abs()
    out["distance_to_lfp_boundary_clustered_eur_kw_year"] = out["top_benefit_eur_kw_year_clustered"].abs()
    out["nearest_lfp_boundary_distance_eur_kw_year"] = out[
        ["distance_to_lfp_boundary_dispersed_eur_kw_year", "distance_to_lfp_boundary_clustered_eur_kw_year"]
    ].min(axis=1)
    out["nearest_catalogue_identity_margin_eur_kw_year"] = out[
        ["top_runner_gap_eur_kw_year_dispersed", "top_runner_gap_eur_kw_year_clustered"]
    ].min(axis=1)
    out["maximum_selected_relative_cost_path_span_eur_kw_year"] = out[
        [
            "top_relative_to_lfp_fixed_cost_span_2025_2035_eur_kw_year_dispersed",
            "top_relative_to_lfp_fixed_cost_span_2025_2035_eur_kw_year_clustered",
        ]
    ].max(axis=1)
    out["nearest_lfp_boundary_within_relative_cost_path_span"] = (
        out["nearest_lfp_boundary_distance_eur_kw_year"]
        <= out["maximum_selected_relative_cost_path_span_eur_kw_year"] + EPS
    )
    out["nearest_catalogue_identity_margin_within_pair_cost_path_span"] = (
        out["top_runner_gap_within_pair_cost_path_span_dispersed"]
        | out["top_runner_gap_within_pair_cost_path_span_clustered"]
    )
    out["maximum_selected_relative_fixed_cost_movement_2025_to_2035_eur_kw_year"] = out[
        [
            "top_absolute_relative_to_lfp_fixed_cost_movement_2025_to_2035_eur_kw_year_dispersed",
            "top_absolute_relative_to_lfp_fixed_cost_movement_2025_to_2035_eur_kw_year_clustered",
        ]
    ].max(axis=1)
    out["absolute_best_catalogue_chronology_effect_eur_kw_year"] = out[
        "best_benefit_change_clustered_minus_dispersed_eur_kw_year"
    ].abs()
    out["chronology_effect_exceeds_2025_2035_relative_fixed_cost_movement"] = (
        out["absolute_best_catalogue_chronology_effect_eur_kw_year"]
        > out["maximum_selected_relative_fixed_cost_movement_2025_to_2035_eur_kw_year"] + EPS
    )
    return out.sort_values(["cost_year", "country_year_id"]).reset_index(drop=True)


def build_summary(
    config_level: pd.DataFrame,
    duration: pd.DataFrame,
    catalogue: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cost_year in EXPECTED_YEARS:
        cf = config_level.loc[config_level["cost_year"].eq(cost_year)]
        du = duration.loc[duration["cost_year"].eq(cost_year)]
        ca = catalogue.loc[catalogue["cost_year"].eq(cost_year)]
        nonflip_ca = ca.loc[~ca["procurement_identity_flip"]]
        boundary_stats = qstats(nonflip_ca["nearest_lfp_boundary_distance_eur_kw_year"])
        identity_stats = qstats(ca.loc[~ca["catalogue_config_identity_flip"], "nearest_catalogue_identity_margin_eur_kw_year"])
        rows.append(
            {
                "cost_year": cost_year,
                "n_country_years": len(ca),
                "n_config_nodes": len(cf),
                "config_node_sign_flips": int(cf["sign_flip"].sum()),
                "config_node_sign_flip_rate": float(cf["sign_flip"].mean()),
                "n_duration_envelope_nodes": len(du),
                "duration_node_sign_flips": int(du["sign_flip"].sum()),
                "duration_node_sign_flip_rate": float(du["sign_flip"].mean()),
                "catalogue_config_identity_flips": int(ca["catalogue_config_identity_flip"].sum()),
                "catalogue_config_identity_flip_rate": float(ca["catalogue_config_identity_flip"].mean()),
                "catalogue_technology_identity_flips": int(ca["catalogue_technology_identity_flip"].sum()),
                "catalogue_technology_identity_flip_rate": float(ca["catalogue_technology_identity_flip"].mean()),
                "catalogue_duration_identity_flips": int(ca["catalogue_duration_identity_flip"].sum()),
                "catalogue_duration_identity_flip_rate": float(ca["catalogue_duration_identity_flip"].mean()),
                "procurement_identity_flips": int(ca["procurement_identity_flip"].sum()),
                "procurement_identity_flip_rate": float(ca["procurement_identity_flip"].mean()),
                "candidate_to_lfp_procurement_flips": int(
                    ca["procurement_flip_direction"].eq("candidate_tail_to_additional_lfp4").sum()
                ),
                "lfp_to_candidate_procurement_flips": int(
                    ca["procurement_flip_direction"].eq("additional_lfp4_to_candidate_tail").sum()
                ),
                "nonflip_country_years": len(nonflip_ca),
                "nonflip_boundary_margin_median_eur_kw_year": boundary_stats["median"],
                "nonflip_boundary_margin_q25_eur_kw_year": boundary_stats["q25"],
                "nonflip_boundary_margin_q75_eur_kw_year": boundary_stats["q75"],
                "nonflip_boundaries_within_relative_cost_path_span": int(
                    nonflip_ca["nearest_lfp_boundary_within_relative_cost_path_span"].sum()
                ),
                "nonflip_boundaries_within_relative_cost_path_span_rate": float(
                    nonflip_ca["nearest_lfp_boundary_within_relative_cost_path_span"].mean()
                ),
                "stable_catalogue_identity_margin_median_eur_kw_year": identity_stats["median"],
                "stable_catalogue_identity_margin_q25_eur_kw_year": identity_stats["q25"],
                "stable_catalogue_identity_margin_q75_eur_kw_year": identity_stats["q75"],
                "catalogue_identity_margin_within_pair_cost_path_span": int(
                    ca["nearest_catalogue_identity_margin_within_pair_cost_path_span"].sum()
                ),
                "catalogue_identity_margin_within_pair_cost_path_span_rate": float(
                    ca["nearest_catalogue_identity_margin_within_pair_cost_path_span"].mean()
                ),
                "chronology_effect_exceeds_2025_2035_relative_fixed_cost_movement": int(
                    ca["chronology_effect_exceeds_2025_2035_relative_fixed_cost_movement"].sum()
                ),
                "chronology_effect_exceeds_2025_2035_relative_fixed_cost_movement_rate": float(
                    ca["chronology_effect_exceeds_2025_2035_relative_fixed_cost_movement"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_country_block_bootstrap(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Equal-country mean and percentile interval from country-block resampling.

    Country-years are repeated observations within countries.  Each endpoint is
    first averaged within country; the 15 country means are then sampled with
    replacement.  The pooled 132-country-year fraction is retained separately
    as a descriptive count/rate.
    """
    work = catalogue.copy()
    work["candidate_selected_dispersed"] = work["procurement_state_dispersed"].eq("candidate_tail")
    work["candidate_selected_clustered"] = work["procurement_state_clustered"].eq("candidate_tail")
    endpoints = [
        "procurement_identity_flip",
        "catalogue_config_identity_flip",
        "catalogue_duration_identity_flip",
        "candidate_selected_dispersed",
        "candidate_selected_clustered",
        "nearest_lfp_boundary_within_relative_cost_path_span",
        "chronology_effect_exceeds_2025_2035_relative_fixed_cost_movement",
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for cost_year in EXPECTED_YEARS:
        year_data = work.loc[work["cost_year"].eq(cost_year)]
        for endpoint in endpoints:
            country_means = year_data.groupby("country", sort=True)[endpoint].mean().to_numpy(dtype=float)
            draws = rng.choice(country_means, size=(BOOTSTRAP_DRAWS, len(country_means)), replace=True).mean(axis=1)
            rows.append(
                {
                    "cost_year": cost_year,
                    "endpoint": endpoint,
                    "pooled_country_year_count": int(year_data[endpoint].sum()),
                    "pooled_country_year_n": int(len(year_data)),
                    "pooled_country_year_rate": float(year_data[endpoint].mean()),
                    "equal_country_mean_rate": float(country_means.mean()),
                    "country_block_ci95_low": float(np.quantile(draws, 0.025)),
                    "country_block_ci95_high": float(np.quantile(draws, 0.975)),
                    "n_countries": int(len(country_means)),
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    cells_all = pd.read_csv(CELLS_PATH)
    costs = pd.read_csv(COST_PATH)
    cells = cells_all.loc[cells_all["revaluation_case_id"].eq("central_actual_year_cost")].copy()

    required = {
        "cost_year", "country", "year", "country_year_id", "chronology",
        "config_id", "tech_id", "duration_h", VALUE,
    }
    missing = required - set(cells.columns)
    assert not missing, f"Missing revaluation columns: {sorted(missing)}"
    assert sorted(cells["cost_year"].unique().tolist()) == EXPECTED_YEARS
    assert set(cells["chronology"].unique().tolist()) == set(EXPECTED_CHRONOLOGIES)
    assert cells["country_year_id"].nunique() == EXPECTED_COUNTRY_YEARS
    assert cells["config_id"].nunique() == EXPECTED_CONFIGS
    assert len(cells) == len(EXPECTED_YEARS) * EXPECTED_COUNTRY_YEARS * len(EXPECTED_CHRONOLOGIES) * EXPECTED_CONFIGS
    assert cells.duplicated(["cost_year", "country_year_id", "chronology", "config_id"]).sum() == 0
    assert np.isfinite(cells[VALUE]).all()

    spans, cost_pivot = build_cost_spans(costs)
    assert set(cells["config_id"]) <= set(spans["config_id"])

    config_level = build_config_level(cells, spans)
    duration = build_duration_envelope(cells, spans)
    catalogue = build_catalogue_choice(cells, spans, cost_pivot)
    summary = build_summary(config_level, duration, catalogue)
    block_bootstrap = build_country_block_bootstrap(catalogue)

    # Frozen-result gates. These are deliberately explicit because the tables
    # will be consumed by manuscript prose and figures.
    assert summary["procurement_identity_flips"].tolist() == [1, 1, 2]
    assert summary["catalogue_config_identity_flips"].tolist() == [13, 9, 7]
    assert summary["catalogue_technology_identity_flips"].tolist() == [0, 0, 0]
    assert summary["duration_node_sign_flips"].tolist() == [1, 2, 4]
    assert summary[
        "chronology_effect_exceeds_2025_2035_relative_fixed_cost_movement"
    ].tolist() == [68, 68, 68]
    actual_procurement_flips = catalogue.loc[catalogue["procurement_identity_flip"]]
    assert actual_procurement_flips["country_year_id"].tolist() == [
        "AT-2024", "AT-2024", "AT-2018", "AT-2024"
    ]
    assert actual_procurement_flips["cost_year"].tolist() == [2025, 2030, 2035, 2035]
    assert actual_procurement_flips["top_config_dispersed"].tolist() == ["caes_24h"] * 4
    assert actual_procurement_flips["top_config_clustered"].tolist() == ["caes_24h"] * 4

    config_path = HERE / "decision_flip_config_nodes.csv.gz"
    duration_path = HERE / "decision_flip_duration_envelopes.csv.gz"
    catalogue_path = HERE / "decision_flip_catalogue_choices.csv"
    summary_path = HERE / "decision_flip_year_summary.csv"
    flip_cases_path = HERE / "decision_flip_observed_cases.csv"
    spans_path = HERE / "cost_path_spans_2025_2035.csv"
    bootstrap_path = HERE / "decision_flip_country_block_bootstrap.csv"

    deterministic_gzip = {"method": "gzip", "compresslevel": 9, "mtime": 0}
    config_level.to_csv(config_path, index=False, compression=deterministic_gzip)
    duration.to_csv(duration_path, index=False, compression=deterministic_gzip)
    catalogue.to_csv(catalogue_path, index=False)
    summary.to_csv(summary_path, index=False)
    spans.to_csv(spans_path, index=False)
    block_bootstrap.to_csv(bootstrap_path, index=False)

    cases = []
    for level, frame, flag in [
        ("config_node", config_level, "sign_flip"),
        ("duration_envelope", duration, "sign_flip"),
        ("catalogue_config_identity", catalogue, "catalogue_config_identity_flip"),
        ("procurement_identity", catalogue, "procurement_identity_flip"),
    ]:
        selected = frame.loc[frame[flag]].copy()
        selected.insert(0, "audit_level", level)
        cases.append(selected)
    pd.concat(cases, ignore_index=True, sort=False).to_csv(flip_cases_path, index=False)

    optimized_outputs = [
        OPTIMIZED_CENTRAL_DIR / "portfolio_arm_results.csv.gz",
        OPTIMIZED_CENTRAL_DIR / "ldes_vs_extra_lfp4_results.csv.gz",
        OPTIMIZED_CENTRAL_DIR / "clustered_vs_dispersed_paired_effects.csv",
    ]
    missing_optimized = [str(path) for path in optimized_outputs if not path.exists()]

    outputs = [
        config_path,
        duration_path,
        catalogue_path,
        summary_path,
        flip_cases_path,
        spans_path,
        bootstrap_path,
    ]
    audit = {
        "analysis_id": "v44_central_cost_decision_flip_audit",
        "analysis_type": "exact post-processing of frozen fixed-power dispatch trajectories",
        "independent_unit": "country-year (132 units nested within 15 countries); node counts are deterministic catalogue evaluations, not independent replicates",
        "central_valuation": {
            "energy_value_eur_mwh": float(cells["revaluation_energy_value_eur_mwh"].iloc[0]),
            "firm_value_eur_kw_year": float(cells["revaluation_firm_value_eur_kw_year"].iloc[0]),
            "charge_price_eur_mwh": float(cells["revaluation_charge_price_eur_mwh"].iloc[0]),
        },
        "cost_years": EXPECTED_YEARS,
        "zero_tolerance_eur_kw_year": EPS,
        "country_block_bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "estimator": "equal-country mean of within-country country-year rates",
        },
        "definitions": {
            "config_node_sign_flip": "same candidate config changes sign relative to additional 4 h LFP between dispersed and clustered chronologies",
            "duration_node_sign_flip": "chronology-specific maximum candidate benefit at the same duration changes sign",
            "catalogue_config_identity_flip": "highest-benefit candidate config differs between chronologies, conditional on selecting a candidate tail even if all candidate benefits are negative",
            "procurement_identity_flip": "maximum candidate benefit crosses zero, switching the preferred tail between any candidate LDES configuration and additional 4 h LFP",
            "distance_to_zero": "absolute maximum candidate-minus-additional-LFP benefit in EUR kW^-1 yr^-1",
            "catalogue_identity_margin": "top candidate benefit minus runner-up candidate benefit in EUR kW^-1 yr^-1",
            "cost_path_span": "deterministic range across the three 2025/2030/2035 annualized fixed-cost points; not a confidence or forecast interval",
        },
        "input_hashes": {
            str(path): sha256(path)
            for path in [CELLS_PATH, COST_PATH, RUN_MANIFEST_PATH, FIXED_RUN_CONTRACT_PATH, FIXED_QA_PATH]
        },
        "qa": {
            "central_rows": int(len(cells)),
            "country_years": int(cells["country_year_id"].nunique()),
            "countries": int(cells["country"].nunique()),
            "candidate_configs": int(cells["config_id"].nunique()),
            "duration_nodes": int(cells["duration_h"].nunique()),
            "duplicate_cell_keys": int(cells.duplicated(["cost_year", "country_year_id", "chronology", "config_id"]).sum()),
            "nonfinite_benefits": int((~np.isfinite(cells[VALUE])).sum()),
            "fixed_run_qa_status": json.loads(FIXED_QA_PATH.read_text())["status"],
        },
        "year_summary": summary.to_dict(orient="records"),
        "minimum_data_gaps": {
            "central_joint_power_duration_choice": (
                "The central optimized-power directory contains a run contract but no result tables; "
                "therefore this audit identifies the best technology-duration configuration at the "
                "fixed common tail-power scale, not a completed central joint power-duration procurement solve."
            ),
            "source_disagreement_uncertainty": (
                "The frozen catalogue supplies one source-aware point per configuration and cost year. "
                "Only the deterministic 2025-2035 path span can be compared with decision margins; "
                "independent source-disagreement or probabilistic cost bands are unavailable."
            ),
            "missing_central_optimized_outputs": missing_optimized,
        },
        "output_hashes": {str(path): sha256(path) for path in outputs},
    }
    audit_path = HERE / "decision_flip_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")

    print(summary.to_string(index=False))
    print(f"\nWrote {len(outputs) + 1} frozen outputs to {HERE}")


if __name__ == "__main__":
    main()
