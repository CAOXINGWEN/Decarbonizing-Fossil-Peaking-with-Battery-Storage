#!/usr/bin/env python3
"""Merge and audit exact bounded-power central-S20 procurement choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V44 = HERE.parent
V43 = V44.parent / "revision_nature_energy_v43"
LONG_CATALOGUE = V43 / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
BASE_REFERENCE = V43 / "portfolio_model" / "full_132cy_fixed_s20_central" / "portfolio_arm_results.csv.gz"
ROOT = HERE / "bounded_s20_highs"
PRIMARY_TOL_EUR = 0.01
# The two-stage LP may return each arm within one cent of its primary optimum;
# two independently solved arms therefore carry a 0.02-EUR numerical floor.
# This is *not* the procurement tie rule.  Manuscript-facing choices use the
# same normalized tolerance as the fixed-power audit, 0.01 EUR kW-1 yr-1,
# converted to an absolute annual amount for each country-year.
SOLVER_PAIR_ABS_FLOOR_EUR = 2.0 * PRIMARY_TOL_EUR
TIE_TOL_PER_KW_YEAR = 0.01
POWER_SHARE_TOL = 1.0e-6
MATERIAL_POWER_SHARE_CHANGE = 0.01
BOOTSTRAP_DRAWS = 10_000
RESIDUAL_TOL = 1.0e-6
# The cost ledger adds terms on the order of 10^8--10^9 EUR.  A 10-micro-euro
# absolute tolerance is still below 10^-13 in relative terms and avoids
# treating floating-point summation order as an accounting failure.
ACCOUNTING_TOL_EUR = 1.0e-5
ACCOUNTING_REL_TOL = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def base_cost(cost_year: int) -> tuple[float, float]:
    catalogue = pd.read_csv(LONG_CATALOGUE)
    row = catalogue[
        catalogue["learning_year"].eq(cost_year)
        & catalogue["config_id"].eq("li_lfp_4h")
    ]
    if len(row) != 1:
        raise RuntimeError("Cost-year LFP4 base row is not unique")
    return (
        float(row.iloc[0]["annual_fixed_cost_eur_source_aware_per_kw_discharge_year"]),
        float(row.iloc[0]["variable_om_eur_source_aware_mwh"]),
    )


def make_base_only(cost_year: int) -> pd.DataFrame:
    frame = pd.read_csv(BASE_REFERENCE)
    frame = frame[frame["arm_type"].eq("base_only")].copy()
    if len(frame) != 264:
        raise RuntimeError(f"Expected 264 base-only chronology rows, found {len(frame)}")
    fixed, vom = base_cost(cost_year)
    frame["learning_year"] = cost_year
    frame["fixed_cost_eur_kw_year"] = 0.0
    frame["vom_eur_mwh"] = vom
    frame["base_fixed_cost_eur"] = 1000.0 * frame["base_power_mw"] * fixed
    frame["tail_fixed_cost_eur"] = 0.0
    frame["tail_vom_cost_eur"] = 0.0
    frame["tail_power_mw"] = 0.0
    frame["tail_power_upper_mw"] = 0.0
    frame["tail_energy_capacity_mwh"] = 0.0
    frame["tail_usable_energy_capacity_mwh"] = 0.0
    frame["storage_cost_eur"] = (
        frame["base_fixed_cost_eur"] + frame["base_vom_cost_eur"]
        + frame["charge_opportunity_cost_eur"]
    )
    frame["system_cost_eur"] = (
        frame["storage_cost_eur"] + frame["residual_energy_cost_eur"]
        + frame["residual_firm_cost_eur"]
    )
    frame["net_benefit_vs_gas_only_eur"] = frame["gas_only_system_cost_eur"] - frame["system_cost_eur"]
    frame["system_cost_eur_per_mwh_task"] = frame["system_cost_eur"] / frame["base_task_mwh"]
    frame["config_id"] = "base_only_lfp4"
    frame["tech_id"] = "none"
    frame["technology_en"] = "No incremental tail asset"
    frame["duration_h"] = 0.0
    frame["tail_power_mode"] = "optimized"
    frame["solver_method"] = "archived_s20_base_only_exact_cost_revaluation"
    frame["secondary_tiebreak_success"] = True
    return frame


def selected_directories(cost_year: int) -> list[Path]:
    all_directories = sorted((ROOT / f"cost_{cost_year}").glob("shard*_of_*"))
    groups: dict[int, list[Path]] = {}
    for directory in all_directories:
        match = re.fullmatch(r"shard\d+_of_(\d+)", directory.name)
        if match:
            groups.setdefault(int(match.group(1)), []).append(directory)
    complete_groups = {
        count: directories for count, directories in groups.items()
        if len(directories) == count
    }
    if complete_groups:
        # Prefer the most highly partitioned complete run and never mix shard
        # families, even if interrupted older runs remain on disk.
        selected_count = max(complete_groups)
        directories = sorted(complete_groups[selected_count])
    else:
        directories = []
    if not directories:
        directories = [ROOT / f"cost_{cost_year}" / f"chunk{chunk}" for chunk in range(1, 5)]
    return directories


def merge_year(cost_year: int) -> pd.DataFrame:
    frames = []
    country_years: set[tuple[str, int]] = set()
    directories = selected_directories(cost_year)
    for chunk, directory in enumerate(directories, start=1):
        qa = json.loads((directory / "qa.json").read_text())
        if qa.get("status") != "pass":
            raise RuntimeError(f"Chunk {chunk} QA did not pass")
        contract = json.loads((directory / "run_contract.json").read_text())
        if int(contract["cost_year"]) != cost_year or int(contract["scenario_id"]) != 20:
            raise RuntimeError(f"{directory.name} has an incompatible model contract")
        if int(contract["candidate_count"]) != 49 or contract["chronologies"] != ["dispersed", "clustered"]:
            raise RuntimeError(f"{directory.name} has an incompatible candidate/chronology contract")
        pairs = {(str(country), int(year)) for country, year in contract["country_years"]}
        if country_years.intersection(pairs):
            raise RuntimeError("Country-year chunks overlap")
        country_years.update(pairs)
        frame = pd.read_csv(directory / "portfolio_arm_results.csv.gz")
        frames.append(frame)
    if len(country_years) != 132:
        raise RuntimeError(f"Expected 132 disjoint country-years, found {len(country_years)}")
    results = pd.concat(frames, ignore_index=True)
    if len(results) != 13200:
        raise RuntimeError(f"Expected 13,200 optimized tail-arm rows, found {len(results)}")
    results = pd.concat([results, make_base_only(cost_year)], ignore_index=True, sort=False)
    if results.duplicated(["country", "year", "chronology", "arm_type", "config_id"]).any():
        raise RuntimeError("Merged arm identities are not unique")
    return results


def merge_chronology_conservation(cost_year: int) -> pd.DataFrame:
    frames = []
    for directory in selected_directories(cost_year):
        frame = pd.read_csv(directory / "chronology_conservation.csv")
        frames.append(frame)
    conservation = pd.concat(frames, ignore_index=True)
    keys = ["country", "year", "chronology"]
    if len(conservation) != 264 or conservation.duplicated(keys).any():
        raise RuntimeError(
            f"Expected 264 unique chronology-conservation rows, found {len(conservation)}"
        )
    return conservation


def classify_rows(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["tail_power_share"] = frame["tail_power_mw"] / frame["base_power_mw"]
    frame["tail_active"] = frame["tail_power_share"] > POWER_SHARE_TOL
    frame["decision_id"] = np.where(frame["tail_active"], frame["config_id"], "no_tail")
    frame["decision_tech"] = np.where(frame["tail_active"], frame["tech_id"], "none")
    frame["decision_duration_h"] = np.where(frame["tail_active"], frame["duration_h"], 0.0)
    return frame


def add_efc_trajectory_account(results: pd.DataFrame) -> pd.DataFrame:
    """Revalue the already system-cost-optimized trajectories with EFC value.

    This is deliberately a trajectory screen, not a claim that tail power was
    optimized against the EFC account.  The physical paths and power choices
    remain those obtained from the residual-peak system-cost LP.
    """
    frame = results.copy()
    frame["efc_trajectory_net_value_eur"] = (
        frame["energy_value_eur_mwh"] * frame["total_discharge_mwh"]
        + 1000.0 * frame["firm_value_eur_kw_year"] * frame["portfolio_efc_mw"]
        - frame["base_fixed_cost_eur"]
        - frame["tail_fixed_cost_eur"]
        - frame["base_vom_cost_eur"]
        - frame["tail_vom_cost_eur"]
        - frame["charge_price_eur_mwh"] * frame["total_charge_mwh"]
    )
    return frame


def accounting_residuals(results: pd.DataFrame) -> dict[str, float]:
    storage_rebuilt = (
        results["base_fixed_cost_eur"] + results["tail_fixed_cost_eur"]
        + results["base_vom_cost_eur"] + results["tail_vom_cost_eur"]
        + results["charge_opportunity_cost_eur"]
    )
    system_rebuilt = (
        storage_rebuilt + results["residual_energy_cost_eur"]
        + results["residual_firm_cost_eur"]
    )
    benefit_rebuilt = results["gas_only_system_cost_eur"] - results["system_cost_eur"]
    efc_energy = results["energy_value_eur_mwh"] * results["total_discharge_mwh"]
    efc_firm = 1000.0 * results["firm_value_eur_kw_year"] * results["portfolio_efc_mw"]
    efc_charge = results["charge_price_eur_mwh"] * results["total_charge_mwh"]
    efc_rebuilt = (
        efc_energy + efc_firm
        - results["base_fixed_cost_eur"] - results["tail_fixed_cost_eur"]
        - results["base_vom_cost_eur"] - results["tail_vom_cost_eur"]
        - efc_charge
    )
    storage_error = np.abs(results["storage_cost_eur"] - storage_rebuilt)
    system_error = np.abs(results["system_cost_eur"] - system_rebuilt)
    benefit_error = np.abs(results["net_benefit_vs_gas_only_eur"] - benefit_rebuilt)
    efc_error = np.abs(results["efc_trajectory_net_value_eur"] - efc_rebuilt)
    storage_scale = np.maximum(np.abs(storage_rebuilt), 1.0)
    system_scale = np.maximum(np.abs(system_rebuilt), 1.0)
    benefit_scale = np.maximum.reduce([
        np.abs(results["gas_only_system_cost_eur"].to_numpy()),
        np.abs(results["system_cost_eur"].to_numpy()),
        np.ones(len(results)),
    ])
    efc_scale = np.maximum(
        np.abs(efc_energy) + np.abs(efc_firm)
        + np.abs(results["base_fixed_cost_eur"])
        + np.abs(results["tail_fixed_cost_eur"])
        + np.abs(results["base_vom_cost_eur"])
        + np.abs(results["tail_vom_cost_eur"])
        + np.abs(efc_charge),
        1.0,
    )
    return {
        "maximum_storage_cost_accounting_residual_eur": float(
            storage_error.max()
        ),
        "maximum_system_cost_accounting_residual_eur": float(
            system_error.max()
        ),
        "maximum_net_benefit_accounting_residual_eur": float(
            benefit_error.max()
        ),
        "maximum_efc_trajectory_accounting_residual_eur": float(
            efc_error.max()
        ),
        "maximum_storage_cost_accounting_relative_residual": float(
            (storage_error / storage_scale).max()
        ),
        "maximum_system_cost_accounting_relative_residual": float(
            (system_error / system_scale).max()
        ),
        "maximum_net_benefit_accounting_relative_residual": float(
            (benefit_error / benefit_scale).max()
        ),
        "maximum_efc_trajectory_accounting_relative_residual": float(
            (efc_error / efc_scale).max()
        ),
    }


def choose_options(
    results: pd.DataFrame,
    account: str,
    criterion: str,
    maximize: bool,
) -> pd.DataFrame:
    keys = ["learning_year", "country", "year", "chronology"]
    rows = []
    for key, group in classify_rows(results).groupby(keys, sort=True):
        # Collapse the many zero-power representations to the explicit no-tail
        # arm before determining the closest genuinely different alternative.
        explicit_base = group[group["arm_type"].eq("base_only")]
        active = group[group["tail_active"]]
        options = pd.concat([explicit_base, active], ignore_index=True)
        options = options.sort_values(
            [criterion, "duration_h", "config_id"],
            ascending=[not maximize, True, True],
            kind="stable",
        )
        optimum = float(options[criterion].max() if maximize else options[criterion].min())
        base_power_mw = float(options["base_power_mw"].iloc[0])
        tie_tolerance_eur = max(
            SOLVER_PAIR_ABS_FLOOR_EUR,
            TIE_TOL_PER_KW_YEAR * 1000.0 * base_power_mw,
        )
        if maximize:
            tie = options[options[criterion] >= optimum - tie_tolerance_eur]
        else:
            tie = options[options[criterion] <= optimum + tie_tolerance_eur]
        base_in_tie = tie[tie["arm_type"].eq("base_only")]
        winner = (base_in_tie.iloc[0] if len(base_in_tie) else tie.iloc[0])
        alternatives = options[options["decision_id"].ne(winner["decision_id"])]
        runner_up = alternatives.iloc[0] if len(alternatives) else None
        if runner_up is None:
            margin = np.nan
        else:
            margin = float(abs(winner[criterion] - runner_up[criterion]))
        decision_ids = sorted({str(value) for value in tie["decision_id"]})
        decision_techs = sorted({str(value) for value in tie["decision_tech"]})
        decision_durations = sorted({float(value) for value in tie["decision_duration_h"]})
        decision_participation = sorted({int(bool(value)) for value in tie["tail_active"]})
        rows.append({
            **dict(zip(keys, key)),
            "selection_account": account,
            "selection_criterion": criterion,
            "selection_is_maximization": maximize,
            "selection_score_eur": float(winner[criterion]),
            "decision_id": str(winner["decision_id"]),
            "decision_tech": str(winner["decision_tech"]),
            "decision_duration_h": float(winner["decision_duration_h"]),
            "tail_active": bool(winner["tail_active"]),
            "tail_power_mw": float(winner["tail_power_mw"]),
            "tail_power_share": float(winner["tail_power_share"]),
            "system_cost_eur": float(winner["system_cost_eur"]),
            "portfolio_efc_mw": float(winner["portfolio_efc_mw"]),
            "unserved_eens_mwh": float(winner["unserved_eens_mwh"]),
            "base_power_mw": base_power_mw,
            "runner_up_id": str(runner_up["decision_id"]) if runner_up is not None else "",
            "runner_up_tech": str(runner_up["decision_tech"]) if runner_up is not None else "",
            "runner_up_duration_h": float(runner_up["decision_duration_h"]) if runner_up is not None else np.nan,
            "runner_up_tail_power_mw": float(runner_up["tail_power_mw"]) if runner_up is not None else np.nan,
            "runner_up_tail_power_share": float(runner_up["tail_power_share"]) if runner_up is not None else np.nan,
            "decision_margin_eur": margin,
            "decision_margin_eur_kw_year": margin / (1000.0 * float(winner["base_power_mw"])) if runner_up is not None else np.nan,
            "tie_option_count": int(len(tie)),
            "tie_tolerance_eur": tie_tolerance_eur,
            "tie_tolerance_eur_kw_year": TIE_TOL_PER_KW_YEAR,
            "tie_decision_set": "|".join(decision_ids),
            "tie_technology_set": "|".join(decision_techs),
            "tie_duration_set": "|".join(f"{value:g}" for value in decision_durations),
            "tie_participation_set": "|".join(str(value) for value in decision_participation),
            "tie_power_share_min": float(tie["tail_power_share"].min()),
            "tie_power_share_max": float(tie["tail_power_share"].max()),
        })
    choices = pd.DataFrame(rows)
    if len(choices) != 264:
        raise RuntimeError(f"Expected 264 chronology choices, found {len(choices)}")
    return choices


def pair_choices(choices: pd.DataFrame) -> pd.DataFrame:
    keys = ["learning_year", "country", "year"]
    values = [
        "selection_account", "selection_criterion", "selection_is_maximization",
        "selection_score_eur", "decision_id", "decision_tech", "decision_duration_h", "tail_active",
        "tail_power_mw", "tail_power_share", "system_cost_eur", "portfolio_efc_mw",
        "unserved_eens_mwh", "runner_up_id", "decision_margin_eur",
        "decision_margin_eur_kw_year", "tie_option_count", "base_power_mw",
        "tie_tolerance_eur", "tie_tolerance_eur_kw_year",
        "tie_decision_set", "tie_technology_set", "tie_duration_set",
        "tie_participation_set", "tie_power_share_min", "tie_power_share_max",
    ]
    dispersed = choices[choices["chronology"].eq("dispersed")][keys + values].copy()
    clustered = choices[choices["chronology"].eq("clustered")][keys + values].copy()
    dispersed = dispersed.rename(columns={column: f"dispersed_{column}" for column in values})
    clustered = clustered.rename(columns={column: f"clustered_{column}" for column in values})
    paired = dispersed.merge(clustered, on=keys, validate="one_to_one")
    def disjoint(left: pd.Series, right: pd.Series) -> pd.Series:
        return pd.Series(
            [set(str(a).split("|")).isdisjoint(set(str(b).split("|"))) for a, b in zip(left, right)],
            index=left.index,
        )

    # Canonical identities remain an audit trail.  A manuscript-facing flip is
    # robust only when the two normalized-tolerance optimal sets are disjoint.
    paired["canonical_identity_change"] = paired["dispersed_decision_id"].ne(paired["clustered_decision_id"])
    paired["canonical_technology_change"] = paired["dispersed_decision_tech"].ne(paired["clustered_decision_tech"])
    paired["canonical_duration_change"] = paired["dispersed_decision_duration_h"].ne(paired["clustered_decision_duration_h"])
    paired["canonical_participation_change"] = paired["dispersed_tail_active"].ne(paired["clustered_tail_active"])
    paired["canonical_power_change"] = (
        paired["clustered_tail_power_share"] - paired["dispersed_tail_power_share"]
    ).abs() > POWER_SHARE_TOL
    paired["identity_flip"] = disjoint(paired["dispersed_tie_decision_set"], paired["clustered_tie_decision_set"])
    paired["technology_flip"] = disjoint(paired["dispersed_tie_technology_set"], paired["clustered_tie_technology_set"])
    paired["duration_flip"] = disjoint(paired["dispersed_tie_duration_set"], paired["clustered_tie_duration_set"])
    paired["participation_flip"] = disjoint(paired["dispersed_tie_participation_set"], paired["clustered_tie_participation_set"])
    paired["tail_to_no_tail"] = (
        paired["dispersed_tie_participation_set"].eq("1")
        & paired["clustered_tie_participation_set"].eq("0")
    )
    paired["no_tail_to_tail"] = (
        paired["dispersed_tie_participation_set"].eq("0")
        & paired["clustered_tie_participation_set"].eq("1")
    )
    paired["delta_tail_power_share"] = paired["clustered_tail_power_share"] - paired["dispersed_tail_power_share"]
    power_gap = np.maximum(
        paired["clustered_tie_power_share_min"] - paired["dispersed_tie_power_share_max"],
        paired["dispersed_tie_power_share_min"] - paired["clustered_tie_power_share_max"],
    )
    paired["minimum_tie_set_power_share_gap"] = power_gap.clip(lower=0.0)
    paired["power_flip"] = paired["minimum_tie_set_power_share_gap"] > POWER_SHARE_TOL
    paired["material_power_change"] = paired["minimum_tie_set_power_share_gap"] > MATERIAL_POWER_SHARE_CHANGE
    paired["any_material_procurement_change"] = paired["identity_flip"] | paired["material_power_change"]
    paired["clustered_minus_dispersed_system_cost_eur"] = paired["clustered_system_cost_eur"] - paired["dispersed_system_cost_eur"]
    paired["clustered_minus_dispersed_selection_score_eur"] = paired["clustered_selection_score_eur"] - paired["dispersed_selection_score_eur"]
    return paired


def ldes_vs_lfp(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["learning_year", "country", "year", "chronology"]
    classified = classify_rows(results)
    comparator = classified[classified["arm_type"].eq("extra_lfp4_comparator")][keys + [
        "system_cost_eur", "tail_power_mw", "tail_power_share"
    ]].rename(columns={
        "system_cost_eur": "lfp_system_cost_eur",
        "tail_power_mw": "lfp_tail_power_mw",
        "tail_power_share": "lfp_tail_power_share",
    })
    candidates = classified[classified["arm_type"].eq("ldes_candidate")].copy()
    candidates = candidates.merge(comparator, on=keys, validate="many_to_one")
    candidates["ldes_benefit_vs_lfp_eur"] = candidates["lfp_system_cost_eur"] - candidates["system_cost_eur"]
    candidates["ldes_benefit_vs_lfp_eur_kw_year"] = candidates["ldes_benefit_vs_lfp_eur"] / (1000.0 * candidates["base_power_mw"])
    pair_keys = ["learning_year", "country", "year", "config_id", "tech_id", "duration_h"]
    dispersed = candidates[candidates["chronology"].eq("dispersed")][pair_keys + ["ldes_benefit_vs_lfp_eur", "ldes_benefit_vs_lfp_eur_kw_year", "tail_power_share"]]
    clustered = candidates[candidates["chronology"].eq("clustered")][pair_keys + ["ldes_benefit_vs_lfp_eur", "ldes_benefit_vs_lfp_eur_kw_year", "tail_power_share"]]
    dispersed = dispersed.rename(columns={column: f"dispersed_{column}" for column in ["ldes_benefit_vs_lfp_eur", "ldes_benefit_vs_lfp_eur_kw_year", "tail_power_share"]})
    clustered = clustered.rename(columns={column: f"clustered_{column}" for column in ["ldes_benefit_vs_lfp_eur", "ldes_benefit_vs_lfp_eur_kw_year", "tail_power_share"]})
    paired = dispersed.merge(clustered, on=pair_keys, validate="one_to_one")
    paired["dispersed_sign"] = np.select(
        [paired["dispersed_ldes_benefit_vs_lfp_eur_kw_year"] > TIE_TOL_PER_KW_YEAR, paired["dispersed_ldes_benefit_vs_lfp_eur_kw_year"] < -TIE_TOL_PER_KW_YEAR],
        [1, -1], default=0,
    )
    paired["clustered_sign"] = np.select(
        [paired["clustered_ldes_benefit_vs_lfp_eur_kw_year"] > TIE_TOL_PER_KW_YEAR, paired["clustered_ldes_benefit_vs_lfp_eur_kw_year"] < -TIE_TOL_PER_KW_YEAR],
        [1, -1], default=0,
    )
    paired["robust_sign_flip"] = paired["dispersed_sign"] * paired["clustered_sign"] == -1
    paired["ldes_to_lfp_flip"] = paired["dispersed_sign"].eq(1) & paired["clustered_sign"].eq(-1)
    paired["lfp_to_ldes_flip"] = paired["dispersed_sign"].eq(-1) & paired["clustered_sign"].eq(1)
    return candidates, paired


def block_bootstrap_rate(frame: pd.DataFrame, column: str, seed: int) -> tuple[float, float, float]:
    countries = np.array(sorted(frame["country"].unique()))
    groups = {country: frame[frame["country"].eq(country)][column].to_numpy(float) for country in countries}
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(countries, size=len(countries), replace=True)
        estimates[draw] = np.concatenate([groups[country] for country in sampled]).mean()
    return float(frame[column].mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarize(
    cost_year: int,
    account: str,
    paired: pd.DataFrame,
    sign_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for index, metric in enumerate([
        "identity_flip", "technology_flip", "duration_flip", "participation_flip",
        "tail_to_no_tail", "no_tail_to_tail", "power_flip", "material_power_change", "any_material_procurement_change",
    ]):
        estimate, low, high = block_bootstrap_rate(paired, metric, 440043 + cost_year + index)
        rows.append({
            "learning_year": cost_year, "account": account, "metric": metric,
            "count": int(paired[metric].sum()), "denominator": len(paired),
            "rate": estimate, "country_block_bootstrap_ci_low": low,
            "country_block_bootstrap_ci_high": high,
        })
    if sign_pairs is not None:
        robust = sign_pairs.groupby(["country", "year"], as_index=False)["robust_sign_flip"].max()
        estimate, low, high = block_bootstrap_rate(robust, "robust_sign_flip", 449943 + cost_year)
        rows.append({
            "learning_year": cost_year, "account": account,
            "metric": "country_year_with_any_ldes_lfp_robust_sign_flip",
            "count": int(robust["robust_sign_flip"].sum()), "denominator": len(robust),
            "rate": estimate, "country_block_bootstrap_ci_low": low,
            "country_block_bootstrap_ci_high": high,
        })
    return pd.DataFrame(rows)


def process(cost_year: int) -> None:
    directory = ROOT / f"cost_{cost_year}"
    results = add_efc_trajectory_account(merge_year(cost_year))
    conservation = merge_chronology_conservation(cost_year)
    choices = choose_options(
        results, "residual_peak_system_cost", "system_cost_eur", maximize=False
    )
    paired = pair_choices(choices)
    trajectory_choices = choose_options(
        results, "efc_equivalent_trajectory_screen", "efc_trajectory_net_value_eur",
        maximize=True,
    )
    trajectory_paired = pair_choices(trajectory_choices)
    comparison, sign_pairs = ldes_vs_lfp(results)
    summary = pd.concat([
        summarize(cost_year, "residual_peak_system_cost", paired, sign_pairs),
        summarize(cost_year, "efc_equivalent_trajectory_screen", trajectory_paired),
    ], ignore_index=True)
    accounting = accounting_residuals(results)
    optimized = results[results["arm_type"].ne("base_only")]
    expected_tie_a = np.maximum(
        SOLVER_PAIR_ABS_FLOOR_EUR,
        TIE_TOL_PER_KW_YEAR * 1000.0 * choices["base_power_mw"],
    )
    expected_tie_b = np.maximum(
        SOLVER_PAIR_ABS_FLOOR_EUR,
        TIE_TOL_PER_KW_YEAR * 1000.0 * trajectory_choices["base_power_mw"],
    )
    tie_formula_residual = float(max(
        np.abs(choices["tie_tolerance_eur"] - expected_tie_a).max(),
        np.abs(trajectory_choices["tie_tolerance_eur"] - expected_tie_b).max(),
    ))
    conservation_columns = [
        "base_task_conservation_error_mwh",
        "tail_task_conservation_error_mwh",
        "recharge_conservation_error_mwh",
    ]
    max_chronology_conservation = float(
        conservation[conservation_columns].abs().to_numpy().max()
    )
    arm_path = directory / "bounded_arm_results_with_base.csv.gz"
    results.to_csv(arm_path, index=False, compression={"method": "gzip", "mtime": 0})
    choices.to_csv(directory / "bounded_joint_choices.csv", index=False)
    paired.to_csv(directory / "bounded_joint_choice_pairs.csv", index=False)
    trajectory_choices.to_csv(directory / "bounded_trajectory_screen_choices.csv", index=False)
    trajectory_paired.to_csv(directory / "bounded_trajectory_screen_choice_pairs.csv", index=False)
    comparison.to_csv(directory / "bounded_ldes_vs_lfp_results.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    sign_pairs.to_csv(directory / "bounded_ldes_vs_lfp_sign_pairs.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary.to_csv(directory / "bounded_joint_choice_summary.csv", index=False)
    qa = {
        "status": "pass",
        "learning_year": cost_year,
        "country_year_count": int(paired[["country", "year"]].drop_duplicates().shape[0]),
        "country_count": int(paired["country"].nunique()),
        "optimized_tail_arm_rows": int(results["arm_type"].ne("base_only").sum()),
        "base_only_rows": int(results["arm_type"].eq("base_only").sum()),
        "joint_choice_rows": len(choices), "paired_choice_rows": len(paired),
        "trajectory_screen_choice_rows": len(trajectory_choices),
        "trajectory_screen_paired_choice_rows": len(trajectory_paired),
        "ldes_lfp_pair_rows": len(sign_pairs),
        "all_optimal": bool(optimized["solver_status_name"].eq("optimal").all()),
        "solver_status_counts": {
            str(key): int(value)
            for key, value in optimized["solver_status_name"].value_counts().to_dict().items()
        },
        "secondary_tiebreak_success_rate": float(optimized["secondary_tiebreak_success"].mean()),
        "maximum_task_balance_residual_mw": float(results["maximum_task_balance_residual_mw"].max()),
        "maximum_shared_recharge_violation_mw": float(results["maximum_shared_recharge_violation_mw"].max()),
        "maximum_soc_balance_residual_mwh": float(max(results["base_max_soc_balance_residual_mwh"].max(), results["tail_max_soc_balance_residual_mwh"].max())),
        "maximum_tail_power_lower_bound_violation_mw": float(max(0.0, -optimized["tail_power_mw"].min())),
        "maximum_tail_power_upper_bound_violation_mw": float(max(0.0, (optimized["tail_power_mw"] - optimized["tail_power_upper_mw"]).max())),
        "chronology_conservation_rows": int(len(conservation)),
        "maximum_chronology_conservation_residual_mwh": max_chronology_conservation,
        **accounting,
        "normalized_tie_formula_maximum_residual_eur": tie_formula_residual,
        "residual_peak_system_cost_flip_counts": {
            metric: int(paired[metric].sum()) for metric in [
                "identity_flip", "technology_flip", "duration_flip",
                "participation_flip", "tail_to_no_tail", "no_tail_to_tail",
                "power_flip", "material_power_change", "any_material_procurement_change",
            ]
        },
        "residual_peak_system_cost_canonical_change_counts": {
            metric: int(paired[metric].sum()) for metric in [
                "canonical_identity_change", "canonical_technology_change",
                "canonical_duration_change", "canonical_participation_change",
                "canonical_power_change",
            ]
        },
        "efc_trajectory_screen_flip_counts": {
            metric: int(trajectory_paired[metric].sum()) for metric in [
                "identity_flip", "technology_flip", "duration_flip",
                "participation_flip", "tail_to_no_tail", "no_tail_to_tail",
                "power_flip", "material_power_change", "any_material_procurement_change",
            ]
        },
        "efc_trajectory_screen_canonical_change_counts": {
            metric: int(trajectory_paired[metric].sum()) for metric in [
                "canonical_identity_change", "canonical_technology_change",
                "canonical_duration_change", "canonical_participation_change",
                "canonical_power_change",
            ]
        },
        "trajectory_screen_role": "revaluation_of_residual_peak_system_cost_optimized_trajectories__not_efc_objective_optimization",
        "solver_pair_absolute_floor_eur": SOLVER_PAIR_ABS_FLOOR_EUR,
        "choice_tolerance_eur_kw_year": TIE_TOL_PER_KW_YEAR,
        "power_share_active_tolerance": POWER_SHARE_TOL,
        "material_power_share_change_threshold": MATERIAL_POWER_SHARE_CHANGE,
        "arm_results_sha256": sha256(arm_path),
    }
    if (
        qa["country_year_count"], qa["country_count"],
        qa["optimized_tail_arm_rows"], qa["base_only_rows"],
        qa["joint_choice_rows"], qa["paired_choice_rows"],
        qa["trajectory_screen_choice_rows"], qa["trajectory_screen_paired_choice_rows"],
        qa["ldes_lfp_pair_rows"],
    ) != (132, 15, 13200, 264, 264, 132, 264, 132, 6468):
        qa["status"] = "fail"
    physical_residual_checks = [
        qa["maximum_task_balance_residual_mw"],
        qa["maximum_shared_recharge_violation_mw"],
        qa["maximum_soc_balance_residual_mwh"],
        qa["maximum_tail_power_lower_bound_violation_mw"],
        qa["maximum_tail_power_upper_bound_violation_mw"],
        qa["maximum_chronology_conservation_residual_mwh"],
        qa["normalized_tie_formula_maximum_residual_eur"],
    ]
    accounting_residual_checks = [
        qa["maximum_storage_cost_accounting_residual_eur"],
        qa["maximum_system_cost_accounting_residual_eur"],
        qa["maximum_net_benefit_accounting_residual_eur"],
        qa["maximum_efc_trajectory_accounting_residual_eur"],
    ]
    accounting_relative_residual_checks = [
        qa["maximum_storage_cost_accounting_relative_residual"],
        qa["maximum_system_cost_accounting_relative_residual"],
        qa["maximum_net_benefit_accounting_relative_residual"],
        qa["maximum_efc_trajectory_accounting_relative_residual"],
    ]
    qa["physical_residual_tolerance"] = RESIDUAL_TOL
    qa["accounting_residual_tolerance_eur"] = ACCOUNTING_TOL_EUR
    qa["accounting_relative_residual_tolerance"] = ACCOUNTING_REL_TOL
    if (
        not qa["all_optimal"]
        or max(physical_residual_checks) > RESIDUAL_TOL
        or max(accounting_residual_checks) > ACCOUNTING_TOL_EUR
        or max(accounting_relative_residual_checks) > ACCOUNTING_REL_TOL
    ):
        qa["status"] = "fail"
    write_json(directory / "bounded_joint_choice_qa.json", qa)
    print(summary.to_string(index=False)); print(json.dumps(qa, indent=2))
    if qa["status"] != "pass":
        raise RuntimeError("Bounded joint-choice postprocessing QA failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-year", type=int, choices=[2025, 2030, 2035], required=True)
    args = parser.parse_args()
    process(args.cost_year)


if __name__ == "__main__":
    main()
