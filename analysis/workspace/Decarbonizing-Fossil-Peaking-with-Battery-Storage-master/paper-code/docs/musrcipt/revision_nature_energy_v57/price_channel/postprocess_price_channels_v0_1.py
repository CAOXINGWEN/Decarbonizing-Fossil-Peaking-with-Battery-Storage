#!/usr/bin/env python3
"""Merge and summarize frozen-dispatch price-channel experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TOL = 0.01
COST_CATALOGUE = HERE.parent / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
PORTFOLIO_MODEL = HERE.parent / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"
BASELINE_EFC_OPTIONS = HERE.parent / "joint_choice" / "fixed_power_option_values.csv.gz"


def load_portfolio_module():
    spec = importlib.util.spec_from_file_location("price_post_portfolio", PORTFOLIO_MODEL)
    if spec is None or spec.loader is None:
        raise RuntimeError(PORTFOLIO_MODEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_country_mean(frame: pd.DataFrame, value: str, draws: int = 10000) -> tuple[float, float, float]:
    country = frame.groupby("country", as_index=False)[value].mean()
    values = country[value].to_numpy(float)
    rng = np.random.default_rng(4601)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(sampled, [0.025, 0.975]).tolist()


def strict_valley(group: pd.DataFrame) -> dict[str, object]:
    ordered = group.sort_values("duration_h", kind="stable")
    values = ordered.benefit_eur_kw_year.to_numpy(float)
    hours = ordered.duration_h.to_numpy(float)
    positive, negative = values > TOL, values < -TOL
    candidates = []
    i = 0
    while i < len(values):
        if not negative[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(values) and negative[i + 1]:
            i += 1
        end = i
        if end - start + 1 >= 2 and start > 0 and end < len(values) - 1 and positive[start - 1] and positive[end + 1]:
            candidates.append((hours[end + 1] - hours[start - 1], start, end))
        i += 1
    if candidates and not np.any(np.abs(values) <= TOL):
        width, start, end = sorted(candidates, reverse=True)[0]
        return {
            "strict_valley": True,
            "left_boundary_h": hours[start - 1],
            "right_boundary_h": hours[end + 1],
            "valley_width_h": width,
            "negative_durations_h": "|".join(f"{x:g}" for x in hours[start:end + 1]),
        }
    return {"strict_valley": False, "left_boundary_h": np.nan, "right_boundary_h": np.nan, "valley_width_h": 0.0, "negative_durations_h": ""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=HERE / "full_chunks")
    parser.add_argument("--output", type=Path, default=HERE / "summary")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_root.glob("chunk*/price_channel_arm_revaluation.csv.gz"))
    if not paths:
        raise RuntimeError("No completed price-channel chunks")
    arms = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    scenarios = [column.removesuffix("_system_cost_eur") for column in arms if column.endswith("_system_cost_eur")]
    expected = arms[["country", "year"]].drop_duplicates().shape[0]
    rows, envelopes, choices, valleys = [], [], [], []
    for scenario in scenarios:
        cost = f"{scenario}_system_cost_eur"
        for (country, year, chronology), group in arms.groupby(["country", "year", "chronology"], sort=False):
            lfp = group[group.arm_type.eq("extra_lfp4_comparator")].iloc[0]
            candidates = group[group.arm_type.eq("ldes_candidate")].copy()
            candidates["benefit_eur_kw_year"] = (float(lfp[cost]) - candidates[cost]) / (candidates.tail_power_mw * 1000.0)
            candidates["scenario"] = scenario
            rows.append(candidates)
            for duration, duration_group in candidates.groupby("duration_h"):
                best = duration_group.sort_values("benefit_eur_kw_year", ascending=False).iloc[0]
                envelopes.append({
                    "scenario": scenario, "country": country, "year": year, "chronology": chronology,
                    "duration_h": duration, "benefit_eur_kw_year": best.benefit_eur_kw_year,
                    "dominant_config_id": best.config_id, "dominant_tech_id": best.tech_id,
                })
            selected = group.sort_values(cost, kind="stable").iloc[0]
            selected_class = "no_tail" if selected.arm_type == "base_only" else ("additional_lfp4" if selected.arm_type == "extra_lfp4_comparator" else "ldes")
            choices.append({
                "scenario": scenario, "country": country, "year": year, "chronology": chronology,
                "selected_class": selected_class, "selected_config_id": selected.config_id,
                "selected_tech_id": selected.tech_id, "selected_duration_h": selected.duration_h,
                "selected_system_cost_eur": selected[cost],
            })
    cells = pd.concat(rows, ignore_index=True)
    envelope = pd.DataFrame(envelopes)
    for key, group in envelope.groupby(["scenario", "country", "year", "chronology"]):
        valleys.append({"scenario": key[0], "country": key[1], "year": key[2], "chronology": key[3], **strict_valley(group)})
    valley = pd.DataFrame(valleys)
    choice = pd.DataFrame(choices)

    paired_cells = cells.pivot(index=["scenario", "country", "year", "config_id", "tech_id", "duration_h"], columns="chronology", values="benefit_eur_kw_year").reset_index()
    paired_cells["chronology_effect_eur_kw_year"] = paired_cells.clustered - paired_cells.dispersed
    paired_env = envelope.pivot(index=["scenario", "country", "year", "duration_h"], columns="chronology", values="benefit_eur_kw_year").reset_index()
    paired_env["chronology_effect_eur_kw_year"] = paired_env.clustered - paired_env.dispersed
    choice_wide = choice.pivot(index=["scenario", "country", "year"], columns="chronology", values=["selected_class", "selected_config_id", "selected_tech_id", "selected_duration_h"]).reset_index()
    choice_wide.columns = ["_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else col for col in choice_wide.columns]
    choice_wide["identity_flip"] = choice_wide.selected_config_id_clustered != choice_wide.selected_config_id_dispersed
    choice_wide["technology_flip"] = choice_wide.selected_tech_id_clustered != choice_wide.selected_tech_id_dispersed
    choice_wide["duration_flip"] = choice_wide.selected_duration_h_clustered != choice_wide.selected_duration_h_dispersed
    choice_wide["participation_flip"] = choice_wide.selected_class_clustered != choice_wide.selected_class_dispersed

    focal = paired_cells[(paired_cells.tech_id.eq("caes")) & (paired_cells.duration_h.eq(24))]
    focal_summary = []
    for scenario, group in focal.groupby("scenario"):
        estimate, low, high = bootstrap_country_mean(group, "chronology_effect_eur_kw_year")
        focal_summary.append({"scenario": scenario, "n_country_years": len(group), "mean_effect_eur_kw_year": estimate, "ci95_low": low, "ci95_high": high})
    duration_summary = []
    for (scenario, duration), group in paired_env.groupby(["scenario", "duration_h"]):
        estimate, low, high = bootstrap_country_mean(group, "chronology_effect_eur_kw_year")
        duration_summary.append({"scenario": scenario, "duration_h": duration, "mean_effect_eur_kw_year": estimate, "ci95_low": low, "ci95_high": high})
    decision_summary = choice_wide.groupby("scenario", as_index=False).agg(
        n_country_years=("country", "size"), identity_flips=("identity_flip", "sum"),
        technology_flips=("technology_flip", "sum"), duration_flips=("duration_flip", "sum"),
        participation_flips=("participation_flip", "sum"),
    )
    valley_summary = valley.groupby(["scenario", "chronology"], as_index=False).agg(
        n_country_years=("country", "size"), strict_valleys=("strict_valley", "sum"),
        mean_width_when_present=("valley_width_h", lambda x: x[x > 0].mean()),
    )
    beta = pd.DataFrame(focal_summary)
    beta = beta[beta.scenario.isin(["S0", "S1_beta_0.25", "S1_beta_0.5", "S1_beta_1"])]
    beta["beta"] = beta.scenario.map({"S0": 0.0, "S1_beta_0.25": 0.25, "S1_beta_0.5": 0.5, "S1_beta_1": 1.0})
    slope, intercept = np.polyfit(beta.beta, beta.mean_effect_eur_kw_year, 1)
    critical = -intercept / slope if slope != 0 and -intercept / slope >= 0 else np.nan
    critical_summary = {"intercept_effect_eur_kw_year": intercept, "slope_per_beta": slope, "critical_beta": critical}

    arms.to_csv(args.output / "price_channel_arms_merged.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    cells.to_csv(args.output / "price_channel_mix_cells.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    envelope.to_csv(args.output / "price_channel_duration_envelope.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    paired_cells.to_csv(args.output / "price_channel_paired_cells.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    paired_env.to_csv(args.output / "price_channel_paired_duration_envelope.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    choice_wide.to_csv(args.output / "price_channel_choice_flips.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    valley.to_csv(args.output / "price_channel_valley_records.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(focal_summary).to_csv(args.output / "price_channel_caes24_summary.csv", index=False)
    pd.DataFrame(duration_summary).to_csv(args.output / "price_channel_duration_summary.csv", index=False)
    decision_summary.to_csv(args.output / "price_channel_decision_summary.csv", index=False)
    valley_summary.to_csv(args.output / "price_channel_valley_summary.csv", index=False)
    (args.output / "price_channel_critical_beta.json").write_text(json.dumps(critical_summary, indent=2) + "\n")

    # Revalue the identical physical schedules under the three registered
    # technology-specific fixed-cost years.  The model catalogue used to
    # recover the schedules supplies the fixed cost already embedded in each
    # arm; it is removed before the 2025/2030/2035 value is added.  Common
    # base-layer cost cancels in every arm comparison and remains untouched.
    portfolio = load_portfolio_module()
    model_catalogue = pd.read_csv(portfolio.DEFAULT_CATALOGUE)
    embedded_fixed = model_catalogue.set_index("config_id")[
        "annual_fixed_cost_eur_per_kw_year_source_aware"
    ].to_dict()
    annual_costs = pd.read_csv(COST_CATALOGUE)[
        ["learning_year", "config_id", "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"]
    ].rename(columns={"annual_fixed_cost_eur_source_aware_per_kw_discharge_year": "annual_fixed_cost_eur_kw_year"})
    annual_map = annual_costs.set_index(["learning_year", "config_id"])["annual_fixed_cost_eur_kw_year"].to_dict()
    arms["embedded_tail_fixed_cost_eur_kw_year"] = arms.config_id.map(embedded_fixed).fillna(0.0)
    year_cells: list[pd.DataFrame] = []
    year_choices: list[dict[str, object]] = []
    year_envelopes: list[dict[str, object]] = []
    for learning_year in (2025, 2030, 2035):
        year_frame = arms.copy()
        year_frame["cost_year"] = learning_year
        year_frame["annual_tail_fixed_cost_eur_kw_year"] = [
            0.0 if arm == "base_only" else annual_map[(learning_year, config)]
            for arm, config in zip(year_frame.arm_type, year_frame.config_id)
        ]
        old_tail = year_frame.tail_power_mw * 1000.0 * year_frame.embedded_tail_fixed_cost_eur_kw_year
        new_tail = year_frame.tail_power_mw * 1000.0 * year_frame.annual_tail_fixed_cost_eur_kw_year
        for scenario in scenarios:
            adjusted = f"{scenario}_system_cost_{learning_year}_eur"
            year_frame[adjusted] = year_frame[f"{scenario}_system_cost_eur"] - old_tail + new_tail
            for (country, year, chronology), group in year_frame.groupby(["country", "year", "chronology"], sort=False):
                lfp = group[group.arm_type.eq("extra_lfp4_comparator")].iloc[0]
                candidates = group[group.arm_type.eq("ldes_candidate")].copy()
                candidates["benefit_eur_kw_year"] = (
                    float(lfp[adjusted]) - candidates[adjusted]
                ) / (candidates.tail_power_mw * 1000.0)
                candidates["scenario"] = scenario
                candidates["cost_year"] = learning_year
                year_cells.append(candidates[[
                    "scenario", "cost_year", "country", "year", "chronology", "config_id",
                    "tech_id", "duration_h", "tail_power_mw", "benefit_eur_kw_year",
                ]])
                for duration, duration_group in candidates.groupby("duration_h"):
                    best = duration_group.sort_values("benefit_eur_kw_year", ascending=False).iloc[0]
                    year_envelopes.append({
                        "scenario": scenario, "cost_year": learning_year, "country": country,
                        "year": year, "chronology": chronology, "duration_h": duration,
                        "benefit_eur_kw_year": best.benefit_eur_kw_year,
                        "dominant_config_id": best.config_id, "dominant_tech_id": best.tech_id,
                    })
                selected = group.sort_values(adjusted, kind="stable").iloc[0]
                selected_class = "no_tail" if selected.arm_type == "base_only" else (
                    "additional_lfp4" if selected.arm_type == "extra_lfp4_comparator" else "ldes"
                )
                year_choices.append({
                    "scenario": scenario, "cost_year": learning_year, "country": country,
                    "year": year, "chronology": chronology, "selected_class": selected_class,
                    "selected_config_id": selected.config_id, "selected_tech_id": selected.tech_id,
                    "selected_duration_h": selected.duration_h, "selected_system_cost_eur": selected[adjusted],
                })
    year_cells_frame = pd.concat(year_cells, ignore_index=True)
    year_envelope = pd.DataFrame(year_envelopes)
    year_choice = pd.DataFrame(year_choices)
    year_valleys = []
    for key, group in year_envelope.groupby(["scenario", "cost_year", "country", "year", "chronology"]):
        year_valleys.append({
            "scenario": key[0], "cost_year": key[1], "country": key[2], "year": key[3],
            "chronology": key[4], **strict_valley(group),
        })
    year_valley = pd.DataFrame(year_valleys)
    year_choice_wide = year_choice.pivot(
        index=["scenario", "cost_year", "country", "year"], columns="chronology",
        values=["selected_class", "selected_config_id", "selected_tech_id", "selected_duration_h"],
    ).reset_index()
    year_choice_wide.columns = [
        "_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else col
        for col in year_choice_wide.columns
    ]
    year_choice_wide["identity_flip"] = year_choice_wide.selected_config_id_clustered != year_choice_wide.selected_config_id_dispersed
    year_choice_wide["technology_flip"] = year_choice_wide.selected_tech_id_clustered != year_choice_wide.selected_tech_id_dispersed
    year_choice_wide["duration_flip"] = year_choice_wide.selected_duration_h_clustered != year_choice_wide.selected_duration_h_dispersed
    year_choice_wide["participation_flip"] = year_choice_wide.selected_class_clustered != year_choice_wide.selected_class_dispersed
    year_decision_summary = year_choice_wide.groupby(["scenario", "cost_year"], as_index=False).agg(
        n_country_years=("country", "size"), identity_flips=("identity_flip", "sum"),
        technology_flips=("technology_flip", "sum"), duration_flips=("duration_flip", "sum"),
        participation_flips=("participation_flip", "sum"),
    )
    year_valley_summary = year_valley.groupby(["scenario", "cost_year", "chronology"], as_index=False).agg(
        n_country_years=("country", "size"), strict_valleys=("strict_valley", "sum"),
        mean_width_when_present=("valley_width_h", lambda x: x[x > 0].mean()),
    )
    year_cells_frame.to_csv(args.output / "price_channel_year_specific_mix_cells.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    year_envelope.to_csv(args.output / "price_channel_year_specific_duration_envelope.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    year_choice_wide.to_csv(args.output / "price_channel_year_specific_choice_flips.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    year_valley.to_csv(args.output / "price_channel_year_specific_valley_records.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    year_decision_summary.to_csv(args.output / "price_channel_year_specific_decision_summary.csv", index=False)
    year_valley_summary.to_csv(args.output / "price_channel_year_specific_valley_summary.csv", index=False)

    # Main decision account: apply only the hourly price-channel increment to
    # the already validated EENS-equivalent option ledger.  This retains the
    # exact 2025/2030/2035 fixed costs, EFC values and 0.01 EUR kW-1 yr-1
    # set-valued tie convention used for the reported 0/3/22 tail-entry flips.
    baseline = pd.read_csv(BASELINE_EFC_OPTIONS)
    available_units = arms[["country", "year"]].drop_duplicates()
    baseline = baseline.merge(available_units, on=["country", "year"], how="inner", validate="many_to_one")
    price_keys = ["country", "year", "chronology", "arm_type", "config_id"]
    price_delta_rows = []
    for scenario in scenarios:
        subset = arms[price_keys].copy()
        subset["scenario"] = scenario
        subset["price_channel_net_value_delta_eur"] = -(
            arms[f"{scenario}_system_cost_eur"] - arms["S0_system_cost_eur"]
        )
        price_delta_rows.append(subset)
    price_delta = pd.concat(price_delta_rows, ignore_index=True)
    efc = baseline.merge(price_delta, on=price_keys, how="inner", validate="many_to_many")
    if len(efc) != len(baseline) * len(scenarios):
        raise RuntimeError("Price-channel rows do not align with the validated EFC option ledger")
    efc["price_channel_net_value_eur"] = (
        efc.efc_net_value_revalued_eur + efc.price_channel_net_value_delta_eur
    )
    group_keys = ["scenario", "cost_year", "country", "year", "chronology"]
    efc["scenario_base_only_net_value_eur"] = efc.groupby(group_keys, sort=False)[
        "price_channel_net_value_eur"
    ].transform(lambda x: x.loc[efc.loc[x.index, "arm_type"].eq("base_only")].iloc[0])
    efc["price_channel_score_eur_kw_year"] = (
        efc.price_channel_net_value_eur - efc.scenario_base_only_net_value_eur
    ) / (efc.base_power_mw * 1000.0)

    # Set-valued optimum.  A chronology pair flips robustly only when its two
    # 0.01-EUR tie sets are disjoint; this reproduces the registered decision
    # rule rather than counting arbitrary lexicographic changes.
    efc["top_score"] = efc.groupby(group_keys, sort=False).price_channel_score_eur_kw_year.transform("max")
    tied = efc.loc[efc.price_channel_score_eur_kw_year.ge(efc.top_score - TOL)].copy()
    tied["class"] = np.select(
        [tied.arm_type.eq("base_only"), tied.arm_type.eq("extra_lfp4_comparator")],
        ["no_tail", "additional_lfp4"], default="ldes",
    )
    tie_sets = tied.groupby(group_keys, as_index=False).agg(
        tie_config_set=("config_id", lambda x: "|".join(sorted(set(x)))),
        tie_class_set=("class", lambda x: "|".join(sorted(set(x)))),
        tie_size=("config_id", "size"),
    )
    canonical = (
        tied.assign(priority=np.where(tied.arm_type.eq("base_only"), 0, 1))
        .sort_values(group_keys + ["priority", "config_id"], kind="stable")
        .drop_duplicates(group_keys)
    )
    selected = canonical[group_keys + ["config_id", "tech_id", "duration_h", "class", "price_channel_score_eur_kw_year"]].merge(
        tie_sets, on=group_keys, how="left", validate="one_to_one"
    )
    selected_wide = selected.pivot(
        index=["scenario", "cost_year", "country", "year"], columns="chronology",
        values=["config_id", "tech_id", "duration_h", "class", "price_channel_score_eur_kw_year", "tie_config_set", "tie_class_set", "tie_size"],
    ).reset_index()
    selected_wide.columns = [
        "_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else col
        for col in selected_wide.columns
    ]
    def disjoint_sets(row, prefix: str) -> bool:
        left = set(str(row[f"{prefix}_dispersed"]).split("|"))
        right = set(str(row[f"{prefix}_clustered"]).split("|"))
        return left.isdisjoint(right)
    selected_wide["robust_identity_flip"] = selected_wide.apply(lambda r: disjoint_sets(r, "tie_config_set"), axis=1)
    selected_wide["robust_participation_flip"] = selected_wide.apply(lambda r: disjoint_sets(r, "tie_class_set"), axis=1)
    selected_wide["canonical_duration_flip"] = selected_wide.duration_h_clustered != selected_wide.duration_h_dispersed
    selected_wide["canonical_technology_flip"] = selected_wide.tech_id_clustered != selected_wide.tech_id_dispersed
    efc_decision_summary = selected_wide.groupby(["scenario", "cost_year"], as_index=False).agg(
        n_country_years=("country", "size"), robust_identity_flips=("robust_identity_flip", "sum"),
        robust_participation_flips=("robust_participation_flip", "sum"),
        canonical_duration_flips=("canonical_duration_flip", "sum"),
        canonical_technology_flips=("canonical_technology_flip", "sum"),
    )

    efc_envelopes = []
    efc_valleys = []
    for key, group in efc.groupby(group_keys, sort=False):
        lfp = group[group.arm_type.eq("extra_lfp4_comparator")].iloc[0]
        candidates = group[group.arm_type.eq("ldes_candidate")].copy()
        candidates["benefit_eur_kw_year"] = (
            candidates.price_channel_score_eur_kw_year - float(lfp.price_channel_score_eur_kw_year)
        )
        duration_rows = []
        for duration, duration_group in candidates.groupby("duration_h"):
            best = duration_group.sort_values("benefit_eur_kw_year", ascending=False).iloc[0]
            duration_rows.append({
                "scenario": key[0], "cost_year": key[1], "country": key[2], "year": key[3],
                "chronology": key[4], "duration_h": duration,
                "benefit_eur_kw_year": best.benefit_eur_kw_year,
                "dominant_config_id": best.config_id, "dominant_tech_id": best.tech_id,
            })
        efc_envelopes.extend(duration_rows)
        efc_valleys.append({
            "scenario": key[0], "cost_year": key[1], "country": key[2], "year": key[3],
            "chronology": key[4], **strict_valley(pd.DataFrame(duration_rows)),
        })
    efc_envelope = pd.DataFrame(efc_envelopes)
    efc_valley = pd.DataFrame(efc_valleys)
    efc_valley_summary = efc_valley.groupby(["scenario", "cost_year", "chronology"], as_index=False).agg(
        n_country_years=("country", "size"), strict_valleys=("strict_valley", "sum"),
        mean_width_when_present=("valley_width_h", lambda x: x[x > 0].mean()),
    )
    efc.to_csv(args.output / "price_channel_efc_option_values.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    selected_wide.to_csv(args.output / "price_channel_efc_choice_flips.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    efc_envelope.to_csv(args.output / "price_channel_efc_duration_envelope.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    efc_valley.to_csv(args.output / "price_channel_efc_valley_records.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    efc_decision_summary.to_csv(args.output / "price_channel_efc_decision_summary.csv", index=False)
    efc_valley_summary.to_csv(args.output / "price_channel_efc_valley_summary.csv", index=False)

    qa = {
        "country_years": expected, "arm_rows": len(arms), "scenarios": scenarios,
        "all_scenarios_132": bool((decision_summary.n_country_years == expected).all()),
        "all_year_scenarios_complete": bool((year_decision_summary.n_country_years == expected).all()),
        "cost_years": [2025, 2030, 2035], "critical_beta": critical,
        "efc_s0_participation_flips": efc_decision_summary.loc[
            efc_decision_summary.scenario.eq("S0"), ["cost_year", "robust_participation_flips"]
        ].set_index("cost_year").robust_participation_flips.astype(int).to_dict(),
    }
    (args.output / "qa.json").write_text(json.dumps(qa, indent=2) + "\n")


if __name__ == "__main__":
    main()
