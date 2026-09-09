#!/usr/bin/env python3
"""Cross-year synthesis for exact bounded-power procurement results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V43 = HERE.parent.parent / "revision_nature_energy_v43"
ROOT = HERE / "bounded_s20_highs"
CATALOGUE = V43 / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
YEARS = (2025, 2030, 2035)
SOLVER_PAIR_ABS_FLOOR_EUR = 0.02
TIE_TOL_PER_KW_YEAR = 0.01
RESIDUAL_TOL = 1.0e-6
ACCOUNTING_TOL_EUR = 1.0e-5
ACCOUNTING_REL_TOL = 1.0e-12


FLIP_METRICS = (
    "identity_flip", "technology_flip", "duration_flip", "participation_flip",
    "tail_to_no_tail", "no_tail_to_tail", "power_flip", "material_power_change",
    "any_material_procurement_change",
)


def metric_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {metric: int(frame[metric].sum()) for metric in FLIP_METRICS}


def canonical_counts(frame: pd.DataFrame) -> dict[str, int]:
    metrics = (
        "canonical_identity_change", "canonical_technology_change",
        "canonical_duration_change", "canonical_participation_change",
        "canonical_power_change",
    )
    return {metric: int(frame[metric].sum()) for metric in metrics}


def tie_formula_max_residual(frame: pd.DataFrame) -> float:
    expected = np.maximum(
        SOLVER_PAIR_ABS_FLOOR_EUR,
        TIE_TOL_PER_KW_YEAR * 1000.0 * frame["base_power_mw"],
    )
    return float(np.abs(frame["tie_tolerance_eur"] - expected).max())


def representative_changes(frame: pd.DataFrame, account: str) -> pd.DataFrame:
    selected_columns = [
        "learning_year", "country", "year",
        "dispersed_decision_id", "clustered_decision_id",
        "dispersed_decision_tech", "clustered_decision_tech",
        "dispersed_decision_duration_h", "clustered_decision_duration_h",
        "dispersed_tail_power_share", "clustered_tail_power_share",
        "dispersed_base_power_mw",
        "dispersed_tie_decision_set", "clustered_tie_decision_set",
        "identity_flip", "technology_flip", "duration_flip",
        "participation_flip", "material_power_change",
        "minimum_tie_set_power_share_gap",
        "clustered_minus_dispersed_selection_score_eur",
        "clustered_minus_dispersed_system_cost_eur",
    ]
    changes = frame[
        frame["identity_flip"] | frame["material_power_change"]
    ][selected_columns].copy()
    changes.insert(1, "account", account)
    changes["absolute_selection_score_change_eur"] = changes[
        "clustered_minus_dispersed_selection_score_eur"
    ].abs()
    changes["selection_score_change_eur_kw_year"] = (
        changes["clustered_minus_dispersed_selection_score_eur"]
        / (1000.0 * changes["dispersed_base_power_mw"])
    )
    changes["system_cost_change_eur_kw_year"] = (
        changes["clustered_minus_dispersed_system_cost_eur"]
        / (1000.0 * changes["dispersed_base_power_mw"])
    )
    changes = changes.sort_values(
        ["learning_year", "identity_flip", "participation_flip",
         "minimum_tie_set_power_share_gap", "absolute_selection_score_change_eur"],
        ascending=[True, False, False, False, False], kind="stable",
    )
    return changes


def cost_lookup() -> tuple[dict[tuple[int, str], float], dict[str, float]]:
    catalogue = pd.read_csv(CATALOGUE)
    lookup = {
        (int(row.learning_year), str(row.config_id)):
        float(row.annual_fixed_cost_eur_source_aware_per_kw_discharge_year)
        for row in catalogue.itertuples(index=False)
    }
    lookup.update({(year, "no_tail"): 0.0 for year in YEARS})
    span: dict[str, float] = {}
    for config_id, group in catalogue.groupby("config_id"):
        span[str(config_id)] = float(
            group["annual_fixed_cost_eur_source_aware_per_kw_discharge_year"].max()
            - group["annual_fixed_cost_eur_source_aware_per_kw_discharge_year"].min()
        )
    span["no_tail"] = 0.0
    return lookup, span


def main() -> None:
    summaries = []
    paired = []
    choices = []
    trajectory_paired = []
    trajectory_choices = []
    sign_pairs = []
    qa_by_year = {}
    for year in YEARS:
        directory = ROOT / f"cost_{year}"
        qa = json.loads((directory / "bounded_joint_choice_qa.json").read_text())
        if qa.get("status") != "pass":
            raise RuntimeError(f"Cost-year {year} QA failed")
        qa_by_year[str(year)] = qa
        summaries.append(pd.read_csv(directory / "bounded_joint_choice_summary.csv"))
        paired.append(pd.read_csv(directory / "bounded_joint_choice_pairs.csv"))
        choices.append(pd.read_csv(directory / "bounded_joint_choices.csv"))
        trajectory_paired.append(pd.read_csv(directory / "bounded_trajectory_screen_choice_pairs.csv"))
        trajectory_choices.append(pd.read_csv(directory / "bounded_trajectory_screen_choices.csv"))
        sign_pairs.append(pd.read_csv(directory / "bounded_ldes_vs_lfp_sign_pairs.csv.gz"))
    summary = pd.concat(summaries, ignore_index=True)
    pairs = pd.concat(paired, ignore_index=True)
    choice = pd.concat(choices, ignore_index=True)
    trajectory_pairs = pd.concat(trajectory_paired, ignore_index=True)
    trajectory_choice = pd.concat(trajectory_choices, ignore_index=True)
    signs = pd.concat(sign_pairs, ignore_index=True)

    transitions = (
        pairs.groupby(
            ["learning_year", "dispersed_decision_id", "clustered_decision_id"],
            as_index=False,
        ).size().rename(columns={"size": "country_years"})
        .sort_values(["learning_year", "country_years"], ascending=[True, False])
    )
    technology_transitions = (
        pairs.groupby(
            ["learning_year", "dispersed_decision_tech", "clustered_decision_tech"],
            as_index=False,
        ).size().rename(columns={"size": "country_years"})
        .sort_values(["learning_year", "country_years"], ascending=[True, False])
    )
    winner_distribution = (
        choice.groupby(
            ["learning_year", "chronology", "decision_id", "decision_tech", "decision_duration_h"],
            as_index=False,
        ).size().rename(columns={"size": "country_years"})
        .sort_values(["learning_year", "chronology", "country_years"], ascending=[True, True, False])
    )
    trajectory_transitions = (
        trajectory_pairs.groupby(
            ["learning_year", "dispersed_decision_id", "clustered_decision_id"],
            as_index=False,
        ).size().rename(columns={"size": "country_years"})
        .sort_values(["learning_year", "country_years"], ascending=[True, False])
    )
    trajectory_winner_distribution = (
        trajectory_choice.groupby(
            ["learning_year", "chronology", "decision_id", "decision_tech", "decision_duration_h"],
            as_index=False,
        ).size().rename(columns={"size": "country_years"})
        .sort_values(["learning_year", "chronology", "country_years"], ascending=[True, True, False])
    )
    trajectory_technology_transitions = (
        trajectory_pairs.groupby(
            ["learning_year", "dispersed_decision_tech", "clustered_decision_tech"],
            as_index=False,
        ).size().rename(columns={"size": "country_years"})
        .sort_values(["learning_year", "country_years"], ascending=[True, False])
    )
    robust_sign_transitions = (
        signs[signs["robust_sign_flip"]]
        .groupby(["learning_year", "config_id", "tech_id", "duration_h"], as_index=False)
        .agg(country_year_config_flips=("robust_sign_flip", "size"), countries=("country", "nunique"))
        .sort_values(["learning_year", "country_year_config_flips"], ascending=[True, False])
    )

    # Compare each realized decision margin to the 2025--2035 span of the
    # winner-versus-runner-up fixed-cost differential, holding the two power
    # shares at their current optimized values.  Both quantities are thereby
    # expressed per kW of the common base asset.  This is a transparent local
    # cost-path perturbation, not a probability interval or a reoptimization.
    lookup, _ = cost_lookup()
    margin_rows = []
    for row in choice.itertuples(index=False):
        winner = str(row.decision_id)
        runner = str(row.runner_up_id)
        if not runner:
            differential_span = np.nan
        else:
            winner_share = float(row.tail_power_share)
            runner_share = float(row.runner_up_tail_power_share)
            differentials = [
                lookup.get((year, winner), np.nan) * winner_share
                - lookup.get((year, runner), np.nan) * runner_share
                for year in YEARS
            ]
            differential_span = float(np.nanmax(differentials) - np.nanmin(differentials))
        margin_rows.append({
            "learning_year": int(row.learning_year), "country": row.country,
            "year": int(row.year), "chronology": row.chronology,
            "decision_id": winner, "runner_up_id": runner,
            "decision_margin_eur_kw_year": float(row.decision_margin_eur_kw_year),
            "winner_runner_cost_differential_2025_2035_span_eur_per_base_kw_year_holding_power": differential_span,
            "margin_within_cost_path_span": bool(
                np.isfinite(differential_span)
                and float(row.decision_margin_eur_kw_year) <= differential_span
            ),
        })
    margins = pd.DataFrame(margin_rows)
    margin_summary = (
        margins.groupby(["learning_year", "chronology"], as_index=False)
        .agg(
            median_decision_margin_eur_kw_year=("decision_margin_eur_kw_year", "median"),
            q25_decision_margin_eur_kw_year=("decision_margin_eur_kw_year", lambda x: x.quantile(0.25)),
            q75_decision_margin_eur_kw_year=("decision_margin_eur_kw_year", lambda x: x.quantile(0.75)),
            median_cost_path_span_eur_per_base_kw_year_holding_power=("winner_runner_cost_differential_2025_2035_span_eur_per_base_kw_year_holding_power", "median"),
            share_margins_within_cost_path_span=("margin_within_cost_path_span", "mean"),
        )
    )

    primary_changes = representative_changes(pairs, "residual_peak_system_cost")
    trajectory_changes = representative_changes(
        trajectory_pairs, "efc_equivalent_trajectory_screen"
    )
    all_changes = pd.concat([primary_changes, trajectory_changes], ignore_index=True)
    representative = (
        all_changes.groupby(["account", "learning_year"], group_keys=False)
        .head(5).reset_index(drop=True)
    )

    summary.to_csv(ROOT / "bounded_joint_choice_summary_all_years.csv", index=False)
    pairs.to_csv(ROOT / "bounded_joint_choice_pairs_all_years.csv", index=False)
    choice.to_csv(ROOT / "bounded_joint_choices_all_years.csv", index=False)
    trajectory_pairs.to_csv(ROOT / "bounded_trajectory_screen_choice_pairs_all_years.csv", index=False)
    trajectory_choice.to_csv(ROOT / "bounded_trajectory_screen_choices_all_years.csv", index=False)
    trajectory_transitions.to_csv(ROOT / "bounded_trajectory_screen_decision_transitions_all_years.csv", index=False)
    trajectory_technology_transitions.to_csv(ROOT / "bounded_trajectory_screen_technology_transitions_all_years.csv", index=False)
    trajectory_winner_distribution.to_csv(ROOT / "bounded_trajectory_screen_winner_distribution_all_years.csv", index=False)
    transitions.to_csv(ROOT / "bounded_decision_transitions_all_years.csv", index=False)
    technology_transitions.to_csv(ROOT / "bounded_technology_transitions_all_years.csv", index=False)
    winner_distribution.to_csv(ROOT / "bounded_winner_distribution_all_years.csv", index=False)
    robust_sign_transitions.to_csv(ROOT / "bounded_ldes_lfp_robust_sign_flips_all_years.csv", index=False)
    margins.to_csv(ROOT / "bounded_decision_margin_vs_cost_path.csv", index=False)
    margin_summary.to_csv(ROOT / "bounded_decision_margin_summary.csv", index=False)
    all_changes.to_csv(ROOT / "bounded_all_material_procurement_changes_all_years.csv", index=False)
    representative.to_csv(ROOT / "bounded_representative_procurement_changes.csv", index=False)

    tie_formula_residuals = {
        "residual_peak_system_cost": tie_formula_max_residual(choice),
        "efc_equivalent_trajectory_screen": tie_formula_max_residual(trajectory_choice),
    }
    exact_primary_counts = {
        str(year): metric_counts(pairs[pairs["learning_year"].eq(year)])
        for year in YEARS
    }
    exact_trajectory_counts = {
        str(year): metric_counts(trajectory_pairs[trajectory_pairs["learning_year"].eq(year)])
        for year in YEARS
    }
    exact_primary_canonical_counts = {
        str(year): canonical_counts(pairs[pairs["learning_year"].eq(year)])
        for year in YEARS
    }
    exact_trajectory_canonical_counts = {
        str(year): canonical_counts(trajectory_pairs[trajectory_pairs["learning_year"].eq(year)])
        for year in YEARS
    }
    per_year_row_checks = {
        str(year): {
            "country_year_pairs": int(pairs["learning_year"].eq(year).sum()),
            "primary_chronology_choices": int(choice["learning_year"].eq(year).sum()),
            "trajectory_screen_chronology_choices": int(
                trajectory_choice["learning_year"].eq(year).sum()
            ),
            "optimized_tail_arm_rows": int(qa_by_year[str(year)]["optimized_tail_arm_rows"]),
            "base_only_rows": int(qa_by_year[str(year)]["base_only_rows"]),
            "all_solver_statuses_optimal": bool(qa_by_year[str(year)]["all_optimal"]),
        }
        for year in YEARS
    }
    maximum_solver_residual = max(
        max(
            float(qa_by_year[str(year)][name])
            for name in (
                "maximum_task_balance_residual_mw",
                "maximum_shared_recharge_violation_mw",
                "maximum_soc_balance_residual_mwh",
                "maximum_tail_power_lower_bound_violation_mw",
                "maximum_tail_power_upper_bound_violation_mw",
                "maximum_chronology_conservation_residual_mwh",
            )
        )
        for year in YEARS
    )
    maximum_accounting_residual = max(
        max(
            float(qa_by_year[str(year)][name])
            for name in (
                "maximum_storage_cost_accounting_residual_eur",
                "maximum_system_cost_accounting_residual_eur",
                "maximum_net_benefit_accounting_residual_eur",
                "maximum_efc_trajectory_accounting_residual_eur",
            )
        )
        for year in YEARS
    )
    maximum_accounting_relative_residual = max(
        max(
            float(qa_by_year[str(year)][name])
            for name in (
                "maximum_storage_cost_accounting_relative_residual",
                "maximum_system_cost_accounting_relative_residual",
                "maximum_net_benefit_accounting_relative_residual",
                "maximum_efc_trajectory_accounting_relative_residual",
            )
        )
        for year in YEARS
    )
    final_qa = {
        "status": "pass",
        "cost_years": list(YEARS),
        "country_years_per_cost_year": 132,
        "total_country_year_pairs": len(pairs),
        "total_trajectory_screen_country_year_pairs": len(trajectory_pairs),
        "primary_account": "residual_peak_system_cost_exact_joint_procurement",
        "secondary_account": "efc_equivalent_trajectory_screen",
        "secondary_account_role": "revaluation_of_residual_peak_system_cost_optimized_trajectories__not_efc_objective_optimization",
        "all_year_QA_passed": all(item["status"] == "pass" for item in qa_by_year.values()),
        "per_year_row_and_solver_checks": per_year_row_checks,
        "maximum_solver_or_conservation_residual": maximum_solver_residual,
        "maximum_accounting_residual_eur": maximum_accounting_residual,
        "maximum_accounting_relative_residual": maximum_accounting_relative_residual,
        "physical_residual_tolerance": RESIDUAL_TOL,
        "accounting_residual_tolerance_eur": ACCOUNTING_TOL_EUR,
        "accounting_relative_residual_tolerance": ACCOUNTING_REL_TOL,
        "normalized_tie_formula": "max(0.02 EUR, 0.01 EUR kW-1 yr-1 multiplied by base power in kW)",
        "normalized_tie_formula_maximum_residual_eur": tie_formula_residuals,
        "residual_peak_system_cost_flip_counts": exact_primary_counts,
        "residual_peak_system_cost_canonical_change_counts": exact_primary_canonical_counts,
        "efc_equivalent_trajectory_screen_flip_counts": exact_trajectory_counts,
        "efc_equivalent_trajectory_screen_canonical_change_counts": exact_trajectory_canonical_counts,
        "robust_LDES_LFP_sign_flip_country_year_counts": {
            str(year): int(
                signs[signs["learning_year"].eq(year)]
                .groupby(["country", "year"])["robust_sign_flip"].max().sum()
            ) for year in YEARS
        },
        "year_QA": qa_by_year,
        "representative_change_cases": json.loads(
            representative.to_json(orient="records")
        ),
    }
    row_contract_pass = all(
        checks == {
            "country_year_pairs": 132,
            "primary_chronology_choices": 264,
            "trajectory_screen_chronology_choices": 264,
            "optimized_tail_arm_rows": 13200,
            "base_only_rows": 264,
            "all_solver_statuses_optimal": True,
        }
        for checks in per_year_row_checks.values()
    )
    uniqueness_pass = (
        not pairs.duplicated(["learning_year", "country", "year"]).any()
        and not trajectory_pairs.duplicated(["learning_year", "country", "year"]).any()
        and not choice.duplicated(["learning_year", "country", "year", "chronology"]).any()
        and not trajectory_choice.duplicated(
            ["learning_year", "country", "year", "chronology"]
        ).any()
    )
    transition_totals_pass = all(
        int(transitions.loc[transitions["learning_year"].eq(year), "country_years"].sum()) == 132
        and int(trajectory_transitions.loc[
            trajectory_transitions["learning_year"].eq(year), "country_years"
        ].sum()) == 132
        for year in YEARS
    )
    contract_pass = (
        len(pairs) == 396 and len(trajectory_pairs) == 396
        and len(choice) == 792 and len(trajectory_choice) == 792
        and len(signs) == 19_404
        and row_contract_pass and uniqueness_pass and transition_totals_pass
        and final_qa["all_year_QA_passed"]
        and maximum_solver_residual <= RESIDUAL_TOL
        and maximum_accounting_residual <= ACCOUNTING_TOL_EUR
        and maximum_accounting_relative_residual <= ACCOUNTING_REL_TOL
        and max(tie_formula_residuals.values()) <= RESIDUAL_TOL
        and set(choice["selection_account"]) == {"residual_peak_system_cost"}
        and set(trajectory_choice["selection_account"]) == {
            "efc_equivalent_trajectory_screen"
        }
    )
    final_qa["row_contract_pass"] = row_contract_pass
    final_qa["uniqueness_pass"] = uniqueness_pass
    final_qa["transition_totals_pass"] = transition_totals_pass
    final_qa["all_contract_checks_pass"] = contract_pass
    if not contract_pass:
        final_qa["status"] = "fail"
    (ROOT / "BOUNDED_ALL_YEAR_FINAL_QA.json").write_text(
        json.dumps(final_qa, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(summary.to_string(index=False))
    print("\nDecision transitions\n", transitions.to_string(index=False))
    print("\nMargin summary\n", margin_summary.to_string(index=False))
    print(json.dumps(final_qa, indent=2))
    if final_qa["status"] != "pass":
        raise RuntimeError("All-year bounded result QA failed")


if __name__ == "__main__":
    main()
