#!/usr/bin/env python3
"""Build the auditable result tables used in the v62 Supplementary Information.

The script only summarizes frozen model outputs.  It does not rerun, alter or
select optimization results.  Outputs from earlier frozen revision directories
are used where the corresponding robustness branch was not copied into v62.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V62 = HERE.parent
MANUSCRIPT_ROOT = V62.parent
V57 = MANUSCRIPT_ROOT / "revision_nature_energy_v57"
V44 = MANUSCRIPT_ROOT / "revision_nature_energy_v44"


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_string(values: dict[int, int]) -> str:
    return "/".join(str(values[year]) for year in (2025, 2030, 2035))


def main() -> None:
    inputs = {
        "discount_sensitivity": V57
        / "sequencing_parity/cost_duration_discount_sensitivity.csv",
        "natural_pairs": V62
        / "natural_year_heatmap/source_data/natural_year_transition_pairs.csv",
        "matched_decades": V62
        / "natural_year_heatmap/source_data/matched_ten_year_transitions.csv",
        "efc_24h": V62 / "figure_work/source_data_v60/figure3_24h_value_records.csv",
        "objective_records": V62
        / "figure_work/source_data_v58/harmonized_parity/three_regime_system_year_duration_records.csv.gz",
        "duration_boundaries": V62
        / "cluster_intensity_surface/source_data/v60_cluster_intensity_boundaries.csv",
        "duration_surface": V62
        / "cluster_intensity_surface/source_data/v60_cluster_intensity_surface.csv",
        "recharge_blocks": V62
        / "figure_work/figure4_relayout_v62/outputs/figure4_relayout_block_summary_source.csv",
        "recharge_composition": V62
        / "figure_work/figure4_relayout_v62/outputs/figure4_relayout_response_composition_source.csv",
        "fixed_procurement": V44 / "joint_choice/fixed_power_paired_decision_flips.csv",
        "fixed_procurement_summary": V44
        / "v44_analysis/decision_flips/portfolio_choice_decision_flip_summary.csv",
        "bounded_procurement": V44
        / "joint_choice/bounded_s20_highs/bounded_joint_choice_summary_all_years.csv",
        "capacity_scan": V44 / "anchor_scan/capacity_product_flip_curve.csv",
        "anchor_summary": V44 / "anchor_scan/anchor_scan_summary.csv",
        "anchor_choices": V44 / "anchor_scan/anchor_scan_paired_decisions.csv.gz",
        "actual_cost_valleys": V44
        / "figure_work/source_data/figure6_panel_a_actual_cost_year_valleys.csv",
        "conditional_valleys": V44
        / "figure_work/source_data/figure6_panel_b_cost_firm_phase.csv",
        "price_fixed": V57
        / "price_channel/empirical_summary_v0_1/price_channel_caes24_summary.csv",
        "price_fixed_flips": V57
        / "price_channel/empirical_summary_v0_1/price_channel_decision_summary.csv",
        "price_reoptimized": V57
        / "price_channel/empirical_reoptimized_summary_v0_1/s3b_caes24_summary.csv",
        "price_reoptimized_duration": V57
        / "price_channel/empirical_reoptimized_summary_v0_1/s3b_duration_summary.csv",
        "price_reoptimized_flips": V57
        / "price_channel/empirical_reoptimized_summary_v0_1/s3b_choice_flips.csv",
        "natural_fixed_effects": V57 / "joint_choice/natural_fixed_effect_coefficients.csv",
        "natural_fixed_effects_qa": V57 / "joint_choice/natural_fixed_effect_qa.json",
    }

    # Cost-scale sensitivity and sequencing-parity counts.
    discount = read_csv(inputs["discount_sensitivity"])
    thresholds = discount.loc[np.isclose(discount["caes_duration_h"], 24.0), [
        "discount_rate",
        "absolute_relative_cost_movement_eur_kw_year",
    ]].sort_values("discount_rate")
    natural_pairs = read_csv(inputs["natural_pairs"])
    matched_decades = read_csv(inputs["matched_decades"])
    efc = read_csv(inputs["efc_24h"])
    objective = read_csv(inputs["objective_records"])
    objective = objective.loc[np.isclose(objective["duration_h"], 24.0)].copy()

    threshold_rows: list[dict[str, object]] = []
    for row in thresholds.itertuples(index=False):
        rate = float(row.discount_rate)
        threshold = float(row.absolute_relative_cost_movement_eur_kw_year)
        out: dict[str, object] = {
            "discount_rate": rate,
            "cost_threshold_eur_kw_year": threshold,
            "natural_pair_hits": int(
                natural_pairs["absolute_weather_value_change_eur_kw_year"].ge(threshold).sum()
            ),
            "natural_pairs": int(len(natural_pairs)),
            "adverse_decade_hits": int(
                matched_decades["weather_value_change_eur_kw_year"].le(-threshold).sum()
            ),
            "beneficial_decade_hits": int(
                matched_decades["weather_value_change_eur_kw_year"].ge(threshold).sum()
            ),
            "matched_decades": int(len(matched_decades)),
        }
        for regime in ("Europe", "China", "Japan"):
            efc_effect = efc.loc[
                efc["regime"].eq(regime), "signed_sequencing_effect_eur_kw_year"
            ]
            objective_effect = objective.loc[
                objective["regime"].eq(regime), "signed_sequencing_effect_eur_kw_year"
            ]
            out[f"efc_{regime.lower()}_hits"] = int(efc_effect.abs().ge(threshold).sum())
            out[f"objective_{regime.lower()}_hits"] = int(
                objective_effect.abs().ge(threshold).sum()
            )
        threshold_rows.append(out)
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(HERE / "cost_threshold_parity_counts.csv", index=False)

    # Four-day duration boundaries and the 12-day stress endpoint.
    boundaries = read_csv(inputs["duration_boundaries"])
    surface = read_csv(inputs["duration_surface"])
    duration_rows: list[dict[str, object]] = []
    for regime in ("Europe", "China", "Japan"):
        four = boundaries.loc[
            boundaries["regime"].eq(regime)
            & boundaries["target_cluster_days"].eq(4)
        ].iloc[0]
        twelve = boundaries.loc[
            boundaries["regime"].eq(regime)
            & boundaries["target_cluster_days"].eq(12)
        ].iloc[0]
        crossing_h = float(four["first_sustained_positive_duration_h"])
        crossing = surface.loc[
            surface["regime"].eq(regime)
            & surface["target_cluster_days"].eq(4)
            & np.isclose(surface["duration_h"], crossing_h)
        ].iloc[0]
        duration_rows.append(
            {
                "regime": regime,
                "four_day_trough_duration_h": float(four["trough_duration_h"]),
                "four_day_trough_effect_eur_kw_year": float(four["trough_effect_eur_kw_year"]),
                "four_day_first_sustained_positive_h": crossing_h,
                "boundary_block_mean_eur_kw_year": float(
                    crossing["block_mean_effect_eur_kw_year"]
                ),
                "boundary_ci95_low": float(crossing["ci95_low"]),
                "boundary_ci95_high": float(crossing["ci95_high"]),
                "twelve_day_trough_duration_h": float(twelve["trough_duration_h"]),
                "twelve_day_trough_effect_eur_kw_year": float(
                    twelve["trough_effect_eur_kw_year"]
                ),
                "twelve_day_first_sustained_positive_h": twelve[
                    "first_sustained_positive_duration_h"
                ],
            }
        )
    duration_table = pd.DataFrame(duration_rows)
    duration_table.to_csv(HERE / "duration_boundary_summary.csv", index=False)

    # Expanded-recharge result and response composition.
    recharge = read_csv(inputs["recharge_blocks"])
    composition = read_csv(inputs["recharge_composition"])
    recharge_rows: list[dict[str, object]] = []
    for regime in ("Europe", "China", "Japan"):
        one = recharge.loc[
            recharge["regime"].eq(regime) & recharge["recharge_multiplier"].eq(1)
        ].iloc[0]
        four = recharge.loc[
            recharge["regime"].eq(regime) & recharge["recharge_multiplier"].eq(4)
        ].iloc[0]
        counts = (
            composition.loc[
                composition["regime"].eq(regime)
                & composition["recharge_multiplier"].eq(4)
            ]
            .set_index("response_state")["n_system_years"]
            .to_dict()
        )
        recharge_rows.append(
            {
                "regime": regime,
                "one_x_block_mean_eur_kw_year": float(
                    one["country_block_mean_effect_eur_kw_year"]
                ),
                "one_x_ci95_low": float(one["ci95_low"]),
                "one_x_ci95_high": float(one["ci95_high"]),
                "four_x_block_mean_eur_kw_year": float(
                    four["country_block_mean_effect_eur_kw_year"]
                ),
                "four_x_ci95_low": float(four["ci95_low"]),
                "four_x_ci95_high": float(four["ci95_high"]),
                "effect_retained_pct": float(four["effect_retained_pct"]),
                "deeper_loss_count": int(counts["deeper loss"]),
                "unchanged_count": int(counts["unchanged"]),
                "weaker_loss_count": int(counts["weaker loss"]),
            }
        )
    recharge_table = pd.DataFrame(recharge_rows)
    recharge_table.to_csv(HERE / "recharge_robustness_summary.csv", index=False)

    # Procurement and value-channel branches declared in Methods.
    audit_rows: list[dict[str, object]] = []
    fixed = read_csv(inputs["fixed_procurement"])
    fixed_voluntary = fixed.loc[
        fixed["account"].eq("efc_value") & fixed["decision_scope"].eq("voluntary_tail")
    ]
    fixed_counts = {
        int(year): int(frame["disjoint_tie_set_flip"].sum())
        for year, frame in fixed_voluntary.groupby("cost_year")
    }
    fixed_mandatory = fixed.loc[
        fixed["account"].eq("efc_value") & fixed["decision_scope"].eq("mandatory_tail")
    ]
    mandatory_counts = {
        int(year): int(frame["technology_flip"].sum())
        for year, frame in fixed_mandatory.groupby("cost_year")
    }
    fixed_summary = read_csv(inputs["fixed_procurement_summary"])
    fixed_summary = fixed_summary.loc[
        fixed_summary["decision_domain"].eq("all_49_ldes_configurations")
    ]
    best_ldes_duration_counts = {
        int(row.cost_year): int(row.shadow_ldes_duration_flip_count)
        for row in fixed_summary.itertuples(index=False)
    }
    audit_rows.append(
        {
            "branch": "Fixed-power catalogue procurement",
            "sample_design": "132 European country-years; 2025/2030/2035 costs; 0.01 EUR kW-1 yr-1 tie tolerance",
            "result": (
                f"voluntary-tail identity flips {count_string(fixed_counts)}/132; "
                f"mandatory-tail technology flips {count_string(mandatory_counts)}/132; "
                f"best-LDES duration changes {count_string(best_ldes_duration_counts)}/132"
            ),
            "inference": "chronology can cross the procurement boundary, with the largest count under 2035 costs",
            "source_key": "fixed_procurement",
        }
    )

    bounded = read_csv(inputs["bounded_procurement"])

    def bounded_counts(account: str, metric: str) -> dict[int, int]:
        selected = bounded.loc[
            bounded["account"].eq(account) & bounded["metric"].eq(metric)
        ]
        return {
            int(row.learning_year): int(row.count)
            for row in selected.itertuples(index=False)
        }

    bounded_identity = bounded_counts("residual_peak_system_cost", "identity_flip")
    bounded_participation = bounded_counts(
        "residual_peak_system_cost", "participation_flip"
    )
    bounded_power = bounded_counts("residual_peak_system_cost", "material_power_change")
    bounded_any = bounded_counts(
        "residual_peak_system_cost", "any_material_procurement_change"
    )
    bounded_efc = bounded_counts("efc_equivalent_trajectory_screen", "identity_flip")
    audit_rows.append(
        {
            "branch": "Bounded-power joint procurement",
            "sample_design": "132 European country-years; power, technology, duration and dispatch jointly optimized under system cost",
            "result": (
                f"objective identity flips {count_string(bounded_identity)}/132; "
                f"participation {count_string(bounded_participation)}/132; material-power "
                f"changes {count_string(bounded_power)}/132; any material procurement "
                f"{count_string(bounded_any)}/132; EFC trajectory-screen identities "
                f"{count_string(bounded_efc)}/132"
            ),
            "inference": "allowing tail power to adjust does not remove chronology-dependent identity changes",
            "source_key": "bounded_procurement",
        }
    )

    capacity = read_csv(inputs["capacity_scan"])
    observed = capacity.loc[capacity["within_official_capacity_price_band"]].copy()
    ranges: dict[int, tuple[int, int]] = {}
    for year, frame in observed.groupby("cost_year"):
        ranges[int(year)] = (
            int(frame["selected_config_flip_count"].min()),
            int(frame["selected_config_flip_count"].max()),
        )
    audit_rows.append(
        {
            "branch": "Capacity-product value scan",
            "sample_design": "official four-priority-market band, 15.6946-90.4316 EUR kW-1 yr-1",
            "result": "configuration-flip ranges "
            + "/".join(
                f"{ranges[y][0]}-{ranges[y][1]}" for y in (2025, 2030, 2035)
            )
            + " of 132",
            "inference": "the observed capacity-price band preserves zero to modest flips and reaches 15 cases in 2035",
            "source_key": "capacity_scan",
        }
    )

    anchor_summary = read_csv(inputs["anchor_summary"])
    voll_summary = anchor_summary.loc[
        anchor_summary["branch_id"].eq("B_country_voll_direct_reliability")
    ]
    voll_duration_flips = {
        int(row.cost_year): int(row.selected_duration_flip_count)
        for row in voll_summary.itertuples(index=False)
    }
    voll_entry_flips = {
        int(row.cost_year): int(row.tail_entry_exit_flip_count)
        for row in voll_summary.itertuples(index=False)
    }
    choices = read_csv(inputs["anchor_choices"], low_memory=False)
    voll_choices = choices.loc[
        choices["branch_id"].eq("B_country_voll_direct_reliability")
    ]
    voll_dispersed_median = float(voll_choices["selected_duration_h_dispersed"].median())
    voll_clustered_median = float(voll_choices["selected_duration_h_clustered"].median())
    audit_rows.append(
        {
            "branch": "Direct-VoLL reliability ledger",
            "sample_design": "132 European country-years; country-specific VoLL; energy and firm-capacity adders set to zero",
            "result": (
                f"duration flips {count_string(voll_duration_flips)}/132; tail entry/exit "
                f"{count_string(voll_entry_flips)}/132; dispersed/clustered median selected "
                f"duration {voll_dispersed_median:.0f}/{voll_clustered_median:.0f} h"
            ),
            "inference": "the reliability ledger changes selected duration without changing tail participation",
            "source_key": "anchor_summary;anchor_choices",
        }
    )

    price = read_csv(inputs["price_fixed"])
    price_flips = read_csv(inputs["price_fixed_flips"])
    price_labels = {
        "S0": "fixed",
        "S1_beta_1": "scarcity beta=1",
        "S2_surplus_linked_charge": "surplus-linked charge",
        "S3a_empirical_price": "empirical day-ahead",
    }
    for scenario, label in price_labels.items():
        result = price.loc[price["scenario"].eq(scenario)].iloc[0]
        flips = price_flips.loc[price_flips["scenario"].eq(scenario)].iloc[0]
        audit_rows.append(
            {
                "branch": f"Fixed-dispatch price channel: {label}",
                "sample_design": "69 common European country-years",
                "result": (
                    f"24 h effect {result.mean_effect_eur_kw_year:.2f} "
                    f"[{result.ci95_low:.2f}, {result.ci95_high:.2f}] EUR kW-1 yr-1; "
                    f"identity flips {int(flips.identity_flips)}/69"
                ),
                "inference": "the mean effect remains negative across the pre-specified fixed-dispatch price channels",
                "source_key": "price_fixed;price_fixed_flips",
            }
        )

    reoptimized = read_csv(inputs["price_reoptimized"]).iloc[0]
    reoptimized_duration = read_csv(inputs["price_reoptimized_duration"])
    long_duration_response = reoptimized_duration.loc[
        reoptimized_duration["duration_h"].ge(24.0)
    ]
    positive_mean_h = float(
        long_duration_response.loc[
            long_duration_response["mean_effect_eur_kw_year"].gt(0), "duration_h"
        ].iloc[0]
    )
    positive_ci_h = float(
        long_duration_response.loc[
            long_duration_response["ci95_low"].gt(0), "duration_h"
        ].iloc[0]
    )
    reoptimized_flips = read_csv(inputs["price_reoptimized_flips"])
    audit_rows.append(
        {
            "branch": "Empirical-price responsive dispatch",
            "sample_design": "39 country-years in Austria, Germany, Denmark and France",
            "result": (
                f"24 h effect {reoptimized.mean_effect_eur_kw_year:.2f} "
                f"[{reoptimized.ci95_low:.2f}, {reoptimized.ci95_high:.2f}] EUR kW-1 yr-1; "
                f"mean/CI-positive boundary {positive_mean_h:.0f}/{positive_ci_h:.0f} h; "
                f"identity flips {int(reoptimized_flips['identity_flip'].sum())}/39"
            ),
            "inference": "price-responsive operation retains the negative 24 h effect and shifts procurement identities in a minority of cases",
            "source_key": "price_reoptimized;price_reoptimized_duration;price_reoptimized_flips",
        }
    )

    fixed_effects = read_csv(inputs["natural_fixed_effects"])
    zero_only = fixed_effects.loc[
        fixed_effects["model_id"].eq("zero_window_only")
        & fixed_effects["predictor"].eq("natural_zero_recovery_window_share")
    ].iloc[0]
    adjusted = fixed_effects.loc[
        fixed_effects["model_id"].eq("joint_geometry_plus_scale")
        & fixed_effects["predictor"].eq("natural_zero_recovery_window_share")
    ].iloc[0]
    audit_rows.append(
        {
            "branch": "Natural-year country fixed effects",
            "sample_design": "132 European country-years; 15 country blocks; 10,000 bootstrap draws",
            "result": (
                f"zero-window-only: {zero_only.coefficient / 10:.2f} "
                f"[{zero_only.bootstrap_ci95_low / 10:.2f}, {zero_only.bootstrap_ci95_high / 10:.2f}]; "
                f"geometry+scale adjusted: {adjusted.coefficient / 10:.2f} "
                f"[{adjusted.bootstrap_ci95_low / 10:.2f}, {adjusted.bootstrap_ci95_high / 10:.2f}] "
                "EUR kW-1 yr-1 per 10 percentage points"
            ),
            "inference": "a larger zero-recovery share remains negatively associated with natural-year relative value after scale adjustment",
            "source_key": "natural_fixed_effects;natural_fixed_effects_qa",
        }
    )

    actual_valleys = read_csv(inputs["actual_cost_valleys"])
    conditional_valleys = read_csv(inputs["conditional_valleys"])
    conditional = conditional_valleys.loc[
        np.isclose(conditional_valleys["firm_value_eur_kw_year"], 575.0)
        & np.isclose(conditional_valleys["ldes_tail_fixed_cost_multiplier"], 0.7)
    ].iloc[0]
    conditional_count = int(round(float(conditional["cluster_induced_rate"]) * 132))
    audit_rows.append(
        {
            "branch": "Strict mid-duration valley",
            "sample_design": "technology-specific 2025/2030/2035 costs; out-of-range structural point reported separately",
            "result": (
                f"observed-cost valleys {int(actual_valleys.strict_valley_count.sum())} across all six chronology-year cells; "
                f"cluster-induced valleys {conditional_count}/132 only at (cost multiplier, firm value)=(0.7, 575)"
            ),
            "inference": "the strict positive-negative-positive valley is not supported under observed cost years and is explicitly conditional outside the observed anchor range",
            "source_key": "actual_cost_valleys;conditional_valleys",
        }
    )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(HERE / "declared_robustness_audit.csv", index=False)

    natural_fe_qa = json.loads(inputs["natural_fixed_effects_qa"].read_text(encoding="utf-8"))
    central = threshold_table.loc[np.isclose(threshold_table["discount_rate"], 0.036)].iloc[0]
    checks = {
        "central_cost_threshold_is_5p296": bool(
            np.isclose(central["cost_threshold_eur_kw_year"], 5.296038099660983)
        ),
        "central_natural_counts_match": bool(
            central["natural_pair_hits"] == 161 and central["adverse_decade_hits"] == 3
        ),
        "central_efc_counts_match": bool(
            [central[f"efc_{r}_hits"] for r in ("europe", "china", "japan")]
            == [68, 26, 11]
        ),
        "central_objective_counts_match": bool(
            [central[f"objective_{r}_hits"] for r in ("europe", "china", "japan")]
            == [52, 26, 7]
        ),
        "four_day_boundaries_match": bool(
            duration_table["four_day_first_sustained_positive_h"].tolist()
            == [96.0, 168.0, 48.0]
        ),
        "twelve_day_has_no_positive_boundary": bool(
            duration_table["twelve_day_first_sustained_positive_h"].isna().all()
        ),
        "four_x_regime_means_remain_negative": bool(
            recharge_table["four_x_block_mean_eur_kw_year"].lt(0).all()
        ),
        "fixed_procurement_counts_match": fixed_counts == {2025: 0, 2030: 3, 2035: 22},
        "best_ldes_duration_counts_match": best_ldes_duration_counts
        == {2025: 13, 2030: 12, 2035: 8},
        "bounded_objective_counts_match": bounded_identity
        == {2025: 2, 2030: 2, 2035: 3},
        "capacity_band_ranges_match": ranges
        == {2025: (0, 0), 2030: (0, 3), 2035: (0, 15)},
        "voll_duration_counts_match": voll_duration_flips
        == {2025: 125, 2030: 125, 2035: 125},
        "price_reoptimized_flips_match": int(reoptimized_flips["identity_flip"].sum()) == 9,
        "price_reoptimized_positive_boundary_matches": positive_mean_h == 96.0
        and positive_ci_h == 100.0,
        "natural_fixed_effect_qa_pass": natural_fe_qa.get("status") == "pass",
        "observed_cost_valleys_zero": int(actual_valleys["strict_valley_count"].sum()) == 0,
        "conditional_valley_count_matches": conditional_count == 92,
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "input_sha256": {key: sha256(path) for key, path in inputs.items()},
        "outputs": [
            "cost_threshold_parity_counts.csv",
            "duration_boundary_summary.csv",
            "recharge_robustness_summary.csv",
            "declared_robustness_audit.csv",
        ],
    }
    (HERE / "qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if qa["status"] != "pass":
        raise RuntimeError("Robustness evidence QA failed")


if __name__ == "__main__":
    main()
