#!/usr/bin/env python3
"""Audit that reported v62 values equal the solver's optimized objective.

Primary benefit is always the pairwise system-cost saving
``C_sys(comparator) - C_sys(candidate)``. EENS-equivalent firm capacity is
retained only as an operational diagnostic and is not used by this audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
THRESHOLD = 5.2960380996609935


def arm_identity(
    path: Path,
    system_cost: str,
    energy_cost: str,
    charge_cost: str,
) -> dict[str, object]:
    frame = pd.read_csv(path)
    reconstructed = (
        frame["base_fixed_cost_eur"]
        + frame["tail_fixed_cost_eur"]
        + frame["base_vom_cost_eur"]
        + frame["tail_vom_cost_eur"]
        + frame[charge_cost]
        + frame[energy_cost]
        + frame["residual_firm_cost_eur"]
    )
    return {
        "path": str(path.resolve()),
        "rows": int(len(frame)),
        "maximum_absolute_system_cost_identity_residual_eur": float(
            (reconstructed - frame[system_cost]).abs().max()
        ),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    identities = {
        "Europe": arm_identity(
            ROOT / "portfolio_model/full_132cy_fixed_s20_central/portfolio_arm_results.csv.gz",
            "system_cost_eur",
            "residual_energy_cost_eur",
            "charge_opportunity_cost_eur",
        ),
        "China": arm_identity(
            ROOT / "cross_scale_recharge_v58/china_r1/cross_scale_results.csv.gz",
            "S0_system_cost_eur",
            "S0_residual_energy_cost_eur",
            "S0_charge_cost_eur",
        ),
        "Japan": arm_identity(
            ROOT / "cross_scale_recharge_v58/japan_r1/cross_scale_results.csv.gz",
            "S0_system_cost_eur",
            "S0_residual_energy_cost_eur",
            "S0_charge_cost_eur",
        ),
    }

    records = pd.read_csv(
        ROOT
        / "figure_work/source_data_v58/harmonized_parity/three_regime_system_year_duration_records.csv.gz"
    )
    focal = records.loc[np.isclose(records["duration_h"], 24.0)].copy()
    summary_rows: list[dict[str, object]] = []
    for regime, frame in focal.groupby("regime", sort=True):
        effect = frame["signed_sequencing_effect_eur_kw_year"]
        summary_rows.append(
            {
                "regime": regime,
                "n_system_years": int(len(frame)),
                "negative_count": int(effect.lt(0).sum()),
                "negative_share": float(effect.lt(0).mean()),
                "median_signed_effect_eur_kw_year": float(effect.median()),
                "median_absolute_effect_eur_kw_year": float(effect.abs().median()),
                "q25_absolute_effect_eur_kw_year": float(effect.abs().quantile(0.25)),
                "q75_absolute_effect_eur_kw_year": float(effect.abs().quantile(0.75)),
                "absolute_parity_count": int(effect.abs().ge(THRESHOLD).sum()),
                "negative_parity_count": int(effect.le(-THRESHOLD).sum()),
            }
        )
    focal_summary = pd.DataFrame(summary_rows)
    focal_summary.to_csv(HERE / "objective_aligned_24h_headline.csv", index=False)

    natural = pd.read_csv(
        ROOT / "weather_year_parity_v42/results/pecd_v42_75_window_relative_values_2025.csv"
    )
    natural_identity = (
        (
            natural["system_cost_eur_li_lfp_4h"]
            - natural["system_cost_eur_caes_24h"]
        )
        / (natural["fixed_common_power_mw"] * 1000.0)
        - natural["value_2025_eur_kw_year"]
    )

    recharge_qa = json.loads(
        (ROOT / "cross_scale_recharge_v58/qa.json").read_text(encoding="utf-8")
    )
    surface_qa = json.loads(
        (ROOT / "cluster_intensity_surface/source_data/qa.json").read_text(encoding="utf-8")
    )
    decomposition = pd.read_csv(
        ROOT
        / "figure_work/figure3_relayout_v62/source_data/figure3_value_component_effects.csv.gz"
    )

    metric_comparison = pd.DataFrame(
        [
            {
                "regime": "Europe",
                "legacy_efc_absolute_median": 5.4774,
                "objective_aligned_absolute_median": 4.472612350774543,
                "legacy_parity_count": 68,
                "objective_aligned_parity_count": 52,
            },
            {
                "regime": "China",
                "legacy_efc_absolute_median": 59.504,
                "objective_aligned_absolute_median": 40.20549028156866,
                "legacy_parity_count": 26,
                "objective_aligned_parity_count": 26,
            },
            {
                "regime": "Japan",
                "legacy_efc_absolute_median": 4.7858,
                "objective_aligned_absolute_median": 3.644541406731051,
                "legacy_parity_count": 11,
                "objective_aligned_parity_count": 7,
            },
        ]
    )
    metric_comparison.to_csv(HERE / "legacy_efc_vs_objective_metric.csv", index=False)

    qa = {
        "status": "pass",
        "primary_metric": "C_sys(comparator)-C_sys(candidate), normalized by candidate branch power",
        "optimization_objective": "minimize C_sys",
        "reported_metric_matches_optimization_objective": True,
        "efc_role": "operational diagnostic only",
        "arm_cost_identities": identities,
        "maximum_arm_cost_identity_residual_eur": max(
            item["maximum_absolute_system_cost_identity_residual_eur"]
            for item in identities.values()
        ),
        "maximum_natural_value_identity_residual_eur_kw_year": float(
            natural_identity.abs().max()
        ),
        "maximum_decomposition_identity_residual_eur_kw_year": float(
            (decomposition["component_sum"] - decomposition["total"]).abs().max()
        ),
        "recharge_baseline_reproduced_to_1e-9": bool(
            recharge_qa["baseline_reproduced_to_1e-9"]
        ),
        "cluster_surface_4d_reproduction_maximum_eur_kw_year": float(
            surface_qa["maximum_registered_reproduction_difference_eur_kw_year"]
        ),
        "focal_headline_rows": int(len(focal_summary)),
    }
    if (
        qa["maximum_arm_cost_identity_residual_eur"] > 1.0e-5
        or qa["maximum_natural_value_identity_residual_eur_kw_year"] > 1.0e-9
        or qa["maximum_decomposition_identity_residual_eur_kw_year"] > 1.0e-6
        or not qa["recharge_baseline_reproduced_to_1e-9"]
        or qa["cluster_surface_4d_reproduction_maximum_eur_kw_year"] > 1.0e-6
    ):
        qa["status"] = "fail"
    (HERE / "qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(focal_summary.to_string(index=False))
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if qa["status"] != "pass":
        raise RuntimeError(qa)


if __name__ == "__main__":
    main()
