#!/usr/bin/env python3
"""Compare paired 4 h and 6 h v62 capacity-expansion solutions descriptively."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTIVE_TOLERANCE = 1.0e-6
MATERIAL_THRESHOLD_PP = 1.0
EXPECTED_COUNTRIES = {
    "AT", "BE", "BG", "CZ", "DE", "DK", "ES", "FR", "GB", "GR",
    "HR", "IT", "PT", "RO", "SI",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-4h", type=Path, required=True)
    parser.add_argument("--pooled-6h", type=Path, required=True)
    parser.add_argument("--chronology-4h", type=Path, required=True)
    parser.add_argument("--chronology-6h", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_passed(root: Path) -> dict[str, Any]:
    qa = json.loads((root / "qa.json").read_text(encoding="utf-8"))
    if qa.get("status") != "pass":
        raise RuntimeError(f"Input has not passed QA: {root}")
    return qa


def sign_agreement(left: pd.Series, right: pd.Series, tolerance: float = 1e-9) -> pd.Series:
    left_sign = np.where(left > tolerance, 1, np.where(left < -tolerance, -1, 0))
    right_sign = np.where(right > tolerance, 1, np.where(right < -tolerance, -1, 0))
    return pd.Series(left_sign == right_sign, index=left.index)


def design_features(root: Path) -> pd.DataFrame:
    summaries = pd.read_csv(root / "design_summary.csv")
    capacities = pd.read_csv(root / "capacity_mix.csv")
    summaries = summaries.loc[summaries["mode"].eq("diversified")].copy()
    rows: list[dict[str, Any]] = []
    for (country, risk_weight), group in capacities.loc[
        capacities["mode"].eq("diversified")
    ].groupby(["country", "risk_weight"], sort=True):
        active = group.loc[
            group["capacity_per_normalized_mw"].gt(ACTIVE_TOLERANCE)
        ].copy()
        dominant = group.sort_values(
            ["capacity_per_normalized_mw", "config_id"],
            ascending=[False, True], kind="stable"
        ).iloc[0]
        summary = summaries.loc[
            summaries["country"].eq(country)
            & summaries["risk_weight"].eq(risk_weight)
        ].iloc[0]
        rows.append({
            "country": str(country),
            "risk_weight": float(risk_weight),
            "total_power_percent_mean_gas": 100.0 * float(
                summary["total_power_capacity_per_normalized_mw"]
            ),
            "total_energy_hours_mean_gas": float(
                summary["total_energy_capacity_mwh_per_normalized_mw"]
            ),
            "risk_adjusted_cost_eur_per_normalized_mw": float(
                summary["risk_adjusted_cost_eur_per_normalized_mw"]
            ),
            "active_set": "+".join(sorted(active["config_id"].astype(str))) or "none",
            "selected_technology_count": int(len(active)),
            "dominant_technology": str(dominant["config_id"]),
        })
    return pd.DataFrame(rows)


def paired_pooled(root4: Path, root6: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature4 = design_features(root4).add_suffix("_4h").rename(columns={
        "country_4h": "country", "risk_weight_4h": "risk_weight"
    })
    feature6 = design_features(root6).add_suffix("_6h").rename(columns={
        "country_6h": "country", "risk_weight_6h": "risk_weight"
    })
    paired = feature6.merge(feature4, on=["country", "risk_weight"], validate="one_to_one")
    paired["power_4h_minus_6h_pp"] = (
        paired["total_power_percent_mean_gas_4h"]
        - paired["total_power_percent_mean_gas_6h"]
    )
    paired["absolute_power_difference_pp"] = paired["power_4h_minus_6h_pp"].abs()
    paired["power_within_1pp"] = paired["absolute_power_difference_pp"].le(
        MATERIAL_THRESHOLD_PP
    )
    paired["active_set_agreement"] = paired["active_set_4h"].eq(paired["active_set_6h"])
    paired["dominant_technology_agreement"] = paired[
        "dominant_technology_4h"
    ].eq(paired["dominant_technology_6h"])
    paired["cost_4h_minus_6h_eur_per_normalized_mw"] = (
        paired["risk_adjusted_cost_eur_per_normalized_mw_4h"]
        - paired["risk_adjusted_cost_eur_per_normalized_mw_6h"]
    )

    shifts4 = pd.read_csv(root4 / "procurement_design_shifts.csv").add_suffix("_4h").rename(
        columns={"country_4h": "country"}
    )
    shifts6 = pd.read_csv(root6 / "procurement_design_shifts.csv").add_suffix("_6h").rename(
        columns={"country_6h": "country"}
    )
    shifts = shifts6.merge(shifts4, on="country", validate="one_to_one")
    shifts["risk_shift_direction_agreement"] = sign_agreement(
        shifts["risk_aware_minus_expected_power_percentage_points_6h"],
        shifts["risk_aware_minus_expected_power_percentage_points_4h"],
    )
    shifts["risk_shift_4h_minus_6h_pp"] = (
        shifts["risk_aware_minus_expected_power_percentage_points_4h"]
        - shifts["risk_aware_minus_expected_power_percentage_points_6h"]
    )

    cap4 = pd.read_csv(root4 / "capacity_mix.csv")
    cap6 = pd.read_csv(root6 / "capacity_mix.csv")
    keys = ["country", "risk_weight", "mode", "config_id"]
    tech = cap6.merge(cap4, on=keys, suffixes=("_6h", "_4h"), validate="one_to_one")
    tech["capacity_4h_minus_6h_pp"] = 100.0 * (
        tech["capacity_per_normalized_mw_4h"]
        - tech["capacity_per_normalized_mw_6h"]
    )
    return paired, shifts, tech


def paired_diversification(root4: Path, root6: Path) -> pd.DataFrame:
    keys = ["country", "risk_weight"]
    frame4 = pd.read_csv(root4 / "diversification_comparison.csv")
    frame6 = pd.read_csv(root6 / "diversification_comparison.csv")
    paired = frame6.merge(frame4, on=keys, suffixes=("_6h", "_4h"), validate="one_to_one")
    paired["best_single_agreement"] = paired["best_single_mode_6h"].eq(
        paired["best_single_mode_4h"]
    )
    expected = paired["risk_weight"].eq(0.0)
    risk = paired["risk_weight"].eq(0.5)
    paired["relevant_diversification_change_6h"] = np.where(
        expected,
        paired["diversified_minus_best_single_mean_cost_eur_per_normalized_mw_6h"],
        paired["diversified_minus_best_single_cvar90_cost_eur_per_normalized_mw_6h"],
    )
    paired["relevant_diversification_change_4h"] = np.where(
        expected,
        paired["diversified_minus_best_single_mean_cost_eur_per_normalized_mw_4h"],
        paired["diversified_minus_best_single_cvar90_cost_eur_per_normalized_mw_4h"],
    )
    paired["diversification_benefit_agreement"] = (
        paired["relevant_diversification_change_6h"].lt(-1.0e-6)
        == paired["relevant_diversification_change_4h"].lt(-1.0e-6)
    )
    paired["relevant_change_4h_minus_6h_eur_per_normalized_mw"] = (
        paired["relevant_diversification_change_4h"]
        - paired["relevant_diversification_change_6h"]
    )
    if not (expected | risk).all():
        raise RuntimeError("Unexpected risk weight in diversification comparison")
    return paired


def paired_chronology(root4: Path, root6: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["country", "risk_weight"]
    direct4 = pd.read_csv(root4 / "direct_chronology_effect.csv")
    direct6 = pd.read_csv(root6 / "direct_chronology_effect.csv")
    direct = direct6.merge(direct4, on=keys, suffixes=("_6h", "_4h"), validate="one_to_one")
    for stem in (
        "clustered_minus_dispersed_total_power_pp",
        "combined_minus_observed_total_power_pp",
    ):
        direct[f"{stem}_direction_agreement"] = sign_agreement(
            direct[f"{stem}_6h"], direct[f"{stem}_4h"]
        )
        direct[f"{stem}_4h_minus_6h_pp"] = direct[f"{stem}_4h"] - direct[f"{stem}_6h"]

    ensemble4 = pd.read_csv(root4 / "ensemble_comparison.csv")
    ensemble6 = pd.read_csv(root6 / "ensemble_comparison.csv")
    ensemble_keys = ["country", "risk_weight", "ensemble"]
    ensemble = ensemble6.merge(
        ensemble4, on=ensemble_keys, suffixes=("_6h", "_4h"), validate="one_to_one"
    )
    ensemble["power_4h_minus_6h_pp"] = (
        ensemble["total_power_capacity_percent_mean_gas_4h"]
        - ensemble["total_power_capacity_percent_mean_gas_6h"]
    )
    ensemble["absolute_power_difference_pp"] = ensemble["power_4h_minus_6h_pp"].abs()
    ensemble["active_set_agreement"] = ensemble["active_set_4h"].eq(
        ensemble["active_set_6h"]
    )
    ensemble["dominant_technology_agreement"] = ensemble[
        "dominant_technology_4h"
    ].eq(ensemble["dominant_technology_6h"])
    return direct, ensemble


def grouped_pooled_metrics(paired: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for weight, label in ((0.0, "expected"), (0.5, "mean_cvar90")):
        subset = paired.loc[paired["risk_weight"].eq(weight)]
        result[label] = {
            "median_4h_minus_6h_power_pp": float(subset["power_4h_minus_6h_pp"].median()),
            "median_absolute_power_difference_pp": float(
                subset["absolute_power_difference_pp"].median()
            ),
            "maximum_absolute_power_difference_pp": float(
                subset["absolute_power_difference_pp"].max()
            ),
            "countries_within_1pp": int(subset["power_within_1pp"].sum()),
            "active_set_agreement_countries": int(subset["active_set_agreement"].sum()),
            "dominant_technology_agreement_countries": int(
                subset["dominant_technology_agreement"].sum()
            ),
            "median_4h_minus_6h_risk_adjusted_cost_eur_per_normalized_mw": float(
                subset["cost_4h_minus_6h_eur_per_normalized_mw"].median()
            ),
        }
    return result


def main() -> int:
    args = parse_args()
    roots = {
        "pooled_4h": args.pooled_4h.resolve(),
        "pooled_6h": args.pooled_6h.resolve(),
        "chronology_4h": args.chronology_4h.resolve(),
        "chronology_6h": args.chronology_6h.resolve(),
    }
    qas = {name: read_passed(root) for name, root in roots.items()}
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    pooled, shifts, tech = paired_pooled(roots["pooled_4h"], roots["pooled_6h"])
    diversification = paired_diversification(roots["pooled_4h"], roots["pooled_6h"])
    chronology, ensembles = paired_chronology(
        roots["chronology_4h"], roots["chronology_6h"]
    )
    pooled.to_csv(output / "pooled_country_pair.csv", index=False)
    shifts.to_csv(output / "risk_shift_country_pair.csv", index=False)
    tech.to_csv(output / "technology_capacity_pair.csv", index=False)
    diversification.to_csv(output / "diversification_country_pair.csv", index=False)
    chronology.to_csv(output / "chronology_effect_country_pair.csv", index=False)
    ensembles.to_csv(output / "chronology_ensemble_country_pair.csv", index=False)

    pooled4_head = json.loads((roots["pooled_4h"] / "headline_metrics.json").read_text())
    pooled6_head = json.loads((roots["pooled_6h"] / "headline_metrics.json").read_text())
    chron4_head = json.loads((roots["chronology_4h"] / "headline_metrics.json").read_text())
    chron6_head = json.loads((roots["chronology_6h"] / "headline_metrics.json").read_text())
    headline: dict[str, Any] = {
        "design": "paired deterministic re-optimization of the same 15 country cases",
        "inferential_tests": "none; countries are fixed model cases, not random replicates",
        "country_count": int(pooled["country"].nunique()),
        "pooled_capacity": grouped_pooled_metrics(pooled),
        "risk_pricing": {
            "material_power_change_countries_6h": int(pooled6_head["material_power_change_countries"]),
            "material_power_change_countries_4h": int(pooled4_head["material_power_change_countries"]),
            "direction_agreement_countries": int(shifts["risk_shift_direction_agreement"].sum()),
            "median_absolute_shift_6h_pp": float(
                shifts["risk_aware_minus_expected_power_percentage_points_6h"].abs().median()
            ),
            "median_absolute_shift_4h_pp": float(
                shifts["risk_aware_minus_expected_power_percentage_points_4h"].abs().median()
            ),
            "technology_set_change_countries_6h": int(pooled6_head["technology_set_change_countries"]),
            "technology_set_change_countries_4h": int(pooled4_head["technology_set_change_countries"]),
            "dominant_technology_change_countries_6h": int(pooled6_head["dominant_technology_change_countries"]),
            "dominant_technology_change_countries_4h": int(pooled4_head["dominant_technology_change_countries"]),
        },
        "portfolio_diversification": {
            "expected_cost_improvement_countries_6h": int(
                pooled6_head["expected_cost_diversification_improvement_countries"]
            ),
            "expected_cost_improvement_countries_4h": int(
                pooled4_head["expected_cost_diversification_improvement_countries"]
            ),
            "risk_cvar90_improvement_countries_6h": int(
                pooled6_head["risk_aware_cvar90_diversification_improvement_countries"]
            ),
            "risk_cvar90_improvement_countries_4h": int(
                pooled4_head["risk_aware_cvar90_diversification_improvement_countries"]
            ),
            "benefit_classification_agreement_designs": int(
                diversification["diversification_benefit_agreement"].sum()
            ),
            "best_single_technology_agreement_designs": int(
                diversification["best_single_agreement"].sum()
            ),
            "design_count": int(len(diversification)),
        },
        "chronology_isolation": {},
    }
    for weight, label in ((0.0, "expected"), (0.5, "mean_cvar90")):
        subset = chronology.loc[chronology["risk_weight"].eq(weight)]
        c4 = chron4_head[label]
        c6 = chron6_head[label]
        headline["chronology_isolation"][label] = {
            "clustered_vs_dispersed_material_countries_6h": int(
                c6["clustered_vs_dispersed_material_power_change_countries"]
            ),
            "clustered_vs_dispersed_material_countries_4h": int(
                c4["clustered_vs_dispersed_material_power_change_countries"]
            ),
            "clustered_vs_dispersed_direction_agreement_countries": int(
                subset["clustered_minus_dispersed_total_power_pp_direction_agreement"].sum()
            ),
            "clustered_vs_dispersed_median_absolute_power_6h_pp": float(
                c6["clustered_vs_dispersed_median_absolute_power_change_pp"]
            ),
            "clustered_vs_dispersed_median_absolute_power_4h_pp": float(
                c4["clustered_vs_dispersed_median_absolute_power_change_pp"]
            ),
            "clustered_vs_dispersed_technology_set_change_countries_6h": int(
                c6["clustered_vs_dispersed_technology_set_change_countries"]
            ),
            "clustered_vs_dispersed_technology_set_change_countries_4h": int(
                c4["clustered_vs_dispersed_technology_set_change_countries"]
            ),
            "clustered_vs_dispersed_dominant_change_countries_6h": int(
                c6["clustered_vs_dispersed_dominant_technology_change_countries"]
            ),
            "clustered_vs_dispersed_dominant_change_countries_4h": int(
                c4["clustered_vs_dispersed_dominant_technology_change_countries"]
            ),
            "combined_vs_observed_material_countries_6h": int(
                c6["combined_vs_observed_material_power_change_countries"]
            ),
            "combined_vs_observed_material_countries_4h": int(
                c4["combined_vs_observed_material_power_change_countries"]
            ),
            "combined_vs_observed_direction_agreement_countries": int(
                subset["combined_minus_observed_total_power_pp_direction_agreement"].sum()
            ),
        }
    write_json(output / "headline_metrics.json", headline)

    countries = set(pooled["country"].astype(str))
    planning_values = {
        "pooled_4h": set(pd.read_csv(roots["pooled_4h"] / "design_summary.csv")["planning_step_hours"]),
        "pooled_6h": set(pd.read_csv(roots["pooled_6h"] / "design_summary.csv")["planning_step_hours"]),
        "chronology_4h": set(pd.read_csv(roots["chronology_4h"] / "design_summary.csv")["planning_step_hours"]),
        "chronology_6h": set(pd.read_csv(roots["chronology_6h"] / "design_summary.csv")["planning_step_hours"]),
    }
    h2_max = {}
    for name, root in roots.items():
        caps = pd.read_csv(root / "capacity_mix.csv")
        h2_max[name] = float(caps.loc[
            caps["config_id"].astype(str).eq("h2_cavern_168h")
            & caps["mode"].eq("diversified"),
            "capacity_per_normalized_mw",
        ].abs().max())
    qa = {
        "status": "pass",
        "country_set": sorted(countries),
        "country_set_complete": countries == EXPECTED_COUNTRIES,
        "paired_pooled_design_count": int(len(pooled)),
        "paired_chronology_contrast_count": int(len(chronology)),
        "paired_ensemble_design_count": int(len(ensembles)),
        "planning_step_values": {key: sorted(map(float, value)) for key, value in planning_values.items()},
        "all_source_qas_pass": all(qa.get("status") == "pass" for qa in qas.values()),
        "maximum_diversified_hydrogen_capacity": h2_max,
        "source_qa_sha256": {name: sha256(root / "qa.json") for name, root in roots.items()},
    }
    qa["status"] = (
        "pass"
        if qa["country_set_complete"]
        and qa["paired_pooled_design_count"] == 30
        and qa["paired_chronology_contrast_count"] == 30
        and qa["paired_ensemble_design_count"] == 120
        and qa["planning_step_values"] == {
            "pooled_4h": [4.0], "pooled_6h": [6.0],
            "chronology_4h": [4.0], "chronology_6h": [6.0]
        }
        and qa["all_source_qas_pass"]
        else "fail"
    )
    qa["output_sha256"] = {
        name: sha256(output / name)
        for name in (
            "pooled_country_pair.csv", "risk_shift_country_pair.csv",
            "technology_capacity_pair.csv", "diversification_country_pair.csv",
            "chronology_effect_country_pair.csv", "chronology_ensemble_country_pair.csv",
            "headline_metrics.json",
        )
    }
    write_json(output / "qa.json", qa)
    print(json.dumps(json_ready({"headline": headline, "qa": qa}), ensure_ascii=False, indent=2))
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
