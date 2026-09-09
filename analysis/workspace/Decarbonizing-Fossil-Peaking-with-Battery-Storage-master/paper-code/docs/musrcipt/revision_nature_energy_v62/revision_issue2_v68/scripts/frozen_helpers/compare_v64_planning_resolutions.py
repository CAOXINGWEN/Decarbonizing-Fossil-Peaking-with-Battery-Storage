#!/usr/bin/env python3
"""Compare 2 h, 4 h and 6 h procurement solutions for manuscript v64."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTIVE_TOLERANCE = 1.0e-6
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
        number = float(value)
        return number if np.isfinite(number) else None
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
    for resolution in (2, 4, 6):
        parser.add_argument(f"--pooled-{resolution}h", type=Path, required=True)
        parser.add_argument(f"--chronology-{resolution}h", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_passed(root: Path, expected_step: int) -> dict[str, Any]:
    qa = json.loads((root / "qa.json").read_text(encoding="utf-8"))
    if qa.get("status") != "pass":
        raise RuntimeError(f"Input has not passed QA: {root}")
    if int(qa.get("planning_step_hours", expected_step)) != expected_step:
        raise RuntimeError(f"Planning-step mismatch: {root}")
    return qa


def sign(values: pd.Series, tolerance: float = 1.0e-9) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    return np.where(array > tolerance, 1, np.where(array < -tolerance, -1, 0))


def active_set(group: pd.DataFrame) -> str:
    selected = group.loc[
        group["capacity_per_normalized_mw"].gt(ACTIVE_TOLERANCE), "config_id"
    ].astype(str)
    return "+".join(sorted(selected)) if len(selected) else "none"


def dominant(group: pd.DataFrame) -> str:
    row = group.sort_values(
        ["capacity_per_normalized_mw", "config_id"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    return (
        str(row["config_id"])
        if float(row["capacity_per_normalized_mw"]) > ACTIVE_TOLERANCE
        else "none"
    )


def pooled_features(root: Path, resolution: int) -> pd.DataFrame:
    summaries = pd.read_csv(root / "design_summary.csv")
    capacities = pd.read_csv(root / "capacity_mix.csv")
    summaries = summaries.loc[summaries["mode"].eq("diversified")].copy()
    capacities = capacities.loc[capacities["mode"].eq("diversified")].copy()
    rows: list[dict[str, Any]] = []
    for (country, risk_weight), group in capacities.groupby(
        ["country", "risk_weight"], sort=True
    ):
        summary = summaries.loc[
            summaries["country"].eq(country)
            & summaries["risk_weight"].eq(risk_weight)
        ]
        if len(summary) != 1:
            raise RuntimeError(f"Incomplete pooled design: {country}, {risk_weight}")
        rows.append(
            {
                "country": str(country),
                "risk_weight": float(risk_weight),
                f"power_{resolution}h_pp": 100.0 * float(
                    summary.iloc[0]["total_power_capacity_per_normalized_mw"]
                ),
                f"active_set_{resolution}h": active_set(group),
                f"dominant_{resolution}h": dominant(group),
            }
        )
    return pd.DataFrame(rows)


def risk_shifts(pooled: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for country, group in pooled.groupby("country", sort=True):
        indexed = group.set_index("risk_weight")
        row: dict[str, Any] = {"country": country}
        for resolution in (2, 4, 6):
            row[f"risk_minus_expected_{resolution}h_pp"] = float(
                indexed.loc[0.5, f"power_{resolution}h_pp"]
                - indexed.loc[0.0, f"power_{resolution}h_pp"]
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["direction_agreement_4h_2h"] = (
        sign(result["risk_minus_expected_4h_pp"])
        == sign(result["risk_minus_expected_2h_pp"])
    )
    result["direction_agreement_6h_2h"] = (
        sign(result["risk_minus_expected_6h_pp"])
        == sign(result["risk_minus_expected_2h_pp"])
    )
    result["direction_agreement_all_three"] = (
        (sign(result["risk_minus_expected_2h_pp"])
         == sign(result["risk_minus_expected_4h_pp"]))
        & (sign(result["risk_minus_expected_2h_pp"])
           == sign(result["risk_minus_expected_6h_pp"]))
    )
    return result


def chronology_features(root: Path, resolution: int) -> pd.DataFrame:
    frame = pd.read_csv(root / "ensemble_comparison.csv")
    required = {
        "country", "risk_weight", "ensemble",
        "total_power_capacity_percent_mean_gas", "active_set",
        "dominant_technology",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Chronology table is missing fields: {root}")
    frame = frame[list(required)].copy()
    return frame.rename(
        columns={
            "total_power_capacity_percent_mean_gas": f"power_{resolution}h_pp",
            "active_set": f"active_set_{resolution}h",
            "dominant_technology": f"dominant_{resolution}h",
        }
    )


def direct_chronology(root: Path, resolution: int) -> pd.DataFrame:
    frame = pd.read_csv(root / "direct_chronology_effect.csv")
    columns = [
        "country", "risk_weight", "clustered_minus_dispersed_total_power_pp"
    ]
    return frame[columns].rename(
        columns={
            "clustered_minus_dispersed_total_power_pp":
                f"clustered_minus_dispersed_{resolution}h_pp"
        }
    )


def add_resolution_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["power_2h_minus_4h_pp"] = result["power_2h_pp"] - result["power_4h_pp"]
    result["power_4h_minus_6h_pp"] = result["power_4h_pp"] - result["power_6h_pp"]
    result["abs_power_2h_minus_4h_pp"] = result["power_2h_minus_4h_pp"].abs()
    result["abs_power_4h_minus_6h_pp"] = result["power_4h_minus_6h_pp"].abs()
    result["stepwise_power_contraction"] = (
        result["abs_power_2h_minus_4h_pp"]
        < result["abs_power_4h_minus_6h_pp"]
    )
    power_columns = ["power_2h_pp", "power_4h_pp", "power_6h_pp"]
    result["power_range_2h_4h_6h_pp"] = (
        result[power_columns].max(axis=1) - result[power_columns].min(axis=1)
    )
    result["active_set_agreement_4h_2h"] = result["active_set_4h"].eq(
        result["active_set_2h"]
    )
    result["active_set_agreement_6h_2h"] = result["active_set_6h"].eq(
        result["active_set_2h"]
    )
    result["active_set_agreement_all_three"] = (
        result["active_set_2h"].eq(result["active_set_4h"])
        & result["active_set_2h"].eq(result["active_set_6h"])
    )
    result["dominant_agreement_4h_2h"] = result["dominant_4h"].eq(
        result["dominant_2h"]
    )
    result["dominant_agreement_6h_2h"] = result["dominant_6h"].eq(
        result["dominant_2h"]
    )
    result["dominant_agreement_all_three"] = (
        result["dominant_2h"].eq(result["dominant_4h"])
        & result["dominant_2h"].eq(result["dominant_6h"])
    )
    return result


def grouped_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "design_count": len(frame),
        "median_within_design_power_range_2h_4h_6h_pp": float(
            frame["power_range_2h_4h_6h_pp"].median()
        ),
        "median_absolute_2h_minus_4h_power_pp": float(
            frame["abs_power_2h_minus_4h_pp"].median()
        ),
        "median_absolute_4h_minus_6h_power_pp": float(
            frame["abs_power_4h_minus_6h_pp"].median()
        ),
        "stepwise_power_contraction_designs": int(
            frame["stepwise_power_contraction"].sum()
        ),
        "active_set_agreement_4h_2h": int(
            frame["active_set_agreement_4h_2h"].sum()
        ),
        "active_set_agreement_6h_2h": int(
            frame["active_set_agreement_6h_2h"].sum()
        ),
        "active_set_agreement_all_three": int(
            frame["active_set_agreement_all_three"].sum()
        ),
        "dominant_agreement_4h_2h": int(
            frame["dominant_agreement_4h_2h"].sum()
        ),
        "dominant_agreement_6h_2h": int(
            frame["dominant_agreement_6h_2h"].sum()
        ),
        "dominant_agreement_all_three": int(
            frame["dominant_agreement_all_three"].sum()
        ),
    }


def command(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def sci(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def write_macros(path: Path, headline: dict[str, Any], qa2: dict[str, Any]) -> None:
    pooled = headline["pooled_all_designs"]
    ensemble = headline["all_chronology_ensembles"]
    event_expected = headline["event_order"]["expected"]
    event_risk = headline["event_order"]["mean_cvar90"]
    lines = [
        "% Generated by compare_v64_planning_resolutions.py.",
        command("ResolutionTwoFourRiskDirectionAgreement", f"{headline['risk_pricing']['direction_agreement_4h_2h']}/15"),
        command("ResolutionSixTwoRiskDirectionAgreement", f"{headline['risk_pricing']['direction_agreement_6h_2h']}/15"),
        command("ResolutionTwoRiskMaterialCount", f"{headline['risk_pricing']['material_power_change_2h']}/15"),
        command("ResolutionTwoFourPooledMedianAbs", f"{pooled['median_absolute_2h_minus_4h_power_pp']:.2f}"),
        command("ResolutionFourSixPooledMedianAbs", f"{pooled['median_absolute_4h_minus_6h_power_pp']:.2f}"),
        command("ResolutionTwoFourPooledContractionCount", f"{pooled['stepwise_power_contraction_designs']}/30"),
        command("ResolutionTwoFourPooledActiveSetAgreement", f"{pooled['active_set_agreement_4h_2h']}/30"),
        command("ResolutionSixTwoPooledActiveSetAgreement", f"{pooled['active_set_agreement_6h_2h']}/30"),
        command("ResolutionTwoFourAllMedianAbs", f"{ensemble['median_absolute_2h_minus_4h_power_pp']:.2f}"),
        command("ResolutionFourSixAllMedianAbs", f"{ensemble['median_absolute_4h_minus_6h_power_pp']:.2f}"),
        command("ResolutionTwoFourAllContractionCount", f"{ensemble['stepwise_power_contraction_designs']}/120"),
        command("ResolutionTwoFourAllActiveSetAgreement", f"{ensemble['active_set_agreement_4h_2h']}/120"),
        command("ResolutionSixTwoAllActiveSetAgreement", f"{ensemble['active_set_agreement_6h_2h']}/120"),
        command("ResolutionAllThreePooledActiveSetAgreement", f"{pooled['active_set_agreement_all_three']}/30"),
        command("ResolutionAllThreeActiveSetAgreement", f"{ensemble['active_set_agreement_all_three']}/120"),
        command("ResolutionAllThreePooledMedianPowerRange", f"{pooled['median_within_design_power_range_2h_4h_6h_pp']:.2f}"),
        command("ResolutionAllThreeMedianPowerRange", f"{ensemble['median_within_design_power_range_2h_4h_6h_pp']:.2f}"),
        command("ResolutionAllThreeRiskDirectionAgreement", f"{headline['risk_pricing']['direction_agreement_all_three']}/15"),
        command("ResolutionGridRiskMaterialRange", headline["risk_pricing"]["material_power_change_range"]),
        command("ResolutionGridEventExpectedMaterialRange", event_expected["material_power_change_range"]),
        command("ResolutionGridEventRiskMaterialRange", event_risk["material_power_change_range"]),
        command("ResolutionAllThreeEventExpectedDirectionAgreement", f"{event_expected['direction_agreement_all_three']}/15"),
        command("ResolutionAllThreeEventRiskDirectionAgreement", f"{event_risk['direction_agreement_all_three']}/15"),
        command("ResolutionTwoFourEventExpectedDirectionAgreement", f"{event_expected['direction_agreement_4h_2h']}/15"),
        command("ResolutionTwoFourEventRiskDirectionAgreement", f"{event_risk['direction_agreement_4h_2h']}/15"),
        command("ResolutionTwoEventExpectedMaterialCount", f"{event_expected['material_power_change_2h']}/15"),
        command("ResolutionTwoEventRiskMaterialCount", f"{event_risk['material_power_change_2h']}/15"),
        command("ResolutionTwoObjectiveError", sci(qa2["maximum_objective_identity_error_eur_per_normalized_mw"])),
        command("ResolutionTwoTaskResidual", sci(qa2["maximum_task_balance_residual"])),
        command("ResolutionTwoRechargeResidual", sci(qa2["maximum_shared_recharge_violation"])),
        command("ResolutionTwoSocResidual", sci(qa2["maximum_state_balance_residual_mwh"])),
        command("ResolutionTwoTaskEnergyResidual", sci(qa2["maximum_chronology_task_energy_residual"])),
        command("ResolutionTwoRechargeEnergyResidual", sci(qa2["maximum_chronology_recharge_energy_residual"])),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    pooled_roots = {
        resolution: getattr(args, f"pooled_{resolution}h").resolve()
        for resolution in (2, 4, 6)
    }
    chronology_roots = {
        resolution: getattr(args, f"chronology_{resolution}h").resolve()
        for resolution in (2, 4, 6)
    }
    pooled_qas = {
        resolution: read_passed(root, resolution)
        for resolution, root in pooled_roots.items()
    }
    chronology_qas = {
        resolution: read_passed(root, resolution)
        for resolution, root in chronology_roots.items()
    }
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise RuntimeError("Output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    pooled_parts = [pooled_features(pooled_roots[r], r) for r in (2, 4, 6)]
    pooled = pooled_parts[0]
    for part in pooled_parts[1:]:
        pooled = pooled.merge(part, on=["country", "risk_weight"], validate="one_to_one")
    pooled = add_resolution_metrics(pooled)
    if set(pooled["country"]) != EXPECTED_COUNTRIES or len(pooled) != 30:
        raise RuntimeError("Pooled resolution panel is incomplete")
    pooled.to_csv(output / "pooled_resolution_convergence.csv", index=False)

    shifts = risk_shifts(pooled)
    shifts.to_csv(output / "risk_pricing_direction_convergence.csv", index=False)

    chronology_parts = [
        chronology_features(chronology_roots[r], r) for r in (2, 4, 6)
    ]
    chronology = chronology_parts[0]
    keys = ["country", "risk_weight", "ensemble"]
    for part in chronology_parts[1:]:
        chronology = chronology.merge(part, on=keys, validate="one_to_one")
    chronology = add_resolution_metrics(chronology)
    if set(chronology["country"]) != EXPECTED_COUNTRIES or len(chronology) != 120:
        raise RuntimeError("Four-ensemble resolution panel is incomplete")
    chronology.to_csv(output / "chronology_ensemble_resolution_convergence.csv", index=False)

    direct_parts = [direct_chronology(chronology_roots[r], r) for r in (2, 4, 6)]
    direct = direct_parts[0]
    direct_keys = ["country", "risk_weight"]
    for part in direct_parts[1:]:
        direct = direct.merge(part, on=direct_keys, validate="one_to_one")
    direct["direction_agreement_4h_2h"] = (
        sign(direct["clustered_minus_dispersed_4h_pp"])
        == sign(direct["clustered_minus_dispersed_2h_pp"])
    )
    direct["direction_agreement_6h_2h"] = (
        sign(direct["clustered_minus_dispersed_6h_pp"])
        == sign(direct["clustered_minus_dispersed_2h_pp"])
    )
    direct["direction_agreement_all_three"] = (
        (sign(direct["clustered_minus_dispersed_2h_pp"])
         == sign(direct["clustered_minus_dispersed_4h_pp"]))
        & (sign(direct["clustered_minus_dispersed_2h_pp"])
           == sign(direct["clustered_minus_dispersed_6h_pp"]))
    )
    direct.to_csv(output / "event_order_direction_convergence.csv", index=False)

    headline: dict[str, Any] = {
        "design": "paired deterministic 2 h, 4 h and 6 h re-optimization",
        "inferential_tests": "none; countries are fixed model cases",
        "country_count": len(EXPECTED_COUNTRIES),
        "pooled_all_designs": grouped_metrics(pooled),
        "all_chronology_ensembles": grouped_metrics(chronology),
        "risk_pricing": {
            "direction_agreement_4h_2h": int(shifts["direction_agreement_4h_2h"].sum()),
            "direction_agreement_6h_2h": int(shifts["direction_agreement_6h_2h"].sum()),
            "direction_agreement_all_three": int(
                shifts["direction_agreement_all_three"].sum()
            ),
            "material_power_change_2h": int(
                shifts["risk_minus_expected_2h_pp"].abs().gt(1.0).sum()
            ),
            "material_power_change_4h": int(
                shifts["risk_minus_expected_4h_pp"].abs().gt(1.0).sum()
            ),
            "material_power_change_6h": int(
                shifts["risk_minus_expected_6h_pp"].abs().gt(1.0).sum()
            ),
        },
        "event_order": {},
        "by_ensemble_and_objective": {},
    }
    risk_material_counts = [
        headline["risk_pricing"][f"material_power_change_{resolution}h"]
        for resolution in (2, 4, 6)
    ]
    headline["risk_pricing"]["material_power_change_range"] = (
        f"{min(risk_material_counts)}--{max(risk_material_counts)}"
    )
    for risk_weight, label in ((0.0, "expected"), (0.5, "mean_cvar90")):
        subset = direct.loc[direct["risk_weight"].eq(risk_weight)]
        material_counts = {
            resolution: int(
                subset[f"clustered_minus_dispersed_{resolution}h_pp"]
                .abs().gt(1.0).sum()
            )
            for resolution in (2, 4, 6)
        }
        headline["event_order"][label] = {
            "direction_agreement_4h_2h": int(subset["direction_agreement_4h_2h"].sum()),
            "direction_agreement_6h_2h": int(subset["direction_agreement_6h_2h"].sum()),
            "direction_agreement_all_three": int(
                subset["direction_agreement_all_three"].sum()
            ),
            "material_power_change_2h": material_counts[2],
            "material_power_change_4h": material_counts[4],
            "material_power_change_6h": material_counts[6],
            "material_power_change_range": (
                f"{min(material_counts.values())}--{max(material_counts.values())}"
            ),
        }
    for (ensemble, risk_weight), group in chronology.groupby(
        ["ensemble", "risk_weight"], sort=True
    ):
        label = "expected" if risk_weight == 0.0 else "mean_cvar90"
        headline["by_ensemble_and_objective"][f"{ensemble}_{label}"] = grouped_metrics(group)
    write_json(output / "headline_metrics.json", headline)
    write_macros(
        output / "resolution_convergence_macros.tex",
        headline,
        pooled_qas[2],
    )

    source_paths = {
        f"pooled_{resolution}h": root / "qa.json"
        for resolution, root in pooled_roots.items()
    }
    source_paths.update(
        {
            f"chronology_{resolution}h": root / "qa.json"
            for resolution, root in chronology_roots.items()
        }
    )
    qa = {
        "status": "pass",
        "source_qa_statuses": {
            **{f"pooled_{r}h": pooled_qas[r]["status"] for r in (2, 4, 6)},
            **{f"chronology_{r}h": chronology_qas[r]["status"] for r in (2, 4, 6)},
        },
        "country_count": int(pooled["country"].nunique()),
        "pooled_design_count": len(pooled),
        "chronology_ensemble_design_count": len(chronology),
        "direct_event_order_design_count": len(direct),
        "source_qa_sha256": {name: sha256(path) for name, path in source_paths.items()},
    }
    output_names = (
        "pooled_resolution_convergence.csv",
        "risk_pricing_direction_convergence.csv",
        "chronology_ensemble_resolution_convergence.csv",
        "event_order_direction_convergence.csv",
        "headline_metrics.json",
        "resolution_convergence_macros.tex",
    )
    qa["output_sha256"] = {name: sha256(output / name) for name in output_names}
    write_json(output / "qa.json", qa)
    print(json.dumps(json_ready({"qa": qa, "headline": headline}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
