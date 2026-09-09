#!/usr/bin/env python3
"""Apply the R3 annual-curtailment allocation admission gate.

The fossil-gas task is not modified.  This audit only tests whether an ACER
2023 annual curtailment total can be allocated beneath the independently
constructed synchronous potential-VRE-surplus upper bound.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL = (
    ROOT
    / "processed"
    / "energy_charts_observed_gas_task_panel_2015_2024.csv.gz"
)
ACER = (
    ROOT
    / "raw"
    / "acer_2024_mmr"
    / "ACER_Figure42_RES_curtailment_2023.csv"
)
RESEARCH = ROOT / "artifacts" / "research"
PANEL_QA = RESEARCH / "energy_charts_observed_gas_task_panel_qa.json"
EXPECTED_PANEL_SHA256 = (
    "5f2e8ab49120050fac2ef8bd1b11e20a58863442631a8d9baccf5b924f0bf3ab"
)
EXPECTED_ACER_SHA256 = (
    "56bd86ad82ded7b238afeb796ac67df08ea87b2a3d291e7a096bbbb564ede59c"
)
ABSOLUTE_TOLERANCE_MWH = 1e-6
RELATIVE_TOLERANCE = 1e-10
HOURLY_BOUND_TOLERANCE_MW = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def main() -> None:
    columns = [
        "timestamp_utc",
        "country",
        "year",
        "system_load_mw",
        "observed_wind_mw",
        "observed_solar_mw",
        "observed_gas_peaking_task_mw",
        "load_mw",
        "gas_task_eligible",
        "gas_task_material_share_ge_1pct",
    ]
    panel = pd.read_csv(PANEL, usecols=columns)
    panel_qa = json.loads(PANEL_QA.read_text(encoding="utf-8"))
    panel_sha256 = sha256(PANEL)
    acer_sha256 = sha256(ACER)
    panel = panel[
        panel["year"].eq(2023)
        & as_bool(panel["gas_task_eligible"])
        & as_bool(panel["gas_task_material_share_ge_1pct"])
    ].copy()
    acer_all = pd.read_csv(ACER)
    acer_unique_country_year = bool(
        acer_all.groupby(["country_code", "year"]).size().eq(1).all()
    )
    acer = acer_all[
        acer_all["year"].eq(2023)
        & acer_all["res_curtailment_gwh"].gt(0)
    ].copy()
    annual_acer = dict(
        zip(
            acer["country_code"].astype(str),
            acer["res_curtailment_gwh"].astype(float) * 1000.0,
        )
    )

    records = []
    for country, group in panel.groupby("country", sort=True):
        if country not in annual_acer:
            continue
        task_identity_error = float(
            (
                group["load_mw"]
                - group["observed_gas_peaking_task_mw"]
            )
            .abs()
            .max()
        )
        hour_count = int(len(group))
        unique_timestamp_count = int(group["timestamp_utc"].nunique())
        hourly_calendar_gate = bool(
            hour_count == 8760 and unique_timestamp_count == 8760
        )
        observed_vre = (
            group["observed_wind_mw"].astype(float)
            + group["observed_solar_mw"].astype(float)
        )
        alpha = float(group["system_load_mw"].sum() / observed_vre.sum())
        potential = np.maximum(
            alpha * observed_vre.to_numpy(float)
            - group["system_load_mw"].to_numpy(float),
            0.0,
        )
        potential_mwh = float(potential.sum())
        annual_acer_mwh = float(annual_acer[country])
        positive_weights = bool(
            potential_mwh > 0.0 and np.any(potential > 0.0)
        )
        volume_gate = bool(
            annual_acer_mwh > 0.0
            and annual_acer_mwh <= potential_mwh
        )
        scale = (
            annual_acer_mwh / potential_mwh
            if positive_weights
            else np.nan
        )
        scale_gate = bool(np.isfinite(scale) and 0.0 < scale <= 1.0)
        calibrated = scale * potential if scale_gate else np.zeros_like(
            potential
        )
        annual_error = float(calibrated.sum() - annual_acer_mwh)
        relative_error = (
            abs(annual_error) / annual_acer_mwh
            if annual_acer_mwh > 0.0
            else np.inf
        )
        maximum_bound_violation = float(
            np.maximum(calibrated - potential, 0.0).max()
        )
        conservation_gate = bool(
            abs(annual_error) <= ABSOLUTE_TOLERANCE_MWH
            and relative_error <= RELATIVE_TOLERANCE
        )
        hourly_bound_gate = bool(
            maximum_bound_violation <= HOURLY_BOUND_TOLERANCE_MW
        )
        admitted = bool(
            positive_weights
            and volume_gate
            and scale_gate
            and conservation_gate
            and hourly_bound_gate
            and task_identity_error == 0.0
            and hourly_calendar_gate
        )
        records.append(
            {
                "country": country,
                "year": 2023,
                "hour_count": hour_count,
                "unique_timestamp_count": unique_timestamp_count,
                "hourly_calendar_gate_passed": hourly_calendar_gate,
                "annual_acer_curtailment_mwh": annual_acer_mwh,
                "annual_potential_vre_surplus_mwh": potential_mwh,
                "positive_potential_surplus_weights": positive_weights,
                "annual_volume_gate_passed": volume_gate,
                "acer_allocation_scale": scale,
                "allocation_scale_gate_passed": scale_gate,
                "acer_annual_allocation_error_mwh": annual_error,
                "acer_annual_allocation_relative_error": relative_error,
                "annual_conservation_gate_passed": conservation_gate,
                "acer_max_hourly_surplus_bound_violation_mw": (
                    maximum_bound_violation
                ),
                "hourly_upper_bound_gate_passed": hourly_bound_gate,
                "task_alias_identity_max_abs_error_mw": task_identity_error,
                "r3_admitted": admitted,
                "interpretation": (
                    "annual-volume-calibrated hourly allocation; not "
                    "observed hourly curtailment"
                ),
            }
        )

    output = pd.DataFrame(records).sort_values(["country", "year"])
    output_path = (
        RESEARCH
        / "acer_r3_observed_gas_candidate_admission.csv"
    )
    output.to_csv(output_path, index=False)
    expected_candidates = {"DE", "ES", "HR", "IT"}
    checks = {
        "candidate_intersection_is_DE_ES_HR_IT": (
            set(output["country"]) == expected_candidates
        ),
        "four_candidate_country_years": len(output) == 4,
        "frozen_task_panel_sha256_matches": (
            panel_sha256 == EXPECTED_PANEL_SHA256
        ),
        "frozen_acer_source_sha256_matches": (
            acer_sha256 == EXPECTED_ACER_SHA256
        ),
        "upstream_task_panel_qa_passed": bool(panel_qa["pass"]),
        "upstream_task_panel_qa_sha_matches": (
            panel_qa["output_sha256"] == panel_sha256
        ),
        "acer_country_year_rows_are_unique": acer_unique_country_year,
        "all_candidates_have_8760_unique_hours": bool(
            output["hourly_calendar_gate_passed"].all()
        ),
        "all_candidates_pass_positive_weight_gate": bool(
            output["positive_potential_surplus_weights"].all()
        ),
        "all_candidates_pass_annual_volume_gate": bool(
            output["annual_volume_gate_passed"].all()
        ),
        "all_allocation_scales_in_zero_one_interval": bool(
            output["allocation_scale_gate_passed"].all()
        ),
        "all_annual_allocations_conserved": bool(
            output["annual_conservation_gate_passed"].all()
        ),
        "all_hourly_allocations_below_potential_surplus": bool(
            output["hourly_upper_bound_gate_passed"].all()
        ),
        "fossil_gas_task_alias_unchanged": bool(
            output["task_alias_identity_max_abs_error_mw"].eq(0.0).all()
        ),
        "all_four_candidates_admitted": bool(output["r3_admitted"].all()),
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "task_panel": str(PANEL),
        "task_panel_sha256": panel_sha256,
        "expected_task_panel_sha256": EXPECTED_PANEL_SHA256,
        "upstream_task_panel_qa": str(PANEL_QA),
        "acer_source": str(ACER),
        "acer_source_sha256": acer_sha256,
        "expected_acer_source_sha256": EXPECTED_ACER_SHA256,
        "absolute_conservation_tolerance_mwh": (
            ABSOLUTE_TOLERANCE_MWH
        ),
        "relative_conservation_tolerance": RELATIVE_TOLERANCE,
        "hourly_bound_tolerance_mw": HOURLY_BOUND_TOLERANCE_MW,
        "admitted_country_years": output.loc[
            output["r3_admitted"], ["country", "year"]
        ].to_dict("records"),
        "interpretation": (
            "Passing R3 permits a bounded annual-volume-calibrated hourly "
            "recharge sensitivity. It does not create observed hourly "
            "curtailment and does not alter the actual fossil-gas task."
        ),
    }
    qa_path = (
        RESEARCH
        / "acer_r3_observed_gas_candidate_admission_qa.json"
    )
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(output.to_string(index=False))
    print(json.dumps(qa, indent=2))
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
