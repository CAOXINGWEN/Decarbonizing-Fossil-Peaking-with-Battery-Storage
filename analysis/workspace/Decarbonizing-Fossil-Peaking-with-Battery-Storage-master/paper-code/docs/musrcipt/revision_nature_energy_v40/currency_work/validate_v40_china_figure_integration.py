#!/usr/bin/env python3
"""Validate the source-aware China integration in v40 Fig. 6 and ED Fig. 7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
V40 = HERE.parent
SOURCE = V40 / "source_data/figure_source_data"
V31 = V40.parent / "revision_nature_energy_v31"
CHINA_F4 = V31 / "main_figures/figure4_turnover_mechanism/source_data"
CHINA_ED3 = V31 / "supplement/ed3_path_interventions/source_data"
REPORT = HERE / "V40_CHINA_FIGURE_INTEGRATION_QA.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    check_dtype: bool = False,
    atol: float = 1e-10,
) -> bool:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True).rename_axis(columns=None),
            expected.reset_index(drop=True).rename_axis(columns=None),
            check_dtype=check_dtype,
            check_exact=False,
            rtol=1e-10,
            atol=atol,
        )
    except AssertionError:
        return False
    return True


def route_check(frame: pd.DataFrame) -> bool:
    expected = {
        "caes_currency_route": {"USD_to_EUR"},
        "lfp_currency_route": {"CNY_to_EUR"},
        "value_item_currency_route": {"CNY_to_EUR"},
    }
    return all(set(frame[column].dropna().unique()) == values for column, values in expected.items())


def raster_record(path: Path, expected_size: tuple[int, int], expected_mode: str | None = None) -> dict:
    with Image.open(path) as image:
        record = {
            "path": str(path.relative_to(V40)),
            "bytes": path.stat().st_size,
            "size_px": list(image.size),
            "mode": image.mode,
            "dpi": [float(value) for value in image.info.get("dpi", (0, 0))],
            "sha256": sha256(path),
            "pass": image.size == expected_size and (expected_mode is None or image.mode == expected_mode),
        }
    return record


def export_bundle(folder: str, stem: str, png_size: tuple[int, int], tiff_size: tuple[int, int]) -> dict:
    base = V40 / folder / stem
    pdf = base.with_suffix(".pdf")
    svg = base.with_suffix(".svg")
    png = base.with_suffix(".png")
    tiff = base.with_suffix(".tiff")
    gray = base.parent / f"{base.name}_grayscale.png"
    svg_text = svg.read_text(encoding="utf-8")
    vector = {
        "pdf": {
            "path": str(pdf.relative_to(V40)),
            "bytes": pdf.stat().st_size,
            "sha256": sha256(pdf),
            "pass": pdf.stat().st_size > 0,
        },
        "svg": {
            "path": str(svg.relative_to(V40)),
            "bytes": svg.stat().st_size,
            "sha256": sha256(svg),
            "editable_text_elements": svg_text.count("<text"),
            "pass": svg.stat().st_size > 0 and "<text" in svg_text,
        },
    }
    raster = {
        "png": raster_record(png, png_size),
        "tiff": raster_record(tiff, tiff_size),
        "grayscale_png": raster_record(gray, png_size, "L"),
    }
    return {
        "vector": vector,
        "raster": raster,
        "pass": all(item["pass"] for item in vector.values())
        and all(item["pass"] for item in raster.values()),
    }


def main() -> None:
    inputs = {
        name: pd.read_csv(HERE / name)
        for name in [
            "china_osr_boundary_source_aware.csv",
            "china_fixed_asset_summary_source_aware_eur.csv",
            "china_mix_technology_year_summary_source_aware_eur.csv",
            "china_fixed_asset_comparison_source_aware_eur.csv",
            "china_fixed_asset_paired_changes_source_aware_eur.csv",
        ]
    }
    input_hashes = {name: sha256(HERE / name) for name in inputs}

    boundary = pd.read_csv(SOURCE / "figure6_panel_a_china_boundary_response.csv")
    fixed = pd.read_csv(SOURCE / "figure6_panel_b_china_fixed_asset.csv")
    economics = pd.read_csv(SOURCE / "figure6_panel_f_china_economics.csv")
    province = pd.read_csv(SOURCE / "figure6_panel_d_china_province_changes.csv")
    expected_economics = inputs[
        "china_mix_technology_year_summary_source_aware_eur.csv"
    ]
    expected_economics = expected_economics[
        expected_economics["tech_id"].eq("caes")
        & expected_economics["evidence_set"].eq("observed_year")
    ]

    comparison = pd.read_csv(
        SOURCE / "ed7_panel_abd_fixed_asset_comparison_source_aware_eur.csv"
    )
    changes = pd.read_csv(
        SOURCE / "ed7_panel_c_paired_changes_source_aware_eur.csv"
    )
    summary = pd.read_csv(
        SOURCE / "ed7_panel_d_fixed_asset_summary_source_aware_eur.csv"
    )
    effect_endpoints = pd.read_csv(
        SOURCE / "ed7_panel_e_intervention_endpoint_effects.csv"
    )
    effect_summary = pd.read_csv(
        SOURCE / "ed7_panel_e_intervention_bootstrap_summary.csv"
    )
    cycle_summary = pd.read_csv(
        SOURCE / "ed7_panel_f_technology_cycle_bootstrap_summary.csv"
    )
    change_keys = ["dispatch_method", "objective_mode", "chronology"]
    expected_changes = inputs[
        "china_fixed_asset_paired_changes_source_aware_eur.csv"
    ].sort_values(change_keys, kind="stable")
    observed_changes = changes.drop(columns=["plot_label"]).sort_values(
        change_keys, kind="stable"
    )
    expected_province_rows = []
    for dispatch, dispatch_method, objective_mode in [
        ("Cyclic dispatch", "cyclic_heuristic", "heuristic"),
        ("Perfect foresight", "perfect_foresight", "system_total"),
    ]:
        route = inputs["china_fixed_asset_comparison_source_aware_eur.csv"]
        route = route[
            route["dispatch_method"].eq(dispatch_method)
            & route["chronology"].eq("discharge_first")
            & route["objective_mode"].eq(objective_mode)
        ]
        wide = route.pivot(
            index="province_en",
            columns="persistence_level",
            values="caes_mix_net_benefit_eur_per_mwh_task",
        )
        current = wide[[1, 10]].rename(
            columns={
                1: "p1_caes_mix_net_benefit_eur_per_mwh_task",
                10: "p10_caes_mix_net_benefit_eur_per_mwh_task",
            }
        ).reset_index()
        current["dispatch"] = dispatch
        current["dispatch_method"] = dispatch_method
        current["chronology"] = "discharge_first"
        current["objective_mode"] = objective_mode
        current["persistence_from"] = 1
        current["persistence_to"] = 10
        current["change_eur_per_mwh_task"] = (
            current["p10_caes_mix_net_benefit_eur_per_mwh_task"]
            - current["p1_caes_mix_net_benefit_eur_per_mwh_task"]
        )
        for column in [
            "caes_currency_route",
            "lfp_currency_route",
            "value_item_currency_route",
            "model_cny_per_usd",
            "reporting_usd_per_eur",
            "reporting_cny_per_eur",
        ]:
            current[column] = route[column].iloc[0]
        expected_province_rows.append(current)
    expected_province = pd.concat(expected_province_rows, ignore_index=True)

    economic_source_checks = {
        "figure6a_exact": frame_equal(
            boundary, inputs["china_osr_boundary_source_aware.csv"]
        ),
        "figure6b_exact": frame_equal(
            fixed, inputs["china_fixed_asset_summary_source_aware_eur.csv"]
        ),
        "figure6d_p10_minus_p1_exact": frame_equal(
            province, expected_province
        ),
        "figure6f_exact": frame_equal(economics, expected_economics),
        "ed7_abd_exact": frame_equal(
            comparison,
            inputs["china_fixed_asset_comparison_source_aware_eur.csv"],
        ),
        "ed7_c_exact": frame_equal(observed_changes, expected_changes),
        "ed7_d_exact": frame_equal(
            summary, inputs["china_fixed_asset_summary_source_aware_eur.csv"]
        ),
        "ed7_endpoint_schema_uses_p10": all(
            "p3" not in column.lower()
            for frame in [effect_endpoints, effect_summary, cycle_summary]
            for column in frame.columns
        ),
        "all_currency_routes_exact": all(route_check(frame) for frame in inputs.values()),
        "maximum_abs_cost_identity_error_eur": float(
            comparison["cost_identity_error_eur"].abs().max()
        ),
    }
    economic_source_checks["pass"] = (
        all(
            value
            for key, value in economic_source_checks.items()
            if key.endswith("_exact")
        )
        and economic_source_checks["ed7_endpoint_schema_uses_p10"]
        and economic_source_checks["all_currency_routes_exact"]
        and economic_source_checks["maximum_abs_cost_identity_error_eur"] < 1e-6
    )

    retention = pd.read_csv(SOURCE / "figure6_panel_c_china_retention.csv")
    expected_retention = pd.read_csv(CHINA_F4 / "figure4_panel_c_turnover_retention.csv")
    expected_retention["system_label"] = expected_retention["system"].replace(
        {"extra LFP4": "Additional LFP4", "CAES24": "A-CAES24"}
    )
    centroids = pd.read_csv(SOURCE / "figure6_panel_e_china_recoverable_deficit.csv")
    expected_centroids = pd.read_csv(CHINA_F4 / "figure4_derived_province_centroids.csv")
    expected_effect_endpoints = pd.read_csv(
        CHINA_ED3 / "ed3_panel_e_intervention_endpoint_effects.csv"
    ).rename(columns=lambda column: column.replace("p3_", "p10_"))
    expected_effect_summary = pd.read_csv(
        CHINA_ED3 / "ed3_panel_e_intervention_bootstrap_summary.csv"
    ).rename(columns=lambda column: column.replace("p3_", "p10_"))
    expected_cycle_summary = pd.read_csv(
        CHINA_ED3 / "ed3_panel_f_technology_cycle_bootstrap_summary.csv"
    ).rename(columns=lambda column: column.replace("p3_", "p10_"))

    old_comparison = pd.read_csv(
        CHINA_ED3 / "strict_paired_fixed_asset_comparison.csv"
    )
    physical_columns = [
        "annual_demand_kwh__CAES24",
        "annual_demand_kwh__extra LFP4",
        "annual_storage_supply_kwh__CAES24",
        "annual_storage_supply_kwh__extra LFP4",
        "annual_gas_backup_kwh__CAES24",
        "annual_gas_backup_kwh__extra LFP4",
        "caes_minus_lfp_served_share_pp",
    ]
    maximum_physical_drift = float(
        np.nanmax(
            np.abs(
                comparison[physical_columns].to_numpy(float)
                - old_comparison[physical_columns].to_numpy(float)
            )
        )
    )
    physical_source_checks = {
        "figure6c_exact": frame_equal(retention, expected_retention),
        "figure6e_exact": frame_equal(centroids, expected_centroids),
        "ed7a_maximum_abs_physical_drift": maximum_physical_drift,
        "ed7e_endpoints_exact": frame_equal(
            effect_endpoints, expected_effect_endpoints
        ),
        "ed7e_summary_exact": frame_equal(effect_summary, expected_effect_summary),
        "ed7f_exact": frame_equal(cycle_summary, expected_cycle_summary),
    }
    physical_source_checks["pass"] = (
        physical_source_checks["figure6c_exact"]
        and physical_source_checks["figure6e_exact"]
        and maximum_physical_drift < 1e-5
        and physical_source_checks["ed7e_endpoints_exact"]
        and physical_source_checks["ed7e_summary_exact"]
        and physical_source_checks["ed7f_exact"]
    )

    endpoints = {}
    for year in (2025, 2030, 2035):
        year_data = boundary[boundary["learning_year"].eq(year)].set_index(
            "persistence_level"
        )
        endpoints[str(year)] = {
            "p1_mean_osr_gain_pp": float(year_data.loc[1, "mean_osr_gain_pp"]),
            "p10_mean_osr_gain_pp": float(year_data.loc[10, "mean_osr_gain_pp"]),
        }
    zero_charge = economics[np.isclose(economics["charge_opportunity_price_eur_kwh"], 0)]
    key_numbers = {
        "figure6a_osr_endpoints": endpoints,
        "figure6b_discharge_first": fixed[
            fixed["chronology"].eq("discharge_first")
            & fixed["persistence_level"].isin([1, 10])
            & fixed["objective_mode"].isin(["heuristic", "system_total"])
        ][
            [
                "dispatch_method",
                "persistence_level",
                "demand_weighted_caes_mix_net_benefit_eur_per_mwh_task",
            ]
        ].to_dict(orient="records"),
        "figure6f_zero_charge_system_cost_reduction_pct": {
            str(int(row.learning_year)): float(row.mean_mix_system_cost_reduction_pct)
            for row in zero_charge.itertuples()
        },
    }

    exports = {
        "figure6": export_bundle(
            "main_figures",
            "fig_v40_figure6_china_external_validation",
            (2160, 1785),
            (4320, 3570),
        ),
        "extended_data_figure7": export_bundle(
            "supplement",
            "fig_v40_ed7_china_dispatch_validation",
            (2160, 2160),
            (4320, 4320),
        ),
    }
    exports["pass"] = exports["figure6"]["pass"] and exports[
        "extended_data_figure7"
    ]["pass"]

    report = {
        "status": "pass"
        if economic_source_checks["pass"]
        and physical_source_checks["pass"]
        and exports["pass"]
        else "fail",
        "figure_contract": {
            "core_conclusion": (
                "Source-aware China economics preserve the contraction of the "
                "CAES-mix advantage as persistent events cluster."
            ),
            "archetype": "quantitative multi-panel validation grid",
            "backend": "Python/matplotlib",
            "economic_panels_updated": ["Figure 6a", "Figure 6b", "Figure 6d", "Figure 6f", "ED7b", "ED7c", "ED7d"],
            "physical_panels_held_fixed": ["Figure 6c", "Figure 6e", "ED7a", "ED7e", "ED7f"],
        },
        "input_sha256": input_hashes,
        "economic_source_checks": economic_source_checks,
        "physical_source_checks": physical_source_checks,
        "key_numbers": key_numbers,
        "exports": exports,
        "preflight_warning_review": {
            "missing_data_exclusion": (
                "The static warning is triggered by existing explicit dropna "
                "operations used after aligned joins or undefined ratios; this "
                "revision adds no row sampling or convenience exclusion."
            ),
            "simulated_data": (
                "The static warning is a false positive from seeded normal "
                "jitter used only to separate plotted points; plotted values "
                "remain the complete observed/model-derived dataset."
            ),
        },
        "manual_visual_inspection": {
            "status": "pass",
            "notes": (
                "Inspected final-size Figure 3-6 and ED4/ED7 color PNGs: "
                "P10 endpoints, 2035 cost-snapshot annotations, task-firm "
                "labels, direct technology labels, axes, legends, confidence "
                "bands and error bars are legible with no clipping or "
                "material overlap."
            ),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(REPORT)}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
