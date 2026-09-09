#!/usr/bin/env python3
"""Independent source-aware re-costing of the frozen China validation.

CAES technology costs use model USD -> EUR (undo 7.2 CNY/USD, then divide by
1.13 USD/EUR). LFP technology costs use CNY -> EUR (divide by
8.1185043137255 CNY/EUR). Gas comparator/value items remain on their inherited
CNY basis and use CNY -> EUR. No v24/v28/v31/v39 file is modified.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MUSCRIPT = HERE.parents[1]
MASTER = HERE.parents[5]
V24_DIR = MUSCRIPT / "revision_nature_energy_v24" / "artifacts" / "mechanism_decisive_tests"
V24_RUNNER = V24_DIR / "run_v24_charge_power_and_foresight_tests.py"
V24_OSR_DETAIL = V24_DIR / "v24_cyclic_persistence_trajectory_by_province.csv"
V28_FIXED_DIR = MUSCRIPT / "revision_nature_energy_v28_mid_duration_valley" / "strict_paired_fixed_asset_economics"
V28_FIXED_RUNNER = V28_FIXED_DIR / "run_strict_paired_fixed_asset_economics_v28.py"
V28_MIX_DIR = MUSCRIPT / "revision_nature_energy_v28_mid_duration_valley" / "mix_economic_metrics"
V28_MIX_RUNNER = V28_MIX_DIR / "run_mix_economic_metrics_v28.py"
V31_F4 = MUSCRIPT / "revision_nature_energy_v31" / "main_figures" / "figure4_turnover_mechanism" / "source_data"
V31_F5 = MUSCRIPT / "revision_nature_energy_v31" / "main_figures" / "figure5_mix_economics" / "source_data"

MODEL_CNY_PER_USD = 7.2
ECB_USD_PER_EUR = 1.13
ECB_CNY_PER_EUR = 8.1185043137255
CAES_RELATIVE_TO_UNIFORM_CNY = ECB_CNY_PER_EUR / (
    MODEL_CNY_PER_USD * ECB_USD_PER_EUR
)
EUR_PER_CNY = 1.0 / ECB_CNY_PER_EUR
EUR_PER_MODEL_CNY_USD_ROUTE = 1.0 / (
    MODEL_CNY_PER_USD * ECB_USD_PER_EUR
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tech_with_factor(tech: Any, factor: float):
    return replace(
        tech,
        fixed_cost_cny_kwh_yr=float(tech.fixed_cost_cny_kwh_yr) * factor,
        variable_om_cny_mwh=float(tech.variable_om_cny_mwh) * factor,
        power_cost_cny_kw=float(tech.power_cost_cny_kw) * factor,
        energy_cost_cny_kwh=float(tech.energy_cost_cny_kwh) * factor,
        fixed_om_cny_kw_yr=float(tech.fixed_om_cny_kw_yr) * factor,
    )


def rename_currency_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            col: col.replace("_cny", "_eur").replace("cny_", "eur_")
            for col in frame.columns
            if "_cny" in col or "cny_" in col
        }
    )


def add_route_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["caes_currency_route"] = "USD_to_EUR"
    out["lfp_currency_route"] = "CNY_to_EUR"
    out["value_item_currency_route"] = "CNY_to_EUR"
    out["model_cny_per_usd"] = MODEL_CNY_PER_USD
    out["reporting_usd_per_eur"] = ECB_USD_PER_EUR
    out["reporting_cny_per_eur"] = ECB_CNY_PER_EUR
    return out


def recompute_osr(v24) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_load = v24.v14.load_techs
    original = original_load()
    source_aware = {
        key: (
            tech_with_factor(value, CAES_RELATIVE_TO_UNIFORM_CNY)
            if key[0] == "caes"
            else value
        )
        for key, value in original.items()
    }
    v24.v14.load_techs = lambda: source_aware
    try:
        manifest = v24.load_full_persistence_manifest()
        _, provinces = v24.load_inputs()
        detail, summary = v24.run_cyclic_persistence_trajectory(
            manifest, provinces, capacity_points=120
        )
    finally:
        v24.v14.load_techs = original_load

    old = pd.read_csv(V31_F4 / "figure4_panel_a_boundary_response.csv")
    keys = ["learning_year", "persistence_level"]
    comparison = old.merge(
        summary,
        on=keys,
        suffixes=("_uniform_v31", "_source_aware_v40"),
        validate="one_to_one",
    )
    comparison["mean_osr_gain_change_pp"] = (
        comparison["mean_osr_gain_pp_source_aware_v40"]
        - comparison["mean_osr_gain_pp_uniform_v31"]
    )
    comparison["ci_low_change_pp"] = (
        comparison["ci_low_source_aware_v40"]
        - comparison["ci_low_uniform_v31"]
    )
    comparison["ci_high_change_pp"] = (
        comparison["ci_high_source_aware_v40"]
        - comparison["ci_high_uniform_v31"]
    )
    old_detail = pd.read_csv(V24_OSR_DETAIL)
    detail_keys = ["province_en", "learning_year", "persistence_level"]
    detail_comparison = old_detail[detail_keys + [
        "extra_lfp4h_osr_pct", "caes24_osr_pct",
        "caes24_minus_lfp4h_osr_gain_pp",
    ]].merge(
        detail[detail_keys + [
            "extra_lfp4h_osr_pct", "caes24_osr_pct",
            "caes24_minus_lfp4h_osr_gain_pp",
        ]],
        on=detail_keys,
        suffixes=("_uniform_v31", "_source_aware_v40"),
        validate="one_to_one",
    )
    for metric in (
        "extra_lfp4h_osr_pct", "caes24_osr_pct",
        "caes24_minus_lfp4h_osr_gain_pp",
    ):
        detail_comparison[f"{metric}_change"] = (
            detail_comparison[f"{metric}_source_aware_v40"]
            - detail_comparison[f"{metric}_uniform_v31"]
        )
    old_gain = detail_comparison[
        "caes24_minus_lfp4h_osr_gain_pp_uniform_v31"
    ]
    new_gain = detail_comparison[
        "caes24_minus_lfp4h_osr_gain_pp_source_aware_v40"
    ]
    detail_comparison["osr_gain_sign_changed"] = (
        (old_gain * new_gain < 0.0)
        & (old_gain.abs() > 1e-9)
        & (new_gain.abs() > 1e-9)
    )
    return (
        add_route_metadata(detail),
        add_route_metadata(summary),
        add_route_metadata(comparison),
        add_route_metadata(detail_comparison),
    )


def recompute_fixed_asset(v24, fixed):
    context = fixed.load_context().copy()
    context["gas_parity_unit_cost_cny_kwh"] *= EUR_PER_CNY
    dispatch = fixed.normalize_dispatch()
    techs = v24.v14.load_techs()
    source_aware_eur = {
        key: tech_with_factor(
            tech,
            EUR_PER_MODEL_CNY_USD_ROUTE if key[0] == "caes" else EUR_PER_CNY,
        )
        for key, tech in techs.items()
    }
    original_load = v24.v14.load_techs
    v24.v14.load_techs = lambda: source_aware_eur
    try:
        costed = fixed.cost_dispatch(dispatch, context, v24)
    finally:
        v24.v14.load_techs = original_load
    comparison = fixed.compare_arms(costed)
    summary, paired = fixed.summarize(comparison)
    costed = add_route_metadata(rename_currency_columns(costed))
    comparison = add_route_metadata(rename_currency_columns(comparison))
    summary = add_route_metadata(rename_currency_columns(summary))
    paired = add_route_metadata(rename_currency_columns(paired))

    old_summary = pd.read_csv(V31_F4 / "figure4_panel_b_fixed_asset_summary.csv")
    keys = [
        "dispatch_method", "chronology", "objective_mode",
        "persistence_level",
    ]
    selected = [
        *keys,
        "demand_weighted_caes_mix_net_benefit_cny_per_mwh_task",
        "aggregate_lfp_system_cost_billion_cny",
        "aggregate_caes_system_cost_billion_cny",
    ]
    old = old_summary[selected].copy()
    for col in selected[len(keys):]:
        old[col.replace("_cny", "_eur").replace("cny_", "eur_")] = (
            old.pop(col) * EUR_PER_CNY
        )
    delta = old.merge(
        summary,
        on=keys,
        suffixes=("_uniform_v31", "_source_aware_v40"),
        validate="one_to_one",
    )
    metric = "demand_weighted_caes_mix_net_benefit_eur_per_mwh_task"
    delta[f"{metric}_change"] = (
        delta[f"{metric}_source_aware_v40"] - delta[f"{metric}_uniform_v31"]
    )
    return costed, comparison, summary, paired, add_route_metadata(delta)


def source_aware_mix_costs(mix) -> dict[tuple[str, int, int], dict[str, float]]:
    costs, _ = mix.load_costs()
    out: dict[tuple[str, int, int], dict[str, float]] = {}
    for key, row in costs.items():
        factor = (
            EUR_PER_MODEL_CNY_USD_ROUTE
            if key[0] == "caes"
            else EUR_PER_CNY
        )
        out[key] = {name: float(value) * factor for name, value in row.items()}
    return out


def recompute_mix_economics(mix, workers: int = 12):
    costs = source_aware_mix_costs(mix)
    jobs = mix.load_jobs(("caes",), None)
    old_prices = tuple(mix.CHARGE_PRICES)
    original_gas_components = mix.gas_components
    mix.CHARGE_PRICES = tuple(value * EUR_PER_CNY for value in old_prices)

    def gas_components_eur(gas_price_cny_m3: float, kappa: float):
        return tuple(
            value * EUR_PER_CNY
            for value in original_gas_components(gas_price_cny_m3, kappa)
        )

    mix.gas_components = gas_components_eur
    detail_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    bridge_rows: list[dict[str, object]] = []

    def task(record: dict[str, object]):
        return mix.process_job(record, costs)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(
                pool.map(task, jobs.to_dict("records")), start=1
            ):
                detail, curve, bridge = result
                detail_rows.extend(detail)
                curve_rows.extend(curve)
                bridge_rows.append(bridge)
                if index % 100 == 0 or index == len(jobs):
                    print(
                        f"[China source-aware economics] {index}/{len(jobs)} grids",
                        flush=True,
                    )
        detail = pd.DataFrame(detail_rows)
        curve = pd.DataFrame(curve_rows)
        bridge = pd.DataFrame(bridge_rows)
        paths = mix.aggregate_paths(detail)
        summary = mix.summarize_paths(paths)
        bootstrap = mix.province_block_bootstrap(detail)
        capacity_summary = mix.summarize_capacity_bridge(bridge)
        persistence, persistence_endpoints = mix.persistence_response(paths)
    finally:
        mix.gas_components = original_gas_components
        mix.CHARGE_PRICES = old_prices

    frames = {
        "china_mix_model_cells_source_aware_eur.csv.gz": detail,
        "china_mix_path_aggregates_source_aware_eur.csv": paths,
        "china_mix_technology_year_summary_source_aware_eur.csv": summary,
        "china_mix_province_bootstrap_source_aware_eur.csv": bootstrap,
        "china_mix_capacity_value_curves_source_aware_eur.csv.gz": curve,
        "china_mix_capacity_thresholds_source_aware_eur.csv": bridge,
        "china_mix_capacity_summary_source_aware_eur.csv": capacity_summary,
        "china_mix_persistence_response_source_aware_eur.csv": persistence,
        "china_mix_persistence_endpoints_source_aware_eur.csv": persistence_endpoints,
    }
    renamed: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        out = rename_currency_columns(frame)
        if "charge_opportunity_price_eur_kwh" in out:
            out["charge_opportunity_price_original_cny_kwh"] = (
                out["charge_opportunity_price_eur_kwh"] * ECB_CNY_PER_EUR
            )
        renamed[name] = add_route_metadata(out)

    old_summary = pd.read_csv(V31_F5 / "figure5_technology_year_summary.csv")
    old_summary = old_summary[
        old_summary["tech_id"].eq("caes")
    ].copy()
    old_summary = rename_currency_columns(old_summary)
    money_cols = [
        col for col in old_summary.columns
        if "_eur" in col and not col.endswith("_pct")
    ]
    for col in money_cols:
        old_summary[col] = old_summary[col] * EUR_PER_CNY
    new_summary = renamed[
        "china_mix_technology_year_summary_source_aware_eur.csv"
    ]
    keys = [
        "tech_id", "learning_year", "evidence_set",
        "charge_opportunity_price_eur_kwh",
    ]
    compare_cols = [
        "mean_mix_benefit_eur_per_mwh_task",
        "mean_mix_system_cost_reduction_pct",
        "mean_annual_mix_net_benefit_billion_eur",
        "mean_ldes_share_energy_capacity_pct",
        "mean_ldes_share_power_pct",
    ]
    delta = old_summary[keys + compare_cols].merge(
        new_summary[keys + compare_cols],
        on=keys,
        suffixes=("_uniform_v31", "_source_aware_v40"),
        validate="one_to_one",
    )
    for col in compare_cols:
        delta[f"{col}_change"] = (
            delta[f"{col}_source_aware_v40"] - delta[f"{col}_uniform_v31"]
        )
    renamed["china_mix_summary_comparison_v31_v40.csv"] = add_route_metadata(delta)
    old_detail = pd.read_csv(V28_MIX_DIR / "mix_economic_metrics_model_cells.csv")
    old_detail = old_detail[old_detail["tech_id"].eq("caes")].copy()
    old_detail = rename_currency_columns(old_detail)
    for col in [
        col for col in old_detail.columns
        if "_eur" in col and not col.endswith("_pct")
    ]:
        old_detail[col] = old_detail[col] * EUR_PER_CNY
    new_detail = renamed["china_mix_model_cells_source_aware_eur.csv.gz"]
    cell_keys = [
        "stem", "province_en", "learning_year", "evidence_set",
        "charge_opportunity_price_eur_kwh",
    ]
    cell_metrics = [
        "hybrid_selected_duration_h",
        "hybrid_base_lfp_capacity_mwh",
        "hybrid_ldes_capacity_mwh",
        "hybrid_storage_supply_kwh",
        "mix_incremental_net_benefit_eur",
        "mix_benefit_eur_per_mwh_task",
        "mix_system_cost_reduction_pct",
    ]
    cell_delta = old_detail[cell_keys + cell_metrics].merge(
        new_detail[cell_keys + cell_metrics],
        on=cell_keys,
        suffixes=("_uniform_v31", "_source_aware_v40"),
        validate="one_to_one",
    )
    selection_metrics = [
        "hybrid_selected_duration_h",
        "hybrid_base_lfp_capacity_mwh",
        "hybrid_ldes_capacity_mwh",
    ]
    cell_delta["selected_configuration_changed"] = np.logical_or.reduce([
        ~np.isclose(
            cell_delta[f"{col}_uniform_v31"],
            cell_delta[f"{col}_source_aware_v40"],
            equal_nan=True,
        )
        for col in selection_metrics
    ])
    for col in (
        "mix_incremental_net_benefit_eur",
        "mix_benefit_eur_per_mwh_task",
        "mix_system_cost_reduction_pct",
    ):
        cell_delta[f"{col}_change"] = (
            cell_delta[f"{col}_source_aware_v40"]
            - cell_delta[f"{col}_uniform_v31"]
        )
    renamed["china_mix_model_cell_comparison_v31_v40.csv.gz"] = (
        add_route_metadata(cell_delta)
    )
    return renamed


def dataframe_record(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    v24 = load_module("v40_china_v24", V24_RUNNER)
    fixed = load_module("v40_china_fixed", V28_FIXED_RUNNER)
    mix = load_module("v40_china_mix", V28_MIX_RUNNER)

    osr_detail, osr_summary, osr_comparison, osr_detail_comparison = (
        recompute_osr(v24)
    )
    fixed_costed, fixed_comparison, fixed_summary, fixed_paired, fixed_delta = (
        recompute_fixed_asset(v24, fixed)
    )
    mix_frames = recompute_mix_economics(mix)

    outputs: dict[str, pd.DataFrame] = {
        "china_osr_boundary_by_province_source_aware.csv": osr_detail,
        "china_osr_boundary_source_aware.csv": osr_summary,
        "china_osr_boundary_comparison_v31_v40.csv": osr_comparison,
        "china_osr_boundary_province_comparison_v31_v40.csv": osr_detail_comparison,
        "china_fixed_asset_arm_costs_source_aware_eur.csv": fixed_costed,
        "china_fixed_asset_comparison_source_aware_eur.csv": fixed_comparison,
        "china_fixed_asset_summary_source_aware_eur.csv": fixed_summary,
        "china_fixed_asset_paired_changes_source_aware_eur.csv": fixed_paired,
        "china_fixed_asset_summary_comparison_v31_v40.csv": fixed_delta,
        **mix_frames,
    }
    for name, frame in outputs.items():
        frame.to_csv(HERE / name, index=False)

    endpoints = osr_comparison[
        osr_comparison["persistence_level"].isin([1, 10])
    ][[
        "learning_year", "persistence_level",
        "mean_osr_gain_pp_uniform_v31",
        "mean_osr_gain_pp_source_aware_v40",
        "mean_osr_gain_change_pp",
    ]].sort_values(["learning_year", "persistence_level"])
    primary_fixed = fixed_summary[
        (
            fixed_summary["dispatch_method"].eq("cyclic_heuristic")
            & fixed_summary["chronology"].eq("discharge_first")
            & fixed_summary["objective_mode"].eq("heuristic")
        )
        | (
            fixed_summary["dispatch_method"].eq("perfect_foresight")
            & fixed_summary["chronology"].eq("discharge_first")
            & fixed_summary["objective_mode"].eq("system_total")
        )
    ].sort_values(["dispatch_method", "persistence_level"])
    mix_summary = mix_frames[
        "china_mix_technology_year_summary_source_aware_eur.csv"
    ]
    primary_mix = mix_summary[
        mix_summary["evidence_set"].eq("observed_year")
        & np.isclose(
            mix_summary["charge_opportunity_price_eur_kwh"], 0.0
        )
    ].sort_values("learning_year")
    old_fixed_comparison = pd.read_csv(
        V28_FIXED_DIR / "strict_paired_fixed_asset_comparison.csv"
    )
    physical_fixed_columns = [
        "annual_demand_kwh__CAES24",
        "annual_demand_kwh__extra LFP4",
        "annual_storage_supply_kwh__CAES24",
        "annual_storage_supply_kwh__extra LFP4",
        "annual_gas_backup_kwh__CAES24",
        "annual_gas_backup_kwh__extra LFP4",
        "caes_minus_lfp_served_share_pp",
    ]
    fixed_keys = [
        "province_en", "persistence_level", "chronology",
        "dispatch_method", "objective_mode", "learning_year",
    ]
    fixed_physical_audit = old_fixed_comparison[
        fixed_keys + physical_fixed_columns
    ].merge(
        fixed_comparison[fixed_keys + physical_fixed_columns],
        on=fixed_keys,
        suffixes=("_uniform_v31", "_source_aware_v40"),
        validate="one_to_one",
    )
    fixed_physical_max_change = max(
        float(np.max(np.abs(
            fixed_physical_audit[f"{col}_source_aware_v40"]
            - fixed_physical_audit[f"{col}_uniform_v31"]
        )))
        for col in physical_fixed_columns
    )
    model_cell_comparison = mix_frames[
        "china_mix_model_cell_comparison_v31_v40.csv.gz"
    ]

    checks = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "exchange_rates": {
            "model_cny_per_usd": MODEL_CNY_PER_USD,
            "ecb_2025_usd_per_eur": ECB_USD_PER_EUR,
            "ecb_2025_cny_per_eur": ECB_CNY_PER_EUR,
            "caes_cost_ratio_vs_uniform_cny_route": CAES_RELATIVE_TO_UNIFORM_CNY,
            "caes_cost_change_pct_vs_uniform_cny_route": (
                CAES_RELATIVE_TO_UNIFORM_CNY - 1.0
            ) * 100.0,
        },
        "osr": {
            "rows": int(len(osr_detail)),
            "summary_rows": int(len(osr_summary)),
            "all_capacity_points_converged": bool(
                osr_detail["all_capacity_points_converged"].astype(bool).all()
            ),
            "maximum_abs_run_length_change_days": float(
                np.max(np.abs(
                    osr_comparison["mean_run_length_days_source_aware_v40"]
                    - osr_comparison["mean_run_length_days_uniform_v31"]
                ))
            ),
            "maximum_abs_mean_osr_gain_change_pp": float(
                osr_comparison["mean_osr_gain_change_pp"].abs().max()
            ),
            "maximum_abs_lfp_osr_change_pp": float(
                osr_detail_comparison["extra_lfp4h_osr_pct_change"].abs().max()
            ),
            "mean_trajectory_points_with_sign_change": int(np.sum(
                (
                    osr_comparison["mean_osr_gain_pp_uniform_v31"]
                    * osr_comparison["mean_osr_gain_pp_source_aware_v40"]
                ) < 0.0
            )),
            "province_cells_with_osr_gain_sign_change": int(
                osr_detail_comparison["osr_gain_sign_changed"].sum()
            ),
            "endpoints": endpoints.to_dict("records"),
        },
        "fixed_asset": {
            "comparison_rows": int(len(fixed_comparison)),
            "maximum_abs_cost_identity_error_eur": float(
                fixed_comparison["cost_identity_error_eur"].abs().max()
            ),
            "maximum_abs_physical_metric_change": fixed_physical_max_change,
            "primary_summary": primary_fixed.to_dict("records"),
        },
        "system_cost": {
            "model_cell_rows": int(len(
                mix_frames["china_mix_model_cells_source_aware_eur.csv.gz"]
            )),
            "maximum_abs_accounting_identity_error_eur": float(
                mix_frames["china_mix_model_cells_source_aware_eur.csv.gz"][
                    "accounting_identity_error_eur"
                ].abs().max()
            ),
            "selected_configuration_changes": int(
                model_cell_comparison["selected_configuration_changed"].sum()
            ),
            "observed_year_selected_configuration_changes": int(
                model_cell_comparison.loc[
                    model_cell_comparison["evidence_set"].eq("observed_year"),
                    "selected_configuration_changed",
                ].sum()
            ),
            "observed_year_zero_charge_summary": primary_mix.to_dict("records"),
        },
        "input_sha256": {
            str(path): digest(path)
            for path in (
                V24_RUNNER,
                V28_FIXED_RUNNER,
                V28_MIX_RUNNER,
                V31_F4 / "figure4_panel_a_boundary_response.csv",
                V31_F4 / "figure4_panel_b_fixed_asset_summary.csv",
                V31_F5 / "figure5_technology_year_summary.csv",
            )
        },
        "outputs": {},
    }
    checks["outputs"] = {
        name: {
            **dataframe_record(frame),
            "sha256": digest(HERE / name),
        }
        for name, frame in outputs.items()
    }
    if checks["osr"]["maximum_abs_run_length_change_days"] > 1e-12:
        raise AssertionError("Event geometry changed during currency-only audit")
    if checks["fixed_asset"]["maximum_abs_cost_identity_error_eur"] > 1e-5:
        raise AssertionError("Fixed-asset accounting identity failed")
    if checks["system_cost"]["maximum_abs_accounting_identity_error_eur"] > 1.0:
        raise AssertionError("System-cost accounting identity failed")
    (HERE / "china_source_aware_currency_QA.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "pass",
        "osr_endpoints": checks["osr"]["endpoints"],
        "fixed_asset_primary_rows": len(primary_fixed),
        "system_cost_primary_rows": len(primary_mix),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
