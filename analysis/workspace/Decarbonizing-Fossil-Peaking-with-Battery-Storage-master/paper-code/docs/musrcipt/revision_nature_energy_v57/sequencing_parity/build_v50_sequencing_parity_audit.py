#!/usr/bin/env python3
"""Build the v50 sequencing-parity and three-regime recovery-window audit."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
V50 = HERE.parent
WORKSPACE = HERE.parents[5]
COSTS = V50 / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
DECISION = V50 / "decision_flip" / "decision_flip_catalogue_choices.csv"
EUROPE_GEOMETRY = V50 / "figure_work" / "source_data" / "figure2_panel_b_country_year_geometry.csv"
CHINA_GEOMETRY = V50 / "figure_work" / "source_data_v49" / "figure2b_china_map.csv"
JAPAN_GEOMETRY = V50 / "figure_work" / "source_data_v49" / "figure2c_japan_map.csv"
EUROPE_PANEL = (
    WORKSPACE
    / "dataset"
    / "external_hourly"
    / "europe_mid_duration_valley_upgrade_v0_1"
    / "processed"
    / "observed_gas_event_task_definitions_v0_2.csv.gz"
)
CHINA_PANEL = V50 / "cross_scale_cases" / "china_province_panel" / "china_province_event_task_panel.csv.gz"
JAPAN_PANEL = V50 / "cross_scale_cases" / "japan_community_panel" / "japan_community_event_task_panel.csv.gz"
RUNNER = V50 / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"

RATES = (0.02, 0.036, 0.06, 0.08)
YEARS = (2025, 2035)
CHRONOLOGY_MEDIAN = 5.48


def crf(rate: float, life: float) -> float:
    return rate * (1.0 + rate) ** life / ((1.0 + rate) ** life - 1.0)


def load_runner():
    spec = importlib.util.spec_from_file_location("v50_portfolio_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_cost_sensitivity() -> pd.DataFrame:
    costs = pd.read_csv(COSTS)
    selected = costs[
        costs["learning_year"].isin(YEARS)
        & (
            costs["config_id"].eq("li_lfp_4h")
            | costs["tech_id"].eq("caes")
        )
    ].copy()
    rows: list[dict[str, float | int | bool]] = []
    for rate in RATES:
        for duration in sorted(selected.loc[selected.tech_id.eq("caes"), "duration_h"].unique()):
            annual: dict[tuple[str, int], float] = {}
            for year in YEARS:
                for config in ("li_lfp_4h", f"caes_{duration:g}h"):
                    row = selected[
                        selected["learning_year"].eq(year)
                        & selected["config_id"].eq(config)
                    ].iloc[0]
                    capex_per_kw = float(row["total_capex_eur_source_aware_kwh"]) * float(row["duration_h"])
                    annual[(config, year)] = (
                        crf(rate, float(row["calendar_life_years"])) * capex_per_kw
                        + float(row["fixed_om_eur_source_aware_kw_yr"])
                    )
            config = f"caes_{duration:g}h"
            rel_2025 = annual[("li_lfp_4h", 2025)] - annual[(config, 2025)]
            rel_2035 = annual[("li_lfp_4h", 2035)] - annual[(config, 2035)]
            movement = abs(rel_2035 - rel_2025)
            rows.append(
                {
                    "discount_rate": rate,
                    "caes_duration_h": float(duration),
                    "lfp_duration_h": 4.0,
                    "lfp_2025_annual_fixed_cost_eur_kw_year": annual[("li_lfp_4h", 2025)],
                    "lfp_2035_annual_fixed_cost_eur_kw_year": annual[("li_lfp_4h", 2035)],
                    "caes_2025_annual_fixed_cost_eur_kw_year": annual[(config, 2025)],
                    "caes_2035_annual_fixed_cost_eur_kw_year": annual[(config, 2035)],
                    "relative_advantage_2025_eur_kw_year": rel_2025,
                    "relative_advantage_2035_eur_kw_year": rel_2035,
                    "absolute_relative_cost_movement_eur_kw_year": movement,
                    "chronology_median_eur_kw_year": CHRONOLOGY_MEDIAN,
                    "chronology_median_at_least_cost_movement": CHRONOLOGY_MEDIAN >= movement,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "cost_duration_discount_sensitivity.csv", index=False)
    central = out[
        np.isclose(out.discount_rate, 0.036)
        & np.isclose(out.caes_duration_h, 24.0)
    ].iloc[0]
    catalogue = selected.pivot(
        index="config_id",
        columns="learning_year",
        values="annual_fixed_cost_eur_source_aware_per_kw_discharge_year",
    )
    expected = abs(
        (catalogue.loc["li_lfp_4h", 2035] - catalogue.loc["caes_24h", 2035])
        - (catalogue.loc["li_lfp_4h", 2025] - catalogue.loc["caes_24h", 2025])
    )
    if not math.isclose(
        float(central.absolute_relative_cost_movement_eur_kw_year),
        float(expected),
        rel_tol=0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("Central 24 h sensitivity does not reproduce the cost catalogue")
    return out


def geometry_correlations() -> pd.DataFrame:
    specs = [
        ("Europe", EUROPE_GEOMETRY, "annual_task_twh", "natural_longest_event_run_days"),
        ("China", CHINA_GEOMETRY, "annual_observed_peaking_task_gwh", "longest_extreme_peaking_run_days"),
        ("Japan", JAPAN_GEOMETRY, "annual_residual_import_mwh", "longest_extreme_import_run_days"),
    ]
    rows = []
    for regime, path, task, event in specs:
        frame = pd.read_csv(path)
        rho, pvalue = spearmanr(frame[task], frame[event], nan_policy="omit")
        rows.append(
            {
                "regime": regime,
                "n_case_years": int(frame[[task, event]].dropna().shape[0]),
                "spearman_rho_annual_task_vs_longest_event": float(rho),
                "two_sided_p_value": float(pvalue),
                "task_min": float(frame[task].min()),
                "task_max": float(frame[task].max()),
                "event_run_min_days": float(frame[event].min()),
                "event_run_median_days": float(frame[event].median()),
                "event_run_max_days": float(frame[event].max()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "three_regime_task_event_correlations.csv", index=False)
    return out


def recovery_case_metrics(frame: pd.DataFrame, id_cols: list[str], runner) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(id_cols, sort=True):
        group = group.sort_values("source_hour_id", kind="stable")
        if len(group) != 8760:
            raise RuntimeError(f"{keys}: expected 8760 hours, got {len(group)}")
        adapter = group.rename(columns={id_cols[0]: "country", id_cols[-1]: "year"})
        orders, _ = runner.day_orders(adapter)
        daily = (
            group.groupby("source_day_id", sort=True)
            .agg(event=("event_flag_q75", "max"), recharge=("r2_potential_recharge_mw", "sum"))
        )
        event = daily["event"].astype(bool).to_dict()
        recharge = daily["recharge"].to_dict()
        for chronology, order in orders.items():
            flags = np.array([event[int(day)] for day in order], dtype=bool)
            recharge_days = np.array([recharge[int(day)] for day in order], dtype=float)
            pos = np.flatnonzero(flags)
            recovery_days = np.diff(pos) - 1
            available = np.array(
                [recharge_days[left + 1 : right].sum() for left, right in zip(pos[:-1], pos[1:])],
                dtype=float,
            )
            row = {col: val for col, val in zip(id_cols, keys if isinstance(keys, tuple) else (keys,))}
            row.update(
                {
                    "chronology": chronology,
                    "event_days": int(flags.sum()),
                    "recovery_window_count": int(len(recovery_days)),
                    "median_recovery_hours": float(np.median(recovery_days * 24.0)),
                    "zero_recovery_window_share": float(np.mean(recovery_days == 0)),
                    "median_window_recharge_mwh": float(np.median(available)),
                    "mean_window_recharge_mwh": float(np.mean(available)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_recovery_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    runner = load_runner()
    europe = pd.read_csv(
        EUROPE_PANEL,
        usecols=["country", "year", "source_hour_id", "source_day_id", "season_segment", "event_flag_q75", "task_c3_q75_mw", "r2_potential_recharge_mw"],
    )
    china = pd.read_csv(
        CHINA_PANEL,
        usecols=["province", "year", "source_hour_id", "source_day_id", "season_segment", "event_flag_q75", "task_c3_q75_mw", "r2_potential_recharge_mw"],
    )
    japan = pd.read_csv(
        JAPAN_PANEL,
        usecols=["prefecture", "mesh_id", "fiscal_year", "source_hour_id", "source_day_id", "season_segment", "event_flag_q75", "task_c3_q75_mw", "r2_potential_recharge_mw"],
    )
    pieces = []
    e = recovery_case_metrics(europe, ["country", "year"], runner)
    e.insert(0, "regime", "Europe")
    pieces.append(e)
    c = recovery_case_metrics(china, ["province", "year"], runner)
    c.insert(0, "regime", "China")
    pieces.append(c)
    j = recovery_case_metrics(japan, ["prefecture", "mesh_id", "fiscal_year"], runner)
    j.insert(0, "regime", "Japan")
    pieces.append(j)
    case = pd.concat(pieces, ignore_index=True, sort=False)
    case.to_csv(HERE / "three_regime_recovery_window_case_year.csv", index=False)
    summary = (
        case.groupby(["regime", "chronology"], sort=False)
        .agg(
            n_case_years=("event_days", "size"),
            median_of_case_median_recovery_hours=("median_recovery_hours", "median"),
            mean_zero_recovery_window_share=("zero_recovery_window_share", "mean"),
            median_zero_recovery_window_share=("zero_recovery_window_share", "median"),
            median_of_case_median_window_recharge_mwh=("median_window_recharge_mwh", "median"),
        )
        .reset_index()
    )
    summary.to_csv(HERE / "three_regime_recovery_window_summary.csv", index=False)
    return case, summary


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    costs = build_cost_sensitivity()
    geometry = geometry_correlations()
    case, recovery = build_recovery_audit()
    choices = pd.read_csv(DECISION)
    at = choices[
        choices["country_year_id"].eq("AT-2024")
        & choices["cost_year"].eq(2035)
    ]
    at.to_csv(HERE / "austria_2024_same_year_pair.csv", index=False)
    central = costs[np.isclose(costs.discount_rate, 0.036)]
    audit = {
        "central_24h_cost_movement_eur_kw_year": float(
            central.loc[np.isclose(central.caes_duration_h, 24), "absolute_relative_cost_movement_eur_kw_year"].iloc[0]
        ),
        "all_duration_rate_cost_movement_min": float(costs.absolute_relative_cost_movement_eur_kw_year.min()),
        "all_duration_rate_cost_movement_max": float(costs.absolute_relative_cost_movement_eur_kw_year.max()),
        "central_rate_duration_cost_movement_min": float(central.absolute_relative_cost_movement_eur_kw_year.min()),
        "central_rate_duration_cost_movement_max": float(central.absolute_relative_cost_movement_eur_kw_year.max()),
        "chronology_median_eur_kw_year": CHRONOLOGY_MEDIAN,
        "sensitivity_cells_with_chronology_at_least_cost_movement": int(costs.chronology_median_at_least_cost_movement.sum()),
        "sensitivity_cells": int(len(costs)),
        "geometry_rows": int(len(geometry)),
        "recovery_case_chronology_rows": int(len(case)),
        "recovery_summary_rows": int(len(recovery)),
        "austria_2024_2035_rows": int(len(at)),
    }
    (HERE / "v50_sequencing_parity_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
