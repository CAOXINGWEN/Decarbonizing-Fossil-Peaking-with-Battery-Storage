"""
M4 economics module for the second paper.

This module consumes the M3 dispatch results and the M2 technology parameter
library, then computes:
  - levelized annual cost curves
  - annual revenue / profit curves
  - marginal cost curves
  - CSR / PSR / OSR boundaries
  - province-weighted national summaries

The implementation deliberately reuses the first-paper 3R definitions from
``metrics.py`` so that the economic boundaries remain comparable across the
two papers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import brentq
import pandas as pd

from config import CONFIG, GAS_POWER_PER_M3, calculate_crf
from ldes_params import (
    SCENARIOS,
    apply_lfp_duration_cost_override,
    load_tech_library,
)
from metrics import aggregate_province_3r, compute_province_3r


PAPER_CODE_DIR = Path(__file__).resolve().parent
MASTER_DIR = PAPER_CODE_DIR.parent
PROJECT_ROOT = MASTER_DIR.parent
DATA_DIR = PROJECT_ROOT / "dataset"

M3_RESULTS_CSV = PAPER_CODE_DIR / "output" / "m3_dispatch" / "m3_dispatch_results.csv"
TECH_LIBRARY_CSV = Path(
    PROJECT_ROOT / "dataset" / "M2_LDES_technology_parameter_library_v0_2.csv"
)
ELECTRICITY_PRICE_XLSX = DATA_DIR / "31省光伏上网.xlsx"
GAS_PRICE_XLSX = DATA_DIR / "主要能源价格.xlsx"

OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m4_economics"

DEFAULT_SCENARIOS = ["baseline", "low_cost", "high_cost"]


def _clean_province(value: object) -> str:
    return str(value).strip()


def load_price_table(
    path: Path,
    province_candidates: Iterable[str],
    value_candidates: Iterable[str],
) -> pd.DataFrame:
    """Load a province-price table and normalize the column names."""
    df = pd.read_excel(path)
    province_col = None
    value_col = None

    for cand in province_candidates:
        for col in df.columns:
            if cand in str(col):
                province_col = col
                break
        if province_col is not None:
            break

    for cand in value_candidates:
        for col in df.columns:
            if cand in str(col):
                value_col = col
                break
        if value_col is not None:
            break

    if province_col is None:
        province_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[1]

    out = df[[province_col, value_col]].copy()
    out.columns = ["province_cn", "value"]
    out["province_cn"] = out["province_cn"].map(_clean_province)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["province_cn", "value"]).drop_duplicates(
        subset=["province_cn"], keep="last"
    )
    return out


def load_electricity_prices() -> pd.DataFrame:
    return load_price_table(
        ELECTRICITY_PRICE_XLSX,
        province_candidates=["省/自治区/直辖市", "省份", "地区"],
        value_candidates=["基准价", "电价", "price"],
    ).rename(columns={"value": "electricity_price_cny_kwh"})


def load_gas_prices() -> pd.DataFrame:
    return load_price_table(
        GAS_PRICE_XLSX,
        province_candidates=["省份", "省/自治区/直辖市", "地区"],
        value_candidates=["天然气价格", "gas", "气价"],
    ).rename(columns={"value": "gas_price_cny_m3"})


def build_cost_scenario_table(
    tech_df: pd.DataFrame,
    cost_scenario: str,
) -> pd.DataFrame:
    """Return one row per technology for the chosen cost scenario."""
    if cost_scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {cost_scenario}")

    base = tech_df.reset_index().copy()
    suffix = SCENARIOS[cost_scenario]["power_suffix"]
    energy_suffix = SCENARIOS[cost_scenario]["energy_suffix"]

    cols = [
        "tech_id",
        "technology_cn",
        "technology_en",
        "family",
        "default_duration_h",
        "duration_min_h",
        "duration_max_h",
        "self_discharge_pct_day",
        "calendar_life_years",
        "cycle_life_cycles",
        "response_time_min",
        "fixed_om_cny_kw_yr",
        "variable_om_cny_mwh",
        "replacement_year",
    ]
    optional_cols = [
        "replacement_component",
        "replacement_fraction_of_component",
        "charge_power_ratio",
    ]
    cols.extend([col for col in optional_cols if col in base.columns])

    scenario_df = base[cols].copy()
    scenario_df["cost_scenario"] = cost_scenario
    scenario_df["power_cost_cny_kw"] = base[f"power_cost_cny_kw_{suffix}"]
    scenario_df["energy_cost_cny_kwh"] = base[f"energy_cost_cny_kwh_{energy_suffix}"]
    return scenario_df


def replacement_present_value(
    replacement_cost_per_event: float,
    replacement_year: int,
    project_life_years: int,
    discount_rate: float,
) -> tuple[float, list[int]]:
    """Present value of repeated replacement events."""
    if replacement_year <= 0 or project_life_years <= replacement_year:
        return 0.0, []

    years = list(range(replacement_year, project_life_years, replacement_year))
    if not years:
        return 0.0, []

    pv = 0.0
    for year in years:
        pv += replacement_cost_per_event / ((1.0 + discount_rate) ** year)
    return pv, years


def replacement_basis_cost(
    component: str,
    capex_power: pd.Series,
    capex_energy: pd.Series,
) -> pd.Series:
    """Return the capex basis that should be replaced for one technology."""
    component = str(component).strip().lower()
    if component == "power":
        return capex_power
    if component == "energy":
        return capex_energy
    if component == "total":
        return capex_power + capex_energy
    return pd.Series(np.zeros(len(capex_power)), index=capex_power.index, dtype=float)


def irr_with_replacements(
    initial_capex: float,
    annual_revenue: float,
    annual_fixed_om: float,
    annual_variable_om: float,
    replacement_years: list[int],
    replacement_cost: float,
    project_life_years: int,
) -> float:
    """Compute IRR from explicit cash flows."""
    if initial_capex <= 0 or project_life_years <= 0:
        return float("nan")

    yearly_cash_flow = annual_revenue - annual_fixed_om - annual_variable_om
    cash_flows = [-initial_capex]
    for year in range(1, project_life_years + 1):
        cf = yearly_cash_flow
        if year in replacement_years:
            cf -= replacement_cost
        cash_flows.append(cf)

    def npv(rate: float) -> float:
        if rate <= -0.999999:
            return np.inf
        denom = (1.0 + rate) ** np.arange(len(cash_flows), dtype=float)
        return float(np.sum(np.asarray(cash_flows, dtype=float) / denom))

    try:
        f_low = npv(-0.9999)
        f_high = npv(10.0)
        if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
            return float("nan")
        irr = brentq(npv, -0.9999, 10.0, maxiter=200)
        return float(irr * 100.0)
    except Exception:
        return float("nan")


def enrich_curve_points(
    group: pd.DataFrame,
    tech_row: pd.Series,
    gas_price_row: pd.Series,
    elec_price_row: pd.Series,
) -> pd.DataFrame:
    """Add all economic columns to a single province-tech-duration curve."""
    out = group.sort_values("capacity_mwh").reset_index(drop=True).copy()

    power_kw = out["power_mw"].astype(float) * 1000.0
    charge_power_ratio = float(tech_row.get("charge_power_ratio", 1.0))
    rated_power_kw = power_kw * max(charge_power_ratio, 1.0)
    capacity_kwh = out["capacity_mwh"].astype(float) * 1000.0
    supply_kwh = out["total_supply_kwh"].astype(float)

    discount_rate = float(CONFIG["cost"]["discount_rate"])
    project_life_years = int(round(float(tech_row["calendar_life_years"])))
    replacement_year = int(round(float(tech_row["replacement_year"])))
    crf = calculate_crf(discount_rate, project_life_years)

    power_cost = float(tech_row["power_cost_cny_kw"])
    energy_cost = float(tech_row["energy_cost_cny_kwh"])
    fixed_om = float(tech_row["fixed_om_cny_kw_yr"])
    duration_h = float(out["duration_h"].iloc[0]) if "duration_h" in out.columns else float("nan")
    power_cost, energy_cost, fixed_om = apply_lfp_duration_cost_override(
        tech_id=str(tech_row.get("tech_id", "")),
        scenario_name=str(tech_row.get("cost_scenario", "")),
        duration_h=duration_h,
        power_cost_cny_kw=power_cost,
        energy_cost_cny_kwh=energy_cost,
        fixed_om_cny_kw_yr=fixed_om,
    )
    variable_om = float(tech_row["variable_om_cny_mwh"])
    dispatch_rte = float(out["rte"].iloc[0]) if "rte" in out.columns else float("nan")
    replacement_component = str(tech_row.get("replacement_component", "energy"))
    replacement_fraction = float(tech_row.get("replacement_fraction_of_component", 1.0))

    gas_price = float(gas_price_row["gas_price_cny_m3"])
    elec_price = float(elec_price_row["electricity_price_cny_kwh"])
    gas_unit_cost = gas_price / GAS_POWER_PER_M3 + float(CONFIG["gas"]["generator_cost"])

    capex_power = power_cost * rated_power_kw
    capex_energy = energy_cost * capacity_kwh
    replacement_basis_cny = replacement_basis_cost(
        replacement_component,
        capex_power,
        capex_energy,
    )
    replacement_cost_per_event_cny = replacement_basis_cny * replacement_fraction
    replacement_discount_factor, replacement_years = replacement_present_value(
        replacement_cost_per_event=1.0,
        replacement_year=replacement_year,
        project_life_years=project_life_years,
        discount_rate=discount_rate,
    )
    replacement_pv = replacement_cost_per_event_cny * replacement_discount_factor

    # Broadcast the per-row capital terms.
    out["power_cost_cny_kw"] = power_cost
    out["energy_cost_cny_kwh"] = energy_cost
    out["fixed_om_cny_kw_yr"] = fixed_om
    out["variable_om_cny_mwh"] = variable_om
    out["calendar_life_years"] = project_life_years
    out["replacement_year"] = replacement_year
    out["replacement_component"] = replacement_component
    out["replacement_fraction_of_component"] = replacement_fraction
    out["discount_rate"] = discount_rate
    out["dispatch_rte"] = dispatch_rte
    out["gas_price_cny_m3"] = gas_price
    out["electricity_price_cny_kwh"] = elec_price
    out["gas_unit_cost_cny_kwh"] = gas_unit_cost

    out["power_capex_cny"] = capex_power
    out["energy_capex_cny"] = capex_energy
    out["replacement_capex_basis_cny"] = replacement_basis_cny
    out["replacement_cost_per_event_cny"] = replacement_cost_per_event_cny
    out["replacement_pv_cny"] = replacement_pv
    out["initial_capex_cny"] = out["power_capex_cny"] + out["energy_capex_cny"]
    out["levelized_capex_pv_cny"] = (
        out["power_capex_cny"] + out["energy_capex_cny"] + out["replacement_pv_cny"]
    )
    out["annualized_capex_cny"] = out["levelized_capex_pv_cny"] * crf

    out["annual_fixed_om_cny"] = fixed_om * rated_power_kw
    out["annual_variable_om_cny"] = variable_om * (supply_kwh / 1000.0)
    out["annual_cost_cny"] = (
        out["annualized_capex_cny"]
        + out["annual_fixed_om_cny"]
        + out["annual_variable_om_cny"]
    )
    out["unit_cost_cny_kwh"] = np.where(
        supply_kwh > 0,
        out["annual_cost_cny"] / supply_kwh,
        np.nan,
    )

    out["annual_revenue_cny"] = supply_kwh * elec_price
    out["annual_net_profit_cny"] = out["annual_revenue_cny"] - out["annual_cost_cny"]
    out["unit_profit_cny_per_kwh_capacity"] = np.where(
        capacity_kwh > 0,
        out["annual_net_profit_cny"] / capacity_kwh,
        np.nan,
    )
    out["unit_margin_to_gas_cny_kwh"] = gas_unit_cost - out["unit_cost_cny_kwh"]
    out["capital_replacement_count"] = len(replacement_years)

    out["annual_replacement_reserve_cny"] = replacement_pv * crf
    out["annual_operating_net_cashflow_cny"] = (
        out["annual_revenue_cny"]
        - out["annual_fixed_om_cny"]
        - out["annual_variable_om_cny"]
    )
    out["irr_pct"] = np.nan
    out["project_life_years"] = project_life_years

    return out


def build_segment_table(group_points: pd.DataFrame) -> pd.DataFrame:
    """Return marginal-cost segments for one economic curve."""
    out = group_points.sort_values("capacity_mwh").reset_index(drop=True)
    if len(out) < 2:
        return pd.DataFrame()

    delta_supply = out["total_supply_kwh"].diff()
    delta_cost = out["annual_cost_cny"].diff()
    marginal_cost = delta_cost / delta_supply.replace(0, np.nan)
    mid_satisfy = (out["satisfy_rate_pct"] + out["satisfy_rate_pct"].shift(1)) / 2.0

    seg = out.iloc[1:].copy().reset_index(drop=True)
    seg["prev_capacity_mwh"] = out["capacity_mwh"].shift(1).iloc[1:].values
    seg["prev_supply_kwh"] = out["total_supply_kwh"].shift(1).iloc[1:].values
    seg["prev_annual_cost_cny"] = out["annual_cost_cny"].shift(1).iloc[1:].values
    seg["delta_supply_kwh"] = delta_supply.iloc[1:].values
    seg["delta_annual_cost_cny"] = delta_cost.iloc[1:].values
    seg["marginal_cost_cny_kwh"] = marginal_cost.iloc[1:].values
    seg["mid_satisfy_rate_pct"] = mid_satisfy.iloc[1:].values
    mid_capacity = (out["capacity_mwh"] + out["capacity_mwh"].shift(1)) / 2.0
    seg["mid_capacity_mwh"] = mid_capacity.iloc[1:].values
    return seg


def summarize_group(group_points: pd.DataFrame) -> dict | None:
    """Compute province-level economics and 3R metrics for one curve."""
    if group_points.empty:
        return None

    summary_input = group_points.rename(
        columns={
            "total_supply_kwh": "peak_supply_kwh",
            "annual_cost_cny": "annual_cost",
            "unit_profit_cny_per_kwh_capacity": "unit_profit",
        }
    ).to_dict("records")

    summary = compute_province_3r(
        summary_input,
        gas_price=float(group_points["gas_price_cny_m3"].iloc[0]),
        elec_price=float(group_points["electricity_price_cny_kwh"].iloc[0]),
        gas_power_per_m3=GAS_POWER_PER_M3,
        generator_cost=float(CONFIG["gas"]["generator_cost"]),
    )
    if summary is None:
        return None

    csr, psr, osr, total_demand = summary
    max_idx = group_points["unit_profit_cny_per_kwh_capacity"].astype(float).idxmax()
    opt_row = group_points.loc[max_idx]

    row = {
        "tech_id": opt_row["tech_id"],
        "technology_cn": opt_row["technology_cn"],
        "cost_scenario": opt_row["cost_scenario"],
        "province_en": opt_row["province_en"],
        "province_cn": opt_row["province_cn"],
        "duration_h": int(opt_row["duration_h"]),
        "total_demand_kwh": float(total_demand),
        "gas_price_cny_m3": float(opt_row["gas_price_cny_m3"]),
        "electricity_price_cny_kwh": float(opt_row["electricity_price_cny_kwh"]),
        "gas_unit_cost_cny_kwh": float(opt_row["gas_unit_cost_cny_kwh"]),
        "csr_pct": float(csr),
        "psr_pct": float(psr),
        "osr_pct": float(osr),
        "max_satisfy_rate_pct": float(group_points["satisfy_rate_pct"].max()),
        "optimal_capacity_mwh": float(opt_row["capacity_mwh"]),
        "optimal_unit_profit_cny_per_kwh_capacity": float(opt_row["unit_profit_cny_per_kwh_capacity"]),
        "optimal_annual_cost_cny": float(opt_row["annual_cost_cny"]),
        "optimal_annual_revenue_cny": float(opt_row["annual_revenue_cny"]),
        "optimal_annual_net_profit_cny": float(opt_row["annual_net_profit_cny"]),
        "optimal_unit_cost_cny_kwh": float(opt_row["unit_cost_cny_kwh"]),
        "optimal_irr_pct": float(
            irr_with_replacements(
                initial_capex=float(opt_row["initial_capex_cny"]),
                annual_revenue=float(opt_row["annual_revenue_cny"]),
                annual_fixed_om=float(opt_row["annual_fixed_om_cny"]),
                annual_variable_om=float(opt_row["annual_variable_om_cny"]),
                replacement_years=list(range(
                    int(round(float(opt_row["replacement_year"]))),
                    int(round(float(opt_row["project_life_years"]))),
                    int(round(float(opt_row["replacement_year"]))),
                )) if int(round(float(opt_row["replacement_year"]))) > 0
                and float(opt_row["replacement_cost_per_event_cny"]) > 0 else [],
                replacement_cost=float(opt_row["replacement_cost_per_event_cny"]),
                project_life_years=int(opt_row["project_life_years"]),
            )
        ),
        "project_life_years": int(opt_row["project_life_years"]),
        "replacement_year": int(opt_row["replacement_year"]),
        "capital_replacement_count": int(opt_row["capital_replacement_count"]),
        "dispatch_rte": float(opt_row["dispatch_rte"]),
    }
    return row


def weighted_national_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate province-level 3R results to national summaries."""
    rows = []
    for keys, group in summary_df.groupby(
        ["tech_id", "technology_cn", "cost_scenario", "duration_h"],
        sort=False,
    ):
        tech_id, technology_cn, cost_scenario, duration_h = keys
        province_3r_list = [
            (r["csr_pct"], r["psr_pct"], r["osr_pct"], r["total_demand_kwh"])
            for _idx, r in group.iterrows()
        ]
        if not province_3r_list:
            continue
        weighted = aggregate_province_3r(province_3r_list)
        rows.append(
            {
                "tech_id": tech_id,
                "technology_cn": technology_cn,
                "cost_scenario": cost_scenario,
                "duration_h": duration_h,
                "province_count": int(group.shape[0]),
                "total_demand_kwh": float(group["total_demand_kwh"].sum()),
                "CSR_weighted_avg": float(weighted["CSR"]),
                "PSR_weighted_avg": float(weighted["PSR"]),
                "OSR_weighted_avg": float(weighted["OSR"]),
                "CSR_median": float(group["csr_pct"].median()),
                "PSR_median": float(group["psr_pct"].median()),
                "OSR_median": float(group["osr_pct"].median()),
                "CSR_q25": float(group["csr_pct"].quantile(0.25)),
                "CSR_q75": float(group["csr_pct"].quantile(0.75)),
                "PSR_q25": float(group["psr_pct"].quantile(0.25)),
                "PSR_q75": float(group["psr_pct"].quantile(0.75)),
                "OSR_q25": float(group["osr_pct"].quantile(0.25)),
                "OSR_q75": float(group["osr_pct"].quantile(0.75)),
            }
        )

    return pd.DataFrame(rows)


def process_cost_scenario(
    results_df: pd.DataFrame,
    tech_df: pd.DataFrame,
    gas_prices: pd.DataFrame,
    elec_prices: pd.DataFrame,
    cost_scenario: str,
    output_tag: str = "",
    save_points: bool = True,
    save_segments: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute all economic outputs for one cost scenario."""
    scenario_tech = build_cost_scenario_table(tech_df, cost_scenario)

    merged = results_df.copy()
    if "scenario" in merged.columns:
        merged = merged.rename(columns={"scenario": "dispatch_scenario"})
    merged["province_cn"] = merged["province_cn"].map(_clean_province)
    merged["province_en"] = merged["province_en"].map(_clean_province)

    merged = merged.merge(
        scenario_tech,
        on="tech_id",
        how="inner",
        suffixes=("", "_tech"),
    )
    merged = merged.merge(
        gas_prices,
        on="province_cn",
        how="left",
    )
    merged = merged.merge(
        elec_prices,
        on="province_cn",
        how="left",
    )

    merged = merged.dropna(subset=["gas_price_cny_m3", "electricity_price_cny_kwh"])

    enriched_groups = []
    summary_rows = []
    segment_rows = []

    group_keys = ["tech_id", "province_en", "province_cn", "duration_h"]
    for _, group in merged.groupby(group_keys, sort=False):
        tech_row = group.iloc[0]
        group_points = enrich_curve_points(group, tech_row, tech_row, tech_row)
        if save_points:
            enriched_groups.append(group_points)

        summary_row = summarize_group(group_points)
        if summary_row is not None:
            summary_rows.append(summary_row)

        if save_segments:
            seg = build_segment_table(group_points)
            if not seg.empty:
                segment_rows.append(seg)

    points_df = pd.concat(enriched_groups, ignore_index=True) if enriched_groups else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    segments_df = pd.concat(segment_rows, ignore_index=True) if segment_rows else pd.DataFrame()

    tag = f"_{output_tag}" if output_tag else ""

    if save_points:
        points_path = OUTPUT_DIR / f"m4_curve_points_{cost_scenario}{tag}.csv"
        points_df.to_csv(points_path, index=False)
    if save_segments:
        segments_path = OUTPUT_DIR / f"m4_curve_segments_{cost_scenario}{tag}.csv"
        segments_df.to_csv(segments_path, index=False)

    summary_path = OUTPUT_DIR / f"m4_3r_summary_{cost_scenario}{tag}.csv"
    summary_df.to_csv(summary_path, index=False)

    national_df = weighted_national_summary(summary_df)
    national_path = OUTPUT_DIR / f"m4_national_3r_summary_{cost_scenario}{tag}.csv"
    national_df.to_csv(national_path, index=False)

    return points_df, summary_df, national_df


def run_all_scenarios(
    m3_results_csv: Path | None = None,
    scenarios: list[str] | None = None,
    output_tag: str = "",
    save_points: bool = True,
    save_segments: bool = True,
) -> None:
    """Main entry point."""
    scenarios = scenarios or DEFAULT_SCENARIOS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = m3_results_csv or M3_RESULTS_CSV
    results_df = pd.read_csv(results_path)
    tech_df = load_tech_library(str(TECH_LIBRARY_CSV))
    gas_prices = load_gas_prices()
    elec_prices = load_electricity_prices()

    all_summary = []
    all_national = []

    for scenario in scenarios:
        print(f"[M4] processing cost scenario: {scenario}")
        points_df, summary_df, national_df = process_cost_scenario(
            results_df=results_df,
            tech_df=tech_df,
            gas_prices=gas_prices,
            elec_prices=elec_prices,
            cost_scenario=scenario,
            output_tag=output_tag,
            save_points=save_points,
            save_segments=save_segments,
        )
        all_summary.append(summary_df)
        all_national.append(national_df)
        print(
            f"  points={len(points_df):,}, province-tech-duration rows={len(summary_df):,}, "
            f"national rows={len(national_df):,}"
        )

    summary_all = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    national_all = pd.concat(all_national, ignore_index=True) if all_national else pd.DataFrame()

    tag = f"_{output_tag}" if output_tag else ""
    summary_all.to_csv(OUTPUT_DIR / f"m4_3r_summary_all_scenarios{tag}.csv", index=False)
    national_all.to_csv(OUTPUT_DIR / f"m4_national_3r_summary_all_scenarios{tag}.csv", index=False)

    manifest = pd.DataFrame(
        {
            "scenario": scenarios,
            "description": [
                {
                    "baseline": "baseline cost assumptions",
                    "low_cost": "low-cost sensitivity",
                    "high_cost": "high-cost sensitivity",
                }.get(s, s)
                for s in scenarios
            ],
        }
    )
    manifest.to_csv(OUTPUT_DIR / f"m4_run_manifest{tag}.csv", index=False)
    print(f"[M4] inputs read from {results_path}")
    print(f"[M4] outputs written to {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M4 economics module.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="Cost scenarios to evaluate: baseline low_cost high_cost",
    )
    parser.add_argument(
        "--no-points",
        action="store_true",
        help="Skip saving the large point-level curve tables.",
    )
    parser.add_argument(
        "--no-segments",
        action="store_true",
        help="Skip saving the marginal-cost segment tables.",
    )
    parser.add_argument(
        "--m3-results-csv",
        type=str,
        default=str(M3_RESULTS_CSV),
        help="Path to the M3 dispatch result table to evaluate.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Suffix tag appended to M4 output filenames, e.g. dw8h.",
    )
    args = parser.parse_args()

    run_all_scenarios(
        m3_results_csv=Path(args.m3_results_csv),
        scenarios=args.scenarios,
        output_tag=args.output_tag,
        save_points=not args.no_points,
        save_segments=not args.no_segments,
    )


if __name__ == "__main__":
    main()
