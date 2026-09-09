"""
Learning-curve versions of LFP+CAES discharge-ratio gas-parity figures.

Unlike a visual-only shift of marginal curves, this script rebuilds the cost
frontier from the original capacity scans:

* single LFP and single CAES curves are rebuilt from ``m4_single_tech_*`` scans;
* hybrid curves are rebuilt from ``m4_hybrid_lfp_caes_discharge_ratio_*``
  capacity frontiers;
* both LFP CAPEX and CAES/LDES CAPEX are reduced by the selected learning-curve
  year factor, while fixed O&M and variable O&M are retained.

The output reproduces the same province/DW ``gas_parity_linear`` figure style
as ``m4_visualize_lfp_caes_discharge_ratio_gas_parity_by_dw.py``.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from config import CONFIG
from ldes_params import calculate_crf, load_tech_library
import m4_hybrid_lfp_caes_discharge_ratio_frontier as hybrid_frontier
import m4_hybrid_search as hybrid
import m4_visualize_freqlen_seasonal_mc_per_province_marginal_osr_no_lfp4h as marginal
import m4_visualize_lfp_caes_discharge_ratio_gas_parity_by_dw as gas_plot
import m4_visualize_freqlen_seasonal_mc_marginal_osr_lfp_caes_discharge_ratio as ratio_plot


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
OUT_ROOT = (
    PAPER_CODE_DIR
    / "output"
    / "m4_economics_learning_curve"
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_lfp_caes_discharge_ratio_learning_curve_gas_parity_linear_by_dw"
)
FIG_PREFIX_BASE = (
    "fig5_freqlen_seasonal_mc_marginal_osr_lfp_caes_discharge_ratio_"
    "learning_curve"
)
SCENARIO = "baseline"
SINGLE_TECHS = ["caes", "li_lfp"]
SINGLE_FILE_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
SINGLE_SCAN_CACHE: dict[tuple[str, str, str, float, int], pd.DataFrame] = {}


def year_costs() -> dict[int, float]:
    return {int(year): float(cost) for year, cost in CONFIG["year_costs"].items()}


def base_storage_cost() -> float:
    return float(CONFIG["cost"]["storage_cost_per_kwh"])


def storage_cost_for_year(year: int) -> float:
    costs = year_costs()
    if int(year) not in costs:
        raise ValueError(f"Year {year} is not in CONFIG['year_costs'].")
    return costs[int(year)]


def learning_factor_for_year(year: int) -> float:
    return storage_cost_for_year(year) / base_storage_cost()


def output_dir(mode: str, year: int, province_en: str) -> Path:
    return OUT_ROOT / f"year_{int(year)}" / mode / gas_plot.slugify(province_en)


def source_dir(mode: str, year: int) -> Path:
    return OUT_ROOT / f"year_{int(year)}" / mode / "source_data"


def fig_prefix(mode: str, year: int) -> str:
    return (
        f"{FIG_PREFIX_BASE}_{mode}_gas_parity_linear_no_lfp4h_"
        f"year{int(year)}"
    )


def output_stem(mode: str, year: int, province_en: str) -> str:
    return f"{fig_prefix(mode, year)}_{gas_plot.slugify(province_en)}"


def tech_params(tech_lib: pd.DataFrame, tech_id: str, duration_h: int):
    tp = hybrid.get_tech_params(tech_lib, tech_id, SCENARIO, int(duration_h))
    if tp is None:
        raise ValueError(f"Missing technology parameters for {tech_id} {duration_h}h.")
    return tp


def fixed_om_from_capacity(tp, capacity_mwh: pd.Series) -> pd.Series:
    capacity = pd.to_numeric(capacity_mwh, errors="coerce").fillna(0.0).astype(float)
    power_mw = capacity / float(tp.duration_h)
    rated_power_kw = power_mw * 1000.0 * max(float(tp.charge_power_ratio), 1.0)
    return rated_power_kw * float(tp.fixed_om_cny_kw_yr)


def annualized_capex(capex_cny: pd.Series, tp, factor: float) -> pd.Series:
    capex = pd.to_numeric(capex_cny, errors="coerce").fillna(0.0).astype(float)
    crf = calculate_crf(float(CONFIG["cost"]["discount_rate"]), int(tp.calendar_life_years))
    return capex * float(factor) * crf


def variable_om_from_supply(tp, annual_supply_kwh: pd.Series) -> pd.Series:
    supply = pd.to_numeric(annual_supply_kwh, errors="coerce").fillna(0.0).astype(float)
    return float(tp.variable_om_cny_mwh) * supply / 1000.0


def apply_single_learning_cost(scan: pd.DataFrame, tech_id: str, duration_h: int, year: int, tech_lib: pd.DataFrame) -> pd.DataFrame:
    tp = tech_params(tech_lib, tech_id, int(duration_h))
    factor = learning_factor_for_year(year)
    out = scan.copy()
    capex = pd.to_numeric(out["capex_cny"], errors="coerce").fillna(0.0)
    supply = pd.to_numeric(out["annual_supply_kwh"], errors="coerce").fillna(0.0)
    out["annual_cost_cny_2025"] = pd.to_numeric(out["annual_cost_cny"], errors="coerce")
    out["capex_cny_2025"] = capex
    out["annualized_capex_cny_learning"] = annualized_capex(capex, tp, factor)
    out["fixed_om_cny_learning"] = fixed_om_from_capacity(tp, out["capacity_mwh"])
    out["variable_om_cny_learning"] = variable_om_from_supply(tp, supply)
    out["annual_cost_cny"] = (
        out["annualized_capex_cny_learning"]
        + out["fixed_om_cny_learning"]
        + out["variable_om_cny_learning"]
    )
    out["unit_cost_cny_kwh"] = np.where(
        supply > 0.0,
        out["annual_cost_cny"].astype(float) / supply,
        np.nan,
    )
    out["cost_ratio_to_gas"] = out["unit_cost_cny_kwh"] / pd.to_numeric(out["gas_unit_cost_cny_kwh"], errors="coerce")
    out["learning_year"] = int(year)
    out["storage_cost_learning_cny_kwh"] = storage_cost_for_year(year)
    out["lfp_cost_factor_vs_2025"] = factor if tech_id == "li_lfp" else np.nan
    out["caes_cost_factor_vs_2025"] = factor if tech_id == "caes" else np.nan
    return out


def apply_hybrid_learning_cost(front: pd.DataFrame, year: int, tech_lib: pd.DataFrame) -> pd.DataFrame:
    if front.empty:
        return front.copy()
    lfp_dur = int(pd.to_numeric(front["lfp_duration_h"], errors="coerce").dropna().iloc[0])
    caes_dur = int(pd.to_numeric(front["caes_duration_h"], errors="coerce").dropna().iloc[0])
    lfp_tp = tech_params(tech_lib, "li_lfp", lfp_dur)
    caes_tp = tech_params(tech_lib, "caes", caes_dur)
    factor = learning_factor_for_year(year)

    out = front.copy()
    lfp_capex = pd.to_numeric(out["capex_lfp_cny"], errors="coerce").fillna(0.0)
    caes_capex = pd.to_numeric(out["capex_ldes_cny"], errors="coerce").fillna(0.0)
    annual_supply = pd.to_numeric(out["annual_supply_kwh"], errors="coerce").fillna(0.0)
    lfp_supply = pd.to_numeric(out["annual_supply_lfp_kwh"], errors="coerce").fillna(0.0)
    caes_supply = pd.to_numeric(out["annual_supply_ldes_kwh"], errors="coerce").fillna(0.0)

    out["annual_cost_cny_2025"] = pd.to_numeric(out["annual_cost_cny"], errors="coerce")
    out["capex_lfp_cny_2025"] = lfp_capex
    out["capex_ldes_cny_2025"] = caes_capex
    out["annualized_capex_lfp_cny_learning"] = annualized_capex(lfp_capex, lfp_tp, factor)
    out["annualized_capex_ldes_cny_learning"] = annualized_capex(caes_capex, caes_tp, factor)
    out["fixed_om_lfp_cny_learning"] = fixed_om_from_capacity(lfp_tp, out["lfp_capacity_mwh"])
    out["fixed_om_ldes_cny_learning"] = fixed_om_from_capacity(caes_tp, out["caes_capacity_mwh"])
    out["variable_om_lfp_cny_learning"] = variable_om_from_supply(lfp_tp, lfp_supply)
    out["variable_om_ldes_cny_learning"] = variable_om_from_supply(caes_tp, caes_supply)
    out["annual_cost_cny"] = (
        out["annualized_capex_lfp_cny_learning"]
        + out["annualized_capex_ldes_cny_learning"]
        + out["fixed_om_lfp_cny_learning"]
        + out["fixed_om_ldes_cny_learning"]
        + out["variable_om_lfp_cny_learning"]
        + out["variable_om_ldes_cny_learning"]
    )
    out["unit_cost_cny_kwh"] = np.where(
        annual_supply > 0.0,
        out["annual_cost_cny"].astype(float) / annual_supply,
        np.nan,
    )
    out["cost_ratio_to_gas"] = out["unit_cost_cny_kwh"] / pd.to_numeric(out["gas_unit_cost_cny_kwh"], errors="coerce")
    out["learning_year"] = int(year)
    out["storage_cost_learning_cny_kwh"] = storage_cost_for_year(year)
    out["lfp_cost_factor_vs_2025"] = factor
    out["caes_cost_factor_vs_2025"] = factor
    return out


def actual_supply_share(front: pd.DataFrame) -> tuple[float, float]:
    lfp = pd.to_numeric(front["annual_supply_lfp_kwh"], errors="coerce").fillna(0.0)
    caes = pd.to_numeric(front["annual_supply_ldes_kwh"], errors="coerce").fillna(0.0)
    total = float((lfp + caes).sum())
    if total <= 0:
        return float("nan"), float("nan")
    lfp_share = float(lfp.sum() / total * 100.0)
    return lfp_share, 100.0 - lfp_share


def source_rows_from_curve(curve: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    out = curve.copy()
    for key, value in meta.items():
        out[key] = value
    return out


def load_single_scan_file(stem: str, tech_id: str, tech_lib: pd.DataFrame) -> pd.DataFrame:
    cache_key = (str(stem), str(tech_id))
    cached = SINGLE_FILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    fp = M4_DIR / f"m4_single_tech_{stem}_{tech_id}.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    usecols = [
        "satisfy_rate_pct",
        "annual_supply_kwh",
        "annual_charge_kwh",
        "annual_spillage_kwh",
        "capex_cny",
        "annual_cost_cny",
        "unit_cost_cny_kwh",
        "gas_unit_cost_cny_kwh",
        "cost_ratio_to_gas",
        "capacity_mwh",
        "demand_stem",
        "scenario",
        "dispatch_window_h",
        "tech_id",
        "duration_h",
        "province_en",
        "province_cn",
    ]
    df = pd.read_csv(fp, usecols=usecols, low_memory=False)
    df = df[df["scenario"].eq(SCENARIO)].copy()
    if tech_id == "li_lfp" and not df.empty:
        df = marginal.plotter.recalculate_lfp_costs(df, tech_lib)
    SINGLE_FILE_CACHE[cache_key] = df
    return df


def load_single_scan(
    stem: str,
    tech_id: str,
    province_en: str,
    dw_h: float,
    duration_h: int,
    tech_lib: pd.DataFrame,
) -> pd.DataFrame:
    cache_key = (str(stem), str(tech_id), str(province_en), float(dw_h), int(duration_h))
    cached = SINGLE_SCAN_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    df = load_single_scan_file(stem, tech_id, tech_lib)
    out = df[
        df["province_en"].astype(str).eq(province_en)
        & df["dispatch_window_h"].astype(float).eq(float(dw_h))
        & df["duration_h"].astype(int).eq(int(duration_h))
    ].copy()
    SINGLE_SCAN_CACHE[cache_key] = out
    return out.copy()


def load_hybrid_frontier(mode: str, province_filter: str | None, dw_values: list[float]) -> pd.DataFrame:
    fp = hybrid_frontier.output_csv_for_mode(mode)
    if not fp.exists():
        raise FileNotFoundError(fp)
    dw_set = {float(x) for x in dw_values}
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(fp, chunksize=200_000, low_memory=False):
        mask = (
            chunk["scenario"].eq(SCENARIO)
            & chunk["dispatch_window_h"].astype(float).isin(dw_set)
        )
        if province_filter:
            needle = province_filter.lower()
            mask &= (
                chunk["province_en"].astype(str).str.lower().str.contains(needle)
                | chunk["province_cn"].astype(str).str.lower().str.contains(needle)
            )
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_summary(mode: str, province_filter: str | None, dw_values: list[float]) -> pd.DataFrame:
    summary = gas_plot.load_mode_summary(mode)
    summary = summary[summary["dispatch_window_h"].astype(float).isin([float(x) for x in dw_values])].copy()
    if province_filter:
        needle = province_filter.lower()
        summary = summary[
            summary["province_en"].astype(str).str.lower().str.contains(needle)
            | summary["province_cn"].astype(str).str.lower().str.contains(needle)
        ].copy()
    if summary.empty:
        raise ValueError(f"No source summary rows matched mode={mode}, province={province_filter}.")
    return summary


def build_learning_source(
    mode: str,
    year: int,
    summary: pd.DataFrame,
    baseline_source: pd.DataFrame,
    hybrid_front: pd.DataFrame,
    province_en: str,
    dw_h: float,
    tech_lib: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_summary = summary[
        summary["province_en"].astype(str).eq(province_en)
        & summary["dispatch_window_h"].astype(float).eq(float(dw_h))
    ].copy()
    source_parts: list[pd.DataFrame] = []
    rebuilt_summary_rows: list[dict] = []

    for row in panel_summary.to_dict(orient="records"):
        row_idx = int(row["row_idx"])
        tech_id = str(row["tech_id"])
        if tech_id in SINGLE_TECHS:
            duration_h = int(float(row["duration_h"]))
            scan = load_single_scan(
                stem=str(row["stem"]),
                tech_id=tech_id,
                province_en=province_en,
                dw_h=float(dw_h),
                duration_h=duration_h,
                tech_lib=tech_lib,
            )
            if scan.empty:
                continue
            shifted = apply_single_learning_cost(scan, tech_id, duration_h, year, tech_lib)
            frontier = marginal.system_cost_frontier_from_scan(shifted)
            curve = marginal.marginal_from_system_cost_frontier(frontier)
            if frontier.empty or curve.empty:
                continue
            osr = marginal.osr_from_marginal_curve(curve)
            meta = dict(row)
            meta.update(
                {
                    "osr_pct": osr,
                    "learning_year": int(year),
                    "storage_cost_learning_cny_kwh": storage_cost_for_year(year),
                    "lfp_cost_factor_vs_2025": learning_factor_for_year(year) if tech_id == "li_lfp" else np.nan,
                    "caes_cost_factor_vs_2025": learning_factor_for_year(year) if tech_id == "caes" else np.nan,
                    "cost_decline_scope": "single_tech_capex_rebuilt",
                    "system_cost_frontier_points_n": int(len(frontier)),
                    "raw_marginal_points_n": int(len(curve)),
                    "smoothed_marginal_points_n": int(len(curve)),
                }
            )
            source_parts.append(source_rows_from_curve(curve, meta))
            rebuilt_summary_rows.append(meta)
            continue

        share = row.get("lfp_discharge_share_pct")
        if pd.isna(share):
            continue
        front = hybrid_front[
            hybrid_front["province_en"].astype(str).eq(province_en)
            & hybrid_front["row_idx"].astype(int).eq(row_idx)
            & hybrid_front["dispatch_window_h"].astype(float).eq(float(dw_h))
            & np.isclose(hybrid_front["lfp_discharge_share_pct"].astype(float), float(share))
        ].copy()
        if front.empty:
            continue
        shifted = apply_hybrid_learning_cost(front, year, tech_lib)
        frontier = marginal.system_cost_frontier_from_scan(shifted)
        curve = marginal.marginal_from_system_cost_frontier(frontier)
        osr = marginal.osr_from_marginal_curve(curve)
        actual_lfp, actual_caes = actual_supply_share(shifted)
        meta = dict(row)
        meta.update(
            {
                "osr_pct": osr,
                "actual_lfp_supply_share_pct": actual_lfp,
                "actual_caes_supply_share_pct": actual_caes,
                "learning_year": int(year),
                "storage_cost_learning_cny_kwh": storage_cost_for_year(year),
                "lfp_cost_factor_vs_2025": learning_factor_for_year(year),
                "caes_cost_factor_vs_2025": learning_factor_for_year(year),
                "cost_decline_scope": "lfp_and_caes_capex_rebuilt",
                "system_cost_frontier_points_n": int(len(frontier)),
                "raw_marginal_points_n": int(len(curve)),
                "smoothed_marginal_points_n": int(len(curve)),
            }
        )
        source_parts.append(source_rows_from_curve(curve, meta))
        rebuilt_summary_rows.append(meta)

    source = pd.concat(source_parts, ignore_index=True) if source_parts else pd.DataFrame()
    rebuilt_summary = pd.DataFrame(rebuilt_summary_rows)
    return source, rebuilt_summary


def render_mode_year(
    mode: str,
    year: int,
    province_filter: str | None,
    dw_values: list[float],
    formats: list[str],
    dpi: int,
    y_max: float,
) -> pd.DataFrame:
    tech_lib = load_tech_library()
    summary = load_summary(mode, province_filter, dw_values)
    hybrid_front = load_hybrid_frontier(mode, province_filter, dw_values)
    provinces = (
        summary[["province_en", "province_cn"]]
        .drop_duplicates()
        .sort_values("province_en")
        .itertuples(index=False)
    )
    provinces = [(str(row.province_en), str(row.province_cn)) for row in provinces]
    sdir = source_dir(mode, year)
    sdir.mkdir(parents=True, exist_ok=True)

    factor = learning_factor_for_year(year)
    cost = storage_cost_for_year(year)
    method_note = (
        f"LFP and CAES/LDES costs are both shifted by the first-paper China storage learning curve: "
        f"{cost:.1f} CNY/kWh equivalent ({factor:.3f}x of 2025). Hybrid curves are rebuilt from LFP and CAES CAPEX components."
    )
    title_suffix = f"year={int(year)}, LFP+CAES CAPEX factor={factor:.3f}"
    summary_extra = {
        "learning_year": int(year),
        "storage_cost_learning_cny_kwh": cost,
        "lfp_cost_factor_vs_2025": factor,
        "caes_cost_factor_vs_2025": factor,
        "cost_decline_scope": "lfp_and_caes_capex",
    }

    all_summary_rows: list[pd.DataFrame] = []
    for idx, (province_en, province_cn) in enumerate(provinces, start=1):
        print(
            f"=== [{mode}/year{year}] rebuild+render {idx}/{len(provinces)}: {province_en} ({province_cn}) ===",
            flush=True,
        )
        baseline_source = gas_plot.load_province_source(mode, province_en)
        province_sources: list[pd.DataFrame] = []
        province_summaries: list[pd.DataFrame] = []
        for dw_h in dw_values:
            source_df, rebuilt_summary = build_learning_source(
                mode=mode,
                year=year,
                summary=summary,
                baseline_source=baseline_source,
                hybrid_front=hybrid_front,
                province_en=province_en,
                dw_h=float(dw_h),
                tech_lib=tech_lib,
            )
            if source_df.empty:
                continue
            province_sources.append(source_df)
            province_summaries.append(rebuilt_summary)
            gas_plot.render_dw_figure(
                mode=mode,
                variant="linear",
                province_en=province_en,
                province_cn=province_cn,
                source_df=source_df,
                dw_h=float(dw_h),
                out_dir=output_dir(mode, year, province_en),
                fig_prefix=fig_prefix(mode, year),
                formats=formats,
                dpi=dpi,
                zoom_y_max=y_max,
                title_suffix=title_suffix,
                method_note=method_note,
                summary_extra=summary_extra,
            )
        if province_sources:
            pd.concat(province_sources, ignore_index=True).to_csv(
                sdir / f"{output_stem(mode, year, province_en)}_source.csv",
                index=False,
            )
        if province_summaries:
            all_summary_rows.append(pd.concat(province_summaries, ignore_index=True))

    all_summary = pd.concat(all_summary_rows, ignore_index=True) if all_summary_rows else pd.DataFrame()
    summary_fp = sdir / f"{fig_prefix(mode, year)}_intersection_summary.csv"
    all_summary.to_csv(summary_fp, index=False)
    print(f"summary: {summary_fp.relative_to(PAPER_CODE_DIR)}", flush=True)
    return all_summary


def write_year_cost_table(years: list[int]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "learning_year": int(year),
            "storage_cost_cny_kwh_equivalent": storage_cost_for_year(year),
            "lfp_cost_factor_vs_2025": learning_factor_for_year(year),
            "caes_cost_factor_vs_2025": learning_factor_for_year(year),
            "cost_decline_scope": "lfp_and_caes_capex",
        }
        for year in years
    ]
    pd.DataFrame(rows).to_csv(OUT_ROOT / "learning_curve_year_costs.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 2025/2030/2035 learning-curve hard/soft gas-parity linear figures with LFP and CAES CAPEX decline."
    )
    parser.add_argument("--modes", nargs="+", default=ratio_plot.QUOTA_MODES, choices=ratio_plot.QUOTA_MODES)
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2030, 2035])
    parser.add_argument("--dw-values", nargs="+", type=float, default=[24.0])
    parser.add_argument("--province", default=None)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--y-max", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = [int(year) for year in args.years]
    dw_values = [float(x) for x in args.dw_values]
    write_year_cost_table(years)

    run_rows: list[dict] = []
    for year in years:
        for mode in args.modes:
            summary = render_mode_year(
                mode=mode,
                year=year,
                province_filter=args.province,
                dw_values=dw_values,
                formats=args.formats,
                dpi=args.dpi,
                y_max=args.y_max,
            )
            factor = learning_factor_for_year(year)
            run_rows.append(
                {
                    "learning_year": int(year),
                    "mode": mode,
                    "province_filter": args.province or "",
                    "dw_values": ",".join(str(int(x)) if float(x).is_integer() else str(x) for x in dw_values),
                    "summary_rows": int(len(summary)),
                    "storage_cost_learning_cny_kwh_equivalent": storage_cost_for_year(year),
                    "lfp_cost_factor_vs_2025": factor,
                    "caes_cost_factor_vs_2025": factor,
                    "cost_decline_scope": "lfp_and_caes_capex",
                }
            )

    pd.DataFrame(run_rows).to_csv(OUT_ROOT / "learning_curve_run_summary.csv", index=False)
    print(f"=== done: {len(run_rows)} mode-year runs ===", flush=True)


if __name__ == "__main__":
    main()
