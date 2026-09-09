"""
Learning-curve gas-parity figures for fixed LFP 4h plus CAES tail supplement.

This sensitivity keeps the first-paper learning curve tied to its original
short-duration BESS anchor:

* single LFP is forced to duration_h = 4;
* CAES duration follows the existing seasonal-MC CAES OSR champion in each
  province / panel / dispatch-window cell;
* LFP CAPEX follows CONFIG["year_costs"] for 2025 / 2030 / 2035;
* CAES CAPEX is held at the 2025 M2 baseline, so the LFP learning curve is not
  transferred to LDES.

The figures use the same gas-parity linear style as the existing
``m4_economics_learning_curve`` output, but are written to a sibling folder:

  output/m4_economics_lfp4h_learning_curve/
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from config import CONFIG
from ldes_params import load_tech_library
import m4_hybrid_lfp4h_caes_discharge_ratio_frontier as lfp4_frontier
import m4_visualize_freqlen_seasonal_mc_per_province_marginal_osr_no_lfp4h as marginal
import m4_visualize_lfp_caes_discharge_ratio_gas_parity_by_dw as gas_plot
import m4_visualize_lfp_caes_discharge_ratio_learning_curve_gas_parity_by_dw as lc
import m4_visualize_freqlen_seasonal_mc_marginal_osr_lfp_caes_discharge_ratio as ratio_plot


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUT_ROOT = (
    PAPER_CODE_DIR
    / "output"
    / "m4_economics_lfp4h_learning_curve"
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_lfp4h_marginal_osr_lfp4h_caes_discharge_ratio_learning_curve_gas_parity_linear_by_dw"
)
FIG_PREFIX_BASE = (
    "fig5_freqlen_seasonal_mc_marginal_osr_lfp4h_caes_discharge_ratio_"
    "lfp4h_learning_curve"
)
SCENARIO = "baseline"
LFP_DURATION_H = 4
SINGLE_TECHS = ["li_lfp", "caes"]


def year_costs() -> dict[int, float]:
    return {int(year): float(cost) for year, cost in CONFIG["year_costs"].items()}


def base_storage_cost() -> float:
    return float(CONFIG["cost"]["storage_cost_per_kwh"])


def storage_cost_for_year(year: int) -> float:
    costs = year_costs()
    if int(year) not in costs:
        raise ValueError(f"Year {year} is not in CONFIG['year_costs'].")
    return costs[int(year)]


def lfp_learning_factor_for_year(year: int) -> float:
    return storage_cost_for_year(year) / base_storage_cost()


def caes_learning_factor_for_year(year: int) -> float:
    _ = year
    return 1.0


def output_dir(mode: str, year: int, province_en: str) -> Path:
    return OUT_ROOT / f"year_{int(year)}" / mode / gas_plot.slugify(province_en)


def source_dir(mode: str, year: int) -> Path:
    return OUT_ROOT / f"year_{int(year)}" / mode / "source_data"


def fig_prefix(mode: str, year: int) -> str:
    return (
        f"{FIG_PREFIX_BASE}_{mode}_gas_parity_linear_"
        f"year{int(year)}"
    )


def output_stem(mode: str, year: int, province_en: str) -> str:
    return f"{fig_prefix(mode, year)}_{gas_plot.slugify(province_en)}"


def cost_decline_scope_for_single(tech_id: str) -> str:
    if tech_id == "li_lfp":
        return "single_lfp4h_capex_learning"
    if tech_id == "caes":
        return "single_caes_fixed_2025"
    return "single_tech"


def apply_single_cost(scan: pd.DataFrame, tech_id: str, duration_h: int, year: int, tech_lib: pd.DataFrame) -> pd.DataFrame:
    tp = lc.tech_params(tech_lib, tech_id, int(duration_h))
    lfp_factor = lfp_learning_factor_for_year(year)
    caes_factor = caes_learning_factor_for_year(year)
    factor = lfp_factor if tech_id == "li_lfp" else caes_factor

    out = scan.copy()
    capex = pd.to_numeric(out["capex_cny"], errors="coerce").fillna(0.0)
    supply = pd.to_numeric(out["annual_supply_kwh"], errors="coerce").fillna(0.0)
    out["annual_cost_cny_2025"] = pd.to_numeric(out["annual_cost_cny"], errors="coerce")
    out["capex_cny_2025"] = capex
    out["annualized_capex_cny_learning"] = lc.annualized_capex(capex, tp, factor)
    out["fixed_om_cny_learning"] = lc.fixed_om_from_capacity(tp, out["capacity_mwh"])
    out["variable_om_cny_learning"] = lc.variable_om_from_supply(tp, supply)
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
    out["lfp_cost_factor_vs_2025"] = lfp_factor if tech_id == "li_lfp" else np.nan
    out["caes_cost_factor_vs_2025"] = caes_factor if tech_id == "caes" else np.nan
    out["cost_decline_scope"] = cost_decline_scope_for_single(tech_id)
    return out


def apply_hybrid_cost(front: pd.DataFrame, year: int, tech_lib: pd.DataFrame) -> pd.DataFrame:
    if front.empty:
        return front.copy()

    lfp_dur = int(pd.to_numeric(front["lfp_duration_h"], errors="coerce").dropna().iloc[0])
    caes_dur = int(pd.to_numeric(front["caes_duration_h"], errors="coerce").dropna().iloc[0])
    if lfp_dur != LFP_DURATION_H:
        raise ValueError(f"Expected LFP {LFP_DURATION_H}h frontier, got {lfp_dur}h.")

    lfp_tp = lc.tech_params(tech_lib, "li_lfp", lfp_dur)
    caes_tp = lc.tech_params(tech_lib, "caes", caes_dur)
    lfp_factor = lfp_learning_factor_for_year(year)
    caes_factor = caes_learning_factor_for_year(year)

    out = front.copy()
    lfp_capex = pd.to_numeric(out["capex_lfp_cny"], errors="coerce").fillna(0.0)
    caes_capex = pd.to_numeric(out["capex_ldes_cny"], errors="coerce").fillna(0.0)
    annual_supply = pd.to_numeric(out["annual_supply_kwh"], errors="coerce").fillna(0.0)
    lfp_supply = pd.to_numeric(out["annual_supply_lfp_kwh"], errors="coerce").fillna(0.0)
    caes_supply = pd.to_numeric(out["annual_supply_ldes_kwh"], errors="coerce").fillna(0.0)

    out["annual_cost_cny_2025"] = pd.to_numeric(out["annual_cost_cny"], errors="coerce")
    out["capex_lfp_cny_2025"] = lfp_capex
    out["capex_ldes_cny_2025"] = caes_capex
    out["annualized_capex_lfp_cny_learning"] = lc.annualized_capex(lfp_capex, lfp_tp, lfp_factor)
    out["annualized_capex_ldes_cny_learning"] = lc.annualized_capex(caes_capex, caes_tp, caes_factor)
    out["fixed_om_lfp_cny_learning"] = lc.fixed_om_from_capacity(lfp_tp, out["lfp_capacity_mwh"])
    out["fixed_om_ldes_cny_learning"] = lc.fixed_om_from_capacity(caes_tp, out["caes_capacity_mwh"])
    out["variable_om_lfp_cny_learning"] = lc.variable_om_from_supply(lfp_tp, lfp_supply)
    out["variable_om_ldes_cny_learning"] = lc.variable_om_from_supply(caes_tp, caes_supply)
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
    out["lfp_cost_factor_vs_2025"] = lfp_factor
    out["caes_cost_factor_vs_2025"] = caes_factor
    out["cost_decline_scope"] = "lfp4h_capex_learning_caes_fixed_2025"
    return out


def source_rows_from_curve(curve: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    out = curve.copy()
    for key, value in meta.items():
        out[key] = value
    return out


def load_hybrid_frontier(mode: str, province_filter: str | None, dw_values: list[float]) -> pd.DataFrame:
    fp = lfp4_frontier.output_csv_for_mode(mode, dw_values)
    if not fp.exists():
        raise FileNotFoundError(
            f"{fp} does not exist. Run m4_hybrid_lfp4h_caes_discharge_ratio_frontier.py first."
        )
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


def cell_metadata_from_champions(
    champion_map: dict[tuple[str, float, str], dict],
    province_filter: str | None,
    dw_values: list[float],
) -> pd.DataFrame:
    rows: list[dict] = []
    for experiment, stem, row_idx, row_label in lfp4_frontier.seasonal_plotter.SCAN_ROWS:
        for (champ_stem, dw_h, province_en), champion in champion_map.items():
            if str(champ_stem) != str(stem):
                continue
            if float(dw_h) not in {float(x) for x in dw_values}:
                continue
            rows.append(
                {
                    "experiment": str(experiment),
                    "stem": str(stem),
                    "row_idx": int(row_idx),
                    "row_label": str(row_label),
                    "dispatch_window_h": float(dw_h),
                    "province_en": str(province_en),
                    "province_cn": str(champion["province_cn"]),
                    "lfp_duration_h": LFP_DURATION_H,
                    "caes_duration_h": int(champion["caes_duration_h"]),
                }
            )
    meta = pd.DataFrame(rows)
    if province_filter and not meta.empty:
        needle = province_filter.lower()
        meta = meta[
            meta["province_en"].astype(str).str.lower().str.contains(needle)
            | meta["province_cn"].astype(str).str.lower().str.contains(needle)
        ].copy()
    return meta.drop_duplicates().sort_values(["province_en", "row_idx", "dispatch_window_h"]).reset_index(drop=True)


def actual_supply_share(front: pd.DataFrame) -> tuple[float, float]:
    lfp = pd.to_numeric(front["annual_supply_lfp_kwh"], errors="coerce").fillna(0.0)
    caes = pd.to_numeric(front["annual_supply_ldes_kwh"], errors="coerce").fillna(0.0)
    total = float((lfp + caes).sum())
    if total <= 0:
        return float("nan"), float("nan")
    lfp_share = float(lfp.sum() / total * 100.0)
    return lfp_share, 100.0 - lfp_share


def add_single_curve(
    source_parts: list[pd.DataFrame],
    summary_rows: list[dict],
    *,
    row: dict,
    tech_id: str,
    duration_h: int,
    mode: str,
    year: int,
    tech_lib: pd.DataFrame,
) -> None:
    scan = lc.load_single_scan(
        stem=str(row["stem"]),
        tech_id=tech_id,
        province_en=str(row["province_en"]),
        dw_h=float(row["dispatch_window_h"]),
        duration_h=int(duration_h),
        tech_lib=tech_lib,
    )
    if scan.empty:
        return

    shifted = apply_single_cost(scan, tech_id, int(duration_h), year, tech_lib)
    frontier = marginal.system_cost_frontier_from_scan(shifted)
    curve = marginal.marginal_from_system_cost_frontier(frontier)
    if frontier.empty or curve.empty:
        return

    osr = marginal.osr_from_marginal_curve(curve)
    tech_label = "LFP" if tech_id == "li_lfp" else "CAES"
    meta = {
        "quota_mode": mode,
        "province_cn": str(row["province_cn"]),
        "province_en": str(row["province_en"]),
        "experiment": str(row["experiment"]),
        "stem": str(row["stem"]),
        "row_idx": int(row["row_idx"]),
        "row_label": str(row["row_label"]),
        "dispatch_window_h": float(row["dispatch_window_h"]),
        "tech_id": tech_id,
        "tech_label": tech_label,
        "plot_label": f"{tech_label} {int(duration_h)}h",
        "duration_h": int(duration_h),
        "lfp_duration_h": int(duration_h) if tech_id == "li_lfp" else np.nan,
        "caes_duration_h": int(duration_h) if tech_id == "caes" else np.nan,
        "osr_pct": osr,
        "series_type": "single",
        "learning_year": int(year),
        "storage_cost_learning_cny_kwh": storage_cost_for_year(year),
        "lfp_cost_factor_vs_2025": lfp_learning_factor_for_year(year) if tech_id == "li_lfp" else np.nan,
        "caes_cost_factor_vs_2025": caes_learning_factor_for_year(year) if tech_id == "caes" else np.nan,
        "cost_decline_scope": cost_decline_scope_for_single(tech_id),
        "system_cost_frontier_points_n": int(len(frontier)),
        "raw_marginal_points_n": int(len(curve)),
        "smoothed_marginal_points_n": int(len(curve)),
    }
    source_parts.append(source_rows_from_curve(curve, meta))
    summary_rows.append(meta)


def add_hybrid_curve(
    source_parts: list[pd.DataFrame],
    summary_rows: list[dict],
    *,
    row: dict,
    mode: str,
    year: int,
    share: int,
    hybrid_front: pd.DataFrame,
    tech_lib: pd.DataFrame,
) -> None:
    front = hybrid_front[
        hybrid_front["province_en"].astype(str).eq(str(row["province_en"]))
        & hybrid_front["row_idx"].astype(int).eq(int(row["row_idx"]))
        & hybrid_front["dispatch_window_h"].astype(float).eq(float(row["dispatch_window_h"]))
        & np.isclose(hybrid_front["lfp_discharge_share_pct"].astype(float), float(share))
    ].copy()
    if front.empty:
        return

    shifted = apply_hybrid_cost(front, year, tech_lib)
    frontier = marginal.system_cost_frontier_from_scan(shifted)
    curve = marginal.marginal_from_system_cost_frontier(frontier)
    if frontier.empty or curve.empty:
        return

    osr = marginal.osr_from_marginal_curve(curve)
    actual_lfp, actual_caes = actual_supply_share(shifted)
    caes_share = 100 - int(share)
    caes_duration_h = int(pd.to_numeric(front["caes_duration_h"], errors="coerce").dropna().iloc[0])
    meta = {
        "quota_mode": mode,
        "province_cn": str(row["province_cn"]),
        "province_en": str(row["province_en"]),
        "experiment": str(row["experiment"]),
        "stem": str(row["stem"]),
        "row_idx": int(row["row_idx"]),
        "row_label": str(row["row_label"]),
        "dispatch_window_h": float(row["dispatch_window_h"]),
        "tech_id": f"hybrid_{mode}_{int(share):02d}",
        "tech_label": ratio_plot.MODE_LABELS[mode],
        "plot_label": f"LFP{int(share)}-CAES{caes_share}",
        "duration_h": f"{LFP_DURATION_H}+{caes_duration_h}",
        "lfp_duration_h": LFP_DURATION_H,
        "caes_duration_h": caes_duration_h,
        "osr_pct": osr,
        "series_type": "discharge_ratio_hybrid",
        "lfp_discharge_share_pct": int(share),
        "caes_discharge_share_pct": caes_share,
        "actual_lfp_supply_share_pct": actual_lfp,
        "actual_caes_supply_share_pct": actual_caes,
        "learning_year": int(year),
        "storage_cost_learning_cny_kwh": storage_cost_for_year(year),
        "lfp_cost_factor_vs_2025": lfp_learning_factor_for_year(year),
        "caes_cost_factor_vs_2025": caes_learning_factor_for_year(year),
        "cost_decline_scope": "lfp4h_capex_learning_caes_fixed_2025",
        "system_cost_frontier_points_n": int(len(frontier)),
        "raw_marginal_points_n": int(len(curve)),
        "smoothed_marginal_points_n": int(len(curve)),
    }
    source_parts.append(source_rows_from_curve(curve, meta))
    summary_rows.append(meta)


def build_learning_source(
    mode: str,
    year: int,
    cell_meta: pd.DataFrame,
    hybrid_front: pd.DataFrame,
    province_en: str,
    dw_h: float,
    tech_lib: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_meta = cell_meta[
        cell_meta["province_en"].astype(str).eq(province_en)
        & cell_meta["dispatch_window_h"].astype(float).eq(float(dw_h))
    ].copy()
    source_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for row in panel_meta.to_dict(orient="records"):
        add_single_curve(
            source_parts,
            summary_rows,
            row=row,
            tech_id="caes",
            duration_h=int(row["caes_duration_h"]),
            mode=mode,
            year=year,
            tech_lib=tech_lib,
        )
        add_single_curve(
            source_parts,
            summary_rows,
            row=row,
            tech_id="li_lfp",
            duration_h=LFP_DURATION_H,
            mode=mode,
            year=year,
            tech_lib=tech_lib,
        )
        for share in ratio_plot.DISCHARGE_LFP_SHARES_PCT:
            add_hybrid_curve(
                source_parts,
                summary_rows,
                row=row,
                mode=mode,
                year=year,
                share=int(share),
                hybrid_front=hybrid_front,
                tech_lib=tech_lib,
            )

    source = pd.concat(source_parts, ignore_index=True) if source_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return source, summary


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
    champion_map = lfp4_frontier.load_lfp4h_caes_champion_map(province_filter, dw_values)
    cell_meta = cell_metadata_from_champions(champion_map, province_filter, dw_values)
    if cell_meta.empty:
        raise ValueError(f"No LFP4h+CAES cells matched mode={mode}, province={province_filter}.")

    hybrid_front = load_hybrid_frontier(mode, province_filter, dw_values)
    provinces = (
        cell_meta[["province_en", "province_cn"]]
        .drop_duplicates()
        .sort_values("province_en")
        .itertuples(index=False)
    )
    provinces = [(str(row.province_en), str(row.province_cn)) for row in provinces]
    sdir = source_dir(mode, year)
    sdir.mkdir(parents=True, exist_ok=True)

    lfp_factor = lfp_learning_factor_for_year(year)
    caes_factor = caes_learning_factor_for_year(year)
    cost = storage_cost_for_year(year)
    method_note = (
        f"Cost scope: LFP4h CAPEX={cost:.1f} CNY/kWh ({lfp_factor:.3f}x); "
        f"CAES CAPEX fixed at 2025 ({caes_factor:.3f}x); O&M retained."
    )
    title_suffix = f"year={int(year)}, LFP4h CAPEX factor={lfp_factor:.3f}, CAES fixed"
    summary_extra = {
        "learning_year": int(year),
        "storage_cost_learning_cny_kwh": cost,
        "lfp_cost_factor_vs_2025": lfp_factor,
        "caes_cost_factor_vs_2025": caes_factor,
        "cost_decline_scope": "lfp4h_learning_caes_fixed_2025",
    }

    all_summary_rows: list[pd.DataFrame] = []
    for idx, (province_en, province_cn) in enumerate(provinces, start=1):
        print(
            f"=== [{mode}/year{year}] rebuild+render {idx}/{len(provinces)}: {province_en} ({province_cn}) ===",
            flush=True,
        )
        province_sources: list[pd.DataFrame] = []
        province_summaries: list[pd.DataFrame] = []
        for dw_h in dw_values:
            source_df, rebuilt_summary = build_learning_source(
                mode=mode,
                year=year,
                cell_meta=cell_meta,
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
            "lfp_cost_factor_vs_2025": lfp_learning_factor_for_year(year),
            "caes_cost_factor_vs_2025": caes_learning_factor_for_year(year),
            "cost_decline_scope": "lfp4h_capex_learning_caes_fixed_2025",
        }
        for year in years
    ]
    pd.DataFrame(rows).to_csv(OUT_ROOT / "learning_curve_year_costs.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create 2025/2030/2035 LFP4h learning-curve gas-parity figures "
            "for single LFP4h, single CAES, and LFP4h+CAES hard/soft quota curves."
        )
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
            run_rows.append(
                {
                    "learning_year": int(year),
                    "mode": mode,
                    "province_filter": args.province or "",
                    "dw_values": ",".join(str(int(x)) if float(x).is_integer() else str(x) for x in dw_values),
                    "summary_rows": int(len(summary)),
                    "storage_cost_learning_cny_kwh_equivalent": storage_cost_for_year(year),
                    "lfp_cost_factor_vs_2025": lfp_learning_factor_for_year(year),
                    "caes_cost_factor_vs_2025": caes_learning_factor_for_year(year),
                    "cost_decline_scope": "lfp4h_capex_learning_caes_fixed_2025",
                }
            )

    pd.DataFrame(run_rows).to_csv(OUT_ROOT / "learning_curve_run_summary.csv", index=False)
    print(f"=== done: {len(run_rows)} mode-year runs ===", flush=True)


if __name__ == "__main__":
    main()
