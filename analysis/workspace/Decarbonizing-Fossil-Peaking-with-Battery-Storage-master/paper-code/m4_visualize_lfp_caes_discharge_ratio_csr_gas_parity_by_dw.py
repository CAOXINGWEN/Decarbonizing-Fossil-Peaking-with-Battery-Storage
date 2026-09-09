"""
CSR gas-parity figures for fixed-discharge-ratio LFP+CAES curves.

This script reads existing source/frontier data only; it does not rerun
dispatch. It produces one province-named folder with four figures per province
(DW=8/12/16/24 h). Each panel plots eleven curves: single CAES, single LFP,
and nine fixed LFP/CAES discharge-ratio hybrids. The y-axis is average unit
cost divided by gas unit cost, so the red y=1 line gives the CSR gas-parity
intersection.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m4_visualize_freqlen_per_province_osr as base
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal
import m4_visualize_freqlen_seasonal_mc_marginal_osr_lfp_caes_discharge_ratio as ratio_plot
import m4_visualize_single_tech_generic as ms


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
NATURE_DIR = M4_DIR / "nature_figure"

SINGLE_SOURCE_DIR = (
    NATURE_DIR
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h"
    / "source_data"
)
SINGLE_SOURCE_PREFIX = "fig5_freqlen_seasonal_mc_osr_no_lfp4h"
PROVINCE_SUMMARY_CSV = (
    NATURE_DIR
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_smoothed_lfp_caes_discharge_ratio_hard_quota"
    / "source_data"
    / "fig5_freqlen_seasonal_mc_marginal_osr_lfp_caes_discharge_ratio_hard_quota_no_lfp4h_summary.csv"
)

DW_VALUES = [8.0, 12.0, 16.0, 24.0]
SINGLE_TECH_ORDER = ["caes", "li_lfp"]
HYBRID_SHARES = ratio_plot.DISCHARGE_LFP_SHARES_PCT

FRONTIER_COLS = [
    "target_supply_pct",
    "annual_supply_kwh",
    "annual_cost_cny",
    "unit_cost_cny_kwh",
    "gas_unit_cost_cny_kwh",
    "cost_ratio_to_gas",
    "scenario",
    "experiment",
    "stem",
    "row_idx",
    "row_label",
    "dispatch_window_h",
    "province_en",
    "province_cn",
    "quota_mode",
    "quota_mode_label",
    "ldes_tech",
    "lfp_duration_h",
    "ldes_duration_h",
    "caes_duration_h",
    "lfp_discharge_share_pct",
    "caes_discharge_share_pct",
    "target_discharge_ratio_label",
    "lfp_single_osr_pct",
    "caes_single_osr_pct",
    "capacity_grid_n",
    "evaluated_pairs_n",
    "total_demand_annual_kwh",
    "frontier_mode",
]


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "province"


def frontier_csv_for_mode(mode: str) -> Path:
    return M4_DIR / f"m4_hybrid_lfp_caes_discharge_ratio_{mode}_frontier_no_lfp4h.csv"


def output_dir_for_mode(mode: str) -> Path:
    return (
        NATURE_DIR
        / f"freqlen_per_province_osr_seasonal_mc_no_lfp4h_csr_lfp_caes_discharge_ratio_{mode}_gas_parity_linear_by_dw"
    )


def fig_prefix_for_mode(mode: str) -> str:
    return f"fig5_freqlen_seasonal_mc_csr_lfp_caes_discharge_ratio_{mode}_gas_parity_linear_no_lfp4h"


def color_for_series(tech_id: str, lfp_share_pct: float | None = None):
    if tech_id in ms.TECH_COLORS:
        return ms.TECH_COLORS[tech_id]
    if lfp_share_pct is not None and np.isfinite(lfp_share_pct):
        return ratio_plot.RATIO_COLORS.get(int(round(float(lfp_share_pct))), "0.35")
    return "0.35"


def style_for_series(tech_id: str) -> dict:
    if tech_id == "li_lfp":
        return {"lw": 2.25, "ls": "--", "alpha": 0.98, "zorder": 8}
    if tech_id == "caes":
        return {"lw": 2.10, "ls": "-", "alpha": 0.98, "zorder": 8}
    return {"lw": 1.45, "ls": "-", "alpha": 0.90, "zorder": 5}


def format_csr_for_label(csr: float) -> str:
    if not np.isfinite(csr):
        return "CSR=NA"
    return f"CSR={csr:.1f}%"


def prepare_average_curve(
    df: pd.DataFrame,
    x_col: str = "supply_pct",
    y_col: str = "cost_ratio_to_gas",
    grid_step_pct: float = 0.5,
) -> pd.DataFrame:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame(columns=["supply_pct", "cost_ratio_to_gas"])

    curve = df[[x_col, y_col]].rename(columns={x_col: "supply_pct", y_col: "cost_ratio_to_gas"}).copy()
    curve["supply_pct"] = pd.to_numeric(curve["supply_pct"], errors="coerce")
    curve["cost_ratio_to_gas"] = pd.to_numeric(curve["cost_ratio_to_gas"], errors="coerce")
    curve = curve.replace([np.inf, -np.inf], np.nan).dropna()
    curve = curve[
        (curve["supply_pct"] > 0)
        & (curve["supply_pct"] <= 100)
        & (curve["cost_ratio_to_gas"] > 0)
    ].copy()
    if curve.empty:
        return pd.DataFrame(columns=["supply_pct", "cost_ratio_to_gas"])

    curve = (
        curve.sort_values(["supply_pct", "cost_ratio_to_gas"])
        .groupby("supply_pct", as_index=False)
        .first()
        .sort_values("supply_pct")
        .reset_index(drop=True)
    )
    if len(curve) < 3:
        return curve

    x = curve["supply_pct"].to_numpy(dtype=float)
    y = curve["cost_ratio_to_gas"].to_numpy(dtype=float)
    x_min, x_max = float(x.min()), float(x.max())
    grid = np.arange(x_min, x_max + grid_step_pct / 2.0, grid_step_pct)
    grid = grid[(grid > 0) & (grid <= 100)]
    if grid.size < 2:
        return curve

    y_grid = np.interp(grid, x, y)
    out = pd.DataFrame({"supply_pct": grid, "cost_ratio_to_gas": y_grid})
    out = out[
        np.isfinite(out["supply_pct"])
        & np.isfinite(out["cost_ratio_to_gas"])
        & (out["cost_ratio_to_gas"] > 0)
    ].copy()
    return out.reset_index(drop=True)


def csr_from_average_curve(curve: pd.DataFrame, y_gas: float = 1.0) -> float:
    if curve.empty:
        return float("nan")
    df = curve[["supply_pct", "cost_ratio_to_gas"]].dropna().sort_values("supply_pct")
    x = df["supply_pct"].to_numpy(dtype=float)
    y = df["cost_ratio_to_gas"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return float("nan")
    if np.all(y > y_gas):
        return 0.0
    if np.all(y <= y_gas):
        return float(min(100.0, x.max()))

    candidates: list[float] = []
    below = y <= y_gas
    if np.any(below):
        candidates.extend(x[below].tolist())
    for i in range(x.size - 1):
        y0 = y[i] - y_gas
        y1 = y[i + 1] - y_gas
        if y0 == 0:
            candidates.append(float(x[i]))
        if y0 * y1 < 0:
            frac = -y0 / (y1 - y0)
            candidates.append(float(x[i] + frac * (x[i + 1] - x[i])))
        elif y1 == 0:
            candidates.append(float(x[i + 1]))
    if not candidates:
        return 0.0
    return float(max(candidates))


def plot_csr_marker(ax, csr: float, color, y_min: float = 0.0, y_gas: float = 1.0) -> None:
    if not np.isfinite(csr) or csr < 0 or csr > 100:
        return
    ax.vlines(csr, y_min, y_gas, color=color, lw=0.75, alpha=0.42, zorder=2)
    ax.scatter([csr], [y_gas], s=20, marker="o", facecolor=color, edgecolor="white",
               linewidth=0.45, alpha=0.98, zorder=12)


def load_province_list(province_filter: str | None) -> list[tuple[str, str]]:
    if not PROVINCE_SUMMARY_CSV.exists():
        raise FileNotFoundError(PROVINCE_SUMMARY_CSV)
    df = pd.read_csv(PROVINCE_SUMMARY_CSV, usecols=["province_en", "province_cn"], low_memory=False)
    df = df.drop_duplicates().sort_values("province_en")
    if province_filter:
        needle = province_filter.lower()
        df = df[
            df["province_en"].astype(str).str.lower().str.contains(needle)
            | df["province_cn"].astype(str).str.lower().str.contains(needle)
        ].copy()
    provinces = [(str(r.province_en), str(r.province_cn)) for r in df.itertuples(index=False)]
    if not provinces:
        raise ValueError(f"No provinces matched filter: {province_filter}")
    return provinces


def load_single_source(province_en: str) -> pd.DataFrame:
    fp = SINGLE_SOURCE_DIR / f"{SINGLE_SOURCE_PREFIX}_{slugify(province_en)}_source.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp, low_memory=False)
    return df[df["tech_id"].isin(SINGLE_TECH_ORDER)].copy()


def load_hybrid_frontiers(mode: str, province_filter: str | None) -> pd.DataFrame:
    fp = frontier_csv_for_mode(mode)
    if not fp.exists():
        raise FileNotFoundError(fp)
    if not province_filter:
        df = pd.read_csv(fp, usecols=FRONTIER_COLS, low_memory=False)
    else:
        needle = province_filter.lower()
        chunks = []
        for chunk in pd.read_csv(fp, usecols=FRONTIER_COLS, chunksize=200_000, low_memory=False):
            mask = (
                chunk["province_en"].astype(str).str.lower().str.contains(needle)
                | chunk["province_cn"].astype(str).str.lower().str.contains(needle)
            )
            if mask.any():
                chunks.append(chunk.loc[mask].copy())
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=FRONTIER_COLS)
    df = df[df["scenario"].eq(ms.SCENARIO)].copy()
    df = df[df["quota_mode"].eq(mode)].copy()
    df = df[df["dispatch_window_h"].isin(DW_VALUES)].copy()
    df = df[df["lfp_discharge_share_pct"].isin(HYBRID_SHARES)].copy()
    return df


def hybrid_panel_curve(
    hybrid_df: pd.DataFrame,
    province_en: str,
    stem: str,
    row_idx: int,
    dw_h: float,
    lfp_share: int,
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    sub = hybrid_df[
        hybrid_df["province_en"].astype(str).eq(province_en)
        & hybrid_df["stem"].astype(str).eq(stem)
        & hybrid_df["row_idx"].eq(int(row_idx))
        & hybrid_df["dispatch_window_h"].eq(float(dw_h))
        & hybrid_df["lfp_discharge_share_pct"].eq(float(lfp_share))
    ].copy()
    if sub.empty:
        return None, None
    curve = prepare_average_curve(sub, x_col="target_supply_pct", y_col="cost_ratio_to_gas")
    if curve.empty:
        return None, None
    caes_share = 100 - int(lfp_share)
    meta = {
        "tech_id": f"hybrid_{sub['quota_mode'].iloc[0]}_{int(lfp_share):02d}",
        "plot_label": f"LFP{int(lfp_share)}-CAES{caes_share}",
        "series_type": "discharge_ratio_hybrid",
        "lfp_discharge_share_pct": float(lfp_share),
        "caes_discharge_share_pct": float(caes_share),
        "lfp_duration_h": float(sub["lfp_duration_h"].iloc[0]),
        "caes_duration_h": float(sub["caes_duration_h"].iloc[0]),
        "capacity_points_n": int(sub["evaluated_pairs_n"].max()) if "evaluated_pairs_n" in sub else len(sub),
        "frontier_points_n": int(len(sub)),
    }
    return curve, meta


def single_panel_curve(
    single_df: pd.DataFrame,
    stem: str,
    dw_h: float,
    tech_id: str,
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    sub = single_df[
        single_df["stem"].astype(str).eq(stem)
        & single_df["dispatch_window_h"].eq(float(dw_h))
        & single_df["tech_id"].astype(str).eq(tech_id)
    ].copy()
    if sub.empty:
        return None, None
    curve = prepare_average_curve(sub, x_col="supply_pct", y_col="cost_ratio_to_gas")
    if curve.empty:
        return None, None
    first = sub.iloc[0]
    duration = int(round(float(first["duration_h"])))
    tech_label = str(first["tech_label"])
    meta = {
        "tech_id": tech_id,
        "plot_label": f"{tech_label} {duration}h",
        "series_type": "single",
        "duration_h": duration,
        "capacity_points_n": int(len(sub)),
        "frontier_points_n": int(len(sub)),
    }
    return curve, meta


def render_dw_figure(
    mode: str,
    province_en: str,
    province_cn: str,
    single_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    dw_h: float,
    out_dir: Path,
    fig_prefix: str,
    formats: list[str],
    dpi: int,
    y_max: float,
) -> list[dict]:
    fig = plt.figure(figsize=(16.5, 19.5))
    gs = fig.add_gridspec(
        4, 2, left=0.070, right=0.985, top=0.915, bottom=0.055,
        wspace=0.18, hspace=0.31,
    )
    mode_label = ratio_plot.MODE_LABELS[mode]
    summary_rows: list[dict] = []

    fig.suptitle(
        f"Gas-parity CSR intersections, {mode_label} - {province_en}, DW={int(dw_h)}h",
        fontsize=15,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.070,
        0.948,
        "CSR view: y-axis is average unit cost / gas cost. Dots and legend values mark the rightmost gas-parity intersection.",
        ha="left",
        va="top",
        fontsize=8.2,
    )

    panel_idx = 0
    for experiment, stem, row_idx, row_label in seasonal.plotter.SCAN_ROWS:
        ax = fig.add_subplot(gs[int(row_idx) // 2, int(row_idx) % 2])
        ax.axhline(1.0, color="#c74f4f", lw=1.15, ls="--", alpha=0.88, zorder=1)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, y_max)
        ax.set_title(
            f"{base.panel_label(panel_idx)} {row_label}",
            loc="left",
            fontsize=9.6,
            fontweight="bold",
        )
        ax.grid(True, which="major", lw=0.35, color="#d8d8d8", alpha=0.80)
        ax.set_xlabel("Peaking demand coverage / supply ratio (%)", fontsize=8.5)
        ax.set_ylabel("Average unit cost / gas cost", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=7.5)

        legend_handles = []
        legend_labels = []
        hybrid_csr = []

        curve_specs: list[tuple[pd.DataFrame, dict]] = []
        for tech_id in SINGLE_TECH_ORDER:
            curve, meta = single_panel_curve(single_df, stem, dw_h, tech_id)
            if curve is not None:
                curve_specs.append((curve, meta))
        for share in HYBRID_SHARES:
            curve, meta = hybrid_panel_curve(hybrid_df, province_en, stem, int(row_idx), dw_h, int(share))
            if curve is not None:
                curve_specs.append((curve, meta))

        if not curve_specs:
            ax.text(0.5, 0.5, "No curve data", transform=ax.transAxes, ha="center", va="center")
            panel_idx += 1
            continue

        for curve, meta in curve_specs:
            tech_id = str(meta["tech_id"])
            color = color_for_series(tech_id, meta.get("lfp_discharge_share_pct"))
            style = style_for_series(tech_id)
            csr = csr_from_average_curve(curve)
            legend_label = f"{meta['plot_label']} ({format_csr_for_label(csr)})"
            line, = ax.plot(
                curve["supply_pct"],
                curve["cost_ratio_to_gas"],
                color=color,
                label=legend_label,
                **style,
            )
            plot_csr_marker(ax, csr, color, y_min=0.0)
            legend_handles.append(line)
            legend_labels.append(legend_label)
            if meta.get("series_type") == "discharge_ratio_hybrid":
                hybrid_csr.append(csr)

            row = {
                "quota_mode": mode,
                "province_cn": province_cn,
                "province_en": province_en,
                "experiment": experiment,
                "stem": stem,
                "row_idx": int(row_idx),
                "row_label": row_label,
                "dispatch_window_h": float(dw_h),
                "tech_id": tech_id,
                "plot_label": meta["plot_label"],
                "csr_pct": csr,
                "series_type": meta.get("series_type"),
                "capacity_points_n": meta.get("capacity_points_n"),
                "frontier_points_n": meta.get("frontier_points_n"),
                "lfp_discharge_share_pct": meta.get("lfp_discharge_share_pct"),
                "caes_discharge_share_pct": meta.get("caes_discharge_share_pct"),
                "duration_h": meta.get("duration_h"),
                "lfp_duration_h": meta.get("lfp_duration_h"),
                "caes_duration_h": meta.get("caes_duration_h"),
                "min_cost_ratio_to_gas": float(curve["cost_ratio_to_gas"].min()),
                "max_cost_ratio_to_gas": float(curve["cost_ratio_to_gas"].max()),
            }
            summary_rows.append(row)

        finite = np.asarray([x for x in hybrid_csr if np.isfinite(x)], dtype=float)
        if finite.size:
            ax.text(
                0.985,
                0.965,
                f"Hybrid CSR range: {finite.min():.1f}-{finite.max():.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.3,
                bbox=dict(facecolor="white", edgecolor="#d6d6d6", linewidth=0.45, alpha=0.88),
            )

        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            fontsize=5.4,
            handlelength=1.55,
            ncol=2,
            columnspacing=0.65,
            handletextpad=0.35,
            borderaxespad=0.30,
            labelspacing=0.18,
            frameon=True,
            framealpha=0.82,
            edgecolor="#d0d0d0",
        )
        panel_idx += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{fig_prefix}_{slugify(province_en)}_dw{int(dw_h)}h"
    for fmt in formats:
        fp = out_dir / f"{stem}.{fmt}"
        fig.savefig(fp, dpi=dpi if fmt.lower() == "png" else None)
        print(f"  saved: {fp.relative_to(PAPER_CODE_DIR)}", flush=True)
    plt.close(fig)
    return summary_rows


def render_mode(
    mode: str,
    province_filter: str | None,
    formats: list[str],
    dpi: int,
    y_max: float,
) -> None:
    provinces = load_province_list(province_filter)
    hybrid_df = load_hybrid_frontiers(mode, province_filter)
    out_root = output_dir_for_mode(mode)
    source_out = out_root / "source_data"
    source_out.mkdir(parents=True, exist_ok=True)
    fig_prefix = fig_prefix_for_mode(mode)
    all_summary_rows: list[dict] = []

    for idx, (province_en, province_cn) in enumerate(provinces, start=1):
        print(f"=== [{mode}/CSR] render province {idx}/{len(provinces)}: {province_en} ({province_cn}) ===", flush=True)
        single_df = load_single_source(province_en)
        province_dir = out_root / slugify(province_en)
        for dw_h in DW_VALUES:
            all_summary_rows.extend(
                render_dw_figure(
                    mode=mode,
                    province_en=province_en,
                    province_cn=province_cn,
                    single_df=single_df,
                    hybrid_df=hybrid_df,
                    dw_h=dw_h,
                    out_dir=province_dir,
                    fig_prefix=fig_prefix,
                    formats=formats,
                    dpi=dpi,
                    y_max=y_max,
                )
            )

    summary_name = f"{fig_prefix}_intersection_summary.csv"
    if province_filter:
        summary_name = f"{fig_prefix}_{slugify(province_filter)}_intersection_summary.csv"
    summary_fp = source_out / summary_name
    pd.DataFrame(all_summary_rows).to_csv(summary_fp, index=False)
    print(f"=== [{mode}/CSR] gas-parity summary: {len(all_summary_rows):,} rows ===", flush=True)
    print(f"summary: {summary_fp.relative_to(PAPER_CODE_DIR)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSR average-cost gas-parity figures for fixed-discharge-ratio LFP+CAES curves."
    )
    parser.add_argument("--modes", nargs="+", default=["hard_quota"], choices=ratio_plot.QUOTA_MODES)
    parser.add_argument("--province", default=None, help="Optional province filter, e.g. Anhui or 安徽")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--y-max", type=float, default=3.0, help="Linear y-axis upper limit in average unit cost / gas cost units.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for mode in args.modes:
        render_mode(
            mode=mode,
            province_filter=args.province,
            formats=args.formats,
            dpi=args.dpi,
            y_max=args.y_max,
        )


if __name__ == "__main__":
    main()
