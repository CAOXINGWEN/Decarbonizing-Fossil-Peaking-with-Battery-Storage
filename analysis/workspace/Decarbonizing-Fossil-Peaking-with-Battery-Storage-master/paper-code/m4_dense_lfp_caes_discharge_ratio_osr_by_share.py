"""
Dense 1%-step OSR-by-share diagnostic for LFP+CAES discharge-ratio dispatch.

This script is intentionally scoped for visual diagnostics. It reruns the true
LFP x CAES capacity-pair dispatch for each target LFP discharge share and then
extracts the gas-parity OSR from the marginal-cost frontier. The 0% and 100%
points are added from the corresponding CAES-only and LFP-only single-tech OSR
benchmarks.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m4_visualize_freqlen_per_province_osr as base
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal
import m4_visualize_freqlen_seasonal_mc_per_province_marginal_osr_no_lfp4h as marginal
import m4_hybrid_lfp_caes_discharge_ratio_frontier as frontier
import m4_hybrid_lfp_caes_seasonal_mc_sweep as sparse


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
OUT_ROOT = (
    M4_DIR
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h_lfp_caes_discharge_ratio_dense_1pct_osr_by_share"
)
FIG_PREFIX = "fig5_freqlen_seasonal_mc_lfp_caes_discharge_ratio_dense_1pct_osr_by_share_no_lfp4h"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "province"


def actual_supply_share(front: pd.DataFrame) -> tuple[float, float]:
    if front.empty:
        return float("nan"), float("nan")
    lfp = pd.to_numeric(front["annual_supply_lfp_kwh"], errors="coerce")
    caes = pd.to_numeric(front["annual_supply_ldes_kwh"], errors="coerce")
    total = float((lfp + caes).sum())
    if total <= 0:
        return float("nan"), float("nan")
    lfp_share = float(lfp.sum() / total * 100.0)
    return lfp_share, 100.0 - lfp_share


def duration_h_value(value: float | int | None) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(round(float(value)))


def duration_label(value: float | int | None) -> str:
    dur = duration_h_value(value)
    return f"{dur}h" if dur is not None else "?h"


def resolve_province(champion_map: dict, province_filter: str, dw_h: float) -> tuple[str, str]:
    candidates = sorted(
        {
            (prov_en, champ["province_cn"])
            for (stem, dw, prov_en), champ in champion_map.items()
            if float(dw) == float(dw_h)
        }
    )
    if not candidates:
        raise ValueError(f"No champion province matched {province_filter!r}, DW={dw_h}")
    return candidates[0]


def evaluate_dense(
    province_filter: str,
    dw_h: float,
    quota_mode: str,
    capacity_grid_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    champion_map = sparse.load_champion_map(province_filter)
    province_en, province_cn = resolve_province(champion_map, province_filter, dw_h)
    gas_map, elec_map = sparse.load_prices()
    tech_lib = sparse.load_tech_library()

    summary_rows: list[dict] = []
    frontier_rows: list[dict] = []
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    for experiment, stem, row_idx, row_label in seasonal.plotter.SCAN_ROWS:
        t0 = time.time()
        key = (stem, float(dw_h), province_en)
        if key not in champion_map:
            print(f"skip missing champion: {stem}, DW={dw_h}, {province_en}", flush=True)
            continue
        champion = champion_map[key]
        lfp_duration_h = duration_h_value(champion["lfp_duration_h"])
        caes_duration_h = duration_h_value(champion["caes_duration_h"])

        if stem not in cache:
            demand_df, curt_df, cw_df = frontier.load_inputs(stem)
            lfp_single = frontier.load_single_tech(stem, "li_lfp")
            caes_single = frontier.load_single_tech(stem, "caes")
            cache[stem] = (demand_df, curt_df, cw_df, lfp_single, caes_single)
        else:
            demand_df, curt_df, cw_df, lfp_single, caes_single = cache[stem]

        panel_rows = [
            {
                "quota_mode": quota_mode,
                "province_cn": province_cn,
                "province_en": province_en,
                "experiment": experiment,
                "stem": stem,
                "row_idx": int(row_idx),
                "row_label": row_label,
                "dispatch_window_h": float(dw_h),
                "lfp_discharge_share_pct": 0.0,
                "caes_discharge_share_pct": 100.0,
                "osr_pct": float(champion["caes_single_osr_pct"]),
                "actual_lfp_supply_share_pct": 0.0,
                "actual_caes_supply_share_pct": 100.0,
                "lfp_duration_h": lfp_duration_h,
                "caes_duration_h": caes_duration_h,
                "point_type": "single_endpoint",
                "capacity_grid_n": int(capacity_grid_n),
            }
        ]

        for share_pct in range(1, 100):
            front_rows = frontier.evaluate_ratio_cell(
                stem=stem,
                row_idx=int(row_idx),
                row_label=row_label,
                dw_h=float(dw_h),
                province_en=province_en,
                champion=champion,
                demand_df=demand_df,
                curt_df=curt_df,
                cw_df=cw_df,
                lfp_single=lfp_single,
                caes_single=caes_single,
                gas_map=gas_map,
                elec_map=elec_map,
                tech_lib=tech_lib,
                quota_mode=quota_mode,
                lfp_discharge_share_pct=int(share_pct),
                capacity_grid_n=int(capacity_grid_n),
            )
            front = pd.DataFrame(front_rows)
            if front.empty:
                osr = float("nan")
                actual_lfp, actual_caes = float("nan"), float("nan")
            else:
                curve = marginal.marginal_from_system_cost_frontier(front)
                osr = marginal.osr_from_marginal_curve(curve)
                actual_lfp, actual_caes = actual_supply_share(front)
                for r in front.to_dict("records"):
                    r["dense_source_row_idx"] = int(row_idx)
                    frontier_rows.append(r)

            panel_rows.append(
                {
                    "quota_mode": quota_mode,
                    "province_cn": province_cn,
                    "province_en": province_en,
                    "experiment": experiment,
                    "stem": stem,
                    "row_idx": int(row_idx),
                    "row_label": row_label,
                    "dispatch_window_h": float(dw_h),
                    "lfp_discharge_share_pct": float(share_pct),
                    "caes_discharge_share_pct": float(100 - share_pct),
                    "osr_pct": float(osr),
                    "actual_lfp_supply_share_pct": actual_lfp,
                    "actual_caes_supply_share_pct": actual_caes,
                    "lfp_duration_h": lfp_duration_h,
                    "caes_duration_h": caes_duration_h,
                    "point_type": "dense_hybrid_ratio",
                    "capacity_grid_n": int(capacity_grid_n),
                }
            )

        panel_rows.append(
            {
                "quota_mode": quota_mode,
                "province_cn": province_cn,
                "province_en": province_en,
                "experiment": experiment,
                "stem": stem,
                "row_idx": int(row_idx),
                "row_label": row_label,
                "dispatch_window_h": float(dw_h),
                "lfp_discharge_share_pct": 100.0,
                "caes_discharge_share_pct": 0.0,
                "osr_pct": float(champion["lfp_single_osr_pct"]),
                "actual_lfp_supply_share_pct": 100.0,
                "actual_caes_supply_share_pct": 0.0,
                "lfp_duration_h": lfp_duration_h,
                "caes_duration_h": caes_duration_h,
                "point_type": "single_endpoint",
                "capacity_grid_n": int(capacity_grid_n),
            }
        )
        summary_rows.extend(panel_rows)
        print(
            f"  dense {quota_mode} {province_en} DW={dw_h:.0f}h {row_label}: "
            f"{len(panel_rows)} points in {time.time() - t0:.1f}s",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    frontiers = pd.DataFrame(frontier_rows)
    return summary, frontiers, province_en, province_cn


def render_figure(
    summary: pd.DataFrame,
    province_en: str,
    province_cn: str,
    dw_h: float,
    quota_mode: str,
    formats: list[str],
    dpi: int,
) -> None:
    mode_label = frontier.QUOTA_MODE_LABELS[quota_mode]
    fig = plt.figure(figsize=(14.5, 17.0))
    gs = fig.add_gridspec(
        4, 2, left=0.075, right=0.985, top=0.925, bottom=0.055,
        wspace=0.18, hspace=0.34,
    )
    fig.suptitle(
        f"Dense 1% OSR at gas parity by target LFP discharge share, {mode_label} - {province_en}, DW={int(dw_h)}h",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.075,
        0.955,
        "Each 1% point reruns LFP x CAES capacity-pair dispatch; 0% is CAES-only and 100% is LFP-only.",
        ha="left",
        va="top",
        fontsize=8.4,
    )

    for panel_idx, (experiment, stem, row_idx, row_label) in enumerate(seasonal.plotter.SCAN_ROWS):
        ax = fig.add_subplot(gs[int(row_idx) // 2, int(row_idx) % 2])
        panel = summary[
            summary["row_idx"].eq(int(row_idx))
            & summary["dispatch_window_h"].eq(float(dw_h))
        ].sort_values("lfp_discharge_share_pct")

        ax.set_title(f"{base.panel_label(panel_idx)} {row_label}", loc="left", fontsize=9.4, fontweight="bold")
        ax.set_xlim(-1, 101)
        ax.set_xticks(list(range(0, 101, 10)))
        ax.set_xlabel("Target LFP discharge share (%)", fontsize=8.2)
        ax.set_ylabel("OSR at gas parity (%)", fontsize=8.2)
        ax.grid(True, which="major", lw=0.35, color="#d8d8d8", alpha=0.82)
        ax.tick_params(axis="both", labelsize=7.4)

        if panel.empty:
            ax.text(0.5, 0.5, "No dense results", transform=ax.transAxes, ha="center", va="center")
            continue
        lfp_duration_h = duration_h_value(panel["lfp_duration_h"].dropna().iloc[0]) if "lfp_duration_h" in panel.columns and panel["lfp_duration_h"].notna().any() else None
        caes_duration_h = duration_h_value(panel["caes_duration_h"].dropna().iloc[0]) if "caes_duration_h" in panel.columns and panel["caes_duration_h"].notna().any() else None
        duration_legend = (
            f"LFP {duration_label(lfp_duration_h)} + CAES {duration_label(caes_duration_h)}"
        )

        ax.plot(
            panel["lfp_discharge_share_pct"],
            panel["osr_pct"],
            color="#2f6f9f",
            lw=1.8,
            marker="o",
            ms=1.7,
            label=f"Dense 1% dispatch ({duration_legend})",
            zorder=5,
        )
        endpoints = panel[panel["point_type"].eq("single_endpoint")]
        if not endpoints.empty:
            ax.scatter(
                endpoints["lfp_discharge_share_pct"],
                endpoints["osr_pct"],
                color="#1c4f73",
                edgecolor="white",
                linewidth=0.75,
                s=36,
                zorder=7,
                label="Single-tech endpoints",
            )

        y_values = panel["osr_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        if not y_values.empty:
            y_min = max(0.0, float(y_values.min()) - 5.0)
            y_max = min(100.0, float(y_values.max()) + 5.0)
            if y_max - y_min < 10:
                center = (y_min + y_max) / 2.0
                y_min = max(0.0, center - 5.0)
                y_max = min(100.0, center + 5.0)
            ax.set_ylim(y_min, y_max)
            spread = float(y_values.max() - y_values.min())
            ax.text(
                0.985,
                0.055,
                f"0-100 spread: {spread:.1f} pp",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.2,
                bbox=dict(facecolor="white", edgecolor="#d5d5d5", linewidth=0.45, alpha=0.88),
            )

        ax.legend(loc="upper left", fontsize=7.0, frameon=True, framealpha=0.86)

    out_dir = OUT_ROOT / quota_mode / slugify(province_en)
    source_dir = OUT_ROOT / quota_mode / "source_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{FIG_PREFIX}_{quota_mode}_{slugify(province_en)}_dw{int(dw_h)}h"
    for fmt in formats:
        fp = out_dir / f"{stem}.{fmt}"
        fig.savefig(fp, dpi=dpi if fmt == "png" else None)
        print(f"  saved: {fp.relative_to(PAPER_CODE_DIR)}", flush=True)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense 1%-step OSR-by-share diagnostic.")
    parser.add_argument("--province", default="Anhui")
    parser.add_argument("--dw", type=float, default=8.0)
    parser.add_argument("--mode", default="hard_quota", choices=sorted(frontier.QUOTA_MODE_CODES))
    parser.add_argument("--capacity-grid-n", type=int, default=21)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, frontiers, province_en, province_cn = evaluate_dense(
        province_filter=args.province,
        dw_h=float(args.dw),
        quota_mode=args.mode,
        capacity_grid_n=int(args.capacity_grid_n),
    )

    source_dir = OUT_ROOT / args.mode / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(province_en)
    summary_fp = source_dir / f"{FIG_PREFIX}_{args.mode}_{slug}_dw{int(args.dw)}h_summary.csv"
    frontier_fp = source_dir / f"{FIG_PREFIX}_{args.mode}_{slug}_dw{int(args.dw)}h_frontier.csv"
    summary.to_csv(summary_fp, index=False)
    frontiers.to_csv(frontier_fp, index=False)
    print(f"summary: {summary_fp.relative_to(PAPER_CODE_DIR)} ({len(summary):,} rows)", flush=True)
    print(f"frontier: {frontier_fp.relative_to(PAPER_CODE_DIR)} ({len(frontiers):,} rows)", flush=True)

    render_figure(
        summary=summary,
        province_en=province_en,
        province_cn=province_cn,
        dw_h=float(args.dw),
        quota_mode=args.mode,
        formats=args.formats,
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()
