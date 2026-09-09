"""
Top-10 1%-optimized LFP+CAES discharge-ratio gas-parity curves.

For each province, discharge window, panel, and quota mode, this script scans
target LFP discharge shares from 1% to 99%, computes the marginal gas-parity
OSR for each share, selects the ten highest-OSR hybrid ratios, and plots only
those ten marginal-cost curves.

Output root:
  output/m4_economics/nature_figure/
    freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_lfp_caes_discharge_ratio_top10_1pct_optimal_by_dw/
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import re
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m4_dense_lfp_caes_discharge_ratio_osr_by_share as dense
import m4_hybrid_lfp_caes_discharge_ratio_frontier as frontier
import m4_hybrid_lfp_caes_seasonal_mc_sweep as sparse
import m4_visualize_freqlen_per_province_osr as base
import m4_visualize_freqlen_seasonal_mc_per_province_marginal_osr_no_lfp4h as marginal
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
OUT_ROOT = (
    M4_DIR
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_lfp_caes_discharge_ratio_top10_1pct_optimal_by_dw"
)
FIG_PREFIX = (
    "fig5_freqlen_seasonal_mc_marginal_osr_lfp_caes_discharge_ratio_"
    "top10_1pct_optimal_gas_parity_linear_no_lfp4h"
)
TOP_K = 10
SHARE_RANGE = range(1, 100)
DW_VALUES = [8.0, 12.0, 16.0, 24.0]


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "province"


def mode_label(mode: str) -> str:
    return frontier.QUOTA_MODE_LABELS.get(mode, mode)


def source_dir(mode: str) -> Path:
    return OUT_ROOT / mode / "source_data"


def province_dir(mode: str, province_en: str) -> Path:
    return OUT_ROOT / mode / slugify(province_en)


def output_stem(mode: str, province_en: str, dw_h: float) -> str:
    return f"{FIG_PREFIX}_{mode}_{slugify(province_en)}_dw{int(dw_h)}h"


def dense_cache_paths(mode: str, province_en: str, dw_h: float) -> tuple[Path, Path]:
    stem = output_stem(mode, province_en, dw_h)
    sdir = source_dir(mode)
    return sdir / f"{stem}_dense_summary.csv", sdir / f"{stem}_dense_frontier_top10.csv"


def selected_paths(mode: str, province_en: str, dw_h: float) -> tuple[Path, Path]:
    stem = output_stem(mode, province_en, dw_h)
    sdir = source_dir(mode)
    return sdir / f"{stem}_selected_top10_summary.csv", sdir / f"{stem}_curve_source.csv"


def existing_dense_paths(mode: str, province_en: str, dw_h: float) -> tuple[Path, Path]:
    slug = slugify(province_en)
    stem = f"{dense.FIG_PREFIX}_{mode}_{slug}_dw{int(dw_h)}h"
    sdir = dense.OUT_ROOT / mode / "source_data"
    return sdir / f"{stem}_summary.csv", sdir / f"{stem}_frontier.csv"


def is_usable_dense_cache(summary: pd.DataFrame, capacity_grid_n: int) -> bool:
    if summary.empty or "capacity_grid_n" not in summary.columns:
        return False
    vals = pd.to_numeric(summary["capacity_grid_n"], errors="coerce").dropna().unique()
    return len(vals) > 0 and int(round(float(vals[0]))) == int(capacity_grid_n)


def load_or_compute_dense(
    province_filter: str,
    dw_h: float,
    mode: str,
    capacity_grid_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    champion_map = sparse.load_champion_map(province_filter)
    province_en, province_cn = dense.resolve_province(champion_map, province_filter, dw_h)

    own_summary_fp, own_frontier_fp = dense_cache_paths(mode, province_en, dw_h)
    if own_summary_fp.exists() and own_frontier_fp.exists():
        summary = pd.read_csv(own_summary_fp, low_memory=False)
        if is_usable_dense_cache(summary, capacity_grid_n):
            return summary, pd.read_csv(own_frontier_fp, low_memory=False), province_en, province_cn

    old_summary_fp, old_frontier_fp = existing_dense_paths(mode, province_en, dw_h)
    if old_summary_fp.exists() and old_frontier_fp.exists():
        summary = pd.read_csv(old_summary_fp, low_memory=False)
        if is_usable_dense_cache(summary, capacity_grid_n):
            return summary, pd.read_csv(old_frontier_fp, low_memory=False), province_en, province_cn

    summary, frontiers, province_en, province_cn = dense.evaluate_dense(
        province_filter=province_filter,
        dw_h=float(dw_h),
        quota_mode=mode,
        capacity_grid_n=int(capacity_grid_n),
    )
    return summary, frontiers, province_en, province_cn


def select_top10(summary: pd.DataFrame, row_idx: int) -> pd.DataFrame:
    panel = summary[
        summary["row_idx"].eq(int(row_idx))
        & summary["point_type"].eq("dense_hybrid_ratio")
    ].copy()
    panel = panel[np.isfinite(pd.to_numeric(panel["osr_pct"], errors="coerce"))]
    if panel.empty:
        return panel
    panel["osr_pct"] = panel["osr_pct"].astype(float)
    panel["lfp_discharge_share_pct"] = panel["lfp_discharge_share_pct"].astype(float)
    panel = panel.sort_values(
        ["osr_pct", "lfp_discharge_share_pct"],
        ascending=[False, True],
    ).head(TOP_K).copy()
    panel.insert(0, "top10_rank", np.arange(1, len(panel) + 1))
    return panel


def actual_ratio_label(row: pd.Series) -> str:
    lfp = row.get("actual_lfp_supply_share_pct")
    caes = row.get("actual_caes_supply_share_pct")
    if pd.notna(lfp) and pd.notna(caes):
        return f"LFP{float(lfp):.1f}-CAES{float(caes):.1f}"
    target = row.get("lfp_discharge_share_pct")
    if pd.notna(target):
        return f"LFP{float(target):.0f}-CAES{100.0 - float(target):.0f}"
    return "LFP+CAES"


def target_ratio_label(row: pd.Series) -> str:
    target = row.get("lfp_discharge_share_pct")
    if pd.isna(target):
        return "T?"
    return f"T{float(target):.0f}/{100.0 - float(target):.0f}"


def panel_top_frontier(frontiers: pd.DataFrame, top: pd.DataFrame, row_idx: int) -> pd.DataFrame:
    if frontiers.empty or top.empty:
        return pd.DataFrame()
    shares = set(top["lfp_discharge_share_pct"].astype(float).round(6).tolist())
    out = frontiers[
        frontiers["row_idx"].eq(int(row_idx))
        & frontiers["lfp_discharge_share_pct"].astype(float).round(6).isin(shares)
    ].copy()
    return out


def build_curve_source(top: pd.DataFrame, frontiers: pd.DataFrame, row_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[pd.DataFrame] = []
    top_frontiers: list[pd.DataFrame] = []
    for rec in top.itertuples(index=False):
        share = float(rec.lfp_discharge_share_pct)
        front = frontiers[
            frontiers["row_idx"].eq(int(row_idx))
            & np.isclose(frontiers["lfp_discharge_share_pct"].astype(float), share)
        ].copy()
        if front.empty:
            continue
        curve = marginal.marginal_from_system_cost_frontier(front)
        if curve.empty:
            continue
        meta = top[top["lfp_discharge_share_pct"].astype(float).eq(share)].iloc[0].to_dict()
        for key, value in meta.items():
            if key not in curve.columns:
                curve[key] = value
        curve_rows.append(curve)
        top_frontiers.append(front)
    curve_source = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    top_frontier = pd.concat(top_frontiers, ignore_index=True) if top_frontiers else pd.DataFrame()
    return curve_source, top_frontier


def y_limits(curves: pd.DataFrame, y_max: float) -> tuple[float, float]:
    y = pd.to_numeric(curves.get("marginal_cost_ratio_to_gas", pd.Series(dtype=float)), errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    if y.empty:
        return 0.0, y_max
    return 0.0, max(float(y_max), min(6.0, float(y.quantile(0.985)) * 1.05))


def render_figure(
    mode: str,
    province_en: str,
    province_cn: str,
    dw_h: float,
    summary: pd.DataFrame,
    frontiers: pd.DataFrame,
    formats: list[str],
    dpi: int,
    y_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fig = plt.figure(figsize=(16.5, 19.5))
    gs = fig.add_gridspec(
        4, 2, left=0.070, right=0.985, top=0.920, bottom=0.055,
        wspace=0.18, hspace=0.31,
    )
    fig.suptitle(
        f"Top-10 1%-optimized LFP+CAES gas-parity OSR curves, {mode_label(mode)} - {province_en}, DW={int(dw_h)}h",
        fontsize=14.0,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.070,
        0.949,
        "Each panel scans target LFP discharge shares from 1% to 99%, selects the ten highest gas-parity OSR ratios, "
        "and plots their marginal cost / gas cost curves. Legend ratios are actual dispatched supply shares; T is the target quota.",
        ha="left",
        va="top",
        fontsize=8.0,
    )

    all_selected = []
    all_curves = []
    all_frontiers = []
    colors = plt.cm.tab10(np.linspace(0, 1, TOP_K))

    for panel_idx, (experiment, stem, row_idx, row_label) in enumerate(seasonal.plotter.SCAN_ROWS):
        ax = fig.add_subplot(gs[int(row_idx) // 2, int(row_idx) % 2])
        top = select_top10(summary, int(row_idx))
        curves, top_frontier = build_curve_source(top, frontiers, int(row_idx))
        if not top.empty:
            top = top.copy()
            top["province_en"] = province_en
            top["province_cn"] = province_cn
            top["quota_mode"] = mode
            top["dispatch_window_h"] = float(dw_h)
            all_selected.append(top)
        if not curves.empty:
            all_curves.append(curves)
        if not top_frontier.empty:
            all_frontiers.append(top_frontier)

        ax.axhline(1.0, color="#c74f4f", lw=1.15, ls="--", alpha=0.88, zorder=1)
        ax.set_xlim(0, 100)
        ax.set_ylim(*y_limits(curves, y_max))
        ax.grid(True, which="major", lw=0.35, color="#d8d8d8", alpha=0.80)
        ax.set_title(
            f"{base.panel_label(panel_idx)} {row_label}",
            loc="left",
            fontsize=9.6,
            fontweight="bold",
        )
        ax.set_xlabel("Peaking demand coverage / supply ratio (%)", fontsize=8.4)
        ax.set_ylabel("Marginal cost / gas cost", fontsize=8.4)
        ax.tick_params(axis="both", labelsize=7.5)

        if top.empty or curves.empty:
            ax.text(0.5, 0.5, "No top-10 curve data", transform=ax.transAxes, ha="center", va="center")
            continue

        legend_handles = []
        legend_labels = []
        for i, rec in enumerate(top.itertuples(index=False)):
            share = float(rec.lfp_discharge_share_pct)
            curve = curves[np.isclose(curves["lfp_discharge_share_pct"].astype(float), share)].copy()
            if curve.empty:
                continue
            curve = curve.sort_values("supply_pct")
            color = colors[(int(getattr(rec, "top10_rank")) - 1) % TOP_K]
            label = (
                f"#{int(getattr(rec, 'top10_rank'))} {actual_ratio_label(pd.Series(rec._asdict()))} "
                f"({target_ratio_label(pd.Series(rec._asdict()))}, OSR={float(rec.osr_pct):.1f}%)"
            )
            line, = ax.plot(
                curve["supply_pct"],
                curve["marginal_cost_ratio_to_gas"],
                color=color,
                lw=1.45 if i else 2.20,
                alpha=0.96,
                zorder=8 if i == 0 else 5,
            )
            osr = float(rec.osr_pct)
            if np.isfinite(osr):
                ax.vlines(osr, 0.0, 1.0, color=color, lw=0.7, alpha=0.45, zorder=2)
                ax.scatter([osr], [1.0], s=20, marker="o", facecolor=color, edgecolor="white",
                           linewidth=0.45, alpha=0.98, zorder=12)
            legend_handles.append(line)
            legend_labels.append(label)

        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            fontsize=5.1,
            handlelength=1.45,
            ncol=2,
            columnspacing=0.55,
            handletextpad=0.32,
            borderaxespad=0.28,
            labelspacing=0.16,
            frameon=True,
            framealpha=0.84,
            edgecolor="#d0d0d0",
        )

    out_dir = province_dir(mode, province_en)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(mode, province_en, dw_h)
    for fmt in formats:
        fp = out_dir / f"{stem}.{fmt}"
        fig.savefig(fp, dpi=dpi if fmt.lower() == "png" else None)
        print(f"  saved: {fp.relative_to(PAPER_CODE_DIR)}", flush=True)
    plt.close(fig)

    selected = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    curve_source = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    top_frontier = pd.concat(all_frontiers, ignore_index=True) if all_frontiers else pd.DataFrame()
    return selected, curve_source, top_frontier


def run_one_task(task: tuple[str, float, str, int, list[str], int, float]) -> dict:
    province_filter, dw_h, mode, capacity_grid_n, formats, dpi, y_max = task
    try:
        summary, frontiers, province_en, province_cn = load_or_compute_dense(
            province_filter=province_filter,
            dw_h=float(dw_h),
            mode=mode,
            capacity_grid_n=int(capacity_grid_n),
        )
        selected, curves, top_frontier = render_figure(
            mode=mode,
            province_en=province_en,
            province_cn=province_cn,
            dw_h=float(dw_h),
            summary=summary,
            frontiers=frontiers,
            formats=formats,
            dpi=int(dpi),
            y_max=float(y_max),
        )

        sdir = source_dir(mode)
        sdir.mkdir(parents=True, exist_ok=True)
        dense_summary_fp, dense_frontier_fp = dense_cache_paths(mode, province_en, dw_h)
        selected_fp, curves_fp = selected_paths(mode, province_en, dw_h)
        top_frontier_fp = sdir / f"{output_stem(mode, province_en, dw_h)}_top10_frontier.csv"

        summary.to_csv(dense_summary_fp, index=False)
        top_frontier.to_csv(dense_frontier_fp, index=False)
        selected.to_csv(selected_fp, index=False)
        curves.to_csv(curves_fp, index=False)
        top_frontier.to_csv(top_frontier_fp, index=False)

        return {
            "ok": True,
            "mode": mode,
            "province_en": province_en,
            "province_cn": province_cn,
            "dispatch_window_h": float(dw_h),
            "selected_rows": int(len(selected)),
            "curve_rows": int(len(curves)),
            "top_frontier_rows": int(len(top_frontier)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "mode": mode,
            "province_filter": province_filter,
            "dispatch_window_h": float(dw_h),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def province_filters_from_champions(province_filter: str | None) -> list[str]:
    cmap = sparse.load_champion_map(province_filter)
    provinces = sorted({prov for (_, _, prov) in cmap})
    return provinces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot top-10 1%-optimized LFP+CAES discharge-ratio gas-parity OSR curves."
    )
    parser.add_argument("--province", default=None, help="Optional province filter, e.g. Anhui")
    parser.add_argument("--modes", nargs="+", default=["hard_quota", "soft_quota"], choices=sorted(frontier.QUOTA_MODE_CODES))
    parser.add_argument("--dw-values", nargs="+", type=float, default=DW_VALUES)
    parser.add_argument("--capacity-grid-n", type=int, default=21)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--y-max", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    provinces = province_filters_from_champions(args.province)
    if not provinces:
        raise SystemExit("No provinces matched.")

    tasks = [
        (prov, float(dw_h), mode, int(args.capacity_grid_n), list(args.formats), int(args.dpi), float(args.y_max))
        for mode in args.modes
        for prov in provinces
        for dw_h in args.dw_values
    ]
    print(
        f"=== top10 1pct optimal gas-parity run: provinces={len(provinces)}, "
        f"modes={args.modes}, dw={args.dw_values}, tasks={len(tasks)}, workers={args.workers} ===",
        flush=True,
    )

    results = []
    if int(args.workers) <= 1:
        for i, task in enumerate(tasks, start=1):
            print(f"=== task {i}/{len(tasks)}: {task[2]} {task[0]} DW={task[1]:.0f}h ===", flush=True)
            res = run_one_task(task)
            results.append(res)
            print(res, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            future_map = {pool.submit(run_one_task, task): task for task in tasks}
            for i, future in enumerate(as_completed(future_map), start=1):
                res = future.result()
                results.append(res)
                status = "ok" if res.get("ok") else "failed"
                print(f"=== completed {i}/{len(tasks)} [{status}]: {res} ===", flush=True)

    summary_fp = OUT_ROOT / "top10_1pct_optimal_run_summary.csv"
    pd.DataFrame(results).to_csv(summary_fp, index=False)
    failures = [r for r in results if not r.get("ok")]
    print(f"=== done: ok={len(results) - len(failures)}, failed={len(failures)} ===", flush=True)
    print(f"summary: {summary_fp.relative_to(PAPER_CODE_DIR)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
