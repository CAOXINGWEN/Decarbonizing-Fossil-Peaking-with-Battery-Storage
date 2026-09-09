"""
M4 LFP+CAES learning-curve gas-parity curves.

This post-processes the hourly-dispatch top-10 hard/soft quota frontiers by
applying the first-paper China storage learning-curve cost factors from
config.CONFIG["year_costs"]. Dispatch, capacity combinations, and supplied
energy stay fixed; annualized storage cost is shifted to each target year and
the marginal gas-parity OSR is recalculated.

Output root:
  output/m4_economics_learning_curve/nature_figure/
    freqlen_per_province_osr_seasonal_mc_no_lfp4h_learning_curve_top10_by_dw/
"""

from __future__ import annotations

from pathlib import Path
import argparse
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG
import m4_hybrid_lfp_caes_discharge_ratio_frontier as frontier
import m4_visualize_freqlen_per_province_osr as base
import m4_visualize_freqlen_seasonal_mc_per_province_marginal_osr_no_lfp4h as marginal
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal
import m4_visualize_lfp_caes_discharge_ratio_top10_optimal_gas_parity_by_dw as baseline_top10


PAPER_CODE_DIR = Path(__file__).resolve().parent
BASE_ROOT = baseline_top10.OUT_ROOT
OUT_ROOT = (
    PAPER_CODE_DIR
    / "output"
    / "m4_economics_learning_curve"
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h_learning_curve_top10_by_dw"
)
FIG_PREFIX = (
    "fig5_freqlen_seasonal_mc_marginal_osr_lfp_caes_learning_curve_"
    "top10_1pct_optimal_gas_parity_linear_full_tail_no_lfp4h"
)


def mode_label(mode: str) -> str:
    return frontier.QUOTA_MODE_LABELS.get(mode, mode)


def year_costs() -> dict[int, float]:
    return {int(year): float(cost) for year, cost in CONFIG["year_costs"].items()}


def cost_factor(year: int) -> float:
    costs = year_costs()
    if int(year) not in costs:
        raise ValueError(f"Year {year} is not in CONFIG['year_costs'].")
    base_cost = float(CONFIG["cost"]["storage_cost_per_kwh"])
    return costs[int(year)] / base_cost


def output_stem(mode: str, province_en: str, dw_h: float, year: int) -> str:
    return (
        f"{FIG_PREFIX}_year{int(year)}_{mode}_"
        f"{baseline_top10.slugify(province_en)}_dw{int(dw_h)}h"
    )


def output_dir(mode: str, province_en: str, year: int) -> Path:
    return OUT_ROOT / f"year_{int(year)}" / mode / baseline_top10.slugify(province_en)


def source_dir(mode: str, year: int) -> Path:
    return OUT_ROOT / f"year_{int(year)}" / mode / "source_data"


def base_source_paths(mode: str, province_en: str, dw_h: float) -> tuple[Path, Path]:
    stem = baseline_top10.output_stem(mode, province_en, float(dw_h))
    sdir = BASE_ROOT / mode / "source_data"
    selected = sdir / f"{stem}_selected_top10_summary.csv"
    frontier_fp = sdir / f"{stem}_top10_frontier.csv"
    return selected, frontier_fp


def apply_learning_cost(front: pd.DataFrame, year: int) -> pd.DataFrame:
    factor = cost_factor(year)
    out = front.copy()
    out["learning_year"] = int(year)
    out["storage_cost_2025_cny_kwh"] = float(CONFIG["cost"]["storage_cost_per_kwh"])
    out["storage_cost_learning_cny_kwh"] = year_costs()[int(year)]
    out["learning_cost_factor_vs_2025"] = factor
    out["annual_cost_cny_2025"] = pd.to_numeric(out["annual_cost_cny"], errors="coerce")
    out["annual_cost_cny"] = out["annual_cost_cny_2025"] * factor

    supply = pd.to_numeric(out["annual_supply_kwh"], errors="coerce")
    gas = pd.to_numeric(out["gas_unit_cost_cny_kwh"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["unit_cost_cny_kwh"] = out["annual_cost_cny"] / supply
        out.loc[supply <= 0, "unit_cost_cny_kwh"] = np.nan
        out["cost_ratio_to_gas"] = out["unit_cost_cny_kwh"] / gas
    return out


def label_for_row(row: pd.Series) -> str:
    lfp_actual = row.get("actual_lfp_supply_share_pct")
    caes_actual = row.get("actual_caes_supply_share_pct")
    target = row.get("lfp_discharge_share_pct")
    lfp_h = row.get("lfp_duration_h")
    caes_h = row.get("caes_duration_h")
    osr = row.get("osr_pct_learning")

    if pd.notna(lfp_actual) and pd.notna(caes_actual):
        actual = f"LFP{float(lfp_actual):.1f}-CAES{float(caes_actual):.1f}"
    elif pd.notna(target):
        actual = f"LFP{float(target):.0f}-CAES{100.0 - float(target):.0f}"
    else:
        actual = "LFP+CAES"

    if pd.notna(lfp_h) and pd.notna(caes_h):
        duration = f"{float(lfp_h):.0f}h/{float(caes_h):.0f}h"
    else:
        duration = "?h/?h"

    if pd.notna(target):
        target_label = f"T{float(target):.0f}/{100.0 - float(target):.0f}"
    else:
        target_label = "T?"

    if pd.notna(osr):
        return f"#{int(row['top10_rank_learning'])} {actual} ({duration}, {target_label}, OSR={float(osr):.1f}%)"
    return f"#{int(row['top10_rank_learning'])} {actual} ({duration}, {target_label})"


def rebuild_panel_curves(
    selected: pd.DataFrame,
    frontiers: pd.DataFrame,
    row_idx: int,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top = selected[selected["row_idx"].eq(int(row_idx))].copy()
    if top.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    curve_rows: list[pd.DataFrame] = []
    frontier_rows: list[pd.DataFrame] = []
    selected_rows: list[dict] = []

    for rec in top.to_dict(orient="records"):
        share = float(rec["lfp_discharge_share_pct"])
        front = frontiers[
            frontiers["row_idx"].eq(int(row_idx))
            & np.isclose(frontiers["lfp_discharge_share_pct"].astype(float), share)
        ].copy()
        if front.empty:
            continue

        shifted_front = apply_learning_cost(front, year)
        curve = marginal.marginal_from_system_cost_frontier(shifted_front)
        osr = marginal.osr_from_marginal_curve(curve)

        rec = dict(rec)
        rec["learning_year"] = int(year)
        rec["storage_cost_learning_cny_kwh"] = year_costs()[int(year)]
        rec["learning_cost_factor_vs_2025"] = cost_factor(year)
        rec["osr_pct_2025_baseline"] = rec.get("osr_pct")
        rec["osr_pct_learning"] = osr
        selected_rows.append(rec)

        if not curve.empty:
            for key, value in rec.items():
                if key not in curve.columns:
                    curve[key] = value
            curve_rows.append(curve)
        frontier_rows.append(shifted_front)

    panel_selected = pd.DataFrame(selected_rows)
    if not panel_selected.empty:
        panel_selected = panel_selected.sort_values(
            ["osr_pct_learning", "lfp_discharge_share_pct"],
            ascending=[False, True],
        ).reset_index(drop=True)
        panel_selected["top10_rank_learning"] = np.arange(1, len(panel_selected) + 1)

    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    if not curves.empty and not panel_selected.empty:
        rank_map = panel_selected.set_index("lfp_discharge_share_pct")[
            ["top10_rank_learning", "osr_pct_learning"]
        ].to_dict(orient="index")
        curves["top10_rank_learning"] = curves["lfp_discharge_share_pct"].map(
            lambda v: rank_map.get(float(v), {}).get("top10_rank_learning", np.nan)
        )
        curves["osr_pct_learning"] = curves["lfp_discharge_share_pct"].map(
            lambda v: rank_map.get(float(v), {}).get("osr_pct_learning", np.nan)
        )

    shifted_frontiers = pd.concat(frontier_rows, ignore_index=True) if frontier_rows else pd.DataFrame()
    return panel_selected, curves, shifted_frontiers


def y_limits(curves: pd.DataFrame) -> tuple[float, float]:
    if curves.empty or "marginal_cost_ratio_to_gas" not in curves.columns:
        return 0.0, 1.25
    y = pd.to_numeric(curves["marginal_cost_ratio_to_gas"], errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    if y.empty:
        return 0.0, 1.25
    return 0.0, max(1.25, float(y.max()) * 1.05)


def render_figure(
    mode: str,
    province_en: str,
    province_cn: str,
    dw_h: float,
    year: int,
    selected: pd.DataFrame,
    frontiers: pd.DataFrame,
    formats: list[str],
    dpi: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fig = plt.figure(figsize=(16.5, 19.5))
    gs = fig.add_gridspec(
        4, 2, left=0.070, right=0.985, top=0.920, bottom=0.055,
        wspace=0.18, hspace=0.31,
    )
    factor = cost_factor(year)
    cost = year_costs()[int(year)]
    fig.suptitle(
        f"LFP+CAES learning-curve marginal gas-parity OSR, {mode_label(mode)} - "
        f"{province_en}, DW={int(dw_h)}h, year={int(year)}",
        fontsize=14.0,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.070,
        0.949,
        f"Storage annualized cost is shifted by the first-paper China storage learning curve "
        f"({cost:.1f} CNY/kWh, {factor:.3f}x of 2025). Dispatch and supply frontiers are kept fixed; "
        "OSR is recalculated from each marginal cost / gas cost curve.",
        ha="left",
        va="top",
        fontsize=8.0,
    )

    all_selected: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    all_frontiers: list[pd.DataFrame] = []
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for panel_idx, (_, _, row_idx, row_label) in enumerate(seasonal.plotter.SCAN_ROWS):
        ax = fig.add_subplot(gs[int(row_idx) // 2, int(row_idx) % 2])
        panel_selected, curves, shifted_frontiers = rebuild_panel_curves(
            selected=selected,
            frontiers=frontiers,
            row_idx=int(row_idx),
            year=int(year),
        )
        if not panel_selected.empty:
            panel_selected["province_en"] = province_en
            panel_selected["province_cn"] = province_cn
            panel_selected["quota_mode"] = mode
            panel_selected["dispatch_window_h"] = float(dw_h)
            all_selected.append(panel_selected)
        if not curves.empty:
            all_curves.append(curves)
        if not shifted_frontiers.empty:
            all_frontiers.append(shifted_frontiers)

        ax.axhline(1.0, color="#c74f4f", lw=1.15, ls="--", alpha=0.88, zorder=1)
        ax.set_xlim(0, 100)
        ax.set_ylim(*y_limits(curves))
        ax.grid(True, which="major", lw=0.35, color="#d8d8d8", alpha=0.80)
        ax.set_title(
            f"{base.panel_label(panel_idx)} {row_label}",
            loc="left",
            fontsize=9.6,
            fontweight="bold",
        )
        ax.set_xlabel("OSR supply ratio (%)", fontsize=8.4)
        ax.set_ylabel("Marginal cost / gas cost", fontsize=8.4)
        ax.tick_params(axis="both", labelsize=7.5)

        if panel_selected.empty or curves.empty:
            ax.text(0.5, 0.5, "No curve data", transform=ax.transAxes, ha="center", va="center")
            continue

        legend_handles = []
        legend_labels = []
        for _, rec in panel_selected.iterrows():
            share = float(rec["lfp_discharge_share_pct"])
            curve = curves[np.isclose(curves["lfp_discharge_share_pct"].astype(float), share)].copy()
            if curve.empty:
                continue
            curve = curve.sort_values("supply_pct")
            rank = int(rec["top10_rank_learning"])
            color = colors[(rank - 1) % len(colors)]
            line, = ax.plot(
                curve["supply_pct"],
                curve["marginal_cost_ratio_to_gas"],
                color=color,
                lw=2.20 if rank == 1 else 1.45,
                alpha=0.96,
                zorder=8 if rank == 1 else 5,
            )
            osr = float(rec["osr_pct_learning"])
            if np.isfinite(osr):
                ax.vlines(osr, 0.0, 1.0, color=color, lw=0.7, alpha=0.45, zorder=2)
                ax.scatter(
                    [osr], [1.0], s=20, marker="o", facecolor=color,
                    edgecolor="white", linewidth=0.45, alpha=0.98, zorder=12,
                )
            legend_handles.append(line)
            legend_labels.append(label_for_row(rec))

        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            fontsize=5.0,
            handlelength=1.35,
            ncol=2,
            columnspacing=0.55,
            handletextpad=0.32,
            borderaxespad=0.28,
            labelspacing=0.16,
            frameon=True,
            framealpha=0.84,
            edgecolor="#d0d0d0",
        )

    out_dir = output_dir(mode, province_en, year)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(mode, province_en, dw_h, year)
    for fmt in formats:
        fp = out_dir / f"{stem}.{fmt}"
        fig.savefig(fp, dpi=dpi if fmt.lower() == "png" else None)
        print(f"  saved: {fp.relative_to(PAPER_CODE_DIR)}", flush=True)
    plt.close(fig)

    selected_out = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    curves_out = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    frontiers_out = pd.concat(all_frontiers, ignore_index=True) if all_frontiers else pd.DataFrame()
    return selected_out, curves_out, frontiers_out


def run_one(mode: str, province_en: str, dw_h: float, year: int, formats: list[str], dpi: int) -> dict:
    try:
        selected_fp, frontier_fp = base_source_paths(mode, province_en, dw_h)
        if not selected_fp.exists():
            raise FileNotFoundError(selected_fp)
        if not frontier_fp.exists():
            raise FileNotFoundError(frontier_fp)

        selected = pd.read_csv(selected_fp, low_memory=False)
        frontiers = pd.read_csv(frontier_fp, low_memory=False)
        province_cn = str(selected["province_cn"].dropna().iloc[0]) if "province_cn" in selected.columns else ""

        selected_out, curves_out, frontiers_out = render_figure(
            mode=mode,
            province_en=province_en,
            province_cn=province_cn,
            dw_h=float(dw_h),
            year=int(year),
            selected=selected,
            frontiers=frontiers,
            formats=formats,
            dpi=int(dpi),
        )

        sdir = source_dir(mode, year)
        sdir.mkdir(parents=True, exist_ok=True)
        stem = output_stem(mode, province_en, dw_h, year)
        selected_out.to_csv(sdir / f"{stem}_selected_summary.csv", index=False)
        curves_out.to_csv(sdir / f"{stem}_curve_source.csv", index=False)
        frontiers_out.to_csv(sdir / f"{stem}_shifted_frontier.csv", index=False)

        return {
            "ok": True,
            "mode": mode,
            "province_en": province_en,
            "province_cn": province_cn,
            "dispatch_window_h": float(dw_h),
            "learning_year": int(year),
            "storage_cost_learning_cny_kwh": year_costs()[int(year)],
            "learning_cost_factor_vs_2025": cost_factor(year),
            "selected_rows": int(len(selected_out)),
            "curve_rows": int(len(curves_out)),
            "frontier_rows": int(len(frontiers_out)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "mode": mode,
            "province_en": province_en,
            "dispatch_window_h": float(dw_h),
            "learning_year": int(year),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def available_provinces(mode: str, dw_h: float) -> list[str]:
    combined = BASE_ROOT / f"top10_1pct_optimal_selected_summary_dw{int(dw_h)}h.csv"
    if combined.exists():
        df = pd.read_csv(combined, usecols=["quota_mode", "province_en", "dispatch_window_h"])
        df = df[
            df["quota_mode"].eq(mode)
            & np.isclose(df["dispatch_window_h"].astype(float), float(dw_h))
        ]
        return sorted(df["province_en"].dropna().astype(str).unique().tolist())

    pattern = f"{baseline_top10.FIG_PREFIX}_{mode}_*_dw{int(dw_h)}h_selected_top10_summary.csv"
    out = []
    for fp in (BASE_ROOT / mode / "source_data").glob(pattern):
        text = fp.name
        prefix = f"{baseline_top10.FIG_PREFIX}_{mode}_"
        suffix = f"_dw{int(dw_h)}h_selected_top10_summary.csv"
        out.append(text.removeprefix(prefix).removesuffix(suffix))
    return sorted(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply 2025-2035 learning-curve cost shifts to top-10 LFP+CAES gas-parity curves."
    )
    parser.add_argument("--province", default=None, help="Optional province filter, e.g. Anhui")
    parser.add_argument("--modes", nargs="+", default=["hard_quota", "soft_quota"], choices=sorted(frontier.QUOTA_MODE_CODES))
    parser.add_argument("--dw-values", nargs="+", type=float, default=[24.0])
    parser.add_argument("--years", nargs="+", type=int, default=sorted(year_costs()))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    cost_rows = [
        {
            "learning_year": int(year),
            "storage_cost_cny_kwh": year_costs()[int(year)],
            "learning_cost_factor_vs_2025": cost_factor(int(year)),
        }
        for year in args.years
    ]
    pd.DataFrame(cost_rows).to_csv(OUT_ROOT / "learning_curve_year_costs.csv", index=False)

    tasks = []
    for mode in args.modes:
        for dw_h in args.dw_values:
            provinces = available_provinces(mode, float(dw_h))
            if args.province:
                needle = args.province.lower()
                provinces = [p for p in provinces if needle in p.lower()]
            for year in args.years:
                for province in provinces:
                    tasks.append((mode, province, float(dw_h), int(year)))

    if not tasks:
        raise SystemExit("No matching baseline top-10 source files found.")

    print(
        f"=== learning-curve top10 gas-parity run: tasks={len(tasks)}, "
        f"modes={args.modes}, dw={args.dw_values}, years={args.years} ===",
        flush=True,
    )
    results = []
    for i, (mode, province, dw_h, year) in enumerate(tasks, start=1):
        print(f"=== task {i}/{len(tasks)}: year={year} {mode} {province} DW={dw_h:.0f}h ===", flush=True)
        res = run_one(mode, province, dw_h, year, list(args.formats), int(args.dpi))
        results.append(res)
        print(res, flush=True)

    run_summary = pd.DataFrame(results)
    run_summary.to_csv(OUT_ROOT / "learning_curve_run_summary.csv", index=False)
    ok = run_summary["ok"].eq(True).sum() if "ok" in run_summary.columns else 0
    print(f"=== done: ok={int(ok)}, failed={len(run_summary) - int(ok)} ===", flush=True)

    selected_frames = []
    for year in args.years:
        for mode in args.modes:
            sdir = source_dir(mode, int(year))
            for fp in sdir.glob("*_selected_summary.csv"):
                selected_frames.append(pd.read_csv(fp, low_memory=False))
    if selected_frames:
        combined = pd.concat(selected_frames, ignore_index=True)
        for dw_h in args.dw_values:
            sub = combined[np.isclose(combined["dispatch_window_h"].astype(float), float(dw_h))]
            sub.to_csv(
                OUT_ROOT / f"learning_curve_selected_summary_dw{int(dw_h)}h.csv",
                index=False,
            )

    failures = [r for r in results if not r.get("ok")]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
