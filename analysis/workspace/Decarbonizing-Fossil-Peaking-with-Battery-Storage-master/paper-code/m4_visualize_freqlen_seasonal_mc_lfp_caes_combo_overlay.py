"""
M4 Fig 5 seasonal MC province sheets with continuous LFP-CAES combo curves.

This script keeps the original eight single-technology OSR-selected curves from
freqlen_per_province_osr_seasonal_mc_no_lfp4h unchanged, then overlays one
continuous LFP-CAES optimal-mix curve.

For each target peaking supply ratio x, the curve searches all allocations
where:

  lfp_supply_pct + caes_supply_pct = x

The plotted y-axis is the portfolio average unit cost ratio:

  combo_cost_ratio =
      (lfp_supply_pct * lfp_cost_ratio
       + caes_supply_pct * caes_cost_ratio)
      / x

The minimum-cost allocation is selected independently at every x. This avoids
hard OSR splicing and makes the combo curve directly comparable to the original
single-technology unit-cost curves on the same axes. It remains a curve-level
proxy, not a co-optimized hourly hybrid dispatch.

Output:
  output/m4_economics/nature_figure/
    freqlen_per_province_osr_seasonal_mc_no_lfp4h_combo_overlay/
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m4_visualize_freqlen_per_province_osr as base
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal
import m4_visualize_single_tech_generic as ms


seasonal_plotter = seasonal.plotter

PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
INPUT_DIR = (
    M4_DIR
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h"
)
INPUT_SOURCE_DIR = INPUT_DIR / "source_data"
INPUT_CHAMPIONS = INPUT_SOURCE_DIR / seasonal_plotter.CHAMPION_FILENAME

OUT_DIR = (
    M4_DIR
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h_combo_overlay"
)
SOURCE_DIR = OUT_DIR / "source_data"
FIG_FILE_PREFIX = "fig5_freqlen_seasonal_mc_osr_no_lfp4h_combo_overlay"
SUMMARY_FILENAME = f"{FIG_FILE_PREFIX}_combo_summary.csv"

COMBO_ID = "lfp_caes_optimal_mix"
COMBO_LABEL = "LFP+CAES optimal mix"
COMBO_COLOR = "#111111"
COMBO_MARKER = "D"
ALLOCATION_GRID_STEP = 0.25

TECH_LABELS = {
    "li_lfp": "LFP",
    "caes": "CAES",
}


def _finite_curve(curve: pd.DataFrame) -> pd.DataFrame:
    out = curve[["supply_pct", "cost_ratio_to_gas"]].dropna().copy()
    out = out[np.isfinite(out["supply_pct"]) & np.isfinite(out["cost_ratio_to_gas"])]
    out = out.sort_values("supply_pct").drop_duplicates("supply_pct", keep="first")
    return out.reset_index(drop=True)


def _curve_with_zero(curve: pd.DataFrame) -> pd.DataFrame:
    """Add a light linear extrapolation anchor at x=0 for continuous combos."""
    curve = _finite_curve(curve)
    if curve.empty:
        return curve
    if curve["supply_pct"].iloc[0] <= 0.0:
        return curve
    if len(curve) < 2:
        first = curve.iloc[0]
        anchor = pd.DataFrame(
            {"supply_pct": [0.0], "cost_ratio_to_gas": [first["cost_ratio_to_gas"]]}
        )
        return pd.concat([anchor, curve], ignore_index=True)

    x0, x1 = curve["supply_pct"].iloc[:2].to_numpy(dtype=float)
    y0, y1 = curve["cost_ratio_to_gas"].iloc[:2].to_numpy(dtype=float)
    if x1 == x0:
        y_at_zero = y0
    else:
        slope = (y1 - y0) / (x1 - x0)
        y_at_zero = y0 - slope * x0
    if not np.isfinite(y_at_zero) or y_at_zero <= 0.0:
        y_at_zero = y0
    y_at_zero = max(float(y_at_zero), 1e-6)
    anchor = pd.DataFrame(
        {"supply_pct": [0.0], "cost_ratio_to_gas": [y_at_zero]}
    )
    return pd.concat([anchor, curve], ignore_index=True)


def _interp_y(curve: pd.DataFrame, x_value: float) -> float:
    curve = _curve_with_zero(curve)
    if curve.empty:
        return float("nan")
    x = curve["supply_pct"].to_numpy(dtype=float)
    y = curve["cost_ratio_to_gas"].to_numpy(dtype=float)
    x_value = float(x_value)
    if x_value <= x.min():
        return float(y[0])
    if x_value >= x.max():
        return float(y[-1])
    return float(np.interp(x_value, x, y))


def _meta_for_tech(source: pd.DataFrame, tech_id: str) -> dict | None:
    sub = source[source["tech_id"].eq(tech_id)]
    if sub.empty:
        return None
    return {
        "tech_id": tech_id,
        "tech_label": str(sub["tech_label"].dropna().iloc[0]),
        "duration_h": int(sub["duration_h"].dropna().iloc[0]),
        "osr_pct": float(sub["osr_pct"].dropna().iloc[0]),
        "curve": sub[["supply_pct", "cost_ratio_to_gas"]].copy(),
    }


def _marginal_osr_from_average_curve(curve: pd.DataFrame) -> float:
    """Approximate OSR from the combo average-cost curve.

    With gas cost normalized to 1, total cost proxy is x * average_cost_ratio.
    The marginal cost ratio is d(total_cost_proxy) / d(x).
    """
    clean = curve[["combo_supply_pct", "cost_ratio_to_gas"]].dropna().copy()
    clean = clean.sort_values("combo_supply_pct").drop_duplicates(
        "combo_supply_pct", keep="first"
    )
    x = clean["combo_supply_pct"].to_numpy(dtype=float)
    y = clean["cost_ratio_to_gas"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return float("nan")

    total_cost = x * y
    dx = np.diff(x)
    dc = np.diff(total_cost)
    valid_dx = dx > 0
    if not np.any(valid_dx):
        return float("nan")
    marginal = dc[valid_dx] / dx[valid_dx]
    x_mid = ((x[:-1] + x[1:]) / 2.0)[valid_dx]

    finite = np.isfinite(marginal) & np.isfinite(x_mid)
    marginal = marginal[finite]
    x_mid = x_mid[finite]
    if len(x_mid) < 2:
        return float("nan")

    diff = marginal - 1.0
    changes = np.where(np.diff(np.sign(diff)))[0]
    for idx in changes:
        x0, x1 = x_mid[idx], x_mid[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        if d1 == d0:
            return float(x0)
        return float(x0 - d0 * (x1 - x0) / (d1 - d0))

    if np.all(diff <= 0):
        return float(x_mid.max())
    return 0.0


def build_optimal_mix_curve(panel_source: pd.DataFrame) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """Build the minimum-average-cost LFP+CAES mix curve for one panel."""
    lfp_meta = _meta_for_tech(panel_source, "li_lfp")
    caes_meta = _meta_for_tech(panel_source, "caes")
    if lfp_meta is None or caes_meta is None:
        return None, None

    lfp_curve = _curve_with_zero(lfp_meta["curve"])
    caes_curve = _curve_with_zero(caes_meta["curve"])
    if lfp_curve.empty or caes_curve.empty:
        return None, None

    lfp_x = lfp_curve["supply_pct"].to_numpy(dtype=float)
    lfp_y = lfp_curve["cost_ratio_to_gas"].to_numpy(dtype=float)
    caes_x = caes_curve["supply_pct"].to_numpy(dtype=float)
    caes_y = caes_curve["cost_ratio_to_gas"].to_numpy(dtype=float)
    lfp_max = float(np.nanmax(lfp_x))
    caes_max = float(np.nanmax(caes_x))
    combo_max = min(100.0, lfp_max + caes_max)
    if not np.isfinite(combo_max) or combo_max <= 0.0:
        return None, None

    x_targets = np.arange(ALLOCATION_GRID_STEP, combo_max + 1e-9, ALLOCATION_GRID_STEP)
    rows = []
    for target in x_targets:
        lo = max(0.0, target - caes_max)
        hi = min(lfp_max, target)
        if hi < lo:
            continue
        lfp_allocs = np.arange(lo, hi + 1e-9, ALLOCATION_GRID_STEP)
        if len(lfp_allocs) == 0 or lfp_allocs[-1] < hi:
            lfp_allocs = np.append(lfp_allocs, hi)
        caes_allocs = target - lfp_allocs

        lfp_cost = np.interp(lfp_allocs, lfp_x, lfp_y)
        caes_cost = np.interp(caes_allocs, caes_x, caes_y)
        avg_cost = (lfp_allocs * lfp_cost + caes_allocs * caes_cost) / target
        finite = np.isfinite(avg_cost)
        if not np.any(finite):
            continue

        best_local = np.argmin(np.where(finite, avg_cost, np.inf))
        best_lfp = float(lfp_allocs[best_local])
        best_caes = float(caes_allocs[best_local])
        rows.append(
            {
                "combo_supply_pct": float(target),
                "cost_ratio_to_gas": float(avg_cost[best_local]),
                "lfp_supply_pct": best_lfp,
                "caes_supply_pct": best_caes,
                "lfp_cost_ratio": float(lfp_cost[best_local]),
                "caes_cost_ratio": float(caes_cost[best_local]),
                "lfp_share": float(best_lfp / target),
                "caes_share": float(best_caes / target),
            }
        )

    if not rows:
        return None, None

    combo = pd.DataFrame(rows)
    combo_osr = _marginal_osr_from_average_curve(combo)
    meta = {
        "lfp_duration_h": int(lfp_meta["duration_h"]),
        "caes_duration_h": int(caes_meta["duration_h"]),
        "lfp_single_osr_pct": float(lfp_meta["osr_pct"]),
        "caes_single_osr_pct": float(caes_meta["osr_pct"]),
        "combo_osr_pct": float(combo_osr),
        "combo_max_supply_pct": float(combo["combo_supply_pct"].max()),
    }
    return combo.reset_index(drop=True), meta


def build_combo_average_curve(
    panel_source: pd.DataFrame,
    first_tech: str,
    second_tech: str,
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    first_meta = _meta_for_tech(panel_source, first_tech)
    second_meta = _meta_for_tech(panel_source, second_tech)
    if first_meta is None or second_meta is None:
        return None, None

    first_curve = _curve_with_zero(first_meta["curve"])
    second_curve = _curve_with_zero(second_meta["curve"])
    if first_curve.empty or second_curve.empty:
        return None, None

    first_osr = min(max(float(first_meta["osr_pct"]), 0.0), 100.0)
    second_max = min(max(float(second_curve["supply_pct"].max()), 0.0), 100.0)

    if first_osr <= 0.0:
        combo = second_curve.rename(columns={"supply_pct": "combo_supply_pct"}).copy()
        combo["first_supply_pct"] = 0.0
        combo["second_residual_supply_pct"] = combo["combo_supply_pct"]
        combo["second_original_curve_supply_pct"] = combo["combo_supply_pct"]
    else:
        y_first_osr = _interp_y(first_curve, first_osr)
        x1 = np.linspace(0.0, first_osr, max(24, int(np.ceil(first_osr * 2.4)) + 1))
        y1 = np.interp(
            x1,
            first_curve["supply_pct"].to_numpy(dtype=float),
            first_curve["cost_ratio_to_gas"].to_numpy(dtype=float),
        )
        stage1 = pd.DataFrame(
            {
                "combo_supply_pct": x1,
                "cost_ratio_to_gas": y1,
                "first_supply_pct": x1,
                "second_residual_supply_pct": 0.0,
                "second_original_curve_supply_pct": 0.0,
            }
        )

        if first_osr >= 100.0 or second_max <= 0.0:
            combo = stage1
        else:
            combo_max = first_osr + (100.0 - first_osr) * second_max / 100.0
            x2 = np.linspace(
                first_osr,
                combo_max,
                max(24, int(np.ceil((combo_max - first_osr) * 2.4)) + 1),
            )
            second_residual_curve_pct = (
                (x2 - first_osr) / (100.0 - first_osr) * 100.0
            )
            y_second = np.interp(
                second_residual_curve_pct,
                second_curve["supply_pct"].to_numpy(dtype=float),
                second_curve["cost_ratio_to_gas"].to_numpy(dtype=float),
            )
            second_supply_pct = x2 - first_osr
            y2 = np.divide(
                first_osr * y_first_osr + second_supply_pct * y_second,
                x2,
                out=np.full_like(x2, y_first_osr, dtype=float),
                where=x2 > 0,
            )
            stage2 = pd.DataFrame(
                {
                    "combo_supply_pct": x2,
                    "cost_ratio_to_gas": y2,
                    "first_supply_pct": first_osr,
                    "second_residual_supply_pct": second_supply_pct,
                    "second_original_curve_supply_pct": second_residual_curve_pct,
                }
            )
            combo = pd.concat([stage1, stage2], ignore_index=True)

    combo = combo.sort_values("combo_supply_pct").drop_duplicates(
        "combo_supply_pct", keep="last"
    )
    combo_osr = _marginal_osr_from_average_curve(combo)
    residual_endpoint_pct = (
        first_osr + (100.0 - first_osr) * min(max(second_meta["osr_pct"], 0.0), 100.0) / 100.0
    )
    meta = {
        "first_tech_id": first_tech,
        "second_tech_id": second_tech,
        "first_duration_h": int(first_meta["duration_h"]),
        "second_duration_h": int(second_meta["duration_h"]),
        "first_osr_pct": first_osr,
        "second_osr_pct": float(second_meta["osr_pct"]),
        "residual_osr_endpoint_pct": float(min(max(residual_endpoint_pct, 0.0), 100.0)),
        "combo_osr_pct": float(combo_osr),
        "combo_max_supply_pct": float(combo["combo_supply_pct"].max()),
    }
    return combo.reset_index(drop=True), meta


def load_province_source(prov_en: str) -> pd.DataFrame:
    slug = base.slugify(prov_en)
    fp = INPUT_SOURCE_DIR / f"{seasonal_plotter.FIG_FILE_PREFIX}_{slug}_source.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_csv(fp)


def render_province(
    prov_cn: str,
    prov_en: str,
    province_source: pd.DataFrame,
    formats: list[str],
    dpi: int,
    save_source: bool,
) -> list[dict]:
    fig = plt.figure(figsize=(18.0, 28.0))
    gs = fig.add_gridspec(
        8,
        4,
        left=0.065,
        right=0.985,
        top=0.960,
        bottom=0.040,
        wspace=0.22,
        hspace=0.42,
    )

    source_rows = []
    summary_rows = []
    panel_idx = 0

    for experiment, stem, row_idx, row_label in seasonal_plotter.SCAN_ROWS:
        for col_idx, dw_h in enumerate(seasonal_plotter.DW_VALUES):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            base.setup_axis(
                ax,
                base.panel_label(panel_idx),
                experiment,
                dw_h,
                show_xlabel=(row_idx == 7),
                show_ylabel=(col_idx == 0),
            )

            panel_source = province_source[
                province_source["stem"].eq(stem)
                & province_source["dispatch_window_h"].eq(float(dw_h))
            ].copy()

            for tech_id in seasonal_plotter.TECH_ORDER:
                rec = panel_source[panel_source["tech_id"].eq(tech_id)]
                if rec.empty:
                    continue
                is_lfp = tech_id == "li_lfp"
                tech_label = str(rec["tech_label"].dropna().iloc[0])
                duration_h = int(rec["duration_h"].dropna().iloc[0])
                osr_pct = float(rec["osr_pct"].dropna().iloc[0])
                label = f"{tech_label} {duration_h}h | OSR {osr_pct:.0f}%"
                ax.plot(
                    rec["supply_pct"],
                    rec["cost_ratio_to_gas"],
                    color=ms.TECH_COLORS[tech_id],
                    lw=1.9 if is_lfp else 1.35,
                    ls="--" if is_lfp else "-",
                    marker="s" if is_lfp else "o",
                    ms=3.0 if is_lfp else 2.4,
                    markevery=30,
                    zorder=3,
                    label=label,
                )
                if save_source:
                    source_rows.append(
                        rec.assign(
                            curve_type="single_tech",
                            combo_id="",
                            combo_label="",
                            combo_osr_pct=np.nan,
                            lfp_duration_h=np.nan,
                            caes_duration_h=np.nan,
                            lfp_single_osr_pct=np.nan,
                            caes_single_osr_pct=np.nan,
                            combo_supply_pct=np.nan,
                            lfp_supply_pct=np.nan,
                            caes_supply_pct=np.nan,
                            lfp_cost_ratio=np.nan,
                            caes_cost_ratio=np.nan,
                            lfp_share=np.nan,
                            caes_share=np.nan,
                        )
                    )

            combo_curve, meta = build_optimal_mix_curve(panel_source)
            if combo_curve is not None:
                label = (
                    f"{COMBO_LABEL} "
                    f"(LFP{meta['lfp_duration_h']}h+CAES{meta['caes_duration_h']}h)"
                    f" | OSR* {meta['combo_osr_pct']:.0f}%"
                )
                ax.plot(
                    combo_curve["combo_supply_pct"],
                    combo_curve["cost_ratio_to_gas"],
                    color=COMBO_COLOR,
                    lw=2.25,
                    ls="-",
                    marker=COMBO_MARKER,
                    ms=3.0,
                    markevery=35,
                    zorder=5,
                    label=label,
                )

                summary_rows.append(
                    {
                        "province_cn": prov_cn,
                        "province_en": prov_en,
                        "experiment": experiment,
                        "stem": stem,
                        "row_idx": row_idx,
                        "row_label": row_label,
                        "dispatch_window_h": float(dw_h),
                        "combo_id": COMBO_ID,
                        "combo_label": COMBO_LABEL,
                        **meta,
                    }
                )
                if save_source:
                    source_rows.append(
                        combo_curve.assign(
                            province_cn=prov_cn,
                            province_en=prov_en,
                            experiment=experiment,
                            stem=stem,
                            row_idx=row_idx,
                            row_label=row_label,
                            dispatch_window_h=float(dw_h),
                            curve_type="combo_average_cost",
                            tech_id="",
                            tech_label="",
                            duration_h=np.nan,
                            osr_pct=np.nan,
                            supply_pct=np.nan,
                            combo_id=COMBO_ID,
                            combo_label=COMBO_LABEL,
                            **meta,
                        )
                    )

            ax.legend(
                loc="upper left",
                fontsize=4.2,
                handlelength=1.05,
                borderaxespad=0.2,
                ncol=1,
                columnspacing=0.35,
                handletextpad=0.30,
                labelspacing=0.14,
            )

            if col_idx == 0:
                ax.text(
                    -0.30,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=9.6,
                    fontweight="bold",
                    color=ms.PALETTE["black"],
                    **ms.cjk_font_kwargs(bold=True),
                )
            panel_idx += 1

    fig.text(
        0.065,
        0.990,
        f"Fig 5 seasonal-MC OSR curves with LFP-CAES optimal mix — {prov_en} ({prov_cn})",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=ms.PALETTE["black"],
        **ms.cjk_font_kwargs(bold=True),
    )
    fig.text(
        0.065,
        0.975,
        "Original eight OSR-selected single-technology curves are unchanged. "
        "Black curve is the continuous LFP+CAES optimal-mix cost proxy: "
        "each x-axis point selects the minimum-cost LFP/CAES allocation at that total supply ratio. "
        "Red dashed line = gas parity.",
        ha="left",
        va="top",
        fontsize=7.4,
        color=ms.PALETTE["black"],
        **ms.cjk_font_kwargs(),
    )

    stem = f"{FIG_FILE_PREFIX}_{base.slugify(prov_en)}"
    for fmt in formats:
        path = OUT_DIR / f"{stem}.{fmt}"
        kw = {"dpi": dpi} if fmt == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved: {path.name}", flush=True)
    plt.close(fig)

    if save_source and source_rows:
        out = pd.concat(source_rows, ignore_index=True)
        out.to_csv(SOURCE_DIR / f"{stem}_source.csv", index=False)

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formats", nargs="+", default=["png"], choices=["png", "svg", "pdf"]
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--province", default=None, help="Optional province_en or province_cn substring."
    )
    parser.add_argument(
        "--no-source", action="store_true", help="Do not write curve source CSVs."
    )
    args = parser.parse_args()

    if not INPUT_CHAMPIONS.exists():
        raise FileNotFoundError(INPUT_CHAMPIONS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    champions = pd.read_csv(INPUT_CHAMPIONS)
    provinces = (
        champions[["province_cn", "province_en"]]
        .drop_duplicates()
        .sort_values("province_en")
        .itertuples(index=False, name=None)
    )
    provinces = list(provinces)
    if args.province:
        needle = args.province.lower()
        provinces = [
            (cn, en)
            for cn, en in provinces
            if needle in str(cn).lower() or needle in str(en).lower()
        ]
    if not provinces:
        raise SystemExit("No province matched.")

    all_summary_rows = []
    for idx, (prov_cn, prov_en) in enumerate(provinces, start=1):
        print(f"=== render {idx}/{len(provinces)}: {prov_en} ({prov_cn}) ===", flush=True)
        source = load_province_source(str(prov_en))
        rows = render_province(
            str(prov_cn),
            str(prov_en),
            source,
            formats=args.formats,
            dpi=args.dpi,
            save_source=not args.no_source,
        )
        all_summary_rows.extend(rows)

    if all_summary_rows:
        summary = pd.DataFrame(all_summary_rows)
        summary = summary.sort_values(
            ["province_en", "row_idx", "dispatch_window_h", "combo_id"]
        )
        summary.to_csv(SOURCE_DIR / SUMMARY_FILENAME, index=False)
        print(f"=== combo summary: {len(summary):,} rows ===", flush=True)


if __name__ == "__main__":
    main()
