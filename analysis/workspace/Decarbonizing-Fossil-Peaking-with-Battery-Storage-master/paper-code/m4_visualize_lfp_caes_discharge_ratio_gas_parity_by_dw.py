"""
Gas-parity diagnostic figures for fixed-discharge-ratio LFP+CAES curves.

This script reads the source data produced by
``m4_visualize_freqlen_seasonal_mc_marginal_osr_lfp_caes_discharge_ratio.py``.
It does not rerun dispatch. For each province and quota mode, it creates one
province-named folder containing four larger figures, one for each discharge
window (8/12/16/24 h). Three variants are supported: a full-tail log-scale
view with a linear gas-parity inset, a linear gas-parity view clipped near the
gas benchmark, and a linear full-tail view with the same gas-parity inset.
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
DW_VALUES = [8.0, 12.0, 16.0, 24.0]
SINGLE_TECH_ORDER = ["caes", "li_lfp"]
HYBRID_SHARES = ratio_plot.DISCHARGE_LFP_SHARES_PCT


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "province"


def source_dir_for_mode(mode: str) -> Path:
    return ratio_plot.output_dir_for_mode(mode) / "source_data"


def source_prefix_for_mode(mode: str) -> str:
    return ratio_plot.fig_prefix_for_mode(mode)


def output_dir_for_mode(mode: str, variant: str = "full_tail") -> Path:
    if variant not in {"full_tail", "linear", "linear_full_tail"}:
        raise ValueError(f"Unknown gas-parity variant: {variant}")
    variant_suffix = {
        "full_tail": "gas_parity_full_tail_by_dw",
        "linear": "gas_parity_linear_by_dw",
        "linear_full_tail": "gas_parity_linear_full_tail_by_dw",
    }[variant]
    return (
        M4_DIR
        / "nature_figure"
        / f"freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_lfp_caes_discharge_ratio_{mode}_{variant_suffix}"
    )


def fig_prefix_for_mode(mode: str, variant: str = "full_tail") -> str:
    if variant not in {"full_tail", "linear", "linear_full_tail"}:
        raise ValueError(f"Unknown gas-parity variant: {variant}")
    variant_token = {
        "full_tail": "gas_parity_full_tail",
        "linear": "gas_parity_linear",
        "linear_full_tail": "gas_parity_linear_full_tail",
    }[variant]
    return f"fig5_freqlen_seasonal_mc_marginal_osr_lfp_caes_discharge_ratio_{mode}_{variant_token}_no_lfp4h"


def load_mode_summary(mode: str) -> pd.DataFrame:
    fp = source_dir_for_mode(mode) / f"{source_prefix_for_mode(mode)}_summary.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp, low_memory=False)
    if "plot_label" not in df.columns:
        df["plot_label"] = df["tech_label"].astype(str)
    return df


def load_province_source(mode: str, province_en: str) -> pd.DataFrame:
    slug = slugify(province_en)
    fp = source_dir_for_mode(mode) / f"{source_prefix_for_mode(mode)}_{slug}_source.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp, low_memory=False)
    if "plot_label" not in df.columns:
        df["plot_label"] = df["tech_label"].astype(str)
    return df


def color_for_series(row: pd.Series):
    tech_id = str(row["tech_id"])
    if tech_id in ms.TECH_COLORS:
        return ms.TECH_COLORS[tech_id]
    share = row.get("lfp_discharge_share_pct")
    if pd.notna(share):
        return ratio_plot.RATIO_COLORS.get(int(round(float(share))), "0.35")
    return "0.35"


def style_for_series(row: pd.Series) -> dict:
    tech_id = str(row["tech_id"])
    if tech_id == "li_lfp":
        return {"lw": 2.25, "ls": "--", "alpha": 0.98, "zorder": 8}
    if tech_id == "caes":
        return {"lw": 2.10, "ls": "-", "alpha": 0.98, "zorder": 8}
    return {"lw": 1.45, "ls": "-", "alpha": 0.90, "zorder": 5}


def format_osr_for_label(osr: float) -> str:
    if not np.isfinite(osr):
        return "OSR=NA"
    return f"OSR={osr:.1f}%"


def label_for_series(row: pd.Series) -> str:
    """Use actual dispatched supply shares for hybrid labels when available."""
    if str(row.get("series_type", "")) != "discharge_ratio_hybrid":
        return str(row["plot_label"])

    lfp_actual = row.get("actual_lfp_supply_share_pct")
    caes_actual = row.get("actual_caes_supply_share_pct")
    if pd.notna(lfp_actual) and pd.notna(caes_actual):
        return f"LFP{float(lfp_actual):.1f}-CAES{float(caes_actual):.1f}"
    return str(row["plot_label"])


def ordered_panel_groups(panel_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    order = []
    for tech_id in SINGLE_TECH_ORDER:
        if panel_df["tech_id"].astype(str).eq(tech_id).any():
            order.append(tech_id)
    for share in HYBRID_SHARES:
        tech_id = f"hybrid_{panel_df['quota_mode'].dropna().iloc[0]}_{share:02d}" if panel_df["quota_mode"].notna().any() else None
        if tech_id and panel_df["tech_id"].astype(str).eq(tech_id).any():
            order.append(tech_id)
    remaining = [
        x for x in panel_df["tech_id"].astype(str).dropna().unique()
        if x not in set(order)
    ]
    order.extend(sorted(remaining))
    return [(tech_id, panel_df[panel_df["tech_id"].astype(str).eq(tech_id)].copy()) for tech_id in order]


def plot_osr_marker(ax, osr: float, color, y_min: float, y_gas: float = 1.0) -> None:
    if not np.isfinite(osr):
        return
    if osr < 0 or osr > 100:
        return
    ax.vlines(osr, y_min, y_gas, color=color, lw=0.75, alpha=0.42, zorder=2)
    ax.scatter([osr], [y_gas], s=20, marker="o", facecolor=color, edgecolor="white",
               linewidth=0.45, alpha=0.98, zorder=12)


def positive_y_limits(panel: pd.DataFrame) -> tuple[float, float]:
    y = pd.to_numeric(panel["marginal_cost_ratio_to_gas"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = y.dropna()
    y = y[y > 0]
    if y.empty:
        return 0.05, 10.0
    y_min = max(0.03, float(y.min()) * 0.80)
    y_max = max(3.0, float(y.max()) * 1.08)
    return y_min, y_max


def render_dw_figure(
    mode: str,
    variant: str,
    province_en: str,
    province_cn: str,
    source_df: pd.DataFrame,
    dw_h: float,
    out_dir: Path,
    fig_prefix: str,
    formats: list[str],
    dpi: int,
    zoom_y_max: float,
    title_suffix: str | None = None,
    method_note: str | None = None,
    summary_extra: dict | None = None,
) -> list[dict]:
    if variant not in {"full_tail", "linear", "linear_full_tail"}:
        raise ValueError(f"Unknown gas-parity variant: {variant}")
    is_linear_zoom = variant == "linear"
    is_linear_full_tail = variant == "linear_full_tail"

    fig = plt.figure(figsize=(16.5, 19.5))
    gs = fig.add_gridspec(
        4, 2, left=0.070, right=0.985, top=0.915, bottom=0.055,
        wspace=0.18, hspace=0.31,
    )

    mode_label = ratio_plot.MODE_LABELS[mode]
    summary_rows: list[dict] = []

    title = f"Gas-parity OSR intersections, {mode_label} - {province_en}, DW={int(dw_h)}h"
    if title_suffix:
        title = f"{title}, {title_suffix}"
    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=0.982,
    )
    note = (
        "Linear y-axis clipped near gas parity. Hybrid legend labels report actual LFP/CAES supply shares; "
        "dots and OSR values mark each curve's gas-cost intersection."
        if is_linear_zoom
        else (
            "Main panels use a linear axis and keep the full high-cost tail. Hybrid legend labels report actual LFP/CAES supply shares; "
            "inset dots mark each curve's OSR where marginal cost equals gas cost."
            if is_linear_full_tail
            else "Main panels keep the full high-cost tail. Hybrid legend labels report actual LFP/CAES supply shares; inset dots mark each curve's OSR where marginal cost equals gas cost."
        )
    )
    if method_note:
        note = f"{method_note} {note}"
    fig.text(
        0.070,
        0.948,
        note,
        ha="left",
        va="top",
        fontsize=8.2,
    )

    panel_idx = 0
    for experiment, stem, row_idx, row_label in seasonal.plotter.SCAN_ROWS:
        ax = fig.add_subplot(gs[int(row_idx) // 2, int(row_idx) % 2])
        panel = source_df[
            source_df["row_idx"].eq(int(row_idx))
            & source_df["dispatch_window_h"].eq(float(dw_h))
        ].copy()

        y_min_full, y_max_full = positive_y_limits(panel) if not panel.empty else (0.05, 10.0)
        ax.axhline(1.0, color="#c74f4f", lw=1.15, ls="--", alpha=0.88, zorder=1)
        ax.set_xlim(0, 100)
        if is_linear_zoom:
            ax.set_ylim(0, zoom_y_max)
        elif is_linear_full_tail:
            ax.set_ylim(0, y_max_full)
        else:
            ax.set_yscale("log")
            ax.set_ylim(y_min_full, y_max_full)
        ax.set_title(
            f"{base.panel_label(panel_idx)} {row_label}",
            loc="left",
            fontsize=9.6,
            fontweight="bold",
        )
        ax.grid(True, which="major", lw=0.35, color="#d8d8d8", alpha=0.80)
        ax.set_xlabel("Peaking demand coverage / supply ratio (%)", fontsize=8.5)
        ax.set_ylabel("Marginal cost / gas cost" + ("" if is_linear_zoom else " (full tail)"), fontsize=8.5)
        ax.tick_params(axis="both", labelsize=7.5)

        if panel.empty:
            ax.text(0.5, 0.5, "No curve data", transform=ax.transAxes, ha="center", va="center")
            panel_idx += 1
            continue

        legend_handles = []
        legend_labels = []
        hybrid_osr = []
        inset = None
        if not is_linear_zoom:
            inset = ax.inset_axes([0.505, 0.075, 0.465, 0.355])
            inset.axhline(1.0, color="#c74f4f", lw=0.85, ls="--", alpha=0.88, zorder=1)
            inset.set_xlim(0, 100)
            inset.set_ylim(0, zoom_y_max)
            inset.grid(True, which="major", lw=0.25, color="#d6d6d6", alpha=0.75)
            inset.tick_params(axis="both", labelsize=5.5, pad=1)
            inset.set_title("gas parity zoom", fontsize=5.8, pad=1.5)
        for tech_id, curve in ordered_panel_groups(panel):
            curve = curve.dropna(subset=["supply_pct", "marginal_cost_ratio_to_gas"]).sort_values("supply_pct")
            if curve.empty:
                continue
            first = curve.iloc[0]
            color = color_for_series(first)
            style = style_for_series(first)
            label = label_for_series(first)
            osr = float(first["osr_pct"]) if pd.notna(first["osr_pct"]) else float("nan")
            legend_label = f"{label} ({format_osr_for_label(osr)})"
            line, = ax.plot(
                curve["supply_pct"],
                curve["marginal_cost_ratio_to_gas"],
                color=color,
                label=legend_label,
                **style,
            )
            if inset is not None:
                inset.plot(
                    curve["supply_pct"],
                    curve["marginal_cost_ratio_to_gas"],
                    color=color,
                    lw=max(0.70, style["lw"] * 0.62),
                    ls=style["ls"],
                    alpha=style["alpha"],
                    zorder=style["zorder"],
                )
            plot_osr_marker(ax, osr, color, y_min=0.0 if (is_linear_zoom or is_linear_full_tail) else y_min_full)
            if inset is not None:
                plot_osr_marker(inset, osr, color, y_min=0.0)
            legend_handles.append(line)
            legend_labels.append(legend_label)
            if str(first.get("series_type", "")) == "discharge_ratio_hybrid":
                hybrid_osr.append(osr)
            row_summary = {
                "quota_mode": mode,
                "province_cn": province_cn,
                "province_en": province_en,
                "experiment": experiment,
                "stem": stem,
                "row_idx": int(row_idx),
                "row_label": row_label,
                "dispatch_window_h": float(dw_h),
                "tech_id": tech_id,
                "plot_label": label,
                "osr_pct": osr,
                "series_type": first.get("series_type"),
                "lfp_discharge_share_pct": first.get("lfp_discharge_share_pct"),
                "caes_discharge_share_pct": first.get("caes_discharge_share_pct"),
                "actual_lfp_supply_share_pct": first.get("actual_lfp_supply_share_pct"),
                "actual_caes_supply_share_pct": first.get("actual_caes_supply_share_pct"),
            }
            if summary_extra:
                row_summary.update(summary_extra)
            summary_rows.append(row_summary)

        if hybrid_osr:
            finite = np.asarray([x for x in hybrid_osr if np.isfinite(x)], dtype=float)
            if finite.size:
                ax.text(
                    0.985,
                    0.965,
                    f"Hybrid OSR range: {finite.min():.1f}-{finite.max():.1f}%",
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
            fontsize=5.4 if is_linear_zoom else 5.8,
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
    variant: str,
    province_filter: str | None,
    formats: list[str],
    dpi: int,
    y_max: float,
) -> None:
    summary = load_mode_summary(mode)
    if province_filter:
        needle = province_filter.lower()
        summary = summary[
            summary["province_en"].astype(str).str.lower().str.contains(needle)
            | summary["province_cn"].astype(str).str.lower().str.contains(needle)
        ].copy()
    province_rows = (
        summary[["province_en", "province_cn"]]
        .drop_duplicates()
        .sort_values("province_en")
        .itertuples(index=False)
    )
    provinces = [(str(r.province_en), str(r.province_cn)) for r in province_rows]
    if not provinces:
        raise ValueError(f"No provinces matched filter: {province_filter}")

    out_root = output_dir_for_mode(mode, variant=variant)
    source_out = out_root / "source_data"
    source_out.mkdir(parents=True, exist_ok=True)
    fig_prefix = fig_prefix_for_mode(mode, variant=variant)
    all_summary_rows: list[dict] = []

    for idx, (province_en, province_cn) in enumerate(provinces, start=1):
        print(f"=== [{mode}/{variant}] render province {idx}/{len(provinces)}: {province_en} ({province_cn}) ===", flush=True)
        province_df = load_province_source(mode, province_en)
        province_dir = out_root / slugify(province_en)
        for dw_h in DW_VALUES:
            all_summary_rows.extend(
                render_dw_figure(
                    mode=mode,
                    variant=variant,
                    province_en=province_en,
                    province_cn=province_cn,
                    source_df=province_df,
                    dw_h=dw_h,
                    out_dir=province_dir,
                    fig_prefix=fig_prefix,
                    formats=formats,
                    dpi=dpi,
                    zoom_y_max=y_max,
                )
            )

    summary_name = f"{fig_prefix}_intersection_summary.csv"
    if province_filter:
        summary_name = f"{fig_prefix}_{slugify(province_filter)}_intersection_summary.csv"
    summary_fp = source_out / summary_name
    pd.DataFrame(all_summary_rows).to_csv(summary_fp, index=False)
    print(f"=== [{mode}/{variant}] gas-parity summary: {len(all_summary_rows):,} rows ===", flush=True)
    print(f"summary: {summary_fp.relative_to(PAPER_CODE_DIR)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-province DW-specific gas-parity diagnostic figures for fixed-discharge-ratio LFP+CAES curves."
    )
    parser.add_argument("--modes", nargs="+", default=ratio_plot.QUOTA_MODES, choices=ratio_plot.QUOTA_MODES)
    parser.add_argument("--variants", nargs="+", default=["full_tail", "linear"], choices=["full_tail", "linear", "linear_full_tail"])
    parser.add_argument("--province", default=None, help="Optional province filter, e.g. Anhui or 安徽")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--y-max", type=float, default=3.0, help="Linear gas-parity y-axis upper limit in marginal cost / gas cost units.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for mode in args.modes:
        for variant in args.variants:
            render_mode(
                mode=mode,
                variant=variant,
                province_filter=args.province,
                formats=args.formats,
                dpi=args.dpi,
                y_max=args.y_max,
            )


if __name__ == "__main__":
    main()
