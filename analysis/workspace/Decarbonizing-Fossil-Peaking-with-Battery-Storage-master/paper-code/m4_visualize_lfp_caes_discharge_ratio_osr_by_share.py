"""
OSR-by-discharge-share diagnostic figures for LFP+CAES fixed-ratio dispatch.

The marginal-cost curves can look visually similar because all scenarios share
the same demand and curtailment shape. This script plots the gas-parity
intersection directly: target LFP discharge share on the x-axis and OSR at gas
parity on the y-axis. Each province receives a folder with four DW figures.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import m4_visualize_freqlen_per_province_osr as base
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal
import m4_visualize_freqlen_seasonal_mc_marginal_osr_lfp_caes_discharge_ratio as ratio_plot


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
DW_VALUES = [8.0, 12.0, 16.0, 24.0]


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "province"


def source_dir_for_mode(mode: str) -> Path:
    return ratio_plot.output_dir_for_mode(mode) / "source_data"


def source_prefix_for_mode(mode: str) -> str:
    return ratio_plot.fig_prefix_for_mode(mode)


def output_dir_for_mode(mode: str) -> Path:
    return (
        M4_DIR
        / "nature_figure"
        / f"freqlen_per_province_osr_seasonal_mc_no_lfp4h_lfp_caes_discharge_ratio_{mode}_osr_by_share_by_dw"
    )


def fig_prefix_for_mode(mode: str) -> str:
    return f"fig5_freqlen_seasonal_mc_lfp_caes_discharge_ratio_{mode}_osr_by_share_no_lfp4h"


def load_summary(mode: str, province_filter: str | None = None) -> pd.DataFrame:
    fp = source_dir_for_mode(mode) / f"{source_prefix_for_mode(mode)}_summary.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp, low_memory=False)
    if province_filter:
        needle = province_filter.lower()
        df = df[
            df["province_en"].astype(str).str.lower().str.contains(needle)
            | df["province_cn"].astype(str).str.lower().str.contains(needle)
        ].copy()
    return df


def first_numeric(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def duration_h_value(value: float | None) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(round(float(value)))


def duration_label(value: float | None) -> str:
    dur = duration_h_value(value)
    return f"{dur}h" if dur is not None else "?h"


def osr_range_label(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return "OSR NA"
    return f"hybrid OSR {float(clean.min()):.1f}-{float(clean.max()):.1f}%"


def province_list(summary: pd.DataFrame) -> list[tuple[str, str]]:
    rows = (
        summary[["province_en", "province_cn"]]
        .drop_duplicates()
        .sort_values("province_en")
        .itertuples(index=False)
    )
    return [(str(r.province_en), str(r.province_cn)) for r in rows]


def render_dw(
    mode: str,
    province_en: str,
    province_cn: str,
    prov_summary: pd.DataFrame,
    dw_h: float,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[dict]:
    mode_label = ratio_plot.MODE_LABELS[mode]
    fig = plt.figure(figsize=(14.5, 17.0))
    gs = fig.add_gridspec(
        4, 2, left=0.075, right=0.985, top=0.925, bottom=0.055,
        wspace=0.18, hspace=0.34,
    )
    fig.suptitle(
        f"OSR at gas parity by target LFP discharge share, {mode_label} - {province_en}, DW={int(dw_h)}h",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.075,
        0.955,
        "Y-axis is the OSR where marginal cost equals gas cost. Endpoints are CAES-only at 0% and LFP-only at 100%.",
        ha="left",
        va="top",
        fontsize=8.4,
    )

    summary_rows: list[dict] = []
    panel_idx = 0
    for experiment, stem, row_idx, row_label in seasonal.plotter.SCAN_ROWS:
        ax = fig.add_subplot(gs[int(row_idx) // 2, int(row_idx) % 2])
        panel = prov_summary[
            prov_summary["row_idx"].eq(int(row_idx))
            & prov_summary["dispatch_window_h"].eq(float(dw_h))
        ].copy()

        hybrid = panel[panel["series_type"].eq("discharge_ratio_hybrid")].copy()
        hybrid = hybrid.dropna(subset=["lfp_discharge_share_pct", "osr_pct"])
        hybrid = hybrid.sort_values("lfp_discharge_share_pct")

        singles = panel[panel["series_type"].eq("single")].copy()
        caes = singles[singles["tech_id"].eq("caes")]
        lfp = singles[singles["tech_id"].eq("li_lfp")]
        lfp_duration_h = first_numeric(hybrid, "lfp_duration_h") or first_numeric(lfp, "duration_h")
        caes_duration_h = first_numeric(hybrid, "caes_duration_h") or first_numeric(caes, "duration_h")
        duration_legend = (
            f"LFP {duration_label(lfp_duration_h)} + CAES {duration_label(caes_duration_h)}"
        )

        ax.set_title(f"{base.panel_label(panel_idx)} {row_label}", loc="left", fontsize=9.4, fontweight="bold")
        ax.set_xlim(-2, 102)
        ax.set_xticks(list(range(0, 101, 10)))
        ax.set_xlabel("Target LFP discharge share (%)", fontsize=8.2)
        ax.set_ylabel("OSR at gas parity (%)", fontsize=8.2)
        ax.grid(True, which="major", lw=0.35, color="#d8d8d8", alpha=0.82)
        ax.tick_params(axis="both", labelsize=7.4)

        y_values = []
        hybrid_plot = hybrid[[
            "lfp_discharge_share_pct",
            "caes_discharge_share_pct",
            "osr_pct",
            "actual_lfp_supply_share_pct",
            "actual_caes_supply_share_pct",
            "lfp_duration_h",
            "caes_duration_h",
        ]].copy() if not hybrid.empty else pd.DataFrame(
            columns=[
                "lfp_discharge_share_pct",
                "caes_discharge_share_pct",
                "osr_pct",
                "actual_lfp_supply_share_pct",
                "actual_caes_supply_share_pct",
                "lfp_duration_h",
                "caes_duration_h",
            ]
        )

        if not caes.empty:
            caes_endpoint = pd.DataFrame(
                [
                    {
                        "lfp_discharge_share_pct": 0.0,
                        "caes_discharge_share_pct": 100.0,
                        "osr_pct": float(caes["osr_pct"].iloc[0]),
                        "actual_lfp_supply_share_pct": 0.0,
                        "actual_caes_supply_share_pct": 100.0,
                        "lfp_duration_h": duration_h_value(lfp_duration_h),
                        "caes_duration_h": duration_h_value(caes_duration_h),
                    }
                ]
            )
            hybrid_plot = pd.concat([caes_endpoint, hybrid_plot], ignore_index=True)
        if not lfp.empty:
            lfp_endpoint = pd.DataFrame(
                [
                    {
                        "lfp_discharge_share_pct": 100.0,
                        "caes_discharge_share_pct": 0.0,
                        "osr_pct": float(lfp["osr_pct"].iloc[0]),
                        "actual_lfp_supply_share_pct": 100.0,
                        "actual_caes_supply_share_pct": 0.0,
                        "lfp_duration_h": duration_h_value(lfp_duration_h),
                        "caes_duration_h": duration_h_value(caes_duration_h),
                    }
                ]
            )
            hybrid_plot = pd.concat([hybrid_plot, lfp_endpoint], ignore_index=True)

        if not hybrid_plot.empty:
            hybrid_plot = hybrid_plot.dropna(subset=["lfp_discharge_share_pct", "osr_pct"])
            hybrid_plot = hybrid_plot.sort_values("lfp_discharge_share_pct")

        if not hybrid_plot.empty:
            hybrid_osr_label = osr_range_label(hybrid["osr_pct"]) if not hybrid.empty else osr_range_label(hybrid_plot["osr_pct"])
            ax.plot(
                hybrid_plot["lfp_discharge_share_pct"],
                hybrid_plot["osr_pct"],
                color="#2f6f9f",
                lw=2.0,
                marker="o",
                ms=4.2,
                label=f"Hybrid + endpoints ({duration_legend}; {hybrid_osr_label})",
                zorder=5,
            )
            ax.scatter(
                [0.0, 100.0],
                [
                    float(hybrid_plot[hybrid_plot["lfp_discharge_share_pct"].eq(0.0)]["osr_pct"].iloc[0])
                    if hybrid_plot["lfp_discharge_share_pct"].eq(0.0).any() else float("nan"),
                    float(hybrid_plot[hybrid_plot["lfp_discharge_share_pct"].eq(100.0)]["osr_pct"].iloc[0])
                    if hybrid_plot["lfp_discharge_share_pct"].eq(100.0).any() else float("nan"),
                ],
                color="#2f6f9f",
                edgecolor="white",
                linewidth=0.75,
                s=36,
                zorder=7,
            )
            y_values.extend(hybrid_plot["osr_pct"].dropna().tolist())
            for row in hybrid_plot.itertuples(index=False):
                point_type = "single_endpoint" if float(row.lfp_discharge_share_pct) in (0.0, 100.0) else "hybrid_ratio"
                summary_rows.append(
                    {
                        "quota_mode": mode,
                        "province_cn": province_cn,
                        "province_en": province_en,
                        "experiment": experiment,
                        "stem": stem,
                        "row_idx": int(row_idx),
                        "row_label": row_label,
                        "dispatch_window_h": float(dw_h),
                        "lfp_discharge_share_pct": float(row.lfp_discharge_share_pct),
                        "caes_discharge_share_pct": float(row.caes_discharge_share_pct),
                        "osr_pct": float(row.osr_pct),
                        "actual_lfp_supply_share_pct": float(row.actual_lfp_supply_share_pct)
                        if pd.notna(row.actual_lfp_supply_share_pct) else None,
                        "actual_caes_supply_share_pct": float(row.actual_caes_supply_share_pct)
                        if pd.notna(row.actual_caes_supply_share_pct) else None,
                        "lfp_duration_h": duration_h_value(row.lfp_duration_h),
                        "caes_duration_h": duration_h_value(row.caes_duration_h),
                        "point_type": point_type,
                    }
                )

        if y_values:
            y_min = max(0.0, min(y_values) - 5.0)
            y_max = min(100.0, max(y_values) + 5.0)
            if y_max - y_min < 10:
                center = (y_min + y_max) / 2.0
                y_min = max(0.0, center - 5.0)
                y_max = min(100.0, center + 5.0)
            ax.set_ylim(y_min, y_max)

        if not hybrid_plot.empty:
            spread = float(hybrid_plot["osr_pct"].max() - hybrid_plot["osr_pct"].min())
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
        panel_idx += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{fig_prefix_for_mode(mode)}_{slugify(province_en)}_dw{int(dw_h)}h"
    for fmt in formats:
        fp = out_dir / f"{stem}.{fmt}"
        fig.savefig(fp, dpi=dpi if fmt == "png" else None)
        print(f"  saved: {fp.relative_to(PAPER_CODE_DIR)}", flush=True)
    plt.close(fig)
    return summary_rows


def render_mode(mode: str, province_filter: str | None, formats: list[str], dpi: int) -> None:
    summary = load_summary(mode, province_filter)
    provinces = province_list(summary)
    if not provinces:
        raise ValueError(f"No province matched filter: {province_filter}")

    out_root = output_dir_for_mode(mode)
    source_out = out_root / "source_data"
    source_out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for idx, (province_en, province_cn) in enumerate(provinces, start=1):
        print(f"=== [{mode}] render province {idx}/{len(provinces)}: {province_en} ({province_cn}) ===", flush=True)
        prov_summary = summary[summary["province_en"].astype(str).eq(province_en)].copy()
        province_dir = out_root / slugify(province_en)
        for dw_h in DW_VALUES:
            all_rows.extend(
                render_dw(
                    mode=mode,
                    province_en=province_en,
                    province_cn=province_cn,
                    prov_summary=prov_summary,
                    dw_h=dw_h,
                    out_dir=province_dir,
                    formats=formats,
                    dpi=dpi,
                )
            )

    summary_fp = source_out / f"{fig_prefix_for_mode(mode)}_summary.csv"
    pd.DataFrame(all_rows).to_csv(summary_fp, index=False)
    print(f"=== [{mode}] OSR-by-share summary: {len(all_rows):,} rows ===", flush=True)
    print(f"summary: {summary_fp.relative_to(PAPER_CODE_DIR)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OSR at gas parity against target LFP discharge share.")
    parser.add_argument("--modes", nargs="+", default=ratio_plot.QUOTA_MODES, choices=ratio_plot.QUOTA_MODES)
    parser.add_argument("--province", default=None)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for mode in args.modes:
        render_mode(mode, args.province, args.formats, args.dpi)


if __name__ == "__main__":
    main()
