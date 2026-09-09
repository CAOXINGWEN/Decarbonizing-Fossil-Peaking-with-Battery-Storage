"""
M4 Fig 5 F + G — province-level OSR curve sheets.

The national Fig 5 F/G panels use 26-province median curves. This script keeps
the same F/G natural-distribution demand nodes but renders one province sheet
per province. Each panel selects the duration by that province's own OSR.

Output:
  output/m4_economics/nature_figure/freqlen_per_province_osr/
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m4_visualize_single_tech_generic as ms


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
OUT_DIR = M4_DIR / "nature_figure" / "freqlen_per_province_osr"
SOURCE_DIR = OUT_DIR / "source_data"

DW_VALUES = ms.DW_VALUES
TECH_ORDER = [
    "caes", "laes", "gravity", "fecr_flow",
    "molten_salt_tes", "vrfb", "h2_cavern", "li_lfp",
]
X_GRID = np.linspace(0.0, 100.0, 241)

SCAN_ROWS = [
    ("F", "F60_real", 0, "F freq=60%"),
    ("F", "F70_real", 1, "F freq=70%"),
    ("F", "F80_real", 2, "F freq=80%"),
    ("F", "F90_real", 3, "F freq=90%"),
    ("G", "G5_real", 4, "G length=5d"),
    ("G", "G7_real", 5, "G length=7d"),
    ("G", "G10_real", 6, "G length=10d"),
    ("G", "G14_real", 7, "G length=14d"),
]

NEEDED_COLS = [
    "scenario", "dispatch_window_h", "duration_h",
    "province_en", "province_cn",
    "satisfy_rate_pct", "annual_supply_kwh",
    "unit_cost_cny_kwh", "gas_unit_cost_cny_kwh",
    "cost_ratio_to_gas",
]


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "province"


def panel_label(idx: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    if idx < len(letters):
        return letters[idx]
    return letters[(idx // len(letters)) - 1] + letters[idx % len(letters)]


def province_champion_duration_osr(df_cell: pd.DataFrame) -> tuple[int | None, float]:
    """Select duration with the highest OSR for one province/tech/dw cell."""
    scores: dict[int, float] = {}
    for dur, sub in df_cell.groupby("duration_h", sort=True):
        osr = ms.compute_osr_for_province(sub)
        if osr is None or not np.isfinite(osr):
            continue
        scores[int(dur)] = float(osr)
    if not scores:
        return None, float("nan")
    best = max(scores, key=scores.get)
    return best, scores[best]


def interpolate_curve(sub: pd.DataFrame) -> pd.DataFrame:
    curve = sub[["satisfy_rate_pct", "cost_ratio_to_gas"]].dropna()
    if curve.empty:
        return pd.DataFrame(columns=["supply_pct", "cost_ratio_to_gas"])
    curve = curve.sort_values("satisfy_rate_pct").drop_duplicates(
        subset=["satisfy_rate_pct"], keep="first")
    if len(curve) < 2:
        return pd.DataFrame(columns=["supply_pct", "cost_ratio_to_gas"])

    x = curve["satisfy_rate_pct"].to_numpy(dtype=float)
    y = curve["cost_ratio_to_gas"].to_numpy(dtype=float)
    valid = (X_GRID >= x.min()) & (X_GRID <= x.max())
    y_grid = np.full_like(X_GRID, np.nan, dtype=float)
    y_grid[valid] = np.interp(X_GRID[valid], x, y)
    keep = np.isfinite(y_grid)
    return pd.DataFrame({
        "supply_pct": X_GRID[keep],
        "cost_ratio_to_gas": y_grid[keep],
    })


def build_province_records() -> tuple[dict, dict, pd.DataFrame]:
    records = defaultdict(dict)
    province_names: dict[str, str] = {}
    champion_rows = []

    for experiment, stem, row_idx, row_label in SCAN_ROWS:
        print(f"=== {stem} ({experiment}) ===")
        for tech_id in TECH_ORDER:
            fp = M4_DIR / f"m4_single_tech_{stem}_{tech_id}.csv"
            if not fp.exists():
                print(f"  WARNING missing {fp.name}")
                continue
            df = pd.read_csv(fp, usecols=NEEDED_COLS, low_memory=False)
            df = df[df["scenario"].eq(ms.SCENARIO)].copy()
            df = df[df["dispatch_window_h"].isin(DW_VALUES)]
            print(f"  {tech_id}: {len(df):,} rows")

            for (prov_cn, prov_en, dw_h), cell in df.groupby(
                ["province_cn", "province_en", "dispatch_window_h"],
                sort=False,
            ):
                province_names[prov_cn] = prov_en
                best_dur, osr_pct = province_champion_duration_osr(cell)
                if best_dur is None:
                    continue
                sub = cell[cell["duration_h"].eq(float(best_dur))]
                curve = interpolate_curve(sub)
                if curve.empty:
                    continue

                key = (experiment, row_idx, float(dw_h), tech_id)
                records[prov_cn][key] = {
                    "province_cn": prov_cn,
                    "province_en": prov_en,
                    "experiment": experiment,
                    "stem": stem,
                    "row_idx": row_idx,
                    "row_label": row_label,
                    "dispatch_window_h": float(dw_h),
                    "tech_id": tech_id,
                    "tech_label": ms.TECH_FILES[tech_id][0],
                    "duration_h": best_dur,
                    "osr_pct": osr_pct,
                    "curve": curve,
                }
                champion_rows.append({
                    "province_cn": prov_cn,
                    "province_en": prov_en,
                    "experiment": experiment,
                    "stem": stem,
                    "row_label": row_label,
                    "dispatch_window_h": float(dw_h),
                    "tech_id": tech_id,
                    "tech_label": ms.TECH_FILES[tech_id][0],
                    "duration_h": best_dur,
                    "osr_pct": osr_pct,
                })

            del df

    champions = pd.DataFrame(champion_rows)
    return records, province_names, champions


def setup_axis(ax, panel: str, experiment: str, dw_h: float,
               show_xlabel: bool, show_ylabel: bool):
    ax.axhline(1.0, color=ms.PALETTE["red"], lw=0.9, ls="--", alpha=0.5)
    ax.fill_between([0, 100], 0.05, 1.0, color=ms.PALETTE["fill_blue"],
                    alpha=0.12, zorder=0)
    ax.fill_between([0, 100], 1.0, 100, color=ms.PALETTE["fill_red"],
                    alpha=0.08, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.15, 30)
    ax.grid(axis="both", which="major", color=ms.PALETTE["grid"], lw=0.5)
    ax.tick_params(labelsize=6.4)
    ax.set_title(f"{panel}  {experiment} dw={int(dw_h)}h",
                 loc="left", fontsize=7.8, pad=2)
    ax.set_xlabel("Peaking supply ratio (%)" if show_xlabel else "",
                  labelpad=2)
    ax.set_ylabel("Unit cost / gas cost" if show_ylabel else "",
                  labelpad=2)


def render_province(prov_cn: str, prov_en: str, prov_records: dict,
                    formats: list[str], dpi: int, save_source: bool):
    fig = plt.figure(figsize=(18.0, 28.0))
    gs = fig.add_gridspec(
        8, 4, left=0.065, right=0.985, top=0.960, bottom=0.040,
        wspace=0.22, hspace=0.42)

    source_rows = []
    panel_idx = 0
    for experiment, stem, row_idx, row_label in SCAN_ROWS:
        for col_idx, dw_h in enumerate(DW_VALUES):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            setup_axis(
                ax, panel_label(panel_idx), experiment, dw_h,
                show_xlabel=(row_idx == 7),
                show_ylabel=(col_idx == 0),
            )

            for tech_id in TECH_ORDER:
                rec = prov_records.get((experiment, row_idx, float(dw_h), tech_id))
                if rec is None:
                    continue
                curve = rec["curve"]
                is_lfp = tech_id == "li_lfp"
                label = (
                    f"{rec['tech_label']} {int(rec['duration_h'])}h"
                    f" | OSR {rec['osr_pct']:.0f}%"
                )
                ax.plot(
                    curve["supply_pct"], curve["cost_ratio_to_gas"],
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
                    source_rows.append(curve.assign(
                        province_cn=prov_cn,
                        province_en=prov_en,
                        experiment=experiment,
                        stem=stem,
                        row_label=row_label,
                        dispatch_window_h=float(dw_h),
                        tech_id=tech_id,
                        tech_label=rec["tech_label"],
                        duration_h=int(rec["duration_h"]),
                        osr_pct=rec["osr_pct"],
                    ))

            ax.legend(
                loc="upper left", fontsize=4.7, handlelength=1.1,
                borderaxespad=0.2, ncol=1, columnspacing=0.4,
                handletextpad=0.35, labelspacing=0.18)

            if col_idx == 0:
                ax.text(
                    -0.30, 0.5, row_label,
                    transform=ax.transAxes,
                    ha="center", va="center", rotation=90,
                    fontsize=9.6, fontweight="bold",
                    color=ms.PALETTE["black"],
                    **ms.cjk_font_kwargs(bold=True),
                )
            panel_idx += 1

    fig.text(
        0.065, 0.990,
        f"Fig 5 province-level OSR curves — {prov_en} ({prov_cn})",
        ha="left", va="top", fontsize=13, fontweight="bold",
        color=ms.PALETTE["black"], **ms.cjk_font_kwargs(bold=True))
    fig.text(
        0.065, 0.975,
        "Rows 1-4: frequency scan; rows 5-8: run-length scan. "
        "Each panel uses this province's own OSR-selected duration; "
        "red dashed line = gas parity.",
        ha="left", va="top", fontsize=7.4,
        color=ms.PALETTE["black"], **ms.cjk_font_kwargs())

    stem = f"fig5_freqlen_osr_{slugify(prov_en)}"
    for fmt in formats:
        path = OUT_DIR / f"{stem}.{fmt}"
        kw = {"dpi": dpi} if fmt == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved: {path.name}")
    plt.close(fig)

    if save_source and source_rows:
        out = pd.concat(source_rows, ignore_index=True)
        out.to_csv(SOURCE_DIR / f"{stem}_source.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formats", nargs="+", default=["png"],
                        choices=["png", "svg", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--province", default=None,
                        help="Optional province_en or province_cn substring.")
    parser.add_argument("--no-source", action="store_true",
                        help="Do not write province-level curve source CSVs.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    records, province_names, champions = build_province_records()
    champions.to_csv(SOURCE_DIR / "fig5_freqlen_province_osr_champions.csv",
                     index=False)
    print(f"=== champions: {len(champions):,} rows ===")

    provinces = sorted(province_names.items(), key=lambda kv: kv[1])
    if args.province:
        needle = args.province.lower()
        provinces = [
            (cn, en) for cn, en in provinces
            if needle in cn.lower() or needle in en.lower()
        ]
    if not provinces:
        raise SystemExit("No province matched.")

    for idx, (prov_cn, prov_en) in enumerate(provinces, start=1):
        print(f"=== render {idx}/{len(provinces)}: {prov_en} ({prov_cn}) ===")
        render_province(
            prov_cn, prov_en, records[prov_cn],
            formats=args.formats, dpi=args.dpi,
            save_source=not args.no_source,
        )


if __name__ == "__main__":
    main()
