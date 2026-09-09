"""
M4 Fig 5 F + G — province-level OSR curve sheets without LFP 4h.

This variant reuses the existing province-level OSR source curves for all
non-LFP technologies and recomputes only the LFP champion duration after
excluding 4h. LFP is therefore selected from 6/8/10h only.

Output:
  output/m4_economics/nature_figure/freqlen_per_province_osr_no_lfp4h/
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m4_visualize_single_tech_generic as ms
import m4_visualize_freqlen_per_province_osr as base


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
OLD_OUT_DIR = M4_DIR / "nature_figure" / "freqlen_per_province_osr"
OLD_SOURCE_DIR = OLD_OUT_DIR / "source_data"
OUT_DIR = M4_DIR / "nature_figure" / "freqlen_per_province_osr_no_lfp4h"
SOURCE_DIR = OUT_DIR / "source_data"

DW_VALUES = base.DW_VALUES
TECH_ORDER = base.TECH_ORDER
SCAN_ROWS = base.SCAN_ROWS
NEEDED_COLS = base.NEEDED_COLS
LFP_ALLOWED_DURATIONS = [6, 8, 10]

SCAN_META_BY_STEM = {
    stem: (experiment, row_idx, row_label)
    for experiment, stem, row_idx, row_label in SCAN_ROWS
}


def load_existing_non_lfp_records() -> tuple[dict, dict, list[dict]]:
    """Load old province source curves and keep only non-LFP technologies."""
    records = defaultdict(dict)
    province_names: dict[str, str] = {}
    champion_rows: list[dict] = []

    champ_fp = OLD_SOURCE_DIR / "fig5_freqlen_province_osr_champions.csv"
    if not champ_fp.exists():
        raise FileNotFoundError(f"Missing old champion source: {champ_fp}")
    old_champions = pd.read_csv(champ_fp)
    old_champions = old_champions[old_champions["tech_id"].ne("li_lfp")]
    champion_rows.extend(old_champions.to_dict("records"))

    source_files = sorted(OLD_SOURCE_DIR.glob("fig5_freqlen_osr_*_source.csv"))
    if not source_files:
        raise FileNotFoundError(f"No old province source CSVs under {OLD_SOURCE_DIR}")

    group_cols = [
        "province_cn", "province_en", "experiment", "stem", "row_label",
        "dispatch_window_h", "tech_id", "tech_label", "duration_h", "osr_pct",
    ]
    for fp in source_files:
        df = pd.read_csv(fp)
        df = df[df["tech_id"].ne("li_lfp")].copy()
        if df.empty:
            continue
        for keys, sub in df.groupby(group_cols, sort=False):
            rec = dict(zip(group_cols, keys))
            stem = str(rec["stem"])
            if stem not in SCAN_META_BY_STEM:
                continue
            experiment, row_idx, row_label = SCAN_META_BY_STEM[stem]
            prov_cn = str(rec["province_cn"])
            prov_en = str(rec["province_en"])
            province_names[prov_cn] = prov_en
            tech_id = str(rec["tech_id"])
            dw_h = float(rec["dispatch_window_h"])
            key = (experiment, int(row_idx), dw_h, tech_id)
            records[prov_cn][key] = {
                "province_cn": prov_cn,
                "province_en": prov_en,
                "experiment": experiment,
                "stem": stem,
                "row_idx": int(row_idx),
                "row_label": row_label,
                "dispatch_window_h": dw_h,
                "tech_id": tech_id,
                "tech_label": str(rec["tech_label"]),
                "duration_h": int(rec["duration_h"]),
                "osr_pct": float(rec["osr_pct"]),
                "curve": sub[["supply_pct", "cost_ratio_to_gas"]].copy(),
            }

    return records, province_names, champion_rows


def recompute_lfp_no4h(records: dict, province_names: dict,
                       champion_rows: list[dict]) -> None:
    """Replace LFP records with OSR champions selected from 6/8/10h only."""
    for experiment, stem, row_idx, row_label in SCAN_ROWS:
        fp = M4_DIR / f"m4_single_tech_{stem}_li_lfp.csv"
        if not fp.exists():
            print(f"WARNING missing {fp.name}")
            continue
        print(f"=== recompute LFP without 4h: {stem} ===")
        df = pd.read_csv(fp, usecols=NEEDED_COLS, low_memory=False)
        df = df[df["scenario"].eq(ms.SCENARIO)].copy()
        df = df[df["dispatch_window_h"].isin(DW_VALUES)]
        df = df[df["duration_h"].isin(LFP_ALLOWED_DURATIONS)]
        print(f"  rows after LFP duration filter {LFP_ALLOWED_DURATIONS}: {len(df):,}")

        for (prov_cn, prov_en, dw_h), cell in df.groupby(
            ["province_cn", "province_en", "dispatch_window_h"],
            sort=False,
        ):
            province_names[str(prov_cn)] = str(prov_en)
            best_dur, osr_pct = base.province_champion_duration_osr(cell)
            if best_dur is None:
                continue
            sub = cell[cell["duration_h"].eq(float(best_dur))]
            curve = base.interpolate_curve(sub)
            if curve.empty:
                continue

            key = (experiment, int(row_idx), float(dw_h), "li_lfp")
            records[str(prov_cn)][key] = {
                "province_cn": str(prov_cn),
                "province_en": str(prov_en),
                "experiment": experiment,
                "stem": stem,
                "row_idx": int(row_idx),
                "row_label": row_label,
                "dispatch_window_h": float(dw_h),
                "tech_id": "li_lfp",
                "tech_label": ms.TECH_FILES["li_lfp"][0],
                "duration_h": int(best_dur),
                "osr_pct": float(osr_pct),
                "curve": curve,
            }
            champion_rows.append({
                "province_cn": str(prov_cn),
                "province_en": str(prov_en),
                "experiment": experiment,
                "stem": stem,
                "row_label": row_label,
                "dispatch_window_h": float(dw_h),
                "tech_id": "li_lfp",
                "tech_label": ms.TECH_FILES["li_lfp"][0],
                "duration_h": int(best_dur),
                "osr_pct": float(osr_pct),
            })

        del df


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
            base.setup_axis(
                ax, base.panel_label(panel_idx), experiment, dw_h,
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
        "LFP 4h is excluded; LFP duration is OSR-selected from 6/8/10h only. "
        "All other technologies reuse their province OSR-selected durations; "
        "red dashed line = gas parity.",
        ha="left", va="top", fontsize=7.4,
        color=ms.PALETTE["black"], **ms.cjk_font_kwargs())

    stem = f"fig5_freqlen_osr_no_lfp4h_{base.slugify(prov_en)}"
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

    records, province_names, champion_rows = load_existing_non_lfp_records()
    recompute_lfp_no4h(records, province_names, champion_rows)

    champions = pd.DataFrame(champion_rows)
    champions = champions.sort_values([
        "province_en", "experiment", "stem", "dispatch_window_h", "tech_id",
    ])
    champions.to_csv(
        SOURCE_DIR / "fig5_freqlen_province_osr_no_lfp4h_champions.csv",
        index=False,
    )
    print(f"=== champions: {len(champions):,} rows ===")
    lfp_counts = champions[champions["tech_id"].eq("li_lfp")][
        "duration_h"
    ].value_counts().sort_index()
    print("=== LFP selected durations after excluding 4h ===")
    print(lfp_counts.to_string())

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
