"""
Seasonal-MC province-level marginal OSR curves: LFP, CAES, and fixed-ratio hybrids.

This figure set is a focused variant of
freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_smoothed:

  - keep only single-technology LFP and CAES curves;
  - add fixed LFP:CAES energy-capacity ratios from 90:10 to 10:90;
  - draw both LFP-first and CAES-first true-dispatch hybrid curves.

Output:
  output/m4_economics/nature_figure/
    freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_smoothed_lfp_caes_fixed_ratio/
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m4_visualize_freqlen_per_province_osr as base
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal
import m4_visualize_freqlen_seasonal_mc_per_province_marginal_osr_no_lfp4h as marginal
import m4_visualize_single_tech_generic as ms


seasonal_plotter = seasonal.plotter

PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
FIXED_RATIO_CSV = M4_DIR / "m4_hybrid_lfp_caes_fixed_ratio_frontier_no_lfp4h.csv"

OUT_DIR = (
    M4_DIR
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_smoothed_lfp_caes_fixed_ratio"
)
SOURCE_DIR = OUT_DIR / "source_data"
FIG_FILE_PREFIX = "fig5_freqlen_seasonal_mc_marginal_osr_lfp_caes_fixed_ratio_no_lfp4h"
SUMMARY_FILENAME = f"{FIG_FILE_PREFIX}_summary.csv"

SINGLE_TECH_ORDER = ["caes", "li_lfp"]
FIXED_LFP_SHARES_PCT = list(range(90, 0, -10))
POLICY_ORDER = ["lfp_first", "caes_first"]
POLICY_LABELS = {
    "lfp_first": "LFP-first",
    "caes_first": "CAES-first",
}
POLICY_LINESTYLES = {
    "lfp_first": "-",
    "caes_first": "--",
}
RATIO_COLORS = {
    share: plt.cm.viridis(i / (len(FIXED_LFP_SHARES_PCT) - 1))
    for i, share in enumerate(FIXED_LFP_SHARES_PCT)
}

FIXED_RATIO_COLS = [
    "target_supply_pct",
    "target_annual_supply_kwh",
    "satisfy_rate_pct",
    "annual_supply_kwh",
    "annual_supply_lfp_kwh",
    "annual_supply_ldes_kwh",
    "lfp_supply_share",
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
    "policy",
    "policy_label",
    "ldes_tech",
    "lfp_duration_h",
    "ldes_duration_h",
    "caes_duration_h",
    "lfp_capacity_mwh",
    "ldes_capacity_mwh",
    "caes_capacity_mwh",
    "lfp_capacity_share_pct",
    "caes_capacity_share_pct",
    "fixed_ratio_label",
    "lfp_single_osr_pct",
    "caes_single_osr_pct",
    "capacity_grid_n",
    "evaluated_points_n",
    "total_demand_annual_kwh",
    "frontier_mode",
]


def load_fixed_ratio_frontiers(fp: Path, province_filter: str | None) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(fp)

    if not province_filter:
        df = pd.read_csv(fp, usecols=FIXED_RATIO_COLS, low_memory=False)
    else:
        needle = province_filter.lower()
        chunks = []
        for chunk in pd.read_csv(fp, usecols=FIXED_RATIO_COLS, chunksize=200_000, low_memory=False):
            mask = (
                chunk["province_en"].astype(str).str.lower().str.contains(needle)
                | chunk["province_cn"].astype(str).str.lower().str.contains(needle)
            )
            if mask.any():
                chunks.append(chunk.loc[mask].copy())
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=FIXED_RATIO_COLS)

    df = df[df["scenario"].eq(ms.SCENARIO)].copy()
    df = df[df["dispatch_window_h"].isin(seasonal_plotter.DW_VALUES)].copy()
    df = df[df["policy"].isin(POLICY_ORDER)].copy()
    df = df[df["lfp_capacity_share_pct"].isin(FIXED_LFP_SHARES_PCT)].copy()
    return df


def fixed_ratio_supply_share(front: pd.DataFrame) -> tuple[float, float]:
    lfp = front["annual_supply_lfp_kwh"].astype(float)
    caes = front["annual_supply_ldes_kwh"].astype(float)
    total = float((lfp + caes).sum())
    if total <= 0:
        return float("nan"), float("nan")
    lfp_share = float(lfp.sum() / total * 100.0)
    return lfp_share, 100.0 - lfp_share


def append_single_records(
    records: dict,
    province_names: dict,
    summary_rows: list[dict],
    champions: pd.DataFrame,
    province_filter: str | None,
) -> None:
    champions = champions[champions["tech_id"].isin(SINGLE_TECH_ORDER)].copy()
    pairs = (
        champions[["stem", "tech_id"]]
        .drop_duplicates()
        .sort_values(["stem", "tech_id"])
        .itertuples(index=False, name=None)
    )
    for stem, tech_id in list(pairs):
        t0 = time.time()
        champ_pair = champions[
            champions["stem"].eq(stem) & champions["tech_id"].eq(tech_id)
        ].copy()
        if champ_pair.empty:
            continue
        print(f"=== load single full scan: {stem} | {tech_id} ===", flush=True)
        df = marginal.load_single_full_scan(stem, tech_id, province_filter)
        if df.empty:
            print("  no rows after filtering", flush=True)
            continue

        champ_keys = defaultdict(list)
        for row in champ_pair.itertuples(index=False):
            key = (
                str(row.province_en),
                float(row.dispatch_window_h),
                int(row.duration_h),
            )
            champ_keys[key].append(row)

        matched = 0
        for (prov_en, dw_h, dur), sub in df.groupby(
            ["province_en", "dispatch_window_h", "duration_h"], sort=False
        ):
            rows = champ_keys.get((str(prov_en), float(dw_h), int(dur)))
            if not rows:
                continue
            cost_front = marginal.system_cost_frontier_from_scan(sub)
            curve = marginal.marginal_from_system_cost_frontier(cost_front)
            if cost_front.empty or curve.empty:
                continue
            prov_cn = str(sub["province_cn"].iloc[0])
            province_names[prov_cn] = str(prov_en)
            matched += len(rows)
            for row in rows:
                rec_key = (int(row.row_idx), float(row.dispatch_window_h), tech_id)
                rec = {
                    "province_cn": prov_cn,
                    "province_en": str(prov_en),
                    "experiment": str(row.experiment),
                    "stem": str(row.stem),
                    "row_idx": int(row.row_idx),
                    "row_label": str(row.row_label),
                    "dispatch_window_h": float(row.dispatch_window_h),
                    "tech_id": tech_id,
                    "tech_label": str(row.tech_label),
                    "duration_h": int(row.duration_h),
                    "osr_pct": float(row.osr_pct),
                    "curve": curve,
                    "capacity_points_n": int(len(sub)),
                    "system_cost_frontier_points_n": int(len(cost_front)),
                    "raw_marginal_points_n": int(len(curve)),
                    "smoothed_marginal_points_n": int(len(curve)),
                    "series_type": "single",
                }
                records[prov_cn][rec_key] = rec
                summary_rows.append({k: v for k, v in rec.items() if k != "curve"})

        print(f"  matched single curves: {matched:,}; elapsed {time.time() - t0:.1f}s", flush=True)
        del df


def append_fixed_ratio_records(
    records: dict,
    province_names: dict,
    summary_rows: list[dict],
    province_filter: str | None,
    fixed_ratio_csv: Path,
) -> None:
    t0 = time.time()
    print(f"=== load fixed-ratio hybrid frontier: {fixed_ratio_csv.name} ===", flush=True)
    df = load_fixed_ratio_frontiers(fixed_ratio_csv, province_filter)
    if df.empty:
        print("  no fixed-ratio rows after filtering", flush=True)
        return

    matched = 0
    group_cols = [
        "province_en",
        "province_cn",
        "stem",
        "row_idx",
        "row_label",
        "dispatch_window_h",
        "policy",
        "lfp_capacity_share_pct",
    ]
    for key, front in df.groupby(group_cols, sort=False):
        prov_en, prov_cn, stem, row_idx, row_label, dw_h, policy, lfp_share = key
        front = front.dropna(
            subset=[
                "target_supply_pct",
                "annual_supply_kwh",
                "annual_cost_cny",
                "gas_unit_cost_cny_kwh",
                "total_demand_annual_kwh",
            ]
        ).copy()
        if front.empty:
            continue
        curve = marginal.marginal_from_system_cost_frontier(front)
        if curve.empty:
            continue

        lfp_share = int(round(float(lfp_share)))
        caes_share = 100 - lfp_share
        lfp_supply_share, caes_supply_share = fixed_ratio_supply_share(front)
        lfp_duration_h = int(front["lfp_duration_h"].iloc[0])
        caes_duration_h = int(front["caes_duration_h"].iloc[0])
        osr_pct = marginal.osr_from_marginal_curve(curve)
        tech_id = f"hybrid_{policy}_{lfp_share:02d}"
        rec_key = (int(row_idx), float(dw_h), tech_id)
        rec = {
            "province_cn": str(prov_cn),
            "province_en": str(prov_en),
            "experiment": str(front["experiment"].iloc[0]),
            "stem": str(stem),
            "row_idx": int(row_idx),
            "row_label": str(row_label),
            "dispatch_window_h": float(dw_h),
            "tech_id": tech_id,
            "tech_label": POLICY_LABELS[str(policy)],
            "policy": str(policy),
            "duration_h": f"{lfp_duration_h}+{caes_duration_h}",
            "lfp_duration_h": lfp_duration_h,
            "caes_duration_h": caes_duration_h,
            "lfp_capacity_share_pct": lfp_share,
            "caes_capacity_share_pct": caes_share,
            "fixed_ratio_label": f"LFP{lfp_share}-CAES{caes_share}",
            "lfp_supply_share_pct": lfp_supply_share,
            "caes_supply_share_pct": caes_supply_share,
            "osr_pct": osr_pct,
            "lfp_single_osr_pct": float(front["lfp_single_osr_pct"].iloc[0]),
            "caes_single_osr_pct": float(front["caes_single_osr_pct"].iloc[0]),
            "curve": curve,
            "capacity_points_n": int(front["evaluated_points_n"].max()),
            "system_cost_frontier_points_n": int(len(front)),
            "raw_marginal_points_n": int(len(curve)),
            "smoothed_marginal_points_n": int(len(curve)),
            "frontier_mode": str(front["frontier_mode"].iloc[0]),
            "series_type": "fixed_ratio_hybrid",
        }
        records[str(prov_cn)][rec_key] = rec
        province_names[str(prov_cn)] = str(prov_en)
        summary_rows.append({k: v for k, v in rec.items() if k != "curve"})
        matched += 1

    print(
        f"  matched fixed-ratio marginal curves: {matched:,}; elapsed {time.time() - t0:.1f}s",
        flush=True,
    )
    del df


def build_records(
    champions: pd.DataFrame,
    province_filter: str | None,
    fixed_ratio_csv: Path,
) -> tuple[dict, dict, pd.DataFrame]:
    records = defaultdict(dict)
    province_names = {}
    summary_rows = []
    append_single_records(records, province_names, summary_rows, champions, province_filter)
    append_fixed_ratio_records(records, province_names, summary_rows, province_filter, fixed_ratio_csv)
    return records, province_names, pd.DataFrame(summary_rows)


def setup_axis(ax, panel: str, experiment: str, dw_h: float, show_xlabel: bool, show_ylabel: bool) -> None:
    marginal.setup_marginal_axis(ax, panel, experiment, dw_h, show_xlabel, show_ylabel)


def render_province(
    prov_cn: str,
    prov_en: str,
    prov_records: dict,
    formats: list[str],
    dpi: int,
    save_source: bool,
) -> None:
    fig = plt.figure(figsize=(18.0, 28.0))
    gs = fig.add_gridspec(
        8, 4, left=0.065, right=0.985, top=0.960, bottom=0.040,
        wspace=0.22, hspace=0.42,
    )

    source_rows = []
    panel_idx = 0
    for experiment, stem, row_idx, row_label in seasonal_plotter.SCAN_ROWS:
        for col_idx, dw_h in enumerate(seasonal_plotter.DW_VALUES):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            setup_axis(
                ax,
                base.panel_label(panel_idx),
                experiment,
                dw_h,
                show_xlabel=(row_idx == 7),
                show_ylabel=(col_idx == 0),
            )

            for tech_id in SINGLE_TECH_ORDER:
                rec = prov_records.get((int(row_idx), float(dw_h), tech_id))
                if rec is None:
                    continue
                curve = rec["curve"]
                is_lfp = tech_id == "li_lfp"
                label = f"{rec['tech_label']} {rec['duration_h']}h"
                ax.plot(
                    curve["supply_pct"],
                    curve["marginal_cost_ratio_to_gas"],
                    color=ms.TECH_COLORS[tech_id],
                    lw=2.1 if is_lfp else 1.8,
                    ls="--" if is_lfp else "-",
                    marker="s" if is_lfp else "o",
                    ms=2.6,
                    markevery=45,
                    zorder=6,
                    label=label,
                )
                if save_source:
                    source_rows.append(
                        curve.assign(
                            province_cn=prov_cn,
                            province_en=prov_en,
                            experiment=experiment,
                            stem=stem,
                            row_idx=int(row_idx),
                            row_label=row_label,
                            dispatch_window_h=float(dw_h),
                            tech_id=tech_id,
                            tech_label=rec["tech_label"],
                            duration_h=rec["duration_h"],
                            osr_pct=rec["osr_pct"],
                            series_type=rec["series_type"],
                        )
                    )

            for policy in POLICY_ORDER:
                for lfp_share in FIXED_LFP_SHARES_PCT:
                    tech_id = f"hybrid_{policy}_{lfp_share:02d}"
                    rec = prov_records.get((int(row_idx), float(dw_h), tech_id))
                    if rec is None:
                        continue
                    curve = rec["curve"]
                    caes_share = 100 - lfp_share
                    label = f"{POLICY_LABELS[policy]} L{lfp_share}/C{caes_share}"
                    ax.plot(
                        curve["supply_pct"],
                        curve["marginal_cost_ratio_to_gas"],
                        color=RATIO_COLORS[lfp_share],
                        lw=1.05,
                        ls=POLICY_LINESTYLES[policy],
                        alpha=0.92,
                        zorder=4 if policy == "lfp_first" else 3,
                        label=label,
                    )
                    if save_source:
                        source_rows.append(
                            curve.assign(
                                province_cn=prov_cn,
                                province_en=prov_en,
                                experiment=experiment,
                                stem=stem,
                                row_idx=int(row_idx),
                                row_label=row_label,
                                dispatch_window_h=float(dw_h),
                                tech_id=tech_id,
                                tech_label=rec["tech_label"],
                                policy=rec["policy"],
                                duration_h=rec["duration_h"],
                                lfp_duration_h=rec["lfp_duration_h"],
                                caes_duration_h=rec["caes_duration_h"],
                                lfp_capacity_share_pct=rec["lfp_capacity_share_pct"],
                                caes_capacity_share_pct=rec["caes_capacity_share_pct"],
                                fixed_ratio_label=rec["fixed_ratio_label"],
                                lfp_supply_share_pct=rec["lfp_supply_share_pct"],
                                caes_supply_share_pct=rec["caes_supply_share_pct"],
                                osr_pct=rec["osr_pct"],
                                series_type=rec["series_type"],
                                frontier_mode=rec["frontier_mode"],
                            )
                        )

            ax.legend(
                loc="upper left",
                fontsize=2.9,
                handlelength=1.1,
                borderaxespad=0.15,
                ncol=2,
                columnspacing=0.35,
                handletextpad=0.25,
                labelspacing=0.10,
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
        f"Fig 5 seasonal-MC fixed-ratio LFP+CAES marginal OSR curves — {prov_en} ({prov_cn})",
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
        "Only single-tech LFP and CAES are retained. Hybrid curves are true-dispatch "
        "fixed energy-capacity ratios LFP:CAES = 90:10 ... 10:90 under LFP-first "
        "and CAES-first policies. Y-axis is marginal cost / gas cost from each "
        "ratio-specific lower convex envelope; red dashed line = OSR gas parity. "
        "LFP 4h is excluded.",
        ha="left",
        va="top",
        fontsize=7.4,
        color=ms.PALETTE["black"],
        **ms.cjk_font_kwargs(),
    )

    stem_out = f"{FIG_FILE_PREFIX}_{base.slugify(prov_en)}"
    for fmt in formats:
        path = OUT_DIR / f"{stem_out}.{fmt}"
        kw = {"dpi": dpi} if fmt == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved: {path.name}", flush=True)
    plt.close(fig)

    if save_source and source_rows:
        out = pd.concat(source_rows, ignore_index=True)
        out.to_csv(SOURCE_DIR / f"{stem_out}_source.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default=None)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "svg", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-source", action="store_true")
    parser.add_argument("--fixed-ratio-csv", default=str(FIXED_RATIO_CSV))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    champions = marginal.load_champions(args.province)
    if champions.empty:
        raise SystemExit("No champion rows matched.")

    records, province_names, summary = build_records(
        champions,
        args.province,
        Path(args.fixed_ratio_csv),
    )
    if summary.empty:
        raise SystemExit("No fixed-ratio marginal curves were produced.")

    summary = summary.sort_values(
        ["province_en", "row_idx", "dispatch_window_h", "series_type", "tech_id"]
    )
    summary.to_csv(SOURCE_DIR / SUMMARY_FILENAME, index=False)

    provinces = sorted(province_names.items(), key=lambda kv: kv[1])
    for idx, (prov_cn, prov_en) in enumerate(provinces, start=1):
        print(f"=== render {idx}/{len(provinces)}: {prov_en} ({prov_cn}) ===", flush=True)
        render_province(
            prov_cn,
            prov_en,
            records.get(prov_cn, {}),
            formats=args.formats,
            dpi=args.dpi,
            save_source=not args.no_source,
        )

    print(f"=== fixed-ratio marginal OSR summary: {len(summary):,} rows ===", flush=True)


if __name__ == "__main__":
    main()
