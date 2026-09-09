"""
Seasonal-MC province-level marginal OSR curves for fixed discharge-share LFP+CAES.

Two output folders are produced when both modes are requested:

  freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_smoothed_lfp_caes_discharge_ratio_hard_quota
  freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_smoothed_lfp_caes_discharge_ratio_soft_quota
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

SINGLE_TECH_ORDER = ["caes", "li_lfp"]
DISCHARGE_LFP_SHARES_PCT = list(range(90, 0, -10))
QUOTA_MODES = ["hard_quota", "soft_quota"]
MODE_LABELS = {
    "hard_quota": "Hard quota",
    "soft_quota": "Soft quota",
}
MODE_DESCRIPTIONS = {
    "hard_quota": (
        "Hard quota: LFP and CAES can only serve their assigned discharge shares; "
        "no cross-resource fallback is allowed."
    ),
    "soft_quota": (
        "Soft quota: LFP and CAES serve target discharge shares first; residual "
        "unmet demand can be supplied by either resource's remaining deliverable energy."
    ),
}
RATIO_COLORS = {
    share: plt.cm.viridis(i / (len(DISCHARGE_LFP_SHARES_PCT) - 1))
    for i, share in enumerate(DISCHARGE_LFP_SHARES_PCT)
}

DISCHARGE_RATIO_COLS = [
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
    "quota_mode",
    "quota_mode_label",
    "ldes_tech",
    "lfp_duration_h",
    "ldes_duration_h",
    "caes_duration_h",
    "lfp_capacity_mwh",
    "ldes_capacity_mwh",
    "caes_capacity_mwh",
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


def frontier_csv_for_mode(mode: str) -> Path:
    return M4_DIR / f"m4_hybrid_lfp_caes_discharge_ratio_{mode}_frontier_no_lfp4h.csv"


def output_dir_for_mode(mode: str) -> Path:
    return (
        M4_DIR
        / "nature_figure"
        / f"freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_smoothed_lfp_caes_discharge_ratio_{mode}"
    )


def fig_prefix_for_mode(mode: str) -> str:
    return f"fig5_freqlen_seasonal_mc_marginal_osr_lfp_caes_discharge_ratio_{mode}_no_lfp4h"


def load_discharge_ratio_frontiers(
    fp: Path,
    quota_mode: str,
    province_filter: str | None,
) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(fp)

    if not province_filter:
        df = pd.read_csv(fp, usecols=DISCHARGE_RATIO_COLS, low_memory=False)
    else:
        needle = province_filter.lower()
        chunks = []
        for chunk in pd.read_csv(fp, usecols=DISCHARGE_RATIO_COLS, chunksize=200_000, low_memory=False):
            mask = (
                chunk["province_en"].astype(str).str.lower().str.contains(needle)
                | chunk["province_cn"].astype(str).str.lower().str.contains(needle)
            )
            if mask.any():
                chunks.append(chunk.loc[mask].copy())
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=DISCHARGE_RATIO_COLS)

    df = df[df["scenario"].eq(ms.SCENARIO)].copy()
    df = df[df["dispatch_window_h"].isin(seasonal_plotter.DW_VALUES)].copy()
    df = df[df["quota_mode"].eq(quota_mode)].copy()
    df = df[df["lfp_discharge_share_pct"].isin(DISCHARGE_LFP_SHARES_PCT)].copy()
    return df


def actual_supply_share(front: pd.DataFrame) -> tuple[float, float]:
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
            key = (str(row.province_en), float(row.dispatch_window_h), int(row.duration_h))
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
                    "plot_label": f"{row.tech_label} {row.duration_h}h",
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


def append_discharge_ratio_records(
    records: dict,
    province_names: dict,
    summary_rows: list[dict],
    province_filter: str | None,
    quota_mode: str,
    frontier_csv: Path,
) -> None:
    t0 = time.time()
    print(f"=== load discharge-ratio frontier: {frontier_csv.name} ===", flush=True)
    df = load_discharge_ratio_frontiers(frontier_csv, quota_mode, province_filter)
    if df.empty:
        print("  no discharge-ratio rows after filtering", flush=True)
        return

    matched = 0
    group_cols = [
        "province_en",
        "province_cn",
        "stem",
        "row_idx",
        "row_label",
        "dispatch_window_h",
        "quota_mode",
        "lfp_discharge_share_pct",
    ]
    for key, front in df.groupby(group_cols, sort=False):
        prov_en, prov_cn, stem, row_idx, row_label, dw_h, mode, lfp_share = key
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
        ratio_label = f"LFP{lfp_share}-CAES{caes_share}"
        actual_lfp_share, actual_caes_share = actual_supply_share(front)
        lfp_duration_h = int(front["lfp_duration_h"].iloc[0])
        caes_duration_h = int(front["caes_duration_h"].iloc[0])
        osr_pct = marginal.osr_from_marginal_curve(curve)
        tech_id = f"hybrid_{quota_mode}_{lfp_share:02d}"
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
            "tech_label": MODE_LABELS[str(mode)],
            "plot_label": ratio_label,
            "quota_mode": str(mode),
            "duration_h": f"{lfp_duration_h}+{caes_duration_h}",
            "lfp_duration_h": lfp_duration_h,
            "caes_duration_h": caes_duration_h,
            "lfp_discharge_share_pct": lfp_share,
            "caes_discharge_share_pct": caes_share,
            "target_discharge_ratio_label": ratio_label,
            "actual_lfp_supply_share_pct": actual_lfp_share,
            "actual_caes_supply_share_pct": actual_caes_share,
            "osr_pct": osr_pct,
            "lfp_single_osr_pct": float(front["lfp_single_osr_pct"].iloc[0]),
            "caes_single_osr_pct": float(front["caes_single_osr_pct"].iloc[0]),
            "curve": curve,
            "capacity_points_n": int(front["evaluated_pairs_n"].max()),
            "system_cost_frontier_points_n": int(len(front)),
            "raw_marginal_points_n": int(len(curve)),
            "smoothed_marginal_points_n": int(len(curve)),
            "frontier_mode": str(front["frontier_mode"].iloc[0]),
            "series_type": "discharge_ratio_hybrid",
        }
        records[str(prov_cn)][rec_key] = rec
        province_names[str(prov_cn)] = str(prov_en)
        summary_rows.append({k: v for k, v in rec.items() if k != "curve"})
        matched += 1

    print(
        f"  matched discharge-ratio marginal curves: {matched:,}; elapsed {time.time() - t0:.1f}s",
        flush=True,
    )
    del df


def build_records(
    champions: pd.DataFrame,
    province_filter: str | None,
    quota_mode: str,
    frontier_csv: Path,
) -> tuple[dict, dict, pd.DataFrame]:
    records = defaultdict(dict)
    province_names = {}
    summary_rows = []
    append_single_records(records, province_names, summary_rows, champions, province_filter)
    append_discharge_ratio_records(
        records, province_names, summary_rows, province_filter, quota_mode, frontier_csv
    )
    return records, province_names, pd.DataFrame(summary_rows)


def setup_axis(ax, panel: str, experiment: str, dw_h: float, show_xlabel: bool, show_ylabel: bool) -> None:
    marginal.setup_marginal_axis(ax, panel, experiment, dw_h, show_xlabel, show_ylabel)


def render_province(
    prov_cn: str,
    prov_en: str,
    prov_records: dict,
    quota_mode: str,
    out_dir: Path,
    source_dir: Path,
    fig_prefix: str,
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
                label = rec["plot_label"]
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
                            plot_label=rec["plot_label"],
                            duration_h=rec["duration_h"],
                            osr_pct=rec["osr_pct"],
                            series_type=rec["series_type"],
                        )
                    )

            for lfp_share in DISCHARGE_LFP_SHARES_PCT:
                tech_id = f"hybrid_{quota_mode}_{lfp_share:02d}"
                rec = prov_records.get((int(row_idx), float(dw_h), tech_id))
                if rec is None:
                    continue
                curve = rec["curve"]
                label = rec["plot_label"]
                ax.plot(
                    curve["supply_pct"],
                    curve["marginal_cost_ratio_to_gas"],
                    color=RATIO_COLORS[lfp_share],
                    lw=1.25,
                    ls="-",
                    alpha=0.93,
                    zorder=4,
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
                            plot_label=rec["plot_label"],
                            quota_mode=rec["quota_mode"],
                            duration_h=rec["duration_h"],
                            lfp_duration_h=rec["lfp_duration_h"],
                            caes_duration_h=rec["caes_duration_h"],
                            lfp_discharge_share_pct=rec["lfp_discharge_share_pct"],
                            caes_discharge_share_pct=rec["caes_discharge_share_pct"],
                            target_discharge_ratio_label=rec["target_discharge_ratio_label"],
                            actual_lfp_supply_share_pct=rec["actual_lfp_supply_share_pct"],
                            actual_caes_supply_share_pct=rec["actual_caes_supply_share_pct"],
                            osr_pct=rec["osr_pct"],
                            series_type=rec["series_type"],
                            frontier_mode=rec["frontier_mode"],
                        )
                    )

            ax.legend(
                loc="upper left",
                fontsize=3.4,
                handlelength=1.15,
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
        f"Fig 5 seasonal-MC fixed-discharge-ratio LFP+CAES marginal OSR curves "
        f"({MODE_LABELS[quota_mode]}) — {prov_en} ({prov_cn})",
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
        "target discharge shares LFP:CAES = 90:10 ... 10:90 with two-dimensional "
        "LFP x CAES capacity search. "
        f"{MODE_DESCRIPTIONS[quota_mode]} "
        "Y-axis is marginal cost / gas cost from each ratio-specific lower convex "
        "envelope; red dashed line = OSR gas parity. LFP 4h is excluded.",
        ha="left",
        va="top",
        fontsize=7.3,
        color=ms.PALETTE["black"],
        **ms.cjk_font_kwargs(),
    )

    stem_out = f"{fig_prefix}_{base.slugify(prov_en)}"
    for fmt in formats:
        path = out_dir / f"{stem_out}.{fmt}"
        kw = {"dpi": dpi} if fmt == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved: {path.name}", flush=True)
    plt.close(fig)

    if save_source and source_rows:
        out = pd.concat(source_rows, ignore_index=True)
        out.to_csv(source_dir / f"{stem_out}_source.csv", index=False)


def render_mode(args, quota_mode: str) -> None:
    out_dir = output_dir_for_mode(quota_mode)
    source_dir = out_dir / "source_data"
    fig_prefix = fig_prefix_for_mode(quota_mode)
    summary_filename = f"{fig_prefix}_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    frontier_csv = Path(args.frontier_csv) if args.frontier_csv else frontier_csv_for_mode(quota_mode)
    champions = marginal.load_champions(args.province)
    if champions.empty:
        raise SystemExit("No champion rows matched.")

    records, province_names, summary = build_records(
        champions,
        args.province,
        quota_mode,
        frontier_csv,
    )
    if summary.empty:
        raise SystemExit(f"No discharge-ratio marginal curves were produced for {quota_mode}.")

    summary = summary.sort_values(
        ["province_en", "row_idx", "dispatch_window_h", "series_type", "tech_id"]
    )
    summary.to_csv(source_dir / summary_filename, index=False)

    provinces = sorted(province_names.items(), key=lambda kv: kv[1])
    for idx, (prov_cn, prov_en) in enumerate(provinces, start=1):
        print(
            f"=== [{quota_mode}] render {idx}/{len(provinces)}: {prov_en} ({prov_cn}) ===",
            flush=True,
        )
        render_province(
            prov_cn,
            prov_en,
            records.get(prov_cn, {}),
            quota_mode,
            out_dir,
            source_dir,
            fig_prefix,
            formats=args.formats,
            dpi=args.dpi,
            save_source=not args.no_source,
        )

    print(f"=== [{quota_mode}] discharge-ratio marginal OSR summary: {len(summary):,} rows ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default=None)
    parser.add_argument("--modes", nargs="+", default=QUOTA_MODES, choices=QUOTA_MODES)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "svg", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-source", action="store_true")
    parser.add_argument(
        "--frontier-csv",
        default=None,
        help="Optional frontier CSV; only valid when a single mode is requested.",
    )
    args = parser.parse_args()

    if args.frontier_csv and len(args.modes) > 1:
        raise SystemExit("--frontier-csv can only be used with a single --modes value")

    for quota_mode in args.modes:
        render_mode(args, quota_mode)


if __name__ == "__main__":
    main()
