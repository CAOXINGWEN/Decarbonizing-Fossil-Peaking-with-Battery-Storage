"""
M4 Fig 5 seasonal-MC province sheets with true LFP+CAES hybrid dispatch.

This visualization keeps the original eight single-technology OSR-selected
curves unchanged and overlays two curves derived from hourly-window hybrid
dispatch results:

  - LFP-first dispatch: LFP charges/discharges first, CAES serves residual.
  - CAES-first dispatch: CAES charges/discharges first, LFP serves residual.

Hybrid curves are built from actual capacity-pair dispatch rows produced by
m4_hybrid_lfp_caes_seasonal_mc_sweep.py. For plotting, raw capacity-pair points
are grouped by peaking-supply ratio bins and the minimum-cost point in each bin
is retained, then interpolated on the same x-axis as the single-tech curves.

Output:
  output/m4_economics/nature_figure/
    freqlen_per_province_osr_seasonal_mc_no_lfp4h_hybrid_dispatch/
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
HYBRID_CSV = M4_DIR / "m4_hybrid_lfp_caes_seasonal_mc_no_lfp4h.csv"

OUT_DIR = (
    M4_DIR
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h_hybrid_dispatch"
)
SOURCE_DIR = OUT_DIR / "source_data"
FIG_FILE_PREFIX = "fig5_freqlen_seasonal_mc_osr_no_lfp4h_hybrid_dispatch"
SUMMARY_FILENAME = f"{FIG_FILE_PREFIX}_summary.csv"

POLICY_STYLE = {
    "lfp_first": {
        "label": "LFP-first hybrid",
        "color": "#111111",
        "ls": "-",
        "marker": "D",
    },
    "caes_first": {
        "label": "CAES-first hybrid",
        "color": "#D55E00",
        "ls": "-.",
        "marker": "P",
    },
}

BIN_WIDTH_PCT = 0.5
STRICT_DIR_SUFFIX = "_strict_combo"


def load_province_source(prov_en: str) -> pd.DataFrame:
    slug = base.slugify(prov_en)
    fp = INPUT_SOURCE_DIR / f"{seasonal_plotter.FIG_FILE_PREFIX}_{slug}_source.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_csv(fp)


def load_hybrid_results(fp: Path, province_filter: str | None) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(fp)
    usecols = [
        "satisfy_rate_pct",
        "annual_supply_kwh",
        "annual_supply_lfp_kwh",
        "annual_supply_ldes_kwh",
        "lfp_supply_share",
        "annual_cost_cny",
        "unit_cost_cny_kwh",
        "gas_unit_cost_cny_kwh",
        "cost_ratio_to_gas",
        "stem",
        "row_idx",
        "row_label",
        "dispatch_window_h",
        "province_en",
        "province_cn",
        "policy",
        "policy_label",
        "lfp_duration_h",
        "caes_duration_h",
        "lfp_capacity_mwh",
        "caes_capacity_mwh",
        "lfp_single_osr_pct",
        "caes_single_osr_pct",
        "lfp_grid_n",
        "caes_grid_n",
    ]
    if not province_filter:
        return pd.read_csv(fp, usecols=usecols, low_memory=False)

    needle = province_filter.lower()
    chunks = []
    for chunk in pd.read_csv(fp, usecols=usecols, chunksize=200_000, low_memory=False):
        mask = (
            chunk["province_en"].astype(str).str.lower().str.contains(needle)
            | chunk["province_cn"].astype(str).str.lower().str.contains(needle)
        )
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def eligible_hybrid_rows(
    raw: pd.DataFrame,
    strict_hybrid: bool,
    min_share_frac: float,
) -> pd.DataFrame:
    clean = raw.dropna(
        subset=[
            "satisfy_rate_pct",
            "cost_ratio_to_gas",
            "annual_supply_kwh",
            "unit_cost_cny_kwh",
            "gas_unit_cost_cny_kwh",
        ]
    ).copy()
    clean = clean[
        np.isfinite(clean["satisfy_rate_pct"])
        & np.isfinite(clean["cost_ratio_to_gas"])
        & (clean["satisfy_rate_pct"] > 0)
        & (clean["cost_ratio_to_gas"] > 0)
    ]
    if clean.empty:
        return clean

    if strict_hybrid:
        clean = clean[
            (clean["lfp_capacity_mwh"] > 0)
            & (clean["caes_capacity_mwh"] > 0)
            & (clean["annual_supply_lfp_kwh"] > 0)
            & (clean["annual_supply_ldes_kwh"] > 0)
        ].copy()
        if clean.empty:
            return clean

    if min_share_frac > 0:
        total_supply = (
            clean["annual_supply_lfp_kwh"].astype(float)
            + clean["annual_supply_ldes_kwh"].astype(float)
        )
        lfp_share = np.divide(
            clean["annual_supply_lfp_kwh"].astype(float),
            total_supply,
            out=np.full(len(clean), np.nan, dtype=float),
            where=total_supply > 0,
        )
        caes_share = 1.0 - lfp_share
        clean = clean[
            np.isfinite(lfp_share)
            & (lfp_share >= min_share_frac)
            & (caes_share >= min_share_frac)
        ].copy()
        if clean.empty:
            return clean

    return clean


def hybrid_frontier(clean: pd.DataFrame) -> pd.DataFrame:
    if clean.empty:
        return clean.copy()
    clean["supply_bin"] = (
        np.round(clean["satisfy_rate_pct"] / BIN_WIDTH_PCT) * BIN_WIDTH_PCT
    )
    clean = clean[(clean["supply_bin"] >= 0) & (clean["supply_bin"] <= 100)]
    idx = clean.groupby("supply_bin")["cost_ratio_to_gas"].idxmin()
    front = clean.loc[idx].copy()
    front = front.sort_values(["supply_bin", "cost_ratio_to_gas"]).drop_duplicates(
        "supply_bin", keep="first"
    )
    front["supply_pct"] = front["supply_bin"].astype(float)
    return front.reset_index(drop=True)


def interpolate_hybrid_curve(front: pd.DataFrame) -> pd.DataFrame:
    if front.empty or front["supply_pct"].nunique() < 2:
        return pd.DataFrame(columns=["supply_pct", "cost_ratio_to_gas"])
    x = front["supply_pct"].to_numpy(dtype=float)
    y = front["cost_ratio_to_gas"].to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]
    valid = (base.X_GRID >= x_u.min()) & (base.X_GRID <= x_u.max())
    y_grid = np.full_like(base.X_GRID, np.nan, dtype=float)
    y_grid[valid] = np.interp(base.X_GRID[valid], x_u, y_u)
    keep = np.isfinite(y_grid) & (y_grid > 0)
    return pd.DataFrame(
        {"supply_pct": base.X_GRID[keep], "cost_ratio_to_gas": y_grid[keep]}
    )


def compute_hybrid_osr(front: pd.DataFrame) -> float:
    if front.empty or front["supply_pct"].nunique() < 3:
        return float("nan")
    df = front[
        ["annual_supply_kwh", "unit_cost_cny_kwh", "gas_unit_cost_cny_kwh"]
    ].copy()
    df["satisfy_rate_pct"] = front["supply_pct"].to_numpy(dtype=float)
    try:
        return float(ms.compute_osr_for_province(df))
    except Exception:
        return float("nan")


def frontier_share_metrics(front: pd.DataFrame) -> dict:
    """Return LFP/CAES supply shares on selected frontier points."""
    if front.empty:
        return {
            "lfp_supply_share_pct": float("nan"),
            "caes_supply_share_pct": float("nan"),
            "lfp_supply_share_mean_pct": float("nan"),
            "caes_supply_share_mean_pct": float("nan"),
        }

    lfp = front["annual_supply_lfp_kwh"].astype(float)
    caes = front["annual_supply_ldes_kwh"].astype(float)
    total = lfp + caes
    total_sum = float(total.sum())
    if total_sum > 0:
        lfp_weighted = float(lfp.sum() / total_sum * 100.0)
    else:
        lfp_weighted = float("nan")

    share = np.divide(
        lfp.to_numpy(dtype=float),
        total.to_numpy(dtype=float),
        out=np.full(len(front), np.nan, dtype=float),
        where=total.to_numpy(dtype=float) > 0,
    )
    lfp_mean = float(np.nanmean(share) * 100.0) if np.isfinite(share).any() else float("nan")
    return {
        "lfp_supply_share_pct": lfp_weighted,
        "caes_supply_share_pct": 100.0 - lfp_weighted if np.isfinite(lfp_weighted) else float("nan"),
        "lfp_supply_share_mean_pct": lfp_mean,
        "caes_supply_share_mean_pct": 100.0 - lfp_mean if np.isfinite(lfp_mean) else float("nan"),
    }


def render_province(
    prov_cn: str,
    prov_en: str,
    province_source: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    formats: list[str],
    dpi: int,
    save_source: bool,
    strict_hybrid: bool,
    min_share_frac: float,
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
                & province_source["row_label"].eq(row_label)
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
                            row_idx=row_idx,
                            policy="",
                            policy_label="",
                            hybrid_osr_pct=np.nan,
                            raw_points_n=np.nan,
                            frontier_points_n=np.nan,
                        )
                    )

            panel_hybrid = hybrid_df[
                hybrid_df["province_en"].eq(prov_en)
                & hybrid_df["stem"].eq(stem)
                & hybrid_df["row_idx"].eq(int(row_idx))
                & hybrid_df["row_label"].eq(row_label)
                & hybrid_df["dispatch_window_h"].eq(float(dw_h))
            ].copy()

            for policy, style in POLICY_STYLE.items():
                raw = panel_hybrid[panel_hybrid["policy"].eq(policy)].copy()
                if raw.empty:
                    continue
                eligible = eligible_hybrid_rows(raw, strict_hybrid, min_share_frac)
                front = hybrid_frontier(eligible)
                curve = interpolate_hybrid_curve(front)
                if curve.empty:
                    continue
                osr = compute_hybrid_osr(front)
                shares = frontier_share_metrics(front)
                lfp_dur = int(raw["lfp_duration_h"].dropna().iloc[0])
                caes_dur = int(raw["caes_duration_h"].dropna().iloc[0])
                label = (
                    f"{style['label']} "
                    f"(LFP{shares['lfp_supply_share_pct']:.0f}%/"
                    f"CAES{shares['caes_supply_share_pct']:.0f}%)"
                    f" | OSR {osr:.0f}%"
                )
                ax.plot(
                    curve["supply_pct"],
                    curve["cost_ratio_to_gas"],
                    color=style["color"],
                    lw=2.2,
                    ls=style["ls"],
                    marker=style["marker"],
                    ms=3.0,
                    markevery=35,
                    zorder=5,
                    label=label,
                )

                summary = {
                    "province_cn": prov_cn,
                    "province_en": prov_en,
                    "experiment": experiment,
                    "stem": stem,
                    "row_idx": row_idx,
                    "row_label": row_label,
                    "dispatch_window_h": float(dw_h),
                    "policy": policy,
                    "policy_label": style["label"],
                    "lfp_duration_h": lfp_dur,
                    "caes_duration_h": caes_dur,
                    "lfp_single_osr_pct": float(raw["lfp_single_osr_pct"].dropna().iloc[0]),
                    "caes_single_osr_pct": float(raw["caes_single_osr_pct"].dropna().iloc[0]),
                    "hybrid_osr_pct": osr,
                    "raw_points_n": int(len(raw)),
                    "eligible_points_n": int(len(eligible)),
                    "frontier_points_n": int(len(front)),
                    "min_cost_ratio": float(front["cost_ratio_to_gas"].min()),
                    "max_supply_pct": float(front["supply_pct"].max()),
                    "lfp_supply_share_pct": shares["lfp_supply_share_pct"],
                    "caes_supply_share_pct": shares["caes_supply_share_pct"],
                    "lfp_supply_share_mean_pct": shares["lfp_supply_share_mean_pct"],
                    "caes_supply_share_mean_pct": shares["caes_supply_share_mean_pct"],
                    "strict_hybrid": bool(strict_hybrid),
                    "min_share_pct": float(min_share_frac * 100.0),
                }
                summary_rows.append(summary)

                if save_source:
                    source_rows.append(
                        curve.assign(
                            province_cn=prov_cn,
                            province_en=prov_en,
                            experiment=experiment,
                            stem=stem,
                            row_idx=row_idx,
                            row_label=row_label,
                            dispatch_window_h=float(dw_h),
                            curve_type="hybrid_dispatch_frontier",
                            tech_id="",
                            tech_label="",
                            duration_h=np.nan,
                            osr_pct=np.nan,
                            policy=policy,
                            policy_label=style["label"],
                            hybrid_osr_pct=osr,
                            raw_points_n=int(len(raw)),
                            eligible_points_n=int(len(eligible)),
                            frontier_points_n=int(len(front)),
                            lfp_duration_h=lfp_dur,
                            caes_duration_h=caes_dur,
                            lfp_supply_share_pct=shares["lfp_supply_share_pct"],
                            caes_supply_share_pct=shares["caes_supply_share_pct"],
                            lfp_supply_share_mean_pct=shares["lfp_supply_share_mean_pct"],
                            caes_supply_share_mean_pct=shares["caes_supply_share_mean_pct"],
                            strict_hybrid=bool(strict_hybrid),
                            min_share_pct=float(min_share_frac * 100.0),
                        )
                    )

            ax.legend(
                loc="upper left",
                fontsize=4.15,
                handlelength=1.05,
                borderaxespad=0.2,
                ncol=1,
                columnspacing=0.35,
                handletextpad=0.30,
                labelspacing=0.13,
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
        f"Fig 5 seasonal-MC OSR curves with true LFP-CAES hybrid dispatch — {prov_en} ({prov_cn})",
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
        "Black/orange curves are minimum-cost frontiers from actual LFP+CAES hybrid dispatch capacity grids. "
        "Legend LFP/CAES percentages are frontier supply shares. "
        "Red dashed line = gas parity.",
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

    return summary_rows


def main():
    global OUT_DIR, SOURCE_DIR, FIG_FILE_PREFIX, SUMMARY_FILENAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-csv", default=str(HYBRID_CSV))
    parser.add_argument(
        "--formats", nargs="+", default=["png"], choices=["png", "svg", "pdf"]
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--province", default=None)
    parser.add_argument("--no-source", action="store_true")
    parser.add_argument(
        "--strict-hybrid",
        action="store_true",
        help="Exclude degenerate LFP-only or CAES-only hybrid capacity points.",
    )
    parser.add_argument(
        "--min-share-pct",
        type=float,
        default=0.0,
        help="Minimum actual supply share required for both LFP and CAES.",
    )
    args = parser.parse_args()

    min_share_frac = max(float(args.min_share_pct), 0.0) / 100.0
    if args.strict_hybrid or min_share_frac > 0:
        suffix = STRICT_DIR_SUFFIX
        if min_share_frac > 0:
            suffix += f"_min{int(round(min_share_frac * 100))}pct"
        OUT_DIR = Path(str(OUT_DIR) + suffix)
        SOURCE_DIR = OUT_DIR / "source_data"
        FIG_FILE_PREFIX = f"{FIG_FILE_PREFIX}{suffix}"
        SUMMARY_FILENAME = f"{FIG_FILE_PREFIX}_summary.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    hybrid_df = load_hybrid_results(Path(args.hybrid_csv), args.province)
    if hybrid_df.empty:
        raise SystemExit("No hybrid dispatch rows matched.")

    provinces = (
        hybrid_df[["province_cn", "province_en"]]
        .drop_duplicates()
        .sort_values("province_en")
        .itertuples(index=False, name=None)
    )
    provinces = list(provinces)

    all_summary = []
    for idx, (prov_cn, prov_en) in enumerate(provinces, start=1):
        print(f"=== render {idx}/{len(provinces)}: {prov_en} ({prov_cn}) ===", flush=True)
        source = load_province_source(str(prov_en))
        rows = render_province(
            str(prov_cn),
            str(prov_en),
            source,
            hybrid_df,
            formats=args.formats,
            dpi=args.dpi,
            save_source=not args.no_source,
            strict_hybrid=bool(args.strict_hybrid),
            min_share_frac=min_share_frac,
        )
        all_summary.extend(rows)

    if all_summary:
        summary = pd.DataFrame(all_summary)
        summary = summary.sort_values(
            ["province_en", "row_idx", "dispatch_window_h", "policy"]
        )
        summary.to_csv(SOURCE_DIR / SUMMARY_FILENAME, index=False)
        print(f"=== hybrid summary: {len(summary):,} rows ===", flush=True)


if __name__ == "__main__":
    main()
