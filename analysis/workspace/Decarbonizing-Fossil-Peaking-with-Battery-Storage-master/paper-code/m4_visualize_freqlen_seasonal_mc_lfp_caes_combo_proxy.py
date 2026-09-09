"""
M4 Fig 5 sequential LFP-CAES residual OSR proxy curves.

This script reads the province-level seasonal MC, OSR-selected, no-LFP4h source
curves and creates two sequential combination curves for each province:

  1. LFP -> CAES: LFP first serves demand up to its single-tech OSR, then CAES
     is scaled to the residual unserved demand.
  2. CAES -> LFP: the same residual stacking logic in the reverse order.

The curves are a source-data proxy, not a co-optimized hourly hybrid dispatch.
The second technology's x-axis contribution is residual-scaled:

  combo_supply_pct = first_osr_pct + (100 - first_osr_pct) * second_supply_pct / 100

Accordingly, the proxy endpoint is:

  first_osr_pct + (100 - first_osr_pct) * second_osr_pct / 100

Output:
  output/m4_economics/nature_figure/
    freqlen_per_province_osr_seasonal_mc_lfp_caes_combo_proxy/
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
    / "freqlen_per_province_osr_seasonal_mc_lfp_caes_combo_proxy"
)
SOURCE_DIR = OUT_DIR / "source_data"
FIG_FILE_PREFIX = "fig5_freqlen_seasonal_mc_lfp_caes_combo_proxy"
SUMMARY_FILENAME = f"{FIG_FILE_PREFIX}_summary.csv"

COMBO_SPECS = [
    ("lfp_then_caes", "LFP -> CAES", "li_lfp", "caes", "#1B6CA8", "-"),
    ("caes_then_lfp", "CAES -> LFP", "caes", "li_lfp", "#C96F2D", "--"),
]

TECH_LABELS = {
    "li_lfp": "LFP",
    "caes": "CAES",
}


def _finite_curve(curve: pd.DataFrame) -> pd.DataFrame:
    out = curve[["supply_pct", "cost_ratio_to_gas"]].dropna().copy()
    out = out[np.isfinite(out["supply_pct"]) & np.isfinite(out["cost_ratio_to_gas"])]
    out = out.sort_values("supply_pct").drop_duplicates("supply_pct", keep="first")
    return out.reset_index(drop=True)


def _interp_y(curve: pd.DataFrame, x_value: float) -> float:
    curve = _finite_curve(curve)
    if curve.empty:
        return float("nan")
    x = curve["supply_pct"].to_numpy(dtype=float)
    y = curve["cost_ratio_to_gas"].to_numpy(dtype=float)
    if x_value <= x.min():
        return float(y[0])
    if x_value >= x.max():
        return float(y[-1])
    return float(np.interp(x_value, x, y))


def clip_curve_at_osr(curve: pd.DataFrame, osr_pct: float) -> pd.DataFrame:
    """Return a component curve from its first available point to its OSR."""
    curve = _finite_curve(curve)
    if curve.empty or not np.isfinite(osr_pct) or osr_pct <= 0:
        return pd.DataFrame(columns=["component_supply_pct", "cost_ratio_to_gas"])

    osr_pct = float(min(max(osr_pct, 0.0), 100.0))
    clipped = curve[curve["supply_pct"].le(osr_pct)].copy()
    if osr_pct >= curve["supply_pct"].min():
        y_osr = _interp_y(curve, osr_pct)
        clipped = pd.concat(
            [
                clipped,
                pd.DataFrame(
                    {"supply_pct": [osr_pct], "cost_ratio_to_gas": [y_osr]}
                ),
            ],
            ignore_index=True,
        )
    clipped = clipped.sort_values("supply_pct").drop_duplicates(
        "supply_pct", keep="last"
    )
    clipped = clipped.rename(columns={"supply_pct": "component_supply_pct"})
    return clipped.reset_index(drop=True)


def build_combo_curve(
    source: pd.DataFrame,
    first_tech: str,
    second_tech: str,
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    first = source[source["tech_id"].eq(first_tech)]
    second = source[source["tech_id"].eq(second_tech)]
    if first.empty or second.empty:
        return None, None

    first_duration = int(first["duration_h"].dropna().iloc[0])
    second_duration = int(second["duration_h"].dropna().iloc[0])
    first_osr = float(first["osr_pct"].dropna().iloc[0])
    second_osr = float(second["osr_pct"].dropna().iloc[0])
    first_osr = min(max(first_osr, 0.0), 100.0)
    second_osr = min(max(second_osr, 0.0), 100.0)
    combo_osr = first_osr + (100.0 - first_osr) * second_osr / 100.0
    combo_osr = min(max(combo_osr, 0.0), 100.0)

    first_seg = clip_curve_at_osr(first, first_osr)
    second_seg = clip_curve_at_osr(second, second_osr)
    pieces = []

    if not first_seg.empty:
        first_seg = first_seg.copy()
        first_seg["combo_supply_pct"] = first_seg["component_supply_pct"]
        first_seg["stage"] = 1
        first_seg["component_tech_id"] = first_tech
        first_seg["component_tech_label"] = TECH_LABELS[first_tech]
        first_seg["component_duration_h"] = first_duration
        pieces.append(first_seg)

    if not second_seg.empty and first_osr < 100.0:
        second_seg = second_seg.copy()
        second_seg["combo_supply_pct"] = (
            first_osr + (100.0 - first_osr) * second_seg["component_supply_pct"] / 100.0
        )
        if second_seg["component_supply_pct"].min() > 0.0:
            second_anchor = pd.DataFrame(
                {
                    "component_supply_pct": [0.0],
                    "cost_ratio_to_gas": [_interp_y(second, 0.0)],
                    "combo_supply_pct": [first_osr],
                }
            )
            second_seg = pd.concat([second_anchor, second_seg], ignore_index=True)
        second_seg["stage"] = 2
        second_seg["component_tech_id"] = second_tech
        second_seg["component_tech_label"] = TECH_LABELS[second_tech]
        second_seg["component_duration_h"] = second_duration
        pieces.append(second_seg)

    if not pieces:
        return None, None

    curve = pd.concat(pieces, ignore_index=True)
    curve["first_tech_id"] = first_tech
    curve["second_tech_id"] = second_tech
    curve["first_osr_pct"] = first_osr
    curve["second_osr_pct"] = second_osr
    curve["combo_osr_proxy_pct"] = combo_osr
    curve["first_duration_h"] = first_duration
    curve["second_duration_h"] = second_duration
    meta = {
        "first_tech_id": first_tech,
        "second_tech_id": second_tech,
        "first_duration_h": first_duration,
        "second_duration_h": second_duration,
        "first_osr_pct": first_osr,
        "second_osr_pct": second_osr,
        "combo_osr_proxy_pct": combo_osr,
    }
    return curve, meta


def load_province_source(prov_en: str) -> pd.DataFrame:
    slug = base.slugify(prov_en)
    fp = INPUT_SOURCE_DIR / f"{seasonal_plotter.FIG_FILE_PREFIX}_{slug}_source.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_csv(fp)


def add_reference_curves(ax, panel_source: pd.DataFrame):
    for tech_id, color, ls in [
        ("li_lfp", ms.TECH_COLORS["li_lfp"], ":"),
        ("caes", ms.TECH_COLORS["caes"], ":"),
    ]:
        sub = panel_source[panel_source["tech_id"].eq(tech_id)]
        if sub.empty:
            continue
        ax.plot(
            sub["supply_pct"],
            sub["cost_ratio_to_gas"],
            color=color,
            lw=0.75,
            ls=ls,
            alpha=0.28,
            zorder=2,
        )


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
            if row_idx == 7:
                ax.set_xlabel("Cumulative peaking supply ratio proxy (%)", labelpad=2)
            if col_idx == 0:
                ax.set_ylabel("Component unit cost / gas cost", labelpad=2)

            panel_source = province_source[
                province_source["stem"].eq(stem)
                & province_source["dispatch_window_h"].eq(float(dw_h))
                & province_source["tech_id"].isin(["li_lfp", "caes"])
            ].copy()
            add_reference_curves(ax, panel_source)

            handles = []
            for combo_id, combo_label, first_tech, second_tech, color, ls in COMBO_SPECS:
                combo_curve, meta = build_combo_curve(
                    panel_source, first_tech=first_tech, second_tech=second_tech
                )
                if combo_curve is None:
                    continue

                label = (
                    f"{combo_label} | proxy {meta['combo_osr_proxy_pct']:.0f}%"
                )
                first_labeled = False
                for stage in [1, 2]:
                    seg = combo_curve[combo_curve["stage"].eq(stage)]
                    if seg.empty:
                        continue
                    line, = ax.plot(
                        seg["combo_supply_pct"],
                        seg["cost_ratio_to_gas"],
                        color=color,
                        lw=2.1 if combo_id == "lfp_then_caes" else 1.9,
                        ls=ls,
                        marker="o" if stage == 1 else "s",
                        ms=2.4,
                        markevery=35,
                        zorder=4,
                        label=label if not first_labeled else None,
                    )
                    if not first_labeled:
                        handles.append(line)
                        first_labeled = True

                end_x = meta["combo_osr_proxy_pct"]
                end_y = _interp_y(
                    panel_source[panel_source["tech_id"].eq(second_tech)],
                    meta["second_osr_pct"],
                )
                if np.isfinite(end_y):
                    ax.scatter(
                        [end_x],
                        [end_y],
                        s=14,
                        color=color,
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=5,
                    )

                row = {
                    "province_cn": prov_cn,
                    "province_en": prov_en,
                    "experiment": experiment,
                    "stem": stem,
                    "row_idx": row_idx,
                    "row_label": row_label,
                    "dispatch_window_h": float(dw_h),
                    "combo_id": combo_id,
                    "combo_label": combo_label,
                    **meta,
                }
                summary_rows.append(row)

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
                            combo_id=combo_id,
                            combo_label=combo_label,
                        )
                    )

            ax.legend(
                loc="upper left",
                fontsize=5.1,
                handlelength=1.35,
                borderaxespad=0.2,
                ncol=1,
                columnspacing=0.4,
                handletextpad=0.35,
                labelspacing=0.18,
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
        f"Fig 5 sequential LFP-CAES residual OSR proxy — {prov_en} ({prov_cn})",
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
        "Proxy stack from single-tech OSR champion curves. "
        "The first resource serves demand to its OSR; the second resource is residual-scaled. "
        "Thin dotted lines are original LFP and CAES single-tech curves; red dashed line = gas parity.",
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
        print(f"  saved: {path.name}")
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
        print(f"=== render {idx}/{len(provinces)}: {prov_en} ({prov_cn}) ===")
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
            [
                "province_en",
                "row_idx",
                "dispatch_window_h",
                "combo_id",
            ]
        )
        summary.to_csv(SOURCE_DIR / SUMMARY_FILENAME, index=False)
        print(f"=== summary: {len(summary):,} rows ===")


if __name__ == "__main__":
    main()
