"""
M4 Fig 5 B + C — province-level Monte Carlo q75 OSR sheets without LFP 4h.

This figure set uses the Monte Carlo + q75 extreme-event augmentation nodes:
frequency scan = freq4/8/12/16_len5 and run-length scan = freq8_len5/7/10/14.

LFP 4h is excluded. LFP 6/8/10h cost columns are recalculated at load time
from the current M2 duration override table, while dispatch/supply results are
reused because cost parameters do not affect single-tech dispatch.

Output:
  output/m4_economics/nature_figure/freqlen_per_province_osr_mc_q75_no_lfp4h/
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

from config import CONFIG, GAS_POWER_PER_M3, calculate_crf
from ldes_params import load_tech_library
from m4_hybrid_search import get_tech_params
from metrics import compute_csr as metrics_compute_csr
import m4_visualize_single_tech_generic as ms
import m4_visualize_freqlen_per_province_osr as base


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
OUT_DIR = M4_DIR / "nature_figure" / "freqlen_per_province_osr_mc_q75_no_lfp4h"
SOURCE_DIR = OUT_DIR / "source_data"
FIG_TITLE_PREFIX = "Fig 5 province-level MC-q75 OSR curves"
FIG_FOOTER = (
    "Rows 1-4: Monte Carlo q75 frequency scan; rows 5-8: q75 run-length scan. "
    "LFP 4h is excluded; LFP duration is OSR-selected from 6/8/10h only "
    "with current M2 duration-cost overrides. Red dashed line = gas parity."
)
FIG_FILE_PREFIX = "fig5_freqlen_mc_q75_osr_no_lfp4h"
CHAMPION_FILENAME = "fig5_freqlen_mc_q75_province_osr_no_lfp4h_champions.csv"
CHAMPION_METRIC = "OSR"

DW_VALUES = base.DW_VALUES
TECH_ORDER = base.TECH_ORDER
LFP_ALLOWED_DURATIONS = [6, 8, 10]
DISCOUNT_RATE = float(CONFIG["cost"]["discount_rate"])

SCAN_ROWS = [
    ("F", "freq4_len5", 0, "F events=4/yr"),
    ("F", "freq8_len5", 1, "F events=8/yr"),
    ("F", "freq12_len5", 2, "F events=12/yr"),
    ("F", "freq16_len5", 3, "F events=16/yr"),
    ("G", "freq8_len5", 4, "G length=5d"),
    ("G", "freq8_len7", 5, "G length=7d"),
    ("G", "freq8_len10", 6, "G length=10d"),
    ("G", "freq8_len14", 7, "G length=14d"),
]

NEEDED_COLS = [
    "scenario", "dispatch_window_h", "duration_h",
    "province_en", "province_cn", "capacity_mwh",
    "satisfy_rate_pct", "annual_supply_kwh",
    "unit_cost_cny_kwh", "gas_unit_cost_cny_kwh",
    "cost_ratio_to_gas",
]


def recalculate_lfp_costs(df: pd.DataFrame, tech_lib: pd.DataFrame) -> pd.DataFrame:
    """Refresh LFP 6/8/10h economics with current M2 duration overrides."""
    out = df.copy()
    out["duration_h"] = out["duration_h"].astype(int)
    for (scenario, dur), idx in out.groupby(["scenario", "duration_h"]).groups.items():
        tp = get_tech_params(tech_lib, "li_lfp", str(scenario), int(dur))
        if tp is None:
            continue
        cap_mwh = out.loc[idx, "capacity_mwh"].to_numpy(dtype=float)
        annual_supply = out.loc[idx, "annual_supply_kwh"].to_numpy(dtype=float)
        gas_unit = out.loc[idx, "gas_unit_cost_cny_kwh"].to_numpy(dtype=float)

        power_mw = cap_mwh / float(tp.duration_h)
        rated_power_kw = power_mw * 1000.0 * max(tp.charge_power_ratio, 1.0)
        capacity_kwh = cap_mwh * 1000.0
        capex = (
            float(tp.power_cost_cny_kw) * rated_power_kw
            + float(tp.energy_cost_cny_kwh) * capacity_kwh
        )
        crf = calculate_crf(DISCOUNT_RATE, int(tp.calendar_life_years))
        annual_fixed = capex * crf + float(tp.fixed_om_cny_kw_yr) * rated_power_kw
        annual_var = float(tp.variable_om_cny_mwh) * annual_supply / 1000.0
        annual_cost = annual_fixed + annual_var

        unit_cost = np.full_like(annual_supply, np.nan, dtype=float)
        valid = annual_supply > 0
        unit_cost[valid] = annual_cost[valid] / annual_supply[valid]

        cost_ratio = np.full_like(unit_cost, np.nan, dtype=float)
        valid_ratio = valid & np.isfinite(gas_unit) & (gas_unit > 0)
        cost_ratio[valid_ratio] = unit_cost[valid_ratio] / gas_unit[valid_ratio]

        out.loc[idx, "capex_cny"] = capex
        out.loc[idx, "annual_cost_cny"] = annual_cost
        out.loc[idx, "unit_cost_cny_kwh"] = unit_cost
        out.loc[idx, "cost_ratio_to_gas"] = cost_ratio

    return out


def load_filtered_csv(fp: Path, province_filter: str | None) -> pd.DataFrame:
    """Load a single-tech CSV, optionally filtering provinces while streaming."""
    if not province_filter:
        return pd.read_csv(fp, usecols=NEEDED_COLS, low_memory=False)

    needle = province_filter.lower()
    chunks = []
    for chunk in pd.read_csv(
        fp, usecols=NEEDED_COLS, chunksize=200_000, low_memory=False
    ):
        mask = (
            chunk["province_cn"].astype(str).str.lower().str.contains(needle)
            | chunk["province_en"].astype(str).str.lower().str.contains(needle)
        )
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        return pd.DataFrame(columns=NEEDED_COLS)
    return pd.concat(chunks, ignore_index=True)


def compute_csr_for_province(prov_df: pd.DataFrame) -> float | None:
    """Compute CSR for one province-duration curve using the 3R metric code."""
    if prov_df.empty or "annual_supply_kwh" not in prov_df.columns:
        return None

    df = prov_df.sort_values("satisfy_rate_pct").drop_duplicates(
        subset=["satisfy_rate_pct"], keep="first").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "satisfy_rate_pct",
            "annual_supply_kwh",
            "unit_cost_cny_kwh",
            "gas_unit_cost_cny_kwh",
        ]
    )
    if len(df) < 2:
        return None

    positive = df["satisfy_rate_pct"].astype(float).gt(0) & df[
        "annual_supply_kwh"
    ].astype(float).gt(0)
    if not positive.any():
        return None

    total_demand = np.nanmedian(
        df.loc[positive, "annual_supply_kwh"].astype(float).to_numpy()
        * 100.0
        / df.loc[positive, "satisfy_rate_pct"].astype(float).to_numpy()
    )
    if not np.isfinite(total_demand) or total_demand <= 0:
        return None

    df = df.copy()
    df["peak_supply_kwh"] = df["annual_supply_kwh"].astype(float)
    df["annual_cost"] = (
        df["annual_supply_kwh"].astype(float)
        * df["unit_cost_cny_kwh"].astype(float)
    )
    df["total_demand_kwh"] = float(total_demand)

    gas_unit_cny = float(df["gas_unit_cost_cny_kwh"].iloc[0])
    if not np.isfinite(gas_unit_cny) or gas_unit_cny <= 0:
        return None
    gen_cost = float(CONFIG["gas"]["generator_cost"])
    fake_gas_price = (gas_unit_cny - gen_cost) * GAS_POWER_PER_M3
    try:
        return float(metrics_compute_csr(
            df, fake_gas_price, GAS_POWER_PER_M3, gen_cost))
    except Exception:
        return None


def province_champion_duration(df_cell: pd.DataFrame) -> tuple[int | None, float]:
    """Select one duration by the configured 3R metric for one province cell."""
    metric = CHAMPION_METRIC.upper()
    if metric not in {"OSR", "CSR"}:
        raise ValueError(f"Unsupported CHAMPION_METRIC: {CHAMPION_METRIC}")

    scores: dict[int, float] = {}
    for dur, sub in df_cell.groupby("duration_h", sort=True):
        if metric == "CSR":
            score = compute_csr_for_province(sub)
        else:
            score = ms.compute_osr_for_province(sub)
        if score is None or not np.isfinite(score):
            continue
        scores[int(dur)] = float(score)

    if not scores:
        return None, float("nan")
    best = max(scores, key=scores.get)
    return best, scores[best]


def build_province_records(province_filter: str | None = None) -> tuple[dict, dict, pd.DataFrame]:
    records = defaultdict(dict)
    province_names: dict[str, str] = {}
    champion_rows = []
    tech_lib = load_tech_library()
    metric = CHAMPION_METRIC.upper()
    score_col = f"{metric.lower()}_pct"

    for experiment, stem, row_idx, row_label in SCAN_ROWS:
        print(f"=== {stem} ({row_label}) ===", flush=True)
        for tech_id in TECH_ORDER:
            fp = M4_DIR / f"m4_single_tech_{stem}_{tech_id}.csv"
            if not fp.exists():
                print(f"  WARNING missing {fp.name}", flush=True)
                continue
            df = load_filtered_csv(fp, province_filter)
            if df.empty:
                print(f"  {tech_id}: 0 rows after province filter", flush=True)
                continue
            df = df[df["scenario"].eq(ms.SCENARIO)].copy()
            df = df[df["dispatch_window_h"].isin(DW_VALUES)]
            if tech_id == "li_lfp":
                df = df[df["duration_h"].isin(LFP_ALLOWED_DURATIONS)].copy()
                df = recalculate_lfp_costs(df, tech_lib)
                print(
                    f"  {tech_id}: {len(df):,} rows "
                    f"(latest cost, durations {LFP_ALLOWED_DURATIONS})",
                    flush=True,
                )
            else:
                print(f"  {tech_id}: {len(df):,} rows", flush=True)

            for (prov_cn, prov_en, dw_h), cell in df.groupby(
                ["province_cn", "province_en", "dispatch_window_h"],
                sort=False,
            ):
                province_names[str(prov_cn)] = str(prov_en)
                best_dur, score_pct = province_champion_duration(cell)
                if best_dur is None:
                    continue
                sub = cell[cell["duration_h"].eq(float(best_dur))]
                curve = base.interpolate_curve(sub)
                if curve.empty:
                    continue

                key = (experiment, int(row_idx), float(dw_h), tech_id)
                records[str(prov_cn)][key] = {
                    "province_cn": str(prov_cn),
                    "province_en": str(prov_en),
                    "experiment": experiment,
                    "stem": stem,
                    "row_idx": int(row_idx),
                    "row_label": row_label,
                    "dispatch_window_h": float(dw_h),
                    "tech_id": tech_id,
                    "tech_label": ms.TECH_FILES[tech_id][0],
                    "duration_h": int(best_dur),
                    "champion_metric": metric,
                    "champion_score_pct": float(score_pct),
                    score_col: float(score_pct),
                    "curve": curve,
                }
                champion_rows.append({
                    "province_cn": str(prov_cn),
                    "province_en": str(prov_en),
                    "experiment": experiment,
                    "stem": stem,
                    "row_idx": int(row_idx),
                    "row_label": row_label,
                    "dispatch_window_h": float(dw_h),
                    "tech_id": tech_id,
                    "tech_label": ms.TECH_FILES[tech_id][0],
                    "duration_h": int(best_dur),
                    "champion_metric": metric,
                    "champion_score_pct": float(score_pct),
                    score_col: float(score_pct),
                })

            del df

    champions = pd.DataFrame(champion_rows)
    return records, province_names, champions


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
                    f" | {rec['champion_metric']} "
                    f"{rec['champion_score_pct']:.0f}%"
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
                    score_col = f"{rec['champion_metric'].lower()}_pct"
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
                        champion_metric=rec["champion_metric"],
                        champion_score_pct=rec["champion_score_pct"],
                        **{score_col: rec["champion_score_pct"]},
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
        f"{FIG_TITLE_PREFIX} — {prov_en} ({prov_cn})",
        ha="left", va="top", fontsize=13, fontweight="bold",
        color=ms.PALETTE["black"], **ms.cjk_font_kwargs(bold=True))
    fig.text(
        0.065, 0.975,
        FIG_FOOTER,
        ha="left", va="top", fontsize=7.4,
        color=ms.PALETTE["black"], **ms.cjk_font_kwargs())

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

    records, province_names, champions = build_province_records(args.province)
    if champions.empty:
        raise SystemExit(
            f"No valid {CHAMPION_METRIC.upper()} champion curves were produced."
        )
    champions = champions.sort_values([
        "province_en", "experiment", "row_idx",
        "dispatch_window_h", "tech_id",
    ])
    champions.to_csv(
        SOURCE_DIR / CHAMPION_FILENAME,
        index=False,
    )
    print(f"=== champions: {len(champions):,} rows ===")
    lfp_counts = champions[champions["tech_id"].eq("li_lfp")][
        "duration_h"
    ].value_counts().sort_index()
    print(f"=== LFP selected durations after excluding 4h ({CHAMPION_METRIC.upper()}) ===")
    print(lfp_counts.to_string())

    provinces = sorted(province_names.items(), key=lambda kv: kv[1])
    if not provinces:
        raise SystemExit("No province matched.")

    for idx, (prov_cn, prov_en) in enumerate(provinces, start=1):
        print(f"=== render {idx}/{len(provinces)}: {prov_en} ({prov_cn}) ===")
        render_province(
            prov_cn, prov_en, records.get(prov_cn, {}),
            formats=args.formats, dpi=args.dpi,
            save_source=not args.no_source,
        )


if __name__ == "__main__":
    main()
