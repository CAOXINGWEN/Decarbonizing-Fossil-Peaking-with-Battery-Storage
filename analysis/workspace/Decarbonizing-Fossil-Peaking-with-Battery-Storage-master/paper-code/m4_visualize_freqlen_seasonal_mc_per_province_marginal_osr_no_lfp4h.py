"""
M4 Fig 5 seasonal-MC province-level marginal OSR curves without LFP 4h.

This figure set uses the same OSR-selected champion durations as
freqlen_per_province_osr_seasonal_mc_no_lfp4h, but plots marginal-cost
curves instead of average unit-cost curves. Single-technology curves come
from the full capacity scan. Hybrid LFP+CAES curves come from the hourly
capacity-pair dispatch frontier.

For each province / row / dispatch-window / technology:

  annual_cost = annual_supply_kwh * unit_cost_cny_kwh
  marginal_cost = d(annual_cost) / d(annual_supply_kwh)
  plotted_y = marginal_cost / gas_unit_cost_cny_kwh

Thus the red dashed line y=1 is the OSR gas-parity threshold.

Output:
  output/m4_economics/nature_figure/
    freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr/
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
from scipy.interpolate import PchipInterpolator

import m4_visualize_freqlen_per_province_osr as base
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal
import m4_visualize_freqlen_mc_q75_per_province_osr_no_lfp4h as plotter
import m4_visualize_single_tech_generic as ms
from ldes_params import load_tech_library


seasonal_plotter = seasonal.plotter

PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
INPUT_SOURCE_DIR = (
    M4_DIR
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h"
    / "source_data"
)
CHAMPIONS_CSV = INPUT_SOURCE_DIR / seasonal_plotter.CHAMPION_FILENAME
HYBRID_CSV = M4_DIR / "m4_hybrid_lfp_caes_seasonal_mc_no_lfp4h.csv"
DENSE_HYBRID_CSV = M4_DIR / "m4_hybrid_lfp_caes_dense_frontier_no_lfp4h.csv"

OUT_DIR = (
    M4_DIR
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h_marginal_osr_smoothed"
)
SOURCE_DIR = OUT_DIR / "source_data"
FIG_FILE_PREFIX = "fig5_freqlen_seasonal_mc_marginal_osr_smoothed_no_lfp4h"
SUMMARY_FILENAME = f"{FIG_FILE_PREFIX}_summary.csv"

SMOOTH_BIN_WIDTH_PCT = 0.5
SMOOTH_ROLLING_BINS = 9
SINGLE_FRONTIER_MODE = "single_full_scan_lower_convex_envelope"

NEEDED_COLS = [
    "scenario",
    "dispatch_window_h",
    "duration_h",
    "province_cn",
    "province_en",
    "capacity_mwh",
    "satisfy_rate_pct",
    "annual_supply_kwh",
    "unit_cost_cny_kwh",
    "gas_unit_cost_cny_kwh",
    "cost_ratio_to_gas",
]

HYBRID_COLS = [
    "satisfy_rate_pct",
    "annual_supply_kwh",
    "annual_supply_lfp_kwh",
    "annual_supply_ldes_kwh",
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
    "lfp_duration_h",
    "caes_duration_h",
    "lfp_capacity_mwh",
    "caes_capacity_mwh",
    "lfp_single_osr_pct",
    "caes_single_osr_pct",
    "lfp_grid_n",
    "caes_grid_n",
]

DENSE_HYBRID_COLS = [
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
    "lfp_single_osr_pct",
    "caes_single_osr_pct",
    "lfp_grid_n",
    "caes_grid_n",
    "evaluated_pairs_n",
    "total_demand_annual_kwh",
    "frontier_mode",
]

HYBRID_STYLE = {
    "hybrid_lfp_first": {
        "policy": "lfp_first",
        "label": "LFP-first hybrid",
        "color": "#111111",
        "ls": "-",
        "marker": "D",
    },
    "hybrid_caes_first": {
        "policy": "caes_first",
        "label": "CAES-first hybrid",
        "color": "#D55E00",
        "ls": "-.",
        "marker": "P",
    },
}
HYBRID_ORDER = list(HYBRID_STYLE)


def format_share_pct(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if 0.0 < value < 10.0:
        return f"{value:.1f}%"
    return f"{value:.0f}%"


def load_champions(province_filter: str | None = None) -> pd.DataFrame:
    if not CHAMPIONS_CSV.exists():
        raise FileNotFoundError(CHAMPIONS_CSV)
    df = pd.read_csv(CHAMPIONS_CSV)
    if province_filter:
        needle = province_filter.lower()
        mask = (
            df["province_en"].astype(str).str.lower().str.contains(needle)
            | df["province_cn"].astype(str).str.lower().str.contains(needle)
        )
        df = df[mask].copy()
    return df


def load_single_full_scan(stem: str, tech_id: str, province_filter: str | None) -> pd.DataFrame:
    fp = M4_DIR / f"m4_single_tech_{stem}_{tech_id}.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)

    df = pd.read_csv(fp, usecols=NEEDED_COLS, low_memory=False)
    if province_filter:
        needle = province_filter.lower()
        mask = (
            df["province_en"].astype(str).str.lower().str.contains(needle)
            | df["province_cn"].astype(str).str.lower().str.contains(needle)
        )
        df = df.loc[mask].copy()

    df = df[df["scenario"].eq(ms.SCENARIO)].copy()
    df = df[df["dispatch_window_h"].isin(seasonal_plotter.DW_VALUES)].copy()
    df["duration_h"] = df["duration_h"].astype(int)
    if tech_id == "li_lfp":
        df = df[df["duration_h"].isin(plotter.LFP_ALLOWED_DURATIONS)].copy()
        df = plotter.recalculate_lfp_costs(df, load_tech_library())
    return df


def load_hybrid_dispatch(province_filter: str | None) -> pd.DataFrame:
    if not HYBRID_CSV.exists():
        raise FileNotFoundError(HYBRID_CSV)

    if not province_filter:
        df = pd.read_csv(HYBRID_CSV, usecols=HYBRID_COLS, low_memory=False)
    else:
        needle = province_filter.lower()
        chunks = []
        for chunk in pd.read_csv(HYBRID_CSV, usecols=HYBRID_COLS, chunksize=200_000, low_memory=False):
            mask = (
                chunk["province_en"].astype(str).str.lower().str.contains(needle)
                | chunk["province_cn"].astype(str).str.lower().str.contains(needle)
            )
            if mask.any():
                chunks.append(chunk.loc[mask].copy())
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=HYBRID_COLS)

    df = df[df["scenario"].eq(ms.SCENARIO)].copy()
    df = df[df["dispatch_window_h"].isin(seasonal_plotter.DW_VALUES)].copy()
    df = df[df["policy"].isin([style["policy"] for style in HYBRID_STYLE.values()])].copy()
    return df


def load_dense_hybrid_frontier(fp: Path, province_filter: str | None) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(fp)

    if not province_filter:
        df = pd.read_csv(fp, usecols=DENSE_HYBRID_COLS, low_memory=False)
    else:
        needle = province_filter.lower()
        chunks = []
        for chunk in pd.read_csv(fp, usecols=DENSE_HYBRID_COLS, chunksize=200_000, low_memory=False):
            mask = (
                chunk["province_en"].astype(str).str.lower().str.contains(needle)
                | chunk["province_cn"].astype(str).str.lower().str.contains(needle)
            )
            if mask.any():
                chunks.append(chunk.loc[mask].copy())
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=DENSE_HYBRID_COLS)

    df = df[df["scenario"].eq(ms.SCENARIO)].copy()
    df = df[df["dispatch_window_h"].isin(seasonal_plotter.DW_VALUES)].copy()
    df = df[df["policy"].isin([style["policy"] for style in HYBRID_STYLE.values()])].copy()
    return df


def system_cost_frontier_from_scan(sub: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "satisfy_rate_pct",
        "annual_supply_kwh",
        "unit_cost_cny_kwh",
        "gas_unit_cost_cny_kwh",
    ]
    df = sub[cols].dropna().copy()
    df = df[
        np.isfinite(df["satisfy_rate_pct"])
        & np.isfinite(df["annual_supply_kwh"])
        & np.isfinite(df["unit_cost_cny_kwh"])
        & np.isfinite(df["gas_unit_cost_cny_kwh"])
        & (df["satisfy_rate_pct"] > 0)
        & (df["annual_supply_kwh"] > 0)
        & (df["unit_cost_cny_kwh"] > 0)
        & (df["gas_unit_cost_cny_kwh"] > 0)
    ]
    if df.empty:
        return pd.DataFrame()

    df["annual_cost_cny"] = df["annual_supply_kwh"] * df["unit_cost_cny_kwh"]
    total_demand = (
        df["annual_supply_kwh"] / (df["satisfy_rate_pct"] / 100.0)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if total_demand.empty:
        return pd.DataFrame()

    total_demand_annual_kwh = float(total_demand.median())
    gas_unit = float(df["gas_unit_cost_cny_kwh"].median())
    df["target_supply_pct"] = df["annual_supply_kwh"] / total_demand_annual_kwh * 100.0
    df["target_annual_supply_kwh"] = df["annual_supply_kwh"]
    df["total_demand_annual_kwh"] = total_demand_annual_kwh
    df["frontier_mode"] = SINGLE_FRONTIER_MODE

    zero = pd.DataFrame(
        [
            {
                "target_supply_pct": 0.0,
                "target_annual_supply_kwh": 0.0,
                "annual_cost_cny": 0.0,
                "gas_unit_cost_cny_kwh": gas_unit,
                "satisfy_rate_pct": 0.0,
                "annual_supply_kwh": 0.0,
                "unit_cost_cny_kwh": np.nan,
                "total_demand_annual_kwh": total_demand_annual_kwh,
                "frontier_mode": SINGLE_FRONTIER_MODE,
            }
        ]
    )
    return lower_convex_envelope_frame(pd.concat([zero, df], ignore_index=True))


def lower_convex_envelope_indices(x: np.ndarray, y: np.ndarray) -> list[int]:
    order = np.lexsort((y, x))
    points: list[tuple[float, float, int]] = []
    for idx in order:
        xi = float(x[idx])
        yi = float(y[idx])
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        if xi < 0 or yi < 0:
            continue
        if points and np.isclose(xi, points[-1][0], rtol=0.0, atol=1e-6):
            if yi < points[-1][1]:
                points[-1] = (xi, yi, int(idx))
            continue
        points.append((xi, yi, int(idx)))
        while len(points) >= 3:
            x1, y1, _ = points[-3]
            x2, y2, _ = points[-2]
            x3, y3, _ = points[-1]
            if x2 <= x1 or x3 <= x2:
                points.pop(-2)
                continue
            s12 = (y2 - y1) / (x2 - x1)
            s23 = (y3 - y2) / (x3 - x2)
            if s23 < s12 - 1e-12:
                points.pop(-2)
            else:
                break
    return [idx for _, _, idx in points]


def lower_convex_envelope_frame(front: pd.DataFrame) -> pd.DataFrame:
    if front.empty:
        return front.copy()
    required = ["annual_supply_kwh", "annual_cost_cny"]
    if not set(required).issubset(front.columns):
        return pd.DataFrame()

    clean = front.copy()
    clean = clean[
        np.isfinite(clean["annual_supply_kwh"])
        & np.isfinite(clean["annual_cost_cny"])
        & (clean["annual_supply_kwh"] >= 0)
        & (clean["annual_cost_cny"] >= 0)
    ].copy()
    if clean.empty:
        return clean

    clean = clean.sort_values(["annual_supply_kwh", "annual_cost_cny"]).drop_duplicates(
        "annual_supply_kwh", keep="first"
    )
    clean = clean.reset_index(drop=True)
    hull_idx = lower_convex_envelope_indices(
        clean["annual_supply_kwh"].to_numpy(dtype=float),
        clean["annual_cost_cny"].to_numpy(dtype=float),
    )
    if not hull_idx:
        return clean.iloc[0:0].copy()
    return clean.iloc[hull_idx].sort_values("annual_supply_kwh").reset_index(drop=True)


def marginal_from_system_cost_frontier(front: pd.DataFrame) -> pd.DataFrame:
    required = ["annual_supply_kwh", "annual_cost_cny", "gas_unit_cost_cny_kwh"]
    if front.empty or not set(required).issubset(front.columns):
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )
    keep_cols = list(dict.fromkeys(required + ["target_supply_pct", "total_demand_annual_kwh"]))
    df = front[keep_cols].dropna(subset=required).copy()
    df = df[
        np.isfinite(df["annual_supply_kwh"])
        & np.isfinite(df["annual_cost_cny"])
        & np.isfinite(df["gas_unit_cost_cny_kwh"])
        & (df["annual_supply_kwh"] >= 0)
        & (df["annual_cost_cny"] >= 0)
        & (df["gas_unit_cost_cny_kwh"] > 0)
    ]
    df = lower_convex_envelope_frame(df)
    if len(df) < 2:
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )

    supply = df["annual_supply_kwh"].to_numpy(dtype=float)
    cost = df["annual_cost_cny"].to_numpy(dtype=float)
    gas = df["gas_unit_cost_cny_kwh"].to_numpy(dtype=float)

    total_demand = pd.Series(dtype=float)
    if "total_demand_annual_kwh" in df.columns:
        total_demand = df["total_demand_annual_kwh"].replace([np.inf, -np.inf], np.nan).dropna()
    if total_demand.empty and "target_supply_pct" in df.columns:
        target = df["target_supply_pct"].replace([np.inf, -np.inf], np.nan)
        total_demand = (
            df["annual_supply_kwh"] / (target / 100.0)
        ).replace([np.inf, -np.inf], np.nan).dropna()
    if total_demand.empty:
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )
    total_demand_annual_kwh = float(total_demand.median())
    if not np.isfinite(total_demand_annual_kwh) or total_demand_annual_kwh <= 0:
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )

    ds = np.diff(supply)
    dc = np.diff(cost)
    valid = (ds > 0) & (dc > 0)
    if not np.any(valid):
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )

    marginal = dc[valid] / ds[valid]
    gas_mid = ((gas[:-1] + gas[1:]) / 2.0)[valid]
    supply_mid = ((supply[:-1] + supply[1:]) / 2.0)[valid]
    x_mid = supply_mid / total_demand_annual_kwh * 100.0
    ratio = marginal / gas_mid

    raw = pd.DataFrame(
        {
            "supply_pct": x_mid,
            "marginal_cost_ratio_to_gas": ratio,
            "marginal_cost_cny_kwh": marginal,
        }
    )
    raw = raw[
        np.isfinite(raw["supply_pct"])
        & np.isfinite(raw["marginal_cost_ratio_to_gas"])
        & np.isfinite(raw["marginal_cost_cny_kwh"])
        & (raw["supply_pct"] >= 0)
        & (raw["supply_pct"] <= 100)
        & (raw["marginal_cost_ratio_to_gas"] > 0)
        & (raw["marginal_cost_cny_kwh"] > 0)
    ].copy()
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )

    endpoint_pct = min(100.0, float(supply[-1] / total_demand_annual_kwh * 100.0))
    anchor = pd.concat(
        [
            pd.DataFrame(
                {
                    "supply_pct": [0.0],
                    "marginal_cost_ratio_to_gas": [float(raw["marginal_cost_ratio_to_gas"].iloc[0])],
                    "marginal_cost_cny_kwh": [float(raw["marginal_cost_cny_kwh"].iloc[0])],
                }
            ),
            raw,
            pd.DataFrame(
                {
                    "supply_pct": [endpoint_pct],
                    "marginal_cost_ratio_to_gas": [float(raw["marginal_cost_ratio_to_gas"].iloc[-1])],
                    "marginal_cost_cny_kwh": [float(raw["marginal_cost_cny_kwh"].iloc[-1])],
                }
            ),
        ],
        ignore_index=True,
    )
    anchor = (
        anchor.sort_values("supply_pct")
        .groupby("supply_pct", as_index=False)
        .median(numeric_only=True)
        .sort_values("supply_pct")
    )
    if len(anchor) < 2:
        return raw.assign(annual_supply_mid_kwh=raw["supply_pct"] / 100.0 * total_demand_annual_kwh)

    x_min = float(anchor["supply_pct"].min())
    x_max = float(anchor["supply_pct"].max())
    grid = np.arange(x_min, x_max + SMOOTH_BIN_WIDTH_PCT / 2.0, SMOOTH_BIN_WIDTH_PCT)
    grid = grid[(grid >= 0) & (grid <= 100)]
    if len(grid) == 0:
        grid = anchor["supply_pct"].to_numpy(dtype=float)

    log_ratio = np.log(anchor["marginal_cost_ratio_to_gas"].to_numpy(dtype=float))
    log_marginal = np.log(anchor["marginal_cost_cny_kwh"].to_numpy(dtype=float))
    x_anchor = anchor["supply_pct"].to_numpy(dtype=float)
    ratio_interp = PchipInterpolator(x_anchor, log_ratio, extrapolate=False)
    marginal_interp = PchipInterpolator(x_anchor, log_marginal, extrapolate=False)

    out = pd.DataFrame(
        {
            "supply_pct": grid,
            "marginal_cost_ratio_to_gas": np.exp(ratio_interp(grid)),
            "marginal_cost_cny_kwh": np.exp(marginal_interp(grid)),
        }
    )
    out["annual_supply_mid_kwh"] = out["supply_pct"] / 100.0 * total_demand_annual_kwh
    out = out[
        np.isfinite(out["supply_pct"])
        & np.isfinite(out["marginal_cost_ratio_to_gas"])
        & (out["marginal_cost_ratio_to_gas"] > 0)
    ]
    return out.reset_index(drop=True)


def marginal_curve(sub: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "capacity_mwh",
        "satisfy_rate_pct",
        "annual_supply_kwh",
        "unit_cost_cny_kwh",
        "gas_unit_cost_cny_kwh",
    ]
    df = sub[cols].dropna().copy()
    df = df[
        np.isfinite(df["annual_supply_kwh"])
        & np.isfinite(df["unit_cost_cny_kwh"])
        & np.isfinite(df["gas_unit_cost_cny_kwh"])
        & (df["annual_supply_kwh"] > 0)
        & (df["gas_unit_cost_cny_kwh"] > 0)
    ]
    if len(df) < 3:
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )

    df["annual_cost_cny"] = df["annual_supply_kwh"] * df["unit_cost_cny_kwh"]
    df = df.sort_values(["annual_supply_kwh", "capacity_mwh"]).drop_duplicates(
        "annual_supply_kwh", keep="first"
    )
    if len(df) < 3:
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )

    supply = df["annual_supply_kwh"].to_numpy(dtype=float)
    cost = df["annual_cost_cny"].to_numpy(dtype=float)
    sat = df["satisfy_rate_pct"].to_numpy(dtype=float)
    gas = df["gas_unit_cost_cny_kwh"].to_numpy(dtype=float)

    ds = np.diff(supply)
    dc = np.diff(cost)
    valid = ds > 0
    if not np.any(valid):
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )

    marginal = dc[valid] / ds[valid]
    x_mid = ((sat[:-1] + sat[1:]) / 2.0)[valid]
    supply_mid = ((supply[:-1] + supply[1:]) / 2.0)[valid]
    gas_mid = ((gas[:-1] + gas[1:]) / 2.0)[valid]
    ratio = marginal / gas_mid

    out = pd.DataFrame(
        {
            "supply_pct": x_mid,
            "marginal_cost_ratio_to_gas": ratio,
            "marginal_cost_cny_kwh": marginal,
            "annual_supply_mid_kwh": supply_mid,
        }
    )
    out = out[
        np.isfinite(out["supply_pct"])
        & np.isfinite(out["marginal_cost_ratio_to_gas"])
        & (out["supply_pct"] >= 0)
        & (out["supply_pct"] <= 100)
        & (out["marginal_cost_ratio_to_gas"] > 0)
    ]
    return out.sort_values("supply_pct").reset_index(drop=True)


def hybrid_eligible_rows(
    sub: pd.DataFrame,
    require_combo_capacity: bool,
    require_component_supply: bool,
) -> pd.DataFrame:
    clean = sub.dropna(
        subset=[
            "satisfy_rate_pct",
            "annual_supply_kwh",
            "annual_supply_lfp_kwh",
            "annual_supply_ldes_kwh",
            "annual_cost_cny",
            "gas_unit_cost_cny_kwh",
            "lfp_capacity_mwh",
            "caes_capacity_mwh",
        ]
    ).copy()
    clean = clean[
        np.isfinite(clean["satisfy_rate_pct"])
        & np.isfinite(clean["annual_supply_kwh"])
        & np.isfinite(clean["annual_cost_cny"])
        & np.isfinite(clean["gas_unit_cost_cny_kwh"])
        & (clean["satisfy_rate_pct"] > 0)
        & (clean["annual_supply_kwh"] > 0)
        & (clean["annual_cost_cny"] > 0)
        & (clean["gas_unit_cost_cny_kwh"] > 0)
    ]
    if require_combo_capacity:
        clean = clean[
            (clean["lfp_capacity_mwh"] > 0)
            & (clean["caes_capacity_mwh"] > 0)
        ]
    if require_component_supply:
        clean = clean[
            (clean["annual_supply_lfp_kwh"] > 0)
            & (clean["annual_supply_ldes_kwh"] > 0)
        ]
    return clean


def hybrid_cost_frontier(
    sub: pd.DataFrame,
    require_combo_capacity: bool = True,
    require_component_supply: bool = True,
) -> pd.DataFrame:
    clean = hybrid_eligible_rows(
        sub,
        require_combo_capacity=require_combo_capacity,
        require_component_supply=require_component_supply,
    )
    if clean.empty:
        return clean

    clean["supply_bin"] = np.floor(clean["satisfy_rate_pct"] / SMOOTH_BIN_WIDTH_PCT).astype(int)
    clean = clean[(clean["supply_bin"] >= 0) & (clean["supply_bin"] <= int(100 / SMOOTH_BIN_WIDTH_PCT))]
    if clean.empty:
        return clean

    idx = (
        clean.sort_values(["annual_cost_cny", "cost_ratio_to_gas"])
        .groupby("supply_bin", sort=True)
        .head(1)
        .index
    )
    front = clean.loc[idx].copy()
    front = front.sort_values(["annual_supply_kwh", "annual_cost_cny"]).drop_duplicates(
        "annual_supply_kwh", keep="first"
    )

    stack = []
    for row in front.itertuples(index=False):
        row_cost = float(row.annual_cost_cny)
        while stack and row_cost <= float(stack[-1].annual_cost_cny):
            stack.pop()
        stack.append(row)
    if not stack:
        return front.iloc[0:0].copy()

    out = pd.DataFrame([r._asdict() for r in stack])
    out["supply_pct"] = out["satisfy_rate_pct"].astype(float)
    return out.sort_values("annual_supply_kwh").reset_index(drop=True)


def hybrid_marginal_curve(front: pd.DataFrame) -> pd.DataFrame:
    if front.empty or len(front) < 3:
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )

    df = front[
        [
            "supply_pct",
            "annual_supply_kwh",
            "annual_cost_cny",
            "gas_unit_cost_cny_kwh",
        ]
    ].copy()
    df = df.dropna().sort_values(["annual_supply_kwh", "annual_cost_cny"])
    df = df.drop_duplicates("annual_supply_kwh", keep="first")
    if len(df) < 3:
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )

    supply = df["annual_supply_kwh"].to_numpy(dtype=float)
    cost = df["annual_cost_cny"].to_numpy(dtype=float)
    sat = df["supply_pct"].to_numpy(dtype=float)
    gas = df["gas_unit_cost_cny_kwh"].to_numpy(dtype=float)
    ds = np.diff(supply)
    dc = np.diff(cost)
    valid = (ds > 0) & (dc > 0)
    if not np.any(valid):
        return pd.DataFrame(
            columns=[
                "supply_pct",
                "marginal_cost_ratio_to_gas",
                "marginal_cost_cny_kwh",
                "annual_supply_mid_kwh",
            ]
        )

    marginal = dc[valid] / ds[valid]
    gas_mid = ((gas[:-1] + gas[1:]) / 2.0)[valid]
    out = pd.DataFrame(
        {
            "supply_pct": ((sat[:-1] + sat[1:]) / 2.0)[valid],
            "marginal_cost_ratio_to_gas": marginal / gas_mid,
            "marginal_cost_cny_kwh": marginal,
            "annual_supply_mid_kwh": ((supply[:-1] + supply[1:]) / 2.0)[valid],
        }
    )
    out = out[
        np.isfinite(out["supply_pct"])
        & np.isfinite(out["marginal_cost_ratio_to_gas"])
        & (out["supply_pct"] >= 0)
        & (out["supply_pct"] <= 100)
        & (out["marginal_cost_ratio_to_gas"] > 0)
    ]
    return out.sort_values("supply_pct").reset_index(drop=True)


def osr_from_marginal_curve(curve: pd.DataFrame) -> float:
    if curve.empty:
        return float("nan")
    below = curve[curve["marginal_cost_ratio_to_gas"] <= 1.0]
    if below.empty:
        return 0.0
    return float(below["supply_pct"].max())


def hybrid_share_metrics(front: pd.DataFrame) -> dict:
    if front.empty:
        return {
            "lfp_supply_share_pct": float("nan"),
            "caes_supply_share_pct": float("nan"),
        }
    lfp = front["annual_supply_lfp_kwh"].astype(float)
    caes = front["annual_supply_ldes_kwh"].astype(float)
    total = float((lfp + caes).sum())
    if total <= 0:
        lfp_share = float("nan")
    else:
        lfp_share = float(lfp.sum() / total * 100.0)
    return {
        "lfp_supply_share_pct": lfp_share,
        "caes_supply_share_pct": 100.0 - lfp_share if np.isfinite(lfp_share) else float("nan"),
    }


def smooth_marginal_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Convert raw adjacent-point marginal costs into a readable supply curve."""
    if curve.empty:
        return curve.copy()

    df = curve[
        [
            "supply_pct",
            "marginal_cost_ratio_to_gas",
            "marginal_cost_cny_kwh",
            "annual_supply_mid_kwh",
        ]
    ].copy()
    df = df[
        np.isfinite(df["supply_pct"])
        & np.isfinite(df["marginal_cost_ratio_to_gas"])
        & (df["supply_pct"] >= 0)
        & (df["supply_pct"] <= 100)
        & (df["marginal_cost_ratio_to_gas"] > 0)
    ]
    if len(df) < 10:
        return df.sort_values("supply_pct").reset_index(drop=True)

    df["bin_id"] = np.floor(df["supply_pct"] / SMOOTH_BIN_WIDTH_PCT).astype(int)
    df["log_ratio"] = np.log(df["marginal_cost_ratio_to_gas"])
    grouped = (
        df.groupby("bin_id", as_index=False)
        .agg(
            supply_pct=("supply_pct", "median"),
            log_ratio=("log_ratio", "median"),
            marginal_cost_cny_kwh=("marginal_cost_cny_kwh", "median"),
            annual_supply_mid_kwh=("annual_supply_mid_kwh", "median"),
            raw_points_in_bin=("supply_pct", "size"),
        )
        .sort_values("supply_pct")
        .reset_index(drop=True)
    )
    if len(grouped) < 5:
        out = grouped.drop(columns=["log_ratio", "raw_points_in_bin"]).copy()
        out["marginal_cost_ratio_to_gas"] = np.exp(grouped["log_ratio"])
        out["smoothing_raw_points_in_bin"] = grouped["raw_points_in_bin"]
        return out

    min_periods = max(3, SMOOTH_ROLLING_BINS // 3)
    smooth_log = (
        grouped["log_ratio"]
        .rolling(SMOOTH_ROLLING_BINS, center=True, min_periods=min_periods)
        .median()
        .fillna(grouped["log_ratio"])
        .rolling(3, center=True, min_periods=1)
        .mean()
    )
    smooth_ratio = np.exp(smooth_log.to_numpy(dtype=float))
    smooth_ratio = np.maximum.accumulate(smooth_ratio)

    out = grouped.drop(columns=["log_ratio"]).copy()
    out["marginal_cost_ratio_to_gas"] = smooth_ratio
    out = out.rename(columns={"raw_points_in_bin": "smoothing_raw_points_in_bin"})
    return out[
        [
            "supply_pct",
            "marginal_cost_ratio_to_gas",
            "marginal_cost_cny_kwh",
            "annual_supply_mid_kwh",
            "smoothing_raw_points_in_bin",
        ]
    ]


def append_hybrid_records(
    records: dict,
    province_names: dict,
    summary_rows: list[dict],
    province_filter: str | None,
    dense_hybrid_csv: Path,
) -> None:
    t0 = time.time()
    print(f"=== load dense hybrid frontier: {dense_hybrid_csv.name} ===", flush=True)
    df = load_dense_hybrid_frontier(dense_hybrid_csv, province_filter)
    if df.empty:
        print("  no dense hybrid frontier rows after filtering", flush=True)
        return

    style_by_policy = {style["policy"]: (tech_id, style) for tech_id, style in HYBRID_STYLE.items()}
    matched = 0
    group_cols = [
        "province_en",
        "province_cn",
        "stem",
        "row_idx",
        "row_label",
        "dispatch_window_h",
        "policy",
    ]
    for key, sub in df.groupby(group_cols, sort=False):
        prov_en, prov_cn, stem, row_idx, row_label, dw_h, policy = key
        style_item = style_by_policy.get(str(policy))
        if style_item is None:
            continue
        tech_id, style = style_item

        front = sub.dropna(
            subset=[
                "target_supply_pct",
                "target_annual_supply_kwh",
                "annual_cost_cny",
                "gas_unit_cost_cny_kwh",
            ]
        ).copy()
        if front.empty:
            continue
        plot_curve = marginal_from_system_cost_frontier(front)
        if plot_curve.empty:
            continue

        shares = hybrid_share_metrics(front)
        lfp_duration_h = int(front["lfp_duration_h"].iloc[0])
        caes_duration_h = int(front["caes_duration_h"].iloc[0])
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
            "tech_label": style["label"],
            "duration_h": f"{lfp_duration_h}+{caes_duration_h}",
            "lfp_duration_h": lfp_duration_h,
            "caes_duration_h": caes_duration_h,
            "osr_pct": osr_from_marginal_curve(plot_curve),
            "lfp_supply_share_pct": shares["lfp_supply_share_pct"],
            "caes_supply_share_pct": shares["caes_supply_share_pct"],
            "lfp_single_osr_pct": float(front["lfp_single_osr_pct"].iloc[0]),
            "caes_single_osr_pct": float(front["caes_single_osr_pct"].iloc[0]),
            "curve": plot_curve,
            "capacity_points_n": int(front["evaluated_pairs_n"].max()) if "evaluated_pairs_n" in front else int(len(sub)),
            "system_cost_frontier_points_n": int(len(front)),
            "hybrid_frontier_points_n": int(len(front)),
            "raw_marginal_points_n": int(len(plot_curve)),
            "smoothed_marginal_points_n": int(len(plot_curve)),
            "smoothing_bin_width_pct": SMOOTH_BIN_WIDTH_PCT,
            "smoothing_rolling_bins": SMOOTH_ROLLING_BINS,
            "hybrid_frontier_mode": str(front["frontier_mode"].iloc[0]),
            "hybrid_strict_combo": True,
            "lfp_grid_n": int(front["lfp_grid_n"].max()) if "lfp_grid_n" in front else np.nan,
            "caes_grid_n": int(front["caes_grid_n"].max()) if "caes_grid_n" in front else np.nan,
        }
        records[str(prov_cn)][rec_key] = rec
        province_names[str(prov_cn)] = str(prov_en)
        summary_rows.append({k: v for k, v in rec.items() if k not in {"curve"}})
        matched += 1

    print(
        f"  matched dense hybrid marginal curves: {matched:,}; elapsed {time.time() - t0:.1f}s",
        flush=True,
    )
    del df


def build_records(
    champions: pd.DataFrame,
    province_filter: str | None,
    dense_hybrid_csv: Path,
) -> tuple[dict, dict, pd.DataFrame]:
    records = defaultdict(dict)
    province_names = {}
    summary_rows = []

    needed_pairs = (
        champions[["stem", "tech_id"]]
        .drop_duplicates()
        .sort_values(["stem", "tech_id"])
        .itertuples(index=False, name=None)
    )

    for stem, tech_id in list(needed_pairs):
        t0 = time.time()
        champ_pair = champions[
            champions["stem"].eq(stem) & champions["tech_id"].eq(tech_id)
        ].copy()
        if champ_pair.empty:
            continue
        print(f"=== load full scan: {stem} | {tech_id} ===", flush=True)
        df = load_single_full_scan(stem, tech_id, province_filter)
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
            key = (str(prov_en), float(dw_h), int(dur))
            rows = champ_keys.get(key)
            if not rows:
                continue
            cost_front = system_cost_frontier_from_scan(sub)
            plot_curve = marginal_from_system_cost_frontier(cost_front)
            if cost_front.empty or plot_curve.empty:
                continue
            matched += len(rows)
            prov_cn = str(sub["province_cn"].iloc[0])
            province_names[prov_cn] = str(prov_en)
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
                    "curve": plot_curve,
                    "capacity_points_n": int(len(sub)),
                    "system_cost_frontier_points_n": int(len(cost_front)),
                    "raw_marginal_points_n": int(len(plot_curve)),
                    "smoothed_marginal_points_n": int(len(plot_curve)),
                    "smoothing_bin_width_pct": SMOOTH_BIN_WIDTH_PCT,
                    "smoothing_rolling_bins": SMOOTH_ROLLING_BINS,
                }
                records[prov_cn][rec_key] = rec
                summary_rows.append(
                    {
                        k: v
                        for k, v in rec.items()
                        if k not in {"curve"}
                    }
                )

        print(
            f"  matched champion curves: {matched:,}; elapsed {time.time() - t0:.1f}s",
            flush=True,
        )
        del df

    if dense_hybrid_csv.exists():
        append_hybrid_records(records, province_names, summary_rows, province_filter, dense_hybrid_csv)
    else:
        print(f"=== dense hybrid frontier not found: {dense_hybrid_csv} ===", flush=True)

    return records, province_names, pd.DataFrame(summary_rows)


def setup_marginal_axis(ax, panel: str, experiment: str, dw_h: float, show_xlabel: bool, show_ylabel: bool):
    ax.axhline(1.0, color=ms.PALETTE["red"], lw=0.9, ls="--", alpha=0.6)
    ax.fill_between([0, 100], 0.02, 1.0, color=ms.PALETTE["fill_blue"], alpha=0.12, zorder=0)
    ax.fill_between([0, 100], 1.0, 1_000, color=ms.PALETTE["fill_red"], alpha=0.08, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.05, 100)
    ax.grid(axis="both", which="major", color=ms.PALETTE["grid"], lw=0.5)
    ax.tick_params(labelsize=6.4)
    ax.set_title(f"{panel}  {experiment} dw={int(dw_h)}h", loc="left", fontsize=7.8, pad=2)
    ax.set_xlabel("Peaking supply ratio (%)" if show_xlabel else "", labelpad=2)
    ax.set_ylabel("Marginal cost / gas cost" if show_ylabel else "", labelpad=2)


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
            setup_marginal_axis(
                ax,
                base.panel_label(panel_idx),
                experiment,
                dw_h,
                show_xlabel=(row_idx == 7),
                show_ylabel=(col_idx == 0),
            )

            for tech_id in seasonal_plotter.TECH_ORDER:
                rec = prov_records.get((int(row_idx), float(dw_h), tech_id))
                if rec is None:
                    continue
                curve = rec["curve"]
                is_lfp = tech_id == "li_lfp"
                label = (
                    f"{rec['tech_label']} {rec['duration_h']}h"
                    f" | OSR {rec['osr_pct']:.0f}%"
                )
                ax.plot(
                    curve["supply_pct"],
                    curve["marginal_cost_ratio_to_gas"],
                    color=ms.TECH_COLORS[tech_id],
                    lw=1.9 if is_lfp else 1.35,
                    ls="--" if is_lfp else "-",
                    marker="s" if is_lfp else "o",
                    ms=2.8 if is_lfp else 2.1,
                    markevery=40,
                    zorder=3,
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
                            capacity_points_n=rec["capacity_points_n"],
                            system_cost_frontier_points_n=rec["system_cost_frontier_points_n"],
                            raw_marginal_points_n=rec["raw_marginal_points_n"],
                            smoothed_marginal_points_n=rec["smoothed_marginal_points_n"],
                            smoothing_bin_width_pct=rec["smoothing_bin_width_pct"],
                            smoothing_rolling_bins=rec["smoothing_rolling_bins"],
                        )
                    )

            for tech_id in HYBRID_ORDER:
                rec = prov_records.get((int(row_idx), float(dw_h), tech_id))
                if rec is None:
                    continue
                curve = rec["curve"]
                style = HYBRID_STYLE[tech_id]
                lfp_share = rec.get("lfp_supply_share_pct", float("nan"))
                caes_share = rec.get("caes_supply_share_pct", float("nan"))
                if np.isfinite(lfp_share) and np.isfinite(caes_share):
                    share_text = (
                        f"LFP{format_share_pct(lfp_share)}/"
                        f"CAES{format_share_pct(caes_share)}"
                    )
                else:
                    share_text = "LFP/CAES"
                label = (
                    f"{rec['tech_label']} ({share_text})"
                    f" | OSR {rec['osr_pct']:.0f}%"
                )
                ax.plot(
                    curve["supply_pct"],
                    curve["marginal_cost_ratio_to_gas"],
                    color=style["color"],
                    lw=2.2,
                    ls=style["ls"],
                    marker=style["marker"],
                    ms=3.0,
                    markevery=20,
                    zorder=5,
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
                            lfp_duration_h=rec["lfp_duration_h"],
                            caes_duration_h=rec["caes_duration_h"],
                            osr_pct=rec["osr_pct"],
                            lfp_supply_share_pct=rec["lfp_supply_share_pct"],
                            caes_supply_share_pct=rec["caes_supply_share_pct"],
                            lfp_single_osr_pct=rec["lfp_single_osr_pct"],
                            caes_single_osr_pct=rec["caes_single_osr_pct"],
                            capacity_points_n=rec["capacity_points_n"],
                            system_cost_frontier_points_n=rec["system_cost_frontier_points_n"],
                            hybrid_frontier_points_n=rec["hybrid_frontier_points_n"],
                            raw_marginal_points_n=rec["raw_marginal_points_n"],
                            smoothed_marginal_points_n=rec["smoothed_marginal_points_n"],
                            smoothing_bin_width_pct=rec["smoothing_bin_width_pct"],
                            smoothing_rolling_bins=rec["smoothing_rolling_bins"],
                            hybrid_frontier_mode=rec["hybrid_frontier_mode"],
                            hybrid_strict_combo=rec["hybrid_strict_combo"],
                            lfp_grid_n=rec["lfp_grid_n"],
                            caes_grid_n=rec["caes_grid_n"],
                        )
                    )

            ax.legend(
                loc="upper left",
                fontsize=3.75,
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
        f"Fig 5 seasonal-MC smoothed marginal OSR curves — {prov_en} ({prov_cn})",
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
        "Single-tech curves use OSR-selected champion durations from full capacity scans; "
        "hybrid curves use dense full-range LFP+CAES lower convex-envelope frontiers. "
        "Y-axis is marginal cost / gas cost from convex-envelope slopes with shape-preserving interpolation; "
        "red dashed line = OSR gas parity. "
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default=None)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "svg", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-source", action="store_true")
    parser.add_argument("--dense-hybrid-csv", default=str(DENSE_HYBRID_CSV))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    champions = load_champions(args.province)
    if champions.empty:
        raise SystemExit("No champion rows matched.")

    records, province_names, summary = build_records(
        champions,
        args.province,
        Path(args.dense_hybrid_csv),
    )
    if summary.empty:
        raise SystemExit("No marginal OSR curves were produced.")
    summary = summary.sort_values(
        ["province_en", "row_idx", "dispatch_window_h", "tech_id"]
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

    print(f"=== marginal OSR summary: {len(summary):,} rows ===", flush=True)


if __name__ == "__main__":
    main()
