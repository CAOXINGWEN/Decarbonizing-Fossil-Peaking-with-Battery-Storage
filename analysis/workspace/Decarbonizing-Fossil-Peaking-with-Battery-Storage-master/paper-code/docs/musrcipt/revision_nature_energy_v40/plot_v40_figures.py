#!/usr/bin/env python3
"""Build the Europe-main v40 Nature-style figure and Source Data bundle.

The physical dispatch ledger is frozen from v39. Technology costs are
re-reported with the source-aware routes requested for v40: LFP and VRFB use
CNY→EUR; Fe–Cr, gravity, molten-salt TES, A-CAES, LAES and cavern hydrogen use
their model-USD equivalents and USD→EUR.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
EU = WORKSPACE / "dataset/external_hourly/europe_mid_duration_valley_upgrade_v0_1"
DEFAULT_RESULTS = HERE / "currency_work"
V39_RESULTS = HERE.parent / "revision_nature_energy_v39/source_data/europe_mainline_results"
DEFAULT_PANEL = EU / "processed/observed_gas_event_task_definitions_v0_2.csv.gz"
DEFAULT_MAP = EU / "raw/gisco_countries_2024/CNTR_RG_20M_2024_4326.geojson"
RUNNER = EU / "scripts/run_europe_observed_gas_mainline_v0_1.py"
CHINA_F4 = HERE.parent / "revision_nature_energy_v31/main_figures/figure4_turnover_mechanism/source_data"
CHINA_F5 = HERE.parent / "revision_nature_energy_v31/main_figures/figure5_mix_economics/source_data"
CHINA_ED3 = (
    HERE.parent
    / "revision_nature_energy_v31/supplement/ed3_path_interventions/source_data"
)
CHINA_ED2 = (
    HERE.parent
    / "revision_nature_energy_v31/supplement/ed2_event_sensitivity/source_data"
)

CNY_PER_EUR = 8.1185043137255
EUR_PER_CNY = 1.0 / CNY_PER_EUR
MODEL_CNY_PER_USD = 7.2
USD_PER_EUR = 1.1300
EUR_PER_MODEL_CNY_USD_ROUTE = 1.0 / (MODEL_CNY_PER_USD * USD_PER_EUR)
FX_SERIES_KEY = "EXR.A.CNY.EUR.SP00.A"
FX_REFERENCE_YEAR = 2025

TECH_ORDER = [
    "li_lfp",
    "vrfb",
    "fecr_flow",
    "gravity",
    "molten_salt_tes",
    "caes",
    "laes",
    "h2_cavern",
]
TECH_LABEL = {
    "li_lfp": "LFP",
    "vrfb": "VRFB",
    "fecr_flow": "Fe–Cr flow",
    "gravity": "Gravity",
    "molten_salt_tes": "Thermal",
    "caes": "A-CAES",
    "laes": "LAES",
    "h2_cavern": "Cavern H$_2$",
}
TECH_COLOR = {
    "li_lfp": "#0072B2",
    "vrfb": "#56B4E9",
    "fecr_flow": "#CC79A7",
    "gravity": "#737373",
    "molten_salt_tes": "#E69F00",
    "caes": "#009E73",
    "laes": "#D55E00",
    "h2_cavern": "#6A51A3",
}
DISPERSED = "#365D8D"
CLUSTERED = "#D95F02"
NATURAL = "#555555"
LIGHT_GREY = "#E6E8EA"
DARK = "#222222"
SELECTED = ["li_lfp_4h", "caes_24h", "h2_cavern_48h", "h2_cavern_168h"]
SELECTED_LABEL = {
    "li_lfp_4h": "LFP 4 h",
    "caes_24h": "A-CAES 24 h",
    "h2_cavern_48h": "Cavern H$_2$ 48 h",
    "h2_cavern_168h": "Cavern H$_2$ 168 h",
}


def add_eur_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain source CNY fields and append fixed-rate EUR reporting fields."""
    converted = frame.copy()
    targets: list[tuple[str, str]] = []
    for column in frame.columns:
        if "_cny_" in column:
            targets.append((column, column.replace("_cny_", "_eur_")))
        elif column.endswith("_cny"):
            targets.append((column, f"{column[:-4]}_eur"))
    for source, target in targets:
        values = pd.to_numeric(frame[source], errors="raise") * EUR_PER_CNY
        if target in converted.columns:
            if not np.allclose(
                pd.to_numeric(converted[target], errors="raise"),
                values,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(f"Inconsistent CNY/EUR pair: {source}, {target}")
        else:
            converted[target] = values
    if targets:
        converted["reporting_fx_cny_per_eur"] = CNY_PER_EUR
        converted["reporting_fx_reference_year"] = FX_REFERENCE_YEAR
        converted["reporting_fx_series_key"] = FX_SERIES_KEY
    return converted


def add_cny_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Append explicitly labelled CNY equivalents back-converted from EUR."""
    converted = frame.copy()
    targets: list[tuple[str, str]] = []
    for column in frame.columns:
        if "_eur_" in column:
            targets.append(
                (
                    column,
                    column.replace(
                        "_eur_",
                        "_cny_equivalent_backconverted_from_eur_",
                    ),
                )
            )
        elif column.endswith("_eur"):
            targets.append(
                (
                    column,
                    f"{column[:-4]}_cny_equivalent_backconverted_from_eur",
                )
            )
    for source, target in targets:
        if target not in converted.columns:
            converted[target] = (
                pd.to_numeric(frame[source], errors="raise") * CNY_PER_EUR
            )
    if targets:
        converted["reporting_fx_cny_per_eur"] = CNY_PER_EUR
        converted["reporting_fx_reference_year"] = FX_REFERENCE_YEAR
        converted["reporting_fx_series_key"] = FX_SERIES_KEY
        converted["cny_equivalent_provenance"] = (
            "backconverted_from_source_aware_eur_not_native_cny"
        )
    return converted


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.3,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_runner():
    spec = importlib.util.spec_from_file_location("europe_runner_v39", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", axes):
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.018,
            pos.y1 + 0.008,
            label,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="bottom",
        )


def export(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    # Preserve the exact 183-mm canvas across every export.  Tight bounding
    # boxes made the v39 figures vary by almost an inch after export.
    fig.savefig(output / f"{stem}.pdf", bbox_inches=None)
    fig.savefig(output / f"{stem}.svg", bbox_inches=None)
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches=None)
    fig.savefig(
        output / f"{stem}.tiff",
        dpi=600,
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(output / f"{stem}.png") as image:
        image.convert("L").save(output / f"{stem}_grayscale.png", dpi=(300, 300))
    plt.close(fig)


def save_source(frame: pd.DataFrame, source_dir: Path, name: str) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source_dir / name, index=False)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def log_duration_axis(ax: plt.Axes, maximum: float = 720.0) -> None:
    ticks = [4, 8, 12, 24, 48, 96, 168]
    if maximum > 168:
        ticks += [336, 720]
    ax.set_xscale("log")
    ax.set_xticks([value for value in ticks if value <= maximum])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(3.6, maximum * 1.08)


def geometry_stats(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        panel.groupby(["country", "year", "source_day_id"], as_index=False)
        .agg(
            task_mwh=("task_c3_q75_mw", "sum"),
            recharge_mwh=("r2_potential_recharge_mw", "sum"),
            event=("event_flag_q75", "max"),
            segment=("season_segment", "first"),
        )
        .sort_values(["country", "year", "source_day_id"], kind="stable")
    )
    rows: list[dict[str, float | int | str]] = []
    for (country, year), group in daily.groupby(["country", "year"], sort=True):
        flag = group["event"].to_numpy(bool)
        segment = group["segment"].to_numpy(int)
        longest = 0
        run = 0
        adjacent = 0
        previous = False
        previous_segment = -1
        for current, current_segment in zip(flag, segment):
            if current_segment != previous_segment:
                run = 0
                previous = False
            if current:
                run += 1
                longest = max(longest, run)
                adjacent += int(previous)
            else:
                run = 0
            previous = bool(current)
            previous_segment = int(current_segment)
        total = float(group["task_mwh"].sum())
        event_energy = float(group.loc[group["event"].astype(bool), "task_mwh"].sum())
        rows.append(
            {
                "country": country,
                "year": int(year),
                "annual_task_twh": total / 1.0e6,
                "event_task_share_pct": 100.0 * event_energy / total if total else 0.0,
                "natural_adjacent_event_pairs": adjacent,
                "natural_longest_event_run_days": longest,
                "annual_recharge_twh": float(group["recharge_mwh"].sum()) / 1.0e6,
                "recharge_to_task_ratio": (
                    float(group["recharge_mwh"].sum()) / total if total else np.nan
                ),
            }
        )
    return daily, pd.DataFrame(rows)


def map_panel(
    ax: plt.Axes,
    geojson_path: Path,
    country_years: pd.Series,
) -> None:
    with geojson_path.open(encoding="utf-8") as handle:
        geo = json.load(handle)
    code_map = {"GB": "UK", "GR": "EL"}
    values = {code_map.get(code, code): value for code, value in country_years.items()}
    norm = mpl.colors.Normalize(vmin=5, vmax=10)
    cmap = mpl.colormaps["YlGnBu"]
    background: list[Polygon] = []
    selected: list[Polygon] = []
    selected_values: list[float] = []
    for feature in geo["features"]:
        code = feature["properties"].get("CNTR_ID")
        geometry = feature.get("geometry")
        if not geometry:
            continue
        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        for polygon in polygons:
            if not polygon:
                continue
            outer = np.asarray(polygon[0], dtype=float)
            if outer.ndim != 2 or outer.shape[1] != 2:
                continue
            if outer[:, 0].max() < -13 or outer[:, 0].min() > 32:
                continue
            if outer[:, 1].max() < 34 or outer[:, 1].min() > 72:
                continue
            patch = Polygon(outer, closed=True)
            if code in values:
                selected.append(patch)
                selected_values.append(values[code])
            else:
                background.append(patch)
    ax.add_collection(
        PatchCollection(
            background,
            facecolor="#F3F3F3",
            edgecolor="white",
            linewidth=0.25,
            zorder=1,
        )
    )
    collection = PatchCollection(
        selected,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    collection.set_array(np.asarray(selected_values))
    ax.add_collection(collection)
    ax.set_xlim(-12, 31)
    ax.set_ylim(35, 71.5)
    ax.set_aspect(1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    inset = ax.inset_axes([0.05, -0.02, 0.45, 0.045])
    mpl.colorbar.ColorbarBase(inset, cmap=cmap, norm=norm, orientation="horizontal")
    inset.set_xlabel("Observed years", labelpad=1)
    inset.tick_params(labelsize=5, length=1.5)


def make_soc_trace(
    task: np.ndarray,
    recharge: np.ndarray,
    duration_h: float,
    rte: float,
    self_discharge_pct_day: float,
    charge_power_ratio: float,
    pdis_mw: float,
) -> np.ndarray:
    energy = duration_h * pdis_mw
    e_min, e_max = 0.05 * energy, 0.95 * energy
    eta = math.sqrt(rte)
    pch = charge_power_ratio * pdis_mw
    retention = (1.0 - self_discharge_pct_day / 100.0) ** (1.0 / 24.0)
    soc = 0.5 * energy
    for _ in range(200):
        start = soc
        for demand, resource in zip(task, recharge):
            soc = max(e_min, soc * retention)
            if demand > 1.0e-12:
                discharge = min(pdis_mw, demand, max(0.0, (soc - e_min) * eta))
                soc -= discharge / eta
            elif resource > 1.0e-12:
                charge = min(pch, resource, max(0.0, (e_max - soc) / eta))
                soc += charge * eta
        if abs(soc - start) <= max(1.0e-9, 1.0e-6 * energy):
            break
    trace = np.empty(task.size + 1)
    trace[0] = soc / energy
    for index, (demand, resource) in enumerate(zip(task, recharge), start=1):
        soc = max(e_min, soc * retention)
        if demand > 1.0e-12:
            discharge = min(pdis_mw, demand, max(0.0, (soc - e_min) * eta))
            soc -= discharge / eta
        elif resource > 1.0e-12:
            charge = min(pch, resource, max(0.0, (e_max - soc) / eta))
            soc += charge * eta
        trace[index] = soc / energy
    return trace


def figure1(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    fig.subplots_adjust(left=0.018, right=0.985, bottom=0.035, top=0.975)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # A continuous, left-to-right evidence path replaces the v39 dashboard.
    ax.add_patch(
        FancyBboxPatch(
            (1.0, 10.5), 98.0, 48.5,
            boxstyle="round,pad=0.4,rounding_size=2.2",
            facecolor="#F7F9FA", edgecolor="#D9E0E4", linewidth=0.75,
        )
    )
    stages = [4.5, 25.0, 47.5, 70.0, 91.0]
    for x0, x1 in zip(stages[:-1], stages[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 13.5, 35), (x1 - 2.2, 35),
                arrowstyle="-|>", mutation_scale=10,
                color="#A8B2B8", linewidth=1.0,
            )
        )

    # Stage 1 | observed synchronous system.
    ax.text(11.0, 55.2, "Observed system", ha="center",
            fontsize=7.4, weight="bold", color=DARK)
    # stylized Europe network rather than a literal map
    nodes = np.array([
        [6.3, 43.0], [9.2, 47.3], [12.7, 45.1], [15.4, 49.2],
        [8.0, 38.4], [12.0, 39.2], [15.5, 41.0], [10.0, 34.0],
        [14.0, 34.8],
    ])
    edges = [(0,1),(1,2),(2,3),(0,4),(1,5),(2,5),(2,6),
             (4,5),(5,6),(4,7),(5,7),(5,8),(6,8),(7,8)]
    for i, j in edges:
        ax.plot(nodes[[i, j], 0], nodes[[i, j], 1],
                color="#AFC4CB", lw=0.8, zorder=1)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=18, color="#277DA1",
               edgecolor="white", linewidth=0.45, zorder=2)
    # demand, renewable and gas glyphs
    ax.plot([4.5, 17.5], [28.2, 28.2], color="#CAD2D7", lw=0.7)
    bx = np.arange(5.2, 17.0, 1.25)
    heights = np.array([2.0, 3.8, 1.8, 4.8, 3.0, 5.5, 2.4, 4.1, 2.7, 4.9])
    ax.bar(bx, heights, width=0.78, bottom=28.2, color="#E69F00",
           alpha=0.85, linewidth=0)
    ax.plot(bx, 30.0 + 1.15 * np.sin(np.linspace(0, 2.6*np.pi, len(bx))),
            color="#009E73", lw=1.05)
    ax.text(11.0, 21.8, "15 countries · 132 country-years",
            ha="center", fontsize=5.7, color="#4F5B62")
    ax.text(11.0, 18.9, "gas · load · wind · solar",
            ha="center", fontsize=5.5, color="#738087")

    # Stage 2 | same blocks, different chronology.
    ax.text(32.8, 55.2, "Paired chronology", ha="center",
            fontsize=7.4, weight="bold", color=DARK)
    block_h = np.array([2.4, 5.2, 3.1, 6.0, 4.2, 5.5, 2.8])
    for y0, color, label, order in [
        (43.8, DISPERSED, "dispersed", [3, 0, 5, 2, 6, 1, 4]),
        (31.3, CLUSTERED, "clustered", [0, 6, 2, 4, 3, 5, 1]),
    ]:
        xs = np.arange(25.4, 41.0, 2.25)
        ax.bar(xs, block_h[order], width=1.15, bottom=y0,
               color=color, alpha=0.88, linewidth=0)
        # green arcs indicate the time available for replenishment
        for x0 in xs[:-1]:
            ax.add_patch(
                mpl.patches.Arc(
                    (x0 + 1.12, y0 - 0.2), 2.0, 2.0,
                    theta1=12, theta2=168, color="#2A9D8F", lw=0.75,
                    alpha=0.9,
                )
            )
        ax.text(24.3, y0 + 2.6, label, color=color, ha="right",
                fontsize=5.8, weight="bold")
    ax.text(32.8, 22.0, "same 24-h blocks and annual energy",
            ha="center", fontsize=5.6, color="#4F5B62")
    ax.text(32.8, 19.0, "only event spacing changes",
            ha="center", fontsize=5.6, color="#738087")

    # Stage 3 | full technology-duration domain.
    ax.text(55.3, 55.2, "Technology spectrum", ha="center",
            fontsize=7.4, weight="bold", color=DARK)
    duration_x = np.linspace(47.3, 64.0, len(TECH_ORDER))
    duration_y = 28.0 + np.array([2.5, 4.2, 6.2, 7.7, 9.4, 11.4, 13.3, 17.0])
    for x0, y0, tech in zip(duration_x, duration_y, TECH_ORDER):
        ax.add_patch(
            FancyBboxPatch(
                (x0 - 1.05, y0 - 2.1), 2.1, 4.2,
                boxstyle="round,pad=0.10,rounding_size=0.35",
                facecolor="white", edgecolor=TECH_COLOR[tech], linewidth=0.9,
            )
        )
        fill = 0.30 + 0.55 * (y0 - duration_y.min()) / (
            duration_y.max() - duration_y.min()
        )
        ax.add_patch(
            Rectangle(
                (x0 - 0.66, y0 - 1.35), 1.32, 2.70 * fill,
                facecolor=TECH_COLOR[tech], edgecolor="none", alpha=0.88,
            )
        )
    ax.annotate(
        "", xy=(64.7, 24.8), xytext=(46.2, 24.8),
        arrowprops=dict(arrowstyle="-|>", color="#7E8A90", lw=0.8),
    )
    ax.text(46.3, 22.0, "4 h", fontsize=5.4, color="#66737A")
    ax.text(64.5, 22.0, "720 h", fontsize=5.4, ha="right", color="#66737A")
    ax.text(55.3, 18.9, "53 feasible configurations",
            ha="center", fontsize=5.6, color="#4F5B62")

    # Stage 4 | mechanistic accounting.
    ax.text(75.2, 55.2, "Physical response", ha="center",
            fontsize=7.4, weight="bold", color=DARK)
    ax.add_patch(
        FancyBboxPatch(
            (68.6, 38.3), 5.0, 10.0,
            boxstyle="round,pad=0.25,rounding_size=0.8",
            facecolor="#E8F2F7", edgecolor=TECH_COLOR["li_lfp"], linewidth=1.0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (77.2, 38.3), 5.0, 10.0,
            boxstyle="round,pad=0.25,rounding_size=0.8",
            facecolor="#E7F4EF", edgecolor=TECH_COLOR["caes"], linewidth=1.0,
        )
    )
    for x0, color, low in [
        (69.2, TECH_COLOR["li_lfp"], False),
        (77.8, TECH_COLOR["caes"], True),
    ]:
        xs = np.linspace(x0, x0 + 3.8, 70)
        curve = 41.4 + 3.2 * np.sin(np.linspace(0, 2.2*np.pi, 70))
        if low:
            curve -= np.linspace(0, 2.0, 70)
        ax.plot(xs, curve, color=color, lw=1.05, clip_on=True)
    ax.text(71.1, 35.2, "cycles", ha="center", color=TECH_COLOR["li_lfp"],
            fontsize=5.6, weight="bold")
    ax.text(79.7, 35.2, "firm supply", ha="center", color=TECH_COLOR["caes"],
            fontsize=5.6, weight="bold")
    ax.plot([70.7, 80.0], [29.3, 29.3], color="#C3CBD0", lw=0.8)
    for x0, label, color in [
        (70.7, "EFC", DISPERSED), (75.35, "EENS", CLUSTERED),
        (80.0, "€", TECH_COLOR["h2_cavern"]),
    ]:
        ax.scatter([x0], [29.3], s=74, color=color, edgecolor="white",
                   linewidth=0.7, zorder=3)
        ax.text(x0, 29.3, label, ha="center", va="center", fontsize=5.0,
                color="white", weight="bold", zorder=4)
    ax.text(75.35, 19.0, "cycles · firm supply · net value",
            ha="center", fontsize=5.5, color="#4F5B62")

    # Stage 5 | conditional economic frontier.
    ax.text(91.0, 55.2, "Value frontier", ha="center",
            fontsize=7.4, weight="bold", color=DARK)
    xx = np.linspace(85.0, 97.2, 160)
    dispersed = 41.5 - 0.40 * (xx - 85.0)
    clustered = dispersed - 3.6 - 2.6 * np.exp(-0.5*((xx-91.0)/1.65)**2)
    inventory = 28.3 + 1.12 * np.maximum(xx - 91.0, 0)
    ax.plot(xx, dispersed, color=DISPERSED, lw=1.45)
    ax.plot(xx, clustered, color=CLUSTERED, lw=1.45)
    ax.plot(xx, inventory, color=TECH_COLOR["h2_cavern"], lw=1.45)
    ax.axhline(31.0, xmin=0.855, xmax=0.977, color="#78848A", lw=0.55)
    valley = (xx >= 89.0) & (xx <= 92.7)
    ax.fill_between(
        xx, 31.0, clustered, where=valley & (clustered < 31.0),
        color="#7E57A5", alpha=0.22, linewidth=0,
    )
    ax.text(90.8, 27.2, "mid-duration\nvalley", ha="center",
            va="center", color="#6A51A3", fontsize=5.6, weight="bold")
    ax.text(84.8, 19.0, "short", fontsize=5.3, color="#66737A")
    ax.text(97.3, 19.0, "long duration", fontsize=5.3, ha="right",
            color="#66737A")

    # Compact evidence strip anchors the schematic in the two empirical tests.
    ax.plot([2.8, 97.2], [14.4, 14.4], color="#D9E0E4", lw=0.65)
    evidence = [
        (13.0, "#277DA1", "Europe", "1,156,320 country-hours"),
        (36.0, DISPERSED, "Paired test", "annual energy conserved"),
        (60.0, CLUSTERED, "Outcomes", "cycles · EENS · net value"),
        (84.0, "#C44E52", "External test", "26 Chinese provinces"),
    ]
    for x0, color, top, bottom in evidence:
        ax.scatter([x0 - 5.0], [7.2], s=75, color=color, edgecolor="white",
                   linewidth=0.7)
        ax.text(x0 - 2.8, 8.4, top, fontsize=5.8, weight="bold",
                color=DARK, va="center")
        ax.text(x0 - 2.8, 5.8, bottom, fontsize=5.2, color="#66737A",
                va="center")
    export(fig, output, "fig_v40_figure1_framework")


def figure2(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    stats: pd.DataFrame,
    geojson: Path,
    runner,
    output: Path,
    source_dir: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 5.55))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.965)
    gs = fig.add_gridspec(
        3, 6, height_ratios=[1.0, 1.0, 0.92],
        hspace=0.54, wspace=0.78,
    )
    axa = fig.add_subplot(gs[0:2, 0:3])
    axb = fig.add_subplot(gs[0, 3:6])
    axc = fig.add_subplot(gs[1, 3:6])
    axd = fig.add_subplot(gs[2, 0:2])
    axe = fig.add_subplot(gs[2, 2:4])
    axf = fig.add_subplot(gs[2, 4:6])
    axes = [axa, axb, axc, axd, axe, axf]
    coverage = panel[["country", "year"]].drop_duplicates()
    year_count = coverage.groupby("country").size()
    map_panel(axa, geojson, year_count)
    axa.text(
        0.03, 0.98, "15 countries\n132 country-years",
        transform=axa.transAxes, va="top", ha="left",
        fontsize=7.0, weight="bold", color=DARK,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white",
            edgecolor="#D5DDE1", alpha=0.92,
        ),
    )

    matrix = (
        coverage.assign(value=1)
        .pivot(index="country", columns="year", values="value")
        .reindex(sorted(coverage["country"].unique(), reverse=True))
    )
    for row, country in enumerate(matrix.index):
        years = matrix.columns[matrix.loc[country].notna()].to_numpy()
        if years.size:
            axb.plot(
                [years.min(), years.max()], [row, row],
                color="#D6DDE1", lw=1.2, zorder=1,
            )
            axb.scatter(
                years, np.full(years.size, row), s=15,
                color="#2A9D8F", edgecolor="white",
                linewidth=0.35, zorder=2,
            )
    axb.set_xticks(np.arange(2015, 2025))
    axb.set_xticklabels([str(y)[-2:] for y in range(2015, 2025)])
    axb.set_yticks(range(matrix.shape[0]))
    axb.set_yticklabels(matrix.index, fontsize=5.4)
    axb.set_xlim(2014.6, 2024.4)
    axb.set_xlabel("Observed year")
    axb.set_ylabel("")
    axb.grid(axis="x", color="#EDF0F2", lw=0.5, zorder=0)
    axb.spines["left"].set_visible(False)
    axb.tick_params(axis="y", length=0)

    example = panel[(panel["country"].eq("DE")) & (panel["year"].eq(2023))].sort_values("source_hour_id")
    orders, _ = runner.day_orders(example)
    base = example["task_c3_q75_mw"].to_numpy(float).reshape(365, 24).sum(axis=1)
    recharge = (
        example["r2_potential_recharge_mw"].to_numpy(float)
        .reshape(365, 24).sum(axis=1)
    )
    base_scale = max(base.max(), 1.0)
    recharge_scale = max(recharge.max(), 1.0)
    chronology_source: dict[str, np.ndarray] = {"day": np.arange(365)}
    for offset, (name, color) in enumerate(
        [("dispersed", DISPERSED), ("clustered", CLUSTERED)]
    ):
        task_values = base[np.asarray(orders[name])] / base_scale
        recharge_values = recharge[np.asarray(orders[name])] / recharge_scale
        x = np.arange(365)
        midline = 1.35 - offset * 1.45
        axc.fill_between(
            x, midline, midline + 0.72 * task_values,
            color=color, linewidth=0, alpha=0.88,
        )
        axc.fill_between(
            x, midline, midline - 0.45 * recharge_values,
            color="#2A9D8F", linewidth=0, alpha=0.32,
        )
        axc.axhline(midline, color="#AEB8BD", lw=0.5)
        axc.text(
            360, midline + 0.50, name.capitalize(), ha="right",
            color=color, weight="bold", fontsize=5.6,
        )
        chronology_source[f"{name}_task_normalized"] = task_values
        chronology_source[f"{name}_recharge_normalized"] = recharge_values
    axc.set_xlim(0, 365)
    axc.set_ylim(-0.58, 2.18)
    axc.set_yticks([])
    axc.set_xlabel("Reordered day in the same German year")
    axc.set_ylabel("task     recharge", fontsize=5.6)
    axc.spines["left"].set_visible(False)
    axc.spines["bottom"].set_color("#CBD2D6")

    longest = stats["natural_longest_event_run_days"].to_numpy(float)
    rng = np.random.default_rng(40)
    axd.scatter(
        longest, rng.normal(0.18, 0.035, len(longest)),
        s=7, color="#4C78A8", alpha=0.22, linewidth=0,
    )
    violin = axd.violinplot(
        longest, positions=[0.52], vert=False, widths=0.34,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("none")
        body.set_alpha(0.30)
    q1, med, q3 = np.quantile(longest, [0.25, 0.5, 0.75])
    axd.plot(
        [q1, q3], [0.52, 0.52], color=DARK, lw=3.2,
        solid_capstyle="round",
    )
    axd.scatter(
        [med], [0.52], s=18, facecolor="white",
        edgecolor=DARK, linewidth=0.7, zorder=3,
    )
    axd.set_xlabel("Longest natural event run (d)")
    axd.set_yticks([])
    axd.set_ylim(0.05, 0.72)
    axd.spines["left"].set_visible(False)

    rho = stats["annual_task_twh"].corr(stats["natural_longest_event_run_days"], method="spearman")
    axe.scatter(
        stats["annual_task_twh"],
        stats["natural_longest_event_run_days"],
        color="#2A9D8F",
        s=14,
        alpha=0.65,
        linewidth=0.2,
        edgecolor="white",
    )
    axe.set_xscale("log")
    axe.set_xlabel("Annual peaking task (TWh)")
    axe.set_ylabel("Longest natural event run (d)")
    axe.text(
        0.05, 0.93, f"Spearman $\\rho$ = {rho:.2f}",
        transform=axe.transAxes, va="top", fontsize=5.7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
    )

    x, y = ecdf(stats["recharge_to_task_ratio"].to_numpy())
    axf.fill_between(x, 0, y, color="#2A9D8F", alpha=0.13)
    axf.plot(x, y, color="#2A9D8F", lw=1.6)
    median = float(stats["recharge_to_task_ratio"].median())
    q25, q75 = stats["recharge_to_task_ratio"].quantile([0.25, 0.75])
    axf.axvspan(q25, q75, color="#2A9D8F", alpha=0.10)
    axf.axvline(median, color=DARK, ls="--", lw=0.8)
    axf.text(
        median * 1.08, 0.10, f"median {median:.1f}",
        rotation=90, va="bottom", fontsize=5.4,
    )
    axf.set_xscale("log")
    axf.set_xlabel("Annual potential recharge / peaking task")
    axf.set_ylabel("Cumulative share of country-years")
    axf.set_ylim(0, 1)
    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_figure2_europe_data_geometry")
    save_source(coverage, source_dir, "figure2_panel_ab_country_year_coverage.csv")
    save_source(stats, source_dir, "figure2_panel_def_event_geometry.csv")
    save_source(
        pd.DataFrame(chronology_source),
        source_dir,
        "figure2_panel_c_germany_2023_chronologies.csv",
    )


def figure3(
    results: pd.DataFrame,
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    source_dir: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 6.05))
    fig.subplots_adjust(left=0.120, right=0.985, bottom=0.085, top=0.965)
    gs = fig.add_gridspec(
        3, 6, height_ratios=[1.08, 0.92, 0.98],
        hspace=0.50, wspace=0.80,
    )
    axa = fig.add_subplot(gs[0, 0:4])
    axb = fig.add_subplot(gs[0, 4:6])
    axc = fig.add_subplot(gs[1, 0:3])
    axd = fig.add_subplot(gs[1, 3:6])
    axe = fig.add_subplot(gs[2, 0:3])
    axf = fig.add_subplot(gs[2, 3:6])
    axes = [axa, axb, axc, axd, axe, axf]

    catalog = (
        results.sort_values(["duration_h", "tech_id"])
        .drop_duplicates("config_id")
        [[
            "config_id", "tech_id", "duration_h",
            "released_library_configuration", "direct_duration_anchor",
        ]]
    )
    for tech in TECH_ORDER:
        group = summary[summary["tech_id"].eq(tech)].sort_values("duration_h")
        if group.empty:
            continue
        axa.fill_between(
            group["duration_h"],
            group["retained_equivalent_full_cycles_ci_low"],
            group["retained_equivalent_full_cycles_ci_high"],
            color=TECH_COLOR[tech], alpha=0.10, linewidth=0,
        )
        axa.plot(
            group["duration_h"],
            group["retained_equivalent_full_cycles_mean"],
            color=TECH_COLOR[tech], marker="o", ms=2.7,
            label=TECH_LABEL[tech],
        )
    axa.axhline(1, color="#8A9499", ls="--", lw=0.65)
    axa.set_ylabel("Clustered/dispersed EFC cycles")
    axa.set_xlabel("Discharge duration (h)")
    axa.set_ylim(0, 1.04)
    log_duration_axis(axa)
    axa.grid(axis="x", color="#ECEFF1", lw=0.45)
    axa.legend(
        loc="lower right", ncol=4, handlelength=1.3,
        columnspacing=0.9, fontsize=5.2,
    )

    prevalence = (
        paired.assign(
            cycle_loss=lambda x: x["delta_equivalent_full_cycles"].lt(0)
        )
        .groupby(["tech_id", "duration_h"], as_index=False)["cycle_loss"]
        .mean()
    )
    for tech in TECH_ORDER:
        group = prevalence[prevalence["tech_id"].eq(tech)].sort_values("duration_h")
        if group.empty:
            continue
        axb.plot(
            group["duration_h"], 100 * group["cycle_loss"],
            color=TECH_COLOR[tech], marker="o", ms=2.5,
        )
    axb.set_ylim(0, 103)
    axb.set_ylabel("Country-years with\nfewer cycles (%)")
    axb.set_xlabel("Duration (h)")
    log_duration_axis(axb)
    axb.grid(axis="x", color="#ECEFF1", lw=0.45)

    selected = paired[paired["config_id"].isin(SELECTED)].copy()
    selected["cost_snapshot_year"] = 2035
    positions = np.arange(len(SELECTED))
    rng = np.random.default_rng(20260727)
    for index, config in enumerate(SELECTED):
        current = selected[selected["config_id"].eq(config)]
        values = current["retained_equivalent_full_cycles"].to_numpy()
        color = TECH_COLOR[current["tech_id"].iloc[0]]
        parts = axc.violinplot(
            values, positions=[index], widths=0.70,
            showextrema=False, showmeans=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.25)
        axc.scatter(
            rng.normal(index, 0.055, size=len(values)), values,
            s=5, alpha=0.13, color=color, linewidth=0,
        )
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axc.plot(
            [index, index], [q1, q3], color=DARK, lw=3.0,
            solid_capstyle="round",
        )
        axc.scatter(
            index, med, s=20, facecolor="white",
            edgecolor=DARK, zorder=3,
        )
    axc.axhline(1, color="#888888", ls="--", lw=0.7)
    axc.set_xticks(positions)
    axc.set_xticklabels(
        [SELECTED_LABEL[c] for c in SELECTED], rotation=18, ha="right",
    )
    axc.set_ylabel("EFC-cycle retention")
    axc.set_ylim(0, 1.05)

    mean_cycles = (
        results[
            results["chronology"].isin(["dispersed", "clustered"])
            & results["config_id"].isin(SELECTED)
        ]
        .groupby(["config_id", "chronology"], as_index=False)[
            "equivalent_full_cycles"
        ]
        .mean()
    )
    for config in SELECTED:
        group = mean_cycles[
            mean_cycles["config_id"].eq(config)
        ].set_index("chronology")
        color = TECH_COLOR[
            selected.loc[selected["config_id"].eq(config), "tech_id"].iloc[0]
        ]
        axd.plot(
            [0, 1],
            [
                group.loc["dispersed", "equivalent_full_cycles"],
                group.loc["clustered", "equivalent_full_cycles"],
            ],
            color=color, marker="o", label=SELECTED_LABEL[config],
        )
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["Dispersed", "Clustered"])
    axd.set_xlim(-0.10, 1.10)
    axd.set_ylabel("Equivalent full cycles (yr$^{-1}$)")
    axd.legend(
        loc="upper right", ncol=2, handlelength=1.3,
        columnspacing=0.8, fontsize=5.2,
    )

    metric_rows: list[dict[str, float | str]] = []
    for config in SELECTED:
        current = selected[selected["config_id"].eq(config)]
        for metric, label, scale in [
            ("delta_served_share_pp", "Task service", 1.0),
            ("delta_efc_ratio", "Task-firm supply", 100.0),
        ]:
            values = scale * current[metric].to_numpy(float)
            metric_rows.append(
                {
                    "config_id": config, "metric": label,
                    "mean": values.mean(),
                    "low": np.quantile(values, 0.025),
                    "high": np.quantile(values, 0.975),
                }
            )
    metric_summary = pd.DataFrame(metric_rows)
    y_positions = np.arange(len(SELECTED))
    for offset, metric, marker in [
        (-0.10, "Task service", "o"),
        (0.10, "Task-firm supply", "s"),
    ]:
        group = (
            metric_summary[metric_summary["metric"].eq(metric)]
            .set_index("config_id").loc[SELECTED]
        )
        axe.errorbar(
            group["mean"], y_positions + offset,
            xerr=[
                group["mean"] - group["low"],
                group["high"] - group["mean"],
            ],
            fmt=marker, color=DISPERSED if metric == "Task service" else CLUSTERED,
            capsize=1.8, ms=3.7, label=metric,
        )
    axe.axvline(0, color="#777777", lw=0.7)
    axe.set_yticks(y_positions)
    axe.set_yticklabels([SELECTED_LABEL[c] for c in SELECTED])
    axe.invert_yaxis()
    axe.set_xlabel("Clustered minus dispersed effect (pp)")
    axe.legend(loc="lower left", ncol=2, fontsize=5.2)

    economic_summary = (
        paired.groupby(["config_id", "tech_id", "duration_h"], as_index=False)
        .agg(
            mean=("delta_joint_net_benefit_eur_per_kw_year", "mean"),
            low=(
                "delta_joint_net_benefit_eur_per_kw_year",
                lambda x: x.quantile(0.025),
            ),
            high=(
                "delta_joint_net_benefit_eur_per_kw_year",
                lambda x: x.quantile(0.975),
            ),
        )
        .assign(cost_snapshot_year=2035)
    )
    for tech in TECH_ORDER:
        group = economic_summary[
            economic_summary["tech_id"].eq(tech)
        ].sort_values("duration_h")
        if group.empty:
            continue
        axf.fill_between(
            group["duration_h"], group["low"], group["high"],
            color=TECH_COLOR[tech], alpha=0.08, linewidth=0,
        )
        axf.plot(
            group["duration_h"], group["mean"],
            color=TECH_COLOR[tech], marker="o", ms=2.5,
        )
    axf.axhline(0, color="#777777", lw=0.7)
    axf.set_xlabel("Discharge duration (h)")
    axf.set_ylabel("$\\Delta$ annual net benefit\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axf.set_title("2035 cost snapshot", loc="left", color="#66737A")
    log_duration_axis(axf)
    axf.grid(axis="x", color="#ECEFF1", lw=0.45)
    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_figure3_duration_spectrum")
    save_source(catalog, source_dir, "figure3_panel_a_configuration_catalog.csv")
    save_source(
        summary.assign(cost_snapshot_year=2035),
        source_dir,
        "figure3_panel_a_configuration_summary.csv",
    )
    save_source(prevalence, source_dir, "figure3_panel_b_cycle_loss_prevalence.csv")
    save_source(selected, source_dir, "figure3_panel_c_selected_configuration_changes.csv")
    save_source(mean_cycles, source_dir, "figure3_panel_d_mean_cycles.csv")
    save_source(metric_summary, source_dir, "figure3_panel_e_physical_effects.csv")
    save_source(economic_summary, source_dir, "figure3_panel_f_net_benefit.csv")


def figure4(
    panel: pd.DataFrame,
    results: pd.DataFrame,
    paired: pd.DataFrame,
    runner,
    output: Path,
    source_dir: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 6.10))
    fig.subplots_adjust(left=0.120, right=0.985, bottom=0.085, top=0.965)
    gs = fig.add_gridspec(
        3, 6, height_ratios=[1.05, 0.95, 1.05],
        hspace=0.50, wspace=0.80,
    )
    axa = fig.add_subplot(gs[0, 0:3])
    axb = fig.add_subplot(gs[0, 3:6])
    axc = fig.add_subplot(gs[1, 0:3])
    axd = fig.add_subplot(gs[1, 3:6])
    axe = fig.add_subplot(gs[2, 0:3])
    axf = fig.add_subplot(gs[2, 3:6])
    axes = [axa, axb, axc, axd, axe, axf]
    example = panel[(panel["country"].eq("DE")) & (panel["year"].eq(2023))].sort_values("source_hour_id")
    orders, _ = runner.day_orders(example)
    task0 = example["task_c3_q75_mw"].to_numpy(float)
    recharge0 = example["r2_potential_recharge_mw"].to_numpy(float)
    cfg = results.drop_duplicates("config_id").set_index("config_id")
    pdis = float(example["forced_discharge_power_mw"].iloc[0])
    traces: list[pd.DataFrame] = []
    daily_clustered_task = (
        runner.reorder_days(task0, orders["clustered"]).reshape(365, 24).sum(axis=1)
    )
    window_days = 45
    rolling = np.convolve(
        daily_clustered_task, np.ones(window_days), mode="valid"
    )
    start = int(np.argmax(rolling))
    stop = start + window_days

    task_clustered = runner.reorder_days(task0, orders["clustered"])
    recharge_clustered = runner.reorder_days(recharge0, orders["clustered"])
    daily_task = task_clustered.reshape(365, 24).sum(axis=1)
    daily_recharge = recharge_clustered.reshape(365, 24).sum(axis=1)
    task_scale = max(daily_task[start:stop].max(), 1.0)
    recharge_scale = max(daily_recharge[start:stop].max(), 1.0)
    days = np.arange(window_days)
    axa.bar(
        days, daily_task[start:stop] / task_scale,
        width=0.82, color=CLUSTERED, alpha=0.82, label="Peaking task",
    )
    axa.fill_between(
        days, 0, -0.62 * daily_recharge[start:stop] / recharge_scale,
        color="#2A9D8F", alpha=0.34, linewidth=0, label="Potential recharge",
    )
    axa.axhline(0, color="#AEB7BC", lw=0.6)
    axa.set_xlim(-0.8, window_days - 0.2)
    axa.set_xlabel("Day within the highest-task 45-d clustered window")
    axa.set_ylabel("Normalized task / recharge")
    axa.set_yticks([-0.6, 0, 0.5, 1.0])
    axa.set_yticklabels(["0.6", "0", "0.5", "1.0"])
    axa.legend(loc="upper left", ncol=2, fontsize=5.3)

    soc_window_source: list[pd.DataFrame] = []
    for config in ["li_lfp_4h", "caes_24h"]:
        row = cfg.loc[config]
        for chronology, color in [("dispersed", DISPERSED), ("clustered", CLUSTERED)]:
            task = runner.reorder_days(task0, orders[chronology])
            recharge = runner.reorder_days(recharge0, orders[chronology])
            trace = make_soc_trace(
                task,
                recharge,
                float(row["duration_h"]),
                float(row["rte"]),
                float(row["self_discharge_pct_day"]),
                float(row["charge_power_ratio"]),
                pdis,
            )
            daily_min = pd.Series(trace[:-1]).groupby(np.arange(8760) // 24).min().to_numpy()
            daily_max = pd.Series(trace[:-1]).groupby(np.arange(8760) // 24).max().to_numpy()
            midpoint = 0.5 * (daily_min + daily_max)
            tech_color = (
                TECH_COLOR["li_lfp"] if config == "li_lfp_4h"
                else TECH_COLOR["caes"]
            )
            linestyle = "--" if chronology == "dispersed" else "-"
            axb.plot(
                days, midpoint[start:stop], color=tech_color,
                ls=linestyle, lw=1.15,
                label=f"{SELECTED_LABEL[config]} · {chronology}",
            )
            if chronology == "clustered":
                axb.fill_between(
                    days, daily_min[start:stop], daily_max[start:stop],
                    color=tech_color, alpha=0.10, linewidth=0,
                )
            traces.append(
                pd.DataFrame(
                    {
                        "config_id": config,
                        "chronology": chronology,
                        "day": np.arange(365),
                        "soc_daily_min": daily_min,
                        "soc_daily_max": daily_max,
                    }
                )
            )
            soc_window_source.append(
                pd.DataFrame(
                    {
                        "config_id": config,
                        "chronology": chronology,
                        "window_day": days,
                        "soc_daily_min": daily_min[start:stop],
                        "soc_daily_mid": midpoint[start:stop],
                        "soc_daily_max": daily_max[start:stop],
                    }
                )
            )
    axb.set_ylim(0, 1)
    axb.set_xlabel("Day within the same 45-d window position")
    axb.set_ylabel("State of charge")
    axb.legend(loc="lower left", ncol=2, fontsize=5.0, columnspacing=0.7)

    selected_results = results[
        results["config_id"].isin(["li_lfp_4h", "caes_24h"])
        & results["chronology"].isin(["dispersed", "clustered"])
    ]
    retention_wide = selected_results.pivot_table(
        index=["country", "year", "config_id", "tech_id"],
        columns="chronology",
        values=["charge_mwh", "equivalent_full_cycles", "served_share", "efc_ratio"],
    ).reset_index()
    retention_wide.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple) else str(col)
        for col in retention_wide.columns
    ]
    metric_map = {
        "charge_mwh": "Charge throughput",
        "equivalent_full_cycles": "Equivalent cycles",
        "served_share": "Task service",
        "efc_ratio": "Task-firm supply",
    }
    retention_rows: list[dict[str, float | str]] = []
    for _, row in retention_wide.iterrows():
        for metric, label in metric_map.items():
            denominator = float(row[f"{metric}_dispersed"])
            ratio = (
                100 * float(row[f"{metric}_clustered"]) / denominator
                if denominator > 0 else np.nan
            )
            retention_rows.append(
                {
                    "country": row["country"],
                    "year": int(row["year"]),
                    "config_id": row["config_id"],
                    "metric": label,
                    "retention_pct": ratio,
                }
            )
    retention_long = pd.DataFrame(retention_rows).dropna()
    retention_summary = (
        retention_long.groupby(["config_id", "metric"], as_index=False)
        .agg(
            mean=("retention_pct", "mean"),
            q025=("retention_pct", lambda x: x.quantile(0.025)),
            q975=("retention_pct", lambda x: x.quantile(0.975)),
        )
    )
    metric_order = list(metric_map.values())
    y = np.arange(len(metric_order))
    for offset, config in [(-0.10, "li_lfp_4h"), (0.10, "caes_24h")]:
        group = (
            retention_summary[retention_summary["config_id"].eq(config)]
            .set_index("metric").loc[metric_order]
        )
        color = TECH_COLOR["li_lfp"] if config.startswith("li_lfp") else TECH_COLOR["caes"]
        axc.errorbar(
            group["mean"], y + offset,
            xerr=[
                group["mean"] - group["q025"],
                group["q975"] - group["mean"],
            ],
            fmt="o",
            color=color,
            capsize=2,
            label=SELECTED_LABEL[config],
        )
    axc.axvline(100, color="#777777", ls="--", lw=0.7)
    axc.set_yticks(y)
    axc.set_yticklabels(metric_order)
    axc.invert_yaxis()
    axc.set_xlabel("Clustered/dispersed retention (%)")
    axc.legend(loc="lower right")

    physical = results[results["chronology"].isin(["dispersed", "clustered"])]
    wide = physical.pivot_table(
        index=["country", "year", "config_id", "tech_id", "duration_h"],
        columns="chronology",
        values=["charge_saturation_hours", "equivalent_full_cycles"],
    ).reset_index()
    wide["delta_charge_saturation_h"] = wide[("charge_saturation_hours", "clustered")] - wide[("charge_saturation_hours", "dispersed")]
    wide["delta_cycles"] = wide[("equivalent_full_cycles", "clustered")] - wide[("equivalent_full_cycles", "dispersed")]
    for tech in TECH_ORDER:
        current = wide[wide[("tech_id", "")].eq(tech)]
        if current.empty:
            continue
        axd.scatter(
            current["delta_charge_saturation_h"],
            current["delta_cycles"],
            s=6, alpha=0.10, linewidth=0, color=TECH_COLOR[tech],
        )
        axd.scatter(
            current["delta_charge_saturation_h"].median(),
            current["delta_cycles"].median(),
            s=24, marker="D", color=TECH_COLOR[tech],
            edgecolor="white", linewidth=0.45, label=TECH_LABEL[tech],
        )
    axd.axhline(0, color="#777777", lw=0.6)
    axd.set_xlabel("$\\Delta$ charge-power saturation (h yr$^{-1}$)")
    axd.set_ylabel("$\\Delta$ equivalent full cycles")
    axd.legend(
        loc="lower right", ncol=2, fontsize=5.0,
        handletextpad=0.3, columnspacing=0.7,
    )

    rel = (
        results[
            results["config_id"].isin(["li_lfp_4h", "caes_24h"])
            & results["chronology"].isin(["dispersed", "clustered"])
        ]
        .pivot_table(
            index=["country", "year", "chronology"],
            columns="config_id",
            values="joint_net_benefit_eur_per_kw_year",
        )
        .reset_index()
    )
    rel["caes_minus_lfp_eur_per_kw_year"] = rel["caes_24h"] - rel["li_lfp_4h"]
    country_rel = rel.groupby(["country", "chronology"], as_index=False)[
        "caes_minus_lfp_eur_per_kw_year"
    ].mean()
    country_wide = country_rel.pivot(
        index="country",
        columns="chronology",
        values="caes_minus_lfp_eur_per_kw_year",
    )
    country_wide["delta"] = country_wide["clustered"] - country_wide["dispersed"]
    country_wide = country_wide.sort_values("delta")
    for index, (country, row) in enumerate(country_wide.iterrows()):
        axe.plot([row["dispersed"], row["clustered"]], [index, index], color="#B0B0B0", lw=0.8)
        axe.scatter(row["dispersed"], index, color=DISPERSED, s=15)
        axe.scatter(row["clustered"], index, color=CLUSTERED, s=15)
    axe.axvline(0, color="#777777", lw=0.7)
    axe.set_yticks(range(len(country_wide)))
    axe.set_yticklabels(country_wide.index)
    axe.set_xlabel("A-CAES24 minus LFP4 joint net benefit\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axe.set_title("2035 cost snapshot", loc="left", color="#66737A")
    axe.scatter([], [], color=DISPERSED, label="Dispersed")
    axe.scatter([], [], color=CLUSTERED, label="Clustered")
    axe.legend(loc="lower left")

    rel_wide = rel.pivot_table(
        index=["country", "year"],
        columns="chronology",
        values="caes_minus_lfp_eur_per_kw_year",
    )
    shift = rel_wide["clustered"] - rel_wide["dispersed"]
    charge = selected_results.pivot_table(
        index=["country", "year", "config_id"],
        columns="chronology", values="charge_mwh",
    )
    charge["retention"] = charge["clustered"] / charge["dispersed"]
    charge_gap = charge["retention"].unstack("config_id")
    charge_gap["caes_minus_lfp_charge_retention"] = (
        charge_gap["caes_24h"] - charge_gap["li_lfp_4h"]
    )
    mechanism = (
        charge_gap[["caes_minus_lfp_charge_retention"]]
        .join(shift.rename("relative_value_shift_eur_per_kw_year"))
        .dropna().reset_index()
    )
    axf.scatter(
        100 * mechanism["caes_minus_lfp_charge_retention"],
        mechanism["relative_value_shift_eur_per_kw_year"],
        s=12, alpha=0.42, color=TECH_COLOR["caes"],
        edgecolor="white", linewidth=0.3,
    )
    coefficient = np.polyfit(
        100 * mechanism["caes_minus_lfp_charge_retention"],
        mechanism["relative_value_shift_eur_per_kw_year"], 1,
    )
    xx = np.linspace(
        100 * mechanism["caes_minus_lfp_charge_retention"].min(),
        100 * mechanism["caes_minus_lfp_charge_retention"].max(), 100,
    )
    axf.plot(xx, np.polyval(coefficient, xx), color=DARK, lw=0.9)
    rho = mechanism["caes_minus_lfp_charge_retention"].corr(
        mechanism["relative_value_shift_eur_per_kw_year"], method="spearman",
    )
    axf.text(
        0.04, 0.95, f"Spearman $\\rho$ = {rho:.2f}",
        transform=axf.transAxes, va="top", fontsize=5.5,
    )
    axf.axhline(0, color="#777777", lw=0.6)
    axf.set_xlabel("A-CAES24 minus LFP4\ncharge-retention gap (pp)")
    axf.set_ylabel("Relative annual-value shift\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axf.set_title("2035 cost snapshot", loc="left", color="#66737A")
    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_figure4_replenishment_mechanism")
    save_source(pd.concat(traces, ignore_index=True), source_dir, "figure4_panel_ab_soc_trace.csv")
    save_source(
        pd.concat(soc_window_source, ignore_index=True),
        source_dir, "figure4_panel_b_soc_window.csv",
    )
    save_source(retention_long, source_dir, "figure4_panel_c_retention_metrics.csv")
    save_source(retention_summary, source_dir, "figure4_panel_c_retention_summary.csv")
    wide_flat = wide.copy()
    wide_flat.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple) else str(col)
        for col in wide_flat.columns
    ]
    save_source(wide_flat, source_dir, "figure4_panel_d_charge_saturation_cycles.csv")
    country_value_source = country_wide.reset_index().rename(
        columns={
            "dispersed": "dispersed_caes_minus_lfp_eur_per_kw_year",
            "clustered": "clustered_caes_minus_lfp_eur_per_kw_year",
            "delta": "delta_caes_minus_lfp_eur_per_kw_year",
        }
    )
    country_value_source["cost_snapshot_year"] = 2035
    country_year_source = rel_wide.reset_index().rename(
        columns={
            "dispersed": "dispersed_caes_minus_lfp_eur_per_kw_year",
            "clustered": "clustered_caes_minus_lfp_eur_per_kw_year",
        }
    )
    country_year_source["delta_caes_minus_lfp_eur_per_kw_year"] = shift.to_numpy()
    country_year_source["cost_snapshot_year"] = 2035
    mechanism["cost_snapshot_year"] = 2035
    save_source(
        add_cny_columns(country_value_source),
        source_dir,
        "figure4_panel_e_country_relative_value.csv",
    )
    save_source(
        add_cny_columns(country_year_source),
        source_dir,
        "figure4_panel_f_country_year_relative_shift.csv",
    )
    save_source(
        mechanism, source_dir,
        "figure4_panel_f_charge_retention_value_link.csv",
    )


def gap_widths(envelope: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for keys, group in envelope.groupby(["scenario_id", "country", "year", "chronology"], sort=False):
        ordered = group.sort_values("duration_h", kind="stable")
        durations = ordered["duration_h"].to_numpy(float)
        positive = ordered["net_benefit_eur_per_kw_year"].to_numpy(float) > 0
        widths: list[float] = []
        for position in range(1, len(positive) - 1):
            if positive[position]:
                continue
            left = np.where(positive[:position])[0]
            right = np.where(positive[position + 1 :])[0]
            if left.size and right.size:
                widths.append(float(durations[position + 1 + right[0]] - durations[left[-1]]))
        rows.append(
            {
                "scenario_id": int(keys[0]),
                "country": keys[1],
                "year": int(keys[2]),
                "chronology": keys[3],
                "gap_width_h": max(widths) if widths else 0.0,
            }
        )
    return pd.DataFrame(rows)


def duration_envelope_stats(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["scenario_id", "chronology", "duration_h"], as_index=False)
        .agg(
            mean_eur_per_kw_year=("net_benefit_eur_per_kw_year", "mean"),
            q25_eur_per_kw_year=(
                "net_benefit_eur_per_kw_year",
                lambda x: x.quantile(0.25),
            ),
            q75_eur_per_kw_year=(
                "net_benefit_eur_per_kw_year",
                lambda x: x.quantile(0.75),
            ),
            positive_share=(
                "net_benefit_eur_per_kw_year",
                lambda x: (x > 0).mean(),
            ),
        )
        .assign(
            cost_snapshot_year=2035,
            reporting_fx_reference_year=FX_REFERENCE_YEAR,
            technology_cost_reporting_routes="technology-specific",
        )
    )


def figure5(
    results: pd.DataFrame,
    envelope: pd.DataFrame,
    gap_summary: pd.DataFrame,
    output: Path,
    source_dir: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 6.15))
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.085, top=0.965)
    gs = fig.add_gridspec(
        3, 6, height_ratios=[1.0, 1.0, 1.0],
        hspace=0.50, wspace=0.86,
    )
    axa = fig.add_subplot(gs[0, 0:2])
    axb = fig.add_subplot(gs[0:2, 2:6])
    axc = fig.add_subplot(gs[1, 0:2])
    axd = fig.add_subplot(gs[2, 0:2])
    axe = fig.add_subplot(gs[2, 2:4])
    axf = fig.add_subplot(gs[2, 4:6])
    axes = [axa, axb, axc, axd, axe, axf]
    fig.text(
        0.982,
        0.986,
        "2035 cost snapshot",
        ha="right",
        va="top",
        fontsize=6.2,
        color="#66737A",
    )
    chosen = envelope[envelope["scenario_id"].isin([20, 31])]
    stats = duration_envelope_stats(chosen)
    for axis, scenario in [(axa, 20), (axb, 31)]:
        current = stats[stats["scenario_id"].eq(scenario)]
        for chronology, color in [("dispersed", DISPERSED), ("clustered", CLUSTERED)]:
            group = current[current["chronology"].eq(chronology)].sort_values("duration_h")
            axis.fill_between(
                group["duration_h"],
                group["q25_eur_per_kw_year"],
                group["q75_eur_per_kw_year"],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            axis.plot(
                group["duration_h"],
                group["mean_eur_per_kw_year"],
                color=color,
                marker="o",
                ms=2.6 if scenario == 31 else 2.1,
                label=chronology.capitalize(),
            )
        axis.axhline(0, color="#777777", lw=0.7)
        axis.set_xlabel("Discharge duration (h)")
        axis.set_ylabel("Best annual net benefit\n(EUR kW$^{-1}$ yr$^{-1}$)")
        log_duration_axis(axis, maximum=168)
        axis.grid(axis="x", color="#ECEFF1", lw=0.45)
    axa.legend(loc="lower left", fontsize=5.1)
    axb.legend(loc="upper right", fontsize=5.4)
    axb.axvspan(28, 40, color="#7E57A5", alpha=0.13)
    axb.text(
        33.5,
        0.95,
        "negative-value interval",
        transform=axb.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#6A51A3",
        fontsize=5.7, weight="bold",
    )
    axb.axvline(44, color=TECH_COLOR["h2_cavern"], ls=":", lw=0.9)
    axb.text(
        44, 0.10, "re-entry begins\nat 44 h",
        transform=axb.get_xaxis_transform(), ha="left", va="bottom",
        fontsize=5.4, color=TECH_COLOR["h2_cavern"],
    )
    dominant = (
        chosen[
            chosen["scenario_id"].eq(31)
            & chosen["chronology"].eq("clustered")
        ]
        .groupby(["duration_h", "tech_id"], as_index=False)
        .size()
        .sort_values(["duration_h", "size"], ascending=[True, False])
        .drop_duplicates("duration_h")
    )
    dominant["cost_snapshot_year"] = 2035
    ymin, ymax = axb.get_ylim()
    band_y = ymin + 0.045 * (ymax - ymin)
    for _, row in dominant.iterrows():
        axb.scatter(
            row["duration_h"], band_y, s=17,
            color=TECH_COLOR[row["tech_id"]], edgecolor="white",
            linewidth=0.35, zorder=5,
        )
    label_y = band_y - 0.025 * (ymax - ymin)
    for tech_id, label in [
        ("li_lfp", "LFP"),
        ("caes", "A-CAES"),
        ("h2_cavern", "Cavern H$_2$"),
    ]:
        durations = dominant.loc[
            dominant["tech_id"].eq(tech_id), "duration_h"
        ].to_numpy(float)
        if durations.size:
            x_label = float(np.exp(np.mean(np.log(durations))))
            axb.text(
                x_label,
                label_y,
                label,
                fontsize=5.0,
                color=TECH_COLOR[tech_id],
                va="top",
                ha="center",
                weight="bold",
            )

    deep = stats[stats["scenario_id"].eq(31)]
    for chronology, color in [("dispersed", DISPERSED), ("clustered", CLUSTERED)]:
        group = deep[deep["chronology"].eq(chronology)].sort_values("duration_h")
        axc.plot(group["duration_h"], 100 * group["positive_share"], color=color, marker="o", ms=2.3, label=chronology.capitalize())
    axc.set_xlabel("Discharge duration (h)")
    axc.set_ylabel("Positive country-years (%)")
    axc.set_ylim(0, 103)
    log_duration_axis(axc, maximum=168)
    axc.axvspan(28, 40, color="#7E57A5", alpha=0.10)
    axc.grid(axis="x", color="#ECEFF1", lw=0.45)

    widths = gap_widths(chosen)
    widths["cost_snapshot_year"] = 2035
    deep_width = widths[widths["scenario_id"].eq(31)].pivot_table(index=["country", "year"], columns="chronology", values="gap_width_h")
    rng = np.random.default_rng(39)
    for index, chronology in enumerate(["dispersed", "clustered"]):
        values = deep_width[chronology].to_numpy()
        color = [DISPERSED, CLUSTERED][index]
        axd.scatter(
            rng.normal(index, 0.055, size=len(values)), values,
            s=6, alpha=0.18, color=color, linewidth=0,
        )
        if np.unique(values).size > 1:
            parts = axd.violinplot(
                values, positions=[index], widths=0.68,
                showextrema=False, showmeans=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("none")
                body.set_alpha(0.20)
        axd.plot(
            [index - 0.16, index + 0.16],
            [np.median(values)] * 2, color=DARK, lw=2,
        )
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["Dispersed", "Clustered"])
    axd.set_ylabel("Valley width (h)")
    axd.set_ylim(-2, max(50, deep_width.max().max() * 1.08))
    axd.text(
        0.98, 0.94, f"mean {deep_width['clustered'].mean():.1f} h",
        transform=axd.transAxes, ha="right", va="top",
        fontsize=5.5, color=CLUSTERED, weight="bold",
    )

    central_firm = gap_summary[
        np.isclose(
            gap_summary["firm_value_eur_kw_year"],
            842.6943979480799 * EUR_PER_CNY,
        )
    ]
    clustered = central_firm[central_firm["chronology"].eq("clustered")].pivot(
        index="charge_price_eur_mwh",
        columns="energy_value_eur_mwh",
        values="country_years_with_gap",
    )
    dispersed = central_firm[central_firm["chronology"].eq("dispersed")].pivot(
        index="charge_price_eur_mwh",
        columns="energy_value_eur_mwh",
        values="country_years_with_gap",
    )
    phase = 100 * (clustered - dispersed) / 132
    norm = mpl.colors.Normalize(vmin=-100, vmax=100)
    cmap = mpl.colormaps["RdBu_r"]
    bubble_rows: list[dict[str, float]] = []
    for i, charge_value in enumerate(phase.index):
        for j, energy_value in enumerate(phase.columns):
            value = float(phase.iloc[i, j])
            axe.scatter(
                j, i, s=22 + 1.8 * abs(value),
                color=cmap(norm(value)), edgecolor="white", linewidth=0.55,
            )
            axe.text(
                j, i, f"{value:.0f}", ha="center", va="center",
                fontsize=5.1, color="white" if abs(value) > 55 else DARK,
                weight="bold" if abs(value) > 70 else "normal",
            )
            bubble_rows.append(
                {
                    "energy_value_eur_per_mwh": float(energy_value),
                    "charge_price_eur_per_mwh": float(charge_value),
                    "valley_prevalence_change_pp": value,
                }
            )
    axe.set_xlim(-0.55, phase.shape[1] - 0.45)
    axe.set_ylim(-0.55, phase.shape[0] - 0.45)
    axe.set_xticks(range(phase.shape[1]))
    axe.set_xticklabels([f"{value:.1f}" for value in phase.columns])
    axe.set_yticks(range(phase.shape[0]))
    axe.set_yticklabels([f"{value:.2f}" for value in phase.index])
    axe.set_xlabel("Energy displacement value (EUR MWh$^{-1}$)")
    axe.set_ylabel("Charging price\n(EUR MWh$^{-1}$)")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axe, fraction=0.05, pad=0.02)
    cbar.set_label("$\\Delta$ valley prevalence (pp)")
    axe.spines["top"].set_visible(False)
    axe.spines["right"].set_visible(False)

    selected_configs = [
        "li_lfp_10h", "caes_24h",
        "h2_cavern_36h", "h2_cavern_60h",
    ]
    current = results[results["chronology"].eq("clustered") & results["config_id"].isin(selected_configs)].copy()
    current["energy_value_eur_per_kw_year"] = (
        current["discharge_mwh"]
        * (800 * EUR_PER_CNY)
        / (current["pdis_mw"] * 1000)
    )
    current["firm_value_eur_per_kw_year"] = (
        current["efc_ratio"] * 842.6943979480799 * EUR_PER_CNY
    )
    current["storage_cost_eur_per_kw_year"] = -(
        current["annual_fixed_cost_eur_per_kw_year"]
        + current["storage_vom_eur_per_kw_year"]
    )
    current["charge_cost_eur_per_kw_year"] = 0.0
    component_columns = [
        "energy_value_eur_per_kw_year",
        "firm_value_eur_per_kw_year",
        "storage_cost_eur_per_kw_year",
        "charge_cost_eur_per_kw_year",
    ]
    components = (
        current.groupby("config_id", as_index=False)[component_columns]
        .mean()
        .set_index("config_id")
        .loc[selected_configs]
    )
    x = np.arange(len(components))
    bottom_pos = np.zeros(len(components))
    for column, color, label in [
        ("energy_value_eur_per_kw_year", "#4C78A8", "Energy"),
        ("firm_value_eur_per_kw_year", "#59A14F", "Task-firm supply"),
    ]:
        axf.bar(x, components[column], bottom=bottom_pos, color=color, width=0.65, label=label)
        bottom_pos += components[column].to_numpy()
    axf.bar(
        x,
        components["storage_cost_eur_per_kw_year"],
        color="#B0B0B0",
        width=0.65,
        label="Storage cost",
    )
    net_value = components[component_columns].sum(axis=1)
    components["net_value_eur_per_kw_year"] = net_value
    components["cost_snapshot_year"] = 2035
    axf.scatter(
        x, net_value, marker="D", s=22,
        color=DARK, edgecolor="white", linewidth=0.45,
        zorder=5, label="Net",
    )
    axf.axhline(0, color="#777777", lw=0.7)
    axf.set_xticks(x)
    axf.set_xticklabels(
        ["LFP 10 h", "A-CAES 24 h", "H$_2$ 36 h", "H$_2$ 60 h"],
        rotation=22, ha="right",
    )
    axf.set_ylabel("Value (EUR kW$^{-1}$ yr$^{-1}$)")
    axf.legend(
        loc="upper left", ncol=2, handlelength=1.1,
        columnspacing=0.8, fontsize=5.0,
    )
    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_figure5_conditional_gap")
    save_source(stats, source_dir, "figure5_panel_abc_duration_value_envelopes.csv")
    save_source(
        dominant, source_dir,
        "figure5_panel_b_dominant_technology_by_duration.csv",
    )
    save_source(widths, source_dir, "figure5_panel_d_gap_widths.csv")
    phase_source = pd.DataFrame(bubble_rows)
    phase_source["cost_snapshot_year"] = 2035
    phase_source = add_cny_columns(phase_source)
    save_source(phase_source, source_dir, "figure5_panel_e_gap_phase_map.csv")
    save_source(
        add_cny_columns(components.reset_index()),
        source_dir,
        "figure5_panel_f_value_components.csv",
    )


def figure6(
    output: Path,
    source_dir: Path,
    china_results: Path = DEFAULT_RESULTS,
) -> None:
    """Build the China validation figure with source-aware economic panels."""
    boundary = pd.read_csv(
        china_results / "china_osr_boundary_source_aware.csv"
    )
    fixed = pd.read_csv(
        china_results / "china_fixed_asset_summary_source_aware_eur.csv"
    )
    retention = pd.read_csv(CHINA_F4 / "figure4_panel_c_turnover_retention.csv")
    comparison = pd.read_csv(
        china_results / "china_fixed_asset_comparison_source_aware_eur.csv"
    )
    province_rows: list[pd.DataFrame] = []
    dispatch_routes = [
        ("Cyclic dispatch", "cyclic_heuristic", "heuristic"),
        ("Perfect foresight", "perfect_foresight", "system_total"),
    ]
    for dispatch, dispatch_method, objective_mode in dispatch_routes:
        route = comparison[
            comparison["dispatch_method"].eq(dispatch_method)
            & comparison["chronology"].eq("discharge_first")
            & comparison["objective_mode"].eq(objective_mode)
        ]
        wide = route.pivot(
            index="province_en",
            columns="persistence_level",
            values="caes_mix_net_benefit_eur_per_mwh_task",
        )
        if set(wide.columns) != {1, 10} or len(wide) != 26:
            raise ValueError(
                f"Figure 6d requires 26 paired P1/P10 provinces for {dispatch}."
            )
        route_metadata = route[
            [
                "caes_currency_route",
                "lfp_currency_route",
                "value_item_currency_route",
                "model_cny_per_usd",
                "reporting_usd_per_eur",
                "reporting_cny_per_eur",
            ]
        ].drop_duplicates()
        if len(route_metadata) != 1:
            raise ValueError(f"Inconsistent Figure 6d currency routes for {dispatch}.")
        current = wide[[1, 10]].rename(
            columns={
                1: "p1_caes_mix_net_benefit_eur_per_mwh_task",
                10: "p10_caes_mix_net_benefit_eur_per_mwh_task",
            }
        ).reset_index()
        current["dispatch"] = dispatch
        current["dispatch_method"] = dispatch_method
        current["chronology"] = "discharge_first"
        current["objective_mode"] = objective_mode
        current["persistence_from"] = 1
        current["persistence_to"] = 10
        current["change_eur_per_mwh_task"] = (
            current["p10_caes_mix_net_benefit_eur_per_mwh_task"]
            - current["p1_caes_mix_net_benefit_eur_per_mwh_task"]
        )
        for column, value in route_metadata.iloc[0].items():
            current[column] = value
        province_rows.append(current)
    province = pd.concat(province_rows, ignore_index=True)
    centroids = pd.read_csv(CHINA_F4 / "figure4_derived_province_centroids.csv")
    spearman = pd.read_csv(CHINA_F4 / "figure4_derived_spearman_summary.csv")
    economic = pd.read_csv(
        china_results / "china_mix_technology_year_summary_source_aware_eur.csv"
    )
    fig = plt.figure(figsize=(7.2, 5.95))
    fig.subplots_adjust(left=0.120, right=0.985, bottom=0.085, top=0.965)
    gs = fig.add_gridspec(
        3, 6, height_ratios=[1.10, 0.95, 1.0],
        hspace=0.52, wspace=0.82,
    )
    axa = fig.add_subplot(gs[0, 0:4])
    axb = fig.add_subplot(gs[0, 4:6])
    axc = fig.add_subplot(gs[1, 0:3])
    axd = fig.add_subplot(gs[1, 3:6])
    axe = fig.add_subplot(gs[2, 0:3])
    axf = fig.add_subplot(gs[2, 3:6])
    axes = [axa, axb, axc, axd, axe, axf]

    year_colors = {2025: "#4C78A8", 2030: "#F2A541", 2035: "#59A14F"}
    for year, group in boundary.groupby("learning_year"):
        group = group.sort_values("mean_run_length_days")
        axa.fill_between(group["mean_run_length_days"], group["ci_low"], group["ci_high"], color=year_colors[year], alpha=0.15, linewidth=0)
        axa.plot(group["mean_run_length_days"], group["mean_osr_gain_pp"], color=year_colors[year], marker="o", ms=2.5, label=str(year))
    axa.axhline(0, color="#777777", lw=0.7)
    axa.set_xlabel("Mean event run length (d)")
    axa.set_ylabel("CAES-mix service advantage (pp)")
    axa.legend(title="Cost year", ncol=3, loc="upper right")
    axa.grid(axis="y", color="#ECEFF1", lw=0.45)

    subset = fixed[
        (fixed["chronology"].eq("discharge_first"))
        & fixed["persistence_level"].isin([1, 10])
        & fixed["objective_mode"].isin(["heuristic", "system_total"])
    ]
    method_colors = {
        "cyclic_heuristic": DISPERSED,
        "perfect_foresight": CLUSTERED,
    }
    for index, method in enumerate(subset["dispatch_method"].unique()):
        group = subset[subset["dispatch_method"].eq(method)].set_index("persistence_level")
        axb.plot(
            [0, 1],
            [
                group.loc[
                    1,
                    "demand_weighted_caes_mix_net_benefit_eur_per_mwh_task",
                ],
                group.loc[
                    10,
                    "demand_weighted_caes_mix_net_benefit_eur_per_mwh_task",
                ],
            ],
            marker="o",
            color=method_colors.get(method, "#777777"),
            label=method.replace("_", " ").title(),
        )
    axb.axhline(0, color="#777777", lw=0.7)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["Dispersed", "Clustered"])
    axb.set_ylabel("CAES-mix benefit\n(EUR MWh$^{-1}$ task)")
    axb.legend(loc="best", fontsize=5.1)

    retention["system_label"] = retention["system"].replace({"extra LFP4": "Additional LFP4", "CAES24": "A-CAES24"})
    metric_order = ["Tail supply", "Charge throughput", "Equivalent full cycles"]
    y = np.arange(len(metric_order))
    for offset, system in [(-0.12, "Additional LFP4"), (0.12, "A-CAES24")]:
        group = retention[retention["system_label"].eq(system)].set_index("label").reindex(metric_order)
        color = TECH_COLOR["li_lfp"] if "LFP" in system else TECH_COLOR["caes"]
        axc.hlines(
            y + offset, 100, group["high_as_percent_of_low"],
            color=color, lw=1.2, alpha=0.65,
        )
        axc.scatter(
            group["high_as_percent_of_low"], y + offset,
            color=color, s=24, label=system, zorder=3,
        )
    axc.axvline(100, color="#777777", ls="--", lw=0.7)
    axc.set_yticks(y)
    axc.set_yticklabels(metric_order)
    axc.invert_yaxis()
    axc.set_xlabel("Clustered/dispersed retention (%)")
    axc.legend(loc="lower right")

    for dispatch, color in [("Cyclic dispatch", DISPERSED), ("Perfect foresight", CLUSTERED)]:
        values = province.loc[
            province["dispatch"].eq(dispatch),
            "change_eur_per_mwh_task",
        ].to_numpy()
        x, y_ecdf = ecdf(values)
        axd.fill_between(x, 0, y_ecdf, color=color, alpha=0.08)
        axd.plot(x, y_ecdf, color=color, label=dispatch)
    axd.axvline(0, color="#777777", lw=0.7)
    axd.set_xlabel("P10 − P1 fixed-asset mix benefit\n(EUR MWh$^{-1}$ task)")
    axd.set_ylabel("Cumulative share of provinces")
    axd.legend(loc="lower right")

    xcol = "caes_minus_lfp__delta_event_recoverable_deficit_weighted"
    ycol = "caes_minus_lfp__delta_system_served_pct"
    axe.scatter(centroids[xcol], centroids[ycol], color=TECH_COLOR["caes"], s=23, alpha=0.8, edgecolor="white", linewidth=0.4)
    coefficient = np.polyfit(centroids[xcol], centroids[ycol], 1)
    xx = np.linspace(centroids[xcol].min(), centroids[xcol].max(), 100)
    axe.plot(xx, np.polyval(coefficient, xx), color=DARK, lw=0.9)
    rho = float(spearman.loc[spearman["outcome"].eq("System served share"), "spearman_rho"].iloc[0])
    axe.text(
        0.04, 0.94, f"Spearman $\\rho$ = {rho:.2f}",
        transform=axe.transAxes, va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.1),
    )
    axe.set_xlabel("$\\Delta$ recoverable event deficit")
    axe.set_ylabel("$\\Delta$ CAES-minus-LFP\nserved share (pp)")

    current = economic[(economic["tech_id"].eq("caes")) & (economic["evidence_set"].eq("observed_year"))]
    for year, group in current.groupby("learning_year"):
        group = group.sort_values("charge_opportunity_price_eur_kwh")
        axf.plot(
            group["charge_opportunity_price_eur_kwh"],
            group["mean_mix_system_cost_reduction_pct"],
            marker="o",
            color=year_colors[year],
            label=str(year),
        )
        axf.fill_between(
            group["charge_opportunity_price_eur_kwh"],
            0,
            group["mean_mix_system_cost_reduction_pct"],
            color=year_colors[year], alpha=0.06,
        )
    axf.axhline(0, color="#777777", lw=0.7)
    axf.set_xlabel("Charging opportunity price (EUR kWh$^{-1}$)")
    axf.set_ylabel("Hybrid system-cost reduction (%)")
    axf.legend(title="Cost year", ncol=3, fontsize=5.1)
    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_figure6_china_external_validation")
    save_source(boundary, source_dir, "figure6_panel_a_china_boundary_response.csv")
    save_source(fixed, source_dir, "figure6_panel_b_china_fixed_asset.csv")
    save_source(
        fixed,
        source_dir,
        "figure6_panel_b_china_fixed_asset_source_aware_eur.csv",
    )
    save_source(retention, source_dir, "figure6_panel_c_china_retention.csv")
    save_source(province, source_dir, "figure6_panel_d_china_province_changes.csv")
    save_source(centroids, source_dir, "figure6_panel_e_china_recoverable_deficit.csv")
    save_source(current, source_dir, "figure6_panel_f_china_economics.csv")
    save_source(
        current,
        source_dir,
        "figure6_panel_f_china_economics_source_aware_eur.csv",
    )


def supplementary_china_event_sensitivity(
    output: Path,
    source_dir: Path,
) -> None:
    """Rebuild the Chinese event-definition sensitivity in the v40 style."""
    grid = pd.read_csv(CHINA_ED2 / "ed2_province_definition_grid.csv")
    summary = pd.read_csv(CHINA_ED2 / "ed2_definition_summary.csv")
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 5.35))
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.11, top=0.95,
        wspace=0.38, hspace=0.44,
    )
    axes = axes_grid.ravel().tolist()
    axa, axb, axc, axd = axes
    q_colors = {0.70: "#4C78A8", 0.75: "#F2A541", 0.80: "#59A14F", 0.85: "#B279A2"}

    focus = grid[grid["min_run_days"].eq(3)]
    rng = np.random.default_rng(406)
    for index, quantile in enumerate(sorted(focus["quantile"].unique())):
        values = focus.loc[
            focus["quantile"].eq(quantile),
            "persistent_fraction_tail_pct",
        ].to_numpy(float)
        color = q_colors[float(quantile)]
        parts = axa.violinplot(
            values, positions=[index], widths=0.72,
            showextrema=False, showmeans=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.25)
        axa.scatter(
            rng.normal(index, 0.055, len(values)), values,
            s=8, color=color, alpha=0.22, linewidth=0,
        )
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axa.plot(
            [index, index], [q1, q3], color=DARK,
            lw=3.0, solid_capstyle="round",
        )
        axa.scatter(
            index, med, s=20, facecolor="white",
            edgecolor=DARK, linewidth=0.6, zorder=3,
        )
    axa.set_xticks(range(4))
    axa.set_xticklabels(["q70", "q75", "q80", "q85"])
    axa.set_ylabel("Tail energy in events $\\geq$3 d (%)")

    for quantile, group in summary.groupby("quantile"):
        group = group.sort_values("min_run_days")
        color = q_colors[float(quantile)]
        axb.fill_between(
            group["min_run_days"],
            group["mean_persistent_fraction_ci_low"],
            group["mean_persistent_fraction_ci_high"],
            color=color, alpha=0.10, linewidth=0,
        )
        axb.plot(
            group["min_run_days"],
            group["mean_persistent_fraction_tail_pct"],
            color=color, marker="o", label=f"q{int(100*quantile)}",
        )
    axb.set_xticks([2, 3, 4, 5])
    axb.set_xlabel("Minimum event run (d)")
    axb.set_ylabel("Mean persistent share of tail energy (%)")
    axb.legend(ncol=2, fontsize=5.3)

    relation = grid[
        np.isclose(grid["quantile"], 0.75)
        & grid["min_run_days"].eq(3)
    ]
    axc.scatter(
        relation["max_persistent_run_length_days"],
        relation["persistent_fraction_tail_pct"],
        s=22, color="#6A51A3", alpha=0.72,
        edgecolor="white", linewidth=0.4,
    )
    coefficient = np.polyfit(
        relation["max_persistent_run_length_days"],
        relation["persistent_fraction_tail_pct"], 1,
    )
    xx = np.linspace(
        relation["max_persistent_run_length_days"].min(),
        relation["max_persistent_run_length_days"].max(), 100,
    )
    axc.plot(xx, np.polyval(coefficient, xx), color=DARK, lw=0.9)
    rho = relation["max_persistent_run_length_days"].corr(
        relation["persistent_fraction_tail_pct"], method="spearman",
    )
    axc.text(
        0.04, 0.94, f"Spearman $\\rho$ = {rho:.2f}",
        transform=axc.transAxes, va="top",
    )
    axc.set_xlabel("Longest persistent event (d)")
    axc.set_ylabel("Persistent share of tail energy (%)")

    run_colors = {2: "#4C78A8", 3: "#F2A541", 4: "#59A14F", 5: "#B279A2"}
    for min_run, group in summary.groupby("min_run_days"):
        group = group.sort_values("quantile")
        x = 100 * group["quantile"]
        axd.plot(
            x, group["pearson_r"], color=run_colors[int(min_run)],
            marker="o", label=f"$\\geq${int(min_run)} d",
        )
        axd.plot(
            x, group["spearman_rho"], color=run_colors[int(min_run)],
            marker="s", ls="--", alpha=0.75,
        )
    axd.axhline(0, color="#777777", lw=0.7)
    axd.set_xticks([70, 75, 80, 85])
    axd.set_xlabel("Event-day quantile threshold")
    axd.set_ylabel("Association coefficient")
    axd.legend(ncol=2, fontsize=5.1)
    axd.text(
        0.98, 0.05, "circles Pearson · squares Spearman",
        transform=axd.transAxes, ha="right", va="bottom",
        fontsize=5.1, color="#66737A",
    )

    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_ed6_china_event_sensitivity")
    save_source(grid, source_dir, "ed6_china_event_definition_grid.csv")
    save_source(summary, source_dir, "ed6_china_event_definition_summary.csv")


def supplementary_china_dispatch_validation(
    output: Path,
    source_dir: Path,
    china_results: Path = DEFAULT_RESULTS,
) -> None:
    """Build Chinese dispatch validation with source-aware economic accounting."""
    comparison = pd.read_csv(
        china_results / "china_fixed_asset_comparison_source_aware_eur.csv"
    )
    changes = pd.read_csv(
        china_results / "china_fixed_asset_paired_changes_source_aware_eur.csv"
    )
    summary = pd.read_csv(
        china_results / "china_fixed_asset_summary_source_aware_eur.csv"
    )
    effect_endpoints = pd.read_csv(
        CHINA_ED3 / "ed3_panel_e_intervention_endpoint_effects.csv"
    ).rename(columns=lambda column: column.replace("p3_", "p10_"))
    effect_summary = pd.read_csv(
        CHINA_ED3 / "ed3_panel_e_intervention_bootstrap_summary.csv"
    ).rename(columns=lambda column: column.replace("p3_", "p10_"))
    cycle_summary = pd.read_csv(
        CHINA_ED3 / "ed3_panel_f_technology_cycle_bootstrap_summary.csv"
    ).rename(columns=lambda column: column.replace("p3_", "p10_"))

    fig, axes_grid = plt.subplots(
        3,
        2,
        figsize=(7.2, 7.2),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.12]},
    )
    fig.subplots_adjust(
        left=0.16,
        right=0.985,
        bottom=0.075,
        top=0.97,
        wspace=0.40,
        hspace=0.58,
    )
    axes = axes_grid.ravel().tolist()
    axa, axb, axc, axd, axe, axf = axes

    primary = comparison[
        comparison["dispatch_method"].eq("cyclic_heuristic")
        & comparison["chronology"].eq("discharge_first")
        & comparison["objective_mode"].eq("heuristic")
    ]
    demand = primary.pivot(
        index="province_en",
        columns="persistence_level",
        values="annual_demand_kwh__CAES24",
    )
    demand_delta = (demand[10] - demand[1]).abs().sort_values()
    axa.scatter(
        np.arange(len(demand_delta)),
        np.maximum(demand_delta.to_numpy(float), 1e-12),
        s=13,
        color="#0072B2",
        alpha=0.75,
    )
    axa.axhline(
        1e-3,
        color="#D55E00",
        ls="--",
        lw=0.8,
        label="Pre-specified tolerance",
    )
    axa.set_yscale("log")
    axa.set_xlabel("Province (ordered by absolute difference)")
    axa.set_ylabel("|P10 − P1 annual task| (kWh)")
    axa.legend(loc="upper left")
    axa.grid(axis="y", color=LIGHT_GREY, lw=0.45)

    residual = np.maximum(
        np.abs(comparison["cost_identity_error_eur"].to_numpy(float)),
        1e-14,
    )
    axb.scatter(
        np.linspace(0.15, 0.85, len(residual)),
        residual,
        s=7,
        color="#009E73",
        alpha=0.28,
        edgecolors="none",
    )
    axb.axhline(
        EUR_PER_CNY,
        color="#D55E00",
        ls="--",
        lw=0.8,
        label=f"{EUR_PER_CNY:.3f} EUR audit tolerance",
    )
    axb.set_yscale("log")
    axb.set_xticks([])
    axb.set_xlabel("All arm comparisons (n = 312)")
    axb.set_ylabel("Absolute cost-identity residual (EUR)")
    axb.legend(loc="upper left")
    axb.grid(axis="y", color=LIGHT_GREY, lw=0.45)

    dispatch_labels = {
        ("cyclic_heuristic", "discharge_first", "heuristic"):
            "Cyclic · discharge-first",
        ("cyclic_heuristic", "charge_first", "heuristic"):
            "Cyclic · charge-first",
        ("perfect_foresight", "discharge_first", "system_total"):
            "Foresight · total",
        ("perfect_foresight", "charge_first", "system_total"):
            "Foresight · total (alt.)",
        ("perfect_foresight", "discharge_first", "base_priority"):
            "Foresight · base",
        ("perfect_foresight", "charge_first", "base_priority"):
            "Foresight · base (alt.)",
    }
    changes["plot_label"] = [
        dispatch_labels[
            (row.dispatch_method, row.chronology, row.objective_mode)
        ]
        for _, row in changes.iterrows()
    ]
    changes = changes.sort_values(
        ["dispatch_method", "objective_mode", "chronology"],
        kind="stable",
    ).reset_index(drop=True)
    y = np.arange(len(changes))
    for yy, (_, row) in zip(y, changes.iterrows()):
        color = "#0072B2" if row.dispatch_method == "cyclic_heuristic" else "#D55E00"
        mean = float(row.province_mean_change_eur_per_mwh_task)
        axc.errorbar(
            mean,
            yy,
            xerr=[
                [mean - float(row.ci_low_eur_per_mwh_task)],
                [float(row.ci_high_eur_per_mwh_task) - mean],
            ],
            fmt="o",
            ms=3.8,
            color=color,
            capsize=2,
            lw=1.1,
        )
    axc.axvline(0, color="#777777", lw=0.7)
    axc.set_yticks(y)
    axc.set_yticklabels(changes["plot_label"])
    axc.tick_params(axis="y", labelsize=6.0)
    axc.invert_yaxis()
    axc.set_xlabel("P10 − P1 direct-cost benefit (EUR MWh$^{-1}$ task)")
    axc.grid(axis="x", color=LIGHT_GREY, lw=0.45)

    selected = summary[
        (
            summary["dispatch_method"].eq("cyclic_heuristic")
            & summary["chronology"].eq("discharge_first")
            & summary["objective_mode"].eq("heuristic")
        )
        | (
            summary["dispatch_method"].eq("perfect_foresight")
            & summary["chronology"].eq("discharge_first")
            & summary["objective_mode"].eq("system_total")
        )
    ]
    for method, group in selected.groupby("dispatch_method"):
        group = group.sort_values("persistence_level")
        color = "#0072B2" if method == "cyclic_heuristic" else "#D55E00"
        label = "Cyclic heuristic" if method == "cyclic_heuristic" else "Perfect foresight"
        axd.plot(
            [0, 1],
            group["demand_weighted_caes_mix_net_benefit_eur_per_mwh_task"],
            marker="o",
            ms=4,
            color=color,
            label=label,
        )
    axd.axhline(0, color="#777777", lw=0.7)
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["Dispersed P1", "Clustered P10"])
    axd.set_xlim(-0.08, 1.08)
    axd.set_ylabel("Direct-cost benefit (EUR MWh$^{-1}$ task)")
    axd.legend(loc="upper right")
    axd.grid(axis="y", color=LIGHT_GREY, lw=0.45)

    intervention_order = [
        "reference",
        "event_front_equalized",
        "no_event_recharge",
        "abundant_daily",
    ]
    intervention_label = {
        "reference": "Reference",
        "event_front_equalized": "Event-front equalized",
        "no_event_recharge": "No event recharge",
        "abundant_daily": "Abundant daily",
    }
    intervention_color = {
        "reference": "#0072B2",
        "event_front_equalized": "#6A51A3",
        "no_event_recharge": "#D55E00",
        "abundant_daily": "#009E73",
    }
    endpoint_matrix = effect_endpoints.pivot(
        index="province_en",
        columns="variant",
        values="endpoint_effect_pp",
    )
    effect_lookup = effect_summary.set_index("variant")
    for yy, variant in enumerate(intervention_order):
        values = endpoint_matrix[variant].sort_values().to_numpy(float)
        color = intervention_color[variant]
        axe.scatter(
            values,
            yy + np.linspace(-0.12, 0.12, len(values)),
            s=8,
            color=color,
            alpha=0.22,
            edgecolors="none",
        )
        row = effect_lookup.loc[variant]
        mean = float(row.mean_endpoint_effect_pp)
        axe.errorbar(
            mean,
            yy,
            xerr=[
                [mean - float(row.paired_bootstrap_ci_low_pp)],
                [float(row.paired_bootstrap_ci_high_pp) - mean],
            ],
            fmt="o",
            ms=4.1,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            capsize=2.0,
            lw=1.2,
            zorder=3,
        )
        axe.text(
            mean + 0.20,
            yy - 0.16,
            f"{mean:+.2f}",
            color=color,
            fontsize=6.0,
            va="center",
        )
    axe.axvline(0, color="#777777", lw=0.75)
    axe.set_yticks(np.arange(len(intervention_order)))
    axe.set_yticklabels([intervention_label[value] for value in intervention_order])
    axe.invert_yaxis()
    axe.set_xlim(-7.8, 4.8)
    axe.set_xlabel("P10 − P1 CAES relative served-share response (pp)")
    axe.grid(axis="x", color=LIGHT_GREY, lw=0.45)

    cycle_lookup = cycle_summary.set_index(["variant", "technology"])
    technology_style = {
        "24 h CAES": ("#D55E00", "o"),
        "4 h LFP": ("#0072B2", "s"),
    }
    variant_x = {
        "reference": np.array([0.0, 1.0]),
        "abundant_daily": np.array([2.8, 3.8]),
    }
    for variant in ("reference", "abundant_daily"):
        for technology in ("4 h LFP", "24 h CAES"):
            row = cycle_lookup.loc[(variant, technology)]
            color, marker = technology_style[technology]
            xx = variant_x[variant]
            values = np.array(
                [
                    row.p1_mean_equivalent_cycles_yr,
                    row.p10_mean_equivalent_cycles_yr,
                ],
                dtype=float,
            )
            lower = np.array([row.p1_ci_low, row.p10_ci_low], dtype=float)
            upper = np.array([row.p1_ci_high, row.p10_ci_high], dtype=float)
            axf.plot(
                xx,
                values,
                color=color,
                marker=marker,
                ms=4.0,
                lw=1.25,
                zorder=3,
            )
            axf.errorbar(
                xx,
                values,
                yerr=np.vstack([values - lower, upper - values]),
                fmt="none",
                ecolor=color,
                elinewidth=0.7,
                capsize=1.7,
                alpha=0.8,
                zorder=2,
            )
            change = float(row.mean_change_equivalent_cycles_yr)
            display_change = 0.0 if abs(change) < 0.005 else change
            axf.text(
                float(xx.mean()),
                float(values.mean()) + (2.1 if technology == "4 h LFP" else -2.1),
                f"$\\Delta$ {display_change:+.2f}",
                color=color,
                fontsize=6.0,
                ha="center",
                va="center",
            )
    axf.axvline(1.9, color=LIGHT_GREY, lw=0.75)
    axf.set_xlim(-0.28, 5.05)
    axf.set_ylim(0, 53)
    axf.set_xticks([0.0, 1.0, 2.8, 3.8])
    axf.set_xticklabels(["P1", "P10", "P1", "P10"])
    axf.text(
        0.5,
        51.2,
        "Reference",
        color=DARK,
        fontsize=6.3,
        ha="center",
        va="top",
        fontweight="bold",
    )
    axf.text(
        3.3,
        51.2,
        "Abundant daily",
        color=DARK,
        fontsize=6.3,
        ha="center",
        va="top",
        fontweight="bold",
    )
    axf.text(
        3.98,
        float(
            cycle_lookup.loc[
                ("abundant_daily", "4 h LFP"),
                "p10_mean_equivalent_cycles_yr",
            ]
        ),
        "4 h LFP",
        color="#0072B2",
        fontsize=6.0,
        va="center",
    )
    axf.text(
        3.98,
        float(
            cycle_lookup.loc[
                ("abundant_daily", "24 h CAES"),
                "p10_mean_equivalent_cycles_yr",
            ]
        ),
        "24 h CAES",
        color="#D55E00",
        fontsize=6.0,
        va="center",
    )
    axf.set_xlabel("Persistence endpoint")
    axf.set_ylabel("Tail equivalent cycles (yr$^{-1}$)")
    axf.grid(axis="y", color=LIGHT_GREY, lw=0.45)

    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_ed7_china_dispatch_validation")
    save_source(
        comparison,
        source_dir,
        "ed7_panel_abd_fixed_asset_comparison_dual_currency.csv",
    )
    save_source(
        comparison,
        source_dir,
        "ed7_panel_abd_fixed_asset_comparison_source_aware_eur.csv",
    )
    save_source(
        changes,
        source_dir,
        "ed7_panel_c_paired_changes_dual_currency.csv",
    )
    save_source(
        changes,
        source_dir,
        "ed7_panel_c_paired_changes_source_aware_eur.csv",
    )
    save_source(
        summary,
        source_dir,
        "ed7_panel_d_fixed_asset_summary_dual_currency.csv",
    )
    save_source(
        summary,
        source_dir,
        "ed7_panel_d_fixed_asset_summary_source_aware_eur.csv",
    )
    save_source(
        effect_endpoints,
        source_dir,
        "ed7_panel_e_intervention_endpoint_effects.csv",
    )
    save_source(
        effect_summary,
        source_dir,
        "ed7_panel_e_intervention_bootstrap_summary.csv",
    )
    save_source(
        cycle_summary,
        source_dir,
        "ed7_panel_f_technology_cycle_bootstrap_summary.csv",
    )


def supplementary_figures(
    panel: pd.DataFrame,
    results: pd.DataFrame,
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
    chronology: pd.DataFrame,
    output: Path,
    source_dir: Path,
) -> None:
    # ED1 is produced by the dedicated 53-configuration, three-cost-year
    # source-aware builder in ``ed1_cost_paths_53``.  The integrated script
    # starts with ED2 to avoid leaving a second, obsolete ED1 in the bundle.
    # ED2 | q70-q80 event-definition audit
    records: list[dict[str, float | int | str]] = []
    for quantile in [70, 75, 80]:
        daily = (
            panel.groupby(["country", "year", "source_day_id"], as_index=False)
            .agg(
                task=(f"task_c3_q{quantile}_mw", "sum"),
                event=(f"event_flag_q{quantile}", "max"),
                above_floor=("above_floor_gas_mw", "sum"),
            )
        )
        for (country, year), group in daily.groupby(["country", "year"]):
            group = group.sort_values("source_day_id")
            task = float(group["task"].sum())
            event = group["event"].astype(int)
            run_ids = event.ne(event.shift(fill_value=0)).cumsum()
            longest = int(
                event.groupby(run_ids).sum().max()
                if event.any() else 0
            )
            records.append(
                {
                    "country": country,
                    "year": year,
                    "quantile": quantile,
                    "event_days": int(group["event"].sum()),
                    "longest_event_run_days": longest,
                    "task_twh": task / 1.0e6,
                    "event_task_share_pct": (
                        100 * task / group["above_floor"].sum()
                        if group["above_floor"].sum() > 0
                        else 0
                    ),
                }
            )
    sensitivity = pd.DataFrame(records)
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    axes = axes_grid.ravel().tolist()
    for position, column, label in [
        (0, "longest_event_run_days", "Longest event run (d)"),
        (1, "task_twh", "Peaking task (TWh)"),
        (2, "event_task_share_pct", "Event-task share (%)"),
    ]:
        ax = axes[position]
        for quantile, color in zip([70, 75, 80], ["#4C78A8", "#F2A541", "#59A14F"]):
            values = sensitivity.loc[sensitivity["quantile"].eq(quantile), column].to_numpy()
            x, y = ecdf(values)
            ax.plot(x, y, color=color, label=f"q{quantile}")
        ax.set_xlabel(label)
        ax.set_ylabel("Cumulative share")
    axes[0].legend(ncol=3)
    pivot = sensitivity.pivot_table(index=["country", "year"], columns="quantile", values="event_task_share_pct")
    axes[3].scatter(pivot[70], pivot[80], s=13, alpha=0.65, color="#6A51A3")
    limits = [min(pivot[70].min(), pivot[80].min()), max(pivot[70].max(), pivot[80].max())]
    axes[3].plot(limits, limits, color="#777777", ls="--", lw=0.7)
    axes[3].set_xlabel("q70 event-task share (%)")
    axes[3].set_ylabel("q80 event-task share (%)")
    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_ed2_event_definition_sensitivity")
    save_source(sensitivity, source_dir, "ed2_event_definition_sensitivity.csv")

    # ED3 | full configuration paired responses
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    axes = axes_grid.ravel().tolist()
    metrics = [
        ("retained_equivalent_full_cycles_mean", "Clustered/dispersed EFC cycles", 1),
        ("delta_served_share_pp_mean", "$\\Delta$ served share (pp)", 0),
        (
            "delta_efc_ratio_mean",
            "$\\Delta$ EENS-equivalent task-firm supply ratio",
            0,
        ),
        (
            "delta_joint_net_benefit_eur_per_kw_year_mean",
            "$\\Delta$ joint net benefit\n(EUR kW$^{-1}$ yr$^{-1}$)",
            0,
        ),
    ]
    for ax, (column, ylabel, reference) in zip(axes, metrics):
        for tech in TECH_ORDER:
            group = summary[summary["tech_id"].eq(tech)].sort_values("duration_h")
            ax.plot(group["duration_h"], group[column], color=TECH_COLOR[tech], marker="o", ms=2.2, label=TECH_LABEL[tech])
        ax.axhline(reference, color="#777777", lw=0.7, ls="--" if reference else "-")
        ax.set_xlabel("Duration (h)")
        ax.set_ylabel(ylabel)
        log_duration_axis(ax)
    axes[0].legend(ncol=4, loc="lower right")
    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_ed4_full_duration_response")

    # ED4 | value scan across the four task-firm-supply values
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    axes = axes_grid.ravel().tolist()
    firm_values = sorted(gap_summary["firm_value_eur_kw_year"].unique())
    norm = mpl.colors.Normalize(vmin=-100, vmax=100)
    cmap = mpl.colormaps["RdBu_r"]
    for ax, firm in zip(axes, firm_values):
        subset = gap_summary[
            np.isclose(gap_summary["firm_value_eur_kw_year"], firm)
        ]
        clustered = subset[subset["chronology"].eq("clustered")].pivot(
            index="charge_price_eur_mwh",
            columns="energy_value_eur_mwh",
            values="country_years_with_gap",
        )
        dispersed = subset[subset["chronology"].eq("dispersed")].pivot(
            index="charge_price_eur_mwh",
            columns="energy_value_eur_mwh",
            values="country_years_with_gap",
        )
        phase = 100 * (clustered - dispersed) / 132
        for i in range(phase.shape[0]):
            for j in range(phase.shape[1]):
                value = float(phase.iloc[i, j])
                ax.scatter(
                    j, i, s=40 + 2.5 * abs(value),
                    color=cmap(norm(value)), edgecolor="white",
                    linewidth=0.6,
                )
                ax.text(
                    j, i, f"{value:.0f}", ha="center", va="center",
                    fontsize=5.4,
                    color="white" if abs(value) > 55 else DARK,
                )
        ax.set_xlim(-0.55, phase.shape[1] - 0.45)
        ax.set_ylim(-0.55, phase.shape[0] - 0.45)
        ax.set_xticks(range(phase.shape[1]))
        ax.set_xticklabels([f"{v:.1f}" for v in phase.columns])
        ax.set_yticks(range(phase.shape[0]))
        ax.set_yticklabels([f"{v:.2f}" for v in phase.index])
        ax.set_title(
            f"Task-firm-supply value = {firm:.1f} EUR kW$^{{-1}}$ yr$^{{-1}}$",
            loc="left",
            weight="bold",
        )
        ax.set_xlabel("Energy value (EUR MWh$^{-1}$)")
        ax.set_ylabel("Charge price (EUR MWh$^{-1}$)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    image = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("$\\Delta$ country-years with a strict valley (pp)")
    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_ed5_value_scan")
    save_source(
        gap_summary,
        source_dir,
        "ed5_value_scan_source_aware_currency.csv",
    )

    # ED5 | conservation and cyclic-SOC QA
    fig, axes_grid = plt.subplots(1, 2, figsize=(7.2, 3.05))
    fig.subplots_adjust(
        left=0.13, right=0.985, bottom=0.20, top=0.92, wspace=0.48,
    )
    axes = axes_grid.ravel().tolist()
    task_error = np.abs(chronology["task_original_conservation_error_mwh"].to_numpy())
    recharge_error = np.abs(chronology["recharge_original_conservation_error_mwh"].to_numpy())
    cyclic = np.abs(results["cyclic_soc_residual_mwh"].to_numpy())
    maxima = pd.DataFrame(
        {
            "metric": [
                "Task energy", "Recharge energy", "Cyclic SOC",
            ],
            "maximum_abs_error_mwh": [
                max(np.max(task_error), 1e-16),
                max(np.max(recharge_error), 1e-16),
                max(np.max(cyclic), 1e-16),
            ],
        }
    )
    qa_colors = [DISPERSED, "#2A9D8F", CLUSTERED]
    y = np.arange(len(maxima))
    axes[0].hlines(
        y, 1e-16, maxima["maximum_abs_error_mwh"],
        color=qa_colors, lw=2.2,
    )
    axes[0].scatter(
        maxima["maximum_abs_error_mwh"], y,
        color=qa_colors, s=28, edgecolor="white", linewidth=0.5,
    )
    axes[0].axvline(1e-6, color="#777777", ls="--", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-16, 1e-5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(maxima["metric"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Maximum absolute numerical error (MWh)")

    overlap_long = results[
        ["country", "year", "chronology", "task_recharge_overlap_hours"]
    ].drop_duplicates(["country", "year", "chronology"])
    chronology_order = ["natural", "dispersed", "clustered"]
    rng = np.random.default_rng(405)
    for index, chronology_name in enumerate(chronology_order):
        values = overlap_long.loc[
            overlap_long["chronology"].eq(chronology_name),
            "task_recharge_overlap_hours",
        ].to_numpy(float)
        color = [NATURAL, DISPERSED, CLUSTERED][index]
        parts = axes[1].violinplot(
            values, positions=[index], widths=0.68,
            showextrema=False, showmeans=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.22)
        axes[1].scatter(
            rng.normal(index, 0.045, len(values)), values,
            s=6, color=color, alpha=0.17, linewidth=0,
        )
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axes[1].plot(
            [index, index], [q1, q3], color=DARK, lw=3.0,
            solid_capstyle="round",
        )
        axes[1].scatter(
            index, med, s=20, facecolor="white",
            edgecolor=DARK, linewidth=0.6, zorder=3,
        )
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels([v.capitalize() for v in chronology_order])
    axes[1].set_ylabel("Task–recharge overlap (h yr$^{-1}$)")
    add_panel_labels(fig, axes)
    export(fig, output, "fig_v40_ed3_numerical_qa")
    save_source(chronology, source_dir, "ed3_chronology_conservation.csv")
    save_source(maxima, source_dir, "ed3_maximum_numerical_errors.csv")
    save_source(overlap_long, source_dir, "ed3_task_recharge_overlap.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output", type=Path, default=HERE / "main_figures")
    parser.add_argument("--supp-output", type=Path, default=HERE / "supplement")
    parser.add_argument("--source-output", type=Path, default=HERE / "source_data/figure_source_data")
    parser.add_argument(
        "--china-only",
        action="store_true",
        help="Rebuild only source-aware China Fig. 6 and Extended Data Fig. 7.",
    )
    args = parser.parse_args()
    style()
    if args.china_only:
        figure6(args.output, args.source_output, args.results)
        supplementary_china_dispatch_validation(
            args.supp_output,
            args.source_output,
            args.results,
        )
        return
    runner = load_runner()
    panel = pd.read_csv(args.panel)
    results = pd.read_csv(
        args.results / "configuration_results_source_aware_eur.csv.gz"
    ).rename(
        columns={
            "annual_fixed_cost_eur_per_kw_year_source_aware":
                "annual_fixed_cost_eur_per_kw_year",
            "storage_vom_eur_per_kw_year_source_aware":
                "storage_vom_eur_per_kw_year",
            "turnover_net_benefit_eur_per_kw_year_source_aware":
                "turnover_net_benefit_eur_per_kw_year",
            "joint_net_benefit_eur_per_kw_year_source_aware":
                "joint_net_benefit_eur_per_kw_year",
        }
    )
    # Retain the richer physical paired ledger from v39 and replace only its
    # monetary delta with the independently re-costed v40 value.
    paired = pd.read_csv(
        V39_RESULTS / "paired_configuration_changes.csv.gz"
    )
    paired_money = pd.read_csv(
        args.results / "paired_configuration_changes_source_aware_eur.csv.gz"
    )[
        [
            "country",
            "year",
            "config_id",
            "delta_joint_net_benefit_eur_per_kw_year_source_aware",
        ]
    ]
    paired = paired.drop(
        columns=["delta_joint_net_benefit_eur_per_kw_year"],
        errors="ignore",
    ).merge(
        paired_money,
        on=["country", "year", "config_id"],
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "delta_joint_net_benefit_eur_per_kw_year_source_aware":
                "delta_joint_net_benefit_eur_per_kw_year"
        }
    )
    summary = pd.read_csv(V39_RESULTS / "paired_configuration_summary.csv")
    summary_money = pd.read_csv(
        args.results / "paired_configuration_summary_source_aware_eur.csv"
    )[
        ["config_id", "source_aware_delta_mean_eur_per_kw_year"]
    ]
    summary = summary.drop(
        columns=["delta_joint_net_benefit_eur_per_kw_year_mean"],
        errors="ignore",
    ).merge(
        summary_money,
        on="config_id",
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "source_aware_delta_mean_eur_per_kw_year":
                "delta_joint_net_benefit_eur_per_kw_year_mean"
        }
    )
    envelope = pd.read_csv(
        args.results / "value_sensitivity_envelope_source_aware_eur.csv.gz"
    ).rename(
        columns={
            "net_benefit_eur_per_kw_year_source_aware":
                "net_benefit_eur_per_kw_year",
            "energy_value_cny_mwh_v39_assumption": "energy_value_cny_mwh",
            "firm_value_cny_kw_year_v39_assumption":
                "firm_value_cny_kw_year",
            "charge_price_cny_mwh_v39_assumption":
                "charge_price_cny_mwh",
        }
    )
    gap_summary = pd.read_csv(
        args.results
        / "value_sensitivity_gap_summary_source_aware_eur.csv"
    )
    chronology = pd.read_csv(V39_RESULTS / "chronology_conservation.csv")
    if (results["duration_h"] <= 0).any():
        raise ValueError("Log-duration figures require strictly positive durations.")
    daily, stats = geometry_stats(panel)
    save_source(daily, args.source_output, "europe_daily_task_recharge.csv")
    save_source(stats, args.source_output, "europe_country_year_event_geometry.csv")
    figure1(args.output)
    figure2(panel, daily, stats, args.map, runner, args.output, args.source_output)
    figure3(results, paired, summary, args.output, args.source_output)
    figure4(panel, results, paired, runner, args.output, args.source_output)
    figure5(results, envelope, gap_summary, args.output, args.source_output)
    figure6(args.output, args.source_output, args.results)
    supplementary_figures(
        panel,
        results,
        paired,
        summary,
        gap_summary,
        chronology,
        args.supp_output,
        args.source_output,
    )
    supplementary_china_event_sensitivity(
        args.supp_output,
        args.source_output,
    )
    supplementary_china_dispatch_validation(
        args.supp_output,
        args.source_output,
        args.results,
    )
    manifest = {
        "status": "pass",
        "main_figures": 6,
        "extended_data_figures_built_by_v40_script": 6,
        "extended_data_figures_total_in_manuscript": 7,
        "extended_data_figure_1_builder":
            "revision_nature_energy_v40/ed1_cost_paths_53/"
            "build_plot_ed1_cost_paths_53.py",
        "countries": int(panel["country"].nunique()),
        "country_years": int(panel[["country", "year"]].drop_duplicates().shape[0]),
        "country_hours": int(len(panel)),
        "configurations": int(results["config_id"].nunique()),
        "map": str(args.map),
        "reporting_currency": "EUR",
        "technology_cost_routes": {
            "LFP_VRFB": "CNY_to_EUR",
            "FeCr_gravity_molten_salt_A-CAES_LAES_H2": "USD_to_EUR",
        },
        "derived_cny_equivalent_field_rule": (
            "Fields named *_cny_equivalent_backconverted_from_eur* are "
            "reporting equivalents derived from source-aware EUR results; "
            "they are not native-CNY model outputs."
        ),
        "cny_per_eur": CNY_PER_EUR,
        "usd_per_eur": 1.13,
        "fx_reference_year": FX_REFERENCE_YEAR,
        "fx_series_key": FX_SERIES_KEY,
    }
    (args.source_output / "figure_build_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
