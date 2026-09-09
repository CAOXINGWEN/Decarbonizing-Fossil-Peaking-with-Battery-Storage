#!/usr/bin/env python3
"""Build the evidence-locked v43 Nature Energy figure package.

The script deliberately separates three evidence classes.

1. Figure 1 is a vector scientific-hypothesis illustration.
2. Figure 2 and ED1--2 use frozen European inputs that remain unchanged.
3. Figure 3--6 and ED3--9 read the registered LFP+LDES portfolio outputs and
   frozen v43 experiment tables; ED10 is the separate ERAA system-consistency
   screen. Missing v43 outputs
   are recorded as pending and are never replaced by v40/v41 numerical
   results.

Run from any directory:

    python plot_v43_figures.py

Use ``--strict`` after the portfolio post-processing finishes to require every
main and Extended Data figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)
from PIL import Image


HERE = Path(__file__).resolve().parent
V43 = HERE.parent
MUSCRIPT = V43.parent
V41 = MUSCRIPT / "revision_nature_energy_v41"
V40 = MUSCRIPT / "revision_nature_energy_v40"
V41_SOURCE = V41 / "source_data" / "figure_source_data"
V40_SOURCE = V40 / "source_data" / "figure_source_data"
V43_SOURCE = V43 / "source_data"
EU = HERE.parents[5] / "dataset" / "external_hourly" / "europe_mid_duration_valley_upgrade_v0_1"
MAP_FILE = EU / "raw" / "gisco_countries_2024" / "CNTR_RG_20M_2024_4326.geojson"
ERAA_METRICS = (
    EU
    / "model_results"
    / "eraa_full_network_v0_2_6_WS01_dispersed_caes24_costpilot"
    / "run_metrics.csv"
)
ERAA_FIRST_STAGE = ERAA_METRICS.parent / "first_stage.csv"
OBSERVED_EVENT_TASK = (
    EU
    / "processed"
    / "observed_gas_event_task_definitions_v0_2.csv.gz"
)

MAIN_OUT = HERE / "main_figures"
ED_OUT = HERE / "extended_data"
SOURCE_OUT = HERE / "source_data"
MANIFEST_OUT = HERE / "manifests"

FIG_WIDTH_MM = 182.88
WIDTH = FIG_WIDTH_MM / 25.4
VECTOR_EXPORT_EXTENSIONS = (".pdf", ".svg")
RASTER_EXPORT_EXTENSIONS = (".png", ".tiff")
SUBMISSION_RASTER_DPI = 600

INK = "#233746"
MUTED = "#697980"
GRID = "#DFE7EA"
PALE = "#F4F7F7"
DISPERSED = "#35618D"
CLUSTERED = "#D8662E"
RECHARGE = "#158F83"
PURPLE = "#7056A3"
GOLD = "#D39B25"
RED = "#BA4A52"
BLUE = "#2F78A8"
VALLEY = "#E9D9CE"

TECH_ORDER = [
    "vrfb",
    "fecr_flow",
    "gravity",
    "molten_salt_tes",
    "caes",
    "laes",
    "h2_cavern",
]
TECH_COLOR = {
    "li_lfp": "#2878B5",
    "lfp": "#2878B5",
    "vrfb": "#4C9FD3",
    "fecr_flow": "#B66A9D",
    "gravity": "#6F787D",
    "molten_salt_tes": "#D99A22",
    "caes": "#16886F",
    "laes": "#C65B36",
    "h2_cavern": "#7256A6",
}
TECH_LABEL = {
    "li_lfp": "4 h LFP",
    "lfp": "4 h LFP",
    "vrfb": "VRFB",
    "fecr_flow": "Fe–Cr flow",
    "gravity": "Gravity",
    "molten_salt_tes": "Thermal",
    "caes": "A-CAES",
    "laes": "LAES",
    "h2_cavern": "Cavern H$_2$",
}


@dataclass
class BuildRecord:
    figure_id: str
    status: str
    outputs: list[str]
    sources: list[str]
    note: str = ""


class MissingEvidence(RuntimeError):
    """Raised when a figure cannot be built without inventing evidence."""


BUILD_RECORDS: list[BuildRecord] = []
MISSING_DATA_AUDIT: list[dict[str, object]] = []


def complete_cases(
    data: pd.DataFrame | pd.Series,
    *,
    label: str,
    subset: Sequence[str] | None = None,
) -> pd.DataFrame | pd.Series:
    """Remove missing plotting coordinates while recording the exact count."""

    n_before = int(len(data))
    if isinstance(data, pd.DataFrame):
        result = data.dropna(subset=list(subset) if subset is not None else None)
    else:
        result = data.dropna()
    n_after = int(len(result))
    MISSING_DATA_AUDIT.append(
        {
            "plot_context": label,
            "n_before": n_before,
            "n_after": n_after,
            "excluded_count": n_before - n_after,
        }
    )
    return result


def deterministic_display_offsets(n: int, half_width: float = 0.11) -> np.ndarray:
    """Return reproducible quasi-uniform offsets used only to prevent point overlap."""

    if n <= 1:
        return np.zeros(max(n, 0), dtype=float)
    phase = (np.arange(n, dtype=float) * 0.6180339887498949) % 1.0
    return (2.0 * phase - 1.0) * half_width


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.labelsize": 6.6,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.55,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.2,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_panel(frame: pd.DataFrame, filename: str) -> Path:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    path = SOURCE_OUT / filename
    frame.to_csv(path, index=False)
    return path


def export(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    figure_id: str,
    sources: Sequence[Path],
    note: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix, kwargs in [
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches=None, **kwargs)
        paths.append(path)
    gray = out_dir / f"{stem}_grayscale.png"
    with Image.open(out_dir / f"{stem}.png") as image:
        image.convert("L").save(gray, dpi=(300, 300))
    paths.append(gray)
    plt.close(fig)
    BUILD_RECORDS.append(
        BuildRecord(
            figure_id=figure_id,
            status="built",
            outputs=[str(path.relative_to(HERE)) for path in paths],
            sources=[str(path) for path in sources],
            note=note,
        )
    )


def panel_labels(fig: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    fig.canvas.draw()
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", axes):
        box = ax.get_position()
        fig.text(
            box.x0 - 0.010,
            box.y1 + 0.007,
            label,
            ha="right",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#111111",
        )


def rounded(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    face: str,
    edge: str = "none",
    linewidth: float = 0.0,
    radius: float = 0.01,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MUTED,
    linewidth: float = 0.9,
    mutation: float = 7,
    zorder: int = 8,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=mutation,
        color=color,
        linewidth=linewidth,
        zorder=zorder,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(patch)
    return patch


def chronology_strip(
    ax: plt.Axes,
    x0: float,
    y0: float,
    width: float,
    clustered: bool,
) -> None:
    positions = (
        [0.02, 0.15, 0.32, 0.50, 0.68, 0.86]
        if not clustered
        else [0.02, 0.09, 0.16, 0.56, 0.63, 0.70]
    )
    heights = [0.036, 0.062, 0.047, 0.069, 0.044, 0.057]
    for pos, height in zip(positions, heights):
        ax.add_patch(
            Rectangle(
                (x0 + pos * width, y0),
                0.035 * width,
                height,
                transform=ax.transAxes,
                facecolor=CLUSTERED,
                edgecolor="white",
                linewidth=0.25,
                zorder=6,
            )
        )
    # Renewable-surplus waves reveal the recharge intervals.
    xx = np.linspace(x0, x0 + width, 160)
    wave = y0 - 0.035 + 0.010 * np.sin(np.linspace(0, 9 * np.pi, 160))
    ax.plot(xx, wave, transform=ax.transAxes, color=RECHARGE, lw=1.0, zorder=5)


def duration_axis(ax: plt.Axes, xmax: float = 180) -> None:
    ax.set_xscale("log")
    ax.set_xlim(3.7, xmax)
    ticks = [4, 8, 12, 24, 40, 72, 120, 168]
    if xmax > 180:
        ticks += [240, 336, 480, 720]
    ticks = [tick for tick in ticks if tick <= xmax]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


PORTFOLIO_FILES = {
    "cells": [
        "v42_fixed_scale_mix_cells.csv.gz",
        "v42_fixed_scale_mix_cells.csv",
        "v42_portfolio_cells.csv.gz",
    ],
    "envelope": [
        "v42_fixed_scale_mix_envelope.csv",
        "v42_portfolio_duration_envelope.csv",
    ],
    "valley": [
        "v42_fixed_scale_mix_valley_records.csv",
        "v42_portfolio_valley_records.csv",
    ],
    "valley_pairs": ["v42_fixed_scale_mix_valley_pairs.csv"],
    "layers": [
        "v42_fixed_scale_mix_layer_metrics.csv",
        "v42_portfolio_layer_metrics.csv",
    ],
    "components": [
        "v42_fixed_scale_mix_value_components.csv",
        "v42_portfolio_value_components.csv",
    ],
    "paired": [
        "v42_fixed_scale_mix_envelope_paired_effects.csv",
        "v42_fixed_scale_mix_paired_effects.csv",
        "v42_portfolio_paired_effects.csv",
    ],
    "config_paired": [
        "v42_fixed_scale_mix_config_paired_effects.csv.gz",
        "v42_fixed_scale_mix_config_paired_effects.csv",
    ],
    "all_cells": [
        "v42_all_cases_mix_efc_discrete_scenario_cells.csv.gz",
        "v42_all_cases_mix_efc_discrete_scenario_cells.csv",
    ],
    "all_envelope": [
        "v42_fixed_scale_mix_efc_discrete_scenario_envelope.csv.gz",
        "v42_fixed_scale_mix_efc_discrete_scenario_envelope.csv",
    ],
    "all_config_paired": [
        "v42_all_cases_mix_efc_discrete_scenario_config_paired_effects.csv.gz",
        "v42_all_cases_mix_efc_discrete_scenario_config_paired_effects.csv",
    ],
    "focal": ["v42_fixed_scale_mix_focal_config_summary.csv"],
    "duration_summary": ["v42_fixed_scale_mix_duration_summary.csv"],
    "bootstrap": [
        "v43_figure5b_duration_envelope_bootstrap_10000.csv",
        "v42_fixed_scale_mix_bootstrap_summary.csv",
    ],
    "primary_shapes": [
        "v42_fixed_scale_mix_efc_primary_scenario_valley_shape_summary.csv"
    ],
    "value_scan_curve": [
        "v42_fixed_scale_mix_efc_value_scan_formation_rate_curve.csv"
    ],
    "value_scan_paired": [
        "v42_fixed_scale_mix_efc_value_scan_paired_formation_rate_curve.csv"
    ],
    "value_scan_records": [
        "v42_fixed_scale_mix_efc_value_scan_valley_records.csv.gz",
        "v42_fixed_scale_mix_efc_value_scan_valley_records.csv",
    ],
    "value_scan_duration": [
        "v42_fixed_scale_mix_efc_value_scan_duration_summary.csv.gz",
        "v42_fixed_scale_mix_efc_value_scan_duration_summary.csv",
    ],
    "firm_thresholds": [
        "v42_fixed_scale_mix_efc_value_scan_minimum_firm_thresholds.csv"
    ],
    "cost_firm_phase": [
        "v42_fixed_scale_mix_efc_cost_firm_phase_formation_rate.csv"
    ],
    "cost_firm_paired": [
        "v42_fixed_scale_mix_efc_cost_firm_phase_paired_formation_rate.csv"
    ],
    "cost_thresholds": [
        "v42_fixed_scale_mix_efc_cost_firm_phase_first_cost_thresholds.csv"
    ],
    "recharge_summary": [
        "v43_recharge_multiplier_paired_summary_10000.csv",
        "v42_recharge_multiplier_paired_summary.csv",
        "v42_s20_recharge_multiplier_paired_summary.csv",
    ],
}

SEARCH_ROOTS = [
    V43 / "source_data",
    V43 / "source_data_s20_central_interim",
    V43 / "postprocess",
    V43 / "portfolio_results",
    V43 / "portfolio_model",
    HERE / "input",
]


def find_portfolio_source(key: str, required: bool = True) -> Path | None:
    candidates = PORTFOLIO_FILES[key]
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for filename in candidates:
            direct = root / filename
            if direct.exists():
                return direct
            matches = sorted(root.rglob(filename))
            if matches:
                # Smoke-test products are never publication evidence.
                matches = [
                    item
                    for item in matches
                    if "smoke" not in "/".join(part.lower() for part in item.parts)
                ]
                if matches:
                    return matches[0]
    if required:
        raise MissingEvidence(
            f"New v42 fixed-scale portfolio source is missing: {candidates[0]}"
        )
    return None


def read_portfolio(key: str, required: bool = True) -> tuple[pd.DataFrame, Path | None]:
    path = find_portfolio_source(key, required=required)
    if path is None:
        return pd.DataFrame(), None
    return pd.read_csv(path), path


ALIASES = {
    "country": ["country", "country_code", "zone"],
    "year": ["year", "weather_year", "climate_year"],
    "chronology": ["chronology", "ordering", "sequence", "chronology_id"],
    "duration_h": ["duration_h", "storage_duration_h", "ldes_duration_h"],
    "technology": [
        "technology",
        "tech_id",
        "ldes_technology",
        "dominant_technology",
        "dominant_tech_id",
    ],
    "mix_benefit": [
        "mix_benefit_eur_per_kw_common_power_year",
        "mix_benefit_eur_per_kw_year",
        "benefit_eur_per_kw_year",
        "net_benefit_eur_per_kw_year",
        "best_mix_benefit_eur_per_kw_common_power_year",
        "best_mix_benefit_eur_per_kw_year",
    ],
    "mix_benefit_task": [
        "mix_benefit_eur_per_mwh_task",
        "benefit_eur_per_mwh_task",
    ],
    "strict_valley": ["strict_valley", "has_strict_valley"],
    "left_boundary_h": [
        "left_boundary_h",
        "left_positive_duration_h",
        "h_t",
    ],
    "right_boundary_h": [
        "right_boundary_h",
        "right_reentry_duration_h",
        "h_i",
    ],
    "valley_width_h": ["valley_width_h", "gap_width_h", "width_h"],
    "valley_class": ["valley_class", "transition_class"],
    "model_mode": ["model_mode", "capacity_mode", "scale_mode"],
    "base_cycles": [
        "base_lfp_equivalent_full_cycles_rated",
        "base_lfp_efc",
        "lfp_equivalent_full_cycles_rated",
        "base_equivalent_full_cycles",
    ],
    "tail_cycles": [
        "ldes_equivalent_full_cycles_rated",
        "tail_equivalent_full_cycles_rated",
        "ldes_efc",
        "tail_equivalent_full_cycles",
    ],
    "base_task": [
        "base_lfp_task_service_mwh",
        "base_lfp_task_supply_mwh",
        "base_task_service_mwh",
        "base_task_mwh",
        "base_discharge_mwh",
    ],
    "tail_task": [
        "ldes_task_service_mwh",
        "tail_task_service_mwh",
        "ldes_task_supply_mwh",
        "tail_task_mwh",
        "tail_discharge_mwh",
    ],
    "base_charge": [
        "base_lfp_charge_grid_mwh",
        "base_lfp_charge_mwh",
        "base_charge_mwh",
    ],
    "tail_charge": [
        "ldes_charge_grid_mwh",
        "ldes_charge_mwh",
        "tail_charge_mwh",
    ],
    "soc_recovery": [
        "ldes_mean_soc_recovery_fraction",
        "mean_soc_recovery_fraction",
    ],
    "charge_saturation_h": [
        "ldes_charge_power_saturation_hours",
        "charge_power_saturation_hours",
        "tail_charge_power_binding_hours",
    ],
    "unabsorbed_recharge": [
        "unused_potential_recharge_mwh",
        "unabsorbed_recharge_mwh",
    ],
    "shared_binding_h": ["shared_recharge_binding_hours"],
    "energy_value": [
        "energy_displacement_value_eur_yr",
        "energy_substitution_value_eur_yr",
    ],
    "firm_value": [
        "task_firm_supply_value_eur_yr",
        "capacity_value_eur_yr",
    ],
    "fixed_cost": [
        "annualized_fixed_cost_eur_yr",
        "storage_fixed_cost_eur_yr",
    ],
    "charge_cost": [
        "charging_opportunity_cost_eur_yr",
        "charge_opportunity_cost_eur_yr",
    ],
    "variable_cost": ["variable_om_eur_yr", "variable_cost_eur_yr"],
    "energy_price": [
        "energy_value_eur_per_mwh",
        "gas_displacement_value_eur_per_mwh",
        "v_e_eur_mwh",
    ],
    "charge_price": [
        "charge_opportunity_price_eur_per_mwh",
        "charging_price_eur_per_mwh",
        "c_q_eur_mwh",
    ],
    "firm_price": [
        "firm_value_eur_per_kw_year",
        "capacity_value_eur_per_kw_year",
        "v_f_eur_kw_year",
    ],
}


def column(frame: pd.DataFrame, key: str, required: bool = True) -> str | None:
    for name in ALIASES.get(key, [key]):
        if name in frame.columns:
            return name
    if required:
        raise MissingEvidence(f"Required column '{key}' not found in {list(frame.columns)}")
    return None


def canonical_chronology(values: pd.Series) -> pd.Series:
    mapping = {
        "d": "dispersed",
        "disp": "dispersed",
        "dispersed": "dispersed",
        "c": "clustered",
        "clus": "clustered",
        "clustered": "clustered",
    }
    return values.astype(str).str.lower().map(mapping).fillna(values.astype(str).str.lower())


def fixed_primary(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the prespecified equal-branch-power model when modes are pooled."""
    mode = column(frame, "model_mode", required=False)
    if mode and frame[mode].astype(str).str.lower().eq("fixed").any():
        return frame[frame[mode].astype(str).str.lower().eq("fixed")].copy()
    return frame.copy()


def central_s20(frame: pd.DataFrame, fixed_only: bool = True) -> pd.DataFrame:
    """Select the registered S20 central physical case when pooled."""
    data = fixed_primary(frame) if fixed_only else frame.copy()
    if "source_label" in data.columns:
        mask = data["source_label"].astype(str).str.lower().str.contains("s20")
        if mask.any():
            data = data[mask].copy()
    return data


def mark_pending(figure_id: str, message: str) -> None:
    BUILD_RECORDS.append(
        BuildRecord(
            figure_id=figure_id,
            status="pending_new_combo_source_data",
            outputs=[],
            sources=[],
            note=message,
        )
    )


def figure1() -> None:
    """Continuous hypothesis illustration with a common 4 h LFP layer."""
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
    fig.subplots_adjust(left=0.010, right=0.992, bottom=0.025, top=0.985)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    field = Polygon(
        [
            (0.015, 0.93),
            (0.26, 0.95),
            (0.52, 0.91),
            (0.74, 0.82),
            (0.98, 0.87),
            (0.98, 0.29),
            (0.76, 0.30),
            (0.52, 0.36),
            (0.26, 0.37),
            (0.035, 0.34),
        ],
        transform=ax.transAxes,
        closed=True,
        facecolor="#F1F6F6",
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(field)

    # 1. Conserved paired chronology.
    ax.text(
        0.035,
        0.907,
        "Conserved task days and annual energy",
        transform=ax.transAxes,
        fontsize=6.6,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.035,
        0.818,
        "dispersed",
        transform=ax.transAxes,
        fontsize=5.9,
        fontweight="bold",
        color=DISPERSED,
    )
    chronology_strip(ax, 0.035, 0.742, 0.275, clustered=False)
    ax.text(
        0.035,
        0.603,
        "clustered",
        transform=ax.transAxes,
        fontsize=5.9,
        fontweight="bold",
        color=CLUSTERED,
    )
    chronology_strip(ax, 0.035, 0.527, 0.275, clustered=True)
    arrow(ax, (0.324, 0.737), (0.324, 0.565), CLUSTERED, 0.95, 7)
    ax.text(
        0.166,
        0.440,
        "same blocks · recovery windows contract",
        transform=ax.transAxes,
        ha="center",
        fontsize=5.4,
        color=MUTED,
    )

    # 2. Every comparison shares the same intraday LFP base.
    arrow(ax, (0.350, 0.675), (0.385, 0.675), MUTED)
    ax.text(
        0.390,
        0.920,
        "Common 4 h LFP base\nwith mutually exclusive tail arms",
        transform=ax.transAxes,
        fontsize=6.1,
        fontweight="bold",
        color=INK,
        linespacing=1.05,
    )
    rounded(ax, (0.405, 0.487), 0.250, 0.108, "#E0ECF4", BLUE, 0.8, 0.014, 4)
    for idx in range(7):
        ax.add_patch(
            Rectangle(
                (0.423 + 0.030 * idx, 0.506),
                0.022,
                0.070,
                transform=ax.transAxes,
                facecolor=BLUE if idx < 5 else "#A8CBDD",
                edgecolor="white",
                linewidth=0.30,
                zorder=6,
            )
        )
    ax.text(
        0.530,
        0.458,
        "shared 4 h LFP base",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.1,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        0.530,
        0.430,
        "identical total branch power in both comparator arms",
        transform=ax.transAxes,
        ha="center",
        fontsize=5.1,
        color=MUTED,
    )
    # Comparator branch: the common base is paired either with extra 4 h LFP
    # or with one member of the LDES candidate set.
    rounded(
        ax,
        (0.392, 0.660),
        0.052,
        0.102,
        mpl.colors.to_rgba(BLUE, 0.11),
        BLUE,
        0.65,
        0.006,
        6,
    )
    ax.text(
        0.418,
        0.642,
        "ARM A",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.0,
        color=BLUE,
        fontweight="bold",
    )
    for idx in range(3):
        ax.add_patch(
            Rectangle(
                (0.400 + idx * 0.012, 0.680),
                0.009,
                0.050,
                transform=ax.transAxes,
                facecolor=BLUE,
                edgecolor="white",
                linewidth=0.25,
                zorder=7,
            )
        )
    ax.text(
        0.418,
        0.781,
        "extra\n4 h LFP",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.0,
        fontweight="bold",
        color=BLUE,
        linespacing=0.9,
    )
    ax.add_patch(
        Circle(
            (0.452, 0.710),
            0.017,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=MUTED,
            lw=0.55,
            zorder=8,
        )
    )
    ax.text(
        0.452,
        0.710,
        "OR",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.0,
        color=INK,
        fontweight="bold",
        zorder=9,
    )
    xchips = np.linspace(0.475, 0.655, 7)
    for x, tech in zip(xchips, TECH_ORDER):
        colour = TECH_COLOR[tech]
        rounded(
            ax,
            (x - 0.021, 0.660),
            0.042,
            0.102,
            mpl.colors.to_rgba(colour, 0.11),
            colour,
            0.65,
            0.006,
            6,
        )
        # Energy-tank fill communicates increasing tail inventory.
        fill = 0.021 + 0.006 * TECH_ORDER.index(tech)
        ax.add_patch(
            Rectangle(
                (x - 0.014, 0.674),
                0.028,
                fill,
                transform=ax.transAxes,
                facecolor=colour,
                edgecolor="none",
                alpha=0.78,
                zorder=7,
            )
        )
        label_y = 0.780 + (0.021 if TECH_ORDER.index(tech) % 2 else 0)
        short_label = {
            "molten_salt_tes": "TES",
            "h2_cavern": "H$_2$",
        }.get(tech, TECH_LABEL[tech].replace(" flow", "").replace("Cavern ", ""))
        ax.text(
            x,
            label_y,
            short_label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.0,
            fontweight="bold",
            color=colour,
        )
    ax.text(
        0.565,
        0.835,
        "one of seven candidate LDES tail layers",
        transform=ax.transAxes,
        ha="center",
        fontsize=5.3,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.565,
        0.642,
        "ARM B",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.0,
        color=PURPLE,
        fontweight="bold",
    )

    # 3. Mechanism: joint dispatch and replenishment.
    arrow(ax, (0.670, 0.675), (0.700, 0.675), MUTED)
    ax.text(
        0.715,
        0.920,
        "Recovery windows meet\nthe charging interface",
        transform=ax.transAxes,
        fontsize=6.1,
        fontweight="bold",
        color=INK,
        linespacing=1.05,
    )
    for y, colour, label in [
        (0.785, BLUE, "additional LFP tail"),
        (0.675, RECHARGE, "candidate LDES tail"),
    ]:
        ax.plot(
            [0.720, 0.765, 0.805],
            [y, y - 0.016, y - 0.005],
            transform=ax.transAxes,
            color=colour,
            lw=1.6,
        )
        ax.add_patch(
            Arc(
                (0.835, y - 0.003),
                0.055,
                0.070,
                theta1=30,
                theta2=330,
                transform=ax.transAxes,
                color=colour,
                lw=1.1,
            )
        )
        arrow(ax, (0.849, y + 0.025), (0.857, y + 0.012), colour, 0.8, 5)
        ax.text(
            0.875,
            y - 0.004,
            label,
            transform=ax.transAxes,
            va="center",
            fontsize=5.2,
            color=colour,
            fontweight="bold",
        )
    ax.text(
        0.720,
        0.565,
        "clustering compresses cycling return at intermediate durations\n"
        "while longer coverage can recover value",
        transform=ax.transAxes,
        fontsize=5.3,
        color=MUTED,
        linespacing=1.2,
    )

    # 4. Conditional valley hypothesis, connected rather than boxed.
    x = np.linspace(0.700, 0.965, 260)
    u = (x - x.min()) / (x.max() - x.min())
    dispersed = 0.390 - 0.060 * u + 0.025 * np.sin(1.2 * np.pi * u)
    clustered = (
        0.365
        - 0.075 * u
        - 0.100 * np.exp(-((u - 0.50) / 0.16) ** 2)
        + 0.120 / (1 + np.exp(-(u - 0.76) * 24))
    )
    zero = 0.275
    ax.plot(x, dispersed, transform=ax.transAxes, color=DISPERSED, lw=1.55, zorder=7)
    ax.plot(x, clustered, transform=ax.transAxes, color=CLUSTERED, lw=1.55, zorder=7)
    ax.plot(
        [0.700, 0.968],
        [zero, zero],
        transform=ax.transAxes,
        color="#758086",
        lw=0.7,
        zorder=3,
    )
    ax.fill_between(
        x,
        clustered,
        zero,
        where=clustered < zero,
        transform=ax.transAxes,
        color=VALLEY,
        alpha=0.85,
        linewidth=0,
        zorder=2,
    )
    ax.text(
        0.704,
        0.420,
        "conditional mix-value hypothesis",
        transform=ax.transAxes,
        fontsize=5.7,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.818,
        0.246,
        "mid-duration valley",
        transform=ax.transAxes,
        ha="center",
        fontsize=5.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.818,
        0.222,
        "only if short and longer durations both re-enter",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.0,
        color=MUTED,
    )
    ax.text(
        0.963,
        0.285,
        "long-duration\nre-entry",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.1,
        color=PURPLE,
    )
    arrow(ax, (0.700, 0.190), (0.967, 0.190), INK, 0.75, 6)
    ax.text(
        0.835,
        0.162,
        "LDES duration",
        transform=ax.transAxes,
        ha="center",
        fontsize=5.3,
        fontweight="bold",
        color=INK,
    )

    # Narrow evidence ribbon. It states design facts, not results.
    rounded(ax, (0.035, 0.055), 0.610, 0.185, "white", GRID, 0.65, 0.014, 2)
    nodes = [
        (0.105, "15 countries", "132 country-years", "#2E83A5"),
        (0.255, "paired days", "conserved chronology", DISPERSED),
        (
            0.410,
            "49 LDES configurations",
            "+ one 4 h LFP tail comparator",
            RECHARGE,
        ),
        (
            0.570,
            "fixed-scale mixes",
            "LFP base + LFP tail\nversus LFP base + LDES tail",
            PURPLE,
        ),
    ]
    for idx, (xnode, title, subtitle, colour) in enumerate(nodes):
        ax.add_patch(
            Circle(
                (xnode, 0.157),
                0.024,
                transform=ax.transAxes,
                facecolor=mpl.colors.to_rgba(colour, 0.13),
                edgecolor=colour,
                lw=0.75,
                zorder=6,
            )
        )
        if idx == 0:
            ax.add_patch(
                Arc(
                    (xnode, 0.157),
                    0.029,
                    0.038,
                    theta1=0,
                    theta2=360,
                    transform=ax.transAxes,
                    color=colour,
                    lw=0.55,
                    zorder=7,
                )
            )
        elif idx == 1:
            arrow(ax, (xnode - 0.013, 0.162), (xnode + 0.013, 0.152), colour, 0.6, 4)
        elif idx == 2:
            for j, height in enumerate([0.010, 0.017, 0.024, 0.031]):
                ax.add_patch(
                    Rectangle(
                        (xnode - 0.014 + j * 0.008, 0.141),
                        0.005,
                        height,
                        transform=ax.transAxes,
                        facecolor=colour,
                        edgecolor="none",
                        zorder=7,
                    )
                )
        else:
            ax.add_patch(
                Rectangle(
                    (xnode - 0.014, 0.148),
                    0.028,
                    0.017,
                    transform=ax.transAxes,
                    facecolor=BLUE,
                    edgecolor="none",
                    zorder=7,
                )
            )
            ax.add_patch(
                Rectangle(
                    (xnode + 0.003, 0.148),
                    0.011,
                    0.017,
                    transform=ax.transAxes,
                    facecolor=PURPLE,
                    edgecolor="none",
                    zorder=8,
                )
            )
        ax.text(
            xnode,
            0.118 if idx == 3 else 0.112,
            title,
            transform=ax.transAxes,
            ha="center",
            fontsize=5.4,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            xnode,
            0.096 if idx == 3 else 0.083,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top" if idx == 3 else "baseline",
            fontsize=5.0,
            color=MUTED,
            linespacing=1.15,
        )
        if idx < len(nodes) - 1:
            arrow(
                ax,
                (xnode + 0.030, 0.157),
                (nodes[idx + 1][0] - 0.030, 0.157),
                "#AEB9BE",
                0.65,
                5,
            )

    contract = pd.DataFrame(
        {
            "stage": [
                "paired_chronology",
                "common_intraday_layer",
                "candidate_tail_layer",
                "joint_dispatch",
                "conditional_value_test",
            ],
            "fixed": [
                "daily blocks; annual task and recharge",
                "4 h LFP present in every portfolio",
                "technology-specific duration and cost",
                "common task, recharge and cyclic SOC",
                "same common-power normalization",
            ],
            "changed_or_tested": [
                "inter-event ordering",
                "none",
                "LDES family and duration",
                "cycling return and coverage return",
                "strict positive–nonpositive–positive structure",
            ],
        }
    )
    source = save_panel(contract, "figure1_scientific_hypothesis_contract.csv")
    export(
        fig,
        MAIN_OUT,
        "fig_v43_figure1_framework",
        "Figure 1",
        [source],
        "Vector hypothesis figure; no empirical values are encoded.",
    )


def _map_shapes(
    stats: pd.DataFrame,
    metric_column: str = "natural_longest_event_run_days",
) -> tuple[list[Polygon], list[Polygon], list[float], dict[str, tuple[float, float]]]:
    if not MAP_FILE.exists():
        raise MissingEvidence(f"GISCO boundary missing: {MAP_FILE}")
    geo = json.loads(MAP_FILE.read_text(encoding="utf-8"))
    metric = stats.groupby("country")[metric_column].median().to_dict()
    code_map = {"GB": "UK", "GR": "EL"}
    metric = {code_map.get(key, key): value for key, value in metric.items()}
    background: list[Polygon] = []
    selected: list[Polygon] = []
    values: list[float] = []
    centroids: dict[str, tuple[float, float]] = {}
    centroid_size: dict[str, float] = {}
    for feature in geo["features"]:
        code = feature["properties"].get("CNTR_ID")
        geometry = feature.get("geometry")
        if not geometry:
            continue
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else []
        )
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
            if code in metric:
                selected.append(patch)
                values.append(float(metric[code]))
                area_proxy = float(np.ptp(outer[:, 0]) * np.ptp(outer[:, 1]))
                if area_proxy > centroid_size.get(code, -1):
                    centroids[code] = (
                        float(np.median(outer[:, 0])),
                        float(np.median(outer[:, 1])),
                    )
                    centroid_size[code] = area_proxy
            else:
                background.append(patch)
    return background, selected, values, centroids


def map_event_geometry(ax: plt.Axes, stats: pd.DataFrame) -> pd.DataFrame:
    background, selected, values, centroids = _map_shapes(stats)
    vmax = max(15, math.ceil(float(np.nanmax(values))))
    norm = mpl.colors.Normalize(vmin=4, vmax=vmax)
    cmap = mpl.colormaps["YlOrBr"]
    ax.add_collection(
        PatchCollection(
            background,
            facecolor="#F0F3F4",
            edgecolor="white",
            linewidth=0.25,
        )
    )
    collection = PatchCollection(
        selected,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.42,
    )
    collection.set_array(np.asarray(values))
    ax.add_collection(collection)

    country = (
        stats.groupby("country", as_index=False)
        .agg(
            median_longest_event_days=("natural_longest_event_run_days", "median"),
            mean_annual_task_twh=("annual_task_twh", "mean"),
            mean_recharge_to_task_ratio=("recharge_to_task_ratio", "mean"),
            country_years=("year", "nunique"),
        )
    )
    country["gisco_code"] = country["country"].replace({"GB": "UK", "GR": "EL"})
    country["lon"] = country["gisco_code"].map(
        {key: value[0] for key, value in centroids.items()}
    )
    country["lat"] = country["gisco_code"].map(
        {key: value[1] for key, value in centroids.items()}
    )
    valid = complete_cases(
        country, subset=["lon", "lat"], label="figure2_country_map_coordinates"
    )
    size = 10 + 60 * np.sqrt(
        valid["mean_annual_task_twh"]
        / max(float(valid["mean_annual_task_twh"].max()), 1e-12)
    )
    ax.scatter(
        valid["lon"],
        valid["lat"],
        s=size,
        facecolor="none",
        edgecolor=INK,
        linewidth=0.55,
        alpha=0.78,
        zorder=5,
    )
    ax.set_xlim(-12, 31)
    ax.set_ylim(35, 71.5)
    ax.set_aspect(1.13)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([0.04, -0.015, 0.47, 0.026])
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cax.set_xlabel("Median longest natural event (d)", labelpad=1)
    cax.tick_params(labelsize=5, length=1.4)
    ax.text(
        0.98,
        0.035,
        "circle area scales with annual peaking task",
        transform=ax.transAxes,
        ha="right",
        fontsize=5.1,
        color=MUTED,
    )
    return country


def observed_event_geometry() -> pd.DataFrame:
    """Recompute Figure 2 geometry from the registered C1/C3/R2 definitions."""
    if not OBSERVED_EVENT_TASK.exists():
        raise MissingEvidence(
            f"Observed European task definitions missing: {OBSERVED_EVENT_TASK}"
        )
    columns = [
        "country",
        "year",
        "source_day_id",
        "task_c1_mw",
        "task_c3_q75_mw",
        "r2_potential_recharge_mw",
        "event_flag_q75",
    ]
    hourly = pd.read_csv(OBSERVED_EVENT_TASK, usecols=columns)
    annual = (
        hourly.groupby(["country", "year"], as_index=False)
        .agg(
            annual_task_mwh=("task_c1_mw", "sum"),
            annual_event_tail_mwh=("task_c3_q75_mw", "sum"),
            annual_recharge_mwh=("r2_potential_recharge_mw", "sum"),
        )
    )
    daily = (
        hourly.groupby(["country", "year", "source_day_id"], as_index=False)[
            "event_flag_q75"
        ]
        .max()
        .sort_values(["country", "year", "source_day_id"])
    )
    event_rows = []
    for (country, year), group in daily.groupby(["country", "year"], sort=True):
        flags = group["event_flag_q75"].astype(bool).to_numpy()
        adjacent = int(np.sum(flags[:-1] & flags[1:])) if len(flags) > 1 else 0
        longest = 0
        current = 0
        for active in flags:
            current = current + 1 if active else 0
            longest = max(longest, current)
        event_rows.append(
            {
                "country": country,
                "year": year,
                "natural_adjacent_event_pairs": adjacent,
                "natural_longest_event_run_days": longest,
            }
        )
    stats = annual.merge(
        pd.DataFrame(event_rows),
        on=["country", "year"],
        how="left",
        validate="one_to_one",
    )
    stats["annual_task_twh"] = stats["annual_task_mwh"] / 1e6
    stats["annual_event_tail_twh"] = stats["annual_event_tail_mwh"] / 1e6
    stats["annual_recharge_twh"] = stats["annual_recharge_mwh"] / 1e6
    stats["event_task_share_pct"] = (
        100 * stats["annual_event_tail_mwh"] / stats["annual_task_mwh"]
    )
    stats["recharge_to_task_ratio"] = (
        stats["annual_recharge_mwh"] / stats["annual_task_mwh"]
    )
    return stats[
        [
            "country",
            "year",
            "annual_task_twh",
            "annual_event_tail_twh",
            "event_task_share_pct",
            "natural_adjacent_event_pairs",
            "natural_longest_event_run_days",
            "annual_recharge_twh",
            "recharge_to_task_ratio",
        ]
    ].copy()


def figure2() -> None:
    chronology_path = V41_SOURCE / "figure2_panel_c_conserved_chronology.csv"
    if not chronology_path.exists():
        raise MissingEvidence("Conserved paired-day chronology source is missing")
    stats = observed_event_geometry()
    chronology = pd.read_csv(chronology_path)
    recovery_ecdf_path = V43_SOURCE / "v43_figure2f_recovery_window_ecdf.csv"
    recovery_summary_path = (
        V43_SOURCE / "v43_figure2f_recovery_window_summary.csv"
    )
    if not recovery_ecdf_path.exists() or not recovery_summary_path.exists():
        raise MissingEvidence("Frozen v43 recovery-window source is missing")
    recovery_ecdf = pd.read_csv(recovery_ecdf_path)
    recovery_summary = pd.read_csv(recovery_summary_path)

    fig = plt.figure(figsize=(WIDTH, 5.6))
    fig.subplots_adjust(left=0.070, right=0.950, bottom=0.080, top=0.965)
    gs = fig.add_gridspec(
        3,
        12,
        height_ratios=[1.00, 0.82, 0.78],
        width_ratios=[1] * 12,
        hspace=0.64,
        wspace=0.95,
    )
    axa = fig.add_subplot(gs[0:2, 0:6])
    axb = fig.add_subplot(gs[0, 6:12])
    axc = fig.add_subplot(gs[1, 6:12])
    axd = fig.add_subplot(gs[2, 0:4])
    axe = fig.add_subplot(gs[2, 4:8])
    axf = fig.add_subplot(gs[2, 8:12])
    axes = [axa, axb, axc, axd, axe, axf]

    country = map_event_geometry(axa, stats)

    x = stats["annual_task_twh"].to_numpy(float)
    y = stats["natural_longest_event_run_days"].to_numpy(float)
    ratio = stats["recharge_to_task_ratio"].to_numpy(float)
    if np.any(x <= 0) or np.any(ratio <= 0):
        raise MissingEvidence("Figure 2 log-scale inputs contain non-positive values")
    colour = np.log10(ratio)
    scatter = axb.scatter(
        x,
        y,
        c=colour,
        cmap="viridis",
        s=18,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.30,
    )
    axb.set_xscale("log")
    rho = stats["annual_task_twh"].corr(
        stats["natural_longest_event_run_days"], method="spearman"
    )
    axb.text(
        0.03,
        0.95,
        f"Spearman $\\rho$ = {rho:.2f}",
        transform=axb.transAxes,
        va="top",
        fontsize=5.4,
    )
    axb.set_xlabel("Annual peaking task (TWh)")
    axb.set_ylabel("Longest natural event (d)")
    cbar = fig.colorbar(scatter, ax=axb, fraction=0.035, pad=0.02)
    cbar.set_label("log$_{10}$(recharge / task)", labelpad=2)

    days = chronology["day"].to_numpy(float)
    for name, base, colour in [
        ("dispersed", 1.0, DISPERSED),
        ("clustered", 0.0, CLUSTERED),
    ]:
        task = chronology[f"{name}_task_normalized"].to_numpy(float)
        recharge = chronology[f"{name}_recharge_normalized"].to_numpy(float)
        axc.fill_between(
            days,
            base,
            base + 0.42 * task,
            color=colour,
            alpha=0.88,
            linewidth=0,
        )
        axc.plot(days, base - 0.25 * recharge, color=RECHARGE, lw=0.62)
        axc.text(
            -8,
            base + 0.14,
            name,
            ha="right",
            va="center",
            color=colour,
            fontsize=5.5,
            fontweight="bold",
        )
    axc.set_xlim(-28, 365)
    axc.set_ylim(-0.31, 1.48)
    axc.set_yticks([])
    axc.set_xticks([0, 90, 181, 273, 364])
    axc.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"])
    axc.set_xlabel("Conserved reordering of identical daily blocks")
    axc.text(
        0.01,
        0.95,
        "task",
        transform=axc.transAxes,
        color=CLUSTERED,
        fontsize=5.2,
        fontweight="bold",
    )
    axc.text(
        0.10,
        0.95,
        "potential recharge",
        transform=axc.transAxes,
        color=RECHARGE,
        fontsize=5.2,
        fontweight="bold",
    )
    axc.text(
        0.99,
        0.04,
        "normalized daily amplitude",
        transform=axc.transAxes,
        ha="right",
        fontsize=5.0,
        color=MUTED,
    )

    longest = np.sort(y)
    longest_ecdf = np.arange(1, len(longest) + 1) / len(longest)
    axd.step(longest, longest_ecdf, where="post", color=GOLD)
    axd.axvline(np.median(longest), color=INK, lw=0.65, ls="--")
    axd.text(
        np.median(longest) + 0.35,
        0.12,
        f"median {np.median(longest):.1f} d",
        ha="left",
        fontsize=5.2,
        color=INK,
    )
    axd.set_xlabel("Longest natural event (d)")
    axd.set_ylabel("Cumulative country-years")
    axd.set_ylim(0, 1.02)

    ratio_sorted = np.sort(ratio)
    ratio_ecdf = np.arange(1, len(ratio_sorted) + 1) / len(ratio_sorted)
    axe.step(ratio_sorted, ratio_ecdf, where="post", color=RECHARGE)
    axe.set_xscale("log")
    axe.axvline(np.median(ratio), color=INK, lw=0.65, ls="--")
    axe.text(
        np.median(ratio) * 1.13,
        0.12,
        f"median {np.median(ratio):.1f}",
        ha="left",
        fontsize=5.2,
        color=INK,
    )
    axe.set_xlabel("Annual recharge / peaking task")
    axe.set_ylabel("Cumulative country-years")
    axe.set_ylim(0, 1.02)

    for chronology_name, colour in [
        ("dispersed", DISPERSED),
        ("clustered", CLUSTERED),
    ]:
        group = recovery_ecdf[
            recovery_ecdf["chronology"].eq(chronology_name)
        ].sort_values("recovery_hours")
        axf.step(
            group["recovery_hours"],
            group["ecdf"],
            where="post",
            color=colour,
            label=chronology_name.capitalize(),
        )
    median_row = recovery_summary[
        recovery_summary["metric"].eq("median_recovery_hours")
    ].iloc[0]
    zero_row = recovery_summary[
        recovery_summary["metric"].eq("zero_recovery_window_share")
    ].iloc[0]
    recharge_window_row = recovery_summary[
        recovery_summary["metric"].eq(
            "median_available_recharge_share_of_annual"
        )
    ].iloc[0]
    axf.axvline(
        median_row["dispersed_country_year_median"],
        color=DISPERSED,
        lw=0.65,
        ls="--",
    )
    axf.text(
        0.98,
        0.22,
        (
            "median 72 → 0 h\n"
            f"zero-window share 0 → "
            f"{100 * zero_row['clustered_country_year_median']:.1f}%\n"
            "recharge/window "
            f"{100 * recharge_window_row['dispersed_country_year_median']:.2f}"
            " → 0% annual"
        ),
        transform=axf.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=INK,
    )
    axf.set_xlim(0, 520)
    axf.set_xticks([0, 72, 168, 336, 504])
    axf.set_ylim(0, 1.02)
    axf.set_xlabel("Recovery window (h)")
    axf.set_ylabel("Cumulative event pairs")
    axf.legend(loc="upper right", fontsize=5.0)
    axf.text(
        0.02,
        0.98,
        "4 d event-cluster intervention",
        transform=axf.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        color=MUTED,
    )

    panel_sources = [
        save_panel(country, "figure2_panel_a_country_map_summary.csv"),
        save_panel(stats, "figure2_panel_b_country_year_geometry.csv"),
        save_panel(chronology, "figure2_panel_c_conserved_chronology.csv"),
        save_panel(
            pd.DataFrame({"longest_event_days": longest, "ecdf": longest_ecdf}),
            "figure2_panel_d_event_run_ecdf.csv",
        ),
        save_panel(
            pd.DataFrame({"recharge_to_task_ratio": ratio_sorted, "ecdf": ratio_ecdf}),
            "figure2_panel_e_recharge_ratio_ecdf.csv",
        ),
        save_panel(
            recovery_ecdf,
            "figure2_panel_f_recovery_window_ecdf.csv",
        ),
        save_panel(
            recovery_summary,
            "figure2_panel_f_recovery_window_summary.csv",
        ),
    ]
    panel_labels(fig, axes)
    export(
        fig,
        MAIN_OUT,
        "fig_v43_figure2_europe_event_design",
        "Figure 2",
        [
            OBSERVED_EVENT_TASK,
            chronology_path,
            recovery_ecdf_path,
            recovery_summary_path,
            MAP_FILE,
            *panel_sources,
        ],
        (
            "C1 annual task, C3-q75 event-tail share and R2 recharge are "
            "recomputed from the registered hourly definitions; panel c "
            "uses the frozen conserved-day illustration; panel f uses the "
            "registered four-day recovery-window intervention."
        ),
    )


def _summarize_envelope(envelope: pd.DataFrame) -> pd.DataFrame:
    chronology = column(envelope, "chronology")
    duration = column(envelope, "duration_h")
    benefit = column(envelope, "mix_benefit")
    tech = column(envelope, "technology", required=False)
    data = envelope.copy()
    data["_chronology"] = canonical_chronology(data[chronology])
    data["_duration_h"] = pd.to_numeric(data[duration], errors="coerce")
    data["_benefit"] = pd.to_numeric(data[benefit], errors="coerce")
    summary = (
        complete_cases(
            data,
            subset=["_duration_h", "_benefit"],
            label="duration_envelope_numeric_coordinates",
        )
        .groupby(["_chronology", "_duration_h"])["_benefit"]
        .agg(
            median="median",
            mean="mean",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            n="size",
        )
        .reset_index()
    )
    if tech:
        dominant = (
            complete_cases(
                data,
                subset=["_duration_h"],
                label="duration_envelope_technology_coordinates",
            )
            .groupby(["_chronology", "_duration_h"])[tech]
            .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
            .rename("dominant_technology")
            .reset_index()
        )
        summary = summary.merge(
            dominant, on=["_chronology", "_duration_h"], how="left"
        )
    return summary


def _transition_classes(valley: pd.DataFrame) -> pd.DataFrame:
    country = column(valley, "country")
    year = column(valley, "year")
    chronology = column(valley, "chronology")
    flag = column(valley, "strict_valley")
    data = valley[[country, year, chronology, flag]].copy()
    data["_chronology"] = canonical_chronology(data[chronology])
    data["_flag"] = data[flag].astype(bool)
    wide = data.pivot_table(
        index=[country, year],
        columns="_chronology",
        values="_flag",
        aggfunc="max",
    ).fillna(False)
    for name in ["dispersed", "clustered"]:
        if name not in wide.columns:
            wide[name] = False
    conditions = [
        (~wide["dispersed"]) & wide["clustered"],
        wide["dispersed"] & wide["clustered"],
        wide["dispersed"] & (~wide["clustered"]),
    ]
    labels = ["formed after clustering", "retained", "disappeared"]
    wide["transition"] = np.select(conditions, labels, default="never formed")
    return wide.reset_index()


def figure3() -> None:
    """Joint-portfolio allocation and focal 24 h A-CAES incremental value."""
    cells, cells_path = read_portfolio("cells")
    config_pairs, config_pairs_path = read_portfolio("config_paired")
    cells = central_s20(cells)
    config_pairs = central_s20(config_pairs)
    if "config_id" not in cells.columns:
        raise MissingEvidence("Figure 3 requires candidate config_id")
    focal_id = "caes_24h"
    if not cells["config_id"].eq(focal_id).any():
        raise MissingEvidence("Focal caes_24h rows are absent")
    focal = cells[cells["config_id"].eq(focal_id)].copy()
    focal["_chronology"] = canonical_chronology(focal["chronology"])
    pairs = config_pairs[config_pairs["config_id"].eq(focal_id)].copy()
    if pairs.empty:
        raise MissingEvidence("Paired caes_24h portfolio effects are absent")

    common_kw = focal["common_base_power_kw"].replace(0, np.nan)
    focal["ldes_portfolio_value_eur_kw_yr"] = (
        focal["portfolio_net_value_efc_eur"] / common_kw
    )
    focal["lfp_tail_portfolio_value_eur_kw_yr"] = (
        focal["extra_lfp4_net_value_efc_eur"] / common_kw
    )

    fig = plt.figure(figsize=(WIDTH, 5.2))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.090, top=0.965)
    gs = fig.add_gridspec(
        2, 14, height_ratios=[1.20, 0.94], hspace=0.58, wspace=1.05
    )
    axa = fig.add_subplot(gs[0, 0:8])
    axb = fig.add_subplot(gs[0, 8:14])
    axc = fig.add_subplot(gs[1, 0:4])
    axd = fig.add_subplot(gs[1, 4:9])
    axe = fig.add_subplot(gs[1, 9:14])
    axes = [axa, axb, axc, axd, axe]

    # a: paired portfolio-arm value distributions, with every line preserving
    # the common LFP base and changing only the added tail branch.
    positions = {
        ("dispersed", "LFP tail"): 0,
        ("dispersed", "A-CAES tail"): 1,
        ("clustered", "LFP tail"): 3,
        ("clustered", "A-CAES tail"): 4,
    }
    arm_rows = []
    for chronology_name, chronology_colour in [
        ("dispersed", DISPERSED),
        ("clustered", CLUSTERED),
    ]:
        group = focal[focal["_chronology"].eq(chronology_name)].copy()
        for label, value_col, face in [
            ("LFP tail", "lfp_tail_portfolio_value_eur_kw_yr", "#A8CBDD"),
            ("A-CAES tail", "ldes_portfolio_value_eur_kw_yr", RECHARGE),
        ]:
            values = complete_cases(
                group[value_col],
                label=f"figure3_arm_value_{chronology_name}_{label}",
            ).to_numpy(float)
            xpos = positions[(chronology_name, label)]
            violin = axa.violinplot(
                values,
                positions=[xpos],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(face)
                body.set_edgecolor(chronology_colour)
                body.set_linewidth(0.65)
                body.set_alpha(0.45)
            jitter = deterministic_display_offsets(len(values), half_width=0.11)
            axa.scatter(
                xpos + jitter,
                values,
                s=5.5,
                facecolor=chronology_colour,
                edgecolor="none",
                alpha=0.28,
                zorder=2,
            )
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            axa.vlines(xpos, q25, q75, color=INK, lw=2.0, zorder=4)
            axa.scatter(xpos, median, s=22, color="white", edgecolor=INK, lw=0.6, zorder=5)
            arm_rows.extend(
                {
                    "chronology": chronology_name,
                    "tail_branch": label,
                    "portfolio_value_eur_per_kw_common_power_year": value,
                }
                for value in values
            )
    axa.set_xticks([0, 1, 3, 4])
    axa.set_xticklabels(
        ["LFP tail", "A-CAES tail", "LFP tail", "A-CAES tail"]
    )
    axa.text(0.5, -0.20, "dispersed", transform=axa.get_xaxis_transform(), ha="center", color=DISPERSED, fontweight="bold")
    axa.text(3.5, -0.20, "clustered", transform=axa.get_xaxis_transform(), ha="center", color=CLUSTERED, fontweight="bold")
    axa.set_ylabel("Portfolio net value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axa.grid(axis="y", color=GRID, lw=0.5)
    axa.text(
        0.02,
        0.98,
        "common 4 h LFP base in every arm",
        transform=axa.transAxes,
        va="top",
        fontsize=5.2,
        color=MUTED,
    )

    # b: power, energy and realized task-service allocations.
    allocation = (
        focal.groupby("_chronology")
        .agg(
            tail_power_share=("tail_power_share", "mean"),
            tail_energy_share=("tail_energy_share", "mean"),
            tail_task_share=("tail_discharge_share", "mean"),
        )
        .reindex(["dispersed", "clustered"])
    )
    metrics = ["tail_power_share", "tail_energy_share", "tail_task_share"]
    metric_labels = ["Power", "Energy", "Task service"]
    xloc = np.arange(len(metrics))
    width = 0.34
    for offset, chronology_name, colour in [
        (-width / 2, "dispersed", DISPERSED),
        (width / 2, "clustered", CLUSTERED),
    ]:
        tail = 100 * allocation.loc[chronology_name, metrics].to_numpy(float)
        axb.bar(
            xloc + offset,
            100 - tail,
            width,
            color=mpl.colors.to_rgba(BLUE, 0.42),
            edgecolor=colour,
            linewidth=0.55,
        )
        axb.bar(
            xloc + offset,
            tail,
            width,
            bottom=100 - tail,
            color=mpl.colors.to_rgba(RECHARGE, 0.82),
            edgecolor=colour,
            linewidth=0.55,
            label=chronology_name,
        )
    axb.set_xticks(xloc)
    axb.set_xticklabels(metric_labels)
    axb.set_ylim(0, 100)
    axb.set_ylabel("Portfolio allocation (%)")
    axb.legend(ncol=2, loc="upper center")

    # c: EFC-valued incremental portfolio value, not residual-peak cost.
    benefit_data = [
        complete_cases(
            focal[focal["_chronology"].eq(name)][
                "mix_benefit_eur_per_kw_common_power_year"
            ],
            label=f"figure3_mix_benefit_{name}",
        )
        for name in ["dispersed", "clustered"]
    ]
    box = axc.boxplot(
        benefit_data,
        positions=[0, 1],
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.0},
        whiskerprops={"color": INK, "linewidth": 0.65},
        capprops={"color": INK, "linewidth": 0.65},
        boxprops={"edgecolor": INK, "linewidth": 0.65},
    )
    for patch, colour in zip(box["boxes"], [DISPERSED, CLUSTERED]):
        patch.set_facecolor(mpl.colors.to_rgba(colour, 0.30))
    axc.axhline(0, color=INK, lw=0.65)
    axc.set_xticks([0, 1])
    axc.set_xticklabels(["Dispersed", "Clustered"])
    axc.set_ylabel(
        "A-CAES-tail incremental value\n(EUR kW$^{-1}$ yr$^{-1}$)"
    )

    # d: paired country-year change in the same primary value metric.
    dcol = next(
        name
        for name in [
            "clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year",
            "clustered_minus_dispersed_best_mix_benefit_eur_per_kw_common_power_year",
        ]
        if name in pairs.columns
    )
    deltas = complete_cases(
        pd.to_numeric(pairs[dcol], errors="coerce"),
        label="figure3_country_year_paired_deltas",
    ).sort_values()
    ypos = np.arange(len(deltas))
    axd.hlines(ypos, 0, deltas, color=np.where(deltas < 0, CLUSTERED, RECHARGE), lw=0.75, alpha=0.7)
    axd.scatter(deltas, ypos, c=np.where(deltas < 0, CLUSTERED, RECHARGE), s=8, edgecolor="none")
    axd.axvline(0, color=INK, lw=0.65)
    axd.set_yticks([])
    axd.set_xlabel(
        "Clustered − dispersed incremental value\n"
        "(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axd.set_ylabel("Country-years")
    axd.text(
        0.98,
        0.05,
        f"median {deltas.median():.2f}",
        transform=axd.transAxes,
        ha="right",
        fontsize=5.2,
        color=MUTED,
    )

    # e: equivalent cycles of the alternative branch and EFC-equivalent gain.
    cycle_long = []
    for chronology_name in ["dispersed", "clustered"]:
        group = focal[focal["_chronology"].eq(chronology_name)]
        cycle_long.extend(
            {
                "chronology": chronology_name,
                "branch": branch,
                "equivalent_full_cycles": value,
            }
            for branch, colname in [
                ("LFP tail", "marginal_extra_lfp4_efc"),
                ("A-CAES tail", "tail_equivalent_full_cycles"),
            ]
            for value in complete_cases(
                group[colname],
                label=f"figure3_cycles_{chronology_name}_{branch}",
            )
        )
    cycle_frame = pd.DataFrame(cycle_long)
    positions_e = [0, 1, 3, 4]
    data_e = [
        cycle_frame[
            cycle_frame["chronology"].eq(chronology_name)
            & cycle_frame["branch"].eq(branch)
        ]["equivalent_full_cycles"]
        for chronology_name, branch in [
            ("dispersed", "LFP tail"),
            ("dispersed", "A-CAES tail"),
            ("clustered", "LFP tail"),
            ("clustered", "A-CAES tail"),
        ]
    ]
    bp = axe.boxplot(
        data_e,
        positions=positions_e,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 0.9},
        whiskerprops={"color": INK, "linewidth": 0.6},
        capprops={"color": INK, "linewidth": 0.6},
        boxprops={"edgecolor": INK, "linewidth": 0.6},
    )
    for patch, colour in zip(
        bp["boxes"], [BLUE, RECHARGE, BLUE, RECHARGE]
    ):
        patch.set_facecolor(mpl.colors.to_rgba(colour, 0.35))
    axe.set_xticks(positions_e)
    axe.set_xticklabels(["LFP", "CAES", "LFP", "CAES"])
    axe.text(0.5, -0.24, "dispersed", transform=axe.get_xaxis_transform(), ha="center", color=DISPERSED, fontsize=5.0)
    axe.text(3.5, -0.24, "clustered", transform=axe.get_xaxis_transform(), ha="center", color=CLUSTERED, fontsize=5.0)
    axe.set_ylabel("Equivalent full cycles")
    efc_inset = axe.inset_axes([0.57, 0.55, 0.40, 0.38])
    efc_values = complete_cases(
        focal["mix_efc_gain_vs_extra_lfp4_mw"],
        label="figure3_efc_gain_inset",
    )
    efc_inset.hist(efc_values, bins=18, color=PURPLE, alpha=0.72)
    efc_inset.axvline(0, color=INK, lw=0.6)
    efc_inset.set_xlabel("EFC gain (MW)", fontsize=5.0)
    efc_inset.set_yticks([])
    efc_inset.tick_params(labelsize=5.0)

    panel_sources = [
        save_panel(pd.DataFrame(arm_rows), "figure3_panel_a_portfolio_arm_values.csv"),
        save_panel(allocation.reset_index(), "figure3_panel_b_component_allocation.csv"),
        save_panel(
            focal[
                [
                    "country",
                    "year",
                    "_chronology",
                    "mix_benefit_eur_per_kw_common_power_year",
                ]
            ],
            "figure3_panel_c_mix_advantage.csv",
        ),
        save_panel(pairs, "figure3_panel_d_paired_mix_advantage.csv"),
        save_panel(cycle_frame, "figure3_panel_e_branch_cycles.csv"),
    ]
    panel_labels(fig, axes)
    export(
        fig,
        MAIN_OUT,
        "fig_v43_figure3_joint_portfolio_value",
        "Figure 3",
        [cells_path, config_pairs_path, *panel_sources],
        "Focal A-CAES 24 h fixed-power comparison; primary benefit uses the EFC value ledger.",
    )


def _paired_metric_summary(
    frame: pd.DataFrame,
    metrics: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for key, label in metrics:
        name = column(frame, key, required=False)
        if not name:
            continue
        values = complete_cases(
            pd.to_numeric(frame[name], errors="coerce"),
            label=f"mechanism_summary_{key}",
        )
        if values.empty:
            continue
        rows.append(
            {
                "metric_key": key,
                "metric_label": label,
                "median": values.median(),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
                "mean": values.mean(),
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def figure4() -> None:
    """Annual recharge-throughput mechanism and the S20 recharge intervention."""
    cells, cells_path = read_portfolio("cells")
    config_pairs, config_pairs_path = read_portfolio("config_paired")
    recharge, recharge_path = read_portfolio("recharge_summary")
    cells = central_s20(cells)
    config_pairs = central_s20(config_pairs)
    if "source_label" in recharge.columns:
        recharge = recharge[
            recharge["source_label"].astype(str).str.lower().eq("fixed_s20_central")
        ].copy()
    focal_id = "caes_24h"
    focal = cells[cells["config_id"].eq(focal_id)].copy()
    focal["_chronology"] = canonical_chronology(focal["chronology"])
    pairs = config_pairs[config_pairs["config_id"].eq(focal_id)].copy()
    if focal.empty or pairs.empty:
        raise MissingEvidence("Figure 4 requires central S20 CAES24 rows")

    fig = plt.figure(figsize=(WIDTH, 5.0))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.090, top=0.965)
    gs = fig.add_gridspec(
        2, 12, height_ratios=[1.16, 0.94], hspace=0.58, wspace=1.0
    )
    axa = fig.add_subplot(gs[0, 0:7])
    axb = fig.add_subplot(gs[0, 7:12])
    axc = fig.add_subplot(gs[1, 0:7])
    axd = fig.add_subplot(gs[1, 7:12])
    axes = [axa, axb, axc, axd]

    # a: paired retention of directly recorded tail-layer quantities.
    retention_specs = [
        (
            "tail_charge_mwh_clustered",
            "tail_charge_mwh_dispersed",
            "Charging",
            RECHARGE,
        ),
        (
            "tail_discharge_mwh_clustered",
            "tail_discharge_mwh_dispersed",
            "Task service",
            CLUSTERED,
        ),
        (
            "tail_equivalent_full_cycles_clustered",
            "tail_equivalent_full_cycles_dispersed",
            "Equivalent cycles",
            PURPLE,
        ),
    ]
    retention_rows = []
    for xpos, (numerator, denominator, label, colour) in enumerate(retention_specs):
        valid = complete_cases(
            pairs[[numerator, denominator]],
            label=f"figure4_retention_{label}",
        )
        valid = valid[valid[denominator].abs() > 1e-12]
        values = 100 * valid[numerator] / valid[denominator]
        retention_rows.extend(
            {"metric": label, "clustered_to_dispersed_pct": value}
            for value in values
        )
        violin = axa.violinplot(
            values,
            positions=[xpos],
            widths=0.72,
            showextrema=False,
            showmedians=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(colour)
            body.set_edgecolor(colour)
            body.set_alpha(0.24)
        axa.scatter(
            xpos + deterministic_display_offsets(len(values), half_width=0.11),
            values,
            s=5,
            color=colour,
            alpha=0.28,
            edgecolor="none",
        )
        q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        axa.vlines(xpos, q25, q75, color=INK, lw=2.0)
        axa.scatter(xpos, med, s=22, facecolor="white", edgecolor=INK, lw=0.6, zorder=5)
    axa.axhline(100, color=INK, lw=0.7, ls="--")
    axa.set_xticks(range(3))
    axa.set_xticklabels([item[2] for item in retention_specs])
    axa.set_ylabel("Clustered / dispersed retention (%)")
    axa.grid(axis="y", color=GRID, lw=0.5)

    # b: only directly output constraint-activity diagnostics.
    binding_country = (
        focal.groupby(["country", "_chronology"], as_index=False)
        .agg(
            shared_recharge_binding_hours=(
                "shared_recharge_binding_hours",
                "mean",
            ),
            tail_charge_power_binding_hours=(
                "tail_charge_power_binding_hours",
                "mean",
            ),
        )
    )
    binding = (
        binding_country.groupby("_chronology")
        [
            [
                "shared_recharge_binding_hours",
                "tail_charge_power_binding_hours",
            ]
        ]
        .mean()
        .reindex(["dispersed", "clustered"])
    )
    ypos = np.arange(2)
    for idx, metric in enumerate(binding.columns):
        xvalues = [
            binding.loc["dispersed", metric],
            binding.loc["clustered", metric],
        ]
        axb.plot(xvalues, [idx, idx], color=GRID, lw=2.2, zorder=1)
        axb.scatter(xvalues[0], idx, color=DISPERSED, s=26, label="Dispersed" if idx == 0 else None, zorder=3)
        axb.scatter(xvalues[1], idx, color=CLUSTERED, s=26, label="Clustered" if idx == 0 else None, zorder=3)
        for chronology_index, value in enumerate(xvalues):
            if idx == 0:
                offset = (-7, 8) if chronology_index == 0 else (7, 8)
                horizontal = "right" if chronology_index == 0 else "left"
                vertical = "bottom"
            else:
                offset = (0, 8) if chronology_index == 0 else (0, -13)
                horizontal = "center"
                vertical = "bottom" if chronology_index == 0 else "top"
            axb.annotate(
                f"{value:.1f}",
                (value, idx),
                xytext=offset,
                textcoords="offset points",
                ha=horizontal,
                va=vertical,
                fontsize=5.0,
                color=MUTED,
            )
    axb.set_yticks(ypos)
    axb.set_yticklabels(["", ""])
    axb.set_xscale("log")
    axb.set_xlim(0.7, 800)
    axb.set_xlabel("Binding hours yr$^{-1}$")
    axb.legend(ncol=2, loc="center")
    axb.text(
        0.98,
        0.03,
        "Available recharge",
        transform=axb.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=INK,
    )
    axb.text(
        0.02,
        0.97,
        "Charging interface",
        transform=axb.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        color=INK,
    )

    # c: S20 1×/2×/4× recharge intervention, already country-block summarized.
    ci_low_col = (
        "ci_low" if "ci_low" in recharge else "country_block_ci_low"
    )
    ci_high_col = (
        "ci_high" if "ci_high" in recharge else "country_block_ci_high"
    )
    needed = {
        "analysis",
        "recharge_power_multiplier",
        "config_id",
        "metric",
        "country_block_mean",
        ci_low_col,
        ci_high_col,
    }
    if not needed.issubset(recharge.columns):
        raise MissingEvidence(
            f"Recharge summary is missing columns: {sorted(needed - set(recharge.columns))}"
        )
    response = recharge[
        recharge["analysis"].eq("clustered_minus_dispersed")
        & recharge["config_id"].isin(["caes_24h", "h2_cavern_48h"])
        & recharge["metric"].eq(
            "mix_benefit_eur_per_kw_common_power_year"
        )
    ].copy()
    if response.empty:
        raise MissingEvidence("S20 paired mix-benefit recharge response is absent")
    did = recharge[
        recharge["analysis"].eq("chronology_difference_in_differences")
        & recharge["config_id"].isin(["caes_24h", "h2_cavern_48h"])
        & recharge["metric"].eq(
            "mix_benefit_eur_per_kw_common_power_year"
        )
    ].copy()
    for config_id, colour, label in [
        ("caes_24h", RECHARGE, "A-CAES 24 h"),
        ("h2_cavern_48h", PURPLE, "Cavern H$_2$ 48 h"),
    ]:
        group = response[response["config_id"].eq(config_id)].sort_values(
            "recharge_power_multiplier"
        )
        if group.empty:
            continue
        yerr = np.vstack(
            [
                group["country_block_mean"] - group[ci_low_col],
                group[ci_high_col] - group["country_block_mean"],
            ]
        )
        axc.errorbar(
            group["recharge_power_multiplier"],
            group["country_block_mean"],
            yerr=yerr,
            color=colour,
            marker="o",
            capsize=2,
            label=label,
        )
    axc.axhline(0, color=INK, lw=0.7)
    axc.set_xticks(sorted(response["recharge_power_multiplier"].unique()))
    axc.set_xlabel("Available renewable-surplus multiplier")
    axc.set_ylabel(
        "Clustered − dispersed incremental value\n"
        "(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axc.legend(ncol=2, loc="best")
    if not did.empty:
        maximum_shift = did["country_block_mean"].abs().max()
        axc.text(
            0.98,
            0.05,
            f"largest change from 1×  {maximum_shift:.02f}",
            transform=axc.transAxes,
            ha="right",
            fontsize=5.0,
            color=MUTED,
        )

    # d: connect the paired physical loss to the paired economic response.
    mechanism = pairs[
        [
            "country",
            "year",
            "clustered_minus_dispersed_tail_charge_mwh",
            "clustered_minus_dispersed_tail_equivalent_full_cycles",
            "clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year",
        ]
    ].copy()
    mechanism["tail_charge_change_gwh"] = (
        mechanism["clustered_minus_dispersed_tail_charge_mwh"] / 1000
    )
    xmetric = "clustered_minus_dispersed_tail_equivalent_full_cycles"
    ymetric = "clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year"
    scatter = axd.scatter(
        mechanism[xmetric],
        mechanism[ymetric],
        c=mechanism["tail_charge_change_gwh"],
        cmap="viridis",
        s=12,
        alpha=0.62,
        edgecolor="none",
    )
    valid = complete_cases(
        mechanism[[xmetric, ymetric]],
        label="figure4_recovery_window_regression",
    )
    slope, intercept = np.polyfit(valid[xmetric], valid[ymetric], 1)
    line_x = np.linspace(valid[xmetric].min(), valid[xmetric].max(), 100)
    axd.plot(line_x, slope * line_x + intercept, color=INK, lw=0.85)
    correlation = valid.corr().iloc[0, 1]
    axd.axhline(0, color=GRID, lw=0.65)
    axd.axvline(0, color=GRID, lw=0.65)
    axd.set_xlabel("Clustered − dispersed equivalent cycles")
    axd.set_ylabel("Clustered − dispersed incremental value")
    cax_d = axd.inset_axes([0.15, 0.925, 0.68, 0.035])
    cbar_d = fig.colorbar(scatter, cax=cax_d, orientation="horizontal")
    cbar_d.ax.tick_params(labelsize=5.0, length=1.2)
    cbar_d.ax.set_title("tail-charge change (GWh)", fontsize=5.0, pad=1)
    axd.text(
        0.97,
        0.05,
        f"Pearson r = {correlation:.2f}",
        transform=axd.transAxes,
        ha="right",
        fontsize=5.0,
        color=MUTED,
    )

    panel_sources = [
        save_panel(pd.DataFrame(retention_rows), "figure4_panel_a_tail_retention.csv"),
        save_panel(binding.reset_index(), "figure4_panel_b_binding_hours.csv"),
        save_panel(response, "figure4_panel_c_recharge_multiplier_response.csv"),
        save_panel(did, "figure4_panel_c_recharge_difference_in_differences.csv"),
        save_panel(mechanism, "figure4_panel_d_physical_economic_link.csv"),
    ]
    panel_labels(fig, axes)
    export(
        fig,
        MAIN_OUT,
        "fig_v43_figure4_shared_recharge_mechanism",
        "Figure 4",
        [cells_path, config_pairs_path, recharge_path, *panel_sources],
        "Central S20 only; no unexported event-level SOC mediator is plotted.",
    )


def _phase_grid(cells: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    energy_price = column(cells, "energy_price")
    charge_price = column(cells, "charge_price")
    chronology = column(cells, "chronology")
    benefit = column(cells, "mix_benefit")
    country = column(cells, "country")
    year = column(cells, "year")
    duration = column(cells, "duration_h")
    data = cells.copy()
    data["_chronology"] = canonical_chronology(data[chronology])
    # Strict-valley prevalence is recomputed from each scenario/country-year.
    rows = []
    group_keys = [energy_price, charge_price, country, year, "_chronology"]
    firm_price = column(data, "firm_price", required=False)
    if firm_price:
        group_keys.insert(2, firm_price)
    for keys, group in data.groupby(group_keys):
        ordered = (
            group.groupby(duration)[benefit].max().sort_index()
        )
        durations = ordered.index.to_numpy(float)
        positive = ordered.to_numpy(float) > 0
        strict = False
        if positive.sum() >= 2:
            first = np.flatnonzero(positive)
            for left_index, right_index in zip(first[:-1], first[1:]):
                if right_index - left_index > 1 and (~positive[left_index + 1 : right_index]).all():
                    strict = True
                    break
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_keys, key_values))
        row["strict_valley"] = strict
        rows.append(row)
    records = pd.DataFrame(rows)
    phase_keys = [energy_price, charge_price, "_chronology"]
    if firm_price:
        phase_keys.insert(2, firm_price)
    phase = (
        records.groupby(phase_keys)["strict_valley"]
        .mean()
        .mul(100)
        .rename("strict_valley_prevalence_pct")
        .reset_index()
    )
    return phase, energy_price, charge_price, firm_price or ""


def figure5() -> None:
    envelope, envelope_path = read_portfolio("envelope")
    valley, valley_path = read_portfolio("valley")
    bootstrap, bootstrap_path = read_portfolio("bootstrap")
    channel_path = (
        V43_SOURCE / "v43_figure5c_channel_decomposition_wide.csv"
    )
    signature_path = (
        V43_SOURCE / "v43_figure5d_cluster_signature_summary.csv"
    )
    crossing_path = (
        V43_SOURCE / "v43_figure5e_zero_crossing_country_year.csv"
    )
    crossing_summary_path = (
        V43_SOURCE / "v43_figure5e_zero_crossing_summary.csv"
    )
    if not channel_path.exists():
        raise MissingEvidence("Frozen v43 signed channel decomposition is missing")
    if (
        not signature_path.exists()
        or not crossing_path.exists()
        or not crossing_summary_path.exists()
    ):
        raise MissingEvidence("Frozen v43 2/4/6-day signature sources are missing")
    channel = pd.read_csv(channel_path)
    channel_plot = channel[channel["duration_h"].le(168)].copy()
    signature = pd.read_csv(signature_path)
    crossings = pd.read_csv(crossing_path)
    crossing_summary = pd.read_csv(crossing_summary_path)
    envelope = central_s20(envelope)
    valley = central_s20(valley)
    bootstrap = central_s20(bootstrap)
    summary = _summarize_envelope(envelope)

    fig = plt.figure(figsize=(WIDTH, 5.65))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.095, top=0.965)
    gs = fig.add_gridspec(
        2, 14, height_ratios=[1.06, 1.0], hspace=0.60, wspace=1.0
    )
    axa = fig.add_subplot(gs[0, 0:5])
    axb = fig.add_subplot(gs[0, 5:14])
    axc = fig.add_subplot(gs[1, 0:5])
    axd = fig.add_subplot(gs[1, 5:10])
    axe = fig.add_subplot(gs[1, 10:14])
    axes = [axa, axb, axc, axd, axe]

    # a: absolute central envelope. Both chronology arms are shown directly,
    # and the zero line prevents the paired effect from being mistaken for
    # central economic entry.
    axa.axhline(0, color=INK, lw=0.70)
    for chronology_name, colour, marker in [
        ("dispersed", DISPERSED, "o"),
        ("clustered", CLUSTERED, "s"),
    ]:
        group = summary[summary["_chronology"].eq(chronology_name)].sort_values(
            "_duration_h"
        )
        axa.fill_between(
            group["_duration_h"],
            group["q25"],
            group["q75"],
            color=colour,
            alpha=0.13,
            linewidth=0,
        )
        axa.plot(
            group["_duration_h"],
            group["median"],
            color=colour,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.55,
            markevery=max(1, len(group) // 8),
            label=chronology_name.capitalize(),
        )
    duration_axis(axa, max(168, float(summary["_duration_h"].max())))
    axa.set_xlabel("LDES duration (h)")
    axa.set_ylabel(
        "Best incremental portfolio value\n(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axa.legend(ncol=2, loc="lower left")
    axa.grid(axis="y", color=GRID, lw=0.45)
    central_counts = (
        valley.groupby("chronology")["strict_valley"].sum().to_dict()
        if "strict_valley" in valley
        else {}
    )
    axa.text(
        0.05,
        0.90,
        "strict valleys\n"
        f"D {int(central_counts.get('dispersed', 0))}/132 · "
        f"C {int(central_counts.get('clustered', 0))}/132",
        transform=axa.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        color=MUTED,
        fontweight="bold",
    )

    # b: hero evidence—the paired clustering response is U-shaped over the
    # technology-duration envelope and crosses zero at long durations.
    effect_metric = (
        "clustered_minus_dispersed_best_mix_benefit_"
        "eur_per_kw_common_power_year"
    )
    effect = bootstrap[
        bootstrap["analysis"].eq("duration_envelope_paired_effect")
        & bootstrap["metric"].eq(effect_metric)
    ].sort_values("duration_h")
    if effect.empty:
        raise MissingEvidence("Figure 5 requires V43 duration-envelope bootstrap effects")
    axb.axhline(0, color=INK, lw=0.75)
    axb.fill_between(
        effect["duration_h"],
        effect["ci_low"],
        effect["ci_high"],
        color=CLUSTERED,
        alpha=0.16,
        linewidth=0,
    )
    axb.plot(
        effect["duration_h"],
        effect["point_estimate"],
        color=CLUSTERED,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.65,
        markevery=max(1, len(effect) // 14),
    )
    duration_axis(axb, max(168, float(effect["duration_h"].max())))
    axb.set_xlabel("LDES duration (h)")
    axb.set_ylabel(
        "Clustered − dispersed incremental value\n"
        "(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axb.grid(axis="y", color=GRID, lw=0.5)
    key_durations = [24, 32, 40, 48, 72, 96, 120, 168]
    key_effect = effect[effect["duration_h"].isin(key_durations)]
    for _, row in key_effect.iterrows():
        axb.annotate(
            f"{row['point_estimate']:+.1f}",
            (row["duration_h"], row["point_estimate"]),
            xytext=(0, 5 if row["point_estimate"] >= 0 else -8),
            textcoords="offset points",
            ha="center",
            va="bottom" if row["point_estimate"] >= 0 else "top",
            fontsize=5.0,
            color=CLUSTERED,
        )

    # c: signed value-channel decomposition in the same units as panel b.
    axc.axhline(0, color=INK, lw=0.65)
    channel_specs = [
        (
            "cycling_channel",
            RECHARGE,
            "Cycling return",
        ),
        (
            "coverage_channel",
            PURPLE,
            "Coverage return",
        ),
    ]
    for prefix, colour, label in channel_specs:
        mean_col = f"{prefix}_country_block_mean"
        low_col = f"{prefix}_ci95_low"
        high_col = f"{prefix}_ci95_high"
        axc.fill_between(
            channel_plot["duration_h"],
            channel_plot[low_col],
            channel_plot[high_col],
            color=colour,
            alpha=0.12,
            linewidth=0,
        )
        axc.plot(
            channel_plot["duration_h"],
            channel_plot[mean_col],
            color=colour,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.55,
            markevery=max(1, len(channel_plot) // 10),
            label=label,
        )
    duration_axis(axc, 168)
    axc.set_xlabel("LDES duration (h)")
    axc.set_ylabel(
        "Signed channel contribution\n"
        "(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axc.grid(axis="y", color=GRID, lw=0.45)
    axc.legend(loc="lower right")

    # d: mechanism signature under registered 2, 4 and 6 day clusters.
    phase_matrix = signature.pivot_table(
        index="target_cluster_days",
        columns="duration_h",
        values="country_block_mean",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    effect_limit = max(
        1.0,
        float(np.nanmax(np.abs(phase_matrix.to_numpy(float)))),
    )
    image = axd.imshow(
        phase_matrix.to_numpy(),
        aspect="auto",
        origin="upper",
        cmap="RdBu_r",
        vmin=-effect_limit,
        vmax=effect_limit,
    )
    signature_durations = phase_matrix.columns.to_numpy(float)
    tick_durations = [24, 32, 48, 72, 96, 120, 168]
    tick_durations = [
        value for value in tick_durations if value in set(signature_durations)
    ]
    axd.set_xticks(
        [
            int(np.flatnonzero(np.isclose(signature_durations, value))[0])
            for value in tick_durations
        ]
    )
    axd.set_xticklabels([f"{value:g}" for value in tick_durations])
    axd.set_yticks(range(len(phase_matrix.index)))
    axd.set_yticklabels([f"{int(value)} d" for value in phase_matrix.index])
    axd.set_xlabel("LDES duration (h)")
    axd.set_ylabel("Controlled cluster length")
    for row_index, cluster_days in enumerate(phase_matrix.index):
        row_summary = crossing_summary[
            crossing_summary["target_cluster_days"].eq(cluster_days)
        ]
        if row_summary.empty or pd.isna(
            row_summary.iloc[0]["median_crossing_h"]
        ):
            continue
        crossing_h = float(row_summary.iloc[0]["median_crossing_h"])
        column_index = int(np.argmin(np.abs(signature_durations - crossing_h)))
        axd.scatter(
            [column_index],
            [row_index],
            s=32,
            facecolor="none",
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
    cax = axd.inset_axes([0.14, 0.925, 0.70, 0.035])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_ticks([-effect_limit, 0, effect_limit])
    cbar.ax.tick_params(labelsize=5.0, length=1.2)
    cbar.ax.set_title(
        "paired effect (EUR kW$^{-1}$ yr$^{-1}$)",
        fontsize=5.0,
        pad=1,
    )
    axd.text(
        0.98,
        0.04,
        "white ring · median country-year crossing",
        transform=axd.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=INK,
    )

    # e: country-year crossing distributions with explicit censoring.
    cluster_colours = {2: BLUE, 4: GOLD, 6: CLUSTERED}
    for row_index, cluster_days in enumerate([2, 4, 6]):
        group = crossings[
            crossings["target_cluster_days"].eq(cluster_days)
        ].copy()
        observed = group[group["crossing_status"].eq("observed")]
        colour = cluster_colours[cluster_days]
        if not observed.empty:
            y = row_index + deterministic_display_offsets(
                len(observed), half_width=0.17
            )
            axe.scatter(
                observed["zero_crossing_duration_h"],
                y,
                s=7,
                color=colour,
                alpha=0.25,
                edgecolor="none",
            )
        summary_row = crossing_summary[
            crossing_summary["target_cluster_days"].eq(cluster_days)
        ].iloc[0]
        if pd.notna(summary_row["median_crossing_h"]):
            axe.hlines(
                row_index,
                summary_row["q25_crossing_h"],
                summary_row["q75_crossing_h"],
                color=colour,
                lw=3.1,
            )
            axe.scatter(
                summary_row["median_crossing_h"],
                row_index,
                s=24,
                facecolor="white",
                edgecolor=colour,
                linewidth=0.8,
                zorder=5,
            )
        right_count = int(summary_row["right_censored"])
        if right_count:
            axe.scatter(
                [176],
                [row_index],
                marker=">",
                s=26,
                facecolor=colour,
                edgecolor=INK,
                linewidth=0.45,
                zorder=6,
            )
            axe.text(
                176,
                row_index + 0.20,
                f"n={right_count}",
                ha="right",
                va="bottom",
                fontsize=5.0,
                color=INK,
            )
        left_count = int(summary_row["left_censored"])
        if left_count:
            axe.scatter(
                [22],
                [row_index],
                marker="<",
                s=26,
                facecolor=colour,
                edgecolor=INK,
                linewidth=0.45,
                zorder=6,
            )
    axe.set_xscale("log")
    axe.set_xlim(20, 185)
    axe.set_xticks([24, 32, 48, 72, 96, 120, 168])
    axe.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    axe.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axe.set_ylim(-0.45, 2.55)
    axe.set_yticks([0, 1, 2])
    axe.set_yticklabels(["2 d", "4 d", "6 d"])
    axe.set_xlabel("Zero-crossing duration (h)")
    axe.set_ylabel("Controlled cluster length")
    axe.grid(axis="x", color=GRID, lw=0.45)
    axe.text(
        0.98,
        0.04,
        "triangles denote censoring",
        transform=axe.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=MUTED,
    )

    panel_sources = [
        save_panel(summary, "figure5_panel_a_absolute_central_envelope.csv"),
        save_panel(effect, "figure5_panel_b_central_clustering_effect.csv"),
        save_panel(
            channel_plot,
            "figure5_panel_c_signed_channel_decomposition.csv",
        ),
        save_panel(signature, "figure5_panel_d_cluster_signature.csv"),
        save_panel(crossings, "figure5_panel_e_zero_crossing_records.csv"),
        save_panel(
            crossing_summary,
            "figure5_panel_e_zero_crossing_summary.csv",
        ),
    ]
    panel_labels(fig, axes)
    export(
        fig,
        MAIN_OUT,
        "fig_v43_figure5_conditional_valley",
        "Figure 5",
        [
            envelope_path,
            valley_path,
            bootstrap_path,
            channel_path,
            signature_path,
            crossing_path,
            crossing_summary_path,
            *panel_sources,
        ],
        (
            "Central absolute entry, the paired duration response, signed "
            "value channels and the registered 2/4/6-day mechanism signature "
            "are kept distinct."
        ),
    )


def _eraa_summary() -> tuple[pd.DataFrame, Path]:
    if not ERAA_METRICS.exists():
        raise MissingEvidence(f"ERAA run metrics missing: {ERAA_METRICS}")
    data = pd.read_csv(ERAA_METRICS)
    if "variant" not in data:
        raise MissingEvidence("ERAA run metrics do not identify baseline and joint variants")
    baseline = data[data["variant"].eq("baseline")].iloc[0]
    joint = data[data["variant"].eq("joint")].iloc[0]
    rows = []
    specs = [
        (
            "Economic objective",
            "economic_objective_excluding_hydro_feasibility_penalties_eur_year",
            1e9,
            "EUR bn yr$^{-1}$",
        ),
        (
            "EENS",
            "deterministic_expected_availability_unserved_energy_mwh",
            1000,
            "GWh",
        ),
        (
            "LOLE",
            "deterministic_expected_availability_loss_of_load_hours",
            1,
            "h",
        ),
    ]
    for metric, colname, scale, unit in specs:
        base = float(baseline[colname]) / scale
        candidate = float(joint[colname]) / scale
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "baseline": base,
                "with_fixed_4p242_gw_caes": candidate,
                "absolute_change": candidate - base,
                "relative_change_pct": 100 * (candidate - base) / base,
            }
        )
    return pd.DataFrame(rows), ERAA_METRICS


def _draw_eraa_small_multiples(
    axes: Sequence[plt.Axes], summary: pd.DataFrame
) -> None:
    for ax, (_, row) in zip(axes, summary.iterrows()):
        values = [row["baseline"], row["with_fixed_4p242_gw_caes"]]
        ax.bar(
            [0, 1],
            values,
            color=["#C7CFD3", RECHARGE],
            width=0.58,
        )
        lower = min(values)
        upper = max(values)
        span = max(upper - lower, abs(upper) * 0.03, 1e-9)
        ax.set_ylim(max(0, lower - span * 0.55), upper + span * 0.95)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "+ A-CAES"], rotation=20, ha="right")
        ax.set_ylabel(row["unit"])
        ax.text(
            0.98,
            0.96,
            f"{row['relative_change_pct']:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=RECHARGE if row["relative_change_pct"] < 0 else RED,
            fontweight="bold",
            fontsize=5.5,
        )


def figure6() -> None:
    cost_phase, cost_phase_path = read_portfolio("cost_firm_paired")
    envelope, envelope_path = read_portfolio("all_envelope")
    cells, cells_path = read_portfolio("all_cells")
    config_pairs, config_pairs_path = read_portfolio("all_config_paired")
    year_duration_path = (
        V43_SOURCE / "v43_year_specific_duration_summary.csv"
    )
    year_valley_path = V43_SOURCE / "v43_year_specific_valley_summary.csv"
    if not year_duration_path.exists() or not year_valley_path.exists():
        raise MissingEvidence("Frozen 2025/2030/2035 cost-year sources are missing")
    year_duration = pd.read_csv(year_duration_path)
    year_valley = pd.read_csv(year_valley_path)

    cost_phase = central_s20(cost_phase)
    # The bounded-power comparison is a registered S31 robustness case.
    def s31(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        mask = pd.Series(True, index=data.index)
        if "source_label" in data:
            mask &= data["source_label"].astype(str).str.lower().str.contains("s31")
        if "valuation_scenario_id" in data:
            mask &= pd.to_numeric(
                data["valuation_scenario_id"], errors="coerce"
            ).eq(31)
        return data[mask].copy()

    robust_envelope = s31(envelope)
    robust_cells = s31(cells)
    robust_pairs = s31(config_pairs)
    mode_col = column(robust_envelope, "model_mode")
    if robust_envelope[mode_col].astype(str).str.lower().nunique() < 2:
        raise MissingEvidence(
            "Figure 6 requires completed fixed and bounded-power S31 combination outputs"
        )

    fig = plt.figure(figsize=(WIDTH, 5.3))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.090, top=0.965)
    gs = fig.add_gridspec(
        2, 12, height_ratios=[1.16, 0.92], hspace=0.58, wspace=1.25
    )
    axa = fig.add_subplot(gs[0, 0:5])
    axb = fig.add_subplot(gs[0, 7:12])
    axc = fig.add_subplot(gs[1, 0:5])
    axd = fig.add_subplot(gs[1, 5:9])
    axe = fig.add_subplot(gs[1, 10:12])
    axes = [axa, axb, axc, axd, axe]

    # a: actual technology-specific 2025, 2030 and 2035 cost paths.
    annual = year_duration[
        year_duration["revaluation_case_id"].eq("central_actual_year_cost")
    ].copy()
    annual["row_label"] = annual.apply(
        lambda row: (
            f"{int(row['cost_year'])} "
            + (
                "D"
                if str(row["chronology"]).lower().startswith("d")
                else "C"
            )
        ),
        axis=1,
    )
    row_order = [
        f"{year} {chronology}"
        for year in [2025, 2030, 2035]
        for chronology in ["D", "C"]
    ]
    annual_matrix = annual.pivot_table(
        index="row_label",
        columns="duration_h",
        values="median",
        aggfunc="mean",
    ).reindex(row_order)
    annual_limit = min(
        -1.0,
        math.floor(float(np.nanmin(annual_matrix.to_numpy(float)))),
    )
    im_a = axa.imshow(
        annual_matrix.to_numpy(),
        aspect="auto",
        origin="upper",
        cmap="Blues_r",
        vmin=annual_limit,
        vmax=0,
    )
    durations = annual_matrix.columns.to_numpy(float)
    shown = [8, 12, 24, 40, 72, 120, 168]
    shown = [value for value in shown if value in set(durations)]
    shown_positions = [
        int(np.flatnonzero(np.isclose(durations, value))[0]) for value in shown
    ]
    axa.set_xticks(shown_positions)
    axa.set_xticklabels([f"{value:g}" for value in shown])
    axa.set_yticks(range(len(annual_matrix)))
    axa.set_yticklabels(annual_matrix.index)
    axa.set_xlabel("LDES duration (h)")
    axa.set_ylabel("")
    axa.text(
        0.0,
        1.012,
        "D dispersed · C clustered · both LDES and 4 h LFP costs updated",
        transform=axa.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=5.0,
    )
    cbar = fig.colorbar(im_a, ax=axa, fraction=0.035, pad=0.025)
    cbar.set_label(
        "Median incremental benefit\n(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    central_year_valleys = year_valley[
        year_valley["revaluation_case_id"].eq("central_actual_year_cost")
    ]
    if not central_year_valleys.empty:
        axa.text(
            0.98,
            0.96,
            "strict valleys 0/132 in all six rows",
            transform=axa.transAxes,
            ha="right",
            va="top",
            fontsize=5.0,
            color=INK,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.1,
            },
        )

    # b: cost × firm-value phase for clustering-induced strict valleys.
    phase_matrix = cost_phase.pivot_table(
        index="ldes_tail_fixed_cost_multiplier",
        columns="firm_value_eur_kw_year",
        values="cluster_induced_rate",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    im_b = axb.imshow(
        100 * phase_matrix.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma_r",
        vmin=0,
        vmax=max(1, 100 * np.nanmax(phase_matrix.to_numpy())),
        extent=[
            float(phase_matrix.columns.min()),
            float(phase_matrix.columns.max()),
            float(phase_matrix.index.min()),
            float(phase_matrix.index.max()),
        ],
    )
    axb.set_xlabel(
        "Task-firm-supply unit value (EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axb.set_ylabel(
        "Multiplier relative to 2035 LDES-tail\nannualized fixed cost"
    )
    axb.scatter(
        [103.8],
        [1.0],
        marker="o",
        s=28,
        facecolor="white",
        edgecolor=INK,
        linewidth=0.75,
        zorder=6,
    )
    axb.annotate(
        "central\n($m_C$=1, $v_F$=103.8)",
        (103.8, 1.0),
        xytext=(7, -5),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=5.0,
        color=INK,
    )
    axb.scatter(
        [575],
        [0.7],
        marker="*",
        s=44,
        facecolor=GOLD,
        edgecolor=INK,
        linewidth=0.55,
        zorder=6,
    )
    axb.annotate(
        "conditional peak\n($m_C$=0.7, $v_F$=575)",
        (575, 0.7),
        xytext=(-5, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=INK,
    )
    axb.text(
        0.02,
        0.03,
        "4 h LFP comparator held at its 2035 cost",
        transform=axb.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=INK,
    )
    cax_b = axb.inset_axes([0.60, 0.92, 0.36, 0.025])
    cbar = fig.colorbar(im_b, cax=cax_b, orientation="horizontal")
    cax_b.tick_params(labelsize=5.0, length=1.2)
    cax_b.set_xlabel(
        "clustering-induced strict valleys (%)", fontsize=5.0, labelpad=0.5
    )
    cax_b.xaxis.set_label_position("top")
    cax_b.xaxis.tick_top()

    # c: fixed versus bounded-power portfolio envelopes.
    robust_envelope = robust_envelope.copy()
    robust_envelope["_chronology"] = canonical_chronology(
        robust_envelope[column(robust_envelope, "chronology")]
    )
    robust_envelope["_mode"] = (
        robust_envelope[mode_col]
        .astype(str)
        .str.lower()
        .replace({"optimized": "bounded"})
    )
    benefit_col = column(robust_envelope, "mix_benefit")
    duration_col = column(robust_envelope, "duration_h")
    envelope_summary = (
        robust_envelope.groupby(["_mode", "_chronology", duration_col])[
            benefit_col
        ]
        .median()
        .rename("median_mix_benefit")
        .reset_index()
    )
    axc.axhline(0, color=INK, lw=0.7)
    for mode, linestyle in [("fixed", "-"), ("bounded", "--")]:
        for chronology_name, colour in [
            ("dispersed", DISPERSED),
            ("clustered", CLUSTERED),
        ]:
            group = envelope_summary[
                envelope_summary["_mode"].eq(mode)
                & envelope_summary["_chronology"].eq(chronology_name)
            ].sort_values(duration_col)
            if group.empty:
                continue
            axc.plot(
                group[duration_col],
                group["median_mix_benefit"],
                color=colour,
                linestyle=linestyle,
                marker="o" if mode == "fixed" else "s",
                markerfacecolor="white",
                markeredgewidth=0.6,
                markevery=max(1, len(group) // 10),
                label=f"{chronology_name[0].upper()} · {mode}",
            )
    duration_axis(axc, max(168, float(robust_envelope[duration_col].max())))
    axc.set_xlabel("LDES duration (h)")
    axc.set_ylabel("Incremental value (EUR kW$^{-1}$ yr$^{-1}$)")
    axc.legend(ncol=2, fontsize=5.0)

    # d: bounded tail-power choice and paired value direction.
    robust_cells = robust_cells.copy()
    robust_cells["_mode"] = robust_cells[column(robust_cells, "model_mode")].astype(str).str.lower()
    optimized = robust_cells[
        robust_cells["_mode"].isin(["optimized", "bounded"])
    ].copy()
    optimized["selected_power_share_of_upper"] = (
        optimized["tail_power_mw"]
        / optimized["tail_power_upper_mw"].replace(0, np.nan)
    )
    robust_pairs = robust_pairs.copy()
    robust_pairs["_mode"] = robust_pairs[column(robust_pairs, "model_mode")].astype(str).str.lower()
    optimized_pairs = robust_pairs[
        robust_pairs["_mode"].isin(["optimized", "bounded"])
    ].copy()
    pair_delta = next(
        name
        for name in [
            "clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year",
            "clustered_minus_dispersed_best_mix_benefit_eur_per_kw_common_power_year",
        ]
        if name in optimized_pairs
    )
    power = (
        optimized.groupby(["country", "year", "config_id", "tech_id"])
        .agg(
            selected_power_share=("selected_power_share_of_upper", "mean"),
        )
        .reset_index()
    )
    direction = optimized_pairs[
        ["country", "year", "config_id", pair_delta]
    ].drop_duplicates()
    power_direction = power.merge(
        direction, on=["country", "year", "config_id"], how="inner"
    )
    for tech, group in power_direction.groupby("tech_id"):
        axd.scatter(
            100 * group["selected_power_share"],
            group[pair_delta],
            color=TECH_COLOR.get(str(tech), MUTED),
            s=9,
            alpha=0.45,
            edgecolor="none",
            label=TECH_LABEL.get(str(tech), str(tech)),
        )
    axd.axhline(0, color=INK, lw=0.65)
    axd.set_xlabel("Selected tail power / upper bound (%)")
    axd.set_ylabel("Paired incremental-value\nchange")
    handles, labels = axd.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.66, 0.49),
        ncol=4,
        fontsize=5.0,
        markerscale=0.9,
        handletextpad=0.25,
        columnspacing=0.65,
        frameon=False,
    )

    # e: numerical QA from completed V43 production runs only.
    qa_paths = [
        V43 / "portfolio_model" / name / "qa_summary.json"
        for name in [
            "full_132cy_fixed_s20_central",
            "full_132cy_optimized_s20_central",
            "full_132cy_fixed",
            "full_132cy_optimized",
        ]
    ]
    qa_paths = [path for path in qa_paths if path.exists()]
    if len(qa_paths) < 2:
        raise MissingEvidence("Completed V43 fixed/bounded QA summaries are missing")
    qa_specs = [
        ("Task", "maximum_task_balance_residual_mw"),
        ("Recharge", "maximum_shared_recharge_violation_mw"),
        ("Base SOC", "maximum_base_soc_balance_residual_mwh"),
        ("Tail SOC", "maximum_tail_soc_balance_residual_mwh"),
        ("Cost identity", "maximum_system_cost_identity_error_eur"),
    ]
    qa_rows = []
    for path in qa_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks = payload.get("checks", payload)
        for label, key in qa_specs:
            aliases = {
                "maximum_base_soc_balance_residual_mwh": (
                    "maximum_cyclic_soc_residual_mwh"
                ),
                "maximum_tail_soc_balance_residual_mwh": (
                    "maximum_cyclic_soc_residual_mwh"
                ),
                "maximum_system_cost_identity_error_eur": (
                    "maximum_accounting_identity_residual_eur"
                ),
            }
            qa_rows.append(
                {
                    "run": path.parent.name,
                    "metric": label,
                    "maximum_absolute_residual": float(
                        checks.get(key, checks.get(aliases.get(key, ""), np.nan))
                    ),
                }
            )
    qa = pd.DataFrame(qa_rows)
    qa_plot = (
        qa.groupby("metric")["maximum_absolute_residual"].max().reindex(
            [item[0] for item in qa_specs]
        )
    )
    plotted_qa = np.log10(np.maximum(qa_plot.to_numpy(float), 1e-16))
    axe.barh(range(len(qa_plot)), plotted_qa, color="#82939B")
    axe.set_yticks(range(len(qa_plot)))
    axe.set_yticklabels(qa_plot.index)
    axe.tick_params(axis="y", labelsize=5.0)
    axe.axvline(-8, color=RED, lw=0.7, ls="--")
    axe.set_xlabel("log$_{10}$(max. residual)")
    axe.invert_yaxis()

    panel_sources = [
        save_panel(annual, "figure6_panel_a_actual_cost_year_envelopes.csv"),
        save_panel(
            central_year_valleys,
            "figure6_panel_a_actual_cost_year_valleys.csv",
        ),
        save_panel(cost_phase, "figure6_panel_b_cost_firm_phase.csv"),
        save_panel(envelope_summary, "figure6_panel_c_fixed_bounded_envelopes.csv"),
        save_panel(power_direction, "figure6_panel_d_bounded_power_direction.csv"),
        save_panel(qa, "figure6_panel_e_v43_numerical_qa.csv"),
    ]
    panel_labels(fig, axes)
    export(
        fig,
        MAIN_OUT,
        "fig_v43_figure6_value_and_power_sensitivity",
        "Figure 6",
        [
            year_duration_path,
            year_valley_path,
            cost_phase_path,
            envelope_path,
            cells_path,
            config_pairs_path,
            *qa_paths,
            *panel_sources,
        ],
        (
            "Panel a uses source-aware technology-specific 2025, 2030 and "
            "2035 costs for both LDES and the 4 h LFP comparator. Panel b "
            "scales only the 2035 LDES-tail annual fixed cost."
        ),
    )


def _release_frozen(
    ed_id: str,
    source_dir: Path,
    source_stem: str,
    target_stem: str,
    source_csvs: Sequence[Path],
    note: str,
) -> None:
    ED_OUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ["pdf", "svg", "png", "tiff"]:
        source = source_dir / f"{source_stem}.{suffix}"
        if not source.exists():
            continue
        target = ED_OUT / f"{target_stem}.{suffix}"
        shutil.copy2(source, target)
        outputs.append(target)
    gray_source = source_dir / f"{source_stem}_grayscale.png"
    if gray_source.exists():
        gray_target = ED_OUT / f"{target_stem}_grayscale.png"
        shutil.copy2(gray_source, gray_target)
        outputs.append(gray_target)
    if not outputs:
        raise MissingEvidence(f"Frozen source figure missing: {source_dir / source_stem}")
    copied_sources = []
    for path in source_csvs:
        if not path.exists():
            continue
        target = SOURCE_OUT / f"{ed_id.lower()}_{path.name}"
        SOURCE_OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied_sources.append(target)
    BUILD_RECORDS.append(
        BuildRecord(
            figure_id=ed_id,
            status="released_frozen",
            outputs=[str(path.relative_to(HERE)) for path in outputs],
            sources=[str(path) for path in source_csvs if path.exists()]
            + [str(path) for path in copied_sources],
            note=note,
        )
    )


def release_ed1_ed2() -> None:
    _release_frozen(
        "ED1",
        V40 / "ed1_cost_paths_53",
        "fig_v40_ed1_cost_paths_53_eur",
        "fig_v43_ed1_cost_paths_53_eur",
        [
            V40
            / "ed1_cost_paths_53"
            / "source_data"
            / "ed1_53_duration_year_cost_paths_eur.csv"
        ],
        "Frozen source-aware EUR catalogue; 53 technology-duration configurations.",
    )
    _release_frozen(
        "ED2",
        V40 / "supplement",
        "fig_v40_ed2_event_definition_sensitivity",
        "fig_v43_ed2_event_definition_sensitivity",
        [V40_SOURCE / "ed2_event_definition_sensitivity.csv"],
        "Frozen European event-definition sensitivity.",
    )


def _v43_qa_inputs() -> list[tuple[str, Path, Path]]:
    """Return completed manuscript-scale V43 runs, never smoke/chunk outputs."""
    candidates = [
        "full_132cy_fixed_s20_central",
        "full_132cy_optimized_s20_central",
        "full_132cy_fixed",
        "full_132cy_optimized",
    ]
    completed = []
    for name in candidates:
        root = V43 / "portfolio_model" / name
        qa = root / "qa_summary.json"
        conservation = root / "chronology_conservation.csv"
        if qa.exists() and conservation.exists():
            completed.append((name, qa, conservation))
    if not completed:
        raise MissingEvidence("Completed manuscript-scale V43 QA outputs are missing")
    return completed


def ed3() -> None:
    """Numerical and paired-chronology audit rebuilt from V43 production runs."""
    inputs = _v43_qa_inputs()
    qa_rows = []
    conservation_frames = []
    for run, qa_path, conservation_path in inputs:
        payload = json.loads(qa_path.read_text(encoding="utf-8"))
        checks = payload.get("checks", payload)
        aliases = [
            (
                "Base SOC",
                [
                    "maximum_base_soc_balance_residual_mwh",
                    "maximum_base_soc_residual_mwh",
                ],
                "MWh",
            ),
            (
                "Tail SOC",
                [
                    "maximum_tail_soc_balance_residual_mwh",
                    "maximum_tail_soc_residual_mwh",
                ],
                "MWh",
            ),
            (
                "Cyclic SOC",
                ["maximum_cyclic_soc_residual_mwh"],
                "MWh",
            ),
            (
                "Task balance",
                ["maximum_task_balance_residual_mw"],
                "MW",
            ),
            (
                "Shared recharge",
                ["maximum_shared_recharge_violation_mw"],
                "MW",
            ),
            (
                "Accounting identity",
                [
                    "maximum_accounting_identity_residual_eur",
                    "maximum_system_cost_identity_error_eur",
                ],
                "EUR",
            ),
        ]
        for label, keys, unit in aliases:
            value = next(
                (
                    float(checks[key])
                    for key in keys
                    if key in checks and pd.notna(checks[key])
                ),
                np.nan,
            )
            qa_rows.append(
                {
                    "run": run,
                    "metric": label,
                    "unit": unit,
                    "maximum_absolute_residual": value,
                }
            )
        frame = pd.read_csv(conservation_path)
        frame["run"] = run
        conservation_frames.append(frame)
    qa = pd.DataFrame(qa_rows)
    conservation = pd.concat(conservation_frames, ignore_index=True)

    fig = plt.figure(figsize=(WIDTH, 4.6))
    fig.subplots_adjust(left=0.080, right=0.985, bottom=0.105, top=0.965)
    gs = fig.add_gridspec(2, 12, hspace=0.62, wspace=1.0)
    axa = fig.add_subplot(gs[:, 0:5])
    axb = fig.add_subplot(gs[0, 5:12])
    axc = fig.add_subplot(gs[1, 5:9])
    axd = fig.add_subplot(gs[1, 9:12])
    axes = [axa, axb, axc, axd]

    conservation_columns = [
        ("base_task_conservation_error_mwh", "C1 annual task", BLUE),
        ("tail_task_conservation_error_mwh", "C3 event tail", CLUSTERED),
        ("recharge_conservation_error_mwh", "R2 recharge", RECHARGE),
    ]
    conservation_long = []
    for name, label, colour in conservation_columns:
        values = np.abs(
            complete_cases(
                pd.to_numeric(conservation[name], errors="coerce"),
                label=f"ed3_conservation_{name}",
            )
        )
        ordered = np.sort(np.maximum(values.to_numpy(float), 1e-16))
        probability = np.arange(1, len(ordered) + 1) / len(ordered)
        axa.plot(ordered, probability, color=colour, label=label)
        conservation_long.extend(
            {"metric": label, "absolute_error_mwh": value} for value in values
        )
    axa.set_xscale("log")
    axa.set_xlim(5e-17, None)
    axa.set_xlabel("Absolute annual conservation error (MWh)")
    axa.set_ylabel("Empirical cumulative probability")
    axa.legend(loc="upper left", bbox_to_anchor=(0.02, 0.86), frameon=False)
    axa.text(
        0.03,
        0.96,
        "C1, C3 and R2 conserved\nwithin every country–year pair",
        transform=axa.transAxes,
        va="top",
        color=MUTED,
    )

    run_order = [item[0] for item in inputs]
    run_labels = {
        "full_132cy_fixed_s20_central": "S20 central",
        "full_132cy_optimized_s20_central": "S20 bounded",
        "full_132cy_fixed": "three value cases",
        "full_132cy_optimized": "three cases bounded",
    }
    run_colours = [DISPERSED, CLUSTERED, RECHARGE, PURPLE]

    def residual_dotplot(
        ax: plt.Axes,
        metrics: Sequence[str],
        xlabel: str,
    ) -> pd.DataFrame:
        view = qa[qa["metric"].isin(metrics)].copy()
        y_positions = np.arange(len(metrics))
        offsets = np.linspace(-0.13, 0.13, max(1, len(run_order)))
        for offset, run, colour in zip(offsets, run_order, run_colours):
            group = view[view["run"].eq(run)].set_index("metric").reindex(metrics)
            values = np.maximum(
                group["maximum_absolute_residual"].to_numpy(float), 1e-16
            )
            ax.scatter(
                values,
                y_positions + offset,
                s=26,
                color=colour,
                edgecolor="white",
                linewidth=0.45,
                label=run_labels.get(run, run),
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(metrics)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=GRID, lw=0.45)
        return view

    soc_view = residual_dotplot(
        axb,
        ["Base SOC", "Tail SOC", "Cyclic SOC"],
        "Maximum absolute SOC residual (MWh)",
    )
    axb.legend(ncol=min(4, len(run_order)), loc="upper right")
    balance_view = residual_dotplot(
        axc,
        ["Task balance", "Shared recharge"],
        "Maximum absolute residual (MW)",
    )
    axc.set_yticklabels(["Task balance", "Recharge balance"])
    account_view = residual_dotplot(
        axd,
        ["Accounting identity"],
        "Maximum absolute residual (EUR)",
    )

    panel_labels(fig, axes)
    source_qa = save_panel(qa, "ed3_v43_numerical_residuals.csv")
    source_conservation = save_panel(
        pd.DataFrame(conservation_long), "ed3_v43_chronology_conservation_errors.csv"
    )
    source_soc = save_panel(soc_view, "ed3_v43_soc_residuals.csv")
    source_balance = save_panel(balance_view, "ed3_v43_balance_residuals.csv")
    source_account = save_panel(account_view, "ed3_v43_accounting_residuals.csv")
    export(
        fig,
        ED_OUT,
        "fig_v43_ed3_numerical_qa",
        "ED3",
        [
            *[item for _, qa, conservation_path in inputs for item in (qa, conservation_path)],
            source_qa,
            source_conservation,
            source_soc,
            source_balance,
            source_account,
        ],
        "V43 production-run numerical and conserved-chronology audit.",
    )


def ed4() -> None:
    """Full technology-duration response of the V43 fixed-scale portfolios."""
    cells, cells_path = read_portfolio("cells")
    pairs, pairs_path = read_portfolio("config_paired")
    cells = central_s20(cells)
    pairs = central_s20(pairs)
    cells = cells.copy()
    cells["_chronology"] = canonical_chronology(cells[column(cells, "chronology")])
    benefit = column(cells, "mix_benefit")
    duration = column(cells, "duration_h")
    tech = column(cells, "technology")
    delta_col = next(
        (
            name
            for name in [
                "clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year",
                "clustered_minus_dispersed_best_mix_benefit_eur_per_kw_common_power_year",
            ]
            if name in pairs
        ),
        None,
    )
    if delta_col is None:
        raise MissingEvidence("ED4 requires paired V43 configuration-level mix benefits")

    summary = (
        cells.groupby([tech, duration, "_chronology"])[benefit]
        .median()
        .rename("median_mix_benefit")
        .reset_index()
    )
    delta = (
        pairs.groupby([column(pairs, "technology"), column(pairs, "duration_h")])[delta_col]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    cycle_ratio_rows = []
    for _, row in pairs.iterrows():
        denominator = row.get("tail_equivalent_full_cycles_dispersed", np.nan)
        numerator = row.get("tail_equivalent_full_cycles_clustered", np.nan)
        if pd.notna(denominator) and abs(denominator) > 1e-12:
            cycle_ratio_rows.append(
                {
                    "tech_id": row[column(pairs, "technology")],
                    "duration_h": row[column(pairs, "duration_h")],
                    "cycle_retention_pct": 100 * numerator / denominator,
                }
            )
    cycle_ratio = pd.DataFrame(cycle_ratio_rows)

    fig = plt.figure(figsize=(WIDTH, 5.0))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.965)
    gs = fig.add_gridspec(
        2, 12, height_ratios=[1.15, 0.92], hspace=0.60, wspace=1.25
    )
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0:4])
    axc = fig.add_subplot(gs[1, 5:8])
    axd = fig.add_subplot(gs[1, 9:12])
    axes = [axa, axb, axc, axd]

    for tech_id, group in summary.groupby(tech):
        colour = TECH_COLOR.get(str(tech_id), MUTED)
        for chronology_name, linestyle in [("dispersed", "-"), ("clustered", "--")]:
            view = group[group["_chronology"].eq(chronology_name)].sort_values(duration)
            if view.empty:
                continue
            axa.plot(
                view[duration],
                view["median_mix_benefit"],
                color=colour,
                ls=linestyle,
                marker="o",
                markerfacecolor="white" if chronology_name == "clustered" else colour,
                markeredgewidth=0.55,
                label=(
                    TECH_LABEL.get(str(tech_id), str(tech_id))
                    if chronology_name == "dispersed"
                    else None
                ),
            )
    axa.axhline(0, color=INK, lw=0.7)
    duration_axis(axa, max(168, float(cells[duration].max())))
    axa.set_xlabel("Candidate LDES duration (h)")
    axa.set_ylabel(
        "Median incremental value\n(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axa.legend(ncol=4, loc="upper center")
    axa.text(0.99, 0.04, "solid dispersed · dashed clustered", transform=axa.transAxes, ha="right", color=MUTED)

    durations = sorted(delta[column(pairs, "duration_h")].unique())
    technologies = [item for item in TECH_ORDER if item in set(delta[column(pairs, "technology")])]
    matrix = delta.pivot(
        index=column(pairs, "technology"),
        columns=column(pairs, "duration_h"),
        values="median",
    ).reindex(index=technologies, columns=durations)
    limit = np.nanmax(np.abs(matrix.to_numpy(float)))
    image = axb.imshow(
        matrix.to_numpy(float),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-max(limit, 1e-9),
        vmax=max(limit, 1e-9),
    )
    axb.set_yticks(range(len(matrix)))
    axb.set_yticklabels([TECH_LABEL.get(str(item), str(item)) for item in matrix.index])
    shown = np.linspace(0, max(0, len(durations) - 1), min(7, len(durations))).round().astype(int)
    axb.set_xticks(shown)
    axb.set_xticklabels([f"{durations[idx]:g}" for idx in shown])
    axb.set_xlabel("Duration (h)")
    cbar = fig.colorbar(image, ax=axb, fraction=0.05, pad=0.03)
    cbar.set_label("Clustered − dispersed\nincremental value")

    box_data = [
        complete_cases(
            pairs.loc[
                pairs[column(pairs, "technology")].eq(item), delta_col
            ],
            label=f"ed4_paired_delta_{item}",
        )
        for item in technologies
    ]
    bp = axc.boxplot(
        box_data,
        vert=False,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 0.9},
        whiskerprops={"color": INK, "linewidth": 0.6},
        capprops={"color": INK, "linewidth": 0.6},
        boxprops={"edgecolor": INK, "linewidth": 0.6},
    )
    for patch, item in zip(bp["boxes"], technologies):
        patch.set_facecolor(mpl.colors.to_rgba(TECH_COLOR.get(item, MUTED), 0.35))
    axc.axvline(0, color=INK, lw=0.65)
    axc.set_yticks(range(1, len(technologies) + 1))
    axc.set_yticklabels([TECH_LABEL.get(item, item) for item in technologies])
    axc.set_xlabel("Paired incremental-value change")

    if cycle_ratio.empty:
        axd.axis("off")
    else:
        cycle_data = [
            complete_cases(
                cycle_ratio.loc[
                    cycle_ratio["tech_id"].eq(item), "cycle_retention_pct"
                ],
                label=f"ed4_cycle_retention_{item}",
            )
            for item in technologies
        ]
        bp2 = axd.boxplot(
            cycle_data,
            vert=False,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 0.9},
            boxprops={"edgecolor": INK, "linewidth": 0.6},
        )
        for patch, item in zip(bp2["boxes"], technologies):
            patch.set_facecolor(mpl.colors.to_rgba(TECH_COLOR.get(item, MUTED), 0.35))
        axd.axvline(100, color=INK, lw=0.7, ls="--")
        axd.set_yticks(range(1, len(technologies) + 1))
        axd.set_yticklabels([])
        axd.set_xlabel("Equivalent-cycle retention (%)")

    panel_labels(fig, axes)
    source_summary = save_panel(summary, "ed4_v43_full_configuration_duration_response.csv")
    source_delta = save_panel(delta, "ed4_v43_configuration_paired_effects.csv")
    source_cycles = save_panel(cycle_ratio, "ed4_v43_tail_cycle_retention.csv")
    export(
        fig,
        ED_OUT,
        "fig_v43_ed4_full_technology_duration_response",
        "ED4",
        [cells_path, pairs_path, source_summary, source_delta, source_cycles],
        "All V43 candidate technologies and registered technology-specific durations.",
    )


def ed5() -> None:
    """Calendar-year costs and the registered conditional strict-valley cell."""
    duration_path = V43_SOURCE / "v43_year_specific_duration_summary.csv"
    valley_path = V43_SOURCE / "v43_year_specific_valley_summary.csv"
    paired_path = (
        V43_SOURCE / "v43_year_specific_paired_valley_summary.csv"
    )
    records_path = V43_SOURCE / "v43_year_specific_valley_records.csv"
    required = [duration_path, valley_path, paired_path, records_path]
    if any(not path.exists() for path in required):
        raise MissingEvidence("ED5 v43 cost-year sources are incomplete")

    duration = pd.read_csv(duration_path)
    valley = pd.read_csv(valley_path)
    paired = pd.read_csv(paired_path)
    records = pd.read_csv(records_path)
    actual_case = "central_actual_year_cost"
    conditional_case = (
        "figure6_conditional_peak_m0p7_vf575_frozen_trajectory"
    )
    actual = duration[duration["revaluation_case_id"].eq(actual_case)].copy()
    actual_valleys = valley[
        valley["revaluation_case_id"].eq(actual_case)
    ].copy()
    conditional = duration[
        duration["revaluation_case_id"].eq(conditional_case)
    ].copy()
    conditional_valleys = valley[
        valley["revaluation_case_id"].eq(conditional_case)
    ].copy()
    conditional_paired = paired[
        paired["revaluation_case_id"].eq(conditional_case)
    ].copy()
    conditional_records = records[
        records["revaluation_case_id"].eq(conditional_case)
    ].copy()
    strict_clustered = conditional_records[
        conditional_records["chronology"].eq("clustered")
        & conditional_records["strict_valley"].astype(bool)
    ].copy()
    if actual.empty or conditional.empty or strict_clustered.empty:
        raise MissingEvidence("ED5 registered annual or conditional rows are absent")

    fig = plt.figure(figsize=(WIDTH, 5.0))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.965)
    gs = fig.add_gridspec(2, 12, height_ratios=[1.12, 0.92], hspace=0.60, wspace=1.0)
    axa = fig.add_subplot(gs[0, 0:7])
    axb = fig.add_subplot(gs[0, 7:12])
    axc = fig.add_subplot(gs[1, 0:7])
    axd = fig.add_subplot(gs[1, 7:12])
    axes = [axa, axb, axc, axd]

    # a: actual technology-specific cost years; line style encodes chronology.
    year_colours = {2025: "#7C8A90", 2030: GOLD, 2035: PURPLE}
    for year in [2025, 2030, 2035]:
        for chronology_name, linestyle in [
            ("dispersed", "-"),
            ("clustered", "--"),
        ]:
            group = actual[
                actual["cost_year"].eq(year)
                & actual["chronology"].eq(chronology_name)
            ].sort_values("duration_h")
            axa.plot(
                group["duration_h"],
                group["median"],
                color=year_colours[year],
                linestyle=linestyle,
                marker="o" if chronology_name == "dispersed" else "s",
                markerfacecolor="white",
                markeredgewidth=0.5,
                markevery=max(1, len(group) // 7),
                label=(
                    f"{year} · "
                    + ("D" if chronology_name == "dispersed" else "C")
                ),
            )
    axa.axhline(0, color=INK, lw=0.7)
    duration_axis(axa, 168)
    axa.set_xlabel("LDES duration (h)")
    axa.set_ylabel(
        "Median incremental benefit\n(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axa.legend(ncol=3, fontsize=5.0)
    axa.text(
        0.98,
        0.96,
        "both the LDES and 4 h LFP comparator costs update by year",
        transform=axa.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color=MUTED,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
            "pad": 1.0,
        },
    )

    # b: strict-valley counts under the three actual cost years.
    valley_matrix = actual_valleys.pivot_table(
        index="cost_year",
        columns="chronology",
        values="strict_valley_count",
        aggfunc="sum",
    ).reindex(index=[2025, 2030, 2035], columns=["dispersed", "clustered"])
    image = axb.imshow(
        valley_matrix.to_numpy(float),
        cmap="Blues",
        vmin=0,
        vmax=132,
        aspect="auto",
        origin="upper",
    )
    for row in range(valley_matrix.shape[0]):
        for col in range(valley_matrix.shape[1]):
            axb.text(
                col,
                row,
                f"{int(valley_matrix.iloc[row, col])}/132",
                ha="center",
                va="center",
                fontsize=6.0,
                color=INK,
                fontweight="bold",
            )
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["Dispersed", "Clustered"])
    axb.set_yticks(range(3))
    axb.set_yticklabels(["2025", "2030", "2035"])
    axb.set_xlabel("Chronology")
    axb.set_ylabel("Cost year")
    cax = axb.inset_axes([0.58, 0.92, 0.38, 0.025])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_ticks([0, 132])
    cax.set_title("strict valleys", fontsize=5.0, pad=1)

    # c: registered conditional peak at 0.7x the 2035 LDES fixed cost.
    for chronology_name, colour in [
        ("dispersed", DISPERSED),
        ("clustered", CLUSTERED),
    ]:
        group = conditional[
            conditional["chronology"].eq(chronology_name)
        ].sort_values("duration_h")
        axc.fill_between(
            group["duration_h"],
            group["q25"],
            group["q75"],
            color=colour,
            alpha=0.14,
            linewidth=0,
        )
        axc.plot(
            group["duration_h"],
            group["median"],
            color=colour,
            marker="o",
            markerfacecolor="white",
            label=chronology_name.capitalize(),
        )
    axc.axhline(0, color=INK, lw=0.7)
    duration_axis(axc, 168)
    axc.set_xlabel("LDES duration (h)")
    axc.set_ylabel(
        "Conditional incremental benefit\n(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axc.legend(ncol=2)
    axc.text(
        0.02,
        0.04,
        "$m_C$=0.7 · $v_F$=575 · 4 h LFP held at 2035 cost",
        transform=axc.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=MUTED,
    )

    # d: country-year boundaries at the registered conditional peak.
    order = strict_clustered.sort_values("valley_width_h").reset_index(drop=True)
    ypos = np.arange(len(order))
    axd.hlines(
        ypos,
        order["left_boundary_h"],
        order["right_boundary_h"],
        color=VALLEY,
        lw=1.4,
    )
    axd.scatter(
        order["left_boundary_h"],
        ypos,
        color=RECHARGE,
        s=7,
        label="Left boundary",
    )
    axd.scatter(
        order["right_boundary_h"],
        ypos,
        color=PURPLE,
        s=7,
        label="Re-entry",
    )
    axd.set_xscale("log")
    axd.set_xlim(20, 180)
    axd.set_xticks([24, 32, 40, 48, 72, 120, 168])
    axd.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    axd.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axd.set_yticks([])
    axd.set_xlabel("Conditional valley boundaries (h)")
    axd.set_ylabel("Clustered country-years")
    axd.legend(ncol=2, fontsize=5.0)
    induced = conditional_paired[
        conditional_paired["paired_state"].eq("cluster_induced")
    ]["n_country_years"].sum()
    axd.text(
        0.98,
        0.04,
        (
            f"induced {int(induced)}/132 · median "
            f"{strict_clustered['left_boundary_h'].median():.0f}→"
            f"{strict_clustered['right_boundary_h'].median():.0f} h"
        ),
        transform=axd.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=INK,
        fontweight="bold",
    )

    panel_labels(fig, axes)
    source_actual = save_panel(actual, "ed5_actual_cost_year_envelopes.csv")
    source_counts = save_panel(
        actual_valleys, "ed5_actual_cost_year_valley_counts.csv"
    )
    source_conditional = save_panel(
        conditional, "ed5_conditional_peak_duration_response.csv"
    )
    source_boundaries = save_panel(
        strict_clustered, "ed5_conditional_peak_boundaries.csv"
    )
    source_paired = save_panel(
        conditional_paired, "ed5_conditional_peak_paired_states.csv"
    )
    export(
        fig,
        ED_OUT,
        "fig_v43_ed5_conditional_phase_details",
        "ED5",
        [
            duration_path,
            valley_path,
            paired_path,
            records_path,
            source_actual,
            source_counts,
            source_conditional,
            source_boundaries,
            source_paired,
        ],
        (
            "Actual 2025/2030/2035 technology-specific cost revaluation and "
            "the registered conditional peak. The multiplier scales only "
            "the 2035 LDES-tail fixed cost; the 4 h LFP comparator remains "
            "at its 2035 cost."
        ),
    )


def ed6() -> None:
    """Fixed versus bounded-tail-power robustness in the V43 combination model."""
    cells, cells_path = read_portfolio("all_cells")
    envelope, envelope_path = read_portfolio("all_envelope")
    pairs, pairs_path = read_portfolio("all_config_paired")

    def select_s31(frame: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=frame.index)
        if "source_label" in frame:
            mask &= frame["source_label"].astype(str).str.lower().str.contains("s31")
        if "valuation_scenario_id" in frame:
            mask &= pd.to_numeric(
                frame["valuation_scenario_id"], errors="coerce"
            ).eq(31)
        return frame[mask].copy()

    cells = select_s31(cells)
    envelope = select_s31(envelope)
    pairs = select_s31(pairs)
    mode = column(cells, "model_mode")
    if cells[mode].astype(str).str.lower().nunique() < 2:
        raise MissingEvidence("ED6 requires both fixed and bounded-tail-power V43 runs")
    cells["_mode"] = cells[mode].astype(str).str.lower().replace({"optimized": "bounded"})
    cells["_chronology"] = canonical_chronology(cells[column(cells, "chronology")])
    envelope["_mode"] = envelope[column(envelope, "model_mode")].astype(str).str.lower().replace({"optimized": "bounded"})
    envelope["_chronology"] = canonical_chronology(envelope[column(envelope, "chronology")])
    pairs["_mode"] = pairs[column(pairs, "model_mode")].astype(str).str.lower().replace({"optimized": "bounded"})

    benefit = column(envelope, "mix_benefit")
    duration = column(envelope, "duration_h")
    summary = (
        envelope.groupby(["_mode", "_chronology", duration])[benefit]
        .median()
        .rename("median_mix_benefit")
        .reset_index()
    )
    bounded = cells[cells["_mode"].eq("bounded")].copy()
    bounded["selected_power_share_of_upper"] = (
        bounded["tail_power_mw"]
        / bounded["tail_power_upper_mw"].replace(0, np.nan)
    )
    power_summary = (
        bounded.groupby(["tech_id", "duration_h"])
        .agg(
            median_tail_power_share=("selected_power_share_of_upper", "median"),
            q25_tail_power_share=("selected_power_share_of_upper", lambda values: values.quantile(0.25)),
            q75_tail_power_share=("selected_power_share_of_upper", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )
    join_keys = ["country", "year", "chronology", "config_id", "valuation_scenario_id"]
    comparison = cells.pivot_table(
        index=[key for key in join_keys if key in cells],
        columns="_mode",
        values="mix_benefit_eur_per_kw_common_power_year",
        aggfunc="mean",
    ).reset_index()
    if {"fixed", "bounded"}.issubset(comparison.columns):
        comparison["bounded_minus_fixed"] = comparison["bounded"] - comparison["fixed"]
    else:
        raise MissingEvidence("ED6 cannot pair fixed and bounded mix benefits")
    delta_col = next(
        (
            name
            for name in [
                "clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year",
                "clustered_minus_dispersed_best_mix_benefit_eur_per_kw_common_power_year",
            ]
            if name in pairs
        ),
        None,
    )
    if delta_col is None:
        raise MissingEvidence("ED6 requires chronology-paired V43 benefits")
    operation_summary = (
        cells.groupby(["_mode", "_chronology"])
        .agg(
            tail_throughput_mwh=("tail_discharge_mwh", "median"),
            eens_equivalent_firm_supply_mwh=(
                "mix_eens_reduction_vs_extra_lfp4_mwh",
                "median",
            ),
        )
        .reset_index()
    )
    sign_keys = [
        key
        for key in [
            "country",
            "year",
            "config_id",
            "tech_id",
            "duration_h",
            "valuation_scenario_id",
        ]
        if key in pairs
    ]
    sign_pairs = (
        pairs.pivot_table(
            index=sign_keys,
            columns="_mode",
            values=delta_col,
            aggfunc="mean",
        )
        .reset_index()
    )
    if not {"fixed", "bounded"}.issubset(sign_pairs.columns):
        raise MissingEvidence("ED6 cannot pair fixed and bounded chronology effects")
    tolerance = 1e-6
    sign_pairs["jointly_zero"] = (
        sign_pairs["fixed"].abs().le(tolerance)
        & sign_pairs["bounded"].abs().le(tolerance)
    )
    sign_pairs["same_direction"] = np.sign(sign_pairs["fixed"]).eq(
        np.sign(sign_pairs["bounded"])
    )
    sign_valid = sign_pairs[~sign_pairs["jointly_zero"]].copy()
    direction_agreement = (
        sign_valid.groupby("tech_id")["same_direction"]
        .agg(agreement_pct=lambda values: 100 * values.mean(), comparisons="size")
        .reset_index()
        .sort_values("agreement_pct")
    )
    overall_agreement = 100 * float(sign_valid["same_direction"].mean())

    fig = plt.figure(figsize=(WIDTH, 4.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.965)
    gs = fig.add_gridspec(2, 12, hspace=0.60, wspace=1.05)
    axa = fig.add_subplot(gs[0, 0:5])
    axb = fig.add_subplot(gs[0, 5:12])
    axc = fig.add_subplot(gs[1, 0:6])
    axd = fig.add_subplot(gs[1, 6:12])
    axes = [axa, axb, axc, axd]

    # a: the bounded mode chooses tail power jointly with energy duration.
    for tech_id, group in power_summary.groupby("tech_id"):
        axa.plot(
            group["duration_h"],
            100 * group["median_tail_power_share"],
            marker="o",
            color=TECH_COLOR.get(str(tech_id), MUTED),
            label=TECH_LABEL.get(str(tech_id), str(tech_id)),
        )
        axa.fill_between(
            group["duration_h"],
            100 * group["q25_tail_power_share"],
            100 * group["q75_tail_power_share"],
            color=TECH_COLOR.get(str(tech_id), MUTED),
            alpha=0.10,
        )
    duration_axis(axa, max(168, float(power_summary["duration_h"].max())))
    axa.set_xlabel("Duration (h)")
    axa.set_ylabel("Selected tail power / bound (%)")
    axa.set_ylim(-2, 103)
    axa.legend(ncol=2, fontsize=5.0)

    # b: conditional S31 portfolio envelopes under fixed and bounded power.
    for mode_name, linestyle in [("fixed", "-"), ("bounded", "--")]:
        for chronology_name, colour in [("dispersed", DISPERSED), ("clustered", CLUSTERED)]:
            group = summary[
                summary["_mode"].eq(mode_name)
                & summary["_chronology"].eq(chronology_name)
            ].sort_values(duration)
            axb.plot(
                group[duration],
                group["median_mix_benefit"],
                color=colour,
                ls=linestyle,
                marker="o" if mode_name == "fixed" else "s",
                markerfacecolor="white",
                label=f"{chronology_name[0].upper()} · {mode_name}",
            )
    axb.axhline(0, color=INK, lw=0.7)
    duration_axis(axb, max(168, float(power_summary["duration_h"].max())))
    axb.set_xlabel("Duration (h)")
    axb.set_ylabel(
        "Median incremental value\n(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axb.legend(ncol=2)

    # c: native-unit throughput and EENS reduction relative to the LFP-tail arm.
    xlabels = ["D · fixed", "D · bounded", "C · fixed", "C · bounded"]
    order = [
        ("fixed", "dispersed"),
        ("bounded", "dispersed"),
        ("fixed", "clustered"),
        ("bounded", "clustered"),
    ]
    xloc = np.arange(len(order))
    throughput_values = []
    firm_values = []
    for mode_name, chronology_name in order:
        row = operation_summary[
            operation_summary["_mode"].eq(mode_name)
            & operation_summary["_chronology"].eq(chronology_name)
        ].iloc[0]
        throughput_values.append(row["tail_throughput_mwh"])
        firm_values.append(row["eens_equivalent_firm_supply_mwh"])
    axc.plot(
        xloc,
        throughput_values,
        color=RECHARGE,
        marker="o",
        label="Tail task service",
    )
    axc.plot(
        xloc,
        firm_values,
        color=PURPLE,
        marker="s",
        label="EENS reduction vs LFP tail",
    )
    axc.set_xticks(xloc)
    axc.set_xticklabels(xlabels)
    axc.set_ylabel("Median annual energy (MWh)")
    axc.legend()

    # d: sign agreement in the paired chronology response.
    ypos = np.arange(len(direction_agreement))
    axd.barh(
        ypos,
        direction_agreement["agreement_pct"],
        color=[
            TECH_COLOR.get(str(item), MUTED)
            for item in direction_agreement["tech_id"]
        ],
    )
    axd.axvline(50, color=INK, lw=0.65, ls="--")
    axd.set_yticks(ypos)
    axd.set_yticklabels(
        [
            TECH_LABEL.get(str(item), str(item))
            for item in direction_agreement["tech_id"]
        ]
    )
    axd.set_xlim(0, 100)
    axd.set_xlabel("Fixed–bounded sign agreement (%)")
    axd.text(
        0.98,
        0.04,
        f"overall {overall_agreement:.1f}%",
        transform=axd.transAxes,
        ha="right",
        color=MUTED,
    )

    panel_labels(fig, axes)
    source_summary = save_panel(summary, "ed6_fixed_bounded_envelopes.csv")
    source_power = save_panel(power_summary, "ed6_bounded_tail_power_selection.csv")
    source_comparison = save_panel(comparison, "ed6_fixed_bounded_value_pairs.csv")
    source_operation = save_panel(
        operation_summary, "ed6_tail_throughput_and_firm_supply.csv"
    )
    source_direction = save_panel(
        direction_agreement, "ed6_fixed_bounded_direction_agreement.csv"
    )
    source_pairs = save_panel(
        pairs[["_mode", "country", "year", "config_id", delta_col]],
        "ed6_fixed_bounded_chronology_effects.csv",
    )
    export(
        fig,
        ED_OUT,
        "fig_v43_ed6_fixed_bounded_tail_power",
        "ED6",
        [
            cells_path,
            envelope_path,
            pairs_path,
            source_summary,
            source_power,
            source_comparison,
            source_operation,
            source_direction,
            source_pairs,
        ],
        "V43 combination-model robustness with bounded tail power and full cost revaluation.",
    )


def ed7() -> None:
    cells, cells_path = read_portfolio("cells")
    layers, layers_path = read_portfolio("layers")
    components, components_path = read_portfolio("components")
    cells = central_s20(cells)
    layers = central_s20(layers)
    components = central_s20(components)
    layers = layers.copy()
    layers["_chronology"] = canonical_chronology(
        layers[column(layers, "chronology")]
    )
    components = components.copy()
    components["_chronology"] = canonical_chronology(
        components[column(components, "chronology")]
    )

    fig = plt.figure(figsize=(WIDTH, 4.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.965)
    gs = fig.add_gridspec(2, 12, hspace=0.55, wspace=0.95)
    axes = [
        fig.add_subplot(gs[0, 0:7]),
        fig.add_subplot(gs[0, 7:12]),
        fig.add_subplot(gs[1, 0:4]),
        fig.add_subplot(gs[1, 4:8]),
        fig.add_subplot(gs[1, 8:12]),
    ]
    axa, axb, axc, axd, axe = axes

    duration = column(layers, "duration_h")
    share_summary = (
        layers.groupby(["_chronology", duration])
        .agg(
            tail_power_share=("tail_power_share", "median"),
            tail_energy_share=("tail_energy_share", "median"),
            tail_task_share=("tail_discharge_share", "median"),
            tail_charge_share=("tail_charge_share", "median"),
        )
        .reset_index()
    )
    for name, colour in [("dispersed", DISPERSED), ("clustered", CLUSTERED)]:
        group = share_summary[share_summary["_chronology"].eq(name)].sort_values(
            duration
        )
        axa.plot(
            group[duration],
            100 * group["tail_task_share"],
            color=colour,
            marker="o",
            label=f"Tail task · {name}",
        )
    energy = (
        share_summary.groupby(duration, as_index=False)["tail_energy_share"]
        .median()
        .sort_values(duration)
    )
    axa.plot(
        energy[duration],
        100 * energy["tail_energy_share"],
        color=PURPLE,
        marker="s",
        markerfacecolor="white",
        label="Tail energy capacity",
    )
    axa.axhline(
        50,
        color=INK,
        lw=0.75,
        ls="--",
        label="Tail power capacity",
    )
    duration_axis(axa, max(168, float(pd.to_numeric(layers[duration]).max())))
    axa.set_xlabel("LDES duration (h)")
    axa.set_ylabel("Tail-layer share (%)")
    axa.set_ylim(0, 102)
    axa.legend(ncol=2, loc="upper left")

    # Cycle intensity is reported for the common 4 h layer and the selected tail.
    cycle_data = []
    for component_name, key in [
        ("4 h LFP", "base_cycles"),
        ("LDES tail", "tail_cycles"),
    ]:
        metric = column(layers, key)
        for chronology_name in ["dispersed", "clustered"]:
            values = complete_cases(
                layers.loc[layers["_chronology"].eq(chronology_name), metric],
                label=f"ed7_cycles_{component_name}_{chronology_name}",
            )
            cycle_data.append(
                {
                    "component": component_name,
                    "chronology": chronology_name,
                    "values": values,
                }
            )
    positions = [0.82, 1.18, 1.82, 2.18]
    boxes = axb.boxplot(
        [item["values"] for item in cycle_data],
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 0.9},
        whiskerprops={"color": INK, "linewidth": 0.6},
        capprops={"color": INK, "linewidth": 0.6},
        boxprops={"edgecolor": INK, "linewidth": 0.6},
    )
    for patch, item in zip(boxes["boxes"], cycle_data):
        colour = (
            DISPERSED if item["chronology"] == "dispersed" else CLUSTERED
        )
        patch.set_facecolor(mpl.colors.to_rgba(colour, 0.35))
    axb.set_xticks([1, 2])
    axb.set_xticklabels(["4 h LFP", "LDES tail"])
    axb.set_ylabel("Equivalent full cycles yr$^{-1}$")
    axb.plot([], [], color=DISPERSED, lw=5, alpha=0.35, label="Dispersed")
    axb.plot([], [], color=CLUSTERED, lw=5, alpha=0.35, label="Clustered")
    axb.legend()

    # Dispatch allocation makes the common-resource competition explicit.
    allocation_rows = []
    for chronology_name in ["dispersed", "clustered"]:
        view = share_summary[share_summary["_chronology"].eq(chronology_name)]
        for metric, label in [
            ("tail_charge_share", "Recharge"),
            ("tail_task_share", "Task service"),
        ]:
            tail = 100 * float(view[metric].median())
            allocation_rows.append(
                {
                    "chronology": chronology_name,
                    "allocation": label,
                    "base_share_pct": 100 - tail,
                    "tail_share_pct": tail,
                }
            )
    allocation = pd.DataFrame(allocation_rows)
    allocation["label"] = allocation.apply(
        lambda row: (
            ("D" if row["chronology"] == "dispersed" else "C")
            + " · "
            + row["allocation"]
        ),
        axis=1,
    )
    xpos = np.arange(len(allocation))
    axc.bar(xpos, allocation["base_share_pct"], color="#AFC7D8", label="4 h LFP")
    axc.bar(
        xpos,
        allocation["tail_share_pct"],
        bottom=allocation["base_share_pct"],
        color=RECHARGE,
        label="LDES tail",
    )
    axc.set_xticks(xpos)
    axc.set_xticklabels(allocation["label"], rotation=35, ha="right")
    axc.set_ylabel("Allocation (%)")
    axc.set_ylim(0, 100)
    axc.legend()

    tech_col = "dominant_tech_id"
    technology_share = (
        layers.groupby(["_chronology", tech_col])
        .size()
        .rename("records")
        .reset_index()
    )
    technology_share["share_pct"] = technology_share.groupby("_chronology")[
        "records"
    ].transform(lambda values: 100 * values / values.sum())
    left = np.zeros(2)
    chronology_order = ["dispersed", "clustered"]
    short_tech_label = {
        "vrfb": "VRFB",
        "fecr_flow": "Fe–Cr",
        "gravity": "Gravity",
        "molten_salt_tes": "TES",
        "caes": "A-CAES",
        "laes": "LAES",
        "h2_cavern": "H$_2$",
    }
    for tech_id in TECH_ORDER:
        values = []
        for chronology_name in chronology_order:
            match = technology_share[
                technology_share["_chronology"].eq(chronology_name)
                & technology_share[tech_col].eq(tech_id)
            ]["share_pct"]
            values.append(float(match.iloc[0]) if not match.empty else 0.0)
        axd.barh(
            [0, 1],
            values,
            left=left,
            color=TECH_COLOR.get(tech_id, MUTED),
            label=TECH_LABEL.get(tech_id, tech_id),
        )
        for row_index, width in enumerate(values):
            if width >= 7:
                axd.text(
                    left[row_index] + width / 2,
                    row_index,
                    short_tech_label.get(tech_id, tech_id),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=5.0,
                    fontweight="bold",
                )
        left += np.asarray(values)
    axd.set_yticks([0, 1])
    axd.set_yticklabels(["Dispersed", "Clustered"])
    axd.set_xlim(0, 100)
    axd.set_xlabel("Dominant configuration share (%)")
    axd.invert_yaxis()

    value_summary = (
        components.groupby(["_chronology", "component"])[
            "component_value_eur_per_kw_common_power_year"
        ]
        .median()
        .unstack("_chronology")
        .reindex(
            [
                "fixed_cost_contribution",
                "variable_om_contribution",
                "charging_cost_contribution",
                "energy_service_contribution",
                "firm_service_contribution",
            ]
        )
    )
    component_labels = [
        "Fixed cost",
        "Variable O&M",
        "Charging",
        "Energy service",
        "Firm supply",
    ]
    ypos = np.arange(len(value_summary))
    height = 0.34
    axe.barh(
        ypos - height / 2,
        value_summary["dispersed"],
        height,
        color=DISPERSED,
        label="Dispersed",
    )
    axe.barh(
        ypos + height / 2,
        value_summary["clustered"],
        height,
        color=CLUSTERED,
        label="Clustered",
    )
    axe.axvline(0, color=INK, lw=0.7)
    axe.set_yticks(ypos)
    axe.set_yticklabels(component_labels)
    axe.invert_yaxis()
    axe.set_xlabel("Median value contribution\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axe.legend()

    panel_labels(fig, axes)
    source_cells = save_panel(
        cells.head(50000), "ed7_panel_a_candidate_cells_excerpt.csv"
    )
    source_shares = save_panel(
        share_summary, "ed7_capacity_and_dispatch_shares.csv"
    )
    source_cycles = save_panel(
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "component": item["component"],
                        "chronology": item["chronology"],
                        "equivalent_full_cycles": item["values"],
                    }
                )
                for item in cycle_data
            ],
            ignore_index=True,
        ),
        "ed7_cycle_distributions.csv",
    )
    source_allocation = save_panel(allocation, "ed7_dispatch_allocation.csv")
    source_technology = save_panel(
        technology_share, "ed7_dominant_technology_shares.csv"
    )
    source_components = save_panel(
        value_summary.reset_index(), "ed7_value_component_decomposition.csv"
    )
    export(
        fig,
        ED_OUT,
        "fig_v43_ed7_fixed_scale_mix_details",
        "ED7",
        [
            cells_path,
            layers_path,
            components_path,
            source_cells,
            source_shares,
            source_cycles,
            source_allocation,
            source_technology,
            source_components,
        ],
        "S20 central fixed-scale portfolio capacity, dispatch and value decomposition.",
    )


def ed8() -> None:
    recharge, recharge_path = read_portfolio("recharge_summary")
    duration_scan, duration_path = read_portfolio("value_scan_duration")
    cost_phase, cost_phase_path = read_portfolio("cost_firm_paired")
    recharge = central_s20(recharge)
    duration_scan = central_s20(duration_scan)
    cost_phase = central_s20(cost_phase)
    if recharge.empty or duration_scan.empty or cost_phase.empty:
        raise MissingEvidence("ED8 requires V43 recharge and economic-sensitivity outputs")

    focal_configs = ["caes_24h", "h2_cavern_48h"]
    config_labels = {"caes_24h": "A-CAES 24 h", "h2_cavern_48h": "Cavern H$_2$ 48 h"}
    chronology_level = recharge[
        recharge["analysis"].eq("chronology_level")
        & recharge["intervention_id"].eq("r1")
        & recharge["config_id"].isin(focal_configs)
    ].copy()
    retention_rows = []
    for config_id in focal_configs:
        for metric, label in [
            ("tail_charge_mwh", "Charging"),
            ("tail_discharge_mwh", "Task service"),
            ("tail_equivalent_full_cycles", "Equivalent cycles"),
        ]:
            view = chronology_level[
                chronology_level["config_id"].eq(config_id)
                & chronology_level["metric"].eq(metric)
            ].set_index("chronology")
            if not {"dispersed", "clustered"}.issubset(view.index):
                continue
            retention_rows.append(
                {
                    "config_id": config_id,
                    "metric": metric,
                    "metric_label": label,
                    "clustered_to_dispersed_pct": 100
                    * float(view.loc["clustered", "country_block_mean"])
                    / float(view.loc["dispersed", "country_block_mean"]),
                }
            )
    retention = pd.DataFrame(retention_rows)

    binding = chronology_level[
        chronology_level["metric"].isin(
            ["shared_recharge_binding_hours", "tail_charge_power_binding_hours"]
        )
    ][
        [
            "config_id",
            "chronology",
            "metric",
            "country_block_mean",
            "country_block_ci_low",
            "country_block_ci_high",
        ]
    ].copy()

    intervention = recharge[
        recharge["analysis"].eq("clustered_minus_dispersed")
        & recharge["metric"].eq("mix_benefit_eur_per_kw_common_power_year")
        & recharge["config_id"].isin(focal_configs)
    ][
        [
            "config_id",
            "intervention_id",
            "recharge_quantity_multiplier",
            "country_block_mean",
            "country_block_ci_low",
            "country_block_ci_high",
        ]
    ].drop_duplicates(["config_id", "intervention_id"])

    # Energy and charging-price sensitivity at the 24 h node and a common
    # task-firm-supply unit value. The panel reports the paired clustered-minus-dispersed
    # median response, not an occurrence probability.
    firm_values = np.sort(duration_scan["firm_value_eur_kw_year"].unique())
    firm_slice = firm_values[np.argmin(np.abs(firm_values - 100))]
    economic_slice = duration_scan[
        np.isclose(duration_scan["duration_h"], 24)
        & np.isclose(duration_scan["firm_value_eur_kw_year"], firm_slice)
        & np.isclose(duration_scan["ldes_tail_fixed_cost_multiplier"], 1)
    ].copy()
    economic_pair = economic_slice.pivot_table(
        index=["energy_value_eur_mwh", "charge_price_eur_mwh"],
        columns="chronology",
        values="median",
        aggfunc="mean",
    ).reset_index()
    if {"dispersed", "clustered"}.issubset(economic_pair.columns):
        economic_pair["clustered_minus_dispersed"] = (
            economic_pair["clustered"] - economic_pair["dispersed"]
        )
    else:
        raise MissingEvidence("ED8 economic scan lacks both paired chronologies")
    economic_matrix = economic_pair.pivot(
        index="charge_price_eur_mwh",
        columns="energy_value_eur_mwh",
        values="clustered_minus_dispersed",
    ).sort_index().sort_index(axis=1)

    phase_curves = cost_phase.copy()
    available_costs = np.sort(
        phase_curves["ldes_tail_fixed_cost_multiplier"].unique()
    )
    selected_costs = []
    for target in [0.4, 0.6, 0.8, 1.0, 1.2]:
        value = float(available_costs[np.argmin(np.abs(available_costs - target))])
        if value not in selected_costs:
            selected_costs.append(value)

    fig = plt.figure(figsize=(WIDTH, 5.0))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.965)
    gs = fig.add_gridspec(
        2, 12, height_ratios=[0.92, 1.08], hspace=0.62, wspace=1.80
    )
    axa = fig.add_subplot(gs[0, 0:3])
    axb = fig.add_subplot(gs[0, 4:7])
    axc = fig.add_subplot(gs[0, 9:12])
    axd = fig.add_subplot(gs[1, 0:5])
    axe = fig.add_subplot(gs[1, 7:12])
    axes = [axa, axb, axc, axd, axe]

    metrics = ["Charging", "Task service", "Equivalent cycles"]
    xpos = np.arange(len(metrics))
    width = 0.34
    for offset, config_id, colour in [
        (-width / 2, "caes_24h", RECHARGE),
        (width / 2, "h2_cavern_48h", PURPLE),
    ]:
        view = retention.set_index(["config_id", "metric_label"]).reindex(
            pd.MultiIndex.from_product([[config_id], metrics])
        )
        axa.bar(
            xpos + offset,
            view["clustered_to_dispersed_pct"].to_numpy(float),
            width,
            color=colour,
            alpha=0.88,
            label=config_labels[config_id],
        )
    axa.axhline(100, color=INK, lw=0.7, ls="--")
    axa.set_xticks(xpos)
    axa.set_xticklabels(metrics, rotation=25, ha="right")
    axa.set_ylabel("Clustered / dispersed retention (%)")
    axa.set_ylim(0, 108)
    axa.legend(loc="upper center")

    binding_specs = [
        ("caes_24h", "shared_recharge_binding_hours", "CAES shared"),
        ("caes_24h", "tail_charge_power_binding_hours", "CAES charge"),
        ("h2_cavern_48h", "shared_recharge_binding_hours", "H$_2$ shared"),
        ("h2_cavern_48h", "tail_charge_power_binding_hours", "H$_2$ charge"),
    ]
    for ypos, (config_id, metric, label) in enumerate(binding_specs):
        view = binding[
            binding["config_id"].eq(config_id) & binding["metric"].eq(metric)
        ].set_index("chronology")
        dispersed_value = float(view.loc["dispersed", "country_block_mean"])
        clustered_value = float(view.loc["clustered", "country_block_mean"])
        axb.plot(
            [dispersed_value, clustered_value],
            [ypos, ypos],
            color="#CBD5DA",
            lw=2.4,
            zorder=1,
        )
        axb.scatter(dispersed_value, ypos, color=DISPERSED, s=28, zorder=2)
        axb.scatter(clustered_value, ypos, color=CLUSTERED, s=28, zorder=2)
    axb.set_xscale("log")
    axb.set_xlim(0.7, 900)
    axb.set_yticks(range(len(binding_specs)))
    axb.set_yticklabels([item[2] for item in binding_specs])
    axb.invert_yaxis()
    axb.set_xlabel("Binding hours yr$^{-1}$")
    axb.scatter([], [], color=DISPERSED, label="Dispersed")
    axb.scatter([], [], color=CLUSTERED, label="Clustered")
    axb.legend(ncol=2)

    for config_id, colour in [("caes_24h", RECHARGE), ("h2_cavern_48h", PURPLE)]:
        view = intervention[intervention["config_id"].eq(config_id)].sort_values(
            "recharge_quantity_multiplier"
        )
        x = view["recharge_quantity_multiplier"].to_numpy(float)
        y = view["country_block_mean"].to_numpy(float)
        lo = view["country_block_ci_low"].to_numpy(float)
        hi = view["country_block_ci_high"].to_numpy(float)
        axc.plot(x, y, color=colour, marker="o", label=config_labels[config_id])
        axc.fill_between(x, lo, hi, color=colour, alpha=0.14)
    axc.axhline(0, color=INK, lw=0.7)
    axc.set_xticks([1, 2, 4])
    axc.set_xlabel("Renewable-surplus multiplier")
    axc.set_ylabel(
        "Clustered − dispersed\nincremental value\n"
        "(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    axc.legend()

    limit = max(1e-9, float(np.nanmax(np.abs(economic_matrix.to_numpy(float)))))
    im_d = axd.imshow(
        economic_matrix.to_numpy(float),
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axd.set_xticks(range(len(economic_matrix.columns)))
    axd.set_xticklabels([f"{value:g}" for value in economic_matrix.columns])
    axd.set_yticks(range(len(economic_matrix.index)))
    axd.set_yticklabels([f"{value:g}" for value in economic_matrix.index])
    axd.set_xlabel("Energy substitution value (EUR MWh$^{-1}$)")
    axd.set_ylabel("Charging opportunity price (EUR MWh$^{-1}$)")
    axd.text(
        0.98,
        0.04,
        f"24 h node · task-firm-supply unit value {firm_slice:g}",
        transform=axd.transAxes,
        ha="right",
        color="white" if limit > 2 else INK,
        fontsize=5.0,
    )
    cax_d = axd.inset_axes([0.59, 0.92, 0.37, 0.025])
    cbar = fig.colorbar(im_d, cax=cax_d, orientation="horizontal")
    cax_d.tick_params(labelsize=5.0, length=1.2)
    cax_d.set_xlabel(
        "paired incremental value", fontsize=5.0, labelpad=0.5, color="white"
    )
    cax_d.xaxis.set_label_position("top")
    cax_d.xaxis.tick_top()

    colours = mpl.colormaps["viridis"](
        np.linspace(0.12, 0.88, len(selected_costs))
    )
    for cost, colour in zip(selected_costs, colours):
        view = phase_curves[
            np.isclose(phase_curves["ldes_tail_fixed_cost_multiplier"], cost)
        ].sort_values("firm_value_eur_kw_year")
        axe.plot(
            view["firm_value_eur_kw_year"],
            100 * view["cluster_induced_rate"],
            marker="o",
            color=colour,
            label=f"{cost:g}× cost",
        )
    axe.set_xlabel("Task-firm-supply unit value (EUR kW$^{-1}$ yr$^{-1}$)")
    axe.set_ylabel("Clustering-induced\nstrict valleys (%)")
    axe.set_ylim(-2, 102)
    axe.legend(ncol=3)

    panel_labels(fig, axes)
    source_retention = save_panel(retention, "ed8_tail_throughput_retention.csv")
    source_binding = save_panel(binding, "ed8_binding_hour_diagnostics.csv")
    source_intervention = save_panel(
        intervention, "ed8_recharge_quantity_multiplier_response.csv"
    )
    source_economic = save_panel(
        economic_pair, "ed8_energy_charge_value_sensitivity.csv"
    )
    source_phase = save_panel(
        phase_curves[
            phase_curves["ldes_tail_fixed_cost_multiplier"].isin(selected_costs)
        ],
        "ed8_firm_value_cost_response.csv",
    )
    export(
        fig,
        ED_OUT,
        "fig_v43_ed8_mix_sensitivity",
        "ED8",
        [
            recharge_path,
            duration_path,
            cost_phase_path,
            source_retention,
            source_binding,
            source_intervention,
            source_economic,
            source_phase,
        ],
        "S20 recharge intervention and deterministic economic sensitivity.",
    )


def ed9() -> None:
    signature_path = V43_SOURCE / "v43_figure5d_cluster_signature_summary.csv"
    crossing_path = (
        V43_SOURCE / "v43_figure5e_zero_crossing_country_year.csv"
    )
    crossing_summary_path = (
        V43_SOURCE / "v43_figure5e_zero_crossing_summary.csv"
    )
    association_path = (
        V43_SOURCE / "v43_ed9_natural_crossing_association_summary.csv"
    )
    records_path = V43_SOURCE / "v43_cluster_signature_country_year.csv.gz"
    required = [
        signature_path,
        crossing_path,
        crossing_summary_path,
        association_path,
        records_path,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise MissingEvidence(
            "ED9 requires the registered 2/4/6-day cluster-signature "
            f"sources: {missing}"
        )
    signature = pd.read_csv(signature_path)
    crossings = pd.read_csv(crossing_path)
    crossing_summary = pd.read_csv(crossing_summary_path)
    association = pd.read_csv(association_path)
    records = pd.read_csv(records_path)
    if set(signature["target_cluster_days"].astype(int)) != {2, 4, 6}:
        raise MissingEvidence("ED9 signature is not the registered 2/4/6-day design")
    if crossings[["country", "year"]].drop_duplicates().shape[0] != 132:
        raise MissingEvidence("ED9 crossing records do not cover 132 country-years")

    observed = crossings[crossings["crossing_status"].eq("observed")].copy()
    country_cluster = (
        observed.groupby(["country", "target_cluster_days"])[
            "zero_crossing_duration_h"
        ]
        .agg(
            median_crossing_h="median",
            q25_crossing_h=lambda values: values.quantile(0.25),
            q75_crossing_h=lambda values: values.quantile(0.75),
            observed_crossings="size",
        )
        .reset_index()
    )
    censor_counts = (
        crossings.groupby(
            ["country", "target_cluster_days", "crossing_status"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for status in ["observed", "left_censored", "right_censored"]:
        if status not in censor_counts:
            censor_counts[status] = 0
    country_cluster = censor_counts.merge(
        country_cluster,
        on=["country", "target_cluster_days"],
        how="left",
        validate="one_to_one",
    )
    country_cluster["country_years"] = country_cluster[
        ["observed", "left_censored", "right_censored"]
    ].sum(axis=1)
    country_cluster["right_censored_share"] = (
        country_cluster["right_censored"]
        / country_cluster["country_years"]
    )

    map_data = country_cluster[
        country_cluster["target_cluster_days"].eq(4)
    ][
        [
            "country",
            "median_crossing_h",
            "observed",
            "right_censored",
            "country_years",
        ]
    ].copy()
    map_observed = map_data.dropna(subset=["median_crossing_h"]).copy()

    status = (
        crossings.groupby(["target_cluster_days", "crossing_status"])
        .size()
        .rename("country_years")
        .reset_index()
    )
    status["share_pct"] = status.groupby("target_cluster_days")[
        "country_years"
    ].transform(lambda values: 100.0 * values / values.sum())

    fig = plt.figure(figsize=(WIDTH, 5.0))
    fig.subplots_adjust(left=0.070, right=0.985, bottom=0.10, top=0.965)
    gs = fig.add_gridspec(
        2, 12, height_ratios=[1.05, 0.95], hspace=0.60, wspace=1.0
    )
    axa = fig.add_subplot(gs[:, 0:6])
    axb = fig.add_subplot(gs[0, 6:9])
    axc = fig.add_subplot(gs[0, 9:12])
    axd = fig.add_subplot(gs[1, 6:12])
    axes = [axa, axb, axc, axd]

    # a: spatial heterogeneity of the registered four-day zero crossing.
    background, selected, map_values, _ = _map_shapes(
        map_observed, "median_crossing_h"
    )
    axa.add_collection(
        PatchCollection(
            background,
            facecolor="#E9EEF0",
            edgecolor="white",
            linewidth=0.25,
        )
    )
    norm = mpl.colors.Normalize(vmin=24, vmax=168)
    collection = PatchCollection(
        selected,
        cmap="YlGnBu",
        norm=norm,
        edgecolor="white",
        linewidth=0.42,
    )
    collection.set_array(np.asarray(map_values))
    axa.add_collection(collection)
    axa.set_xlim(-12, 31)
    axa.set_ylim(35, 71.5)
    axa.set_aspect(1.13)
    axa.set_xticks([])
    axa.set_yticks([])
    for spine in axa.spines.values():
        spine.set_visible(False)
    cax = axa.inset_axes([0.05, 0.01, 0.57, 0.026])
    colour_bar = mpl.colorbar.ColorbarBase(
        cax, cmap="YlGnBu", norm=norm, orientation="horizontal"
    )
    colour_bar.set_ticks([24, 48, 72, 96, 120, 168])
    cax.set_xlabel(
        "Four-day intervention · country median observed\n"
        "zero-crossing duration (h)",
        labelpad=1,
    )
    cax.tick_params(labelsize=5.0, length=1.4)
    axa.text(
        0.98,
        0.02,
        "grey · no observed crossing",
        transform=axa.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=MUTED,
    )

    # b: all censoring states are retained rather than silently discarded.
    cluster_order = [2, 4, 6]
    status_order = ["left_censored", "observed", "right_censored"]
    status_colours = {
        "left_censored": BLUE,
        "observed": RECHARGE,
        "right_censored": "#9AAAB1",
    }
    bottoms = np.zeros(len(cluster_order))
    for state in status_order:
        values = []
        for cluster_days in cluster_order:
            match = status[
                status["target_cluster_days"].eq(cluster_days)
                & status["crossing_status"].eq(state)
            ]["share_pct"]
            values.append(float(match.iloc[0]) if not match.empty else 0.0)
        axb.bar(
            np.arange(len(cluster_order)),
            values,
            bottom=bottoms,
            color=status_colours[state],
            label=state.replace("_", " "),
            width=0.72,
        )
        bottoms += np.asarray(values)
    axb.set_xticks(range(len(cluster_order)))
    axb.set_xticklabels([f"{value} d" for value in cluster_order])
    axb.set_ylim(0, 100)
    axb.set_ylabel("Country-years (%)")
    axb.set_xlabel("Controlled cluster length")
    axb.legend(fontsize=5.0, loc="upper left")

    # c: observed zero-crossing distribution; censoring counts remain visible.
    observed_by_cluster = [
        complete_cases(
            observed.loc[
                observed["target_cluster_days"].eq(cluster_days),
                "zero_crossing_duration_h",
            ],
            label=f"ed9_zero_crossing_{cluster_days}d",
        )
        for cluster_days in cluster_order
    ]
    violins = axc.violinplot(
        observed_by_cluster,
        positions=np.arange(len(cluster_order)),
        widths=0.74,
        showmedians=True,
        showextrema=False,
    )
    for body, colour in zip(
        violins["bodies"], [BLUE, GOLD, CLUSTERED]
    ):
        body.set_facecolor(colour)
        body.set_edgecolor(colour)
        body.set_alpha(0.28)
    violins["cmedians"].set_color(INK)
    violins["cmedians"].set_linewidth(0.9)
    axc.set_xticks(range(len(cluster_order)))
    axc.set_xticklabels([f"{value} d" for value in cluster_order])
    axc.set_yscale("log")
    axc.set_ylim(20, 185)
    axc.set_yticks([24, 32, 48, 72, 96, 120, 168])
    axc.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    axc.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axc.set_ylabel("Observed zero crossing (h)")
    axc.set_xlabel("Controlled cluster length")
    for position, cluster_days in enumerate(cluster_order):
        row = crossing_summary[
            crossing_summary["target_cluster_days"].eq(cluster_days)
        ].iloc[0]
        axc.text(
            position,
            176,
            f"R {int(row['right_censored'])}",
            ha="center",
            va="top",
            fontsize=5.0,
            color=MUTED,
        )

    # d: natural-event association is retained as a falsification check.
    association = association.sort_values("target_cluster_days")
    y_position = np.arange(len(association))
    point = association[
        "spearman_rho_natural_longest_event_vs_crossing"
    ].to_numpy(float)
    low = association["country_block_ci95_low"].to_numpy(float)
    high = association["country_block_ci95_high"].to_numpy(float)
    axd.errorbar(
        point,
        y_position,
        xerr=np.vstack([point - low, high - point]),
        fmt="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor=INK,
        ecolor=INK,
        elinewidth=1.1,
        capsize=2.5,
    )
    axd.axvline(0, color=INK, lw=0.65)
    axd.set_xlim(-0.78, 0.28)
    axd.set_yticks(y_position)
    axd.set_yticklabels(
        [
            f"{int(value)} d"
            for value in association["target_cluster_days"]
        ]
    )
    axd.invert_yaxis()
    axd.set_xlabel(
        "Spearman $\\rho$\n"
        "natural longest event vs controlled zero crossing"
    )
    axd.set_ylabel("Controlled cluster length")
    axd.grid(axis="x", color=GRID, lw=0.45)
    for y_value, estimate in zip(y_position, point):
        axd.text(
            estimate,
            y_value - 0.18,
            f"{estimate:+.2f}",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color=INK,
        )
    axd.text(
        0.98,
        0.50,
        "country-block 95% CI",
        transform=axd.transAxes,
        ha="right",
        va="center",
        fontsize=5.0,
        color=MUTED,
    )

    panel_labels(fig, axes)
    source_map = save_panel(
        map_data, "ed9_four_day_country_crossing_map.csv"
    )
    source_status = save_panel(
        status, "ed9_crossing_censoring_status.csv"
    )
    source_country = save_panel(
        country_cluster, "ed9_country_crossing_summary.csv"
    )
    export(
        fig,
        ED_OUT,
        "fig_v43_ed9_mix_heterogeneity",
        "ED9",
        [
            signature_path,
            crossing_path,
            crossing_summary_path,
            association_path,
            records_path,
            MAP_FILE,
            source_map,
            source_status,
            source_country,
        ],
        (
            "Registered 2/4/6-day zero-crossing heterogeneity with explicit "
            "left and right censoring."
        ),
    )


def ed10() -> None:
    summary, source_path = _eraa_summary()
    if not ERAA_FIRST_STAGE.exists():
        raise MissingEvidence(f"ERAA first-stage deployment missing: {ERAA_FIRST_STAGE}")
    metrics = pd.read_csv(source_path)
    first_stage = pd.read_csv(ERAA_FIRST_STAGE)
    baseline = metrics[metrics["variant"].eq("baseline")].iloc[0]
    joint = metrics[metrics["variant"].eq("joint")].iloc[0]
    deployed = first_stage[
        first_stage["variant"].eq("joint")
        & first_stage["candidate_discharge_power_mw"].gt(1e-9)
    ].copy()
    deployed["country"] = (
        deployed["zone"]
        .str.slice(0, 2)
        .replace({"UK": "GB", "EL": "GR"})
    )
    deployment = (
        deployed.groupby("country", as_index=False)
        .agg(
            candidate_nodes=("zone", "nunique"),
            candidate_power_mw=("candidate_discharge_power_mw", "sum"),
            candidate_charge_power_mw=("candidate_charge_power_mw", "sum"),
        )
    )
    deployment["candidate_power_gw"] = deployment["candidate_power_mw"] / 1000

    operation = pd.DataFrame(
        [
            {
                "metric": "Candidate charging",
                "change_gwh": (
                    float(joint["candidate_charge_mwh"])
                    - float(baseline["candidate_charge_mwh"])
                )
                / 1000,
            },
            {
                "metric": "VRE spill",
                "change_gwh": (
                    float(joint["vre_spill_mwh"]) - float(baseline["vre_spill_mwh"])
                )
                / 1000,
            },
            {
                "metric": "Fossil-gas generation",
                "change_gwh": (
                    float(joint["simulated_fossil_gas_generation_mwh"])
                    - float(baseline["simulated_fossil_gas_generation_mwh"])
                )
                / 1000,
            },
        ]
    )

    fig = plt.figure(figsize=(WIDTH, 4.35))
    fig.subplots_adjust(left=0.070, right=0.985, bottom=0.10, top=0.965)
    gs = fig.add_gridspec(
        2, 12, height_ratios=[0.92, 1.08], hspace=0.62, wspace=1.15
    )
    axa = fig.add_subplot(gs[:, 0:5])
    axb = fig.add_subplot(gs[0, 5:7])
    axc = fig.add_subplot(gs[0, 7:9])
    axd = fig.add_subplot(gs[0, 9:12])
    axe = fig.add_subplot(gs[1, 5:12])
    axes = [axa, axb, axc, axd, axe]

    background, selected, values, centroids = _map_shapes(
        deployment, "candidate_power_gw"
    )
    axa.add_collection(
        PatchCollection(
            background,
            facecolor="#F0F3F4",
            edgecolor="white",
            linewidth=0.25,
        )
    )
    norm = mpl.colors.Normalize(
        vmin=0,
        vmax=max(0.2, float(np.nanmax(deployment["candidate_power_gw"]))),
    )
    collection = PatchCollection(
        selected,
        cmap="BuGn",
        norm=norm,
        edgecolor="white",
        linewidth=0.42,
    )
    collection.set_array(np.asarray(values))
    axa.add_collection(collection)
    deployment["gisco_code"] = deployment["country"].replace(
        {"GB": "UK", "GR": "EL"}
    )
    deployment["lon"] = deployment["gisco_code"].map(
        {key: value[0] for key, value in centroids.items()}
    )
    deployment["lat"] = deployment["gisco_code"].map(
        {key: value[1] for key, value in centroids.items()}
    )
    valid = complete_cases(
        deployment,
        subset=["lon", "lat"],
        label="ed10_deployment_map_coordinates",
    )
    size = 12 + 60 * np.sqrt(
        valid["candidate_nodes"] / max(float(valid["candidate_nodes"].max()), 1)
    )
    axa.scatter(
        valid["lon"],
        valid["lat"],
        s=size,
        facecolor="white",
        edgecolor=INK,
        linewidth=0.6,
        alpha=0.82,
        zorder=5,
    )
    axa.set_xlim(-12, 31)
    axa.set_ylim(35, 71.5)
    axa.set_aspect(1.13)
    axa.set_xticks([])
    axa.set_yticks([])
    for spine in axa.spines.values():
        spine.set_visible(False)
    cax = axa.inset_axes([0.05, 0.01, 0.52, 0.026])
    mpl.colorbar.ColorbarBase(cax, cmap="BuGn", norm=norm, orientation="horizontal")
    cax.set_xlabel("Candidate A-CAES power (GW)", labelpad=1)
    cax.tick_params(labelsize=5, length=1.4)
    axa.text(
        0.97,
        0.04,
        f"{len(deployed)} candidate nodes · {deployed['candidate_discharge_power_mw'].sum()/1000:.3f} GW",
        transform=axa.transAxes,
        ha="right",
        color=MUTED,
        fontsize=5.1,
    )

    _draw_eraa_small_multiples([axb, axc, axd], summary)

    colours = [RECHARGE if value >= 0 else CLUSTERED for value in operation["change_gwh"]]
    ypos = np.arange(len(operation))
    axe.barh(ypos, operation["change_gwh"], color=colours)
    axe.axvline(0, color=INK, lw=0.7)
    axe.set_yticks(ypos)
    axe.set_yticklabels(operation["metric"])
    axe.invert_yaxis()
    axe.set_xlabel("Change after the fixed 24 h A-CAES tranche (GWh yr$^{-1}$)")
    for y, value in zip(ypos, operation["change_gwh"]):
        axe.text(
            value,
            y,
            f" {value:+,.0f}",
            ha="left" if value >= 0 else "right",
            va="center",
            color=INK,
            fontsize=5.2,
        )

    panel_labels(fig, axes)
    source = save_panel(summary, "ed10_eraa_system_screen.csv")
    source_deployment = save_panel(deployment, "ed10_eraa_candidate_deployment.csv")
    source_operation = save_panel(operation, "ed10_eraa_operational_changes.csv")
    export(
        fig,
        ED_OUT,
        "fig_v43_ed10_eraa_system_screen",
        "ED10",
        [
            source_path,
            ERAA_FIRST_STAGE,
            MAP_FILE,
            source,
            source_deployment,
            source_operation,
        ],
        "Native-unit ERAA WS01 fixed-capacity system-consistency screen.",
    )


def write_manifest() -> None:
    MANIFEST_OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    missing_audit_path = SOURCE_OUT / "v43_missing_data_exclusion_audit.csv"
    pd.DataFrame(MISSING_DATA_AUDIT).to_csv(missing_audit_path, index=False)
    payload = {
        "schema_version": "v43.figure-build.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figure_width_mm": FIG_WIDTH_MM,
        "missing_data_exclusion_audit": str(missing_audit_path.relative_to(HERE)),
        "portfolio_policy": (
            "Figures 3--6 and ED3--9 accept only V43 combination-model "
            "sources; ED1--2 alone are frozen, and no V41 numerical fallback "
            "is permitted."
        ),
        "records": [record.__dict__ for record in BUILD_RECORDS],
    }
    source_hashes = {}
    used_sources: set[Path] = set()
    for record in BUILD_RECORDS:
        for raw in record.sources:
            path = Path(raw).resolve()
            if path.exists() and path.is_file():
                source_hashes[str(path)] = sha256(path)
                used_sources.add(path)
    payload["source_sha256"] = source_hashes
    registry_rows = []
    registry_candidates = [
        (path, "figure_panel_source")
        for path in sorted(SOURCE_OUT.glob("*"))
        if path.is_file()
        and path.name != "v43_source_data_registry.csv"
    ]
    registry_candidates.extend(
        (path, "frozen_v43_evidence")
        for path in sorted(V43_SOURCE.glob("*"))
        if path.is_file()
    )
    for path, scope in registry_candidates:
        resolved = path.resolve()
        registry_rows.append(
            {
                "source_file": path.name,
                "source_path": str(path.relative_to(V43)),
                "source_scope": scope,
                "v43_manifest_status": (
                    "included" if resolved in used_sources else "excluded_orphan"
                ),
                "sha256": sha256(path),
            }
        )
    source_registry_path = SOURCE_OUT / "v43_source_data_registry.csv"
    pd.DataFrame(registry_rows).to_csv(source_registry_path, index=False)
    payload["source_data_registry"] = str(
        source_registry_path.relative_to(HERE)
    )
    (MANIFEST_OUT / "v43_figure_build_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame([record.__dict__ for record in BUILD_RECORDS]).to_csv(
        MANIFEST_OUT / "v43_figure_status.csv",
        index=False,
    )


def build_safely(figure_id: str, function, strict: bool) -> None:
    try:
        function()
    except MissingEvidence as exc:
        mark_pending(figure_id, str(exc))
        if strict:
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any requested figure lacks new evidence.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["figure1", "figure2", "figure3", "figure4", "figure5", "figure6", "ed"],
        help="Subset of figure1 ... figure6 and/or ed.",
    )
    args = parser.parse_args()
    configure()
    MAIN_OUT.mkdir(parents=True, exist_ok=True)
    ED_OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    functions = {
        "figure1": ("Figure 1", figure1),
        "figure2": ("Figure 2", figure2),
        "figure3": ("Figure 3", figure3),
        "figure4": ("Figure 4", figure4),
        "figure5": ("Figure 5", figure5),
        "figure6": ("Figure 6", figure6),
    }
    for target in args.targets:
        if target in functions:
            figure_id, function = functions[target]
            build_safely(figure_id, function, args.strict)
        elif target == "ed":
            build_safely("ED1--ED2", release_ed1_ed2, args.strict)
            for figure_id, function in [
                ("ED3", ed3),
                ("ED4", ed4),
                ("ED5", ed5),
                ("ED6", ed6),
                ("ED7", ed7),
                ("ED8", ed8),
                ("ED9", ed9),
                ("ED10", ed10),
            ]:
                build_safely(figure_id, function, args.strict)
        else:
            raise SystemExit(f"Unknown target: {target}")
    write_manifest()
    built = sum(record.status in {"built", "released_frozen"} for record in BUILD_RECORDS)
    pending = sum(record.status.startswith("pending") for record in BUILD_RECORDS)
    print(f"v43 figure build complete: {built} built/released; {pending} pending")


if __name__ == "__main__":
    main()
