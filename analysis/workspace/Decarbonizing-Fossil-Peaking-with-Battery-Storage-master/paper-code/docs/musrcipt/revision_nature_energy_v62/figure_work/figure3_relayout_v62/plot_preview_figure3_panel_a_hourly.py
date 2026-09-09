#!/usr/bin/env python3
"""Standalone Figure 3a preview using unsmoothed hourly dispatch."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plot_preview_figure3_recovery_relayout import (
    BASE_CHARGE,
    BASE_DISCHARGE,
    INK,
    MM,
    OUT,
    PROFILE_SOURCE,
    SOC_FILL,
    TAIL_CHARGE,
    TAIL_DISCHARGE,
    configure_publication,
    maximum_activity_window,
)


WIDTH_MM = 183.0
HEIGHT_MM = 170.0


def configure_hourly_preview() -> None:
    configure_publication()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_hourly_bundle(fig: plt.Figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_v62_preview_figure3_panel_a_hourly_dispatch"
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(
        path.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(path.with_suffix(".png"), dpi=360)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(
        path.with_name(path.name + "_grayscale").with_suffix(".png"),
        grey,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def hourly_small_multiple(
    ax: plt.Axes,
    frame: pd.DataFrame,
    power: float,
    regime: str,
    chronology: str,
    row: int,
    column: int,
) -> None:
    window = maximum_activity_window(frame, days=14)
    x = window["window_hour"].to_numpy(float)
    base_discharge = window["base_discharge_mw"].to_numpy(float) / power
    tail_discharge = window["tail_discharge_mw"].to_numpy(float) / power
    base_charge = window["base_charge_mw"].to_numpy(float) / power
    tail_charge = window["tail_charge_mw"].to_numpy(float) / power

    inventory_ax = ax.twinx()
    inventory_ax.fill_between(
        x,
        0,
        window["tail_soc_pct"].to_numpy(float),
        step="post",
        color=SOC_FILL,
        alpha=0.27,
        lw=0,
        zorder=0,
    )
    inventory_ax.set_ylim(0, 102)
    inventory_ax.set_zorder(ax.get_zorder() - 1)
    inventory_ax.patch.set_visible(False)
    ax.patch.set_alpha(0)
    if row % 2 == 1:
        inventory_ax.set_yticks([0, 50, 100])
    else:
        inventory_ax.set_yticks([])
    inventory_ax.spines[["left", "top", "bottom"]].set_visible(False)
    inventory_ax.spines["right"].set_linewidth(0.5)

    ax.fill_between(
        x,
        0,
        base_discharge,
        step="post",
        color=BASE_DISCHARGE,
        lw=0,
    )
    ax.fill_between(
        x,
        base_discharge,
        base_discharge + tail_discharge,
        step="post",
        color=TAIL_DISCHARGE,
        lw=0,
    )
    ax.fill_between(
        x,
        0,
        -base_charge,
        step="post",
        color=BASE_CHARGE,
        lw=0,
    )
    ax.fill_between(
        x,
        -base_charge,
        -(base_charge + tail_charge),
        step="post",
        color=TAIL_CHARGE,
        lw=0,
    )
    ax.axhline(0, color=INK, lw=0.48)
    ax.set_xlim(0, 336)
    ax.set_ylim(-2.75, 2.20)
    ax.set_xticks(np.arange(0, 337, 24))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.0, width=0.55, pad=1.8)
    ax.grid(axis="x", color="#E5E7E8", lw=0.35, zorder=-2)

    if row % 2 == 0:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_yticks([-2, 0, 2])
    chronology_label = (
        "dispersed"
        if chronology == "dispersed"
        else r"controlled clustering ($\leq$4 d)"
    )
    ax.set_title(
        f"{regime} · {chronology_label}",
        loc="left",
        pad=2.0,
        color="#4D4D4D",
        fontweight="bold",
    )


def main() -> None:
    configure_hourly_preview()
    profiles = pd.read_csv(
        PROFILE_SOURCE / "figure3_representative_dispatch_profiles.csv.gz"
    )
    selection = pd.read_csv(
        PROFILE_SOURCE / "figure3_representative_dispatch_selection.csv"
    )

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        8,
        1,
        left=0.075,
        right=0.93,
        bottom=0.075,
        top=0.90,
        hspace=0.20,
        height_ratios=[1, 1, 0.22, 1, 1, 0.22, 1, 1],
    )
    grid_rows = [0, 1, 3, 4, 6, 7]
    axes = []
    for grid_row in grid_rows:
        share = axes[0] if axes else None
        axes.append(fig.add_subplot(grid[grid_row, 0], sharex=share))

    panel_order = [
        ("Europe", "dispersed"),
        ("Europe", "clustered"),
        ("China", "dispersed"),
        ("China", "clustered"),
        ("Japan", "dispersed"),
        ("Japan", "clustered"),
    ]
    for row, (regime, chronology) in enumerate(panel_order):
        frame = profiles.loc[
            profiles["regime"].eq(regime)
            & profiles["chronology"].eq(chronology)
        ]
        power = float(
            selection.loc[
                selection["regime"].eq(regime)
                & selection["chronology"].eq(chronology),
                "tail_power_mw",
            ].iloc[0]
        )
        hourly_small_multiple(
            axes[row],
            frame,
            power,
            regime,
            chronology,
            row,
            0,
        )

    fig.text(0.022, 0.900, "a", fontsize=8, fontweight="bold", ha="left", va="top")
    fig.text(
        0.018,
        0.48,
        r"hourly normalized power  $P/P^*$",
        rotation=90,
        ha="center",
        va="center",
    )
    fig.text(
        0.50,
        0.025,
        "hour within most active 14-day window",
        ha="center",
        va="center",
    )
    fig.text(
        0.982,
        0.49,
        "A-CAES inventory (%)",
        rotation=270,
        ha="center",
        va="center",
    )
    fig.legend(
        handles=[
            Patch(facecolor=BASE_DISCHARGE, edgecolor="none", label="4 h LFP discharge"),
            Patch(facecolor=TAIL_DISCHARGE, edgecolor="none", label="24 h A-CAES discharge"),
            Patch(facecolor=BASE_CHARGE, edgecolor="none", label="4 h LFP charging"),
            Patch(facecolor=TAIL_CHARGE, edgecolor="none", label="24 h A-CAES charging"),
            Patch(
                facecolor=mpl.colors.to_rgba(SOC_FILL, 0.27),
                edgecolor="none",
                label="A-CAES inventory (hourly)",
            ),
        ],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        handlelength=1.25,
        columnspacing=0.9,
        borderaxespad=0,
    )

    save_hourly_bundle(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
