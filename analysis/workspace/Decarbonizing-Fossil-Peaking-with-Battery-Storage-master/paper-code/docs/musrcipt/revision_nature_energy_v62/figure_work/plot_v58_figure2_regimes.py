#!/usr/bin/env python3
"""Figure 2: parallel empirical profiles and the conserved sequencing design."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from plot_v57_reference_style_figures import china_map, europe_map, japan_map


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data_v58"
SOURCE_OLD = HERE / "source_data"
SOURCE49 = HERE / "source_data_v49"
OUT = HERE / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
WIDTH_MM = 183
HEIGHT_MM = 190

INK = "#2C393E"
MUTED = "#6E777A"
GRID = "#DDE1DE"
LIGHT = "#F2F4F1"
EUROPE = "#D77A66"
CHINA = "#D0A13C"
JAPAN = "#4E8FAC"
RECHARGE = "#4A9487"
COLORS = {"Europe": EUROPE, "China": CHINA, "Japan": JAPAN}
MARKERS = {"Europe": "o", "China": "s", "Japan": "^"}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.65,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.2, fontweight="bold", ha="left", va="bottom", color=INK)


def quiet(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.45)
        ax.set_axisbelow(True)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=360)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(path.with_name(path.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def plot_maps(fig: plt.Figure, spec) -> list[plt.Axes]:
    eu = pd.read_csv(SOURCE49 / "figure2a_europe_map.csv")
    cn = pd.read_csv(SOURCE49 / "figure2b_china_map.csv")
    jp = pd.read_csv(SOURCE49 / "figure2c_japan_map.csv")
    sub = spec.subgridspec(1, 3, wspace=0.20)
    axes = [fig.add_subplot(sub[0, index]) for index in range(3)]
    cmap = LinearSegmentedColormap.from_list("event_duration", [JAPAN, "#F5F1E9", EUROPE])
    norm = Normalize(4, 16)
    europe_map(axes[0], eu.set_index("gisco_code")["median_longest_event_days"].to_dict(), cmap, norm)
    china_map(axes[1], cn.set_index("province")["longest_extreme_peaking_run_days"].to_dict(), cmap, norm)
    community = jp.groupby(["mesh_id", "latitude", "longitude"], as_index=False).agg(
        event_run_days=("longest_extreme_import_run_days", "median")
    )
    japan_map(axes[2], community, community["event_run_days"], cmap, norm)
    titles = [
        ("Europe | national", "15 countries · 132 country-years"),
        ("China | provincial", "26 provinces · 26 province-years"),
        ("Japan | community", "19 grids · 27 community-years"),
    ]
    for index, (ax, (title, subtitle), regime) in enumerate(zip(axes, titles, ["Europe", "China", "Japan"])):
        ax.set_title(f"{title}\n{subtitle}", color=COLORS[regime], fontweight="bold", fontsize=6.1, pad=1.5, linespacing=1.25)
        ax.set_anchor("C")
        panel_label(ax, chr(ord("a") + index), -0.05, 1.03)
    cax = fig.add_axes([0.405, 0.895, 0.19, 0.009])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("longest observed event (d)", fontsize=5.3, labelpad=1.0)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.1, length=1.6, pad=1)
    return axes


def plot_task_composition(ax: plt.Axes, contract: pd.DataFrame) -> None:
    order = ["Europe", "China", "Japan"]
    for position, regime in enumerate(order):
        row = contract.loc[contract["regime"].eq(regime)].iloc[0]
        share = float(row["event_task_share_pct"])
        colour = COLORS[regime]
        ax.barh(position, share, height=0.54, color=colour, alpha=0.86, edgecolor="white", lw=0.4)
        ax.barh(position, 100 - share, left=share, height=0.54, color=LIGHT, edgecolor="#C8CECA", lw=0.35, hatch="//")
        ax.text(share / 2, position, f"{share:.1f}% event", color="white", fontweight="bold", ha="center", va="center", fontsize=5.5)
        task = float(row["annual_task_value"])
        unit = str(row["annual_task_unit"])
        task_label = f"{task:,.1f} {unit}" if task >= 100 else f"{task:.2f} {unit}"
        ax.text(102, position, task_label, color=colour, va="center", fontweight="bold", fontsize=5.6)
    ax.set_yticks(range(3), order)
    ax.invert_yaxis()
    ax.set_xlim(0, 132)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("annual task carried on q75 event days (%)")
    ax.text(103, -0.74, "observed task", color=MUTED, fontsize=5.1, ha="left")
    quiet(ax, "x")
    panel_label(ax, "d", -0.055, 1.04)


def plot_task_event(fig: plt.Figure, spec, records: pd.DataFrame, summary: pd.DataFrame) -> None:
    sub = spec.subgridspec(1, 3, wspace=0.22)
    axes = [fig.add_subplot(sub[0, index]) for index in range(3)]
    y_max = max(24, float(records["event_run_days"].max()) + 1)
    for index, (ax, regime) in enumerate(zip(axes, ["Europe", "China", "Japan"])):
        frame = records.loc[records["regime"].eq(regime)]
        colour = COLORS[regime]
        marker = MARKERS[regime]
        ax.scatter(
            frame["annual_task_percentile"],
            frame["event_run_days"],
            s=13,
            marker=marker,
            facecolor=mpl.colors.to_rgba(colour, 0.45),
            edgecolor="white",
            lw=0.25,
            rasterized=True,
        )
        if frame["annual_task_percentile"].nunique() > 1:
            fit = np.polyfit(frame["annual_task_percentile"], frame["event_run_days"], 1)
            xx = np.linspace(0, 100, 100)
            ax.plot(xx, np.polyval(fit, xx), color=colour, lw=0.9)
        rho = float(summary.loc[summary["regime"].eq(regime), "spearman_rho_task_vs_event"].iloc[0])
        ax.text(0.96, 0.93, rf"$\rho={rho:+.2f}$", transform=ax.transAxes, ha="right", va="top", color=colour, fontweight="bold")
        ax.set_xlim(0, 103)
        ax.set_ylim(0, y_max)
        ax.set_xticks([0, 50, 100])
        ax.set_title(regime, color=colour, fontweight="bold", pad=2)
        ax.set_xlabel("task percentile")
        if index == 0:
            ax.set_ylabel("longest event (d)")
            panel_label(ax, "e", -0.25, 1.10)
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        quiet(ax, "y")
    axes[0].text(0.00, 1.16, "Annual task does not determine event duration", transform=axes[0].transAxes, color=INK, fontweight="bold", va="bottom")


def plot_conservation(ax: plt.Axes, chronology: pd.DataFrame, summary: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "f", -0.015, 1.01)
    ax.text(0.00, 0.99, "One intervention across all regimes: reorder intact 24 h blocks", color=INK, fontweight="bold", va="top")
    columns = [
        ("dispersed_task_normalized", "clustered_task_normalized", "task", EUROPE),
        ("dispersed_recharge_normalized", "clustered_recharge_normalized", "recharge", RECHARGE),
    ]
    max_days = len(chronology)
    for row_index, (dispersed, clustered, label, colour) in enumerate(columns):
        y_top = 0.73 - row_index * 0.31
        ax.text(0.00, y_top - 0.03, label, color=colour, fontweight="bold", va="center")
        for x0, source, title in [
            (0.12, dispersed, "dispersed"),
            (0.44, clustered, "clustered"),
        ]:
            values = chronology[source].to_numpy(float)
            values = np.abs(values) / max(float(np.nanmax(np.abs(values))), 1e-12)
            image = np.tile(values, (6, 1))
            cmap = LinearSegmentedColormap.from_list(f"{label}_{title}", ["#F5F5F2", colour])
            ax.imshow(image, extent=[x0, x0 + 0.27, y_top - 0.085, y_top + 0.085], aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest", rasterized=True)
            ax.add_patch(Rectangle((x0, y_top - 0.085), 0.27, 0.17, facecolor="none", edgecolor="#CBD0CC", lw=0.45))
            if row_index == 0:
                ax.text(x0 + 0.135, y_top + 0.12, title, ha="center", color=COLORS["Europe"] if title == "clustered" else COLORS["Japan"], fontweight="bold")
        ax.annotate("", xy=(0.43, y_top), xytext=(0.40, y_top), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.75})

    ax.text(0.12, 0.08, f"{max_days} day blocks · annual energy and within-day profiles conserved", color=MUTED, fontsize=5.7)
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        row = summary.loc[summary["regime"].eq(regime)].iloc[0]
        x = [0.76, 0.86, 0.96][index]
        ax.text(x, 0.77, regime, color=COLORS[regime], fontweight="bold", ha="center", fontsize=5.4)
        ax.text(x, 0.61, "72→0 h", color=INK, ha="center", fontweight="bold", fontsize=6.0)
        ax.text(x, 0.49, "recovery", color=MUTED, ha="center", fontsize=5.0)
        ax.text(x, 0.28, f"{row.zero_window_clustered_pct:.1f}%", color=COLORS[regime], ha="center", fontweight="bold", fontsize=6.0)
        ax.text(x, 0.16, "zero-window", color=MUTED, ha="center", fontsize=5.0)


def main() -> None:
    configure()
    contract = pd.read_csv(SOURCE / "figure2_task_contract.csv")
    records = pd.read_csv(SOURCE / "figure2_task_event_records.csv")
    summary = pd.read_csv(SOURCE / "figure2_regime_summary.csv")
    chronology = pd.read_csv(SOURCE_OLD / "figure2_panel_c_conserved_chronology.csv")

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        4,
        12,
        left=0.065,
        right=0.985,
        bottom=0.052,
        top=0.965,
        height_ratios=[1.35, 0.72, 1.02, 0.95],
        hspace=0.48,
        wspace=0.72,
    )
    plot_maps(fig, grid[0, :])
    plot_task_composition(fig.add_subplot(grid[1, :]), contract)
    plot_task_event(fig, grid[2, :], records, summary)
    plot_conservation(fig.add_subplot(grid[3, :]), chronology, summary)
    save_bundle(fig, "fig_v58_figure2_three_balancing_regimes")
    plt.close(fig)


if __name__ == "__main__":
    main()
