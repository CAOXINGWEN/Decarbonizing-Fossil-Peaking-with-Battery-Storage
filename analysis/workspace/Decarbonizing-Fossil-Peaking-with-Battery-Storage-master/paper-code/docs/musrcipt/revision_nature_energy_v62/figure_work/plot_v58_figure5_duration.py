#!/usr/bin/env python3
"""Figure 5: duration reversal, controlled dose and parity exposure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data_v58"
OUT = HERE / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
WIDTH_MM = 183
HEIGHT_MM = 224

INK = "#2B383E"
MUTED = "#6D777B"
GRID = "#DDE2DF"
EUROPE = "#D87863"
CHINA = "#D0A13F"
JAPAN = "#4E8FAC"
COLORS = {"Europe": EUROPE, "China": CHINA, "Japan": JAPAN}
MARKERS = {"Europe": "o", "China": "s", "Japan": "^"}
REGIME_LABELS = {"Europe": "NVSB", "China": "PLRP", "Japan": "CLDB"}
DOSE_COLORS = {2: "#4E8FAC", 4: "#D0A13F", 6: "#D87863"}
THRESHOLD = 5.2960380996609935


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.4,
            "axes.labelsize": 6.4,
            "axes.titlesize": 6.7,
            "xtick.labelsize": 5.3,
            "ytick.labelsize": 5.2,
            "legend.fontsize": 5.3,
            "legend.frameon": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.17, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.2, fontweight="bold", ha="left", va="bottom", color=INK)


def quiet(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.43)
        ax.set_axisbelow(True)


def deterministic_jitter(n: int, scale: float = 0.13) -> np.ndarray:
    return scale * np.sin((np.arange(n) + 1) * 2.399963)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=360)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(path.with_name(path.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def plot_response(ax: plt.Axes, frame: pd.DataFrame, regime: str, label: str) -> None:
    frame = frame.sort_values("duration_h")
    x = frame["duration_h"].to_numpy(float)
    y = frame["block_mean_effect_eur_kw_year"].to_numpy(float)
    low = frame["ci95_low"].to_numpy(float)
    high = frame["ci95_high"].to_numpy(float)
    valid = np.isfinite(x) & (x > 0)
    x, y, low, high = x[valid], y[valid], low[valid], high[valid]
    if np.any(x <= 0):
        raise ValueError("Storage duration must be strictly positive on a logarithmic axis.")
    colour = COLORS[regime]
    ax.axvspan(8, 24, color=mpl.colors.to_rgba("#D5AA4A", 0.10), lw=0)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(-THRESHOLD, color="#7065A0", lw=0.75, ls=(0, (3, 2)))
    ax.fill_between(x, low, high, color=mpl.colors.to_rgba(colour, 0.15), lw=0)
    ax.fill_between(x, y, 0, where=y < 0, color=mpl.colors.to_rgba(colour, 0.09), lw=0)
    ax.plot(x, y, color=colour, lw=1.25)
    focus = np.isin(x, [8, 24, 48, 96, 240])
    ax.scatter(x[focus], y[focus], s=18, marker=MARKERS[regime], color=colour, edgecolor="white", lw=0.45, zorder=4)
    minimum = int(np.nanargmin(y))
    ax.annotate(
        f"minimum {x[minimum]:g} h",
        xy=(x[minimum], y[minimum]),
        xytext=(0.96, 0.08),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": colour, "lw": 0.65},
        color=colour,
        ha="right",
        va="bottom",
        fontsize=5.2,
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xlim(3.5, 780)
    ax.set_xticks([4, 8, 24, 48, 96, 240, 720], ["4", "8", "24", "48", "96", "240", "720"])
    ax.set_xlabel("storage duration (h)")
    if regime == "Europe":
        ax.set_ylabel("clustered − dispersed value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_title(REGIME_LABELS[regime], color=colour, fontweight="bold", pad=3)
    ax.text(0.03, 0.95, f"n = {int(frame.n_system_years.iloc[0])}", transform=ax.transAxes, color=MUTED, va="top", fontsize=5.1)
    quiet(ax, "y")
    panel_label(ax, label, -0.18, 1.06)


def clean_system_label(regime: str, system_id: str) -> str:
    if regime == "Europe":
        return system_id
    if regime == "China":
        return system_id.replace("CN-", "")
    parts = system_id.split("-")
    return f"{parts[1]}-{parts[-1][-3:]}" if len(parts) >= 3 else system_id


def plot_exposure_heatmap(
    ax: plt.Axes,
    frame: pd.DataFrame,
    regime: str,
    label: str,
    cmap: LinearSegmentedColormap,
) -> mpl.image.AxesImage:
    durations = [8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48]
    frame = frame.loc[frame["duration_h"].isin(durations)].copy()
    pivot = frame.pivot(index="system_id", columns="duration_h", values="parity_exposure_frequency")
    pivot = pivot.reindex(columns=durations)
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    image = ax.imshow(pivot.to_numpy(float), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1, rasterized=True)
    ax.set_xticks(range(len(durations)), [str(value) for value in durations], rotation=90)
    ax.set_yticks(range(len(pivot)), [clean_system_label(regime, value) for value in pivot.index])
    heatmap_label_size = 4.1 if regime == "China" else 4.6
    ax.tick_params(axis="both", length=0, pad=1.2, labelsize=heatmap_label_size)
    if regime == "Europe":
        ax.set_ylabel("system")
    ax.set_xlabel("storage duration (h)")
    ax.set_title(f"{REGIME_LABELS[regime]} parity exposure", color=COLORS[regime], fontweight="bold", pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, label, -0.18, 1.04)
    return image


def plot_dose(ax: plt.Axes, dose: pd.DataFrame, crossing: pd.DataFrame) -> None:
    for cluster_days in [2, 4, 6]:
        frame = dose.loc[dose["target_cluster_days"].eq(cluster_days)].sort_values("duration_h")
        colour = DOSE_COLORS[cluster_days]
        x = frame["duration_h"].to_numpy(float)
        y = frame["country_block_mean"].to_numpy(float)
        low = frame["ci95_low"].to_numpy(float)
        high = frame["ci95_high"].to_numpy(float)
        valid = np.isfinite(x) & (x > 0)
        x, y, low, high = x[valid], y[valid], low[valid], high[valid]
        if np.any(x <= 0):
            raise ValueError("Storage duration must be strictly positive on a logarithmic axis.")
        ax.fill_between(x, low, high, color=mpl.colors.to_rgba(colour, 0.12), lw=0)
        ax.plot(x, y, color=colour, lw=1.2, marker="o", ms=2.8, mfc=colour, mec="white", mew=0.35, label=f"{cluster_days} d cluster")
        cross = crossing.loc[crossing["target_cluster_days"].eq(cluster_days)].iloc[0]
        median = float(cross["median_crossing_h"])
        ax.axvline(median, color=colour, lw=0.7, ls=(0, (2, 2)), alpha=0.8)
        ax.text(median, 0.97, f"{median:.0f} h", transform=ax.get_xaxis_transform(), color=colour, ha="center", va="top", fontweight="bold", fontsize=5.2)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(20, 270)
    ax.set_xticks([24, 48, 60, 96, 120, 168, 240], ["24", "48", "60", "96", "120", "168", "240"])
    ax.set_xlabel("storage duration (h)")
    ax.set_ylabel("clustered − dispersed value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.legend(loc="lower right", ncol=3, handlelength=1.3, columnspacing=0.8, handletextpad=0.3)
    quiet(ax, "y")
    panel_label(ax, "g", -0.09, 1.04)


def plot_parity(ax: plt.Axes, records: pd.DataFrame) -> None:
    order = ["Europe", "China", "Japan"]
    for y, regime in enumerate(order):
        values = records.loc[records["regime"].eq(regime), "sequencing_shift_eur_kw_year"].to_numpy(float)
        values = values[np.isfinite(values) & (values > 0)]
        if np.any(values <= 0):
            raise ValueError("Sequencing shifts must be strictly positive on a logarithmic axis.")
        colour = COLORS[regime]
        ax.scatter(values, y + deterministic_jitter(len(values), 0.13), s=9, color=mpl.colors.to_rgba(colour, 0.38), edgecolor="white", lw=0.2, rasterized=True)
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.plot([q1, q3], [y, y], color=colour, lw=3.3, solid_capstyle="round")
        ax.scatter(median, y, s=33, marker=MARKERS[regime], color=colour, edgecolor="white", lw=0.55, zorder=4)
        hits = int(np.sum(values >= THRESHOLD))
        ax.text(0.98, y, f"{hits}/{len(values)}", transform=ax.get_yaxis_transform(), ha="right", va="center", color=colour, fontweight="bold", fontsize=5.5)
    ax.axvline(THRESHOLD, color="#7065A0", lw=0.9, ls=(0, (3, 2)))
    ax.text(THRESHOLD, 1.05, f"cost scale\n{THRESHOLD:.2f}", transform=ax.get_xaxis_transform(), color="#7065A0", ha="center", va="bottom", fontsize=5.2, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.25, 105)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100], ["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_yticks(range(3), [REGIME_LABELS[regime] for regime in order])
    for tick, regime in zip(ax.get_yticklabels(), order):
        tick.set_color(COLORS[regime])
        tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("absolute 24 h sequencing shift\n(EUR kW$^{-1}$ yr$^{-1}$; log scale)")
    quiet(ax, "x")
    panel_label(ax, "h", -0.18, 1.04)


def main() -> None:
    configure()
    duration = pd.read_csv(SOURCE / "figure5_duration_response_summary.csv")
    exposure = pd.read_csv(SOURCE / "figure5_system_duration_exposure.csv")
    dose = pd.read_csv(SOURCE / "figure5_controlled_cluster_dose.csv")
    crossing = pd.read_csv(SOURCE / "figure5_controlled_crossing_summary.csv")
    records = pd.read_csv(SOURCE / "figure3_24h_value_records.csv")
    cmap = LinearSegmentedColormap.from_list("parity_exposure", ["#F2F4F1", "#E5C778", "#C86153"])

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.075,
        right=0.985,
        bottom=0.06,
        top=0.95,
        height_ratios=[0.84, 1.48, 0.92],
        hspace=0.52,
        wspace=0.85,
    )
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        plot_response(
            fig.add_subplot(grid[0, index * 4 : (index + 1) * 4]),
            duration.loc[duration["regime"].eq(regime)],
            regime,
            chr(ord("a") + index),
        )
    images = []
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        images.append(
            plot_exposure_heatmap(
                fig.add_subplot(grid[1, index * 4 : (index + 1) * 4]),
                exposure.loc[exposure["regime"].eq(regime)],
                regime,
                chr(ord("d") + index),
                cmap,
            )
        )
    cax = fig.add_axes([0.345, 0.318, 0.31, 0.009])
    cbar = fig.colorbar(images[0], cax=cax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0", "50", "100"])
    cbar.set_label("system-year share reaching matched cost scale (%)", labelpad=1.0, fontsize=5.4)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=5.1, length=1.5, pad=1)
    plot_dose(fig.add_subplot(grid[2, :8]), dose, crossing)
    plot_parity(fig.add_subplot(grid[2, 9:]), records)
    save_bundle(fig, "fig_v58_figure5_duration_parity_atlas")
    plt.close(fig)


if __name__ == "__main__":
    main()
