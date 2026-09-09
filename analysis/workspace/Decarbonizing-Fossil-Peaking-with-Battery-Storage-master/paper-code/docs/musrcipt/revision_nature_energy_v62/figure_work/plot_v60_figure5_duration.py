#!/usr/bin/env python3
"""Figure 5: duration reversal, controlled dose and parity exposure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data_v60"
OUT = HERE / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
WIDTH_MM = 183
HEIGHT_MM = 218

INK = "#2B383E"
MUTED = "#6D777B"
GRID = "#DDE2DF"
EUROPE = "#D87863"
CHINA = "#D0A13F"
JAPAN = "#4E8FAC"
COLORS = {"Europe": EUROPE, "China": CHINA, "Japan": JAPAN}
MARKERS = {"Europe": "o", "China": "s", "Japan": "^"}
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
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(-THRESHOLD, color="#7065A0", lw=0.75, ls=(0, (3, 2)))
    ax.fill_between(x, low, high, color=mpl.colors.to_rgba(colour, 0.16), lw=0)
    ax.fill_between(x, y, 0, where=y < 0, color=mpl.colors.to_rgba(colour, 0.22), lw=0)
    ax.fill_between(x, y, 0, where=y >= 0, color=mpl.colors.to_rgba("#3E8A78", 0.14), lw=0)
    ax.plot(x, y, color=colour, lw=1.05)
    ax.scatter(
        x,
        y,
        s=8.5,
        marker=MARKERS[regime],
        color=mpl.colors.to_rgba(colour, 0.62),
        edgecolor="white",
        lw=0.25,
        rasterized=True,
        zorder=3,
    )
    minimum = int(np.nanargmin(y))
    ax.vlines(x[minimum], y[minimum], 0, color=colour, lw=0.72, ls=(0, (2, 2)))
    ax.scatter(
        x[minimum],
        y[minimum],
        s=31,
        marker=MARKERS[regime],
        color=colour,
        edgecolor="white",
        lw=0.5,
        zorder=4,
    )
    vertical_span = max(float(high.max() - low.min()), 1.0)
    trough_label_y = y[minimum] - 0.055 * vertical_span if regime == "Japan" else y[minimum] + 0.08 * vertical_span
    trough_label_va = "top" if regime == "Japan" else "bottom"
    ax.text(
        x[minimum],
        trough_label_y,
        f"trough {x[minimum]:g} h",
        color=colour,
        ha="center",
        va=trough_label_va,
        fontsize=5.5,
        fontweight="bold",
    )
    after = np.flatnonzero((np.arange(len(y)) > minimum) & (y >= 0))
    if after.size:
        crossing_index = int(after[0])
        ax.scatter(
            x[crossing_index],
            y[crossing_index],
            s=24,
            marker="D",
            color="#3E8A78",
            edgecolor="white",
            lw=0.45,
            zorder=4,
        )
        ax.text(
            x[crossing_index],
            y[crossing_index] + 0.05 * vertical_span,
            f"reversal {x[crossing_index]:g} h",
            color="#3E8A78",
            ha="center",
            va="bottom",
            fontsize=5.5,
            fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xlim(3.5, 780)
    ax.set_xticks([4, 8, 24, 48, 96, 240, 720], ["4", "8", "24", "48", "96", "240", "720"])
    ax.set_xlabel("storage duration (h)")
    if regime == "Europe":
        ax.set_ylabel("clustered − dispersed value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_title(regime, color=colour, fontweight="bold", pad=3)
    ax.text(0.03, 0.95, f"n = {int(frame.n_system_years.iloc[0])}", transform=ax.transAxes, color=MUTED, va="top", fontsize=5.5)
    for lo, hi, zone_colour, zone_label in [
        (4, 12, "#D0A13F", "diurnal"),
        (12, 72, "#4E8FAC", "recovery"),
        (72, 720, "#3E8A78", "cluster span"),
    ]:
        ax.fill_between(
            [lo, hi],
            [0.90, 0.90],
            [0.94, 0.94],
            transform=ax.get_xaxis_transform(),
            color=mpl.colors.to_rgba(zone_colour, 0.62),
            lw=0,
            clip_on=False,
        )
        ax.text(
            np.sqrt(lo * hi),
            0.965,
            zone_label,
            transform=ax.get_xaxis_transform(),
            color=zone_colour,
            ha="center",
            va="bottom",
            fontsize=5.5,
            fontweight="bold",
        )
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
    ax.set_title(f"{regime} parity exposure", color=COLORS[regime], fontweight="bold", pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, label, -0.18, 1.04)
    return image


def plot_exposure_panel(
    fig: plt.Figure,
    spec,
    frame: pd.DataFrame,
    regime: str,
    label: str,
    cmap: LinearSegmentedColormap,
) -> mpl.collections.QuadMesh:
    durations = [4, 8, 12, 16, 24, 48, 96, 168]
    frame = frame.loc[frame["duration_h"].isin(durations)].copy()
    pivot = frame.pivot(
        index="system_id",
        columns="duration_h",
        values="parity_exposure_frequency",
    ).reindex(columns=durations)
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    sub = spec.subgridspec(2, 1, height_ratios=[0.26, 1.0], hspace=0.05)
    profile_ax = fig.add_subplot(sub[0, 0])
    heat_ax = fig.add_subplot(sub[1, 0], sharex=profile_ax)

    totals = (
        frame.groupby("duration_h")[["parity_years", "n_observation_years"]]
        .sum()
        .reindex(durations)
    )
    profile = 100 * totals["parity_years"].to_numpy(float) / totals["n_observation_years"].to_numpy(float)
    x = np.arange(len(durations))
    colour = COLORS[regime]
    profile_ax.fill_between(
        x,
        profile,
        0,
        color=mpl.colors.to_rgba(colour, 0.18),
        lw=0,
    )
    profile_ax.plot(x, profile, color=colour, lw=0.95)
    profile_ax.set_xlim(-0.5, len(durations) - 0.5)
    profile_ax.set_ylim(0, 105)
    profile_ax.set_yticks([0, 50, 100])
    profile_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if regime == "Europe":
        profile_ax.set_ylabel("systems (%)", labelpad=2)
    else:
        profile_ax.tick_params(axis="y", labelleft=False)
    profile_ax.spines[["top", "right", "bottom"]].set_visible(False)
    hit_row = frame.loc[frame["duration_h"].eq(24)]
    hits = int(hit_row["parity_years"].sum())
    total = int(hit_row["n_observation_years"].sum())
    profile_ax.set_title(
        f"{regime}  |  {hits}/{total} at 24 h",
        color=colour,
        fontweight="bold",
        pad=2,
    )
    panel_label(profile_ax, label, -0.18, 1.05)

    edges_x = np.arange(len(durations) + 1) - 0.5
    edges_y = np.arange(len(pivot) + 1) - 0.5
    image = heat_ax.pcolormesh(
        edges_x,
        edges_y,
        pivot.to_numpy(float),
        cmap=cmap,
        vmin=0,
        vmax=1,
        shading="flat",
        edgecolors="white",
        linewidth=0.34,
    )
    heat_ax.set_xlim(-0.5, len(durations) - 0.5)
    heat_ax.set_ylim(len(pivot) - 0.5, -0.5)
    heat_ax.set_xticks(x, [str(value) for value in durations])
    heat_ax.set_yticks(
        np.arange(len(pivot)),
        [clean_system_label(regime, value) for value in pivot.index],
    )
    label_size = 4.35 if regime == "China" else 4.6
    heat_ax.tick_params(axis="both", length=0, pad=1.2, labelsize=label_size)
    if regime == "Europe":
        heat_ax.set_ylabel("system")
    heat_ax.set_xlabel("storage duration (h)")
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    return image


def plot_dose(ax: plt.Axes, dose: pd.DataFrame, crossing: pd.DataFrame) -> None:
    order = [2, 4, 6]
    durations = np.array(sorted(dose["duration_h"].unique()), float)
    matrix = (
        dose.pivot(index="target_cluster_days", columns="duration_h", values="country_block_mean")
        .reindex(index=order, columns=durations)
        .to_numpy(float)
    )
    edges = np.empty(len(durations) + 1)
    edges[1:-1] = np.sqrt(durations[:-1] * durations[1:])
    edges[0] = durations[0] ** 2 / edges[1]
    edges[-1] = durations[-1] ** 2 / edges[-2]
    limit = float(np.nanmax(np.abs(matrix)))
    cmap = LinearSegmentedColormap.from_list("dose_shift", ["#D76F54", "#F5F3EF", "#3E8A78"])
    image = ax.pcolormesh(
        edges,
        np.arange(4) - 0.5,
        matrix,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        shading="flat",
        rasterized=True,
    )
    for row, cluster_days in enumerate(order):
        cross = crossing.loc[crossing["target_cluster_days"].eq(cluster_days)].iloc[0]
        median = float(cross["median_crossing_h"])
        ax.scatter(
            median,
            row,
            s=42,
            marker="D",
            color=DOSE_COLORS[cluster_days],
            edgecolor="white",
            lw=0.65,
            zorder=4,
        )
        ax.text(
            median,
            row - 0.28,
            f"{median:.0f} h",
            color=DOSE_COLORS[cluster_days],
            ha="center",
            va="bottom",
            fontsize=5.5,
            fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks([24, 36, 48, 60, 72, 96, 120, 168], ["24", "36", "48", "60", "72", "96", "120", "168"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_yticks(range(3), ["2 d", "4 d", "6 d"])
    for tick, cluster_days in zip(ax.get_yticklabels(), order):
        tick.set_color(DOSE_COLORS[cluster_days])
        tick.set_fontweight("bold")
    ax.set_ylim(2.5, -0.5)
    ax.set_xlabel("storage duration (h)")
    ax.set_ylabel("controlled cluster length")
    ax.text(
        0.0,
        1.04,
        "Cluster length shifts the value-reversal duration",
        transform=ax.transAxes,
        color=INK,
        fontweight="bold",
        va="bottom",
    )
    cax = ax.inset_axes([0.70, 1.055, 0.27, 0.035])
    cb = plt.colorbar(image, cax=cax, orientation="horizontal")
    cb.set_ticks([-limit, 0, limit])
    cb.set_ticklabels([f"−{limit:.0f}", "0", f"+{limit:.0f}"])
    cb.ax.tick_params(labelsize=5.5, length=1.2, pad=1)
    cb.ax.set_title("value shift (EUR kW$^{-1}$ yr$^{-1}$)", fontsize=5.5, pad=1, color=MUTED)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "g", -0.09, 1.04)


def log_density(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_values = np.log10(values)
    lo, hi = float(log_values.min()), float(log_values.max())
    hist, edges = np.histogram(log_values, bins=58, range=(lo - 0.08, hi + 0.08), density=True)
    kernel_x = np.arange(-8, 9)
    kernel = np.exp(-0.5 * (kernel_x / 2.6) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(hist, kernel, mode="same")
    if density.max() > 0:
        density /= density.max()
    return 10 ** ((edges[:-1] + edges[1:]) / 2), density


def plot_parity(ax: plt.Axes, records: pd.DataFrame) -> None:
    order = ["Europe", "China", "Japan"]
    positions = [2, 1, 0]
    for y, regime in zip(positions, order):
        values = records.loc[records["regime"].eq(regime), "sequencing_shift_eur_kw_year"].to_numpy(float)
        values = values[np.isfinite(values) & (values > 0)]
        if np.any(values <= 0):
            raise ValueError("Sequencing shifts must be strictly positive on a logarithmic axis.")
        colour = COLORS[regime]
        x_density, density = log_density(values)
        ax.fill_between(
            x_density,
            y,
            y + 0.48 * density,
            color=mpl.colors.to_rgba(colour, 0.25),
            lw=0,
        )
        ax.plot(x_density, y + 0.48 * density, color=colour, lw=0.9)
        ax.scatter(
            values,
            y - 0.10 + deterministic_jitter(len(values), 0.07),
            s=8,
            color=mpl.colors.to_rgba(colour, 0.32),
            edgecolor="white",
            lw=0.18,
            rasterized=True,
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.plot([q1, q3], [y + 0.57, y + 0.57], color=colour, lw=3.0, solid_capstyle="round")
        ax.scatter(median, y + 0.57, s=31, marker=MARKERS[regime], color=colour, edgecolor="white", lw=0.55, zorder=4)
        hits = int(np.sum(values >= THRESHOLD))
        ax.text(
            0.98,
            y + 0.22,
            f"{hits}/{len(values)}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color=colour,
            fontweight="bold",
            fontsize=5.5,
        )
    ax.axvline(THRESHOLD, color="#7065A0", lw=0.9, ls=(0, (3, 2)))
    ax.text(THRESHOLD, 1.03, f"cost scale\n{THRESHOLD:.2f}", transform=ax.get_xaxis_transform(), color="#7065A0", ha="center", va="bottom", fontsize=5.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.25, 105)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100], ["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_yticks(positions, order)
    for tick, regime in zip(ax.get_yticklabels(), order):
        tick.set_color(COLORS[regime])
        tick.set_fontweight("bold")
    ax.set_ylim(-0.32, 2.75)
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
            plot_exposure_panel(
                fig,
                grid[1, index * 4 : (index + 1) * 4],
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
    save_bundle(fig, "fig_v60_figure5_duration_parity_atlas")
    plt.close(fig)


if __name__ == "__main__":
    main()
