#!/usr/bin/env python3
"""Plot the 1950-2025 natural-weather transition parity heatmap."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
REVISION = ROOT.parent
RESULTS = REVISION / "weather_year_parity_v42" / "results"
OUT = ROOT / "figures"
SOURCE = ROOT / "source_data"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

VALUES_PATH = RESULTS / "pecd_v42_75_window_relative_values_2025.csv"
PAIRS_PATH = RESULTS / "pecd_v42_75_window_all_pairs.csv.gz"
DECADES_PATH = RESULTS / "pecd_v42_65_matched_decade_transitions.csv"
PREPARED_PATH = RESULTS / "pecd_v42_75_window_prepared_summary.csv"
SUMMARY_PATH = RESULTS / "pecd_v42_75y_parity_summary.json"

MM = 1.0 / 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 151.0

INK = "#26343A"
MUTED = "#6E7C80"
GRID = "#DDE3E3"
RED = "#8F2D2A"
CORAL = "#D87869"
NEUTRAL = "#F7F6F1"
MINT = "#83C6BE"
TEAL = "#197C75"
GOLD = "#D6A12B"
GOLD_DARK = "#765719"

PARITY_CMAP = LinearSegmentedColormap.from_list(
    "natural_transition_parity",
    [RED, CORAL, NEUTRAL, MINT, TEAL],
    N=256,
)
PARITY_CMAP.set_bad("white")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.8,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 6.1,
        "legend.fontsize": 5.9,
        "legend.frameon": False,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def panel_label(ax: plt.Axes, label: str, x: float = -0.075, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = [VALUES_PATH, PAIRS_PATH, DECADES_PATH, PREPARED_PATH, SUMMARY_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing natural-weather source data:\n" + "\n".join(missing))

    values = pd.read_csv(VALUES_PATH).sort_values("window_start_year").reset_index(drop=True)
    pairs = pd.read_csv(PAIRS_PATH).sort_values(["start_year", "end_year"]).reset_index(drop=True)
    decades = pd.read_csv(DECADES_PATH).sort_values("start_year").reset_index(drop=True)
    prepared = pd.read_csv(PREPARED_PATH).sort_values("window_start_year").reset_index(drop=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    n_years = len(values)
    expected_pairs = n_years * (n_years - 1) // 2
    if n_years != 75 or values.window_start_year.nunique() != n_years:
        raise ValueError(f"Expected 75 unique operating years, found {n_years}")
    if len(pairs) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} natural-year pairs, found {len(pairs)}")
    if len(decades) != 65:
        raise ValueError(f"Expected 65 matched ten-year transitions, found {len(decades)}")
    if not np.array_equal(values.window_start_year.to_numpy(), prepared.window_start_year.to_numpy()):
        raise ValueError("Prepared event geometry does not align with the value chronology")
    if not np.all(pairs.end_year.to_numpy() > pairs.start_year.to_numpy()):
        raise ValueError("Natural-year pair table must contain earlier-to-later transitions only")
    return values, pairs, decades, prepared, summary


def prepare_source_data(
    values: pd.DataFrame,
    pairs: pd.DataFrame,
    decades: pd.DataFrame,
    prepared: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    years = values.window_start_year.to_numpy(int)
    year_to_index = {year: index for index, year in enumerate(years)}
    pair_data = pairs.copy()
    pair_data["start_index"] = pair_data.start_year.map(year_to_index).astype(int)
    pair_data["end_index"] = pair_data.end_year.map(year_to_index).astype(int)
    pair_data["transition_over_cost_scale"] = (
        pair_data.weather_value_change_eur_kw_year / threshold
    )
    pair_data["adverse_cost_scale_crossing"] = pair_data.transition_over_cost_scale.le(-1.0)
    pair_data["beneficial_cost_scale_crossing"] = pair_data.transition_over_cost_scale.ge(1.0)
    pair_data["matched_ten_year_transition"] = pair_data.end_year.sub(pair_data.start_year).eq(10)

    decade_keys = set(zip(decades.start_year.astype(int), decades.end_year.astype(int)))
    pair_data["in_matched_decade_table"] = [
        (int(start), int(end)) in decade_keys
        for start, end in zip(pair_data.start_year, pair_data.end_year)
    ]
    if not pair_data.matched_ten_year_transition.equals(pair_data.in_matched_decade_table):
        raise ValueError("The 65 matched-decade transitions do not align with the all-pairs table")

    matrix = np.full((len(years), len(years)), np.nan, dtype=float)
    matrix[pair_data.end_index, pair_data.start_index] = pair_data.transition_over_cost_scale

    marginals = values[
        [
            "window_start_year",
            "weather_window",
            "value_2025_eur_kw_year",
            "natural_zero_recovery_window_share",
            "natural_longest_event_run_days",
        ]
    ].merge(
        prepared[
            [
                "window_start_year",
                "event_days",
                "annual_base_task_mwh",
                "annual_tail_task_mwh",
                "annual_recharge_mwh",
            ]
        ],
        on="window_start_year",
        how="left",
        validate="one_to_one",
    )

    pair_data.to_csv(SOURCE / "natural_year_transition_pairs.csv", index=False)
    marginals.to_csv(SOURCE / "natural_year_marginals.csv", index=False)
    pd.DataFrame(matrix, index=values.weather_window, columns=values.weather_window).to_csv(
        SOURCE / "natural_year_transition_matrix_normalized.csv",
        index_label="later_weather_window",
    )
    decades.to_csv(SOURCE / "matched_ten_year_transitions.csv", index=False)
    return marginals, matrix


def draw_value_chronology(ax: plt.Axes, marginals: pd.DataFrame) -> None:
    x = np.arange(len(marginals))
    value = marginals.value_2025_eur_kw_year.to_numpy(float)
    median = float(np.median(value))
    ax.fill_between(x, median, value, where=value >= median, color=MINT, alpha=0.64, linewidth=0)
    ax.fill_between(x, median, value, where=value < median, color=CORAL, alpha=0.52, linewidth=0)
    ax.plot(x, value, color=INK, lw=0.75, marker="o", ms=1.6, mfc="white", mew=0.35)
    ax.axhline(median, color=MUTED, lw=0.55, ls=(0, (3, 2)))
    ax.text(
        len(marginals) - 0.8,
        median + 0.22,
        "75-year median",
        color=MUTED,
        fontsize=5.6,
        ha="right",
        va="bottom",
    )

    best = int(np.argmax(value))
    worst = int(np.argmin(value))
    for index, color, offset in [(best, TEAL, 0.65), (worst, RED, 0.45)]:
        ax.scatter(index, value[index], s=24, color=color, edgecolor="white", linewidth=0.55, zorder=4)
        ax.text(
            index,
            value[index] + offset,
            str(marginals.weather_window.iloc[index]),
            color=color,
            fontsize=5.8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    ax.set_xlim(-0.5, len(marginals) - 0.5)
    padding = 0.13 * float(np.ptp(value))
    ax.set_ylim(float(value.min() - padding), float(value.max() + padding))
    tick_index = np.array([0, 15, 30, 45, 60, 74])
    tick_labels = marginals.weather_window.iloc[tick_index].to_list()
    ax.set_xticks(tick_index, tick_labels)
    ax.set_xlabel("Natural operating year (July–June)", labelpad=2.0)
    ax.set_ylabel(
        "Relative value\n"
        "(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    ax.grid(axis="y", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.set_title(
        "24 h A-CAES relative value vs 4 h LFP (2025 costs)",
        loc="left",
        fontweight="bold",
        pad=2.0,
    )


def draw_transition_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    marginals: pd.DataFrame,
    pairs: pd.DataFrame,
    decades: pd.DataFrame,
    threshold: float,
) -> mpl.image.AxesImage:
    norm = TwoSlopeNorm(vmin=-2.05, vcenter=0.0, vmax=2.05)
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap=PARITY_CMAP,
        norm=norm,
        rasterized=False,
    )

    n_years = len(marginals)
    ax.plot([-0.5, n_years - 0.5], [-0.5, n_years - 0.5], color=INK, lw=0.75, zorder=4)
    ten_x = np.arange(0, n_years - 10)
    ten_y = ten_x + 10
    ax.plot(ten_x, ten_y, color=GOLD, lw=0.8, alpha=0.95, zorder=5)

    adverse = pairs.loc[pairs.transition_over_cost_scale.le(-1.0)]
    ax.scatter(
        adverse.start_index,
        adverse.end_index,
        marker="s",
        s=11.0,
        facecolors="none",
        edgecolors="white",
        linewidths=0.55,
        zorder=6,
    )

    erasing = decades.loc[decades.erases_2025_2035_learning].copy()
    year_to_index = dict(zip(marginals.window_start_year.astype(int), np.arange(n_years)))
    erasing_x = erasing.start_year.map(year_to_index).to_numpy(int)
    erasing_y = erasing.end_year.map(year_to_index).to_numpy(int)
    ax.scatter(
        erasing_x,
        erasing_y,
        marker="D",
        s=29,
        facecolors=GOLD,
        edgecolors=GOLD_DARK,
        linewidths=0.65,
        zorder=7,
    )

    worst = erasing.loc[erasing.weather_value_change_eur_kw_year.idxmin()]
    target_x = year_to_index[int(worst.start_year)]
    target_y = year_to_index[int(worst.end_year)]
    ratio = float(worst.weather_value_change_eur_kw_year / threshold)
    ax.annotate(
        f"{worst.start_window} to {worst.end_window}\n"
        f"{ratio:.2f} × cost scale\n"
        f"2035 net {worst.net_2025_to_2035_value_change_eur_kw_year:.2f}",
        xy=(target_x, target_y),
        xytext=(53.0, 34.0),
        color=INK,
        fontsize=5.8,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": GOLD_DARK, "lw": 0.75},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.3},
        zorder=8,
    )

    tick_index = np.array([0, 10, 20, 30, 40, 50, 60, 70, 74])
    tick_labels = marginals.weather_window.iloc[tick_index].to_list()
    ax.set_xticks(tick_index, tick_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(tick_index, tick_labels)
    ax.set_xlim(-0.5, n_years - 0.5)
    ax.set_ylim(-0.5, n_years - 0.5)
    ax.set_xlabel("Earlier operating year")
    ax.set_ylabel("Later operating year")
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_title("75 natural years | 2,775 earlier-to-later transitions", loc="left", fontweight="bold", pad=3.0)

    handles = [
        Line2D([0], [0], color=INK, lw=0.8, label="Same-year reference"),
        Line2D([0], [0], color=GOLD, lw=1.0, label="Matched ten-year transitions"),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=4.0,
            markerfacecolor=RED,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label="Adverse shift ≥ cost scale",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markersize=4.2,
            markerfacecolor=GOLD,
            markeredgecolor=GOLD_DARK,
            markeredgewidth=0.6,
            label="Ten-year shift erases learning",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.025),
        handlelength=1.4,
        borderaxespad=0,
        labelspacing=0.55,
    )
    return image


def draw_recovery_structure(ax: plt.Axes, marginals: pd.DataFrame) -> None:
    y = np.arange(len(marginals))
    share = 100.0 * marginals.natural_zero_recovery_window_share.to_numpy(float)
    longest = marginals.natural_longest_event_run_days.to_numpy(float)
    size = 6.0 + 1.25 * (longest - longest.min())

    ax.plot(share, y, color=GRID, lw=0.7, zorder=1)
    ax.scatter(share, y, s=size, color=MUTED, edgecolor="white", linewidth=0.25, alpha=0.85, zorder=2)
    ax.axvline(float(np.median(share)), color=GOLD_DARK, lw=0.65, ls=(0, (3, 2)))
    ax.set_xlim(48, 72)
    ax.set_xticks([50, 60, 70])
    ax.set_xlabel("Zero-recovery\nwindows (%)")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.set_title("Natural clustering", loc="left", fontweight="bold", pad=3.0)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=np.sqrt(6.0 + 1.25 * (days - longest.min())),
            markerfacecolor=MUTED,
            markeredgecolor="white",
            markeredgewidth=0.25,
            label=f"{days} d",
        )
        for days in [5, 10, 15]
    ]
    ax.legend(
        handles=legend_handles,
        title="Longest run",
        title_fontsize=5.9,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        columnspacing=0.45,
        handletextpad=0.20,
    )


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    rgba = plt.imread(stem.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(stem.with_name(stem.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def main() -> None:
    values, pairs, decades, prepared, summary = load_data()
    threshold = float(summary["cost_improvement_2025_2035_eur_kw_year"])
    marginals, matrix = prepare_source_data(values, pairs, decades, prepared, threshold)
    pair_source = pd.read_csv(SOURCE / "natural_year_transition_pairs.csv")

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.115,
        right=0.985,
        bottom=0.115,
        top=0.955,
        width_ratios=[1.0, 0.19],
        height_ratios=[0.24, 1.0, 0.065],
        wspace=0.16,
        hspace=0.48,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1], sharey=ax_b)
    cax = fig.add_subplot(grid[2, 0])
    fig.add_subplot(grid[0, 1]).axis("off")
    fig.add_subplot(grid[2, 1]).axis("off")

    draw_value_chronology(ax_a, marginals)
    image = draw_transition_heatmap(ax_b, matrix, marginals, pair_source, decades, threshold)
    draw_recovery_structure(ax_c, marginals)

    # Treat the chronology and clustering axes as aligned marginals of one panel.
    fig.canvas.draw()
    a_box = ax_a.get_position()
    b_box = ax_b.get_position()
    c_box = ax_c.get_position()
    cb_box = cax.get_position()
    ax_a.set_position([b_box.x0, a_box.y0, b_box.width, a_box.height])
    ax_c.set_position([b_box.x1 + 0.035, b_box.y0, c_box.width, b_box.height])
    cax.set_position([b_box.x0, cb_box.y0, b_box.width, cb_box.height])
    fig.text(
        b_box.x0 - 0.032,
        a_box.y1 + 0.006,
        "a",
        fontsize=8.4,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    cbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[-2, -1, 0, 1, 2])
    cbar.outline.set_linewidth(0.55)
    cbar.ax.tick_params(width=0.5, length=2.4, pad=1.6)
    cbar.ax.axvline(-1, color="white", lw=1.0)
    cbar.set_label(
        "Natural-year value transition / 2025–2035 cost scale  (later − earlier)",
        labelpad=2.0,
    )

    adverse = int(pair_source.adverse_cost_scale_crossing.sum())
    beneficial = int(pair_source.beneficial_cost_scale_crossing.sum())
    erasing = int(decades.erases_2025_2035_learning.sum())
    qa = {
        "operating_years": int(len(marginals)),
        "all_earlier_to_later_pairs": int(len(pair_source)),
        "cost_scale_eur_kw_year": threshold,
        "minimum_normalized_transition": float(np.nanmin(matrix)),
        "maximum_normalized_transition": float(np.nanmax(matrix)),
        "adverse_cost_scale_crossings": adverse,
        "beneficial_cost_scale_crossings": beneficial,
        "all_absolute_cost_scale_crossings": adverse + beneficial,
        "matched_ten_year_transitions": int(len(decades)),
        "matched_ten_year_transitions_erasing_learning": erasing,
        "matrix_non_missing_cells": int(np.isfinite(matrix).sum()),
    }
    (SOURCE / "natural_year_transition_heatmap_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    stem = OUT / "fig_v60_natural_year_transition_heatmap"
    save_bundle(fig, stem)
    plt.close(fig)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
