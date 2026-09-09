#!/usr/bin/env python3
"""Preview a data-rich relayout of Figure 4 without changing the v62 manuscript."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch


HERE = Path(__file__).resolve().parent
FIGURE_WORK = HERE.parent
REVISION = FIGURE_WORK.parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 198.0
MULTIPLIERS = [1, 2, 4]
REGIMES = ["Europe", "China", "Japan"]
REGIME_LABELS = {"Europe": "NVSB", "China": "PLRP", "Japan": "CLDB"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(FIGURE_WORK))
base = load_module("v62_figure4_base", FIGURE_WORK / "plot_v60_figure4_recharge.py")

COLORS = {
    "Europe": base.CORAL,
    "China": base.GOLD,
    "Japan": base.BLUE,
}
MARKERS = {
    "Europe": "o",
    "China": "s",
    "Japan": "^",
}
RESPONSE_COLORS = {
    "deeper loss": base.CORAL,
    "unchanged": "#E5E2DC",
    "weaker loss": base.TEAL,
}
RESPONSE_HATCHES = {
    "deeper loss": None,
    "unchanged": None,
    "weaker loss": "////",
}


def configure() -> None:
    base.configure_publication()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.1,
            "axes.labelsize": 6.2,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "legend.fontsize": 5.1,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.17, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        color=base.INK,
        ha="left",
        va="bottom",
    )


def signed_value(value: float) -> str:
    """Format small effects without publishing visually distracting signed zeros."""
    if abs(value) < 0.005:
        return "0.00"
    return f"{value:+.2f}"


def save_bundle(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(
        FIGURE_WORK / "source_data_v58" / "figure4_recharge_intervention_summary.csv"
    )
    recovery = pd.read_csv(
        FIGURE_WORK / "source_data_v58" / "figure3_recovery_window_records.csv"
    )

    records = pd.read_csv(
        REVISION / "cross_scale_recharge_v58" / "three_regime_recharge_quantity_records.csv.gz"
    )
    records["regime"] = records["application_case"].str.capitalize()
    records["observation_id"] = (
        records["case_id"].astype(str) + "|" + records["case_year"].astype(str)
    )
    records = records.rename(columns={"block": "inference_block"})[
        [
            "regime",
            "observation_id",
            "inference_block",
            "recharge_multiplier",
            "chronology_effect",
        ]
    ]
    records["recharge_multiplier"] = records["recharge_multiplier"].astype(int)

    expected = {"Europe": 132, "China": 26, "Japan": 27}
    for regime, expected_n in expected.items():
        frame = records.loc[records["regime"].eq(regime)]
        counts = frame.groupby("recharge_multiplier")["observation_id"].nunique()
        if counts.reindex(MULTIPLIERS).isna().any() or not np.all(
            counts.reindex(MULTIPLIERS).to_numpy(int) == expected_n
        ):
            raise ValueError(f"Incomplete recharge response records for {regime}: {counts.to_dict()}")
        duplicates = frame.duplicated(["observation_id", "recharge_multiplier"])
        if duplicates.any():
            raise ValueError(f"Duplicate system-year multiplier records for {regime}.")

    if set(summary["regime"]) != set(REGIMES):
        raise ValueError("Recharge summary does not contain all three regimes.")
    if len(recovery) != 370:
        raise ValueError(f"Expected 370 chronology records, found {len(recovery)}.")

    records.to_csv(OUT / "figure4_relayout_system_year_source.csv", index=False)
    summary.to_csv(OUT / "figure4_relayout_block_summary_source.csv", index=False)
    return summary, recovery, records


def build_response_composition(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tolerance = 1e-6
    for regime in REGIMES:
        pivot = records.loc[records["regime"].eq(regime)].pivot(
            index="observation_id",
            columns="recharge_multiplier",
            values="chronology_effect",
        )
        for multiplier in (2, 4):
            delta = (pivot[multiplier] - pivot[1]).to_numpy(float)
            states = {
                "deeper loss": delta < -tolerance,
                "unchanged": np.abs(delta) <= tolerance,
                "weaker loss": delta > tolerance,
            }
            for state, mask in states.items():
                rows.append(
                    {
                        "regime": regime,
                        "recharge_multiplier": multiplier,
                        "response_state": state,
                        "n_system_years": int(mask.sum()),
                        "share_pct": 100 * float(mask.mean()),
                        "unchanged_tolerance_eur_kw_year": tolerance,
                    }
                )
    composition = pd.DataFrame(rows)
    composition.to_csv(
        OUT / "figure4_relayout_response_composition_source.csv", index=False
    )
    return composition


def draw_refill_phase_compact(fig: plt.Figure, spec) -> None:
    ax = fig.add_subplot(spec)
    window = np.linspace(0.0, 72.0, 241)
    resource = np.linspace(0.8, 4.2, 171)
    refill = np.minimum(resource[:, None] * window[None, :] / 36.0, 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "refill_phase_compact",
        [base.CORAL, "#F1D8CF", "#F7F5F0", "#A8CEC2", base.TEAL],
    )
    image = ax.pcolormesh(
        window,
        resource,
        refill,
        shading="auto",
        cmap=cmap,
        norm=Normalize(0, 1),
        rasterized=True,
    )
    ax.contour(
        window,
        resource,
        refill,
        levels=[0.25, 0.50, 0.75, 0.99],
        colors=base.INK,
        linewidths=0.48,
        alpha=0.72,
    )
    ax.text(
        0.98,
        0.035,
        "refill contours   25%   50%   75%   full",
        transform=ax.transAxes,
        color=base.INK,
        ha="right",
        va="bottom",
        fontsize=5.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
    )
    ax.axvline(0, color=base.INK, lw=0.7)
    ax.text(
        2.2,
        3.88,
        "zero-hour gate:\nno refill at any multiplier",
        color=base.CORAL,
        fontweight="bold",
        va="top",
        fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
    )
    ax.set_xlim(0, 72)
    ax.set_ylim(0.8, 4.2)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.set_yticks([1, 2, 3, 4], ["1×", "2×", "3×", "4×"])
    ax.set_xlabel("inter-event recovery window (h)")
    ax.set_ylabel("available-recharge multiplier")
    panel_label(ax, "a", x=-0.12, y=1.04)
    ax.spines[["top", "right"]].set_visible(False)

    cax = ax.inset_axes([1.008, 0.0, 0.019, 1.0])
    colorbar = fig.colorbar(image, cax=cax, orientation="vertical")
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["0", "50", "100"])
    colorbar.set_label("maximum inventory rebuilt (%)", fontsize=5.2, labelpad=1.5)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.tick_params(labelsize=5.1, length=1.5, pad=1.0)


def draw_window_transition(fig: plt.Figure, spec, recovery: pd.DataFrame) -> None:
    sub = spec.subgridspec(2, 1, height_ratios=[0.58, 1.0], hspace=0.48)
    window_ax = fig.add_subplot(sub[0, 0])
    state_ax = fig.add_subplot(sub[1, 0])

    medians = (
        recovery.groupby("chronology")
        .agg(
            median_window=("median_recovery_hours", "median"),
            zero_share=("zero_recovery_window_share", "median"),
        )
        .reindex(["dispersed", "clustered"])
    )
    dispersed_h = float(medians.loc["dispersed", "median_window"])
    clustered_h = float(medians.loc["clustered", "median_window"])

    window_ax.add_patch(
        FancyArrowPatch(
            (dispersed_h, 0.50),
            (clustered_h, 0.50),
            arrowstyle="-|>",
            mutation_scale=8,
            color=base.INK,
            lw=0.85,
            shrinkA=5,
            shrinkB=5,
        )
    )
    window_ax.scatter(
        [clustered_h, dispersed_h],
        [0.50, 0.50],
        s=[33, 33],
        color=[base.CORAL, base.TEAL],
        edgecolor="white",
        lw=0.55,
        zorder=3,
    )
    window_ax.text(
        clustered_h,
        0.76,
        f"clustered  {clustered_h:.0f} h",
        ha="left",
        va="center",
        color=base.CORAL,
        fontweight="bold",
        fontsize=5.5,
    )
    window_ax.text(
        dispersed_h,
        0.76,
        f"dispersed  {dispersed_h:.0f} h",
        ha="right",
        va="center",
        color=base.TEAL,
        fontweight="bold",
        fontsize=5.5,
    )
    window_ax.set_xlim(-3, 75)
    window_ax.set_ylim(0.2, 0.92)
    window_ax.set_xticks([0, 24, 48, 72])
    window_ax.set_yticks([])
    window_ax.set_xlabel("median inter-event recovery window (h)", labelpad=1.5)
    window_ax.spines[["left", "top", "right"]].set_visible(False)
    window_ax.tick_params(axis="x", pad=1.2)
    panel_label(window_ax, "b", x=-0.19, y=1.10)
    window_ax.text(
        1.0,
        1.08,
        "185 conserved sequence pairs",
        transform=window_ax.transAxes,
        ha="right",
        va="bottom",
        color=base.MUTED,
        fontsize=5.2,
    )

    zero = 100 * medians["zero_share"].to_numpy(float)
    open_share = 100 - zero
    y = np.array([1, 0])
    state_ax.barh(
        y,
        zero,
        color=base.CORAL,
        alpha=0.86,
        height=0.46,
        hatch="////",
        edgecolor="white",
        linewidth=0.35,
        label="zero-window",
    )
    state_ax.barh(
        y,
        open_share,
        left=zero,
        color=base.TEAL,
        alpha=0.86,
        height=0.46,
        label="open window",
    )
    for yi, zero_value, open_value in zip(y, zero, open_share):
        chronology = "dispersed" if yi == 1 else "clustered"
        state_ax.text(
            2.0,
            yi,
            chronology,
            color="white",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=5.4,
        )
        if zero_value >= 8:
            state_ax.text(
                0.5 * zero_value,
                yi,
                f"{zero_value:.1f}%",
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=5.6,
            )
        state_ax.text(
            zero_value + 0.5 * open_value,
            yi,
            f"{open_value:.1f}%",
            color="white" if open_value >= 18 else base.INK,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=5.6,
        )
    state_ax.set_xlim(0, 100)
    state_ax.set_yticks([])
    state_ax.set_xlabel("recovery-window state share (%)", labelpad=1.8)
    state_ax.set_xticks([0, 25, 50, 75, 100])
    state_ax.spines[["top", "right", "left"]].set_visible(False)
    state_ax.legend(
        handles=[
            Patch(facecolor=base.TEAL, edgecolor="none", label="open window"),
            Patch(
                facecolor=base.CORAL,
                edgecolor="white",
                hatch="////",
                label="zero-window",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.46),
        ncol=2,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.6,
    )


def plot_regime_increment(
    ax: plt.Axes,
    raw: pd.DataFrame,
    regime: str,
    label: str,
) -> None:
    colour = COLORS[regime]
    pivot = raw.pivot(
        index="observation_id",
        columns="recharge_multiplier",
        values="chronology_effect",
    ).reindex(columns=MULTIPLIERS)
    if pivot.isna().any().any():
        raise ValueError(f"Incomplete paired multiplier response for {regime}.")
    delta_two = pivot[2] - pivot[1]
    delta_four = pivot[4] - pivot[1]
    order = np.argsort(delta_four.to_numpy(float), kind="mergesort")
    ranked_two = delta_two.to_numpy(float)[order]
    ranked_four = delta_four.to_numpy(float)[order]
    percentile = np.linspace(0, 100, len(order))

    ax.fill_between(
        percentile,
        ranked_two,
        ranked_four,
        color=mpl.colors.to_rgba(colour, 0.15),
        lw=0,
    )
    ax.plot(
        percentile,
        ranked_two,
        color=mpl.colors.to_rgba(colour, 0.62),
        lw=0.95,
        ls=(0, (3, 2)),
    )
    ax.plot(percentile, ranked_four, color=colour, lw=1.15)
    ax.scatter(
        percentile,
        ranked_four,
        s=5.0,
        color=mpl.colors.to_rgba(colour, 0.52),
        edgecolor="white",
        lw=0.16,
        rasterized=True,
        zorder=3,
    )
    ax.axhline(0, color=base.INK, lw=0.65, ls=(0, (3, 2)))
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("system-years ranked by 4× response (%)", labelpad=2)
    if regime == "Europe":
        ax.set_ylabel(
            "incremental sequencing effect vs 1×\n(EUR kW$^{-1}$ yr$^{-1}$)",
            labelpad=2,
        )
    ax.set_title(
        REGIME_LABELS[regime],
        loc="left",
        x=0.085,
        color=colour,
        fontweight="bold",
        y=1.105,
        pad=0,
    )
    panel_label(ax, label, x=0.0, y=1.09)
    q05, median, q95 = np.quantile(ranked_four, [0.05, 0.50, 0.95])
    ax.text(
        0.03,
        1.015,
        f"n = {len(pivot)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=base.MUTED,
        fontsize=5.2,
        clip_on=False,
    )
    ax.text(
        0.97,
        1.015,
        f"4× median {signed_value(median)}  [5-95% {signed_value(q05)}, {signed_value(q95)}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=colour,
        fontweight="bold",
        fontsize=5.0,
        clip_on=False,
    )
    lower = min(float(ranked_two.min()), float(ranked_four.min()), 0.0)
    upper = max(float(ranked_two.max()), float(ranked_four.max()), 0.0)
    pad = 0.08 * max(upper - lower, 0.05)
    ax.set_ylim(lower - pad, upper + pad)
    ax.grid(axis="y", color=base.GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if regime == "Europe":
        ax.legend(
            handles=[
                Line2D([0], [0], color=mpl.colors.to_rgba(colour, 0.62), lw=1.0, ls=(0, (3, 2)), label="2× - 1×"),
                Line2D([0], [0], color=colour, lw=1.2, label="4× - 1×"),
            ],
            loc="lower left",
            frameon=False,
            ncol=2,
            handlelength=1.5,
            columnspacing=1.2,
        )


def plot_integrated_response(ax: plt.Axes, records: pd.DataFrame) -> None:
    panel_label(ax, "f", x=-0.13, y=1.04)
    cost_scale = 5.296
    summaries: list[list[np.ndarray]] = []

    for regime in REGIMES:
        frame = records.loc[records["regime"].eq(regime)].pivot(
            index="observation_id",
            columns="recharge_multiplier",
            values="chronology_effect",
        )
        summaries.append(
            [
                np.quantile(frame[1].to_numpy(float), [0.05, 0.50, 0.95]),
                np.quantile(
                (frame[4] - frame[1]).to_numpy(float),
                    [0.05, 0.50, 0.95],
                ),
            ]
        )

    medians = np.asarray([[cell[1] for cell in row] for row in summaries])
    normalized = np.clip(medians / cost_scale, -1.0, 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "cost_scale_effect",
        [base.CORAL, "#F6F4F0", base.TEAL],
    )
    image = ax.imshow(
        normalized,
        cmap=cmap,
        norm=Normalize(-1, 1),
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks([0, 1], ["1× effect", "4× - 1× response"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=True, bottom=False, length=0, pad=4)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    ax.set_yticks(np.arange(len(REGIMES)), [REGIME_LABELS[regime] for regime in REGIMES])
    ax.tick_params(axis="y", length=0, pad=4)
    for tick, regime in zip(ax.get_yticklabels(), REGIMES):
        tick.set_color(COLORS[regime])
        tick.set_fontweight("bold")

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)

    for row_index, row in enumerate(summaries):
        for column_index, quantiles in enumerate(row):
            q05, median, q95 = quantiles
            text_color = "white" if abs(normalized[row_index, column_index]) >= 0.58 else base.INK
            ax.text(
                column_index,
                row_index - 0.10,
                signed_value(float(median)),
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=6.0,
            )
            ax.text(
                column_index,
                row_index + 0.18,
                f"[{signed_value(float(q05))}, {signed_value(float(q95))}]",
                ha="center",
                va="center",
                color=text_color,
                fontsize=5.0,
            )

    cax = ax.inset_axes([0.16, -0.21, 0.68, 0.065])
    colorbar = ax.figure.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_ticklabels(["≤-1×", "0", "≥+1×"])
    colorbar.set_label("median effect / 2025-2035 cost scale", fontsize=5.0, labelpad=1.2)
    colorbar.ax.tick_params(labelsize=5.0, length=1.5, pad=1.0)

def plot_response_composition(ax: plt.Axes, composition: pd.DataFrame) -> None:
    order = [
        ("Europe", 2),
        ("Europe", 4),
        ("China", 2),
        ("China", 4),
        ("Japan", 2),
        ("Japan", 4),
    ]
    y = np.array([5.4, 4.6, 3.1, 2.3, 0.8, 0.0])
    states = ["deeper loss", "unchanged", "weaker loss"]
    left = np.zeros(len(order))
    for state in states:
        values = []
        for regime, multiplier in order:
            match = composition.loc[
                composition["regime"].eq(regime)
                & composition["recharge_multiplier"].eq(multiplier)
                & composition["response_state"].eq(state),
                "share_pct",
            ]
            values.append(float(match.iloc[0]))
        values_array = np.asarray(values)
        ax.barh(
            y,
            values_array,
            left=left,
            height=0.58,
            color=RESPONSE_COLORS[state],
            edgecolor="white",
            linewidth=0.45,
            hatch=RESPONSE_HATCHES[state],
            label=state,
        )
        for yi, start, value in zip(y, left, values_array):
            if value >= 13:
                text_color = base.INK if state == "unchanged" else "white"
                ax.text(
                    start + 0.5 * value,
                    yi,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=5.0,
                )
        left += values_array
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 5.95)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks(
        y,
        ["NVSB  2×", "NVSB  4×", "PLRP  2×", "PLRP  4×", "CLDB  2×", "CLDB  4×"],
    )
    ax.set_xlabel("system-years (%)", labelpad=2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        frameon=False,
        handlelength=0.9,
        columnspacing=0.7,
        borderaxespad=0,
        fontsize=5.0,
    )
    panel_label(ax, "g", x=-0.15, y=1.08)
    ax.spines[["top", "right", "left"]].set_visible(False)


def plot_effect_retention(ax: plt.Axes, summary: pd.DataFrame) -> None:
    positions = {"Europe": 2.0, "China": 1.0, "Japan": 0.0}
    ax.axvline(100, color=base.INK, lw=0.7, ls=(0, (3, 2)))
    for regime in REGIMES:
        frame = summary.loc[summary["regime"].eq(regime)].set_index(
            "recharge_multiplier"
        )
        value_two = float(frame.loc[2, "effect_retained_pct"])
        value_four = float(frame.loc[4, "effect_retained_pct"])
        yi = positions[regime]
        y_two = yi + 0.11
        y_four = yi - 0.11
        colour = COLORS[regime]
        ax.plot([100, value_two], [y_two, y_two], color=colour, lw=1.15, alpha=0.58)
        ax.plot([100, value_four], [y_four, y_four], color=colour, lw=2.0, alpha=0.72)
        ax.scatter(
            value_two,
            y_two,
            s=31,
            marker="o",
            facecolor="white",
            edgecolor=colour,
            lw=1.15,
            zorder=3,
        )
        ax.scatter(
            value_four,
            y_four,
            s=34,
            marker="D",
            facecolor=colour,
            edgecolor="white",
            lw=0.55,
            zorder=4,
        )
        ax.text(
            value_four + 1.5,
            y_four,
            f"{value_four:.1f}%",
            color=colour,
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=5.2,
        )
    ax.annotate(
        "deeper loss",
        xy=(188, 2.42),
        xytext=(119, 2.42),
        ha="left",
        va="center",
        color=base.MUTED,
        fontsize=5.0,
        arrowprops={"arrowstyle": "-|>", "color": base.MUTED, "lw": 0.55},
    )
    ax.set_xlim(94, 198)
    ax.set_ylim(-0.48, 2.63)
    ax.set_xticks([100, 125, 150, 175])
    ax.set_yticks([2, 1, 0], ["NVSB", "PLRP", "CLDB"])
    ax.set_xlabel("sequencing effect retained (%)", labelpad=2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", mfc="white", mec=base.INK, mew=0.9,
                color="none", label="2×", markersize=4.2,
            ),
            Line2D(
                [0], [0], marker="D", mfc=base.INK, mec="white", mew=0.5,
                color="none", label="4×", markersize=4.2,
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0,
        fontsize=5.0,
    )
    panel_label(ax, "h", x=-0.15, y=1.08)
    ax.grid(axis="x", color=base.GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)


def main() -> None:
    configure()
    summary, recovery, records = load_data()
    composition = build_response_composition(records)

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.085,
        right=0.982,
        bottom=0.073,
        top=0.958,
        height_ratios=[1.02, 0.92, 0.78],
        hspace=0.40,
        wspace=0.72,
    )

    draw_refill_phase_compact(fig, grid[0, :7])
    draw_window_transition(fig, grid[0, 7:12], recovery)

    for index, regime in enumerate(REGIMES):
        plot_regime_increment(
            fig.add_subplot(grid[1, index * 4 : (index + 1) * 4]),
            records.loc[records["regime"].eq(regime)],
            regime,
            chr(ord("c") + index),
        )

    bottom = grid[2, :].subgridspec(
        1,
        2,
        width_ratios=[1.15, 2.15],
        wspace=0.27,
    )
    ax_f = fig.add_subplot(bottom[0, 0])
    plot_integrated_response(ax_f, records)
    bottom_right = bottom[0, 1].subgridspec(
        1,
        2,
        width_ratios=[1.0, 1.30],
        wspace=0.16,
    )
    ax_g = fig.add_subplot(bottom_right[0, 0])
    ax_h = fig.add_subplot(bottom_right[0, 1])
    plot_response_composition(ax_g, composition)
    plot_effect_retention(ax_h, summary)

    # Close only the middle-to-bottom gutter without disturbing the upper legend strip.
    for axis in (ax_f, ax_g, ax_h):
        position = axis.get_position()
        axis.set_position(
            [position.x0, position.y0 + 0.020, position.width, position.height]
        )
    save_bundle(fig, "fig_v62_preview_figure4_recharge_relayout")
    plt.close(fig)


if __name__ == "__main__":
    main()
