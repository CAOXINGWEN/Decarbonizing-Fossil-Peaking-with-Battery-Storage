#!/usr/bin/env python3
"""Build the three v62 supplementary robustness figures from frozen summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = HERE / "source_data"
OUT = HERE / "figures"

MM = 1.0 / 25.4
WIDTH_MM = 183.0
INK = "#26353C"
MUTED = "#6C777A"
GRID = "#D9DFDD"
PALE = "#F3F5F3"
CORAL = "#D76F54"
BLUE = "#4A90A8"
TEAL = "#3B8F7E"
GOLD = "#D1A13A"
PURPLE = "#7364A8"
REGIME_COLORS = {"Europe": CORAL, "China": GOLD, "Japan": BLUE}
REGIME_MARKERS = {"Europe": "o", "China": "s", "Japan": "^"}
REGIME_LABELS = {"Europe": "NVSB", "China": "PLRP", "Japan": "CLDB"}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.labelsize": 6.3,
            "axes.titlesize": 6.7,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 5.4,
            "legend.frameon": False,
            "axes.linewidth": 0.62,
            "lines.linewidth": 1.0,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.major.width": 0.50,
            "ytick.major.width": 0.50,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def quiet(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.42)
    ax.set_axisbelow(True)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=360)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(path.with_name(path.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def validate_inputs(
    cost: pd.DataFrame,
    counts: pd.DataFrame,
    duration: pd.DataFrame,
    cells: pd.DataFrame,
    contrasts: pd.DataFrame,
    cost_uncertainty: pd.DataFrame,
    natural_pairs: pd.DataFrame,
) -> None:
    assert len(cost) == 12 and not cost.isna().any().any()
    assert len(counts) == 4 and not counts.isna().any().any()
    assert len(duration) == 3
    assert duration["twelve_day_first_sustained_positive_h"].isna().all()
    assert not duration.drop(columns=["twelve_day_first_sustained_positive_h"]).isna().any().any()
    assert len(cells) == 24 and not cells.isna().any().any()
    assert len(contrasts) == 21 and not contrasts.isna().any().any()
    assert len(cost_uncertainty) == 540 and not cost_uncertainty.isna().any().any()
    assert len(natural_pairs) == 2_775
    assert set(cost_uncertainty["analysis_set"]) == {
        "cost_parameters_fixed_discount",
        "cost_parameters_and_discount_rate",
    }
    expected_cells = {(e, p, w) for e in (1.0, 4.0) for p in (1.0, 4.0) for w in ("observed", "expanded")}
    for regime in ("europe", "china", "japan"):
        observed = set(
            cells.loc[cells["regime"].eq(regime), ["energy_multiplier", "charge_power_multiplier", "window_mode"]]
            .itertuples(index=False, name=None)
        )
        assert observed == expected_cells


def plot_discount_sensitivity(
    cost: pd.DataFrame,
    counts: pd.DataFrame,
    cost_uncertainty: pd.DataFrame,
    natural_pairs: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH_MM * MM, 166 * MM), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flat
    x_ticks = [2.0, 3.6, 6.0, 8.0]
    duration_style = {
        8: (BLUE, "o"),
        16: (TEAL, "s"),
        24: (PURPLE, "^"),
    }
    for duration_h, part in cost.groupby("duration_h", sort=True):
        colour, marker = duration_style[int(duration_h)]
        x = 100 * part["discount_rate"].to_numpy()
        y = part["cost_scale_eur_kw_year"].to_numpy()
        ax_a.plot(x, y, marker=marker, color=colour, label=f"{int(duration_h)} h A-CAES")
    central = cost.loc[(cost["duration_h"].eq(24)) & np.isclose(cost["discount_rate"], 0.036)].iloc[0]
    ax_a.scatter(3.6, central["cost_scale_eur_kw_year"], s=42, facecolor="white", edgecolor=PURPLE, lw=1.2, zorder=5)
    ax_a.annotate("central: 5.30", (3.6, 5.30), xytext=(4.05, 3.7), color=PURPLE, arrowprops={"arrowstyle": "-", "color": PURPLE, "lw": 0.7})
    ax_a.set(xlabel="real discount rate (%)", ylabel="2025–2035 cost comparator\n(EUR kW$^{-1}$ yr$^{-1}$)", xticks=x_ticks)
    ax_a.legend(loc="upper left")
    quiet(ax_a, "y")
    panel_label(ax_a, "a")

    x = 100 * counts["discount_rate"].to_numpy()
    natural = [
        ("all natural-year pairs", 100 * counts["natural_pair_hits"] / counts["natural_pairs"], CORAL, "o"),
        ("adverse 10-year transitions", 100 * counts["adverse_decade_hits"] / counts["matched_decades"], TEAL, "s"),
    ]
    for label, y, colour, marker in natural:
        ax_b.plot(x, y, marker=marker, color=colour, label=label)
    ax_b.set(xlabel="real discount rate (%)", ylabel="contrasts reaching parity (%)", xticks=x_ticks, ylim=(-1, 40))
    ax_b.legend(loc="upper right")
    quiet(ax_b, "y")
    panel_label(ax_b, "b")

    denominators = {"Europe": 132, "China": 26, "Japan": 27}
    ledger_columns = {
        "EFC": {"Europe": "efc_europe_hits", "China": "efc_china_hits", "Japan": "efc_japan_hits"},
        "System cost": {"Europe": "objective_europe_hits", "China": "objective_china_hits", "Japan": "objective_japan_hits"},
    }
    for ax, (ledger, columns), label in zip((ax_c, ax_d), ledger_columns.items(), ("c", "d")):
        for regime in ("Europe", "China", "Japan"):
            y = 100 * counts[columns[regime]] / denominators[regime]
            ax.plot(x, y, marker=REGIME_MARKERS[regime], color=REGIME_COLORS[regime], label=REGIME_LABELS[regime])
        ax.set(xlabel="real discount rate (%)", ylabel="system–years reaching parity (%)", xticks=x_ticks, ylim=(-3, 105))
        ax.set_title(f"{ledger} ledger", loc="left", color=INK, fontweight="bold")
        quiet(ax, "y")
        panel_label(ax, label)
    ax_c.legend(loc="lower left", ncol=3, columnspacing=1.0, handletextpad=0.4)

    first_year = int(min(natural_pairs["start_year"].min(), natural_pairs["end_year"].min()))
    anchored = natural_pairs.loc[natural_pairs["start_year"].eq(first_year)].sort_values("end_year")
    annual_values = np.r_[0.0, anchored["weather_value_change_eur_kw_year"].to_numpy(float)]
    natural_span = float(np.quantile(annual_values, 0.95) - np.quantile(annual_values, 0.05))
    natural_pair_values = natural_pairs["absolute_weather_value_change_eur_kw_year"].to_numpy(float)
    fixed_values = cost_uncertainty.loc[
        cost_uncertainty["analysis_set"].eq("cost_parameters_fixed_discount"),
        "absolute_relative_cost_movement_eur_kw_year",
    ].to_numpy(float)
    joint_values = cost_uncertainty.loc[
        cost_uncertainty["analysis_set"].eq("cost_parameters_and_discount_rate"),
        "absolute_relative_cost_movement_eur_kw_year",
    ].to_numpy(float)

    interval_rows = [
        ("Natural-year pairs\n(n=2,775)", natural_pair_values, CORAL, "o"),
        ("Cost parameters\n(n=108)", fixed_values, PURPLE, "s"),
        ("+ discount rate\n(n=432)", joint_values, TEAL, "D"),
    ]
    for y, (label, values, colour, marker) in enumerate(interval_rows[::-1]):
        q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        ax_e.hlines(y, q05, q95, color=colour, lw=1.0)
        ax_e.hlines(y, q25, q75, color=colour, lw=4.0, alpha=0.72)
        ax_e.scatter(median, y, s=28, marker=marker, color=colour, edgecolor="white", lw=0.55, zorder=4)
    ax_e.axvline(natural_span, color=INK, lw=0.75, ls=(0, (3, 2)))
    ax_e.text(natural_span + 0.55, 2.23, f"natural span {natural_span:.2f}", color=INK, fontsize=5.2)
    ax_e.set(
        xlabel="absolute value or cost movement (EUR kW$^{-1}$ yr$^{-1}$)",
        yticks=[0, 1, 2],
        yticklabels=[item[0] for item in interval_rows[::-1]],
        ylim=(-0.55, 2.55),
        xlim=(0, 35),
    )
    quiet(ax_e, "x")
    panel_label(ax_e, "e")

    for values, colour, line_style, marker, label in (
        (fixed_values, PURPLE, "-", "s", "cost parameters (n=108)"),
        (joint_values, TEAL, (0, (4, 2)), "D", "+ discount rate (n=432)"),
    ):
        ordered = np.sort(values)
        cumulative = 100.0 * np.arange(1, len(ordered) + 1) / len(ordered)
        ax_f.step(ordered, cumulative, where="post", color=colour, ls=line_style, label=label)
        share = 100.0 * np.mean(values <= natural_span + 1e-12)
        ax_f.scatter(natural_span, share, s=27, marker=marker, color=colour, edgecolor="white", lw=0.5, zorder=4)
        ax_f.text(natural_span + 0.65, share, f"{share:.1f}%", va="center", color=colour, fontsize=5.4)
    ax_f.axvline(natural_span, color=INK, lw=0.75, ls=(0, (3, 2)))
    ax_f.set(
        xlabel="cost-movement scale (EUR kW$^{-1}$ yr$^{-1}$)",
        ylabel="scenario cells no greater than scale (%)",
        xlim=(0, 35),
        ylim=(-2, 102),
    )
    ax_f.legend(loc="lower right")
    quiet(ax_f, "both")
    panel_label(ax_f, "f")
    save_bundle(fig, "fig_v62_supplementary_figure1_discount_sensitivity")
    plt.close(fig)


def plot_duration_boundaries(duration: pd.DataFrame) -> None:
    order = ["Europe", "China", "Japan"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_MM * MM, 66 * MM), constrained_layout=True)
    for ax, regime, label in zip(axes, order, ("a", "b", "c")):
        row = duration.loc[duration["regime"].eq(regime)].iloc[0]
        colour = REGIME_COLORS[regime]
        trough_4 = (row["four_day_trough_duration_h"], row["four_day_trough_effect_eur_kw_year"])
        boundary_4 = (row["four_day_first_sustained_positive_h"], row["boundary_block_mean_eur_kw_year"])
        trough_12 = (row["twelve_day_trough_duration_h"], row["twelve_day_trough_effect_eur_kw_year"])
        ax.axhline(0, color=INK, lw=0.65)
        ax.plot([trough_4[0], boundary_4[0]], [trough_4[1], boundary_4[1]], color=GRID, lw=0.8, zorder=1)
        ax.scatter(*trough_4, s=28, marker="o", color=colour, edgecolor="white", lw=0.5, zorder=3, label="4 d trough")
        ax.errorbar(
            boundary_4[0],
            boundary_4[1],
            yerr=[[boundary_4[1] - row["boundary_ci95_low"]], [row["boundary_ci95_high"] - boundary_4[1]]],
            fmt="D",
            ms=4.2,
            color=TEAL,
            ecolor=TEAL,
            elinewidth=0.9,
            capsize=2.2,
            label="4 d positive boundary",
            zorder=4,
        )
        ax.scatter(*trough_12, s=31, marker="s", color=PURPLE, edgecolor="white", lw=0.5, zorder=3, label="12 d trough")
        low = min(trough_4[1], trough_12[1], row["boundary_ci95_low"])
        high = max(boundary_4[1], row["boundary_ci95_high"], 0)
        pad = max(1.0, 0.13 * (high - low))
        ax.set_ylim(low - pad, high + 1.8 * pad)
        ax.annotate(
            "12 d positive boundary >168 h",
            xy=(166, high + 0.15 * pad),
            xytext=(84, high + 0.75 * pad),
            ha="center",
            color=PURPLE,
            fontsize=5.2,
            arrowprops={"arrowstyle": "-|>", "color": PURPLE, "lw": 0.7},
        )
        ax.set(xlim=(20, 172), xticks=[24, 48, 96, 144, 168], xlabel="storage duration (h)")
        ax.set_title(REGIME_LABELS[regime], loc="left", color=colour, fontweight="bold")
        quiet(ax, "y")
        panel_label(ax, label)
    axes[0].set_ylabel("clustered – dispersed system-cost effect\n(EUR kW$^{-1}$ yr$^{-1}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, handletextpad=0.4, columnspacing=1.1)
    save_bundle(fig, "fig_v62_supplementary_figure2_duration_boundaries")
    plt.close(fig)


def plot_factorial(cells: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(WIDTH_MM * MM, 132 * MM), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.08, 1.0])
    regimes = [("Europe", "europe", 132), ("China", "china", 26), ("Japan", "japan", 27)]
    conditions = [
        ("observed", 1.0, 1.0, "Observed | E1 P1"),
        ("observed", 4.0, 1.0, "Observed | E4 P1"),
        ("observed", 1.0, 4.0, "Observed | E1 P4"),
        ("observed", 4.0, 4.0, "Observed | E4 P4"),
        ("expanded", 1.0, 1.0, "Expanded | E1 P1"),
        ("expanded", 4.0, 1.0, "Expanded | E4 P1"),
        ("expanded", 1.0, 4.0, "Expanded | E1 P4"),
        ("expanded", 4.0, 4.0, "Expanded | E4 P4"),
    ]
    y_positions = np.array([7, 6, 5, 4, 2.5, 1.5, 0.5, -0.5])
    for col, (display, key, n_systems) in enumerate(regimes):
        ax = fig.add_subplot(grid[0, col])
        part = cells.loc[cells["regime"].eq(key)]
        for y, (window, energy, power, _) in zip(y_positions, conditions):
            row = part.loc[
                part["window_mode"].eq(window)
                & part["energy_multiplier"].eq(energy)
                & part["charge_power_multiplier"].eq(power)
            ].iloc[0]
            mean = row["block_mean_effect_eur_kw_year"]
            low = row["ci95_low"]
            high = row["ci95_high"]
            marker = "s" if power == 4.0 else "o"
            face = REGIME_COLORS[display] if energy == 4.0 else "white"
            ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="none", color=REGIME_COLORS[display], lw=0.85, capsize=1.8)
            ax.scatter(mean, y, s=26, marker=marker, facecolor=face, edgecolor=REGIME_COLORS[display], lw=0.8, zorder=3)
        ax.axvline(0, color=INK, lw=0.65)
        ax.axhspan(2.9, 7.5, color=BLUE, alpha=0.045, lw=0)
        ax.axhspan(-1.0, 2.9, color=PURPLE, alpha=0.045, lw=0)
        ax.set_yticks(y_positions)
        if col == 0:
            ax.set_yticklabels([item[3] for item in conditions])
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-1.0, 7.5)
        ax.set_xlabel("system-cost sequencing effect\n(EUR kW$^{-1}$ yr$^{-1}$)")
        ax.set_title(f"{REGIME_LABELS[display]}  (n={n_systems})", loc="left", color=REGIME_COLORS[display], fontweight="bold")
        quiet(ax, "x")
        panel_label(ax, chr(ord("a") + col), x=-0.18 if col == 0 else -0.10)

    ax_d = fig.add_subplot(grid[1, :])
    contrast_order = [
        ("energy_4x_minus_1x", "Energy 4× – 1×"),
        ("charge_power_4x_minus_1x", "Charge power 4× – 1×"),
        ("expanded_minus_observed_windows", "Expanded – observed hours"),
        ("energy_by_power_interaction", "Energy × power"),
        ("energy_by_window_interaction", "Energy × hours"),
        ("power_by_window_interaction", "Power × hours"),
        ("three_way_interaction", "Energy × power × hours"),
    ]
    y_base = np.arange(len(contrast_order))[::-1]
    offsets = {"Europe": 0.18, "China": 0.0, "Japan": -0.18}
    for display, key, _ in regimes:
        part = contrasts.loc[contrasts["regime"].eq(key)]
        for y, (contrast_key, _) in zip(y_base, contrast_order):
            row = part.loc[part["contrast"].eq(contrast_key)].iloc[0]
            mean = row["block_mean_contrast_eur_kw_year"]
            low = row["ci95_low"]
            high = row["ci95_high"]
            yp = y + offsets[display]
            ax_d.errorbar(mean, yp, xerr=[[mean - low], [high - mean]], fmt="none", color=REGIME_COLORS[display], lw=0.85, capsize=1.7)
            ax_d.scatter(mean, yp, s=24, marker=REGIME_MARKERS[display], color=REGIME_COLORS[display], edgecolor="white", lw=0.45, zorder=3, label=REGIME_LABELS[display] if y == y_base[0] else None)
    ax_d.axvline(0, color=INK, lw=0.70)
    ax_d.axhspan(3.5, 6.5, color=TEAL, alpha=0.045, lw=0)
    ax_d.set_yticks(y_base, [item[1] for item in contrast_order])
    ax_d.set_ylim(-0.6, 6.6)
    ax_d.set_xlabel("factorial contrast in system-cost sequencing effect (EUR kW$^{-1}$ yr$^{-1}$)")
    ax_d.legend(loc="lower left", ncol=3, columnspacing=1.0, handletextpad=0.4)
    quiet(ax_d, "x")
    panel_label(ax_d, "d", x=-0.04)
    save_bundle(fig, "fig_v62_supplementary_figure3_recharge_factorial")
    plt.close(fig)


def main() -> None:
    configure()
    cost = pd.read_csv(SOURCE / "discount_duration_cost_scale.csv")
    counts = pd.read_csv(ROOT / "robustness_evidence" / "cost_threshold_parity_counts.csv")
    duration = pd.read_csv(ROOT / "robustness_evidence" / "duration_boundary_summary.csv")
    cells = pd.read_csv(ROOT / "recharge_factorial" / "factorial_cell_summary.csv")
    contrasts = pd.read_csv(ROOT / "recharge_factorial" / "factorial_contrast_summary.csv")
    cost_uncertainty = pd.read_csv(
        ROOT / "cost_uncertainty_v62" / "results" / "cost_comparator_scenario_ensemble.csv"
    )
    natural_pairs = pd.read_csv(
        ROOT / "natural_year_heatmap" / "source_data" / "natural_year_transition_pairs.csv"
    )
    validate_inputs(cost, counts, duration, cells, contrasts, cost_uncertainty, natural_pairs)
    plot_discount_sensitivity(cost, counts, cost_uncertainty, natural_pairs)
    plot_duration_boundaries(duration)
    plot_factorial(cells, contrasts)


if __name__ == "__main__":
    main()
