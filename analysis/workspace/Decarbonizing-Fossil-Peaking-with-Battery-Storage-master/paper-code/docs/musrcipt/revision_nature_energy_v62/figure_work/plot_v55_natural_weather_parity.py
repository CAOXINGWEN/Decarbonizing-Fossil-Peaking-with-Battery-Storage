#!/usr/bin/env python3
"""Build v55 Figure 2: natural European weather-year sequencing parity."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
V55 = ROOT.parent
WEATHER = V55 / "weather_year_parity"
EUROPE = WEATHER / "results_europe"
COUNTRY = WEATHER / "results"
PREPARED = WEATHER / "prepared_europe"
OUT = ROOT / "main_figures"
SOURCE = ROOT / "source_data_v55"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 186.0

INK = "#23343F"
MUTED = "#6F7D84"
GRID = "#D9E0E3"
PALE = "#F3F6F7"
BLUE = "#2F7FAE"
TEAL = "#15988E"
CORAL = "#E56B2F"
PURPLE = "#7564A8"
GOLD = "#C99A28"
RED = "#C94B45"
GREEN = "#3D8F69"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.3,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "legend.fontsize": 5.7,
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
})


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="bottom", ha="left")


def quiet_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    rgba = plt.imread(stem.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(stem.with_name(stem.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def load_data() -> dict[str, pd.DataFrame]:
    values_2025 = pd.read_csv(EUROPE / "pecd_europe_34_window_relative_values_2025.csv")
    by_cost = pd.read_csv(EUROPE / "pecd_europe_34_window_relative_values_by_cost_year.csv")
    scope_risk = pd.read_csv(EUROPE / "pecd_europe_34_window_directional_risk.csv")
    country_risk = pd.read_csv(EUROPE / "pecd_country_35y_directional_risk_pooled.csv")
    country_span = pd.read_csv(COUNTRY / "pecd_35y_country_parity_summary.csv")
    pairs = pd.read_csv(COUNTRY / "pecd_35y_within_country_year_pairs.csv.gz")
    costs = pd.read_csv(EUROPE / "pecd_focal_cost_path_2025_2035.csv")
    return {
        "values_2025": values_2025,
        "by_cost": by_cost,
        "scope_risk": scope_risk,
        "country_risk": country_risk,
        "country_span": country_span,
        "pairs": pairs,
        "costs": costs,
    }


def make_daily_source(values_2025: pd.DataFrame) -> pd.DataFrame:
    data = np.load(PREPARED / "Europe33_copperplate_pecd_1982_2016_jul_jun_standardized.npz")
    starts = data["window_start_years"].astype(int)
    rows = []
    lookup = values_2025[values_2025.scope.eq("Europe33_copperplate")].set_index("window_start_year")
    for start in [1988, 1998]:
        idx = int(np.flatnonzero(starts == start)[0])
        task = data["base_task_mw"][idx].reshape(365, 24).sum(axis=1)
        recharge = data["recharge_mw"][idx].reshape(365, 24).sum(axis=1)
        event = data["event_day"][idx].astype(bool)
        task_scale = float(np.quantile(task, 0.99))
        recharge_scale = float(np.quantile(recharge, 0.99))
        for day in range(365):
            rows.append({
                "window_start_year": start,
                "weather_window": f"{start}/{str(start + 1)[-2:]}",
                "day_of_operating_year": day + 1,
                "daily_task_mwh": float(task[day]),
                "daily_recharge_mwh": float(recharge[day]),
                "normalized_task": float(task[day] / max(task_scale, 1.0e-12)),
                "normalized_recharge": float(recharge[day] / max(recharge_scale, 1.0e-12)),
                "event_day": bool(event[day]),
                "value_2025_eur_kw_year": float(lookup.loc[start, "value_2025_eur_kw_year"]),
                "zero_window_share": float(lookup.loc[start, "natural_zero_recovery_window_share"]),
                "longest_event_run_days": int(lookup.loc[start, "natural_longest_event_run_days"]),
            })
    return pd.DataFrame(rows)


def plot_panel_a(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    group = data["by_cost"]
    group = group[group.scope.eq("Europe33_copperplate")]
    pivot = group.pivot(index=["window_start_year", "weather_window"], columns="cost_year", values="caes24_minus_lfp4_eur_kw_year").reset_index()
    x = np.arange(len(pivot))
    ax.fill_between(x, pivot[2025], pivot[2035], color="#DCEDE9", alpha=0.85, linewidth=0)
    ax.plot(x, pivot[2025], color=INK, lw=0.85, marker="o", ms=2.1, mfc="white", mew=0.55, label="2025 cost")
    ax.plot(x, pivot[2035], color=TEAL, lw=1.15, marker="o", ms=2.2, mfc=TEAL, mew=0, label="2035 cost")
    ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(len(x) - 0.5, 0.45, "economic crossover", ha="right", va="bottom", color=MUTED, fontsize=5.5)

    for start, color in [(1988, GREEN), (1998, RED)]:
        idx = int(np.flatnonzero(pivot.window_start_year.to_numpy() == start)[0])
        ax.scatter(idx, pivot.loc[idx, 2025], s=24, color=color, edgecolor="white", linewidth=0.55, zorder=6)
        ax.scatter(idx, pivot.loc[idx, 2035], s=24, color=color, edgecolor="white", linewidth=0.55, zorder=6)
    i0 = int(np.flatnonzero(pivot.window_start_year.to_numpy() == 1988)[0])
    i1 = int(np.flatnonzero(pivot.window_start_year.to_numpy() == 1998)[0])
    ax.annotate(
        "natural decade\nchange = -7.60",
        xy=(i1, pivot.loc[i1, 2035]),
        xytext=(i0 + 1.4, pivot.loc[i0, 2025] + 3.5),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.8},
        color=RED,
        fontsize=5.6,
        ha="left",
        va="bottom",
    )
    ax.annotate(
        "cost gain = +5.30",
        xy=(len(x) - 2, pivot.loc[len(x) - 2, 2035]),
        xytext=(len(x) - 2, pivot.loc[len(x) - 2, 2025]),
        arrowprops={"arrowstyle": "<->", "color": PURPLE, "lw": 0.75},
        color=PURPLE,
        fontsize=5.5,
        ha="right",
        va="center",
    )
    ticks = np.arange(0, len(x), 5)
    ax.set_xticks(ticks, pivot.weather_window.iloc[ticks], rotation=0)
    ax.set_xlim(-0.8, len(x) - 0.2)
    ax.set_ylim(-27, 2)
    ax.set_ylabel("24 h A-CAES relative value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_xlabel("July-June operating year")
    quiet_grid(ax)
    ax.legend(loc="lower right", ncol=2, handlelength=1.4, columnspacing=0.8)
    panel_label(ax, "a", x=-0.075)
    pivot.to_csv(SOURCE / "figure2a_europe33_natural_chronology.csv", index=False)


def plot_panel_b(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    frame = data["by_cost"]
    frame = frame[frame.scope.eq("Europe33_copperplate")].copy()
    for pos, year in enumerate([2025, 2030, 2035]):
        values = frame.loc[frame.cost_year.eq(year), "caes24_minus_lfp4_eur_kw_year"].to_numpy(float)
        order = np.argsort(values, kind="mergesort")
        jitter = np.empty(len(values), dtype=float)
        jitter[order] = np.linspace(-0.13, 0.13, len(values))
        ax.scatter(np.full(len(values), pos) + jitter, values, s=8, color=mpl.colors.to_rgba(BLUE, 0.46), edgecolor="none", zorder=2)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        low, high = values.min(), values.max()
        ax.plot([pos, pos], [low, high], color=MUTED, lw=0.65, zorder=3)
        ax.add_patch(Rectangle((pos - 0.22, q1), 0.44, q3 - q1, facecolor="#E9EEF0", edgecolor=MUTED, linewidth=0.65, zorder=3))
        ax.plot([pos - 0.22, pos + 0.22], [med, med], color=INK, lw=1.05, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(2.35, 0.45, "0/34", color=RED, ha="right", va="bottom", fontweight="bold", fontsize=6.0)
    ax.text(2.35, -1.1, "crossovers at each cost year", color=MUTED, ha="right", va="top", fontsize=5.3)
    ax.set_xticks([0, 1, 2], ["2025", "2030", "2035"])
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-27, 2)
    ax.set_xlabel("technology cost year")
    quiet_grid(ax)
    panel_label(ax, "b", x=-0.18)
    frame.to_csv(SOURCE / "figure2b_cost_year_distributions.csv", index=False)


def plot_panel_c(ax_top: plt.Axes, ax_bottom: plt.Axes, daily: pd.DataFrame) -> None:
    for ax, start, color in [(ax_top, 1988, GREEN), (ax_bottom, 1998, RED)]:
        group = daily[daily.window_start_year.eq(start)]
        x = group.day_of_operating_year.to_numpy()
        task = group.normalized_task.to_numpy()
        recharge = group.normalized_recharge.to_numpy()
        event = group.event_day.to_numpy(bool)
        ax.fill_between(x, 0, task, color=mpl.colors.to_rgba(CORAL, 0.38), linewidth=0)
        ax.plot(x, task, color=CORAL, lw=0.42)
        ax.fill_between(x, 0, -recharge, color=mpl.colors.to_rgba(TEAL, 0.34), linewidth=0)
        ax.plot(x, -recharge, color=TEAL, lw=0.42)
        event_y = np.full(event.sum(), 1.10)
        ax.scatter(x[event], event_y, marker="|", s=8, color=INK, linewidth=0.55)
        label = str(group.weather_window.iloc[0])
        value = float(group.value_2025_eur_kw_year.iloc[0])
        ax.text(4, 1.16, f"{label}  |  value {value:.2f}", ha="left", va="bottom", color=color, fontweight="bold", fontsize=5.6)
        ax.text(362, 1.16, f"zero window {100*group.zero_window_share.iloc[0]:.1f}%  |  longest run {int(group.longest_event_run_days.iloc[0])} d", ha="right", va="bottom", color=MUTED, fontsize=5.2)
        ax.axhline(0, color=GRID, lw=0.5)
        ax.set_xlim(1, 365)
        ax.set_ylim(-1.18, 1.32)
        ax.set_yticks([-1, 0, 1], ["recharge", "0", "task"])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis="x", length=0)
    ax_bottom.set_xticks([1, 62, 123, 184, 245, 306, 365], ["Jul", "Sep", "Nov", "Jan", "Mar", "May", "Jun"])
    ax_bottom.set_xlabel("day of operating year")
    panel_label(ax_top, "c", x=-0.075, y=1.18)
    daily.to_csv(SOURCE / "figure2c_parity_pair_daily_sequences.csv", index=False)


def plot_panel_d(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    frame = data["values_2025"]
    frame = frame[frame.scope.eq("Europe33_copperplate")].sort_values("window_start_year")
    value = frame.value_2025_eur_kw_year.to_numpy(float)
    labels = frame.weather_window.astype(str).to_numpy()
    delta = value[10:] - value[:-10]
    x = np.arange(len(delta))
    colors = np.where(delta <= -5.296038099, RED, np.where(delta >= 5.296038099, GREEN, BLUE))
    ax.vlines(x, 0, delta, color=[mpl.colors.to_rgba(c, 0.65) for c in colors], lw=0.8)
    ax.scatter(x, delta, s=16, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.55)
    ax.axhline(-5.296038099, color=PURPLE, lw=0.8, ls=(0, (3, 2)))
    ax.text(len(delta) - 0.3, -5.05, "-5.30 cost-improvement line", ha="right", va="bottom", color=PURPLE, fontsize=5.3)
    hit = int(np.flatnonzero(delta <= -5.296038099)[0])
    ax.annotate(
        f"{labels[hit]} to {labels[hit + 10]}\n{delta[hit]:.2f}",
        xy=(hit, delta[hit]),
        xytext=(hit + 1.4, delta[hit] - 0.2),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.7},
        color=RED,
        fontsize=5.4,
        ha="left",
        va="top",
    )
    ax.set_xticks(np.arange(0, len(delta), 4), labels[:len(delta):4])
    ax.set_xlim(-0.7, len(delta) - 0.3)
    ax.set_ylim(-9.1, 8.2)
    ax.set_ylabel("weather-induced change\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_xlabel("ten-year weather transition (start window)")
    quiet_grid(ax)
    panel_label(ax, "d", x=-0.14)
    pd.DataFrame({
        "start_window": labels[:-10],
        "end_window": labels[10:],
        "weather_induced_value_change_eur_kw_year": delta,
        "erases_2025_2035_learning": delta <= -5.296038099,
    }).to_csv(SOURCE / "figure2d_ten_year_transitions.csv", index=False)


def plot_panel_e(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    countries = data["country_span"][[
        "country", "natural_full_span_eur_kw_year", "natural_q95_q05_span_eur_kw_year"
    ]].copy()
    europe = data["scope_risk"]
    europe = europe[
        europe.unit_id.eq("Europe33_copperplate")
        & europe.start_cost_year.eq(2025)
        & europe.end_cost_year.eq(2035)
    ].iloc[0]
    countries = pd.concat([
        pd.DataFrame([{
            "country": "Europe 33",
            "natural_full_span_eur_kw_year": europe.full_natural_span_eur_kw_year,
            "natural_q95_q05_span_eur_kw_year": europe.q95_q05_natural_span_eur_kw_year,
        }]),
        countries.sort_values("natural_full_span_eur_kw_year", ascending=False),
    ], ignore_index=True)
    y = np.arange(len(countries))[::-1]
    full = countries.natural_full_span_eur_kw_year.to_numpy(float)
    central = countries.natural_q95_q05_span_eur_kw_year.to_numpy(float)
    colors = [INK] + [BLUE] * (len(countries) - 1)
    ax.hlines(y, 0, full, color=[mpl.colors.to_rgba(c, 0.38) for c in colors], lw=1.0)
    ax.hlines(y, 0, central, color=colors, lw=3.0)
    ax.scatter(full, y, s=13, facecolor="white", edgecolor=colors, linewidth=0.7, zorder=3)
    ax.scatter(central, y, s=13, color=colors, edgecolor="white", linewidth=0.35, zorder=4)
    ax.axvline(5.296038099, color=PURPLE, lw=0.8, ls=(0, (3, 2)))
    ax.text(5.45, y.max() + 0.45, "5.30", color=PURPLE, fontsize=5.3, va="bottom")
    ax.set_yticks(y, countries.country)
    ax.set_xlim(0, max(13.5, full.max() + 0.7))
    ax.set_xlabel("natural value span (EUR kW$^{-1}$ yr$^{-1}$)")
    quiet_grid(ax, axis="x")
    legend = [
        Line2D([0], [0], color=BLUE, lw=3, marker="o", ms=3, label="central 90%"),
        Line2D([0], [0], color=mpl.colors.to_rgba(BLUE, 0.4), lw=1, marker="o", markerfacecolor="white", ms=3, label="full span"),
    ]
    ax.legend(handles=legend, loc="lower right", ncol=2, handlelength=1.3, columnspacing=0.7)
    panel_label(ax, "e", x=-0.12)
    countries.to_csv(SOURCE / "figure2e_country_and_europe_spans.csv", index=False)


def plot_panel_f(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    scope = data["scope_risk"]
    scope = scope[scope.unit_id.eq("Europe33_copperplate")].sort_values(["start_cost_year", "end_cost_year"])
    order = [(2025, 2030), (2030, 2035), (2025, 2035)]
    scope = scope.set_index(["start_cost_year", "end_cost_year"]).loc[order].reset_index()
    country = data["country_risk"].set_index(["start_cost_year", "end_cost_year"]).loc[order].reset_index()
    pairs = data["pairs"].absolute_value_difference_eur_kw_year.to_numpy(float)
    thresholds = scope.relative_cost_improvement_eur_kw_year.to_numpy(float)
    country_pair = np.array([np.mean(pairs >= threshold) for threshold in thresholds])
    x = np.arange(3)
    ax.plot(x, 100 * scope.all_pair_fraction_absolute_shift_reaching_cost_improvement, color=PURPLE, marker="o", ms=4, lw=1.15, label="Europe 33: any pair")
    ax.plot(x, 100 * scope.horizon_matched_adverse_fraction_erasing_learning, color=TEAL, marker="s", ms=3.8, lw=1.15, label="Europe 33: matched adverse")
    ax.plot(x, 100 * country_pair, color=mpl.colors.to_rgba(PURPLE, 0.48), marker="o", mfc="white", ms=3.8, lw=0.9, label="15 countries: any pair")
    ax.plot(x, 100 * country.horizon_matched_adverse_fraction_erasing_learning, color=CORAL, marker="s", ms=3.8, lw=1.15, label="15 countries: matched adverse")
    for xpos, value in enumerate(100 * scope.horizon_matched_adverse_fraction_erasing_learning):
        ax.text(xpos, value + 2.3, f"{value:.1f}%", ha="center", va="bottom", color=TEAL, fontsize=5.2)
    ax.set_xticks(x, ["2025-30\n5 yr", "2030-35\n5 yr", "2025-35\n10 yr"])
    ax.set_ylim(0, 76)
    ax.set_ylabel("sequences reaching parity (%)", labelpad=2)
    quiet_grid(ax)
    ax.legend(loc="upper right", ncol=2, handlelength=1.5, columnspacing=0.7)
    panel_label(ax, "f", x=-0.085)
    source = pd.DataFrame({
        "interval": ["2025-2030", "2030-2035", "2025-2035"],
        "threshold_eur_kw_year": thresholds,
        "europe33_any_pair_fraction": scope.all_pair_fraction_absolute_shift_reaching_cost_improvement.to_numpy(float),
        "europe33_horizon_matched_adverse_fraction": scope.horizon_matched_adverse_fraction_erasing_learning.to_numpy(float),
        "country15_any_pair_fraction": country_pair,
        "country15_horizon_matched_adverse_fraction": country.horizon_matched_adverse_fraction_erasing_learning.to_numpy(float),
    })
    source.to_csv(SOURCE / "figure2f_parity_exposure_rates.csv", index=False)


def plot_panel_g(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    frame = data["values_2025"].copy()
    styles = {
        "Europe33_copperplate": (TEAL, "Europe 33"),
        "Europe15_copperplate": (CORAL, "Europe 15"),
    }
    source_rows = []
    for scope, (color, label) in styles.items():
        group = frame[frame.scope.eq(scope)].copy()
        x = 100 * group.natural_zero_recovery_window_share.to_numpy(float)
        y = group.value_2025_eur_kw_year.to_numpy(float)
        ax.scatter(x, y, s=14, color=mpl.colors.to_rgba(color, 0.62), edgecolor="white", linewidth=0.35, label=label)
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, intercept + slope * xx, color=color, lw=1.0)
        group["zero_window_share_percentage_points"] = x
        group["ols_slope_eur_kw_year_per_percentage_point"] = slope
        source_rows.append(group)
    ax.set_xlabel("zero-recovery-window share (%)")
    ax.set_ylabel("")
    ax.text(
        0.015,
        0.92,
        "2025 relative value\n(EUR kW$^{-1}$ yr$^{-1}$)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=INK,
    )
    quiet_grid(ax)
    ax.legend(loc="upper right")
    panel_label(ax, "g", x=-0.085)
    pd.concat(source_rows, ignore_index=True).to_csv(SOURCE / "figure2g_recovery_geometry_relation.csv", index=False)


def build() -> None:
    data = load_data()
    daily = make_daily_source(data["values_2025"])

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        12,
        left=0.075,
        right=0.985,
        bottom=0.065,
        top=0.975,
        wspace=1.65,
        hspace=0.82,
        height_ratios=[1.18, 0.92, 1.30, 1.02],
    )
    ax_a = fig.add_subplot(gs[0, :8])
    ax_b = fig.add_subplot(gs[0, 8:])
    c_spec = gs[1, :].subgridspec(2, 1, hspace=0.06)
    ax_c1 = fig.add_subplot(c_spec[0, 0])
    ax_c2 = fig.add_subplot(c_spec[1, 0])
    ax_d = fig.add_subplot(gs[2, :6])
    ax_e = fig.add_subplot(gs[2, 6:])
    ax_f = fig.add_subplot(gs[3, :6])
    ax_g = fig.add_subplot(gs[3, 6:])

    plot_panel_a(ax_a, data)
    plot_panel_b(ax_b, data)
    plot_panel_c(ax_c1, ax_c2, daily)
    plot_panel_d(ax_d, data)
    plot_panel_e(ax_e, data)
    plot_panel_f(ax_f, data)
    plot_panel_g(ax_g, data)

    stem = OUT / "fig_v55_figure2_natural_weather_parity"
    save_bundle(fig, stem)
    plt.close(fig)


if __name__ == "__main__":
    build()
