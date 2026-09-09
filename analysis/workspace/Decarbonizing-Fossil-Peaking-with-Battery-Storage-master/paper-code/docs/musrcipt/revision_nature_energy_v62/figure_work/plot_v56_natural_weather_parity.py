#!/usr/bin/env python3
"""Build v56 Figure 2 from the 75-window PECD v4.2 analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
V56 = ROOT.parent
WEATHER = V56 / "weather_year_parity_v42"
RESULTS = WEATHER / "results"
PREPARED = WEATHER / "prepared" / "Europe33_pecd_v42_1950_2025_75_windows.npz"
OUT = ROOT / "main_figures"
SOURCE = ROOT / "source_data_v56"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 190.0

INK = "#20333D"
MUTED = "#66777E"
GRID = "#D7DFE2"
PALE = "#F2F5F5"
BLUE = "#2678A5"
TEAL = "#138A7E"
CORAL = "#D8643C"
PURPLE = "#6E63A5"
GOLD = "#C5972C"
RED = "#B94A48"
GREEN = "#378261"
YEAR_CMAP = LinearSegmentedColormap.from_list("weather_year", [BLUE, GOLD, CORAL])

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.4,
    "axes.labelsize": 6.6,
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
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.4, fontweight="bold", va="bottom", ha="left")


def quiet_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    rgba = plt.imread(stem.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(stem.with_name(stem.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def load_data() -> dict[str, object]:
    required = {
        "values": RESULTS / "pecd_v42_75_window_relative_values_2025.csv",
        "by_cost": RESULTS / "pecd_v42_75_window_relative_values_by_cost_year.csv",
        "decades": RESULTS / "pecd_v42_65_matched_decade_transitions.csv",
        "pairs": RESULTS / "pecd_v42_75_window_all_pairs.csv.gz",
        "costs": RESULTS / "pecd_v42_focal_cost_path_2025_2035.csv",
        "summary": RESULTS / "pecd_v42_75y_parity_summary.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing or not PREPARED.exists():
        raise FileNotFoundError("Run the PECD v4.2 build and parity pipeline first:\n" + "\n".join(missing))
    output: dict[str, object] = {
        name: pd.read_csv(path) for name, path in required.items() if name != "summary"
    }
    output["summary"] = json.loads(required["summary"].read_text(encoding="utf-8"))
    return output


def daily_contrast(values: pd.DataFrame, decades: pd.DataFrame) -> pd.DataFrame:
    data = np.load(PREPARED)
    starts = data["window_start_years"].astype(int)
    worst = decades.loc[decades.weather_value_change_eur_kw_year.idxmin()]
    selected = [int(worst.start_year), int(worst.end_year)]
    lookup = values.set_index("window_start_year")
    rows: list[dict[str, object]] = []
    for start in selected:
        index = int(np.flatnonzero(starts == start)[0])
        task = (data["base_task_mw"][index] + data["tail_task_mw"][index]).reshape(365, 24).sum(axis=1)
        recharge = data["recharge_mw"][index].reshape(365, 24).sum(axis=1)
        event = data["event_day"][index].astype(bool)
        task_scale = max(float(np.quantile(task, 0.99)), 1.0e-12)
        recharge_scale = max(float(np.quantile(recharge, 0.99)), 1.0e-12)
        meta = lookup.loc[start]
        for day in range(365):
            rows.append({
                "window_start_year": start,
                "weather_window": str(meta.weather_window),
                "day_of_operating_year": day + 1,
                "daily_task_mwh": float(task[day]),
                "daily_recharge_mwh": float(recharge[day]),
                "normalized_task": float(task[day] / task_scale),
                "normalized_recharge": float(recharge[day] / recharge_scale),
                "event_day": bool(event[day]),
                "value_2025_eur_kw_year": float(meta.value_2025_eur_kw_year),
                "zero_recovery_window_share": float(meta.natural_zero_recovery_window_share),
                "longest_event_run_days": int(meta.natural_longest_event_run_days),
            })
    return pd.DataFrame(rows)


def plot_a(ax: plt.Axes, values: pd.DataFrame, by_cost: pd.DataFrame, decades: pd.DataFrame, threshold: float) -> None:
    pivot = by_cost.pivot(index=["window_start_year", "weather_window"], columns="cost_year", values="caes24_minus_lfp4_eur_kw_year").reset_index()
    pivot = pivot.sort_values("window_start_year").reset_index(drop=True)
    x = np.arange(len(pivot))
    ax.fill_between(x, pivot[2025], pivot[2035], color="#DCEBE8", alpha=0.85, linewidth=0)
    ax.plot(x, pivot[2025], color=INK, lw=0.8, marker="o", ms=1.7, mfc="white", mew=0.45, label="2025 cost")
    ax.plot(x, pivot[2035], color=TEAL, lw=1.0, marker="o", ms=1.8, mfc=TEAL, mew=0, label="2035 cost")
    ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    worst = decades.loc[decades.weather_value_change_eur_kw_year.idxmin()]
    i0 = int(np.flatnonzero(pivot.window_start_year.to_numpy(int) == int(worst.start_year))[0])
    i1 = int(np.flatnonzero(pivot.window_start_year.to_numpy(int) == int(worst.end_year))[0])
    for index, color in [(i0, GREEN), (i1, RED)]:
        ax.axvline(index, color=color, lw=0.65, alpha=0.65)
        ax.scatter(index, pivot.loc[index, 2025], s=22, color=color, edgecolor="white", linewidth=0.5, zorder=6)
        ax.scatter(index, pivot.loc[index, 2035], s=22, color=color, edgecolor="white", linewidth=0.5, zorder=6)
    y_anchor = min(float(pivot[2025].max()), float(pivot.loc[i0, 2025] + 0.18 * np.ptp(pivot[2025])))
    ax.annotate(
        f"most adverse matched decade  {worst.weather_value_change_eur_kw_year:.2f}",
        xy=(i1, pivot.loc[i1, 2025]), xytext=(i0 + 1.0, y_anchor),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.75}, color=RED, fontsize=5.6,
        ha="left", va="bottom",
    )
    arrow_x = len(pivot) - 2.2
    base_y = float(pivot[2025].iloc[-2])
    ax.annotate("", xy=(arrow_x, base_y + threshold), xytext=(arrow_x, base_y), arrowprops={"arrowstyle": "<->", "color": PURPLE, "lw": 0.8})
    ax.text(arrow_x - 0.8, base_y + threshold / 2, f"learning +{threshold:.2f}", color=PURPLE, fontsize=5.4, ha="right", va="center")
    tick_index = np.arange(0, len(pivot), 10)
    ax.set_xticks(tick_index, pivot.weather_window.iloc[tick_index])
    ax.set_xlim(-1.0, len(pivot) - 0.1)
    pad = max(1.0, 0.10 * float(np.ptp(np.r_[pivot[2025], pivot[2035]])))
    ax.set_ylim(float(np.min(np.r_[pivot[2025], pivot[2035]]) - pad), float(np.max(np.r_[pivot[2025], pivot[2035]]) + pad))
    ax.set_ylabel("24 h A-CAES relative value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_xlabel("July-June operating year")
    quiet_grid(ax)
    ax.legend(loc="lower right", ncol=2, handlelength=1.5, columnspacing=0.9)
    panel_label(ax, "a", x=-0.055)
    pivot.to_csv(SOURCE / "figure2a_75_window_chronology.csv", index=False)


def plot_b(ax: plt.Axes, by_cost: pd.DataFrame) -> None:
    years = [2025, 2030, 2035]
    colours = [INK, BLUE, TEAL]
    arrays = [by_cost.loc[by_cost.cost_year.eq(year), "caes24_minus_lfp4_eur_kw_year"].to_numpy(float) for year in years]
    parts = ax.violinplot(arrays, positions=np.arange(3), widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, colour in zip(parts["bodies"], colours):
        body.set_facecolor(colour)
        body.set_edgecolor(colour)
        body.set_alpha(0.18)
        body.set_linewidth(0.65)
    rng = np.random.default_rng(5602)
    for pos, (values, colour) in enumerate(zip(arrays, colours)):
        jitter = rng.uniform(-0.19, 0.19, len(values))
        ax.scatter(pos + jitter, values, s=6.5, color=mpl.colors.to_rgba(colour, 0.48), edgecolor="none", zorder=2)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([pos, pos], [q1, q3], color=colour, lw=3.0, solid_capstyle="butt", zorder=3)
        ax.scatter(pos, median, s=17, facecolor="white", edgecolor=colour, linewidth=0.8, zorder=4)
        crossings = int(np.sum(values >= 0.0))
        ax.text(pos, float(values.max()) + 0.05 * max(np.ptp(values), 1.0), f"{crossings}/75", color=colour, ha="center", va="bottom", fontsize=5.6, fontweight="bold")
    ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(np.arange(3), [str(year) for year in years])
    ax.set_xlabel("technology cost year")
    ax.set_ylabel("relative value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    quiet_grid(ax)
    panel_label(ax, "b", x=-0.18)
    by_cost.to_csv(SOURCE / "figure2b_cost_year_distributions.csv", index=False)


def plot_c(axes: tuple[plt.Axes, plt.Axes], daily: pd.DataFrame) -> None:
    colours = [GREEN, RED]
    for index, ((_, group), ax, colour) in enumerate(zip(daily.groupby("window_start_year", sort=True), axes, colours)):
        day = group.day_of_operating_year.to_numpy(int)
        task = group.normalized_task.to_numpy(float)
        recharge = -group.normalized_recharge.to_numpy(float)
        event = group.event_day.to_numpy(bool)
        ax.fill_between(day, 0, task, color=mpl.colors.to_rgba(CORAL, 0.52), linewidth=0)
        ax.fill_between(day, 0, recharge, color=mpl.colors.to_rgba(BLUE, 0.48), linewidth=0)
        ax.vlines(day[event], 1.03, 1.13, color=INK, lw=0.35)
        meta = group.iloc[0]
        ax.text(3, 1.15, f"{meta.weather_window}  |  value {meta.value_2025_eur_kw_year:.2f}", color=colour, fontweight="bold", fontsize=5.6, ha="left", va="bottom")
        ax.text(363, 1.15, f"zero recovery {100 * meta.zero_recovery_window_share:.1f}%  |  longest {int(meta.longest_event_run_days)} d", color=MUTED, fontsize=5.2, ha="right", va="bottom")
        ax.axhline(0, color=MUTED, lw=0.45)
        ax.set_xlim(1, 365)
        ax.set_ylim(-1.08, 1.30)
        ax.set_yticks([-1, 0, 1], ["recharge", "0", "task"])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if index == 0:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            ax.set_xticks([1, 62, 123, 184, 245, 306, 365], ["Jul", "Sep", "Nov", "Jan", "Mar", "May", "Jun"])
            ax.set_xlabel("day of operating year")
    axes[0].legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor=mpl.colors.to_rgba(CORAL, 0.52), edgecolor="none", label="task"),
            Rectangle((0, 0), 1, 1, facecolor=mpl.colors.to_rgba(BLUE, 0.48), edgecolor="none", label="recharge"),
            Line2D([0], [0], marker="|", color=INK, linestyle="none", label="event day"),
        ],
        loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False, handlelength=1.1, columnspacing=0.8,
    )
    panel_label(axes[0], "c", x=-0.075, y=1.27)
    daily.to_csv(SOURCE / "figure2c_worst_decade_daily_sequences.csv", index=False)


def plot_d(ax: plt.Axes, decades: pd.DataFrame, threshold: float) -> None:
    frame = decades.sort_values("start_year").reset_index(drop=True)
    value = frame.weather_value_change_eur_kw_year.to_numpy(float)
    x = np.arange(len(frame))
    colours = np.where(value <= -threshold, RED, np.where(value >= threshold, GREEN, BLUE))
    ax.vlines(x, 0, value, color=[mpl.colors.to_rgba(c, 0.58) for c in colours], lw=0.75)
    ax.scatter(x, value, s=13, color=colours, edgecolor="white", linewidth=0.35, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.55)
    ax.axhline(-threshold, color=PURPLE, lw=0.8, ls=(0, (3, 2)))
    ax.axhline(threshold, color=PURPLE, lw=0.55, ls=(0, (1.5, 2.5)), alpha=0.8)
    adverse = int(frame.erases_2025_2035_learning.sum())
    beneficial = int(frame.exceeds_learning_beneficially.sum())
    ax.text(len(frame) - 0.5, -threshold, f"{adverse}/65 erase learning", color=RED, ha="right", va="top", fontsize=5.5, fontweight="bold")
    ax.text(len(frame) - 0.5, threshold, f"{beneficial}/65 reinforce learning", color=GREEN, ha="right", va="bottom", fontsize=5.5, fontweight="bold")
    worst = frame.loc[frame.weather_value_change_eur_kw_year.idxmin()]
    i = int(worst.name)
    ax.annotate(
        f"{worst.start_window} to {worst.end_window}\n{worst.weather_value_change_eur_kw_year:.2f}",
        xy=(i, worst.weather_value_change_eur_kw_year), xytext=(min(i + 4, 48), worst.weather_value_change_eur_kw_year),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.7}, color=RED, fontsize=5.4, ha="left", va="center",
    )
    ticks = np.arange(0, len(frame), 10)
    ax.set_xticks(ticks, frame.start_window.iloc[ticks])
    ax.set_xlim(-1, len(frame))
    pad = max(0.8, 0.12 * float(np.ptp(value)))
    ax.set_ylim(min(float(value.min() - pad), -threshold - pad), max(float(value.max() + pad), threshold + pad))
    ax.set_ylabel("weather-induced change\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_xlabel("matched ten-year transition (start window)")
    quiet_grid(ax)
    panel_label(ax, "d", x=-0.12)
    frame.to_csv(SOURCE / "figure2d_65_matched_decades.csv", index=False)


def plot_e(ax: plt.Axes, values: pd.DataFrame, threshold: float) -> None:
    frame = values.sort_values("window_start_year").reset_index(drop=True)
    value = frame.value_2025_eur_kw_year.to_numpy(float)
    matrix = np.abs(value[:, None] - value[None, :])
    mask = np.tril(np.ones_like(matrix, dtype=bool), 0)
    shown = np.ma.array(matrix, mask=mask)
    vmax = max(float(np.nanmax(matrix)), threshold)
    image = ax.imshow(shown, origin="upper", cmap="magma_r", vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
    hit_y, hit_x = np.where((matrix >= threshold) & np.triu(np.ones_like(matrix, dtype=bool), 1))
    ax.scatter(hit_x, hit_y, marker="s", s=2.0, facecolors="none", edgecolors=mpl.colors.to_rgba(INK, 0.34), linewidths=0.25)
    ticks = np.array([0, 14, 29, 44, 59, 74])
    labels = frame.weather_window.iloc[ticks].to_list()
    ax.set_xticks(ticks, labels, rotation=45, ha="right")
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("later operating year")
    ax.set_ylabel("earlier operating year")
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.044, pad=0.025)
    cbar.set_label("absolute value difference\n(EUR kW$^{-1}$ yr$^{-1}$)", fontsize=5.7)
    cbar.ax.tick_params(labelsize=5.2, width=0.5, length=2.3)
    cbar.ax.axhline(threshold, color="white", lw=1.0)
    panel_label(ax, "e", x=-0.12)
    pd.DataFrame(matrix, index=frame.weather_window, columns=frame.weather_window).to_csv(SOURCE / "figure2e_all_pair_matrix.csv")


def plot_f(ax: plt.Axes, summary: dict[str, object]) -> None:
    threshold = float(summary["cost_improvement_2025_2035_eur_kw_year"])
    spans = np.array([
        float(summary["natural_full_span_eur_kw_year"]),
        float(summary["natural_q95_q05_span_eur_kw_year"]),
    ])
    y = np.array([1.0, 0.0])
    ax.barh(y, spans, height=0.46, color=[INK, TEAL], alpha=0.88)
    ax.axvline(threshold, color=PURPLE, lw=0.9, ls=(0, (3, 2)))
    ax.text(threshold, 1.42, f"cost scale {threshold:.2f}", color=PURPLE, fontsize=5.4, ha="center", va="bottom")
    for yi, value in zip(y, spans):
        ax.text(value + 0.12, yi, f"{value:.2f}", ha="left", va="center", color=INK, fontsize=5.7, fontweight="bold")
    ax.set_yticks(y, ["full span", "5-95% span"])
    ax.set_xlabel("natural value span (EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_xlim(0, max(float(spans.max()), threshold) * 1.30)
    ax.set_ylim(-0.82, 1.58)
    quiet_grid(ax, axis="x")
    pair_n = int(summary["all_pairs_reaching_cost_scale"])
    pair_total = int(summary["all_unordered_weather_pairs"])
    decade_n = int(summary["matched_decades_erasing_learning"])
    decade_total = int(summary["matched_ten_year_transitions"])
    ax.text(0.03, 1.16, f"all pairs  {pair_n}/{pair_total} ({100 * pair_n / pair_total:.1f}%)", transform=ax.transAxes, ha="left", va="bottom", color=CORAL, fontsize=5.5, fontweight="bold", clip_on=False)
    ax.text(0.03, 1.06, f"adverse decades  {decade_n}/{decade_total} ({100 * decade_n / decade_total:.1f}%)", transform=ax.transAxes, ha="left", va="bottom", color=RED, fontsize=5.5, fontweight="bold", clip_on=False)
    panel_label(ax, "f", x=-0.16)
    pd.DataFrame({
        "metric": ["full_span", "q95_q05_span", "cost_improvement", "all_pair_fraction", "adverse_decade_fraction"],
        "value": [spans[0], spans[1], threshold, pair_n / pair_total, decade_n / decade_total],
    }).to_csv(SOURCE / "figure2f_risk_accounting.csv", index=False)


def plot_g(ax: plt.Axes, values: pd.DataFrame) -> None:
    frame = values.sort_values("window_start_year").copy()
    x = 100.0 * frame.natural_zero_recovery_window_share.to_numpy(float)
    y = frame.value_2025_eur_kw_year.to_numpy(float)
    years = frame.window_start_year.to_numpy(int)
    norm = Normalize(vmin=int(years.min()), vmax=int(years.max()))
    ax.scatter(x, y, c=years, cmap=YEAR_CMAP, norm=norm, s=22, edgecolor="white", linewidth=0.45, alpha=0.9, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(xx, intercept + slope * xx, color=INK, lw=1.0)
    r = float(np.corrcoef(x, y)[0, 1])
    ax.text(0.98, 0.95, f"Pearson r = {r:.2f}", transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=5.7)
    ax.set_xlabel("zero-recovery-window share (%)")
    ax.set_ylabel("2025 relative value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    quiet_grid(ax)
    anchors = [1950, 1987, 2024]
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markersize=4.0,
                   markerfacecolor=YEAR_CMAP(norm(year)), markeredgecolor="white",
                   markeredgewidth=0.4, label=str(year))
            for year in anchors
        ],
        title="operating-year start", title_fontsize=5.5, loc="lower right",
        frameon=False, ncol=3, handletextpad=0.25, columnspacing=0.55,
    )
    panel_label(ax, "g", x=-0.13)
    frame[["window_start_year", "weather_window", "natural_zero_recovery_window_share", "value_2025_eur_kw_year"]].to_csv(SOURCE / "figure2g_recovery_value_relation.csv", index=False)


def main() -> None:
    data = load_data()
    values = data["values"]
    by_cost = data["by_cost"]
    decades = data["decades"]
    summary = data["summary"]
    assert isinstance(values, pd.DataFrame)
    assert isinstance(by_cost, pd.DataFrame)
    assert isinstance(decades, pd.DataFrame)
    assert isinstance(summary, dict)
    threshold = float(summary["cost_improvement_2025_2035_eur_kw_year"])
    daily = daily_contrast(values, decades)

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    outer = fig.add_gridspec(
        4, 12, left=0.105, right=0.985, bottom=0.075, top=0.965,
        hspace=0.72, wspace=1.25, height_ratios=[0.85, 0.92, 1.18, 0.92],
    )
    ax_a = fig.add_subplot(outer[0, :])
    ax_b = fig.add_subplot(outer[1, :4])
    cgrid = outer[1, 4:].subgridspec(2, 1, hspace=0.08)
    ax_c1 = fig.add_subplot(cgrid[0, 0])
    ax_c2 = fig.add_subplot(cgrid[1, 0], sharex=ax_c1)
    ax_d = fig.add_subplot(outer[2, :6])
    ax_e = fig.add_subplot(outer[2, 7:])
    ax_f = fig.add_subplot(outer[3, :5])
    ax_g = fig.add_subplot(outer[3, 6:])

    plot_a(ax_a, values, by_cost, decades, threshold)
    plot_b(ax_b, by_cost)
    plot_c((ax_c1, ax_c2), daily)
    plot_d(ax_d, decades, threshold)
    plot_e(ax_e, values, threshold)
    plot_f(ax_f, summary)
    plot_g(ax_g, values)

    stem = OUT / "fig_v56_figure2_natural_weather_parity"
    save_bundle(fig, stem)
    plt.close(fig)
    (SOURCE / "figure2_source_data_manifest.txt").write_text(
        "Figure 2 source tables are generated directly by plot_v56_natural_weather_parity.py\n"
        "from weather_year_parity_v42/results and the standardized prepared NPZ.\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
