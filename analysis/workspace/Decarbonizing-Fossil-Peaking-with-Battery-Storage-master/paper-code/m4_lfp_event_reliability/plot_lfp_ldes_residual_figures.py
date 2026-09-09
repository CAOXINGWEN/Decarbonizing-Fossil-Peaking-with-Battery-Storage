#!/usr/bin/env python3
"""Publication figures for LFP-only vs baseline LFP + LDES residual demand."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm


MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_DIR / "output"
FIGURE_DIR = MODULE_DIR / "figures"
SOURCE_DIR = FIGURE_DIR / "source_data"

PROVINCES = ["Beijing", "Jiangsu", "Fujian", "Zhejiang", "Sichuan"]
WINDOWS = [8, 12, 16, 24]

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "teal": "#42949E",
    "green": "#2E8B57",
    "red": "#B64342",
    "orange": "#D8842F",
    "violet": "#7C6CCF",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#8A8A8A",
    "neutral_dark": "#4D4D4D",
    "black": "#272727",
}

PROVINCE_COLORS = {
    "Beijing": "#0F4D92",
    "Jiangsu": "#42949E",
    "Fujian": "#D8842F",
    "Zhejiang": "#7C6CCF",
    "Sichuan": "#2E8B57",
}

TECH_COLORS = {
    "caes": "#16456A",
    "h2_cavern": "#7B68B3",
    "laes": "#C77C2E",
    "gravity": "#3E8B72",
    "fecr_flow": "#4E8B95",
    "molten_salt_tes": "#8A8A8A",
    "vrfb": "#A05FB5",
}

TECH_LABELS = {
    "caes": "CAES",
    "h2_cavern": "H2 cavern",
    "laes": "LAES",
    "gravity": "Gravity",
    "fecr_flow": "FeCr flow",
    "molten_salt_tes": "Molten salt TES",
    "vrfb": "VRFB",
}


def apply_publication_style(font_size: int = 8, axes_linewidth: float = 1.0) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["black"],
    )


def save_figure(fig, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf", "png"]:
        path = FIGURE_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
    plt.close(fig)


def ordered_osr(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["boundary_metric"].eq("osr")].copy()
    out["province_en"] = pd.Categorical(out["province_en"], PROVINCES, ordered=True)
    out["peaking_window_h"] = pd.Categorical(out["peaking_window_h"], WINDOWS, ordered=True)
    return out.sort_values(["province_en", "peaking_window_h"]).reset_index(drop=True)


def matrix(df: pd.DataFrame, value: str) -> np.ndarray:
    pivot = (
        df.pivot(index="province_en", columns="peaking_window_h", values=value)
        .reindex(index=PROVINCES, columns=WINDOWS)
    )
    return pivot.to_numpy(dtype=float)


def annotate_heatmap(ax, data: np.ndarray, fmt: str, color_threshold: float | None = None) -> None:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if not np.isfinite(value):
                label = "NA"
            elif fmt == "ratio":
                label = f"{value:.2f}"
            elif fmt == "pct":
                label = f"{value:.0f}%"
            elif fmt == "gwh":
                label = f"{value:.0f}"
            else:
                label = f"{value:.1f}"
            text_color = "white" if color_threshold is not None and value > color_threshold else PALETTE["black"]
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color=text_color)


def style_matrix_axes(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_xticks(np.arange(len(WINDOWS)))
    ax.set_xticklabels([f"{w} h" for w in WINDOWS])
    ax.set_yticks(np.arange(len(PROVINCES)))
    ax.set_yticklabels(PROVINCES)
    ax.set_xlabel("Peaking window")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9, pad=8)
    ax.tick_params(length=0)


def plot_residual_summary(summary: pd.DataFrame) -> None:
    osr = ordered_osr(summary)
    osr_source = osr.copy()
    osr_source["total_lfp_only_unserved_GWh"] = (
        osr_source["total_lfp_only_unserved_extreme_demand_kwh"] / 1e6
    )
    osr_source["capacity_ratio_LDES_to_LFP_overbuild"] = (
        osr_source["mean_required_LDES_MWh"] / osr_source["mean_required_LFP_overbuild_MWh"]
    )
    osr_source.to_csv(SOURCE_DIR / "fig_lfp_ldes_residual_summary_osr.csv", index=False)

    completion = (
        osr.groupby("province_en", observed=False)[
            ["lfp_only_event_completion_rate", "lfp_ldes_event_completion_rate"]
        ]
        .mean()
        .reindex(PROVINCES)
    )
    unserved = matrix(osr_source, "total_lfp_only_unserved_GWh")
    cap_ratio = matrix(osr_source, "capacity_ratio_LDES_to_LFP_overbuild")
    cost_ratio = matrix(osr_source, "mean_ldes_to_lfp_overbuild_cost_ratio")

    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    x = np.arange(len(PROVINCES))
    width = 0.34
    ax_a.bar(
        x - width / 2,
        completion["lfp_only_event_completion_rate"],
        width,
        label="LFP-only",
        color=PALETTE["neutral_mid"],
    )
    ax_a.bar(
        x + width / 2,
        completion["lfp_ldes_event_completion_rate"],
        width,
        label="LFP + LDES",
        color=PALETTE["blue_main"],
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(PROVINCES, rotation=30, ha="right")
    ax_a.set_ylabel("Event completion rate (%)")
    ax_a.set_ylim(0, 110)
    ax_a.set_title("Reliability target", loc="left", fontsize=9, pad=8)
    ax_a.legend(loc="upper left", ncols=1)
    ax_a.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    add_panel_label(ax_a, "a")

    blue_cmap = LinearSegmentedColormap.from_list(
        "blue_scale", ["#F4F7FB", "#9CC3E6", "#0F4D92"]
    )
    im_b = ax_b.imshow(unserved, cmap=blue_cmap, aspect="auto")
    style_matrix_axes(ax_b, "LFP-only unserved demand", ylabel=None)
    annotate_heatmap(ax_b, unserved, "gwh", color_threshold=np.nanmax(unserved) * 0.55)
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.03)
    cbar_b.set_label("GWh")
    add_panel_label(ax_b, "b")

    ratio_cmap = LinearSegmentedColormap.from_list(
        "capacity_ratio", ["#F7FBF7", "#AADCA9", "#2E8B57"]
    )
    im_c = ax_c.imshow(cap_ratio, cmap=ratio_cmap, aspect="auto")
    style_matrix_axes(ax_c, "LDES energy / LFP overbuild", ylabel=None)
    annotate_heatmap(ax_c, cap_ratio, "ratio", color_threshold=np.nanmax(cap_ratio) * 0.92)
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.03)
    cbar_c.set_label("MWh ratio")
    add_panel_label(ax_c, "c")

    vmax = max(2.1, float(np.nanmax(cost_ratio)))
    norm = TwoSlopeNorm(vmin=0.75, vcenter=1.0, vmax=vmax)
    cost_cmap = LinearSegmentedColormap.from_list(
        "cost_ratio", ["#2E8B57", "#F5F5F5", "#B64342"]
    )
    im_d = ax_d.imshow(cost_ratio, cmap=cost_cmap, norm=norm, aspect="auto")
    style_matrix_axes(ax_d, "LDES cost / LFP overbuild cost", ylabel=None)
    annotate_heatmap(ax_d, cost_ratio, "ratio", color_threshold=None)
    cbar_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.03)
    cbar_d.set_label("Ratio")
    add_panel_label(ax_d, "d")

    save_figure(fig, "fig_m4_lfp_ldes_residual_summary")


def plot_event_cost_scatter(best_event: pd.DataFrame) -> None:
    osr = ordered_osr(best_event)
    osr = osr[
        [
            "event_id",
            "province_en",
            "peaking_window_h",
            "lfp_only_overbuild_annual_cost_cny",
            "ldes_annual_cost_cny",
            "ldes_tech_id",
            "ldes_duration_h",
            "incremental_cost_saving_pct",
        ]
    ].copy()
    osr["lfp_only_overbuild_annual_cost_billion_cny"] = (
        osr["lfp_only_overbuild_annual_cost_cny"] / 1e9
    )
    osr["ldes_annual_cost_billion_cny"] = osr["ldes_annual_cost_cny"] / 1e9
    osr.to_csv(SOURCE_DIR / "fig_event_cost_scatter_osr.csv", index=False)

    marker_map = {8: "o", 12: "s", 16: "^", 24: "D"}
    fig, ax = plt.subplots(figsize=(4.1, 3.7), constrained_layout=True)
    for province in PROVINCES:
        sub = osr[osr["province_en"].eq(province)]
        for window, marker in marker_map.items():
            part = sub[sub["peaking_window_h"].astype(int).eq(window)]
            if part.empty:
                continue
            ax.scatter(
                part["lfp_only_overbuild_annual_cost_billion_cny"],
                part["ldes_annual_cost_billion_cny"],
                s=32,
                marker=marker,
                color=PROVINCE_COLORS[province],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
            )

    min_value = min(
        osr["lfp_only_overbuild_annual_cost_billion_cny"].min(),
        osr["ldes_annual_cost_billion_cny"].min(),
    )
    max_value = max(
        osr["lfp_only_overbuild_annual_cost_billion_cny"].max(),
        osr["ldes_annual_cost_billion_cny"].max(),
    )
    line = np.logspace(np.log10(min_value * 0.8), np.log10(max_value * 1.2), 100)
    ax.plot(line, line, color=PALETTE["neutral_dark"], linewidth=1.0, linestyle="--")
    ax.fill_between(line, min_value * 0.5, line, color="#DDF3DE", alpha=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("LFP-only overbuild annual cost (billion CNY yr$^{-1}$)")
    ax.set_ylabel("Best LDES annual cost (billion CNY yr$^{-1}$)")
    ax.set_title("Event-level marginal reliability cost", loc="left", fontsize=9, pad=8)
    ax.grid(True, which="major", color="#E6E6E6", linewidth=0.6)
    add_panel_label(ax, "a")

    province_handles = [
        ax.scatter([], [], s=32, color=PROVINCE_COLORS[p], label=p) for p in PROVINCES
    ]
    marker_handles = [
        ax.scatter([], [], s=32, color=PALETTE["neutral_dark"], marker=m, label=f"{w} h")
        for w, m in marker_map.items()
    ]
    leg1 = ax.legend(handles=province_handles, loc="lower right", fontsize=7, title="Province")
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="upper left", fontsize=7, title="Window")

    save_figure(fig, "fig_m4_event_cost_scatter")


def plot_best_tech_map(counts: pd.DataFrame, summary: pd.DataFrame) -> None:
    osr_counts = ordered_osr(counts)
    event_counts = ordered_osr(summary)[
        ["province_en", "peaking_window_h", "event_count"]
    ].drop_duplicates()
    dominant = (
        osr_counts.sort_values(
            ["province_en", "peaking_window_h", "best_event_count"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["province_en", "peaking_window_h"], keep="first")
        .merge(event_counts, on=["province_en", "peaking_window_h"], how="left")
    )
    dominant["dominant_label"] = dominant.apply(
        lambda r: (
            f"{TECH_LABELS.get(r['ldes_tech_id'], r['ldes_tech_id'])}\n"
            f"{int(r['ldes_duration_h'])} h"
        ),
        axis=1,
    )
    dominant["share_label"] = dominant.apply(
        lambda r: f"{int(r['best_event_count'])}/{int(r['event_count'])}", axis=1
    )
    dominant.to_csv(SOURCE_DIR / "fig_best_ldes_tech_map_osr.csv", index=False)

    tech_order = [tech for tech in TECH_COLORS if tech in set(dominant["ldes_tech_id"])]
    tech_to_idx = {tech: idx for idx, tech in enumerate(tech_order)}
    color_grid = np.full((len(PROVINCES), len(WINDOWS)), np.nan)
    label_grid = [["" for _ in WINDOWS] for _ in PROVINCES]
    share_grid = [["" for _ in WINDOWS] for _ in PROVINCES]
    for _, row in dominant.iterrows():
        i = PROVINCES.index(row["province_en"])
        j = WINDOWS.index(int(row["peaking_window_h"]))
        color_grid[i, j] = tech_to_idx[row["ldes_tech_id"]]
        label_grid[i][j] = row["dominant_label"]
        share_grid[i][j] = row["share_label"]

    cmap = ListedColormap([TECH_COLORS[tech] for tech in tech_order])
    fig, ax = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
    ax.imshow(color_grid, cmap=cmap, vmin=-0.5, vmax=len(tech_order) - 0.5, aspect="auto")
    style_matrix_axes(ax, "Dominant least-cost LDES option", ylabel=None)
    for i in range(len(PROVINCES)):
        for j in range(len(WINDOWS)):
            ax.text(
                j,
                i - 0.08,
                label_grid[i][j],
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                fontweight="bold",
            )
            ax.text(
                j,
                i + 0.26,
                share_grid[i][j],
                ha="center",
                va="center",
                color="white",
                fontsize=6,
            )
    add_panel_label(ax, "a")

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, color=TECH_COLORS[tech])
        for tech in tech_order
    ]
    labels = [TECH_LABELS.get(tech, tech) for tech in tech_order]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=3, fontsize=7)
    save_figure(fig, "fig_m4_best_ldes_tech_map")


def plot_cost_saving_lines(summary: pd.DataFrame) -> None:
    osr = ordered_osr(summary)
    source = osr[
        [
            "province_en",
            "peaking_window_h",
            "mean_incremental_cost_saving_pct",
            "mean_ldes_to_lfp_overbuild_cost_ratio",
        ]
    ].copy()
    source.to_csv(SOURCE_DIR / "fig_cost_saving_by_window_osr.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    for province in PROVINCES:
        sub = source[source["province_en"].eq(province)]
        ax.plot(
            sub["peaking_window_h"].astype(int),
            sub["mean_incremental_cost_saving_pct"],
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            color=PROVINCE_COLORS[province],
            label=province,
        )
    ax.axhline(0, color=PALETTE["neutral_dark"], linewidth=1.0, linestyle="--")
    ax.set_xticks(WINDOWS)
    ax.set_xlabel("Peaking window (h)")
    ax.set_ylabel("Cost saving vs LFP overbuild (%)")
    ax.set_title("LDES advantage strengthens in longer windows", loc="left", fontsize=9, pad=8)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.legend(loc="lower right", ncols=1, fontsize=7)
    add_panel_label(ax, "a")
    save_figure(fig, "fig_m4_cost_saving_by_window")


def main() -> None:
    apply_publication_style()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(OUTPUT_DIR / "lfp_only_vs_lfp_ldes_best_summary.csv")
    best_event = pd.read_csv(OUTPUT_DIR / "lfp_only_vs_lfp_ldes_best_event.csv")
    counts = pd.read_csv(OUTPUT_DIR / "lfp_ldes_best_tech_counts.csv")

    plot_residual_summary(summary)
    plot_event_cost_scatter(best_event)
    plot_best_tech_map(counts, summary)
    plot_cost_saving_lines(summary)

    print("Wrote figures to:")
    for path in sorted(FIGURE_DIR.glob("fig_m4_*.*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
