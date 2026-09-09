#!/usr/bin/env python3
"""Nature-style main figures for the European observed-gas experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RESULTS = ROOT / "model_results" / "europe_observed_gas_mainline_v0_1"
DEFAULT_PANEL = ROOT / "processed" / "observed_gas_event_task_definitions_v0_2.csv.gz"
DEFAULT_OUTPUT = ROOT / "figures" / "europe_observed_gas_mainline_v0_1"

TECH_ORDER = [
    "li_lfp",
    "vrfb",
    "fecr_flow",
    "gravity",
    "molten_salt_tes",
    "caes",
    "laes",
    "h2_cavern",
]
TECH_LABELS = {
    "li_lfp": "LFP",
    "vrfb": "VRFB",
    "fecr_flow": "Fe–Cr flow",
    "gravity": "Gravity",
    "molten_salt_tes": "Thermal",
    "caes": "A-CAES",
    "laes": "LAES",
    "h2_cavern": "Cavern H$_2$",
}
TECH_COLORS = {
    "li_lfp": "#0072B2",
    "vrfb": "#56B4E9",
    "fecr_flow": "#CC79A7",
    "gravity": "#999999",
    "molten_salt_tes": "#E69F00",
    "caes": "#009E73",
    "laes": "#D55E00",
    "h2_cavern": "#6A51A3",
}
DISPERSED = "#4D4D4D"
CLUSTERED = "#009E73"
ACCENT = "#D55E00"


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", axes):
        position = ax.get_position()
        fig.text(
            position.x0 - 0.025,
            position.y1 + 0.008,
            label,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def export(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", bbox_inches="tight", dpi=600)
    fig.savefig(
        output / f"{stem}.tiff",
        bbox_inches="tight",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(output / f"{stem}.png") as image:
        image.convert("L").save(
            output / f"{stem}_grayscale.png",
            dpi=(300, 300),
        )


def make_orders(frame: pd.DataFrame) -> dict[str, list[int]]:
    daily = (
        frame.groupby("source_day_id", sort=True)
        .agg(
            segment=("season_segment", "first"),
            event=("event_flag_q75", "max"),
        )
        .reset_index()
    )

    def interleave(extreme: list[int], normal: list[int], cluster: bool) -> list[int]:
        groups = (
            [extreme[i : i + 4] for i in range(0, len(extreme), 4)]
            if cluster
            else [[item] for item in extreme]
        )
        slots: list[list[int]] = [[] for _ in range(len(normal) + 1)]
        if groups:
            positions = np.linspace(1, max(1, len(normal) - 1), len(groups))
            for group, position in zip(groups, np.rint(positions).astype(int)):
                slots[int(np.clip(position, 0, len(normal)))].extend(group)
        order: list[int] = []
        for index, day in enumerate(normal):
            order.extend(slots[index])
            order.append(day)
        order.extend(slots[-1])
        return order

    orders = {"natural": list(range(365))}
    for label, cluster in (("dispersed", False), ("clustered", True)):
        order: list[int] = []
        for segment in sorted(daily["segment"].unique()):
            subset = daily[daily["segment"].eq(segment)]
            extreme = subset.loc[
                subset["event"].astype(bool), "source_day_id"
            ].astype(int).tolist()
            normal = subset.loc[
                ~subset["event"].astype(bool), "source_day_id"
            ].astype(int).tolist()
            order.extend(interleave(extreme, normal, cluster))
        orders[label] = order
    return orders


def figure2(
    panel: pd.DataFrame,
    results: pd.DataFrame,
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 7.0))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[0.86, 1.05, 1.05], hspace=0.48, wspace=0.34
    )
    axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]
    axa, axb, axc, axd, axe, axf = axes

    # a | coverage matrix
    coverage = (
        panel[["country", "year"]]
        .drop_duplicates()
        .assign(value=1)
        .pivot(index="country", columns="year", values="value")
        .reindex(sorted(panel["country"].unique()))
    )
    axa.imshow(
        coverage.notna().to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=mpl.colors.ListedColormap(["#F2F2F2", "#4C9F91"]),
        vmin=0,
        vmax=1,
    )
    axa.set_xticks(range(len(coverage.columns)))
    axa.set_xticklabels(coverage.columns, rotation=45, ha="right")
    axa.set_yticks(range(len(coverage.index)))
    axa.set_yticklabels(coverage.index)
    axa.set_xlabel("Observed year")
    axa.set_ylabel("Country")
    axa.text(
        0.99,
        0.02,
        "15 countries | 132 country-years",
        transform=axa.transAxes,
        ha="right",
        va="bottom",
        color="#333333",
        fontsize=6,
    )
    for spine in axa.spines.values():
        spine.set_visible(False)

    # b | example conserved chronology
    example = panel[(panel["country"].eq("DE")) & (panel["year"].eq(2023))].copy()
    example = example.sort_values("source_hour_id", kind="stable")
    orders = make_orders(example)
    for offset, (label, color) in enumerate(
        (("dispersed", DISPERSED), ("clustered", CLUSTERED))
    ):
        task = (
            example["task_c3_q75_mw"]
            .to_numpy(float)
            .reshape(365, 24)[orders[label]]
            .sum(axis=1)
        )
        task = task / max(task.max(), 1.0)
        axb.fill_between(
            np.arange(110),
            offset * 1.3,
            offset * 1.3 + task[:110],
            color=color,
            alpha=0.82,
            linewidth=0,
        )
        axb.text(
            108,
            offset * 1.3 + 0.85,
            label.capitalize(),
            color=color,
            ha="right",
            va="center",
            fontweight="bold",
        )
    axb.set_xlim(0, 110)
    axb.set_ylim(-0.05, 2.45)
    axb.set_yticks([])
    axb.set_xlabel("Reordered day")
    axb.set_ylabel("Daily event task\n(normalized)")
    axb.spines["left"].set_visible(False)

    # c | physical duration response. Configurations with the same duration have
    # identical power and energy constraints in this experiment, so show the
    # common response once rather than hiding it beneath overlapping tech lines.
    duration_response = (
        summary.groupby("duration_h", as_index=False)
        .agg(
            retained=("retained_equivalent_full_cycles_mean", "mean"),
            low=("retained_equivalent_full_cycles_ci_low", "mean"),
            high=("retained_equivalent_full_cycles_ci_high", "mean"),
        )
        .sort_values("duration_h")
    )
    axc.plot(
        duration_response["duration_h"],
        100 * duration_response["retained"],
        color="#6A51A3",
        marker="o",
    )
    axc.fill_between(
        duration_response["duration_h"].to_numpy(float),
        100 * duration_response["low"].to_numpy(float),
        100 * duration_response["high"].to_numpy(float),
        color="#6A51A3",
        alpha=0.12,
        linewidth=0,
    )
    axc.axhline(100, color="#BDBDBD", lw=0.7, ls=":")
    axc.set_xscale("log")
    axc.set_xlim(3.5, 800)
    axc.set_ylim(20, 102)
    axc.set_xlabel("Nominal discharge duration (h)")
    axc.set_ylabel("Cycles retained after clustering (%)")
    axc.text(
        0.97,
        0.05,
        "53 feasible duration–technology configurations",
        transform=axc.transAxes,
        ha="right",
        va="bottom",
        color="#444444",
        fontsize=5.5,
    )

    # d | country-paired CAES-minus-LFP value
    key = results[
        results["config_id"].isin(["li_lfp_4h", "caes_24h"])
        & results["chronology"].isin(["dispersed", "clustered"])
    ]
    wide = key.pivot_table(
        index=["country", "year", "chronology"],
        columns="config_id",
        values="joint_net_benefit_cny_per_kw_year",
    ).reset_index()
    wide["relative"] = wide["caes_24h"] - wide["li_lfp_4h"]
    country = (
        wide.groupby(["country", "chronology"])["relative"]
        .mean()
        .unstack("chronology")
    )
    for _, row in country.iterrows():
        axd.plot(
            [0, 1],
            [row["dispersed"], row["clustered"]],
            color="#C7C7C7",
            lw=0.7,
            zorder=1,
        )
    axd.scatter(
        np.zeros(len(country)),
        country["dispersed"],
        color=DISPERSED,
        s=17,
        zorder=2,
        label="Dispersed",
    )
    axd.scatter(
        np.ones(len(country)),
        country["clustered"],
        color=CLUSTERED,
        s=17,
        zorder=2,
        label="Clustered",
    )
    axd.axhline(0, color="#777777", lw=0.7, ls="--")
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["Dispersed", "Clustered"])
    axd.set_xlim(-0.35, 1.35)
    axd.set_ylabel("CAES24 − LFP4 net benefit\n(CNY kW$^{-1}$ yr$^{-1}$)")
    axd.text(
        0.03,
        0.96,
        f"Mean: {country['dispersed'].mean():+.0f} → "
        f"{country['clustered'].mean():+.0f}",
        transform=axd.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # e | country-level relative value shift, years visible as a range
    shift = (
        wide.pivot_table(
            index=["country", "year"], columns="chronology", values="relative"
        )
        .assign(delta=lambda x: x["clustered"] - x["dispersed"])
        .reset_index()
    )
    country_shift = (
        shift.groupby("country")["delta"].agg(["mean", "min", "max"]).sort_values("mean")
    )
    y = np.arange(len(country_shift))
    axe.hlines(
        y,
        country_shift["min"],
        country_shift["max"],
        color="#BDBDBD",
        lw=1.0,
    )
    axe.scatter(country_shift["mean"], y, color=ACCENT, s=18, zorder=2)
    axe.axvline(0, color="#777777", lw=0.7, ls="--")
    axe.set_yticks(y)
    axe.set_yticklabels(country_shift.index)
    axe.set_xlabel("Clustered − dispersed relative value\n(CNY kW$^{-1}$ yr$^{-1}$)")
    axe.set_ylabel("Country")

    # f | replenishment interface response
    physical = results[
        results["chronology"].isin(["dispersed", "clustered"])
    ].pivot_table(
        index=["country", "year", "config_id", "tech_id", "duration_h"],
        columns="chronology",
        values=["charge_saturation_hours", "equivalent_full_cycles"],
    )
    physical.columns = ["_".join(item) for item in physical.columns]
    physical = physical.reset_index()
    physical["delta_charge_saturation_h"] = (
        physical["charge_saturation_hours_clustered"]
        - physical["charge_saturation_hours_dispersed"]
    )
    physical["delta_cycles"] = (
        physical["equivalent_full_cycles_clustered"]
        - physical["equivalent_full_cycles_dispersed"]
    )
    hb = axf.hexbin(
        physical["delta_charge_saturation_h"],
        physical["delta_cycles"],
        gridsize=32,
        mincnt=1,
        cmap="viridis",
        linewidths=0,
    )
    colorbar = fig.colorbar(hb, ax=axf, pad=0.02, fraction=0.05)
    colorbar.set_label("Country-year configurations")
    axf.axhline(0, color="#777777", lw=0.6, ls=":")
    axf.axvline(0, color="#777777", lw=0.6, ls=":")
    axf.set_xlabel("Change in charge-power saturation (h yr$^{-1}$)")
    axf.set_ylabel("Change in equivalent full cycles (yr$^{-1}$)")

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.07, top=0.98)
    panel_labels(fig, axes)
    export(fig, output, "fig_v38_europe_observed_gas_reuse")
    plt.close(fig)

    shift.to_csv(output / "fig_v38_source_data_relative_value_shift.csv", index=False)
    country_shift.reset_index().to_csv(
        output / "fig_v38_source_data_country_shift.csv", index=False
    )
    physical.to_csv(
        output / "fig_v38_source_data_replenishment_response.csv.gz",
        index=False,
    )


def figure3(
    results: pd.DataFrame,
    envelope: pd.DataFrame,
    gap: pd.DataFrame,
    output: Path,
) -> None:
    scenario = 31
    current = envelope[envelope["scenario_id"].eq(scenario)].copy()
    fig = plt.figure(figsize=(7.2, 7.1))
    grid = fig.add_gridspec(
        3, 3, height_ratios=[1.18, 0.92, 1.0], hspace=0.52, wspace=0.42
    )
    axa = fig.add_subplot(grid[0, :2])
    axb = fig.add_subplot(grid[0, 2])
    axc = fig.add_subplot(grid[1, 0])
    axd = fig.add_subplot(grid[1, 1])
    axe = fig.add_subplot(grid[1, 2])
    axf = fig.add_subplot(grid[2, :])
    axes = [axa, axb, axc, axd, axe, axf]

    # a | hero envelope
    for chronology, color, marker in (
        ("dispersed", DISPERSED, "o"),
        ("clustered", CLUSTERED, "s"),
    ):
        group = current[current["chronology"].eq(chronology)]
        statistics = group.groupby("duration_h")[
            "net_benefit_cny_per_kw_year"
        ].agg(
            mean="mean",
            low=lambda x: np.quantile(x, 0.1),
            high=lambda x: np.quantile(x, 0.9),
        )
        axa.plot(
            statistics.index,
            statistics["mean"],
            color=color,
            marker=marker,
            label=chronology.capitalize(),
        )
        axa.fill_between(
            statistics.index,
            statistics["low"],
            statistics["high"],
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    axa.axhline(0, color="#777777", lw=0.7, ls="--")
    axa.axvspan(28, 40, color=ACCENT, alpha=0.10, lw=0)
    axa.text(
        33.5,
        axa.get_ylim()[0] * 0.75,
        "mid-duration\ngap",
        color=ACCENT,
        ha="center",
        va="bottom",
        fontsize=6,
    )
    axa.set_xscale("log")
    axa.set_xlim(3.7, 180)
    axa.set_xlabel("Nominal discharge duration (h)")
    axa.set_ylabel("Joint net benefit (CNY kW$^{-1}$ yr$^{-1}$)")
    axa.legend(loc="upper left")

    # b | positive country-years
    counts = (
        current.assign(positive=current["net_benefit_cny_per_kw_year"] > 0)
        .groupby(["chronology", "duration_h"])["positive"]
        .sum()
        .reset_index()
    )
    for chronology, color, marker in (
        ("dispersed", DISPERSED, "o"),
        ("clustered", CLUSTERED, "s"),
    ):
        group = counts[counts["chronology"].eq(chronology)]
        axb.plot(
            group["duration_h"],
            group["positive"],
            color=color,
            marker=marker,
        )
    axb.set_xscale("log")
    axb.set_xlim(3.7, 180)
    axb.set_ylim(-3, 135)
    axb.set_yticks([0, 44, 88, 132])
    axb.set_xlabel("Duration (h)")
    axb.set_ylabel("Country-years with\npositive net benefit")

    # c | dominant technology along clustered envelope
    clustered = current[current["chronology"].eq("clustered")].copy()
    winners = (
        clustered.groupby("duration_h")["tech_id"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    net = clustered.groupby("duration_h")["net_benefit_cny_per_kw_year"].mean()
    axc.scatter(
        winners["duration_h"],
        net.reindex(winners["duration_h"]).to_numpy(),
        c=[TECH_COLORS[item] for item in winners["tech_id"]],
        s=24,
        edgecolor="white",
        linewidth=0.4,
    )
    axc.plot(winners["duration_h"], net.reindex(winners["duration_h"]), color="#A0A0A0", lw=0.7)
    axc.axhline(0, color="#777777", lw=0.6, ls="--")
    axc.set_xscale("log")
    axc.set_xlim(3.7, 180)
    axc.set_xlabel("Duration (h)")
    axc.set_ylabel("Clustered envelope\n(CNY kW$^{-1}$ yr$^{-1}$)")

    # d | country mean gap width
    scenario_rows = current.copy()
    records: list[dict[str, float | str | int]] = []
    for (country, year, chronology), group in scenario_rows.groupby(
        ["country", "year", "chronology"], sort=False
    ):
        ordered = group.sort_values("duration_h")
        durations = ordered["duration_h"].to_numpy(float)
        positive = ordered["net_benefit_cny_per_kw_year"].to_numpy(float) > 0
        widths: list[float] = []
        for position in range(1, len(positive) - 1):
            if positive[position]:
                continue
            left = np.flatnonzero(positive[:position])
            right = np.flatnonzero(positive[position + 1 :])
            if len(left) and len(right):
                widths.append(
                    durations[position + 1 + right[0]] - durations[left[-1]]
                )
        records.append(
            {
                "country": country,
                "year": year,
                "chronology": chronology,
                "gap_width_h": max(widths, default=0.0),
            }
        )
    gap_width = pd.DataFrame(records)
    plot_gap = gap_width[gap_width["chronology"].eq("clustered")]
    sns.boxplot(
        data=plot_gap,
        y="gap_width_h",
        color="#D9EEE8",
        width=0.42,
        fliersize=0,
        linewidth=0.7,
        ax=axd,
    )
    sns.stripplot(
        data=plot_gap,
        y="gap_width_h",
        color=CLUSTERED,
        alpha=0.25,
        size=2,
        jitter=0.18,
        ax=axd,
    )
    axd.set_xticks([])
    axd.set_xlabel("Clustered chronology\n(n = 132 country-years)")
    axd.set_ylabel("Strict gap width (h)")

    # e | gap prevalence phase matrix, clustered minus dispersed
    firm = 842.6943979480799
    phase = gap[np.isclose(gap["firm_value_cny_kw_year"], firm)].copy()
    phase["share"] = 100 * phase["country_years_with_gap"] / phase["country_years"]
    phase_wide = phase.pivot_table(
        index=["energy_value_cny_mwh", "charge_price_cny_mwh"],
        columns="chronology",
        values="share",
    ).reset_index()
    phase_wide["difference"] = phase_wide["clustered"] - phase_wide["dispersed"]
    matrix = phase_wide.pivot(
        index="energy_value_cny_mwh",
        columns="charge_price_cny_mwh",
        values="difference",
    ).sort_index(ascending=False)
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Change in gap prevalence (pp)", "shrink": 0.75},
        linewidths=0.5,
        linecolor="white",
        ax=axe,
    )
    axe.set_xticklabels([f"{float(item.get_text()):.0f}" for item in axe.get_xticklabels()])
    axe.set_yticklabels([f"{float(item.get_text()):.0f}" for item in axe.get_yticklabels()])
    axe.set_xlabel("Charging value\n(CNY MWh$^{-1}$)")
    axe.set_ylabel("Energy value\n(CNY MWh$^{-1}$)")

    # f | economic composition at clustered endpoint
    selected_ids = ["li_lfp_4h", "caes_24h", "h2_cavern_48h", "h2_cavern_168h"]
    selected = results[
        results["chronology"].eq("clustered")
        & results["config_id"].isin(selected_ids)
    ].copy()
    selected["energy_value"] = (
        selected["discharge_mwh"] * 800.0 / (selected["pdis_mw"] * 1000.0)
    )
    selected["firm_value"] = (
        selected["efc_ratio"] * 842.6943979480799
    )
    selected["cost"] = -(
        selected["annual_fixed_cost_cny_per_kw_year"]
        + selected["storage_vom_cny_per_kw_year"]
    )
    composition = (
        selected.groupby("config_id")[["energy_value", "firm_value", "cost"]]
        .mean()
        .reindex(selected_ids)
    )
    labels = ["LFP 4 h", "A-CAES 24 h", "Cavern H$_2$ 48 h", "Cavern H$_2$ 168 h"]
    x = np.arange(len(labels))
    axf.bar(
        x,
        composition["energy_value"],
        color="#56B4E9",
        label="Energy displacement",
    )
    axf.bar(
        x,
        composition["firm_value"],
        bottom=composition["energy_value"],
        color="#009E73",
        label="Firm-capacity value",
    )
    axf.bar(x, composition["cost"], color="#D55E00", label="Storage cost")
    net = composition.sum(axis=1)
    axf.scatter(x, net, color="black", marker="D", s=20, label="Net", zorder=3)
    axf.axhline(0, color="#777777", lw=0.7)
    axf.set_xticks(x)
    axf.set_xticklabels(labels)
    axf.set_ylabel("Value component (CNY kW$^{-1}$ yr$^{-1}$)")
    axf.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.07, top=0.97)
    panel_labels(fig, axes)
    export(fig, output, "fig_v38_conditional_mid_duration_valley")
    plt.close(fig)

    current.to_csv(output / "fig_v38_source_data_scenario31_envelope.csv.gz", index=False)
    counts.to_csv(output / "fig_v38_source_data_positive_country_years.csv", index=False)
    gap_width.to_csv(output / "fig_v38_source_data_gap_width.csv", index=False)
    composition.reset_index().to_csv(
        output / "fig_v38_source_data_value_composition.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    style()
    panel = pd.read_csv(
        args.panel,
        usecols=[
            "country",
            "year",
            "source_hour_id",
            "source_day_id",
            "season_segment",
            "event_flag_q75",
            "task_c3_q75_mw",
        ],
    )
    results = pd.read_csv(args.results / "configuration_results.csv.gz")
    paired = pd.read_csv(args.results / "paired_configuration_changes.csv.gz")
    summary = pd.read_csv(args.results / "paired_configuration_summary.csv")
    envelope = pd.read_csv(args.results / "value_sensitivity_envelope.csv.gz")
    gap = pd.read_csv(args.results / "value_sensitivity_gap_summary.csv")
    if np.any(summary["duration_h"].to_numpy(float) <= 0):
        raise ValueError("Log-duration figures require strictly positive durations.")
    args.output.mkdir(parents=True, exist_ok=True)
    figure2(panel, results, paired, summary, args.output)
    figure3(results, envelope, gap, args.output)


if __name__ == "__main__":
    main()
