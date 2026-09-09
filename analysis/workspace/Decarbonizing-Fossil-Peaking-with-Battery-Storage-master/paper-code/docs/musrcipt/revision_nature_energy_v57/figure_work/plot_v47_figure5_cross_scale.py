#!/usr/bin/env python3
"""Build v47 Figure 5 from audited European, Chinese and Japanese results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "main_figures"
SOURCE = HERE / "source_data_v47"
OUT.mkdir(exist_ok=True)
SOURCE.mkdir(exist_ok=True)

EUROPE = ROOT / "source_data" / "v43_figure5b_duration_envelope_bootstrap_10000.csv"
PRICE = ROOT / "price_channel" / "empirical_summary_v0_1" / "price_channel_caes24_summary.csv"
S3B = ROOT / "price_channel" / "empirical_reoptimized_summary_v0_1" / "s3b_caes24_summary.csv"
CROSS = ROOT / "cross_scale_cases" / "summary_v0_2"
EU_PHASE = ROOT / "source_data" / "v42_fixed_scale_mix_efc_cost_firm_phase_formation_rate.csv"

NAVY = "#17324D"
BLUE = "#2878B5"
ORANGE = "#E76F27"
TEAL = "#139A8C"
PURPLE = "#7761A8"
GOLD = "#D39B22"
GREY = "#738592"
LIGHT = "#EDF3F5"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.2,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold")


def clean(ax: mpl.axes.Axes, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#DCE5E8", linewidth=0.5, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


def save_source(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(SOURCE / name, index=False)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eu = pd.read_csv(EUROPE).sort_values("duration_h")
    price = pd.read_csv(PRICE)
    s3b = pd.read_csv(S3B)
    cross_duration = pd.read_csv(CROSS / "cross_scale_efc_duration_summary.csv")
    cross_valley = pd.read_csv(CROSS / "cross_scale_efc_valley_summary.csv")
    phase = pd.read_csv(CROSS / "cross_scale_efc_phase_scan.csv")
    eu_phase = pd.read_csv(EU_PHASE)
    eu_phase = eu_phase[
        eu_phase.source_label.eq("fixed_s20_central")
        & eu_phase.model_mode.eq("fixed")
        & eu_phase.chronology.eq("clustered")
        & np.isclose(eu_phase.ldes_tail_fixed_cost_multiplier, 1.0)
    ].copy()
    eu_phase = eu_phase.rename(
        columns={
            "firm_value_eur_kw_year": "firm_value_eur_kw_year",
            "country_years_with_strict_valley": "strict_valleys",
            "n_country_years": "n_case_years",
        }
    )
    eu_phase["application_case"] = "europe"
    eu_phase["strict_valley_pct"] = 100 * eu_phase.strict_valleys / eu_phase.n_case_years
    other = phase[
        phase.chronology.eq("clustered") & np.isclose(phase.ldes_cost_multiplier, 1.0)
    ].copy()
    other["strict_valley_pct"] = 100 * other.strict_valleys / other.n_case_years
    phase_all = pd.concat(
        [
            eu_phase[["application_case", "firm_value_eur_kw_year", "strict_valleys", "n_case_years", "strict_valley_pct"]],
            other[["application_case", "firm_value_eur_kw_year", "strict_valleys", "n_case_years", "strict_valley_pct"]],
        ],
        ignore_index=True,
    )
    price = pd.concat([price, s3b], ignore_index=True, sort=False)
    return eu, price, cross_duration, cross_valley, phase_all


def main() -> None:
    setup()
    eu, price, cross_duration, cross_valley, phase = load_data()
    save_source(eu, "figure5a_europe_duration_effect.csv")

    selected_scenarios = ["S0", "S1_beta_1", "S2_surplus_linked_charge", "S3a_empirical_price", "S3b_empirical_reoptimized"]
    price_plot = price[price.scenario.isin(selected_scenarios)].copy()
    label_map = {
        "S0": "fixed (69)",
        "S1_beta_1": "event uplift (69)",
        "S2_surplus_linked_charge": "surplus price (69)",
        "S3a_empirical_price": "empirical fixed (69)",
        "S3b_empirical_reoptimized": "empirical optimized (39)",
    }
    price_plot["label"] = price_plot.scenario.map(label_map)
    order = [label_map[x] for x in selected_scenarios]
    price_plot["label"] = pd.Categorical(price_plot.label, order, ordered=True)
    price_plot = price_plot.sort_values("label")
    save_source(price_plot, "figure5b_price_channel_effects.csv")

    cross_plot = cross_duration[
        cross_duration.scenario.eq("S0")
        & cross_duration.cost_year.eq(2035)
        & cross_duration.duration_h.ge(8)
    ].copy()
    save_source(cross_plot, "figure5c_cross_scale_duration_effects.csv")
    save_source(phase, "figure5d_strict_valley_visibility.csv")

    central_valleys = cross_valley[
        cross_valley.scenario.eq("S0")
        & (
            (cross_valley.application_case.eq("china") & cross_valley.cost_year.eq(2030))
            | (cross_valley.application_case.eq("japan") & cross_valley.cost_year.eq(2035))
        )
    ].copy()
    save_source(central_valleys, "figure5c_central_strict_valley_counts.csv")

    fig = plt.figure(figsize=(7.205, 6.35), facecolor="white")
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1.0, 1.08], hspace=0.42, wspace=0.95)
    axa = fig.add_subplot(gs[0, :7])
    axb = fig.add_subplot(gs[0, 8:])
    cgs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :7], hspace=0.24)
    axc1 = fig.add_subplot(cgs[0])
    axc2 = fig.add_subplot(cgs[1], sharex=axc1)
    axd = fig.add_subplot(gs[1, 8:])

    # a: European cross-duration response.
    axa.fill_between(eu.duration_h, eu.ci_low, eu.ci_high, color=BLUE, alpha=0.16, linewidth=0)
    axa.plot(eu.duration_h, eu.point_estimate, color=BLUE, lw=1.8, marker="o", ms=3.0, mec="white", mew=0.45, zorder=3)
    axa.axhline(0, color=NAVY, lw=0.75)
    axa.axvspan(28, 40, color=ORANGE, alpha=0.08, zorder=0)
    axa.annotate("strongest cycling loss", xy=(36, -7.69), xytext=(17, -10.0), color=ORANGE,
                 arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.7}, fontsize=5.7)
    axa.annotate("positive beyond 96 h", xy=(96, 1.27), xytext=(84, -1.4), ha="right", color=PURPLE,
                 arrowprops={"arrowstyle": "-", "color": PURPLE, "lw": 0.7}, fontsize=5.7)
    axa.set_xscale("log")
    axa.set_xlim(4, 180)
    axa.set_ylim(-11.2, 8.6)
    axa.set_xticks([4, 8, 16, 24, 40, 72, 96, 168], ["4", "8", "16", "24", "40", "72", "96", "168"])
    axa.set_ylabel("Chronology effect (EUR kW$^{-1}$ yr$^{-1}$)")
    axa.set_xlabel("Storage duration (h)")
    axa.text(0.01, 0.96, "Europe · 132 country-years", transform=axa.transAxes, va="top", color=NAVY, fontweight="bold")
    clean(axa)
    panel_label(axa, "a")

    # b: empirical and structured price-channel tests for 24 h A-CAES.
    y = np.arange(len(price_plot))[::-1]
    values = price_plot.mean_effect_eur_kw_year.to_numpy()
    lows = price_plot.ci95_low.to_numpy()
    highs = price_plot.ci95_high.to_numpy()
    colours = [GREY, ORANGE, TEAL, PURPLE, NAVY]
    axb.axvline(0, color=NAVY, lw=0.75)
    for yi, value, low, high, colour in zip(y, values, lows, highs, colours):
        axb.plot([low, high], [yi, yi], color=colour, lw=1.4, solid_capstyle="round")
        axb.scatter(value, yi, s=28, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
    axb.set_yticks(y, price_plot.label.astype(str), fontsize=5.5)
    axb.set_xlim(-10.8, 0.8)
    axb.set_xlabel("24 h A-CAES effect\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axb.text(0.02, 1.02, "7 priced markets · 4 reoptimized", transform=axb.transAxes, va="bottom", color=NAVY, fontweight="bold")
    axb.text(0.98, 0.42, r"critical event-run $\beta=19.6$", transform=axb.transAxes, ha="right", color=GREY, fontsize=5.4)
    clean(axb, grid="x")
    panel_label(axb, "b")

    # c: aligned application-scale responses without a dual y axis.
    app_specs = [
        (axc1, "china", ORANGE, "China · 26 province-years"),
        (axc2, "japan", TEAL, "Japan · 27 community-years in 5 prefectures"),
    ]
    for ax, app, colour, label in app_specs:
        sub = cross_plot[cross_plot.application_case.eq(app)].sort_values("duration_h")
        ax.fill_between(sub.duration_h, sub.ci95_low, sub.ci95_high, color=colour, alpha=0.15, linewidth=0)
        ax.plot(sub.duration_h, sub.mean_effect_eur_kw_year, color=colour, lw=1.55, marker="o", ms=2.4, mec="white", mew=0.35)
        ax.axhline(0, color=NAVY, lw=0.7)
        ax.set_xscale("log")
        ax.set_xlim(8, 760)
        ax.text(0.01, 0.90, label, transform=ax.transAxes, va="top", color=NAVY, fontweight="bold")
        clean(ax)
    axc1.set_ylim(-85, 28)
    axc2.set_ylim(-16, 18)
    axc1.tick_params(labelbottom=False)
    axc2.set_xlabel("Storage duration (h)")
    axc2.set_xticks([8, 16, 24, 40, 72, 168, 336, 720], ["8", "16", "24", "40", "72", "168", "336", "720"])
    axc1.text(0.97, 0.09, "2030 valleys  24/26 → 0/26", transform=axc1.transAxes,
              ha="right", va="bottom", color=ORANGE, fontsize=5.4, fontweight="bold")
    axc1.text(-0.13, 1.12, "c", transform=axc1.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold")
    fig.text(0.034, 0.235, "Chronology effect (EUR kW$^{-1}$ yr$^{-1}$)", rotation=90, ha="center", va="center", fontsize=6.5)

    # d: economic visibility of a strict valley under unchanged 2035 tail costs.
    specs = [("europe", BLUE, "Europe"), ("china", ORANGE, "China"), ("japan", TEAL, "Japan")]
    for app, colour, label in specs:
        sub = phase[phase.application_case.eq(app)].sort_values("firm_value_eur_kw_year")
        axd.plot(sub.firm_value_eur_kw_year, sub.strict_valley_pct, color=colour, lw=1.6,
                 marker={"europe": "o", "china": "s", "japan": "^"}[app], ms=2.8,
                 mec="white", mew=0.35, label=label)
    axd.axvline(103.799218, color=NAVY, ls="--", lw=0.8)
    axd.text(109, 96, "central firm value", rotation=90, va="top", color=NAVY, fontsize=5.4)
    axd.set_xlim(0, 610)
    axd.set_ylim(-3, 104)
    axd.set_xlabel("Firm-supply value (EUR kW$^{-1}$ yr$^{-1}$)")
    axd.set_ylabel("Strict valleys (%)", labelpad=1.0)
    axd.legend(loc="center left", bbox_to_anchor=(0.02, 0.57), frameon=False, handlelength=2.0)
    axd.text(0.02, 1.02, "clustered chronology · 2035 tail costs", transform=axd.transAxes, va="bottom", color=NAVY, fontweight="bold")
    clean(axd)
    panel_label(axd, "d")

    fig.subplots_adjust(left=0.088, right=0.985, top=0.965, bottom=0.092)
    stem = OUT / "fig_v47_figure5_price_and_cross_scale"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    with mpl.rc_context({"image.cmap": "gray"}):
        image = plt.imread(stem.with_suffix(".png"))
        rgb = image[..., :3]
        grey = np.dot(rgb, [0.2126, 0.7152, 0.0722])
        plt.imsave(OUT / "fig_v47_figure5_price_and_cross_scale_grayscale.png", grey, cmap="gray")

    manifest = {
        "figure": "Figure 5",
        "independent_blocks": {"europe": 15, "china": 26, "japan": 5},
        "observational_units": {"europe_country_years": 132, "china_province_years": 26, "japan_community_years": 27},
        "price_subset_country_years": sorted(price_plot.n_country_years.astype(int).unique().tolist()),
        "price_scenarios_main": selected_scenarios,
        "strict_valley_definition": "positive-negative-negative-positive duration envelope",
    }
    (SOURCE / "figure5_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
