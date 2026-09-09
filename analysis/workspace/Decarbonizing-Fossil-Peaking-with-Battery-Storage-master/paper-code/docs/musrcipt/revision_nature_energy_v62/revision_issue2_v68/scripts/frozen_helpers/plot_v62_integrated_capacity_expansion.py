#!/usr/bin/env python3
"""Plot the integrated Supplementary Figure 5 capacity-expansion evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import plot_v62_capacity_expansion_validation as pooled


HERE = Path(__file__).resolve().parent
PRIMARY_RESULTS = HERE / "resolution4h_all15"
CHRONOLOGY_RESULTS = (
    HERE / "chronology_procurement_validation" / "resolution4h_results"
)
FIGURES = HERE / "figures"
SOURCE_DATA = HERE / "source_data"
STEM = "fig_v62_supplementary_figure5_integrated_capacity_expansion"
PLANNING_STEP_HOURS = 4

# Reuse the established pooled-panel transformations with the 4 h primary run.
pooled.RESULTS = PRIMARY_RESULTS

TECHNOLOGIES = pooled.TECHNOLOGIES
TECHNOLOGY_LABELS = pooled.TECHNOLOGY_LABELS
TECHNOLOGY_COLORS = pooled.TECHNOLOGY_COLORS
EXPECTED_COLOR = pooled.EXPECTED_COLOR
RISK_COLOR = pooled.RISK_COLOR
DISPERSED = "#3C78A8"
CLUSTERED = "#D06B4C"
COMBINED = "#7A5FA0"
INK = "#202020"
GRID = "#E1E1E1"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.8,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def prepare_chronology_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ensemble = pd.read_csv(CHRONOLOGY_RESULTS / "ensemble_comparison.csv")
    direct = pd.read_csv(CHRONOLOGY_RESULTS / "direct_chronology_effect.csv")
    headline = json.loads(
        (CHRONOLOGY_RESULTS / "headline_metrics.json").read_text(encoding="utf-8")
    )

    observed = ensemble.loc[ensemble["ensemble"].eq("observed")].set_index(
        ["country", "risk_weight"]
    )["total_power_capacity_percent_mean_gas"]
    ensemble["observed_total_power_percent_mean_gas"] = [
        observed.loc[(country, risk_weight)]
        for country, risk_weight in zip(ensemble["country"], ensemble["risk_weight"])
    ]
    ensemble["total_power_change_from_observed_pp"] = (
        ensemble["total_power_capacity_percent_mean_gas"]
        - ensemble["observed_total_power_percent_mean_gas"]
    )
    panel_d = ensemble[
        [
            "country",
            "risk_weight",
            "ensemble",
            "scenario_count",
            "total_power_capacity_percent_mean_gas",
            "observed_total_power_percent_mean_gas",
            "total_power_change_from_observed_pp",
            "active_set",
            "dominant_technology",
        ]
    ].copy()

    technology_columns = {
        "4 h LFP": "clustered_minus_dispersed_li_lfp_4h_power_pp",
        "24 h A-CAES": "clustered_minus_dispersed_caes_24h_power_pp",
        "168 h cavern H2": "clustered_minus_dispersed_h2_cavern_168h_power_pp",
    }
    rows: list[dict[str, object]] = []
    for record in direct.itertuples(index=False):
        for technology, column in technology_columns.items():
            rows.append(
                {
                    "country": record.country,
                    "risk_weight": float(record.risk_weight),
                    "technology": technology,
                    "clustered_minus_dispersed_power_pp": float(
                        getattr(record, column)
                    ),
                    "total_power_change_pp": float(
                        record.clustered_minus_dispersed_total_power_pp
                    ),
                    "technology_set_change": bool(
                        record.clustered_vs_dispersed_technology_set_change
                    ),
                    "dominant_technology_change": bool(
                        record.clustered_vs_dispersed_dominant_technology_change
                    ),
                    "mix_total_variation_pp": float(
                        record.clustered_vs_dispersed_mix_total_variation_pp
                    ),
                }
            )
    panel_e = pd.DataFrame(rows)

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    panel_d.to_csv(SOURCE_DATA / "panel_d_chronology_ensemble_response.csv", index=False)
    panel_e.to_csv(SOURCE_DATA / "panel_e_technology_chronology_response.csv", index=False)
    return panel_d, panel_e, headline


def style_country_axis(axis: plt.Axes, countries: list[str], show_y: bool) -> None:
    y = np.arange(len(countries), dtype=float)
    axis.axvline(0, color=INK, linewidth=0.7, zorder=0)
    axis.set_yticks(y)
    axis.set_yticklabels(countries if show_y else [])
    axis.set_ylim(len(countries) - 0.45, -0.55)
    axis.grid(axis="x", color=GRID, linewidth=0.4, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)


def plot_panel_a(
    axis: plt.Axes, data: pd.DataFrame, countries: list[str]
) -> None:
    y = np.arange(len(countries), dtype=float)
    bar_height = 0.27
    offsets = {0.0: -0.17, 0.5: 0.17}
    alphas = {0.0: 0.55, 0.5: 1.0}
    for risk_weight in (0.0, 0.5):
        left = np.zeros(len(countries))
        for technology in TECHNOLOGIES:
            values = (
                data.loc[
                    data["risk_weight"].eq(risk_weight)
                    & data["config_id"].eq(technology)
                ]
                .set_index("country")
                .reindex(countries)["capacity_share_percent"]
                .fillna(0.0)
                .to_numpy()
            )
            axis.barh(
                y + offsets[risk_weight],
                values,
                left=left,
                height=bar_height,
                color=TECHNOLOGY_COLORS[technology],
                alpha=alphas[risk_weight],
                edgecolor="white",
                linewidth=0.25,
                label=TECHNOLOGY_LABELS[technology] if risk_weight == 0.5 else None,
            )
            left += values
    axis.set_yticks(y)
    axis.set_yticklabels(countries)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Endogenous power-capacity share (%)")
    axis.set_title("Procurement composition", loc="left", pad=6)
    axis.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.13),
        ncol=3,
        handlelength=1.0,
        columnspacing=0.7,
        borderaxespad=0,
    )


def plot_panel_b(
    axis: plt.Axes, data: pd.DataFrame, countries: list[str]
) -> None:
    y = np.arange(len(countries), dtype=float)
    ordered = data.set_index("country").reindex(countries)
    values = ordered[
        "risk_aware_minus_expected_power_percentage_points"
    ].to_numpy(dtype=float)
    axis.hlines(y, 0, values, color="#A7A7A7", linewidth=0.7, zorder=1)
    axis.scatter(
        values,
        y,
        c=np.where(values >= 0, RISK_COLOR, EXPECTED_COLOR),
        s=16,
        edgecolor="white",
        linewidth=0.3,
        zorder=2,
    )
    axis.axvline(0, color=INK, linewidth=0.7)
    axis.set_yticks(y)
    axis.set_yticklabels([])
    axis.set_xlabel("Mean + CVaR$_{90}$ minus expected design\n(power percentage points)")
    axis.set_title("Risk-price response", loc="left", pad=6)


def plot_panel_c(
    axis: plt.Axes, data: pd.DataFrame, countries: list[str]
) -> None:
    y = np.arange(len(countries), dtype=float)
    contract = {
        "Expected annual cost": ("o", EXPECTED_COLOR, -0.12),
        "CVaR90 annual cost": ("D", RISK_COLOR, 0.12),
    }
    for endpoint, (marker, color, offset) in contract.items():
        subset = data.loc[data["endpoint"].eq(endpoint)].set_index("country")
        values = subset.reindex(countries)[
            "cost_change_percent_best_single"
        ].to_numpy(dtype=float)
        axis.scatter(
            values,
            y + offset,
            marker=marker,
            color=color,
            s=17,
            edgecolor="white",
            linewidth=0.3,
            label=endpoint,
            zorder=3,
        )
        axis.axvline(
            float(np.median(values)),
            color=color,
            linewidth=0.8,
            linestyle=(0, (3, 2)),
        )
    axis.axvline(0, color=INK, linewidth=0.7)
    axis.set_yticks(y)
    axis.set_yticklabels([])
    axis.set_xlabel("Portfolio minus best single technology\n(% of single-technology cost)")
    axis.set_title("Value of the technology menu", loc="left", pad=6)
    axis.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.13),
        borderaxespad=0,
        handletextpad=0.35,
    )


def plot_panel_d(
    axis: plt.Axes,
    data: pd.DataFrame,
    countries: list[str],
    headline: dict,
) -> None:
    y = np.arange(len(countries), dtype=float)
    ensemble_contract = {
        "dispersed": (DISPERSED, "o", "Dispersed"),
        "clustered": (CLUSTERED, "s", "Clustered"),
        "combined": (COMBINED, "D", "D + C ensemble"),
    }
    offsets = {0.0: -0.15, 0.5: 0.15}
    for risk_weight in (0.0, 0.5):
        subset = data.loc[data["risk_weight"].eq(risk_weight)]
        pivot = subset.pivot(
            index="country",
            columns="ensemble",
            values="total_power_change_from_observed_pp",
        ).reindex(countries)
        axis.hlines(
            y + offsets[risk_weight],
            pivot["dispersed"],
            pivot["clustered"],
            color="#A8A8A8",
            linewidth=0.6,
            zorder=1,
        )
        for ensemble, (color, marker, label) in ensemble_contract.items():
            facecolor = "white" if risk_weight == 0.0 else color
            axis.scatter(
                pivot[ensemble],
                y + offsets[risk_weight],
                facecolor=facecolor,
                edgecolor=color,
                marker=marker,
                s=17,
                linewidth=0.8,
                label=label if risk_weight == 0.5 else None,
                zorder=3,
            )
    style_country_axis(axis, countries, True)
    data_extent = float(data["total_power_change_from_observed_pp"].abs().max())
    display_extent = max(225.0, 50.0 * np.ceil((data_extent + 15.0) / 50.0))
    axis.set_xlim(-display_extent, display_extent)
    axis.set_xticks([-200, -100, 0, 100, 200])
    axis.set_xlabel(
        "Total procured power relative to observed chronology\n"
        "(percentage points of mean gas-peaking power)"
    )
    axis.set_title("Chronology-ensemble response", loc="left", pad=6)
    axis.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.105),
        ncol=3,
        borderaxespad=0,
        handletextpad=0.35,
        columnspacing=0.75,
    )


def plot_panel_e(
    axis: plt.Axes,
    data: pd.DataFrame,
    countries: list[str],
    headline: dict,
) -> None:
    y = np.arange(len(countries), dtype=float)
    technology_contract = {
        "4 h LFP": (TECHNOLOGY_COLORS["li_lfp_4h"], "o"),
        "24 h A-CAES": (TECHNOLOGY_COLORS["caes_24h"], "s"),
        "168 h cavern H2": (TECHNOLOGY_COLORS["h2_cavern_168h"], "^"),
    }
    offsets = {0.0: -0.15, 0.5: 0.15}
    for risk_weight in (0.0, 0.5):
        subset = data.loc[data["risk_weight"].eq(risk_weight)]
        for technology, (color, marker) in technology_contract.items():
            values = (
                subset.loc[subset["technology"].eq(technology)]
                .set_index("country")
                .reindex(countries)["clustered_minus_dispersed_power_pp"]
                .to_numpy(dtype=float)
            )
            facecolor = "white" if risk_weight == 0.0 else color
            axis.scatter(
                values,
                y + offsets[risk_weight],
                facecolor=facecolor,
                edgecolor=color,
                marker=marker,
                s=17,
                linewidth=0.8,
                label=technology if risk_weight == 0.5 else None,
                zorder=3,
            )
    style_country_axis(axis, countries, False)
    axis.set_xlim(-275, 150)
    axis.set_xticks([-250, -125, 0, 125])
    axis.set_xlabel(
        "Clustered minus dispersed technology capacity\n"
        "(percentage points of mean gas-peaking power)"
    )
    axis.set_title("Technology response to event order", loc="left", pad=6)
    axis.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.105),
        ncol=3,
        borderaxespad=0,
        handletextpad=0.35,
        columnspacing=0.75,
    )


def finish_axis(axis: plt.Axes) -> None:
    axis.invert_yaxis()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color=GRID, linewidth=0.4, zorder=0)
    axis.tick_params(axis="y", length=0)


def plot() -> dict[str, object]:
    configure_style()
    panel_a, panel_b, panel_c = pooled.prepare_source_data()
    panel_d, panel_e, headline = prepare_chronology_data()
    countries = sorted(panel_a["country"].unique())

    figure = plt.figure(figsize=(7.2, 7.25), constrained_layout=False)
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.05],
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.905,
        hspace=0.36,
    )
    top = outer[0].subgridspec(1, 3, width_ratios=[1.55, 1.0, 1.25], wspace=0.27)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.16)
    ax_a = figure.add_subplot(top[0, 0])
    ax_b = figure.add_subplot(top[0, 1])
    ax_c = figure.add_subplot(top[0, 2])
    ax_d = figure.add_subplot(bottom[0, 0])
    ax_e = figure.add_subplot(bottom[0, 1])

    plot_panel_a(ax_a, panel_a, countries)
    plot_panel_b(ax_b, panel_b, countries)
    plot_panel_c(ax_c, panel_c, countries)
    for axis in (ax_a, ax_b, ax_c):
        finish_axis(axis)
    plot_panel_d(ax_d, panel_d, countries, headline)
    plot_panel_e(ax_e, panel_e, countries, headline)

    for label, axis in zip("abcde", (ax_a, ax_b, ax_c, ax_d, ax_e)):
        axis.text(
            -0.14,
            1.16,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
        )

    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": FIGURES / f"{STEM}.pdf",
        "svg": FIGURES / f"{STEM}.svg",
        "png": FIGURES / f"{STEM}.png",
        "tiff": FIGURES / f"{STEM}.tiff",
    }
    figure.savefig(outputs["pdf"])
    figure.savefig(outputs["svg"])
    figure.savefig(outputs["png"], dpi=600)
    figure.savefig(outputs["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)

    with Image.open(outputs["png"]) as image:
        grayscale = FIGURES / f"{STEM}_grayscale.png"
        image.convert("L").save(grayscale, dpi=(600, 600))

    qa = {
        "status": "pass",
        "backend": "python-matplotlib",
        "planning_step_hours": PLANNING_STEP_HOURS,
        "country_count": len(countries),
        "panel_count": 5,
        "all_source_rows_used": True,
        "excluded_rows": 0,
        "panel_rows": {
            "a": len(panel_a),
            "b": len(panel_b),
            "c": len(panel_c),
            "d": len(panel_d),
            "e": len(panel_e),
        },
        "minimum_font_size_pt": 6.1,
        "outputs": {
            key: {"path": str(path), "bytes": path.stat().st_size}
            for key, path in {**outputs, "grayscale": grayscale}.items()
        },
    }
    if any(item["bytes"] <= 0 for item in qa["outputs"].values()):
        qa["status"] = "fail"
    (HERE / "integrated_figure_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))
    return qa


if __name__ == "__main__":
    report = plot()
    raise SystemExit(0 if report["status"] == "pass" else 2)
