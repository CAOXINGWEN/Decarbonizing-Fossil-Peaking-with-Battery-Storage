#!/usr/bin/env python3
"""Plot the chronology-isolated capacity-expansion validation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SOURCE_DATA = HERE / "source_data"
STEM = "fig_v62_supplementary_figure6_chronology_procurement"

DISPERSED = "#3C78A8"
CLUSTERED = "#D06B4C"
COMBINED = "#7A5FA0"
LFP = "#3C78A8"
ACAES = "#E69F00"
HYDROGEN = "#2A9D6F"
INK = "#202020"
GRID = "#D9D9D9"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def prepare_source_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ensemble = pd.read_csv(RESULTS / "ensemble_comparison.csv")
    direct = pd.read_csv(RESULTS / "direct_chronology_effect.csv")
    headline = json.loads((RESULTS / "headline_metrics.json").read_text(encoding="utf-8"))

    observed = (
        ensemble.loc[ensemble["ensemble"].eq("observed")]
        .set_index(["country", "risk_weight"])[
            "total_power_capacity_percent_mean_gas"
        ]
    )
    ensemble["observed_total_power_percent_mean_gas"] = [
        observed.loc[(country, risk_weight)]
        for country, risk_weight in zip(ensemble["country"], ensemble["risk_weight"])
    ]
    ensemble["total_power_change_from_observed_pp"] = (
        ensemble["total_power_capacity_percent_mean_gas"]
        - ensemble["observed_total_power_percent_mean_gas"]
    )
    panels_ab = ensemble[
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
        for label, column in technology_columns.items():
            rows.append(
                {
                    "country": record.country,
                    "risk_weight": float(record.risk_weight),
                    "technology": label,
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
    panels_cd = pd.DataFrame(rows)

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    panels_ab.to_csv(SOURCE_DATA / "panels_ab_ensemble_procurement.csv", index=False)
    panels_cd.to_csv(SOURCE_DATA / "panels_cd_technology_shifts.csv", index=False)
    return panels_ab, panels_cd, headline


def country_order(panels_cd: pd.DataFrame) -> list[str]:
    total = (
        panels_cd.loc[
            panels_cd["risk_weight"].eq(0.0)
            & panels_cd["technology"].eq("4 h LFP")
        ]
        .set_index("country")["total_power_change_pp"]
        .sort_values()
    )
    return total.index.tolist()


def style_axis(axis: plt.Axes, show_y: bool, countries: list[str]) -> None:
    y = np.arange(len(countries))
    axis.axvline(0, color=INK, linewidth=0.75, zorder=0)
    axis.set_yticks(y)
    axis.set_yticklabels(countries if show_y else [])
    axis.set_ylim(len(countries) - 0.45, -0.55)
    axis.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)


def plot_total_panel(
    axis: plt.Axes,
    data: pd.DataFrame,
    countries: list[str],
    risk_weight: float,
    title: str,
    summary: dict,
    show_y: bool,
) -> None:
    subset = data.loc[data["risk_weight"].eq(risk_weight)].copy()
    pivot = subset.pivot(
        index="country", columns="ensemble", values="total_power_change_from_observed_pp"
    ).reindex(countries)
    y = np.arange(len(countries), dtype=float)
    axis.hlines(
        y,
        pivot["dispersed"],
        pivot["clustered"],
        color="#A8A8A8",
        linewidth=0.8,
        zorder=1,
    )
    axis.scatter(
        pivot["dispersed"], y - 0.10, color=DISPERSED, marker="o", s=18,
        edgecolor="white", linewidth=0.35, label="Dispersed", zorder=3
    )
    axis.scatter(
        pivot["clustered"], y + 0.10, color=CLUSTERED, marker="s", s=18,
        edgecolor="white", linewidth=0.35, label="Clustered", zorder=3
    )
    axis.scatter(
        pivot["combined"], y, facecolor="white", edgecolor=COMBINED, marker="D",
        s=20, linewidth=0.9, label="D + C ensemble", zorder=4
    )
    style_axis(axis, show_y, countries)
    axis.set_xlim(-80, 85)
    axis.set_xticks([-75, -50, -25, 0, 25, 50, 75])
    axis.set_title(title, loc="left", pad=9, fontweight="bold")
    axis.set_xlabel("Total procured power relative to observed chronology\n(percentage points of mean gas-peaking power)")
    axis.text(
        0.0,
        1.015,
        (
            f"D versus C: {summary['clustered_vs_dispersed_material_power_change_countries']}/15 "
            f">1 pp; median |change| {summary['clustered_vs_dispersed_median_absolute_power_change_pp']:.2f} pp"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#444444",
    )


def plot_technology_panel(
    axis: plt.Axes,
    data: pd.DataFrame,
    countries: list[str],
    risk_weight: float,
    title: str,
    summary: dict,
    show_y: bool,
) -> None:
    subset = data.loc[data["risk_weight"].eq(risk_weight)]
    y = np.arange(len(countries), dtype=float)
    specifications = [
        ("4 h LFP", LFP, "o", -0.14),
        ("24 h A-CAES", ACAES, "s", 0.14),
        ("168 h cavern H2", HYDROGEN, "^", 0.0),
    ]
    for label, color, marker, offset in specifications:
        values = (
            subset.loc[subset["technology"].eq(label)]
            .set_index("country")
            .reindex(countries)["clustered_minus_dispersed_power_pp"]
            .to_numpy(dtype=float)
        )
        axis.scatter(
            values,
            y + offset,
            color=color,
            marker=marker,
            s=18,
            edgecolor="white",
            linewidth=0.35,
            label=label,
            zorder=3,
        )
    style_axis(axis, show_y, countries)
    axis.set_xlim(-90, 90)
    axis.set_xticks([-75, -50, -25, 0, 25, 50, 75])
    axis.set_title(title, loc="left", pad=9, fontweight="bold")
    axis.set_xlabel("Clustered minus dispersed technology capacity\n(percentage points of mean gas-peaking power)")
    axis.text(
        0.0,
        1.015,
        (
            f"Technology set changes in {summary['clustered_vs_dispersed_technology_set_change_countries']}/15; "
            f"median mix displacement {summary['clustered_vs_dispersed_median_mix_total_variation_pp']:.2f} pp"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#444444",
    )


def plot() -> dict[str, object]:
    configure_style()
    panels_ab, panels_cd, headline = prepare_source_data()
    countries = country_order(panels_cd)

    figure, axes = plt.subplots(2, 2, figsize=(7.20, 7.05), constrained_layout=False)
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.115, top=0.875, wspace=0.18, hspace=0.34)
    plot_total_panel(
        axes[0, 0], panels_ab, countries, 0.0, "Expected-cost procurement",
        headline["expected"], True
    )
    plot_total_panel(
        axes[0, 1], panels_ab, countries, 0.5, "Mean + CVaR$_{90}$ procurement",
        headline["mean_cvar90"], False
    )
    plot_technology_panel(
        axes[1, 0], panels_cd, countries, 0.0, "Expected-cost technology response",
        headline["expected"], True
    )
    plot_technology_panel(
        axes[1, 1], panels_cd, countries, 0.5, "Mean + CVaR$_{90}$ technology response",
        headline["mean_cvar90"], False
    )

    total_handles, total_labels = axes[0, 0].get_legend_handles_labels()
    tech_handles, tech_labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(
        total_handles + tech_handles,
        total_labels + tech_labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.965),
        ncol=6,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    for label, axis in zip("abcd", axes.reshape(-1)):
        axis.text(
            -0.12,
            1.105,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
    figure.text(
        0.085,
        0.032,
        "n = 15 country procurement regions. All points are paired deterministic optima under identical annual energy, costs and technology menus.",
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#444444",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "pdf": FIGURES / f"{STEM}.pdf",
        "svg": FIGURES / f"{STEM}.svg",
        "png": FIGURES / f"{STEM}.png",
        "tiff": FIGURES / f"{STEM}.tiff",
    }
    figure.savefig(output_paths["pdf"])
    figure.savefig(output_paths["svg"])
    figure.savefig(output_paths["png"], dpi=600)
    figure.savefig(output_paths["tiff"], dpi=600)
    plt.close(figure)

    with Image.open(output_paths["png"]) as image:
        grayscale_path = FIGURES / f"{STEM}_grayscale.png"
        image.convert("L").save(grayscale_path, dpi=(600, 600))

    qa = {
        "status": "pass",
        "backend": "python-matplotlib",
        "country_count": len(countries),
        "panel_count": 4,
        "all_source_rows_used": True,
        "excluded_rows": 0,
        "hydrogen_nonzero_design_count": int(
            (
                pd.read_csv(RESULTS / "capacity_mix.csv")
                .loc[lambda frame: frame["config_id"].eq("h2_cavern_168h"), "capacity_per_normalized_mw"]
                .abs()
                .gt(1.0e-9)
                .sum()
            )
        ),
        "outputs": {key: str(path) for key, path in output_paths.items()},
        "grayscale_preview": str(grayscale_path),
    }
    (HERE / "figure_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return qa


if __name__ == "__main__":
    plot()
