#!/usr/bin/env python3
"""Plot Supplementary Figure 6 comparing the 4 h and 6 h planning grids."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
COMPARISON = HERE / "resolution4h_vs6h"
FIGURES = HERE / "figures"
SOURCE_DATA = HERE / "source_data"
STEM = "fig_v62_supplementary_figure6_planning_resolution"

FOUR_H = "#0072B2"
SIX_H = "#767676"
EXPECTED = "#3C5488"
RISK = "#B24745"
INK = "#202020"
GRID = "#E1E1E1"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6.8,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
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


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    risk = pd.read_csv(COMPARISON / "risk_shift_country_pair.csv")
    pooled = pd.read_csv(COMPARISON / "pooled_country_pair.csv")
    chronology = pd.read_csv(COMPARISON / "chronology_effect_country_pair.csv")
    countries = sorted(risk["country"].astype(str).unique())
    if countries != sorted(pooled["country"].astype(str).unique()):
        raise RuntimeError("Pooled and risk-response country sets differ")
    if countries != sorted(chronology["country"].astype(str).unique()):
        raise RuntimeError("Chronology and risk-response country sets differ")
    if len(risk) != 15 or len(pooled) != 30 or len(chronology) != 30:
        raise RuntimeError("Resolution comparison is incomplete")

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    risk.to_csv(
        SOURCE_DATA / "supplementary_figure6_panel_a_risk_response.csv", index=False
    )
    pooled.to_csv(
        SOURCE_DATA / "supplementary_figure6_panel_b_capacity_shift.csv", index=False
    )
    chronology.to_csv(
        SOURCE_DATA / "supplementary_figure6_panels_c_d_chronology_response.csv",
        index=False,
    )
    return risk, pooled, chronology


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


def paired_resolution_axis(
    axis: plt.Axes,
    data: pd.DataFrame,
    countries: list[str],
    column_6h: str,
    column_4h: str,
    title: str,
    xlabel: str,
    show_y: bool,
    xlim: tuple[float, float],
    xticks: list[float],
) -> None:
    ordered = data.set_index("country").reindex(countries)
    values_6h = ordered[column_6h].to_numpy(dtype=float)
    values_4h = ordered[column_4h].to_numpy(dtype=float)
    y = np.arange(len(countries), dtype=float)
    axis.hlines(y, values_6h, values_4h, color="#B8B8B8", linewidth=0.7, zorder=1)
    axis.scatter(
        values_6h,
        y,
        marker="x",
        color=SIX_H,
        s=18,
        linewidth=0.9,
        label="6 h sensitivity",
        zorder=3,
    )
    axis.scatter(
        values_4h,
        y,
        marker="o",
        facecolor=FOUR_H,
        edgecolor="white",
        s=20,
        linewidth=0.35,
        label="4 h primary",
        zorder=4,
    )
    style_country_axis(axis, countries, show_y)
    axis.set_xlim(*xlim)
    axis.set_xticks(xticks)
    axis.set_title(title, loc="left", pad=6)
    axis.set_xlabel(xlabel)


def plot() -> dict[str, object]:
    configure_style()
    risk, pooled, chronology = load_source_data()
    countries = sorted(risk["country"].astype(str).unique())

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 7.3),
        gridspec_kw={
            "left": 0.08,
            "right": 0.985,
            "bottom": 0.08,
            "top": 0.91,
            "hspace": 0.34,
            "wspace": 0.18,
        },
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    paired_resolution_axis(
        ax_a,
        risk,
        countries,
        "risk_aware_minus_expected_power_percentage_points_6h",
        "risk_aware_minus_expected_power_percentage_points_4h",
        "Risk-price response",
        "Mean + CVaR$_{90}$ minus expected design\n(power percentage points)",
        True,
        (-215, 90),
        [-200, -100, 0, 75],
    )

    expected = pooled.loc[pooled["risk_weight"].eq(0.0)].set_index("country")
    risk_aware = pooled.loc[pooled["risk_weight"].eq(0.5)].set_index("country")
    y = np.arange(len(countries), dtype=float)
    ax_b.scatter(
        expected.reindex(countries)["power_4h_minus_6h_pp"],
        y - 0.12,
        marker="o",
        facecolor="white",
        edgecolor=EXPECTED,
        s=20,
        linewidth=0.9,
        label="Expected cost",
        zorder=3,
    )
    ax_b.scatter(
        risk_aware.reindex(countries)["power_4h_minus_6h_pp"],
        y + 0.12,
        marker="D",
        facecolor=RISK,
        edgecolor="white",
        s=19,
        linewidth=0.35,
        label="Mean + CVaR$_{90}$",
        zorder=3,
    )
    style_country_axis(ax_b, countries, False)
    ax_b.set_xlim(-25, 280)
    ax_b.set_xticks([0, 100, 200])
    ax_b.set_title("Resolution shift in total procurement", loc="left", pad=6)
    ax_b.set_xlabel(
        "4 h minus 6 h total procured power\n(percentage points of mean gas peaking)"
    )
    ax_b.legend(
        frameon=False,
        loc="upper right",
        ncol=1,
        borderaxespad=0.3,
        columnspacing=0.7,
        handletextpad=0.35,
    )

    expected_chronology = chronology.loc[chronology["risk_weight"].eq(0.0)]
    risk_chronology = chronology.loc[chronology["risk_weight"].eq(0.5)]
    chronology_columns = (
        "clustered_minus_dispersed_total_power_pp_6h",
        "clustered_minus_dispersed_total_power_pp_4h",
    )
    paired_resolution_axis(
        ax_c,
        expected_chronology,
        countries,
        chronology_columns[0],
        chronology_columns[1],
        "Event-order response: expected cost",
        "Clustered minus dispersed total power\n(percentage points of mean gas peaking)",
        True,
        (-225, 100),
        [-200, -100, 0, 75],
    )
    paired_resolution_axis(
        ax_d,
        risk_chronology,
        countries,
        chronology_columns[0],
        chronology_columns[1],
        "Event-order response: mean + CVaR$_{90}$",
        "Clustered minus dispersed total power\n(percentage points of mean gas peaking)",
        False,
        (-225, 100),
        [-200, -100, 0, 75],
    )

    handles, labels = ax_a.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.985),
        ncol=2,
        borderaxespad=0,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(
            -0.13,
            1.09,
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

    comparison_qa = json.loads((COMPARISON / "qa.json").read_text(encoding="utf-8"))
    qa = {
        "status": "pass",
        "backend": "python-matplotlib",
        "source_qa_status": comparison_qa.get("status"),
        "country_count": len(countries),
        "panel_count": 4,
        "all_source_rows_used": True,
        "excluded_rows": 0,
        "panel_rows": {
            "a": len(risk),
            "b": len(pooled),
            "c": len(expected_chronology),
            "d": len(risk_chronology),
        },
        "minimum_font_size_pt": 6.2,
        "outputs": {
            key: {"path": str(path), "bytes": path.stat().st_size}
            for key, path in {**outputs, "grayscale": grayscale}.items()
        },
    }
    if qa["source_qa_status"] != "pass" or any(
        item["bytes"] <= 0 for item in qa["outputs"].values()
    ):
        qa["status"] = "fail"
    (HERE / "resolution_figure_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))
    return qa


if __name__ == "__main__":
    report = plot()
    raise SystemExit(0 if report["status"] == "pass" else 2)
