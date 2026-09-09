#!/usr/bin/env python3
"""Plot the v64 Supplementary Figure 6 resolution-convergence analysis."""

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
STEM = "fig_v64_supplementary_figure6_resolution_convergence"

TWO_H = "#CC79A7"
FOUR_H = "#0072B2"
SIX_H = "#767676"
EXPECTED = "#3C5488"
RISK = "#B24745"
INK = "#202020"
GRID = "#E2E2E2"
ENSEMBLE_LABELS = {
    "observed": "Observed",
    "dispersed": "Dispersed",
    "clustered": "Clustered",
    "combined": "Pooled",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6.6,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.6,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 6.0,
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = pd.read_csv(RESULTS / "pooled_resolution_convergence.csv")
    risk = pd.read_csv(RESULTS / "risk_pricing_direction_convergence.csv")
    ensembles = pd.read_csv(RESULTS / "chronology_ensemble_resolution_convergence.csv")
    event = pd.read_csv(RESULTS / "event_order_direction_convergence.csv")
    if len(pooled) != 30 or len(risk) != 15 or len(ensembles) != 120 or len(event) != 30:
        raise RuntimeError("Resolution-convergence source data are incomplete")
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(SOURCE_DATA / "panel_b_pooled_three_grid_trajectories.csv", index=False)
    risk.to_csv(SOURCE_DATA / "panel_a_risk_direction.csv", index=False)
    ensembles.to_csv(
        SOURCE_DATA / "panels_c_f_all_chronology_ensembles.csv", index=False
    )
    event.to_csv(SOURCE_DATA / "panels_d_e_event_order_direction.csv", index=False)
    return pooled, risk, ensembles, event


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color=GRID, linewidth=0.4, zorder=0)


def country_resolution_axis(
    axis: plt.Axes,
    frame: pd.DataFrame,
    countries: list[str],
    stems: dict[int, str],
    title: str,
    xlabel: str,
    show_y: bool,
) -> None:
    indexed = frame.set_index("country").reindex(countries)
    y = np.arange(len(countries), dtype=float)
    values = {resolution: indexed[column].to_numpy(dtype=float) for resolution, column in stems.items()}
    axis.hlines(y, values[6], values[4], color="#C7C7C7", linewidth=0.65, zorder=1)
    axis.hlines(y, values[4], values[2], color="#8BB9D7", linewidth=0.75, zorder=1)
    axis.scatter(values[6], y, marker="x", color=SIX_H, s=16, linewidth=0.8, label="6 h", zorder=3)
    axis.scatter(values[4], y, marker="o", facecolor=FOUR_H, edgecolor="white", s=18, linewidth=0.35, label="4 h", zorder=4)
    axis.scatter(values[2], y, marker="s", facecolor=TWO_H, edgecolor="white", s=17, linewidth=0.35, label="2 h", zorder=5)
    axis.axvline(0, color=INK, linewidth=0.65, zorder=0)
    axis.set_yticks(y)
    axis.set_yticklabels(countries if show_y else [])
    axis.set_ylim(len(countries) - 0.45, -0.55)
    axis.tick_params(axis="y", length=0)
    axis.set_title(title, loc="left", pad=4)
    axis.set_xlabel(xlabel)
    clean_axis(axis)


def plot() -> dict[str, object]:
    configure_style()
    pooled, risk, ensembles, event = load_data()
    countries = sorted(risk["country"].astype(str).unique())

    figure = plt.figure(figsize=(7.2, 8.45))
    grid = figure.add_gridspec(
        3,
        2,
        left=0.09,
        right=0.985,
        bottom=0.065,
        top=0.935,
        hspace=0.42,
        wspace=0.26,
        height_ratios=[1.0, 1.0, 0.78],
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_d = figure.add_subplot(grid[1, 0])
    ax_e = figure.add_subplot(grid[1, 1])
    ax_c = figure.add_subplot(grid[2, 0])
    ax_f = figure.add_subplot(grid[2, 1])

    country_resolution_axis(
        ax_a,
        risk,
        countries,
        {
            2: "risk_minus_expected_2h_pp",
            4: "risk_minus_expected_4h_pp",
            6: "risk_minus_expected_6h_pp",
        },
        "Risk-price response",
        "Mean + CVaR$_{90}$ minus expected design\n(power percentage points)",
        True,
    )

    grid_hours = np.array([2, 4, 6], dtype=float)
    power_columns = ["power_2h_pp", "power_4h_pp", "power_6h_pp"]
    pooled_power = pooled[power_columns].to_numpy(dtype=float)
    if np.any(pooled_power <= 0):
        raise RuntimeError("Pooled procurement must be positive for the log-scale grid comparison")
    for risk_weight, marker, colour, label in (
        (0.0, "o", EXPECTED, "Expected cost"),
        (0.5, "D", RISK, "Mean + CVaR$_{90}$"),
    ):
        subset = pooled.loc[pooled["risk_weight"].eq(risk_weight)]
        for _, row in subset.iterrows():
            ax_b.plot(
                grid_hours,
                row[power_columns].to_numpy(dtype=float),
                color=colour,
                linewidth=0.55,
                alpha=0.18,
                zorder=1,
            )
        values = subset[power_columns].to_numpy(dtype=float)
        lower, median, upper = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
        ax_b.vlines(grid_hours, lower, upper, color=colour, linewidth=2.0, zorder=3)
        ax_b.plot(
            grid_hours,
            median,
            color=colour,
            marker=marker,
            markersize=4.2,
            linewidth=1.15,
            label=label,
            zorder=4,
        )
    ax_b.set_yscale("log")
    ax_b.set_xticks(grid_hours)
    ax_b.set_xticklabels(["2 h", "4 h", "6 h"])
    ax_b.set_title("Pooled procurement across planning grids", loc="left", pad=4)
    ax_b.set_xlabel("Planning-grid interval")
    ax_b.set_ylabel("Total procured power\n(% of mean gas-peaking power)")
    ax_b.legend(frameon=False, loc="upper right", borderaxespad=0.3, handletextpad=0.35)
    clean_axis(ax_b)
    ax_b.grid(axis="y", color=GRID, linewidth=0.4, zorder=0)

    expected_event = event.loc[event["risk_weight"].eq(0.0)]
    risk_event = event.loc[event["risk_weight"].eq(0.5)]
    event_stems = {
        2: "clustered_minus_dispersed_2h_pp",
        4: "clustered_minus_dispersed_4h_pp",
        6: "clustered_minus_dispersed_6h_pp",
    }
    country_resolution_axis(
        ax_d,
        expected_event,
        countries,
        event_stems,
        "Event-order response: expected cost",
        "Clustered minus dispersed total power\n(percentage points)",
        True,
    )
    country_resolution_axis(
        ax_e,
        risk_event,
        countries,
        event_stems,
        "Event-order response: mean + CVaR$_{90}$",
        "Clustered minus dispersed total power\n(percentage points)",
        False,
    )

    ensemble_order = ["observed", "dispersed", "clustered", "combined"]
    objective_order = [(0.0, "Exp."), (0.5, "Risk")]
    x_ensembles = np.arange(len(ensemble_order), dtype=float)
    bar_width = 0.34
    for ensemble_index, ensemble in enumerate(ensemble_order):
        for objective_index, (risk_weight, objective_label) in enumerate(objective_order):
            subset = ensembles.loc[
                ensembles["ensemble"].eq(ensemble)
                & ensembles["risk_weight"].eq(risk_weight)
            ]
            count = int(subset["active_set_agreement_all_three"].sum())
            offset = (objective_index - 0.5) * bar_width
            colour = EXPECTED if risk_weight == 0.0 else RISK
            ax_c.bar(
                ensemble_index + offset,
                count,
                width=bar_width,
                color=colour,
                alpha=0.9,
                label=objective_label if ensemble_index == 0 else None,
                zorder=2,
            )
            ax_c.text(
                ensemble_index + offset,
                count + 0.35,
                f"{count}/15",
                ha="center",
                va="bottom",
                fontsize=5.7,
            )
    ax_c.set_xticks(x_ensembles)
    ax_c.set_xticklabels([ENSEMBLE_LABELS[item] for item in ensemble_order])
    ax_c.set_ylim(0, 16.2)
    ax_c.set_yticks([0, 5, 10, 15])
    ax_c.set_ylabel("Countries")
    ax_c.set_title("Same active technology set on all three grids", loc="left", pad=4)
    ax_c.legend(frameon=False, loc="lower left", ncol=2, handletextpad=0.35, columnspacing=0.8)
    clean_axis(ax_c)
    ax_c.grid(axis="y", color=GRID, linewidth=0.4, zorder=0)

    for ensemble_index, ensemble in enumerate(ensemble_order):
        subset = ensembles.loc[ensembles["ensemble"].eq(ensemble)]
        for offset, risk_weight, colour, marker, label in (
            (-0.15, 0.0, EXPECTED, "o", "Expected cost"),
            (0.15, 0.5, RISK, "D", "Mean + CVaR$_{90}$"),
        ):
            x = ensemble_index + offset
            values = subset.loc[
                subset["risk_weight"].eq(risk_weight),
                "power_range_2h_4h_6h_pp",
            ].to_numpy(dtype=float)
            jitter = np.linspace(-0.075, 0.075, num=len(values))
            ax_f.scatter(
                np.full(len(values), x) + jitter,
                values,
                s=11,
                marker=marker,
                facecolor="none" if marker == "o" else colour,
                edgecolor=colour,
                linewidth=0.55,
                alpha=0.55,
                zorder=2,
            )
            lower, median, upper = np.quantile(values, [0.25, 0.5, 0.75])
            ax_f.vlines(x, lower, upper, color=colour, linewidth=2.0, zorder=4)
            ax_f.scatter(x, median, s=25, marker=marker, color=colour, linewidth=0.8, zorder=5)
    ax_f.set_yscale("symlog", linthresh=1.0, linscale=0.7)
    ax_f.set_xticks(np.arange(4))
    ax_f.set_xticklabels([ENSEMBLE_LABELS[item] for item in ensemble_order])
    ax_f.set_ylabel("Within-design power range across\n2 h, 4 h and 6 h grids (pp)")
    ax_f.set_title("Planning-grid range across ensembles", loc="left", pad=4)
    ax_f.grid(axis="y", color=GRID, linewidth=0.4, zorder=0)
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)

    handles, labels_legend = ax_a.get_legend_handles_labels()
    figure.legend(
        handles,
        labels_legend,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.09, 0.985),
        ncol=3,
        borderaxespad=0,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    panel_axes = {"a": ax_a, "b": ax_b, "c": ax_c, "d": ax_d, "e": ax_e, "f": ax_f}
    for label, axis in panel_axes.items():
        axis.text(
            -0.15,
            1.08,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.3,
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

    with Image.open(outputs["png"]) as rendered:
        grayscale = FIGURES / f"{STEM}_grayscale.png"
        rendered.convert("L").save(grayscale, dpi=(600, 600))

    comparison_qa = json.loads((RESULTS / "qa.json").read_text(encoding="utf-8"))
    qa = {
        "status": "pass",
        "backend": "python-matplotlib",
        "source_qa_status": comparison_qa.get("status"),
        "country_count": len(countries),
        "pooled_rows": len(pooled),
        "risk_rows": len(risk),
        "chronology_ensemble_rows": len(ensembles),
        "event_order_rows": len(event),
        "all_source_rows_used": True,
        "excluded_rows": 0,
        "panel_count": 6,
        "minimum_font_size_pt": 5.7,
        "outputs": {
            key: {"path": str(path), "bytes": path.stat().st_size}
            for key, path in {**outputs, "grayscale": grayscale}.items()
        },
    }
    if qa["source_qa_status"] != "pass" or any(
        item["bytes"] <= 0 for item in qa["outputs"].values()
    ):
        qa["status"] = "fail"
    (HERE / "figure_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))
    return qa


if __name__ == "__main__":
    report = plot()
    raise SystemExit(0 if report["status"] == "pass" else 2)
