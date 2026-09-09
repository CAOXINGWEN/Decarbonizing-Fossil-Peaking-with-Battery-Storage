#!/usr/bin/env python3
"""Build v44 Figure 6 from the frozen decision-facing evidence package.

The figure retains the v43 palette, typography, dimensions and asymmetric
five-panel layout, while replacing the unanchored high-value phase map with
direct procurement decisions and observed European capacity-price anchors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
FIGURE_WORK = HERE.parent
V44 = FIGURE_WORK.parent

JOINT = V44 / "joint_choice"
ANCHOR_SCAN = V44 / "anchor_scan"
DECISION = V44 / "v44_analysis" / "decision_flips"
VALUE_ANCHORS = V44 / "v44_analysis" / "value_anchors"

MAIN_OUT = FIGURE_WORK / "main_figures"
SOURCE_OUT = FIGURE_WORK / "source_data"

CHOICES_FILE = JOINT / "fixed_power_joint_choices.csv"
FLIP_SUMMARY_FILE = JOINT / "joint_choice_flip_summary.csv"
OPTION_VALUES_FILE = JOINT / "fixed_power_option_values.csv.gz"
NATURAL_PANEL_FILE = JOINT / "natural_fixed_effect_panel.csv"
NATURAL_COEF_FILE = JOINT / "natural_fixed_effect_coefficients.csv"
COST_SPAN_FILE = DECISION / "portfolio_cost_path_span_comparison.csv"
CAPACITY_CURVE_FILE = ANCHOR_SCAN / "capacity_product_flip_curve.csv"
CAPACITY_ANCHORS_FILE = VALUE_ANCHORS / "capacity_market_anchors.csv"

WIDTH = 182.88 / 25.4
HEIGHT = 5.30
PNG_DPI = 300
TIFF_DPI = 600
TIE_TOL = 0.01
CAPACITY_PRIORITY_LOW = 15.6946
CAPACITY_PRIORITY_HIGH = 90.4316
NGCC_NEW_ENTRY_FIXED_COST = 103.79921785880974

INK = "#233746"
MUTED = "#697980"
GRID = "#DFE7EA"
PALE = "#F4F7F7"
DISPERSED = "#35618D"
CLUSTERED = "#D8662E"
RECHARGE = "#158F83"
PURPLE = "#7056A3"
GOLD = "#D39B25"
RED = "#BA4A52"

NO_TAIL = "#CFD7DA"
LFP = DISPERSED
LDES = RECHARGE


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.labelsize": 6.6,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.55,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.2,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": TIFF_DPI,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_inputs() -> None:
    missing = [
        path
        for path in [
            CHOICES_FILE,
            FLIP_SUMMARY_FILE,
            OPTION_VALUES_FILE,
            NATURAL_PANEL_FILE,
            NATURAL_COEF_FILE,
            COST_SPAN_FILE,
            CAPACITY_CURVE_FILE,
            CAPACITY_ANCHORS_FILE,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing frozen Figure 6 inputs: " + ", ".join(map(str, missing)))


def save_source(frame: pd.DataFrame, name: str) -> Path:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    path = SOURCE_OUT / name
    frame.to_csv(path, index=False)
    return path


def panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    for label, ax in zip("abcde", axes):
        box = ax.get_position()
        fig.text(
            box.x0 - 0.010,
            box.y1 + 0.007,
            label,
            ha="right",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#111111",
        )


def build_panel_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    choices = pd.read_csv(CHOICES_FILE)
    selected = choices[
        choices["model_mode"].eq("fixed")
        & choices["account"].eq("efc_value")
        & choices["decision_scope"].eq("voluntary_tail")
    ].copy()
    expected_rows = 132 * 2 * 3
    if len(selected) != expected_rows:
        raise RuntimeError(f"Panel a expected {expected_rows} choice rows, found {len(selected)}")
    if selected[["country", "year"]].drop_duplicates().shape[0] != 132:
        raise RuntimeError("Panel a does not contain 132 unique country-years")

    selected["selection"] = selected["choice_family"].replace(
        {"no_tail": "No tail", "LFP": "4 h LFP", "LDES": "LDES"}
    )
    composition = (
        selected.groupby(["cost_year", "chronology", "selection"], observed=True)
        .size()
        .rename("country_year_count")
        .reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [[2025, 2030, 2035], ["dispersed", "clustered"], ["No tail", "4 h LFP", "LDES"]],
        names=["cost_year", "chronology", "selection"],
    ).to_frame(index=False)
    composition = complete.merge(
        composition,
        on=["cost_year", "chronology", "selection"],
        how="left",
        validate="one_to_one",
    )
    composition["country_year_count"] = composition["country_year_count"].fillna(0).astype(int)
    composition["share_percent"] = 100.0 * composition["country_year_count"] / 132.0
    if not composition.groupby(["cost_year", "chronology"])["country_year_count"].sum().eq(132).all():
        raise RuntimeError("Panel a composition does not close to 132 in every row")

    summary = pd.read_csv(FLIP_SUMMARY_FILE)
    summary = summary[
        summary["model_mode"].eq("fixed")
        & summary["account"].eq("efc_value")
        & summary["decision_scope"].eq("voluntary_tail")
    ][
        [
            "cost_year",
            "n_country_years",
            "disjoint_tie_set_flips",
            "disjoint_tie_set_flip_rate",
            "country_block_bootstrap_ci95_low",
            "country_block_bootstrap_ci95_high",
            "tail_to_no_tail_flips",
            "no_tail_to_tail_flips",
        ]
    ].sort_values("cost_year")
    if summary["disjoint_tie_set_flips"].astype(int).tolist() != [0, 3, 22]:
        raise RuntimeError("Panel a frozen robust flip counts differ from 0, 3 and 22")
    return composition, summary


def build_panel_b() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(COST_SPAN_FILE)
    data = data[
        data["decision_domain"].eq("all_49_ldes_configurations")
        & data["cost_year"].eq(2035)
    ].copy()
    if len(data) != 132 or data[["country", "year"]].duplicated().any():
        raise RuntimeError("Panel b expected one row for each of 132 country-years")
    data = data.sort_values("absolute_cluster_effect_eur_kw_year").reset_index(drop=True)
    data["ecdf_percent"] = 100.0 * np.arange(1, len(data) + 1) / len(data)
    summary = pd.DataFrame(
        [
            {
                "n_country_years": len(data),
                "median_absolute_clustering_effect_eur_kw_year": data[
                    "absolute_cluster_effect_eur_kw_year"
                ].median(),
                "minimum_cost_path_span_eur_kw_year": data[
                    "mean_chronology_cost_path_span_eur_kw_year"
                ].min(),
                "median_cost_path_span_eur_kw_year": data[
                    "mean_chronology_cost_path_span_eur_kw_year"
                ].median(),
                "maximum_cost_path_span_eur_kw_year": data[
                    "mean_chronology_cost_path_span_eur_kw_year"
                ].max(),
                "clustering_effect_at_least_cost_path_span_count": int(
                    data["cluster_effect_at_least_cost_path_span"].sum()
                ),
                "clustering_effect_at_least_cost_path_span_rate": float(
                    data["cluster_effect_at_least_cost_path_span"].mean()
                ),
            }
        ]
    )
    if int(summary.iloc[0]["clustering_effect_at_least_cost_path_span_count"]) != 60:
        raise RuntimeError("Panel b expected 60 country-years above the cost-path span")
    return data, summary


def build_panel_c() -> tuple[pd.DataFrame, pd.DataFrame]:
    curve = pd.read_csv(CAPACITY_CURVE_FILE).sort_values(["cost_year", "firm_value_eur_kw_year"])
    needed = {
        "firm_value_eur_kw_year",
        "cost_year",
        "selected_class_flip_count",
        "selected_class_flip_rate",
        "selected_class_flip_untoleranced_count",
    }
    if not needed.issubset(curve.columns):
        raise RuntimeError(f"Panel c capacity curve lacks fields: {sorted(needed - set(curve.columns))}")
    if curve.groupby("cost_year").size().to_dict() != {2025: 29, 2030: 29, 2035: 29}:
        raise RuntimeError("Panel c capacity scan is not 29 anchors by three cost years")
    curve["robust_flip_percent"] = 100.0 * curve["selected_class_flip_rate"]

    # The comparable four-market envelope is audited separately from the raw
    # minimum/maximum of every retained sensitivity observation.  The 103.8
    # value is an NGCC new-entry fixed-cost reference, not a clearing price.
    band_rows = curve.loc[
        curve["firm_value_eur_kw_year"].isin(
            [CAPACITY_PRIORITY_LOW, CAPACITY_PRIORITY_HIGH]
        )
    ]
    if band_rows.groupby("cost_year").size().to_dict() != {2025: 2, 2030: 2, 2035: 2}:
        raise RuntimeError("Panel c lacks the audited 15.6946 and 90.4316 capacity anchors")
    ngcc_rows = curve.loc[
        np.isclose(
            curve["firm_value_eur_kw_year"],
            NGCC_NEW_ENTRY_FIXED_COST,
            atol=1e-6,
        )
    ]
    if ngcc_rows.groupby("cost_year").size().to_dict() != {2025: 1, 2030: 1, 2035: 1}:
        raise RuntimeError("Panel c lacks the 103.799 NGCC fixed-cost reference")

    anchors = pd.read_csv(CAPACITY_ANCHORS_FILE)
    if "converted_eur_kw_year" not in anchors.columns:
        raise RuntimeError("Panel c audited capacity-anchor file lacks converted_eur_kw_year")
    anchors["eur_per_kw_year"] = anchors["converted_eur_kw_year"].astype(float)
    anchors["anchor_scope"] = np.where(
        anchors["jurisdiction"].eq("All-island Single Electricity Market"),
        "contextual_SEM_high",
        "four_priority_markets",
    )
    priority = anchors.loc[anchors["anchor_scope"].eq("four_priority_markets")]
    sem = anchors.loc[anchors["anchor_scope"].eq("contextual_SEM_high")]
    if not (
        np.isclose(priority["eur_per_kw_year"].min(), CAPACITY_PRIORITY_LOW, atol=1e-4)
        and np.isclose(priority["eur_per_kw_year"].max(), CAPACITY_PRIORITY_HIGH, atol=1e-4)
    ):
        raise RuntimeError("Panel c four-priority-market range is not 15.6946--90.4316")
    if len(sem) != 1 or not np.isclose(sem["eur_per_kw_year"].iloc[0], 160.577, atol=1e-4):
        raise RuntimeError("Panel c contextual SEM high anchor is not 160.577")
    anchors["within_priority_market_range"] = anchors["anchor_scope"].eq(
        "four_priority_markets"
    )
    anchors = anchors.sort_values("eur_per_kw_year").reset_index(drop=True)
    return curve, anchors


def build_panel_d() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(NATURAL_PANEL_FILE)
    if len(panel) != 132 or panel["country"].nunique() != 15:
        raise RuntimeError("Panel d natural-year panel is not 132 country-years across 15 countries")
    panel["zero_window_share_within_country_pp"] = 100.0 * (
        panel["natural_zero_recovery_window_share"]
        - panel.groupby("country")["natural_zero_recovery_window_share"].transform("mean")
    )
    panel["caes24_minus_lfp4_within_country_eur_kw_year"] = (
        panel["natural_caes24_minus_lfp4_eur_kw_year"]
        - panel.groupby("country")["natural_caes24_minus_lfp4_eur_kw_year"].transform("mean")
    )
    panel["country_year_id"] = panel["country"] + "-" + panel["year"].astype(int).astype(str)

    coef = pd.read_csv(NATURAL_COEF_FILE)
    coef = coef[
        coef["model_id"].eq("zero_window_only")
        & coef["predictor"].eq("natural_zero_recovery_window_share")
    ].copy()
    if len(coef) != 1 or int(coef.iloc[0]["country_years"]) != 132:
        raise RuntimeError("Panel d zero-window fixed-effect coefficient is missing or duplicated")
    coef["coefficient_per_10pp_eur_kw_year"] = 0.1 * coef["coefficient"]
    coef["ci95_low_per_10pp_eur_kw_year"] = 0.1 * coef["bootstrap_ci95_low"]
    coef["ci95_high_per_10pp_eur_kw_year"] = 0.1 * coef["bootstrap_ci95_high"]
    return panel, coef


def build_panel_e() -> pd.DataFrame:
    values = pd.read_csv(OPTION_VALUES_FILE)
    values = values[
        values["model_mode"].eq("fixed")
        & values["cost_year"].eq(2035)
        & ~values["arm_type"].eq("base_only")
    ].copy()
    best = (
        values.groupby(["country", "year", "chronology"], observed=True)[
            "efc_value_score_eur_kw_year"
        ]
        .max()
        .unstack("chronology")
        .reset_index()
    )
    if len(best) != 132 or not {"dispersed", "clustered"}.issubset(best.columns):
        raise RuntimeError("Panel e could not reconstruct 132 paired tail-entry margins")
    best = best.rename(
        columns={
            "dispersed": "tail_entry_margin_dispersed_eur_kw_year",
            "clustered": "tail_entry_margin_clustered_eur_kw_year",
        }
    )
    best["clustering_margin_shift_eur_kw_year"] = (
        best["tail_entry_margin_clustered_eur_kw_year"]
        - best["tail_entry_margin_dispersed_eur_kw_year"]
    )
    best["robust_tail_exit"] = (
        best["tail_entry_margin_dispersed_eur_kw_year"].gt(TIE_TOL)
        & best["tail_entry_margin_clustered_eur_kw_year"].lt(-TIE_TOL)
    )
    best["country_year_id"] = best["country"] + "-" + best["year"].astype(int).astype(str)
    if int(best["robust_tail_exit"].sum()) != 22:
        raise RuntimeError("Panel e expected 22 robust 2035 tail exits")
    return best.sort_values(["robust_tail_exit", "country", "year"]).reset_index(drop=True)


def draw_panel_a(ax: plt.Axes, composition: pd.DataFrame, flips: pd.DataFrame) -> None:
    row_order = [
        (year, chronology)
        for year in [2025, 2030, 2035]
        for chronology in ["dispersed", "clustered"]
    ]
    category_order = ["No tail", "4 h LFP", "LDES"]
    colors = {"No tail": NO_TAIL, "4 h LFP": LFP, "LDES": LDES}
    y = np.arange(len(row_order), dtype=float)
    left = np.zeros(len(row_order), dtype=float)
    for category in category_order:
        values = []
        for year, chronology in row_order:
            value = composition[
                composition["cost_year"].eq(year)
                & composition["chronology"].eq(chronology)
                & composition["selection"].eq(category)
            ]["country_year_count"].iloc[0]
            values.append(int(value))
        ax.barh(
            y,
            values,
            left=left,
            height=0.68,
            color=colors[category],
            edgecolor="white",
            linewidth=0.35,
            label=category,
        )
        left += np.asarray(values, dtype=float)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{year}  {'D' if chronology == 'dispersed' else 'C'}" for year, chronology in row_order]
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 153)
    ax.set_xticks([0, 33, 66, 99, 132])
    ax.set_xlabel("Country–years selected (n = 132)")
    ax.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        0.0,
        1.014,
        "D dispersed · C clustered",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=5.1,
    )
    ax.text(
        1.0,
        1.014,
        "robust flips",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=5.1,
    )
    for i, year in enumerate([2025, 2030, 2035]):
        row = flips[flips["cost_year"].eq(year)].iloc[0]
        count = int(row["disjoint_tie_set_flips"])
        rate = 100.0 * float(row["disjoint_tie_set_flip_rate"])
        ax.text(
            151.5,
            2 * i + 0.5,
            f"{count}\n({rate:.1f}%)",
            ha="right",
            va="center",
            fontsize=5.1,
            color=CLUSTERED if count else MUTED,
            fontweight="bold" if count else "normal",
        )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, -0.34),
        ncol=3,
        fontsize=5.1,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.8,
    )


def draw_panel_b(ax: plt.Axes, ecdf: pd.DataFrame, summary: pd.DataFrame) -> None:
    x = ecdf["absolute_cluster_effect_eur_kw_year"].to_numpy(float)
    y = ecdf["ecdf_percent"].to_numpy(float)
    row = summary.iloc[0]
    low = float(row["minimum_cost_path_span_eur_kw_year"])
    med = float(row["median_cost_path_span_eur_kw_year"])
    high = float(row["maximum_cost_path_span_eur_kw_year"])
    ax.axvspan(low, high, color=GOLD, alpha=0.26, lw=0)
    ax.axvline(med, color=GOLD, lw=1.0, ls="--")
    ax.step(x, y, where="post", color=DISPERSED, lw=1.5)
    ax.scatter(x, y, s=4.5, color=DISPERSED, alpha=0.45, edgecolor="none", rasterized=False)
    ax.set_xlim(0, max(23.0, float(x.max()) * 1.04))
    ax.set_ylim(0, 101)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Absolute clustering value shift (EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_ylabel("Cumulative share of country–years (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.annotate(
        "2025–2035 relative-cost\nmovement",
        xy=(med, 61),
        xytext=(8.2, 80),
        fontsize=5.0,
        color=INK,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": GOLD, "lw": 0.75},
    )
    n = int(row["clustering_effect_at_least_cost_path_span_count"])
    rate = 100.0 * float(row["clustering_effect_at_least_cost_path_span_rate"])
    ax.text(
        0.98,
        0.08,
        f"{n}/132 ({rate:.1f}%) exceed\nthe published cost-path movement",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=INK,
        fontweight="bold",
    )


def draw_panel_c(ax: plt.Axes, curve: pd.DataFrame, anchors: pd.DataFrame) -> None:
    low = CAPACITY_PRIORITY_LOW
    high = CAPACITY_PRIORITY_HIGH
    ax.axvspan(low, high, color=GOLD, alpha=0.13, lw=0, zorder=0)
    year_styles = {
        2025: (MUTED, "o", ":"),
        2030: (DISPERSED, "s", "--"),
        2035: (CLUSTERED, "o", "-"),
    }
    for year, (color, marker, linestyle) in year_styles.items():
        group = curve[curve["cost_year"].eq(year)].sort_values("firm_value_eur_kw_year")
        ax.plot(
            group["firm_value_eur_kw_year"],
            group["robust_flip_percent"],
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.45,
            markersize=2.5,
            linestyle=linestyle,
            linewidth=1.2,
            label=str(year),
            zorder=3,
        )
    ngcc_reference = NGCC_NEW_ENTRY_FIXED_COST
    ax.axvline(ngcc_reference, color=PURPLE, lw=0.75, ls="--", zorder=1)
    point = curve[
        curve["cost_year"].eq(2035)
        & np.isclose(curve["firm_value_eur_kw_year"], 103.799, atol=0.002)
    ].iloc[0]
    ax.scatter(
        [ngcc_reference],
        [point["robust_flip_percent"]],
        s=25,
        facecolor="white",
        edgecolor=PURPLE,
        linewidth=0.85,
        zorder=5,
    )
    ax.annotate(
        "NGCC new-entry fixed-cost\nreference 103.8 · 22/132",
        xy=(ngcc_reference, point["robust_flip_percent"]),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=PURPLE,
    )
    ax.text(
        (low + high) / 2,
        27.8,
        "four priority markets\n15.7–90.4",
        ha="center",
        va="top",
        fontsize=5.0,
        color=GOLD,
        fontweight="bold",
    )
    rug_y = -0.35
    ax.vlines(
        anchors.loc[anchors["within_priority_market_range"], "eur_per_kw_year"],
        rug_y,
        rug_y + 0.55,
        color=GOLD,
        alpha=0.48,
        lw=0.55,
        clip_on=False,
    )
    ax.set_xlim(0, 125)
    ax.set_ylim(-0.5, 29)
    ax.set_xlabel("Task-firm-supply unit value (EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_ylabel("Robust purchase flips (%)")
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=1, fontsize=5.0, handlelength=1.6)


def draw_panel_d(ax: plt.Axes, panel: pd.DataFrame, coef: pd.DataFrame) -> None:
    x = panel["zero_window_share_within_country_pp"].to_numpy(float)
    y = panel["caes24_minus_lfp4_within_country_eur_kw_year"].to_numpy(float)
    row = coef.iloc[0]
    slope = float(row["coefficient"]) / 100.0
    slope_low = float(row["bootstrap_ci95_low"]) / 100.0
    slope_high = float(row["bootstrap_ci95_high"]) / 100.0
    grid = np.linspace(float(x.min()), float(x.max()), 200)
    line = slope * grid
    lower = np.minimum(slope_low * grid, slope_high * grid)
    upper = np.maximum(slope_low * grid, slope_high * grid)
    ax.fill_between(grid, lower, upper, color=RECHARGE, alpha=0.14, lw=0)
    ax.plot(grid, line, color=RECHARGE, lw=1.45)
    ax.scatter(x, y, s=8, color=MUTED, alpha=0.48, edgecolor="white", linewidth=0.18)
    ax.axhline(0, color=INK, lw=0.55)
    ax.axvline(0, color=INK, lw=0.55)
    ax.set_xlim(-14.7, 14.2)
    ax.set_ylim(-5.1, 5.9)
    ax.set_xlabel("Within-country zero-window shift (pp)")
    ax.set_ylabel(
        "CAES24–LFP4 value residual\n(EUR kW$^{-1}$ yr$^{-1}$)",
        labelpad=1.0,
    )
    ax.text(
        0.98,
        0.96,
        (
            f"−{abs(float(row['coefficient_per_10pp_eur_kw_year'])):.2f} per 10 pp\n"
            f"95% CI {float(row['ci95_low_per_10pp_eur_kw_year']):.2f} to "
            f"{float(row['ci95_high_per_10pp_eur_kw_year']):.2f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.04,
        "country fixed effects · n = 132",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=MUTED,
    )


def draw_panel_e(ax: plt.Axes, boundary: pd.DataFrame) -> None:
    x = boundary["tail_entry_margin_dispersed_eur_kw_year"].to_numpy(float)
    y = boundary["clustering_margin_shift_eur_kw_year"].to_numpy(float)
    flip = boundary["robust_tail_exit"].to_numpy(bool)
    xline = np.linspace(0, 9.0, 150)
    ax.fill_between(xline, -16.5, -xline, color=CLUSTERED, alpha=0.10, lw=0)
    ax.plot(xline, -xline, color=CLUSTERED, lw=0.85, ls="--")
    ax.axvline(0, color=INK, lw=0.6)
    ax.axhline(0, color=INK, lw=0.6)
    ax.scatter(x[~flip], y[~flip], s=8, color="#9EAAAF", alpha=0.50, edgecolor="none")
    ax.scatter(
        x[flip],
        y[flip],
        s=13,
        color=CLUSTERED,
        alpha=0.84,
        edgecolor="white",
        linewidth=0.25,
        label="Tail exits",
    )
    at = boundary[boundary["country_year_id"].eq("AT-2024")].iloc[0]
    ax.scatter(
        [at["tail_entry_margin_dispersed_eur_kw_year"]],
        [at["clustering_margin_shift_eur_kw_year"]],
        marker="*",
        s=38,
        facecolor=GOLD,
        edgecolor=INK,
        linewidth=0.45,
        zorder=5,
    )
    ax.annotate(
        "AT 2024",
        xy=(
            at["tail_entry_margin_dispersed_eur_kw_year"],
            at["clustering_margin_shift_eur_kw_year"],
        ),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=INK,
    )
    ax.text(
        0.95,
        0.50,
        "22/132 cross\nthe no-tail boundary",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=CLUSTERED,
        fontweight="bold",
    )
    ax.text(4.8, -4.5, "exit boundary", color=CLUSTERED, fontsize=5.0, rotation=-35)
    ax.set_xlim(-9.2, 9.2)
    ax.set_ylim(-16.2, 0.8)
    ax.set_xlabel("Dispersed tail-entry margin\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_ylabel("Margin shift\n(EUR kW$^{-1}$ yr$^{-1}$)", labelpad=1.0)


def export_figure(fig: plt.Figure, stem: str) -> list[Path]:
    MAIN_OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = MAIN_OUT / f"{stem}.pdf"
    svg_path = MAIN_OUT / f"{stem}.svg"
    png_path = MAIN_OUT / f"{stem}.png"
    tiff_path = MAIN_OUT / f"{stem}.tiff"
    fig.savefig(pdf_path, bbox_inches=None)
    fig.savefig(svg_path, bbox_inches=None)
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches=None)
    fig.savefig(
        tiff_path,
        dpi=TIFF_DPI,
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    outputs = [pdf_path, svg_path, png_path, tiff_path]
    gray = MAIN_OUT / f"{stem}_grayscale.png"
    with Image.open(MAIN_OUT / f"{stem}.png") as image:
        image.convert("L").save(gray, dpi=(PNG_DPI, PNG_DPI))
    outputs.append(gray)
    return outputs


def main() -> None:
    configure()
    require_inputs()
    MAIN_OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)

    panel_a, panel_a_flips = build_panel_a()
    panel_b, panel_b_summary = build_panel_b()
    panel_c, panel_c_anchors = build_panel_c()
    panel_d, panel_d_coef = build_panel_d()
    panel_e = build_panel_e()

    source_paths = [
        save_source(panel_a, "figure6_panel_a_purchase_composition.csv"),
        save_source(panel_a_flips, "figure6_panel_a_robust_flip_summary.csv"),
        save_source(panel_b, "figure6_panel_b_value_shift_cost_path_ecdf.csv"),
        save_source(panel_b_summary, "figure6_panel_b_value_shift_cost_path_summary.csv"),
        save_source(panel_c, "figure6_panel_c_capacity_anchor_flip_curve.csv"),
        save_source(panel_c_anchors, "figure6_panel_c_capacity_market_anchors.csv"),
        save_source(panel_d, "figure6_panel_d_natural_fixed_effect_residuals.csv"),
        save_source(panel_d_coef, "figure6_panel_d_natural_fixed_effect_coefficient.csv"),
        save_source(panel_e, "figure6_panel_e_2035_tail_decision_boundary.csv"),
    ]

    fig = plt.figure(figsize=(7.2, 5.30))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.090, top=0.965)
    gs = fig.add_gridspec(
        2,
        12,
        height_ratios=[1.16, 0.92],
        hspace=0.60,
        wspace=1.45,
    )
    axa = fig.add_subplot(gs[0, 0:5])
    axb = fig.add_subplot(gs[0, 7:12])
    axc = fig.add_subplot(gs[1, 0:4])
    axd = fig.add_subplot(gs[1, 5:8])
    axe = fig.add_subplot(gs[1, 9:12])
    axes = [axa, axb, axc, axd, axe]

    draw_panel_a(axa, panel_a, panel_a_flips)
    draw_panel_b(axb, panel_b, panel_b_summary)
    draw_panel_c(axc, panel_c, panel_c_anchors)
    draw_panel_d(axd, panel_d, panel_d_coef)
    draw_panel_e(axe, panel_e)
    panel_labels(fig, axes)

    stem = "fig_v44_figure6_decision_consequences"
    output_paths = export_figure(fig, stem)
    plt.close(fig)

    registry = pd.DataFrame(
        [
            {
                "figure": "Figure 6",
                "panel": panel,
                "source_data_file": path.name,
                "rows": len(pd.read_csv(path)),
                "sha256": sha256(path),
            }
            for panel, path in zip(
                ["a", "a", "b", "b", "c", "c", "d", "d", "e"], source_paths
            )
        ]
    )
    registry_path = save_source(registry, "figure6_v44_source_data_registry.csv")
    source_paths.append(registry_path)

    contract = {
        "figure": "Figure 6",
        "core_conclusion": (
            "Event clustering changes actual tail procurement when chronology-driven value losses "
            "cross the cost-and-service decision boundary."
        ),
        "archetype": "asymmetric mixed-modality quantitative figure",
        "backend": "Python matplotlib only",
        "final_size_inches": [WIDTH, HEIGHT],
        "panel_map": {
            "a": "selected no-tail, 4 h LFP and LDES identities under 2025/2030/2035 costs",
            "b": "clustering value-shift distribution against the published 2025-2035 relative-cost movement",
            "c": "robust decision flips across the four-priority-market capacity-price range, with the NGCC new-entry fixed-cost reference shown separately",
            "d": "within-country natural-year recovery-window relationship",
            "e": "2035 tail-entry margin and the chronology-induced exit boundary",
        },
        "statistics": {
            "panel_a": "country-block bootstrap 95% interval is available in Source Data; n=132 country-years",
            "panel_d": "country fixed effects; 10,000 country-block bootstrap draws over 15 countries",
            "decision_tie_tolerance_eur_kw_year": TIE_TOL,
        },
        "input_sha256": {str(path.relative_to(V44)): sha256(path) for path in [
            CHOICES_FILE,
            FLIP_SUMMARY_FILE,
            OPTION_VALUES_FILE,
            NATURAL_PANEL_FILE,
            NATURAL_COEF_FILE,
            COST_SPAN_FILE,
            CAPACITY_CURVE_FILE,
            CAPACITY_ANCHORS_FILE,
        ]},
        "source_data_sha256": {path.name: sha256(path) for path in source_paths},
        "output_sha256": {path.name: sha256(path) for path in output_paths},
        "qa_counts": {
            "country_years": 132,
            "countries": 15,
            "robust_flips_2025_2030_2035": [0, 3, 22],
            "effect_at_least_cost_path_span": 60,
            "natural_fixed_effect_country_blocks": 15,
            "robust_2035_tail_exits": 22,
        },
    }
    manifest_path = SOURCE_OUT / "figure6_v44_build_manifest.json"
    manifest_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "outputs": [str(path) for path in output_paths]}, indent=2))


if __name__ == "__main__":
    main()
