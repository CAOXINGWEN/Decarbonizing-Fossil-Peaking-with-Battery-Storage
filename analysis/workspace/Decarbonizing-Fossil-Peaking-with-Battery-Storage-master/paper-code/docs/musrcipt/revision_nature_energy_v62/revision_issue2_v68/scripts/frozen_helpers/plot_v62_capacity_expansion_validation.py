#!/usr/bin/env python3
"""Plot Supplementary Figure 5 from the frozen capacity-expansion outputs."""

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
STEM = "fig_v62_supplementary_figure5_capacity_expansion"

TECHNOLOGIES = ["li_lfp_4h", "caes_24h", "h2_cavern_168h"]
TECHNOLOGY_LABELS = {
    "li_lfp_4h": "4 h LFP",
    "caes_24h": "24 h A-CAES",
    "h2_cavern_168h": "168 h cavern H$_2$",
}
TECHNOLOGY_COLORS = {
    "li_lfp_4h": "#0072B2",
    "caes_24h": "#E69F00",
    "h2_cavern_168h": "#009E73",
}
EXPECTED_COLOR = "#3C5488"
RISK_COLOR = "#B24745"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.7,
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


def prepare_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    capacities = pd.read_csv(RESULTS / "capacity_mix.csv")
    summaries = pd.read_csv(RESULTS / "design_summary.csv")
    comparisons = pd.read_csv(RESULTS / "diversification_comparison.csv")
    shifts = pd.read_csv(RESULTS / "procurement_design_shifts.csv")

    panel_a = capacities.loc[capacities["mode"].eq("diversified")].copy()
    panel_a["design"] = np.where(
        panel_a["risk_weight"].eq(0.0), "Expected cost", "Mean + CVaR90"
    )
    panel_a["capacity_share_percent"] = panel_a["capacity_share"] * 100.0
    panel_a = panel_a[
        [
            "country",
            "risk_weight",
            "design",
            "config_id",
            "tech_id",
            "duration_h",
            "capacity_per_normalized_mw",
            "capacity_share_percent",
        ]
    ].sort_values(["country", "risk_weight", "duration_h"])

    panel_b = shifts[
        [
            "country",
            "expected_total_power_percent_mean_gas",
            "risk_aware_total_power_percent_mean_gas",
            "risk_aware_minus_expected_power_percentage_points",
            "technology_set_change",
            "dominant_technology_change",
            "material_power_change",
        ]
    ].copy()

    panel_c_rows: list[dict[str, object]] = []
    for row in comparisons.itertuples(index=False):
        group = summaries.loc[
            summaries["country"].eq(row.country)
            & summaries["risk_weight"].eq(row.risk_weight)
        ]
        best = group.loc[group["mode"].eq(row.best_single_mode)].iloc[0]
        if float(row.risk_weight) == 0.0:
            endpoint = "Expected annual cost"
            delta = float(
                row.diversified_minus_best_single_mean_cost_eur_per_normalized_mw
            )
            denominator = float(best["mean_total_cost_eur_per_normalized_mw"])
        else:
            endpoint = "CVaR90 annual cost"
            delta = float(
                row.diversified_minus_best_single_cvar90_cost_eur_per_normalized_mw
            )
            denominator = float(best["cvar90_total_cost_eur_per_normalized_mw"])
        panel_c_rows.append(
            {
                "country": row.country,
                "risk_weight": float(row.risk_weight),
                "endpoint": endpoint,
                "best_single_mode": row.best_single_mode,
                "cost_change_eur_per_normalized_mw": delta,
                "best_single_cost_eur_per_normalized_mw": denominator,
                "cost_change_percent_best_single": 100.0 * delta / denominator,
            }
        )
    panel_c = pd.DataFrame(panel_c_rows)

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    panel_a.to_csv(SOURCE_DATA / "panel_a_capacity_composition.csv", index=False)
    panel_b.to_csv(SOURCE_DATA / "panel_b_capacity_shift.csv", index=False)
    panel_c.to_csv(SOURCE_DATA / "panel_c_diversification_value.csv", index=False)
    return panel_a, panel_b, panel_c


def plot() -> dict[str, object]:
    configure_style()
    panel_a, panel_b, panel_c = prepare_source_data()
    countries = sorted(panel_a["country"].unique())
    y = np.arange(len(countries), dtype=float)
    lookup = {country: index for index, country in enumerate(countries)}

    figure = plt.figure(figsize=(7.2, 7.6), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[1.55, 1.0, 1.25],
        left=0.075,
        right=0.985,
        bottom=0.175,
        top=0.835,
        wspace=0.27,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    ax_a, ax_b, ax_c = axes

    bar_height = 0.27
    offsets = {0.0: -0.17, 0.5: 0.17}
    alphas = {0.0: 0.58, 0.5: 1.0}
    for risk_weight in (0.0, 0.5):
        left = np.zeros(len(countries))
        for config_id in TECHNOLOGIES:
            values = (
                panel_a.loc[
                    panel_a["risk_weight"].eq(risk_weight)
                    & panel_a["config_id"].eq(config_id)
                ]
                .set_index("country")
                .reindex(countries)["capacity_share_percent"]
                .fillna(0.0)
                .to_numpy()
            )
            ax_a.barh(
                y + offsets[risk_weight],
                values,
                left=left,
                height=bar_height,
                color=TECHNOLOGY_COLORS[config_id],
                alpha=alphas[risk_weight],
                edgecolor="white",
                linewidth=0.25,
                label=(
                    TECHNOLOGY_LABELS[config_id]
                    if risk_weight == 0.5
                    else None
                ),
            )
            left += values
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(countries)
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Endogenous power-capacity share (%)")
    ax_a.set_title("Procurement composition", loc="left", pad=7)
    ax_a.text(
        0.0,
        1.002,
        "pale: expected cost    saturated: mean + CVaR90",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color="#444444",
    )
    ax_a.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.125),
        ncol=3,
        handlelength=1.1,
        columnspacing=0.9,
        borderaxespad=0,
    )

    ordered_b = panel_b.set_index("country").reindex(countries)
    change = ordered_b[
        "risk_aware_minus_expected_power_percentage_points"
    ].to_numpy(dtype=float)
    ax_b.hlines(y, 0, change, color="#A7A7A7", linewidth=0.7, zorder=1)
    ax_b.scatter(
        change,
        y,
        c=np.where(change >= 0, RISK_COLOR, EXPECTED_COLOR),
        s=18,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    ax_b.axvline(0, color="#333333", linewidth=0.7)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([])
    ax_b.set_xlabel("CVaR design $-$ expected design\n(power percentage points)")
    ax_b.set_title("Procured-power response", loc="left", pad=7)

    marker_contract = {
        "Expected annual cost": ("o", EXPECTED_COLOR, -0.12),
        "CVaR90 annual cost": ("D", RISK_COLOR, 0.12),
    }
    for endpoint, (marker, color, offset) in marker_contract.items():
        subset = panel_c.loc[panel_c["endpoint"].eq(endpoint)].set_index("country")
        values = subset.reindex(countries)[
            "cost_change_percent_best_single"
        ].to_numpy(dtype=float)
        ax_c.scatter(
            values,
            y + offset,
            marker=marker,
            color=color,
            s=20,
            edgecolor="white",
            linewidth=0.35,
            label=endpoint,
            zorder=3,
        )
        median = float(np.median(values))
        ax_c.axvline(median, color=color, linewidth=0.9, linestyle=(0, (3, 2)))
    ax_c.axvline(0, color="#333333", linewidth=0.7)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([])
    ax_c.set_xlabel("Diversified $-$ best single technology\n(% of single-technology cost)")
    ax_c.set_title("Value of the technology menu", loc="left", pad=7)
    ax_c.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.125),
        borderaxespad=0,
        handletextpad=0.4,
    )

    for axis in axes:
        axis.invert_yaxis()
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="x", color="#E3E3E3", linewidth=0.45, zorder=0)
        axis.tick_params(axis="y", length=0)
    for letter, axis in zip("abc", axes):
        axis.text(
            -0.15,
            1.16,
            letter,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    figure.text(
        0.075,
        0.025,
        "Negative values in c denote savings from access to the three-technology menu; dashed lines show country medians.",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#444444",
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

    image = Image.open(outputs["png"]).convert("L")
    grayscale = FIGURES / f"{STEM}_grayscale.png"
    image.save(grayscale)
    qa = {
        "status": "pass",
        "figure": STEM,
        "country_count": len(countries),
        "panel_a_row_count": len(panel_a),
        "panel_b_row_count": len(panel_b),
        "panel_c_row_count": len(panel_c),
        "maximum_capacity_share_sum_error_percentage_points": float(
            panel_a.groupby(["country", "risk_weight"])[
                "capacity_share_percent"
            ].sum().sub(100.0).abs().max()
        ),
        "all_panel_b_values_finite": bool(
            np.isfinite(
                panel_b["risk_aware_minus_expected_power_percentage_points"]
            ).all()
        ),
        "all_panel_c_values_finite": bool(
            np.isfinite(panel_c["cost_change_percent_best_single"]).all()
        ),
        "minimum_font_size_pt": 6.3,
        "outputs": {
            key: {"path": str(path), "bytes": path.stat().st_size}
            for key, path in {**outputs, "grayscale": grayscale}.items()
        },
    }
    if (
        qa["maximum_capacity_share_sum_error_percentage_points"] > 1.0e-6
        or not qa["all_panel_b_values_finite"]
        or not qa["all_panel_c_values_finite"]
        or any(item["bytes"] <= 0 for item in qa["outputs"].values())
    ):
        qa["status"] = "fail"
    (HERE / "figure_preflight.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return qa


if __name__ == "__main__":
    report = plot()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 2)
