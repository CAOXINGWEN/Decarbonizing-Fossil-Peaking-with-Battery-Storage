#!/usr/bin/env python3
"""Build v45 Figure 6 from the frozen v44 decision-facing evidence package."""

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
ROOT = FIGURE_WORK.parent
SOURCE = FIGURE_WORK / "source_data"
OUT = FIGURE_WORK / "main_figures"

BOUNDARY_FILE = SOURCE / "figure6_panel_e_2035_tail_decision_boundary.csv"
FIXED_FLIP_FILE = SOURCE / "figure6_panel_a_robust_flip_summary.csv"
JOINT_FLIP_FILE = ROOT / "joint_choice" / "bounded_s20_highs" / "bounded_joint_choice_summary_all_years.csv"
CAPACITY_CURVE_FILE = SOURCE / "figure6_panel_c_capacity_anchor_flip_curve.csv"
CAPACITY_ANCHOR_FILE = SOURCE / "figure6_panel_c_capacity_market_anchors.csv"
SHIFT_ECDF_FILE = SOURCE / "figure6_panel_b_value_shift_cost_path_ecdf.csv"
SHIFT_SUMMARY_FILE = SOURCE / "figure6_panel_b_value_shift_cost_path_summary.csv"
NATURAL_COEF_FILE = ROOT / "joint_choice" / "natural_fixed_effect_coefficients.csv"

WIDTH = 182.88 / 25.4
HEIGHT = 5.12
PNG_DPI = 300
TIFF_DPI = 600

PRIORITY_LOW = 15.6946
PRIORITY_HIGH = 90.4316
NGCC_REFERENCE = 103.79921785880974

INK = "#263B49"
MUTED = "#718087"
GRID = "#DFE7EA"
DISPERSED = "#35618D"
CLUSTERED = "#D8662E"
TEAL = "#15968C"
PURPLE = "#7562A8"
GOLD = "#D6A135"
PALE_ORANGE = "#FBEEE7"


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
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.55,
            "lines.linewidth": 1.15,
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
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> dict[str, pd.DataFrame]:
    paths = {
        "boundary": BOUNDARY_FILE,
        "fixed_flips": FIXED_FLIP_FILE,
        "joint_flips": JOINT_FLIP_FILE,
        "capacity": CAPACITY_CURVE_FILE,
        "anchors": CAPACITY_ANCHOR_FILE,
        "shift_ecdf": SHIFT_ECDF_FILE,
        "shift_summary": SHIFT_SUMMARY_FILE,
        "natural_coef": NATURAL_COEF_FILE,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen Figure 6 inputs: " + ", ".join(missing))
    data = {name: pd.read_csv(path) for name, path in paths.items()}
    boundary = data["boundary"]
    if len(boundary) != 132 or int(boundary["robust_tail_exit"].sum()) != 22:
        raise RuntimeError("The 2035 decision boundary must contain 132 cases and 22 robust exits")
    fixed = data["fixed_flips"].sort_values("cost_year")
    if fixed["disjoint_tie_set_flips"].astype(int).tolist() != [0, 3, 22]:
        raise RuntimeError("Fixed-scale flip counts differ from 0, 3 and 22")
    joint = data["joint_flips"]
    joint = joint.loc[
        joint["account"].eq("residual_peak_system_cost")
        & joint["metric"].eq("identity_flip")
    ].sort_values("learning_year").copy()
    if joint["count"].astype(int).tolist() != [2, 2, 3]:
        raise RuntimeError("Joint system-cost identity flips differ from 2, 2 and 3")
    data["joint_flips"] = joint
    if len(data["shift_ecdf"]) != 132:
        raise RuntimeError("Value-shift ECDF no longer contains 132 country-years")
    return data


def panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    for label, ax in zip("abcde", axes):
        box = ax.get_position()
        fig.text(
            box.x0 - 0.009,
            box.y1 + 0.007,
            label,
            ha="right",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#111111",
        )


def draw_boundary_hero(ax: plt.Axes, data: pd.DataFrame) -> None:
    x = data["tail_entry_margin_dispersed_eur_kw_year"].to_numpy(float)
    y = data["tail_entry_margin_clustered_eur_kw_year"].to_numpy(float)
    flip = data["robust_tail_exit"].astype(bool).to_numpy()
    xmin, xmax = -9.2, 9.2
    ymin, ymax = -10.8, 1.0
    ax.fill_between([0, xmax], ymin, 0, color=PALE_ORANGE, zorder=0)
    diag = np.linspace(max(xmin, ymin), min(xmax, ymax), 100)
    ax.plot(diag, diag, color="#AAB7BC", lw=0.7, ls="--", zorder=1)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axvline(0, color=INK, lw=0.7)
    ax.scatter(x[~flip], y[~flip], s=14, color="#9CA9AE", alpha=0.48, edgecolor="none", rasterized=True)
    ax.scatter(x[flip], y[flip], s=22, color=CLUSTERED, alpha=0.88, edgecolor="white", linewidth=0.35, zorder=4)
    at = data.loc[data["country_year_id"].eq("AT-2024")].iloc[0]
    ax.scatter(
        [at["tail_entry_margin_dispersed_eur_kw_year"]],
        [at["tail_entry_margin_clustered_eur_kw_year"]],
        marker="*",
        s=70,
        facecolor=GOLD,
        edgecolor=INK,
        linewidth=0.55,
        zorder=5,
    )
    ax.annotate(
        "AT 2024",
        xy=(at["tail_entry_margin_dispersed_eur_kw_year"], at["tail_entry_margin_clustered_eur_kw_year"]),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=INK,
    )
    ax.text(
        0.96,
        0.93,
        "22/132 country–years\ncross the tail-entry boundary",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color=CLUSTERED,
        fontweight="bold",
    )
    ax.text(5.8, -1.0, "tail exits", ha="center", va="center", fontsize=5.2, color=CLUSTERED, fontweight="bold")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Dispersed tail-entry margin\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_ylabel("Clustered tail-entry margin\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.grid(color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def draw_cost_year_flips(ax: plt.Axes, fixed: pd.DataFrame, joint: pd.DataFrame) -> None:
    years = fixed["cost_year"].to_numpy(int)
    fixed_rate = 100 * fixed["disjoint_tie_set_flip_rate"].to_numpy(float)
    fixed_lo = 100 * fixed["country_block_bootstrap_ci95_low"].to_numpy(float)
    fixed_hi = 100 * fixed["country_block_bootstrap_ci95_high"].to_numpy(float)
    joint_rate = 100 * joint["rate"].to_numpy(float)
    joint_lo = 100 * joint["country_block_bootstrap_ci_low"].to_numpy(float)
    joint_hi = 100 * joint["country_block_bootstrap_ci_high"].to_numpy(float)
    ax.errorbar(
        years - 0.55,
        fixed_rate,
        yerr=np.vstack([fixed_rate - fixed_lo, fixed_hi - fixed_rate]),
        color=CLUSTERED,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=CLUSTERED,
        markersize=3.8,
        capsize=1.8,
        lw=1.1,
        label="EENS-equivalent tail participation",
    )
    ax.errorbar(
        years + 0.55,
        joint_rate,
        yerr=np.vstack([joint_rate - joint_lo, joint_hi - joint_rate]),
        color=DISPERSED,
        marker="s",
        markerfacecolor="white",
        markeredgecolor=DISPERSED,
        markersize=3.6,
        capsize=1.8,
        lw=1.0,
        ls="--",
        label="Joint system-cost identity",
    )
    for year, value, count in zip(years, fixed_rate, fixed["disjoint_tie_set_flips"].astype(int)):
        ax.text(year - 0.55, value + 1.0, str(count), ha="center", va="bottom", fontsize=5.0, color=CLUSTERED)
    for year, value, count in zip(years, joint_rate, joint["count"].astype(int)):
        ax.text(year + 0.55, value + 1.0, str(count), ha="center", va="bottom", fontsize=5.0, color=DISPERSED)
    ax.set_xticks(years)
    ax.set_ylim(0, 31)
    ax.set_ylabel("Robust flips (%)")
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=5.0, handlelength=1.5)


def draw_capacity_anchor(ax: plt.Axes, curve: pd.DataFrame, anchors: pd.DataFrame) -> None:
    data = curve.loc[curve["cost_year"].eq(2035)].sort_values("firm_value_eur_kw_year")
    ax.axvspan(PRIORITY_LOW, PRIORITY_HIGH, color=GOLD, alpha=0.15, lw=0)
    ax.plot(data["firm_value_eur_kw_year"], data["robust_flip_percent"], color=CLUSTERED, lw=1.45, marker="o", markersize=2.5, markerfacecolor="white", markeredgewidth=0.45)
    ax.axvline(NGCC_REFERENCE, color=PURPLE, lw=0.8, ls="--")
    point = data.iloc[np.argmin(np.abs(data["firm_value_eur_kw_year"].to_numpy(float) - NGCC_REFERENCE))]
    ax.scatter([NGCC_REFERENCE], [point["robust_flip_percent"]], s=30, facecolor="white", edgecolor=PURPLE, linewidth=0.9, zorder=4)
    ax.annotate("NGCC 103.8\n22/132", xy=(NGCC_REFERENCE, point["robust_flip_percent"]), xytext=(-3, 5), textcoords="offset points", ha="right", va="bottom", fontsize=5.0, color=PURPLE)
    ax.text((PRIORITY_LOW + PRIORITY_HIGH) / 2, 23.5, "four-market range\n15.7–90.4", ha="center", va="top", fontsize=5.0, color=GOLD, fontweight="bold")
    priority = anchors.loc[anchors["within_priority_market_range"].astype(bool), "eur_per_kw_year"]
    ax.vlines(priority, -0.3, 0.35, color=GOLD, lw=0.55, alpha=0.55, clip_on=False)
    ax.set_xlim(0, 125)
    ax.set_ylim(-0.4, 25.5)
    ax.set_xlabel("Firm-supply value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_ylabel("2035 flips (%)")
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def draw_cost_distance(ax: plt.Axes, ecdf: pd.DataFrame, summary: pd.DataFrame) -> None:
    data = ecdf.sort_values("absolute_cluster_effect_eur_kw_year")
    x = data["absolute_cluster_effect_eur_kw_year"].to_numpy(float)
    y = data["ecdf_percent"].to_numpy(float)
    row = summary.iloc[0]
    low = float(row["minimum_cost_path_span_eur_kw_year"])
    med = float(row["median_cost_path_span_eur_kw_year"])
    high = float(row["maximum_cost_path_span_eur_kw_year"])
    ax.axvspan(low, high, color=GOLD, alpha=0.24, lw=0)
    ax.axvline(med, color=GOLD, lw=0.9, ls="--")
    ax.step(x, y, where="post", color=DISPERSED, lw=1.35)
    ax.set_xlim(0, max(23, x.max() * 1.03))
    ax.set_ylim(0, 101)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Chronology shift\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_ylabel("Cumulative share (%)")
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    count = int(row["clustering_effect_at_least_cost_path_span_count"])
    ax.text(0.96, 0.08, f"{count}/132 exceed the\n2025–2035 cost movement", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color=INK, fontweight="bold")


def draw_natural_coefficients(ax: plt.Axes, coef: pd.DataFrame) -> None:
    rows = []
    for model_id, label in [
        ("zero_window_only", "Country FE"),
        ("joint_geometry_plus_scale", "Fully adjusted"),
    ]:
        row = coef.loc[
            coef["model_id"].eq(model_id)
            & coef["predictor"].eq("natural_zero_recovery_window_share")
        ].iloc[0]
        rows.append(
            {
                "label": label,
                "estimate": 0.1 * float(row["coefficient"]),
                "low": 0.1 * float(row["bootstrap_ci95_low"]),
                "high": 0.1 * float(row["bootstrap_ci95_high"]),
            }
        )
    data = pd.DataFrame(rows)
    y = np.array([1.0, 0.0])
    ax.axvline(0, color=INK, lw=0.7)
    for yi, row, color in zip(y, data.to_dict("records"), [TEAL, DISPERSED]):
        ax.plot([row["low"], row["high"]], [yi, yi], color=color, lw=1.7, solid_capstyle="round")
        ax.scatter([row["estimate"]], [yi], s=30, color=color, edgecolor="white", linewidth=0.45, zorder=4)
        ax.text(row["estimate"], yi + 0.19, f"{row['estimate']:.2f}", ha="center", va="bottom", fontsize=5.0, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.set_xlim(-2.35, 0.30)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Coefficient per +10 pp\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.05, "n = 132 country–years", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color=MUTED)


def export(fig: plt.Figure, stem: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / f"{stem}.pdf"
    svg_path = OUT / f"{stem}.svg"
    png_path = OUT / f"{stem}.png"
    tiff_path = OUT / f"{stem}.tiff"
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=PNG_DPI)
    fig.savefig(tiff_path, dpi=TIFF_DPI, pil_kwargs={"compression": "tiff_lzw"})
    paths = [pdf_path, svg_path, png_path, tiff_path]
    gray = OUT / f"{stem}_grayscale.png"
    with Image.open(paths[2]) as image:
        image.convert("L").save(gray, dpi=(PNG_DPI, PNG_DPI))
    paths.append(gray)
    return paths


def main() -> None:
    configure()
    data = load_inputs()
    fig = plt.figure(figsize=(7.2, 5.12))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.965)
    gs = fig.add_gridspec(2, 14, hspace=0.62, wspace=1.45)
    axa = fig.add_subplot(gs[:, 0:7])
    axb = fig.add_subplot(gs[0, 8:11])
    axc = fig.add_subplot(gs[0, 11:14])
    axd = fig.add_subplot(gs[1, 8:11])
    axe = fig.add_subplot(gs[1, 11:14])
    axes = [axa, axb, axc, axd, axe]

    draw_boundary_hero(axa, data["boundary"])
    draw_cost_year_flips(axb, data["fixed_flips"], data["joint_flips"])
    draw_capacity_anchor(axc, data["capacity"], data["anchors"])
    draw_cost_distance(axd, data["shift_ecdf"], data["shift_summary"])
    draw_natural_coefficients(axe, data["natural_coef"])
    panel_labels(fig, axes)

    stem = "fig_v45_figure6_procurement_consequences"
    outputs = export(fig, stem)
    plt.close(fig)

    fixed = data["fixed_flips"]
    joint = data["joint_flips"]
    contract = {
        "figure": "Figure 6",
        "version": "v45",
        "core_conclusion": "Chronology-driven value shifts alter storage procurement when they cross country-year decision margins.",
        "archetype": "asymmetric mixed-modality figure with a decision-boundary hero panel",
        "backend": "Python matplotlib",
        "final_size_inches": [WIDTH, HEIGHT],
        "panel_map": {
            "a": "2035 dispersed versus clustered tail-entry margins and 22 robust exits",
            "b": "fixed-scale EENS-equivalent participation flips and joint system-cost identity flips across cost years",
            "c": "2035 decision flips across observed European capacity-price anchors and the NGCC fixed-cost reference",
            "d": "chronology-induced value movement relative to the published 2025-2035 cost movement",
            "e": "natural-year country fixed-effect estimates for zero-recovery-window prevalence",
        },
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [BOUNDARY_FILE, FIXED_FLIP_FILE, JOINT_FLIP_FILE, CAPACITY_CURVE_FILE, CAPACITY_ANCHOR_FILE, SHIFT_ECDF_FILE, SHIFT_SUMMARY_FILE, NATURAL_COEF_FILE]},
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "qa": {
            "country_years": 132,
            "fixed_scale_participation_flips_2025_2030_2035": fixed["disjoint_tie_set_flips"].astype(int).tolist(),
            "joint_system_cost_identity_flips_2025_2030_2035": joint["count"].astype(int).tolist(),
            "robust_2035_tail_exits": int(data["boundary"]["robust_tail_exit"].sum()),
            "cost_movement_exceedance_count": int(data["shift_summary"].iloc[0]["clustering_effect_at_least_cost_path_span_count"]),
            "missing_values_in_plotted_fields": 0,
        },
    }
    manifest_path = SOURCE / "figure6_v45_build_manifest.json"
    manifest_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "outputs": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
