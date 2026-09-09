#!/usr/bin/env python3
"""Plot Supplementary Figure 7 for the recovery-window model comparison."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
SOURCE = HERE / "source_data"
FIGURES = HERE / "figures"
ASSET = ROOT / "figure_work" / "submission_layout_v65" / "tight_assets"

INK = "#222222"
MUTED = "#777777"
PALE = "#D9DEE5"
BLUE = "#3676A8"
MAGENTA = "#A33C73"
TEAL = "#2E8074"

MODEL_LABELS = {
    "longest_event": "Longest event",
    "recovery_window": "Recovery-window closure",
    "joint_geometry": "Joint geometry",
    "joint_controlled": "Joint + annual controls",
}
MODEL_ORDER = list(MODEL_LABELS)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.2,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, fontweight="bold")


def draw(color: bool, destination: Path) -> None:
    comparison = pd.read_csv(RESULTS / "model_comparison.csv").set_index("model_id").loc[MODEL_ORDER]
    coefficients = pd.read_csv(RESULTS / "coefficient_summary.csv")
    predictions = pd.read_csv(SOURCE / "loco_predictions.csv")
    joint_predictions = predictions.loc[predictions.model_id.eq("joint_geometry")].copy()

    signal = BLUE if color else "#3B3B3B"
    accent = MAGENTA if color else "#8A8A8A"
    secondary = TEAL if color else "#666666"

    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.22, 1.0], height_ratios=[1.0, 1.0], hspace=0.52, wspace=0.46)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    # a | Model fit and held-out prediction.
    y = np.arange(len(MODEL_ORDER))[::-1]
    for yi, model_id in zip(y, MODEL_ORDER):
        row = comparison.loc[model_id]
        ax_a.plot([row.loco_r2, row.within_r2], [yi, yi], color=PALE if color else "#C9C9C9", lw=1.6, zorder=1)
        ax_a.errorbar(
            row.loco_r2,
            yi,
            xerr=np.array([[row.loco_r2 - row.loco_r2_ci95_low], [row.loco_r2_ci95_high - row.loco_r2]]),
            fmt="s",
            ms=4.1,
            mfc="white",
            mec=accent,
            mew=0.85,
            ecolor=accent,
            elinewidth=0.75,
            capsize=1.8,
            zorder=3,
        )
        ax_a.plot(row.within_r2, yi, "o", ms=4.4, color=signal, zorder=4)
        ax_a.text(row.within_r2 + 0.007, yi + 0.11, f"{row.within_r2:.3f}", color=signal, fontsize=5.7, va="bottom")
        label_offset = -0.07 if yi == 0 else -0.13
        ax_a.text(row.loco_r2 + 0.007, yi + label_offset, f"{row.loco_r2:.3f}", color=accent, fontsize=5.7, va="top")
    ax_a.axvline(0, color=MUTED, lw=0.65, ls=(0, (2, 2)), zorder=0)
    ax_a.set_yticks(y, [MODEL_LABELS[key] for key in MODEL_ORDER])
    ax_a.set_xlim(-0.065, 0.225)
    ax_a.set_xticks([-0.05, 0.00, 0.05, 0.10, 0.15, 0.20])
    ax_a.set_xlabel(r"Explained variance, $R^2$")
    ax_a.set_title("Recovery-window closure adds annual value information", loc="left", fontweight="bold", pad=7)
    ax_a.plot([], [], "o", color=signal, ms=4.4, label=r"Within-country $R^2$")
    ax_a.plot([], [], "s", mfc="white", mec=accent, mew=0.85, ms=4.1, label=r"Leave-one-country-out $R^2$ (95% CI)")
    ax_a.legend(loc="center left", bbox_to_anchor=(0.02, 0.50), handletextpad=0.5, borderaxespad=0.2)
    panel_label(ax_a, "a")

    # b | Recovery-window coefficient across specifications.
    coefficient_models = ["recovery_window", "joint_geometry", "joint_controlled"]
    coefficient_labels = ["Recovery window", "Joint geometry", "Joint + controls"]
    sub = coefficients.loc[
        coefficients.model_id.isin(coefficient_models)
        & coefficients.predictor.eq("natural_zero_recovery_window_share")
    ].set_index("model_id").loc[coefficient_models]
    yb = np.arange(len(sub))[::-1]
    estimate = sub.coefficient.to_numpy(float) / 10.0
    low = sub.bootstrap_ci95_low.to_numpy(float) / 10.0
    high = sub.bootstrap_ci95_high.to_numpy(float) / 10.0
    ax_b.errorbar(
        estimate,
        yb,
        xerr=np.vstack([estimate - low, high - estimate]),
        fmt="o",
        color=secondary,
        ecolor=secondary,
        ms=4.2,
        elinewidth=1.0,
        capsize=2.0,
    )
    for xval, yi in zip(estimate, yb):
        ax_b.text(xval - 0.035, yi + 0.17, f"{xval:.2f}", color=secondary, fontsize=5.8, ha="right")
    ax_b.axvline(0, color=MUTED, lw=0.65, ls=(0, (2, 2)))
    ax_b.set_yticks(yb, coefficient_labels)
    ax_b.set_xlim(-2.35, 0.15)
    ax_b.set_xlabel("Value change per 10-pp increase\nin zero-window share\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax_b.set_title("Recovery-window coefficient", loc="left", fontweight="bold", pad=7)
    panel_label(ax_b, "b")

    # c | Observation-level held-out predictions.
    x = joint_predictions.observed_country_centred_value_eur_kw_year.to_numpy(float)
    yhat = joint_predictions.predicted_country_centred_value_eur_kw_year.to_numpy(float)
    limit = 5.8
    ax_c.plot([-limit, limit], [-limit, limit], color=MUTED, lw=0.75, ls=(0, (3, 2)), zorder=0)
    ax_c.axhline(0, color=PALE if color else "#D0D0D0", lw=0.55, zorder=0)
    ax_c.axvline(0, color=PALE if color else "#D0D0D0", lw=0.55, zorder=0)
    ax_c.scatter(x, yhat, s=12, color=signal, alpha=0.62, edgecolors="white", linewidths=0.25, rasterized=True)
    ax_c.set_xlim(-limit, limit)
    ax_c.set_ylim(-limit, limit)
    ax_c.set_xticks([-5, 0, 5])
    ax_c.set_yticks([-5, 0, 5])
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_xlabel("Observed annual value deviation\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax_c.set_ylabel("Held-out prediction\n(EUR kW$^{-1}$ yr$^{-1}$)")
    r2 = float(comparison.loc["joint_geometry", "loco_r2"])
    corr = float(comparison.loc["joint_geometry", "loco_pearson_r"])
    ax_c.text(0.04, 0.94, f"$R^2$ = {r2:.3f}\n$r$ = {corr:.3f}\n$n$ = {len(joint_predictions)}", transform=ax_c.transAxes, ha="left", va="top", fontsize=6.1)
    ax_c.set_title("Joint-geometry held-out predictions", loc="left", fontweight="bold", pad=7)
    panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.14, right=0.985, top=0.93, bottom=0.14)
    suffix = destination.suffix.lower()
    dpi = 600 if suffix == ".tiff" else 300 if suffix == ".png" else None
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    ASSET.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "fig_v65_supplementary_figure7_recovery_window_incremental"
    draw(True, stem.with_suffix(".svg"))
    draw(True, stem.with_suffix(".pdf"))
    draw(True, stem.with_suffix(".png"))
    draw(True, stem.with_suffix(".tiff"))
    draw(False, FIGURES / f"{stem.name}_grayscale.png")
    shutil.copy2(stem.with_suffix(".pdf"), ASSET / "supp_figure7.pdf")

    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    coefficients = pd.read_csv(RESULTS / "coefficient_summary.csv")
    predictions = pd.read_csv(SOURCE / "loco_predictions.csv")
    qa = {
        "status": "pass",
        "backend": "python-matplotlib",
        "source_country_years": 132,
        "source_countries": 15,
        "models": 4,
        "coefficient_rows": len(coefficients),
        "prediction_rows": len(predictions),
        "expected_prediction_rows": 132 * 4,
        "excluded_observations": 0,
        "all_models_present": set(comparison.model_id) == set(MODEL_ORDER),
        "all_predictions_present": len(predictions) == 132 * 4,
        "exports": ["svg", "pdf", "png", "tiff", "grayscale_png"],
    }
    if not all([qa["all_models_present"], qa["all_predictions_present"]]):
        raise RuntimeError(f"figure QA failed: {qa}")
    (FIGURES / "figure_qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
