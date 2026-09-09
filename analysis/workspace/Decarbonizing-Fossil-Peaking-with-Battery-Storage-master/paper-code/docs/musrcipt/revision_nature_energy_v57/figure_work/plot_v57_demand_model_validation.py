#!/usr/bin/env python3
"""Build v57 Extended Data Figure 1 for demand reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
V57 = ROOT.parent
PROCESSED = V57 / "weather_year_parity_v42" / "processed"
OUT = ROOT / "extended_data_figures"
SOURCE = ROOT / "source_data_v57"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
INK = "#20333D"
MUTED = "#66777E"
GRID = "#D7DFE2"
PALE = "#F2F5F5"
BLUE = "#2678A5"
TEAL = "#138A7E"
CORAL = "#D8643C"
PURPLE = "#6E63A5"
GOLD = "#C5972C"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.3,
    "axes.labelsize": 6.5,
    "axes.titlesize": 6.7,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def label(ax: plt.Axes, text: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.3, fontweight="bold", ha="left", va="bottom")


def grid(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.45, alpha=0.85)
    ax.set_axisbelow(True)


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    rgba = plt.imread(stem.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(
        stem.with_name(stem.name + "_grayscale").with_suffix(".png"),
        grey,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def require_inputs() -> dict[str, Path]:
    files = {
        "fit": PROCESSED / "demand_model_holdout_country_year_fit.csv",
        "sample": PROCESSED / "demand_model_holdout_sample.csv.gz",
        "surface": PROCESSED / "demand_model_holdout_month_hour_surface.csv",
        "qa": PROCESSED / "demand_model_qa.json",
        "panel": PROCESSED / "pecd_v42_europe33_hourly_1950_2025.parquet",
        "panel_qa": PROCESSED / "pecd_v42_europe33_hourly_qa.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Run build_pecd_v42_hourly_panel.py first:\n" + "\n".join(missing))
    return files


def plot_a(ax: plt.Axes, fit: pd.DataFrame) -> None:
    summary = fit.groupby("country", as_index=False).agg(
        weighted_r2=("weighted_r2", "mean"),
        weighted_pearson_r=("weighted_pearson_r", "mean"),
        weighted_rmse_pu=("weighted_rmse_pu", "mean"),
    ).sort_values("weighted_r2")
    y = np.arange(len(summary))
    ax.hlines(y, summary.weighted_r2, summary.weighted_pearson_r, color=GRID, lw=1.0)
    ax.scatter(summary.weighted_r2, y, marker="s", s=15, color=BLUE, edgecolor="white", linewidth=0.35, label="$R^2$")
    ax.scatter(summary.weighted_pearson_r, y, marker="o", s=16, color=CORAL, edgecolor="white", linewidth=0.35, label="Pearson $r$")
    ax.axvline(0.45, color=PURPLE, lw=0.7, ls=(0, (3, 2)))
    ax.axvline(0.70, color=PURPLE, lw=0.7, ls=(0, (1.5, 2.5)))
    ax.set_yticks(y, summary.country)
    ax.set_xlim(min(0.35, float(summary.weighted_r2.min()) - 0.03), 1.01)
    ax.set_xlabel("country temporal-validation fit")
    grid(ax)
    ax.legend(loc="lower right", frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.35)
    label(ax, "a", x=-0.19)
    summary.to_csv(SOURCE / "extended_data_figure1a_country_holdout_fit.csv", index=False)


def plot_b(ax: plt.Axes, sample: pd.DataFrame, qa: dict[str, object]) -> None:
    x = sample.load_pu.to_numpy(float)
    y = sample.predicted_load_pu.to_numpy(float)
    low = max(0.0, float(min(x.min(), y.min())))
    high = float(max(x.max(), y.max()))
    ax.hexbin(x, y, gridsize=65, mincnt=1, bins="log", cmap="Blues", linewidths=0)
    ax.plot([low, high], [low, high], color=CORAL, lw=0.9, ls=(0, (3, 2)))
    holdout = qa["holdout"]
    assert isinstance(holdout, dict)
    ax.text(
        0.04, 0.96,
        f"$R^2$ = {float(holdout['weighted_r2']):.2f}\n$r$ = {float(holdout['weighted_pearson_r']):.2f}\nRMSE = {float(holdout['weighted_rmse_pu']):.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=5.7, color=INK,
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("observed normalized load")
    ax.set_ylabel("predicted normalized load")
    label(ax, "b", x=-0.15)
    sample[["timestamp_utc", "country", "year", "load_pu", "predicted_load_pu", "error_pu"]].to_csv(
        SOURCE / "extended_data_figure1b_holdout_sample.csv.gz", index=False, compression="gzip"
    )


def plot_c(axes: tuple[plt.Axes, plt.Axes], surface: pd.DataFrame) -> None:
    observed = surface.pivot(index="month", columns="hour", values="observed_load_pu").sort_index()
    predicted = surface.pivot(index="month", columns="hour", values="predicted_load_pu").sort_index()
    vmin = float(min(observed.min().min(), predicted.min().min()))
    vmax = float(max(observed.max().max(), predicted.max().max()))
    images = []
    for ax, matrix, title in zip(axes, [observed, predicted], ["Observed", "Predicted"]):
        image = ax.imshow(matrix, origin="upper", aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax, interpolation="nearest")
        images.append(image)
        ax.set_title(title, pad=2)
        ax.set_xticks([0, 6, 12, 18, 23], ["00", "06", "12", "18", "23"])
        ax.set_yticks([0, 2, 5, 8, 11], ["Jan", "Mar", "Jun", "Sep", "Dec"])
        ax.set_xlabel("hour (UTC)")
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel("month")
    cbar = axes[1].figure.colorbar(images[-1], ax=list(axes), fraction=0.038, pad=0.025)
    cbar.set_label("normalized load", fontsize=5.6)
    cbar.ax.tick_params(labelsize=5.2, length=2.2, width=0.5)
    label(axes[0], "c", x=-0.22)
    surface.to_csv(SOURCE / "extended_data_figure1c_month_hour_surface.csv", index=False)


def plot_d(ax: plt.Axes, sample: pd.DataFrame) -> None:
    summary = sample.groupby("hour").error_pu.agg(
        median="median",
        q10=lambda values: np.quantile(values, 0.10),
        q90=lambda values: np.quantile(values, 0.90),
    ).reset_index()
    ax.fill_between(summary.hour, summary.q10, summary.q90, color=mpl.colors.to_rgba(BLUE, 0.18), linewidth=0)
    ax.plot(summary.hour, summary["median"], color=BLUE, lw=1.0, marker="o", ms=2.1)
    ax.axhline(0, color=MUTED, lw=0.65)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("hour (UTC)")
    ax.set_ylabel("prediction error (p.u.)")
    grid(ax, axis="y")
    label(ax, "d", x=-0.18)
    summary.to_csv(SOURCE / "extended_data_figure1d_hourly_error.csv", index=False)


def plot_e(ax: plt.Axes, panel: pd.DataFrame) -> None:
    panel["timestamp_utc"] = pd.to_datetime(panel.timestamp_utc, utc=True)
    panel["date"] = panel.timestamp_utc.dt.floor("D")
    annual = panel.groupby("calendar_year", as_index=False).demand_mw.agg(
        mean="mean",
        p05=lambda values: np.quantile(values, 0.05),
        p95=lambda values: np.quantile(values, 0.95),
    )
    daily = panel.groupby(["calendar_year", "date"], as_index=False).demand_mw.mean()
    daily_peak = daily.groupby("calendar_year").demand_mw.max().rename("maximum_daily_mean").reset_index()
    annual = annual.merge(daily_peak, on="calendar_year", validate="one_to_one")
    annual["p95_p05_width_pu"] = (annual.p95 - annual.p05) / annual["mean"]
    annual["maximum_daily_mean_pu"] = annual.maximum_daily_mean / annual["mean"]
    ax.plot(annual.calendar_year, annual.p95_p05_width_pu, color=TEAL, lw=1.0, marker="o", ms=1.8, label="P95-P05 width")
    ax.plot(annual.calendar_year, annual.maximum_daily_mean_pu, color=GOLD, lw=0.9, marker="s", ms=1.7, label="maximum daily mean")
    ax.set_xlim(1949, 2026)
    ax.set_xticks([1950, 1975, 2000, 2025])
    ax.set_xlabel("calendar year")
    ax.set_ylabel("normalized load-shape\nmetric")
    grid(ax, axis="y")
    ax.legend(loc="upper right", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.5)
    label(ax, "e", x=-0.09)
    annual.to_csv(SOURCE / "extended_data_figure1e_reconstructed_load_chronology.csv", index=False)


def plot_f(ax: plt.Axes, qa: dict[str, object], panel_qa: dict[str, object]) -> None:
    ax.set_axis_off()
    rows = [
        ("Climate", "PECD v4.2, ERA5"),
        ("Weather years", "1950-2025; 75 windows"),
        ("Wind / solar", "v4.2 CF; ERAA 2025 weights"),
        ("Demand model", "observed load, 2015-2024"),
        ("Validation years", "2023-2024"),
        ("Temperature", "population-weighted; trailing 24 h"),
        ("Normalization", "country-year mean 1; no leap day"),
        ("Residual", "load - annual-matched VRE"),
    ]
    y_top = 0.96
    row_height = 0.100
    ax.add_patch(Rectangle((0, y_top - row_height + 0.008), 1, row_height, transform=ax.transAxes, facecolor=INK, edgecolor="none"))
    ax.text(0.02, y_top - 0.5 * row_height + 0.008, "Input and reconstruction audit", transform=ax.transAxes, color="white", fontweight="bold", fontsize=6.3, ha="left", va="center")
    for index, (field, value) in enumerate(rows):
        y = y_top - (index + 1.5) * row_height
        if index % 2 == 0:
            ax.add_patch(Rectangle((0, y - 0.5 * row_height), 1, row_height, transform=ax.transAxes, facecolor=PALE, edgecolor="none"))
        ax.text(0.02, y, field, transform=ax.transAxes, color=MUTED, fontsize=5.1, ha="left", va="center")
        ax.text(0.36, y, value, transform=ax.transAxes, color=INK, fontsize=5.1, ha="left", va="center")
    ax.text(0.02, 0.012, f"QA: demand {qa['status']}  |  panel {panel_qa['status']}", transform=ax.transAxes, color=TEAL, fontsize=5.4, fontweight="bold", ha="left", va="bottom")
    label(ax, "f", x=-0.10, y=1.02)
    pd.DataFrame(rows, columns=["field", "definition"]).to_csv(SOURCE / "extended_data_figure1f_provenance.csv", index=False)


def main() -> None:
    files = require_inputs()
    fit = pd.read_csv(files["fit"])
    sample = pd.read_csv(files["sample"])
    sample["timestamp_utc"] = pd.to_datetime(sample.timestamp_utc, utc=True)
    surface = pd.read_csv(files["surface"])
    panel = pd.read_parquet(files["panel"], columns=["timestamp_utc", "calendar_year", "demand_mw"])
    qa = json.loads(files["qa"].read_text(encoding="utf-8"))
    panel_qa = json.loads(files["panel_qa"].read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(7.20, 6.06))
    outer = fig.add_gridspec(
        3, 12, left=0.07, right=0.985, bottom=0.07, top=0.95,
        hspace=0.64, wspace=1.15, height_ratios=[1.28, 0.92, 0.92],
    )
    ax_a = fig.add_subplot(outer[0, :5])
    ax_b = fig.add_subplot(outer[0, 6:])
    cgrid = outer[1, :7].subgridspec(1, 2, wspace=0.18)
    ax_c1 = fig.add_subplot(cgrid[0, 0])
    ax_c2 = fig.add_subplot(cgrid[0, 1])
    ax_d = fig.add_subplot(outer[1, 8:])
    ax_e = fig.add_subplot(outer[2, :8])
    ax_f = fig.add_subplot(outer[2, 8:])

    plot_a(ax_a, fit)
    plot_b(ax_b, sample, qa)
    plot_c((ax_c1, ax_c2), surface)
    plot_d(ax_d, sample)
    plot_e(ax_e, panel)
    plot_f(ax_f, qa, panel_qa)

    stem = OUT / "ed_fig_v57_demand_model_validation"
    save_bundle(fig, stem)
    plt.close(fig)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
