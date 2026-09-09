#!/usr/bin/env python3
"""Build v45 Figure 4 from the frozen v44 mechanism source data.

The redesign keeps every observation and every registered estimate while giving
the country-year retention distribution the visual lead.  The surplus-quantity
intervention is shown as a difference-from-reference forest plot rather than a
line chart, because the four estimates are discrete counterfactual contrasts.
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
ROOT = FIGURE_WORK.parent
SOURCE = FIGURE_WORK / "source_data"
OUT = FIGURE_WORK / "main_figures"

RETENTION_FILE = SOURCE / "figure4_panel_a_tail_retention.csv"
BINDING_FILE = SOURCE / "figure4_panel_b_binding_hours.csv"
DID_FILE = SOURCE / "figure4_panel_c_recharge_difference_in_differences.csv"
LINK_FILE = SOURCE / "figure4_panel_d_physical_economic_link.csv"

WIDTH = 182.88 / 25.4
HEIGHT = 4.82
PNG_DPI = 300
TIFF_DPI = 600

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
            "font.size": 6.5,
            "axes.labelsize": 6.7,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.3,
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [RETENTION_FILE, BINDING_FILE, DID_FILE, LINK_FILE]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen Figure 4 inputs: " + ", ".join(missing))
    retention = pd.read_csv(RETENTION_FILE)
    binding = pd.read_csv(BINDING_FILE)
    did = pd.read_csv(DID_FILE)
    link = pd.read_csv(LINK_FILE)
    if len(retention) != 396 or retention["metric"].value_counts().to_dict() != {
        "Charging": 132,
        "Task service": 132,
        "Equivalent cycles": 132,
    }:
        raise RuntimeError("Retention panel no longer contains 132 observations for all three metrics")
    if set(binding["_chronology"]) != {"dispersed", "clustered"}:
        raise RuntimeError("Binding-hour source lacks the paired chronologies")
    if len(did) != 4 or set(did["recharge_quantity_multiplier"]) != {2, 4}:
        raise RuntimeError("Surplus intervention source is not the frozen 2x/4x contrast")
    if len(link) != 132 or link[["country", "year"]].duplicated().any():
        raise RuntimeError("Physical-economic link source is not one row per country-year")
    return retention, binding, did, link


def panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    for label, ax in zip("abcd", axes):
        box = ax.get_position()
        fig.text(
            box.x0 - 0.010,
            box.y1 + 0.008,
            label,
            ha="right",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#111111",
        )


def draw_retention(ax: plt.Axes, data: pd.DataFrame) -> None:
    order = ["Charging", "Task service", "Equivalent cycles"]
    colors = [TEAL, CLUSTERED, PURPLE]
    positions = np.arange(len(order))[::-1]
    arrays = [data.loc[data["metric"].eq(metric), "clustered_to_dispersed_pct"].to_numpy() for metric in order]
    violins = ax.violinplot(
        arrays,
        positions=positions,
        vert=False,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.24,
    )
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(0.8)

    for pos, values, color in zip(positions, arrays, colors):
        jitter = 0.22 * np.sin(np.arange(len(values)) * 2.399963229728653)
        ax.scatter(values, pos + jitter, s=8, color=color, alpha=0.32, edgecolor="none", rasterized=True)
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([q25, q75], [pos, pos], color=INK, lw=2.1, solid_capstyle="round", zorder=5)
        ax.scatter([median], [pos], s=27, facecolor="white", edgecolor=INK, linewidth=0.7, zorder=6)
        ax.text(q75 + 1.2, pos, f"{median:.1f}%", ha="left", va="center", fontsize=5.4, color=INK)

    ax.axvline(100, color=INK, lw=0.8, ls="--")
    ax.text(99.2, 2.42, "no loss", ha="right", va="bottom", fontsize=5.1, color=MUTED)
    ax.set_xlim(40, 103)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(positions)
    ax.set_yticklabels(order)
    ax.set_xlabel("Clustered / dispersed retention (%)")
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.02,
        "Dots are 132 paired country–years; white circles show medians",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=5.0,
    )


def draw_binding(ax: plt.Axes, data: pd.DataFrame) -> None:
    row_d = data.loc[data["_chronology"].eq("dispersed")].iloc[0]
    row_c = data.loc[data["_chronology"].eq("clustered")].iloc[0]
    specs = [
        ("Tail charging\ninterface", "tail_charge_power_binding_hours", 1.0),
        ("Shared surplus\nlimit", "shared_recharge_binding_hours", 0.0),
    ]
    for label, field, y in specs:
        vd, vc = float(row_d[field]), float(row_c[field])
        ax.plot([vc, vd], [y, y], color="#AAB7BC", lw=1.8, solid_capstyle="round")
        ax.scatter([vd], [y], s=33, color=DISPERSED, edgecolor="white", linewidth=0.4, zorder=4)
        ax.scatter([vc], [y], s=33, color=CLUSTERED, edgecolor="white", linewidth=0.4, zorder=4)
        ax.text(vd, y + 0.16, f"{vd:.1f}", ha="center", va="bottom", fontsize=5.1, color=DISPERSED)
        ax.text(vc, y - 0.16, f"{vc:.1f}", ha="center", va="top", fontsize=5.1, color=CLUSTERED)
    ax.set_xscale("log")
    ax.set_xlim(0.75, 760)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([specs[0][0], specs[1][0]])
    ax.set_xlabel("Binding hours yr$^{-1}$")
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=24, color=DISPERSED, label="Dispersed")
    ax.scatter([], [], s=24, color=CLUSTERED, label="Clustered")
    ax.legend(loc="upper left", ncol=2, handletextpad=0.3, columnspacing=0.7)


def draw_surplus_intervention(ax: plt.Axes, data: pd.DataFrame) -> None:
    label_map = {
        ("caes", 2): "A-CAES 24 h\n2× surplus",
        ("caes", 4): "A-CAES 24 h\n4× surplus",
        ("h2_cavern", 2): "Cavern H$_2$ 48 h\n2× surplus",
        ("h2_cavern", 4): "Cavern H$_2$ 48 h\n4× surplus",
    }
    plot = data.copy()
    plot["plot_label"] = [label_map[(tech, int(mult))] for tech, mult in zip(plot["tech_id"], plot["recharge_quantity_multiplier"])]
    order = [
        "A-CAES 24 h\n2× surplus",
        "A-CAES 24 h\n4× surplus",
        "Cavern H$_2$ 48 h\n2× surplus",
        "Cavern H$_2$ 48 h\n4× surplus",
    ]
    colors = {"caes": TEAL, "h2_cavern": PURPLE}
    y = np.arange(len(order))[::-1]
    ax.axvline(0, color=INK, lw=0.7)
    for yi, label in zip(y, order):
        row = plot.loc[plot["plot_label"].eq(label)].iloc[0]
        mean = float(row["country_block_mean"])
        low = float(row["country_block_ci_low"])
        high = float(row["country_block_ci_high"])
        color = colors[row["tech_id"]]
        ax.plot([low, high], [yi, yi], color=color, lw=1.5, solid_capstyle="round")
        ax.scatter([mean], [yi], s=25, facecolor=color, edgecolor="white", linewidth=0.45, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(-0.050, 0.012)
    ax.set_xlabel("Change from the 1× surplus case\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_axisbelow(True)


def draw_cycle_value_link(ax: plt.Axes, data: pd.DataFrame) -> None:
    x = data["clustered_minus_dispersed_tail_equivalent_full_cycles"].to_numpy(float)
    y = data["clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year"].to_numpy(float)
    charge_loss = -data["tail_charge_change_gwh"].to_numpy(float)
    coef = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 200)
    norm = mpl.colors.Normalize(vmin=0, vmax=np.quantile(charge_loss, 0.97))
    scatter = ax.scatter(
        x,
        y,
        c=np.clip(charge_loss, 0, norm.vmax),
        cmap="cividis",
        norm=norm,
        s=14,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.25,
        rasterized=True,
    )
    ax.plot(grid, np.polyval(coef, grid), color=INK, lw=1.25)
    ax.axhline(0, color=INK, lw=0.55)
    ax.axvline(0, color=INK, lw=0.55)
    ax.set_xlabel("Change in A-CAES equivalent full cycles")
    ax.set_ylabel("Change in incremental portfolio value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_xlim(x.min() - 0.5, 0.8)
    ax.set_ylim(y.min() - 0.8, y.max() + 0.8)
    ax.grid(color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.97, 0.06, f"Pearson r = {r:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color=INK, fontweight="bold")
    cbar = ax.figure.colorbar(scatter, ax=ax, orientation="vertical", fraction=0.050, pad=0.025, aspect=24)
    cbar.set_label("Tail-charge loss (GWh)", fontsize=5.2, labelpad=2.0, rotation=270, va="bottom")
    cbar.ax.tick_params(labelsize=5.0, length=1.8)
    cbar.outline.set_linewidth(0.45)


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
    retention, binding, did, link = load_inputs()
    fig = plt.figure(figsize=(7.2, 4.82))
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.115, top=0.965)
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.0], hspace=0.58, wspace=1.05)
    axa = fig.add_subplot(gs[0, 0:7])
    axb = fig.add_subplot(gs[0, 8:12])
    axc = fig.add_subplot(gs[1, 0:5])
    axd = fig.add_subplot(gs[1, 6:12])
    axes = [axa, axb, axc, axd]

    draw_retention(axa, retention)
    draw_binding(axb, binding)
    draw_surplus_intervention(axc, did)
    draw_cycle_value_link(axd, link)
    panel_labels(fig, axes)

    stem = "fig_v45_figure4_recovery_interface_mechanism"
    outputs = export(fig, stem)
    plt.close(fig)

    manifest = {
        "figure": "Figure 4",
        "version": "v45",
        "core_conclusion": "Event clustering compresses cycling return by removing recovery time at the charging interface, not by reducing annual surplus quantity.",
        "archetype": "asymmetric quantitative grid with a distribution-led hero panel",
        "backend": "Python matplotlib",
        "final_size_inches": [WIDTH, HEIGHT],
        "panel_map": {
            "a": "country-year retention distributions for charging, tail service and equivalent cycles",
            "b": "paired binding hours for the shared surplus limit and tail charging interface",
            "c": "2x and 4x surplus-quantity contrasts relative to the 1x chronology effect",
            "d": "country-year closure from cycle loss to incremental portfolio value",
        },
        "statistics": "Panel a shows all 132 country-years. Panel c reports country-block means and percentile 95% intervals from 10,000 resamples of 15 countries.",
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [RETENTION_FILE, BINDING_FILE, DID_FILE, LINK_FILE]},
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "qa": {
            "retention_rows": len(retention),
            "intervention_rows": len(did),
            "country_year_rows": len(link),
            "missing_values_in_plotted_fields": int(
                retention["clustered_to_dispersed_pct"].isna().sum()
                + did[["country_block_mean", "country_block_ci_low", "country_block_ci_high"]].isna().sum().sum()
                + link[["clustered_minus_dispersed_tail_equivalent_full_cycles", "clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year"]].isna().sum().sum()
            ),
        },
    }
    manifest_path = SOURCE / "figure4_v45_build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "outputs": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
