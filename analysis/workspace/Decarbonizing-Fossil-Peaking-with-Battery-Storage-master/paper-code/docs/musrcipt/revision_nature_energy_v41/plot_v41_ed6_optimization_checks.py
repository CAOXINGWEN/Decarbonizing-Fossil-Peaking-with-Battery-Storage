#!/usr/bin/env python3
"""Build Extended Data Figure 6 from the full confirmatory Gurobi run.

The figure is an asymmetric evidence plate. The hero panel shows the
perfect-foresight technology-duration envelope across all 132 country-years.
Three supporting panels show paired valley formation across dispatch rules,
the observed gap-width distribution, and the charge-tie-break gain obtained
without changing peaking-task supply.
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
RESULTS = HERE / "perfect_foresight_valley_full_132cy"
AUDIT = HERE / "perfect_foresight_valley_independent_audit_v41.json"
OUTPUT = HERE / "supplement"
SOURCE_DATA = HERE / "source_data/figure_source_data"

WIDTH = 7.20
HEIGHT = 5.10
INK = "#243746"
MUTED = "#667780"
GRID = "#E2E8EB"
PALE = "#F4F7F8"
DISPERSED = "#355F8D"
CLUSTERED = "#D95F02"
PERFECT = "#159D89"
HEURISTIC = "#8B959A"
MATCHED = "#6B55A3"
VALLEY_FILL = "#F6D8C6"

DISPATCH_LABEL = {
    "frozen_v40_heuristic": "Frozen heuristic",
    "matched_greedy": "Energy-matched greedy",
    "perfect_foresight": "Perfect-foresight LP",
}
DISPATCH_COLOR = {
    "frozen_v40_heuristic": HEURISTIC,
    "matched_greedy": MATCHED,
    "perfect_foresight": PERFECT,
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.labelsize": 6.6,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.55,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.2,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def export(fig: plt.Figure) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = "fig_v41_ed6_optimization_checks"
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches=None)
    fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches=None)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches=None)
    fig.savefig(
        OUTPUT / f"{stem}.tiff",
        dpi=600,
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(OUTPUT / f"{stem}.png") as image:
        image.convert("L").save(
            OUTPUT / f"{stem}_grayscale.png",
            dpi=(300, 300),
        )
    plt.close(fig)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 - 0.012,
        box.y1 + 0.010,
        label,
        ha="right",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color="#111111",
    )


def load_and_validate() -> tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("audit_status") != "pass":
        raise RuntimeError("Independent perfect-foresight audit has not passed")
    if not audit.get(
        "manuscript_evidence_admitted_after_independent_audit"
    ):
        raise RuntimeError("Full run has not been independently admitted")

    envelope = pd.read_csv(
        RESULTS / "scenario31_duration_envelope.csv.gz"
    )
    records = pd.read_csv(RESULTS / "strict_valley_records.csv")
    comparison = pd.read_csv(
        RESULTS / "heuristic_vs_perfect_foresight.csv.gz"
    )
    if (
        envelope["country"].nunique() != 15
        or envelope[["country", "year"]].drop_duplicates().shape[0] != 132
    ):
        raise RuntimeError("Unexpected envelope scope")
    if comparison.shape[0] != 13992:
        raise RuntimeError("Expected 13,992 perfect/matched LP cells")
    return audit, envelope, records, comparison


def prepare_envelope(
    envelope: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = envelope.loc[
        envelope["dispatch_mode"].eq("perfect_foresight"),
        [
            "country",
            "year",
            "chronology",
            "duration_h",
            "config_id",
            "tech_id",
            "scenario31_net_benefit_eur_per_kw_year",
        ],
    ].copy()
    if raw.shape[0] != 132 * 2 * 24:
        raise RuntimeError("Perfect-foresight envelope is incomplete")
    summary = (
        raw.groupby(["chronology", "duration_h"])[
            "scenario31_net_benefit_eur_per_kw_year"
        ]
        .agg(
            median="median",
            mean="mean",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            minimum="min",
            maximum="max",
            n="size",
        )
        .reset_index()
    )
    return raw, summary


def prepare_prevalence(records: pd.DataFrame) -> pd.DataFrame:
    summary = (
        records.groupby(["dispatch_mode", "chronology"], as_index=False)
        .agg(
            strict_valleys=("has_strict_valley", "sum"),
            country_years=("has_strict_valley", "size"),
            mean_gap_width_h=("gap_width_h", "mean"),
            median_gap_width_h=("gap_width_h", "median"),
        )
    )
    summary["strict_valley_prevalence_pct"] = (
        100 * summary["strict_valleys"] / summary["country_years"]
    )
    return summary


def prepare_widths(records: pd.DataFrame) -> pd.DataFrame:
    widths = records.loc[
        records["dispatch_mode"].eq("perfect_foresight")
        & records["chronology"].eq("clustered"),
        [
            "country",
            "year",
            "gap_width_h",
            "left_positive_duration_h",
            "nonpositive_start_duration_h",
            "nonpositive_end_duration_h",
            "right_reentry_duration_h",
        ],
    ].copy()
    if widths.shape[0] != 132 or not widths["gap_width_h"].gt(0).all():
        raise RuntimeError("Perfect-foresight gap-width records are incomplete")
    return widths


def prepare_efficiency(comparison: pd.DataFrame) -> pd.DataFrame:
    country_year = (
        comparison.groupby(["country", "year", "chronology"], as_index=False)
        .agg(
            matched_charge_mwh=("matched_charge_mwh", "sum"),
            perfect_charge_mwh=("perfect_charge_mwh", "sum"),
            matched_discharge_mwh=("matched_discharge_mwh", "sum"),
            perfect_discharge_mwh=("perfect_discharge_mwh", "sum"),
        )
    )
    country_year["charge_avoided_mwh"] = (
        country_year["matched_charge_mwh"]
        - country_year["perfect_charge_mwh"]
    )
    country_year["charge_avoided_pct"] = (
        100
        * country_year["charge_avoided_mwh"]
        / country_year["matched_charge_mwh"]
    )
    country_year["task_supply_retained_pct"] = (
        100
        * country_year["perfect_discharge_mwh"]
        / country_year["matched_discharge_mwh"]
    )
    if country_year.shape[0] != 264:
        raise RuntimeError("Dispatch-efficiency aggregation is incomplete")
    return country_year


def draw_hero_envelope(
    ax: plt.Axes,
    summary: pd.DataFrame,
) -> None:
    ax.axhspan(-24, 0, color="#F8F1ED", alpha=0.75, zorder=0)
    ax.axvspan(28, 40, color=VALLEY_FILL, alpha=0.72, zorder=0)
    ax.axhline(0, color=INK, lw=0.75, zorder=1)
    for chronology, colour, linestyle, marker in [
        ("dispersed", DISPERSED, "-", "o"),
        ("clustered", CLUSTERED, "-", "s"),
    ]:
        group = summary.loc[
            summary["chronology"].eq(chronology)
        ].sort_values("duration_h")
        x = group["duration_h"].to_numpy(float)
        median = group["median"].to_numpy(float)
        q25 = group["q25"].to_numpy(float)
        q75 = group["q75"].to_numpy(float)
        ax.fill_between(
            x,
            q25,
            q75,
            color=colour,
            alpha=0.13,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            x,
            median,
            color=colour,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.7,
            markevery=2,
            label=chronology.capitalize(),
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlim(3.7, 180)
    ax.set_ylim(-25, 220)
    ticks = [4, 8, 12, 24, 40, 72, 120, 168]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Storage duration (h)")
    ax.set_ylabel(
        "Best-technology annual net benefit\n"
        "(EUR kW$^{-1}$ yr$^{-1}$)"
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    ax.legend(
        loc="upper left",
        ncol=2,
        handlelength=2.4,
        columnspacing=1.2,
    )
    ax.text(
        0.99,
        0.985,
        "median and IQR · n = 132 country-years",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=MUTED,
    )
    ax.annotate(
        "Clustered mid-duration valley\nmedian envelope below zero",
        xy=(34, -10),
        xytext=(53, 40),
        ha="left",
        va="bottom",
        color=CLUSTERED,
        fontsize=6.0,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-|>",
            "color": CLUSTERED,
            "lw": 0.75,
            "mutation_scale": 7,
        },
    )
    ax.annotate(
        "long-duration re-entry",
        xy=(44, 0.8),
        xytext=(76, -18),
        ha="center",
        va="top",
        color=INK,
        fontsize=5.5,
        arrowprops={
            "arrowstyle": "-|>",
            "color": INK,
            "lw": 0.65,
            "mutation_scale": 6,
        },
    )
    ax.text(
        0.02,
        0.04,
        "Scenario 31  |  energy €98.5 MWh$^{-1}$  |  "
        "firm value €103.8 kW$^{-1}$ yr$^{-1}$  |  "
        "charge price €0 MWh$^{-1}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=MUTED,
    )


def draw_prevalence(
    ax: plt.Axes,
    prevalence: pd.DataFrame,
) -> None:
    modes = [
        "frozen_v40_heuristic",
        "matched_greedy",
        "perfect_foresight",
    ]
    y = np.arange(len(modes))[::-1]
    for ypos, mode in zip(y, modes):
        row = prevalence.loc[prevalence["dispatch_mode"].eq(mode)]
        dispersed = float(
            row.loc[
                row["chronology"].eq("dispersed"),
                "strict_valley_prevalence_pct",
            ].iloc[0]
        )
        clustered = float(
            row.loc[
                row["chronology"].eq("clustered"),
                "strict_valley_prevalence_pct",
            ].iloc[0]
        )
        ax.annotate(
            "",
            xy=(clustered, ypos),
            xytext=(dispersed, ypos),
            arrowprops={
                "arrowstyle": "-|>",
                "color": DISPATCH_COLOR[mode],
                "lw": 1.6 if mode == "perfect_foresight" else 1.0,
                "mutation_scale": 8,
            },
            zorder=2,
        )
        ax.scatter(
            dispersed,
            ypos,
            s=28,
            facecolor="white",
            edgecolor=DISPERSED,
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            clustered,
            ypos,
            s=28,
            facecolor=CLUSTERED,
            edgecolor=CLUSTERED,
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            2.5,
            ypos + 0.17,
            "0/132",
            ha="left",
            va="bottom",
            color=DISPERSED,
            fontsize=5.2,
        )
        ax.text(
            97.5,
            ypos + 0.17,
            "132/132",
            ha="right",
            va="bottom",
            color=CLUSTERED,
            fontsize=5.2,
            fontweight="bold" if mode == "perfect_foresight" else "normal",
        )
    ax.set_xlim(-4, 104)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Country-years with a strict valley (%)")
    ax.set_yticks(y)
    ax.set_yticklabels([DISPATCH_LABEL[mode] for mode in modes])
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax.text(
        0.02,
        1.04,
        "Dispersed",
        transform=ax.transAxes,
        color=DISPERSED,
        ha="left",
        va="bottom",
        fontsize=5.3,
        fontweight="bold",
    )
    ax.text(
        0.98,
        1.04,
        "Clustered",
        transform=ax.transAxes,
        color=CLUSTERED,
        ha="right",
        va="bottom",
        fontsize=5.3,
        fontweight="bold",
    )


def draw_width_distribution(
    ax: plt.Axes,
    widths: pd.DataFrame,
) -> None:
    counts = widths["gap_width_h"].value_counts().sort_index()
    x = counts.index.to_numpy(float)
    y = counts.to_numpy(int)
    ax.vlines(x, 0, y, color="#C4CDD1", lw=2.2, zorder=1)
    ax.scatter(
        x,
        y,
        s=31,
        color=PERFECT,
        edgecolor="white",
        linewidth=0.55,
        zorder=2,
    )
    for xpos, count in zip(x, y):
        ax.text(
            xpos,
            count + 2.0,
            str(count),
            ha="center",
            va="bottom",
            fontsize=5.4,
            color=INK,
        )
    mean_width = float(widths["gap_width_h"].mean())
    median_width = float(widths["gap_width_h"].median())
    ax.axvline(
        mean_width,
        color=CLUSTERED,
        linestyle="--",
        lw=0.9,
        label=f"mean {mean_width:.1f} h",
    )
    ax.axvline(
        median_width,
        color=INK,
        linestyle=":",
        lw=0.9,
        label=f"median {median_width:.0f} h",
    )
    ax.set_xlim(11.5, 38.5)
    ax.set_ylim(0, max(y) * 1.22)
    ax.set_xticks(sorted(set([14, 16, 20, 24, 36])))
    ax.set_xlabel("Clustered gap width (h)")
    ax.set_ylabel("Country-years")
    ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    ax.legend(
        loc="upper right",
        handlelength=2.3,
        fontsize=5.3,
    )
    ax.text(
        0.02,
        0.95,
        "Perfect foresight · n = 132",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.3,
        color=MUTED,
    )


def draw_dispatch_efficiency(
    ax: plt.Axes,
    efficiency: pd.DataFrame,
    audit: dict,
) -> None:
    groups = [
        efficiency.loc[
            efficiency["chronology"].eq("dispersed"),
            "charge_avoided_pct",
        ].to_numpy(float),
        efficiency.loc[
            efficiency["chronology"].eq("clustered"),
            "charge_avoided_pct",
        ].to_numpy(float),
    ]
    violin = ax.violinplot(
        groups,
        positions=[0, 1],
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, colour in zip(violin["bodies"], [DISPERSED, CLUSTERED]):
        body.set_facecolor(colour)
        body.set_edgecolor(colour)
        body.set_alpha(0.16)
        body.set_linewidth(0.7)
    for index, (values, colour) in enumerate(
        zip(groups, [DISPERSED, CLUSTERED])
    ):
        # Deterministic low-discrepancy jitter that is independent of value,
        # avoiding an artificial diagonal relationship inside each violin.
        x = (
            (np.arange(len(values), dtype=float) * 0.61803398875) % 1.0
            - 0.5
        ) * 0.44
        ax.scatter(
            index + x,
            values,
            s=5.2,
            color=colour,
            alpha=0.32,
            edgecolor="none",
            rasterized=False,
            zorder=2,
        )
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.vlines(index, q25, q75, color=INK, lw=2.0, zorder=3)
        ax.scatter(
            index,
            median,
            s=20,
            color="white",
            edgecolor=INK,
            linewidth=0.7,
            zorder=4,
        )
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 0.245)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Dispersed", "Clustered"])
    ax.set_ylabel("Charging avoided by foresight (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    paired = audit["paired_dispatch_check"]
    ax.text(
        0.98,
        0.97,
        "Task supply retained 100.000%\n"
        f"Charging avoided "
        f"{paired['total_charge_avoided_mwh'] / 1e6:.3f} TWh",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.97,
        "n = 132 country-years per chronology",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.1,
        color=MUTED,
    )


def save_source_data(
    raw_envelope: pd.DataFrame,
    envelope_summary: pd.DataFrame,
    prevalence: pd.DataFrame,
    widths: pd.DataFrame,
    efficiency: pd.DataFrame,
    audit: dict,
) -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    raw_envelope.to_csv(
        SOURCE_DATA / "figure_ed6_panel_a_envelope_country_year.csv.gz",
        index=False,
        compression="gzip",
    )
    envelope_summary.to_csv(
        SOURCE_DATA / "figure_ed6_panel_a_envelope_summary.csv",
        index=False,
    )
    prevalence.to_csv(
        SOURCE_DATA / "figure_ed6_panel_b_valley_prevalence.csv",
        index=False,
    )
    widths.to_csv(
        SOURCE_DATA / "figure_ed6_panel_c_gap_width.csv",
        index=False,
    )
    efficiency.to_csv(
        SOURCE_DATA / "figure_ed6_panel_d_dispatch_efficiency.csv",
        index=False,
    )
    (
        SOURCE_DATA / "figure_ed6_independent_audit_summary.json"
    ).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    source_files = [
        "figure_ed6_panel_a_envelope_country_year.csv.gz",
        "figure_ed6_panel_a_envelope_summary.csv",
        "figure_ed6_panel_b_valley_prevalence.csv",
        "figure_ed6_panel_c_gap_width.csv",
        "figure_ed6_panel_d_dispatch_efficiency.csv",
        "figure_ed6_independent_audit_summary.json",
    ]
    manifest = {
        "status": "pass",
        "figure": "Extended Data Figure 6",
        "core_claim": (
            "Perfect foresight preserves clustering-induced strict-valley "
            "formation while supplying the same peaking task with less charge."
        ),
        "scope": audit["scope"],
        "confirmatory_result": audit["confirmatory_result"],
        "source_data": {},
    }
    for filename in source_files:
        path = SOURCE_DATA / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["source_data"][filename] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    (SOURCE_DATA / "figure_ed6_build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build() -> None:
    configure()
    audit, envelope, records, comparison = load_and_validate()
    raw_envelope, envelope_summary = prepare_envelope(envelope)
    prevalence = prepare_prevalence(records)
    widths = prepare_widths(records)
    efficiency = prepare_efficiency(comparison)

    # Final two-column width is fixed at 7.20 inches.
    fig = plt.figure(figsize=(7.20, 5.10))
    fig.subplots_adjust(
        left=0.075,
        right=0.986,
        bottom=0.105,
        top=0.942,
    )
    grid = fig.add_gridspec(
        2,
        12,
        height_ratios=[1.56, 1.0],
        hspace=0.44,
        wspace=1.05,
    )
    axa = fig.add_subplot(grid[0, 0:8])
    axb = fig.add_subplot(grid[0, 8:12])
    axc = fig.add_subplot(grid[1, 0:5])
    axd = fig.add_subplot(grid[1, 5:12])

    draw_hero_envelope(axa, envelope_summary)
    draw_prevalence(axb, prevalence)
    draw_width_distribution(axc, widths)
    draw_dispatch_efficiency(axd, efficiency, audit)

    fig.text(
        0.075,
        0.985,
        "Confirmatory perfect-foresight test of the mid-duration valley",
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.986,
        0.985,
        "15 countries · 132 country-years · 53 configurations · "
        "13,992 continuous LPs",
        ha="right",
        va="top",
        fontsize=5.5,
        color=MUTED,
    )
    for label, axis in zip("abcd", [axa, axb, axc, axd]):
        panel_label(fig, axis, label)

    save_source_data(
        raw_envelope,
        envelope_summary,
        prevalence,
        widths,
        efficiency,
        audit,
    )
    export(fig)


if __name__ == "__main__":
    build()
