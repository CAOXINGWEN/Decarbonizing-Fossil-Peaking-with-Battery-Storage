#!/usr/bin/env python3
"""Rebuild Extended Data Figures 1-10 in the v54 Nature Energy visual system."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch

from plot_v51_main_figures import WORLD_SHP, add_shape, deterministic_jitter
from plot_v54_reference_style_figures import (
    BLUE,
    CORAL,
    GOLD,
    GREY,
    GRID,
    INK,
    LIGHT,
    MUTED,
    SAGE,
    TAUPE,
    configure,
    label_panel,
    quiet,
)

ORANGE = CORAL
PALE = LIGHT
PURPLE = TAUPE
RED = "#B85C5C"
TEAL = SAGE
CMAP_SEQ = mpl.colors.LinearSegmentedColormap.from_list("v54_seq", [BLUE, SAGE, GOLD, CORAL])
CMAP_DIV = mpl.colors.LinearSegmentedColormap.from_list("v54_div", [CORAL, LIGHT, BLUE])


def panel(ax, letter: str, x: float = -0.08, y: float = 1.04) -> None:
    label_panel(ax, letter, x, y)


def clean(ax, grid: str | None = "y") -> None:
    quiet(ax, grid)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "source_data"
OUT = HERE / "extended_data"
OUT.mkdir(parents=True, exist_ok=True)

TECH_ORDER = ["li_lfp", "vrfb", "fecr_flow", "gravity", "molten_salt_tes", "caes", "laes", "h2_cavern"]
TECH_LABELS = {
    "li_lfp": "LFP",
    "vrfb": "VRFB",
    "fecr_flow": "Fe-Cr flow",
    "gravity": "Gravity",
    "molten_salt_tes": "Molten-salt TES",
    "caes": "A-CAES",
    "laes": "LAES",
    "h2_cavern": "Cavern H$_2$",
}
TECH_COLOURS = {
    "li_lfp": BLUE,
    "vrfb": "#90B8C9",
    "fecr_flow": "#A9C8D3",
    "gravity": GOLD,
    "molten_salt_tes": CORAL,
    "caes": SAGE,
    "laes": TAUPE,
    "h2_cavern": "#716A63",
}


def save_ed(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(OUT / f"{stem}_grayscale.png", grey, cmap="gray", vmin=0, vmax=1)
    plt.close(fig)


def ecdf(ax, values, colour, label, lw=1.35) -> None:
    values = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(values) + 1) / len(values) * 100
    ax.step(values, y, where="post", color=colour, lw=lw, label=label)


def ed1() -> None:
    data = pd.read_csv(SRC / "ed1_ed1_53_duration_year_cost_paths_eur.csv")
    nonpositive_capex = data.total_capex_eur_kwh <= 0
    if nonpositive_capex.any():
        raise ValueError("Installed CAPEX must be positive before log scaling")
    fig = plt.figure(figsize=(7.20, 6.15))
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.30, 1.05, 0.92], hspace=0.55, wspace=0.95,
                  left=0.075, right=0.985, top=0.96, bottom=0.085)
    evidence_markers = {
        "Direct evidence duration": "o",
        "Released configuration": "s",
        "Within-domain component interpolation": "D",
        "H2 structural extrapolation": "^",
    }

    axa = fig.add_subplot(gs[0:2, :7])
    for tech in TECH_ORDER:
        q = data[data.tech_id == tech]
        for _, line in q.groupby("duration_h"):
            line = line.sort_values("learning_year")
            axa.plot(line.learning_year, line.total_capex_eur_kwh, color=TECH_COLOURS[tech],
                     lw=0.82, alpha=0.40, ls=":" if line.structural_extrapolation.any() else "-")
        med = q.groupby("learning_year").total_capex_eur_kwh.median()
        axa.plot(med.index, med.values, color=TECH_COLOURS[tech], lw=1.65, label=TECH_LABELS[tech])
    axa.set_yscale("log"); axa.set_xticks([2025, 2030, 2035])
    axa.set_xlabel("Cost snapshot"); axa.set_ylabel("Installed CAPEX (EUR kWh$^{-1}$)")
    axa.legend(ncol=2, loc="lower left", columnspacing=0.7, handlelength=1.2)
    clean(axa); panel(axa, "a", -0.09, 1.02)

    axb = fig.add_subplot(gs[0, 7:])
    d25 = data[data.learning_year == 2025].set_index(["tech_id", "duration_h"])
    d35 = data[data.learning_year == 2035].set_index(["tech_id", "duration_h"])
    idx = d25.index.intersection(d35.index)
    for tech in TECH_ORDER:
        ids = [key for key in idx if key[0] == tech]
        if not ids:
            continue
        x = d25.loc[ids, "total_capex_eur_kwh"].to_numpy()
        y = d35.loc[ids, "total_capex_eur_kwh"].to_numpy()
        axb.scatter(x, y, s=12, color=TECH_COLOURS[tech], edgecolor="white", lw=0.3, label=TECH_LABELS[tech])
    lo = min(d25.loc[idx, "total_capex_eur_kwh"].min(), d35.loc[idx, "total_capex_eur_kwh"].min())
    hi = max(d25.loc[idx, "total_capex_eur_kwh"].max(), d35.loc[idx, "total_capex_eur_kwh"].max())
    axb.plot([lo, hi], [lo, hi], color=INK, ls="--", lw=0.65)
    axb.set_xscale("log"); axb.set_yscale("log"); axb.set_xlabel("2025 CAPEX"); axb.set_ylabel("2035 CAPEX")
    clean(axb); panel(axb, "b", -0.16, 1.02)

    axc = fig.add_subplot(gs[1, 7:])
    reduction = (1 - d35.loc[idx, "total_capex_eur_kwh"] / d25.loc[idx, "total_capex_eur_kwh"]) * 100
    for i, tech in enumerate(TECH_ORDER):
        vals = reduction[[key[0] == tech for key in reduction.index]].to_numpy()
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        axc.plot([q1, q3], [i, i], color=TECH_COLOURS[tech], lw=2.6)
        axc.scatter(med, i, s=28, color=TECH_COLOURS[tech], edgecolor="white", lw=0.4, zorder=3)
    axc.set_yticks(range(8), [TECH_LABELS[t] for t in TECH_ORDER]); axc.invert_yaxis()
    axc.set_xlabel("2025-2035 CAPEX reduction (%)")
    clean(axc, "x"); panel(axc, "c", -0.16, 1.02)

    axd = fig.add_subplot(gs[2, :4])
    for tech in TECH_ORDER:
        q = data[(data.tech_id == tech) & (data.learning_year == 2035)].sort_values("duration_h")
        axd.plot(q.duration_h, q.total_capex_eur_kwh, color=TECH_COLOURS[tech], lw=1.0,
                 marker="o", ms=2.2, label=TECH_LABELS[tech])
    axd.set_xscale("log"); axd.set_yscale("log"); axd.set_xlabel("Discharge duration (h)")
    axd.set_ylabel("2035 CAPEX"); clean(axd); panel(axd, "d", -0.18, 1.03)

    axe = fig.add_subplot(gs[2, 4:8])
    counts = data[data.learning_year == 2025].groupby(["tech_id", "evidence_class"]).size().unstack(fill_value=0)
    left = np.zeros(len(TECH_ORDER))
    evidence_colours = [BLUE, SAGE, GOLD, CORAL]
    for colour, evidence in zip(evidence_colours, evidence_markers):
        vals = counts.reindex(TECH_ORDER).get(evidence, pd.Series(0, index=TECH_ORDER)).to_numpy()
        axe.barh(range(8), vals, left=left, color=colour, height=0.58, label=evidence)
        left += vals
    axe.set_yticks(range(8), [TECH_LABELS[t] for t in TECH_ORDER]); axe.invert_yaxis()
    axe.set_xlabel("Configurations"); axe.legend(loc="lower right", fontsize=5.0)
    clean(axe, "x"); panel(axe, "e", -0.18, 1.03)

    axf = fig.add_subplot(gs[2, 8:])
    q = data[data.learning_year == 2035]
    for i, tech in enumerate(TECH_ORDER):
        vals = q.loc[q.tech_id == tech, "total_capex_eur_kwh"].to_numpy()
        jitter = deterministic_jitter(len(vals), 0.13)
        axf.scatter(vals, i + jitter, s=8, color=TECH_COLOURS[tech], alpha=0.48, edgecolor="none")
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        axf.plot([q1, q3], [i, i], color=TECH_COLOURS[tech], lw=2.3)
        axf.scatter(med, i, s=22, color=INK, edgecolor="white", lw=0.35, zorder=3)
    axf.set_xscale("log"); axf.set_yticks(range(8), [TECH_LABELS[t] for t in TECH_ORDER]); axf.invert_yaxis()
    axf.set_xlabel("2035 CAPEX distribution")
    clean(axf, "x"); panel(axf, "f", -0.18, 1.03)
    save_ed(fig, "fig_v54_ed1_cost_paths_53_eur")


def ed2() -> None:
    data = pd.read_csv(SRC / "ed2_ed2_event_definition_sensitivity.csv")
    fig = plt.figure(figsize=(7.20, 5.10))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[0.82, 1.30], hspace=0.48, wspace=0.65,
                  left=0.08, right=0.985, top=0.96, bottom=0.10)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]
    colours = {70: BLUE, 75: GOLD, 80: TEAL}
    metrics = [
        ("longest_event_run_days", "Longest event run (d)"),
        ("task_twh", "Identified tail task (TWh)"),
        ("event_task_share_pct", "Event-task share (%)"),
    ]
    for i, (ax, (metric, xlabel)) in enumerate(zip(axes, metrics)):
        for quantile in [70, 75, 80]:
            ecdf(ax, data.loc[data["quantile"] == quantile, metric], colours[quantile], f"q{quantile}")
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel("Cumulative share (%)")
            ax.legend(ncol=3, loc="lower right", handlelength=1.0, columnspacing=0.7)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(0, 101); clean(ax); panel(ax, chr(ord("a") + i), -0.19)

    axd = fig.add_subplot(gs[1, :])
    q70 = data[data["quantile"] == 70].set_index(["country", "year"])
    q80 = data[data["quantile"] == 80].set_index(["country", "year"])
    idx = q70.index.intersection(q80.index)
    x = q70.loc[idx, "event_task_share_pct"].to_numpy()
    y = q80.loc[idx, "event_task_share_pct"].to_numpy()
    task = data[data["quantile"] == 75].set_index(["country", "year"]).loc[idx, "task_twh"].to_numpy()
    lo = min(x.min(), y.min()) - 2; hi = max(x.max(), y.max()) + 2
    sc = axd.scatter(x, y, c=task, cmap=CMAP_SEQ, s=11 + 5 * np.sqrt(task), alpha=0.72,
                     edgecolor="white", lw=0.25)
    axd.plot([lo, hi], [lo, hi], color=INK, ls="--", lw=0.8)
    axd.fill_between([lo, hi], [lo, hi], [hi, hi], color=mpl.colors.to_rgba(ORANGE, 0.06))
    axd.text(0.03, 0.94, "q80 allocates less annual task to events", transform=axd.transAxes,
             color=ORANGE, fontsize=5.4, va="top")
    axd.set_xlim(lo, hi); axd.set_ylim(lo, hi)
    axd.set_xlabel("q70 event-task share (%)"); axd.set_ylabel("q80 event-task share (%)")
    cb = fig.colorbar(sc, ax=axd, fraction=0.025, pad=0.015); cb.set_label("q75 tail task (TWh)")
    clean(axd); panel(axd, "d", -0.07)
    save_ed(fig, "fig_v54_ed2_event_definition_sensitivity")


def residual_strip(ax, data, colour, title, unit) -> None:
    metrics = list(data.metric.drop_duplicates())
    runs = list(data.run.drop_duplicates())
    xpos = np.arange(len(runs))
    for j, metric in enumerate(metrics):
        q = data[data.metric == metric].set_index("run").reindex(runs)
        vals = q.maximum_absolute_residual.to_numpy(float)
        offset = (j - (len(metrics) - 1) / 2) * 0.16
        valid = np.isfinite(vals)
        ax.scatter(xpos[valid] + offset, vals[valid], s=31, color=mpl.colors.to_rgba(colour, 0.82),
                   marker=["o", "s", "D"][j % 3], edgecolor="white", lw=0.4, label=metric)
        for x0 in xpos[~valid] + offset:
            ax.text(x0, 1e-15, "n.a.", ha="center", va="bottom", color=MUTED, fontsize=5.0, rotation=90)
    ax.set_yscale("symlog", linthresh=1e-15)
    ax.set_xticks(xpos, ["central", "full fixed", "optimized"][:len(runs)], rotation=18, ha="right")
    ax.set_ylabel(f"Maximum residual ({unit})")
    ax.set_title(title, loc="left", fontsize=5.7, color=colour, fontweight="bold")
    ax.legend(loc="upper left", fontsize=5.0, handletextpad=0.2)
    clean(ax)


def ed3() -> None:
    conservation = pd.read_csv(SRC / "ed3_v43_chronology_conservation_errors.csv")
    soc = pd.read_csv(SRC / "ed3_v43_soc_residuals.csv")
    balance = pd.read_csv(SRC / "ed3_v43_balance_residuals.csv")
    accounting = pd.read_csv(SRC / "ed3_v43_accounting_residuals.csv")
    fig = plt.figure(figsize=(7.20, 5.15))
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1.25, 1.0], hspace=0.52, wspace=0.95,
                  left=0.08, right=0.985, top=0.96, bottom=0.11)
    axa = fig.add_subplot(gs[0, :])
    for metric, colour in zip(conservation.metric.unique(), [BLUE, ORANGE, TEAL]):
        vals = conservation.loc[conservation.metric == metric, "absolute_error_mwh"].to_numpy()
        ecdf(axa, vals, colour, metric, 1.45)
    axa.axvspan(0, 1e-8, color=mpl.colors.to_rgba(TEAL, 0.08), label="numerical tolerance zone")
    axa.set_xscale("symlog", linthresh=1e-15)
    axa.set_xlabel("Absolute annual conservation error (MWh; symlog)")
    axa.set_ylabel("Empirical cumulative probability (%)")
    axa.legend(ncol=4, loc="lower right", handlelength=1.0, columnspacing=0.8)
    clean(axa); panel(axa, "a", -0.06)
    axb, axc, axd = [fig.add_subplot(gs[1, sl]) for sl in [slice(0, 4), slice(4, 8), slice(8, 12)]]
    residual_strip(axb, soc, BLUE, "State-of-charge closure", "MWh")
    residual_strip(axc, balance, TEAL, "Task and recharge balance", "MW")
    residual_strip(axd, accounting, ORANGE, "Portfolio accounting", "EUR")
    for ax, letter in zip([axb, axc, axd], "bcd"):
        panel(ax, letter, -0.18)
    save_ed(fig, "fig_v54_ed3_numerical_qa")


def ed4() -> None:
    response = pd.read_csv(SRC / "ed4_v43_full_configuration_duration_response.csv")
    paired = pd.read_csv(SRC / "ed4_v43_configuration_paired_effects.csv")
    cycles = pd.read_csv(SRC / "ed4_v43_tail_cycle_retention.csv")
    techs = [t for t in TECH_ORDER if t != "li_lfp"]
    fig = plt.figure(figsize=(7.20, 5.70))
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1.35, 1.0], hspace=0.52, wspace=0.95,
                  left=0.085, right=0.985, top=0.965, bottom=0.10)
    axa = fig.add_subplot(gs[0, :])
    for tech in techs:
        colour = TECH_COLOURS[tech]
        for chronology, ls, alpha in [("dispersed", "-", 0.95), ("clustered", "--", 0.72)]:
            q = response[(response.tech_id == tech) & (response._chronology == chronology)].sort_values("duration_h")
            axa.plot(q.duration_h, q.median_mix_benefit, color=colour, ls=ls, alpha=alpha, lw=1.25,
                     marker="o" if chronology == "dispersed" else "s", ms=2.4, mec="white", mew=0.25)
        qend = response[(response.tech_id == tech) & (response._chronology == "dispersed")].sort_values("duration_h")
        axa.text(qend.duration_h.iloc[-1] * 1.02, qend.median_mix_benefit.iloc[-1], TECH_LABELS[tech],
                 color=colour, va="center", fontsize=5.0)
    axa.axhline(0, color=INK, lw=0.75); axa.set_xscale("log")
    axa.set_xlim(response.duration_h.min() * 0.92, response.duration_h.max() * 1.30)
    axa.set_xlabel("Candidate LDES duration (h)")
    axa.set_ylabel("Median LFP-tail-relative value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axa.text(0.77, 0.94, "solid / circle: dispersed    dashed / square: clustered",
             transform=axa.transAxes, ha="center", color=MUTED, fontsize=5.0)
    clean(axa); panel(axa, "a", -0.07)

    axb = fig.add_subplot(gs[1, :5])
    duration_order = sorted(paired.duration_h.unique())
    matrix = paired.pivot(index="tech_id", columns="duration_h", values="median").reindex(techs)
    vmax = np.nanmax(np.abs(matrix.to_numpy()))
    im = axb.imshow(matrix.to_numpy(), aspect="auto", cmap=CMAP_DIV, vmin=-vmax, vmax=vmax)
    axb.set_yticks(range(len(techs)), [TECH_LABELS[t] for t in techs])
    tick_idx = np.linspace(0, len(duration_order) - 1, min(7, len(duration_order))).round().astype(int)
    axb.set_xticks(tick_idx, [str(int(duration_order[i])) for i in tick_idx])
    axb.set_xlabel("Duration (h)"); axb.tick_params(length=0)
    cax = axb.inset_axes([0.20, 1.03, 0.62, 0.055])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Clustered - dispersed value", fontsize=5.0); cb.ax.tick_params(labelsize=5.0, length=1.2)
    panel(axb, "b", -0.15)

    axc = fig.add_subplot(gs[1, 5:8])
    summaries = []
    for tech in techs:
        vals = paired.loc[paired.tech_id == tech, "median"].to_numpy()
        summaries.append((tech, np.quantile(vals, .25), np.median(vals), np.quantile(vals, .75)))
    for i, (tech, q1, med, q3) in enumerate(summaries[::-1]):
        colour = TECH_COLOURS[tech]
        axc.plot([q1, q3], [i, i], color=colour, lw=3.2, solid_capstyle="round")
        axc.scatter(med, i, s=30, color=colour, edgecolor="white", lw=0.4, zorder=3)
    axc.axvline(0, color=INK, lw=0.75)
    axc.set_yticks(range(len(techs)), [TECH_LABELS[t] for t in techs[::-1]])
    axc.set_xlabel("Median paired effect")
    clean(axc, "x"); panel(axc, "c", -0.24)

    axd = fig.add_subplot(gs[1, 8:])
    for i, tech in enumerate(techs):
        vals = cycles.loc[cycles.tech_id == tech, "cycle_retention_pct"].to_numpy()
        vp = axd.violinplot(vals, positions=[i], vert=False, widths=0.72, showextrema=False, points=60)
        for body in vp["bodies"]:
            body.set_facecolor(TECH_COLOURS[tech]); body.set_edgecolor("none"); body.set_alpha(0.22)
        q1, med, q3 = np.quantile(vals, [.25, .5, .75])
        axd.plot([q1, q3], [i, i], color=TECH_COLOURS[tech], lw=3.0, solid_capstyle="round")
        axd.scatter(med, i, s=25, color=INK, edgecolor="white", lw=0.35, zorder=3)
    axd.axvline(100, color=INK, ls="--", lw=0.7)
    axd.set_yticks(range(len(techs)), [TECH_LABELS[t] for t in techs])
    axd.set_xlabel("Equivalent-cycle retention (%)")
    clean(axd, "x"); panel(axd, "d", -0.22)
    save_ed(fig, "fig_v54_ed4_full_technology_duration_response")


def ed5() -> None:
    actual = pd.read_csv(SRC / "ed5_actual_cost_year_envelopes.csv")
    valleys = pd.read_csv(SRC / "ed5_actual_cost_year_valley_counts.csv")
    conditional = pd.read_csv(SRC / "ed5_conditional_peak_duration_response.csv")
    boundaries = pd.read_csv(SRC / "ed5_conditional_peak_boundaries.csv")
    states = pd.read_csv(SRC / "ed5_conditional_peak_paired_states.csv")
    fig = plt.figure(figsize=(7.20, 5.65))
    gs = GridSpec(2, 12, figure=fig, hspace=0.50, wspace=1.00,
                  left=0.085, right=0.985, top=0.965, bottom=0.10)
    axa = fig.add_subplot(gs[0, :8]); axb = fig.add_subplot(gs[0, 8:])
    axe = fig.add_subplot(gs[1, :8]); axd = fig.add_subplot(gs[1, 8:])
    year_cols = {2025: "#9BA8AE", 2030: GOLD, 2035: PURPLE}
    for year in [2025, 2030, 2035]:
        for chronology, ls in [("dispersed", "-"), ("clustered", "--")]:
            q = actual[(actual.cost_year == year) & (actual.chronology == chronology)].sort_values("duration_h")
            axa.plot(q.duration_h, q["median"], color=year_cols[year], ls=ls, lw=1.25,
                     marker="o" if chronology == "dispersed" else "s", ms=2.2, mec="white", mew=0.2)
            if chronology == "clustered":
                axa.fill_between(q.duration_h, q.q25, q.q75, color=mpl.colors.to_rgba(year_cols[year], 0.07), lw=0)
    axa.axhline(0, color=INK, lw=0.75); axa.set_xscale("log")
    axa.set_xlabel("LDES duration (h)")
    axa.set_ylabel("Median incremental benefit\n(EUR kW$^{-1}$ yr$^{-1}$)")
    handles = [mpl.lines.Line2D([], [], color=year_cols[y], lw=1.5, label=str(y)) for y in year_cols]
    handles += [mpl.lines.Line2D([], [], color=INK, ls="-", label="D"),
                mpl.lines.Line2D([], [], color=INK, ls="--", label="C")]
    axa.legend(handles=handles, ncol=5, loc="lower left", handlelength=1.0, columnspacing=0.7)
    clean(axa); panel(axa, "a", -0.07)

    matrix = valleys.pivot(index="cost_year", columns="chronology", values="strict_valley_rate").reindex([2025,2030,2035])
    im = axb.imshow(100 * matrix.to_numpy(), cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "valleys", [PALE, GOLD, ORANGE]), vmin=0, vmax=100, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = valleys[(valleys.cost_year == matrix.index[i]) & (valleys.chronology == matrix.columns[j])].strict_valley_count.iloc[0]
            axb.text(j, i, f"{int(count)}/132", ha="center", va="center", color=INK, fontweight="bold")
    axb.set_xticks(range(2), ["clustered", "dispersed"], rotation=18)
    axb.set_yticks(range(3), matrix.index); axb.tick_params(length=0)
    axb.text(0.5, -0.28, "actual cost years", transform=axb.transAxes, ha="center", color=MUTED, fontsize=5.0)
    panel(axb, "b", -0.24)

    for chronology, colour, marker in [("dispersed", BLUE, "o"), ("clustered", ORANGE, "s")]:
        q = conditional[conditional.chronology == chronology].sort_values("duration_h")
        axe.fill_between(q.duration_h, q.q25, q.q75, color=mpl.colors.to_rgba(colour, 0.14), lw=0)
        axe.plot(q.duration_h, q["median"], color=colour, marker=marker, ms=2.7, lw=1.4,
                 mec="white", mew=0.25, label=chronology)
    axe.axhline(0, color=INK, lw=0.75); axe.set_xscale("log")
    axe.set_xlabel("LDES duration (h)"); axe.set_ylabel("Conditional incremental benefit\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axe.legend(loc="lower right"); clean(axe); panel(axe, "c", -0.07)

    boundary_observed = boundaries.right_boundary_h.notna()
    boundary_missing_count = int((~boundary_observed).sum())
    vals = boundaries.loc[boundary_observed, "right_boundary_h"].to_numpy()
    vp = axd.violinplot(vals, positions=[0], vert=False, widths=0.70, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(PURPLE); body.set_edgecolor("none"); body.set_alpha(0.22)
    axd.scatter(vals, deterministic_jitter(len(vals), 0.18), s=7, color=mpl.colors.to_rgba(PURPLE, .35), edgecolor="none")
    q1, med, q3 = np.quantile(vals, [.25, .5, .75])
    axd.plot([q1, q3], [0, 0], color=PURPLE, lw=4.0, solid_capstyle="round")
    axd.scatter(med, 0, s=36, color=INK, edgecolor="white", lw=.4, zorder=4)
    induced = int(states.loc[states.paired_state == "cluster_induced", "n_country_years"].iloc[0])
    axd.text(med, 0.28, f"median {med:.0f} h", ha="center", fontweight="bold", fontsize=5.3)
    axd.text(0.98, 0.92, f"{induced}/132 cluster-induced valleys", transform=axd.transAxes,
             ha="right", color=ORANGE, fontsize=5.1)
    axd.set_yticks([]); axd.set_xlabel("Long-duration re-entry boundary (h)")
    clean(axd, "x"); panel(axd, "d", -0.20)
    save_ed(fig, "fig_v54_ed5_conditional_phase_details")


def ed6() -> None:
    composition = pd.read_csv(SRC / "ed6a_bounded_procurement_composition.csv")
    rates = pd.read_csv(SRC / "ed6b_bounded_procurement_change_rates.csv")
    transitions = pd.read_csv(SRC / "ed6c_bounded_duration_transitions.csv")
    power = pd.read_csv(SRC / "ed6d_bounded_tail_power_changes.csv")
    fig = plt.figure(figsize=(7.20, 5.80))
    gs = GridSpec(2, 12, figure=fig, height_ratios=[0.95, 1.20], hspace=0.52, wspace=1.00,
                  left=0.08, right=0.985, top=0.96, bottom=0.10)
    axa = fig.add_subplot(gs[0, :6]); axb = fig.add_subplot(gs[0, 6:])
    trans_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :8], wspace=0.16)
    trans_axes = [fig.add_subplot(trans_grid[0, i]) for i in range(3)]
    axd = fig.add_subplot(gs[1, 9:])
    family_cols = {"No tail": "#D8E0E3", "4 h LFP": BLUE, "LDES": TEAL}
    labels = []
    for k, (year, chronology) in enumerate([(y, c) for y in [2025,2030,2035] for c in ["dispersed","clustered"]]):
        q = composition[(composition.learning_year == year) & (composition.chronology == chronology)]
        bottom = 0
        for family in ["No tail", "4 h LFP", "LDES"]:
            val = float(q.loc[q.choice_family == family, "share_percent"].iloc[0])
            axa.bar(k, val, bottom=bottom, color=family_cols[family], width=.74, edgecolor="white", lw=.35)
            bottom += val
        labels.append(f"{year}\n{'D' if chronology == 'dispersed' else 'C'}")
    axa.set_xticks(range(6), labels); axa.set_ylim(0,100); axa.set_ylabel("Selected portfolios (%)")
    axa.legend(handles=[Patch(color=family_cols[k], label=k) for k in family_cols], ncol=3, loc="upper left")
    clean(axa); panel(axa, "a", -0.12)

    metrics = ["identity_flip", "duration_flip", "participation_flip", "material_power_change"]
    metric_labels = ["identity", "duration", "participation", "tail power"]
    year_cols = {2025: BLUE, 2030: GOLD, 2035: PURPLE}
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        for offset, year in zip([-0.18, 0, 0.18], [2025,2030,2035]):
            row = rates[(rates.learning_year == year) & (rates.metric == metric)].iloc[0]
            y0 = i + offset
            axb.plot([row.ci_low_percent, row.ci_high_percent], [y0, y0], color=year_cols[year], lw=1.6)
            axb.scatter(row.rate_percent, y0, color=year_cols[year], s=24, marker={2025:"o",2030:"s",2035:"D"}[year],
                        edgecolor="white", lw=.35, zorder=3)
    axb.set_yticks(range(4), metric_labels); axb.set_xlabel("Paired country-years changed (%)")
    axb.legend(handles=[mpl.lines.Line2D([],[],marker={2025:"o",2030:"s",2035:"D"}[y],color=year_cols[y],lw=0,label=str(y))
                        for y in [2025,2030,2035]], ncol=3, loc="lower right", handletextpad=.2)
    clean(axb, "x"); panel(axb, "b", -0.18)

    duration_nodes = [0,4,24,60]
    node_labels = ["none","4","24","60"]
    for ax, year, letter in zip(trans_axes, [2025,2030,2035], ["c", "", ""]):
        q = transitions[transitions.learning_year == year]
        for row in q.itertuples():
            xi = duration_nodes.index(int(row.dispersed_decision_duration_h))
            yi = duration_nodes.index(int(row.clustered_decision_duration_h))
            ax.scatter(xi, yi, s=18 + 2.2 * row.country_year_count, color=mpl.colors.to_rgba(TEAL, .68),
                       edgecolor="white", lw=.4)
            if row.country_year_count < 120:
                ax.text(xi, yi, str(int(row.country_year_count)), ha="center", va="center", fontsize=5.0, color=INK)
        ax.plot([-0.3,3.3],[-0.3,3.3],color=INK,ls="--",lw=.6)
        ax.set_xlim(-.35,3.35); ax.set_ylim(-.35,3.35); ax.set_aspect("equal")
        ax.set_xticks(range(4),node_labels); ax.set_yticks(range(4),node_labels if year==2025 else [])
        ax.set_title(str(year), fontsize=5.7, color=year_cols[year], fontweight="bold")
        ax.set_xlabel("D duration (h)")
        if year==2025: ax.set_ylabel("C duration (h)")
        clean(ax, None)
        if letter: panel(ax, letter, -0.28)

    for i, year in enumerate([2025,2030,2035]):
        vals = power.loc[power.learning_year == year, "delta_tail_power_percentage_points"].to_numpy()
        vp = axd.violinplot(vals, positions=[i], widths=.72, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(year_cols[year]); body.set_edgecolor("none"); body.set_alpha(.22)
        axd.scatter(i + deterministic_jitter(len(vals), .16), vals, s=4.5,
                    color=mpl.colors.to_rgba(year_cols[year], .28), edgecolor="none")
        axd.scatter(i, np.median(vals), s=25, color=INK, edgecolor="white", lw=.35, zorder=3)
    axd.axhline(0,color=INK,lw=.7); axd.set_xticks(range(3),["2025","2030","2035"])
    axd.set_ylabel("C - D tail power (pp)")
    clean(axd); panel(axd, "d", -0.28)
    save_ed(fig, "fig_v54_ed6_bounded_joint_procurement")


def ed7() -> None:
    shares = pd.read_csv(SRC / "ed7_capacity_and_dispatch_shares.csv")
    cycles = pd.read_csv(SRC / "ed7_cycle_distributions.csv")
    allocation = pd.read_csv(SRC / "ed7_dispatch_allocation.csv")
    technology = pd.read_csv(SRC / "ed7_dominant_technology_shares.csv")
    values = pd.read_csv(SRC / "ed7_value_component_decomposition.csv")
    fig = plt.figure(figsize=(7.20, 5.75))
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.20, 0.92, 0.92], hspace=0.58, wspace=1.00,
                  left=0.08, right=0.985, top=0.965, bottom=0.09)
    axa = fig.add_subplot(gs[0, :])
    metric_specs = [
        ("tail_energy_share", "energy capacity", PURPLE, "-"),
        ("tail_task_share", "task service", BLUE, "-"),
        ("tail_charge_share", "charging", TEAL, ":"),
        ("tail_power_share", "power capacity", MUTED, "--"),
    ]
    for metric, label, colour, ls in metric_specs:
        for chronology, marker, alpha in [("dispersed","o",.95),("clustered","s",.72)]:
            q = shares[shares._chronology == chronology].sort_values("duration_h")
            axa.plot(q.duration_h, 100*q[metric], color=colour, ls=ls, alpha=alpha, lw=1.25,
                     marker=marker, ms=2.3, mec="white", mew=.2,
                     label=f"{label} - {chronology}" if chronology=="dispersed" else None)
    axa.set_xscale("log"); axa.set_ylim(0,104)
    axa.set_xlabel("LDES duration (h)"); axa.set_ylabel("Tail-layer share (%)")
    axa.text(.99,.93,"circles: dispersed    squares: clustered",transform=axa.transAxes,ha="right",color=MUTED,fontsize=5.0)
    axa.legend(ncol=4,loc="lower right",handlelength=1.4,columnspacing=.8)
    clean(axa); panel(axa,"a",-.06)

    axb = fig.add_subplot(gs[1, :5])
    groups = [("4 h LFP","dispersed",BLUE),("4 h LFP","clustered",ORANGE),
              ("LDES tail","dispersed",TEAL),("LDES tail","clustered",PURPLE)]
    for i,(component,chronology,colour) in enumerate(groups):
        vals=cycles[(cycles.component==component)&(cycles.chronology==chronology)].equivalent_full_cycles.to_numpy()
        vp=axb.violinplot(vals,positions=[i],vert=False,widths=.68,showextrema=False,points=60)
        for body in vp["bodies"]:
            body.set_facecolor(colour);body.set_edgecolor("none");body.set_alpha(.20)
        q1,med,q3=np.quantile(vals,[.25,.5,.75]);axb.plot([q1,q3],[i,i],color=colour,lw=3.0,solid_capstyle="round")
        axb.scatter(med,i,s=25,color=INK,edgecolor="white",lw=.35,zorder=3)
    axb.set_yticks(range(4),["LFP D","LFP C","LDES D","LDES C"]);axb.invert_yaxis()
    axb.set_xlabel("Equivalent full cycles yr$^{-1}$");clean(axb,"x");panel(axb,"b",-.18)

    axc=fig.add_subplot(gs[1,6:])
    for i,row in enumerate(allocation.itertuples()):
        axc.barh(i,row.base_share_pct,color=BLUE,height=.62)
        axc.barh(i,row.tail_share_pct,left=row.base_share_pct,color=TEAL,height=.62)
        axc.text(row.base_share_pct/2,i,f"{row.base_share_pct:.0f}%",ha="center",va="center",color="white",fontweight="bold")
        axc.text(row.base_share_pct+row.tail_share_pct/2,i,f"{row.tail_share_pct:.0f}%",ha="center",va="center",color="white",fontweight="bold")
    axc.set_yticks(range(len(allocation)),allocation.label);axc.set_xlim(0,100);axc.set_xlabel("Allocation share (%)")
    axc.legend(handles=[Patch(color=BLUE,label="4 h LFP"),Patch(color=TEAL,label="LDES tail")],ncol=2,loc="lower right")
    clean(axc,"x");panel(axc,"c",-.17)

    axd=fig.add_subplot(gs[2,:5])
    techs=["vrfb","caes","h2_cavern"]
    for i,chronology in enumerate(["dispersed","clustered"]):
        left=0
        for tech in techs:
            val=float(technology[(technology._chronology==chronology)&(technology.dominant_tech_id==tech)].share_pct.iloc[0])
            axd.barh(i,val,left=left,color=TECH_COLOURS[tech],height=.60,edgecolor="white",lw=.35)
            left+=val
    axd.set_yticks([0,1],["dispersed","clustered"]);axd.set_xlim(0,100);axd.set_xlabel("Dominant configuration share (%)")
    axd.legend(handles=[Patch(color=TECH_COLOURS[t],label=TECH_LABELS[t]) for t in techs],ncol=3,loc="lower right")
    clean(axd,"x");panel(axd,"d",-.18)

    axe=fig.add_subplot(gs[2,6:])
    component_labels={
        "fixed_cost_contribution":"fixed cost","variable_om_contribution":"variable O&M",
        "charging_cost_contribution":"charging","energy_service_contribution":"energy service",
        "firm_service_contribution":"firm service"}
    y=np.arange(len(values))
    for offset,(chronology,colour) in zip([-.14,.14],[('dispersed',BLUE),('clustered',ORANGE)]):
        vals=values[chronology].to_numpy();axe.barh(y+offset,vals,height=.24,color=colour,alpha=.82,label=chronology)
    axe.axvline(0,color=INK,lw=.75);axe.set_yticks(y,[component_labels[c] for c in values.component]);axe.invert_yaxis()
    axe.set_xlabel("Median value contribution (EUR kW$^{-1}$ yr$^{-1}$)");axe.legend(loc="lower right")
    clean(axe,"x");panel(axe,"e",-.17)
    save_ed(fig,"fig_v54_ed7_fixed_scale_mix_details")


def ed8() -> None:
    retention=pd.read_csv(SRC/"ed8_tail_throughput_retention.csv")
    binding=pd.read_csv(SRC/"ed8_binding_hour_diagnostics.csv")
    recharge=pd.read_csv(SRC/"ed8_recharge_quantity_multiplier_response.csv")
    economic=pd.read_csv(SRC/"ed8_energy_charge_value_sensitivity.csv")
    firm=pd.read_csv(SRC/"ed8_firm_value_cost_response.csv")
    fig=plt.figure(figsize=(7.20,5.75))
    gs=GridSpec(2,12,figure=fig,height_ratios=[.95,1.18],hspace=.52,wspace=1.00,
                left=.08,right=.985,top=.96,bottom=.10)
    axa=fig.add_subplot(gs[0,:4]);axb=fig.add_subplot(gs[0,4:8]);axc=fig.add_subplot(gs[0,8:])
    axd=fig.add_subplot(gs[1,:5]);axe=fig.add_subplot(gs[1,5:])
    configs=[("caes_24h",BLUE,"A-CAES 24 h"),("h2_cavern_48h",PURPLE,"Cavern H$_2$ 48 h")]
    metrics=["Charging","Task service","Equivalent cycles"]
    for j,(config,colour,label) in enumerate(configs):
        q=retention[retention.config_id==config].set_index("metric_label").reindex(metrics)
        x=np.arange(3)+(.11 if j else -.11)
        axa.plot(x,q.clustered_to_dispersed_pct,color=colour,lw=1.2,marker="D" if j else "o",ms=4.2,label=label)
    axa.axhline(100,color=INK,ls="--",lw=.7);axa.set_xticks(range(3),["charge","service","cycles"])
    axa.set_ylabel("C / D retention (%)");axa.legend(loc="lower left",fontsize=5.0);clean(axa);panel(axa,"a",-.20)

    for i,metric in enumerate(["shared_recharge_binding_hours","tail_charge_power_binding_hours"]):
        for config,colour,label in configs:
            q=binding[(binding.config_id==config)&(binding.metric==metric)]
            d=q[q.chronology=="dispersed"].iloc[0];c=q[q.chronology=="clustered"].iloc[0]
            y0=i+(-.12 if config=="caes_24h" else .12)
            axb.annotate("",xy=(c.country_block_mean,y0),xytext=(d.country_block_mean,y0),
                         arrowprops=dict(arrowstyle="-|>",color=colour,lw=1.5,mutation_scale=8))
            axb.scatter([d.country_block_mean,c.country_block_mean],[y0,y0],s=24,color=colour,edgecolor="white",lw=.35)
        
    axb.set_xscale("log");axb.set_yticks([0,1],["surplus ceiling","charge interface"]);axb.set_xlabel("Binding hours yr$^{-1}$")
    clean(axb,"x");panel(axb,"b",-.23)

    for config,colour,label in configs:
        q=recharge[recharge.config_id==config].sort_values("recharge_quantity_multiplier")
        axc.fill_between(q.recharge_quantity_multiplier,q.country_block_ci_low,q.country_block_ci_high,
                         color=mpl.colors.to_rgba(colour,.12),lw=0)
        axc.plot(q.recharge_quantity_multiplier,q.country_block_mean,color=colour,lw=1.4,marker="o",ms=3.8,label=label)
    axc.axhline(0,color=INK,lw=.7);axc.set_xticks([1,2,4],["1x","2x","4x"])
    axc.set_xlabel("Recharge multiplier");axc.set_ylabel("C - D value");axc.legend(loc="lower left",fontsize=5.0)
    clean(axc);panel(axc,"c",-.20)

    pivot=economic.pivot(index="charge_price_eur_mwh",columns="energy_value_eur_mwh",values="clustered_minus_dispersed")
    im=axd.imshow(pivot.to_numpy(),origin="lower",aspect="auto",cmap=CMAP_DIV)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            axd.text(j,i,f"{pivot.iloc[i,j]:.1f}",ha="center",va="center",color="white" if pivot.iloc[i,j]<-6 else INK,fontweight="bold")
    axd.set_xticks(range(pivot.shape[1]),[f"{v:.0f}" for v in pivot.columns]);axd.set_yticks(range(pivot.shape[0]),[f"{v:.2g}" for v in pivot.index])
    axd.set_xlabel("Energy value (EUR MWh$^{-1}$)");axd.set_ylabel("Charge price (EUR MWh$^{-1}$)");axd.tick_params(length=0)
    cax=axd.inset_axes([.18,1.03,.62,.055]);cb=fig.colorbar(im,cax=cax,orientation="horizontal");cb.set_label("C - D value",fontsize=5.0)
    panel(axd,"d",-.15)

    for mult,colour in zip(sorted(firm.ldes_tail_fixed_cost_multiplier.unique()),CMAP_SEQ(np.linspace(.08,.92,5))):
        q=firm[firm.ldes_tail_fixed_cost_multiplier==mult].sort_values("firm_value_eur_kw_year")
        axe.plot(q.firm_value_eur_kw_year,100*q.cluster_induced_rate,color=colour,lw=1.25,marker="o",ms=2.2,label=f"{mult:.1f}x cost")
    axe.set_xlabel("Firm-supply unit value (EUR kW$^{-1}$ yr$^{-1}$)");axe.set_ylabel("Cluster-induced valleys (%)")
    axe.legend(ncol=5,loc="upper right",handlelength=1.0,columnspacing=.6,fontsize=5.0);clean(axe);panel(axe,"e",-.10)
    save_ed(fig,"fig_v54_ed8_mix_sensitivity")


def european_map(ax, values, cmap, norm, missing="#EEF2F3") -> None:
    reader=shapefile.Reader(str(WORLD_SHP),encoding="utf-8")
    for sr in reader.iterShapeRecords():
        code=sr.record.as_dict().get("ISO_A2")
        face=cmap(norm(values[code])) if code in values and np.isfinite(values[code]) else missing
        add_shape(ax,sr.shape,face,"white",.18,2 if code in values else 1)
    ax.set_xlim(-12,32);ax.set_ylim(34,72);ax.set_aspect("equal");ax.axis("off")


def ed9() -> None:
    mapping=pd.read_csv(SRC/"ed9_four_day_country_crossing_map.csv")
    status=pd.read_csv(SRC/"ed9_crossing_censoring_status.csv")
    summary=pd.read_csv(SRC/"ed9_country_crossing_summary.csv")
    records=pd.read_csv(SRC/"figure5_panel_e_zero_crossing_records.csv")
    natural=pd.read_csv(SRC/"figure2_panel_b_country_year_geometry.csv")
    fig=plt.figure(figsize=(7.20,5.75))
    gs=GridSpec(2,12,figure=fig,height_ratios=[1.18,1.0],hspace=.44,wspace=.95,
                left=.07,right=.985,top=.96,bottom=.10)
    axa=fig.add_subplot(gs[0,:7]);axb=fig.add_subplot(gs[0,8:])
    axc=fig.add_subplot(gs[1,:7]);axd=fig.add_subplot(gs[1,8:])
    cmap=mpl.colors.LinearSegmentedColormap.from_list("crossing",[BLUE,LIGHT,GOLD,CORAL]);norm=mpl.colors.Normalize(36,144)
    european_map(axa,mapping.set_index("country").median_crossing_h.to_dict(),cmap,norm)
    cax=axa.inset_axes([.18,.05,.50,.045]);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal")
    cb.set_label("Four-day controlled crossing (h)",fontsize=5.0);cb.ax.tick_params(labelsize=5.0,length=1.3)
    panel(axa,"a",0.0,.98)

    status_cols={"left_censored":"#B9C4C9","observed":TEAL,"right_censored":ORANGE}
    for i,days in enumerate([2,4,6]):
        left=0
        for state in ["left_censored","observed","right_censored"]:
            q=status[(status.target_cluster_days==days)&(status.crossing_status==state)]
            val=float(q.share_pct.iloc[0]) if len(q) else 0
            axb.barh(i,val,left=left,color=status_cols[state],height=.62,edgecolor="white",lw=.35)
            if val>3: axb.text(left+val/2,i,f"{val:.0f}%",ha="center",va="center",color="white" if state=="observed" else INK,fontweight="bold")
            left+=val
    axb.set_yticks(range(3),["2 d","4 d","6 d"]);axb.set_xlim(0,100);axb.set_xlabel("Country-years (%)")
    axb.legend(handles=[Patch(color=status_cols[k],label=k.replace("_"," ")) for k in status_cols],loc="lower left",fontsize=5.0)
    clean(axb,"x");panel(axb,"b",-.24)

    countries=list(mapping.sort_values("median_crossing_h").country)
    colours={2:BLUE,4:ORANGE,6:PURPLE}
    for i,country in enumerate(countries):
        for days,offset in [(2,-.17),(4,0),(6,.17)]:
            row=summary[(summary.country==country)&(summary.target_cluster_days==days)].iloc[0]
            axc.plot([row.q25_crossing_h,row.q75_crossing_h],[i+offset,i+offset],color=colours[days],lw=1.4)
            axc.scatter(row.median_crossing_h,i+offset,s=16,color=colours[days],marker={2:"o",4:"s",6:"D"}[days],edgecolor="white",lw=.3)
    axc.set_yticks(range(len(countries)),countries);axc.set_xlabel("First sustained positive duration (h)")
    axc.legend(handles=[mpl.lines.Line2D([],[],marker={2:"o",4:"s",6:"D"}[d],color=colours[d],lw=0,label=f"{d} d") for d in [2,4,6]],ncol=3,loc="lower right")
    clean(axc,"x");panel(axc,"c",-.10)

    q=records[records.target_cluster_days==4].merge(natural[["country","year","natural_longest_event_run_days"]],on=["country","year"],how="left")
    q=q[np.isfinite(q.zero_crossing_duration_h)&np.isfinite(q.natural_longest_event_run_days)]
    x=q.natural_longest_event_run_days.to_numpy();y=q.zero_crossing_duration_h.to_numpy()
    axd.scatter(x,y,s=12,color=mpl.colors.to_rgba(ORANGE,.46),edgecolor="white",lw=.25)
    fit=np.polyfit(x,y,1);xx=np.linspace(x.min(),x.max(),80);axd.plot(xx,np.polyval(fit,xx),color=INK,lw=1.1)
    rho=pd.Series(x).corr(pd.Series(y),method="spearman")
    axd.text(.05,.92,f"Spearman rho = {rho:+.2f}",transform=axd.transAxes,fontweight="bold",fontsize=5.2)
    axd.set_xlabel("Unrearranged longest event (d)");axd.set_ylabel("Controlled crossing (h)")
    clean(axd);panel(axd,"d",-.20)
    save_ed(fig,"fig_v54_ed9_mix_heterogeneity")


def ed10() -> None:
    deployment=pd.read_csv(SRC/"ed10_eraa_candidate_deployment.csv")
    screen=pd.read_csv(SRC/"ed10_eraa_system_screen.csv")
    operation=pd.read_csv(SRC/"ed10_eraa_operational_changes.csv")
    fig=plt.figure(figsize=(7.20,5.05))
    gs=GridSpec(2,12,figure=fig,height_ratios=[1.15,.90],hspace=.50,wspace=.90,
                left=.07,right=.985,top=.96,bottom=.11)
    axa=fig.add_subplot(gs[:, :7])
    reader=shapefile.Reader(str(WORLD_SHP),encoding="utf-8")
    target=set(deployment.country)
    power=deployment.set_index("country").candidate_power_gw.to_dict()
    norm=mpl.colors.Normalize(0,max(power.values()));cmap=mpl.colors.LinearSegmentedColormap.from_list("eraa",[LIGHT,SAGE,"#4F7765"])
    for sr in reader.iterShapeRecords():
        code=sr.record.as_dict().get("ISO_A2")
        face=cmap(norm(power[code])) if code in target and code in power else "#EEF2F3"
        add_shape(axa,sr.shape,face,"white",.18,2 if code in target else 1)
    location_observed = deployment[["lon", "lat"]].notna().all(axis=1)
    location_missing_count = int((~location_observed).sum())
    valid=deployment.loc[location_observed]
    axa.scatter(valid.lon,valid.lat,s=18+115*np.sqrt(valid.candidate_power_gw/max(valid.candidate_power_gw.max(),1e-9)),
                facecolor="white",edgecolor=INK,lw=.45,zorder=5)
    axa.set_xlim(-12,32);axa.set_ylim(34,72);axa.set_aspect("equal");axa.axis("off")
    axa.text(.03,.06,f"{deployment.candidate_power_gw.sum():.3f} GW fixed candidate tranche",transform=axa.transAxes,color=TEAL,fontweight="bold",fontsize=5.6)
    cax=axa.inset_axes([.18,.02,.48,.035]);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal")
    cb.set_label("Candidate A-CAES power (GW)",fontsize=5.0);panel(axa,"a",0,.98)

    mini=GridSpecFromSubplotSpec(1,3,subplot_spec=gs[0,7:],wspace=.75)
    for ax,row,letter in zip([fig.add_subplot(mini[0,i]) for i in range(3)],screen.itertuples(),"bcd"):
        ax.plot([0,1],[row.baseline,row.with_fixed_4p242_gw_caes],color=MUTED,lw=1.5)
        ax.scatter(0,row.baseline,s=38,color="#AEBBC1",edgecolor="white",lw=.4,zorder=3)
        ax.scatter(1,row.with_fixed_4p242_gw_caes,s=38,color=TEAL,marker="s",edgecolor="white",lw=.4,zorder=3)
        ax.text(.5,max(row.baseline,row.with_fixed_4p242_gw_caes)*1.01,f"{row.relative_change_pct:+.1f}%",ha="center",color=TEAL,fontweight="bold",fontsize=5.0)
        ax.set_xticks([0,1],["base","+A-CAES"],rotation=25,ha="right")
        ax.set_ylabel(f"{row.metric}\n({row.unit})");clean(ax);panel(ax,letter,-.28)

    axe=fig.add_subplot(gs[1,7:])
    vals=operation.change_gwh.to_numpy();labels=operation.metric.to_list();cols=[TEAL,BLUE,ORANGE]
    y=np.arange(len(vals));axe.barh(y,vals,color=cols,alpha=.86)
    axe.axvline(0,color=INK,lw=.75);axe.set_yticks(y,labels);axe.invert_yaxis();axe.set_xlabel("Operational change (GWh yr$^{-1}$)")
    axe.margins(x=.12)
    for i,v in enumerate(vals):axe.text(v+(240 if v>=0 else -240),i,f"{v:+,.0f}",ha="left" if v>=0 else "right",va="center",color=cols[i],fontweight="bold",fontsize=5.0)
    clean(axe,"x");panel(axe,"e",-.20)
    save_ed(fig,"fig_v54_ed10_eraa_system_screen")


def main() -> None:
    configure()
    ed1(); ed2(); ed3(); ed4(); ed5(); ed6(); ed7(); ed8(); ed9(); ed10()
    manifest={
        "backend":"python-matplotlib",
        "figures":[f"fig_v54_ed{i}" for i in range(1,11)],
        "note":"All Extended Data figures were rebuilt from registered v54 source tables."
    }
    (HERE/"manifests/v54_extended_figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    main()
