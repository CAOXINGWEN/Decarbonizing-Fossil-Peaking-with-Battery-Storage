#!/usr/bin/env python3
"""Rebuild v54 Figures 2-6 with a parity-led Nature page layout."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle, Patch, Polygon, Rectangle, Wedge

from plot_v51_main_figures import (
    CHINA_GEO,
    JAPAN_GEO_ROOT,
    WORLD_SHP,
    add_geojson,
    add_shape,
    deterministic_jitter,
    duration_edges,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "source_data"
SRC49 = HERE / "source_data_v49"
OUT = HERE / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#3D3A36"
MUTED = "#77736D"
GRID = "#E5E2DC"
LIGHT = "#F5F3EF"
CORAL = "#D98678"
PEACH = "#EAB59A"
BLUE = "#76A9C5"
BLUE_DARK = "#4B83A4"
SAGE = "#72A190"
GREEN = "#688F69"
GOLD = "#D9AD52"
TAUPE = "#9A8065"
GREY = "#B9B5AE"


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.1,
        "axes.labelsize": 6.2,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.2,
        "legend.fontsize": 5.0,
        "axes.linewidth": 0.55,
        "lines.linewidth": 1.0,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def label_panel(ax, letter: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.0,
            fontweight="bold", va="bottom", color=INK)


def quiet(ax, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.45)
        ax.set_axisbelow(True)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=450)
    fig.savefig(path.with_suffix(".tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(OUT / f"{stem}_grayscale.png", grey, cmap="gray", vmin=0, vmax=1)
    plt.close(fig)


def ecdf(ax, values, colour, label=None, lw=1.25) -> None:
    values = np.sort(np.asarray(values, float))
    y = np.arange(1, len(values) + 1) / len(values) * 100
    ax.step(values, y, where="post", color=colour, lw=lw, label=label)


def interval_mark(ax, values, y, colour, marker="o", raw=True, width=0.12) -> None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if raw:
        ax.scatter(values, y + deterministic_jitter(len(values), width), s=5.2,
                   color=mpl.colors.to_rgba(colour, 0.26), edgecolor="none", rasterized=True)
    q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    ax.plot([q1, q3], [y, y], color=colour, lw=2.7, solid_capstyle="round")
    ax.scatter(med, y, s=27, marker=marker, color=colour, edgecolor="white", lw=0.45, zorder=5)


def density_curve(values, bins=90, smooth=5):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    lo, hi = values.min(), values.max()
    if np.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi), density=True)
    kernel_x = np.arange(-smooth * 3, smooth * 3 + 1)
    kernel = np.exp(-0.5 * (kernel_x / max(smooth, 1)) ** 2)
    kernel /= kernel.sum()
    hist = np.convolve(hist, kernel, mode="same")
    centres = (edges[:-1] + edges[1:]) / 2
    if hist.max() > 0:
        hist /= hist.max()
    return centres, hist


def ridge(ax, values, y, colour, height=0.68, label=None) -> None:
    x, d = density_curve(values)
    ax.fill_between(x, y, y + height * d, color=mpl.colors.to_rgba(colour, 0.30), lw=0)
    ax.plot(x, y + height * d, color=colour, lw=0.9)
    med = float(np.nanmedian(values))
    ax.scatter(med, y + height * np.interp(med, x, d), s=18, color=colour,
               edgecolor="white", lw=0.35, zorder=4)
    if label:
        ax.text(x.min(), y + 0.10, label, ha="left", va="bottom", color=MUTED, fontsize=5.0)


def ring_summary(ax, fraction, colour, centre=(0.5, 0.5), radius=0.32, text="") -> None:
    ax.add_patch(Wedge(centre, radius, 0, 360, width=radius * 0.26,
                       facecolor="#E7E3DC", edgecolor="none"))
    ax.add_patch(Wedge(centre, radius, 90, 90 + 360 * fraction, width=radius * 0.26,
                       facecolor=colour, edgecolor="none"))
    ax.text(*centre, text, ha="center", va="center", color=colour,
            fontsize=5.4, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")


def triangle_quantity(ax, y, width, colour, label, value) -> None:
    x0 = 0.50
    h = 0.15
    poly = Polygon([(x0 - width / 2, y), (x0 + width / 2, y),
                    (x0 + width * 0.40, y + h), (x0 - width * 0.40, y + h)],
                   closed=True, facecolor=colour, edgecolor="white", lw=0.45)
    ax.add_patch(poly)
    ax.text(0.02, y + h / 2, label, ha="left", va="center", color=INK, fontsize=5.1)
    ax.text(x0, y + h / 2, value, ha="center", va="center", color="white",
            fontsize=5.0, fontweight="bold")


def europe_map(ax, values=None, cmap=None, norm=None, edge="#FFFFFF", missing="#F0F1EE") -> None:
    reader = shapefile.Reader(str(WORLD_SHP), encoding="utf-8")
    values = values or {}
    for sr in reader.iterShapeRecords():
        code = sr.record.as_dict().get("ISO_A2")
        if code in values and np.isfinite(values[code]):
            face = cmap(norm(values[code])) if cmap is not None else BLUE
            z = 3
        else:
            face = missing
            z = 1
        add_shape(ax, sr.shape, face, edge, 0.20, z)
    ax.set_xlim(-12, 32); ax.set_ylim(34, 72); ax.set_aspect("equal"); ax.axis("off")


def china_map(ax, values, cmap, norm) -> None:
    zh = {"Anhui":"安徽省","Beijing":"北京市","Chongqing":"重庆市","Fujian":"福建省","Gansu":"甘肃省",
          "Guangdong":"广东省","Guangxi":"广西壮族自治区","Guizhou":"贵州省","Hainan":"海南省","Hebei":"河北省",
          "Heilongjiang":"黑龙江省","Henan":"河南省","Hubei":"湖北省","Hunan":"湖南省","Inner Mongolia":"内蒙古自治区",
          "Jiangsu":"江苏省","Jiangxi":"江西省","Jilin":"吉林省","Liaoning":"辽宁省","Ningxia":"宁夏回族自治区",
          "Qinghai":"青海省","Shaanxi":"陕西省","Shandong":"山东省","Shanxi":"山西省","Sichuan":"四川省",
          "Zhejiang":"浙江省","Tianjin":"天津市","Xinjiang":"新疆维吾尔自治区","Yunnan":"云南省"}
    lookup = {zh[k]: v for k, v in values.items() if k in zh}
    geo = json.loads(CHINA_GEO.read_text(encoding="utf-8"))
    for feat in geo["features"]:
        v = lookup.get(feat["properties"].get("name"))
        add_geojson(ax, feat["geometry"], cmap(norm(v)) if v is not None else "#F0F1EE",
                    edge="white", lw=0.20)
    ax.set_xlim(73, 136); ax.set_ylim(17, 54); ax.set_aspect("equal"); ax.axis("off")


def japan_map(ax, points, colour_values, cmap, norm) -> None:
    bounds = (137.55, 140.15, 34.45, 36.05)
    for code in (11, 12, 13, 14, 22):
        path = JAPAN_GEO_ROOT / f"N03-20240101_{code}_GML/N03-20240101_{code}.geojson"
        for feat in json.loads(path.read_text(encoding="utf-8"))["features"]:
            add_geojson(ax, feat["geometry"], "#F0F1EE", edge="#D4D2CC", lw=0.14, bounds=bounds)
    ax.scatter(points.longitude, points.latitude, c=colour_values, cmap=cmap, norm=norm,
               s=14, edgecolor="white", lw=0.35, zorder=5)
    ax.set_xlim(*bounds[:2]); ax.set_ylim(*bounds[2:]); ax.set_aspect("equal"); ax.axis("off")


def figure2() -> None:
    eu = pd.read_csv(SRC49 / "figure2a_europe_map.csv")
    cn = pd.read_csv(SRC49 / "figure2b_china_map.csv")
    jp = pd.read_csv(SRC49 / "figure2c_japan_map.csv")
    runs = pd.read_csv(SRC49 / "figure2d_three_scale_event_runs.csv")
    eu_cy = pd.read_csv(SRC / "figure2_panel_b_country_year_geometry.csv")
    chronology = pd.read_csv(SRC / "figure2_panel_c_conserved_chronology.csv")
    recovery = pd.read_csv(SRC / "figure2_panel_f_recovery_window_ecdf.csv")
    correlations = pd.read_csv(ROOT / "sequencing_parity/three_regime_task_event_correlations.csv")

    fig = plt.figure(figsize=(7.20, 7.48))
    gs = GridSpec(4, 12, figure=fig, height_ratios=[1.45, 1.12, 1.12, 1.20],
                  hspace=0.43, wspace=0.75, left=0.07, right=0.985, top=0.97, bottom=0.06)

    top = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, :], height_ratios=[4.2, 1.0],
                                  hspace=0.05, wspace=0.20)
    axes_maps = [fig.add_subplot(top[0, i]) for i in range(3)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("event", [BLUE, "#F3EEE6", CORAL])
    norm = mpl.colors.Normalize(4, 14)
    europe_map(axes_maps[0], eu.set_index("gisco_code").median_longest_event_days.to_dict(), cmap, norm)
    china_map(axes_maps[1], cn.set_index("province").longest_extreme_peaking_run_days.to_dict(), cmap, norm)
    comm = jp.groupby(["mesh_id", "latitude", "longitude"], as_index=False).agg(
        event=("longest_extreme_import_run_days", "median"))
    japan_map(axes_maps[2], comm, comm.event, cmap, norm)
    axes_maps[0].set_anchor("W")
    axes_maps[1].set_anchor("C")
    axes_maps[2].set_anchor("E")
    for ax, title, sub in zip(axes_maps,
                              ["S1 European national", "S2 Chinese provincial", "S3 Japanese community"],
                              ["132 country-years", "26 provinces", "27 community-years"]):
        ax.set_title(f"{title}\n{sub}", fontsize=5.3, pad=1.5, color=INK, linespacing=1.25)
    label_panel(axes_maps[0], "a", -0.08, 1.02)
    cax = axes_maps[1].inset_axes([0.25, 0.080, 0.50, 0.028])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("Longest event (d)", fontsize=5.0, labelpad=1); cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.0, length=1.5)

    strip = fig.add_subplot(top[1, :])
    regime_cols = {"Europe": CORAL, "China": GOLD, "Japan": BLUE}
    for i, name in enumerate(["Europe", "China", "Japan"]):
        vals = runs.loc[runs.scale == name, "event_run_days"].to_numpy()
        q1, med, q3 = np.quantile(vals, [.25, .5, .75])
        strip.plot([i - .24, i + .24], [med, med], color=regime_cols[name], lw=0.8)
        strip.plot([i, i], [q1, q3], color=regime_cols[name], lw=1.1)
        strip.scatter(i, med, s=24, marker=["o", "s", "^"][i], color=regime_cols[name],
                      edgecolor="white", lw=.35, zorder=4)
    strip.set_xticks(range(3), ["Europe", "China", "Japan"]); strip.set_ylabel("event run (d)")
    quiet(strip, "y")

    bgrid = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1, :8], hspace=0.26, wspace=0.42)
    regime_data = [
        (runs.loc[runs.scale == "Europe", "event_run_days"], eu_cy.recharge_to_task_ratio, CORAL, "Europe"),
        (cn.longest_extreme_peaking_run_days, cn.recharge_to_peaking_ratio, GOLD, "China"),
        (jp.longest_extreme_import_run_days, jp.pv_reverse_flow_to_task_ratio, BLUE, "Japan"),
    ]
    for j, (event, ratio, colour, name) in enumerate(regime_data):
        for r, (vals, xlabel) in enumerate([(event, "event run (d)"), (ratio, "recharge / task")]):
            if r == 1 and np.any(np.asarray(vals) <= 0):
                raise ValueError(f"{name} recharge/task contains nonpositive values")
            ax = fig.add_subplot(bgrid[r, j])
            interval_mark(ax, vals, 0, colour, marker=["o", "s", "^"][j], raw=True, width=.11)
            ax.axvline(np.median(vals), color=INK, ls="--", lw=.55)
            if r == 1:
                ax.set_xscale("log")
            ax.set_yticks([]); ax.set_xlabel(xlabel if r else "")
            if r == 0: ax.set_title(name, color=colour, fontsize=5.5, pad=2)
            quiet(ax, "x")
            if j == 0 and r == 0: label_panel(ax, "b", -.28, 1.14)

    axc = fig.add_subplot(gs[1:3, 8:])
    x = (eu_cy.event_task_share_pct - eu_cy.event_task_share_pct.mean()) / eu_cy.event_task_share_pct.std()
    if np.any(eu_cy.recharge_to_task_ratio <= 0):
        raise ValueError("European recharge/task contains nonpositive values")
    logr = np.log(eu_cy.recharge_to_task_ratio)
    y = (logr - logr.mean()) / logr.std()
    high = eu_cy.natural_longest_event_run_days >= eu_cy.natural_longest_event_run_days.median()
    axc.scatter(x[~high], y[~high], s=9, color=mpl.colors.to_rgba(GREY, .34), edgecolor="none", rasterized=True)
    axc.scatter(x[high], y[high], s=10, color=mpl.colors.to_rgba(CORAL, .40), edgecolor="none", rasterized=True)
    fit = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 100)
    axc.plot(xx, np.polyval(fit, xx), color=INK, lw=1.05)
    axc.axhline(0, color=INK, ls="--", lw=.55); axc.axvline(0, color=INK, ls="--", lw=.55)
    axc.text(.04, .94, "event-dependent", transform=axc.transAxes, color=CORAL, va="top", fontsize=5.1)
    axc.text(.97, .94, "recharge-buffered", transform=axc.transAxes, color=GREEN, va="top", ha="right", fontsize=5.1)
    axc.set_xlabel("Event concentration (z)")
    axc.set_ylabel("Recharge access (z)")
    quiet(axc); label_panel(axc, "c", -.10, 1.02)

    axd = fig.add_subplot(gs[2, :4])
    shares = [eu_cy.event_task_share_pct.median(), cn.extreme_peaking_share_pct.median(),
              jp.extreme_import_share_pct.median()]
    for i, (name, share, colour) in enumerate(zip(["Europe", "China", "Japan"], shares, [CORAL, GOLD, BLUE])):
        axd.barh(i, share, color=mpl.colors.to_rgba(colour, .72), height=.58, edgecolor="white", lw=.4)
        axd.barh(i, 100-share, left=share, color="#E8E5DE", height=.58, edgecolor="white", lw=.4, hatch="//")
        axd.text(share/2, i, f"{share:.0f}% event", ha="center", va="center", color="white", fontweight="bold", fontsize=5.0)
    axd.set_yticks(range(3), ["Europe", "China", "Japan"]); axd.set_xlim(0,100)
    axd.set_xlabel("Annual task composition (%)"); axd.invert_yaxis(); quiet(axd, "x")
    label_panel(axd, "d", -.22, 1.03)

    axe = fig.add_subplot(gs[2, 4:8])
    frames = []
    for vals, name in [(eu_cy[["annual_task_twh","natural_longest_event_run_days"]], "Europe"),
                       (cn[["annual_observed_peaking_task_gwh","longest_extreme_peaking_run_days"]], "China"),
                       (jp[["annual_residual_import_mwh","longest_extreme_import_run_days"]], "Japan")]:
        q = vals.copy(); q.columns = ["task","event"]; q["regime"] = name; frames.append(q)
    rank = pd.concat(frames, ignore_index=True); rank["pct"] = rank.groupby("regime").task.rank(pct=True)*100
    for name, colour, marker in [("Europe",CORAL,"o"),("China",GOLD,"s"),("Japan",BLUE,"^")]:
        q=rank[rank.regime==name]
        axe.scatter(q.pct,q.event,s=8,color=mpl.colors.to_rgba(colour,.48),marker=marker,edgecolor="none",label=name)
    axe.set_xlabel("Within-regime annual-task percentile"); axe.set_ylabel("Longest event (d)")
    axe.legend(ncol=3, loc="upper right", handletextpad=.2, columnspacing=.6)
    quiet(axe); label_panel(axe,"e",-.18,1.03)

    axf = fig.add_subplot(gs[3, :3])
    rho = correlations.set_index("regime").spearman_rho_annual_task_vs_longest_event
    meds = runs.groupby("scale").event_run_days.median()
    for name, colour, marker in [("Europe",CORAL,"o"),("China",GOLD,"s"),("Japan",BLUE,"^")]:
        axf.plot([0,rho[name]],[0,meds[name]],color=mpl.colors.to_rgba(colour,.45),lw=.9,ls="--")
        axf.scatter(rho[name],meds[name],s=34,color=colour,marker=marker,edgecolor="white",lw=.45,zorder=4)
        axf.text(rho[name],meds[name]+.65,name,ha="center",color=colour,fontsize=5.0)
    axf.axvline(0,color=INK,lw=.55); axf.axhline(0,color=INK,lw=.55)
    axf.set_xlabel("Task-event Spearman rho"); axf.set_ylabel("Median run (d)")
    quiet(axf); label_panel(axf,"f",-.18,1.03)

    axg = fig.add_subplot(gs[3, 3:]); axg.set_xlim(0, 1); axg.set_ylim(0, 1); axg.axis("off")
    axg.text(.36,.94,"dispersed",ha="center",va="top",color=BLUE,fontweight="bold",fontsize=6.0)
    axg.text(.78,.94,"clustered",ha="center",va="top",color=CORAL,fontweight="bold",fontsize=6.0)
    for label, col_d, col_c, y0 in [
        ("C1 annual task", "dispersed_task_normalized", "clustered_task_normalized", .58),
        ("R2 annual recharge", "dispersed_recharge_normalized", "clustered_recharge_normalized", .22)]:
        axg.text(.01,y0+.10,label,ha="left",va="center",color=INK,fontweight="bold",fontsize=5.8)
        for x0, col, colour in [(.23,col_d,BLUE),(.65,col_c,CORAL)]:
            vals=np.asarray(chronology[col],float)
            scale=np.abs(vals)/(np.nanmax(np.abs(vals)) or 1)
            width=.30/len(vals)
            for i,v in enumerate(scale):
                axg.add_patch(Rectangle((x0+i*width,y0),width+.0004,.20,
                                        facecolor=mpl.colors.to_rgba(colour,.22+.73*v),
                                        edgecolor="white",lw=.10))
        axg.annotate("",xy=(.62,y0+.10),xytext=(.56,y0+.10),
                     arrowprops={"arrowstyle":"-|>","color":MUTED,"lw":.75})
    zero=float(recovery[(recovery.chronology=="clustered")&(recovery.recovery_hours==0)].ecdf.iloc[0])*100
    axg.text(.44,.05,"same annual content",ha="center",va="bottom",color=MUTED,fontsize=5.3)
    axg.text(.80,.05,f"{zero:.1f}% zero-window mass",ha="center",va="bottom",color=CORAL,fontsize=5.5,fontweight="bold")
    label_panel(axg,"g",-.015,1.00)

    save_bundle(fig, "fig_v54_figure2_three_regime_event_atlas")


def figure3() -> None:
    eu = pd.read_csv(SRC49 / "figure2a_europe_map.csv")
    arms = pd.read_csv(SRC / "figure3_panel_a_portfolio_arm_values.csv")
    alloc = pd.read_csv(SRC / "figure3_panel_b_component_allocation.csv")
    paired = pd.read_csv(SRC / "figure3_panel_d_paired_mix_advantage.csv")
    cycles = pd.read_csv(SRC / "figure3_panel_e_branch_cycles.csv")

    fig = plt.figure(figsize=(7.20, 7.65))
    gs = GridSpec(4, 12, figure=fig, height_ratios=[1.08,1.18,1.16,1.28],
                  hspace=.53,wspace=.78,left=.07,right=.985,top=.96,bottom=.06)

    axa=fig.add_subplot(gs[0,:7])
    dispersed=alloc[alloc._chronology=="dispersed"].iloc[0]
    clustered=alloc[alloc._chronology=="clustered"].iloc[0]
    layer_rows=[("installed P",50.0),
                ("installed E",100/7),
                ("task · D",100*(1-dispersed.tail_task_share)),
                ("task · C",100*(1-clustered.tail_task_share))]
    for i,(name,common) in enumerate(layer_rows):
        tail=100-common
        axa.barh(i,common,color=BLUE,height=.62,edgecolor="white",lw=.4)
        axa.barh(i,tail,left=common,color=SAGE,height=.62,edgecolor="white",lw=.4)
        axa.text(common/2,i,f"{common:.1f}%",ha="center",va="center",color="white",fontweight="bold",fontsize=5.0)
        axa.text(common+tail/2,i,f"{tail:.1f}%",ha="center",va="center",color="white",fontweight="bold",fontsize=5.0)
    axa.axhline(1.5,color=GRID,lw=.7)
    axa.set_yticks(range(4),[x[0] for x in layer_rows]);axa.set_xlim(0,100);axa.invert_yaxis()
    axa.set_xlabel("Layer share (%)");quiet(axa,"x")
    axa.text(.25,1.04,"4 h LFP",transform=axa.transAxes,ha="center",color=BLUE,fontweight="bold",fontsize=5.2)
    axa.text(.75,1.04,"24 h A-CAES",transform=axa.transAxes,ha="center",color=SAGE,fontweight="bold",fontsize=5.2)
    label_panel(axa,"a",-.13,1.08)

    axb=fig.add_subplot(gs[0,7:])
    value_sets=[]
    for chrono in ["dispersed","clustered"]:
        value_sets.append(arms[(arms.tail_branch=="A-CAES tail")&(arms.chronology==chrono)].portfolio_value_eur_per_kw_common_power_year.to_numpy())
    violins=axb.violinplot(value_sets,positions=[0,1],vert=False,widths=.78,showextrema=False)
    for body,colour in zip(violins["bodies"],[BLUE,CORAL]):
        body.set_facecolor(mpl.colors.to_rgba(colour,.30));body.set_edgecolor(colour);body.set_linewidth(.7)
    for i,(vals,colour,marker) in enumerate(zip(value_sets,[BLUE,CORAL],["o","s"])):
        interval_mark(axb,vals,i,colour,marker=marker,raw=True,width=.12)
    frac=float((paired.clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year<0).mean())
    axb.axvline(0,color=INK,ls="--",lw=.6)
    axb.text(.97,.91,f"{100*frac:.1f}% lose value",transform=axb.transAxes,
             ha="right",va="top",color=CORAL,fontsize=5.0,fontweight="bold")
    axb.set_yticks([0,1],["dispersed","clustered"]);axb.set_xlabel("A-CAES portfolio value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    quiet(axb,"x");label_panel(axb,"b",-.18,1.08)

    axc=fig.add_subplot(gs[1:3,4:])
    country_med=paired.groupby("country").clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year.median().to_dict()
    cmap=mpl.colors.LinearSegmentedColormap.from_list("effect",[CORAL,"#F4F1EC",BLUE])
    vmax=max(abs(min(country_med.values())),abs(max(country_med.values())));norm=mpl.colors.TwoSlopeNorm(0,-vmax,vmax)
    europe_map(axc,country_med,cmap,norm)
    axc.set_anchor("E")
    cent=eu.set_index("gisco_code")[["lon","lat"]]
    for row in paired.itertuples():
        if row.country not in cent.index: continue
        base=cent.loc[row.country]; angle=(int(row.year)%11)/11*2*np.pi; radius=.28+.06*(int(row.year)%3)
        axc.scatter(base.lon+radius*np.cos(angle),base.lat+radius*np.sin(angle),s=13,
                    color=cmap(norm(row.clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year)),
                    edgecolor="white",lw=.25,zorder=5)
    cax=axc.inset_axes([.22,.94,.50,.035]);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal")
    cb.set_label("Clustered - dispersed value",fontsize=5.0);cb.ax.tick_params(labelsize=5.0,length=1.2)
    label_panel(axc,"c",0.0,.98)

    axd=fig.add_subplot(gs[1,:4])
    order=[("LFP tail","dispersed",BLUE,"4 h LFP · D"),("LFP tail","clustered",CORAL,"4 h LFP · C"),
           ("A-CAES tail","dispersed",BLUE,"A-CAES · D"),("A-CAES tail","clustered",CORAL,"A-CAES · C")]
    for i,(branch,chrono,colour,label) in enumerate(order):
        vals=arms[(arms.tail_branch==branch)&(arms.chronology==chrono)].portfolio_value_eur_per_kw_common_power_year
        interval_mark(axd,vals,i,colour,marker="o" if chrono=="dispersed" else "s")
    axd.axvline(0,color=INK,lw=.6);axd.set_yticks(range(4),[x[3] for x in order]);axd.invert_yaxis();axd.set_xlabel("Portfolio value")
    quiet(axd,"x");label_panel(axd,"d",-.26,1.02)

    egrid=GridSpecFromSubplotSpec(2,1,subplot_spec=gs[2,:4],height_ratios=[.8,1.1],hspace=.32)
    axe1=fig.add_subplot(egrid[0,0]);axe2=fig.add_subplot(egrid[1,0])
    for i,chrono in enumerate(["dispersed","clustered"]):
        r=alloc[alloc._chronology==chrono].iloc[0]; axe1.barh(i,100*(1-r.tail_task_share),color=BLUE,height=.58)
        axe1.barh(i,100*r.tail_task_share,left=100*(1-r.tail_task_share),color=SAGE,height=.58)
    axe1.set_yticks([0,1],["D","C"]);axe1.set_xlim(0,100);axe1.set_xlabel("Task allocation (%)");quiet(axe1,"x");label_panel(axe1,"e",-.20,1.10)
    vals=100*paired.tail_discharge_share_clustered
    axe2.hist(vals,bins=12,color=mpl.colors.to_rgba(SAGE,.60),edgecolor="white",lw=.3)
    axe2.axvline(np.median(vals),color=INK,lw=.8);axe2.set_xlabel("Tail service share, clustered (%)");axe2.set_ylabel("count");quiet(axe2,"y")

    fgrid=GridSpecFromSubplotSpec(3,1,subplot_spec=gs[3,:5],hspace=.20)
    diagnostics=[("tail charge",paired.tail_charge_mwh_dispersed,paired.tail_charge_mwh_clustered,1e-3),
                 ("equivalent cycles",paired.tail_equivalent_full_cycles_dispersed,paired.tail_equivalent_full_cycles_clustered,1),
                 ("portfolio value",paired.mix_benefit_eur_per_kw_common_power_year_dispersed,paired.mix_benefit_eur_per_kw_common_power_year_clustered,1)]
    for i,(name,d,c,scale) in enumerate(diagnostics):
        ax=fig.add_subplot(fgrid[i,0]); x=np.asarray(d)*scale;y=np.asarray(c)*scale
        ax.scatter(x,y,s=6,color=mpl.colors.to_rgba(CORAL,.22),edgecolor="none",rasterized=True)
        lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],color=INK,ls="--",lw=.55)
        ax.text(.02,.80,name,transform=ax.transAxes,ha="left",va="top",fontsize=5.0,color=MUTED)
        if i<2: ax.tick_params(labelbottom=False)
        else: ax.set_xlabel("dispersed → clustered")
        quiet(ax)
        if i==0:label_panel(ax,"f",-.22,1.05)

    axg=fig.add_subplot(gs[3,5:9])
    tail=float(alloc.loc[alloc._chronology=="dispersed","tail_task_share"].iloc[0])
    shares=[50.0,85.7,100*tail];labels=["Power","Energy","Tail task"];cols=[BLUE,SAGE,CORAL]
    for i,(value,label,colour) in enumerate(zip(shares,labels,cols)):
        axg.plot([0,100],[i,i],color=GRID,lw=4.8,solid_capstyle="round")
        axg.plot([0,value],[i,i],color=colour,lw=4.8,solid_capstyle="round")
        axg.scatter(value,i,s=32,color=colour,edgecolor="white",lw=.45,zorder=4)
        axg.text(min(value+4,96),i,f"{value:.1f}%",va="center",ha="left" if value<90 else "right",
                 color=colour,fontweight="bold",fontsize=5.2)
    axg.set_xlim(0,103);axg.set_ylim(-.6,2.6);axg.set_yticks(range(3),labels);axg.invert_yaxis()
    axg.set_xlabel("Portfolio share (%)");quiet(axg,"x");label_panel(axg,"g",-.24,1.03)

    axh=fig.add_subplot(gs[3,9:])
    retain=np.column_stack([100*paired.tail_charge_mwh_clustered/paired.tail_charge_mwh_dispersed,
                            100*paired.tail_discharge_mwh_clustered/paired.tail_discharge_mwh_dispersed,
                            100*paired.tail_equivalent_full_cycles_clustered/paired.tail_equivalent_full_cycles_dispersed])
    retain_sets=[retain[np.isfinite(retain[:,i]),i] for i in range(3)]
    violins=axh.violinplot(retain_sets,positions=range(3),widths=.72,showextrema=False)
    for body,colour in zip(violins["bodies"],[SAGE,BLUE,TAUPE]):
        body.set_facecolor(mpl.colors.to_rgba(colour,.28));body.set_edgecolor(colour);body.set_linewidth(.65)
    for i,(values,colour) in enumerate(zip(retain_sets,[SAGE,BLUE,TAUPE])):
        axh.scatter(i+deterministic_jitter(len(values),.10),values,s=5.0,
                    color=mpl.colors.to_rgba(colour,.24),edgecolor="none",rasterized=True)
        q1,med,q3=np.quantile(values,[.25,.5,.75])
        axh.plot([i,i],[q1,q3],color=colour,lw=2.7,solid_capstyle="round")
        axh.scatter(i,med,s=27,color=colour,edgecolor="white",lw=.45,zorder=5)
    lfp_d=cycles[(cycles.branch=="LFP tail")&(cycles.chronology=="dispersed")].equivalent_full_cycles
    lfp_c=cycles[(cycles.branch=="LFP tail")&(cycles.chronology=="clustered")].equivalent_full_cycles
    lfp_retention=100*np.median(lfp_c)/np.median(lfp_d)
    axh.scatter(2,lfp_retention,s=31,marker="D",color=BLUE_DARK,edgecolor="white",lw=.4,zorder=6)
    axh.text(2,lfp_retention-4,"4 h LFP",ha="center",va="top",color=BLUE_DARK,fontsize=5.0)
    axh.axhline(100,color=INK,ls=":",lw=.6);axh.set_xticks(range(3),["charge","service","cycles"],rotation=20,ha="right")
    axh.set_ylabel("");axh.text(.02,.92,"C / D retention (%)",transform=axh.transAxes,
                                ha="left",va="top",color=INK,fontsize=5.0)
    quiet(axh,"y");label_panel(axh,"h",-.28,1.03)

    save_bundle(fig,"fig_v54_figure3_portfolio_cycling_ledger")


def figure4() -> None:
    eu=pd.read_csv(SRC49/"figure2a_europe_map.csv")
    eu_cy=pd.read_csv(SRC/"figure2_panel_b_country_year_geometry.csv")
    chronology=pd.read_csv(SRC/"figure2_panel_c_conserved_chronology.csv")
    retention=pd.read_csv(SRC/"figure4_panel_a_tail_retention.csv")
    binding=pd.read_csv(SRC/"figure4_panel_b_binding_hours.csv")
    multiplier=pd.read_csv(SRC/"figure4_panel_c_recharge_multiplier_response.csv")
    link=pd.read_csv(SRC/"figure4_panel_d_physical_economic_link.csv")
    paired=pd.read_csv(SRC/"figure3_panel_d_paired_mix_advantage.csv")
    residuals=pd.read_csv(SRC/"figure6_panel_d_natural_fixed_effect_residuals.csv")
    coeff=pd.read_csv(ROOT/"joint_choice/natural_fixed_effect_coefficients.csv")

    fig=plt.figure(figsize=(7.20,8.35))
    gs=GridSpec(5,12,figure=fig,height_ratios=[.82,1.36,1.36,1.02,1.16],
                hspace=.62,wspace=.72,left=.07,right=.985,top=.97,bottom=.055)

    axa=fig.add_subplot(gs[0,:8]); axam=fig.add_subplot(gs[0,9:])
    mat=np.vstack([chronology.dispersed_task_normalized,chronology.dispersed_recharge_normalized,
                   chronology.clustered_task_normalized,chronology.clustered_recharge_normalized])
    axa.imshow(mat,aspect="auto",cmap="RdBu_r",vmin=-max(abs(mat.min()),abs(mat.max())),vmax=max(abs(mat.min()),abs(mat.max())))
    axa.set_yticks(range(4),["D task","D recharge","C task","C recharge"]);axa.set_xlabel("Day of year")
    axa.tick_params(length=0);label_panel(axa,"a",-.09,1.04)
    for row,colour,label in [(mat[0],CORAL,"task"),(mat[1],BLUE,"recharge")]:
        x,d=density_curve(row);axam.plot(d,x,color=colour,lw=1.1,label=label)
    axam.set_xlabel("density");axam.set_ylabel("normalized level");axam.legend(loc="upper right");quiet(axam)

    natural=link.merge(residuals[["country","year","natural_zero_recovery_window_share"]],on=["country","year"])

    axb=fig.add_subplot(gs[1:3,:6])
    country=link.groupby("country").clustered_minus_dispersed_tail_equivalent_full_cycles.median().to_dict()
    vmax=max(abs(min(country.values())),abs(max(country.values())));norm=mpl.colors.TwoSlopeNorm(0,-vmax,vmax)
    cmap=mpl.colors.LinearSegmentedColormap.from_list("constraint",[CORAL,"#F2EFE8",BLUE])
    europe_map(axb,country,cmap,norm)
    axb.set_anchor("C")
    cent=eu.set_index("gisco_code")[["lon","lat"]]
    for r in link.itertuples():
        if r.country not in cent.index:continue
        c=cent.loc[r.country];ang=(int(r.year)%11)/11*2*np.pi
        axb.scatter(c.lon+.33*np.cos(ang),c.lat+.33*np.sin(ang),s=12,color=cmap(norm(r.clustered_minus_dispersed_tail_equivalent_full_cycles)),edgecolor="white",lw=.25,zorder=5)
    cax=axb.inset_axes([.24,.94,.48,.035]);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal")
    cb.set_label("Change in A-CAES equivalent cycles",fontsize=5.0);cb.ax.tick_params(labelsize=5.0,length=1.2)
    label_panel(axb,"b",0,.99)

    axc=fig.add_subplot(gs[1:3,6:])
    temp=natural.copy()
    temp["event_high"]=temp.clustered_minus_dispersed_tail_equivalent_full_cycles<temp.clustered_minus_dispersed_tail_equivalent_full_cycles.median()
    temp["value_high"]=temp.clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year<temp.clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year.median()
    temp["class"]=np.select([temp.event_high&temp.value_high,temp.event_high&~temp.value_high,~temp.event_high&temp.value_high],[3,2,1],default=0)
    cls=temp.groupby("country")["class"].agg(lambda s:int(s.mode().iloc[0])).to_dict();cols={0:BLUE,1:GOLD,2:SAGE,3:CORAL}
    europe_map(axc,cls,mpl.colors.ListedColormap([cols[i] for i in range(4)]),mpl.colors.BoundaryNorm([-.5,.5,1.5,2.5,3.5],4))
    axc.set_anchor("C")
    axc.legend(handles=[Patch(color=cols[i],label=l) for i,l in enumerate(["low loss","value loss","cycle loss","high-high"])],
               ncol=2,loc="upper center",bbox_to_anchor=(.5,1.055),handlelength=.8,columnspacing=.65)
    label_panel(axc,"c",0,.99)

    dgrid=GridSpecFromSubplotSpec(2,1,subplot_spec=gs[3,:4],height_ratios=[1.05,.78],hspace=.36)
    axc1=fig.add_subplot(dgrid[0,0]);axc2=fig.add_subplot(dgrid[1,0])
    for i,metric in enumerate(["Charging","Task service","Equivalent cycles"]):
        vals=retention[retention.metric==metric].clustered_to_dispersed_pct
        interval_mark(axc1,vals,i,[SAGE,BLUE,TAUPE][i],raw=True)
    axc1.axvline(100,color=INK,ls="--",lw=.55);axc1.set_yticks(range(3),["charge","service","cycles"]);axc1.set_xlabel("C / D retention (%)")
    quiet(axc1,"x");label_panel(axc1,"d",-.24,1.08)
    for i,metric in enumerate(["shared_recharge_binding_hours","tail_charge_power_binding_hours"]):
        d=float(binding.loc[binding._chronology=="dispersed",metric].iloc[0]);c=float(binding.loc[binding._chronology=="clustered",metric].iloc[0])
        total=max(d,c);axc2.barh(i,100*min(d,c)/total,color=SAGE,height=.55);axc2.barh(i,100*(1-min(d,c)/total),left=100*min(d,c)/total,color=CORAL,height=.55)
    axc2.set_yticks([0,1],["surplus","interface"]);axc2.set_xlim(0,100);axc2.set_xlabel("Retained / changed binding (%)");quiet(axc2,"x")

    axd=fig.add_subplot(gs[3,4:8])
    merged=link.merge(eu_cy[["country","year","natural_longest_event_run_days"]],on=["country","year"])
    bins=pd.cut(merged.natural_longest_event_run_days,[0,6,9,12,30],labels=["≤6 d","7–9 d","10–12 d",">12 d"])
    for i,(label,colour) in enumerate(zip(bins.cat.categories,[BLUE,SAGE,GOLD,CORAL])):
        vals=-merged.loc[bins==label,"tail_charge_change_gwh"].to_numpy()
        ridge(axd,vals,i,colour,.70,str(label))
    axd.set_yticks([]);axd.set_xlabel("Tail-charge loss (GWh)");quiet(axd,"x");label_panel(axd,"e",-.18,1.03)

    axe=fig.add_subplot(gs[3,8:])
    for chrono,colour,marker in [("dispersed",BLUE,"o"),("clustered",CORAL,"s")]:
        x=paired[f"shared_recharge_binding_hours_{chrono}"]+1;y=paired[f"tail_charge_power_binding_hours_{chrono}"]+1
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("Binding-hour pseudocount produced nonpositive values")
        axe.scatter(x,y,s=8,color=mpl.colors.to_rgba(colour,.26),marker=marker,edgecolor="none",rasterized=True,label=chrono)
    axe.set_xscale("log");axe.set_yscale("log");axe.set_xlabel("Shared-surplus binding h + 1");axe.set_ylabel("")
    axe.text(.03,.96,"Charge-interface\nbinding h + 1",transform=axe.transAxes,
             ha="left",va="top",color=INK,fontsize=5.0)
    axe.legend(loc="lower right");quiet(axe);label_panel(axe,"f",-.18,1.03)

    fgrid=GridSpecFromSubplotSpec(3,1,subplot_spec=gs[4,:6],hspace=.18)
    metrics=[("value",natural.clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year),
             ("tail charge",natural.tail_charge_change_gwh),
             ("cycles",natural.clustered_minus_dispersed_tail_equivalent_full_cycles)]
    for i,(label,y) in enumerate(metrics):
        ax=fig.add_subplot(fgrid[i,0]);x=100*natural.natural_zero_recovery_window_share
        ax.scatter(x,y,s=6,color=mpl.colors.to_rgba(CORAL,.22),edgecolor="none",rasterized=True)
        fit=np.polyfit(x,y,1);xx=np.linspace(x.min(),x.max(),80);ax.plot(xx,np.polyval(fit,xx),color=INK,lw=.9)
        ax.axhline(0,color=INK,lw=.5);ax.set_ylabel("");quiet(ax)
        ax.text(.01,.80,label,transform=ax.transAxes,ha="left",va="top",color=MUTED,fontsize=5.0)
        if i<2:ax.set_xticklabels([])
        else:ax.set_xlabel("Natural zero-window share (%)")
        if i==0:label_panel(ax,"g",-.12,1.03)

    axh=fig.add_subplot(gs[4,6:])
    models=[("zero_window_only","Unadjusted"),("joint_geometry","+ event geometry"),("joint_geometry_plus_scale","+ annual scale")]
    found=[]
    for mid,label in models:
        q=coeff[(coeff.model_id==mid)&(coeff.predictor=="natural_zero_recovery_window_share")]
        if len(q):found.append((q.iloc[0],label))
    ypos=np.arange(len(found))[::-1];est=np.array([.1*r.coefficient for r,_ in found]);lo=np.array([.1*r.bootstrap_ci95_low for r,_ in found]);hi=np.array([.1*r.bootstrap_ci95_high for r,_ in found])
    colours=[BLUE,GOLD,CORAL]
    left_edge=float(lo.min()-.68)
    for y,e,l,h,colour,(_,model_label) in zip(ypos,est,lo,hi,colours,found):
        axh.plot([l,h],[y,y],color=mpl.colors.to_rgba(colour,.30),lw=6.0,solid_capstyle="round",zorder=2)
        axh.plot([l,h],[y,y],color=colour,lw=1.4,solid_capstyle="round",zorder=3)
        axh.scatter(e,y,s=38,color=colour,edgecolor="white",lw=.55,zorder=4)
        axh.text(left_edge+.04,y,model_label,ha="left",va="center",color=INK,fontsize=5.0)
        axh.text(h+.04,y,f"{e:.2f}",ha="left",va="center",color=colour,fontsize=5.0,fontweight="bold")
    axh.axvline(0,color=INK,ls="--",lw=.65);axh.set_yticks([]);axh.set_xlim(left_edge,.08)
    axh.set_xlabel("Effect per +10 pp (EUR kW$^{-1}$ yr$^{-1}$)");axh.set_ylim(-.55,len(found)-.45)
    quiet(axh,"x");label_panel(axh,"h",-.15,1.03)

    save_bundle(fig,"fig_v54_figure4_recovery_interface_tests")


def figure5() -> None:
    eu=pd.read_csv(SRC49/"figure2a_europe_map.csv")
    eu_resp=pd.read_csv(SRC49/"figure5a_europe_duration_response.csv").sort_values("duration_h")
    cross=pd.read_csv(SRC49/"figure5bc_china_japan_duration_response.csv")
    decomp=pd.read_csv(SRC/"figure5_panel_c_signed_channel_decomposition.csv")
    signature=pd.read_csv(SRC/"figure5_panel_d_cluster_signature.csv")
    crossing=pd.read_csv(SRC/"figure5_panel_e_zero_crossing_summary.csv")
    records=pd.read_csv(SRC/"figure5_panel_e_zero_crossing_records.csv")
    mapdata=pd.read_csv(SRC/"ed9_four_day_country_crossing_map.csv")
    status=pd.read_csv(SRC/"ed9_crossing_censoring_status.csv")
    valleys=pd.read_csv(SRC/"figure5_panel_e_conditional_valley_records.csv")
    natural=pd.read_csv(SRC/"figure2_panel_b_country_year_geometry.csv")

    fig=plt.figure(figsize=(7.20,8.30))
    gs=GridSpec(5,12,figure=fig,height_ratios=[.86,1.30,1.30,1.02,1.18],
                hspace=.62,wspace=.72,left=.07,right=.985,top=.97,bottom=.055)

    axa=fig.add_subplot(gs[0,:8]);axam=fig.add_subplot(gs[0,9:])
    durations=sorted(set(eu_resp.duration_h).union(cross.duration_h))
    rows=[]
    for name in ["Europe","China","Japan"]:
        if name=="Europe": q=eu_resp.set_index("duration_h").point_estimate.reindex(durations).interpolate().to_numpy()
        else:q=cross[cross.application_case==name.lower()].set_index("duration_h").mean_effect_eur_kw_year.reindex(durations).interpolate().to_numpy()
        rows.append(q)
    matrix=np.vstack(rows);vmax=np.nanmax(abs(matrix));edges=duration_edges(np.array(durations))
    axa.pcolormesh(edges,np.arange(4)-.5,matrix,cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="flat")
    axa.set_xscale("log");axa.set_yticks(range(3),["Europe","China","Japan"]);axa.set_xlabel("Storage duration (h)")
    axa.axvline(24,color=INK,ls="--",lw=.55);label_panel(axa,"a",-.09,1.03)
    for q,colour,label in zip(rows,[BLUE,CORAL,SAGE],["Europe","China","Japan"]):
        axam.plot(q,durations,color=colour,lw=1.15,label=label)
    axam.axvline(0,color=INK,lw=.55);axam.set_yscale("log");axam.set_ylabel("duration (h)");axam.set_xlabel("effect");axam.legend(loc="lower right");quiet(axam)

    axb=fig.add_subplot(gs[1:3,:6]); vals=mapdata.set_index("country").median_crossing_h.to_dict();norm=mpl.colors.Normalize(48,144);cmap=mpl.colors.LinearSegmentedColormap.from_list("cross",[BLUE,"#EEE8D8",CORAL])
    europe_map(axb,vals,cmap,norm)
    axb.set_anchor("C")
    cent=eu.set_index("gisco_code")[["lon","lat"]]
    for r in mapdata.itertuples():
        if r.country in cent.index:
            c=cent.loc[r.country];axb.scatter(c.lon,c.lat,s=18+3*r.country_years,color=cmap(norm(r.median_crossing_h)),edgecolor="white",lw=.4,zorder=5)
    cax=axb.inset_axes([.24,.94,.48,.035]);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal")
    cb.set_label("Four-day crossover (h)",fontsize=5.0);cb.ax.tick_params(labelsize=5.0,length=1.2);label_panel(axb,"b",0,.99)

    axg=fig.add_subplot(gs[1:3,6:])
    v=valleys.groupby("country").strict_valley.mean().to_dict();normg=mpl.colors.Normalize(0,max(v.values()) if max(v.values())>0 else 1)
    cmapg=mpl.colors.LinearSegmentedColormap.from_list("valley",["#F2F0EA",GOLD,CORAL]);europe_map(axg,v,cmapg,normg)
    axg.set_anchor("C")
    cax=axg.inset_axes([.24,.94,.48,.035]);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=normg,cmap=cmapg),cax=cax,orientation="horizontal")
    cb.set_label("Strict-valley share",fontsize=5.0,labelpad=1);cb.ax.xaxis.set_label_position("top")
    label_panel(axg,"c",0,.99)

    cgrid=GridSpecFromSubplotSpec(2,1,subplot_spec=gs[3,:4],height_ratios=[1.0,.8],hspace=.34)
    axc1=fig.add_subplot(cgrid[0,0]);axc2=fig.add_subplot(cgrid[1,0])
    for i,(days,colour) in enumerate([(2,BLUE),(4,GOLD),(6,CORAL)]):
        r=crossing[crossing.target_cluster_days==days].iloc[0]
        axc1.plot([r.q25_crossing_h,r.q75_crossing_h],[i,i],color=colour,lw=2.6);axc1.scatter(r.median_crossing_h,i,s=29,color=colour,edgecolor="white",lw=.4,zorder=4)
        axc1.text(r.median_crossing_h,i+.22,f"{int(r.median_crossing_h)} h",ha="center",fontsize=5.0)
    axc1.set_yticks(range(3),["2 d","4 d","6 d"]);axc1.set_xlabel("First sustained positive duration (h)");quiet(axc1,"x");label_panel(axc1,"d",-.24,1.08)
    statecols={"left_censored":GREY,"observed":SAGE,"right_censored":CORAL}
    for i,days in enumerate([2,4,6]):
        left=0
        for state in statecols:
            q=status[(status.target_cluster_days==days)&(status.crossing_status==state)];v=float(q.share_pct.iloc[0]) if len(q) else 0
            axc2.barh(i,v,left=left,color=statecols[state],height=.55);left+=v
    axc2.set_yticks(range(3),["2 d","4 d","6 d"]);axc2.set_xlim(0,100);axc2.set_xlabel("Observed / censored share (%)");quiet(axc2,"x")

    axd=fig.add_subplot(gs[3,4:8])
    for i,(days,colour) in enumerate([(2,BLUE),(4,GOLD),(6,CORAL)]):
        observed=records[(records.target_cluster_days==days)&(records.crossing_status=="observed")]
        if observed.zero_crossing_duration_h.isna().any():
            raise ValueError(f"Observed {days}-day crossings contain missing durations")
        q=observed.zero_crossing_duration_h.to_numpy()
        ridge(axd,q,i,colour,.72,f"{days} d")
    axd.set_yticks([]);axd.set_xlabel("Crossover duration (h)");quiet(axd,"x");label_panel(axd,"e",-.18,1.03)

    axe=fig.add_subplot(gs[3,8:])
    for col,colour,label,ls in [("cycling_channel_country_block_mean",BLUE,"cycling","-"),("coverage_channel_country_block_mean",CORAL,"coverage","--"),("total_effect_country_block_mean",INK,"total","-")]:
        axe.plot(decomp.duration_h,decomp[col],color=colour,lw=1.25,ls=ls,label=label)
    axe.axhline(0,color=INK,lw=.55);axe.set_xscale("log");axe.set_xlabel("Storage duration (h)");axe.set_ylabel("Chronology effect");axe.legend(ncol=3,loc="upper left");quiet(axe);label_panel(axe,"f",-.18,1.03)

    fgrid=GridSpecFromSubplotSpec(3,1,subplot_spec=gs[4,:7],hspace=.18)
    for i,(name,colour) in enumerate([("Europe",BLUE),("China",CORAL),("Japan",SAGE)]):
        ax=fig.add_subplot(fgrid[i,0])
        if name=="Europe":q=eu_resp
        else:q=cross[cross.application_case==name.lower()].rename(columns={"mean_effect_eur_kw_year":"point_estimate","ci95_low":"ci_low","ci95_high":"ci_high"})
        ax.fill_between(q.duration_h,q.ci_low,q.ci_high,color=mpl.colors.to_rgba(colour,.15),lw=0);ax.plot(q.duration_h,q.point_estimate,color=colour,lw=1.1)
        ax.axhline(0,color=INK,lw=.5);ax.set_xscale("log");ax.set_ylabel("");quiet(ax)
        ax.text(.01,.80,name,transform=ax.transAxes,ha="left",va="top",color=colour,fontsize=5.0)
        if i<2:ax.set_xticklabels([])
        else:ax.set_xlabel("duration (h)")
        if i==0:label_panel(ax,"g",-.12,1.03)

    hgrid=GridSpecFromSubplotSpec(2,1,subplot_spec=gs[4,7:],height_ratios=[1.2,.65],hspace=.35)
    axh1=fig.add_subplot(hgrid[0,0]);axh2=fig.add_subplot(hgrid[1,0])
    q=records[records.target_cluster_days==4].merge(natural[["country","year","natural_longest_event_run_days"]],on=["country","year"])
    q=q[q.crossing_status=="observed"]
    if q.zero_crossing_duration_h.isna().any():
        raise ValueError("Observed four-day crossings contain missing durations")
    axh1.scatter(q.natural_longest_event_run_days,q.zero_crossing_duration_h,s=7,color=mpl.colors.to_rgba(CORAL,.35),edgecolor="none",rasterized=True)
    fit=np.polyfit(q.natural_longest_event_run_days,q.zero_crossing_duration_h,1);xx=np.linspace(q.natural_longest_event_run_days.min(),q.natural_longest_event_run_days.max(),80);axh1.plot(xx,np.polyval(fit,xx),color=INK,lw=.9)
    axh1.set_xlabel("Natural longest event (d)");axh1.set_ylabel("")
    axh1.text(.02,.96,"4 d crossover (h)",transform=axh1.transAxes,ha="left",va="top",fontsize=5.0,color=INK)
    quiet(axh1);label_panel(axh1,"h",-.12,1.04)
    axh2.plot(crossing.target_cluster_days,crossing.median_crossing_h,color=GOLD,marker="o",ms=3.5);axh2.set_xticks([2,4,6]);axh2.set_xlabel("Cluster length (d)");axh2.set_ylabel("")
    axh2.text(.02,.88,"median h",transform=axh2.transAxes,ha="left",va="top",fontsize=5.0,color=MUTED);quiet(axh2)

    save_bundle(fig,"fig_v54_figure5_cross_duration_reversal")


def figure6() -> None:
    parity=pd.read_csv(SRC/"figure6_panel_b_value_shift_cost_path_ecdf.csv")
    flips=pd.read_csv(SRC/"figure6_panel_a_robust_flip_summary.csv")
    curve=pd.read_csv(SRC/"figure6_panel_c_capacity_anchor_flip_curve.csv")
    anchors=pd.read_csv(SRC/"figure6_panel_c_capacity_market_anchors.csv")
    sensitivity=pd.read_csv(ROOT/"sequencing_parity/cost_duration_discount_sensitivity.csv")
    parity_dir=SRC/"v54_parity_exposure"
    exposure=pd.read_csv(parity_dir/"figure6cde_system_duration_exposure.csv")
    parity_records=pd.read_csv(parity_dir/"figure6cde_system_year_parity_records.csv")
    parity_summary=pd.read_csv(parity_dir/"figure6b_24h_cross_regime_parity_summary.csv").set_index("regime")
    central_movement=float(sensitivity[
        np.isclose(sensitivity.discount_rate,.036)
        & np.isclose(sensitivity.caes_duration_h,24.0)
    ].absolute_relative_cost_movement_eur_kw_year.iloc[0])

    fig=plt.figure(figsize=(7.20,8.10))
    gs=GridSpec(3,12,figure=fig,height_ratios=[1.20,3.30,1.18],hspace=.46,wspace=.78,
                left=.075,right=.985,top=.965,bottom=.065)

    axa=fig.add_subplot(gs[0,:5])
    piv=sensitivity.pivot(index="discount_rate",columns="caes_duration_h",values="absolute_relative_cost_movement_eur_kw_year")
    im=axa.imshow(piv.values,origin="lower",aspect="auto",cmap="cividis",
                  extent=[piv.columns.min(),piv.columns.max(),100*piv.index.min(),100*piv.index.max()])
    axa.contour(piv.columns,100*piv.index,piv.values,levels=[5.48],
                colors=["white"],linewidths=.8)
    axa.contour(piv.columns,100*piv.index,piv.values,levels=[central_movement],
                colors=[CORAL],linewidths=.8)
    axa.scatter(24,3.6,marker="*",s=52,color=CORAL,edgecolor="white",lw=.4)
    axa.set_xlabel("A-CAES duration (h)");axa.set_ylabel("Discount rate (%)")
    cax=axa.inset_axes([.20,1.035,.60,.060]);cb=fig.colorbar(im,cax=cax,orientation="horizontal")
    cb.set_label("2025–2035 cost movement",fontsize=5.0);cb.ax.tick_params(labelsize=5.0,length=1.2)
    label_panel(axa,"a",-.17,1.10)

    axb=fig.add_subplot(gs[0,5:])
    focus=parity_records[np.isclose(parity_records.duration_h,24.0)].copy()
    order=["Europe","China","Japan"];colours=[BLUE,GOLD,SAGE]
    transformed=[]
    for regime in order:
        transformed.append(np.log10(1+focus.loc[focus.regime.eq(regime),"sequencing_shift_eur_kw_year"].to_numpy()))
    parts=axb.violinplot(transformed,positions=np.arange(3),vert=False,widths=.72,
                         showmeans=False,showmedians=False,showextrema=False)
    for body,colour in zip(parts["bodies"],colours):
        body.set_facecolor(colour);body.set_edgecolor(colour);body.set_alpha(.28);body.set_linewidth(.7)
    for i,(regime,colour,values) in enumerate(zip(order,colours,transformed)):
        raw=np.log10(1+focus.loc[focus.regime.eq(regime),"sequencing_shift_eur_kw_year"].to_numpy())
        axb.scatter(raw,i+deterministic_jitter(len(raw),.13),s=5,color=mpl.colors.to_rgba(colour,.28),
                    edgecolor="none",rasterized=True)
        med=np.median(raw);axb.scatter(med,i,s=24,color=colour,edgecolor="white",lw=.4,zorder=5)
        row=parity_summary.loc[regime]
        axb.text(np.log10(76),i,f"{int(row.above_parity_count)}/{int(row.n_system_years)}",
                 ha="right",va="center",color=colour,fontsize=5.2,fontweight="bold",
                 bbox={"facecolor":"white","edgecolor":"none","alpha":.90,"pad":.35},zorder=10)
    cost_line=np.log10(1+central_movement);axb.axvspan(cost_line,np.log10(76),color=mpl.colors.to_rgba(CORAL,.06),lw=0)
    axb.axvline(cost_line,color=INK,ls="--",lw=.75)
    ticks=np.array([0,2,6,10,20,40,70]);axb.set_xticks(np.log10(1+ticks),[str(v) for v in ticks])
    axb.set_xlim(0,np.log10(76));axb.set_yticks(range(3),order);axb.invert_yaxis()
    axb.set_xlabel("Absolute 24 h sequencing shift (EUR kW$^{-1}$ yr$^{-1}$)")
    axb.text(cost_line,.03,f"{central_movement:.2f} cost path",transform=axb.get_xaxis_transform(),ha="left",va="bottom",fontsize=5.0,color=INK)
    quiet(axb,"x");label_panel(axb,"b",-.12,1.04)

    common_durations=[4,6,8,10,12,14,16,18,20,22,24,28,32,36,40,44,48,60,72,96,100,120,144,168]
    exposure_cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "parity_exposure_v54",["#F3F1EC","#B9D2D8",GOLD,CORAL]
    )
    heatmaps=[];heat_axes=[]

    def short_label(regime: str, label: str) -> str:
        if regime=="China":
            return label.replace("Inner Mongolia","Inner Mong.")
        if regime=="Japan":
            prefecture,mesh=label.rsplit("-",1)
            return f"{prefecture[:4]} {mesh[-3:]}"
        return label

    def parity_heatmap(spec,regime: str,letter: str):
        sub=GridSpecFromSubplotSpec(2,1,subplot_spec=spec,height_ratios=[.48,3.0],hspace=.08)
        profile_ax=fig.add_subplot(sub[0,0]);heat_ax=fig.add_subplot(sub[1,0])
        q=exposure[exposure.regime.eq(regime)&exposure.duration_h.isin(common_durations)].copy()
        system_order=(q.groupby(["system_id","system_label"]).apply(
            lambda g: np.average(g.parity_exposure_frequency,weights=g.n_observation_years),
            include_groups=False).sort_values(ascending=False).index)
        ids=[item[0] for item in system_order];labels=[short_label(regime,item[1]) for item in system_order]
        matrix=q.pivot(index="system_id",columns="duration_h",values="parity_exposure_frequency").reindex(index=ids,columns=common_durations)
        weighted=(q.groupby("duration_h").apply(
            lambda g: np.average(g.parity_exposure_frequency,weights=g.n_observation_years),
            include_groups=False).reindex(common_durations).fillna(0)*100)
        x=np.arange(len(common_durations));profile_ax.fill_between(x,0,weighted.values,color=mpl.colors.to_rgba(CORAL,.23),lw=0)
        profile_ax.plot(x,weighted.values,color=CORAL,lw=.85);profile_ax.set_ylim(0,105);profile_ax.set_xlim(-.5,len(x)-.5)
        profile_ax.axhline(50,color=GRID,lw=.5);profile_ax.set_xticks([]);profile_ax.set_yticks([0,50,100] if regime=="Europe" else [])
        if regime=="Europe":profile_ax.set_ylabel("systems (%)",fontsize=5.0,labelpad=2)
        profile_ax.set_title(f"{regime}  |  {int(parity_summary.loc[regime].above_parity_count)}/{int(parity_summary.loc[regime].n_system_years)} at 24 h",
                             fontsize=6.0,fontweight="bold",color=INK,pad=2)
        quiet(profile_ax);label_panel(profile_ax,letter,-.17,1.22)
        values=np.ma.masked_invalid(matrix.values)
        x_edges=np.arange(values.shape[1]+1)-.5;y_edges=np.arange(values.shape[0]+1)-.5
        image=heat_ax.pcolormesh(x_edges,y_edges,values,cmap=exposure_cmap,vmin=0,vmax=1,
                                 shading="flat",edgecolors="white",linewidth=.08,
                                 antialiased=False,rasterized=False,snap=True)
        heat_ax.set_xlim(-.5,values.shape[1]-.5);heat_ax.set_ylim(values.shape[0]-.5,-.5)
        heat_ax.set_yticks(np.arange(len(labels)),labels,fontsize=5.0);heat_ax.set_xticks(np.arange(len(common_durations)))
        shown={4,8,12,16,24,48,96,168};heat_ax.set_xticklabels([str(v) if v in shown else "" for v in common_durations],rotation=0)
        if regime!="China":heat_ax.set_xlabel("Storage duration (h)")
        heat_ax.tick_params(length=0,pad=1.2)
        heat_ax.spines[:].set_visible(False);heatmaps.append(image);heat_axes.append(heat_ax)
        return heat_ax

    hero=GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1,:],wspace=.58)
    parity_heatmap(hero[0,0],"Europe","c")
    parity_heatmap(hero[0,1],"China","d")
    parity_heatmap(hero[0,2],"Japan","e")
    cb=fig.colorbar(heatmaps[0],ax=heat_axes,orientation="horizontal",ticks=[0,.5,1],
                    fraction=.020,pad=.055,aspect=34)
    cb.set_ticklabels(["0","50","100"]);cb.set_label("Historical observations above parity (%)",fontsize=5.2,labelpad=1)
    cb.ax.tick_params(labelsize=5.0,length=1.2)

    axf=fig.add_subplot(gs[2,:4])
    at=parity[(parity.country.eq("AT"))&(parity.year.eq(2024))].iloc[0]
    x_d=float(at.benefit_dispersed_eur_kw_year);x_c=float(at.benefit_clustered_eur_kw_year)
    axf.plot([x_c,x_d],[0,0],color=GREY,lw=3.0,solid_capstyle="round")
    axf.scatter([x_d,x_c],[0,0],s=38,color=[BLUE,CORAL],edgecolor="white",lw=.5,zorder=4)
    axf.text(x_d,.12,f"dispersed\n{x_d:+.2f}",ha="center",va="bottom",color=BLUE,fontsize=5.1)
    axf.text(x_c,.12,f"clustered\n{x_c:+.2f}",ha="center",va="bottom",color=CORAL,fontsize=5.1)
    axf.axvline(0,color=INK,lw=.65);axf.set_ylim(-.35,.52);axf.set_yticks([])
    axf.set_xlabel("AT-2024 A-CAES relative value");axf.text((x_c+x_d)/2,-.20,"21.77 shift",ha="center",color=INK,fontsize=5.2,fontweight="bold")
    quiet(axf,"x");label_panel(axf,"f",-.16,1.03)

    axg=fig.add_subplot(gs[2,4:8])
    for i,r in enumerate(flips.itertuples()):
        lo=100*r.country_block_bootstrap_ci95_low;hi=100*r.country_block_bootstrap_ci95_high;est=100*r.disjoint_tie_set_flip_rate
        colour=[BLUE,GOLD,CORAL][i];axg.plot([lo,hi],[i,i],color=colour,lw=2.6)
        axg.scatter(est,i,s=31,color=colour,edgecolor="white",lw=.4,zorder=4)
        axg.text(hi+.7,i,f"{r.disjoint_tie_set_flips}/132",va="center",color=colour,fontsize=5.0)
    axg.set_yticks(range(3),["2025","2030","2035"]);axg.set_xlim(-1,33);axg.set_xlabel("Procurement-identity changes (%)")
    quiet(axg,"x");label_panel(axg,"g",-.18,1.03)

    axh=fig.add_subplot(gs[2,8:])
    for year,colour,marker in [(2025,BLUE,"o"),(2030,GOLD,"s"),(2035,CORAL,"^")]:
        q=curve[curve.cost_year==year].sort_values("firm_value_eur_kw_year");axh.plot(q.firm_value_eur_kw_year,100*q.tail_entry_exit_flip_rate,color=colour,lw=1.15,marker=marker,ms=2.2,label=str(year))
    band=anchors[anchors.within_priority_market_range].eur_per_kw_year;axh.axvspan(band.min(),band.max(),color=mpl.colors.to_rgba(GOLD,.16),lw=0);axh.axvline(103.8,color=INK,ls="--",lw=.6)
    axh.set_xlabel("Firm-supply unit value");axh.set_ylabel("Tail flips (%)");axh.legend(loc="upper left",ncol=3)
    quiet(axh,"y");label_panel(axh,"h",-.17,1.03)

    save_bundle(fig,"fig_v54_figure6_sequencing_parity_exposure")


def main() -> None:
    configure()
    figure2(); figure3(); figure4(); figure5(); figure6()
    manifest={"backend":"python-matplotlib","template":"author-provided Nature page layouts",
              "figures":[2,3,4,5,6],"panel_limit":8}
    (HERE/"manifests/v54_reference_style_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    main()
