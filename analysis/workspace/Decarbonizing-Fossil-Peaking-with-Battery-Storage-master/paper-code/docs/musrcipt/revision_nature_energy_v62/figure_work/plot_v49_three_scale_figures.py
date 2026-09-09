#!/usr/bin/env python3
"""Build v49 Figures 2 and 5 from audited three-scale source data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parents[4]
SRC = HERE / "source_data_v49"
OUT = HERE / "main_figures"
OUT.mkdir(exist_ok=True)

EUROPE_GEO = WORKSPACE / "dataset/external_hourly/europe_mid_duration_valley_upgrade_v0_1/raw/gisco_countries_2024/CNTR_RG_20M_2024_4326.geojson"
CHINA_GEO = WORKSPACE / "dataset/中国_省.geojson"
JAPAN_GEO_ROOT = WORKSPACE / "dataset/external_geospatial/japan_mlit_n03_2024"
WORLD_SHP = WORKSPACE / "dataset/external_geospatial/world_natural_earth_110m_v5_1_1/ne_110m_admin_0_countries.shp"

NAVY = "#17324D"
BLUE = "#2878B5"
ORANGE = "#E76F27"
TEAL = "#139A8C"
PURPLE = "#7761A8"
GREY = "#738592"
LIGHT = "#EDF3F5"
GRID = "#D7E2E6"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def panel_label(ax, label: str, x: float = -0.06, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def clean(ax, grid="y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#DCE5E8", lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def rings_to_path(rings) -> MplPath:
    vertices, codes = [], []
    for ring in rings:
        if len(ring) < 3:
            continue
        vertices.extend(ring)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(np.asarray(vertices, float), np.asarray(codes))


def geometry_rings(geometry):
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    return []


def bbox_intersects(rings, bounds) -> bool:
    xs, ys = [], []
    for ring in rings:
        for x, y in ring[0]:
            xs.append(x); ys.append(y)
    if not xs:
        return False
    x0, x1, y0, y1 = bounds
    return max(xs) >= x0 and min(xs) <= x1 and max(ys) >= y0 and min(ys) <= y1


def add_feature(ax, geometry, face, edge="white", lw=0.25, alpha=1.0, bounds=None, z=1):
    for polygon in geometry_rings(geometry):
        if bounds is not None and not bbox_intersects([polygon], bounds):
            continue
        path = rings_to_path(polygon)
        ax.add_patch(PathPatch(path, facecolor=face, edgecolor=edge, lw=lw, alpha=alpha, zorder=z))


def save_bundle(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    rgba = plt.imread(stem.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(stem.parent / f"{stem.name}_grayscale.png", grey, cmap="gray", vmin=0, vmax=1)
    plt.close(fig)


def add_shapefile_polygon(ax, shape, face, edge="white", lw=0.25, alpha=1.0, z=1):
    """Draw one pyshp polygon while preserving multipart country geometry."""
    points = np.asarray(shape.points, float)
    parts = list(shape.parts) + [len(points)]
    for start, stop in zip(parts[:-1], parts[1:]):
        ring = points[start:stop]
        if len(ring) < 3:
            continue
        ax.fill(ring[:, 0], ring[:, 1], facecolor=face, edgecolor=edge,
                linewidth=lw, alpha=alpha, zorder=z)


def draw_effect_distribution(ax, data, durations, colour, xlim, xlabel):
    """Raw case effects with mean and 95% interval for selected durations."""
    for yi, duration in enumerate(durations):
        sub = data[np.isclose(data.duration_h, duration)].copy()
        values = sub.chronology_effect_eur_kw_year.to_numpy(float)
        jitter = np.linspace(-0.15, 0.15, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(values, yi + jitter, s=9, color=mpl.colors.to_rgba(colour, 0.42),
                   edgecolor="white", linewidth=0.2, rasterized=True, zorder=2)
        mean = values.mean()
        lo, hi = np.quantile(values, [0.025, 0.975])
        ax.plot([lo, hi], [yi, yi], color=colour, lw=1.25, zorder=3)
        ax.scatter(mean, yi, s=28, color=colour, edgecolor="white", linewidth=0.45, zorder=4)
    ax.axvline(0, color=NAVY, lw=0.7)
    ax.set_yticks(range(len(durations)), [f"{int(v)} h" for v in durations])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    clean(ax, grid="x")


def build_figure2_v48_legacy() -> None:
    europe = pd.read_csv(SRC / "figure2a_europe_map.csv")
    china = pd.read_csv(SRC / "figure2b_china_map.csv")
    japan = pd.read_csv(SRC / "figure2c_japan_map.csv")
    runs = pd.read_csv(SRC / "figure2d_three_scale_event_runs.csv")
    contract = pd.read_csv(SRC / "figure2e_three_scale_data_contract.csv")
    cmap = mpl.colormaps["cividis"]
    norm = mpl.colors.Normalize(vmin=3, vmax=23)

    fig = plt.figure(figsize=(7.205, 5.2), facecolor="white")
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1.05, 1], hspace=0.22, wspace=0.45)
    axa = fig.add_subplot(gs[0, :7])
    axb = fig.add_subplot(gs[0, 7:])
    axc = fig.add_subplot(gs[1, :7])
    axd = fig.add_subplot(gs[1, 7:10])
    axe = fig.add_subplot(gs[1, 10:])

    # Europe country map.
    lookup = europe.set_index("gisco_code").median_longest_event_days.to_dict()
    geo = json.loads(EUROPE_GEO.read_text(encoding="utf-8"))
    for feature in geo["features"]:
        code = feature["properties"].get("CNTR_ID")
        if code not in lookup:
            continue
        add_feature(axa, feature["geometry"], cmap(norm(lookup[code])), edge="white", lw=0.35)
    axa.set_xlim(-11, 31); axa.set_ylim(34, 58); axa.set_aspect("equal"); axa.axis("off")
    for row in europe.itertuples():
        axa.scatter(row.lon, row.lat, s=4 + 2.2 * np.sqrt(row.mean_annual_task_twh), color=NAVY,
                    alpha=0.72, edgecolor="white", linewidth=0.25, zorder=5)
    panel_label(axa, "a", -0.01, 0.98)

    # China province map.  The observed gas-peaking task is used directly.
    zh = {
        "Anhui":"安徽省","Beijing":"北京市","Chongqing":"重庆市","Fujian":"福建省","Gansu":"甘肃省",
        "Guangdong":"广东省","Guangxi":"广西壮族自治区","Guizhou":"贵州省","Hainan":"海南省","Hebei":"河北省",
        "Heilongjiang":"黑龙江省","Henan":"河南省","Hubei":"湖北省","Hunan":"湖南省","Inner Mongolia":"内蒙古自治区",
        "Jiangsu":"江苏省","Jiangxi":"江西省","Jilin":"吉林省","Liaoning":"辽宁省","Ningxia":"宁夏回族自治区",
        "Qinghai":"青海省","Shaanxi":"陕西省","Shandong":"山东省","Shanxi":"山西省","Sichuan":"四川省","Zhejiang":"浙江省",
        "Tianjin":"天津市","Xinjiang":"新疆维吾尔自治区","Yunnan":"云南省",
    }
    c_lookup = {zh[k]: v for k, v in china.set_index("province").longest_extreme_peaking_run_days.to_dict().items()}
    geo = json.loads(CHINA_GEO.read_text(encoding="utf-8"))
    for feature in geo["features"]:
        name = feature["properties"].get("name")
        value = c_lookup.get(name)
        add_feature(axb, feature["geometry"], cmap(norm(value)) if value is not None else "#EEF2F3", edge="white", lw=0.25)
    axb.set_xlim(73, 136); axb.set_ylim(17, 54); axb.set_aspect("equal"); axb.axis("off")
    panel_label(axb, "b", -0.02, 0.98)

    # Japan official MLIT administrative boundaries and community mesh centres.
    bounds = (137.55, 140.15, 34.45, 36.05)
    for code in (11, 12, 13, 14, 22):
        path = JAPAN_GEO_ROOT / f"N03-20240101_{code}_GML/N03-20240101_{code}.geojson"
        geo = json.loads(path.read_text(encoding="utf-8"))
        for feature in geo["features"]:
            add_feature(axc, feature["geometry"], "#F3F6F7", edge="#C9D5D9", lw=0.15, bounds=bounds)
    community = japan.groupby(["prefecture", "mesh_id", "latitude", "longitude"], as_index=False).agg(
        event_run_days=("longest_extreme_import_run_days", "median"),
        annual_import_mwh=("annual_residual_import_mwh", "mean"),
    )
    axc.scatter(community.longitude, community.latitude, c=community.event_run_days, cmap=cmap, norm=norm,
                s=11 + 1.2 * np.sqrt(community.annual_import_mwh), edgecolor="white", linewidth=0.45, zorder=6)
    axc.set_xlim(bounds[0], bounds[1]); axc.set_ylim(bounds[2], bounds[3]); axc.set_aspect("equal"); axc.axis("off")
    panel_label(axc, "c", -0.01, 0.98)
    cax = axc.inset_axes([0.58, 0.06, 0.30, 0.028])
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cax.tick_params(labelsize=5.0, length=1.5)
    cax.set_xlabel("Longest extreme-event run (d)", fontsize=5.0, labelpad=1)

    # Raw distributions, not means-only bars.
    colours = {"Europe": BLUE, "China": ORANGE, "Japan": TEAL}
    ypos = {"Europe": 2, "China": 1, "Japan": 0}
    for scale, group in runs.groupby("scale"):
        # Deterministic spreading exposes every observation without simulating data.
        y = ypos[scale] + np.linspace(-0.12, 0.12, len(group))
        axd.scatter(group.event_run_days, y, s=10, color=colours[scale], alpha=0.70,
                    edgecolor="white", linewidth=0.25)
        med = group.event_run_days.median()
        axd.plot([med, med], [ypos[scale]-0.22, ypos[scale]+0.22], color=NAVY, lw=1.2)
    axd.set_yticks([0,1,2], ["Japan", "China", "Europe"])
    axd.set_xlabel("Longest extreme-event run (d)")
    axd.set_xlim(0, 25)
    clean(axd, grid="x"); panel_label(axd, "d", -0.16, 1.02)

    # Compact data-contract card; definitions differ by empirical scale.
    axe.axis("off"); panel_label(axe, "e", -0.15, 1.02)
    short = {
        "Europe": ("hourly gas above q10", "wind–solar balance", BLUE),
        "China": ("observed gas peaking", "wind–solar curtailment", ORANGE),
        "Japan": ("community import residual", "PV reverse flow", TEAL),
    }
    for j, row in enumerate(contract.itertuples()):
        y = 0.75 - 0.29*j
        task, recharge, colour = short[row.scale]
        card = FancyBboxPatch((0.02,y),0.95,0.22,boxstyle="round,pad=0.012,rounding_size=0.025",
                              transform=axe.transAxes,facecolor=mpl.colors.to_rgba(colour,0.08),edgecolor=colour,lw=0.7)
        axe.add_patch(card)
        axe.text(0.07,y+0.165,row.scale,transform=axe.transAxes,fontweight="bold",color=colour,fontsize=6.0)
        axe.text(0.07,y+0.105,task,transform=axe.transAxes,color=NAVY,fontsize=5.0)
        axe.text(0.07,y+0.054,recharge,transform=axe.transAxes,color=GREY,fontsize=5.0)
        axe.text(0.93,y+0.165,f"n={row.case_years}",transform=axe.transAxes,ha="right",color=NAVY,fontsize=5.0)

    fig.subplots_adjust(left=0.035, right=0.985, top=0.98, bottom=0.06)
    save_bundle(fig, OUT / "fig_v48_figure2_three_scale_empirical_settings")


def build_figure5_v48_legacy() -> None:
    europe = pd.read_csv(SRC / "figure5a_europe_duration_response.csv").sort_values("duration_h")
    cross = pd.read_csv(SRC / "figure5bc_china_japan_duration_response.csv")
    counts = pd.read_csv(SRC / "figure5d_three_scale_strict_valley_counts.csv")
    phase_eu = pd.read_csv(ROOT / "source_data/v42_fixed_scale_mix_efc_cost_firm_phase_formation_rate.csv")
    phase_cross = pd.read_csv(ROOT / "cross_scale_cases/summary_v0_2/cross_scale_efc_phase_scan.csv")

    fig = plt.figure(figsize=(7.205, 7.0), facecolor="white")
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.05, 0.9, 0.92], hspace=0.44, wspace=0.80)
    axa = fig.add_subplot(gs[0, :8]); axp = fig.add_subplot(gs[0, 9:])
    axb = fig.add_subplot(gs[1, :6]); axc = fig.add_subplot(gs[1, 6:])
    axd = fig.add_subplot(gs[2, :7]); axe = fig.add_subplot(gs[2, 8:])

    def duration_panel(ax, data, colour, ylim, label):
        # Strictly positive duration guard for the logarithmic axis.
        assert (data.duration_h > 0).all(), "Log-duration panels require positive durations"
        ax.fill_between(data.duration_h, data.ci95_low if "ci95_low" in data else data.ci_low,
                        data.ci95_high if "ci95_high" in data else data.ci_high,
                        color=colour, alpha=0.15, linewidth=0)
        y = data.mean_effect_eur_kw_year if "mean_effect_eur_kw_year" in data else data.point_estimate
        ax.plot(data.duration_h, y, color=colour, lw=1.7, marker="o", ms=2.8, mec="white", mew=0.4)
        ax.axhline(0, color=NAVY, lw=0.75)
        ax.set_xscale("log"); ax.set_xlim(max(4,data.duration_h.min()), data.duration_h.max()*1.04); ax.set_ylim(*ylim)
        ax.set_xlabel("Storage duration (h)"); ax.set_ylabel("Chronology effect\n(EUR kW$^{-1}$ yr$^{-1}$)")
        ax.text(0.02,0.94,label,transform=ax.transAxes,va="top",fontweight="bold",color=NAVY)
        clean(ax)

    duration_panel(axa, europe, BLUE, (-11.5, 8.5), "Europe · national energy balance")
    axa.axvspan(28,40,color=ORANGE,alpha=0.08)
    axa.set_xticks([4,8,16,24,40,72,96,168],["4","8","16","24","40","72","96","168"])
    panel_label(axa,"a")

    # Price-channel robustness retained from v47 as a compact inset.
    price = pd.read_csv(ROOT / "price_channel/empirical_summary_v0_1/price_channel_caes24_summary.csv")
    s3b = pd.read_csv(ROOT / "price_channel/empirical_reoptimized_summary_v0_1/s3b_caes24_summary.csv")
    price = pd.concat([price,s3b],ignore_index=True)
    scenarios=["S0","S1_beta_1","S2_surplus_linked_charge","S3a_empirical_price","S3b_empirical_reoptimized"]
    labels=["fixed","event uplift","surplus price","empirical fixed","empirical optimized"]
    p=price.set_index("scenario").loc[scenarios].reset_index(); y=np.arange(5)[::-1]
    axp.axvline(0,color=NAVY,lw=0.75)
    for yi,row,col in zip(y,p.itertuples(),[GREY,ORANGE,TEAL,PURPLE,NAVY]):
        axp.plot([row.ci95_low,row.ci95_high],[yi,yi],color=col,lw=1.3)
        axp.scatter(row.mean_effect_eur_kw_year,yi,s=23,color=col,edgecolor="white",lw=0.4,zorder=3)
    axp.set_yticks(y,labels); axp.set_xlabel("24 h effect (EUR kW$^{-1}$ yr$^{-1}$)")
    clean(axp,grid="x"); panel_label(axp,"b",-0.20,1.04)

    china=cross[cross.application_case.eq("china")].sort_values("duration_h")
    japan=cross[cross.application_case.eq("japan")].sort_values("duration_h")
    duration_panel(axb,china,ORANGE,(-86,30),"China · provincial gas peaking")
    duration_panel(axc,japan,TEAL,(-17,19),"Japan · community PV prosumption")
    axb.set_xticks([8,16,24,40,72,168,336,720],["8","16","24","40","72","168","336","720"])
    axc.set_xticks([8,16,24,40,72,168,336,720],["8","16","24","40","72","168","336","720"])
    panel_label(axb,"c"); panel_label(axc,"d")

    # Topology changes in selected, fully specified economic states.
    selected = counts[counts.selection_label.isin([
        "E=98.54, Q=0, F=600, 2035 costs", "China central 2030",
        "China clustered-valley phase", "Japan clustered-valley phase",
    ])].copy()
    order=["Europe\nservice phase","China\n2030 central","China\nservice phase","Japan\nservice phase"]
    label_map={
        "E=98.54, Q=0, F=600, 2035 costs":order[0],"China central 2030":order[1],
        "China clustered-valley phase":order[2],"Japan clustered-valley phase":order[3],
    }
    selected["plot_label"]=selected.selection_label.map(label_map)
    xpos=np.arange(4)
    for j,chron in enumerate(["dispersed","clustered"]):
        sub=selected[selected.chronology.eq(chron)].set_index("plot_label").reindex(order)
        vals=100*sub.strict_valleys/sub.n_cases
        x=xpos+(j-0.5)*0.19
        axd.scatter(x,vals,s=35,marker="o" if chron=="dispersed" else "s",
                    color=BLUE if chron=="dispersed" else ORANGE,edgecolor="white",lw=0.5,zorder=4,label=chron.capitalize())
        for xx,v,n,k in zip(x,vals,sub.n_cases,sub.strict_valleys):
            axd.text(xx,v+4.5,f"{int(k)}/{int(n)}",ha="center",fontsize=5.0,color=NAVY)
    axd.set_xticks(xpos,order); axd.set_ylabel("Cases with a strict valley (%)"); axd.set_ylim(-4,112)
    axd.legend(frameon=False,ncol=2,loc="upper center"); clean(axd); panel_label(axd,"e")

    # Formation thresholds at unchanged 2035 LDES costs.
    eu=phase_eu[
        phase_eu.source_label.eq("fixed_s20_central")
        & phase_eu.model_mode.eq("fixed")
        & phase_eu.chronology.eq("clustered")
        & np.isclose(phase_eu.energy_value_eur_mwh,98.540318399)
        & np.isclose(phase_eu.charge_price_eur_mwh,0)
        & np.isclose(phase_eu.ldes_tail_fixed_cost_multiplier,1)
    ].sort_values("firm_value_eur_kw_year")
    axe.plot(eu.firm_value_eur_kw_year,100*eu.strict_valley_rate,color=BLUE,lw=1.6,label="Europe")
    for app,col,label in [("china",ORANGE,"China"),("japan",TEAL,"Japan")]:
        sub=phase_cross[phase_cross.application_case.eq(app)&phase_cross.chronology.eq("clustered")&np.isclose(phase_cross.ldes_cost_multiplier,1)].sort_values("firm_value_eur_kw_year")
        axe.plot(sub.firm_value_eur_kw_year,100*sub.strict_valleys/sub.n_case_years,color=col,lw=1.6,label=label)
    axe.set_xlim(0,610); axe.set_ylim(-3,104); axe.set_xlabel("Firm-supply value (EUR kW$^{-1}$ yr$^{-1}$)")
    axe.set_ylabel("Strict valleys (%)"); axe.legend(frameon=False,loc="upper left"); clean(axe); panel_label(axe,"f",-0.18,1.04)

    fig.subplots_adjust(left=0.085,right=0.985,top=0.975,bottom=0.07)
    save_bundle(fig,OUT/"fig_v48_figure5_three_scale_valley")


def build_figure2() -> None:
    """World-led spatial atlas with nested province and community evidence."""
    europe = pd.read_csv(SRC / "figure2a_europe_map.csv")
    china = pd.read_csv(SRC / "figure2b_china_map.csv")
    japan = pd.read_csv(SRC / "figure2c_japan_map.csv")
    runs = pd.read_csv(SRC / "figure2d_three_scale_event_runs.csv")

    fig = plt.figure(figsize=(7.205, 5.75), facecolor="white")
    gs = GridSpec(2, 12, figure=fig, height_ratios=[2.05, 1.0], hspace=0.16, wspace=0.50)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, :4])
    axc = fig.add_subplot(gs[1, 4:8])
    axd = fig.add_subplot(gs[1, 8:])

    # a, world context. Natural Earth is the global base; thematic detail is
    # supplied by the audited European, Chinese and Japanese source tables.
    reader = shapefile.Reader(str(WORLD_SHP), encoding="utf-8")
    eu_codes = set(europe.gisco_code)
    europe_lookup = europe.set_index("gisco_code").median_longest_event_days.to_dict()
    eu_norm = mpl.colors.Normalize(vmin=4, vmax=23)
    eu_cmap = mpl.colors.LinearSegmentedColormap.from_list("eu_blue", ["#D9EAF3", BLUE, NAVY])
    for record in reader.iterShapeRecords():
        props = record.record.as_dict()
        code = props.get("ISO_A2")
        face, edge, lw, z = "#F1F4F5", "white", 0.20, 1
        if code in eu_codes:
            face = eu_cmap(eu_norm(europe_lookup[code])); edge = "white"; lw = 0.32; z = 3
        elif code == "CN":
            face = mpl.colors.to_rgba(ORANGE, 0.80); edge = "white"; lw = 0.35; z = 3
        elif code == "JP":
            face = mpl.colors.to_rgba(TEAL, 0.90); edge = "white"; lw = 0.35; z = 4
        add_shapefile_polygon(axa, record.shape, face, edge=edge, lw=lw, z=z)
    axa.set_xlim(-180, 180); axa.set_ylim(-58, 84); axa.set_aspect("equal", adjustable="box"); axa.axis("off")
    axa.scatter(europe.lon, europe.lat,
                s=5 + 1.8 * np.sqrt(europe.mean_annual_task_twh), color=NAVY,
                edgecolor="white", linewidth=0.35, alpha=0.82, zorder=8)
    axa.scatter([104.0], [35.0], s=34, marker="s", color=ORANGE,
                edgecolor="white", linewidth=0.55, zorder=9)
    axa.scatter([138.0], [37.0], s=38, marker="D", color=TEAL,
                edgecolor="white", linewidth=0.55, zorder=9)
    panel_label(axa, "a", 0.005, 0.96)

    # Direct labels make the global hierarchy legible without a detached legend.
    callouts = [
        (7.5, 54.0, -18, 70, "Europe", "15 countries · 132 country–years", BLUE),
        (104.0, 35.0, 68, 4, "China", "26 provinces · observed gas peaking", ORANGE),
        (138.0, 37.0, 151, 17, "Japan", "19 communities · 27 community–years", TEAL),
    ]
    for x, y, tx, ty, head, sub, colour in callouts:
        axa.annotate("", xy=(x, y), xytext=(tx, ty),
                     arrowprops=dict(arrowstyle="-", lw=0.75, color=colour,
                                     connectionstyle="arc3,rad=0.12"), zorder=10)
        axa.text(tx, ty + 3.4, head, ha="center", va="bottom", color=colour,
                 fontsize=6.7, fontweight="bold", zorder=10)
        axa.text(tx, ty - 1.0, sub, ha="center", va="top", color=NAVY,
                 fontsize=5.3, zorder=10)
    cax = axa.inset_axes([0.065, 0.08, 0.18, 0.025])
    mpl.colorbar.ColorbarBase(cax, cmap=eu_cmap, norm=eu_norm, orientation="horizontal")
    cax.tick_params(labelsize=5.0, length=1.5)
    cax.set_xlabel("Europe · median longest event (d)", fontsize=5.0, labelpad=1)

    # b, Chinese provincial detail. The input is already a peaking-task series;
    # q75 labels extreme peaking days and no q10 baseline is subtracted.
    zh = {
        "Anhui":"安徽省","Beijing":"北京市","Chongqing":"重庆市","Fujian":"福建省","Gansu":"甘肃省",
        "Guangdong":"广东省","Guangxi":"广西壮族自治区","Guizhou":"贵州省","Hainan":"海南省","Hebei":"河北省",
        "Heilongjiang":"黑龙江省","Henan":"河南省","Hubei":"湖北省","Hunan":"湖南省","Inner Mongolia":"内蒙古自治区",
        "Jiangsu":"江苏省","Jiangxi":"江西省","Jilin":"吉林省","Liaoning":"辽宁省","Ningxia":"宁夏回族自治区",
        "Qinghai":"青海省","Shaanxi":"陕西省","Shandong":"山东省","Shanxi":"山西省","Sichuan":"四川省","Zhejiang":"浙江省",
        "Tianjin":"天津市","Xinjiang":"新疆维吾尔自治区","Yunnan":"云南省",
    }
    c_lookup = {zh[k]: v for k, v in china.set_index("province").longest_extreme_peaking_run_days.to_dict().items()}
    c_norm = mpl.colors.Normalize(vmin=3, vmax=6)
    c_cmap = mpl.colors.LinearSegmentedColormap.from_list("china_orange", ["#FBE5D9", "#F2A36F", ORANGE])
    geo = json.loads(CHINA_GEO.read_text(encoding="utf-8"))
    for feature in geo["features"]:
        name = feature["properties"].get("name")
        value = c_lookup.get(name)
        add_feature(axb, feature["geometry"], c_cmap(c_norm(value)) if value is not None else "#F0F3F4",
                    edge="white", lw=0.22)
    axb.set_xlim(73, 136); axb.set_ylim(17, 54); axb.set_aspect("equal"); axb.axis("off")
    panel_label(axb, "b", -0.01, 0.98)
    cb = axb.inset_axes([0.09, 0.01, 0.53, 0.035])
    mpl.colorbar.ColorbarBase(cb, cmap=c_cmap, norm=c_norm, orientation="horizontal")
    cb.set_xticks([3, 4, 5, 6]); cb.tick_params(labelsize=5.0, length=1.2)
    cb.set_xlabel("Longest extreme gas-peaking event (d)", fontsize=5.0, labelpad=1)
    axb.text(0.98, 0.06, "26 provinces\n10.71 TWh observed peaking task",
             transform=axb.transAxes, ha="right", va="bottom", fontsize=5.1, color=NAVY)

    # c, official Japan boundaries with a community-scale zoom.
    bounds = (137.55, 140.15, 34.45, 36.05)
    for code in (11, 12, 13, 14, 22):
        path = JAPAN_GEO_ROOT / f"N03-20240101_{code}_GML/N03-20240101_{code}.geojson"
        geo = json.loads(path.read_text(encoding="utf-8"))
        for feature in geo["features"]:
            add_feature(axc, feature["geometry"], "#F3F6F7", edge="#C7D2D6", lw=0.14, bounds=bounds)
    community = japan.groupby(["prefecture", "mesh_id", "latitude", "longitude"], as_index=False).agg(
        event_run_days=("longest_extreme_import_run_days", "median"),
        annual_import_mwh=("annual_residual_import_mwh", "mean"),
    )
    j_norm = mpl.colors.Normalize(vmin=6, vmax=21)
    j_cmap = mpl.colors.LinearSegmentedColormap.from_list("japan_teal", ["#D8F0EC", "#68C2B8", TEAL, "#075E58"])
    axc.scatter(community.longitude, community.latitude, c=community.event_run_days,
                cmap=j_cmap, norm=j_norm, s=10 + 1.15 * np.sqrt(community.annual_import_mwh),
                edgecolor="white", linewidth=0.4, zorder=6)
    axc.set_xlim(*bounds[:2]); axc.set_ylim(*bounds[2:]); axc.set_aspect("equal"); axc.axis("off")
    panel_label(axc, "c", -0.01, 0.98)
    cj = axc.inset_axes([0.08, 0.01, 0.53, 0.035])
    mpl.colorbar.ColorbarBase(cj, cmap=j_cmap, norm=j_norm, orientation="horizontal")
    cj.set_xticks([6, 11, 16, 21]); cj.tick_params(labelsize=5.0, length=1.2)
    cj.set_xlabel("Longest extreme community-import event (d)", fontsize=5.0, labelpad=1)
    axc.text(0.98, 0.06, "5 prefectures · 19 meshes\n236,520 community–hours",
             transform=axc.transAxes, ha="right", va="bottom", fontsize=5.1, color=NAVY)

    # d, every observed event-run statistic is shown; medians are dark strokes.
    colours = {"Europe": BLUE, "China": ORANGE, "Japan": TEAL}
    ypos = {"Europe": 2, "China": 1, "Japan": 0}
    for scale, group in runs.groupby("scale"):
        y = ypos[scale] + np.linspace(-0.13, 0.13, len(group))
        axd.scatter(group.event_run_days, y, s=10, color=colours[scale], alpha=0.66,
                    edgecolor="white", linewidth=0.25)
        med = group.event_run_days.median()
        axd.plot([med, med], [ypos[scale]-0.24, ypos[scale]+0.24], color=NAVY, lw=1.25)
    axd.set_yticks([0, 1, 2], ["Japan", "China", "Europe"])
    axd.set_xlabel("Longest extreme-event run (d)")
    axd.set_xlim(0, 25); clean(axd, grid="x"); panel_label(axd, "d", -0.13, 1.02)

    fig.subplots_adjust(left=0.028, right=0.985, top=0.985, bottom=0.055)
    save_bundle(fig, OUT / "fig_v49_figure2_world_three_scale_empirical_settings")


def build_figure5() -> None:
    """Eight-panel cross-scale mechanism and strict-valley evidence figure."""
    europe = pd.read_csv(SRC / "figure5a_europe_duration_response.csv").sort_values("duration_h")
    cross_summary = pd.read_csv(SRC / "figure5bc_china_japan_duration_response.csv")
    paired = pd.read_csv(ROOT / "cross_scale_cases/summary_v0_2/cross_scale_efc_paired_duration.csv.gz")
    selected_records = pd.read_csv(SRC / "figure5e_selected_cross_scale_valley_records.csv")
    decomp = pd.read_csv(ROOT / "cost_and_decomposition/figure5c_channel_decomposition_wide.csv")
    eu_records = pd.read_csv(HERE / "source_data/figure5_panel_e_conditional_valley_records.csv")

    fig = plt.figure(figsize=(7.205, 8.75), facecolor="white")
    gs = GridSpec(4, 12, figure=fig, height_ratios=[1.06, 0.78, 0.92, 0.88], hspace=0.52, wspace=0.82)
    axa = fig.add_subplot(gs[0, :8])
    axb = fig.add_subplot(gs[0, 8:])
    axc = fig.add_subplot(gs[1, :])
    axd = fig.add_subplot(gs[2, :7])
    axe = fig.add_subplot(gs[2, 7:])
    axf = fig.add_subplot(gs[3, :5])
    axg = fig.add_subplot(gs[3, 5:8])
    axh = fig.add_subplot(gs[3, 8:])

    # a, European mean response and country-block bootstrap interval.
    axa.fill_between(europe.duration_h, europe.ci_low, europe.ci_high,
                     color=BLUE, alpha=0.15, linewidth=0)
    axa.plot(europe.duration_h, europe.point_estimate, color=BLUE, lw=1.65,
             marker="o", ms=2.7, mec="white", mew=0.35)
    axa.axhline(0, color=NAVY, lw=0.72)
    axa.axvspan(28, 40, color=ORANGE, alpha=0.08)
    axa.set_xscale("log"); axa.set_xlim(4, 174); axa.set_ylim(-11.2, 8.3)
    axa.set_xticks([4, 8, 16, 24, 40, 72, 96, 168], ["4", "8", "16", "24", "40", "72", "96", "168"])
    axa.set_xlabel("Storage duration (h)")
    axa.set_ylabel("Chronology effect\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axa.text(0.02, 0.95, "Europe · cross-duration value reversal", transform=axa.transAxes,
             va="top", color=NAVY, fontweight="bold")
    clean(axa); panel_label(axa, "a")

    # b, exact two-channel decomposition of the same response.
    selected_h = [24, 32, 40, 96, 168]
    ds = decomp.set_index("duration_h").loc[selected_h].reset_index()
    x = np.arange(len(ds))
    cycling = ds.cycling_channel_country_block_mean.to_numpy()
    coverage = ds.coverage_channel_country_block_mean.to_numpy()
    for values, colour, label in [(cycling, BLUE, "cycling return"), (coverage, PURPLE, "coverage return")]:
        positive = np.where(values > 0, values, 0)
        negative = np.where(values < 0, values, 0)
        bottom_pos = np.zeros_like(values) if label == "cycling return" else np.where(cycling > 0, cycling, 0)
        bottom_neg = np.zeros_like(values) if label == "cycling return" else np.where(cycling < 0, cycling, 0)
        axb.bar(x, positive, bottom=bottom_pos, width=0.62, color=colour, alpha=0.88, label=label)
        axb.bar(x, negative, bottom=bottom_neg, width=0.62, color=colour, alpha=0.88)
    axb.scatter(x, ds.total_effect_country_block_mean, marker="D", s=18,
                color=NAVY, edgecolor="white", lw=0.35, zorder=5, label="total")
    axb.axhline(0, color=NAVY, lw=0.72)
    axb.set_xticks(x, [f"{v} h" for v in selected_h], rotation=28, ha="right")
    axb.set_ylabel("Effect (EUR kW$^{-1}$ yr$^{-1}$)")
    axb.legend(frameon=False, fontsize=5.0, ncol=1, loc="upper left")
    clean(axb); panel_label(axb, "b", -0.16, 1.04)

    # c, each horizontal segment is one strict European mid-duration valley.
    eu_strict = eu_records[(eu_records.chronology == "clustered") & eu_records.strict_valley].copy()
    eu_strict = eu_strict.sort_values(["left_boundary_h", "right_boundary_h", "country", "year"]).reset_index(drop=True)
    for yi, row in eu_strict.iterrows():
        axc.plot([row.left_boundary_h, row.right_boundary_h], [yi, yi],
                 color=mpl.colors.to_rgba(BLUE, 0.58), lw=1.25, solid_capstyle="round")
        axc.scatter([row.left_boundary_h, row.right_boundary_h], [yi, yi], s=6,
                    color=[ORANGE, PURPLE], edgecolor="white", linewidth=0.2, zorder=4)
    axc.set_xscale("log"); axc.set_xlim(18, 190); axc.set_ylim(-3, len(eu_strict)+2)
    axc.set_xticks([24, 32, 40, 60, 96, 120, 168], ["24", "32", "40", "60", "96", "120", "168"])
    axc.set_yticks([]); axc.set_xlabel("Strict-valley interval across storage duration (h)")
    axc.text(0.01, 0.93, f"Europe · 85 of 132 clustered country–years", transform=axc.transAxes,
             color=NAVY, fontweight="bold", va="top")
    axc.text(0.99, 0.93, "left entry boundary   →   long-duration re-entry", transform=axc.transAxes,
             color=GREY, ha="right", va="top", fontsize=5.0)
    clean(axc, grid="x"); panel_label(axc, "c", -0.035, 1.03)

    # d, all 26 Chinese provincial effects at the durations that define the trough and recovery.
    china_raw = paired[(paired.application_case == "china") & (paired.scenario == "S0") & (paired.cost_year == 2035)]
    draw_effect_distribution(axd, china_raw, [24, 40, 168, 240], ORANGE, (-115, 45),
                             "China chronology effect (EUR kW$^{-1}$ yr$^{-1}$)")
    axd.text(0.02, 0.94, "China · observed provincial gas peaking", transform=axd.transAxes,
             va="top", color=NAVY, fontweight="bold")
    panel_label(axd, "d", -0.08, 1.04)

    # e, the same Chinese physical response changes topology across two value structures.
    china_records = selected_records[selected_records.application_case == "china"]
    structures = ["China central 2030", "China clustered-valley phase"]
    labels = ["2030 central", "2035 coverage"]
    for yi, (structure, label) in enumerate(zip(structures, labels)):
        sub = china_records[china_records.selection_label == structure]
        d = 100 * sub[sub.chronology == "dispersed"].strict_valley.mean()
        c = 100 * sub[sub.chronology == "clustered"].strict_valley.mean()
        ax_e_colour = ORANGE if c >= d else BLUE
        axe.annotate("", xy=(c, yi), xytext=(d, yi),
                     arrowprops=dict(arrowstyle="-|>", color=ax_e_colour, lw=1.35, mutation_scale=8))
        axe.scatter(d, yi, s=30, color=BLUE, edgecolor="white", lw=0.45, zorder=4)
        axe.scatter(c, yi, s=30, marker="s", color=ORANGE, edgecolor="white", lw=0.45, zorder=4)
        axe.text(d, yi+0.17, f"{int(round(d*26/100))}/26", ha="center", fontsize=5.0, color=BLUE)
        axe.text(c, yi-0.18, f"{int(round(c*26/100))}/26", ha="center", va="top", fontsize=5.0, color=ORANGE)
    axe.set_yticks([0, 1], labels); axe.set_xlim(-6, 106); axe.set_ylim(-0.45, 1.45)
    axe.set_xlabel("Provinces with a strict valley (%)")
    clean(axe, grid="x"); panel_label(axe, "e", -0.19, 1.04)

    # f, all 27 Japanese community-year effects at the observed sign-reversal nodes.
    japan_raw = paired[(paired.application_case == "japan") & (paired.scenario == "S0") & (paired.cost_year == 2035)]
    draw_effect_distribution(axf, japan_raw, [32, 72], TEAL, (-28, 24),
                             "Japan effect\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axf.text(0.02, 0.94, "Japan · community PV", transform=axf.transAxes,
             va="top", color=NAVY, fontweight="bold")
    panel_label(axf, "f", -0.14, 1.04)

    # g, clustered strict-valley incidence by Japanese prefecture.
    japan_records = selected_records[(selected_records.application_case == "japan") &
                                      (selected_records.selection_label == "Japan clustered-valley phase") &
                                      (selected_records.chronology == "clustered")].copy()
    japan_records["prefecture"] = japan_records.case_id.str.split("-").str[1]
    pref = japan_records.groupby("prefecture").strict_valley.agg(["sum", "size"]).sort_values("sum")
    yy = np.arange(len(pref))
    rates = 100 * pref["sum"] / pref["size"]
    axg.hlines(yy, 0, rates, color=mpl.colors.to_rgba(TEAL, 0.45), lw=3.0)
    axg.scatter(rates, yy, s=24, color=TEAL, edgecolor="white", lw=0.45, zorder=4)
    for yv, rate, k, n in zip(yy, rates, pref["sum"], pref["size"]):
        axg.text(rate + 3, yv, f"{int(k)}/{int(n)}", va="center", fontsize=5.0, color=NAVY)
    axg.set_yticks(yy, pref.index); axg.set_xlim(0, 112)
    axg.set_xlabel("Clustered strict valleys (%)")
    clean(axg, grid="x"); panel_label(axg, "g", -0.24, 1.04)

    # h, cross-scale width distributions show the economic interval opened by clustering.
    width_groups = [
        ("Europe", eu_strict.valley_width_h.to_numpy(float), BLUE),
        ("China", china_records[(china_records.selection_label == "China clustered-valley phase") &
                                 (china_records.chronology == "clustered") &
                                 china_records.strict_valley].valley_width_h.to_numpy(float), ORANGE),
        ("Japan", japan_records[japan_records.strict_valley].valley_width_h.to_numpy(float), TEAL),
    ]
    for xi, (name, values, colour) in enumerate(width_groups):
        jitter = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else np.zeros(1)
        axh.scatter(xi + jitter, values, s=8, color=mpl.colors.to_rgba(colour, 0.45),
                    edgecolor="white", linewidth=0.2, rasterized=True)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axh.plot([xi, xi], [q1, q3], color=colour, lw=4.0, solid_capstyle="round")
        axh.scatter(xi, med, s=26, color=NAVY, edgecolor="white", lw=0.4, zorder=5)
    axh.set_xticks(range(3), ["Europe\n85/132", "China\n26/26", "Japan\n8/27"])
    axh.set_ylabel("Strict-valley width (h)")
    clean(axh); panel_label(axh, "h", -0.18, 1.04)

    fig.subplots_adjust(left=0.082, right=0.985, top=0.982, bottom=0.055)
    save_bundle(fig, OUT / "fig_v49_figure5_three_scale_mid_duration_valley")


def main() -> None:
    setup(); build_figure2(); build_figure5()
    manifest={
        "figure2_archetype":"world-led asymmetric spatial atlas with province and community zooms",
        "figure5_archetype":"eight-panel cross-scale mechanism and strict-valley evidence composite",
        "source_directory":str(SRC),
        "independent_blocks":{"Europe":15,"China":26,"Japan":5},
        "case_years":{"Europe":132,"China":26,"Japan":27},
        "world_boundary_source":"Natural Earth admin-0 countries v5.1.1",
        "japan_boundary_source":"MLIT National Land Numerical Information N03-2024",
    }
    (SRC/"v49_figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    main()
