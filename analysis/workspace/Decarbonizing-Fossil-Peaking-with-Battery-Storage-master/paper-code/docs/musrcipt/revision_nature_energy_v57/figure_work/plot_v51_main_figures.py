#!/usr/bin/env python3
"""Build the v51 mixed-modality Nature Energy main figures 2–6.

Every panel reads a registered table produced by the existing analysis.  The
script changes visual form and hierarchy only; it does not recompute model
results or synthesize observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, PathPatch, Patch, Rectangle
from matplotlib.path import Path as MplPath


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parents[4]
SRC = HERE / "source_data"
SRC49 = HERE / "source_data_v49"
OUT = HERE / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

WORLD_SHP = WORKSPACE / "dataset/external_geospatial/world_natural_earth_110m_v5_1_1/ne_110m_admin_0_countries.shp"
CHINA_GEO = WORKSPACE / "dataset/中国_省.geojson"
JAPAN_GEO_ROOT = WORKSPACE / "dataset/external_geospatial/japan_mlit_n03_2024"

INK = "#203747"
MUTED = "#667985"
GRID = "#DCE5E8"
PALE = "#F3F6F7"
BLUE = "#2C78A8"
ORANGE = "#E26D32"
TEAL = "#168F83"
PURPLE = "#7259A5"
GOLD = "#D39A2C"
RED = "#B8525A"


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.labelsize": 6.7,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.5,
        "axes.linewidth": 0.55,
        "lines.linewidth": 1.15,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def panel(ax, letter: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="bottom")


def clean(ax, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.48, alpha=0.85)
    ax.set_axisbelow(True)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(OUT / f"{stem}_grayscale.png", grey, cmap="gray", vmin=0, vmax=1)
    plt.close(fig)


def geometry_rings(geometry):
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    return []


def add_geojson(ax, geometry, face, edge="white", lw=0.2, bounds=None, z=1):
    for polygon in geometry_rings(geometry):
        ring = np.asarray(polygon[0], float)
        if len(ring) < 3:
            continue
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            if ring[:, 0].max() < x0 or ring[:, 0].min() > x1 or ring[:, 1].max() < y0 or ring[:, 1].min() > y1:
                continue
        vertices = np.vstack([ring, ring[0]])
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=face, edgecolor=edge, lw=lw, zorder=z))


def add_shape(ax, shape, face, edge="white", lw=0.2, z=1):
    pts = np.asarray(shape.points, float)
    parts = list(shape.parts) + [len(pts)]
    for lo, hi in zip(parts[:-1], parts[1:]):
        ring = pts[lo:hi]
        if len(ring) >= 3:
            ax.fill(ring[:, 0], ring[:, 1], facecolor=face, edgecolor=edge, lw=lw, zorder=z)


def deterministic_jitter(n: int, width: float = 0.14) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    phase = (np.arange(n) * 0.61803398875) % 1
    return (2 * phase - 1) * width


def duration_edges(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(sorted(set(values)), dtype=float)
    if len(vals) == 1:
        return np.array([vals[0] * 0.85, vals[0] * 1.15])
    mids = np.sqrt(vals[:-1] * vals[1:])
    return np.r_[vals[0] ** 2 / mids[0], mids, vals[-1] ** 2 / mids[-1]]


def summary_segment(ax, vals, y, colour, label=None, height=0.13, s=7) -> None:
    vals = np.asarray(vals, dtype=float)
    ax.scatter(vals, y + deterministic_jitter(len(vals), height), s=s,
               color=mpl.colors.to_rgba(colour, 0.34), edgecolor="none")
    q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
    ax.plot([q1, q3], [y, y], color=colour, lw=3.4, solid_capstyle="round")
    ax.scatter(med, y, s=24, color=INK, edgecolor="white", lw=0.35, zorder=5)
    if label:
        ax.text(q3, y + 0.22, label, color=colour, fontsize=5.1, ha="right")


def flow_ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, colour, alpha=0.18, zorder=1) -> None:
    """Draw a smooth allocation ribbon between two vertical intervals."""
    dx = (x1 - x0) * 0.46
    verts = [
        (x0, y0a), (x0 + dx, y0a), (x1 - dx, y1a), (x1, y1a),
        (x1, y1b), (x1 - dx, y1b), (x0 + dx, y0b), (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=colour, edgecolor="none",
                               alpha=alpha, zorder=zorder))


def figure2() -> None:
    eu = pd.read_csv(SRC49 / "figure2a_europe_map.csv")
    cn = pd.read_csv(SRC49 / "figure2b_china_map.csv")
    jp = pd.read_csv(SRC49 / "figure2c_japan_map.csv")
    runs = pd.read_csv(SRC49 / "figure2d_three_scale_event_runs.csv")
    eu_cy = pd.read_csv(SRC / "figure2_panel_b_country_year_geometry.csv")
    recovery = pd.read_csv(SRC / "figure2_panel_f_recovery_window_ecdf.csv")
    correlations = pd.read_csv(ROOT / "sequencing_parity/three_regime_task_event_correlations.csv")

    fig = plt.figure(figsize=(7.20, 6.35))
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.0, 1.0, 0.90], hspace=0.28, wspace=0.65)
    axa = fig.add_subplot(gs[0, :8])
    axb = fig.add_subplot(gs[0, 8:])
    axc = fig.add_subplot(gs[1, :4])
    axd = fig.add_subplot(gs[1, 4:8])
    axe = fig.add_subplot(gs[1, 8:])
    axf = fig.add_subplot(gs[2, :7])
    axg = fig.add_subplot(gs[2, 8:])

    reader = shapefile.Reader(str(WORLD_SHP), encoding="utf-8")
    eu_codes = set(eu.gisco_code)
    for sr in reader.iterShapeRecords():
        code = sr.record.as_dict().get("ISO_A2")
        face, edge, lw, z = "#EEF2F3", "white", 0.16, 1
        if code in eu_codes:
            face, z = mpl.colors.to_rgba(BLUE, 0.72), 3
        elif code == "CN":
            face, z = mpl.colors.to_rgba(ORANGE, 0.80), 4
        elif code == "JP":
            face, z = mpl.colors.to_rgba(TEAL, 0.92), 4
        add_shape(axa, sr.shape, face, edge, lw, z)
    axa.set_xlim(-25, 160); axa.set_ylim(12, 75); axa.set_aspect("equal"); axa.axis("off")
    axa.scatter(eu.lon, eu.lat, s=7 + 2 * np.sqrt(eu.mean_annual_task_twh), color=INK,
                edgecolor="white", lw=0.3, zorder=8)
    for x, y, tx, ty, name, sub, colour in [
        (8, 51, -8, 68, "Europe", "15 countries · 132 country–years", BLUE),
        (104, 35, 92, 18, "China", "26 provinces", ORANGE),
        (138, 37, 149, 57, "Japan", "19 communities · 27 community–years", TEAL),
    ]:
        axa.annotate("", xy=(x, y), xytext=(tx, ty), arrowprops=dict(arrowstyle="-", color=colour, lw=0.7))
        axa.text(tx, ty + 2.0, name, ha="center", color=colour, fontweight="bold", fontsize=6.6)
        axa.text(tx, ty - 1.2, sub, ha="center", color=INK, fontsize=5.2)
    panel(axa, "a", 0.0, 0.98)

    zh = {"Anhui":"安徽省","Beijing":"北京市","Chongqing":"重庆市","Fujian":"福建省","Gansu":"甘肃省",
          "Guangdong":"广东省","Guangxi":"广西壮族自治区","Guizhou":"贵州省","Hainan":"海南省","Hebei":"河北省",
          "Heilongjiang":"黑龙江省","Henan":"河南省","Hubei":"湖北省","Hunan":"湖南省","Inner Mongolia":"内蒙古自治区",
          "Jiangsu":"江苏省","Jiangxi":"江西省","Jilin":"吉林省","Liaoning":"辽宁省","Ningxia":"宁夏回族自治区",
          "Qinghai":"青海省","Shaanxi":"陕西省","Shandong":"山东省","Shanxi":"山西省","Sichuan":"四川省",
          "Zhejiang":"浙江省","Tianjin":"天津市","Xinjiang":"新疆维吾尔自治区","Yunnan":"云南省"}
    lookup = {zh[k]: v for k, v in cn.set_index("province").longest_extreme_peaking_run_days.to_dict().items()}
    geo = json.loads(CHINA_GEO.read_text(encoding="utf-8"))
    norm = mpl.colors.Normalize(3, 6)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("cn", ["#FCE7DB", "#F1A16F", ORANGE])
    for feat in geo["features"]:
        v = lookup.get(feat["properties"].get("name"))
        add_geojson(axb, feat["geometry"], cmap(norm(v)) if v is not None else "#EFF3F4", lw=0.20)
    axb.set_xlim(73, 136); axb.set_ylim(17, 54); axb.set_aspect("equal"); axb.axis("off")
    axb.text(0.04, 0.08, "provincial gas peaking", transform=axb.transAxes, color=ORANGE, fontweight="bold")
    panel(axb, "b", -0.02, 0.98)

    bounds = (137.55, 140.15, 34.45, 36.05)
    for code in (11, 12, 13, 14, 22):
        path = JAPAN_GEO_ROOT / f"N03-20240101_{code}_GML/N03-20240101_{code}.geojson"
        for feat in json.loads(path.read_text(encoding="utf-8"))["features"]:
            add_geojson(axc, feat["geometry"], "#F1F5F5", edge="#C8D3D6", lw=0.13, bounds=bounds)
    comm = jp.groupby(["mesh_id", "latitude", "longitude"], as_index=False).agg(
        event=("longest_extreme_import_run_days", "median"), task=("annual_residual_import_mwh", "mean"))
    axc.scatter(comm.longitude, comm.latitude, c=comm.event, cmap="viridis", s=10 + 1.1*np.sqrt(comm.task),
                edgecolor="white", lw=0.35, zorder=5)
    axc.set_xlim(bounds[0], bounds[1]); axc.set_ylim(bounds[2], bounds[3]); axc.set_aspect("equal"); axc.axis("off")
    axc.text(0.04, 0.08, "community PV balancing", transform=axc.transAxes, color=TEAL, fontweight="bold")
    panel(axc, "c", -0.02, 0.98)

    colours = {"Europe": BLUE, "China": ORANGE, "Japan": TEAL}
    for y, name in enumerate(["Japan", "China", "Europe"]):
        vals = runs.loc[runs.scale == name, "event_run_days"].to_numpy(float)
        axd.scatter(vals, y + deterministic_jitter(len(vals), 0.13), s=11,
                    color=mpl.colors.to_rgba(colours[name], 0.62), edgecolor="white", lw=0.25)
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        axd.plot([q1, q3], [y, y], color=colours[name], lw=3.3, solid_capstyle="round")
        axd.scatter(med, y, color=INK, s=20, edgecolor="white", lw=0.3, zorder=5)
    axd.set_yticks(range(3), ["Japan", "China", "Europe"]); axd.set_xlim(0, 25)
    axd.set_xlabel("Longest natural event (d)"); clean(axd, "x"); panel(axd, "d", -0.16)

    # Within-regime task rank removes incomparable energy units while preserving the observed ordering.
    frames = []
    e = eu_cy[["annual_task_twh", "natural_longest_event_run_days"]].copy()
    e.columns = ["task", "event"]; e["regime"] = "Europe"; frames.append(e)
    c = cn[["annual_observed_peaking_task_gwh", "longest_extreme_peaking_run_days"]].copy()
    c.columns = ["task", "event"]; c["regime"] = "China"; frames.append(c)
    j = jp[["annual_residual_import_mwh", "longest_extreme_import_run_days"]].copy()
    j.columns = ["task", "event"]; j["regime"] = "Japan"; frames.append(j)
    rank = pd.concat(frames, ignore_index=True)
    rank["task_percentile"] = rank.groupby("regime").task.rank(pct=True) * 100
    rho = correlations.set_index("regime").spearman_rho_annual_task_vs_longest_event.to_dict()
    for name in ["Europe", "China", "Japan"]:
        q = rank[rank.regime == name]
        axe.scatter(q.task_percentile, q.event, s=9, color=mpl.colors.to_rgba(colours[name], 0.55),
                    edgecolor="white", lw=0.2, label=f"{name}  ρ={rho[name]:+.2f}")
    axe.set_xlabel("Within-regime annual-task percentile"); axe.set_ylabel("Longest event (d)")
    axe.legend(loc="upper right", handletextpad=0.2); clean(axe); panel(axe, "e", -0.14)

    chronology = pd.read_csv(SRC / "figure2_panel_c_conserved_chronology.csv")
    sl = chronology.iloc[:70]
    for base_y, task_col, recharge_col, label, colour in [
        (1.25, "dispersed_task_normalized", "dispersed_recharge_normalized", "dispersed", BLUE),
        (0.25, "clustered_task_normalized", "clustered_recharge_normalized", "clustered", ORANGE),
    ]:
        axf.fill_between(sl.day, base_y, base_y + 0.52*sl[task_col], step="mid", color=colour, alpha=0.9)
        axf.plot(sl.day, base_y - 0.23*sl[recharge_col], color=TEAL, lw=0.75)
        axf.text(-2, base_y + 0.18, label, ha="right", color=colour, fontweight="bold")
    axf.set_xlim(-5, 70); axf.set_ylim(-0.2, 2.1); axf.set_yticks([]); axf.set_xlabel("Identical day blocks rearranged within season")
    axf.spines[["left", "top", "right"]].set_visible(False); panel(axf, "f", -0.06)

    for y0, name, colour, marker in [(1, "dispersed", BLUE, "o"), (0, "clustered", ORANGE, "s")]:
        q = recovery[recovery.chronology == name].sort_values("recovery_hours").copy()
        shares = q.window_count / q.window_count.sum()
        axg.hlines(y0, 0, 520, color=mpl.colors.to_rgba(colour, 0.22), lw=2.0, zorder=1)
        axg.vlines(q.recovery_hours, y0 - 0.10, y0 + 0.10, color=colour, lw=0.75, alpha=0.55)
        axg.scatter(q.recovery_hours, np.full(len(q), y0), s=24 + 250 * shares,
                    marker=marker, color=mpl.colors.to_rgba(colour, 0.82), edgecolor="white",
                    lw=0.45, zorder=4)
    axg.annotate("", xy=(0, 0.50), xytext=(72, 0.50),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=9))
    axg.text(34, 0.58, "median 72 h → 0 h", color=INK, ha="center", fontsize=5.4, fontweight="bold")
    axg.text(0, -0.24, "74.7% zero-window mass", color=ORANGE, ha="left", fontsize=5.2)
    axg.set_xscale("symlog", linthresh=24, linscale=1.0)
    axg.set_xlim(-2, 560); axg.set_ylim(-0.42, 1.35)
    axg.set_xticks([0, 72, 168, 504], ["0", "72", "168", "504"])
    axg.set_yticks([0, 1], ["clustered", "dispersed"])
    axg.set_xlabel("Recovery window (h); area ∝ count")
    axg.spines[["left", "top", "right"]].set_visible(False); axg.tick_params(axis="y", length=0)
    axg.grid(axis="x", color=GRID, lw=0.45); panel(axg, "g", -0.17)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.985, bottom=0.055)
    save_bundle(fig, "fig_v51_figure2_three_regime_event_atlas")


def figure3() -> None:
    arms = pd.read_csv(SRC / "figure3_panel_a_portfolio_arm_values.csv")
    alloc = pd.read_csv(SRC / "figure3_panel_b_component_allocation.csv")
    adv = pd.read_csv(SRC / "figure3_panel_c_mix_advantage.csv")
    paired = pd.read_csv(SRC / "figure3_panel_d_paired_mix_advantage.csv")
    cycles = pd.read_csv(SRC / "figure3_panel_e_branch_cycles.csv")
    decomp = pd.read_csv(SRC / "figure5_panel_c_signed_channel_decomposition.csv")
    d24 = decomp[np.isclose(decomp.duration_h, 24)].iloc[0]

    fig = plt.figure(figsize=(7.20, 6.85))
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.0, 1.18, 0.98], hspace=0.52, wspace=1.18)
    axa = fig.add_subplot(gs[0, :4])
    axb = fig.add_subplot(gs[0, 5:])
    axc = fig.add_subplot(gs[1, :4])
    axd = fig.add_subplot(gs[1, 4:8])
    axe = fig.add_subplot(gs[1, 8:])
    axf = fig.add_subplot(gs[2, 1:11])

    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.axis("off")
    tail_d = float(alloc.loc[alloc._chronology == "dispersed", "tail_task_share"].iloc[0])
    tail_c = float(alloc.loc[alloc._chronology == "clustered", "tail_task_share"].iloc[0])
    axa.text(0.04, 0.92, "installed layers", color=INK, fontweight="bold", fontsize=5.7)
    axa.text(0.76, 0.92, "annual task allocation", color=INK, fontweight="bold", fontsize=5.7, ha="center")
    for y0, title, common, tail in [(0.73, "power", 0.50, 0.50), (0.53, "energy", 1/7, 6/7)]:
        axa.text(0.04, y0 + 0.055, title, color=MUTED, va="center", fontsize=5.2)
        x0, width, height = 0.18, 0.27, 0.11
        axa.add_patch(Rectangle((x0, y0), width * common, height, facecolor=BLUE, edgecolor="white", lw=0.5))
        axa.add_patch(Rectangle((x0 + width * common, y0), width * tail, height, facecolor=TEAL, edgecolor="white", lw=0.5))
        axa.text(x0 + width * common / 2, y0 + height / 2, f"{100*common:.0f}%", color="white",
                 ha="center", va="center", fontweight="bold", fontsize=5.0)
        axa.text(x0 + width * (common + tail / 2), y0 + height / 2, f"{100*tail:.1f}%", color="white",
                 ha="center", va="center", fontweight="bold", fontsize=5.0)
    for y0, name, tail, colour in [(0.68, "dispersed", tail_d, BLUE), (0.25, "clustered", tail_c, ORANGE)]:
        x0, width, height = 0.58, 0.37, 0.13
        common = 1 - tail
        axa.add_patch(Rectangle((x0, y0), width * common, height, facecolor=BLUE,
                                edgecolor="white", lw=0.5, zorder=3))
        axa.add_patch(Rectangle((x0 + width * common, y0), width * tail, height, facecolor=TEAL,
                                edgecolor="white", lw=0.5, zorder=3))
        if name == "dispersed":
            label_y, va = y0 + height + 0.025, "bottom"
        else:
            label_y, va = y0 - 0.025, "top"
        axa.text(x0 + width / 2, label_y, name, color=colour, ha="center", va=va,
                 fontsize=5.3, fontweight="bold")
        axa.text(x0 + width * common / 2, y0 + height / 2, f"{100*common:.1f}", color="white",
                 ha="center", va="center", fontweight="bold", fontsize=5.0)
        axa.text(x0 + width * (common + tail / 2), y0 + height / 2, f"{100*tail:.1f}", color="white",
                 ha="center", va="center", fontweight="bold", fontsize=5.0)
    flow_ribbon(axa, 0.45, 0.58, 0.53, 0.64, 0.68, 0.81, TEAL, 0.20)
    flow_ribbon(axa, 0.45, 0.58, 0.53, 0.64, 0.25, 0.38, TEAL, 0.12)
    axa.annotate("", xy=(0.93, 0.44), xytext=(0.93, 0.66),
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0, mutation_scale=9))
    axa.text(0.91, 0.46, f"-{100*(tail_d-tail_c):.1f} pp\ntail task", ha="right", va="center",
             color=ORANGE, fontsize=5.4, fontweight="bold")
    axa.text(0.21, 0.43, "4 h LFP", color=BLUE, fontsize=5.0, fontweight="bold", ha="center")
    axa.text(0.37, 0.43, "24 h A-CAES", color=TEAL, fontsize=5.0, fontweight="bold", ha="center")
    panel(axa, "a", -0.02, 0.98)

    p = paired.sort_values("mix_benefit_eur_per_kw_common_power_year_dispersed").reset_index(drop=True)
    rank = np.arange(1, len(p) + 1)
    yd = p.mix_benefit_eur_per_kw_common_power_year_dispersed.to_numpy()
    yc = p.mix_benefit_eur_per_kw_common_power_year_clustered.to_numpy()
    axb.fill_between(rank, yd, yc, color=mpl.colors.to_rgba(ORANGE, 0.18))
    axb.plot(rank, yd, color=BLUE, lw=1.45, label="dispersed")
    axb.plot(rank, yc, color=ORANGE, lw=1.45, label="clustered")
    axb.scatter(rank[::6], yd[::6], s=8, color=BLUE, edgecolor="white", lw=0.2, zorder=4)
    axb.scatter(rank[::6], yc[::6], s=8, color=ORANGE, edgecolor="white", lw=0.2, zorder=4)
    axb.axhline(0, color=INK, lw=0.75)
    axb.annotate("paired value loss under clustering", xy=(95, yc[94]), xytext=(62, -2.5),
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8),
                 color=ORANGE, fontsize=5.4, ha="center")
    axb.set_xlim(1, len(p)); axb.set_xlabel("Country-years sorted by dispersed A-CAES advantage")
    axb.set_ylabel("A-CAES advantage\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axb.legend(loc="upper left", ncol=2, handlelength=1.4); clean(axb); panel(axb, "b", -0.06)

    arm_order = [("LFP tail", "dispersed", BLUE), ("LFP tail", "clustered", ORANGE),
                 ("A-CAES tail", "dispersed", BLUE), ("A-CAES tail", "clustered", ORANGE)]
    labels = ["4 h LFP · D", "4 h LFP · C", "A-CAES · D", "A-CAES · C"]
    for i, (branch, chrono, colour) in enumerate(arm_order):
        vals = arms[(arms.tail_branch == branch) & (arms.chronology == chrono)].portfolio_value_eur_per_kw_common_power_year
        summary_segment(axc, vals, i, colour)
    axc.axvline(0, color=INK, lw=0.75)
    axc.set_yticks(range(4), labels); axc.invert_yaxis()
    axc.set_xlabel("Portfolio value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    clean(axc, "x"); panel(axc, "c", -0.18)

    for i, branch in enumerate(["LFP tail", "A-CAES tail"]):
        d = cycles[(cycles.branch == branch) & (cycles.chronology == "dispersed")].equivalent_full_cycles.to_numpy()
        c = cycles[(cycles.branch == branch) & (cycles.chronology == "clustered")].equivalent_full_cycles.to_numpy()
        for vals, y0, colour in [(d, i * 2, BLUE), (c, i * 2 + 0.55, ORANGE)]:
            q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
            axd.plot([q1, q3], [y0, y0], color=colour, lw=4.0, solid_capstyle="round")
            axd.scatter(med, y0, s=28, color=INK, edgecolor="white", lw=0.35, zorder=5)
            axd.scatter(vals, y0 + deterministic_jitter(len(vals), 0.11), s=5,
                        color=mpl.colors.to_rgba(colour, 0.24), edgecolor="none")
        axd.add_patch(FancyArrowPatch((np.median(d), i * 2 + 0.19), (np.median(c), i * 2 + 0.38),
                                      arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                      color=ORANGE, alpha=0.9))
    axd.set_yticks([0.25, 2.25], ["4 h LFP", "24 h A-CAES"])
    axd.set_xlabel("Equivalent full cycles yr$^{-1}$")
    axd.legend(handles=[Patch(color=BLUE, label="dispersed"), Patch(color=ORANGE, label="clustered")],
               loc="upper right", bbox_to_anchor=(1.01, 1.02), ncol=2, handlelength=0.8, columnspacing=0.8)
    clean(axd, "x"); panel(axd, "d", -0.16)

    retention_matrix = np.column_stack([
        100 * paired.tail_charge_mwh_clustered / paired.tail_charge_mwh_dispersed,
        100 * paired.tail_discharge_mwh_clustered / paired.tail_discharge_mwh_dispersed,
        100 * paired.tail_equivalent_full_cycles_clustered / paired.tail_equivalent_full_cycles_dispersed,
    ])
    for row in retention_matrix:
        axe.plot(range(3), row, color=mpl.colors.to_rgba(INK, 0.055), lw=0.45, zorder=1)
    q25, med, q75 = np.quantile(retention_matrix, [0.25, 0.5, 0.75], axis=0)
    axe.fill_between(range(3), q25, q75, color=mpl.colors.to_rgba(PURPLE, 0.20), zorder=2)
    axe.plot(range(3), med, color=PURPLE, lw=2.2, marker="o", ms=4.6, mec="white", mew=0.45, zorder=4)
    axe.fill_between(range(3), med, 100, color=mpl.colors.to_rgba(ORANGE, 0.08), zorder=0)
    axe.axhline(100, color=INK, ls="--", lw=0.75)
    axe.text(1.95, med[-1] + 2.2, f"median retained {med[-1]:.0f}%", color=PURPLE,
             ha="right", fontweight="bold", fontsize=5.3)
    axe.text(1.95, 95, f"{100-med[-1]:.0f}% lost", color=ORANGE, ha="right", fontsize=5.2)
    axe.set_xticks(range(3), ["tail\ncharge", "task\nservice", "equivalent\ncycles"])
    axe.set_ylim(38, 104); axe.set_ylabel("Clustered / dispersed use (%)")
    clean(axe); panel(axe, "e", -0.20)

    values = np.array([d24.cycling_channel_country_block_mean, d24.coverage_channel_country_block_mean])
    cumulative = np.r_[0, np.cumsum(values)]
    xx = np.array([0, 1, 2])
    axf.fill_between(xx, cumulative, 0, color=mpl.colors.to_rgba(BLUE, 0.11), zorder=0)
    axf.plot(xx, cumulative, color=INK, lw=1.25, zorder=2)
    for i, (start, end, colour, label) in enumerate(zip(cumulative[:-1], cumulative[1:],
                                                        [BLUE, PURPLE], ["cycling", "coverage"])):
        axf.annotate("", xy=(i + 1, end), xytext=(i, start),
                     arrowprops=dict(arrowstyle="-|>", color=colour, lw=4.2,
                                     mutation_scale=11, shrinkA=5, shrinkB=5))
        axf.text(i + 0.50, (start + end) / 2 + 0.48, f"{label}  {end-start:+.2f}",
                 color=colour, ha="center", va="bottom", fontsize=5.4, fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5))
    axf.scatter(xx, cumulative, s=[28, 36, 54], color=[INK, BLUE, INK], edgecolor="white", lw=0.5, zorder=4)
    axf.text(2, cumulative[-1] - 0.26, f"total  {cumulative[-1]:+.2f}", color=INK,
             ha="center", va="top", fontsize=6.0, fontweight="bold")
    axf.axhline(0, color=INK, lw=0.75)
    axf.set_xticks(xx, ["dispersed\nledger", "after cycling", "clustered\nledger"])
    axf.set_xlim(-0.18, 2.18); axf.set_ylim(cumulative[-1] - 0.9, 0.8)
    axf.set_ylabel("24 h chronology effect (EUR kW$^{-1}$ yr$^{-1}$)")
    clean(axf); panel(axf, "f", -0.06)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.982, bottom=0.06)
    save_bundle(fig, "fig_v51_figure3_portfolio_cycling_ledger")


def figure4() -> None:
    retention = pd.read_csv(SRC / "figure4_panel_a_tail_retention.csv")
    binding = pd.read_csv(SRC / "figure4_panel_b_binding_hours.csv")
    multiplier = pd.read_csv(SRC / "figure4_panel_c_recharge_multiplier_response.csv")
    link = pd.read_csv(SRC / "figure4_panel_d_physical_economic_link.csv")
    price = pd.read_csv(ROOT / "price_channel/empirical_summary_v0_1/price_channel_caes24_summary.csv")
    reopt = pd.read_csv(ROOT / "price_channel/empirical_reoptimized_summary_v0_1/s3b_caes24_summary.csv")
    coeff = pd.read_csv(ROOT / "joint_choice/natural_fixed_effect_coefficients.csv")

    fig = plt.figure(figsize=(7.20, 6.85))
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.05, 1.0, 1.0], hspace=0.50, wspace=1.08)
    axa = fig.add_subplot(gs[0, :7]); axb = fig.add_subplot(gs[0, 8:])
    axc = fig.add_subplot(gs[1, :6])
    joint = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 6:], width_ratios=[4.2, 1.0],
                                   height_ratios=[1.0, 4.2], hspace=0.04, wspace=0.04)
    axd_top = fig.add_subplot(joint[0, 0])
    axd = fig.add_subplot(joint[1, 0])
    axd_right = fig.add_subplot(joint[1, 1])
    axe = fig.add_subplot(gs[2, :7]); axf = fig.add_subplot(gs[2, 8:])

    for y, metric in enumerate(["Charging", "Task service", "Equivalent cycles"]):
        vals = retention[retention.metric == metric].clustered_to_dispersed_pct.to_numpy()
        parts = axa.violinplot(vals, positions=[y], vert=False, widths=0.72, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor([TEAL, BLUE, PURPLE][y]); b.set_alpha(0.25); b.set_edgecolor("none")
        axa.scatter(vals, y + deterministic_jitter(len(vals), 0.15), s=7,
                    color=mpl.colors.to_rgba([TEAL, BLUE, PURPLE][y], 0.35), edgecolor="none")
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        axa.plot([q1,q3],[y,y], color=[TEAL,BLUE,PURPLE][y], lw=3.5, solid_capstyle="round")
        axa.scatter(med,y,s=24,color=INK,edgecolor="white",lw=0.3,zorder=5)
    axa.axvline(100, color=INK, ls="--", lw=0.7); axa.set_yticks(range(3), ["charging", "task service", "equivalent cycles"])
    axa.set_xlabel("Clustered / dispersed retention (%)"); axa.set_xlim(35,105); clean(axa,"x"); panel(axa,"a",-0.11)

    y = np.arange(2)
    for i, metric in enumerate(["shared_recharge_binding_hours", "tail_charge_power_binding_hours"]):
        d = float(binding.loc[binding._chronology == "dispersed", metric].iloc[0])
        c = float(binding.loc[binding._chronology == "clustered", metric].iloc[0])
        axb.annotate("", xy=(c, i), xytext=(d, i),
                     arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.7, mutation_scale=9))
        axb.scatter(d,i,s=42,color=BLUE,marker="o",edgecolor="white",lw=0.45,zorder=4)
        axb.scatter(c,i,s=42,color=ORANGE,marker="s",edgecolor="white",lw=0.45,zorder=4)
        change = 100 * (c / d - 1)
        label_x = max(d, c) * (0.90 if max(d, c) > 50 else 1.18)
        ha = "right" if max(d, c) > 50 else "left"
        axb.text(label_x, i + 0.13, f"{change:+.0f}%", va="bottom", ha=ha,
                 fontsize=5.4, color=ORANGE, fontweight="bold")
        axb.text(d, i + 0.12, f"{d:.1f}", color=BLUE, ha="left", va="bottom", fontsize=5.0)
        axb.text(c, i - 0.12, f"{c:.1f}", color=ORANGE, ha="right", va="top", fontsize=5.0)
    axb.set_xscale("log"); axb.set_yticks(y,["surplus ceiling","charge interface"]); axb.set_xlabel("Binding hours yr$^{-1}$")
    axb.set_xlim(0.7, 900)
    axb.text(0.98, 1.02, "● D    ■ C", transform=axb.transAxes, ha="right", va="bottom",
             color=INK, fontsize=5.0)
    axb.set_ylim(-0.30, 1.35)
    clean(axb,"x"); panel(axb,"b",-0.25)

    multiplier = multiplier[multiplier.metric == "mix_benefit_eur_per_kw_common_power_year"].copy()
    multiplier["tech_label"] = np.where(multiplier.tech_id == "caes", "A-CAES 24 h", "Cavern H$_2$ 48 h")
    for tech, colour, marker, label in [("A-CAES 24 h", BLUE, "o", "A-CAES 24 h"),
                                        ("Cavern H$_2$ 48 h", PURPLE, "D", "Cavern H$_2$ 48 h")]:
        q = multiplier[multiplier.tech_label == tech].sort_values("recharge_quantity_multiplier")
        xq = q.recharge_quantity_multiplier.to_numpy(float)
        mean = q.country_block_mean.to_numpy(float)
        lo = q.country_block_ci_low.to_numpy(float)
        hi = q.country_block_ci_high.to_numpy(float)
        axc.fill_between(xq, lo, hi, color=mpl.colors.to_rgba(colour, 0.12), lw=0)
        axc.plot(xq, mean, color=colour, marker=marker, ms=4.2, lw=1.5,
                 mec="white", mew=0.45)
        axc.text(xq[-1] + 0.10, mean[-1], label, color=colour, va="center", fontsize=5.2)
    axc.axhline(0, color=INK, lw=0.75)
    axc.axhspan(-0.25, 0.25, color=mpl.colors.to_rgba(TEAL, 0.08), zorder=0)
    axc.text(2.5, -0.45, "additional annual recharge does not restore value", color=MUTED,
             ha="center", va="top", fontsize=5.1)
    axc.set_xticks([1,2,4],["1×","2×","4×"]); axc.set_xlim(0.8,4.85)
    axc.set_xlabel("Recharge quantity at already available hours")
    axc.set_ylabel("Chronology effect (EUR kW$^{-1}$ yr$^{-1}$)")
    clean(axc); panel(axc,"c",-0.16)

    x=link.clustered_minus_dispersed_tail_equivalent_full_cycles.to_numpy()
    yv=link.clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year.to_numpy()
    loss=-link.tail_charge_change_gwh.to_numpy()
    axd.hexbin(x, yv, gridsize=20, cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "density", ["#F5F7F7", "#B8D6E5", BLUE]), mincnt=1, linewidths=0, alpha=0.82)
    sizes = 7 + 36 * np.sqrt(np.clip(loss, 0, None) / max(loss.max(), 1e-12))
    axd.scatter(x, yv, s=sizes, facecolor=mpl.colors.to_rgba(ORANGE, 0.18),
                edgecolor=mpl.colors.to_rgba(INK, 0.45), lw=0.28, zorder=3)
    fit=np.polyfit(x,yv,1); xx=np.linspace(x.min(),x.max(),100); axd.plot(xx,np.polyval(fit,xx),color=INK,lw=1.25)
    r=np.corrcoef(x,yv)[0,1]; axd.text(0.04,0.91,f"r = {r:.2f}",transform=axd.transAxes,fontweight="bold")
    axd.text(0.04,0.79,"area ∝ tail-charge loss",transform=axd.transAxes,color=MUTED,fontsize=5.0)
    axd.set_xlabel("Change in equivalent cycles"); axd.set_ylabel("Value change\n(EUR kW$^{-1}$ yr$^{-1}$)", labelpad=0.5)
    clean(axd); panel(axd,"d",-0.18,1.30)
    axd_top.hist(x, bins=18, color=mpl.colors.to_rgba(BLUE, 0.58), edgecolor="white", lw=0.25)
    axd_right.hist(yv, bins=16, orientation="horizontal", color=mpl.colors.to_rgba(ORANGE, 0.55),
                    edgecolor="white", lw=0.25)
    for marginal in [axd_top, axd_right]:
        marginal.axis("off")
    axd_top.set_xlim(axd.get_xlim()); axd_right.set_ylim(axd.get_ylim())

    p = price[price.scenario.isin(["S0","S1_beta_1","S2_surplus_linked_charge","S3a_empirical_price"])].copy()
    p = pd.concat([p, reopt.rename(columns={"scenario":"scenario"})], ignore_index=True)
    order=["S0","S1_beta_1","S2_surplus_linked_charge","S3a_empirical_price","S3b_empirical_reoptimized"]
    names={"S0":"common price","S1_beta_1":"event scarcity premium","S2_surplus_linked_charge":"surplus-linked charge","S3a_empirical_price":"day-ahead revaluation","S3b_empirical_reoptimized":"price-responsive dispatch"}
    p["ord"]=p.scenario.map({v:i for i,v in enumerate(order)}); p=p.sort_values("ord")
    ys = np.arange(len(p))[::-1]
    cols = [BLUE, BLUE, TEAL, PURPLE, ORANGE]
    means = p.mean_effect_eur_kw_year.to_numpy()
    lows = p.ci95_low.to_numpy(); highs = p.ci95_high.to_numpy()
    axe.plot(means, ys, color=mpl.colors.to_rgba(INK, 0.35), lw=0.8, zorder=1)
    for i, (row, y0) in enumerate(zip(p.itertuples(), ys)):
        axe.plot([row.ci95_low, row.ci95_high], [y0, y0], color=cols[i], lw=2.3,
                 solid_capstyle="round", zorder=2)
        axe.scatter(row.mean_effect_eur_kw_year, y0, s=38, color=cols[i], marker=["o","s","D","^","P"][i],
                    edgecolor="white", lw=0.45, zorder=4)
        axe.text(row.ci95_high + 0.18, y0, f"{row.mean_effect_eur_kw_year:+.1f}",
                 color=cols[i], va="center", fontsize=5.1, fontweight="bold")
    axe.axvline(0, color=INK, lw=0.75)
    axe.set_yticks(ys, [names[v] for v in p.scenario])
    axe.set_xlabel("24 h chronology effect (EUR kW$^{-1}$ yr$^{-1}$)")
    clean(axe,"x"); panel(axe,"e",-0.08)

    fe=[]
    for model,label in [("zero_window_only","unadjusted"),("joint_geometry_plus_scale","adjusted")]:
        row=coeff[(coeff.model_id==model)&(coeff.predictor=="natural_zero_recovery_window_share")].iloc[0]
        fe.append((label,0.1*row.coefficient,0.1*row.bootstrap_ci95_low,0.1*row.bootstrap_ci95_high))
    for i,(label,est,lo,hi) in enumerate(fe[::-1]):
        colour = ORANGE if label == "adjusted" else BLUE
        axf.plot([lo, hi], [i, i], color=colour, lw=3.0, solid_capstyle="round")
        axf.scatter(est, i, s=48, color=colour, marker="D" if label == "adjusted" else "o",
                    edgecolor="white", lw=0.5, zorder=4)
        axf.text(hi + 0.05, i, f"{est:.2f}", color=colour, va="center", fontsize=5.3, fontweight="bold")
    axf.axvline(0,color=INK,lw=0.75)
    axf.set_yticks(range(2),[x[0] for x in fe[::-1]])
    axf.set_xlabel("Effect per +10 pp zero-window share\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axf.set_xlim(min(lo for _,_,lo,_ in fe)-0.25,0.35)
    clean(axf,"x"); panel(axf,"f",-0.23)

    fig.subplots_adjust(left=0.10,right=0.985,top=0.98,bottom=0.06)
    save_bundle(fig,"fig_v51_figure4_recovery_interface_tests")


def figure5() -> None:
    eu=pd.read_csv(SRC49/"figure5a_europe_duration_response.csv").sort_values("duration_h")
    cross=pd.read_csv(SRC49/"figure5bc_china_japan_duration_response.csv")
    decomp=pd.read_csv(SRC/"figure5_panel_c_signed_channel_decomposition.csv")
    signature=pd.read_csv(SRC/"figure5_panel_d_cluster_signature.csv")
    crossing=pd.read_csv(SRC/"figure5_panel_e_zero_crossing_summary.csv")
    crossing_records=pd.read_csv(SRC/"figure5_panel_e_zero_crossing_records.csv")
    counts=pd.read_csv(SRC49/"figure5d_three_scale_strict_valley_counts.csv")
    eu_records=pd.read_csv(SRC/"figure5_panel_e_conditional_valley_records.csv")

    fig=plt.figure(figsize=(7.20,7.65))
    gs=GridSpec(3,12,figure=fig,height_ratios=[1.55,1.08,1.0],hspace=0.50,wspace=0.90)
    hero=GridSpecFromSubplotSpec(3,1,subplot_spec=gs[0,:8],hspace=0.08)
    axes=[fig.add_subplot(hero[i,0]) for i in range(3)]
    axb=fig.add_subplot(gs[0,8:]); axc=fig.add_subplot(gs[1,:7]); axd=fig.add_subplot(gs[1,8:])
    axe=fig.add_subplot(gs[2,:7]); axf=fig.add_subplot(gs[2,8:])

    series=[("Europe",eu,"point_estimate","ci_low","ci_high",BLUE),
            ("China",cross[(cross.application_case=="china")&(cross.scenario=="S0")&(cross.cost_year==2035)],"mean_effect_eur_kw_year","ci95_low","ci95_high",ORANGE),
            ("Japan",cross[(cross.application_case=="japan")&(cross.scenario=="S0")&(cross.cost_year==2035)],"mean_effect_eur_kw_year","ci95_low","ci95_high",TEAL)]
    for i,(name,df,ycol,lcol,hcol,colour) in enumerate(series):
        df=df.sort_values("duration_h")
        extent=[df.duration_h.min(),df.duration_h.max(),df[ycol].min(),df[ycol].max()]
        neg = df[ycol].to_numpy() < 0
        axes[i].fill_between(df.duration_h,df[lcol],df[hcol],color=colour,alpha=0.16,lw=0)
        axes[i].fill_between(df.duration_h, 0, df[ycol], where=neg, color=mpl.colors.to_rgba(colour, 0.13), lw=0)
        axes[i].fill_between(df.duration_h, 0, df[ycol], where=~neg, color=mpl.colors.to_rgba(GOLD, 0.16), lw=0)
        axes[i].plot(df.duration_h,df[ycol],color=colour,lw=1.45,marker="o",ms=2.2,mec="white",mew=0.25)
        axes[i].axhline(0,color=INK,lw=0.65); axes[i].set_xscale("log"); axes[i].set_xlim(4,250)
        axes[i].text(0.01,0.82,name,transform=axes[i].transAxes,color=colour,fontweight="bold")
        axes[i].spines[["top","right"]].set_visible(False); axes[i].grid(axis="x",color=GRID,lw=0.4)
        if i<2: axes[i].set_xticklabels([])
    axes[-1].set_xticks([4,8,16,24,40,72,96,168,240],["4","8","16","24","40","72","96","168","240"])
    axes[-1].set_xlabel("Storage duration (h)"); axes[1].set_ylabel("Clustered − dispersed value (EUR kW$^{-1}$ yr$^{-1}$)")
    panel(axes[0],"a",-0.08,1.04)

    sel=decomp.sort_values("duration_h").copy()
    dur=sel.duration_h.to_numpy()
    cyc=sel.cycling_channel_country_block_mean.to_numpy()
    cov=sel.coverage_channel_country_block_mean.to_numpy()
    total=sel.total_effect_country_block_mean.to_numpy()
    axb.fill_between(dur,0,cyc,color=mpl.colors.to_rgba(BLUE,0.30),label="cycling")
    axb.fill_between(dur,cyc,cyc+cov,color=mpl.colors.to_rgba(PURPLE,0.36),label="coverage")
    axb.plot(dur,total,color=INK,lw=1.35,marker="D",ms=2.6,mec="white",mew=0.25,label="total")
    axb.axhline(0,color=INK,lw=0.7); axb.set_xscale("log")
    axb.set_xticks([24,32,40,60,96,168],["24","32","40","60","96","168"])
    axb.set_ylabel("Effect\n(EUR kW$^{-1}$ yr$^{-1}$)", labelpad=0.5)
    axb.legend(loc="upper left", handlelength=1.0); clean(axb,"x"); panel(axb,"b",-0.18)

    for days,colour in [(2,BLUE),(4,ORANGE),(6,PURPLE)]:
        q=signature[signature.target_cluster_days==days].sort_values("duration_h")
        axc.fill_between(q.duration_h,q.ci95_low,q.ci95_high,color=colour,alpha=0.10,lw=0)
        axc.plot(q.duration_h,q.country_block_mean,color=colour,lw=1.35,label=f"{days} d cluster")
    axc.axhline(0,color=INK,lw=0.7); axc.set_xscale("log"); axc.set_xticks([24,40,60,72,96,120,168],["24","40","60","72","96","120","168"])
    axc.set_xlabel("Storage duration (h)"); axc.set_ylabel("Chronology effect (EUR kW$^{-1}$ yr$^{-1}$)")
    axc.legend(ncol=3,loc="lower right"); clean(axc); panel(axc,"c",-0.09)

    for i, (days, colour) in enumerate([(2,BLUE),(4,ORANGE),(6,PURPLE)]):
        q = crossing_records[crossing_records.target_cluster_days == days]
        observed = q.loc[q.crossing_status != "right_censored", "zero_crossing_duration_h"].dropna().to_numpy()
        vp = axd.violinplot(observed, positions=[i], vert=False, widths=0.72, showextrema=False,
                           points=80, bw_method=0.28)
        for body in vp["bodies"]:
            body.set_facecolor(colour); body.set_edgecolor("none"); body.set_alpha(0.20)
        axd.scatter(observed, i + deterministic_jitter(len(observed), 0.18), s=5.2,
                    color=mpl.colors.to_rgba(colour, 0.30), edgecolor="none", zorder=2)
        row = crossing[crossing.target_cluster_days == days].iloc[0]
        axd.plot([row.q25_crossing_h,row.q75_crossing_h],[i,i],color=colour,lw=4.0,solid_capstyle="round",zorder=3)
        axd.scatter(row.median_crossing_h,i,s=34,color=INK,edgecolor="white",lw=0.4,zorder=5)
        axd.text(row.median_crossing_h,i+0.25,f"{int(row.median_crossing_h)} h",ha="center",fontsize=5.2,
                 color=INK,fontweight="bold")
        censored = int((q.crossing_status == "right_censored").sum())
        if censored:
            axd.scatter(np.repeat(168,censored), i + deterministic_jitter(censored,0.10), marker=">",s=22,
                        color=colour,edgecolor="white",lw=0.35,zorder=5)
            axd.text(166,i-0.28,f"{censored} censored",ha="right",fontsize=5.0,color=colour)
    axd.set_yticks(range(3),["2 d","4 d","6 d"]); axd.set_xlim(20,174)
    axd.set_xlabel("First sustained positive duration (h)")
    clean(axd,"x"); panel(axd,"d",-0.24)

    eu_strict=eu_records[(eu_records.chronology=="clustered")&eu_records.strict_valley].copy()
    all_durations=sorted({int(v) for text in eu_strict.nonpositive_durations_h.dropna()
                          for v in str(text).split("|") if str(v).strip()})
    occupancy=[]
    for duration in all_durations:
        count=sum(str(duration) in str(text).split("|") for text in eu_strict.nonpositive_durations_h.dropna())
        occupancy.append((duration,count,100*count/132))
    occ=pd.DataFrame(occupancy,columns=["duration_h","country_years","share_pct"])
    bars=axe.bar(occ.duration_h.astype(str),occ.share_pct,color=mpl.colors.to_rgba(BLUE,0.70),edgecolor="white",lw=0.45)
    for bar,row in zip(bars,occ.itertuples()):
        if row.country_years >= 8:
            axe.text(bar.get_x()+bar.get_width()/2,row.share_pct+1.2,f"{row.country_years}",ha="center",fontsize=5.1,color=INK)
    axe.set_ylabel("Clustered valley occupancy (%)")
    axe.set_xlabel("Non-positive duration node (h)")
    axe.set_ylim(0,max(occ.share_pct.max()+8,10)); clean(axe); panel(axe,"e",0.0,1.04)

    selected=counts[((counts.scale=="Europe")&(counts.selection_label.str.contains("E=98")))|
                    ((counts.scale=="China")&(counts.selection_label=="China clustered-valley phase"))|
                    ((counts.scale=="Japan")&(counts.selection_label=="Japan clustered-valley phase"))]
    matrix=selected.pivot(index="scale",columns="chronology",values="strict_valleys").reindex(["Europe","China","Japan"])[["dispersed","clustered"]]
    denominators=np.array([132,26,27])
    rate=matrix.to_numpy()/denominators[:,None]*100
    scale_cols=[BLUE,ORANGE,TEAL]
    for i, name in enumerate(matrix.index):
        y0, y1 = rate[i,0], rate[i,1]
        axf.annotate("",xy=(1,y1),xytext=(0,y0),
                     arrowprops=dict(arrowstyle="-|>",color=scale_cols[i],lw=2.5,mutation_scale=10,alpha=0.82))
        axf.scatter(0,y0,s=34,marker="o",color=scale_cols[i],edgecolor="white",lw=0.5,zorder=4)
        axf.scatter(1,y1,s=55,marker="s",color=scale_cols[i],edgecolor="white",lw=0.5,zorder=4)
        axf.text(1.04,y1,f"{name}  {int(matrix.iloc[i,1])}/{denominators[i]}",ha="left",va="center",
                 color=scale_cols[i],fontsize=5.2,fontweight="bold")
    axf.text(-0.04,4.0,"all 0",ha="right",va="bottom",color=MUTED,fontsize=5.0,fontweight="bold")
    axf.set_xticks([0,1],["dispersed","clustered"]); axf.set_xlim(-0.18,1.62); axf.set_ylim(-5,108)
    axf.set_ylabel("Strict-valley incidence (%)")
    clean(axf); panel(axf,"f",-0.25)

    fig.subplots_adjust(left=0.09,right=0.985,top=0.985,bottom=0.06)
    save_bundle(fig,"fig_v51_figure5_cross_duration_reversal")


def figure6() -> None:
    parity=pd.read_csv(SRC/"figure6_panel_b_value_shift_cost_path_ecdf.csv")
    flips=pd.read_csv(SRC/"figure6_panel_a_robust_flip_summary.csv")
    curve=pd.read_csv(SRC/"figure6_panel_c_capacity_anchor_flip_curve.csv")
    anchors=pd.read_csv(SRC/"figure6_panel_c_capacity_market_anchors.csv")
    sensitivity=pd.read_csv(ROOT/"sequencing_parity/cost_duration_discount_sensitivity.csv")
    at=pd.read_csv(ROOT/"sequencing_parity/austria_2024_same_year_pair.csv").iloc[0]
    coeff=pd.read_csv(ROOT/"joint_choice/natural_fixed_effect_coefficients.csv")

    fig=plt.figure(figsize=(7.20,7.05))
    gs=GridSpec(3,12,figure=fig,height_ratios=[1.16,1.0,1.0],hspace=0.50,wspace=0.80)
    axa=fig.add_subplot(gs[0,:7]); axb=fig.add_subplot(gs[0,8:])
    axc=fig.add_subplot(gs[1,:5]); axd=fig.add_subplot(gs[1,5:])
    axe=fig.add_subplot(gs[2,:7]); axf=fig.add_subplot(gs[2,8:])

    vals=np.sort(parity.absolute_cluster_effect_eur_kw_year.to_numpy())
    y=np.arange(1,len(vals)+1)/len(vals)*100
    axa.fill_between(vals,0,y,color=mpl.colors.to_rgba(BLUE,0.16),step="pre")
    axa.plot(vals,y,color=BLUE,lw=1.7)
    rug_y = -2.5 + deterministic_jitter(len(vals), 1.0)
    axa.scatter(vals, rug_y, s=3.5, color=mpl.colors.to_rgba(INK, 0.22), edgecolor="none", clip_on=False)
    axa.axvline(6.03,color=ORANGE,lw=1.3,ls="--")
    axa.axvline(np.median(vals),color=INK,lw=1.0)
    axa.text(6.03,8,"2025–2035\ncost movement\n6.03",color=ORANGE,ha="center",va="bottom",fontsize=5.3)
    axa.text(np.median(vals),78,f"median\n{np.median(vals):.2f}",color=INK,ha="center")
    axa.set_xlabel("Absolute chronology-induced value shift (EUR kW$^{-1}$ yr$^{-1}$)"); axa.set_ylabel("Cumulative share (%)")
    clean(axa); panel(axa,"a",-0.10)

    axb.axhspan(0, 7.8, color=mpl.colors.to_rgba(TEAL, 0.10), lw=0)
    axb.axhspan(-19, 0, color=mpl.colors.to_rgba(ORANGE, 0.09), lw=0)
    y0=at.top_benefit_eur_kw_year_dispersed; y1=at.top_benefit_eur_kw_year_clustered
    axb.annotate("",xy=(1,y1),xytext=(0,y0),
                 arrowprops=dict(arrowstyle="-|>",color=ORANGE,lw=3.2,mutation_scale=13,
                                 connectionstyle="arc3,rad=-0.06"))
    axb.scatter(0,y0,s=82,color=TEAL,marker="o",edgecolor="white",lw=0.65,zorder=5)
    axb.scatter(1,y1,s=82,color=ORANGE,marker="s",edgecolor="white",lw=0.65,zorder=5)
    axb.axhline(0,color=INK,lw=0.85); axb.set_xticks([0,1],["dispersed","clustered"])
    axb.set_ylabel("A-CAES margin over 4 h LFP\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axb.text(0,y0+1.0,f"A-CAES selected\n{y0:+.2f}",ha="center",va="bottom",color=TEAL,fontweight="bold",fontsize=5.4)
    axb.text(1,y1-1.0,f"4 h LFP selected\n{y1:+.2f}",ha="center",va="top",color=ORANGE,fontweight="bold",fontsize=5.4)
    axb.text(0.52,-4.0,"procurement boundary crossed",ha="center",color=INK,fontsize=5.0)
    axb.set_ylim(-19,8.0)
    clean(axb); panel(axb,"b",-0.22)

    rates = 100 * flips.disjoint_tie_set_flip_rate
    lows = 100 * flips.country_block_bootstrap_ci95_low
    highs = 100 * flips.country_block_bootstrap_ci95_high
    axc.fill_between(flips.cost_year, lows, highs, color=mpl.colors.to_rgba(ORANGE, 0.14))
    axc.plot(flips.cost_year, rates, color=ORANGE, lw=1.5)
    sizes = 36 + 4.2 * flips.disjoint_tie_set_flips
    axc.scatter(flips.cost_year, rates, s=sizes, color=ORANGE, edgecolor="white", lw=0.55, zorder=4)
    for r in flips.itertuples():
        axc.text(r.cost_year, 100*r.disjoint_tie_set_flip_rate + 2.0, f"{r.disjoint_tie_set_flips}/132",
                 ha="center", fontsize=5.2, color=INK)
    axc.set_xticks([2025,2030,2035]); axc.set_ylim(-1,34); axc.set_ylabel("Tail-participation flips (%)")
    clean(axc); panel(axc,"c",-0.17)

    q=curve[(curve.cost_year==2035)&(curve.branch_id.str.startswith("A_capacity"))].sort_values("firm_value_eur_kw_year")
    axd.plot(q.firm_value_eur_kw_year,100*q.tail_entry_exit_flip_rate,color=PURPLE,lw=1.6)
    axd.fill_between(q.firm_value_eur_kw_year, 0, 100*q.tail_entry_exit_flip_rate,
                     color=mpl.colors.to_rgba(PURPLE, 0.10))
    band=anchors[anchors.within_priority_market_range].eur_per_kw_year
    axd.axvspan(band.min(),band.max(),color=GOLD,alpha=0.16)
    axd.axvline(103.8,color=INK,ls="--",lw=0.8)
    axd.text((band.min()+band.max())/2,2,"observed capacity\nmarket range",ha="center",va="bottom",fontsize=5.0,color=GOLD)
    axd.set_xlabel("Firm-supply unit value (EUR kW$^{-1}$ yr$^{-1}$)"); axd.set_ylabel("2035 flips (%)")
    clean(axd); panel(axd,"d",-0.15)

    pivot=sensitivity.pivot(index="discount_rate",columns="caes_duration_h",values="absolute_relative_cost_movement_eur_kw_year")
    im=axe.imshow(pivot.values,origin="lower",aspect="auto",cmap="cividis",
                  extent=[pivot.columns.min()-1,pivot.columns.max()+1,100*pivot.index.min()-0.5,100*pivot.index.max()+0.5])
    cs=axe.contour(pivot.columns,100*pivot.index,pivot.values,levels=[5.48, 6.03],colors=["white", ORANGE],linewidths=[1.0, 1.0])
    axe.clabel(cs,fmt={5.48:"sequencing median", 6.03:"24 h pair"},fontsize=5.0)
    axe.scatter([24],[3.6],marker="*",s=70,color=ORANGE,edgecolor="white",lw=0.5,zorder=5)
    axe.set_xlabel("A-CAES duration (h)"); axe.set_ylabel("Real discount rate (%)")
    cax = axe.inset_axes([0.79, 0.08, 0.025, 0.82])
    cb=fig.colorbar(im,cax=cax); cb.set_label("cost movement", fontsize=5.0)
    panel(axe,"e",-0.10)

    increments = [
        ("zero-window share\n(+10 pp)", "natural_zero_recovery_window_share", 0.1, ORANGE, "D"),
        ("longest event\n(+1 d)", "natural_longest_event_run_days", 1.0, BLUE, "o"),
        ("annual task\n(+1 log unit)", "log_annual_task_mwh", 1.0, PURPLE, "s"),
        ("recharge/task\n(+1 log unit)", "log_recharge_to_task_ratio", 1.0, TEAL, "^")]
    model = coeff[coeff.model_id == "joint_geometry_plus_scale"]
    for i,(label,predictor,scale,colour,marker) in enumerate(increments[::-1]):
        r=model[model.predictor==predictor].iloc[0]
        est,lo,hi=scale*r.coefficient,scale*r.bootstrap_ci95_low,scale*r.bootstrap_ci95_high
        axf.plot([lo,hi],[i,i],color=colour,lw=2.7,solid_capstyle="round")
        axf.scatter(est,i,s=42,color=colour,marker=marker,edgecolor="white",lw=0.5,zorder=4)
        axf.text(hi+0.08,i,f"{est:+.2f}",color=colour,va="center",fontsize=5.0,fontweight="bold")
    axf.axvline(0,color=INK,lw=0.75)
    axf.set_yticks(range(4),[x[0] for x in increments[::-1]])
    axf.set_xlabel("Adjusted effect (EUR kW$^{-1}$ yr$^{-1}$)")
    axf.set_xlim(-7.2,4.6)
    clean(axf,"x"); panel(axf,"f",-0.28)

    fig.subplots_adjust(left=0.085,right=0.985,top=0.98,bottom=0.06)
    save_bundle(fig,"fig_v51_figure6_sequencing_parity_choices")


def main() -> None:
    from plot_v51_reference_style_figures import main as reference_style_main

    reference_style_main()


if __name__=="__main__":
    main()
