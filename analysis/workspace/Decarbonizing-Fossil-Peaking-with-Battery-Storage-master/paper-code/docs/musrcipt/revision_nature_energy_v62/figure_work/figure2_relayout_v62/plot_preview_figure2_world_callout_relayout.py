#!/usr/bin/env python3
"""Preview a regime-led Figure 2 for three empirical balancing architectures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, ConnectionPatch, Polygon, Rectangle
from scipy.stats import gaussian_kde


HERE = Path(__file__).resolve().parent
FIGURE_WORK = HERE.parent
sys.path.insert(0, str(FIGURE_WORK))

from plot_v51_main_figures import (  # noqa: E402
    CHINA_GEO,
    JAPAN_GEO_ROOT,
    WORLD_SHP,
    add_geojson,
    add_shape,
)
from v60_figure_style import (  # noqa: E402
    GRID,
    INK,
    MM,
    MUTED,
    PALE,
    REGIME_COLORS,
    REGIME_MARKERS,
    configure,
    panel_label,
    quiet,
)


SOURCE = FIGURE_WORK / "source_data_v60"
CALENDAR_SOURCE = HERE / "source_data" / "figure2a_regime_calendar_daily.csv.gz"
OUT = HERE / "outputs"
WIDTH_MM = 183.0
HEIGHT_MM = 225.0

LAND = "#E8EBE9"
COAST = "#FFFFFF"
EVENT_COLOURS = ["#F3E5A7", "#C8DEA0", "#8FC48A", "#559B70", "#2D6B55"]
EVENT_BOUNDS = np.array([3, 6, 9, 12, 16, 24], dtype=float)
EVENT_CMAP = ListedColormap(EVENT_COLOURS, name="event_duration")
EVENT_NORM = BoundaryNorm(EVENT_BOUNDS, EVENT_CMAP.N)
EVENT_DAY_CMAP = ListedColormap(["#F7F8F7", "#7098B0"], name="event_day")

REGIMES = ["Europe", "China", "Japan"]
REGIME_LABELS = {"Europe": "NVSB", "China": "PLRP", "Japan": "CLDB"}
TASK_COLOUR = "#C76755"
RECHARGE_COLOUR = "#4C9489"

CHINA_NAMES = {
    "Anhui": "安徽省", "Beijing": "北京市", "Chongqing": "重庆市", "Fujian": "福建省",
    "Gansu": "甘肃省", "Guangdong": "广东省", "Guangxi": "广西壮族自治区", "Guizhou": "贵州省",
    "Hainan": "海南省", "Hebei": "河北省", "Heilongjiang": "黑龙江省", "Henan": "河南省",
    "Hubei": "湖北省", "Hunan": "湖南省", "Inner Mongolia": "内蒙古自治区", "Jiangsu": "江苏省",
    "Jiangxi": "江西省", "Jilin": "吉林省", "Liaoning": "辽宁省", "Ningxia": "宁夏回族自治区",
    "Qinghai": "青海省", "Shaanxi": "陕西省", "Shandong": "山东省", "Shanxi": "山西省",
    "Sichuan": "四川省", "Tianjin": "天津市", "Xinjiang": "新疆维吾尔自治区", "Yunnan": "云南省",
    "Zhejiang": "浙江省",
}


def configure_publication() -> None:
    configure()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def export_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(
        path.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(path.with_suffix(".png"), dpi=360)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(
        path.with_name(path.name + "_grayscale").with_suffix(".png"),
        grey,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def label_map(ax: plt.Axes, name: str, detail: str) -> None:
    colour = REGIME_COLORS[name]
    ax.text(
        0.025,
        0.965,
        name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=colour,
        fontsize=6.7,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.35},
        zorder=15,
    )
    ax.text(
        0.975,
        0.965,
        detail,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
        fontsize=5.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.35},
        zorder=15,
    )


def normalized_density(values: np.ndarray, edges: np.ndarray, sigma_bins: float = 1.05) -> tuple[np.ndarray, np.ndarray]:
    histogram, _ = np.histogram(values, bins=edges, density=True)
    radius = int(np.ceil(4.0 * sigma_bins))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(histogram.astype(float), kernel, mode="same")
    if density.max() > 0:
        density /= density.max()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def draw_event_marginals(
    duration_ax: plt.Axes,
    concentration_ax: plt.Axes,
    records: pd.DataFrame,
) -> None:
    duration_edges = np.arange(2.5, 24.5, 1.0)
    duration_baselines = {"Europe": 2.0, "China": 1.0, "Japan": 0.0}
    for regime in REGIMES:
        values = records.loc[records["regime"].eq(regime), "event_run_days"].to_numpy(float)
        x, density = normalized_density(values, duration_edges, sigma_bins=1.05)
        baseline = duration_baselines[regime]
        colour = REGIME_COLORS[regime]
        visible = density > 0.01
        duration_ax.fill_between(
            x[visible],
            baseline,
            baseline + 0.72 * density[visible],
            color=mpl.colors.to_rgba(colour, 0.28),
            lw=0,
        )
        duration_ax.plot(x[visible], baseline + 0.72 * density[visible], color=colour, lw=0.90)
    duration_ax.set_xlim(3, 23)
    duration_ax.set_ylim(-0.08, 2.80)
    duration_ax.set_xticks([4, 8, 12, 16, 20, 23])
    duration_ax.set_xlabel("")
    duration_ax.set_yticks([0, 1, 2], ["CLDB  n=27", "PLRP  n=26", "NVSB  n=132"])
    for tick, regime in zip(duration_ax.get_yticklabels(), ["Japan", "China", "Europe"]):
        tick.set_color(REGIME_COLORS[regime])
        tick.set_fontweight("bold")
    duration_ax.tick_params(axis="y", length=0, pad=3)
    duration_ax.spines[["top", "right", "left"]].set_visible(False)
    duration_ax.text(
        1.0,
        1.02,
        "longest event duration (d)",
        transform=duration_ax.transAxes,
        ha="right",
        va="bottom",
        color=INK,
        fontsize=5.5,
    )
    panel_label(duration_ax, "a", -0.035, 1.02)

    share_grid = np.linspace(40, 100, 301)
    share_edges = np.linspace(40, 100, 31)
    for regime in REGIMES:
        values = records.loc[records["regime"].eq(regime), "event_task_share_pct"].to_numpy(float)
        y, density = normalized_density(values, share_edges, sigma_bins=1.05)
        profile = np.interp(share_grid, y, density, left=0.0, right=0.0)
        colour = REGIME_COLORS[regime]
        concentration_ax.fill_betweenx(share_grid, 0, profile, color=mpl.colors.to_rgba(colour, 0.12), lw=0)
        concentration_ax.plot(profile, share_grid, color=colour, lw=0.82)
    concentration_ax.set_xlim(0, 1.04)
    concentration_ax.set_ylim(40, 100)
    concentration_ax.set_xticks([])
    concentration_ax.set_yticks([40, 60, 80, 100])
    concentration_ax.set_ylabel("q75 task share (%)", labelpad=2.0)
    concentration_ax.yaxis.set_label_position("right")
    concentration_ax.yaxis.tick_right()
    concentration_ax.tick_params(axis="y", pad=1.5)
    concentration_ax.spines[["top", "left"]].set_visible(False)


def draw_world_locator(
    fig: plt.Figure,
    ax: plt.Axes,
    inset_axes: dict[str, plt.Axes],
    europe: pd.DataFrame,
) -> None:
    reader = shapefile.Reader(str(WORLD_SHP), encoding="utf-8")
    eu_codes = set(europe["gisco_code"])
    for record in reader.iterShapeRecords():
        code = record.record.as_dict().get("ISO_A2")
        face, edge, lw, z = LAND, COAST, 0.18, 1
        if code in eu_codes:
            face, edge, lw, z = mpl.colors.to_rgba(REGIME_COLORS["Europe"], 0.74), "white", 0.28, 4
        elif code == "CN":
            face, edge, lw, z = mpl.colors.to_rgba(REGIME_COLORS["China"], 0.78), "white", 0.30, 4
        elif code == "JP":
            face, edge, lw, z = mpl.colors.to_rgba(REGIME_COLORS["Japan"], 0.90), "white", 0.34, 5
        add_shape(ax, record.shape, face, edge=edge, lw=lw, z=z)

    china_geo = json.loads(CHINA_GEO.read_text(encoding="utf-8"))
    china_face = mpl.colors.to_rgba(REGIME_COLORS["China"], 1.0)
    for feature in china_geo["features"]:
        if feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}:
            add_geojson(
                ax,
                feature["geometry"],
                china_face,
                edge=china_face,
                lw=0.08,
                z=6,
            )

    ax.set_xlim(-13, 150)
    ax.set_ylim(17, 74)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("S")
    ax.axis("off")

    boxes = {
        "Europe": (-13.0, 34.0, 46.0, 38.0),
        "China": (72.0, 17.5, 65.0, 39.0),
        "Japan": (128.0, 29.0, 20.0, 19.0),
    }
    anchors = {
        "Europe": ((-13.0, 34.0), (0.50, 1.00)),
        "China": ((72.0, 56.5), (0.36, 1.00)),
        "Japan": ((148.0, 29.0), (0.50, 1.00)),
    }
    for name in REGIMES:
        colour = REGIME_COLORS[name]
        x, y, width, height = boxes[name]
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=mpl.colors.to_rgba(colour, 0.035),
                edgecolor=colour,
                lw=0.82,
                zorder=8,
            )
        )
        ax.text(
            x + 0.5 * width,
            min(y + height + 1.2, 74.2),
            name,
            ha="center",
            va="bottom",
            color=colour,
            fontsize=5.8,
            fontweight="bold",
            zorder=9,
        )
        xy_world, xy_inset = anchors[name]
        connection = ConnectionPatch(
            xyA=xy_world,
            coordsA="data",
            axesA=ax,
            xyB=xy_inset,
            coordsB="axes fraction",
            axesB=inset_axes[name],
            color=mpl.colors.to_rgba(INK, 0.66),
            lw=0.64,
            zorder=13,
            clip_on=False,
        )
        fig.add_artist(connection)


def draw_europe(ax: plt.Axes, europe: pd.DataFrame) -> None:
    lookup = europe.set_index("gisco_code")["median_longest_event_days"].to_dict()
    reader = shapefile.Reader(str(WORLD_SHP), encoding="utf-8")
    for record in reader.iterShapeRecords():
        code = record.record.as_dict().get("ISO_A2")
        value = lookup.get(code)
        face = EVENT_CMAP(EVENT_NORM(value)) if value is not None else LAND
        edge = "white" if value is not None else "#CFD5D2"
        add_shape(ax, record.shape, face, edge=edge, lw=0.30 if value is not None else 0.16, z=3 if value is not None else 1)
    ax.set_xlim(-12, 32)
    ax.set_ylim(34, 72)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_china(ax: plt.Axes, china: pd.DataFrame) -> None:
    values = china.set_index("province")["longest_extreme_peaking_run_days"].to_dict()
    lookup = {CHINA_NAMES[name]: value for name, value in values.items() if name in CHINA_NAMES}
    geo = json.loads(CHINA_GEO.read_text(encoding="utf-8"))
    hainan_geometry = None
    for feature in geo["features"]:
        name = feature["properties"].get("name")
        value = lookup.get(name)
        face = EVENT_CMAP(EVENT_NORM(value)) if value is not None else LAND
        add_geojson(ax, feature["geometry"], face, edge="white", lw=0.25, z=3 if value is not None else 1)
        if name == "海南省":
            hainan_geometry = feature["geometry"]

    if hainan_geometry is not None:
        add_geojson(
            ax,
            hainan_geometry,
            LAND,
            edge="#9FA9A4",
            lw=0.30,
            z=5,
        )
    south_boundary = next(
        feature
        for feature in geo["features"]
        if feature["properties"].get("name") == "境界线"
        and min(point[1] for line in feature["geometry"]["coordinates"] for point in line) < 5
    )
    for line in south_boundary["geometry"]["coordinates"][:9]:
        points = np.asarray(line, dtype=float)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color="#9FA9A4",
            lw=0.60,
            solid_capstyle="round",
            zorder=6,
        )
    ax.set_xlim(73, 136)
    ax.set_ylim(2.5, 54)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_japan_region(
    ax: plt.Axes,
    points: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    label: str,
) -> None:
    for code in (11, 12, 13, 14, 22):
        path = JAPAN_GEO_ROOT / f"N03-20240101_{code}_GML/N03-20240101_{code}.geojson"
        for feature in json.loads(path.read_text(encoding="utf-8"))["features"]:
            add_geojson(ax, feature["geometry"], LAND, edge="#C5CFCA", lw=0.18, bounds=bounds)
    inside = points.loc[
        points["longitude"].between(bounds[0], bounds[1])
        & points["latitude"].between(bounds[2], bounds[3])
    ]
    ax.scatter(
        inside["longitude"],
        inside["latitude"],
        c=inside["event_run_days"],
        cmap=EVENT_CMAP,
        norm=EVENT_NORM,
        s=39,
        marker="h",
        edgecolor="white",
        lw=0.48,
        zorder=6,
    )
    ax.set_xlim(bounds[:2])
    ax.set_ylim(bounds[2:])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0.03,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=5.0,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.18},
        zorder=10,
    )


def draw_japan(ax: plt.Axes, japan: pd.DataFrame) -> None:
    community = japan.groupby(
        ["prefecture", "mesh_id", "latitude", "longitude"],
        as_index=False,
    ).agg(event_run_days=("longest_extreme_import_run_days", "median"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    positions = [
        [0.00, 0.51, 0.33, 0.46],
        [0.335, 0.51, 0.33, 0.46],
        [0.67, 0.51, 0.33, 0.46],
        [0.14, 0.03, 0.35, 0.46],
        [0.51, 0.03, 0.35, 0.46],
    ]
    for prefecture, position in zip(
        ["Shizuoka", "Saitama", "Tokyo", "Kanagawa", "Chiba"],
        positions,
    ):
        local = community.loc[community["prefecture"].eq(prefecture)].copy()
        lon_span = float(local["longitude"].max() - local["longitude"].min())
        lat_span = float(local["latitude"].max() - local["latitude"].min())
        lon_margin = max(0.022, 0.65 * lon_span)
        lat_margin = max(0.018, 0.65 * lat_span)
        bounds = (
            float(local["longitude"].min() - lon_margin),
            float(local["longitude"].max() + lon_margin),
            float(local["latitude"].min() - lat_margin),
            float(local["latitude"].max() + lat_margin),
        )
        draw_japan_region(ax.inset_axes(position), local, bounds, prefecture)


def draw_event_scale(ax: plt.Axes) -> None:
    labels = ["3–5", "6–8", "9–11", "12–15", "16–23"]
    ax.set_xlim(0, len(EVENT_COLOURS))
    ax.set_ylim(-0.45, 0.82)
    for index, (colour, label) in enumerate(zip(EVENT_COLOURS, labels)):
        ax.add_patch(
            Rectangle(
                (index, 0.10),
                1.0,
                0.25,
                facecolor=colour,
                edgecolor="white",
                lw=0.36,
            )
        )
        ax.text(index + 0.5, -0.02, label, ha="center", va="top", fontsize=5.2, color=INK)
    ax.text(
        2.5,
        0.54,
        "longest observed event (d)",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=5.6,
    )
    ax.axis("off")


def draw_map_panel(
    fig: plt.Figure,
    spec,
    europe: pd.DataFrame,
    china: pd.DataFrame,
    japan: pd.DataFrame,
) -> None:
    top = spec.subgridspec(
        3,
        12,
        height_ratios=[0.94, 0.77, 0.08],
        hspace=0.02,
        wspace=0.08,
    )
    world = fig.add_subplot(top[0, :])
    parents = {
        "Europe": fig.add_subplot(top[1, 0:4]),
        "China": fig.add_subplot(top[1, 4:8]),
        "Japan": fig.add_subplot(top[1, 8:12]),
    }
    map_axes = {
        name: parent.inset_axes([0.025, 0.025, 0.95, 0.91])
        for name, parent in parents.items()
    }
    for parent in parents.values():
        parent.set_xlim(0, 1)
        parent.set_ylim(0, 1)
        parent.axis("off")
        parent.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=parent.transAxes,
                fill=False,
                edgecolor="#AEB8B3",
                lw=0.42,
                clip_on=False,
                zorder=12,
            )
        )
    draw_world_locator(fig, world, parents, europe)
    draw_europe(map_axes["Europe"], europe)
    draw_china(map_axes["China"], china)
    draw_japan(map_axes["Japan"], japan)
    label_map(parents["Europe"], "Europe", "national | 15 countries | n=132")
    label_map(parents["China"], "China", "provincial | 26 provinces | n=26")
    label_map(parents["Japan"], "Japan", "community | 19 grids | n=27")
    panel_label(world, "a", -0.035, 1.02)
    draw_event_scale(fig.add_subplot(top[2, :]))


def draw_national_boundary_icon(ax: plt.Axes, colour: str) -> None:
    nodes = np.array(
        [
            [0.18, 0.57],
            [0.34, 0.69],
            [0.50, 0.56],
            [0.66, 0.69],
            [0.82, 0.57],
        ]
    )
    links = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (2, 4)]
    for start, end in links:
        ax.plot(
            nodes[[start, end], 0],
            nodes[[start, end], 1],
            color=mpl.colors.to_rgba(colour, 0.58),
            lw=0.78,
            zorder=2,
        )
    for index, (x_value, y_value) in enumerate(nodes):
        ax.add_patch(
            Circle(
                (x_value, y_value),
                0.030 if index == 2 else 0.024,
                facecolor=colour if index == 2 else "white",
                edgecolor=colour,
                lw=0.75,
                zorder=3,
            )
        )


def draw_provincial_boundary_icon(ax: plt.Axes, colour: str) -> None:
    hub = np.array([0.50, 0.61])
    nodes = np.array([[0.22, 0.69], [0.30, 0.52], [0.70, 0.52], [0.78, 0.69]])
    for node in nodes:
        ax.plot(
            [hub[0], node[0]],
            [hub[1], node[1]],
            color=mpl.colors.to_rgba(colour, 0.58),
            lw=0.78,
            zorder=2,
        )
        ax.add_patch(
            Rectangle(
                (node[0] - 0.025, node[1] - 0.025),
                0.050,
                0.050,
                facecolor="white",
                edgecolor=colour,
                lw=0.72,
                zorder=3,
            )
        )
    ax.add_patch(
        Rectangle(
            (hub[0] - 0.038, hub[1] - 0.038),
            0.076,
            0.076,
            facecolor=colour,
            edgecolor=colour,
            lw=0.75,
            zorder=4,
        )
    )


def draw_community_boundary_icon(ax: plt.Axes, colour: str) -> None:
    bus_y = 0.52
    ax.plot([0.16, 0.84], [bus_y, bus_y], color=mpl.colors.to_rgba(colour, 0.68), lw=0.95)
    for x_value in [0.26, 0.50, 0.74]:
        ax.plot([x_value, x_value], [bus_y, 0.59], color=mpl.colors.to_rgba(colour, 0.68), lw=0.72)
        ax.add_patch(
            Rectangle(
                (x_value - 0.055, 0.59),
                0.11,
                0.075,
                facecolor="white",
                edgecolor=colour,
                lw=0.70,
                zorder=3,
            )
        )
        ax.add_patch(
            Polygon(
                [[x_value - 0.067, 0.665], [x_value, 0.725], [x_value + 0.067, 0.665]],
                closed=True,
                facecolor=mpl.colors.to_rgba(colour, 0.20),
                edgecolor=colour,
                lw=0.70,
                zorder=3,
            )
        )
        ax.plot(
            [x_value - 0.030, x_value + 0.030],
            [0.676, 0.676],
            color=colour,
            lw=1.25,
            solid_capstyle="butt",
            zorder=4,
        )


def draw_regime_column(
    ax: plt.Axes,
    regime: str,
    scale_label: str,
    sample_label: str,
    hourly_pairs: int,
    task_label: str,
    recharge_label: str,
) -> None:
    colour = REGIME_COLORS[regime]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0.0, 0.84),
            1.0,
            0.16,
            facecolor=mpl.colors.to_rgba(colour, 0.085),
            edgecolor="none",
            zorder=0,
        )
    )
    ax.text(0.05, 0.945, scale_label, ha="left", va="top", color=INK, fontsize=7.2, fontweight="bold")
    ax.text(0.05, 0.865, f"{REGIME_LABELS[regime]}  |  {sample_label}", ha="left", va="bottom", color=colour, fontsize=5.6, fontweight="bold")

    if regime == "Europe":
        draw_national_boundary_icon(ax, colour)
    elif regime == "China":
        draw_provincial_boundary_icon(ax, colour)
    else:
        draw_community_boundary_icon(ax, colour)

    ax.text(0.05, 0.405, "task", ha="left", va="center", color=TASK_COLOUR, fontsize=5.4, fontweight="bold")
    ax.plot([0.235, 0.295], [0.405, 0.405], color=TASK_COLOUR, lw=2.5, solid_capstyle="butt")
    ax.text(0.315, 0.405, task_label, ha="left", va="center", color=INK, fontsize=5.25)
    ax.text(0.05, 0.285, "recharge", ha="left", va="center", color=RECHARGE_COLOUR, fontsize=5.4, fontweight="bold")
    ax.plot([0.235, 0.295], [0.285, 0.285], color=RECHARGE_COLOUR, lw=2.5, solid_capstyle="butt")
    ax.text(0.315, 0.285, recharge_label, ha="left", va="center", color=INK, fontsize=5.25)
    ax.text(
        0.50,
        0.105,
        f"{hourly_pairs:,} hourly task–recharge pairs",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=5.15,
    )
    ax.plot([0.04, 0.96], [0.035, 0.035], color="#D5DBD8", lw=0.48)


def draw_shared_regime_contract(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centres = [1 / 6, 0.5, 5 / 6]
    for x_value, regime in zip(centres, REGIMES):
        colour = REGIME_COLORS[regime]
        ax.plot([x_value, x_value], [1.00, 0.73], color=mpl.colors.to_rgba(colour, 0.70), lw=0.72)
        ax.add_patch(Circle((x_value, 0.70), 0.014, facecolor=colour, edgecolor="white", lw=0.35, zorder=3))
    ax.plot([centres[0], centres[-1]], [0.70, 0.70], color="#AEB7B3", lw=0.62, zorder=1)
    ax.text(0.24, 0.34, "system–year", ha="center", va="center", color=INK, fontsize=5.7, fontweight="bold")
    ax.annotate(
        "",
        xy=(0.42, 0.34),
        xytext=(0.34, 0.34),
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.72, "mutation_scale": 7},
    )
    ax.text(0.61, 0.34, "365 intact paired 24 h blocks", ha="center", va="center", color=INK, fontsize=5.7, fontweight="bold")
    ax.annotate(
        "",
        xy=(0.88, 0.34),
        xytext=(0.80, 0.34),
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.72, "mutation_scale": 7},
    )
    ax.text(0.92, 0.34, "common\ntest", ha="center", va="center", color=MUTED, fontsize=5.15, linespacing=0.92)


def draw_regime_architecture(fig: plt.Figure, spec, task_contract: pd.DataFrame) -> None:
    container = fig.add_subplot(spec)
    container.set_axis_off()
    panel_label(container, "a", -0.035, 1.015)
    layout = spec.subgridspec(2, 3, height_ratios=[0.84, 0.16], hspace=0.00, wspace=0.10)
    definitions = {
        "Europe": {
            "scale": "National VRE self-balancing",
            "sample": "15 countries  |  n=132",
            "task": "gas above stable baseline",
            "recharge": "synchronous wind + PV surplus",
        },
        "China": {
            "scale": "Provincial last-resort peaking",
            "sample": "26 provinces  |  n=26",
            "task": "observed gas-peaking output",
            "recharge": "wind + PV curtailment",
        },
        "Japan": {
            "scale": "Community-level distributed balancing",
            "sample": "19 grids  |  n=27",
            "task": "grid import above stable baseline",
            "recharge": "distributed-PV reverse flow",
        },
    }
    contract = task_contract.set_index("regime")
    for index, regime in enumerate(REGIMES):
        values = definitions[regime]
        axis = fig.add_subplot(layout[0, index])
        draw_regime_column(
            axis,
            regime=regime,
            scale_label=values["scale"],
            sample_label=values["sample"],
            hourly_pairs=int(contract.loc[regime, "hourly_observations"]),
            task_label=values["task"],
            recharge_label=values["recharge"],
        )
        if index < 2:
            axis.plot([1.02, 1.02], [0.07, 0.97], color="#D8DDDA", lw=0.48, clip_on=False)
    draw_shared_regime_contract(fig.add_subplot(layout[1, :]))


def draw_regime_header(ax: plt.Axes, regime: str, scale: str, n_label: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.76, scale, ha="left", va="center", color=INK, fontsize=6.2, fontweight="bold")
    ax.text(0.0, 0.06, f"{REGIME_LABELS[regime]}  |  {n_label}", ha="left", va="bottom", color=REGIME_COLORS[regime], fontsize=5.2, fontweight="bold")


def draw_observed_profile(ax: plt.Axes, frame: pd.DataFrame, regime: str) -> None:
    profile = (
        frame.groupby("source_day_id", as_index=False)
        .agg(
            task=("base_task_display_norm", "mean"),
            recharge=("recharge_display_norm", "mean"),
        )
        .sort_values("source_day_id")
    )
    x_values = profile["source_day_id"].to_numpy(float)
    task = profile["task"].to_numpy(float)
    recharge = profile["recharge"].to_numpy(float)
    ax.fill_between(x_values, 0, task, color=mpl.colors.to_rgba(TASK_COLOUR, 0.62), lw=0)
    ax.plot(x_values, task, color=TASK_COLOUR, lw=0.58)
    ax.fill_between(x_values, 0, -recharge, color=mpl.colors.to_rgba(RECHARGE_COLOUR, 0.52), lw=0)
    ax.plot(x_values, -recharge, color=RECHARGE_COLOUR, lw=0.58)
    ax.axhline(0, color=INK, lw=0.45)
    ax.set_xlim(-0.5, 364.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_yticklabels(["−0.5", "0", "0.5"])
    ax.tick_params(axis="y", labelsize=5.0, pad=1.5)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.grid(False)
    ax.text(0.012, 0.91, "task", transform=ax.transAxes, ha="left", va="top", color=TASK_COLOUR, fontsize=5.0, fontweight="bold")
    ax.text(0.012, 0.08, "recharge", transform=ax.transAxes, ha="left", va="bottom", color=RECHARGE_COLOUR, fontsize=5.0, fontweight="bold")


def draw_event_calendar(
    ax: plt.Axes,
    margin_ax: plt.Axes,
    frame: pd.DataFrame,
    regime: str,
    show_xlabel: bool,
) -> None:
    keys = ["system_id", "observation_year"]
    metrics = (
        frame[keys + ["annual_task_percentile", "event_run_days"]]
        .drop_duplicates()
        .sort_values(["annual_task_percentile", "system_id", "observation_year"], ascending=[False, True, True])
    )
    metrics["case_key"] = metrics["system_id"].astype(str) + "|" + metrics["observation_year"].astype(str)
    work = frame.copy()
    work["case_key"] = work["system_id"].astype(str) + "|" + work["observation_year"].astype(str)
    matrix = (
        work.pivot(index="case_key", columns="source_day_id", values="event_flag_q75")
        .reindex(index=metrics["case_key"], columns=np.arange(365))
        .to_numpy(float)
    )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Incomplete event-calendar matrix for {regime}")
    ax.imshow(
        matrix,
        cmap=EVENT_DAY_CMAP,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
        rasterized=True,
    )
    ax.set_xlim(-0.5, 364.5)
    if show_xlabel:
        ax.set_xticks([0, 59, 119, 179, 239, 299, 364], ["1", "60", "120", "180", "240", "300", "365"])
        ax.set_xlabel("day of operating year", labelpad=1.5, fontsize=5.2)
    else:
        ax.set_xticks([])
        ax.set_xlabel("")
    n_rows = len(metrics)
    ax.set_yticks([0, n_rows - 1], ["high task", "low task"])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)

    duration = metrics["event_run_days"].to_numpy(float)[:, None]
    margin_ax.imshow(
        duration,
        cmap=EVENT_CMAP,
        norm=EVENT_NORM,
        interpolation="nearest",
        aspect="auto",
        rasterized=True,
    )
    margin_ax.set_xticks([])
    margin_ax.set_yticks([])
    for spine in margin_ax.spines.values():
        spine.set_visible(False)
    margin_ax.text(0.5, 1.012, "run", transform=margin_ax.transAxes, ha="center", va="bottom", color=MUTED, fontsize=5.0)


def draw_regime_calendar_atlas(fig: plt.Figure, spec, calendar: pd.DataFrame) -> None:
    container = fig.add_subplot(spec)
    container.set_axis_off()
    panel_label(container, "a", -0.035, 1.02)
    layout = spec.subgridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.24], hspace=0.27)
    labels = {
        "Europe": ("National VRE self-balancing", "132 country–years"),
        "China": ("Provincial last-resort peaking", "26 province–years"),
        "Japan": ("Community-level distributed balancing", "27 community–years"),
    }
    for index, regime in enumerate(REGIMES):
        inner = layout[index, 0].subgridspec(
            3,
            2,
            height_ratios=[0.20, 0.36, 0.44],
            width_ratios=[0.975, 0.025],
            hspace=0.08,
            wspace=0.025,
        )
        frame = calendar.loc[calendar["regime"].eq(regime)].copy()
        scale, n_label = labels[regime]
        draw_regime_header(fig.add_subplot(inner[0, :]), regime, scale, n_label)
        draw_observed_profile(fig.add_subplot(inner[1, 0]), frame, regime)
        draw_event_calendar(
            fig.add_subplot(inner[2, 0]),
            fig.add_subplot(inner[2, 1]),
            frame,
            regime,
            show_xlabel=True,
        )

    legend_layout = layout[3, 0].subgridspec(1, 2, width_ratios=[0.975, 0.025], wspace=0.025)
    legend_ax = fig.add_subplot(legend_layout[0, 0])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.add_patch(Rectangle((0.0, 0.42), 0.030, 0.20, facecolor=EVENT_DAY_CMAP(1), edgecolor="none"))
    legend_ax.text(0.039, 0.52, "q75 event day", ha="left", va="center", color=INK, fontsize=5.1)
    labels_duration = ["3–5", "6–8", "9–11", "12–15", "16–23"]
    start = 0.31
    width = (1.0 - start) / len(labels_duration)
    legend_ax.text(0.18, 0.52, "longest event (d)", ha="left", va="center", color=INK, fontsize=5.0)
    for offset, (colour, label) in enumerate(zip(EVENT_COLOURS, labels_duration)):
        x_value = start + offset * width
        legend_ax.add_patch(Rectangle((x_value, 0.42), width, 0.20, facecolor=colour, edgecolor="white", lw=0.30))
        legend_ax.text(x_value + width / 2, 0.34, label, ha="center", va="top", color=INK, fontsize=5.0)


def draw_task_rank_pulses(ax: plt.Axes, records: pd.DataFrame) -> None:
    panel_label(ax, "c", -0.08, 1.04)
    y_positions = {"Europe": 2.0, "China": 1.0, "Japan": 0.0}
    duration_min = float(EVENT_BOUNDS[0])
    duration_max = float(EVENT_BOUNDS[-1] - 1)

    for regime in REGIMES:
        frame = records.loc[records["regime"].eq(regime)].sort_values("annual_task_percentile")
        task_rank = frame["annual_task_percentile"].to_numpy(float)
        duration = frame["event_run_days"].to_numpy(float)
        amplitude = 0.10 + 0.62 * np.clip(
            (duration - duration_min) / (duration_max - duration_min),
            0.0,
            1.0,
        )
        baseline = y_positions[regime]
        colour = REGIME_COLORS[regime]

        ax.hlines(baseline, 0, 100, color="#D8DDDA", lw=0.48, zorder=1)
        ax.vlines(
            task_rank,
            baseline,
            baseline + amplitude,
            color=mpl.colors.to_rgba(colour, 0.30),
            lw=0.52,
            zorder=2,
        )
        ax.scatter(
            task_rank,
            baseline + amplitude,
            c=duration,
            cmap=EVENT_CMAP,
            norm=EVENT_NORM,
            s=7.2 if regime == "Europe" else 10.5,
            marker=REGIME_MARKERS[regime],
            edgecolor="white",
            lw=0.24,
            zorder=3,
        )

        quintile = pd.cut(task_rank, bins=[-0.1, 20, 40, 60, 80, 100.1], labels=False)
        median_x: list[float] = []
        median_y: list[float] = []
        for index in range(5):
            values = duration[quintile == index]
            if len(values) == 0:
                continue
            median_duration = float(np.median(values))
            median_amplitude = 0.10 + 0.62 * np.clip(
                (median_duration - duration_min) / (duration_max - duration_min),
                0.0,
                1.0,
            )
            median_x.append(10.0 + 20.0 * index)
            median_y.append(baseline + median_amplitude)
        ax.scatter(
            median_x,
            median_y,
            s=17,
            marker="D",
            facecolor="white",
            edgecolor=colour,
            lw=0.72,
            zorder=5,
        )

        rho = float(frame[["annual_task_percentile", "event_run_days"]].corr(method="spearman").iloc[0, 1])
        ax.text(
            99.0,
            baseline + 0.67,
            f"$\\rho$={rho:+.2f}",
            ha="right",
            va="top",
            color=colour,
            fontsize=5.3,
            fontweight="bold",
        )

    for regime in REGIMES:
        ax.text(
            1.5,
            y_positions[regime] - 0.055,
            f"{REGIME_LABELS[regime]}  n={len(records.loc[records['regime'].eq(regime)])}",
            ha="left",
            va="top",
            color=REGIME_COLORS[regime],
            fontsize=5.2,
            fontweight="bold",
        )
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.18, 2.78)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("within-regime annual-task percentile")
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(False)
    ax.text(
        0.0,
        1.025,
        "pulse height and colour: longest event; diamonds: quintile median",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=5.0,
    )


def draw_event_geometry(ax: plt.Axes, records: pd.DataFrame) -> None:
    panel_label(ax, "d", -0.13, 1.04)
    x_grid = np.linspace(38, 100, 260)
    y_grid = np.linspace(2, 24, 220)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
    label_offsets = {"Europe": (7, -9), "China": (7, 5), "Japan": (7, 7)}
    for regime in REGIMES:
        frame = records.loc[records["regime"].eq(regime)]
        colour = REGIME_COLORS[regime]
        marker = REGIME_MARKERS[regime]
        x_values = frame["event_task_share_pct"].to_numpy(float)
        y_values = frame["event_run_days"].to_numpy(float)
        density = gaussian_kde(np.vstack([x_values, y_values]), bw_method="scott")(positions).reshape(grid_x.shape)
        ordered = np.sort(density.ravel())[::-1]
        cumulative = np.cumsum(ordered) / ordered.sum()
        threshold_50 = ordered[min(np.searchsorted(cumulative, 0.50), len(ordered) - 1)]
        threshold_80 = ordered[min(np.searchsorted(cumulative, 0.80), len(ordered) - 1)]
        ax.contourf(
            grid_x,
            grid_y,
            density,
            levels=[threshold_80, threshold_50, density.max() * 1.0001],
            colors=[mpl.colors.to_rgba(colour, 0.16), mpl.colors.to_rgba(colour, 0.38)],
            antialiased=True,
            zorder=2,
        )
        ax.contour(
            grid_x,
            grid_y,
            density,
            levels=[threshold_80, threshold_50],
            colors=[mpl.colors.to_rgba(colour, 0.72)],
            linewidths=[0.52, 0.78],
            zorder=3,
        )
        x_q25, x_median, x_q75 = np.quantile(x_values, [0.25, 0.50, 0.75])
        y_q25, y_median, y_q75 = np.quantile(y_values, [0.25, 0.50, 0.75])
        ax.errorbar(
            x_median,
            y_median,
            xerr=[[x_median - x_q25], [x_q75 - x_median]],
            yerr=[[y_median - y_q25], [y_q75 - y_median]],
            fmt=marker,
            ms=6.0,
            mfc=colour,
            mec="white",
            mew=0.62,
            ecolor=colour,
            elinewidth=0.82,
            capsize=2.0,
            zorder=6,
        )
        ax.annotate(
            REGIME_LABELS[regime],
            (x_median, y_median),
            xytext=label_offsets[regime],
            textcoords="offset points",
            color=colour,
            fontweight="bold",
            fontsize=5.7,
        )
    ax.text(0.98, 0.97, "50% / 80% density contours", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=5.2)
    ax.set_xlim(38, 100)
    ax.set_ylim(2, 24)
    ax.set_xticks([40, 60, 80, 100])
    ax.set_yticks([4, 8, 12, 16, 20, 24])
    ax.set_xlabel("annual task on q75 event days (%)")
    ax.set_ylabel("longest event (d)")
    quiet(ax, "both")


def barcode_rgba(values: np.ndarray, colour: str) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    rgb = np.asarray(mpl.colors.to_rgb(colour))
    strength = np.where(values > 0, 0.16 + 0.84 * values, 0.025)
    image = np.ones((1, len(values), 4), dtype=float)
    image[0, :, :3] = 1.0 - strength[:, None] * (1.0 - rgb[None, :])
    image[0, :, 3] = 1.0
    return image


def chronology_keys(task: np.ndarray, recharge: np.ndarray) -> list[tuple[float, float, int]]:
    occurrence: dict[tuple[float, float], int] = {}
    keys: list[tuple[float, float, int]] = []
    for task_value, recharge_value in zip(task, recharge):
        pair = (round(float(task_value), 10), round(float(recharge_value), 10))
        index = occurrence.get(pair, 0)
        keys.append((pair[0], pair[1], index))
        occurrence[pair] = index + 1
    return keys


def draw_conserved_sequences(fig: plt.Figure, spec, chronology: pd.DataFrame, summary: pd.DataFrame) -> plt.Axes:
    del summary
    ax = fig.add_subplot(spec)
    task_colour = TASK_COLOUR
    recharge_colour = RECHARGE_COLOUR
    dispersed_task = chronology["dispersed_task_normalized"].to_numpy(float)
    dispersed_recharge = chronology["dispersed_recharge_normalized"].to_numpy(float)
    clustered_task = chronology["clustered_task_normalized"].to_numpy(float)
    clustered_recharge = chronology["clustered_recharge_normalized"].to_numpy(float)

    rows = [
        (3.32, dispersed_task, task_colour),
        (2.72, dispersed_recharge, recharge_colour),
        (1.32, clustered_task, task_colour),
        (0.72, clustered_recharge, recharge_colour),
    ]
    for y, values, colour in rows:
        ax.imshow(
            barcode_rgba(values, colour),
            extent=(0.5, 365.5, y - 0.22, y + 0.22),
            aspect="auto",
            interpolation="nearest",
            rasterized=True,
            zorder=3,
        )
        ax.add_patch(
            Rectangle(
                (0.5, y - 0.22),
                365,
                0.44,
                fill=False,
                edgecolor="#CDD3D0",
                lw=0.38,
                zorder=4,
            )
        )

    dispersed_keys = chronology_keys(dispersed_task, dispersed_recharge)
    clustered_keys = chronology_keys(clustered_task, clustered_recharge)
    clustered_positions = {key: index + 1 for index, key in enumerate(clustered_keys)}
    candidates = np.flatnonzero(dispersed_task >= np.quantile(dispersed_task[dispersed_task > 0], 0.80))
    selected = candidates[np.unique(np.rint(np.linspace(0, len(candidates) - 1, 6)).astype(int))]
    for index in selected:
        x_top = index + 1
        x_bottom = clustered_positions[dispersed_keys[index]]
        ax.plot([x_top, x_bottom], [2.45, 1.59], color=mpl.colors.to_rgba(MUTED, 0.48), lw=0.48, zorder=2)
        ax.scatter([x_top, x_bottom], [2.45, 1.59], s=5.0, color=MUTED, edgecolor="white", lw=0.25, zorder=5)

    ax.text(0.012, 0.975, "dispersed", transform=ax.transAxes, ha="left", va="top", color=INK, fontweight="bold", fontsize=5.8)
    ax.text(0.012, 0.535, "controlled clustering (≤4 d)", transform=ax.transAxes, ha="left", va="top", color=INK, fontweight="bold", fontsize=5.8)
    panel_label(ax, "b", -0.035, 1.04)

    ax.set_xlim(0.5, 365.5)
    ax.set_ylim(-0.25, 3.95)
    ax.set_xticks([1, 60, 120, 180, 240, 300, 365])
    ax.set_xlabel("reordered day-block position", labelpad=2.0)
    ax.set_yticks([3.32, 2.72, 1.32, 0.72], ["task", "recharge", "task", "recharge"])
    for tick, colour in zip(ax.get_yticklabels(), [task_colour, recharge_colour] * 2):
        tick.set_color(colour)
        tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(False)

    ax.plot([], [], color=task_colour, lw=4, label="task intensity")
    ax.plot([], [], color=recharge_colour, lw=4, label="recharge intensity")
    ax.plot([], [], color=MUTED, lw=0.65, marker="o", ms=2.2, label="selected intact-day link")
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1.055), ncol=3, handlelength=1.3, columnspacing=1.1)

    ledger = [
        (0.16, "365 blocks = 365 blocks"),
        (0.50, "annual task  100% = 100%"),
        (0.84, "annual recharge  100% = 100%"),
    ]
    for x, text_value in ledger:
        ax.text(x, 0.035, text_value, transform=ax.transAxes, ha="center", va="bottom", color=MUTED, fontsize=5.4, fontweight="bold")
    return ax


def main() -> None:
    configure_publication()
    calendar = pd.read_csv(CALENDAR_SOURCE)
    records = pd.read_csv(SOURCE / "figure2_task_event_records.csv")
    summary = pd.read_csv(SOURCE / "figure2_regime_summary.csv")
    chronology = pd.read_csv(SOURCE / "figure2_conserved_chronology.csv")

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        5,
        12,
        left=0.095,
        right=0.950,
        bottom=0.050,
        top=0.982,
        height_ratios=[1.50, 0.04, 0.56, 0.18, 0.58],
        hspace=0.0,
        wspace=0.86,
    )
    draw_regime_calendar_atlas(fig, grid[0, :], calendar)
    barcode_ax = draw_conserved_sequences(fig, grid[2, :], chronology, summary)
    draw_task_rank_pulses(fig.add_subplot(grid[4, 0:5]), records)
    draw_event_geometry(fig.add_subplot(grid[4, 6:12]), records)
    export_figure(fig, "fig_v62_preview_figure2_world_callout_relayout")
    plt.close(fig)


if __name__ == "__main__":
    main()
