"""Shared Nature-style visual language for the v60 manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


MM = 1.0 / 25.4
WIDTH_MM = 183.0

INK = "#26353C"
MUTED = "#6C777A"
GRID = "#D9DFDD"
PALE = "#F3F5F3"
WHITE = "#FFFFFF"

CORAL = "#D76F54"
CORAL_LIGHT = "#EDB7A9"
BLUE = "#4A90A8"
BLUE_LIGHT = "#A8CBDA"
TEAL = "#3B8F7E"
GOLD = "#D1A13A"
PURPLE = "#7364A8"
GREEN = "#4F8B66"

REGIME_COLORS = {"Europe": CORAL, "China": GOLD, "Japan": BLUE}
REGIME_MARKERS = {"Europe": "o", "China": "s", "Japan": "^"}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.labelsize": 6.3,
            "axes.titlesize": 6.7,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.5,
            "legend.frameon": False,
            "axes.linewidth": 0.62,
            "lines.linewidth": 1.0,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.major.width": 0.50,
            "ytick.major.width": 0.50,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def quiet(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.42)
        ax.set_axisbelow(True)


def deterministic_jitter(n: int, scale: float = 0.10) -> np.ndarray:
    return scale * np.sin((np.arange(n) + 1) * 2.399963)


def density_curve(values: np.ndarray, bins: int = 90, smooth: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0, 1.0]), np.zeros(2)
    lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.05 * (hi - lo)
    hist, edges = np.histogram(values, bins=bins, range=(lo - pad, hi + pad), density=True)
    kx = np.arange(-14, 15)
    kernel = np.exp(-0.5 * (kx / smooth) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(hist, kernel, mode="same")
    if density.max() > 0:
        density /= density.max()
    return (edges[:-1] + edges[1:]) / 2.0, density


def interval_mark(
    ax: plt.Axes,
    values: np.ndarray,
    y: float,
    colour: str,
    marker: str = "o",
    raw: bool = True,
    jitter: float = 0.10,
) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if raw:
        ax.scatter(
            values,
            y + deterministic_jitter(len(values), jitter),
            s=7.0,
            color=mpl.colors.to_rgba(colour, 0.30),
            edgecolor="white",
            lw=0.18,
            rasterized=True,
            zorder=2,
        )
    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    ax.plot([q1, q3], [y, y], color=colour, lw=3.0, solid_capstyle="round", zorder=3)
    ax.scatter(median, y, s=30, marker=marker, color=colour, edgecolor="white", lw=0.5, zorder=4)
    return float(q1), float(median), float(q3)


def ridge(
    ax: plt.Axes,
    values: np.ndarray,
    baseline: float,
    colour: str,
    height: float = 0.65,
    direction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    x, density = density_curve(values)
    y = baseline + direction * height * density
    ax.fill_between(x, baseline, y, color=mpl.colors.to_rgba(colour, 0.24), lw=0)
    ax.plot(x, y, color=colour, lw=0.9)
    return x, density


def save_bundle(fig: plt.Figure, out: Path, stem: str, png_dpi: int = 360) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=png_dpi)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(path.with_name(path.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)
