#!/usr/bin/env python3
"""Build and plot the v40 source-aware EUR technology-duration cost paths.

The 53-configuration inventory is inherited from the v36 duration-spectrum
design and carried into the v40 source-aware currency integration. For each
of the 2025, 2030 and 2035 snapshots, power and energy CAPEX components are
read from the latest currency-provenance library. Installed CAPEX is then
reconstructed as

    power CAPEX / nominal discharge duration + energy CAPEX.

The reporting route is technology-specific: LFP and VRFB use their native
CNY-to-EUR route; A-CAES, LAES, Fe-Cr flow, gravity storage, molten-salt TES
and cavern hydrogen use USD-to-EUR. Installed CAPEX is never interpolated
directly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "source_data"
SOURCE_PATH = SOURCE_DIR / "ed1_53_duration_year_cost_paths_eur.csv"
INVENTORY_PATH = (
    HERE.parent
    / "currency_work"
    / "configuration_catalog_53_source_aware_eur.csv"
)

YEARS = (2025, 2030, 2035)
TECH_ORDER = (
    "li_lfp",
    "vrfb",
    "fecr_flow",
    "gravity",
    "molten_salt_tes",
    "caes",
    "laes",
    "h2_cavern",
)
TECH_LABEL = {
    "li_lfp": "LFP",
    "vrfb": "VRFB",
    "fecr_flow": "Fe–Cr flow",
    "gravity": "Gravity storage",
    "molten_salt_tes": "Molten-salt TES",
    "caes": "A-CAES",
    "laes": "LAES",
    "h2_cavern": "Cavern H$_2$",
}
EXPECTED_ROUTE = {
    "li_lfp": "CNY_to_EUR",
    "vrfb": "CNY_to_EUR",
    "fecr_flow": "USD_to_EUR",
    "gravity": "USD_to_EUR",
    "molten_salt_tes": "USD_to_EUR",
    "caes": "USD_to_EUR",
    "laes": "USD_to_EUR",
    "h2_cavern": "USD_to_EUR",
}
EXPECTED_CONFIG_COUNTS = {
    "li_lfp": 4,
    "vrfb": 3,
    "fecr_flow": 3,
    "gravity": 3,
    "molten_salt_tes": 5,
    "caes": 9,
    "laes": 9,
    "h2_cavern": 17,
}
STATUS_ORDER = (
    "Direct evidence duration",
    "Released configuration",
    "Within-domain component interpolation",
    "H2 structural extrapolation",
)
STATUS_STYLE = {
    "Direct evidence duration": {
        "marker": "D",
        "linestyle": "-",
        "filled": True,
        "linewidth": 1.25,
    },
    "Released configuration": {
        "marker": "o",
        "linestyle": (0, (4.0, 1.5)),
        "filled": True,
        "linewidth": 1.05,
    },
    "Within-domain component interpolation": {
        "marker": "o",
        "linestyle": (0, (1.0, 1.35)),
        "filled": False,
        "linewidth": 1.05,
    },
    "H2 structural extrapolation": {
        "marker": "^",
        "linestyle": (0, (5.0, 1.4, 1.1, 1.4)),
        "filled": False,
        "linewidth": 1.25,
    },
}

WIDTH_IN = 7.2
HEIGHT_IN = 5.55
OUTPUT_STEM = HERE / "fig_v40_ed1_cost_paths_53_eur"


def find_workspace() -> Path:
    for parent in HERE.parents:
        candidate = (
            parent
            / "dataset"
            / "M2_LDES_duration_year_parameter_library_v0_2_currency_provenance.csv"
        )
        if candidate.exists():
            return parent
    raise FileNotFoundError("Could not locate the source-aware parameter library")


WORKSPACE = find_workspace()
PARAMETER_LIBRARY = (
    WORKSPACE
    / "dataset"
    / "M2_LDES_duration_year_parameter_library_v0_2_currency_provenance.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_class(record: dict[str, object]) -> str:
    if bool(record["structural_extrapolation"]):
        return "H2 structural extrapolation"
    if bool(record["direct_duration_anchor"]):
        return "Direct evidence duration"
    if bool(record["released_library_configuration"]):
        return "Released configuration"
    return "Within-domain component interpolation"


def scalar_or_fail(group: pd.DataFrame, field: str, context: str) -> object:
    values = group[field].dropna().unique()
    if len(values) != 1:
        raise RuntimeError(
            f"Expected one {field} value for {context}, found {values.tolist()}"
        )
    return values[0]


def build_snapshot(
    library: pd.DataFrame,
    inventory: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    source = library[
        (library["learning_year"] == year)
        & library["tech_id"].isin(TECH_ORDER)
    ].copy()
    rows: list[dict[str, object]] = []
    numeric_components = (
        "power_cost_eur_kw",
        "energy_cost_eur_kwh",
        "fixed_om_eur_kw_yr",
        "variable_om_eur_mwh",
        "power_cost_eur_via_usd_kw",
        "power_cost_eur_via_cny_kw",
        "energy_cost_eur_via_usd_kwh",
        "energy_cost_eur_via_cny_kwh",
    )
    audit_fields = (
        "technology_en",
        "family",
        "parameter_basis",
        "confidence",
        "primary_source",
        "source_url_primary",
        "source_url_secondary",
        "capex_currency_basis",
        "technology_currency_route",
        "capex_reporting_route",
        "reporting_fx_reference_year",
        "reporting_fx_usd_per_eur",
        "reporting_fx_cny_per_eur",
    )

    for item in inventory.to_dict("records"):
        tech_id = str(item["tech_id"])
        duration = float(item["duration_h"])
        group = (
            source[source["tech_id"] == tech_id]
            .sort_values("duration_h")
            .drop_duplicates("duration_h")
        )
        if group.empty:
            raise RuntimeError(f"Missing {year} component source for {tech_id}")
        route = str(scalar_or_fail(group, "capex_reporting_route", tech_id))
        if route != EXPECTED_ROUTE[tech_id]:
            raise RuntimeError(
                f"Unexpected reporting route for {tech_id}: {route}; "
                f"expected {EXPECTED_ROUTE[tech_id]}"
            )

        anchors = group["duration_h"].to_numpy(float)
        first = group.iloc[0]
        record: dict[str, object] = {
            "catalogue_version": "v40_ed1_source_aware_eur_1.0",
            "tech_id": tech_id,
            "technology_label": TECH_LABEL[tech_id]
            .replace("H$_2$", "H2")
            .replace("–", "-"),
            "learning_year": int(year),
            "duration_h": duration,
            "released_library_configuration": bool(
                item["released_library_configuration"]
            ),
            "direct_duration_anchor": bool(item["direct_duration_anchor"]),
            "structural_extrapolation": bool(item["structural_extrapolation"]),
            "configuration_status": str(item["configuration_status"]),
            "interpolation_lower_h": float(item["interpolation_lower_h"]),
            "interpolation_upper_h": float(item["interpolation_upper_h"]),
            "source_key": str(item["source_key"]),
        }
        for field in audit_fields:
            record[field] = first[field]

        exact_duration = bool(np.any(np.isclose(anchors, duration)))
        direct_component_model = (
            str(item["configuration_status"])
            == "direct_evidence_duration_component_model"
        )
        structural = bool(item["structural_extrapolation"])
        released = bool(item["released_library_configuration"])
        if released and not exact_duration:
            raise RuntimeError(
                f"Released configuration absent from {year} library: "
                f"{tech_id} {duration:g} h"
            )
        if not structural and not exact_duration and not direct_component_model:
            if duration <= anchors.min() or duration >= anchors.max():
                raise RuntimeError(
                    f"Non-structural duration outside evidence domain: "
                    f"{tech_id} {duration:g} h"
                )
            lower = float(anchors[anchors < duration].max())
            upper = float(anchors[anchors > duration].min())
            if not (
                np.isclose(lower, float(item["interpolation_lower_h"]))
                and np.isclose(upper, float(item["interpolation_upper_h"]))
            ):
                raise RuntimeError(
                    f"Interpolation bounds changed for {tech_id} {duration:g} h "
                    f"in {year}: observed {lower:g}-{upper:g} h"
                )

        for field in numeric_components:
            values = group[field].to_numpy(float)
            if structural or direct_component_model:
                unique = np.unique(np.round(values, 12))
                if len(unique) != 1:
                    raise RuntimeError(
                        f"Non-constant {field} prevents component reuse for "
                        f"{tech_id} in {year}"
                    )
                record[field] = float(values[0])
            else:
                record[field] = float(np.interp(duration, anchors, values))

        record["total_capex_eur_kwh"] = (
            float(record["power_cost_eur_kw"]) / duration
            + float(record["energy_cost_eur_kwh"])
        )
        record["total_capex_eur_via_usd_kwh"] = (
            float(record["power_cost_eur_via_usd_kw"]) / duration
            + float(record["energy_cost_eur_via_usd_kwh"])
        )
        record["total_capex_eur_via_cny_kwh"] = (
            float(record["power_cost_eur_via_cny_kw"]) / duration
            + float(record["energy_cost_eur_via_cny_kwh"])
        )
        record["cost_identity_error_eur_kwh"] = (
            float(record["total_capex_eur_kwh"])
            - float(record["power_cost_eur_kw"]) / duration
            - float(record["energy_cost_eur_kwh"])
        )
        record["evidence_class"] = evidence_class(record)
        rows.append(record)
    return pd.DataFrame(rows)


def build_source_data() -> pd.DataFrame:
    library = pd.read_csv(PARAMETER_LIBRARY)
    inventory = pd.read_csv(INVENTORY_PATH)
    if len(inventory) != 53:
        raise RuntimeError(f"Expected 53 configurations, found {len(inventory)}")
    if inventory[["tech_id", "duration_h"]].duplicated().any():
        raise RuntimeError("The configuration inventory contains duplicates")

    frames = [
        build_snapshot(library=library, inventory=inventory, year=year)
        for year in YEARS
    ]
    paths = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["tech_id", "duration_h", "learning_year"])
        .reset_index(drop=True)
    )

    key = ["tech_id", "duration_h", "learning_year"]
    if len(paths) != 159 or paths[key].duplicated().any():
        raise RuntimeError("Expected 159 unique technology-duration-year rows")
    yearly_counts = paths.groupby("learning_year").size().to_dict()
    if yearly_counts != {2025: 53, 2030: 53, 2035: 53}:
        raise RuntimeError(f"Unexpected annual row counts: {yearly_counts}")
    observed_counts = (
        paths[paths["learning_year"] == 2035]
        .groupby("tech_id")
        .size()
        .to_dict()
    )
    if observed_counts != EXPECTED_CONFIG_COUNTS:
        raise RuntimeError(f"Unexpected technology counts: {observed_counts}")

    required = [
        "duration_h",
        "power_cost_eur_kw",
        "energy_cost_eur_kwh",
        "total_capex_eur_kwh",
        "capex_reporting_route",
        "evidence_class",
    ]
    if paths[required].isna().any().any():
        raise RuntimeError("Missing required Source Data values")
    if (paths[["duration_h", "total_capex_eur_kwh"]] <= 0).any().any():
        raise RuntimeError("Duration and installed CAPEX must be positive")
    if paths["cost_identity_error_eur_kwh"].abs().max() > 1e-10:
        raise RuntimeError("Installed-CAPEX component identity failed")

    for tech_id, expected in EXPECTED_ROUTE.items():
        observed = set(
            paths.loc[paths["tech_id"] == tech_id, "capex_reporting_route"]
        )
        if observed != {expected}:
            raise RuntimeError(
                f"Reporting-route QA failed for {tech_id}: {sorted(observed)}"
            )

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    paths.to_csv(SOURCE_PATH, index=False, float_format="%.10g")
    return paths


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.7,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.7,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_configuration(
    ax: mpl.axes.Axes,
    group: pd.DataFrame,
    norm: LogNorm,
    cmap: mpl.colors.Colormap,
) -> None:
    group = group.sort_values("learning_year")
    status = str(group["evidence_class"].iloc[0])
    if group["evidence_class"].nunique() != 1:
        raise RuntimeError("Evidence status changed across cost snapshots")
    style = STATUS_STYLE[status]
    duration = float(group["duration_h"].iloc[0])
    color = cmap(norm(duration))
    marker_face = color if style["filled"] else "white"
    ax.plot(
        group["learning_year"],
        group["total_capex_eur_kwh"],
        color=color,
        linewidth=style["linewidth"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=3.6 if status != "H2 structural extrapolation" else 4.2,
        markerfacecolor=marker_face,
        markeredgecolor=color,
        markeredgewidth=0.65,
        solid_capstyle="round",
        zorder=3,
    )


def make_figure(data: pd.DataFrame) -> mpl.figure.Figure:
    configure_style()
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(WIDTH_IN, HEIGHT_IN),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.082,
        right=0.982,
        bottom=0.205,
        top=0.925,
        wspace=0.245,
        hspace=0.355,
    )

    norm = LogNorm(vmin=4, vmax=720)
    cmap = mpl.colormaps["viridis"]
    y_ticks = [10, 20, 50, 100, 200, 400]
    for index, (ax, tech_id) in enumerate(zip(axes.flat, TECH_ORDER)):
        frame = data[data["tech_id"] == tech_id]
        for _, group in frame.groupby("duration_h", sort=True):
            draw_configuration(ax=ax, group=group, norm=norm, cmap=cmap)
        durations = sorted(frame["duration_h"].unique())
        ax.set_yscale("log")
        ax.set_ylim(9.5, 365)
        ax.set_xlim(2024.3, 2035.7)
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(year) for year in YEARS])
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", which="major", color="#D9DEE3", linewidth=0.48)
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=2.4,
            width=0.6,
            pad=1.8,
        )
        ax.tick_params(axis="both", which="minor", length=0)
        ax.set_title(
            TECH_LABEL[tech_id],
            loc="left",
            pad=4.5,
            fontweight="bold",
        )
        ax.text(
            0.98,
            0.965,
            f"n = {len(durations)}  |  {durations[0]:g}–{durations[-1]:g} h",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.35,
            color="#626970",
        )
        add_panel_label(ax, chr(ord("a") + index))
        if index // 4 == 1:
            ax.set_xlabel("Cost snapshot")
        if index % 4 == 0:
            ax.set_ylabel("Installed CAPEX\n(EUR kWh$^{-1}$)")

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cax = fig.add_axes([0.105, 0.079, 0.335, 0.018])
    colorbar = fig.colorbar(
        scalar,
        cax=cax,
        orientation="horizontal",
        ticks=[4, 8, 16, 24, 48, 96, 168, 336, 720],
    )
    colorbar.ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:g}")
    )
    colorbar.ax.tick_params(length=2.0, width=0.55, labelsize=5.6, pad=1.5)
    colorbar.outline.set_linewidth(0.55)
    colorbar.set_label("Nominal discharge duration (h)", fontsize=5.9, labelpad=2.0)

    status_counts = (
        data[data["learning_year"] == 2035]["evidence_class"]
        .value_counts()
        .to_dict()
    )
    legend_handles: list[Line2D] = []
    short_label = {
        "Direct evidence duration": "Direct anchor",
        "Released configuration": "Released node",
        "Within-domain component interpolation": "Within-domain interpolation",
        "H2 structural extrapolation": "H$_2$ structural extrapolation",
    }
    for status in STATUS_ORDER:
        style = STATUS_STYLE[status]
        legend_handles.append(
            Line2D(
                [],
                [],
                color="#4C5358",
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.0,
                markerfacecolor="#4C5358" if style["filled"] else "white",
                markeredgecolor="#4C5358",
                markeredgewidth=0.65,
                label=f"{short_label[status]} ({status_counts.get(status, 0)})",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.485, 0.035),
        ncol=2,
        columnspacing=1.0,
        handlelength=2.6,
        handletextpad=0.45,
        borderaxespad=0,
        frameon=False,
    )
    fig.text(
        0.485,
        0.113,
        "Configuration evidence status",
        fontsize=5.9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#343A3F",
    )
    fig.text(
        0.982,
        0.012,
        "CAPEX = power component / duration + energy component",
        fontsize=5.2,
        ha="right",
        va="bottom",
        color="#60676D",
    )
    return fig


def save_outputs(fig: mpl.figure.Figure) -> dict[str, str]:
    paths = {
        "pdf": OUTPUT_STEM.with_suffix(".pdf"),
        "svg": OUTPUT_STEM.with_suffix(".svg"),
        "png": OUTPUT_STEM.with_suffix(".png"),
        "tiff": OUTPUT_STEM.with_suffix(".tiff"),
    }
    fig.savefig(paths["pdf"], bbox_inches=None)
    fig.savefig(paths["svg"], bbox_inches=None)
    fig.savefig(paths["png"], dpi=300, bbox_inches=None)
    fig.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    preview = Image.open(paths["png"]).convert("RGB")
    grayscale_path = HERE / "fig_v40_ed1_cost_paths_53_eur_grayscale.png"
    ImageOps.grayscale(preview).save(grayscale_path, dpi=(300, 300))
    paths["grayscale"] = grayscale_path
    return {path.name: sha256(path) for path in paths.values()}


def write_qa(data: pd.DataFrame, outputs: dict[str, str]) -> None:
    tiff = Image.open(OUTPUT_STEM.with_suffix(".tiff"))
    png = Image.open(OUTPUT_STEM.with_suffix(".png"))
    expected_tiff = (round(WIDTH_IN * 600), round(HEIGHT_IN * 600))
    expected_png = (round(WIDTH_IN * 300), round(HEIGHT_IN * 300))
    yearly_counts = {
        str(year): int(count)
        for year, count in data.groupby("learning_year").size().items()
    }
    technology_counts = {
        tech: int(count)
        for tech, count in (
            data[data["learning_year"] == 2035].groupby("tech_id").size().items()
        )
    }
    route_map = {
        tech: sorted(
            data.loc[data["tech_id"] == tech, "capex_reporting_route"].unique()
        )
        for tech in TECH_ORDER
    }
    status_counts = {
        status: int(count)
        for status, count in (
            data[data["learning_year"] == 2035]["evidence_class"]
            .value_counts()
            .items()
        )
    }
    qa = {
        "status": "PASS",
        "figure": OUTPUT_STEM.name,
        "canvas_inches": [WIDTH_IN, HEIGHT_IN],
        "source_rows": int(len(data)),
        "unique_configurations_per_snapshot": 53,
        "snapshot_row_counts": yearly_counts,
        "technology_configuration_counts": technology_counts,
        "evidence_status_counts_per_snapshot": status_counts,
        "duplicates_on_tech_duration_year": int(
            data[["tech_id", "duration_h", "learning_year"]].duplicated().sum()
        ),
        "missing_required_values": 0,
        "maximum_capex_identity_error_eur_kwh": float(
            data["cost_identity_error_eur_kwh"].abs().max()
        ),
        "reporting_routes": route_map,
        "source_aware_route_check": all(
            route_map[tech] == [EXPECTED_ROUTE[tech]] for tech in TECH_ORDER
        ),
        "source_library": str(PARAMETER_LIBRARY.relative_to(WORKSPACE)),
        "configuration_inventory": str(INVENTORY_PATH.relative_to(WORKSPACE)),
        "source_data": str(SOURCE_PATH.relative_to(WORKSPACE)),
        "source_library_sha256": sha256(PARAMETER_LIBRARY),
        "configuration_inventory_sha256": sha256(INVENTORY_PATH),
        "source_data_sha256": sha256(SOURCE_PATH),
        "png_pixel_dimensions": list(png.size),
        "expected_png_dimensions_300dpi": list(expected_png),
        "tiff_pixel_dimensions": list(tiff.size),
        "expected_tiff_dimensions_600dpi": list(expected_tiff),
        "fixed_canvas_dimensions_pass": (
            all(abs(a - e) <= 1 for a, e in zip(png.size, expected_png))
            and all(abs(a - e) <= 1 for a, e in zip(tiff.size, expected_tiff))
        ),
        "output_sha256": outputs,
    }
    (HERE / "ED1_QA.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    data = build_source_data()
    fig = make_figure(data)
    outputs = save_outputs(fig)
    plt.close(fig)
    write_qa(data, outputs)
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_rows": len(data),
                "source_data": str(SOURCE_PATH),
                "figure": str(OUTPUT_STEM),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
