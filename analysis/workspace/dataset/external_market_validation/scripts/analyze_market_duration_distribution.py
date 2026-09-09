#!/usr/bin/env python3
"""Reproducible audit of observed storage-duration deployment distributions.

The script uses two official project inventories:

1. U.S. EIA Form EIA-860 final 2024 energy-storage schedule.
2. AEMO April 2026 NEM Generation Information workbook.

It produces cleaned project/unit tables, requested duration-bin summaries,
threshold-sensitivity diagnostics, checksums, and two publication-grade figures.
The analysis is descriptive: it does not identify why the observed deployment
distribution arose and does not attribute it to scarcity-event clustering.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
DERIVED = ROOT / "derived"
FIGURES = ROOT / "figures"

EIA_ZIP = RAW / "eia860_2024_final.zip"
EIA_XLSX = RAW / "eia860_2024_final" / "3_4_Energy_Storage_Y2024.xlsx"
AEMO_XLSX = RAW / "aemo_nem_generation_information_apr_2026.xlsx"

STANDARD_LABELS = ["0–4 h", ">4–8 h", ">8–24 h", ">24 h"]
STANDARD_EDGES = [0.0, 4.0, 8.0, 24.0, np.inf]
STATUS_ORDER = ["Operating", "Planned"]
STATUS_COLORS = {"Operating": "#0F4D92", "Planned": "#D8843F"}

EIA_TECH_CODES = {
    "LIB": "Lithium-ion battery",
    "FLB": "Flow battery",
    "MAB": "Metal-air battery",
    "NIB": "Nickel-based battery",
    "NAB": "Sodium-based battery",
    "PBB": "Lead-acid battery",
    "ECC": "Electrochemical capacitor",
    "OTH": "Other battery/storage",
}


def configure_matplotlib() -> None:
    """Apply a compact Nature-style plotting theme."""
    # Mandatory editable-SVG rules from the plotting contract.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 7,
        }
    )


def ensure_dirs() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_header_row(path: Path, sheet_name: str, token: str) -> int:
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=12)
    for idx, row in preview.iterrows():
        values = row.astype(str).str.strip()
        if values.eq(token).any():
            return int(idx)
    raise ValueError(f"Could not find header token {token!r} in {path} / {sheet_name}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def assign_standard_bin(duration_h: pd.Series) -> pd.Categorical:
    return pd.cut(
        duration_h,
        bins=STANDARD_EDGES,
        labels=STANDARD_LABELS,
        right=True,
        include_lowest=False,
        ordered=True,
    )


def eia_technology_group(row: pd.Series) -> str:
    technology = str(row.get("Technology", "")).strip()
    code = str(row.get("Storage Technology 1", "")).strip()
    if "Compressed Air" in technology:
        return "Compressed-air energy storage"
    if "Flywheel" in technology:
        return "Flywheel"
    if "Solar Thermal" in technology:
        return "Solar thermal with storage"
    if technology == "Batteries":
        return EIA_TECH_CODES.get(code, "Battery, unspecified")
    return technology or "Unspecified"


def read_eia_units() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned EIA unit records and an exclusion audit."""
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    active_planned = {"P", "L", "T", "U", "V", "TS", "OT"}

    for sheet, status_group in [("Operable", "Operating"), ("Proposed", "Planned")]:
        header = find_header_row(EIA_XLSX, sheet, "Utility ID")
        raw = pd.read_excel(EIA_XLSX, sheet_name=sheet, header=header)
        raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
        n_input = len(raw)

        if sheet == "Operable":
            selected = raw.loc[raw["Status"].astype(str).str.strip().eq("OP")].copy()
        else:
            selected = raw.loc[
                raw["Status"].astype(str).str.strip().isin(active_planned)
            ].copy()

        selected["power_mw"] = numeric(selected["Nameplate Capacity (MW)"])
        selected["energy_mwh"] = numeric(
            selected["Nameplate Energy Capacity (MWh)"]
        )
        valid = (
            selected["power_mw"].gt(0)
            & selected["energy_mwh"].gt(0)
            & np.isfinite(selected["power_mw"])
            & np.isfinite(selected["energy_mwh"])
        )

        audit_rows.append(
            {
                "dataset": "EIA-860 final 2024",
                "sheet": sheet,
                "input_rows": n_input,
                "status_selected_rows": len(selected),
                "valid_power_energy_rows": int(valid.sum()),
                "excluded_missing_or_nonpositive_power_energy": int((~valid).sum()),
            }
        )

        selected = selected.loc[valid].copy()
        selected["dataset"] = "EIA-860 final 2024"
        selected["market"] = "United States"
        selected["status_group"] = status_group
        selected["source_status"] = selected["Status"].astype(str).str.strip()
        selected["unit_id"] = (
            selected["Plant Code"].astype("Int64").astype(str)
            + "::"
            + selected["Generator ID"].astype(str).str.strip()
        )
        selected["project_proxy_id"] = selected["Plant Code"].astype(
            "Int64"
        ).astype(str)
        selected["project_name"] = selected["Plant Name"].astype(str).str.strip()
        selected["region"] = selected["State"].astype(str).str.strip()
        selected["duration_h"] = selected["energy_mwh"] / selected["power_mw"]
        selected["duration_bin"] = assign_standard_bin(selected["duration_h"])
        selected["technology_group"] = selected.apply(
            eia_technology_group, axis=1
        )
        selected["technology_scope"] = np.where(
            selected["Technology"].astype(str).eq("Batteries"),
            "Battery only",
            "Other storage",
        )
        selected["unit_count"] = 1

        keep = [
            "dataset",
            "market",
            "status_group",
            "source_status",
            "unit_id",
            "project_proxy_id",
            "project_name",
            "region",
            "technology_group",
            "technology_scope",
            "power_mw",
            "energy_mwh",
            "duration_h",
            "duration_bin",
            "unit_count",
        ]
        frames.append(selected[keep])

    units = pd.concat(frames, ignore_index=True)
    units["duration_bin"] = assign_standard_bin(units["duration_h"])
    audit = pd.DataFrame(audit_rows)
    return units, audit


def aggregate_eia_sites(units: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        units.groupby(
            [
                "dataset",
                "market",
                "status_group",
                "project_proxy_id",
                "project_name",
                "region",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            power_mw=("power_mw", "sum"),
            energy_mwh=("energy_mwh", "sum"),
            unit_count=("unit_id", "nunique"),
            technology_group=("technology_group", lambda x: "; ".join(sorted(set(x)))),
            battery_units=(
                "technology_scope",
                lambda x: int(np.sum(np.asarray(x) == "Battery only")),
            ),
        )
        .copy()
    )
    grouped["duration_h"] = grouped["energy_mwh"] / grouped["power_mw"]
    grouped["duration_bin"] = assign_standard_bin(grouped["duration_h"])
    grouped["technology_scope"] = np.where(
        grouped["battery_units"].eq(grouped["unit_count"]),
        "Battery only",
        "All detailed-storage records",
    )
    grouped["project_proxy_count"] = 1
    return grouped


def read_aemo_sites() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean AEMO storage records and aggregate them to site-stage projects."""
    header = find_header_row(AEMO_XLSX, "Generator Information", "AEMO Survey ID")
    raw = pd.read_excel(AEMO_XLSX, sheet_name="Generator Information", header=header)
    raw["row_power_mw"] = numeric(raw["Agg Nameplate Capacity (MW AC)"])
    raw["row_energy_mwh"] = numeric(raw["Agg Nameplate Storage Capacity (MWh)"])
    selected = raw.loc[
        raw["row_power_mw"].gt(0) & raw["row_energy_mwh"].gt(0)
    ].copy()

    audit = pd.DataFrame(
        [
            {
                "dataset": "AEMO NEM April 2026",
                "sheet": "Generator Information",
                "input_rows": len(raw),
                "status_selected_rows": len(raw),
                "valid_power_energy_rows": len(selected),
                "excluded_missing_or_nonpositive_power_energy": len(raw)
                - len(selected),
            }
        ]
    )

    selected["source_status"] = (
        selected["Commitment Status"].astype(str).str.strip()
    )
    selected["technology_group"] = np.select(
        [
            selected["Technology Detail"].astype(str).eq("Pumped Hydro"),
            selected["Technology Detail"].astype(str).eq("Compressed Air"),
            selected["Technology Detail"].astype(str).eq("Lithium-ion"),
        ],
        ["Pumped hydro", "Compressed-air energy storage", "Lithium-ion battery"],
        default=selected["Technology Type"].astype(str).str.strip(),
    )

    def aggregate_site_stage(group: pd.DataFrame) -> pd.Series:
        power = float(group["row_power_mw"].sum())
        energy_parts: list[float] = []
        for tech, tech_group in group.groupby("technology_group", dropna=False):
            values = tech_group.loc[
                tech_group["row_energy_mwh"].notna(), "row_energy_mwh"
            ].to_numpy(float)
            values = values[values > 0]
            if tech == "Pumped hydro":
                # AEMO repeats a shared reservoir capacity across turbine rows.
                energy_parts.append(float(np.unique(values).sum()))
            else:
                energy_parts.append(float(values.sum()))
        energy = float(np.sum(energy_parts))
        return pd.Series(
            {
                "project_name": str(group["Site Name"].iloc[0]).strip(),
                "region": str(group["Region"].iloc[0]).strip(),
                "power_mw": power,
                "energy_mwh": energy,
                "unit_count": int(group["Gen Info Unit ID"].nunique()),
                "technology_group": "; ".join(
                    sorted(
                        set(
                            group.loc[
                                group["technology_group"].notna(),
                                "technology_group",
                            ].astype(str)
                        )
                    )
                ),
                "battery_units": int(
                    group["Technology Type"].astype(str).eq("Battery Storage").sum()
                ),
                "row_count": len(group),
            }
        )

    site = (
        selected.groupby(
            ["AEMO Survey ID", "source_status"],
            dropna=False,
            observed=True,
        )
        .apply(aggregate_site_stage, include_groups=False)
        .reset_index()
    )
    site = site.rename(columns={"AEMO Survey ID": "aemo_survey_id"})
    site["dataset"] = "AEMO NEM April 2026"
    site["market"] = "Australia NEM"
    site["project_proxy_id"] = (
        site["aemo_survey_id"].astype("Int64").astype(str)
        + "::"
        + site["source_status"].astype(str)
    )
    site["status_group"] = np.where(
        site["source_status"].eq("In Service"), "Operating", "Planned"
    )
    site["pipeline_stage_group"] = np.select(
        [
            site["source_status"].eq("In Service"),
            site["source_status"].eq("In Commissioning"),
            site["source_status"].isin(["Committed", "Committed*", "Anticipated"]),
            site["source_status"].eq("Publicly Announced"),
        ],
        [
            "In service",
            "In commissioning",
            "Committed or anticipated",
            "Publicly announced",
        ],
        default="Other",
    )
    site["duration_h"] = site["energy_mwh"] / site["power_mw"]
    site["duration_bin"] = assign_standard_bin(site["duration_h"])
    site["data_quality_flag"] = np.where(
        site["duration_h"].gt(24)
        & ~site["technology_group"].str.contains("Pumped hydro", na=False),
        "Non-pumped-storage duration >24 h; manual source-record review recommended",
        "",
    )
    site["technology_scope"] = np.where(
        site["technology_group"].eq("Lithium-ion battery"),
        "Battery only",
        "All storage",
    )
    site["project_proxy_count"] = 1

    keep = [
        "dataset",
        "market",
        "status_group",
        "pipeline_stage_group",
        "source_status",
        "project_proxy_id",
        "project_name",
        "region",
        "technology_group",
        "technology_scope",
        "power_mw",
        "energy_mwh",
        "duration_h",
        "duration_bin",
        "data_quality_flag",
        "unit_count",
        "row_count",
    ]
    return site[keep], audit


def summary_for_records(
    records: pd.DataFrame,
    dataset: str,
    aggregation_level: str,
    scope_name: str,
    scope_mask: pd.Series | np.ndarray,
) -> pd.DataFrame:
    subset = records.loc[np.asarray(scope_mask)].copy()
    subset["duration_bin"] = assign_standard_bin(subset["duration_h"])
    rows: list[pd.DataFrame] = []
    for status in STATUS_ORDER:
        part = subset.loc[subset["status_group"].eq(status)].copy()
        grouped = (
            part.groupby("duration_bin", observed=False)
            .agg(
                project_proxy_count=("project_proxy_id", "nunique"),
                unit_count=("unit_count", "sum"),
                power_mw=("power_mw", "sum"),
                energy_mwh=("energy_mwh", "sum"),
            )
            .reindex(STANDARD_LABELS, fill_value=0)
            .reset_index()
        )
        grouped["dataset"] = dataset
        grouped["aggregation_level"] = aggregation_level
        grouped["technology_scope"] = scope_name
        grouped["status_group"] = status
        for metric in [
            "project_proxy_count",
            "unit_count",
            "power_mw",
            "energy_mwh",
        ]:
            total = grouped[metric].sum()
            grouped[f"{metric}_share_pct"] = np.where(
                total > 0, grouped[metric] / total * 100.0, np.nan
            )
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def make_all_summaries(
    eia_units: pd.DataFrame, eia_sites: pd.DataFrame, aemo_sites: pd.DataFrame
) -> pd.DataFrame:
    summaries = [
        summary_for_records(
            eia_units,
            "EIA-860 final 2024",
            "Generator unit",
            "All detailed-storage records",
            np.ones(len(eia_units), dtype=bool),
        ),
        summary_for_records(
            eia_units,
            "EIA-860 final 2024",
            "Generator unit",
            "Battery only",
            eia_units["technology_scope"].eq("Battery only"),
        ),
        summary_for_records(
            eia_sites,
            "EIA-860 final 2024",
            "Plant-site project proxy",
            "All detailed-storage records",
            np.ones(len(eia_sites), dtype=bool),
        ),
        summary_for_records(
            eia_sites,
            "EIA-860 final 2024",
            "Plant-site project proxy",
            "Battery only",
            eia_sites["technology_scope"].eq("Battery only"),
        ),
        summary_for_records(
            aemo_sites,
            "AEMO NEM April 2026",
            "Site-stage project proxy",
            "All storage",
            np.ones(len(aemo_sites), dtype=bool),
        ),
        summary_for_records(
            aemo_sites,
            "AEMO NEM April 2026",
            "Site-stage project proxy",
            "Battery only",
            aemo_sites["technology_scope"].eq("Battery only"),
        ),
    ]
    out = pd.concat(summaries, ignore_index=True)
    out["duration_bin"] = pd.Categorical(
        out["duration_bin"],
        categories=STANDARD_LABELS,
        ordered=True,
    )
    return out.sort_values(
        [
            "dataset",
            "aggregation_level",
            "technology_scope",
            "status_group",
            "duration_bin",
        ]
    )


def threshold_sensitivity(
    records: pd.DataFrame,
    dataset: str,
    aggregation_level: str,
    scope_name: str,
    scope_mask: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Test a strict three-band valley over alternative duration thresholds.

    A strict observational valley requires the middle-band deployment to be
    lower than both the short- and long-duration bands.  This is deliberately
    stronger than merely finding an empty narrow sub-bin.
    """
    subset = records.loc[np.asarray(scope_mask)].copy()
    metrics = {
        "project_proxy_count": None,
        "power_mw": "power_mw",
        "energy_mwh": "energy_mwh",
    }
    rows: list[dict[str, object]] = []
    for status in STATUS_ORDER:
        part = subset.loc[subset["status_group"].eq(status)].copy()
        for short_cut in [2.0, 4.0, 6.0]:
            for long_cut in [8.0, 12.0, 16.0, 24.0, 32.0, 48.0]:
                if long_cut <= short_cut:
                    continue
                masks = {
                    "short": part["duration_h"].le(short_cut),
                    "middle": part["duration_h"].gt(short_cut)
                    & part["duration_h"].le(long_cut),
                    "long": part["duration_h"].gt(long_cut),
                }
                for metric_name, value_col in metrics.items():
                    values: dict[str, float] = {}
                    for band, mask in masks.items():
                        if metric_name == "project_proxy_count":
                            values[band] = float(
                                part.loc[mask, "project_proxy_id"].nunique()
                            )
                        else:
                            values[band] = float(part.loc[mask, value_col].sum())
                    flank_min = min(values["short"], values["long"])
                    ratio = (
                        values["middle"] / flank_min if flank_min > 0 else np.nan
                    )
                    strict = bool(
                        flank_min > 0
                        and values["middle"] < values["short"]
                        and values["middle"] < values["long"]
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "aggregation_level": aggregation_level,
                            "technology_scope": scope_name,
                            "status_group": status,
                            "short_cut_h": short_cut,
                            "long_cut_h": long_cut,
                            "metric": metric_name,
                            "short_value": values["short"],
                            "middle_value": values["middle"],
                            "long_value": values["long"],
                            "middle_to_smaller_flank_ratio": ratio,
                            "strict_valley": strict,
                        }
                    )
    return pd.DataFrame(rows)


def local_trough_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    key_cols = [
        "dataset",
        "aggregation_level",
        "technology_scope",
        "status_group",
    ]
    metrics = ["project_proxy_count", "power_mw", "energy_mwh"]
    for keys, group in summary.groupby(key_cols, observed=True):
        indexed = group.set_index("duration_bin")
        for metric in metrics:
            b4_8 = float(indexed.loc[">4–8 h", metric])
            b8_24 = float(indexed.loc[">8–24 h", metric])
            b24 = float(indexed.loc[">24 h", metric])
            adjacent_min = min(b4_8, b24)
            local_ratio = b8_24 / adjacent_min if adjacent_min > 0 else np.nan
            combined_middle = b4_8 + b8_24
            short = float(indexed.loc["0–4 h", metric])
            long = b24
            flank_min = min(short, long)
            combined_ratio = (
                combined_middle / flank_min if flank_min > 0 else np.nan
            )
            rows.append(
                {
                    **dict(zip(key_cols, keys)),
                    "metric": metric,
                    "value_4_8": b4_8,
                    "value_8_24": b8_24,
                    "value_gt24": b24,
                    "local_8_24_trough": bool(
                        adjacent_min > 0 and b8_24 < b4_8 and b8_24 < b24
                    ),
                    "local_8_24_to_smaller_adjacent_ratio": local_ratio,
                    "combined_middle_4_24": combined_middle,
                    "short_0_4": short,
                    "long_gt24": long,
                    "strict_combined_4_24_valley": bool(
                        flank_min > 0
                        and combined_middle < short
                        and combined_middle < long
                    ),
                    "combined_middle_to_smaller_flank_ratio": combined_ratio,
                }
            )
    return pd.DataFrame(rows)


def equal_width_summary(records: pd.DataFrame, dataset: str) -> pd.DataFrame:
    labels = [
        "0–4 h",
        ">4–8 h",
        ">8–12 h",
        ">12–16 h",
        ">16–20 h",
        ">20–24 h",
        ">24 h",
    ]
    edges = [0, 4, 8, 12, 16, 20, 24, np.inf]
    frame = records.copy()
    frame["equal_width_bin"] = pd.cut(
        frame["duration_h"],
        bins=edges,
        labels=labels,
        include_lowest=False,
        right=True,
        ordered=True,
    )
    out = (
        frame.groupby(["status_group", "equal_width_bin"], observed=False)
        .agg(
            project_proxy_count=("project_proxy_id", "nunique"),
            power_mw=("power_mw", "sum"),
            energy_mwh=("energy_mwh", "sum"),
        )
        .reset_index()
    )
    out["dataset"] = dataset
    return out[
        [
            "dataset",
            "status_group",
            "equal_width_bin",
            "project_proxy_count",
            "power_mw",
            "energy_mwh",
        ]
    ]


def technology_summary(records: pd.DataFrame, dataset: str) -> pd.DataFrame:
    frame = records.copy()
    frame["duration_bin"] = assign_standard_bin(frame["duration_h"])
    out = (
        frame.groupby(
            ["status_group", "duration_bin", "technology_group"],
            observed=False,
        )
        .agg(
            project_proxy_count=("project_proxy_id", "nunique"),
            power_mw=("power_mw", "sum"),
            energy_mwh=("energy_mwh", "sum"),
        )
        .reset_index()
    )
    out["dataset"] = dataset
    return out


def save_figure_bundle(fig: plt.Figure, stem: str) -> None:
    fig.savefig(
        FIGURES / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURES / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURES / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURES / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )


def plot_deployment_profiles(summary: pd.DataFrame) -> None:
    """Six-panel inventory profile; zeros remain defined through log10(x+1)."""
    configure_matplotlib()
    datasets = [
        (
            "EIA-860 final 2024",
            "Plant-site project proxy",
            "All detailed-storage records",
            "United States · EIA-860",
        ),
        (
            "AEMO NEM April 2026",
            "Site-stage project proxy",
            "All storage",
            "Australia NEM · AEMO",
        ),
    ]
    metrics = [
        ("project_proxy_count", "Project-site proxies"),
        ("power_mw", "Power (MW)"),
        ("energy_mwh", "Energy (MWh)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55), sharex=True)
    x = np.arange(len(STANDARD_LABELS))
    labels = iter("abcdef")

    for row, (dataset, level, scope, row_title) in enumerate(datasets):
        subset = summary.loc[
            summary["dataset"].eq(dataset)
            & summary["aggregation_level"].eq(level)
            & summary["technology_scope"].eq(scope)
        ].copy()
        for col, (metric, metric_title) in enumerate(metrics):
            ax = axes[row, col]
            ax.axvspan(0.5, 2.5, color="#ECE8E2", alpha=0.65, zorder=0)
            for status in STATUS_ORDER:
                values = (
                    subset.loc[subset["status_group"].eq(status)]
                    .set_index("duration_bin")
                    .reindex(STANDARD_LABELS)[metric]
                    .fillna(0)
                    .to_numpy(float)
                )
                transformed = np.log10(values + 1.0)
                ax.plot(
                    x,
                    transformed,
                    color=STATUS_COLORS[status],
                    marker="o",
                    ms=4.0,
                    lw=1.4,
                    label=status,
                    zorder=3,
                )
                for xpos, raw, yval in zip(x, values, transformed):
                    if raw == 0:
                        ax.plot(
                            xpos,
                            0,
                            marker="x",
                            ms=4,
                            mew=0.8,
                            color=STATUS_COLORS[status],
                            zorder=4,
                        )
            ax.set_xticks(x)
            ax.set_xticklabels(STANDARD_LABELS, rotation=22, ha="right")
            ax.set_ylabel(f"log$_{{10}}$({metric_title} + 1)")
            ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.8)
            ax.text(
                -0.13,
                1.04,
                next(labels),
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=9,
                va="bottom",
            )
            if row == 0:
                ax.set_title(metric_title, pad=7)
            if col == 0:
                ax.text(
                    0.98,
                    0.97,
                    row_title,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    color="#3E3E3E",
                    fontsize=6.8,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                    },
                )

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.70, 1.015),
        ncol=2,
        handlelength=2.0,
    )
    fig.text(
        0.5,
        0.01,
        "Duration bin (nameplate energy ÷ nameplate discharge power)",
        ha="center",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.18, wspace=0.38, hspace=0.45)
    save_figure_bundle(fig, "market_storage_duration_inventory_profiles")
    plt.close(fig)


def plot_valley_sensitivity(sensitivity: pd.DataFrame) -> None:
    """Show whether a strict middle-band valley survives threshold choices."""
    configure_matplotlib()
    planned = sensitivity.loc[
        sensitivity["status_group"].eq("Planned")
        & sensitivity["short_cut_h"].eq(4.0)
        & sensitivity["technology_scope"].isin(
            ["All detailed-storage records", "All storage"]
        )
        & sensitivity["aggregation_level"].isin(
            ["Plant-site project proxy", "Site-stage project proxy"]
        )
    ].copy()

    metric_titles = {
        "project_proxy_count": "Project-site proxies",
        "power_mw": "Power",
        "energy_mwh": "Energy",
    }
    dataset_colors = {
        "EIA-860 final 2024": "#0F4D92",
        "AEMO NEM April 2026": "#2B8C7F",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharex=True)
    labels = iter("abc")

    for ax, metric in zip(axes, metric_titles):
        part = planned.loc[planned["metric"].eq(metric)].copy()
        for dataset in ["EIA-860 final 2024", "AEMO NEM April 2026"]:
            curve = part.loc[part["dataset"].eq(dataset)].sort_values("long_cut_h")
            y = curve["middle_to_smaller_flank_ratio"].to_numpy(float)
            y_plot = np.log10(np.clip(y, 1e-3, None))
            ax.plot(
                curve["long_cut_h"],
                y_plot,
                marker="o",
                ms=4,
                lw=1.4,
                color=dataset_colors[dataset],
                label=dataset,
            )
        ax.axhline(0, color="#B64342", lw=0.9, ls="--")
        ax.fill_between(
            [7, 49],
            -3,
            0,
            color="#F6CFCB",
            alpha=0.4,
            zorder=0,
        )
        ax.set_xlim(7, 49)
        ax.set_ylim(-3, None)
        ax.set_xticks([8, 12, 16, 24, 32, 48])
        ax.set_xlabel("Long-duration cutoff (h)")
        ax.set_ylabel("log$_{10}$(middle / smaller flank)")
        ax.set_title(metric_titles[metric], pad=7)
        ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.8)
        ax.text(
            -0.13,
            1.04,
            next(labels),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )
    axes[0].text(
        8.5,
        -0.15,
        "strict valley",
        color="#9B3A39",
        fontsize=6.8,
        va="top",
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.66, 1.02),
        ncol=2,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.22, wspace=0.38)
    save_figure_bundle(fig, "market_storage_duration_valley_sensitivity")
    plt.close(fig)


def write_source_manifest() -> None:
    rows = [
        {
            "source": "U.S. EIA Form EIA-860 final 2024",
            "analysis_role": "Primary U.S. operating and proposed detailed-storage inventory",
            "official_landing_page": "https://www.eia.gov/electricity/data/eia860/",
            "download_url": "https://www.eia.gov/electricity/data/eia860/xls/eia8602024.zip",
            "local_file": str(EIA_ZIP.relative_to(ROOT)),
            "sha256": sha256(EIA_ZIP),
        },
        {
            "source": "AEMO NEM Generation Information April 2026",
            "analysis_role": "Independent market replication including batteries and pumped hydro",
            "official_landing_page": "https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/nem-forecasting-and-planning/forecasting-and-planning-data/generation-information",
            "download_url": "https://www.aemo.com.au/-/media/files/electricity/nem/planning_and_forecasting/generation_information/2026/nem-generation-information-apr-2026.xlsx",
            "local_file": str(AEMO_XLSX.relative_to(ROOT)),
            "sha256": sha256(AEMO_XLSX),
        },
    ]
    pd.DataFrame(rows).to_csv(DERIVED / "source_manifest_sha256.csv", index=False)


def main() -> None:
    ensure_dirs()
    for required in [EIA_ZIP, EIA_XLSX, AEMO_XLSX]:
        if not required.exists():
            raise FileNotFoundError(required)

    eia_units, eia_audit = read_eia_units()
    eia_sites = aggregate_eia_sites(eia_units)
    aemo_sites, aemo_audit = read_aemo_sites()

    eia_units.to_csv(DERIVED / "eia860_2024_storage_units_clean.csv", index=False)
    eia_sites.to_csv(
        DERIVED / "eia860_2024_storage_site_projects_clean.csv", index=False
    )
    aemo_sites.to_csv(
        DERIVED / "aemo_apr2026_storage_site_projects_clean.csv", index=False
    )
    pd.concat([eia_audit, aemo_audit], ignore_index=True).to_csv(
        DERIVED / "record_exclusion_audit.csv", index=False
    )

    summary = make_all_summaries(eia_units, eia_sites, aemo_sites)
    summary.to_csv(DERIVED / "duration_bin_summary.csv", index=False)

    sensitivity = pd.concat(
        [
            threshold_sensitivity(
                eia_units,
                "EIA-860 final 2024",
                "Generator unit",
                "All detailed-storage records",
                np.ones(len(eia_units), dtype=bool),
            ),
            threshold_sensitivity(
                eia_sites,
                "EIA-860 final 2024",
                "Plant-site project proxy",
                "All detailed-storage records",
                np.ones(len(eia_sites), dtype=bool),
            ),
            threshold_sensitivity(
                eia_sites,
                "EIA-860 final 2024",
                "Plant-site project proxy",
                "Battery only",
                eia_sites["technology_scope"].eq("Battery only"),
            ),
            threshold_sensitivity(
                aemo_sites,
                "AEMO NEM April 2026",
                "Site-stage project proxy",
                "All storage",
                np.ones(len(aemo_sites), dtype=bool),
            ),
            threshold_sensitivity(
                aemo_sites,
                "AEMO NEM April 2026",
                "Site-stage project proxy",
                "Battery only",
                aemo_sites["technology_scope"].eq("Battery only"),
            ),
        ],
        ignore_index=True,
    )
    sensitivity.to_csv(
        DERIVED / "three_band_threshold_sensitivity.csv", index=False
    )

    local = local_trough_diagnostics(summary)
    local.to_csv(DERIVED / "standard_bin_valley_diagnostics.csv", index=False)

    sensitivity.assign(
        eligible_valley_test=sensitivity[
            "middle_to_smaller_flank_ratio"
        ].notna()
    ).groupby(
        [
            "dataset",
            "aggregation_level",
            "technology_scope",
            "status_group",
            "metric",
        ],
        observed=True,
        as_index=False,
    ).agg(
        threshold_combinations=("strict_valley", "size"),
        eligible_threshold_combinations=("eligible_valley_test", "sum"),
        strict_valley_combinations=("strict_valley", "sum"),
    ).assign(
        strict_valley_share_of_eligible=lambda x: np.where(
            x["eligible_threshold_combinations"].gt(0),
            x["strict_valley_combinations"]
            / x["eligible_threshold_combinations"],
            np.nan,
        )
    ).to_csv(DERIVED / "valley_threshold_robustness_summary.csv", index=False)

    flagged = aemo_sites.loc[aemo_sites["data_quality_flag"].ne("")].copy()
    flagged.to_csv(DERIVED / "aemo_duration_data_quality_flags.csv", index=False)
    summary_for_records(
        aemo_sites.loc[aemo_sites["data_quality_flag"].eq("")],
        "AEMO NEM April 2026",
        "Site-stage project proxy",
        "All storage excluding flagged non-pumped >24 h records",
        np.ones(int(aemo_sites["data_quality_flag"].eq("").sum()), dtype=bool),
    ).to_csv(
        DERIVED / "aemo_standard_bin_summary_excluding_flagged_record.csv",
        index=False,
    )

    pd.concat(
        [
            equal_width_summary(eia_sites, "EIA-860 final 2024"),
            equal_width_summary(aemo_sites, "AEMO NEM April 2026"),
        ],
        ignore_index=True,
    ).to_csv(DERIVED / "equal_width_4h_bin_summary.csv", index=False)

    pd.concat(
        [
            technology_summary(eia_sites, "EIA-860 final 2024"),
            technology_summary(aemo_sites, "AEMO NEM April 2026"),
        ],
        ignore_index=True,
    ).to_csv(DERIVED / "technology_by_duration_bin_summary.csv", index=False)

    write_source_manifest()
    plot_deployment_profiles(summary)
    plot_valley_sensitivity(sensitivity)

    print("Outputs written to", ROOT)
    print("EIA units:", len(eia_units), "EIA site projects:", len(eia_sites))
    print("AEMO site-stage projects:", len(aemo_sites))


if __name__ == "__main__":
    main()
