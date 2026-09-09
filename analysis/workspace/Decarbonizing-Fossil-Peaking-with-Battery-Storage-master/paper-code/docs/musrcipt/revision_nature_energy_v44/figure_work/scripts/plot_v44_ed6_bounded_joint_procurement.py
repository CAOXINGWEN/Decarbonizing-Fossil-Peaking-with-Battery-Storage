#!/usr/bin/env python3
"""Build v44 Extended Data Figure 6 from exact bounded-power choices.

The script is deliberately evidence-gated.  It renders only after the
cross-year bounded-power QA record reports ``status=pass`` and after all
source tables satisfy the registered 15-country, 132-country-year contract.
No fallback, partial, simulated or fixed-power values are accepted.

Figure contract
---------------
Core claim
    Under the exact residual-peak system-cost account, when the model jointly
    chooses no tail, an additional 4 h LFP tail, or one of 49 LDES
    configurations together with 0--100% tail power, clustering produces sparse
    but non-zero portfolio-identity, duration and power changes, while tail
    participation does not change in the three registered cost years.
Evidence chain
    a, winning portfolio composition by cost year and chronology;
    b, paired change rates with country-block bootstrap intervals;
    c, dispersed-to-clustered selected-duration transitions;
    d, the full country-year distribution of selected-power changes.
Archetype
    Asymmetric quantitative grid, inheriting the v43 ED6 visual system.
Export contract
    Nature double-column width, editable PDF/SVG, 300-dpi PNG, 600-dpi TIFF,
    grayscale preview and panel-specific Source Data.
"""

from __future__ import annotations

import hashlib
import json
import sys
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
V44 = FIGURE_WORK.parent
BOUNDED = V44 / "joint_choice" / "bounded_s20_highs"

QA_FILE = BOUNDED / "BOUNDED_ALL_YEAR_FINAL_QA.json"
CHOICES_FILE = BOUNDED / "bounded_joint_choices_all_years.csv"
PAIRS_FILE = BOUNDED / "bounded_joint_choice_pairs_all_years.csv"
SUMMARY_FILE = BOUNDED / "bounded_joint_choice_summary_all_years.csv"

OUT_DIR = FIGURE_WORK / "extended_data"
SOURCE_DIR = FIGURE_WORK / "source_data"
MANIFEST_DIR = FIGURE_WORK / "manifests"
STEM = "fig_v44_ed6_bounded_joint_procurement"

YEARS = (2025, 2030, 2035)
CHRONOLOGIES = ("dispersed", "clustered")
N_COUNTRY_YEARS = 132
EXPECTED_CHOICES = len(YEARS) * len(CHRONOLOGIES) * N_COUNTRY_YEARS
EXPECTED_PAIRS = len(YEARS) * N_COUNTRY_YEARS

WIDTH = 182.88 / 25.4
HEIGHT = 4.80
PNG_DPI = 300
TIFF_DPI = 600

# Frozen v43 visual language.
INK = "#233746"
MUTED = "#697980"
GRID = "#DFE7EA"
PALE = "#F4F7F7"
DISPERSED = "#35618D"
CLUSTERED = "#D8662E"
RECHARGE = "#158F83"
PURPLE = "#7056A3"
GOLD = "#D39B25"
RED = "#BA4A52"

NO_TAIL = "#CFD7DA"
LFP = DISPERSED
LDES = RECHARGE
YEAR_COLOUR = {2025: GOLD, 2030: DISPERSED, 2035: PURPLE}
YEAR_MARKER = {2025: "o", 2030: "s", 2035: "^"}


class EvidencePending(RuntimeError):
    """Raised when exact bounded-power outputs have not passed QA yet."""


def configure() -> None:
    """Match v43 ED6 typography, line weights and export settings."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.labelsize": 6.6,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.55,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.2,
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
            "savefig.dpi": TIFF_DPI,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def primary_identity_flip_counts(qa: dict) -> dict[str, int]:
    """Read the exact system-cost count map, with legacy compatibility."""
    if "residual_peak_system_cost_flip_counts" in qa:
        counts = qa["residual_peak_system_cost_flip_counts"]
        return {str(year): int(counts[str(year)]["identity_flip"]) for year in YEARS}
    if "identity_flip_counts" in qa:
        counts = qa["identity_flip_counts"]
        return {str(year): int(counts[str(year)]) for year in YEARS}
    raise RuntimeError("Final QA does not expose exact residual-peak identity counts")


def load_evidence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load only a complete, QA-passed cross-year bounded-power result."""
    if not QA_FILE.exists():
        raise EvidencePending(f"waiting for {QA_FILE.name}")
    qa = json.loads(QA_FILE.read_text())
    if qa.get("status") != "pass" or not qa.get("all_year_QA_passed", False):
        raise EvidencePending("bounded-power all-year QA has not passed")
    missing = [path for path in (CHOICES_FILE, PAIRS_FILE, SUMMARY_FILE) if not path.exists()]
    if missing:
        raise EvidencePending("QA exists but final source tables are incomplete: " + ", ".join(path.name for path in missing))

    choices = pd.read_csv(CHOICES_FILE)
    pairs = pd.read_csv(PAIRS_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    required_choice = {
        "learning_year", "country", "year", "chronology", "decision_id",
        "decision_tech", "decision_duration_h", "tail_active",
        "tail_power_share", "system_cost_eur", "selection_account",
    }
    required_pair = {
        "learning_year", "country", "year", "identity_flip",
        "technology_flip", "duration_flip", "participation_flip",
        "material_power_change", "dispersed_decision_duration_h",
        "clustered_decision_duration_h", "dispersed_tail_power_share",
        "clustered_tail_power_share", "delta_tail_power_share",
        "dispersed_selection_account", "clustered_selection_account",
    }
    required_summary = {
        "learning_year", "account", "metric", "count", "denominator", "rate",
        "country_block_bootstrap_ci_low", "country_block_bootstrap_ci_high",
    }
    for label, frame, required in [
        ("choices", choices, required_choice),
        ("pairs", pairs, required_pair),
        ("summary", summary, required_summary),
    ]:
        absent = sorted(required.difference(frame.columns))
        if absent:
            raise RuntimeError(f"{label} table is missing columns: {absent}")

    if len(choices) != EXPECTED_CHOICES or len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_CHOICES} choices and {EXPECTED_PAIRS} pairs; "
            f"found {len(choices)} and {len(pairs)}"
        )
    if set(choices["learning_year"].astype(int)) != set(YEARS):
        raise RuntimeError("Choice table does not contain exactly the three registered cost years")
    if set(choices["chronology"].astype(str)) != set(CHRONOLOGIES):
        raise RuntimeError("Choice table chronology labels violate the registered paired design")
    if not choices["selection_account"].eq("residual_peak_system_cost").all():
        raise RuntimeError("Primary bounded choices mix in a non-system-cost account")
    if choices.duplicated(["learning_year", "country", "year", "chronology"]).any():
        raise RuntimeError("Choice identities are not unique")
    if pairs.duplicated(["learning_year", "country", "year"]).any():
        raise RuntimeError("Paired country-year identities are not unique")
    for account_column in ("dispersed_selection_account", "clustered_selection_account"):
        if not pairs[account_column].eq("residual_peak_system_cost").all():
            raise RuntimeError("Paired table mixes in an EFC trajectory screen")
    for learning_year in YEARS:
        year_choices = choices[choices["learning_year"].eq(learning_year)]
        year_pairs = pairs[pairs["learning_year"].eq(learning_year)]
        if len(year_pairs) != N_COUNTRY_YEARS:
            raise RuntimeError(f"Cost year {learning_year} does not contain 132 paired country-years")
        if year_pairs["country"].nunique() != 15:
            raise RuntimeError(f"Cost year {learning_year} does not contain 15 countries")
        for chronology in CHRONOLOGIES:
            if len(year_choices[year_choices["chronology"].eq(chronology)]) != N_COUNTRY_YEARS:
                raise RuntimeError(f"Cost year {learning_year}, {chronology} does not contain 132 choices")

    # Recompute headline counts and compare them with both summary and final QA.
    qa_identity_counts = primary_identity_flip_counts(qa)
    for learning_year in YEARS:
        frame = pairs[pairs["learning_year"].eq(learning_year)]
        for metric in ("identity_flip", "participation_flip", "duration_flip", "material_power_change"):
            row = summary[
                summary["learning_year"].eq(learning_year)
                & summary["account"].eq("residual_peak_system_cost")
                & summary["metric"].eq(metric)
            ]
            if len(row) != 1:
                raise RuntimeError(f"Summary row {learning_year}/{metric} is not unique")
            observed = int(pd.to_numeric(frame[metric], errors="raise").astype(bool).sum())
            if int(row.iloc[0]["count"]) != observed or int(row.iloc[0]["denominator"]) != N_COUNTRY_YEARS:
                raise RuntimeError(f"Summary mismatch for {learning_year}/{metric}")
        qa_identity = qa_identity_counts[str(learning_year)]
        if qa_identity != int(frame["identity_flip"].astype(bool).sum()):
            raise RuntimeError(f"Final QA identity count mismatch for {learning_year}")
        year_qa = qa["year_QA"][str(learning_year)]
        if not np.isclose(float(year_qa["choice_tolerance_eur_kw_year"]), 0.01):
            raise RuntimeError("Procurement tie tolerance is not 0.01 EUR kW-1 yr-1")
        if year_qa.get("trajectory_screen_role") != (
            "revaluation_of_residual_peak_system_cost_optimized_trajectories__not_efc_objective_optimization"
        ):
            raise RuntimeError("Residual-peak optimization and EFC trajectory screen are not separated")

    return choices, pairs, summary, qa


def classify_family(frame: pd.DataFrame) -> pd.Series:
    active = frame["tail_active"].astype(bool)
    tech = frame["decision_tech"].astype(str).str.lower()
    return pd.Series(
        np.select(
            [~active, tech.isin(["li_lfp", "lfp"])],
            ["No tail", "4 h LFP"],
            default="LDES",
        ),
        index=frame.index,
        name="choice_family",
    )


def make_sources(
    choices: pd.DataFrame,
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    selected = choices.copy()
    selected["choice_family"] = classify_family(selected)
    composition = (
        selected.groupby(["learning_year", "chronology", "choice_family"], observed=True)
        .size()
        .rename("country_year_count")
        .reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [YEARS, CHRONOLOGIES, ["No tail", "4 h LFP", "LDES"]],
        names=["learning_year", "chronology", "choice_family"],
    ).to_frame(index=False)
    composition = complete.merge(
        composition,
        how="left",
        on=["learning_year", "chronology", "choice_family"],
        validate="one_to_one",
    )
    composition["country_year_count"] = composition["country_year_count"].fillna(0).astype(int)
    composition["share_percent"] = 100.0 * composition["country_year_count"] / N_COUNTRY_YEARS
    closed = composition.groupby(["learning_year", "chronology"])["country_year_count"].sum()
    if not closed.eq(N_COUNTRY_YEARS).all():
        raise RuntimeError("Procurement composition does not close to 132")

    metrics = ["identity_flip", "participation_flip", "duration_flip", "material_power_change"]
    rates = summary[
        summary["account"].eq("residual_peak_system_cost")
        & summary["metric"].isin(metrics)
    ].copy()
    rates["rate_percent"] = 100.0 * rates["rate"]
    rates["ci_low_percent"] = 100.0 * rates["country_block_bootstrap_ci_low"]
    rates["ci_high_percent"] = 100.0 * rates["country_block_bootstrap_ci_high"]

    transitions = (
        pairs.groupby(
            ["learning_year", "dispersed_decision_duration_h", "clustered_decision_duration_h"],
            observed=True,
        )
        .size()
        .rename("country_year_count")
        .reset_index()
    )

    power = pairs[
        [
            "learning_year", "country", "year",
            "dispersed_tail_power_share", "clustered_tail_power_share",
            "delta_tail_power_share", "material_power_change", "identity_flip",
        ]
    ].copy()
    power["delta_tail_power_percentage_points"] = 100.0 * power["delta_tail_power_share"]
    power["dispersed_tail_power_percent"] = 100.0 * power["dispersed_tail_power_share"]
    power["clustered_tail_power_percent"] = 100.0 * power["clustered_tail_power_share"]

    return {
        "composition": composition,
        "rates": rates,
        "transitions": transitions,
        "power": power,
        "choices": selected,
    }


def panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    for label, ax in zip("abcd", axes):
        box = ax.get_position()
        fig.text(
            box.x0 - 0.010,
            box.y1 + 0.007,
            label,
            ha="right",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#111111",
        )


def deterministic_offsets(n: int, half_width: float = 0.13) -> np.ndarray:
    """Deterministic display-only offsets; source values remain unchanged."""
    if n <= 1:
        return np.zeros(max(n, 0), dtype=float)
    phase = (np.arange(n, dtype=float) * 0.6180339887498949) % 1.0
    return (2.0 * phase - 1.0) * half_width


def draw_panel_a(ax: plt.Axes, composition: pd.DataFrame, rates: pd.DataFrame) -> None:
    x = np.array([0.0, 0.72, 2.05, 2.77, 4.10, 4.82])
    order = [(year, chronology) for year in YEARS for chronology in CHRONOLOGIES]
    bottoms = np.zeros(len(x))
    category_order = ["No tail", "4 h LFP", "LDES"]
    colours = {"No tail": NO_TAIL, "4 h LFP": LFP, "LDES": LDES}
    for category in category_order:
        heights = []
        for learning_year, chronology in order:
            row = composition[
                composition["learning_year"].eq(learning_year)
                & composition["chronology"].eq(chronology)
                & composition["choice_family"].eq(category)
            ]
            heights.append(float(row.iloc[0]["share_percent"]))
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            width=0.55,
            color=colours[category],
            edgecolor=INK if category == "LDES" else "white",
            linewidth=0.35,
            hatch="////" if category == "LDES" else None,
            label=category,
        )
        bottoms += np.asarray(heights)
    ax.set_xticks(x)
    ax.set_xticklabels(["Disp.", "Clust."] * 3)
    for midpoint, learning_year in zip([0.36, 2.41, 4.46], YEARS):
        ax.text(midpoint, -0.18, str(learning_year), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.7)
        flips = int(
            rates[
                rates["learning_year"].eq(learning_year)
                & rates["metric"].eq("identity_flip")
            ].iloc[0]["count"]
        )
        ax.text(midpoint, 102.0, f"{flips}/132", ha="center", va="bottom", fontsize=5.2, color=INK, fontweight="bold")
    ax.set_ylim(0, 111)
    ax.set_xlim(-0.48, 5.30)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Selected portfolios (%)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.50, 1.01), columnspacing=0.9, handlelength=1.2)


def draw_panel_b(ax: plt.Axes, rates: pd.DataFrame) -> None:
    metric_order = ["identity_flip", "participation_flip", "duration_flip", "material_power_change"]
    metric_label = {
        "identity_flip": "Portfolio identity",
        "participation_flip": "Tail participation",
        "duration_flip": "Selected duration",
        "material_power_change": "Tail power >1 pp",
    }
    base_y = np.arange(len(metric_order))[::-1].astype(float)
    offsets = {2025: 0.20, 2030: 0.0, 2035: -0.20}
    for learning_year in YEARS:
        frame = rates[rates["learning_year"].eq(learning_year)].set_index("metric").loc[metric_order]
        y = base_y + offsets[learning_year]
        estimate = frame["rate_percent"].to_numpy(float)
        low = frame["ci_low_percent"].to_numpy(float)
        high = frame["ci_high_percent"].to_numpy(float)
        ax.errorbar(
            estimate,
            y,
            xerr=np.vstack([estimate - low, high - estimate]),
            fmt=YEAR_MARKER[learning_year],
            color=YEAR_COLOUR[learning_year],
            markerfacecolor="white",
            markeredgewidth=0.8,
            elinewidth=0.85,
            capsize=1.7,
            label=str(learning_year),
            zorder=3,
        )
        for xx, yy, count in zip(estimate, y, frame["count"].astype(int)):
            ax.annotate(str(count), (xx, yy), xytext=(3.2, 0), textcoords="offset points", va="center", fontsize=5.0, color=YEAR_COLOUR[learning_year])
    ax.axvline(0, color=INK, lw=0.65)
    ax.set_yticks(base_y)
    ax.set_yticklabels([metric_label[item] for item in metric_order])
    maximum = max(7.5, float(rates["ci_high_percent"].max()) + 1.1)
    ax.set_xlim(-0.5, maximum)
    ax.set_xlabel("Paired country-years changed (%)")
    ax.legend(ncol=3, loc="lower right", handletextpad=0.35, columnspacing=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)


def draw_panel_c(ax: plt.Axes, transitions: pd.DataFrame) -> None:
    max_duration = max(
        float(transitions["dispersed_decision_duration_h"].max()),
        float(transitions["clustered_decision_duration_h"].max()),
        168.0,
    )
    ax.plot([0, max_duration], [0, max_duration], color=MUTED, lw=0.7, ls="--", zorder=1)
    for learning_year in YEARS:
        frame = transitions[transitions["learning_year"].eq(learning_year)].copy()
        size = 7.0 + 5.8 * np.sqrt(frame["country_year_count"].to_numpy(float))
        ax.scatter(
            frame["dispersed_decision_duration_h"],
            frame["clustered_decision_duration_h"],
            s=size,
            marker=YEAR_MARKER[learning_year],
            facecolor="white",
            edgecolor=YEAR_COLOUR[learning_year],
            linewidth=0.75,
            alpha=0.88,
            label=str(learning_year),
            zorder=3,
        )
    ticks = [0, 4, 8, 12, 24, 48, 96, 168, 336, 720]
    ticks = [item for item in ticks if item <= max_duration]
    ax.set_xscale("symlog", linthresh=4, linscale=0.65, base=2)
    ax.set_yscale("symlog", linthresh=4, linscale=0.65, base=2)
    ax.set_xlim(-0.5, max_duration * 1.12)
    ax.set_ylim(-0.5, max_duration * 1.12)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Dispersed selected duration (h)")
    ax.set_ylabel("Clustered selected duration (h)")
    ax.legend(ncol=3, loc="upper left")
    ax.text(0.98, 0.03, "0 h denotes no tail", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color=MUTED)


def draw_panel_d(ax: plt.Axes, power: pd.DataFrame) -> None:
    for xloc, learning_year in enumerate(YEARS):
        frame = power[power["learning_year"].eq(learning_year)].sort_values(
            ["delta_tail_power_percentage_points", "country", "year"], kind="stable"
        )
        values = frame["delta_tail_power_percentage_points"].to_numpy(float)
        offsets = deterministic_offsets(len(frame))
        ax.scatter(
            xloc + offsets,
            values,
            s=7.0,
            color=YEAR_COLOUR[learning_year],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.48,
            rasterized=False,
            zorder=2,
        )
        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.vlines(xloc, q25, q75, color=INK, linewidth=2.5, zorder=4)
        ax.scatter([xloc], [median], s=16, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=5)
        changed = int(frame["material_power_change"].astype(bool).sum())
        y_anchor = max(float(np.nanmax(values)), 0.0)
        ax.annotate(
            f"{changed}/132",
            (xloc, y_anchor),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color=YEAR_COLOUR[learning_year],
            fontweight="bold",
        )
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels([str(item) for item in YEARS])
    ax.set_xlim(-0.45, len(YEARS) - 0.55)
    values = power["delta_tail_power_percentage_points"].to_numpy(float)
    lower = min(-2.0, float(np.nanmin(values)))
    upper = max(2.0, float(np.nanmax(values)))
    data_span = max(4.0, upper - lower)
    ax.set_ylim(lower - 0.10 * data_span, upper + 0.17 * data_span)
    ax.set_xlabel("Cost year")
    ax.set_ylabel("Clustered − dispersed tail power\n(percentage points)")
    ax.text(0.99, 0.03, "dots, country-years   bar, IQR   circle, median", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color=MUTED)


def export_figure(fig: plt.Figure, source_paths: list[Path], qa: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{STEM}.pdf"
    svg = OUT_DIR / f"{STEM}.svg"
    png = OUT_DIR / f"{STEM}.png"
    tiff = OUT_DIR / f"{STEM}.tiff"
    fig.savefig(pdf, bbox_inches=None)
    fig.savefig(svg, bbox_inches=None)
    fig.savefig(png, dpi=PNG_DPI, bbox_inches=None)
    fig.savefig(
        tiff,
        dpi=TIFF_DPI,
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    # Nature-ready line-art TIFF without an alpha channel.
    with Image.open(tiff) as image:
        rgb_tiff = image.convert("RGB")
    rgb_tiff.save(tiff, compression="tiff_lzw", dpi=(TIFF_DPI, TIFF_DPI))
    outputs: list[Path] = [pdf, svg, png, tiff]
    grayscale = OUT_DIR / f"{STEM}_grayscale.png"
    with Image.open(OUT_DIR / f"{STEM}.png") as image:
        image.convert("L").save(grayscale, dpi=(PNG_DPI, PNG_DPI))
    outputs.append(grayscale)
    manifest = {
        "status": "pass",
        "figure": "Extended Data Figure 6",
        "stem": STEM,
        "core_claim": "Residual-peak system-cost optimization identifies sparse chronology-dependent changes in portfolio identity, duration and tail power, but no tail-participation changes, after endogenous tail-power selection.",
        "n_countries": 15,
        "n_country_years_per_cost_year": N_COUNTRY_YEARS,
        "cost_years": list(YEARS),
        "chronologies": list(CHRONOLOGIES),
        "backend": "python-matplotlib",
        "selection_account": "residual_peak_system_cost",
        "efc_trajectory_screen_included": False,
        "choice_tolerance_eur_kw_year": 0.01,
        "style_inheritance": "v43 ED6 typography, palette, line weights and asymmetric 2x2 layout",
        "redundant_encoding": [
            "cost year uses both colour and marker shape",
            "LDES portfolio share uses both colour and diagonal hatching",
        ],
        "outputs": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs},
        "source_data": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in source_paths},
        "upstream_QA_sha256": sha256(QA_FILE),
        "upstream_QA_identity_flips": primary_identity_flip_counts(qa),
        "data_exclusions": 0,
        "display_only_transformations": [
            "panel c uses a symlog duration scale with a 0 h no-tail anchor",
            "panel d uses deterministic horizontal offsets solely to reveal overlapping observations",
        ],
    }
    (MANIFEST_DIR / f"{STEM}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> None:
    configure()
    choices, pairs, summary, qa = load_evidence()
    sources = make_sources(choices, pairs, summary)

    fig = plt.figure(figsize=(7.2, 4.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.955)
    gs = fig.add_gridspec(2, 12, hspace=0.60, wspace=1.05)
    axa = fig.add_subplot(gs[0, 0:5])
    axb = fig.add_subplot(gs[0, 5:12])
    axc = fig.add_subplot(gs[1, 0:6])
    axd = fig.add_subplot(gs[1, 6:12])
    axes = [axa, axb, axc, axd]

    draw_panel_a(axa, sources["composition"], sources["rates"])
    draw_panel_b(axb, sources["rates"])
    draw_panel_c(axc, sources["transitions"])
    draw_panel_d(axd, sources["power"])
    panel_labels(fig, axes)

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "composition": "ed6a_bounded_procurement_composition.csv",
        "rates": "ed6b_bounded_procurement_change_rates.csv",
        "transitions": "ed6c_bounded_duration_transitions.csv",
        "power": "ed6d_bounded_tail_power_changes.csv",
        "choices": "ed6_bounded_joint_choices_country_year_chronology.csv",
    }
    source_paths: list[Path] = []
    for key, filename in files.items():
        path = SOURCE_DIR / filename
        sources[key].to_csv(path, index=False)
        source_paths.append(path)
    export_figure(fig, source_paths, qa)
    plt.close(fig)
    print(f"Built {OUT_DIR / (STEM + '.pdf')}")


if __name__ == "__main__":
    try:
        main()
    except EvidencePending as exc:
        print(f"PENDING: {exc}", file=sys.stderr)
        raise SystemExit(2)
