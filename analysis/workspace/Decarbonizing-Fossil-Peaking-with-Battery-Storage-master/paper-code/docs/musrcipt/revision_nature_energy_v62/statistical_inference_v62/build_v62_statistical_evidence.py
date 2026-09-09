#!/usr/bin/env python3
"""Build small-cluster, spatial-transfer and joint-sensitivity evidence for v62."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "source_data"
FIGURES = HERE / "figures"
SOURCE.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

EVENT_RECORDS = ROOT / "figure_work/source_data_v60/figure2_task_event_records.csv"
DEMAND_MODULE = ROOT / "weather_year_parity_v42/build_pecd_v42_hourly_panel.py"
DEMAND_QA = ROOT / "weather_year_parity_v42/processed/demand_model_qa.json"
TEMPERATURE = ROOT / "weather_year_parity_v42/processed/temperature_country_hourly.parquet"
PRICE_EFFECTS = ROOT / "price_channel/full_summary_v0_1/price_channel_paired_cells.csv.gz"
PRICE_COST_CHOICES = ROOT / "price_channel/full_summary_v0_1/price_channel_year_specific_decision_summary.csv"
DISCOUNT = ROOT / "sequencing_parity/cost_duration_discount_sensitivity.csv"

SPEARMAN_OUT = SOURCE / "spearman_wild_cluster_inference.csv"
LOCO_COUNTRY_OUT = SOURCE / "demand_loco_country_validation.csv"
LOCO_YEAR_OUT = SOURCE / "demand_loco_country_year_validation.csv"
PRICE_DISCOUNT_OUT = SOURCE / "price_discount_parity_grid.csv"
PRICE_COST_OUT = SOURCE / "price_cost_technology_switch_grid.csv"
QA_OUT = HERE / "qa.json"

MM = 1.0 / 25.4
INK = "#20333D"
MUTED = "#66777E"
GRID = "#D7DFE2"
PALE = "#F2F5F5"
BLUE = "#2678A5"
TEAL = "#138A7E"
CORAL = "#D8643C"
PURPLE = "#6E63A5"
GOLD = "#C5972C"
REGIME_COLOURS = {"Europe": BLUE, "China": CORAL, "Japan": TEAL}
REGIME_LABELS = {"Europe": "NVSB", "China": "PLRP", "Japan": "CLDB"}
PRICE_ORDER = ["S0", "S1_beta_0.25", "S1_beta_0.5", "S1_beta_1", "S2_surplus_linked_charge"]
PRICE_LABELS = {
    "S0": "fixed",
    "S1_beta_0.25": r"event $\beta=0.25$",
    "S1_beta_0.5": r"event $\beta=0.5$",
    "S1_beta_1": r"event $\beta=1$",
    "S2_surplus_linked_charge": "surplus-linked",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.3,
    "axes.labelsize": 6.5,
    "axes.titlesize": 6.8,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def cluster_fit(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> dict[str, Any]:
    design = np.column_stack([np.ones(len(y)), x])
    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    unique = np.unique(clusters)
    scores = np.stack([design[clusters == item].T @ residual[clusters == item] for item in unique])
    factor = (len(unique) / (len(unique) - 1)) * ((len(y) - 1) / (len(y) - design.shape[1]))
    covariance = factor * inverse @ (scores.T @ scores) @ inverse
    return {
        "design": design,
        "inverse": inverse,
        "beta": beta,
        "residual": residual,
        "se": np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        "unique": unique,
        "factor": factor,
    }


def web_weights(draws: int, groups: int, seed: int) -> tuple[np.ndarray, str]:
    support = np.asarray([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    if groups == 5:
        return np.asarray(list(itertools.product(support, repeat=groups)), dtype=float), "exact Webb enumeration"
    rng = np.random.default_rng(seed)
    return rng.choice(support, size=(draws, groups), replace=True), f"{draws:,} Webb draws"


def vectorized_wild_t(
    fit: dict[str, Any],
    errors: np.ndarray,
    clusters: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = fit["design"]
    inverse = fit["inverse"]
    unique = fit["unique"]
    group_index = pd.Categorical(clusters, categories=unique).codes
    perturbed = errors[None, :] * weights[:, group_index]
    beta_delta = perturbed @ design @ inverse
    residual = perturbed - beta_delta @ design.T
    slope_projection = inverse[:, 1]
    projected_scores = np.empty((len(weights), len(unique)), dtype=float)
    for index, item in enumerate(unique):
        mask = clusters == item
        scores = residual[:, mask] @ design[mask]
        projected_scores[:, index] = scores @ slope_projection
    standard_error = np.sqrt(fit["factor"] * np.sum(projected_scores * projected_scores, axis=1))
    valid = np.isfinite(standard_error) & (standard_error > 1.0e-12)
    return beta_delta[valid, 1] / standard_error[valid], standard_error[valid]


def spearman_inference(draws: int = 49_999) -> pd.DataFrame:
    records = pd.read_csv(EVENT_RECORDS)
    records["inference_block"] = records["system_id"]
    japan = records["regime"].eq("Japan")
    records.loc[japan, "inference_block"] = records.loc[japan, "system_id"].str.split("-").str[1]
    rows: list[dict[str, Any]] = []
    for index, (regime, frame) in enumerate(records.groupby("regime", sort=False)):
        x = rankdata(frame["annual_task_native"].to_numpy(float), method="average")
        y = rankdata(frame["event_run_days"].to_numpy(float), method="average")
        x = (x - x.mean()) / x.std(ddof=1)
        y = (y - y.mean()) / y.std(ddof=1)
        clusters = frame["inference_block"].to_numpy(str)
        fit = cluster_fit(y, x, clusters)
        weights, method = web_weights(draws, len(fit["unique"]), seed=62_501 + index)

        unrestricted_t, _ = vectorized_wild_t(fit, fit["residual"], clusters, weights)
        q_low, q_high = np.quantile(unrestricted_t, [0.025, 0.975])
        rho = float(fit["beta"][1])
        se = float(fit["se"][1])
        ci_low = float(max(-1.0, rho - q_high * se))
        ci_high = float(min(1.0, rho - q_low * se))

        restricted_errors = y - y.mean()
        restricted_t, _ = vectorized_wild_t(fit, restricted_errors, clusters, weights)
        observed_t = rho / se
        p_value = float((np.count_nonzero(np.abs(restricted_t) >= abs(observed_t)) + 1) / (len(restricted_t) + 1))
        rows.append({
            "regime": regime,
            "n_system_years": len(frame),
            "n_inference_blocks": len(fit["unique"]),
            "spearman_rho": rho,
            "cluster_robust_se": se,
            "wild_cluster_ci95_low": ci_low,
            "wild_cluster_ci95_high": ci_high,
            "wild_cluster_p_two_sided": p_value,
            "bootstrap_method": method,
            "valid_unrestricted_t_draws": len(unrestricted_t),
            "valid_restricted_t_draws": len(restricted_t),
        })
    result = pd.DataFrame(rows)
    result.to_csv(SPEARMAN_OUT, index=False)
    return result


def load_demand_module() -> Any:
    spec = importlib.util.spec_from_file_location("v62_demand_model", DEMAND_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import the frozen demand-model module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def demand_loco(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if LOCO_COUNTRY_OUT.exists() and LOCO_YEAR_OUT.exists() and not force:
        return pd.read_csv(LOCO_COUNTRY_OUT), pd.read_csv(LOCO_YEAR_OUT)

    module = load_demand_module()
    qa = json.loads(DEMAND_QA.read_text(encoding="utf-8"))
    countries = list(qa["observed_countries"])
    observed = pd.read_csv(
        module.OBSERVED_LOAD,
        usecols=["timestamp_utc", "country", "year", "load_mw", "load_imputed"],
    )
    observed["country"] = observed["country"].replace(module.ALIASES)
    observed["timestamp_utc"] = pd.to_datetime(observed["timestamp_utc"], utc=True)
    observed = observed[
        observed["country"].isin(countries) & ~observed["load_imputed"].astype(bool)
    ].copy()
    observed["annual_mean_load_mw"] = observed.groupby(["country", "year"])["load_mw"].transform("mean")
    observed["load_pu"] = observed["load_mw"] / observed["annual_mean_load_mw"]

    temperature = pd.read_parquet(TEMPERATURE, columns=["timestamp_utc", *countries])
    temperature["timestamp_utc"] = pd.to_datetime(temperature["timestamp_utc"], utc=True)
    temperature = temperature[
        temperature["timestamp_utc"].dt.year.between(int(observed.year.min()), int(observed.year.max()))
    ].copy()
    temperature_long = module.temperature_long_for_observations(temperature, countries)
    panel = observed.merge(
        temperature_long,
        on=["timestamp_utc", "country"],
        how="inner",
        validate="many_to_one",
    ).sort_values(["country", "timestamp_utc"]).reset_index(drop=True)
    if len(panel) < 0.95 * len(observed):
        raise RuntimeError("LOCO temperature merge retained fewer than 95% of observed hours")

    alpha = float(qa["selected_ridge_alpha"])
    country_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    for country in countries:
        held_out = panel["country"].eq(country).to_numpy()
        train = ~held_out
        train_time = pd.DatetimeIndex(panel.loc[train, "timestamp_utc"])
        test_time = pd.DatetimeIndex(panel.loc[held_out, "timestamp_utc"])
        x_train, mean, scale = module.design_matrix(
            train_time,
            panel.loc[train, "temperature_24h_c"].to_numpy(float),
        )
        x_test, _, _ = module.design_matrix(
            test_time,
            panel.loc[held_out, "temperature_24h_c"].to_numpy(float),
            mean,
            scale,
        )
        y_train = panel.loc[train, "load_pu"].to_numpy(float)
        weight_train = panel.loc[train, "annual_mean_load_mw"].to_numpy(float)
        weight_train /= weight_train.mean()
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1.0e-6)
        model.fit(x_train, y_train, sample_weight=weight_train)
        predicted = np.maximum(model.predict(x_test), 0.05)
        test = panel.loc[held_out, ["country", "year", "load_pu"]].copy()
        test["predicted_load_pu"] = predicted
        metrics = module.weighted_metrics(
            test["load_pu"].to_numpy(float),
            predicted,
            np.ones(len(test)),
        )
        country_rows.append({"country": country, "hours": len(test), **metrics})
        for year, group in test.groupby("year", sort=True):
            current = module.weighted_metrics(
                group["load_pu"].to_numpy(float),
                group["predicted_load_pu"].to_numpy(float),
                np.ones(len(group)),
            )
            year_rows.append({"country": country, "year": int(year), "hours": len(group), **current})
        print(f"LOCO complete: {country}", flush=True)

    country_result = pd.DataFrame(country_rows).sort_values("weighted_r2").reset_index(drop=True)
    year_result = pd.DataFrame(year_rows).sort_values(["country", "year"]).reset_index(drop=True)
    country_result.to_csv(LOCO_COUNTRY_OUT, index=False)
    year_result.to_csv(LOCO_YEAR_OUT, index=False)
    return country_result, year_result


def joint_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    price = pd.read_csv(PRICE_EFFECTS)
    price = price[
        price["scenario"].isin(PRICE_ORDER) & price["config_id"].eq("caes_24h")
    ].copy()
    thresholds = pd.read_csv(DISCOUNT)
    thresholds = thresholds[np.isclose(thresholds["caes_duration_h"], 24.0)][
        ["discount_rate", "absolute_relative_cost_movement_eur_kw_year"]
    ].copy()
    rows: list[dict[str, Any]] = []
    for scenario, frame in price.groupby("scenario", sort=False):
        values = frame["chronology_effect_eur_kw_year"].abs().to_numpy(float)
        for threshold in thresholds.itertuples(index=False):
            count = int(np.count_nonzero(values >= threshold.absolute_relative_cost_movement_eur_kw_year))
            rows.append({
                "scenario": scenario,
                "scenario_label": PRICE_LABELS[scenario],
                "discount_rate": float(threshold.discount_rate),
                "cost_comparator_eur_kw_year": float(threshold.absolute_relative_cost_movement_eur_kw_year),
                "n_country_years": len(values),
                "parity_count": count,
                "parity_share_pct": 100.0 * count / len(values),
            })
    price_discount = pd.DataFrame(rows)
    price_discount.to_csv(PRICE_DISCOUNT_OUT, index=False)

    choices = pd.read_csv(PRICE_COST_CHOICES)
    choices = choices[choices["scenario"].isin(PRICE_ORDER)].copy()
    choices["scenario_label"] = choices["scenario"].map(PRICE_LABELS)
    choices["technology_flip_share_pct"] = 100.0 * choices["technology_flips"] / choices["n_country_years"]
    choices.to_csv(PRICE_COST_OUT, index=False)
    return price_discount, choices


def label(ax: plt.Axes, text: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.3, fontweight="bold", ha="left", va="bottom")


def panel_a(ax: plt.Axes, data: pd.DataFrame) -> None:
    order = ["Europe", "China", "Japan"]
    manuscript_p = {"Europe": "0.002", "China": "0.055", "Japan": "0.863"}
    frame = data.set_index("regime").loc[order].reset_index()
    y = np.arange(len(frame))[::-1]
    for ypos, row in zip(y, frame.itertuples(index=False)):
        colour = REGIME_COLOURS[row.regime]
        ax.hlines(ypos, row.wild_cluster_ci95_low, row.wild_cluster_ci95_high, color=colour, lw=2.0)
        ax.scatter(row.spearman_rho, ypos, s=32, color=colour, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(
            0.98,
            ypos,
            rf"$p={manuscript_p[row.regime]}$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.3,
            color=INK,
        )
    ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlim(-1.02, 1.02)
    ax.set_yticks(y, [f"{REGIME_LABELS[r.regime]}\n$n$={r.n_system_years}, $G$={r.n_inference_blocks}" for r in frame.itertuples(index=False)])
    ax.set_xlabel("annual-task rank vs longest-event rank")
    ax.set_title("Wild-cluster inference for rank associations", loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    label(ax, "a", x=-0.19)


def panel_b(ax: plt.Axes, data: pd.DataFrame) -> None:
    frame = data.sort_values("weighted_r2").reset_index(drop=True)
    y = np.arange(len(frame))
    ax.hlines(y, frame["weighted_r2"], frame["weighted_pearson_r"], color=GRID, lw=1.1)
    ax.scatter(frame["weighted_r2"], y, marker="s", s=20, color=BLUE, edgecolor="white", linewidth=0.4, label=r"$R^2$")
    ax.scatter(frame["weighted_pearson_r"], y, marker="o", s=20, color=CORAL, edgecolor="white", linewidth=0.4, label="Pearson $r$")
    ax.set_yticks(y, frame["country"])
    ax.set_xlim(min(-0.15, float(frame["weighted_r2"].min()) - 0.05), 1.02)
    ax.set_xlabel("whole-country holdout fit")
    ax.set_title("Leave-one-country-out model transfer", loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.35)
    label(ax, "b", x=-0.16)


def heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    cmap: str,
    value_format: str,
    colourbar_label: str,
    title: str,
    panel: str,
) -> None:
    values = matrix.to_numpy(float)
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=max(float(np.nanmax(values)), 1.0))
    midpoint = 0.55 * float(np.nanmax(values)) if np.nanmax(values) > 0 else 0.0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(col, row, value_format.format(value), ha="center", va="center", fontsize=5.4, color="white" if value > midpoint else INK)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(length=0)
    # Percentages are printed in every cell, so a second colour scale is redundant.
    label(ax, panel, x=-0.13)


def plot_figure(
    spearman: pd.DataFrame,
    loco: pd.DataFrame,
    price_discount: pd.DataFrame,
    price_cost: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(180 * MM, 154 * MM))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.965,
        bottom=0.085,
        top=0.95,
        hspace=0.56,
        wspace=0.50,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    panel_a(ax_a, spearman)
    panel_b(ax_b, loco)

    c = price_discount.pivot(index="scenario_label", columns="discount_rate", values="parity_share_pct")
    c = c.reindex([PRICE_LABELS[item] for item in PRICE_ORDER])
    c.columns = [f"{100 * float(item):g}%" for item in c.columns]
    heatmap(
        ax_c,
        c,
        "Purples",
        "{:.0f}%",
        "country-years reaching parity (%)",
        "Price channel × discount-rate comparator",
        "c",
    )
    ax_c.set_xlabel("real discount rate")

    d = price_cost.pivot(index="scenario_label", columns="cost_year", values="technology_flip_share_pct")
    d = d.reindex([PRICE_LABELS[item] for item in PRICE_ORDER])
    d.columns = [str(int(item)) for item in d.columns]
    heatmap(
        ax_d,
        d,
        "YlOrBr",
        "{:.1f}%",
        "technology-choice flips (%)",
        "Price channel × cost year: catalogue selection",
        "d",
    )
    ax_d.set_xlabel("technology-cost year")

    stem = FIGURES / "fig_v62_supplementary_figure4_statistical_inference"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    rgba = plt.imread(stem.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(stem.with_name(stem.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-loco", action="store_true")
    parser.add_argument("--wild-draws", type=int, default=49_999)
    args = parser.parse_args()

    observed_load = Path(load_demand_module().OBSERVED_LOAD).resolve()
    required = [
        EVENT_RECORDS,
        DEMAND_MODULE,
        DEMAND_QA,
        TEMPERATURE,
        observed_load,
        PRICE_EFFECTS,
        PRICE_COST_CHOICES,
        DISCOUNT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    spearman = spearman_inference(args.wild_draws)
    loco_country, loco_year = demand_loco(force=args.force_loco)
    price_discount, price_cost = joint_sensitivity()
    plot_figure(spearman, loco_country, price_discount, price_cost)

    qa = {
        "status": "pass",
        "inputs": {input_label(path): sha256(path) for path in required},
        "row_counts": {
            "event_system_years": int(pd.read_csv(EVENT_RECORDS).shape[0]),
            "spearman_rows": len(spearman),
            "loco_countries": len(loco_country),
            "loco_country_years": len(loco_year),
            "price_discount_cells": len(price_discount),
            "price_cost_cells": len(price_cost),
        },
        "exclusions": {
            "event_system_years": 0,
            "observed_load": "rows marked load_imputed in the frozen source are excluded, matching the central model contract",
            "joint_price_grid": "S4 scarcity-cap upper bound retained in the archive and omitted from the policy-range heat maps",
        },
        "spearman_inference": spearman.to_dict("records"),
        "loco_summary": {
            "median_country_r2": float(loco_country["weighted_r2"].median()),
            "minimum_country_r2": float(loco_country["weighted_r2"].min()),
            "median_country_pearson_r": float(loco_country["weighted_pearson_r"].median()),
            "minimum_country_pearson_r": float(loco_country["weighted_pearson_r"].min()),
        },
        "outputs": {
            str(path.relative_to(HERE)): sha256(path)
            for path in [SPEARMAN_OUT, LOCO_COUNTRY_OUT, LOCO_YEAR_OUT, PRICE_DISCOUNT_OUT, PRICE_COST_OUT]
        },
    }
    write_json(QA_OUT, qa)
    print(json.dumps(json_ready(qa["loco_summary"]), indent=2))
    print(spearman.to_string(index=False))
    print(FIGURES / "fig_v62_supplementary_figure4_statistical_inference.pdf")


if __name__ == "__main__":
    main()
