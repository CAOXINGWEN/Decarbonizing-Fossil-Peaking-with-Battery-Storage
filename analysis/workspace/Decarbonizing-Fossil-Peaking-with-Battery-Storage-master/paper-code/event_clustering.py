"""
M1 Event Clustering: Extreme peaking demand temporal clustering analysis.

Classifies daily peaking demand into three states (zero / low / extreme),
extracts consecutive extreme-day runs, computes clustering metrics per
province and season, and fits run-length distributions (geometric vs
discrete Weibull).

Outputs:
  - province_extreme_event_runs.csv      (individual runs)
  - province_clustering_metrics.csv      (annual per province)
  - province_season_clustering.csv       (season × province)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import skew, kurtosis

from monte_carlo_simulation import (
    classify_states,
    estimate_transition_matrix,
)
from config import MC_CONFIG


def load_demand_matrix(filepath):
    """Load 26×365 Excel into (365×26) DataFrame with month column.

    The Excel has provinces as row index and MMDD strings as columns.
    Returns DataFrame with days as rows, province names as columns,
    plus a 'month' column (int 1-12).
    """
    raw = pd.read_excel(filepath, index_col=0)
    df = raw.T  # now: rows = MMDD day strings, columns = province names
    df.index.name = "mmdd"
    df["month"] = df.index.str[:2].astype(int)
    return df

# ───────────────────────────────────────────────────────────────────
# Province name mapping (Excel English → config Chinese)
# ───────────────────────────────────────────────────────────────────

EN_TO_CN = {
    "Hebei": "河北省",
    "Shandong": "山东省",
    "Liaoning": "辽宁省",
    "Shanxi": "山西省",
    "Shaanxi": "陕西省",
    "Tianjin": "天津市",
    "Guangxi": "广西壮族自治区",
    "Guangdong": "广东省",
    "Jilin": "吉林省",
    "Inner Mongolia": "内蒙古自治区",
    "Henan": "河南省",
    "Jiangsu": "江苏省",
    "Gansu": "甘肃省",
    "Ningxia": "宁夏回族自治区",
    "Xinjiang": "新疆维吾尔自治区",
    "Qinghai": "青海省",
    "Beijing": "北京市",
    "Hubei": "湖北省",
    "Hunan": "湖南省",
    "Jiangxi": "江西省",
    "Zhejiang": "浙江省",
    "Fujian": "福建省",
    "Anhui": "安徽省",
    "Guizhou": "贵州省",
    "Sichuan": "四川省",
    "Yunnan": "云南省",
}

CN_TO_EN = {v: k for k, v in EN_TO_CN.items()}

# ───────────────────────────────────────────────────────────────────
# Run-length extraction
# ───────────────────────────────────────────────────────────────────

def extract_run_lengths(states, target_state=2):
    """Return list of consecutive-run lengths for *target_state*.

    Also returns a list of (start_index, length) tuples for each run.
    """
    runs = []
    current_len = 0
    start_idx = None
    for i, s in enumerate(states):
        if s == target_state:
            if current_len == 0:
                start_idx = i
            current_len += 1
        else:
            if current_len > 0:
                runs.append((start_idx, current_len))
            current_len = 0
    if current_len > 0:
        runs.append((start_idx, current_len))
    return runs  # list of (start_index, length)


# ───────────────────────────────────────────────────────────────────
# Clustering metrics
# ───────────────────────────────────────────────────────────────────

def compute_clustering_metrics(run_tuples):
    """Compute summary metrics from run (start, length) tuples."""
    if not run_tuples:
        return {
            "n_runs": 0,
            "total_extreme_days": 0,
            "mean_run_length": np.nan,
            "median_run_length": np.nan,
            "max_run_length": 0,
            "frac_runs_ge3": np.nan,
        }
    lengths = np.array([t[1] for t in run_tuples])
    return {
        "n_runs": len(lengths),
        "total_extreme_days": int(lengths.sum()),
        "mean_run_length": float(lengths.mean()),
        "median_run_length": float(np.median(lengths)),
        "max_run_length": int(lengths.max()),
        "frac_runs_ge3": float((lengths >= 3).sum() / len(lengths)),
    }


def compute_p22(transition_matrix):
    """Extract P(state 2 → state 2) from 3×3 matrix."""
    return float(transition_matrix[2, 2])


# ───────────────────────────────────────────────────────────────────
# Distribution fitting
# ───────────────────────────────────────────────────────────────────

def _logit(x):
    x = float(np.clip(x, 1e-12, 1.0 - 1e-12))
    return np.log(x / (1.0 - x))


def _aic(log_likelihood, n_params):
    if not np.isfinite(log_likelihood):
        return np.nan
    return float(2 * n_params - 2 * log_likelihood)


def _discrete_weibull_loglik(lengths, q, beta):
    """Type-I discrete Weibull log-likelihood for lengths on {1,2,...}.

    P(L=k) = q^((k-1)^beta) - q^(k^beta), 0 < q < 1, beta > 0.
    beta = 1 gives the geometric distribution with p = 1 - q.
    """
    lengths = np.asarray(lengths, dtype=float)
    if not (0.0 < q < 1.0) or beta <= 0:
        return -np.inf

    log_q = np.log(q)
    left = np.power(lengths - 1.0, beta) * log_q
    right = np.power(lengths, beta) * log_q
    diff = right - left
    diff = np.minimum(diff, -1e-15)
    log_probs = left + np.log1p(-np.exp(diff))
    if not np.all(np.isfinite(log_probs)):
        return -np.inf
    return float(np.sum(log_probs))


def _fit_discrete_weibull_mle(lengths, geom_p):
    """Fit a two-parameter discrete Weibull by numerical MLE."""
    lengths = np.asarray(lengths, dtype=int)
    if len(lengths) == 0:
        return np.nan, np.nan, np.nan

    q0 = float(np.clip(1.0 - geom_p, 1e-6, 1.0 - 1e-6))
    starts = [(_logit(q0), np.log(beta0)) for beta0 in (0.5, 0.8, 1.0, 1.4, 2.0)]
    bounds = [(-20.0, 20.0), (np.log(0.05), np.log(20.0))]

    best = None
    for start in starts:
        def objective(theta):
            q = float(expit(theta[0]))
            beta = float(np.exp(theta[1]))
            ll = _discrete_weibull_loglik(lengths, q, beta)
            return -ll if np.isfinite(ll) else np.inf

        opt = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not opt.success or not np.isfinite(opt.fun):
            continue
        if best is None or opt.fun < best.fun:
            best = opt

    if best is None:
        return np.nan, np.nan, np.nan

    q_hat = float(expit(best.x[0]))
    beta_hat = float(np.exp(best.x[1]))
    ll_hat = float(-best.fun)
    return q_hat, beta_hat, ll_hat


def fit_run_length_distribution(run_tuples):
    """Fit geometric and discrete Weibull duration models; select by AIC.

    Geometric: P(X=k) = (1-p)^{k-1} p,  k = 1,2,...  (scipy.stats.geom)
    DWeibull:  P(X=k) = q^((k-1)^beta) - q^(k^beta)

    Returns dict with fitted parameters and AIC values.
    """
    result = {
        "best_fit_model": np.nan,
        "geom_p": np.nan,
        "dweibull_q": np.nan,
        "dweibull_beta": np.nan,
        "aic_geom": np.nan,
        "aic_dweibull": np.nan,
    }
    if not run_tuples:
        return result

    lengths = np.array([t[1] for t in run_tuples])
    n = len(lengths)
    if n < 5:
        geom_p = 1.0 / lengths.mean() if lengths.mean() > 0 else 1.0
        geom_p = np.clip(geom_p, 1e-6, 1.0 - 1e-6)
        result["geom_p"] = float(geom_p)
        result["best_fit_model"] = "geometric"
        ll = np.sum(st.geom.logpmf(lengths, geom_p))
        result["aic_geom"] = _aic(ll, 1)
        return result

    # --- Geometric MLE ---
    geom_p = 1.0 / lengths.mean()
    geom_p = np.clip(geom_p, 1e-6, 1.0 - 1e-6)
    ll_geom = np.sum(st.geom.logpmf(lengths, geom_p))
    aic_geom = _aic(ll_geom, 1)

    result["geom_p"] = float(geom_p)
    result["aic_geom"] = float(aic_geom) if np.isfinite(aic_geom) else np.nan

    # --- Discrete Weibull MLE
    dweibull_q, dweibull_beta, ll_dweibull = _fit_discrete_weibull_mle(lengths, geom_p)
    if np.isfinite(ll_dweibull):
        result["dweibull_q"] = float(dweibull_q)
        result["dweibull_beta"] = float(dweibull_beta)
        result["aic_dweibull"] = _aic(ll_dweibull, 2)

    model_aics = {
        "geometric": result["aic_geom"],
        "discrete_weibull": result["aic_dweibull"],
    }
    finite_aics = {model: aic for model, aic in model_aics.items() if np.isfinite(aic)}
    result["best_fit_model"] = min(finite_aics, key=finite_aics.get) if finite_aics else np.nan

    return result


# ───────────────────────────────────────────────────────────────────
# Per-province analysis
# ───────────────────────────────────────────────────────────────────

def analyze_province(df, province_en, seasons, quantile):
    """Run full clustering analysis for one province.

    Returns (annual_dict, list_of_season_dicts, list_of_run_detail_dicts).
    """
    province_cn = EN_TO_CN.get(province_en, province_en)
    values = df[province_en].values.astype(float)
    months = df["month"].values

    # ── Annual analysis ──
    nonzero = values[values > 0]
    q75 = np.percentile(nonzero, quantile * 100) if len(nonzero) > 0 else 0.0
    states = classify_states(values, q75)
    run_tuples = extract_run_lengths(states, target_state=2)
    metrics = compute_clustering_metrics(run_tuples)
    T = estimate_transition_matrix(states, n_states=3)
    p22 = compute_p22(T)
    fit = fit_run_length_distribution(run_tuples)

    annual_row = {
        "province_en": province_en,
        "province_cn": province_cn,
        "n_extreme_days": metrics["total_extreme_days"],
        **metrics,
        "p22": p22,
        # 需求值分布形态指标（全量非零值）
        "demand_skewness": float(skew(nonzero)) if len(nonzero) > 2 else np.nan,
        "demand_kurtosis": float(kurtosis(nonzero)) if len(nonzero) > 2 else np.nan,
        # 极端值分布形态指标（仅 state=2 的值）
        "extreme_skewness": float(skew(values[states == 2])) if (states == 2).sum() > 2 else np.nan,
        "extreme_kurtosis": float(kurtosis(values[states == 2])) if (states == 2).sum() > 2 else np.nan,
        # 运行长度分布形态指标
        "runlen_skewness": float(skew([t[1] for t in run_tuples])) if len(run_tuples) > 2 else np.nan,
        "runlen_kurtosis": float(kurtosis([t[1] for t in run_tuples])) if len(run_tuples) > 2 else np.nan,
        **fit,
    }
    del annual_row["total_extreme_days"]

    # ── Run detail rows (annual) ──
    run_details = []
    for rid, (start, length) in enumerate(run_tuples):
        mean_intensity = float(values[start : start + length].mean())
        run_details.append({
            "province_en": province_en,
            "province_cn": province_cn,
            "season": "annual",
            "run_id": rid,
            "start_day_index": start,
            "run_length": length,
            "mean_intensity_kwh": mean_intensity,
        })

    # ── Seasonal analysis ──
    season_rows = []
    for season_name, season_months in seasons.items():
        mask = np.isin(months, season_months)
        season_vals = values[mask]
        if len(season_vals) == 0:
            continue

        nonzero_s = season_vals[season_vals > 0]
        q75_s = (
            np.percentile(nonzero_s, quantile * 100) if len(nonzero_s) > 0 else 0.0
        )
        states_s = classify_states(season_vals, q75_s)
        run_tuples_s = extract_run_lengths(states_s, target_state=2)
        metrics_s = compute_clustering_metrics(run_tuples_s)
        T_s = estimate_transition_matrix(states_s, n_states=3)
        p22_s = compute_p22(T_s)

        season_rows.append({
            "province_en": province_en,
            "province_cn": province_cn,
            "season": season_name,
            "n_extreme_days": metrics_s["total_extreme_days"],
            "n_runs": metrics_s["n_runs"],
            "mean_run_length": metrics_s["mean_run_length"],
            "median_run_length": metrics_s["median_run_length"],
            "max_run_length": metrics_s["max_run_length"],
            "frac_runs_ge3": metrics_s["frac_runs_ge3"],
            "p22": p22_s,
        })

        # seasonal run details
        day_indices = np.where(mask)[0]
        for rid, (start_in_season, length) in enumerate(run_tuples_s):
            actual_start = int(day_indices[start_in_season])
            mean_intensity = float(
                values[actual_start : actual_start + length].mean()
            )
            run_details.append({
                "province_en": province_en,
                "province_cn": province_cn,
                "season": season_name,
                "run_id": rid,
                "start_day_index": actual_start,
                "run_length": length,
                "mean_intensity_kwh": mean_intensity,
            })

    return annual_row, season_rows, run_details


# ───────────────────────────────────────────────────────────────────
# Main orchestrator
# ───────────────────────────────────────────────────────────────────

def main(data_path=None, output_dir=None):
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__),
            "..", "data",
            "Daily Natural Gas Peaking Power Generation by Province in China.xlsx",
        )
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), "output", "m1_clustering"
        )
    os.makedirs(output_dir, exist_ok=True)

    seasons = MC_CONFIG["seasons"]
    quantile = MC_CONFIG["intensity_quantile"]

    print(f"Loading data from {data_path} ...")
    df = load_demand_matrix(data_path)
    province_columns = [c for c in df.columns if c in EN_TO_CN]
    print(f"Found {len(province_columns)} provinces: {province_columns}")

    all_annual = []
    all_seasonal = []
    all_runs = []

    for prov in province_columns:
        annual, seasonal, runs = analyze_province(df, prov, seasons, quantile)
        all_annual.append(annual)
        all_seasonal.extend(seasonal)
        all_runs.extend(runs)

    df_annual = pd.DataFrame(all_annual)
    df_seasonal = pd.DataFrame(all_seasonal)
    df_runs = pd.DataFrame(all_runs)

    annual_cols = [
        "province_en", "province_cn", "n_extreme_days", "n_runs",
        "mean_run_length", "median_run_length", "max_run_length",
        "frac_runs_ge3", "p22",
        "demand_skewness", "demand_kurtosis",
        "extreme_skewness", "extreme_kurtosis",
        "runlen_skewness", "runlen_kurtosis",
        "best_fit_model", "geom_p",
        "dweibull_q", "dweibull_beta",
        "aic_geom", "aic_dweibull",
    ]
    df_annual = df_annual[[c for c in annual_cols if c in df_annual.columns]]

    season_cols = [
        "province_en", "province_cn", "season", "n_extreme_days", "n_runs",
        "mean_run_length", "median_run_length", "max_run_length",
        "frac_runs_ge3", "p22",
    ]
    df_seasonal = df_seasonal[[c for c in season_cols if c in df_seasonal.columns]]

    run_cols = [
        "province_en", "province_cn", "season", "run_id",
        "start_day_index", "run_length", "mean_intensity_kwh",
    ]
    df_runs = df_runs[[c for c in run_cols if c in df_runs.columns]]

    path_annual = os.path.join(output_dir, "province_clustering_metrics.csv")
    path_seasonal = os.path.join(output_dir, "province_season_clustering.csv")
    path_runs = os.path.join(output_dir, "province_extreme_event_runs.csv")

    df_annual.to_csv(path_annual, index=False)
    df_seasonal.to_csv(path_seasonal, index=False)
    df_runs.to_csv(path_runs, index=False)

    print(f"\n=== M1 Event Clustering Results ===")
    print(f"Annual metrics : {path_annual}  ({len(df_annual)} provinces)")
    print(f"Seasonal metrics: {path_seasonal}  ({len(df_seasonal)} rows)")
    print(f"Run details     : {path_runs}  ({len(df_runs)} runs)")
    print()
    print(df_annual[["province_en", "n_runs", "mean_run_length",
                      "max_run_length", "p22", "best_fit_model"]].to_string())

    return df_annual, df_seasonal, df_runs


if __name__ == "__main__":
    main()
