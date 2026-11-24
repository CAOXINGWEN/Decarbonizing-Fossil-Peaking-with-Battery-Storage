# The economic value of long-duration energy storage under stochastic balancing requirements

Public source data, 58-configuration storage technology parameter catalog, and reproduction code accompanying the *Joule* manuscript:

> **Xingwen Cao** and **Haoran Zhang**\*  
> *The economic value of long-duration energy storage under stochastic balancing requirements*  
> Lead Contact & Correspondence: Haoran Zhang ([h.zhang@pku.edu.cn](mailto:h.zhang@pku.edu.cn))

---

## Overview

This repository archives the processed source data, technology physical and cost parameter catalogs, and analysis/visualization scripts underlying all main-text and supplemental results across **11,054 complete annual hourly grid records**:
- **Chinese provincial power systems (`n = 310`, 31 mainland provinces, 2015–2024)**: electricity load peaking tasks paired with annual wind and solar curtailment budgets (`178` positive-curtailment and `132` zero-curtailment province-years; generation-proportional and monthly peak-clipping reconstructions).
- **European national grids (`n = 132`, 15 countries, 2015–2024)**: gas-fired generation peaking tasks paired with synchronous wind and solar surpluses.
- **Japanese community grids (`n = 10,612`, 5,306 500-m community mesh grids across 5 Kanto–Shizuoka prefectures, FY2023–FY2024)**: low-voltage grid import peaking tasks paired with daytime rooftop PV exports.
- **External validations & extensions**: 75 European historical weather years (`1950/51–2024/25`, PECD v4.2 / ERA5), 32 US regional electricity market records (`CISO`, `ERCO`, `MISO`, EIA-930), 58 storage configurations across 8 technologies (`4–720 h` rated duration, `641,132` full-year optimal dispatch evaluations), and multi-year endogenous capacity planning (`mean` and `CVaR90`).

---

## Repository Structure

```text
├── source_data/
│   ├── figures/                  # Source data CSVs for Main Figures 1–5, Table 1, and Figures S1–S4, S20–S21
│   ├── tables/                   # Processed record-level evaluation tables (11,054 main + external records,
│   │                             #   conditional date rearrangements, 1–12 d event extensions, window recursion,
│   │                             #   equal-energy constant-power service valuation, price/rule/threshold sweeps)
│   ├── technology_catalog_58/    # Complete 641,132-row (11,054 × 58) catalog results & multi-year planning tables
│   └── robustness_checks/        # Japanese 5-prefecture regional aggregation, Netted-all extension & VRE depth checks
├── parameters/
│   ├── parameters.json           # Canonical physical & cost parameters (2025, 2030, 2035) across all 8 technologies
│   ├── cost_catalogue_2025_2035.csv
│   ├── technology_catalogue.json # 58-configuration catalog specification (LFP, VRFB, Fe-Cr, Gravity, TES, A-CAES, LAES, 200 bar H2)
│   ├── technology_costs_2035.csv # 2035 annualized fixed-cost breakdown at 3.6% real cost of capital (Table S3)
│   └── source_library/           # Multi-source technology cost learning curves & replacement-aware parameter library
├── code/
│   ├── preprocessing_and_regimes/# Input series construction, baseline extraction, curtailment reconstruction & demand validation
│   ├── dispatch_and_chronology/  # Hourly LP dispatch optimization, 11.054M intra-seasonal date rearrangements,
│   │                             #   1–12 d event extensions & 2×2×2 charging-window factorial experiments
│   ├── technology_catalog_and_capacity/ # 58-configuration solver, multi-year CVaR90 capacity planning, near-optimal
│   │                             #   duration intervals & equal-energy constant-power service valuation
│   ├── robustness/               # Prefectural aggregation, service-eligibility extension & penetration transition scripts
│   └── figures/                  # Exact Matplotlib/vector scripts generating Main Figures 1–5, Table 1 & Figures S1–S21
├── reproducibility/
│   ├── verify_joule_results.py   # Fast standalone verification of headline manuscript statistics (no solver license needed)
│   └── requirements.txt          # Python package dependencies
├── verification/
│   └── joule_headline_verification.json
├── metadata/
│   ├── analysis_entrypoints.json # Stage-by-stage script index
│   └── file_provenance.csv       # Byte sizes and SHA-256 hashes of all archived files
├── CITATION.cff
├── LICENSE
└── checksums.sha256
```

---

## Quick Verification (No Commercial Solver Required)

With Python 3.10+ and `numpy`, `pandas`, and `scipy` installed (`pip install -r reproducibility/requirements.txt`), run from the repository root:

```bash
python reproducibility/verify_joule_results.py
```

This script verifies directly from `source_data/` and `parameters/`:
1. Exact sample counts across all three regimes (`310` Chinese province-years with `178` positive-curtailment records, `132` European country-years, and `10,612` Japanese community-years; `N = 11,054`).
2. Historical win counts and shares at 2035 technology costs and a 3.6% real cost of capital (`67/310 = 21.6%`, `24/132 = 18.2%`, `1,468/10,612 = 13.8%`).
3. Mean hybrid win rates across `11,054,000` intra-seasonal conditional date rearrangements (`22%` in China, `34%` in Europe, `56%` in Japan).
4. Negative median net savings `B < 0` across all three regimes once continuous peaking calls reach `3 days` in controlled event-duration extensions.
5. Completeness of the `641,132` (`11,054 × 58`) technology catalog evaluations and convergence of cross-technology least-cost selections onto A-CAES or LFP.
6. Rank consistency between baseline net system cost savings `B` and equal-energy constant-power service valuation (Spearman `|ρ| = 0.88`, `0.78`, and `1.00` in China, Europe, and Japan).

---

## Reproducing Figures and Full Optimization Reruns

- **Figure & Table Generation**: All main-text and supplemental figure scripts reside in `code/figures/` (using processed CSVs in `source_data/`). Set `JOULE_MODE=1` to export exact `174.0 mm` Cell Press *Joule* vector PDFs, SVGs, and 600 DPI PNGs/TIFFs.
- **Full Dispatch & Capacity-Planning Optimization**: Full LP/MIP re-optimization across the 11.054 million date rearrangements, 58-configuration catalog, and multi-year CVaR90 capacity expansion uses `scipy.optimize.linprog` (HiGHS), Cardinal Optimizer (`coptpy`), or Gurobi (`gurobipy`); entry points are indexed in `metadata/analysis_entrypoints.json`.

---

## Data Governance and Access

Public datasets for European power grids (Fraunhofer ISE Energy-Charts), historical meteorological reanalyses (Copernicus ERA5 / PECD v4.2, ENTSO-E ERAA), US balancing authorities (EIA-930), and storage technology cost catalogs (NREL ATB, PNNL ESGC, US DOE, Danish Energy Agency, CNESA) are available from their original providers as cited in the manuscript. Raw hourly provincial electricity load and curtailment records for China and raw community meter/grid records for Japan are governed by provider access agreements and are not redistributed in raw hourly form; all processed annual/daily source-data tables required to reproduce every figure, table, and statistical test in the paper are included in `source_data/`. Requests for additional information should be directed to the Lead Contact, Prof. Haoran Zhang ([h.zhang@pku.edu.cn](mailto:h.zhang@pku.edu.cn)).
