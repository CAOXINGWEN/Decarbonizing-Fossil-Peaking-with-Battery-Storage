#!/usr/bin/env python3
"""Summarize the representative-country price-responsive dispatch experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "empirical_reoptimized" / "empirical_reoptimized_arms.csv.gz"
OUTPUT = HERE / "empirical_reoptimized_summary_v0_1"
COST = "S3b_empirical_reoptimized_system_cost_eur"


def block_interval(frame: pd.DataFrame, value: str, draws: int = 10_000) -> tuple[float, float, float]:
    blocks = frame.groupby("country", as_index=False)[value].mean()[value].to_numpy(float)
    rng = np.random.default_rng(4703)
    sampled = blocks[rng.integers(0, len(blocks), size=(draws, len(blocks)))].mean(axis=1)
    return float(blocks.mean()), *np.quantile(sampled, [0.025, 0.975]).tolist()


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    arms = pd.read_csv(SOURCE)
    cells = []
    choices = []
    for (country, year, chronology), group in arms.groupby(["country", "year", "chronology"], sort=False):
        lfp = group[group.arm_type.eq("extra_lfp4_comparator")].iloc[0]
        candidates = group[group.arm_type.eq("ldes_candidate")].copy()
        candidates["benefit_eur_kw_year"] = (float(lfp[COST]) - candidates[COST]) / (candidates.tail_power_mw * 1000.0)
        cells.append(candidates)
        selected = group.sort_values(COST, kind="stable").iloc[0]
        choices.append({
            "country": country, "year": year, "chronology": chronology,
            "config_id": selected.config_id, "tech_id": selected.tech_id,
            "duration_h": selected.duration_h, "arm_type": selected.arm_type,
        })
    cells = pd.concat(cells, ignore_index=True)
    paired = cells.pivot(
        index=["country", "year", "config_id", "tech_id", "duration_h"],
        columns="chronology", values="benefit_eur_kw_year",
    ).reset_index()
    paired["chronology_effect_eur_kw_year"] = paired.clustered - paired.dispersed
    caes = paired[paired.tech_id.eq("caes") & paired.duration_h.eq(24)].copy()
    estimate, low, high = block_interval(caes, "chronology_effect_eur_kw_year")
    caes_summary = pd.DataFrame([{
        "scenario": "S3b_empirical_reoptimized", "n_country_years": len(caes), "n_countries": caes.country.nunique(),
        "mean_effect_eur_kw_year": estimate, "ci95_low": low, "ci95_high": high,
    }])

    envelopes = []
    for (country, year, chronology, duration), group in cells.groupby(["country", "year", "chronology", "duration_h"]):
        best = group.loc[group.benefit_eur_kw_year.idxmax()]
        envelopes.append({
            "country": country, "year": year, "chronology": chronology, "duration_h": duration,
            "benefit_eur_kw_year": best.benefit_eur_kw_year, "tech_id": best.tech_id, "config_id": best.config_id,
        })
    envelope = pd.DataFrame(envelopes)
    paired_envelope = envelope.pivot(index=["country", "year", "duration_h"], columns="chronology", values="benefit_eur_kw_year").reset_index()
    paired_envelope["chronology_effect_eur_kw_year"] = paired_envelope.clustered - paired_envelope.dispersed
    duration_summary = []
    for duration, group in paired_envelope.groupby("duration_h"):
        estimate, low, high = block_interval(group, "chronology_effect_eur_kw_year")
        duration_summary.append({"duration_h": duration, "mean_effect_eur_kw_year": estimate, "ci95_low": low, "ci95_high": high})

    choice = pd.DataFrame(choices).pivot(index=["country", "year"], columns="chronology", values=["config_id", "tech_id", "duration_h", "arm_type"]).reset_index()
    choice.columns = ["_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else col for col in choice.columns]
    choice["identity_flip"] = choice.config_id_clustered != choice.config_id_dispersed
    choice["duration_flip"] = choice.duration_h_clustered != choice.duration_h_dispersed
    choice["participation_flip"] = choice.arm_type_clustered != choice.arm_type_dispersed

    caes_summary.to_csv(OUTPUT / "s3b_caes24_summary.csv", index=False)
    pd.DataFrame(duration_summary).to_csv(OUTPUT / "s3b_duration_summary.csv", index=False)
    choice.to_csv(OUTPUT / "s3b_choice_flips.csv", index=False)
    paired.to_csv(OUTPUT / "s3b_paired_cells.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (OUTPUT / "qa.json").write_text(json.dumps({
        "country_years": int(arms[["country", "year"]].drop_duplicates().shape[0]),
        "countries": sorted(arms.country.unique().tolist()), "configuration_nodes": 49,
        "identity_flips": int(choice.identity_flip.sum()), "participation_flips": int(choice.participation_flip.sum()),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
