#!/usr/bin/env python3
"""Summarize the Chinese provincial and Japanese community applications."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PRICE_POST = HERE.parent / "price_channel" / "postprocess_price_channels_v0_1.py"
spec = importlib.util.spec_from_file_location("cross_scale_price_post", PRICE_POST)
if spec is None or spec.loader is None:
    raise RuntimeError(PRICE_POST)
post = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = post
spec.loader.exec_module(post)


def block_bootstrap(group: pd.DataFrame, block: str, value: str, draws: int = 10000) -> tuple[float, float, float]:
    blocks = group.groupby(block, as_index=False)[value].mean()[value].to_numpy(float)
    rng = np.random.default_rng(4702)
    estimates = blocks[rng.integers(0, len(blocks), size=(draws, len(blocks)))].mean(axis=1)
    return float(blocks.mean()), *np.quantile(estimates, [0.025, 0.975]).tolist()


def main() -> None:
    paths = sorted(HERE.glob("full_*_shard*/cross_scale_results.csv.gz"))
    if len(paths) != 4:
        raise RuntimeError(f"Expected four completed shards, found {len(paths)}")
    arms = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    cells, envelope, valleys = [], [], []
    for (case, case_id, year, chronology), group in arms.groupby(["application_case", "case_id", "case_year", "chronology"], sort=False):
        lfp = group[group.arm_type.eq("extra_lfp4_comparator")].iloc[0]
        candidates = group[group.arm_type.eq("ldes_candidate")].copy()
        candidates["benefit_eur_kw_year"] = (float(lfp.S0_system_cost_eur) - candidates.S0_system_cost_eur) / (candidates.tail_power_mw * 1000.0)
        cells.append(candidates)
        duration_rows = []
        for duration, duration_group in candidates.groupby("duration_h"):
            best = duration_group.sort_values("benefit_eur_kw_year", ascending=False).iloc[0]
            duration_rows.append({
                "application_case": case, "case_id": case_id, "case_year": year,
                "chronology": chronology, "duration_h": duration,
                "benefit_eur_kw_year": best.benefit_eur_kw_year,
                "dominant_config_id": best.config_id, "dominant_tech_id": best.tech_id,
            })
        envelope.extend(duration_rows)
        valleys.append({
            "application_case": case, "case_id": case_id, "case_year": year, "chronology": chronology,
            **post.strict_valley(pd.DataFrame(duration_rows)),
        })
    cells = pd.concat(cells, ignore_index=True)
    envelope = pd.DataFrame(envelope)
    valleys = pd.DataFrame(valleys)
    paired_cells = cells.pivot(index=["application_case", "case_id", "case_year", "config_id", "tech_id", "duration_h"], columns="chronology", values="benefit_eur_kw_year").reset_index()
    paired_cells["chronology_effect_eur_kw_year"] = paired_cells.clustered - paired_cells.dispersed
    paired_envelope = envelope.pivot(index=["application_case", "case_id", "case_year", "duration_h"], columns="chronology", values="benefit_eur_kw_year").reset_index()
    paired_envelope["chronology_effect_eur_kw_year"] = paired_envelope.clustered - paired_envelope.dispersed
    focal = paired_cells[paired_cells.tech_id.eq("caes") & paired_cells.duration_h.eq(24)].copy()
    focal["block"] = np.where(focal.application_case.eq("china"), focal.case_id, focal.case_id.str.split("-").str[1])
    summaries = []
    for case, group in focal.groupby("application_case"):
        estimate, low, high = block_bootstrap(group, "block", "chronology_effect_eur_kw_year")
        summaries.append({"application_case": case, "n_case_years": len(group), "mean_caes24_effect_eur_kw_year": estimate, "ci95_low": low, "ci95_high": high})
    duration_summary = []
    paired_envelope["block"] = np.where(paired_envelope.application_case.eq("china"), paired_envelope.case_id, paired_envelope.case_id.str.split("-").str[1])
    for (case, duration), group in paired_envelope.groupby(["application_case", "duration_h"]):
        estimate, low, high = block_bootstrap(group, "block", "chronology_effect_eur_kw_year")
        duration_summary.append({"application_case": case, "duration_h": duration, "mean_effect_eur_kw_year": estimate, "ci95_low": low, "ci95_high": high})
    valley_summary = valleys.groupby(["application_case", "chronology"], as_index=False).agg(n_case_years=("case_id", "size"), strict_valleys=("strict_valley", "sum"), mean_width_when_present=("valley_width_h", lambda x: x[x > 0].mean()))
    output = HERE / "summary"
    output.mkdir(exist_ok=True)
    arms.to_csv(output / "cross_scale_arms.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    cells.to_csv(output / "cross_scale_mix_cells.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    envelope.to_csv(output / "cross_scale_duration_envelope.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    paired_cells.to_csv(output / "cross_scale_paired_cells.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    paired_envelope.to_csv(output / "cross_scale_paired_duration_envelope.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    valleys.to_csv(output / "cross_scale_valley_records.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(summaries).to_csv(output / "cross_scale_caes24_summary.csv", index=False)
    pd.DataFrame(duration_summary).to_csv(output / "cross_scale_duration_summary.csv", index=False)
    valley_summary.to_csv(output / "cross_scale_valley_summary.csv", index=False)
    qa = {
        "applications": sorted(arms.application_case.unique()),
        "case_years": arms[["application_case", "case_id", "case_year"]].drop_duplicates().groupby("application_case").size().to_dict(),
        "expected_arms_per_case_chronology": 51,
        "complete_arm_groups": bool(arms.groupby(["application_case", "case_id", "case_year", "chronology"]).size().eq(51).all()),
    }
    (output / "qa.json").write_text(json.dumps(qa, indent=2) + "\n")


if __name__ == "__main__":
    main()
