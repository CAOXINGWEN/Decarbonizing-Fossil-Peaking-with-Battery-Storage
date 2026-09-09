#!/usr/bin/env python3
"""Revalue completed cluster-intensity arms with the solver objective.

The optimization runs already freeze ``system_cost_eur`` for every arm.  This
script therefore does not re-solve or alter any schedule.  It replaces the
legacy EFC-derived benefit with the exact pairwise objective difference

    C_sys(additional 4 h LFP) - C_sys(LDES candidate),

and writes a parallel, auditable result tree.  The original run outputs remain
untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs"
OUT = HERE / "runs_objective_aligned"
PARITY_THRESHOLD = 5.2960380996609935
GROUP = [
    "regime",
    "system_id",
    "observation_year",
    "inference_block",
    "target_cluster_days",
]
GEOMETRY = [
    "event_days_dispersed",
    "longest_event_run_days_dispersed",
    "zero_recovery_window_share_dispersed",
    "median_recovery_hours_dispersed",
    "mean_recovery_hours_dispersed",
    "event_days_clustered",
    "longest_event_run_days_clustered",
    "zero_recovery_window_share_clustered",
    "median_recovery_hours_clustered",
    "mean_recovery_hours_clustered",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_gzip(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def revalue_shard(directory: Path) -> dict[str, object]:
    arms_path = directory / "arm_results.csv.gz"
    effects_path = directory / "system_duration_effects.csv.gz"
    arms = pd.read_csv(arms_path)
    old_effects = pd.read_csv(effects_path)
    keys = [*GROUP, "chronology"]

    comparator = arms.loc[
        arms["arm_type"].eq("extra_lfp4_comparator"),
        [*keys, "system_cost_eur"],
    ].rename(columns={"system_cost_eur": "comparator_system_cost_eur"})
    candidates = arms.loc[arms["arm_type"].eq("ldes_candidate")].merge(
        comparator,
        on=keys,
        validate="many_to_one",
    )
    candidates["benefit_vs_extra_lfp4_eur_kw_year"] = (
        candidates["comparator_system_cost_eur"] - candidates["system_cost_eur"]
    ) / (candidates["tail_power_mw"] * 1000.0)
    envelope = (
        candidates.sort_values(
            [*keys, "duration_h", "benefit_vs_extra_lfp4_eur_kw_year"],
            ascending=[True] * (len(keys) + 1) + [False],
            kind="stable",
        )
        .groupby([*keys, "duration_h"], as_index=False, sort=True)
        .first()
    )

    keep = [
        *GROUP,
        "duration_h",
        "config_id",
        "tech_id",
        "benefit_vs_extra_lfp4_eur_kw_year",
        "tail_equivalent_full_cycles",
    ]
    dispersed = envelope.loc[envelope["chronology"].eq("dispersed"), keep]
    clustered = envelope.loc[envelope["chronology"].eq("clustered"), keep]
    paired = dispersed.merge(
        clustered,
        on=[*GROUP, "duration_h"],
        suffixes=("_dispersed", "_clustered"),
        validate="one_to_one",
    )
    geometry = old_effects[[*GROUP, "duration_h", *GEOMETRY]].drop_duplicates()
    paired = paired.merge(
        geometry,
        on=[*GROUP, "duration_h"],
        validate="one_to_one",
    )
    paired["sequencing_effect_eur_kw_year"] = (
        paired["benefit_vs_extra_lfp4_eur_kw_year_clustered"]
        - paired["benefit_vs_extra_lfp4_eur_kw_year_dispersed"]
    )
    paired["absolute_sequencing_shift_eur_kw_year"] = paired[
        "sequencing_effect_eur_kw_year"
    ].abs()
    paired["reaches_cost_scale"] = paired[
        "absolute_sequencing_shift_eur_kw_year"
    ].ge(PARITY_THRESHOLD)

    ordered = [
        "duration_h",
        "config_id_dispersed",
        "tech_id_dispersed",
        "benefit_vs_extra_lfp4_eur_kw_year_dispersed",
        "tail_equivalent_full_cycles_dispersed",
        "config_id_clustered",
        "tech_id_clustered",
        "benefit_vs_extra_lfp4_eur_kw_year_clustered",
        "tail_equivalent_full_cycles_clustered",
        *GROUP,
        *GEOMETRY,
        "sequencing_effect_eur_kw_year",
        "absolute_sequencing_shift_eur_kw_year",
        "reaches_cost_scale",
    ]
    paired = paired[ordered].sort_values(
        ["regime", "system_id", "observation_year", "target_cluster_days", "duration_h"]
    )

    output = OUT / directory.name
    output.mkdir(parents=True, exist_ok=True)
    write_gzip(paired, output / "system_duration_effects.csv.gz")
    maximum_cluster1 = float(
        paired.loc[
            paired["target_cluster_days"].eq(1), "sequencing_effect_eur_kw_year"
        ].abs().max()
    )
    qa: dict[str, object] = {
        "status": "pass",
        "source_shard": directory.name,
        "source_arm_results": str(arms_path.resolve()),
        "source_arm_results_sha256": sha256(arms_path),
        "source_effect_geometry": str(effects_path.resolve()),
        "source_effect_geometry_sha256": sha256(effects_path),
        "source_rows": int(len(old_effects)),
        "output_rows": int(len(paired)),
        "value_metric": "C_sys(additional 4 h LFP)-C_sys(LDES candidate)",
        "efc_role": "operational diagnostic only",
        "maximum_absolute_cluster1_effect_eur_kw_year": maximum_cluster1,
        "all_effects_finite": bool(
            np.isfinite(paired["sequencing_effect_eur_kw_year"]).all()
        ),
    }
    if (
        qa["source_rows"] != qa["output_rows"]
        or maximum_cluster1 > 1.0e-9
        or not qa["all_effects_finite"]
    ):
        qa["status"] = "fail"
    (output / "qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return qa


def main() -> None:
    directories = sorted(
        path for path in SOURCE.iterdir() if (path / "arm_results.csv.gz").exists()
    )
    if not directories:
        raise RuntimeError(f"No completed runs found under {SOURCE}")
    results = [revalue_shard(path) for path in directories]
    summary = {
        "status": "pass" if all(item["status"] == "pass" for item in results) else "fail",
        "shards": len(results),
        "source_tree": str(SOURCE.resolve()),
        "output_tree": str(OUT.resolve()),
        "value_metric": "exact pairwise solver-system-cost difference",
        "original_outputs_modified": False,
    }
    (OUT / "qa.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise RuntimeError(summary)


if __name__ == "__main__":
    main()
