#!/usr/bin/env python3
"""Prepare compact, inference-free v43 figure source tables.

This script performs deterministic tabulation only. Inferential summaries are
read from their registered experiment outputs and are not recomputed here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V43 = HERE.parent
SOURCE = V43 / "source_data"
RECOVERY = (
    V43
    / "experiments_signature"
    / "results_recovery_windows"
    / "recovery_windows.csv.gz"
)
RECOVERY_SUMMARY = (
    V43
    / "experiments_signature"
    / "results_recovery_windows"
    / "recovery_window_overall_summary.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recovery_window_ecdf() -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = pd.read_csv(RECOVERY)
    windows = windows[windows["target_cluster_days"].eq(4)].copy()
    rows: list[dict[str, object]] = []
    for chronology, group in windows.groupby("chronology", sort=True):
        counts = (
            group["recovery_hours"]
            .value_counts()
            .sort_index()
            .rename_axis("recovery_hours")
            .rename("window_count")
            .reset_index()
        )
        counts["chronology"] = chronology
        counts["ecdf"] = counts["window_count"].cumsum() / counts[
            "window_count"
        ].sum()
        rows.extend(counts.to_dict("records"))
    ecdf = pd.DataFrame(rows)[
        ["chronology", "recovery_hours", "window_count", "ecdf"]
    ]

    summary = pd.read_csv(RECOVERY_SUMMARY)
    summary = summary[
        summary["target_cluster_days"].eq(4)
        & summary["metric"].isin(
            [
                "median_recovery_hours",
                "zero_recovery_window_share",
                "median_available_recharge_share_of_annual",
            ]
        )
    ].copy()
    if len(summary) != 3:
        raise RuntimeError("Expected three registered 4-day recovery summaries")
    return ecdf, summary


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    ecdf, summary = recovery_window_ecdf()
    ecdf_path = SOURCE / "v43_figure2f_recovery_window_ecdf.csv"
    summary_path = SOURCE / "v43_figure2f_recovery_window_summary.csv"
    ecdf.to_csv(ecdf_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "analysis_id": "v43_compact_figure_source_preparation",
        "inputs": {
            str(RECOVERY): sha256(RECOVERY),
            str(RECOVERY_SUMMARY): sha256(RECOVERY_SUMMARY),
        },
        "outputs": {
            str(ecdf_path): sha256(ecdf_path),
            str(summary_path): sha256(summary_path),
        },
        "transforms": {
            "figure2f": (
                "Exact empirical CDF of recovery-window hours for the "
                "registered 4-day chronology intervention; inferential "
                "summaries are copied from the country-block bootstrap output."
            )
        },
    }
    manifest_path = SOURCE / "v43_compact_figure_source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
