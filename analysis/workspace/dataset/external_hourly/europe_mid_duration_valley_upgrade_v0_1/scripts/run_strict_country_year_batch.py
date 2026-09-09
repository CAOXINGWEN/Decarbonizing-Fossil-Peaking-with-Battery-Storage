#!/usr/bin/env python3
"""Parallel strict-country-year batch runner and result aggregator.

Every country-year is assigned an independent adapter run directory.  This
avoids concurrent writes to a shared CSV and makes interruption/resume safe.
The batch first supports a seven-node physical pilot and then the complete
53-configuration point screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
UPGRADE = HERE.parent
ADAPTER = HERE / "run_europe_duration_spectrum_pilot.py"
DEFAULT_PANEL = (
    UPGRADE
    / "processed"
    / "opsd_europe_hourly_eligible_country_years_2015_2019.csv.gz"
)
MODEL_ROOT = UPGRADE / "model_adapter"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def strict_pairs(panel_path: Path, flags: list[str]) -> list[tuple[str, int]]:
    columns = pd.read_csv(panel_path, nrows=0).columns.tolist()
    absent = [flag for flag in flags if flag not in columns]
    if absent:
        raise RuntimeError(f"Strict flags absent from panel: {absent}")
    use = ["country", "year", *flags]
    frame = pd.read_csv(panel_path, usecols=use)
    keep = pd.Series(True, index=frame.index)
    for flag in flags:
        keep &= frame[flag].map(as_bool)
    pairs = (
        frame.loc[keep, ["country", "year"]]
        .drop_duplicates()
        .sort_values(["country", "year"])
    )
    return [(str(row.country), int(row.year)) for row in pairs.itertuples()]


def run_one(
    pair: tuple[str, int],
    mode: str,
    panel_path: Path,
    flags: list[str],
    batch_id: str,
    max_duration_h: float | None,
    vre_energy_share: float | None,
    max_gas_pu: float | None,
    solve_chronologies: str | None,
    extreme_quantile: float | None,
    cluster_days: int | None,
) -> dict[str, Any]:
    country, year = pair
    run_id = f"{batch_id}/{country}_{year}"
    run_dir = MODEL_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ADAPTER),
        "--mode",
        mode,
        "--panel",
        str(panel_path),
        "--sample-flag",
        ",".join(flags),
        "--unit",
        "country_year",
        "--country-years",
        f"{country}:{year}",
        "--run-id",
        run_id,
        "--resume",
    ]
    if max_duration_h is not None:
        command.extend(["--max-duration-h", f"{max_duration_h:g}"])
    if vre_energy_share is not None:
        command.extend(["--vre-energy-share", f"{vre_energy_share:g}"])
    if max_gas_pu is not None:
        command.extend(["--max-gas-pu", f"{max_gas_pu:g}"])
    if solve_chronologies is not None:
        command.extend(["--solve-chronologies", solve_chronologies])
    if extreme_quantile is not None:
        command.extend(["--extreme-quantile", f"{extreme_quantile:g}"])
    if cluster_days is not None:
        command.extend(["--cluster-days", str(cluster_days)])
    started = time.time()
    result = subprocess.run(
        command,
        cwd=UPGRADE.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "country": country,
        "year": year,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "returncode": int(result.returncode),
        "elapsed_seconds": time.time() - started,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-4000:],
    }
    (run_dir / "batch_subprocess_receipt.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def concat_existing(
    run_dirs: list[Path], filename: str, required: bool = True
) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        path = run_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "source_run_dir", str(run_dir))
            frames.append(frame)
        elif required:
            raise FileNotFoundError(path)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def distribution(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "n": 0,
            "negative_n": 0,
            "negative_fraction": None,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    return {
        "n": int(len(clean)),
        "negative_n": int((clean < 0).sum()),
        "negative_fraction": float((clean < 0).mean()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
    }


def aggregate(
    receipts: list[dict[str, Any]],
    mode: str,
    panel_path: Path,
    flags: list[str],
    batch_id: str,
    elapsed_seconds: float,
    max_duration_h: float | None,
    vre_energy_share: float | None,
    max_gas_pu: float | None,
    solve_chronologies: str | None,
    extreme_quantile: float | None,
    cluster_days: int | None,
) -> dict[str, Any]:
    aggregate_dir = MODEL_ROOT / f"{batch_id}_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    receipt_frame = pd.DataFrame(receipts).sort_values(["country", "year"])
    receipt_frame.to_csv(aggregate_dir / "batch_receipts.csv", index=False)
    failures = receipt_frame[receipt_frame["returncode"] != 0]
    if len(failures):
        failures.to_csv(aggregate_dir / "failed_runs.csv", index=False)
        raise RuntimeError(
            f"{len(failures)} country-year runs failed; see "
            f"{aggregate_dir / 'failed_runs.csv'}"
        )
    run_dirs = [Path(path) for path in receipt_frame["run_dir"]]
    physical = concat_existing(run_dirs, "paired_physical_value_changes.csv")
    boundaries = concat_existing(run_dirs, "boundary_diagnostics.csv")
    chronology_qa = concat_existing(run_dirs, "chronology_conservation_qa.csv")
    envelopes = concat_existing(run_dirs, "duration_envelope.csv")
    configuration_metrics = concat_existing(
        run_dirs, "configuration_screen_metrics.csv"
    )
    physical.to_csv(
        aggregate_dir / "unit_configuration_paired_changes.csv", index=False
    )
    boundaries.to_csv(
        aggregate_dir / "all_boundary_diagnostics.csv", index=False
    )
    chronology_qa.to_csv(
        aggregate_dir / "all_chronology_conservation_qa.csv", index=False
    )
    envelopes.to_csv(aggregate_dir / "all_duration_envelopes.csv", index=False)
    configuration_metrics.to_csv(
        aggregate_dir / "all_configuration_screen_metrics.csv", index=False
    )

    delta_columns = {
        "EFC": "delta_efc_clustered_minus_dispersed",
        "joint_benefit": "delta_joint_benefit_clustered_minus_dispersed",
        "turnover_benefit": (
            "delta_turnover_benefit_clustered_minus_dispersed"
        ),
    }
    unit_rows = []
    for unit_id, group in physical.groupby("unit_id", sort=True):
        row: dict[str, Any] = {
            "unit_id": unit_id,
            "configuration_count": int(group["config_id"].nunique()),
        }
        for label, column in delta_columns.items():
            stats = distribution(group[column])
            for key, value in stats.items():
                row[f"{label}_{key}"] = value
        unit_rows.append(row)
    unit_summary = pd.DataFrame(unit_rows)
    unit_summary.to_csv(
        aggregate_dir / "country_year_direction_summary.csv", index=False
    )

    pooled = {label: distribution(physical[column]) for label, column in delta_columns.items()}
    point = boundaries[boundaries["full_configuration_set"].map(as_bool)]
    boundary_summary = {
        "chronology_unit_screens": int(len(point)),
        "point_screen_complete_valley_n": int(
            point["point_screen_complete_valley"].map(as_bool).sum()
        )
        if len(point)
        else 0,
        "confirmatory_complete_valley_n": 0,
        "h_T_censoring_status_counts": (
            point["h_T_censoring_status"].value_counts().to_dict()
            if len(point)
            else {}
        ),
        "h_I_censoring_status_counts": (
            point["h_I_censoring_status"].value_counts().to_dict()
            if len(point)
            else {}
        ),
        "nonconfirmation_reason_counts": (
            point["nonconfirmation_reasons"].value_counts().to_dict()
            if len(point)
            else {}
        ),
    }
    configuration_counts = []
    for run_dir in run_dirs:
        adapter_qa = json.loads((run_dir / "QA.json").read_text(encoding="utf-8"))
        configuration_counts.append(int(adapter_qa["configuration_count_run"]))
    if len(set(configuration_counts)) != 1:
        raise RuntimeError(
            f"Inconsistent configuration counts across runs: "
            f"{sorted(set(configuration_counts))}"
        )
    qa = {
        "batch_id": batch_id,
        "mode": mode,
        "panel": str(panel_path),
        "panel_sha256": sha256(panel_path),
        "sample_flags_AND": flags,
        "country_year_count": int(len(receipt_frame)),
        "configuration_count_per_country_year": configuration_counts[0],
        "maximum_duration_h": max_duration_h,
        "vre_energy_share": vre_energy_share,
        "maximum_gas_capacity_pu_mean_task": max_gas_pu,
        "chronologies_solved": (
            [
                item.strip()
                for item in solve_chronologies.split(",")
                if item.strip()
            ]
            if solve_chronologies
            else ["natural", "dispersed", "clustered"]
        ),
        "extreme_quantile": (
            extreme_quantile if extreme_quantile is not None else 0.75
        ),
        "cluster_days": cluster_days if cluster_days is not None else 4,
        "elapsed_seconds": elapsed_seconds,
        "subprocess_returncodes_all_zero": bool(
            (receipt_frame["returncode"] == 0).all()
        ),
        "all_country_year_adapter_qa_passed": True,
        "chronology_conservation_all_passed": bool(
            chronology_qa["day_multiset_matches_natural"].map(as_bool).all()
        ),
        "clustered_run_longer_than_dispersed_all_units": bool(
            (
                chronology_qa.pivot_table(
                    index=["unit_id", "year"],
                    columns="chronology",
                    values="mean_extreme_run_d",
                )["clustered"]
                > chronology_qa.pivot_table(
                    index=["unit_id", "year"],
                    columns="chronology",
                    values="mean_extreme_run_d",
                )["dispersed"]
            ).all()
        ),
        "pooled_direction_distributions": pooled,
        "pooled_direction_distributions_interpretation": (
            "Descriptive fixed configuration-design nodes only. The n values "
            "are not empirical sample sizes and must not be used for "
            "confidence intervals or significance tests. Empirical summaries "
            "must first reduce configurations within country-year."
        ),
        "boundary_summary": boundary_summary,
        "confirmatory_complete_valley_supported": False,
        "confirmatory_reason": (
            "This batch is a point physical/value screen without networked "
            "ELCC or hierarchical confidence intervals."
        ),
    }
    for run_dir in run_dirs:
        adapter_qa = json.loads((run_dir / "QA.json").read_text(encoding="utf-8"))
        qa["all_country_year_adapter_qa_passed"] &= bool(
            adapter_qa.get("passed", False)
        )
    qa["passed"] = bool(
        qa["subprocess_returncodes_all_zero"]
        and qa["all_country_year_adapter_qa_passed"]
        and qa["chronology_conservation_all_passed"]
        and qa["clustered_run_longer_than_dispersed_all_units"]
    )
    (aggregate_dir / "AGGREGATE_QA_AND_SUMMARY.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )

    lines = [
        f"# Strict European country-year {mode} summary",
        "",
        f"- Country-years: {qa['country_year_count']}.",
        f"- Configurations per country-year: "
        f"{qa['configuration_count_per_country_year']}.",
        f"- Batch QA passed: {qa['passed']}.",
        "",
        "## Clustered minus dispersed directions",
        "",
        (
            "The following counts describe fixed configuration-design nodes, "
            "not independent empirical observations. Country-year/country-"
            "block summaries are produced by the dedicated screen summarizer."
        ),
        "",
    ]
    for label, stats in pooled.items():
        lines.append(
            f"- {label}: {stats['negative_n']}/{stats['n']} negative "
            f"({stats['negative_fraction']:.1%}); median={stats['median']:.3f}; "
            f"mean={stats['mean']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Mid-duration-valley status",
            "",
            f"- Point screens with a complete valley: "
            f"{boundary_summary['point_screen_complete_valley_n']}/"
            f"{boundary_summary['chronology_unit_screens']}.",
            "- Confirmatory complete valleys: 0 by design. Networked ELCC "
            "and hierarchical interval criteria are outside this adapter.",
            "",
            "Negative EFC or turnover shifts support the narrower claim that "
            "clustering compresses repeat-use value. They do not alone prove "
            "a positive-negative-positive duration-value valley.",
        ]
    )
    (aggregate_dir / "AGGREGATE_INTERPRETATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if not qa["passed"]:
        raise RuntimeError(f"Aggregate QA failed: {aggregate_dir}")
    return qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "screen"), required=True)
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument(
        "--sample-flags",
        default="strict_confirmatory_eligible_country_year",
        help="Comma-separated boolean columns combined with AND",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-duration-h",
        type=float,
        help=(
            "Optional upper duration bound forwarded to screen runs; use 168 "
            "for the 50-node evidence-supported catalogue."
        ),
    )
    parser.add_argument(
        "--vre-energy-share",
        type=float,
        help=(
            "Optional value forwarded to the adapter. Leave unset to retain "
            "the adapter default. Use 0 only when the panel load field is "
            "itself an observed task series and wind/solar must not be "
            "subtracted from that task."
        ),
    )
    parser.add_argument(
        "--max-gas-pu",
        type=float,
        help=(
            "Optional numerical gas-capacity bound forwarded to the adapter. "
            "It must exceed the maximum task peak-to-mean ratio plus charging "
            "headroom and is audited for nonbindingness after the run."
        ),
    )
    parser.add_argument(
        "--solve-chronologies",
        help=(
            "Optional comma-separated chronology subset forwarded to the "
            "adapter. Use dispersed,clustered for a paired-only batch."
        ),
    )
    parser.add_argument(
        "--extreme-quantile",
        type=float,
        help=(
            "Optional scarcity-day quantile forwarded to the adapter. "
            "The primary specification is 0.75."
        ),
    )
    parser.add_argument(
        "--cluster-days",
        type=int,
        help=(
            "Optional target clustered-run length forwarded to the adapter. "
            "The primary specification is 4."
        ),
    )
    parser.add_argument("--batch-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel).resolve()
    flags = [item.strip() for item in args.sample_flags.split(",") if item.strip()]
    pairs = strict_pairs(panel_path, flags)
    if not pairs:
        raise RuntimeError("No country-years pass the requested strict flags")
    batch_id = args.batch_id or f"strict_{len(pairs)}cy_{args.mode}"
    started = time.time()
    receipts = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                pair,
                args.mode,
                panel_path,
                flags,
                batch_id,
                args.max_duration_h,
                args.vre_energy_share,
                args.max_gas_pu,
                args.solve_chronologies,
                args.extreme_quantile,
                args.cluster_days,
            ): pair
            for pair in pairs
        }
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            status = "ok" if receipt["returncode"] == 0 else "FAILED"
            print(
                f"{receipt['country']}-{receipt['year']} {status} "
                f"{receipt['elapsed_seconds']:.1f}s",
                flush=True,
            )
    qa = aggregate(
        receipts,
        mode=args.mode,
        panel_path=panel_path,
        flags=flags,
        batch_id=batch_id,
        elapsed_seconds=time.time() - started,
        max_duration_h=args.max_duration_h,
        vre_energy_share=args.vre_energy_share,
        max_gas_pu=args.max_gas_pu,
        solve_chronologies=args.solve_chronologies,
        extreme_quantile=args.extreme_quantile,
        cluster_days=args.cluster_days,
    )
    print(MODEL_ROOT / f"{batch_id}_aggregate")
    print(json.dumps(qa["pooled_direction_distributions"], indent=2))


if __name__ == "__main__":
    main()
