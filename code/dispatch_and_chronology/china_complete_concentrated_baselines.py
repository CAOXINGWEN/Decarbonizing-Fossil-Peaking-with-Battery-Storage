"""Extend the Chinese monthly clipping experiment to the six remaining load baselines."""

import argparse
import fcntl
import json
import os
from pathlib import Path

import china_mc as C
import numpy as np
import summarize_china


YEARS = list(range(2015, 2025))
BASELINES = (5, 10, 20, 25, 30, 35, 50)
RESULTS = C.WORK / "results"
ARCHIVE = RESULTS / "yearly_concentrated"
DESTINATION = RESULTS / "yearly_concentrated_all_q_20261120"


def read_json(path):
    return json.loads(path.read_text())


def configure():
    C.setup("fixed_2020")
    C.SCENARIO = "yearly_concentrated"
    C.OUT = DESTINATION
    C.QUANTILES = BASELINES
    archive_contract = read_json(ARCHIVE / "contract.json")
    archive_summary = read_json(ARCHIVE / "years_2015_2024/complete.json")
    assert archive_summary["source_contract_sha256"] == C.file_digest(ARCHIVE / "contract.json")
    assert archive_summary["baselines"] == [50] and archive_summary["placements"] == 310000
    japan = C.REV / "japan_baseline_mc_10612_20261118/code"
    files = [
        Path(__file__).with_name("run_reconstructed_mc.py"), Path(C.__file__),
        Path(C.model.__file__), japan / "compact_dispatch.py", japan / "run_mc.py",
        C.DATA / "records.csv", C.DATA / "china_2015_2024_load_mw.npy",
    ]
    hashes = {path.name: C.file_digest(path) for path in files}
    supplies = []
    for year in YEARS:
        chunk = ARCHIVE / "year_chunks" / str(year) / "contract.json"
        assert C.file_digest(chunk) == archive_contract["annual_contracts_sha256"][str(year)]
        contract = read_json(chunk)
        source = C.DATA / "reconstruction" / f"{year}_clipped.npy"
        receipt = source.with_name(f"{year}_complete.json")
        assert contract["source_sha256"] == hashes
        assert contract["input_resource_sha256"] == C.file_digest(source)
        assert contract["annual_calibration_sha256"] == C.file_digest(receipt)
        assert contract["resource"] == C.SCENARIO and contract["quantiles"] == [50]
        assert contract["seeds_per_record"] == C.NSEEDS and contract["seeds"] == [0, 999]
        assert contract["technical_cost_year"] == 2035 and contract["real_financing_rate"] == .036
        records = C.META[C.META.year == year]
        assert np.array_equal(records.province_idx, np.arange(31))
        assert np.array_equal(records.row_idx, (year - YEARS[0]) * 31 + np.arange(31))
        values = np.load(source)
        assert values.shape == (31, 8760) and np.isfinite(values).all() and (values >= 0).all()
        supplies.append(values)
    C.SUPPLY = np.concatenate(supplies)
    run_contract = dict(archive_contract, quantiles=list(BASELINES),
                        reused_q50_contract_sha256=archive_summary["source_contract_sha256"],
                        extension_script_sha256=C.file_digest(Path(__file__)))
    C.contract = lambda: run_contract


def reuse_q50():
    cases = DESTINATION / "cases"
    cases.mkdir(exist_ok=True)
    for idx in C.META.row_idx:
        target = C.case_path(int(idx), 50)
        source = ARCHIVE / "cases" / target.name
        if target.exists():
            assert os.path.samefile(source, target) or C.file_digest(source) == C.file_digest(target)
        else:
            os.link(source, target)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    DESTINATION.mkdir(exist_ok=True)
    with (DESTINATION / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure()
        reuse_q50()
        C.run(YEARS, args.workers)
        summarize_china.publish(YEARS)


if __name__ == "__main__":
    main()
