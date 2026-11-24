"""Compare both charging reconstructions on matched province-years and event placements."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


WORK = Path(__file__).resolve().parents[1]
RESULTS = WORK / "results"
YEARS = "years_2015_2024"
BASELINES = (5, 10, 20, 25, 30, 35, 50)
REPORTS = WORK.parent / "中日基线与蒙特卡洛实验报告_20261118"
OUTPUT = REPORTS / "中国_两种充电方法_七个负荷基线_混合组合胜出比例.csv"


def input_tables(scenario, baselines):
    folder = RESULTS / scenario / YEARS
    complete = json.loads((folder / "complete.json").read_text())
    assert complete["status"] == "complete" and complete["baselines"] == list(baselines)
    assert complete["annual_records"] == 310 and complete["seeds_per_record_baseline"] == 1000
    assert complete["placements"] == 310000 * len(baselines)
    assert complete["max_direct_B_error"] < 1e-6 and complete["max_compact_original_B_error"] < 1e-6
    for filename in ("china_records.csv.gz", "china_baseline_summary.csv", "china_seed_matrices.h5"):
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == complete["files_sha256"][filename]
    frame = pd.read_csv(folder / "china_records.csv.gz", dtype={"uid": str})
    assert len(frame) == 310 * len(baselines) and not frame.duplicated(["q", "uid"]).any()
    frame = frame.set_index(["q", "row_idx"]).sort_index()
    summary = pd.read_csv(folder / "china_baseline_summary.csv")
    return frame, summary[summary.cohort == "all_31"].set_index("q"), folder / "china_seed_matrices.h5"


def verify_axes(h5, records):
    assert np.array_equal(h5["row_idx"][:], np.arange(310))
    assert np.array_equal(h5["seed"][:], np.arange(1000))
    assert np.array_equal(h5["uid"][:].astype(str), records.uid.to_numpy())
    assert h5.attrs["technology_cost_year"] == 2035 and h5.attrs["real_financing_rate"] == .036


def wins(h5, q, records, summary):
    b = h5[f"q{q:02d}/B"][:]
    assert b.shape == (310, 1000) and np.isfinite(b).all()
    assert np.isfinite(records.real_B).all() and np.array_equal(records.index, np.arange(310))
    assert np.allclose(records.lfp_cost - records.hybrid_cost, records.real_B, atol=1e-8, rtol=0)
    original, draws = records.real_B.to_numpy() > 0, b > 0
    assert np.array_equal(draws.sum(axis=1), records.mc_winning_draws)
    assert original.sum() == summary.original_wins and draws.sum() == summary.mc_wins
    assert summary.annual_records == original.size and summary.placements == draws.size
    return original, draws


def compare(label, prop, clipped):
    n = prop.size
    p, c = int(prop.sum()), int(clipped.sum())
    prop_only = int((prop & ~clipped).sum())
    clip_only = int((clipped & ~prop).sum())
    assert c - p == clip_only - prop_only
    return {
        f"{label}总数": n,
        f"{label}按出力比例分配胜出数": p,
        f"{label}按出力比例分配胜出率(%)": round(100 * p / n, 6),
        f"{label}月度削峰胜出数": c,
        f"{label}月度削峰胜出率(%)": round(100 * c / n, 6),
        f"{label}胜出率差(月度削峰减比例分配,百分点)": round(100 * (c - p) / n, 6),
        f"{label}仅按出力比例分配胜出数": prop_only,
        f"{label}仅月度削峰胜出数": clip_only,
    }


def main():
    prop, prop_summary, prop_file = input_tables("yearly_surplus", BASELINES)
    clip, clip_summary, clip_file = input_tables("yearly_concentrated_all_q_20261120", BASELINES)
    old, _, old_file = input_tables("yearly_concentrated", (50,))
    task_fields = ["uid", "province", "year", "ref_power_mw", "load_mwh", "task_mwh", "task_hours",
                   "events", "event_days", "mean_event_days", "longest_event_days"]
    pd.testing.assert_frame_equal(prop[task_fields], clip[task_fields], check_exact=True)
    pd.testing.assert_frame_equal(clip.loc[50], old.loc[50], check_exact=True)
    rows = []
    with h5py.File(prop_file) as p, h5py.File(clip_file) as c, h5py.File(old_file) as archived:
        for matrix, frame in ((p, prop), (c, clip), (archived, old)):
            verify_axes(matrix, frame.loc[50])
        for name in ("B", "long_windows"):
            assert np.array_equal(c[f"q50/{name}"][:], archived[f"q50/{name}"][:])
        for q in BASELINES:
            verify_axes(p, prop.loc[q])
            verify_axes(c, clip.loc[q])
            p_real, p_mc = wins(p, q, prop.loc[q], prop_summary.loc[q])
            c_real, c_mc = wins(c, q, clip.loc[q], clip_summary.loc[q])
            rows.append({"负荷基线": f"q{q}", **compare("原始日期", p_real, c_real),
                         **compare("蒙特卡洛", p_mc, c_mc)})
            print(q, int(p_real.sum()), int(c_real.sum()), int(p_mc.sum()), int(c_mc.sum()),
                  "original switches", np.count_nonzero(p_real != c_real),
                  "min |B|", min(np.abs(p[f"q{q:02d}/B"][:]).min(), np.abs(c[f"q{q:02d}/B"][:]).min()))
    table = pd.DataFrame(rows)
    temporary = OUTPUT.with_suffix(".partial.csv")
    table.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
