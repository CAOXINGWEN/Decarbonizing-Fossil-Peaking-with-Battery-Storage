"""Shared provenance, finance summaries and bounded execution for the supplemental suites."""
from pathlib import Path
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"
WORK = Path(__file__).resolve().parents[3]
REV = WORK.parent
sys.path.insert(0, str(WORK / "code"))
import china_mc as C
import numpy as np
import pandas as pd
import psutil

M = C.model
RM = M.RM
VERSION = ".".join(str(getattr(M.cp.COPT, "VERSION_" + s)) for s in ("MAJOR", "MINOR", "TECHNICAL"))
assert VERSION == "8.0.7"
ROOT = M.ROOT


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def signatures(paths):
    return {os.path.relpath(p, ROOT): sha(p) for p in sorted(set(map(Path, paths)))}


def contract(out, evidence):
    path = out / "run_contract.json"
    if path.exists() and json.loads(path.read_text()) != evidence:
        raise ValueError(f"Saved checkpoints have a different contract: {path}")
    write_json(path, evidence)


def worker_init(max_as):
    resource.setrlimit(resource.RLIMIT_AS, (max_as, max_as))
    C.init_worker()


def parallel(out, tasks, function, workers):
    """Bound aggregate worker address space as well as sampling full tree RSS."""
    start = time.monotonic()
    workers = min(M.limit_workers(workers), max(1, len(tasks)))
    max_as = int(24e9 / workers)
    peak = 0
    proc = psutil.Process()
    def sample():
        nonlocal peak
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        peak = max(peak, rss)
        if rss >= 29e9:
            raise MemoryError(f"Aggregate RSS reached {rss} bytes")
    sample()
    done = 0
    previous = start
    with mp.get_context("fork").Pool(workers, initializer=worker_init, initargs=(max_as,), maxtasksperchild=12) as pool:
        iterator = pool.imap_unordered(function, tasks, chunksize=1)
        while done < len(tasks):
            try:
                iterator.next(timeout=.4)
                done += 1
            except mp.TimeoutError:
                pass
            sample()
            if time.monotonic() - previous > 20 or done == len(tasks):
                previous = time.monotonic()
                status = dict(finished=done, scheduled=len(tasks), workers=workers, peak_rss_bytes=peak,
                              elapsed_seconds=round(previous-start, 2))
                write_json(out/"progress.json", status)
                print(json.dumps(status), flush=True)
    return dict(workers=workers, worker_address_space_limit_bytes=max_as, peak_rss_bytes=peak,
                elapsed_seconds=round(time.monotonic()-start, 2), cases_processed=done)


def finance(records, b, windows, q, group, ranks):
    """Use the 47-column registered reporting schema. Window metrics use the requested definitions."""
    op = records.B.to_numpy() + RM.PREM36[2035]
    op_mc = b + RM.PREM36[2035]
    window_med = np.median(windows, axis=1)
    rows = []
    for yr in RM.YEARS:
        roi = RM.roi(op, yr)
        mc = RM.roi(op_mc, yr)
        quantiles = np.array([RM.ordered_quantile(v, [.05,.5,.95]) for v in mc])
        lo, mid, hi = quantiles.T
        valid = np.isfinite(lo) & np.isfinite(hi)
        finite = np.isfinite(roi)
        rmed, mmed = RM.ordered_quantile(roi,[.5])[0], RM.ordered_quantile(mid,[.5])[0]
        row = dict(group=group, q=q, year=yr, n=len(roi), real_roi_med=rmed, mc_roi_med=mmed,
                   real_med=rmed, mc_med=mmed, real_min=min(roi[finite],default=np.nan),
                   real_max=max(roi[finite],default=np.nan), real_none=int((~finite).sum()),
                   mc_none_share=(~np.isfinite(mc)).mean(), mc_width_med=np.median(hi[valid]-lo[valid]) if valid.any() else np.nan,
                   mc_width_n=int(valid.sum()), real_full_med=records.hybrid_window_full.median(),
                   mc_full_med=np.median(window_med))
        for rate in RM.RATES:
            threshold = RM.premium(yr,rate/100)
            row[f"real_win_{rate:g}"] = int((op >= threshold).sum())
            row[f"mc_win_{rate:g}"] = (op_mc >= threshold).mean(axis=1).mean()
        probability = (op_mc >= RM.PREM36[yr]).mean(axis=1)
        row.update(disagree_36=int(((probability>.5)!=(op>=RM.PREM36[yr])).sum()),
                   sensitive_36=int(((probability>.05)&(probability<.95)).sum()), real_pct_med=np.median(ranks))
        rows.append(row)
        if yr == 2035:
            placements = pd.DataFrame(dict(uid=records.uid.to_numpy(), mc_q05=lo, mc_q50=mid,
                mc_q95=hi, mc_win36=probability, real_pct=ranks, roi_real=roi, group=group,
                lw_real=records.long_windows.to_numpy(), lw_mc_med=window_med))
    summary = pd.DataFrame(rows)
    other = REV/"japan_5309_audit_20261117/results/full_10612/japan_primary_summary.csv"
    assert summary.columns.tolist() == pd.read_csv(other, nrows=0).columns.tolist() and summary.shape[1] == 47
    return summary, placements
