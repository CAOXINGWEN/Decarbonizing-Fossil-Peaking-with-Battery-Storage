#!/usr/bin/env python3
"""Selectively extract the ERAA 2025 members required by the 2035 phase-1 model.

The script reads only the remote ZIP central directories and the compressed byte
ranges of selected members. It does not download or modify the concurrent full
archive downloads. Every extracted member is verified against the CRC and
uncompressed size stored in the official ZIP central directory.
"""

from __future__ import annotations

import argparse
import binascii
from bisect import bisect_right
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urlparse
import zipfile
import zlib

import fsspec
import requests


BASE_URL = (
    "https://eepublicdownloads.blob.core.windows.net/"
    "public-cdn-container/clean-documents/sdc-documents/ERAA"
)

ARCHIVES = {
    "weather": {
        "filename": "ERAA_2025_PECDWeather.zip",
        "url": f"{BASE_URL}/ERAA_2025_PECDWeather.zip",
        "expected_size": 2_144_730_060,
        "expected_count": 1,
    },
    "demand": {
        "filename": "ERAA_2025_DemandData.zip",
        "url": f"{BASE_URL}/ERAA_2025_DemandData.zip",
        "expected_size": 442_714_786,
        "expected_count": 52,
    },
    "pecdres": {
        "filename": "ERAA_2025_PECDRES.zip",
        "url": f"{BASE_URL}/ERAA_2025_PECDRES.zip",
        "expected_size": 1_274_578_535,
        "expected_count": 455,
    },
}

WEATHER_MEMBER = "PECD - weather/WeatherScenarios_Mapping.xlsx"
DEMAND_BTM_2035_PREFIX = (
    "Demand data/Behind the meter PV-Battery system additions/"
)
DEMAND_BTM_2035_COUNT = 8
LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_SIGNATURE = 0x04034B50

MANIFEST_FIELDS = [
    "archive_name",
    "archive_url",
    "archive_size",
    "archive_etag",
    "archive_last_modified",
    "member_path",
    "output_path",
    "member_crc32",
    "compress_type",
    "compressed_size",
    "uncompressed_size",
    "compressed_sha256",
    "sha256",
    "source_transport",
    "status",
    "downloaded_at_utc",
    "error",
]


@dataclass(frozen=True)
class ArchiveMetadata:
    name: str
    filename: str
    url: str
    size: int
    etag: str
    last_modified: str


@dataclass(frozen=True)
class MemberTask:
    archive: ArchiveMetadata
    member_path: str
    header_offset: int
    crc32: int
    compress_type: int
    compressed_size: int
    uncompressed_size: int
    flag_bits: int
    priority: int


class RangeCache:
    """Read completed byte ranges produced by a concurrent ranged downloader."""

    def __init__(self, directory: Path, archive: ArchiveMetadata):
        self.directory = directory
        metadata_path = directory / "ranges.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        same_object_path = (
            normalized_object_path(str(metadata["url"]))
            == normalized_object_path(archive.url)
        )
        if int(metadata["size"]) != archive.size or not same_object_path:
            raise ValueError(f"range cache does not match archive: {directory}")
        self.jobs = sorted(metadata["jobs"], key=lambda item: int(item["start"]))
        self.starts = [int(item["start"]) for item in self.jobs]

    @property
    def label(self) -> str:
        return f"range_cache:{self.directory.name}"

    def read(self, start: int, end: int) -> bytes | None:
        if end < start:
            return b""
        index = max(bisect_right(self.starts, start) - 1, 0)
        position = start
        chunks: list[bytes] = []
        while position <= end and index < len(self.jobs):
            job = self.jobs[index]
            job_start = int(job["start"])
            job_end = int(job["end"])
            if position < job_start or position > job_end:
                return None
            path = self.directory / str(job["name"])
            expected_size = job_end - job_start + 1
            try:
                if path.stat().st_size != expected_size:
                    return None
                read_start = position - job_start
                read_end = min(end, job_end) - job_start
                with path.open("rb") as handle:
                    handle.seek(read_start)
                    chunk = handle.read(read_end - read_start + 1)
                if len(chunk) != read_end - read_start + 1:
                    return None
            except FileNotFoundError:
                return None
            chunks.append(chunk)
            position = job_start + read_end + 1
            index += 1
        if position != end + 1:
            return None
        return b"".join(chunks)


class FileRangeCache:
    """Read ranges from a completed primary or edge copy of an official ZIP."""

    def __init__(self, path: Path, archive: ArchiveMetadata):
        self.path = path
        self.archive = archive

    @property
    def label(self) -> str:
        return f"local_archive:{self.path.name}"

    def read(self, start: int, end: int) -> bytes | None:
        try:
            if self.path.stat().st_size != self.archive.size:
                return None
            with self.path.open("rb") as handle:
                handle.seek(start)
                chunk = handle.read(end - start + 1)
        except FileNotFoundError:
            return None
        if len(chunk) != end - start + 1:
            return None
        return chunk


_thread_local = threading.local()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalized_object_path(url: str) -> str:
    """Return the canonical Azure object path across blob/CDN host aliases."""

    path = urlparse(url).path
    prefix = "/public-cdn-container"
    return path[len(prefix) :] if path.startswith(prefix + "/") else path


def session() -> requests.Session:
    value = getattr(_thread_local, "session", None)
    if value is None:
        value = requests.Session()
        value.headers.update({"User-Agent": "ERAA-2025-selective-extractor/1.0"})
        _thread_local.session = value
    return value


def safe_member_path(member: str) -> PurePosixPath:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe ZIP member path: {member!r}")
    return path


def head_archive(
    name: str, spec: dict[str, object], timeout: float, retries: int
) -> ArchiveMetadata:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.head(str(spec["url"]), timeout=(30, timeout))
            response.raise_for_status()
            size = int(response.headers["Content-Length"])
            if size != int(spec["expected_size"]):
                raise ValueError(
                    f"{name}: remote size {size} != expected {spec['expected_size']}"
                )
            return ArchiveMetadata(
                name=name,
                filename=str(spec["filename"]),
                url=str(spec["url"]),
                size=size,
                etag=response.headers.get("ETag", "").strip('"'),
                last_modified=response.headers.get("Last-Modified", ""),
            )
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"HEAD failed for {name}: {error}") from error


def select_member(
    name: str, member: str, include_demand_btm: bool = False
) -> tuple[bool, int]:
    if name == "weather":
        return member == WEATHER_MEMBER, 0
    if name == "demand":
        total_demand = (
            member.startswith("Demand data/Demand timeseries/")
            and member.endswith("_Demand_total_2035_National Trends.csv")
        )
        if total_demand:
            return True, 1
        btm_addition = (
            include_demand_btm
            and member.startswith(DEMAND_BTM_2035_PREFIX)
            and member.endswith("_2035_National Trends.csv")
        )
        return btm_addition, 2
    if name == "pecdres":
        if (
            member.startswith("PECD - RES/Hydro Constraints_250716/")
            and member.endswith("_2035.csv")
        ):
            return True, 2
        if (
            member.startswith("PECD - RES/Hydro Inflows_250704/")
            and member.endswith("_2035.csv")
        ):
            return True, 3
        if (
            member.startswith("PECD - RES/Capacity Factors_250716/")
            and member.endswith("_2035.csv")
        ):
            return True, 4
    return False, 99


def load_archive_tasks(
    metadata: ArchiveMetadata,
    expected_count: int,
    block_size: int,
    include_demand_btm: bool = False,
) -> list[MemberTask]:
    remote = fsspec.open(metadata.url, mode="rb", block_size=block_size).open()
    try:
        with zipfile.ZipFile(remote) as archive:
            tasks: list[MemberTask] = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                selected, priority = select_member(
                    metadata.name,
                    info.filename,
                    include_demand_btm=include_demand_btm,
                )
                if not selected:
                    continue
                if info.flag_bits & 0x1:
                    raise ValueError(f"encrypted member is unsupported: {info.filename}")
                tasks.append(
                    MemberTask(
                        archive=metadata,
                        member_path=info.filename,
                        header_offset=info.header_offset,
                        crc32=info.CRC,
                        compress_type=info.compress_type,
                        compressed_size=info.compress_size,
                        uncompressed_size=info.file_size,
                        flag_bits=info.flag_bits,
                        priority=priority,
                    )
                )
    finally:
        remote.close()
    if len(tasks) != expected_count:
        raise ValueError(
            f"{metadata.name}: selected {len(tasks)} members, expected {expected_count}"
        )
    return tasks


def request_range(
    url: str, start: int, end: int, total_size: int, timeout: float, retries: int
) -> bytes:
    error: Exception | None = None
    expected_length = end - start + 1
    for attempt in range(retries):
        try:
            response = session().get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(30, timeout),
            )
            if response.status_code != 206:
                raise RuntimeError(
                    f"range request returned HTTP {response.status_code}, not 206"
                )
            content_range = response.headers.get("Content-Range", "")
            expected_range = f"bytes {start}-{end}/{total_size}"
            if content_range != expected_range:
                raise RuntimeError(
                    f"unexpected Content-Range {content_range!r}; "
                    f"expected {expected_range!r}"
                )
            if len(response.content) != expected_length:
                raise RuntimeError(
                    f"range length {len(response.content)} != {expected_length}"
                )
            return response.content
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"range request failed after {retries} attempts: {start}-{end}: {error}"
    ) from error


def decompress_member(task: MemberTask, blob: bytes) -> tuple[bytes, bytes]:
    if len(blob) < LOCAL_FILE_HEADER.size:
        raise ValueError("range is shorter than a local ZIP header")
    (
        signature,
        _version,
        flag_bits,
        compress_type,
        _mtime,
        _mdate,
        _local_crc,
        _local_compressed_size,
        _local_uncompressed_size,
        name_length,
        extra_length,
    ) = LOCAL_FILE_HEADER.unpack_from(blob, 0)
    if signature != LOCAL_FILE_SIGNATURE:
        raise ValueError(f"invalid local ZIP signature: 0x{signature:08x}")
    if flag_bits != task.flag_bits:
        raise ValueError(f"flag-bits mismatch for {task.member_path}")
    if compress_type != task.compress_type:
        raise ValueError(f"compression-method mismatch for {task.member_path}")
    data_start = LOCAL_FILE_HEADER.size + name_length + extra_length
    data_end = data_start + task.compressed_size
    if data_end > len(blob):
        raise ValueError(
            f"range does not contain complete compressed payload for {task.member_path}"
        )
    compressed = blob[data_start:data_end]
    if task.compress_type == zipfile.ZIP_STORED:
        raw = compressed
    elif task.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(
            f"unsupported compression type {task.compress_type}: {task.member_path}"
        )
    return compressed, raw


def local_data_start(task: MemberTask, blob: bytes) -> int:
    if len(blob) < LOCAL_FILE_HEADER.size:
        raise ValueError("range is shorter than a local ZIP header")
    (
        signature,
        _version,
        flag_bits,
        compress_type,
        _mtime,
        _mdate,
        _local_crc,
        _local_compressed_size,
        _local_uncompressed_size,
        name_length,
        extra_length,
    ) = LOCAL_FILE_HEADER.unpack_from(blob, 0)
    if signature != LOCAL_FILE_SIGNATURE:
        raise ValueError(f"invalid local ZIP signature: 0x{signature:08x}")
    if flag_bits != task.flag_bits or compress_type != task.compress_type:
        raise ValueError(f"local-header metadata mismatch for {task.member_path}")
    return LOCAL_FILE_HEADER.size + name_length + extra_length


def verify_raw(task: MemberTask, raw: bytes) -> None:
    if len(raw) != task.uncompressed_size:
        raise ValueError(
            f"uncompressed size {len(raw)} != {task.uncompressed_size}: "
            f"{task.member_path}"
        )
    crc = binascii.crc32(raw) & 0xFFFFFFFF
    if crc != task.crc32:
        raise ValueError(
            f"CRC32 {crc:08x} != {task.crc32:08x}: {task.member_path}"
        )


def valid_existing(task: MemberTask, output_path: Path) -> tuple[bool, str]:
    if not output_path.is_file() or output_path.stat().st_size != task.uncompressed_size:
        return False, ""
    sha = hashlib.sha256()
    crc = 0
    with output_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
            crc = binascii.crc32(chunk, crc)
    return (crc & 0xFFFFFFFF) == task.crc32, sha.hexdigest()


def record_for_task(
    task: MemberTask,
    output_path: Path,
    compressed_sha256: str,
    sha256: str,
    source_transport: str,
    status: str,
    error: str = "",
) -> dict[str, object]:
    try:
        relative_output = output_path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        relative_output = output_path.as_posix()
    return {
        "archive_name": task.archive.filename,
        "archive_url": task.archive.url,
        "archive_size": task.archive.size,
        "archive_etag": task.archive.etag,
        "archive_last_modified": task.archive.last_modified,
        "member_path": task.member_path,
        "output_path": relative_output,
        "member_crc32": f"{task.crc32:08x}",
        "compress_type": task.compress_type,
        "compressed_size": task.compressed_size,
        "uncompressed_size": task.uncompressed_size,
        "compressed_sha256": compressed_sha256,
        "sha256": sha256,
        "source_transport": source_transport,
        "status": status,
        "downloaded_at_utc": utc_now(),
        "error": error,
    }


def extract_one(
    task: MemberTask,
    output_root: Path,
    timeout: float,
    retries: int,
    prior: dict[tuple[str, str], dict[str, object]],
    range_caches: list[RangeCache | FileRangeCache],
) -> dict[str, object]:
    member = safe_member_path(task.member_path)
    output_path = output_root / task.archive.filename.removesuffix(".zip") / Path(
        *member.parts
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ok, existing_sha = valid_existing(task, output_path)
    prior_record = prior.get((task.archive.filename, task.member_path), {})
    if existing_ok and prior_record.get("compressed_sha256"):
        return record_for_task(
            task,
            output_path,
            str(prior_record["compressed_sha256"]),
            existing_sha,
            str(prior_record.get("source_transport", "existing_file")),
            "verified_existing",
        )

    # Four KiB of header slack covers the local filename and extra fields.
    range_end = (
        task.header_offset
        + LOCAL_FILE_HEADER.size
        + 4096
        + task.compressed_size
        + 32
        - 1
    )
    range_end = min(range_end, task.archive.size - 1)
    blob: bytes | None = None
    source_transport = ""
    header_end = min(task.header_offset + 4095, task.archive.size - 1)
    for cache in range_caches:
        header = cache.read(task.header_offset, header_end)
        if header is None:
            continue
        data_start = local_data_start(task, header)
        exact_end = task.header_offset + data_start + task.compressed_size - 1
        candidate = cache.read(task.header_offset, exact_end)
        if candidate is not None:
            blob = candidate
            source_transport = cache.label
            break
    if blob is None:
        blob = request_range(
            task.archive.url,
            task.header_offset,
            range_end,
            task.archive.size,
            timeout,
            retries,
        )
        source_transport = "remote_http_range"
    compressed, raw = decompress_member(task, blob)
    verify_raw(task, raw)
    compressed_sha = hashlib.sha256(compressed).hexdigest()
    raw_sha = hashlib.sha256(raw).hexdigest()

    temporary = output_path.with_name(output_path.name + ".part")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return record_for_task(
        task,
        output_path,
        compressed_sha,
        raw_sha,
        source_transport,
        "downloaded_verified",
    )


def load_prior(jsonl_path: Path) -> dict[tuple[str, str], dict[str, object]]:
    latest: dict[tuple[str, str], dict[str, object]] = {}
    if not jsonl_path.exists():
        return latest
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = (str(record["archive_name"]), str(record["member_path"]))
                latest[key] = record
            except (json.JSONDecodeError, KeyError):
                continue
    return latest


def write_csv(
    csv_path: Path, records: Iterable[dict[str, object]]
) -> None:
    temporary = csv_path.with_name(csv_path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            sorted(
                records,
                key=lambda row: (
                    str(row["archive_name"]),
                    str(row["member_path"]),
                ),
            )
        )
    os.replace(temporary, csv_path)


def discover_range_caches(
    root: Path, archive: ArchiveMetadata
) -> list[RangeCache | FileRangeCache]:
    caches: list[RangeCache | FileRangeCache] = []
    for directory in sorted(root.glob(".*.zip.ranges")):
        metadata_path = directory / "ranges.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                int(metadata.get("size", -1)) == archive.size
                and normalized_object_path(str(metadata.get("url", "")))
                == normalized_object_path(archive.url)
            ):
                caches.append(RangeCache(directory, archive))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    stem = archive.filename.removesuffix(".zip")
    for path in (root / archive.filename, root / f"{stem}_edge.zip"):
        caches.append(FileRangeCache(path, archive))
    return caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "dataset/external_hourly/"
            "europe_mid_duration_valley_upgrade_v0_1/raw/"
            "eraa_2025/selective_2035"
        ),
    )
    parser.add_argument(
        "--range-cache-root",
        type=Path,
        default=Path(
            "dataset/external_hourly/"
            "europe_mid_duration_valley_upgrade_v0_1/raw/eraa_2025"
        ),
        help="reuse completed byte ranges from concurrent full-archive downloads",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "dataset/external_hourly/"
            "europe_mid_duration_valley_upgrade_v0_1/artifacts/research/"
            "ERAA_2025_SELECTIVE_2035_MEMBER_MANIFEST.csv"
        ),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=2 * 1024 * 1024)
    parser.add_argument(
        "--archives",
        nargs="+",
        choices=("weather", "demand", "pecdres"),
        default=("weather", "demand", "pecdres"),
        help="archive groups to process (default: all three)",
    )
    parser.add_argument(
        "--include-demand-btm",
        action="store_true",
        help=(
            "also select the eight 2035 behind-the-meter PV/battery additions "
            "for AT00, CH00, DE00 and HU00"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.manifest.with_suffix(".jsonl")
    prior = load_prior(jsonl_path)

    tasks: list[MemberTask] = []
    caches_by_archive: dict[str, list[RangeCache | FileRangeCache]] = {}
    for name in args.archives:
        spec = ARCHIVES[name]
        metadata = head_archive(name, spec, args.timeout, args.retries)
        expected_count = int(spec["expected_count"])
        if name == "demand" and args.include_demand_btm:
            expected_count += DEMAND_BTM_2035_COUNT
        selected = load_archive_tasks(
            metadata,
            expected_count,
            args.block_size,
            include_demand_btm=args.include_demand_btm,
        )
        print(
            f"{metadata.filename}: selected {len(selected)} members; "
            f"ETag={metadata.etag}; size={metadata.size}",
            flush=True,
        )
        tasks.extend(selected)
        caches = discover_range_caches(args.range_cache_root, metadata)
        caches_by_archive[metadata.filename] = caches
        print(
            "  local range sources: "
            + (", ".join(cache.label for cache in caches) or "none"),
            flush=True,
        )
    tasks.sort(key=lambda task: (task.priority, task.member_path))
    print(
        f"Starting {len(tasks)} selective extractions with {args.workers} workers",
        flush=True,
    )

    records = dict(prior)
    successes = 0
    failures = 0
    started = time.monotonic()
    last_report = started
    with jsonl_path.open("a", encoding="utf-8") as checkpoint:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(
                    extract_one,
                    task,
                    args.output_root,
                    args.timeout,
                    args.retries,
                    prior,
                    caches_by_archive[task.archive.filename],
                ): task
                for task in tasks
            }
            for completed, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    record = future.result()
                    successes += 1
                except Exception as exc:  # noqa: BLE001
                    output_path = (
                        args.output_root
                        / task.archive.filename.removesuffix(".zip")
                        / Path(*safe_member_path(task.member_path).parts)
                    )
                    record = record_for_task(
                        task,
                        output_path,
                        "",
                        "",
                        "",
                        "failed",
                        f"{type(exc).__name__}: {exc}",
                    )
                    failures += 1
                records[(task.archive.filename, task.member_path)] = record
                checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                checkpoint.flush()
                now = time.monotonic()
                if completed % 10 == 0 or now - last_report >= 30 or completed == len(tasks):
                    elapsed = max(now - started, 1e-6)
                    print(
                        f"progress {completed}/{len(tasks)}; "
                        f"verified={successes}; failed={failures}; "
                        f"rate={completed / elapsed:.2f} members/s",
                        flush=True,
                    )
                    last_report = now

    selected_records = [
        records[(task.archive.filename, task.member_path)] for task in tasks
    ]
    write_csv(args.manifest, selected_records)
    if failures:
        print(
            f"Completed with {failures} failures; rerun to resume. "
            f"Manifest: {args.manifest}",
            flush=True,
        )
        return 2
    print(
        f"All {len(tasks)} members verified. Manifest: {args.manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
