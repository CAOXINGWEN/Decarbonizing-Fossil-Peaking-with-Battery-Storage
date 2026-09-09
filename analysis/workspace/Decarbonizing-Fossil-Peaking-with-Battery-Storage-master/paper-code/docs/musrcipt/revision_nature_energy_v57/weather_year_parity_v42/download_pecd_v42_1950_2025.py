#!/usr/bin/env python3
"""Download the PECD v4.2 inputs used by the v56 weather-year experiment.

The experiment uses a single PECD generation lineage for every calendar year:
ERA5 historical PECD v4.2 capacity factors for representative 2030 wind and
solar technologies. Population-weighted temperature is retrieved from the same
PECD release for the hourly demand reconstruction. The CDS catalogue exposes
corrected fv2 temperature files through 2021 and fv1 annual files thereafter;
these are the current catalogue revisions of one PECD v4.2 product, not a splice
between PECD generations.

CDS access requires a personal access token in ~/.cdsapirc. The script is
restartable: verified archives and extracted files are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from zipfile import BadZipFile, ZipFile

import requests


HERE = Path(__file__).resolve().parent
ARCHIVES = HERE / "raw_archives"
EXTRACTED = HERE / "extracted"
MANIFEST = HERE / "pecd_v42_download_manifest.json"

DATASET = "sis-energy-pecd"
CDS_API = "https://cds.climate.copernicus.eu/api"
YEARS = tuple(range(1950, 2026))


@dataclass(frozen=True)
class Product:
    product_id: str
    variable: str
    spatial_resolution: str
    years: tuple[int, ...]
    technology: str | None = None
    energy_scenario: str | None = None
    file_version: str = "fv1"


PRODUCTS = (
    Product(
        product_id="wind_onshore_tech35_regrb_peon_fv1",
        variable="wind_power_onshore_capacity_factor",
        spatial_resolution="peon",
        technology="35",
        energy_scenario="resource_grade_b",
        years=YEARS,
    ),
    Product(
        product_id="wind_offshore_tech21_regrb_peof_fv1",
        variable="wind_power_offshore_capacity_factor",
        spatial_resolution="peof",
        technology="21",
        energy_scenario="resource_grade_b",
        years=YEARS,
    ),
    Product(
        product_id="solar_utility_fixed_tech62_peon_fv1",
        variable="solar_photovoltaic_generation_capacity_factor",
        spatial_resolution="peon",
        technology="62",
        years=YEARS,
    ),
    Product(
        product_id="population_weighted_temperature_szon_fv2",
        variable="population_weighted_temperature",
        spatial_resolution="szon",
        file_version="fv2",
        years=tuple(range(1950, 2022)),
    ),
    Product(
        product_id="population_weighted_temperature_szon_fv1",
        variable="population_weighted_temperature",
        spatial_resolution="szon",
        file_version="fv1",
        years=tuple(range(2022, 2026)),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunks(values: tuple[int, ...], size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def request_for(product: Product, years: tuple[int, ...]) -> dict[str, object]:
    request: dict[str, object] = {
        "pecd_version": "pecd4_2",
        "temporal_period": ["historical"],
        "origin": ["era5_reanalysis"],
        "variable": [product.variable],
        "spatial_resolution": [product.spatial_resolution],
        "year": [str(year) for year in years],
        "file_version": [product.file_version],
    }
    if product.technology is not None:
        request["technology"] = [product.technology]
    if product.energy_scenario is not None:
        request["energy_scenario"] = [product.energy_scenario]
    return request


def archive_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None and bool(archive.namelist())
    except BadZipFile:
        return False


def safe_extract(archive_path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    extracted: list[Path] = []
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise RuntimeError(f"Unsafe path in {archive_path.name}: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not target.exists() or target.stat().st_size != member.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as sink:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        sink.write(block)
            extracted.append(target)
    return extracted


def load_existing_manifest() -> list[dict[str, object]]:
    if not MANIFEST.exists():
        return []
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return list(payload.get("archives", []))


def write_manifest(rows: list[dict[str, object]]) -> None:
    payload = {
        "dataset": DATASET,
        "pecd_version": "PECD v4.2",
        "origin": "ERA5 reanalysis",
        "calendar_years": [min(YEARS), max(YEARS)],
        "complete_july_june_operating_years": 75,
        "technology_contract": {
            "onshore": "technology 35 (SP277 W m-2, 150 m hub), Resource Grade B",
            "offshore": "technology 21 (SP316 W m-2, 155 m hub), Resource Grade B",
            "solar": "technology 62 (utility-scale fixed)",
        },
        "archives": sorted(rows, key=lambda row: str(row["archive"])),
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_cds_token() -> str:
    for name in ("ECMWF_DATASTORES_KEY", "CDSAPI_KEY"):
        if value := os.environ.get(name):
            return value.strip()
    candidates = [
        Path(os.environ.get("ECMWF_DATASTORES_RC_FILE", Path.home() / ".ecmwfdatastoresrc")),
        Path(os.environ.get("CDSAPI_RC", Path.home() / ".cdsapirc")),
    ]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("key:"):
                return line.split(":", 1)[1].strip()
    raise RuntimeError(
        "CDS credentials are missing. Create ~/.cdsapirc with the personal access token "
        "from https://cds.climate.copernicus.eu/profile and accept the PECD licence."
    )


def api_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    json_body: dict[str, object] | None = None,
    timeout: tuple[int, int] = (30, 180),
) -> requests.Response:
    delay = 5.0
    for attempt in range(8):
        response = session.request(method, url, json=json_body, timeout=timeout)
        if response.status_code not in {408, 429, 500, 502, 503, 504}:
            if response.status_code >= 400:
                try:
                    detail = json.dumps(response.json(), ensure_ascii=False)
                except ValueError:
                    detail = response.text[:2000]
                raise RuntimeError(f"CDS API {response.status_code} for {url}: {detail}")
            return response
        if attempt == 7:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        time.sleep(float(retry_after) if retry_after else delay)
        delay = min(delay * 1.7, 120.0)
    raise AssertionError("unreachable")


def link_href(payload: dict[str, object], relation: str) -> str | None:
    for link in payload.get("links", []):
        if isinstance(link, dict) and link.get("rel") == relation:
            return str(link["href"])
    return None


def retrieve(
    session: requests.Session,
    request: dict[str, object],
    target: Path,
) -> None:
    monitor = submit_job(session, request)

    sleep = 2.0
    while True:
        status_payload = api_request(session, "GET", monitor).json()
        status = status_payload.get("status")
        if status == "successful":
            break
        if status in {"failed", "rejected", "dismissed", "deleted"}:
            results_url = link_href(status_payload, "results") or f"{monitor}/results"
            failed = api_request(session, "GET", results_url).json()
            raise RuntimeError(f"CDS job {status}: {json.dumps(failed, ensure_ascii=False)}")
        if status not in {"accepted", "running"}:
            raise RuntimeError(f"Unknown CDS job state: {status_payload}")
        time.sleep(sleep)
        sleep = min(sleep * 1.5, 90.0)

    download_successful_job(session, monitor, status_payload, target)


def submit_job(session: requests.Session, request: dict[str, object]) -> str:
    process_url = f"{CDS_API}/retrieve/v1/processes/{DATASET}"
    submit = api_request(
        session,
        "POST",
        f"{process_url}/execution",
        json_body={"inputs": request},
    ).json()
    monitor = link_href(submit, "monitor")
    if monitor is None:
        raise RuntimeError(f"CDS response has no monitor link: {submit}")
    print(f"submitted CDS job {monitor.rpartition('/')[2]}", flush=True)
    return monitor


def fail_if_terminal(status: str | None, session: requests.Session, monitor: str) -> None:
    if status in {"failed", "rejected", "dismissed", "deleted"}:
        results_url = f"{monitor}/results"
        failed = api_request(session, "GET", results_url).json()
        raise RuntimeError(f"CDS job {status}: {json.dumps(failed, ensure_ascii=False)}")
    if status not in {"accepted", "running", "successful"}:
        raise RuntimeError(f"Unknown CDS job state for {monitor}: {status}")


def download_successful_job(
    session: requests.Session,
    monitor: str,
    status_payload: dict[str, object],
    target: Path,
) -> None:
    results_url = link_href(status_payload, "results") or f"{monitor}/results"
    results = api_request(session, "GET", results_url).json()
    asset = results.get("asset", {}).get("value", {})
    href = asset.get("href")
    expected_size = int(asset.get("file:size", 0))
    if not href or expected_size <= 0:
        raise RuntimeError(f"Invalid CDS result asset: {results}")
    download_url = urljoin(results_url, str(href))
    with session.get(download_url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)
    if target.stat().st_size != expected_size:
        raise RuntimeError(
            f"Incomplete download for {target.name}: {target.stat().st_size} of {expected_size} bytes"
        )


def finalize_archive(
    archive: Path,
    product: Product,
    years: tuple[int, ...],
    request: dict[str, object],
    rows_by_archive: dict[str, dict[str, object]],
) -> None:
    if not archive_is_valid(archive):
        raise RuntimeError(f"Invalid CDS archive: {archive}")
    files = safe_extract(archive, EXTRACTED / product.product_id)
    row = {
        "archive": archive.relative_to(HERE).as_posix(),
        "product_id": product.product_id,
        "years": list(years),
        "request": request,
        "size_bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "extracted_files": [path.relative_to(HERE).as_posix() for path in files],
    }
    rows_by_archive[str(row["archive"])] = row
    write_manifest(list(rows_by_archive.values()))
    print(f"verified {archive.name}: {len(files)} files", flush=True)


def pending_tasks(chunk_size: int, selected: set[str] | None) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for product in PRODUCTS:
        if selected and product.product_id not in selected:
            continue
        for years in chunks(product.years, chunk_size):
            year_tag = f"{years[0]}_{years[-1]}"
            archive = ARCHIVES / f"{product.product_id}_{year_tag}.zip"
            request = request_for(product, years)
            tasks.append({
                "product": product,
                "years": years,
                "year_tag": year_tag,
                "archive": archive,
                "request": request,
            })
    return tasks


def run(chunk_size: int, selected: set[str] | None, max_active: int) -> None:
    token = read_cds_token()
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    rows_by_archive = {str(row["archive"]): row for row in load_existing_manifest()}
    session = requests.Session()
    session.headers.update({
        "PRIVATE-TOKEN": token,
        "User-Agent": "pecd-v42-v56-reproducible-downloader/1.0",
    })

    queue = pending_tasks(chunk_size, selected)
    active: dict[str, dict[str, object]] = {}
    while queue or active:
        while queue and len(active) < max_active:
            task = queue.pop(0)
            product = task["product"]
            archive = task["archive"]
            if not isinstance(product, Product) or not isinstance(archive, Path):
                raise TypeError("Internal task construction failed")
            if archive_is_valid(archive):
                finalize_archive(
                    archive,
                    product,
                    task["years"],  # type: ignore[arg-type]
                    task["request"],  # type: ignore[arg-type]
                    rows_by_archive,
                )
                continue
            print(f"requesting {product.product_id} {task['year_tag']}", flush=True)
            monitor = submit_job(session, task["request"])  # type: ignore[arg-type]
            task["monitor"] = monitor
            active[monitor] = task

        completed: list[str] = []
        for monitor, task in list(active.items()):
            status_payload = api_request(session, "GET", monitor).json()
            status = status_payload.get("status")
            fail_if_terminal(status, session, monitor)
            if status != "successful":
                continue
            archive = task["archive"]
            product = task["product"]
            if not isinstance(archive, Path) or not isinstance(product, Product):
                raise TypeError("Internal active task construction failed")
            download_successful_job(session, monitor, status_payload, archive)
            finalize_archive(
                archive,
                product,
                task["years"],  # type: ignore[arg-type]
                task["request"],  # type: ignore[arg-type]
                rows_by_archive,
            )
            completed.append(monitor)
        for monitor in completed:
            active.pop(monitor, None)
        if active:
            time.sleep(20.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--max-active", type=int, default=1)
    parser.add_argument(
        "--product",
        action="append",
        choices=[product.product_id for product in PRODUCTS],
        help="Download only a selected product; may be supplied more than once.",
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if args.max_active < 1:
        raise ValueError("--max-active must be positive")
    run(args.chunk_size, set(args.product) if args.product else None, args.max_active)


if __name__ == "__main__":
    main()
