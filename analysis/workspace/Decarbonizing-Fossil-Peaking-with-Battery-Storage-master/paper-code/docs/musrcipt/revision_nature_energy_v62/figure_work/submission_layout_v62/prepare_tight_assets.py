#!/usr/bin/env python3
"""Create non-destructive, tightly bounded figure assets for v62 typesetting."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "tight_assets"
PADDING_PT = 2.8
RENDER_DPI = 300
INK_THRESHOLD = 8


PDF_ASSETS = {
    "main_figure2.pdf": ROOT
    / "figure_work/figure2_relayout_v62/outputs/fig_v62_preview_figure2_world_callout_relayout.pdf",
    "main_figure3.pdf": ROOT
    / "figure_work/figure3_relayout_v62/outputs/fig_v62_preview_figure3_recovery_relayout.pdf",
    "main_figure4.pdf": ROOT
    / "figure_work/figure4_relayout_v62/outputs/fig_v62_preview_figure4_recharge_relayout.pdf",
    "main_figure5.pdf": ROOT
    / "figure_work/main_figures/fig_v62_figure5_duration_parity_atlas.pdf",
    "main_figure6.pdf": ROOT
    / "figure_work/main_figures/fig_v62_figure6_temporal_parity.pdf",
    "supp_figure1.pdf": ROOT
    / "figure_work/supplementary_figures_v62/figures/fig_v62_supplementary_figure1_discount_sensitivity.pdf",
    "supp_figure2.pdf": ROOT
    / "figure_work/supplementary_figures_v62/figures/fig_v62_supplementary_figure2_duration_boundaries.pdf",
    "supp_figure3.pdf": ROOT
    / "figure_work/supplementary_figures_v62/figures/fig_v62_supplementary_figure3_recharge_factorial.pdf",
    "supp_figure4.pdf": ROOT
    / "statistical_inference_v62/figures/fig_v62_supplementary_figure4_statistical_inference.pdf",
    "supp_figure5.pdf": ROOT
    / "capacity_expansion_validation/figures/fig_v62_supplementary_figure5_integrated_capacity_expansion.pdf",
    "supp_figure6.pdf": ROOT
    / "capacity_expansion_validation/figures/fig_v62_supplementary_figure6_planning_resolution.pdf",
    "extended_data_figure1.pdf": ROOT
    / "figure_work/extended_data_figures/ed_fig_v57_demand_model_validation.pdf",
    "extended_data_figure2.pdf": ROOT
    / "figure_work/extended_data_figures/ed_fig_v57_pecd_inputs_operating_years.pdf",
    "extended_data_figure3.pdf": ROOT
    / "figure_work/extended_data/fig_v54_ed1_cost_paths_53_eur.pdf",
    "extended_data_figure4.pdf": ROOT
    / "figure_work/extended_data/fig_v54_ed2_event_definition_sensitivity.pdf",
    "extended_data_figure5.pdf": ROOT
    / "figure_work/extended_data/fig_v54_ed3_numerical_qa.pdf",
    "extended_data_figure6.pdf": ROOT
    / "figure_work/extended_data/fig_v54_ed4_full_technology_duration_response.pdf",
    "extended_data_figure7.pdf": ROOT
    / "figure_work/extended_data_figure7_v62/outputs/ed_fig_v62_three_regime_cross_duration_heterogeneity.pdf",
}

PNG_ASSETS = {
    "main_figure1.png": ROOT
    / "figure_work/main_figures/fig_v62_figure1_sequencing_parity_framework_regimes.png"
}


def ink_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    rgba = image.convert("RGBA")
    white = Image.new("RGB", rgba.size, "white")
    white.paste(rgba, mask=rgba.getchannel("A"))
    difference = ImageChops.difference(white, Image.new("RGB", rgba.size, "white"))
    mask = difference.convert("L").point(
        lambda value: 255 if value > INK_THRESHOLD else 0
    )
    bbox = mask.getbbox()
    if bbox is None:
        raise RuntimeError("No non-white figure content detected")
    return bbox


def render_pdf(source: Path, destination_prefix: Path) -> Path:
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            "1",
            "-singlefile",
            "-png",
            "-r",
            str(RENDER_DPI),
            str(source),
            str(destination_prefix),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return destination_prefix.with_suffix(".png")


def crop_pdf(source: Path, destination: Path) -> dict[str, float | str]:
    reader = PdfReader(str(source))
    if len(reader.pages) != 1:
        raise ValueError(f"Expected a one-page figure PDF: {source}")

    page = reader.pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)

    with tempfile.TemporaryDirectory() as tmp:
        preview = render_pdf(source, Path(tmp) / "preview")
        with Image.open(preview) as image:
            x0, y0, x1, y1 = ink_bbox(image)
            width_px = image.width
            height_px = image.height

    left_pt = max(0.0, x0 / width_px * width_pt - PADDING_PT)
    right_pt = min(width_pt, x1 / width_px * width_pt + PADDING_PT)
    bottom_pt = max(0.0, (height_px - y1) / height_px * height_pt - PADDING_PT)
    top_pt = min(height_pt, (height_px - y0) / height_px * height_pt + PADDING_PT)
    box = RectangleObject((left_pt, bottom_pt, right_pt, top_pt))
    page.mediabox = box
    page.cropbox = box
    page.trimbox = box
    page.bleedbox = box

    writer = PdfWriter()
    writer.add_page(page)
    if reader.metadata:
        writer.add_metadata(
            {key: value for key, value in reader.metadata.items() if value is not None}
        )
    with destination.open("wb") as handle:
        writer.write(handle)

    cropped_width_pt = right_pt - left_pt
    cropped_height_pt = top_pt - bottom_pt
    return {
        "source": str(source.relative_to(ROOT)),
        "output": str(destination.relative_to(ROOT)),
        "source_width_pt": round(width_pt, 3),
        "cropped_width_pt": round(cropped_width_pt, 3),
        "source_height_pt": round(height_pt, 3),
        "cropped_height_pt": round(cropped_height_pt, 3),
        "horizontal_content_gain_percent": round(
            (width_pt / cropped_width_pt - 1.0) * 100.0, 2
        ),
    }


def crop_png(source: Path, destination: Path) -> dict[str, float | str]:
    with Image.open(source) as image:
        x0, _, x1, _ = ink_bbox(image)
        padding_px = max(8, round(PADDING_PT / 72.0 * 600.0))
        left = max(0, x0 - padding_px)
        right = min(image.width, x1 + padding_px)
        cropped = image.crop((left, 0, right, image.height))
        cropped.save(destination, dpi=(600, 600), optimize=True)
        source_width = image.width
        cropped_width = cropped.width

    return {
        "source": str(source.relative_to(ROOT)),
        "output": str(destination.relative_to(ROOT)),
        "source_width_px": source_width,
        "cropped_width_px": cropped_width,
        "horizontal_content_gain_percent": round(
            (source_width / cropped_width - 1.0) * 100.0, 2
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, float | str]] = []

    for output_name, source in PDF_ASSETS.items():
        if not source.exists():
            raise FileNotFoundError(source)
        report.append(crop_pdf(source, OUTPUT / output_name))

    for output_name, source in PNG_ASSETS.items():
        if not source.exists():
            raise FileNotFoundError(source)
        report.append(crop_png(source, OUTPUT / output_name))

    report_path = Path(__file__).resolve().parent / "tight_asset_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(report)} tight assets in {OUTPUT}")
    print(f"QA report: {report_path}")


if __name__ == "__main__":
    main()
