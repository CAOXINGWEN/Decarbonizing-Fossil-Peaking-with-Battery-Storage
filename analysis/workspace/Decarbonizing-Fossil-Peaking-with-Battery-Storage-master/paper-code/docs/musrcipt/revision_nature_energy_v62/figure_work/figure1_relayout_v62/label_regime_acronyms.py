#!/usr/bin/env python3
"""Replace geographic display labels in Figure 1 with balancing-regime acronyms.

The operation is deliberately limited to the three label boxes. All illustrated
evidence, geometry and numerical sample sizes remain unchanged.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FIGURE_WORK = Path(__file__).resolve().parents[1]
SOURCE = FIGURE_WORK / "main_figures" / "fig_v62_figure1_sequencing_parity_framework.png"
OUTPUT = FIGURE_WORK / "main_figures" / "fig_v62_figure1_sequencing_parity_framework_regimes.png"

LABELS = (
    (68, "NVSB", 132, (5, 582, 132, 657)),
    (211, "PLRP", 26, (145, 582, 278, 657)),
    (349, "CLDB", 27, (286, 582, 415, 657)),
)


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
    count_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 19)

    for centre_x, acronym, sample_size, box in LABELS:
        draw.rectangle(box, fill="white")
        draw.text((centre_x, 591), acronym, font=label_font, fill="black", anchor="ma")
        draw.text((centre_x, 620), f"n={sample_size}", font=count_font, fill="black", anchor="ma")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, dpi=(300, 300), optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
