"""
M4 Fig 5 province-level season-aware MC CSR-selected sheets without LFP 4h.

This script reuses the MC-q75 province plotting implementation but points it to
the dual-sided seasonal Monte Carlo freqlen nodes. The plotted curves remain
average unit-cost ratios, while each province/technology/dispatch-window cell
selects its champion duration by CSR instead of OSR.

Output:
  output/m4_economics/nature_figure/freqlen_per_province_csr_seasonal_mc_no_lfp4h/
"""

from pathlib import Path

import m4_visualize_freqlen_mc_q75_per_province_osr_no_lfp4h as plotter


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"

plotter.CHAMPION_METRIC = "CSR"
plotter.OUT_DIR = (
    M4_DIR / "nature_figure" / "freqlen_per_province_csr_seasonal_mc_no_lfp4h"
)
plotter.SOURCE_DIR = plotter.OUT_DIR / "source_data"

plotter.SCAN_ROWS = [
    ("F", "seasmc_freq4_len5", 0, "F events=4/yr"),
    ("F", "seasmc_freq8_len5", 1, "F events=8/yr"),
    ("F", "seasmc_freq12_len5", 2, "F events=12/yr"),
    ("F", "seasmc_freq16_len5", 3, "F events=16/yr"),
    ("G", "seasmc_freq8_len5", 4, "G length=5d"),
    ("G", "seasmc_freq8_len7", 5, "G length=7d"),
    ("G", "seasmc_freq8_len10", 6, "G length=10d"),
    ("G", "seasmc_freq8_len14", 7, "G length=14d"),
]

plotter.FIG_TITLE_PREFIX = (
    "Fig 5 province-level seasonal-MC CSR-selected cost curves"
)
plotter.FIG_FOOTER = (
    "Rows 1-4: season-aware MC frequency scan; rows 5-8: run-length scan. "
    "Demand side = 1 real year + 4 seasonal Markov MC years; charge side = "
    "1 real year + 4 seasonal block-bootstrap MC years. LFP 4h is excluded; "
    "each province-tech-dw cell uses its own CSR-selected duration. Curves "
    "show average unit cost / gas cost; red dashed line = gas parity."
)
plotter.FIG_FILE_PREFIX = "fig5_freqlen_seasonal_mc_csr_no_lfp4h"
plotter.CHAMPION_FILENAME = (
    "fig5_freqlen_seasonal_mc_province_csr_no_lfp4h_champions.csv"
)


if __name__ == "__main__":
    plotter.main()
