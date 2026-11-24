"""Render the v86 main figures 1-5 and Table 1 with all nature-figure gates.

  python run_all.py            all figures and Table 1
  python run_all.py 2 4        selected figures
Outputs: figures/output (PDF, SVG, PNG 300 dpi, TIFF 600 dpi, table1.*), figures/qa (alignment, collision and glyph
audits), source_data (per-panel CSV). Reads only delivered result packages; no optimisation is solved.
"""
import sys
import importlib

import data as D

if __name__ == '__main__':
    D.check()
    sel = sys.argv[1:] or ['1', '2', '3', '4', '5', 'table1', 'appendix_europe_us', 'appendix_provincial_allocation', 'appendix_event_extension', 'appendix_factorial', 'appendix_capacity', 'appendix_cost_paths']
    for k in sel:
        importlib.import_module(k if not k.isdigit() else f'fig{k}').main()
