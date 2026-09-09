"""Render the two updated procurement figures using frozen plotting definitions."""
from pathlib import Path
import argparse
import difflib
import json
import shutil
import sys

from run_capacity_v68 import WORK,CAP,module,sha,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--only',choices=['5','6'])
    args=parser.parse_args()
    qa=json.loads((WORK/'qa/model_verification.json').read_text())
    assert qa['status']=='pass' and qa['design_count']==540
    helpers=WORK/'scripts/frozen_helpers'
    helpers.mkdir(exist_ok=True)
    sources=['plot_v62_capacity_expansion_validation.py','plot_v62_integrated_capacity_expansion.py',
             'resolution_convergence_v64/plot_v64_resolution_convergence.py']
    for relative in sources:
        shutil.copy2(CAP/relative,helpers/Path(relative).name)
    helper5=helpers/'plot_v62_integrated_capacity_expansion.py'
    original=helper5.read_text()
    old='    axis.set_xlim(-225, 225)'
    new='''    data_extent = float(data["total_power_change_from_observed_pp"].abs().max())
    display_extent = max(225.0, 50.0 * np.ceil((data_extent + 15.0) / 50.0))
    axis.set_xlim(-display_extent, display_extent)'''
    assert original.count(old)==1
    updated=original.replace(old,new)
    helper5.write_text(updated)
    (WORK/'figure_layout_changes.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True),updated.splitlines(keepends=True),
        fromfile='original_supp_figure5_plotter',tofile='v68_supp_figure5_plotter')))
    sys.path.insert(0,str(helpers))
    figures=WORK/'figures';figures.mkdir(exist_ok=True)
    s5=module('v68_supp5',helpers/'plot_v62_integrated_capacity_expansion.py')
    s5.HERE=WORK/'qa';s5.PRIMARY_RESULTS=WORK/'results/pooled_4h'
    s5.CHRONOLOGY_RESULTS=WORK/'results/chronology_4h'
    s5.FIGURES=figures;s5.SOURCE_DATA=WORK/'source_data/supp_figure5'
    s5.STEM='supp_figure5'
    s5.pooled.HERE=WORK/'qa';s5.pooled.RESULTS=s5.PRIMARY_RESULTS
    s5.pooled.SOURCE_DATA=s5.SOURCE_DATA
    reports_path=WORK/'qa/figure_export_reports.json'
    reports=json.loads(reports_path.read_text()) if reports_path.exists() else {}
    if args.only!='6':
        reports['supp_figure5']=s5.plot()
    s6=module('v68_supp6',helpers/'plot_v64_resolution_convergence.py')
    s6.HERE=WORK/'qa';s6.RESULTS=WORK/'results/comparison_2h_4h_6h'
    s6.FIGURES=figures;s6.SOURCE_DATA=WORK/'source_data/supp_figure6'
    s6.STEM='supp_figure6'
    if args.only!='5':
        reports['supp_figure6']=s6.plot()
    reports['source_sha256']={p.name:sha(p) for p in sorted(helpers.glob('plot*.py'))}
    write_json(reports_path,reports)
    print('Rendered updated supplementary figure(s):',args.only or '5 and 6')


if __name__=='__main__':
    main()
