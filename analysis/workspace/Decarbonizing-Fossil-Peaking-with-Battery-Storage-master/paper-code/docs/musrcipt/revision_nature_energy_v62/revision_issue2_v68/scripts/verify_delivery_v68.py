"""Check manuscript consistency, unchanged EFC assets, figures, and PDF layout."""
from pathlib import Path
import hashlib
import json
import re
import sys

import fitz
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
REV=WORK.parent
QA=WORK/'qa'
checks={}


def check(name,value,detail=None):
    checks[name]={'pass':bool(value),'detail':detail}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(text,start):
    assert text[start]=='{'
    depth=1;i=start+1
    while depth:
        depth+=(text[i]=='{')-(text[i]=='}');i+=1
    return text[start+1:i-1],i


def macros(path):
    text=path.read_text();values={}
    for match in re.finditer(r'\\newcommand\{\\([A-Za-z]+)\}',text):
        if text[match.end():match.end()+1]=='{':
            value,_=group(text,match.end());values[match[1]]=value
    return values


def main():
    model=json.loads((QA/'model_verification.json').read_text())
    check('all_540_designs_verified',model['status']=='pass' and model['design_count']==540)
    from run_capacity_v68 import module
    base=module('delivery_input_verification',HERE/'capacity_model_v68.py')
    expected=json.loads((WORK/'scenario_cache/manifest.json').read_text())['provenance']
    input_paths={'model':HERE/'capacity_model_v68.py',
                 'chronology':REV/'capacity_expansion_validation/run_v62_chronology_procurement_counterfactual.py',
                 'panel':base.PANEL_PATH,'costs':base.COST_CATALOGUE_PATH,
                 'values':base.VALUE_SCENARIO_PATH,'registered_runner':base.REGISTERED_RUNNER_PATH}
    check('all_registered_model_and_data_hashes_preserved',
          all(sha(path)==expected[key] for key,path in input_paths.items()))
    previous=json.loads((WORK/'before/manuscript_dependency_hashes.json').read_text())
    changed=[name for name,digest in previous.items() if sha(REV/name)!=digest]
    check('all_v67_manuscript_dependencies_preserved',not changed,changed)
    old_model=REV/'capacity_expansion_validation/run_v62_multiyear_capacity_expansion.py'
    check('frozen_original_model_preserved',sha(old_model)==sha(WORK/'before'/old_model.name))
    new_model=(HERE/'capacity_model_v68.py').read_text()
    check('task_peak_capacity_bound_removed','bounds[int(index)] = (0.0, maximum_task)' not in new_model
          and 'ub=maximum_task' not in new_model and '[(0.0, None)] * variable_count' in new_model)
    check('obsolete_bound_qa_removed','minimum_capacity_upper_bound_slack' not in new_model)
    maintex=(REV/'musript_new_v68.tex').read_text()
    sitex=(REV/'musript_new_v68_supplementary.tex').read_text()
    check('method_matches_nonnegative_capacity_domain',r'P_{i,k}\geq0' in maintex
          and '精确冗余' not in maintex+sitex and '采购功率为非负内生决策变量' in sitex)
    check('independent_design_count_in_both_manuscripts','540' in maintex and '540' in sitex)
    check('efc_main_figure_paths_retained',all(f'revision_issue1_v67/figures/main_figure{i}.pdf' in maintex for i in (3,4,5,6)))
    check('updated_procurement_figure_paths',all(f'revision_issue2_v68/figures/supp_figure{i}.pdf' in sitex for i in (5,6)))
    check('no_duplicate_supplements_8_to_11',not re.search(r'Supplementary Figure (?:8|9|10|11)\b',sitex))
    pattern=re.compile(r'防御|审稿|质疑|为回应|不影响|不改变|仍然成立|更严格|并非|而非|不意味着|不能直接|仅作为|仅用于|保守|精确冗余|in response to the reviewer|does not undermine|remains robust|more stringent|despite this limitation',re.I)
    candidates=[]
    for name,text in [('main',maintex),('supplement',sitex)]:
        for n,line in enumerate(text.splitlines(),1):
            if pattern.search(line):candidates.append({'file':name,'line':n,'text':line})
    check('defensive_expression_screen',not candidates,candidates)
    macro_sets=[
        ('capacity_expansion_validation/chronology_procurement_validation/resolution4h_results/chronology_procurement_macros.tex','results/chronology_4h/chronology_procurement_macros.tex'),
        ('capacity_expansion_validation/resolution4h_vs6h/planning_resolution_macros.tex','results/comparison_4h_6h/planning_resolution_macros.tex'),
        ('capacity_expansion_validation/resolution_convergence_v64/results/resolution_convergence_macros.tex','results/comparison_2h_4h_6h/resolution_convergence_macros.tex')]
    changes=[];new_macros={}
    for old,new in macro_sets:
        a=macros(REV/old);b=macros(WORK/new)
        check('macro_keys_'+Path(new).stem,set(a)==set(b))
        new_macros.update(b)
        changes.extend({'macro':key,'v67':a[key],'v68':b[key]} for key in a if a[key]!=b[key])
        for text in (maintex,sitex):
            check('current_macro_path_'+Path(new).stem+'_'+('main' if text==maintex else 'si'),
                  'revision_issue2_v68/'+new in text and old not in text)
    (QA/'macro_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
    pdfs={};preview=QA/'previews';preview.mkdir(exist_ok=True)
    for stem,text in [('musript_new_v68',maintex),('musript_new_v68_supplementary',sitex)]:
        log=(REV/(stem+'.log')).read_text(errors='replace')
        bad=[x for x in ['Undefined control sequence','Missing character:','Float too large',
                        'Overfull \\hbox','Overfull \\vbox','undefined references','undefined citations'] if x in log]
        check(stem+'_compile_clean',not bad,bad)
        doc=fitz.open(REV/(stem+'.pdf'));full='\n'.join(p.get_text() for p in doc)
        (QA/(stem+'_extracted.txt')).write_text(full)
        check(stem+'_no_empty_pages',all(len(p.get_text().strip())>50 for p in doc))
        outside=[]
        for i,page in enumerate(doc):
            for x0,y0,x1,y1,word,*_ in page.get_text('words'):
                if x0<-.5 or y0<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:
                    outside.append({'page':i+1,'word':word})
        check(stem+'_text_within_page',not outside,outside)
        compact=re.sub(r'\s+','',full)
        check(stem+'_540_designs_in_pdf','540个' in compact)
        if stem.endswith('supplementary'):
            check('system_cost_comparison_retained',all(x in compact for x in ['5.43','161/2775','3/65','−0.032','−0.004']))
            for i in (5,6):
                pages=[j for j,p in enumerate(doc) if f'Supplementary Figure {i}' in p.get_text()]
                check(f'supp_figure{i}_single_caption',len(pages)==1,pages)
                if pages:
                    doc[pages[0]].get_pixmap(dpi=135).save(preview/f'v68_supp_figure{i}_page.png')
        else:
            check('efc_primary_values_in_pdf',all(x in compact for x in ['7.64','496/2,775','6/65','16/75']))
        cells=[]
        for i,p in enumerate(doc):
            pix=p.get_pixmap(dpi=35)
            thumb=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
            cell=Image.new('RGB',(thumb.width,thumb.height+22),'white');cell.paste(thumb,(0,22))
            ImageDraw.Draw(cell).text((8,4),f'Page {i+1}',fill='black');cells.append(cell)
        width=max(x.width for x in cells);height=max(x.height for x in cells)
        sheet=Image.new('RGB',(width*4,height*((len(cells)+3)//4)),(230,230,230))
        for i,cell in enumerate(cells):sheet.paste(cell,((i%4)*width,(i//4)*height))
        sheet.save(preview/(stem+'_contact.png'))
        pdfs[stem]={'pages':len(doc),'sha256':sha(REV/(stem+'.pdf')),'bytes':(REV/(stem+'.pdf')).stat().st_size}
    figures={}
    for i in (5,6):
        stem=f'supp_figure{i}'
        check(stem+'_export_bundle',all((WORK/'figures'/f'{stem}.{ext}').exists() for ext in ['pdf','svg','png','tiff']))
        doc=fitz.open(WORK/'figures'/f'{stem}.pdf');page=doc[0]
        fonts=page.get_fonts(full=True)
        check(stem+'_editable_pdf',len(page.get_text())>100 and all(f[2]!='Type3' for f in fonts))
        check(stem+'_editable_svg','<text' in (WORK/'figures'/f'{stem}.svg').read_text())
        page.get_pixmap(dpi=150).save(preview/(stem+'_final.png'))
        figures[stem]={'width_mm':page.rect.width*25.4/72,'height_mm':page.rect.height*25.4/72,
                       'sha256':sha(WORK/'figures'/f'{stem}.pdf')}
    s=pd.read_csv(WORK/'source_data/all_540_design_summary.csv')
    p5=pd.read_csv(WORK/'source_data/supp_figure5/panel_a_capacity_composition.csv')
    check('supp5_country_and_composition_coverage',p5.country.nunique()==15 and len(p5)==90)
    p6=pd.read_csv(WORK/'source_data/supp_figure6/panels_c_f_all_chronology_ensembles.csv')
    check('supp6_full_design_coverage',p6.country.nunique()==15 and len(p6)==120)
    status='pass' if all(v['pass'] for v in checks.values()) else 'fail'
    result={'status':status,'checks':checks,'pdfs':pdfs,'figures':figures,
            'manual_visual_review':'required alongside these deterministic checks',
            'scope':'Second reviewer issue; existing EFC analysis preserved.'}
    (QA/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':status,'checks':len(checks),'failed':[k for k,v in checks.items() if not v['pass']],
                      'pdfs':pdfs,'figures':figures},ensure_ascii=False,indent=2))
    return 0 if status=='pass' else 2


if __name__=='__main__':
    raise SystemExit(main())
