"""Verify the issue-1 data/figure/text contract and preservation of v66."""
from pathlib import Path
import json,hashlib,re
import numpy as np
import pandas as pd
import fitz

WORK=Path(__file__).resolve().parents[1];ROOT=WORK.parent
SRC=WORK/'source_data';NAT=WORK/'natural_results';QA=WORK/'qa'
checks={}
def check(name,value,detail=None):
    checks[name]={'pass':bool(value),'detail':detail}
    assert value,(name,detail)

hashes=json.loads((QA/'v66_original_hashes.json').read_text())
for name,expected in hashes.items():
    check('preserved_'+name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected)

natural=json.loads((NAT/'pecd_v42_75y_parity_summary.json').read_text())
check('natural_primary_span',np.isclose(natural['natural_q95_q05_span_eur_kw_year'],7.63922612726801,atol=1e-10))
check('natural_primary_pairs',natural['all_pairs_reaching_cost_scale']==496)
check('natural_primary_decades',natural['matched_decades_erasing_learning']==6)
check('natural_primary_crossings',natural['economic_crossovers_by_cost_year']=={'2025':1,'2030':1,'2035':16})
check('natural_worst_net',np.isclose(natural['worst_matched_decade']['net_change_after_learning_eur_kw_year'],-5.973347323392501))

focal=pd.read_csv(SRC/'figure3_24h_value_records.csv')
check('all_controlled_observations_retained',focal.groupby('regime').size().to_dict()=={'China':26,'Europe':132,'Japan':27})
expected={'China':(59.503987,26),'Europe':(5.477395,68),'Japan':(4.785756,11)}
for regime,(median,hits) in expected.items():
    g=focal[focal.regime.eq(regime)]
    check('preserved_efc_headline_'+regime,np.isclose(g.sequencing_shift_eur_kw_year.median(),median,atol=1e-5) and g.above_parity.sum()==hits)
check('all_cycle_effects_negative',focal.cycle_effect.lt(0).all())
decomp=pd.read_csv(SRC/'figure3_value_component_effects.csv.gz')
err=float(abs(decomp[['fixed_cost','variable_om','charging','energy_service','firm_service']].sum(axis=1)-decomp.total).max())
check('efc_decomposition_closes',err<1e-8,err)
recharge=pd.read_csv(SRC/'recharge_efc_records.csv')
p=recharge[recharge.recharge_multiplier.eq(1)].merge(focal,on=['regime','system_id','observation_year'],validate='1:1')
err=float(abs(p.chronology_effect-p.signed_sequencing_effect_eur_kw_year).max())
check('recharge_baseline_reproduced',len(p)==185 and err<1e-6,err)
surface=json.loads((WORK/'surface/qa.json').read_text())
check('full_surface_reproduced',surface['status']=='pass' and surface['actual_system_duration_rows']==31080,surface['maximum_registered_reproduction_difference_eur_kw_year'])

models=pd.read_csv(SRC/'model_comparison_both_ledgers_and_difference.csv')
for response,target in [('efc_value',.1103169056),('system_cost_value',-.0318882645)]:
    joint=models[models.response.eq(response)&models.model.eq('joint_geometry')].iloc[0]
    check(response+'_loco_joint',np.isclose(joint.loco_r2,target,atol=1e-9))
    check(response+'_all_four_models',models[models.response.eq(response)].shape[0]==4)
ledger=json.loads((SRC/'diagnostic_summary.json').read_text())
check('efc_natural_response_preserved',ledger['efc_original_response_max_error']<1e-10)
check('paired_ledger_accounting_identity',ledger['identity_max_error_eur_kw_year']<1e-8)

pdf_summary={}
for name in ['musript_new_v67','musript_new_v67_supplementary']:
    tex=(ROOT/f'{name}.tex').read_text();log=(ROOT/f'{name}.log').read_text()
    check(name+'_compile_no_errors',not any(s in log for s in ['Undefined control sequence','Missing character:','Float too large','Overfull \\hbox','Overfull \\vbox']))
    doc=fitz.open(ROOT/f'{name}.pdf')
    full='\n'.join(page.get_text() for page in doc)
    (QA/f'{name}_extracted.txt').write_text(full)
    check(name+'_no_empty_pages',all(len(page.get_text().strip())>50 for page in doc))
    outside=[]
    for i,page in enumerate(doc):
        for word in page.get_text('words'):
            x0,y0,x1,y1=word[:4]
            if x0<-.5 or y0<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:outside.append((i+1,word[4]))
    check(name+'_text_within_page',not outside,outside)
    if name=='musript_new_v67':
        check('main_no_old_natural_endpoint',not any(x in tex for x in ['5.43','均为 0/75','有 72 个不利转移','3/65']))
        check('main_primary_natural_values_present',all(x in full for x in ['7.64','496/2,775','6/65','16/75']))
        check('main_efc_workflow_and_comparison_reference',all(x in full for x in ['系统成本', '最小化确定调度', '补充表7', '国家内年度偏差']))
    else:
        check('si_alternative_natural_values_present',all(x in full for x in ['5.43','161/2775','3/65']))
        check('si_single_comparison_table_and_negative_prediction_result','Supplementary Table 7' in full and 'Supplementary Table 8' not in full and all(x in full for x in ['−0.032', '−0.004', '−0.016', '−0.001']))
    pdf_summary[name]={'pages':len(doc),'bytes':(ROOT/f'{name}.pdf').stat().st_size,'sha256':hashlib.sha256((ROOT/f'{name}.pdf').read_bytes()).hexdigest()}

figure_summary={}
for name in ['main_figure3','main_figure4','main_figure5','main_figure6','supp_figure1','supp_figure7']:
    check(name+'_export_bundle',all((WORK/'figures'/f'{name}.{ext}').exists() for ext in ['pdf','svg','png']))
    doc=fitz.open(WORK/'figures'/f'{name}.pdf')
    fonts=doc[0].get_fonts(full=True)
    check(name+'_editable_pdf_text',len(doc[0].get_text())>100 and all(f[2]!='Type3' for f in fonts))
    figure_summary[name]={'width_mm':doc[0].rect.width*25.4/72,'height_mm':doc[0].rect.height*25.4/72}

result={'status':'pass','checks':checks,'pdfs':pdf_summary,'figures':figure_summary,
    'scope_limit':'First reviewer issue only; this is not a full submission-readiness certification.'}
(QA/'v67_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'status':'pass','checks':len(checks),'pdfs':pdf_summary},ensure_ascii=False,indent=2))
