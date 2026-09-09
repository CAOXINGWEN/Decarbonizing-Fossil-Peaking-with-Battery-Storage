from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parents[1]
REV = ROOT/'Decarbonizing-Fossil-Peaking-with-Battery-Storage-master/paper-code/docs/musrcipt/revision_nature_energy_v62'
BASE = ROOT/'submission/Nature_Energy_v70'
RELEASE = ROOT/'submission/Nature_Energy_v71'
DOC = REV
SOURCE_DATA_DIR = WORK/'source_data'
if WORK.parent.name == 'revision':
    RELEASE = WORK.parents[1]
    ROOT = RELEASE/'reproducibility/workspace'
    REV = ROOT/'Decarbonizing-Fossil-Peaking-with-Battery-Storage-master/paper-code/docs/musrcipt/revision_nature_energy_v62'
    BASE = WORK/'baseline_v70'
    DOC = RELEASE/'manuscript'
    SOURCE_DATA_DIR = RELEASE/'source_data'
USD_PER_EUR = 1.1300
CNY_PER_EUR = 8.1185043137255
CNY_PER_USD = CNY_PER_EUR / USD_PER_EUR
for p in ['qa','figures','source_data','evidence','baseline_figures']:
    (WORK/p).mkdir(parents=True, exist_ok=True)
