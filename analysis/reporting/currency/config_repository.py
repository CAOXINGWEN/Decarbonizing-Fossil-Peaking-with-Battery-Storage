from pathlib import Path
REPOSITORY = Path(__file__).resolve().parents[3]
ROOT = REPOSITORY / "analysis/workspace"
WORK = REPOSITORY / "generated/currency"
REV = ROOT / "Decarbonizing-Fossil-Peaking-with-Battery-Storage-master/paper-code/docs/musrcipt/revision_nature_energy_v62"
BASE = REPOSITORY / "native_reporting_inputs"
RELEASE = REPOSITORY
DOC = WORK
SOURCE_DATA_DIR = WORK / "source_data"
USD_PER_EUR = 1.1300
CNY_PER_EUR = 8.1185043137255
CNY_PER_USD = CNY_PER_EUR / USD_PER_EUR
for name in ["qa", "figures", "source_data", "evidence", "baseline_figures"]:
    (WORK / name).mkdir(parents=True, exist_ok=True)
