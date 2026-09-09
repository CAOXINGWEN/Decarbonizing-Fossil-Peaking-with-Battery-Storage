"""Run the frozen USD reporting scripts with repository-local paths."""
from pathlib import Path
import argparse, importlib.util, runpy, sys
repo = Path(__file__).resolve().parents[1]
folder = repo / 'analysis/reporting/currency'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('stage', choices=['figures', 'convert'])
p.add_argument('arguments', nargs='*')
a = p.parse_args()
sys.path.insert(0, str(folder))
spec = importlib.util.spec_from_file_location('config', folder / 'config_repository.py')
config = importlib.util.module_from_spec(spec)
sys.modules['config'] = config
spec.loader.exec_module(config)
script = folder / ('plot_figures_v71.py' if a.stage == 'figures' else 'convert_source_data.py')
if a.stage == 'convert' and not (config.BASE / 'source_data').is_dir():
    raise SystemExit('Place native-currency reporting inputs in native_reporting_inputs/source_data before conversion.')
sys.argv = [str(script), *a.arguments]
runpy.run_path(str(script), run_name='__main__')
