"""Current R5: all eight technologies, duration expansion and cost decomposition."""
from pathlib import Path
import runpy

def main():
    source = Path(__file__).resolve().parents[2] / 'revision/full_manuscript_R5_20261114/code/figure_r5.py'
    runpy.run_path(str(source), run_name='__main__')

if __name__ == '__main__':
    main()
