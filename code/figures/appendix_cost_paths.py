"""Entry point for the eight-technology cost-path appendix."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'revision/hydrogen_tank_only_20261114/code'))
from cost_paths import main

if __name__=='__main__':main()
