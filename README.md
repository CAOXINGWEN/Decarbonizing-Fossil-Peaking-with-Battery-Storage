# Clustered energy droughts can erase a decade of long-duration storage learning

Research data and analysis code accompanying the manuscript by **Xingwen Cao** and **Haoran Zhang**. Correspondence to Haoran Zhang at [h.zhang@pku.edu.cn](mailto:h.zhang@pku.edu.cn).

## Contents

- `source_data/` contains the processed data underlying the main and supplementary results, including the USD valuation ledger, natural-year comparisons, 540 procurement designs, regression summaries and figure inputs.
- `analysis/workspace/` contains the frozen analysis source and parameter tables. Historical directory names retain the module imports and source-to-result paths used in the study. The release associated with the present manuscript is **v75**.
- `analysis/reporting/` contains the USD reporting scripts and the updated geographical layout code.
- `reproducibility/` contains numerical checks and recorded software requirements.
- `metadata/` provides the analysis entry points and file provenance. `checksums.sha256` records this release's file hashes.

## Verify the reported results

With Python 3.9 or a compatible environment, install `reproducibility/requirements-verification.txt`, then run from the repository root

```bash
python reproducibility/verify_data.py
python reproducibility/verify_headlines.py
```

These commands use the included processed data. They check the EFC and system-cost ledger identities, the fixed and configuration-matched cost scales, natural-year comparisons and procurement feasibility. Results are written into `verification/`.

## Analysis workflows

`metadata/analysis_entrypoints.json` identifies the original scripts for weather-input reconstruction, storage dispatch, EFC postprocessing, capacity expansion and figure generation. Their paths are relative to `analysis/workspace/`. Full analysis uses the dependencies in `requirements-core.txt`; raw weather and geographical preprocessing additionally uses `requirements-raw.txt`. Gurobi-based dispatch requires a Gurobi licence. Capacity expansion uses HiGHS through SciPy.

To rerun an original analysis, obtain its input files using the source registry and place them at the paths relative to `analysis/workspace/`. Run the selected entry point from that directory. Natural-year dispatch precedes EFC postprocessing. The capacity runner covers 540 designs on the 2, 4 and 6 h planning grids; its cached scenario files and solved outputs are created during execution. Figure scripts read the derived tables produced by those analysis stages.

The model's historical calculations use their recorded native monetary fields. The manuscript and `source_data/` report USD using 1.1300 USD per EUR. `analysis/reporting/currency/convert_source_data.py` includes the corrected monetary fields from the paired ledger. For repository-local reporting runs, `reproducibility/run_reporting.py` selects `config_repository.py`; the corresponding native reporting inputs belong in `native_reporting_inputs/source_data/`. Reconstructed analysis tables remain under `analysis/workspace/`. Output is written to `generated/currency/`.

## Data sources and access

The source registry in `source_data/cost_sources/source_registry.csv` records the 91 source assets, providers, file checksums, original paths and access conditions. `technology_cost_source_map_53.csv` maps the 53 technology configurations to their sources. Public weather and power-system inputs can be obtained from their listed providers.

The Chinese provincial gas-peaking and curtailment inputs were supplied to the authors by State Grid Corporation of China under non-public access arrangements. Access to the Japanese community records follows the original provider's conditions. Enquiries about these inputs should be directed to Haoran Zhang at h.zhang@pku.edu.cn. This repository distributes the authors' processed research outputs and analysis code; provider-controlled original files follow those access routes.

## Version and verification

The processed Source Data are preserved byte for byte from the checked v74 research results. The v75 manuscript updates the title, authors, affiliations, correspondence, acknowledgements and repository statements. The repository verification records state which checks were run. Full optimization is separate from the arithmetic and archived feasibility checks.
