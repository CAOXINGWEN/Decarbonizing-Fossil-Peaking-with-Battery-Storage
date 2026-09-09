# EFC Source Data

All monetary columns in `paired_ledger_panel.csv`, including valuation differences and capacity-service credits, are in USD kW⁻¹ yr⁻¹. The reporting conversion is 1.1300 USD per EUR. `paired_ledger_panel.units.json` defines the monetary fields and their identities.

The paired ledger satisfies `efc_value − system_cost_value = accounting_difference` and `incremental_efc_credit − avoided_residual_peak_credit = accounting_difference_from_capacity_terms`. The two differences agree to numerical precision.

For Figure 5a and c, `all_duration_efc_records.csv.gz` records the selected technology and configuration separately for each chronology and the matched cost scale for each paired observation. `figure5_system_duration_exposure.csv` summarizes the within-system exposure. `figure5_configuration_cost_scales.csv` supplies the configuration-level fixed costs, relative gaps and absolute decade movements. The scale is the mean absolute 2025–2035 movement in the annualized fixed-cost gap of the two selected configurations relative to additional 4 h LFP.

For Figure 5b, `../event_cluster_surface/v60_cluster_intensity_system_year.csv.gz` records the selected configurations. Its surface uses 24 h A-CAES and 28–168 h cavern hydrogen. The fixed normalization benchmark is recorded in `../event_cluster_surface/qa.json`.
