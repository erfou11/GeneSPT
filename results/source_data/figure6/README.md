# Figure 6 Panel D source data

These lightweight files correspond to the frozen Protocol A MHPR/MERFISH
completed-matrix clustering analysis:

- `mhpr_completed_matrix_fold_metrics.csv`: seven methods by five frozen folds;
- `mhpr_completed_matrix_summary.csv`: mean, sample SD and seven-method ranks;
- `mhpr_completed_matrix_input_hashes.csv`: protocol and archive-input hashes;
- `mhpr_completed_matrix_run_manifest.json`: output hashes and run checks.
- `mhpr_completed_matrix_repeatability_check.json`: deterministic rerun check.
- `mhpr_completed_matrix_test_gene_participation.csv`: predicted test-gene
  participation in the completed matrices.

The same completed-matrix rule and clustering configuration are applied to all
methods. Author `Cell_class` labels are used only after clustering to calculate
ARI, AMI, NMI and homogeneity. The large consolidated source table for Figure 6
Panels B-C and the prediction matrices are distributed through the Zenodo
archive.
