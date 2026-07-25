# Supplementary Table S2

`supplementary_table_s2_formal_benchmark.csv` is the single formal benchmark table. Its 42 rows cover six datasets and seven methods, with one row per dataset-method pair.

The table summarizes verified strict whole-gene holdout evidence.

## Reported fields

`dataset` is the manuscript-facing dataset name, `dataset_id` is the benchmark dataset identifier, and `role` distinguishes primary and cross-platform benchmark groups. `folds` is the number of frozen folds. `coverage` is the minimum fold coverage.

For SPCC, RMSE, JSD, and SSIM, each `*_mean` field is the arithmetic mean across the five fold summaries and each `*_fold_sd_ddof0` field is their population standard deviation. SPCC and SSIM ranks are descending; RMSE and JSD ranks are ascending. Rank 1 is best, and each `*_is_second_best` field marks rank 2.

Public filenames, byte sizes, row counts, and SHA256 checksums are recorded in `manifest.json`.
