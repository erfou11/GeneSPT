# Reproducing Manuscript Analyses

This repository now separates two kinds of reproduction paths:

1. anchored final local scripts copied into `main/` and `scripts/`;
2. lightweight public evaluator wrappers under `src/genespt/` and
   `scripts/evaluate_predictions.py`.

Large processed matrices, frozen splits, prediction matrices, and figure/table
source data are archived on Zenodo rather than tracked in GitHub.

Data archive:

- DOI: <https://doi.org/10.5281/zenodo.20630224>
- Archive: `GeneSPT_manuscript_data_20260610.zip`
- Internal manifest: `FILE_MANIFEST_SHA256.csv`
- Dataset manifest: `DATASET_MANIFEST.csv`

Prediction-matrix addendum prepared for upload:

- Archive: `GeneSPT_zenodo_addendum_prediction_matrices_20260626.zip`
- SHA256: `8c771f0f72a3a5cc533fc620c74eca3329a90ef8b596b9c1e427c15b8581e93f`
- DOI/record: pending Zenodo upload; replace after publication.

See `docs/SOURCE_ANCHOR.md` before treating a script as public-runnable. It
records which files are byte-for-byte copies from the final local source tree
and which dependency is still missing.

## First-Line Main Performance Check

The first reviewer-facing reproduction check is the main benchmark source-value
verification, not the PSP ablation. After extracting the Zenodo archive, run:

```bash
docker run --rm \
  -v "$PWD:/workspace/GeneSPT" \
  -v "/path/to/GeneSPT_manuscript_data_20260610:/data" \
  -w /workspace/GeneSPT \
  genespt:paper \
  python scripts/verify_main_performance_from_zenodo.py --zenodo-root /data
```

This command verifies the archived primary benchmark source values used for
Table 2/Figure 2. It writes the legacy Figure 2 input expected by the anchored
final script, reruns Figure 2 source generation, and checks that the generated
source CSV exactly matches the Zenodo final Figure 2 source CSV.

Expected report:

```text
results/reproduction/main_performance_source_check/README.md
```

Boundary:

- This is a source-value verification path.
- It reports the main SPCC, SSIM, RMSE, and JS/JSD values for primary GeneSPT.
- It does not retrain models.
- It is complemented by the prediction-matrix addendum check below, which
  recomputes metrics from archived saved predictions.

## Prediction-Matrix Addendum Check

After extracting `GeneSPT_zenodo_addendum_prediction_matrices_20260626.zip`,
run:

```bash
docker run --rm \
  -v "$PWD:/workspace/GeneSPT" \
  -v "/path/to/GeneSPT_zenodo_addendum_prediction_matrices_20260626:/addendum" \
  -w /workspace/GeneSPT \
  genespt:paper \
  python scripts/verify_prediction_addendum.py --addendum-root /addendum
```

The script reads:

```text
manifests/PREDICTION_MATRIX_MANIFEST.csv
ground_truth/<dataset_id>/st_log1p_cpm.npy
prediction_matrices/<dataset_id>/<method>/fold<k>/prediction.npz
prediction_matrices/<dataset_id>/<method>/fold<k>/test_gene_idx.npy
```

It writes:

```text
results/reproduction/prediction_matrix_addendum_check/fold_metrics.csv
results/reproduction/prediction_matrix_addendum_check/aggregate_metrics.csv
```

Quick check:

```bash
docker run --rm \
  -v "$PWD:/workspace/GeneSPT" \
  -v "/path/to/GeneSPT_zenodo_addendum_prediction_matrices_20260626:/addendum" \
  -w /workspace/GeneSPT \
  genespt:paper \
  python scripts/verify_prediction_addendum.py \
    --addendum-root /addendum \
    --datasets MHPR_current_panel \
    --methods GeneSPT
```

## Expected Local Layout

The public evaluator expects:

```text
data/processed/<dataset_id>/
  st_matrix.npy
  scrna_matrix.npy
  coordinates.npy
  gene_names.txt

splits/<dataset_id>/
  fold0_train_gene_idx.npy
  fold0_val_gene_idx.npy
  fold0_test_gene_idx.npy

results/predictions/<dataset_id>/<method>/fold0/
  prediction_test.npy
```

The anchored final scripts preserve the local final-workbench convention:

```text
/workspace/GeneSPT/data/<dataset_id>/
  Spatial_count.txt
  scRNA_count.txt
  Locations.txt

/workspace/GeneSPT/results/imformation/
  final_multidataset_masks/
  final_recomputed_prediction_matrices/
  final_manuscript_figures/
```

Using the Docker command in `README.md` mounts the repository at
`/workspace/GeneSPT` and preserves those defaults.

## Table 2: Primary Benchmark

Purpose: primary strict whole-gene benchmark on Vis9A, HBC, and Cell2location
mouse brain.

Anchored scripts:

```text
main/build_final_four_metric_available_benchmark.py
main/recompute_final_benchmark_from_predictions.py
```

Current status:

- These files are preserved as byte-identical final local source copies.
- They import `main/run_final_multidataset_fold0_gate.py`.
- `main/run_final_multidataset_fold0_gate.py` imports
  `run_final_dataset_audit.py`, which was not present in the local final source
  tree at refactor time.
- Therefore, do not claim the full Table 2 builder is one-command runnable
  until the missing dataset-audit dependency or equivalent manifest is restored.

Current reviewer-facing check:

```bash
python scripts/verify_main_performance_from_zenodo.py --zenodo-root /data
```

This validates the archived primary benchmark source values and the Figure 2
source-generation path.

Prediction-matrix recomputation check:

```bash
python scripts/verify_prediction_addendum.py --addendum-root /addendum
```

This recomputes fold-level and aggregate metrics from the saved prediction
matrices in the addendum. The original anchored Table 2 builder is still kept
as a source-anchored artifact and still depends on the final-workbench audit
module noted above.

Available public metric recomputation for one saved method/fold:

```bash
python scripts/evaluate_predictions.py \
  --true data/processed/Vis9A_D7_spaim_effective4470/st_matrix.npy \
  --pred results/predictions/Vis9A_D7_spaim_effective4470/GeneSPT/fold0/prediction_test.npy \
  --test-indices splits/Vis9A_D7_spaim_effective4470/fold0_test_gene_idx.npy \
  --out results/evaluation/Vis9A_D7_spaim_effective4470/GeneSPT/fold0_summary.csv
```

Sanity checks:

- SPCC is Spearman correlation across spatial locations.
- RMSE, JS/JSD, and SSIM come from the anchored final evaluator logic.
- SSIM is not multiplied by 10.
- Evaluated genes must match the frozen test-gene index vector.

## Figure 2: Primary Benchmark Plot

Purpose: visualize Table 2 primary benchmark results.

Anchored script:

```bash
python main/generate_figure2_primary_dotplot.py
```

Expected final-workbench inputs:

```text
results/imformation/table1_primary_benchmark_final.csv
```

Expected outputs:

```text
final_output/final_main_results/figure2_primary_benchmark_dotplot.pdf
final_output/final_main_results/figure2_primary_benchmark_dotplot.png
final_output/final_main_results/figure2_primary_benchmark_dotplot_source.csv
```

## Figure 3: Descriptor and PSP Ablations

Purpose: descriptor controls, GeneSPT-GC versus full GeneSPT, and PSP mechanism
controls.

Anchored method scripts:

```bash
python main/run_gene_conditioned_decoder_folds012_validation.py --folds 0,1,2
python main/run_predictable_spatial_program_folds012.py --folds 0 1 2
```

Anchored fold0 PSP script:

```bash
python main/run_predictable_spatial_program_transfer_fold0.py
```

Zenodo/source-data files:

```text
results_source_data/figure3_panelB_psp_ablation_clean.csv
results_source_data/supplementary_table_s3_descriptor_psp_and_mechanism_controls.csv
```

Sanity checks:

- PSP spatial programs are estimated from training genes only.
- PSP component selection uses validation genes only.
- Test genes are evaluated after model and readout choices are fixed.

## Figure 4: HBC Representative Maps

Purpose: show HBC representative held-out gene spatial maps.

Anchored script:

```bash
python scripts/generate_figure4_hbc_representative_maps.py
```

Expected final-workbench inputs include HBC processed matrices, frozen split
files, final prediction matrices, and prior source CSVs under `final_output/`
or `results/imformation/`.

Sanity checks:

- Figure labels should use manuscript terminology.
- Maps are visualization artifacts; reported SPCC should come from the
  centralized evaluator, not from plotted pixels.

## Figure 5: Cross-Platform Distributions

Purpose: summarize per-gene performance on seqFISH+ cortex/SVZ, MHPR/MERFISH,
and MVC/STARmap.

Anchored script:

```bash
python main/generate_figure5_cross_platform_per_gene_violins.py
```

Expected final-workbench inputs include:

```text
final_output/figure5_cross_platform_raw_metrics_source.csv
results/imformation/final_available_datasets_four_metric_gene_level.csv
```

Sanity checks:

- Cross-platform datasets are evaluated with the same central metric code.
- Per-gene distributions should be based on per-gene evaluator/source CSVs.

## Figure 6: Downstream Validation

Purpose: evaluate downstream utility of predicted held-out genes.

Anchored script:

```bash
python scripts/generate_main_downstream_figure6_final.py
```

Expected final-workbench inputs include downstream source tables under:

```text
final_output/downstream_validation_supplement/
final_output/downstream_upgraded_labels/
final_output/downstream_leiden_sensitivity/
final_output/downstream_leiden_primary_sensitivity/
```

Sanity checks:

- Downstream labels and annotations must trace to label-provenance files.
- Predicted genes used downstream must be from frozen held-out test genes.

## Supplementary Tables S1-S3

Anchored script:

```bash
python scripts/generate_supplyment_tables.py
```

Zenodo source files:

```text
results_source_data/supplementary_table_s1_dataset_provenance_preprocessing_splits_label_provenance.csv
results_source_data/supplementary_table_s2_full_benchmark_metrics_and_method_availability.csv
results_source_data/supplementary_table_s3_descriptor_psp_and_mechanism_controls.csv
```

Sanity checks:

- S1 covers provenance, preprocessing, frozen splits, and label provenance.
- S2 covers full benchmark metrics and method availability.
- S3 covers descriptor ablation, PSP ablation, and mechanism controls.

## Integrity Rules

- Test genes must not be used for model fitting, PSP program estimation,
  validation screening, fusion/readout selection, or hyperparameter tuning.
- Validation genes may be used for model selection, PSP component screening,
  and fusion/readout selection only.
- Reported SPCC, RMSE, JS/JSD, and SSIM values should come from the anchored
  final evaluator.
