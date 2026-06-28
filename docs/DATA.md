# Data Availability

GitHub stores source code and lightweight metadata only. Large data are
available from Zenodo. The reviewer-facing upload is a single combined archive:

- Archive name: `GeneSPT.zip`
- DOI: <https://doi.org/10.5281/zenodo.20965448>
- Zenodo record: <https://zenodo.org/records/20965448>
- MD5: `bab51420a0a961dc6a9d85f1f980b393`
- SHA256: `0ae1a6384aca693c173b61a315c6426174106a863d07ae5523b8cc2ac7fa0351`

`GeneSPT.zip` contains:

- `README_combined_upload.txt`
- `GeneSPT_manuscript_data_20260610.zip`
- `GeneSPT_zenodo_addendum_prediction_matrices_20260626.zip`

The original data-only archive remains available for provenance:

- DOI: <https://doi.org/10.5281/zenodo.20630224>
- Archive name: `GeneSPT_manuscript_data_20260610.zip`
- Zenodo MD5: `872c96ff8d5bd6ac565b1843f46145c8`
- SHA256: `E932BD33D04CE14D2AF4CEB33F7F0376B1867EE5CFA2F177E17D2C702993E701`

The prediction-matrix inner archive is:

- Archive name: `GeneSPT_zenodo_addendum_prediction_matrices_20260626.zip`
- SHA256: `8c771f0f72a3a5cc533fc620c74eca3329a90ef8b596b9c1e427c15b8581e93f`

## Expected Zenodo Internal Folders

After extracting `GeneSPT.zip`, extract
`GeneSPT_manuscript_data_20260610.zip`. The extracted manuscript data archive
uses these top-level entries:

```text
processed_datasets/
frozen_splits/
label_provenance/
provenance_reports/
results_source_data/
CHECKSUMS_SHA256.txt
DATASET_MANIFEST.csv
FILE_MANIFEST_SHA256.csv
PACKAGE_BUILD_SUMMARY.json
README.md
```

`FILE_MANIFEST_SHA256.csv` is the file-level checksum manifest. Use it to
verify archive completeness after download.

## Expected Local Runtime Layout

The command-line examples in this repository expect a local runtime layout:

```text
data/processed/<dataset_id>/
splits/<dataset_id>/
results/predictions/<dataset_id>/<method>/fold<k>/
results/evaluation/<dataset_id>/<method>/
```

The current Zenodo package stores processed matrices as the final-workbench
text files under `processed_datasets/`. The main benchmark source-value check
can be run directly against the extracted Zenodo folder:

```bash
python scripts/verify_main_performance_from_zenodo.py --zenodo-root /data
```

After extracting `GeneSPT.zip`, extract
`GeneSPT_zenodo_addendum_prediction_matrices_20260626.zip`. The extracted
prediction-matrix archive uses these top-level entries:

```text
prediction_matrices/
mechanism_ablation_prediction_matrices/
ground_truth/
manifests/
ADDENDUM_VALIDATION_REPORT.md
CHECKSUMS_SHA256.txt
PACKAGE_BUILD_SUMMARY.json
README.md
```

The public evaluator wrapper under `scripts/evaluate_predictions.py` requires a
saved prediction matrix. The addendum provides final benchmark prediction
matrices plus evaluator-ready ground-truth arrays.

## Manuscript Datasets

Primary datasets:

- Vis9A: `Vis9A_D7_spaim_effective4470`
- HBC: `HBC_shared16112`
- Cell2location mouse brain: `Cell2location_mouse_brain_ST8059048_shared12819`

Cross-platform datasets:

- seqFISH+ cortex/SVZ:
  `seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000`
- MHPR/MERFISH: `MHPR_current_panel`
- MVC/STARmap: `MVC_shared981`

## Original Source Notes

- Vis9A: spatial source `GSE161318/GSM4904761`; scRNA reference
  `GSE159500/GSM4831163`.
- HBC: spatial section `1142243F`; scRNA reference `CID3586/GSE176078`.
- Cell2location mouse brain: Kleshchevnikov et al., Nature Biotechnology
  40, 661-671 (2022), DOI `10.1038/s41587-021-01139-4`; mouse brain Visium
  `E-MTAB-11114`; mouse brain snRNA-seq `E-MTAB-11115`.
- seqFISH+ cortex/SVZ: Eng et al. seqFISH+ mouse cortex/SVZ data with internal
  processed ID `seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000`.
- MHPR/MERFISH: Dryad DOI `10.5061/dryad.8t8s248`.
- MVC/STARmap: STARmap mouse visual cortex archive, Zenodo record `10698912`.

## Source Tables

The current Zenodo package includes these source-data CSVs under
`results_source_data/`:

- `benchmark_metrics_primary_clean.csv`
- `benchmark_metrics_cross_platform_clean.csv`
- `figure3_panelB_psp_ablation_clean.csv`
- `supplementary_table_s1_dataset_provenance_preprocessing_splits_label_provenance.csv`
- `supplementary_table_s2_full_benchmark_metrics_and_method_availability.csv`
- `supplementary_table_s3_descriptor_psp_and_mechanism_controls.csv`

## Prediction Matrix Addendum

The addendum includes:

- `prediction_matrices/<dataset_id>/<method>/fold<k>/prediction.npz`
- `prediction_matrices/<dataset_id>/<method>/fold<k>/test_gene_idx.npy`
- `prediction_matrices/<dataset_id>/<method>/fold<k>/metadata.json`
- `ground_truth/<dataset_id>/st_log1p_cpm.npy`
- `ground_truth/<dataset_id>/gene_names.txt`
- `manifests/PREDICTION_MATRIX_MANIFEST.csv`
- `manifests/MECHANISM_ABLATION_MATRIX_MANIFEST.csv`
- `manifests/DATASET_AUDIT_MANIFEST.csv`
- `manifests/GROUND_TRUTH_MATRIX_MANIFEST.csv`

Use the extracted prediction-matrix archive as `--addendum-root`:

```bash
python scripts/verify_prediction_addendum.py --addendum-root /path/to/GeneSPT_zenodo_addendum_prediction_matrices_20260626
```

The script writes fold-level metrics and dataset-method aggregate metrics to
`results/reproduction/prediction_matrix_addendum_check/`.
