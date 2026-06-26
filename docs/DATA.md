# Data Availability

GitHub stores source code and lightweight metadata only. Large data are
available from Zenodo:

- DOI: <https://doi.org/10.5281/zenodo.20630224>
- Archive name: `GeneSPT_manuscript_data_20260610.zip`
- Zenodo MD5: `872c96ff8d5bd6ac565b1843f46145c8`
- SHA256: `E932BD33D04CE14D2AF4CEB33F7F0376B1867EE5CFA2F177E17D2C702993E701`

## Expected Zenodo Internal Folders

The local package staged for Zenodo uses these top-level entries:

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

The public evaluator wrapper under `scripts/evaluate_predictions.py` requires a
saved prediction matrix. The current Zenodo package does not include the full
set of final prediction matrices.

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

TODO: if final prediction matrices are archived under a separate folder, add
the exact internal path and checksum manifest here after that archive is
finalized.
