# Reproduction Status

This document states what the public repository can and cannot reproduce from
the current GitHub code and the combined Zenodo data package, `GeneSPT.zip`.
The combined package contains the original manuscript data archive and the
prediction-matrix archive as two inner ZIP files.

## Publicly Runnable Checks

### Environment Smoke Test

```bash
python tests/smoke_test.py
```

Expected output:

```text
smoke_test=ok
```

### Main Performance Source-Value Check

```bash
docker run --rm \
  -v "$PWD:/workspace/GeneSPT" \
  -v "/path/to/GeneSPT_manuscript_data_20260610:/data" \
  -w /workspace/GeneSPT \
  genespt:paper \
  python scripts/verify_main_performance_from_zenodo.py --zenodo-root /data
```

This verifies the primary benchmark source values used for Table 2/Figure 2.
Use the extracted inner `GeneSPT_manuscript_data_20260610.zip` folder as
`/data`.
It also reruns the anchored Figure 2 source-generation script and compares the
generated source CSV to the Zenodo final Figure 2 source CSV.

Expected output includes:

```text
figure2_exact_hash_match=True
```

Report path:

```text
results/reproduction/main_performance_source_check/README.md
```

### Prediction-Matrix Addendum Check

```bash
docker run --rm \
  -v "$PWD:/workspace/GeneSPT" \
  -v "/path/to/GeneSPT_zenodo_addendum_prediction_matrices_20260626:/addendum" \
  -w /workspace/GeneSPT \
  genespt:paper \
  python scripts/verify_prediction_addendum.py --addendum-root /addendum
```

This recomputes SPCC, RMSE, JS/JSD, and SSIM from archived saved prediction
matrices and evaluator-ready ground truth arrays.
Use the extracted inner
`GeneSPT_zenodo_addendum_prediction_matrices_20260626.zip` folder as
`/addendum`.

Quick local validation performed on 2026-06-26:

```text
MHPR_current_panel / GeneSPT / 5 folds:
SPCC_mean=0.247304, RMSE_mean=1.221133, JS/JSD_mean=0.296925, SSIM_mean=0.183256
manifest deltas: SPCC=0.0, RMSE=-9.91e-11, JS/JSD=1.18e-11, SSIM=5.71e-11

MHPR_current_panel / SpaGE / 5 folds:
SPCC_mean=0.213669, RMSE_mean=1.234999, JS/JSD_mean=0.324172, SSIM_mean=0.177536
manifest deltas: SPCC=1.11e-16, RMSE=-2.26e-10, JS/JSD=4.69e-11, SSIM=1.63e-10
```

## Primary GeneSPT Values Checked

The main source-value check reports the primary GeneSPT metrics:

```text
dataset                   SPCC_mean           raw_SSIM_mean        RMSE_mean          JS_JSD_mean
Vis9A                     0.1929435518593177  0.0569371313190875  1.3010974202927632 0.4523600946347454
HBC                       0.1192097487302816  0.0335107004404702  1.347076722599781  0.4889606245129387
Cell2location mouse brain 0.18163487598437938 0.0509362568311357  1.2925126466596484 0.34293048964922246
```

## Method/Ablation Startup Checks

These commands are secondary checks. They validate that the method code can read
processed matrices and frozen splits when explicit Zenodo paths are provided.
They are not the first-line main performance reproduction path.

GeneSPT-GC:

```bash
python main/run_gene_conditioned_decoder_folds012_validation.py --folds 0,1,2
```

PSP ablation/mechanism validation:

```bash
python main/run_predictable_spatial_program_folds012.py --folds 0 1 2
```

The PSP script expects space-separated fold IDs.

## Current Full-Recompute Boundary

The combined Zenodo package `GeneSPT.zip` contains two inner archives. The
manuscript data inner archive contains processed datasets, frozen splits,
provenance files, source tables, and final figure source CSVs. The prediction
matrix inner archive contains final saved prediction matrices, evaluator-ready
ground truth matrices, and public manifests.

Therefore:

- Table 2/Figure 2 source values can be verified from the main package.
- Saved prediction-matrix metrics can be recomputed from the addendum with
  `scripts/verify_prediction_addendum.py`.
- One-off prediction matrices can be evaluated with
  `scripts/evaluate_predictions.py`.
- Full training-from-raw regeneration of every final prediction matrix remains
  outside the lightweight public check path.

The anchored full benchmark scripts are preserved:

```text
main/recompute_final_benchmark_from_predictions.py
main/build_final_four_metric_available_benchmark.py
```

They still require the missing final-workbench dataset-audit dependency or an
equivalent public manifest if those exact anchored builders are used directly.
The prediction-matrix inner archive provides a public manifest for the
wrapper-based recomputation path.

## Required Upgrade For Full Recompute

To upgrade beyond saved-prediction recomputation, add one of the following:

1. a complete training-to-prediction command chain that regenerates the final
   prediction matrices; plus
2. a public replacement for the missing final dataset audit manifest if the
   exact anchored final benchmark builders remain the first-line path.
