# Reproduction Status

This document states what the public repository can and cannot reproduce from
the current GitHub code and Zenodo data package.

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

The archived Zenodo package contains processed datasets, frozen splits,
provenance files, source tables, and final figure source CSVs. It does not
include the complete final prediction matrices for every method/fold.

Therefore:

- Table 2/Figure 2 source values can be verified.
- A single prediction matrix can be evaluated with
  `scripts/evaluate_predictions.py` if that matrix is supplied separately.
- The full Table 2 benchmark cannot currently be recomputed end-to-end from
  GitHub plus the current Zenodo package alone.

The anchored full benchmark scripts are preserved:

```text
main/recompute_final_benchmark_from_predictions.py
main/build_final_four_metric_available_benchmark.py
```

They still require final prediction matrices and the missing final-workbench
dataset-audit dependency or an equivalent public manifest.

## Required Upgrade For Full Recompute

To upgrade from source-value verification to full benchmark recomputation, add
one of the following:

1. a prediction-matrix archive with checksums and documented layout; or
2. a complete training-to-prediction command chain that regenerates the final
   prediction matrices; plus
3. a public replacement for the missing final dataset audit manifest.
