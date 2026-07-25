# Reproduction status

## Available without data

- import and environment smoke tests;
- synthetic centralized-metric tests;
- exact GeneSPT formal source, six baseline adapters and runtime patches or wrappers;
- machine-readable dataset, method and environment configuration;
- plotting and downstream-analysis source.

## Available with the data archive

- SHA256 verification of 210 benchmark predictions and 30 fold-specific truth
  matrices;
- SHA256 verification of 90 Figure 3 mechanism predictions;
- one-command recomputation of benchmark and mechanism SPCC, standardized
  RMSE, JSD and SSIM;
- all 180 external-baseline task commands and audit fields;
- all frozen train/validation/test split hashes;
- all 30 readout locks and 60 candidate tables for GeneSPT/GeneSPT-GC;
- source data for Supplementary Tables S1-S3 and Figures 2-6.

## Full training requirements

Metric recomputation uses the archived prediction and truth matrices. Full
model training requires the recorded upstream repositories, processed inputs,
CUDA environment, and compute budget.

See `docs/REPRODUCE_PROTOCOL_A.md` for executable commands and
`docs/BASELINE_ADAPTATION.md` for baseline fairness details.
