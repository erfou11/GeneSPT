# Reviewer audit and metric reproduction

The public audit path operates on frozen evaluator-ready matrices. It does not
train a model, require the private workbench, or modify an extracted archive.

## Formal benchmark audit

Run the fail-closed archive verifier first:

```bash
python scripts/reproducibility/verify_protocol_a_release.py \
  --archive-root ../GeneSPT_reviewer_archive
```

The verifier requires:

- 210 prediction matrices: 6 datasets x 5 folds x 7 benchmark methods;
- 30 fold-specific final-test truth matrices;
- 180 completed external-baseline task records;
- 30 frozen input/split records;
- 60 GeneSPT/GeneSPT-GC readout-selection records;
- complete finite predictions with no formal fallback; and
- matching SHA256 commitments for arrays, indices and public selection locks.

It also rejects unexpected methods, missing coverage, workstation paths and
stale method labels in the reviewer-facing Protocol A tree.

Recompute all four metrics and benchmark ranks with:

```bash
python scripts/reproducibility/recompute_protocol_a_benchmark.py \
  --archive-root ../GeneSPT_reviewer_archive \
  --output-dir results/recomputed_protocol_a
```

Metric calculations come only from `src/genespt/metrics.py`. Eligibility is
defined by truth, so a method cannot improve coverage by producing missing or
constant predictions. Fold values are medians across truth-eligible genes;
reported five-fold values are arithmetic means of the five fold medians.

## Validation-selection evidence

The archive stores one public selection lock for every dataset and fold. Each
lock contains the fixed 57-candidate family for GeneSPT and GeneSPT-GC, the
selected row, guard results, code hashes and explicit flags showing that final-
test predictions and truth were not accessed before selection. Candidate
tables, validation metric audits and train-gene fitting traces are stored under
`protocol_a_reproducibility/readout_selections/`.

External baselines do not receive this model-specific readout and remain at raw
identity output. Figure 3 mechanism controls also remain identity-readout
comparisons so the PSP toggle is not confounded by a different output layer.

## Baseline adaptation evidence

The original 150 scheduler-managed tasks are indexed in
`FORMAL_BASELINE_RUN_MANIFEST.csv`; the 30 stAI folds and their raw output
hashes are indexed in `STAI_FORMAL_ADOPTION_MANIFEST.json`. The release
exporter combines them into the 180-row archive task manifest. Upstream
revisions and parameters are in
`configs/protocol_a_baseline_versions.yaml`; runtime changes are exposed as
small patches under `baseline_adapters/patches/`. Every formal adapter fails on
axis mismatch, non-finite output, incomplete final-test coverage or hidden-gene
overlap. No truth-copy, zero-fill, KMeans or checkpoint-reuse fallback is
accepted.

## Figure 3 mechanism audit

Descriptor controls, matched full-versus-GC comparisons and PSP controls are
reported separately from the main benchmark. Their fold-level and five-fold
values are in Supplementary Table S3 and the current Figure 3 source data.
Figure 3B/3C use identity readout; the main benchmark readout must not be mixed
into those mechanism comparisons.

Recompute all 90 mechanism rows and verify them against S3 with:

```bash
python scripts/reproducibility/recompute_protocol_a_mechanism.py \
  --archive-root ../GeneSPT_reviewer_archive \
  --output-dir results/recomputed_figure3
```

## Claim boundary

Array hashes and saved manifests establish that the released metrics match the
frozen formal outputs. They do not replace independent end-to-end retraining of
third-party packages. The repository therefore claims one-command matrix-level
verification and metric reproduction, while full retraining requires the
recorded upstream repositories and CUDA environment.
