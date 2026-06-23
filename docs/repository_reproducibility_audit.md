# Repository Reproducibility Audit

Date: 2026-06-23

## Executive Summary

The repository has been refactored from a simplified rebuilt implementation
into an anchored release candidate. Manuscript-facing method and reproduction
scripts are now copied from the final local GeneSPT source tree, and the public
metric wrapper calls the anchored final evaluator logic.

This fixes the prior high-risk mismatch where GitHub computed Pearson
correlation under the `SPCC` label.

The repository is improved but is not fully public-release ready because one
anchored final script chain still depends on `run_final_dataset_audit.py`, which
was not present in the local final source tree at refactor time.

## Overall Score

Score: 82 / 100

Decision: Review-ready with remaining blocker for full Table 2 one-command
reproduction.

## Mandatory Blockers

- Full Table 2 builder is source-anchored but not yet fully runnable because
  `main/run_final_multidataset_fold0_gate.py` imports missing
  `run_final_dataset_audit.py`.
- The Zenodo package-to-final-workbench path conversion still needs a small
  checked script if the archive layout differs from the `/workspace/GeneSPT`
  defaults.

## What Changed

- Added anchored final local scripts under `main/`.
- Added anchored final figure/table scripts under `scripts/`.
- Replaced `src/genespt/metrics.py` with a thin wrapper around
  `main/run_strict_gene_conditioned_decoder_gate.py::gene_metrics`.
- Removed the old hand-written release-skeleton GC/PSP implementation from
  `src/genespt/`.
- Removed old `scripts/run_genespt_gc.py` and `scripts/run_full_genespt.py`.
- Added `docs/SOURCE_ANCHOR.md`.
- Updated `README.md` and `docs/REPRODUCE_MANUSCRIPT.md` to point to anchored
  code paths.

## Detailed Score Table

| Category | Score | Notes |
|---|---:|---|
| Documentation and readability | 13 / 15 | README, data docs, reproduction docs, and source-anchor docs are present. |
| Environment and installation | 8 / 10 | Environment files exist; tests require dependencies not present in the bundled bare Python. |
| Data layout, Zenodo, splits, provenance | 12 / 15 | DOI and archive layout documented; conversion/check helper still needed. |
| Model training code completeness | 13 / 15 | GC and PSP final local scripts copied into `main/`; some defaults retain final-workbench paths. |
| Centralized evaluation and metrics | 15 / 15 | Public evaluator calls anchored final evaluator; SPCC uses Spearman. |
| Manuscript reproduction scripts/docs | 12 / 15 | Figure and supplement scripts anchored; Table 2 full builder has missing dependency. |
| Tests and smoke examples | 7 / 10 | Evaluator smoke tests exist; full model smoke is intentionally not faked. |
| Hygiene and terminology consistency | 2 / 5 | Public docs improved; anchored historical scripts still contain old internal labels where preserved exactly. |

## Anchored Files Found

Anchored final local `main/` scripts:

```text
main/run_strict_gene_conditioned_decoder_gate.py
main/run_gene_conditioned_mlp_controls_stabilization.py
main/run_gene_conditioned_decoder_folds012_validation.py
main/run_gc_spatiality_aware_training.py
main/run_gc_spatial_residual_basis_fold0.py
main/run_st_spatial_program_decoder_fold0.py
main/run_predictable_spatial_program_transfer_fold0.py
main/run_predictable_spatial_program_folds012.py
main/run_psp_spatiality_gated_readout.py
main/run_msr_structural_readout.py
main/run_final_multidataset_fold0_gate.py
main/recompute_final_benchmark_from_predictions.py
main/build_final_four_metric_available_benchmark.py
main/generate_figure2_primary_dotplot.py
main/generate_figure5_cross_platform_per_gene_violins.py
```

Anchored final local `scripts/` files:

```text
scripts/generate_figure4_hbc_representative_maps.py
scripts/generate_main_downstream_figure6_final.py
scripts/generate_supplyment_tables.py
```

## Commands to Use

Evaluator smoke:

```bash
python tests/smoke_test.py
```

Latest local bare-runtime result:

```text
ModuleNotFoundError: No module named 'scipy'
```

This was run with the bundled bare Python runtime, not the declared
`environment.yml` or Docker environment. Syntax compilation of the anchored
Python files passed.

Validation in the existing local development Docker image passed:

```text
image: topodist/cuda-dev:pytorch2.1.2-cu118
python tests/smoke_test.py -> smoke_test=ok
python scripts/env_smoke_test.py -> environment_smoke_test=ok
python scripts/evaluate_predictions.py --help -> ok
```

One saved prediction matrix:

```bash
python scripts/evaluate_predictions.py \
  --true data/processed/Vis9A_D7_spaim_effective4470/st_matrix.npy \
  --pred results/predictions/Vis9A_D7_spaim_effective4470/GeneSPT/fold0/prediction_test.npy \
  --test-indices splits/Vis9A_D7_spaim_effective4470/fold0_test_gene_idx.npy \
  --out results/evaluation/Vis9A_D7_spaim_effective4470/GeneSPT/fold0_summary.csv
```

Anchored GC validation:

```bash
python main/run_gene_conditioned_decoder_folds012_validation.py --folds 0,1,2
```

Anchored PSP validation:

```bash
python main/run_predictable_spatial_program_folds012.py --folds 0,1,2
```

## Missing Reproduction Paths

- Full Table 2 rebuild needs the missing `run_final_dataset_audit.py` or an
  equivalent restored dataset manifest.
- Full one-command conversion from Zenodo archive layout to the anchored
  final-workbench path layout is not yet implemented.

## Metric Findings

The public evaluator now resolves to:

```text
main/run_strict_gene_conditioned_decoder_gate.py::gene_metrics
```

This anchored function computes SPCC using:

```text
scipy.stats.spearmanr
```

No public `src/genespt` model or PSP implementation remains that can silently
reintroduce Pearson-labeled SPCC.

## Required Fixes Before Public Release

1. Recover or reconstruct from archived provenance the missing
   `run_final_dataset_audit.py` equivalent without inventing data.
2. Add a checked Zenodo extraction/conversion script.
3. Optionally add CI or a smaller CPU-only smoke environment so reviewers do
   not need to build the CUDA image just to check the evaluator.
4. Confirm final manuscript-facing terminology in public docs.
5. Commit and push the anchored refactor.

## Recommended Fixes After Release

1. Add a compact `docs/manuscript_artifact_map.md` mapping each figure/table to
   code and Zenodo source files.
2. Add CI that runs evaluator smoke tests.
3. Add hash manifest for copied anchored scripts in the release tag.
