# Source Anchor

This repository has been refactored so that manuscript-facing method and
reproduction code is anchored to the final local GeneSPT workbench rather than
rewritten as a separate clean-room implementation.

## Anchor Policy

- Core method and reproduction scripts under `main/` are byte-for-byte copies
  from the final local `GeneSPT/main` source tree as of 2026-06-23.
- Figure and supplementary-table scripts under `scripts/` are byte-for-byte
  copies from the final local `GeneSPT/scripts` source tree as of 2026-06-23.
- `src/genespt/metrics.py` is only a thin public wrapper. It calls
  `main/run_strict_gene_conditioned_decoder_gate.py::gene_metrics`, which is
  the final local evaluator logic.
- The older hand-written release-skeleton GC/PSP implementation was removed
  from `src/genespt/` to avoid divergence from the final local code.

## Anchored Main Scripts

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

## Anchored Script Copies

```text
scripts/generate_figure4_hbc_representative_maps.py
scripts/generate_main_downstream_figure6_final.py
scripts/generate_supplyment_tables.py
```

## Public Verification Wrappers

These files are reviewer-facing wrappers around archived source values or the
anchored evaluator. They are not alternative implementations of the GeneSPT
method:

```text
scripts/evaluate_predictions.py
scripts/env_smoke_test.py
scripts/verify_main_performance_from_zenodo.py
tests/smoke_test.py
tests/test_metrics.py
src/genespt/
```

`scripts/verify_main_performance_from_zenodo.py` verifies the Table 2/Figure 2
main performance source values from the Zenodo archive and reruns the anchored
Figure 2 source-generation path. It does not retrain models and does not
recompute all benchmark rows from unavailable prediction matrices.

## Known Unresolved Dependency

`main/run_final_multidataset_fold0_gate.py` imports
`run_final_dataset_audit.py`. That source file was not present in the local
final source tree at the time of this refactor. Therefore, scripts that import
`run_final_multidataset_fold0_gate.py` are preserved as source-anchored
artifacts, but should not be advertised as fully runnable until that missing
dependency is recovered or the final dataset manifest is restored from the
archived data package.

Because the anchored files are preserved exactly, some internal historical
labels may still appear inside source comments, fallback branches, or archived
source-table handling. Public-facing manuscript terminology should be governed
by `README.md`, `docs/REPRODUCE_MANUSCRIPT.md`, `docs/DATA.md`, and
`docs/BASELINES.md` until the upstream final source is cleaned and recopied.

## Metric Anchor

The public evaluator intentionally uses the final local metric implementation:

```text
main/run_strict_gene_conditioned_decoder_gate.py::gene_metrics
```

That implementation computes SPCC with `scipy.stats.spearmanr`, summarizes
per-gene metrics by median in the final local helper, and uses the final local
definitions for RMSE, JS, and SSIM.
