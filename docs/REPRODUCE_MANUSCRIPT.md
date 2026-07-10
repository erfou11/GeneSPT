# Manuscript reproduction map

The public repository provides active source and lightweight checks. Large
inputs and frozen prediction matrices are supplied through Zenodo DOI
`10.5281/zenodo.21223023`.

| Manuscript item | Active source path | External inputs |
| --- | --- | --- |
| Figure 2 | `main/generate_figure2_primary_dotplot.py` | primary benchmark source table |
| Figure 3 | `main/run_gc_mlp_descriptor_controls_5fold.py`, `main/run_predictable_spatial_program_folds012.py` | frozen splits and canonical GC cache |
| Figure 4 | `scripts/generate_figure4_hbc_representative_maps.py` | HBC ground truth and saved predictions |
| Figure 5 | `main/generate_figure5_cross_platform_per_gene_violins.py` | cross-platform per-gene metrics |
| Figure 6 | `scripts/generate_main_downstream_figure6_final.py` | MHPR measured and predicted matrices plus labels |

The centralized evaluator recomputes metrics from original-scale saved
predictions. Visualization normalization is not used for quantitative metrics.

This repository does not claim one-command end-to-end retraining of every
external baseline. Baseline versions and archived predictions must be combined
with their official implementations.
