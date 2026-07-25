# Manuscript reproduction map

Large arrays are stored in the separate data archive. Every path below is
relative to the extracted archive root.

| Manuscript item | Active source | Archive evidence | Reproduction level |
| --- | --- | --- | --- |
| Supplementary Table S1 | dataset and split manifests | `results_source_data/supplementary/`; `processed_datasets/`; `frozen_splits/`; `label_provenance/` | Provenance, preprocessing, split and label audit |
| Supplementary Table S2 | `scripts/reproducibility/recompute_protocol_a_benchmark.py` | `prediction_matrices/`; `ground_truth_protocol_a/`; `results_source_data/supplementary/` | Full matrix-level metric and rank recomputation |
| Supplementary Table S3 | formal mechanism scripts under `scripts/protocol_a_full/` | `mechanism_ablation_prediction_matrices/`; `results_source_data/supplementary/`; `results_source_data/figures/figure3/` | Archived fold-level mechanism evidence |
| Figure 2 | `scripts/protocol_a_full/generate_protocol_a_figure2.py` | `results_source_data/figures/figure2/` | Seven-method source table and plotting provenance |
| Figure 3 | `scripts/protocol_a_full/generate_protocol_a_figure3_s3.py` | `results_source_data/figures/figure3/` | Source table plus mechanism matrices |
| Figure 4 | `scripts/protocol_a_full/generate_protocol_a_figure4.py` | `results_source_data/figures/figure4/`; HBC truth and prediction matrices | Four genes by eight columns, including stAI |
| Figure 5 | `scripts/protocol_a_full/generate_protocol_a_figure5_s2.py` | `results_source_data/figures/figure5/` | Seven-method per-gene source table and plotting provenance |
| Figure 6 | `scripts/protocol_a_full/generate_protocol_a_figure6.py` | `results_source_data/figures/figure6/`; MHPR labels and completed matrices | Panel source tables, hashes and downstream metrics |

The complete Protocol A verification and matrix-level recomputation commands
are in `docs/REPRODUCE_PROTOCOL_A.md`.

## Reproduction boundary

The archive and centralized evaluator reproduce reported metrics from frozen
final-test matrices. The repository also includes the exact formal GeneSPT
sources, baseline adapters, upstream patches, parameters and run manifests.
Full end-to-end retraining still requires external baseline repositories,
processed inputs, CUDA compute and training time; it is not presented as a
single lightweight command.
