# Configuration boundary

The six dataset YAML files in this directory are logical data-path manifests.
They identify dataset IDs, aligned inputs, frozen split paths, and output
locations, but they are not parsed by the retained historical experiment
scripts and must not be treated as executable training configurations.

The verified, code-matched GC/PSP mechanism configuration is recorded in
`canonical_gc_psp.yaml` and explained in `docs/METHOD_CONFIGURATION.md`.
`protocol_a_baseline_versions.yaml` pins all six external methods, including
the official stAI commit and its exact formal parameters.
Dataset- and fold-specific validation-selected readout traces belong to the
reviewer archive under `results_source_data/validation_selection/`.

`downstream_mhpr_completed_matrix_louvain.yaml` is the active Figure 6 Panel D
configuration. It constructs each fold-specific matrix from measured outer
train/validation genes and method-predicted outer test genes, then applies the
same HVG=100, PCA=30, k=15 and weighted Louvain settings to every method.

The public repository supports no-data smoke tests and matrix-level metric
reproduction. It does not claim one-command recreation of every archived
prediction matrix from these YAML files.
