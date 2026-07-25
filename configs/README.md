# Configuration boundary

The six dataset YAML files in this directory are logical data-path manifests.
They identify dataset IDs, aligned inputs, frozen split paths, and output
locations. They document the data contract rather than serving as standalone
training commands.

The verified, code-matched GC/PSP mechanism configuration is recorded in
`canonical_gc_psp.yaml` and explained in `docs/METHOD_CONFIGURATION.md`.
`protocol_a_baseline_versions.yaml` pins all six external methods, including
the official stAI commit and its exact formal parameters.
Dataset- and fold-specific validation-selected readout traces belong to the
Zenodo archive under `protocol_a_reproducibility/readout_selections/`.

`downstream_mhpr_completed_matrix_louvain.yaml` is the active Figure 6 Panel D
configuration. It constructs each fold-specific matrix from measured outer
train/validation genes and method-predicted outer test genes, then applies the
same HVG=100, PCA=30, k=15 and weighted Louvain settings to every method.

Use `docs/REPRODUCE_PROTOCOL_A.md` for training and matrix-recomputation
commands.
