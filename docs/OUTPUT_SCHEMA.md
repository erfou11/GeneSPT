# Output schema

Large prediction matrices are stored in the data archive, not GitHub.

## Formal benchmark prediction

Each dataset, method and fold has:

```text
prediction_matrices/<dataset_id>/<method>/fold<fold>/
  prediction.npz
  test_gene_idx.npy
  metadata.json
```

`prediction.npz` contains one float32 array named `prediction` with shape
`spots x final_test_genes`. Rows follow the original ST spot/cell order.
Columns follow `test_gene_idx.npy`, whose values index the dataset-wide
`ground_truth/<dataset_id>/gene_names.txt` axis. The metadata records dataset,
role, method, fold, result layer, shape, source-matrix SHA256, compact-matrix
SHA256, truth SHA256, coverage and the frozen fold metrics.

GeneSPT rows use `validation_selected_readout_genespt57`. External methods use
`raw_identity`. This field is mandatory and prevents output layers from being
silently mixed.

## Fold-specific truth

Protocol A normalization depends on the inner-training gene set, so truth is
also stored by fold:

```text
ground_truth_protocol_a/<dataset_id>/fold<fold>/
  truth.npz
  test_gene_idx.npy
  metadata.json
```

`truth.npz` contains `truth`, aligned exactly to the prediction matrix. Its
normalization is `log1p(CPM)` with the CPM denominator computed from inner-
training genes only and applied to all columns. A single dataset-wide truth
matrix must not be substituted for this fold-specific Protocol A truth.

## Public manifests

`PREDICTION_MATRIX_MANIFEST.csv` binds all 210 prediction files to their test
indices, truth matrices, source hashes and reported fold metrics.
`PROTOCOL_A_TRUTH_MATRIX_MANIFEST.csv` binds all 30 truth files. Paths are
archive-relative and SHA256 verification is fail-closed.

## Mechanism matrices

Figure 3 mechanism controls are stored separately under
`mechanism_ablation_prediction_matrices/` and indexed by
`MECHANISM_ABLATION_MATRIX_MANIFEST.csv`. They use identity readout and must not
be substituted with the main benchmark readout matrices.
