# Output Schema

Prediction matrices are stored outside GitHub.

## Prediction Matrix

- Rows: spots or cells in the spatial dataset.
- Columns: held-out test genes unless explicitly stored as a full gene matrix.
- Values: predicted nonnegative expression on the evaluator's expected scale.

Recommended metadata:

| field | meaning |
| --- | --- |
| `dataset_id` | internal dataset identifier |
| `fold` | strict whole-gene fold |
| `method` | method label, e.g. `GeneSPT` or `GeneSPT-GC` |
| `test_gene_indices` | indices into the dataset gene list |
| `test_gene_names` | gene symbols or IDs |
| `spot_ids` | spatial spot/cell identifiers |
| `checksum` | SHA256 checksum of the prediction file |
| `readout` | quantitative readout applied to the saved prediction |
| `posthoc_calibration` | post-hoc calibration, or `none` |
| `base_prediction` | matched frozen GC prediction when the archive contains a PSP toggle |

## Strict PSP labels

For the Cell2location matched toggle, `GeneSPT-GC` is reserved for the PSP-off
`gc_mlp_base` prediction and `GeneSPT-GC+PSP` is reserved for the PSP-on
`predictable_spatial_program_selected_correct` prediction. Both use the same
GC base cache output, the same frozen split, identity readout, and no post-hoc
calibration.

A legacy prediction that includes PSP must not be relabeled as `GeneSPT-GC` or
otherwise treated as GC-only. The internal model field and saved prediction
provenance take precedence over an ambiguous historical display label.

