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

