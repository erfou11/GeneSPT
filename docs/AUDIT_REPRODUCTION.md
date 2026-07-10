# Reviewer metric and PSP audits

The public audit path operates on evaluator-ready matrices. It does not train a
model, require a private workbench, or modify a Zenodo extraction. Prediction
matrices and ground truth remain external to GitHub.

## No-data checks

Both public entry points include deterministic synthetic checks:

```bash
python scripts/audit_complete_set_metrics.py --self-test
python scripts/compare_cell2location_strict_psp.py --self-test
```

The first exercises the complete-set edge cases. The second creates a tiny
two-fold bundle under the ignored `results/` directory, validates every strict
PSP invariant, recomputes metrics, and removes the temporary bundle.

## Complete-set matrix audit

Use `scripts/audit_complete_set_metrics.py` for one truth matrix and any number
of aligned predictions:

```bash
python scripts/audit_complete_set_metrics.py \
  --truth /path/to/truth.npy \
  --gene-names /path/to/gene_names.txt \
  --test-indices /path/to/test_gene_idx.npy \
  --prediction method-a /path/to/method_a_prediction.npz \
  --prediction method-b /path/to/method_b_prediction.npy \
  --out-dir results/complete_set_audit
```

NPZ predictions use the `prediction` key by default. Inputs are `spots x
genes`; `--test-indices` subsets a full truth matrix into the common prediction
column order. The script writes per-gene metrics, method summaries with
coverage, and SHA256 input provenance. All metric calculations come from
`src/genespt/metrics.py`; the CLI does not carry a second evaluator.

The authoritative eligibility and edge-case rules are in
`docs/metric_policy.md`. In particular, a prediction cannot alter the eligible
truth set or disappear through a method-specific finite-value filter.

## Cell2location strict PSP comparison

After extracting Zenodo record `10.5281/zenodo.21223023`, set `ZENODO` to the
root of `GeneSPT_zenodo_unified_20260706` and run:

```bash
DATASET=Cell2location_mouse_brain_ST8059048_shared12819

python scripts/compare_cell2location_strict_psp.py \
  --truth "$ZENODO/ground_truth/$DATASET/st_log1p_cpm.npy" \
  --gene-names "$ZENODO/ground_truth/$DATASET/gene_names.txt" \
  --prediction-root "$ZENODO/mechanism_ablation_prediction_matrices/$DATASET" \
  --split-dir "$ZENODO/frozen_splits/primary/$DATASET/gene_index_masks" \
  --reference-fold-metrics "$ZENODO/results_source_data/cell2location_psp_ablation_strict/cell2location_strict_psp_toggle_fold_metrics.csv" \
  --write-gene-level \
  --out-dir results/cell2location_strict_psp_recomputed
```

The comparison fails unless every fold satisfies all of these conditions:

- `GeneSPT-GC` is the internal `gc_mlp_base` output.
- `GeneSPT-GC+PSP` is the internal
  `predictable_spatial_program_selected_correct` output.
- The GC prediction, the GC archive's `base_prediction`, and the PSP archive's
  `base_prediction` are elementwise identical. This is the archived check that
  both settings use the same frozen GC cache output.
- Train, validation, and test indices are identical between settings and match
  the published frozen split arrays.
- Both settings use an identity readout and `posthoc_calibration=none`.
- Predictions are finite, test-gene names are aligned, and truth-defined metric
  eligibility is identical between settings.
- Recomputed fold metrics match both the per-archive metadata and, when passed,
  the archived reference fold table within `--atol`. The default is `1e-8` to
  tolerate low-order floating differences across supported NumPy/SciPy builds;
  split arrays, method identities, shapes, and coverage counts remain exact.

The expected five-fold means are:

| Setting | SPCC | RMSE | JSD | SSIM |
| --- | ---: | ---: | ---: | ---: |
| GeneSPT-GC | 0.162491 | 1.332672 | 0.355543 | 0.026551 |
| GeneSPT-GC+PSP | 0.182097 | 1.316650 | 0.353203 | 0.031518 |

Fold values are medians over each metric's truth-eligible genes. The reported
five-fold value is the arithmetic mean of the five fold medians. Improvement
is PSP minus GC for SPCC and SSIM, and GC minus PSP for RMSE and JSD.

Outputs include fold metrics and coverage, fold improvements, five-fold
summary, paired descriptive tests, strict validation checks, and a SHA256 run
manifest. Per-gene output is optional because it is generated evidence rather
than source code.

## Label boundary

`GeneSPT-GC` means PSP-off only. `GeneSPT-GC+PSP` means the matched PSP-on
setting. A historical or legacy result that contains PSP must remain labeled as
GC+PSP (or with its original PSP-on label); it must never be relabeled,
pooled, or cited as GC-only. Similar names in old tables do not override the
saved internal model and prediction provenance.

## Source selection

These lightweight scripts retain the reusable parts of the local
`main/recompute_complete_set_metrics.py`, the frozen archive contract in
`main/run_predictable_spatial_program_folds012.py`, and the strict checks in
`main/update_cell2location_strict_psp_submission_assets.py`. Local hard-coded
paths, manuscript mutation, archive synchronization, and duplicate metric
implementations were intentionally excluded.
