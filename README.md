# GeneSPT

Reviewer-facing code repository for:

**GeneSPT: Gene-Conditioned Spatial Program Transfer for Unmeasured Gene Prediction in Spatial Transcriptomics**

This repository is now anchored to the final local GeneSPT manuscript code. The
method/reproduction scripts in `main/` and selected scripts in `scripts/` were
copied from the final local workbench rather than rewritten as a separate
simplified implementation.

See `docs/SOURCE_ANCHOR.md` for the source-anchor policy and known dependency
limits.

## What This Repository Contains

- anchored final local code for strict whole-gene GC, PSP, evaluator, and
  manuscript figure/source-table paths;
- a thin public evaluator wrapper under `src/genespt/` that calls the anchored
  final metric implementation;
- dataset path templates under `configs/`;
- documentation for the Zenodo data archive and manuscript reproduction paths.

Large datasets, processed matrices, checkpoints, prediction matrices, and
figure outputs are not tracked in GitHub. Processed data, source tables, and
final prediction matrices are archived on Zenodo in a single combined package,
`GeneSPT.zip`.

## Repository Layout

```text
main/               Anchored final local method and reproduction scripts
src/genespt/        Thin public evaluator/io wrappers only
scripts/            Public evaluator plus anchored final figure/table scripts
configs/            Dataset path templates
docs/               Data, archive, baseline, source-anchor, and reproduction notes
data/               Local data mount point; ignored by Git
splits/             Local fixed split mount point; binary files ignored
results/            Local outputs; predictions/checkpoints ignored
figures/            Lightweight figure notes/source tables only
tests/              Synthetic evaluator smoke tests
baseline_adapters/  Notes for external baseline adapters
```

## Installation

```bash
conda env create -f environment.yml
conda activate genespt
pip install -e .
```

Docker is also supported:

```bash
docker build -t genespt:paper .
docker run --gpus all --rm -it -v "$PWD:/workspace/GeneSPT" genespt:paper bash
```

The anchored final scripts use the original final-workbench convention
`/workspace/GeneSPT` in several defaults. Running in the Docker command above
keeps that convention intact.

## Smoke Test

The smoke test uses synthetic matrices and checks the anchored evaluator path:

```bash
python tests/smoke_test.py
```

## Main Performance Verification

For reviewer-facing checks, start with the main benchmark source values, not the
PSP ablation scripts. Download `GeneSPT.zip` from Zenodo, extract it, then
extract the inner archive `GeneSPT_manuscript_data_20260610.zip`. Mount that
extracted inner folder as `/data` inside Docker and run:

```bash
docker run --rm \
  -v "$PWD:/workspace/GeneSPT" \
  -v "/path/to/GeneSPT_manuscript_data_20260610:/data" \
  -w /workspace/GeneSPT \
  genespt:paper \
  python scripts/verify_main_performance_from_zenodo.py --zenodo-root /data
```

This command:

- reads the Zenodo primary benchmark source table;
- writes the legacy Figure 2 input expected by the anchored final script;
- reruns `main/generate_figure2_primary_dotplot.py`;
- verifies that the generated Figure 2 source CSV matches the archived final
  Figure 2 source CSV;
- reports the primary GeneSPT SPCC, SSIM, RMSE, and JS/JSD values.

Expected report:

```text
results/reproduction/main_performance_source_check/README.md
```

This is a source-value verification path. It does not retrain models. To
recompute metrics from saved prediction matrices, use the saved-prediction
verification command below.

## Saved Prediction-Matrix Verification

After extracting `GeneSPT.zip`, extract the inner archive
`GeneSPT_zenodo_addendum_prediction_matrices_20260626.zip`. Mount that
extracted inner folder as `/addendum` and run:

```bash
docker run --rm \
  -v "$PWD:/workspace/GeneSPT" \
  -v "/path/to/GeneSPT_zenodo_addendum_prediction_matrices_20260626:/addendum" \
  -w /workspace/GeneSPT \
  genespt:paper \
  python scripts/verify_prediction_addendum.py --addendum-root /addendum
```

For a quick check on one dataset-method pair:

```bash
docker run --rm \
  -v "$PWD:/workspace/GeneSPT" \
  -v "/path/to/GeneSPT_zenodo_addendum_prediction_matrices_20260626:/addendum" \
  -w /workspace/GeneSPT \
  genespt:paper \
  python scripts/verify_prediction_addendum.py \
    --addendum-root /addendum \
    --datasets MHPR_current_panel \
    --methods GeneSPT
```

This writes fold-level and dataset-method aggregate metrics under
`results/reproduction/prediction_matrix_addendum_check/`.

## Data Layout

For the public evaluator wrapper, place archived inputs locally as:

```text
data/processed/<dataset_id>/
  st_matrix.npy          # spots x genes
  scrna_matrix.npy       # cells x genes, if needed by the selected script
  coordinates.npy
  gene_names.txt

splits/<dataset_id>/
  fold0_train_gene_idx.npy
  fold0_val_gene_idx.npy
  fold0_test_gene_idx.npy
```

The anchored final scripts in `main/` may instead expect the final-workbench
text layout under `/workspace/GeneSPT/data/<dataset_id>/`:

```text
Spatial_count.txt
scRNA_count.txt
Locations.txt
```

## Manuscript Datasets

Primary datasets:

- Vis9A
- HBC
- Cell2location mouse brain

Cross-platform datasets:

- seqFISH+ cortex/SVZ
- MHPR/MERFISH
- MVC/STARmap

Dataset-specific path templates are available under `configs/`.

## Anchored Method Commands

These are method and ablation validation paths. They are not the first-line
main performance reproduction path.

GeneSPT-GC validation path from the final local code:

```bash
python main/run_gene_conditioned_decoder_folds012_validation.py --folds 0,1,2
```

PSP validation path from the final local code:

```bash
python main/run_predictable_spatial_program_folds012.py --folds 0 1 2
```

These scripts preserve final local defaults. Use explicit `--counts-path`,
`--scrna-counts-path`, `--locations-path`, `--mask-dir`, and `--out-dir`
arguments if your extracted Zenodo files are not mounted at `/workspace/GeneSPT`.

## Centralized Evaluation

`scripts/evaluate_predictions.py` is a lightweight public entry point. The
metric logic comes from:

```text
main/run_strict_gene_conditioned_decoder_gate.py::gene_metrics
```

SPCC is Spearman correlation across spatial locations. SSIM is not multiplied
by 10.

```bash
python scripts/evaluate_predictions.py \
  --true data/processed/Vis9A_D7_spaim_effective4470/st_matrix.npy \
  --pred results/predictions/Vis9A_D7_spaim_effective4470/GeneSPT/fold0/prediction_test.npy \
  --test-indices splits/Vis9A_D7_spaim_effective4470/fold0_test_gene_idx.npy \
  --out results/evaluation/Vis9A_fold0_GeneSPT_summary.csv
```

The evaluator command requires a saved prediction matrix. Use the prediction
matrix inner archive from `GeneSPT.zip` for the archived final benchmark
matrices and evaluator-ready `log1p(CPM)` ground-truth arrays.

## Manuscript Reproduction

Start with:

```text
docs/REPRODUCE_MANUSCRIPT.md
docs/SOURCE_ANCHOR.md
docs/DATA.md
docs/REPRODUCTION_STATUS.md
docs/TERMINOLOGY.md
```

Some final benchmark/table scripts are preserved as source-anchored artifacts
but still require a missing local dependency, `run_final_dataset_audit.py`, or
the equivalent dataset manifest from the archived package. This is documented
in `docs/SOURCE_ANCHOR.md`; no replacement dependency has been invented.

## Data Availability

The reviewer-facing data package is a single combined Zenodo upload:

- File: `GeneSPT.zip`
- Zenodo DOI/record: pending final publication; replace this line after
  publication.
- MD5: `bab51420a0a961dc6a9d85f1f980b393`
- SHA256: `0ae1a6384aca693c173b61a315c6426174106a863d07ae5523b8cc2ac7fa0351`

`GeneSPT.zip` contains:

- `README_combined_upload.txt`
- `GeneSPT_manuscript_data_20260610.zip`
- `GeneSPT_zenodo_addendum_prediction_matrices_20260626.zip`

The original data-only Zenodo record is retained for provenance:

- DOI: <https://doi.org/10.5281/zenodo.20630224>
- File: `GeneSPT_manuscript_data_20260610.zip`
- Zenodo MD5: `872c96ff8d5bd6ac565b1843f46145c8`
- SHA256: `E932BD33D04CE14D2AF4CEB33F7F0376B1867EE5CFA2F177E17D2C702993E701`

## Citation

The manuscript DOI will be added after preprint or journal release. Until then,
cite the GitHub repository and the Zenodo data archive.

## License

Apache-2.0. Repository owner should confirm the final license before public
release.
