# GeneSPT

Reviewer-facing reference implementation for:

**GeneSPT: Gene-Conditioned Spatial Program Transfer for Unmeasured Gene Prediction in Spatial Transcriptomics**

This repository is rebuilt around the method described in the manuscript. It is
not a dump of historical exploratory branches.

## What This Repository Contains

- scRNA-derived gene descriptor construction;
- GeneSPT-GC, the shared gene-conditioned decoder;
- Predictable Spatial Program Transfer (PSP);
- validation-selected fusion/readout;
- centralized strict whole-gene evaluation for SPCC, RMSE, JS/JSD, and SSIM;
- documentation for the manuscript data archive.

Large datasets, processed matrices, checkpoints, prediction matrices, and
figure outputs are not tracked in GitHub. They are archived separately on
Zenodo.

## Method Summary

GeneSPT predicts held-out genes under strict whole-gene holdout. For each fold,
training genes are used for model fitting and spatial-program estimation,
validation genes are used for PSP component screening and fusion/readout
selection, and test genes are used only after all choices are fixed.

The full model has two complementary branches:

1. **GeneSPT-GC:** a gene-conditioned decoder that maps a spot representation
   and target-gene descriptor to expression.
2. **PSP:** a validation-screened transfer branch that extracts spatial
   programs from training-gene ST maps and predicts target-gene coefficients
   from scRNA-derived descriptors.

## Repository Layout

```text
src/genespt/        Paper-aligned reference implementation
scripts/            CLI entry points for GC, full GeneSPT, and evaluation
configs/            Example configuration files
docs/               Data, archive, baseline, and reproduction notes
data/               Local data mount point; ignored by Git
splits/             Local fixed split mount point; binary files ignored
results/            Local outputs; predictions/checkpoints ignored
figures/            Lightweight figure notes/source tables only
tests/              Synthetic smoke test
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

## Smoke Test

The smoke test uses synthetic matrices only:

```bash
python tests/smoke_test.py
```

## Data Layout

Place archived inputs locally as:

```text
data/processed/<dataset_id>/
  st_matrix.npy          # spots x genes
  scrna_matrix.npy       # cells x genes
  gene_names.txt
  spot_ids.txt

splits/<dataset_id>/
  fold0_train_gene_idx.npy
  fold0_val_gene_idx.npy
  fold0_test_gene_idx.npy
```

The manuscript datasets are:

- Primary: Vis9A, HBC, Cell2location mouse brain.
- Cross-platform: seqFISH+ cortex/SVZ, MHPR/MERFISH, MVC/STARmap.

## Run GeneSPT-GC

```bash
python scripts/run_genespt_gc.py --config configs/genespt_gc.example.yaml
```

## Run Full GeneSPT

```bash
python scripts/run_full_genespt.py --config configs/full_genespt.example.yaml
```

## Centralized Evaluation

```bash
python scripts/evaluate_predictions.py \
  --true data/processed/Vis9A_D7_spaim_effective4470/st_matrix.npy \
  --pred results/predictions/Vis9A_D7_spaim_effective4470/GeneSPT/fold0_pred.npy \
  --test-indices splits/Vis9A_D7_spaim_effective4470/fold0_test_gene_idx.npy \
  --out results/evaluation/Vis9A_fold0_GeneSPT_summary.csv
```

## Data Availability

The reviewer-facing data package is available on Zenodo:

- DOI: <https://doi.org/10.5281/zenodo.20630224>
- Public file: `GeneSPT_manuscript_data_20260610.zip`
- Zenodo MD5: `872c96ff8d5bd6ac565b1843f46145c8`
- SHA256: `E932BD33D04CE14D2AF4CEB33F7F0376B1867EE5CFA2F177E17D2C702993E701`

## Citation

The manuscript DOI will be added after preprint or journal release. Until then,
cite the GitHub repository and the Zenodo data archive.

## License

Apache-2.0. Repository owner should confirm the final license before public
release.
