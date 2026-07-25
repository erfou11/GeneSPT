# GeneSPT

Official implementation and reproducibility materials for:

**GeneSPT: Gene-Conditioned Decoding and Predictable Spatial Program Transfer
for Unmeasured Gene Expression Prediction in Spatial Transcriptomics**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21550226.svg)](https://doi.org/10.5281/zenodo.21550226)

GeneSPT evaluates unmeasured-gene prediction under strict whole-gene holdout.
The model combines:

1. scRNA-derived target-gene descriptors;
2. a shared gene-conditioned decoder (GeneSPT-GC);
3. Predictable Spatial Program Transfer (PSP), which predicts gene-specific
   coefficients for spatial programs estimated from training-gene ST maps.

Model selection uses validation genes and is fixed before final-test
evaluation. SPCC, RMSE, JS/JSD, and SSIM are computed by one centralized
evaluator.

## Layout

```text
main/               GeneSPT and PSP implementation modules used by Protocol A
scripts/protocol_a_full/  Benchmark schedulers, evaluators, controls, and Figures 2-6
scripts/reproducibility/  Archive export, verification, and matrix recomputation
src/genespt/        Lightweight public I/O and metric helpers
configs/            Dataset configuration templates
docs/               Reproduction, provenance, and output documentation
data/               Local data mount; large contents are ignored
frozen_inputs/      Local frozen caches and binary splits; ignored
splits/             Frozen split mount; binary files are ignored
results/            Local experiment outputs; generated contents are ignored
figures/            Local rendered figures; generated contents are ignored
tests/              Synthetic smoke and metric tests
baseline_adapters/  Exact formal adapters and upstream runtime patches
manifests/           Public Protocol A run, split, matrix, and selection indices
```

## Installation and no-data smoke test

The repository includes its own `compose.yaml`; no parent-directory Compose
file or Dev Container configuration is required. From the public repository
root, build the image and run the environment smoke test with:

```bash
docker compose build genespt
docker compose run --rm genespt python scripts/env_smoke_test.py
```

The smoke test checks imports and reports CUDA visibility, but it does not
require a GPU or any biological data.

See `ENVIRONMENT.md` for the container contract.

For a standalone environment:

```bash
conda env create -f environment.yml
conda activate genespt
pip install -e .
```

## Dataset Set

Primary datasets:

- Vis9A
- HBC
- Cell2location mouse brain

Cross-platform datasets:

- seqFISH+ cortex/SVZ
- MHPR/MERFISH
- MVC/STARmap

Expected local paths are documented in `configs/` and `docs/DATA.md`.

## Strict Holdout Boundary

Training-gene ST expression may be used for model fitting and spatial-program
estimation. Validation-gene ST expression may be used for model and component
selection. Test-gene ST expression is reserved for final evaluation only.
scRNA-derived descriptors are external gene-identity inputs and do not contain
test-gene ST spatial expression.

## Benchmark Entry Point

The GeneSPT/GeneSPT-GC benchmark scheduler is:

```bash
python scripts/protocol_a_full/run_protocol_a_genespt.py --run --resume
```

It uses the frozen dataset configuration and produces the matched GC-only and
full PSP outputs for all six datasets and five folds. Full commands and their
data requirements are documented in `docs/REPRODUCE_PROTOCOL_A.md`.

## Evaluation

The centralized evaluator is:

- `src/genespt/metrics.py`

The no-data metric smoke test is:

```bash
python tests/smoke_test.py
```

The release includes archive verification and centralized matrix-level
recomputation:

```bash
python scripts/reproducibility/verify_protocol_a_release.py \
  --archive-root ../GeneSPT_archive
python scripts/reproducibility/recompute_protocol_a_benchmark.py \
  --archive-root ../GeneSPT_archive \
  --output-dir results/recomputed_protocol_a
python scripts/reproducibility/recompute_protocol_a_mechanism.py \
  --archive-root ../GeneSPT_archive \
  --output-dir results/recomputed_figure3
```

The directory name above is an example extraction location adjacent to the
repository. Every reported required or missing path is relative to the
extracted archive root.

See `docs/metric_policy.md` for metric eligibility,
`docs/REPRODUCE_PROTOCOL_A.md` for the benchmark workflow, and
`docs/AUDIT_REPRODUCTION.md` for matrix-level verification. The entry-point
map is in `docs/ACTIVE_CODE_MAP.md`.

Saved prediction matrices must retain dataset ID, fold ID, method, spot IDs,
test-gene indices, and gene identifiers. See `docs/OUTPUT_SCHEMA.md`.

## Manuscript Outputs

- Figure 2: `scripts/protocol_a_full/generate_protocol_a_figure2.py`
- Figure 3 and Supplementary Table S3: `scripts/protocol_a_full/generate_protocol_a_figure3_s3.py`
- Figure 4: `scripts/protocol_a_full/generate_protocol_a_figure4.py`
- Figure 5 and Supplementary Table S2: `scripts/protocol_a_full/generate_protocol_a_figure5_s2.py`
- Figure 6: `scripts/protocol_a_full/generate_protocol_a_figure6.py`

## Reproducibility

The repository and Zenodo archive support archive validation, centralized
metric recomputation, and regeneration of the reported figures and
supplementary tables from saved matrices. Training entry points, parameters,
six external-method adapters, upstream revisions, runtime patches, and task
manifests are provided. Full training additionally requires the referenced
upstream repositories, processed inputs, and CUDA environment. See
`docs/REPRODUCTION_STATUS.md` and `docs/REPRODUCE_MANUSCRIPT.md`.

## External Baselines

The manuscript compares GeneSPT with Tangram, TransImp, SpaIM, SpaGE, stPlus,
and stAI using the same frozen whole-gene splits and centralized evaluator.
Upstream versions and adaptations are documented in
`docs/BASELINE_ADAPTATION.md`.

## Citation and License

See `CITATION.cff`. The code is distributed under the Apache-2.0 license.
Data retain the terms of their original sources and the Zenodo record.
