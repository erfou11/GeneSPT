# GeneSPT

Reviewer-facing source repository for:

**GeneSPT: Gene-Conditioned Decoding and Predictable Spatial Program Transfer
for Unmeasured Gene Expression Prediction in Spatial Transcriptomics**

GeneSPT evaluates unmeasured-gene prediction under strict whole-gene holdout.
The model combines:

1. scRNA-derived target-gene descriptors;
2. a shared gene-conditioned decoder (GeneSPT-GC);
3. Predictable Spatial Program Transfer (PSP), which predicts gene-specific
   coefficients for spatial programs estimated from training-gene ST maps;
4. a validation-selected model readout locked before final-test evaluation; and
5. a centralized evaluator for SPCC, RMSE, JS/JSD, and SSIM.

## Repository Scope

This checkout is anchored to the completed formal Protocol A workbench. Active
method and figure code is under `main/` and `scripts/`.
Superseded model routes, mistaken configurations, historical results, raw
data, prediction matrices, and checkpoints are intentionally not tracked.

Large reviewer-facing data and result artifacts are prepared as a separate
archive. A replacement Zenodo DOI will be added after the new record is
published; this repository does not currently claim an active archive DOI.

GitHub does not contain raw or processed expression matrices.

## Layout

```text
main/               GeneSPT and PSP implementation modules used by Protocol A
scripts/protocol_a_full/  Formal schedulers, evaluators, controls, and Figures 2-6
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

## Formal Experiment Entry Point

The retained formal GeneSPT/GeneSPT-GC scheduler is:

```bash
python scripts/protocol_a_full/run_protocol_a_genespt.py --run --resume
```

It uses the frozen dataset configuration and produces the matched GC-only and
full PSP outputs for all six datasets and five folds. Full commands and their
data requirements are documented in `docs/REPRODUCE_PROTOCOL_A.md`.

## Evaluation

The authoritative public complete-set evaluator is:

- `src/genespt/metrics.py`

The no-data metric smoke test is:

```bash
python tests/smoke_test.py
```

The complete formal benchmark has a separate fail-closed verifier and
centralized matrix-level recomputation:

```bash
python scripts/reproducibility/verify_protocol_a_release.py \
  --archive-root ../GeneSPT_reviewer_archive
python scripts/reproducibility/recompute_protocol_a_benchmark.py \
  --archive-root ../GeneSPT_reviewer_archive \
  --output-dir results/recomputed_protocol_a
python scripts/reproducibility/recompute_protocol_a_mechanism.py \
  --archive-root ../GeneSPT_reviewer_archive \
  --output-dir results/recomputed_figure3
```

The directory name above is an example extraction location adjacent to the
repository. Every reported required or missing path is relative to the
extracted archive root.

See `docs/metric_policy.md` for edge-case eligibility,
`docs/REPRODUCE_PROTOCOL_A.md` for the formal benchmark, and
`docs/AUDIT_REPRODUCTION.md` for the matrix-level audit. The complete active
entry-point map is in `docs/ACTIVE_CODE_MAP.md`. Public audits call the
centralized evaluator instead of copying formulas.

Saved prediction matrices must retain dataset ID, fold ID, method, spot IDs,
test-gene indices, and gene identifiers. See `docs/OUTPUT_SCHEMA.md`.

## Manuscript Outputs

- Figure 2: `scripts/protocol_a_full/generate_protocol_a_figure2.py`
- Figure 3 and Supplementary Table S3: `scripts/protocol_a_full/generate_protocol_a_figure3_s3.py`
- Figure 4: `scripts/protocol_a_full/generate_protocol_a_figure4.py`
- Figure 5 and Supplementary Table S2: `scripts/protocol_a_full/generate_protocol_a_figure5_s2.py`
- Figure 6: `scripts/protocol_a_full/generate_protocol_a_figure6.py`

## Reproduction Boundary

With the Zenodo archive, this repository supports **matrix-level
reproduction**: validating the archive layout and recomputing centralized
metrics from archived ground-truth, prediction, and frozen-split files. This
does not retrain a model or recreate the archived prediction matrices.

The repository does **not** claim one-command full-method retraining. Raw
downloads, training caches/checkpoints, and complete third-party baseline
repositories are not included. The exact six adapters, upstream revisions,
runtime patches or wrappers, parameters and 180 task records are included. Figure 3
matrix-level verification is supported by 90 identity-readout mechanism
matrices; retraining those controls additionally needs the canonical frozen GC
cache described above. See
`docs/REPRODUCTION_STATUS.md` and `docs/REPRODUCE_MANUSCRIPT.md` for the exact
boundary.

## External Baselines

The manuscript compares GeneSPT with Tangram, TransImp, SpaIM, SpaGE, stPlus,
and stAI using the same frozen whole-gene splits and centralized evaluator.
Third-party repositories are not vendored here. See
`docs/BASELINE_ADAPTATION.md` for the complete adaptation contract.

## Citation and License

See `CITATION.cff`. The repository is distributed under the Apache-2.0
license; data retain the terms of their original sources and the Zenodo record.
