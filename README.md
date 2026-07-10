# GeneSPT

Reviewer-facing source repository for:

**GeneSPT: Gene-Conditioned Spatial Program Transfer for Unmeasured Gene
Prediction in Spatial Transcriptomics**

GeneSPT evaluates unmeasured-gene prediction under strict whole-gene holdout.
The model combines:

1. scRNA-derived target-gene descriptors;
2. a shared gene-conditioned decoder (GeneSPT-GC);
3. Predictable Spatial Program Transfer (PSP), which predicts gene-specific
   coefficients for spatial programs estimated from training-gene ST maps;
4. pre-specified validation-selected fusion/readout rules frozen before test
   evaluation; and
5. a centralized evaluator for SPCC, RMSE, JS/JSD, and SSIM.

## Repository Scope

This checkout is anchored to the cleaned local manuscript workbench on
2026-07-10. Active method and figure code is under `main/` and `scripts/`.
Superseded model routes, mistaken configurations, historical results, raw
data, prediction matrices, and checkpoints are intentionally not tracked.

Large reviewer-facing data and result artifacts are archived at:

- Zenodo DOI: <https://doi.org/10.5281/zenodo.21223023>
- Zenodo record: <https://zenodo.org/records/21223023>

GitHub does not contain raw or processed expression matrices.

## Layout

```text
main/               Active GeneSPT, PSP, evaluator, and benchmark scripts
scripts/            Active manuscript figure and downstream-analysis scripts
src/genespt/        Lightweight public I/O and metric helpers
configs/            Dataset configuration templates
docs/               Reproduction, provenance, and output documentation
data/               Local data mount; large contents are ignored
frozen_inputs/      Local frozen caches and binary splits; ignored
splits/             Frozen split mount; binary files are ignored
results/            Local experiment outputs; generated contents are ignored
figures/            Local rendered figures; generated contents are ignored
tests/              Synthetic smoke and metric tests
baseline_adapters/  Notes for external baseline adapters
```

## Installation

The manuscript experiments use the CUDA Docker environment documented in
`ENVIRONMENT.md`.

```powershell
cd D:\TESTWORK001
docker compose up -d genespt
docker compose exec genespt bash -lc "cd /workspace/GeneSPT && python scripts/env_smoke_test.py"
```

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

## PSP Experiment Entry Point

The guarded five-fold PSP runner is:

```bash
python main/run_predictable_spatial_program_folds012.py \
  --folds 0 1 2 3 4 \
  --base-cache-dir /path/to/frozen/canonical_gc_cache \
  --psp-descriptor pca32_nmf32 \
  --save-prediction-matrices \
  --out-dir /path/to/output
```

The runner refuses implicit replacement-base training and non-canonical PSP
descriptors unless explicit override flags are supplied. The cache must use:

```text
foldN/gc_mlp_pca32_softplus_correct.npz
```

## Evaluation

Metric implementations used by the active workbench are in:

- `main/evaluate_spatial_pattern_metrics.py`
- `main/metrics_refstyle.py`
- `main/run_strict_gene_conditioned_decoder_gate.py`

Saved prediction matrices must retain dataset ID, fold ID, method, spot IDs,
test-gene indices, and gene identifiers. See `docs/OUTPUT_SCHEMA.md`.

## Manuscript Outputs

- Figure 2: `main/generate_figure2_primary_dotplot.py`
- Figure 4: `scripts/generate_figure4_hbc_representative_maps.py`
- Figure 5: `main/generate_figure5_cross_platform_per_gene_violins.py`
- Figure 6: `scripts/generate_main_downstream_figure6_final.py`

Figure 3 mechanism analyses require the frozen GC cache and strict split files
from the local/Zenodo provenance bundle. This repository does not claim a
one-command full manuscript reproduction while those large inputs remain
external.

## External Baselines

The manuscript compares GeneSPT with Tangram, TransImp, SpaIM, SpaGE, and
stPlus using the same frozen whole-gene splits and centralized evaluator.
Third-party repositories are not vendored here.

## Citation and License

See `CITATION.cff`. The repository is distributed under the Apache-2.0
license; data retain the terms of their original sources and the Zenodo record.
