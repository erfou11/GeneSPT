# Reproduce the formal Protocol A benchmark

## Recompute saved-matrix results

After extracting the data archive next to this repository, run:

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

The first command checks 210 compact prediction matrices, 30 fold-specific
truth matrices, 180 baseline task records, 30 input/split records and 60
GeneSPT/GeneSPT-GC readout-selection records. The second command recomputes
SPCC, standardized RMSE, JSD and SSIM from the saved final-test matrices and
checks every benchmark fold against the frozen formal source table. The third
command independently recomputes the 90 Figure 3A-C mechanism rows and checks
them against Supplementary Table S3.

Expected output files:

- `recomputed_fold_metrics.csv`
- `recomputed_five_fold_metrics.csv`
- `recomputed_benchmark_ranks.csv`
- `recompute_summary.json`

Use `--save-gene-level` only when the large per-gene table is needed.

## Formal training commands

The following commands are the benchmark training entry points. They require the
processed inputs, frozen splits, recorded third-party revisions and a CUDA
environment.

```bash
python scripts/protocol_a_full/prepare_protocol_a_inputs.py
python scripts/protocol_a_full/run_protocol_a_genespt.py --run --resume
python scripts/protocol_a_full/run_protocol_a_baselines.py --run --resume
python scripts/protocol_a_full/evaluate_protocol_a_raw_predictions.py
```

The GeneSPT command creates matched GeneSPT-GC and full GeneSPT outputs from
the same folds and descriptor cache. The baseline scheduler creates exactly
`6 datasets x 5 folds x 5 methods = 150` fail-closed tasks for Tangram,
TransImp, SpaIM, SpaGE and stPlus. stAI uses the separately retained
truth-isolated three-stage adapter:

```bash
python baseline_adapters/stai/run_stai_protocol_a.py prepare \
  --dataset Vis9A --fold 0 --output-dir results/stai/Vis9A/fold0
python baseline_adapters/stai/run_stai_protocol_a.py run \
  --dataset Vis9A --fold 0 --output-dir results/stai/Vis9A/fold0
python baseline_adapters/stai/run_stai_protocol_a.py evaluate \
  --dataset Vis9A --fold 0 --output-dir results/stai/Vis9A/fold0
```

Running these stages for all six datasets and five folds contributes the
remaining 30 external-baseline tasks. The official checkout and label-source
settings are documented in `baseline_adapters/stai/README.md`.

The model-specific readout is selected and applied in physically separated
phases:

```bash
python scripts/protocol_a_full/run_protocol_a_validation_readout.py --prepare --all
python scripts/protocol_a_full/run_protocol_a_validation_readout.py --select --all --resume
python scripts/protocol_a_full/run_protocol_a_validation_readout.py --apply --all
python scripts/protocol_a_full/run_protocol_a_validation_readout.py --evaluate --all
```

Each GeneSPT and GeneSPT-GC fold evaluates 57 predefined candidates using
training and validation genes. The selection lock records that neither
final-test predictions nor final-test truth were accessed before locking. The
external baselines remain at raw identity output.

## Evidence map

| Question | Evidence |
| --- | --- |
| Which genes were train/validation/test? | `manifests/protocol_a/INPUT_SPLIT_MANIFEST.csv` and archive `frozen_splits/` |
| How was each baseline invoked? | `manifests/protocol_a/FORMAL_BASELINE_RUN_MANIFEST.csv` and `manifests/protocol_a/STAI_FORMAL_ADOPTION_MANIFEST.json` |
| Which upstream code and parameters were used? | `configs/protocol_a_baseline_versions.yaml`, `baseline_adapters/patches/` and `baseline_adapters/stai/README.md` |
| Which output matrix produced each number? | `manifests/protocol_a/PREDICTION_MATRIX_MANIFEST.csv` |
| Which matrices support Figure 3 and S3? | `manifests/protocol_a/MECHANISM_ABLATION_MATRIX_MANIFEST.csv` |
| How was GeneSPT readout selected? | `manifests/protocol_a/READOUT_SELECTION_MANIFEST.csv` and archive `protocol_a_reproducibility/readout_selections/` |
| Can the metrics be independently recomputed? | `recompute_protocol_a_benchmark.py` and `recompute_protocol_a_mechanism.py` |

The verification and recomputation commands operate directly on the frozen
matrices. Full baseline training uses the pinned upstream implementations and
adapters listed above.
