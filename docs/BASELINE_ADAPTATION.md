# Baseline adaptation under strict whole-gene holdout

## Shared information boundary

All six external methods use the same five frozen outer folds. For each fold,
only training-gene ST expression is exposed to normalization and model fitting.
Validation- and final-test-gene ST expression is hidden from fitting. External
scRNA expression for all shared genes remains available as reference
information, consistent with the benchmark information boundary. The CPM
denominator is computed from training genes only and applied to every ST column
before `log1p`. The final-test ST truth is opened only by the centralized
evaluator after model outputs have been fixed.

Every adapter saves a complete finite matrix in the original spot order. A
missing gene, duplicate axis entry, non-finite value, hidden-gene overlap,
unexpected shape or input-hash mismatch is a hard failure. No truth-copy,
zero-fill, KMeans or checkpoint-reuse fallback contributed to the formal 180
baseline tasks.

## Method-specific adaptations

| Method | Upstream mode | Strict-holdout adaptation | Frozen parameters |
| --- | --- | --- | --- |
| Tangram 1.0.4 | Cluster mapping and gene projection | Mapping genes are restricted to the training index. Gene filtering is disabled so a frozen held-out axis cannot change by method. | clusters, RNA-count density prior, Leiden 0.5, 1000 epochs, lr 0.1, seed 42 |
| TransImp 0.2.0 | Low-rank translation | Translation signatures are fitted on training genes; validation and test genes are prediction targets only. The runtime patch changes a diagnostic label, not numerical operations. | lowrank/cell, 256 dimensions, 2000 epochs, lr 0.01, weight decay 0.01, spatial weight 0.1, k=8, seed 42 |
| SpaIM V1.0.0 | Style-transfer imputation | Native `min_cells`/`min_genes` filtering is disabled because it would otherwise make coverage method-dependent. Native metrics are skipped. Hidden-gene zero fill, truth copy and KMeans fallback are disabled. | 50 epochs, batch 500, style dim 1, lr 0.001, layers 256/512, Leiden 0.5, seed 42 |
| SpaGE | Principal-vector transfer | Principal vectors are fitted on training genes only; all validation/test genes are predicted. Constant-feature z-scores are mapped to zero to keep outputs finite. Cache reuse is disabled. | 30 principal vectors, seed 42 |
| stPlus | Autoencoder plus weighted neighbors | Training input contains training genes only; validation/test genes are prediction targets. Each fold uses a fresh checkpoint scope and deterministic data loading. | top-k request 2000, t-min 5, 50 neighbors, batch 512, max 10000 epochs, seed 42 |
| stAI | Supervised joint reference/spatial encoding and top-k imputation | A truth-free package contains outer-training ST genes, the allowed scRNA reference, spatial coordinates and author-provided scRNA cell labels. Validation- and test-gene ST expression is absent during `prepare` and `run`; only `evaluate` opens final-test ST truth. No pseudo-labels are used. | official commit `3376cc1`, 500 epochs, five internal models, top-k 50, spatial kNN 10, seed 8848 |

The exact upstream revisions and all parameters are machine-readable in
`configs/protocol_a_baseline_versions.yaml`. Runtime differences from upstream
are supplied as reviewable patches in `baseline_adapters/patches/`. stAI uses
the official source without an upstream patch; its three-stage wrapper is
`baseline_adapters/stai/run_stai_protocol_a.py`. Adapter file hashes, commands,
input hashes, output hashes, coverage and failure flags are recorded for every
dataset-fold task in the release manifests.

## Fairness boundary

External baselines retain their native model objectives and default model
families. The shared adaptation changes only the information boundary, axis
contract, deterministic execution and centralized reporting needed for the
same strict whole-gene test. GeneSPT's validation-selected readout is applied
only to GeneSPT and GeneSPT-GC; external methods are evaluated at raw identity
output. This distinction is explicit in every prediction-matrix manifest row.

## Reproduction level

Full retraining uses the recorded upstream revisions together with the
supplied adapters and patches. Reported benchmark metrics can be recomputed
from the archived matrices using the command in
`docs/REPRODUCE_PROTOCOL_A.md`.
