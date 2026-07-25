# Verified method configuration

This document separates parameters verified from executable source and
archived matrices from logical path templates. It does not expand the public
claim beyond matrix-level reproduction.

## Expression and descriptors

ST and scRNA expression are library-size normalized to 10,000 counts per
spatial location or cell and transformed with `log1p`. Gene descriptors are
formed from the gene-by-cell scRNA matrix. PCA32 uses scikit-learn
`TruncatedSVD(n_components=32)`. NMF32 uses
`MiniBatchNMF(n_components=32, init="nndsvda", max_iter=250,
batch_size=512, beta_loss="frobenius")`. The canonical GC branch uses PCA32;
the matched PSP branch uses concatenated PCA32 and NMF32 descriptors.

## Gene-conditioned decoder

The spot encoder is `LayerNorm -> Linear(input,256) -> GELU -> Linear(256,128)
-> LayerNorm`. Spatial and gene inputs are projected to 96 dimensions. Their
concatenation is decoded by `Linear(192,192) -> GELU -> Linear(192,64) -> GELU
-> Linear(64,1)` followed by Softplus. The model minimizes pairwise mean
squared error with AdamW (`lr=2e-3`, `weight_decay=1e-4`), gradient clipping at
5, 800 steps, batches of 65,536 spot-gene pairs, and validation every 100
steps. Validation selects the checkpoint; test-gene ST is not used.

## Predictable Spatial Program Transfer

The matched mechanism analysis applies rank-64 `TruncatedSVD` to the raw
log1p-CPM train-gene ST matrix. A ridge model (`alpha=10`) predicts program
coefficients from PCA32+NMF32 descriptors. Components are ranked on validation
genes by coefficient Spearman correlation multiplied by
`log1p(oracle coefficient variance)`, and the top 32 are retained.

Training-gene Moran's I values on an 8-nearest-neighbour graph fit a separate
descriptor ridge model (`alpha=1`). Validation descriptor-predicted spatiality
defines tertile thresholds. Test genes are assigned to low, middle, or high
spatiality bins only from descriptor-predicted spatiality; test-gene ST is not
read. Validation searches 48 fixed `(lambda_low, lambda_mid, lambda_high)`
triples from the grids recorded in `configs/canonical_gc_psp.yaml`. A strict
score improvement is required, so exact ties retain the first fixed-traversal
candidate. The validation score combines whole-panel Pearson correlation,
high-spatiality Pearson correlation, RMSE, and JSD. The source field formerly
named `SPCC` in this fast selection trace is Pearson correlation, not the
Spearman SPCC reported by the centralized evaluator. The implemented score
also contains an SSIM guard, but both candidate and baseline SSIM are `NaN` in
the archived fast trace, so that term is inactive for all 240 candidates. The
fixed candidate grid and selection logic are retained in
`configs/canonical_gc_psp.yaml` and
`main/run_predictable_spatial_program_folds012.py`. The reviewer archive
preserves the resulting fold-specific prediction matrices and hashes; the
intermediate fast selection trace is outside the stated matrix-level
reproduction boundary.

Figure 3B/3C uses identity readout with no post-hoc calibration. The main
benchmark uses a separate model-specific validation-selected readout that is
not presented as a GeneSPT innovation. For every dataset and fold, GeneSPT and
GeneSPT-GC use the same fixed family of 57 candidates. Selection is completed
and hashed before final-test predictions or truth are opened. All 30 locks,
60 candidate tables and selected rows are published under
`protocol_a_reproducibility/readout_selections/` in the data archive. External
baselines remain at raw identity output.

## Reproduction boundary

The retained repository can validate archive layout and recompute centralized
metrics from saved matrices. Raw downloads, complete training caches,
checkpoints, and all third-party implementations are not included, so the
repository does not claim one-command end-to-end retraining.
