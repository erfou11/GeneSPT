# Reproducing Manuscript Analyses

This repository provides the paper-aligned implementation and the command
surface. The archived matrices, splits, prediction outputs, and source tables
are stored on Zenodo.

## Table 2

Primary strict whole-gene benchmark on Vis9A, HBC, and Cell2location mouse
brain. Use the fixed split files from the Zenodo package and evaluate all
methods with `scripts/evaluate_predictions.py`.

## Figure 2

Primary benchmark visualization from Table 2 source values.

## Figure 3

Mechanistic analysis:

- descriptor controls;
- GeneSPT-GC versus full GeneSPT;
- PSP controls for learned train-gene spatial programs and
  descriptor-conditioned coefficient prediction.

## Figure 4

HBC representative held-out gene maps. Full matrix outputs stay in Zenodo;
GitHub may contain only lightweight source tables or plotting code.

## Figure 5

Cross-platform distributions for seqFISH+ cortex/SVZ, MHPR/MERFISH, and
MVC/STARmap.

## Figure 6

Downstream analyses using augmented matrices constructed from measured
train/validation genes and predicted held-out genes.

## Integrity Rules

- Test genes must not be used for model fitting, PSP program estimation,
  validation screening, fusion/readout selection, or hyperparameter tuning.
- Validation genes may be used for model selection, PSP component screening,
  and fusion/readout selection only.
- Reported metrics should come from the centralized evaluator.

