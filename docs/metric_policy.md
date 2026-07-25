# Centralized Metric Policy

The public evaluator accepts aligned `spots x genes` truth and prediction
matrices. The overall evaluation set is fixed by the truth: every gene with a
fully finite truth vector is eligible. A nonfinite prediction for one of these
genes is an evaluation failure and raises `ValueError`; it is never omitted by
`nanmedian` or another method-dependent filter.

## Metric Eligibility

| Metric | Truth-defined eligibility | Edge-case policy |
| --- | --- | --- |
| SPCC | Finite, nonconstant truth | Spearman rank correlation; a finite constant prediction scores exactly `0` |
| RMSE | Any finite truth | Population z-score RMSE; a constant vector has an all-zero z-score |
| SSIM | Any finite truth | Existing reference-style global formula after per-vector maximum scaling |
| JS/JSD | Finite truth with positive mass after nonnegative clipping | Natural logarithms; a zero-mass prediction scores `ln(2)` |

Truth with no positive mass after nonnegative clipping has method-independent
JSD `N/A`. The optional
`uniform_zero_jensen_shannon_divergence` function exposes a zero-to-uniform
sensitivity convention, but it is not included in the primary per-gene table
or summary.

The SSIM implementation intentionally retains the existing GeneSPT
reference-style global formula. It is a single global comparison over the
per-gene vectors, not a windowed image SSIM, and this repair does not change its
mathematical definition.

## Coverage Fields

`evaluate_prediction` returns `(per_gene, summary)`. The one-row summary has
median primary metrics plus these overall fields:

- `total`: all matrix columns.
- `eligible`: columns with fully finite truth.
- `scored`: eligible columns accepted by the evaluator.
- `constant_prediction`: eligible columns with a finite constant prediction.
- `coverage`: `scored / eligible` (`N/A` when no truth column is eligible).

Verbose `*_genes` aliases are also provided. Because structural eligibility
differs by metric, the summary additionally reports `<METRIC>_eligible`,
`<METRIC>_scored`, `<METRIC>_constant_prediction`, and `<METRIC>_coverage` for
SPCC, RMSE, SSIM, and JSD. `JS` and `JS/JSD` remain aliases of `JSD` for output
compatibility.

## Reproducible audit entry points

`scripts/reproducibility/recompute_protocol_a_benchmark.py` applies this
evaluator to all 210 formal benchmark matrices and verifies the fold metrics
against the frozen source table.

`scripts/reproducibility/recompute_protocol_a_mechanism.py` applies the same
policy to all 90 Figure 3 mechanism matrices. Neither entry point relaxes the
metric policy or filters genes separately by method.
