"""Centralized, standalone metrics for aligned spatial expression matrices.

All public matrix functions expect ``spots x genes`` inputs. Metric eligibility
is determined from the truth only, so changing a method's prediction cannot
change the evaluation gene set or hide a failed prediction.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


CONSTANT_ATOL = 1e-12
JSD_MAX_NATURAL_LOG = math.log(2.0)


def _as_float_vector(values: Any, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {vector.shape}")
    if vector.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return vector


def _validate_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true = _as_float_vector(y_true, "y_true")
    pred = _as_float_vector(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {true.shape} and {pred.shape}"
        )
    return true, pred


def _validate_matrices(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim != 2 or pred.ndim != 2:
        raise ValueError(
            "y_true and y_pred must be two-dimensional; "
            f"got {true.shape} and {pred.shape}"
        )
    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shape; "
            f"got {true.shape} and {pred.shape}"
        )
    if true.shape[0] == 0 or true.shape[1] == 0:
        raise ValueError("Metric matrices must contain at least one spot and one gene")
    return true, pred


def _is_constant(values: np.ndarray, *, atol: float = CONSTANT_ATOL) -> bool:
    """Return whether a finite vector is constant at a scale-aware tolerance."""

    if not np.isfinite(values).all():
        return False
    maximum = float(np.max(values))
    minimum = float(np.min(values))
    scale = max(1.0, abs(maximum), abs(minimum))
    return bool(maximum / scale - minimum / scale <= float(atol))


def _zscore_finite(values: np.ndarray, *, atol: float) -> np.ndarray:
    if _is_constant(values, atol=atol):
        return np.zeros_like(values, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(values))))
    scaled = values / scale
    centered = scaled - float(np.mean(scaled))
    standard_deviation = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise FloatingPointError(
            "Finite nonconstant vector produced an invalid standard deviation"
        )
    return centered / standard_deviation


def _spcc(true: np.ndarray, pred: np.ndarray, *, constant_atol: float) -> float:
    if _is_constant(pred, atol=constant_atol):
        return 0.0

    true_rank = rankdata(true, method="average")
    pred_rank = rankdata(pred, method="average")
    true_centered = true_rank - float(np.mean(true_rank))
    pred_centered = pred_rank - float(np.mean(pred_rank))
    denominator = float(
        np.sqrt(
            np.dot(true_centered, true_centered) * np.dot(pred_centered, pred_centered)
        )
    )
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise FloatingPointError(
            "Nonconstant finite ranks produced an invalid denominator"
        )
    value = float(np.dot(true_centered, pred_centered) / denominator)
    if not np.isfinite(value):
        raise FloatingPointError("SPCC was nonfinite for finite nonconstant inputs")
    return float(np.clip(value, -1.0, 1.0))


def _rmse(true: np.ndarray, pred: np.ndarray, *, constant_atol: float) -> float:
    true_z = _zscore_finite(true, atol=constant_atol)
    pred_z = _zscore_finite(pred, atol=constant_atol)
    value = float(np.sqrt(np.mean((true_z - pred_z) ** 2)))
    if not np.isfinite(value):
        raise FloatingPointError("RMSE was nonfinite for finite inputs")
    return value


def _reference_ssim_scale(values: np.ndarray, *, atol: float) -> np.ndarray:
    maximum_absolute_value = float(np.max(np.abs(values)))
    if maximum_absolute_value == 0.0:
        return np.zeros_like(values, dtype=np.float64)
    positive_maximum = float(np.max(values))
    if positive_maximum > float(atol) * maximum_absolute_value:
        denominator = positive_maximum
    else:
        denominator = maximum_absolute_value
    return values / denominator


def _reference_global_ssim(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    constant_atol: float,
) -> float:
    """Existing reference-style global SSIM, including its scaling convention."""

    first = _reference_ssim_scale(true, atol=constant_atol)
    second = _reference_ssim_scale(pred, atol=constant_atol)
    data_range = max(
        float(np.max(first)),
        float(np.max(second)),
        float(constant_atol),
    )

    mean_first = float(np.mean(first))
    mean_second = float(np.mean(second))
    centered_first = first - mean_first
    centered_second = second - mean_second
    sigma_first = float(np.sqrt(np.mean(centered_first * centered_first)))
    sigma_second = float(np.sqrt(np.mean(centered_second * centered_second)))
    covariance = float(np.mean(centered_first * centered_second))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    c3 = c2 / 2.0

    luminance = (2.0 * mean_first * mean_second + c1) / (
        mean_first * mean_first + mean_second * mean_second + c1
    )
    contrast = (2.0 * sigma_first * sigma_second + c2) / (
        sigma_first * sigma_first + sigma_second * sigma_second + c2
    )
    structure = (covariance + c3) / (sigma_first * sigma_second + c3)
    value = float(luminance * contrast * structure)
    if not np.isfinite(value):
        raise FloatingPointError("SSIM was nonfinite for finite inputs")
    return value


def _normalize_nonnegative_probability(
    values: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    if not np.isfinite(values).all():
        raise ValueError("Probability normalization requires finite input")
    nonnegative = np.maximum(values, 0.0)
    maximum = float(np.max(nonnegative))
    if maximum == 0.0:
        return None, True
    scaled = nonnegative / maximum
    total = float(np.sum(scaled, dtype=np.float64))
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError(
            "Positive finite vector produced invalid probability mass"
        )
    return scaled / total, False


def _jsd_from_probabilities(
    true_probability: np.ndarray,
    pred_probability: np.ndarray,
) -> float:
    midpoint = 0.5 * (true_probability + pred_probability)
    true_mask = true_probability > 0.0
    pred_mask = pred_probability > 0.0
    value = 0.5 * float(
        np.sum(
            true_probability[true_mask]
            * np.log(true_probability[true_mask] / midpoint[true_mask])
        )
    )
    value += 0.5 * float(
        np.sum(
            pred_probability[pred_mask]
            * np.log(pred_probability[pred_mask] / midpoint[pred_mask])
        )
    )
    if not np.isfinite(value):
        raise FloatingPointError("JSD was nonfinite for finite probability inputs")
    return float(np.clip(value, 0.0, JSD_MAX_NATURAL_LOG))


def jensen_shannon_divergence(y_true: Any, y_pred: Any) -> float:
    """Return natural-log JSD under the main evaluation policy.

    Values are clipped to nonnegative mass before normalization. Truth with no
    positive mass is method-independent N/A and returns ``NaN``. Positive-mass
    truth paired with a zero-mass prediction returns exactly ``ln(2)``.
    """

    true, pred = _validate_pair(y_true, y_pred)
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("JSD requires finite y_true and y_pred")
    true_probability, true_zero_mass = _normalize_nonnegative_probability(true)
    pred_probability, pred_zero_mass = _normalize_nonnegative_probability(pred)
    if true_zero_mass:
        return float("nan")
    if pred_zero_mass:
        return JSD_MAX_NATURAL_LOG
    assert true_probability is not None and pred_probability is not None
    return _jsd_from_probabilities(true_probability, pred_probability)


def uniform_zero_jensen_shannon_divergence(y_true: Any, y_pred: Any) -> float:
    """Return the optional zero-to-uniform JSD sensitivity analysis.

    This convention is deliberately separate from the primary JSD and is never
    included in :func:`evaluate_prediction` summaries.
    """

    true, pred = _validate_pair(y_true, y_pred)
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("JSD sensitivity analysis requires finite y_true and y_pred")
    true_probability, true_zero_mass = _normalize_nonnegative_probability(true)
    pred_probability, pred_zero_mass = _normalize_nonnegative_probability(pred)
    uniform = np.full(true.size, 1.0 / float(true.size), dtype=np.float64)
    if true_zero_mass:
        true_probability = uniform
    if pred_zero_mass:
        pred_probability = uniform
    assert true_probability is not None and pred_probability is not None
    return _jsd_from_probabilities(true_probability, pred_probability)


def _finite_mean_and_std(values: np.ndarray) -> tuple[float, float]:
    if not np.isfinite(values).all():
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def gene_metrics(
    y_true: Any,
    y_pred: Any,
    gene_names: Sequence[Any] | None = None,
    *,
    constant_atol: float = CONSTANT_ATOL,
) -> pd.DataFrame:
    """Compute per-gene metrics for aligned ``spots x genes`` matrices.

    A nonfinite prediction for any finite-truth gene raises ``ValueError``.
    Structural N/A values are retained only where the truth makes a metric
    undefined: constant truth for SPCC and zero-mass truth for JSD.
    """

    if constant_atol <= 0.0 or not np.isfinite(constant_atol):
        raise ValueError("constant_atol must be finite and greater than zero")
    true, pred = _validate_matrices(y_true, y_pred)
    if gene_names is None:
        names = [f"gene_{position}" for position in range(true.shape[1])]
    else:
        names = list(gene_names)
        if len(names) != true.shape[1]:
            raise ValueError(
                f"gene_names length {len(names)} does not match "
                f"{true.shape[1]} matrix columns"
            )

    rows: list[dict[str, Any]] = []
    for position, raw_gene_name in enumerate(names):
        gene_name = str(raw_gene_name)
        truth = true[:, position]
        prediction = pred[:, position]
        truth_finite = bool(np.isfinite(truth).all())
        prediction_finite = bool(np.isfinite(prediction).all())
        if truth_finite and not prediction_finite:
            raise ValueError(
                "Nonfinite prediction for finite-truth gene "
                f"{gene_name!r} at column {position}"
            )

        truth_constant = bool(truth_finite and _is_constant(truth, atol=constant_atol))
        prediction_constant = bool(
            prediction_finite and _is_constant(prediction, atol=constant_atol)
        )
        prediction_all_zero = bool(
            prediction_finite and np.count_nonzero(prediction) == 0
        )

        truth_zero_mass = False
        if truth_finite:
            _, truth_zero_mass = _normalize_nonnegative_probability(truth)
        prediction_zero_mass = False
        if prediction_finite:
            _, prediction_zero_mass = _normalize_nonnegative_probability(prediction)

        spcc_eligible = bool(truth_finite and not truth_constant)
        rmse_eligible = truth_finite
        ssim_eligible = truth_finite
        jsd_eligible = bool(truth_finite and not truth_zero_mass)

        spcc = float("nan")
        rmse = float("nan")
        ssim = float("nan")
        jsd = float("nan")
        if truth_finite:
            rmse = _rmse(truth, prediction, constant_atol=constant_atol)
            ssim = _reference_global_ssim(
                truth,
                prediction,
                constant_atol=constant_atol,
            )
            if spcc_eligible:
                spcc = _spcc(truth, prediction, constant_atol=constant_atol)
            if jsd_eligible:
                jsd = jensen_shannon_divergence(truth, prediction)

        expected_values = [rmse, ssim]
        if spcc_eligible:
            expected_values.append(spcc)
        if jsd_eligible:
            expected_values.append(jsd)
        if truth_finite and not np.isfinite(expected_values).all():
            raise FloatingPointError(
                f"Finite eligible inputs produced nonfinite metrics for gene {gene_name!r}"
            )

        true_mean, true_std = _finite_mean_and_std(truth)
        pred_mean, pred_std = _finite_mean_and_std(prediction)
        rows.append(
            {
                "gene_idx": int(position),
                "gene": gene_name,
                "subgroup": "all",
                "truth_finite": truth_finite,
                "truth_constant": truth_constant,
                "truth_zero_mass_after_nonnegative_clipping": bool(truth_zero_mass),
                "eligible_truth": truth_finite,
                "prediction_finite": prediction_finite,
                "prediction_constant": prediction_constant,
                "prediction_all_zero": prediction_all_zero,
                "prediction_zero_mass_after_nonnegative_clipping": bool(
                    prediction_zero_mass
                ),
                "scored": bool(truth_finite and prediction_finite),
                "exclusion_reason": "" if truth_finite else "truth_nonfinite",
                "SPCC_eligible": spcc_eligible,
                "RMSE_eligible": rmse_eligible,
                "SSIM_eligible": ssim_eligible,
                "JSD_eligible": jsd_eligible,
                "SPCC": spcc,
                "SSIM": ssim,
                "RMSE": rmse,
                "JS": jsd,
                "JSD": jsd,
                "JS/JSD": jsd,
                "true_mean": true_mean,
                "true_std": true_std,
                "pred_mean": pred_mean,
                "pred_std": pred_std,
            }
        )
    return pd.DataFrame(rows)


def _metric_summary(
    per_gene: pd.DataFrame,
    metric: str,
) -> tuple[float, int, int, int, float]:
    eligible = per_gene[f"{metric}_eligible"].to_numpy(dtype=bool)
    values = per_gene[metric].to_numpy(dtype=np.float64)
    scored = eligible & np.isfinite(values)
    eligible_count = int(eligible.sum())
    scored_count = int(scored.sum())
    if scored_count != eligible_count:
        raise FloatingPointError(
            f"{metric} scored {scored_count} of {eligible_count} truth-eligible genes"
        )
    median = float(np.median(values[scored])) if scored_count else float("nan")
    constant_prediction_count = int(
        (eligible & per_gene["prediction_constant"].to_numpy(dtype=bool)).sum()
    )
    coverage = float(scored_count / eligible_count) if eligible_count else float("nan")
    return (
        median,
        eligible_count,
        scored_count,
        constant_prediction_count,
        coverage,
    )


def evaluate_prediction(
    y_true: Any,
    y_pred: Any,
    gene_names: Sequence[Any] | None = None,
    *,
    constant_atol: float = CONSTANT_ATOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-gene metrics and a one-row median/coverage summary."""

    per_gene = gene_metrics(
        y_true,
        y_pred,
        gene_names,
        constant_atol=constant_atol,
    )
    summary: dict[str, float | int] = {}
    metric_counts: dict[str, tuple[int, int, int, float]] = {}
    for metric in ("SPCC", "RMSE", "SSIM", "JSD"):
        median, eligible, scored, constant_prediction, coverage = _metric_summary(
            per_gene,
            metric,
        )
        summary[metric] = median
        metric_counts[metric] = (
            eligible,
            scored,
            constant_prediction,
            coverage,
        )

    summary["JS"] = summary["JSD"]
    summary["JS/JSD"] = summary["JSD"]
    eligible_mask = per_gene["eligible_truth"].to_numpy(dtype=bool)
    scored_mask = eligible_mask & per_gene["scored"].to_numpy(dtype=bool)
    total = int(len(per_gene))
    eligible = int(eligible_mask.sum())
    scored = int(scored_mask.sum())
    constant_prediction = int(
        (eligible_mask & per_gene["prediction_constant"].to_numpy(dtype=bool)).sum()
    )
    coverage = float(scored / eligible) if eligible else float("nan")
    summary.update(
        {
            "total": total,
            "eligible": eligible,
            "scored": scored,
            "constant_prediction": constant_prediction,
            "coverage": coverage,
            "total_genes": total,
            "eligible_genes": eligible,
            "scored_genes": scored,
            "constant_prediction_genes": constant_prediction,
        }
    )
    for metric, counts in metric_counts.items():
        metric_eligible, metric_scored, metric_constant, metric_coverage = counts
        summary[f"{metric}_eligible"] = metric_eligible
        summary[f"{metric}_scored"] = metric_scored
        summary[f"{metric}_constant_prediction"] = metric_constant
        summary[f"{metric}_coverage"] = metric_coverage
    return per_gene, pd.DataFrame([summary])
