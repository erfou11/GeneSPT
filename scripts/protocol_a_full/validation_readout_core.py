#!/usr/bin/env python3
"""Frozen GeneSPT validation-selected readout algorithms.

This module contains only the model-specific 57-candidate family copied from
the Protocol Boundary pilot.  It has no dataset paths and no test-truth I/O.
The orchestration and integrity boundary live in
``run_protocol_a_validation_readout.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import NearestNeighbors


FEATURE_KINDS = ("pca32", "nmf32", "pca32_nmf32")
TARGET_MODES = ("affine", "mean_only", "scale_only")
MODEL_GRID = (
    ("ridge", 0.1),
    ("ridge", 1.0),
    ("ridge", 10.0),
    ("ridge", 100.0),
    ("elasticnet", 0.001),
    ("elasticnet", 0.01),
)
METRICS = ("SPCC", "RMSE", "JSD", "SSIM")
GUARD_TOLERANCES = {"SPCC": 0.0015, "RMSE": 0.0015, "JSD": 0.0015}
CONSTANT_ATOL = 1e-12
EPS = 1e-12
JSD_MAX = math.log(2.0)
EXPECTED_CANDIDATE_COUNT = 3 + len(FEATURE_KINDS) * len(TARGET_MODES) * len(MODEL_GRID)


class ReadoutError(RuntimeError):
    """Raised when a frozen readout invariant is violated."""


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        json_ready(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_metrics_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "genespt_protocol_a_validation_readout_metrics", path
    )
    if spec is None or spec.loader is None:
        raise ReadoutError(f"Cannot load centralized evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "evaluate_prediction", None)):
        raise ReadoutError(
            f"Centralized evaluator has no callable evaluate_prediction: {path}"
        )
    return module


def _column_is_constant(matrix: np.ndarray, atol: float = CONSTANT_ATOL) -> np.ndarray:
    maximum = np.max(matrix, axis=0)
    minimum = np.min(matrix, axis=0)
    scale = np.maximum.reduce([np.ones_like(maximum), np.abs(maximum), np.abs(minimum)])
    return maximum / scale - minimum / scale <= float(atol)


def _stable_zscore_columns(matrix: np.ndarray, constant: np.ndarray) -> np.ndarray:
    scale = np.maximum(1.0, np.max(np.abs(matrix), axis=0))
    scaled = matrix / scale[None, :]
    centered = scaled - np.mean(scaled, axis=0, keepdims=True)
    standard_deviation = np.sqrt(np.mean(centered * centered, axis=0))
    safe = np.where(standard_deviation > 0.0, standard_deviation, 1.0)
    result = centered / safe[None, :]
    result[:, constant] = 0.0
    return result


def _reference_scale_columns(matrix: np.ndarray, atol: float) -> np.ndarray:
    maximum_absolute = np.max(np.abs(matrix), axis=0)
    positive_maximum = np.max(matrix, axis=0)
    denominator = np.where(
        positive_maximum > float(atol) * maximum_absolute,
        positive_maximum,
        maximum_absolute,
    )
    safe = np.where(denominator != 0.0, denominator, 1.0)
    result = matrix / safe[None, :]
    result[:, maximum_absolute == 0.0] = 0.0
    return result


def _probability_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nonnegative = np.maximum(matrix, 0.0)
    maximum = np.max(nonnegative, axis=0)
    zero_mass = maximum == 0.0
    safe_maximum = np.where(zero_mass, 1.0, maximum)
    scaled = nonnegative / safe_maximum[None, :]
    total = np.sum(scaled, axis=0, dtype=np.float64)
    safe_total = np.where(total > 0.0, total, 1.0)
    probability = scaled / safe_total[None, :]
    probability[:, zero_mass] = 0.0
    return probability, zero_mass


def fast_metric_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    constant_atol: float = CONSTANT_ATOL,
) -> dict[str, float]:
    """Vectorized equivalent of centralized ``evaluate_prediction`` medians."""

    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim != 2 or pred.ndim != 2 or true.shape != pred.shape:
        raise ValueError(f"Aligned 2D matrices required; got {true.shape} and {pred.shape}")
    if true.size == 0:
        raise ValueError("Metric matrices cannot be empty")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Validation metrics require finite truth and prediction")

    truth_constant = _column_is_constant(true, constant_atol)
    prediction_constant = _column_is_constant(pred, constant_atol)
    true_rank = rankdata(true, axis=0, method="average")
    pred_rank = rankdata(pred, axis=0, method="average")
    true_centered = true_rank - np.mean(true_rank, axis=0, keepdims=True)
    pred_centered = pred_rank - np.mean(pred_rank, axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(true_centered * true_centered, axis=0)
        * np.sum(pred_centered * pred_centered, axis=0)
    )
    spcc = np.divide(
        np.sum(true_centered * pred_centered, axis=0),
        denominator,
        out=np.zeros(true.shape[1], dtype=np.float64),
        where=denominator > 0.0,
    )
    spcc[prediction_constant] = 0.0
    spcc = np.clip(spcc, -1.0, 1.0)

    true_z = _stable_zscore_columns(true, truth_constant)
    pred_z = _stable_zscore_columns(pred, prediction_constant)
    rmse = np.sqrt(np.mean((true_z - pred_z) ** 2, axis=0))

    first = _reference_scale_columns(true, constant_atol)
    second = _reference_scale_columns(pred, constant_atol)
    data_range = np.maximum.reduce(
        [
            np.max(first, axis=0),
            np.max(second, axis=0),
            np.full(true.shape[1], float(constant_atol)),
        ]
    )
    mean_first = np.mean(first, axis=0)
    mean_second = np.mean(second, axis=0)
    centered_first = first - mean_first[None, :]
    centered_second = second - mean_second[None, :]
    sigma_first = np.sqrt(np.mean(centered_first * centered_first, axis=0))
    sigma_second = np.sqrt(np.mean(centered_second * centered_second, axis=0))
    covariance = np.mean(centered_first * centered_second, axis=0)
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
    ssim = luminance * contrast * structure

    true_probability, truth_zero_mass = _probability_columns(true)
    pred_probability, prediction_zero_mass = _probability_columns(pred)
    midpoint = 0.5 * (true_probability + pred_probability)
    with np.errstate(divide="ignore", invalid="ignore"):
        true_terms = np.where(
            true_probability > 0.0,
            true_probability * np.log(true_probability / midpoint),
            0.0,
        )
        pred_terms = np.where(
            pred_probability > 0.0,
            pred_probability * np.log(pred_probability / midpoint),
            0.0,
        )
    jsd = 0.5 * np.sum(true_terms, axis=0) + 0.5 * np.sum(pred_terms, axis=0)
    jsd = np.clip(jsd, 0.0, JSD_MAX)
    jsd[prediction_zero_mass & ~truth_zero_mass] = JSD_MAX
    jsd[truth_zero_mass] = np.nan

    eligible_spcc = ~truth_constant
    eligible_jsd = ~truth_zero_mass
    summary = {
        "SPCC": float(np.median(spcc[eligible_spcc]))
        if np.any(eligible_spcc)
        else float("nan"),
        "RMSE": float(np.median(rmse)),
        "JSD": float(np.median(jsd[eligible_jsd]))
        if np.any(eligible_jsd)
        else float("nan"),
        "SSIM": float(np.median(ssim)),
    }
    summary["JS"] = summary["JSD"]
    return summary


def metric_summaries_close(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    atol: float = 2e-12,
) -> tuple[bool, dict[str, float]]:
    differences: dict[str, float] = {}
    matched = True
    for metric in METRICS:
        a = float(first[metric])
        b = float(second[metric])
        if math.isnan(a) and math.isnan(b):
            differences[metric] = 0.0
            continue
        difference = abs(a - b)
        differences[metric] = difference
        matched = matched and math.isfinite(difference) and difference <= atol
    return matched, differences


def make_knn_edges(coordinates: np.ndarray, k: int = 8) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise ReadoutError("Coordinates must contain at least two spots and two columns")
    if not np.isfinite(points).all():
        raise ReadoutError("Coordinates contain nonfinite values")
    neighbors = NearestNeighbors(
        n_neighbors=min(int(k) + 1, points.shape[0]), metric="euclidean"
    )
    neighbors.fit(points)
    _, indices = neighbors.kneighbors(points)
    edges: set[tuple[int, int]] = set()
    for first in range(points.shape[0]):
        for raw_second in indices[first, 1:]:
            second = int(raw_second)
            edges.add(tuple(sorted((int(first), second))))
    return np.asarray(sorted(edges), dtype=np.int64)


def moran_i(values: np.ndarray, edges: np.ndarray) -> float:
    vector = np.asarray(values, dtype=np.float64)
    centered = vector - float(np.mean(vector))
    denominator = float(np.sum(centered * centered))
    if denominator < EPS or edges.size == 0:
        return float("nan")
    weight_sum = 2.0 * float(edges.shape[0])
    numerator = float(2.0 * np.sum(centered[edges[:, 0]] * centered[edges[:, 1]]))
    return float((len(vector) / weight_sum) * (numerator / denominator))


def graph_smoothness(values: np.ndarray, edges: np.ndarray) -> float:
    if edges.size == 0:
        return float("nan")
    vector = np.asarray(values, dtype=np.float64)
    return float(np.mean((vector[edges[:, 0]] - vector[edges[:, 1]]) ** 2))


def predicted_spatiality(
    train_truth: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    descriptors: Mapping[str, np.ndarray],
    edges: np.ndarray,
) -> np.ndarray:
    train_moran = np.asarray(
        [moran_i(train_truth[:, position], edges) for position in range(train_truth.shape[1])],
        dtype=np.float32,
    )
    finite = np.isfinite(train_moran)
    if int(finite.sum()) < 2:
        raise ReadoutError("Spatiality predictor needs at least two finite train-gene Moran values")
    target = train_moran.copy()
    target[~finite] = float(np.nanmedian(target[finite]))
    descriptor = np.asarray(descriptors["pca32_nmf32"], dtype=np.float32)
    finite_descriptor = np.isfinite(descriptor[train_idx]).all(axis=1)
    usable = finite & finite_descriptor
    if int(usable.sum()) < 2:
        raise ReadoutError("Spatiality predictor has fewer than two usable training genes")
    model = Ridge(alpha=1.0)
    model.fit(descriptor[train_idx][usable], target[usable])
    return model.predict(descriptor[eval_idx]).astype(np.float32)


def prediction_features(
    descriptor_kind: str,
    prediction: np.ndarray,
    gene_idx: np.ndarray,
    descriptors: Mapping[str, np.ndarray],
    spatiality: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float32)
    if pred.shape[1] != len(gene_idx) or pred.shape[1] != len(spatiality):
        raise ValueError("Prediction, index, and spatiality columns must align")
    rows: list[list[float]] = []
    for position in range(pred.shape[1]):
        values = np.clip(pred[:, position], 0.0, None)
        quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append(
            [
                float(np.mean(values)),
                float(np.std(values)),
                float(np.min(values)),
                float(np.max(values)),
                *[float(value) for value in quantiles],
                float(np.mean(values <= 1e-6)),
                float(moran_i(values, edges)),
                float(graph_smoothness(values, edges)),
                float(spatiality[position]),
            ]
        )
    descriptor = np.asarray(descriptors[descriptor_kind], dtype=np.float32)[gene_idx]
    features = np.concatenate([descriptor, np.asarray(rows, dtype=np.float32)], axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _legacy_metric_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    true_std = np.std(true, axis=0)
    pred_std = np.std(pred, axis=0)
    true_rank = rankdata(true, axis=0)
    pred_rank = rankdata(pred, axis=0)
    true_rank -= np.mean(true_rank, axis=0, keepdims=True)
    pred_rank -= np.mean(pred_rank, axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(true_rank * true_rank, axis=0) * np.sum(pred_rank * pred_rank, axis=0)
    )
    spcc = np.divide(
        np.sum(true_rank * pred_rank, axis=0),
        denominator,
        out=np.full(true.shape[1], np.nan),
        where=denominator > EPS,
    )
    spcc[(true_std <= EPS) | (pred_std <= EPS)] = np.nan
    true_z = (true - np.mean(true, axis=0, keepdims=True)) / np.maximum(true_std, EPS)
    pred_z = (pred - np.mean(pred, axis=0, keepdims=True)) / np.maximum(pred_std, EPS)
    rmse = np.sqrt(np.mean((true_z - pred_z) ** 2, axis=0))
    true_mass = true / np.maximum(np.sum(true, axis=0, keepdims=True), EPS)
    pred_mass = pred / np.maximum(np.sum(pred, axis=0, keepdims=True), EPS)
    midpoint = 0.5 * (true_mass + pred_mass)
    with np.errstate(divide="ignore", invalid="ignore"):
        jsd = 0.5 * np.sum(
            np.where(
                true_mass > 0.0,
                true_mass * np.log(np.maximum(true_mass, EPS) / np.maximum(midpoint, EPS)),
                0.0,
            ),
            axis=0,
        )
        jsd += 0.5 * np.sum(
            np.where(
                pred_mass > 0.0,
                pred_mass * np.log(np.maximum(pred_mass, EPS) / np.maximum(midpoint, EPS)),
                0.0,
            ),
            axis=0,
        )
    true_scaled = true / np.maximum(np.max(true, axis=0, keepdims=True), EPS)
    pred_scaled = pred / np.maximum(np.max(pred, axis=0, keepdims=True), EPS)
    data_range = np.maximum.reduce(
        [
            np.max(true_scaled, axis=0),
            np.max(pred_scaled, axis=0),
            np.full(true.shape[1], EPS),
        ]
    )
    mean_true = np.mean(true_scaled, axis=0)
    mean_pred = np.mean(pred_scaled, axis=0)
    sigma_true = np.sqrt(
        np.mean((true_scaled - mean_true[None, :]) ** 2, axis=0)
    )
    sigma_pred = np.sqrt(
        np.mean((pred_scaled - mean_pred[None, :]) ** 2, axis=0)
    )
    covariance = np.mean(
        (true_scaled - mean_true[None, :])
        * (pred_scaled - mean_pred[None, :]),
        axis=0,
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    c3 = c2 / 2.0
    luminance = (2.0 * mean_true * mean_pred + c1) / (
        mean_true * mean_true + mean_pred * mean_pred + c1
    )
    contrast = (2.0 * sigma_true * sigma_pred + c2) / (
        sigma_true * sigma_true + sigma_pred * sigma_pred + c2
    )
    structure = (covariance + c3) / (sigma_true * sigma_pred + c3)
    return {"SPCC": spcc, "RMSE": rmse, "JSD": jsd, "SSIM": luminance * contrast * structure}


def uniform_train_oracle_choices(
    train_truth: np.ndarray,
    pred_train: np.ndarray,
    train_idx: np.ndarray,
    *,
    fold: int,
) -> pd.DataFrame:
    """Build the frozen per-train-gene affine targets without test information."""

    true = np.asarray(train_truth, dtype=np.float64)
    pred = np.clip(np.asarray(pred_train, dtype=np.float64), 0.0, None)
    if true.shape != pred.shape or pred.shape[1] != len(train_idx):
        raise ValueError("Train truth, prediction, and indices must align")
    base = _legacy_metric_arrays(true, pred)
    n_genes = pred.shape[1]
    best_metrics = {key: values.copy() for key, values in base.items()}
    best_method = np.full(n_genes, "identity", dtype=object)
    best_a = np.ones(n_genes, dtype=np.float64)
    best_b = np.zeros(n_genes, dtype=np.float64)
    best_ssim = base["SSIM"].copy()
    mean_true = np.mean(true, axis=0)
    mean_pred = np.mean(pred, axis=0)
    std_true = np.std(true, axis=0)
    std_pred = np.std(pred, axis=0)
    span = np.maximum(
        np.quantile(pred, 0.95, axis=0) - np.quantile(pred, 0.05, axis=0),
        np.maximum(std_pred, 1e-3),
    )
    variance_pred = np.var(pred, axis=0)
    covariance = np.mean((pred - mean_pred[None, :]) * (true - mean_true[None, :]), axis=0)
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    scales = [
        np.ones(n_genes),
        np.maximum(mean_true / np.maximum(mean_pred, EPS), 1e-4),
        np.maximum(std_true / np.maximum(std_pred, EPS), 1e-4),
        *[np.full(n_genes, value, dtype=np.float64) for value in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0)],
    ]
    for scale in scales:
        candidates.append(("scale_only", np.clip(scale, 1e-4, 10.0), np.zeros(n_genes)))
    candidates.append(("shift_only", np.ones(n_genes), mean_true - mean_pred))
    for fraction in np.linspace(-0.75, 0.75, 13):
        candidates.append(("shift_only", np.ones(n_genes), fraction * span))
    least_squares = np.divide(
        covariance, variance_pred, out=np.ones(n_genes), where=variance_pred > EPS
    )
    candidates.append(
        ("positive_affine", np.clip(least_squares, 1e-4, 10.0), mean_true - least_squares * mean_pred)
    )
    mean_std = np.divide(std_true, std_pred, out=np.ones(n_genes), where=std_pred > EPS)
    candidates.append(
        ("positive_affine", np.clip(mean_std, 1e-4, 10.0), mean_true - mean_std * mean_pred)
    )
    for value in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        scale = np.full(n_genes, value, dtype=np.float64)
        mean_match = mean_true - value * mean_pred
        candidates.append(("positive_affine", scale, mean_match))
        for fraction in np.linspace(-0.5, 0.5, 7):
            candidates.append(("positive_affine", scale, mean_match + fraction * span))

    for method, scale, shift in candidates:
        candidate = np.clip(pred * scale[None, :] + shift[None, :], 0.0, None)
        metrics = _legacy_metric_arrays(true, candidate)
        guard = (
            np.nan_to_num(metrics["SPCC"], nan=-1.0)
            >= np.nan_to_num(base["SPCC"], nan=-1.0) - 1e-10
        )
        guard &= metrics["RMSE"] <= base["RMSE"] + 1e-10
        guard &= metrics["JSD"] <= base["JSD"] + 0.0015
        better = guard & (metrics["SSIM"] > best_ssim)
        if np.any(better):
            best_ssim[better] = metrics["SSIM"][better]
            best_method[better] = method
            best_a[better] = scale[better]
            best_b[better] = shift[better]
            for key in best_metrics:
                best_metrics[key][better] = metrics[key][better]

    rows: list[dict[str, Any]] = []
    for position, gene_index in enumerate(train_idx):
        rows.append(
            {
                "fold": int(fold),
                "split": "train",
                "gene_idx": int(gene_index),
                "gene_pos": int(position),
                "method": str(best_method[position]),
                "a": float(best_a[position]),
                "b": float(best_b[position]),
                "guard_pass": True,
                **{key: float(best_metrics[key][position]) for key in METRICS},
            }
        )
    return pd.DataFrame(rows)


def target_parameters(train_choices: pd.DataFrame, pred_train: np.ndarray) -> np.ndarray:
    ordered = train_choices.sort_values("gene_pos")
    if len(ordered) != pred_train.shape[1]:
        raise ReadoutError(
            f"Train choices {len(ordered)} do not match prediction columns {pred_train.shape[1]}"
        )
    parameters: list[list[float]] = []
    for _, row in ordered.iterrows():
        position = int(row["gene_pos"])
        standard_deviation = float(np.std(pred_train[:, position]))
        parameters.append(
            [
                math.log(max(float(row["a"]), 1e-4)),
                float(row["b"]) / max(standard_deviation, 1e-3),
            ]
        )
    return np.asarray(parameters, dtype=np.float32)


def fit_predict_parameters(
    model_kind: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    eval_features: np.ndarray,
    alpha: float,
    seed: int,
) -> np.ndarray:
    if model_kind == "ridge":
        model = Ridge(alpha=float(alpha))
        model.fit(train_features, train_targets)
        return model.predict(eval_features).astype(np.float32)
    if model_kind == "elasticnet":
        outputs: list[np.ndarray] = []
        for target_position in range(train_targets.shape[1]):
            model = ElasticNet(
                alpha=float(alpha),
                l1_ratio=0.2,
                max_iter=5000,
                random_state=int(seed) + target_position,
            )
            model.fit(train_features, train_targets[:, target_position])
            outputs.append(model.predict(eval_features))
        return np.vstack(outputs).T.astype(np.float32)
    raise ValueError(f"Unknown parameter model {model_kind!r}")


def apply_predicted_parameters(
    prediction: np.ndarray, parameters: np.ndarray, mode: str
) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float32)
    if parameters.shape != (pred.shape[1], 2):
        raise ValueError(
            f"Parameter shape {parameters.shape} does not match {pred.shape[1]} genes"
        )
    output = pred.copy()
    for position in range(pred.shape[1]):
        log_scale, normalized_shift = parameters[position]
        scale = float(np.clip(np.exp(log_scale), 0.2, 5.0))
        shift = float(normalized_shift) * max(float(np.std(pred[:, position])), 1e-3)
        if mode == "mean_only":
            scale = 1.0
        elif mode == "scale_only":
            shift = 0.0
        output[:, position] = np.clip(scale * pred[:, position] + shift, 0.0, None)
    return output.astype(np.float32)


def apply_global_affine(prediction: np.ndarray, scale: float, shift: float) -> np.ndarray:
    return np.clip(
        float(scale) * np.asarray(prediction, dtype=np.float64) + float(shift),
        0.0,
        None,
    ).astype(np.float32)


def _guard_component(candidate: float, base: float, *, lower: bool, tolerance: float) -> bool:
    if math.isnan(candidate) and math.isnan(base):
        return True
    if not math.isfinite(candidate) or not math.isfinite(base):
        return False
    return candidate >= base - tolerance if lower else candidate <= base + tolerance


def validation_guard(candidate: Mapping[str, float], base: Mapping[str, float]) -> bool:
    return (
        _guard_component(
            float(candidate["SPCC"]),
            float(base["SPCC"]),
            lower=True,
            tolerance=GUARD_TOLERANCES["SPCC"],
        )
        and _guard_component(
            float(candidate["RMSE"]),
            float(base["RMSE"]),
            lower=False,
            tolerance=GUARD_TOLERANCES["RMSE"],
        )
        and _guard_component(
            float(candidate["JSD"]),
            float(base["JSD"]),
            lower=False,
            tolerance=GUARD_TOLERANCES["JSD"],
        )
    )


def _raw_tie_score(metrics: Mapping[str, float]) -> float:
    values = [float(metrics[key]) for key in ("SPCC", "RMSE", "JSD")]
    if not all(math.isfinite(value) for value in values):
        return -math.inf
    return values[0] - values[1] - values[2]


def candidate_names() -> list[str]:
    names = ["identity", "global_affine", "positive_global_affine"]
    for descriptor_kind in FEATURE_KINDS:
        for target in TARGET_MODES:
            for model_kind, alpha in MODEL_GRID:
                names.append(
                    f"{descriptor_kind}_{model_kind}_{target}_alpha{alpha:g}"
                )
    if len(names) != EXPECTED_CANDIDATE_COUNT or len(set(names)) != len(names):
        raise AssertionError("Candidate family is not the frozen 57-row grid")
    return names


def frozen_protocol_definition() -> dict[str, Any]:
    return {
        "readout_layer": "model_specific_validation_selected_genespt57",
        "applies_to": ["GeneSPT", "GeneSPT-GC"],
        "candidate_family": candidate_names(),
        "feature_kinds": list(FEATURE_KINDS),
        "target_modes": list(TARGET_MODES),
        "model_grid": [[kind, alpha] for kind, alpha in MODEL_GRID],
        "guard_tolerances": dict(GUARD_TOLERANCES),
        "selection": "maximum_validation_SSIM_subject_to_guards_then_raw_tie_score",
        "seed": 42,
        "test_status": "additional_method_layer_not_preregistered",
        "external_baselines": "raw_identity_only_no_descriptor_readout",
    }


def build_validation_candidates(
    *,
    method: str,
    fold: int,
    train_truth: np.ndarray,
    val_truth: np.ndarray,
    pred_train: np.ndarray,
    pred_val: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    descriptors: Mapping[str, np.ndarray],
    spatiality_train: np.ndarray,
    spatiality_val: np.ndarray,
    edges: np.ndarray,
    seed: int,
    metrics_module: Any,
    train_choices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    """Fit and select validation candidates; this function has no test inputs."""

    raw_train = np.clip(np.asarray(pred_train, dtype=np.float32), 0.0, None)
    raw_val = np.clip(np.asarray(pred_val, dtype=np.float32), 0.0, None)
    choices = (
        uniform_train_oracle_choices(train_truth, raw_train, train_idx, fold=fold)
        if train_choices is None
        else train_choices.sort_values("gene_pos").reset_index(drop=True)
    )
    targets = target_parameters(choices, raw_train)
    feature_cache: dict[tuple[str, str], np.ndarray] = {}

    def features(kind: str, split: str) -> np.ndarray:
        key = (kind, split)
        if key not in feature_cache:
            feature_cache[key] = prediction_features(
                kind,
                raw_train if split == "train" else raw_val,
                train_idx if split == "train" else val_idx,
                descriptors,
                spatiality_train if split == "train" else spatiality_val,
                edges,
            )
        return feature_cache[key]

    base = fast_metric_summary(val_truth, raw_val)
    rows: list[dict[str, Any]] = []
    metric_audits: list[dict[str, Any]] = []
    selected_name = "identity"
    selected_prediction = raw_val
    best_ssim = float(base["SSIM"])
    best_tie = _raw_tie_score(base)

    def add_candidate(
        name: str,
        prediction: np.ndarray | None,
        detail: Mapping[str, Any],
        *,
        error: str = "",
    ) -> None:
        nonlocal selected_name, selected_prediction, best_ssim, best_tie
        row: dict[str, Any] = {
            "method": method,
            "fold": int(fold),
            "candidate_order": len(rows),
            "calibration": name,
            "status": "ok" if prediction is not None else "failed",
            "error": error,
            "guard_pass": False,
            "selected": False,
            "val_SPCC": float("nan"),
            "val_RMSE": float("nan"),
            "val_JSD": float("nan"),
            "val_JS": float("nan"),
            "val_SSIM": float("nan"),
            "delta_val_SPCC": float("nan"),
            "delta_val_RMSE": float("nan"),
            "delta_val_JSD": float("nan"),
            "delta_val_JS": float("nan"),
            "delta_val_SSIM": float("nan"),
            "feature_kind": detail.get("feature_kind", ""),
            "model_kind": detail.get("model_kind", ""),
            "target": detail.get("target", ""),
            "mode": detail.get("mode", ""),
            "alpha": detail.get("alpha", float("nan")),
            "a": detail.get("a", float("nan")),
            "b": detail.get("b", float("nan")),
            "fit_truth_scope": detail.get("fit_truth_scope", ""),
        }
        if prediction is not None:
            metrics = fast_metric_summary(val_truth, prediction)
            guard = validation_guard(metrics, base)
            row.update(
                {
                    "guard_pass": bool(guard),
                    **{f"val_{key}": float(metrics[key]) for key in METRICS},
                    "val_JS": float(metrics["JSD"]),
                    **{
                        f"delta_val_{key}": float(metrics[key]) - float(base[key])
                        for key in METRICS
                    },
                    "delta_val_JS": float(metrics["JSD"]) - float(base["JSD"]),
                }
            )
            score = float(metrics["SSIM"])
            tie = _raw_tie_score(metrics)
            better = guard and (
                score > best_ssim + 1e-12
                or (abs(score - best_ssim) <= 1e-12 and tie > best_tie)
            )
            if better:
                selected_name = name
                selected_prediction = prediction.copy()
                best_ssim = score
                best_tie = tie
        rows.append(row)

    add_candidate("identity", raw_val, {"fit_truth_scope": "none"})
    flattened_true = np.asarray(val_truth, dtype=np.float64).reshape(-1)
    flattened_pred = np.asarray(raw_val, dtype=np.float64).reshape(-1)
    variance = float(np.var(flattened_pred))
    scale = (
        float(np.cov(flattened_pred, flattened_true, bias=True)[0, 1] / variance)
        if variance > EPS
        else 1.0
    )
    shift = float(np.mean(flattened_true) - scale * np.mean(flattened_pred))
    add_candidate(
        "global_affine",
        apply_global_affine(raw_val, scale, shift),
        {"a": scale, "b": shift, "fit_truth_scope": "validation"},
    )
    positive_scale = max(scale, 1e-4)
    positive_shift = float(
        np.mean(flattened_true) - positive_scale * np.mean(flattened_pred)
    )
    add_candidate(
        "positive_global_affine",
        apply_global_affine(raw_val, positive_scale, positive_shift),
        {"a": positive_scale, "b": positive_shift, "fit_truth_scope": "validation"},
    )

    for descriptor_kind in FEATURE_KINDS:
        train_features = features(descriptor_kind, "train")
        val_features = features(descriptor_kind, "val")
        for target in TARGET_MODES:
            mode = target
            for model_kind, alpha in MODEL_GRID:
                name = f"{descriptor_kind}_{model_kind}_{target}_alpha{alpha:g}"
                detail = {
                    "feature_kind": descriptor_kind,
                    "model_kind": model_kind,
                    "target": target,
                    "mode": mode,
                    "alpha": float(alpha),
                    "fit_truth_scope": "training_genes",
                }
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        parameters = fit_predict_parameters(
                            model_kind,
                            train_features,
                            targets,
                            val_features,
                            alpha,
                            seed + 101 * fold,
                        )
                    candidate = apply_predicted_parameters(raw_val, parameters, mode)
                    add_candidate(name, candidate, detail)
                except Exception as exc:
                    add_candidate(
                        name,
                        None,
                        detail,
                        error=f"{type(exc).__name__}: {exc}",
                    )

    if [row["calibration"] for row in rows] != candidate_names():
        raise AssertionError("Constructed candidate order differs from the frozen family")
    selected_matches = [row for row in rows if row["calibration"] == selected_name]
    if len(selected_matches) != 1:
        raise AssertionError(f"Selected candidate {selected_name!r} is not unique")
    selected_matches[0]["selected"] = True

    for label, prediction, expected in (
        ("identity", raw_val, base),
        (selected_name, selected_prediction, fast_metric_summary(val_truth, selected_prediction)),
    ):
        _, central_frame = metrics_module.evaluate_prediction(val_truth, prediction)
        central = central_frame.iloc[0].to_dict()
        matched, differences = metric_summaries_close(expected, central)
        metric_audits.append(
            {
                "method": method,
                "fold": fold,
                "calibration": label,
                "fast_vs_centralized_match": bool(matched),
                **{f"abs_diff_{key}": differences[key] for key in METRICS},
            }
        )
        if not matched:
            raise ReadoutError(
                f"Fast validation metrics differ from centralized evaluator for "
                f"{method} fold{fold} {label}: {differences}"
            )

    selected_row = dict(selected_matches[0])
    selected_row["test_prediction_array_materialized_before_selection"] = False
    selected_row["test_truth_array_materialized_before_selection"] = False
    return pd.DataFrame(rows), selected_row, choices, metric_audits


def apply_selected_candidate(
    *,
    selected: Mapping[str, Any],
    pred_train: np.ndarray,
    pred_test: np.ndarray,
    train_choices: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    descriptors: Mapping[str, np.ndarray],
    spatiality_train: np.ndarray,
    spatiality_test: np.ndarray,
    edges: np.ndarray,
    seed: int,
    fold: int,
) -> np.ndarray:
    name = str(selected["calibration"])
    raw_test = np.clip(np.asarray(pred_test, dtype=np.float32), 0.0, None)
    if name == "identity":
        return raw_test
    if name in {"global_affine", "positive_global_affine"}:
        return apply_global_affine(raw_test, float(selected["a"]), float(selected["b"]))
    descriptor_kind = str(selected["feature_kind"])
    model_kind = str(selected["model_kind"])
    mode = str(selected["mode"])
    alpha = float(selected["alpha"])
    targets = target_parameters(train_choices, np.asarray(pred_train, dtype=np.float32))
    train_features = prediction_features(
        descriptor_kind,
        pred_train,
        train_idx,
        descriptors,
        spatiality_train,
        edges,
    )
    test_features = prediction_features(
        descriptor_kind,
        raw_test,
        test_idx,
        descriptors,
        spatiality_test,
        edges,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        parameters = fit_predict_parameters(
            model_kind,
            train_features,
            targets,
            test_features,
            alpha,
            seed + 1000 + fold,
        )
    return apply_predicted_parameters(raw_test, parameters, mode)
