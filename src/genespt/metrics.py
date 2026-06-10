"""Centralized metrics for strict whole-gene prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr


def _spcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(pearsonr(y_true, y_pred).statistic)


def _jsd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, None)
    b = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    if a.sum() <= 0:
        a = np.ones_like(a)
    if b.sum() <= 0:
        b = np.ones_like(b)
    return float(jensenshannon(a / a.sum(), b / b.sum(), base=2.0) ** 2)


def _ssim_vector(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    x = np.asarray(y_true, dtype=np.float64)
    y = np.asarray(y_pred, dtype=np.float64)
    data_range = max(float(np.max([x.max(), y.max()]) - np.min([x.min(), y.min()])), 1e-6)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mux, muy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mux) * (y - muy))
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))


def gene_metrics(y_true: np.ndarray, y_pred: np.ndarray, gene_names: list[str] | None = None) -> pd.DataFrame:
    """Compute per-gene SPCC, RMSE, JS/JSD, and raw-scale SSIM."""

    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: true={true.shape}, pred={pred.shape}")
    names = gene_names or [f"gene_{i}" for i in range(true.shape[1])]
    rows = []
    for j, name in enumerate(names):
        yt = true[:, j]
        yp = pred[:, j]
        rows.append(
            {
                "gene": name,
                "SPCC": _spcc(yt, yp),
                "RMSE": float(np.sqrt(np.mean((yp - yt) ** 2))),
                "JS/JSD": _jsd(yt, yp),
                "SSIM": _ssim_vector(yt, yp),
            }
        )
    return pd.DataFrame(rows)


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray, gene_names: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-gene metrics and one-row summary metrics."""

    per_gene = gene_metrics(y_true, y_pred, gene_names)
    summary = per_gene[["SPCC", "RMSE", "JS/JSD", "SSIM"]].mean().to_frame().T
    return per_gene, summary

