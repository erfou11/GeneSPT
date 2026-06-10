"""Predictable Spatial Program Transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge


@dataclass
class PSPModel:
    spatial_programs: np.ndarray
    ridge: Ridge
    selected_components: np.ndarray
    component_scores: np.ndarray

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        params = self.ridge.predict(np.asarray(descriptors, dtype=np.float64))
        means = params[:, 0]
        coeffs = params[:, 1:]
        mask = np.zeros(coeffs.shape[1], dtype=np.float64)
        mask[self.selected_components] = 1.0
        pred = means[None, :] + self.spatial_programs @ (coeffs * mask).T
        return np.clip(pred, 0.0, None).astype(np.float32)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(pearsonr(x, y).statistic)


def _project_coefficients(spatial_programs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = y.mean(axis=0)
    centered = y - means[None, :]
    coeffs = np.linalg.pinv(spatial_programs) @ centered
    return means, coeffs.T


def fit_psp(
    train_st: np.ndarray,
    train_descriptors: np.ndarray,
    val_st: np.ndarray,
    val_descriptors: np.ndarray,
    *,
    n_components: int = 16,
    ridge_alpha: float = 10.0,
    min_component_corr: float = 0.0,
    top_k: int | None = None,
) -> PSPModel:
    """Fit descriptor-predictable, validation-screened PSP.

    Spatial programs are estimated from training genes only. Ridge coefficient
    prediction is fitted on training genes only. Validation genes are used only
    to decide which spatial components are predictably transferable.
    """

    x_train = np.asarray(train_st, dtype=np.float64)
    d_train = np.asarray(train_descriptors, dtype=np.float64)
    x_val = np.asarray(val_st, dtype=np.float64)
    d_val = np.asarray(val_descriptors, dtype=np.float64)
    k = int(max(1, min(n_components, x_train.shape[0], x_train.shape[1])))

    train_means = x_train.mean(axis=0)
    centered = x_train - train_means[None, :]
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    spatial_programs = u[:, :k] * s[:k][None, :]
    train_coeffs = vt[:k, :].T
    train_targets = np.column_stack([train_means, train_coeffs])

    ridge = Ridge(alpha=float(ridge_alpha))
    ridge.fit(d_train, train_targets)

    val_means, val_coeffs = _project_coefficients(spatial_programs, x_val)
    true_val_targets = np.column_stack([val_means, val_coeffs])
    pred_val_targets = ridge.predict(d_val)
    component_scores = np.array(
        [_safe_corr(true_val_targets[:, j + 1], pred_val_targets[:, j + 1]) for j in range(k)],
        dtype=np.float64,
    )

    if top_k is not None:
        selected = np.argsort(component_scores)[::-1][: max(1, min(int(top_k), k))]
    else:
        selected = np.flatnonzero(component_scores >= float(min_component_corr))
        if selected.size == 0:
            selected = np.array([int(np.argmax(component_scores))], dtype=int)

    return PSPModel(
        spatial_programs=spatial_programs.astype(np.float32),
        ridge=ridge,
        selected_components=np.asarray(selected, dtype=int),
        component_scores=component_scores.astype(np.float32),
    )

