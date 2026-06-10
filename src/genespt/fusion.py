"""Validation-selected fusion/readout utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FusionResult:
    psp_weight: float
    validation_rmse: float


def apply_fusion(gc_pred: np.ndarray, psp_pred: np.ndarray, psp_weight: float) -> np.ndarray:
    """Return ``(1 - weight) * GC + weight * PSP`` with nonnegative clipping."""

    w = float(psp_weight)
    fused = (1.0 - w) * np.asarray(gc_pred) + w * np.asarray(psp_pred)
    return np.clip(fused, 0.0, None).astype(np.float32)


def select_global_fusion(
    gc_val_pred: np.ndarray,
    psp_val_pred: np.ndarray,
    val_true: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> FusionResult:
    """Select a global PSP weight using validation genes only."""

    weights = np.asarray(grid if grid is not None else np.linspace(0.0, 1.0, 21), dtype=np.float64)
    best = FusionResult(psp_weight=0.0, validation_rmse=float("inf"))
    y = np.asarray(val_true, dtype=np.float64)
    for w in weights:
        pred = apply_fusion(gc_val_pred, psp_val_pred, float(w))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        if rmse < best.validation_rmse:
            best = FusionResult(psp_weight=float(w), validation_rmse=rmse)
    return best

