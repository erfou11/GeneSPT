"""Thin public wrapper around the anchored final manuscript evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANCHORED_MAIN = _REPO_ROOT / "main"
if str(_ANCHORED_MAIN) not in sys.path:
    sys.path.insert(0, str(_ANCHORED_MAIN))

from run_strict_gene_conditioned_decoder_gate import gene_metrics as _anchored_gene_metrics  # noqa: E402


def gene_metrics(y_true: np.ndarray, y_pred: np.ndarray, gene_names: list[str] | None = None) -> pd.DataFrame:
    """Compute metrics with the final local manuscript evaluator logic."""

    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: true={true.shape}, pred={pred.shape}")
    names = gene_names or [f"gene_{i}" for i in range(true.shape[1])]
    idx = np.arange(true.shape[1], dtype=np.int64)
    per_gene = _anchored_gene_metrics(true, pred, idx, list(names)).copy()
    per_gene["JS/JSD"] = per_gene["JS"]
    return per_gene


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray, gene_names: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-gene metrics and one-row median summary metrics."""

    per_gene = gene_metrics(y_true, y_pred, gene_names)
    summary = {
        metric: float(np.nanmedian(per_gene[metric]))
        for metric in ["SPCC", "RMSE", "JS", "SSIM"]
    }
    summary["JS/JSD"] = summary["JS"]
    return per_gene, pd.DataFrame([summary])
