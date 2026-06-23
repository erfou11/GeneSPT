from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genespt.metrics import evaluate_prediction, gene_metrics


def test_spcc_is_rank_based_not_pearson() -> None:
    y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_pred = np.array([[1.0], [4.0], [9.0], [16.0], [100.0]])

    per_gene, summary = evaluate_prediction(y_true, y_pred)

    assert np.isclose(per_gene.loc[0, "SPCC"], 1.0)
    assert np.isclose(summary.loc[0, "SPCC"], 1.0)
    assert not np.isclose(np.corrcoef(y_true[:, 0], y_pred[:, 0])[0, 1], 1.0)


def test_spcc_reverse_order_is_negative_one() -> None:
    y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_pred = np.array([[5.0], [4.0], [3.0], [2.0], [1.0]])

    per_gene, summary = evaluate_prediction(y_true, y_pred)

    assert np.isclose(per_gene.loc[0, "SPCC"], -1.0)
    assert np.isclose(summary.loc[0, "SPCC"], -1.0)


def test_spcc_constant_vector_returns_nan_and_summary_uses_final_median() -> None:
    y_true = np.column_stack(
        [
            np.array([1, 1, 1, 1, 1], dtype=float),
            np.array([1, 2, 3, 4, 5], dtype=float),
        ]
    )
    y_pred = np.column_stack(
        [
            np.array([1, 2, 3, 4, 5], dtype=float),
            np.array([1, 4, 9, 16, 100], dtype=float),
        ]
    )

    per_gene = gene_metrics(y_true, y_pred)
    _, summary = evaluate_prediction(y_true, y_pred)

    assert np.isnan(per_gene.loc[0, "SPCC"])
    assert np.isclose(summary.loc[0, "SPCC"], 1.0)
