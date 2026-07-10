"""Synthetic smoke test for the centralized public evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genespt.metrics import evaluate_prediction


def main() -> None:
    y_true = np.column_stack(
        [
            np.array([1, 2, 3, 4, 5], dtype=float),
            np.array([5, 4, 3, 2, 1], dtype=float),
        ]
    )
    y_pred = np.column_stack(
        [
            np.array([1, 4, 9, 16, 100], dtype=float),
            np.array([1, 2, 3, 4, 5], dtype=float),
        ]
    )

    per_gene, summary = evaluate_prediction(y_true, y_pred, ["monotone", "reverse"])
    assert list(per_gene["gene"]) == ["monotone", "reverse"]
    assert np.isclose(per_gene.loc[0, "SPCC"], 1.0)
    assert np.isclose(per_gene.loc[1, "SPCC"], -1.0)
    assert np.isfinite(summary[["RMSE", "JS", "JS/JSD", "SSIM"]].to_numpy()).all()
    print("smoke_test=ok")


if __name__ == "__main__":
    main()
