from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genespt.metrics import evaluate_prediction, gene_metrics  # noqa: E402


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


def test_constant_truth_only_makes_spcc_ineligible() -> None:
    y_true = np.column_stack(
        [
            np.full(5, 3.0),
            np.array([1, 2, 3, 4, 5], dtype=float),
        ]
    )
    y_pred = np.column_stack(
        [
            np.array([1, 2, 3, 4, 5], dtype=float),
            np.array([1, 4, 9, 16, 100], dtype=float),
        ]
    )

    per_gene, summary = evaluate_prediction(y_true, y_pred)

    assert np.isnan(per_gene.loc[0, "SPCC"])
    assert np.isfinite(per_gene.loc[0, ["RMSE", "SSIM"]].to_numpy(float)).all()
    assert np.isclose(summary.loc[0, "SPCC"], 1.0)
    assert summary.loc[0, "SPCC_eligible"] == 1
    assert summary.loc[0, "RMSE_eligible"] == 2
    assert summary.loc[0, "SSIM_eligible"] == 2


def test_constant_prediction_spcc_is_zero() -> None:
    y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_pred = np.full((5, 1), 7.0)

    per_gene, summary = evaluate_prediction(y_true, y_pred)

    assert per_gene.loc[0, "prediction_constant"]
    assert per_gene.loc[0, "SPCC"] == 0.0
    assert summary.loc[0, "SPCC"] == 0.0
    assert summary.loc[0, "constant_prediction"] == 1


def test_positive_truth_and_all_zero_prediction_has_log_two_jsd() -> None:
    y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_pred = np.zeros((4, 1), dtype=float)

    per_gene, summary = evaluate_prediction(y_true, y_pred)

    expected = math.log(2.0)
    assert np.isclose(per_gene.loc[0, "JSD"], expected, atol=1e-15, rtol=0.0)
    assert per_gene.loc[0, "JS"] == per_gene.loc[0, "JSD"]
    assert per_gene.loc[0, "JS/JSD"] == per_gene.loc[0, "JSD"]
    assert np.isclose(summary.loc[0, "JSD"], expected, atol=1e-15, rtol=0.0)
    assert summary.loc[0, "JS"] == summary.loc[0, "JSD"]
    assert summary.loc[0, "JS/JSD"] == summary.loc[0, "JSD"]


def test_all_zero_truth_jsd_is_method_independent_na() -> None:
    y_true = np.zeros((4, 2), dtype=float)
    y_pred = np.column_stack(
        [
            np.zeros(4, dtype=float),
            np.array([1.0, 2.0, 3.0, 4.0]),
        ]
    )

    per_gene, summary = evaluate_prediction(y_true, y_pred)

    assert per_gene["JSD"].isna().all()
    assert np.isnan(summary.loc[0, "JSD"])
    assert summary.loc[0, "JSD_eligible"] == 0
    assert summary.loc[0, "JSD_scored"] == 0


def test_rmse_and_ssim_remain_finite_for_zero_constant_truth() -> None:
    y_true = np.zeros((5, 1), dtype=float)
    y_pred = np.array([[0.0], [1.0], [0.0], [2.0], [0.0]])

    per_gene, summary = evaluate_prediction(y_true, y_pred)

    assert np.isfinite(per_gene.loc[0, ["RMSE", "SSIM"]].to_numpy(float)).all()
    assert np.isfinite(summary.loc[0, ["RMSE", "SSIM"]].to_numpy(float)).all()


def test_nonfinite_prediction_for_finite_truth_fails() -> None:
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[1.0], [np.nan], [3.0]])

    try:
        evaluate_prediction(y_true, y_pred, ["bad_prediction"])
    except ValueError as error:
        assert "Nonfinite prediction" in str(error)
        assert "bad_prediction" in str(error)
    else:
        raise AssertionError("Expected a nonfinite prediction to fail evaluation")


def test_summary_reports_truth_fixed_coverage() -> None:
    y_true = np.column_stack(
        [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.full(4, 2.0),
            np.full(4, np.nan),
        ]
    )
    y_pred = np.column_stack(
        [
            np.array([4.0, 3.0, 2.0, 1.0]),
            np.full(4, 5.0),
            np.full(4, np.nan),
        ]
    )

    per_gene, summary = evaluate_prediction(y_true, y_pred)

    assert per_gene["eligible_truth"].tolist() == [True, True, False]
    assert summary.loc[0, "total"] == 3
    assert summary.loc[0, "eligible"] == 2
    assert summary.loc[0, "scored"] == 2
    assert summary.loc[0, "constant_prediction"] == 1
    assert summary.loc[0, "coverage"] == 1.0
    assert summary.loc[0, "total_genes"] == 3
    assert summary.loc[0, "eligible_genes"] == 2
    assert summary.loc[0, "scored_genes"] == 2
    assert summary.loc[0, "constant_prediction_genes"] == 1
    assert summary.loc[0, "SPCC_eligible"] == 1
    assert summary.loc[0, "SPCC_scored"] == 1
    assert summary.loc[0, "SPCC_coverage"] == 1.0
    assert summary.loc[0, "RMSE_eligible"] == 2
    assert summary.loc[0, "RMSE_scored"] == 2
    assert summary.loc[0, "RMSE_coverage"] == 1.0


def test_gene_names_length_is_validated() -> None:
    y_true = np.ones((3, 2), dtype=float)
    y_pred = np.ones((3, 2), dtype=float)

    try:
        gene_metrics(y_true, y_pred, ["only_one_name"])
    except ValueError as error:
        assert "gene_names length" in str(error)
    else:
        raise AssertionError("Expected mismatched gene_names to fail evaluation")
