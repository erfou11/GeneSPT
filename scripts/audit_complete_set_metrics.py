#!/usr/bin/env python3
"""Audit aligned predictions with the centralized complete-set evaluator.

The CLI accepts one truth matrix and one or more already aligned prediction
matrices. It writes per-gene metrics, method summaries, and an input manifest;
it never trains a model or mutates an archived result package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genespt.io import load_array, load_gene_names  # noqa: E402
from genespt.metrics import evaluate_prediction  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_matrix(path: Path, key: str | None) -> np.ndarray:
    matrix = np.asarray(load_array(path, key=key))
    if matrix.ndim != 2:
        raise ValueError(
            f"{path} must contain a two-dimensional matrix; got {matrix.shape}"
        )
    return matrix


def audit_predictions(
    truth: np.ndarray,
    predictions: Sequence[tuple[str, np.ndarray]],
    gene_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate multiple methods against one fixed truth matrix."""

    if not predictions:
        raise ValueError("At least one prediction is required")
    methods = [method for method, _ in predictions]
    if len(methods) != len(set(methods)):
        raise ValueError("Prediction method labels must be unique")

    per_gene_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    eligibility_reference: pd.DataFrame | None = None
    eligibility_columns = [
        "truth_finite",
        "truth_constant",
        "truth_zero_mass_after_nonnegative_clipping",
        "eligible_truth",
        "SPCC_eligible",
        "RMSE_eligible",
        "SSIM_eligible",
        "JSD_eligible",
    ]

    for method, prediction in predictions:
        per_gene, summary = evaluate_prediction(truth, prediction, gene_names)
        current_eligibility = per_gene.loc[:, eligibility_columns].reset_index(
            drop=True
        )
        if eligibility_reference is None:
            eligibility_reference = current_eligibility
        elif not current_eligibility.equals(eligibility_reference):
            raise AssertionError("Truth-defined eligibility changed between methods")

        per_gene.insert(0, "method", method)
        summary.insert(0, "method", method)
        per_gene_frames.append(per_gene)
        summary_frames.append(summary)

    return (
        pd.concat(per_gene_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
    )


def run_self_test() -> dict[str, int | str]:
    """Exercise the reviewer-facing edge cases without external data."""

    truth = np.column_stack(
        [
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            np.zeros(4, dtype=float),
            np.full(4, 2.0),
        ]
    )
    predictions = [
        (
            "constant-edge",
            np.column_stack(
                [
                    np.zeros(4, dtype=float),
                    np.asarray([0.0, 1.0, 0.0, 2.0]),
                    np.full(4, 2.0),
                ]
            ),
        ),
        ("identity", truth.copy()),
    ]
    per_gene, summary = audit_predictions(
        truth,
        predictions,
        ["positive_truth", "zero_truth", "constant_positive_truth"],
    )
    edge = per_gene[per_gene["method"].eq("constant-edge")].reset_index(drop=True)
    edge_summary = summary[summary["method"].eq("constant-edge")].iloc[0]

    assert edge.loc[0, "SPCC"] == 0.0
    assert np.isclose(edge.loc[0, "JSD"], math.log(2.0), atol=1e-15, rtol=0.0)
    assert np.isnan(edge.loc[1, "JSD"])
    assert np.isfinite(edge.loc[1, ["RMSE", "SSIM"]].to_numpy(float)).all()
    assert int(edge_summary["SPCC_eligible"]) == 1
    assert int(edge_summary["JSD_eligible"]) == 2
    assert float(edge_summary["coverage"]) == 1.0
    return {"status": "PASS", "methods": 2, "genes": 3}


def run_audit(
    *,
    truth_path: Path,
    truth_key: str | None,
    prediction_specs: Sequence[Sequence[str]],
    prediction_key: str | None,
    gene_names_path: Path | None,
    test_indices_path: Path | None,
    output_dir: Path,
) -> dict[str, int | str]:
    truth = _load_matrix(truth_path, truth_key)
    test_indices: np.ndarray | None = None
    if test_indices_path is not None:
        test_indices = np.asarray(
            load_array(test_indices_path), dtype=np.int64
        ).reshape(-1)
        if len(np.unique(test_indices)) != len(test_indices):
            raise ValueError("test indices contain duplicates")
        if np.any(test_indices < 0) or np.any(test_indices >= truth.shape[1]):
            raise ValueError("test indices are outside the truth matrix gene axis")
        truth = truth[:, test_indices]

    gene_names = load_gene_names(gene_names_path, test_indices)
    if gene_names is not None and len(gene_names) != truth.shape[1]:
        raise ValueError(
            f"gene name count {len(gene_names)} does not match {truth.shape[1]} truth columns"
        )

    prediction_paths: list[tuple[str, Path]] = []
    predictions: list[tuple[str, np.ndarray]] = []
    for raw_method, raw_path in prediction_specs:
        method = str(raw_method)
        path = Path(raw_path)
        prediction_paths.append((method, path))
        predictions.append((method, _load_matrix(path, prediction_key)))

    per_gene, summary = audit_predictions(truth, predictions, gene_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_gene.to_csv(output_dir / "complete_set_per_gene_metrics.csv", index=False)
    summary.to_csv(output_dir / "complete_set_summary_metrics.csv", index=False)

    inputs = [
        {
            "role": "truth",
            "path": str(truth_path.resolve()),
            "bytes": truth_path.stat().st_size,
            "sha256": _sha256(truth_path),
        }
    ]
    for method, path in prediction_paths:
        inputs.append(
            {
                "role": f"prediction:{method}",
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    for role, path in (
        ("gene_names", gene_names_path),
        ("test_indices", test_indices_path),
    ):
        if path is not None:
            inputs.append(
                {
                    "role": role,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

    manifest = {
        "policy": "truth-defined complete set; see docs/metric_policy.md",
        "matrix_orientation": "spots x genes",
        "truth_shape_after_subsetting": list(map(int, truth.shape)),
        "methods": [method for method, _ in predictions],
        "nonfinite_prediction_policy": "raise for any truth-eligible gene",
        "inputs": inputs,
    }
    (output_dir / "complete_set_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "methods": len(predictions),
        "genes": int(truth.shape[1]),
        "output_dir": str(output_dir),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth", type=Path, help="Truth matrix (.npy, .npz, .csv, or .tsv)"
    )
    parser.add_argument(
        "--truth-key", default=None, help="Array key for a multi-array truth NPZ"
    )
    parser.add_argument(
        "--prediction",
        action="append",
        nargs=2,
        metavar=("METHOD", "PATH"),
        help="Repeat for each aligned prediction matrix",
    )
    parser.add_argument(
        "--prediction-key",
        default="prediction",
        help="Array key used for prediction NPZ files (ignored for other formats)",
    )
    parser.add_argument("--gene-names", type=Path, default=None)
    parser.add_argument("--test-indices", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        print(json.dumps(run_self_test(), indent=2))
        return 0
    missing = [
        name
        for name, value in (
            ("--truth", args.truth),
            ("--prediction", args.prediction),
            ("--out-dir", args.out_dir),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            f"Missing required arguments outside --self-test: {', '.join(missing)}"
        )
    result = run_audit(
        truth_path=args.truth,
        truth_key=args.truth_key,
        prediction_specs=args.prediction,
        prediction_key=args.prediction_key,
        gene_names_path=args.gene_names,
        test_indices_path=args.test_indices,
        output_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
