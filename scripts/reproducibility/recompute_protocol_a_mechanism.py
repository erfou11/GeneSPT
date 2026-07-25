#!/usr/bin/env python3
"""Recompute Figure 3 and Supplementary Table S3 metrics from release matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from genespt.metrics import evaluate_prediction  # noqa: E402


FOLDS = (0, 1, 2, 3, 4)
METRICS = ("SPCC", "RMSE", "JSD", "SSIM")
CHUNK_BYTES = 8 * 1024 * 1024


class MechanismRecomputeError(RuntimeError):
    """Raised when the release mechanism package cannot be reproduced."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise MechanismRecomputeError(f"Refusing to write an empty table: {path}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_matrix(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise MechanismRecomputeError(f"Missing {key!r} array: {path}")
        values = np.asarray(archive[key], dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise MechanismRecomputeError(f"Invalid matrix: {path}")
    return values


def load_gene_names(path: Path) -> np.ndarray:
    names = np.asarray(
        [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()],
        dtype=str,
    )
    if names.ndim != 1 or names.size == 0:
        raise MechanismRecomputeError(f"Invalid gene-name axis: {path}")
    return names


def expected_index(path: Path) -> dict[tuple[str, str, int, str], dict[str, str]]:
    index: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for row in read_csv(path):
        key = (row["panel"], row["dataset_id"], int(row["fold"]), row["control"])
        if key in index:
            raise MechanismRecomputeError(f"Duplicate expected S3 row: {key}")
        index[key] = row
    return index


def truth_index(archive_root: Path) -> dict[tuple[str, int], dict[str, str]]:
    path = (
        archive_root
        / "prediction_matrix_manifests"
        / "PROTOCOL_A_TRUTH_MATRIX_MANIFEST.csv"
    )
    return {
        (row["dataset_id"], int(row["fold"])): row for row in read_csv(path)
    }


def compare_metrics(
    observed: Mapping[str, Any], expected: Mapping[str, str], tolerance: float
) -> None:
    identity = (
        observed["panel"],
        observed["dataset_id"],
        int(observed["fold"]),
        observed["control"],
    )
    for metric in METRICS:
        if not math.isclose(
            float(observed[metric]),
            float(expected[metric]),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise MechanismRecomputeError(
                f"Metric mismatch for {identity} {metric}: "
                f"observed={observed[metric]}, expected={expected[metric]}"
            )
    if not math.isclose(
        float(observed["coverage"]),
        float(expected["coverage"]),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise MechanismRecomputeError(f"Coverage mismatch for {identity}")


def recompute(
    *,
    archive_root: Path,
    manifest_rows: Sequence[Mapping[str, str]],
    expected_rows: Mapping[tuple[str, str, int, str], Mapping[str, str]],
    verify_hashes: bool,
    tolerance: float,
) -> list[dict[str, Any]]:
    truth_rows = truth_index(archive_root)
    truth_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    gene_cache: dict[str, np.ndarray] = {}
    output: list[dict[str, Any]] = []
    for row in manifest_rows:
        panel = row["panel"]
        dataset_id = row["dataset_id"]
        fold = int(row["fold"])
        control = row["control"]
        identity = (panel, dataset_id, fold, control)
        if row["readout"] != "identity" or row["posthoc_calibration"] != "none":
            raise MechanismRecomputeError(f"Non-identity mechanism row: {identity}")

        prediction_path = archive_root / row["matrix_path"]
        prediction_idx_path = archive_root / row["test_gene_idx_path"]
        if verify_hashes:
            if sha256_file(prediction_path) != row["compact_prediction_sha256"]:
                raise MechanismRecomputeError(f"Prediction hash mismatch: {identity}")
            if sha256_file(prediction_idx_path) != row["test_gene_idx_sha256"]:
                raise MechanismRecomputeError(f"Prediction index hash mismatch: {identity}")

        truth_row = truth_rows[(dataset_id, fold)]
        if (dataset_id, fold) not in truth_cache:
            truth_path = archive_root / truth_row["truth_path"]
            truth_idx_path = archive_root / truth_row["test_gene_idx_path"]
            if verify_hashes:
                if sha256_file(truth_path) != truth_row["truth_sha256"]:
                    raise MechanismRecomputeError(f"Truth hash mismatch: {(dataset_id, fold)}")
                if sha256_file(truth_idx_path) != truth_row["test_gene_idx_sha256"]:
                    raise MechanismRecomputeError(
                        f"Truth index hash mismatch: {(dataset_id, fold)}"
                    )
            truth_cache[(dataset_id, fold)] = (
                load_matrix(truth_path, "truth"),
                np.asarray(np.load(truth_idx_path, allow_pickle=False), dtype=np.int64),
            )
        truth, truth_idx = truth_cache[(dataset_id, fold)]
        prediction = load_matrix(prediction_path, "prediction")
        prediction_idx = np.asarray(
            np.load(prediction_idx_path, allow_pickle=False), dtype=np.int64
        )
        if prediction.shape != truth.shape or not np.array_equal(prediction_idx, truth_idx):
            raise MechanismRecomputeError(f"Prediction/truth alignment mismatch: {identity}")

        if dataset_id not in gene_cache:
            gene_cache[dataset_id] = load_gene_names(
                archive_root / row["gene_names_path"]
            )
        all_names = gene_cache[dataset_id]
        if truth_idx.max(initial=-1) >= all_names.size:
            raise MechanismRecomputeError(f"Gene index exceeds name axis: {dataset_id}")
        _, summary = evaluate_prediction(truth, prediction, all_names[truth_idx])
        values = summary.iloc[0]
        result: dict[str, Any] = {
            "panel": panel,
            "dataset": row["dataset"],
            "dataset_id": dataset_id,
            "fold": fold,
            "setting": row["setting"],
            "control": control,
            "model": row["model"],
            "result_layer": row["result_layer"],
            "readout": row["readout"],
            "posthoc_calibration": row["posthoc_calibration"],
            "coverage": float(values["coverage"]),
            "eligible_genes": int(values["eligible_genes"]),
            "scored_genes": int(values["scored_genes"]),
            "constant_prediction_genes": int(values["constant_prediction_genes"]),
            "SPCC": float(values["SPCC"]),
            "RMSE": float(values["RMSE"]),
            "JSD": float(values["JSD"]),
            "SSIM": float(values["SSIM"]),
        }
        expected = expected_rows.get(identity)
        if expected is None:
            raise MechanismRecomputeError(f"Missing expected S3 row: {identity}")
        compare_metrics(result, expected, tolerance)
        output.append(result)
        print(f"[verified] Figure 3{panel} {dataset_id} fold{fold} {control}", flush=True)
    if len(output) != 90:
        raise MechanismRecomputeError(f"Expected 90 mechanism rows, found {len(output)}")
    return output


def five_fold_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    groups = [
        "panel",
        "dataset",
        "dataset_id",
        "setting",
        "control",
        "model",
        "result_layer",
        "readout",
        "posthoc_calibration",
    ]
    for key, group in frame.groupby(groups, sort=True):
        if sorted(group["fold"].astype(int).tolist()) != list(FOLDS):
            raise MechanismRecomputeError(f"Incomplete mechanism group: {key}")
        row = dict(zip(groups, key, strict=True))
        row["n_folds"] = len(group)
        row["coverage"] = float(group["coverage"].mean())
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[metric] = float(values.mean())
            row[f"{metric}_fold_sd_ddof0"] = float(values.std(ddof=0))
        output.append(row)
    if len(output) != 18:
        raise MechanismRecomputeError(f"Expected 18 five-fold groups, found {len(output)}")
    return output


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-fold-metrics", type=Path)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    archive_root = args.archive_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    manifest = (
        archive_root
        / "prediction_matrix_manifests"
        / "MECHANISM_ABLATION_MATRIX_MANIFEST.csv"
    )
    expected_path = args.expected_fold_metrics or (
        archive_root
        / "results_source_data"
        / "supplementary"
        / "supplementary_table_s3_fold_level.csv"
    )
    manifest_rows = read_csv(manifest)
    expected_rows = expected_index(expected_path)
    rows = recompute(
        archive_root=archive_root,
        manifest_rows=manifest_rows,
        expected_rows=expected_rows,
        verify_hashes=not args.skip_hash_check,
        tolerance=args.tolerance,
    )
    summaries = five_fold_summary(rows)
    write_csv(output_dir / "mechanism_fold_metrics.csv", rows)
    write_csv(output_dir / "mechanism_five_fold_summary.csv", summaries)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "matrix_rows": len(rows),
        "five_fold_rows": len(summaries),
        "expected_fold_metrics_verified": True,
        "hashes_verified": not args.skip_hash_check,
        "tolerance": args.tolerance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "RECOMPUTE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
