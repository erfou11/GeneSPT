#!/usr/bin/env python3
"""Recompute the formal Protocol A benchmark from compact archived matrices."""

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
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from genespt.metrics import evaluate_prediction  # noqa: E402


METHODS = (
    "GeneSPT",
    "Tangram",
    "TransImp",
    "SpaIM",
    "SpaGE",
    "stPlus",
    "stAI",
)
METRICS = ("SPCC", "RMSE", "JSD", "SSIM")
LOWER_IS_BETTER = {"RMSE", "JSD"}
FOLDS = (0, 1, 2, 3, 4)
CHUNK_BYTES = 8 * 1024 * 1024


class RecomputeError(RuntimeError):
    """Raised when the public matrix contract cannot be verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RecomputeError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_archive_path(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise RecomputeError(f"Manifest path escapes archive root: {value}") from exc
    if not candidate.is_file():
        raise RecomputeError(f"Manifest file is missing: {candidate}")
    return candidate


def verify_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RecomputeError(
            f"SHA256 mismatch for {label}: expected {expected}, observed {observed}"
        )


def load_gene_names(path: Path) -> np.ndarray:
    values = [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]
    if not values or len(set(values)) != len(values):
        raise RecomputeError(f"Gene-name axis is empty or non-unique: {path}")
    return np.asarray(values, dtype=str)


def validate_prediction_manifest(rows: Sequence[Mapping[str, str]]) -> None:
    expected = 6 * len(FOLDS) * len(METHODS)
    if len(rows) != expected:
        raise RecomputeError(f"Expected {expected} prediction rows, found {len(rows)}")
    keys: set[tuple[str, int, str]] = set()
    for row in rows:
        key = (row["dataset_id"], int(row["fold"]), row["method"])
        if key in keys:
            raise RecomputeError(f"Duplicate prediction row: {key}")
        keys.add(key)
        if row["method"] not in METHODS:
            raise RecomputeError(f"Unexpected method: {row['method']}")
        if int(row["fold"]) not in FOLDS:
            raise RecomputeError(f"Unexpected fold: {row['fold']}")
        if row["matrix_scope"] != "final_test_genes":
            raise RecomputeError(f"Unexpected matrix scope for {key}")
    for dataset_id in sorted({row["dataset_id"] for row in rows}):
        for fold in FOLDS:
            observed = {method for ds, current_fold, method in keys if ds == dataset_id and current_fold == fold}
            if observed != set(METHODS):
                raise RecomputeError(
                    f"Incomplete method set for {dataset_id} fold{fold}: {sorted(observed)}"
                )


def truth_index(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, int], Mapping[str, str]]:
    if len(rows) != 6 * len(FOLDS):
        raise RecomputeError(f"Expected 30 truth rows, found {len(rows)}")
    result: dict[tuple[str, int], Mapping[str, str]] = {}
    for row in rows:
        key = (row["dataset_id"], int(row["fold"]))
        if key in result:
            raise RecomputeError(f"Duplicate truth row: {key}")
        result[key] = row
    return result


def expected_index(path: Path | None) -> dict[tuple[str, int, str], Mapping[str, str]]:
    if path is None:
        return {}
    result: dict[tuple[str, int, str], Mapping[str, str]] = {}
    for row in read_csv(path):
        key = (row["dataset_id"], int(row["fold"]), row["method"])
        result[key] = row
    return result


def load_compact(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise RecomputeError(f"{path} does not contain array {key!r}")
        values = np.asarray(archive[key], dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RecomputeError(f"Invalid compact array in {path}")
    return values


def compare_expected(
    observed: Mapping[str, Any],
    expected: Mapping[str, str],
    tolerance: float,
) -> None:
    identity = (observed["dataset_id"], int(observed["fold"]), observed["method"])
    for metric in METRICS:
        first = float(observed[metric])
        second = float(expected[metric])
        if not math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance):
            raise RecomputeError(
                f"Metric mismatch for {identity} {metric}: observed {first}, expected {second}"
            )


def recompute(
    *,
    archive_root: Path,
    prediction_rows: Sequence[Mapping[str, str]],
    truth_rows: Mapping[tuple[str, int], Mapping[str, str]],
    expected_rows: Mapping[tuple[str, int, str], Mapping[str, str]],
    verify_hashes: bool,
    tolerance: float,
    save_gene_level: bool,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    fold_rows: list[dict[str, Any]] = []
    gene_frames: list[pd.DataFrame] = []
    gene_name_cache: dict[str, np.ndarray] = {}
    truth_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for row in sorted(
        prediction_rows,
        key=lambda item: (item["dataset_id"], int(item["fold"]), METHODS.index(item["method"])),
    ):
        dataset_id = row["dataset_id"]
        fold = int(row["fold"])
        method = row["method"]
        key = (dataset_id, fold)
        truth_row = truth_rows[key]
        if key not in truth_cache:
            truth_path = resolve_archive_path(archive_root, truth_row["truth_path"])
            truth_idx_path = resolve_archive_path(archive_root, truth_row["test_gene_idx_path"])
            if verify_hashes:
                verify_sha(truth_path, truth_row["truth_sha256"], f"truth {key}")
                verify_sha(
                    truth_idx_path,
                    truth_row["test_gene_idx_sha256"],
                    f"truth test index {key}",
                )
            truth_cache[key] = (
                load_compact(truth_path, "truth"),
                np.asarray(np.load(truth_idx_path, allow_pickle=False), dtype=np.int64),
            )
        truth, truth_idx = truth_cache[key]

        prediction_path = resolve_archive_path(archive_root, row["matrix_path"])
        prediction_idx_path = resolve_archive_path(archive_root, row["test_gene_idx_path"])
        if verify_hashes:
            verify_sha(
                prediction_path,
                row["compact_prediction_sha256"],
                f"prediction {(dataset_id, fold, method)}",
            )
            verify_sha(
                prediction_idx_path,
                row["test_gene_idx_sha256"],
                f"prediction test index {(dataset_id, fold, method)}",
            )
        prediction = load_compact(prediction_path, "prediction")
        prediction_idx = np.asarray(
            np.load(prediction_idx_path, allow_pickle=False), dtype=np.int64
        )
        if prediction.shape != truth.shape:
            raise RecomputeError(
                f"Shape mismatch for {(dataset_id, fold, method)}: "
                f"prediction {prediction.shape}, truth {truth.shape}"
            )
        if not np.array_equal(prediction_idx, truth_idx):
            raise RecomputeError(f"Test-index mismatch for {(dataset_id, fold, method)}")

        if dataset_id not in gene_name_cache:
            gene_name_cache[dataset_id] = load_gene_names(
                resolve_archive_path(archive_root, row["gene_names_path"])
            )
        all_gene_names = gene_name_cache[dataset_id]
        if truth_idx.max(initial=-1) >= len(all_gene_names):
            raise RecomputeError(f"Test index exceeds gene-name axis for {dataset_id}")
        test_gene_names = all_gene_names[truth_idx]
        per_gene, summary = evaluate_prediction(truth, prediction, test_gene_names)
        values = summary.iloc[0]
        fold_row: dict[str, Any] = {
            "dataset": row["dataset"],
            "dataset_id": dataset_id,
            "role": row["role"],
            "fold": fold,
            "method": method,
            "result_layer": row["result_layer"],
            "test_gene_count": int(truth.shape[1]),
            "coverage": float(values["coverage"]),
            "SPCC": float(values["SPCC"]),
            "RMSE": float(values["RMSE"]),
            "JSD": float(values["JSD"]),
            "SSIM": float(values["SSIM"]),
            "constant_prediction_genes": int(values["constant_prediction_genes"]),
        }
        expected = expected_rows.get((dataset_id, fold, method))
        if expected is not None:
            compare_expected(fold_row, expected, tolerance)
        fold_rows.append(fold_row)
        if save_gene_level:
            per_gene.insert(0, "method", method)
            per_gene.insert(0, "fold", fold)
            per_gene.insert(0, "dataset_id", dataset_id)
            per_gene.insert(0, "dataset", row["dataset"])
            gene_frames.append(per_gene)
        print(f"[evaluated] {method} {dataset_id} fold{fold}", flush=True)
    return fold_rows, gene_frames


def five_fold_summary(fold_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(fold_rows)
    rows: list[dict[str, Any]] = []
    for (dataset, dataset_id, role, method, layer), group in frame.groupby(
        ["dataset", "dataset_id", "role", "method", "result_layer"], sort=True
    ):
        if sorted(group["fold"].astype(int).tolist()) != list(FOLDS):
            raise RecomputeError(f"Incomplete five-fold group: {(dataset_id, method)}")
        row: dict[str, Any] = {
            "dataset": dataset,
            "dataset_id": dataset_id,
            "role": role,
            "method": method,
            "result_layer": layer,
            "fold_count": len(group),
            "coverage": float(group["coverage"].mean()),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    if len(rows) != 6 * len(METHODS):
        raise RecomputeError(
            f"Expected {6 * len(METHODS)} five-fold rows, found {len(rows)}"
        )
    return rows


def rank_summary(five_fold_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(five_fold_rows)
    rows: list[dict[str, Any]] = []
    for (dataset, dataset_id, role), group in frame.groupby(
        ["dataset", "dataset_id", "role"], sort=True
    ):
        for metric in METRICS:
            ascending = metric in LOWER_IS_BETTER
            ranks = group[metric].rank(method="min", ascending=ascending)
            for (_, item), rank in zip(group.iterrows(), ranks, strict=True):
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_id": dataset_id,
                        "role": role,
                        "metric": metric,
                        "method": item["method"],
                        "value": float(item[metric]),
                        "rank": int(rank),
                    }
                )
    return rows


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument("--truth-manifest", type=Path)
    parser.add_argument("--expected-fold-metrics", type=Path)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--save-gene-level", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    archive_root = args.archive_root.resolve(strict=True)
    prediction_manifest = args.prediction_manifest or (
        archive_root / "prediction_matrix_manifests" / "PREDICTION_MATRIX_MANIFEST.csv"
    )
    truth_manifest = args.truth_manifest or (
        archive_root
        / "prediction_matrix_manifests"
        / "PROTOCOL_A_TRUTH_MATRIX_MANIFEST.csv"
    )
    expected_path = args.expected_fold_metrics or (
        archive_root
        / "protocol_a_reproducibility"
        / "source_data"
        / "formal_fold_metrics.csv"
    )
    prediction_rows = read_csv(prediction_manifest)
    validate_prediction_manifest(prediction_rows)
    truth_rows = truth_index(read_csv(truth_manifest))
    expected_rows = expected_index(expected_path if expected_path.is_file() else None)
    fold_rows, gene_frames = recompute(
        archive_root=archive_root,
        prediction_rows=prediction_rows,
        truth_rows=truth_rows,
        expected_rows=expected_rows,
        verify_hashes=not args.skip_hash_check,
        tolerance=args.tolerance,
        save_gene_level=args.save_gene_level,
    )
    five_fold_rows = five_fold_summary(fold_rows)
    rank_rows = rank_summary(five_fold_rows)
    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "recomputed_fold_metrics.csv", fold_rows)
    write_csv(output_dir / "recomputed_five_fold_metrics.csv", five_fold_rows)
    write_csv(output_dir / "recomputed_benchmark_ranks.csv", rank_rows)
    if args.save_gene_level:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(gene_frames, ignore_index=True).to_csv(
            output_dir / "recomputed_gene_level_metrics.csv", index=False
        )
    rank_frame = pd.DataFrame(rank_rows)
    genespt_wins = (
        rank_frame.loc[rank_frame["method"] == "GeneSPT"]
        .groupby("role")["rank"]
        .apply(lambda values: int((values == 1).sum()))
        .to_dict()
    )
    summary = {
        "status": "complete",
        "prediction_rows": len(prediction_rows),
        "truth_rows": len(truth_rows),
        "fold_metric_rows": len(fold_rows),
        "five_fold_rows": len(five_fold_rows),
        "expected_fold_metrics_verified": bool(expected_rows),
        "hashes_verified": not args.skip_hash_check,
        "genespt_rank1_items": genespt_wins,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recompute_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecomputeError as exc:
        raise SystemExit(f"ERROR: {exc}")
