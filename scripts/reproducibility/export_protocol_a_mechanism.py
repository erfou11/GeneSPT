#!/usr/bin/env python3
"""Export the formal Protocol A Figure 3 mechanism matrices.

The exporter keeps only final-test predictions and frozen test-gene indices.
It reads the completed formal run, writes into a separate release build, and
never mutates the formal experiment directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FOLDS = (0, 1, 2, 3, 4)
PANELS = ("A", "B", "C")
CHUNK_BYTES = 8 * 1024 * 1024


class MechanismExportError(RuntimeError):
    """Raised when a formal mechanism artifact violates the release contract."""


def io_path(path: Path) -> Path:
    """Return a Windows extended-length path for long formal artifact names."""

    if os.name == "nt" and path.is_absolute():
        value = str(path)
        if not value.startswith("\\\\?\\"):
            return Path("\\\\?\\" + value)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with io_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise MechanismExportError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise MechanismExportError(f"Refusing to write an empty manifest: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise MechanismExportError(f"Inconsistent columns while writing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def save_prediction(path: Path, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, prediction=prediction)
    temporary.replace(path)


def archive_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise MechanismExportError(f"Path is outside the release root: {path}") from exc


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not io_path(path).is_file():
        raise MechanismExportError(f"Missing source prediction: {path}")
    with np.load(io_path(path), allow_pickle=False) as archive:
        if "prediction" not in archive.files or "test_gene_idx" not in archive.files:
            raise MechanismExportError(
                f"Source must contain prediction and test_gene_idx arrays: {path}"
            )
        prediction = np.asarray(archive["prediction"], dtype=np.float32)
        test_idx = np.asarray(archive["test_gene_idx"], dtype=np.int64)
    if prediction.ndim != 2 or test_idx.ndim != 1:
        raise MechanismExportError(f"Invalid source array dimensions: {path}")
    if prediction.shape[1] != test_idx.size:
        raise MechanismExportError(f"Prediction/test-index shape mismatch: {path}")
    if not np.isfinite(prediction).all():
        raise MechanismExportError(f"Non-finite prediction values: {path}")
    return prediction, test_idx


def truth_index(build_root: Path) -> dict[tuple[str, int], dict[str, str]]:
    path = (
        build_root
        / "prediction_matrix_manifests"
        / "PROTOCOL_A_TRUTH_MATRIX_MANIFEST.csv"
    )
    rows = read_csv(path)
    index = {(row["dataset_id"], int(row["fold"])): row for row in rows}
    if len(index) != 30:
        raise MechanismExportError(f"Expected 30 formal truth rows, found {len(index)}")
    return index


def s3_index(formal_root: Path) -> dict[tuple[str, str, int, str], dict[str, str]]:
    path = formal_root / "supplementary" / "S3" / "supplementary_table_s3_fold_level.csv"
    rows = read_csv(path)
    index: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for row in rows:
        key = (row["panel"], row["dataset_id"], int(row["fold"]), row["control"])
        if key in index:
            raise MechanismExportError(f"Duplicate S3 fold row: {key}")
        index[key] = row
    return index


def benchmark_source(formal_root: Path, dataset_id: str, fold: int, model: str) -> Path:
    return (
        formal_root
        / "genespt"
        / "benchmark"
        / dataset_id
        / f"fold{fold}"
        / "protocol_a_genespt_prediction_matrices"
        / model
        / f"fold{fold}"
        / "prediction.npz"
    )


def figure3a_source(formal_root: Path, fold: int, control: str, model: str) -> Path:
    if control == "correct":
        return benchmark_source(
            formal_root, "Vis9A_D7_spaim_effective4470", fold, "gc_mlp_base"
        )
    return (
        formal_root
        / "mechanism"
        / "figure3_a_descriptor_controls"
        / f"fold{fold}"
        / control
        / "prediction.npz"
    )


def formal_relative_path(value: str, formal_root: Path) -> Path:
    normalized = value.replace("\\", "/")
    marker = "results/protocol_a_full_rerun_20260711/"
    if marker not in normalized:
        raise MechanismExportError(f"Unrecognized formal source path: {value}")
    return formal_root / normalized.split(marker, 1)[1]


def figure3c_sources(formal_root: Path, fold: int) -> dict[str, tuple[Path, str]]:
    manifest = read_json(
        formal_root
        / "mechanism"
        / "figure3_c_primary_mechanism_controls"
        / f"fold{fold}"
        / "prediction_source_manifest.json"
    )
    result: dict[str, tuple[Path, str]] = {}
    predictions = manifest.get("predictions")
    if not isinstance(predictions, list):
        raise MechanismExportError(f"Invalid Figure 3C source manifest for fold {fold}")
    for item in predictions:
        if not isinstance(item, dict):
            raise MechanismExportError(f"Invalid Figure 3C source row for fold {fold}")
        result[str(item["control"])] = (
            formal_relative_path(str(item["path"]), formal_root),
            str(item["sha256"]),
        )
    if len(result) != 8:
        raise MechanismExportError(
            f"Expected eight Figure 3C controls for fold {fold}, found {len(result)}"
        )
    return result


def expected_rows(formal_root: Path) -> list[dict[str, Any]]:
    s3 = read_csv(
        formal_root / "supplementary" / "S3" / "supplementary_table_s3_fold_level.csv"
    )
    rows: list[dict[str, Any]] = []
    for row in s3:
        panel = row["panel"]
        if panel not in PANELS:
            continue
        rows.append(
            {
                "panel": panel,
                "dataset": row["dataset"],
                "dataset_id": row["dataset_id"],
                "fold": int(row["fold"]),
                "setting": row["setting"],
                "control": row["control"],
                "model": row["model"],
                "source_kind": row["source_kind"],
                "result_layer": row["result_layer"],
                "readout": row["readout"],
                "posthoc_calibration": row["posthoc_calibration"],
                "reported_SPCC": row["SPCC"],
                "reported_RMSE": row["RMSE"],
                "reported_JSD": row["JSD"],
                "reported_SSIM": row["SSIM"],
                "reported_coverage": row["coverage"],
            }
        )
    rows.sort(key=lambda row: (PANELS.index(row["panel"]), row["dataset_id"], row["fold"], row["control"]))
    expected_count = 20 + 30 + 40
    if len(rows) != expected_count:
        raise MechanismExportError(
            f"Expected {expected_count} formal mechanism rows, found {len(rows)}"
        )
    return rows


def source_for_row(
    formal_root: Path,
    row: Mapping[str, Any],
    figure3c_cache: dict[int, dict[str, tuple[Path, str]]],
) -> tuple[Path, str]:
    panel = str(row["panel"])
    fold = int(row["fold"])
    control = str(row["control"])
    model = str(row["model"])
    if panel == "A":
        source = figure3a_source(formal_root, fold, control, model)
        return source, sha256_file(source)
    if panel == "B":
        source = benchmark_source(formal_root, str(row["dataset_id"]), fold, model)
        return source, sha256_file(source)
    if fold not in figure3c_cache:
        figure3c_cache[fold] = figure3c_sources(formal_root, fold)
    try:
        source, expected_hash = figure3c_cache[fold][control]
    except KeyError as exc:
        raise MechanismExportError(
            f"Figure 3C source is missing control {control!r} for fold {fold}"
        ) from exc
    observed_hash = sha256_file(source)
    if observed_hash != expected_hash:
        raise MechanismExportError(f"Figure 3C source hash mismatch: {source}")
    return source, observed_hash


def export(formal_root: Path, build_root: Path) -> list[dict[str, Any]]:
    truth = truth_index(build_root)
    s3 = s3_index(formal_root)
    rows = expected_rows(formal_root)
    output_root = build_root / "mechanism_ablation_prediction_matrices"
    if output_root.exists() and any(output_root.iterdir()):
        raise MechanismExportError(
            f"Output directory is not empty; isolate it before rebuilding: {output_root}"
        )
    figure3c_cache: dict[int, dict[str, tuple[Path, str]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        panel = str(row["panel"])
        dataset_id = str(row["dataset_id"])
        fold = int(row["fold"])
        control = str(row["control"])
        s3_row = s3[(panel, dataset_id, fold, control)]
        truth_row = truth[(dataset_id, fold)]
        truth_idx_path = build_root / truth_row["test_gene_idx_path"]
        truth_idx = np.asarray(np.load(truth_idx_path, allow_pickle=False), dtype=np.int64)
        source, source_hash = source_for_row(formal_root, row, figure3c_cache)
        prediction, source_idx = load_prediction(source)
        if not np.array_equal(source_idx, truth_idx):
            raise MechanismExportError(
                f"Frozen test-index mismatch for panel {panel}, {dataset_id}, fold {fold}, {control}"
            )

        destination = (
            output_root
            / f"panel_{panel.lower()}"
            / dataset_id
            / control
            / f"fold{fold}"
        )
        matrix_path = destination / "prediction.npz"
        index_path = destination / "test_gene_idx.npy"
        metadata_path = destination / "metadata.json"
        save_prediction(matrix_path, prediction)
        save_npy(index_path, source_idx)
        metadata = {
            "schema_version": 1,
            "protocol": "Protocol A strict whole-gene holdout",
            "panel": f"Figure 3{panel}",
            "dataset": row["dataset"],
            "dataset_id": dataset_id,
            "fold": fold,
            "setting": row["setting"],
            "control": control,
            "model": row["model"],
            "source_kind": row["source_kind"],
            "result_layer": row["result_layer"],
            "readout": "identity",
            "posthoc_calibration": "none",
            "matrix_scope": "final_test_genes",
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype),
            "source_prediction_relpath": source.relative_to(formal_root).as_posix(),
            "source_prediction_sha256": source_hash,
            "source_prediction_array_sha256": sha256_array(prediction),
            "frozen_test_gene_idx_sha256": sha256_array(source_idx),
            "truth_path": truth_row["truth_path"],
            "truth_sha256": truth_row["truth_sha256"],
            "gene_names_path": truth_row["gene_names_path"],
            "reported_fold_metrics": {
                "coverage": float(s3_row["coverage"]),
                "SPCC": float(s3_row["SPCC"]),
                "RMSE": float(s3_row["RMSE"]),
                "JSD": float(s3_row["JSD"]),
                "SSIM": float(s3_row["SSIM"]),
            },
        }
        write_json(metadata_path, metadata)
        manifest_rows.append(
            {
                "panel": panel,
                "dataset": row["dataset"],
                "dataset_id": dataset_id,
                "fold": fold,
                "setting": row["setting"],
                "control": control,
                "model": row["model"],
                "source_kind": row["source_kind"],
                "result_layer": row["result_layer"],
                "readout": "identity",
                "posthoc_calibration": "none",
                "matrix_scope": "final_test_genes",
                "matrix_path": archive_relative(matrix_path, build_root),
                "metadata_path": archive_relative(metadata_path, build_root),
                "test_gene_idx_path": archive_relative(index_path, build_root),
                "truth_path": truth_row["truth_path"],
                "gene_names_path": truth_row["gene_names_path"],
                "shape": json.dumps(list(prediction.shape), separators=(",", ":")),
                "dtype": str(prediction.dtype),
                "compact_prediction_sha256": sha256_file(matrix_path),
                "compact_prediction_bytes": matrix_path.stat().st_size,
                "metadata_sha256": sha256_file(metadata_path),
                "test_gene_idx_sha256": sha256_file(index_path),
                "source_prediction_relpath": source.relative_to(formal_root).as_posix(),
                "source_prediction_sha256": source_hash,
                "source_prediction_array_sha256": sha256_array(prediction),
                "truth_sha256": truth_row["truth_sha256"],
                "reported_SPCC": row["reported_SPCC"],
                "reported_RMSE": row["reported_RMSE"],
                "reported_JSD": row["reported_JSD"],
                "reported_SSIM": row["reported_SSIM"],
                "reported_coverage": row["reported_coverage"],
            }
        )
        print(f"[exported] Figure 3{panel} {dataset_id} fold{fold} {control}", flush=True)
    return manifest_rows


README = """# Figure 3 mechanism prediction matrices

This directory contains compact final-test prediction matrices for the formal
Protocol A mechanism analyses reported in Figure 3 and Supplementary Table S3.

- `panel_a/` contains the Vis9A gene-descriptor controls.
- `panel_b/` contains matched GeneSPT-GC versus GeneSPT-GC+PSP comparisons for
  Vis9A, HBC and Cell2location mouse brain.
- `panel_c/` contains the Vis9A PSP mechanism controls.

Every matrix uses identity readout and no post-hoc calibration. Rows are spatial
spots or cells; columns follow the accompanying frozen `test_gene_idx.npy`.
Fold-specific ground truth is stored separately under `ground_truth_protocol_a/`.
The complete file and source hashes are listed in
`prediction_matrix_manifests/MECHANISM_ABLATION_MATRIX_MANIFEST.csv`.
"""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    formal_root = args.formal_root.resolve(strict=True)
    build_root = args.build_root.resolve(strict=True)
    rows = export(formal_root, build_root)
    manifest_path = (
        build_root
        / "prediction_matrix_manifests"
        / "MECHANISM_ABLATION_MATRIX_MANIFEST.csv"
    )
    write_csv(manifest_path, rows)
    write_text(build_root / "mechanism_ablation_prediction_matrices" / "README.md", README)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "matrix_rows": len(rows),
        "panel_counts": {
            panel: sum(row["panel"] == panel for row in rows) for panel in PANELS
        },
        "readout": "identity",
        "posthoc_calibration": "none",
        "manifest_path": archive_relative(manifest_path, build_root),
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json(
        build_root
        / "protocol_a_reproducibility"
        / "MECHANISM_EXPORT_SUMMARY.json",
        summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
