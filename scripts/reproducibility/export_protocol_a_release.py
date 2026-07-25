#!/usr/bin/env python3
"""Build a reviewer-facing compact Protocol A evidence bundle.

The exporter reads only the completed formal run and the frozen archive inputs.
It writes into a separate build directory; it never mutates either source tree.
Only final-test columns are retained in the compact prediction and truth arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


METHODS = (
    "GeneSPT",
    "Tangram",
    "TransImp",
    "SpaIM",
    "SpaGE",
    "stPlus",
    "stAI",
)
BASELINES = METHODS[1:]
SCHEDULER_BASELINES = BASELINES[:-1]
STAI_METHOD = "stAI"
FOLDS = (0, 1, 2, 3, 4)
CHUNK_BYTES = 8 * 1024 * 1024
STAI_OFFICIAL_COMMIT = "3376cc16cc6d8461edafc0aeb4519b92d18474b7"
STAI_FORMAL_ADAPTER_SHA256 = (
    "8f00df9d89a2d8fc9fa8e3201c6e4b71ff2d0521e48cd687c9e7718ae741dec4"
)


class ExportError(RuntimeError):
    """Raised when a formal artifact does not satisfy the release contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ExportError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ExportError(f"Refusing to write an empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows:
        if list(row) != fieldnames:
            raise ExportError(f"Inconsistent columns while writing {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def save_compressed(path: Path, key: str, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: values})
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def posix(path: Path) -> str:
    return path.as_posix()


def archive_relative(path: Path, root: Path) -> str:
    try:
        return posix(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ExportError(f"Path is outside the build root: {path}") from exc


def sanitize_string(value: str) -> str:
    replacements = (
        (
            "/workspace/zenodo_upload/GeneSPT_zenodo_unified_20260706/",
            "${ARCHIVE_ROOT}/",
        ),
        (
            "/workspace/GeneSPT/results/protocol_a_full_rerun_20260711/",
            "${FORMAL_RUN_ROOT}/",
        ),
        ("/workspace/GeneSPT_github_main_rebuild/", "${REPOSITORY_ROOT}/"),
        ("/workspace/GeneSPT/", "${REPOSITORY_ROOT}/"),
        ("/opt/conda/bin/python", "python"),
    )
    result = value.replace("\\", "/")
    for source, target in replacements:
        result = result.replace(source, target)
    if ":/" in result[:4] or result.startswith("/workspace/"):
        raise ExportError(f"Unresolved absolute path in public value: {value}")
    return result


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 6:
        raise ExportError("Protocol A config must define exactly six datasets")
    folds = tuple(int(value) for value in config.get("folds", []))
    if folds != FOLDS:
        raise ExportError(f"Unexpected fold contract: {folds}")
    return config


def metric_index(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    rows = read_csv(path)
    index: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset_id"], int(row["fold"]), row["method"])
        if key in index:
            raise ExportError(f"Duplicate formal metric row: {key}")
        index[key] = row
    expected = 6 * len(FOLDS) * len(METHODS)
    if len(index) != expected:
        raise ExportError(f"Expected {expected} formal metric rows, found {len(index)}")
    return index


def expected_source_hash(metric_row: Mapping[str, str]) -> str:
    value = metric_row.get("prediction_sha256", "")
    if len(value) != 64:
        raise ExportError("Formal metric row is missing prediction_sha256")
    return value


def compact_source_prediction(
    source: Path,
    method: str,
    test_idx: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    if method == "GeneSPT":
        with np.load(source, allow_pickle=False) as archive:
            prediction = np.asarray(archive["prediction"], dtype=np.float32)
            source_idx = np.asarray(archive["test_gene_idx"], dtype=np.int64)
        if not np.array_equal(source_idx, test_idx):
            raise ExportError(f"GeneSPT test index mismatch: {source}")
    else:
        matrix = np.load(source, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2:
            raise ExportError(f"Prediction is not two-dimensional: {source}")
        if matrix.shape[1] == n_genes:
            prediction = np.asarray(matrix[:, test_idx], dtype=np.float32)
        elif matrix.shape[1] == test_idx.size:
            prediction = np.asarray(matrix, dtype=np.float32)
        else:
            raise ExportError(
                f"Unexpected prediction shape {matrix.shape} for {source}; "
                f"expected {n_genes} or {test_idx.size} columns"
            )
    if prediction.ndim != 2 or prediction.shape[1] != test_idx.size:
        raise ExportError(f"Compact prediction shape mismatch: {source}")
    if not np.isfinite(prediction).all():
        raise ExportError(f"Non-finite prediction values: {source}")
    return prediction


def compact_truth(source: Path, test_idx: np.ndarray, n_genes: int) -> np.ndarray:
    matrix = np.load(source, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[1] != n_genes:
        raise ExportError(f"Unexpected formal truth shape {matrix.shape}: {source}")
    truth = np.asarray(matrix[:, test_idx], dtype=np.float32)
    if not np.isfinite(truth).all():
        raise ExportError(f"Non-finite truth values: {source}")
    return truth


def source_prediction_path(formal_root: Path, dataset_id: str, fold: int, method: str) -> Path:
    if method == "GeneSPT":
        return (
            formal_root
            / "evaluation"
            / "validation_selected_readout_genespt57"
            / "test_predictions"
            / dataset_id
            / f"fold{fold}"
            / "GeneSPT.npz"
        )
    return formal_root / "baselines" / method / dataset_id / f"fold{fold}" / "imputed_expression.npy"


def build_arrays(
    *,
    formal_root: Path,
    archive_root: Path,
    build_root: Path,
    config: Mapping[str, Any],
    metrics: Mapping[tuple[str, int, str], Mapping[str, str]],
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        dataset_name = str(dataset["name"])
        role = "Primary" if str(dataset["role"]) == "primary" else "Cross-platform"
        n_spots, n_genes = (int(value) for value in dataset["expected_st_shape"])
        gene_names_rel = str(dataset["gene_names"])
        gene_names_path = archive_root / gene_names_rel
        if not gene_names_path.is_file():
            raise ExportError(f"Missing gene names: {gene_names_path}")
        public_gene_names_path = build_root / gene_names_rel
        if not public_gene_names_path.is_file():
            copy_file(gene_names_path, public_gene_names_path)
        if sha256_file(public_gene_names_path) != sha256_file(gene_names_path):
            raise ExportError(f"Gene-name copy mismatch: {public_gene_names_path}")
        for fold in FOLDS:
            test_mask_rel = str(dataset["test_mask"]).format(fold=fold)
            test_mask_source = archive_root / test_mask_rel
            test_idx = np.asarray(np.load(test_mask_source, allow_pickle=False), dtype=np.int64)
            if test_idx.ndim != 1 or len(np.unique(test_idx)) != test_idx.size:
                raise ExportError(f"Invalid test index: {test_mask_source}")
            if test_idx.size == 0 or test_idx.min() < 0 or test_idx.max() >= n_genes:
                raise ExportError(f"Out-of-range test index: {test_mask_source}")

            truth_source = formal_root / "inputs" / dataset_id / f"fold{fold}" / "full_truth.npy"
            truth_dir = build_root / "ground_truth_protocol_a" / dataset_id / f"fold{fold}"
            truth_path = truth_dir / "truth.npz"
            truth_idx_path = truth_dir / "test_gene_idx.npy"
            truth_metadata_path = truth_dir / "metadata.json"
            truth_source_sha = sha256_file(truth_source)
            if not (resume and truth_path.is_file() and truth_idx_path.is_file()):
                truth = compact_truth(truth_source, test_idx, n_genes)
                if truth.shape != (n_spots, test_idx.size):
                    raise ExportError(f"Truth shape mismatch for {dataset_id} fold{fold}")
                save_compressed(truth_path, "truth", truth)
                save_npy(truth_idx_path, test_idx)
            with np.load(truth_path, allow_pickle=False) as packed_truth:
                truth_shape = tuple(int(value) for value in packed_truth["truth"].shape)
            if truth_shape != (n_spots, test_idx.size):
                raise ExportError(f"Packed truth shape mismatch: {truth_path}")
            truth_metadata = {
                "schema_version": 1,
                "protocol": "A",
                "dataset": dataset_name,
                "dataset_id": dataset_id,
                "role": role,
                "fold": fold,
                "matrix_scope": "final_test_genes",
                "shape": list(truth_shape),
                "dtype": "float32",
                "normalization": "log1p_cpm_with_inner_train_gene_library_size",
                "source_full_truth_sha256": truth_source_sha,
                "truth_sha256": sha256_file(truth_path),
                "test_gene_idx_sha256": sha256_file(truth_idx_path),
                "frozen_test_mask_source_sha256": sha256_file(test_mask_source),
                "gene_names_path": gene_names_rel,
                "spot_axis": "row order of processed Spatial_count.txt and Locations.txt",
            }
            write_json(truth_metadata_path, truth_metadata)
            truth_rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_id": dataset_id,
                    "role": role,
                    "fold": fold,
                    "truth_path": archive_relative(truth_path, build_root),
                    "metadata_path": archive_relative(truth_metadata_path, build_root),
                    "test_gene_idx_path": archive_relative(truth_idx_path, build_root),
                    "gene_names_path": gene_names_rel,
                    "shape": json.dumps(list(truth_shape), separators=(",", ":")),
                    "dtype": "float32",
                    "truth_sha256": sha256_file(truth_path),
                    "truth_bytes": truth_path.stat().st_size,
                    "test_gene_idx_sha256": sha256_file(truth_idx_path),
                    "source_full_truth_sha256": truth_source_sha,
                    "normalization": "log1p_cpm_with_inner_train_gene_library_size",
                }
            )

            for method in METHODS:
                metric_row = metrics[(dataset_id, fold, method)]
                source = source_prediction_path(formal_root, dataset_id, fold, method)
                source_sha = sha256_file(source)
                if source_sha != expected_source_hash(metric_row):
                    raise ExportError(
                        f"Formal source hash mismatch for {method} {dataset_id} fold{fold}"
                    )
                output_dir = build_root / "prediction_matrices" / dataset_id / method / f"fold{fold}"
                output_path = output_dir / "prediction.npz"
                output_idx_path = output_dir / "test_gene_idx.npy"
                metadata_path = output_dir / "metadata.json"
                if not (resume and output_path.is_file() and output_idx_path.is_file()):
                    prediction = compact_source_prediction(source, method, test_idx, n_genes)
                    if prediction.shape != (n_spots, test_idx.size):
                        raise ExportError(
                            f"Prediction shape mismatch for {method} {dataset_id} fold{fold}: "
                            f"{prediction.shape}"
                        )
                    save_compressed(output_path, "prediction", prediction)
                    save_npy(output_idx_path, test_idx)
                with np.load(output_path, allow_pickle=False) as packed:
                    output_shape = tuple(int(value) for value in packed["prediction"].shape)
                    if not np.isfinite(packed["prediction"]).all():
                        raise ExportError(f"Non-finite compact prediction: {output_path}")
                if output_shape != (n_spots, test_idx.size):
                    raise ExportError(f"Packed prediction shape mismatch: {output_path}")
                if not np.array_equal(np.load(output_idx_path, allow_pickle=False), test_idx):
                    raise ExportError(f"Packed test index mismatch: {output_idx_path}")
                result_layer = str(metric_row["result_layer"])
                metadata = {
                    "schema_version": 2,
                    "protocol": "A",
                    "dataset": dataset_name,
                    "dataset_id": dataset_id,
                    "role": role,
                    "fold": fold,
                    "method": method,
                    "result_layer": result_layer,
                    "matrix_scope": "final_test_genes",
                    "shape": list(output_shape),
                    "dtype": "float32",
                    "source_prediction_sha256": source_sha,
                    "compact_prediction_sha256": sha256_file(output_path),
                    "test_gene_idx_sha256": sha256_file(output_idx_path),
                    "truth_path": archive_relative(truth_path, build_root),
                    "truth_sha256": sha256_file(truth_path),
                    "gene_names_path": gene_names_rel,
                    "reported_fold_metrics": {
                        key: float(metric_row[key]) for key in ("SPCC", "RMSE", "JSD", "SSIM")
                    },
                    "coverage": float(metric_row["coverage"]),
                    "source_type": (
                        "validation_selected_readout" if method == "GeneSPT" else "raw_identity"
                    ),
                }
                write_json(metadata_path, metadata)
                prediction_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_id": dataset_id,
                        "role": role,
                        "method": method,
                        "fold": fold,
                        "result_layer": result_layer,
                        "matrix_path": archive_relative(output_path, build_root),
                        "metadata_path": archive_relative(metadata_path, build_root),
                        "test_gene_idx_path": archive_relative(output_idx_path, build_root),
                        "truth_path": archive_relative(truth_path, build_root),
                        "gene_names_path": gene_names_rel,
                        "matrix_scope": "final_test_genes",
                        "shape": json.dumps(list(output_shape), separators=(",", ":")),
                        "dtype": "float32",
                        "compact_prediction_sha256": sha256_file(output_path),
                        "compact_prediction_bytes": output_path.stat().st_size,
                        "source_prediction_sha256": source_sha,
                        "test_gene_idx_sha256": sha256_file(output_idx_path),
                        "truth_sha256": sha256_file(truth_path),
                        "reported_SPCC": metric_row["SPCC"],
                        "reported_RMSE": metric_row["RMSE"],
                        "reported_JSD": metric_row["JSD"],
                        "reported_SSIM": metric_row["SSIM"],
                        "coverage": metric_row["coverage"],
                    }
                )
                print(f"[packed] {method} {dataset_id} fold{fold}", flush=True)
    return prediction_rows, truth_rows


def fallback_fields(audit: Mapping[str, Any]) -> tuple[bool, str]:
    used = bool(
        audit.get("fallback_used", False)
        or audit.get("hidden_gene_zero_fallback_used", False)
        or audit.get("truth_copy_fallback_used", False)
    )
    details: list[str] = []
    if isinstance(audit.get("fallback"), dict):
        details.append(json.dumps(audit["fallback"], sort_keys=True, separators=(",", ":")))
    if audit.get("fallback_policy") is not None:
        details.append(str(audit["fallback_policy"]))
    if audit.get("native_filtering_disabled") is not None:
        details.append(f"native_filtering_disabled={bool(audit['native_filtering_disabled'])}")
    return used, "; ".join(details)


def build_run_manifests(
    *,
    formal_root: Path,
    archive_root: Path,
    build_root: Path,
    config: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_index = {
        (str(row["method"]), str(row["dataset_id"]), int(row["fold"])): row
        for row in prediction_rows
    }
    role_index = {
        str(item["dataset_id"]): (
            "Primary" if str(item["role"]) == "primary" else "Cross-platform"
        )
        for item in config["datasets"]
    }
    dataset_name_index = {str(item["dataset_id"]): str(item["name"]) for item in config["datasets"]}
    baseline_rows: list[dict[str, Any]] = []
    for status_path in sorted((formal_root / "baselines" / "_scheduler" / "status").glob("*.json")):
        status = load_json(status_path)
        method = str(status["method"])
        dataset_id = str(status["dataset_id"])
        fold = int(status["fold"])
        if method not in SCHEDULER_BASELINES or fold not in FOLDS:
            raise ExportError(f"Unexpected baseline task: {status_path}")
        if status.get("status") != "completed" or int(status.get("returncode", -1)) != 0:
            raise ExportError(f"Incomplete baseline task: {status_path}")
        audit_source = formal_root / "baselines" / method / dataset_id / f"fold{fold}" / "adapter_run_audit.json"
        audit = load_json(audit_source)
        fallback_used, fallback_policy = fallback_fields(audit)
        compact = prediction_index[(method, dataset_id, fold)]
        baseline_rows.append(
            {
                "task": f"{method}__{dataset_id}__fold{fold}",
                "method": method,
                "dataset": dataset_name_index[dataset_id],
                "dataset_id": dataset_id,
                "role": role_index[dataset_id],
                "fold": fold,
                "protocol": "A",
                "status": "completed",
                "returncode": 0,
                "command": json.dumps(sanitize(status["command"]), separators=(",", ":")),
                "adapter_sha256": status["adapter_sha256"],
                "config_sha256": status["config_sha256"],
                "input_sha256": status["input_sha256"],
                "frozen_split_sha256": status["input_file_sha256"]["frozen_split"],
                "train_mask_sha256": status["input_file_sha256"]["train_mask"],
                "val_mask_sha256": status["input_file_sha256"]["val_mask"],
                "test_mask_sha256": status["input_file_sha256"]["test_mask"],
                "source_prediction_sha256": status["prediction_sha256"],
                "compact_prediction_path": compact["matrix_path"],
                "compact_prediction_sha256": compact["compact_prediction_sha256"],
                "shape": compact["shape"],
                "finite": True,
                "complete_test_coverage": bool(
                    audit.get("test_coverage_complete", audit.get("complete_test_gene_coverage", True))
                ),
                "fallback_used": fallback_used,
                "fallback_policy": fallback_policy,
                "adapter_audit_source_sha256": sha256_file(audit_source),
            }
        )
    stai_root = formal_root / "baselines" / STAI_METHOD
    adoption_path = stai_root / "formal_adoption_manifest.json"
    adoption = load_json(adoption_path)
    if (
        adoption.get("status") != "complete"
        or adoption.get("method") != STAI_METHOD
        or adoption.get("official_stai_commit") != STAI_OFFICIAL_COMMIT
    ):
        raise ExportError(f"Invalid formal stAI adoption manifest: {adoption_path}")
    adoption_records = {
        (str(row["dataset_id"]), int(row["fold"])): row
        for row in adoption.get("folds", [])
    }
    if len(adoption_records) != 6 * len(FOLDS):
        raise ExportError("Formal stAI adoption manifest must contain 30 folds")

    version_config = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "protocol_a_baseline_versions.yaml"
    )
    for dataset in config["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        dataset_name = str(dataset["name"])
        role = role_index[dataset_id]
        for fold in FOLDS:
            audit_source = (
                stai_root / dataset_id / f"fold{fold}" / "adapter_run_audit.json"
            )
            audit = load_json(audit_source)
            record = adoption_records[(dataset_id, fold)]
            if (
                audit.get("status") != "complete"
                or audit.get("method") != STAI_METHOD
                or audit.get("official_stai_commit") != STAI_OFFICIAL_COMMIT
                or bool(audit.get("test_st_truth_accessed_during_model_stage"))
                or bool(audit.get("validation_st_expression_used_during_model_stage"))
                or bool(audit.get("pseudo_labels_used"))
            ):
                raise ExportError(f"Invalid formal stAI audit: {audit_source}")
            compact = prediction_index[(STAI_METHOD, dataset_id, fold)]
            if (
                str(record["prediction_sha256"])
                != str(audit["output_matrix_sha256"])
                or str(compact["source_prediction_sha256"])
                != str(audit["output_matrix_sha256"])
            ):
                raise ExportError(
                    f"stAI prediction provenance mismatch: {dataset_id} fold{fold}"
                )
            split_path = (
                formal_root / "inputs" / dataset_id / f"fold{fold}" / "mode_a_split.json"
            )
            train_mask = archive_root / str(dataset["train_mask"]).format(fold=fold)
            val_mask = archive_root / str(dataset["val_mask"]).format(fold=fold)
            test_mask = archive_root / str(dataset["test_mask"]).format(fold=fold)
            output_dir = f"${{FORMAL_RUN_ROOT}}/baselines/stAI/{dataset_id}/fold{fold}"
            command = [
                [
                    "python",
                    "${REPOSITORY_ROOT}/baseline_adapters/stai/run_stai_protocol_a.py",
                    stage,
                    "--dataset",
                    dataset_name,
                    "--fold",
                    str(fold),
                    "--output-dir",
                    output_dir,
                ]
                for stage in ("prepare", "run", "evaluate")
            ]
            baseline_rows.append(
                {
                    "task": f"{STAI_METHOD}__{dataset_id}__fold{fold}",
                    "method": STAI_METHOD,
                    "dataset": dataset_name,
                    "dataset_id": dataset_id,
                    "role": role,
                    "fold": fold,
                    "protocol": "A",
                    "status": "completed",
                    "returncode": 0,
                    "command": json.dumps(command, separators=(",", ":")),
                    "adapter_sha256": STAI_FORMAL_ADAPTER_SHA256,
                    "config_sha256": sha256_file(version_config),
                    "input_sha256": sha256_file(split_path),
                    "frozen_split_sha256": sha256_file(split_path),
                    "train_mask_sha256": sha256_file(train_mask),
                    "val_mask_sha256": sha256_file(val_mask),
                    "test_mask_sha256": sha256_file(test_mask),
                    "source_prediction_sha256": audit["output_matrix_sha256"],
                    "compact_prediction_path": compact["matrix_path"],
                    "compact_prediction_sha256": compact["compact_prediction_sha256"],
                    "shape": compact["shape"],
                    "finite": True,
                    "complete_test_coverage": float(audit.get("coverage", 0.0)) == 1.0,
                    "fallback_used": False,
                    "fallback_policy": "none; pseudo-labels disabled",
                    "adapter_audit_source_sha256": sha256_file(audit_source),
                }
            )

    expected = len(BASELINES) * 6 * len(FOLDS)
    if len(baseline_rows) != expected:
        raise ExportError(f"Expected {expected} baseline status rows, found {len(baseline_rows)}")

    input_rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        for fold in FOLDS:
            manifest_path = formal_root / "inputs" / dataset_id / f"fold{fold}" / "artifact_manifest.json"
            manifest = load_json(manifest_path)
            artifacts = manifest["input_artifacts"]
            outputs = manifest["output_artifacts"]
            train_idx = np.load(archive_root / str(dataset["train_mask"]).format(fold=fold), allow_pickle=False)
            val_idx = np.load(archive_root / str(dataset["val_mask"]).format(fold=fold), allow_pickle=False)
            test_idx = np.load(archive_root / str(dataset["test_mask"]).format(fold=fold), allow_pickle=False)
            input_rows.append(
                {
                    "dataset": str(dataset["name"]),
                    "dataset_id": dataset_id,
                    "role": "Primary" if str(dataset["role"]) == "primary" else "Cross-platform",
                    "fold": fold,
                    "train_gene_count": int(train_idx.size),
                    "validation_gene_count": int(val_idx.size),
                    "test_gene_count": int(test_idx.size),
                    "train_mask_path": str(artifacts["train_mask"]["path"]),
                    "train_mask_sha256": str(artifacts["train_mask"]["sha256"]),
                    "val_mask_path": str(artifacts["val_mask"]["path"]),
                    "val_mask_sha256": str(artifacts["val_mask"]["sha256"]),
                    "test_mask_path": str(artifacts["test_mask"]["path"]),
                    "test_mask_sha256": str(artifacts["test_mask"]["sha256"]),
                    "raw_counts_sha256": str(artifacts["raw_counts"]["sha256"]),
                    "scrna_counts_sha256": str(artifacts["scrna_counts"]["sha256"]),
                    "locations_sha256": str(artifacts["locations"]["sha256"]),
                    "mode_a_split_sha256": str(outputs["mode_a_split"]["sha256"]),
                    "full_truth_sha256": str(outputs["full_truth"]["sha256"]),
                    "normalization_audit_sha256": str(outputs["normalization_audit"]["sha256"]),
                    "normalization_policy": "inner_train_gene_library_size_applied_to_all_columns",
                    "artifact_manifest_sha256": sha256_file(manifest_path),
                }
            )
    if len(input_rows) != 6 * len(FOLDS):
        raise ExportError("Input manifest does not cover all 30 dataset-fold tasks")
    return baseline_rows, input_rows


def export_readout_selections(formal_root: Path, build_root: Path) -> list[dict[str, Any]]:
    source_root = formal_root / "evaluation" / "validation_selected_readout_genespt57" / "selections"
    destination_root = build_root / "protocol_a_reproducibility" / "readout_selections"
    rows: list[dict[str, Any]] = []
    for source_lock in sorted(source_root.glob("*/fold*/selection_lock.json")):
        lock = load_json(source_lock)
        dataset_id = str(lock["identity"]["dataset_id"])
        fold = int(lock["identity"]["fold"])
        destination = destination_root / dataset_id / f"fold{fold}"
        public_lock = sanitize(lock)
        public_lock["source_selection_lock_sha256"] = sha256_file(source_lock)
        public_lock_path = destination / "selection_lock_public.json"
        write_json(public_lock_path, public_lock)
        contract_source = source_lock.parent.parent.parent / "selection_inputs" / dataset_id / f"fold{fold}" / "selection_contract.json"
        if not contract_source.is_file():
            contract_source = (
                formal_root
                / "evaluation"
                / "validation_selected_readout_genespt57"
                / "selection_inputs"
                / dataset_id
                / f"fold{fold}"
                / "selection_contract.json"
            )
        public_contract_path = destination / "selection_contract_public.json"
        write_json(public_contract_path, sanitize(load_json(contract_source)))
        for method in ("GeneSPT", "GeneSPT-GC"):
            method_source = source_lock.parent / method
            method_destination = destination / method
            for filename in (
                "validation_candidates.csv",
                "validation_metric_audit.csv",
                "selected.json",
                "train_oracle_choices.csv",
            ):
                copy_file(method_source / filename, method_destination / filename)
            candidate_rows = read_csv(method_source / "validation_candidates.csv")
            if len(candidate_rows) != 57:
                raise ExportError(
                    f"Expected 57 readout candidates for {method} {dataset_id} fold{fold}"
                )
            selected = lock["selected"][method]
            rows.append(
                {
                    "dataset": str(lock["identity"]["dataset"]),
                    "dataset_id": dataset_id,
                    "role": str(lock["identity"]["role"]),
                    "fold": fold,
                    "method": method,
                    "candidate_count": len(candidate_rows),
                    "selected_calibration": str(selected["calibration"]),
                    "feature_kind": str(selected.get("feature_kind", "")),
                    "model_kind": str(selected.get("model_kind", "")),
                    "target_mode": str(selected.get("target", "")),
                    "alpha": selected.get("alpha"),
                    "selection_rule": str(lock["protocol_definition"]["selection"]),
                    "guard_pass": bool(selected["guard_pass"]),
                    "seed": int(lock["protocol_definition"]["seed"]),
                    "test_prediction_accessed_before_lock": bool(
                        lock["test_prediction_accessed_before_lock"]
                    ),
                    "test_truth_accessed_before_lock": bool(lock["test_truth_accessed_before_lock"]),
                    "source_selection_lock_sha256": sha256_file(source_lock),
                    "public_selection_lock_path": archive_relative(public_lock_path, build_root),
                    "public_selection_lock_sha256": sha256_file(public_lock_path),
                    "candidate_table_path": archive_relative(
                        method_destination / "validation_candidates.csv", build_root
                    ),
                }
            )
    if len(rows) != 6 * len(FOLDS) * 2:
        raise ExportError(f"Expected 60 readout rows, found {len(rows)}")
    return rows


def checksum_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "BUNDLE_FILE_MANIFEST_SHA256.csv":
            continue
        rows.append(
            {
                "path": archive_relative(path, root),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-run-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    formal_root = args.formal_run_root.resolve(strict=True)
    archive_root = args.archive_root.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    build_root = args.output_root.resolve()
    if build_root.exists() and any(build_root.iterdir()) and not args.resume:
        raise ExportError(f"Output directory is not empty; use --resume: {build_root}")
    build_root.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    formal_metrics_path = (
        formal_root / "evaluation" / "formal_benchmark_evidence" / "formal_fold_metrics.csv"
    )
    metrics = metric_index(formal_metrics_path)
    prediction_rows, truth_rows = build_arrays(
        formal_root=formal_root,
        archive_root=archive_root,
        build_root=build_root,
        config=config,
        metrics=metrics,
        resume=args.resume,
    )
    baseline_rows, input_rows = build_run_manifests(
        formal_root=formal_root,
        archive_root=archive_root,
        build_root=build_root,
        config=config,
        prediction_rows=prediction_rows,
    )
    readout_rows = export_readout_selections(formal_root, build_root)

    matrix_manifest_root = build_root / "prediction_matrix_manifests"
    provenance_root = build_root / "protocol_a_reproducibility" / "manifests"
    write_csv(matrix_manifest_root / "PREDICTION_MATRIX_MANIFEST.csv", prediction_rows)
    write_csv(matrix_manifest_root / "PROTOCOL_A_TRUTH_MATRIX_MANIFEST.csv", truth_rows)
    write_csv(provenance_root / "FORMAL_BASELINE_RUN_MANIFEST.csv", baseline_rows)
    write_csv(provenance_root / "INPUT_SPLIT_MANIFEST.csv", input_rows)
    write_csv(provenance_root / "READOUT_SELECTION_MANIFEST.csv", readout_rows)

    source_data_root = build_root / "protocol_a_reproducibility" / "source_data"
    for filename in (
        "formal_five_fold_metrics.csv",
        "formal_paired_bootstrap.csv",
    ):
        copy_file(
            formal_root / "evaluation" / "formal_benchmark_evidence" / filename,
            source_data_root / filename,
        )
    public_fold_rows = [sanitize(row) for row in read_csv(formal_metrics_path)]
    write_csv(source_data_root / "formal_fold_metrics.csv", public_fold_rows)
    public_evidence_manifest = load_json(
        formal_root
        / "evaluation"
        / "formal_benchmark_evidence"
        / "formal_evidence_manifest.json"
    )
    # The quarantine-side metrics file was used only to cross-check adoption.
    # The public package anchors stAI to the formal adoption manifest instead.
    public_evidence_manifest.get("inputs", {}).pop("stai_reference", None)
    public_evidence_manifest = sanitize(public_evidence_manifest)
    write_json(source_data_root / "formal_evidence_manifest.json", public_evidence_manifest)
    copy_file(
        formal_root / "evaluation" / "formal_benchmark_evidence" / "formal_gene_level_metrics.csv",
        source_data_root / "formal_gene_level_metrics.csv",
    )
    write_csv(
        build_root / "protocol_a_reproducibility" / "BUNDLE_FILE_MANIFEST_SHA256.csv",
        checksum_manifest(build_root),
    )
    summary = {
        "schema_version": 1,
        "protocol": "A",
        "datasets": 6,
        "folds_per_dataset": 5,
        "benchmark_methods": list(METHODS),
        "prediction_matrix_count": len(prediction_rows),
        "truth_matrix_count": len(truth_rows),
        "baseline_task_count": len(baseline_rows),
        "readout_selection_row_count": len(readout_rows),
        "formal_metrics_sha256": sha256_file(formal_metrics_path),
        "formal_config_sha256": sha256_file(config_path),
        "status": "complete",
    }
    write_json(build_root / "protocol_a_reproducibility" / "BUILD_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        raise SystemExit(f"ERROR: {exc}")
