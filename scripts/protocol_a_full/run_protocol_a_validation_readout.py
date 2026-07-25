#!/usr/bin/env python3
"""Auditable four-stage Protocol A GeneSPT validation readout.

The stages are intentionally separate processes:

* prepare: copy only train/validation truth and predictions into isolated files;
* select: choose from the frozen 57-candidate family and write a lock;
* apply: verify the lock/hash chain, then read test predictions (never truth);
* evaluate: read test truth for the first time and call centralized metrics.py.

External baselines remain raw identity outputs and are never passed through this
model-specific readout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import validation_readout_core as core


PROJECT_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "protocol_a_datasets.yaml"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "protocol_a_full_rerun_20260711"
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_RESULTS_ROOT
    / "evaluation"
    / "validation_selected_readout_genespt57"
)
DEFAULT_METRICS = (
    WORKSPACE_ROOT / "GeneSPT_github_main_rebuild" / "src" / "genespt" / "metrics.py"
)
DEFAULT_RAW_REPORT = (
    DEFAULT_RESULTS_ROOT / "evaluation" / "protocol_a_raw_evaluation_report.json"
)

METHODS = ("GeneSPT", "GeneSPT-GC")
EXTERNAL_BASELINES = ("Tangram", "TransImp", "SpaIM", "SpaGE", "stPlus")
FOLDS = (0, 1, 2, 3, 4)
SEED = 42

SOURCE_KEYS = {
    "prediction", "base_prediction", "train_gene_idx", "val_gene_idx",
    "test_gene_idx", "test_genes", "model", "fold", "base_descriptor",
    "psp_descriptor", "component_keep", "lambda_low", "lambda_mid",
    "lambda_high", "spatiality_q1", "spatiality_q2", "posthoc_calibration",
    "readout", "base_prediction_val", "base_prediction_test",
    "base_prediction_train", "selected_prediction_val",
    "selected_prediction_test", "selected_rule_frozen_from_split",
    "selected_prediction_train", "selected_train_coefficient_source",
}
SOURCE_REQUIRED = {
    "train_gene_idx", "val_gene_idx", "test_gene_idx",
    "base_prediction_train", "base_prediction_val", "base_prediction_test",
    "selected_prediction_train", "selected_prediction_val",
    "selected_prediction_test",
}
COMMON_KEYS = {
    "train_truth", "val_truth", "train_idx", "val_idx", "coordinates",
    "pca32", "nmf32", "pca32_nmf32",
}
METHOD_INPUT_KEYS = {"pred_train", "pred_val"}
SELECTED_TEST_KEYS = {
    "prediction", "test_gene_idx", "method", "dataset_id", "fold",
    "selected_calibration", "selection_lock_sha256",
}


class ProtocolError(RuntimeError):
    """Fail-closed Protocol A readout error."""


@dataclass(frozen=True)
class Task:
    dataset: str
    dataset_id: str
    role: str
    fold: int
    n_spots: int
    n_genes: int
    config_path: Path
    archive_root: Path
    input_dir: Path
    completion_path: Path
    gene_names_path: Path
    locations_path: Path
    train_mask_path: Path
    val_mask_path: Path
    test_mask_path: Path
    prediction_path: Path
    prediction_sha256: str
    descriptor_path: Path
    descriptor_sha256: str
    gene_axis_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\n")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def raw_array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProtocolError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(core.json_ready(value), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def source_record(path: Path, *, include_sha: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    record: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
    }
    try:
        record["project_relative"] = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        pass
    try:
        record["workspace_relative"] = resolved.relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        pass
    if include_sha:
        record["sha256"] = sha256_file(resolved)
    return record


def resolve_record(record: Mapping[str, Any]) -> Path:
    candidates = [Path(str(record.get("path", "")))]
    if record.get("project_relative"):
        candidates.append(PROJECT_ROOT / str(record["project_relative"]))
    if record.get("workspace_relative"):
        candidates.append(WORKSPACE_ROOT / str(record["workspace_relative"]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ProtocolError(f"Recorded source is unavailable: {record}")


def verify_record(record: Mapping[str, Any], *, hash_file: bool = True) -> Path:
    path = resolve_record(record)
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise ProtocolError(f"File size changed: {path}")
    if hash_file and str(record.get("sha256", "")) != sha256_file(path):
        raise ProtocolError(f"File hash changed: {path}")
    return path


def audit_npz(path: Path, *, allowed: set[str], required: set[str]) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except Exception as exc:
        raise ProtocolError(f"Invalid NPZ archive {path}: {exc}") from exc
    if len(names) != len(set(names)):
        raise ProtocolError(f"Duplicate NPZ members: {path}")
    keys: list[str] = []
    for name in names:
        member = Path(name)
        if member.name != name or member.suffix != ".npy":
            raise ProtocolError(f"Non-whitelisted NPZ member {name!r}: {path}")
        keys.append(member.stem)
    observed = set(keys)
    unexpected = observed - allowed
    missing = required - observed
    if unexpected or missing:
        raise ProtocolError(
            f"NPZ key contract failed for {path}; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return tuple(keys)


def load_npz(path: Path, keys: Iterable[str], *, allowed: set[str]) -> dict[str, np.ndarray]:
    required = set(keys)
    audit_npz(path, allowed=allowed, required=required)
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in required}


def translate_manifest_path(value: str) -> Path:
    normalized = value.replace("\\", "/")
    if normalized.startswith("/workspace/GeneSPT/"):
        return PROJECT_ROOT / normalized.removeprefix("/workspace/GeneSPT/")
    if normalized == "/workspace/GeneSPT":
        return PROJECT_ROOT
    if normalized.startswith("/workspace/"):
        return WORKSPACE_ROOT / normalized.removeprefix("/workspace/")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _command_argument(command: Sequence[Any], flag: str) -> str:
    values = [str(value) for value in command]
    if values.count(flag) != 1:
        raise ProtocolError(f"Completion command must contain one {flag}")
    position = values.index(flag)
    if position + 1 >= len(values):
        raise ProtocolError(f"Completion command has no value after {flag}")
    return values[position + 1]


def _prediction_file_sha(completion: Mapping[str, Any], path: Path) -> str:
    records = completion.get("outputs", {}).get("files", [])
    matches = []
    for record in records:
        if not isinstance(record, dict) or not record.get("path"):
            continue
        if translate_manifest_path(str(record["path"])).resolve() == path.resolve():
            matches.append(str(record.get("sha256", "")))
    if len(matches) != 1 or len(matches[0]) != 64:
        raise ProtocolError(f"Prediction file is not uniquely hashed in completion manifest: {path}")
    return matches[0]


def discover_tasks(
    *,
    config_path: Path = DEFAULT_CONFIG,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    datasets: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
) -> list[Task]:
    config = read_json(config_path)
    archive_root = (config_path.resolve().parent.parent / str(config["archive"]["root"])).resolve()
    configured = config.get("datasets")
    if not isinstance(configured, list) or len(configured) != 6:
        raise ProtocolError("Protocol A config must contain the six manuscript datasets")
    selectors = {str(value).casefold() for value in datasets or ()}
    selected_folds = tuple(int(value) for value in (folds or FOLDS))
    if any(value not in FOLDS for value in selected_folds):
        raise ProtocolError("Fold selectors must be in 0..4")
    tasks: list[Task] = []
    matched: set[str] = set()
    for item in configured:
        name = str(item["name"])
        dataset_id = str(item["dataset_id"])
        if selectors and name.casefold() not in selectors and dataset_id.casefold() not in selectors:
            continue
        matched.update({name.casefold(), dataset_id.casefold()} & selectors)
        n_spots, n_genes = (int(value) for value in item["expected_st_shape"])
        for fold in selected_folds:
            completion_path = (
                results_root / "genespt" / "benchmark" / dataset_id
                / f"fold{fold}" / "completion_manifest.json"
            )
            completion = read_json(completion_path)
            if completion.get("status") != "complete" or completion.get("protocol") != "A":
                raise ProtocolError(f"Incomplete Protocol A completion manifest: {completion_path}")
            if completion.get("dataset_id") != dataset_id or int(completion.get("fold", -1)) != fold:
                raise ProtocolError(f"Completion identity mismatch: {completion_path}")
            payload = completion["outputs"]["prediction_payloads"]["canonical_full_gc_payload"]
            if payload.get("train_validation_test_complete") is not True:
                raise ProtocolError(f"Incomplete GeneSPT/GC payload: {completion_path}")
            prediction_path = translate_manifest_path(str(payload["path"])).resolve()
            descriptor = completion["provenance"]["descriptor"]
            descriptor_record = descriptor["descriptor_file"]
            descriptor_path = translate_manifest_path(str(descriptor_record["path"])).resolve()
            command_descriptor = translate_manifest_path(
                _command_argument(completion["command"], "--descriptor-cache")
            ).resolve()
            if command_descriptor != descriptor_path:
                raise ProtocolError(f"Descriptor command/provenance mismatch: {completion_path}")
            tasks.append(
                Task(
                    dataset=name,
                    dataset_id=dataset_id,
                    role=str(item["role"]),
                    fold=fold,
                    n_spots=n_spots,
                    n_genes=n_genes,
                    config_path=config_path.resolve(),
                    archive_root=archive_root,
                    input_dir=results_root / "inputs" / dataset_id / f"fold{fold}",
                    completion_path=completion_path,
                    gene_names_path=archive_root / str(item["gene_names"]),
                    locations_path=archive_root / str(item["locations"]),
                    train_mask_path=archive_root / str(item["train_mask"]).format(fold=fold),
                    val_mask_path=archive_root / str(item["val_mask"]).format(fold=fold),
                    test_mask_path=archive_root / str(item["test_mask"]).format(fold=fold),
                    prediction_path=prediction_path,
                    prediction_sha256=_prediction_file_sha(completion, prediction_path),
                    descriptor_path=descriptor_path,
                    descriptor_sha256=str(descriptor_record["sha256"]),
                    gene_axis_sha256=str(descriptor["gene_axis_sha256"]),
                )
            )
    if selectors and len(matched) != len(selectors):
        raise ProtocolError(f"Unknown dataset selector(s): {sorted(selectors - matched)}")
    return tasks


def preflight_task(task: Task) -> dict[str, Any]:
    required = [
        task.config_path, task.input_dir / "full_truth.npy",
        task.input_dir / "mode_a_split.json", task.input_dir / "artifact_manifest.json",
        task.completion_path, task.gene_names_path, task.locations_path,
        task.train_mask_path, task.val_mask_path, task.test_mask_path,
        task.prediction_path, task.descriptor_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProtocolError(f"Missing task artifacts for {task.dataset_id} fold{task.fold}: {missing}")
    prediction_keys = audit_npz(
        task.prediction_path, allowed=SOURCE_KEYS, required=SOURCE_REQUIRED
    )
    descriptor_keys = audit_npz(
        task.descriptor_path,
        allowed={"pca32", "nmf32", "pca32_nmf32"},
        required={"pca32", "nmf32", "pca32_nmf32"},
    )
    return {
        "dataset": task.dataset,
        "dataset_id": task.dataset_id,
        "fold": task.fold,
        "status": "ready",
        "prediction_key_count": len(prediction_keys),
        "descriptor_keys": list(descriptor_keys),
    }


def _genes(path: Path, count: int) -> list[str]:
    genes = path.read_text(encoding="utf-8-sig").splitlines()
    if len(genes) != count or len(set(genes)) != count or any(not value for value in genes):
        raise ProtocolError(f"Invalid gene axis: {path}")
    return genes


def _mask(path: Path, count: int) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ProtocolError(f"Invalid mask: {path}")
    result = values.astype(np.int64, copy=False)
    if len(result) == 0 or len(np.unique(result)) != len(result):
        raise ProtocolError(f"Invalid mask: {path}")
    if int(result.min()) < 0 or int(result.max()) >= count:
        raise ProtocolError(f"Out-of-range mask: {path}")
    return result


def task_subdir(root: Path, task_or_identity: Task | Mapping[str, Any]) -> Path:
    dataset_id = (
        task_or_identity.dataset_id
        if isinstance(task_or_identity, Task)
        else str(task_or_identity["dataset_id"])
    )
    fold = (
        task_or_identity.fold
        if isinstance(task_or_identity, Task)
        else int(task_or_identity["fold"])
    )
    return root / dataset_id / f"fold{fold}"


def prepare_task(task: Task, prepared_dir: Path) -> dict[str, Any]:
    preflight_task(task)
    artifact_path = task.input_dir / "artifact_manifest.json"
    split_path = task.input_dir / "mode_a_split.json"
    truth_path = task.input_dir / "full_truth.npy"
    artifact = read_json(artifact_path)
    split = read_json(split_path)
    completion = read_json(task.completion_path)

    sources = {
        "config": source_record(task.config_path),
        "completion_manifest": source_record(task.completion_path),
        "artifact_manifest": source_record(artifact_path),
        "mode_a_split": source_record(split_path),
        "full_truth": source_record(truth_path),
        "gene_names": source_record(task.gene_names_path),
        "locations": source_record(task.locations_path),
        "train_mask": source_record(task.train_mask_path),
        "val_mask": source_record(task.val_mask_path),
        "test_mask": source_record(task.test_mask_path),
        "descriptor": source_record(task.descriptor_path),
        "raw_prediction": source_record(task.prediction_path),
    }
    if sources["raw_prediction"]["sha256"] != task.prediction_sha256:
        raise ProtocolError(f"Prediction SHA differs from completion manifest: {task.prediction_path}")
    if sources["descriptor"]["sha256"] != task.descriptor_sha256:
        raise ProtocolError(f"Descriptor SHA differs from completion manifest: {task.descriptor_path}")
    expected_artifact = artifact.get("output_artifacts", {}).get("full_truth", {}).get("sha256")
    if expected_artifact != sources["full_truth"]["sha256"]:
        raise ProtocolError(f"Full truth SHA differs from artifact manifest: {truth_path}")

    genes = _genes(task.gene_names_path, task.n_genes)
    gene_axis_sha = core.canonical_hash(genes)
    if gene_axis_sha != task.gene_axis_sha256 or split.get("gene_axis_sha256") != gene_axis_sha:
        raise ProtocolError(f"Gene-axis provenance mismatch: {task.dataset_id} fold{task.fold}")
    train_idx = _mask(task.train_mask_path, task.n_genes)
    val_idx = _mask(task.val_mask_path, task.n_genes)
    test_idx = _mask(task.test_mask_path, task.n_genes)
    partition = np.concatenate([train_idx, val_idx, test_idx])
    if len(partition) != task.n_genes or len(np.unique(partition)) != task.n_genes:
        raise ProtocolError(f"Masks do not partition gene axis: {task.dataset_id} fold{task.fold}")
    for key, expected in (
        ("inner_train_gene_idx", train_idx),
        ("inner_validation_gene_idx", val_idx),
        ("final_test_gene_idx", test_idx),
    ):
        if not np.array_equal(np.asarray(split.get(key), dtype=np.int64), expected):
            raise ProtocolError(f"Mode-A split mismatch for {key}: {task.dataset_id} fold{task.fold}")

    truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
    if truth.shape != (task.n_spots, task.n_genes):
        raise ProtocolError(f"Truth shape mismatch: {truth_path}")
    train_truth = np.asarray(truth[:, train_idx], dtype=np.float32).copy()
    val_truth = np.asarray(truth[:, val_idx], dtype=np.float32).copy()
    del truth
    if not np.isfinite(train_truth).all() or not np.isfinite(val_truth).all():
        raise ProtocolError("Prepared train/validation truth contains nonfinite values")

    coordinates = pd.read_csv(task.locations_path, sep="\t").apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float32)
    if coordinates.shape[0] != task.n_spots or coordinates.shape[1] < 2:
        raise ProtocolError(f"Coordinate shape mismatch: {task.locations_path}")
    descriptors = load_npz(
        task.descriptor_path,
        ("pca32", "nmf32", "pca32_nmf32"),
        allowed={"pca32", "nmf32", "pca32_nmf32"},
    )
    descriptor_provenance = completion["provenance"]["descriptor"]["arrays"]
    for key in ("pca32", "nmf32", "pca32_nmf32"):
        value = np.asarray(descriptors[key], dtype=np.float32)
        if value.shape[0] != task.n_genes or not np.isfinite(value).all():
            raise ProtocolError(f"Invalid descriptor {key}: {task.descriptor_path}")
        if raw_array_sha256(value) != descriptor_provenance[key]["payload_sha256"]:
            raise ProtocolError(f"Descriptor payload SHA mismatch for {key}")
        descriptors[key] = value

    source_arrays = load_npz(
        task.prediction_path,
        (
            "train_gene_idx", "val_gene_idx",
            "selected_prediction_train", "selected_prediction_val",
            "base_prediction_train", "base_prediction_val",
        ),
        allowed=SOURCE_KEYS,
    )
    if not np.array_equal(source_arrays["train_gene_idx"].astype(np.int64), train_idx):
        raise ProtocolError("Prediction train indices differ from frozen split")
    if not np.array_equal(source_arrays["val_gene_idx"].astype(np.int64), val_idx):
        raise ProtocolError("Prediction validation indices differ from frozen split")

    prepared_dir.mkdir(parents=True, exist_ok=True)
    common_path = prepared_dir / "common.npz"
    save_npz(
        common_path,
        train_truth=train_truth,
        val_truth=val_truth,
        train_idx=train_idx,
        val_idx=val_idx,
        coordinates=coordinates,
        pca32=descriptors["pca32"],
        nmf32=descriptors["nmf32"],
        pca32_nmf32=descriptors["pca32_nmf32"],
    )
    method_paths: dict[str, Path] = {}
    for method, prefix in (("GeneSPT", "selected"), ("GeneSPT-GC", "base")):
        pred_train = np.asarray(source_arrays[f"{prefix}_prediction_train"], dtype=np.float32)
        pred_val = np.asarray(source_arrays[f"{prefix}_prediction_val"], dtype=np.float32)
        if pred_train.shape != train_truth.shape or pred_val.shape != val_truth.shape:
            raise ProtocolError(f"Prepared prediction shape mismatch for {method}")
        if not np.isfinite(pred_train).all() or not np.isfinite(pred_val).all():
            raise ProtocolError(f"Prepared prediction is nonfinite for {method}")
        path = prepared_dir / f"{method}.npz"
        save_npz(path, pred_train=pred_train, pred_val=pred_val)
        method_paths[method] = path

    outputs = {
        "common": source_record(common_path),
        "methods": {method: source_record(path) for method, path in method_paths.items()},
    }
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "identity": {
            "dataset": task.dataset, "dataset_id": task.dataset_id,
            "role": task.role, "fold": task.fold,
            "n_spots": task.n_spots, "n_genes": task.n_genes,
        },
        "gene_axis_sha256": gene_axis_sha,
        "split_sha256": {
            "train": sources["train_mask"]["sha256"],
            "validation": sources["val_mask"]["sha256"],
            "test": sources["test_mask"]["sha256"],
        },
        "descriptor_sha256": sources["descriptor"]["sha256"],
        "raw_prediction_sha256": sources["raw_prediction"]["sha256"],
        "sources": sources,
        "source_npz_member_whitelist": sorted(SOURCE_KEYS),
        "source_prediction_keys_loaded": [
            "train_gene_idx", "val_gene_idx", "selected_prediction_train",
            "selected_prediction_val", "base_prediction_train", "base_prediction_val",
        ],
        "test_arrays_materialized": False,
        "outputs": outputs,
        "output_array_sha256": {
            "train_truth": array_sha256(train_truth),
            "val_truth": array_sha256(val_truth),
            **{
                f"{method}:{split_name}": array_sha256(
                    np.asarray(source_arrays[f"{prefix}_prediction_{split_name}"], dtype=np.float32)
                )
                for method, prefix in (("GeneSPT", "selected"), ("GeneSPT-GC", "base"))
                for split_name in ("train", "val")
            },
        },
    }
    manifest_path = prepared_dir / "prepare_manifest.json"
    write_json(manifest_path, manifest)
    warning = (
        f"low_validation_sample_count:{len(val_idx)}"
        if task.dataset_id == "MHPR_current_panel"
        else ""
    )
    contract = {
        "schema_version": 1,
        "status": "selection_input_ready",
        "identity": manifest["identity"],
        "gene_axis_sha256": gene_axis_sha,
        "split_sha256": manifest["split_sha256"],
        "descriptor_sha256": manifest["descriptor_sha256"],
        "raw_prediction_sha256": manifest["raw_prediction_sha256"],
        "prepare_manifest_sha256": sha256_file(manifest_path),
        "prepared_outputs": outputs,
        "protocol_definition_sha256": core.canonical_hash(core.frozen_protocol_definition()),
        "contains_test_st_expression": False,
        "contains_test_prediction": False,
        "test_gene_descriptors_allowed": True,
        "warnings": [warning] if warning else [],
    }
    write_json(prepared_dir / "selection_contract.json", contract)
    return manifest


def _verify_prepared(contract_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    contract = read_json(contract_path)
    if contract.get("contains_test_st_expression") is not False or contract.get("contains_test_prediction") is not False:
        raise ProtocolError("Selection contract is not physically test-isolated")
    outputs = contract["prepared_outputs"]
    common_path = verify_record(outputs["common"])
    common = load_npz(common_path, COMMON_KEYS, allowed=COMMON_KEYS)
    methods: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        path = verify_record(outputs["methods"][method])
        methods[method] = load_npz(path, METHOD_INPUT_KEYS, allowed=METHOD_INPUT_KEYS)
    return contract, common, methods


def select_task(
    prepared_task_dir: Path,
    selection_task_dir: Path,
    metrics_path: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    """Select a readout using only isolated train/validation artifacts."""

    if seed != SEED:
        raise ProtocolError("The frozen readout seed is 42")
    contract_path = prepared_task_dir / "selection_contract.json"
    contract, common, methods = _verify_prepared(contract_path)
    identity = contract["identity"]
    metrics_module = core.load_metrics_module(metrics_path)
    descriptors = {key: np.asarray(common[key], dtype=np.float32) for key in core.FEATURE_KINDS}
    train_truth = np.asarray(common["train_truth"], dtype=np.float32)
    val_truth = np.asarray(common["val_truth"], dtype=np.float32)
    train_idx = np.asarray(common["train_idx"], dtype=np.int64)
    val_idx = np.asarray(common["val_idx"], dtype=np.int64)
    edges = core.make_knn_edges(np.asarray(common["coordinates"], dtype=np.float32))
    spatiality_train = core.predicted_spatiality(train_truth, train_idx, train_idx, descriptors, edges)
    spatiality_val = core.predicted_spatiality(train_truth, train_idx, val_idx, descriptors, edges)
    selection_task_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, Any] = {}
    method_artifacts: dict[str, Any] = {}
    for method in METHODS:
        result = core.build_validation_candidates(
            method=method,
            fold=int(identity["fold"]),
            train_truth=train_truth,
            val_truth=val_truth,
            pred_train=np.asarray(methods[method]["pred_train"], dtype=np.float32),
            pred_val=np.asarray(methods[method]["pred_val"], dtype=np.float32),
            train_idx=train_idx,
            val_idx=val_idx,
            descriptors=descriptors,
            spatiality_train=spatiality_train,
            spatiality_val=spatiality_val,
            edges=edges,
            seed=seed,
            metrics_module=metrics_module,
        )
        candidates, selected_row, choices, metric_audits = result
        method_dir = selection_task_dir / method
        candidates_path = method_dir / "validation_candidates.csv"
        choices_path = method_dir / "train_oracle_choices.csv"
        audits_path = method_dir / "validation_metric_audit.csv"
        selected_path = method_dir / "selected.json"
        write_csv(candidates_path, candidates)
        write_csv(choices_path, choices)
        write_csv(audits_path, pd.DataFrame(metric_audits))
        write_json(selected_path, selected_row)
        selected[method] = core.json_ready(selected_row)
        method_artifacts[method] = {
            "candidates": source_record(candidates_path),
            "train_oracle_choices": source_record(choices_path),
            "metric_audit": source_record(audits_path),
            "selected": source_record(selected_path),
        }
    code = {
        "core": source_record(Path(core.__file__)),
        "runner": source_record(Path(__file__)),
        "centralized_metrics": source_record(metrics_path),
    }
    lock = {
        "schema_version": 1,
        "status": "selection_locked",
        "created_at_utc": utc_now(),
        "identity": identity,
        "readout_layer": "model_specific_validation_selected_genespt57",
        "not_preregistered": True,
        "external_baselines": "raw_identity_only_no_descriptor_readout",
        "selection_contract": source_record(contract_path),
        "protocol_definition": core.frozen_protocol_definition(),
        "protocol_definition_sha256": core.canonical_hash(core.frozen_protocol_definition()),
        "candidate_family_sha256": core.canonical_hash(core.candidate_names()),
        "selected": selected,
        "method_artifacts": method_artifacts,
        "code": code,
        "warnings": contract.get("warnings", []),
        "test_prediction_accessed_before_lock": False,
        "test_truth_accessed_before_lock": False,
    }
    lock_path = selection_task_dir / "selection_lock.json"
    write_json(lock_path, lock)
    receipt = {
        "schema_version": 1,
        "selection_lock_sha256": sha256_file(lock_path),
        "selection_lock_bytes": lock_path.stat().st_size,
        "created_at_utc": utc_now(),
    }
    write_json(selection_task_dir / "selection_receipt.json", receipt)
    for warning in contract.get("warnings", []):
        print(f"WARNING {identity['dataset_id']} fold{identity['fold']}: {warning}", file=sys.stderr)
    return lock


def _validate_lock(
    prepared_task_dir: Path, selection_task_dir: Path, metrics_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    lock_path = selection_task_dir / "selection_lock.json"
    receipt_path = selection_task_dir / "selection_receipt.json"
    if not lock_path.is_file() or not receipt_path.is_file():
        raise ProtocolError("Selection lock and receipt are required before apply")
    receipt = read_json(receipt_path)
    if receipt.get("selection_lock_sha256") != sha256_file(lock_path):
        raise ProtocolError("Selection lock is missing or was tampered with")
    lock = read_json(lock_path)
    if lock.get("status") != "selection_locked":
        raise ProtocolError("Selection lock status is invalid")
    for key, path in (
        ("core", Path(core.__file__)),
        ("runner", Path(__file__)),
        ("centralized_metrics", metrics_path),
    ):
        record = lock["code"][key]
        if record.get("sha256") != sha256_file(path):
            raise ProtocolError(f"Code/input hash changed after selection lock: {key}")
    if lock.get("candidate_family_sha256") != core.canonical_hash(core.candidate_names()):
        raise ProtocolError("Candidate family changed after selection lock")
    contract_path = prepared_task_dir / "selection_contract.json"
    if lock["selection_contract"]["sha256"] != sha256_file(contract_path):
        raise ProtocolError("Selection contract changed after selection lock")
    contract, common, methods = _verify_prepared(contract_path)
    if contract["protocol_definition_sha256"] != core.canonical_hash(core.frozen_protocol_definition()):
        raise ProtocolError("Frozen protocol definition changed")
    prepare_manifest_path = prepared_task_dir / "prepare_manifest.json"
    if contract["prepare_manifest_sha256"] != sha256_file(prepare_manifest_path):
        raise ProtocolError("Prepare manifest changed after selection")
    prepare_manifest = read_json(prepare_manifest_path)
    for method in METHODS:
        artifacts = lock["method_artifacts"][method]
        for record in artifacts.values():
            verify_record(record)
        candidates = pd.read_csv(resolve_record(artifacts["candidates"]))
        selected_mask = candidates["selected"].astype(str).str.casefold().eq("true")
        if int(selected_mask.sum()) != 1:
            raise ProtocolError(f"Selection CSV is not unique for {method}")
        if str(candidates.loc[selected_mask, "calibration"].iloc[0]) != str(lock["selected"][method]["calibration"]):
            raise ProtocolError(f"Selection CSV/lock mismatch for {method}")
    return lock, prepare_manifest, common, methods


def apply_task(
    prepared_task_dir: Path,
    selection_task_dir: Path,
    apply_task_dir: Path,
    metrics_path: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    """Apply a locked readout to test predictions without reading test truth."""

    if seed != SEED:
        raise ProtocolError("The frozen readout seed is 42")
    lock, manifest, common, methods = _validate_lock(
        prepared_task_dir, selection_task_dir, metrics_path
    )
    sources = manifest["sources"]
    # Deliberately omit full_truth here. Test truth is first opened in evaluate.
    for key in (
        "config", "completion_manifest", "artifact_manifest", "mode_a_split",
        "gene_names", "locations", "train_mask", "val_mask", "test_mask",
        "descriptor", "raw_prediction",
    ):
        verify_record(sources[key])
    raw_prediction_path = resolve_record(sources["raw_prediction"])
    raw_sha_before = sha256_file(raw_prediction_path)
    audit_npz(raw_prediction_path, allowed=SOURCE_KEYS, required=SOURCE_REQUIRED)
    with np.load(raw_prediction_path, allow_pickle=False) as archive:
        test_idx = np.asarray(archive["test_gene_idx"], dtype=np.int64)
        raw_test = {
            "GeneSPT": np.asarray(archive["selected_prediction_test"], dtype=np.float32),
            "GeneSPT-GC": np.asarray(archive["base_prediction_test"], dtype=np.float32),
        }
    test_mask = _mask(resolve_record(sources["test_mask"]), int(manifest["identity"]["n_genes"]))
    if not np.array_equal(test_idx, test_mask):
        raise ProtocolError("Test prediction indices differ from frozen test mask")
    descriptors = {key: np.asarray(common[key], dtype=np.float32) for key in core.FEATURE_KINDS}
    train_truth = np.asarray(common["train_truth"], dtype=np.float32)
    train_idx = np.asarray(common["train_idx"], dtype=np.int64)
    edges = core.make_knn_edges(np.asarray(common["coordinates"], dtype=np.float32))
    spatiality_train = core.predicted_spatiality(train_truth, train_idx, train_idx, descriptors, edges)
    spatiality_test = core.predicted_spatiality(train_truth, train_idx, test_idx, descriptors, edges)
    lock_sha = sha256_file(selection_task_dir / "selection_lock.json")
    outputs: dict[str, Any] = {}
    apply_task_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        prediction = raw_test[method]
        if prediction.shape != (int(manifest["identity"]["n_spots"]), len(test_idx)):
            raise ProtocolError(f"Test prediction shape mismatch for {method}")
        choices = pd.read_csv(
            resolve_record(lock["method_artifacts"][method]["train_oracle_choices"])
        )
        selected_prediction = core.apply_selected_candidate(
            selected=lock["selected"][method],
            pred_train=np.asarray(methods[method]["pred_train"], dtype=np.float32),
            pred_test=prediction,
            train_choices=choices,
            train_idx=train_idx,
            test_idx=test_idx,
            descriptors=descriptors,
            spatiality_train=spatiality_train,
            spatiality_test=spatiality_test,
            edges=edges,
            seed=seed,
            fold=int(manifest["identity"]["fold"]),
        )
        if not np.isfinite(selected_prediction).all():
            raise ProtocolError(f"Selected test prediction is nonfinite for {method}")
        output_path = apply_task_dir / f"{method}.npz"
        save_npz(
            output_path,
            prediction=selected_prediction,
            test_gene_idx=test_idx,
            method=np.asarray(method),
            dataset_id=np.asarray(manifest["identity"]["dataset_id"]),
            fold=np.asarray(int(manifest["identity"]["fold"]), dtype=np.int64),
            selected_calibration=np.asarray(lock["selected"][method]["calibration"]),
            selection_lock_sha256=np.asarray(lock_sha),
        )
        outputs[method] = source_record(output_path)
    raw_sha_after = sha256_file(raw_prediction_path)
    if raw_sha_after != raw_sha_before:
        raise ProtocolError("Raw prediction matrix changed during apply")
    apply_manifest = {
        "schema_version": 1,
        "status": "applied",
        "created_at_utc": utc_now(),
        "identity": manifest["identity"],
        "selection_lock_sha256": lock_sha,
        "raw_prediction_sha256_before": raw_sha_before,
        "raw_prediction_sha256_after": raw_sha_after,
        "raw_prediction_unchanged": True,
        "test_truth_accessed": False,
        "test_prediction_loaded_after_lock_verification": True,
        "outputs": outputs,
    }
    write_json(apply_task_dir / "apply_manifest.json", apply_manifest)
    return apply_manifest


def evaluate_task(
    prepared_task_dir: Path,
    selection_task_dir: Path,
    apply_task_dir: Path,
    evaluation_task_dir: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    """Evaluate locked test outputs; this is the first stage to read test truth."""

    _, manifest, _, _ = _validate_lock(prepared_task_dir, selection_task_dir, metrics_path)
    lock_sha = sha256_file(selection_task_dir / "selection_lock.json")
    apply_manifest_path = apply_task_dir / "apply_manifest.json"
    apply_manifest = read_json(apply_manifest_path)
    if apply_manifest.get("status") != "applied" or apply_manifest.get("test_truth_accessed") is not False:
        raise ProtocolError("Apply manifest does not satisfy the no-test-truth contract")
    if apply_manifest.get("selection_lock_sha256") != lock_sha:
        raise ProtocolError("Apply manifest does not match the active selection lock")
    expected_raw_sha = str(manifest["raw_prediction_sha256"])
    if (
        apply_manifest.get("raw_prediction_sha256_before") != expected_raw_sha
        or apply_manifest.get("raw_prediction_sha256_after") != expected_raw_sha
        or apply_manifest.get("raw_prediction_unchanged") is not True
    ):
        raise ProtocolError("Apply manifest raw-prediction hash chain is invalid")
    sources = manifest["sources"]
    raw_prediction_path = verify_record(sources["raw_prediction"])
    if sha256_file(raw_prediction_path) != expected_raw_sha:
        raise ProtocolError("Raw prediction changed before final evaluation")
    truth_path = verify_record(sources["full_truth"])
    gene_path = verify_record(sources["gene_names"])
    test_mask_path = verify_record(sources["test_mask"])
    n_genes = int(manifest["identity"]["n_genes"])
    n_spots = int(manifest["identity"]["n_spots"])
    test_idx = _mask(test_mask_path, n_genes)
    truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
    if truth.shape != (n_spots, n_genes):
        raise ProtocolError("Full truth shape changed before evaluation")
    test_truth = np.asarray(truth[:, test_idx], dtype=np.float32).copy()
    del truth
    genes = _genes(gene_path, n_genes)
    test_genes = [genes[int(index)] for index in test_idx]
    metrics_module = core.load_metrics_module(metrics_path)
    gene_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    output_records: dict[str, Any] = {}
    for method in METHODS:
        output_path = verify_record(apply_manifest["outputs"][method])
        arrays = load_npz(output_path, SELECTED_TEST_KEYS, allowed=SELECTED_TEST_KEYS)
        if not np.array_equal(np.asarray(arrays["test_gene_idx"], dtype=np.int64), test_idx):
            raise ProtocolError(f"Selected test indices changed for {method}")
        if str(np.asarray(arrays["method"]).item()) != method:
            raise ProtocolError(f"Selected prediction method identity changed for {method}")
        if str(np.asarray(arrays["dataset_id"]).item()) != str(manifest["identity"]["dataset_id"]):
            raise ProtocolError(f"Selected prediction dataset identity changed for {method}")
        if int(np.asarray(arrays["fold"]).item()) != int(manifest["identity"]["fold"]):
            raise ProtocolError(f"Selected prediction fold identity changed for {method}")
        if str(np.asarray(arrays["selection_lock_sha256"]).item()) != lock_sha:
            raise ProtocolError(f"Selected prediction lock identity changed for {method}")
        prediction = np.asarray(arrays["prediction"], dtype=np.float32)
        if prediction.shape != (n_spots, len(test_idx)) or not np.isfinite(prediction).all():
            raise ProtocolError(f"Selected prediction matrix is invalid for {method}")
        per_gene, summary_frame = metrics_module.evaluate_prediction(
            test_truth, prediction, test_genes
        )
        per_gene = per_gene.rename(columns={"gene_idx": "gene_pos"})
        per_gene.insert(0, "gene_idx", test_idx)
        per_gene.insert(0, "fold", int(manifest["identity"]["fold"]))
        per_gene.insert(0, "method", method)
        per_gene.insert(0, "dataset_id", manifest["identity"]["dataset_id"])
        per_gene.insert(0, "dataset", manifest["identity"]["dataset"])
        gene_frames.append(per_gene)
        summary = summary_frame.iloc[0].to_dict()
        fold_rows.append(
            {
                **manifest["identity"],
                "method": method,
                "result_layer": "validation_selected_readout_genespt57",
                "selected_calibration": str(np.asarray(arrays["selected_calibration"]).item()),
                **{metric: float(summary[metric]) for metric in core.METRICS},
                "coverage": float(summary["coverage"]),
                "eligible_genes": int(summary["eligible_genes"]),
                "scored_genes": int(summary["scored_genes"]),
            }
        )
        for metric in core.METRICS:
            coverage_rows.append(
                {
                    "dataset": manifest["identity"]["dataset"],
                    "dataset_id": manifest["identity"]["dataset_id"],
                    "fold": int(manifest["identity"]["fold"]),
                    "method": method,
                    "metric": metric,
                    "eligible": int(summary[f"{metric}_eligible"]),
                    "scored": int(summary[f"{metric}_scored"]),
                    "coverage": float(summary[f"{metric}_coverage"]),
                    "constant_prediction": int(summary[f"{metric}_constant_prediction"]),
                }
            )
        output_records[method] = source_record(output_path)
    evaluation_task_dir.mkdir(parents=True, exist_ok=True)
    gene_path_out = evaluation_task_dir / "gene_metrics.csv"
    fold_path_out = evaluation_task_dir / "fold_metrics.csv"
    coverage_path_out = evaluation_task_dir / "coverage.csv"
    write_csv(gene_path_out, pd.concat(gene_frames, ignore_index=True))
    write_csv(fold_path_out, pd.DataFrame(fold_rows))
    write_csv(coverage_path_out, pd.DataFrame(coverage_rows))
    evaluation_manifest = {
        "schema_version": 1,
        "status": "evaluated",
        "created_at_utc": utc_now(),
        "identity": manifest["identity"],
        "centralized_metrics": source_record(metrics_path),
        "test_truth_first_access_stage": "evaluate",
        "selected_predictions": output_records,
        "outputs": {
            "gene_metrics": source_record(gene_path_out),
            "fold_metrics": source_record(fold_path_out),
            "coverage": source_record(coverage_path_out),
        },
    }
    write_json(evaluation_task_dir / "evaluation_manifest.json", evaluation_manifest)
    return evaluation_manifest


def aggregate_selection(output_root: Path, tasks: Sequence[Task]) -> None:
    candidates: list[pd.DataFrame] = []
    selected: list[dict[str, Any]] = []
    for task in tasks:
        directory = task_subdir(output_root / "selections", task)
        lock = read_json(directory / "selection_lock.json")
        for method in METHODS:
            frame = pd.read_csv(directory / method / "validation_candidates.csv")
            frame.insert(0, "dataset_id", task.dataset_id)
            frame.insert(0, "dataset", task.dataset)
            candidates.append(frame)
            selected.append({"dataset": task.dataset, "dataset_id": task.dataset_id, **lock["selected"][method]})
    write_csv(output_root / "validation_candidates.csv", pd.concat(candidates, ignore_index=True))
    write_csv(output_root / "selected_readouts.csv", pd.DataFrame(selected))


def aggregate_evaluation(
    output_root: Path, tasks: Sequence[Task], raw_report_path: Path
) -> dict[str, Any]:
    gene_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    for task in tasks:
        directory = task_subdir(output_root / "evaluation_by_fold", task)
        gene_frames.append(pd.read_csv(directory / "gene_metrics.csv"))
        fold_frames.append(pd.read_csv(directory / "fold_metrics.csv"))
        coverage_frames.append(pd.read_csv(directory / "coverage.csv"))
    genes = pd.concat(gene_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    write_csv(output_root / "gene_level_metrics.csv", genes)
    write_csv(output_root / "fold_metrics.csv", folds)
    write_csv(output_root / "coverage.csv", coverage)
    five_rows: list[dict[str, Any]] = []
    for (dataset, dataset_id, role, method), group in folds.groupby(
        ["dataset", "dataset_id", "role", "method"], sort=False
    ):
        observed = sorted(group["fold"].astype(int).tolist())
        if observed != list(FOLDS):
            raise ProtocolError(f"Five-fold aggregation incomplete for {dataset_id} {method}: {observed}")
        five_rows.append(
            {
                "dataset": dataset, "dataset_id": dataset_id, "role": role,
                "method": method, "result_layer": "validation_selected_readout_genespt57",
                "folds": 5,
                **{metric: float(group[metric].mean()) for metric in core.METRICS},
                **{f"{metric}_std_ddof0": float(group[metric].std(ddof=0)) for metric in core.METRICS},
                "coverage": float(group["coverage"].mean()),
            }
        )
    five = pd.DataFrame(five_rows)
    write_csv(output_root / "five_fold_metrics.csv", five)

    raw_report = read_json(raw_report_path)
    raw_rows = [
        {
            "dataset": row["dataset"], "dataset_id": row["dataset_id"],
            "role": row["role"], "method": row["method"],
            "result_layer": "raw_identity", "folds": int(row["folds_evaluated"]),
            **{metric: float(row[metric]) for metric in core.METRICS},
            "coverage": float(row["coverage"]),
        }
        for row in raw_report["five_fold_summary"]
        if row.get("status") == "complete" and row.get("method") in EXTERNAL_BASELINES
    ]
    combined = pd.concat([five, pd.DataFrame(raw_rows)], ignore_index=True, sort=False)
    write_csv(output_root / "combined_five_fold_metrics.csv", combined)
    rank_rows: list[dict[str, Any]] = []
    for dataset_id, group in combined[combined["method"] != "GeneSPT-GC"].groupby("dataset_id", sort=False):
        for metric in core.METRICS:
            ascending = metric in {"RMSE", "JSD"}
            ranks = group[metric].rank(method="min", ascending=ascending)
            for (_, row), rank in zip(group.iterrows(), ranks):
                rank_rows.append(
                    {
                        "dataset": row["dataset"], "dataset_id": dataset_id,
                        "role": row["role"], "method": row["method"],
                        "metric": metric, "value": row[metric], "rank": int(rank),
                        "result_layer": row["result_layer"],
                    }
                )
    ranks = pd.DataFrame(rank_rows)
    write_csv(output_root / "benchmark_ranks.csv", ranks)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "readout_layer": "model_specific_validation_selected_genespt57",
        "not_preregistered": True,
        "methods_with_readout": list(METHODS),
        "external_baselines": "raw_identity_only_no_descriptor_readout",
        "task_count": len(tasks),
        "fold_method_evaluations": len(folds),
        "five_fold_summaries": len(five),
        "protocol_definition": core.frozen_protocol_definition(),
        "raw_report": source_record(raw_report_path),
        "outputs": {
            name: source_record(output_root / name)
            for name in (
                "validation_candidates.csv", "selected_readouts.csv",
                "gene_level_metrics.csv", "fold_metrics.csv", "coverage.csv",
                "five_fold_metrics.csv", "combined_five_fold_metrics.csv",
                "benchmark_ranks.csv",
            )
            if (output_root / name).is_file()
        },
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--preflight", action="store_true")
    phase.add_argument("--prepare", action="store_true")
    phase.add_argument("--select", action="store_true")
    phase.add_argument("--apply", action="store_true")
    phase.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--fold", action="append", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--raw-report", type=Path, default=DEFAULT_RAW_REPORT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="For --select only, verify and skip already locked folds.",
    )
    args = parser.parse_args(argv)
    if not any((args.prepare, args.select, args.apply, args.evaluate, args.preflight)):
        args.preflight = True
    if args.seed != SEED:
        parser.error("The frozen readout seed is 42")
    if args.resume and not args.select:
        parser.error("--resume is supported only with --select")
    if args.all and (args.dataset or args.fold):
        parser.error("--all cannot be combined with --dataset/--fold")
    if not args.preflight and not args.all and (not args.dataset or args.fold is None):
        parser.error("Mutating phases require --all or both --dataset and --fold")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = discover_tasks(
        config_path=args.config,
        results_root=args.results_root,
        datasets=None if args.all else args.dataset,
        folds=None if args.all else args.fold,
    )
    if not tasks:
        raise ProtocolError("No tasks selected")
    if args.preflight:
        rows = [preflight_task(task) for task in tasks]
        print(json.dumps({"status": "ready", "task_count": len(rows), "tasks": rows}, indent=2))
        return 0
    for task in tasks:
        prepared = task_subdir(args.output_root / "selection_inputs", task)
        selected = task_subdir(args.output_root / "selections", task)
        applied = task_subdir(args.output_root / "test_predictions", task)
        evaluated = task_subdir(args.output_root / "evaluation_by_fold", task)
        phase_name = next(
            name for name in ("prepare", "select", "apply", "evaluate")
            if getattr(args, name)
        )
        print(
            f"[{phase_name}] {task.dataset_id} fold{task.fold}",
            flush=True,
        )
        if args.prepare:
            prepare_task(task, prepared)
        elif args.select:
            if args.resume and (selected / "selection_lock.json").is_file():
                _validate_lock(prepared, selected, args.metrics_path)
                print(
                    f"[select-resume-verified] {task.dataset_id} fold{task.fold}",
                    flush=True,
                )
                continue
            select_task(prepared, selected, args.metrics_path, args.seed)
        elif args.apply:
            apply_task(prepared, selected, applied, args.metrics_path, args.seed)
        elif args.evaluate:
            evaluate_task(prepared, selected, applied, evaluated, args.metrics_path)
        print(
            f"[{phase_name}-complete] {task.dataset_id} fold{task.fold}",
            flush=True,
        )
    if args.select and args.all:
        aggregate_selection(args.output_root, tasks)
    if args.evaluate and args.all:
        aggregate_evaluation(args.output_root, tasks, args.raw_report)
    print(json.dumps({"status": "complete", "phase": phase_name, "task_count": len(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
