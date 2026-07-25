#!/usr/bin/env python3
"""Fail-closed scheduler for the five formal Protocol A baseline adapters.

The default mode is a read-only preflight.  Model processes are started only
with ``--run``.  Every adapter command uses the fixed container namespace and
the explicit inner-train, validation, and final-test gene-index files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np


GENESPT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = GENESPT_ROOT / "configs" / "protocol_a_datasets.yaml"
INPUTS_RELATIVE = PurePosixPath("results/protocol_a_full_rerun_20260711/inputs")
OUTPUT_RELATIVE = PurePosixPath("results/protocol_a_full_rerun_20260711/baselines")
CONTAINER_PROJECT_ROOT = PurePosixPath("/workspace/GeneSPT")
CONTAINER_WORKSPACE_ROOT = PurePosixPath("/workspace")
CONTAINER_PYTHON = "/opt/conda/bin/python"
PROTOCOL_POLICY = "inner_train_gene_library_size_applied_to_all_columns"
REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_DATASETS = (
    ("Vis9A", "Vis9A_D7_spaim_effective4470"),
    ("HBC", "HBC_shared16112"),
    ("Cell2location", "Cell2location_mouse_brain_ST8059048_shared12819"),
    (
        "seqFISH+",
        "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000",
    ),
    ("MHPR", "MHPR_current_panel"),
    ("MVC", "MVC_shared981"),
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.+-]+$")
CHUNK_BYTES = 8 * 1024 * 1024


class SchedulerError(RuntimeError):
    """Raised when preflight or execution cannot satisfy the frozen contract."""


@dataclass(frozen=True)
class AdapterSpec:
    method: str
    relative_script: PurePosixPath
    gpu: bool
    accepts_normalization_scope: bool
    device_arguments: tuple[str, ...] = ()
    protocol_arguments: tuple[str, ...] = ()


ADAPTERS = (
    AdapterSpec(
        "Tangram",
        PurePosixPath("baseline/tangram/run_tangram_mhpr_fold_from_split.py"),
        True,
        True,
        ("--device", "cuda:0"),
    ),
    AdapterSpec(
        "TransImp",
        PurePosixPath("baseline/tranSpa-main/run_transpa_mhpr_fold_from_split.py"),
        True,
        True,
        ("--device", "cuda:0"),
    ),
    AdapterSpec(
        "SpaIM",
        PurePosixPath("baseline/SpaIM-main/run_spaim_mhpr_fold_from_split.py"),
        True,
        True,
        device_arguments=("--gpu", "0"),
        protocol_arguments=("--disable-native-filtering",),
    ),
    AdapterSpec(
        "SpaGE",
        PurePosixPath("baseline/SpaGE/run_spage_mhpr_fold_from_split.py"),
        False,
        False,
    ),
    AdapterSpec(
        "stPlus",
        PurePosixPath("baseline/stPlus/run_stplus_mhpr_fold_from_split.py"),
        True,
        False,
    ),
)
METHODS = tuple(spec.method for spec in ADAPTERS)
ADAPTER_BY_METHOD = {spec.method: spec for spec in ADAPTERS}


@dataclass(frozen=True)
class FileFingerprint:
    path: Path
    size_bytes: int
    mtime_ns: int
    sha256: str


class FileHasher:
    """Hash each immutable preflight input once and retain a stat guard."""

    def __init__(self) -> None:
        self._records: dict[Path, FileFingerprint] = {}

    def hash(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
        label: str | None = None,
    ) -> str:
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise SchedulerError(f"Expected file is missing: {resolved}")
        before = resolved.stat()
        cached = self._records.get(resolved)
        if cached is None:
            digest = hashlib.sha256()
            with resolved.open("rb") as handle:
                for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
                    digest.update(block)
            after = resolved.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise SchedulerError(f"File changed while hashing: {resolved}")
            cached = FileFingerprint(
                path=resolved,
                size_bytes=int(after.st_size),
                mtime_ns=int(after.st_mtime_ns),
                sha256=digest.hexdigest(),
            )
            self._records[resolved] = cached
        context = label or str(resolved)
        if expected_bytes is not None and cached.size_bytes != int(expected_bytes):
            raise SchedulerError(
                f"{context} byte count mismatch: {cached.size_bytes} != {expected_bytes}"
            )
        if expected_sha256 is not None:
            _validate_sha256(expected_sha256, context=f"{context} declared SHA256")
            if cached.sha256 != expected_sha256:
                raise SchedulerError(f"{context} SHA256 mismatch")
        return cached.sha256

    def fingerprint(self, path: str | Path) -> FileFingerprint:
        resolved = Path(path).resolve(strict=True)
        if resolved not in self._records:
            self.hash(resolved)
        return self._records[resolved]

    def assert_unchanged(self) -> None:
        for record in self._records.values():
            current = record.path.stat()
            if (
                current.st_size != record.size_bytes
                or current.st_mtime_ns != record.mtime_ns
            ):
                raise SchedulerError(
                    f"Preflight input changed after hashing: {record.path}"
                )


@dataclass(frozen=True)
class FoldBundle:
    dataset_name: str
    dataset_id: str
    fold: int
    expected_shape: tuple[int, int]
    train_gene_count: int
    validation_gene_count: int
    test_gene_count: int
    command_inputs: Mapping[str, Path]
    package_inputs: Mapping[str, Path]
    input_file_sha256: Mapping[str, str]
    input_sha256: str


@dataclass(frozen=True)
class BaselineTask:
    method: str
    dataset_name: str
    dataset_id: str
    fold: int
    gpu: bool
    command: tuple[str, ...]
    output_dir: Path
    expected_shape: tuple[int, int]
    train_gene_count: int
    validation_gene_count: int
    test_gene_count: int
    command_input_sha256: Mapping[str, str]
    input_file_sha256: Mapping[str, str]
    input_sha256: str
    adapter_path: Path
    adapter_sha256: str
    config_sha256: str
    protocol_helper_sha256: str
    input_preparer_sha256: str

    @property
    def key(self) -> str:
        return f"{self.method}__{self.dataset_id}__fold{self.fold}"


@dataclass
class PreparedSchedule:
    project_root: Path
    inputs_root: Path
    output_root: Path
    config_path: Path
    config_sha256: str
    protocol_helper_sha256: str
    input_preparer_sha256: str
    adapter_sha256: Mapping[str, str]
    datasets: tuple[str, ...]
    methods: tuple[str, ...]
    folds: tuple[int, ...]
    tasks: tuple[BaselineTask, ...]
    hasher: FileHasher


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise SchedulerError(f"{context} is not a lowercase SHA256")
    return value


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: str | Path, *, context: str) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise SchedulerError(f"Could not read {context}: {source}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as error:
            raise SchedulerError(
                f"{context} is not JSON-form YAML and PyYAML is unavailable"
            ) from error
        try:
            payload = yaml.safe_load(text)
        except Exception as error:
            raise SchedulerError(f"Could not parse {context}: {source}") from error
    if not isinstance(payload, dict):
        raise SchedulerError(f"{context} must contain one mapping")
    return payload


def _safe_relative_path(value: object, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise SchedulerError(f"{context} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise SchedulerError(f"{context} must not escape its declared root")
    return path


def _join_relative(root: Path, value: object, *, context: str) -> Path:
    relative = _safe_relative_path(value, context=context)
    candidate = root.joinpath(*relative.parts).resolve(strict=False)
    _require_within(candidate, root, context=context)
    return candidate


def _require_within(path: Path, root: Path, *, context: str) -> None:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise SchedulerError(f"{context} is outside {resolved_root}: {path}") from error


def _container_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve(strict=False)
    project = project_root.resolve(strict=False)
    workspace = project.parent
    try:
        relative = resolved.relative_to(project)
        container = CONTAINER_PROJECT_ROOT.joinpath(*relative.parts)
    except ValueError:
        try:
            relative = resolved.relative_to(workspace)
        except ValueError as error:
            raise SchedulerError(
                f"Path cannot be represented in the fixed container mount: {resolved}"
            ) from error
        container = CONTAINER_WORKSPACE_ROOT.joinpath(*relative.parts)
    if not container.is_absolute():
        raise AssertionError("Container path must be absolute")
    return container.as_posix()


def _record_hash(
    record: object,
    path: Path,
    hasher: FileHasher,
    *,
    context: str,
    expected_record_path: str | None = None,
) -> str:
    if not isinstance(record, dict):
        raise SchedulerError(f"{context} record must be a mapping")
    if expected_record_path is not None and record.get("path") != expected_record_path:
        raise SchedulerError(
            f"{context} path mismatch: {record.get('path')!r} != {expected_record_path!r}"
        )
    sha = _validate_sha256(record.get("sha256"), context=f"{context} SHA256")
    size = record.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise SchedulerError(f"{context} byte count is invalid")
    return hasher.hash(
        path,
        expected_sha256=sha,
        expected_bytes=size,
        label=context,
    )


def _validate_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if config.get("protocol") != "A":
        raise SchedulerError("Config protocol must be A")
    if config.get("output_root") != INPUTS_RELATIVE.as_posix():
        raise SchedulerError("Config output_root is not the frozen Protocol A inputs root")
    if config.get("folds") != list(REQUIRED_FOLDS):
        raise SchedulerError("Config must declare exactly folds 0 through 4")
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not all(
        isinstance(dataset, dict) for dataset in datasets
    ):
        raise SchedulerError("Config datasets must be a list of mappings")
    observed = tuple((dataset.get("name"), dataset.get("dataset_id")) for dataset in datasets)
    if set(observed) != set(REQUIRED_DATASETS) or len(observed) != len(REQUIRED_DATASETS):
        raise SchedulerError("Config must contain the frozen six-dataset set exactly once")
    for dataset in datasets:
        dataset_id = dataset.get("dataset_id")
        if not isinstance(dataset_id, str) or not SAFE_COMPONENT.fullmatch(dataset_id):
            raise SchedulerError(f"Unsafe dataset_id: {dataset_id!r}")
        shape = dataset.get("expected_st_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise SchedulerError(f"{dataset_id}: expected_st_shape is invalid")
        for key in (
            "raw_counts",
            "scrna_counts",
            "locations",
            "gene_names",
            "frozen_split",
            "train_mask",
            "val_mask",
            "test_mask",
        ):
            if key not in dataset:
                raise SchedulerError(f"{dataset_id}: config is missing {key}")
    return tuple(datasets)


def _select_datasets(
    datasets: Sequence[Mapping[str, Any]], selectors: Sequence[str] | None
) -> tuple[dict[str, Any], ...]:
    if not selectors:
        return tuple(dict(dataset) for dataset in datasets)
    lookup: dict[str, Mapping[str, Any]] = {}
    for dataset in datasets:
        for value in (dataset["name"], dataset["dataset_id"]):
            key = str(value).casefold()
            if key in lookup and lookup[key] is not dataset:
                raise SchedulerError(f"Ambiguous dataset selector: {value}")
            lookup[key] = dataset
    requested: set[str] = set()
    for selector in selectors:
        match = lookup.get(str(selector).casefold())
        if match is None:
            raise SchedulerError(f"Unknown dataset selector: {selector}")
        requested.add(str(match["dataset_id"]))
    return tuple(
        dict(dataset)
        for dataset in datasets
        if str(dataset["dataset_id"]) in requested
    )


def _select_methods(selectors: Sequence[str] | None) -> tuple[AdapterSpec, ...]:
    if not selectors:
        return ADAPTERS
    lookup = {method.casefold(): method for method in METHODS}
    selected: set[str] = set()
    for selector in selectors:
        method = lookup.get(str(selector).casefold())
        if method is None:
            raise SchedulerError(f"Unknown method selector: {selector}")
        selected.add(method)
    return tuple(spec for spec in ADAPTERS if spec.method in selected)


def _select_folds(selectors: Sequence[int] | None) -> tuple[int, ...]:
    if not selectors:
        return REQUIRED_FOLDS
    selected = tuple(sorted(set(selectors)))
    if any(fold not in REQUIRED_FOLDS for fold in selected):
        raise SchedulerError(f"Fold selectors must be in {list(REQUIRED_FOLDS)}")
    return selected


def _index_list(payload: Mapping[str, Any], key: str, n_genes: int) -> np.ndarray:
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise SchedulerError(f"Mode-A split field {key} must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise SchedulerError(f"Mode-A split field {key} must contain integers")
    array = np.asarray(values, dtype=np.int64)
    if np.any(array < 0) or np.any(array >= n_genes):
        raise SchedulerError(f"Mode-A split field {key} contains an out-of-range index")
    if np.unique(array).size != array.size:
        raise SchedulerError(f"Mode-A split field {key} contains duplicates")
    return array


def _load_mask(path: Path, *, label: str) -> np.ndarray:
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as error:
        raise SchedulerError(f"Could not load {label} mask: {path}") from error
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise SchedulerError(f"{label} mask must be a one-dimensional integer array")
    return values.astype(np.int64, copy=False)


def _global_fold_summary(
    manifest: Mapping[str, Any], dataset_id: str, fold: int
) -> Mapping[str, Any]:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise SchedulerError("Materialization manifest datasets field is invalid")
    matches = [item for item in datasets if isinstance(item, dict) and item.get("dataset_id") == dataset_id]
    if len(matches) != 1:
        raise SchedulerError(f"Materialization manifest is missing {dataset_id}")
    folds = matches[0].get("folds")
    if not isinstance(folds, list):
        raise SchedulerError(f"Materialization manifest folds are invalid for {dataset_id}")
    fold_matches = [item for item in folds if isinstance(item, dict) and item.get("fold") == fold]
    if len(fold_matches) != 1:
        raise SchedulerError(f"Materialization manifest is missing {dataset_id} fold{fold}")
    return fold_matches[0]


def _validate_generated_record(
    records: Mapping[str, Any],
    key: str,
    path: Path,
    relative_path: str,
    hasher: FileHasher,
    *,
    context: str,
) -> str:
    if key not in records:
        raise SchedulerError(f"{context} is missing output record {key}")
    return _record_hash(
        records[key],
        path,
        hasher,
        context=f"{context} {key}",
        expected_record_path=relative_path,
    )


def _prepare_fold_bundle(
    *,
    project_root: Path,
    archive_root: Path,
    inputs_root: Path,
    config_path: Path,
    protocol_helper_path: Path,
    dataset: Mapping[str, Any],
    fold: int,
    materialization_manifest_path: Path,
    materialization_manifest: Mapping[str, Any],
    hasher: FileHasher,
) -> FoldBundle:
    dataset_name = str(dataset["name"])
    dataset_id = str(dataset["dataset_id"])
    expected_shape = tuple(int(value) for value in dataset["expected_st_shape"])
    fold_dir = inputs_root / dataset_id / f"fold{fold}"
    artifact_path = fold_dir / "artifact_manifest.json"
    split_path = fold_dir / "mode_a_split.json"
    normalization_path = fold_dir / "normalization_audit.json"
    truth_path = fold_dir / "full_truth.npy"

    artifact = _load_mapping(artifact_path, context=f"{dataset_id} fold{fold} artifact manifest")
    if artifact.get("dataset_id") != dataset_id or artifact.get("fold") != fold:
        raise SchedulerError(f"{dataset_id} fold{fold}: artifact identity mismatch")
    input_records = artifact.get("input_artifacts")
    output_records = artifact.get("output_artifacts")
    if not isinstance(input_records, dict) or not isinstance(output_records, dict):
        raise SchedulerError(f"{dataset_id} fold{fold}: artifact records are invalid")

    source_paths = {
        "raw_counts": _join_relative(archive_root, dataset["raw_counts"], context="raw_counts"),
        "scrna_counts": _join_relative(archive_root, dataset["scrna_counts"], context="scrna_counts"),
        "locations": _join_relative(archive_root, dataset["locations"], context="locations"),
        "gene_names": _join_relative(archive_root, dataset["gene_names"], context="gene_names"),
        "frozen_split": _join_relative(
            archive_root,
            str(dataset["frozen_split"]).format(fold=fold),
            context="frozen_split",
        ),
        "train_mask": _join_relative(
            archive_root,
            str(dataset["train_mask"]).format(fold=fold),
            context="train_mask",
        ),
        "val_mask": _join_relative(
            archive_root,
            str(dataset["val_mask"]).format(fold=fold),
            context="val_mask",
        ),
        "test_mask": _join_relative(
            archive_root,
            str(dataset["test_mask"]).format(fold=fold),
            context="test_mask",
        ),
    }
    expected_input_keys = {*source_paths, "config", "protocol_a_preprocessing"}
    if set(input_records) != expected_input_keys:
        raise SchedulerError(f"{dataset_id} fold{fold}: input artifact key set mismatch")
    source_hashes: dict[str, str] = {}
    for key, path in source_paths.items():
        configured = dataset[key if key not in {"train_mask", "val_mask", "test_mask", "frozen_split"} else key]
        expected_path = str(configured).format(fold=fold)
        source_hashes[key] = _record_hash(
            input_records[key],
            path,
            hasher,
            context=f"{dataset_id} fold{fold} {key}",
            expected_record_path=expected_path,
        )

    config_relative = config_path.relative_to(project_root).as_posix()
    helper_relative = protocol_helper_path.relative_to(project_root).as_posix()
    _record_hash(
        input_records["config"],
        config_path,
        hasher,
        context=f"{dataset_id} fold{fold} config",
        expected_record_path=config_relative,
    )
    _record_hash(
        input_records["protocol_a_preprocessing"],
        protocol_helper_path,
        hasher,
        context=f"{dataset_id} fold{fold} protocol helper",
        expected_record_path=helper_relative,
    )

    base_relative = f"{dataset_id}/fold{fold}"
    generated_paths = {
        "full_truth": truth_path,
        "mode_a_split": split_path,
        "normalization_audit": normalization_path,
    }
    generated_hashes = {
        key: _validate_generated_record(
            output_records,
            key,
            path,
            f"{base_relative}/{path.name}",
            hasher,
            context=f"{dataset_id} fold{fold}",
        )
        for key, path in generated_paths.items()
    }

    summary = _global_fold_summary(materialization_manifest, dataset_id, fold)
    summary_outputs = summary.get("output_artifacts")
    if not isinstance(summary_outputs, dict):
        raise SchedulerError(f"{dataset_id} fold{fold}: global output records are invalid")
    for key, path in generated_paths.items():
        summary_hash = _validate_generated_record(
            summary_outputs,
            key,
            path,
            f"{base_relative}/{path.name}",
            hasher,
            context=f"global {dataset_id} fold{fold}",
        )
        if summary_hash != generated_hashes[key]:
            raise SchedulerError(f"{dataset_id} fold{fold}: global and fold hashes disagree")
    artifact_hash = _validate_generated_record(
        summary_outputs,
        "artifact_manifest",
        artifact_path,
        f"{base_relative}/artifact_manifest.json",
        hasher,
        context=f"global {dataset_id} fold{fold}",
    )

    split = _load_mapping(split_path, context=f"{dataset_id} fold{fold} Mode-A split")
    if (
        split.get("protocol") != "A"
        or split.get("protocol_role") != "strict_primary_modeA"
        or split.get("dataset") != dataset_name
        or split.get("dataset_id") != dataset_id
        or split.get("fold") != fold
    ):
        raise SchedulerError(f"{dataset_id} fold{fold}: Mode-A split identity mismatch")
    n_genes = expected_shape[1]
    train = _index_list(split, "inner_train_gene_idx", n_genes)
    validation = _index_list(split, "inner_validation_gene_idx", n_genes)
    test = _index_list(split, "final_test_gene_idx", n_genes)
    if split.get("train_gene_idx") != train.tolist() or split.get("val_gene_idx") != validation.tolist():
        raise SchedulerError(f"{dataset_id} fold{fold}: duplicate split fields disagree")
    hidden = np.concatenate((validation, test))
    if split.get("test_gene_idx") != hidden.tolist() or split.get("hidden_gene_idx") != hidden.tolist():
        raise SchedulerError(f"{dataset_id} fold{fold}: hidden genes are not validation plus test")
    covered = np.concatenate((train, validation, test))
    if covered.size != n_genes or not np.array_equal(np.sort(covered), np.arange(n_genes)):
        raise SchedulerError(f"{dataset_id} fold{fold}: split is not a complete disjoint partition")
    visibility = split.get("visibility")
    if not isinstance(visibility, dict) or visibility.get("visible_st_gene_idx") != train.tolist():
        raise SchedulerError(f"{dataset_id} fold{fold}: visible ST genes are not inner-train only")
    if visibility.get("hidden_st_gene_idx") != hidden.tolist() or not all(
        visibility.get(key) is True
        for key in (
            "model_fit_uses_inner_train_only",
            "validation_st_hidden_from_model_fit",
            "final_test_st_hidden_from_all_fit_and_selection",
        )
    ):
        raise SchedulerError(f"{dataset_id} fold{fold}: Mode-A visibility contract failed")

    for label, path, expected in (
        ("train", source_paths["train_mask"], train),
        ("validation", source_paths["val_mask"], validation),
        ("test", source_paths["test_mask"], test),
    ):
        observed = _load_mask(path, label=label)
        if not np.array_equal(observed, expected):
            raise SchedulerError(f"{dataset_id} fold{fold}: {label} mask disagrees with Mode-A split")

    try:
        truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise SchedulerError(f"{dataset_id} fold{fold}: could not inspect full truth") from error
    if tuple(truth.shape) != expected_shape or truth.dtype != np.float32:
        raise SchedulerError(f"{dataset_id} fold{fold}: full truth shape/dtype mismatch")
    del truth

    normalization = _load_mapping(
        normalization_path,
        context=f"{dataset_id} fold{fold} normalization audit",
    )
    if normalization.get("dataset_id") != dataset_id or normalization.get("fold") != fold:
        raise SchedulerError(f"{dataset_id} fold{fold}: normalization audit identity mismatch")
    protocol_a = normalization.get("protocol_a")
    if not isinstance(protocol_a, dict) or protocol_a.get("policy") != PROTOCOL_POLICY:
        raise SchedulerError(f"{dataset_id} fold{fold}: normalization policy mismatch")
    if protocol_a.get("denominator_gene_count") != int(train.size):
        raise SchedulerError(f"{dataset_id} fold{fold}: denominator gene count mismatch")
    if normalization.get("input_artifacts") != input_records:
        raise SchedulerError(f"{dataset_id} fold{fold}: normalization input audit mismatch")
    output_sha = normalization.get("output_sha256")
    if not isinstance(output_sha, dict) or (
        output_sha.get("full_truth_npy") != generated_hashes["full_truth"]
        or output_sha.get("mode_a_split_json") != generated_hashes["mode_a_split"]
    ):
        raise SchedulerError(f"{dataset_id} fold{fold}: normalization output hash mismatch")
    if summary.get("shape") != list(expected_shape):
        raise SchedulerError(f"{dataset_id} fold{fold}: global shape mismatch")
    if (
        summary.get("inner_train_gene_count") != int(train.size)
        or summary.get("inner_validation_gene_count") != int(validation.size)
        or summary.get("final_test_gene_count") != int(test.size)
    ):
        raise SchedulerError(f"{dataset_id} fold{fold}: global split counts mismatch")

    command_inputs = {
        "locations_path": source_paths["locations"],
        "st_data": source_paths["raw_counts"],
        "sc_data": source_paths["scrna_counts"],
        "gene_split_json": split_path,
        "train_gene_idx_path": source_paths["train_mask"],
        "val_gene_idx_path": source_paths["val_mask"],
        "test_gene_idx_path": source_paths["test_mask"],
    }
    package_inputs = {
        **source_paths,
        "full_truth": truth_path,
        "mode_a_split": split_path,
        "normalization_audit": normalization_path,
        "artifact_manifest": artifact_path,
        "materialization_manifest": materialization_manifest_path,
    }
    package_hashes = {
        key: hasher.hash(path, label=f"{dataset_id} fold{fold} package input {key}")
        for key, path in package_inputs.items()
    }
    if package_hashes["artifact_manifest"] != artifact_hash:
        raise AssertionError("Artifact manifest hash cache disagrees")
    return FoldBundle(
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        fold=fold,
        expected_shape=expected_shape,
        train_gene_count=int(train.size),
        validation_gene_count=int(validation.size),
        test_gene_count=int(test.size),
        command_inputs=command_inputs,
        package_inputs=package_inputs,
        input_file_sha256=package_hashes,
        input_sha256=canonical_sha256(package_hashes),
    )


def _build_command(
    spec: AdapterSpec,
    bundle: FoldBundle,
    *,
    project_root: Path,
    output_dir: Path,
) -> tuple[str, ...]:
    adapter = project_root.joinpath(*spec.relative_script.parts)
    command = [
        CONTAINER_PYTHON,
        _container_path(adapter, project_root),
        "--locations-path",
        _container_path(bundle.command_inputs["locations_path"], project_root),
        "--st-data",
        _container_path(bundle.command_inputs["st_data"], project_root),
        "--sc-data",
        _container_path(bundle.command_inputs["sc_data"], project_root),
        "--gene-split-json",
        _container_path(bundle.command_inputs["gene_split_json"], project_root),
        "--train-gene-idx-path",
        _container_path(bundle.command_inputs["train_gene_idx_path"], project_root),
        "--val-gene-idx-path",
        _container_path(bundle.command_inputs["val_gene_idx_path"], project_root),
        "--test-gene-idx-path",
        _container_path(bundle.command_inputs["test_gene_idx_path"], project_root),
    ]
    if spec.accepts_normalization_scope:
        command.extend(("--st-normalization-scope", "train_genes"))
    command.extend(("--model-gene-scope", "train_indices"))
    command.extend(spec.device_arguments)
    command.extend(spec.protocol_arguments)
    command.extend(
        (
            "--output-dir",
            _container_path(output_dir, project_root),
            "--skip-adapter-metrics",
        )
    )
    return tuple(command)


def prepare_schedule(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = GENESPT_ROOT,
    datasets: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
) -> PreparedSchedule:
    """Validate the complete frozen package and return selected model tasks."""

    project = Path(project_root).resolve(strict=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = project / config_file
    config_file = config_file.resolve(strict=True)
    _require_within(config_file, project, context="Config")
    hasher = FileHasher()
    config_sha = hasher.hash(config_file, label="Protocol A config")
    config = _load_mapping(config_file, context="Protocol A config")
    configured_datasets = _validate_config(config)

    archive = config.get("archive")
    if not isinstance(archive, dict) or not isinstance(archive.get("root"), str):
        raise SchedulerError("Config archive.root is invalid")
    archive_root = (project / archive["root"]).resolve(strict=True)
    _require_within(archive_root, project.parent, context="Archive root")
    inputs_root = project.joinpath(*INPUTS_RELATIVE.parts).resolve(strict=True)
    output_root = project.joinpath(*OUTPUT_RELATIVE.parts).resolve(strict=False)
    _require_within(inputs_root, project, context="Inputs root")
    _require_within(output_root, project, context="Baseline output root")

    protocol_helper_path = project / "main" / "protocol_a_preprocessing.py"
    input_preparer_path = project / "scripts" / "protocol_a_full" / "prepare_protocol_a_inputs.py"
    helper_sha = hasher.hash(protocol_helper_path, label="Protocol A helper")
    preparer_sha = hasher.hash(input_preparer_path, label="Protocol A input preparer")
    adapter_hashes: dict[str, str] = {}
    adapter_paths: dict[str, Path] = {}
    for spec in ADAPTERS:
        path = project.joinpath(*spec.relative_script.parts)
        adapter_paths[spec.method] = path
        adapter_hashes[spec.method] = hasher.hash(path, label=f"{spec.method} adapter")

    materialization_path = inputs_root / "materialization_manifest.json"
    materialization_sha = hasher.hash(materialization_path, label="Materialization manifest")
    materialization = _load_mapping(materialization_path, context="materialization manifest")
    if (
        materialization.get("protocol") != "A"
        or materialization.get("mode") != "materialized"
        or materialization.get("model_started") is not False
        or materialization.get("dataset_count") != len(REQUIRED_DATASETS)
        or materialization.get("folds_per_dataset") != len(REQUIRED_FOLDS)
        or materialization.get("output_root") != INPUTS_RELATIVE.as_posix()
    ):
        raise SchedulerError("Materialization manifest header is not the frozen six-dataset package")
    code_artifacts = materialization.get("code_artifacts")
    if not isinstance(code_artifacts, dict):
        raise SchedulerError("Materialization code_artifacts is invalid")
    for key, path, expected_sha in (
        ("config", config_file, config_sha),
        ("prepare_script", input_preparer_path, preparer_sha),
        ("protocol_a_preprocessing", protocol_helper_path, helper_sha),
    ):
        record = code_artifacts.get(key)
        observed = _record_hash(
            record,
            path,
            hasher,
            context=f"materialization {key}",
            expected_record_path=path.relative_to(project).as_posix(),
        )
        if observed != expected_sha:
            raise SchedulerError(f"Materialization {key} hash disagrees with current code")

    selected_datasets = _select_datasets(configured_datasets, datasets)
    selected_methods = _select_methods(methods)
    selected_folds = _select_folds(folds)
    bundles: list[FoldBundle] = []
    for dataset in selected_datasets:
        for fold in selected_folds:
            bundles.append(
                _prepare_fold_bundle(
                    project_root=project,
                    archive_root=archive_root,
                    inputs_root=inputs_root,
                    config_path=config_file,
                    protocol_helper_path=protocol_helper_path,
                    dataset=dataset,
                    fold=fold,
                    materialization_manifest_path=materialization_path,
                    materialization_manifest=materialization,
                    hasher=hasher,
                )
            )

    tasks: list[BaselineTask] = []
    for spec in selected_methods:
        for bundle in bundles:
            output_dir = output_root / spec.method / bundle.dataset_id / f"fold{bundle.fold}"
            _require_within(output_dir, output_root, context="Task output")
            command = _build_command(
                spec,
                bundle,
                project_root=project,
                output_dir=output_dir,
            )
            command_input_sha = {
                key: hasher.hash(path) for key, path in bundle.command_inputs.items()
            }
            tasks.append(
                BaselineTask(
                    method=spec.method,
                    dataset_name=bundle.dataset_name,
                    dataset_id=bundle.dataset_id,
                    fold=bundle.fold,
                    gpu=spec.gpu,
                    command=command,
                    output_dir=output_dir,
                    expected_shape=bundle.expected_shape,
                    train_gene_count=bundle.train_gene_count,
                    validation_gene_count=bundle.validation_gene_count,
                    test_gene_count=bundle.test_gene_count,
                    command_input_sha256=command_input_sha,
                    input_file_sha256=dict(bundle.input_file_sha256),
                    input_sha256=bundle.input_sha256,
                    adapter_path=adapter_paths[spec.method],
                    adapter_sha256=adapter_hashes[spec.method],
                    config_sha256=config_sha,
                    protocol_helper_sha256=helper_sha,
                    input_preparer_sha256=preparer_sha,
                )
            )
    if hasher.hash(materialization_path) != materialization_sha:
        raise AssertionError("Materialization manifest hash cache disagrees")
    hasher.assert_unchanged()
    return PreparedSchedule(
        project_root=project,
        inputs_root=inputs_root,
        output_root=output_root,
        config_path=config_file,
        config_sha256=config_sha,
        protocol_helper_sha256=helper_sha,
        input_preparer_sha256=preparer_sha,
        adapter_sha256=adapter_hashes,
        datasets=tuple(str(dataset["name"]) for dataset in selected_datasets),
        methods=tuple(spec.method for spec in selected_methods),
        folds=selected_folds,
        tasks=tuple(tasks),
        hasher=hasher,
    )


def task_provenance(task: BaselineTask) -> dict[str, object]:
    return {
        "command": list(task.command),
        "command_sha256": canonical_sha256(list(task.command)),
        "input_sha256": task.input_sha256,
        "input_file_sha256": dict(task.input_file_sha256),
        "adapter_sha256": task.adapter_sha256,
        "config_sha256": task.config_sha256,
        "protocol_helper_sha256": task.protocol_helper_sha256,
        "input_preparer_sha256": task.input_preparer_sha256,
    }


def _declared_output_sha256(audit: Mapping[str, Any]) -> str:
    direct = audit.get("output_matrix_sha256")
    if isinstance(direct, str):
        return _validate_sha256(direct, context="adapter output_matrix_sha256")
    outputs = audit.get("output_sha256")
    if not isinstance(outputs, dict) or "imputed_expression.npy" not in outputs:
        raise SchedulerError("Adapter audit does not declare the prediction SHA256")
    value = outputs["imputed_expression.npy"]
    if isinstance(value, dict):
        value = value.get("sha256")
    return _validate_sha256(value, context="adapter prediction SHA256")


def validate_task_output(task: BaselineTask) -> dict[str, object]:
    """Recompute prediction/audit hashes and validate the strict adapter audit."""

    _require_within(task.output_dir, task.output_dir.parent.parent.parent.parent, context="Task output")
    prediction_path = task.output_dir / "imputed_expression.npy"
    audit_path = task.output_dir / "adapter_run_audit.json"
    if not prediction_path.is_file() or not audit_path.is_file():
        raise SchedulerError(f"{task.key}: prediction or adapter audit is missing")
    audit = _load_mapping(audit_path, context=f"{task.key} adapter audit")
    if audit.get("adapter") != task.method:
        raise SchedulerError(f"{task.key}: adapter audit method mismatch")
    if audit.get("protocol_role") != "strict_primary_modeA":
        raise SchedulerError(f"{task.key}: adapter audit is not strict Mode-A")
    if audit.get("eligible_for_strict_primary") is not True:
        raise SchedulerError(f"{task.key}: adapter audit is not formally eligible")
    if audit.get("model_gene_scope") != "train_indices":
        raise SchedulerError(f"{task.key}: adapter exposed genes outside train_indices")
    if audit.get("st_normalization_scope") != "train_genes":
        raise SchedulerError(f"{task.key}: adapter normalization scope mismatch")
    if task.method == "SpaIM" and audit.get("native_filtering_disabled") is not True:
        raise SchedulerError(
            f"{task.key}: SpaIM native filtering was not disabled for the frozen gene universe"
        )
    skipped = audit.get("adapter_metrics_skipped")
    if skipped is None and isinstance(audit.get("scope"), dict):
        skipped = audit["scope"].get("adapter_metrics_skipped")
    if skipped is not True:
        raise SchedulerError(f"{task.key}: adapter-local metrics were not disabled")
    source_sha = audit.get("adapter_source_sha256")
    if source_sha is not None and source_sha != task.adapter_sha256:
        raise SchedulerError(f"{task.key}: adapter source hash mismatch")
    input_hashes = audit.get("input_sha256")
    if not isinstance(input_hashes, dict):
        raise SchedulerError(f"{task.key}: adapter input hash audit is missing")
    if dict(input_hashes) != dict(task.command_input_sha256):
        raise SchedulerError(f"{task.key}: adapter input hashes do not match preflight")
    normalization = audit.get("normalization_audit")
    if not isinstance(normalization, dict) or normalization.get("policy") != PROTOCOL_POLICY:
        raise SchedulerError(f"{task.key}: adapter normalization audit mismatch")
    if normalization.get("denominator_gene_count") != task.train_gene_count:
        raise SchedulerError(f"{task.key}: adapter denominator count mismatch")
    for key, expected in (
        ("validation_gene_count", task.validation_gene_count),
        ("final_test_gene_count", task.test_gene_count),
    ):
        if key in audit and audit[key] != expected:
            raise SchedulerError(f"{task.key}: adapter {key} mismatch")
    shape = audit.get("imputed_matrix_shape")
    if shape != list(task.expected_shape):
        raise SchedulerError(f"{task.key}: adapter prediction shape audit mismatch")
    if audit.get("prediction_finite") is False:
        raise SchedulerError(f"{task.key}: adapter reported non-finite predictions")
    try:
        prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise SchedulerError(f"{task.key}: prediction is not a valid NumPy array") from error
    if tuple(prediction.shape) != task.expected_shape or prediction.dtype != np.float32:
        raise SchedulerError(f"{task.key}: prediction shape/dtype mismatch")
    del prediction
    output_sha = sha256_file(prediction_path)
    if output_sha != _declared_output_sha256(audit):
        raise SchedulerError(f"{task.key}: prediction hash disagrees with adapter audit")
    audit_sha = sha256_file(audit_path)
    return {
        "prediction_path": str(prediction_path),
        "prediction_sha256": output_sha,
        "prediction_bytes": int(prediction_path.stat().st_size),
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha,
        "audit_bytes": int(audit_path.stat().st_size),
    }


def _status_path(schedule: PreparedSchedule, task: BaselineTask) -> Path:
    return schedule.output_root / "_scheduler" / "status" / f"{task.key}.json"


def _log_path(schedule: PreparedSchedule, task: BaselineTask) -> Path:
    return schedule.output_root / "_scheduler" / "logs" / f"{task.key}.log"


def _atomic_write_json(path: Path, payload: Mapping[str, Any], root: Path) -> None:
    _require_within(path, root, context="Scheduler JSON output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    _require_within(temporary, root, context="Scheduler temporary output")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def resume_decision(
    task: BaselineTask, status_path: Path
) -> tuple[bool, str, dict[str, object] | None]:
    if not status_path.is_file():
        return False, "status_missing", None
    try:
        status = _load_mapping(status_path, context=f"{task.key} status")
        if status.get("status") != "completed":
            return False, f"status_{status.get('status', 'unknown')}", None
        expected = task_provenance(task)
        for key, value in expected.items():
            if status.get(key) != value:
                return False, f"{key}_mismatch", None
        output = validate_task_output(task)
        if status.get("prediction_sha256") != output["prediction_sha256"]:
            return False, "prediction_sha256_mismatch", None
        if status.get("audit_sha256") != output["audit_sha256"]:
            return False, "audit_sha256_mismatch", None
        return True, "complete_and_verified", output
    except Exception as error:
        return False, f"verification_failed: {type(error).__name__}: {error}", None


def _remove_task_output(task: BaselineTask, output_root: Path) -> None:
    if not task.output_dir.exists():
        return
    _require_within(task.output_dir, output_root, context="Stale task output")
    expected = output_root / task.method / task.dataset_id / f"fold{task.fold}"
    if task.output_dir.resolve(strict=False) != expected.resolve(strict=False):
        raise SchedulerError(f"Refusing to clear unexpected task path: {task.output_dir}")
    shutil.rmtree(task.output_dir)


def execute_schedule(
    schedule: PreparedSchedule,
    *,
    resume: bool,
    continue_on_error: bool,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, object]:
    """Execute tasks synchronously; all GPU consumers are therefore serialized."""

    schedule.output_root.mkdir(parents=True, exist_ok=True)
    _require_within(schedule.output_root, schedule.project_root, context="Baseline output root")
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []
    attempted = 0
    for task in schedule.tasks:
        schedule.hasher.assert_unchanged()
        if sha256_file(task.adapter_path) != task.adapter_sha256:
            raise SchedulerError(f"{task.key}: adapter changed after preflight")
        if sha256_file(schedule.config_path) != task.config_sha256:
            raise SchedulerError(f"{task.key}: config changed after preflight")
        helper_path = schedule.project_root / "main" / "protocol_a_preprocessing.py"
        if sha256_file(helper_path) != task.protocol_helper_sha256:
            raise SchedulerError(f"{task.key}: protocol helper changed after preflight")

        status_path = _status_path(schedule, task)
        log_path = _log_path(schedule, task)
        if resume:
            can_skip, reason, _output = resume_decision(task, status_path)
            if can_skip:
                skipped.append(task.key)
                print(f"[resume] {task.key}: verified output skipped", flush=True)
                continue
            print(f"[resume] {task.key}: rerun required ({reason})", flush=True)
            _remove_task_output(task, schedule.output_root)
        elif task.output_dir.exists() or status_path.exists():
            error = "existing output/status requires --resume"
            failed.append({"task": task.key, "error": error})
            if not continue_on_error:
                break
            continue

        task.output_dir.mkdir(parents=True, exist_ok=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        attempted += 1
        started = utc_now()
        record: dict[str, Any] = {
            "schema_version": 1,
            "protocol": "A",
            "task": task.key,
            "method": task.method,
            "dataset": task.dataset_name,
            "dataset_id": task.dataset_id,
            "fold": task.fold,
            "gpu_task": task.gpu,
            "execution_policy": "synchronous_serial",
            "status": "running",
            "started_utc": started,
            "log_path": _container_path(log_path, schedule.project_root),
            **task_provenance(task),
        }
        _atomic_write_json(status_path, record, schedule.output_root)
        print(f"[run] {task.key}", flush=True)
        error_text: str | None = None
        returncode: int | None = None
        output_record: dict[str, object] | None = None
        try:
            log_mode = "a" if resume and log_path.exists() else "w"
            with log_path.open(log_mode, encoding="utf-8") as log_handle:
                log_handle.write(f"\n[{started}] {' '.join(task.command)}\n")
                log_handle.flush()
                environment = os.environ.copy()
                environment["PYTHONPATH"] = "/workspace/GeneSPT/main:/workspace/GeneSPT/scripts"
                environment["PYTHONUNBUFFERED"] = "1"
                if task.gpu:
                    environment["CUDA_VISIBLE_DEVICES"] = "0"
                process = runner(
                    list(task.command),
                    cwd=CONTAINER_PROJECT_ROOT.as_posix(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                    env=environment,
                )
                returncode = int(process.returncode)
            if returncode != 0:
                error_text = f"adapter exited with return code {returncode}"
            else:
                output_record = validate_task_output(task)
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"

        record["ended_utc"] = utc_now()
        record["returncode"] = returncode
        if log_path.is_file():
            record["log_sha256"] = sha256_file(log_path)
            record["log_bytes"] = int(log_path.stat().st_size)
        if error_text is None and output_record is not None:
            record["status"] = "completed"
            record.update(output_record)
            completed.append(task.key)
        else:
            record["status"] = "failed"
            record["error"] = error_text or "unknown output validation failure"
            failed.append({"task": task.key, "error": record["error"]})
        _atomic_write_json(status_path, record, schedule.output_root)
        if error_text is not None and not continue_on_error:
            break

    return {
        "status": "ok" if not failed else "failed",
        "execution_policy": "synchronous_serial",
        "selected_tasks": len(schedule.tasks),
        "attempted": attempted,
        "completed": completed,
        "skipped": skipped,
        "failures": failed,
        "stopped_early": bool(failed and not continue_on_error),
    }


def preflight_summary(schedule: PreparedSchedule) -> dict[str, object]:
    return {
        "status": "ok",
        "mode": "preflight_only",
        "model_started": False,
        "protocol": "A",
        "datasets": list(schedule.datasets),
        "methods": list(schedule.methods),
        "folds": list(schedule.folds),
        "task_count": len(schedule.tasks),
        "gpu_task_count": sum(task.gpu for task in schedule.tasks),
        "execution_policy": "synchronous_serial",
        "container_python": CONTAINER_PYTHON,
        "container_project_root": CONTAINER_PROJECT_ROOT.as_posix(),
        "output_root": CONTAINER_PROJECT_ROOT.joinpath(*OUTPUT_RELATIVE.parts).as_posix(),
        "config_sha256": schedule.config_sha256,
        "protocol_helper_sha256": schedule.protocol_helper_sha256,
        "input_preparer_sha256": schedule.input_preparer_sha256,
        "adapter_sha256": dict(schedule.adapter_sha256),
    }


def _verify_container_runtime(schedule: PreparedSchedule) -> None:
    expected = Path(CONTAINER_PROJECT_ROOT.as_posix()).resolve(strict=True)
    if schedule.project_root.resolve(strict=True) != expected:
        raise SchedulerError("--run is allowed only from /workspace/GeneSPT in the container")
    if not Path(CONTAINER_PYTHON).is_file():
        raise SchedulerError(f"Container Python is missing: {CONTAINER_PYTHON}")
    if any(task.gpu for task in schedule.tasks):
        probe = subprocess.run(
            [
                CONTAINER_PYTHON,
                "-c",
                "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)",
            ],
            cwd=CONTAINER_PROJECT_ROOT.as_posix(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        if probe.returncode != 0:
            raise SchedulerError("CUDA is unavailable for the selected GPU baseline tasks")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight or explicitly run the five formal Protocol A baselines."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--datasets", "--dataset", nargs="+", default=None)
    parser.add_argument("--methods", "--method", nargs="+", default=None)
    parser.add_argument("--folds", "--fold", nargs="+", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    failures = parser.add_mutually_exclusive_group()
    failures.add_argument("--continue-on-error", action="store_true")
    failures.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        args.preflight_only = True
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if (args.continue_on_error or args.stop_on_error) and not args.run:
        parser.error("failure policy flags require --run")
    args.continue_on_error = bool(args.continue_on_error)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        schedule = prepare_schedule(
            args.config,
            project_root=GENESPT_ROOT,
            datasets=args.datasets,
            methods=args.methods,
            folds=args.folds,
        )
        summary = preflight_summary(schedule)
        print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        _verify_container_runtime(schedule)
        result = execute_schedule(
            schedule,
            resume=bool(args.resume),
            continue_on_error=bool(args.continue_on_error),
        )
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True), flush=True)
        return 0 if result["status"] == "ok" else 1
    except Exception as error:
        print(f"[failed] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
