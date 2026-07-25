#!/usr/bin/env python3
"""Strict six-dataset GeneSPT/GC scheduler for Protocol A.

The default mode is a read-only preflight.  ``--run`` is required before any
descriptor is built or any model process is started.  Every model job is one
dataset/fold pair so completion can be resumed only after exact provenance and
output validation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 2
PROTOCOL = "A"
REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_DATASETS = (
    ("Vis9A", "Vis9A_D7_spaim_effective4470", "primary"),
    ("HBC", "HBC_shared16112", "primary"),
    (
        "Cell2location",
        "Cell2location_mouse_brain_ST8059048_shared12819",
        "primary",
    ),
    (
        "seqFISH+",
        "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000",
        "cross_platform",
    ),
    ("MHPR", "MHPR_current_panel", "cross_platform"),
    ("MVC", "MVC_shared981", "cross_platform"),
)

CONTAINER_WORKSPACE_ROOT = PurePosixPath("/workspace")
CONTAINER_PROJECT_ROOT = CONTAINER_WORKSPACE_ROOT / "GeneSPT"
CONTAINER_PYTHON = PurePosixPath("/opt/conda/bin/python")
CONFIG_RELATIVE = PurePosixPath("configs/protocol_a_datasets.yaml")
CORE_RELATIVE = PurePosixPath(
    "main/run_predictable_spatial_program_folds012.py"
)
PREPARE_HELPER_RELATIVE = PurePosixPath(
    "scripts/protocol_a_full/prepare_protocol_a_inputs.py"
)
SCHEDULER_RELATIVE = PurePosixPath(
    "scripts/protocol_a_full/run_protocol_a_genespt.py"
)
REQUIREMENTS_RELATIVE = PurePosixPath("requirements.txt")
OUTPUT_RELATIVE = PurePosixPath(
    "results/protocol_a_full_rerun_20260711/genespt"
)
CONFIG_INPUT_OUTPUT_RELATIVE = PurePosixPath(
    "results/protocol_a_full_rerun_20260711/inputs"
)

BENCHMARK_MODE = "benchmark"
CONTROL_MODE = "primary_mechanism_controls"
BENCHMARK_PREFIX = "protocol_a_genespt"
CONTROL_PREFIX = "protocol_a_genespt_primary_controls"
DESCRIPTOR_FILENAME = "descriptors_pca32_nmf32.npz"
DESCRIPTOR_MANIFEST_FILENAME = "descriptor_manifest.json"
COMPLETION_MANIFEST_FILENAME = "completion_manifest.json"
FAILURE_MANIFEST_FILENAME = "run_failure.json"

CORE_ENVIRONMENT = {
    "MPLBACKEND": "Agg",
    "PYTHONPATH": "/workspace/GeneSPT/main:/workspace/GeneSPT/scripts",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}

EXPECTED_CONTROL_MODELS = (
    "predictable_spatial_program_shuffled_descriptor_control",
    "predictable_spatial_program_random_descriptor_control",
    "predictable_spatial_program_permuted_labels_control",
    "predictable_spatial_program_random_spatial_basis_control",
    "predictable_spatial_program_spot_permuted_spatial_program_control",
    "predictable_spatial_program_mean_coefficient_baseline_control",
)
BASE_MODEL = "gc_mlp_base"
FULL_MODEL = "predictable_spatial_program_selected_correct"

CHUNK_BYTES = 4 * 1024 * 1024
SHA256_LENGTH = 64


class PreflightError(ValueError):
    """Raised when a frozen Protocol A invariant is violated."""


class StaleCacheError(PreflightError):
    """Raised when an existing cache/output cannot be resumed exactly."""


@dataclass(frozen=True)
class Layout:
    """Host/container path mapping for the one mounted workspace."""

    project_root: Path
    workspace_root: Path
    container_workspace_root: PurePosixPath = CONTAINER_WORKSPACE_ROOT

    @property
    def output_root(self) -> Path:
        return self.project_root.joinpath(*OUTPUT_RELATIVE.parts)

    @property
    def container_project_root(self) -> PurePosixPath:
        return self.container_workspace_root / "GeneSPT"

    @property
    def container_output_root(self) -> PurePosixPath:
        return self.container_project_root.joinpath(*OUTPUT_RELATIVE.parts)

    def host_to_container(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        workspace = self.workspace_root.resolve(strict=True)
        try:
            relative = resolved.relative_to(workspace)
        except ValueError as error:
            raise PreflightError(
                f"Path is outside the mounted workspace: {resolved}"
            ) from error
        return str(
            self.container_workspace_root.joinpath(*relative.parts)
        )

    def assert_container_runtime(self) -> None:
        expected = Path(str(self.container_project_root)).resolve(strict=True)
        actual = self.project_root.resolve(strict=True)
        if actual != expected:
            raise PreflightError(
                "--run must execute inside the mounted container at "
                f"{self.container_project_root}; observed {actual}"
            )
        python = Path(str(CONTAINER_PYTHON))
        if not python.is_absolute() or not python.is_file():
            raise PreflightError(
                f"Frozen container Python is missing: {CONTAINER_PYTHON}"
            )


@dataclass(frozen=True)
class Artifact:
    host_path: Path
    container_path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.container_path,
            "bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TableInspection:
    artifact: Artifact
    header: tuple[str, ...]
    data_rows: int
    first_column: tuple[str, ...] | None


@dataclass(frozen=True)
class FoldContext:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    artifacts: Mapping[str, Artifact]

    @property
    def mask_dir_host(self) -> Path:
        return self.artifacts["train_mask"].host_path.parent

    @property
    def mask_dir_container(self) -> str:
        return str(PurePosixPath(self.artifacts["train_mask"].container_path).parent)


@dataclass(frozen=True)
class DatasetContext:
    spec: Mapping[str, Any]
    genes: tuple[str, ...]
    gene_axis_sha256: str
    artifacts: Mapping[str, Artifact]
    folds: Mapping[int, FoldContext]

    @property
    def name(self) -> str:
        return str(self.spec["name"])

    @property
    def dataset_id(self) -> str:
        return str(self.spec["dataset_id"])

    @property
    def role(self) -> str:
        return str(self.spec["role"])


@dataclass(frozen=True)
class DescriptorContext:
    status: str
    source_kind: str
    host_path: Path
    container_path: str
    provenance: Mapping[str, Any]

    @property
    def ready(self) -> bool:
        return self.status == "ready"


@dataclass(frozen=True)
class JobSpec:
    mode: str
    dataset_id: str
    fold: int
    output_dir_host: Path
    output_dir_container: str
    output_prefix: str
    command: tuple[str, ...]
    cwd: str
    environment: Mapping[str, str]
    provenance: Mapping[str, Any]
    job_signature_sha256: str

    def preview(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "fold": self.fold,
            "mode": self.mode,
            "output_dir": self.output_dir_container,
            "command": list(self.command),
            "command_sha256": canonical_json_sha256(list(self.command)),
            "job_signature_sha256": self.job_signature_sha256,
        }


@dataclass(frozen=True)
class FrozenDescriptorSpec:
    relative_path: PurePosixPath
    size_bytes: int
    sha256: str
    gene_axis_sha256: str


FROZEN_DESCRIPTOR_SPECS: Mapping[str, FrozenDescriptorSpec] = {
    "Vis9A_D7_spaim_effective4470": FrozenDescriptorSpec(
        PurePosixPath(
            "frozen_inputs/vis9a_psp_canonical_20260710/"
            "descriptors/descriptors_pca32_nmf32.npz"
        ),
        2_112_885,
        "f7833e0a485ac441f6815050171802cfb322207dd59749556d66b5868595b529",
        "f615ec76a9e0d1483c784ae5877d8a5785e2e032386dda6851ad19d98a4ff2a0",
    ),
    "Cell2location_mouse_brain_ST8059048_shared12819": FrozenDescriptorSpec(
        PurePosixPath(
            "frozen_inputs/cell2location_psp_canonical_20260710/"
            "descriptors/descriptors_pca32_nmf32.npz"
        ),
        6_178_636,
        "78786f7b5feeea688ae0e1723201b1164471933b2d693c86d510a587f704d964",
        "c2f0327d6213afa6d080cfb4044fb96f5552af5a7617409e846e2b0fd5cdf3d8",
    ),
}


def default_layout() -> Layout:
    project = Path(__file__).resolve().parents[2]
    return Layout(project_root=project, workspace_root=project.parent)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_payload_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_package_path(value: object, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PreflightError(f"{context} must be a non-empty path")
    if "\\" in value:
        raise PreflightError(f"{context} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise PreflightError(f"{context} is not package-relative: {value}")
    return path


def _format_fold_path(value: object, fold: int, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or value.count("{fold}") != 1:
        raise PreflightError(f"{context} must contain exactly one {{fold}}")
    try:
        formatted = value.format(fold=fold)
    except (KeyError, ValueError) as error:
        raise PreflightError(f"Invalid fold template for {context}") from error
    return _safe_package_path(formatted, context=context)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PreflightError(f"Could not read config: {path}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as error:
            raise PreflightError(
                f"{path} is not JSON-form YAML and PyYAML is unavailable"
            ) from error
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise PreflightError(f"Expected a mapping in {path}")
    return payload


def _artifact(
    path: Path,
    layout: Layout,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    observed_sha256: str | None = None,
) -> Artifact:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"Required file is missing: {path}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise PreflightError(f"Required input is not a regular file: {resolved}")
    size = resolved.stat().st_size
    if expected_size is not None and size != expected_size:
        raise PreflightError(
            f"Size mismatch for {resolved}: {size} != {expected_size}"
        )
    digest = observed_sha256 or sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise PreflightError(f"SHA256 mismatch for {resolved}")
    return Artifact(
        host_path=resolved,
        container_path=layout.host_to_container(resolved),
        size_bytes=size,
        sha256=digest,
    )


def _inspect_tsv(
    path: Path,
    layout: Layout,
    *,
    expected_size: int,
    expected_sha256: str,
    capture_first_column: bool = False,
) -> TableInspection:
    resolved = path.resolve(strict=True)
    if resolved.stat().st_size != expected_size:
        raise PreflightError(f"Size mismatch for {resolved}")
    digest = hashlib.sha256()
    header: tuple[str, ...] | None = None
    rows = 0
    first_column: list[str] | None = [] if capture_first_column else None
    with resolved.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            stripped = raw.rstrip(b"\r\n")
            if not stripped:
                raise PreflightError(
                    f"Blank row {line_number} in {layout.host_to_container(resolved)}"
                )
            if line_number == 1:
                try:
                    header = tuple(stripped.decode("utf-8-sig").split("\t"))
                except UnicodeDecodeError as error:
                    raise PreflightError(f"Invalid UTF-8 header: {resolved}") from error
                continue
            rows += 1
            if first_column is not None:
                raw_first = stripped.partition(b"\t")[0]
                try:
                    first_column.append(raw_first.decode("utf-8"))
                except UnicodeDecodeError as error:
                    raise PreflightError(
                        f"Invalid UTF-8 first column in {resolved} row {line_number}"
                    ) from error
    observed_hash = digest.hexdigest()
    if observed_hash != expected_sha256.lower():
        raise PreflightError(f"SHA256 mismatch for {resolved}")
    if header is None:
        raise PreflightError(f"Empty table: {resolved}")
    record = _artifact(
        resolved,
        layout,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        observed_sha256=observed_hash,
    )
    return TableInspection(
        artifact=record,
        header=header,
        data_rows=rows,
        first_column=tuple(first_column) if first_column is not None else None,
    )


def _record_from_archive(
    archive_root: Path,
    relative: PurePosixPath,
    checksums: Mapping[str, tuple[int, str]],
    layout: Layout,
) -> Artifact:
    key = relative.as_posix()
    if key not in checksums:
        raise PreflightError(f"Input is absent from archive checksum manifest: {key}")
    size, digest = checksums[key]
    path = archive_root.joinpath(*relative.parts)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(archive_root.resolve(strict=True))
    except ValueError as error:
        raise PreflightError(f"Archive path escapes root: {key}") from error
    return _artifact(
        resolved,
        layout,
        expected_size=size,
        expected_sha256=digest,
    )


def _inspect_from_archive(
    archive_root: Path,
    relative: PurePosixPath,
    checksums: Mapping[str, tuple[int, str]],
    layout: Layout,
    *,
    capture_first_column: bool = False,
) -> TableInspection:
    key = relative.as_posix()
    if key not in checksums:
        raise PreflightError(f"Input is absent from archive checksum manifest: {key}")
    size, digest = checksums[key]
    return _inspect_tsv(
        archive_root.joinpath(*relative.parts),
        layout,
        expected_size=size,
        expected_sha256=digest,
        capture_first_column=capture_first_column,
    )


def _load_archive_checksums(
    config: Mapping[str, Any],
    config_path: Path,
    layout: Layout,
) -> tuple[Path, Mapping[str, tuple[int, str]], Artifact]:
    archive = config.get("archive")
    if not isinstance(archive, dict):
        raise PreflightError("config.archive must be a mapping")
    configured_root = archive.get("root")
    if not isinstance(configured_root, str) or not configured_root:
        raise PreflightError("config.archive.root must be a non-empty path")
    archive_root = Path(configured_root)
    if not archive_root.is_absolute():
        archive_root = layout.project_root / archive_root
    archive_root = archive_root.resolve(strict=True)
    if not archive_root.is_dir():
        raise PreflightError(f"Archive root is not a directory: {archive_root}")
    layout.host_to_container(archive_root)

    manifest_rel = _safe_package_path(
        archive.get("checksum_manifest"), context="archive.checksum_manifest"
    )
    manifest_path = archive_root.joinpath(*manifest_rel.parts)
    expected_size = archive.get("checksum_manifest_bytes")
    expected_hash = archive.get("checksum_manifest_sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise PreflightError("archive.checksum_manifest_bytes must be positive")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != SHA256_LENGTH
    ):
        raise PreflightError("archive.checksum_manifest_sha256 is invalid")
    manifest_record = _artifact(
        manifest_path,
        layout,
        expected_size=expected_size,
        expected_sha256=expected_hash,
    )

    records: dict[str, tuple[int, str]] = {}
    with manifest_record.host_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["relative_path", "size_bytes", "sha256"]:
            raise PreflightError("Archive checksum manifest header is invalid")
        for row_number, row in enumerate(reader, start=2):
            relative = _safe_package_path(
                row.get("relative_path"), context=f"checksum row {row_number}"
            ).as_posix()
            if relative in records:
                raise PreflightError(f"Duplicate checksum entry: {relative}")
            try:
                size = int(str(row.get("size_bytes")))
            except ValueError as error:
                raise PreflightError(
                    f"Invalid size in checksum row {row_number}"
                ) from error
            digest = str(row.get("sha256", "")).lower()
            if size < 0 or len(digest) != SHA256_LENGTH:
                raise PreflightError(f"Invalid checksum row {row_number}")
            records[relative] = (size, digest)
    return archive_root, records, manifest_record


def _validate_config(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise PreflightError("Config must declare schema_version=1, protocol='A'")
    if config.get("folds") != list(REQUIRED_FOLDS):
        raise PreflightError(f"Config folds must be exactly {list(REQUIRED_FOLDS)}")
    if config.get("output_root") != CONFIG_INPUT_OUTPUT_RELATIVE.as_posix():
        raise PreflightError("Config input output_root is not the frozen Protocol A root")
    if config.get("zero_row_policy") != {
        "raw_all_gene_zero_rows": "error",
        "zero_train_library_rows": "record_and_zero",
    }:
        raise PreflightError("Config zero-row policy is not frozen Protocol A")
    datasets = config.get("datasets")
    if not isinstance(datasets, list):
        raise PreflightError("config.datasets must be a list")
    identity = tuple(
        (
            item.get("name") if isinstance(item, dict) else None,
            item.get("dataset_id") if isinstance(item, dict) else None,
            item.get("role") if isinstance(item, dict) else None,
        )
        for item in datasets
    )
    if identity != REQUIRED_DATASETS:
        raise PreflightError("Config does not contain the frozen six-dataset order")
    required = {
        "name",
        "dataset_id",
        "role",
        "raw_counts",
        "scrna_counts",
        "locations",
        "gene_names",
        "frozen_split",
        "train_mask",
        "val_mask",
        "test_mask",
        "expected_st_shape",
        "expected_locations_header",
        "scrna_orientation",
        "expected_scrna_shape",
        "expected_scrna_index_header",
    }
    for index, item in enumerate(datasets):
        if not isinstance(item, dict):
            raise PreflightError(f"datasets[{index}] must be a mapping")
        missing = sorted(required.difference(item))
        if missing:
            raise PreflightError(
                f"datasets[{index}] is missing: {', '.join(missing)}"
            )
        for key in ("raw_counts", "scrna_counts", "locations", "gene_names"):
            _safe_package_path(item[key], context=f"datasets[{index}].{key}")
        for key in ("frozen_split", "train_mask", "val_mask", "test_mask"):
            for fold in REQUIRED_FOLDS:
                _format_fold_path(
                    item[key], fold, context=f"datasets[{index}].{key}"
                )
        if item["scrna_orientation"] not in {
            "cells_by_genes",
            "genes_by_cells",
        }:
            raise PreflightError(f"datasets[{index}].scrna_orientation is invalid")
    return tuple(datasets)


def _load_gene_axis(
    record: Artifact,
    expected_gene_count: int,
) -> tuple[tuple[str, ...], str]:
    try:
        genes = tuple(
            record.host_path.read_text(encoding="utf-8-sig").splitlines()
        )
    except UnicodeDecodeError as error:
        raise PreflightError(f"Invalid gene-name encoding: {record.container_path}") from error
    if len(genes) != expected_gene_count:
        raise PreflightError(
            f"Gene-axis length mismatch for {record.container_path}: "
            f"{len(genes)} != {expected_gene_count}"
        )
    if any(not gene for gene in genes) or len(set(genes)) != len(genes):
        raise PreflightError(f"Gene axis is blank or non-unique: {record.container_path}")
    return genes, canonical_json_sha256(list(genes))


def _shape2(value: object, *, context: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise PreflightError(f"{context} must be two positive integers")
    return int(value[0]), int(value[1])


def _load_mask(record: Artifact, *, split: str, n_genes: int) -> np.ndarray:
    try:
        values = np.load(record.host_path, allow_pickle=False)
    except Exception as error:
        raise PreflightError(f"Could not load {split} mask: {record.container_path}") from error
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 1
        or not np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.bool_)
        or values.size == 0
    ):
        raise PreflightError(f"{split} mask must be a non-empty integer vector")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= n_genes):
        raise PreflightError(f"{split} mask contains out-of-range indices")
    if np.any(np.diff(values) <= 0):
        raise PreflightError(f"{split} mask must be strictly increasing")
    return values


def _validate_fold(
    dataset: Mapping[str, Any],
    fold: int,
    genes: Sequence[str],
    archive_root: Path,
    checksums: Mapping[str, tuple[int, str]],
    layout: Layout,
) -> FoldContext:
    artifact_records: dict[str, Artifact] = {}
    masks: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        key = f"{split}_mask"
        relative = _format_fold_path(
            dataset[key], fold, context=f"{dataset['dataset_id']}.{key}"
        )
        expected_name = f"fold{fold}_{split}_gene_idx.npy"
        if relative.name != expected_name:
            raise PreflightError(
                f"Core requires frozen mask basename {expected_name}: {relative}"
            )
        record = _record_from_archive(
            archive_root, relative, checksums, layout
        )
        artifact_records[key] = record
        masks[split] = _load_mask(record, split=split, n_genes=len(genes))

    mask_parents = {
        artifact_records[f"{split}_mask"].host_path.parent
        for split in ("train", "val", "test")
    }
    if len(mask_parents) != 1:
        raise PreflightError("Frozen train/val/test masks must share one directory")
    merged = np.concatenate([masks["train"], masks["val"], masks["test"]])
    if not np.array_equal(np.sort(merged), np.arange(len(genes), dtype=np.int64)):
        raise PreflightError(
            f"{dataset['dataset_id']} fold{fold}: masks overlap or do not cover all genes"
        )

    split_relative = _format_fold_path(
        dataset["frozen_split"],
        fold,
        context=f"{dataset['dataset_id']}.frozen_split",
    )
    split_record = _record_from_archive(
        archive_root, split_relative, checksums, layout
    )
    artifact_records["frozen_split"] = split_record
    try:
        split_payload = json.loads(split_record.host_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(
            f"Invalid frozen split JSON: {split_record.container_path}"
        ) from error
    if not isinstance(split_payload, dict):
        raise PreflightError("Frozen split must be a mapping")
    if (
        "dataset" in split_payload
        and split_payload["dataset"]
        not in {dataset["name"], dataset["dataset_id"]}
    ):
        raise PreflightError("Frozen split dataset identity mismatch")
    if split_payload.get("fold") != fold:
        raise PreflightError("Frozen split fold mismatch")
    for split in ("train", "val", "test"):
        expected_idx = masks[split].tolist()
        expected_genes = [genes[index] for index in expected_idx]
        if split_payload.get(f"{split}_gene_idx") != expected_idx:
            raise PreflightError(
                f"Frozen split {split} indices disagree with the explicit mask"
            )
        if split_payload.get(f"{split}_genes") != expected_genes:
            raise PreflightError(
                f"Frozen split {split} gene order disagrees with the dataset axis"
            )
    return FoldContext(
        fold=fold,
        train_idx=masks["train"],
        val_idx=masks["val"],
        test_idx=masks["test"],
        artifacts=artifact_records,
    )


def _preflight_dataset(
    dataset: Mapping[str, Any],
    selected_folds: Sequence[int],
    archive_root: Path,
    checksums: Mapping[str, tuple[int, str]],
    layout: Layout,
) -> DatasetContext:
    st_shape = _shape2(
        dataset["expected_st_shape"],
        context=f"{dataset['dataset_id']}.expected_st_shape",
    )
    gene_record = _record_from_archive(
        archive_root,
        _safe_package_path(dataset["gene_names"], context="gene_names"),
        checksums,
        layout,
    )
    genes, gene_axis_sha256 = _load_gene_axis(gene_record, st_shape[1])

    raw = _inspect_from_archive(
        archive_root,
        _safe_package_path(dataset["raw_counts"], context="raw_counts"),
        checksums,
        layout,
    )
    if raw.header != genes or raw.data_rows != st_shape[0]:
        raise PreflightError(
            f"{dataset['dataset_id']}: raw ST shape/order does not match the config"
        )

    locations = _inspect_from_archive(
        archive_root,
        _safe_package_path(dataset["locations"], context="locations"),
        checksums,
        layout,
    )
    expected_locations_header = tuple(dataset["expected_locations_header"])
    if (
        locations.header != expected_locations_header
        or locations.data_rows != st_shape[0]
    ):
        raise PreflightError(
            f"{dataset['dataset_id']}: locations shape/header mismatch"
        )

    orientation = str(dataset["scrna_orientation"])
    scrna_shape = _shape2(
        dataset["expected_scrna_shape"],
        context=f"{dataset['dataset_id']}.expected_scrna_shape",
    )
    scrna = _inspect_from_archive(
        archive_root,
        _safe_package_path(dataset["scrna_counts"], context="scrna_counts"),
        checksums,
        layout,
        capture_first_column=orientation == "genes_by_cells",
    )
    expected_index_header = str(dataset["expected_scrna_index_header"])
    if orientation == "cells_by_genes":
        expected_header = (expected_index_header, *genes)
        if (
            scrna.header != expected_header
            or scrna.data_rows != scrna_shape[0]
            or scrna_shape[1] != len(genes)
        ):
            raise PreflightError(
                f"{dataset['dataset_id']}: cells-by-genes scRNA shape/order mismatch"
            )
    else:
        if (
            not scrna.header
            or scrna.header[0] != expected_index_header
            or len(scrna.header) - 1 != scrna_shape[1]
            or scrna.data_rows != scrna_shape[0]
            or scrna.first_column is None
        ):
            raise PreflightError(
                f"{dataset['dataset_id']}: genes-by-cells scRNA shape/header mismatch"
            )
        source_genes = scrna.first_column
        if len(set(source_genes)) != len(source_genes):
            raise PreflightError("genes-by-cells scRNA gene identifiers are not unique")
        missing = sorted(set(genes).difference(source_genes))
        if missing:
            raise PreflightError(
                f"{dataset['dataset_id']}: scRNA is missing {len(missing)} dataset genes"
            )

    folds = {
        fold: _validate_fold(
            dataset,
            fold,
            genes,
            archive_root,
            checksums,
            layout,
        )
        for fold in selected_folds
    }
    if tuple(selected_folds) == REQUIRED_FOLDS:
        test_union = np.concatenate([folds[fold].test_idx for fold in REQUIRED_FOLDS])
        if not np.array_equal(
            np.sort(test_union), np.arange(len(genes), dtype=np.int64)
        ):
            raise PreflightError(
                f"{dataset['dataset_id']}: five test folds are not a complete partition"
            )

    return DatasetContext(
        spec=dataset,
        genes=genes,
        gene_axis_sha256=gene_axis_sha256,
        artifacts={
            "gene_names": gene_record,
            "raw_counts": raw.artifact,
            "scrna_counts": scrna.artifact,
            "locations": locations.artifact,
        },
        folds=folds,
    )


def _local_imports(path: Path, main_dir: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise PreflightError(f"Could not parse Python dependency: {path}") from error
    result: set[Path] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    for name in names:
        candidate = main_dir / f"{name}.py"
        if candidate.is_file():
            result.add(candidate.resolve(strict=True))
    return result


def collect_code_provenance(layout: Layout) -> Mapping[str, Any]:
    core = layout.project_root.joinpath(*CORE_RELATIVE.parts).resolve(strict=True)
    main_dir = layout.project_root / "main"
    pending = list(_local_imports(core, main_dir))
    dependencies: set[Path] = set()
    while pending:
        dependency = pending.pop()
        if dependency == core or dependency in dependencies:
            continue
        dependencies.add(dependency)
        pending.extend(_local_imports(dependency, main_dir).difference(dependencies))

    scheduler = layout.project_root.joinpath(*SCHEDULER_RELATIVE.parts)
    prepare_helper = layout.project_root.joinpath(*PREPARE_HELPER_RELATIVE.parts)
    requirements = layout.project_root.joinpath(*REQUIREMENTS_RELATIVE.parts)
    return {
        "core": _artifact(core, layout).as_dict(),
        "dependencies": [
            _artifact(path, layout).as_dict()
            for path in sorted(dependencies, key=lambda item: str(item))
        ],
        "helpers": [
            _artifact(path, layout).as_dict()
            for path in (scheduler, prepare_helper)
        ],
        "requirements": _artifact(requirements, layout).as_dict(),
    }


def _validate_descriptor_arrays(
    path: Path,
    *,
    n_genes: int,
) -> Mapping[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"pca32", "nmf32", "pca32_nmf32"}
            if set(payload.files) != required:
                raise PreflightError(
                    f"Descriptor cache keys must be exactly {sorted(required)}: {path}"
                )
            arrays = {key: payload[key] for key in sorted(required)}
    except PreflightError:
        raise
    except Exception as error:
        raise PreflightError(f"Could not load descriptor cache: {path}") from error

    expected_shapes = {
        "pca32": (n_genes, 32),
        "nmf32": (n_genes, 32),
        "pca32_nmf32": (n_genes, 64),
    }
    metadata: dict[str, Any] = {}
    for key, value in arrays.items():
        if value.shape != expected_shapes[key] or value.dtype != np.float32:
            raise PreflightError(
                f"Descriptor {key} shape/dtype mismatch: "
                f"{value.shape}/{value.dtype} != {expected_shapes[key]}/float32"
            )
        if not np.isfinite(value).all():
            raise PreflightError(f"Descriptor {key} contains non-finite values")
        metadata[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "payload_sha256": _array_payload_sha256(value),
        }
    concatenated = np.concatenate(
        [arrays["pca32"], arrays["nmf32"]], axis=1
    ).astype(np.float32, copy=False)
    if not np.array_equal(concatenated, arrays["pca32_nmf32"]):
        raise PreflightError("pca32_nmf32 is not the exact ordered concatenation")
    return metadata


def _descriptor_dir(
    layout: Layout,
    mode: str,
    dataset_id: str,
) -> Path:
    return layout.output_root / mode / dataset_id / "descriptor_cache"


def _descriptor_expected_provenance(
    dataset: DatasetContext,
    code_provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    dependency_by_name = {
        PurePosixPath(str(record["path"])).name: record
        for record in code_provenance["dependencies"]
    }
    builder = dependency_by_name.get(
        "run_strict_gene_conditioned_decoder_gate.py"
    )
    if builder is None:
        raise PreflightError("Descriptor builder dependency is absent from code provenance")
    return {
        "dataset_id": dataset.dataset_id,
        "gene_axis_sha256": dataset.gene_axis_sha256,
        "gene_count": len(dataset.genes),
        "source_artifacts": {
            "scrna_counts": dataset.artifacts["scrna_counts"].as_dict(),
            "gene_names": dataset.artifacts["gene_names"].as_dict(),
        },
        "algorithm": {
            "normalization": "log1p_cpm_target_sum_10000",
            "pca_dims": [32],
            "nmf_dims": [32],
            "seed": 42,
            "combined_order": ["pca32", "nmf32"],
        },
        "builder_dependency": builder,
        "scheduler_helper": code_provenance["helpers"][0],
    }


def _assert_output_path(path: Path, layout: Layout) -> None:
    root = layout.output_root.resolve(strict=False)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PreflightError(f"Output path escapes the frozen GeneSPT root: {path}") from error


def _validate_built_descriptor(
    dataset: DatasetContext,
    mode: str,
    code_provenance: Mapping[str, Any],
    layout: Layout,
) -> DescriptorContext:
    directory = _descriptor_dir(layout, mode, dataset.dataset_id)
    cache_path = directory / DESCRIPTOR_FILENAME
    manifest_path = directory / DESCRIPTOR_MANIFEST_FILENAME
    if not cache_path.is_file() or not manifest_path.is_file():
        raise StaleCacheError(
            f"Partial or old descriptor cache must be removed explicitly: "
            f"{layout.host_to_container(directory)}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StaleCacheError(f"Invalid descriptor manifest: {manifest_path}") from error
    expected_provenance = _descriptor_expected_provenance(dataset, code_provenance)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise StaleCacheError("Descriptor cache schema is stale")
    if manifest.get("status") != "complete":
        raise StaleCacheError("Descriptor cache is not complete")
    if manifest.get("provenance") != expected_provenance:
        raise StaleCacheError("Descriptor cache provenance/order binding is stale")
    cache_record = _artifact(cache_path, layout)
    if manifest.get("descriptor_file") != cache_record.as_dict():
        raise StaleCacheError("Descriptor cache file hash/size is stale")
    arrays = _validate_descriptor_arrays(cache_path, n_genes=len(dataset.genes))
    if manifest.get("arrays") != arrays:
        raise StaleCacheError("Descriptor cache array metadata is stale")
    manifest_record = _artifact(manifest_path, layout)
    return DescriptorContext(
        status="ready",
        source_kind="isolated_built",
        host_path=cache_path.resolve(strict=True),
        container_path=cache_record.container_path,
        provenance={
            "status": "ready",
            "source_kind": "isolated_built",
            "gene_axis_sha256": dataset.gene_axis_sha256,
            "order_validated": True,
            "descriptor_file": cache_record.as_dict(),
            "descriptor_manifest": manifest_record.as_dict(),
            "arrays": arrays,
        },
    )


def resolve_descriptor(
    dataset: DatasetContext,
    mode: str,
    code_provenance: Mapping[str, Any],
    layout: Layout,
) -> DescriptorContext:
    built_dir = _descriptor_dir(layout, mode, dataset.dataset_id)
    _assert_output_path(built_dir, layout)
    built_entries = list(built_dir.iterdir()) if built_dir.is_dir() else []

    frozen_spec = FROZEN_DESCRIPTOR_SPECS.get(dataset.dataset_id)
    if frozen_spec is not None:
        frozen_path = layout.project_root.joinpath(*frozen_spec.relative_path.parts)
        if frozen_path.exists():
            if built_entries:
                raise StaleCacheError(
                    "An isolated descriptor cache exists while the trusted frozen "
                    f"cache is available: {layout.host_to_container(built_dir)}"
                )
            if dataset.gene_axis_sha256 != frozen_spec.gene_axis_sha256:
                raise PreflightError(
                    f"Frozen descriptor order binding does not match {dataset.dataset_id}"
                )
            record = _artifact(
                frozen_path,
                layout,
                expected_size=frozen_spec.size_bytes,
                expected_sha256=frozen_spec.sha256,
            )
            arrays = _validate_descriptor_arrays(
                record.host_path, n_genes=len(dataset.genes)
            )
            return DescriptorContext(
                status="ready",
                source_kind="trusted_frozen",
                host_path=record.host_path,
                container_path=record.container_path,
                provenance={
                    "status": "ready",
                    "source_kind": "trusted_frozen",
                    "gene_axis_sha256": dataset.gene_axis_sha256,
                    "order_validated": True,
                    "descriptor_file": record.as_dict(),
                    "arrays": arrays,
                },
            )

    if built_entries:
        return _validate_built_descriptor(
            dataset, mode, code_provenance, layout
        )
    planned_path = built_dir / DESCRIPTOR_FILENAME
    return DescriptorContext(
        status="build_required",
        source_kind="isolated_build_required",
        host_path=planned_path,
        container_path=layout.host_to_container(planned_path),
        provenance={
            "status": "build_required",
            "source_kind": "isolated_build_required",
            "path": layout.host_to_container(planned_path),
            "gene_axis_sha256": dataset.gene_axis_sha256,
            "expected_shapes": {
                "pca32": [len(dataset.genes), 32],
                "nmf32": [len(dataset.genes), 32],
                "pca32_nmf32": [len(dataset.genes), 64],
            },
            "order_will_be": "explicit_dataset_gene_axis",
        },
    )


def _parse_numeric_row(raw: bytes, expected: int, *, context: str) -> np.ndarray:
    try:
        values = np.fromstring(raw.decode("ascii"), sep="\t", dtype=np.float32)
    except UnicodeDecodeError as error:
        raise PreflightError(f"Non-ASCII numeric values in {context}") from error
    if values.size != expected:
        raise PreflightError(
            f"Numeric width mismatch in {context}: {values.size} != {expected}"
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise PreflightError(f"Invalid count values in {context}")
    return values


def _load_scrna_for_descriptors(dataset: DatasetContext) -> np.ndarray:
    path = dataset.artifacts["scrna_counts"].host_path
    orientation = str(dataset.spec["scrna_orientation"])
    expected_shape = _shape2(
        dataset.spec["expected_scrna_shape"], context="expected_scrna_shape"
    )
    if orientation == "cells_by_genes":
        matrix = np.empty(expected_shape, dtype=np.float32)
        with path.open("rb") as handle:
            header = handle.readline().rstrip(b"\r\n").decode("utf-8-sig").split("\t")
            if tuple(header[1:]) != dataset.genes:
                raise PreflightError("scRNA header changed after preflight")
            for row_index, raw in enumerate(handle):
                if row_index >= expected_shape[0]:
                    raise PreflightError("scRNA row count grew after preflight")
                _cell_id, separator, numeric = raw.rstrip(b"\r\n").partition(b"\t")
                if not separator:
                    raise PreflightError("scRNA row is missing its cell identifier")
                matrix[row_index] = _parse_numeric_row(
                    numeric,
                    expected_shape[1],
                    context=f"{dataset.dataset_id} scRNA row {row_index + 2}",
                )
            if row_index + 1 != expected_shape[0]:
                raise PreflightError("scRNA row count changed after preflight")
        return matrix

    n_cells = expected_shape[1]
    matrix = np.empty((n_cells, len(dataset.genes)), dtype=np.float32)
    wanted = {gene: index for index, gene in enumerate(dataset.genes)}
    found: set[str] = set()
    with path.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n").decode("utf-8-sig").split("\t")
        if len(header) - 1 != n_cells:
            raise PreflightError("genes-by-cells scRNA header changed after preflight")
        for line_number, raw in enumerate(handle, start=2):
            raw_gene, separator, numeric = raw.rstrip(b"\r\n").partition(b"\t")
            if not separator:
                raise PreflightError(f"Malformed scRNA row {line_number}")
            try:
                gene = raw_gene.decode("utf-8")
            except UnicodeDecodeError as error:
                raise PreflightError(f"Invalid scRNA gene at row {line_number}") from error
            if gene not in wanted:
                continue
            if gene in found:
                raise PreflightError(f"Duplicate selected scRNA gene: {gene}")
            matrix[:, wanted[gene]] = _parse_numeric_row(
                numeric,
                n_cells,
                context=f"{dataset.dataset_id} scRNA gene {gene}",
            )
            found.add(gene)
    missing = sorted(set(wanted).difference(found))
    if missing:
        raise PreflightError(
            f"Descriptor build is missing {len(missing)} genes after preflight"
        )
    return matrix


def _write_atomic(path: Path, payload: bytes, layout: Layout) -> None:
    _assert_output_path(path, layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    _assert_output_path(temporary, layout)
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_descriptor_cache(
    dataset: DatasetContext,
    mode: str,
    code_provenance: Mapping[str, Any],
    layout: Layout,
) -> DescriptorContext:
    existing = resolve_descriptor(dataset, mode, code_provenance, layout)
    if existing.ready:
        return existing
    directory = _descriptor_dir(layout, mode, dataset.dataset_id)
    _assert_output_path(directory, layout)
    if directory.exists() and any(directory.iterdir()):
        raise StaleCacheError(f"Descriptor directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    main_dir = layout.project_root / "main"
    sys.path.insert(0, str(main_dir))
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from run_strict_gene_conditioned_decoder_gate import (  # type: ignore
            build_descriptors,
            log1p_cpm,
        )
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if sys.path and sys.path[0] == str(main_dir):
            sys.path.pop(0)

    counts = _load_scrna_for_descriptors(dataset)
    normalized = log1p_cpm(counts)
    descriptors = build_descriptors(
        normalized,
        pca_dims=[32],
        nmf_dims=[32],
        seed=42,
    )
    descriptors["pca32_nmf32"] = np.concatenate(
        [descriptors["pca32"], descriptors["nmf32"]], axis=1
    ).astype(np.float32)
    cache_path = directory / DESCRIPTOR_FILENAME
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **descriptors)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    del counts, normalized, descriptors

    arrays = _validate_descriptor_arrays(cache_path, n_genes=len(dataset.genes))
    descriptor_record = _artifact(cache_path, layout)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": PROTOCOL,
        "created_at_utc": _utc_now(),
        "provenance": _descriptor_expected_provenance(dataset, code_provenance),
        "descriptor_file": descriptor_record.as_dict(),
        "arrays": arrays,
    }
    _write_atomic(
        directory / DESCRIPTOR_MANIFEST_FILENAME,
        canonical_json_bytes(manifest),
        layout,
    )
    return _validate_built_descriptor(
        dataset, mode, code_provenance, layout
    )


def _job_output_dir(
    layout: Layout,
    mode: str,
    dataset_id: str,
    fold: int,
) -> Path:
    return layout.output_root / mode / dataset_id / f"fold{fold}"


def _build_command(
    dataset: DatasetContext,
    fold_context: FoldContext,
    descriptor: DescriptorContext,
    mode: str,
    layout: Layout,
) -> tuple[tuple[str, ...], str, str]:
    controls = mode == CONTROL_MODE
    output_prefix = CONTROL_PREFIX if controls else BENCHMARK_PREFIX
    output_dir = _job_output_dir(
        layout, mode, dataset.dataset_id, fold_context.fold
    )
    output_dir_container = layout.host_to_container(output_dir)
    command = (
        str(CONTAINER_PYTHON),
        str(layout.container_project_root.joinpath(*CORE_RELATIVE.parts)),
        "--folds",
        str(fold_context.fold),
        "--counts-path",
        dataset.artifacts["raw_counts"].container_path,
        "--scrna-counts-path",
        dataset.artifacts["scrna_counts"].container_path,
        "--locations-path",
        dataset.artifacts["locations"].container_path,
        "--mask-dir",
        fold_context.mask_dir_container,
        "--st-normalization-scope",
        "train_genes",
        "--out-dir",
        output_dir_container,
        "--steps",
        "800",
        "--batch-size",
        "65536",
        "--eval-every",
        "100",
        "--lr",
        "0.002",
        "--seed",
        "42",
        "--no-reuse-base",
        "--allow-train-base",
        "--descriptor-cache",
        descriptor.container_path,
        "--psp-descriptor",
        "pca32_nmf32",
        "--save-prediction-matrices",
        "--run-controls" if controls else "--no-run-controls",
        "--output-prefix",
        output_prefix,
    )
    path_options = {
        "--counts-path",
        "--scrna-counts-path",
        "--locations-path",
        "--mask-dir",
        "--out-dir",
        "--descriptor-cache",
    }
    for index, token in enumerate(command[:-1]):
        if token in path_options:
            value = PurePosixPath(command[index + 1])
            if not value.is_absolute():
                raise PreflightError(f"Core command path is not absolute: {value}")
    return command, output_prefix, output_dir_container


def build_job_spec(
    dataset: DatasetContext,
    fold: int,
    descriptor: DescriptorContext,
    mode: str,
    config_record: Artifact,
    archive_manifest_record: Artifact,
    code_provenance: Mapping[str, Any],
    layout: Layout,
) -> JobSpec:
    fold_context = dataset.folds[fold]
    command, output_prefix, output_dir_container = _build_command(
        dataset, fold_context, descriptor, mode, layout
    )
    inputs = {
        key: artifact.as_dict()
        for key, artifact in dataset.artifacts.items()
    }
    masks = {
        key: artifact.as_dict()
        for key, artifact in fold_context.artifacts.items()
    }
    provenance = {
        "config": config_record.as_dict(),
        "archive_checksum_manifest": archive_manifest_record.as_dict(),
        "inputs": inputs,
        "masks": masks,
        "gene_axis_sha256": dataset.gene_axis_sha256,
        "code": code_provenance,
        "descriptor": dict(descriptor.provenance),
    }
    environment = {
        **CORE_ENVIRONMENT,
        "MPLCONFIGDIR": str(
            PurePosixPath(output_dir_container) / ".matplotlib"
        ),
        "XDG_CACHE_HOME": str(
            PurePosixPath(output_dir_container) / ".cache"
        ),
    }
    signature_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "mode": mode,
        "dataset_id": dataset.dataset_id,
        "fold": fold,
        "command": list(command),
        "cwd": str(layout.container_project_root / "main"),
        "environment": environment,
        "provenance": provenance,
    }
    return JobSpec(
        mode=mode,
        dataset_id=dataset.dataset_id,
        fold=fold,
        output_dir_host=_job_output_dir(
            layout, mode, dataset.dataset_id, fold
        ),
        output_dir_container=output_dir_container,
        output_prefix=output_prefix,
        command=command,
        cwd=str(layout.container_project_root / "main"),
        environment=environment,
        provenance=provenance,
        job_signature_sha256=canonical_json_sha256(signature_payload),
    )


def _scalar(payload: Mapping[str, np.ndarray], key: str) -> object:
    if key not in payload:
        raise PreflightError(f"Prediction payload is missing scalar {key}")
    value = np.asarray(payload[key])
    if value.size != 1:
        raise PreflightError(f"Prediction scalar {key} is not scalar")
    return value.reshape(()).item()


def _require_prediction_array(
    payload: Mapping[str, np.ndarray],
    key: str,
    shape: tuple[int, int],
) -> np.ndarray:
    if key not in payload:
        raise PreflightError(f"Prediction payload is missing {key}")
    value = np.asarray(payload[key])
    if value.shape != shape or value.dtype != np.float32:
        raise PreflightError(
            f"Prediction {key} shape/dtype mismatch: {value.shape}/{value.dtype}"
        )
    if not np.isfinite(value).all():
        raise PreflightError(f"Prediction {key} contains non-finite values")
    return value


def _inspect_prediction_payload(
    path: Path,
    *,
    model: str,
    dataset: DatasetContext,
    fold_context: FoldContext,
    require_full_splits: bool,
) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    try:
        loaded = np.load(path, allow_pickle=True)
        payload = {key: loaded[key] for key in loaded.files}
        loaded.close()
    except Exception as error:
        raise PreflightError(f"Could not load prediction payload: {path}") from error
    fold = fold_context.fold
    if str(_scalar(payload, "model")) != model or int(_scalar(payload, "fold")) != fold:
        raise PreflightError(f"Prediction identity mismatch: {path}")
    if str(_scalar(payload, "psp_descriptor")) != "pca32_nmf32":
        raise PreflightError("Prediction PSP descriptor is not frozen")
    if str(_scalar(payload, "posthoc_calibration")) != "none":
        raise PreflightError("Prediction payload has post-hoc calibration")
    if str(_scalar(payload, "readout")) != "identity":
        raise PreflightError("Prediction payload readout is not identity")

    for key, expected in (
        ("train_gene_idx", fold_context.train_idx),
        ("val_gene_idx", fold_context.val_idx),
        ("test_gene_idx", fold_context.test_idx),
    ):
        observed = np.asarray(payload.get(key))
        if observed.dtype != np.int64 or not np.array_equal(observed, expected):
            raise PreflightError(f"Prediction payload {key} is not the frozen mask")
    expected_test_genes = np.asarray(
        [dataset.genes[int(index)] for index in fold_context.test_idx],
        dtype=object,
    )
    if not np.array_equal(np.asarray(payload.get("test_genes")), expected_test_genes):
        raise PreflightError("Prediction test gene order mismatch")

    n_spots = int(dataset.spec["expected_st_shape"][0])
    shapes = {
        "train": (n_spots, int(fold_context.train_idx.size)),
        "val": (n_spots, int(fold_context.val_idx.size)),
        "test": (n_spots, int(fold_context.test_idx.size)),
    }
    test_prediction = _require_prediction_array(payload, "prediction", shapes["test"])
    base_test = _require_prediction_array(payload, "base_prediction", shapes["test"])
    if model == BASE_MODEL and not np.array_equal(test_prediction, base_test):
        raise PreflightError("GC legacy prediction differs from base_prediction")

    split_metadata: dict[str, Any] = {}
    if require_full_splits:
        base_arrays = {
            split: _require_prediction_array(
                payload, f"base_prediction_{split}", shapes[split]
            )
            for split in ("train", "val", "test")
        }
        if not np.array_equal(base_arrays["test"], base_test):
            raise PreflightError("GC test payload aliases disagree")
        split_metadata["gc"] = {
            split: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for split, value in base_arrays.items()
        }
        if model == FULL_MODEL:
            full_arrays = {
                split: _require_prediction_array(
                    payload, f"selected_prediction_{split}", shapes[split]
                )
                for split in ("train", "val", "test")
            }
            if not np.array_equal(full_arrays["test"], test_prediction):
                raise PreflightError("Full test payload aliases disagree")
            if str(_scalar(payload, "selected_rule_frozen_from_split")) != "validation":
                raise PreflightError("Full prediction rule was not frozen on validation")
            if (
                str(_scalar(payload, "selected_train_coefficient_source"))
                != "ridge_descriptor_prediction_on_train_genes"
            ):
                raise PreflightError("Full train coefficient source is not frozen")
            split_metadata["full"] = {
                split: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for split, value in full_arrays.items()
            }
    return (
        {
            "model": model,
            "fold": fold,
            "legacy_test_shape": list(test_prediction.shape),
            "split_payload": split_metadata,
        },
        payload,
    )


def _validate_core_config(
    path: Path,
    job: JobSpec,
    dataset: DatasetContext,
    fold_context: FoldContext,
    descriptor: DescriptorContext,
) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"Invalid core run config: {path}") from error
    expected = {
        "folds": [fold_context.fold],
        "seed": 42,
        "counts_path": dataset.artifacts["raw_counts"].container_path,
        "scrna_counts_path": dataset.artifacts["scrna_counts"].container_path,
        "locations_path": dataset.artifacts["locations"].container_path,
        "mask_dir": fold_context.mask_dir_container,
        "out_dir": job.output_dir_container,
        "descriptor_cache": descriptor.container_path,
        "output_prefix": job.output_prefix,
        "st_normalization_scope": "train_genes",
        "psp_descriptor": "pca32_nmf32",
        "posthoc_calibration": "none",
        "readout": "identity",
        "prediction_matrices_saved": True,
        "run_controls": job.mode == CONTROL_MODE,
        "allow_train_base": True,
        "base_cache_dir": None,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PreflightError(
                f"Core run config mismatch for {key}: {payload.get(key)!r} != {value!r}"
            )
    normalizations = payload.get("st_normalization_by_fold")
    if not isinstance(normalizations, list) or len(normalizations) != 1:
        raise PreflightError("Core normalization audit must contain exactly one fold")
    normalization = normalizations[0]
    if (
        normalization.get("fold") != fold_context.fold
        or normalization.get("scope") != "train_genes"
        or normalization.get("denominator_gene_count")
        != int(fold_context.train_idx.size)
        or normalization.get("eligible_for_strict_primary") is not True
    ):
        raise PreflightError("Core train-gene normalization metadata is invalid")
    expected_mask_paths = {
        split: fold_context.artifacts[f"{split}_mask"].container_path
        for split in ("train", "val", "test")
    }
    if normalization.get("mask_paths") != expected_mask_paths:
        raise PreflightError("Core normalization mask paths are not the frozen masks")
    split_payload = payload.get("prediction_matrix_split_payload")
    required_split_flags = {
        "base_train_saved_when_available": True,
        "base_val_saved_for_core_models": True,
        "selected_train_saved_when_base_train_available": True,
        "selected_val_saved": True,
        "selected_rule_frozen_from_split": "validation",
        "selected_train_coefficient_source": "ridge_descriptor_prediction_on_train_genes",
    }
    if not isinstance(split_payload, dict) or any(
        split_payload.get(key) != value
        for key, value in required_split_flags.items()
    ):
        raise PreflightError("Core prediction split contract is incomplete")
    return {
        "path": str(PurePosixPath(job.output_dir_container) / path.name),
        "validated_fields": sorted(expected),
        "train_gene_denominator_count": int(fold_context.train_idx.size),
    }


def _collect_output_records(job: JobSpec, layout: Layout) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if not job.output_dir_host.is_dir():
        raise PreflightError(f"Job output directory is missing: {job.output_dir_container}")
    for path in sorted(job.output_dir_host.rglob("*"), key=lambda item: str(item)):
        if path.name == COMPLETION_MANIFEST_FILENAME:
            continue
        if path.is_symlink():
            raise PreflightError(f"Output symlink is forbidden: {path}")
        if path.is_file():
            records.append(_artifact(path, layout).as_dict())
    if not records:
        raise PreflightError(f"Job produced no output files: {job.output_dir_container}")
    return records


def validate_core_outputs(
    job: JobSpec,
    dataset: DatasetContext,
    descriptor: DescriptorContext,
    layout: Layout,
) -> Mapping[str, Any]:
    fold_context = dataset.folds[job.fold]
    root = job.output_dir_host
    config_path = root / f"{job.output_prefix}_run_config.json"
    config_metadata = _validate_core_config(
        config_path, job, dataset, fold_context, descriptor
    )
    prediction_root = root / f"{job.output_prefix}_prediction_matrices"
    expected_models = {BASE_MODEL, FULL_MODEL}
    if job.mode == CONTROL_MODE:
        expected_models.update(EXPECTED_CONTROL_MODELS)
    observed_models = (
        {path.name for path in prediction_root.iterdir() if path.is_dir()}
        if prediction_root.is_dir()
        else set()
    )
    if observed_models != expected_models:
        raise PreflightError(
            "Prediction model directories do not match the mode contract: "
            f"{sorted(observed_models)} != {sorted(expected_models)}"
        )

    payload_metadata: dict[str, Any] = {}
    loaded_payloads: dict[str, Mapping[str, np.ndarray]] = {}
    for model in sorted(expected_models):
        path = prediction_root / model / f"fold{job.fold}" / "prediction.npz"
        metadata, payload = _inspect_prediction_payload(
            path,
            model=model,
            dataset=dataset,
            fold_context=fold_context,
            require_full_splits=model in {BASE_MODEL, FULL_MODEL},
        )
        metadata = dict(metadata)
        metadata["path"] = layout.host_to_container(path)
        payload_metadata[model] = metadata
        loaded_payloads[model] = payload

    base = loaded_payloads[BASE_MODEL]
    full = loaded_payloads[FULL_MODEL]
    for split in ("train", "val", "test"):
        if not np.array_equal(
            base[f"base_prediction_{split}"],
            full[f"base_prediction_{split}"],
        ):
            raise PreflightError(
                f"GC {split} predictions disagree between GC and full payloads"
            )
    payload_metadata["canonical_full_gc_payload"] = {
        "path": payload_metadata[FULL_MODEL]["path"],
        "full_keys": [
            "selected_prediction_train",
            "selected_prediction_val",
            "selected_prediction_test",
        ],
        "gc_keys": [
            "base_prediction_train",
            "base_prediction_val",
            "base_prediction_test",
        ],
        "train_validation_test_complete": True,
    }
    return {
        "core_run_config": config_metadata,
        "prediction_payloads": payload_metadata,
        "files": _collect_output_records(job, layout),
    }


def _completion_path(job: JobSpec) -> Path:
    return job.output_dir_host / COMPLETION_MANIFEST_FILENAME


def _expected_completion_identity(job: JobSpec) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": PROTOCOL,
        "mode": job.mode,
        "dataset_id": job.dataset_id,
        "fold": job.fold,
        "command": list(job.command),
        "command_sha256": canonical_json_sha256(list(job.command)),
        "cwd": job.cwd,
        "environment": dict(job.environment),
        "provenance": job.provenance,
        "job_signature_sha256": job.job_signature_sha256,
    }


def write_completion_manifest(
    job: JobSpec,
    dataset: DatasetContext,
    descriptor: DescriptorContext,
    layout: Layout,
) -> Mapping[str, Any]:
    outputs = validate_core_outputs(job, dataset, descriptor, layout)
    manifest = {
        **_expected_completion_identity(job),
        "completed_at_utc": _utc_now(),
        "outputs": outputs,
    }
    _write_atomic(
        _completion_path(job), canonical_json_bytes(manifest), layout
    )
    return manifest


def validate_resume(
    job: JobSpec,
    dataset: DatasetContext,
    descriptor: DescriptorContext,
    layout: Layout,
) -> Mapping[str, Any]:
    path = _completion_path(job)
    if not path.is_file():
        raise StaleCacheError(
            f"Existing job output has no completion manifest: {job.output_dir_container}"
        )
    try:
        observed = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StaleCacheError(f"Invalid completion manifest: {path}") from error
    expected_identity = _expected_completion_identity(job)
    for key, expected in expected_identity.items():
        if observed.get(key) != expected:
            raise StaleCacheError(
                f"Resume rejected: stale {key} for {job.dataset_id} fold{job.fold}"
            )
    current_outputs = validate_core_outputs(job, dataset, descriptor, layout)
    if observed.get("outputs") != current_outputs:
        raise StaleCacheError(
            f"Resume rejected: output metadata/hash changed for "
            f"{job.dataset_id} fold{job.fold}"
        )
    return observed


def _select_datasets(
    datasets: Sequence[Mapping[str, Any]],
    requested: Sequence[str] | None,
    *,
    controls: bool,
) -> tuple[Mapping[str, Any], ...]:
    if requested:
        aliases: dict[str, Mapping[str, Any]] = {}
        for dataset in datasets:
            aliases[str(dataset["name"]).casefold()] = dataset
            aliases[str(dataset["dataset_id"]).casefold()] = dataset
        selected: list[Mapping[str, Any]] = []
        for value in requested:
            key = value.casefold()
            if key not in aliases:
                raise PreflightError(f"Unknown dataset selector: {value}")
            dataset = aliases[key]
            if dataset not in selected:
                selected.append(dataset)
        selected.sort(key=lambda item: datasets.index(item))
    else:
        selected = [
            dataset
            for dataset in datasets
            if not controls or dataset["role"] == "primary"
        ]
    if controls:
        cross_platform = [
            str(dataset["name"])
            for dataset in selected
            if dataset["role"] != "primary"
        ]
        if cross_platform:
            raise PreflightError(
                "Primary mechanism controls cannot target cross-platform datasets: "
                + ", ".join(cross_platform)
            )
    return tuple(selected)


def _select_folds(requested: Sequence[int] | None) -> tuple[int, ...]:
    if not requested:
        return REQUIRED_FOLDS
    invalid = sorted(set(requested).difference(REQUIRED_FOLDS))
    if invalid:
        raise PreflightError(f"Fold selectors are invalid: {invalid}")
    return tuple(fold for fold in REQUIRED_FOLDS if fold in set(requested))


def _job_state(
    job: JobSpec,
    dataset: DatasetContext,
    descriptor: DescriptorContext,
    layout: Layout,
) -> str:
    if not job.output_dir_host.exists():
        return "planned"
    if not job.output_dir_host.is_dir():
        raise StaleCacheError(f"Job output path is not a directory: {job.output_dir_container}")
    validate_resume(job, dataset, descriptor, layout)
    return "complete_valid"


def preflight_protocol_a(
    *,
    layout: Layout | None = None,
    datasets: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
    primary_mechanism_controls: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, DatasetContext], Mapping[str, DescriptorContext], Mapping[str, JobSpec]]:
    active_layout = layout or default_layout()
    config_path = active_layout.project_root.joinpath(*CONFIG_RELATIVE.parts)
    config = _load_mapping(config_path)
    configured_datasets = _validate_config(config)
    selected = _select_datasets(
        configured_datasets,
        datasets,
        controls=primary_mechanism_controls,
    )
    selected_folds = _select_folds(folds)
    config_record = _artifact(config_path, active_layout)
    archive_root, checksums, archive_manifest_record = _load_archive_checksums(
        config, config_path, active_layout
    )
    code_provenance = collect_code_provenance(active_layout)
    mode = CONTROL_MODE if primary_mechanism_controls else BENCHMARK_MODE

    contexts: dict[str, DatasetContext] = {}
    descriptors_by_dataset: dict[str, DescriptorContext] = {}
    jobs: dict[str, JobSpec] = {}
    dataset_reports: list[Mapping[str, Any]] = []
    for dataset_spec in selected:
        dataset_id = str(dataset_spec["dataset_id"])
        if progress:
            progress(f"[preflight] {dataset_id}: inputs, axes, frozen masks")
        context = _preflight_dataset(
            dataset_spec,
            selected_folds,
            archive_root,
            checksums,
            active_layout,
        )
        descriptor = resolve_descriptor(
            context, mode, code_provenance, active_layout
        )
        contexts[dataset_id] = context
        descriptors_by_dataset[dataset_id] = descriptor
        fold_reports: list[Mapping[str, Any]] = []
        for fold in selected_folds:
            job = build_job_spec(
                context,
                fold,
                descriptor,
                mode,
                config_record,
                archive_manifest_record,
                code_provenance,
                active_layout,
            )
            key = f"{dataset_id}/fold{fold}"
            jobs[key] = job
            state = _job_state(job, context, descriptor, active_layout)
            fold_reports.append({**job.preview(), "state": state})
        dataset_reports.append(
            {
                "name": context.name,
                "dataset_id": dataset_id,
                "role": context.role,
                "gene_count": len(context.genes),
                "gene_axis_sha256": context.gene_axis_sha256,
                "descriptor": dict(descriptor.provenance),
                "folds": fold_reports,
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "protocol": PROTOCOL,
        "operation": "preflight_only",
        "model_started": False,
        "mode": mode,
        "output_root": str(active_layout.container_output_root),
        "dataset_count": len(contexts),
        "job_count": len(jobs),
        "folds": list(selected_folds),
        "config": config_record.as_dict(),
        "archive_checksum_manifest": archive_manifest_record.as_dict(),
        "code": code_provenance,
        "datasets": dataset_reports,
    }
    return report, contexts, descriptors_by_dataset, jobs


def _default_executor(
    command: Sequence[str],
    *,
    cwd: str,
    environment: Mapping[str, str],
) -> int:
    env = os.environ.copy()
    env.update(environment)
    completed = subprocess.run(list(command), cwd=cwd, env=env, check=False)
    return int(completed.returncode)


def run_protocol_a(
    *,
    layout: Layout | None = None,
    datasets: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
    primary_mechanism_controls: bool = False,
    resume: bool = False,
    executor: Callable[..., int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Mapping[str, Any]:
    active_layout = layout or default_layout()
    preflight, contexts, descriptors, jobs = preflight_protocol_a(
        layout=active_layout,
        datasets=datasets,
        folds=folds,
        primary_mechanism_controls=primary_mechanism_controls,
        progress=progress,
    )
    active_layout.assert_container_runtime()
    mode = str(preflight["mode"])
    runner = executor or _default_executor
    config_record = Artifact(
        host_path=active_layout.project_root.joinpath(*CONFIG_RELATIVE.parts),
        container_path=str(preflight["config"]["path"]),
        size_bytes=int(preflight["config"]["bytes"]),
        sha256=str(preflight["config"]["sha256"]),
    )
    archive_manifest_record = Artifact(
        host_path=Path("."),
        container_path=str(preflight["archive_checksum_manifest"]["path"]),
        size_bytes=int(preflight["archive_checksum_manifest"]["bytes"]),
        sha256=str(preflight["archive_checksum_manifest"]["sha256"]),
    )
    code_provenance = preflight["code"]

    completed_jobs: list[str] = []
    skipped_jobs: list[str] = []
    for dataset_report in preflight["datasets"]:
        dataset_id = str(dataset_report["dataset_id"])
        context = contexts[dataset_id]
        descriptor = descriptors[dataset_id]
        if not descriptor.ready:
            if progress:
                progress(f"[descriptor] {dataset_id}: building isolated cache")
            descriptor = build_descriptor_cache(
                context, mode, code_provenance, active_layout
            )
            descriptors[dataset_id] = descriptor

        for fold in preflight["folds"]:
            fold = int(fold)
            key = f"{dataset_id}/fold{fold}"
            job = build_job_spec(
                context,
                fold,
                descriptor,
                mode,
                config_record,
                archive_manifest_record,
                code_provenance,
                active_layout,
            )
            if job.output_dir_host.exists():
                validate_resume(job, context, descriptor, active_layout)
                if not resume:
                    raise FileExistsError(
                        "Validated output already exists; pass --resume to skip it: "
                        f"{job.output_dir_container}"
                    )
                skipped_jobs.append(key)
                if progress:
                    progress(f"[resume] {key}: exact completion validated, skipped")
                continue

            _assert_output_path(job.output_dir_host, active_layout)
            job.output_dir_host.mkdir(parents=True, exist_ok=False)
            if progress:
                progress(f"[run] {key}: starting core process")
            return_code = runner(
                job.command,
                cwd=job.cwd,
                environment=job.environment,
            )
            if return_code != 0:
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "dataset_id": dataset_id,
                    "fold": fold,
                    "return_code": return_code,
                    "command": list(job.command),
                    "failed_at_utc": _utc_now(),
                }
                _write_atomic(
                    job.output_dir_host / FAILURE_MANIFEST_FILENAME,
                    canonical_json_bytes(failure),
                    active_layout,
                )
                raise RuntimeError(
                    f"Core process failed for {key} with return code {return_code}"
                )
            write_completion_manifest(
                job, context, descriptor, active_layout
            )
            completed_jobs.append(key)
            if progress:
                progress(f"[complete] {key}: outputs and provenance validated")

    return {
        **preflight,
        "operation": "run",
        "model_started": bool(completed_jobs),
        "completed_jobs": completed_jobs,
        "resumed_jobs": skipped_jobs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or explicitly run the strict six-dataset GeneSPT/GC "
            "Protocol A schedule."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start model jobs after preflight. Without this flag nothing is written.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only jobs whose command, provenance, hashes, and outputs validate exactly.",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        action="extend",
        nargs="+",
        default=None,
        help="Dataset name or frozen dataset_id; repeat or list values.",
    )
    parser.add_argument(
        "--fold",
        "--folds",
        dest="folds",
        action="extend",
        nargs="+",
        type=int,
        default=None,
        help="Fold(s) from 0 through 4; repeat or list values.",
    )
    parser.add_argument(
        "--primary-mechanism-controls",
        "--run-primary-mechanism-controls",
        action="store_true",
        help=(
            "Run the primary-dataset mechanism controls in their isolated output tree. "
            "Benchmark mode never runs controls."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    try:
        if args.run:
            report = run_protocol_a(
                datasets=args.datasets,
                folds=args.folds,
                primary_mechanism_controls=bool(
                    args.primary_mechanism_controls
                ),
                resume=bool(args.resume),
                progress=progress,
            )
        else:
            report, _contexts, _descriptors, _jobs = preflight_protocol_a(
                datasets=args.datasets,
                folds=args.folds,
                primary_mechanism_controls=bool(
                    args.primary_mechanism_controls
                ),
                progress=progress,
            )
    except Exception as error:
        print(
            f"[failed] {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    summary = {
        "status": report["status"],
        "operation": report["operation"],
        "mode": report["mode"],
        "datasets": report["dataset_count"],
        "jobs": report["job_count"],
        "folds": report["folds"],
        "model_started": report["model_started"],
        "output_root": report["output_root"],
    }
    if args.run:
        summary["completed_jobs"] = len(report["completed_jobs"])
        summary["resumed_jobs"] = len(report["resumed_jobs"])
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
