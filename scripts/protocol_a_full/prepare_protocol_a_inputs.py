#!/usr/bin/env python3
"""Fail-closed input preparation for the six-dataset Protocol A rerun.

The default operation is a read-only preflight.  Full fold-specific truth
matrices and their companion metadata are written only when
``--materialize-truth`` is supplied.  This module never starts a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import string
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


GENESPT_ROOT = Path(__file__).resolve().parents[2]
if str(GENESPT_ROOT) not in sys.path:
    sys.path.insert(0, str(GENESPT_ROOT))

from main.protocol_a_preprocessing import (  # noqa: E402
    PROTOCOL_A_POLICY,
    PROTOCOL_A_TARGET_SUM,
    ZERO_TRAIN_LIBRARY_POLICY,
    normalize_st_protocol_a,
    validate_gene_splits,
)


DEFAULT_CONFIG = GENESPT_ROOT / "configs" / "protocol_a_datasets.yaml"
OUTPUT_RELATIVE = PurePosixPath(
    "results/protocol_a_full_rerun_20260711/inputs"
)
REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_DATASETS = (
    ("Vis9A", "Vis9A_D7_spaim_effective4470"),
    ("HBC", "HBC_shared16112"),
    (
        "Cell2location",
        "Cell2location_mouse_brain_ST8059048_shared12819",
    ),
    (
        "seqFISH+",
        "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000",
    ),
    ("MHPR", "MHPR_current_panel"),
    ("MVC", "MVC_shared981"),
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CHUNK_BYTES = 1024 * 1024
ARRAY_ROW_CHUNK = 256


class PreflightError(ValueError):
    """Raised when an input violates a frozen Protocol A invariant."""


@dataclass(frozen=True)
class ArtifactRecord:
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.relative_path,
            "bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TableInspection:
    artifact: ArtifactRecord
    header: tuple[str, ...]
    data_rows: int
    first_column: tuple[str, ...] | None


@dataclass(frozen=True)
class FoldInputs:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    artifacts: Mapping[str, ArtifactRecord]


@dataclass(frozen=True)
class DatasetInputs:
    config: Mapping[str, Any]
    genes: tuple[str, ...]
    gene_axis_sha256: str
    artifacts: Mapping[str, ArtifactRecord]
    folds: tuple[FoldInputs, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    compact = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(compact).hexdigest()


def _load_serialized_mapping(path: Path) -> Mapping[str, Any]:
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
        try:
            payload = yaml.safe_load(text)
        except Exception as error:
            raise PreflightError(f"Could not parse YAML config: {path}") from error
    if not isinstance(payload, dict):
        raise PreflightError("Protocol A config must contain one mapping")
    return payload


def _require_keys(
    payload: Mapping[str, Any],
    required: Sequence[str],
    *,
    context: str,
    optional: Sequence[str] = (),
) -> None:
    missing = sorted(set(required).difference(payload))
    unknown = sorted(set(payload).difference((*required, *optional)))
    if missing:
        raise PreflightError(f"{context} is missing key(s): {', '.join(missing)}")
    if unknown:
        raise PreflightError(f"{context} has unknown key(s): {', '.join(unknown)}")


def _positive_shape(value: object, *, context: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise PreflightError(f"{context} must be a two-integer list")
    shape: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise PreflightError(f"{context} must contain integers")
        if dimension <= 0:
            raise PreflightError(f"{context} dimensions must be positive")
        shape.append(dimension)
    return shape[0], shape[1]


def _package_relative(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PreflightError(f"{context} must be a non-empty package-relative path")
    if "\\" in value:
        raise PreflightError(f"{context} must use package POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise PreflightError(f"{context} is not a safe package-relative path: {value}")
    return path.as_posix()


def _validate_fold_template(value: object, *, context: str) -> str:
    template = _package_relative(value, context=context)
    fields = [
        field
        for _, field, _, _ in string.Formatter().parse(template)
        if field is not None
    ]
    if fields != ["fold"]:
        raise PreflightError(f"{context} must contain exactly one {{fold}} field")
    for fold in REQUIRED_FOLDS:
        _package_relative(template.format(fold=fold), context=context)
    return template


def _validated_config(config_path: Path) -> Mapping[str, Any]:
    config = _load_serialized_mapping(config_path)
    _require_keys(
        config,
        (
            "schema_version",
            "protocol",
            "archive",
            "output_root",
            "folds",
            "zero_row_policy",
            "datasets",
        ),
        context="config",
    )
    if config["schema_version"] != 1 or config["protocol"] != "A":
        raise PreflightError("Config must declare schema_version=1 and protocol='A'")
    if config["output_root"] != OUTPUT_RELATIVE.as_posix():
        raise PreflightError(
            f"output_root is frozen to {OUTPUT_RELATIVE.as_posix()}"
        )
    if config["folds"] != list(REQUIRED_FOLDS):
        raise PreflightError(f"folds must be exactly {list(REQUIRED_FOLDS)}")

    archive = config["archive"]
    if not isinstance(archive, dict):
        raise PreflightError("archive must be a mapping")
    _require_keys(
        archive,
        (
            "root",
            "checksum_manifest",
            "checksum_manifest_bytes",
            "checksum_manifest_sha256",
        ),
        context="archive",
    )
    if not isinstance(archive["root"], str) or not archive["root"]:
        raise PreflightError("archive.root must be a non-empty path")
    _package_relative(
        archive["checksum_manifest"], context="archive.checksum_manifest"
    )
    if (
        isinstance(archive["checksum_manifest_bytes"], bool)
        or not isinstance(archive["checksum_manifest_bytes"], int)
        or archive["checksum_manifest_bytes"] <= 0
    ):
        raise PreflightError("archive.checksum_manifest_bytes must be positive")
    manifest_hash = archive["checksum_manifest_sha256"]
    if not isinstance(manifest_hash, str) or not SHA256_PATTERN.fullmatch(
        manifest_hash
    ):
        raise PreflightError("archive.checksum_manifest_sha256 is invalid")

    zero_policy = config["zero_row_policy"]
    if not isinstance(zero_policy, dict):
        raise PreflightError("zero_row_policy must be a mapping")
    _require_keys(
        zero_policy,
        ("raw_all_gene_zero_rows", "zero_train_library_rows"),
        context="zero_row_policy",
    )
    if zero_policy != {
        "raw_all_gene_zero_rows": "error",
        "zero_train_library_rows": "record_and_zero",
    }:
        raise PreflightError("Protocol A zero-row policies are frozen")

    datasets = config["datasets"]
    if not isinstance(datasets, list):
        raise PreflightError("datasets must be a list")
    observed_identity: list[tuple[str, str]] = []
    dataset_required = (
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
    )
    for index, dataset in enumerate(datasets):
        context = f"datasets[{index}]"
        if not isinstance(dataset, dict):
            raise PreflightError(f"{context} must be a mapping")
        _require_keys(dataset, dataset_required, context=context)
        observed_identity.append((dataset["name"], dataset["dataset_id"]))
        if dataset["role"] not in {"primary", "cross_platform"}:
            raise PreflightError(f"{context}.role is invalid")
        for key in ("raw_counts", "scrna_counts", "locations", "gene_names"):
            _package_relative(dataset[key], context=f"{context}.{key}")
        for key in ("frozen_split", "train_mask", "val_mask", "test_mask"):
            _validate_fold_template(dataset[key], context=f"{context}.{key}")
        _positive_shape(dataset["expected_st_shape"], context=f"{context}.expected_st_shape")
        _positive_shape(
            dataset["expected_scrna_shape"],
            context=f"{context}.expected_scrna_shape",
        )
        header = dataset["expected_locations_header"]
        if (
            not isinstance(header, list)
            or len(header) != 2
            or not all(isinstance(value, str) and value for value in header)
        ):
            raise PreflightError(
                f"{context}.expected_locations_header must contain two names"
            )
        if dataset["scrna_orientation"] not in {
            "cells_by_genes",
            "genes_by_cells",
        }:
            raise PreflightError(f"{context}.scrna_orientation is invalid")
        if not isinstance(dataset["expected_scrna_index_header"], str):
            raise PreflightError(
                f"{context}.expected_scrna_index_header must be a string"
            )

    if tuple(observed_identity) != REQUIRED_DATASETS:
        raise PreflightError(
            "datasets must contain the frozen six-dataset order: "
            + ", ".join(name for name, _ in REQUIRED_DATASETS)
        )
    return config


def _resolve_archive_root(project_root: Path, configured_root: str) -> Path:
    root = Path(configured_root)
    if not root.is_absolute():
        root = project_root / root
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"Archive root does not exist: {root}") from error
    if not resolved.is_dir():
        raise PreflightError(f"Archive root is not a directory: {resolved}")
    return resolved


def _resolve_package_file(archive_root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    candidate = archive_root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"Required package file is missing: {relative_path}") from error
    try:
        resolved.relative_to(archive_root)
    except ValueError as error:
        raise PreflightError(f"Package path escapes archive root: {relative_path}") from error
    if not resolved.is_file():
        raise PreflightError(f"Required package path is not a file: {relative_path}")
    return resolved


def _load_checksum_manifest(
    archive_root: Path,
    archive_config: Mapping[str, Any],
) -> tuple[dict[str, tuple[int, str]], ArtifactRecord]:
    relative_path = _package_relative(
        archive_config["checksum_manifest"], context="archive.checksum_manifest"
    )
    path = _resolve_package_file(archive_root, relative_path)
    observed_bytes = path.stat().st_size
    if observed_bytes != archive_config["checksum_manifest_bytes"]:
        raise PreflightError(
            f"Checksum manifest size mismatch: {observed_bytes} != "
            f"{archive_config['checksum_manifest_bytes']}"
        )
    observed_hash = sha256_file(path)
    if observed_hash != archive_config["checksum_manifest_sha256"]:
        raise PreflightError("Checksum manifest SHA256 does not match the config pin")

    records: dict[str, tuple[int, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["relative_path", "size_bytes", "sha256"]:
            raise PreflightError("Checksum manifest header is not the frozen schema")
        for row_number, row in enumerate(reader, start=2):
            relative = _package_relative(
                row["relative_path"], context=f"checksum row {row_number}"
            )
            if relative in records:
                raise PreflightError(f"Duplicate checksum entry: {relative}")
            try:
                size = int(row["size_bytes"])
            except (TypeError, ValueError) as error:
                raise PreflightError(
                    f"Invalid byte size in checksum row {row_number}"
                ) from error
            sha256 = str(row["sha256"]).lower()
            if size < 0 or not SHA256_PATTERN.fullmatch(sha256):
                raise PreflightError(f"Invalid checksum row {row_number}")
            records[relative] = (size, sha256)
    if not records:
        raise PreflightError("Checksum manifest is empty")
    return records, ArtifactRecord(relative_path, path, observed_bytes, observed_hash)


def _expected_checksum(
    checksum_records: Mapping[str, tuple[int, str]], relative_path: str
) -> tuple[int, str]:
    try:
        return checksum_records[relative_path]
    except KeyError as error:
        raise PreflightError(
            f"Configured input is absent from the frozen checksum manifest: {relative_path}"
        ) from error


def _verify_artifact(
    archive_root: Path,
    checksum_records: Mapping[str, tuple[int, str]],
    relative_path: str,
) -> ArtifactRecord:
    relative_path = _package_relative(relative_path, context="input path")
    expected_bytes, expected_hash = _expected_checksum(
        checksum_records, relative_path
    )
    path = _resolve_package_file(archive_root, relative_path)
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise PreflightError(
            f"Input size mismatch for {relative_path}: "
            f"{observed_bytes} != {expected_bytes}"
        )
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise PreflightError(f"Input SHA256 mismatch for {relative_path}")
    return ArtifactRecord(relative_path, path, observed_bytes, observed_hash)


def _decode_tsv_line(raw: bytes, *, path: str, row: str) -> str:
    try:
        return raw.rstrip(b"\r\n").decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise PreflightError(f"Invalid UTF-8 in {path} ({row})") from error


def _inspect_tsv(
    archive_root: Path,
    checksum_records: Mapping[str, tuple[int, str]],
    relative_path: str,
    *,
    capture_first_column: bool,
) -> TableInspection:
    relative_path = _package_relative(relative_path, context="table path")
    expected_bytes, expected_hash = _expected_checksum(
        checksum_records, relative_path
    )
    path = _resolve_package_file(archive_root, relative_path)
    if path.stat().st_size != expected_bytes:
        raise PreflightError(f"Input size mismatch for {relative_path}")

    digest = hashlib.sha256()
    data_rows = 0
    first_column: list[str] | None = [] if capture_first_column else None
    observed_bytes = 0
    with path.open("rb") as handle:
        header_raw = handle.readline()
        if not header_raw:
            raise PreflightError(f"Table is empty: {relative_path}")
        digest.update(header_raw)
        observed_bytes += len(header_raw)
        header_text = _decode_tsv_line(
            header_raw, path=relative_path, row="header"
        )
        header = tuple(header_text.split("\t"))
        for raw_line in handle:
            digest.update(raw_line)
            observed_bytes += len(raw_line)
            if raw_line.rstrip(b"\r\n") == b"":
                raise PreflightError(
                    f"Blank data row {data_rows + 2} in {relative_path}"
                )
            if first_column is not None:
                first_raw = raw_line.partition(b"\t")[0]
                first_column.append(
                    _decode_tsv_line(
                        first_raw,
                        path=relative_path,
                        row=f"row {data_rows + 2} first column",
                    )
                )
            data_rows += 1
    observed_hash = digest.hexdigest()
    if observed_bytes != expected_bytes or observed_hash != expected_hash:
        raise PreflightError(f"Input SHA256 mismatch for {relative_path}")
    artifact = ArtifactRecord(
        relative_path, path, observed_bytes, observed_hash
    )
    return TableInspection(
        artifact,
        header,
        data_rows,
        tuple(first_column) if first_column is not None else None,
    )


def _read_gene_names(record: ArtifactRecord) -> tuple[str, ...]:
    try:
        genes = tuple(record.path.read_text(encoding="utf-8-sig").splitlines())
    except UnicodeDecodeError as error:
        raise PreflightError(
            f"Gene-name file is not UTF-8: {record.relative_path}"
        ) from error
    if not genes or any(not gene for gene in genes):
        raise PreflightError(f"Gene-name file has blank entries: {record.relative_path}")
    if len(set(genes)) != len(genes):
        raise PreflightError(f"Gene names are not unique: {record.relative_path}")
    return genes


def _strict_json_int_list(
    payload: Mapping[str, Any], key: str, *, context: str
) -> np.ndarray:
    if key not in payload:
        raise PreflightError(f"{context} is missing {key}")
    values = payload[key]
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise PreflightError(f"{context}.{key} must be an integer list")
    return np.asarray(values, dtype=np.int64)


def _load_index_mask(record: ArtifactRecord, *, label: str) -> np.ndarray:
    try:
        values = np.load(record.path, allow_pickle=False)
    except Exception as error:
        raise PreflightError(
            f"Could not load {label} mask: {record.relative_path}"
        ) from error
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise PreflightError(f"{label} mask must be a one-dimensional NumPy array")
    if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(
        values.dtype, np.bool_
    ):
        raise PreflightError(f"{label} mask must contain integer indices")
    return values


def _validate_frozen_split_json(
    record: ArtifactRecord,
    *,
    dataset_name: str,
    dataset_id: str,
    fold: int,
    genes: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    try:
        payload = json.loads(record.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(
            f"Invalid frozen split JSON: {record.relative_path}"
        ) from error
    if not isinstance(payload, dict):
        raise PreflightError(f"Frozen split must be an object: {record.relative_path}")
    if payload.get("fold") != fold:
        raise PreflightError(f"Frozen split fold mismatch: {record.relative_path}")
    if "dataset" in payload and payload["dataset"] not in {
        dataset_name,
        dataset_id,
    }:
        raise PreflightError(f"Frozen split dataset mismatch: {record.relative_path}")

    context = f"{dataset_id} fold{fold} frozen split"
    for key, expected in (
        ("train_gene_idx", train_idx),
        ("val_gene_idx", val_idx),
        ("test_gene_idx", test_idx),
    ):
        observed = _strict_json_int_list(payload, key, context=context)
        if not np.array_equal(observed, expected):
            raise PreflightError(f"{context}.{key} disagrees with the .npy mask")
    for key, expected_idx in (
        ("train_genes", train_idx),
        ("val_genes", val_idx),
        ("test_genes", test_idx),
    ):
        if key not in payload or not isinstance(payload[key], list):
            raise PreflightError(f"{context} is missing {key}")
        expected_names = [genes[int(index)] for index in expected_idx]
        observed_names = [str(value) for value in payload[key]]
        if observed_names != expected_names:
            raise PreflightError(f"{context}.{key} disagrees with ST gene order")


def _validate_dataset_sources(
    dataset: Mapping[str, Any],
    *,
    archive_root: Path,
    checksum_records: Mapping[str, tuple[int, str]],
) -> DatasetInputs:
    dataset_id = str(dataset["dataset_id"])
    expected_spots, expected_genes = _positive_shape(
        dataset["expected_st_shape"], context=f"{dataset_id}.expected_st_shape"
    )
    expected_scrna_rows, expected_scrna_columns = _positive_shape(
        dataset["expected_scrna_shape"],
        context=f"{dataset_id}.expected_scrna_shape",
    )

    gene_record = _verify_artifact(
        archive_root, checksum_records, dataset["gene_names"]
    )
    genes = _read_gene_names(gene_record)
    if len(genes) != expected_genes:
        raise PreflightError(
            f"{dataset_id}: gene-name count {len(genes)} != {expected_genes}"
        )
    gene_axis_sha256 = canonical_json_sha256(list(genes))

    raw = _inspect_tsv(
        archive_root,
        checksum_records,
        dataset["raw_counts"],
        capture_first_column=False,
    )
    if raw.data_rows != expected_spots or raw.header != genes:
        raise PreflightError(
            f"{dataset_id}: raw counts shape or gene order does not match "
            f"the frozen [{expected_spots}, {expected_genes}] axis"
        )
    if len(set(raw.header)) != len(raw.header):
        raise PreflightError(f"{dataset_id}: raw counts have duplicate genes")

    capture_scrna_genes = dataset["scrna_orientation"] == "genes_by_cells"
    scrna = _inspect_tsv(
        archive_root,
        checksum_records,
        dataset["scrna_counts"],
        capture_first_column=capture_scrna_genes,
    )
    expected_index_header = dataset["expected_scrna_index_header"]
    if not scrna.header or scrna.header[0] != expected_index_header:
        raise PreflightError(f"{dataset_id}: scRNA index header mismatch")
    if dataset["scrna_orientation"] == "cells_by_genes":
        if (
            scrna.data_rows != expected_scrna_rows
            or len(scrna.header) - 1 != expected_scrna_columns
        ):
            raise PreflightError(f"{dataset_id}: scRNA shape mismatch")
        if tuple(scrna.header[1:]) != genes:
            raise PreflightError(f"{dataset_id}: scRNA gene order differs from ST")
    else:
        if (
            scrna.data_rows != expected_scrna_rows
            or len(scrna.header) - 1 != expected_scrna_columns
            or scrna.first_column is None
        ):
            raise PreflightError(f"{dataset_id}: gene-by-cell scRNA shape mismatch")
        source_genes = scrna.first_column
        if len(set(source_genes)) != len(source_genes):
            raise PreflightError(f"{dataset_id}: scRNA row gene names are not unique")
        source_gene_set = set(source_genes)
        missing = [gene for gene in genes if gene not in source_gene_set]
        if missing:
            raise PreflightError(
                f"{dataset_id}: scRNA is missing ST gene(s): {', '.join(missing[:5])}"
            )

    locations = _inspect_tsv(
        archive_root,
        checksum_records,
        dataset["locations"],
        capture_first_column=False,
    )
    expected_location_header = tuple(dataset["expected_locations_header"])
    if locations.data_rows != expected_spots or locations.header != expected_location_header:
        raise PreflightError(f"{dataset_id}: locations shape/header mismatch")

    static_artifacts = {
        "raw_counts": raw.artifact,
        "scrna_counts": scrna.artifact,
        "locations": locations.artifact,
        "gene_names": gene_record,
    }
    folds: list[FoldInputs] = []
    all_outer_test: list[np.ndarray] = []
    for fold in REQUIRED_FOLDS:
        fold_records: dict[str, ArtifactRecord] = {}
        for key in ("frozen_split", "train_mask", "val_mask", "test_mask"):
            relative = dataset[key].format(fold=fold)
            fold_records[key] = _verify_artifact(
                archive_root, checksum_records, relative
            )
        train_raw = _load_index_mask(fold_records["train_mask"], label="train")
        val_raw = _load_index_mask(fold_records["val_mask"], label="validation")
        test_raw = _load_index_mask(fold_records["test_mask"], label="test")
        try:
            train_idx, val_idx, test_idx = validate_gene_splits(
                expected_genes,
                train_gene_idx=train_raw,
                val_gene_idx=val_raw,
                test_gene_idx=test_raw,
                require_complete_coverage=True,
            )
        except (TypeError, ValueError) as error:
            raise PreflightError(f"{dataset_id} fold{fold}: {error}") from error
        if val_idx.size == 0 or test_idx.size == 0:
            raise PreflightError(
                f"{dataset_id} fold{fold}: validation and final-test masks must be non-empty"
            )
        _validate_frozen_split_json(
            fold_records["frozen_split"],
            dataset_name=str(dataset["name"]),
            dataset_id=dataset_id,
            fold=fold,
            genes=genes,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        folds.append(
            FoldInputs(
                fold=fold,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                artifacts=fold_records,
            )
        )
        all_outer_test.append(test_idx)

    combined_test = np.concatenate(all_outer_test)
    if (
        combined_test.size != expected_genes
        or np.unique(combined_test).size != expected_genes
        or not np.array_equal(
            np.sort(combined_test), np.arange(expected_genes, dtype=np.int64)
        )
    ):
        raise PreflightError(
            f"{dataset_id}: final-test masks do not form an exact five-fold gene partition"
        )
    return DatasetInputs(
        config=dataset,
        genes=genes,
        gene_axis_sha256=gene_axis_sha256,
        artifacts=static_artifacts,
        folds=tuple(folds),
    )


def _load_numeric_inputs(dataset: DatasetInputs) -> tuple[np.ndarray, np.ndarray]:
    dataset_id = str(dataset.config["dataset_id"])
    expected_shape = tuple(dataset.config["expected_st_shape"])
    try:
        frame = pd.read_csv(
            dataset.artifacts["raw_counts"].path,
            sep="\t",
            dtype=np.float64,
            na_filter=False,
            low_memory=False,
            memory_map=True,
        )
    except Exception as error:
        raise PreflightError(f"{dataset_id}: could not parse raw counts") from error
    if tuple(frame.shape) != expected_shape:
        raise PreflightError(
            f"{dataset_id}: parsed raw shape {tuple(frame.shape)} != {expected_shape}"
        )
    if tuple(str(value) for value in frame.columns) != dataset.genes:
        raise PreflightError(f"{dataset_id}: parsed raw gene order changed")
    counts = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(counts).all():
        raise PreflightError(f"{dataset_id}: raw counts contain non-finite values")
    if np.any(counts < 0):
        raise PreflightError(f"{dataset_id}: raw counts contain negative values")

    zero_rows: list[int] = []
    for start in range(0, counts.shape[0], ARRAY_ROW_CHUNK):
        stop = min(start + ARRAY_ROW_CHUNK, counts.shape[0])
        local_zero = np.flatnonzero(~np.any(counts[start:stop] != 0.0, axis=1))
        zero_rows.extend(int(start + row) for row in local_zero)
    if zero_rows:
        preview = ", ".join(str(row) for row in zero_rows[:5])
        raise PreflightError(
            f"{dataset_id}: raw counts contain all-gene zero row(s): {preview}"
        )

    try:
        location_frame = pd.read_csv(
            dataset.artifacts["locations"].path,
            sep="\t",
            dtype=np.float64,
            na_filter=False,
            low_memory=False,
        )
    except Exception as error:
        raise PreflightError(f"{dataset_id}: could not parse locations") from error
    expected_locations_shape = (expected_shape[0], 2)
    if tuple(location_frame.shape) != expected_locations_shape:
        raise PreflightError(
            f"{dataset_id}: parsed locations shape mismatch"
        )
    if tuple(str(value) for value in location_frame.columns) != tuple(
        dataset.config["expected_locations_header"]
    ):
        raise PreflightError(f"{dataset_id}: parsed locations header mismatch")
    locations = location_frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(locations).all():
        raise PreflightError(f"{dataset_id}: locations contain non-finite values")
    return counts, locations


def _expected_zero_train_rows(
    counts: np.ndarray, train_idx: np.ndarray
) -> np.ndarray:
    rows: list[int] = []
    for start in range(0, counts.shape[0], ARRAY_ROW_CHUNK):
        stop = min(start + ARRAY_ROW_CHUNK, counts.shape[0])
        library = counts[start:stop][:, train_idx].sum(axis=1, dtype=np.float64)
        rows.extend(int(start + value) for value in np.flatnonzero(library == 0.0))
    return np.asarray(rows, dtype=np.int64)


def _float32_payload_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": "float32", "shape": [int(value) for value in array.shape]},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(b"\n")
    for start in range(0, array.shape[0], ARRAY_ROW_CHUNK):
        stop = min(start + ARRAY_ROW_CHUNK, array.shape[0])
        chunk = np.ascontiguousarray(array[start:stop], dtype="<f4")
        if not np.isfinite(chunk).all():
            raise PreflightError("float32 truth conversion produced a non-finite value")
        digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def _mode_a_split_payload(
    dataset: DatasetInputs, fold_input: FoldInputs
) -> dict[str, object]:
    genes = dataset.genes
    train = fold_input.train_idx
    validation = fold_input.val_idx
    final_test = fold_input.test_idx
    hidden = np.concatenate((validation, final_test))
    return {
        "schema_version": 1,
        "protocol": "A",
        "protocol_role": "strict_primary_modeA",
        "dataset": dataset.config["name"],
        "dataset_id": dataset.config["dataset_id"],
        "fold": fold_input.fold,
        "gene_count": len(genes),
        "gene_axis_sha256": dataset.gene_axis_sha256,
        "purpose": "Mode-A: inner-train ST visible; validation and final-test ST hidden",
        "train_gene_idx": train.tolist(),
        "train_genes": [genes[int(index)] for index in train],
        "inner_train_gene_idx": train.tolist(),
        "inner_train_genes": [genes[int(index)] for index in train],
        "val_gene_idx": validation.tolist(),
        "val_genes": [genes[int(index)] for index in validation],
        "inner_validation_gene_idx": validation.tolist(),
        "inner_validation_genes": [genes[int(index)] for index in validation],
        "final_test_gene_idx": final_test.tolist(),
        "final_test_genes": [genes[int(index)] for index in final_test],
        "test_gene_idx": hidden.tolist(),
        "test_genes": [genes[int(index)] for index in hidden],
        "hidden_gene_idx": hidden.tolist(),
        "hidden_genes": [genes[int(index)] for index in hidden],
        "test_gene_idx_semantics": "ordered_inner_validation_plus_final_test",
        "visibility": {
            "visible_st_gene_idx": train.tolist(),
            "hidden_st_gene_idx": hidden.tolist(),
            "model_fit_uses_inner_train_only": True,
            "validation_st_hidden_from_model_fit": True,
            "final_test_st_hidden_from_all_fit_and_selection": True,
        },
        "source_frozen_sha256": {
            key: fold_input.artifacts[key].sha256
            for key in ("frozen_split", "train_mask", "val_mask", "test_mask")
        },
    }


def _validate_normalized_fold(
    counts: np.ndarray,
    normalized: np.ndarray,
    protocol_a_audit: Mapping[str, Any],
    fold_input: FoldInputs,
    *,
    dataset_id: str,
) -> np.ndarray:
    if normalized.shape != counts.shape or normalized.dtype != np.float64:
        raise PreflightError(
            f"{dataset_id} fold{fold_input.fold}: unexpected normalized shape/dtype"
        )
    if not np.isfinite(normalized).all() or np.any(normalized < 0.0):
        raise PreflightError(
            f"{dataset_id} fold{fold_input.fold}: invalid normalized values"
        )
    expected_zero = _expected_zero_train_rows(counts, fold_input.train_idx)
    observed_zero = np.asarray(
        protocol_a_audit.get("zero_train_library_rows", []), dtype=np.int64
    )
    if not np.array_equal(observed_zero, expected_zero):
        raise PreflightError(
            f"{dataset_id} fold{fold_input.fold}: zero-library audit mismatch"
        )
    if expected_zero.size and np.any(normalized[expected_zero] != 0.0):
        raise PreflightError(
            f"{dataset_id} fold{fold_input.fold}: zero-library rows are not zero"
        )
    if protocol_a_audit.get("policy") != PROTOCOL_A_POLICY:
        raise PreflightError(
            f"{dataset_id} fold{fold_input.fold}: Protocol A policy mismatch"
        )
    return expected_zero


def _assert_output_path(path: Path, output_root: Path) -> None:
    resolved_root = output_root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    if resolved_path != resolved_root:
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as error:
            raise PreflightError(f"Output escapes the frozen inputs root: {path}") from error


def _write_bytes_atomic(path: Path, payload: bytes, output_root: Path) -> None:
    _assert_output_path(path, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    _assert_output_path(temporary, output_root)
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_truth_atomic(
    path: Path, normalized: np.ndarray, output_root: Path
) -> None:
    _assert_output_path(path, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    _assert_output_path(temporary, output_root)
    matrix = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=normalized.shape,
        fortran_order=False,
    )
    for start in range(0, normalized.shape[0], ARRAY_ROW_CHUNK):
        stop = min(start + ARRAY_ROW_CHUNK, normalized.shape[0])
        matrix[start:stop] = normalized[start:stop]
    matrix.flush()
    del matrix
    os.replace(temporary, path)
    observed = np.load(path, allow_pickle=False, mmap_mode="r")
    if observed.dtype != np.float32 or observed.shape != normalized.shape:
        raise PreflightError(f"Materialized truth verification failed: {path}")
    for start in range(0, normalized.shape[0], ARRAY_ROW_CHUNK):
        stop = min(start + ARRAY_ROW_CHUNK, normalized.shape[0])
        expected = np.asarray(normalized[start:stop], dtype=np.float32)
        if not np.array_equal(observed[start:stop], expected):
            raise PreflightError(
                f"Materialized truth values differ after float32 conversion: {path}"
            )
    del observed


def _output_record(
    path: Path,
    output_root: Path,
    *,
    display_root: Path | None = None,
) -> dict[str, object]:
    _assert_output_path(path, output_root)
    if not path.is_file():
        raise PreflightError(f"Expected output is missing: {path}")
    relative_root = output_root if display_root is None else display_root
    try:
        display_path = path.relative_to(relative_root).as_posix()
    except ValueError as error:
        raise PreflightError(
            f"Output is outside its display root: {path}"
        ) from error
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _local_record(path: Path, project_root: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    try:
        display_path = resolved.relative_to(project_root.resolve(strict=True)).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _fold_input_records(
    dataset: DatasetInputs,
    fold_input: FoldInputs,
    *,
    config_record: Mapping[str, object],
    preprocessing_record: Mapping[str, object],
) -> dict[str, object]:
    records = {
        key: value.as_dict() for key, value in dataset.artifacts.items()
    }
    records.update(
        {key: value.as_dict() for key, value in fold_input.artifacts.items()}
    )
    records["config"] = dict(config_record)
    records["protocol_a_preprocessing"] = dict(preprocessing_record)
    return records


def _process_fold(
    dataset: DatasetInputs,
    fold_input: FoldInputs,
    counts: np.ndarray,
    *,
    stage_root: Path | None,
    output_root: Path,
    config_record: Mapping[str, object],
    preprocessing_record: Mapping[str, object],
) -> dict[str, object]:
    dataset_id = str(dataset.config["dataset_id"])
    try:
        normalized, protocol_a_audit = normalize_st_protocol_a(
            counts,
            inner_train_gene_idx=fold_input.train_idx,
            val_gene_idx=fold_input.val_idx,
            test_gene_idx=fold_input.test_idx,
            require_complete_coverage=True,
            target_sum=PROTOCOL_A_TARGET_SUM,
        )
    except (TypeError, ValueError) as error:
        raise PreflightError(
            f"{dataset_id} fold{fold_input.fold}: normalization failed: {error}"
        ) from error
    zero_train_rows = _validate_normalized_fold(
        counts,
        normalized,
        protocol_a_audit,
        fold_input,
        dataset_id=dataset_id,
    )
    payload_sha256 = _float32_payload_sha256(normalized)
    split_payload = _mode_a_split_payload(dataset, fold_input)
    split_bytes = canonical_json_bytes(split_payload)
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    input_records = _fold_input_records(
        dataset,
        fold_input,
        config_record=config_record,
        preprocessing_record=preprocessing_record,
    )
    summary: dict[str, object] = {
        "fold": fold_input.fold,
        "shape": [int(value) for value in normalized.shape],
        "truth_dtype": "float32",
        "truth_payload_sha256": payload_sha256,
        "mode_a_split_sha256": split_sha256,
        "inner_train_gene_count": int(fold_input.train_idx.size),
        "inner_validation_gene_count": int(fold_input.val_idx.size),
        "final_test_gene_count": int(fold_input.test_idx.size),
        "hidden_gene_count": int(fold_input.val_idx.size + fold_input.test_idx.size),
        "zero_train_library_spot_count": int(zero_train_rows.size),
        "zero_train_library_rows": zero_train_rows.tolist(),
    }

    if stage_root is not None:
        fold_dir = stage_root / dataset_id / f"fold{fold_input.fold}"
        truth_path = fold_dir / "full_truth.npy"
        split_path = fold_dir / "mode_a_split.json"
        audit_path = fold_dir / "normalization_audit.json"
        manifest_path = fold_dir / "artifact_manifest.json"
        _write_truth_atomic(truth_path, normalized, output_root)
        _write_bytes_atomic(split_path, split_bytes, output_root)
        truth_record = _output_record(
            truth_path, output_root, display_root=stage_root
        )
        split_record = _output_record(
            split_path, output_root, display_root=stage_root
        )
        normalization_audit = {
            "schema_version": 1,
            "protocol": "A",
            "dataset": dataset.config["name"],
            "dataset_id": dataset_id,
            "fold": fold_input.fold,
            "full_truth": {
                "shape": [int(value) for value in normalized.shape],
                "dtype": "float32",
                "gene_axis_sha256": dataset.gene_axis_sha256,
                "payload_sha256": payload_sha256,
                "file": truth_record,
            },
            "mode_a_split": split_record,
            "split_validation": {
                "complete_coverage": True,
                "mutually_disjoint": True,
                "visible_scope": "inner_train_only",
                "hidden_scope": "inner_validation_plus_final_test",
                "inner_train_gene_count": int(fold_input.train_idx.size),
                "inner_validation_gene_count": int(fold_input.val_idx.size),
                "final_test_gene_count": int(fold_input.test_idx.size),
            },
            "zero_rows": {
                "raw_all_gene_zero_rows": [],
                "raw_all_gene_zero_row_policy": "error",
                "zero_train_library_rows": zero_train_rows.tolist(),
                "zero_train_library_spot_count": int(zero_train_rows.size),
                "zero_train_library_policy": ZERO_TRAIN_LIBRARY_POLICY,
                "materialized_zero_rows_verified": True,
            },
            "protocol_a": dict(protocol_a_audit),
            "input_artifacts": input_records,
            "output_sha256": {
                "full_truth_npy": truth_record["sha256"],
                "mode_a_split_json": split_record["sha256"],
            },
        }
        _write_bytes_atomic(
            audit_path, canonical_json_bytes(normalization_audit), output_root
        )
        audit_record = _output_record(
            audit_path, output_root, display_root=stage_root
        )
        artifact_manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "fold": fold_input.fold,
            "input_artifacts": input_records,
            "output_artifacts": {
                "full_truth": truth_record,
                "mode_a_split": split_record,
                "normalization_audit": audit_record,
            },
        }
        _write_bytes_atomic(
            manifest_path, canonical_json_bytes(artifact_manifest), output_root
        )
        summary["output_artifacts"] = {
            **artifact_manifest["output_artifacts"],
            "artifact_manifest": _output_record(
                manifest_path, output_root, display_root=stage_root
            ),
        }
    del normalized
    return summary


def _expected_output_root(project_root: Path) -> Path:
    return project_root.joinpath(*OUTPUT_RELATIVE.parts).resolve(strict=False)


def prepare_protocol_a_inputs(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    materialize_truth: bool = False,
    project_root: str | Path = GENESPT_ROOT,
    quiet: bool = False,
) -> dict[str, object]:
    """Preflight all six datasets and optionally materialize fold artifacts.

    ``project_root`` exists for isolated tests.  The CLI does not expose it, so
    production outputs remain frozen to the repository's one allowed root.
    """

    project = Path(project_root).resolve(strict=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (project / config_file).resolve(strict=False)
    config_file = config_file.resolve(strict=True)
    config = _validated_config(config_file)
    archive_root = _resolve_archive_root(project, config["archive"]["root"])
    checksum_records, checksum_manifest_record = _load_checksum_manifest(
        archive_root, config["archive"]
    )
    output_root = _expected_output_root(project)
    configured_output = project.joinpath(*PurePosixPath(config["output_root"]).parts)
    if configured_output.resolve(strict=False) != output_root:
        raise PreflightError("Configured output does not resolve to the frozen inputs root")

    config_record = _local_record(config_file, project)
    preprocessing_path = GENESPT_ROOT / "main" / "protocol_a_preprocessing.py"
    preprocessing_record = _local_record(preprocessing_path, GENESPT_ROOT)
    script_record = _local_record(Path(__file__), GENESPT_ROOT)

    validated_datasets: list[DatasetInputs] = []
    for dataset_config in config["datasets"]:
        if not quiet:
            print(
                f"[static] {dataset_config['dataset_id']}: paths, SHA256, axes, masks",
                flush=True,
            )
        validated_datasets.append(
            _validate_dataset_sources(
                dataset_config,
                archive_root=archive_root,
                checksum_records=checksum_records,
            )
        )

    if materialize_truth and output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Protocol A inputs root: {output_root}"
        )

    created_output_root = False
    stage_root: Path | None = None
    if materialize_truth:
        output_root.mkdir(parents=True, exist_ok=False)
        created_output_root = True
        stage_root = output_root / ".staging"
        stage_root.mkdir()

    dataset_summaries: list[dict[str, object]] = []
    try:
        for dataset in validated_datasets:
            dataset_id = str(dataset.config["dataset_id"])
            if not quiet:
                print(
                    f"[numeric] {dataset_id}: raw values, locations, five folds",
                    flush=True,
                )
            counts, _locations = _load_numeric_inputs(dataset)
            fold_summaries: list[dict[str, object]] = []
            for fold_input in dataset.folds:
                if not quiet:
                    print(
                        f"[normalize] {dataset_id} fold{fold_input.fold}",
                        flush=True,
                    )
                fold_summaries.append(
                    _process_fold(
                        dataset,
                        fold_input,
                        counts,
                        stage_root=stage_root,
                        output_root=output_root,
                        config_record=config_record,
                        preprocessing_record=preprocessing_record,
                    )
                )
            dataset_summaries.append(
                {
                    "name": dataset.config["name"],
                    "dataset_id": dataset_id,
                    "shape": list(dataset.config["expected_st_shape"]),
                    "scrna_shape": list(dataset.config["expected_scrna_shape"]),
                    "scrna_orientation": dataset.config["scrna_orientation"],
                    "gene_axis_sha256": dataset.gene_axis_sha256,
                    "cross_fold_final_test_partition_complete": True,
                    "input_artifacts": {
                        key: value.as_dict()
                        for key, value in dataset.artifacts.items()
                    },
                    "folds": fold_summaries,
                }
            )
            del counts, _locations

        result: dict[str, object] = {
            "schema_version": 1,
            "protocol": "A",
            "mode": "materialized" if materialize_truth else "preflight_only",
            "model_started": False,
            "dataset_count": len(dataset_summaries),
            "folds_per_dataset": len(REQUIRED_FOLDS),
            "output_root": OUTPUT_RELATIVE.as_posix(),
            "checksum_trust_root": checksum_manifest_record.as_dict(),
            "code_artifacts": {
                "config": config_record,
                "prepare_script": script_record,
                "protocol_a_preprocessing": preprocessing_record,
            },
            "datasets": dataset_summaries,
        }
        if stage_root is not None:
            manifest_path = stage_root / "materialization_manifest.json"
            _write_bytes_atomic(
                manifest_path, canonical_json_bytes(result), output_root
            )
            for child in sorted(stage_root.iterdir(), key=lambda value: value.name):
                destination = output_root / child.name
                _assert_output_path(destination, output_root)
                os.replace(child, destination)
            stage_root.rmdir()
        return result
    except BaseException:
        if created_output_root and output_root.exists():
            expected = _expected_output_root(project)
            if output_root.resolve(strict=False) != expected:
                raise RuntimeError("Refusing cleanup outside frozen output root")
            shutil.rmtree(output_root)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight the frozen six-dataset Protocol A inputs. No model is started."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Read and validate everything without creating output files (default).",
    )
    mode.add_argument(
        "--materialize-truth",
        action="store_true",
        help="Write fold-specific float32 truth and audits under the frozen inputs root.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = prepare_protocol_a_inputs(
            args.config,
            materialize_truth=bool(args.materialize_truth),
            project_root=GENESPT_ROOT,
        )
    except Exception as error:
        print(f"[failed] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    zero_train_total = sum(
        int(fold["zero_train_library_spot_count"])
        for dataset in result["datasets"]
        for fold in dataset["folds"]
    )
    zero_train_details = [
        {
            "dataset_id": dataset["dataset_id"],
            "fold": fold["fold"],
            "rows": fold["zero_train_library_rows"],
        }
        for dataset in result["datasets"]
        for fold in dataset["folds"]
        if fold["zero_train_library_spot_count"]
    ]
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": result["mode"],
                "datasets": result["dataset_count"],
                "folds": result["dataset_count"] * result["folds_per_dataset"],
                "zero_train_library_spot_rows_across_folds": zero_train_total,
                "zero_train_library_details": zero_train_details,
                "model_started": False,
                "output_root": result["output_root"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
