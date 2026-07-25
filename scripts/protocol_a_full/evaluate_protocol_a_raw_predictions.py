#!/usr/bin/env python3
"""Read-only integrity audit and centralized evaluation for Protocol A outputs.

The evaluator never chooses a readout, calibrates a matrix, or fills a missing
prediction.  A prediction reaches ``evaluate_prediction`` only after its input
package, completion metadata, identity, shape, finite values, hashes, and
frozen final-test indices have all been validated.

Missing scheduler products are reported as missing by default so this command
can monitor an in-progress full rerun.  ``--require-complete`` changes only the
exit policy; it does not change what is read or evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np


GENESPT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = GENESPT_ROOT / "configs" / "protocol_a_datasets.yaml"
DEFAULT_RESULTS_ROOT = (
    GENESPT_ROOT / "results" / "protocol_a_full_rerun_20260711"
)
DEFAULT_INPUTS_ROOT = DEFAULT_RESULTS_ROOT / "inputs"
DEFAULT_GENESPT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "genespt"
DEFAULT_BASELINE_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "baselines"
DEFAULT_METRICS_PATH = (
    GENESPT_ROOT.parent
    / "GeneSPT_github_main_rebuild"
    / "src"
    / "genespt"
    / "metrics.py"
)

PROTOCOL = "A"
PROTOCOL_POLICY = "inner_train_gene_library_size_applied_to_all_columns"
REQUIRED_FOLDS = (0, 1, 2, 3, 4)
METRICS = ("SPCC", "RMSE", "JSD", "SSIM")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHUNK_BYTES = 8 * 1024 * 1024
ARRAY_ROW_CHUNK = 256

GENESPT_PREFIX = "protocol_a_genespt"
GENESPT_MODE = "benchmark"
GENESPT_GC_MODEL = "gc_mlp_base"
GENESPT_FULL_MODEL = "predictable_spatial_program_selected_correct"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    model: str | None = None


METHOD_SPECS = (
    MethodSpec("GeneSPT-GC", "genespt", GENESPT_GC_MODEL),
    MethodSpec("GeneSPT", "genespt", GENESPT_FULL_MODEL),
    MethodSpec("Tangram", "baseline"),
    MethodSpec("TransImp", "baseline"),
    MethodSpec("SpaIM", "baseline"),
    MethodSpec("SpaGE", "baseline"),
    MethodSpec("stPlus", "baseline"),
)


class AuditError(RuntimeError):
    """Base class for a reportable Protocol A audit outcome."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class MissingArtifact(AuditError):
    """A scheduler product is not complete yet."""


class IntegrityError(AuditError):
    """An existing artifact violates the frozen contract."""


@dataclass(frozen=True)
class FileFingerprint:
    size_bytes: int
    mtime_ns: int
    sha256: str


class FileHasher:
    """Hash files once and detect mutation during a single audit process."""

    def __init__(self) -> None:
        self._cache: dict[Path, FileFingerprint] = {}

    def hash(self, path: Path, *, context: str) -> str:
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise MissingArtifact("file_missing", f"{context} is missing: {path}") from error
        if not resolved.is_file() or resolved.is_symlink():
            raise IntegrityError(
                "not_regular_file", f"{context} is not a regular file: {resolved}"
            )
        before = resolved.stat()
        cached = self._cache.get(resolved)
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
                raise IntegrityError(
                    "file_changed_while_hashing",
                    f"{context} changed while being hashed: {resolved}",
                )
            cached = FileFingerprint(
                size_bytes=int(after.st_size),
                mtime_ns=int(after.st_mtime_ns),
                sha256=digest.hexdigest(),
            )
            self._cache[resolved] = cached
        elif (
            before.st_size != cached.size_bytes
            or before.st_mtime_ns != cached.mtime_ns
        ):
            raise IntegrityError(
                "file_changed_during_audit",
                f"{context} changed after its first audit read: {resolved}",
            )
        return cached.sha256

    def fingerprint(self, path: Path, *, context: str) -> FileFingerprint:
        self.hash(path, context=context)
        return self._cache[path.resolve(strict=True)]

    def assert_unchanged(self) -> None:
        for path, record in self._cache.items():
            try:
                current = path.stat()
            except OSError as error:
                raise IntegrityError(
                    "file_removed_during_audit",
                    f"Audited file disappeared during evaluation: {path}",
                ) from error
            if (
                current.st_size != record.size_bytes
                or current.st_mtime_ns != record.mtime_ns
            ):
                raise IntegrityError(
                    "file_changed_during_audit",
                    f"Audited file changed during evaluation: {path}",
                )


@dataclass(frozen=True)
class GlobalInputs:
    project_root: Path
    config: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    archive_root: Path
    archive_records: Mapping[str, tuple[int, str]]
    archive_manifest_path: Path
    archive_manifest_sha256: str
    inputs_root: Path
    materialization_path: Path
    materialization: Mapping[str, Any]
    materialization_sha256: str


@dataclass(frozen=True)
class FoldInput:
    dataset_name: str
    dataset_id: str
    role: str
    fold: int
    expected_shape: tuple[int, int]
    gene_names: tuple[str, ...]
    gene_axis_sha256: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    truth_path: Path
    truth_sha256: str
    split_path: Path
    split_sha256: str
    normalization_path: Path
    normalization_sha256: str
    artifact_manifest_path: Path
    artifact_manifest_sha256: str
    materialization_sha256: str
    source_sha256: Mapping[str, str]

    @property
    def test_gene_names(self) -> tuple[str, ...]:
        return tuple(self.gene_names[int(index)] for index in self.test_idx)


@dataclass
class InputOutcome:
    dataset_name: str
    dataset_id: str
    role: str
    fold: int
    status: str
    context: FoldInput | None = None
    issue_code: str = ""
    issue: str = ""


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_mapping(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise MissingArtifact("manifest_missing", f"{context} is missing: {path}") from error
    except OSError as error:
        raise IntegrityError("manifest_unreadable", f"Could not read {context}: {path}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as error:
            raise IntegrityError(
                "mapping_parse_error",
                f"{context} is not JSON-form YAML and PyYAML is unavailable: {path}",
            ) from error
        try:
            payload = yaml.safe_load(text)
        except Exception as error:
            raise IntegrityError(
                "mapping_parse_error", f"Could not parse {context}: {path}"
            ) from error
    if not isinstance(payload, dict):
        raise IntegrityError("mapping_type_error", f"{context} must be an object: {path}")
    return payload


def _safe_relative(value: object, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise IntegrityError("invalid_relative_path", f"{context} is not a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise IntegrityError("invalid_relative_path", f"{context} is unsafe: {value!r}")
    return path


def _format_fold_path(value: object, fold: int, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or value.count("{fold}") != 1:
        raise IntegrityError(
            "invalid_fold_template", f"{context} must contain exactly one {{fold}}"
        )
    try:
        formatted = value.format(fold=fold)
    except (KeyError, ValueError) as error:
        raise IntegrityError("invalid_fold_template", f"Could not format {context}") from error
    return _safe_relative(formatted, context=context)


def _normal_path(value: object) -> str:
    return str(value).replace("\\", "/").rstrip("/")


def _record_path_matches(recorded: object, expected: str, *, suffix_ok: bool) -> bool:
    observed = _normal_path(recorded).lstrip("./")
    wanted = _normal_path(expected).lstrip("./")
    return observed == wanted or (suffix_ok and observed.endswith("/" + wanted))


def _declared_record(
    record: object,
    *,
    expected_path: str,
    context: str,
    suffix_ok: bool = False,
) -> tuple[int, str]:
    if not isinstance(record, dict):
        raise IntegrityError("manifest_record_missing", f"{context} record is missing")
    if not _record_path_matches(record.get("path"), expected_path, suffix_ok=suffix_ok):
        raise IntegrityError(
            "manifest_path_mismatch",
            f"{context} path {record.get('path')!r} does not match {expected_path!r}",
        )
    size = record.get("bytes")
    digest = record.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise IntegrityError("manifest_size_invalid", f"{context} has an invalid byte count")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise IntegrityError("manifest_sha_invalid", f"{context} has an invalid SHA256")
    return int(size), digest


def _verify_record(
    record: object,
    path: Path,
    *,
    expected_path: str,
    context: str,
    hasher: FileHasher,
    suffix_ok: bool = False,
) -> str:
    size, expected_sha = _declared_record(
        record,
        expected_path=expected_path,
        context=context,
        suffix_ok=suffix_ok,
    )
    actual_sha = hasher.hash(path, context=context)
    fingerprint = hasher.fingerprint(path, context=context)
    if fingerprint.size_bytes != size:
        raise IntegrityError(
            "manifest_size_mismatch",
            f"{context} byte count {fingerprint.size_bytes} does not match {size}",
        )
    if actual_sha != expected_sha:
        raise IntegrityError("manifest_sha_mismatch", f"{context} SHA256 mismatch")
    return actual_sha


def _validate_config(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise IntegrityError(
            "config_identity_mismatch", "Config must declare schema_version=1 and protocol='A'"
        )
    if config.get("folds") != list(REQUIRED_FOLDS):
        raise IntegrityError(
            "config_folds_mismatch", f"Config folds must be exactly {list(REQUIRED_FOLDS)}"
        )
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise IntegrityError("config_datasets_invalid", "config.datasets must be non-empty")
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
    }
    identities: set[str] = set()
    names: set[str] = set()
    checked: list[Mapping[str, Any]] = []
    for position, item in enumerate(datasets):
        if not isinstance(item, dict) or not required.issubset(item):
            raise IntegrityError(
                "config_dataset_invalid", f"Config dataset {position} is incomplete"
            )
        name = item["name"]
        dataset_id = item["dataset_id"]
        if not isinstance(name, str) or not name or not isinstance(dataset_id, str) or not dataset_id:
            raise IntegrityError(
                "config_dataset_identity_invalid", f"Config dataset {position} has no identity"
            )
        if name.casefold() in names or dataset_id.casefold() in identities:
            raise IntegrityError("config_dataset_duplicate", "Config dataset identities are not unique")
        names.add(name.casefold())
        identities.add(dataset_id.casefold())
        shape = item["expected_st_shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise IntegrityError(
                "config_shape_invalid", f"{dataset_id}.expected_st_shape is invalid"
            )
        for key in ("gene_names",):
            _safe_relative(item[key], context=f"{dataset_id}.{key}")
        for key in ("frozen_split", "train_mask", "val_mask", "test_mask"):
            _format_fold_path(item[key], 0, context=f"{dataset_id}.{key}")
        checked.append(item)
    return tuple(checked)


def _select_datasets(
    datasets: Sequence[Mapping[str, Any]], selectors: Sequence[str] | None
) -> tuple[Mapping[str, Any], ...]:
    if not selectors:
        return tuple(datasets)
    lookup: dict[str, Mapping[str, Any]] = {}
    for dataset in datasets:
        lookup[str(dataset["name"]).casefold()] = dataset
        lookup[str(dataset["dataset_id"]).casefold()] = dataset
    selected: list[Mapping[str, Any]] = []
    for selector in selectors:
        dataset = lookup.get(str(selector).casefold())
        if dataset is None:
            raise IntegrityError("unknown_dataset", f"Unknown dataset selector: {selector}")
        if dataset not in selected:
            selected.append(dataset)
    selected.sort(key=lambda item: datasets.index(item))
    return tuple(selected)


def _select_methods(selectors: Sequence[str] | None) -> tuple[MethodSpec, ...]:
    if not selectors:
        return METHOD_SPECS
    lookup = {spec.name.casefold(): spec for spec in METHOD_SPECS}
    selected: list[MethodSpec] = []
    for selector in selectors:
        spec = lookup.get(str(selector).casefold())
        if spec is None:
            raise IntegrityError("unknown_method", f"Unknown method selector: {selector}")
        if spec not in selected:
            selected.append(spec)
    selected.sort(key=METHOD_SPECS.index)
    return tuple(selected)


def _resolve_archive_root(config: Mapping[str, Any], project_root: Path) -> Path:
    archive = config.get("archive")
    if not isinstance(archive, dict) or not isinstance(archive.get("root"), str):
        raise IntegrityError("archive_config_invalid", "config.archive.root is invalid")
    root = Path(archive["root"])
    if not root.is_absolute():
        root = project_root / root
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise IntegrityError("archive_root_missing", f"Archive root is missing: {root}") from error
    if not resolved.is_dir():
        raise IntegrityError("archive_root_invalid", f"Archive root is not a directory: {resolved}")
    return resolved


def _load_archive_manifest(
    config: Mapping[str, Any], project_root: Path, hasher: FileHasher
) -> tuple[Path, Path, Mapping[str, tuple[int, str]], str]:
    archive = config.get("archive")
    assert isinstance(archive, dict)
    root = _resolve_archive_root(config, project_root)
    relative = _safe_relative(
        archive.get("checksum_manifest"), context="archive.checksum_manifest"
    )
    path = root.joinpath(*relative.parts)
    expected_size = archive.get("checksum_manifest_bytes")
    expected_sha = archive.get("checksum_manifest_sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or not isinstance(expected_sha, str)
        or not SHA256_RE.fullmatch(expected_sha)
    ):
        raise IntegrityError(
            "archive_manifest_pin_invalid", "Config archive checksum pin is invalid"
        )
    actual_sha = hasher.hash(path, context="archive checksum manifest")
    fingerprint = hasher.fingerprint(path, context="archive checksum manifest")
    if fingerprint.size_bytes != expected_size or actual_sha != expected_sha:
        raise IntegrityError(
            "archive_manifest_pin_mismatch", "Archive checksum manifest differs from config pin"
        )
    records: dict[str, tuple[int, str]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["relative_path", "size_bytes", "sha256"]:
                raise IntegrityError(
                    "archive_manifest_header_invalid", "Archive checksum manifest header is invalid"
                )
            for row_number, row in enumerate(reader, start=2):
                key = _safe_relative(
                    row.get("relative_path"), context=f"archive checksum row {row_number}"
                ).as_posix()
                if key in records:
                    raise IntegrityError(
                        "archive_manifest_duplicate", f"Duplicate archive record: {key}"
                    )
                try:
                    size = int(str(row.get("size_bytes")))
                except ValueError as error:
                    raise IntegrityError(
                        "archive_manifest_size_invalid",
                        f"Invalid archive byte count at row {row_number}",
                    ) from error
                digest = str(row.get("sha256", ""))
                if size < 0 or not SHA256_RE.fullmatch(digest):
                    raise IntegrityError(
                        "archive_manifest_record_invalid",
                        f"Invalid archive checksum record at row {row_number}",
                    )
                records[key] = (size, digest)
    except OSError as error:
        raise IntegrityError(
            "archive_manifest_unreadable", f"Could not read archive manifest: {path}"
        ) from error
    return root, path, records, actual_sha


def _project_relative(path: Path, project_root: Path, *, context: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(project_root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as error:
        raise IntegrityError(
            "path_outside_project", f"{context} is outside the project root: {path}"
        ) from error


def _prepare_global_inputs(
    *,
    config_path: Path,
    project_root: Path,
    inputs_root: Path,
    hasher: FileHasher,
) -> GlobalInputs:
    config = _load_mapping(config_path, context="Protocol A config")
    datasets = _validate_config(config)
    del datasets
    config_sha = hasher.hash(config_path, context="Protocol A config")
    archive_root, archive_manifest_path, archive_records, archive_sha = _load_archive_manifest(
        config, project_root, hasher
    )

    configured_output = _safe_relative(config.get("output_root"), context="config.output_root")
    expected_inputs_root = project_root.joinpath(*configured_output.parts).resolve(strict=False)
    if inputs_root.resolve(strict=False) != expected_inputs_root:
        raise IntegrityError(
            "inputs_root_mismatch",
            f"Inputs root {inputs_root} does not match config output_root {expected_inputs_root}",
        )

    materialization_path = inputs_root / "materialization_manifest.json"
    materialization = _load_mapping(
        materialization_path, context="Protocol A materialization manifest"
    )
    materialization_sha = hasher.hash(
        materialization_path, context="Protocol A materialization manifest"
    )
    if (
        materialization.get("schema_version") != 1
        or materialization.get("protocol") != PROTOCOL
        or materialization.get("mode") != "materialized"
        or materialization.get("model_started") is not False
        or materialization.get("folds_per_dataset") != len(REQUIRED_FOLDS)
        or materialization.get("output_root") != configured_output.as_posix()
    ):
        raise IntegrityError(
            "materialization_identity_mismatch",
            "Materialization manifest header does not match Protocol A",
        )
    config_datasets = config.get("datasets")
    if materialization.get("dataset_count") != len(config_datasets):
        raise IntegrityError(
            "materialization_dataset_count_mismatch",
            "Materialization dataset count does not match config",
        )
    code = materialization.get("code_artifacts")
    if not isinstance(code, dict):
        raise IntegrityError(
            "materialization_code_missing", "Materialization code_artifacts is missing"
        )
    config_relative = _project_relative(config_path, project_root, context="Protocol A config")
    _verify_record(
        code.get("config"),
        config_path,
        expected_path=config_relative,
        context="materialization config",
        hasher=hasher,
    )
    checksum_record = materialization.get("checksum_trust_root")
    _verify_record(
        checksum_record,
        archive_manifest_path,
        expected_path=PurePosixPath(str(config["archive"]["checksum_manifest"])).as_posix(),
        context="materialization checksum trust root",
        hasher=hasher,
    )
    return GlobalInputs(
        project_root=project_root,
        config=config,
        config_path=config_path,
        config_sha256=config_sha,
        archive_root=archive_root,
        archive_records=archive_records,
        archive_manifest_path=archive_manifest_path,
        archive_manifest_sha256=archive_sha,
        inputs_root=inputs_root,
        materialization_path=materialization_path,
        materialization=materialization,
        materialization_sha256=materialization_sha,
    )


def _materialization_dataset(
    materialization: Mapping[str, Any], dataset_id: str
) -> Mapping[str, Any]:
    datasets = materialization.get("datasets")
    if not isinstance(datasets, list):
        raise IntegrityError(
            "materialization_datasets_invalid", "Materialization datasets field is invalid"
        )
    matches = [
        item
        for item in datasets
        if isinstance(item, dict) and item.get("dataset_id") == dataset_id
    ]
    if len(matches) != 1:
        raise IntegrityError(
            "materialization_dataset_missing",
            f"Materialization must contain exactly one dataset {dataset_id}",
        )
    return matches[0]


def _materialization_fold(
    dataset_summary: Mapping[str, Any], dataset_id: str, fold: int
) -> Mapping[str, Any]:
    folds = dataset_summary.get("folds")
    if not isinstance(folds, list):
        raise IntegrityError(
            "materialization_folds_invalid", f"Materialization folds are invalid for {dataset_id}"
        )
    matches = [item for item in folds if isinstance(item, dict) and item.get("fold") == fold]
    if len(matches) != 1:
        raise IntegrityError(
            "materialization_fold_missing",
            f"Materialization must contain exactly one {dataset_id} fold{fold}",
        )
    return matches[0]


def _archive_record_matches(
    record: object,
    relative: PurePosixPath,
    global_inputs: GlobalInputs,
    *,
    context: str,
) -> tuple[int, str]:
    key = relative.as_posix()
    if key not in global_inputs.archive_records:
        raise IntegrityError(
            "archive_record_missing", f"{context} is absent from the pinned archive manifest"
        )
    declared_size, declared_sha = _declared_record(
        record, expected_path=key, context=context
    )
    expected_size, expected_sha = global_inputs.archive_records[key]
    if declared_size != expected_size or declared_sha != expected_sha:
        raise IntegrityError(
            "archive_record_mismatch", f"{context} disagrees with the pinned archive manifest"
        )
    return expected_size, expected_sha


def _load_gene_names(path: Path, expected_count: int) -> tuple[str, ...]:
    try:
        genes = tuple(path.read_text(encoding="utf-8-sig").splitlines())
    except (OSError, UnicodeDecodeError) as error:
        raise IntegrityError("gene_names_unreadable", f"Could not read gene names: {path}") from error
    if len(genes) != expected_count or any(not gene for gene in genes):
        raise IntegrityError(
            "gene_axis_length_mismatch",
            f"Gene-name count {len(genes)} does not match {expected_count}",
        )
    if len(set(genes)) != len(genes):
        raise IntegrityError("gene_axis_duplicate", "Gene names are not unique")
    return genes


def _load_mask(path: Path, n_genes: int, *, context: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as error:
        raise IntegrityError("mask_load_error", f"Could not load {context}: {path}") from error
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
        raise IntegrityError("mask_type_error", f"{context} must be a 1-D integer array")
    result = value.astype(np.int64, copy=False)
    if result.size == 0:
        raise IntegrityError("mask_empty", f"{context} is empty")
    if np.unique(result).size != result.size:
        raise IntegrityError("mask_duplicate", f"{context} contains duplicate indices")
    if int(result.min()) < 0 or int(result.max()) >= n_genes:
        raise IntegrityError("mask_out_of_range", f"{context} contains an out-of-range index")
    return result


def _strict_int_list(
    payload: Mapping[str, Any], key: str, n_genes: int, *, context: str
) -> np.ndarray:
    values = payload.get(key)
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise IntegrityError("split_index_invalid", f"{context}.{key} must be an integer list")
    result = np.asarray(values, dtype=np.int64)
    if result.size and (int(result.min()) < 0 or int(result.max()) >= n_genes):
        raise IntegrityError("split_index_out_of_range", f"{context}.{key} is out of range")
    if np.unique(result).size != result.size:
        raise IntegrityError("split_index_duplicate", f"{context}.{key} has duplicates")
    return result


def _assert_names(
    payload: Mapping[str, Any], key: str, expected: Sequence[str], *, context: str
) -> None:
    observed = payload.get(key)
    if not isinstance(observed, list) or [str(value) for value in observed] != list(expected):
        raise IntegrityError("split_gene_order_mismatch", f"{context}.{key} differs from gene axis")


def _float32_payload_sha256(matrix: np.ndarray) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": "float32", "shape": [int(value) for value in matrix.shape]},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(b"\n")
    nonfinite = 0
    negative = 0
    for start in range(0, matrix.shape[0], ARRAY_ROW_CHUNK):
        stop = min(start + ARRAY_ROW_CHUNK, matrix.shape[0])
        chunk = np.ascontiguousarray(matrix[start:stop], dtype="<f4")
        nonfinite += int((~np.isfinite(chunk)).sum())
        negative += int((chunk < 0.0).sum())
        digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest(), nonfinite, negative


def _verify_source_file(
    *,
    key: str,
    relative: PurePosixPath,
    record: object,
    global_inputs: GlobalInputs,
    hasher: FileHasher,
    context: str,
) -> tuple[Path, str]:
    expected_size, expected_sha = _archive_record_matches(
        record, relative, global_inputs, context=context
    )
    path = global_inputs.archive_root.joinpath(*relative.parts)
    actual_sha = hasher.hash(path, context=context)
    fingerprint = hasher.fingerprint(path, context=context)
    if fingerprint.size_bytes != expected_size or actual_sha != expected_sha:
        raise IntegrityError("source_sha_mismatch", f"{context} differs from archive checksum pin")
    return path, actual_sha


def _validate_fold_input(
    dataset: Mapping[str, Any],
    fold: int,
    global_inputs: GlobalInputs,
    hasher: FileHasher,
) -> FoldInput:
    dataset_name = str(dataset["name"])
    dataset_id = str(dataset["dataset_id"])
    role = str(dataset["role"])
    n_spots, n_genes = (int(value) for value in dataset["expected_st_shape"])
    base_relative = f"{dataset_id}/fold{fold}"
    fold_dir = global_inputs.inputs_root / dataset_id / f"fold{fold}"
    truth_path = fold_dir / "full_truth.npy"
    split_path = fold_dir / "mode_a_split.json"
    normalization_path = fold_dir / "normalization_audit.json"
    artifact_path = fold_dir / "artifact_manifest.json"

    artifact = _load_mapping(artifact_path, context=f"{dataset_id} fold{fold} artifact manifest")
    if artifact.get("schema_version") != 1 or artifact.get("dataset_id") != dataset_id or artifact.get("fold") != fold:
        raise IntegrityError(
            "artifact_manifest_identity_mismatch",
            f"{dataset_id} fold{fold} artifact manifest identity mismatch",
        )
    input_records = artifact.get("input_artifacts")
    output_records = artifact.get("output_artifacts")
    if not isinstance(input_records, dict) or not isinstance(output_records, dict):
        raise IntegrityError(
            "artifact_manifest_records_invalid", f"{dataset_id} fold{fold} records are invalid"
        )

    material_dataset = _materialization_dataset(global_inputs.materialization, dataset_id)
    if (
        material_dataset.get("name") != dataset_name
        or material_dataset.get("shape") != [n_spots, n_genes]
    ):
        raise IntegrityError(
            "materialization_dataset_identity_mismatch",
            f"Materialization identity/shape mismatch for {dataset_id}",
        )
    material_fold = _materialization_fold(material_dataset, dataset_id, fold)
    if material_fold.get("shape") != [n_spots, n_genes] or material_fold.get("truth_dtype") != "float32":
        raise IntegrityError(
            "materialization_fold_shape_mismatch",
            f"Materialization shape/dtype mismatch for {dataset_id} fold{fold}",
        )
    global_outputs = material_fold.get("output_artifacts")
    if not isinstance(global_outputs, dict):
        raise IntegrityError(
            "materialization_output_records_invalid",
            f"Materialization outputs are invalid for {dataset_id} fold{fold}",
        )

    truth_sha = _verify_record(
        output_records.get("full_truth"),
        truth_path,
        expected_path=f"{base_relative}/full_truth.npy",
        context=f"{dataset_id} fold{fold} full truth",
        hasher=hasher,
    )
    split_sha = _verify_record(
        output_records.get("mode_a_split"),
        split_path,
        expected_path=f"{base_relative}/mode_a_split.json",
        context=f"{dataset_id} fold{fold} Mode-A split",
        hasher=hasher,
    )
    normalization_sha = _verify_record(
        output_records.get("normalization_audit"),
        normalization_path,
        expected_path=f"{base_relative}/normalization_audit.json",
        context=f"{dataset_id} fold{fold} normalization audit",
        hasher=hasher,
    )
    artifact_sha = _verify_record(
        global_outputs.get("artifact_manifest"),
        artifact_path,
        expected_path=f"{base_relative}/artifact_manifest.json",
        context=f"{dataset_id} fold{fold} artifact manifest",
        hasher=hasher,
    )
    for key, actual in (
        ("full_truth", truth_sha),
        ("mode_a_split", split_sha),
        ("normalization_audit", normalization_sha),
    ):
        local_size, local_sha = _declared_record(
            output_records.get(key),
            expected_path=f"{base_relative}/{Path(str(output_records[key]['path'])).name}",
            context=f"{dataset_id} fold{fold} local {key}",
        )
        global_size, global_sha = _declared_record(
            global_outputs.get(key),
            expected_path=f"{base_relative}/{Path(str(global_outputs[key]['path'])).name}",
            context=f"{dataset_id} fold{fold} global {key}",
        )
        if (local_size, local_sha) != (global_size, global_sha) or local_sha != actual:
            raise IntegrityError(
                "materialization_record_disagreement",
                f"Local/global {key} records disagree for {dataset_id} fold{fold}",
            )

    required_inputs = {
        "raw_counts",
        "scrna_counts",
        "locations",
        "gene_names",
        "frozen_split",
        "train_mask",
        "val_mask",
        "test_mask",
        "config",
        "protocol_a_preprocessing",
    }
    if set(input_records) != required_inputs:
        raise IntegrityError(
            "input_record_keys_mismatch",
            f"Input record keys differ from Protocol A for {dataset_id} fold{fold}",
        )
    config_relative = _project_relative(
        global_inputs.config_path,
        global_inputs.project_root,
        context="Protocol A config",
    )
    _verify_record(
        input_records.get("config"),
        global_inputs.config_path,
        expected_path=config_relative,
        context=f"{dataset_id} fold{fold} config",
        hasher=hasher,
    )

    source_relatives: dict[str, PurePosixPath] = {
        "raw_counts": _safe_relative(dataset["raw_counts"], context=f"{dataset_id}.raw_counts"),
        "scrna_counts": _safe_relative(dataset["scrna_counts"], context=f"{dataset_id}.scrna_counts"),
        "locations": _safe_relative(dataset["locations"], context=f"{dataset_id}.locations"),
        "gene_names": _safe_relative(dataset["gene_names"], context=f"{dataset_id}.gene_names"),
        "frozen_split": _format_fold_path(dataset["frozen_split"], fold, context=f"{dataset_id}.frozen_split"),
        "train_mask": _format_fold_path(dataset["train_mask"], fold, context=f"{dataset_id}.train_mask"),
        "val_mask": _format_fold_path(dataset["val_mask"], fold, context=f"{dataset_id}.val_mask"),
        "test_mask": _format_fold_path(dataset["test_mask"], fold, context=f"{dataset_id}.test_mask"),
    }
    source_declared_sha: dict[str, str] = {}
    for key, relative in source_relatives.items():
        _size, digest = _archive_record_matches(
            input_records.get(key),
            relative,
            global_inputs,
            context=f"{dataset_id} fold{fold} source {key}",
        )
        source_declared_sha[key] = digest

    gene_path, gene_sha = _verify_source_file(
        key="gene_names",
        relative=source_relatives["gene_names"],
        record=input_records["gene_names"],
        global_inputs=global_inputs,
        hasher=hasher,
        context=f"{dataset_id} gene names",
    )
    genes = _load_gene_names(gene_path, n_genes)
    gene_axis_sha = canonical_json_sha256(list(genes))
    if material_dataset.get("gene_axis_sha256") != gene_axis_sha:
        raise IntegrityError(
            "gene_axis_sha_mismatch", f"Materialization gene axis mismatch for {dataset_id}"
        )

    mask_arrays: dict[str, np.ndarray] = {}
    source_actual_sha: dict[str, str] = {
        **source_declared_sha,
        "gene_names": gene_sha,
    }
    for key, label in (
        ("train_mask", "train mask"),
        ("val_mask", "validation mask"),
        ("test_mask", "final-test mask"),
    ):
        path, digest = _verify_source_file(
            key=key,
            relative=source_relatives[key],
            record=input_records[key],
            global_inputs=global_inputs,
            hasher=hasher,
            context=f"{dataset_id} fold{fold} {label}",
        )
        mask_arrays[key] = _load_mask(path, n_genes, context=f"{dataset_id} fold{fold} {label}")
        source_actual_sha[key] = digest
    frozen_path, frozen_sha = _verify_source_file(
        key="frozen_split",
        relative=source_relatives["frozen_split"],
        record=input_records["frozen_split"],
        global_inputs=global_inputs,
        hasher=hasher,
        context=f"{dataset_id} fold{fold} frozen split",
    )
    source_actual_sha["frozen_split"] = frozen_sha
    frozen = _load_mapping(frozen_path, context=f"{dataset_id} fold{fold} frozen split")
    if frozen.get("fold") != fold or (
        "dataset" in frozen and frozen.get("dataset") not in {dataset_name, dataset_id}
    ):
        raise IntegrityError(
            "frozen_split_identity_mismatch", f"Frozen split identity mismatch for {dataset_id} fold{fold}"
        )
    train_idx = mask_arrays["train_mask"]
    val_idx = mask_arrays["val_mask"]
    test_idx = mask_arrays["test_mask"]
    for key, expected in (
        ("train_gene_idx", train_idx),
        ("val_gene_idx", val_idx),
        ("test_gene_idx", test_idx),
    ):
        observed = _strict_int_list(frozen, key, n_genes, context="frozen split")
        if not np.array_equal(observed, expected):
            raise IntegrityError(
                "frozen_split_mask_mismatch", f"Frozen split {key} differs from .npy mask"
            )
    for key, expected_idx in (
        ("train_genes", train_idx),
        ("val_genes", val_idx),
        ("test_genes", test_idx),
    ):
        _assert_names(frozen, key, [genes[int(index)] for index in expected_idx], context="frozen split")

    covered = np.concatenate((train_idx, val_idx, test_idx))
    if covered.size != n_genes or not np.array_equal(np.sort(covered), np.arange(n_genes)):
        raise IntegrityError(
            "fold_partition_mismatch", f"Masks do not form a complete partition for {dataset_id} fold{fold}"
        )

    split = _load_mapping(split_path, context=f"{dataset_id} fold{fold} Mode-A split")
    if (
        split.get("schema_version") != 1
        or split.get("protocol") != PROTOCOL
        or split.get("protocol_role") != "strict_primary_modeA"
        or split.get("dataset") != dataset_name
        or split.get("dataset_id") != dataset_id
        or split.get("fold") != fold
        or split.get("gene_count") != n_genes
        or split.get("gene_axis_sha256") != gene_axis_sha
    ):
        raise IntegrityError(
            "mode_a_split_identity_mismatch", f"Mode-A split identity mismatch for {dataset_id} fold{fold}"
        )
    split_train = _strict_int_list(split, "inner_train_gene_idx", n_genes, context="Mode-A split")
    split_val = _strict_int_list(split, "inner_validation_gene_idx", n_genes, context="Mode-A split")
    split_test = _strict_int_list(split, "final_test_gene_idx", n_genes, context="Mode-A split")
    hidden = np.concatenate((val_idx, test_idx))
    if not (
        np.array_equal(split_train, train_idx)
        and np.array_equal(split_val, val_idx)
        and np.array_equal(split_test, test_idx)
        and np.array_equal(_strict_int_list(split, "train_gene_idx", n_genes, context="Mode-A split"), train_idx)
        and np.array_equal(_strict_int_list(split, "val_gene_idx", n_genes, context="Mode-A split"), val_idx)
        and np.array_equal(_strict_int_list(split, "test_gene_idx", n_genes, context="Mode-A split"), hidden)
        and np.array_equal(_strict_int_list(split, "hidden_gene_idx", n_genes, context="Mode-A split"), hidden)
    ):
        raise IntegrityError(
            "mode_a_split_mask_mismatch", f"Mode-A split differs from frozen masks for {dataset_id} fold{fold}"
        )
    for key, expected_idx in (
        ("inner_train_genes", train_idx),
        ("inner_validation_genes", val_idx),
        ("final_test_genes", test_idx),
        ("hidden_genes", hidden),
        ("test_genes", hidden),
    ):
        _assert_names(split, key, [genes[int(index)] for index in expected_idx], context="Mode-A split")
    if split.get("source_frozen_sha256") != {
        key: source_actual_sha[key]
        for key in ("frozen_split", "train_mask", "val_mask", "test_mask")
    }:
        raise IntegrityError(
            "mode_a_source_sha_mismatch", f"Mode-A source hashes mismatch for {dataset_id} fold{fold}"
        )
    visibility = split.get("visibility")
    if not isinstance(visibility, dict) or (
        visibility.get("visible_st_gene_idx") != train_idx.tolist()
        or visibility.get("hidden_st_gene_idx") != hidden.tolist()
        or visibility.get("model_fit_uses_inner_train_only") is not True
        or visibility.get("validation_st_hidden_from_model_fit") is not True
        or visibility.get("final_test_st_hidden_from_all_fit_and_selection") is not True
    ):
        raise IntegrityError(
            "mode_a_visibility_mismatch", f"Mode-A visibility contract failed for {dataset_id} fold{fold}"
        )

    normalization = _load_mapping(
        normalization_path, context=f"{dataset_id} fold{fold} normalization audit"
    )
    if (
        normalization.get("schema_version") != 1
        or normalization.get("protocol") != PROTOCOL
        or normalization.get("dataset") != dataset_name
        or normalization.get("dataset_id") != dataset_id
        or normalization.get("fold") != fold
        or normalization.get("input_artifacts") != input_records
    ):
        raise IntegrityError(
            "normalization_identity_mismatch", f"Normalization audit mismatch for {dataset_id} fold{fold}"
        )
    protocol_a = normalization.get("protocol_a")
    if (
        not isinstance(protocol_a, dict)
        or protocol_a.get("protocol") != PROTOCOL
        or protocol_a.get("policy") != PROTOCOL_POLICY
        or protocol_a.get("denominator_gene_count") != int(train_idx.size)
    ):
        raise IntegrityError(
            "normalization_policy_mismatch", f"Protocol A denominator mismatch for {dataset_id} fold{fold}"
        )
    full_truth_meta = normalization.get("full_truth")
    if (
        not isinstance(full_truth_meta, dict)
        or full_truth_meta.get("shape") != [n_spots, n_genes]
        or full_truth_meta.get("dtype") != "float32"
        or full_truth_meta.get("gene_axis_sha256") != gene_axis_sha
    ):
        raise IntegrityError(
            "truth_metadata_mismatch", f"Truth metadata mismatch for {dataset_id} fold{fold}"
        )
    _verify_record(
        full_truth_meta.get("file"),
        truth_path,
        expected_path=f"{base_relative}/full_truth.npy",
        context=f"{dataset_id} fold{fold} truth metadata record",
        hasher=hasher,
    )
    _verify_record(
        normalization.get("mode_a_split"),
        split_path,
        expected_path=f"{base_relative}/mode_a_split.json",
        context=f"{dataset_id} fold{fold} normalization split record",
        hasher=hasher,
    )
    output_sha = normalization.get("output_sha256")
    if not isinstance(output_sha, dict) or output_sha.get("full_truth_npy") != truth_sha or output_sha.get("mode_a_split_json") != split_sha:
        raise IntegrityError(
            "normalization_output_sha_mismatch", f"Normalization output hashes mismatch for {dataset_id} fold{fold}"
        )

    try:
        truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise IntegrityError("truth_load_error", f"Could not load truth: {truth_path}") from error
    if tuple(truth.shape) != (n_spots, n_genes) or truth.dtype != np.float32:
        raise IntegrityError(
            "truth_shape_dtype_mismatch",
            f"Truth shape/dtype {truth.shape}/{truth.dtype} differs from {(n_spots, n_genes)}/float32",
        )
    payload_sha, nonfinite_count, negative_count = _float32_payload_sha256(truth)
    del truth
    if nonfinite_count:
        raise IntegrityError(
            "truth_nonfinite", f"Truth contains {nonfinite_count} non-finite values"
        )
    if negative_count:
        raise IntegrityError("truth_negative", f"Truth contains {negative_count} negative values")
    if (
        full_truth_meta.get("payload_sha256") != payload_sha
        or material_fold.get("truth_payload_sha256") != payload_sha
    ):
        raise IntegrityError(
            "truth_payload_sha_mismatch", f"Truth payload hash mismatch for {dataset_id} fold{fold}"
        )
    if material_fold.get("mode_a_split_sha256") != split_sha:
        raise IntegrityError(
            "mode_a_payload_sha_mismatch", f"Mode-A split payload hash mismatch for {dataset_id} fold{fold}"
        )
    if (
        material_fold.get("inner_train_gene_count") != int(train_idx.size)
        or material_fold.get("inner_validation_gene_count") != int(val_idx.size)
        or material_fold.get("final_test_gene_count") != int(test_idx.size)
    ):
        raise IntegrityError(
            "materialization_split_count_mismatch",
            f"Materialization split counts mismatch for {dataset_id} fold{fold}",
        )

    source_actual_sha.update(
        {
            "full_truth": truth_sha,
            "mode_a_split": split_sha,
            "normalization_audit": normalization_sha,
            "artifact_manifest": artifact_sha,
        }
    )
    return FoldInput(
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        role=role,
        fold=fold,
        expected_shape=(n_spots, n_genes),
        gene_names=genes,
        gene_axis_sha256=gene_axis_sha,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        truth_path=truth_path,
        truth_sha256=truth_sha,
        split_path=split_path,
        split_sha256=split_sha,
        normalization_path=normalization_path,
        normalization_sha256=normalization_sha,
        artifact_manifest_path=artifact_path,
        artifact_manifest_sha256=artifact_sha,
        materialization_sha256=global_inputs.materialization_sha256,
        source_sha256=source_actual_sha,
    )


def _input_report(outcome: InputOutcome) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": outcome.dataset_name,
        "dataset_id": outcome.dataset_id,
        "role": outcome.role,
        "fold": outcome.fold,
        "status": outcome.status,
        "issue_code": outcome.issue_code,
        "issue": outcome.issue,
    }
    if outcome.context is not None:
        context = outcome.context
        row.update(
            {
                "truth_path": str(context.truth_path),
                "truth_sha256": context.truth_sha256,
                "truth_shape": list(context.expected_shape),
                "train_gene_count": int(context.train_idx.size),
                "validation_gene_count": int(context.val_idx.size),
                "final_test_gene_count": int(context.test_idx.size),
                "fixed_test_indices_verified": True,
                "gene_axis_sha256": context.gene_axis_sha256,
            }
        )
    return row


def _validate_all_inputs(
    datasets: Sequence[Mapping[str, Any]],
    global_inputs: GlobalInputs,
    hasher: FileHasher,
    progress: Callable[[str], None] | None,
) -> dict[tuple[str, int], InputOutcome]:
    outcomes: dict[tuple[str, int], InputOutcome] = {}
    for dataset in datasets:
        dataset_id = str(dataset["dataset_id"])
        dataset_name = str(dataset["name"])
        role = str(dataset["role"])
        for fold in REQUIRED_FOLDS:
            if progress:
                progress(f"[input] {dataset_id} fold{fold}")
            try:
                context = _validate_fold_input(dataset, fold, global_inputs, hasher)
            except MissingArtifact as error:
                outcome = InputOutcome(
                    dataset_name, dataset_id, role, fold, "missing", issue_code=error.code, issue=str(error)
                )
            except IntegrityError as error:
                outcome = InputOutcome(
                    dataset_name, dataset_id, role, fold, "invalid", issue_code=error.code, issue=str(error)
                )
            else:
                outcome = InputOutcome(dataset_name, dataset_id, role, fold, "ready", context=context)
            outcomes[(dataset_id, fold)] = outcome

        dataset_outcomes = [outcomes[(dataset_id, fold)] for fold in REQUIRED_FOLDS]
        if all(item.status == "ready" and item.context is not None for item in dataset_outcomes):
            tests = np.concatenate([item.context.test_idx for item in dataset_outcomes if item.context])
            n_genes = int(dataset["expected_st_shape"][1])
            if tests.size != n_genes or not np.array_equal(np.sort(tests), np.arange(n_genes)):
                for item in dataset_outcomes:
                    item.status = "invalid"
                    item.issue_code = "cross_fold_test_partition_mismatch"
                    item.issue = (
                        f"Final-test masks across five folds do not partition the gene axis for {dataset_id}"
                    )
                    item.context = None
    return outcomes


def _manifest_file_record(
    files: object, expected_suffix: str, *, context: str
) -> Mapping[str, Any]:
    if not isinstance(files, list):
        raise IntegrityError("completion_files_invalid", f"{context} files list is invalid")
    matches = [
        item
        for item in files
        if isinstance(item, dict)
        and _record_path_matches(item.get("path"), expected_suffix, suffix_ok=True)
    ]
    if len(matches) != 1:
        raise IntegrityError(
            "completion_file_record_missing",
            f"{context} must contain exactly one record ending in {expected_suffix}",
        )
    return matches[0]


def _npz_scalar(archive: Any, key: str, *, context: str) -> object:
    if key not in archive.files:
        raise IntegrityError("prediction_scalar_missing", f"{context} is missing scalar {key}")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise IntegrityError("prediction_scalar_invalid", f"{context}.{key} is not scalar")
    return value.reshape(()).item()


def _npz_index(archive: Any, key: str, expected: np.ndarray, *, context: str) -> None:
    if key not in archive.files:
        raise IntegrityError("prediction_index_missing", f"{context} is missing {key}")
    observed = np.asarray(archive[key])
    if observed.dtype != np.int64 or not np.array_equal(observed, expected):
        raise IntegrityError(
            "prediction_index_mismatch", f"{context}.{key} differs from frozen indices"
        )


def _load_genespt_prediction(
    method: MethodSpec,
    fold_input: FoldInput,
    genespt_root: Path,
    global_inputs: GlobalInputs,
    hasher: FileHasher,
) -> tuple[np.ndarray, dict[str, Any]]:
    assert method.model is not None
    job_root = (
        genespt_root
        / GENESPT_MODE
        / fold_input.dataset_id
        / f"fold{fold_input.fold}"
    )
    completion_path = job_root / "completion_manifest.json"
    if not completion_path.is_file():
        failure_path = job_root / "run_failure.json"
        code = "genespt_failed" if failure_path.is_file() else "genespt_completion_missing"
        detail = "failure manifest present" if failure_path.is_file() else "completion manifest absent"
        raise MissingArtifact(code, f"{method.name} {fold_input.dataset_id} fold{fold_input.fold}: {detail}")
    completion = _load_mapping(
        completion_path,
        context=f"{method.name} {fold_input.dataset_id} fold{fold_input.fold} completion manifest",
    )
    expected_identity = {
        "schema_version": 2,
        "status": "complete",
        "protocol": PROTOCOL,
        "mode": GENESPT_MODE,
        "dataset_id": fold_input.dataset_id,
        "fold": fold_input.fold,
    }
    for key, expected in expected_identity.items():
        if completion.get(key) != expected:
            raise IntegrityError(
                "genespt_completion_identity_mismatch",
                f"{method.name} completion {key}={completion.get(key)!r}, expected {expected!r}",
            )
    signature_payload = {
        key: completion.get(key)
        for key in (
            "schema_version",
            "protocol",
            "mode",
            "dataset_id",
            "fold",
            "command",
            "cwd",
            "environment",
            "provenance",
        )
    }
    if completion.get("job_signature_sha256") != canonical_json_sha256(signature_payload):
        raise IntegrityError(
            "genespt_job_signature_mismatch", f"{method.name} completion job signature mismatch"
        )
    command = completion.get("command")
    if not isinstance(command, list) or "--st-normalization-scope" not in command:
        raise IntegrityError(
            "genespt_command_invalid", f"{method.name} completion command is invalid"
        )
    scope_position = command.index("--st-normalization-scope") + 1
    if scope_position >= len(command) or command[scope_position] != "train_genes":
        raise IntegrityError(
            "genespt_normalization_scope_mismatch", f"{method.name} did not use train_genes normalization"
        )
    provenance = completion.get("provenance")
    if not isinstance(provenance, dict):
        raise IntegrityError("genespt_provenance_missing", "GeneSPT provenance is missing")
    config_record = provenance.get("config")
    _size, config_sha = _declared_record(
        config_record,
        expected_path=_project_relative(
            global_inputs.config_path,
            global_inputs.project_root,
            context="Protocol A config",
        ),
        context="GeneSPT config provenance",
        suffix_ok=True,
    )
    if config_sha != global_inputs.config_sha256:
        raise IntegrityError("genespt_config_sha_mismatch", "GeneSPT config hash mismatch")
    if provenance.get("gene_axis_sha256") != fold_input.gene_axis_sha256:
        raise IntegrityError("genespt_gene_axis_mismatch", "GeneSPT gene axis hash mismatch")
    masks = provenance.get("masks")
    if not isinstance(masks, dict):
        raise IntegrityError("genespt_masks_missing", "GeneSPT mask provenance is missing")
    for key in ("frozen_split", "train_mask", "val_mask", "test_mask"):
        record = masks.get(key)
        if not isinstance(record, dict) or record.get("sha256") != fold_input.source_sha256[key]:
            raise IntegrityError(
                "genespt_mask_sha_mismatch", f"GeneSPT {key} hash differs from frozen input"
            )

    relative_prediction = (
        f"{GENESPT_PREFIX}_prediction_matrices/{method.model}/"
        f"fold{fold_input.fold}/prediction.npz"
    )
    prediction_path = job_root.joinpath(*PurePosixPath(relative_prediction).parts)
    outputs = completion.get("outputs")
    if not isinstance(outputs, dict):
        raise IntegrityError("genespt_outputs_missing", "GeneSPT completion outputs are missing")
    payloads = outputs.get("prediction_payloads")
    if not isinstance(payloads, dict) or not isinstance(payloads.get(method.model), dict):
        raise IntegrityError(
            "genespt_payload_manifest_missing", f"Completion has no payload metadata for {method.model}"
        )
    payload_meta = payloads[method.model]
    if (
        payload_meta.get("model") != method.model
        or payload_meta.get("fold") != fold_input.fold
        or payload_meta.get("legacy_test_shape")
        != [fold_input.expected_shape[0], int(fold_input.test_idx.size)]
        or not _record_path_matches(payload_meta.get("path"), relative_prediction, suffix_ok=True)
    ):
        raise IntegrityError(
            "genespt_payload_manifest_mismatch", f"Payload metadata mismatch for {method.model}"
        )
    file_record = _manifest_file_record(
        outputs.get("files"), relative_prediction, context=f"{method.name} completion"
    )
    prediction_sha = _verify_record(
        file_record,
        prediction_path,
        expected_path=relative_prediction,
        context=f"{method.name} prediction",
        hasher=hasher,
        suffix_ok=True,
    )
    completion_sha = hasher.hash(completion_path, context=f"{method.name} completion manifest")

    try:
        with np.load(prediction_path, allow_pickle=True) as archive:
            if str(_npz_scalar(archive, "model", context=method.name)) != method.model:
                raise IntegrityError("prediction_method_mismatch", f"{method.name} model identity mismatch")
            if int(_npz_scalar(archive, "fold", context=method.name)) != fold_input.fold:
                raise IntegrityError("prediction_fold_mismatch", f"{method.name} fold identity mismatch")
            if str(_npz_scalar(archive, "psp_descriptor", context=method.name)) != "pca32_nmf32":
                raise IntegrityError("prediction_descriptor_mismatch", f"{method.name} descriptor mismatch")
            if str(_npz_scalar(archive, "posthoc_calibration", context=method.name)) != "none":
                raise IntegrityError("prediction_calibration_present", f"{method.name} has post-hoc calibration")
            if str(_npz_scalar(archive, "readout", context=method.name)) != "identity":
                raise IntegrityError("prediction_readout_mismatch", f"{method.name} readout is not identity")
            _npz_index(archive, "train_gene_idx", fold_input.train_idx, context=method.name)
            _npz_index(archive, "val_gene_idx", fold_input.val_idx, context=method.name)
            _npz_index(archive, "test_gene_idx", fold_input.test_idx, context=method.name)
            if "test_genes" not in archive.files or [str(value) for value in np.asarray(archive["test_genes"]).tolist()] != list(fold_input.test_gene_names):
                raise IntegrityError(
                    "prediction_test_gene_order_mismatch", f"{method.name} test gene order mismatch"
                )
            if "prediction" not in archive.files:
                raise IntegrityError("prediction_matrix_missing", f"{method.name} has no prediction key")
            prediction = np.asarray(archive["prediction"])
            expected_shape = (fold_input.expected_shape[0], int(fold_input.test_idx.size))
            if prediction.shape != expected_shape or prediction.dtype != np.float32:
                raise IntegrityError(
                    "prediction_shape_dtype_mismatch",
                    f"{method.name} prediction is {prediction.shape}/{prediction.dtype}; expected {expected_shape}/float32",
                )
            if not np.isfinite(prediction).all():
                raise IntegrityError("prediction_nonfinite", f"{method.name} prediction contains NaN/Inf")
            alias_key = (
                "base_prediction_test"
                if method.model == GENESPT_GC_MODEL
                else "selected_prediction_test"
            )
            if alias_key not in archive.files:
                raise IntegrityError(
                    "prediction_raw_alias_missing", f"{method.name} is missing {alias_key}"
                )
            alias = np.asarray(archive[alias_key])
            if alias.shape != prediction.shape or alias.dtype != np.float32 or not np.array_equal(alias, prediction):
                raise IntegrityError(
                    "prediction_raw_alias_mismatch", f"{method.name} prediction differs from {alias_key}"
                )
            if method.model == GENESPT_FULL_MODEL and (
                str(_npz_scalar(archive, "selected_rule_frozen_from_split", context=method.name))
                != "validation"
                or str(_npz_scalar(archive, "selected_train_coefficient_source", context=method.name))
                != "ridge_descriptor_prediction_on_train_genes"
            ):
                raise IntegrityError(
                    "prediction_selection_provenance_mismatch",
                    "GeneSPT selected rule was not frozen from validation",
                )
            prediction_copy = prediction.astype(np.float32, copy=True)
    except AuditError:
        raise
    except Exception as error:
        raise IntegrityError(
            "prediction_npz_load_error", f"Could not load {method.name} prediction: {prediction_path}"
        ) from error
    return prediction_copy, {
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "completion_manifest": str(completion_path),
        "completion_manifest_sha256": completion_sha,
        "matrix_scope": "frozen_final_test_genes",
        "prediction_key": "prediction",
        "raw_alias_key": (
            "base_prediction_test"
            if method.model == GENESPT_GC_MODEL
            else "selected_prediction_test"
        ),
        "model": method.model,
        "readout": "identity",
        "posthoc_calibration": "none",
    }


def _matrix_finite(matrix: np.ndarray) -> bool:
    for start in range(0, matrix.shape[0], ARRAY_ROW_CHUNK):
        if not np.isfinite(matrix[start : start + ARRAY_ROW_CHUNK]).all():
            return False
    return True


def _declared_baseline_output_sha(audit: Mapping[str, Any]) -> str:
    direct = audit.get("output_matrix_sha256")
    if isinstance(direct, str) and SHA256_RE.fullmatch(direct):
        return direct
    outputs = audit.get("output_sha256")
    if isinstance(outputs, dict):
        value = outputs.get("imputed_expression.npy")
        if isinstance(value, dict):
            value = value.get("sha256")
        if isinstance(value, str) and SHA256_RE.fullmatch(value):
            return value
    raise IntegrityError(
        "baseline_output_sha_missing", "Baseline adapter audit has no valid prediction SHA256"
    )


def _adapter_metrics_were_skipped(audit: Mapping[str, Any]) -> bool:
    value = audit.get("adapter_metrics_skipped")
    scope = audit.get("scope")
    if value is None and isinstance(scope, dict):
        value = scope.get("adapter_metrics_skipped")
    return value is True


def _adapter_declares_nonfinite_output(audit: Mapping[str, Any]) -> bool:
    declarations = []
    if "prediction_finite" in audit:
        declarations.append(audit.get("prediction_finite"))
    finite_checks = audit.get("finite_checks")
    if isinstance(finite_checks, dict):
        declarations.extend(
            finite_checks.get(key) for key in ("normalized", "predicted", "imputed")
        )
    for key in ("normalized_finite", "predicted_finite", "imputed_finite"):
        if key in audit:
            declarations.append(audit.get(key))
    return any(value is False for value in declarations)


def _adapter_reports_fallback(audit: Mapping[str, Any]) -> bool:
    for key in (
        "fallback_used",
        "truth_copy_fallback_used",
        "hidden_gene_zero_fallback_used",
    ):
        if audit.get(key) is True:
            return True
    for key in ("fallback", "truth_fallback"):
        value = audit.get(key)
        if value is True:
            return True
        if isinstance(value, dict) and value.get("used") is True:
            return True
    return False


def _load_baseline_prediction(
    method: MethodSpec,
    fold_input: FoldInput,
    baseline_root: Path,
    global_inputs: GlobalInputs,
    hasher: FileHasher,
) -> tuple[np.ndarray, dict[str, Any]]:
    task_key = f"{method.name}__{fold_input.dataset_id}__fold{fold_input.fold}"
    output_dir = baseline_root / method.name / fold_input.dataset_id / f"fold{fold_input.fold}"
    prediction_path = output_dir / "imputed_expression.npy"
    audit_path = output_dir / "adapter_run_audit.json"
    status_path = baseline_root / "_scheduler" / "status" / f"{task_key}.json"
    if not status_path.is_file():
        raise MissingArtifact(
            "baseline_status_missing", f"{task_key}: scheduler completion status is absent"
        )
    status = _load_mapping(status_path, context=f"{task_key} scheduler status")
    scheduler_state = status.get("status")
    if scheduler_state != "completed":
        raise MissingArtifact(
            f"baseline_status_{scheduler_state or 'unknown'}",
            f"{task_key}: scheduler status is {scheduler_state!r}",
        )
    expected_identity = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "task": task_key,
        "method": method.name,
        "dataset": fold_input.dataset_name,
        "dataset_id": fold_input.dataset_id,
        "fold": fold_input.fold,
    }
    for key, expected in expected_identity.items():
        if status.get(key) != expected:
            raise IntegrityError(
                "baseline_status_identity_mismatch",
                f"{task_key}: status {key}={status.get(key)!r}, expected {expected!r}",
            )
    command = status.get("command")
    if not isinstance(command, list) or status.get("command_sha256") != canonical_json_sha256(command):
        raise IntegrityError("baseline_command_sha_mismatch", f"{task_key}: command hash mismatch")
    input_hashes = status.get("input_file_sha256")
    if not isinstance(input_hashes, dict) or status.get("input_sha256") != canonical_json_sha256(input_hashes):
        raise IntegrityError("baseline_input_signature_mismatch", f"{task_key}: input signature mismatch")
    if status.get("config_sha256") != global_inputs.config_sha256:
        raise IntegrityError("baseline_config_sha_mismatch", f"{task_key}: config hash mismatch")
    required_input_hashes = {
        "full_truth": fold_input.truth_sha256,
        "mode_a_split": fold_input.split_sha256,
        "normalization_audit": fold_input.normalization_sha256,
        "artifact_manifest": fold_input.artifact_manifest_sha256,
        "materialization_manifest": fold_input.materialization_sha256,
        "raw_counts": fold_input.source_sha256["raw_counts"],
        "scrna_counts": fold_input.source_sha256["scrna_counts"],
        "locations": fold_input.source_sha256["locations"],
        "gene_names": fold_input.source_sha256["gene_names"],
        "frozen_split": fold_input.source_sha256["frozen_split"],
        "train_mask": fold_input.source_sha256["train_mask"],
        "val_mask": fold_input.source_sha256["val_mask"],
        "test_mask": fold_input.source_sha256["test_mask"],
    }
    for key, expected_sha in required_input_hashes.items():
        if input_hashes.get(key) != expected_sha:
            raise IntegrityError(
                "baseline_input_sha_mismatch", f"{task_key}: {key} hash differs from evaluated input"
            )
    if not prediction_path.is_file() or not audit_path.is_file():
        raise IntegrityError(
            "baseline_completed_output_missing", f"{task_key}: completed status points to missing output/audit"
        )
    prediction_sha = hasher.hash(prediction_path, context=f"{task_key} prediction")
    audit_sha = hasher.hash(audit_path, context=f"{task_key} adapter audit")
    prediction_fp = hasher.fingerprint(prediction_path, context=f"{task_key} prediction")
    audit_fp = hasher.fingerprint(audit_path, context=f"{task_key} adapter audit")
    if (
        status.get("prediction_sha256") != prediction_sha
        or status.get("prediction_bytes") != prediction_fp.size_bytes
        or status.get("audit_sha256") != audit_sha
        or status.get("audit_bytes") != audit_fp.size_bytes
    ):
        raise IntegrityError(
            "baseline_status_output_sha_mismatch", f"{task_key}: scheduler output hashes/bytes mismatch"
        )
    if not _record_path_matches(
        status.get("prediction_path"),
        f"{method.name}/{fold_input.dataset_id}/fold{fold_input.fold}/imputed_expression.npy",
        suffix_ok=True,
    ) or not _record_path_matches(
        status.get("audit_path"),
        f"{method.name}/{fold_input.dataset_id}/fold{fold_input.fold}/adapter_run_audit.json",
        suffix_ok=True,
    ):
        raise IntegrityError("baseline_status_path_mismatch", f"{task_key}: status output paths mismatch")

    audit = _load_mapping(audit_path, context=f"{task_key} adapter audit")
    if (
        audit.get("adapter") != method.name
        or audit.get("protocol_role") != "strict_primary_modeA"
        or audit.get("eligible_for_strict_primary") is not True
        or audit.get("model_gene_scope") != "train_indices"
        or audit.get("st_normalization_scope") != "train_genes"
        or not _adapter_metrics_were_skipped(audit)
        or _adapter_declares_nonfinite_output(audit)
        or audit.get("imputed_matrix_shape") != list(fold_input.expected_shape)
    ):
        raise IntegrityError(
            "baseline_adapter_contract_mismatch", f"{task_key}: adapter audit contract mismatch"
        )
    if _adapter_reports_fallback(audit):
        raise IntegrityError(
            "baseline_adapter_fallback_used",
            f"{task_key}: adapter audit reports a fallback in the formal prediction",
        )
    normalization = audit.get("normalization_audit")
    if (
        not isinstance(normalization, dict)
        or normalization.get("policy") != PROTOCOL_POLICY
        or normalization.get("denominator_gene_count") != int(fold_input.train_idx.size)
    ):
        raise IntegrityError(
            "baseline_normalization_mismatch", f"{task_key}: normalization audit mismatch"
        )
    if "validation_gene_count" in audit and audit.get("validation_gene_count") != int(fold_input.val_idx.size):
        raise IntegrityError("baseline_validation_count_mismatch", f"{task_key}: validation count mismatch")
    if "final_test_gene_count" in audit and audit.get("final_test_gene_count") != int(fold_input.test_idx.size):
        raise IntegrityError("baseline_test_count_mismatch", f"{task_key}: final-test count mismatch")
    if _declared_baseline_output_sha(audit) != prediction_sha:
        raise IntegrityError("baseline_adapter_output_sha_mismatch", f"{task_key}: adapter output hash mismatch")
    adapter_inputs = audit.get("input_sha256")
    expected_adapter_inputs = {
        "locations_path": input_hashes.get("locations"),
        "st_data": input_hashes.get("raw_counts"),
        "sc_data": input_hashes.get("scrna_counts"),
        "gene_split_json": input_hashes.get("mode_a_split"),
        "train_gene_idx_path": input_hashes.get("train_mask"),
        "val_gene_idx_path": input_hashes.get("val_mask"),
        "test_gene_idx_path": input_hashes.get("test_mask"),
    }
    if adapter_inputs != expected_adapter_inputs:
        raise IntegrityError(
            "baseline_adapter_input_sha_mismatch", f"{task_key}: adapter input hashes mismatch"
        )

    try:
        matrix = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise IntegrityError("baseline_prediction_load_error", f"Could not load {task_key} matrix") from error
    if tuple(matrix.shape) != fold_input.expected_shape or matrix.dtype != np.float32:
        raise IntegrityError(
            "prediction_shape_dtype_mismatch",
            f"{task_key}: matrix is {matrix.shape}/{matrix.dtype}, expected {fold_input.expected_shape}/float32",
        )
    if not _matrix_finite(matrix):
        raise IntegrityError("prediction_nonfinite", f"{task_key}: matrix contains NaN/Inf")
    prediction = np.asarray(matrix[:, fold_input.test_idx], dtype=np.float32)
    del matrix
    expected_test_shape = (fold_input.expected_shape[0], int(fold_input.test_idx.size))
    if prediction.shape != expected_test_shape:
        raise IntegrityError(
            "prediction_test_extraction_mismatch",
            f"{task_key}: final-test extraction is {prediction.shape}, expected {expected_test_shape}",
        )
    return prediction, {
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "adapter_audit": str(audit_path),
        "adapter_audit_sha256": audit_sha,
        "scheduler_status": str(status_path),
        "scheduler_status_sha256": hasher.hash(status_path, context=f"{task_key} scheduler status"),
        "matrix_scope": "full_frozen_gene_axis",
        "extraction": "columns_at_frozen_final_test_gene_idx",
        "readout": "adapter_raw_output",
        "posthoc_calibration": "none_per_evaluator",
    }


def _load_metrics_module(path: Path) -> Any:
    if not path.is_file():
        raise IntegrityError("metrics_module_missing", f"Centralized evaluator is missing: {path}")
    spec = importlib.util.spec_from_file_location("protocol_a_centralized_metrics", path)
    if spec is None or spec.loader is None:
        raise IntegrityError("metrics_module_import_error", f"Cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise IntegrityError("metrics_module_import_error", f"Cannot import evaluator: {path}") from error
    finally:
        sys.dont_write_bytecode = previous
    if not callable(getattr(module, "evaluate_prediction", None)):
        raise IntegrityError(
            "metrics_function_missing", f"Evaluator has no callable evaluate_prediction: {path}"
        )
    return module


def _metric_value(summary: Mapping[str, Any], metric: str) -> float:
    candidates = (metric,) if metric != "JSD" else ("JSD", "JS", "JS/JSD")
    for key in candidates:
        if key in summary:
            return float(summary[key])
    raise IntegrityError("metric_summary_missing", f"Centralized evaluator summary is missing {metric}")


def _evaluate_aligned(
    metrics_module: Any,
    truth: np.ndarray,
    prediction: np.ndarray,
    gene_names: Sequence[str],
) -> dict[str, Any]:
    if truth.shape != prediction.shape:
        raise IntegrityError(
            "aligned_shape_mismatch", f"Aligned truth/prediction shapes differ: {truth.shape} vs {prediction.shape}"
        )
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise IntegrityError("aligned_nonfinite", "Aligned truth/prediction contains NaN/Inf")
    try:
        per_gene, summary_frame = metrics_module.evaluate_prediction(
            truth, prediction, gene_names=list(gene_names)
        )
    except Exception as error:
        raise IntegrityError(
            "centralized_evaluator_error", f"evaluate_prediction failed: {type(error).__name__}: {error}"
        ) from error
    try:
        if len(summary_frame) != 1 or len(per_gene) != truth.shape[1]:
            raise ValueError("unexpected result dimensions")
        summary = summary_frame.iloc[0].to_dict()
        total = int(summary["total_genes"])
        eligible = int(summary["eligible_genes"])
        scored = int(summary["scored_genes"])
        constant = int(summary["constant_prediction_genes"])
        coverage = float(summary["coverage"])
    except Exception as error:
        raise IntegrityError(
            "centralized_summary_invalid", "Centralized evaluator returned an invalid summary"
        ) from error
    if total != truth.shape[1] or eligible != total or scored != eligible:
        raise IntegrityError(
            "centralized_coverage_incomplete",
            f"Centralized evaluator scored {scored}/{eligible}/{total} genes",
        )
    metric_values: dict[str, float] = {}
    metric_counts: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        value = _metric_value(summary, metric)
        metric_eligible = int(summary[f"{metric}_eligible"])
        metric_scored = int(summary[f"{metric}_scored"])
        metric_constant = int(summary[f"{metric}_constant_prediction"])
        metric_coverage = float(summary[f"{metric}_coverage"])
        if metric_scored != metric_eligible:
            raise IntegrityError(
                "centralized_metric_coverage_incomplete",
                f"{metric} scored {metric_scored}/{metric_eligible} eligible genes",
            )
        metric_values[metric] = value
        metric_counts[metric] = {
            "eligible_gene_count": metric_eligible,
            "valid_gene_count": metric_scored,
            "missing_gene_count": metric_eligible - metric_scored,
            "constant_prediction_count": metric_constant,
            "coverage": metric_coverage,
        }
    return {
        "SPCC": metric_values["SPCC"],
        "RMSE": metric_values["RMSE"],
        "JSD": metric_values["JSD"],
        "JS": metric_values["JSD"],
        "JS/JSD": metric_values["JSD"],
        "SSIM": metric_values["SSIM"],
        "expected_gene_count": total,
        "eligible_gene_count": eligible,
        "valid_gene_count": scored,
        "missing_gene_count": total - scored,
        "constant_prediction_count": constant,
        "coverage": coverage,
        "metric_gene_counts": metric_counts,
    }


def _base_prediction_row(method: MethodSpec, fold_input: FoldInput) -> dict[str, Any]:
    return {
        "dataset": fold_input.dataset_name,
        "dataset_id": fold_input.dataset_id,
        "role": fold_input.role,
        "fold": fold_input.fold,
        "method": method.name,
        "method_kind": method.kind,
        "status": "missing",
        "issue_code": "",
        "issue": "",
        "fixed_test_indices_verified": True,
        "test_gene_count": int(fold_input.test_idx.size),
        "SPCC": None,
        "RMSE": None,
        "JSD": None,
        "JS": None,
        "JS/JSD": None,
        "SSIM": None,
        "expected_gene_count": int(fold_input.test_idx.size),
        "eligible_gene_count": None,
        "valid_gene_count": None,
        "missing_gene_count": None,
        "constant_prediction_count": None,
        "coverage": None,
        "metric_gene_counts": {},
    }


def _blocked_prediction_row(method: MethodSpec, outcome: InputOutcome) -> dict[str, Any]:
    status = "invalid" if outcome.status == "invalid" else "missing"
    return {
        "dataset": outcome.dataset_name,
        "dataset_id": outcome.dataset_id,
        "role": outcome.role,
        "fold": outcome.fold,
        "method": method.name,
        "method_kind": method.kind,
        "status": status,
        "issue_code": f"input_{outcome.issue_code or outcome.status}",
        "issue": f"Prediction not evaluated because fold input is {outcome.status}: {outcome.issue}",
        "fixed_test_indices_verified": False,
        "test_gene_count": None,
        "SPCC": None,
        "RMSE": None,
        "JSD": None,
        "JS": None,
        "JS/JSD": None,
        "SSIM": None,
        "expected_gene_count": None,
        "eligible_gene_count": None,
        "valid_gene_count": None,
        "missing_gene_count": None,
        "constant_prediction_count": None,
        "coverage": None,
        "metric_gene_counts": {},
    }


def _aggregate_rows(
    fold_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[Mapping[str, Any]],
    methods: Sequence[MethodSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_id = str(dataset["dataset_id"])
        for method in methods:
            group = [
                row
                for row in fold_rows
                if row["dataset_id"] == dataset_id and row["method"] == method.name
            ]
            evaluated = [row for row in group if row["status"] == "evaluated"]
            invalid = [row for row in group if row["status"] == "invalid"]
            finite_metrics = len(evaluated) == len(REQUIRED_FOLDS) and all(
                row[metric] is not None and math.isfinite(float(row[metric]))
                for row in evaluated
                for metric in METRICS
            )
            complete = (
                finite_metrics
                and len(group) == len(REQUIRED_FOLDS)
                and {int(row["fold"]) for row in evaluated} == set(REQUIRED_FOLDS)
            )
            status = "complete" if complete else ("invalid" if invalid else "missing")
            aggregate: dict[str, Any] = {
                "dataset": dataset["name"],
                "dataset_id": dataset_id,
                "role": dataset["role"],
                "method": method.name,
                "status": status,
                "aggregation": "arithmetic_mean_of_five_fold_medians",
                "folds_expected": len(REQUIRED_FOLDS),
                "folds_evaluated": len(evaluated),
                "missing_fold_count": len(REQUIRED_FOLDS) - len(evaluated),
                "SPCC": None,
                "RMSE": None,
                "JSD": None,
                "JS": None,
                "JS/JSD": None,
                "SSIM": None,
                "metric_std_ddof0": {},
                "expected_gene_count": None,
                "eligible_gene_count": None,
                "valid_gene_count": None,
                "missing_gene_count": None,
                "constant_prediction_count": None,
                "coverage": None,
                "metric_gene_counts": {},
            }
            if complete:
                means = {
                    metric: float(np.mean([float(row[metric]) for row in evaluated]))
                    for metric in METRICS
                }
                aggregate.update(means)
                aggregate["JS"] = means["JSD"]
                aggregate["JS/JSD"] = means["JSD"]
                aggregate["metric_std_ddof0"] = {
                    metric: float(np.std([float(row[metric]) for row in evaluated], ddof=0))
                    for metric in METRICS
                }
                for key in (
                    "expected_gene_count",
                    "eligible_gene_count",
                    "valid_gene_count",
                    "missing_gene_count",
                    "constant_prediction_count",
                ):
                    aggregate[key] = int(sum(int(row[key]) for row in evaluated))
                eligible = int(aggregate["eligible_gene_count"])
                valid = int(aggregate["valid_gene_count"])
                aggregate["coverage"] = float(valid / eligible) if eligible else None
                metric_counts: dict[str, dict[str, Any]] = {}
                for metric in METRICS:
                    eligible_metric = int(
                        sum(int(row["metric_gene_counts"][metric]["eligible_gene_count"]) for row in evaluated)
                    )
                    valid_metric = int(
                        sum(int(row["metric_gene_counts"][metric]["valid_gene_count"]) for row in evaluated)
                    )
                    metric_counts[metric] = {
                        "eligible_gene_count": eligible_metric,
                        "valid_gene_count": valid_metric,
                        "missing_gene_count": eligible_metric - valid_metric,
                        "constant_prediction_count": int(
                            sum(
                                int(row["metric_gene_counts"][metric]["constant_prediction_count"])
                                for row in evaluated
                            )
                        ),
                        "coverage": (
                            float(valid_metric / eligible_metric) if eligible_metric else None
                        ),
                    }
                aggregate["metric_gene_counts"] = metric_counts
            rows.append(aggregate)
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def evaluate_protocol_a(
    *,
    config_path: Path = DEFAULT_CONFIG,
    project_root: Path = GENESPT_ROOT,
    inputs_root: Path = DEFAULT_INPUTS_ROOT,
    genespt_root: Path = DEFAULT_GENESPT_OUTPUT_ROOT,
    baseline_root: Path = DEFAULT_BASELINE_OUTPUT_ROOT,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    dataset_selectors: Sequence[str] | None = None,
    method_selectors: Sequence[str] | None = None,
    metrics_module: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Audit selected Protocol A runs and return a JSON-safe report.

    Selection is for incremental inspection only.  Folds are intentionally not
    selectable: every selected dataset/method is always assessed as a five-fold
    unit.
    """

    project = Path(project_root).resolve(strict=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = project / config_file
    config_file = config_file.resolve(strict=True)
    active_inputs = Path(inputs_root).resolve(strict=False)
    active_genespt = Path(genespt_root).resolve(strict=False)
    active_baselines = Path(baseline_root).resolve(strict=False)
    active_metrics = Path(metrics_path).resolve(strict=False)
    hasher = FileHasher()

    global_inputs = _prepare_global_inputs(
        config_path=config_file,
        project_root=project,
        inputs_root=active_inputs,
        hasher=hasher,
    )
    configured = _validate_config(global_inputs.config)
    datasets = _select_datasets(configured, dataset_selectors)
    methods = _select_methods(method_selectors)

    if metrics_module is None:
        evaluator_sha = hasher.hash(active_metrics, context="centralized evaluator")
        evaluator = _load_metrics_module(active_metrics)
        evaluator_path_value = str(active_metrics)
    else:
        evaluator = metrics_module
        if not callable(getattr(evaluator, "evaluate_prediction", None)):
            raise IntegrityError(
                "metrics_function_missing", "Injected evaluator has no evaluate_prediction"
            )
        evaluator_sha = None
        evaluator_path_value = "<injected>"

    input_outcomes = _validate_all_inputs(datasets, global_inputs, hasher, progress)
    fold_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_id = str(dataset["dataset_id"])
        for fold in REQUIRED_FOLDS:
            outcome = input_outcomes[(dataset_id, fold)]
            if outcome.status != "ready" or outcome.context is None:
                fold_rows.extend(_blocked_prediction_row(method, outcome) for method in methods)
                continue
            fold_input = outcome.context
            try:
                truth_matrix = np.load(fold_input.truth_path, mmap_mode="r", allow_pickle=False)
                truth = np.asarray(truth_matrix[:, fold_input.test_idx], dtype=np.float32)
                del truth_matrix
            except Exception as error:
                for method in methods:
                    row = _base_prediction_row(method, fold_input)
                    row.update(
                        {
                            "status": "invalid",
                            "issue_code": "truth_reload_error",
                            "issue": f"Could not reload aligned truth: {type(error).__name__}: {error}",
                        }
                    )
                    fold_rows.append(row)
                continue
            for method in methods:
                if progress:
                    progress(f"[evaluate] {dataset_id} fold{fold} {method.name}")
                row = _base_prediction_row(method, fold_input)
                try:
                    if method.kind == "genespt":
                        prediction, provenance = _load_genespt_prediction(
                            method, fold_input, active_genespt, global_inputs, hasher
                        )
                    else:
                        prediction, provenance = _load_baseline_prediction(
                            method, fold_input, active_baselines, global_inputs, hasher
                        )
                    metrics = _evaluate_aligned(
                        evaluator, truth, prediction, fold_input.test_gene_names
                    )
                except MissingArtifact as error:
                    row.update(
                        {"status": "missing", "issue_code": error.code, "issue": str(error)}
                    )
                except IntegrityError as error:
                    row.update(
                        {"status": "invalid", "issue_code": error.code, "issue": str(error)}
                    )
                else:
                    row.update(provenance)
                    row.update(metrics)
                    row["status"] = "evaluated"
                finally:
                    if "prediction" in locals():
                        del prediction
                fold_rows.append(row)
            del truth

    aggregates = _aggregate_rows(fold_rows, datasets, methods)
    try:
        hasher.assert_unchanged()
    except IntegrityError as error:
        raise IntegrityError(
            error.code,
            f"Read-only audit cannot certify a concurrently changed artifact: {error}",
        ) from error

    expected_runs = len(datasets) * len(methods) * len(REQUIRED_FOLDS)
    evaluated_runs = sum(row["status"] == "evaluated" for row in fold_rows)
    missing_runs = sum(row["status"] == "missing" for row in fold_rows)
    invalid_runs = sum(row["status"] == "invalid" for row in fold_rows)
    invalid_inputs = sum(outcome.status == "invalid" for outcome in input_outcomes.values())
    missing_inputs = sum(outcome.status == "missing" for outcome in input_outcomes.values())
    aggregates_complete = all(row["status"] == "complete" for row in aggregates)
    if invalid_runs or invalid_inputs:
        status = "invalid"
    elif evaluated_runs == expected_runs and aggregates_complete:
        status = "complete"
    else:
        status = "missing"

    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "operation": "raw_prediction_integrity_and_centralized_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "complete": status == "complete",
        "read_only": True,
        "source_results_modified": False,
        "evaluation_policy": {
            "prediction_policy": "raw_scheduler_output_only",
            "readout_selection_performed": False,
            "posthoc_calibration_performed": False,
            "missing_prediction_fill_performed": False,
            "matrix_orientation": "spots_x_genes",
            "evaluated_gene_axis": "frozen_final_test_gene_idx_in_original_order",
            "fold_metric": "centralized_evaluator_median_across_truth_eligible_genes",
            "five_fold_metric": "arithmetic_mean_of_exactly_five_finite_fold_medians",
        },
        "paths": {
            "project_root": str(project),
            "config": str(config_file),
            "inputs_root": str(active_inputs),
            "genespt_root": str(active_genespt),
            "baseline_root": str(active_baselines),
            "centralized_evaluator": evaluator_path_value,
        },
        "sha256": {
            "config": global_inputs.config_sha256,
            "archive_checksum_manifest": global_inputs.archive_manifest_sha256,
            "materialization_manifest": global_inputs.materialization_sha256,
            "centralized_evaluator": evaluator_sha,
        },
        "selection": {
            "datasets": [str(dataset["dataset_id"]) for dataset in datasets],
            "methods": [method.name for method in methods],
            "folds": list(REQUIRED_FOLDS),
        },
        "naming_contract": {
            "inputs": "inputs/<dataset_id>/fold<fold>/{full_truth.npy,mode_a_split.json,normalization_audit.json,artifact_manifest.json}",
            "genespt": (
                "genespt/benchmark/<dataset_id>/fold<fold>/"
                "protocol_a_genespt_prediction_matrices/<model>/fold<fold>/prediction.npz"
            ),
            "baselines": "baselines/<Method>/<dataset_id>/fold<fold>/imputed_expression.npy",
            "baseline_status": "baselines/_scheduler/status/<Method>__<dataset_id>__fold<fold>.json",
        },
        "counts": {
            "expected_input_folds": len(datasets) * len(REQUIRED_FOLDS),
            "ready_input_folds": sum(outcome.status == "ready" for outcome in input_outcomes.values()),
            "missing_input_folds": missing_inputs,
            "invalid_input_folds": invalid_inputs,
            "expected_runs": expected_runs,
            "evaluated_runs": evaluated_runs,
            "missing_runs": missing_runs,
            "invalid_runs": invalid_runs,
            "expected_five_fold_summaries": len(datasets) * len(methods),
            "complete_five_fold_summaries": sum(row["status"] == "complete" for row in aggregates),
        },
        "input_folds": [_input_report(input_outcomes[key]) for key in sorted(input_outcomes)],
        "fold_metrics": fold_rows,
        "five_fold_summary": aggregates,
    }
    return _json_safe(report)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=GENESPT_ROOT)
    parser.add_argument("--inputs-root", type=Path, default=DEFAULT_INPUTS_ROOT)
    parser.add_argument("--genespt-root", type=Path, default=DEFAULT_GENESPT_OUTPUT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_OUTPUT_ROOT)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        action="extend",
        nargs="+",
        default=None,
        help="Dataset name or dataset_id. Every selected dataset still uses all five folds.",
    )
    parser.add_argument(
        "--method",
        "--methods",
        dest="methods",
        action="extend",
        nargs="+",
        default=None,
        help="Formal method name. Defaults to GeneSPT-GC, GeneSPT, and five baselines.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return nonzero when any selected input/run/five-fold summary is missing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-fold progress to stderr; the JSON report remains on stdout.",
    )
    return parser.parse_args(argv)


def exit_code_for_report(report: Mapping[str, Any], *, require_complete: bool) -> int:
    if report.get("status") == "invalid":
        return 2
    if require_complete and report.get("status") != "complete":
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    progress = (
        (lambda message: print(message, file=sys.stderr, flush=True))
        if args.verbose
        else None
    )
    try:
        report = evaluate_protocol_a(
            config_path=args.config,
            project_root=args.project_root,
            inputs_root=args.inputs_root,
            genespt_root=args.genespt_root,
            baseline_root=args.baseline_root,
            metrics_path=args.metrics_path,
            dataset_selectors=args.datasets,
            method_selectors=args.methods,
            progress=progress,
        )
    except AuditError as error:
        report = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "operation": "raw_prediction_integrity_and_centralized_evaluation",
            "status": "invalid",
            "complete": False,
            "read_only": True,
            "source_results_modified": False,
            "fatal_issue": {"code": error.code, "message": str(error)},
        }
    except Exception as error:
        report = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "operation": "raw_prediction_integrity_and_centralized_evaluation",
            "status": "invalid",
            "complete": False,
            "read_only": True,
            "source_results_modified": False,
            "fatal_issue": {
                "code": "unexpected_error",
                "message": f"{type(error).__name__}: {error}",
            },
        }
    print(
        json.dumps(_json_safe(report), ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return exit_code_for_report(report, require_complete=bool(args.require_complete))


if __name__ == "__main__":
    raise SystemExit(main())
