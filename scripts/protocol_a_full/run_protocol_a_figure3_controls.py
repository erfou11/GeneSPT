#!/usr/bin/env python3
"""Protocol A Figure 3A controls and Figure 3C centralized readout.

Figure 3A trains architecture-matched GC descriptor controls from the frozen
fold-specific ``full_truth.npy`` matrices.  It never opens or normalizes the
legacy full-panel count matrix.  The correct-descriptor arm is referenced from
the audited Protocol A benchmark; every trained control calls the exact formal
GC training function with the same per-fold initialization/sampling seed.

Figure 3C model execution remains owned by ``run_protocol_a_genespt.py``.  This
runner only validates those formal outputs and materializes centralized
fold/gene metrics plus a hash-complete source manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
PROTOCOL = "A"
DATASET = "Vis9A"
DATASET_ID = "Vis9A_D7_spaim_effective4470"
REQUIRED_FOLDS = (0, 1, 2, 3, 4)

FORMAL_BASE_SEED = 42
FORMAL_STEPS = 800
FORMAL_BATCH_SIZE = 65536
FORMAL_EVAL_EVERY = 100
FORMAL_LR = 0.002
FORMAL_WEIGHT_DECAY = 1e-4
FORMAL_DESCRIPTOR_SEED_BASE = 420_000

PROTOCOL_A_NORMALIZATION_POLICY = (
    "inner_train_gene_library_size_applied_to_all_columns"
)
ZERO_TRAIN_LIBRARY_POLICY = "set_entire_normalized_spot_row_to_zero"
BENCHMARK_MODEL = "gc_mlp_base"
BENCHMARK_PREFIX = "protocol_a_genespt"
CONTROL_PREFIX = "protocol_a_genespt_primary_controls"
COMPLETION_MANIFEST = "completion_manifest.json"
FAILURE_MANIFEST = "run_failure.json"

EXPECTED_DESCRIPTOR_SHA256 = (
    "f7833e0a485ac441f6815050171802cfb322207dd59749556d66b5868595b529"
)
EXPECTED_GENE_AXIS_SHA256 = (
    "f615ec76a9e0d1483c784ae5877d8a5785e2e032386dda6851ad19d98a4ff2a0"
)
EXPECTED_METRICS_SHA256 = (
    "f2570c61225dc5f211a38843859c556aa1f221a567935378c577f288aebfe5c1"
)

METRICS = ("SPCC", "RMSE", "JSD", "SSIM")
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class DescriptorControlSpec:
    control: str
    model: str
    transform_seed_offset: int


FIGURE3A_CONTROL_SPECS = (
    DescriptorControlSpec(
        "shuffled", "mlp_pca32_softplus_shuffled", 11
    ),
    DescriptorControlSpec("random", "mlp_pca32_softplus_random", 29),
    DescriptorControlSpec(
        "permuted_labels", "mlp_pca32_softplus_permuted_labels", 47
    ),
)
FIGURE3A_SETTINGS = ("correct",) + tuple(
    spec.control for spec in FIGURE3A_CONTROL_SPECS
)
FIGURE3A_MODEL_BY_SETTING = {
    "correct": "mlp_pca32_softplus_correct",
    **{spec.control: spec.model for spec in FIGURE3A_CONTROL_SPECS},
}

FIGURE3C_MODELS = (
    BENCHMARK_MODEL,
    "predictable_spatial_program_selected_correct",
    "predictable_spatial_program_shuffled_descriptor_control",
    "predictable_spatial_program_random_descriptor_control",
    "predictable_spatial_program_permuted_labels_control",
    "predictable_spatial_program_random_spatial_basis_control",
    "predictable_spatial_program_spot_permuted_spatial_program_control",
    "predictable_spatial_program_mean_coefficient_baseline_control",
)
FIGURE3C_CONTROL_BY_MODEL = {
    BENCHMARK_MODEL: "base",
    "predictable_spatial_program_selected_correct": "correct",
    "predictable_spatial_program_shuffled_descriptor_control": "shuffled_descriptor",
    "predictable_spatial_program_random_descriptor_control": "random_descriptor",
    "predictable_spatial_program_permuted_labels_control": "permuted_labels",
    "predictable_spatial_program_random_spatial_basis_control": "random_spatial_basis",
    "predictable_spatial_program_spot_permuted_spatial_program_control": "spot_permuted_spatial_program",
    "predictable_spatial_program_mean_coefficient_baseline_control": "mean_coefficient",
}


class Figure3Error(RuntimeError):
    """Raised when a frozen Figure 3 contract is invalid."""


class StaleResultError(Figure3Error):
    """Raised when an existing result cannot be resumed exactly."""


@dataclass(frozen=True)
class Figure3Layout:
    project_root: Path
    results_root: Path
    metrics_path: Path
    descriptor_path: Path
    expected_descriptor_sha256: str | None = EXPECTED_DESCRIPTOR_SHA256
    expected_gene_axis_sha256: str | None = EXPECTED_GENE_AXIS_SHA256
    expected_metrics_sha256: str | None = EXPECTED_METRICS_SHA256

    @classmethod
    def default(cls) -> "Figure3Layout":
        project = Path(__file__).resolve().parents[2]
        return cls(
            project_root=project,
            results_root=(
                project / "results" / "protocol_a_full_rerun_20260711"
            ),
            metrics_path=(
                project.parent
                / "GeneSPT_github_main_rebuild"
                / "src"
                / "genespt"
                / "metrics.py"
            ),
            descriptor_path=(
                project
                / "frozen_inputs"
                / "vis9a_psp_canonical_20260710"
                / "descriptors"
                / "descriptors_pca32_nmf32.npz"
            ),
        )

    @property
    def workspace_root(self) -> Path:
        return self.project_root.parent

    @property
    def inputs_root(self) -> Path:
        return self.results_root / "inputs" / DATASET_ID

    @property
    def benchmark_root(self) -> Path:
        return self.results_root / "genespt" / "benchmark" / DATASET_ID

    @property
    def primary_controls_root(self) -> Path:
        return (
            self.results_root
            / "genespt"
            / "primary_mechanism_controls"
            / DATASET_ID
        )

    @property
    def mechanism_root(self) -> Path:
        return self.results_root / "mechanism"

    @property
    def figure3a_root(self) -> Path:
        return self.mechanism_root / "figure3_a_descriptor_controls"

    @property
    def figure3c_root(self) -> Path:
        return self.mechanism_root / "figure3_c_primary_mechanism_controls"


@dataclass(frozen=True)
class FoldInput:
    fold: int
    truth_path: Path
    split_path: Path
    normalization_path: Path
    artifact_manifest_path: Path
    truth_record: Mapping[str, Any]
    split_record: Mapping[str, Any]
    normalization_record: Mapping[str, Any]
    artifact_manifest_record: Mapping[str, Any]
    n_spots: int
    n_genes: int
    gene_axis_sha256: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    test_genes: tuple[str, ...]
    truth_source: str = "full_truth.npy"


@dataclass(frozen=True)
class Figure3APlan:
    fold_input: FoldInput
    descriptor_record: Mapping[str, Any]
    descriptor_array_sha256: str
    benchmark_prediction_path: Path
    benchmark_prediction_record: Mapping[str, Any]
    benchmark_prediction_array_sha256: str
    benchmark_completion_record: Mapping[str, Any]
    benchmark_run_config_record: Mapping[str, Any]
    benchmark_job_signature_sha256: str
    correct_prediction: np.ndarray
    metrics_record: Mapping[str, Any]
    code_records: tuple[Mapping[str, Any], ...]
    output_dir: Path
    training_seed: int
    correct_source: str = "audited_protocol_a_benchmark_gc"

    @property
    def fold(self) -> int:
        return self.fold_input.fold

    @property
    def train_idx(self) -> np.ndarray:
        return self.fold_input.train_idx

    @property
    def val_idx(self) -> np.ndarray:
        return self.fold_input.val_idx

    @property
    def test_idx(self) -> np.ndarray:
        return self.fold_input.test_idx

    @property
    def test_genes(self) -> tuple[str, ...]:
        return self.fold_input.test_genes

    @property
    def truth_source(self) -> str:
        return self.fold_input.truth_source


@dataclass(frozen=True)
class Figure3CPlan:
    fold_input: FoldInput
    completion_record: Mapping[str, Any]
    job_signature_sha256: str
    prediction_paths: Mapping[str, Path]
    prediction_records: Mapping[str, Mapping[str, Any]]
    predictions: Mapping[str, np.ndarray]
    metrics_record: Mapping[str, Any]
    code_records: tuple[Mapping[str, Any], ...]
    output_dir: Path

    @property
    def fold(self) -> int:
        return self.fold_input.fold


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def _relative_path(path: Path, layout: Figure3Layout) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(layout.workspace_root.resolve(strict=True)).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: Path, layout: Figure3Layout) -> dict[str, Any]:
    if not path.is_file():
        raise Figure3Error(f"Required file is missing: {path}")
    return {
        "path": _relative_path(path, layout),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _record_path(record: Mapping[str, Any], layout: Figure3Layout) -> Path:
    raw = str(record["path"])
    path = Path(raw)
    if path.is_absolute():
        return path
    pure = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise StaleResultError(f"Unsafe manifest path: {raw}")
    return layout.workspace_root.joinpath(*pure.parts)


def _validate_file_record(
    record: Mapping[str, Any], path: Path, *, context: str
) -> None:
    if not isinstance(record, Mapping):
        raise Figure3Error(f"{context} record is missing")
    if not path.is_file():
        raise Figure3Error(f"{context} is missing: {path}")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise Figure3Error(f"{context} byte count mismatch")
    if str(record.get("sha256", "")) != sha256_file(path):
        raise Figure3Error(f"{context} SHA256 mismatch")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: object) -> None:
    _write_atomic(path, canonical_json_bytes(value))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _write_atomic(path, payload)


def write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise Figure3Error(f"Required JSON is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Figure3Error(f"Invalid JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise Figure3Error(f"JSON root must be an object: {path}")
    return value


def _strict_indices(
    payload: Mapping[str, Any], key: str, n_genes: int
) -> np.ndarray:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise Figure3Error(f"Mode-A split {key} must be a non-empty list")
    if any(type(value) is not int for value in raw):
        raise Figure3Error(f"Mode-A split {key} contains a non-integer")
    result = np.asarray(raw, dtype=np.int64)
    if np.unique(result).size != result.size:
        raise Figure3Error(f"Mode-A split {key} contains duplicates")
    if int(result.min()) < 0 or int(result.max()) >= n_genes:
        raise Figure3Error(f"Mode-A split {key} contains an out-of-range index")
    return result


def _strict_names(
    payload: Mapping[str, Any], key: str, expected: int
) -> tuple[str, ...]:
    raw = payload.get(key)
    if (
        not isinstance(raw, list)
        or len(raw) != expected
        or any(not isinstance(value, str) or not value for value in raw)
    ):
        raise Figure3Error(f"Mode-A split {key} is invalid")
    return tuple(raw)


def _validate_expected_hash(
    observed: str, expected: str | None, *, context: str
) -> None:
    if expected is not None and observed != expected:
        raise Figure3Error(
            f"{context} is not the frozen audited file: {observed} != {expected}"
        )


def _inspect_fold_input(layout: Figure3Layout, fold: int) -> FoldInput:
    directory = layout.inputs_root / f"fold{fold}"
    truth_path = directory / "full_truth.npy"
    split_path = directory / "mode_a_split.json"
    normalization_path = directory / "normalization_audit.json"
    artifact_path = directory / "artifact_manifest.json"
    artifact = _load_json(artifact_path)
    if artifact.get("dataset_id") != DATASET_ID or artifact.get("fold") != fold:
        raise Figure3Error(f"Input artifact identity mismatch for fold{fold}")
    outputs = artifact.get("output_artifacts")
    if not isinstance(outputs, Mapping):
        raise Figure3Error(f"Input artifact outputs are missing for fold{fold}")
    for key, path in (
        ("full_truth", truth_path),
        ("mode_a_split", split_path),
        ("normalization_audit", normalization_path),
    ):
        _validate_file_record(
            outputs.get(key, {}), path, context=f"fold{fold} {key}"
        )

    split = _load_json(split_path)
    if (
        split.get("schema_version") != 1
        or split.get("protocol") != PROTOCOL
        or split.get("dataset") != DATASET
        or split.get("dataset_id") != DATASET_ID
        or split.get("fold") != fold
    ):
        raise Figure3Error(f"Mode-A identity mismatch for fold{fold}")
    n_genes = int(split.get("gene_count", 0))
    if n_genes <= 0:
        raise Figure3Error(f"Mode-A gene count is invalid for fold{fold}")
    gene_axis_sha = str(split.get("gene_axis_sha256", ""))
    _validate_expected_hash(
        gene_axis_sha,
        layout.expected_gene_axis_sha256,
        context="Vis9A gene-axis SHA256",
    )
    train_idx = _strict_indices(split, "inner_train_gene_idx", n_genes)
    val_idx = _strict_indices(split, "inner_validation_gene_idx", n_genes)
    test_idx = _strict_indices(split, "final_test_gene_idx", n_genes)
    hidden_idx = _strict_indices(split, "hidden_gene_idx", n_genes)
    visible_alias = _strict_indices(split, "train_gene_idx", n_genes)
    val_alias = _strict_indices(split, "val_gene_idx", n_genes)
    hidden_alias = _strict_indices(split, "test_gene_idx", n_genes)
    expected_hidden = np.concatenate([val_idx, test_idx])
    if not np.array_equal(hidden_idx, expected_hidden):
        raise Figure3Error(
            f"fold{fold} hidden genes are not ordered validation plus final test"
        )
    if not np.array_equal(hidden_alias, expected_hidden):
        raise Figure3Error(f"fold{fold} Mode-A test alias is invalid")
    if not np.array_equal(visible_alias, train_idx) or not np.array_equal(
        val_alias, val_idx
    ):
        raise Figure3Error(f"fold{fold} Mode-A train/validation aliases changed")
    combined = np.concatenate([train_idx, val_idx, test_idx])
    if np.unique(combined).size != n_genes or set(combined.tolist()) != set(
        range(n_genes)
    ):
        raise Figure3Error(
            f"fold{fold} Mode-A train/validation/final-test coverage is invalid"
        )
    test_genes = _strict_names(split, "final_test_genes", test_idx.size)

    try:
        truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise Figure3Error(f"Could not load full_truth for fold{fold}") from error
    if truth.ndim != 2 or truth.shape[1] != n_genes or truth.dtype != np.float32:
        raise Figure3Error(
            f"fold{fold} full_truth shape/dtype mismatch: {truth.shape}/{truth.dtype}"
        )
    if not np.isfinite(truth).all():
        raise Figure3Error(f"fold{fold} full_truth contains non-finite values")
    n_spots = int(truth.shape[0])
    del truth

    normalization = _load_json(normalization_path)
    protocol_a = normalization.get("protocol_a")
    split_validation = normalization.get("split_validation")
    full_truth_meta = normalization.get("full_truth")
    if (
        normalization.get("protocol") != PROTOCOL
        or normalization.get("dataset_id") != DATASET_ID
        or normalization.get("fold") != fold
        or not isinstance(protocol_a, Mapping)
        or protocol_a.get("policy") != PROTOCOL_A_NORMALIZATION_POLICY
        or protocol_a.get("zero_train_library_policy")
        != ZERO_TRAIN_LIBRARY_POLICY
        or int(protocol_a.get("denominator_gene_count", -1)) != train_idx.size
        or list(protocol_a.get("shape", [])) != [n_spots, n_genes]
        or not isinstance(split_validation, Mapping)
        or split_validation.get("complete_coverage") is not True
        or split_validation.get("mutually_disjoint") is not True
        or not isinstance(full_truth_meta, Mapping)
        or full_truth_meta.get("gene_axis_sha256") != gene_axis_sha
        or list(full_truth_meta.get("shape", [])) != [n_spots, n_genes]
        or full_truth_meta.get("dtype") != "float32"
    ):
        raise Figure3Error(f"Protocol A normalization audit failed for fold{fold}")

    return FoldInput(
        fold=fold,
        truth_path=truth_path,
        split_path=split_path,
        normalization_path=normalization_path,
        artifact_manifest_path=artifact_path,
        truth_record=file_record(truth_path, layout),
        split_record=file_record(split_path, layout),
        normalization_record=file_record(normalization_path, layout),
        artifact_manifest_record=file_record(artifact_path, layout),
        n_spots=n_spots,
        n_genes=n_genes,
        gene_axis_sha256=gene_axis_sha,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        test_genes=test_genes,
    )


def _inspect_descriptor(
    layout: Figure3Layout, n_genes: int, gene_axis_sha256: str
) -> tuple[Mapping[str, Any], str]:
    record = file_record(layout.descriptor_path, layout)
    _validate_expected_hash(
        str(record["sha256"]),
        layout.expected_descriptor_sha256,
        context="PCA32 descriptor file SHA256",
    )
    _validate_expected_hash(
        gene_axis_sha256,
        layout.expected_gene_axis_sha256,
        context="PCA32 descriptor gene-axis binding",
    )
    try:
        with np.load(layout.descriptor_path, allow_pickle=False) as payload:
            if "pca32" not in payload.files:
                raise Figure3Error("Frozen descriptor file has no pca32 array")
            descriptor = np.asarray(payload["pca32"])
    except (OSError, ValueError) as error:
        raise Figure3Error("Could not load frozen PCA32 descriptor") from error
    if descriptor.shape != (n_genes, 32) or descriptor.dtype != np.float32:
        raise Figure3Error(
            f"Frozen PCA32 shape/dtype mismatch: {descriptor.shape}/{descriptor.dtype}"
        )
    if not np.isfinite(descriptor).all():
        raise Figure3Error("Frozen PCA32 contains non-finite values")
    return record, sha256_array(descriptor)


def _command_value(command: Sequence[Any], option: str) -> str:
    tokens = [str(value) for value in command]
    if tokens.count(option) != 1:
        raise Figure3Error(f"Benchmark command must contain exactly one {option}")
    index = tokens.index(option)
    if index + 1 >= len(tokens):
        raise Figure3Error(f"Benchmark command option has no value: {option}")
    return tokens[index + 1]


def _scalar(payload: Mapping[str, np.ndarray], key: str) -> Any:
    if key not in payload:
        raise Figure3Error(f"Prediction payload is missing scalar {key}")
    value = np.asarray(payload[key])
    if value.size != 1:
        raise Figure3Error(f"Prediction payload {key} is not scalar")
    return value.reshape(()).item()


def formal_training_seed(fold: int) -> int:
    return FORMAL_BASE_SEED + 1701 * int(fold)


def formal_transform_seed(fold: int, spec: DescriptorControlSpec) -> int:
    return (
        FORMAL_DESCRIPTOR_SEED_BASE
        + 1701 * int(fold)
        + int(spec.transform_seed_offset)
    )


def build_descriptor_control(
    descriptor: np.ndarray, control: str, seed: int
) -> tuple[np.ndarray, np.ndarray | None, str]:
    source = np.asarray(descriptor, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 32:
        raise Figure3Error(f"PCA32 descriptor has invalid shape: {source.shape}")
    rng = np.random.default_rng(int(seed))
    if control == "shuffled":
        permutation = rng.permutation(source.shape[0]).astype(np.int64)
        return (
            source[permutation].astype(np.float32, copy=True),
            permutation,
            "global gene-row shuffle",
        )
    if control == "random":
        transformed = rng.normal(
            loc=0.0,
            scale=float(np.std(source, dtype=np.float64) + 1e-6),
            size=source.shape,
        ).astype(np.float32)
        return transformed, None, "Gaussian random PCA32-shaped descriptor"
    if control == "permuted_labels":
        permutation = rng.permutation(source.shape[0]).astype(np.int64)
        transformed = source.copy()
        transformed[:] = source[permutation]
        return transformed, permutation, "permuted gene-label descriptor assignment"
    raise Figure3Error(f"Unknown Figure 3A descriptor control: {control}")


def _inspect_benchmark(
    layout: Figure3Layout, fold_input: FoldInput
) -> tuple[
    Path,
    Mapping[str, Any],
    str,
    Mapping[str, Any],
    Mapping[str, Any],
    str,
    np.ndarray,
]:
    fold = fold_input.fold
    root = layout.benchmark_root / f"fold{fold}"
    completion_path = root / COMPLETION_MANIFEST
    config_path = root / f"{BENCHMARK_PREFIX}_run_config.json"
    prediction_path = (
        root
        / f"{BENCHMARK_PREFIX}_prediction_matrices"
        / BENCHMARK_MODEL
        / f"fold{fold}"
        / "prediction.npz"
    )
    completion = _load_json(completion_path)
    if (
        completion.get("status") != "complete"
        or completion.get("protocol") != PROTOCOL
        or completion.get("mode") != "benchmark"
        or completion.get("dataset_id") != DATASET_ID
        or completion.get("fold") != fold
    ):
        raise Figure3Error(f"Audited benchmark completion identity failed for fold{fold}")
    command = completion.get("command")
    if not isinstance(command, list):
        raise Figure3Error(f"Benchmark command is missing for fold{fold}")
    expected_options = {
        "--folds": str(fold),
        "--st-normalization-scope": "train_genes",
        "--steps": str(FORMAL_STEPS),
        "--batch-size": str(FORMAL_BATCH_SIZE),
        "--eval-every": str(FORMAL_EVAL_EVERY),
        "--lr": str(FORMAL_LR),
        "--seed": str(FORMAL_BASE_SEED),
    }
    for option, expected in expected_options.items():
        if _command_value(command, option) != expected:
            raise Figure3Error(
                f"Benchmark command {option} changed for fold{fold}"
            )
    command_tokens = {str(value) for value in command}
    required_flags = {
        "--no-reuse-base",
        "--allow-train-base",
        "--save-prediction-matrices",
        "--no-run-controls",
        "--descriptor-cache",
    }
    if not required_flags.issubset(command_tokens):
        raise Figure3Error(f"Benchmark GC command contract failed for fold{fold}")

    run_config = _load_json(config_path)
    if (
        run_config.get("folds") != [fold]
        or run_config.get("seed") != FORMAL_BASE_SEED
        or run_config.get("st_normalization_scope") != "train_genes"
        or run_config.get("base_descriptor") != "pca32"
        or run_config.get("readout") != "identity"
        or run_config.get("posthoc_calibration") != "none"
        or run_config.get("prediction_matrices_saved") is not True
    ):
        raise Figure3Error(f"Benchmark GC run config failed for fold{fold}")

    try:
        with np.load(prediction_path, allow_pickle=True) as payload:
            prediction = np.asarray(payload["prediction"])
            embedded = np.asarray(payload["base_prediction_test"])
            train_idx = np.asarray(payload["train_gene_idx"], dtype=np.int64)
            val_idx = np.asarray(payload["val_gene_idx"], dtype=np.int64)
            test_idx = np.asarray(payload["test_gene_idx"], dtype=np.int64)
            identity = {
                "model": str(_scalar(payload, "model")),
                "fold": int(_scalar(payload, "fold")),
                "base_descriptor": str(_scalar(payload, "base_descriptor")),
                "readout": str(_scalar(payload, "readout")),
                "posthoc_calibration": str(
                    _scalar(payload, "posthoc_calibration")
                ),
            }
    except (OSError, ValueError, KeyError) as error:
        raise Figure3Error(
            f"Could not load audited benchmark GC prediction for fold{fold}"
        ) from error
    expected_shape = (fold_input.n_spots, fold_input.test_idx.size)
    if (
        prediction.shape != expected_shape
        or prediction.dtype != np.float32
        or not np.isfinite(prediction).all()
        or not np.array_equal(prediction, embedded)
        or not np.array_equal(train_idx, fold_input.train_idx)
        or not np.array_equal(val_idx, fold_input.val_idx)
        or not np.array_equal(test_idx, fold_input.test_idx)
        or identity
        != {
            "model": BENCHMARK_MODEL,
            "fold": fold,
            "base_descriptor": "pca32",
            "readout": "identity",
            "posthoc_calibration": "none",
        }
    ):
        raise Figure3Error(f"Audited benchmark GC prediction failed for fold{fold}")
    job_signature = str(completion.get("job_signature_sha256", ""))
    if len(job_signature) != 64:
        raise Figure3Error(f"Benchmark job signature is invalid for fold{fold}")
    return (
        prediction_path,
        file_record(prediction_path, layout),
        sha256_array(prediction),
        file_record(completion_path, layout),
        file_record(config_path, layout),
        job_signature,
        prediction.astype(np.float32, copy=True),
    )


def _load_metrics_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "protocol_a_figure3_centralized_metrics", path
    )
    if spec is None or spec.loader is None:
        raise Figure3Error(f"Cannot load centralized metrics: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise Figure3Error(f"Cannot import centralized metrics: {path}") from error
    finally:
        sys.dont_write_bytecode = previous
    if not callable(getattr(module, "evaluate_prediction", None)):
        raise Figure3Error("Centralized evaluator has no evaluate_prediction")
    return module


def _inspect_metrics(layout: Figure3Layout) -> Mapping[str, Any]:
    record = file_record(layout.metrics_path, layout)
    _validate_expected_hash(
        str(record["sha256"]),
        layout.expected_metrics_sha256,
        context="centralized metrics SHA256",
    )
    _load_metrics_module(layout.metrics_path)
    return record


def _code_records(layout: Figure3Layout) -> tuple[Mapping[str, Any], ...]:
    source_project = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        source_project / "main" / "run_gc_spatial_residual_basis_fold0.py",
        source_project
        / "main"
        / "run_gene_conditioned_mlp_controls_stabilization.py",
        source_project / "main" / "run_strict_gene_conditioned_decoder_gate.py",
        source_project
        / "scripts"
        / "protocol_a_full"
        / "run_protocol_a_genespt.py",
    )
    return tuple(file_record(path, layout) for path in paths)


def _load_scheduler_module(layout: Figure3Layout) -> Any:
    path = (
        layout.project_root
        / "scripts"
        / "protocol_a_full"
        / "run_protocol_a_genespt.py"
    )
    spec = importlib.util.spec_from_file_location(
        "protocol_a_figure3_scheduler_audit", path
    )
    if spec is None or spec.loader is None:
        raise Figure3Error(f"Cannot load Protocol A scheduler: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _audit_scheduler(
    layout: Figure3Layout, folds: Sequence[int], *, controls: bool
) -> Mapping[str, Any]:
    scheduler = _load_scheduler_module(layout)
    scheduler_layout = scheduler.Layout(
        project_root=layout.project_root,
        workspace_root=layout.workspace_root,
    )
    report, _contexts, _descriptors, _jobs = scheduler.preflight_protocol_a(
        layout=scheduler_layout,
        datasets=[DATASET],
        folds=folds,
        primary_mechanism_controls=controls,
    )
    expected_mode = (
        scheduler.CONTROL_MODE if controls else scheduler.BENCHMARK_MODE
    )
    if report.get("mode") != expected_mode:
        raise Figure3Error("Protocol A scheduler returned the wrong mode")
    if controls:
        return report
    states = {
        str(item["state"])
        for dataset in report["datasets"]
        for item in dataset["folds"]
    }
    if states != {"complete_valid"}:
        raise Figure3Error(
            f"Audited benchmark GC is not complete for every requested fold: {states}"
        )
    return report


def _select_folds(folds: Sequence[int] | None) -> tuple[int, ...]:
    if not folds:
        return REQUIRED_FOLDS
    invalid = sorted(set(int(value) for value in folds).difference(REQUIRED_FOLDS))
    if invalid:
        raise Figure3Error(f"Invalid fold selectors: {invalid}")
    return tuple(fold for fold in REQUIRED_FOLDS if fold in set(map(int, folds)))


def _figure3a_identity(plan: Figure3APlan) -> dict[str, Any]:
    fold_input = plan.fold_input
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "panel": "Figure3A",
        "dataset": DATASET,
        "dataset_id": DATASET_ID,
        "fold": plan.fold,
        "input_policy": {
            "matrix": "fold_specific_full_truth.npy",
            "split": "mode_a_inner_train_validation_final_test",
            "legacy_full_panel_cpm_used": False,
        },
        "inputs": {
            "full_truth": dict(fold_input.truth_record),
            "mode_a_split": dict(fold_input.split_record),
            "normalization_audit": dict(fold_input.normalization_record),
            "artifact_manifest": dict(fold_input.artifact_manifest_record),
            "shape": [fold_input.n_spots, fold_input.n_genes],
            "gene_axis_sha256": fold_input.gene_axis_sha256,
            "train_gene_idx_sha256": sha256_array(fold_input.train_idx),
            "validation_gene_idx_sha256": sha256_array(fold_input.val_idx),
            "final_test_gene_idx_sha256": sha256_array(fold_input.test_idx),
        },
        "descriptor": {
            "source": dict(plan.descriptor_record),
            "array": "pca32",
            "shape": [fold_input.n_genes, 32],
            "array_sha256": plan.descriptor_array_sha256,
            "gene_axis_sha256": fold_input.gene_axis_sha256,
        },
        "architecture": {
            "implementation": "train_canonical_base",
            "model": "FlexibleMLPDecoder",
            "output_mode": "softplus",
            "optimizer": "AdamW",
            "weight_decay": FORMAL_WEIGHT_DECAY,
            "gradient_clip_norm": 5.0,
            "checkpoint_selection": "validation_genes",
        },
        "hyperparameters": {
            "steps": FORMAL_STEPS,
            "batch_size": FORMAL_BATCH_SIZE,
            "eval_every": FORMAL_EVAL_EVERY,
            "lr": FORMAL_LR,
            "base_seed": FORMAL_BASE_SEED,
            "training_seed": plan.training_seed,
        },
        "correct": {
            "source": plan.correct_source,
            "prediction": dict(plan.benchmark_prediction_record),
            "prediction_array_sha256": plan.benchmark_prediction_array_sha256,
            "completion_manifest": dict(plan.benchmark_completion_record),
            "run_config": dict(plan.benchmark_run_config_record),
            "job_signature_sha256": plan.benchmark_job_signature_sha256,
            "training_seed": plan.training_seed,
        },
        "controls": [
            {
                "control": spec.control,
                "model": spec.model,
                "only_changed_input": "pca32_descriptor",
                "transform_seed": formal_transform_seed(plan.fold, spec),
                "training_seed": plan.training_seed,
            }
            for spec in FIGURE3A_CONTROL_SPECS
        ],
        "centralized_metrics": dict(plan.metrics_record),
        "code": [dict(record) for record in plan.code_records],
    }


def _validate_completion(
    output_dir: Path,
    expected_identity: Mapping[str, Any],
    layout: Figure3Layout,
) -> Mapping[str, Any]:
    manifest_path = output_dir / COMPLETION_MANIFEST
    if not manifest_path.is_file():
        raise StaleResultError(
            f"Existing output has no completion manifest: {output_dir}"
        )
    observed = _load_json(manifest_path)
    if observed.get("status") != "complete":
        raise StaleResultError(f"Completion status is not complete: {output_dir}")
    if observed.get("identity") != expected_identity:
        raise StaleResultError(f"Resume rejected: stale identity in {output_dir}")
    outputs = observed.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise StaleResultError(f"Resume rejected: output records missing in {output_dir}")
    resolved_root = output_dir.resolve(strict=True)
    for record in outputs:
        if not isinstance(record, Mapping):
            raise StaleResultError("Resume rejected: malformed output record")
        path = _record_path(record, layout)
        try:
            path.resolve(strict=False).relative_to(resolved_root)
        except ValueError as error:
            raise StaleResultError(
                f"Resume rejected: output escapes fold directory: {path}"
            ) from error
        try:
            _validate_file_record(record, path, context="resumed output")
        except Figure3Error as error:
            raise StaleResultError(str(error)) from error
    return observed


def preflight_figure3a(
    *,
    layout: Figure3Layout | None = None,
    folds: Sequence[int] | None = None,
    audit_scheduler: bool = True,
    progress: Callable[[str], None] | None = None,
) -> tuple[Mapping[str, Any], Mapping[int, Figure3APlan]]:
    active = layout or Figure3Layout.default()
    selected = _select_folds(folds)
    if audit_scheduler:
        if progress:
            progress("[Figure3A preflight] validating audited benchmark schedule")
        _audit_scheduler(active, selected, controls=False)
    metrics_record = _inspect_metrics(active)
    code_records = _code_records(active)
    plans: dict[int, Figure3APlan] = {}
    states: dict[str, str] = {}
    descriptor_identity: tuple[Mapping[str, Any], str] | None = None
    for fold in selected:
        if progress:
            progress(f"[Figure3A preflight] fold{fold}: Mode-A inputs and benchmark GC")
        fold_input = _inspect_fold_input(active, fold)
        if descriptor_identity is None:
            descriptor_identity = _inspect_descriptor(
                active, fold_input.n_genes, fold_input.gene_axis_sha256
            )
        descriptor_record, descriptor_array_sha = descriptor_identity
        (
            benchmark_path,
            benchmark_record,
            benchmark_array_sha,
            benchmark_completion_record,
            benchmark_config_record,
            benchmark_job_signature,
            correct_prediction,
        ) = _inspect_benchmark(active, fold_input)
        plan = Figure3APlan(
            fold_input=fold_input,
            descriptor_record=descriptor_record,
            descriptor_array_sha256=descriptor_array_sha,
            benchmark_prediction_path=benchmark_path,
            benchmark_prediction_record=benchmark_record,
            benchmark_prediction_array_sha256=benchmark_array_sha,
            benchmark_completion_record=benchmark_completion_record,
            benchmark_run_config_record=benchmark_config_record,
            benchmark_job_signature_sha256=benchmark_job_signature,
            correct_prediction=correct_prediction,
            metrics_record=metrics_record,
            code_records=code_records,
            output_dir=active.figure3a_root / f"fold{fold}",
            training_seed=formal_training_seed(fold),
        )
        plans[fold] = plan
        if not plan.output_dir.exists():
            states[f"fold{fold}"] = "planned"
        elif not plan.output_dir.is_dir():
            raise StaleResultError(f"Figure3A output is not a directory: {plan.output_dir}")
        else:
            _validate_completion(
                plan.output_dir, _figure3a_identity(plan), active
            )
            states[f"fold{fold}"] = "complete_valid"
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "operation": "preflight_only",
            "model_started": False,
            "panel": "Figure3A",
            "dataset": DATASET,
            "dataset_id": DATASET_ID,
            "folds": list(selected),
            "states": states,
            "output_root": str(active.figure3a_root),
            "legacy_full_panel_cpm_used": False,
        },
        plans,
    )


def _evaluate_prediction(
    metrics_module: Any,
    *,
    fold_input: FoldInput,
    prediction: np.ndarray,
    model: str,
    control: str,
    source_kind: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    truth = np.load(fold_input.truth_path, mmap_mode="r", allow_pickle=False)
    test_truth = np.asarray(truth[:, fold_input.test_idx], dtype=np.float32)
    del truth
    try:
        per_gene, summary_frame = metrics_module.evaluate_prediction(
            test_truth,
            prediction,
            gene_names=list(fold_input.test_genes),
        )
    except Exception as error:
        raise Figure3Error(
            f"Centralized evaluation failed for fold{fold_input.fold} {control}"
        ) from error
    if len(per_gene) != fold_input.test_idx.size or len(summary_frame) != 1:
        raise Figure3Error("Centralized evaluator returned unexpected dimensions")
    summary = summary_frame.iloc[0].to_dict()
    if (
        int(summary.get("total_genes", -1)) != fold_input.test_idx.size
        or int(summary.get("eligible_genes", -1)) != fold_input.test_idx.size
        or int(summary.get("scored_genes", -1)) != fold_input.test_idx.size
        or float(summary.get("coverage", 0.0)) != 1.0
    ):
        raise Figure3Error(
            f"Centralized evaluation coverage is incomplete for fold{fold_input.fold} {control}"
        )
    gene_frame = per_gene.rename(columns={"gene_idx": "gene_pos"}).copy()
    gene_frame.insert(0, "gene_idx", fold_input.test_idx)
    gene_frame.insert(0, "source_kind", source_kind)
    gene_frame.insert(0, "control", control)
    gene_frame.insert(0, "model", model)
    gene_frame.insert(0, "fold", fold_input.fold)
    gene_frame.insert(0, "dataset_id", DATASET_ID)
    gene_frame.insert(0, "dataset", DATASET)
    fold_row = {
        "dataset": DATASET,
        "dataset_id": DATASET_ID,
        "fold": fold_input.fold,
        "model": model,
        "control": control,
        "source_kind": source_kind,
        "SPCC": float(summary["SPCC"]),
        "RMSE": float(summary["RMSE"]),
        "JSD": float(summary["JSD"]),
        "JS": float(summary["JSD"]),
        "SSIM": float(summary["SSIM"]),
        "coverage": float(summary["coverage"]),
        "eligible_genes": int(summary["eligible_genes"]),
        "scored_genes": int(summary["scored_genes"]),
        "constant_prediction_genes": int(summary["constant_prediction_genes"]),
    }
    coverage = []
    for metric in METRICS:
        eligible = int(summary[f"{metric}_eligible"])
        scored = int(summary[f"{metric}_scored"])
        if eligible != scored:
            raise Figure3Error(
                f"Centralized {metric} coverage incomplete for fold{fold_input.fold} {control}"
            )
        coverage.append(
            {
                "dataset": DATASET,
                "dataset_id": DATASET_ID,
                "fold": fold_input.fold,
                "model": model,
                "control": control,
                "metric": metric,
                "eligible": eligible,
                "scored": scored,
                "coverage": float(summary[f"{metric}_coverage"]),
                "constant_prediction": int(
                    summary[f"{metric}_constant_prediction"]
                ),
            }
        )
    return gene_frame, fold_row, coverage


def _output_records(directory: Path, layout: Figure3Layout) -> list[Mapping[str, Any]]:
    records = []
    for path in sorted(directory.rglob("*"), key=lambda value: str(value)):
        if path.is_file() and path.name not in {COMPLETION_MANIFEST, FAILURE_MANIFEST}:
            records.append(file_record(path, layout))
    if not records:
        raise Figure3Error(f"No outputs were written in {directory}")
    return records


def _execute_figure3a_fold(
    plan: Figure3APlan,
    layout: Figure3Layout,
    *,
    trainer: Callable[..., Any] | None,
    progress: Callable[[str], None] | None,
) -> Mapping[str, Any]:
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        truth = np.load(plan.fold_input.truth_path, allow_pickle=False)
        with np.load(layout.descriptor_path, allow_pickle=False) as payload:
            descriptor = np.asarray(payload["pca32"], dtype=np.float32)
        if sha256_array(descriptor) != plan.descriptor_array_sha256:
            raise Figure3Error("Frozen PCA32 changed after preflight")
        metrics_module = _load_metrics_module(layout.metrics_path)
        test_index_path = plan.output_dir / "test_gene_idx.npy"
        write_npy(test_index_path, plan.test_idx.astype(np.int64))

        gene_frames: list[pd.DataFrame] = []
        fold_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        correct_gene, correct_row, correct_coverage = _evaluate_prediction(
            metrics_module,
            fold_input=plan.fold_input,
            prediction=plan.correct_prediction,
            model=FIGURE3A_MODEL_BY_SETTING["correct"],
            control="correct",
            source_kind=plan.correct_source,
        )
        gene_frames.append(correct_gene)
        fold_rows.append(correct_row)
        coverage_rows.extend(correct_coverage)
        correct_reference_path = plan.output_dir / "correct_benchmark_reference.json"
        write_json(
            correct_reference_path,
            {
                "source": plan.correct_source,
                "prediction": dict(plan.benchmark_prediction_record),
                "prediction_array_sha256": plan.benchmark_prediction_array_sha256,
                "completion_manifest": dict(plan.benchmark_completion_record),
                "run_config": dict(plan.benchmark_run_config_record),
                "test_gene_idx_sha256": sha256_array(plan.test_idx),
            },
        )

        if trainer is None:
            try:
                import torch

                main_dir = Path(__file__).resolve().parents[2] / "main"
                if str(main_dir) not in sys.path:
                    sys.path.insert(0, str(main_dir))
                from run_gc_spatial_residual_basis_fold0 import (
                    train_canonical_base,
                )
            except Exception as error:
                raise Figure3Error("Cannot load formal GC trainer") from error
            if not torch.cuda.is_available():
                raise Figure3Error(
                    "Formal Figure3A controls require CUDA to match the audited GC run"
                )
            active_trainer = train_canonical_base
            device: Any = torch.device("cuda")
        else:
            active_trainer = trainer
            device = "synthetic-test-device"

        train_values = truth[:, plan.train_idx].reshape(-1)
        output_low = float(np.quantile(train_values, 0.001))
        output_high = float(np.quantile(train_values, 0.999))
        result_details = []
        for spec in FIGURE3A_CONTROL_SPECS:
            if progress:
                progress(
                    f"[Figure3A run] fold{plan.fold} {spec.control}: formal GC training"
                )
            transform_seed = formal_transform_seed(plan.fold, spec)
            transformed, permutation, note = build_descriptor_control(
                descriptor, spec.control, transform_seed
            )
            if np.array_equal(transformed, descriptor):
                raise Figure3Error(
                    f"Descriptor control did not change PCA32: {spec.control}"
                )
            setting_dir = plan.output_dir / spec.control
            setting_dir.mkdir(parents=True, exist_ok=False)
            descriptor_path = setting_dir / "descriptor_control.npz"
            write_npz(
                descriptor_path,
                descriptor=transformed,
                permutation=(
                    permutation
                    if permutation is not None
                    else np.empty(0, dtype=np.int64)
                ),
                control=np.asarray(spec.control),
                source_descriptor_sha256=np.asarray(plan.descriptor_array_sha256),
                transform_seed=np.asarray(transform_seed, dtype=np.int64),
                training_seed=np.asarray(plan.training_seed, dtype=np.int64),
                note=np.asarray(note),
            )
            predictions, history, best_val_score = active_trainer(
                X=truth,
                desc_np=transformed,
                train_idx=plan.train_idx,
                val_idx=plan.val_idx,
                test_idx=plan.test_idx,
                output_low=output_low,
                output_high=output_high,
                device=device,
                steps=FORMAL_STEPS,
                batch_size=FORMAL_BATCH_SIZE,
                eval_every=FORMAL_EVAL_EVERY,
                lr=FORMAL_LR,
                seed=plan.training_seed,
            )
            if not isinstance(predictions, Mapping):
                raise Figure3Error(f"Trainer returned invalid predictions: {spec.control}")
            expected_shapes = {
                "train": (plan.fold_input.n_spots, plan.train_idx.size),
                "val": (plan.fold_input.n_spots, plan.val_idx.size),
                "test": (plan.fold_input.n_spots, plan.test_idx.size),
            }
            normalized_predictions: dict[str, np.ndarray] = {}
            for split, expected_shape in expected_shapes.items():
                if split not in predictions:
                    raise Figure3Error(
                        f"Trainer prediction is missing {split}: {spec.control}"
                    )
                value = np.asarray(predictions[split])
                if (
                    value.shape != expected_shape
                    or value.dtype != np.float32
                    or not np.isfinite(value).all()
                ):
                    raise Figure3Error(
                        f"Trainer {split} prediction is invalid: {spec.control}"
                    )
                normalized_predictions[split] = value
            prediction = normalized_predictions["test"]
            prediction_path = setting_dir / "prediction.npz"
            write_npz(
                prediction_path,
                prediction=prediction,
                train_gene_idx=plan.train_idx.astype(np.int64),
                val_gene_idx=plan.val_idx.astype(np.int64),
                test_gene_idx=plan.test_idx.astype(np.int64),
                test_genes=np.asarray(plan.test_genes),
                dataset_id=np.asarray(DATASET_ID),
                fold=np.asarray(plan.fold, dtype=np.int64),
                model=np.asarray(spec.model),
                control=np.asarray(spec.control),
                descriptor=np.asarray("pca32"),
                descriptor_source_sha256=np.asarray(
                    plan.descriptor_array_sha256
                ),
                descriptor_control_sha256=np.asarray(
                    sha256_array(transformed)
                ),
                transform_seed=np.asarray(transform_seed, dtype=np.int64),
                training_seed=np.asarray(plan.training_seed, dtype=np.int64),
                steps=np.asarray(FORMAL_STEPS, dtype=np.int64),
                batch_size=np.asarray(FORMAL_BATCH_SIZE, dtype=np.int64),
                eval_every=np.asarray(FORMAL_EVAL_EVERY, dtype=np.int64),
                lr=np.asarray(FORMAL_LR, dtype=np.float64),
                output_mode=np.asarray("softplus"),
                readout=np.asarray("identity"),
                posthoc_calibration=np.asarray("none"),
            )
            history_frame = pd.DataFrame(history).copy()
            history_frame.insert(0, "training_seed", plan.training_seed)
            history_frame.insert(0, "transform_seed", transform_seed)
            history_frame.insert(0, "control", spec.control)
            history_frame.insert(0, "model", spec.model)
            history_frame.insert(0, "fold", plan.fold)
            history_path = setting_dir / "training_history.csv"
            write_csv(history_path, history_frame)
            genes, row, coverage = _evaluate_prediction(
                metrics_module,
                fold_input=plan.fold_input,
                prediction=prediction,
                model=spec.model,
                control=spec.control,
                source_kind="formal_descriptor_only_control",
            )
            gene_frames.append(genes)
            fold_rows.append(row)
            coverage_rows.extend(coverage)
            result_details.append(
                {
                    "control": spec.control,
                    "model": spec.model,
                    "descriptor_control": file_record(descriptor_path, layout),
                    "descriptor_control_array_sha256": sha256_array(transformed),
                    "prediction": file_record(prediction_path, layout),
                    "prediction_array_sha256": sha256_array(prediction),
                    "training_history": file_record(history_path, layout),
                    "transform_seed": transform_seed,
                    "training_seed": plan.training_seed,
                    "best_validation_score": float(best_val_score),
                }
            )

        fold_metrics_path = plan.output_dir / "fold_metrics.csv"
        gene_metrics_path = plan.output_dir / "gene_level_metrics.csv"
        coverage_path = plan.output_dir / "coverage.csv"
        write_csv(fold_metrics_path, pd.DataFrame(fold_rows))
        write_csv(gene_metrics_path, pd.concat(gene_frames, ignore_index=True))
        write_csv(coverage_path, pd.DataFrame(coverage_rows))
        identity = _figure3a_identity(plan)
        outputs = _output_records(plan.output_dir, layout)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "identity": identity,
            "results": result_details,
            "outputs": outputs,
        }
        write_json(plan.output_dir / COMPLETION_MANIFEST, manifest)
        return manifest
    except Exception as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "panel": "Figure3A",
            "dataset_id": DATASET_ID,
            "fold": plan.fold,
            "failed_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(plan.output_dir / FAILURE_MANIFEST, failure)
        raise


def _aggregate_panel(
    *,
    panel: str,
    root: Path,
    folds: Sequence[int],
    layout: Figure3Layout,
    correct_control: str,
) -> Mapping[str, Any]:
    gene_frames = []
    fold_frames = []
    coverage_frames = []
    fold_manifest_records = []
    for fold in folds:
        directory = root / f"fold{fold}"
        gene_frames.append(pd.read_csv(directory / "gene_level_metrics.csv"))
        fold_frames.append(pd.read_csv(directory / "fold_metrics.csv"))
        coverage_frames.append(pd.read_csv(directory / "coverage.csv"))
        fold_manifest_records.append(
            file_record(directory / COMPLETION_MANIFEST, layout)
        )
    genes = pd.concat(gene_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    genes_path = root / "gene_level_metrics.csv"
    folds_path = root / "fold_metrics.csv"
    coverage_path = root / "coverage.csv"
    write_csv(genes_path, genes)
    write_csv(folds_path, fold_metrics)
    write_csv(coverage_path, coverage)

    summary_rows = []
    for (model, control), frame in fold_metrics.groupby(
        ["model", "control"], sort=False
    ):
        row: dict[str, Any] = {
            "dataset": DATASET,
            "dataset_id": DATASET_ID,
            "panel": panel,
            "model": model,
            "control": control,
            "folds": int(frame["fold"].nunique()),
            "aggregation": "arithmetic_mean_of_fold_medians",
            "coverage": float(frame["coverage"].mean()),
        }
        for metric in METRICS:
            row[metric] = float(frame[metric].mean())
            row[f"{metric}_std_ddof0"] = float(
                frame[metric].to_numpy(dtype=float).std(ddof=0)
            )
        row["JS"] = row["JSD"]
        row["JS_std_ddof0"] = row["JSD_std_ddof0"]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = root / "five_fold_metrics.csv"
    write_csv(summary_path, summary)

    correct = summary[summary["control"].eq(correct_control)]
    if len(correct) != 1:
        raise Figure3Error(
            f"{panel} aggregate requires one correct setting, found {len(correct)}"
        )
    correct_row = correct.iloc[0]
    delta_rows = []
    for _, row in summary.iterrows():
        if row["control"] == correct_control:
            continue
        delta_rows.append(
            {
                "panel": panel,
                "comparison": f"correct_vs_{row['control']}",
                "control": row["control"],
                "delta_SPCC_correct_minus_control": float(correct_row["SPCC"])
                - float(row["SPCC"]),
                "delta_RMSE_correct_minus_control": float(correct_row["RMSE"])
                - float(row["RMSE"]),
                "delta_JSD_correct_minus_control": float(correct_row["JSD"])
                - float(row["JSD"]),
                "delta_SSIM_correct_minus_control": float(correct_row["SSIM"])
                - float(row["SSIM"]),
            }
        )
    deltas_path = root / "correct_vs_controls_deltas.csv"
    write_csv(deltas_path, pd.DataFrame(delta_rows))
    output_paths = (
        genes_path,
        folds_path,
        coverage_path,
        summary_path,
        deltas_path,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if tuple(folds) == REQUIRED_FOLDS else "partial_complete",
        "panel": panel,
        "dataset": DATASET,
        "dataset_id": DATASET_ID,
        "folds": list(folds),
        "generated_at_utc": _utc_now(),
        "aggregation": "arithmetic_mean_of_fold_medians",
        "fold_manifests": fold_manifest_records,
        "outputs": [file_record(path, layout) for path in output_paths],
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def run_figure3a(
    *,
    layout: Figure3Layout | None = None,
    folds: Sequence[int] | None = None,
    resume: bool = False,
    audit_scheduler: bool = True,
    trainer: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Mapping[str, Any]:
    active = layout or Figure3Layout.default()
    preflight, plans = preflight_figure3a(
        layout=active,
        folds=folds,
        audit_scheduler=audit_scheduler,
        progress=progress,
    )
    selected = tuple(int(value) for value in preflight["folds"])
    completed = []
    resumed = []
    for fold in selected:
        plan = plans[fold]
        state = str(preflight["states"][f"fold{fold}"])
        if state == "complete_valid":
            if not resume:
                raise FileExistsError(
                    f"Validated Figure3A fold{fold} exists; pass --resume to skip"
                )
            resumed.append(fold)
            if progress:
                progress(f"[Figure3A resume] fold{fold}: exact completion skipped")
            continue
        _execute_figure3a_fold(
            plan,
            active,
            trainer=trainer,
            progress=progress,
        )
        completed.append(fold)
    aggregate = _aggregate_panel(
        panel="Figure3A",
        root=active.figure3a_root,
        folds=selected,
        layout=active,
        correct_control="correct",
    )
    return {
        **preflight,
        "operation": "run",
        "model_started": bool(completed),
        "completed_folds": completed,
        "resumed_folds": resumed,
        "aggregate_status": aggregate["status"],
    }


def _figure3c_identity(plan: Figure3CPlan) -> dict[str, Any]:
    fold_input = plan.fold_input
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "panel": "Figure3C",
        "dataset": DATASET,
        "dataset_id": DATASET_ID,
        "fold": plan.fold,
        "formal_source_mode": "primary_mechanism_controls",
        "formal_source_runner": "run_protocol_a_genespt.py --primary-mechanism-controls --dataset Vis9A",
        "inputs": {
            "full_truth": dict(fold_input.truth_record),
            "mode_a_split": dict(fold_input.split_record),
            "normalization_audit": dict(fold_input.normalization_record),
            "artifact_manifest": dict(fold_input.artifact_manifest_record),
            "final_test_gene_idx_sha256": sha256_array(fold_input.test_idx),
        },
        "formal_completion_manifest": dict(plan.completion_record),
        "job_signature_sha256": plan.job_signature_sha256,
        "prediction_sources": [
            {
                "model": model,
                "control": FIGURE3C_CONTROL_BY_MODEL[model],
                **dict(plan.prediction_records[model]),
                "prediction_array_sha256": sha256_array(plan.predictions[model]),
            }
            for model in FIGURE3C_MODELS
        ],
        "centralized_metrics": dict(plan.metrics_record),
        "code": [dict(record) for record in plan.code_records],
    }


def _inspect_figure3c_fold(
    layout: Figure3Layout,
    fold_input: FoldInput,
    metrics_record: Mapping[str, Any],
    code_records: tuple[Mapping[str, Any], ...],
) -> Figure3CPlan:
    fold = fold_input.fold
    root = layout.primary_controls_root / f"fold{fold}"
    completion_path = root / COMPLETION_MANIFEST
    completion = _load_json(completion_path)
    if (
        completion.get("status") != "complete"
        or completion.get("protocol") != PROTOCOL
        or completion.get("mode") != "primary_mechanism_controls"
        or completion.get("dataset_id") != DATASET_ID
        or completion.get("fold") != fold
    ):
        raise Figure3Error(f"Formal Figure3C completion identity failed for fold{fold}")
    job_signature = str(completion.get("job_signature_sha256", ""))
    if len(job_signature) != 64:
        raise Figure3Error(f"Formal Figure3C job signature invalid for fold{fold}")
    prediction_root = root / f"{CONTROL_PREFIX}_prediction_matrices"
    observed_models = (
        {path.name for path in prediction_root.iterdir() if path.is_dir()}
        if prediction_root.is_dir()
        else set()
    )
    if observed_models != set(FIGURE3C_MODELS):
        raise Figure3Error(
            f"Formal Figure3C model set mismatch for fold{fold}: {sorted(observed_models)}"
        )
    predictions: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    records: dict[str, Mapping[str, Any]] = {}
    expected_shape = (fold_input.n_spots, fold_input.test_idx.size)
    for model in FIGURE3C_MODELS:
        path = prediction_root / model / f"fold{fold}" / "prediction.npz"
        try:
            with np.load(path, allow_pickle=True) as payload:
                prediction = np.asarray(payload["prediction"])
                train_idx = np.asarray(payload["train_gene_idx"], dtype=np.int64)
                val_idx = np.asarray(payload["val_gene_idx"], dtype=np.int64)
                test_idx = np.asarray(payload["test_gene_idx"], dtype=np.int64)
                identity = {
                    "model": str(_scalar(payload, "model")),
                    "fold": int(_scalar(payload, "fold")),
                    "readout": str(_scalar(payload, "readout")),
                    "posthoc_calibration": str(
                        _scalar(payload, "posthoc_calibration")
                    ),
                }
        except (OSError, ValueError, KeyError) as error:
            raise Figure3Error(
                f"Could not load formal Figure3C prediction fold{fold} {model}"
            ) from error
        if (
            prediction.shape != expected_shape
            or prediction.dtype != np.float32
            or not np.isfinite(prediction).all()
            or not np.array_equal(train_idx, fold_input.train_idx)
            or not np.array_equal(val_idx, fold_input.val_idx)
            or not np.array_equal(test_idx, fold_input.test_idx)
            or identity
            != {
                "model": model,
                "fold": fold,
                "readout": "identity",
                "posthoc_calibration": "none",
            }
        ):
            raise Figure3Error(
                f"Formal Figure3C prediction contract failed fold{fold} {model}"
            )
        paths[model] = path
        records[model] = file_record(path, layout)
        predictions[model] = prediction.astype(np.float32, copy=True)
    return Figure3CPlan(
        fold_input=fold_input,
        completion_record=file_record(completion_path, layout),
        job_signature_sha256=job_signature,
        prediction_paths=paths,
        prediction_records=records,
        predictions=predictions,
        metrics_record=metrics_record,
        code_records=code_records,
        output_dir=layout.figure3c_root / f"fold{fold}",
    )


def preflight_figure3c(
    *,
    layout: Figure3Layout | None = None,
    folds: Sequence[int] | None = None,
    audit_scheduler: bool = True,
    progress: Callable[[str], None] | None = None,
) -> tuple[Mapping[str, Any], Mapping[int, Figure3CPlan]]:
    active = layout or Figure3Layout.default()
    selected = _select_folds(folds)
    if audit_scheduler:
        if progress:
            progress("[Figure3C preflight] validating formal primary-control schedule")
        scheduler_report = _audit_scheduler(active, selected, controls=True)
        states = {
            str(item["state"])
            for dataset in scheduler_report["datasets"]
            for item in dataset["folds"]
        }
        if states != {"complete_valid"}:
            raise Figure3Error(
                f"Formal Figure3C outputs are not complete for requested folds: {states}"
            )
    metrics_record = _inspect_metrics(active)
    code_records = _code_records(active)
    plans = {}
    states = {}
    for fold in selected:
        if progress:
            progress(f"[Figure3C preflight] fold{fold}: matrices and manifests")
        fold_input = _inspect_fold_input(active, fold)
        plan = _inspect_figure3c_fold(
            active, fold_input, metrics_record, code_records
        )
        plans[fold] = plan
        if not plan.output_dir.exists():
            states[f"fold{fold}"] = "planned"
        elif not plan.output_dir.is_dir():
            raise StaleResultError(f"Figure3C output is not a directory: {plan.output_dir}")
        else:
            _validate_completion(
                plan.output_dir, _figure3c_identity(plan), active
            )
            states[f"fold{fold}"] = "complete_valid"
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "operation": "preflight_only",
            "model_started": False,
            "panel": "Figure3C",
            "dataset": DATASET,
            "dataset_id": DATASET_ID,
            "folds": list(selected),
            "states": states,
            "output_root": str(active.figure3c_root),
            "formal_prediction_root": str(active.primary_controls_root),
        },
        plans,
    )


def _execute_figure3c_fold(
    plan: Figure3CPlan,
    layout: Figure3Layout,
    *,
    progress: Callable[[str], None] | None,
) -> Mapping[str, Any]:
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        metrics_module = _load_metrics_module(layout.metrics_path)
        test_index_path = plan.output_dir / "test_gene_idx.npy"
        write_npy(test_index_path, plan.fold_input.test_idx.astype(np.int64))
        genes = []
        fold_rows = []
        coverage_rows = []
        for model in FIGURE3C_MODELS:
            control = FIGURE3C_CONTROL_BY_MODEL[model]
            if progress:
                progress(
                    f"[Figure3C metrics] fold{plan.fold} {control}: centralized evaluator"
                )
            gene_frame, row, coverage = _evaluate_prediction(
                metrics_module,
                fold_input=plan.fold_input,
                prediction=plan.predictions[model],
                model=model,
                control=control,
                source_kind="formal_primary_mechanism_controls",
            )
            genes.append(gene_frame)
            fold_rows.append(row)
            coverage_rows.extend(coverage)
        write_csv(plan.output_dir / "fold_metrics.csv", pd.DataFrame(fold_rows))
        write_csv(
            plan.output_dir / "gene_level_metrics.csv",
            pd.concat(genes, ignore_index=True),
        )
        write_csv(plan.output_dir / "coverage.csv", pd.DataFrame(coverage_rows))
        write_json(
            plan.output_dir / "prediction_source_manifest.json",
            {
                "formal_completion_manifest": dict(plan.completion_record),
                "job_signature_sha256": plan.job_signature_sha256,
                "predictions": [
                    {
                        "model": model,
                        "control": FIGURE3C_CONTROL_BY_MODEL[model],
                        **dict(plan.prediction_records[model]),
                        "prediction_array_sha256": sha256_array(
                            plan.predictions[model]
                        ),
                    }
                    for model in FIGURE3C_MODELS
                ],
            },
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "identity": _figure3c_identity(plan),
            "outputs": _output_records(plan.output_dir, layout),
        }
        write_json(plan.output_dir / COMPLETION_MANIFEST, manifest)
        return manifest
    except Exception as error:
        write_json(
            plan.output_dir / FAILURE_MANIFEST,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "panel": "Figure3C",
                "dataset_id": DATASET_ID,
                "fold": plan.fold,
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def run_figure3c(
    *,
    layout: Figure3Layout | None = None,
    folds: Sequence[int] | None = None,
    resume: bool = False,
    audit_scheduler: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Mapping[str, Any]:
    active = layout or Figure3Layout.default()
    preflight, plans = preflight_figure3c(
        layout=active,
        folds=folds,
        audit_scheduler=audit_scheduler,
        progress=progress,
    )
    selected = tuple(int(value) for value in preflight["folds"])
    completed = []
    resumed = []
    for fold in selected:
        plan = plans[fold]
        state = str(preflight["states"][f"fold{fold}"])
        if state == "complete_valid":
            if not resume:
                raise FileExistsError(
                    f"Validated Figure3C fold{fold} exists; pass --resume to skip"
                )
            resumed.append(fold)
            continue
        _execute_figure3c_fold(plan, active, progress=progress)
        completed.append(fold)
    aggregate = _aggregate_panel(
        panel="Figure3C",
        root=active.figure3c_root,
        folds=selected,
        layout=active,
        correct_control="correct",
    )
    return {
        **preflight,
        "operation": "materialize_centralized_metrics",
        "model_started": False,
        "completed_folds": completed,
        "resumed_folds": resumed,
        "aggregate_status": aggregate["status"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight/run Protocol A Figure 3A controls or summarize formal Figure 3C controls."
    )
    parser.add_argument(
        "--panel", choices=("3a", "3c"), default="3a"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run Figure3A controls or materialize Figure3C centralized metrics.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only folds whose identities and output hashes validate exactly.",
    )
    parser.add_argument(
        "--fold",
        "--folds",
        dest="folds",
        action="extend",
        nargs="+",
        type=int,
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    try:
        if args.panel == "3a":
            if args.run:
                report = run_figure3a(
                    folds=args.folds,
                    resume=bool(args.resume),
                    progress=progress,
                )
            else:
                report, _plans = preflight_figure3a(
                    folds=args.folds, progress=progress
                )
        else:
            if args.run:
                report = run_figure3c(
                    folds=args.folds,
                    resume=bool(args.resume),
                    progress=progress,
                )
            else:
                report, _plans = preflight_figure3c(
                    folds=args.folds, progress=progress
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
        "panel": report["panel"],
        "folds": report["folds"],
        "model_started": report["model_started"],
        "output_root": report["output_root"],
    }
    if args.run:
        summary["completed_folds"] = report["completed_folds"]
        summary["resumed_folds"] = report["resumed_folds"]
        summary["aggregate_status"] = report["aggregate_status"]
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
