"""Fail-closed, fold-aware ST preprocessing for Protocol A.

Matrices are expected to have spatial spots in rows and genes in columns.
Protocol A computes each spot's library size from the explicitly supplied
inner-training genes, then applies that denominator to every gene column.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np


PROTOCOL_A_TARGET_SUM = 10_000.0
PROTOCOL_A_POLICY = "inner_train_gene_library_size_applied_to_all_columns"
ZERO_TRAIN_LIBRARY_POLICY = "set_entire_normalized_spot_row_to_zero"

__all__ = [
    "PROTOCOL_A_POLICY",
    "PROTOCOL_A_TARGET_SUM",
    "ZERO_TRAIN_LIBRARY_POLICY",
    "normalize_st_protocol_a",
    "validate_gene_splits",
]


GeneIndices = Sequence[int] | np.ndarray


def _validate_index_array(
    values: GeneIndices,
    *,
    name: str,
    n_genes: int,
    allow_empty: bool,
) -> np.ndarray:
    """Return a private int64 copy after strict index validation."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of integers")
    if array.size == 0:
        if allow_empty:
            return np.empty(0, dtype=np.int64)
        raise ValueError(f"{name} must contain at least one gene index")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain integers, got dtype {array.dtype}")
    if np.any(array < 0) or np.any(array >= n_genes):
        raise ValueError(f"{name} contains an out-of-range gene index for {n_genes} genes")

    validated = array.astype(np.int64, copy=True)
    if np.unique(validated).size != validated.size:
        raise ValueError(f"{name} contains duplicate gene indices")
    return validated


def validate_gene_splits(
    n_genes: int,
    *,
    train_gene_idx: GeneIndices,
    val_gene_idx: GeneIndices,
    test_gene_idx: GeneIndices,
    require_complete_coverage: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and canonicalize train/validation/test gene indices.

    Each split must be one-dimensional, integer-valued, in range, internally
    unique, and mutually disjoint. Validation and test splits may be empty;
    the training split may not, because it defines Protocol A's denominator.
    When ``require_complete_coverage`` is true, the three splits must also
    partition every matrix column.
    """

    if isinstance(n_genes, (bool, np.bool_)) or not isinstance(
        n_genes, (int, np.integer)
    ):
        raise TypeError("n_genes must be an integer")
    n_genes = int(n_genes)
    if n_genes <= 0:
        raise ValueError("n_genes must be positive")
    if not isinstance(require_complete_coverage, (bool, np.bool_)):
        raise TypeError("require_complete_coverage must be a boolean")

    train = _validate_index_array(
        train_gene_idx,
        name="train_gene_idx",
        n_genes=n_genes,
        allow_empty=False,
    )
    validation = _validate_index_array(
        val_gene_idx,
        name="val_gene_idx",
        n_genes=n_genes,
        allow_empty=True,
    )
    test = _validate_index_array(
        test_gene_idx,
        name="test_gene_idx",
        n_genes=n_genes,
        allow_empty=True,
    )

    for left_name, left, right_name, right in (
        ("train_gene_idx", train, "val_gene_idx", validation),
        ("train_gene_idx", train, "test_gene_idx", test),
        ("val_gene_idx", validation, "test_gene_idx", test),
    ):
        overlap = np.intersect1d(left, right, assume_unique=True)
        if overlap.size:
            overlap_text = ", ".join(str(int(value)) for value in overlap[:5])
            raise ValueError(
                f"gene splits overlap: {left_name} and {right_name} share "
                f"index/indices {overlap_text}"
            )

    if require_complete_coverage:
        covered = np.concatenate((train, validation, test))
        if covered.size != n_genes:
            missing = np.setdiff1d(
                np.arange(n_genes, dtype=np.int64), covered, assume_unique=False
            )
            missing_text = ", ".join(str(int(value)) for value in missing[:5])
            raise ValueError(
                "gene splits do not completely cover all matrix columns; "
                f"missing index/indices {missing_text}"
            )

    return train, validation, test


def _denominator_gene_index_sha256(indices: np.ndarray) -> str:
    """Hash the ordered index list using canonical compact JSON."""

    encoded = json.dumps(indices.tolist(), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_target_sum(target_sum: float) -> float:
    if isinstance(target_sum, (bool, np.bool_, str, bytes)):
        raise TypeError("target_sum must be a real number")
    try:
        value = float(target_sum)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("target_sum must be a real number") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("target_sum must be finite and positive")
    if value != PROTOCOL_A_TARGET_SUM:
        raise ValueError(
            f"Protocol A requires target_sum={PROTOCOL_A_TARGET_SUM}, got {value}"
        )
    return value


def normalize_st_protocol_a(
    counts: np.ndarray,
    *,
    inner_train_gene_idx: GeneIndices,
    val_gene_idx: GeneIndices,
    test_gene_idx: GeneIndices,
    require_complete_coverage: bool = True,
    target_sum: float = PROTOCOL_A_TARGET_SUM,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply Protocol A normalization and return a deterministic audit.

    ``inner_train_gene_idx`` is deliberately required and keyword-only. There
    is no all-gene fallback. Validation and test indices are accepted only so
    their fold partition can be checked before any normalization is performed.

    The returned matrix has ``float64`` dtype. The audit contains only Python
    scalars and containers and can therefore be passed directly to
    :func:`json.dumps`.
    """

    if np.ma.isMaskedArray(counts):
        raise TypeError("counts must not be a masked array")
    raw = np.asarray(counts)
    if raw.ndim != 2:
        raise ValueError(f"counts must be a two-dimensional spot-by-gene matrix, got {raw.shape}")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"counts must have a real numeric dtype, got {raw.dtype}")
    if not np.isfinite(raw).all():
        raise ValueError("counts must contain only finite values")
    if np.any(raw < 0):
        raise ValueError("counts must be nonnegative")

    train, _, _ = validate_gene_splits(
        raw.shape[1],
        train_gene_idx=inner_train_gene_idx,
        val_gene_idx=val_gene_idx,
        test_gene_idx=test_gene_idx,
        require_complete_coverage=require_complete_coverage,
    )
    target = _validated_target_sum(target_sum)
    matrix = raw.astype(np.float64, copy=False)

    try:
        with np.errstate(over="raise", invalid="raise"):
            library_sizes = matrix[:, train].sum(axis=1, dtype=np.float64)
    except FloatingPointError as error:
        raise ValueError("inner-train library-size calculation overflowed") from error
    if not np.isfinite(library_sizes).all():
        raise ValueError("inner-train library sizes must be finite")

    zero_library = library_sizes == 0.0
    nonzero_library = ~zero_library
    normalized = np.zeros(matrix.shape, dtype=np.float64)
    if np.any(nonzero_library):
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                scaled = (
                    matrix[nonzero_library]
                    / library_sizes[nonzero_library, None]
                    * target
                )
                normalized[nonzero_library] = np.log1p(scaled)
        except FloatingPointError as error:
            raise ValueError("Protocol A normalization produced a non-finite value") from error

    zero_rows = np.flatnonzero(zero_library).astype(np.int64, copy=False)
    audit: dict[str, object] = {
        "protocol": "A",
        "policy": PROTOCOL_A_POLICY,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "target_sum": target,
        "denominator_gene_count": int(train.size),
        "denominator_gene_index_sha256": _denominator_gene_index_sha256(train),
        "zero_train_library_rows": [int(row) for row in zero_rows],
        "zero_train_library_spot_count": int(zero_rows.size),
        "zero_train_library_policy": ZERO_TRAIN_LIBRARY_POLICY,
    }
    return normalized, audit
