from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = PROJECT_ROOT / "main"
for path in [PROJECT_ROOT, MAIN_ROOT]:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from protocol_a_preprocessing import (  # noqa: E402
    PROTOCOL_A_POLICY,
    PROTOCOL_A_TARGET_SUM,
    normalize_st_protocol_a,
    validate_gene_splits,
)


STRICT_MODEL_GENE_SCOPE = "train_indices"
LEGACY_MODEL_GENE_SCOPE = "non_test"
RUN_AUDIT_NAME = "adapter_run_audit.json"
ADAPTER_VERSION = "protocol_a_v1"


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--locations-path", type=str, required=True)
    p.add_argument("--st-data", type=str, required=True)
    p.add_argument("--sc-data", type=str, required=True)
    p.add_argument("--gene-split-json", type=str, required=True)
    p.add_argument("--train-gene-idx-path", type=str, required=True)
    p.add_argument("--val-gene-idx-path", type=str, required=True)
    p.add_argument("--test-gene-idx-path", type=str, required=True)
    p.add_argument(
        "--model-gene-scope",
        type=str,
        default=STRICT_MODEL_GENE_SCOPE,
        choices=[STRICT_MODEL_GENE_SCOPE, LEGACY_MODEL_GENE_SCOPE],
        help=(
            "train_indices is formal Protocol A. non_test exposes validation ST "
            "genes to fitting and is always diagnostic-only."
        ),
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--scrna-max-cells", type=int, default=5000)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--t-min", type=int, default=5)
    p.add_argument("--n-neighbors", type=int, default=50)
    p.add_argument("--max-epoch-num", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--converge-ratio", type=float, default=0.004)
    p.add_argument("--weight-decay", type=float, default=0.0002)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-adapter-metrics", action="store_true")
    return p.parse_args(argv)


def to_dense(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def artifact_record(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def collect_artifact_records(
    paths: Mapping[str, str | Path | None],
) -> dict[str, dict[str, object] | None]:
    return {
        str(name): (artifact_record(path) if path is not None else None)
        for name, path in paths.items()
    }


def normalize_st_counts(
    count_matrix,
    *,
    inner_train_gene_idx,
    val_gene_idx,
    test_gene_idx,
):
    """Normalize ST counts through the single public Protocol A implementation."""

    return normalize_st_protocol_a(
        count_matrix,
        inner_train_gene_idx=inner_train_gene_idx,
        val_gene_idx=val_gene_idx,
        test_gene_idx=test_gene_idx,
        require_complete_coverage=True,
        target_sum=PROTOCOL_A_TARGET_SUM,
    )


def normalize_st_counts_protocol_a(
    count_matrix,
    train_idx,
    val_idx,
    test_idx,
):
    return normalize_st_counts(
        count_matrix,
        inner_train_gene_idx=train_idx,
        val_gene_idx=val_idx,
        test_gene_idx=test_idx,
    )


def normalize_from_counts(
    count_matrix,
    template_adata,
    *,
    inner_train_gene_idx,
    val_gene_idx,
    test_gene_idx,
):
    import anndata as ad

    normalized, normalization_audit = normalize_st_counts(
        count_matrix,
        inner_train_gene_idx=inner_train_gene_idx,
        val_gene_idx=val_gene_idx,
        test_gene_idx=test_gene_idx,
    )
    adata = ad.AnnData(X=normalized)
    adata.obs_names = template_adata.obs_names.copy()
    adata.var_names = template_adata.var_names.copy()
    adata.obs = template_adata.obs.copy()
    adata.var = template_adata.var.copy()
    adata.obsm["spatial"] = np.asarray(
        template_adata.obsm["spatial"], dtype=np.float32
    )
    adata.layers["counts"] = np.asarray(count_matrix).copy()
    return adata, normalization_audit


def _load_split_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        split_obj = json.load(handle)
    if not isinstance(split_obj, dict):
        raise ValueError(f"Gene split must be a JSON object: {path}")
    return split_obj


def _validated_gene_name_list(value, *, key: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a JSON list")
    names = [str(item) for item in value]
    if len(names) != len(set(names)):
        raise ValueError(f"{key} contains duplicate gene names")
    return names


def resolve_gene_split(
    split_obj: Mapping[str, object],
    original_genes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve an explicit, complete train/validation/test partition.

    Protocol A deliberately accepts no complement-derived or test-only split.
    When redundant gene-name lists are present, they must agree exactly with
    the indexed ST column order.
    """

    if not isinstance(split_obj, Mapping):
        raise TypeError("Gene split must be a mapping")
    genes = [str(gene) for gene in original_genes]
    if not genes:
        raise ValueError("Original ST gene axis must not be empty")
    if len(genes) != len(set(genes)):
        raise ValueError("Original ST gene axis contains duplicate gene names")

    required_keys = ("train_gene_idx", "val_gene_idx", "test_gene_idx")
    missing_keys = [key for key in required_keys if key not in split_obj]
    if missing_keys:
        raise KeyError(
            "Protocol A requires explicit train/val/test masks; missing "
            + ", ".join(missing_keys)
        )

    train_idx, val_idx, test_idx = validate_gene_splits(
        len(genes),
        train_gene_idx=split_obj["train_gene_idx"],
        val_gene_idx=split_obj["val_gene_idx"],
        test_gene_idx=split_obj["test_gene_idx"],
        require_complete_coverage=True,
    )

    name_definitions = (
        ("train", train_idx, ("train_genes",)),
        ("validation", val_idx, ("val_genes", "validation_genes")),
        ("test", test_idx, ("test_genes", "test_target_genes")),
    )
    for label, indices, keys in name_definitions:
        expected = [genes[int(index)] for index in indices]
        observed_lists = []
        for key in keys:
            if key in split_obj:
                observed_lists.append(
                    (key, _validated_gene_name_list(split_obj[key], key=key))
                )
        for key, observed in observed_lists:
            if observed != expected:
                raise ValueError(
                    f"{key} does not match {label} indices in the original ST column order"
                )
        if len(observed_lists) > 1:
            reference_key, reference = observed_lists[0]
            for key, observed in observed_lists[1:]:
                if observed != reference:
                    raise ValueError(
                        f"Split gene lists {reference_key} and {key} disagree"
                    )

    return train_idx, val_idx, test_idx


def _load_index_array(path: str | Path, *, label: str) -> np.ndarray:
    try:
        values = np.load(path, allow_pickle=False)
    except ValueError as error:
        raise ValueError(f"Could not load {label} as a non-pickle NumPy array") from error
    return np.asarray(values)


def load_and_validate_gene_splits(
    *,
    train_gene_idx_path: str | Path,
    val_gene_idx_path: str | Path,
    test_gene_idx_path: str | Path,
    n_genes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx, test_idx = validate_gene_splits(
        n_genes,
        train_gene_idx=_load_index_array(
            train_gene_idx_path, label="train mask"
        ),
        val_gene_idx=_load_index_array(
            val_gene_idx_path, label="validation mask"
        ),
        test_gene_idx=_load_index_array(test_gene_idx_path, label="test mask"),
        require_complete_coverage=True,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _metadata_indices(values, *, key: str, n_genes: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{key} must be a one-dimensional integer list")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{key} must contain integer gene indices, not booleans")
    array = array.astype(np.int64, copy=False)
    if np.unique(array).size != array.size:
        raise ValueError(f"{key} contains duplicate gene indices")
    if array.size and (np.any(array < 0) or np.any(array >= n_genes)):
        raise ValueError(f"{key} contains an out-of-range gene index")
    return array


def _metadata_scope(key: str, observed, candidates) -> str:
    observed_list = list(observed)
    for scope, expected in candidates:
        if observed_list == list(expected):
            return scope
    expected_scopes = ", ".join(scope for scope, _ in candidates)
    raise ValueError(
        f"Gene split metadata field {key} disagrees with explicit masks "
        f"(expected scope: {expected_scopes})"
    )


def validate_gene_split_metadata(
    split_obj: Mapping[str, object],
    original_genes: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, str]:
    """Cross-check known split schemas against the authoritative mask files."""

    if not isinstance(split_obj, Mapping):
        raise TypeError("Gene split must be a mapping")
    genes = [str(gene) for gene in original_genes]
    if len(genes) != len(set(genes)):
        raise ValueError("Original ST gene axis contains duplicate gene names")
    hidden_idx = np.concatenate((val_idx, test_idx))
    non_test_idx = np.sort(np.concatenate((train_idx, val_idx)))

    index_candidates = {
        "train_gene_idx": (
            ("train", train_idx),
            ("non_test", non_test_idx),
        ),
        "train_idx": (("train", train_idx), ("non_test", non_test_idx)),
        "inner_train_gene_idx": (("train", train_idx),),
        "val_gene_idx": (("validation", val_idx),),
        "validation_gene_idx": (("validation", val_idx),),
        "val_idx": (("validation", val_idx),),
        "validation_idx": (("validation", val_idx),),
        "inner_validation_gene_idx": (("validation", val_idx),),
        "final_test_gene_idx": (("test", test_idx),),
        "test_gene_idx": (
            ("test", test_idx),
            ("validation_and_test", hidden_idx),
        ),
        "test_idx": (
            ("test", test_idx),
            ("validation_and_test", hidden_idx),
        ),
    }
    name_candidates = {
        "train_genes": (
            ("train", [genes[index] for index in train_idx]),
            ("non_test", [genes[index] for index in non_test_idx]),
        ),
        "inner_train_genes": (
            ("train", [genes[index] for index in train_idx]),
        ),
        "val_genes": (("validation", [genes[index] for index in val_idx]),),
        "validation_genes": (
            ("validation", [genes[index] for index in val_idx]),
        ),
        "inner_validation_genes": (
            ("validation", [genes[index] for index in val_idx]),
        ),
        "final_test_genes": (("test", [genes[index] for index in test_idx]),),
        "test_genes": (
            ("test", [genes[index] for index in test_idx]),
            ("validation_and_test", [genes[index] for index in hidden_idx]),
        ),
        "test_target_genes": (
            ("test", [genes[index] for index in test_idx]),
            ("validation_and_test", [genes[index] for index in hidden_idx]),
        ),
    }

    validated = {}
    for key, candidates in index_candidates.items():
        if key in split_obj:
            observed = _metadata_indices(
                split_obj[key], key=key, n_genes=len(genes)
            )
            validated[key] = _metadata_scope(key, observed, candidates)
    for key, candidates in name_candidates.items():
        if key in split_obj:
            observed = _validated_gene_name_list(split_obj[key], key=key)
            validated[key] = _metadata_scope(key, observed, candidates)
    if not validated:
        raise ValueError(
            "Gene split JSON contains no recognized fields to validate against masks"
        )
    return validated


def load_gene_split(
    path: str | Path,
    original_genes: Sequence[str],
    *,
    train_gene_idx_path: str | Path | None = None,
    val_gene_idx_path: str | Path | None = None,
    test_gene_idx_path: str | Path | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    split_obj = _load_split_json(path)
    mask_paths = (train_gene_idx_path, val_gene_idx_path, test_gene_idx_path)
    if any(mask_path is not None for mask_path in mask_paths):
        if not all(mask_path is not None for mask_path in mask_paths):
            raise ValueError(
                "Explicit mask files are all-or-none: provide train, val, and test paths"
            )
        split_indices = load_and_validate_gene_splits(
            train_gene_idx_path=train_gene_idx_path,
            val_gene_idx_path=val_gene_idx_path,
            test_gene_idx_path=test_gene_idx_path,
            n_genes=len(original_genes),
        )
        validate_gene_split_metadata(
            split_obj,
            original_genes,
            split_indices[0],
            split_indices[1],
            split_indices[2],
        )
    else:
        split_indices = resolve_gene_split(split_obj, original_genes)

    return split_obj, *split_indices


def load_test_genes(path: str):
    """Backward-compatible reader; the strict main path uses load_gene_split."""

    split_obj = _load_split_json(path)
    for key in ("test_genes", "test_target_genes"):
        if key in split_obj:
            return _validated_gene_name_list(split_obj[key], key=key)
    raise KeyError(f"Could not find test gene list in {path}. Keys: {list(split_obj)}")


def resolve_model_gene_scope(
    model_gene_scope: str,
) -> tuple[str, bool]:
    if model_gene_scope == STRICT_MODEL_GENE_SCOPE:
        return STRICT_MODEL_GENE_SCOPE, False
    if model_gene_scope == LEGACY_MODEL_GENE_SCOPE:
        return LEGACY_MODEL_GENE_SCOPE, True
    raise ValueError(f"Unsupported model gene scope: {model_gene_scope!r}")


def model_gene_indices(
    model_gene_scope: str,
    train_gene_idx,
    val_gene_idx,
    test_gene_idx,
) -> tuple[np.ndarray, np.ndarray]:
    n_genes = int(
        len(train_gene_idx) + len(val_gene_idx) + len(test_gene_idx)
    )
    train_idx, val_idx, test_idx = validate_gene_splits(
        n_genes,
        train_gene_idx=train_gene_idx,
        val_gene_idx=val_gene_idx,
        test_gene_idx=test_gene_idx,
        require_complete_coverage=True,
    )
    canonical_scope, _ = resolve_model_gene_scope(model_gene_scope)
    if canonical_scope == STRICT_MODEL_GENE_SCOPE:
        fit_idx = train_idx.copy()
        predict_idx = np.concatenate((val_idx, test_idx))
    else:
        fit_idx = np.sort(np.concatenate((train_idx, val_idx)))
        predict_idx = test_idx.copy()
    if predict_idx.size == 0:
        raise ValueError(
            f"No genes remain hidden under model gene scope {model_gene_scope}"
        )
    validate_gene_splits(
        n_genes,
        train_gene_idx=fit_idx,
        val_gene_idx=np.empty(0, dtype=np.int64),
        test_gene_idx=predict_idx,
        require_complete_coverage=True,
    )
    return fit_idx, predict_idx


def assemble_full_prediction_matrix(
    normalized_st,
    *,
    fit_idx,
    predict_idx,
    predicted_values,
    zero_rows,
) -> np.ndarray:
    normalized = np.asarray(normalized_st)
    prediction = np.asarray(predicted_values)
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    predict_idx = np.asarray(predict_idx, dtype=np.int64)
    if normalized.ndim != 2:
        raise ValueError("normalized_st must be a two-dimensional matrix")
    if prediction.shape != (normalized.shape[0], predict_idx.size):
        raise ValueError(
            "stPlus prediction shape does not match spots and hidden genes: "
            f"{prediction.shape} != {(normalized.shape[0], predict_idx.size)}"
        )
    validate_gene_splits(
        normalized.shape[1],
        train_gene_idx=fit_idx,
        val_gene_idx=np.empty(0, dtype=np.int64),
        test_gene_idx=predict_idx,
        require_complete_coverage=True,
    )
    if not np.isfinite(normalized).all() or not np.isfinite(prediction).all():
        raise ValueError("Normalized ST and stPlus predictions must be finite")
    output = np.zeros(normalized.shape, dtype=np.float32)
    output[:, fit_idx] = normalized[:, fit_idx].astype(np.float32, copy=False)
    output[:, predict_idx] = prediction.astype(np.float32, copy=False)
    zero_rows = np.asarray(zero_rows, dtype=np.int64)
    if zero_rows.ndim != 1:
        raise ValueError("zero_rows must be one-dimensional")
    if zero_rows.size and (
        np.any(zero_rows < 0) or np.any(zero_rows >= normalized.shape[0])
    ):
        raise ValueError("zero_rows contains an out-of-range spot index")
    if zero_rows.size:
        output[zero_rows, :] = 0.0
    return output


def validate_prediction_contract(
    pred_df,
    *,
    target_genes: Sequence[str],
    n_spots: int,
) -> np.ndarray:
    targets = [str(gene) for gene in target_genes]
    columns = [str(gene) for gene in pred_df.columns]
    if len(columns) != len(set(columns)):
        raise ValueError("stPlus prediction contains duplicate gene columns")
    missing = [gene for gene in targets if gene not in set(columns)]
    if missing:
        raise ValueError(
            f"stPlus did not predict every test gene; missing {missing[:10]}"
        )
    if len(pred_df) != int(n_spots):
        raise ValueError(
            f"stPlus prediction row count {len(pred_df)} does not match {n_spots} spots"
        )
    prediction = pred_df.loc[:, targets].to_numpy(dtype=np.float32)
    if not np.isfinite(prediction).all():
        raise ValueError("stPlus prediction must contain only finite values")
    return prediction


def build_split_audit(
    *,
    n_genes: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_records: Mapping[str, dict[str, object] | None],
    metadata_fields: Mapping[str, str],
) -> dict[str, object]:
    canonical = {
        "train_gene_idx": [int(value) for value in train_idx],
        "val_gene_idx": [int(value) for value in val_idx],
        "test_gene_idx": [int(value) for value in test_idx],
    }
    split_records = {}
    for scope, indices, input_key in (
        ("train", train_idx, "train_gene_idx_path"),
        ("validation", val_idx, "val_gene_idx_path"),
        ("test", test_idx, "test_gene_idx_path"),
    ):
        source_record = input_records.get(input_key)
        if source_record is None:
            raise ValueError(f"Missing input audit record for {input_key}")
        split_records[scope] = {
            "path": source_record["path"],
            "file_sha256": source_record["sha256"],
            "canonical_index_sha256": _canonical_json_sha256(
                [int(value) for value in indices]
            ),
            "gene_count": int(indices.size),
        }
    return {
        "n_genes": int(n_genes),
        "require_complete_coverage": True,
        "complete_coverage": True,
        "mutually_disjoint": True,
        "canonical_sha256": _canonical_json_sha256(canonical),
        "metadata_fields_validated": dict(metadata_fields),
        **split_records,
    }


def build_model_audit(
    *,
    model_gene_scope: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    fit_idx: np.ndarray,
    predict_idx: np.ndarray,
) -> dict[str, object]:
    diagnostic = model_gene_scope == LEGACY_MODEL_GENE_SCOPE
    return {
        "gene_scope": model_gene_scope,
        "fit_gene_count": int(fit_idx.size),
        "predict_gene_count": int(predict_idx.size),
        "output_gene_count": int(train_idx.size + val_idx.size + test_idx.size),
        "train_index_gene_count": int(train_idx.size),
        "validation_index_gene_count": int(val_idx.size),
        "test_index_gene_count": int(test_idx.size),
        "fit_train_gene_count": int(train_idx.size),
        "fit_validation_gene_count": int(val_idx.size if diagnostic else 0),
        "fit_test_gene_count": 0,
        "hidden_validation_gene_count": int(0 if diagnostic else val_idx.size),
        "hidden_test_gene_count": int(test_idx.size),
        "fit_hidden_overlap_count": int(
            np.intersect1d(fit_idx, predict_idx).size
        ),
    }


def build_run_audit(
    *,
    input_records: Mapping[str, dict[str, object] | None],
    output_paths: Mapping[str, str | Path],
    prediction_output_key: str,
    prediction_shape: Sequence[int],
    prediction_dtype: str,
    normalization_audit: Mapping[str, object],
    scope: Mapping[str, object],
    counts: Mapping[str, object],
    hyperparameters: Mapping[str, object],
    split_audit: Mapping[str, object] | None = None,
    model_audit: Mapping[str, object] | None = None,
    zero_library_spot_names: Sequence[str] = (),
) -> dict[str, object]:
    output_records = {
        str(name): artifact_record(path) for name, path in output_paths.items()
    }
    if prediction_output_key not in output_records:
        raise KeyError(
            f"Prediction output key {prediction_output_key!r} is absent from output paths"
        )
    prediction_record = dict(output_records[prediction_output_key])
    prediction_record["shape"] = [int(value) for value in prediction_shape]
    prediction_record["dtype"] = str(prediction_dtype)
    output_records[prediction_output_key] = prediction_record
    input_hashes = {
        str(name): (record["sha256"] if record is not None else None)
        for name, record in input_records.items()
    }
    output_hashes = {
        str(name): record["sha256"] for name, record in output_records.items()
    }
    scope_dict = dict(scope)
    counts_dict = dict(counts)
    diagnostic = bool(scope_dict.get("diagnostic_only", False))
    input_paths = {
        str(name): (record["path"] if record is not None else None)
        for name, record in input_records.items()
    }
    audit = {
        "adapter": "stPlus",
        "method": "stPlus",
        "adapter_version": ADAPTER_VERSION,
        "protocol": "A",
        "protocol_role": (
            "explicit_non_test_diagnostic" if diagnostic else "strict_primary_modeA"
        ),
        "eligible_for_strict_primary": not diagnostic,
        "formal_protocol_a_run": not diagnostic,
        "run_mode": (
            "diagnostic_non_test" if diagnostic else "formal_protocol_a"
        ),
        "diagnostic": {
            "enabled": diagnostic,
            "reason": (
                "validation ST genes exposed by explicit non_test opt-in"
                if diagnostic
                else None
            ),
        },
        "fit_visibility": (
            "explicit_diagnostic_scope"
            if diagnostic
            else "inner_train_genes_only; validation_and_test_hidden"
        ),
        "hashes": {"inputs": input_hashes, "outputs": output_hashes},
        "input": dict(input_records),
        "input_paths": input_paths,
        "scope": scope_dict,
        "counts": counts_dict,
        "normalization": dict(normalization_audit),
        "normalization_audit": dict(normalization_audit),
        "hyperparameters": dict(hyperparameters),
        "output": {
            "directory": str(Path(prediction_record["path"]).parent),
            "prediction_artifact": prediction_output_key,
            "prediction": prediction_record,
            "artifacts": output_records,
        },
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "outputs": output_records,
        "output_files": output_records,
        "st_normalization_scope": scope_dict.get("st_normalization_scope"),
        "model_gene_scope": scope_dict.get("model_gene_scope"),
        "diagnostic_only": diagnostic,
        "train_gene_count": counts_dict.get("inner_train_gene_count"),
        "validation_gene_count": counts_dict.get("validation_gene_count"),
        "test_gene_count": counts_dict.get("test_gene_count"),
        "model_train_gene_count": counts_dict.get("model_train_gene_count"),
        "zero_library_spot_count": counts_dict.get("zero_train_library_spot_count"),
        "zero_library_spots": [str(name) for name in zero_library_spot_names],
        "imputed_matrix_shape": [int(value) for value in prediction_shape],
        "cache": {
            "enabled": False,
            "preexisting_checkpoint_read": False,
            "cross_scope_reuse_allowed": False,
            "policy": "fresh_output_scope_required",
        },
    }
    if split_audit is not None:
        audit["split"] = dict(split_audit)
        audit["split_sha256"] = split_audit.get("canonical_sha256")
    if model_audit is not None:
        audit["model"] = dict(model_audit)
        audit["model_fit_gene_count"] = model_audit.get("fit_gene_count")
        audit["model_predict_gene_count"] = model_audit.get("predict_gene_count")
    return audit


def write_run_audit(output_dir: str | Path, audit: Mapping[str, object]) -> Path:
    audit_path = Path(output_dir) / RUN_AUDIT_NAME
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(audit), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return audit_path


def load_scrna_full_downsampled(
    path,
    target_genes=None,
    max_cells=5000,
    seed=42,
    target_sum=1e4,
):
    import pandas as pd

    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if target_genes is not None:
        target_set = set(map(str, target_genes))
        col_overlap = len(set(map(str, df.columns)).intersection(target_set))
        idx_overlap = len(set(map(str, df.index)).intersection(target_set))
        if idx_overlap > col_overlap:
            keep_genes = [g for g in df.index if str(g) in target_set]
            if len(keep_genes) == 0:
                raise ValueError("No overlapping genes found on scRNA row index")
            df = df.loc[keep_genes].T
        else:
            keep_genes = [g for g in df.columns if str(g) in target_set]
            if len(keep_genes) == 0:
                raise ValueError("No overlapping genes found on scRNA columns")
            df = df[keep_genes]
    else:
        df = df.T

    if df.columns.duplicated().any():
        duplicate = list(map(str, df.columns[df.columns.duplicated()].tolist()))
        raise ValueError(f"scRNA matrix contains duplicate gene columns: {duplicate[:10]}")

    if max_cells is not None and max_cells > 0 and len(df) > max_cells:
        rng = np.random.default_rng(seed)
        keep_idx = np.sort(rng.choice(len(df), size=max_cells, replace=False))
        df = df.iloc[keep_idx].copy()

    x = df.to_numpy(dtype=np.float32)
    lib = np.clip(x.sum(axis=1, keepdims=True), 1.0, None)
    x = x / lib * float(target_sum)
    x = np.log1p(x)
    return pd.DataFrame(
        x,
        index=df.index.astype(str),
        columns=df.columns.astype(str),
    )


def cal_ssim_ref(im1, im2, m_val):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, length = 0.01, 0.03, m_val
    c1 = (k1 * length) ** 2
    c2 = (k2 * length) ** 2
    c3 = c2 / 2
    l12 = (2 * mu1 * mu2 + c1) / (mu1**2 + mu2**2 + c1)
    c12 = (2 * sigma1 * sigma2 + c2) / (sigma1**2 + sigma2**2 + c2)
    s12 = (sigma12 + c3) / (sigma1 * sigma2 + c3)
    return l12 * c12 * s12


def scale_max_df(df):
    import pandas as pd

    result = pd.DataFrame(index=df.index)
    for label, content in df.items():
        denom = float(content.max())
        if abs(denom) < 1e-12:
            denom = 1.0
        result[label] = content / denom
    return result


def scale_z_score_df(df):
    import pandas as pd
    import scipy.stats as st

    result = pd.DataFrame(index=df.index)
    for label, content in df.items():
        z = st.zscore(content)
        result[label] = np.nan_to_num(z, nan=0.0)
    return result


def scale_plus_df(df):
    import pandas as pd

    result = pd.DataFrame(index=df.index)
    for label, content in df.items():
        denom = float(content.sum())
        if abs(denom) < 1e-12:
            denom = 1.0
        result[label] = content / denom
    return result


def compute_stdiff_style_gene_metrics(x_true, x_pred, genes, target_gene_names):
    import pandas as pd
    import scipy.stats as st
    from scipy.stats import spearmanr

    raw = pd.DataFrame(x_true, columns=genes)
    raw.columns = [str(x).upper() for x in raw.columns]
    raw = raw.T.loc[~raw.T.index.duplicated(keep="first")].T.fillna(1e-20)

    imp = pd.DataFrame(x_pred, columns=genes)
    imp.columns = [str(x).upper() for x in imp.columns]
    imp = imp.T.loc[~imp.T.index.duplicated(keep="first")].T.fillna(1e-20)

    target_gene_names = [str(g).upper() for g in target_gene_names]
    cols = [g for g in target_gene_names if g in raw.columns and g in imp.columns]

    raw_spcc = raw[cols]
    imp_spcc = imp[cols]
    raw_ssim = scale_max_df(raw[cols].copy())
    imp_ssim = scale_max_df(imp[cols].copy())
    raw_js = scale_plus_df(raw[cols].copy())
    imp_js = scale_plus_df(imp[cols].copy())
    raw_rmse = scale_z_score_df(raw[cols].copy())
    imp_rmse = scale_z_score_df(imp[cols].copy())

    rows = []
    for gene in cols:
        try:
            spcc = float(
                spearmanr(
                    raw_spcc[gene].fillna(1e-20),
                    imp_spcc[gene].fillna(1e-20),
                )[0]
            )
        except Exception:
            spcc = np.nan

        raw_col = raw_ssim[gene].fillna(1e-20)
        imp_col = imp_ssim[gene].fillna(1e-20)
        m_val = max(float(raw_col.max()), float(imp_col.max()))
        try:
            ssim = float(
                cal_ssim_ref(
                    raw_col.to_numpy().reshape(-1, 1),
                    imp_col.to_numpy().reshape(-1, 1),
                    m_val,
                )
            )
        except Exception:
            ssim = np.nan

        raw_col_js = raw_js[gene].fillna(1e-20)
        imp_col_js = imp_js[gene].fillna(1e-20)
        mid = (raw_col_js + imp_col_js) / 2.0
        try:
            js = float(
                0.5 * st.entropy(raw_col_js, mid)
                + 0.5 * st.entropy(imp_col_js, mid)
            )
        except Exception:
            js = np.nan

        raw_col_rmse = raw_rmse[gene].fillna(1e-20)
        imp_col_rmse = imp_rmse[gene].fillna(1e-20)
        try:
            rmse = float(np.sqrt(((raw_col_rmse - imp_col_rmse) ** 2).mean()))
        except Exception:
            rmse = np.nan

        rows.append({"gene": gene, "SPCC": spcc, "SSIM": ssim, "RMSE": rmse, "JS": js})

    gene_df = pd.DataFrame(rows)
    summary = {
        "SPCC_gene_median_stdiff_style": (
            float(np.nanmedian(gene_df["SPCC"])) if len(gene_df) else np.nan
        ),
        "SSIM_gene_median_stdiff_style": (
            float(np.nanmedian(gene_df["SSIM"])) if len(gene_df) else np.nan
        ),
        "RMSE_gene_median_stdiff_style": (
            float(np.nanmedian(gene_df["RMSE"])) if len(gene_df) else np.nan
        ),
        "JS_gene_median_stdiff_style": (
            float(np.nanmedian(gene_df["JS"])) if len(gene_df) else np.nan
        ),
    }
    return gene_df, summary


def main(
    argv: Sequence[str] | None = None,
    *,
    stplus_fn=None,
    st_loader=None,
    scrna_loader=None,
):
    import pandas as pd

    if stplus_fn is None:
        from baseline.stPlus.model import stPlus as stplus_fn
    if st_loader is None:
        from utils import load_mhpr_from_txt as st_loader
    if scrna_loader is None:
        scrna_loader = load_scrna_full_downsampled

    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "locations_path": args.locations_path,
        "st_data": args.st_data,
        "sc_data": args.sc_data,
        "gene_split_json": args.gene_split_json,
        "train_gene_idx_path": args.train_gene_idx_path,
        "val_gene_idx_path": args.val_gene_idx_path,
        "test_gene_idx_path": args.test_gene_idx_path,
    }
    input_records = collect_artifact_records(input_paths)

    adata_raw = st_loader(
        locations_path=args.locations_path,
        counts_path=args.st_data,
        normalize=False,
        store_raw_layer=True,
    )
    x_counts = to_dense(
        adata_raw.layers["counts"]
        if "counts" in adata_raw.layers
        else adata_raw.X
    ).astype(np.float32)
    original_genes = list(map(str, adata_raw.var_names))
    split_obj, train_idx, val_idx, test_idx = load_gene_split(
        args.gene_split_json,
        original_genes,
        train_gene_idx_path=args.train_gene_idx_path,
        val_gene_idx_path=args.val_gene_idx_path,
        test_gene_idx_path=args.test_gene_idx_path,
    )
    metadata_fields = validate_gene_split_metadata(
        split_obj,
        original_genes,
        train_idx,
        val_idx,
        test_idx,
    )
    fit_idx, predict_idx = model_gene_indices(
        args.model_gene_scope, train_idx, val_idx, test_idx
    )
    model_scope, diagnostic_only = resolve_model_gene_scope(
        args.model_gene_scope
    )

    normalized_st, normalization_audit = normalize_st_counts(
        x_counts,
        inner_train_gene_idx=train_idx,
        val_gene_idx=val_idx,
        test_gene_idx=test_idx,
    )
    model_genes = [original_genes[int(index)] for index in fit_idx]
    predict_genes = [original_genes[int(index)] for index in predict_idx]
    test_genes = [original_genes[int(index)] for index in test_idx]

    spatial_df = pd.DataFrame(
        normalized_st[:, fit_idx].astype(np.float32),
        index=list(map(str, adata_raw.obs_names)),
        columns=model_genes,
    )
    scrna_df = scrna_loader(
        path=args.sc_data,
        target_genes=original_genes,
        max_cells=(
            None
            if args.scrna_max_cells is not None and args.scrna_max_cells <= 0
            else args.scrna_max_cells
        ),
        seed=args.seed,
    )

    scrna_gene_set = set(map(str, scrna_df.columns))
    missing_model_genes = [gene for gene in model_genes if gene not in scrna_gene_set]
    if missing_model_genes:
        raise ValueError(
            "Model fitting genes are absent from the scRNA matrix: "
            f"{missing_model_genes[:10]}"
        )
    missing_predict_genes = [
        gene for gene in predict_genes if gene not in scrna_gene_set
    ]
    if missing_predict_genes:
        raise ValueError(
            "Protocol A requires complete hidden-gene prediction coverage; "
            f"scRNA is missing {missing_predict_genes[:10]}"
        )

    other_gene_count = len(
        np.setdiff1d(
            scrna_df.columns.values,
            np.hstack(
                (
                    spatial_df.columns.values,
                    np.asarray(predict_genes, dtype=object),
                )
            ),
        )
    )
    top_k = min(int(args.top_k), max(other_gene_count, 0))
    print(
        f"[stPlus] model genes = {spatial_df.shape[1]}, "
        f"hidden genes = {len(predict_genes)}, final test genes = {len(test_genes)}, "
        f"extra genes = {other_gene_count}, top_k = {top_k}, scope = {model_scope}"
    )

    save_prefix = str(output_dir / "stplus_model")
    existing_checkpoints = sorted(
        output_dir.glob(f"stplus_model-{args.t_min}min*.pt")
    )
    if existing_checkpoints:
        raise FileExistsError(
            "Refusing to reuse pre-existing stPlus checkpoints across run scopes: "
            f"{[path.name for path in existing_checkpoints[:10]]}"
        )
    pred_df = stplus_fn(
        spatial_df=spatial_df,
        scrna_df=scrna_df,
        genes_to_predict=np.asarray(predict_genes, dtype=object),
        save_path_prefix=save_prefix,
        top_k=top_k,
        t_min=args.t_min,
        data_quality=None,
        random_seed=args.seed,
        verbose=True,
        n_neighbors=args.n_neighbors,
        converge_ratio=args.converge_ratio,
        max_epoch_num=args.max_epoch_num,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    prediction = validate_prediction_contract(
        pred_df,
        target_genes=predict_genes,
        n_spots=normalized_st.shape[0],
    )

    imputed = assemble_full_prediction_matrix(
        normalized_st,
        fit_idx=fit_idx,
        predict_idx=predict_idx,
        predicted_values=prediction,
        zero_rows=normalization_audit["zero_train_library_rows"],
    )

    prediction_path = output_dir / "imputed_expression.npy"
    split_copy_path = output_dir / "gene_split.json"
    np.save(prediction_path, imputed.astype(np.float32))
    with split_copy_path.open("w", encoding="utf-8") as handle:
        json.dump(split_obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    output_paths: dict[str, Path] = {
        "imputed_expression.npy": prediction_path,
        "gene_split.json": split_copy_path,
    }
    summary = None
    if not args.skip_adapter_metrics:
        gene_df, summary = compute_stdiff_style_gene_metrics(
            x_true=normalized_st.astype(np.float32),
            x_pred=imputed.astype(np.float32),
            genes=original_genes,
            target_gene_names=test_genes,
        )
        gene_metrics_path = output_dir / "gene_level_metrics_stdiff_style.csv"
        stdiff_result_path = output_dir / "final_result_stdiff_style.csv"
        final_result_path = output_dir / "final_result.csv"
        gene_df.to_csv(gene_metrics_path, index=False)
        pd.DataFrame([summary]).to_csv(stdiff_result_path, index=False)
        pd.DataFrame(
            [
                {
                    "method": "stPlus",
                    "SPCC": summary["SPCC_gene_median_stdiff_style"],
                    "SSIM": summary["SSIM_gene_median_stdiff_style"],
                    "RMSE": summary["RMSE_gene_median_stdiff_style"],
                    "JS": summary["JS_gene_median_stdiff_style"],
                }
            ]
        ).to_csv(final_result_path, index=False)
        output_paths.update(
            {
                "gene_level_metrics_stdiff_style.csv": gene_metrics_path,
                "final_result_stdiff_style.csv": stdiff_result_path,
                "final_result.csv": final_result_path,
            }
        )

    for checkpoint_path in sorted(output_dir.glob("stplus_model-*.pt")):
        output_paths[checkpoint_path.name] = checkpoint_path

    val_model_overlap = np.intersect1d(fit_idx, val_idx)
    test_model_overlap = np.intersect1d(fit_idx, test_idx)
    scope = {
        "adapter_version": ADAPTER_VERSION,
        "adapter_source_sha256": sha256_file(Path(__file__).resolve()),
        "st_normalization_scope": "train_genes",
        "normalization_policy": PROTOCOL_A_POLICY,
        "split_scope": "explicit_complete_train_val_test_partition",
        "model_gene_scope": model_scope,
        "model_fit_uses_inner_train_only": model_scope == STRICT_MODEL_GENE_SCOPE,
        "validation_st_hidden_from_model_fit": int(val_model_overlap.size) == 0,
        "test_st_hidden_from_model_fit": int(test_model_overlap.size) == 0,
        "legacy_non_test_available_only_as_diagnostic": True,
        "diagnostic_only": bool(diagnostic_only),
        "formal_reporting_eligible": not bool(diagnostic_only),
        "adapter_metrics_skipped": bool(args.skip_adapter_metrics),
    }
    counts = {
        "spatial_spot_count": int(x_counts.shape[0]),
        "st_gene_count": int(x_counts.shape[1]),
        "inner_train_gene_count": int(train_idx.size),
        "validation_gene_count": int(val_idx.size),
        "test_gene_count": int(test_idx.size),
        "split_covered_gene_count": int(
            train_idx.size + val_idx.size + test_idx.size
        ),
        "hidden_gene_count": int(predict_idx.size),
        "model_train_gene_count": int(fit_idx.size),
        "model_validation_overlap_count": int(val_model_overlap.size),
        "model_test_overlap_count": int(test_model_overlap.size),
        "scrna_cell_count": int(scrna_df.shape[0]),
        "scrna_gene_count": int(scrna_df.shape[1]),
        "prediction_gene_count": int(prediction.shape[1]),
        "zero_train_library_spot_count": int(
            normalization_audit["zero_train_library_spot_count"]
        ),
    }
    hyperparameters = {
        "scrna_max_cells": args.scrna_max_cells,
        "top_k_requested": args.top_k,
        "top_k_effective": top_k,
        "t_min": args.t_min,
        "n_neighbors": args.n_neighbors,
        "max_epoch_num": args.max_epoch_num,
        "batch_size": args.batch_size,
        "converge_ratio": args.converge_ratio,
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    split_audit = build_split_audit(
        n_genes=len(original_genes),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        input_records=input_records,
        metadata_fields=metadata_fields,
    )
    model_audit = build_model_audit(
        model_gene_scope=model_scope,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        fit_idx=fit_idx,
        predict_idx=predict_idx,
    )
    audit = build_run_audit(
        input_records=input_records,
        output_paths=output_paths,
        prediction_output_key="imputed_expression.npy",
        prediction_shape=imputed.shape,
        prediction_dtype=str(imputed.dtype),
        normalization_audit=normalization_audit,
        scope=scope,
        counts=counts,
        hyperparameters=hyperparameters,
        split_audit=split_audit,
        model_audit=model_audit,
        zero_library_spot_names=[
            str(adata_raw.obs_names[row])
            for row in normalization_audit["zero_train_library_rows"]
        ],
    )
    write_run_audit(output_dir, audit)

    if summary is not None:
        print(pd.DataFrame([summary]).to_string(index=False))
    print(f"[DONE] results saved to: {output_dir}")


if __name__ == "__main__":
    main()
