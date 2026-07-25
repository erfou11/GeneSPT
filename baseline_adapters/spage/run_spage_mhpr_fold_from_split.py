import argparse
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import sparse
from scipy.stats import spearmanr


THIS_DIR = Path(__file__).resolve().parent
SPAGE_ROOT = THIS_DIR / "SpaGE-master"
PROJECT_ROOT = THIS_DIR.parents[1]
MAIN_ROOT = PROJECT_ROOT / "main"
for path in [SPAGE_ROOT, MAIN_ROOT]:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from SpaGE.main import SpaGE
from protocol_a_preprocessing import (
    PROTOCOL_A_POLICY,
    normalize_st_protocol_a,
    validate_gene_splits,
)
from scrna_reference import load_scrna_from_txt
from utils import load_mhpr_from_txt


FORMAL_MODEL_GENE_SCOPE = "train_indices"
DIAGNOSTIC_MODEL_GENE_SCOPE = "non_test"
ADAPTER_VERSION = "protocol_a_v1"


def parse_args(argv=None):
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
        default=FORMAL_MODEL_GENE_SCOPE,
        choices=[FORMAL_MODEL_GENE_SCOPE, DIAGNOSTIC_MODEL_GENE_SCOPE],
        help=(
            "Genes exposed to SpaGE fitting. 'train_indices' is the formal Protocol A "
            "path. 'non_test' exposes validation genes and is diagnostic-only."
        ),
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--n-pv", type=int, default=30)
    p.add_argument("--scrna-max-cells", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-adapter-metrics",
        action="store_true",
        help="Save prediction matrix only; use the central evaluator for final metrics.",
    )
    return p.parse_args(argv)


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_gene_split(path):
    with open(path, "r", encoding="utf-8") as handle:
        split_obj = json.load(handle)
    if not isinstance(split_obj, dict):
        raise ValueError(f"Gene split must be a JSON object: {path}")
    return split_obj


def _load_gene_index_file(path, label):
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"Could not load {label} gene indices from {path}") from error
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{label} gene index path must contain one NumPy array: {path}")
    return values


def load_and_validate_gene_splits(
    *,
    train_gene_idx_path,
    val_gene_idx_path,
    test_gene_idx_path,
    n_genes,
):
    train_raw = _load_gene_index_file(train_gene_idx_path, "train")
    val_raw = _load_gene_index_file(val_gene_idx_path, "validation")
    test_raw = _load_gene_index_file(test_gene_idx_path, "test")
    train_idx, val_idx, test_idx = validate_gene_splits(
        n_genes,
        train_gene_idx=train_raw,
        val_gene_idx=val_raw,
        test_gene_idx=test_raw,
        require_complete_coverage=True,
    )
    # Canonical column order makes model input and semantic split hashes stable.
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _metadata_indices(values, key, n_genes):
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


def _require_metadata_match(key, observed, candidates):
    for scope, expected in candidates:
        if list(observed) == list(expected):
            return scope
    expected_scopes = ", ".join(scope for scope, _ in candidates)
    raise ValueError(
        f"Gene split metadata field {key} disagrees with explicit index paths "
        f"(expected scope: {expected_scopes})"
    )


def validate_gene_split_metadata(split_obj, original_genes, train_idx, val_idx, test_idx):
    if not isinstance(split_obj, dict):
        raise TypeError("split_obj must be a JSON object")
    genes = [str(gene) for gene in original_genes]
    if len(genes) != len(set(genes)):
        raise ValueError("Original ST columns contain duplicate gene names")

    hidden_idx = np.concatenate((val_idx, test_idx))
    index_candidates = {
        "train_gene_idx": [("train", train_idx)],
        "train_idx": [("train", train_idx)],
        "inner_train_gene_idx": [("train", train_idx)],
        "val_gene_idx": [("validation", val_idx)],
        "validation_gene_idx": [("validation", val_idx)],
        "val_idx": [("validation", val_idx)],
        "validation_idx": [("validation", val_idx)],
        "inner_validation_gene_idx": [("validation", val_idx)],
        "final_test_gene_idx": [("test", test_idx)],
        # Older outer-fold JSON files call the complete hidden set "test".
        "test_gene_idx": [("test", test_idx), ("validation_and_test", hidden_idx)],
        "test_idx": [("test", test_idx), ("validation_and_test", hidden_idx)],
    }
    gene_candidates = {
        "train_genes": [("train", [genes[idx] for idx in train_idx])],
        "inner_train_genes": [("train", [genes[idx] for idx in train_idx])],
        "val_genes": [("validation", [genes[idx] for idx in val_idx])],
        "validation_genes": [("validation", [genes[idx] for idx in val_idx])],
        "inner_validation_genes": [("validation", [genes[idx] for idx in val_idx])],
        "final_test_genes": [("test", [genes[idx] for idx in test_idx])],
        "test_genes": [
            ("test", [genes[idx] for idx in test_idx]),
            ("validation_and_test", [genes[idx] for idx in hidden_idx]),
        ],
        "test_target_genes": [
            ("test", [genes[idx] for idx in test_idx]),
            ("validation_and_test", [genes[idx] for idx in hidden_idx]),
        ],
    }

    validated_fields = {}
    for key, candidates in index_candidates.items():
        if key in split_obj:
            observed = _metadata_indices(split_obj[key], key, len(genes))
            validated_fields[key] = _require_metadata_match(key, observed, candidates)
    for key, candidates in gene_candidates.items():
        if key in split_obj:
            observed = [str(value) for value in split_obj[key]]
            if len(observed) != len(set(observed)):
                raise ValueError(f"{key} contains duplicate genes")
            validated_fields[key] = _require_metadata_match(key, observed, candidates)
    return validated_fields


def normalize_st_counts_protocol_a(count_matrix, train_idx, val_idx, test_idx):
    normalized, normalization_audit = normalize_st_protocol_a(
        count_matrix,
        inner_train_gene_idx=train_idx,
        val_gene_idx=val_idx,
        test_gene_idx=test_idx,
        require_complete_coverage=True,
    )
    if not np.isfinite(np.asarray(normalized)).all():
        raise ValueError("Protocol A normalized ST matrix contains non-finite values")
    return normalized, normalization_audit


def normalize_from_counts(count_matrix, template_adata, train_idx, val_idx, test_idx):
    truth_spot_names = [str(name) for name in template_adata.obs_names]
    _validate_spot_axis(truth_spot_names, truth_spot_names, label="ST truth")
    normalized, normalization_audit = normalize_st_counts_protocol_a(
        count_matrix,
        train_idx,
        val_idx,
        test_idx,
    )
    adata = ad.AnnData(X=np.asarray(count_matrix, dtype=np.float32))
    adata.obs_names = template_adata.obs_names.copy()
    adata.var_names = template_adata.var_names.copy()
    adata.obs = template_adata.obs.copy()
    adata.var = template_adata.var.copy()
    adata.obsm["spatial"] = np.asarray(template_adata.obsm["spatial"], dtype=np.float32)
    adata.layers["counts"] = np.asarray(count_matrix, dtype=np.float32).copy()
    adata.X = normalized
    if not np.isfinite(to_dense(adata.X)).all():
        raise ValueError("Normalized ST AnnData contains non-finite values")
    return adata, normalization_audit


def _spot_axis_sha256(spot_names):
    return _canonical_json_sha256([str(name) for name in spot_names])


def _resolve_spot_axis(observed_spot_names, truth_spot_names, *, label):
    observed = [str(name) for name in observed_spot_names]
    truth = [str(name) for name in truth_spot_names]
    if len(observed) != len(set(observed)):
        raise ValueError(f"{label} spot IDs are not unique")
    if len(truth) != len(set(truth)):
        raise ValueError("ST truth spot IDs are not unique")

    resolved = observed
    alias_rule = None
    if set(observed) != set(truth):
        truth_aliases = {
            name.removeprefix("spot_"): name
            for name in truth
            if name.startswith("spot_")
        }
        if (
            len(truth_aliases) == len(truth)
            and set(observed) == set(truth_aliases)
        ):
            resolved = [truth_aliases[name] for name in observed]
            alias_rule = "numeric_id_to_spot_prefix"
        else:
            missing = sorted(set(truth).difference(observed))
            unexpected = sorted(set(observed).difference(truth))
            raise ValueError(
                f"{label} spot IDs do not match ST obs_names; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

    if set(resolved) != set(truth):
        missing = sorted(set(truth).difference(observed))
        unexpected = sorted(set(observed).difference(truth))
        raise ValueError(
            f"{label} spot IDs do not match ST obs_names; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    audit = {
        "spot_axis_sha256": _spot_axis_sha256(truth),
        "observed_spot_axis_sha256": _spot_axis_sha256(observed),
        "resolved_spot_axis_sha256": _spot_axis_sha256(resolved),
        "spot_axis_reordered": resolved != truth,
        "spot_id_alias_normalized": alias_rule is not None,
        "spot_id_alias_rule": alias_rule,
        "spot_count": len(truth),
    }
    return resolved, audit


def _validate_spot_axis(observed_spot_names, truth_spot_names, *, label):
    _, audit = _resolve_spot_axis(observed_spot_names, truth_spot_names, label=label)
    return audit


def align_prediction_to_truth(prediction, truth_spot_names, *, spot_names=None):
    """Validate and explicitly align a prediction DataFrame or matrix to ST truth."""
    truth = [str(name) for name in truth_spot_names]
    if isinstance(prediction, pd.DataFrame):
        observed = [str(name) for name in prediction.index]
        resolved, audit = _resolve_spot_axis(
            observed, truth, label="SpaGE prediction"
        )
        aligned = prediction.copy()
        aligned.index = resolved
        return aligned.loc[truth], audit

    values = np.asarray(prediction)
    if values.ndim != 2:
        raise ValueError("Prediction matrix must be two-dimensional")
    if spot_names is None:
        raise ValueError("Prediction matrix spot IDs are required for alignment")
    observed = [str(name) for name in spot_names]
    resolved, audit = _resolve_spot_axis(
        observed, truth, label="Prediction matrix"
    )
    positions = {name: position for position, name in enumerate(resolved)}
    return values[[positions[name] for name in truth]], audit


def build_split_audit(args, n_genes, train_idx, val_idx, test_idx, input_hashes, metadata_fields):
    canonical_split = {
        "train_gene_idx": [int(value) for value in train_idx],
        "val_gene_idx": [int(value) for value in val_idx],
        "test_gene_idx": [int(value) for value in test_idx],
    }
    split_records = {}
    for scope, indices, arg_name in (
        ("train", train_idx, "train_gene_idx_path"),
        ("validation", val_idx, "val_gene_idx_path"),
        ("test", test_idx, "test_gene_idx_path"),
    ):
        split_records[scope] = {
            "path": str(Path(getattr(args, arg_name)).resolve()),
            "file_sha256": input_hashes[arg_name],
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
        "canonical_sha256": _canonical_json_sha256(canonical_split),
        "metadata_fields_validated": dict(metadata_fields),
        **split_records,
    }


def collect_input_audit(args):
    paths = {
        "locations_path": args.locations_path,
        "st_data": args.st_data,
        "sc_data": args.sc_data,
        "gene_split_json": args.gene_split_json,
        "train_gene_idx_path": args.train_gene_idx_path,
        "val_gene_idx_path": args.val_gene_idx_path,
        "test_gene_idx_path": args.test_gene_idx_path,
    }
    resolved_paths = {key: str(Path(value).resolve()) for key, value in paths.items()}
    hashes = {key: sha256_file(value) for key, value in resolved_paths.items()}
    return resolved_paths, hashes


def model_gene_indices(model_gene_scope, train_idx, val_idx, test_idx):
    if model_gene_scope == FORMAL_MODEL_GENE_SCOPE:
        fit_idx = train_idx
        predict_idx = np.concatenate((val_idx, test_idx))
    elif model_gene_scope == DIAGNOSTIC_MODEL_GENE_SCOPE:
        fit_idx = np.sort(np.concatenate((train_idx, val_idx)))
        predict_idx = test_idx
    else:
        raise ValueError(f"Unsupported model gene scope: {model_gene_scope}")
    if predict_idx.size == 0:
        raise ValueError(f"No genes remain hidden under model gene scope {model_gene_scope}")
    validate_gene_splits(
        int(train_idx.size + val_idx.size + test_idx.size),
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
):
    normalized = np.asarray(normalized_st)
    predicted = np.asarray(predicted_values)
    if normalized.ndim != 2:
        raise ValueError("normalized_st must be a two-dimensional matrix")
    if not np.isfinite(normalized).all():
        raise ValueError("normalized_st contains non-finite values")
    if predicted.shape != (normalized.shape[0], len(predict_idx)):
        raise ValueError(
            "SpaGE prediction shape does not match spots and hidden gene count: "
            f"{predicted.shape} != {(normalized.shape[0], len(predict_idx))}"
        )
    validate_gene_splits(
        normalized.shape[1],
        train_gene_idx=fit_idx,
        val_gene_idx=np.empty(0, dtype=np.int64),
        test_gene_idx=predict_idx,
        require_complete_coverage=True,
    )
    if not np.isfinite(predicted).all():
        raise ValueError("SpaGE predictions contain non-finite values")

    output = np.zeros(normalized.shape, dtype=np.float32)
    output[:, fit_idx] = normalized[:, fit_idx].astype(np.float32, copy=False)
    output[:, predict_idx] = predicted.astype(np.float32, copy=False)
    zero_rows = np.asarray(zero_rows, dtype=np.int64)
    if zero_rows.size:
        output[zero_rows, :] = 0.0
    if not np.isfinite(output).all():
        raise ValueError("Final imputed matrix contains non-finite values")
    return output


def _model_audit(model_gene_scope, train_idx, val_idx, test_idx, fit_idx, predict_idx):
    diagnostic = model_gene_scope == DIAGNOSTIC_MODEL_GENE_SCOPE
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
        "fit_hidden_overlap_count": int(np.intersect1d(fit_idx, predict_idx).size),
    }


def build_run_audit(
    *,
    args,
    input_paths,
    input_hashes,
    split_audit,
    normalization_audit,
    model_audit,
    output_hashes,
    zero_library_spot_names=None,
    spot_axis_audit=None,
    finite_audit=None,
):
    diagnostic = args.model_gene_scope == DIAGNOSTIC_MODEL_GENE_SCOPE
    adapter_source_sha256 = sha256_file(Path(__file__).resolve())
    scope_payload = {
        "adapter_version": ADAPTER_VERSION,
        "adapter_source_sha256": adapter_source_sha256,
        "protocol": "A",
        "normalization_policy": PROTOCOL_A_POLICY,
        "model_gene_scope": args.model_gene_scope,
        "n_pv": int(args.n_pv),
        "scrna_max_cells": int(args.scrna_max_cells),
        "seed": int(args.seed),
        "input_sha256": input_hashes,
        "split_sha256": split_audit["canonical_sha256"],
    }
    scope_sha256 = _canonical_json_sha256(scope_payload)
    diagnostic_reason = (
        "validation ST genes are exposed to model fitting by explicit non_test opt-in"
        if diagnostic
        else None
    )
    zero_library_spot_names = list(zero_library_spot_names or [])
    output_sha256 = {
        name: record["sha256"]
        for name, record in output_hashes.items()
        if isinstance(record, dict) and "sha256" in record
    }
    matrix_record = output_hashes.get("imputed_expression.npy", {})
    spot_axis_audit = dict(spot_axis_audit or {})
    finite_audit = dict(finite_audit or {})
    test_gene_count = int(model_audit["test_index_gene_count"])
    return {
        "adapter": "SpaGE",
        "method": "SpaGE",
        "adapter_version": ADAPTER_VERSION,
        "adapter_source_sha256": adapter_source_sha256,
        "protocol": "A",
        "protocol_role": (
            "explicit_non_test_diagnostic" if diagnostic else "strict_primary_modeA"
        ),
        "eligible_for_strict_primary": not diagnostic,
        "diagnostic_only": diagnostic,
        "formal_protocol_a_run": not diagnostic,
        "run_mode": "diagnostic_non_test" if diagnostic else "formal_protocol_a",
        "diagnostic": {"enabled": diagnostic, "reason": diagnostic_reason},
        "fit_visibility": (
            "explicit_diagnostic_scope"
            if diagnostic
            else "inner_train_genes_only; validation_and_test_hidden"
        ),
        "st_normalization_scope": "train_genes",
        "model_gene_scope": args.model_gene_scope,
        "input_paths": input_paths,
        "input_sha256": input_hashes,
        "split": split_audit,
        "split_sha256": split_audit["canonical_sha256"],
        "normalization_audit": normalization_audit,
        "normalization_denominator_gene_count": int(
            normalization_audit["denominator_gene_count"]
        ),
        "model": model_audit,
        "model_fit_gene_count": int(model_audit["fit_gene_count"]),
        "model_predict_gene_count": int(model_audit["predict_gene_count"]),
        "model_train_gene_count": int(model_audit["fit_gene_count"]),
        "train_index_gene_count": int(model_audit["train_index_gene_count"]),
        "validation_gene_count": int(model_audit["validation_index_gene_count"]),
        "final_test_gene_count": int(model_audit["test_index_gene_count"]),
        "hidden_validation_plus_test_gene_count": int(
            model_audit["validation_index_gene_count"]
            + model_audit["test_index_gene_count"]
        ),
        "test_gene_count": int(model_audit["predict_gene_count"]),
        "model_train_hidden_overlap_count": int(
            model_audit["fit_hidden_overlap_count"]
        ),
        "train_test_overlap_count": int(model_audit["fit_hidden_overlap_count"]),
        "zero_library_spot_count": int(
            normalization_audit.get(
                "zero_train_library_spot_count",
                len(zero_library_spot_names),
            )
        ),
        "zero_library_spots": zero_library_spot_names,
        "spot_axis": spot_axis_audit,
        "spot_axis_sha256": spot_axis_audit.get("spot_axis_sha256"),
        "truth_spot_axis_sha256": spot_axis_audit.get("spot_axis_sha256"),
        "prediction_spot_axis_sha256": spot_axis_audit.get(
            "observed_spot_axis_sha256"
        ),
        "spot_axis_reordered": bool(
            spot_axis_audit.get("spot_axis_reordered", False)
        ),
        "prediction_spot_axis_reordered": bool(
            spot_axis_audit.get("spot_axis_reordered", False)
        ),
        "finite_checks": finite_audit,
        "normalized_finite": bool(finite_audit.get("normalized", False)),
        "predicted_finite": bool(finite_audit.get("predicted", False)),
        "imputed_finite": bool(finite_audit.get("imputed", False)),
        "test_coverage": {
            "complete": True,
            "expected_gene_count": test_gene_count,
            "predicted_gene_count": test_gene_count,
            "missing_gene_count": 0,
            "unexpected_gene_count": 0,
        },
        "test_coverage_complete": True,
        "fallback": {"used": False, "policy": "disabled"},
        "fallback_used": False,
        "fallbacks": [],
        "st_data_sha256": input_hashes.get("st_data"),
        "gene_split_json_sha256": input_hashes.get("gene_split_json"),
        "train_gene_idx_sha256": input_hashes.get("train_gene_idx_path"),
        "val_gene_idx_sha256": input_hashes.get("val_gene_idx_path"),
        "test_gene_idx_sha256": input_hashes.get("test_gene_idx_path"),
        "cache": {
            "enabled": False,
            "read": False,
            "write": False,
            "cross_scope_reuse_allowed": False,
            "scope_sha256": scope_sha256,
            "policy": "cache_disabled",
        },
        "adapter_metrics_skipped": bool(args.skip_adapter_metrics),
        "imputed_matrix_shape": matrix_record.get("shape"),
        "output_files": dict(output_hashes),
        "output_sha256": output_sha256,
        "outputs": dict(output_hashes),
    }


def write_run_audit(output_dir, audit):
    audit_path = Path(output_dir) / "adapter_run_audit.json"
    temporary_path = audit_path.with_suffix(".json.tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(audit_path)


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
    l12 = (2 * mu1 * mu2 + c1) / (mu1 ** 2 + mu2 ** 2 + c1)
    c12 = (2 * sigma1 * sigma2 + c2) / (sigma1 ** 2 + sigma2 ** 2 + c2)
    s12 = (sigma12 + c3) / (sigma1 * sigma2 + c3)
    return l12 * c12 * s12


def scale_max_df(df):
    result = pd.DataFrame(index=df.index)
    for label, content in df.items():
        denom = float(content.max())
        if abs(denom) < 1e-12:
            denom = 1.0
        result[label] = content / denom
    return result


def scale_z_score_df(df):
    result = pd.DataFrame(index=df.index)
    for label, content in df.items():
        z = st.zscore(content)
        result[label] = np.nan_to_num(z, nan=0.0)
    return result


def scale_plus_df(df):
    result = pd.DataFrame(index=df.index)
    for label, content in df.items():
        denom = float(content.sum())
        if abs(denom) < 1e-12:
            denom = 1.0
        result[label] = content / denom
    return result


def compute_stdiff_style_gene_metrics(x_true, x_pred, genes, target_gene_names):
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
            spcc = float(spearmanr(raw_spcc[gene].fillna(1e-20), imp_spcc[gene].fillna(1e-20))[0])
        except Exception:
            spcc = np.nan

        raw_col = raw_ssim[gene].fillna(1e-20)
        imp_col = imp_ssim[gene].fillna(1e-20)
        m_val = max(float(raw_col.max()), float(imp_col.max()))
        try:
            ssim = float(cal_ssim_ref(raw_col.to_numpy().reshape(-1, 1), imp_col.to_numpy().reshape(-1, 1), m_val))
        except Exception:
            ssim = np.nan

        raw_col_js = raw_js[gene].fillna(1e-20)
        imp_col_js = imp_js[gene].fillna(1e-20)
        mid = (raw_col_js + imp_col_js) / 2.0
        try:
            js = float(0.5 * st.entropy(raw_col_js, mid) + 0.5 * st.entropy(imp_col_js, mid))
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
        "SPCC_gene_median_stdiff_style": float(np.nanmedian(gene_df["SPCC"])) if len(gene_df) else np.nan,
        "SSIM_gene_median_stdiff_style": float(np.nanmedian(gene_df["SSIM"])) if len(gene_df) else np.nan,
        "RMSE_gene_median_stdiff_style": float(np.nanmedian(gene_df["RMSE"])) if len(gene_df) else np.nan,
        "JS_gene_median_stdiff_style": float(np.nanmedian(gene_df["JS"])) if len(gene_df) else np.nan,
    }
    return gene_df, summary


def main(argv=None):
    args = parse_args(argv)
    if args.n_pv <= 0:
        raise ValueError("--n-pv must be positive")

    input_paths, input_hashes = collect_input_audit(args)
    split_obj = load_gene_split(args.gene_split_json)
    adata_raw = load_mhpr_from_txt(
        locations_path=args.locations_path,
        counts_path=args.st_data,
        normalize=False,
        store_raw_layer=True,
    )
    x_counts = to_dense(
        adata_raw.layers["counts"] if "counts" in adata_raw.layers else adata_raw.X
    ).astype(np.float32)
    all_genes = list(map(str, adata_raw.var_names))
    truth_spot_names = [str(name) for name in adata_raw.obs_names]
    _validate_spot_axis(truth_spot_names, truth_spot_names, label="ST truth")
    if x_counts.ndim != 2 or x_counts.shape[1] != len(all_genes):
        raise ValueError(
            "ST count matrix shape does not match the original ST gene columns: "
            f"{x_counts.shape} versus {len(all_genes)} genes"
        )
    if len(all_genes) != len(set(all_genes)):
        raise ValueError("Original ST columns contain duplicate gene names")

    train_idx, val_idx, test_idx = load_and_validate_gene_splits(
        train_gene_idx_path=args.train_gene_idx_path,
        val_gene_idx_path=args.val_gene_idx_path,
        test_gene_idx_path=args.test_gene_idx_path,
        n_genes=len(all_genes),
    )
    metadata_fields = validate_gene_split_metadata(
        split_obj,
        all_genes,
        train_idx,
        val_idx,
        test_idx,
    )
    adata_gt, normalization_audit = normalize_from_counts(
        x_counts,
        adata_raw,
        train_idx,
        val_idx,
        test_idx,
    )

    adata_scrna = load_scrna_from_txt(
        counts_path=args.sc_data,
        normalize=True,
        store_raw_layer=True,
        target_genes=all_genes,
        max_cells=(None if args.scrna_max_cells <= 0 else args.scrna_max_cells),
        seed=args.seed,
        maxabs_scale=False,
    )
    scrna_genes = list(map(str, adata_scrna.var_names))
    if len(scrna_genes) != len(set(scrna_genes)):
        raise ValueError("scRNA reference contains duplicate gene names")
    scrna_gene_set = set(scrna_genes)
    missing_scrna_genes = [gene for gene in all_genes if gene not in scrna_gene_set]
    if missing_scrna_genes:
        raise ValueError(
            "scRNA reference is missing genes required for a full Protocol A output: "
            f"{missing_scrna_genes[:10]}"
        )
    adata_scrna = adata_scrna[:, all_genes].copy()

    fit_idx, predict_idx = model_gene_indices(
        args.model_gene_scope,
        train_idx,
        val_idx,
        test_idx,
    )
    fit_genes = [all_genes[idx] for idx in fit_idx]
    predict_genes = [all_genes[idx] for idx in predict_idx]
    n_pv = min(int(args.n_pv), len(fit_genes))
    print(
        f"[SpaGE] mode = {args.model_gene_scope}, fit genes = {len(fit_genes)}, "
        f"hidden genes = {len(predict_genes)}, n_pv = {n_pv}"
    )

    spatial_df = pd.DataFrame(
        to_dense(adata_gt.X)[:, fit_idx].astype(np.float32),
        index=list(map(str, adata_gt.obs_names)),
        columns=fit_genes,
    )
    scrna_df = pd.DataFrame(
        to_dense(adata_scrna.X).astype(np.float32),
        index=list(map(str, adata_scrna.obs_names)),
        columns=list(map(str, adata_scrna.var_names)),
    )

    pred_df = SpaGE(
        Spatial_data=spatial_df,
        RNA_data=scrna_df,
        n_pv=n_pv,
        genes_to_predict=np.asarray(predict_genes, dtype=object),
    )
    if not isinstance(pred_df, pd.DataFrame):
        raise TypeError("SpaGE must return predictions as a pandas DataFrame")
    if pred_df.shape[0] != spatial_df.shape[0]:
        raise ValueError(
            f"SpaGE returned {pred_df.shape[0]} rows for {spatial_df.shape[0]} ST spots"
        )
    pred_df.columns = [str(column) for column in pred_df.columns]
    if len(pred_df.columns) != len(set(pred_df.columns)):
        raise ValueError("SpaGE returned duplicate prediction columns")
    if set(pred_df.columns) != set(predict_genes):
        missing = sorted(set(predict_genes).difference(pred_df.columns))
        unexpected = sorted(set(pred_df.columns).difference(predict_genes))
        raise ValueError(
            "SpaGE prediction columns do not match the hidden gene set; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    pred_df = pred_df.loc[:, predict_genes]
    pred_df, spot_axis_audit = align_prediction_to_truth(
        pred_df,
        truth_spot_names,
    )
    predicted_values = pred_df.to_numpy(dtype=np.float32)
    if not np.isfinite(predicted_values).all():
        raise ValueError("SpaGE predictions contain non-finite values")

    imputed = assemble_full_prediction_matrix(
        to_dense(adata_gt.X),
        fit_idx=fit_idx,
        predict_idx=predict_idx,
        predicted_values=predicted_values,
        zero_rows=normalization_audit["zero_train_library_rows"],
    )
    if not np.isfinite(imputed).all():
        raise ValueError("Final imputed matrix contains non-finite values")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "imputed_expression.npy"
    saved_split_path = output_dir / "gene_split.json"
    if not np.isfinite(to_dense(adata_gt.X)).all():
        raise ValueError("Normalized ST matrix is non-finite before saving outputs")
    if not np.isfinite(predicted_values).all():
        raise ValueError("Predicted matrix is non-finite before saving outputs")
    if not np.isfinite(imputed).all():
        raise ValueError("Final imputed matrix is non-finite before saving outputs")
    np.save(prediction_path, imputed)
    with open(saved_split_path, "w", encoding="utf-8") as handle:
        json.dump(split_obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    split_audit = build_split_audit(
        args,
        len(all_genes),
        train_idx,
        val_idx,
        test_idx,
        input_hashes,
        metadata_fields,
    )
    model_audit = _model_audit(
        args.model_gene_scope,
        train_idx,
        val_idx,
        test_idx,
        fit_idx,
        predict_idx,
    )
    model_audit["requested_n_pv"] = int(args.n_pv)
    model_audit["effective_n_pv"] = int(n_pv)
    output_hashes = {
        "imputed_expression.npy": {
            "sha256": sha256_file(prediction_path),
            "shape": [int(value) for value in imputed.shape],
            "dtype": str(imputed.dtype),
        },
        "gene_split.json": {"sha256": sha256_file(saved_split_path)},
    }
    audit = build_run_audit(
        args=args,
        input_paths=input_paths,
        input_hashes=input_hashes,
        split_audit=split_audit,
        normalization_audit=normalization_audit,
        model_audit=model_audit,
        output_hashes=output_hashes,
        spot_axis_audit=spot_axis_audit,
        finite_audit={
            "normalized": True,
            "predicted": True,
            "imputed": True,
        },
        zero_library_spot_names=[
            str(adata_raw.obs_names[row])
            for row in normalization_audit["zero_train_library_rows"]
        ],
    )
    write_run_audit(output_dir, audit)

    if args.skip_adapter_metrics:
        print(f"[DONE] prediction matrix saved to: {prediction_path}")
        return

    test_genes = [all_genes[idx] for idx in test_idx]
    gene_df, summary = compute_stdiff_style_gene_metrics(
        x_true=to_dense(adata_gt.X).astype(np.float32),
        x_pred=imputed,
        genes=all_genes,
        target_gene_names=test_genes,
    )
    gene_df.to_csv(output_dir / "gene_level_metrics_stdiff_style.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "final_result_stdiff_style.csv", index=False)

    pd.DataFrame(
        [
            {
                "method": "SpaGE",
                "SPCC": summary["SPCC_gene_median_stdiff_style"],
                "SSIM": summary["SSIM_gene_median_stdiff_style"],
                "RMSE": summary["RMSE_gene_median_stdiff_style"],
                "JS": summary["JS_gene_median_stdiff_style"],
            }
        ]
    ).to_csv(output_dir / "final_result.csv", index=False)

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"[DONE] results saved to: {output_dir}")


if __name__ == "__main__":
    main()
