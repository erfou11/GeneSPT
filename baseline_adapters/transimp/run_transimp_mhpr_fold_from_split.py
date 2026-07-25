import os
import sys
import json
import argparse
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as st
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.neighbors import kneighbors_graph
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = PROJECT_ROOT / "main"
TRANSPA_ROOT = Path(__file__).resolve().parent
for path in [PROJECT_ROOT, MAIN_ROOT, TRANSPA_ROOT]:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from transpa.util import expTransImp
from utils import load_mhpr_from_txt
from protocol_a_preprocessing import normalize_st_protocol_a, validate_gene_splits


METHOD_DISPLAY_NAME = "TransImp"
IMPLEMENTATION_PROVENANCE = {
    "repository": "tranSpa",
    "python_package": "transpa",
    "entrypoint": "transpa.util.expTransImp",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--locations-path", type=str, required=True)
    p.add_argument("--st-data", type=str, required=True)
    p.add_argument("--sc-data", type=str, required=True)
    p.add_argument("--gene-split-json", type=str, required=True)
    p.add_argument(
        "--train-gene-idx-path",
        "--train-mask-path",
        dest="train_gene_idx_path",
        type=str,
        default=None,
    )
    p.add_argument(
        "--val-gene-idx-path",
        "--val-mask-path",
        dest="val_gene_idx_path",
        type=str,
        default=None,
    )
    p.add_argument(
        "--test-gene-idx-path",
        "--test-mask-path",
        dest="test_gene_idx_path",
        type=str,
        default=None,
    )
    p.add_argument(
        "--st-normalization-scope",
        type=str,
        default="train_genes",
        choices=["all_genes", "train_genes"],
        help="all_genes is retained only for an explicit legacy diagnostic run.",
    )
    p.add_argument(
        "--model-gene-scope",
        type=str,
        default="train_indices",
        choices=["non_test", "train_indices"],
        help="non_test is retained only for an explicit diagnostic run.",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--scrna-max-cells", type=int, default=5000)
    p.add_argument("--mapping-mode", type=str, default="lowrank")
    p.add_argument("--signature-mode", type=str, default="cell")
    p.add_argument("--mapping-lowdim", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--wt-spa", type=float, default=0.1)
    p.add_argument("--wt-js", type=float, default=0.0)
    p.add_argument("--knn", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-adapter-metrics", action="store_true")
    return p.parse_args()


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


def _coerce_gene_indices(values, label, n_genes, require_nonempty=True):
    indices = np.asarray(values)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a one-dimensional integer array")
    indices = indices.astype(np.int64, copy=True)
    if require_nonempty and indices.size == 0:
        raise ValueError(f"{label} must not be empty")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{label} contains duplicate indices")
    if indices.size and (
        int(indices.min()) < 0 or int(indices.max()) >= int(n_genes)
    ):
        raise ValueError(f"{label} contains indices outside [0, {n_genes})")
    return indices


def load_gene_indices(path, n_genes, label, require_nonempty=True):
    indices = _coerce_gene_indices(
        np.load(path, allow_pickle=False),
        f"{label} gene indices",
        n_genes,
        require_nonempty=require_nonempty,
    )
    return np.sort(indices)


def load_protocol_a_gene_indices(
    train_path,
    val_path,
    test_path,
    n_genes,
):
    """Load the three explicit masks and require a complete gene partition."""

    train_idx = load_gene_indices(train_path, n_genes, "train")
    val_idx = load_gene_indices(val_path, n_genes, "validation")
    test_idx = load_gene_indices(test_path, n_genes, "test")
    return validate_gene_splits(
        n_genes,
        train_gene_idx=train_idx,
        val_gene_idx=val_idx,
        test_gene_idx=test_idx,
        require_complete_coverage=True,
    )


def protocol_role(args):
    strict = (
        args.st_normalization_scope == "train_genes"
        and args.model_gene_scope == "train_indices"
    )
    return strict, (
        "strict_primary_modeA" if strict else "explicit_legacy_diagnostic"
    )


def validate_run_mode(args):
    """Fail closed unless all inputs needed by the selected mode are explicit."""

    strict_mode, role = protocol_role(args)
    if strict_mode:
        missing = [
            option
            for option, value in (
                ("--train-gene-idx-path", args.train_gene_idx_path),
                ("--val-gene-idx-path", args.val_gene_idx_path),
                ("--test-gene-idx-path", args.test_gene_idx_path),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Strict Protocol A requires explicit train/validation/test gene "
                f"index paths; missing {', '.join(missing)}"
            )
    elif (
        args.st_normalization_scope == "train_genes"
        or args.model_gene_scope == "train_indices"
    ) and args.train_gene_idx_path is None:
        raise ValueError(
            "--train-gene-idx-path is required when a diagnostic run uses "
            "train-gene normalization or train-index model fitting"
        )
    return strict_mode, role


def normalize_from_counts(count_matrix, template_adata, denominator_idx=None):
    """Legacy diagnostic normalization retained for explicit non-primary runs."""

    adata = ad.AnnData(X=np.asarray(count_matrix, dtype=np.float32))
    adata.obs_names = template_adata.obs_names.copy()
    adata.var_names = template_adata.var_names.copy()
    adata.obs = template_adata.obs.copy()
    adata.var = template_adata.var.copy()
    adata.obsm["spatial"] = np.asarray(template_adata.obsm["spatial"], dtype=np.float32)
    adata.layers["counts"] = np.asarray(count_matrix, dtype=np.float32).copy()
    if denominator_idx is None:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        indices = np.asarray(denominator_idx, dtype=np.int64)
        counts = np.asarray(count_matrix, dtype=np.float64)
        library_sizes = counts[:, indices].sum(axis=1, dtype=np.float64)
        normalized = np.zeros(counts.shape, dtype=np.float64)
        nonzero = library_sizes > 0.0
        if np.any(nonzero):
            normalized[nonzero] = np.log1p(
                counts[nonzero] / library_sizes[nonzero, None] * 1e4
            )
        adata.X = normalized.astype(np.float32)
    return adata


def _consistent_gene_list(split_obj, keys, label):
    found = []
    for key in keys:
        if key not in split_obj:
            continue
        values = [str(value) for value in split_obj[key]]
        if len(values) != len(set(values)):
            raise ValueError(f"{key} contains duplicate genes")
        found.append((key, values))
    if not found:
        return None
    reference_key, reference = found[0]
    for key, values in found[1:]:
        if values != reference:
            raise ValueError(f"Split gene lists {reference_key} and {key} disagree")
    return reference


def load_gene_split(path):
    with open(path, "r", encoding="utf-8") as handle:
        split_obj = json.load(handle)
    if not isinstance(split_obj, dict):
        raise ValueError(f"Gene split must be a JSON object: {path}")
    hidden_genes = _consistent_gene_list(
        split_obj,
        ["test_genes", "test_target_genes"],
        "hidden",
    )
    if hidden_genes is None:
        raise KeyError(
            f"Could not find hidden gene list in {path}. Keys: {list(split_obj.keys())}"
        )
    return split_obj, hidden_genes


def load_test_genes(path: str):
    return load_gene_split(path)[1]


def _indices_from_split_genes(split_obj, keys, gene_to_idx):
    genes = _consistent_gene_list(split_obj, keys, keys[0])
    if genes is None:
        return None
    missing = [gene for gene in genes if gene not in gene_to_idx]
    if missing:
        raise ValueError(f"Split genes are absent from ST columns: {missing[:10]}")
    return np.asarray([gene_to_idx[gene] for gene in genes], dtype=np.int64)


def _indices_from_split_indices(split_obj, keys, n_genes):
    found = []
    for key in keys:
        if key in split_obj:
            found.append(
                (
                    key,
                    _coerce_gene_indices(
                        split_obj[key], key, n_genes, require_nonempty=True
                    ),
                )
            )
    if not found:
        return None
    reference_key, reference = found[0]
    for key, values in found[1:]:
        if not np.array_equal(np.sort(values), np.sort(reference)):
            raise ValueError(f"Split index lists {reference_key} and {key} disagree")
    return reference


def _require_same_indices(actual, expected, label):
    if actual is not None and not np.array_equal(
        np.sort(actual), np.sort(np.asarray(expected, dtype=np.int64))
    ):
        raise ValueError(f"Split JSON {label} does not match the explicit mask path")


def validate_protocol_a_split_json(
    split_obj,
    original_genes,
    train_idx,
    val_idx,
    test_idx,
):
    """Cross-check JSON names/indices against the explicit Protocol A masks."""

    genes = [str(gene) for gene in original_genes]
    if len(genes) != len(set(genes)):
        raise ValueError("Original ST columns contain duplicate gene names")
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    n_genes = len(genes)
    hidden_idx = np.concatenate((val_idx, test_idx))

    definitions = (
        (
            "train genes",
            ["train_gene_idx", "train_idx", "inner_train_gene_idx"],
            ["train_genes", "inner_train_genes"],
            train_idx,
        ),
        (
            "validation genes",
            [
                "val_gene_idx",
                "validation_gene_idx",
                "val_idx",
                "validation_idx",
                "inner_validation_gene_idx",
            ],
            ["val_genes", "validation_genes", "inner_validation_genes"],
            val_idx,
        ),
        (
            "final test genes",
            ["final_test_gene_idx", "final_test_idx"],
            ["final_test_genes"],
            test_idx,
        ),
        (
            "hidden validation-plus-test genes",
            ["test_gene_idx", "test_idx"],
            ["test_genes", "test_target_genes"],
            hidden_idx,
        ),
    )
    for label, index_keys, gene_keys, expected in definitions:
        declared_indices = _indices_from_split_indices(
            split_obj, index_keys, n_genes
        )
        declared_genes = _indices_from_split_genes(
            split_obj, gene_keys, gene_to_idx
        )
        _require_same_indices(declared_indices, expected, label)
        _require_same_indices(declared_genes, expected, label)

    declared_hidden = _indices_from_split_genes(
        split_obj, ["test_genes", "test_target_genes"], gene_to_idx
    )
    if declared_hidden is None:
        raise ValueError("Split JSON must explicitly declare all hidden genes")
    return hidden_idx


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
    hashes = {
        key: (sha256_file(path) if path is not None else None)
        for key, path in paths.items()
    }
    return paths, hashes


def collect_output_audit(output_dir, filenames):
    records = {}
    for filename in filenames:
        path = Path(output_dir) / filename
        records[filename] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return records


def write_run_audit(output_dir, audit):
    with open(
        Path(output_dir) / "adapter_run_audit.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)


def load_scrna_full_downsampled(path, target_genes=None, max_cells=5000, seed=42, target_sum=1e4):
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

    if max_cells is not None and max_cells > 0 and len(df) > max_cells:
        rng = np.random.default_rng(seed)
        keep_idx = np.sort(rng.choice(len(df), size=max_cells, replace=False))
        df = df.iloc[keep_idx].copy()

    x = df.to_numpy(dtype=np.float32)
    lib = np.clip(x.sum(axis=1, keepdims=True), 1.0, None)
    x = x / lib * float(target_sum)
    x = np.log1p(x)
    return pd.DataFrame(x, index=df.index.astype(str), columns=df.columns.astype(str))


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


def build_spatial_adj(coords, knn=8):
    graph = kneighbors_graph(
        np.asarray(coords, dtype=np.float32),
        n_neighbors=int(knn),
        mode="connectivity",
        include_self=False,
    )
    graph = graph.maximum(graph.T).tocoo()
    return graph


def coerce_transimp_prediction(prediction, row_names, hidden_genes):
    """Bind expTransImp's positional output to the input ST row/gene axes."""

    expected_rows = [str(value) for value in row_names]
    expected_genes = [str(value) for value in hidden_genes]
    if len(expected_genes) != len(set(expected_genes)):
        raise ValueError("TransImp hidden gene list contains duplicate genes")

    if isinstance(prediction, pd.DataFrame):
        if list(map(str, prediction.index)) != expected_rows:
            raise ValueError(
                "TransImp prediction row labels do not exactly match input ST order"
            )
        if list(map(str, prediction.columns)) != expected_genes:
            raise ValueError(
                "TransImp prediction columns do not exactly match hidden gene order"
            )
        values = prediction.to_numpy(dtype=np.float32, copy=True)
        row_axis_contract = "explicit_prediction_labels_match_input_ST"
    else:
        values = np.asarray(prediction, dtype=np.float32)
        row_axis_contract = (
            "unlabelled_expTransImp_array_rows_inherit_input_ST_df_tgt_order"
        )

    expected_shape = (len(expected_rows), len(expected_genes))
    if values.shape != expected_shape:
        raise ValueError(
            f"TransImp returned prediction shape {values.shape}, expected {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("TransImp returned non-finite predictions")

    bound = pd.DataFrame(values, index=expected_rows, columns=expected_genes)
    return bound, {
        "row_axis_contract": row_axis_contract,
        "row_count": int(len(expected_rows)),
        "row_order_sha256": hashlib.sha256(
            json.dumps(expected_rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "prediction_gene_count": int(len(expected_genes)),
        "prediction_gene_order": expected_genes,
        "prediction_gene_order_sha256": hashlib.sha256(
            json.dumps(expected_genes, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "prediction_source": "expTransImp",
    }


def assemble_imputed_matrix(base_matrix, prediction_df, hidden_idx, zero_library_idx):
    """Replace every hidden column from predictions, with no truth fallback."""

    base = np.asarray(base_matrix, dtype=np.float32)
    if base.ndim != 2 or not np.isfinite(base).all():
        raise ValueError("Input matrix for imputation must be a finite 2-D matrix")
    prediction = prediction_df.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(prediction).all():
        raise ValueError("non-finite predictions before imputation")
    hidden_idx = np.asarray(hidden_idx, dtype=np.int64)
    if prediction.shape[1] != hidden_idx.size:
        raise ValueError("Prediction columns do not cover every hidden gene")

    imputed = base.copy()
    imputed[:, hidden_idx] = prediction
    if len(zero_library_idx):
        imputed[np.asarray(zero_library_idx, dtype=np.int64), :] = 0.0
    if not np.isfinite(imputed).all():
        raise ValueError("Final imputed matrix is non-finite before saving")

    covered = {
        str(gene): {
            "source": "expTransImp",
            "output_column_index": int(index),
            "truth_fallback": False,
        }
        for gene, index in zip(prediction_df.columns, hidden_idx)
    }
    return imputed, covered


def main():
    args = parse_args()
    strict_mode, role = validate_run_mode(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_obj, declared_hidden_genes = load_gene_split(args.gene_split_json)

    adata_raw = load_mhpr_from_txt(
        locations_path=args.locations_path,
        counts_path=args.st_data,
        normalize=False,
        store_raw_layer=True,
    )
    x_counts = to_dense(adata_raw.layers["counts"] if "counts" in adata_raw.layers else adata_raw.X).astype(np.float32)
    original_genes = list(map(str, adata_raw.var_names))
    if len(original_genes) != len(set(original_genes)):
        raise ValueError("Original ST columns contain duplicate gene names")
    gene_to_idx = {gene: idx for idx, gene in enumerate(original_genes)}

    train_idx = None
    val_idx = None
    final_test_idx = None
    normalization_audit = None
    if strict_mode:
        train_idx, val_idx, final_test_idx = load_protocol_a_gene_indices(
            args.train_gene_idx_path,
            args.val_gene_idx_path,
            args.test_gene_idx_path,
            len(original_genes),
        )
        hidden_idx = validate_protocol_a_split_json(
            split_obj,
            original_genes,
            train_idx,
            val_idx,
            final_test_idx,
        )
        normalized, normalization_audit = normalize_st_protocol_a(
            x_counts,
            inner_train_gene_idx=train_idx,
            val_gene_idx=val_idx,
            test_gene_idx=final_test_idx,
            require_complete_coverage=True,
        )
        adata_gt = adata_raw.copy()
        adata_gt.X = normalized.astype(np.float32)
        zero_library_idx = np.asarray(
            normalization_audit["zero_train_library_rows"], dtype=np.int64
        )
        denominator_idx = train_idx
        train_genes = [original_genes[idx] for idx in train_idx]
        hidden_genes = [original_genes[idx] for idx in hidden_idx]
        metric_target_genes = [original_genes[idx] for idx in final_test_idx]
    else:
        missing_hidden = [
            gene for gene in declared_hidden_genes if gene not in gene_to_idx
        ]
        if missing_hidden:
            raise ValueError(
                f"Diagnostic split genes are absent from ST columns: {missing_hidden[:10]}"
            )
        hidden_genes = list(declared_hidden_genes)
        hidden_idx = np.asarray(
            [gene_to_idx[gene] for gene in hidden_genes], dtype=np.int64
        )
        if args.train_gene_idx_path is not None:
            train_idx = load_gene_indices(
                args.train_gene_idx_path,
                len(original_genes),
                "train",
            )
        denominator_idx = (
            train_idx if args.st_normalization_scope == "train_genes" else None
        )
        adata_gt = normalize_from_counts(
            x_counts,
            adata_raw,
            denominator_idx=denominator_idx,
        )
        denominator_for_zero = (
            np.arange(len(original_genes), dtype=np.int64)
            if denominator_idx is None
            else denominator_idx
        )
        library_sizes = x_counts[:, denominator_for_zero].sum(
            axis=1, dtype=np.float64
        )
        zero_library_idx = np.flatnonzero(library_sizes == 0.0)
        if args.model_gene_scope == "train_indices":
            train_genes = [original_genes[idx] for idx in train_idx]
        else:
            hidden_gene_set = set(hidden_genes)
            train_genes = [
                gene for gene in original_genes if gene not in hidden_gene_set
            ]
        metric_target_genes = hidden_genes

    if not train_genes:
        raise ValueError("No model training genes remain")
    overlap = set(train_genes).intersection(hidden_genes)
    if overlap:
        raise ValueError(
            f"Model training genes overlap hidden validation/test genes: {sorted(overlap)[:10]}"
        )

    spatial_df = pd.DataFrame(
        to_dense(adata_gt.X)[:, [gene_to_idx[gene] for gene in train_genes]].astype(
            np.float32
        ),
        index=list(map(str, adata_gt.obs_names)),
        columns=train_genes,
    )
    scrna_df = load_scrna_full_downsampled(
        path=args.sc_data,
        target_genes=list(map(str, adata_raw.var_names)),
        max_cells=(None if args.scrna_max_cells is not None and args.scrna_max_cells <= 0 else args.scrna_max_cells),
        seed=args.seed,
    )

    if scrna_df.columns.has_duplicates:
        raise ValueError("scRNA reference contains duplicate gene columns")
    required_reference_genes = train_genes + hidden_genes
    reference_gene_set = set(map(str, scrna_df.columns))
    missing_reference = [
        gene for gene in required_reference_genes if gene not in reference_gene_set
    ]
    if missing_reference:
        raise ValueError(
            "scRNA reference is missing required train/hidden genes: "
            f"{missing_reference[:10]}"
        )
    scrna_df = scrna_df.loc[:, required_reference_genes]
    spatial_adj = build_spatial_adj(adata_gt.obsm["spatial"], knn=args.knn)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    print(
        f"[{METHOD_DISPLAY_NAME}] train genes = {len(train_genes)}, "
        f"hidden validation+test genes = {len(hidden_genes)}, "
        f"n_epochs = {args.n_epochs}, device = {device}"
    )
    pred = expTransImp(
        df_ref=scrna_df,
        df_tgt=spatial_df,
        train_gene=train_genes,
        test_gene=hidden_genes,
        signature_mode=args.signature_mode,
        mapping_mode=args.mapping_mode,
        mapping_lowdim=args.mapping_lowdim,
        wt_spa=args.wt_spa,
        wt_js=args.wt_js,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        spa_adj=spatial_adj,
        device=device,
        seed=args.seed,
    )

    prediction_df, prediction_audit = coerce_transimp_prediction(
        pred, spatial_df.index, hidden_genes
    )
    final_test_genes = (
        [original_genes[idx] for idx in final_test_idx]
        if final_test_idx is not None
        else []
    )
    if final_test_genes and not set(final_test_genes).issubset(
        prediction_df.columns
    ):
        raise ValueError("Every final test gene must be covered by prediction")
    imputed, prediction_coverage = assemble_imputed_matrix(
        to_dense(adata_gt.X), prediction_df, hidden_idx, zero_library_idx
    )

    pd.DataFrame(imputed, index=adata_gt.obs_names, columns=adata_gt.var_names).to_csv(output_dir / "imputed_expression.csv")
    np.save(output_dir / "imputed_expression.npy", imputed)
    with open(output_dir / "gene_split.json", "w", encoding="utf-8") as f:
        json.dump(split_obj, f, ensure_ascii=False, indent=2)

    output_filenames = [
        "imputed_expression.npy",
        "imputed_expression.csv",
        "gene_split.json",
    ]
    summary = None
    if not args.skip_adapter_metrics:
        gene_df, summary = compute_stdiff_style_gene_metrics(
            x_true=to_dense(adata_gt.X).astype(np.float32),
            x_pred=imputed,
            genes=original_genes,
            target_gene_names=metric_target_genes,
        )
        gene_df.to_csv(
            output_dir / "gene_level_metrics_stdiff_style.csv", index=False
        )
        pd.DataFrame([summary]).to_csv(
            output_dir / "final_result_stdiff_style.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "method": METHOD_DISPLAY_NAME,
                    "SPCC": summary["SPCC_gene_median_stdiff_style"],
                    "SSIM": summary["SSIM_gene_median_stdiff_style"],
                    "RMSE": summary["RMSE_gene_median_stdiff_style"],
                    "JS": summary["JS_gene_median_stdiff_style"],
                }
            ]
        ).to_csv(output_dir / "final_result.csv", index=False)
        output_filenames.extend(
            [
                "gene_level_metrics_stdiff_style.csv",
                "final_result_stdiff_style.csv",
                "final_result.csv",
            ]
        )

    input_paths, input_hashes = collect_input_audit(args)
    output_files = collect_output_audit(output_dir, output_filenames)
    audit = {
        "adapter": METHOD_DISPLAY_NAME,
        "method": METHOD_DISPLAY_NAME,
        "method_display": METHOD_DISPLAY_NAME,
        "package_provenance": "tranSpa/transpa",
        "implementation_provenance": IMPLEMENTATION_PROVENANCE.copy(),
        "adapter_source_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": "A" if strict_mode else "legacy_diagnostic",
        "protocol_role": role,
        "eligible_for_strict_primary": bool(strict_mode),
        "diagnostic_only": not strict_mode,
        "fit_visibility": (
            "inner_train_genes_only; validation_and_test_hidden"
            if strict_mode
            else "explicit_diagnostic_scope"
        ),
        "st_normalization_scope": args.st_normalization_scope,
        "model_gene_scope": args.model_gene_scope,
        "normalization_denominator_gene_count": int(
            len(original_genes) if denominator_idx is None else len(denominator_idx)
        ),
        "model_train_gene_count": int(len(train_genes)),
        "train_index_gene_count": (
            int(len(train_idx)) if train_idx is not None else None
        ),
        "hidden_validation_plus_test_gene_count": int(len(hidden_idx)),
        "hidden_gene_order": hidden_genes,
        "hidden_gene_count_matches_prediction": bool(
            len(hidden_genes) == prediction_audit["prediction_gene_count"]
        ),
        "prediction_audit": prediction_audit,
        "prediction_coverage": prediction_coverage,
        "final_test_gene_order": final_test_genes,
        "final_test_gene_count_matches_coverage": bool(
            all(
                gene in prediction_coverage
                for gene in final_test_genes
            )
        ),
        "truth_fallback": False,
        "row_axis_contract": prediction_audit["row_axis_contract"],
        "fallback_policy": "none; fail closed on missing or misordered prediction axes",
        "validation_gene_count": int(len(val_idx)) if val_idx is not None else None,
        "final_test_gene_count": (
            int(len(final_test_idx)) if final_test_idx is not None else None
        ),
        "model_train_hidden_overlap_count": 0,
        "zero_library_spot_count": int(len(zero_library_idx)),
        "zero_library_spots": [
            str(adata_raw.obs_names[idx]) for idx in zero_library_idx
        ],
        "normalization_audit": normalization_audit,
        "input_paths": input_paths,
        "input_sha256": input_hashes,
        "st_data_sha256": input_hashes["st_data"],
        "gene_split_json_sha256": input_hashes["gene_split_json"],
        "train_gene_idx_sha256": input_hashes["train_gene_idx_path"],
        "val_gene_idx_sha256": input_hashes["val_gene_idx_path"],
        "test_gene_idx_sha256": input_hashes["test_gene_idx_path"],
        "output_files": output_files,
        "output_sha256": {
            filename: record["sha256"]
            for filename, record in output_files.items()
        },
        "imputed_matrix_shape": [int(value) for value in imputed.shape],
        "imputed_matrix_dtype": str(imputed.dtype),
        "adapter_metrics_skipped": bool(args.skip_adapter_metrics),
    }
    write_run_audit(output_dir, audit)

    if summary is not None:
        print(pd.DataFrame([summary]).to_string(index=False))
    print(f"[DONE] results saved to: {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
