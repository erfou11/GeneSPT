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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = PROJECT_ROOT / "main"
for path in [PROJECT_ROOT, MAIN_ROOT]:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

import baseline.tangram.mapping_utils as tg_map
import baseline.tangram.utils as tg_utils
from utils import load_mhpr_from_txt
from scrna_reference import load_scrna_from_txt, align_scrna_to_st
from protocol_a_preprocessing import normalize_st_protocol_a, validate_gene_splits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--locations-path", type=str, required=True)
    p.add_argument("--st-data", type=str, required=True)
    p.add_argument("--sc-data", type=str, required=True)
    p.add_argument("--gene-split-json", type=str, required=True)
    p.add_argument("--train-gene-idx-path", type=str, default=None)
    p.add_argument("--val-gene-idx-path", type=str, default=None)
    p.add_argument("--test-gene-idx-path", type=str, default=None)
    p.add_argument(
        "--st-normalization-scope",
        type=str,
        default="train_genes",
        choices=["all_genes", "train_genes"],
    )
    p.add_argument(
        "--model-gene-scope",
        type=str,
        default="train_indices",
        choices=["non_test", "train_indices"],
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-epochs", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--scrna-max-cells", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="clusters", choices=["cells", "clusters"])
    p.add_argument("--density-prior", type=str, default="rna_count_based", choices=["rna_count_based", "uniform"])
    p.add_argument("--cluster-label", type=str, default="leiden")
    p.add_argument("--leiden-resolution", type=float, default=0.5)
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


def insert_complete_test_predictions(
    base_matrix,
    prediction_frame,
    expected_spot_names,
    all_gene_names,
    test_gene_names,
    analytic_zero_genes=(),
):
    """Insert every held-out prediction after fail-closed row/column alignment."""
    output = np.asarray(base_matrix, dtype=np.float32).copy()
    expected_spots = [str(value) for value in expected_spot_names]
    expected_genes = [str(value) for value in all_gene_names]
    requested_test = [str(value).lower() for value in test_gene_names]
    allowed_analytic_zeros = {str(value).lower() for value in analytic_zero_genes}

    if output.shape != (len(expected_spots), len(expected_genes)):
        raise ValueError(
            "Tangram base matrix shape does not match the declared spot/gene axes"
        )
    if len(set(expected_spots)) != len(expected_spots):
        raise ValueError("Expected spatial spot identifiers are not unique")
    if prediction_frame.index.has_duplicates:
        raise ValueError("Tangram prediction spot identifiers are not unique")

    observed_spots = [str(value) for value in prediction_frame.index]
    if set(observed_spots) != set(expected_spots):
        missing = sorted(set(expected_spots).difference(observed_spots))
        extra = sorted(set(observed_spots).difference(expected_spots))
        raise ValueError(
            "Tangram prediction spot axis differs from the ST matrix: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    aligned = prediction_frame.copy()
    aligned.index = observed_spots
    aligned = aligned.loc[expected_spots]

    predicted_columns = {}
    duplicate_columns = []
    for column in aligned.columns:
        key = str(column).lower()
        if key in predicted_columns:
            duplicate_columns.append(key)
        predicted_columns[key] = column
    if duplicate_columns:
        raise ValueError(
            "Tangram prediction gene axis is not unique after case folding: "
            f"{sorted(set(duplicate_columns))[:10]}"
        )

    gene_to_index = {}
    duplicate_genes = []
    for index, gene in enumerate(expected_genes):
        key = gene.lower()
        if key in gene_to_index:
            duplicate_genes.append(key)
        gene_to_index[key] = index
    if duplicate_genes:
        raise ValueError(
            "ST gene axis is not unique after case folding: "
            f"{sorted(set(duplicate_genes))[:10]}"
        )
    if len(set(requested_test)) != len(requested_test):
        raise ValueError("Held-out Tangram gene list is not unique after case folding")

    missing_from_st = [gene for gene in requested_test if gene not in gene_to_index]
    missing_predictions = [gene for gene in requested_test if gene not in predicted_columns]
    analytic_zero_predictions = [
        gene for gene in missing_predictions if gene in allowed_analytic_zeros
    ]
    unexpected_missing = [
        gene for gene in missing_predictions if gene not in allowed_analytic_zeros
    ]
    if missing_from_st or unexpected_missing:
        raise ValueError(
            "Tangram did not provide complete held-out-gene coverage: "
            f"missing_from_st={missing_from_st[:10]}, "
            f"unexpected_missing_predictions={unexpected_missing[:10]}"
        )

    for gene in requested_test:
        if gene in analytic_zero_predictions:
            output[:, gene_to_index[gene]] = 0.0
            continue
        values = aligned[predicted_columns[gene]].to_numpy(dtype=np.float32)
        if values.shape != (len(expected_spots),) or not np.isfinite(values).all():
            raise ValueError(f"Tangram produced invalid values for held-out gene {gene}")
        output[:, gene_to_index[gene]] = values

    return output, {
        "requested_test_gene_count": int(len(requested_test)),
        "projected_test_gene_count": int(
            len(requested_test) - len(analytic_zero_predictions)
        ),
        "analytic_zero_test_gene_count": int(len(analytic_zero_predictions)),
        "analytic_zero_test_genes": analytic_zero_predictions,
        "analytic_zero_policy": (
            "all-zero scRNA input vectors map analytically to zero spatial predictions"
        ),
        "predicted_test_gene_count": int(len(requested_test)),
        "missing_test_gene_count": 0,
        "truth_copy_fallback_used": False,
        "prediction_spot_axis_reordered": observed_spots != expected_spots,
    }


def st_library_sizes(count_matrix, denominator_idx=None):
    counts = np.asarray(count_matrix, dtype=np.float32)
    selected = counts if denominator_idx is None else counts[:, np.asarray(denominator_idx, dtype=np.int64)]
    return np.asarray(selected.sum(axis=1, dtype=np.float64), dtype=np.float64)


def normalize_st_counts(count_matrix, denominator_idx, log1p, target_sum=1e4):
    """Normalize every ST column with library sizes computed from denominator_idx only."""
    counts = np.asarray(count_matrix, dtype=np.float32)
    if counts.ndim != 2:
        raise ValueError(f"ST count matrix must be 2D, got shape {counts.shape}")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("ST count matrix must contain finite nonnegative values")

    library_sizes = st_library_sizes(counts, denominator_idx)
    scales = np.ones(library_sizes.shape, dtype=np.float64)
    nonzero = library_sizes > 0.0
    scales[nonzero] = float(target_sum) / library_sizes[nonzero]
    normalized = (counts.astype(np.float64) * scales[:, None]).astype(np.float32)
    normalized[~nonzero, :] = 0.0
    if log1p:
        np.log1p(normalized, out=normalized)
    return normalized


def normalize_from_counts(count_matrix, template_adata, denominator_idx=None):
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
        adata.X = normalize_st_counts(count_matrix, denominator_idx, log1p=True)
    return adata


def _consistent_gene_list(split_obj, keys, label):
    found = []
    for key in keys:
        if key in split_obj:
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


def _coerce_indices(values, label, n_genes, require_nonempty=False):
    indices = np.asarray(values)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a one-dimensional integer array")
    indices = indices.astype(np.int64, copy=False)
    if require_nonempty and indices.size == 0:
        raise ValueError(f"{label} must not be empty")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{label} contains duplicate indices")
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= int(n_genes)):
        raise ValueError(f"{label} contains indices outside [0, {n_genes})")
    return indices


def _consistent_index_list(split_obj, keys, label, n_genes):
    found = []
    for key in keys:
        if key in split_obj:
            found.append((key, _coerce_indices(split_obj[key], key, n_genes)))
    if not found:
        return None
    reference_key, reference = found[0]
    for key, values in found[1:]:
        if not np.array_equal(values, reference):
            raise ValueError(f"Split index lists {reference_key} and {key} disagree")
    return reference


def load_gene_split(path):
    with open(path, "r", encoding="utf-8") as handle:
        split_obj = json.load(handle)
    if not isinstance(split_obj, dict):
        raise ValueError(f"Gene split must be a JSON object: {path}")
    test_genes = _consistent_gene_list(split_obj, ["test_genes", "test_target_genes"], "test")
    if test_genes is None:
        raise KeyError(f"Could not find test gene list in {path}. Keys: {list(split_obj.keys())}")
    return split_obj, test_genes


def load_test_genes(path: str):
    return load_gene_split(path)[1]


def load_gene_indices(path, n_genes, label, require_nonempty=True):
    indices = _coerce_indices(
        np.load(path, allow_pickle=False),
        f"{label} gene indices",
        n_genes,
        require_nonempty=require_nonempty,
    )
    return np.sort(indices)


def load_train_gene_indices(path, n_genes):
    return load_gene_indices(path, n_genes, "train", require_nonempty=True)


def validate_gene_split(split_obj, original_genes, test_genes, train_idx=None):
    original_genes = [str(gene) for gene in original_genes]
    if len(original_genes) != len(set(original_genes)):
        raise ValueError("Original ST columns contain duplicate gene names")
    gene_to_idx = {gene: idx for idx, gene in enumerate(original_genes)}
    n_genes = len(original_genes)

    mode_a_hidden_union = (
        split_obj.get("protocol") == "A"
        and split_obj.get("test_gene_idx_semantics")
        == "ordered_inner_validation_plus_final_test"
    )
    if mode_a_hidden_union:
        definitions = {
            "train": (
                ["inner_train_gene_idx", "train_gene_idx", "train_idx"],
                ["inner_train_genes", "train_genes"],
            ),
            "validation": (
                [
                    "inner_validation_gene_idx",
                    "val_gene_idx",
                    "validation_gene_idx",
                    "val_idx",
                    "validation_idx",
                ],
                ["inner_validation_genes", "val_genes", "validation_genes"],
            ),
            "final_test": (
                ["final_test_gene_idx", "final_test_idx"],
                ["final_test_genes"],
            ),
            "hidden": (
                ["hidden_gene_idx", "test_gene_idx", "test_idx"],
                ["hidden_genes", "test_genes", "test_target_genes"],
            ),
        }
    else:
        definitions = {
            "train": (["train_gene_idx", "train_idx"], ["train_genes"]),
            "validation": (
                ["val_gene_idx", "validation_gene_idx", "val_idx", "validation_idx"],
                ["val_genes", "validation_genes"],
            ),
            "test": (
                ["test_gene_idx", "test_idx"],
                ["test_genes", "test_target_genes"],
            ),
        }
    collections = {}
    for label, (index_keys, gene_keys) in definitions.items():
        split_indices = _consistent_index_list(split_obj, index_keys, label, n_genes)
        split_genes = _consistent_gene_list(split_obj, gene_keys, label)
        mapped_indices = None
        if split_genes is not None:
            missing = [gene for gene in split_genes if gene not in gene_to_idx]
            if missing:
                raise ValueError(f"{label} genes are absent from original ST columns: {missing[:10]}")
            mapped_indices = np.asarray([gene_to_idx[gene] for gene in split_genes], dtype=np.int64)
        if split_indices is not None and mapped_indices is not None and not np.array_equal(split_indices, mapped_indices):
            raise ValueError(f"{label} gene names do not match {label} indices in original ST column order")
        effective = mapped_indices if mapped_indices is not None else split_indices
        if effective is not None:
            collections[label] = effective

    test_idx = np.asarray([gene_to_idx[gene] for gene in test_genes], dtype=np.int64)
    target_label = "hidden" if mode_a_hidden_union else "test"
    if target_label in collections and not np.array_equal(
        collections[target_label], test_idx
    ):
        raise ValueError(
            f"Frozen target genes disagree with the split {target_label} collection"
        )

    if mode_a_hidden_union:
        required = ("train", "validation", "final_test", "hidden")
        missing_roles = [role for role in required if role not in collections]
        if missing_roles:
            raise ValueError(f"Mode A split is missing collections: {missing_roles}")
        expected_hidden = np.concatenate(
            [collections["validation"], collections["final_test"]]
        )
        if not np.array_equal(collections["hidden"], expected_hidden):
            raise ValueError(
                "Mode A hidden collection must be validation followed by final test"
            )
        partition = np.concatenate(
            [
                collections["train"],
                collections["validation"],
                collections["final_test"],
            ]
        )
        if len(np.unique(partition)) != n_genes or set(partition.tolist()) != set(
            range(n_genes)
        ):
            raise ValueError(
                "Mode A train/validation/final-test collections are not a complete disjoint partition"
            )
        labels = ["train", "validation", "final_test"]
    else:
        labels = list(collections)
    for left_pos, left in enumerate(labels):
        for right in labels[left_pos + 1 :]:
            overlap = np.intersect1d(collections[left], collections[right])
            if overlap.size:
                raise ValueError(f"Split collections {left} and {right} overlap at indices {overlap[:10].tolist()}")

    if train_idx is not None:
        overlap = np.intersect1d(train_idx, test_idx)
        if overlap.size:
            raise ValueError(f"Train and test indices overlap at {overlap[:10].tolist()}")
        if "validation" in collections:
            overlap = np.intersect1d(train_idx, collections["validation"])
            if overlap.size:
                raise ValueError(f"Train index input overlaps validation indices at {overlap[:10].tolist()}")
        if "train" in collections:
            split_train = collections["train"]
            if "validation" in collections:
                if set(train_idx.tolist()) != set(split_train.tolist()):
                    raise ValueError("Train index input disagrees with the split train collection")
            elif not set(train_idx.tolist()).issubset(set(split_train.tolist())):
                raise ValueError("Train index input is not a subset of the split non-test train collection")
    return test_idx, collections


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
    hashes = {key: (sha256_file(path) if path is not None else None) for key, path in paths.items()}
    return paths, hashes


def write_run_audit(output_dir, audit):
    with open(Path(output_dir) / "adapter_run_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)


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


def ensure_cluster_labels(adata_sc, train_genes, cluster_label, leiden_resolution):
    if cluster_label in adata_sc.obs.columns:
        adata_sc.obs[cluster_label] = adata_sc.obs[cluster_label].astype(str)
        return adata_sc

    if "counts" in adata_sc.layers:
        X_counts = to_dense(adata_sc[:, train_genes].layers["counts"]).astype(np.float32)
    else:
        X_counts = to_dense(adata_sc[:, train_genes].X).astype(np.float32)

    cluster_adata = ad.AnnData(X=X_counts)
    cluster_adata.obs_names = adata_sc.obs_names.copy()
    cluster_adata.var_names = list(map(str, train_genes))
    cluster_adata.obs = adata_sc.obs.copy()

    sc.pp.normalize_total(cluster_adata, target_sum=1e4)
    sc.pp.log1p(cluster_adata)
    sc.pp.highly_variable_genes(cluster_adata)
    if "highly_variable" in cluster_adata.var and int(cluster_adata.var["highly_variable"].sum()) > 0:
        cluster_adata = cluster_adata[:, cluster_adata.var["highly_variable"]].copy()
    sc.pp.scale(cluster_adata, max_value=10)
    sc.tl.pca(cluster_adata)
    sc.pp.neighbors(cluster_adata)
    sc.tl.leiden(cluster_adata, resolution=float(leiden_resolution))

    adata_sc.obs[cluster_label] = cluster_adata.obs["leiden"].astype(str).values
    return adata_sc


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_mode = (
        args.st_normalization_scope == "train_genes"
        and args.model_gene_scope == "train_indices"
    )
    train_indices_required = (
        args.st_normalization_scope == "train_genes"
        or args.model_gene_scope == "train_indices"
    )
    if train_indices_required and args.train_gene_idx_path is None:
        raise ValueError(
            "--train-gene-idx-path is required when --st-normalization-scope=train_genes "
            "or --model-gene-scope=train_indices"
        )
    if strict_mode and (args.val_gene_idx_path is None or args.test_gene_idx_path is None):
        raise ValueError(
            "Strict Protocol A requires --val-gene-idx-path and "
            "--test-gene-idx-path in addition to --train-gene-idx-path"
        )

    split_obj, test_genes = load_gene_split(args.gene_split_json)
    test_gene_set = {str(g).lower() for g in test_genes}

    adata_raw = load_mhpr_from_txt(
        locations_path=args.locations_path,
        counts_path=args.st_data,
        normalize=False,
        store_raw_layer=True,
    )
    x_counts = to_dense(adata_raw.layers["counts"] if "counts" in adata_raw.layers else adata_raw.X).astype(np.float32)
    original_genes = list(map(str, adata_raw.var_names))
    train_idx = None
    val_idx = None
    final_test_idx = None
    if args.train_gene_idx_path is not None:
        train_idx = load_train_gene_indices(args.train_gene_idx_path, len(original_genes))
    hidden_idx, _ = validate_gene_split(
        split_obj, original_genes, test_genes, train_idx=train_idx
    )

    normalization_audit = None
    if strict_mode:
        val_idx = load_gene_indices(
            args.val_gene_idx_path,
            len(original_genes),
            "validation",
            require_nonempty=True,
        )
        final_test_idx = load_gene_indices(
            args.test_gene_idx_path,
            len(original_genes),
            "test",
            require_nonempty=True,
        )
        train_idx, val_idx, final_test_idx = validate_gene_splits(
            len(original_genes),
            train_gene_idx=train_idx,
            val_gene_idx=val_idx,
            test_gene_idx=final_test_idx,
            require_complete_coverage=True,
        )
        expected_hidden = np.sort(np.concatenate([val_idx, final_test_idx]))
        if not np.array_equal(np.sort(hidden_idx), expected_hidden):
            raise ValueError(
                "Mode-A split JSON hidden genes must equal validation plus final test indices"
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
    else:
        denominator_idx = train_idx if args.st_normalization_scope == "train_genes" else None
        library_sizes = st_library_sizes(x_counts, denominator_idx)
        zero_library_idx = np.flatnonzero(library_sizes == 0.0)
        adata_gt = normalize_from_counts(
            x_counts, adata_raw, denominator_idx=denominator_idx
        )

    adata_scrna = load_scrna_from_txt(
        counts_path=args.sc_data,
        normalize=True,
        store_raw_layer=True,
        target_genes=list(map(str, adata_raw.var_names)),
        max_cells=(None if args.scrna_max_cells is not None and args.scrna_max_cells <= 0 else args.scrna_max_cells),
        seed=args.seed,
        maxabs_scale=False,
    )
    adata_gt, adata_scrna = align_scrna_to_st(adata_gt, adata_scrna)

    # Tangram's mapper has a bad dense-array branch; keep scRNA sparse to use the stable code path.
    adata_sc = adata_scrna.copy()
    adata_sp = adata_gt.copy()
    adata_sc.X = sparse.csr_matrix(to_dense(adata_sc.X).astype(np.float32))
    adata_sp.X = sparse.csr_matrix(to_dense(adata_sp.X).astype(np.float32))
    reference_counts = to_dense(
        adata_sc.layers["counts"] if "counts" in adata_sc.layers else adata_sc.X
    ).astype(np.float32)
    zero_reference_genes = {
        str(adata_sc.var_names[index]).lower()
        for index in np.flatnonzero(reference_counts.sum(axis=0, dtype=np.float64) == 0.0)
    }

    if args.model_gene_scope == "non_test":
        train_genes = [str(g) for g in adata_gt.var_names if str(g).lower() not in test_gene_set]
    else:
        indexed_train_genes = [original_genes[idx] for idx in train_idx]
        aligned_gene_set = set(map(str, adata_gt.var_names))
        missing_aligned = [gene for gene in indexed_train_genes if gene not in aligned_gene_set]
        if missing_aligned:
            raise ValueError(f"Indexed model training genes are absent after ST/scRNA alignment: {missing_aligned[:10]}")
        train_genes = indexed_train_genes
    model_test_overlap = {gene.lower() for gene in train_genes}.intersection(test_gene_set)
    if model_test_overlap:
        raise ValueError(f"Model train and test genes overlap: {sorted(model_test_overlap)[:10]}")
    sc_train_df = pd.DataFrame(
        to_dense(adata_sc[:, train_genes].X).astype(np.float32),
        columns=train_genes,
    )
    sp_train_df = pd.DataFrame(
        to_dense(adata_sp[:, train_genes].X).astype(np.float32),
        columns=train_genes,
    )
    valid_train_genes = [
        gene
        for gene in train_genes
        if float(sc_train_df[gene].sum()) > 0.0 and float(sp_train_df[gene].sum()) > 0.0
    ]
    if len(valid_train_genes) == 0:
        raise ValueError("No valid nonzero training genes remain for Tangram.")

    input_paths, input_hashes = collect_input_audit(args)
    audit = {
        "adapter": "Tangram",
        "st_normalization_scope": args.st_normalization_scope,
        "model_gene_scope": args.model_gene_scope,
        "normalization_denominator_gene_count": int(len(original_genes) if denominator_idx is None else len(denominator_idx)),
        "train_gene_count": int(len(train_genes)),
        "model_train_gene_count": int(len(train_genes)),
        "valid_model_train_gene_count": int(len(valid_train_genes)),
        "train_index_gene_count": (int(len(train_idx)) if train_idx is not None else None),
        "hidden_validation_plus_test_gene_count": int(len(hidden_idx)),
        "validation_gene_count": (int(len(val_idx)) if val_idx is not None else None),
        "final_test_gene_count": (
            int(len(final_test_idx)) if final_test_idx is not None else None
        ),
        "train_test_overlap_count": 0,
        "zero_library_spot_count": int(len(zero_library_idx)),
        "zero_library_spots": [str(adata_raw.obs_names[idx]) for idx in zero_library_idx],
        "input_paths": input_paths,
        "input_sha256": input_hashes,
        "st_data_sha256": input_hashes["st_data"],
        "gene_split_json_sha256": input_hashes["gene_split_json"],
        "train_gene_idx_sha256": input_hashes["train_gene_idx_path"],
        "val_gene_idx_sha256": input_hashes["val_gene_idx_path"],
        "test_gene_idx_sha256": input_hashes["test_gene_idx_path"],
        "normalization_audit": normalization_audit,
        "protocol_role": (
            "strict_primary_modeA" if strict_mode else "explicit_legacy_diagnostic"
        ),
        "eligible_for_strict_primary": bool(strict_mode),
        "adapter_metrics_skipped": bool(args.skip_adapter_metrics),
    }
    write_run_audit(output_dir, audit)

    print(
        f"[Tangram] train genes = {len(train_genes)}, valid train genes = {len(valid_train_genes)}, "
        f"test genes = {len(test_gene_set)}, epochs = {args.num_epochs}, mode = {args.mode}"
    )

    if args.mode == "clusters":
        adata_sc = ensure_cluster_labels(
            adata_sc=adata_sc,
            train_genes=valid_train_genes,
            cluster_label=args.cluster_label,
            leiden_resolution=args.leiden_resolution,
        )

    tg_map.pp_adatas(adata_sc, adata_sp, genes=valid_train_genes, gene_to_lowercase=False)
    adata_map = tg_map.map_cells_to_space(
        adata_sc=adata_sc,
        adata_sp=adata_sp,
        cv_train_genes=valid_train_genes,
        cluster_label=(args.cluster_label if args.mode == "clusters" else None),
        mode=args.mode,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        random_state=args.seed,
        verbose=True,
        density_prior=args.density_prior,
    )
    adata_ge = tg_utils.project_genes(
        adata_map,
        adata_sc,
        cluster_label=(args.cluster_label if args.mode == "clusters" else None),
    )

    imputed = to_dense(adata_gt.X).astype(np.float32).copy()
    pred_df = pd.DataFrame(to_dense(adata_ge.X), index=list(map(str, adata_ge.obs_names)), columns=list(map(str, adata_ge.var_names)))
    imputed, coverage_audit = insert_complete_test_predictions(
        base_matrix=imputed,
        prediction_frame=pred_df,
        expected_spot_names=adata_gt.obs_names,
        all_gene_names=adata_gt.var_names,
        test_gene_names=test_genes,
        analytic_zero_genes=zero_reference_genes,
    )
    if not np.isfinite(imputed).all():
        raise ValueError("Tangram full output matrix contains nonfinite values")

    output_matrix_path = output_dir / "imputed_expression.npy"
    np.save(output_matrix_path, imputed.astype(np.float32))
    audit.update(
        {
            "imputed_matrix_shape": list(imputed.shape),
            "prediction_finite": bool(np.isfinite(imputed).all()),
            "output_matrix_path": str(output_matrix_path),
            "output_matrix_sha256": sha256_file(output_matrix_path),
            "held_out_prediction_coverage": coverage_audit,
        }
    )
    write_run_audit(output_dir, audit)
    with open(output_dir / "gene_split.json", "w", encoding="utf-8") as f:
        json.dump(json.load(open(args.gene_split_json, "r", encoding="utf-8")), f, indent=2, ensure_ascii=False)

    if args.skip_adapter_metrics:
        print(f"[DONE] prediction matrix saved to: {output_matrix_path}")
        return

    gene_df, summary = compute_stdiff_style_gene_metrics(
        x_true=to_dense(adata_gt.X).astype(np.float32),
        x_pred=imputed.astype(np.float32),
        genes=list(map(str, adata_gt.var_names)),
        target_gene_names=test_genes,
    )
    gene_df.to_csv(output_dir / "gene_level_metrics_stdiff_style.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "final_result_stdiff_style.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "Tangram",
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
