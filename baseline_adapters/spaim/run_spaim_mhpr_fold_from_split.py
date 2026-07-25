import os
import sys
import json
import argparse
import hashlib
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as st
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = PROJECT_ROOT / "main"
SPAIM_ROOT = Path(__file__).resolve().parent
SPAIM_SRC = SPAIM_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from utils import load_mhpr_from_txt
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
    p.add_argument("--scrna-max-cells", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--style-dim", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-layers", type=str, default="256, 512")
    p.add_argument("--leiden-resolution", type=float, default=0.5)
    p.add_argument("--kmeans-fallback-clusters", type=int, default=12)
    p.add_argument(
        "--allow-kmeans-cluster-fallback",
        action="store_true",
        help="Legacy diagnostic only: permit KMeans if scRNA Leiden construction fails.",
    )
    p.add_argument(
        "--disable-native-filtering",
        action="store_true",
        help="Disable SpaIM's internal min_cells/min_genes filtering so frozen strict test genes remain evaluable.",
    )
    p.add_argument(
        "--skip-adapter-metrics",
        action="store_true",
        help="Save full prediction matrices only; central evaluator should compute final metrics.",
    )
    p.add_argument(
        "--missing-hidden-fallback",
        type=str,
        default="error",
        choices=["error", "zero"],
        help="Fallback for hidden test genes removed by SpaIM's native filtering. 'zero' keeps full frozen test set without using test truth.",
    )
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


def normalize_total_no_log1p(adata_in, denominator_idx=None):
    adata = adata_in.copy()
    if denominator_idx is None:
        sc.pp.normalize_total(adata, target_sum=1e4)
        adata.X = np.asarray(adata.X, dtype=np.float32)
    else:
        adata.X = normalize_st_counts(to_dense(adata.X), denominator_idx, log1p=False)
    return adata


def load_scrna_target_genes_fast(path, target_genes, max_cells=5000, seed=42, target_sum=1e4):
    header = pd.read_csv(path, sep="\t", nrows=0)
    columns = list(header.columns)
    if len(columns) <= 1:
        raise ValueError("scRNA header is empty or malformed")

    target_set = set(map(str, target_genes))
    gene_cols = [c for c in columns[1:] if str(c).strip() in target_set]
    col_overlap = len(gene_cols)

    first_col = pd.read_csv(path, sep="\t", usecols=[columns[0]])
    first_col_values = first_col.iloc[:, 0].astype(str).str.strip()
    idx_overlap = len(set(first_col_values).intersection(target_set))

    rng = np.random.default_rng(seed)
    if col_overlap >= idx_overlap:
        # Format: rows are cells/spots and columns are genes.
        if len(gene_cols) == 0:
            raise ValueError("No overlapping genes found on scRNA columns")
        df = pd.read_csv(path, sep="\t", usecols=[columns[0]] + gene_cols, index_col=0)
        df.index = df.index.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()
        if max_cells is not None and max_cells > 0 and df.shape[0] > max_cells:
            keep_idx = np.sort(rng.choice(df.shape[0], size=max_cells, replace=False))
            df = df.iloc[keep_idx]
    else:
        # Format: rows are genes and columns are cells.
        cell_cols = columns[1:]
        if max_cells is not None and max_cells > 0 and len(cell_cols) > max_cells:
            keep_idx = np.sort(rng.choice(len(cell_cols), size=max_cells, replace=False))
            keep_cols = [cell_cols[i] for i in keep_idx]
        else:
            keep_cols = cell_cols
        df = pd.read_csv(path, sep="\t", usecols=[columns[0]] + keep_cols, index_col=0)
        df.index = df.index.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()
        keep_genes = [g for g in df.index if str(g) in target_set]
        if len(keep_genes) == 0:
            raise ValueError("No overlapping genes found on scRNA row index")
        df = df.loc[keep_genes].T

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    adata = ad.AnnData(df.to_numpy(dtype=np.float32))
    adata.obs_names = df.index.astype(str)
    adata.var_names = df.columns.astype(str)
    adata.layers["counts"] = df.to_numpy(dtype=np.float32).copy()
    sc.pp.normalize_total(adata, target_sum=float(target_sum))
    adata.X = np.asarray(adata.X, dtype=np.float32)
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


def assemble_spaim_predictions(
    base_matrix,
    impute_df,
    expected_spot_names,
    all_gene_names,
    test_gene_names,
    *,
    strict_protocol_a=True,
    missing_hidden_fallback="error",
):
    """Align SpaIM output and insert complete held-out-gene predictions."""
    output = np.asarray(base_matrix, dtype=np.float32).copy()
    expected_spots = [str(value) for value in expected_spot_names]
    expected_genes = [str(value) for value in all_gene_names]
    requested_test = [str(value) for value in test_gene_names]

    if output.shape != (len(expected_spots), len(expected_genes)):
        raise ValueError("SpaIM base matrix shape does not match the declared spot/gene axes")
    if len(set(expected_spots)) != len(expected_spots):
        raise ValueError("Expected spatial spot identifiers are not unique")
    if len(set(expected_genes)) != len(expected_genes):
        raise ValueError("ST gene identifiers are not unique")
    if len(set(requested_test)) != len(requested_test):
        raise ValueError("Held-out SpaIM gene list is not unique")

    observed_spots = [str(value) for value in impute_df.index]
    if len(set(observed_spots)) != len(observed_spots):
        raise ValueError("SpaIM prediction spot identifiers are not unique")
    if set(observed_spots) != set(expected_spots):
        missing = sorted(set(expected_spots).difference(observed_spots))
        extra = sorted(set(observed_spots).difference(expected_spots))
        raise ValueError(
            "SpaIM prediction spot axis differs from the ST matrix: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    aligned = impute_df.copy()
    aligned.index = observed_spots
    aligned = aligned.loc[expected_spots]

    predicted_columns = {}
    for column in aligned.columns:
        key = str(column)
        if key in predicted_columns:
            raise ValueError(f"SpaIM prediction gene columns are not unique: {key}")
        predicted_columns[key] = column

    gene_to_index = {gene: index for index, gene in enumerate(expected_genes)}
    missing_from_st = [gene for gene in requested_test if gene not in gene_to_index]
    missing_predictions = [gene for gene in requested_test if gene not in predicted_columns]
    if missing_from_st:
        raise ValueError(
            "SpaIM held-out genes are absent from the ST matrix: "
            f"{missing_from_st[:10]}"
        )
    if missing_predictions:
        if strict_protocol_a or missing_hidden_fallback == "error":
            raise ValueError(
                "SpaIM did not provide complete held-out-gene coverage: "
                f"missing_from_st={missing_from_st[:10]}, "
                f"missing_predictions={missing_predictions[:10]}"
            )
        for gene in missing_predictions:
            output[:, gene_to_index[gene]] = 0.0

    for gene in requested_test:
        if gene not in predicted_columns:
            continue
        values = aligned[predicted_columns[gene]].to_numpy(dtype=np.float32)
        if values.shape != (len(expected_spots),) or not np.isfinite(values).all():
            raise ValueError(f"SpaIM produced invalid values for held-out gene {gene}")
        output[:, gene_to_index[gene]] = values

    if not np.isfinite(output).all():
        raise ValueError("SpaIM final prediction matrix contains non-finite values")

    fallback_used = bool(missing_predictions and not strict_protocol_a and missing_hidden_fallback != "error")
    return output, {
        "prediction_spot_axis_reordered": observed_spots != expected_spots,
        "spot_id_unique": True,
        "spot_id_set_matches_truth": True,
        "complete_test_gene_coverage": not missing_from_st and not missing_predictions,
        "requested_test_gene_count": int(len(requested_test)),
        "predicted_test_gene_count": int(len(requested_test) - len(missing_predictions)),
        "missing_test_gene_count": int(len(missing_predictions)),
        "truth_copy_fallback_used": False,
        "hidden_gene_zero_fallback_used": fallback_used,
        "prediction_finite": True,
    }


def maybe_build_leiden_labels(
    adata_sc_normcounts,
    seed,
    resolution,
    kmeans_fallback_clusters,
    allow_kmeans_fallback=False,
):
    work = adata_sc_normcounts.copy()
    sc.pp.log1p(work)
    try:
        sc.pp.highly_variable_genes(work, n_top_genes=min(128, work.n_vars))
        if "highly_variable" in work.var and work.var["highly_variable"].sum() > 2:
            work = work[:, work.var["highly_variable"]].copy()
    except Exception:
        pass
    try:
        sc.pp.scale(work, max_value=10)
    except Exception:
        pass
    try:
        sc.tl.pca(work, svd_solver="arpack")
        sc.pp.neighbors(work)
        sc.tl.leiden(work, resolution=float(resolution), random_state=int(seed))
        labels = work.obs["leiden"].astype("category")
        return labels
    except Exception as error:
        if not allow_kmeans_fallback:
            raise RuntimeError(
                "SpaIM scRNA Leiden label construction failed; "
                "formal Protocol A forbids a silent KMeans substitution"
            ) from error
        n_clusters = max(2, min(int(kmeans_fallback_clusters), work.n_obs))
        x = to_dense(work.X).astype(np.float32)
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=int(seed), batch_size=min(1024, x.shape[0]), n_init=5)
        lab = km.fit_predict(x)
        return pd.Categorical(lab.astype(str))


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
    denom = df.max(axis=0).astype(float)
    denom = denom.mask(denom.abs() < 1e-12, 1.0)
    return df.divide(denom, axis=1)


def scale_z_score_df(df):
    z = st.zscore(df.to_numpy(dtype=float), axis=0, nan_policy="omit")
    return pd.DataFrame(np.nan_to_num(z, nan=0.0), index=df.index, columns=df.columns)


def scale_plus_df(df):
    denom = df.sum(axis=0).astype(float)
    denom = denom.mask(denom.abs() < 1e-12, 1.0)
    return df.divide(denom, axis=1)


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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_mode = (
        args.st_normalization_scope == "train_genes"
        and args.model_gene_scope == "train_indices"
    )
    if strict_mode and args.allow_kmeans_cluster_fallback:
        raise ValueError(
            "Strict Protocol A forbids --allow-kmeans-cluster-fallback"
        )
    if strict_mode and args.missing_hidden_fallback != "error":
        raise ValueError(
            "Strict Protocol A forbids --missing-hidden-fallback zero"
        )
    if strict_mode and not args.disable_native_filtering:
        raise ValueError(
            "Strict Protocol A requires --disable-native-filtering because native "
            "min_cells/min_genes filtering depends on the frozen hidden-gene matrix"
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

    dataset_root = output_dir / "_spaim_dataset_root"
    dataset_name = "fold0"
    dataset_dir = dataset_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    split_obj, test_genes = load_gene_split(args.gene_split_json)
    test_gene_set = set(test_genes)

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
        normalized_log1p, normalization_audit = normalize_st_protocol_a(
            x_counts,
            inner_train_gene_idx=train_idx,
            val_gene_idx=val_idx,
            test_gene_idx=final_test_idx,
            require_complete_coverage=True,
        )
        adata_gt = adata_raw.copy()
        adata_gt.X = normalized_log1p.astype(np.float32)
        adata_st_normcounts = adata_raw.copy()
        adata_st_normcounts.X = np.expm1(normalized_log1p).astype(np.float32)
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
        adata_st_counts = adata_raw.copy()
        adata_st_counts.X = x_counts.astype(np.float32)
        adata_st_normcounts = normalize_total_no_log1p(
            adata_st_counts, denominator_idx=denominator_idx
        )

    adata_sc_normcounts = load_scrna_target_genes_fast(
        path=args.sc_data,
        target_genes=list(map(str, adata_raw.var_names)),
        max_cells=(None if args.scrna_max_cells is not None and args.scrna_max_cells <= 0 else args.scrna_max_cells),
        seed=args.seed,
    )
    adata_sc_normcounts.obs["leiden"] = maybe_build_leiden_labels(
        adata_sc_normcounts,
        seed=args.seed,
        resolution=args.leiden_resolution,
        kmeans_fallback_clusters=args.kmeans_fallback_clusters,
        allow_kmeans_fallback=args.allow_kmeans_cluster_fallback,
    )

    if args.model_gene_scope == "non_test":
        train_genes = [str(g) for g in adata_gt.var_names if str(g) not in test_gene_set]
    else:
        train_genes = [original_genes[idx] for idx in train_idx]
    model_test_overlap = set(train_genes).intersection(test_gene_set)
    if model_test_overlap:
        raise ValueError(f"Model train and test genes overlap: {sorted(model_test_overlap)[:10]}")

    input_paths, input_hashes = collect_input_audit(args)
    audit = {
        "adapter": "SpaIM",
        "st_normalization_scope": args.st_normalization_scope,
        "model_gene_scope": args.model_gene_scope,
        "normalization_denominator_gene_count": int(len(original_genes) if denominator_idx is None else len(denominator_idx)),
        "train_gene_count": int(len(train_genes)),
        "model_train_gene_count": int(len(train_genes)),
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
        "kmeans_cluster_fallback_allowed": bool(args.allow_kmeans_cluster_fallback),
        "native_filtering_disabled": bool(args.disable_native_filtering),
    }
    write_run_audit(output_dir, audit)

    st_h5ad = dataset_dir / "Insitu_count.h5ad"
    sc_h5ad = dataset_dir / "scRNA_count_cluster.h5ad"
    adata_st_normcounts.write_h5ad(st_h5ad)
    adata_sc_normcounts.write_h5ad(sc_h5ad)
    np.save(dataset_dir / "train_list.npy", np.asarray([train_genes], dtype=object), allow_pickle=True)
    np.save(dataset_dir / "test_list.npy", np.asarray([test_genes], dtype=object), allow_pickle=True)

    save_root = output_dir / "_spaim_runs"
    save_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SPAIM_SRC / "main.py"),
        "--root",
        str(dataset_root),
        "--dataset_name",
        dataset_name,
        "--kfold",
        "0",
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--style_dim",
        str(args.style_dim),
        "--lr",
        str(args.lr),
        "--save_path",
        str(save_root),
        "--seed",
        str(args.seed),
        "--parallel",
        "0",
        "--cluster",
        "leiden",
        "--gpu",
        str(args.gpu),
        "--model_layers",
        args.model_layers,
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    if args.disable_native_filtering:
        env["SPAIM_DISABLE_NATIVE_FILTERING"] = "1"
    if args.skip_adapter_metrics:
        env["SPAIM_SKIP_NATIVE_METRICS"] = "1"
    print("[SpaIM] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(SPAIM_SRC), env=env)

    run_dir = save_root / dataset_name
    impute_path = run_dir / "impute_result_0.pkl"
    if not impute_path.exists():
        raise FileNotFoundError(f"SpaIM did not produce {impute_path}")

    impute_df = pd.read_pickle(impute_path)
    imputed, prediction_audit = assemble_spaim_predictions(
        to_dense(adata_gt.X),
        impute_df,
        adata_gt.obs_names,
        adata_gt.var_names,
        test_genes,
        strict_protocol_a=strict_mode,
        missing_hidden_fallback=args.missing_hidden_fallback,
    )
    if prediction_audit["hidden_gene_zero_fallback_used"]:
        with open(output_dir / "missing_hidden_gene_fallback.json", "w", encoding="utf-8") as f:
            json.dump(
                {"fallback": args.missing_hidden_fallback},
                f,
                ensure_ascii=False,
                indent=2,
            )

    output_matrix_path = output_dir / "imputed_expression.npy"
    if not np.isfinite(imputed).all():
        raise ValueError("SpaIM final prediction matrix contains non-finite values")
    np.save(output_matrix_path, imputed)
    pd.DataFrame(imputed, index=adata_gt.obs_names, columns=adata_gt.var_names).to_csv(output_dir / "imputed_expression.csv")
    with open(output_dir / "gene_split.json", "w", encoding="utf-8") as f:
        json.dump({"test_genes": test_genes}, f, ensure_ascii=False, indent=2)
    audit.update(
        {
            "imputed_matrix_shape": list(imputed.shape),
            "prediction_finite": bool(np.isfinite(imputed).all()),
            "prediction_spot_axis_reordered": prediction_audit["prediction_spot_axis_reordered"],
            "complete_test_gene_coverage": prediction_audit["complete_test_gene_coverage"],
            "truth_copy_fallback_used": prediction_audit["truth_copy_fallback_used"],
            "hidden_gene_zero_fallback_used": prediction_audit["hidden_gene_zero_fallback_used"],
            "prediction_alignment": prediction_audit,
            "output_matrix_path": str(output_matrix_path),
            "output_matrix_sha256": sha256_file(output_matrix_path),
        }
    )
    write_run_audit(output_dir, audit)

    if args.skip_adapter_metrics:
        audit.update(
            {
                "reason": "Final benchmark metrics are recomputed by the central evaluator from imputed_expression.npy.",
                "imputed_matrix_shape": list(imputed.shape),
            }
        )
        write_run_audit(output_dir, audit)
        print(f"[DONE] prediction matrix saved to: {output_dir}")
        return

    gene_df, summary = compute_stdiff_style_gene_metrics(
        x_true=to_dense(adata_gt.X).astype(np.float32),
        x_pred=imputed,
        genes=[str(g) for g in adata_gt.var_names],
        target_gene_names=test_genes,
    )
    gene_df.to_csv(output_dir / "gene_level_metrics_stdiff_style.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "final_result_stdiff_style.csv", index=False)
    pd.DataFrame([summary]).rename(
        columns={
            "SPCC_gene_median_stdiff_style": "SPCC",
            "SSIM_gene_median_stdiff_style": "SSIM",
            "RMSE_gene_median_stdiff_style": "RMSE",
            "JS_gene_median_stdiff_style": "JS",
        }
    ).to_csv(output_dir / "final_result.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"[DONE] results saved to: {output_dir}")


if __name__ == "__main__":
    main()
