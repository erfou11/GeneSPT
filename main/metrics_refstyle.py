import numpy as np
import scipy.stats as st
import scanpy as sc
import anndata as ad
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _safe_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _zscore_per_gene(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    return (X - mu) / sigma


def pcc_gene_mean_refstyle(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """
    Align with the provided CalculateMeteics.PCC logic:
    1) z-score each gene independently
    2) compute Pearson correlation gene-by-gene across spots
    3) return the mean across genes
    """
    Xt = _zscore_per_gene(X_true)
    Xp = _zscore_per_gene(X_pred)

    vals = []
    for j in range(Xt.shape[1]):
        a = Xt[:, j]
        b = Xp[:, j]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        r, _ = st.pearsonr(a, b)
        vals.append(r)
    return float(np.nanmean(vals)) if vals else np.nan


def rmse_gene_mean_refstyle(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """
    Align with the provided CalculateMeteics.RMSE logic:
    1) z-score each gene independently
    2) compute RMSE gene-by-gene across spots
    3) return the mean across genes
    """
    Xt = _zscore_per_gene(X_true)
    Xp = _zscore_per_gene(X_pred)
    rmse_per_gene = np.sqrt(((Xt - Xp) ** 2).mean(axis=0))
    return float(np.nanmean(rmse_per_gene))


def leiden_labels_refstyle(adata_template: ad.AnnData, X_input, key_added: str):
    """
    Align with the provided clustering code:
    - PCA
    - neighbors(n_pcs=30, n_neighbors=30)
    - leiden
    """
    adata_tmp = adata_template.copy()
    adata_tmp.X = np.asarray(X_input, dtype=np.float32)
    sc.tl.pca(adata_tmp)
    sc.pp.neighbors(adata_tmp, n_pcs=30, n_neighbors=30)
    sc.tl.leiden(adata_tmp, key_added=key_added)
    return adata_tmp.obs[key_added].astype(str).values


def ari_nmi_refstyle(adata_template: ad.AnnData, X_true, X_pred):
    """
    Align with the provided cluster() logic:
    - compute Leiden on original expression
    - compute Leiden on imputed expression
    - compare the two with ARI/NMI
    """
    true_labels = leiden_labels_refstyle(adata_template, X_true, key_added="leiden_true_refstyle")
    pred_labels = leiden_labels_refstyle(adata_template, X_pred, key_added="leiden_pred_refstyle")
    ari = float(adjusted_rand_score(true_labels, pred_labels))
    nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
    return ari, nmi


def evaluate_refstyle_metrics(adata_gt: ad.AnnData, X_pred: np.ndarray) -> dict:
    """
    Convenience wrapper for use inside run_mhpr_full_pipeline.py.
    Uses the FULL matrix, matching the provided reference code style.
    """
    X_true = _safe_dense(adata_gt.X).astype(np.float32)
    X_pred = np.asarray(X_pred, dtype=np.float32)
    if X_true.shape != X_pred.shape:
        raise ValueError(f"Shape mismatch: X_true {X_true.shape} vs X_pred {X_pred.shape}")

    ari, nmi = ari_nmi_refstyle(adata_gt, X_true, X_pred)
    return {
        "PCC_gene_zscore_mean": pcc_gene_mean_refstyle(X_true, X_pred),
        "RMSE_gene_zscore_mean": rmse_gene_mean_refstyle(X_true, X_pred),
        "ARI_refstyle": ari,
        "NMI_refstyle": nmi,
        "ARI_NMI_refstyle_mode": "proxy:original_expression_leiden_npcs30_nneighbors30",
    }
