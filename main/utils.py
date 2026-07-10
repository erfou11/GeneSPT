import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics import mean_squared_error
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# ===============================
# MHPR 数据加载
# ===============================
def load_mhpr_from_txt(locations_path, counts_path,
                       normalize=True,
                       store_raw_layer=True):

    loc = pd.read_csv(locations_path, sep="\t")
    coords = loc.values.astype(np.float32)

    counts = pd.read_csv(counts_path, sep="\t")
    gene_names = counts.columns
    X = counts.values.astype(np.float32)

    if coords.shape[0] != X.shape[0]:
        raise ValueError(
            f"Spot mismatch: {coords.shape[0]} vs {X.shape[0]}"
        )

    adata = ad.AnnData(X)

    adata.var_names = gene_names
    adata.obs_names = [f"spot_{i}" for i in range(X.shape[0])]
    adata.obsm["spatial"] = coords

    if store_raw_layer:
        adata.layers["counts"] = X.copy()

    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    return adata

# ===============================
# mask生成
# ===============================
def make_gene_mask(shape, ratio=0.1, seed=0):

    rng = np.random.default_rng(seed)
    mask = rng.random(shape) > ratio
    return mask.astype(float)

# ===============================
# 应用mask
# ===============================
def apply_mask(X, mask):

    masked = X.copy()
    masked[mask == 0] = 0

    return masked

# ===============================
# 插补评价
# ===============================
def evaluate_imputation(gt, pred, mask):

    target = mask == 0

    gt_vals = gt[target]
    pred_vals = pred[target]

    mse = mean_squared_error(gt_vals, pred_vals)

    # 避免极端情况下常数输入导致 pearsonr/spearmanr 返回 nan
    if gt_vals.size == 0:
        return np.nan, np.nan, np.nan

    if np.std(gt_vals) == 0 or np.std(pred_vals) == 0:
        pcc = np.nan
    else:
        pcc = pearsonr(gt_vals, pred_vals)[0]

    if len(np.unique(gt_vals)) <= 1 or len(np.unique(pred_vals)) <= 1:
        spcc = np.nan
    else:
        spcc = spearmanr(gt_vals, pred_vals)[0]

    return mse, pcc, spcc

# ===============================
# 聚类评价（稳定版，向后兼容）
# ===============================
def clustering_metrics(
    adata,
    label_key="region",
    n_pcs=30,
    n_neighbors=15,
    leiden_resolution=1.0,
    random_state=0,
    compute_umap=False,
    copy=True,
):
    """
    向后兼容：
      - 旧调用 clustering_metrics(adata) 仍可用
      - 默认不删功能，只是把 UMAP 改成可选，避免无关随机性
      - 默认 copy=True，避免原地污染传入的 adata
    """

    adata_eval = adata.copy() if copy else adata

    random.seed(random_state)
    np.random.seed(random_state)

    # 更稳定的 PCA
    sc.pp.pca(adata_eval, svd_solver="arpack")

    # 固定 neighbors 参数，减少评估抖动
    sc.pp.neighbors(adata_eval, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # 固定 Leiden 随机种子
    sc.tl.leiden(
        adata_eval,
        resolution=leiden_resolution,
        random_state=random_state
    )

    # 保留原功能：需要时仍可算 UMAP
    if compute_umap:
        sc.tl.umap(adata_eval, random_state=random_state)

    if label_key in adata_eval.obs:
        labels_true = adata_eval.obs[label_key].astype(str)
    else:
        # 兼容旧逻辑，但显式提示这只是自对比，通常没有真实评价意义
        labels_true = adata_eval.obs["leiden"].astype(str)

    labels_pred = adata_eval.obs["leiden"].astype(str)

    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)

    return ari, nmi
