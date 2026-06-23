#!/usr/bin/env python3
"""Build final available-dataset four-metric strict benchmark.

This table intentionally uses only the traditional four metrics:
SPCC, RMSE, JS and SpaIM/stDiff source-style SSIM. All values are recomputed
centrally from saved prediction matrices and frozen strict gene splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import spearmanr


ROOT = Path("/workspace/GeneSPT")
INFO = ROOT / "results" / "imformation"
FIG_DIR = INFO / "final_manuscript_figures"
FOLDS = range(5)

if str(ROOT / "main") not in sys.path:
    sys.path.insert(0, str(ROOT / "main"))

from recompute_final_benchmark_from_predictions import ensure_psp_prediction  # noqa: E402
from run_final_multidataset_fold0_gate import dataset_paths, load_masks, read_coords, read_st  # noqa: E402


DATASETS = {
    "Vis9A_D7_spaim_effective4470": {
        "display": "Vis9A",
        "technology": "Sequencing / 10X Visium",
        "roots": {
            "SpaGE": "strict_vis9a_spage_gene5cv",
            "Tangram": "strict_vis9a_tangram_gene5cv",
            "TransPA": "strict_vis9a_transpa_gene5cv",
            "stPlus": "strict_vis9a_stplus_gene5cv",
            "SpaIM": "strict_vis9a_spaim_gene5cv",
            # No strict whole-gene stDiff five-fold prediction matrix exists.
            "stDiff": None,
        },
    },
    "MHPR_current_panel": {
        "display": "MHPR",
        "technology": "Image-based / MERFISH",
        "roots": {
            "SpaGE": "mhpr_spage_gene5cv_native_allinone",
            "Tangram": "mhpr_tangram_gene5cv_native_allinone",
            "TransPA": "mhpr_transpa_gene5cv_native_allinone",
            "stPlus": "mhpr_stplus_gene5cv_native_allinone",
            "SpaIM": "mhpr_spaim_gene5cv_native_allinone",
            "stDiff": "mhpr_stdiff_gene5cv_native_allinone",
        },
    },
    "MVC_shared981": {
        "display": "MVC",
        "technology": "Image-based / STARmap",
        "roots": {
            "SpaGE": "mvc_shared981_spage_gene5cv_exact_allinone",
            "Tangram": "mvc_shared981_tangram_gene5cv_exact_allinone",
            "TransPA": "mvc_shared981_transpa_gene5cv_exact_allinone",
            "stPlus": "mvc_shared981_stplus_gene5cv_exact_allinone",
            "SpaIM": None,
            "stDiff": "mvc_shared981_stdiff_gene5cv_exact_allinone",
        },
    },
    "MG_shared347": {
        "display": "MG",
        "technology": "Other public / seqFISH",
        "roots": {
            "SpaGE": "seqfish_mg347_spage_gene5cv",
            "Tangram": "seqfish_mg347_tangram_gene5cv",
            "TransPA": "seqfish_mg347_transpa_gene5cv",
            "stPlus": "seqfish_mg347_stplus_gene5cv",
            "SpaIM": "seqfish_mg347_spaim_gene5cv",
            "stDiff": "seqfish_mg347_stdiff_gene5cv",
        },
    },
}

METHOD_ORDER = [
    "GeneSPT-GC-PSP",
    "GC-MLP-PCA32-softplus",
    "SpaGE",
    "Tangram",
    "TransPA",
    "stPlus",
    "stDiff",
    "SpaIM",
]


def log1p_cpm(x: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lib = x.sum(axis=1, keepdims=True)
    lib = np.where(lib > 0, lib, 1.0)
    return np.log1p(x / lib * float(target_sum)).astype(np.float32)


def safe_z(x: np.ndarray) -> np.ndarray:
    z = st.zscore(x)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def scale_max(x: np.ndarray) -> np.ndarray:
    denom = float(np.nanmax(x))
    if abs(denom) < 1e-12 or not np.isfinite(denom):
        denom = 1.0
    return x / denom


def scale_sum(x: np.ndarray) -> np.ndarray:
    denom = float(np.nansum(x))
    if abs(denom) < 1e-12 or not np.isfinite(denom):
        denom = 1.0
    return x / denom


def cal_ssim_ref(im1: np.ndarray, im2: np.ndarray, m_val: float) -> float:
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, length = 0.01, 0.03, float(m_val)
    c1 = (k1 * length) ** 2
    c2 = (k2 * length) ** 2
    c3 = c2 / 2
    l12 = (2 * mu1 * mu2 + c1) / (mu1**2 + mu2**2 + c1)
    c12 = (2 * sigma1 * sigma2 + c2) / (sigma1**2 + sigma2**2 + c2)
    s12 = (sigma12 + c3) / (sigma1 * sigma2 + c3)
    return float(l12 * c12 * s12)


def metric_one_gene(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.nan_to_num(np.asarray(y_true, dtype=np.float64), nan=1e-20)
    y_pred = np.nan_to_num(np.asarray(y_pred, dtype=np.float64), nan=1e-20)

    try:
        spcc = float(spearmanr(y_true, y_pred)[0])
    except Exception:
        spcc = np.nan

    raw_ssim = scale_max(y_true).reshape(-1, 1)
    pred_ssim = scale_max(y_pred).reshape(-1, 1)
    m_val = max(float(raw_ssim.max()), float(pred_ssim.max()))
    try:
        ssim = cal_ssim_ref(raw_ssim, pred_ssim, m_val)
    except Exception:
        ssim = np.nan

    raw_js = scale_sum(y_true)
    pred_js = scale_sum(y_pred)
    mid = (raw_js + pred_js) / 2.0
    try:
        js = float(0.5 * st.entropy(raw_js, mid) + 0.5 * st.entropy(pred_js, mid))
    except Exception:
        js = np.nan

    raw_rmse = safe_z(y_true)
    pred_rmse = safe_z(y_pred)
    try:
        rmse = float(np.sqrt(((raw_rmse - pred_rmse) ** 2).mean()))
    except Exception:
        rmse = np.nan

    return {"SPCC": spcc, "RMSE": rmse, "JS": js, "SSIM": ssim}


def summarize_prediction(
    dataset: str,
    method: str,
    fold: int,
    x_true: np.ndarray,
    pred_test: np.ndarray,
    test_idx: np.ndarray,
    genes: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for j, gene_idx in enumerate(test_idx):
        row = {
            "dataset": dataset,
            "method": method,
            "fold": int(fold),
            "gene": genes[int(gene_idx)],
            "gene_idx": int(gene_idx),
        }
        row.update(metric_one_gene(x_true[:, int(gene_idx)], pred_test[:, j]))
        rows.append(row)
    gene_df = pd.DataFrame(rows)
    summary = {m: float(np.nanmedian(gene_df[m])) for m in ["SPCC", "RMSE", "JS", "SSIM"]}
    return gene_df, summary


def load_stdiff_csv(path: Path, n_spots: int, genes: list[str], test_idx: np.ndarray) -> tuple[np.ndarray | None, str]:
    if not path.exists():
        return None, "missing_prediction"
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    test_genes = [genes[int(i)] for i in test_idx]

    if df.shape[0] == n_spots and set(test_genes).issubset(set(df.columns)):
        return df.loc[:, test_genes].to_numpy(dtype=np.float32), "ready_test_columns"
    if df.shape[1] == n_spots and set(test_genes).issubset(set(df.index)):
        return df.loc[test_genes, :].T.to_numpy(dtype=np.float32), "ready_test_rows"
    if df.shape[0] == n_spots and df.shape[1] == len(test_idx):
        return df.to_numpy(dtype=np.float32), "ready_test_column_order"
    if df.shape[1] == n_spots and df.shape[0] == len(test_idx):
        return df.T.to_numpy(dtype=np.float32), "ready_test_row_order"
    if df.shape[0] == n_spots and set(genes).issubset(set(df.columns)):
        return df.loc[:, test_genes].to_numpy(dtype=np.float32), "ready_full_columns"
    if df.shape[1] == n_spots and set(genes).issubset(set(df.index)):
        return df.loc[test_genes, :].T.to_numpy(dtype=np.float32), "ready_full_rows"
    return None, f"shape_or_gene_mismatch:{df.shape}"


def load_external_prediction(
    dataset: str,
    method: str,
    fold: int,
    x: np.ndarray,
    test_idx: np.ndarray,
    genes: list[str],
) -> tuple[np.ndarray | None, str, str]:
    root_name = DATASETS[dataset]["roots"].get(method)
    if root_name is None:
        return None, "unavailable", ""
    fold_dir = ROOT / "results" / root_name / f"fold{fold}"

    npy_path = fold_dir / "imputed_expression.npy"
    if npy_path.exists():
        arr = np.load(npy_path)
        if arr.shape == x.shape:
            return arr[:, test_idx].astype(np.float32), "ready_full_matrix", str(npy_path)
        if arr.shape == (x.shape[0], len(test_idx)):
            return arr.astype(np.float32), "ready_test_matrix", str(npy_path)
        if arr.shape == (len(test_idx), x.shape[0]):
            return arr.T.astype(np.float32), "ready_test_matrix_transposed", str(npy_path)
        return None, f"shape_mismatch:{arr.shape}", str(npy_path)

    csv_path = fold_dir / "stDiff_impute.csv"
    pred, status = load_stdiff_csv(csv_path, x.shape[0], genes, test_idx)
    if pred is not None:
        return pred, status, str(csv_path)

    return None, "missing_prediction", str(npy_path)


def load_prediction(
    dataset: str,
    method: str,
    fold: int,
    x: np.ndarray,
    coords: np.ndarray,
    test_idx: np.ndarray,
    genes: list[str],
) -> tuple[np.ndarray | None, str, str]:
    if method == "GC-MLP-PCA32-softplus":
        path = INFO / "final_multidataset_cache" / dataset / f"fold{fold}" / "gc_mlp_pca32_softplus_correct.npz"
        if not path.exists():
            return None, "missing_prediction", str(path)
        z = np.load(path)
        if not np.array_equal(z["test_idx"].astype(np.int64), test_idx.astype(np.int64)):
            return None, "split_mismatch", str(path)
        return z["pred_test"].astype(np.float32), "ready", str(path)

    if method == "GeneSPT-GC-PSP":
        path = ensure_psp_prediction(dataset, fold, x, coords)
        z = np.load(path)
        if not np.array_equal(z["test_idx"].astype(np.int64), test_idx.astype(np.int64)):
            return None, "split_mismatch", str(path)
        return z["pred_test"].astype(np.float32), "ready", str(path)

    return load_external_prediction(dataset, method, fold, x, test_idx, genes)


def write_table(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Final available-dataset strict whole-gene benchmark",
        "",
        "Traditional metrics only: SPCC, RMSE, JS, and SpaIM/stDiff source-style SSIM. All values are centrally recomputed from saved prediction matrices and the same frozen strict gene splits.",
        "",
        "| Dataset | Technology | Method | Folds | SPCC ↑ | RMSE ↓ | JS ↓ | SSIM ↑ |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    df = summary.copy()
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(["dataset_display", "method"])
    for _, r in df.iterrows():
        if r["status"] != "complete":
            lines.append(
                f"| {r['dataset_display']} | {r['technology']} | {r['method']} | "
                f"{int(r['n_done_folds'])}/5 | unavailable | unavailable | unavailable | unavailable |"
            )
            continue
        lines.append(
            f"| {r['dataset_display']} | {r['technology']} | {r['method']} | {int(r['n_done_folds'])}/5 | "
            f"{r['SPCC_mean']:.4f} ± {r['SPCC_std']:.4f} | "
            f"{r['RMSE_mean']:.4f} ± {r['RMSE_std']:.4f} | "
            f"{r['JS_mean']:.4f} ± {r['JS_std']:.4f} | "
            f"{r['SSIM_mean']:.4f} ± {r['SSIM_std']:.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    long_rows = []
    gene_rows = []
    inv_rows = []

    for dataset, meta in DATASETS.items():
        paths = dataset_paths(dataset, INFO / "final_multidataset_masks")
        x_counts, genes = read_st(paths.st_path)
        x = log1p_cpm(x_counts)
        coords = read_coords(paths.loc_path)
        for fold in FOLDS:
            _, _, test_idx = load_masks(paths.mask_dir, fold)
            for method in METHOD_ORDER:
                pred, status, path = load_prediction(dataset, method, fold, x, coords, test_idx, genes)
                inv = {
                    "dataset": dataset,
                    "dataset_display": meta["display"],
                    "technology": meta["technology"],
                    "method": method,
                    "fold": int(fold),
                    "pred_path": path,
                    "status": status,
                    "n_test_genes": int(len(test_idx)),
                }
                if pred is None:
                    inv_rows.append(inv)
                    continue
                inv.update(
                    {
                        "pred_shape": str(tuple(pred.shape)),
                        "contains_nan": bool(np.isnan(pred).any()),
                        "contains_inf": bool(np.isinf(pred).any()),
                        "negative_fraction": float(np.mean(pred < 0)),
                    }
                )
                inv_rows.append(inv)
                gene_df, metrics = summarize_prediction(dataset, method, fold, x, pred, test_idx, genes)
                row = {
                    "dataset": dataset,
                    "dataset_display": meta["display"],
                    "technology": meta["technology"],
                    "method": method,
                    "fold": int(fold),
                    "status": status,
                }
                row.update(metrics)
                long_rows.append(row)
                gene_rows.append(gene_df)

    long_df = pd.DataFrame(long_rows)
    inv_df = pd.DataFrame(inv_rows)
    summary_rows = []
    for dataset, meta in DATASETS.items():
        for method in METHOD_ORDER:
            sub = long_df[(long_df["dataset"].eq(dataset)) & (long_df["method"].eq(method))]
            row = {
                "dataset": dataset,
                "dataset_display": meta["display"],
                "technology": meta["technology"],
                "method": method,
                "n_done_folds": int(sub["fold"].nunique()) if not sub.empty else 0,
                "n_failed_or_missing_folds": int(5 - (sub["fold"].nunique() if not sub.empty else 0)),
                "status": "complete" if (not sub.empty and sub["fold"].nunique() == 5) else "unavailable",
            }
            for m in ["SPCC", "RMSE", "JS", "SSIM"]:
                row[f"{m}_mean"] = float(sub[m].mean()) if not sub.empty else np.nan
                row[f"{m}_std"] = float(sub[m].std()) if len(sub) > 1 else np.nan
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    rank_rows = []
    for dataset, grp in summary[summary["status"].eq("complete")].groupby("dataset"):
        for metric, ascending in [("SPCC_mean", False), ("RMSE_mean", True), ("JS_mean", True), ("SSIM_mean", False)]:
            tmp = grp.dropna(subset=[metric]).copy()
            tmp["metric"] = metric.replace("_mean", "")
            tmp["rank"] = tmp[metric].rank(method="min", ascending=ascending)
            rank_rows.append(tmp[["dataset", "dataset_display", "technology", "method", "metric", "rank", metric]])
    rank_df = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()

    INFO.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(INFO / "final_available_datasets_four_metric_inventory.csv", index=False)
    long_df.to_csv(INFO / "final_available_datasets_four_metric_long.csv", index=False)
    summary.to_csv(INFO / "final_available_datasets_four_metric_summary.csv", index=False)
    rank_df.to_csv(INFO / "final_available_datasets_four_metric_rank.csv", index=False)
    if gene_rows:
        pd.concat(gene_rows, ignore_index=True).to_csv(INFO / "final_available_datasets_four_metric_gene_level.csv", index=False)
    write_table(summary, INFO / "table_final_available_datasets_four_metric.md")

    print(summary[summary["status"].eq("complete")].sort_values(["dataset_display", "SPCC_mean"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
