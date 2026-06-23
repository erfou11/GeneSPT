#!/usr/bin/env python3
"""Recompute the final strict whole-gene benchmark from prediction matrices.

This script intentionally does not trust adapter-level summary files.  Every
ready method/fold is evaluated from a saved prediction matrix, the same ST
ground-truth matrix, the same frozen gene split, and the same metric code.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from run_final_multidataset_fold0_gate import (
    dataset_paths,
    finite_target,
    fit_svd_raw_basis,
    load_masks,
    read_coords,
    read_st,
)
from run_gc_spatiality_aware_training import compute_spatiality
from run_msr_structural_readout import ssim_components
from run_predictable_spatial_program_folds012 import apply_bin_lambdas
from run_predictable_spatial_program_transfer_fold0 import (
    bin_lambdas_from_val,
    component_stats,
    fit_spatiality_predictor,
    selected_component_prediction,
)
from run_st_spatial_program_decoder_fold0 import fit_predict_coeff, project_coeff, preprocess_train
from run_strict_gene_conditioned_decoder_gate import (
    EPS,
    cal_ssim_ref,
    log1p_cpm,
    make_knn_edges,
    scale_max,
    scale_plus,
    scale_z,
    subgroup_indices,
)


ROOT = Path("/workspace/GeneSPT")
INFO = ROOT / "results" / "imformation"
MASK_ROOT = INFO / "final_multidataset_masks"
OUT_MATRIX_ROOT = INFO / "final_recomputed_prediction_matrices"

DATASETS = [
    "Vis9A_D7_spaim_effective4470",
    "HBC_shared16112",
    "MHM_shared14780",
]

FOLDS = range(5)

METHODS = [
    "GeneSPT-GC-PSP",
    "GC-MLP-PCA32-softplus",
    "SpaIM",
    "Tangram",
    "SpaGE",
    "stPlus",
    "TransPA",
]

EXTERNAL_ROOTS = {
    "Vis9A_D7_spaim_effective4470": {
        "SpaIM": ROOT / "results" / "strict_vis9a_spaim_gene5cv",
        "Tangram": ROOT / "results" / "strict_vis9a_tangram_gene5cv",
        "SpaGE": ROOT / "results" / "strict_vis9a_spage_gene5cv",
        "stPlus": ROOT / "results" / "strict_vis9a_stplus_gene5cv",
        "TransPA": ROOT / "results" / "strict_vis9a_transpa_gene5cv",
    },
    "HBC_shared16112": {
        "SpaIM": ROOT / "results" / "final_hbc_spaim_gene5cv",
        "Tangram": ROOT / "results" / "final_hbc_tangram_gene5cv",
        "SpaGE": ROOT / "results" / "final_hbc_spage_gene5cv",
        "stPlus": ROOT / "results" / "final_hbc_stplus_gene5cv",
        "TransPA": ROOT / "results" / "final_hbc_transpa_gene5cv",
    },
    "MHM_shared14780": {
        "SpaIM": ROOT / "results" / "final_mhm14780_spaim_gene5cv",
        "Tangram": ROOT / "results" / "final_mhm14780_tangram_gene5cv",
        "SpaGE": ROOT / "results" / "final_mhm14780_spage_gene5cv",
        "stPlus": ROOT / "results" / "final_mhm14780_stplus_gene5cv",
        "TransPA": ROOT / "results" / "final_mhm14780_transpa_gene5cv",
    },
}

SUMMARY_METRICS = [
    "SPCC",
    "RMSE",
    "JS",
    "SSIM",
    "luminance",
    "contrast",
    "structure",
    "low_expr_SPCC",
    "high_spatial_SPCC",
    "high_spatial_RMSE",
]

RANK_METRICS = {
    "SPCC": False,
    "RMSE": True,
    "JS": True,
    "SSIM": False,
    "high_spatial_SPCC": False,
}


def rel_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def load_dataset(dataset: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    paths = dataset_paths(dataset, MASK_ROOT)
    X_counts, genes = read_st(paths.st_path)
    X = log1p_cpm(X_counts)
    coords = read_coords(paths.loc_path)
    if coords.shape[0] != X.shape[0]:
        raise ValueError(f"{dataset}: coords rows {coords.shape[0]} != spots {X.shape[0]}")
    return X.astype(np.float32), genes, coords.astype(np.float32)


def load_split(dataset: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_masks(MASK_ROOT / dataset, fold)


def gc_base_path(dataset: str, fold: int) -> Path:
    return INFO / "final_multidataset_cache" / dataset / f"fold{fold}" / "gc_mlp_pca32_softplus_correct.npz"


def descriptor_path(dataset: str, fold: int) -> Path:
    return INFO / "final_multidataset_cache" / dataset / f"fold{fold}" / "descriptors_pca32_nmf32.npz"


def psp_path(dataset: str, fold: int) -> Path:
    return OUT_MATRIX_ROOT / dataset / f"fold{fold}" / "genespt_gc_psp_correct.npz"


def external_path(dataset: str, method: str, fold: int) -> Path:
    return EXTERNAL_ROOTS[dataset][method] / f"fold{fold}" / "imputed_expression.npy"


def _same_index(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a.astype(np.int64), b.astype(np.int64)))


def ensure_psp_prediction(dataset: str, fold: int, X: np.ndarray, coords: np.ndarray) -> Path:
    """Create a saved PSP test prediction matrix from frozen cached inputs.

    The GC-MLP base is loaded from the final multidataset cache.  PSP uses the
    frozen selected mechanism: SVD raw K<=64, PCA32+NMF32 descriptors, Ridge
    alpha=10, top predictable components<=32, and val-selected spatiality-bin
    lambdas.  No test-gene truth is used.
    """

    out = psp_path(dataset, fold)
    if out.exists():
        return out

    cache = np.load(gc_base_path(dataset, fold))
    desc = np.load(descriptor_path(dataset, fold))
    train_idx, val_idx, test_idx = load_split(dataset, fold)
    if not _same_index(cache["train_idx"], train_idx) or not _same_index(cache["val_idx"], val_idx) or not _same_index(cache["test_idx"], test_idx):
        raise ValueError(f"{dataset} fold{fold}: GC cache split does not match frozen split")

    base_val = cache["pred_val"].astype(np.float32)
    base_test = cache["pred_test"].astype(np.float32)
    D = desc["pca32_nmf32"].astype(np.float32)

    k_eff = min(64, max(2, len(train_idx) // 2), max(2, X.shape[0] - 2))
    basis = fit_svd_raw_basis(X[:, train_idx], k=k_eff, seed=42 + 1701 * fold)

    X_val_proc, val_meta = preprocess_train(X[:, val_idx], "raw")
    C_val_oracle = project_coeff(basis.A, X_val_proc)
    C_val_pred = fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[val_idx], alpha=10.0, seed=42 + 4000 + fold)
    comp_df = component_stats(C_val_pred, C_val_oracle)
    comp_df["rank_score"] = comp_df["component_spearman"].fillna(-1.0) * np.log1p(comp_df["oracle_coeff_var"].fillna(0.0))
    keep = comp_df.sort_values("rank_score", ascending=False)["component"].to_numpy(dtype=np.int64)[: min(32, basis.k)]
    keep = np.sort(keep.astype(np.int64))
    program_val = selected_component_prediction(basis.A, C_val_pred, val_meta, keep)

    low_val_idx, high_val_idx = subgroup_indices(X, val_idx, coords)
    edges = make_knn_edges(coords, k=8)
    train_spatiality = compute_spatiality(X, train_idx, edges)
    train_moran = finite_target(train_spatiality["MoranI"])
    pred_sp_val = fit_spatiality_predictor(D[train_idx], train_moran, D[val_idx])
    lambdas, lambda_score, lambda_df = bin_lambdas_from_val(
        X,
        base_val,
        program_val,
        val_idx,
        low_val_idx,
        high_val_idx,
        [str(i) for i in range(X.shape[1])],
        pred_sp_val,
    )
    q1, q2 = np.quantile(pred_sp_val, [1 / 3, 2 / 3])

    C_test_pred = fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[test_idx], alpha=10.0, seed=42 + 5000 + fold)
    program_test = selected_component_prediction(basis.A, C_test_pred, basis.meta, keep)
    pred_sp_test = fit_spatiality_predictor(D[train_idx], train_moran, D[test_idx])
    psp_test = apply_bin_lambdas(base_test, program_test, pred_sp_test, (float(q1), float(q2)), lambdas).astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        pred_test=psp_test,
        base_test=base_test,
        program_test=program_test,
        test_idx=test_idx,
        val_idx=val_idx,
        train_idx=train_idx,
        keep_components=keep,
        lambda_low=float(lambdas["low"]),
        lambda_mid=float(lambdas["mid"]),
        lambda_high=float(lambdas["high"]),
        lambda_score=float(lambda_score),
        spatiality_q1=float(q1),
        spatiality_q2=float(q2),
    )
    lambda_df.to_csv(out.with_name("genespt_gc_psp_val_lambda_grid.csv"), index=False)
    comp_df.to_csv(out.with_name("genespt_gc_psp_component_predictability.csv"), index=False)
    return out


def load_method_prediction(dataset: str, method: str, fold: int, X: np.ndarray, coords: np.ndarray) -> tuple[Path | None, np.ndarray | None, np.ndarray | None, str]:
    train_idx, val_idx, test_idx = load_split(dataset, fold)
    if method == "GC-MLP-PCA32-softplus":
        path = gc_base_path(dataset, fold)
        if not path.exists():
            return path, None, test_idx, "missing_prediction"
        z = np.load(path)
        if not _same_index(z["test_idx"], test_idx):
            return path, z["pred_test"], test_idx, "gene_order_issue"
        return path, z["pred_test"].astype(np.float32), test_idx, "ready"

    if method == "GeneSPT-GC-PSP":
        try:
            path = ensure_psp_prediction(dataset, fold, X, coords)
        except Exception as exc:
            return psp_path(dataset, fold), None, test_idx, f"psp_rebuild_failed:{exc}"
        z = np.load(path)
        if not _same_index(z["test_idx"], test_idx):
            return path, z["pred_test"], test_idx, "gene_order_issue"
        return path, z["pred_test"].astype(np.float32), test_idx, "ready"

    path = external_path(dataset, method, fold)
    if not path.exists():
        return path, None, test_idx, "missing_prediction"
    pred = np.load(path)
    if pred.ndim != 2:
        return path, pred, test_idx, "shape_mismatch"
    if pred.shape != X.shape:
        return path, pred, test_idx, "shape_mismatch"
    return path, pred[:, test_idx].astype(np.float32), test_idx, "ready"


def metric_one_gene(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(p)
    if not np.all(finite):
        y = y[finite]
        p = p[finite]
    if y.size == 0:
        return {k: np.nan for k in ["SPCC", "SSIM", "RMSE", "JS", "luminance", "contrast", "structure"]}

    if np.nanstd(y) < EPS or np.nanstd(p) < EPS:
        spcc = np.nan
    else:
        spcc = float(st.spearmanr(y, p).correlation)

    raw_ssim = scale_max(y)
    pred_ssim = scale_max(p)
    m_val = max(float(np.nanmax(raw_ssim)), float(np.nanmax(pred_ssim)), 1e-12)
    try:
        ssim = cal_ssim_ref(raw_ssim, pred_ssim, m_val)
    except Exception:
        ssim = np.nan

    raw_js = scale_plus(y)
    pred_js = scale_plus(p)
    mid = 0.5 * (raw_js + pred_js)
    try:
        js = float(0.5 * st.entropy(raw_js, mid) + 0.5 * st.entropy(pred_js, mid))
    except Exception:
        js = np.nan

    rmse = float(np.sqrt(((scale_z(y) - scale_z(p)) ** 2).mean()))
    try:
        lum, con, stru, _ = ssim_components(y, p)
    except Exception:
        lum, con, stru = np.nan, np.nan, np.nan

    return {
        "SPCC": spcc,
        "SSIM": ssim,
        "RMSE": rmse,
        "JS": js,
        "luminance": float(lum),
        "contrast": float(con),
        "structure": float(stru),
    }


def evaluate_prediction(
    dataset: str,
    method: str,
    fold: int,
    X: np.ndarray,
    genes: list[str],
    pred_test: np.ndarray,
    test_idx: np.ndarray,
    low_idx: np.ndarray,
    high_idx: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    low_set = set(map(int, low_idx))
    high_set = set(map(int, high_idx))
    rows = []
    for j, g in enumerate(test_idx.astype(np.int64)):
        y = X[:, int(g)]
        p = pred_test[:, j]
        row = {
            "dataset": dataset,
            "method": method,
            "fold": int(fold),
            "gene_idx": int(g),
            "gene": str(genes[int(g)]),
            "is_low_expr": int(int(g) in low_set),
            "is_high_spatial": int(int(g) in high_set),
            "true_mean": float(np.nanmean(y)),
            "true_std": float(np.nanstd(y)),
            "pred_mean": float(np.nanmean(p)),
            "pred_std": float(np.nanstd(p)),
        }
        row.update(metric_one_gene(y, p))
        rows.append(row)
    gene_df = pd.DataFrame(rows)

    def med(col: str, mask: pd.Series | np.ndarray | None = None) -> float:
        if mask is None:
            vals = gene_df[col].to_numpy(dtype=np.float64)
        else:
            vals = gene_df.loc[mask, col].to_numpy(dtype=np.float64)
        if vals.size == 0:
            return np.nan
        return float(np.nanmedian(vals))

    summary = {
        "dataset": dataset,
        "method": method,
        "fold": int(fold),
        "SPCC": med("SPCC"),
        "RMSE": med("RMSE"),
        "JS": med("JS"),
        "SSIM": med("SSIM"),
        "luminance": med("luminance"),
        "contrast": med("contrast"),
        "structure": med("structure"),
        "low_expr_SPCC": med("SPCC", gene_df["is_low_expr"].eq(1)),
        "high_spatial_SPCC": med("SPCC", gene_df["is_high_spatial"].eq(1)),
        "high_spatial_RMSE": med("RMSE", gene_df["is_high_spatial"].eq(1)),
        "n_test_genes": int(len(test_idx)),
        "n_low_expr_genes": int(gene_df["is_low_expr"].sum()),
        "n_high_spatial_genes": int(gene_df["is_high_spatial"].sum()),
    }
    return summary, gene_df


def inventory_row(
    dataset: str,
    method: str,
    fold: int,
    path: Path | None,
    X: np.ndarray,
    pred_test: np.ndarray | None,
    test_idx: np.ndarray,
    status: str,
) -> dict:
    row = {
        "dataset": dataset,
        "method": method,
        "fold": int(fold),
        "pred_path": rel_path(path),
        "pred_shape": "",
        "gt_shape": str(tuple(X.shape)),
        "n_test_genes": int(len(test_idx)),
        "n_predicted_test_genes": 0,
        "all_test_genes_present": False,
        "gene_order_exact": False,
        "contains_nan": np.nan,
        "contains_inf": np.nan,
        "negative_fraction": np.nan,
        "status": status,
    }
    if pred_test is None:
        if path is not None and path.exists():
            try:
                arr = np.load(path)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    shape = arr["pred_test"].shape if "pred_test" in arr.files else "npz_no_pred_test"
                else:
                    shape = arr.shape
                row["pred_shape"] = str(shape)
            except Exception:
                pass
        return row

    row["pred_shape"] = str(tuple(pred_test.shape))
    row["n_predicted_test_genes"] = int(pred_test.shape[1]) if pred_test.ndim == 2 else 0
    row["all_test_genes_present"] = bool(pred_test.ndim == 2 and pred_test.shape[1] == len(test_idx))
    row["gene_order_exact"] = bool(row["all_test_genes_present"] and status == "ready")
    row["contains_nan"] = bool(np.isnan(pred_test).any())
    row["contains_inf"] = bool(np.isinf(pred_test).any())
    finite = pred_test[np.isfinite(pred_test)]
    row["negative_fraction"] = float(np.mean(finite < 0.0)) if finite.size else np.nan
    return row


def summarize_long(long_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), g in long_df.groupby(["dataset", "method"], sort=True):
        row = {"dataset": dataset, "method": method}
        for metric in SUMMARY_METRICS:
            vals = g[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.nanmean(vals)) if vals.size else np.nan
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan
            row[f"{metric}_median"] = float(np.nanmedian(vals)) if vals.size else np.nan
        row["n_done_folds"] = int(g["fold"].nunique())
        inv = inventory_df[(inventory_df["dataset"].eq(dataset)) & (inventory_df["method"].eq(method))]
        row["n_failed_or_missing_folds"] = int((inv["status"] != "ready").sum()) if not inv.empty else 5
        row["status"] = "complete" if row["n_done_folds"] == 5 else ("partial" if row["n_done_folds"] else "unavailable")
        rows.append(row)

    # Add unavailable method rows so the final table is explicit.
    present = {(r["dataset"], r["method"]) for r in rows}
    for dataset in DATASETS:
        for method in METHODS:
            if (dataset, method) in present:
                continue
            inv = inventory_df[(inventory_df["dataset"].eq(dataset)) & (inventory_df["method"].eq(method))]
            row = {"dataset": dataset, "method": method}
            for metric in SUMMARY_METRICS:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_median"] = np.nan
            row["n_done_folds"] = 0
            row["n_failed_or_missing_folds"] = int((inv["status"] != "ready").sum()) if not inv.empty else 5
            row["status"] = "unavailable"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def build_rank_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rank_rows = []
    for dataset, g in summary_df.groupby("dataset", sort=True):
        complete = g[g["status"].eq("complete")].copy()
        for _, row in g.iterrows():
            out = {"dataset": dataset, "method": row["method"], "status": row["status"]}
            for metric, ascending in RANK_METRICS.items():
                col = f"{metric}_mean"
                if row["status"] != "complete" or not np.isfinite(row.get(col, np.nan)):
                    out[f"{metric}_rank"] = np.nan
                    continue
                ranks = complete[col].rank(method="min", ascending=ascending)
                out[f"{metric}_rank"] = int(ranks.loc[row.name])
            rank_rows.append(out)
    return pd.DataFrame(rank_rows)


def build_paired_tests(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in long_df.groupby("dataset", sort=True):
        topo = g[g["method"].eq("GeneSPT-GC-PSP")].set_index("fold")
        if topo.empty:
            continue
        for method, h in g.groupby("method", sort=True):
            if method == "GeneSPT-GC-PSP":
                continue
            other = h.set_index("fold")
            folds = sorted(set(topo.index).intersection(set(other.index)))
            if len(folds) < 2:
                continue
            for metric in RANK_METRICS:
                a = topo.loc[folds, metric].to_numpy(dtype=np.float64)
                b = other.loc[folds, metric].to_numpy(dtype=np.float64)
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 2:
                    continue
                diff = a[mask] - b[mask]
                try:
                    t_p = float(st.ttest_rel(a[mask], b[mask], nan_policy="omit").pvalue)
                except Exception:
                    t_p = np.nan
                try:
                    w_p = float(st.wilcoxon(diff).pvalue) if np.any(np.abs(diff) > 1e-12) else 1.0
                except Exception:
                    w_p = np.nan
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"GeneSPT-GC-PSP_vs_{method}",
                        "method": method,
                        "metric": metric,
                        "n_folds": int(mask.sum()),
                        "genespt_mean": float(np.nanmean(a[mask])),
                        "other_mean": float(np.nanmean(b[mask])),
                        "genespt_minus_other": float(np.nanmean(diff)),
                        "paired_t_p": t_p,
                        "wilcoxon_p": w_p,
                    }
                )
    return pd.DataFrame(rows)


def model_alias(name: str) -> str:
    aliases = {
        "gc_mlp_pca32_softplus_correct": "GC-MLP-PCA32-softplus",
        "genespt_gc_psp_correct": "GeneSPT-GC-PSP",
    }
    return aliases.get(str(name), str(name))


def build_old_vs_new_diff(summary_df: pd.DataFrame) -> pd.DataFrame:
    old_path = INFO / "final_main_benchmark_merged_summary.csv"
    if not old_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(old_path)
    old["method"] = old["model"].map(model_alias)
    metrics = ["SPCC", "RMSE", "JS", "SSIM", "luminance", "contrast", "structure", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]
    rows = []
    for _, new_row in summary_df.iterrows():
        old_match = old[(old["dataset"].eq(new_row["dataset"])) & (old["method"].eq(new_row["method"]))]
        for metric in metrics:
            new_col = f"{metric}_mean"
            old_col = f"{metric}_mean"
            old_value = float(old_match.iloc[0][old_col]) if (not old_match.empty and old_col in old_match.columns) else np.nan
            new_value = float(new_row[new_col]) if new_col in new_row and pd.notna(new_row[new_col]) else np.nan
            abs_diff = abs(new_value - old_value) if np.isfinite(new_value) and np.isfinite(old_value) else np.nan
            rel_diff = abs_diff / max(abs(old_value), 1e-12) if np.isfinite(abs_diff) else np.nan
            rows.append(
                {
                    "dataset": new_row["dataset"],
                    "method": new_row["method"],
                    "metric": metric,
                    "old_value": old_value,
                    "recomputed_value": new_value,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "large_diff": bool(np.isfinite(abs_diff) and abs_diff > 1e-6),
                    "ssim_mismatch": bool(metric == "SSIM" and np.isfinite(abs_diff) and abs_diff > 1e-6),
                    "old_available": bool(not old_match.empty),
                    "new_status": new_row["status"],
                }
            )
    return pd.DataFrame(rows)


def fmt(x: float, digits: int = 4) -> str:
    if x is None or not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table writer."""
    if df.empty:
        return "_No rows._"
    safe = df.copy()
    safe = safe.astype(object).where(pd.notna(safe), "NA")
    cols = list(safe.columns)
    lines = [
        "| " + " | ".join(map(str, cols)) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in safe.iterrows():
        vals = [str(row[c]).replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown_table(summary_df: pd.DataFrame, rank_df: pd.DataFrame) -> None:
    out = INFO / "table_main_recomputed_benchmark.md"
    cols = [
        "dataset",
        "method",
        "SPCC_mean",
        "RMSE_mean",
        "JS_mean",
        "SSIM_mean",
        "low_expr_SPCC_mean",
        "high_spatial_SPCC_mean",
        "status",
    ]
    table = summary_df[cols].copy()
    for col in [c for c in table.columns if c.endswith("_mean")]:
        table[col] = table[col].map(lambda v: fmt(v, 4))
    lines = [
        "# Main Recomputed Benchmark Table",
        "",
        "Caption: All metrics are recomputed centrally from saved prediction matrices using the same evaluator. Methods missing the full frozen test gene set are marked unavailable and are not evaluated on a subset.",
        "",
        md_table(table),
        "",
        "## GeneSPT Ranks",
        "",
    ]
    topo_ranks = rank_df[rank_df["method"].eq("GeneSPT-GC-PSP")].copy()
    if not topo_ranks.empty:
        lines.append(md_table(topo_ranks))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(summary_df: pd.DataFrame, rank_df: pd.DataFrame, diff_df: pd.DataFrame, inventory_df: pd.DataFrame) -> None:
    audit_lines = [
        "# Final Benchmark Recompute Audit",
        "",
        "## What Changed",
        "",
        "- The final benchmark no longer trusts adapter-level `final_result_stdiff_style.csv` summaries.",
        "- Every ready method/fold is evaluated from a saved prediction matrix, the same log1p(CPM) ST ground truth, the same frozen `test_gene_idx`, and the same central evaluator.",
        "- GeneSPT-GC-PSP matrices were materialized under `results/imformation/final_recomputed_prediction_matrices/` from the frozen GC cache plus fixed PSP readout parameters; no new model training was run.",
        "",
        "## Inventory Status",
        "",
    ]
    inv_status = inventory_df.groupby(["dataset", "method", "status"]).size().reset_index(name="n_folds")
    audit_lines.append(md_table(inv_status))
    audit_lines += [
        "",
        "## Old vs Recomputed Differences",
        "",
    ]
    if diff_df.empty:
        audit_lines.append("Old merged table was not found.")
    else:
        large = diff_df[diff_df["large_diff"]]
        audit_lines.append(f"Rows with abs diff > 1e-6: {len(large)} / {len(diff_df)}.")
        ssim = diff_df[diff_df["ssim_mismatch"]]
        audit_lines.append(f"SSIM mismatches: {len(ssim)}.")
        if not ssim.empty:
            audit_lines.append("")
            audit_lines.append(md_table(ssim.sort_values(["dataset", "method"]).head(30)))
    (INFO / "final_benchmark_recompute_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    complete = summary_df[summary_df["status"].eq("complete")]
    topo_rank = rank_df[rank_df["method"].eq("GeneSPT-GC-PSP")].copy()
    rank1_top2 = {}
    for metric in RANK_METRICS:
        col = f"{metric}_rank"
        vals = topo_rank[col].dropna().to_numpy(dtype=float)
        rank1_top2[metric] = {
            "rank1": int(np.sum(vals == 1)),
            "top2": int(np.sum(vals <= 2)),
            "n_datasets": int(vals.size),
        }
    unavailable = inventory_df[inventory_df["status"] != "ready"].groupby(["dataset", "method"])["fold"].nunique().reset_index(name="n_unavailable_folds")

    claims_lines = [
        "# Claims After Recomputed Benchmark",
        "",
        "## GeneSPT Rank By Dataset",
        "",
        md_table(topo_rank),
        "",
        "## Rank-1 / Top-2 Counts",
        "",
        md_table(pd.DataFrame([{"metric": k, **v} for k, v in rank1_top2.items()])),
        "",
        "## SSIM Limitation",
        "",
    ]
    if not topo_rank.empty:
        ssim_top2 = rank1_top2["SSIM"]["top2"]
        n_ds = rank1_top2["SSIM"]["n_datasets"]
        claims_lines.append(
            f"GeneSPT-GC-PSP is top-2 by SSIM on {ssim_top2}/{n_ds} recomputed datasets. Treat SSIM as a remaining limitation unless this count supports a stronger claim."
        )
    claims_lines += [
        "",
        "## Unavailable External Methods",
        "",
        md_table(unavailable) if not unavailable.empty else "None.",
    ]
    (INFO / "final_claims_after_recomputed_benchmark.md").write_text("\n".join(claims_lines) + "\n", encoding="utf-8")

    safe = [
        "# Safe Claims Recomputed",
        "",
        "- Metrics in the final table are centrally recomputed from saved prediction matrices with one evaluator.",
        "- GeneSPT-GC-PSP can be compared on Vis9A, HBC, and MHM14780 only against methods that predicted the full frozen test gene set.",
        "- SpaIM is unavailable on HBC/MHM14780 in this strict table if it lacks the full frozen test gene prediction matrix.",
        "- Use the recomputed rank table, not adapter summaries, for SPCC/RMSE/JS/SSIM/high-spatial claims.",
    ]
    (INFO / "final_safe_claims_recomputed.md").write_text("\n".join(safe) + "\n", encoding="utf-8")

    unsafe = [
        "# Unsafe Claims Recomputed",
        "",
        "- Do not claim metrics from adapter `final_result_stdiff_style.csv` as final benchmark evidence.",
        "- Do not evaluate unavailable SpaIM HBC/MHM14780 on a smaller gene subset.",
        "- Do not claim GeneSPT is best on every metric unless the recomputed rank table shows rank 1.",
        "- Do not use calibration or legacy entry-MNAR results to support the strict whole-gene main table.",
    ]
    (INFO / "final_unsafe_claims_recomputed.md").write_text("\n".join(unsafe) + "\n", encoding="utf-8")

    write_markdown_table(summary_df, rank_df)


def main() -> None:
    inventory_rows = []
    long_rows = []
    gene_rows = []

    for dataset in DATASETS:
        print(f"[recompute] loading {dataset}", flush=True)
        X, genes, coords = load_dataset(dataset)
        for fold in FOLDS:
            train_idx, val_idx, test_idx = load_split(dataset, fold)
            low_idx, high_idx = subgroup_indices(X, test_idx, coords)
            for method in METHODS:
                print(f"[recompute] {dataset} fold{fold} {method}", flush=True)
                path, pred_test, pred_test_idx, status = load_method_prediction(dataset, method, fold, X, coords)
                if pred_test is not None and (pred_test.ndim != 2 or pred_test.shape[0] != X.shape[0] or pred_test.shape[1] != len(test_idx)):
                    status = "shape_mismatch" if status == "ready" else status
                inventory_rows.append(inventory_row(dataset, method, fold, path, X, pred_test, test_idx, status))
                if status != "ready" or pred_test is None:
                    continue
                summary, gdf = evaluate_prediction(dataset, method, fold, X, genes, pred_test, test_idx, low_idx, high_idx)
                summary["pred_path"] = rel_path(path)
                long_rows.append(summary)
                gene_rows.append(gdf)

    inventory_df = pd.DataFrame(inventory_rows)
    long_df = pd.DataFrame(long_rows)
    gene_df = pd.concat(gene_rows, ignore_index=True) if gene_rows else pd.DataFrame()
    summary_df = summarize_long(long_df, inventory_df)
    paired_df = build_paired_tests(long_df)
    rank_df = build_rank_table(summary_df)
    diff_df = build_old_vs_new_diff(summary_df)

    inventory_df.to_csv(INFO / "final_prediction_matrix_inventory.csv", index=False)
    long_df.to_csv(INFO / "final_main_benchmark_recomputed_long.csv", index=False)
    summary_df.to_csv(INFO / "final_main_benchmark_recomputed_summary.csv", index=False)
    paired_df.to_csv(INFO / "final_main_benchmark_recomputed_paired_tests.csv", index=False)
    rank_df.to_csv(INFO / "final_main_benchmark_recomputed_rank_table.csv", index=False)
    diff_df.to_csv(INFO / "final_benchmark_recompute_diff.csv", index=False)
    summary_df.to_csv(INFO / "table_main_recomputed_benchmark.csv", index=False)
    if not gene_df.empty:
        gene_df.to_csv(INFO / "final_main_benchmark_recomputed_gene_level.csv", index=False)
    write_reports(summary_df, rank_df, diff_df, inventory_df)
    print("[recompute] complete", flush=True)


if __name__ == "__main__":
    main()
