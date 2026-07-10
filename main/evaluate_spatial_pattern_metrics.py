#!/usr/bin/env python3
"""Evaluate hidden-gene spatial pattern metrics for a completed GeneSPT fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from utils import load_mhpr_from_txt


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    return float(pearsonr(x, y).statistic)


def _build_knn_edges(coords: np.ndarray, k: int, undirected: bool = True) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    k = int(max(1, min(k, n - 1)))
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    nbr = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
    edges = {(int(i), int(j)) for i in range(n) for j in nbr[i] if i != int(j)}
    if undirected:
        undirected_edges = set()
        for i, j in edges:
            a, b = (i, j) if i < j else (j, i)
            undirected_edges.add((a, b))
        edges = undirected_edges
    return np.asarray(sorted(edges), dtype=np.int64)


def _moran_i_by_gene(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n, g = x.shape
    out = np.full(g, np.nan, dtype=np.float64)
    if edges.size == 0:
        return out
    src = edges[:, 0]
    dst = edges[:, 1]
    s0 = float(len(src) * 2)
    for col in range(g):
        v = x[:, col]
        valid = np.isfinite(v)
        if valid.sum() < 3:
            continue
        z = v - np.nanmean(v)
        denom = np.nansum(z * z)
        if denom <= 1e-12:
            continue
        pair_valid = valid[src] & valid[dst]
        if pair_valid.sum() == 0:
            continue
        numerator = 2.0 * np.nansum(z[src[pair_valid]] * z[dst[pair_valid]])
        out[col] = (float(n) / s0) * (numerator / denom)
    return out


def _edge_gradient_metrics(pred: np.ndarray, true: np.ndarray, edges: np.ndarray) -> dict:
    if edges.size == 0:
        return {
            "graph_gradient_corr_mean": np.nan,
            "graph_gradient_corr_median": np.nan,
            "graph_gradient_error_abs_mean": np.nan,
        }
    src = edges[:, 0]
    dst = edges[:, 1]
    pred_grad = pred[src] - pred[dst]
    true_grad = true[src] - true[dst]
    error_abs = np.abs(pred_grad - true_grad)
    corr = np.asarray([_safe_corr(pred_grad[:, j], true_grad[:, j]) for j in range(pred_grad.shape[1])])
    return {
        "graph_gradient_corr_mean": float(np.nanmean(corr)) if np.isfinite(corr).any() else np.nan,
        "graph_gradient_corr_median": float(np.nanmedian(corr)) if np.isfinite(corr).any() else np.nan,
        "graph_gradient_error_abs_mean": float(np.nanmean(error_abs)),
    }


def _normalized_laplacian_apply(values: np.ndarray, edges: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.float64)
    if edges.size == 0:
        return values.copy(), 0
    n = values.shape[0]
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n) & (src != dst)
    if valid.sum() == 0:
        return values.copy(), 0
    src_valid = src[valid]
    dst_valid = dst[valid]
    # Use the same undirected normalized Laplacian convention as training.
    src = np.concatenate([src_valid, dst_valid], axis=0)
    dst = np.concatenate([dst_valid, src_valid], axis=0)
    degree = np.zeros(n, dtype=np.float64)
    np.add.at(degree, src, 1.0)
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, eps))
    norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    aggregated = np.zeros_like(values, dtype=np.float64)
    np.add.at(aggregated, src, values[dst] * norm[:, None])
    return values - aggregated, int(src.size)


def _spectral_moment_metrics(pred: np.ndarray, true: np.ndarray, edges: np.ndarray, eps: float = 1e-12) -> dict:
    if edges.size == 0:
        return {
            "spectral_E1_rel_error": np.nan,
            "spectral_E2_rel_error": np.nan,
            "spectral_num_edges": 0,
            "spectral_num_valid_genes": 0,
        }
    pred_c = pred - np.nanmean(pred, axis=0, keepdims=True)
    true_c = true - np.nanmean(true, axis=0, keepdims=True)
    lap_pred, num_edges = _normalized_laplacian_apply(pred_c, edges, eps=eps)
    lap_true, _ = _normalized_laplacian_apply(true_c, edges, eps=eps)
    e0_pred = np.nansum(pred_c * pred_c, axis=0)
    e0_true = np.nansum(true_c * true_c, axis=0)
    e1_pred = np.nansum(pred_c * lap_pred, axis=0)
    e1_true = np.nansum(true_c * lap_true, axis=0)
    e2_pred = np.nansum(lap_pred * lap_pred, axis=0)
    e2_true = np.nansum(lap_true * lap_true, axis=0)
    valid = (
        np.isfinite(e0_pred)
        & np.isfinite(e0_true)
        & np.isfinite(e1_pred)
        & np.isfinite(e1_true)
        & np.isfinite(e2_pred)
        & np.isfinite(e2_true)
        & (e0_pred > eps)
        & (e0_true > eps)
        & (e1_true > eps)
        & (e2_true > eps)
    )
    if valid.sum() == 0:
        return {
            "spectral_E1_rel_error": np.nan,
            "spectral_E2_rel_error": np.nan,
            "spectral_num_edges": int(num_edges),
            "spectral_num_valid_genes": 0,
        }
    return {
        "spectral_E1_rel_error": float(np.nanmean(np.abs(e1_pred[valid] - e1_true[valid]) / np.maximum(np.abs(e1_true[valid]), eps))),
        "spectral_E2_rel_error": float(np.nanmean(np.abs(e2_pred[valid] - e2_true[valid]) / np.maximum(np.abs(e2_true[valid]), eps))),
        "spectral_E0_pred_mean": float(np.nanmean(e0_pred[valid])),
        "spectral_E0_true_mean": float(np.nanmean(e0_true[valid])),
        "spectral_E1_pred_mean": float(np.nanmean(e1_pred[valid])),
        "spectral_E1_true_mean": float(np.nanmean(e1_true[valid])),
        "spectral_E2_pred_mean": float(np.nanmean(e2_pred[valid])),
        "spectral_E2_true_mean": float(np.nanmean(e2_true[valid])),
        "spectral_num_edges": int(num_edges),
        "spectral_num_valid_genes": int(valid.sum()),
    }


def _last_stage_b_stats(history_path: Path) -> dict:
    if not history_path.exists():
        return {}
    hist = pd.read_csv(history_path)
    if "stage" in hist.columns:
        stage_b = hist[hist["stage"].astype(str) == "stage_b_st"]
        if len(stage_b) > 0:
            hist = stage_b
    if len(hist) == 0:
        return {}
    row = hist.tail(1).iloc[0]

    def get_float(name: str, default=np.nan) -> float:
        if name not in row.index:
            return float(default)
        try:
            return float(row[name])
        except Exception:
            return float(default)

    residual_abs = get_float("stage_b_neighbor_residual_mean_abs")
    gate_mean = get_float("stage_b_gate_mean")
    return {
        "spp_loss_grad": get_float("spp_loss_grad", 0.0),
        "spp_grad_pred_abs_mean": get_float("spp_grad_pred_abs_mean", 0.0),
        "spp_grad_true_abs_mean": get_float("spp_grad_true_abs_mean", 0.0),
        "spp_grad_error_abs_mean": get_float("spp_grad_error_abs_mean", 0.0),
        "residual_abs_mean": residual_abs,
        "gate_mean": gate_mean,
        "gated_residual_abs_mean": residual_abs * gate_mean if np.isfinite(residual_abs) and np.isfinite(gate_mean) else np.nan,
        "gspc_loss": get_float("gspc_loss", 0.0),
        "gspc_E0_pred_mean": get_float("gspc_E0_pred_mean", np.nan),
        "gspc_E0_true_mean": get_float("gspc_E0_true_mean", np.nan),
        "gspc_E1_pred_mean": get_float("gspc_E1_pred_mean", np.nan),
        "gspc_E1_true_mean": get_float("gspc_E1_true_mean", np.nan),
        "gspc_E2_pred_mean": get_float("gspc_E2_pred_mean", np.nan),
        "gspc_E2_true_mean": get_float("gspc_E2_true_mean", np.nan),
        "gspc_E1_rel_error_mean": get_float("gspc_E1_rel_error_mean", np.nan),
        "gspc_E2_rel_error_mean": get_float("gspc_E2_rel_error_mean", np.nan),
        "gspc_num_hidden_genes": get_float("gspc_num_hidden_genes", 0.0),
        "gspc_num_edges": get_float("gspc_num_edges", 0.0),
    }


def evaluate_fold(
    fold_dir: Path,
    counts_path: Path,
    locations_path: Path,
    spatial_knn: int = 15,
    output_path: Path | None = None,
) -> dict:
    eval_dir = fold_dir / "eval"
    train_dir = fold_dir / "train"
    pred = np.load(eval_dir / "imputed.npy").astype(np.float64)
    test_mask = np.load(eval_dir / "test_target_mask.npy").astype(bool)
    hidden_idx = np.where(test_mask.sum(axis=0) > 0)[0]
    if hidden_idx.size == 0:
        raise ValueError(f"No hidden genes found in {eval_dir / 'test_target_mask.npy'}")
    spot_idx = np.where(test_mask[:, hidden_idx].any(axis=1))[0]
    if spot_idx.size == 0:
        raise ValueError(f"No evaluated spots found in {eval_dir / 'test_target_mask.npy'}")

    adata_true = load_mhpr_from_txt(str(locations_path), str(counts_path), normalize=True, store_raw_layer=False)
    true = np.asarray(adata_true.X, dtype=np.float64)
    coords = np.asarray(adata_true.obsm["spatial"], dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, true={true.shape}")

    coords_eval = coords[spot_idx]
    edges = _build_knn_edges(coords_eval, k=int(spatial_knn), undirected=True)
    pred_h = pred[np.ix_(spot_idx, hidden_idx)]
    true_h = true[np.ix_(spot_idx, hidden_idx)]
    moran_pred = _moran_i_by_gene(pred_h, edges)
    moran_true = _moran_i_by_gene(true_h, edges)
    moran_valid = np.isfinite(moran_pred) & np.isfinite(moran_true)

    final_metrics_path = eval_dir / "final_result_stdiff_style.csv"
    final_metrics = pd.read_csv(final_metrics_path).iloc[0].to_dict() if final_metrics_path.exists() else {}
    result = {
        "fold_dir": str(fold_dir),
        "metrics_file": str(final_metrics_path),
        "training_history_file": str(train_dir / "training_history.csv"),
        "n_hidden_genes": int(hidden_idx.size),
        "n_evaluated_spots": int(spot_idx.size),
        "spatial_knn": int(spatial_knn),
        "num_undirected_edges": int(edges.shape[0]),
        "SPCC": float(final_metrics.get("SPCC_gene_median_stdiff_style", np.nan)),
        "SSIM": float(final_metrics.get("SSIM_gene_median_stdiff_style", np.nan)),
        "RMSE": float(final_metrics.get("RMSE_gene_median_stdiff_style", np.nan)),
        "JS": float(final_metrics.get("JS_gene_median_stdiff_style", np.nan)),
        "MoranI_gt_pred_spearman": _safe_corr(moran_true[moran_valid], moran_pred[moran_valid], method="spearman"),
        "MoranI_gt_pred_pearson": _safe_corr(moran_true[moran_valid], moran_pred[moran_valid], method="pearson"),
        "MoranI_MAE": float(np.nanmean(np.abs(moran_pred[moran_valid] - moran_true[moran_valid]))) if moran_valid.any() else np.nan,
    }
    result.update(_edge_gradient_metrics(pred_h, true_h, edges))
    result.update(_spectral_moment_metrics(pred_h, true_h, edges))
    result.update(_last_stage_b_stats(train_dir / "training_history.csv"))

    config_path = fold_dir / "config.json"
    if config_path.exists():
        cfg = json.load(open(config_path, "r", encoding="utf-8"))
        for key in [
            "requested_prior_mode",
            "effective_prior_mode",
            "effective_stage_b_module_mode",
            "effective_stage_b_process_mode",
            "effective_stage_b_hidden_supervision_mode",
            "effective_stage_b_spatial_pattern_loss_weight",
            "effective_stage_b_spatial_pattern_loss_type",
            "effective_stage_b_graph_spectral_loss_weight",
            "effective_stage_b_graph_spectral_loss_type",
        ]:
            result[key] = cfg.get(key)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--counts-path", type=Path, required=True)
    parser.add_argument("--locations-path", type=Path, required=True)
    parser.add_argument("--spatial-knn", type=int, default=15)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate_fold(
        fold_dir=args.fold_dir,
        counts_path=args.counts_path,
        locations_path=args.locations_path,
        spatial_knn=args.spatial_knn,
        output_path=args.output_json,
    )
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(args.output_csv, index=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
