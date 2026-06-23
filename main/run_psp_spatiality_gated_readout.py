#!/usr/bin/env python3
"""PSP spatiality-gated graph diffusion readout gate.

This script is a post-training diagnostic/readout only. It reuses cached
canonical GC-MLP predictions, reconstructs the fixed PSP predictions, selects
graph smoothing parameters on val genes, and evaluates test genes once.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.stats as st
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from run_gc_spatiality_aware_training import compute_spatiality
from run_st_spatial_program_decoder_fold0 import Basis, fit_predict_coeff, project_coeff
from run_st_spatial_program_decoder_fold0 import preprocess_train, reconstruct
from run_predictable_spatial_program_folds012 import (
    apply_bin_lambdas,
    fit_svd_raw_basis,
    load_or_train_base_cached,
    summarize_model,
)
from run_predictable_spatial_program_transfer_fold0 import (
    bin_lambdas_from_val,
    component_stats,
    fit_spatiality_predictor,
    selected_component_prediction,
)
from run_strict_gene_conditioned_decoder_gate import (
    build_descriptors,
    gene_metrics,
    load_matrix,
    log1p_cpm,
    make_knn_edges,
    subgroup_indices,
    summarize_gene_df,
)


INFO = Path("/workspace/GeneSPT/results/imformation")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def knn_weight_graph(coords: np.ndarray, k: int = 8) -> sp.csr_matrix:
    nn = NearestNeighbors(n_neighbors=min(k + 1, coords.shape[0]), metric="euclidean")
    nn.fit(coords)
    dist, ind = nn.kneighbors(coords)
    sigma = float(np.median(dist[:, 1:])) if dist.shape[1] > 1 else 1.0
    sigma = max(sigma, 1e-6)
    rows, cols, data = [], [], []
    for i in range(coords.shape[0]):
        for d, j in zip(dist[i, 1:], ind[i, 1:]):
            w = float(np.exp(-(d * d) / (2.0 * sigma * sigma)))
            rows.extend([i, int(j)])
            cols.extend([int(j), i])
            data.extend([w, w])
    W = sp.coo_matrix((data, (rows, cols)), shape=(coords.shape[0], coords.shape[0])).tocsr()
    W.sum_duplicates()
    W.data = np.maximum(W.data, 0.0)
    return W


def random_graph_like(n: int, k: int, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    rows, cols, data = [], [], []
    for i in range(n):
        choices = rng.choice(np.delete(np.arange(n), i), size=min(k, n - 1), replace=False)
        for j in choices:
            rows.extend([i, int(j)])
            cols.extend([int(j), i])
            data.extend([1.0, 1.0])
    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    W.sum_duplicates()
    W.data[:] = 1.0
    return W


def spot_permuted_graph(W: sp.csr_matrix, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(W.shape[0])
    return W[perm][:, perm].tocsr()


def bilateral_graph(coords: np.ndarray, X_train: np.ndarray, k: int = 8, seed: int = 42) -> sp.csr_matrix:
    W_sp = knn_weight_graph(coords, k=k).tocoo()
    Xc = X_train.astype(np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    k_svd = min(16, Xc.shape[0] - 2, Xc.shape[1] - 2)
    if k_svd >= 2:
        emb = TruncatedSVD(n_components=k_svd, random_state=seed).fit_transform(Xc)
    else:
        emb = Xc
    diffs = emb[W_sp.row] - emb[W_sp.col]
    ed = np.sqrt(np.sum(diffs * diffs, axis=1))
    sig = max(float(np.median(ed)), 1e-6)
    weights = W_sp.data * np.exp(-(ed * ed) / (2.0 * sig * sig))
    W = sp.coo_matrix((weights, (W_sp.row, W_sp.col)), shape=W_sp.shape).tocsr()
    W.sum_duplicates()
    return W


def row_normalized_with_self(W: sp.csr_matrix, self_weight: float = 1.0) -> sp.csr_matrix:
    P = W.tocsr() + sp.eye(W.shape[0], format="csr") * float(self_weight)
    deg = np.asarray(P.sum(axis=1)).reshape(-1)
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg > 0)
    return sp.diags(inv).dot(P).tocsr()


def graph_smooth(W: sp.csr_matrix, Y: np.ndarray, method: str, param: float | int) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float32)
    if method in {"rw", "bilateral_rw"}:
        P = row_normalized_with_self(W)
        out = Y.copy()
        for _ in range(int(param)):
            out = P.dot(out).astype(np.float32)
        return np.clip(out, 0.0, None).astype(np.float32)
    if method == "heat":
        mu = float(param)
        deg = np.asarray(W.sum(axis=1)).reshape(-1)
        L = sp.diags(deg) - W
        A = sp.eye(W.shape[0], format="csr") + mu * L
        out = spla.spsolve(A.tocsc(), Y)
        return np.clip(np.asarray(out, dtype=np.float32), 0.0, None)
    raise ValueError(method)


@dataclass
class PspFold:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    low_val_idx: np.ndarray
    high_val_idx: np.ndarray
    low_test_idx: np.ndarray
    high_test_idx: np.ndarray
    psp_val: np.ndarray
    psp_test: np.ndarray
    pred_sp_val: np.ndarray
    pred_sp_test: np.ndarray
    sp_thresholds: tuple[float, float]
    spatial_graph: sp.csr_matrix
    bilateral_graph: sp.csr_matrix


def build_psp_fold(
    fold: int,
    X: np.ndarray,
    genes: list[str],
    coords: np.ndarray,
    edges: np.ndarray,
    desc: dict[str, np.ndarray],
    args,
    device: torch.device,
) -> PspFold:
    train_idx = np.load(args.mask_dir / f"fold{fold}_train_gene_idx.npy")
    val_idx = np.load(args.mask_dir / f"fold{fold}_val_gene_idx.npy")
    test_idx = np.load(args.mask_dir / f"fold{fold}_test_gene_idx.npy")
    low_val_idx, high_val_idx = subgroup_indices(X, val_idx, coords)
    low_test_idx, high_test_idx = subgroup_indices(X, test_idx, coords)
    base, _ = load_or_train_base_cached(X, desc["pca32"], train_idx, val_idx, test_idx, device, args, fold)
    D = desc["pca32_nmf32"]
    basis = fit_svd_raw_basis(X[:, train_idx], k=64, seed=args.seed + fold)
    X_val_proc, val_meta = preprocess_train(X[:, val_idx], "raw")
    C_val_oracle = project_coeff(basis.A, X_val_proc)
    C_val_pred = fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[val_idx], 10.0, seed=args.seed + fold)
    comp = component_stats(C_val_pred, C_val_oracle)
    comp["rank_score"] = comp["component_spearman"].fillna(-1.0) * np.log1p(comp["oracle_coeff_var"].clip(lower=0))
    keep = comp.sort_values("rank_score", ascending=False).head(min(32, basis.k))["component"].to_numpy(dtype=np.int64)
    program_val = selected_component_prediction(basis.A, C_val_pred, val_meta, keep)
    train_sp = compute_spatiality(X, train_idx, edges)
    train_moran = train_sp["MoranI"].to_numpy(dtype=np.float32)
    pred_sp_val = fit_spatiality_predictor(D[train_idx], train_moran, D[val_idx])
    lambdas, _, _ = bin_lambdas_from_val(X, base["val"], program_val, val_idx, low_val_idx, high_val_idx, genes, pred_sp_val)
    q1, q2 = np.quantile(pred_sp_val, [1 / 3, 2 / 3])
    psp_val = apply_bin_lambdas(base["val"], program_val, pred_sp_val, (q1, q2), lambdas)
    C_test_pred = fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[test_idx], 10.0, seed=args.seed + 99 + fold)
    program_test = selected_component_prediction(basis.A, C_test_pred, basis.meta, keep)
    pred_sp_test = fit_spatiality_predictor(D[train_idx], train_moran, D[test_idx])
    psp_test = apply_bin_lambdas(base["test"], program_test, pred_sp_test, (q1, q2), lambdas)
    W = knn_weight_graph(coords, k=8)
    W_bi = bilateral_graph(coords, X[:, train_idx], k=8, seed=args.seed + fold)
    return PspFold(
        fold=fold,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        low_val_idx=low_val_idx,
        high_val_idx=high_val_idx,
        low_test_idx=low_test_idx,
        high_test_idx=high_test_idx,
        psp_val=psp_val,
        psp_test=psp_test,
        pred_sp_val=pred_sp_val,
        pred_sp_test=pred_sp_test,
        sp_thresholds=(float(q1), float(q2)),
        spatial_graph=W,
        bilateral_graph=W_bi,
    )


def apply_lambda_mode(
    base_pred: np.ndarray,
    smooth_pred: np.ndarray,
    pred_sp: np.ndarray,
    thresholds: tuple[float, float],
    lambda_mode: str,
    lambda_spec: dict[str, float] | float,
) -> np.ndarray:
    if lambda_mode == "global":
        lam = float(lambda_spec)
        return (1.0 - lam) * base_pred + lam * smooth_pred
    return apply_bin_lambdas(base_pred, smooth_pred, pred_sp, thresholds, lambda_spec)  # type: ignore[arg-type]


def apply_lambda_mode_with_labels(
    base_pred: np.ndarray,
    smooth_pred: np.ndarray,
    labels: np.ndarray,
    lambdas: dict[str, float],
) -> np.ndarray:
    out = base_pred.copy()
    for name, lam in lambdas.items():
        mask = labels == name
        out[:, mask] = (1.0 - float(lam)) * base_pred[:, mask] + float(lam) * smooth_pred[:, mask]
    return out


def spatiality_labels(pred_sp: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    q1, q2 = thresholds
    labels = np.full(pred_sp.shape, "mid", dtype=object)
    labels[pred_sp <= q1] = "low"
    labels[pred_sp > q2] = "high"
    return labels


def metric_row(model: str, X: np.ndarray, pred: np.ndarray, idx: np.ndarray, low_idx: np.ndarray, high_idx: np.ndarray, genes: list[str], extra: dict) -> tuple[dict, pd.DataFrame]:
    return summarize_model(model, X, pred, idx, low_idx, high_idx, genes, extra)


def _nanmedian_metric(vals: np.ndarray) -> float:
    return float(np.nanmedian(vals)) if vals.size else float("nan")


def _fast_gene_metrics(Y: np.ndarray, P: np.ndarray) -> dict[str, np.ndarray]:
    Y = np.asarray(Y, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    eps = 1e-12

    Yr = st.rankdata(Y, axis=0)
    Pr = st.rankdata(P, axis=0)
    Yrc = Yr - Yr.mean(axis=0, keepdims=True)
    Prc = Pr - Pr.mean(axis=0, keepdims=True)
    spcc = np.sum(Yrc * Prc, axis=0) / np.sqrt(np.maximum(np.sum(Yrc * Yrc, axis=0) * np.sum(Prc * Prc, axis=0), eps))

    Yz = (Y - Y.mean(axis=0, keepdims=True)) / np.maximum(Y.std(axis=0, keepdims=True), eps)
    Pz = (P - P.mean(axis=0, keepdims=True)) / np.maximum(P.std(axis=0, keepdims=True), eps)
    rmse = np.sqrt(np.mean((Yz - Pz) ** 2, axis=0))

    Ys = Y / np.maximum(Y.sum(axis=0, keepdims=True), eps)
    Ps = P / np.maximum(P.sum(axis=0, keepdims=True), eps)
    M = 0.5 * (Ys + Ps)
    js = 0.5 * np.sum(np.where(Ys > 0, Ys * np.log(np.maximum(Ys, eps) / np.maximum(M, eps)), 0.0), axis=0)
    js += 0.5 * np.sum(np.where(Ps > 0, Ps * np.log(np.maximum(Ps, eps) / np.maximum(M, eps)), 0.0), axis=0)

    Ymax = np.maximum(np.nanmax(Y, axis=0, keepdims=True), eps)
    Pmax = np.maximum(np.nanmax(P, axis=0, keepdims=True), eps)
    Yss = Y / Ymax
    Pss = P / Pmax
    L = np.maximum.reduce([np.nanmax(Yss, axis=0), np.nanmax(Pss, axis=0), np.full(Y.shape[1], eps)])
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2.0
    mu_y = Yss.mean(axis=0)
    mu_p = Pss.mean(axis=0)
    sig_y = np.sqrt(np.mean((Yss - mu_y) ** 2, axis=0))
    sig_p = np.sqrt(np.mean((Pss - mu_p) ** 2, axis=0))
    cov = np.mean((Yss - mu_y) * (Pss - mu_p), axis=0)
    lum = (2 * mu_y * mu_p + C1) / (mu_y * mu_y + mu_p * mu_p + C1)
    con = (2 * sig_y * sig_p + C2) / (sig_y * sig_y + sig_p * sig_p + C2)
    stru = (cov + C3) / (sig_y * sig_p + C3)
    ssim = lum * con * stru
    return {"SPCC": spcc, "RMSE": rmse, "JS": js, "SSIM": ssim}


def fast_summary_row(model: str, X: np.ndarray, pred: np.ndarray, idx: np.ndarray, low_idx: np.ndarray, high_idx: np.ndarray, extra: dict) -> dict:
    idx = np.asarray(idx, dtype=np.int64)
    pos = {int(g): i for i, g in enumerate(idx)}
    low_pos = np.asarray([pos[int(g)] for g in low_idx if int(g) in pos], dtype=np.int64)
    high_pos = np.asarray([pos[int(g)] for g in high_idx if int(g) in pos], dtype=np.int64)
    metrics = _fast_gene_metrics(X[:, idx], pred)
    row = {"model": model, **extra}
    for key, vals in metrics.items():
        row[key] = _nanmedian_metric(vals)
    for prefix, sub_pos in [("low_expr_", low_pos), ("high_spatial_", high_pos)]:
        for key, vals in metrics.items():
            row[f"{prefix}{key}"] = _nanmedian_metric(vals[sub_pos]) if sub_pos.size else float("nan")
    return row


def candidate_graph(fold_data: PspFold, graph_name: str) -> sp.csr_matrix:
    if graph_name == "spatial":
        return fold_data.spatial_graph
    if graph_name == "bilateral":
        return fold_data.bilateral_graph
    raise ValueError(graph_name)


def select_readout_for_fold(fold_data: PspFold, X: np.ndarray, genes: list[str]) -> tuple[dict, pd.DataFrame]:
    psp_val_row = fast_summary_row(
        "psp_val",
        X,
        fold_data.psp_val,
        fold_data.val_idx,
        fold_data.low_val_idx,
        fold_data.high_val_idx,
        {"fold": fold_data.fold, "split": "val", "role": "base"},
    )
    candidates = []
    smooth_cache: dict[tuple[str, str, float], np.ndarray] = {}
    method_grid = [
        ("spatial", "rw", [1, 2, 3, 5]),
        ("spatial", "heat", [0.01, 0.05, 0.1, 0.2]),
        ("bilateral", "bilateral_rw", [1, 2, 3]),
    ]
    global_lams = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    bin_lows = [0.0, 0.05]
    bin_mids = [0.05, 0.1, 0.2]
    bin_highs = [0.2, 0.3, 0.5, 0.6]
    for graph_name, smooth_method, params in method_grid:
        W = candidate_graph(fold_data, graph_name)
        for param in params:
            key = (graph_name, smooth_method, float(param))
            smooth = smooth_cache.get(key)
            if smooth is None:
                smooth = graph_smooth(W, fold_data.psp_val, smooth_method, param)
                smooth_cache[key] = smooth
            for lam in global_lams:
                pred = apply_lambda_mode(fold_data.psp_val, smooth, fold_data.pred_sp_val, fold_data.sp_thresholds, "global", lam)
                row = fast_summary_row(
                    f"val_{graph_name}_{smooth_method}_{param}_global_{lam}",
                    X,
                    pred,
                    fold_data.val_idx,
                    fold_data.low_val_idx,
                    fold_data.high_val_idx,
                    {
                        "fold": fold_data.fold,
                        "split": "val",
                        "role": "candidate",
                        "graph": graph_name,
                        "smooth_method": smooth_method,
                        "smooth_param": param,
                        "lambda_mode": "global",
                        "lambda_spec": json.dumps(lam),
                    },
                )
                candidates.append(row)
            for lam_low in bin_lows:
                for lam_mid in bin_mids:
                    for lam_high in bin_highs:
                        lambdas = {"low": lam_low, "mid": lam_mid, "high": lam_high}
                        pred = apply_lambda_mode(fold_data.psp_val, smooth, fold_data.pred_sp_val, fold_data.sp_thresholds, "predicted_spatiality_bins", lambdas)
                        row = fast_summary_row(
                            f"val_{graph_name}_{smooth_method}_{param}_bins_{lam_low}_{lam_mid}_{lam_high}",
                            X,
                            pred,
                            fold_data.val_idx,
                            fold_data.low_val_idx,
                            fold_data.high_val_idx,
                            {
                                "fold": fold_data.fold,
                                "split": "val",
                                "role": "candidate",
                                "graph": graph_name,
                                "smooth_method": smooth_method,
                                "smooth_param": param,
                                "lambda_mode": "predicted_spatiality_bins",
                                "lambda_spec": json.dumps(lambdas, sort_keys=True),
                            },
                        )
                        candidates.append(row)
    val_df = pd.DataFrame(candidates)
    for metric in ["SPCC", "SSIM", "RMSE", "JS", "high_spatial_SPCC"]:
        val_df[f"delta_{metric}_vs_psp"] = val_df[metric].astype(float) - float(psp_val_row[metric])
    val_df["guard_pass"] = (
        (val_df["delta_SPCC_vs_psp"] >= -0.002)
        & (val_df["delta_RMSE_vs_psp"] <= 0.0015)
        & (val_df["delta_JS_vs_psp"] <= 0.0015)
        & (val_df["delta_high_spatial_SPCC_vs_psp"] >= -1e-12)
    )
    val_df["selection_score"] = val_df["SSIM"].astype(float)
    valid = val_df[val_df["guard_pass"]]
    if valid.empty:
        selected = val_df.sort_values("SSIM", ascending=False).iloc[0].to_dict()
        selected["selected_with_guard"] = False
    else:
        selected = valid.sort_values(["SSIM", "SPCC"], ascending=False).iloc[0].to_dict()
        selected["selected_with_guard"] = True
    return selected, pd.concat([pd.DataFrame([psp_val_row]), val_df], ignore_index=True)


def parse_lambda_spec(lambda_spec: str) -> dict[str, float] | float:
    obj = json.loads(lambda_spec)
    if isinstance(obj, dict):
        return {str(k): float(v) for k, v in obj.items()}
    return float(obj)


def apply_selected_readout(
    pred: np.ndarray,
    fold_data: PspFold,
    selected: dict,
    graph_override: sp.csr_matrix | None = None,
    pred_sp_override: np.ndarray | None = None,
    inverse_bins: bool = False,
) -> np.ndarray:
    W = graph_override if graph_override is not None else candidate_graph(fold_data, selected["graph"])
    smooth = graph_smooth(W, pred, selected["smooth_method"], selected["smooth_param"])
    lambda_spec = parse_lambda_spec(selected["lambda_spec"])
    if selected["lambda_mode"] == "global":
        return apply_lambda_mode(pred, smooth, fold_data.pred_sp_test, fold_data.sp_thresholds, "global", lambda_spec)
    lambdas = dict(lambda_spec)  # type: ignore[arg-type]
    if inverse_bins:
        lambdas = {"low": lambdas["high"], "mid": lambdas["mid"], "high": lambdas["low"]}
    pred_sp = fold_data.pred_sp_test if pred_sp_override is None else pred_sp_override
    return apply_lambda_mode(pred, smooth, pred_sp, fold_data.sp_thresholds, "predicted_spatiality_bins", lambdas)


def apply_selected_readout_external(
    pred: np.ndarray,
    fold_data: PspFold,
    selected: dict,
) -> np.ndarray:
    return apply_selected_readout(pred[:, fold_data.test_idx], fold_data, selected)


def evaluate_readout_fold(fold_data: PspFold, selected: dict, X: np.ndarray, genes: list[str], seed: int) -> tuple[list[dict], list[pd.DataFrame]]:
    rows, genes_rows = [], []
    base_row, base_gene = metric_row(
        "psp_base",
        X,
        fold_data.psp_test,
        fold_data.test_idx,
        fold_data.low_test_idx,
        fold_data.high_test_idx,
        genes,
        {"fold": fold_data.fold, "split": "test", "role": "base", "control": "base"},
    )
    rows.append(base_row)
    genes_rows.append(base_gene)
    correct = apply_selected_readout(fold_data.psp_test, fold_data, selected)
    row, gene_df = metric_row(
        "psp_spatiality_gated_readout_correct",
        X,
        correct,
        fold_data.test_idx,
        fold_data.low_test_idx,
        fold_data.high_test_idx,
        genes,
        {
            "fold": fold_data.fold,
            "split": "test",
            "role": "selected",
            "control": "correct",
            "graph": selected["graph"],
            "smooth_method": selected["smooth_method"],
            "smooth_param": selected["smooth_param"],
            "lambda_mode": selected["lambda_mode"],
            "lambda_spec": selected["lambda_spec"],
            "selected_with_guard": selected["selected_with_guard"],
        },
    )
    rows.append(row)
    genes_rows.append(gene_df)
    controls = []
    controls.append(("random_graph", random_graph_like(fold_data.spatial_graph.shape[0], 8, seed + fold_data.fold)))
    controls.append(("spot_permuted_graph", spot_permuted_graph(candidate_graph(fold_data, selected["graph"]), seed + 17 + fold_data.fold)))
    rng = np.random.default_rng(seed + 31 + fold_data.fold)
    controls_pred = []
    for name, graph in controls:
        controls_pred.append((name, apply_selected_readout(fold_data.psp_test, fold_data, selected, graph_override=graph)))
    controls_pred.append(
        (
            "shuffled_spatiality_bins",
            apply_selected_readout(
                fold_data.psp_test,
                fold_data,
                selected,
                pred_sp_override=fold_data.pred_sp_test[rng.permutation(len(fold_data.pred_sp_test))],
            ),
        )
    )
    controls_pred.append(("inverse_spatiality_bins", apply_selected_readout(fold_data.psp_test, fold_data, selected, inverse_bins=True)))
    W = candidate_graph(fold_data, selected["graph"])
    over_smooth = graph_smooth(W, fold_data.psp_test, "rw" if selected["graph"] == "spatial" else "bilateral_rw", 20)
    controls_pred.append(("over_smoothing_extreme", 0.2 * fold_data.psp_test + 0.8 * over_smooth))
    for name, pred in controls_pred:
        row, gene_df = metric_row(
            f"psp_spatiality_gated_readout_{name}_control",
            X,
            pred,
            fold_data.test_idx,
            fold_data.low_test_idx,
            fold_data.high_test_idx,
            genes,
            {
                "fold": fold_data.fold,
                "split": "test",
                "role": "control",
                "control": name,
                "graph": selected["graph"],
                "smooth_method": selected["smooth_method"],
                "smooth_param": selected["smooth_param"],
                "lambda_mode": selected["lambda_mode"],
                "lambda_spec": selected["lambda_spec"],
                "selected_with_guard": selected["selected_with_guard"],
            },
        )
        rows.append(row)
        genes_rows.append(gene_df)
    return rows, genes_rows


def add_deltas(df: pd.DataFrame, group_col: str = "fold") -> pd.DataFrame:
    out = df.copy()
    for fold, sub in out.groupby(group_col):
        if not sub["role"].eq("base").any():
            continue
        base = sub[sub["role"].eq("base")].iloc[0]
        for metric in ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]:
            out.loc[sub.index, f"delta_{metric}_vs_psp"] = sub[metric].astype(float) - float(base[metric])
    return out


def summarize_long(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]
    summary = df.groupby(["model", "role", "control"], as_index=False)[metrics].agg(["mean", "std", "median"])
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns.to_flat_index()]
    return summary


def external_diagnostic(
    fold_data_by_fold: dict[int, PspFold],
    selected_by_fold: dict[int, dict],
    X: np.ndarray,
    genes: list[str],
    args,
) -> pd.DataFrame:
    method_dirs = {
        "SpaIM": Path("/workspace/GeneSPT/results/strict_vis9a_spaim_gene5cv"),
        "Tangram": Path("/workspace/GeneSPT/results/strict_vis9a_tangram_gene5cv"),
        "SpaGE": Path("/workspace/GeneSPT/results/strict_vis9a_spage_gene5cv"),
        "stPlus": Path("/workspace/GeneSPT/results/strict_vis9a_stplus_gene5cv"),
        "TransPA": Path("/workspace/GeneSPT/results/strict_vis9a_transpa_gene5cv"),
    }
    rows = []
    for method, root in method_dirs.items():
        for fold, fold_data in fold_data_by_fold.items():
            pred_path = root / f"fold{fold}" / "imputed_expression.npy"
            if not pred_path.exists():
                rows.append({"method": method, "fold": fold, "status": "missing_prediction", "prediction_path": str(pred_path)})
                continue
            pred_full = np.load(pred_path).astype(np.float32)
            pred_test = pred_full[:, fold_data.test_idx]
            smoothed = apply_selected_readout_external(pred_full, fold_data, selected_by_fold[fold])
            base_row, _ = metric_row(
                f"{method}_original",
                X,
                pred_test,
                fold_data.test_idx,
                fold_data.low_test_idx,
                fold_data.high_test_idx,
                genes,
                {"method": method, "fold": fold, "status": "done", "variant": "original", "prediction_path": str(pred_path)},
            )
            smooth_row, _ = metric_row(
                f"{method}_same_readout",
                X,
                smoothed,
                fold_data.test_idx,
                fold_data.low_test_idx,
                fold_data.high_test_idx,
                genes,
                {"method": method, "fold": fold, "status": "done", "variant": "same_readout", "prediction_path": str(pred_path)},
            )
            for metric in ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]:
                smooth_row[f"delta_{metric}_vs_original"] = float(smooth_row[metric]) - float(base_row[metric])
            rows.extend([base_row, smooth_row])
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]
    rows = []
    for metric in metrics:
        piv = df[df["role"].isin(["base", "selected"])].pivot(index="fold", columns="role", values=metric).dropna()
        if piv.shape[0] < 2:
            continue
        diff = piv["selected"].to_numpy(float) - piv["base"].to_numpy(float)
        rows.append(
            {
                "metric": metric,
                "n_folds": int(piv.shape[0]),
                "mean_base": float(piv["base"].mean()),
                "mean_selected": float(piv["selected"].mean()),
                "mean_delta": float(diff.mean()),
                "median_delta": float(np.median(diff)),
                "paired_t_p": float(st.ttest_rel(piv["selected"], piv["base"]).pvalue),
                "wilcoxon_p": float(st.wilcoxon(diff).pvalue) if np.any(np.abs(diff) > 1e-12) else 1.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Spatial_count.txt"))
    ap.add_argument("--scrna-counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/scRNA_count.txt"))
    ap.add_argument("--locations-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Locations.txt"))
    ap.add_argument("--mask-dir", type=Path, default=INFO / "strict_whole_gene_masks")
    ap.add_argument("--out-dir", type=Path, default=INFO)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-base", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_counts, genes, _ = load_matrix(args.counts_path, index_col=None)
    X = log1p_cpm(X_counts)
    coords = pd.read_csv(args.locations_path, sep="\t").to_numpy(dtype=np.float32)
    edges = make_knn_edges(coords, k=8)
    X_sc_counts, sc_genes, _ = load_matrix(args.scrna_counts_path, index_col=0)
    if list(sc_genes) != list(genes):
        sc_map = {g: i for i, g in enumerate(sc_genes)}
        X_sc_counts = X_sc_counts[:, [sc_map[g] for g in genes]]
    X_sc = log1p_cpm(X_sc_counts)
    desc = build_descriptors(X_sc, pca_dims=[32], nmf_dims=[32], seed=args.seed)
    desc["pca32_nmf32"] = np.concatenate([desc["pca32"], desc["nmf32"]], axis=1).astype(np.float32)

    all_test_rows, all_gene_rows, all_val_rows = [], [], []
    fold_data_by_fold, selected_by_fold = {}, {}
    for fold in args.folds:
        print(f"[Readout] fold{fold}: building PSP predictions", flush=True)
        fold_data = build_psp_fold(fold, X, genes, coords, edges, desc, args, device)
        fold_data_by_fold[fold] = fold_data
        selected, val_df = select_readout_for_fold(fold_data, X, genes)
        selected_by_fold[fold] = selected
        all_val_rows.append(val_df)
        print(
            f"[Readout] fold{fold}: selected {selected['graph']} {selected['smooth_method']} {selected['smooth_param']} "
            f"{selected['lambda_mode']} {selected['lambda_spec']} guard={selected['selected_with_guard']}",
            flush=True,
        )
        rows, gene_rows = evaluate_readout_fold(fold_data, selected, X, genes, args.seed)
        all_test_rows.extend(rows)
        all_gene_rows.extend(gene_rows)

    long_df = add_deltas(pd.DataFrame(all_test_rows))
    gene_df = pd.concat(all_gene_rows, ignore_index=True)
    val_df = pd.concat(all_val_rows, ignore_index=True)
    summary = summarize_long(long_df)
    paired = paired_tests(long_df)

    long_path = args.out_dir / "psp_spatiality_gated_readout_5fold_long.csv"
    summary_path = args.out_dir / "psp_spatiality_gated_readout_5fold_summary.csv"
    decision_path = args.out_dir / "psp_spatiality_gated_readout_decision.md"
    gene_path = args.out_dir / "psp_spatiality_gated_readout_5fold_gene_level.csv"
    val_path = args.out_dir / "psp_spatiality_gated_readout_val_selection_long.csv"
    paired_path = args.out_dir / "psp_spatiality_gated_readout_5fold_paired_tests.csv"
    long_df.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    gene_df.to_csv(gene_path, index=False)
    val_df.to_csv(val_path, index=False)
    paired.to_csv(paired_path, index=False)

    external_df = external_diagnostic(fold_data_by_fold, selected_by_fold, X, genes, args)
    external_path = args.out_dir / "external_baseline_same_smoothing_diagnostic.csv"
    external_df.to_csv(external_path, index=False)

    selected_summary = summary[summary["role"].eq("selected")].iloc[0]
    base_summary = summary[summary["role"].eq("base")].iloc[0]
    control_summary = summary[summary["role"].eq("control")]
    delta_ssim = float(selected_summary["SSIM_mean"] - base_summary["SSIM_mean"])
    delta_spcc = float(selected_summary["SPCC_mean"] - base_summary["SPCC_mean"])
    delta_rmse = float(selected_summary["RMSE_mean"] - base_summary["RMSE_mean"])
    delta_js = float(selected_summary["JS_mean"] - base_summary["JS_mean"])
    delta_hs = float(selected_summary["high_spatial_SPCC_mean"] - base_summary["high_spatial_SPCC_mean"])
    raw_guard = bool(delta_spcc >= -0.002 and delta_rmse <= 0.0015 and delta_js <= 0.0015 and delta_hs >= -1e-12)
    guarded_controls = control_summary[
        (control_summary["SPCC_mean"] - base_summary["SPCC_mean"] >= -0.002)
        & (control_summary["RMSE_mean"] - base_summary["RMSE_mean"] <= 0.0015)
        & (control_summary["JS_mean"] - base_summary["JS_mean"] <= 0.0015)
        & (control_summary["high_spatial_SPCC_mean"] - base_summary["high_spatial_SPCC_mean"] >= -1e-12)
    ]
    controls_ok = bool(guarded_controls.empty or selected_summary["SSIM_mean"] > guarded_controls["SSIM_mean"].max())
    if delta_ssim >= 0.004 and raw_guard and controls_ok:
        decision = "PSP_READOUT_CONTINUE"
    elif delta_ssim > 0 and controls_ok:
        decision = "PSP_READOUT_AUXILIARY"
    else:
        decision = "PSP_READOUT_FAILED"

    # Generic smoothing diagnostic summary.
    ext_done = external_df[external_df.get("status", "").eq("done")] if "status" in external_df else pd.DataFrame()
    ext_summary = pd.DataFrame()
    if not ext_done.empty and "variant" in ext_done:
        smooth = ext_done[ext_done["variant"].eq("same_readout")]
        ext_summary = smooth.groupby("method", as_index=False)[
            ["delta_SPCC_vs_original", "delta_SSIM_vs_original", "delta_RMSE_vs_original", "delta_JS_vs_original", "delta_high_spatial_SPCC_vs_original"]
        ].mean()
        ext_summary.to_csv(args.out_dir / "external_baseline_same_smoothing_diagnostic_summary.csv", index=False)

    decision_path.write_text(
        "\n".join(
            [
                "# PSP Spatiality-Gated Graph Diffusion Readout Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "## Gate Result",
                f"delta_SSIM_mean = {delta_ssim:.6f}",
                f"delta_SPCC_mean = {delta_spcc:.6f}",
                f"delta_RMSE_mean = {delta_rmse:.6f}",
                f"delta_JS_mean = {delta_js:.6f}",
                f"delta_high_spatial_SPCC_mean = {delta_hs:.6f}",
                f"raw_guard = `{raw_guard}`",
                f"controls_ok = `{controls_ok}`",
                "",
                "## Summary",
                summary[["model", "role", "control", "SPCC_mean", "SSIM_mean", "RMSE_mean", "JS_mean", "high_spatial_SPCC_mean"]].to_string(index=False),
                "",
                "## External Same-Smoothing Diagnostic",
                ext_summary.to_string(index=False) if not ext_summary.empty else "External smoothing diagnostic unavailable.",
                "",
                "## Outputs",
                f"- {long_path}",
                f"- {summary_path}",
                f"- {paired_path}",
                f"- {gene_path}",
                f"- {val_path}",
                f"- {external_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary[["model", "SPCC_mean", "SSIM_mean", "RMSE_mean", "JS_mean", "high_spatial_SPCC_mean"]].to_string(index=False))
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
