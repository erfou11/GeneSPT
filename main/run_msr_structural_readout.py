#!/usr/bin/env python3
"""Multi-scale Structural Readout (MSR) gate for GC-PSP predictions.

No training of the main model is performed. The script reconstructs cached
GC-MLP/PSP predictions, audits SSIM components, builds a multi-scale transform
bank, checks a val-gene oracle upper bound, and only then fits a simple
validation-selected readout-choice predictor.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.stats as st
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from run_psp_spatiality_gated_readout import (
    build_psp_fold,
    bilateral_graph,
    graph_smooth,
    knn_weight_graph,
    random_graph_like,
)
from run_gc_spatiality_aware_training import compute_spatiality
from run_predictable_spatial_program_folds012 import load_or_train_base_cached, summarize_model
from run_predictable_spatial_program_folds012 import apply_bin_lambdas, fit_svd_raw_basis
from run_predictable_spatial_program_transfer_fold0 import (
    bin_lambdas_from_val,
    component_stats,
    fit_spatiality_predictor,
    selected_component_prediction,
)
from run_st_spatial_program_decoder_fold0 import fit_predict_coeff, preprocess_train, project_coeff
from run_strict_gene_conditioned_decoder_gate import (
    build_descriptors,
    gene_metrics,
    load_matrix,
    log1p_cpm,
    make_knn_edges,
    moran_i,
    subgroup_indices,
    summarize_gene_df,
)


INFO = Path("/workspace/GeneSPT/results/imformation")
EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ssim_components(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    y = y / max(float(np.nanmax(y)), EPS)
    p = p / max(float(np.nanmax(p)), EPS)
    L = max(float(np.nanmax(y)), float(np.nanmax(p)), EPS)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2.0
    muy = float(np.mean(y))
    mup = float(np.mean(p))
    sigy = float(np.sqrt(np.mean((y - muy) ** 2)))
    sigp = float(np.sqrt(np.mean((p - mup) ** 2)))
    cov = float(np.mean((y - muy) * (p - mup)))
    lum = (2 * muy * mup + C1) / (muy * muy + mup * mup + C1)
    con = (2 * sigy * sigp + C2) / (sigy * sigy + sigp * sigp + C2)
    stru = (cov + C3) / (sigy * sigp + C3)
    return float(lum), float(con), float(stru), float(lum * con * stru)


def row_normalized(W: sp.csr_matrix) -> sp.csr_matrix:
    P = W.tocsr() + sp.eye(W.shape[0], format="csr")
    deg = np.asarray(P.sum(axis=1)).reshape(-1)
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg > 0)
    return sp.diags(inv).dot(P).tocsr()


def local_stats_errors(y: np.ndarray, p: np.ndarray, P: sp.csr_matrix) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    my = P.dot(y)
    mp = P.dot(p)
    vy = np.maximum(P.dot(y * y) - my * my, 0.0)
    vp = np.maximum(P.dot(p * p) - mp * mp, 0.0)
    cov = P.dot(y * p) - my * mp
    struct = cov / np.maximum(np.sqrt(vy * vp), EPS)
    return {
        "local_mean_abs_error": float(np.mean(np.abs(my - mp))),
        "local_variance_abs_error": float(np.mean(np.abs(vy - vp))),
        "local_covariance_mean": float(np.mean(cov)),
        "local_structure_mean": float(np.mean(struct)),
        "local_contrast_pred": float(np.mean(np.sqrt(vp))),
        "local_contrast_true": float(np.mean(np.sqrt(vy))),
    }


def graph_smoothness(y: np.ndarray, edges: np.ndarray) -> float:
    if edges.size == 0:
        return float("nan")
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((y[edges[:, 0]] - y[edges[:, 1]]) ** 2))


def audit_model_components(
    model: str,
    fold: int,
    X: np.ndarray,
    pred_sub: np.ndarray,
    gene_idx: np.ndarray,
    genes: list[str],
    P: sp.csr_matrix,
    edges: np.ndarray,
) -> list[dict]:
    rows = []
    for pos, g in enumerate(gene_idx):
        y = X[:, int(g)]
        p = pred_sub[:, pos]
        lum, con, stru, ssim = ssim_components(y, p)
        row = {
            "model": model,
            "fold": fold,
            "gene_idx": int(g),
            "gene": genes[int(g)],
            "ssim_luminance": lum,
            "ssim_contrast": con,
            "ssim_structure": stru,
            "SSIM": ssim,
            "graph_smoothness_pred": graph_smoothness(p, edges),
            "graph_smoothness_true": graph_smoothness(y, edges),
            "pred_moranI": moran_i(p, edges),
            "true_moranI": moran_i(y, edges),
        }
        row.update(local_stats_errors(y, p, P))
        rows.append(row)
    return rows


def rankdata_corr(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    Yr = st.rankdata(Y, axis=0)
    Pr = st.rankdata(P, axis=0)
    Yc = Yr - Yr.mean(axis=0, keepdims=True)
    Pc = Pr - Pr.mean(axis=0, keepdims=True)
    return np.sum(Yc * Pc, axis=0) / np.sqrt(np.maximum(np.sum(Yc * Yc, axis=0) * np.sum(Pc * Pc, axis=0), EPS))


def fast_gene_metrics(Y: np.ndarray, P: np.ndarray) -> dict[str, np.ndarray]:
    Y = np.asarray(Y, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    spcc = rankdata_corr(Y, P)
    Yz = (Y - Y.mean(axis=0, keepdims=True)) / np.maximum(Y.std(axis=0, keepdims=True), EPS)
    Pz = (P - P.mean(axis=0, keepdims=True)) / np.maximum(P.std(axis=0, keepdims=True), EPS)
    rmse = np.sqrt(np.mean((Yz - Pz) ** 2, axis=0))
    Ys = Y / np.maximum(Y.sum(axis=0, keepdims=True), EPS)
    Ps = P / np.maximum(P.sum(axis=0, keepdims=True), EPS)
    M = 0.5 * (Ys + Ps)
    js = 0.5 * np.sum(np.where(Ys > 0, Ys * np.log(np.maximum(Ys, EPS) / np.maximum(M, EPS)), 0.0), axis=0)
    js += 0.5 * np.sum(np.where(Ps > 0, Ps * np.log(np.maximum(Ps, EPS) / np.maximum(M, EPS)), 0.0), axis=0)
    ssim = np.asarray([ssim_components(Y[:, j], P[:, j])[3] for j in range(Y.shape[1])], dtype=np.float64)
    return {"SPCC": spcc, "RMSE": rmse, "JS": js, "SSIM": ssim}


def transform_bank(pred: np.ndarray, program_pred: np.ndarray, W: sp.csr_matrix, W_bi: sp.csr_matrix) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"identity": np.clip(pred, 0.0, None).astype(np.float32)}
    rw1 = graph_smooth(W, pred, "rw", 1)
    rw2 = graph_smooth(W, pred, "rw", 2)
    rw3 = graph_smooth(W, pred, "rw", 3)
    rw5 = graph_smooth(W, pred, "rw", 5)
    out.update({"rw1": rw1, "rw2": rw2, "rw3": rw3, "rw5": rw5})
    out["bilateral_rw2"] = graph_smooth(W_bi, pred, "bilateral_rw", 2)
    local_mean = rw1
    for alpha in [1.1, 1.25, 1.5]:
        out[f"contrast_{alpha:g}"] = np.clip(local_mean + alpha * (pred - local_mean), 0.0, None).astype(np.float32)
    for beta in [0.1, 0.25, 0.5]:
        out[f"unsharp_{beta:g}"] = np.clip(pred + beta * (pred - rw1), 0.0, None).astype(np.float32)
    out["program_pred"] = np.clip(program_pred, 0.0, None).astype(np.float32)
    # Convex mixtures with identity.
    base_items = [(k, v) for k, v in out.items() if k != "identity"]
    for name, arr in base_items:
        for lam in [0.25, 0.5, 0.75, 1.0]:
            out[f"mix_{name}_{lam:g}"] = np.clip((1.0 - lam) * pred + lam * arr, 0.0, None).astype(np.float32)
    # A small three-way structural mixture.
    out["mix_smooth_contrast"] = np.clip(0.5 * pred + 0.25 * rw2 + 0.25 * out["contrast_1.25"], 0.0, None).astype(np.float32)
    return out


def build_fold_payload(fold: int, X: np.ndarray, genes: list[str], coords: np.ndarray, edges: np.ndarray, desc: dict[str, np.ndarray], args, device) -> dict:
    psp_fold = build_psp_fold(fold, X, genes, coords, edges, desc, args, device)
    train_idx = psp_fold.train_idx
    val_idx = psp_fold.val_idx
    test_idx = psp_fold.test_idx
    base, _ = load_or_train_base_cached(X, desc["pca32"], train_idx, val_idx, test_idx, device, args, fold)
    D = desc["pca32_nmf32"]
    basis = fit_svd_raw_basis(X[:, train_idx], k=64, seed=args.seed + fold)
    X_val_proc, val_meta = preprocess_train(X[:, val_idx], "raw")
    C_val_oracle = project_coeff(basis.A, X_val_proc)
    C_val_pred = fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[val_idx], 10.0, seed=args.seed + fold)
    comp = component_stats(C_val_pred, C_val_oracle)
    comp["rank_score"] = comp["component_spearman"].fillna(-1.0) * np.log1p(comp["oracle_coeff_var"].clip(lower=0))
    keep = comp.sort_values("rank_score", ascending=False).head(min(32, basis.k))["component"].to_numpy(dtype=np.int64)
    program_train = selected_component_prediction(
        basis.A,
        fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[train_idx], 10.0, seed=args.seed + 77 + fold),
        basis.meta,
        keep,
    )
    program_val = selected_component_prediction(basis.A, C_val_pred, val_meta, keep)
    program_test = selected_component_prediction(
        basis.A,
        fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[test_idx], 10.0, seed=args.seed + 99 + fold),
        basis.meta,
        keep,
    )
    train_sp = compute_spatiality(X, train_idx, edges)
    train_moran = train_sp["MoranI"].to_numpy(dtype=np.float32)
    pred_sp_train = fit_spatiality_predictor(D[train_idx], train_moran, D[train_idx])
    pred_sp_val = fit_spatiality_predictor(D[train_idx], train_moran, D[val_idx])
    pred_sp_test = fit_spatiality_predictor(D[train_idx], train_moran, D[test_idx])
    lambdas, _, _ = bin_lambdas_from_val(X, base["val"], program_val, val_idx, psp_fold.low_val_idx, psp_fold.high_val_idx, genes, pred_sp_val)
    q1, q2 = np.quantile(pred_sp_val, [1 / 3, 2 / 3])
    psp_train = apply_bin_lambdas(base["train"], program_train, pred_sp_train, (q1, q2), lambdas)
    psp_val = apply_bin_lambdas(base["val"], program_val, pred_sp_val, (q1, q2), lambdas)
    psp_test = apply_bin_lambdas(base["test"], program_test, pred_sp_test, (q1, q2), lambdas)
    return {
        "fold": fold,
        "psp": psp_fold,
        "gc_val": base["val"],
        "gc_test": base["test"],
        "psp_train": psp_train,
        "psp_val": psp_val,
        "psp_test": psp_test,
        "program_train": program_train,
        "program_val": program_val,
        "program_test": program_test,
        "pred_sp_train": pred_sp_train,
        "pred_sp_val": pred_sp_val,
        "pred_sp_test": pred_sp_test,
        "psp_lambdas": lambdas,
        "psp_spatiality_thresholds": (float(q1), float(q2)),
        "W": psp_fold.spatial_graph,
        "W_bi": psp_fold.bilateral_graph,
    }


def oracle_select(Y: np.ndarray, base_pred: np.ndarray, bank: dict[str, np.ndarray], guard: bool = True) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    base_m = fast_gene_metrics(Y, base_pred)
    best_pred = base_pred.copy()
    best_name = np.full(Y.shape[1], "identity", dtype=object)
    best_ssim = base_m["SSIM"].copy()
    rows = []
    for name, pred in bank.items():
        m = fast_gene_metrics(Y, pred)
        if guard:
            ok = (
                (m["SPCC"] - base_m["SPCC"] >= -0.002)
                & (m["RMSE"] - base_m["RMSE"] <= 0.0015)
                & (m["JS"] - base_m["JS"] <= 0.0015)
            )
        else:
            ok = np.ones(Y.shape[1], dtype=bool)
        better = ok & (m["SSIM"] > best_ssim)
        best_pred[:, better] = pred[:, better]
        best_name[better] = name
        best_ssim[better] = m["SSIM"][better]
        rows.append(
            {
                "transform": name,
                "SSIM_median": float(np.nanmedian(m["SSIM"])),
                "SPCC_median": float(np.nanmedian(m["SPCC"])),
                "RMSE_median": float(np.nanmedian(m["RMSE"])),
                "JS_median": float(np.nanmedian(m["JS"])),
                "guard_gene_fraction": float(np.mean(ok)),
                "chosen_gene_fraction": float(np.mean(better)),
            }
        )
    return best_pred, best_name, pd.DataFrame(rows)


def feature_matrix(desc: dict[str, np.ndarray], D_name: str, idx: np.ndarray, pred: np.ndarray, pred_sp: np.ndarray, edges: np.ndarray) -> np.ndarray:
    D = desc[D_name][idx].astype(np.float32)
    stats = []
    for j in range(pred.shape[1]):
        x = pred[:, j]
        stats.append(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.max(x)),
                float(np.mean(x <= 1e-6)),
                graph_smoothness(x, edges),
                moran_i(x, edges),
                float(pred_sp[j]) if pred_sp is not None and len(pred_sp) == pred.shape[1] else 0.0,
            ]
        )
    X_feat = np.concatenate([D, np.asarray(stats, dtype=np.float32)], axis=1)
    return np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class ConstantChoice:
    def __init__(self, value: str):
        self.value = value

    def predict(self, X):
        return np.full(X.shape[0], self.value, dtype=object)


def fit_choice_model(X_train: np.ndarray, y_train: np.ndarray):
    classes = np.unique(y_train)
    if len(classes) == 1:
        return ConstantChoice(str(classes[0]))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced"),
    )
    model.fit(X_train, y_train)
    return model


def apply_choices(bank: dict[str, np.ndarray], choices: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    pred = fallback.copy()
    for name in np.unique(choices):
        if name not in bank:
            continue
        mask = choices == name
        pred[:, mask] = bank[name][:, mask]
    return np.clip(pred, 0.0, None).astype(np.float32)


def row_from_pred(model: str, X: np.ndarray, pred: np.ndarray, idx: np.ndarray, low_idx: np.ndarray, high_idx: np.ndarray, genes: list[str], extra: dict):
    return summarize_model(model, X, pred, idx, low_idx, high_idx, genes, extra)


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for fold, sub in out.groupby("fold"):
        base = sub[sub["role"].eq("base")].iloc[0]
        for metric in ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC"]:
            out.loc[sub.index, f"delta_{metric}_vs_psp"] = sub[metric].astype(float) - float(base[metric])
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC"]
    s = df.groupby(["model", "role", "control"], as_index=False)[metrics].agg(["mean", "std", "median"])
    s.columns = ["_".join(c).rstrip("_") for c in s.columns.to_flat_index()]
    return s


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
    P = row_normalized(knn_weight_graph(coords, k=8))
    X_sc_counts, sc_genes, _ = load_matrix(args.scrna_counts_path, index_col=0)
    if list(sc_genes) != list(genes):
        sc_map = {g: i for i, g in enumerate(sc_genes)}
        X_sc_counts = X_sc_counts[:, [sc_map[g] for g in genes]]
    X_sc = log1p_cpm(X_sc_counts)
    desc = build_descriptors(X_sc, pca_dims=[32], nmf_dims=[32], seed=args.seed)
    desc["pca32_nmf32"] = np.concatenate([desc["pca32"], desc["nmf32"]], axis=1).astype(np.float32)

    payloads = {}
    audit_rows = []
    for fold in args.folds:
        print(f"[MSR] fold{fold}: building cached PSP payload", flush=True)
        payload = build_fold_payload(fold, X, genes, coords, edges, desc, args, device)
        payloads[fold] = payload
        psp = payload["psp"]
        audit_rows.extend(audit_model_components("GC-MLP", fold, X, payload["gc_test"], psp.test_idx, genes, P, edges))
        audit_rows.extend(audit_model_components("GeneSPT-GC-PSP", fold, X, payload["psp_test"], psp.test_idx, genes, P, edges))
        for method, root in {
            "SpaIM": "/workspace/GeneSPT/results/strict_vis9a_spaim_gene5cv",
            "Tangram": "/workspace/GeneSPT/results/strict_vis9a_tangram_gene5cv",
            "SpaGE": "/workspace/GeneSPT/results/strict_vis9a_spage_gene5cv",
            "stPlus": "/workspace/GeneSPT/results/strict_vis9a_stplus_gene5cv",
            "TransPA": "/workspace/GeneSPT/results/strict_vis9a_transpa_gene5cv",
        }.items():
            pred_path = Path(root) / f"fold{fold}" / "imputed_expression.npy"
            if pred_path.exists():
                pred_full = np.load(pred_path).astype(np.float32)
                audit_rows.extend(audit_model_components(method, fold, X, pred_full[:, psp.test_idx], psp.test_idx, genes, P, edges))
    audit_df = pd.DataFrame(audit_rows)
    audit_path = args.out_dir / "psp_ssim_component_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    audit_summary = (
        audit_df.groupby("model", as_index=False)[
            ["ssim_luminance", "ssim_contrast", "ssim_structure", "SSIM", "local_mean_abs_error", "local_variance_abs_error", "pred_moranI"]
        ]
        .median()
        .sort_values("SSIM", ascending=False)
    )
    audit_report = args.out_dir / "psp_ssim_component_audit_report.md"
    audit_report.write_text(
        "# PSP SSIM Component Audit\n\n"
        + audit_summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )

    transform_audit_rows = []
    oracle_rows = []
    oracle_pred_rows = []
    oracle_pass_any = False
    predictive_rows = []
    predictive_gene_rows = []
    external_rows = []

    for fold, payload in payloads.items():
        psp = payload["psp"]
        print(f"[MSR] fold{fold}: transform bank and val oracle", flush=True)
        bank_train = transform_bank(payload["psp_train"], payload["program_train"], payload["W"], payload["W_bi"])
        bank_val = transform_bank(payload["psp_val"], payload["program_val"], payload["W"], payload["W_bi"])
        bank_test = transform_bank(payload["psp_test"], payload["program_test"], payload["W"], payload["W_bi"])
        transform_audit_rows.extend(
            [{"fold": fold, "transform": k, "shape": str(v.shape), "pred_min": float(np.min(v)), "pred_max": float(np.max(v))} for k, v in bank_val.items()]
        )
        train_oracle_pred, train_oracle_choice, train_oracle_audit = oracle_select(X[:, psp.train_idx], payload["psp_train"], bank_train, guard=True)
        val_oracle_pred, val_oracle_choice, val_oracle_audit = oracle_select(X[:, psp.val_idx], payload["psp_val"], bank_val, guard=True)
        val_oracle_audit.insert(0, "fold", fold)
        oracle_rows.append(val_oracle_audit)
        base_row, _ = row_from_pred(
            "psp_base_val",
            X,
            payload["psp_val"],
            psp.val_idx,
            psp.low_val_idx,
            psp.high_val_idx,
            genes,
            {"fold": fold, "split": "val", "role": "base", "control": "base"},
        )
        oracle_row, _ = row_from_pred(
            "msr_val_oracle",
            X,
            val_oracle_pred,
            psp.val_idx,
            psp.low_val_idx,
            psp.high_val_idx,
            genes,
            {"fold": fold, "split": "val", "role": "oracle", "control": "oracle"},
        )
        for metric in ["SPCC", "SSIM", "RMSE", "JS", "high_spatial_SPCC"]:
            oracle_row[f"delta_{metric}_vs_psp"] = float(oracle_row[metric]) - float(base_row[metric])
        oracle_pred_rows.extend([base_row, oracle_row])
        if (
            oracle_row["delta_SSIM_vs_psp"] >= 0.004
            and oracle_row["delta_SPCC_vs_psp"] >= -0.002
            and oracle_row["delta_RMSE_vs_psp"] <= 0.0015
            and oracle_row["delta_JS_vs_psp"] <= 0.0015
        ):
            oracle_pass_any = True

        # Predictive readout: run regardless, but decision will respect oracle.
        Xtr = feature_matrix(desc, "pca32_nmf32", psp.train_idx, payload["psp_train"], payload["pred_sp_train"], edges)
        Xva = feature_matrix(desc, "pca32_nmf32", psp.val_idx, payload["psp_val"], payload["pred_sp_val"], edges)
        Xte = feature_matrix(desc, "pca32_nmf32", psp.test_idx, payload["psp_test"], payload["pred_sp_test"], edges)
        model = fit_choice_model(Xtr, train_oracle_choice)
        val_choice = model.predict(Xva)
        val_pred = apply_choices(bank_val, val_choice, payload["psp_val"])
        val_row, _ = row_from_pred(
            "msr_predictive_val",
            X,
            val_pred,
            psp.val_idx,
            psp.low_val_idx,
            psp.high_val_idx,
            genes,
            {"fold": fold, "split": "val", "role": "predictive_val", "control": "correct"},
        )
        test_choice = model.predict(Xte)
        test_pred = apply_choices(bank_test, test_choice, payload["psp_test"])
        base_test_row, base_gene = row_from_pred(
            "psp_base",
            X,
            payload["psp_test"],
            psp.test_idx,
            psp.low_test_idx,
            psp.high_test_idx,
            genes,
            {"fold": fold, "split": "test", "role": "base", "control": "base"},
        )
        test_row, test_gene = row_from_pred(
            "msr_predictive_correct",
            X,
            test_pred,
            psp.test_idx,
            psp.low_test_idx,
            psp.high_test_idx,
            genes,
            {"fold": fold, "split": "test", "role": "selected", "control": "correct"},
        )
        predictive_rows.extend([base_test_row, test_row])
        predictive_gene_rows.extend([base_gene, test_gene])

        rng = np.random.default_rng(args.seed + fold)
        controls = {
            "random_transform_choice": rng.choice(np.array(list(bank_test.keys()), dtype=object), size=len(psp.test_idx)),
            "shuffled_gene_descriptors": fit_choice_model(
                Xtr[rng.permutation(Xtr.shape[0])],
                train_oracle_choice,
            ).predict(Xte),
            "random_spatiality_score": model.predict(Xte.copy()),
        }
        for name, choice in controls.items():
            if name == "random_spatiality_score":
                Xte_ctrl = Xte.copy()
                Xte_ctrl[:, -1] = rng.normal(size=Xte_ctrl.shape[0])
                choice = model.predict(Xte_ctrl)
            pred_ctrl = apply_choices(bank_test, choice, payload["psp_test"])
            row, gene_df = row_from_pred(
                f"msr_{name}_control",
                X,
                pred_ctrl,
                psp.test_idx,
                psp.low_test_idx,
                psp.high_test_idx,
                genes,
                {"fold": fold, "split": "test", "role": "control", "control": name},
            )
            predictive_rows.append(row)
            predictive_gene_rows.append(gene_df)
        # Random graph bank control.
        W_rand = random_graph_like(payload["W"].shape[0], 8, args.seed + fold)
        bank_rand = transform_bank(payload["psp_test"], payload["program_test"], W_rand, W_rand)
        pred_rand_bank = apply_choices(bank_rand, test_choice, payload["psp_test"])
        row, gene_df = row_from_pred(
            "msr_random_graph_transform_bank_control",
            X,
            pred_rand_bank,
            psp.test_idx,
            psp.low_test_idx,
            psp.high_test_idx,
            genes,
            {"fold": fold, "split": "test", "role": "control", "control": "random_graph_transform_bank"},
        )
        predictive_rows.append(row)
        predictive_gene_rows.append(gene_df)

        # External diagnostic applies the same predicted readout classifier.
        for method, root in {
            "SpaIM": "/workspace/GeneSPT/results/strict_vis9a_spaim_gene5cv",
            "Tangram": "/workspace/GeneSPT/results/strict_vis9a_tangram_gene5cv",
            "SpaGE": "/workspace/GeneSPT/results/strict_vis9a_spage_gene5cv",
            "stPlus": "/workspace/GeneSPT/results/strict_vis9a_stplus_gene5cv",
            "TransPA": "/workspace/GeneSPT/results/strict_vis9a_transpa_gene5cv",
        }.items():
            pred_path = Path(root) / f"fold{fold}" / "imputed_expression.npy"
            if not pred_path.exists():
                continue
            ext_full = np.load(pred_path).astype(np.float32)
            ext_test = ext_full[:, psp.test_idx]
            bank_ext = transform_bank(ext_test, ext_test, payload["W"], payload["W_bi"])
            ext_feat = feature_matrix(desc, "pca32_nmf32", psp.test_idx, ext_test, payload["pred_sp_test"], edges)
            ext_choice = model.predict(ext_feat)
            ext_read = apply_choices(bank_ext, ext_choice, ext_test)
            orig_row, _ = row_from_pred(
                f"{method}_original",
                X,
                ext_test,
                psp.test_idx,
                psp.low_test_idx,
                psp.high_test_idx,
                genes,
                {"method": method, "fold": fold, "variant": "original"},
            )
            read_row, _ = row_from_pred(
                f"{method}_msr",
                X,
                ext_read,
                psp.test_idx,
                psp.low_test_idx,
                psp.high_test_idx,
                genes,
                {"method": method, "fold": fold, "variant": "msr_readout"},
            )
            for metric in ["SPCC", "SSIM", "RMSE", "JS", "high_spatial_SPCC"]:
                read_row[f"delta_{metric}_vs_original"] = float(read_row[metric]) - float(orig_row[metric])
            external_rows.extend([orig_row, read_row])

    pd.DataFrame(transform_audit_rows).to_csv(args.out_dir / "msr_transform_bank_audit.csv", index=False)
    oracle_audit_df = pd.concat(oracle_rows, ignore_index=True)
    oracle_audit_df.to_csv(args.out_dir / "msr_val_oracle_transform_audit.csv", index=False)
    oracle_pred_df = pd.DataFrame(oracle_pred_rows)
    oracle_pred_df = add_deltas(oracle_pred_df)
    oracle_pred_df.to_csv(args.out_dir / "msr_val_oracle_upper_bound.csv", index=False)
    oracle_summary = summarize(oracle_pred_df)
    oracle_summary.to_csv(args.out_dir / "msr_val_oracle_summary.csv", index=False)

    predictive_df = add_deltas(pd.DataFrame(predictive_rows))
    predictive_summary = summarize(predictive_df)
    predictive_gene = pd.concat(predictive_gene_rows, ignore_index=True)
    predictive_df.to_csv(args.out_dir / "msr_predictive_readout_5fold_long.csv", index=False)
    predictive_summary.to_csv(args.out_dir / "msr_predictive_readout_5fold_summary.csv", index=False)
    predictive_gene.to_csv(args.out_dir / "msr_predictive_readout_5fold_gene_level.csv", index=False)

    external_df = pd.DataFrame(external_rows)
    external_df.to_csv(args.out_dir / "msr_external_baseline_diagnostic.csv", index=False)
    if not external_df.empty:
        ext_delta = external_df[external_df["variant"].eq("msr_readout")].groupby("method", as_index=False)[
            ["delta_SPCC_vs_original", "delta_SSIM_vs_original", "delta_RMSE_vs_original", "delta_JS_vs_original", "delta_high_spatial_SPCC_vs_original"]
        ].mean()
        ext_delta.to_csv(args.out_dir / "msr_external_baseline_diagnostic_summary.csv", index=False)
    else:
        ext_delta = pd.DataFrame()

    oracle_sel = oracle_summary[oracle_summary["role"].eq("oracle")].iloc[0]
    oracle_base = oracle_summary[oracle_summary["role"].eq("base")].iloc[0]
    oracle_delta_ssim = float(oracle_sel["SSIM_mean"] - oracle_base["SSIM_mean"])
    oracle_delta_spcc = float(oracle_sel["SPCC_mean"] - oracle_base["SPCC_mean"])
    oracle_delta_rmse = float(oracle_sel["RMSE_mean"] - oracle_base["RMSE_mean"])
    oracle_delta_js = float(oracle_sel["JS_mean"] - oracle_base["JS_mean"])
    oracle_pass = bool(oracle_delta_ssim >= 0.004 and oracle_delta_spcc >= -0.002 and oracle_delta_rmse <= 0.0015 and oracle_delta_js <= 0.0015)

    pred_sel = predictive_summary[predictive_summary["role"].eq("selected")].iloc[0]
    pred_base = predictive_summary[predictive_summary["role"].eq("base")].iloc[0]
    pred_ctrl = predictive_summary[predictive_summary["role"].eq("control")]
    d_ssim = float(pred_sel["SSIM_mean"] - pred_base["SSIM_mean"])
    d_spcc = float(pred_sel["SPCC_mean"] - pred_base["SPCC_mean"])
    d_rmse = float(pred_sel["RMSE_mean"] - pred_base["RMSE_mean"])
    d_js = float(pred_sel["JS_mean"] - pred_base["JS_mean"])
    d_hs = float(pred_sel["high_spatial_SPCC_mean"] - pred_base["high_spatial_SPCC_mean"])
    controls_fail = bool(pred_ctrl.empty or pred_sel["SSIM_mean"] > pred_ctrl["SSIM_mean"].max())
    if d_ssim >= 0.004 and d_spcc >= -0.002 and d_rmse <= 0.0015 and d_js <= 0.0015 and d_hs >= -1e-12 and controls_fail:
        decision = "MSR_CONTINUE"
    elif d_ssim > 0 and controls_fail:
        decision = "MSR_AUXILIARY"
    else:
        decision = "MSR_FAILED"

    (args.out_dir / "msr_val_oracle_decision.md").write_text(
        "\n".join(
            [
                "# MSR Val Oracle Decision",
                "",
                f"Decision: `{'MSR_ORACLE_PASS' if oracle_pass else 'MSR_ORACLE_FAILED'}`",
                f"delta_SSIM_mean = {oracle_delta_ssim:.6f}",
                f"delta_SPCC_mean = {oracle_delta_spcc:.6f}",
                f"delta_RMSE_mean = {oracle_delta_rmse:.6f}",
                f"delta_JS_mean = {oracle_delta_js:.6f}",
                "",
                oracle_summary[["model", "role", "SPCC_mean", "SSIM_mean", "RMSE_mean", "JS_mean", "high_spatial_SPCC_mean"]].to_string(index=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "msr_predictive_readout_decision.md").write_text(
        "\n".join(
            [
                "# MSR Predictive Readout Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "## Oracle Upper Bound",
                f"oracle_pass = `{oracle_pass}`",
                f"oracle_delta_SSIM_mean = {oracle_delta_ssim:.6f}",
                f"oracle_delta_SPCC_mean = {oracle_delta_spcc:.6f}",
                f"oracle_delta_RMSE_mean = {oracle_delta_rmse:.6f}",
                f"oracle_delta_JS_mean = {oracle_delta_js:.6f}",
                "",
                "## Predictive Test Result",
                f"delta_SSIM_mean = {d_ssim:.6f}",
                f"delta_SPCC_mean = {d_spcc:.6f}",
                f"delta_RMSE_mean = {d_rmse:.6f}",
                f"delta_JS_mean = {d_js:.6f}",
                f"delta_high_spatial_SPCC_mean = {d_hs:.6f}",
                f"controls_fail = `{controls_fail}`",
                "",
                predictive_summary[["model", "role", "control", "SPCC_mean", "SSIM_mean", "RMSE_mean", "JS_mean", "high_spatial_SPCC_mean"]].to_string(index=False),
                "",
                "## External Baseline Diagnostic",
                ext_delta.to_string(index=False) if not ext_delta.empty else "Not available.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print((args.out_dir / "msr_predictive_readout_decision.md").read_text())


if __name__ == "__main__":
    main()
