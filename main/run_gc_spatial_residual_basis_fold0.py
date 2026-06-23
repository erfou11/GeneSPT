#!/usr/bin/env python3
"""Fold0 descriptor-conditioned spatial residual basis gate.

This script is deliberately outside the legacy fixed-output GeneSPT stack.
It tests whether train-gene residual spatial maps from the canonical
gene-conditioned MLP can be transferred to strict held-out genes through
scRNA-derived gene descriptors.
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
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from torch import nn

from run_gene_conditioned_mlp_controls_stabilization import (
    MLPVariant,
    FlexibleMLPDecoder,
    make_descriptor_control,
)
from run_strict_gene_conditioned_decoder_gate import (
    build_descriptors,
    gene_metrics,
    load_matrix,
    log1p_cpm,
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


def train_canonical_base(
    X: np.ndarray,
    desc_np: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_low: float,
    output_high: float,
    device: torch.device,
    steps: int,
    batch_size: int,
    eval_every: int,
    lr: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, float]:
    """Train canonical GC-MLP PCA32 softplus and return train/val/test preds."""
    set_seed(seed)
    x_input_np = X[:, train_idx].astype(np.float32)
    x_input = torch.tensor(x_input_np, device=device)
    y_full = torch.tensor(X, device=device)
    desc = torch.tensor(desc_np.astype(np.float32), device=device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    model = FlexibleMLPDecoder(
        input_dim=x_input_np.shape[1],
        desc_dim=desc_np.shape[1],
        output_mode="softplus",
        output_low=output_low,
        output_high=output_high,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_score = -np.inf
    best_state = None
    hist = []
    for step in range(1, steps + 1):
        spot_idx = torch.randint(0, X.shape[0], (batch_size,), device=device)
        gene_pos = torch.randint(0, len(train_idx), (batch_size,), device=device)
        gene_idx = train_idx_t[gene_pos]
        pred = model.forward_pairs(x_input, spot_idx, desc[gene_idx])
        target = y_full[spot_idx, gene_idx]
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                pred_val = model.predict_matrix(x_input, desc[val_idx_t], chunk=128).detach().cpu().numpy()
            X_pred_val = np.zeros_like(X, dtype=np.float32)
            X_pred_val[:, val_idx] = pred_val
            val_df = gene_metrics(X, X_pred_val, val_idx, [str(i) for i in range(X.shape[1])])
            val_summary = summarize_gene_df(val_df)
            score = (
                np.nan_to_num(val_summary["SPCC"], nan=-1.0)
                - 0.05 * np.nan_to_num(val_summary["RMSE"], nan=10.0)
                - 0.05 * np.nan_to_num(val_summary["JS"], nan=10.0, posinf=10.0)
            )
            hist.append({"step": step, "loss": float(loss.detach().cpu()), **{f"val_{k}": v for k, v in val_summary.items()}, "score": float(score)})
            if score > best_score:
                best_score = float(score)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    preds = {}
    with torch.no_grad():
        for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            preds[name] = (
                model.predict_matrix(x_input, desc[torch.tensor(idx, device=device)], chunk=128)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return preds, pd.DataFrame(hist), float(best_score)


def make_knn_laplacian(coords: np.ndarray, k: int = 8) -> sp.csr_matrix:
    nn = NearestNeighbors(n_neighbors=min(k + 1, coords.shape[0]), metric="euclidean")
    nn.fit(coords)
    _, ind = nn.kneighbors(coords)
    rows = []
    cols = []
    data = []
    for i in range(coords.shape[0]):
        for j in ind[i, 1:]:
            rows.extend([i, int(j)])
            cols.extend([int(j), i])
            data.extend([1.0, 1.0])
    W = sp.coo_matrix((data, (rows, cols)), shape=(coords.shape[0], coords.shape[0])).tocsr()
    W.data[:] = 1.0
    deg = np.asarray(W.sum(axis=1)).reshape(-1)
    return sp.diags(deg) - W


def orthonormalize(U: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(U, dtype=np.float64))
    return q.astype(np.float32)


def build_basis_bank(residual_train: np.ndarray, coords: np.ndarray, max_k: int, seed: int) -> dict[str, np.ndarray]:
    n_spots = residual_train.shape[0]
    max_k = min(max_k, n_spots - 2)
    U_pca, _, _ = randomized_svd(residual_train.astype(np.float64), n_components=max_k, random_state=seed)
    U_pca = orthonormalize(U_pca)
    L = make_knn_laplacian(coords, k=8)
    try:
        vals, vecs = spla.eigsh(L.astype(np.float64), k=max_k, which="SM")
        order = np.argsort(vals)
        U_lap = orthonormalize(vecs[:, order])
    except Exception:
        vals, vecs = np.linalg.eigh(L.toarray().astype(np.float64))
        U_lap = orthonormalize(vecs[:, np.argsort(vals)[:max_k]])
    U_hybrid = orthonormalize(np.concatenate([U_pca[:, : max_k // 2], U_lap[:, : max_k - max_k // 2]], axis=1))
    return {"pca": U_pca, "laplacian": U_lap, "hybrid": U_hybrid}


def make_random_basis(n_spots: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return orthonormalize(rng.normal(size=(n_spots, k)))


def coefficients(U: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Return gene x K coefficients for spot x gene residual matrix."""
    return (U.T @ residual).T.astype(np.float32)


def reconstruct(U: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return (U @ coeff.T).astype(np.float32)


def fit_predict_coeff(
    predictor: str,
    desc_train: np.ndarray,
    coeff_train: np.ndarray,
    desc_eval: np.ndarray,
    alpha: float,
    seed: int,
) -> np.ndarray:
    if predictor == "ridge":
        model = Ridge(alpha=float(alpha), random_state=seed)
        model.fit(desc_train, coeff_train)
        return model.predict(desc_eval).astype(np.float32)
    raise ValueError(predictor)


class CoeffMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_predict_coeff_mlp(
    desc_train: np.ndarray,
    coeff_train: np.ndarray,
    desc_eval: np.ndarray,
    seed: int,
    steps: int = 800,
    lr: float = 1e-3,
) -> np.ndarray:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(desc_train.astype(np.float32), device=device)
    y = torch.tensor(coeff_train.astype(np.float32), device=device)
    xe = torch.tensor(desc_eval.astype(np.float32), device=device)
    model = CoeffMLP(x.shape[1], y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        return model(xe).detach().cpu().numpy().astype(np.float32)


def assemble_prediction(X_shape: tuple[int, int], idx: np.ndarray, pred_sub: np.ndarray) -> np.ndarray:
    out = np.zeros(X_shape, dtype=np.float32)
    out[:, idx] = pred_sub.astype(np.float32)
    return out


def summarize_subsets(
    model: str,
    X: np.ndarray,
    pred_sub: np.ndarray,
    idx: np.ndarray,
    low_idx: np.ndarray,
    high_idx: np.ndarray,
    genes: list[str],
    extra: dict,
) -> dict:
    X_pred = assemble_prediction(X.shape, idx, pred_sub)
    row = {"model": model, **extra}
    row.update(summarize_gene_df(gene_metrics(X, X_pred, idx, genes)))
    row.update(summarize_gene_df(gene_metrics(X, X_pred, low_idx, genes), prefix="low_expr_"))
    row.update(summarize_gene_df(gene_metrics(X, X_pred, high_idx, genes), prefix="high_spatial_"))
    return row


def quick_gene_summary(X: np.ndarray, pred_sub: np.ndarray, idx: np.ndarray, low_idx: np.ndarray, high_idx: np.ndarray, prefix: str = "") -> dict:
    """Fast val-selection proxy using vectorized Pearson/RMSE/JS-like metrics.

    Full Spearman/SSIM metrics are still computed for final test rows and oracle
    diagnostics. This proxy keeps the fold0 gate lightweight.
    """

    pos = {int(g): j for j, g in enumerate(idx)}

    def subset_positions(sub_idx: np.ndarray) -> np.ndarray:
        return np.asarray([pos[int(g)] for g in sub_idx if int(g) in pos], dtype=np.int64)

    def summarize_cols(cols: np.ndarray) -> dict[str, float]:
        if cols.size == 0:
            return {"SPCC": np.nan, "SSIM": np.nan, "RMSE": np.nan, "JS": np.nan}
        y = X[:, idx[cols]].astype(np.float64)
        p = pred_sub[:, cols].astype(np.float64)
        yc = y - y.mean(axis=0, keepdims=True)
        pc = p - p.mean(axis=0, keepdims=True)
        denom = np.sqrt(np.sum(yc * yc, axis=0) * np.sum(pc * pc, axis=0))
        corr = np.divide(np.sum(yc * pc, axis=0), denom, out=np.full(cols.size, np.nan), where=denom > 1e-12)
        ysd = np.maximum(y.std(axis=0, keepdims=True), 1e-12)
        psd = np.maximum(p.std(axis=0, keepdims=True), 1e-12)
        rmse = np.sqrt(np.mean(((y - y.mean(axis=0, keepdims=True)) / ysd - (p - p.mean(axis=0, keepdims=True)) / psd) ** 2, axis=0))
        yp = np.clip(y, 0.0, None)
        pp = np.clip(p, 0.0, None)
        yp = yp / np.maximum(yp.sum(axis=0, keepdims=True), 1e-12)
        pp = pp / np.maximum(pp.sum(axis=0, keepdims=True), 1e-12)
        mid = 0.5 * (yp + pp)
        js = 0.5 * np.sum(np.where(yp > 0, yp * np.log(np.maximum(yp, 1e-12) / np.maximum(mid, 1e-12)), 0.0), axis=0)
        js += 0.5 * np.sum(np.where(pp > 0, pp * np.log(np.maximum(pp, 1e-12) / np.maximum(mid, 1e-12)), 0.0), axis=0)
        return {"SPCC": float(np.nanmedian(corr)), "SSIM": np.nan, "RMSE": float(np.nanmedian(rmse)), "JS": float(np.nanmedian(js))}

    all_cols = np.arange(len(idx), dtype=np.int64)
    out = {f"{prefix}{k}": v for k, v in summarize_cols(all_cols).items()}
    out.update({f"{prefix}low_expr_{k}": v for k, v in summarize_cols(subset_positions(low_idx)).items()})
    out.update({f"{prefix}high_spatial_{k}": v for k, v in summarize_cols(subset_positions(high_idx)).items()})
    return out


def candidate_score(row: dict, base_row: dict) -> float:
    # Val-only score favoring the stated target: high-spatial and raw improvement
    # while mildly penalizing SSIM drops and distribution errors.
    return float(
        np.nan_to_num(row["SPCC"], nan=-1.0)
        + 0.5 * np.nan_to_num(row["high_spatial_SPCC"], nan=-1.0)
        - 0.04 * np.nan_to_num(row["RMSE"], nan=10.0)
        - 0.04 * np.nan_to_num(row["JS"], nan=10.0, posinf=10.0)
        - (0.25 * max(0.0, float(base_row["SSIM"] - row["SSIM"])) if np.isfinite(row.get("SSIM", np.nan)) else 0.0)
    )


@dataclass
class Selected:
    basis_type: str
    k: int
    descriptor: str
    predictor: str
    alpha: float
    lam: float
    score: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
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
    residual_dir = args.out_dir / "gc_residual_maps"
    residual_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_counts, genes, _ = load_matrix(args.counts_path, index_col=None)
    X = log1p_cpm(X_counts)
    coords = pd.read_csv(args.locations_path, sep="\t").to_numpy(dtype=np.float32)
    X_sc_counts, sc_genes, _ = load_matrix(args.scrna_counts_path, index_col=0)
    if list(sc_genes) != list(genes):
        sc_map = {g: i for i, g in enumerate(sc_genes)}
        X_sc_counts = X_sc_counts[:, [sc_map[g] for g in genes]]
    X_sc = log1p_cpm(X_sc_counts)
    desc = build_descriptors(X_sc, pca_dims=[32], nmf_dims=[32], seed=args.seed)
    desc["pca32_nmf32"] = np.concatenate([desc["pca32"], desc["nmf32"]], axis=1).astype(np.float32)

    fold = int(args.fold)
    train_idx = np.load(args.mask_dir / f"fold{fold}_train_gene_idx.npy")
    val_idx = np.load(args.mask_dir / f"fold{fold}_val_gene_idx.npy")
    test_idx = np.load(args.mask_dir / f"fold{fold}_test_gene_idx.npy")
    low_val_idx, high_val_idx = subgroup_indices(X, val_idx, coords)
    low_test_idx, high_test_idx = subgroup_indices(X, test_idx, coords)
    train_values = X[:, train_idx].reshape(-1)
    q001 = float(np.quantile(train_values, 0.001))
    q999 = float(np.quantile(train_values, 0.999))

    residual_path = residual_dir / f"fold{fold}_canonical_gc_mlp_residual_maps.npz"
    if args.reuse_base and residual_path.exists():
        print(f"Reusing cached base residual maps: {residual_path}", flush=True)
        cached = np.load(residual_path)
        base_preds = {"train": cached["pred_train"], "val": cached["pred_val"], "test": cached["pred_test"]}
        residual_train = cached["residual_train"]
        residual_val = cached["residual_val"]
        residual_test = cached["residual_test"]
        base_val_score = float("nan")
    else:
        print("Training canonical GC-MLP base for residual maps", flush=True)
        base_preds, base_hist, base_val_score = train_canonical_base(
            X=X,
            desc_np=desc["pca32"],
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            output_low=q001,
            output_high=q999,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            lr=args.lr,
            seed=args.seed + 1701 * fold,
        )
        base_hist.to_csv(args.out_dir / "gc_spatial_residual_basis_base_training_history.csv", index=False)

        residual_train = X[:, train_idx].astype(np.float32) - base_preds["train"]
        residual_val = X[:, val_idx].astype(np.float32) - base_preds["val"]
        residual_test = X[:, test_idx].astype(np.float32) - base_preds["test"]
        np.savez_compressed(
            residual_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            pred_train=base_preds["train"],
            pred_val=base_preds["val"],
            pred_test=base_preds["test"],
            residual_train=residual_train,
            residual_val=residual_val,
            residual_test=residual_test,
        )

    base_train = summarize_subsets("base_train", X, base_preds["train"], train_idx, train_idx[:0], train_idx[:0], genes, {"split": "train", "best_val_score": base_val_score})
    base_val = summarize_subsets("base_val", X, base_preds["val"], val_idx, low_val_idx, high_val_idx, genes, {"split": "val", "best_val_score": base_val_score})
    base_test = summarize_subsets("base_test", X, base_preds["test"], test_idx, low_test_idx, high_test_idx, genes, {"split": "test", "best_val_score": base_val_score})
    pd.DataFrame([base_train, base_val, base_test]).to_csv(args.out_dir / "gc_residual_basis_base_audit.csv", index=False)

    print("Building residual PCA/Laplacian/hybrid basis bank", flush=True)
    basis_bank = build_basis_bank(residual_train, coords, max_k=64, seed=args.seed)
    ks = [8, 16, 32, 64]
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    descriptors = ["pca32", "nmf32", "pca32_nmf32"]
    ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    basis_audit_rows = []
    val_rows = []
    best: Selected | None = None
    best_row = None
    for basis_type, U_full in basis_bank.items():
        print(f"Evaluating basis type: {basis_type}", flush=True)
        for k in ks:
            U = U_full[:, :k].astype(np.float32)
            coeff_train = coefficients(U, residual_train)
            coeff_val_oracle = coefficients(U, residual_val)
            oracle_val_pred = base_preds["val"] + reconstruct(U, coeff_val_oracle)
            oracle_row = summarize_subsets(
                f"oracle_{basis_type}_k{k}",
                X,
                oracle_val_pred,
                val_idx,
                low_val_idx,
                high_val_idx,
                genes,
                {"split": "val_oracle", "basis_type": basis_type, "K": k, "descriptor": "oracle", "predictor": "oracle", "alpha": np.nan, "lambda": 1.0},
            )
            oracle_row["delta_high_spatial_SPCC_vs_base"] = float(oracle_row["high_spatial_SPCC"] - base_val["high_spatial_SPCC"])
            oracle_row["delta_SPCC_vs_base"] = float(oracle_row["SPCC"] - base_val["SPCC"])
            oracle_row["delta_RMSE_vs_base"] = float(oracle_row["RMSE"] - base_val["RMSE"])
            basis_audit_rows.append(oracle_row)
            for desc_name in descriptors:
                desc_train = desc[desc_name][train_idx]
                desc_val = desc[desc_name][val_idx]
                for predictor in ["ridge"]:
                    alpha_grid = ridge_alphas
                    for alpha in alpha_grid:
                        try:
                            coeff_val_pred = fit_predict_coeff(predictor, desc_train, coeff_train, desc_val, alpha=alpha, seed=args.seed)
                        except Exception as exc:
                            val_rows.append(
                                {
                                    "model": f"{basis_type}_k{k}_{desc_name}_{predictor}_alpha{alpha}",
                                    "split": "val",
                                    "basis_type": basis_type,
                                    "K": k,
                                    "descriptor": desc_name,
                                    "predictor": predictor,
                                    "alpha": alpha,
                                    "lambda": np.nan,
                                    "error": repr(exc),
                                }
                            )
                            continue
                        residual_val_pred = reconstruct(U, coeff_val_pred)
                        for lam in lams:
                            pred_val = base_preds["val"] + float(lam) * residual_val_pred
                            row = {
                                "model": f"{basis_type}_k{k}_{desc_name}_{predictor}_alpha{alpha}_lambda{lam}",
                                "split": "val",
                                "basis_type": basis_type,
                                "K": k,
                                "descriptor": desc_name,
                                "predictor": predictor,
                                "alpha": alpha,
                                "lambda": lam,
                                "error": "",
                                "selection_metric_mode": "fast_pearson_proxy",
                                **quick_gene_summary(X, pred_val, val_idx, low_val_idx, high_val_idx),
                            }
                            row["delta_SPCC_vs_base"] = float(row["SPCC"] - base_val["SPCC"])
                            row["delta_RMSE_vs_base"] = float(row["RMSE"] - base_val["RMSE"])
                            row["delta_JS_vs_base"] = float(row["JS"] - base_val["JS"])
                            row["delta_SSIM_vs_base"] = float(row["SSIM"] - base_val["SSIM"])
                            row["delta_high_spatial_SPCC_vs_base"] = float(row["high_spatial_SPCC"] - base_val["high_spatial_SPCC"])
                            row["selection_score"] = candidate_score(row, base_val)
                            val_rows.append(row)
                            if best is None or row["selection_score"] > best.score:
                                best = Selected(basis_type, k, desc_name, predictor, float(alpha), float(lam), float(row["selection_score"]))
                                best_row = row

    basis_audit = pd.DataFrame(basis_audit_rows)
    basis_audit.to_csv(args.out_dir / "gc_spatial_residual_basis_audit.csv", index=False)
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(args.out_dir / "gc_spatial_residual_basis_fold0_long.csv", index=False)
    if best is None:
        raise RuntimeError("No valid residual-basis candidate was selected.")

    # Add one MLP coefficient predictor diagnostic for the selected basis/descriptor.
    U_sel = basis_bank[best.basis_type][:, : best.k].astype(np.float32)
    coeff_train_sel = coefficients(U_sel, residual_train)
    coeff_val_mlp = fit_predict_coeff_mlp(desc[best.descriptor][train_idx], coeff_train_sel, desc[best.descriptor][val_idx], seed=args.seed + 911)
    mlp_val_pred = base_preds["val"] + best.lam * reconstruct(U_sel, coeff_val_mlp)
    mlp_val_row = {
        "model": f"{best.basis_type}_k{best.k}_{best.descriptor}_mlp_lambda{best.lam}",
        "split": "val",
        "basis_type": best.basis_type,
        "K": best.k,
        "descriptor": best.descriptor,
        "predictor": "mlp",
        "alpha": np.nan,
        "lambda": best.lam,
        "error": "",
        "selection_metric_mode": "fast_pearson_proxy",
        **quick_gene_summary(X, mlp_val_pred, val_idx, low_val_idx, high_val_idx),
    }
    mlp_val_row["delta_SPCC_vs_base"] = float(mlp_val_row["SPCC"] - base_val["SPCC"])
    mlp_val_row["delta_RMSE_vs_base"] = float(mlp_val_row["RMSE"] - base_val["RMSE"])
    mlp_val_row["delta_JS_vs_base"] = float(mlp_val_row["JS"] - base_val["JS"])
    mlp_val_row["delta_SSIM_vs_base"] = float(mlp_val_row["SSIM"] - base_val["SSIM"])
    mlp_val_row["delta_high_spatial_SPCC_vs_base"] = float(mlp_val_row["high_spatial_SPCC"] - base_val["high_spatial_SPCC"])
    mlp_val_row["selection_score"] = candidate_score(mlp_val_row, base_val)
    if mlp_val_row["selection_score"] > best.score:
        best = Selected(best.basis_type, best.k, best.descriptor, "mlp", np.nan, best.lam, float(mlp_val_row["selection_score"]))
        best_row = mlp_val_row
    pd.concat([val_df, pd.DataFrame([mlp_val_row])], ignore_index=True).to_csv(args.out_dir / "gc_spatial_residual_basis_fold0_long.csv", index=False)

    def predict_with_selected(desc_control: np.ndarray, basis_override: np.ndarray | None = None, mean_coeff: bool = False, seed_offset: int = 0) -> np.ndarray:
        U = (basis_override if basis_override is not None else basis_bank[best.basis_type][:, : best.k]).astype(np.float32)
        coeff_train = coefficients(U, residual_train)
        if mean_coeff:
            coeff_test = np.repeat(coeff_train.mean(axis=0, keepdims=True), len(test_idx), axis=0).astype(np.float32)
        elif best.predictor == "mlp":
            coeff_test = fit_predict_coeff_mlp(desc_control[train_idx], coeff_train, desc_control[test_idx], seed=args.seed + 1300 + seed_offset)
        else:
            coeff_test = fit_predict_coeff(best.predictor, desc_control[train_idx], coeff_train, desc_control[test_idx], alpha=best.alpha, seed=args.seed + seed_offset)
        return base_preds["test"] + best.lam * reconstruct(U, coeff_test)

    selected_desc_correct, _ = make_descriptor_control(desc[best.descriptor], "correct", seed=args.seed + 1)
    test_rows = [
        summarize_subsets(
            "canonical_gc_mlp_pca32_softplus_base",
            X,
            base_preds["test"],
            test_idx,
            low_test_idx,
            high_test_idx,
            genes,
            {"split": "test", "candidate_role": "base", "basis_type": "none", "K": 0, "descriptor": "pca32", "predictor": "none", "control": "none", "lambda": 0.0},
        )
    ]
    selected_test = predict_with_selected(selected_desc_correct)
    test_rows.append(
        summarize_subsets(
            "spatial_residual_basis_selected_correct",
            X,
            selected_test,
            test_idx,
            low_test_idx,
            high_test_idx,
            genes,
            {
                "split": "test",
                "candidate_role": "selected",
                "basis_type": best.basis_type,
                "K": best.k,
                "descriptor": best.descriptor,
                "predictor": best.predictor,
                "control": "correct",
                "lambda": best.lam,
                "alpha": best.alpha,
            },
        )
    )
    control_specs = [
        ("shuffled_descriptor_control", "shuffled", None, False),
        ("random_descriptor_control", "random", None, False),
        ("permuted_label_descriptor_control", "permuted_labels", None, False),
        ("random_spatial_basis_control", "correct", make_random_basis(X.shape[0], best.k, args.seed + 404), False),
        ("spot_permuted_spatial_basis_control", "correct", basis_bank[best.basis_type][np.random.default_rng(args.seed + 505).permutation(X.shape[0]), : best.k], False),
        ("mean_residual_coefficient_baseline", "correct", None, True),
    ]
    for i, (name, control, basis_override, mean_coeff) in enumerate(control_specs):
        desc_control, _ = make_descriptor_control(desc[best.descriptor], control, seed=args.seed + 300 + i)
        pred = predict_with_selected(desc_control, basis_override=basis_override, mean_coeff=mean_coeff, seed_offset=200 + i)
        test_rows.append(
            summarize_subsets(
                name,
                X,
                pred,
                test_idx,
                low_test_idx,
                high_test_idx,
                genes,
                {
                    "split": "test",
                    "candidate_role": "control",
                    "basis_type": best.basis_type if basis_override is None else name,
                    "K": best.k,
                    "descriptor": best.descriptor,
                    "predictor": best.predictor,
                    "control": control if not mean_coeff else "mean_coeff",
                    "lambda": best.lam,
                    "alpha": best.alpha,
                },
            )
        )

    test_df = pd.DataFrame(test_rows)
    base = test_df[test_df["model"].eq("canonical_gc_mlp_pca32_softplus_base")].iloc[0]
    for metric in ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]:
        test_df[f"delta_{metric}_vs_base"] = test_df[metric].astype(float) - float(base[metric])
    test_df.to_csv(args.out_dir / "gc_spatial_residual_basis_fold0_summary.csv", index=False)
    controls_df = test_df[test_df["candidate_role"].eq("control")].copy()
    controls_df.to_csv(args.out_dir / "gc_spatial_residual_basis_controls_fold0.csv", index=False)

    selected = test_df[test_df["model"].eq("spatial_residual_basis_selected_correct")].iloc[0]
    control_max_spcc = float(controls_df["SPCC"].max()) if not controls_df.empty else -np.inf
    control_min_rmse = float(controls_df["RMSE"].min()) if not controls_df.empty else np.inf
    control_min_js = float(controls_df["JS"].min()) if not controls_df.empty else np.inf
    correct_beats_controls = bool(
        float(selected["SPCC"]) > control_max_spcc
        and float(selected["RMSE"]) <= control_min_rmse + 1e-12
        and float(selected["JS"]) <= control_min_js + 1e-12
    )
    gate_main = bool(
        (
            float(selected["delta_high_spatial_SPCC_vs_base"]) >= 0.01
            or float(selected["delta_SPCC_vs_base"]) >= 0.002
            or (float(selected["delta_RMSE_vs_base"]) < -0.001 and float(selected["delta_JS_vs_base"]) <= 0.0 and float(selected["delta_SPCC_vs_base"]) >= -0.0005)
        )
        and float(selected["delta_SSIM_vs_base"]) >= -0.002
        and correct_beats_controls
    )
    gate_aux = bool(
        not gate_main
        and correct_beats_controls
        and float(selected["delta_JS_vs_base"]) <= 0.003
        and float(selected["delta_SPCC_vs_base"]) >= -0.001
        and (float(selected["delta_RMSE_vs_base"]) < 0 or float(selected["delta_JS_vs_base"]) < 0 or float(selected["delta_high_spatial_SPCC_vs_base"]) > 0)
        and float(selected["delta_SSIM_vs_base"]) >= -0.002
    )
    if gate_main:
        decision = "SPATIAL_RESIDUAL_BASIS_CONTINUE"
    elif gate_aux:
        decision = "SPATIAL_RESIDUAL_BASIS_AUXILIARY"
    else:
        decision = "SPATIAL_RESIDUAL_BASIS_FAILED"

    manifest = {
        "fold": fold,
        "mask_safety": {
            "basis_fit": "train_gene residuals only",
            "coefficient_predictor_fit": "train_gene descriptors to train residual coefficients only",
            "hyperparameter_selection": "val genes only",
            "test_usage": "final evaluation only",
        },
        "selected_by_val": best.__dict__,
        "selected_val_row": best_row,
        "base_val": base_val,
        "base_test": base_test,
        "decision": decision,
    }
    (args.out_dir / "gc_spatial_residual_basis_fold0_manifest.json").write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    lines = [
        "# GC Spatial Residual Basis Fold0 Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "## Mask Safety",
        "- Basis fitted from train-gene residual maps only.",
        "- Coefficient predictor fitted from train-gene descriptors and train-gene residual coefficients only.",
        "- Val genes used for basis/K/predictor/lambda selection.",
        "- Test genes used only once for final evaluation.",
        "",
        "## Selected Candidate",
        f"- basis_type: `{best.basis_type}`",
        f"- K: `{best.k}`",
        f"- descriptor: `{best.descriptor}`",
        f"- predictor: `{best.predictor}`",
        f"- alpha: `{best.alpha}`",
        f"- lambda: `{best.lam}`",
        f"- val selection score: `{best.score:.6f}`",
        "",
        "## Test Summary",
        test_df[["model", "SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "delta_SPCC_vs_base", "delta_SSIM_vs_base", "delta_RMSE_vs_base", "delta_JS_vs_base", "delta_high_spatial_SPCC_vs_base"]].to_string(index=False),
        "",
        f"Correct beats all controls on SPCC/RMSE/JS: `{correct_beats_controls}`",
    ]
    (args.out_dir / "gc_spatial_residual_basis_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(test_df[["model", "SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "delta_SPCC_vs_base", "delta_SSIM_vs_base", "delta_RMSE_vs_base", "delta_JS_vs_base"]].to_string(index=False))
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
