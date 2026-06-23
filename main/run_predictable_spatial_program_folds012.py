#!/usr/bin/env python3
"""Folds0-2 validation for predictable spatial program transfer.

This validates the fold0-selected mechanism without introducing a new module:

  - SVD spatial programs learned from train genes only, raw expression, K=64.
  - Coefficients predicted from pca32+nmf32 descriptors with Ridge alpha=10.
  - Top-32 predictable components selected on val genes.
  - Spatiality-bin lambdas selected on val genes.
  - Test genes are evaluated once per fold.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

from run_gene_conditioned_mlp_controls_stabilization import make_descriptor_control
from run_gc_spatial_residual_basis_fold0 import train_canonical_base
from run_gc_spatiality_aware_training import compute_spatiality
from run_st_spatial_program_decoder_fold0 import (
    Basis,
    assemble_prediction,
    fit_predict_coeff,
    project_coeff,
    reconstruct,
    summarize_pred,
)
from run_st_spatial_program_decoder_fold0 import preprocess_train
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


def fit_svd_raw_basis(X_train: np.ndarray, k: int, seed: int) -> Basis:
    Xp, meta = preprocess_train(X_train, "raw")
    k_eff = min(int(k), X_train.shape[0] - 2, X_train.shape[1] - 2)
    svd = TruncatedSVD(n_components=k_eff, random_state=seed + k_eff)
    c_train = svd.fit_transform(Xp.T).astype(np.float32)
    a = svd.components_.T.astype(np.float32)
    return Basis(method="svd", preprocess="raw", k=k_eff, A=a, C_train=c_train, meta=meta)


def load_or_train_base_cached(
    X: np.ndarray,
    desc_pca32: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    args,
    fold: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    residual_dir = args.out_dir / "gc_residual_maps"
    residual_dir.mkdir(parents=True, exist_ok=True)
    cache = residual_dir / f"fold{fold}_canonical_gc_mlp_residual_maps.npz"
    if cache.exists() and args.reuse_base:
        cached = np.load(cache)
        return (
            {"train": cached["pred_train"], "val": cached["pred_val"], "test": cached["pred_test"]},
            pd.DataFrame([{"fold": fold, "source": "cache", "path": str(cache)}]),
        )
    train_values = X[:, train_idx].reshape(-1)
    preds, hist, _ = train_canonical_base(
        X=X,
        desc_np=desc_pca32,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        output_low=float(np.quantile(train_values, 0.001)),
        output_high=float(np.quantile(train_values, 0.999)),
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        lr=args.lr,
        seed=args.seed + 1701 * fold,
    )
    np.savez_compressed(
        cache,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        pred_train=preds["train"],
        pred_val=preds["val"],
        pred_test=preds["test"],
        residual_train=X[:, train_idx].astype(np.float32) - preds["train"],
        residual_val=X[:, val_idx].astype(np.float32) - preds["val"],
        residual_test=X[:, test_idx].astype(np.float32) - preds["test"],
    )
    hist = hist.copy()
    hist.insert(0, "fold", fold)
    hist.to_csv(args.out_dir / f"predictable_spatial_program_base_training_history_fold{fold}.csv", index=False)
    return preds, hist


def apply_bin_lambdas(
    base_pred: np.ndarray,
    program_pred: np.ndarray,
    pred_spatiality: np.ndarray,
    val_spatiality_thresholds: tuple[float, float],
    lambdas: dict[str, float],
) -> np.ndarray:
    q1, q2 = val_spatiality_thresholds
    bins = {
        "low": pred_spatiality <= q1,
        "mid": (pred_spatiality > q1) & (pred_spatiality <= q2),
        "high": pred_spatiality > q2,
    }
    out = base_pred.copy()
    for name, lam in lambdas.items():
        out[:, bins[name]] = (1.0 - float(lam)) * base_pred[:, bins[name]] + float(lam) * program_pred[:, bins[name]]
    return out


def summarize_model(
    model: str,
    X: np.ndarray,
    pred_sub: np.ndarray,
    test_idx: np.ndarray,
    low_idx: np.ndarray,
    high_idx: np.ndarray,
    genes: list[str],
    extra: dict,
) -> tuple[dict, pd.DataFrame]:
    full = assemble_prediction(X.shape, test_idx, pred_sub)
    all_gene = gene_metrics(X, full, test_idx, genes)
    row = {"model": model, **extra}
    row.update(summarize_gene_df(all_gene))
    row.update(summarize_gene_df(gene_metrics(X, full, low_idx, genes), prefix="low_expr_"))
    row.update(summarize_gene_df(gene_metrics(X, full, high_idx, genes), prefix="high_spatial_"))
    all_gene = all_gene.copy()
    all_gene.insert(0, "model", model)
    for k, v in extra.items():
        all_gene[k] = v
    low_set = set(map(int, low_idx))
    high_set = set(map(int, high_idx))
    all_gene["is_low_expr_subgroup"] = all_gene["gene_idx"].astype(int).isin(low_set)
    all_gene["is_high_spatial_subgroup"] = all_gene["gene_idx"].astype(int).isin(high_set)
    return row, all_gene


def paired_tests(long_df: pd.DataFrame, base_model: str, selected_model: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for other in [selected_model] + sorted(m for m in long_df["model"].unique() if m not in {base_model, selected_model}):
        for metric in metrics:
            pivot = long_df[long_df["model"].isin([base_model, other])].pivot(index="fold", columns="model", values=metric).dropna()
            if pivot.shape[0] < 2:
                continue
            diff = pivot[other].to_numpy(dtype=float) - pivot[base_model].to_numpy(dtype=float)
            try:
                t_p = float(st.ttest_rel(pivot[other], pivot[base_model], nan_policy="omit").pvalue)
            except Exception:
                t_p = np.nan
            try:
                w_p = float(st.wilcoxon(diff).pvalue) if np.any(np.abs(diff) > 1e-12) else 1.0
            except Exception:
                w_p = np.nan
            rows.append(
                {
                    "comparison": f"{other} vs {base_model}",
                    "model": other,
                    "metric": metric,
                    "n_folds": int(pivot.shape[0]),
                    "mean_base": float(pivot[base_model].mean()),
                    "mean_model": float(pivot[other].mean()),
                    "mean_delta": float(np.mean(diff)),
                    "median_delta": float(np.median(diff)),
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                }
            )
    return pd.DataFrame(rows)


def write_figures(
    fig_dir: Path,
    coords: np.ndarray,
    figure_payloads: list[dict],
    gene_level: pd.DataFrame,
    long_df: pd.DataFrame,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    first = figure_payloads[0]
    A = first["basis_A"]
    keep = first["component_keep"]
    n_show = min(6, len(keep))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for ax, comp in zip(axes.ravel(), keep[:n_show]):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=A[:, comp], s=8, cmap="coolwarm")
        ax.set_title(f"program {int(comp)}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(fig_dir / "learned_spatial_program_examples_fold0.png", dpi=180)
    plt.close(fig)
    np.save(fig_dir / "learned_spatial_program_examples_fold0_A.npy", A[:, keep[:n_show]])

    comp_df = pd.concat([p["component_df"] for p in figure_payloads], ignore_index=True)
    comp_df.to_csv(fig_dir / "component_predictability_all_folds.csv", index=False)
    top = (
        comp_df.groupby("component", as_index=False)["component_spearman"]
        .mean()
        .sort_values("component_spearman", ascending=False)
        .head(24)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(top["component"].astype(str), top["component_spearman"])
    ax.set_ylabel("Val Spearman")
    ax.set_xlabel("Spatial program component")
    ax.set_title("Predictable components")
    fig.tight_layout()
    fig.savefig(fig_dir / "component_predictability_bar.png", dpi=180)
    plt.close(fig)

    lambda_rows = []
    for payload in figure_payloads:
        for bin_name, lam in payload["lambdas"].items():
            lambda_rows.append(
                {
                    "fold": payload["fold"],
                    "bin": bin_name,
                    "lambda": lam,
                    "n_test_genes": int(payload["test_bin_counts"][bin_name]),
                }
            )
    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(fig_dir / "lambda_bin_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for bin_name, grp in lambda_df.groupby("bin"):
        ax.scatter(grp["fold"], grp["lambda"], s=80, label=bin_name)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Selected lambda")
    ax.set_title("Lambda by predicted spatiality bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "lambda_bin_distribution.png", dpi=180)
    plt.close(fig)

    base = gene_level[gene_level["model"].eq("gc_mlp_base")][["fold", "gene_idx", "SPCC", "is_high_spatial_subgroup"]]
    psp = gene_level[gene_level["model"].eq("predictable_spatial_program_selected_correct")][["fold", "gene_idx", "SPCC"]]
    scatter = base.merge(psp, on=["fold", "gene_idx"], suffixes=("_base", "_psp"))
    scatter.to_csv(fig_dir / "per_gene_spcc_base_vs_psp.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = np.where(scatter["is_high_spatial_subgroup"], "#d95f02", "#1b9e77")
    ax.scatter(scatter["SPCC_base"], scatter["SPCC_psp"], c=colors, s=12, alpha=0.75)
    lo = float(np.nanmin([scatter["SPCC_base"].min(), scatter["SPCC_psp"].min()]))
    hi = float(np.nanmax([scatter["SPCC_base"].max(), scatter["SPCC_psp"].max()]))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xlabel("Base per-gene SPCC")
    ax.set_ylabel("PSP per-gene SPCC")
    ax.set_title("Per-gene improvement")
    fig.tight_layout()
    fig.savefig(fig_dir / "per_gene_improvement_scatter.png", dpi=180)
    plt.close(fig)

    hs = long_df[long_df["model"].isin(["gc_mlp_base", "predictable_spatial_program_selected_correct"])]
    hs_summary = hs.groupby("model", as_index=False)["high_spatial_SPCC"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(hs_summary["model"], hs_summary["mean"], yerr=hs_summary["std"].fillna(0), color=["#7570b3", "#e7298a"])
    ax.set_ylabel("high_spatial_SPCC")
    ax.set_title("High-spatial subgroup")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "high_spatial_subgroup_bar.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
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
    ap.add_argument("--output-prefix", type=str, default="predictable_spatial_program_folds012")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "psp_mechanism_figures"

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
    D = desc["pca32_nmf32"]

    long_rows = []
    gene_rows = []
    comp_rows = []
    figure_payloads = []
    lambda_selection_rows = []

    for fold in args.folds:
        print(f"[PSP] fold{fold}: loading masks/base", flush=True)
        train_idx = np.load(args.mask_dir / f"fold{fold}_train_gene_idx.npy")
        val_idx = np.load(args.mask_dir / f"fold{fold}_val_gene_idx.npy")
        test_idx = np.load(args.mask_dir / f"fold{fold}_test_gene_idx.npy")
        low_val_idx, high_val_idx = subgroup_indices(X, val_idx, coords)
        low_test_idx, high_test_idx = subgroup_indices(X, test_idx, coords)
        base, _ = load_or_train_base_cached(X, desc["pca32"], train_idx, val_idx, test_idx, device, args, fold)

        base_val_row = summarize_pred(
            "gc_mlp_base_val",
            X,
            base["val"],
            val_idx,
            low_val_idx,
            high_val_idx,
            genes,
            {"fold": fold, "split": "val", "role": "base", "control": "base"},
        )
        base_test_row, base_gene = summarize_model(
            "gc_mlp_base",
            X,
            base["test"],
            test_idx,
            low_test_idx,
            high_test_idx,
            genes,
            {"fold": fold, "split": "test", "role": "base", "control": "base"},
        )

        print(f"[PSP] fold{fold}: fitting SVD raw K=64 and val component predictability", flush=True)
        basis = fit_svd_raw_basis(X[:, train_idx], k=64, seed=args.seed + fold)
        X_val_proc, val_meta = preprocess_train(X[:, val_idx], "raw")
        C_val_oracle = project_coeff(basis.A, X_val_proc)
        C_val_pred = fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[val_idx], 10.0, seed=args.seed + fold)
        comp_df = component_stats(C_val_pred, C_val_oracle)
        comp_df.insert(0, "fold", fold)
        comp_df.insert(1, "method", "svd")
        comp_df.insert(2, "preprocess", "raw")
        comp_df.insert(3, "K", basis.k)
        comp_df.insert(4, "descriptor", "pca32_nmf32")
        comp_df.insert(5, "predictor", "ridge")
        comp_df.insert(6, "alpha", 10.0)
        comp_df["rank_score"] = comp_df["component_spearman"].fillna(-1.0) * np.log1p(comp_df["oracle_coeff_var"].clip(lower=0))
        comp_rows.append(comp_df)
        keep = comp_df.sort_values("rank_score", ascending=False).head(min(32, basis.k))["component"].to_numpy(dtype=np.int64)
        program_val = selected_component_prediction(basis.A, C_val_pred, val_meta, keep)

        train_sp = compute_spatiality(X, train_idx, edges)
        pred_sp_val = fit_spatiality_predictor(D[train_idx], train_sp["MoranI"].to_numpy(dtype=np.float32), D[val_idx])
        lambdas, lambda_score, bin_val_df = bin_lambdas_from_val(
            X, base["val"], program_val, val_idx, low_val_idx, high_val_idx, genes, pred_sp_val
        )
        bin_val_df.insert(0, "fold", fold)
        bin_val_df["selected_lambda_score"] = lambda_score
        lambda_selection_rows.append(bin_val_df)
        q1, q2 = np.quantile(pred_sp_val, [1 / 3, 2 / 3])
        C_test_pred = fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[test_idx], 10.0, seed=args.seed + 99 + fold)
        program_test = selected_component_prediction(basis.A, C_test_pred, basis.meta, keep)
        pred_sp_test = fit_spatiality_predictor(D[train_idx], train_sp["MoranI"].to_numpy(dtype=np.float32), D[test_idx])
        selected_test = apply_bin_lambdas(base["test"], program_test, pred_sp_test, (q1, q2), lambdas)

        test_bin_counts = {
            "low": int(np.sum(pred_sp_test <= q1)),
            "mid": int(np.sum((pred_sp_test > q1) & (pred_sp_test <= q2))),
            "high": int(np.sum(pred_sp_test > q2)),
        }
        selected_meta = {
            "fold": fold,
            "split": "test",
            "role": "selected",
            "control": "correct",
            "method": "svd",
            "preprocess": "raw",
            "K": basis.k,
            "descriptor": "pca32_nmf32",
            "predictor": "ridge",
            "alpha": 10.0,
            "topK_pred": len(keep),
            "lambda_mode": "predicted_spatiality_bins",
            "lambda_selected": json.dumps(lambdas, sort_keys=True),
            "component_keep": json.dumps([int(x) for x in keep]),
        }
        selected_row, selected_gene = summarize_model(
            "predictable_spatial_program_selected_correct",
            X,
            selected_test,
            test_idx,
            low_test_idx,
            high_test_idx,
            genes,
            selected_meta,
        )

        controls = [
            ("shuffled_descriptor", make_descriptor_control(D, "shuffled", seed=args.seed + 1 + fold)[0], basis, False),
            ("random_descriptor", make_descriptor_control(D, "random", seed=args.seed + 2 + fold)[0], basis, False),
            ("permuted_labels", make_descriptor_control(D, "permuted_labels", seed=args.seed + 3 + fold)[0], basis, False),
        ]
        rng = np.random.default_rng(args.seed + 444 + fold)
        A_rand = rng.normal(0, float(np.std(basis.A) + 1e-6), size=basis.A.shape).astype(np.float32)
        rand_basis = Basis("random_basis", "raw", basis.k, A_rand, project_coeff(A_rand, X[:, train_idx]), basis.meta)
        A_perm = basis.A[rng.permutation(basis.A.shape[0])].astype(np.float32)
        perm_basis = Basis("spot_permuted_basis", "raw", basis.k, A_perm, project_coeff(A_perm, X[:, train_idx]), basis.meta)
        controls.extend(
            [
                ("random_spatial_basis", D, rand_basis, False),
                ("spot_permuted_spatial_program", D, perm_basis, False),
                ("mean_coefficient_baseline", D, basis, True),
            ]
        )

        fold_rows = [base_test_row, selected_row]
        fold_gene = [base_gene, selected_gene]
        for name, D_ctrl, b_ctrl, mean_coeff in controls:
            if mean_coeff:
                C_ctrl = np.repeat(b_ctrl.C_train.mean(axis=0, keepdims=True), len(test_idx), axis=0).astype(np.float32)
            else:
                C_ctrl = fit_predict_coeff("ridge", D_ctrl[train_idx], b_ctrl.C_train, D_ctrl[test_idx], 10.0, seed=args.seed + 199 + fold)
            prog_ctrl = selected_component_prediction(b_ctrl.A, C_ctrl, b_ctrl.meta, keep)
            pred_ctrl = apply_bin_lambdas(base["test"], prog_ctrl, pred_sp_test, (q1, q2), lambdas)
            row, gene_df = summarize_model(
                f"predictable_spatial_program_{name}_control",
                X,
                pred_ctrl,
                test_idx,
                low_test_idx,
                high_test_idx,
                genes,
                {
                    "fold": fold,
                    "split": "test",
                    "role": "control",
                    "control": name,
                    "method": "svd",
                    "preprocess": "raw",
                    "K": basis.k,
                    "descriptor": "pca32_nmf32",
                    "predictor": "ridge",
                    "alpha": 10.0,
                    "topK_pred": len(keep),
                    "lambda_mode": "predicted_spatiality_bins",
                    "lambda_selected": json.dumps(lambdas, sort_keys=True),
                    "component_keep": json.dumps([int(x) for x in keep]),
                },
            )
            fold_rows.append(row)
            fold_gene.append(gene_df)

        fold_df = pd.DataFrame(fold_rows)
        base_row = fold_df[fold_df["role"].eq("base")].iloc[0]
        for metric in ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]:
            fold_df[f"delta_{metric}_vs_base"] = fold_df[metric].astype(float) - float(base_row[metric])
        long_rows.append(fold_df)
        gene_rows.append(pd.concat(fold_gene, ignore_index=True))
        figure_payloads.append(
            {
                "fold": fold,
                "basis_A": basis.A,
                "component_keep": keep,
                "component_df": comp_df,
                "lambdas": lambdas,
                "test_bin_counts": test_bin_counts,
            }
        )
        print(
            fold_df[["model", "SPCC", "SSIM", "RMSE", "JS", "high_spatial_SPCC", "delta_SPCC_vs_base", "delta_RMSE_vs_base"]].to_string(index=False),
            flush=True,
        )

    long_df = pd.concat(long_rows, ignore_index=True)
    gene_df = pd.concat(gene_rows, ignore_index=True)
    comp_all = pd.concat(comp_rows, ignore_index=True)
    lambda_all = pd.concat(lambda_selection_rows, ignore_index=True)

    prefix = args.output_prefix
    long_path = args.out_dir / f"{prefix}_long.csv"
    summary_path = args.out_dir / f"{prefix}_summary.csv"
    paired_path = args.out_dir / f"{prefix}_paired_tests.csv"
    decision_path = args.out_dir / f"{prefix}_decision.md"
    gene_path = args.out_dir / f"{prefix}_gene_level.csv"
    comp_path = args.out_dir / f"{prefix}_component_predictability.csv"
    lambda_path = args.out_dir / f"{prefix}_lambda_selection.csv"

    long_df.to_csv(long_path, index=False)
    gene_df.to_csv(gene_path, index=False)
    comp_all.to_csv(comp_path, index=False)
    lambda_all.to_csv(lambda_path, index=False)

    metrics = ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]
    summary = (
        long_df.groupby(["model", "role", "control"], as_index=False)[metrics]
        .agg(["mean", "std", "median"])
    )
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(summary_path, index=False)
    if prefix == "predictable_spatial_program_5fold":
        summary.to_csv(args.out_dir / "final_strict_gc_psp_benchmark_summary.csv", index=False)
    paired = paired_tests(long_df, "gc_mlp_base", "predictable_spatial_program_selected_correct", metrics)
    paired.to_csv(paired_path, index=False)

    selected_mean = summary[summary["model"].eq("predictable_spatial_program_selected_correct")].iloc[0]
    base_mean = summary[summary["model"].eq("gc_mlp_base")].iloc[0]
    control_summary = summary[summary["role"].eq("control")]
    delta_spcc = float(selected_mean["SPCC_mean"] - base_mean["SPCC_mean"])
    delta_hs = float(selected_mean["high_spatial_SPCC_mean"] - base_mean["high_spatial_SPCC_mean"])
    delta_rmse = float(selected_mean["RMSE_mean"] - base_mean["RMSE_mean"])
    delta_js = float(selected_mean["JS_mean"] - base_mean["JS_mean"])
    delta_ssim = float(selected_mean["SSIM_mean"] - base_mean["SSIM_mean"])
    controls_ok = bool(
        selected_mean["SPCC_mean"] > control_summary["SPCC_mean"].max()
        and selected_mean["RMSE_mean"] <= control_summary["RMSE_mean"].min() + 1e-12
        and selected_mean["JS_mean"] <= control_summary["JS_mean"].min() + 1e-12
    )
    gate = bool(
        (
            delta_spcc >= 0.002
            or delta_hs >= 0.005
            or (delta_rmse <= -0.001 and delta_spcc >= -0.001)
        )
        and controls_ok
        and delta_ssim >= -0.002
        and delta_js <= 0.003
    )
    aux = bool(not gate and controls_ok and (delta_hs > 0 or delta_rmse < 0) and delta_ssim >= -0.002 and delta_js <= 0.003)
    decision = "PSP_CONTINUE" if gate else ("PSP_AUXILIARY" if aux else "PSP_STOP")

    write_figures(fig_dir, coords, figure_payloads, gene_df, long_df)

    decision_path.write_text(
        "\n".join(
            [
                "# Predictable Spatial Program Folds0-2 Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "## Fixed Mechanism",
                "- basis: SVD raw K=64 learned from train genes only",
                "- descriptor: pca32_nmf32",
                "- coefficient predictor: ridge alpha=10.0 trained on train genes only",
                "- component filter: top-32 predictable components ranked on val genes",
                "- lambda mode: predicted spatiality bins selected on val genes",
                "- test genes: final evaluation only",
                "",
                "## Mean Test Summary",
                summary[
                    [
                        "model",
                        "role",
                        "control",
                        "SPCC_mean",
                        "SSIM_mean",
                        "RMSE_mean",
                        "JS_mean",
                        "low_expr_SPCC_mean",
                        "high_spatial_SPCC_mean",
                        "high_spatial_RMSE_mean",
                    ]
                ].to_string(index=False),
                "",
                "## Correct vs GC-MLP Base",
                f"delta_SPCC_mean = {delta_spcc:.6f}",
                f"delta_SSIM_mean = {delta_ssim:.6f}",
                f"delta_RMSE_mean = {delta_rmse:.6f}",
                f"delta_JS_mean = {delta_js:.6f}",
                f"delta_high_spatial_SPCC_mean = {delta_hs:.6f}",
                f"controls_ok = `{controls_ok}`",
                "",
                "## Output Files",
                f"- {long_path}",
                f"- {summary_path}",
                f"- {paired_path}",
                f"- {gene_path}",
                f"- {comp_path}",
                f"- {lambda_path}",
                f"- {fig_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(summary[["model", "SPCC_mean", "SSIM_mean", "RMSE_mean", "JS_mean", "high_spatial_SPCC_mean"]].to_string(index=False))
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
