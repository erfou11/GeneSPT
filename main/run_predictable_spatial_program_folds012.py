#!/usr/bin/env python3
"""Multi-fold validation for predictable spatial program transfer.

This validates the fold0-selected mechanism without introducing a new module:

  - SVD spatial programs learned from train genes only, raw expression, K=64.
  - Coefficients predicted from a configurable descriptor with Ridge alpha=10.
  - Top-32 predictable components selected on val genes.
  - Spatiality-bin lambdas selected on val genes.
  - Test genes are evaluated once per fold.

Dataset-specific matrices, frozen masks, GC caches, and descriptor caches can
be supplied explicitly so the same controlled PSP comparison is reusable
without retraining the gene-conditioned decoder.
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
from protocol_a_preprocessing import normalize_st_protocol_a


INFO = Path("/workspace/GeneSPT/results/imformation")
ST_TARGET_SUM = 1e4
ZERO_TRAIN_LIBRARY_STRATEGY = "set_entire_normalized_spot_row_to_zero"


def log1p_cpm_with_denominator_genes(
    X_counts: np.ndarray,
    denominator_gene_idx: np.ndarray,
    target_sum: float = ST_TARGET_SUM,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a full count matrix using libraries from a gene subset only."""

    counts = np.asarray(X_counts, dtype=np.float32)
    idx = np.asarray(denominator_gene_idx, dtype=np.int64)
    if counts.ndim != 2:
        raise ValueError(f"ST count matrix must be 2D, got shape {counts.shape}")
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("denominator_gene_idx must be a non-empty 1D array")
    if np.unique(idx).size != idx.size:
        raise ValueError("denominator_gene_idx contains duplicate genes")
    if int(idx.min()) < 0 or int(idx.max()) >= counts.shape[1]:
        raise IndexError("denominator_gene_idx contains an out-of-range gene index")
    if not np.isfinite(counts).all():
        raise ValueError("ST count matrix contains non-finite values")
    if np.any(counts < 0):
        raise ValueError("ST count matrix contains negative values")
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be finite and positive")

    library_sizes = counts[:, idx].sum(axis=1, dtype=np.float64)
    zero_library = library_sizes == 0.0
    safe_library_sizes = library_sizes.copy()
    safe_library_sizes[zero_library] = 1.0

    normalized = np.empty_like(counts, dtype=np.float32)
    np.divide(counts, safe_library_sizes.astype(np.float32)[:, None], out=normalized)
    normalized *= np.float32(target_sum)
    np.log1p(normalized, out=normalized)
    normalized[zero_library, :] = 0.0
    return normalized, library_sizes


def self_check_train_gene_normalization() -> None:
    """Check that held-out counts cannot affect train-gene normalization."""

    counts = np.asarray(
        [
            [4.0, 6.0, 2.0, 1.0],
            [0.0, 0.0, 7.0, 8.0],
            [1.0, 3.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    train_idx = np.asarray([0, 1], dtype=np.int64)
    changed = counts.copy()
    changed[:, 2:] = np.asarray([[200.0, 100.0], [70.0, 80.0], [90.0, 50.0]], dtype=np.float32)

    normalized, libraries = log1p_cpm_with_denominator_genes(counts, train_idx)
    changed_normalized, changed_libraries = log1p_cpm_with_denominator_genes(changed, train_idx)

    np.testing.assert_array_equal(libraries, changed_libraries)
    np.testing.assert_array_equal(normalized[:, train_idx], changed_normalized[:, train_idx])
    if int(np.count_nonzero(libraries == 0.0)) != 1:
        raise AssertionError("expected exactly one zero train-library spot")
    if np.any(normalized[1]) or np.any(changed_normalized[1]):
        raise AssertionError("zero train-library spots must normalize to an all-zero row")
    if np.array_equal(normalized[[0, 2], 2:], changed_normalized[[0, 2], 2:]):
        raise AssertionError("held-out values should still be normalized in nonzero-library spots")


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
) -> tuple[dict[str, np.ndarray | None], pd.DataFrame]:
    cache_candidates: list[Path] = []
    if args.base_cache_dir is not None:
        cache_candidates = [
            args.base_cache_dir / f"fold{fold}" / "gc_mlp_pca32_softplus_correct.npz",
            args.base_cache_dir / f"fold{fold}_canonical_gc_mlp_residual_maps.npz",
        ]

    if args.reuse_base:
        for cache in cache_candidates:
            if not cache.exists():
                continue
            cached = np.load(cache)
            if args.st_normalization_scope == "train_genes":
                required_normalization_meta = {
                    "st_normalization_scope",
                    "normalization_denominator_gene_idx",
                    "zero_train_library_strategy",
                }
                missing_normalization_meta = sorted(required_normalization_meta.difference(cached.files))
                if missing_normalization_meta:
                    print(
                        f"[PSP] fold{fold}: skipping base cache without strict ST normalization metadata: {cache}",
                        flush=True,
                    )
                    cached.close()
                    continue
                cache_scope = str(np.asarray(cached["st_normalization_scope"]).item())
                cache_strategy = str(np.asarray(cached["zero_train_library_strategy"]).item())
                cache_denominator_idx = cached["normalization_denominator_gene_idx"].astype(np.int64)
                if (
                    cache_scope != "train_genes"
                    or cache_strategy != ZERO_TRAIN_LIBRARY_STRATEGY
                    or not np.array_equal(cache_denominator_idx, train_idx.astype(np.int64))
                ):
                    print(
                        f"[PSP] fold{fold}: skipping base cache with incompatible strict ST normalization: {cache}",
                        flush=True,
                    )
                    cached.close()
                    continue
            required = {"pred_val", "pred_test"}
            missing = sorted(required.difference(cached.files))
            if missing:
                raise KeyError(f"GC cache {cache} is missing required arrays: {missing}")
            for key, expected in (("train_idx", train_idx), ("val_idx", val_idx), ("test_idx", test_idx)):
                if key in cached.files and not np.array_equal(cached[key].astype(np.int64), expected.astype(np.int64)):
                    raise ValueError(f"GC cache split mismatch for {key}: {cache}")
            return (
                {
                    "train": cached["pred_train"] if "pred_train" in cached.files else None,
                    "val": cached["pred_val"],
                    "test": cached["pred_test"],
                },
                pd.DataFrame([{"fold": fold, "source": "canonical_cache", "path": str(cache)}]),
            )

    if not args.allow_train_base:
        searched = ", ".join(str(path) for path in cache_candidates) or "no --base-cache-dir supplied"
        if args.st_normalization_scope == "train_genes":
            raise FileNotFoundError(
                "No GC cache with matching train-gene ST normalization metadata was found. "
                "Use a compatible cache or pass --allow-train-base to explicitly retrain the GC base "
                f"on the fold-specific X. Searched: {searched}"
            )
        raise FileNotFoundError(
            "Canonical GC cache was not found. Supply --base-cache-dir pointing to the frozen "
            "Vis9A final_multidataset_cache dataset directory. Refusing to train a replacement GC "
            f"model implicitly. Searched: {searched}"
        )

    residual_dir = args.out_dir / "gc_residual_maps_explicit_retrain"
    residual_dir.mkdir(parents=True, exist_ok=True)
    cache = residual_dir / f"fold{fold}_canonical_gc_mlp_residual_maps.npz"
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
    cache_payload = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "pred_train": preds["train"],
        "pred_val": preds["val"],
        "pred_test": preds["test"],
        "residual_train": X[:, train_idx].astype(np.float32) - preds["train"],
        "residual_val": X[:, val_idx].astype(np.float32) - preds["val"],
        "residual_test": X[:, test_idx].astype(np.float32) - preds["test"],
    }
    if args.st_normalization_scope == "train_genes":
        cache_payload.update(
            {
                "st_normalization_scope": np.asarray("train_genes"),
                "normalization_denominator_gene_idx": train_idx.astype(np.int64),
                "zero_train_library_strategy": np.asarray(ZERO_TRAIN_LIBRARY_STRATEGY),
            }
        )
    np.savez_compressed(cache, **cache_payload)
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
            mean_delta = float(np.mean(diff))
            delta_sd = float(np.std(diff, ddof=1)) if diff.size > 1 else np.nan
            delta_se = float(delta_sd / np.sqrt(diff.size)) if diff.size > 1 else np.nan
            if diff.size > 1 and np.isfinite(delta_se):
                half_width = float(st.t.ppf(0.975, df=diff.size - 1) * delta_se)
                ci95_low = mean_delta - half_width
                ci95_high = mean_delta + half_width
            else:
                ci95_low = np.nan
                ci95_high = np.nan
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
                    "mean_delta": mean_delta,
                    "median_delta": float(np.median(diff)),
                    "delta_sd": delta_sd,
                    "delta_se": delta_se,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                }
            )
    return pd.DataFrame(rows)


def save_prediction_matrix(
    root: Path,
    model: str,
    fold: int,
    prediction: np.ndarray,
    base_prediction: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    genes: list[str],
    descriptor: str,
    keep: np.ndarray,
    lambdas: dict[str, float],
    q1: float,
    q2: float,
    base_prediction_train: np.ndarray | None = None,
    base_prediction_val: np.ndarray | None = None,
    selected_prediction_train: np.ndarray | None = None,
    selected_prediction_val: np.ndarray | None = None,
    selected_prediction_test: np.ndarray | None = None,
) -> Path:
    path = root / model / f"fold{fold}" / "prediction.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Legacy test-only keys remain unchanged for downstream compatibility.
        "prediction": prediction.astype(np.float32),
        "base_prediction": base_prediction.astype(np.float32),
        "train_gene_idx": train_idx.astype(np.int64),
        "val_gene_idx": val_idx.astype(np.int64),
        "test_gene_idx": test_idx.astype(np.int64),
        "test_genes": np.asarray([genes[int(i)] for i in test_idx], dtype=object),
        "model": np.asarray(model, dtype=object),
        "fold": np.asarray(fold, dtype=np.int64),
        "base_descriptor": np.asarray("pca32", dtype=object),
        "psp_descriptor": np.asarray(descriptor, dtype=object),
        "component_keep": keep.astype(np.int64),
        "lambda_low": np.asarray(lambdas["low"], dtype=np.float64),
        "lambda_mid": np.asarray(lambdas["mid"], dtype=np.float64),
        "lambda_high": np.asarray(lambdas["high"], dtype=np.float64),
        "spatiality_q1": np.asarray(q1, dtype=np.float64),
        "spatiality_q2": np.asarray(q2, dtype=np.float64),
        "posthoc_calibration": np.asarray("none", dtype=object),
        "readout": np.asarray("identity", dtype=object),
    }
    if base_prediction_val is not None:
        payload["base_prediction_val"] = base_prediction_val.astype(np.float32)
        payload["base_prediction_test"] = base_prediction.astype(np.float32)
    if base_prediction_train is not None:
        payload["base_prediction_train"] = base_prediction_train.astype(np.float32)
    if selected_prediction_val is not None and selected_prediction_test is not None:
        payload["selected_prediction_val"] = selected_prediction_val.astype(np.float32)
        payload["selected_prediction_test"] = selected_prediction_test.astype(np.float32)
        payload["selected_rule_frozen_from_split"] = np.asarray("validation", dtype=object)
    if selected_prediction_train is not None:
        payload["selected_prediction_train"] = selected_prediction_train.astype(np.float32)
        payload["selected_train_coefficient_source"] = np.asarray(
            "ridge_descriptor_prediction_on_train_genes", dtype=object
        )
    np.savez_compressed(path, **payload)
    return path


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
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Spatial_count.txt"))
    ap.add_argument("--scrna-counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/scRNA_count.txt"))
    ap.add_argument("--locations-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Locations.txt"))
    ap.add_argument("--mask-dir", type=Path, default=INFO / "strict_whole_gene_masks")
    ap.add_argument(
        "--st-normalization-scope",
        choices=["all_genes", "train_genes"],
        default="train_genes",
        help=(
            "Genes used for each spot's ST library-size denominator. "
            "train_genes is the strict default; all_genes is retained only for labeled legacy diagnostics."
        ),
    )
    ap.add_argument(
        "--self-check-st-normalization",
        action="store_true",
        help="Run the lightweight train-only ST normalization invariance check and exit.",
    )
    ap.add_argument("--out-dir", type=Path, default=INFO)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-base", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--base-cache-dir",
        type=Path,
        default=None,
        help=(
            "Frozen canonical GC cache dataset directory. Expected layout: "
            "foldN/gc_mlp_pca32_softplus_correct.npz."
        ),
    )
    ap.add_argument(
        "--descriptor-cache",
        type=Path,
        default=None,
        help=(
            "Optional frozen descriptor NPZ containing pca32, nmf32, and "
            "pca32_nmf32 arrays in dataset gene order. When supplied, the "
            "scRNA count matrix is not reprocessed."
        ),
    )
    ap.add_argument(
        "--allow-train-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly permit retraining a replacement GC base when the frozen cache is unavailable.",
    )
    ap.add_argument(
        "--psp-descriptor",
        choices=["pca32", "nmf32", "pca32_nmf32"],
        default="pca32_nmf32",
    )
    ap.add_argument(
        "--allow-noncanonical-psp-descriptor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Permit descriptor sensitivity runs other than the frozen PCA32+NMF32 PSP configuration.",
    )
    ap.add_argument(
        "--save-prediction-matrices",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument(
        "--run-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run descriptor/basis/mean-coefficient PSP controls in addition to GC versus GC+PSP.",
    )
    ap.add_argument("--output-prefix", type=str, default="predictable_spatial_program_folds012")
    args = ap.parse_args()
    if args.self_check_st_normalization:
        self_check_train_gene_normalization()
        print("ST train-gene normalization self-check passed.")
        return
    if args.psp_descriptor != "pca32_nmf32" and not args.allow_noncanonical_psp_descriptor:
        ap.error(
            "The frozen PSP configuration uses --psp-descriptor pca32_nmf32. "
            "Pass --allow-noncanonical-psp-descriptor only for an explicitly labeled sensitivity run."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "psp_mechanism_figures"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_counts, genes, _ = load_matrix(args.counts_path, index_col=None)
    X_all_genes = log1p_cpm(X_counts) if args.st_normalization_scope == "all_genes" else None
    coords = pd.read_csv(args.locations_path, sep="\t").to_numpy(dtype=np.float32)
    edges = make_knn_edges(coords, k=8)
    if args.descriptor_cache is not None:
        cached_desc = np.load(args.descriptor_cache)
        required_desc = {"pca32", "nmf32", "pca32_nmf32"}
        missing_desc = sorted(required_desc.difference(cached_desc.files))
        if missing_desc:
            raise KeyError(f"Descriptor cache is missing required arrays: {missing_desc}")
        desc = {key: cached_desc[key].astype(np.float32) for key in required_desc}
        for key, value in desc.items():
            if value.shape[0] != len(genes):
                raise ValueError(
                    f"Descriptor cache gene count mismatch for {key}: "
                    f"{value.shape[0]} rows versus {len(genes)} dataset genes"
                )
    else:
        X_sc_counts, sc_genes, _ = load_matrix(args.scrna_counts_path, index_col=0)
        if list(sc_genes) != list(genes):
            sc_map = {g: i for i, g in enumerate(sc_genes)}
            X_sc_counts = X_sc_counts[:, [sc_map[g] for g in genes]]
        X_sc = log1p_cpm(X_sc_counts)
        desc = build_descriptors(X_sc, pca_dims=[32], nmf_dims=[32], seed=args.seed)
        desc["pca32_nmf32"] = np.concatenate([desc["pca32"], desc["nmf32"]], axis=1).astype(np.float32)
    D = desc[args.psp_descriptor]

    prediction_root = args.out_dir / f"{args.output_prefix}_prediction_matrices"

    long_rows = []
    gene_rows = []
    comp_rows = []
    figure_payloads = []
    lambda_selection_rows = []
    st_normalization_by_fold = []

    for fold in args.folds:
        print(f"[PSP] fold{fold}: loading masks/base", flush=True)
        mask_paths = {
            "train": args.mask_dir / f"fold{fold}_train_gene_idx.npy",
            "val": args.mask_dir / f"fold{fold}_val_gene_idx.npy",
            "test": args.mask_dir / f"fold{fold}_test_gene_idx.npy",
        }
        train_idx = np.load(mask_paths["train"])
        val_idx = np.load(mask_paths["val"])
        test_idx = np.load(mask_paths["test"])

        if args.st_normalization_scope == "all_genes":
            if X_all_genes is None:
                raise RuntimeError("legacy all-gene ST normalization was not initialized")
            X_fold = X_all_genes
            denominator_gene_count = int(X_counts.shape[1])
            all_gene_libraries = np.asarray(X_counts, dtype=np.float32).sum(axis=1, dtype=np.float64)
            zero_library_spot_count = int(np.count_nonzero(all_gene_libraries == 0.0))
            zero_library_strategy = "legacy_clip_library_size_to_at_least_1.0"
        else:
            X_fold_raw, normalization_audit = normalize_st_protocol_a(
                X_counts,
                inner_train_gene_idx=train_idx,
                val_gene_idx=val_idx,
                test_gene_idx=test_idx,
                require_complete_coverage=True,
            )
            X_fold = X_fold_raw.astype(np.float32)
            denominator_gene_count = int(
                normalization_audit["denominator_gene_count"]
            )
            zero_library_spot_count = int(
                normalization_audit["zero_train_library_spot_count"]
            )
            zero_library_strategy = str(
                normalization_audit["zero_train_library_policy"]
            )
            print(
                f"[PSP] fold{fold}: ST train-gene denominator uses {denominator_gene_count} genes; "
                f"zero train-library spots={zero_library_spot_count}; strategy={zero_library_strategy}",
                flush=True,
            )

        st_normalization_by_fold.append(
            {
                "fold": int(fold),
                "scope": args.st_normalization_scope,
                "denominator_gene_count": denominator_gene_count,
                "zero_library_spot_count": zero_library_spot_count,
                "zero_train_library_spot_count": (
                    zero_library_spot_count if args.st_normalization_scope == "train_genes" else None
                ),
                "zero_library_strategy": zero_library_strategy,
                "normalization_audit": (
                    normalization_audit
                    if args.st_normalization_scope == "train_genes"
                    else None
                ),
                "eligible_for_strict_primary": bool(
                    args.st_normalization_scope == "train_genes"
                ),
                "mask_paths": {name: str(path) for name, path in mask_paths.items()},
            }
        )

        low_val_idx, high_val_idx = subgroup_indices(X_fold, val_idx, coords)
        low_test_idx, high_test_idx = subgroup_indices(X_fold, test_idx, coords)
        base, _ = load_or_train_base_cached(
            X_fold, desc["pca32"], train_idx, val_idx, test_idx, device, args, fold
        )

        base_val_row = summarize_pred(
            "gc_mlp_base_val",
            X_fold,
            base["val"],
            val_idx,
            low_val_idx,
            high_val_idx,
            genes,
            {"fold": fold, "split": "val", "role": "base", "control": "base"},
        )
        base_test_row, base_gene = summarize_model(
            "gc_mlp_base",
            X_fold,
            base["test"],
            test_idx,
            low_test_idx,
            high_test_idx,
            genes,
            {"fold": fold, "split": "test", "role": "base", "control": "base"},
        )

        print(f"[PSP] fold{fold}: fitting SVD raw K=64 and val component predictability", flush=True)
        basis = fit_svd_raw_basis(X_fold[:, train_idx], k=64, seed=args.seed + fold)
        X_val_proc, val_meta = preprocess_train(X_fold[:, val_idx], "raw")
        C_val_oracle = project_coeff(basis.A, X_val_proc)
        C_val_pred = fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[val_idx], 10.0, seed=args.seed + fold)
        comp_df = component_stats(C_val_pred, C_val_oracle)
        comp_df.insert(0, "fold", fold)
        comp_df.insert(1, "method", "svd")
        comp_df.insert(2, "preprocess", "raw")
        comp_df.insert(3, "K", basis.k)
        comp_df.insert(4, "descriptor", args.psp_descriptor)
        comp_df.insert(5, "predictor", "ridge")
        comp_df.insert(6, "alpha", 10.0)
        comp_df["rank_score"] = comp_df["component_spearman"].fillna(-1.0) * np.log1p(comp_df["oracle_coeff_var"].clip(lower=0))
        comp_rows.append(comp_df)
        keep = comp_df.sort_values("rank_score", ascending=False).head(min(32, basis.k))["component"].to_numpy(dtype=np.int64)
        program_val = selected_component_prediction(basis.A, C_val_pred, val_meta, keep)

        train_sp = compute_spatiality(X_fold, train_idx, edges)
        pred_sp_val = fit_spatiality_predictor(D[train_idx], train_sp["MoranI"].to_numpy(dtype=np.float32), D[val_idx])
        lambdas, lambda_score, bin_val_df = bin_lambdas_from_val(
            X_fold,
            base["val"],
            program_val,
            val_idx,
            low_val_idx,
            high_val_idx,
            genes,
            pred_sp_val,
        )
        bin_val_df.insert(0, "fold", fold)
        bin_val_df["selected_lambda_score"] = lambda_score
        lambda_selection_rows.append(bin_val_df)
        q1, q2 = np.quantile(pred_sp_val, [1 / 3, 2 / 3])
        C_test_pred = fit_predict_coeff("ridge", D[train_idx], basis.C_train, D[test_idx], 10.0, seed=args.seed + 99 + fold)
        program_test = selected_component_prediction(basis.A, C_test_pred, basis.meta, keep)
        pred_sp_test = fit_spatiality_predictor(D[train_idx], train_sp["MoranI"].to_numpy(dtype=np.float32), D[test_idx])
        selected_test = apply_bin_lambdas(base["test"], program_test, pred_sp_test, (q1, q2), lambdas)
        selected_val = None
        selected_train = None
        if args.save_prediction_matrices:
            selected_val = apply_bin_lambdas(base["val"], program_val, pred_sp_val, (q1, q2), lambdas)
            if base["train"] is not None:
                C_train_pred = fit_predict_coeff(
                    "ridge",
                    D[train_idx],
                    basis.C_train,
                    D[train_idx],
                    10.0,
                    seed=args.seed + fold,
                )
                program_train = selected_component_prediction(basis.A, C_train_pred, basis.meta, keep)
                pred_sp_train = fit_spatiality_predictor(
                    D[train_idx],
                    train_sp["MoranI"].to_numpy(dtype=np.float32),
                    D[train_idx],
                )
                selected_train = apply_bin_lambdas(
                    base["train"], program_train, pred_sp_train, (q1, q2), lambdas
                )

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
            "descriptor": args.psp_descriptor,
            "predictor": "ridge",
            "alpha": 10.0,
            "topK_pred": len(keep),
            "lambda_mode": "predicted_spatiality_bins",
            "lambda_selected": json.dumps(lambdas, sort_keys=True),
            "component_keep": json.dumps([int(x) for x in keep]),
        }
        selected_row, selected_gene = summarize_model(
            "predictable_spatial_program_selected_correct",
            X_fold,
            selected_test,
            test_idx,
            low_test_idx,
            high_test_idx,
            genes,
            selected_meta,
        )

        fold_predictions = {
            "gc_mlp_base": base["test"],
            "predictable_spatial_program_selected_correct": selected_test,
        }
        controls = []
        if args.run_controls:
            controls = [
                ("shuffled_descriptor", make_descriptor_control(D, "shuffled", seed=args.seed + 1 + fold)[0], basis, False),
                ("random_descriptor", make_descriptor_control(D, "random", seed=args.seed + 2 + fold)[0], basis, False),
                ("permuted_labels", make_descriptor_control(D, "permuted_labels", seed=args.seed + 3 + fold)[0], basis, False),
            ]
            rng = np.random.default_rng(args.seed + 444 + fold)
            A_rand = rng.normal(0, float(np.std(basis.A) + 1e-6), size=basis.A.shape).astype(np.float32)
            rand_basis = Basis(
                "random_basis", "raw", basis.k, A_rand, project_coeff(A_rand, X_fold[:, train_idx]), basis.meta
            )
            A_perm = basis.A[rng.permutation(basis.A.shape[0])].astype(np.float32)
            perm_basis = Basis(
                "spot_permuted_basis", "raw", basis.k, A_perm, project_coeff(A_perm, X_fold[:, train_idx]), basis.meta
            )
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
                X_fold,
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
                    "descriptor": args.psp_descriptor,
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
            fold_predictions[f"predictable_spatial_program_{name}_control"] = pred_ctrl

        if args.save_prediction_matrices:
            for model_name, prediction in fold_predictions.items():
                is_base_model = model_name == "gc_mlp_base"
                is_selected_model = model_name == "predictable_spatial_program_selected_correct"
                include_base_splits = is_base_model or is_selected_model
                save_prediction_matrix(
                    prediction_root,
                    model_name,
                    fold,
                    prediction,
                    base["test"],
                    train_idx,
                    val_idx,
                    test_idx,
                    genes,
                    args.psp_descriptor,
                    keep,
                    lambdas,
                    float(q1),
                    float(q2),
                    base_prediction_train=base["train"] if include_base_splits else None,
                    base_prediction_val=base["val"] if include_base_splits else None,
                    selected_prediction_train=selected_train if is_selected_model else None,
                    selected_prediction_val=selected_val if is_selected_model else None,
                    selected_prediction_test=selected_test if is_selected_model else None,
                )

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

    metric_columns = ["SPCC", "SSIM", "RMSE", "JS"]
    completeness_rows = []
    for (model, fold), group in gene_df.groupby(["model", "fold"], sort=True):
        row = {
            "model": model,
            "fold": int(fold),
            "total_test_genes": int(len(group)),
        }
        for metric in metric_columns:
            finite = np.isfinite(group[metric].to_numpy(dtype=float))
            row[f"valid_{metric}"] = int(finite.sum())
            row[f"missing_{metric}"] = int((~finite).sum())
        completeness_rows.append(row)
    completeness = pd.DataFrame(completeness_rows)
    for fold, fold_completeness in completeness.groupby("fold", sort=True):
        if fold_completeness["total_test_genes"].nunique() != 1:
            raise RuntimeError(f"Model-dependent test-gene counts detected in fold {fold}")

    prefix = args.output_prefix
    long_path = args.out_dir / f"{prefix}_long.csv"
    summary_path = args.out_dir / f"{prefix}_summary.csv"
    paired_path = args.out_dir / f"{prefix}_paired_tests.csv"
    decision_path = args.out_dir / f"{prefix}_decision.md"
    gene_path = args.out_dir / f"{prefix}_gene_level.csv"
    comp_path = args.out_dir / f"{prefix}_component_predictability.csv"
    lambda_path = args.out_dir / f"{prefix}_lambda_selection.csv"
    completeness_path = args.out_dir / f"{prefix}_metric_completeness.csv"
    config_path = args.out_dir / f"{prefix}_run_config.json"

    long_df.to_csv(long_path, index=False)
    gene_df.to_csv(gene_path, index=False)
    comp_all.to_csv(comp_path, index=False)
    lambda_all.to_csv(lambda_path, index=False)
    completeness.to_csv(completeness_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "folds": [int(x) for x in args.folds],
                "seed": int(args.seed),
                "counts_path": str(args.counts_path),
                "scrna_counts_path": str(args.scrna_counts_path),
                "st_normalization_scope": args.st_normalization_scope,
                "st_normalization_target_sum": float(ST_TARGET_SUM),
                "st_normalization_by_fold": st_normalization_by_fold,
                "descriptor_cache": str(args.descriptor_cache) if args.descriptor_cache else None,
                "locations_path": str(args.locations_path),
                "mask_dir": str(args.mask_dir),
                "out_dir": str(args.out_dir),
                "output_prefix": args.output_prefix,
                "base_descriptor": "pca32",
                "psp_descriptor": args.psp_descriptor,
                "basis": "TruncatedSVD raw K=64",
                "coefficient_predictor": "ridge alpha=10",
                "component_selection": "top-32 ranked on validation genes",
                "fusion": "predicted-spatiality bins selected on validation genes",
                "posthoc_calibration": "none",
                "readout": "identity",
                "test_gene_metric_policy": (
                    "legacy metric table records non-finite values explicitly; final complete-set "
                    "evaluation is recomputed from saved prediction matrices"
                ),
                "prediction_matrices_saved": bool(args.save_prediction_matrices),
                "prediction_matrix_split_payload": {
                    "legacy_test_keys_preserved": True,
                    "base_train_saved_when_available": True,
                    "base_val_saved_for_core_models": True,
                    "selected_train_saved_when_base_train_available": True,
                    "selected_val_saved": True,
                    "selected_rule_frozen_from_split": "validation",
                    "selected_train_coefficient_source": "ridge_descriptor_prediction_on_train_genes",
                },
                "run_controls": bool(args.run_controls),
                "base_cache_dir": str(args.base_cache_dir) if args.base_cache_dir else None,
                "allow_train_base": bool(args.allow_train_base),
                "allow_noncanonical_psp_descriptor": bool(args.allow_noncanonical_psp_descriptor),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

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
    controls_ok = True if control_summary.empty else bool(
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
                "# Predictable Spatial Program Five-Fold Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "## Fixed Mechanism",
                "- basis: SVD raw K=64 learned from train genes only",
                f"- base descriptor: pca32",
                f"- PSP coefficient descriptor: {args.psp_descriptor}",
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
                f"- {completeness_path}",
                f"- {config_path}",
                f"- {prediction_root if args.save_prediction_matrices else 'prediction matrices not requested'}",
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
