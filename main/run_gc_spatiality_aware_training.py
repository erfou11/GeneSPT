#!/usr/bin/env python3
"""Gene-spatiality-aware gene-conditioned learning for strict whole-gene.

This is an isolated strict whole-gene diagnostic. It does not use old
fixed-output legacy modules. Test-gene spatiality is used only for
final reporting subgroups, never for training or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from run_gene_conditioned_mlp_controls_stabilization import FlexibleMLPDecoder, make_descriptor_control
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


def moran_i(y: np.ndarray, edges: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    yc = y - y.mean()
    denom = float(np.sum(yc * yc))
    if denom < 1e-12 or edges.size == 0:
        return np.nan
    wij = 2.0 * edges.shape[0]
    num = float(np.sum(yc[edges[:, 0]] * yc[edges[:, 1]]) * 2.0)
    return float((len(y) / wij) * (num / denom))


def geary_c(y: np.ndarray, edges: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    yc = y - y.mean()
    denom = float(np.sum(yc * yc))
    if denom < 1e-12 or edges.size == 0:
        return np.nan
    diff2 = (y[edges[:, 0]] - y[edges[:, 1]]) ** 2
    return float(((len(y) - 1) / (2.0 * edges.shape[0])) * (np.sum(diff2) / denom))


def compute_spatiality(X: np.ndarray, gene_idx: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    rows = []
    for g in gene_idx:
        y = X[:, int(g)]
        rows.append({"gene_idx": int(g), "MoranI": moran_i(y, edges), "GearyC": geary_c(y, edges), "mean_expr": float(y.mean())})
    return pd.DataFrame(rows)


def load_descriptors(X_sc: np.ndarray, genes: list[str], seed: int) -> dict[str, np.ndarray]:
    desc = build_descriptors(X_sc, pca_dims=[32], nmf_dims=[32], seed=seed)
    desc["pca32_nmf32"] = np.concatenate([desc["pca32"], desc["nmf32"]], axis=1).astype(np.float32)
    desc["scrna_mean1"] = X_sc.mean(axis=0).astype(np.float32)[:, None]
    cluster_path = INFO / "strict_gene_descriptors_extended" / "cluster32_mean.npy"
    if cluster_path.exists():
        arr = np.load(cluster_path).astype(np.float32)
        if arr.shape[0] == len(genes):
            desc["cluster32"] = arr
    return desc


def descriptor_control(desc: np.ndarray, control: str, seed: int) -> np.ndarray:
    if control == "correct":
        return desc.astype(np.float32)
    if control == "permuted_labels":
        return make_descriptor_control(desc, "permuted_labels", seed=seed)[0]
    return make_descriptor_control(desc, control, seed=seed)[0]


def safe_auc(y_true: np.ndarray, score: np.ndarray, kind: str) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    if kind == "roc":
        return float(roc_auc_score(y_true, score))
    return float(average_precision_score(y_true, score))


def run_predictability_audit(args) -> tuple[pd.DataFrame, str]:
    X_counts, genes, _ = load_matrix(args.counts_path, index_col=None)
    X = log1p_cpm(X_counts)
    coords = pd.read_csv(args.locations_path, sep="\t").to_numpy(dtype=np.float32)
    edges = make_knn_edges(coords, k=8)
    X_sc_counts, sc_genes, _ = load_matrix(args.scrna_counts_path, index_col=0)
    if list(sc_genes) != list(genes):
        sc_map = {g: i for i, g in enumerate(sc_genes)}
        X_sc_counts = X_sc_counts[:, [sc_map[g] for g in genes]]
    X_sc = log1p_cpm(X_sc_counts)
    desc = load_descriptors(X_sc, genes, seed=args.seed)
    rows = []
    for fold in range(5):
        train_idx = np.load(args.mask_dir / f"fold{fold}_train_gene_idx.npy")
        val_idx = np.load(args.mask_dir / f"fold{fold}_val_gene_idx.npy")
        train_sp = compute_spatiality(X, train_idx, edges)
        val_sp = compute_spatiality(X, val_idx, edges)
        high_thr = float(np.nanquantile(train_sp["MoranI"], 0.70))
        y_train = train_sp["MoranI"].to_numpy(dtype=np.float64)
        y_val = val_sp["MoranI"].to_numpy(dtype=np.float64)
        high_train = (y_train >= high_thr).astype(int)
        high_val = (y_val >= high_thr).astype(int)
        for desc_name, D0 in desc.items():
            for control in ["correct", "shuffled", "random", "permuted_labels"]:
                D = descriptor_control(D0, control, seed=args.seed + 1000 * fold + len(desc_name) + len(control))
                Xtr = D[train_idx]
                Xva = D[val_idx]
                ridge = Ridge(alpha=1.0)
                ridge.fit(Xtr, y_train)
                pred_val = ridge.predict(Xva)
                try:
                    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
                    clf.fit(Xtr, high_train)
                    prob_val = clf.predict_proba(Xva)[:, 1]
                except Exception:
                    prob_val = pred_val
                rows.append(
                    {
                        "fold": fold,
                        "descriptor": desc_name,
                        "control": control,
                        "n_train": int(len(train_idx)),
                        "n_val": int(len(val_idx)),
                        "train_high_threshold": high_thr,
                        "moran_spearman": float(st.spearmanr(y_val, pred_val, nan_policy="omit").correlation),
                        "moran_pearson": float(np.corrcoef(y_val, pred_val)[0, 1]) if np.std(pred_val) > 1e-12 else np.nan,
                        "high_spatial_auroc": safe_auc(high_val, prob_val, "roc"),
                        "high_spatial_auprc": safe_auc(high_val, prob_val, "pr"),
                        "val_high_rate": float(high_val.mean()),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "gene_spatiality_predictability_audit.csv", index=False)
    summary = (
        df.groupby(["descriptor", "control"], as_index=False)
        .agg(
            moran_spearman_mean=("moran_spearman", "mean"),
            moran_spearman_std=("moran_spearman", "std"),
            auroc_mean=("high_spatial_auroc", "mean"),
            auprc_mean=("high_spatial_auprc", "mean"),
        )
        .sort_values(["moran_spearman_mean", "auroc_mean"], ascending=False)
    )
    best_correct = summary[summary["control"].eq("correct")].iloc[0]
    best_control = summary[~summary["control"].eq("correct")].iloc[0]
    passes = bool(
        best_correct["moran_spearman_mean"] > best_control["moran_spearman_mean"] + 0.02
        and best_correct["auroc_mean"] >= max(0.55, best_control["auroc_mean"] + 0.01)
    )
    decision = "SPATIALITY_PREDICTABLE_CONTINUE" if passes else "SPATIALITY_PREDICTABILITY_FAILED"
    lines = [
        "# Gene Spatiality Predictability Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "## Summary",
        summary.to_string(index=False),
        "",
        "## Best Correct Descriptor",
        best_correct.to_string(),
        "",
        "## Best Control",
        best_control.to_string(),
        "",
        "Continue only if correct descriptors predict train/val spatiality better than matched controls.",
    ]
    (args.out_dir / "gene_spatiality_predictability_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df, decision


@dataclass
class TrainVariant:
    name: str
    mode: str
    value: float = 0.0
    control: str = "correct"
    descriptor_extra: str = "none"


def train_variant(
    variant: TrainVariant,
    X: np.ndarray,
    desc_np: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    train_spatiality: np.ndarray,
    train_bins: np.ndarray,
    low_test_idx: np.ndarray,
    high_test_idx: np.ndarray,
    genes: list[str],
    device: torch.device,
    steps: int,
    batch_size: int,
    eval_every: int,
    lr: float,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    set_seed(seed)
    x_input_np = X[:, train_idx].astype(np.float32)
    x_input = torch.tensor(x_input_np, device=device)
    y_full = torch.tensor(X.astype(np.float32), device=device)
    desc = torch.tensor(desc_np.astype(np.float32), device=device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    model = FlexibleMLPDecoder(input_dim=x_input_np.shape[1], desc_dim=desc_np.shape[1], output_mode="softplus").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    spatiality = np.asarray(train_spatiality, dtype=np.float32)
    spatiality_norm = (spatiality - np.nanmin(spatiality)) / max(float(np.nanmax(spatiality) - np.nanmin(spatiality)), 1e-6)
    weights = np.ones(len(train_idx), dtype=np.float32)
    if variant.mode == "high_weight":
        weights[train_bins == 2] = float(variant.value)
    elif variant.mode == "continuous":
        weights = 1.0 + float(variant.value) * spatiality_norm
    elif variant.mode == "shuffled_weight":
        w = np.ones(len(train_idx), dtype=np.float32)
        w[train_bins == 2] = float(variant.value)
        weights = np.random.default_rng(seed + 7).permutation(w).astype(np.float32)
    elif variant.mode == "random_weight":
        weights = np.random.default_rng(seed + 9).uniform(1.0, float(variant.value), size=len(train_idx)).astype(np.float32)
    elif variant.mode == "inverse_low_weight":
        weights[train_bins == 0] = float(variant.value)
    weights_t = torch.tensor(weights, device=device)
    bin_pos = {b: np.where(train_bins == b)[0] for b in [0, 1, 2]}
    best_score = -np.inf
    best_state = None
    hist = []
    for step in range(1, steps + 1):
        spot_idx = torch.randint(0, X.shape[0], (batch_size,), device=device)
        if variant.mode == "balanced_bins":
            per = batch_size // 3
            chosen = []
            rng = np.random.default_rng(seed + step)
            for b in [0, 1, 2]:
                pool = bin_pos[b] if len(bin_pos[b]) else np.arange(len(train_idx))
                chosen.append(rng.choice(pool, size=per, replace=True))
            rem = batch_size - per * 3
            if rem > 0:
                chosen.append(rng.choice(np.arange(len(train_idx)), size=rem, replace=True))
            gene_pos_np = np.concatenate(chosen)
            rng.shuffle(gene_pos_np)
            gene_pos = torch.tensor(gene_pos_np, dtype=torch.long, device=device)
        else:
            gene_pos = torch.randint(0, len(train_idx), (batch_size,), device=device)
        gene_idx = train_idx_t[gene_pos]
        pred = model.forward_pairs(x_input, spot_idx, desc[gene_idx])
        target = y_full[spot_idx, gene_idx]
        loss_vec = (pred - target) ** 2
        if variant.mode in {"high_weight", "continuous", "shuffled_weight", "random_weight", "inverse_low_weight"}:
            loss = torch.mean(loss_vec * weights_t[gene_pos])
        else:
            loss = torch.mean(loss_vec)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                pred_val = model.predict_matrix(x_input, desc[val_idx_t], chunk=128).detach().cpu().numpy()
            X_pred_val = np.zeros_like(X, dtype=np.float32)
            X_pred_val[:, val_idx] = pred_val
            val_summary = summarize_gene_df(gene_metrics(X, X_pred_val, val_idx, genes))
            val_score = (
                np.nan_to_num(val_summary["SPCC"], nan=-1.0)
                - 0.05 * np.nan_to_num(val_summary["RMSE"], nan=10.0)
                - 0.05 * np.nan_to_num(val_summary["JS"], nan=10.0, posinf=10.0)
            )
            hist.append({"step": step, "loss": float(loss.detach().cpu()), **{f"val_{k}": v for k, v in val_summary.items()}, "score": float(val_score)})
            if val_score > best_score:
                best_score = float(val_score)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred_test = model.predict_matrix(x_input, desc[torch.tensor(test_idx, device=device)], chunk=128).detach().cpu().numpy()
    X_pred = np.zeros_like(X, dtype=np.float32)
    X_pred[:, test_idx] = pred_test
    row = {
        "model": variant.name,
        "mode": variant.mode,
        "value": float(variant.value),
        "control": variant.control,
        "descriptor_extra": variant.descriptor_extra,
        "best_val_score": best_score,
        **summarize_gene_df(gene_metrics(X, X_pred, test_idx, genes)),
        **summarize_gene_df(gene_metrics(X, X_pred, low_test_idx, genes), prefix="low_expr_"),
        **summarize_gene_df(gene_metrics(X, X_pred, high_test_idx, genes), prefix="high_spatial_"),
    }
    return row, pd.DataFrame(hist)


def run_training_gate(args) -> pd.DataFrame:
    X_counts, genes, _ = load_matrix(args.counts_path, index_col=None)
    X = log1p_cpm(X_counts)
    coords = pd.read_csv(args.locations_path, sep="\t").to_numpy(dtype=np.float32)
    edges = make_knn_edges(coords, k=8)
    X_sc_counts, sc_genes, _ = load_matrix(args.scrna_counts_path, index_col=0)
    if list(sc_genes) != list(genes):
        sc_map = {g: i for i, g in enumerate(sc_genes)}
        X_sc_counts = X_sc_counts[:, [sc_map[g] for g in genes]]
    X_sc = log1p_cpm(X_sc_counts)
    desc = load_descriptors(X_sc, genes, seed=args.seed)
    fold = int(args.fold)
    train_idx = np.load(args.mask_dir / f"fold{fold}_train_gene_idx.npy")
    val_idx = np.load(args.mask_dir / f"fold{fold}_val_gene_idx.npy")
    test_idx = np.load(args.mask_dir / f"fold{fold}_test_gene_idx.npy")
    low_test_idx, high_test_idx = subgroup_indices(X, test_idx, coords)
    train_sp = compute_spatiality(X, train_idx, edges)
    moran = train_sp["MoranI"].to_numpy(dtype=np.float32)
    q30, q70 = np.nanquantile(moran, [0.30, 0.70])
    bins = np.zeros(len(train_idx), dtype=np.int64)
    bins[moran >= q70] = 2
    bins[(moran > q30) & (moran < q70)] = 1
    # Predicted spatiality scalar from train genes only.
    pred_model = Ridge(alpha=1.0).fit(desc["pca32"][train_idx], moran)
    pred_sp = pred_model.predict(desc["pca32"]).astype(np.float32)
    pred_sp = ((pred_sp - pred_sp[train_idx].mean()) / max(float(pred_sp[train_idx].std()), 1e-6))[:, None]
    rng = np.random.default_rng(args.seed + 99)
    pred_sp_shuf = rng.permutation(pred_sp).astype(np.float32)
    desc_base = desc["pca32"].astype(np.float32)
    variants = [
        (TrainVariant("canonical_gc_mlp_baseline", "baseline"), desc_base),
        (TrainVariant("high_spatial_weight_1p5", "high_weight", 1.5), desc_base),
        (TrainVariant("high_spatial_weight_2p0", "high_weight", 2.0), desc_base),
        (TrainVariant("high_spatial_weight_3p0", "high_weight", 3.0), desc_base),
        (TrainVariant("balanced_spatiality_bin_sampling", "balanced_bins"), desc_base),
        (TrainVariant("continuous_moran_weight_0p5", "continuous", 0.5), desc_base),
        (TrainVariant("continuous_moran_weight_1p0", "continuous", 1.0), desc_base),
        (TrainVariant("continuous_moran_weight_2p0", "continuous", 2.0), desc_base),
        (TrainVariant("predicted_spatiality_feature", "baseline", 0.0, "correct", "predicted_spatiality"), np.concatenate([desc_base, pred_sp], axis=1).astype(np.float32)),
        (TrainVariant("shuffled_spatiality_weight_control", "shuffled_weight", 2.0, "shuffled_weight"), desc_base),
        (TrainVariant("random_weight_control", "random_weight", 3.0, "random_weight"), desc_base),
        (TrainVariant("inverse_low_spatial_weight_control", "inverse_low_weight", 2.0, "inverse_low"), desc_base),
        (TrainVariant("predicted_spatiality_shuffled_descriptor_control", "baseline", 0.0, "shuffled_pred_spatiality", "predicted_spatiality_shuffled"), np.concatenate([desc_base, pred_sp_shuf], axis=1).astype(np.float32)),
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    histories = []
    for i, (variant, D) in enumerate(variants):
        row, hist = train_variant(
            variant,
            X,
            D,
            train_idx,
            val_idx,
            test_idx,
            moran,
            bins,
            low_test_idx,
            high_test_idx,
            genes,
            device,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            lr=args.lr,
            seed=args.seed + 100 * i,
        )
        rows.append(row)
        hist.insert(0, "model", variant.name)
        histories.append(hist)
        print(f"completed {variant.name}: SPCC={row['SPCC']:.6f} high_spatial={row['high_spatial_SPCC']:.6f}", flush=True)
    df = pd.DataFrame(rows)
    base = df[df["model"].eq("canonical_gc_mlp_baseline")].iloc[0]
    for metric in ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]:
        df[f"delta_{metric}_vs_base"] = df[metric].astype(float) - float(base[metric])
    df.to_csv(args.out_dir / "gc_spatiality_aware_training_fold0_long.csv", index=False)
    df.to_csv(args.out_dir / "gc_spatiality_aware_training_fold0_summary.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(args.out_dir / "gc_spatiality_aware_training_history_fold0.csv", index=False)
    controls = df[df["control"].ne("correct") & ~df["model"].eq("canonical_gc_mlp_baseline")]
    candidates = df[df["control"].eq("correct") & ~df["model"].eq("canonical_gc_mlp_baseline")]
    if candidates.empty:
        decision = "SPATIALITY_AWARE_FAILED"
        best = base
        controls_ok = False
    else:
        best = candidates.sort_values(["high_spatial_SPCC", "SPCC"], ascending=[False, False]).iloc[0]
        controls_ok = bool(
            best["high_spatial_SPCC"] > controls["high_spatial_SPCC"].max()
            and best["SPCC"] > controls["SPCC"].max()
        ) if not controls.empty else False
        main_gate = bool(
            (best["delta_high_spatial_SPCC_vs_base"] >= 0.01 or best["delta_high_spatial_RMSE_vs_base"] <= -0.005)
            and best["delta_SPCC_vs_base"] >= -0.001
            and best["delta_RMSE_vs_base"] <= 0.002
            and best["delta_JS_vs_base"] <= 0.003
            and controls_ok
        )
        aux_gate = bool(
            not main_gate
            and best["delta_high_spatial_SPCC_vs_base"] > 0
            and best["delta_SPCC_vs_base"] >= -0.002
            and best["delta_JS_vs_base"] <= 0.003
        )
        decision = "SPATIALITY_AWARE_CONTINUE" if main_gate else ("SPATIALITY_AWARE_AUXILIARY" if aux_gate else "SPATIALITY_AWARE_FAILED")
    lines = [
        "# GC Spatiality-Aware Training Fold0 Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "## Mask Safety",
        "- Train-gene MoranI/Geary are used only for train-gene weighting/sampling.",
        "- Val genes select variants/checkpoints.",
        "- Test-gene spatiality is used only for final subgroup reporting.",
        "- No legacy fixed-output modules are used.",
        "",
        "## Fold0 Summary",
        df[["model", "SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "delta_SPCC_vs_base", "delta_RMSE_vs_base", "delta_JS_vs_base", "delta_high_spatial_SPCC_vs_base"]].to_string(index=False),
        "",
        f"Best candidate: `{best['model']}`",
        f"Controls OK: `{controls_ok}`",
    ]
    (args.out_dir / "gc_spatiality_aware_training_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Spatial_count.txt"))
    ap.add_argument("--scrna-counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/scRNA_count.txt"))
    ap.add_argument("--locations-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Locations.txt"))
    ap.add_argument("--mask-dir", type=Path, default=INFO / "strict_whole_gene_masks")
    ap.add_argument("--out-dir", type=Path, default=INFO)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, audit_decision = run_predictability_audit(args)
    if args.audit_only or audit_decision != "SPATIALITY_PREDICTABLE_CONTINUE":
        print(f"Audit decision: {audit_decision}")
        return
    run_training_gate(args)


if __name__ == "__main__":
    main()
