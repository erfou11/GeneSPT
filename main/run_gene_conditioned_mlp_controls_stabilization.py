#!/usr/bin/env python3
"""Architecture-matched MLP controls and pointwise stabilization for strict genes."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from sklearn.linear_model import Ridge
from torch import nn

from run_strict_gene_conditioned_decoder_gate import (
    build_descriptors,
    gene_metrics,
    load_matrix,
    log1p_cpm,
    subgroup_indices,
    summarize_gene_df,
)


EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpotEncoder(nn.Module):
    def __init__(self, input_dim: int, z_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlexibleMLPDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        desc_dim: int,
        z_dim: int = 128,
        emb_dim: int = 96,
        output_mode: str = "linear",
        output_low: float = 0.0,
        output_high: float = 8.0,
    ):
        super().__init__()
        self.output_mode = str(output_mode)
        self.output_low = float(output_low)
        self.output_high = float(output_high)
        self.spot_encoder = SpotEncoder(input_dim, z_dim=z_dim)
        self.spot_proj = nn.Linear(z_dim, emb_dim)
        self.gene_proj = nn.Sequential(nn.LayerNorm(desc_dim), nn.Linear(desc_dim, emb_dim), nn.GELU())
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim * 2, 192),
            nn.GELU(),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "linear":
            return raw
        if self.output_mode == "softplus":
            return torch.nn.functional.softplus(raw)
        if self.output_mode == "sigmoid_range":
            return self.output_low + (self.output_high - self.output_low) * torch.sigmoid(raw)
        raise ValueError(f"Unknown output_mode={self.output_mode}")

    def forward_pairs(self, x_input: torch.Tensor, spot_idx: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        z_all = self.spot_encoder(x_input)
        z = self.spot_proj(z_all[spot_idx])
        e = self.gene_proj(desc)
        raw = self.decoder(torch.cat([z, e], dim=-1)).squeeze(-1)
        return self._activate(raw)

    def predict_matrix(self, x_input: torch.Tensor, desc_all: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        z_all = self.spot_proj(self.spot_encoder(x_input))
        outs = []
        for start in range(0, desc_all.shape[0], chunk):
            desc = desc_all[start : start + chunk]
            e = self.gene_proj(desc)
            z = z_all[:, None, :].expand(-1, e.shape[0], -1)
            ee = e[None, :, :].expand(z_all.shape[0], -1, -1)
            raw = self.decoder(torch.cat([z, ee], dim=-1).reshape(-1, z.shape[-1] + ee.shape[-1]))
            outs.append(self._activate(raw.reshape(z_all.shape[0], e.shape[0])))
        return torch.cat(outs, dim=1)


@dataclass
class MLPVariant:
    name: str
    descriptor: str
    control: str
    output_mode: str = "linear"
    dist_loss_weight: float = 0.0


def make_descriptor_control(desc: np.ndarray, control: str, seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    if control == "correct":
        return desc.astype(np.float32), "correct descriptor rows"
    if control == "shuffled":
        perm = rng.permutation(desc.shape[0])
        return desc[perm].astype(np.float32), "global gene-row shuffle"
    if control == "random":
        return rng.normal(loc=0.0, scale=float(np.std(desc) + 1e-6), size=desc.shape).astype(np.float32), "Gaussian random descriptor"
    if control == "permuted_labels":
        perm = rng.permutation(desc.shape[0])
        out = desc.copy()
        out[:] = desc[perm]
        return out.astype(np.float32), "permuted gene-label descriptor assignment"
    raise ValueError(control)


def train_mlp_variant(
    variant: MLPVariant,
    desc_np: np.ndarray,
    X: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_idx: np.ndarray,
    genes: list[str],
    low_test_idx: np.ndarray,
    high_spatial_idx: np.ndarray,
    output_low: float,
    output_high: float,
    device: torch.device,
    steps: int,
    batch_size: int,
    eval_every: int,
    lr: float,
    seed: int,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    set_seed(seed)
    x_input_np = X[:, input_idx].astype(np.float32)
    x_input = torch.tensor(x_input_np, device=device)
    y_full = torch.tensor(X, device=device)
    desc = torch.tensor(desc_np.astype(np.float32), device=device)
    model = FlexibleMLPDecoder(
        input_dim=x_input_np.shape[1],
        desc_dim=desc_np.shape[1],
        output_mode=variant.output_mode,
        output_low=output_low,
        output_high=output_high,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    best_score = -np.inf
    best_state = None
    history = []
    for step in range(1, steps + 1):
        spot_idx = torch.randint(0, X.shape[0], (batch_size,), device=device)
        gene_pos = torch.randint(0, len(train_idx), (batch_size,), device=device)
        gene_idx = train_idx_t[gene_pos]
        pred = model.forward_pairs(x_input, spot_idx, desc[gene_idx])
        target = y_full[spot_idx, gene_idx]
        loss = torch.mean((pred - target) ** 2)
        if float(variant.dist_loss_weight) > 0.0:
            n_gene = min(64, len(train_idx))
            sub_pos = torch.randperm(len(train_idx), device=device)[:n_gene]
            sub_gene = train_idx_t[sub_pos]
            pred_mat = model.predict_matrix(x_input, desc[sub_gene], chunk=64)
            true_mat = y_full[:, sub_gene]
            mean_loss = torch.mean((pred_mat.mean(dim=0) - true_mat.mean(dim=0)) ** 2)
            std_loss = torch.mean((pred_mat.std(dim=0) - true_mat.std(dim=0)) ** 2)
            loss = loss + float(variant.dist_loss_weight) * (mean_loss + std_loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                pred_val = model.predict_matrix(x_input, desc[val_idx_t], chunk=128).detach().cpu().numpy()
            X_pred_val = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
            X_pred_val[:, val_idx] = pred_val
            val_df = gene_metrics(X, X_pred_val, val_idx, genes)
            val_summary = summarize_gene_df(val_df)
            score = (
                np.nan_to_num(val_summary["SPCC"], nan=-1.0)
                - 0.05 * np.nan_to_num(val_summary["RMSE"], nan=10.0)
                - 0.05 * np.nan_to_num(val_summary["JS"], nan=10.0, posinf=10.0)
            )
            history.append({"step": step, "loss": float(loss.detach().cpu()), **{f"val_{k}": v for k, v in val_summary.items()}, "score": float(score)})
            if score > best_score:
                best_score = float(score)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred_test = model.predict_matrix(x_input, desc[torch.tensor(test_idx, device=device)], chunk=128).detach().cpu().numpy()
        pred_train = model.predict_matrix(x_input, desc[torch.tensor(train_idx, device=device)], chunk=128).detach().cpu().numpy()
    X_pred = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
    X_pred[:, test_idx] = pred_test
    summary = {
        "model": variant.name,
        "descriptor": variant.descriptor,
        "control": variant.control,
        "output_mode": variant.output_mode,
        "dist_loss_weight": float(variant.dist_loss_weight),
        "selected_by": "val_genes",
        "best_val_score": best_score,
        **summarize_gene_df(gene_metrics(X, X_pred, test_idx, genes)),
        **summarize_gene_df(gene_metrics(X, X_pred, low_test_idx, genes), prefix="low_expr_"),
        **summarize_gene_df(gene_metrics(X, X_pred, high_spatial_idx, genes), prefix="high_spatial_"),
    }
    return summary, pd.DataFrame(history), X_pred, pred_train


def diagnostic_predictions(X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, np.ndarray]:
    preds = {}
    spot_mean = X[:, train_idx].mean(axis=1, keepdims=True)
    pred = np.zeros_like(X, dtype=np.float32)
    pred[:, test_idx] = np.repeat(spot_mean, len(test_idx), axis=1)
    preds["spot_mean_diagnostic"] = pred
    global_mean = float(X[:, train_idx].mean())
    pred2 = np.zeros_like(X, dtype=np.float32)
    pred2[:, test_idx] = global_mean
    preds["global_mean_diagnostic"] = pred2
    return preds


def per_gene_distribution_audit(
    X: np.ndarray,
    X_pred: np.ndarray,
    test_idx: np.ndarray,
    genes: list[str],
    model: str,
) -> pd.DataFrame:
    metric_df = gene_metrics(X, X_pred, test_idx, genes)
    rows = []
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    metric_map = metric_df.set_index("gene_idx")
    for g in test_idx:
        y = X[:, g].astype(np.float64)
        p = X_pred[:, g].astype(np.float64)
        qerr = float(np.mean(np.abs(np.quantile(p, quantiles) - np.quantile(y, quantiles))))
        true_std = float(np.std(y))
        pred_std = float(np.std(p))
        metric = metric_map.loc[int(g)]
        rows.append(
            {
                "model": model,
                "gene_idx": int(g),
                "gene": str(genes[int(g)]),
                "pred_mean": float(np.mean(p)),
                "true_mean": float(np.mean(y)),
                "pred_std": pred_std,
                "true_std": true_std,
                "mean_error": float(np.mean(p) - np.mean(y)),
                "std_ratio": float(pred_std / max(true_std, EPS)),
                "pred_min": float(np.min(p)),
                "pred_max": float(np.max(p)),
                "negative_fraction": float(np.mean(p < 0.0)),
                "near_zero_fraction": float(np.mean(np.abs(p) < 1e-6)),
                "quantile_error": qerr,
                "per_gene_JS": float(metric["JS"]),
                "per_gene_SPCC": float(metric["SPCC"]),
                "per_gene_RMSE": float(metric["RMSE"]),
            }
        )
    return pd.DataFrame(rows)


def apply_clamp(X_pred: np.ndarray, train_values: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    lo = float(np.quantile(train_values, 0.001))
    hi = float(np.quantile(train_values, 0.999))
    out = X_pred.copy()
    out[:, test_idx] = np.clip(out[:, test_idx], lo, hi)
    return out


def fit_global_affine(train_true: np.ndarray, train_pred: np.ndarray, X_pred: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    p = train_pred.reshape(-1).astype(np.float64)
    y = train_true.reshape(-1).astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    out = X_pred.copy()
    out[:, test_idx] = coef[0] * out[:, test_idx] + coef[1]
    return out


def fit_descriptor_affine(
    train_true: np.ndarray,
    train_pred: np.ndarray,
    desc: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_pred: np.ndarray,
    shuffled: bool,
    seed: int,
) -> np.ndarray:
    a_targets = []
    b_targets = []
    for j in range(train_pred.shape[1]):
        p = train_pred[:, j].astype(np.float64)
        y = train_true[:, j].astype(np.float64)
        A = np.stack([p, np.ones_like(p)], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a_targets.append(float(np.clip(coef[0], -2.0, 4.0)))
        b_targets.append(float(np.clip(coef[1], -4.0, 8.0)))
    target = np.stack([a_targets, b_targets], axis=1)
    train_desc = desc[train_idx].copy()
    test_desc = desc[test_idx].copy()
    if shuffled:
        rng = np.random.default_rng(seed)
        train_desc = train_desc[rng.permutation(train_desc.shape[0])]
        test_desc = test_desc[rng.permutation(test_desc.shape[0])]
    ridge = Ridge(alpha=1.0)
    ridge.fit(train_desc, target)
    ab = ridge.predict(test_desc)
    a = np.clip(ab[:, 0], -2.0, 4.0)
    b = np.clip(ab[:, 1], -4.0, 8.0)
    out = X_pred.copy()
    out[:, test_idx] = out[:, test_idx] * a[None, :] + b[None, :]
    return out


def summarize_prediction(
    name: str,
    X: np.ndarray,
    X_pred: np.ndarray,
    test_idx: np.ndarray,
    low_test_idx: np.ndarray,
    high_spatial_idx: np.ndarray,
    genes: list[str],
    extra: dict | None = None,
) -> dict:
    row = {"model": name, **summarize_gene_df(gene_metrics(X, X_pred, test_idx, genes))}
    row.update(summarize_gene_df(gene_metrics(X, X_pred, low_test_idx, genes), prefix="low_expr_"))
    row.update(summarize_gene_df(gene_metrics(X, X_pred, high_spatial_idx, genes), prefix="high_spatial_"))
    if extra:
        row.update(extra)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Spatial_count.txt"))
    ap.add_argument("--scrna-counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/scRNA_count.txt"))
    ap.add_argument("--locations-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Locations.txt"))
    ap.add_argument("--mask-dir", type=Path, default=Path("/workspace/GeneSPT/results/imformation/strict_whole_gene_masks"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/GeneSPT/results/imformation"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_counts, genes, _ = load_matrix(args.counts_path, index_col=None)
    X = log1p_cpm(X_counts)
    coords = pd.read_csv(args.locations_path, sep="\t").to_numpy(dtype=np.float32)
    train_idx = np.load(args.mask_dir / f"fold{args.fold}_train_gene_idx.npy")
    val_idx = np.load(args.mask_dir / f"fold{args.fold}_val_gene_idx.npy")
    test_idx = np.load(args.mask_dir / f"fold{args.fold}_test_gene_idx.npy")
    low_test_idx, high_spatial_idx = subgroup_indices(X, test_idx, coords)
    train_values = X[:, train_idx].reshape(-1)

    X_sc_counts, sc_genes, _ = load_matrix(args.scrna_counts_path, index_col=0)
    if list(sc_genes) != list(genes):
        sc_map = {g: i for i, g in enumerate(sc_genes)}
        keep = [sc_map[g] for g in genes]
        X_sc_counts = X_sc_counts[:, keep]
    X_sc = log1p_cpm(X_sc_counts)
    descriptors = build_descriptors(X_sc, pca_dims=[16, 32, 64], nmf_dims=[32], seed=args.seed)
    descriptors["scrna_mean1"] = X_sc.mean(axis=0).astype(np.float32)[:, None]

    q001 = float(np.quantile(train_values, 0.001))
    q999 = float(np.quantile(train_values, 0.999))
    controls = [
        MLPVariant("mlp_pca32_correct", "pca32", "correct"),
        MLPVariant("mlp_pca32_shuffled", "pca32", "shuffled"),
        MLPVariant("mlp_pca32_random", "pca32", "random"),
        MLPVariant("mlp_pca32_permuted_labels", "pca32", "permuted_labels"),
        MLPVariant("mlp_pca16_correct", "pca16", "correct"),
        MLPVariant("mlp_pca64_correct", "pca64", "correct"),
        MLPVariant("mlp_nmf32_correct", "nmf32", "correct"),
        MLPVariant("mlp_scrna_mean1_correct", "scrna_mean1", "correct"),
    ]
    summaries = []
    histories = []
    preds = {}
    train_preds = {}
    for i, variant in enumerate(controls):
        desc_control, note = make_descriptor_control(descriptors[variant.descriptor], variant.control, seed=args.seed + 100 + i)
        summary, hist, pred, pred_train = train_mlp_variant(
            variant=variant,
            desc_np=desc_control,
            X=X,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            input_idx=train_idx,
            genes=genes,
            low_test_idx=low_test_idx,
            high_spatial_idx=high_spatial_idx,
            output_low=q001,
            output_high=q999,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            lr=args.lr,
            seed=args.seed + i * 17,
        )
        summary["control_note"] = note
        summaries.append(summary)
        hist["model"] = variant.name
        histories.append(hist)
        preds[variant.name] = pred
        train_preds[variant.name] = pred_train
    control_df = pd.DataFrame(summaries).sort_values("SPCC", ascending=False)
    control_df.to_csv(args.out_dir / "gene_conditioned_mlp_controls_fold0_long.csv", index=False)
    control_df.to_csv(args.out_dir / "gene_conditioned_mlp_controls_fold0_summary.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(args.out_dir / "gene_conditioned_mlp_controls_training_history.csv", index=False)

    correct_pca32 = control_df[control_df["model"] == "mlp_pca32_correct"].iloc[0]
    control_only = control_df[control_df["control"].isin(["shuffled", "random", "permuted_labels"])]
    best_control = control_only.sort_values("SPCC", ascending=False).iloc[0]
    controls_pass = (
        float(correct_pca32["SPCC"]) > float(best_control["SPCC"])
        and float(correct_pca32["RMSE"]) < float(best_control["RMSE"])
    )
    decision = "MLP_DESCRIPTOR_CONTROLS_PASS" if controls_pass else "MLP_DESCRIPTOR_CONTROLS_FAILED"
    (args.out_dir / "gene_conditioned_mlp_controls_decision.md").write_text(
        "\n".join(
            [
                "# Gene-Conditioned MLP Controls Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "## PCA32 Correct",
                correct_pca32.to_frame().T.to_string(index=False),
                "",
                "## Best Matched Control",
                best_control.to_frame().T.to_string(index=False),
                "",
                "Gate requires MLP PCA32 correct to beat MLP shuffled/random/permuted controls on SPCC and RMSE.",
            ]
        ),
        encoding="utf-8",
    )

    # Distribution diagnostic.
    diag_preds = diagnostic_predictions(X, train_idx, test_idx)
    best_control_name = str(best_control["model"])
    audit_parts = [
        per_gene_distribution_audit(X, preds["mlp_pca32_correct"], test_idx, genes, "mlp_pca32_correct"),
        per_gene_distribution_audit(X, diag_preds["spot_mean_diagnostic"], test_idx, genes, "spot_mean_diagnostic"),
        per_gene_distribution_audit(X, preds[best_control_name], test_idx, genes, best_control_name),
    ]
    audit_df = pd.concat(audit_parts, ignore_index=True)
    audit_df.to_csv(args.out_dir / "gene_conditioned_output_distribution_audit.csv", index=False)
    audit_summary = (
        audit_df.groupby("model")
        .agg(
            pred_mean_median=("pred_mean", "median"),
            mean_abs_error_median=("mean_error", lambda x: float(np.nanmedian(np.abs(x)))),
            std_ratio_median=("std_ratio", "median"),
            negative_fraction_median=("negative_fraction", "median"),
            near_zero_fraction_median=("near_zero_fraction", "median"),
            quantile_error_median=("quantile_error", "median"),
            inf_js_rate=("per_gene_JS", lambda x: float(np.mean(~np.isfinite(x)))),
            per_gene_JS_median=("per_gene_JS", lambda x: float(np.nanmedian(np.where(np.isfinite(x), x, np.nan)))),
            per_gene_SPCC_median=("per_gene_SPCC", "median"),
            per_gene_RMSE_median=("per_gene_RMSE", "median"),
        )
        .reset_index()
    )
    failure_lines = ["# Gene-Conditioned JS Failure Report", "", "## Distribution Summary", "", audit_summary.to_string(index=False), ""]
    mlp_audit = audit_summary[audit_summary["model"] == "mlp_pca32_correct"].iloc[0]
    spot_audit = audit_summary[audit_summary["model"] == "spot_mean_diagnostic"].iloc[0]
    causes = []
    if float(mlp_audit["negative_fraction_median"]) > 0.0:
        causes.append("negative/out-of-range predictions")
    if float(mlp_audit["std_ratio_median"]) < 0.75 or float(mlp_audit["std_ratio_median"]) > 1.35:
        causes.append("variance mismatch")
    if float(mlp_audit["mean_abs_error_median"]) > float(spot_audit["mean_abs_error_median"]):
        causes.append("mean shift")
    if float(mlp_audit["quantile_error_median"]) > float(spot_audit["quantile_error_median"]):
        causes.append("quantile/distribution mismatch")
    failure_lines.extend(
        [
            "## Likely JS Failure Sources",
            "",
            ", ".join(causes) if causes else "No single dominant distribution failure source identified.",
        ]
    )
    (args.out_dir / "gene_conditioned_js_failure_report.md").write_text("\n".join(failure_lines), encoding="utf-8")

    # Stabilization variants.
    stab_rows = []
    raw_pred = preds["mlp_pca32_correct"]
    raw_train_pred = train_preds["mlp_pca32_correct"]
    stab_rows.append(summarize_prediction("raw_linear_baseline", X, raw_pred, test_idx, low_test_idx, high_spatial_idx, genes, {"type": "baseline"}))
    stab_rows.append(summarize_prediction("posthoc_train_quantile_clamp", X, apply_clamp(raw_pred, train_values, test_idx), test_idx, low_test_idx, high_spatial_idx, genes, {"type": "range"}))
    stab_rows.append(
        summarize_prediction(
            "global_affine_train_genes",
            X,
            fit_global_affine(X[:, train_idx], raw_train_pred, raw_pred, test_idx),
            test_idx,
            low_test_idx,
            high_spatial_idx,
            genes,
            {"type": "affine"},
        )
    )
    stab_rows.append(
        summarize_prediction(
            "descriptor_affine_pca32",
            X,
            fit_descriptor_affine(X[:, train_idx], raw_train_pred, descriptors["pca32"], train_idx, test_idx, raw_pred, shuffled=False, seed=args.seed),
            test_idx,
            low_test_idx,
            high_spatial_idx,
            genes,
            {"type": "affine"},
        )
    )
    stab_rows.append(
        summarize_prediction(
            "descriptor_affine_pca32_shuffled_control",
            X,
            fit_descriptor_affine(X[:, train_idx], raw_train_pred, descriptors["pca32"], train_idx, test_idx, raw_pred, shuffled=True, seed=args.seed),
            test_idx,
            low_test_idx,
            high_spatial_idx,
            genes,
            {"type": "affine_control"},
        )
    )
    train_variants = [
        MLPVariant("softplus_output", "pca32", "correct", output_mode="softplus"),
        MLPVariant("sigmoid_train_quantile_range", "pca32", "correct", output_mode="sigmoid_range"),
        MLPVariant("mixed_loss_1e4", "pca32", "correct", output_mode="linear", dist_loss_weight=1e-4),
        MLPVariant("mixed_loss_3e4", "pca32", "correct", output_mode="linear", dist_loss_weight=3e-4),
        MLPVariant("mixed_loss_1e3", "pca32", "correct", output_mode="linear", dist_loss_weight=1e-3),
    ]
    for j, variant in enumerate(train_variants):
        summary, hist, pred, _ = train_mlp_variant(
            variant=variant,
            desc_np=descriptors["pca32"],
            X=X,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            input_idx=train_idx,
            genes=genes,
            low_test_idx=low_test_idx,
            high_spatial_idx=high_spatial_idx,
            output_low=q001,
            output_high=q999,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            lr=args.lr,
            seed=args.seed + 500 + j * 17,
        )
        summary["type"] = "trained_stabilization"
        stab_rows.append(summary)
    stab_df = pd.DataFrame(stab_rows)
    baseline = stab_df[stab_df["model"] == "raw_linear_baseline"].iloc[0]
    stab_df["delta_SPCC"] = stab_df["SPCC"] - float(baseline["SPCC"])
    stab_df["delta_RMSE"] = stab_df["RMSE"] - float(baseline["RMSE"])
    stab_df["delta_JS"] = stab_df["JS"] - float(baseline["JS"])
    stab_df["delta_SSIM"] = stab_df["SSIM"] - float(baseline["SSIM"])
    stab_df["gate_candidate"] = (
        (stab_df["SPCC"] >= float(baseline["SPCC"]) - 0.001)
        & (stab_df["RMSE"] <= float(baseline["RMSE"]) + 0.002)
        & (stab_df["JS"] < float(baseline["JS"]))
    )
    stab_df = stab_df.sort_values(["gate_candidate", "JS", "SPCC"], ascending=[False, True, False])
    stab_df.to_csv(args.out_dir / "gene_conditioned_decoder_stabilization_fold0.csv", index=False)
    best_stab = stab_df.iloc[0]
    stab_decision = "GENE_CONDITIONED_STABILIZATION_PASS" if bool(best_stab["gate_candidate"]) and controls_pass else "GENE_CONDITIONED_STABILIZATION_FAILED"
    (args.out_dir / "gene_conditioned_decoder_stabilization_decision.md").write_text(
        "\n".join(
            [
                "# Gene-Conditioned Decoder Stabilization Decision",
                "",
                f"Decision: `{stab_decision}`",
                "",
                "## Baseline",
                baseline.to_frame().T.to_string(index=False),
                "",
                "## Best Stabilized Candidate",
                best_stab.to_frame().T.to_string(index=False),
                "",
                "Gate requires SPCC >= raw baseline - 0.001, RMSE close/improved, JS improved, and descriptor controls passed.",
            ]
        ),
        encoding="utf-8",
    )

    print("Controls decision:", decision)
    print(control_df[["model", "SPCC", "RMSE", "JS", "SSIM", "low_expr_SPCC", "high_spatial_SPCC"]].to_string(index=False))
    print("Stabilization decision:", stab_decision)
    print(stab_df[["model", "SPCC", "RMSE", "JS", "SSIM", "delta_SPCC", "delta_RMSE", "delta_JS", "gate_candidate"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
