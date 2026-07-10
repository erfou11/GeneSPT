#!/usr/bin/env python3
"""Fold0 strict whole-gene gene-conditioned decoder gate.

This script is intentionally independent from the main GeneSPT training
pipeline.  The goal is to test whether a shared spot-gene decoder can predict
strict held-out genes using descriptors that are available for both train and
test genes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from sklearn.decomposition import MiniBatchNMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from torch import nn


EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_matrix(path: Path, index_col: int | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    df = pd.read_csv(path, sep="\t", index_col=index_col)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    obs_names = list(map(str, df.index)) if index_col is not None else [str(i) for i in range(df.shape[0])]
    return df.to_numpy(dtype=np.float32), list(map(str, df.columns)), obs_names


def log1p_cpm(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    lib = np.clip(X.sum(axis=1, keepdims=True), 1.0, None)
    return np.log1p((X / lib) * float(target_sum)).astype(np.float32)


def scale_max(x: np.ndarray) -> np.ndarray:
    denom = float(np.nanmax(x)) if x.size else 0.0
    if abs(denom) < 1e-20:
        denom = 1.0
    return np.asarray(x, dtype=np.float64) / denom


def scale_plus(x: np.ndarray) -> np.ndarray:
    denom = float(np.nansum(x)) if x.size else 0.0
    if abs(denom) < 1e-20:
        denom = 1.0
    return np.asarray(x, dtype=np.float64) / denom


def scale_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.nanstd(x))
    if sd < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - float(np.nanmean(x))) / sd


def cal_ssim_ref(im1: np.ndarray, im2: np.ndarray, M: float) -> float:
    im1 = np.asarray(im1, dtype=np.float64).reshape(-1)
    im2 = np.asarray(im2, dtype=np.float64).reshape(-1)
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return float(l12 * c12 * s12)


def gene_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    gene_idx: np.ndarray,
    gene_names: list[str],
    subgroup: str = "all",
) -> pd.DataFrame:
    rows = []
    for g in gene_idx:
        y = np.asarray(X_true[:, g], dtype=np.float64)
        p = np.asarray(X_pred[:, g], dtype=np.float64)
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
        rows.append(
            {
                "gene_idx": int(g),
                "gene": str(gene_names[int(g)]),
                "subgroup": subgroup,
                "SPCC": spcc,
                "SSIM": ssim,
                "RMSE": rmse,
                "JS": js,
                "true_mean": float(np.mean(y)),
                "true_std": float(np.std(y)),
                "pred_mean": float(np.mean(p)),
                "pred_std": float(np.std(p)),
            }
        )
    return pd.DataFrame(rows)


def summarize_gene_df(df: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    if df.empty:
        return {f"{prefix}{k}": np.nan for k in ["SPCC", "SSIM", "RMSE", "JS"]}
    return {
        f"{prefix}SPCC": float(np.nanmedian(df["SPCC"])),
        f"{prefix}SSIM": float(np.nanmedian(df["SSIM"])),
        f"{prefix}RMSE": float(np.nanmedian(df["RMSE"])),
        f"{prefix}JS": float(np.nanmedian(df["JS"])),
    }


def make_knn_edges(coords: np.ndarray, k: int = 8) -> np.ndarray:
    nbr = NearestNeighbors(n_neighbors=min(k + 1, coords.shape[0]), metric="euclidean")
    nbr.fit(coords)
    _, ind = nbr.kneighbors(coords)
    edges = set()
    for i in range(coords.shape[0]):
        for j in ind[i, 1:]:
            a, b = sorted((int(i), int(j)))
            edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int64)


def moran_i(y: np.ndarray, edges: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    yc = y - y.mean()
    denom = float(np.sum(yc * yc))
    if denom < EPS or edges.size == 0:
        return np.nan
    wij = 2.0 * edges.shape[0]
    num = float(np.sum(yc[edges[:, 0]] * yc[edges[:, 1]]) * 2.0)
    return float((len(y) / wij) * (num / denom))


def subgroup_indices(X: np.ndarray, test_idx: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expr_mean = X[:, test_idx].mean(axis=0)
    low_cut = np.nanquantile(expr_mean, 0.30)
    low_expr = test_idx[expr_mean <= low_cut]
    edges = make_knn_edges(coords, k=8)
    morans = np.asarray([moran_i(X[:, g], edges) for g in test_idx], dtype=np.float64)
    high_cut = np.nanquantile(morans, 0.70)
    high_spatial = test_idx[morans >= high_cut]
    return low_expr.astype(np.int64), high_spatial.astype(np.int64)


def build_descriptors(
    X_scrna: np.ndarray,
    pca_dims: Iterable[int],
    nmf_dims: Iterable[int],
    seed: int,
) -> dict[str, np.ndarray]:
    descriptors: dict[str, np.ndarray] = {}
    gene_by_cell = np.asarray(X_scrna.T, dtype=np.float32)
    for dim in pca_dims:
        svd = TruncatedSVD(n_components=int(dim), random_state=seed)
        descriptors[f"pca{dim}"] = svd.fit_transform(gene_by_cell).astype(np.float32)
    for dim in nmf_dims:
        nmf = MiniBatchNMF(
            n_components=int(dim),
            random_state=seed,
            max_iter=250,
            batch_size=512,
            init="nndsvda",
            beta_loss="frobenius",
        )
        descriptors[f"nmf{dim}"] = nmf.fit_transform(np.clip(gene_by_cell, 0.0, None)).astype(np.float32)
    return descriptors


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


class BilinearDecoder(nn.Module):
    def __init__(self, input_dim: int, desc_dim: int, z_dim: int = 128, factor_dim: int = 96):
        super().__init__()
        self.spot_encoder = SpotEncoder(input_dim, z_dim=z_dim)
        self.spot_proj = nn.Linear(z_dim, factor_dim)
        self.gene_proj = nn.Sequential(nn.LayerNorm(desc_dim), nn.Linear(desc_dim, factor_dim))
        self.gene_bias = nn.Sequential(nn.LayerNorm(desc_dim), nn.Linear(desc_dim, 1))
        self.global_bias = nn.Parameter(torch.zeros(()))

    def forward_pairs(self, x_input: torch.Tensor, spot_idx: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        z_all = self.spot_encoder(x_input)
        z = self.spot_proj(z_all[spot_idx])
        e = self.gene_proj(desc)
        b = self.gene_bias(desc).squeeze(-1)
        return (z * e).sum(dim=-1) / math.sqrt(z.shape[-1]) + b + self.global_bias

    def predict_matrix(self, x_input: torch.Tensor, desc_all: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        z = self.spot_proj(self.spot_encoder(x_input))
        outs = []
        for start in range(0, desc_all.shape[0], chunk):
            desc = desc_all[start : start + chunk]
            e = self.gene_proj(desc)
            b = self.gene_bias(desc).squeeze(-1)
            outs.append(z @ e.T / math.sqrt(z.shape[-1]) + b.unsqueeze(0) + self.global_bias)
        return torch.cat(outs, dim=1)


class MLPDecoder(nn.Module):
    def __init__(self, input_dim: int, desc_dim: int, z_dim: int = 128, emb_dim: int = 96):
        super().__init__()
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

    def forward_pairs(self, x_input: torch.Tensor, spot_idx: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        z_all = self.spot_encoder(x_input)
        z = self.spot_proj(z_all[spot_idx])
        e = self.gene_proj(desc)
        return self.decoder(torch.cat([z, e], dim=-1)).squeeze(-1)

    def predict_matrix(self, x_input: torch.Tensor, desc_all: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        z_all = self.spot_proj(self.spot_encoder(x_input))
        outs = []
        for start in range(0, desc_all.shape[0], chunk):
            desc = desc_all[start : start + chunk]
            e = self.gene_proj(desc)
            z = z_all[:, None, :].expand(-1, e.shape[0], -1)
            ee = e[None, :, :].expand(z_all.shape[0], -1, -1)
            out = self.decoder(torch.cat([z, ee], dim=-1).reshape(-1, z.shape[-1] + ee.shape[-1]))
            outs.append(out.reshape(z_all.shape[0], e.shape[0]))
        return torch.cat(outs, dim=1)


@dataclass
class Variant:
    name: str
    decoder: str
    descriptor: str
    control: str = "correct"


def make_control_descriptor(desc: np.ndarray, control: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if control == "correct":
        return desc.astype(np.float32)
    if control == "shuffled":
        perm = rng.permutation(desc.shape[0])
        return desc[perm].astype(np.float32)
    if control == "random":
        return rng.normal(size=desc.shape).astype(np.float32)
    raise ValueError(f"Unknown descriptor control: {control}")


def train_variant(
    variant: Variant,
    desc_np: np.ndarray,
    X: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_idx: np.ndarray,
    genes: list[str],
    low_test_idx: np.ndarray,
    high_spatial_idx: np.ndarray,
    device: torch.device,
    steps: int,
    batch_size: int,
    eval_every: int,
    lr: float,
    seed: int,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    set_seed(seed)
    x_input_np = X[:, input_idx].astype(np.float32)
    x_input = torch.tensor(x_input_np, device=device)
    y_full = torch.tensor(X, device=device)
    desc = torch.tensor(desc_np, device=device)
    desc_dim = int(desc_np.shape[1])
    if variant.decoder == "bilinear" or variant.decoder == "lowrank":
        model = BilinearDecoder(input_dim=x_input_np.shape[1], desc_dim=desc_dim).to(device)
    elif variant.decoder == "mlp":
        model = MLPDecoder(input_dim=x_input_np.shape[1], desc_dim=desc_dim).to(device)
    else:
        raise ValueError(f"Unknown decoder: {variant.decoder}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_np = val_idx.astype(np.int64)
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
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                pred_val = model.predict_matrix(x_input, desc[torch.tensor(val_idx_np, device=device)]).detach().cpu().numpy()
            X_pred_val = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
            X_pred_val[:, val_idx_np] = pred_val
            val_df = gene_metrics(X, X_pred_val, val_idx_np, genes)
            val_summary = summarize_gene_df(val_df)
            score = (
                np.nan_to_num(val_summary["SPCC"], nan=-1.0)
                - 0.05 * np.nan_to_num(val_summary["RMSE"], nan=10.0)
                - 0.05 * np.nan_to_num(val_summary["JS"], nan=10.0)
            )
            history.append({"step": step, "loss": float(loss.detach().cpu()), **{f"val_{k}": v for k, v in val_summary.items()}, "score": float(score)})
            if score > best_score:
                best_score = float(score)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred_test = model.predict_matrix(x_input, desc[torch.tensor(test_idx, device=device)]).detach().cpu().numpy()
    X_pred = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
    X_pred[:, test_idx] = pred_test
    test_df = gene_metrics(X, X_pred, test_idx, genes)
    low_df = gene_metrics(X, X_pred, low_test_idx, genes, subgroup="low_expression")
    high_df = gene_metrics(X, X_pred, high_spatial_idx, genes, subgroup="high_spatial")
    summary = {
        "model": variant.name,
        "decoder": variant.decoder,
        "descriptor": variant.descriptor,
        "control": variant.control,
        "selected_by": "val_genes",
        "best_val_score": best_score,
        **summarize_gene_df(test_df),
        **summarize_gene_df(low_df, prefix="low_expr_"),
        **summarize_gene_df(high_df, prefix="high_spatial_"),
    }
    hist_df = pd.DataFrame(history)
    return summary, hist_df, X_pred


def baseline_predictions(X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, sc_desc: np.ndarray | None = None) -> dict[str, np.ndarray]:
    preds = {}
    spot_mean = X[:, train_idx].mean(axis=1, keepdims=True)
    pred = np.zeros_like(X, dtype=np.float32)
    pred[:, test_idx] = np.repeat(spot_mean, len(test_idx), axis=1)
    preds["spot_mean_train_gene_default"] = pred
    gene_mean = X[:, train_idx].mean(axis=0)
    global_mean = float(np.mean(gene_mean))
    pred2 = np.zeros_like(X, dtype=np.float32)
    pred2[:, test_idx] = global_mean
    preds["global_train_gene_mean_default"] = pred2
    if sc_desc is not None:
        train_sc_mean = sc_desc[train_idx, 0]
        test_sc_mean = sc_desc[test_idx, 0]
        A = np.stack([train_sc_mean, np.ones_like(train_sc_mean)], axis=1)
        coef, *_ = np.linalg.lstsq(A, gene_mean, rcond=None)
        test_gene_mean = np.stack([test_sc_mean, np.ones_like(test_sc_mean)], axis=1) @ coef
        centered_spot = spot_mean - spot_mean.mean()
        scale = float(np.std(X[:, train_idx]) / max(np.std(centered_spot), 1e-6)) * 0.15
        pred3 = np.zeros_like(X, dtype=np.float32)
        pred3[:, test_idx] = test_gene_mean[None, :] + scale * centered_spot
        preds["scrna_gene_mean_plus_spot_factor"] = pred3
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Spatial_count.txt"))
    ap.add_argument("--scrna-counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/scRNA_count.txt"))
    ap.add_argument("--locations-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Locations.txt"))
    ap.add_argument("--mask-dir", type=Path, default=Path("/workspace/GeneSPT/results/imformation/strict_whole_gene_masks"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/GeneSPT/results/imformation"))
    ap.add_argument("--pca-dims", type=str, default="16,32,64,128")
    ap.add_argument("--nmf-dims", type=str, default="32")
    ap.add_argument("--run-variants", type=str, default="bilinear:pca32:correct,mlp:pca32:correct,lowrank:pca64:correct,bilinear:nmf32:correct,bilinear:pca32:shuffled,bilinear:pca32:random")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    desc_dir = args.out_dir / "strict_gene_descriptors"
    desc_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_counts, genes, _ = load_matrix(args.counts_path, index_col=None)
    X = log1p_cpm(X_counts)
    coords = pd.read_csv(args.locations_path, sep="\t").to_numpy(dtype=np.float32)
    train_idx = np.load(args.mask_dir / f"fold{args.fold}_train_gene_idx.npy")
    val_idx = np.load(args.mask_dir / f"fold{args.fold}_val_gene_idx.npy")
    test_idx = np.load(args.mask_dir / f"fold{args.fold}_test_gene_idx.npy")
    input_idx = train_idx.copy()
    low_test_idx, high_spatial_idx = subgroup_indices(X, test_idx, coords)

    X_sc_counts, sc_genes, _ = load_matrix(args.scrna_counts_path, index_col=0)
    if list(sc_genes) != list(genes):
        sc_map = {g: i for i, g in enumerate(sc_genes)}
        keep = [sc_map[g] for g in genes]
        X_sc_counts = X_sc_counts[:, keep]
        sc_genes = [sc_genes[i] for i in keep]
    X_sc = log1p_cpm(X_sc_counts)
    pca_dims = [int(x) for x in args.pca_dims.split(",") if x.strip()]
    nmf_dims = [int(x) for x in args.nmf_dims.split(",") if x.strip()]
    descriptors = build_descriptors(X_sc, pca_dims=pca_dims, nmf_dims=nmf_dims, seed=args.seed)
    sc_gene_mean = X_sc.mean(axis=0, keepdims=False).astype(np.float32)[:, None]
    descriptors["scrna_mean1"] = sc_gene_mean
    for name, arr in descriptors.items():
        np.savez_compressed(desc_dir / f"{name}.npz", descriptor=arr, genes=np.asarray(genes, dtype=object))

    audit_rows = []
    for name, arr in descriptors.items():
        audit_rows.append(
            {
                "descriptor": name,
                "shape": str(tuple(arr.shape)),
                "source": "scRNA_count.txt",
                "n_genes": arr.shape[0],
                "covers_all_test_genes": bool(arr.shape[0] == len(genes) and np.isfinite(arr[test_idx]).all()),
                "uses_test_ST_values": False,
                "control_alignment_available": name.startswith("pca") or name.startswith("nmf"),
            }
        )
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(desc_dir / "strict_gene_descriptor_inventory.csv", index=False)
    audit_md = [
        "# Strict Gene Descriptor Audit",
        "",
        f"- Dataset: Vis9A fold{args.fold}.",
        f"- scRNA source: `{args.scrna_counts_path}`.",
        f"- ST source: `{args.counts_path}`.",
        "- Preprocessing: per-cell library-size normalization to 1e4 followed by log1p.",
        "- Descriptors are computed from scRNA only and are available for train, val, and test genes.",
        "- No test ST expression values are used to construct descriptors.",
        "",
        audit_df.to_string(index=False),
    ]
    (args.out_dir / "strict_gene_descriptor_audit.md").write_text("\n".join(audit_md), encoding="utf-8")

    # Part 1 diagnostic baselines.
    diag_rows = [
        {
            "model": "legacy_fixed_output_model",
            "status": "not_run_current_arch_not_zero_shot",
            "reason": "Fixed output columns and SBD gene-index parameters receive no strict held-out gene ST gradients.",
            "SPCC": np.nan,
            "SSIM": np.nan,
            "RMSE": np.nan,
            "JS": np.nan,
        }
    ]
    for name, pred in baseline_predictions(X, train_idx, test_idx, sc_desc=descriptors["scrna_mean1"]).items():
        df = gene_metrics(X, pred, test_idx, genes)
        row = {"model": name, "status": "completed", "reason": "simple diagnostic baseline", **summarize_gene_df(df)}
        row.update(summarize_gene_df(gene_metrics(X, pred, low_test_idx, genes), prefix="low_expr_"))
        row.update(summarize_gene_df(gene_metrics(X, pred, high_spatial_idx, genes), prefix="high_spatial_"))
        diag_rows.append(row)
    pd.DataFrame(diag_rows).to_csv(args.out_dir / "strict_fixed_output_diagnostic_fold0.csv", index=False)

    variants = []
    for raw in [x for x in args.run_variants.split(",") if x.strip()]:
        dec, desc_name, control = raw.split(":")
        variants.append(Variant(name=f"{dec}_{desc_name}_{control}", decoder=dec, descriptor=desc_name, control=control))

    summaries = []
    histories = []
    gene_rows = []
    for variant in variants:
        if variant.descriptor not in descriptors:
            raise KeyError(f"Descriptor {variant.descriptor} not available. Available: {sorted(descriptors)}")
        desc_np = make_control_descriptor(descriptors[variant.descriptor], variant.control, seed=args.seed + 13)
        summary, hist, pred = train_variant(
            variant=variant,
            desc_np=desc_np,
            X=X,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            input_idx=input_idx,
            genes=genes,
            low_test_idx=low_test_idx,
            high_spatial_idx=high_spatial_idx,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            lr=args.lr,
            seed=args.seed + len(summaries) * 17,
        )
        summaries.append(summary)
        hist["model"] = variant.name
        histories.append(hist)
        gdf = gene_metrics(X, pred, test_idx, genes)
        gdf["model"] = variant.name
        gene_rows.append(gdf)

    long_df = pd.DataFrame(summaries)
    diag_df = pd.DataFrame(diag_rows)
    diag_for_merge = diag_df.assign(decoder="diagnostic", descriptor="none", control="none", selected_by="none", best_val_score=np.nan)
    all_long = pd.concat([diag_for_merge, long_df], ignore_index=True, sort=False)
    all_long.to_csv(args.out_dir / "gene_conditioned_decoder_fold0_long.csv", index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(args.out_dir / "gene_conditioned_decoder_fold0_training_history.csv", index=False)
    if gene_rows:
        pd.concat(gene_rows, ignore_index=True).to_csv(args.out_dir / "gene_conditioned_decoder_fold0_gene_metrics.csv", index=False)

    summary_cols = ["model", "decoder", "descriptor", "control", "SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "best_val_score"]
    summary_df = all_long[[c for c in summary_cols if c in all_long.columns]].sort_values("SPCC", ascending=False, na_position="last")
    summary_df.to_csv(args.out_dir / "gene_conditioned_decoder_fold0_summary.csv", index=False)

    correct = long_df[long_df["control"] == "correct"].copy()
    controls = long_df[long_df["control"] != "correct"].copy()
    best_correct = correct.sort_values("SPCC", ascending=False).head(1)
    best_control_spcc = float(np.nanmax(controls["SPCC"])) if len(controls) else np.nan
    best_diag_spcc = float(np.nanmax(diag_df["SPCC"])) if "SPCC" in diag_df else np.nan
    if len(best_correct):
        b = best_correct.iloc[0]
        pass_gate = (
            np.isfinite(b["SPCC"])
            and (not np.isfinite(best_diag_spcc) or float(b["SPCC"]) > best_diag_spcc)
            and (not np.isfinite(best_control_spcc) or float(b["SPCC"]) > best_control_spcc)
            and np.isfinite(b["RMSE"])
            and np.isfinite(b["JS"])
        )
        decision = "GENE_CONDITIONED_CONTINUE" if pass_gate else "GENE_CONDITIONED_FAILED"
    else:
        b = None
        decision = "GENE_CONDITIONED_FAILED"
    lines = [
        "# Gene-Conditioned Decoder Fold0 Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "## Best Correct Descriptor Model",
    ]
    if b is not None:
        lines.append(best_correct.to_string(index=False))
    else:
        lines.append("No correct descriptor model completed.")
    lines.extend(
        [
            "",
            "## Control Comparison",
            f"- Best diagnostic SPCC: {best_diag_spcc:.6f}" if np.isfinite(best_diag_spcc) else "- Best diagnostic SPCC: NA",
            f"- Best shuffled/random control SPCC: {best_control_spcc:.6f}" if np.isfinite(best_control_spcc) else "- Best shuffled/random control SPCC: NA",
            "",
            "## Notes",
            "- Current fixed-output GeneSPT is recorded as diagnostic-only and not run as a promoted strict model.",
            "- All trained variants use train genes for fitting, val genes for model selection, and test genes only for final reporting.",
        ]
    )
    (args.out_dir / "gene_conditioned_decoder_decision.md").write_text("\n".join(lines), encoding="utf-8")

    baseline_plan = """# Strict Whole-Gene Baseline Alignment Plan

## Frozen Protocol
Use `/workspace/GeneSPT/results/imformation/strict_whole_gene_masks/` for all methods.

## Rules
- Final inference input genes: `train_gene_idx + val_gene_idx`.
- Test genes: `test_gene_idx`, hidden from input and hyperparameter selection.
- Hyperparameters / checkpoints: selected on `val_gene_idx`.
- Evaluation: same test genes and same metrics for all methods.

## Methods To Align
- SpaIM
- Tangram
- SpaGE
- stPlus
- TransPA
- stDiff
- stImpute if feasible
- TISSUE if feasible

## Reporting
Report SPCC, SSIM, RMSE, JS, per-gene SPCC, high-spatial subgroup, and low-expression subgroup. Record whether each baseline uses scRNA/reference and whether held-out ST test-gene values are excluded from every fitting or selection path.
"""
    (args.out_dir / "strict_whole_gene_baseline_alignment_plan.md").write_text(baseline_plan, encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
