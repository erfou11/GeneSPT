#!/usr/bin/env python3
"""Gene-descriptor-guided ST spatial program decoder fold0 gate.

Spatial programs are learned from train-gene ST maps only. Val genes are used
for oracle upper-bound and hyperparameter selection. Test genes are evaluated
only once for the selected descriptor-to-coefficient model and controls.
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
from sklearn.decomposition import MiniBatchNMF, TruncatedSVD
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.cross_decomposition import PLSRegression
from torch import nn

from run_gene_conditioned_mlp_controls_stabilization import make_descriptor_control
from run_gc_spatial_residual_basis_fold0 import quick_gene_summary, train_canonical_base
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


def assemble_prediction(shape: tuple[int, int], idx: np.ndarray, pred_sub: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    out[:, idx] = pred_sub.astype(np.float32)
    return out


def summarize_pred(model: str, X: np.ndarray, pred_sub: np.ndarray, idx: np.ndarray, low_idx: np.ndarray, high_idx: np.ndarray, genes: list[str], extra: dict) -> dict:
    X_pred = assemble_prediction(X.shape, idx, pred_sub)
    row = {"model": model, **extra}
    row.update(summarize_gene_df(gene_metrics(X, X_pred, idx, genes)))
    row.update(summarize_gene_df(gene_metrics(X, X_pred, low_idx, genes), prefix="low_expr_"))
    row.update(summarize_gene_df(gene_metrics(X, X_pred, high_idx, genes), prefix="high_spatial_"))
    return row


def load_or_train_base(
    X: np.ndarray,
    desc_pca32: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    args,
) -> dict[str, np.ndarray]:
    cache = INFO / "gc_residual_maps" / f"fold{args.fold}_canonical_gc_mlp_residual_maps.npz"
    if cache.exists():
        c = np.load(cache)
        return {"train": c["pred_train"], "val": c["pred_val"], "test": c["pred_test"]}
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
        seed=args.seed + 1701 * args.fold,
    )
    hist.to_csv(args.out_dir / "st_spatial_program_base_training_history.csv", index=False)
    return preds


@dataclass
class Basis:
    method: str
    preprocess: str
    k: int
    A: np.ndarray
    C_train: np.ndarray
    meta: dict


def preprocess_train(X_train: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    X_train = np.asarray(X_train, dtype=np.float32)
    if mode == "raw":
        return X_train.copy(), {"mode": mode}
    if mode == "centered":
        mu = X_train.mean(axis=0, keepdims=True)
        return (X_train - mu).astype(np.float32), {"mode": mode, "mean": mu.astype(np.float32)}
    if mode == "standardized":
        mu = X_train.mean(axis=0, keepdims=True)
        sd = np.clip(X_train.std(axis=0, keepdims=True), 1e-6, None)
        return ((X_train - mu) / sd).astype(np.float32), {"mode": mode, "mean": mu.astype(np.float32), "std": sd.astype(np.float32)}
    if mode == "shifted_nonnegative":
        mn = X_train.min(axis=0, keepdims=True)
        return (X_train - mn).astype(np.float32), {"mode": mode, "min": mn.astype(np.float32)}
    raise ValueError(mode)


def apply_preprocess(X_sub: np.ndarray, meta: dict) -> np.ndarray:
    mode = meta["mode"]
    if mode == "raw":
        return X_sub.astype(np.float32)
    if mode == "centered":
        return (X_sub - meta["mean"]).astype(np.float32)
    if mode == "standardized":
        return ((X_sub - meta["mean"]) / meta["std"]).astype(np.float32)
    if mode == "shifted_nonnegative":
        return np.clip(X_sub - meta["min"], 0.0, None).astype(np.float32)
    raise ValueError(mode)


def invert_preprocess(X_proc: np.ndarray, meta: dict) -> np.ndarray:
    mode = meta["mode"]
    if mode == "raw":
        return X_proc.astype(np.float32)
    if mode == "centered":
        return (X_proc + meta["mean"]).astype(np.float32)
    if mode == "standardized":
        return (X_proc * meta["std"] + meta["mean"]).astype(np.float32)
    if mode == "shifted_nonnegative":
        return (X_proc + meta["min"]).astype(np.float32)
    raise ValueError(mode)


def orth_basis(A: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    return q.astype(np.float32)


def fit_bases(X_train: np.ndarray, max_k: int, seed: int) -> list[Basis]:
    bases: list[Basis] = []
    for preprocess in ["raw", "centered", "standardized"]:
        Xp, meta = preprocess_train(X_train, preprocess)
        for method in ["svd", "nmf"]:
            # ST log1p(CPM) maps are already nonnegative, so raw NMF is
            # directly usable for held-out genes without needing test-gene
            # mean/std/min statistics.
            if method == "nmf" and preprocess != "raw":
                continue
            for k in [8, 16, 32, 64]:
                k_eff = min(k, max_k, X_train.shape[0] - 2, X_train.shape[1] - 2)
                if k_eff < 2:
                    continue
                try:
                    if method == "svd":
                        svd = TruncatedSVD(n_components=k_eff, random_state=seed)
                        C_train = svd.fit_transform(Xp.T).astype(np.float32)
                        A = svd.components_.T.astype(np.float32)
                    elif method == "nmf":
                        nmf = MiniBatchNMF(n_components=k_eff, random_state=seed + k_eff, max_iter=300, batch_size=512, init="nndsvda")
                        C_train = nmf.fit_transform(np.clip(Xp.T, 0.0, None)).astype(np.float32)
                        A = nmf.components_.T.astype(np.float32)
                    bases.append(Basis(method=method, preprocess=preprocess, k=k_eff, A=A, C_train=C_train, meta=meta))
                except Exception:
                    continue
    return bases


def project_coeff(A: np.ndarray, X_sub_proc: np.ndarray) -> np.ndarray:
    return (np.linalg.pinv(A) @ X_sub_proc).T.astype(np.float32)


def reconstruct(A: np.ndarray, coeff: np.ndarray, meta: dict) -> np.ndarray:
    proc = A @ coeff.T
    return np.clip(invert_preprocess(proc, meta), 0.0, None).astype(np.float32)


def val_score(row: dict, base: dict) -> float:
    return float(
        np.nan_to_num(row["SPCC"], nan=-1.0)
        + 0.6 * np.nan_to_num(row["high_spatial_SPCC"], nan=-1.0)
        - 0.04 * np.nan_to_num(row["RMSE"], nan=10.0)
        - 0.04 * np.nan_to_num(row["JS"], nan=10.0, posinf=10.0)
        - 0.2 * max(0.0, float(base["SSIM"] - row["SSIM"]))
    )


def summarize_val_fast(model: str, X: np.ndarray, pred_sub: np.ndarray, idx: np.ndarray, low_idx: np.ndarray, high_idx: np.ndarray, extra: dict) -> dict:
    row = {"model": model, **extra}
    row.update(quick_gene_summary(X, pred_sub, idx, low_idx, high_idx))
    return row


def fit_predict_coeff(kind: str, D_train: np.ndarray, C_train: np.ndarray, D_eval: np.ndarray, alpha: float, seed: int) -> np.ndarray:
    if kind == "ridge":
        model = Ridge(alpha=float(alpha))
        model.fit(D_train, C_train)
        return model.predict(D_eval).astype(np.float32)
    if kind == "elasticnet":
        outs = []
        for j in range(C_train.shape[1]):
            m = ElasticNet(alpha=float(alpha), l1_ratio=0.2, max_iter=2000, random_state=seed + j)
            m.fit(D_train, C_train[:, j])
            outs.append(m.predict(D_eval))
        return np.vstack(outs).T.astype(np.float32)
    if kind == "pls":
        n_comp = min(int(alpha), D_train.shape[1], C_train.shape[1], D_train.shape[0] - 1)
        model = PLSRegression(n_components=max(1, n_comp))
        model.fit(D_train, C_train)
        return model.predict(D_eval).astype(np.float32)
    raise ValueError(kind)


class CoeffMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.net(x)


def fit_predict_mlp(D_train: np.ndarray, C_train: np.ndarray, D_eval: np.ndarray, seed: int) -> np.ndarray:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(D_train.astype(np.float32), device=device)
    y = torch.tensor(C_train.astype(np.float32), device=device)
    xe = torch.tensor(D_eval.astype(np.float32), device=device)
    model = CoeffMLP(x.shape[1], y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(800):
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        return model(xe).detach().cpu().numpy().astype(np.float32)


@dataclass
class Selected:
    method: str
    preprocess: str
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
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
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

    train_idx = np.load(args.mask_dir / f"fold{args.fold}_train_gene_idx.npy")
    val_idx = np.load(args.mask_dir / f"fold{args.fold}_val_gene_idx.npy")
    test_idx = np.load(args.mask_dir / f"fold{args.fold}_test_gene_idx.npy")
    low_val_idx, high_val_idx = subgroup_indices(X, val_idx, coords)
    low_test_idx, high_test_idx = subgroup_indices(X, test_idx, coords)
    base = load_or_train_base(X, desc["pca32"], train_idx, val_idx, test_idx, device, args)
    base_val = summarize_pred("gc_mlp_base_val", X, base["val"], val_idx, low_val_idx, high_val_idx, genes, {"split": "val", "role": "base"})
    base_test = summarize_pred("gc_mlp_base", X, base["test"], test_idx, low_test_idx, high_test_idx, genes, {"split": "test", "role": "base"})

    bases = fit_bases(X[:, train_idx], max_k=128, seed=args.seed)
    basis_rows = []
    oracle_rows = [base_val]
    best_oracle = None
    best_oracle_row = None
    for b in bases:
        # Oracle may use val-gene maps, but preprocessing must be computed per
        # val gene rather than reusing train-gene centering vectors. This keeps
        # the upper-bound diagnostic well-defined without leaking into test.
        X_val_proc, val_meta = preprocess_train(X[:, val_idx], b.preprocess)
        C_val = project_coeff(b.A, X_val_proc)
        oracle = reconstruct(b.A, C_val, val_meta)
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pred = (1.0 - lam) * base["val"] + lam * oracle
            row = summarize_val_fast(
                f"oracle_{b.method}_{b.preprocess}_k{b.k}_lambda{lam:g}",
                X,
                pred,
                val_idx,
                low_val_idx,
                high_val_idx,
                {"split": "val", "role": "oracle", "method": b.method, "preprocess": b.preprocess, "K": b.k, "lambda": lam},
            )
            row["delta_high_spatial_SPCC_vs_base"] = row["high_spatial_SPCC"] - base_val["high_spatial_SPCC"]
            row["delta_RMSE_vs_base"] = row["RMSE"] - base_val["RMSE"]
            row["delta_JS_vs_base"] = row["JS"] - base_val["JS"]
            row["selection_score"] = val_score(row, base_val)
            oracle_rows.append(row)
            if best_oracle is None or row["selection_score"] > best_oracle.score:
                best_oracle = Selected(b.method, b.preprocess, b.k, "oracle", "oracle", np.nan, lam, float(row["selection_score"]))
                best_oracle_row = row
        recon_train = reconstruct(b.A, b.C_train, b.meta)
        basis_rows.append(
            {
                "method": b.method,
                "preprocess": b.preprocess,
                "K": b.k,
                "train_recon_mse": float(np.mean((X[:, train_idx] - recon_train) ** 2)),
                "A_shape": str(tuple(b.A.shape)),
                "C_train_shape": str(tuple(b.C_train.shape)),
            }
        )
    pd.DataFrame(basis_rows).to_csv(args.out_dir / "st_spatial_program_basis_audit.csv", index=False)
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(args.out_dir / "st_spatial_program_oracle_val.csv", index=False)
    oracle_gain = bool(
        best_oracle_row["delta_high_spatial_SPCC_vs_base"] >= 0.01
        or best_oracle_row["delta_RMSE_vs_base"] <= -0.005
        or best_oracle_row["delta_JS_vs_base"] <= -0.003
    )
    (args.out_dir / "st_spatial_program_oracle_decision.md").write_text(
        "\n".join(
            [
                "# ST Spatial Program Oracle Decision",
                "",
                f"Decision: `{'ORACLE_SPATIAL_PROGRAM_HAS_SIGNAL' if oracle_gain else 'ORACLE_SPATIAL_PROGRAM_FAILED'}`",
                "",
                "## Best Oracle",
                pd.Series(best_oracle_row).to_string(),
                "",
                "Stop if oracle spatial-program reconstruction cannot improve high_spatial_SPCC/RMSE/JS on val genes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not oracle_gain:
        print("Oracle upper-bound failed; stopping before descriptor-to-coefficient predictor.")
        return

    # Descriptor-to-coefficient predictor uses only train genes; selection on val genes.
    basis_lookup = {(b.method, b.preprocess, b.k): b for b in bases}
    rows = []
    best: Selected | None = None
    best_row = None
    descriptor_names = ["pca32", "nmf32", "pca32_nmf32"]
    predictors = [("ridge", [0.01, 0.1, 1.0, 10.0, 100.0]), ("elasticnet", [0.001, 0.01, 0.1]), ("pls", [4, 8, 16])]
    # Actual descriptor-to-coefficient prediction must not require test-gene
    # true mean/std/min for inverse transforms. Use raw-space bases only.
    candidate_bases = sorted(
        oracle_df[oracle_df["role"].eq("oracle") & oracle_df["preprocess"].eq("raw")]
        .sort_values("selection_score", ascending=False)[["method", "preprocess", "K"]]
        .drop_duplicates()
        .head(6)
        .itertuples(index=False, name=None)
    )
    for method, preprocess, k in candidate_bases:
        b = basis_lookup[(method, preprocess, int(k))]
        for desc_name in descriptor_names:
            D_train = desc[desc_name][train_idx]
            D_val = desc[desc_name][val_idx]
            for pred_kind, alphas in predictors:
                for alpha in alphas:
                    try:
                        C_val_hat = fit_predict_coeff(pred_kind, D_train, b.C_train, D_val, float(alpha), seed=args.seed)
                    except Exception:
                        continue
                    program_val = reconstruct(b.A, C_val_hat, b.meta)
                    for lam in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
                        pred = (1.0 - lam) * base["val"] + lam * program_val
                        row = summarize_val_fast(
                            f"{method}_{preprocess}_k{k}_{desc_name}_{pred_kind}_alpha{alpha}_lambda{lam:g}",
                            X,
                            pred,
                            val_idx,
                            low_val_idx,
                            high_val_idx,
                            {"split": "val", "role": "candidate", "method": method, "preprocess": preprocess, "K": int(k), "descriptor": desc_name, "predictor": pred_kind, "alpha": float(alpha), "lambda": lam, "control": "correct"},
                        )
                        row["delta_SPCC_vs_base"] = row["SPCC"] - base_val["SPCC"]
                        row["delta_high_spatial_SPCC_vs_base"] = row["high_spatial_SPCC"] - base_val["high_spatial_SPCC"]
                        row["delta_RMSE_vs_base"] = row["RMSE"] - base_val["RMSE"]
                        row["delta_JS_vs_base"] = row["JS"] - base_val["JS"]
                        row["selection_score"] = val_score(row, base_val)
                        rows.append(row)
                        if best is None or row["selection_score"] > best.score:
                            best = Selected(method, preprocess, int(k), desc_name, pred_kind, float(alpha), float(lam), float(row["selection_score"]))
                            best_row = row
    if best is None:
        raise RuntimeError("No descriptor-to-coefficient candidate completed.")

    # Optional MLP coefficient predictor for selected basis/descriptor.
    b_sel = basis_lookup[(best.method, best.preprocess, best.k)]
    try:
        C_val_mlp = fit_predict_mlp(desc[best.descriptor][train_idx], b_sel.C_train, desc[best.descriptor][val_idx], seed=args.seed + 700)
        program_val = reconstruct(b_sel.A, C_val_mlp, b_sel.meta)
        pred = (1.0 - best.lam) * base["val"] + best.lam * program_val
        row = summarize_val_fast(
            f"{best.method}_{best.preprocess}_k{best.k}_{best.descriptor}_mlp_lambda{best.lam:g}",
            X,
            pred,
            val_idx,
            low_val_idx,
            high_val_idx,
            {"split": "val", "role": "candidate", "method": best.method, "preprocess": best.preprocess, "K": best.k, "descriptor": best.descriptor, "predictor": "mlp", "alpha": np.nan, "lambda": best.lam, "control": "correct"},
        )
        row["delta_SPCC_vs_base"] = row["SPCC"] - base_val["SPCC"]
        row["delta_high_spatial_SPCC_vs_base"] = row["high_spatial_SPCC"] - base_val["high_spatial_SPCC"]
        row["delta_RMSE_vs_base"] = row["RMSE"] - base_val["RMSE"]
        row["delta_JS_vs_base"] = row["JS"] - base_val["JS"]
        row["selection_score"] = val_score(row, base_val)
        rows.append(row)
        if row["selection_score"] > best.score:
            best = Selected(best.method, best.preprocess, best.k, best.descriptor, "mlp", np.nan, best.lam, float(row["selection_score"]))
            best_row = row
    except Exception:
        pass

    val_long = pd.DataFrame(rows)
    val_long.to_csv(args.out_dir / "st_spatial_program_descriptor_gate_fold0_long.csv", index=False)

    def selected_test(D_all: np.ndarray, basis_override: Basis | None = None, control: str = "correct") -> np.ndarray:
        b = b_sel if basis_override is None else basis_override
        if control == "mean_coeff":
            C_hat = np.repeat(b.C_train.mean(axis=0, keepdims=True), len(test_idx), axis=0).astype(np.float32)
        elif best.predictor == "mlp":
            C_hat = fit_predict_mlp(D_all[train_idx], b.C_train, D_all[test_idx], seed=args.seed + 900)
        else:
            C_hat = fit_predict_coeff(best.predictor, D_all[train_idx], b.C_train, D_all[test_idx], best.alpha, seed=args.seed + 900)
        program = reconstruct(b.A, C_hat, b.meta)
        return (1.0 - best.lam) * base["test"] + best.lam * program

    test_rows = [base_test]
    D_correct = desc[best.descriptor].astype(np.float32)
    pred_correct = selected_test(D_correct)
    test_rows.append(
        summarize_pred(
            "st_spatial_program_selected_correct",
            X,
            pred_correct,
            test_idx,
            low_test_idx,
            high_test_idx,
            genes,
            {"split": "test", "role": "selected", "method": best.method, "preprocess": best.preprocess, "K": best.k, "descriptor": best.descriptor, "predictor": best.predictor, "alpha": best.alpha, "lambda": best.lam, "control": "correct"},
        )
    )
    for control in ["shuffled", "random", "permuted_labels"]:
        D_ctrl = make_descriptor_control(D_correct, control, seed=args.seed + len(control))[0]
        test_rows.append(
            summarize_pred(
                f"st_spatial_program_{control}_descriptor_control",
                X,
                selected_test(D_ctrl),
                test_idx,
                low_test_idx,
                high_test_idx,
                genes,
                {"split": "test", "role": "control", "method": best.method, "preprocess": best.preprocess, "K": best.k, "descriptor": best.descriptor, "predictor": best.predictor, "alpha": best.alpha, "lambda": best.lam, "control": control},
            )
        )
    rng = np.random.default_rng(args.seed + 123)
    A_rand = rng.normal(0, float(np.std(b_sel.A) + 1e-6), size=b_sel.A.shape).astype(np.float32)
    rand_basis = Basis("random_basis", best.preprocess, best.k, A_rand, project_coeff(A_rand, apply_preprocess(X[:, train_idx], b_sel.meta)), b_sel.meta)
    A_perm = b_sel.A[rng.permutation(b_sel.A.shape[0])].astype(np.float32)
    perm_basis = Basis("spot_permuted_basis", best.preprocess, best.k, A_perm, project_coeff(A_perm, apply_preprocess(X[:, train_idx], b_sel.meta)), b_sel.meta)
    for name, b_ctrl in [("random_spatial_basis", rand_basis), ("spot_permuted_spatial_program", perm_basis)]:
        test_rows.append(
            summarize_pred(
                f"st_spatial_program_{name}_control",
                X,
                selected_test(D_correct, basis_override=b_ctrl),
                test_idx,
                low_test_idx,
                high_test_idx,
                genes,
                {"split": "test", "role": "control", "method": name, "preprocess": best.preprocess, "K": best.k, "descriptor": best.descriptor, "predictor": best.predictor, "alpha": best.alpha, "lambda": best.lam, "control": name},
            )
        )
    test_rows.append(
        summarize_pred(
            "st_spatial_program_mean_coefficient_baseline",
            X,
            selected_test(D_correct, control="mean_coeff"),
            test_idx,
            low_test_idx,
            high_test_idx,
            genes,
            {"split": "test", "role": "control", "method": best.method, "preprocess": best.preprocess, "K": best.k, "descriptor": best.descriptor, "predictor": "mean_coeff", "alpha": np.nan, "lambda": best.lam, "control": "mean_coeff"},
        )
    )
    test_df = pd.DataFrame(test_rows)
    base_row = test_df[test_df["role"].eq("base")].iloc[0]
    for m in ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]:
        test_df[f"delta_{m}_vs_base"] = test_df[m].astype(float) - float(base_row[m])
    test_df.to_csv(args.out_dir / "st_spatial_program_descriptor_gate_fold0_summary.csv", index=False)
    selected = test_df[test_df["role"].eq("selected")].iloc[0]
    controls = test_df[test_df["role"].eq("control")]
    controls_ok = bool(
        selected["SPCC"] > controls["SPCC"].max()
        and selected["RMSE"] <= controls["RMSE"].min() + 1e-12
        and selected["JS"] <= controls["JS"].min() + 1e-12
    )
    gate = bool(
        (
            selected["delta_high_spatial_SPCC_vs_base"] >= 0.01
            or selected["delta_SPCC_vs_base"] >= 0.002
            or (selected["delta_RMSE_vs_base"] <= -0.003 and selected["delta_SPCC_vs_base"] >= -0.001)
            or (selected["delta_JS_vs_base"] <= -0.003 and selected["delta_SPCC_vs_base"] >= -0.001)
        )
        and selected["delta_JS_vs_base"] <= 0.003
        and controls_ok
    )
    decision = "ST_SPATIAL_PROGRAM_CONTINUE" if gate else "ST_SPATIAL_PROGRAM_FAILED"
    lines = [
        "# ST Spatial Program Descriptor Gate Fold0 Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "## Mask Safety",
        "- Spatial programs are learned from train-gene ST maps only.",
        "- Descriptor-to-coefficient predictor is trained on train genes only.",
        "- Val genes select basis/preprocess/K/predictor/lambda.",
        "- Test genes are used only for final evaluation.",
        "",
        "## Best Oracle",
        pd.Series(best_oracle_row).to_string(),
        "",
        "## Selected Candidate",
        json.dumps(best.__dict__, indent=2),
        "",
        "## Test Summary",
        test_df[["model", "SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "delta_SPCC_vs_base", "delta_RMSE_vs_base", "delta_JS_vs_base", "delta_high_spatial_SPCC_vs_base"]].to_string(index=False),
        "",
        f"Controls OK: `{controls_ok}`",
    ]
    (args.out_dir / "st_spatial_program_descriptor_gate_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(test_df[["model", "SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "delta_SPCC_vs_base", "delta_RMSE_vs_base", "delta_JS_vs_base"]].to_string(index=False))
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
