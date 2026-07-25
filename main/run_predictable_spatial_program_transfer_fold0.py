#!/usr/bin/env python3
"""Predictable spatial program transfer fold0 gate.

Follow-up to the ST spatial program decoder gate. It transfers only spatial
program components whose coefficients are predictable from scRNA gene
descriptors on val genes, then selects global/bin-specific lambda on val.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Ridge

from run_gene_conditioned_mlp_controls_stabilization import make_descriptor_control
from run_gc_spatiality_aware_training import compute_spatiality
from run_st_spatial_program_decoder_fold0 import (
    Basis,
    assemble_prediction,
    fit_predict_coeff,
    load_or_train_base,
    project_coeff,
    reconstruct,
    summarize_pred,
    val_score,
)
from run_st_spatial_program_decoder_fold0 import preprocess_train
from run_gc_spatial_residual_basis_fold0 import quick_gene_summary
from run_strict_gene_conditioned_decoder_gate import build_descriptors, load_matrix, log1p_cpm, make_knn_edges, subgroup_indices
from sklearn.decomposition import TruncatedSVD


INFO = Path("/workspace/GeneSPT/results/imformation")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def component_stats(C_pred: np.ndarray, C_oracle: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(C_oracle.shape[1]):
        y = C_oracle[:, k].astype(float)
        p = C_pred[:, k].astype(float)
        if np.std(y) < 1e-12 or np.std(p) < 1e-12:
            spcc = np.nan
            r2 = np.nan
        else:
            spcc = float(st.spearmanr(y, p, nan_policy="omit").correlation)
            ss = float(np.sum((y - p) ** 2))
            tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss / max(tot, 1e-12)
        rows.append(
            {
                "component": k,
                "component_spearman": spcc,
                "component_r2": r2,
                "component_mae": float(np.mean(np.abs(y - p))),
                "oracle_coeff_var": float(np.var(y)),
                "pred_coeff_var": float(np.var(p)),
            }
        )
    return pd.DataFrame(rows)


def selected_component_prediction(A: np.ndarray, C_hat: np.ndarray, meta: dict, keep: np.ndarray) -> np.ndarray:
    C = np.zeros_like(C_hat, dtype=np.float32)
    C[:, keep] = C_hat[:, keep]
    return reconstruct(A, C, meta)


def summarize_val_fast(model: str, X: np.ndarray, pred_sub: np.ndarray, idx: np.ndarray, low_idx: np.ndarray, high_idx: np.ndarray, extra: dict) -> dict:
    row = {"model": model, **extra}
    row.update(quick_gene_summary(X, pred_sub, idx, low_idx, high_idx))
    return row


def fit_svd_bases_only(X_train: np.ndarray, seed: int) -> list[Basis]:
    bases = []
    for preprocess in ["raw"]:
        Xp, meta = preprocess_train(X_train, preprocess)
        for k in [32, 64]:
            k_eff = min(k, X_train.shape[0] - 2, X_train.shape[1] - 2)
            svd = TruncatedSVD(n_components=k_eff, random_state=seed + k_eff)
            C_train = svd.fit_transform(Xp.T).astype(np.float32)
            A = svd.components_.T.astype(np.float32)
            bases.append(Basis(method="svd", preprocess=preprocess, k=k_eff, A=A, C_train=C_train, meta=meta))
    return bases


def fit_spatiality_predictor(D_train: np.ndarray, train_moran: np.ndarray, D_eval: np.ndarray) -> np.ndarray:
    finite_desc = np.isfinite(D_train).all(axis=1)
    finite_target = np.isfinite(train_moran)
    if int(finite_target.sum()) < 2:
        raise ValueError("Spatiality predictor requires at least two finite training-gene Moran's I values")
    target = np.asarray(train_moran, dtype=np.float32).copy()
    target[~finite_target] = float(np.nanmedian(target[finite_target]))
    model = Ridge(alpha=1.0)
    model.fit(D_train[finite_desc], target[finite_desc])
    return model.predict(D_eval).astype(np.float32)


def bin_lambdas_from_val(
    X: np.ndarray,
    base_val: np.ndarray,
    program_val: np.ndarray,
    val_idx: np.ndarray,
    low_val_idx: np.ndarray,
    high_val_idx: np.ndarray,
    genes: list[str],
    pred_spatiality_val: np.ndarray,
) -> tuple[dict[str, float], float, pd.DataFrame]:
    q1, q2 = np.quantile(pred_spatiality_val, [1 / 3, 2 / 3])
    bins = {
        "low": pred_spatiality_val <= q1,
        "mid": (pred_spatiality_val > q1) & (pred_spatiality_val <= q2),
        "high": pred_spatiality_val > q2,
    }
    grids = {"low": [0.0, 0.1, 0.25], "mid": [0.0, 0.1, 0.25, 0.5], "high": [0.25, 0.5, 0.75, 1.0]}
    best = None
    rows = []
    for lam_low in grids["low"]:
        for lam_mid in grids["mid"]:
            for lam_high in grids["high"]:
                pred = base_val.copy()
                for name, lam in [("low", lam_low), ("mid", lam_mid), ("high", lam_high)]:
                    pred[:, bins[name]] = (1.0 - lam) * base_val[:, bins[name]] + lam * program_val[:, bins[name]]
                row = summarize_val_fast(
                    f"val_bin_lambda_low{lam_low}_mid{lam_mid}_high{lam_high}",
                    X,
                    pred,
                    val_idx,
                    low_val_idx,
                    high_val_idx,
                    {"split": "val", "lambda_low": lam_low, "lambda_mid": lam_mid, "lambda_high": lam_high},
                )
                row["selection_score"] = val_score(row, summarize_val_fast("base_val", X, base_val, val_idx, low_val_idx, high_val_idx, {}))
                rows.append(row)
                if best is None or row["selection_score"] > best[0]:
                    best = (float(row["selection_score"]), {"low": lam_low, "mid": lam_mid, "high": lam_high})
    return best[1], best[0], pd.DataFrame(rows)


@dataclass
class Selected:
    method: str
    preprocess: str
    k: int
    descriptor: str
    predictor: str
    alpha: float
    topk: int
    lambda_mode: str
    lambda_value: float
    score: float
    component_keep: list[int]


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
    edges = make_knn_edges(coords, k=8)
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
    base_val_row = summarize_pred("gc_mlp_base_val", X, base["val"], val_idx, low_val_idx, high_val_idx, genes, {"split": "val", "role": "base"})
    base_test_row = summarize_pred("gc_mlp_base", X, base["test"], test_idx, low_test_idx, high_test_idx, genes, {"split": "test", "role": "base"})

    basis_keep = fit_svd_bases_only(X[:, train_idx], seed=args.seed)
    descriptor_names = ["pca32", "nmf32", "pca32_nmf32"]
    predictors = [("ridge", [1.0, 10.0, 100.0]), ("pls", [8, 16])]
    comp_rows = []
    val_rows = []
    best: Selected | None = None
    best_payload = None

    for b in basis_keep:
        X_val_proc, val_meta = __import__("run_st_spatial_program_decoder_fold0").preprocess_train(X[:, val_idx], b.preprocess)
        C_val_oracle = project_coeff(b.A, X_val_proc)
        for desc_name in descriptor_names:
            D_train = desc[desc_name][train_idx]
            D_val = desc[desc_name][val_idx]
            for pred_kind, alphas in predictors:
                for alpha in alphas:
                    try:
                        C_val_pred = fit_predict_coeff(pred_kind, D_train, b.C_train, D_val, float(alpha), seed=args.seed)
                    except Exception:
                        continue
                    stats_df = component_stats(C_val_pred, C_val_oracle)
                    stats_df.insert(0, "method", b.method)
                    stats_df.insert(1, "preprocess", b.preprocess)
                    stats_df.insert(2, "K", b.k)
                    stats_df.insert(3, "descriptor", desc_name)
                    stats_df.insert(4, "predictor", pred_kind)
                    stats_df.insert(5, "alpha", float(alpha))
                    comp_rows.append(stats_df)
                    # Predictability score: positive Spearman and non-trivial oracle variance.
                    tmp = stats_df.copy()
                    tmp["rank_score"] = tmp["component_spearman"].fillna(-1.0) * np.log1p(tmp["oracle_coeff_var"].clip(lower=0))
                    ranked = tmp.sort_values("rank_score", ascending=False)
                    for topk in [4, 8, 16, 32]:
                        keep = ranked.head(min(topk, b.k))["component"].to_numpy(dtype=np.int64)
                        program_val = selected_component_prediction(b.A, C_val_pred, val_meta, keep)
                        for lam in [0.0, 0.1, 0.25, 0.5, 0.75]:
                            pred = (1.0 - lam) * base["val"] + lam * program_val
                            row = summarize_val_fast(
                                f"{b.method}_{b.preprocess}_k{b.k}_{desc_name}_{pred_kind}_alpha{alpha}_top{topk}_lambda{lam:g}",
                                X,
                                pred,
                                val_idx,
                                low_val_idx,
                                high_val_idx,
                                {"split": "val", "method": b.method, "preprocess": b.preprocess, "K": b.k, "descriptor": desc_name, "predictor": pred_kind, "alpha": float(alpha), "topK_pred": topk, "lambda_mode": "global", "lambda": lam, "control": "correct"},
                            )
                            row["delta_SPCC_vs_base"] = row["SPCC"] - base_val_row["SPCC"]
                            row["delta_RMSE_vs_base"] = row["RMSE"] - base_val_row["RMSE"]
                            row["delta_JS_vs_base"] = row["JS"] - base_val_row["JS"]
                            row["delta_high_spatial_SPCC_vs_base"] = row["high_spatial_SPCC"] - base_val_row["high_spatial_SPCC"]
                            row["selection_score"] = val_score(row, base_val_row)
                            val_rows.append(row)
                            if best is None or row["selection_score"] > best.score:
                                best = Selected(b.method, b.preprocess, b.k, desc_name, pred_kind, float(alpha), topk, "global", float(lam), float(row["selection_score"]), keep.tolist())
                                best_payload = (b, C_val_pred, val_meta, keep, row)

    if comp_rows:
        comp_df = pd.concat(comp_rows, ignore_index=True)
    else:
        comp_df = pd.DataFrame()
    comp_df.to_csv(args.out_dir / "spatial_program_component_predictability_fold0.csv", index=False)
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(args.out_dir / "predictable_spatial_program_transfer_fold0_val.csv", index=False)
    if best is None:
        raise RuntimeError("No predictable spatial-program candidates completed.")

    # Component predictability report.
    comp_summary = (
        comp_df.groupby(["method", "preprocess", "K", "descriptor", "predictor", "alpha"], as_index=False)
        .agg(
            mean_positive_spearman=("component_spearman", lambda x: float(np.nanmean(np.maximum(x, 0)))),
            n_components_spearman_gt_02=("component_spearman", lambda x: int(np.sum(np.asarray(x, dtype=float) > 0.2))),
            max_component_spearman=("component_spearman", "max"),
        )
        .sort_values(["n_components_spearman_gt_02", "mean_positive_spearman"], ascending=False)
    )
    decision_predictable = "COMPONENTS_PREDICTABLE_CONTINUE" if int(comp_summary["n_components_spearman_gt_02"].max()) > 0 else "COMPONENTS_NOT_PREDICTABLE"
    (args.out_dir / "spatial_program_component_predictability_report.md").write_text(
        "\n".join(
            [
                "# Spatial Program Component Predictability Report",
                "",
                f"Decision: `{decision_predictable}`",
                "",
                "## Top Component Predictability Settings",
                comp_summary.head(20).to_string(index=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Bin-specific lambda for selected candidate.
    b_sel, _, val_meta_sel, keep_sel, _ = best_payload
    D_sel = desc[best.descriptor]
    C_val_sel = fit_predict_coeff(best.predictor, D_sel[train_idx], b_sel.C_train, D_sel[val_idx], best.alpha, seed=args.seed)
    program_val_sel = selected_component_prediction(b_sel.A, C_val_sel, val_meta_sel, np.asarray(best.component_keep, dtype=np.int64))
    train_sp = compute_spatiality(X, train_idx, edges)
    pred_sp_val = fit_spatiality_predictor(D_sel[train_idx], train_sp["MoranI"].to_numpy(dtype=np.float32), D_sel[val_idx])
    lambdas_bin, bin_score, bin_val_df = bin_lambdas_from_val(X, base["val"], program_val_sel, val_idx, low_val_idx, high_val_idx, genes, pred_sp_val)
    bin_val_df.to_csv(args.out_dir / "predictable_spatial_program_transfer_fold0_bin_lambda_val.csv", index=False)
    if bin_score > best.score:
        best.lambda_mode = "predicted_spatiality_bins"
        best.lambda_value = np.nan
        best.score = bin_score

    def make_program_test(D_all: np.ndarray, basis_override: Basis | None = None, keep_override: np.ndarray | None = None, mean_coeff: bool = False) -> np.ndarray:
        b = b_sel if basis_override is None else basis_override
        keep = np.asarray(best.component_keep if keep_override is None else keep_override, dtype=np.int64)
        if mean_coeff:
            C_hat = np.repeat(b.C_train.mean(axis=0, keepdims=True), len(test_idx), axis=0).astype(np.float32)
        else:
            C_hat = fit_predict_coeff(best.predictor, D_all[train_idx], b.C_train, D_all[test_idx], best.alpha, seed=args.seed + 99)
        # Raw bases are used for actual test prediction; no test-gene true stats required.
        return selected_component_prediction(b.A, C_hat, b.meta, keep)

    program_test = make_program_test(D_sel)
    if best.lambda_mode == "predicted_spatiality_bins":
        pred_sp_test = fit_spatiality_predictor(D_sel[train_idx], train_sp["MoranI"].to_numpy(dtype=np.float32), D_sel[test_idx])
        q1, q2 = np.quantile(pred_sp_val, [1 / 3, 2 / 3])
        bins_test = {"low": pred_sp_test <= q1, "mid": (pred_sp_test > q1) & (pred_sp_test <= q2), "high": pred_sp_test > q2}
        final_test = base["test"].copy()
        for name, lam in lambdas_bin.items():
            final_test[:, bins_test[name]] = (1.0 - lam) * base["test"][:, bins_test[name]] + lam * program_test[:, bins_test[name]]
        selected_lambda_desc = json.dumps(lambdas_bin)
    else:
        final_test = (1.0 - best.lambda_value) * base["test"] + best.lambda_value * program_test
        selected_lambda_desc = str(best.lambda_value)

    test_rows = [base_test_row]
    test_rows.append(
        summarize_pred(
            "predictable_spatial_program_selected_correct",
            X,
            final_test,
            test_idx,
            low_test_idx,
            high_test_idx,
            genes,
            {"split": "test", "role": "selected", **best.__dict__, "control": "correct", "lambda_selected": selected_lambda_desc},
        )
    )
    controls = [
        ("shuffled_descriptor", make_descriptor_control(D_sel, "shuffled", seed=args.seed + 1)[0], None, False),
        ("random_descriptor", make_descriptor_control(D_sel, "random", seed=args.seed + 2)[0], None, False),
        ("permuted_labels", make_descriptor_control(D_sel, "permuted_labels", seed=args.seed + 3)[0], None, False),
    ]
    rng = np.random.default_rng(args.seed + 444)
    A_rand = rng.normal(0, float(np.std(b_sel.A) + 1e-6), size=b_sel.A.shape).astype(np.float32)
    rand_basis = Basis("random_basis", b_sel.preprocess, b_sel.k, A_rand, project_coeff(A_rand, X[:, train_idx]), b_sel.meta)
    A_perm = b_sel.A[rng.permutation(b_sel.A.shape[0])].astype(np.float32)
    perm_basis = Basis("spot_permuted_basis", b_sel.preprocess, b_sel.k, A_perm, project_coeff(A_perm, X[:, train_idx]), b_sel.meta)
    controls.extend(
        [
            ("random_basis", D_sel, rand_basis, False),
            ("spot_permuted_basis", D_sel, perm_basis, False),
            ("mean_coeff", D_sel, None, True),
        ]
    )
    for name, D_ctrl, b_ctrl, mean_coeff in controls:
        prog = make_program_test(D_ctrl, basis_override=b_ctrl, mean_coeff=mean_coeff)
        pred = (1.0 - (0.0 if best.lambda_mode == "predicted_spatiality_bins" else best.lambda_value)) * base["test"] + (0.0 if best.lambda_mode == "predicted_spatiality_bins" else best.lambda_value) * prog
        if best.lambda_mode == "predicted_spatiality_bins" and name != "mean_coeff":
            # For controls, use the same bin lambdas and test bins from correct descriptors.
            pred = base["test"].copy()
            for bin_name, lam in lambdas_bin.items():
                pred[:, bins_test[bin_name]] = (1.0 - lam) * base["test"][:, bins_test[bin_name]] + lam * prog[:, bins_test[bin_name]]
        test_rows.append(
            summarize_pred(
                f"predictable_spatial_program_{name}_control",
                X,
                pred,
                test_idx,
                low_test_idx,
                high_test_idx,
                genes,
                {"split": "test", "role": "control", **best.__dict__, "control": name, "lambda_selected": selected_lambda_desc},
            )
        )

    test_df = pd.DataFrame(test_rows)
    base_row = test_df[test_df["role"].eq("base")].iloc[0]
    for m in ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "high_spatial_RMSE"]:
        test_df[f"delta_{m}_vs_base"] = test_df[m].astype(float) - float(base_row[m])
    test_df.to_csv(args.out_dir / "predictable_spatial_program_transfer_fold0_test.csv", index=False)
    selected = test_df[test_df["role"].eq("selected")].iloc[0]
    ctrl = test_df[test_df["role"].eq("control")]
    controls_ok = bool(
        selected["SPCC"] > ctrl["SPCC"].max()
        and selected["RMSE"] <= ctrl["RMSE"].min() + 1e-12
        and selected["JS"] <= ctrl["JS"].min() + 1e-12
    )
    gate = bool(
        (
            selected["delta_high_spatial_SPCC_vs_base"] >= 0.01
            or selected["delta_SPCC_vs_base"] >= 0.002
            or (selected["delta_RMSE_vs_base"] <= -0.003 and selected["delta_SPCC_vs_base"] >= -0.001)
            or (selected["delta_JS_vs_base"] <= -0.003 and selected["delta_SPCC_vs_base"] >= -0.001)
        )
        and selected["delta_JS_vs_base"] <= 0.003
        and selected["delta_SSIM_vs_base"] >= -0.002
        and controls_ok
    )
    aux = bool(not gate and controls_ok and selected["delta_SPCC_vs_base"] > 0 and selected["delta_RMSE_vs_base"] < 0 and selected["delta_JS_vs_base"] < 0)
    decision = "PREDICTABLE_SPATIAL_PROGRAM_CONTINUE" if gate else ("PREDICTABLE_SPATIAL_PROGRAM_AUXILIARY" if aux else "PREDICTABLE_SPATIAL_PROGRAM_FAILED")
    (args.out_dir / "predictable_spatial_program_transfer_decision.md").write_text(
        "\n".join(
            [
                "# Predictable Spatial Program Transfer Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "## Component Predictability",
                f"Component predictability decision: `{decision_predictable}`",
                comp_summary.head(10).to_string(index=False),
                "",
                "## Selected Candidate",
                json.dumps(best.__dict__, indent=2),
                "",
                "## Test Summary",
                test_df[["model", "SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "delta_SPCC_vs_base", "delta_SSIM_vs_base", "delta_RMSE_vs_base", "delta_JS_vs_base", "delta_high_spatial_SPCC_vs_base"]].to_string(index=False),
                "",
                f"Controls OK: `{controls_ok}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(test_df[["model", "SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC", "delta_SPCC_vs_base", "delta_RMSE_vs_base", "delta_JS_vs_base"]].to_string(index=False))
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
