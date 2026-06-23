#!/usr/bin/env python3
"""Folds0-2 validation for selected strict gene-conditioned pointwise decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_gene_conditioned_mlp_controls_stabilization import (
    MLPVariant,
    diagnostic_predictions,
    make_descriptor_control,
    train_mlp_variant,
)
from run_strict_gene_conditioned_decoder_gate import (
    build_descriptors,
    gene_metrics,
    load_matrix,
    log1p_cpm,
    subgroup_indices,
    summarize_gene_df,
)


def summarize_prediction(name, X, X_pred, test_idx, low_test_idx, high_spatial_idx, genes, fold, kind):
    row = {
        "fold": int(fold),
        "model": name,
        "kind": kind,
        **summarize_gene_df(gene_metrics(X, X_pred, test_idx, genes)),
    }
    row.update(summarize_gene_df(gene_metrics(X, X_pred, low_test_idx, genes), prefix="low_expr_"))
    row.update(summarize_gene_df(gene_metrics(X, X_pred, high_spatial_idx, genes), prefix="high_spatial_"))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="0,1,2")
    ap.add_argument("--counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Spatial_count.txt"))
    ap.add_argument("--scrna-counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/scRNA_count.txt"))
    ap.add_argument("--locations-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Locations.txt"))
    ap.add_argument("--mask-dir", type=Path, default=Path("/workspace/GeneSPT/results/imformation/strict_whole_gene_masks"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/GeneSPT/results/imformation"))
    ap.add_argument("--output-prefix", type=str, default="gene_conditioned_decoder_folds012")
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
        keep = [sc_map[g] for g in genes]
        X_sc_counts = X_sc_counts[:, keep]
    X_sc = log1p_cpm(X_sc_counts)
    descriptors = build_descriptors(X_sc, pca_dims=[32], nmf_dims=[], seed=args.seed)
    desc_pca32 = descriptors["pca32"]

    fold_ids = [int(x) for x in args.folds.split(",") if x.strip()]
    rows = []
    histories = []
    for fold in fold_ids:
        train_idx = np.load(args.mask_dir / f"fold{fold}_train_gene_idx.npy")
        val_idx = np.load(args.mask_dir / f"fold{fold}_val_gene_idx.npy")
        test_idx = np.load(args.mask_dir / f"fold{fold}_test_gene_idx.npy")
        low_test_idx, high_spatial_idx = subgroup_indices(X, test_idx, coords)
        train_values = X[:, train_idx].reshape(-1)
        q001 = float(np.quantile(train_values, 0.001))
        q999 = float(np.quantile(train_values, 0.999))

        for name, pred in diagnostic_predictions(X, train_idx, test_idx).items():
            if name != "spot_mean_diagnostic":
                continue
            rows.append(summarize_prediction(name, X, pred, test_idx, low_test_idx, high_spatial_idx, genes, fold, "diagnostic"))

        variants = [
            MLPVariant("mlp_pca32_raw", "pca32", "correct", output_mode="linear"),
            MLPVariant("mlp_pca32_softplus", "pca32", "correct", output_mode="softplus"),
            MLPVariant("mlp_pca32_shuffled", "pca32", "shuffled", output_mode="linear"),
            MLPVariant("mlp_pca32_random", "pca32", "random", output_mode="linear"),
        ]
        for i, variant in enumerate(variants):
            desc_control, _ = make_descriptor_control(desc_pca32, variant.control, seed=args.seed + 1000 * fold + i)
            summary, hist, _, _ = train_mlp_variant(
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
                seed=args.seed + 1000 * fold + 17 * i,
            )
            summary.update({"fold": int(fold), "kind": "gene_conditioned"})
            rows.append(summary)
            hist["fold"] = int(fold)
            hist["model"] = variant.name
            histories.append(hist)

    long_df = pd.DataFrame(rows)
    prefix = str(args.output_prefix)
    long_df.to_csv(args.out_dir / f"{prefix}_long.csv", index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(args.out_dir / f"{prefix}_training_history.csv", index=False)
    metric_cols = ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC"]
    summary = (
        long_df.groupby("model", as_index=False)
        .agg(**{f"{m}_mean": (m, "mean") for m in metric_cols}, n_folds=("fold", "nunique"))
        .sort_values("SPCC_mean", ascending=False)
    )
    summary.to_csv(args.out_dir / f"{prefix}_summary.csv", index=False)

    soft = summary[summary["model"] == "mlp_pca32_softplus"]
    raw = summary[summary["model"] == "mlp_pca32_raw"]
    ctrl = summary[summary["model"].isin(["mlp_pca32_shuffled", "mlp_pca32_random"])]
    if len(soft) and len(raw) and len(ctrl):
        s = soft.iloc[0]
        r = raw.iloc[0]
        c = ctrl.sort_values("SPCC_mean", ascending=False).iloc[0]
        confirmed = (
            float(s["SPCC_mean"]) > float(c["SPCC_mean"])
            and float(s["RMSE_mean"]) < float(c["RMSE_mean"])
            and float(s["JS_mean"]) < float(r["JS_mean"])
        )
    else:
        confirmed = False
    decision = "GENE_CONDITIONED_POINTWISE_CONFIRMED" if confirmed else "GENE_CONDITIONED_STOP"
    (args.out_dir / f"{prefix}_decision.md").write_text(
        "\n".join(
            [
                "# Gene-Conditioned Decoder Folds0-2 Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "## Summary",
                summary.to_string(index=False),
                "",
                "Confirmation requires softplus MLP to beat shuffled/random controls on SPCC/RMSE and improve JS relative to raw MLP.",
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
