#!/usr/bin/env python3
"""Five-fold architecture-matched descriptor controls for strict GC-MLP.

All variants use the same gene-conditioned MLP decoder and softplus output.
Only the scRNA PCA32 descriptor assignment changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_gene_conditioned_mlp_controls_stabilization import (
    MLPVariant,
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


def summarize_prediction(name, X, X_pred, test_idx, low_test_idx, high_spatial_idx, genes, fold, variant):
    row = {
        "fold": int(fold),
        "model": name,
        "descriptor": variant.descriptor,
        "control": variant.control,
        "output_mode": variant.output_mode,
        **summarize_gene_df(gene_metrics(X, X_pred, test_idx, genes)),
    }
    row.update(summarize_gene_df(gene_metrics(X, X_pred, low_test_idx, genes), prefix="low_expr_"))
    row.update(summarize_gene_df(gene_metrics(X, X_pred, high_spatial_idx, genes), prefix="high_spatial_"))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_counts, genes, _ = load_matrix(args.counts_path, index_col=None)
    X = log1p_cpm(X_counts)
    coords = pd.read_csv(args.locations_path, sep="\t").to_numpy(dtype=np.float32)

    X_sc_counts, sc_genes, _ = load_matrix(args.scrna_counts_path, index_col=0)
    if list(sc_genes) != list(genes):
        sc_map = {g: i for i, g in enumerate(sc_genes)}
        X_sc_counts = X_sc_counts[:, [sc_map[g] for g in genes]]
    X_sc = log1p_cpm(X_sc_counts)
    desc_pca32 = build_descriptors(X_sc, pca_dims=[32], nmf_dims=[], seed=args.seed)["pca32"]

    variants = [
        MLPVariant("mlp_pca32_softplus_correct", "pca32", "correct", output_mode="softplus"),
        MLPVariant("mlp_pca32_softplus_shuffled", "pca32", "shuffled", output_mode="softplus"),
        MLPVariant("mlp_pca32_softplus_random", "pca32", "random", output_mode="softplus"),
        MLPVariant("mlp_pca32_softplus_permuted_labels", "pca32", "permuted_labels", output_mode="softplus"),
    ]

    rows = []
    histories = []
    gene_rows = []
    fold_ids = [int(x) for x in args.folds.split(",") if x.strip()]
    for fold in fold_ids:
        train_idx = np.load(args.mask_dir / f"fold{fold}_train_gene_idx.npy")
        val_idx = np.load(args.mask_dir / f"fold{fold}_val_gene_idx.npy")
        test_idx = np.load(args.mask_dir / f"fold{fold}_test_gene_idx.npy")
        low_test_idx, high_spatial_idx = subgroup_indices(X, test_idx, coords)
        train_values = X[:, train_idx].reshape(-1)
        q001 = float(np.quantile(train_values, 0.001))
        q999 = float(np.quantile(train_values, 0.999))

        for i, variant in enumerate(variants):
            desc_control, note = make_descriptor_control(desc_pca32, variant.control, seed=args.seed + 1000 * fold + 17 * i)
            summary, hist, pred, _ = train_mlp_variant(
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
                seed=args.seed + 1000 * fold + 53 * i,
            )
            summary.update({"fold": int(fold), "descriptor_note": note})
            rows.append(summary)
            hist["fold"] = int(fold)
            hist["model"] = variant.name
            histories.append(hist)

            gdf = gene_metrics(X, pred, test_idx, genes)
            gdf.insert(0, "fold", int(fold))
            gdf.insert(1, "model", variant.name)
            gdf.insert(2, "control", variant.control)
            gene_rows.append(gdf)

    long_df = pd.DataFrame(rows)
    long_df.to_csv(args.out_dir / "gc_mlp_descriptor_controls_5fold_long.csv", index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(args.out_dir / "gc_mlp_descriptor_controls_5fold_training_history.csv", index=False)
    if gene_rows:
        pd.concat(gene_rows, ignore_index=True).to_csv(args.out_dir / "gc_mlp_descriptor_controls_5fold_gene_level.csv", index=False)

    metric_cols = ["SPCC", "SSIM", "RMSE", "JS", "low_expr_SPCC", "high_spatial_SPCC"]
    summary = (
        long_df.groupby(["model", "control", "output_mode"], as_index=False)
        .agg(
            **{f"{m}_mean": (m, "mean") for m in metric_cols},
            **{f"{m}_std": (m, "std") for m in metric_cols},
            n_folds=("fold", "nunique"),
        )
        .sort_values("SPCC_mean", ascending=False)
    )
    summary.to_csv(args.out_dir / "gc_mlp_descriptor_controls_5fold_summary.csv", index=False)

    correct = summary[summary["control"].eq("correct")]
    controls = summary[summary["control"].isin(["shuffled", "random", "permuted_labels"])]
    if len(correct) and len(controls):
        c = correct.iloc[0]
        pass_mask = (
            (float(c["SPCC_mean"]) > controls["SPCC_mean"].max())
            and (float(c["RMSE_mean"]) < controls["RMSE_mean"].min())
            and (float(c["JS_mean"]) < controls["JS_mean"].min())
        )
        best_control = controls.sort_values("SPCC_mean", ascending=False).iloc[0]
    else:
        pass_mask = False
        best_control = None
    decision = "DESCRIPTOR_SIGNAL_CONFIRMED" if pass_mask else "DESCRIPTOR_SIGNAL_NOT_CONFIRMED"
    lines = [
        "# GC-MLP Descriptor Controls Five-Fold Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "## Summary",
        summary.to_string(index=False),
    ]
    if len(correct) and best_control is not None:
        lines.extend(
            [
                "",
                "## Correct vs Best Control",
                f"- Best control: `{best_control['model']}`",
                f"- Delta SPCC: {float(correct.iloc[0]['SPCC_mean']) - float(best_control['SPCC_mean']):+.6f}",
                f"- Delta RMSE: {float(correct.iloc[0]['RMSE_mean']) - float(best_control['RMSE_mean']):+.6f}",
                f"- Delta JS: {float(correct.iloc[0]['JS_mean']) - float(best_control['JS_mean']):+.6f}",
            ]
        )
    (args.out_dir / "gc_mlp_descriptor_controls_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
