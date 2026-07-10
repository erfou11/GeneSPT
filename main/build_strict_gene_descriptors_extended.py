#!/usr/bin/env python3
"""Build extended scRNA gene descriptors for strict whole-gene experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import MiniBatchNMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler

from run_strict_gene_conditioned_decoder_gate import load_matrix, log1p_cpm


def standardize_desc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return StandardScaler(with_mean=True, with_std=True).fit_transform(x).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--st-counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/Spatial_count.txt"))
    ap.add_argument("--scrna-counts-path", type=Path, default=Path("/workspace/GeneSPT/data/Vis9A_D7_spaim_effective4470/scRNA_count.txt"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/GeneSPT/results/imformation/strict_gene_descriptors_extended"))
    ap.add_argument("--audit-md", type=Path, default=Path("/workspace/GeneSPT/results/imformation/strict_gene_descriptor_extended_audit.md"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell-pca-dim", type=int, default=32)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, st_genes, _ = load_matrix(args.st_counts_path, index_col=None)
    X_sc_counts, sc_genes, sc_cells = load_matrix(args.scrna_counts_path, index_col=0)
    if list(sc_genes) != list(st_genes):
        sc_map = {g: i for i, g in enumerate(sc_genes)}
        missing = [g for g in st_genes if g not in sc_map]
        if missing:
            raise ValueError(f"Missing ST genes in scRNA descriptors: {missing[:10]}")
        X_sc_counts = X_sc_counts[:, [sc_map[g] for g in st_genes]]
        sc_genes = list(st_genes)
    X_sc = log1p_cpm(X_sc_counts)

    descriptors: dict[str, np.ndarray] = {}
    gene_by_cell = X_sc.T.astype(np.float32)
    for dim in [16, 32, 64]:
        svd = TruncatedSVD(n_components=dim, random_state=args.seed)
        descriptors[f"pca{dim}"] = standardize_desc(svd.fit_transform(gene_by_cell))
    for dim in [16, 32, 64]:
        nmf = MiniBatchNMF(
            n_components=dim,
            random_state=args.seed,
            max_iter=300,
            batch_size=512,
            init="nndsvda",
            beta_loss="frobenius",
        )
        descriptors[f"nmf{dim}"] = standardize_desc(nmf.fit_transform(np.clip(gene_by_cell, 0.0, None)))

    cell_svd = TruncatedSVD(n_components=min(args.cell_pca_dim, X_sc.shape[1] - 1), random_state=args.seed)
    cell_embed = cell_svd.fit_transform(X_sc.astype(np.float32))
    for k in [16, 32, 64]:
        km = MiniBatchKMeans(n_clusters=k, random_state=args.seed + k, batch_size=min(2048, X_sc.shape[0]), n_init=5)
        labels = km.fit_predict(cell_embed)
        means = np.zeros((k, X_sc.shape[1]), dtype=np.float32)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        for c in range(k):
            if counts[c] > 0:
                means[c] = X_sc[labels == c].mean(axis=0)
        descriptors[f"cluster{k}_mean"] = standardize_desc(means.T)
        pd.DataFrame({"cell": sc_cells, "cluster": labels}).to_csv(args.out_dir / f"cell_cluster_k{k}.csv", index=False)

    descriptors["scrna_mean1"] = standardize_desc(X_sc.mean(axis=0)[:, None])
    descriptors["pca32_nmf32"] = standardize_desc(np.concatenate([descriptors["pca32"], descriptors["nmf32"]], axis=1))
    descriptors["pca32_scrna_mean1"] = standardize_desc(np.concatenate([descriptors["pca32"], descriptors["scrna_mean1"]], axis=1))
    descriptors["pca32_cluster32"] = standardize_desc(np.concatenate([descriptors["pca32"], descriptors["cluster32_mean"]], axis=1))

    for name, arr in descriptors.items():
        np.save(args.out_dir / f"{name}.npy", arr.astype(np.float32))
    pd.Series(st_genes, name="gene").to_csv(args.out_dir / "genes.csv", index=False)
    manifest = {
        "st_counts_path": str(args.st_counts_path),
        "scrna_counts_path": str(args.scrna_counts_path),
        "n_cells": int(X_sc.shape[0]),
        "n_genes": int(X_sc.shape[1]),
        "descriptors": {k: list(map(int, v.shape)) for k, v in descriptors.items()},
        "preprocessing": "scRNA and ST genes aligned by name; scRNA counts normalized by log1p(CPM*1e4); descriptor columns standardized.",
        "mask_safety": "No ST test-gene values are used; descriptors are derived from scRNA_count.txt only.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# Strict Gene Descriptor Extended Audit",
        "",
        f"scRNA cells: `{X_sc.shape[0]}`",
        f"genes: `{X_sc.shape[1]}`",
        "",
        "## Preprocessing",
        manifest["preprocessing"],
        "",
        "## Mask Safety",
        manifest["mask_safety"],
        "",
        "## Descriptors",
    ]
    for k, shape in manifest["descriptors"].items():
        lines.append(f"- `{k}`: shape `{shape}`")
    args.audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest["descriptors"], indent=2))


if __name__ == "__main__":
    main()
