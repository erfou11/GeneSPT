#!/usr/bin/env python3
"""Prepare a full-context dataset while preserving shared-gene evaluation.

The existing manuscript benchmark writes only the shared ST/scRNA genes into
Spatial_count.txt and scRNA_count.txt.  This adapter keeps those files for
backward compatibility, but also writes full ST and full scRNA context matrices
so prototype models can use non-shared genes as auxiliary information without
changing the evaluation target.

Outputs:
  Spatial_count.txt              shared evaluation genes, ST units x genes
  scRNA_count.txt                shared evaluation genes, scRNA cells x genes
  ST_context_count.txt           all retained ST genes, ST units x genes
  scRNA_context_count.txt        all retained scRNA genes, scRNA cells x genes
  Locations.txt                  copied from source
  gene_context_manifest.json     gene-set provenance and context categories

Split JSON files are written over shared evaluation genes only.  Test gene
ground truth remains reserved for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-name", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-data-dir", type=Path, required=True)
    parser.add_argument("--output-split-dir", type=Path, required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction-within-nontest", type=float, default=0.20)
    parser.add_argument(
        "--drop-zero-st-shared-genes",
        action="store_true",
        help="Drop shared evaluation genes whose ST column sum is zero.",
    )
    parser.add_argument(
        "--drop-zero-st-context-genes",
        action="store_true",
        help="Drop ST context genes whose ST column sum is zero.",
    )
    return parser.parse_args()


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return next(csv.reader(handle, delimiter="\t"))


def detect_scrna_orientation(st_genes: list[str], scrna_path: Path) -> str:
    df = pd.read_csv(scrna_path, sep="\t", index_col=0, nrows=50)
    st_set = set(map(str, st_genes))
    row_overlap = len(set(map(str, df.index)) & st_set)
    col_overlap = len(set(map(str, df.columns)) & st_set)
    return "genes_x_cells" if row_overlap >= col_overlap else "cells_x_genes"


def load_scrna(path: Path, orientation: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    if orientation == "genes_x_cells":
        df = df.T
    df.index = list(map(str, df.index))
    df.columns = list(map(str, df.columns))
    df.index.name = "cell_id"
    return df


def make_gene5cv_folds(
    genes: list[str],
    n_splits: int,
    seed: int,
    validation_fraction_within_nontest: float,
) -> list[dict]:
    all_idx = np.arange(len(genes))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (_, test_idx) in enumerate(kf.split(all_idx)):
        test_idx = np.asarray(test_idx, dtype=np.int64)
        nontest_idx = np.setdiff1d(all_idx, test_idx)
        train_idx, val_idx = train_test_split(
            nontest_idx,
            test_size=validation_fraction_within_nontest,
            random_state=seed + fold_idx,
            shuffle=True,
        )
        train_idx = np.asarray(sorted(train_idx), dtype=np.int64)
        val_idx = np.asarray(sorted(val_idx), dtype=np.int64)
        test_idx = np.asarray(sorted(test_idx), dtype=np.int64)
        folds.append(
            {
                "dataset": None,
                "fold": int(fold_idx),
                "train_gene_idx": train_idx.tolist(),
                "val_gene_idx": val_idx.tolist(),
                "test_gene_idx": test_idx.tolist(),
                "train_genes": [genes[i] for i in train_idx],
                "val_genes": [genes[i] for i in val_idx],
                "test_genes": [genes[i] for i in test_idx],
                "split_scope": "shared_eval_genes_only",
            }
        )
    return folds


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    out_data = args.output_data_dir
    out_split = args.output_split_dir
    out_data.mkdir(parents=True, exist_ok=True)
    out_split.mkdir(parents=True, exist_ok=True)

    spatial_path = source_dir / "Spatial_count.txt"
    scrna_path = source_dir / "scRNA_count.txt"
    locations_path = source_dir / "Locations.txt"
    if not spatial_path.exists() or not scrna_path.exists() or not locations_path.exists():
        raise FileNotFoundError("source-dir must contain Spatial_count.txt, scRNA_count.txt and Locations.txt")

    st_header = list(map(str, read_header(spatial_path)))
    scrna_orientation = detect_scrna_orientation(st_header, scrna_path)

    st = pd.read_csv(spatial_path, sep="\t")
    st.columns = list(map(str, st.columns))
    scrna = load_scrna(scrna_path, scrna_orientation)

    st_genes_all = list(st.columns)
    scrna_genes_all = list(scrna.columns)
    st_set = set(st_genes_all)
    scrna_set = set(scrna_genes_all)
    shared_eval_genes = [g for g in st_genes_all if g in scrna_set]

    zero_sum_st_shared: list[str] = []
    if args.drop_zero_st_shared_genes:
        sums = st[shared_eval_genes].sum(axis=0)
        zero_sum_st_shared = [g for g in shared_eval_genes if float(sums[g]) == 0.0]
        shared_eval_genes = [g for g in shared_eval_genes if g not in set(zero_sum_st_shared)]

    st_context_genes = list(st_genes_all)
    zero_sum_st_context: list[str] = []
    if args.drop_zero_st_context_genes:
        sums = st[st_context_genes].sum(axis=0)
        zero_sum_st_context = [g for g in st_context_genes if float(sums[g]) == 0.0]
        st_context_genes = [g for g in st_context_genes if g not in set(zero_sum_st_context)]

    st_shared = st[shared_eval_genes].copy()
    scrna_shared = scrna[shared_eval_genes].copy()
    st_context = st[st_context_genes].copy()
    scrna_context = scrna[scrna_genes_all].copy()

    st_shared.to_csv(out_data / "Spatial_count.txt", sep="\t", index=False)
    scrna_shared.to_csv(out_data / "scRNA_count.txt", sep="\t", index=True)
    st_context.to_csv(out_data / "ST_context_count.txt", sep="\t", index=False)
    scrna_context.to_csv(out_data / "scRNA_context_count.txt", sep="\t", index=True)
    pd.read_csv(locations_path, sep="\t").to_csv(out_data / "Locations.txt", sep="\t", index=False)

    st_only_context_genes = [g for g in st_context_genes if g not in scrna_set]
    scrna_only_context_genes = [g for g in scrna_genes_all if g not in st_set]
    manifest = {
        "source_dataset": args.source_dataset_name,
        "source_dir": str(source_dir),
        "spatial_count_source": str(spatial_path),
        "scrna_count_source": str(scrna_path),
        "sc_orientation_source": scrna_orientation,
        "sc_orientation_output": "cells_x_genes",
        "n_spatial_units": int(st.shape[0]),
        "n_scrna_cells": int(scrna.shape[0]),
        "n_st_genes_all": int(len(st_genes_all)),
        "n_scrna_genes_all": int(len(scrna_genes_all)),
        "n_shared_eval_genes": int(len(shared_eval_genes)),
        "n_st_context_genes": int(len(st_context_genes)),
        "n_scrna_context_genes": int(len(scrna_genes_all)),
        "n_st_only_context_genes": int(len(st_only_context_genes)),
        "n_scrna_only_context_genes": int(len(scrna_only_context_genes)),
        "shared_eval_genes": shared_eval_genes,
        "st_only_context_genes": st_only_context_genes,
        "scrna_only_context_genes": scrna_only_context_genes,
        "zero_sum_st_shared_genes_removed": zero_sum_st_shared,
        "zero_sum_st_context_genes_removed": zero_sum_st_context,
        "evaluation_policy": "Metrics and frozen gene5cv splits are defined only on shared_eval_genes.",
        "context_policy": "ST-only and scRNA-only genes are retained as auxiliary context only; they are not evaluated as held-out targets unless separate ground truth exists.",
        "test_gene_use_policy": "Held-out shared test-gene ST values must not be used for training, validation selection or context construction.",
        "backward_compatible_files": {
            "Spatial_count.txt": "shared evaluation genes only",
            "scRNA_count.txt": "shared evaluation genes only",
        },
        "full_context_files": {
            "ST_context_count.txt": "all retained ST genes",
            "scRNA_context_count.txt": "all retained scRNA genes",
        },
    }
    (out_data / "gene_context_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    folds = make_gene5cv_folds(
        shared_eval_genes,
        n_splits=args.cv_folds,
        seed=args.seed,
        validation_fraction_within_nontest=args.validation_fraction_within_nontest,
    )
    for fold in folds:
        fold["dataset"] = args.source_dataset_name
        (out_split / f"fold{fold['fold']}_split.json").write_text(
            json.dumps(fold, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "output_data_dir": str(out_data),
                "output_split_dir": str(out_split),
                "n_shared_eval_genes": manifest["n_shared_eval_genes"],
                "n_st_only_context_genes": manifest["n_st_only_context_genes"],
                "n_scrna_only_context_genes": manifest["n_scrna_only_context_genes"],
                "n_spatial_units": manifest["n_spatial_units"],
                "n_scrna_cells": manifest["n_scrna_cells"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
