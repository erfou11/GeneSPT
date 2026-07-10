import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-name", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-data-dir", required=True)
    parser.add_argument("--output-split-dir", required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop-zero-st-genes",
        action="store_true",
        help="Drop shared genes whose ST column sum is zero before writing the adapted dataset.",
    )
    return parser.parse_args()


def read_spatial_header(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return next(csv.reader(f, delimiter="\t"))


def read_scrna_gene_axis(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        first_col = header[0] if header else ""
        row_labels = [row[0] for row in reader if row]
    return first_col, row_labels


def detect_scrna_orientation(st_genes, scrna_path: Path):
    df = pd.read_csv(scrna_path, sep="\t", index_col=0, nrows=50)
    st_gene_set = set(map(str, st_genes))
    row_overlap = len(set(map(str, df.index)) & st_gene_set)
    col_overlap = len(set(map(str, df.columns)) & st_gene_set)
    if row_overlap >= col_overlap:
        return "genes_x_cells"
    return "cells_x_genes"


def load_spatial_counts(path: Path):
    return pd.read_csv(path, sep="\t")


def load_scrna_counts(path: Path, orientation: str):
    df = pd.read_csv(path, sep="\t", index_col=0)
    if orientation == "genes_x_cells":
        return df.T
    return df


def make_gene5cv_folds(var_names, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_idx = np.arange(len(var_names))
    folds = []
    for fold_idx, (_, test_idx) in enumerate(kf.split(all_idx)):
        test_idx = np.asarray(test_idx, dtype=np.int64)
        train_idx = np.setdiff1d(all_idx, test_idx)
        folds.append(
            {
                "fold_index": fold_idx,
                "train_gene_idx": train_idx.tolist(),
                "test_gene_idx": test_idx.tolist(),
                "train_genes": [str(var_names[i]) for i in train_idx],
                "test_genes": [str(var_names[i]) for i in test_idx],
            }
        )
    return folds


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_data_dir = Path(args.output_data_dir)
    out_split_dir = Path(args.output_split_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_split_dir.mkdir(parents=True, exist_ok=True)

    locations_path = source_dir / "Locations.txt"
    spatial_path = source_dir / "Spatial_count.txt"
    scrna_path = source_dir / "scRNA_count.txt"

    st_genes = list(map(str, read_spatial_header(spatial_path)))
    scrna_orientation = detect_scrna_orientation(st_genes, scrna_path)

    spatial_df = load_spatial_counts(spatial_path)
    scrna_df = load_scrna_counts(scrna_path, scrna_orientation)
    spatial_df.columns = list(map(str, spatial_df.columns))
    scrna_df.columns = list(map(str, scrna_df.columns))
    scrna_df.index = list(map(str, scrna_df.index))

    shared_genes = [g for g in spatial_df.columns if g in set(scrna_df.columns)]
    missing_st_only = [g for g in spatial_df.columns if g not in set(scrna_df.columns)]
    zero_sum_st_shared = []
    if args.drop_zero_st_genes:
        st_shared_sum = spatial_df[shared_genes].sum(axis=0)
        zero_sum_st_shared = [g for g in shared_genes if float(st_shared_sum[g]) == 0.0]
        shared_genes = [g for g in shared_genes if g not in set(zero_sum_st_shared)]

    spatial_shared = spatial_df[shared_genes].copy()
    scrna_shared = scrna_df[shared_genes].copy()
    final_genes = list(map(str, spatial_shared.columns))
    scrna_shared.index.name = "cell_id"

    spatial_shared.to_csv(out_data_dir / "Spatial_count.txt", sep="\t", index=False)
    scrna_shared.to_csv(out_data_dir / "scRNA_count.txt", sep="\t", index=True)
    pd.read_csv(locations_path, sep="\t").to_csv(out_data_dir / "Locations.txt", sep="\t", index=False)

    manifest = {
        "source_dataset": args.source_dataset_name,
        "st_genes": int(len(spatial_df.columns)),
        "sc_genes": int(scrna_df.shape[1]),
        "shared_genes": int(len(final_genes)),
        "shared_ratio": float(len(final_genes) / max(len(spatial_df.columns), 1)),
        "sc_orientation_source": scrna_orientation,
        "sc_orientation_output": "cells_x_genes",
        "missing_st_only_genes": missing_st_only,
        "drop_zero_st_genes": bool(args.drop_zero_st_genes),
        "zero_sum_st_shared_genes_removed": zero_sum_st_shared,
        "n_spots": int(spatial_df.shape[0]),
        "n_scrna_cells": int(scrna_df.shape[0]),
    }
    (out_data_dir / "shared_gene_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    folds = make_gene5cv_folds(final_genes, n_splits=args.cv_folds, seed=args.seed)
    for fold in folds:
        fold_path = out_split_dir / f"fold{fold['fold_index']}.json"
        fold_path.write_text(json.dumps(fold, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(
        {
            "output_data_dir": str(out_data_dir),
            "output_split_dir": str(out_split_dir),
            "shared_genes": len(final_genes),
            "st_genes": len(spatial_df.columns),
            "sc_genes": int(scrna_df.shape[1]),
            "n_spots": int(spatial_df.shape[0]),
            "n_scrna_cells": int(scrna_df.shape[0]),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
