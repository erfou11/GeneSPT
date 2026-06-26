#!/usr/bin/env python3
"""Recompute benchmark metrics from the Zenodo prediction-matrix addendum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genespt.io import load_array, load_gene_names  # noqa: E402
from genespt.metrics import evaluate_prediction  # noqa: E402


METRICS = ["SPCC", "RMSE", "JS", "SSIM", "JS/JSD"]
REFERENCE_COLUMNS = {
    "SPCC": "benchmark_SPCC_mean",
    "RMSE": "benchmark_RMSE_mean",
    "JS/JSD": "benchmark_JS_JSD_mean",
    "SSIM": "benchmark_raw_SSIM_mean",
}


def comma_values(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_folds(value: str | None) -> set[int] | None:
    if not value:
        return None
    folds: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        folds.add(int(item))
    return folds


def resolve_required(root: Path, rel: str) -> Path:
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def evaluate_manifest_row(addendum_root: Path, row: pd.Series) -> dict[str, object]:
    dataset_id = str(row["dataset_id"])
    pred_path = resolve_required(addendum_root, str(row["matrix_path"]))
    test_idx_path = resolve_required(addendum_root, str(row["test_gene_idx_path"]))
    true_path = resolve_required(addendum_root, f"ground_truth/{dataset_id}/st_log1p_cpm.npy")
    gene_names_path = resolve_required(addendum_root, f"ground_truth/{dataset_id}/gene_names.txt")

    test_idx = load_array(test_idx_path).astype(int)
    true_full = load_array(true_path)
    pred = load_array(pred_path)
    true_test = true_full[:, test_idx]
    if pred.shape[1] == true_full.shape[1]:
        pred = pred[:, test_idx]
    if pred.shape != true_test.shape:
        raise ValueError(
            f"{dataset_id} {row['method']} fold{row['fold']}: "
            f"prediction shape {pred.shape} does not match test shape {true_test.shape}"
        )

    names = load_gene_names(gene_names_path, test_idx)
    _, summary = evaluate_prediction(true_test, pred, names)
    out = {
        "dataset": row.get("dataset", ""),
        "dataset_id": dataset_id,
        "role": row.get("role", ""),
        "method": row["method"],
        "fold": int(row["fold"]),
        "matrix_scope": row.get("matrix_scope", ""),
        "n_spots": int(pred.shape[0]),
        "n_test_genes": int(pred.shape[1]),
        "source_benchmark_csv": row.get("source_benchmark_csv", ""),
    }
    for metric in METRICS:
        out[metric] = float(summary.loc[0, metric])
    return out


def add_reference_deltas(aggregate: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    ref_cols = ["dataset_id", "method"] + list(REFERENCE_COLUMNS.values())
    refs = manifest[ref_cols].drop_duplicates(["dataset_id", "method"]).copy()
    merged = aggregate.merge(refs, on=["dataset_id", "method"], how="left")
    for metric, ref_col in REFERENCE_COLUMNS.items():
        if ref_col in merged:
            merged[f"{metric}_minus_manifest_reference"] = merged[f"{metric}_mean"] - merged[ref_col]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GeneSPT prediction-matrix addendum metrics.")
    parser.add_argument("--addendum-root", type=Path, required=True, help="Extracted GeneSPT Zenodo prediction-matrix addendum.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/reproduction/prediction_matrix_addendum_check"))
    parser.add_argument("--manifest", default="manifests/PREDICTION_MATRIX_MANIFEST.csv")
    parser.add_argument("--datasets", help="Comma-separated dataset_id filter.")
    parser.add_argument("--methods", help="Comma-separated method filter.")
    parser.add_argument("--folds", help="Comma-separated fold filter, for example 0,1,2.")
    parser.add_argument("--max-rows", type=int, help="Evaluate only the first N selected manifest rows.")
    args = parser.parse_args()

    addendum_root = args.addendum_root.resolve()
    manifest_path = resolve_required(addendum_root, args.manifest)
    manifest = pd.read_csv(manifest_path)

    selected = manifest.copy()
    datasets = comma_values(args.datasets)
    methods = comma_values(args.methods)
    folds = parse_folds(args.folds)
    if datasets is not None:
        selected = selected[selected["dataset_id"].isin(datasets)]
    if methods is not None:
        selected = selected[selected["method"].isin(methods)]
    if folds is not None:
        selected = selected[selected["fold"].astype(int).isin(folds)]
    if args.max_rows is not None:
        selected = selected.head(args.max_rows)
    if selected.empty:
        raise ValueError("No manifest rows selected.")

    rows = [evaluate_manifest_row(addendum_root, row) for _, row in selected.iterrows()]
    fold_metrics = pd.DataFrame(rows)

    group_cols = ["dataset", "dataset_id", "role", "method", "source_benchmark_csv"]
    aggregate = (
        fold_metrics.groupby(group_cols, dropna=False)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in aggregate.columns.to_flat_index()
    ]
    aggregate = add_reference_deltas(aggregate, selected)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / "fold_metrics.csv"
    aggregate_path = out_dir / "aggregate_metrics.csv"
    fold_metrics.to_csv(fold_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    print(f"selected_rows={len(selected)}")
    print(f"fold_metrics={fold_path}")
    print(f"aggregate_metrics={aggregate_path}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
