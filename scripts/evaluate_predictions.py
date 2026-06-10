"""Centralized strict whole-gene evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from genespt.io import load_array, load_gene_names
from genespt.metrics import evaluate_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one prediction matrix.")
    parser.add_argument("--true", required=True, help="Full ST matrix, spots x genes.")
    parser.add_argument("--pred", required=True, help="Prediction matrix, spots x test genes or spots x all genes.")
    parser.add_argument("--test-indices", required=True)
    parser.add_argument("--gene-names")
    parser.add_argument("--out", required=True, help="Summary CSV path.")
    parser.add_argument("--per-gene-out")
    args = parser.parse_args()

    test_idx = load_array(args.test_indices).astype(int)
    true_full = load_array(args.true)
    pred = load_array(args.pred)
    true_test = true_full[:, test_idx]
    if pred.shape[1] == true_full.shape[1]:
        pred = pred[:, test_idx]
    if pred.shape != true_test.shape:
        raise ValueError(f"prediction shape {pred.shape} does not match test shape {true_test.shape}")

    names = load_gene_names(args.gene_names, test_idx)
    per_gene, summary = evaluate_prediction(true_test, pred, names)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    per_gene_path = Path(args.per_gene_out) if args.per_gene_out else out.with_name(out.stem + "_per_gene.csv")
    per_gene.to_csv(per_gene_path, index=False)


if __name__ == "__main__":
    main()

