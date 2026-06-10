"""Run GeneSPT-GC for one strict whole-gene fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from genespt.descriptors import build_gene_descriptors
from genespt.io import load_array
from genespt.training import GCTrainingConfig, predict_gc, train_gc_decoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    st = load_array(cfg["st_matrix"])
    scrna = load_array(cfg["scrna_matrix"])
    train_idx = load_array(cfg["train_gene_indices"]).astype(int)
    val_idx = load_array(cfg["val_gene_indices"]).astype(int)
    test_idx = load_array(cfg["test_gene_indices"]).astype(int)

    desc = build_gene_descriptors(
        scrna,
        method=cfg.get("descriptor_method", "pca32_nmf32"),
        n_pca=int(cfg.get("n_pca", 32)),
        n_nmf=int(cfg.get("n_nmf", 32)),
        random_state=int(cfg.get("seed", 0)),
    ).values

    train_cfg = GCTrainingConfig(**cfg.get("training", {}))
    model = train_gc_decoder(
        st[:, train_idx],
        desc[train_idx],
        val_st=st[:, val_idx],
        val_descriptors=desc[val_idx],
        config=train_cfg,
    )
    pred_test = predict_gc(model, st[:, train_idx], desc[test_idx], device=train_cfg.device)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "prediction_test.npy", pred_test)
    (out_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "method": "GeneSPT-GC",
                "n_spots": int(st.shape[0]),
                "n_train_genes": int(train_idx.size),
                "n_val_genes": int(val_idx.size),
                "n_test_genes": int(test_idx.size),
                "descriptor_method": cfg.get("descriptor_method", "pca32_nmf32"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

