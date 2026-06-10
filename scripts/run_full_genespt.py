"""Run full GeneSPT for one strict whole-gene fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from genespt.descriptors import build_gene_descriptors
from genespt.fusion import apply_fusion, select_global_fusion
from genespt.io import load_array
from genespt.psp import fit_psp
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

    train_st = st[:, train_idx]
    val_st = st[:, val_idx]
    test_desc = desc[test_idx]

    train_cfg = GCTrainingConfig(**cfg.get("training", {}))
    gc_model = train_gc_decoder(
        train_st,
        desc[train_idx],
        val_st=val_st,
        val_descriptors=desc[val_idx],
        config=train_cfg,
    )

    gc_val = predict_gc(gc_model, train_st, desc[val_idx], device=train_cfg.device)
    gc_test = predict_gc(gc_model, train_st, test_desc, device=train_cfg.device)

    psp_cfg = cfg.get("psp", {})
    psp_model = fit_psp(
        train_st,
        desc[train_idx],
        val_st,
        desc[val_idx],
        n_components=int(psp_cfg.get("n_components", 16)),
        ridge_alpha=float(psp_cfg.get("ridge_alpha", 10.0)),
        min_component_corr=float(psp_cfg.get("min_component_corr", 0.0)),
        top_k=psp_cfg.get("top_k"),
    )
    psp_val = psp_model.predict(desc[val_idx])
    psp_test = psp_model.predict(test_desc)

    fusion = select_global_fusion(gc_val, psp_val, val_st)
    pred_test = apply_fusion(gc_test, psp_test, fusion.psp_weight)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "prediction_test.npy", pred_test)
    np.save(out_dir / "prediction_test_gc.npy", gc_test)
    np.save(out_dir / "prediction_test_psp.npy", psp_test)
    np.save(out_dir / "psp_component_scores.npy", psp_model.component_scores)
    (out_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "method": "GeneSPT",
                "n_spots": int(st.shape[0]),
                "n_train_genes": int(train_idx.size),
                "n_val_genes": int(val_idx.size),
                "n_test_genes": int(test_idx.size),
                "descriptor_method": cfg.get("descriptor_method", "pca32_nmf32"),
                "selected_psp_components": psp_model.selected_components.tolist(),
                "psp_weight": fusion.psp_weight,
                "validation_rmse": fusion.validation_rmse,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

