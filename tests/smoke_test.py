"""Synthetic end-to-end smoke test for the paper-aligned GeneSPT package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genespt.descriptors import build_gene_descriptors
from genespt.fusion import apply_fusion, select_global_fusion
from genespt.metrics import evaluate_prediction
from genespt.psp import fit_psp
from genespt.training import GCTrainingConfig, predict_gc, train_gc_decoder


def main() -> None:
    rng = np.random.default_rng(0)
    n_spots, n_cells, n_genes = 24, 40, 18
    latent_spot = rng.normal(size=(n_spots, 3))
    latent_gene = rng.normal(size=(n_genes, 3))
    st = np.clip(latent_spot @ latent_gene.T + rng.normal(scale=0.2, size=(n_spots, n_genes)) + 2.0, 0.0, None)
    scrna = rng.poisson(lam=np.clip(rng.lognormal(mean=1.0, sigma=0.5, size=(n_cells, n_genes)), 0.0, 20.0))
    train_idx = np.arange(0, 10)
    val_idx = np.arange(10, 14)
    test_idx = np.arange(14, 18)

    desc = build_gene_descriptors(scrna, n_pca=4, n_nmf=4, random_state=0).values
    cfg = GCTrainingConfig(epochs=5, spot_hidden_dim=16, gene_hidden_dim=8, decoder_hidden_dim=16, progress=False)
    gc = train_gc_decoder(st[:, train_idx], desc[train_idx], val_st=st[:, val_idx], val_descriptors=desc[val_idx], config=cfg)
    gc_val = predict_gc(gc, st[:, train_idx], desc[val_idx])
    gc_test = predict_gc(gc, st[:, train_idx], desc[test_idx])

    psp = fit_psp(st[:, train_idx], desc[train_idx], st[:, val_idx], desc[val_idx], n_components=3, top_k=2)
    psp_val = psp.predict(desc[val_idx])
    psp_test = psp.predict(desc[test_idx])
    fusion = select_global_fusion(gc_val, psp_val, st[:, val_idx])
    pred = apply_fusion(gc_test, psp_test, fusion.psp_weight)

    per_gene, summary = evaluate_prediction(st[:, test_idx], pred)
    assert pred.shape == st[:, test_idx].shape
    assert len(per_gene) == len(test_idx)
    assert np.isfinite(summary[["SPCC", "RMSE", "JS/JSD", "SSIM"]].to_numpy()).all()
    print("smoke_test=ok")


if __name__ == "__main__":
    main()

