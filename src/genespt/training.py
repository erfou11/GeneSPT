"""Training helpers for GeneSPT-GC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange

from .models import GeneConditionedDecoder


@dataclass(frozen=True)
class GCTrainingConfig:
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    spot_hidden_dim: int = 128
    gene_hidden_dim: int = 64
    decoder_hidden_dim: int = 128
    dropout: float = 0.05
    device: str = "auto"
    seed: int = 0
    progress: bool = True


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_gc_decoder(
    train_st: np.ndarray,
    train_descriptors: np.ndarray,
    *,
    val_st: np.ndarray | None = None,
    val_descriptors: np.ndarray | None = None,
    config: GCTrainingConfig | None = None,
) -> GeneConditionedDecoder:
    """Train the GeneSPT-GC branch on training genes only."""

    cfg = config or GCTrainingConfig()
    torch.manual_seed(int(cfg.seed))
    device = _device(cfg.device)

    x_train = torch.as_tensor(train_st, dtype=torch.float32, device=device)
    d_train = torch.as_tensor(train_descriptors, dtype=torch.float32, device=device)
    model = GeneConditionedDecoder(
        n_train_genes=x_train.shape[1],
        descriptor_dim=d_train.shape[1],
        spot_hidden_dim=cfg.spot_hidden_dim,
        gene_hidden_dim=cfg.gene_hidden_dim,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    iterator = trange(cfg.epochs, disable=not cfg.progress, desc="train GeneSPT-GC")
    best_state = None
    best_val = float("inf")

    val_tensors = None
    if val_st is not None and val_descriptors is not None:
        val_tensors = (
            torch.as_tensor(val_st, dtype=torch.float32, device=device),
            torch.as_tensor(val_descriptors, dtype=torch.float32, device=device),
        )

    for _ in iterator:
        model.train()
        pred = model(x_train, d_train)
        loss = F.mse_loss(pred, x_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if val_tensors is not None:
            model.eval()
            with torch.no_grad():
                y_val, d_val = val_tensors
                val_loss = F.mse_loss(model(x_train, d_val), y_val).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            iterator.set_postfix(train=float(loss.item()), val=best_val)
        else:
            iterator.set_postfix(train=float(loss.item()))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_gc(
    model: GeneConditionedDecoder,
    train_st: np.ndarray,
    query_descriptors: np.ndarray,
    *,
    device: str = "auto",
) -> np.ndarray:
    """Predict query-gene expression with a trained GeneSPT-GC model."""

    dev = _device(device)
    model = model.to(dev)
    x_train = torch.as_tensor(train_st, dtype=torch.float32, device=dev)
    d_query = torch.as_tensor(query_descriptors, dtype=torch.float32, device=dev)
    return model(x_train, d_query).detach().cpu().numpy()

