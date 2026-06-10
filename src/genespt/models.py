"""Gene-conditioned decoder used by GeneSPT-GC."""

from __future__ import annotations

import torch
from torch import nn


class GeneConditionedDecoder(nn.Module):
    """Shared decoder for spot-by-gene conditional prediction.

    The model consumes the observed training-gene ST matrix to encode each spot
    and consumes a descriptor for each queried gene. It then predicts expression
    for all spot/gene query pairs with shared parameters, so held-out genes do
    not require fixed output columns.
    """

    def __init__(
        self,
        n_train_genes: int,
        descriptor_dim: int,
        *,
        spot_hidden_dim: int = 128,
        gene_hidden_dim: int = 64,
        decoder_hidden_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.spot_encoder = nn.Sequential(
            nn.Linear(n_train_genes, spot_hidden_dim),
            nn.LayerNorm(spot_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spot_hidden_dim, spot_hidden_dim),
            nn.GELU(),
        )
        self.gene_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, gene_hidden_dim),
            nn.LayerNorm(gene_hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(spot_hidden_dim + gene_hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, train_expression: torch.Tensor, gene_descriptors: torch.Tensor) -> torch.Tensor:
        """Predict expression for query genes.

        Parameters
        ----------
        train_expression:
            Tensor of shape ``spots x train_genes``.
        gene_descriptors:
            Tensor of shape ``query_genes x descriptor_dim``.
        """

        spot_z = self.spot_encoder(train_expression)
        gene_z = self.gene_encoder(gene_descriptors)
        n_spots, n_genes = spot_z.shape[0], gene_z.shape[0]
        spot_rep = spot_z[:, None, :].expand(n_spots, n_genes, spot_z.shape[1])
        gene_rep = gene_z[None, :, :].expand(n_spots, n_genes, gene_z.shape[1])
        joined = torch.cat([spot_rep, gene_rep], dim=-1)
        return self.decoder(joined).squeeze(-1)

