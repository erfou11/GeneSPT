"""scRNA-derived gene descriptor construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF, PCA


@dataclass(frozen=True)
class DescriptorResult:
    values: np.ndarray
    names: list[str]
    method: str


def log_cpm_normalize(counts: np.ndarray, scale: float = 1e4) -> np.ndarray:
    """Return log1p(CPM) normalized cell-by-gene counts."""

    x = np.asarray(counts, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("counts must be a 2D cells x genes matrix")
    totals = np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    return np.log1p(x / totals * float(scale))


def standardize_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Column-standardize an array without changing row order."""

    arr = np.asarray(x, dtype=np.float64)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    return (arr - mean) / np.maximum(std, eps)


def _component_count(requested: int, matrix: np.ndarray) -> int:
    return int(max(1, min(requested, matrix.shape[0], matrix.shape[1])))


def build_gene_descriptors(
    scrna_counts: np.ndarray,
    *,
    method: str = "pca32_nmf32",
    n_pca: int = 32,
    n_nmf: int = 32,
    random_state: int = 0,
) -> DescriptorResult:
    """Build gene descriptors from scRNA-seq reference counts.

    Parameters
    ----------
    scrna_counts:
        Cell-by-gene count matrix aligned to the ST gene order.
    method:
        One of ``pca32``, ``nmf32``, ``pca32_nmf32``, or ``mean1``.
    """

    normalized = log_cpm_normalize(scrna_counts)
    gene_by_cell = normalized.T
    pieces: list[np.ndarray] = []
    names: list[str] = []

    if "pca" in method:
        k = _component_count(int(n_pca), gene_by_cell)
        pca = PCA(n_components=k, random_state=random_state)
        pieces.append(pca.fit_transform(gene_by_cell))
        names.extend([f"pca{i + 1}" for i in range(k)])

    if "nmf" in method:
        k = _component_count(int(n_nmf), gene_by_cell)
        nmf = NMF(
            n_components=k,
            init="nndsvda",
            random_state=random_state,
            max_iter=500,
        )
        pieces.append(nmf.fit_transform(np.clip(gene_by_cell, 0.0, None)))
        names.extend([f"nmf{i + 1}" for i in range(k)])

    if method == "mean1" or not pieces:
        pieces.append(normalized.mean(axis=0, keepdims=False)[:, None])
        names.append("scrna_mean")

    values = standardize_columns(np.concatenate(pieces, axis=1))
    return DescriptorResult(values=values.astype(np.float32), names=names, method=method)


def shuffled_descriptor_control(values: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Return a row-shuffled descriptor matrix for mechanism controls."""

    rng = np.random.default_rng(random_state)
    order = rng.permutation(values.shape[0])
    return np.asarray(values)[order].copy()


def random_descriptor_control(values: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Return random descriptors with the same shape and column scale."""

    rng = np.random.default_rng(random_state)
    random_values = rng.normal(size=np.asarray(values).shape)
    return standardize_columns(random_values).astype(np.float32)

