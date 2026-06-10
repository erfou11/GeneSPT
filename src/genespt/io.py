"""Small I/O helpers used by CLI scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_array(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if key is None:
            if len(data.files) != 1:
                raise ValueError(f"{path} has multiple arrays; pass a key")
            key = data.files[0]
        return data[key]
    if path.suffix in {".csv", ".txt", ".tsv"}:
        delimiter = "," if path.suffix == ".csv" else None
        return np.loadtxt(path, delimiter=delimiter)
    raise ValueError(f"unsupported array format: {path}")


def load_gene_names(path: str | Path | None, indices: np.ndarray | None = None) -> list[str] | None:
    if path is None:
        return None
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if indices is not None:
        return [names[int(i)] for i in indices]
    return names

