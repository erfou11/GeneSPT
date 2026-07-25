#!/usr/bin/env python3
"""Strict whole-gene-holdout adapter for the official stAI model.

The adapter has three explicit stages:

1. ``prepare`` extracts outer-train ST expression and allowed scRNA reference
   information into a truth-free package.
2. ``run`` trains the official stAI core from that package and predicts only
   the frozen outer-test genes.  It cannot access test-gene ST expression.
3. ``evaluate`` opens the frozen truth and invokes the same centralized
   evaluator used by Protocol A.

The adapter imports the official stAI network, datasets and losses at the
pinned source commit.  Minibatching and chunked inference use interfaces
already exposed by the official implementation and preserve its top-k
Euclidean attention rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
GENESPT_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE = GENESPT_ROOT.parent
STAI_ROOT = Path(
    os.environ.get(
        "STAI_ROOT",
        WORKSPACE / "stAI",
    )
).expanduser().resolve()
ARCHIVE_ROOT = Path(
    os.environ.get(
        "GENESPT_ARCHIVE_ROOT",
        WORKSPACE / "GeneSPT_reviewer_archive",
    )
).expanduser().resolve()
PROTOCOL_INPUT_ROOT = Path(
    os.environ.get(
        "GENESPT_PROTOCOL_INPUT_ROOT",
        GENESPT_ROOT / "results" / "protocol_a_full_rerun_20260711" / "inputs",
    )
).expanduser().resolve()
METRICS_PATH = Path(
    os.environ.get(
        "GENESPT_METRICS_PATH",
        GENESPT_ROOT / "src" / "genespt" / "metrics.py",
    )
).expanduser().resolve()
OFFICIAL_COMMIT = "3376cc16cc6d8461edafc0aeb4519b92d18474b7"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: str
    role: str
    scrna_orientation: str
    label_kind: str
    label_path: Path
    label_column: str
    label_alignment: str


def configured_path(environment_variable: str, default: Path) -> Path:
    return Path(
        os.environ.get(environment_variable, default)
    ).expanduser().resolve()


DATASETS = {
    "Vis9A": DatasetSpec(
        "Vis9A",
        "Vis9A_D7_spaim_effective4470",
        "primary",
        "cells_by_genes",
        "csv",
        configured_path(
            "STAI_VIS9A_LABEL_PATH",
            GENESPT_ROOT
            / "data/GSE161318_raw_probe/extracted/GSE159500/"
            "GSM4831163_D7_Ev3_meta.csv.gz",
        ),
        "cell.types",
        "strip_10x_suffix_and_subset_to_author_annotated_cells",
    ),
    "HBC": DatasetSpec(
        "HBC",
        "HBC_shared16112",
        "primary",
        "cells_by_genes",
        "h5ad",
        configured_path(
            "STAI_HBC_LABEL_PATH",
            GENESPT_ROOT
            / "data/Processed Data/dataset21-30/Dataset28/"
            "scRNA_count_cluster.h5ad",
        ),
        "merge_cell_type",
        "exact_cell_id",
    ),
    "Cell2location": DatasetSpec(
        "Cell2location",
        "Cell2location_mouse_brain_ST8059048_shared12819",
        "primary",
        "cells_by_genes",
        "h5ad",
        configured_path(
            "STAI_CELL2LOCATION_LABEL_PATH",
            GENESPT_ROOT
            / "data/downloads/cell2location_mouse_brain/"
            "regression_model/sc.h5ad",
        ),
        "annotation_1",
        "exact_cell_id",
    ),
    "seqFISH+": DatasetSpec(
        "seqFISH+",
        "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000",
        "cross_platform",
        "cells_by_genes",
        "h5ad",
        configured_path(
            "STAI_SEQFISH_LABEL_PATH",
            STAI_ROOT / "data/osmFISH_Zeisel/zeisel_scRNA.h5ad",
        ),
        "celltype",
        "exact_cell_id",
    ),
    "MHPR": DatasetSpec(
        "MHPR",
        "MHPR_current_panel",
        "cross_platform",
        "genes_by_cells",
        "h5ad",
        configured_path(
            "STAI_MHPR_LABEL_PATH",
            GENESPT_ROOT
            / "data/Processed Data/dataset1-10/Dataset6/"
            "scRNA_count_cluster.h5ad",
        ),
        "merge_cell_type",
        "exact_cell_id",
    ),
    "MVC": DatasetSpec(
        "MVC",
        "MVC_shared981",
        "cross_platform",
        "cells_by_genes",
        "h5ad",
        configured_path(
            "STAI_MVC_LABEL_PATH",
            GENESPT_ROOT
            / "data/Processed Data/dataset1-10/Dataset10/"
            "scRNA_count_cluster.h5ad",
        ),
        "merge_cell_type",
        "verified_row_order",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def dataset_paths(spec: DatasetSpec) -> dict[str, Path]:
    base = ARCHIVE_ROOT / "processed_datasets" / spec.role / spec.dataset_id
    return {
        "base": base,
        "st": base / "Spatial_count.txt",
        "scrna": base / "scRNA_count.txt",
        "locations": base / "Locations.txt",
    }


def split_path(spec: DatasetSpec, fold: int) -> Path:
    return PROTOCOL_INPUT_ROOT / spec.dataset_id / f"fold{fold}" / "mode_a_split.json"


def truth_path(spec: DatasetSpec, fold: int) -> Path:
    return PROTOCOL_INPUT_ROOT / spec.dataset_id / f"fold{fold}" / "full_truth.npy"


def split_arrays(split: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantic_keys = (
        "inner_train_gene_idx",
        "inner_validation_gene_idx",
        "final_test_gene_idx",
    )
    fallback_keys = ("train_gene_idx", "val_gene_idx", "test_gene_idx")
    keys = semantic_keys if all(key in split for key in semantic_keys) else fallback_keys
    arrays = []
    for key in keys:
        if key not in split:
            raise KeyError(f"Frozen split is missing {key}")
        array = np.asarray(split[key], dtype=np.int64)
        if array.ndim != 1 or len(array) != len(np.unique(array)):
            raise ValueError(f"Invalid {key}")
        arrays.append(array)
    train_idx, val_idx, test_idx = arrays
    if any(np.intersect1d(a, b).size for a, b in ((train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx))):
        raise ValueError("Frozen train/validation/test sets overlap")
    return train_idx, val_idx, test_idx


def split_gene_names(split: dict[str, Any], key: str, indices: np.ndarray) -> list[str]:
    candidates = {
        "train": ("inner_train_genes", "train_genes"),
        "validation": (
            "inner_validation_genes",
            "val_genes",
            "validation_genes",
        ),
        "test": ("final_test_genes", "test_target_genes", "test_genes"),
    }[key]
    for candidate in candidates:
        if candidate in split:
            names = [str(value) for value in split[candidate]]
            if len(names) != len(indices):
                raise ValueError(f"{candidate} length does not match its indices")
            return names
    all_genes = split.get("gene_names")
    if not isinstance(all_genes, list):
        raise KeyError(f"Frozen split has no {key} gene-name list")
    return [str(all_genes[int(index)]) for index in indices]


def _has_index_column(header: Sequence[str]) -> bool:
    return bool(header) and str(header[0]).strip().casefold() in {
        "",
        "cell_id",
        "spot_id",
        "barcode",
        "index",
        "unnamed: 0",
    }


def read_cells_by_genes(path: Path, selected_genes: Sequence[str] | None = None):
    header = pd.read_csv(path, sep="\t", nrows=0).columns.astype(str).tolist()
    if not header:
        raise ValueError(f"Invalid cells-by-genes table: {path}")
    has_index = _has_index_column(header)
    id_column = header[0] if has_index else None
    genes = header[1:] if has_index else header
    selected = genes if selected_genes is None else [str(gene) for gene in selected_genes]
    missing = sorted(set(selected) - set(genes))
    if missing:
        raise ValueError(f"Missing genes in {path.name}: {missing[:10]}")
    usecols = ([id_column] if id_column is not None else []) + selected
    frame = pd.read_csv(path, sep="\t", usecols=usecols)
    cell_ids = (
        frame[id_column].astype(str).to_numpy(dtype=object)
        if id_column is not None
        else np.asarray([f"row_{index}" for index in range(len(frame))], dtype=object)
    )
    matrix = frame.loc[:, selected].to_numpy(dtype=np.float32)
    return matrix, selected, cell_ids, genes


def cells_by_genes_axis(path: Path) -> list[str]:
    header = pd.read_csv(path, sep="\t", nrows=0).columns.astype(str).tolist()
    if not header:
        raise ValueError(f"Invalid cells-by-genes table: {path}")
    return header[1:] if _has_index_column(header) else header


def read_genes_by_cells(path: Path, selected_genes: Sequence[str] | None = None):
    wanted = None if selected_genes is None else {str(gene) for gene in selected_genes}
    rows: dict[str, np.ndarray] = {}
    order: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        cell_ids = np.asarray(header[1:], dtype=object)
        for row in reader:
            gene = str(row[0])
            order.append(gene)
            if wanted is None or gene in wanted:
                rows[gene] = np.asarray(row[1:], dtype=np.float32)
    selected = order if selected_genes is None else [str(gene) for gene in selected_genes]
    missing = [gene for gene in selected if gene not in rows]
    if missing:
        raise ValueError(f"Missing genes in {path.name}: {missing[:10]}")
    matrix = np.stack([rows[gene] for gene in selected], axis=1)
    return matrix, selected, cell_ids, order


def read_genes_by_cells_reference(path: Path, selected_genes: Sequence[str]):
    """Read selected genes while deriving library sizes from the full reference."""

    selected = [str(gene) for gene in selected_genes]
    wanted = set(selected)
    rows: dict[str, np.ndarray] = {}
    source_gene_count = 0
    library_size: np.ndarray | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        cell_ids = np.asarray(header[1:], dtype=object)
        library_size = np.zeros(len(cell_ids), dtype=np.float64)
        for row in reader:
            source_gene_count += 1
            gene = str(row[0])
            values = np.asarray(row[1:], dtype=np.float32)
            if len(values) != len(cell_ids):
                raise ValueError(f"Malformed scRNA row for {gene}")
            library_size += values
            if gene in wanted:
                rows[gene] = values
    missing = [gene for gene in selected if gene not in rows]
    if missing:
        raise ValueError(f"Missing genes in {path.name}: {missing[:10]}")
    matrix = np.stack([rows[gene] for gene in selected], axis=1)
    return matrix, selected, cell_ids, source_gene_count, library_size


def read_scrna(spec: DatasetSpec, path: Path, selected_genes: Sequence[str]):
    if spec.scrna_orientation == "cells_by_genes":
        matrix, genes, cell_ids, source_genes = read_cells_by_genes(
            path, selected_genes
        )
        return matrix, genes, cell_ids, len(source_genes), None
    if spec.scrna_orientation == "genes_by_cells":
        return read_genes_by_cells_reference(path, selected_genes)
    raise ValueError(f"Unknown scRNA orientation: {spec.scrna_orientation}")


def load_locations(path: Path, expected_rows: int) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t")
    if len(frame) != expected_rows or frame.shape[1] < 2:
        frame = pd.read_csv(path, sep=r"\s+", engine="python")
    if len(frame) != expected_rows or frame.shape[1] < 2:
        raise ValueError(f"Location shape mismatch: {frame.shape}, expected {expected_rows} rows")
    values = frame.iloc[:, :2].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Locations contain NaN/Inf")
    return values


def load_reference_labels(
    spec: DatasetSpec, cell_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ids = [str(value) for value in cell_ids]
    if spec.label_kind == "csv":
        frame = pd.read_csv(spec.label_path)
        source_id_column = str(frame.columns[0])
        labels_by_id = {
            str(row[source_id_column]): row[spec.label_column]
            for _, row in frame.iterrows()
        }
        selected: list[int] = []
        labels: list[str] = []
        for index, cell_id in enumerate(ids):
            lookup = cell_id[:-2] if cell_id.endswith("-1") else cell_id
            value = labels_by_id.get(lookup)
            if value is not None and not pd.isna(value):
                selected.append(index)
                labels.append(str(value))
    elif spec.label_kind == "h5ad":
        import anndata as ad

        adata = ad.read_h5ad(spec.label_path, backed="r")
        if spec.label_column not in adata.obs:
            raise KeyError(f"{spec.label_column} absent from {spec.label_path}")
        source_ids = adata.obs_names.astype(str).tolist()
        source_labels = adata.obs[spec.label_column].astype(str).tolist()
        if spec.label_alignment == "verified_row_order":
            if len(source_ids) != len(ids):
                raise ValueError("Label source row count differs from reference matrix")
            selected = list(range(len(ids)))
            labels = source_labels
        else:
            mapping = dict(zip(source_ids, source_labels, strict=True))
            missing = [cell_id for cell_id in ids if cell_id not in mapping]
            if missing:
                raise ValueError(f"Label source misses reference cells: {missing[:10]}")
            selected = list(range(len(ids)))
            labels = [mapping[cell_id] for cell_id in ids]
    else:
        raise ValueError(f"Unknown label kind: {spec.label_kind}")

    selected_array = np.asarray(selected, dtype=np.int64)
    labels_array = np.asarray(labels, dtype=object)
    if selected_array.size == 0 or len(labels_array) != len(selected_array):
        raise ValueError("No usable scRNA labels")
    categories = list(dict.fromkeys(labels_array.tolist()))
    label_to_int = {label: index for index, label in enumerate(categories)}
    encoded = np.asarray([label_to_int[label] for label in labels_array], dtype=np.int64)
    audit = {
        "source": str(spec.label_path),
        "source_sha256": sha256_file(spec.label_path),
        "column": spec.label_column,
        "alignment": spec.label_alignment,
        "reference_cell_count_before_label_requirement": len(ids),
        "reference_cell_count_used": int(len(selected_array)),
        "reference_cells_excluded_without_author_label": int(len(ids) - len(selected_array)),
        "class_count": len(categories),
        "classes": categories,
        "pseudo_labels_used": False,
    }
    return selected_array, encoded, audit


def normalize_scrna(
    counts: np.ndarray, library_size: np.ndarray | None = None
) -> np.ndarray:
    counts64 = np.asarray(counts, dtype=np.float64)
    library = counts64.sum(axis=1) if library_size is None else np.asarray(
        library_size, dtype=np.float64
    )
    if library.shape != (counts64.shape[0],):
        raise ValueError("scRNA library-size vector has the wrong shape")
    if np.any(library <= 0):
        raise ValueError("scRNA reference contains zero-library cells")
    normalized = np.log1p(counts64 * (10000.0 / library[:, None]))
    return normalized.astype(np.float32)


def scale_columns(values: np.ndarray) -> np.ndarray:
    import anndata as ad
    import scanpy as sc

    adata = ad.AnnData(X=np.asarray(values, dtype=np.float32))
    sc.pp.scale(adata, max_value=10)
    scaled = np.asarray(adata.X, dtype=np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled expression contains NaN/Inf")
    return scaled


def prepare(spec: DatasetSpec, fold: int, output_dir: Path) -> Path:
    started = time.time()
    paths = dataset_paths(spec)
    frozen_split_path = split_path(spec, fold)
    for path in (*paths.values(), frozen_split_path, spec.label_path):
        if not path.exists():
            raise FileNotFoundError(path)

    split = load_json(frozen_split_path)
    train_idx, val_idx, test_idx = split_arrays(split)
    split_train_genes = split_gene_names(split, "train", train_idx)
    split_val_genes = split_gene_names(split, "validation", val_idx)
    split_test_genes = split_gene_names(split, "test", test_idx)

    full_st_gene_order = cells_by_genes_axis(paths["st"])
    train_genes = [full_st_gene_order[int(index)] for index in train_idx]
    val_genes = [full_st_gene_order[int(index)] for index in val_idx]
    test_genes = [full_st_gene_order[int(index)] for index in test_idx]
    for label, split_names, table_names in (
        ("train", split_train_genes, train_genes),
        ("validation", split_val_genes, val_genes),
        ("test", split_test_genes, test_genes),
    ):
        if [name.casefold() for name in split_names] != [
            name.casefold() for name in table_names
        ]:
            raise ValueError(f"Frozen {label} names disagree with ST indices")

    st_train, st_genes, spot_ids, loaded_st_gene_order = read_cells_by_genes(
        paths["st"], train_genes
    )
    if st_genes != train_genes:
        raise ValueError("ST train-gene order changed")
    if loaded_st_gene_order != full_st_gene_order:
        raise ValueError("ST gene axis changed while reading")
    all_split_indices = np.concatenate([train_idx, val_idx, test_idx])
    if sorted(all_split_indices.tolist()) != list(range(len(full_st_gene_order))):
        raise ValueError("Frozen split is not a complete partition of the ST gene axis")
    locations = load_locations(paths["locations"], len(spot_ids))
    (
        scrna_counts,
        scrna_genes,
        cell_ids,
        scrna_source_gene_count,
        scrna_library_size,
    ) = read_scrna(spec, paths["scrna"], full_st_gene_order)
    if scrna_genes != full_st_gene_order:
        raise ValueError("scRNA and ST gene axes differ")
    selected_cells, labels, label_audit = load_reference_labels(spec, cell_ids)
    scrna_counts = scrna_counts[selected_cells]
    if scrna_library_size is not None:
        scrna_library_size = scrna_library_size[selected_cells]
    used_cell_ids = cell_ids[selected_cells]

    scrna_log = normalize_scrna(scrna_counts, scrna_library_size)
    st_train_scaled = scale_columns(st_train)
    scrna_train_scaled = scale_columns(scrna_log[:, train_idx])
    scrna_test_reference = np.asarray(scrna_log[:, test_idx], dtype=np.float32)
    del st_train, scrna_counts, scrna_log

    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / "truth_free_training_package.npz"
    np.savez_compressed(
        package_path,
        st_train_scaled=st_train_scaled,
        scrna_train_scaled=scrna_train_scaled,
        scrna_test_reference=scrna_test_reference,
        locations=locations,
        labels=labels,
        train_gene_idx=train_idx,
        val_gene_idx=val_idx,
        test_gene_idx=test_idx,
        train_genes=np.asarray(train_genes, dtype=str),
        val_genes=np.asarray(val_genes, dtype=str),
        test_genes=np.asarray(test_genes, dtype=str),
        spot_ids=np.asarray(spot_ids, dtype=str),
        reference_cell_ids=np.asarray(used_cell_ids, dtype=str),
    )
    package_hash = sha256_file(package_path)
    manifest = {
        "stage": "prepare",
        "dataset": spec.name,
        "dataset_id": spec.dataset_id,
        "fold": fold,
        "protocol": "A",
        "official_stai_commit": OFFICIAL_COMMIT,
        "adapter_source": str(SCRIPT_PATH),
        "adapter_source_sha256": sha256_file(SCRIPT_PATH),
        "frozen_split": str(frozen_split_path),
        "frozen_split_sha256": sha256_file(frozen_split_path),
        "train_gene_count": len(train_genes),
        "validation_gene_count": len(val_genes),
        "test_gene_count": len(test_genes),
        "spot_count": len(spot_ids),
        "reference_cell_count": len(used_cell_ids),
        "scrna_source_gene_count_for_library_size": int(scrna_source_gene_count),
        "scrna_shared_gene_count_retained": len(full_st_gene_order),
        "split_name_case_only_differences": {
            "train": int(
                sum(a != b for a, b in zip(split_train_genes, train_genes, strict=True))
            ),
            "validation": int(
                sum(a != b for a, b in zip(split_val_genes, val_genes, strict=True))
            ),
            "test": int(
                sum(a != b for a, b in zip(split_test_genes, test_genes, strict=True))
            ),
        },
        "label_provenance": label_audit,
        "information_boundary": {
            "test_gene_st_expression_in_package": False,
            "validation_gene_st_expression_in_package": False,
            "outer_train_gene_st_expression_in_package": True,
            "test_gene_scrna_expression_in_package": True,
            "test_gene_scrna_role": "allowed external reference expression",
            "pseudo_labels_used": False,
        },
        "preprocessing": {
            "scrna": "normalize_total_1e4_then_log1p; train columns scaled with scanpy max_value=10",
            "st": "outer-train columns only; scanpy scale max_value=10",
            "test_st": "not read by model-stage process",
        },
        "package": str(package_path),
        "package_sha256": package_hash,
        "package_keys": sorted(np.load(package_path, allow_pickle=False).files),
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "prepare_manifest.json", manifest)
    return package_path


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def official_modules():
    if str(STAI_ROOT) not in sys.path:
        sys.path.insert(0, str(STAI_ROOT))
    from stAI.datasets import rnaCollate, rnaDataset, spatialCollate, spatialDataset
    from stAI.stai import stAI
    from stAI.utils import location_to_edge

    return stAI, spatialDataset, rnaDataset, spatialCollate, rnaCollate, location_to_edge


def chunked_impute(
    model,
    st_fit: np.ndarray,
    sc_fit: np.ndarray,
    sc_reference: np.ndarray,
    locations: np.ndarray,
    *,
    spatial_knn: int,
    topk: int,
    query_chunk: int,
    encoder_chunk: int,
    device,
    location_to_edge,
) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        edge = location_to_edge(locations, spatial_knn).to(device)
        st_tensor = torch.from_numpy(st_fit).to(device)
        st_latent = model.ST_encoder(st_tensor, edge)
        del st_tensor, edge

        sc_latent_parts = []
        for start in range(0, len(sc_fit), encoder_chunk):
            batch = torch.from_numpy(sc_fit[start : start + encoder_chunk]).to(device)
            sc_latent_parts.append(model.SC_encoder(batch).cpu())
        sc_latent = torch.cat(sc_latent_parts, dim=0).to(device)
        del sc_latent_parts

        outputs = []
        effective_topk = min(int(topk), int(sc_latent.shape[0]))
        for start in range(0, len(st_fit), query_chunk):
            query = st_latent[start : start + query_chunk]
            negative_distance = -torch.cdist(query, sc_latent, p=2)
            values, indices = torch.topk(negative_distance, k=effective_topk, dim=1)
            weights = torch.softmax(values, dim=1)
            reference = torch.from_numpy(sc_reference[indices.cpu().numpy()]).to(device)
            prediction = torch.sum(reference * weights[:, :, None], dim=1)
            outputs.append(prediction.cpu().numpy().astype(np.float32))
            del negative_distance, values, indices, weights, reference, prediction
        return np.concatenate(outputs, axis=0)


def run_model(
    spec: DatasetSpec,
    fold: int,
    output_dir: Path,
    *,
    epochs: int,
    internal_folds: int,
    spatial_batch_size: int,
    scrna_batch_size: int,
    spatial_knn: int,
    topk: int,
    seed: int,
    query_chunk: int,
    encoder_chunk: int,
) -> Path:
    import torch
    from sklearn.model_selection import KFold
    from torch.utils.data import DataLoader

    package_path = output_dir / "truth_free_training_package.npz"
    prepare_manifest_path = output_dir / "prepare_manifest.json"
    if not package_path.is_file() or not prepare_manifest_path.is_file():
        raise FileNotFoundError("Run prepare before model training")
    prepare_manifest = load_json(prepare_manifest_path)
    if sha256_file(package_path) != prepare_manifest["package_sha256"]:
        raise ValueError("Truth-free package hash mismatch")

    package = np.load(package_path, allow_pickle=False)
    forbidden = {"truth", "test_truth", "full_truth", "st_test"}
    if forbidden.intersection(package.files):
        raise ValueError("Model-stage package contains forbidden test ST truth")
    st_train = np.asarray(package["st_train_scaled"], dtype=np.float32)
    sc_train = np.asarray(package["scrna_train_scaled"], dtype=np.float32)
    sc_test = np.asarray(package["scrna_test_reference"], dtype=np.float32)
    locations = np.asarray(package["locations"], dtype=np.float32)
    labels = np.asarray(package["labels"], dtype=np.int64)
    train_genes = np.asarray(package["train_genes"], dtype=str)
    test_genes = np.asarray(package["test_genes"], dtype=str)
    if st_train.shape[1] != len(train_genes) or sc_train.shape[1] != len(train_genes):
        raise ValueError("Training matrix/gene shape mismatch")
    if sc_test.shape != (sc_train.shape[0], len(test_genes)):
        raise ValueError("scRNA test-reference shape mismatch")
    if not all(np.isfinite(array).all() for array in (st_train, sc_train, sc_test, locations)):
        raise ValueError("Model-stage arrays contain NaN/Inf")

    stAI, spatialDataset, rnaDataset, spatialCollate, rnaCollate, location_to_edge = official_modules()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_parameters = {
        "d_hidden": 64,
        "d_latent": 16,
        "lam_clf": 0.1,
        "lam_cos": 1.0,
        "lam_genegraph": 0.0,
        "lam_impute": 1.0,
        "lam_mmd": 0.1,
        "lam_recon": 1.0,
        "topk": topk,
    }
    training_parameters = {
        "epochs": epochs,
        "internal_folds": internal_folds,
        "learning_rate": 0.002,
        "spatial_batch_size": spatial_batch_size,
        "scrna_batch_size": scrna_batch_size,
        "spatial_knn": spatial_knn,
        "seed": seed,
        "query_chunk": query_chunk,
        "encoder_chunk": encoder_chunk,
    }

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    submodel_dir = output_dir / "submodels"
    submodel_dir.mkdir(exist_ok=True)
    splitter = KFold(n_splits=internal_folds, shuffle=True, random_state=seed)
    predictions = []
    submodel_records = []

    for submodel, (fit_idx, calibration_idx) in enumerate(splitter.split(train_genes)):
        sub_started = time.time()
        set_seed(seed)
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        calibration_idx = np.asarray(calibration_idx, dtype=np.int64)
        spatial_data = spatialDataset(
            st_train[:, fit_idx], st_train[:, calibration_idx], locations
        )
        rna_data = rnaDataset(
            sc_train[:, fit_idx], sc_train[:, calibration_idx], labels
        )
        effective_spatial_batch = min(spatial_batch_size, len(spatial_data))
        effective_scrna_batch = min(scrna_batch_size, len(rna_data))
        spatial_loader = DataLoader(
            spatial_data,
            batch_size=effective_spatial_batch,
            collate_fn=spatialCollate(device=device, knn=spatial_knn),
            drop_last=True,
            shuffle=True,
        )
        rna_loader = DataLoader(
            rna_data,
            batch_size=effective_scrna_batch,
            collate_fn=rnaCollate(device=device),
            drop_last=True,
            shuffle=True,
        )
        model = stAI(
            d_input=len(fit_idx),
            n_classes=len(np.unique(labels)),
            **model_parameters,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        epoch_losses = []
        for epoch in range(epochs):
            model.train()
            losses = []
            for spatial_batch, rna_batch in zip(spatial_loader, rna_loader):
                st_fit_batch, st_calibration_batch, st_edge_batch = spatial_batch
                sc_fit_batch, sc_calibration_batch, sc_label_batch, sc_genegraph_batch = rna_batch
                optimizer.zero_grad(set_to_none=True)
                loss, *_ = model(
                    st_fit_batch,
                    st_calibration_batch,
                    st_edge_batch,
                    sc_fit_batch,
                    sc_calibration_batch,
                    sc_label_batch,
                    sc_genegraph_batch,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in submodel {submodel}, epoch {epoch}")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if not losses:
                raise RuntimeError("No stAI minibatches were produced")
            epoch_losses.append(float(np.mean(losses)))

        checkpoint_path = submodel_dir / f"submodel_{submodel}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        prediction = chunked_impute(
            model,
            st_train[:, fit_idx],
            sc_train[:, fit_idx],
            sc_test,
            locations,
            spatial_knn=spatial_knn,
            topk=topk,
            query_chunk=query_chunk,
            encoder_chunk=encoder_chunk,
            device=device,
            location_to_edge=location_to_edge,
        )
        if prediction.shape != (st_train.shape[0], len(test_genes)):
            raise ValueError("Submodel prediction shape mismatch")
        if not np.isfinite(prediction).all():
            raise ValueError("Submodel prediction contains NaN/Inf")
        sub_prediction_path = submodel_dir / f"submodel_{submodel}_test_prediction.npy"
        np.save(sub_prediction_path, prediction.astype(np.float32))
        predictions.append(prediction)
        submodel_records.append(
            {
                "submodel": submodel,
                "fit_gene_count": len(fit_idx),
                "calibration_gene_count": len(calibration_idx),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "prediction": str(sub_prediction_path),
                "prediction_sha256": sha256_file(sub_prediction_path),
                "first_epoch_loss": epoch_losses[0],
                "final_epoch_loss": epoch_losses[-1],
                "elapsed_seconds": time.time() - sub_started,
            }
        )
        del model, optimizer, prediction
        torch.cuda.empty_cache()

    ensemble = np.mean(np.stack(predictions, axis=0), axis=0).astype(np.float32)
    prediction_path = output_dir / "stai_test_prediction.npy"
    np.save(prediction_path, ensemble)
    run_manifest = {
        "stage": "run",
        "dataset": spec.name,
        "dataset_id": spec.dataset_id,
        "fold": fold,
        "protocol": "A",
        "official_stai_commit": OFFICIAL_COMMIT,
        "official_core_files": {
            "stai.py": sha256_file(STAI_ROOT / "stAI/stai.py"),
            "datasets.py": sha256_file(STAI_ROOT / "stAI/datasets.py"),
            "layers.py": sha256_file(STAI_ROOT / "stAI/layers.py"),
            "losses.py": sha256_file(STAI_ROOT / "stAI/losses.py"),
        },
        "adapter_source_sha256": sha256_file(SCRIPT_PATH),
        "truth_free_package_sha256": sha256_file(package_path),
        "test_st_truth_accessed": False,
        "validation_st_expression_used": False,
        "outer_train_only_internal_calibration": True,
        "model_parameters": model_parameters,
        "training_parameters": training_parameters,
        "prediction": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_shape": list(ensemble.shape),
        "prediction_dtype": str(ensemble.dtype),
        "submodels": submodel_records,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    return prediction_path


def load_metrics_module():
    spec = importlib.util.spec_from_file_location("genespt_centralized_metrics", METRICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(METRICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(spec: DatasetSpec, fold: int, output_dir: Path) -> dict[str, Any]:
    package_path = output_dir / "truth_free_training_package.npz"
    prediction_path = output_dir / "stai_test_prediction.npy"
    run_manifest_path = output_dir / "run_manifest.json"
    if not all(path.is_file() for path in (package_path, prediction_path, run_manifest_path)):
        raise FileNotFoundError("Prepare and run stages must finish before evaluation")
    run_manifest = load_json(run_manifest_path)
    if run_manifest.get("test_st_truth_accessed") is not False:
        raise ValueError("Run manifest does not prove a truth-free model stage")
    if sha256_file(prediction_path) != run_manifest["prediction_sha256"]:
        raise ValueError("Prediction hash mismatch")

    package = np.load(package_path, allow_pickle=False)
    test_idx = np.asarray(package["test_gene_idx"], dtype=np.int64)
    test_genes = np.asarray(package["test_genes"], dtype=str).tolist()
    prediction = np.load(prediction_path, allow_pickle=False)
    truth = np.load(truth_path(spec, fold), mmap_mode="r", allow_pickle=False)
    truth_test = np.asarray(truth[:, test_idx], dtype=np.float32)
    if prediction.shape != truth_test.shape:
        raise ValueError(f"Prediction/truth shape mismatch: {prediction.shape} vs {truth_test.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(truth_test).all():
        raise ValueError("Prediction/truth contains NaN/Inf")

    metrics = load_metrics_module()
    per_gene, summary = metrics.evaluate_prediction(
        truth_test, prediction, gene_names=test_genes
    )
    if len(summary) != 1 or len(per_gene) != len(test_genes):
        raise ValueError("Centralized evaluator returned unexpected dimensions")
    per_gene_path = output_dir / "centralized_metrics_per_gene.csv"
    summary_path = output_dir / "centralized_metrics_summary.csv"
    per_gene.to_csv(per_gene_path, index=False)
    summary.to_csv(summary_path, index=False)
    row = summary.iloc[0].to_dict()
    evaluation_manifest = {
        "stage": "evaluate",
        "dataset": spec.name,
        "dataset_id": spec.dataset_id,
        "fold": fold,
        "protocol": "A",
        "truth_access_first_allowed_stage": "evaluate",
        "truth": str(truth_path(spec, fold)),
        "truth_sha256": sha256_file(truth_path(spec, fold)),
        "prediction_sha256": sha256_file(prediction_path),
        "metrics_source": str(METRICS_PATH),
        "metrics_source_sha256": sha256_file(METRICS_PATH),
        "summary": row,
        "per_gene_csv": str(per_gene_path),
        "per_gene_csv_sha256": sha256_file(per_gene_path),
        "summary_csv": str(summary_path),
        "summary_csv_sha256": sha256_file(summary_path),
    }
    write_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
    return evaluation_manifest


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--internal-folds", type=int, default=5)
    parser.add_argument("--spatial-batch-size", type=int, default=512)
    parser.add_argument("--scrna-batch-size", type=int, default=512)
    parser.add_argument("--spatial-knn", type=int, default=10)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=8848)
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--encoder-chunk", type=int, default=1024)
    args = parser.parse_args(argv)
    for name in (
        "epochs",
        "internal_folds",
        "spatial_batch_size",
        "scrna_batch_size",
        "spatial_knn",
        "topk",
        "query_chunk",
        "encoder_chunk",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = DATASETS[args.dataset]
    output_dir = args.output_dir.resolve()
    if args.stage == "prepare":
        result = prepare(spec, args.fold, output_dir)
        print(result)
    elif args.stage == "run":
        result = run_model(
            spec,
            args.fold,
            output_dir,
            epochs=args.epochs,
            internal_folds=args.internal_folds,
            spatial_batch_size=args.spatial_batch_size,
            scrna_batch_size=args.scrna_batch_size,
            spatial_knn=args.spatial_knn,
            topk=args.topk,
            seed=args.seed,
            query_chunk=args.query_chunk,
            encoder_chunk=args.encoder_chunk,
        )
        print(result)
    else:
        result = evaluate(spec, args.fold, output_dir)
        print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
