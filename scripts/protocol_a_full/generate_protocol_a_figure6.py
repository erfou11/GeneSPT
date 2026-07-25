#!/usr/bin/env python3
"""Rebuild Protocol A Figure 6 from the formal prediction matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "protocol_a_full_rerun_20260711"
ARCHIVE_ROOT = (
    WORKSPACE_ROOT / "zenodo_upload" / "GeneSPT_reviewer_archive_20260713"
)
OUTPUT_DIR = RESULTS_ROOT / "figures" / "figure6"

DATASET = "seqFISH+ cortex/SVZ"
DATASET_ID = "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000"
METHODS = ("GeneSPT", "Tangram", "TransImp", "SpaIM", "SpaGE", "stPlus", "stAI")
BASELINES = METHODS[1:]
FOLDS = (0, 1, 2, 3, 4)
EXPECTED_CELL_COUNT = 913
EXPECTED_GENE_COUNT = 10000
EXPECTED_CELL_TYPE_COUNT = 10
EXPECTED_TEST_GENES_PER_FOLD = 2000
EXPECTED_RECORDS_PER_METHOD = 50
PANEL_C_SAMPLE_SIZE = 6000
PANEL_C_SEED = 20260712
EFFECT_FORMULA = "(mean_in-mean_out)/sqrt((var_in+var_out)/2)"
VARIANCE_DDOF = 0

D_PROTOCOL_ID = "protocol_a_figure6d_mhpr_all154_pca30_k15_weighted_louvain_v1"
D_MATRIX_MODE = "protocol_a_completed_outer_fold_matrix"
D_DATASET = "MHPR/MERFISH"
D_DATASET_ID = "MHPR_current_panel"
D_METRICS = ("ARI", "AMI", "NMI", "homogeneity")

COLORS = {
    "GeneSPT": "#c7352f",
    "Tangram": "#4f8f46",
    "TransImp": "#7a5ca8",
    "SpaIM": "#3b8b8c",
    "SpaGE": "#607d9a",
    "stPlus": "#c9852c",
    "stAI": "#c95f8f",
}
GRID_COLOR = "#e8e8e8"
SOURCE_SCHEMA = "protocol_a_figure6_source_v2"

OUTPUT_FILENAMES = {
    "pdf": "protocol_a_figure6.pdf",
    "png": "protocol_a_figure6.png",
    "source": "protocol_a_figure6_source.csv",
    "panel_b_records": "protocol_a_figure6_panel_b_records.csv",
    "panel_c_effects": "protocol_a_figure6_panel_c_effects.csv",
    "caption": "protocol_a_figure6_caption_draft.md",
    "hash_manifest": "protocol_a_figure6_hash_manifest.csv",
}

B_METRICS = (
    ("effect_spearman", "Effect Spearman", "higher"),
    ("effect_mae", "Effect MAE", "lower"),
    ("top20_overlap_count", "Top-20 positive-effect overlap", "higher"),
    ("top50_overlap_count", "Top-50 positive-effect overlap", "higher"),
)

# Keep all four metrics in the source tables, while the main figure gives
# Panels B and C distinct roles: marker prioritization and pooled effect ranking.
B_DISPLAY_METRICS = B_METRICS[2:]

SOURCE_COLUMNS = (
    "source_schema_version",
    "panel",
    "row_type",
    "dataset",
    "dataset_id",
    "method",
    "result_layer",
    "fold",
    "source_key",
    "metric",
    "metric_direction",
    "value",
    "sd",
    "n_records",
    "n_effect_pairs",
    "n_valid_effect_pairs",
    "n_plotted",
    "panel_c_seed",
    "display_effect_min",
    "display_effect_max",
    "effect_formula",
    "variance_ddof",
    "folds",
    "protocol_id",
    "configuration",
    "workflow_order",
    "workflow_step",
    "split_artifact_id",
    "truth_artifact_id",
    "prediction_artifact_id",
    "prediction_audit_artifact_id",
    "label_artifact_id",
    "locations_artifact_id",
    "detail_artifact_id",
    "source_artifact_id",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_artifact_id(path: Path, project_root: Path, archive_root: Path) -> str:
    """Return a stable package-relative ID and reject unscoped local paths."""

    resolved = path.resolve(strict=False)
    for prefix, root in (
        ("repository", project_root.resolve(strict=True)),
        ("archive", archive_root.resolve(strict=True)),
    ):
        try:
            return f"{prefix}/{resolved.relative_to(root).as_posix()}"
        except ValueError:
            continue
    raise ValueError(f"Artifact is outside the repository and archive packages: {path.name}")


def hash_row(
    path: Path,
    *,
    scope: str,
    role: str,
    project_root: Path,
    archive_root: Path,
    method: str = "",
    fold: int | str = "",
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "artifact_scope": scope,
        "artifact_role": role,
        "method": method,
        "fold": fold,
        "artifact_id": package_artifact_id(path, project_root, archive_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return payload


def nested_value(payload: Mapping[str, Any], fields: Sequence[str]) -> Any:
    value: Any = payload
    for field in fields:
        if not isinstance(value, Mapping) or field not in value:
            raise KeyError(".".join(fields))
        value = value[field]
    return value


def render_workspace_path(
    workspace_root: Path,
    template: str | Path,
    **context: Any,
) -> Path:
    rendered = str(template).format(**context)
    candidate = Path(rendered)
    path = candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()
    try:
        path.relative_to(workspace_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"Configured artifact escapes the workspace package: {template}") from exc
    return path


def load_fixed_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict) or protocol.get("protocol_id") != D_PROTOCOL_ID:
        raise ValueError("Figure 6 requires the audited all154 Protocol A configuration")

    method_specs = protocol.get("methods", [])
    names = tuple(str(item.get("name")) for item in method_specs)
    if names != METHODS:
        raise ValueError(f"Formal method order differs from {METHODS}: {names}")
    expected_layers = {"GeneSPT": "validation_selected_readout_genespt57"}
    expected_layers.update({method: "raw_identity" for method in BASELINES})
    for method_spec in method_specs:
        name = str(method_spec["name"])
        if method_spec.get("result_layer") != expected_layers[name]:
            raise ValueError(f"{name} is not on its formal result layer")
        expected_axis = (
            "test_gene_axis" if name in {"GeneSPT", "stAI"} else "full_gene_axis"
        )
        if method_spec.get("prediction", {}).get("gene_axis") != expected_axis:
            raise ValueError(f"{name} is not configured on {expected_axis}")

    if list(protocol.get("folds", [])) != list(FOLDS):
        raise ValueError("The fixed protocol must contain folds 0-4")
    if int(protocol["dataset"]["expected_gene_count"]) != 154:
        raise ValueError("Panel D must use all 154 MHPR genes")
    preprocessing = protocol["preprocessing"]
    if (
        preprocessing.get("feature_policy") != "all_genes"
        or preprocessing.get("standardize_features") is not True
        or preprocessing.get("drop_nonfinite_features") is not False
        or preprocessing.get("drop_zero_variance_features") is not False
    ):
        raise ValueError("Panel D must pass all 154 standardized genes to PCA")
    if int(protocol["pca"]["n_components"]) != 30:
        raise ValueError("Panel D requires PCA30")
    if (
        int(protocol["neighbors"]["n_neighbors"]) != 15
        or str(protocol["neighbors"]["metric"]).lower() != "euclidean"
    ):
        raise ValueError("Panel D requires a k15 Euclidean graph")
    clustering = protocol["clustering"]
    if (
        clustering.get("algorithm") != "igraph_multilevel_louvain"
        or clustering.get("weighted") is not True
        or clustering.get("method_specific_parameters") is not False
        or int(clustering.get("rng_seed", -1)) != 0
    ):
        raise ValueError("Panel D requires seed-0 weighted Louvain for every method")
    return protocol


def load_labels(
    label_path: Path,
    locations_path: Path,
) -> pd.DataFrame:
    labels = pd.read_csv(label_path)
    required = {"matrix_row", "x", "y", "ID", "FOV", "cell_types"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Matched labels are missing columns: {sorted(missing)}")
    if len(labels) != EXPECTED_CELL_COUNT:
        raise ValueError(f"Expected {EXPECTED_CELL_COUNT} matched labels, found {len(labels)}")
    rows = pd.to_numeric(labels["matrix_row"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(rows, np.arange(EXPECTED_CELL_COUNT, dtype=np.int64)):
        raise ValueError("Matched labels do not cover matrix rows 0-912 in order")
    if labels["cell_types"].isna().any() or labels["cell_types"].astype(str).str.len().eq(0).any():
        raise ValueError("Every seqFISH+ matrix row must have a nonempty cell-type label")
    if labels["cell_types"].nunique() != EXPECTED_CELL_TYPE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CELL_TYPE_COUNT} cell types, found {labels['cell_types'].nunique()}"
        )
    expected_ids = np.asarray([f"cell_{index}" for index in range(1, EXPECTED_CELL_COUNT + 1)])
    if not np.array_equal(labels["ID"].astype(str).to_numpy(), expected_ids):
        raise ValueError("Matched label IDs are not aligned to the spatial matrix rows")

    locations = pd.read_csv(locations_path, sep="\t")
    if list(locations.columns) != ["x", "y"] or len(locations) != EXPECTED_CELL_COUNT:
        raise ValueError("seqFISH+ Locations.txt does not have the expected 913 x/y rows")
    if not np.allclose(
        labels[["x", "y"]].to_numpy(dtype=np.float64),
        locations[["x", "y"]].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Matched labels and seqFISH+ locations are not row-aligned")
    return labels


def load_fold_split(path: Path, fold: int) -> tuple[np.ndarray, np.ndarray]:
    payload = read_json(path)
    if (
        payload.get("dataset_id") != DATASET_ID
        or int(payload.get("fold", -1)) != fold
        or int(payload.get("gene_count", -1)) != EXPECTED_GENE_COUNT
    ):
        raise ValueError(f"fold{fold}: split identity differs from the seqFISH+ contract")
    test_idx = np.asarray(payload.get("final_test_gene_idx", []), dtype=np.int64)
    genes = np.asarray(payload.get("final_test_genes", []), dtype=str)
    if len(test_idx) != EXPECTED_TEST_GENES_PER_FOLD or len(genes) != len(test_idx):
        raise ValueError(f"fold{fold}: expected exactly 2,000 final test genes")
    if len(np.unique(test_idx)) != len(test_idx) or np.any(test_idx < 0) or np.any(
        test_idx >= EXPECTED_GENE_COUNT
    ):
        raise ValueError(f"fold{fold}: invalid or duplicate final test-gene indices")
    if np.any(np.char.str_len(genes) == 0):
        raise ValueError(f"fold{fold}: empty test-gene names are not allowed")
    return test_idx, genes


def audit_prediction_source(
    audit_path: Path,
    audit_spec: Mapping[str, Any],
    matrix_sha256: str,
    method: str,
    fold: int,
    test_axis_sha256: str | None = None,
) -> None:
    payload = read_json(audit_path)
    kind = str(audit_spec["kind"])
    if kind == "validation_selected_readout":
        identity = payload.get("identity", {})
        if (
            identity.get("dataset_id") != DATASET_ID
            or int(identity.get("fold", -1)) != fold
            or int(identity.get("n_spots", -1)) != EXPECTED_CELL_COUNT
            or int(identity.get("n_genes", -1)) != EXPECTED_GENE_COUNT
            or payload.get("status") != "applied"
            or payload.get("test_truth_accessed") is not False
            or payload.get("raw_prediction_unchanged") is not True
            or payload.get("test_prediction_loaded_after_lock_verification") is not True
        ):
            raise ValueError(f"{method} fold{fold}: invalid validation-readout audit")
        expected_sha256 = nested_value(payload, ("outputs", method, "sha256"))
    elif kind == "protocol_a_baseline_raw_identity":
        if payload.get("eligible_for_strict_primary") is not True:
            raise ValueError(f"{method} fold{fold}: baseline is not strict-primary eligible")
        expected_sha256 = nested_value(payload, tuple(audit_spec["sha256_path"]))
    elif kind == "protocol_a_stai_raw_identity":
        if (
            payload.get("dataset_id") != DATASET_ID
            or int(payload.get("fold", -1)) != fold
            or payload.get("method") != "stAI"
            or payload.get("status") != "complete"
        ):
            raise ValueError(f"{method} fold{fold}: invalid stAI audit identity")
        expected_sha256 = nested_value(payload, tuple(audit_spec["sha256_path"]))
        expected_axis_sha256 = nested_value(
            payload, tuple(audit_spec["test_axis_sha256_path"])
        )
        if test_axis_sha256 is None or str(expected_axis_sha256) != test_axis_sha256:
            raise ValueError(f"{method} fold{fold}: test-gene axis hash differs")
    else:
        raise ValueError(f"{method}: unsupported audit kind {kind!r}")

    if str(expected_sha256) != matrix_sha256:
        raise ValueError(f"{method} fold{fold}: prediction hash differs from its source audit")
    for check in audit_spec.get("required_values", []):
        actual = nested_value(payload, tuple(check["path"]))
        if actual != check["equals"]:
            raise ValueError(
                f"{method} fold{fold}: audit field {'.'.join(check['path'])} "
                f"is {actual!r}, expected {check['equals']!r}"
            )


def load_formal_prediction(
    workspace_root: Path,
    method_spec: Mapping[str, Any],
    fold: int,
    test_idx: np.ndarray,
    *,
    project_root: Path,
    archive_root: Path,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    method = str(method_spec["name"])
    context = {"method": method, "fold": fold, "dataset_id": DATASET_ID}
    prediction_spec = method_spec["prediction"]
    audit_spec = method_spec["audit"]
    matrix_path = render_workspace_path(workspace_root, prediction_spec["path"], **context)
    audit_path = render_workspace_path(workspace_root, audit_spec["path"], **context)
    axis_path = (
        render_workspace_path(
            workspace_root, prediction_spec["test_index_path"], **context
        )
        if prediction_spec.get("test_index_path")
        else None
    )
    required_paths = [matrix_path, audit_path]
    if axis_path is not None:
        required_paths.append(axis_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    matrix_sha256 = sha256_file(matrix_path)
    matrix_format = str(prediction_spec["format"])
    embedded_idx: np.ndarray | None = None
    if matrix_format == "npz":
        with np.load(matrix_path, allow_pickle=False) as payload:
            key = str(prediction_spec.get("array_key", "prediction"))
            index_key = str(prediction_spec.get("test_index_key", "test_gene_idx"))
            if key not in payload.files or index_key not in payload.files:
                raise KeyError(f"{method} fold{fold}: formal NPZ payload is incomplete")
            raw = np.asarray(payload[key], dtype=np.float32)
            embedded_idx = np.asarray(payload[index_key], dtype=np.int64)
            identities = {"method": method, "dataset_id": DATASET_ID, "fold": fold}
            for identity_key, expected in identities.items():
                if identity_key in payload.files and np.asarray(payload[identity_key]).item() != expected:
                    raise ValueError(
                        f"{method} fold{fold}: embedded {identity_key} identity differs"
                    )
    elif matrix_format == "npy":
        raw = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    else:
        raise ValueError(f"{method}: unsupported prediction format {matrix_format!r}")

    axis_sha256: str | None = None
    if axis_path is not None:
        axis_sha256 = sha256_file(axis_path)
        with np.load(axis_path, allow_pickle=False) as payload:
            index_key = str(prediction_spec.get("test_index_key", "test_gene_idx"))
            if index_key not in payload.files:
                raise KeyError(f"{method} fold{fold}: test-axis payload is incomplete")
            embedded_idx = np.asarray(payload[index_key], dtype=np.int64)
            identities = {"dataset_id": DATASET_ID, "fold": fold}
            for identity_key, expected in identities.items():
                if (
                    identity_key in payload.files
                    and np.asarray(payload[identity_key]).item() != expected
                ):
                    raise ValueError(
                        f"{method} fold{fold}: test-axis {identity_key} differs"
                    )

    gene_axis = str(prediction_spec["gene_axis"])
    if gene_axis == "test_gene_axis":
        if raw.shape != (EXPECTED_CELL_COUNT, len(test_idx)):
            raise ValueError(
                f"{method} fold{fold}: test-axis prediction has shape {raw.shape}"
            )
        if embedded_idx is None or not np.array_equal(embedded_idx, test_idx):
            raise ValueError(f"{method} fold{fold}: embedded test-gene axis differs")
        prediction = np.asarray(raw, dtype=np.float32)
    elif gene_axis == "full_gene_axis":
        if raw.shape != (EXPECTED_CELL_COUNT, EXPECTED_GENE_COUNT):
            raise ValueError(
                f"{method} fold{fold}: full-axis prediction has shape {raw.shape}"
            )
        prediction = np.asarray(raw[:, test_idx], dtype=np.float32)
    else:
        raise ValueError(f"{method}: unsupported gene axis {gene_axis!r}")

    audit_prediction_source(
        audit_path,
        audit_spec,
        matrix_sha256,
        method,
        fold,
        test_axis_sha256=axis_sha256,
    )
    if not np.isfinite(prediction).all():
        raise ValueError(f"{method} fold{fold}: nonfinite formal predictions are not allowed")

    matrix_id = package_artifact_id(matrix_path, project_root, archive_root)
    audit_id = package_artifact_id(audit_path, project_root, archive_root)
    source = {
        "source_key": f"{DATASET_ID}.fold{fold}.{method}",
        "method": method,
        "fold": fold,
        "result_layer": str(method_spec["result_layer"]),
        "prediction_artifact_id": matrix_id,
        "prediction_audit_artifact_id": audit_id,
    }
    hashes = [
        {
            "artifact_scope": "input",
            "artifact_role": "formal_prediction",
            "method": method,
            "fold": fold,
            "artifact_id": matrix_id,
            "bytes": matrix_path.stat().st_size,
            "sha256": matrix_sha256,
        },
        hash_row(
            audit_path,
            scope="input",
            role="prediction_audit",
            method=method,
            fold=fold,
            project_root=project_root,
            archive_root=archive_root,
        ),
    ]
    if axis_path is not None:
        hashes.append(
            hash_row(
                axis_path,
                scope="input",
                role="test_gene_axis",
                method=method,
                fold=fold,
                project_root=project_root,
                archive_root=archive_root,
            )
        )
    return prediction, source, hashes


def standardized_mean_effect(matrix: np.ndarray, in_group: np.ndarray) -> np.ndarray:
    """Compute the fixed effect using population variances and no regularizer."""

    values = np.asarray(matrix, dtype=np.float64)
    mask = np.asarray(in_group, dtype=bool)
    if values.ndim != 2 or mask.ndim != 1 or len(mask) != values.shape[0]:
        raise ValueError("Effect calculation requires a cells-by-genes matrix and row mask")
    if not mask.any() or mask.all():
        raise ValueError("Effect calculation requires nonempty in and out groups")
    mean_in = np.mean(values[mask], axis=0, dtype=np.float64)
    mean_out = np.mean(values[~mask], axis=0, dtype=np.float64)
    var_in = np.var(values[mask], axis=0, dtype=np.float64, ddof=VARIANCE_DDOF)
    var_out = np.var(values[~mask], axis=0, dtype=np.float64, ddof=VARIANCE_DDOF)
    denominator = np.sqrt((var_in + var_out) / 2.0)
    return np.divide(
        mean_in - mean_out,
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def largest_positive_gene_set(
    effect: np.ndarray,
    gene_indices: np.ndarray,
    n: int,
    jointly_valid: np.ndarray,
) -> set[int]:
    effect_values = np.asarray(effect, dtype=np.float64)
    indices = np.asarray(gene_indices, dtype=np.int64)
    eligible = jointly_valid & np.isfinite(effect_values) & (effect_values > 0.0)
    eligible_effect = effect_values[eligible]
    eligible_indices = indices[eligible]
    if not len(eligible_indices):
        return set()
    order = np.lexsort((eligible_indices, -eligible_effect))
    return set(int(value) for value in eligible_indices[order[:n]])


def effect_record(
    true_effect: np.ndarray,
    predicted_effect: np.ndarray,
    gene_indices: np.ndarray,
) -> dict[str, Any]:
    true_values = np.asarray(true_effect, dtype=np.float64)
    predicted_values = np.asarray(predicted_effect, dtype=np.float64)
    valid = np.isfinite(true_values) & np.isfinite(predicted_values)
    if int(valid.sum()) < 2:
        raise ValueError("At least two finite effect pairs are required")
    rho = float(spearmanr(true_values[valid], predicted_values[valid]).statistic)
    mae = float(np.mean(np.abs(true_values[valid] - predicted_values[valid])))
    if not math.isfinite(rho) or not math.isfinite(mae):
        raise ValueError("Fold-cell-type effect metrics must be finite")

    output: dict[str, Any] = {
        "test_gene_count": len(gene_indices),
        "valid_effect_pair_count": int(valid.sum()),
        "invalid_effect_pair_count": int((~valid).sum()),
        "true_positive_gene_count": int(np.sum(valid & (true_values > 0.0))),
        "predicted_positive_gene_count": int(np.sum(valid & (predicted_values > 0.0))),
        "effect_spearman": rho,
        "effect_mae": mae,
    }
    for n in (20, 50):
        true_top = largest_positive_gene_set(true_values, gene_indices, n, valid)
        predicted_top = largest_positive_gene_set(predicted_values, gene_indices, n, valid)
        output[f"true_top{n}_gene_count"] = len(true_top)
        output[f"predicted_top{n}_gene_count"] = len(predicted_top)
        output[f"top{n}_overlap_count"] = len(true_top.intersection(predicted_top))
    return output


def compute_effect_tables(
    *,
    project_root: Path,
    archive_root: Path,
    results_root: Path,
    protocol: Mapping[str, Any],
    labels: pd.DataFrame,
    input_hashes: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workspace_root = project_root.parent.resolve(strict=True)
    label_values = labels["cell_types"].astype(str).to_numpy()
    cell_types = sorted(labels["cell_types"].astype(str).unique().tolist())
    method_specs = {str(spec["name"]): spec for spec in protocol["methods"]}
    label_path = (
        archive_root
        / "label_provenance"
        / "matched_labels"
        / "seqfish_plus_cortex_svz_matched_cell_types.csv"
    )
    locations_path = (
        archive_root
        / "processed_datasets"
        / "cross_platform"
        / DATASET_ID
        / "Locations.txt"
    )
    label_id = package_artifact_id(label_path, project_root, archive_root)
    locations_id = package_artifact_id(locations_path, project_root, archive_root)

    record_rows: list[dict[str, Any]] = []
    effect_frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    all_test_indices: list[np.ndarray] = []

    for fold in FOLDS:
        fold_root = results_root / "inputs" / DATASET_ID / f"fold{fold}"
        split_path = fold_root / "mode_a_split.json"
        truth_path = fold_root / "full_truth.npy"
        test_idx, genes = load_fold_split(split_path, fold)
        all_test_indices.append(test_idx)
        split_id = package_artifact_id(split_path, project_root, archive_root)
        truth_id = package_artifact_id(truth_path, project_root, archive_root)
        input_hashes.extend(
            [
                hash_row(
                    split_path,
                    scope="input",
                    role="frozen_split",
                    fold=fold,
                    project_root=project_root,
                    archive_root=archive_root,
                ),
                hash_row(
                    truth_path,
                    scope="input",
                    role="formal_truth",
                    fold=fold,
                    project_root=project_root,
                    archive_root=archive_root,
                ),
            ]
        )

        truth_full = np.load(truth_path, mmap_mode="r", allow_pickle=False)
        if truth_full.shape != (EXPECTED_CELL_COUNT, EXPECTED_GENE_COUNT):
            raise ValueError(f"fold{fold}: formal truth has shape {truth_full.shape}")
        truth = np.asarray(truth_full[:, test_idx], dtype=np.float32)
        if not np.isfinite(truth).all():
            raise ValueError(f"fold{fold}: nonfinite formal truth values are not allowed")
        truth_effects = {
            cell_type: standardized_mean_effect(truth, label_values == cell_type)
            for cell_type in cell_types
        }

        for method in METHODS:
            prediction, prediction_source, prediction_hashes = load_formal_prediction(
                workspace_root,
                method_specs[method],
                fold,
                test_idx,
                project_root=project_root,
                archive_root=archive_root,
            )
            input_hashes.extend(prediction_hashes)
            source = {
                "panel": "B,C",
                "row_type": "formal_effect_input",
                "dataset": DATASET,
                "dataset_id": DATASET_ID,
                **prediction_source,
                "split_artifact_id": split_id,
                "truth_artifact_id": truth_id,
                "label_artifact_id": label_id,
                "locations_artifact_id": locations_id,
                "effect_formula": EFFECT_FORMULA,
                "variance_ddof": VARIANCE_DDOF,
            }
            source_rows.append(source)

            for cell_type in cell_types:
                mask = label_values == cell_type
                true_effect = truth_effects[cell_type]
                predicted_effect = standardized_mean_effect(prediction, mask)
                valid = np.isfinite(true_effect) & np.isfinite(predicted_effect)
                metrics = effect_record(true_effect, predicted_effect, test_idx)
                record_rows.append(
                    {
                        "dataset": DATASET,
                        "dataset_id": DATASET_ID,
                        "method": method,
                        "result_layer": prediction_source["result_layer"],
                        "fold": fold,
                        "cell_type": cell_type,
                        "n_in": int(mask.sum()),
                        "n_out": int((~mask).sum()),
                        "source_key": prediction_source["source_key"],
                        "effect_formula": EFFECT_FORMULA,
                        "variance_ddof": VARIANCE_DDOF,
                        "split_artifact_id": split_id,
                        "truth_artifact_id": truth_id,
                        "prediction_artifact_id": prediction_source[
                            "prediction_artifact_id"
                        ],
                        "prediction_audit_artifact_id": prediction_source[
                            "prediction_audit_artifact_id"
                        ],
                        "label_artifact_id": label_id,
                        **metrics,
                    }
                )
                effect_frames.append(
                    pd.DataFrame(
                        {
                            "dataset": DATASET,
                            "dataset_id": DATASET_ID,
                            "method": method,
                            "result_layer": prediction_source["result_layer"],
                            "fold": fold,
                            "cell_type": cell_type,
                            "n_in": int(mask.sum()),
                            "n_out": int((~mask).sum()),
                            "gene_index": test_idx,
                            "gene": genes,
                            "true_cell_type_effect": true_effect,
                            "predicted_cell_type_effect": predicted_effect,
                            "pair_valid": valid,
                            "source_key": prediction_source["source_key"],
                        }
                    )
                )
            del prediction
        del truth, truth_full

    combined_test_indices = np.concatenate(all_test_indices)
    if not np.array_equal(
        np.sort(combined_test_indices), np.arange(EXPECTED_GENE_COUNT, dtype=np.int64)
    ):
        raise ValueError("The five frozen test folds must partition all 10,000 genes")

    records = pd.DataFrame(record_rows)
    effects = pd.concat(effect_frames, ignore_index=True, copy=False)
    sources = pd.DataFrame(source_rows)
    return records, effects, sources


def summarize_panel_b(records: pd.DataFrame) -> pd.DataFrame:
    expected_rows = len(METHODS) * EXPECTED_RECORDS_PER_METHOD
    if len(records) != expected_rows:
        raise ValueError(f"Panel B requires {expected_rows} records, found {len(records)}")
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = records[records["method"].eq(method)]
        if len(method_rows) != EXPECTED_RECORDS_PER_METHOD:
            raise ValueError(f"{method}: Panel B requires exactly 50 fold-cell-type records")
        if set(method_rows["fold"].astype(int)) != set(FOLDS):
            raise ValueError(f"{method}: Panel B is missing a frozen fold")
        per_fold = method_rows.groupby("fold", observed=True)["cell_type"].nunique()
        if not per_fold.eq(EXPECTED_CELL_TYPE_COUNT).all():
            raise ValueError(f"{method}: Panel B does not contain all ten cell types per fold")
        result_layers = method_rows["result_layer"].unique()
        if len(result_layers) != 1:
            raise ValueError(f"{method}: mixed result layers are not allowed")
        for column, label, direction in B_METRICS:
            values = pd.to_numeric(method_rows[column], errors="raise").to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(f"{method}: Panel B {column} contains nonfinite values")
            rows.append(
                {
                    "method": method,
                    "result_layer": str(result_layers[0]),
                    "metric_column": column,
                    "metric": label,
                    "metric_direction": direction,
                    "value": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "n_records": len(values),
                }
            )
    return pd.DataFrame(rows)


def panel_c_display_limits(effects: pd.DataFrame) -> tuple[float, float]:
    valid = effects["pair_valid"].astype(bool).to_numpy()
    pooled = np.concatenate(
        [
            effects.loc[valid, "true_cell_type_effect"].to_numpy(dtype=np.float64),
            effects.loc[valid, "predicted_cell_type_effect"].to_numpy(dtype=np.float64),
        ]
    )
    low, high = np.quantile(pooled, [0.005, 0.995])
    span = float(high - low)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("Panel C requires a nonzero finite effect range")
    padding = 0.05 * span
    return float(low - padding), float(high + padding)


def prepare_panel_c(
    effects: pd.DataFrame,
    *,
    sample_size: int = PANEL_C_SAMPLE_SIZE,
    seed: int = PANEL_C_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_size < 1:
        raise ValueError("Panel C sample size must be positive")
    output = effects.copy()
    output["plotted"] = False
    display_min, display_max = panel_c_display_limits(output)
    summaries: list[dict[str, Any]] = []

    for method_index, method in enumerate(METHODS):
        method_mask = output["method"].eq(method).to_numpy()
        valid_mask = method_mask & output["pair_valid"].astype(bool).to_numpy()
        valid_indices = output.index[valid_mask].to_numpy(dtype=np.int64)
        if len(valid_indices) < sample_size:
            raise ValueError(
                f"{method}: only {len(valid_indices)} finite effect pairs for a {sample_size}-point display"
            )
        rng = np.random.default_rng(np.random.SeedSequence([seed, method_index]))
        chosen = rng.choice(valid_indices, size=sample_size, replace=False)
        output.loc[chosen, "plotted"] = True

        true_values = output.loc[valid_indices, "true_cell_type_effect"].to_numpy(
            dtype=np.float64
        )
        predicted_values = output.loc[
            valid_indices, "predicted_cell_type_effect"
        ].to_numpy(dtype=np.float64)
        rho = float(spearmanr(true_values, predicted_values).statistic)
        mae = float(np.mean(np.abs(true_values - predicted_values)))
        if not math.isfinite(rho) or not math.isfinite(mae):
            raise ValueError(f"{method}: Panel C full-pair metrics must be finite")
        method_rows = output.loc[method_mask]
        layers = method_rows["result_layer"].unique()
        if len(layers) != 1:
            raise ValueError(f"{method}: mixed Panel C result layers are not allowed")
        summaries.append(
            {
                "method": method,
                "result_layer": str(layers[0]),
                "effect_spearman": rho,
                "effect_mae": mae,
                "n_effect_pairs": int(method_mask.sum()),
                "n_valid_effect_pairs": len(valid_indices),
                "n_plotted": int(output.loc[method_mask, "plotted"].sum()),
                "panel_c_seed": seed,
                "display_effect_min": display_min,
                "display_effect_max": display_max,
            }
        )
    summary = pd.DataFrame(summaries)
    if summary["n_plotted"].nunique() != 1 or int(summary["n_plotted"].iloc[0]) != sample_size:
        raise ValueError("Panel C must plot an equal number of points for every method")
    return output, summary


def load_panel_d(
    summary_path: Path,
    run_manifest_path: Path,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    manifest = read_json(run_manifest_path)
    if manifest.get("protocol_id") != D_PROTOCOL_ID:
        raise ValueError("Panel D run manifest is not the audited all154 protocol")
    output_info = manifest.get("outputs", {}).get(summary_path.name, {})
    if output_info.get("sha256") != sha256_file(summary_path):
        raise ValueError("Panel D summary hash differs from its audit manifest")
    checks = manifest.get("checks", {})
    required_checks = (
        "all_metrics_finite",
        "common_configuration_all_methods",
        "all_154_genes_enter_preprocessing",
        "measured_train_and_validation_columns_retained",
        "only_test_columns_replaced_by_predictions",
    )
    if any(checks.get(key) is not True for key in required_checks):
        raise ValueError("Panel D audit manifest does not certify the fixed all154 analysis")

    frame = pd.read_csv(summary_path)
    required_columns = {
        "dataset",
        "dataset_id",
        "method",
        "prediction_result_layer",
        "matrix_mode",
        "folds",
        *D_METRICS,
        *(f"{metric}_sd" for metric in D_METRICS),
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Panel D summary is missing columns: {sorted(missing)}")
    if len(frame) != len(METHODS) or set(frame["method"]) != set(METHODS):
        raise ValueError("Panel D requires exactly one audited row for each formal method")
    expected_layers = {"GeneSPT": "validation_selected_readout_genespt57"}
    expected_layers.update({method: "raw_identity" for method in BASELINES})
    for method in METHODS:
        row = frame[frame["method"].eq(method)].iloc[0]
        if (
            row["dataset"] != D_DATASET
            or row["dataset_id"] != D_DATASET_ID
            or row["matrix_mode"] != D_MATRIX_MODE
            or int(row["folds"]) != len(FOLDS)
            or row["prediction_result_layer"] != expected_layers[method]
        ):
            raise ValueError(f"{method}: Panel D summary identity or result layer differs")
        values = np.asarray(
            [row[metric] for metric in D_METRICS]
            + [row[f"{metric}_sd"] for metric in D_METRICS],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values[len(D_METRICS) :] < 0.0):
            raise ValueError(f"{method}: Panel D mean/SD values are invalid")

    order = pd.Categorical(frame["method"], categories=METHODS, ordered=True)
    frame = frame.assign(_method_order=order).sort_values("_method_order").drop(
        columns="_method_order"
    )
    if int(protocol["pca"]["n_components"]) != 30:
        raise ValueError("Panel D protocol drifted after initial validation")
    return frame.reset_index(drop=True)


def assert_public_export(frame: pd.DataFrame) -> None:
    methods = set(frame["method"].dropna().astype(str))
    unexpected = methods.difference(METHODS)
    if unexpected:
        raise ValueError(f"Unexpected method names in Figure 6 source: {sorted(unexpected)}")
    internal_patterns = (
        r"[A-Za-z]:[\\/]",
        r"(?:^|\s)/(?:workspace|home|mnt|root)/",
    )
    text_columns = frame.select_dtypes(include=["object", "string"])
    for column in text_columns.columns:
        values = text_columns[column].fillna("").astype(str)
        for pattern in internal_patterns:
            if values.str.contains(pattern, case=False, regex=True).any():
                raise ValueError("Figure 6 public source contains an internal absolute path")
    artifact_columns = [column for column in frame.columns if column.endswith("artifact_id")]
    for column in artifact_columns:
        values = frame[column].dropna().astype(str)
        populated = values[values.ne("")]
        if not populated.str.match(r"^(repository|archive)/").all():
            raise ValueError(f"{column} contains a non-package-relative artifact ID")


def build_public_source(
    *,
    formal_sources: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
    panel_b_detail_id: str,
    panel_c_detail_id: str,
    panel_d_source_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    workflow = (
        "Measured train/validation genes",
        "Predicted held-out test genes",
        "Completed fold matrix",
        "Cell-type effect recovery",
        "Clustering agreement",
    )
    for order, step in enumerate(workflow, start=1):
        rows.append(
            {
                "panel": "A",
                "row_type": "workflow_step",
                "workflow_order": order,
                "workflow_step": step,
            }
        )

    rows.extend(formal_sources.to_dict(orient="records"))
    for row in panel_b.to_dict(orient="records"):
        rows.append(
            {
                "panel": "B",
                "row_type": "fold_cell_type_metric_summary",
                "dataset": DATASET,
                "dataset_id": DATASET_ID,
                "method": row["method"],
                "result_layer": row["result_layer"],
                "metric": row["metric"],
                "metric_direction": row["metric_direction"],
                "value": row["value"],
                "sd": row["sd"],
                "n_records": row["n_records"],
                "effect_formula": EFFECT_FORMULA,
                "variance_ddof": VARIANCE_DDOF,
                "detail_artifact_id": panel_b_detail_id,
            }
        )
    for row in panel_c.to_dict(orient="records"):
        for column, label, direction in (
            ("effect_spearman", "Effect Spearman", "higher"),
            ("effect_mae", "Effect MAE", "lower"),
        ):
            rows.append(
                {
                    "panel": "C",
                    "row_type": "all_effect_pairs_metric",
                    "dataset": DATASET,
                    "dataset_id": DATASET_ID,
                    "method": row["method"],
                    "result_layer": row["result_layer"],
                    "metric": label,
                    "metric_direction": direction,
                    "value": row[column],
                    "n_effect_pairs": row["n_effect_pairs"],
                    "n_valid_effect_pairs": row["n_valid_effect_pairs"],
                    "n_plotted": row["n_plotted"],
                    "panel_c_seed": row["panel_c_seed"],
                    "display_effect_min": row["display_effect_min"],
                    "display_effect_max": row["display_effect_max"],
                    "effect_formula": EFFECT_FORMULA,
                    "variance_ddof": VARIANCE_DDOF,
                    "detail_artifact_id": panel_c_detail_id,
                }
            )
    configuration = "all154; standardized PCA30; k15 Euclidean; seed0 weighted Louvain"
    for row in panel_d.to_dict(orient="records"):
        for metric in D_METRICS:
            rows.append(
                {
                    "panel": "D",
                    "row_type": "audited_five_fold_metric",
                    "dataset": D_DATASET,
                    "dataset_id": D_DATASET_ID,
                    "method": row["method"],
                    "result_layer": row["prediction_result_layer"],
                    "metric": metric,
                    "metric_direction": "higher",
                    "value": row[metric],
                    "sd": row[f"{metric}_sd"],
                    "folds": row["folds"],
                    "protocol_id": D_PROTOCOL_ID,
                    "configuration": configuration,
                    "source_artifact_id": panel_d_source_id,
                }
            )

    source = pd.DataFrame(rows)
    source["source_schema_version"] = SOURCE_SCHEMA
    for column in SOURCE_COLUMNS:
        if column not in source.columns:
            source[column] = pd.NA
    source = source[list(SOURCE_COLUMNS)]
    assert_public_export(source)
    return source


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.065) -> None:
    ax.text(
        x,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_a(ax: plt.Axes) -> None:
    ax.axis("off")
    panel_label(ax, "A")
    ax.set_title("Downstream workflow", fontsize=9.2, fontweight="bold", pad=5)

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.85,
            edgecolor="#b8b8b8",
            facecolor=color,
            transform=ax.transAxes,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.9,
            linespacing=1.05,
        )

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.85,
                color="#777777",
                transform=ax.transAxes,
                shrinkA=3,
                shrinkB=4,
                zorder=3,
            )
        )

    box(0.02, 0.65, 0.31, 0.18, "Measured train +\nvalidation genes", "#f1f1f1")
    box(0.02, 0.28, 0.31, 0.18, "Predicted held-out\ntest genes", "#ffe9e7")
    box(0.38, 0.47, 0.24, 0.19, "Completed\nfold matrix", "#fff7f6")
    # Keep the rounded-box padding inside the axes so the right border is not clipped.
    box(0.68, 0.63, 0.29, 0.22, "Cell-type\neffects\n(Panels B, C)", "#f5f5f5")
    box(0.68, 0.26, 0.29, 0.22, "Clustering\nagreement\n(Panel D)", "#f5f5f5")
    arrow((0.33, 0.74), (0.38, 0.59))
    arrow((0.33, 0.37), (0.38, 0.54))
    arrow((0.62, 0.58), (0.68, 0.70))
    arrow((0.62, 0.54), (0.68, 0.40))


def draw_panel_b(fig: plt.Figure, spec: Any, summary: pd.DataFrame) -> None:
    outer = fig.add_subplot(spec)
    outer.axis("off")
    panel_label(outer, "B")
    inner = spec.subgridspec(2, 2, height_ratios=[0.30, 1.0], hspace=0.30, wspace=0.38)
    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.82,
        "seqFISH+ positive-effect marker recovery",
        fontsize=9.2,
        fontweight="bold",
        ha="left",
        va="center",
    )
    title_ax.text(
        0.0,
        0.20,
        r"Mean $\pm$ sample SD across 50 fold-cell-type records",
        fontsize=5.9,
        color="#555555",
        ha="left",
        va="center",
    )

    y = np.arange(len(METHODS))
    titles = {
        "top20_overlap_count": "Top-20 positive\noverlap (higher)",
        "top50_overlap_count": "Top-50 positive\noverlap (higher)",
    }
    for index, (metric_column, _label, direction) in enumerate(B_DISPLAY_METRICS):
        ax = fig.add_subplot(inner[1, index])
        metric_rows = summary[summary["metric_column"].eq(metric_column)].set_index("method")
        all_bounds: list[float] = []
        for yi, method in enumerate(METHODS):
            row = metric_rows.loc[method]
            value = float(row["value"])
            sd = float(row["sd"])
            all_bounds.extend([value - sd, value + sd])
            ax.errorbar(
                value,
                yi,
                xerr=sd,
                fmt="o",
                markersize=4.2,
                color=COLORS[method],
                markeredgecolor=COLORS[method],
                markeredgewidth=0.0,
                elinewidth=0.75,
                capsize=1.8,
                capthick=0.7,
                zorder=4,
            )
        best_method = (
            metric_rows["value"].astype(float).idxmax()
            if direction == "higher"
            else metric_rows["value"].astype(float).idxmin()
        )
        best_value = float(metric_rows.loc[best_method, "value"])
        ax.scatter(
            [best_value],
            [METHODS.index(best_method)],
            s=68,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.0,
            zorder=5,
        )
        low, high = min(all_bounds), max(all_bounds)
        span = high - low
        padding = 0.09 * span if span > 0.0 else max(abs(high), 1.0) * 0.1
        ax.set_xlim(low - padding, high + padding)
        ax.set_ylim(len(METHODS) - 0.45, -0.55)
        ax.set_yticks(y)
        ax.set_yticklabels(METHODS if index == 0 else [])
        ax.set_title(titles[metric_column], fontsize=6.4, pad=2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis="both", labelsize=5.7, length=2.2)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        clean_axes(ax)


def draw_panel_c(
    fig: plt.Figure,
    spec: Any,
    effects: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    outer = fig.add_subplot(spec)
    outer.axis("off")
    panel_label(outer, "C", x=-0.10)
    outer.text(
        0.08,
        1.045,
        "Held-out effect pairs across all seven methods",
        fontsize=9.2,
        fontweight="bold",
        ha="left",
        va="top",
        transform=outer.transAxes,
        clip_on=False,
    )
    inner = spec.subgridspec(2, 4, hspace=0.44, wspace=0.34)
    indexed_summary = summary.set_index("method")
    display_min = float(summary["display_effect_min"].iloc[0])
    display_max = float(summary["display_effect_max"].iloc[0])
    display_range = display_max - display_min
    display_pad = max(0.025 * display_range, np.finfo(float).eps)
    axis_min = display_min - display_pad
    axis_max = display_max + display_pad

    for index, method in enumerate(METHODS):
        ax = fig.add_subplot(inner[index // 4, index % 4])
        plotted = effects[
            effects["method"].eq(method)
            & effects["pair_valid"].astype(bool)
            & effects["plotted"].astype(bool)
        ]
        plotted = plotted[
            plotted["true_cell_type_effect"].between(display_min, display_max)
            & plotted["predicted_cell_type_effect"].between(display_min, display_max)
        ]
        ax.scatter(
            plotted["true_cell_type_effect"],
            plotted["predicted_cell_type_effect"],
            s=3.1,
            alpha=0.16,
            color=COLORS[method],
            linewidth=0,
            rasterized=True,
        )
        ax.plot(
            [display_min, display_max],
            [display_min, display_max],
            linestyle="--",
            color="#888888",
            linewidth=0.75,
            zorder=2,
        )
        row = indexed_summary.loc[method]
        ax.text(
            0.04,
            0.96,
            rf"$\rho$={float(row['effect_spearman']):.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
            color="#333333",
        )
        ax.set_title(method, fontsize=7.2, pad=2)
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_box_aspect(1.0)
        if index // 4 == 1:
            ax.set_xlabel("True effect", fontsize=6.0)
        else:
            ax.set_xticklabels([])
        if index % 4 == 0:
            ax.set_ylabel("Predicted effect", fontsize=6.0)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", labelsize=5.2, length=2.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.grid(color=GRID_COLOR, linewidth=0.5)
        clean_axes(ax)


def draw_panel_d(ax: plt.Axes, panel_d: pd.DataFrame) -> None:
    panel_label(ax, "D")
    ax.set_title("MHPR/MERFISH clustering agreement", fontsize=9.2, fontweight="bold", pad=7)
    indexed = panel_d.set_index("method")
    metric_labels = ("ARI", "AMI", "NMI", "Homogeneity")
    x = np.arange(len(D_METRICS))
    width = 0.105
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width
    upper: list[float] = []
    for method_index, method in enumerate(METHODS):
        row = indexed.loc[method]
        means = np.asarray([row[metric] for metric in D_METRICS], dtype=np.float64)
        errors = np.asarray([row[f"{metric}_sd"] for metric in D_METRICS], dtype=np.float64)
        upper.extend((means + errors).tolist())
        ax.bar(
            x + offsets[method_index],
            means,
            yerr=errors,
            width=width * 0.86,
            color=COLORS[method],
            edgecolor="#222222" if method == "GeneSPT" else "none",
            linewidth=0.45 if method == "GeneSPT" else 0.0,
            capsize=1.7,
            error_kw={"elinewidth": 0.55, "capthick": 0.55, "ecolor": "#444444"},
            zorder=3,
        )
    ax.set_ylim(0.0, max(0.69, max(upper) + 0.03))
    ax.set_xlim(-0.58, len(D_METRICS) - 0.42)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Clustering agreement")
    ax.text(
        0.015,
        0.965,
        "all 154 genes | PCA 30 | k=15 Euclidean\nseed 0 weighted Louvain",
        transform=ax.transAxes,
        fontsize=5.7,
        color="#555555",
        ha="left",
        va="top",
    )
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.65)
    ax.grid(axis="x", visible=False)
    clean_axes(ax)


def plot_figure(
    panel_b: pd.DataFrame,
    panel_c_effects: pd.DataFrame,
    panel_c_summary: pd.DataFrame,
    panel_d: pd.DataFrame,
    *,
    pdf_path: Path,
    png_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.1,
            "axes.titlesize": 8.8,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(9.0, 6.6), facecolor="white", constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[0.78, 1.48],
        width_ratios=[0.94, 1.34],
        left=0.065,
        right=0.975,
        bottom=0.075,
        top=0.90,
        hspace=0.37,
        wspace=0.28,
    )
    draw_panel_a(figure.add_subplot(grid[0, 0]))
    draw_panel_b(figure, grid[0, 1], panel_b)
    draw_panel_c(figure, grid[1, 0], panel_c_effects, panel_c_summary)
    draw_panel_d(figure.add_subplot(grid[1, 1]), panel_d)

    method_handles = [Patch(color=COLORS[method], label=method) for method in METHODS]
    best_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="#111111",
        markerfacecolor="none",
        linestyle="none",
        markersize=6.4,
        label="Best mean",
    )
    figure.legend(
        handles=method_handles + [best_handle],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.972),
        ncol=8,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.25,
    )
    figure.savefig(
        pdf_path,
        metadata={
            "Title": "Protocol A Figure 6",
            "Creator": "generate_protocol_a_figure6.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(png_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def caption_text(panel_c_sample_size: int, panel_c_seed: int) -> str:
    return (
        "Figure 6. Downstream recovery using predicted held-out genes. "
        "A, Measured train/validation genes and method-predicted held-out test genes "
        "form completed fold matrices for downstream evaluation. "
        "B, Positive-effect marker recovery in seqFISH+ cortex/SVZ across five frozen gene "
        "folds and ten matched cell types (913 of 913 cells labeled). For each test gene, "
        f"the effect is {EFFECT_FORMULA}, with population variances. Points and error "
        "bars show the mean +/- sample SD across 50 fold-cell-type records for intersections "
        "between up to 20 or 50 genes with the largest strictly positive true and predicted effects. "
        "C, True versus predicted test-gene effects for all seven methods. Each small panel "
        f"shows an equal fixed-seed sample of {panel_c_sample_size:,} finite pairs "
        f"(seed {panel_c_seed}); rho is calculated from every finite effect pair, "
        "while undefined zero-variance predicted effects remain recorded in the source "
        "table together with the complete record-level and pooled MAE results. All panels "
        "use a display window based on the pooled 0.5th and 99.5th "
        "effect percentiles. "
        "D, Audited five-fold MHPR/MERFISH clustering agreement using all 154 genes, "
        "standardized PCA with 30 components, a k=15 Euclidean graph, and seed-0 weighted "
        "Louvain for every method. Bars show mean +/- sample SD across the five frozen "
        "gene folds; author labels are used only after clustering to calculate ARI, AMI, "
        "NMI, and homogeneity.\n"
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        na_rep="NA",
        float_format="%.17g",
        lineterminator="\n",
    )


def generate_figure6(
    *,
    project_root: Path = PROJECT_ROOT,
    archive_root: Path = ARCHIVE_ROOT,
    results_root: Path = RESULTS_ROOT,
    output_dir: Path = OUTPUT_DIR,
    panel_c_sample_size: int = PANEL_C_SAMPLE_SIZE,
    panel_c_seed: int = PANEL_C_SEED,
    dpi: int = 480,
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    archive_root = archive_root.resolve(strict=True)
    results_root = results_root.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    try:
        output_dir.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Figure 6 output must remain inside the repository package") from exc
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_path = project_root / "configs" / "protocol_a_figure6d_mhpr_all154_louvain.yaml"
    protocol = load_fixed_protocol(protocol_path)
    label_path = (
        archive_root
        / "label_provenance"
        / "matched_labels"
        / "seqfish_plus_cortex_svz_matched_cell_types.csv"
    )
    locations_path = (
        archive_root
        / "processed_datasets"
        / "cross_platform"
        / DATASET_ID
        / "Locations.txt"
    )
    labels = load_labels(label_path, locations_path)
    input_hashes = [
        hash_row(
            protocol_path,
            scope="input",
            role="fixed_panel_d_protocol",
            project_root=project_root,
            archive_root=archive_root,
        ),
        hash_row(
            label_path,
            scope="input",
            role="matched_cell_type_labels",
            project_root=project_root,
            archive_root=archive_root,
        ),
        hash_row(
            locations_path,
            scope="input",
            role="label_row_alignment_locations",
            project_root=project_root,
            archive_root=archive_root,
        ),
    ]

    records, effects, formal_sources = compute_effect_tables(
        project_root=project_root,
        archive_root=archive_root,
        results_root=results_root,
        protocol=protocol,
        labels=labels,
        input_hashes=input_hashes,
    )
    panel_b = summarize_panel_b(records)
    effects, panel_c = prepare_panel_c(
        effects,
        sample_size=panel_c_sample_size,
        seed=panel_c_seed,
    )

    panel_d_root = results_root / "downstream" / "figure6d_protocol_a_all154"
    panel_d_summary_path = panel_d_root / "protocol_a_figure6d_mhpr_all154_summary.csv"
    panel_d_manifest_path = panel_d_root / "protocol_a_figure6d_mhpr_all154_run_manifest.json"
    panel_d = load_panel_d(panel_d_summary_path, panel_d_manifest_path, protocol)
    input_hashes.extend(
        [
            hash_row(
                panel_d_summary_path,
                scope="input",
                role="audited_panel_d_summary",
                project_root=project_root,
                archive_root=archive_root,
            ),
            hash_row(
                panel_d_manifest_path,
                scope="input",
                role="panel_d_audit_manifest",
                project_root=project_root,
                archive_root=archive_root,
            ),
        ]
    )

    paths = {key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()}
    panel_b_detail_id = package_artifact_id(
        paths["panel_b_records"], project_root, archive_root
    )
    panel_c_detail_id = package_artifact_id(
        paths["panel_c_effects"], project_root, archive_root
    )
    panel_d_source_id = package_artifact_id(
        panel_d_summary_path, project_root, archive_root
    )

    source = build_public_source(
        formal_sources=formal_sources,
        panel_b=panel_b,
        panel_c=panel_c,
        panel_d=panel_d,
        panel_b_detail_id=panel_b_detail_id,
        panel_c_detail_id=panel_c_detail_id,
        panel_d_source_id=panel_d_source_id,
    )
    assert_public_export(records)
    assert_public_export(effects)
    write_csv(records, paths["panel_b_records"])
    write_csv(effects, paths["panel_c_effects"])
    write_csv(source, paths["source"])
    paths["caption"].write_text(
        caption_text(panel_c_sample_size, panel_c_seed), encoding="utf-8", newline="\n"
    )
    plot_figure(
        panel_b,
        effects,
        panel_c,
        panel_d,
        pdf_path=paths["pdf"],
        png_path=paths["png"],
        dpi=dpi,
    )

    output_hashes = [
        hash_row(
            paths[key],
            scope="output",
            role=key,
            project_root=project_root,
            archive_root=archive_root,
        )
        for key in (
            "pdf",
            "png",
            "source",
            "panel_b_records",
            "panel_c_effects",
            "caption",
        )
    ]
    manifest = pd.DataFrame(input_hashes + output_hashes)
    manifest = manifest.drop_duplicates(subset=["artifact_scope", "artifact_id"]).sort_values(
        ["artifact_scope", "artifact_role", "method", "fold", "artifact_id"],
        kind="stable",
    )
    write_csv(manifest, paths["hash_manifest"])

    return {
        "output_dir": package_artifact_id(output_dir, project_root, archive_root),
        "panel_b_records": len(records),
        "panel_c_effect_rows": len(effects),
        "panel_c_plotted_per_method": panel_c_sample_size,
        "panel_d_methods": len(panel_d),
        "outputs": {key: package_artifact_id(path, project_root, archive_root) for key, path in paths.items()},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--panel-c-sample-size", type=int, default=PANEL_C_SAMPLE_SIZE)
    parser.add_argument("--panel-c-seed", type=int, default=PANEL_C_SEED)
    parser.add_argument("--dpi", type=int, default=480)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate_figure6(
        project_root=args.project_root,
        archive_root=args.archive_root,
        results_root=args.results_root,
        output_dir=args.output_dir,
        panel_c_sample_size=args.panel_c_sample_size,
        panel_c_seed=args.panel_c_seed,
        dpi=args.dpi,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
