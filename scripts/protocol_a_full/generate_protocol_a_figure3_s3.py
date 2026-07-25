#!/usr/bin/env python3
"""Generate Protocol A Figure 3 and Supplementary Table S3.

Panels A and C use the completed centralized mechanism-control evidence. Panel
B is a matched identity-readout comparison of GeneSPT-GC and GeneSPT, with the
fold-level records recovered from the internal audit report and the five-fold
means checked against its designated summary CSV.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "protocol_a_full_rerun_20260711"

PANEL_A_ROOT = RESULTS_ROOT / "mechanism" / "figure3_a_descriptor_controls"
PANEL_C_ROOT = (
    RESULTS_ROOT / "mechanism" / "figure3_c_primary_mechanism_controls"
)
RAW_IDENTITY_ROOT = RESULTS_ROOT / "evaluation" / "raw_identity"
RAW_IDENTITY_FIVE_FOLD = RAW_IDENTITY_ROOT / "raw_identity_five_fold_metrics.csv"
RAW_IDENTITY_OVERVIEW = RAW_IDENTITY_ROOT / "raw_identity_overview.json"
RAW_EVALUATION_REPORT = (
    RESULTS_ROOT / "evaluation" / "protocol_a_raw_evaluation_report.json"
)

FIGURE_DIR = RESULTS_ROOT / "figures" / "figure3"
S3_DIR = RESULTS_ROOT / "supplementary" / "S3"

FIGURE_STEM = "figure3_protocol_a_mechanism_controls"
FIGURE_SOURCE_NAME = f"{FIGURE_STEM}_source.csv"
FIGURE_PNG_NAME = f"{FIGURE_STEM}.png"
FIGURE_PDF_NAME = f"{FIGURE_STEM}.pdf"
FIGURE_CAPTION_NAME = f"{FIGURE_STEM}_caption_draft.md"

S3_FOLD_LEVEL_NAME = "supplementary_table_s3_fold_level.csv"
S3_SUMMARY_NAME = "supplementary_table_s3_five_fold_summary.csv"
S3_B_IMPROVEMENT_NAME = "supplementary_table_s3b_improvements.csv"
S3_README_NAME = "README.md"
S3_MANIFEST_NAME = "manifest.json"

FOLDS = (0, 1, 2, 3, 4)
METRICS = ("SPCC", "RMSE", "JSD", "SSIM")
VIS9A_ID = "Vis9A_D7_spaim_effective4470"
PANEL_B_RESULT_LAYER = "matched_identity_readout"
PANEL_B_SOURCE_KIND = "matched_identity_readout_psp_pair"

PANEL_A_SETTINGS = (
    ("correct", "Correct", "mlp_pca32_softplus_correct"),
    ("random", "Random", "mlp_pca32_softplus_random"),
    ("shuffled", "Shuffled", "mlp_pca32_softplus_shuffled"),
    (
        "permuted_labels",
        "Permuted",
        "mlp_pca32_softplus_permuted_labels",
    ),
)
PANEL_B_DATASETS = (
    ("Vis9A", VIS9A_ID, "Vis9A"),
    ("HBC", "HBC_shared16112", "HBC"),
    (
        "Cell2location",
        "Cell2location_mouse_brain_ST8059048_shared12819",
        "Cell2location",
    ),
)
PANEL_B_METHODS = (
    ("GeneSPT-GC", "base", "gc_mlp_base"),
    (
        "GeneSPT",
        "correct",
        "predictable_spatial_program_selected_correct",
    ),
)
PANEL_C_SETTINGS = (
    ("correct", "Correct PSP", "predictable_spatial_program_selected_correct"),
    ("base", "GC-only", "gc_mlp_base"),
    (
        "shuffled_descriptor",
        "Shuffled descriptor",
        "predictable_spatial_program_shuffled_descriptor_control",
    ),
    (
        "random_descriptor",
        "Random descriptor",
        "predictable_spatial_program_random_descriptor_control",
    ),
    (
        "permuted_labels",
        "Permuted labels",
        "predictable_spatial_program_permuted_labels_control",
    ),
    (
        "random_spatial_basis",
        "Random spatial basis",
        "predictable_spatial_program_random_spatial_basis_control",
    ),
    (
        "spot_permuted_spatial_program",
        "Spot-permuted program",
        "predictable_spatial_program_spot_permuted_spatial_program_control",
    ),
    (
        "mean_coefficient",
        "Mean coefficient",
        "predictable_spatial_program_mean_coefficient_baseline_control",
    ),
)

RED = "#D4362E"
DARK = "#40464D"
GREY = "#A9B0B8"
BLUE_GREY = "#72879A"
GRID = "#E5E8EC"
ZERO = "#A7AFB8"
LINE = "#E2AAA5"

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

PUBLIC_PROVENANCE_COLUMNS = (
    "source_metrics_path",
    "source_metrics_sha256",
    "source_index_path",
    "source_index_sha256",
    "source_manifest_path",
    "source_manifest_sha256",
    "prediction_path",
    "prediction_sha256",
    "reference_five_fold_path",
    "reference_five_fold_sha256",
)
PUBLIC_SOURCE_KIND_REPLACEMENTS = {
    "audited_protocol_a_benchmark_gc": "matched_gene_conditioned_baseline",
}
PUBLIC_S3_FORBIDDEN_PATTERNS = {
    "raw identity label": re.compile(r"raw[\s_-]*identity", re.IGNORECASE),
    "internal protocol label": re.compile(r"protocol[\s_-]*a", re.IGNORECASE),
    "Windows absolute path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"),
    "UNC path": re.compile(r"\\\\[^\\\s]+[\\/]"),
    "Unix internal absolute path": re.compile(
        r"(?<![A-Za-z0-9_])/(?:workspace|home|mnt|tmp)/", re.IGNORECASE
    ),
    "internal evaluation path": re.compile(r"evaluation[\\/]", re.IGNORECASE),
}


class EvidenceError(RuntimeError):
    """Raised when formal Figure 3 evidence violates its frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"Required JSON is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"Invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise EvidenceError(f"JSON root must be an object: {path}")
    return payload


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise EvidenceError(f"{label} is missing columns: {sorted(missing)}")


def project_relative(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(project_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise EvidenceError(f"Path is outside the project root: {path}") from error


def _relative_reported_path(raw_path: str, project_root: Path) -> PurePosixPath:
    text = str(raw_path).replace("\\", "/")
    workspace_prefix = "/workspace/GeneSPT/"
    if text.startswith(workspace_prefix):
        text = text[len(workspace_prefix) :]
    elif text == "/workspace/GeneSPT":
        text = ""
    elif text.startswith("GeneSPT/"):
        text = text[len("GeneSPT/") :]
    else:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            try:
                return PurePosixPath(
                    candidate.resolve(strict=False)
                    .relative_to(project_root.resolve(strict=True))
                    .as_posix()
                )
            except ValueError as error:
                raise EvidenceError(
                    f"Reported path is outside the project root: {raw_path}"
                ) from error
    relative = PurePosixPath(text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise EvidenceError(f"Unsafe reported path: {raw_path}")
    return relative


def resolve_reported_path(raw_path: str, project_root: Path = PROJECT_ROOT) -> Path:
    relative = _relative_reported_path(raw_path, project_root)
    return project_root.joinpath(*relative.parts)


def portable_reported_path(raw_path: str, project_root: Path = PROJECT_ROOT) -> str:
    return _relative_reported_path(raw_path, project_root).as_posix()


def validate_sha256(value: object, context: str) -> str:
    result = str(value)
    if not SHA256_PATTERN.fullmatch(result):
        raise EvidenceError(f"{context} has no valid SHA256")
    return result


def verify_file_record(
    record: Mapping[str, Any],
    *,
    context: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    if not isinstance(record, Mapping):
        raise EvidenceError(f"{context} file record is missing")
    path = resolve_reported_path(str(record.get("path", "")), project_root)
    if not path.is_file():
        raise EvidenceError(f"{context} is missing: {path}")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise EvidenceError(f"{context} byte count mismatch")
    expected = validate_sha256(record.get("sha256"), context)
    observed = sha256_file(path)
    if observed != expected:
        raise EvidenceError(f"{context} SHA256 mismatch: {observed} != {expected}")
    return path


def _output_record_for(
    manifest: Mapping[str, Any],
    expected_path: Path,
    *,
    context: str,
    project_root: Path,
) -> Mapping[str, Any]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise EvidenceError(f"{context} has no output records")
    matches = []
    for record in outputs:
        if not isinstance(record, Mapping):
            continue
        path = resolve_reported_path(str(record.get("path", "")), project_root)
        if path.resolve(strict=False) == expected_path.resolve(strict=False):
            matches.append(record)
    if len(matches) != 1:
        raise EvidenceError(
            f"{context} must audit exactly one {expected_path.name} record"
        )
    verify_file_record(matches[0], context=context, project_root=project_root)
    return matches[0]


def _load_mechanism_root(
    root: Path,
    *,
    expected_panel: str,
    project_root: Path,
) -> tuple[
    dict[str, Any],
    dict[int, tuple[Path, str, dict[str, Any]]],
    str,
    str,
]:
    root_manifest_path = root / "manifest.json"
    root_manifest = load_json(root_manifest_path)
    if (
        root_manifest.get("status") != "complete"
        or root_manifest.get("panel") != expected_panel
        or root_manifest.get("dataset") != "Vis9A"
        or root_manifest.get("dataset_id") != VIS9A_ID
        or tuple(root_manifest.get("folds", [])) != FOLDS
    ):
        raise EvidenceError(f"{expected_panel} root manifest identity is invalid")

    fold_metrics_path = root / "fold_metrics.csv"
    five_fold_path = root / "five_fold_metrics.csv"
    _output_record_for(
        root_manifest,
        fold_metrics_path,
        context=f"{expected_panel} fold metrics",
        project_root=project_root,
    )
    _output_record_for(
        root_manifest,
        five_fold_path,
        context=f"{expected_panel} five-fold metrics",
        project_root=project_root,
    )

    records = root_manifest.get("fold_manifests")
    if not isinstance(records, list) or len(records) != len(FOLDS):
        raise EvidenceError(f"{expected_panel} must have five fold manifests")
    fold_manifests: dict[int, tuple[Path, str, dict[str, Any]]] = {}
    for record in records:
        path = verify_file_record(
            record,
            context=f"{expected_panel} fold completion manifest",
            project_root=project_root,
        )
        payload = load_json(path)
        identity = payload.get("identity")
        if (
            payload.get("status") != "complete"
            or not isinstance(identity, Mapping)
            or identity.get("panel") != expected_panel
            or identity.get("dataset_id") != VIS9A_ID
        ):
            raise EvidenceError(f"Invalid {expected_panel} completion manifest: {path}")
        fold = int(identity.get("fold", -1))
        if fold not in FOLDS or fold in fold_manifests:
            raise EvidenceError(f"Invalid or duplicate {expected_panel} fold: {fold}")
        fold_manifests[fold] = (
            path,
            validate_sha256(record.get("sha256"), str(path)),
            payload,
        )
    if set(fold_manifests) != set(FOLDS):
        raise EvidenceError(f"{expected_panel} fold-manifest coverage is incomplete")
    return (
        root_manifest,
        fold_manifests,
        sha256_file(root_manifest_path),
        sha256_file(fold_metrics_path),
    )


def _numeric_metrics(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in (*METRICS, "coverage"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise EvidenceError(f"{label} contains non-finite {column} values")
    for column in ("fold", "eligible_genes", "scored_genes", "constant_prediction_genes"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(int)
    return result


def _validate_mechanism_frame(
    frame: pd.DataFrame,
    *,
    settings: Sequence[tuple[str, str, str]],
    label: str,
) -> pd.DataFrame:
    require_columns(
        frame,
        {
            "dataset",
            "dataset_id",
            "fold",
            "model",
            "control",
            "source_kind",
            "SPCC",
            "RMSE",
            "JSD",
            "JS",
            "SSIM",
            "coverage",
            "eligible_genes",
            "scored_genes",
            "constant_prediction_genes",
        },
        label,
    )
    result = _numeric_metrics(frame, label)
    expected = {
        (fold, control, model)
        for fold in FOLDS
        for control, _setting, model in settings
    }
    observed = set(zip(result["fold"], result["control"], result["model"]))
    if len(result) != len(expected) or observed != expected:
        raise EvidenceError(
            f"{label} row contract differs: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    if result.duplicated(["fold", "control"]).any():
        raise EvidenceError(f"{label} contains duplicate fold-control rows")
    if not result["dataset"].eq("Vis9A").all() or not result["dataset_id"].eq(
        VIS9A_ID
    ).all():
        raise EvidenceError(f"{label} dataset identity changed")
    if not np.allclose(result["coverage"], 1.0, rtol=0.0, atol=1e-12):
        raise EvidenceError(f"{label} does not have complete coverage")
    if not (result["eligible_genes"] == result["scored_genes"]).all():
        raise EvidenceError(f"{label} has incomplete eligible-gene scoring")
    if not np.allclose(result["JS"], result["JSD"], rtol=0.0, atol=1e-15):
        raise EvidenceError(f"{label} JS alias differs from JSD")
    return result


def _validate_mechanism_summary(
    fold_frame: pd.DataFrame,
    summary_path: Path,
    *,
    settings: Sequence[tuple[str, str, str]],
    label: str,
) -> None:
    summary = pd.read_csv(summary_path)
    require_columns(
        summary,
        {
            "control",
            "model",
            "folds",
            "coverage",
            *METRICS,
            *(f"{metric}_std_ddof0" for metric in METRICS),
        },
        f"{label} five-fold summary",
    )
    expected_pairs = {(control, model) for control, _setting, model in settings}
    observed_pairs = set(zip(summary["control"], summary["model"]))
    if len(summary) != len(settings) or observed_pairs != expected_pairs:
        raise EvidenceError(f"{label} five-fold setting set changed")
    for control, _setting, model in settings:
        folds = fold_frame[
            fold_frame["control"].eq(control) & fold_frame["model"].eq(model)
        ]
        aggregate = summary[
            summary["control"].eq(control) & summary["model"].eq(model)
        ]
        if len(folds) != 5 or len(aggregate) != 1:
            raise EvidenceError(f"{label} summary multiplicity failed for {control}")
        row = aggregate.iloc[0]
        if int(row["folds"]) != 5 or not np.isclose(
            float(row["coverage"]), 1.0, rtol=0.0, atol=1e-12
        ):
            raise EvidenceError(f"{label} summary coverage failed for {control}")
        for metric in METRICS:
            values = folds[metric].to_numpy(dtype=float)
            if not np.isclose(
                float(row[metric]), values.mean(), rtol=0.0, atol=1e-12
            ) or not np.isclose(
                float(row[f"{metric}_std_ddof0"]),
                values.std(ddof=0),
                rtol=0.0,
                atol=1e-12,
            ):
                raise EvidenceError(
                    f"{label} five-fold {metric} does not reproduce for {control}"
                )


def _prediction_record_metadata(
    record: Mapping[str, Any], *, context: str, project_root: Path
) -> tuple[str, str]:
    if not isinstance(record, Mapping):
        raise EvidenceError(f"{context} prediction record is missing")
    if int(record.get("bytes", 0)) <= 0:
        raise EvidenceError(f"{context} prediction byte count is invalid")
    path = portable_reported_path(str(record.get("path", "")), project_root)
    digest = validate_sha256(record.get("sha256"), f"{context} prediction")
    return path, digest


def _base_fold_row(
    row: pd.Series,
    *,
    panel: str,
    setting: str,
    result_layer: str,
    source_metrics_path: Path,
    source_metrics_sha256: str,
    source_index_path: Path,
    source_index_sha256: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    prediction_path: str,
    prediction_sha256: str,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "panel": panel,
        "dataset": str(row["dataset"]),
        "dataset_id": str(row["dataset_id"]),
        "fold": int(row["fold"]),
        "setting": setting,
        "control": str(row["control"]),
        "model": str(row["model"]),
        "source_kind": str(row["source_kind"]),
        "result_layer": result_layer,
        "readout": "identity",
        "posthoc_calibration": "none",
        "aggregation_unit": "fold_median_across_eligible_genes",
        "SPCC": float(row["SPCC"]),
        "RMSE": float(row["RMSE"]),
        "JSD": float(row["JSD"]),
        "SSIM": float(row["SSIM"]),
        "coverage": float(row["coverage"]),
        "eligible_genes": int(row["eligible_genes"]),
        "scored_genes": int(row["scored_genes"]),
        "constant_prediction_genes": int(row["constant_prediction_genes"]),
        "source_metrics_path": project_relative(source_metrics_path, project_root),
        "source_metrics_sha256": source_metrics_sha256,
        "source_index_path": project_relative(source_index_path, project_root),
        "source_index_sha256": source_index_sha256,
        "source_manifest_path": project_relative(source_manifest_path, project_root),
        "source_manifest_sha256": source_manifest_sha256,
        "prediction_path": prediction_path,
        "prediction_sha256": prediction_sha256,
    }


def load_panel_a_fold_rows(
    root: Path = PANEL_A_ROOT, *, project_root: Path = PROJECT_ROOT
) -> pd.DataFrame:
    root = Path(root)
    root_manifest, manifests, root_hash, metrics_hash = _load_mechanism_root(
        root, expected_panel="Figure3A", project_root=project_root
    )
    del root_manifest
    metrics_path = root / "fold_metrics.csv"
    frame = _validate_mechanism_frame(
        pd.read_csv(metrics_path), settings=PANEL_A_SETTINGS, label="Figure 3A"
    )
    _validate_mechanism_summary(
        frame,
        root / "five_fold_metrics.csv",
        settings=PANEL_A_SETTINGS,
        label="Figure 3A",
    )

    rows = []
    for control, setting, model in PANEL_A_SETTINGS:
        selected = frame[frame["control"].eq(control)].sort_values("fold")
        for _, row in selected.iterrows():
            fold = int(row["fold"])
            manifest_path, manifest_hash, manifest = manifests[fold]
            identity = manifest["identity"]
            if control == "correct":
                prediction_record = identity.get("correct", {}).get("prediction")
            else:
                results = manifest.get("results")
                if not isinstance(results, list):
                    raise EvidenceError(f"Figure 3A fold{fold} control results are missing")
                matches = [
                    item
                    for item in results
                    if isinstance(item, Mapping)
                    and item.get("control") == control
                    and item.get("model") == model
                ]
                if len(matches) != 1:
                    raise EvidenceError(
                        f"Figure 3A fold{fold} prediction provenance failed for {control}"
                    )
                prediction_record = matches[0].get("prediction")
            prediction_path, prediction_hash = _prediction_record_metadata(
                prediction_record,
                context=f"Figure 3A fold{fold} {control}",
                project_root=project_root,
            )
            rows.append(
                _base_fold_row(
                    row,
                    panel="A",
                    setting=setting,
                    result_layer="figure3_a_descriptor_controls",
                    source_metrics_path=metrics_path,
                    source_metrics_sha256=metrics_hash,
                    source_index_path=root / "manifest.json",
                    source_index_sha256=root_hash,
                    source_manifest_path=manifest_path,
                    source_manifest_sha256=manifest_hash,
                    prediction_path=prediction_path,
                    prediction_sha256=prediction_hash,
                    project_root=project_root,
                )
            )
    return pd.DataFrame(rows)


def load_panel_c_fold_rows(
    root: Path = PANEL_C_ROOT, *, project_root: Path = PROJECT_ROOT
) -> pd.DataFrame:
    root = Path(root)
    root_manifest, manifests, root_hash, metrics_hash = _load_mechanism_root(
        root, expected_panel="Figure3C", project_root=project_root
    )
    del root_manifest
    metrics_path = root / "fold_metrics.csv"
    frame = _validate_mechanism_frame(
        pd.read_csv(metrics_path), settings=PANEL_C_SETTINGS, label="Figure 3C"
    )
    _validate_mechanism_summary(
        frame,
        root / "five_fold_metrics.csv",
        settings=PANEL_C_SETTINGS,
        label="Figure 3C",
    )

    rows = []
    for control, setting, model in PANEL_C_SETTINGS:
        selected = frame[frame["control"].eq(control)].sort_values("fold")
        for _, row in selected.iterrows():
            fold = int(row["fold"])
            manifest_path, manifest_hash, manifest = manifests[fold]
            sources = manifest["identity"].get("prediction_sources")
            if not isinstance(sources, list):
                raise EvidenceError(f"Figure 3C fold{fold} prediction sources are missing")
            matches = [
                item
                for item in sources
                if isinstance(item, Mapping)
                and item.get("control") == control
                and item.get("model") == model
            ]
            if len(matches) != 1:
                raise EvidenceError(
                    f"Figure 3C fold{fold} prediction provenance failed for {control}"
                )
            prediction_path, prediction_hash = _prediction_record_metadata(
                matches[0],
                context=f"Figure 3C fold{fold} {control}",
                project_root=project_root,
            )
            rows.append(
                _base_fold_row(
                    row,
                    panel="C",
                    setting=setting,
                    result_layer="figure3_c_primary_mechanism_controls",
                    source_metrics_path=metrics_path,
                    source_metrics_sha256=metrics_hash,
                    source_index_path=root / "manifest.json",
                    source_index_sha256=root_hash,
                    source_manifest_path=manifest_path,
                    source_manifest_sha256=manifest_hash,
                    prediction_path=prediction_path,
                    prediction_sha256=prediction_hash,
                    project_root=project_root,
                )
            )
    return pd.DataFrame(rows)


def _validate_raw_identity_summary(
    fold_frame: pd.DataFrame, summary_path: Path
) -> None:
    summary = pd.read_csv(summary_path)
    require_columns(
        summary,
        {
            "dataset",
            "dataset_id",
            "role",
            "method",
            "status",
            *METRICS,
            "coverage",
        },
        "identity-readout audit five-fold summary",
    )
    expected = {
        (dataset, dataset_id, method)
        for dataset, dataset_id, _label in PANEL_B_DATASETS
        for method, _control, _model in PANEL_B_METHODS
    }
    selected = summary[
        summary["dataset"].isin(dataset for dataset, _id, _label in PANEL_B_DATASETS)
        & summary["method"].isin(method for method, _control, _model in PANEL_B_METHODS)
    ].copy()
    observed = set(zip(selected["dataset"], selected["dataset_id"], selected["method"]))
    if len(selected) != 6 or observed != expected:
        raise EvidenceError(
            "Identity-readout audit summary lacks the six matched Panel B rows"
        )
    if not selected["role"].eq("primary").all() or not selected["status"].eq(
        "complete"
    ).all():
        raise EvidenceError(
            "Identity-readout audit Panel B summaries are not complete primary rows"
        )
    for dataset, dataset_id, _label in PANEL_B_DATASETS:
        for method, _control, _model in PANEL_B_METHODS:
            folds = fold_frame[
                fold_frame["dataset"].eq(dataset)
                & fold_frame["dataset_id"].eq(dataset_id)
                & fold_frame["setting"].eq(method)
            ]
            aggregate = selected[
                selected["dataset"].eq(dataset)
                & selected["dataset_id"].eq(dataset_id)
                & selected["method"].eq(method)
            ]
            if len(folds) != 5 or len(aggregate) != 1:
                raise EvidenceError(
                    f"Identity-readout audit multiplicity failed for {dataset}/{method}"
                )
            row = aggregate.iloc[0]
            for metric in METRICS:
                observed_value = folds[metric].to_numpy(dtype=float).mean()
                if not np.isclose(
                    float(row[metric]), observed_value, rtol=0.0, atol=1e-12
                ):
                    raise EvidenceError(
                        f"Identity-readout audit {metric} does not reproduce for "
                        f"{dataset}/{method}"
                    )
            if not np.isclose(
                float(row["coverage"]), folds["coverage"].mean(), rtol=0.0, atol=1e-12
            ):
                raise EvidenceError(
                    f"Identity-readout audit coverage does not reproduce for "
                    f"{dataset}/{method}"
                )


def load_panel_b_fold_rows(
    *,
    summary_path: Path = RAW_IDENTITY_FIVE_FOLD,
    overview_path: Path = RAW_IDENTITY_OVERVIEW,
    report_path: Path = RAW_EVALUATION_REPORT,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    summary_path = Path(summary_path)
    overview_path = Path(overview_path)
    report_path = Path(report_path)
    if not summary_path.is_file() or not report_path.is_file():
        raise EvidenceError("Panel B identity-readout audit inputs are missing")
    overview = load_json(overview_path)
    reported_source = resolve_reported_path(
        str(overview.get("source_report", "")), project_root
    )
    if reported_source.resolve(strict=False) != report_path.resolve(strict=False):
        raise EvidenceError(
            "Identity-readout audit overview points to a different evaluation report"
        )
    report = load_json(report_path)
    if (
        report.get("status") != "complete"
        or report.get("complete") is not True
        or overview.get("source_report_sha256") != report.get("sha256")
    ):
        raise EvidenceError(
            "Identity-readout audit report or overview is not complete and matched"
        )
    policy = report.get("evaluation_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("five_fold_metric")
        != "arithmetic_mean_of_exactly_five_finite_fold_medians"
        or policy.get("posthoc_calibration_performed") is not False
        or policy.get("readout_selection_performed") is not False
    ):
        raise EvidenceError("Identity-readout audit evaluation policy changed")
    records = report.get("fold_metrics")
    if not isinstance(records, list):
        raise EvidenceError("Identity-readout audit report has no fold metrics")
    frame = pd.DataFrame(records)
    require_columns(
        frame,
        {
            "dataset",
            "dataset_id",
            "role",
            "fold",
            "method",
            "model",
            "method_kind",
            "status",
            "readout",
            "posthoc_calibration",
            "matrix_scope",
            "fixed_test_indices_verified",
            *METRICS,
            "coverage",
            "eligible_gene_count",
            "valid_gene_count",
            "constant_prediction_count",
            "completion_manifest",
            "completion_manifest_sha256",
            "prediction_path",
            "prediction_sha256",
        },
        "identity-readout audit fold metrics",
    )
    dataset_names = {dataset for dataset, _id, _label in PANEL_B_DATASETS}
    method_names = {method for method, _control, _model in PANEL_B_METHODS}
    selected = frame[
        frame["dataset"].isin(dataset_names) & frame["method"].isin(method_names)
    ].copy()
    expected = {
        (dataset, dataset_id, fold, method, model)
        for dataset, dataset_id, _label in PANEL_B_DATASETS
        for fold in FOLDS
        for method, _control, model in PANEL_B_METHODS
    }
    observed = set(
        zip(
            selected["dataset"],
            selected["dataset_id"],
            selected["fold"],
            selected["method"],
            selected["model"],
        )
    )
    if len(selected) != 30 or observed != expected:
        raise EvidenceError(
            f"Panel B fold contract differs: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    if selected.duplicated(["dataset_id", "fold", "method"]).any():
        raise EvidenceError("Panel B contains duplicate matched fold rows")
    selected = selected.rename(
        columns={
            "eligible_gene_count": "eligible_genes",
            "valid_gene_count": "scored_genes",
            "constant_prediction_count": "constant_prediction_genes",
        }
    )
    selected["control"] = selected["method"].map(
        {method: control for method, control, _model in PANEL_B_METHODS}
    )
    selected["source_kind"] = PANEL_B_SOURCE_KIND
    selected = _numeric_metrics(selected, "Panel B matched identity-readout folds")
    if (
        not selected["role"].eq("primary").all()
        or not selected["method_kind"].eq("genespt").all()
        or not selected["status"].eq("evaluated").all()
        or not selected["readout"].eq("identity").all()
        or not selected["posthoc_calibration"].eq("none").all()
        or not selected["matrix_scope"].eq("frozen_final_test_genes").all()
        or not selected["fixed_test_indices_verified"].eq(True).all()
        or not np.allclose(selected["coverage"], 1.0, rtol=0.0, atol=1e-12)
        or not (selected["eligible_genes"] == selected["scored_genes"]).all()
    ):
        raise EvidenceError("Panel B is not a complete matched identity-readout comparison")

    report_hash = sha256_file(report_path)
    overview_hash = sha256_file(overview_path)
    summary_hash = sha256_file(summary_path)
    verified_manifests: dict[tuple[str, str], Path] = {}
    rows = []
    for dataset, dataset_id, _label in PANEL_B_DATASETS:
        dataset_frame = selected[
            selected["dataset"].eq(dataset) & selected["dataset_id"].eq(dataset_id)
        ]
        for fold in FOLDS:
            pair = dataset_frame[dataset_frame["fold"].eq(fold)]
            if pair["completion_manifest"].nunique() != 1 or pair[
                "completion_manifest_sha256"
            ].nunique() != 1:
                raise EvidenceError(
                    f"Panel B {dataset} fold{fold} does not share one frozen run manifest"
                )
        for method, control, model in PANEL_B_METHODS:
            method_frame = dataset_frame[dataset_frame["method"].eq(method)].sort_values(
                "fold"
            )
            for _, row in method_frame.iterrows():
                manifest_raw = str(row["completion_manifest"])
                manifest_hash = validate_sha256(
                    row["completion_manifest_sha256"],
                    f"Panel B {dataset} fold{int(row['fold'])} manifest",
                )
                cache_key = (manifest_raw, manifest_hash)
                if cache_key not in verified_manifests:
                    manifest_path = resolve_reported_path(manifest_raw, project_root)
                    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_hash:
                        raise EvidenceError(
                            f"Panel B completion manifest hash failed: {manifest_raw}"
                        )
                    verified_manifests[cache_key] = manifest_path
                manifest_path = verified_manifests[cache_key]
                prediction_hash = validate_sha256(
                    row["prediction_sha256"],
                    f"Panel B {dataset} fold{int(row['fold'])} prediction",
                )
                rows.append(
                    {
                        "panel": "B",
                        "dataset": dataset,
                        "dataset_id": dataset_id,
                        "fold": int(row["fold"]),
                        "setting": method,
                        "control": control,
                        "model": model,
                        "source_kind": PANEL_B_SOURCE_KIND,
                        "result_layer": PANEL_B_RESULT_LAYER,
                        "readout": "identity",
                        "posthoc_calibration": "none",
                        "aggregation_unit": "fold_median_across_eligible_genes",
                        "SPCC": float(row["SPCC"]),
                        "RMSE": float(row["RMSE"]),
                        "JSD": float(row["JSD"]),
                        "SSIM": float(row["SSIM"]),
                        "coverage": float(row["coverage"]),
                        "eligible_genes": int(row["eligible_genes"]),
                        "scored_genes": int(row["scored_genes"]),
                        "constant_prediction_genes": int(
                            row["constant_prediction_genes"]
                        ),
                        "source_metrics_path": project_relative(report_path, project_root),
                        "source_metrics_sha256": report_hash,
                        "source_index_path": project_relative(overview_path, project_root),
                        "source_index_sha256": overview_hash,
                        "source_manifest_path": project_relative(
                            manifest_path, project_root
                        ),
                        "source_manifest_sha256": manifest_hash,
                        "prediction_path": portable_reported_path(
                            str(row["prediction_path"]), project_root
                        ),
                        "prediction_sha256": prediction_hash,
                        "reference_five_fold_path": project_relative(
                            summary_path, project_root
                        ),
                        "reference_five_fold_sha256": summary_hash,
                    }
                )
    result = pd.DataFrame(rows)
    _validate_raw_identity_summary(result, summary_path)
    return result


def load_fold_level_table(
    *,
    panel_a_root: Path = PANEL_A_ROOT,
    panel_c_root: Path = PANEL_C_ROOT,
    raw_summary_path: Path = RAW_IDENTITY_FIVE_FOLD,
    raw_overview_path: Path = RAW_IDENTITY_OVERVIEW,
    raw_report_path: Path = RAW_EVALUATION_REPORT,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    panel_a = load_panel_a_fold_rows(panel_a_root, project_root=project_root)
    panel_b = load_panel_b_fold_rows(
        summary_path=raw_summary_path,
        overview_path=raw_overview_path,
        report_path=raw_report_path,
        project_root=project_root,
    )
    panel_c = load_panel_c_fold_rows(panel_c_root, project_root=project_root)
    for column in ("reference_five_fold_path", "reference_five_fold_sha256"):
        if column not in panel_a:
            panel_a[column] = ""
        if column not in panel_c:
            panel_c[column] = ""
    result = pd.concat([panel_a, panel_b, panel_c], ignore_index=True)
    expected_counts = {"A": 20, "B": 30, "C": 40}
    if result.groupby("panel").size().to_dict() != expected_counts or len(result) != 90:
        raise EvidenceError("S3 fold-level panel dimensions are not 20, 30, and 40")
    if result.duplicated(["panel", "dataset_id", "fold", "setting"]).any():
        raise EvidenceError("S3 fold-level table contains duplicate formal rows")
    return result


def build_five_fold_summary(fold_level: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "panel",
        "dataset",
        "dataset_id",
        "setting",
        "control",
        "model",
        "source_kind",
        "result_layer",
        "readout",
        "posthoc_calibration",
    ]
    require_columns(
        fold_level,
        set(group_columns)
        | set(METRICS)
        | {
            "fold",
            "coverage",
            "eligible_genes",
            "scored_genes",
            "constant_prediction_genes",
        },
        "S3 fold-level table",
    )
    rows = []
    for keys, frame in fold_level.groupby(group_columns, sort=False, dropna=False):
        folds = tuple(sorted(frame["fold"].astype(int).unique()))
        if len(frame) != 5 or folds != FOLDS:
            raise EvidenceError(f"Five-fold summary group is incomplete: {keys}")
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_folds": 5,
                "aggregation": "arithmetic_mean_of_fold_medians",
                "fold_sd_definition": "population_sd_across_five_fold_medians_ddof0",
                "coverage_mean": float(frame["coverage"].mean()),
                "coverage_min": float(frame["coverage"].min()),
                "eligible_genes_total": int(frame["eligible_genes"].sum()),
                "scored_genes_total": int(frame["scored_genes"].sum()),
                "constant_prediction_genes_total": int(
                    frame["constant_prediction_genes"].sum()
                ),
            }
        )
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=float)
            row[metric] = float(values.mean())
            row[f"{metric}_fold_sd_ddof0"] = float(values.std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    expected_counts = {"A": 4, "B": 6, "C": 8}
    if summary.groupby("panel").size().to_dict() != expected_counts or len(summary) != 18:
        raise EvidenceError("S3 summary dimensions are not 4, 6, and 8")
    return summary


def build_panel_b_improvements(fold_level: pd.DataFrame) -> pd.DataFrame:
    panel_b = fold_level[fold_level["panel"].eq("B")]
    rows = []
    for dataset, dataset_id, _label in PANEL_B_DATASETS:
        frame = panel_b[
            panel_b["dataset"].eq(dataset) & panel_b["dataset_id"].eq(dataset_id)
        ]
        base = frame[frame["setting"].eq("GeneSPT-GC")].sort_values("fold")
        full = frame[frame["setting"].eq("GeneSPT")].sort_values("fold")
        if len(base) != 5 or len(full) != 5 or not np.array_equal(
            base["fold"].to_numpy(dtype=int), full["fold"].to_numpy(dtype=int)
        ):
            raise EvidenceError(f"Panel B pairing failed for {dataset}")
        delta_spcc = full["SPCC"].to_numpy(dtype=float) - base["SPCC"].to_numpy(
            dtype=float
        )
        rmse_improvement = base["RMSE"].to_numpy(dtype=float) - full[
            "RMSE"
        ].to_numpy(dtype=float)
        jsd_improvement = base["JSD"].to_numpy(dtype=float) - full["JSD"].to_numpy(
            dtype=float
        )
        delta_ssim = full["SSIM"].to_numpy(dtype=float) - base["SSIM"].to_numpy(
            dtype=float
        )
        row: dict[str, Any] = {
            "dataset": dataset,
            "dataset_id": dataset_id,
            "result_layer": PANEL_B_RESULT_LAYER,
            "readout": "identity",
            "posthoc_calibration": "none",
            "matched_difference": "GeneSPT adds PSP to the frozen GeneSPT-GC base",
            "n_folds": 5,
            "aggregation": "mean_of_paired_fold_improvements",
            "genespt_gc_SPCC": float(base["SPCC"].mean()),
            "genespt_SPCC": float(full["SPCC"].mean()),
            "delta_SPCC": float(delta_spcc.mean()),
            "delta_SPCC_fold_sd_ddof0": float(delta_spcc.std(ddof=0)),
            "genespt_gc_RMSE": float(base["RMSE"].mean()),
            "genespt_RMSE": float(full["RMSE"].mean()),
            "RMSE_improvement": float(rmse_improvement.mean()),
            "RMSE_improvement_fold_sd_ddof0": float(
                rmse_improvement.std(ddof=0)
            ),
            "genespt_gc_JSD": float(base["JSD"].mean()),
            "genespt_JSD": float(full["JSD"].mean()),
            "JSD_improvement": float(jsd_improvement.mean()),
            "JSD_improvement_fold_sd_ddof0": float(
                jsd_improvement.std(ddof=0)
            ),
            "genespt_gc_SSIM": float(base["SSIM"].mean()),
            "genespt_SSIM": float(full["SSIM"].mean()),
            "delta_SSIM": float(delta_ssim.mean()),
            "delta_SSIM_fold_sd_ddof0": float(delta_ssim.std(ddof=0)),
            "delta_SPCC_definition": "GeneSPT_SPCC_minus_GeneSPT-GC_SPCC",
            "RMSE_improvement_definition": "GeneSPT-GC_RMSE_minus_GeneSPT_RMSE",
            "JSD_improvement_definition": "GeneSPT-GC_JSD_minus_GeneSPT_JSD",
            "delta_SSIM_definition": "GeneSPT_SSIM_minus_GeneSPT-GC_SSIM",
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    display_values = result[["delta_SPCC", "RMSE_improvement", "JSD_improvement"]]
    if not (display_values.to_numpy(dtype=float) > 0.0).all():
        raise EvidenceError("A source-derived Panel B display improvement is not positive")
    return result


def _public_s3_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove internal trace fields while preserving all scientific values."""
    public = frame.drop(
        columns=[column for column in PUBLIC_PROVENANCE_COLUMNS if column in frame],
        errors="raise",
    ).copy()
    if "source_kind" in public:
        public["source_kind"] = public["source_kind"].replace(
            PUBLIC_SOURCE_KIND_REPLACEMENTS
        )
    return public


def build_public_s3_tables(
    fold_level: pd.DataFrame,
    summary: pd.DataFrame,
    panel_b: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Project verified internal evidence into the reviewer-facing S3 schema."""
    public_fold_level = _public_s3_frame(fold_level)
    public_summary = _public_s3_frame(summary)
    public_panel_b = _public_s3_frame(panel_b)
    if any(column in public_fold_level for column in PUBLIC_PROVENANCE_COLUMNS):
        raise EvidenceError("Public S3 fold-level table retains internal provenance")
    return public_fold_level, public_summary, public_panel_b


def build_figure_source(
    summary: pd.DataFrame,
    panel_b_improvements: pd.DataFrame,
    *,
    summary_path: Path,
    panel_b_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    summary_hash = sha256_file(summary_path)
    panel_b_hash = sha256_file(panel_b_path)
    rows: list[dict[str, Any]] = []
    panel_a = summary[summary["panel"].eq("A")]
    for _control, setting, _model in PANEL_A_SETTINGS:
        row = panel_a[panel_a["setting"].eq(setting)].iloc[0]
        rows.append(
            {
                "panel": "A",
                "dataset": "Vis9A",
                "metric": "SPCC",
                "metric_label": "SPCC",
                "setting": setting,
                "control": row["control"],
                "value": float(row["SPCC"]),
                "fold_sd_ddof0": float(row["SPCC_fold_sd_ddof0"]),
                "baseline_value": np.nan,
                "improvement": np.nan,
                "improvement_fold_sd_ddof0": np.nan,
                "improvement_definition": "",
                "n_folds": 5,
                "result_layer": row["result_layer"],
                "uncertainty_displayed": "fold_sd_ddof0",
                "horizontal_line_semantics": "fold_sd_error_bar",
                "source_data_path": project_relative(summary_path, project_root),
                "source_data_sha256": summary_hash,
            }
        )
    for dataset, _dataset_id, _label in PANEL_B_DATASETS:
        row = panel_b_improvements[
            panel_b_improvements["dataset"].eq(dataset)
        ].iloc[0]
        specs = (
            (
                "SPCC",
                "Delta SPCC",
                "genespt_SPCC",
                "genespt_gc_SPCC",
                "delta_SPCC",
                "delta_SPCC_fold_sd_ddof0",
                "delta_SPCC_definition",
            ),
        )
        for metric, label, value_col, base_col, delta_col, sd_col, definition_col in specs:
            rows.append(
                {
                    "panel": "B",
                    "dataset": dataset,
                    "metric": metric,
                    "metric_label": label,
                    "setting": "GeneSPT",
                    "control": "GeneSPT-GC",
                    "value": float(row[value_col]),
                    "fold_sd_ddof0": float(
                        summary[
                            summary["panel"].eq("B")
                            & summary["dataset"].eq(dataset)
                            & summary["setting"].eq("GeneSPT")
                        ].iloc[0][f"{metric}_fold_sd_ddof0"]
                    ),
                    "baseline_value": float(row[base_col]),
                    "improvement": float(row[delta_col]),
                    "improvement_fold_sd_ddof0": float(row[sd_col]),
                    "improvement_definition": str(row[definition_col]),
                    "n_folds": 5,
                    "result_layer": PANEL_B_RESULT_LAYER,
                    "uncertainty_displayed": "none",
                    "horizontal_line_semantics": "zero_reference_connector",
                    "source_data_path": project_relative(panel_b_path, project_root),
                    "source_data_sha256": panel_b_hash,
                }
            )
    panel_c = summary[summary["panel"].eq("C")]
    baseline = panel_c[panel_c["control"].eq("base")].iloc[0]
    for _control, setting, _model in PANEL_C_SETTINGS:
        row = panel_c[panel_c["setting"].eq(setting)].iloc[0]
        rows.append(
            {
                "panel": "C",
                "dataset": "Vis9A",
                "metric": "SPCC",
                "metric_label": "SPCC improvement over GeneSPT-GC",
                "setting": setting,
                "control": row["control"],
                "value": float(row["SPCC"]),
                "fold_sd_ddof0": float(row["SPCC_fold_sd_ddof0"]),
                "baseline_value": float(baseline["SPCC"]),
                "improvement": float(row["SPCC"] - baseline["SPCC"]),
                "improvement_fold_sd_ddof0": np.nan,
                "improvement_definition": "setting_SPCC_minus_GeneSPT-GC_SPCC",
                "n_folds": 5,
                "result_layer": row["result_layer"],
                "uncertainty_displayed": "none",
                "horizontal_line_semantics": "zero_reference_connector",
                "source_data_path": project_relative(summary_path, project_root),
                "source_data_sha256": summary_hash,
            }
        )
    source = pd.DataFrame(rows)
    if source.groupby("panel").size().to_dict() != {"A": 4, "B": 3, "C": 8}:
        raise EvidenceError("Figure 3 source dimensions are not 4, 3, and 8")
    return source


def clean_axis(axis: plt.Axes, grid_axis: str = "x") -> None:
    axis.set_facecolor("white")
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    if grid_axis != "y":
        axis.grid(axis="y", visible=False)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.spines["left"].set_color("#C7CDD4")
    axis.spines["bottom"].set_color("#C7CDD4")
    axis.tick_params(colors="#3F454C", labelsize=10.5)


def panel_label(axis: plt.Axes, text: str, y: float = 1.08) -> None:
    axis.text(
        0.0,
        y,
        text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="#20242A",
    )


def padded_limits(values: list[float], *, minimum_pad: float = 0.0002) -> tuple[float, float]:
    low = min([0.0, *values])
    high = max([0.0, *values])
    span = max(high - low, abs(low), abs(high), minimum_pad)
    pad = max(0.14 * span, minimum_pad)
    return low - pad, high + pad


def plot_figure(source: pd.DataFrame, *, pdf_path: Path, png_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(12.8, 7.8), facecolor="white")
    grid = GridSpec(
        2,
        2,
        figure=figure,
        width_ratios=[1.0, 1.18],
        height_ratios=[1.0, 1.25],
        left=0.09,
        right=0.985,
        top=0.92,
        bottom=0.105,
        wspace=0.36,
        hspace=0.55,
    )

    axis_a = figure.add_subplot(grid[0, 0])
    clean_axis(axis_a)
    panel_label(axis_a, "A. Gene descriptor controls (Vis9A)")
    panel_a = source[source["panel"].eq("A")].copy()
    order_a = [setting for _control, setting, _model in PANEL_A_SETTINGS]
    panel_a["setting"] = pd.Categorical(panel_a["setting"], order_a, ordered=True)
    panel_a = panel_a.sort_values("setting")
    y_positions = np.arange(len(panel_a))[::-1]
    for y_position, (_, row) in zip(y_positions, panel_a.iterrows()):
        setting = str(row["setting"])
        color = (
            RED
            if setting == "Correct"
            else BLUE_GREY
            if setting in {"Shuffled", "Permuted"}
            else GREY
        )
        marker = "D" if setting == "Shuffled" else "o"
        axis_a.errorbar(
            float(row["value"]),
            y_position,
            xerr=float(row["fold_sd_ddof0"]),
            fmt=marker,
            markersize=7.5,
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=3,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
    axis_a.set_yticks(y_positions)
    axis_a.set_yticklabels(panel_a["setting"].astype(str).tolist())
    axis_a.set_xlabel("SPCC (higher is better)")
    left_extent = float((panel_a["value"] - panel_a["fold_sd_ddof0"]).min())
    right_extent = float((panel_a["value"] + panel_a["fold_sd_ddof0"]).max())
    span = max(right_extent - left_extent, 0.002)
    axis_a.set_xlim(left_extent - 0.12 * span, right_extent + 0.12 * span)

    axis_b = figure.add_subplot(grid[0, 1])
    clean_axis(axis_b)
    panel_label(axis_b, "B. PSP contribution across primary datasets")
    panel_b = source[source["panel"].eq("B")].copy()
    y_positions = np.arange(len(PANEL_B_DATASETS))[::-1]
    values: list[float] = []
    for row_index, (dataset, _dataset_id, label) in enumerate(PANEL_B_DATASETS):
        match = panel_b[panel_b["dataset"].eq(dataset)]
        if len(match) != 1:
            raise EvidenceError(f"Figure 3B source row is missing: {dataset}/SPCC")
        improvement = float(match.iloc[0]["improvement"])
        values.append(improvement)
        axis_b.plot(
            [0, improvement],
            [y_positions[row_index], y_positions[row_index]],
            color=LINE,
            linewidth=2.2,
            zorder=2,
        )
        axis_b.scatter(
            improvement,
            y_positions[row_index],
            s=62,
            color=RED,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    axis_b.axvline(0, color=ZERO, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    axis_b.set_yticks(y_positions)
    axis_b.set_yticklabels(
        [label for _dataset, _dataset_id, label in PANEL_B_DATASETS]
    )
    axis_b.set_xlabel("Delta SPCC (GeneSPT - GeneSPT-GC)")
    axis_b.set_xlim(*padded_limits(values))

    axis_c = figure.add_subplot(grid[1, :])
    clean_axis(axis_c)
    panel_label(axis_c, "C. PSP control tests (Vis9A)")
    panel_c = source[source["panel"].eq("C")].copy()
    order_c = [setting for _control, setting, _model in PANEL_C_SETTINGS]
    panel_c["setting"] = pd.Categorical(panel_c["setting"], order_c, ordered=True)
    panel_c = panel_c.sort_values("setting")
    y_positions = np.arange(len(panel_c))[::-1]
    values = []
    for y_position, (_, row) in zip(y_positions, panel_c.iterrows()):
        setting = str(row["setting"])
        improvement = float(row["improvement"])
        values.append(improvement)
        if setting == "Correct PSP":
            color, size = RED, 70
        elif setting == "GC-only":
            color, size = DARK, 55
        else:
            color, size = BLUE_GREY, 55
        axis_c.plot(
            [0, improvement],
            [y_position, y_position],
            color=color,
            alpha=0.52,
            linewidth=2.2,
            zorder=2,
        )
        axis_c.scatter(
            improvement,
            y_position,
            s=size,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    axis_c.axvline(0, color=ZERO, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    axis_c.set_yticks(y_positions)
    axis_c.set_yticklabels(panel_c["setting"].astype(str).tolist())
    axis_c.set_xlabel("Delta SPCC relative to GeneSPT-GC")
    axis_c.set_xlim(*padded_limits(values, minimum_pad=0.0005))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", na_rep="")


def write_caption(path: Path) -> None:
    caption = (
        "**Figure 3. Mechanistic validation of target-gene descriptors and PSP.** "
        "A, Vis9A controls compare PCA32 with random, shuffled, and permuted-label "
        "descriptors; points show five-fold mean SPCC ± SD. B, matched GeneSPT-GC and "
        "GeneSPT comparisons across the three primary datasets report ΔSPCC (GeneSPT "
        "− GeneSPT-GC). C, Vis9A PSP controls test learned programs, spatial "
        "correspondence, and gene-specific coefficients; values are ΔSPCC relative to "
        "GeneSPT-GC. Complete SPCC, RMSE, JS/JSD, SSIM, and fold-level results are in "
        "Supplementary Table S3.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption, encoding="utf-8")


def write_readme(
    path: Path,
    *,
    fold_level_path: Path,
    summary_path: Path,
    panel_b_path: Path,
) -> None:
    text = f"""# Supplementary Table S3

Generated from verified strict whole-gene holdout evidence.

## Fold-level table

`{fold_level_path.name}` is the 90-row table. Panel A contains four descriptor settings by five folds (20 rows), Panel B contains three datasets by two matched identity-readout settings by five folds (30 rows), and Panel C contains eight Vis9A mechanism settings by five folds (40 rows). Every row reports SPCC, RMSE, JSD, SSIM, coverage, eligible/scored gene counts, and the corresponding model/control definition.

## Five-fold summary

`{summary_path.name}` contains 18 setting-level rows. Metric values are arithmetic means of the five fold medians. Each `*_fold_sd_ddof0` column is the population SD across those five fold medians. Coverage is reported as both the fold mean and fold minimum.

## Panel B improvements

`{panel_b_path.name}` contains one matched identity-readout comparison for each of Vis9A, HBC, and Cell2location. Delta SPCC and delta SSIM are GeneSPT minus GeneSPT-GC. RMSE and JSD improvements are GeneSPT-GC minus GeneSPT. Means and paired-fold SDs are computed from the fold-level values and independently verified against the frozen five-fold summary before export. Panel B is a mechanism-isolation result: the matched model difference is the added PSP.

## Figure semantics

Panel A displays mean +/- fold SD. Panels B and C display point estimates only; their horizontal segments connect values to the zero reference and do not encode uncertainty.

Public artifact row counts and SHA256 checksums are recorded in `{S3_MANIFEST_NAME}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def public_file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def write_generation_manifest(
    path: Path,
    *,
    outputs: Mapping[str, Path],
    fold_level_rows: int,
    summary_rows: int,
) -> None:
    payload = {
        "schema_version": 2,
        "status": "complete",
        "description": (
            "Public Supplementary Table S3 artifacts for descriptor controls, "
            "matched GeneSPT-GC versus GeneSPT PSP ablation, and mechanism controls "
            "under strict whole-gene holdout."
        ),
        "artifacts": {
            "fold_level": public_file_record(
                outputs["s3_fold_level"], rows=fold_level_rows
            ),
            "five_fold_summary": public_file_record(
                outputs["s3_summary"], rows=summary_rows
            ),
            "panel_b_improvements": public_file_record(
                outputs["s3b_improvements"], rows=3
            ),
            "readme": public_file_record(outputs["readme"]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _public_s3_outputs(s3_dir: Path) -> dict[str, Path]:
    return {
        "s3_fold_level": s3_dir / S3_FOLD_LEVEL_NAME,
        "s3_summary": s3_dir / S3_SUMMARY_NAME,
        "s3b_improvements": s3_dir / S3_B_IMPROVEMENT_NAME,
        "readme": s3_dir / S3_README_NAME,
        "manifest": s3_dir / S3_MANIFEST_NAME,
    }


def _assert_public_frame_matches(
    actual_path: Path, expected: pd.DataFrame, *, label: str
) -> None:
    actual = pd.read_csv(actual_path, keep_default_na=False, float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise EvidenceError(f"{label} changed during public export") from error


def validate_public_s3_artifacts(
    *,
    public_fold_level: pd.DataFrame,
    public_summary: pd.DataFrame,
    public_panel_b: pd.DataFrame,
    outputs: Mapping[str, Path],
) -> None:
    expected_keys = {
        "s3_fold_level",
        "s3_summary",
        "s3b_improvements",
        "readme",
        "manifest",
    }
    if set(outputs) != expected_keys or not all(path.is_file() for path in outputs.values()):
        raise EvidenceError("Public S3 artifact set is incomplete")
    if len(public_fold_level) != 90 or public_fold_level.groupby("panel").size().to_dict() != {
        "A": 20,
        "B": 30,
        "C": 40,
    }:
        raise EvidenceError("Public S3 fold-level row contract changed")
    if len(public_summary) != 18 or public_summary.groupby("panel").size().to_dict() != {
        "A": 4,
        "B": 6,
        "C": 8,
    }:
        raise EvidenceError("Public S3 summary row contract changed")
    if len(public_panel_b) != 3:
        raise EvidenceError("Public S3 Panel B improvement row contract changed")
    for frame in (public_fold_level, public_summary, public_panel_b):
        forbidden_columns = set(frame.columns) & set(PUBLIC_PROVENANCE_COLUMNS)
        if forbidden_columns:
            raise EvidenceError(
                f"Public S3 retains internal provenance columns: {sorted(forbidden_columns)}"
            )

    _assert_public_frame_matches(
        outputs["s3_fold_level"], public_fold_level, label="S3 fold-level table"
    )
    _assert_public_frame_matches(
        outputs["s3_summary"], public_summary, label="S3 five-fold summary"
    )
    _assert_public_frame_matches(
        outputs["s3b_improvements"],
        public_panel_b,
        label="S3 Panel B improvements",
    )

    for artifact_path in outputs.values():
        text = artifact_path.read_text(encoding="utf-8")
        for label, pattern in PUBLIC_S3_FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                raise EvidenceError(
                    f"Public S3 artifact {artifact_path.name} contains {label}"
                )

    manifest = load_json(outputs["manifest"])
    if set(manifest) != {"schema_version", "status", "description", "artifacts"}:
        raise EvidenceError("Public S3 manifest exposes non-public metadata")
    if manifest.get("schema_version") != 2 or manifest.get("status") != "complete":
        raise EvidenceError("Public S3 manifest identity is invalid")
    artifact_records = manifest.get("artifacts")
    expected_records = {
        "fold_level": (outputs["s3_fold_level"], 90),
        "five_fold_summary": (outputs["s3_summary"], 18),
        "panel_b_improvements": (outputs["s3b_improvements"], 3),
        "readme": (outputs["readme"], None),
    }
    if not isinstance(artifact_records, Mapping) or set(artifact_records) != set(
        expected_records
    ):
        raise EvidenceError("Public S3 manifest artifact set changed")
    for key, (artifact_path, rows) in expected_records.items():
        record = artifact_records[key]
        if (
            not isinstance(record, Mapping)
            or record.get("file") != artifact_path.name
            or "/" in str(record.get("file", ""))
            or "\\" in str(record.get("file", ""))
            or int(record.get("bytes", -1)) != artifact_path.stat().st_size
            or record.get("sha256") != sha256_file(artifact_path)
        ):
            raise EvidenceError(f"Public S3 manifest record is invalid for {key}")
        if rows is None:
            if "rows" in record:
                raise EvidenceError(f"Public S3 non-tabular record has rows for {key}")
        elif int(record.get("rows", -1)) != rows:
            raise EvidenceError(f"Public S3 row count is invalid for {key}")


def write_public_s3_bundle(
    *,
    fold_level: pd.DataFrame,
    summary: pd.DataFrame,
    panel_b: pd.DataFrame,
    s3_dir: Path,
) -> dict[str, Path]:
    public_fold_level, public_summary, public_panel_b = build_public_s3_tables(
        fold_level, summary, panel_b
    )
    outputs = _public_s3_outputs(Path(s3_dir))
    Path(s3_dir).mkdir(parents=True, exist_ok=True)
    write_csv(public_fold_level, outputs["s3_fold_level"])
    write_csv(public_summary, outputs["s3_summary"])
    write_csv(public_panel_b, outputs["s3b_improvements"])
    write_readme(
        outputs["readme"],
        fold_level_path=outputs["s3_fold_level"],
        summary_path=outputs["s3_summary"],
        panel_b_path=outputs["s3b_improvements"],
    )
    write_generation_manifest(
        outputs["manifest"],
        outputs=outputs,
        fold_level_rows=len(public_fold_level),
        summary_rows=len(public_summary),
    )
    validate_public_s3_artifacts(
        public_fold_level=public_fold_level,
        public_summary=public_summary,
        public_panel_b=public_panel_b,
        outputs=outputs,
    )
    return outputs


def validate_generated_artifacts(
    *,
    fold_level: pd.DataFrame,
    summary: pd.DataFrame,
    panel_b: pd.DataFrame,
    figure_source: pd.DataFrame,
    outputs: Mapping[str, Path],
) -> None:
    if len(fold_level) != 90 or fold_level.groupby("panel").size().to_dict() != {
        "A": 20,
        "B": 30,
        "C": 40,
    }:
        raise EvidenceError("Generated S3 fold-level dimensions changed")
    if len(summary) != 18 or summary.groupby("panel").size().to_dict() != {
        "A": 4,
        "B": 6,
        "C": 8,
    }:
        raise EvidenceError("Generated S3 summary dimensions changed")
    if len(panel_b) != 3 or len(figure_source) != 15:
        raise EvidenceError("Generated improvement or figure source dimensions changed")
    if not np.allclose(fold_level["coverage"], 1.0, rtol=0.0, atol=1e-12):
        raise EvidenceError("Generated S3 contains incomplete coverage")
    for column in (
        "source_metrics_sha256",
        "source_index_sha256",
        "source_manifest_sha256",
        "prediction_sha256",
    ):
        if not fold_level[column].astype(str).map(SHA256_PATTERN.fullmatch).map(bool).all():
            raise EvidenceError(f"Generated S3 contains invalid provenance in {column}")
    panel_a_source = figure_source[figure_source["panel"].eq("A")]
    if (
        not panel_a_source["uncertainty_displayed"].eq("fold_sd_ddof0").all()
        or not (panel_a_source["fold_sd_ddof0"] > 0.0).all()
    ):
        raise EvidenceError("Figure 3A does not expose fold SD correctly")
    point_only = figure_source[figure_source["panel"].isin(["B", "C"])]
    if (
        not point_only["uncertainty_displayed"].eq("none").all()
        or not point_only["horizontal_line_semantics"].eq(
            "zero_reference_connector"
        ).all()
    ):
        raise EvidenceError("Figure 3B/C connector semantics changed")
    display_improvements = panel_b[
        ["delta_SPCC", "RMSE_improvement", "JSD_improvement"]
    ].to_numpy(dtype=float)
    if not (display_improvements > 0.0).all():
        raise EvidenceError("Panel B source-derived display improvements changed sign")

    with Image.open(outputs["figure_png"]) as image:
        if image.width < 3800 or image.height < 2100:
            raise EvidenceError(f"Figure 3 PNG is not high resolution: {image.size}")
        extrema = image.convert("RGB").getextrema()
        if all(low == high for low, high in extrema):
            raise EvidenceError("Figure 3 PNG is blank")
    pdf_path = outputs["figure_pdf"]
    if pdf_path.stat().st_size < 10_000 or not pdf_path.read_bytes().startswith(b"%PDF"):
        raise EvidenceError("Figure 3 PDF is missing or invalid")
    manifest = load_json(outputs["manifest"])
    if manifest.get("status") != "complete":
        raise EvidenceError("Generated S3 manifest is not complete")


def generate(
    *,
    panel_a_root: Path = PANEL_A_ROOT,
    panel_c_root: Path = PANEL_C_ROOT,
    raw_summary_path: Path = RAW_IDENTITY_FIVE_FOLD,
    raw_overview_path: Path = RAW_IDENTITY_OVERVIEW,
    raw_report_path: Path = RAW_EVALUATION_REPORT,
    figure_dir: Path = FIGURE_DIR,
    s3_dir: Path = S3_DIR,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
    panel_a_root = Path(panel_a_root)
    panel_c_root = Path(panel_c_root)
    raw_summary_path = Path(raw_summary_path)
    raw_overview_path = Path(raw_overview_path)
    raw_report_path = Path(raw_report_path)
    figure_dir = Path(figure_dir)
    s3_dir = Path(s3_dir)

    fold_level = load_fold_level_table(
        panel_a_root=panel_a_root,
        panel_c_root=panel_c_root,
        raw_summary_path=raw_summary_path,
        raw_overview_path=raw_overview_path,
        raw_report_path=raw_report_path,
        project_root=project_root,
    )
    summary = build_five_fold_summary(fold_level)
    panel_b = build_panel_b_improvements(fold_level)

    s3_outputs = write_public_s3_bundle(
        fold_level=fold_level,
        summary=summary,
        panel_b=panel_b,
        s3_dir=s3_dir,
    )
    outputs = {
        "figure_source": figure_dir / FIGURE_SOURCE_NAME,
        "figure_png": figure_dir / FIGURE_PNG_NAME,
        "figure_pdf": figure_dir / FIGURE_PDF_NAME,
        "figure_caption": figure_dir / FIGURE_CAPTION_NAME,
        **s3_outputs,
    }
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_source = build_figure_source(
        summary,
        panel_b,
        summary_path=outputs["s3_summary"],
        panel_b_path=outputs["s3b_improvements"],
        project_root=project_root,
    )
    write_csv(figure_source, outputs["figure_source"])
    plot_figure(
        figure_source,
        pdf_path=outputs["figure_pdf"],
        png_path=outputs["figure_png"],
    )
    write_caption(outputs["figure_caption"])
    validate_generated_artifacts(
        fold_level=fold_level,
        summary=summary,
        panel_b=panel_b,
        figure_source=figure_source,
        outputs=outputs,
    )
    return outputs


def generate_public_s3_only(
    *,
    panel_a_root: Path = PANEL_A_ROOT,
    panel_c_root: Path = PANEL_C_ROOT,
    raw_summary_path: Path = RAW_IDENTITY_FIVE_FOLD,
    raw_overview_path: Path = RAW_IDENTITY_OVERVIEW,
    raw_report_path: Path = RAW_EVALUATION_REPORT,
    s3_dir: Path = S3_DIR,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
    """Regenerate only reviewer-facing S3 files, leaving Figure 3 untouched."""
    fold_level = load_fold_level_table(
        panel_a_root=Path(panel_a_root),
        panel_c_root=Path(panel_c_root),
        raw_summary_path=Path(raw_summary_path),
        raw_overview_path=Path(raw_overview_path),
        raw_report_path=Path(raw_report_path),
        project_root=project_root,
    )
    summary = build_five_fold_summary(fold_level)
    panel_b = build_panel_b_improvements(fold_level)
    return write_public_s3_bundle(
        fold_level=fold_level,
        summary=summary,
        panel_b=panel_b,
        s3_dir=Path(s3_dir),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-a-root", type=Path, default=PANEL_A_ROOT)
    parser.add_argument("--panel-c-root", type=Path, default=PANEL_C_ROOT)
    parser.add_argument("--raw-summary", type=Path, default=RAW_IDENTITY_FIVE_FOLD)
    parser.add_argument("--raw-overview", type=Path, default=RAW_IDENTITY_OVERVIEW)
    parser.add_argument("--raw-report", type=Path, default=RAW_EVALUATION_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--s3-dir", type=Path, default=S3_DIR)
    parser.add_argument(
        "--s3-only",
        action="store_true",
        help="Regenerate only the public Supplementary Table S3 bundle.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    common = {
        "panel_a_root": args.panel_a_root,
        "panel_c_root": args.panel_c_root,
        "raw_summary_path": args.raw_summary,
        "raw_overview_path": args.raw_overview,
        "raw_report_path": args.raw_report,
        "s3_dir": args.s3_dir,
    }
    if args.s3_only:
        outputs = generate_public_s3_only(**common)
    else:
        outputs = generate(figure_dir=args.figure_dir, **common)
    improvements = pd.read_csv(outputs["s3b_improvements"])
    for row in improvements.itertuples(index=False):
        print(
            f"Panel B {row.dataset}: Delta SPCC={row.delta_SPCC:.8f}, "
            f"RMSE improvement={row.RMSE_improvement:.8f}, "
            f"JSD improvement={row.JSD_improvement:.8f}"
        )
    if "figure_pdf" in outputs:
        print(f"Figure 3: {outputs['figure_pdf']}")
    print(f"Supplementary Table S3: {outputs['s3_fold_level']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
