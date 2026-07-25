#!/usr/bin/env python3
"""Generate the fixed Protocol A Figure 4 HBC spatial-map grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT.parent
RUN_ROOT = PROJECT_ROOT / "results" / "protocol_a_full_rerun_20260711"
CONFIG_PATH = PROJECT_ROOT / "configs" / "protocol_a_datasets.formal.yaml"
OUTPUT_DIR = RUN_ROOT / "figures" / "figure4"

DATASET_ID = "HBC_shared16112"
FORMAL_GENESPT_LAYER = "validation_selected_readout_genespt57"
RAW_BASELINE_LAYER = "raw_identity"
METHOD_ORDER = (
    "Ground truth",
    "GeneSPT",
    "Tangram",
    "TransImp",
    "SpaIM",
    "SpaGE",
    "stPlus",
    "stAI",
)
EXTERNAL_METHODS = METHOD_ORDER[2:]
FIGURE_STEM = "figure4_hbc_representative_maps"


@dataclass(frozen=True)
class PanelSpec:
    gene: str
    fold: int
    gene_idx: int
    pattern_type: str


FIXED_PANELS = (
    PanelSpec("B2M", 4, 11120, "broad immune-associated pattern"),
    PanelSpec("CD74", 1, 4810, "immune local enrichment"),
    PanelSpec("COL3A1", 3, 2451, "stromal ECM gradient"),
    PanelSpec("KRT17", 0, 12644, "tumor epithelial enrichment"),
)


VIRIDIS = np.asarray(
    [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ],
    dtype=np.float32,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_relative_path(path: Path, package_root: Path = PACKAGE_ROOT) -> str:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(package_root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"Path is outside the package root: {resolved}") from error
    return relative.as_posix()


class ArtifactRegistry:
    """Hash each unique input once and retain portable manifest records."""

    def __init__(self, package_root: Path) -> None:
        self.package_root = Path(package_root)
        self._records: dict[Path, dict[str, Any]] = {}

    def register(
        self,
        path: Path,
        *,
        kind: str,
        declared_sha256: str | None = None,
    ) -> dict[str, Any]:
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"Required Figure 4 artifact is not a file: {resolved}")
        if resolved not in self._records:
            self._records[resolved] = {
                "package_relative_path": package_relative_path(
                    resolved, self.package_root
                ),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
                "kinds": set(),
            }
        record = self._records[resolved]
        record["kinds"].add(kind)
        if declared_sha256 is not None:
            if len(declared_sha256) != 64:
                raise ValueError(f"Invalid declared SHA-256 for {resolved}")
            if record["sha256"] != declared_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {resolved}: declared={declared_sha256}, "
                    f"observed={record['sha256']}"
                )
        return record

    def manifest_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for record in self._records.values():
            serializable = dict(record)
            serializable["kinds"] = sorted(serializable["kinds"])
            records.append(serializable)
        return sorted(records, key=lambda item: item["package_relative_path"])


def read_json(path: Path, registry: ArtifactRegistry, *, kind: str) -> dict[str, Any]:
    registry.register(path, kind=kind)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _declared_baseline_sha256(audit: dict[str, Any], path: Path) -> str:
    candidates = [
        audit.get("output_matrix_sha256"),
        (audit.get("output_sha256") or {}).get("imputed_expression.npy"),
        (audit.get("output_files") or {}).get("imputed_expression.npy", {}).get(
            "sha256"
        ),
        (audit.get("outputs") or {}).get("imputed_expression.npy", {}).get(
            "sha256"
        ),
        (audit.get("output") or {}).get("prediction", {}).get("sha256"),
    ]
    hashes = {str(value) for value in candidates if value}
    if len(hashes) != 1:
        raise ValueError(
            f"Could not resolve one declared imputed-expression SHA-256 from {path}"
        )
    return hashes.pop()


def load_hbc_dataset_config(
    config_path: Path,
    *,
    project_root: Path,
    registry: ArtifactRegistry,
) -> tuple[Path, dict[str, Any]]:
    config = read_json(config_path, registry, kind="protocol_a_dataset_config")
    datasets = [
        item
        for item in config.get("datasets", [])
        if item.get("dataset_id") == DATASET_ID
    ]
    if len(datasets) != 1:
        raise ValueError(f"Expected one {DATASET_ID} entry in {config_path}")
    archive_setting = Path(config["archive"]["root"])
    archive_root = (
        archive_setting
        if archive_setting.is_absolute()
        else Path(project_root) / archive_setting
    ).resolve()
    if not archive_root.is_dir():
        relative_location = Path(datasets[0]["locations"])
        candidates = sorted(
            (Path(project_root).parent / "zenodo_upload").glob(
                f"*/{relative_location.as_posix()}"
            )
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Could not resolve one current Protocol A archive from the "
                f"configured location or reviewer archives: {archive_root}"
            )
        archive_root = candidates[0].parents[len(relative_location.parts) - 1]
    return archive_root, datasets[0]


def _validate_fixed_gene(split: dict[str, Any], spec: PanelSpec) -> int:
    if int(split.get("fold", -1)) != spec.fold:
        raise ValueError(f"Fold identity mismatch for {spec.gene}: {spec.fold}")
    genes = list(map(str, split.get("final_test_genes", [])))
    indices = [int(value) for value in split.get("final_test_gene_idx", [])]
    if len(genes) != len(indices):
        raise ValueError(f"Malformed final-test gene mapping for fold{spec.fold}")
    matches = [index for index, gene in zip(indices, genes) if gene == spec.gene]
    if matches != [spec.gene_idx]:
        raise ValueError(
            f"Fixed panel {spec.gene} must be outer-test gene index {spec.gene_idx} "
            f"in fold{spec.fold}; observed={matches}"
        )
    return spec.gene_idx


def _load_fold_contract(
    *,
    run_root: Path,
    fold: int,
    coordinate_path: Path,
    registry: ArtifactRegistry,
) -> dict[str, Any]:
    input_dir = run_root / "inputs" / DATASET_ID / f"fold{fold}"
    artifact_path = input_dir / "artifact_manifest.json"
    artifact = read_json(
        artifact_path, registry, kind="protocol_a_input_artifact_manifest"
    )
    if artifact.get("dataset_id") != DATASET_ID or int(artifact.get("fold", -1)) != fold:
        raise ValueError(f"Input artifact manifest identity mismatch: {artifact_path}")

    split_path = input_dir / "mode_a_split.json"
    truth_path = input_dir / "full_truth.npy"
    output_artifacts = artifact.get("output_artifacts", {})
    input_artifacts = artifact.get("input_artifacts", {})
    registry.register(
        split_path,
        kind="mode_a_outer_test_split",
        declared_sha256=output_artifacts["mode_a_split"]["sha256"],
    )
    registry.register(
        truth_path,
        kind="protocol_a_full_truth",
        declared_sha256=output_artifacts["full_truth"]["sha256"],
    )
    registry.register(
        coordinate_path,
        kind="hbc_spatial_coordinates",
        declared_sha256=input_artifacts["locations"]["sha256"],
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))

    selected_dir = (
        run_root
        / "evaluation"
        / FORMAL_GENESPT_LAYER
        / "test_predictions"
        / DATASET_ID
        / f"fold{fold}"
    )
    apply_path = selected_dir / "apply_manifest.json"
    apply_manifest = read_json(
        apply_path, registry, kind="validation_selected_apply_manifest"
    )
    identity = apply_manifest.get("identity", {})
    if identity.get("dataset_id") != DATASET_ID or int(identity.get("fold", -1)) != fold:
        raise ValueError(f"Selected-readout manifest identity mismatch: {apply_path}")
    selected_path = selected_dir / "GeneSPT.npz"
    selected_output = apply_manifest.get("outputs", {}).get("GeneSPT", {})
    registry.register(
        selected_path,
        kind="validation_selected_genespt_prediction",
        declared_sha256=selected_output.get("sha256"),
    )

    baseline_paths: dict[str, Path] = {}
    for method in EXTERNAL_METHODS:
        baseline_dir = run_root / "baselines" / method / DATASET_ID / f"fold{fold}"
        audit_path = baseline_dir / "adapter_run_audit.json"
        audit = read_json(audit_path, registry, kind=f"{method}_adapter_run_audit")
        if audit.get("adapter") not in (None, method):
            raise ValueError(f"Baseline adapter identity mismatch: {audit_path}")
        baseline_path = baseline_dir / "imputed_expression.npy"
        registry.register(
            baseline_path,
            kind=f"{method}_raw_full_matrix",
            declared_sha256=_declared_baseline_sha256(audit, audit_path),
        )
        baseline_paths[method] = baseline_path

    return {
        "split": split,
        "split_path": split_path,
        "truth_path": truth_path,
        "selected_path": selected_path,
        "baseline_paths": baseline_paths,
    }


def _finite_vector(values: np.ndarray, *, context: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"Expected a one-dimensional vector for {context}: {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"Non-finite values found in {context}")
    return vector.copy()


def load_truth_vector(path: Path, gene_idx: int) -> np.ndarray:
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2 or not 0 <= gene_idx < matrix.shape[1]:
        raise ValueError(f"Invalid truth matrix or gene index for {path}: {matrix.shape}")
    return _finite_vector(matrix[:, gene_idx], context=f"truth {path} column {gene_idx}")


def load_selected_genespt_vector(
    path: Path,
    *,
    spec: PanelSpec,
) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as selected:
        required = {"prediction", "test_gene_idx", "method", "dataset_id", "fold"}
        if not required.issubset(selected.files):
            raise ValueError(f"Selected GeneSPT NPZ is missing keys at {path}")
        if str(selected["method"].item()) != "GeneSPT":
            raise ValueError(f"Selected NPZ method identity mismatch: {path}")
        if str(selected["dataset_id"].item()) != DATASET_ID:
            raise ValueError(f"Selected NPZ dataset identity mismatch: {path}")
        if int(selected["fold"].item()) != spec.fold:
            raise ValueError(f"Selected NPZ fold identity mismatch: {path}")
        test_gene_idx = np.asarray(selected["test_gene_idx"], dtype=np.int64)
        positions = np.flatnonzero(test_gene_idx == spec.gene_idx)
        if positions.size != 1:
            raise ValueError(
                f"Expected one selected prediction column for {spec.gene} in {path}"
            )
        position = int(positions[0])
        prediction = np.asarray(selected["prediction"])
        if prediction.ndim != 2 or prediction.shape[1] != test_gene_idx.size:
            raise ValueError(f"Selected prediction shape mismatch: {path}")
        vector = _finite_vector(
            prediction[:, position], context=f"selected GeneSPT {spec.gene}"
        )
    return vector, position


def load_baseline_vector(
    path: Path,
    gene_idx: int,
    *,
    method: str,
    test_gene_idx: np.ndarray,
) -> tuple[np.ndarray, int, str]:
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Invalid {method} prediction matrix: {path}")
    if matrix.shape[1] == len(test_gene_idx):
        positions = np.flatnonzero(test_gene_idx == gene_idx)
        if positions.size != 1:
            raise ValueError(
                f"Expected one frozen test-gene position for {method}: {gene_idx}"
            )
        column_index = int(positions[0])
        semantics = "frozen_test_gene_position"
    elif 0 <= gene_idx < matrix.shape[1]:
        column_index = gene_idx
        semantics = "global_gene_idx"
    else:
        raise ValueError(f"Invalid {method} matrix or gene index: {path}")
    vector = _finite_vector(
        matrix[:, column_index],
        context=f"{method} {path} column {column_index}",
    )
    return vector, column_index, semantics


def normalize_panel(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Apply independent 2nd-98th percentile clipping and min-max scaling."""

    vector = _finite_vector(values, context="visualization panel")
    low, high = np.percentile(vector, [2.0, 98.0])
    low, high = float(low), float(high)
    if high <= low:
        return np.zeros_like(vector, dtype=np.float32), low, high
    normalized = np.clip((vector - low) / (high - low), 0.0, 1.0)
    return normalized.astype(np.float32), low, high


def _source_record(
    *,
    row_index: int,
    column_index: int,
    spec: PanelSpec,
    method: str,
    source_path: Path,
    source_record: dict[str, Any],
    source_array: str,
    source_column_index: int,
    source_column_semantics: str,
    coordinate_record: dict[str, Any],
    raw_values: np.ndarray,
    visual_low: float,
    visual_high: float,
) -> dict[str, Any]:
    if method == "Ground truth":
        result_layer = "protocol_a_full_truth"
    elif method == "GeneSPT":
        result_layer = FORMAL_GENESPT_LAYER
    else:
        result_layer = RAW_BASELINE_LAYER
    return {
        "figure": "Figure 4",
        "panel_id": f"R{row_index + 1}C{column_index + 1}",
        "panel_row": row_index + 1,
        "panel_column": column_index + 1,
        "dataset_id": DATASET_ID,
        "gene": spec.gene,
        "fold": spec.fold,
        "gene_idx": spec.gene_idx,
        "pattern_type": spec.pattern_type,
        "method": method,
        "result_layer": result_layer,
        "source_package_relative_path": source_record["package_relative_path"],
        "source_sha256": source_record["sha256"],
        "source_array": source_array,
        "source_column_index": source_column_index,
        "source_column_semantics": source_column_semantics,
        "coordinate_package_relative_path": coordinate_record[
            "package_relative_path"
        ],
        "coordinate_sha256": coordinate_record["sha256"],
        "spot_count": int(raw_values.size),
        "raw_min": float(np.min(raw_values)),
        "raw_max": float(np.max(raw_values)),
        "visual_clip_lower_percentile": 2.0,
        "visual_clip_upper_percentile": 98.0,
        "visual_vmin": visual_low,
        "visual_vmax": visual_high,
        "visualization_transform": (
            "independent 2nd-98th percentile clipping then min-max scaling to [0,1]"
        ),
        "quantitative_metric_annotation": False,
    }


def collect_panel_data(
    *,
    run_root: Path = RUN_ROOT,
    config_path: Path = CONFIG_PATH,
    project_root: Path = PROJECT_ROOT,
    package_root: Path = PACKAGE_ROOT,
    panel_specs: Sequence[PanelSpec] = FIXED_PANELS,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, ArtifactRegistry]:
    """Load fixed panel vectors and return normalized rows plus source metadata."""

    if tuple(panel_specs) != FIXED_PANELS:
        raise ValueError("Figure 4 gene panels are frozen and cannot be replaced.")
    registry = ArtifactRegistry(Path(package_root))
    archive_root, dataset_config = load_hbc_dataset_config(
        Path(config_path), project_root=Path(project_root), registry=registry
    )
    coordinate_path = archive_root.joinpath(
        *Path(dataset_config["locations"]).parts
    ).resolve(strict=True)

    fold_contracts = {
        fold: _load_fold_contract(
            run_root=Path(run_root),
            fold=fold,
            coordinate_path=coordinate_path,
            registry=registry,
        )
        for fold in sorted({spec.fold for spec in panel_specs})
    }
    coordinates = pd.read_csv(coordinate_path, sep="\t")
    if list(coordinates.columns) != ["x", "y"]:
        raise ValueError(f"HBC coordinates must have x/y columns: {coordinate_path}")
    if not np.isfinite(coordinates[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError(f"HBC coordinates contain non-finite values: {coordinate_path}")
    coordinate_record = registry.register(
        coordinate_path, kind="hbc_spatial_coordinates"
    )

    figure_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for row_index, spec in enumerate(panel_specs):
        contract = fold_contracts[spec.fold]
        _validate_fixed_gene(contract["split"], spec)
        truth = load_truth_vector(contract["truth_path"], spec.gene_idx)
        genespt, selected_position = load_selected_genespt_vector(
            contract["selected_path"], spec=spec
        )
        raw_arrays: dict[str, np.ndarray] = {
            "Ground truth": truth,
            "GeneSPT": genespt,
        }
        baseline_columns: dict[str, tuple[int, str]] = {}
        frozen_test_idx = np.asarray(
            contract["split"]["final_test_gene_idx"],
            dtype=np.int64,
        )
        for method in EXTERNAL_METHODS:
            vector, source_column_index, column_semantics = load_baseline_vector(
                contract["baseline_paths"][method],
                spec.gene_idx,
                method=method,
                test_gene_idx=frozen_test_idx,
            )
            raw_arrays[method] = vector
            baseline_columns[method] = (source_column_index, column_semantics)
        spot_counts = {values.size for values in raw_arrays.values()}
        if spot_counts != {len(coordinates)}:
            raise ValueError(
                f"Spot-axis mismatch for {spec.gene}: arrays={spot_counts}, "
                f"coordinates={len(coordinates)}"
            )

        normalized_arrays: dict[str, np.ndarray] = {}
        for column_index, method in enumerate(METHOD_ORDER):
            raw_values = raw_arrays[method]
            normalized, visual_low, visual_high = normalize_panel(raw_values)
            normalized_arrays[method] = normalized
            if method == "Ground truth":
                source_path = contract["truth_path"]
                source_array = "full_truth"
                source_column_index = spec.gene_idx
                column_semantics = "global_gene_idx"
            elif method == "GeneSPT":
                source_path = contract["selected_path"]
                source_array = "prediction"
                source_column_index = selected_position
                column_semantics = "selected_test_gene_position"
            else:
                source_path = contract["baseline_paths"][method]
                source_array = (
                    "test_prediction"
                    if baseline_columns[method][1] == "frozen_test_gene_position"
                    else "imputed_expression"
                )
                source_column_index, column_semantics = baseline_columns[method]
            artifact_record = registry.register(
                source_path, kind=f"figure4_panel_source_{method}"
            )
            source_rows.append(
                _source_record(
                    row_index=row_index,
                    column_index=column_index,
                    spec=spec,
                    method=method,
                    source_path=source_path,
                    source_record=artifact_record,
                    source_array=source_array,
                    source_column_index=source_column_index,
                    source_column_semantics=column_semantics,
                    coordinate_record=coordinate_record,
                    raw_values=raw_values,
                    visual_low=visual_low,
                    visual_high=visual_high,
                )
            )
        figure_rows.append({"spec": spec, "arrays": normalized_arrays})

    source = pd.DataFrame(source_rows)
    if len(source) != 32:
        raise ValueError(f"Figure 4 source must contain exactly 32 rows, got {len(source)}")
    if tuple(source["gene"].drop_duplicates()) != tuple(
        spec.gene for spec in FIXED_PANELS
    ):
        raise ValueError("Figure 4 source gene order changed.")
    if tuple(source[source["panel_row"].eq(1)]["method"]) != METHOD_ORDER:
        raise ValueError("Figure 4 source method order changed.")
    return figure_rows, source, coordinates, registry


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    text_font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap a short panel label without changing the fixed grid geometry."""

    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        bounds = draw.textbbox((0, 0), candidate, font=text_font)
        if current and bounds[2] - bounds[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def viridis(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    position = clipped * (len(VIRIDIS) - 1)
    lower = np.floor(position).astype(int)
    upper = np.clip(lower + 1, 0, len(VIRIDIS) - 1)
    weight = (position - lower)[..., None]
    return (VIRIDIS[lower] * (1.0 - weight) + VIRIDIS[upper] * weight).astype(
        np.uint8
    )


def coordinate_pixels(
    coordinates: pd.DataFrame, size: int, *, padding: int = 18
) -> tuple[np.ndarray, np.ndarray]:
    x = coordinates["x"].to_numpy(dtype=np.float32)
    y = coordinates["y"].to_numpy(dtype=np.float32)
    x_pixels = padding + (x - x.min()) / max(float(np.ptp(x)), 1.0) * (
        size - 2 * padding
    )
    y_pixels = padding + (y - y.min()) / max(float(np.ptp(y)), 1.0) * (
        size - 2 * padding
    )
    return x_pixels, y_pixels


def draw_map(
    draw: ImageDraw.ImageDraw,
    *,
    x_origin: int,
    y_origin: int,
    size: int,
    coordinate_pixel_values: tuple[np.ndarray, np.ndarray],
    values: np.ndarray,
    radius: int,
) -> None:
    x_pixels, y_pixels = coordinate_pixel_values
    colors = viridis(values)
    for index in np.argsort(values):
        center_x = int(round(x_origin + x_pixels[index]))
        center_y = int(round(y_origin + y_pixels[index]))
        color = tuple(int(value) for value in colors[index])
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=color,
        )
    draw.rectangle(
        (x_origin, y_origin, x_origin + size, y_origin + size),
        outline=(228, 228, 228),
        width=1,
    )


def render_grid(
    rows: list[dict[str, Any]],
    coordinates: pd.DataFrame,
    *,
    png_path: Path,
    pdf_path: Path,
) -> None:
    map_size = 360
    label_width = 225
    column_gap = 22
    row_gap = 52
    top, left, right, bottom = 78, 34, 40, 42
    width = (
        left
        + label_width
        + len(METHOD_ORDER) * map_size
        + (len(METHOD_ORDER) - 1) * column_gap
        + right
    )
    height = top + len(rows) * map_size + (len(rows) - 1) * row_gap + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    header_font = font(25, bold=True)
    gene_font = font(28, bold=True)
    role_font = font(17)
    small_font = font(17)
    pixels = coordinate_pixels(coordinates, map_size)

    for column_index, method in enumerate(METHOD_ORDER):
        x_origin = left + label_width + column_index * (map_size + column_gap)
        box = draw.textbbox((0, 0), method, font=header_font)
        text_width = box[2] - box[0]
        draw.text(
            (x_origin + map_size / 2 - text_width / 2, 26),
            method,
            fill=(25, 25, 25),
            font=header_font,
        )

    for row_index, row in enumerate(rows):
        spec: PanelSpec = row["spec"]
        y_origin = top + row_index * (map_size + row_gap)
        label_y = y_origin + map_size * 0.36
        draw.text((left, label_y), spec.gene, fill=(20, 20, 20), font=gene_font)
        pattern_y = label_y + 42
        pattern_lines = wrap_text(
            draw,
            spec.pattern_type,
            text_font=role_font,
            max_width=label_width - 18,
        )
        line_height = 22
        for line_index, line in enumerate(pattern_lines):
            draw.text(
                (left, pattern_y + line_index * line_height),
                line,
                fill=(88, 88, 88),
                font=role_font,
            )
        draw.text(
            (left, pattern_y + len(pattern_lines) * line_height + 8),
            f"fold{spec.fold} test gene",
            fill=(105, 105, 105),
            font=small_font,
        )
        for column_index, method in enumerate(METHOD_ORDER):
            x_origin = left + label_width + column_index * (map_size + column_gap)
            draw_map(
                draw,
                x_origin=x_origin,
                y_origin=y_origin,
                size=map_size,
                coordinate_pixel_values=pixels,
                values=row["arrays"][method],
                radius=max(2, int(round(map_size / 125))),
            )

    image.save(png_path, dpi=(300, 300))
    image.save(pdf_path, "PDF", resolution=300.0)


def write_manifest(
    *,
    manifest_path: Path,
    output_paths: Sequence[Path],
    registry: ArtifactRegistry,
    source_row_count: int,
    package_root: Path,
) -> None:
    generator_path = Path(__file__).resolve(strict=True)
    payload = {
        "schema_version": 1,
        "figure": "Figure 4",
        "protocol": "A",
        "dataset_id": DATASET_ID,
        "genespt_result_layer": FORMAL_GENESPT_LAYER,
        "external_baseline_result_layer": RAW_BASELINE_LAYER,
        "gene_panels": [asdict(spec) for spec in FIXED_PANELS],
        "method_order": list(METHOD_ORDER),
        "layout": {"rows": 4, "columns": 8},
        "visualization_policy": {
            "scope": "independent_per_panel",
            "clip_percentiles": [2.0, 98.0],
            "post_clip_transform": "min-max to [0,1]",
            "quantitative_metric_annotation": False,
        },
        "source_data_row_count": source_row_count,
        "generator": {
            "package_relative_path": package_relative_path(
                generator_path, package_root
            ),
            "sha256": sha256_file(generator_path),
        },
        "inputs": registry.manifest_records(),
        "outputs": [
            {
                "package_relative_path": package_relative_path(path, package_root),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate(
    *,
    run_root: Path = RUN_ROOT,
    config_path: Path = CONFIG_PATH,
    output_dir: Path = OUTPUT_DIR,
    project_root: Path = PROJECT_ROOT,
    package_root: Path = PACKAGE_ROOT,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{FIGURE_STEM}.png"
    pdf_path = output_dir / f"{FIGURE_STEM}.pdf"
    source_path = output_dir / f"{FIGURE_STEM}_source.csv"
    manifest_path = output_dir / f"{FIGURE_STEM}_manifest.json"

    rows, source, coordinates, registry = collect_panel_data(
        run_root=Path(run_root),
        config_path=Path(config_path),
        project_root=Path(project_root),
        package_root=Path(package_root),
    )
    source.to_csv(source_path, index=False)
    render_grid(rows, coordinates, png_path=png_path, pdf_path=pdf_path)
    write_manifest(
        manifest_path=manifest_path,
        output_paths=(png_path, pdf_path, source_path),
        registry=registry,
        source_row_count=len(source),
        package_root=Path(package_root),
    )
    return {
        "png": png_path,
        "pdf": pdf_path,
        "source": source_path,
        "manifest": manifest_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = generate(
        run_root=args.run_root,
        config_path=args.config,
        output_dir=args.output_dir,
        project_root=args.project_root,
        package_root=args.package_root,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
