#!/usr/bin/env python3
"""Recompute the archived Cell2location strict GC versus GC+PSP comparison.

The script consumes evaluator-ready truth plus the two published five-fold NPZ
trees. It verifies the matched-toggle invariants before computing any summary.
No model is trained and no archive is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genespt.io import load_array, load_gene_names  # noqa: E402
from genespt.metrics import evaluate_prediction  # noqa: E402


DATASET = "Cell2location mouse brain"
DATASET_ID = "Cell2location_mouse_brain_ST8059048_shared12819"
GC_METHOD = "GeneSPT-GC"
PSP_METHOD = "GeneSPT-GC+PSP"
METHODS = (GC_METHOD, PSP_METHOD)
INTERNAL_MODELS = {
    GC_METHOD: "gc_mlp_base",
    PSP_METHOD: "predictable_spatial_program_selected_correct",
}
METRICS = ("SPCC", "RMSE", "JSD", "SSIM")


@dataclass
class FoldArchive:
    method: str
    fold: int
    path: Path
    metadata_path: Path
    prediction: np.ndarray
    base_prediction: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    test_genes: list[str]
    model: str
    readout: str
    posthoc_calibration: str
    metadata: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(saved: Any, key: str) -> Any:
    value = np.asarray(saved[key])
    _require(value.size == 1, f"NPZ field {key!r} must be scalar; got {value.shape}")
    return value.reshape(()).item()


def _load_fold_archive(root: Path, method: str, fold: int) -> FoldArchive:
    fold_dir = root / method / f"fold{fold}"
    path = fold_dir / "prediction.npz"
    metadata_path = fold_dir / "metadata.json"
    _require(path.is_file(), f"Missing prediction archive: {path}")
    _require(metadata_path.is_file(), f"Missing prediction metadata: {metadata_path}")

    required = {
        "prediction",
        "base_prediction",
        "train_gene_idx",
        "val_gene_idx",
        "test_gene_idx",
        "test_genes",
        "model",
        "fold",
        "readout",
        "posthoc_calibration",
    }
    with np.load(path, allow_pickle=True) as saved:
        missing = sorted(required.difference(saved.files))
        _require(not missing, f"{path} is missing NPZ fields: {missing}")
        archive = FoldArchive(
            method=method,
            fold=fold,
            path=path,
            metadata_path=metadata_path,
            prediction=np.asarray(saved["prediction"]),
            base_prediction=np.asarray(saved["base_prediction"]),
            train_idx=np.asarray(saved["train_gene_idx"], dtype=np.int64).reshape(-1),
            val_idx=np.asarray(saved["val_gene_idx"], dtype=np.int64).reshape(-1),
            test_idx=np.asarray(saved["test_gene_idx"], dtype=np.int64).reshape(-1),
            test_genes=[
                str(value) for value in np.asarray(saved["test_genes"]).tolist()
            ],
            model=str(_scalar(saved, "model")),
            readout=str(_scalar(saved, "readout")),
            posthoc_calibration=str(_scalar(saved, "posthoc_calibration")),
            metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        )
        saved_fold = int(_scalar(saved, "fold"))
    _require(saved_fold == fold, f"{path} stores fold {saved_fold}, expected {fold}")

    external_test_idx = fold_dir / "test_gene_idx.npy"
    if external_test_idx.is_file():
        published = np.asarray(np.load(external_test_idx), dtype=np.int64).reshape(-1)
        _require(
            np.array_equal(published, archive.test_idx),
            f"External test indices disagree with {path}",
        )
    return archive


def _validate_partition(archive: FoldArchive, n_genes: int) -> None:
    split_arrays = {
        "train": archive.train_idx,
        "validation": archive.val_idx,
        "test": archive.test_idx,
    }
    for split_name, indices in split_arrays.items():
        _require(
            len(indices) == len(np.unique(indices)),
            f"{archive.method} fold{archive.fold} {split_name} indices contain duplicates",
        )
        _require(
            bool(np.all((indices >= 0) & (indices < n_genes))),
            f"{archive.method} fold{archive.fold} {split_name} indices are out of range",
        )
    combined = np.concatenate(list(split_arrays.values()))
    _require(
        len(combined) == n_genes
        and np.array_equal(np.sort(combined), np.arange(n_genes)),
        f"{archive.method} fold{archive.fold} train/validation/test do not partition all genes",
    )


def _validate_fold_pair(
    gc: FoldArchive,
    psp: FoldArchive,
    *,
    truth_shape: tuple[int, int],
    gene_names: Sequence[str],
    split_dir: Path | None,
    dataset_id: str,
) -> list[dict[str, Any]]:
    n_spots, n_genes = truth_shape
    checks: list[dict[str, Any]] = []

    for archive in (gc, psp):
        _validate_partition(archive, n_genes)
        expected_shape = (n_spots, len(archive.test_idx))
        _require(
            archive.prediction.shape == expected_shape,
            f"{archive.path} prediction shape {archive.prediction.shape} != {expected_shape}",
        )
        _require(
            archive.base_prediction.shape == expected_shape,
            f"{archive.path} base_prediction shape {archive.base_prediction.shape} != {expected_shape}",
        )
        _require(
            np.isfinite(archive.prediction).all()
            and np.isfinite(archive.base_prediction).all(),
            f"{archive.path} contains a nonfinite prediction",
        )
        _require(
            archive.model == INTERNAL_MODELS[archive.method],
            f"{archive.path} model {archive.model!r} is not {INTERNAL_MODELS[archive.method]!r}",
        )
        _require(
            archive.readout == "identity", f"{archive.path} readout is not identity"
        )
        _require(
            archive.posthoc_calibration == "none",
            f"{archive.path} uses post-hoc calibration {archive.posthoc_calibration!r}",
        )
        metadata = archive.metadata
        _require(
            metadata.get("method") == archive.method,
            f"Metadata method mismatch in {archive.metadata_path}",
        )
        _require(
            int(metadata.get("fold", -1)) == archive.fold,
            f"Metadata fold mismatch in {archive.metadata_path}",
        )
        _require(
            metadata.get("dataset_id") == dataset_id,
            f"Metadata dataset mismatch in {archive.metadata_path}",
        )
        _require(
            metadata.get("comparison") == "strict PSP toggle",
            f"Metadata comparison is not strict in {archive.metadata_path}",
        )
        _require(
            metadata.get("readout") == "identity",
            f"Metadata readout is not identity in {archive.metadata_path}",
        )
        _require(
            metadata.get("posthoc_calibration") == "none",
            f"Metadata calibration is not none in {archive.metadata_path}",
        )
        _require(
            metadata.get("gc_base_shared_between_settings") is True,
            f"Metadata does not declare a shared GC base in {archive.metadata_path}",
        )
        _require(
            list(metadata.get("shape", [])) == list(expected_shape),
            f"Metadata shape mismatch in {archive.metadata_path}",
        )
        expected_genes = [str(gene_names[int(index)]) for index in archive.test_idx]
        _require(
            archive.test_genes == expected_genes,
            f"Test-gene order mismatch in {archive.path}",
        )

    for split_name, gc_indices, psp_indices in (
        ("train", gc.train_idx, psp.train_idx),
        ("validation", gc.val_idx, psp.val_idx),
        ("test", gc.test_idx, psp.test_idx),
    ):
        _require(
            np.array_equal(gc_indices, psp_indices),
            f"Fold {gc.fold} {split_name} indices differ between GC and GC+PSP",
        )
        if split_dir is not None:
            frozen_path = (
                split_dir
                / f"fold{gc.fold}_{'val' if split_name == 'validation' else split_name}_gene_idx.npy"
            )
            _require(frozen_path.is_file(), f"Missing frozen split: {frozen_path}")
            frozen = np.asarray(np.load(frozen_path), dtype=np.int64).reshape(-1)
            _require(
                np.array_equal(gc_indices, frozen),
                f"Fold {gc.fold} {split_name} indices differ from {frozen_path}",
            )

    _require(
        np.array_equal(gc.prediction, gc.base_prediction),
        f"{gc.path} is not a pure GC identity prediction",
    )
    _require(
        np.array_equal(gc.prediction, psp.base_prediction),
        f"Fold {gc.fold} does not reuse the identical frozen GC base in GC+PSP",
    )

    checks.extend(
        [
            {
                "fold": gc.fold,
                "check": "same_frozen_gene_split",
                "passed": True,
                "detail": "train/validation/test arrays match both paths and published splits",
            },
            {
                "fold": gc.fold,
                "check": "same_gc_base_prediction",
                "passed": True,
                "detail": "GC prediction equals GC base_prediction and GC+PSP base_prediction exactly",
            },
            {
                "fold": gc.fold,
                "check": "identity_readout_no_posthoc_calibration",
                "passed": True,
                "detail": "both archives and metadata record identity/none",
            },
            {
                "fold": gc.fold,
                "check": "strict_method_labels",
                "passed": True,
                "detail": f"{GC_METHOD}=gc_mlp_base; {PSP_METHOD}=predictable_spatial_program_selected_correct",
            },
        ]
    )
    return checks


def _validate_metadata_metrics(
    archive: FoldArchive, summary: pd.Series, atol: float
) -> float:
    metadata_metrics = archive.metadata.get("metrics")
    if not isinstance(metadata_metrics, dict):
        return 0.0
    key_map = {"SPCC": "SPCC", "RMSE": "RMSE", "JSD": "JS/JSD", "SSIM": "SSIM"}
    differences = []
    for metric, metadata_key in key_map.items():
        expected = float(metadata_metrics[metadata_key])
        observed = float(summary[metric])
        differences.append(abs(observed - expected))
        _require(
            math.isclose(observed, expected, abs_tol=atol, rel_tol=0.0),
            f"{archive.method} fold{archive.fold} {metric} differs from metadata: {observed} vs {expected}",
        )

    metadata_coverage = archive.metadata.get("coverage", {})
    for metric in METRICS:
        if metric not in metadata_coverage:
            continue
        expected = metadata_coverage[metric]
        for suffix in ("eligible", "scored"):
            _require(
                int(summary[f"{metric}_{suffix}"]) == int(expected[suffix]),
                f"{archive.method} fold{archive.fold} {metric} {suffix} differs from metadata",
            )
        _require(
            math.isclose(
                float(summary[f"{metric}_coverage"]),
                float(expected["coverage"]),
                abs_tol=atol,
                rel_tol=0.0,
            ),
            f"{archive.method} fold{archive.fold} {metric} coverage differs from metadata",
        )
    return max(differences, default=0.0)


def _fold_improvements(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    gc = fold_metrics[fold_metrics["method"].eq(GC_METHOD)].set_index("fold")
    psp = fold_metrics[fold_metrics["method"].eq(PSP_METHOD)].set_index("fold")
    _require(gc.index.equals(psp.index), "GC and GC+PSP fold sets differ")
    return pd.DataFrame(
        {
            "fold": gc.index.astype(int),
            "delta_SPCC": psp["SPCC"] - gc["SPCC"],
            "RMSE_improvement": gc["RMSE"] - psp["RMSE"],
            "JSD_improvement": gc["JSD"] - psp["JSD"],
            "SSIM_improvement": psp["SSIM"] - gc["SSIM"],
        }
    ).reset_index(drop=True)


def _comparison_summary(fold_metrics: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        group = fold_metrics[fold_metrics["method"].eq(method)]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "SPCC_mean": float(group["SPCC"].mean()),
                "SPCC_std": float(group["SPCC"].std(ddof=1)),
                "RMSE_mean": float(group["RMSE"].mean()),
                "RMSE_std": float(group["RMSE"].std(ddof=1)),
                "JS_JSD_mean": float(group["JSD"].mean()),
                "JS_JSD_std": float(group["JSD"].std(ddof=1)),
                "SSIM_mean": float(group["SSIM"].mean()),
                "SSIM_std": float(group["SSIM"].std(ddof=1)),
                "folds": int(group["fold"].nunique()),
                "readout": "identity",
                "posthoc_calibration": "none",
            }
        )
    return pd.DataFrame(rows)


def _paired_tests(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    gc = fold_metrics[fold_metrics["method"].eq(GC_METHOD)].sort_values("fold")
    psp = fold_metrics[fold_metrics["method"].eq(PSP_METHOD)].sort_values("fold")
    rows = []
    for metric in METRICS:
        base = gc[metric].to_numpy(float)
        model = psp[metric].to_numpy(float)
        improvement = model - base if metric in {"SPCC", "SSIM"} else base - model
        n_folds = len(improvement)
        mean = float(np.mean(improvement))
        if n_folds > 1:
            sd = float(np.std(improvement, ddof=1))
            se = sd / math.sqrt(n_folds)
            critical = float(stats.t.ppf(0.975, n_folds - 1))
            ci_low, ci_high = mean - critical * se, mean + critical * se
        else:
            sd = se = ci_low = ci_high = float("nan")
        if np.allclose(improvement, 0.0, atol=0.0, rtol=0.0):
            paired_t_p = wilcoxon_p = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                paired_t_p = float(stats.ttest_1samp(improvement, popmean=0.0).pvalue)
            wilcoxon_p = float(stats.wilcoxon(improvement, method="exact").pvalue)
        rows.append(
            {
                "metric": metric,
                "n_folds": n_folds,
                "GeneSPT-GC_mean": float(np.mean(base)),
                "GeneSPT-GC+PSP_mean": float(np.mean(model)),
                "mean_improvement": mean,
                "median_improvement": float(np.median(improvement)),
                "improvement_sd": sd,
                "improvement_se": se,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_t_p": paired_t_p,
                "wilcoxon_p": wilcoxon_p,
                "improvement_definition": (
                    "GC+PSP minus GC"
                    if metric in {"SPCC", "SSIM"}
                    else "GC minus GC+PSP"
                ),
            }
        )
    return pd.DataFrame(rows)


def _validate_reference_fold_metrics(
    observed: pd.DataFrame,
    reference_path: Path,
    folds: Sequence[int],
    atol: float,
) -> float:
    reference = pd.read_csv(reference_path)
    reference = reference[
        reference["method"].isin(METHODS) & reference["fold"].isin(folds)
    ]
    keys = ["method", "fold"]
    columns = [
        "SPCC",
        "RMSE",
        "SSIM",
        "JSD",
        "SPCC_eligible",
        "SPCC_scored",
        "RMSE_eligible",
        "RMSE_scored",
        "SSIM_eligible",
        "SSIM_scored",
        "JSD_eligible",
        "JSD_scored",
    ]
    missing = [column for column in keys + columns if column not in reference.columns]
    _require(not missing, f"Reference fold table is missing columns: {missing}")
    merged = observed[keys + columns].merge(
        reference[keys + columns],
        on=keys,
        suffixes=("_observed", "_reference"),
        validate="one_to_one",
    )
    _require(
        len(merged) == len(METHODS) * len(folds), "Reference fold table is incomplete"
    )
    differences = []
    for column in columns:
        delta = np.abs(
            merged[f"{column}_observed"].to_numpy(float)
            - merged[f"{column}_reference"].to_numpy(float)
        )
        differences.extend(delta.tolist())
        _require(
            bool(np.all(delta <= atol)),
            f"Reference mismatch for {column}; maximum absolute difference={float(np.max(delta))}",
        )
    return max(differences, default=0.0)


def run_comparison(
    *,
    truth_path: Path,
    gene_names_path: Path,
    prediction_root: Path,
    split_dir: Path | None,
    output_dir: Path,
    folds: Sequence[int],
    dataset: str = DATASET,
    dataset_id: str = DATASET_ID,
    reference_fold_metrics: Path | None = None,
    write_gene_level: bool = False,
    require_complete_test_partition: bool = True,
    atol: float = 1e-8,
) -> dict[str, Any]:
    truth = np.asarray(load_array(truth_path))
    _require(truth.ndim == 2, f"Truth matrix must be spots x genes; got {truth.shape}")
    gene_names = load_gene_names(gene_names_path)
    _require(gene_names is not None, "Gene names are required")
    _require(
        len(gene_names) == truth.shape[1],
        "Gene name count does not match truth columns",
    )
    _require(len(folds) == len(set(folds)), "Fold list contains duplicates")

    fold_metric_frames: list[pd.DataFrame] = []
    gene_level_frames: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    test_indices_by_fold: list[np.ndarray] = []
    input_paths: list[Path] = [truth_path, gene_names_path]
    maximum_metadata_difference = 0.0

    for fold in folds:
        gc = _load_fold_archive(prediction_root, GC_METHOD, int(fold))
        psp = _load_fold_archive(prediction_root, PSP_METHOD, int(fold))
        validation_rows.extend(
            _validate_fold_pair(
                gc,
                psp,
                truth_shape=tuple(map(int, truth.shape)),
                gene_names=gene_names,
                split_dir=split_dir,
                dataset_id=dataset_id,
            )
        )
        test_indices_by_fold.append(gc.test_idx)
        test_truth = truth[:, gc.test_idx]
        test_names = [gene_names[int(index)] for index in gc.test_idx]
        method_per_gene: dict[str, pd.DataFrame] = {}

        for archive in (gc, psp):
            per_gene, summary = evaluate_prediction(
                test_truth,
                archive.prediction,
                test_names,
            )
            maximum_metadata_difference = max(
                maximum_metadata_difference,
                _validate_metadata_metrics(archive, summary.iloc[0], atol),
            )
            per_gene.insert(0, "method", archive.method)
            per_gene.insert(1, "fold", int(fold))
            per_gene.insert(2, "global_gene_idx", archive.test_idx.astype(int))
            summary.insert(0, "method", archive.method)
            summary.insert(1, "fold", int(fold))
            method_per_gene[archive.method] = per_gene
            gene_level_frames.append(per_gene)
            fold_metric_frames.append(summary)

        eligibility_columns = [
            "global_gene_idx",
            "truth_finite",
            "truth_constant",
            "truth_zero_mass_after_nonnegative_clipping",
            "SPCC_eligible",
            "RMSE_eligible",
            "SSIM_eligible",
            "JSD_eligible",
        ]
        _require(
            method_per_gene[GC_METHOD][eligibility_columns]
            .reset_index(drop=True)
            .equals(
                method_per_gene[PSP_METHOD][eligibility_columns].reset_index(drop=True)
            ),
            f"Truth-defined eligibility differs between methods in fold {fold}",
        )
        validation_rows.append(
            {
                "fold": int(fold),
                "check": "same_truth_defined_eligibility",
                "passed": True,
                "detail": "all metric eligibility masks match",
            }
        )
        for archive in (gc, psp):
            input_paths.extend([archive.path, archive.metadata_path])
            external_idx = archive.path.parent / "test_gene_idx.npy"
            if external_idx.is_file():
                input_paths.append(external_idx)
        if split_dir is not None:
            for split_name in ("train", "val", "test"):
                input_paths.append(split_dir / f"fold{fold}_{split_name}_gene_idx.npy")

    if require_complete_test_partition:
        combined_test = np.concatenate(test_indices_by_fold)
        _require(
            len(combined_test) == truth.shape[1]
            and np.array_equal(np.sort(combined_test), np.arange(truth.shape[1])),
            "Requested fold test sets do not partition every truth gene exactly once",
        )
        validation_rows.append(
            {
                "fold": "all",
                "check": "test_folds_partition_all_genes",
                "passed": True,
                "detail": f"{len(combined_test)} unique test assignments for {truth.shape[1]} genes",
            }
        )

    fold_metrics = pd.concat(fold_metric_frames, ignore_index=True)
    gene_level = pd.concat(gene_level_frames, ignore_index=True)
    improvements = _fold_improvements(fold_metrics)
    summary = _comparison_summary(fold_metrics, dataset)
    paired = _paired_tests(fold_metrics)

    maximum_reference_difference: float | None = None
    if reference_fold_metrics is not None:
        maximum_reference_difference = _validate_reference_fold_metrics(
            fold_metrics,
            reference_fold_metrics,
            folds,
            atol,
        )
        input_paths.append(reference_fold_metrics)
        validation_rows.append(
            {
                "fold": "all",
                "check": "reference_fold_metrics_reproduced",
                "passed": True,
                "detail": f"maximum absolute difference={maximum_reference_difference:.3g}",
            }
        )

    validation_rows.append(
        {
            "fold": "all",
            "check": "metadata_metrics_reproduced",
            "passed": True,
            "detail": f"maximum absolute difference={maximum_metadata_difference:.3g}",
        }
    )
    validation = pd.DataFrame(validation_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "complete_set_fold_metrics.csv", index=False)
    improvements.to_csv(output_dir / "complete_set_fold_improvements.csv", index=False)
    summary.to_csv(output_dir / "cell2location_strict_psp_summary.csv", index=False)
    paired.to_csv(output_dir / "cell2location_strict_psp_paired_tests.csv", index=False)
    validation.to_csv(output_dir / "strict_validation.csv", index=False)
    if write_gene_level:
        gene_level.to_csv(
            output_dir / "complete_set_gene_level_metrics.csv", index=False
        )

    unique_paths = sorted({path.resolve() for path in input_paths}, key=str)
    inventory = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in unique_paths
    ]
    manifest = {
        "dataset": dataset,
        "dataset_id": dataset_id,
        "comparison": "strict PSP toggle",
        "methods": list(METHODS),
        "invariants": {
            "same_gc_base_prediction": True,
            "same_frozen_train_validation_test_indices": True,
            "readout": "identity",
            "posthoc_calibration": "none",
            "legacy_gc_plus_psp_must_not_be_labeled_gc_only": True,
        },
        "metric_policy": "docs/metric_policy.md",
        "metric_comparison_absolute_tolerance": atol,
        "folds": [int(fold) for fold in folds],
        "write_gene_level": bool(write_gene_level),
        "maximum_metadata_metric_difference": maximum_metadata_difference,
        "maximum_reference_metric_difference": maximum_reference_difference,
        "inputs": inventory,
    }
    (output_dir / "strict_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "folds": len(folds),
        "method_fold_rows": len(fold_metrics),
        "gene_level_rows": len(gene_level),
        "maximum_metadata_metric_difference": maximum_metadata_difference,
        "maximum_reference_metric_difference": maximum_reference_difference,
        "output_dir": str(output_dir),
    }


def _write_synthetic_archive(
    root: Path,
    *,
    method: str,
    fold: int,
    prediction: np.ndarray,
    base_prediction: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    gene_names: Sequence[str],
) -> None:
    fold_dir = root / method / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fold_dir / "prediction.npz",
        prediction=prediction,
        base_prediction=base_prediction,
        train_gene_idx=train_idx,
        val_gene_idx=val_idx,
        test_gene_idx=test_idx,
        test_genes=np.asarray(
            [gene_names[int(index)] for index in test_idx], dtype=object
        ),
        model=np.asarray(INTERNAL_MODELS[method], dtype=object),
        fold=np.asarray(fold),
        readout=np.asarray("identity", dtype=object),
        posthoc_calibration=np.asarray("none", dtype=object),
    )
    np.save(fold_dir / "test_gene_idx.npy", test_idx)
    metadata = {
        "dataset": DATASET,
        "dataset_id": DATASET_ID,
        "method": method,
        "fold": fold,
        "comparison": "strict PSP toggle",
        "readout": "identity",
        "posthoc_calibration": "none",
        "gc_base_shared_between_settings": True,
        "shape": list(map(int, prediction.shape)),
    }
    (fold_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def run_self_test() -> dict[str, Any]:
    """Build a tiny two-fold bundle and exercise every strict invariant."""

    results_root = ROOT / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="_strict_psp_selftest_", dir=results_root
    ) as raw_tmp:
        tmp = Path(raw_tmp)
        truth = np.column_stack(
            [
                np.asarray([0.0, 1.0, 2.0, 3.0]),
                np.zeros(4, dtype=float),
                np.asarray([3.0, 1.0, 4.0, 2.0]),
                np.asarray([1.0, 3.0, 2.0, 4.0]),
            ]
        )
        gene_names = ["gene0", "gene1", "gene2", "gene3"]
        truth_path = tmp / "truth.npy"
        names_path = tmp / "gene_names.txt"
        prediction_root = tmp / "predictions"
        split_dir = tmp / "splits"
        output_dir = tmp / "output"
        np.save(truth_path, truth)
        names_path.write_text("\n".join(gene_names) + "\n", encoding="utf-8")
        split_dir.mkdir()

        folds = {
            0: (
                np.asarray([2], dtype=np.int64),
                np.asarray([3], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
            ),
            1: (
                np.asarray([0], dtype=np.int64),
                np.asarray([1], dtype=np.int64),
                np.asarray([2, 3], dtype=np.int64),
            ),
        }
        for fold, (train_idx, val_idx, test_idx) in folds.items():
            for split_name, indices in (
                ("train", train_idx),
                ("val", val_idx),
                ("test", test_idx),
            ):
                np.save(split_dir / f"fold{fold}_{split_name}_gene_idx.npy", indices)
            test_truth = truth[:, test_idx]
            gc_prediction = np.flip(test_truth, axis=0).copy()
            psp_prediction = test_truth.copy()
            for method, prediction in (
                (GC_METHOD, gc_prediction),
                (PSP_METHOD, psp_prediction),
            ):
                _write_synthetic_archive(
                    prediction_root,
                    method=method,
                    fold=fold,
                    prediction=prediction,
                    base_prediction=gc_prediction,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    gene_names=gene_names,
                )

        result = run_comparison(
            truth_path=truth_path,
            gene_names_path=names_path,
            prediction_root=prediction_root,
            split_dir=split_dir,
            output_dir=output_dir,
            folds=[0, 1],
            write_gene_level=True,
        )
        assert result["status"] == "PASS"
        assert result["method_fold_rows"] == 4
        assert result["gene_level_rows"] == 8
        assert (output_dir / "strict_validation.csv").is_file()
        return {key: value for key, value in result.items() if key != "output_dir"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--gene-names", type=Path, default=None)
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--reference-fold-metrics", type=Path, default=None)
    parser.add_argument("--write-gene-level", action="store_true")
    parser.add_argument("--allow-partial-folds", action="store_true")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for floating metric-table checks; indices and counts remain exact",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        print(json.dumps(run_self_test(), indent=2))
        return 0
    missing = [
        name
        for name, value in (
            ("--truth", args.truth),
            ("--gene-names", args.gene_names),
            ("--prediction-root", args.prediction_root),
            ("--out-dir", args.out_dir),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(
            f"Missing required arguments outside --self-test: {', '.join(missing)}"
        )
    _require(
        args.atol >= 0.0 and math.isfinite(args.atol),
        "--atol must be finite and nonnegative",
    )
    result = run_comparison(
        truth_path=args.truth,
        gene_names_path=args.gene_names,
        prediction_root=args.prediction_root,
        split_dir=args.split_dir,
        output_dir=args.out_dir,
        folds=args.folds,
        dataset=args.dataset,
        dataset_id=args.dataset_id,
        reference_fold_metrics=args.reference_fold_metrics,
        write_gene_level=args.write_gene_level,
        require_complete_test_partition=not args.allow_partial_folds,
        atol=args.atol,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
