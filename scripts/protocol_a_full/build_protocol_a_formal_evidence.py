#!/usr/bin/env python3
"""Build one audited source table for the formal Protocol A benchmark.

GeneSPT uses its locked validation-selected readout. External methods retain
their raw adapter outputs. The script never writes or modifies predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "protocol_a_full_rerun_20260711"
RAW_REPORT = RESULTS_ROOT / "evaluation" / "protocol_a_raw_evaluation_report.json"
READOUT_ROOT = RESULTS_ROOT / "evaluation" / "validation_selected_readout_genespt57"
CONFIG_PATH = PROJECT_ROOT / "configs" / "protocol_a_datasets.formal.yaml"
METRICS_PATH = PROJECT_ROOT / "src" / "genespt" / "metrics.py"
OUTPUT_ROOT = RESULTS_ROOT / "evaluation" / "formal_benchmark_evidence"
STAI_ROOT = RESULTS_ROOT / "baselines" / "stAI"
STAI_REFERENCE = (
    PROJECT_ROOT
    / "results"
    / "source_data"
    / "protocol_a"
    / "benchmark"
    / "formal_five_fold_metrics.csv"
)

FORMAL_METHODS = (
    "GeneSPT",
    "Tangram",
    "TransImp",
    "SpaIM",
    "SpaGE",
    "stPlus",
    "stAI",
)
EXTERNAL_METHODS = FORMAL_METHODS[1:]
METRICS = ("SPCC", "RMSE", "JSD", "SSIM")
LOWER_IS_BETTER = {"RMSE", "JSD"}


class EvidenceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_path(value: str) -> Path:
    normalized = str(value).replace("\\", "/")
    if normalized.startswith("/workspace/GeneSPT/"):
        return PROJECT_ROOT / normalized.removeprefix("/workspace/GeneSPT/")
    if normalized.startswith("/workspace/"):
        return WORKSPACE_ROOT / normalized.removeprefix("/workspace/")
    return Path(value)


def load_metrics_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("genespt_formal_metrics", path)
    if spec is None or spec.loader is None:
        raise EvidenceError(f"Cannot import centralized evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "evaluate_prediction", None)):
        raise EvidenceError("Centralized evaluator has no evaluate_prediction")
    return module


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "A":
        raise EvidenceError("Expected Protocol A configuration")
    return payload


def load_gene_names(config: dict[str, Any]) -> dict[str, list[str]]:
    archive_root = (PROJECT_ROOT / config["archive"]["root"]).resolve()
    result: dict[str, list[str]] = {}
    for dataset in config["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        path = archive_root / dataset["gene_names"]
        expected = int(dataset["expected_st_shape"][1])
        if path.is_file():
            genes = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            split_path = (
                RESULTS_ROOT
                / "inputs"
                / dataset_id
                / "fold0"
                / "mode_a_split.json"
            )
            split = json.loads(split_path.read_text(encoding="utf-8"))
            genes_by_index: list[str | None] = [None] * expected
            for indices_key, names_key in (
                ("inner_train_gene_idx", "inner_train_genes"),
                ("hidden_gene_idx", "hidden_genes"),
            ):
                indices = split[indices_key]
                names = split[names_key]
                if len(indices) != len(names):
                    raise EvidenceError(
                        f"Frozen split gene names are misaligned: {split_path}"
                    )
                for index, name in zip(indices, names):
                    position = int(index)
                    if position < 0 or position >= expected:
                        raise EvidenceError(
                            f"Frozen split gene index is out of range: {split_path}"
                        )
                    value = str(name)
                    current = genes_by_index[position]
                    if current is not None and current != value:
                        raise EvidenceError(
                            f"Conflicting frozen gene names at index {position}: "
                            f"{split_path}"
                        )
                    genes_by_index[position] = value
            if any(value is None for value in genes_by_index):
                raise EvidenceError(
                    f"Frozen split does not reconstruct the full gene axis: {split_path}"
                )
            genes = [str(value) for value in genes_by_index]
        if len(genes) != expected or len(set(genes)) != expected:
            raise EvidenceError(f"Invalid gene axis for {dataset_id}")
        for fold in range(5):
            split_path = (
                RESULTS_ROOT
                / "inputs"
                / dataset_id
                / f"fold{fold}"
                / "mode_a_split.json"
            )
            split = json.loads(split_path.read_text(encoding="utf-8"))
            observed = [
                genes[int(index)] for index in split["final_test_gene_idx"]
            ]
            expected_test_genes = [str(value) for value in split["final_test_genes"]]
            if observed != expected_test_genes:
                raise EvidenceError(
                    f"Frozen test gene names differ from the reconstructed axis: "
                    f"{split_path}"
                )
        result[dataset_id] = genes
    return result


def read_test_indices(dataset_id: str, fold: int, n_genes: int) -> np.ndarray:
    path = RESULTS_ROOT / "inputs" / dataset_id / f"fold{fold}" / "mode_a_split.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    test_idx = np.asarray(payload["final_test_gene_idx"], dtype=np.int64)
    if test_idx.ndim != 1 or len(test_idx) == 0:
        raise EvidenceError(f"Invalid test index vector: {dataset_id} fold{fold}")
    if np.any(test_idx < 0) or np.any(test_idx >= n_genes) or len(np.unique(test_idx)) != len(test_idx):
        raise EvidenceError(f"Out-of-range or duplicate test indices: {dataset_id} fold{fold}")
    return test_idx


def load_formal_prediction(
    method: str,
    row: dict[str, Any],
    dataset_id: str,
    fold: int,
    test_idx: np.ndarray,
    truth_shape: tuple[int, int],
) -> tuple[np.ndarray, str, str]:
    if method == "GeneSPT":
        path = READOUT_ROOT / "test_predictions" / dataset_id / f"fold{fold}" / "GeneSPT.npz"
        archive = np.load(path, allow_pickle=False)
        observed_idx = np.asarray(archive["test_gene_idx"], dtype=np.int64)
        if not np.array_equal(observed_idx, test_idx):
            raise EvidenceError(f"Readout test indices changed: {dataset_id} fold{fold}")
        prediction = np.asarray(archive["prediction"], dtype=np.float32)
        layer = "validation_selected_readout_genespt57"
    elif method == "stAI":
        baseline_dir = STAI_ROOT / dataset_id / f"fold{fold}"
        path = baseline_dir / "imputed_expression.npy"
        audit_path = baseline_dir / "adapter_run_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("status") != "complete"
            or audit.get("method") != "stAI"
            or audit.get("dataset_id") != dataset_id
            or int(audit.get("fold", -1)) != fold
        ):
            raise EvidenceError(f"Invalid formal stAI audit: {audit_path}")
        if sha256_file(path) != audit.get("output_matrix_sha256"):
            raise EvidenceError(f"Formal stAI prediction SHA mismatch: {path}")
        axis_path = baseline_dir / str(audit.get("test_gene_axis_file"))
        if sha256_file(axis_path) != audit.get("test_gene_axis_sha256"):
            raise EvidenceError(f"Formal stAI axis SHA mismatch: {axis_path}")
        with np.load(axis_path, allow_pickle=False) as axis:
            observed_idx = np.asarray(axis["test_gene_idx"], dtype=np.int64)
            observed_dataset = str(axis["dataset_id"].item())
            observed_fold = int(axis["fold"].item())
        if (
            observed_dataset != dataset_id
            or observed_fold != fold
            or not np.array_equal(observed_idx, test_idx)
        ):
            raise EvidenceError(f"Formal stAI test axis changed: {axis_path}")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        prediction = np.asarray(matrix, dtype=np.float32).copy()
        layer = "raw_identity"
    else:
        path = host_path(row["prediction_path"])
        if sha256_file(path) != row["prediction_sha256"]:
            raise EvidenceError(f"Raw prediction SHA mismatch: {method} {dataset_id} fold{fold}")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(matrix.shape) == truth_shape:
            prediction = np.asarray(matrix[:, test_idx], dtype=np.float32).copy()
        elif tuple(matrix.shape) == (truth_shape[0], len(test_idx)):
            prediction = np.asarray(matrix, dtype=np.float32).copy()
        else:
            raise EvidenceError(
                f"Unexpected prediction shape for {method} {dataset_id} fold{fold}: {matrix.shape}"
            )
        layer = "raw_identity"
    expected = (truth_shape[0], len(test_idx))
    if prediction.shape != expected or not np.isfinite(prediction).all():
        raise EvidenceError(f"Invalid formal prediction: {method} {dataset_id} fold{fold}")
    return prediction, str(path), layer


def evaluate_all(
    raw_report: dict[str, Any],
    gene_axes: dict[str, list[str]],
    metrics_module: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_rows = {
        (row["dataset_id"], int(row["fold"])): row
        for row in raw_report["input_folds"]
    }
    raw_rows = {
        (row["dataset_id"], int(row["fold"]), row["method"]): row
        for row in raw_report["fold_metrics"]
        if row["method"] in FORMAL_METHODS
    }
    gene_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for (dataset_id, fold), input_row in sorted(input_rows.items()):
        truth_path = host_path(input_row["truth_path"])
        if sha256_file(truth_path) != input_row["truth_sha256"]:
            raise EvidenceError(f"Truth SHA mismatch: {dataset_id} fold{fold}")
        truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
        truth_shape = tuple(int(value) for value in input_row["truth_shape"])
        if tuple(truth.shape) != truth_shape:
            raise EvidenceError(f"Truth shape mismatch: {dataset_id} fold{fold}")
        genes = gene_axes[dataset_id]
        test_idx = read_test_indices(dataset_id, fold, len(genes))
        test_truth = np.asarray(truth[:, test_idx], dtype=np.float32).copy()
        test_genes = [genes[int(index)] for index in test_idx]
        for method in FORMAL_METHODS:
            raw_row = raw_rows.get(
                (dataset_id, fold, method),
                {"prediction_path": "", "prediction_sha256": ""},
            )
            prediction, prediction_path, layer = load_formal_prediction(
                method, raw_row, dataset_id, fold, test_idx, truth_shape
            )
            per_gene, summary_frame = metrics_module.evaluate_prediction(
                test_truth, prediction, gene_names=test_genes
            )
            if len(per_gene) != len(test_idx) or len(summary_frame) != 1:
                raise EvidenceError(f"Invalid evaluator output: {method} {dataset_id} fold{fold}")
            per_gene = per_gene.rename(columns={"gene_idx": "gene_pos"})
            per_gene.insert(0, "gene_idx", test_idx)
            per_gene.insert(0, "result_layer", layer)
            per_gene.insert(0, "method", method)
            per_gene.insert(0, "fold", fold)
            per_gene.insert(0, "role", input_row["role"])
            per_gene.insert(0, "dataset_id", dataset_id)
            per_gene.insert(0, "dataset", input_row["dataset"])
            keep = [
                "dataset", "dataset_id", "role", "fold", "method", "result_layer",
                "gene_idx", "gene_pos", "gene", "eligible_truth", "prediction_constant",
                "prediction_all_zero", "SPCC", "RMSE", "JSD", "SSIM",
            ]
            gene_frames.append(per_gene[keep])
            summary = summary_frame.iloc[0]
            fold_rows.append(
                {
                    "dataset": input_row["dataset"],
                    "dataset_id": dataset_id,
                    "role": input_row["role"],
                    "fold": fold,
                    "method": method,
                    "result_layer": layer,
                    "prediction_path": prediction_path,
                    "prediction_sha256": sha256_file(Path(prediction_path)),
                    "test_gene_count": len(test_idx),
                    "coverage": float(summary["coverage"]),
                    **{metric: float(summary[metric]) for metric in METRICS},
                }
            )
    genes = pd.concat(gene_frames, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    five_rows: list[dict[str, Any]] = []
    for keys, frame in folds.groupby(["dataset", "dataset_id", "role", "method", "result_layer"], sort=False):
        dataset, dataset_id, role, method, layer = keys
        if set(frame["fold"].astype(int)) != {0, 1, 2, 3, 4}:
            raise EvidenceError(f"Incomplete five-fold group: {dataset_id} {method}")
        row: dict[str, Any] = {
            "dataset": dataset,
            "dataset_id": dataset_id,
            "role": role,
            "method": method,
            "result_layer": layer,
            "folds": 5,
            "coverage": float(frame["coverage"].min()),
        }
        for metric in METRICS:
            values = frame.sort_values("fold")[metric].to_numpy(dtype=float)
            row[metric] = float(values.mean())
            row[f"{metric}_std_ddof0"] = float(values.std(ddof=0))
        five_rows.append(row)
    five = pd.DataFrame(five_rows)
    return genes, folds, five


def validate_formal_summary(
    five: pd.DataFrame,
    reference_path: Path,
    stai_reference_path: Path,
) -> dict[str, float]:
    reference = pd.read_csv(reference_path)
    reference_methods = tuple(method for method in FORMAL_METHODS if method != "stAI")
    reference = reference[reference["method"].isin(reference_methods)].copy()
    existing = five[five["method"].isin(reference_methods)].copy()
    merged = existing.merge(
        reference,
        on=["dataset", "dataset_id", "role", "method", "result_layer"],
        suffixes=("_new", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(existing) or len(merged) != len(reference):
        raise EvidenceError("Formal five-fold summary coverage mismatch")
    differences = []
    for metric in METRICS:
        differences.extend(
            np.abs(merged[f"{metric}_new"] - merged[f"{metric}_reference"]).tolist()
        )
    existing_maximum = float(max(differences, default=0.0))
    if existing_maximum > 1e-10:
        raise EvidenceError(
            "Formal summary differs from audited reference by "
            f"{existing_maximum:.3e}"
        )

    stai_reference = pd.read_csv(stai_reference_path)
    stai = five[five["method"].eq("stAI")].copy()
    stai_reference = stai_reference[stai_reference["method"].eq("stAI")].copy()
    stai_merged = stai.merge(
        stai_reference,
        on=["dataset", "dataset_id", "role", "method"],
        suffixes=("_new", "_reference"),
        validate="one_to_one",
    )
    if len(stai_merged) != 6 or len(stai) != 6 or len(stai_reference) != 6:
        raise EvidenceError("Formal stAI five-fold summary coverage mismatch")
    stai_differences = []
    for metric in METRICS:
        stai_differences.extend(
            np.abs(
                stai_merged[f"{metric}_new"]
                - stai_merged[f"{metric}_reference"]
            ).tolist()
        )
    stai_maximum = float(max(stai_differences, default=0.0))
    if stai_maximum > 1e-10:
        raise EvidenceError(
            "Formal stAI summary differs from the audited candidate reference by "
            f"{stai_maximum:.3e}"
        )
    return {
        "existing_methods": existing_maximum,
        "stAI": stai_maximum,
    }


def oriented_difference(metric: str, genespt: np.ndarray, baseline: np.ndarray) -> float:
    if metric in LOWER_IS_BETTER:
        return float(np.median(baseline) - np.median(genespt))
    return float(np.median(genespt) - np.median(baseline))


def stratified_bootstrap(
    gene_level: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 64,
) -> pd.DataFrame:
    if replicates < 1 or chunk_size < 1:
        raise ValueError("Bootstrap replicates and chunk size must be positive")
    rows: list[dict[str, Any]] = []
    for dataset_id, dataset_frame in gene_level.groupby("dataset_id", sort=False):
        dataset = str(dataset_frame["dataset"].iloc[0])
        role = str(dataset_frame["role"].iloc[0])
        genespt = dataset_frame[dataset_frame["method"].eq("GeneSPT")]
        for baseline_name in EXTERNAL_METHODS:
            baseline = dataset_frame[dataset_frame["method"].eq(baseline_name)]
            for metric in METRICS:
                fold_arrays: list[tuple[np.ndarray, np.ndarray]] = []
                fold_differences: list[float] = []
                for fold in range(5):
                    left = genespt[genespt["fold"].eq(fold)][["gene_idx", metric]]
                    right = baseline[baseline["fold"].eq(fold)][["gene_idx", metric]]
                    merged = left.merge(right, on="gene_idx", suffixes=("_genespt", "_baseline"), validate="one_to_one")
                    a = merged[f"{metric}_genespt"].to_numpy(dtype=float)
                    b = merged[f"{metric}_baseline"].to_numpy(dtype=float)
                    finite = np.isfinite(a) & np.isfinite(b)
                    a, b = a[finite], b[finite]
                    if len(a) == 0:
                        raise EvidenceError(f"No paired genes: {dataset_id} {baseline_name} {metric} fold{fold}")
                    fold_arrays.append((a, b))
                    fold_differences.append(oriented_difference(metric, a, b))
                key = f"{dataset_id}|{baseline_name}|{metric}|{seed}".encode("utf-8")
                local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
                rng = np.random.default_rng(local_seed)
                fold_samples = np.empty((len(fold_arrays), replicates), dtype=np.float64)
                for fold_position, (a, b) in enumerate(fold_arrays):
                    for start in range(0, replicates, chunk_size):
                        stop = min(start + chunk_size, replicates)
                        idx = rng.integers(
                            0,
                            len(a),
                            size=(stop - start, len(a)),
                            dtype=np.int32,
                        )
                        genespt_median = np.median(a[idx], axis=1)
                        baseline_median = np.median(b[idx], axis=1)
                        if metric in LOWER_IS_BETTER:
                            fold_samples[fold_position, start:stop] = (
                                baseline_median - genespt_median
                            )
                        else:
                            fold_samples[fold_position, start:stop] = (
                                genespt_median - baseline_median
                            )
                samples = fold_samples.mean(axis=0)
                point = float(np.mean(fold_differences))
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_id": dataset_id,
                        "role": role,
                        "method": "GeneSPT",
                        "comparator": baseline_name,
                        "metric": metric,
                        "positive_favors_genespt": True,
                        "paired_estimate": point,
                        "ci95_low": float(np.quantile(samples, 0.025)),
                        "ci95_high": float(np.quantile(samples, 0.975)),
                        "bootstrap_probability_positive": float(np.mean(samples > 0.0)),
                        "folds_positive": int(np.sum(np.asarray(fold_differences) > 0.0)),
                        "folds_total": 5,
                        "bootstrap_replicates": replicates,
                        "bootstrap_seed": seed,
                        "estimand": "mean_across_folds_of_paired_fold_median_difference",
                    }
                )
    return pd.DataFrame(rows)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw_report = json.loads(args.raw_report.read_text(encoding="utf-8"))
    counts = raw_report.get("counts", {})
    if (
        raw_report.get("status") != "complete"
        or raw_report.get("complete") is not True
        or int(counts.get("invalid_runs", 1)) != 0
        or int(counts.get("missing_runs", 1)) != 0
        or int(counts.get("invalid_input_folds", 1)) != 0
        or int(counts.get("missing_input_folds", 1)) != 0
    ):
        raise EvidenceError("Raw Protocol A report is not complete and valid")
    config = load_config(args.config)
    metrics_module = load_metrics_module(args.metrics)
    genes, folds, five = evaluate_all(raw_report, load_gene_names(config), metrics_module)
    maximum_differences = validate_formal_summary(
        five,
        args.formal_reference,
        args.stai_reference,
    )
    uncertainty = stratified_bootstrap(
        genes,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        chunk_size=args.bootstrap_chunk_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "formal_gene_level_metrics.csv": genes,
        "formal_fold_metrics.csv": folds,
        "formal_five_fold_metrics.csv": five,
        "formal_paired_bootstrap.csv": uncertainty,
    }
    for name, frame in outputs.items():
        write_csv(args.output_dir / name, frame)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "formal_methods": list(FORMAL_METHODS),
        "genespt_result_layer": "validation_selected_readout_genespt57",
        "external_result_layer": "raw_identity",
        "maximum_reference_difference": maximum_differences,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_chunk_size": args.bootstrap_chunk_size,
        "inputs": {
            "raw_report": {"path": str(args.raw_report), "sha256": sha256_file(args.raw_report)},
            "formal_reference": {"path": str(args.formal_reference), "sha256": sha256_file(args.formal_reference)},
            "metrics": {"path": str(args.metrics), "sha256": sha256_file(args.metrics)},
            "stai_formal_adoption": {
                "path": str(args.stai_root / "formal_adoption_manifest.json"),
                "sha256": sha256_file(
                    args.stai_root / "formal_adoption_manifest.json"
                ),
            },
            "stai_reference": {
                "path": str(args.stai_reference),
                "sha256": sha256_file(args.stai_reference),
            },
        },
        "outputs": {
            name: {"rows": len(frame), "sha256": sha256_file(args.output_dir / name)}
            for name, frame in outputs.items()
        },
    }
    manifest_path = args.output_dir / "formal_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", type=Path, default=RAW_REPORT)
    parser.add_argument("--readout-root", type=Path, default=READOUT_ROOT)
    parser.add_argument("--formal-reference", type=Path, default=READOUT_ROOT / "combined_five_fold_metrics.csv")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--stai-root", type=Path, default=STAI_ROOT)
    parser.add_argument("--stai-reference", type=Path, default=STAI_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=64)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    global READOUT_ROOT, STAI_ROOT
    READOUT_ROOT = args.readout_root.resolve()
    STAI_ROOT = args.stai_root.resolve()
    manifest = build(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
