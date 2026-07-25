#!/usr/bin/env python3
"""Verify the reviewer-facing Protocol A code and archive contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METHODS = (
    "GeneSPT",
    "Tangram",
    "TransImp",
    "SpaIM",
    "SpaGE",
    "stPlus",
    "stAI",
)
BASELINES = METHODS[1:]
FOLDS = (0, 1, 2, 3, 4)
CHUNK_BYTES = 8 * 1024 * 1024


class VerifyError(RuntimeError):
    """Raised when release evidence is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise VerifyError(f"Manifest path escapes archive root: {relative}") from exc
    if not path.is_file():
        raise VerifyError(f"Missing release file: {path}")
    return path


def require_count(rows: Sequence[Any], expected: int, label: str) -> None:
    if len(rows) != expected:
        raise VerifyError(f"Expected {expected} {label}, found {len(rows)}")


def verify_archive(archive_root: Path, verify_hashes: bool) -> dict[str, Any]:
    manifests = archive_root / "prediction_matrix_manifests"
    provenance = archive_root / "protocol_a_reproducibility" / "manifests"
    predictions = read_csv(manifests / "PREDICTION_MATRIX_MANIFEST.csv")
    truths = read_csv(manifests / "PROTOCOL_A_TRUTH_MATRIX_MANIFEST.csv")
    baselines = read_csv(provenance / "FORMAL_BASELINE_RUN_MANIFEST.csv")
    inputs = read_csv(provenance / "INPUT_SPLIT_MANIFEST.csv")
    selections = read_csv(provenance / "READOUT_SELECTION_MANIFEST.csv")
    mechanisms = read_csv(manifests / "MECHANISM_ABLATION_MATRIX_MANIFEST.csv")
    require_count(predictions, 6 * len(FOLDS) * len(METHODS), "prediction rows")
    require_count(truths, 6 * len(FOLDS), "truth rows")
    require_count(baselines, 6 * len(FOLDS) * len(BASELINES), "baseline task rows")
    require_count(inputs, 6 * len(FOLDS), "input/split rows")
    require_count(selections, 6 * len(FOLDS) * 2, "readout selection rows")
    require_count(mechanisms, 90, "mechanism prediction rows")

    prediction_keys: set[tuple[str, int, str]] = set()
    for row in predictions:
        key = (row["dataset_id"], int(row["fold"]), row["method"])
        if key in prediction_keys:
            raise VerifyError(f"Duplicate prediction key: {key}")
        prediction_keys.add(key)
        if row["method"] not in METHODS or int(row["fold"]) not in FOLDS:
            raise VerifyError(f"Invalid prediction identity: {key}")
        matrix = resolve(archive_root, row["matrix_path"])
        test_idx = resolve(archive_root, row["test_gene_idx_path"])
        truth = resolve(archive_root, row["truth_path"])
        if verify_hashes:
            expected = {
                matrix: row["compact_prediction_sha256"],
                test_idx: row["test_gene_idx_sha256"],
                truth: row["truth_sha256"],
            }
            for path, digest in expected.items():
                if sha256_file(path) != digest:
                    raise VerifyError(f"SHA256 mismatch: {path}")

    for row in truths:
        truth = resolve(archive_root, row["truth_path"])
        test_idx = resolve(archive_root, row["test_gene_idx_path"])
        if verify_hashes:
            if sha256_file(truth) != row["truth_sha256"]:
                raise VerifyError(f"Truth SHA256 mismatch: {truth}")
            if sha256_file(test_idx) != row["test_gene_idx_sha256"]:
                raise VerifyError(f"Truth index SHA256 mismatch: {test_idx}")

    for row in baselines:
        if row["method"] not in BASELINES:
            raise VerifyError(f"Unexpected baseline method: {row['method']}")
        if row["status"] != "completed" or int(row["returncode"]) != 0:
            raise VerifyError(f"Incomplete baseline task: {row['task']}")
        if row["fallback_used"].lower() != "false":
            raise VerifyError(f"Fallback used in formal task: {row['task']}")
        if row["finite"].lower() != "true":
            raise VerifyError(f"Non-finite formal task: {row['task']}")
        if row["complete_test_coverage"].lower() != "true":
            raise VerifyError(f"Incomplete test coverage: {row['task']}")
        command = json.loads(row["command"])
        command_text = " ".join(str(item) for item in command)
        if "/workspace/" in command_text or ":/" in command_text:
            raise VerifyError(f"Unsanitized command path: {row['task']}")

    for row in selections:
        if int(row["candidate_count"]) != 57:
            raise VerifyError(
                f"Unexpected readout candidate count for {row['dataset_id']} fold{row['fold']}"
            )
        if row["test_prediction_accessed_before_lock"].lower() != "false":
            raise VerifyError("Readout lock accessed final-test predictions before selection")
        if row["test_truth_accessed_before_lock"].lower() != "false":
            raise VerifyError("Readout lock accessed final-test truth before selection")
        public_lock = resolve(archive_root, row["public_selection_lock_path"])
        if verify_hashes and sha256_file(public_lock) != row["public_selection_lock_sha256"]:
            raise VerifyError(f"Public readout lock SHA256 mismatch: {public_lock}")

    mechanism_keys: set[tuple[str, str, int, str]] = set()
    panel_counts = {"A": 0, "B": 0, "C": 0}
    for row in mechanisms:
        key = (row["panel"], row["dataset_id"], int(row["fold"]), row["control"])
        if key in mechanism_keys:
            raise VerifyError(f"Duplicate mechanism key: {key}")
        mechanism_keys.add(key)
        if row["panel"] not in panel_counts or int(row["fold"]) not in FOLDS:
            raise VerifyError(f"Invalid mechanism identity: {key}")
        panel_counts[row["panel"]] += 1
        if row["readout"] != "identity" or row["posthoc_calibration"] != "none":
            raise VerifyError(f"Non-identity mechanism result: {key}")
        matrix = resolve(archive_root, row["matrix_path"])
        metadata = resolve(archive_root, row["metadata_path"])
        test_idx = resolve(archive_root, row["test_gene_idx_path"])
        resolve(archive_root, row["truth_path"])
        resolve(archive_root, row["gene_names_path"])
        if verify_hashes:
            expected = {
                matrix: row["compact_prediction_sha256"],
                metadata: row["metadata_sha256"],
                test_idx: row["test_gene_idx_sha256"],
            }
            for path, digest in expected.items():
                if sha256_file(path) != digest:
                    raise VerifyError(f"Mechanism SHA256 mismatch: {path}")
    if panel_counts != {"A": 20, "B": 30, "C": 40}:
        raise VerifyError(f"Unexpected Figure 3 panel counts: {panel_counts}")

    text_roots = (
        archive_root / "protocol_a_reproducibility",
        archive_root / "prediction_matrix_manifests",
        archive_root / "results_source_data",
        archive_root / "figures",
        archive_root / "provenance_reports",
        archive_root / "label_provenance",
    )
    banned = re.compile(
        r"stDiff|TransPA|GeneSPT-LCR|\bLCR\b|D:/TESTWORK001|D:\\TESTWORK001|/workspace/|topodist"
    )
    for text_root in text_roots:
        for path in text_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".csv", ".json", ".yaml", ".yml"}:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            match = banned.search(content)
            if match:
                raise VerifyError(f"Banned release token {match.group(0)!r} in {path}")
    for path in (archive_root / "README.md", archive_root / "DATASET_MANIFEST.csv"):
        match = banned.search(path.read_text(encoding="utf-8", errors="replace"))
        if match:
            raise VerifyError(f"Banned release token {match.group(0)!r} in {path}")
    return {
        "prediction_rows": len(predictions),
        "truth_rows": len(truths),
        "baseline_tasks": len(baselines),
        "input_split_rows": len(inputs),
        "readout_selection_rows": len(selections),
        "mechanism_prediction_rows": len(mechanisms),
        "mechanism_panel_counts": panel_counts,
        "hashes_verified": verify_hashes,
    }


def verify_repository(repository_root: Path) -> dict[str, Any]:
    adapter_paths = {
        "Tangram": repository_root / "baseline_adapters" / "tangram" / "run_tangram_mhpr_fold_from_split.py",
        "TransImp": repository_root / "baseline_adapters" / "transimp" / "run_transimp_mhpr_fold_from_split.py",
        "SpaIM": repository_root / "baseline_adapters" / "spaim" / "run_spaim_mhpr_fold_from_split.py",
        "SpaGE": repository_root / "baseline_adapters" / "spage" / "run_spage_mhpr_fold_from_split.py",
        "stPlus": repository_root / "baseline_adapters" / "stplus" / "run_stplus_mhpr_fold_from_split.py",
        "stAI": repository_root / "baseline_adapters" / "stai" / "run_stai_protocol_a.py",
    }
    for method, path in adapter_paths.items():
        if not path.is_file():
            raise VerifyError(f"Missing {method} adapter: {path}")
    required = (
        repository_root / "scripts" / "reproducibility" / "recompute_protocol_a_benchmark.py",
        repository_root / "scripts" / "reproducibility" / "recompute_protocol_a_mechanism.py",
        repository_root / "scripts" / "reproducibility" / "export_protocol_a_release.py",
        repository_root / "scripts" / "reproducibility" / "export_protocol_a_mechanism.py",
        repository_root / "scripts" / "reproducibility" / "regenerate_release_metadata.py",
        repository_root / "docs" / "BASELINE_ADAPTATION.md",
        repository_root / "docs" / "REPRODUCE_PROTOCOL_A.md",
        repository_root / "configs" / "protocol_a_baseline_versions.yaml",
        repository_root / "manifests" / "protocol_a" / "MECHANISM_ABLATION_MATRIX_MANIFEST.csv",
        repository_root / "manifests" / "protocol_a" / "STAI_FORMAL_ADOPTION_MANIFEST.json",
    )
    for path in required:
        if not path.is_file():
            raise VerifyError(f"Missing reviewer-facing repository file: {path}")
    return {
        "adapter_count": len(adapter_paths),
        "required_repository_files": len(required),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    archive = verify_archive(args.archive_root.resolve(strict=True), not args.skip_hash_check)
    repository = verify_repository(args.repository_root.resolve(strict=True))
    summary = {"status": "verified", "archive": archive, "repository": repository}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerifyError as exc:
        raise SystemExit(f"ERROR: {exc}")
