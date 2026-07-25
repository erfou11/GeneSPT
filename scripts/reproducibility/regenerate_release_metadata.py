#!/usr/bin/env python3
"""Regenerate package structure, summary, and SHA256 manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


EXCLUDED_CHECKSUM_FILES = {"CHECKSUMS_SHA256.txt", "FILE_MANIFEST_SHA256.csv"}
CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_files(root: Path, *, include_checksums: bool) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file()]
    if not include_checksums:
        files = [path for path in files if path.name not in EXCLUDED_CHECKSUM_FILES]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().lower())


def csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def build_structure(root: Path) -> str:
    entries = [
        "CHECKSUMS_SHA256.txt\tgenerated; excluded from its own checksum manifest",
        "FILE_MANIFEST_SHA256.csv\tgenerated; excluded from its own checksum manifest",
    ]
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in package_files(root, include_checksums=False):
        grouped[path.relative_to(root).parts[0]].append(path)
    for name in sorted(grouped, key=str.lower):
        paths = grouped[name]
        target = root / name
        size_gib = sum(path.stat().st_size for path in paths) / (1024**3)
        if target.is_dir():
            entries.append(f"{name}/\t{len(paths)} files\t{size_gib:.3f} GiB")
        else:
            entries.append(f"{name}\t{size_gib:.6f} GiB")
    return "\n".join(entries) + "\n"


def manifest_counts(root: Path) -> dict[str, int]:
    manifest_root = root / "prediction_matrix_manifests"
    protocol_root = root / "protocol_a_reproducibility" / "manifests"
    paths = {
        "benchmark_prediction_rows": manifest_root / "PREDICTION_MATRIX_MANIFEST.csv",
        "fold_truth_rows": manifest_root / "PROTOCOL_A_TRUTH_MATRIX_MANIFEST.csv",
        "mechanism_prediction_rows": manifest_root / "MECHANISM_ABLATION_MATRIX_MANIFEST.csv",
        "baseline_task_rows": protocol_root / "FORMAL_BASELINE_RUN_MANIFEST.csv",
        "input_split_rows": protocol_root / "INPUT_SPLIT_MANIFEST.csv",
        "readout_selection_rows": protocol_root / "READOUT_SELECTION_MANIFEST.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release manifests: {missing}")
    return {name: csv_count(path) for name, path in paths.items()}


def build_summary(
    root: Path, *, zenodo_doi: str | None, zenodo_url: str | None
) -> dict[str, Any]:
    files = package_files(root, include_checksums=False)
    return {
        "schema_version": 1,
        "package_name": root.name,
        "build_date_local": date.today().isoformat(),
        "purpose": (
            "Reviewer-facing GeneSPT archive containing processed inputs, frozen "
            "whole-gene splits, formal Protocol A predictions, fold-specific truth, "
            "Figure 3 mechanism matrices, source data and provenance."
        ),
        "formal_contract": {
            "datasets": 6,
            "folds_per_dataset": 5,
            "benchmark_methods": [
                "GeneSPT",
                "Tangram",
                "TransImp",
                "SpaIM",
                "SpaGE",
                "stPlus",
                "stAI",
            ],
            "genespt_result_layer": "validation-selected readout locked before test evaluation",
            "external_baseline_result_layer": "raw identity output",
            "figure3_result_layer": "matched identity readout without post-hoc calibration",
        },
        "manifest_counts": manifest_counts(root),
        "manifested_file_count": len(files),
        "manifested_size_bytes": sum(path.stat().st_size for path in files),
        "generated_checksum_files_excluded_from_manifest": sorted(
            EXCLUDED_CHECKSUM_FILES
        ),
        "checksum_policy": (
            "FILE_MANIFEST_SHA256.csv and CHECKSUMS_SHA256.txt are generated last "
            "and intentionally exclude themselves to avoid circular hashes."
        ),
        "excluded_artifacts": [
            "raw public-source downloads",
            "h5ad files",
            "training checkpoints and caches",
            "superseded experiments",
        ],
        "zenodo_record_status": (
            "published" if zenodo_doi and zenodo_url else "local release candidate"
        ),
        "zenodo_record_doi": zenodo_doi,
        "zenodo_record_url": zenodo_url,
    }


def stabilize_metadata(
    root: Path, *, zenodo_doi: str | None, zenodo_url: str | None
) -> None:
    summary_path = root / "PACKAGE_BUILD_SUMMARY.json"
    structure_path = root / "PACKAGE_STRUCTURE.txt"
    for _ in range(12):
        old_summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        old_structure = structure_path.read_text(encoding="utf-8") if structure_path.exists() else ""
        summary_path.write_text(
            json.dumps(
                build_summary(root, zenodo_doi=zenodo_doi, zenodo_url=zenodo_url),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        structure_path.write_text(build_structure(root), encoding="utf-8")
        if (
            summary_path.read_text(encoding="utf-8") == old_summary
            and structure_path.read_text(encoding="utf-8") == old_structure
        ):
            return
    raise RuntimeError("Package metadata did not stabilize")


def write_checksums(root: Path) -> None:
    rows: list[tuple[str, int, str]] = []
    for index, path in enumerate(
        package_files(root, include_checksums=False), start=1
    ):
        relative = path.relative_to(root).as_posix()
        rows.append((relative, path.stat().st_size, sha256_file(path)))
        if index % 100 == 0:
            print(f"[hashed] {index}", flush=True)
    with (root / "FILE_MANIFEST_SHA256.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        writer.writerows(rows)
    (root / "CHECKSUMS_SHA256.txt").write_text(
        "".join(f"{digest}  {relative}\n" for relative, _, digest in rows),
        encoding="utf-8",
    )
    print(f"manifested_files={len(rows)}")
    print(f"manifested_bytes={sum(size for _, size, _ in rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--zenodo-doi")
    parser.add_argument("--zenodo-url")
    args = parser.parse_args()
    if bool(args.zenodo_doi) != bool(args.zenodo_url):
        parser.error("--zenodo-doi and --zenodo-url must be supplied together")
    if args.zenodo_doi and not args.zenodo_doi.startswith("https://doi.org/"):
        parser.error("--zenodo-doi must be a full DOI URL")
    if args.zenodo_url and not args.zenodo_url.startswith("https://zenodo.org/"):
        parser.error("--zenodo-url must be a full Zenodo URL")
    return args


def main() -> int:
    args = parse_args()
    root = args.root.resolve(strict=True)
    if not (root / "README.md").is_file():
        raise FileNotFoundError(f"Not a GeneSPT reviewer archive: {root}")
    stabilize_metadata(
        root, zenodo_doi=args.zenodo_doi, zenodo_url=args.zenodo_url
    )
    write_checksums(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
