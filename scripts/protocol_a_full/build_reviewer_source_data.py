#!/usr/bin/env python3
"""Build the reviewer-facing source-data bundle from finalized tables."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "protocol_a_full_rerun_20260711"
OUTPUT_ROOT = RESULTS_ROOT / "reviewer_source_data"

FORMAL_ROOT = RESULTS_ROOT / "evaluation" / "formal_benchmark_evidence"
PAIRED_BOOTSTRAP_SOURCE = FORMAL_ROOT / "formal_paired_bootstrap.csv"
GENE_METRICS_SOURCE = FORMAL_ROOT / "formal_gene_level_metrics.csv"
FIGURE4_SOURCE = (
    RESULTS_ROOT
    / "figures"
    / "figure4"
    / "figure4_hbc_representative_maps_source.csv"
)
SELECTED_GENES_SOURCE = (
    WORKSPACE_ROOT / "final_manuscript" / "figures" / "figure4_selected_genes.csv"
)
FIGURE6_ROOT = RESULTS_ROOT / "figures" / "figure6"
PANEL_B_SOURCE = FIGURE6_ROOT / "protocol_a_figure6_panel_b_records.csv"
PANEL_C_EFFECTS_SOURCE = FIGURE6_ROOT / "protocol_a_figure6_panel_c_effects.csv"
FIGURE6_SOURCE = FIGURE6_ROOT / "protocol_a_figure6_source.csv"
PANEL_D_ROOT = RESULTS_ROOT / "downstream" / "figure6d_protocol_a_all154"

BENCHMARK_METHODS = (
    "GeneSPT",
    "Tangram",
    "TransImp",
    "SpaIM",
    "SpaGE",
    "stPlus",
    "stAI",
)
FIGURE4_METHODS = BENCHMARK_METHODS
FIGURE6_METHODS = ("GeneSPT", "Tangram", "TransImp", "SpaIM", "SpaGE", "stPlus")
BASELINES = BENCHMARK_METHODS[1:]
EXPECTED_SELECTED_GENES = 4
EXPECTED_PANEL_B_ROWS = 300
EXPECTED_PANEL_B_ROWS_PER_METHOD = 50
EXPECTED_PANEL_C_DISPLAY_ROWS_PER_METHOD = 6000
EXPECTED_PANEL_C_SUMMARY_ROWS = 12
EXPECTED_PANEL_D_FOLD_ROWS = 30
EXPECTED_PANEL_D_SUMMARY_ROWS = 6
PANEL_C_SEED = "20260712"
EXPECTED_CELL_COUNT = 913
EXPECTED_CELL_TYPE_COUNT = 10

NEUTRAL_LAYERS = {
    "Ground truth": "observed_truth",
    "GeneSPT": "validation_selected_prediction",
    **{method: "external_method_prediction" for method in BASELINES},
}
FORMAL_LAYERS = {
    "Ground truth": "protocol_a_full_truth",
    "GeneSPT": "validation_selected_readout_genespt57",
    **{method: "raw_identity" for method in BASELINES},
}
FORMAL_MATRIX_MODE = "protocol_a_completed_outer_fold_matrix"
PUBLIC_MATRIX_MODE = "completed_outer_fold_matrix"

PUBLIC_CSV_PATHS = (
    "benchmark/paired_gene_bootstrap.csv",
    "figure4/selected_genes.csv",
    "figure4/panel_summary.csv",
    "figure4/per_gene_all_method_metrics.csv",
    "figure6/panel_b_fold_cell_type_records.csv",
    "figure6/panel_c_display_pairs.csv",
    "figure6/panel_c_all_pairs_summary.csv",
    "figure6/panel_d_fold_metrics.csv",
    "figure6/panel_d_summary.csv",
)

FIGURE4_DROP_COLUMNS = {
    "source_package_relative_path",
    "source_sha256",
    "coordinate_package_relative_path",
    "coordinate_sha256",
}
PANEL_C_SUMMARY_COLUMNS = (
    "panel",
    "row_type",
    "dataset",
    "dataset_id",
    "method",
    "result_layer",
    "metric",
    "metric_direction",
    "value",
    "n_effect_pairs",
    "n_valid_effect_pairs",
    "n_plotted",
    "panel_c_seed",
    "display_effect_min",
    "display_effect_max",
    "effect_formula",
    "variance_ddof",
)

FORBIDDEN_PUBLIC_PATTERNS = (
    re.compile(r"raw[_ ]identity", re.IGNORECASE),
    re.compile(r"protocol[_ ]a", re.IGNORECASE),
    re.compile(r"evaluation[\\/]", re.IGNORECASE),
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"(?:^|[\s\"'(])/(?:home|workspace|mnt|root|users|tmp)/", re.IGNORECASE),
    re.compile(r"(?:repository|archive|results|inputs)[\\/]", re.IGNORECASE),
    re.compile(r"zenodo_upload", re.IGNORECASE),
)


class BundleError(RuntimeError):
    """Raised when a finalized input violates the public bundle contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_reader(path: Path) -> tuple[list[str], Iterator[dict[str, str]]]:
    handle = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        handle.close()
        raise BundleError(f"CSV has no header: {path.name}")

    def rows() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                if None in row:
                    raise BundleError(f"Malformed CSV row in {path.name}")
                yield dict(row)
        finally:
            handle.close()

    return list(reader.fieldnames), rows()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames, rows = csv_reader(path)
    return fieldnames, list(rows)


def write_csv(
    relative_path: str,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, str]],
) -> int:
    path = OUTPUT_ROOT / Path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
            count += 1
    return count


def count_csv_rows(path: Path) -> int:
    _, rows = csv_reader(path)
    return sum(1 for _ in rows)


def require_columns(
    fieldnames: Sequence[str], required: Iterable[str], source_name: str
) -> None:
    missing = set(required).difference(fieldnames)
    if missing:
        raise BundleError(f"{source_name} is missing columns: {sorted(missing)}")


def require_single_match(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise BundleError(
            f"Expected one {pattern} input in {root.name}, found {len(matches)}"
        )
    return matches[0]


def neutralize_result_layer(row: dict[str, str], column: str) -> dict[str, str]:
    method = row.get("method", "")
    if method not in NEUTRAL_LAYERS:
        raise BundleError(f"Unexpected method while neutralizing {column}: {method}")
    formal_layer = row.get(column, "")
    expected = FORMAL_LAYERS[method]
    if formal_layer != expected:
        raise BundleError(
            f"{method} has unexpected finalized layer {formal_layer!r} in {column}"
        )
    row[column] = NEUTRAL_LAYERS[method]
    return row


def copy_public_csv(source: Path, relative_path: str) -> int:
    destination = OUTPUT_ROOT / Path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return count_csv_rows(destination)


def build_figure4_panel_summary() -> int:
    fieldnames, source_rows = read_csv(FIGURE4_SOURCE)
    require_columns(
        fieldnames,
        {"panel_id", "gene", "fold", "gene_idx", "method", "result_layer"},
        FIGURE4_SOURCE.name,
    )
    public_fields = [
        column for column in fieldnames if column not in FIGURE4_DROP_COLUMNS
    ]
    if any("path" in column.lower() for column in public_fields):
        raise BundleError("Figure 4 panel summary still contains a path column")

    public_rows: list[dict[str, str]] = []
    method_counts: Counter[str] = Counter()
    for source_row in source_rows:
        row = {column: source_row[column] for column in public_fields}
        neutralize_result_layer(row, "result_layer")
        method_counts[row["method"]] += 1
        public_rows.append(row)
    expected_methods = ("Ground truth", *FIGURE4_METHODS)
    if len(public_rows) != EXPECTED_SELECTED_GENES * len(expected_methods):
        raise BundleError("Figure 4 panel summary must contain 32 rows")
    if method_counts != Counter({method: 4 for method in expected_methods}):
        raise BundleError("Figure 4 panel summary must contain four rows per method")
    return write_csv("figure4/panel_summary.csv", public_fields, public_rows)


def selected_gene_keys() -> tuple[list[str], dict[str, str]]:
    fieldnames, rows = read_csv(SELECTED_GENES_SOURCE)
    require_columns(fieldnames, {"gene", "fold_id"}, SELECTED_GENES_SOURCE.name)
    if len(rows) != EXPECTED_SELECTED_GENES:
        raise BundleError("Figure 4 selected-gene table must contain four genes")
    genes = [row["gene"] for row in rows]
    if len(set(genes)) != EXPECTED_SELECTED_GENES:
        raise BundleError("Figure 4 selected genes must be unique")
    return genes, {row["gene"]: row["fold_id"] for row in rows}


def build_figure4_gene_metrics() -> int:
    genes, fold_by_gene = selected_gene_keys()
    fieldnames, source_rows = csv_reader(GENE_METRICS_SOURCE)
    require_columns(
        fieldnames,
        {
            "dataset_id",
            "fold",
            "method",
            "result_layer",
            "gene",
            "gene_idx",
            "SPCC",
            "RMSE",
            "JSD",
            "SSIM",
        },
        GENE_METRICS_SOURCE.name,
    )
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for source_row in source_rows:
        gene = source_row["gene"]
        method = source_row["method"]
        if (
            source_row["dataset_id"] != "HBC_shared16112"
            or gene not in fold_by_gene
            or source_row["fold"] != fold_by_gene[gene]
            or method not in FIGURE4_METHODS
        ):
            continue
        key = (gene, method)
        if key in selected:
            raise BundleError(f"Duplicate finalized gene metric: {gene}, {method}")
        row = dict(source_row)
        neutralize_result_layer(row, "result_layer")
        selected[key] = row

    expected_keys = {
        (gene, method) for gene in genes for method in FIGURE4_METHODS
    }
    if set(selected) != expected_keys:
        missing = sorted(expected_keys.difference(selected))
        raise BundleError(f"Missing finalized Figure 4 gene metrics: {missing}")
    ordered_rows = [
        selected[(gene, method)]
        for gene in genes
        for method in FIGURE4_METHODS
    ]
    return write_csv(
        "figure4/per_gene_all_method_metrics.csv", fieldnames, ordered_rows
    )


def build_panel_b() -> int:
    fieldnames, source_rows = read_csv(PANEL_B_SOURCE)
    require_columns(
        fieldnames,
        {
            "method",
            "result_layer",
            "fold",
            "cell_type",
            "n_in",
            "n_out",
            "effect_formula",
            "variance_ddof",
        },
        PANEL_B_SOURCE.name,
    )
    public_fields = [
        column
        for column in fieldnames
        if "artifact" not in column.lower()
        and "path" not in column.lower()
        and column.lower() != "source_key"
    ]
    public_rows: list[dict[str, str]] = []
    method_counts: Counter[str] = Counter()
    fold_cell_types: dict[tuple[str, str], set[str]] = {}
    for source_row in source_rows:
        row = {column: source_row[column] for column in public_fields}
        neutralize_result_layer(row, "result_layer")
        if int(row["n_in"]) + int(row["n_out"]) != EXPECTED_CELL_COUNT:
            raise BundleError("Panel B row does not cover 913 cells")
        method_counts[row["method"]] += 1
        fold_cell_types.setdefault((row["method"], row["fold"]), set()).add(
            row["cell_type"]
        )
        public_rows.append(row)
    if len(public_rows) != EXPECTED_PANEL_B_ROWS:
        raise BundleError("Panel B must contain 300 fold-cell-type rows")
    if method_counts != Counter(
        {method: EXPECTED_PANEL_B_ROWS_PER_METHOD for method in FIGURE6_METHODS}
    ):
        raise BundleError("Panel B must contain 50 rows per method")
    if len(fold_cell_types) != len(FIGURE6_METHODS) * 5 or any(
        len(cell_types) != EXPECTED_CELL_TYPE_COUNT
        for cell_types in fold_cell_types.values()
    ):
        raise BundleError("Panel B must contain ten matched cell types per fold")
    return write_csv(
        "figure6/panel_b_fold_cell_type_records.csv", public_fields, public_rows
    )


def build_panel_c_display_pairs() -> tuple[int, Counter[str]]:
    fieldnames, source_rows = csv_reader(PANEL_C_EFFECTS_SOURCE)
    require_columns(
        fieldnames,
        {
            "method",
            "result_layer",
            "true_cell_type_effect",
            "predicted_cell_type_effect",
            "pair_valid",
            "source_key",
            "plotted",
        },
        PANEL_C_EFFECTS_SOURCE.name,
    )
    public_fields = [column for column in fieldnames if column != "source_key"]
    counts: Counter[str] = Counter()

    def public_rows() -> Iterator[dict[str, str]]:
        for source_row in source_rows:
            if source_row["plotted"].casefold() != "true":
                continue
            if source_row["pair_valid"].casefold() != "true":
                raise BundleError("A plotted Panel C pair is not jointly finite")
            row = {column: source_row[column] for column in public_fields}
            neutralize_result_layer(row, "result_layer")
            counts[row["method"]] += 1
            yield row

    row_count = write_csv(
        "figure6/panel_c_display_pairs.csv", public_fields, public_rows()
    )
    expected = Counter(
        {
            method: EXPECTED_PANEL_C_DISPLAY_ROWS_PER_METHOD
            for method in FIGURE6_METHODS
        }
    )
    if counts != expected:
        raise BundleError("Panel C must contain 6,000 plotted pairs per method")
    return row_count, counts


def build_panel_c_summary(display_counts: Counter[str]) -> int:
    fieldnames, source_rows = read_csv(FIGURE6_SOURCE)
    require_columns(fieldnames, PANEL_C_SUMMARY_COLUMNS, FIGURE6_SOURCE.name)
    public_rows: list[dict[str, str]] = []
    metric_counts: Counter[str] = Counter()
    for source_row in source_rows:
        if (
            source_row["panel"] != "C"
            or source_row["row_type"] != "all_effect_pairs_metric"
        ):
            continue
        row = {column: source_row[column] for column in PANEL_C_SUMMARY_COLUMNS}
        neutralize_result_layer(row, "result_layer")
        if row["panel_c_seed"] != PANEL_C_SEED:
            raise BundleError("Panel C summary does not use fixed seed 20260712")
        if int(row["n_plotted"]) != display_counts[row["method"]]:
            raise BundleError("Panel C summary/display point counts disagree")
        metric_counts[row["method"]] += 1
        public_rows.append(row)
    if len(public_rows) != EXPECTED_PANEL_C_SUMMARY_ROWS:
        raise BundleError("Panel C all-pairs summary must contain 12 rows")
    if metric_counts != Counter({method: 2 for method in FIGURE6_METHODS}):
        raise BundleError("Panel C summary must contain two metrics per method")
    return write_csv(
        "figure6/panel_c_all_pairs_summary.csv",
        PANEL_C_SUMMARY_COLUMNS,
        public_rows,
    )


def build_panel_d_table(source: Path, relative_path: str, expected_rows: int) -> int:
    fieldnames, source_rows = read_csv(source)
    require_columns(
        fieldnames,
        {"method", "prediction_result_layer", "matrix_mode"},
        source.name,
    )
    public_rows: list[dict[str, str]] = []
    method_counts: Counter[str] = Counter()
    for source_row in source_rows:
        row = dict(source_row)
        neutralize_result_layer(row, "prediction_result_layer")
        if row["matrix_mode"] != FORMAL_MATRIX_MODE:
            raise BundleError(f"Unexpected finalized matrix mode in {source.name}")
        row["matrix_mode"] = PUBLIC_MATRIX_MODE
        method_counts[row["method"]] += 1
        public_rows.append(row)
    if len(public_rows) != expected_rows or set(method_counts) != set(
        FIGURE6_METHODS
    ):
        raise BundleError(f"Unexpected method coverage in {source.name}")
    expected_per_method = expected_rows // len(FIGURE6_METHODS)
    if method_counts != Counter(
        {method: expected_per_method for method in FIGURE6_METHODS}
    ):
        raise BundleError(f"Unequal method coverage in {source.name}")
    return write_csv(relative_path, fieldnames, public_rows)


def public_csv_metadata(row_counts: Mapping[str, int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative_path in PUBLIC_CSV_PATHS:
        path = OUTPUT_ROOT / Path(relative_path)
        records.append(
            {
                "filename": relative_path,
                "rows": row_counts[relative_path],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def readme_text(records: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Reviewer source data",
        "",
        "This bundle republishes finalized source values without recomputing any result. Labels describing result layers are neutralized, and local provenance paths are omitted.",
        "",
        "## Analysis notes",
        "",
        "- Panels B and C use 913/913 cells and 10 matched types.",
        "- Cell-type effects use population variance (ddof=0); zero-variance effects remain invalid.",
        "- Top-k genes are selected from jointly finite, strictly positive effects, ordered by descending effect with gene index used to break ties.",
        "- Panel C fixed seed 20260712 selects 6,000 displayed pairs independently for each method; all-pairs summaries retain every jointly finite pair.",
        "- Panel D uses all154/PCA30/k15/seed0 weighted Louvain and matrix mode completed_outer_fold_matrix.",
        "",
        "## File inventory",
        "",
        "| filename | rows | bytes | SHA256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['filename']} | {record['rows']} | {record['bytes']} | {record['sha256']} |"
        )
    lines.extend(
        [
            "",
            "The JSON manifest inventories every other public file and does not self-hash.",
            "",
        ]
    )
    return "\n".join(lines)


def write_readme_and_manifest(
    csv_records: Sequence[Mapping[str, object]],
) -> None:
    readme_path = OUTPUT_ROOT / "README.md"
    readme_path.write_text(readme_text(csv_records), encoding="utf-8", newline="\n")
    readme_record: dict[str, object] = {
        "filename": "README.md",
        "rows": len(readme_path.read_text(encoding="utf-8").splitlines()),
        "bytes": readme_path.stat().st_size,
        "sha256": sha256_file(readme_path),
    }
    manifest = {"files": [readme_record, *csv_records]}
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def assert_public_bundle() -> None:
    expected_files = {
        "README.md",
        "manifest.json",
        *PUBLIC_CSV_PATHS,
    }
    actual_files = {
        path.relative_to(OUTPUT_ROOT).as_posix()
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise BundleError(
            f"Unexpected public files: missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )

    for relative_path in sorted(actual_files):
        path = OUTPUT_ROOT / Path(relative_path)
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PUBLIC_PATTERNS:
            if pattern.search(text):
                raise BundleError(
                    f"Forbidden public text {pattern.pattern!r} in {relative_path}"
                )
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise BundleError(f"CSV has no header: {relative_path}")
                fieldnames = list(reader.fieldnames)
            forbidden_columns = [
                column
                for column in fieldnames
                if "artifact" in column.lower()
                or "path" in column.lower()
                or column.lower() == "source_key"
            ]
            if forbidden_columns:
                raise BundleError(
                    f"Internal provenance columns in {relative_path}: {forbidden_columns}"
                )


def build_bundle() -> dict[str, int]:
    required_inputs = (
        PAIRED_BOOTSTRAP_SOURCE,
        GENE_METRICS_SOURCE,
        FIGURE4_SOURCE,
        SELECTED_GENES_SOURCE,
        PANEL_B_SOURCE,
        PANEL_C_EFFECTS_SOURCE,
        FIGURE6_SOURCE,
    )
    missing = [path for path in required_inputs if not path.is_file()]
    if missing:
        raise BundleError(f"Missing finalized inputs: {[path.name for path in missing]}")

    panel_d_fold_source = require_single_match(PANEL_D_ROOT, "*fold_metrics.csv")
    panel_d_summary_source = require_single_match(PANEL_D_ROOT, "*summary.csv")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    row_counts: dict[str, int] = {}
    row_counts[PUBLIC_CSV_PATHS[0]] = copy_public_csv(
        PAIRED_BOOTSTRAP_SOURCE, PUBLIC_CSV_PATHS[0]
    )
    row_counts[PUBLIC_CSV_PATHS[1]] = copy_public_csv(
        SELECTED_GENES_SOURCE, PUBLIC_CSV_PATHS[1]
    )
    row_counts[PUBLIC_CSV_PATHS[2]] = build_figure4_panel_summary()
    row_counts[PUBLIC_CSV_PATHS[3]] = build_figure4_gene_metrics()
    row_counts[PUBLIC_CSV_PATHS[4]] = build_panel_b()
    panel_c_rows, display_counts = build_panel_c_display_pairs()
    row_counts[PUBLIC_CSV_PATHS[5]] = panel_c_rows
    row_counts[PUBLIC_CSV_PATHS[6]] = build_panel_c_summary(display_counts)
    row_counts[PUBLIC_CSV_PATHS[7]] = build_panel_d_table(
        panel_d_fold_source,
        PUBLIC_CSV_PATHS[7],
        EXPECTED_PANEL_D_FOLD_ROWS,
    )
    row_counts[PUBLIC_CSV_PATHS[8]] = build_panel_d_table(
        panel_d_summary_source,
        PUBLIC_CSV_PATHS[8],
        EXPECTED_PANEL_D_SUMMARY_ROWS,
    )

    expected_counts = {
        PUBLIC_CSV_PATHS[0]: 144,
        PUBLIC_CSV_PATHS[1]: 4,
        PUBLIC_CSV_PATHS[2]: 32,
        PUBLIC_CSV_PATHS[3]: 28,
        PUBLIC_CSV_PATHS[4]: 300,
        PUBLIC_CSV_PATHS[5]: 36000,
        PUBLIC_CSV_PATHS[6]: 12,
        PUBLIC_CSV_PATHS[7]: 30,
        PUBLIC_CSV_PATHS[8]: 6,
    }
    if row_counts != expected_counts:
        raise BundleError(f"Unexpected output row counts: {row_counts}")

    csv_records = public_csv_metadata(row_counts)
    write_readme_and_manifest(csv_records)
    assert_public_bundle()
    return row_counts


def main() -> None:
    row_counts = build_bundle()
    print(f"reviewer_source_data={OUTPUT_ROOT}")
    for relative_path in PUBLIC_CSV_PATHS:
        print(f"{relative_path}: {row_counts[relative_path]} rows")


if __name__ == "__main__":
    main()
