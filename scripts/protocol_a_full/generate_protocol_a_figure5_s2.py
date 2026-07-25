#!/usr/bin/env python3
"""Generate Protocol A Figure 5 and Supplementary Table S2.

Figure 5 is rebuilt only from the formal per-gene evidence. Supplementary
Table S2 is the single complete six-dataset by seven-method formal benchmark.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "protocol_a_full_rerun_20260711"
EVALUATION_ROOT = RESULTS_ROOT / "evaluation"
FORMAL_ROOT = EVALUATION_ROOT / "formal_benchmark_evidence"
READOUT_ROOT = EVALUATION_ROOT / "validation_selected_readout_genespt57"

FORMAL_MANIFEST = FORMAL_ROOT / "formal_evidence_manifest.json"
FORMAL_GENE_LEVEL = FORMAL_ROOT / "formal_gene_level_metrics.csv"
FORMAL_FIVE_FOLD = FORMAL_ROOT / "formal_five_fold_metrics.csv"
AUDITED_COMBINED_FIVE_FOLD = READOUT_ROOT / "combined_five_fold_metrics.csv"

FIGURE_DIR = RESULTS_ROOT / "figures" / "figure5"
S2_DIR = RESULTS_ROOT / "supplementary" / "S2"
FIGURE_STEM = "figure5_protocol_a_cross_platform_per_gene_violins"
FIGURE_SOURCE = FIGURE_DIR / f"{FIGURE_STEM}_source.csv"
FIGURE_PNG = FIGURE_DIR / f"{FIGURE_STEM}.png"
FIGURE_PDF = FIGURE_DIR / f"{FIGURE_STEM}.pdf"
FIGURE_CAPTION = FIGURE_DIR / f"{FIGURE_STEM}_caption_draft.md"
S2_CSV = S2_DIR / "supplementary_table_s2_formal_benchmark.csv"
S2_README = S2_DIR / "README.md"
S2_MANIFEST = S2_DIR / "manifest.json"

FORMAL_METHODS = (
    "GeneSPT",
    "Tangram",
    "TransImp",
    "SpaIM",
    "SpaGE",
    "stPlus",
    "stAI",
)
METHOD_ORDER = (
    "GeneSPT",
    "SpaIM",
    "Tangram",
    "TransImp",
    "SpaGE",
    "stPlus",
    "stAI",
)
METHOD_COLORS = {
    "GeneSPT": "#b2182b",
    "SpaIM": "#4c9b9b",
    "Tangram": "#59a14f",
    "TransImp": "#8b6fb3",
    "SpaGE": "#4c78a8",
    "stPlus": "#e6862f",
    "stAI": "#c49a00",
}
FORBIDDEN_METHODS = frozenset({"stDiff", "TransPA", "LCR", "GeneSPT-GC"})

DATASETS = (
    ("Vis9A", "Vis9A_D7_spaim_effective4470", "primary"),
    ("HBC", "HBC_shared16112", "primary"),
    (
        "Cell2location",
        "Cell2location_mouse_brain_ST8059048_shared12819",
        "primary",
    ),
    (
        "seqFISH+",
        "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000",
        "cross_platform",
    ),
    ("MHPR", "MHPR_current_panel", "cross_platform"),
    ("MVC", "MVC_shared981", "cross_platform"),
)
DATASET_ORDER = tuple(dataset for dataset, _, _ in DATASETS)
DATASET_ID_ORDER = tuple(dataset_id for _, dataset_id, _ in DATASETS)
FIGURE_DATASET_ORDER = ("seqFISH+", "MHPR", "MVC")
FIGURE_DATASET_LABELS = {
    "seqFISH+": "seqFISH+ cortex/SVZ",
    "MHPR": "MHPR/MERFISH",
    "MVC": "MVC/STARmap",
}

METRICS = ("SPCC", "RMSE", "JSD", "SSIM")
FIGURE_METRICS = ("SPCC", "RMSE", "JSD", "SSIM")
LOWER_IS_BETTER = frozenset({"RMSE", "JSD"})
DISPLAY_INVERTED_METRICS = frozenset({"SPCC", "SSIM"})
FIGURE_METRIC_LABELS = {
    "SPCC": "1 - SPCC \u2193",
    "RMSE": "RMSE \u2193",
    "JSD": "JS/JSD \u2193",
    "SSIM": "1 - SSIM \u2193",
}
SELECTED_LAYER = "validation_selected_readout_genespt57"
RAW_LAYER = "raw_identity"

S2_PUBLIC_COLUMNS = (
    "dataset",
    "dataset_id",
    "role",
    "method",
    "folds",
    "coverage",
) + tuple(
    field
    for metric in METRICS
    for field in (
        f"{metric}_mean",
        f"{metric}_fold_sd_ddof0",
        f"{metric}_rank",
        f"{metric}_is_second_best",
    )
)

PUBLIC_S2_FORBIDDEN_PATTERNS = {
    "raw identity label": re.compile(r"raw[\s_-]*identity", re.IGNORECASE),
    "internal protocol label": re.compile(r"protocol[\s_-]*a", re.IGNORECASE),
    "Windows absolute path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"),
    "UNC path": re.compile(r"\\\\[^\\\s]+[\\/]"),
    "Unix absolute path": re.compile(
        r"(?<![A-Za-z0-9_])/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+"
    ),
    "internal evaluation path": re.compile(r"evaluation[\\/]", re.IGNORECASE),
}


class EvidenceError(RuntimeError):
    """Raised when an input or generated artifact violates the formal contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceError(f"Expected a JSON object: {path}")
    return payload


def wait_for_formal_evidence(
    *, timeout_seconds: float, poll_seconds: float
) -> dict[str, Any]:
    if timeout_seconds < 0 or poll_seconds <= 0:
        raise ValueError("Wait timeout must be nonnegative and poll interval positive")
    deadline = time.monotonic() + timeout_seconds
    last_reason = "formal evidence has not appeared"
    while True:
        if FORMAL_MANIFEST.is_file():
            try:
                manifest = load_json(FORMAL_MANIFEST)
            except (json.JSONDecodeError, OSError, EvidenceError) as exc:
                last_reason = f"formal manifest is not readable: {exc}"
            else:
                if manifest.get("status") != "complete":
                    last_reason = "formal manifest status is not complete"
                elif not FORMAL_GENE_LEVEL.is_file():
                    last_reason = "formal per-gene source is missing"
                else:
                    return manifest
        else:
            last_reason = "formal manifest is missing"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EvidenceError(
                f"Formal evidence is not ready ({last_reason}); no older source is allowed"
            )
        sleep_seconds = min(poll_seconds, remaining)
        print(
            f"Waiting {sleep_seconds:.0f}s for formal Protocol A evidence: {last_reason}",
            flush=True,
        )
        time.sleep(sleep_seconds)


def verify_manifest_file(path: Path, entry: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise EvidenceError(f"Missing {label}: {path}")
    expected_hash = str(entry.get("sha256", ""))
    if len(expected_hash) != 64:
        raise EvidenceError(f"Manifest has no valid SHA256 for {label}")
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise EvidenceError(
            f"{label} SHA256 mismatch: {observed_hash} != {expected_hash}"
        )


def resolve_five_fold_source(
    manifest: dict[str, Any],
    *,
    formal_path: Path = FORMAL_FIVE_FOLD,
    combined_path: Path = AUDITED_COMBINED_FIVE_FOLD,
) -> Path:
    """Select only an approved formal summary, never an older table source."""
    if formal_path.is_file():
        entry = manifest.get("outputs", {}).get(formal_path.name)
        if not isinstance(entry, dict):
            raise EvidenceError("Formal manifest does not audit formal_five_fold_metrics.csv")
        verify_manifest_file(formal_path, entry, "formal five-fold summary")
        return formal_path
    if combined_path.is_file():
        entry = manifest.get("inputs", {}).get("formal_reference")
        if not isinstance(entry, dict):
            raise EvidenceError("Formal manifest does not audit the combined reference")
        verify_manifest_file(combined_path, entry, "audited combined five-fold summary")
        return combined_path
    raise EvidenceError(
        "Neither formal_five_fold_metrics.csv nor its audited combined reference exists"
    )


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise EvidenceError(f"{label} lacks columns: {sorted(missing)}")


def validate_dataset_metadata(frame: pd.DataFrame, label: str) -> None:
    observed = set(
        frame[["dataset", "dataset_id", "role"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected = set(DATASETS)
    if observed != expected:
        raise EvidenceError(f"{label} dataset metadata differs: {observed ^ expected}")


def expected_layer(method: str) -> str:
    return SELECTED_LAYER if method == "GeneSPT" else RAW_LAYER


def validate_result_layers(frame: pd.DataFrame, label: str) -> None:
    pairs = set(
        frame[["method", "result_layer"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected = {(method, expected_layer(method)) for method in FORMAL_METHODS}
    if pairs != expected:
        raise EvidenceError(f"{label} result-layer contract differs: {pairs ^ expected}")


def validate_export_methods(frame: pd.DataFrame, label: str) -> None:
    present = set(frame["method"].dropna().astype(str))
    forbidden = present & FORBIDDEN_METHODS
    if forbidden:
        raise EvidenceError(f"{label} contains forbidden methods: {sorted(forbidden)}")
    unexpected = present - set(FORMAL_METHODS)
    if unexpected:
        raise EvidenceError(f"{label} contains unexpected methods: {sorted(unexpected)}")


def validate_formal_gene_level(
    frame: pd.DataFrame, manifest: dict[str, Any]
) -> pd.DataFrame:
    required = {
        "dataset",
        "dataset_id",
        "role",
        "fold",
        "method",
        "result_layer",
        "gene_idx",
        "gene_pos",
        "gene",
        "eligible_truth",
        "prediction_constant",
        "prediction_all_zero",
        *METRICS,
    }
    require_columns(frame, required, "formal per-gene source")
    expected_rows = manifest.get("outputs", {}).get(FORMAL_GENE_LEVEL.name, {}).get("rows")
    if not isinstance(expected_rows, int) or len(frame) != expected_rows:
        raise EvidenceError(
            f"Formal per-gene row count is {len(frame):,}; manifest expects {expected_rows}"
        )
    validate_dataset_metadata(frame, "formal per-gene source")
    validate_export_methods(frame, "formal per-gene source")
    if set(frame["method"].astype(str)) != set(FORMAL_METHODS):
        raise EvidenceError("Formal per-gene source does not contain all seven formal methods")
    validate_result_layers(frame, "formal per-gene source")
    if frame.duplicated(["dataset_id", "method", "gene_idx"]).any():
        raise EvidenceError("Formal per-gene source duplicates a dataset/method/gene")
    folds = pd.to_numeric(frame["fold"], errors="coerce")
    if folds.isna().any() or not folds.isin(range(5)).all():
        raise EvidenceError("Formal per-gene source contains an invalid fold")
    metric_values = frame[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    metric_array = metric_values.to_numpy(dtype=float)
    if np.isinf(metric_array).any():
        raise EvidenceError("Formal per-gene source contains an infinite metric")
    if not np.isfinite(metric_values[["RMSE", "SSIM"]].to_numpy(dtype=float)).all():
        raise EvidenceError("Formal per-gene RMSE/SSIM contains a nonfinite value")
    figure_mask = frame["dataset"].isin(FIGURE_DATASET_ORDER)
    if not np.isfinite(
        metric_values.loc[figure_mask, list(FIGURE_METRICS)].to_numpy(dtype=float)
    ).all():
        raise EvidenceError("Figure 5 formal per-gene metrics contain a nonfinite value")
    if frame["gene"].isna().any() or frame["gene"].astype(str).eq("").any():
        raise EvidenceError("Formal per-gene source contains an empty gene identifier")

    for dataset_id in DATASET_ID_ORDER:
        dataset_frame = frame[frame["dataset_id"].eq(dataset_id)]
        canonical = (
            dataset_frame[dataset_frame["method"].eq(FORMAL_METHODS[0])]
            .sort_values("gene_idx", kind="stable")
            [["gene_idx", "fold", "gene"]]
            .reset_index(drop=True)
        )
        if canonical.empty:
            raise EvidenceError(f"No formal per-gene rows for {dataset_id}")
        for method in FORMAL_METHODS[1:]:
            observed = (
                dataset_frame[dataset_frame["method"].eq(method)]
                .sort_values("gene_idx", kind="stable")
                [["gene_idx", "fold", "gene"]]
                .reset_index(drop=True)
            )
            if not canonical.equals(observed):
                raise EvidenceError(
                    f"Formal gene/fold axis differs for {dataset_id} and {method}"
                )
    result = frame.copy()
    result[list(METRICS)] = metric_values
    result["fold"] = folds.astype(int)
    return result


def validate_formal_five_fold(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "dataset_id",
        "role",
        "method",
        "folds",
        "coverage",
        *METRICS,
        *(f"{metric}_std_ddof0" for metric in METRICS),
    }
    require_columns(frame, required, "formal five-fold summary")
    formal = frame[frame["method"].isin(FORMAL_METHODS)].copy()
    if len(formal) != len(DATASETS) * len(FORMAL_METHODS):
        raise EvidenceError(f"Formal five-fold summary has {len(formal)} formal rows")
    if formal.duplicated(["dataset_id", "method"]).any():
        raise EvidenceError("Formal five-fold summary duplicates a dataset/method")
    validate_dataset_metadata(formal, "formal five-fold summary")
    validate_export_methods(formal, "formal five-fold summary")
    if set(formal["method"].astype(str)) != set(FORMAL_METHODS):
        raise EvidenceError("Formal five-fold summary does not contain all formal methods")
    validate_result_layers(formal, "formal five-fold summary")
    folds = pd.to_numeric(formal["folds"], errors="coerce")
    if folds.isna().any() or not folds.eq(5).all():
        raise EvidenceError("Formal five-fold summary is not complete across five folds")
    numeric_columns = [
        "coverage",
        *METRICS,
        *(f"{metric}_std_ddof0" for metric in METRICS),
    ]
    numeric = formal[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise EvidenceError("Formal five-fold summary contains a nonfinite value")
    if not numeric["coverage"].between(0.0, 1.0, inclusive="both").all():
        raise EvidenceError("Formal five-fold coverage falls outside [0, 1]")
    sd_columns = [f"{metric}_std_ddof0" for metric in METRICS]
    if numeric[sd_columns].lt(0.0).any().any():
        raise EvidenceError("Formal five-fold summary contains a negative fold SD")
    formal[numeric_columns] = numeric
    formal["folds"] = folds.astype(int)
    return formal


def build_figure_source(gene_level: pd.DataFrame) -> pd.DataFrame:
    subset = gene_level[gene_level["dataset"].isin(FIGURE_DATASET_ORDER)].copy()
    if not np.isfinite(subset[list(FIGURE_METRICS)].to_numpy(dtype=float)).all():
        raise EvidenceError("Figure 5 source contains a nonfinite formal value")
    id_columns = [
        "dataset",
        "dataset_id",
        "role",
        "method",
        "fold",
        "gene_idx",
        "gene_pos",
        "gene",
        "eligible_truth",
        "prediction_constant",
        "prediction_all_zero",
    ]
    source = subset.melt(
        id_vars=id_columns,
        value_vars=list(FIGURE_METRICS),
        var_name="raw_metric_name",
        value_name="raw_metric_value",
    )
    source["dataset_label"] = source["dataset"].map(FIGURE_DATASET_LABELS)
    source["display_metric"] = source["raw_metric_name"].map(
        {
            "SPCC": "1 - SPCC",
            "RMSE": "RMSE",
            "JSD": "JS/JSD",
            "SSIM": "1 - SSIM",
        }
    )
    inverted = source["raw_metric_name"].isin(DISPLAY_INVERTED_METRICS)
    source["display_transform"] = np.where(
        inverted, "1 - raw_metric_value", "identity"
    )
    source["display_value"] = np.where(
        inverted,
        1.0 - source["raw_metric_value"].to_numpy(dtype=float),
        source["raw_metric_value"].to_numpy(dtype=float),
    )
    source["lower_is_better"] = True
    source["_dataset_order"] = source["dataset"].map(
        {name: index for index, name in enumerate(FIGURE_DATASET_ORDER)}
    )
    source["_metric_order"] = source["raw_metric_name"].map(
        {name: index for index, name in enumerate(FIGURE_METRICS)}
    )
    source["_method_order"] = source["method"].map(
        {name: index for index, name in enumerate(METHOD_ORDER)}
    )
    source = source.sort_values(
        ["_dataset_order", "_metric_order", "_method_order", "fold", "gene_pos"],
        kind="stable",
    ).drop(columns=["_dataset_order", "_metric_order", "_method_order"])
    columns = [
        "dataset",
        "dataset_label",
        "dataset_id",
        "role",
        "method",
        "fold",
        "gene_idx",
        "gene_pos",
        "gene",
        "eligible_truth",
        "prediction_constant",
        "prediction_all_zero",
        "raw_metric_name",
        "raw_metric_value",
        "display_metric",
        "display_transform",
        "display_value",
        "lower_is_better",
    ]
    source = source[columns].reset_index(drop=True)

    expected_rows = len(subset) * len(FIGURE_METRICS)
    if len(source) != expected_rows or source["dataset_label"].isna().any():
        raise EvidenceError("Figure 5 source did not retain every formal per-gene value")
    expected_group_sizes = subset.groupby(["dataset_id", "method"]).size()
    observed_group_sizes = source.groupby(
        ["dataset_id", "method", "raw_metric_name"]
    ).size()
    for (dataset_id, method), count in expected_group_sizes.items():
        for metric in FIGURE_METRICS:
            if int(observed_group_sizes.get((dataset_id, method, metric), -1)) != int(count):
                raise EvidenceError(
                    f"Figure 5 lost rows for {dataset_id}/{method}/{metric}"
                )
    for metric in DISPLAY_INVERTED_METRICS:
        rows = source[source["raw_metric_name"].eq(metric)]
        if not np.allclose(
            rows["display_value"],
            1.0 - rows["raw_metric_value"],
            rtol=0.0,
            atol=1e-15,
        ):
            raise EvidenceError(
                f"Figure 5 {metric} display transform is not 1-{metric}"
            )
    return source


def build_s2(formal_five_fold: pd.DataFrame) -> pd.DataFrame:
    table = formal_five_fold.copy()
    for metric in METRICS:
        table[f"{metric}_mean"] = table[metric].astype(float)
        table[f"{metric}_fold_sd_ddof0"] = table[
            f"{metric}_std_ddof0"
        ].astype(float)
        ranks = table.groupby("dataset_id", sort=False)[metric].rank(
            method="min", ascending=metric in LOWER_IS_BETTER
        )
        if not np.equal(ranks, np.floor(ranks)).all():
            raise EvidenceError(f"Nonintegral {metric} rank")
        table[f"{metric}_rank"] = ranks.astype(int)
        table[f"{metric}_is_second_best"] = ranks.eq(2.0)
        for _, group in table.groupby("dataset_id", sort=False):
            if set(group[f"{metric}_rank"].astype(int)) != set(
                range(1, len(FORMAL_METHODS) + 1)
            ):
                raise EvidenceError(f"A {metric} tie makes second-best ambiguous")

    table["_dataset_order"] = table["dataset"].map(
        {name: index for index, name in enumerate(DATASET_ORDER)}
    )
    table["_method_order"] = table["method"].map(
        {name: index for index, name in enumerate(FORMAL_METHODS)}
    )
    table = table.sort_values(
        ["_dataset_order", "_method_order"], kind="stable"
    ).drop(columns=["_dataset_order", "_method_order"])
    result = table[list(S2_PUBLIC_COLUMNS)].reset_index(drop=True)
    validate_export_methods(result, "S2")
    return result


def configure_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titleweight": "regular",
            "axes.labelweight": "regular",
        }
    )


def plot_figure(source: pd.DataFrame, *, png_path: Path, pdf_path: Path) -> None:
    configure_figure_style()
    fig, axes = plt.subplots(
        3, 4, figsize=(15.8, 8.72), constrained_layout=False
    )
    fig.patch.set_facecolor("white")

    for row, dataset in enumerate(FIGURE_DATASET_ORDER):
        for column, metric in enumerate(FIGURE_METRICS):
            ax = axes[row, column]
            subset = source[
                source["dataset"].eq(dataset)
                & source["raw_metric_name"].eq(metric)
            ]
            if subset.empty:
                raise EvidenceError(f"No Figure 5 rows for {dataset}/{metric}")
            sns.violinplot(
                data=subset,
                x="method",
                y="display_value",
                order=list(METHOD_ORDER),
                palette=METHOD_COLORS,
                hue="method",
                hue_order=list(METHOD_ORDER),
                legend=False,
                inner="quartile",
                cut=0,
                density_norm="width",
                width=0.72,
                linewidth=0.65,
                saturation=1.0,
                ax=ax,
            )
            for index, collection in enumerate(ax.collections[: len(METHOD_ORDER)]):
                collection.set_alpha(0.72 if index == 0 else 0.45)

            ax.set_title(FIGURE_METRIC_LABELS[metric], fontsize=12.2, pad=8)
            ax.set_xlabel("")
            ax.set_xlim(-0.66, len(METHOD_ORDER) - 0.32)
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.455)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#777777")
            ax.spines["bottom"].set_color("#777777")
            if column == 0:
                ax.set_ylabel(
                    FIGURE_DATASET_LABELS[dataset],
                    fontsize=11.5,
                    fontweight="bold",
                    labelpad=12.5,
                )
            else:
                ax.set_ylabel("")
            ax.set_xticks(range(len(METHOD_ORDER)))
            if row == len(FIGURE_DATASET_ORDER) - 1:
                ax.set_xticklabels(METHOD_ORDER, rotation=38, ha="right", fontsize=9.3)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=9.2)

    handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="#333333",
            alpha=0.75,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=7,
        frameon=False,
        fontsize=10.6,
        handlelength=1.6,
        columnspacing=1.8,
    )
    fig.text(
        0.5,
        0.024,
        "Lower values indicate better prediction for all displayed metrics.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#333333",
    )
    fig.subplots_adjust(
        left=0.052,
        right=0.989,
        top=0.895,
        bottom=0.1385,
        wspace=0.29,
        hspace=0.46,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_caption(source: pd.DataFrame, path: Path) -> None:
    counts: dict[str, int] = {}
    gene_rows = source[source["raw_metric_name"].eq(FIGURE_METRICS[0])]
    for dataset in FIGURE_DATASET_ORDER:
        counts[dataset] = int(
            gene_rows[
                gene_rows["dataset"].eq(dataset)
                & gene_rows["method"].eq("GeneSPT")
            ]["gene_idx"].nunique()
        )
    caption = (
        "Figure 5. Per-gene prediction performance across cross-platform spatial "
        "transcriptomics datasets. Rows show seqFISH+ cortex/SVZ "
        f"({counts['seqFISH+']:,} genes), MHPR/MERFISH ({counts['MHPR']:,} genes), "
        f"and MVC/STARmap ({counts['MVC']:,} genes); columns show 1-SPCC, RMSE, "
        "JSD, and 1-SSIM. Violin plots contain every formal held-out per-gene value "
        "across the five frozen folds, without subsampling. SPCC and SSIM are "
        "displayed as 1-SPCC and 1-SSIM so lower values indicate better performance "
        "for all columns. Dashed interior "
        "lines denote Q1, the median, and Q3; y-axis ranges are scaled separately "
        "for readability.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption, encoding="utf-8")


def write_s2_readme(path: Path) -> None:
    text = f"""# Supplementary Table S2

`{S2_CSV.name}` is the single formal benchmark table. Its 42 rows cover six datasets and seven methods, with one row per dataset-method pair.

The table summarizes verified strict whole-gene holdout evidence.

## Reported fields

`dataset` is the manuscript-facing dataset name, `dataset_id` is the benchmark dataset identifier, and `role` distinguishes primary and cross-platform benchmark groups. `folds` is the number of frozen folds. `coverage` is the minimum fold coverage.

For SPCC, RMSE, JSD, and SSIM, each `*_mean` field is the arithmetic mean across the five fold summaries and each `*_fold_sd_ddof0` field is their population standard deviation. SPCC and SSIM ranks are descending; RMSE and JSD ranks are ascending. Rank 1 is best, and each `*_is_second_best` field marks rank 2.

Public filenames, byte sizes, row counts, and SHA256 checksums are recorded in `{S2_MANIFEST.name}`.
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
    *,
    path: Path,
    s2_path: Path,
    readme_path: Path,
    s2_rows: int,
) -> None:
    payload = {
        "schema_version": 2,
        "status": "complete",
        "description": (
            "Public Supplementary Table S2 artifacts for the six-dataset, "
            "seven-method formal benchmark."
        ),
        "artifacts": {
            "benchmark_table": public_file_record(s2_path, rows=s2_rows),
            "readme": public_file_record(readme_path),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _public_s2_outputs(s2_dir: Path) -> dict[str, Path]:
    return {
        "benchmark_table": s2_dir / S2_CSV.name,
        "readme": s2_dir / S2_README.name,
        "manifest": s2_dir / S2_MANIFEST.name,
    }


def _assert_public_s2_matches(path: Path, expected: pd.DataFrame) -> None:
    actual = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise EvidenceError("S2 scientific values changed during public export") from error


def validate_public_s2_artifacts(
    *, s2: pd.DataFrame, outputs: Mapping[str, Path]
) -> None:
    expected_keys = {"benchmark_table", "readme", "manifest"}
    if set(outputs) != expected_keys or not all(path.is_file() for path in outputs.values()):
        raise EvidenceError("Public S2 artifact set is incomplete")
    if list(s2.columns) != list(S2_PUBLIC_COLUMNS):
        raise EvidenceError("Public S2 schema changed")
    if len(s2) != len(DATASETS) * len(FORMAL_METHODS):
        raise EvidenceError(
            "Public S2 output does not contain one row per dataset-method pair"
        )

    expected_axis = [
        (dataset, dataset_id, role, method)
        for dataset, dataset_id, role in DATASETS
        for method in FORMAL_METHODS
    ]
    observed_axis = list(
        s2[["dataset", "dataset_id", "role", "method"]].itertuples(
            index=False, name=None
        )
    )
    if observed_axis != expected_axis:
        raise EvidenceError("Public S2 dataset/method order changed")
    for metric in METRICS:
        ranks = s2[f"{metric}_rank"].astype(int)
        expected_second = ranks.eq(2)
        if not s2[f"{metric}_is_second_best"].astype(bool).equals(expected_second):
            raise EvidenceError(f"Public S2 {metric} second-best flags changed")
        for _, group in s2.groupby("dataset_id", sort=False):
            if set(group[f"{metric}_rank"].astype(int)) != set(
                range(1, len(FORMAL_METHODS) + 1)
            ):
                raise EvidenceError(f"Public S2 {metric} rank contract changed")

    _assert_public_s2_matches(outputs["benchmark_table"], s2)

    s2_dir = outputs["manifest"].parent.resolve()
    expected_files = {path.resolve() for path in outputs.values()}
    observed_files = {
        path.resolve() for path in s2_dir.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        unexpected = sorted(str(path) for path in observed_files ^ expected_files)
        raise EvidenceError(f"Official S2 directory contains unexpected files: {unexpected}")

    for artifact_path in outputs.values():
        text = artifact_path.read_text(encoding="utf-8")
        for label, pattern in PUBLIC_S2_FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                raise EvidenceError(
                    f"Public S2 artifact {artifact_path.name} contains {label}"
                )

    manifest = load_json(outputs["manifest"])
    if set(manifest) != {"schema_version", "status", "description", "artifacts"}:
        raise EvidenceError("Public S2 manifest exposes non-public metadata")
    if manifest.get("schema_version") != 2 or manifest.get("status") != "complete":
        raise EvidenceError("Public S2 manifest identity is invalid")
    if not isinstance(manifest.get("description"), str) or not manifest["description"]:
        raise EvidenceError("Public S2 manifest description is missing")

    artifact_records = manifest.get("artifacts")
    expected_records = {
        "benchmark_table": (outputs["benchmark_table"], len(s2)),
        "readme": (outputs["readme"], None),
    }
    if not isinstance(artifact_records, Mapping) or set(artifact_records) != set(
        expected_records
    ):
        raise EvidenceError("Public S2 manifest artifact set changed")
    for key, (artifact_path, rows) in expected_records.items():
        record = artifact_records[key]
        required_fields = {"file", "bytes", "sha256"}
        if rows is not None:
            required_fields.add("rows")
        if not isinstance(record, Mapping) or set(record) != required_fields:
            raise EvidenceError(f"Public S2 manifest fields are invalid for {key}")
        if (
            record.get("file") != artifact_path.name
            or "/" in str(record.get("file", ""))
            or "\\" in str(record.get("file", ""))
            or int(record.get("bytes", -1)) != artifact_path.stat().st_size
            or record.get("sha256") != sha256_file(artifact_path)
        ):
            raise EvidenceError(f"Public S2 manifest record is invalid for {key}")
        if rows is not None and int(record.get("rows", -1)) != rows:
            raise EvidenceError(f"Public S2 row count is invalid for {key}")


def write_public_s2_bundle(
    *, s2: pd.DataFrame, s2_dir: Path = S2_DIR
) -> dict[str, Path]:
    outputs = _public_s2_outputs(Path(s2_dir))
    Path(s2_dir).mkdir(parents=True, exist_ok=True)
    write_csv(s2, outputs["benchmark_table"])
    write_s2_readme(outputs["readme"])
    write_generation_manifest(
        path=outputs["manifest"],
        s2_path=outputs["benchmark_table"],
        readme_path=outputs["readme"],
        s2_rows=len(s2),
    )
    validate_public_s2_artifacts(s2=s2, outputs=outputs)
    return outputs


def validate_figure_artifacts(figure_source: pd.DataFrame) -> None:
    expected_figure_rows = int(
        figure_source.groupby(["dataset_id", "method", "raw_metric_name"]).size().sum()
    )
    if len(figure_source) != expected_figure_rows:
        raise EvidenceError("Figure source row accounting failed")
    validate_export_methods(figure_source, "Figure 5 source")

    with Image.open(FIGURE_PNG) as image:
        if image.width < 3500 or image.height < 2400:
            raise EvidenceError(f"Figure 5 PNG is not high resolution: {image.size}")
        extrema = image.convert("RGB").getextrema()
        if all(low == high for low, high in extrema):
            raise EvidenceError("Figure 5 PNG is visually blank")
    if FIGURE_PDF.stat().st_size < 10_000 or not FIGURE_PDF.read_bytes().startswith(b"%PDF"):
        raise EvidenceError("Figure 5 PDF is missing or invalid")

    for output in (FIGURE_CAPTION,):
        content = output.read_text(encoding="utf-8")
        leaked = [name for name in FORBIDDEN_METHODS if name in content]
        if leaked:
            raise EvidenceError(f"Forbidden method labels leaked into {output}: {leaked}")


def validate_generated_artifacts(
    figure_source: pd.DataFrame,
    s2: pd.DataFrame,
    *,
    s2_outputs: Mapping[str, Path],
) -> None:
    validate_figure_artifacts(figure_source)
    if len(s2) != len(DATASETS) * len(FORMAL_METHODS):
        raise EvidenceError(
            "Formal S2 output does not contain one row per dataset-method pair"
        )
    validate_export_methods(s2, "S2")
    validate_public_s2_artifacts(s2=s2, outputs=s2_outputs)

    for output in s2_outputs.values():
        content = output.read_text(encoding="utf-8")
        leaked = [name for name in FORBIDDEN_METHODS if name in content]
        if leaked:
            raise EvidenceError(f"Forbidden method labels leaked into {output}: {leaked}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=900.0,
        help="Maximum time to wait for completed formal evidence (default: 900)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=15.0,
        help="Formal-evidence polling interval while waiting (default: 15)",
    )
    parser.add_argument(
        "--s2-only",
        action="store_true",
        help="Regenerate only the public Supplementary Table S2 bundle.",
    )
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="Regenerate only Figure 5 and its source/caption without rewriting S2.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.s2_only and args.figure_only:
        raise EvidenceError("--s2-only and --figure-only are mutually exclusive")
    formal_manifest = wait_for_formal_evidence(
        timeout_seconds=args.wait_timeout_seconds, poll_seconds=args.poll_seconds
    )
    five_fold_source = resolve_five_fold_source(formal_manifest)
    formal_five_fold = validate_formal_five_fold(
        pd.read_csv(five_fold_source, low_memory=False)
    )
    s2 = build_s2(formal_five_fold)
    if args.s2_only:
        write_public_s2_bundle(s2=s2, s2_dir=S2_DIR)
        print(f"Generated formal S2 rows: {len(s2)}")
        print(f"S2 directory: {S2_DIR}")
        return 0

    gene_entry = formal_manifest.get("outputs", {}).get(FORMAL_GENE_LEVEL.name)
    if not isinstance(gene_entry, dict):
        raise EvidenceError("Formal manifest does not audit formal_gene_level_metrics.csv")
    verify_manifest_file(FORMAL_GENE_LEVEL, gene_entry, "formal per-gene source")

    gene_level = validate_formal_gene_level(
        pd.read_csv(FORMAL_GENE_LEVEL, low_memory=False), formal_manifest
    )

    figure_source = build_figure_source(gene_level)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(figure_source, FIGURE_SOURCE)
    plot_figure(figure_source, png_path=FIGURE_PNG, pdf_path=FIGURE_PDF)
    write_caption(figure_source, FIGURE_CAPTION)
    if args.figure_only:
        validate_figure_artifacts(figure_source)
        print(f"Generated Figure 5 source rows: {len(figure_source):,}")
        print(f"Figure directory: {FIGURE_DIR}")
        return 0
    s2_outputs = write_public_s2_bundle(s2=s2, s2_dir=S2_DIR)
    validate_generated_artifacts(
        figure_source,
        s2,
        s2_outputs=s2_outputs,
    )

    print(f"Generated Figure 5 source rows: {len(figure_source):,}")
    print(f"Generated formal S2 rows: {len(s2)}")
    print(f"Figure directory: {FIGURE_DIR}")
    print(f"S2 directory: {S2_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
