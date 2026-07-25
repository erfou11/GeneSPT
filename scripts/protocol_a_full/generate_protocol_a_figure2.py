#!/usr/bin/env python3
"""Generate Protocol A Figure 2 from the frozen formal evaluation table."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT
RUN_ROOT = PROJECT_ROOT / "results" / "protocol_a_full_rerun_20260711"
METRICS_CSV = (
    PROJECT_ROOT
    / "results"
    / "source_data"
    / "protocol_a"
    / "benchmark"
    / "formal_five_fold_metrics.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "figures" / "figure2"

FIGURE_STEM = "figure2_primary_benchmark_dotplot"
FORMAL_GENESPT_LAYER = "validation_selected_readout_genespt57"
RAW_BASELINE_LAYER = "raw_identity"

DATASET_ORDER = ("Vis9A", "HBC", "Cell2location")
DATASET_LABELS = {
    "Vis9A": "Vis9A",
    "HBC": "HBC",
    "Cell2location": "Cell2location\nmouse brain",
}
METHOD_ORDER = (
    "GeneSPT",
    "SpaIM",
    "Tangram",
    "TransImp",
    "SpaGE",
    "stPlus",
    "stAI",
)
EXTERNAL_METHODS = ("Tangram", "TransImp", "SpaIM", "SpaGE", "stPlus", "stAI")
METHOD_COLORS = {
    "GeneSPT": "#b2182b",
    "SpaIM": "#4c9b9b",
    "Tangram": "#59a14f",
    "TransImp": "#8b6fb3",
    "SpaGE": "#4c78a8",
    "stPlus": "#e6862f",
    "stAI": "#c49a00",
}
EXPECTED_GENESPT_SSIM_RANK = {"HBC": 4, "Cell2location": 2}

SOURCE_COLUMNS = (
    "dataset",
    "dataset_id",
    "role",
    "method",
    "result_layer",
    "folds",
    "SPCC",
    "RMSE",
    "JSD",
    "SSIM",
    "SPCC_std_ddof0",
    "RMSE_std_ddof0",
    "JSD_std_ddof0",
    "SSIM_std_ddof0",
    "coverage",
)


@dataclass(frozen=True)
class MetricSpec:
    column: str
    title: str
    higher_is_better: bool

    @property
    def rank_column(self) -> str:
        return f"{self.column}_rank"


METRICS = (
    MetricSpec("SPCC", "SPCC \u2191", True),
    MetricSpec("RMSE", "RMSE \u2193", False),
    MetricSpec("JSD", "JSD \u2193", False),
    MetricSpec("SSIM", "SSIM \u2191", True),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_relative_path(path: Path, package_root: Path = PACKAGE_ROOT) -> str:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(package_root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"Path is outside the package root: {resolved}") from error
    return relative.as_posix()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Figure 2 source is missing columns {missing}: {path}")


def _validate_coverage(frame: pd.DataFrame) -> None:
    folds = pd.to_numeric(frame["folds"], errors="raise").to_numpy(dtype=float)
    if not np.all(folds == 5):
        raise ValueError("Every Figure 2 row must be a five-fold summary.")
    coverage = pd.to_numeric(frame["coverage"], errors="raise").to_numpy(dtype=float)
    if not np.allclose(coverage, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Every Figure 2 row must have complete held-out coverage.")


def _validate_result_layers(frame: pd.DataFrame) -> None:
    genespt = frame[frame["method"].eq("GeneSPT")]
    if not genespt["result_layer"].eq(FORMAL_GENESPT_LAYER).all():
        raise ValueError(f"GeneSPT must come only from {FORMAL_GENESPT_LAYER}.")
    baselines = frame[frame["method"].isin(EXTERNAL_METHODS)]
    if not baselines["result_layer"].eq(RAW_BASELINE_LAYER).all():
        raise ValueError("Every external baseline must come from raw_identity.")


def _add_metric_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    for metric in METRICS:
        values = pd.to_numeric(ranked[metric.column], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Figure 2 metric {metric.column} contains non-finite values.")
        ranked[metric.column] = values.astype(float)
        ranked[metric.rank_column] = (
            ranked.assign(_metric_value=values)
            .groupby("dataset", sort=False)["_metric_value"]
            .rank(method="min", ascending=not metric.higher_is_better)
            .astype(int)
        )
    return ranked


def _validate_genespt_ssim_ranks(frame: pd.DataFrame) -> None:
    for dataset, expected_rank in EXPECTED_GENESPT_SSIM_RANK.items():
        row = frame[frame["dataset"].eq(dataset) & frame["method"].eq("GeneSPT")]
        observed = int(row.iloc[0]["SSIM_rank"])
        if observed != expected_rank:
            raise ValueError(
                f"Formal Figure 2 contract expects GeneSPT SSIM rank {expected_rank} "
                f"for {dataset}, observed {observed}."
            )


def load_primary_metrics(
    metrics_csv: Path = METRICS_CSV,
    *,
    package_root: Path = PACKAGE_ROOT,
) -> pd.DataFrame:
    """Load and validate the 21 formal primary benchmark rows."""

    metrics_csv = Path(metrics_csv)
    if not metrics_csv.is_file():
        raise FileNotFoundError(
            "Protocol A Figure 2 requires the formal combined five-fold table: "
            f"{metrics_csv}"
        )

    source = pd.read_csv(metrics_csv)
    _require_columns(source, SOURCE_COLUMNS, metrics_csv)
    primary = source[
        source["role"].eq("primary")
        & source["dataset"].isin(DATASET_ORDER)
        & source["method"].isin(METHOD_ORDER)
    ].copy()

    expected = {
        (dataset, method) for dataset in DATASET_ORDER for method in METHOD_ORDER
    }
    observed = set(zip(primary["dataset"], primary["method"]))
    if len(primary) != 21 or observed != expected:
        raise ValueError(
            "Figure 2 requires exactly one row for each primary dataset-method pair. "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}, "
            f"rows={len(primary)}"
        )
    if primary.duplicated(["dataset", "method"]).any():
        raise ValueError("Figure 2 source contains duplicate primary dataset-method rows.")

    _validate_coverage(primary)
    _validate_result_layers(primary)
    if not primary["role"].eq("primary").all():
        raise ValueError("Figure 2 contains a non-primary row.")

    dataset_position = {name: index for index, name in enumerate(DATASET_ORDER)}
    method_position = {name: index for index, name in enumerate(METHOD_ORDER)}
    primary["_dataset_order"] = primary["dataset"].map(dataset_position)
    primary["_method_order"] = primary["method"].map(method_position)
    primary = primary.sort_values(["_dataset_order", "_method_order"]).drop(
        columns=["_dataset_order", "_method_order"]
    )
    primary = _add_metric_ranks(primary).reset_index(drop=True)
    _validate_genespt_ssim_ranks(primary)

    source_hash = sha256_file(metrics_csv)
    primary["metric_source_package_relative_path"] = package_relative_path(
        metrics_csv, package_root
    )
    primary["metric_source_sha256"] = source_hash
    output_columns = list(SOURCE_COLUMNS) + [metric.rank_column for metric in METRICS]
    output_columns += [
        "metric_source_package_relative_path",
        "metric_source_sha256",
    ]
    return primary[output_columns]


def configure_fonts() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def axis_limits(values: pd.Series) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    low, high = float(np.min(array)), float(np.max(array))
    span = high - low
    if span == 0.0:
        span = max(abs(high), 1.0) * 0.05
    padding = span * 0.22
    return low - padding, high + padding


def plot_figure(frame: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_fonts()
    fig, axes = plt.subplots(
        len(DATASET_ORDER),
        len(METRICS),
        figsize=(14.8, 8.8),
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Primary sequencing-based benchmark performance",
        fontsize=18,
        fontweight="bold",
        y=0.988,
    )

    y_positions = np.arange(len(METHOD_ORDER))[::-1]
    y_lookup = dict(zip(METHOD_ORDER, y_positions))
    for row_index, dataset in enumerate(DATASET_ORDER):
        subset = frame[frame["dataset"].eq(dataset)]
        for column_index, metric in enumerate(METRICS):
            axis = axes[row_index, column_index]
            for method in METHOD_ORDER:
                record = subset[subset["method"].eq(method)].iloc[0]
                value = float(record[metric.column])
                size = 95 if method == "GeneSPT" else 70
                axis.scatter(
                    value,
                    y_lookup[method],
                    s=size,
                    color=METHOD_COLORS[method],
                    edgecolor="#333333",
                    linewidth=0.65,
                    zorder=3,
                )
                if int(record[metric.rank_column]) == 1:
                    axis.scatter(
                        value,
                        y_lookup[method],
                        s=size + 95,
                        facecolors="none",
                        edgecolors="#111111",
                        linewidth=1.25,
                        zorder=4,
                    )

            axis.set_xlim(*axis_limits(subset[metric.column]))
            axis.set_ylim(-0.6, len(METHOD_ORDER) - 0.4)
            axis.set_title(metric.title, fontsize=12.0, pad=7)
            axis.grid(axis="x", color="#dddddd", linewidth=0.65, alpha=0.7)
            axis.grid(axis="y", color="#eeeeee", linewidth=0.5, alpha=0.55)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#888888")
            axis.spines["bottom"].set_color("#888888")
            axis.tick_params(axis="x", labelsize=8.8)
            axis.tick_params(axis="y", length=0)
            axis.set_yticks(y_positions)
            if column_index == 0:
                axis.set_yticklabels(METHOD_ORDER, fontsize=9.6)
                axis.set_ylabel(
                    DATASET_LABELS[dataset],
                    fontsize=12.0,
                    fontweight="bold",
                    labelpad=16,
                )
            else:
                axis.set_yticklabels([])

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="#333333",
            markersize=7.5 if method == "GeneSPT" else 6.8,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    best_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="none",
        markeredgecolor="#111111",
        markeredgewidth=1.25,
        markersize=9.0,
        label="Best method",
    )
    fig.legend(
        handles=method_handles + [best_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.943),
        ncol=8,
        frameon=False,
        fontsize=10.2,
        columnspacing=1.45,
        handletextpad=0.55,
    )
    fig.text(
        0.5,
        0.025,
        "Dots show five-fold mean metrics on the original evaluation scale. Axes are locally scaled within panels for readability.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#333333",
    )
    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.86,
        bottom=0.095,
        wspace=0.27,
        hspace=0.44,
    )
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_caption(path: Path) -> None:
    path.write_text(
        "**Figure 2. Primary sequencing-based benchmark performance under "
        "strict whole-gene holdout.** Dots show five-fold mean metrics on the "
        "original evaluation scale for Vis9A, HBC, and Cell2location mouse "
        "brain. The formal comparison includes GeneSPT, Tangram, TransImp, "
        "SpaIM, SpaGE, stPlus, and stAI, evaluated with the unified evaluator; "
        "the black open circle marks the best method within each "
        "dataset and metric. Higher values are better for SPCC and SSIM, and "
        "lower values are better for RMSE and JSD. GeneSPT ranks fourth for HBC "
        "SSIM and second for Cell2location SSIM. Axes are locally scaled within "
        "panels for readability.\n",
        encoding="utf-8",
    )


def generate(
    *,
    metrics_csv: Path = METRICS_CSV,
    output_dir: Path = OUTPUT_DIR,
    package_root: Path = PACKAGE_ROOT,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"{FIGURE_STEM}_source.csv"
    pdf_path = output_dir / f"{FIGURE_STEM}.pdf"
    png_path = output_dir / f"{FIGURE_STEM}.png"
    caption_path = output_dir / f"{FIGURE_STEM}_caption_draft.md"

    frame = load_primary_metrics(Path(metrics_csv), package_root=Path(package_root))
    frame.to_csv(source_path, index=False)
    plot_figure(frame, pdf_path, png_path)
    write_caption(caption_path)
    return {
        "pdf": pdf_path,
        "png": png_path,
        "source": source_path,
        "caption": caption_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = generate(
        metrics_csv=args.metrics_csv,
        output_dir=args.output_dir,
        package_root=args.package_root,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
