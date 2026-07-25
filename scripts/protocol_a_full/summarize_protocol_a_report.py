#!/usr/bin/env python3
"""Create compact ranking tables from a complete Protocol A evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_DIRECTIONS = {
    "SPCC": "higher",
    "RMSE": "lower",
    "JSD": "lower",
    "SSIM": "higher",
}
ABLATION_METHODS = {"GeneSPT-GC"}
TARGET_METHOD = "GeneSPT"


def load_complete_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("complete") is not True:
        raise ValueError("Protocol A report is not complete")
    counts = report.get("counts", {})
    if counts.get("missing_runs") != 0 or counts.get("invalid_runs") != 0:
        raise ValueError("Protocol A report contains missing or invalid runs")
    return report


def build_metric_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    benchmark = summary[~summary["method"].isin(ABLATION_METHODS)].copy()
    rows = []
    for (dataset, dataset_id, role), group in benchmark.groupby(
        ["dataset", "dataset_id", "role"], sort=False
    ):
        for metric, direction in METRIC_DIRECTIONS.items():
            ordered = group.sort_values(
                [metric, "method"],
                ascending=[direction == "lower", True],
                kind="mergesort",
            ).reset_index(drop=True)
            ordered["rank"] = range(1, len(ordered) + 1)
            for record in ordered.to_dict(orient="records"):
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_id": dataset_id,
                        "role": role,
                        "metric": metric,
                        "direction": direction,
                        "method": record["method"],
                        "value": record[metric],
                        "rank": record["rank"],
                        "n_benchmark_methods": len(ordered),
                    }
                )
    return pd.DataFrame(rows)


def build_genespt_comparison(ranks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, dataset_id, role, metric), group in ranks.groupby(
        ["dataset", "dataset_id", "role", "metric"], sort=False
    ):
        target = group[group["method"].eq(TARGET_METHOD)]
        if len(target) != 1:
            raise ValueError(f"Expected one {TARGET_METHOD} row for {dataset}/{metric}")
        target_row = target.iloc[0]
        external = group[~group["method"].eq(TARGET_METHOD)].sort_values("rank")
        best_external = external.iloc[0]
        direction = str(target_row["direction"])
        gap = (
            float(target_row["value"]) - float(best_external["value"])
            if direction == "higher"
            else float(best_external["value"]) - float(target_row["value"])
        )
        rows.append(
            {
                "dataset": dataset,
                "dataset_id": dataset_id,
                "role": role,
                "metric": metric,
                "direction": direction,
                "GeneSPT_value": float(target_row["value"]),
                "GeneSPT_rank": int(target_row["rank"]),
                "best_external_method": str(best_external["method"]),
                "best_external_value": float(best_external["value"]),
                "GeneSPT_oriented_gap_vs_best_external": gap,
                "GeneSPT_rank1": int(target_row["rank"]) == 1,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(report_path: Path, output_dir: Path) -> None:
    report = load_complete_report(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(report["five_fold_summary"])
    columns = [
        "dataset",
        "dataset_id",
        "role",
        "method",
        "status",
        "SPCC",
        "RMSE",
        "JSD",
        "SSIM",
        "valid_gene_count",
        "missing_gene_count",
        "constant_prediction_count",
        "coverage",
    ]
    summary[columns].to_csv(output_dir / "raw_identity_five_fold_metrics.csv", index=False)
    ranks = build_metric_ranks(summary)
    ranks.to_csv(output_dir / "raw_identity_benchmark_ranks.csv", index=False)
    comparison = build_genespt_comparison(ranks)
    comparison.to_csv(output_dir / "raw_identity_genespt_comparison.csv", index=False)

    role_counts = (
        comparison.groupby("role", sort=False)["GeneSPT_rank1"]
        .agg(["sum", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    overview = {
        "source_report": str(report_path.resolve()),
        "source_report_sha256": report.get("sha256"),
        "report_counts": report["counts"],
        "benchmark_method_policy": "GeneSPT plus five external methods; GeneSPT-GC excluded as ablation",
        "rank1_counts": role_counts,
        "overall_rank1_items": int(comparison["GeneSPT_rank1"].sum()),
        "overall_items": int(len(comparison)),
    }
    (output_dir / "raw_identity_overview.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    write_outputs(args.report, args.output_dir)


if __name__ == "__main__":
    main()
