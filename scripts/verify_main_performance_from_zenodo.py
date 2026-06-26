#!/usr/bin/env python3
"""Verify main benchmark source values from the Zenodo data package.

This is a reviewer-facing source-value check. It does not retrain models and it
does not claim to recompute metrics from missing prediction matrices. Instead,
it verifies that the archived Zenodo source tables can drive the public Figure 2
source-generation path and reproduce the final primary benchmark source values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_INFO = ROOT / "results" / "imformation"
FIGURE2_OUT = ROOT / "final_output" / "final_main_results" / "figure2_primary_benchmark_dotplot_source.csv"
LEGACY_TABLE1 = RESULTS_INFO / "table1_primary_benchmark_final.csv"

PRIMARY_DATASETS = ["Vis9A", "HBC", "Cell2location mouse brain"]
PRIMARY_DATASET_TO_FIGURE2 = {"Cell2location mouse brain": "Cell2location"}
FIGURE2_METRIC_TO_PRIMARY = {
    "SPCC": "SPCC_mean",
    "RMSE": "RMSE_mean",
    "JS": "JS_JSD_mean",
    "raw_SSIM": "raw_SSIM_mean",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")


def read_zenodo_sources(zenodo_root: Path) -> dict[str, Path]:
    paths = {
        "primary": zenodo_root / "results_source_data" / "benchmark_metrics_primary_clean.csv",
        "cross_platform": zenodo_root / "results_source_data" / "benchmark_metrics_cross_platform_clean.csv",
        "figure2": zenodo_root
        / "results_source_data"
        / "final_figure_source_csv"
        / "figure2_primary_benchmark_dotplot_source.csv",
        "figure3_panel_b": zenodo_root / "results_source_data" / "figure3_panelB_psp_ablation_clean.csv",
        "supp_s2": zenodo_root
        / "results_source_data"
        / "supplementary_table_s2_full_benchmark_metrics_and_method_availability.csv",
    }
    for path in paths.values():
        require_file(path)
    return paths


def write_legacy_table1(primary_df: pd.DataFrame) -> Path:
    RESULTS_INFO.mkdir(parents=True, exist_ok=True)
    out = primary_df.copy()
    out["dataset"] = out["dataset"].replace(PRIMARY_DATASET_TO_FIGURE2)
    out = out.rename(
        columns={
            "JS_JSD_mean": "JS_mean",
            "raw_SSIM_mean": "SSIM_mean",
        }
    )
    keep = ["dataset", "method", "SPCC_mean", "RMSE_mean", "JS_mean", "SSIM_mean", "status", "folds"]
    out[keep].to_csv(LEGACY_TABLE1, index=False)
    return LEGACY_TABLE1


def run_figure2_script() -> None:
    cmd = [sys.executable, str(ROOT / "main" / "generate_figure2_primary_dotplot.py")]
    subprocess.run(cmd, cwd=ROOT, check=True)


def compare_figure2_to_primary(figure2_df: pd.DataFrame, primary_df: pd.DataFrame) -> pd.DataFrame:
    primary = primary_df.copy()
    primary["dataset_figure2"] = primary["dataset"].replace(PRIMARY_DATASET_TO_FIGURE2)
    rows = []
    for row in figure2_df.itertuples(index=False):
        primary_col = FIGURE2_METRIC_TO_PRIMARY[str(row.metric)]
        matched = primary[
            primary["dataset_figure2"].eq(str(row.dataset))
            & primary["method"].eq(str(row.method))
        ]
        if matched.empty:
            rows.append(
                {
                    "dataset": row.dataset,
                    "method": row.method,
                    "metric": row.metric,
                    "figure2_value": float(row.raw_value),
                    "primary_value": None,
                    "abs_diff": None,
                    "status": "missing_primary_row",
                }
            )
            continue
        primary_value = float(matched.iloc[0][primary_col])
        diff = abs(float(row.raw_value) - primary_value)
        rows.append(
            {
                "dataset": row.dataset,
                "method": row.method,
                "metric": row.metric,
                "figure2_value": float(row.raw_value),
                "primary_value": primary_value,
                "abs_diff": diff,
                "status": "ok" if diff <= 1e-12 else "diff",
            }
        )
    return pd.DataFrame(rows)


def format_primary_genespt_table(primary_df: pd.DataFrame) -> str:
    cols = ["dataset", "SPCC_mean", "raw_SSIM_mean", "RMSE_mean", "JS_JSD_mean", "folds"]
    rows = primary_df[
        primary_df["dataset"].isin(PRIMARY_DATASETS) & primary_df["method"].eq("GeneSPT")
    ][cols].copy()
    rows = rows.set_index("dataset").loc[PRIMARY_DATASETS].reset_index()
    return rows.to_string(index=False)


def write_report(
    out_dir: Path,
    zenodo_root: Path,
    paths: dict[str, Path],
    primary_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    consistency: pd.DataFrame,
    figure2_hash_match: bool | None,
    generated_hash: str | None,
    zenodo_hash: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_genespt = primary_df[
        primary_df["dataset"].isin(PRIMARY_DATASETS) & primary_df["method"].eq("GeneSPT")
    ].copy()
    primary_genespt.to_csv(out_dir / "primary_genespt_metrics.csv", index=False)
    consistency.to_csv(out_dir / "figure2_vs_primary_consistency.csv", index=False)

    summary = {
        "zenodo_root": str(zenodo_root),
        "primary_rows": int(len(primary_df)),
        "cross_platform_rows": int(len(cross_df)),
        "figure2_vs_primary_rows": int(len(consistency)),
        "figure2_vs_primary_max_abs_diff": None
        if consistency["abs_diff"].dropna().empty
        else float(consistency["abs_diff"].dropna().max()),
        "figure2_generated_sha256": generated_hash,
        "figure2_zenodo_sha256": zenodo_hash,
        "figure2_exact_hash_match": figure2_hash_match,
        "full_prediction_matrix_recompute": "not_attempted_missing_prediction_matrices",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Main Performance Source-Value Verification",
        "",
        f"Zenodo root: `{zenodo_root}`",
        "",
        "## Decision",
        "",
    ]
    if (consistency["status"] == "ok").all() and (figure2_hash_match is not False):
        lines.append("PASS: archived main benchmark source values are internally consistent.")
    else:
        lines.append("FAIL: at least one main benchmark source-value check failed.")
    lines.extend(
        [
            "",
            "This report verifies source values, not full metric recomputation from prediction matrices.",
            "Full recomputation requires final prediction matrices, which are not included in the Zenodo data package.",
            "",
            "## Primary GeneSPT Metrics",
            "",
            "```text",
            format_primary_genespt_table(primary_df),
            "```",
            "",
            "## Figure 2 Source Check",
            "",
            f"Zenodo Figure 2 source SHA256: `{zenodo_hash}`",
        ]
    )
    if generated_hash is None:
        lines.append("Generated Figure 2 source SHA256: not generated in this run.")
    else:
        lines.append(f"Generated Figure 2 source SHA256: `{generated_hash}`")
        lines.append(f"Exact hash match: `{figure2_hash_match}`")
    lines.extend(
        [
            "",
            f"Figure 2 vs primary table rows checked: `{len(consistency)}`",
            f"Maximum absolute difference: `{summary['figure2_vs_primary_max_abs_diff']}`",
            "",
            "## Source Files",
            "",
        ]
    )
    for key, path in paths.items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This script checks the primary benchmark source values used by Table 2/Figure 2.",
            "- It does not run full model training.",
            "- It does not recompute all methods from prediction matrices because those matrices are not archived here.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify main GeneSPT source values from a Zenodo extraction.")
    parser.add_argument("--zenodo-root", type=Path, required=True, help="Extracted GeneSPT_manuscript_data_20260610 directory.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "reproduction" / "main_performance_source_check",
        help="Directory for verification reports.",
    )
    parser.add_argument(
        "--write-legacy-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write results/imformation/table1_primary_benchmark_final.csv for the anchored Figure 2 script.",
    )
    parser.add_argument(
        "--run-figure2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run main/generate_figure2_primary_dotplot.py and compare the generated source CSV.",
    )
    args = parser.parse_args()

    zenodo_root = args.zenodo_root.resolve()
    paths = read_zenodo_sources(zenodo_root)
    primary_df = pd.read_csv(paths["primary"])
    cross_df = pd.read_csv(paths["cross_platform"])

    required_primary_methods = {"GeneSPT", "GeneSPT-GC", "SpaGE", "Tangram", "stPlus", "TransPA", "SpaIM"}
    observed_primary = set(primary_df["method"].astype(str))
    missing_methods = sorted(required_primary_methods - observed_primary)
    if missing_methods:
        raise ValueError(f"Missing primary benchmark methods: {missing_methods}")

    if args.write_legacy_inputs:
        legacy = write_legacy_table1(primary_df)
        print(f"[verify] wrote legacy Figure 2 input: {legacy}")

    generated_hash = None
    figure2_hash_match = None
    if args.run_figure2:
        if not LEGACY_TABLE1.exists():
            raise FileNotFoundError(f"Figure 2 input is missing: {LEGACY_TABLE1}")
        run_figure2_script()
        require_file(FIGURE2_OUT)
        generated_hash = sha256(FIGURE2_OUT)
        figure2_hash_match = generated_hash == sha256(paths["figure2"])

    figure2_df = pd.read_csv(FIGURE2_OUT if args.run_figure2 else paths["figure2"])
    consistency = compare_figure2_to_primary(figure2_df, primary_df)
    zenodo_hash = sha256(paths["figure2"])
    write_report(
        args.out_dir,
        zenodo_root,
        paths,
        primary_df,
        cross_df,
        consistency,
        figure2_hash_match,
        generated_hash,
        zenodo_hash,
    )

    print("[verify] primary GeneSPT metrics")
    print(format_primary_genespt_table(primary_df))
    print(f"[verify] figure2_vs_primary_max_abs_diff={consistency['abs_diff'].dropna().max():.6g}")
    if generated_hash is not None:
        print(f"[verify] figure2_generated_sha256={generated_hash}")
        print(f"[verify] figure2_zenodo_sha256={zenodo_hash}")
        print(f"[verify] figure2_exact_hash_match={figure2_hash_match}")
    print(f"[verify] wrote report: {args.out_dir / 'README.md'}")

    if not (consistency["status"] == "ok").all():
        raise SystemExit(1)
    if figure2_hash_match is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
