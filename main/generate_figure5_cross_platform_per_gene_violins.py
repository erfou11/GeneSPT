#!/usr/bin/env python3
"""Generate cross-platform per-gene violin plots from existing metric files."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_output"

RAW_FIG5_SOURCE = OUT / "figure5_cross_platform_raw_metrics_source.csv"
SEQ_GENE = (
    OUT
    / "seqfish_trials/seqFISH_plus_cortex_svz/quick_run_zeisel_sccortex/genespt_quick_gene_level.csv"
)
SEQ_EXTERNAL_ROOT = (
    OUT / "seqfish_trials/seqFISH_plus_cortex_svz/external_baselines_zeisel_sccortex"
)
FINAL_AVAILABLE = ROOT / "results/imformation/final_available_datasets_four_metric_gene_level.csv"
MVC_SPAIM_FIX = OUT / "mvc_spaim_fix/mvc_spaim_gene_level_metrics.csv"
PREVIOUS_VIOLIN_AUDIT = OUT / "supp_figure_per_gene_metric_violins_source.csv"

METHOD_ORDER = ["GeneSPT", "SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus"]
METHOD_LABELS = {
    "GeneSPT": "GeneSPT",
    "SpaIM": "SpaIM",
    "Tangram": "Tangram",
    "TransPA": "TransImp",
    "SpaGE": "SpaGE",
    "stPlus": "stPlus",
}
METHOD_COLORS = {
    "GeneSPT": "#b2182b",
    "SpaIM": "#4c9b9b",
    "Tangram": "#59a14f",
    "TransPA": "#8b6fb3",
    "SpaGE": "#4c78a8",
    "stPlus": "#e6862f",
}
DATASETS = [
    ("seqFISH+ cortex/SVZ", "seqFISH+ cortex/SVZ", "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000"),
    ("MHPR", "MHPR/MERFISH", "MHPR_current_panel"),
    ("MVC", "MVC/STARmap", "MVC_shared981"),
]
METRICS = [
    ("1 - SPCC", "SPCC", "1 - SPCC", lambda s: 1.0 - s),
    ("RMSE", "RMSE", "RMSE", lambda s: s),
    ("JS/JSD", "JS", "JS/JSD", lambda s: s),
    ("1 - SSIM", "SSIM", "1 - SSIM", lambda s: 1.0 - s),
]


def configure_fonts() -> None:
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


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")


def standardize_gene_metrics(
    df: pd.DataFrame,
    dataset: str,
    dataset_label: str,
    dataset_id: str,
    method: str,
    source_file: Path,
    fold_col: str = "fold",
) -> pd.DataFrame:
    needed = {fold_col, "gene", "SPCC", "RMSE", "JS", "SSIM"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{source_file} missing required columns: {sorted(missing)}")
    out = df[[fold_col, "gene", "SPCC", "RMSE", "JS", "SSIM"]].copy()
    out = out.rename(columns={fold_col: "fold"})
    out["fold"] = out["fold"].astype(int)
    out["dataset"] = dataset
    out["dataset_label"] = dataset_label
    out["dataset_id"] = dataset_id
    out["method"] = method
    out["source_file"] = str(source_file)
    return out[
        [
            "dataset",
            "dataset_label",
            "dataset_id",
            "method",
            "fold",
            "gene",
            "SPCC",
            "RMSE",
            "JS",
            "SSIM",
            "source_file",
        ]
    ]


def load_seqfishplus() -> list[pd.DataFrame]:
    require(SEQ_GENE)
    gene = pd.read_csv(SEQ_GENE)
    gene = gene[
        gene["model"].isin(["genespt_gc_psp_correct", "topodist_gc_psp_correct"])
        & gene["split"].eq("test")
        & gene["role"].eq("selected")
    ].copy()
    frames = [
        standardize_gene_metrics(
            gene,
            "seqFISH+ cortex/SVZ",
            "seqFISH+ cortex/SVZ",
            "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000",
            "GeneSPT",
            SEQ_GENE,
        )
    ]

    for method in ["SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus"]:
        method_frames = []
        for fold in range(5):
            path = SEQ_EXTERNAL_ROOT / method / f"fold{fold}" / "gene_level_metrics_stdiff_style.csv"
            require(path)
            df = pd.read_csv(path)
            df["fold"] = fold
            method_frames.append(df)
        frames.append(
            standardize_gene_metrics(
                pd.concat(method_frames, ignore_index=True),
                "seqFISH+ cortex/SVZ",
                "seqFISH+ cortex/SVZ",
                "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000",
                method,
                SEQ_EXTERNAL_ROOT / method,
            )
        )
    return frames


def load_mhpr_mvc() -> list[pd.DataFrame]:
    require(FINAL_AVAILABLE)
    df = pd.read_csv(FINAL_AVAILABLE)
    frames = []
    method_map = {
        "TopoDiST-GC-PSP": "GeneSPT",
        "SpaIM": "SpaIM",
        "Tangram": "Tangram",
        "TransPA": "TransPA",
        "SpaGE": "SpaGE",
        "stPlus": "stPlus",
    }
    for dataset, label, dataset_id in [
        ("MHPR", "MHPR/MERFISH", "MHPR_current_panel"),
        ("MVC", "MVC/STARmap", "MVC_shared981"),
    ]:
        for raw_method, display_method in method_map.items():
            if dataset == "MVC" and display_method == "SpaIM":
                continue
            sub = df[df["dataset"].eq(dataset_id) & df["method"].eq(raw_method)].copy()
            if sub.empty:
                raise RuntimeError(f"No per-gene rows found for {dataset_id} / {raw_method}")
            frames.append(
                standardize_gene_metrics(
                    sub,
                    dataset,
                    label,
                    dataset_id,
                    display_method,
                    FINAL_AVAILABLE,
                )
            )

    require(MVC_SPAIM_FIX)
    mvc_spaim = pd.read_csv(MVC_SPAIM_FIX)
    frames.append(
        standardize_gene_metrics(
            mvc_spaim,
            "MVC",
            "MVC/STARmap",
            "MVC_shared981",
            "SpaIM",
            MVC_SPAIM_FIX,
        )
    )
    return frames


def build_per_gene_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = load_seqfishplus() + load_mhpr_mvc()
    wide = pd.concat(frames, ignore_index=True)
    wide = wide[wide["method"].isin(METHOD_ORDER)].copy()

    records = []
    for display_metric, raw_metric, metric_label, transform in METRICS:
        tmp = wide[
            [
                "dataset",
                "dataset_label",
                "dataset_id",
                "method",
                "fold",
                "gene",
                raw_metric,
                "source_file",
            ]
        ].copy()
        tmp = tmp.rename(columns={raw_metric: "raw_metric_value"})
        tmp["metric"] = display_metric
        tmp["metric_label"] = metric_label
        tmp["raw_metric_name"] = raw_metric
        tmp["value"] = transform(tmp["raw_metric_value"].astype(float))
        tmp["lower_is_better"] = True
        records.append(tmp)
    long = pd.concat(records, ignore_index=True)
    long["method"] = pd.Categorical(long["method"], METHOD_ORDER, ordered=True)
    long["dataset"] = pd.Categorical(long["dataset"], [d[0] for d in DATASETS], ordered=True)
    long["metric"] = pd.Categorical(long["metric"], [m[0] for m in METRICS], ordered=True)
    long = long.sort_values(["dataset", "metric", "method", "fold", "gene"]).reset_index(drop=True)
    return wide, long


def validate_against_raw_source(wide: pd.DataFrame) -> pd.DataFrame:
    require(RAW_FIG5_SOURCE)
    raw = pd.read_csv(RAW_FIG5_SOURCE)
    checks = []
    for (dataset, method), sub in wide.groupby(["dataset", "method"], observed=True):
        raw_sub = raw[raw["dataset"].eq(dataset) & raw["method"].eq(method)]
        for metric, raw_col, _, _ in METRICS:
            expected = raw_sub.loc[raw_sub["metric"].eq(raw_col if raw_col != "JS" else "JS/JSD"), "raw_value"]
            if expected.empty:
                expected_value = np.nan
            else:
                expected_value = float(expected.iloc[0])
            observed_value = float(sub.groupby("fold")[raw_col].median().mean())
            checks.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "raw_metric": raw_col,
                    "folds": int(sub["fold"].nunique()),
                    "genes_total": int(len(sub)),
                    "genes_by_fold": ";".join(
                        f"fold{int(k)}={int(v)}" for k, v in sub.groupby("fold").size().items()
                    ),
                    "observed_fold_median_mean": observed_value,
                    "figure5_raw_source_value": expected_value,
                    "abs_diff_vs_source": abs(observed_value - expected_value),
                    "complete_five_folds": sub["fold"].nunique() == 5,
                }
            )
    return pd.DataFrame(checks)


def plot_violins(long: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16.4, 9.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.suptitle("Per-gene cross-platform prediction errors", fontsize=17, fontweight="bold", y=0.985)

    positions = np.arange(1, len(METHOD_ORDER) + 1)
    for r, (dataset, label, _) in enumerate(DATASETS):
        for c, (display_metric, _, metric_label, _) in enumerate(METRICS):
            ax = axes[r, c]
            subset = long[
                long["dataset"].astype(str).eq(dataset)
                & long["metric"].astype(str).eq(display_metric)
            ].copy()
            arrays = [
                subset.loc[subset["method"].astype(str).eq(method), "value"].replace([np.inf, -np.inf], np.nan).dropna().values
                for method in METHOD_ORDER
            ]
            vp = ax.violinplot(
                arrays,
                positions=positions,
                widths=0.78,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, method in zip(vp["bodies"], METHOD_ORDER):
                body.set_facecolor(METHOD_COLORS[method])
                body.set_edgecolor("#333333")
                body.set_linewidth(0.65)
                body.set_alpha(0.72 if method == "GeneSPT" else 0.48)

            bp = ax.boxplot(
                arrays,
                positions=positions,
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "#555555", "linewidth": 0.7},
                capprops={"color": "#555555", "linewidth": 0.7},
            )
            for patch, method in zip(bp["boxes"], METHOD_ORDER):
                patch.set_facecolor("white")
                patch.set_edgecolor(METHOD_COLORS[method])
                patch.set_linewidth(1.0)
                patch.set_alpha(0.95)

            ax.set_title(metric_label, fontsize=12.2, pad=8)
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.75)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#777777")
            ax.spines["bottom"].set_color("#777777")
            if c == 0:
                ax.set_ylabel(label, fontsize=11.5, fontweight="bold", labelpad=14)
            if r == len(DATASETS) - 1:
                ax.set_xticks(positions)
                ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=38, ha="right", fontsize=9.3)
            else:
                ax.set_xticks(positions)
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=9.2)

    handles = [
        Patch(facecolor=METHOD_COLORS[m], edgecolor="#333333", alpha=0.72, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=6,
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
    fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.13, wspace=0.28, hspace=0.43)
    fig.savefig(OUT / "figure5_cross_platform_per_gene_violins.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure5_cross_platform_per_gene_violins.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(long: pd.DataFrame, checks: pd.DataFrame) -> None:
    source_path = OUT / "figure5_cross_platform_per_gene_violins_source.csv"
    long.to_csv(source_path, index=False)
    checks.to_csv(OUT / "figure5_cross_platform_per_gene_violins_completeness_check.csv", index=False)

    (OUT / "figure5_cross_platform_per_gene_violins_caption.md").write_text(
        "Figure 5. Per-gene prediction performance across cross-platform spatial "
        "transcriptomics datasets. Violin plots show metric distributions over held-out "
        "test genes under the strict whole-gene evaluation protocol. SPCC and SSIM "
        "are shown as 1-SPCC and 1-SSIM so that lower values consistently indicate "
        "better prediction across all panels. GeneSPT is compared with complete external "
        "baselines using the same frozen test-gene splits.\n"
    )

    files_used = sorted(long["source_file"].unique())
    counts = (
        long.drop_duplicates(["dataset", "method", "fold", "gene"])
        .groupby(["dataset_label", "method"], observed=True)
        .agg(folds=("fold", "nunique"), genes=("gene", "size"))
        .reset_index()
    )
    counts_text = "\n".join(
        f"- {row.dataset_label} / {row.method}: {int(row.genes)} genes across {int(row.folds)} folds"
        for row in counts.itertuples(index=False)
    )
    files_text = "\n".join(f"- `{path}`" for path in files_used)
    max_diff = checks["abs_diff_vs_source"].max()
    all_six_note = (
        "The optional all-six-dataset supplement was not generated because existing "
        "audit/source files indicate the final primary-dataset readout gene-level "
        "metrics are incomplete or version-mismatched relative to the final summary "
        "tables; no values were fabricated."
    )
    if PREVIOUS_VIOLIN_AUDIT.exists():
        all_six_note += f" Prior audit file consulted: `{PREVIOUS_VIOLIN_AUDIT}`."

    (OUT / "figure5_per_gene_violin_changelog.md").write_text(
        "# Figure 5 Per-Gene Violin Changelog\n\n"
        "## Per-gene metric files used\n"
        f"{files_text}\n\n"
        "## Completeness\n"
        f"{counts_text}\n"
        f"- Maximum absolute difference between fold-median means from the per-gene files "
        f"and `figure5_cross_platform_raw_metrics_source.csv`: {max_diff:.6g}.\n"
        "- All included dataset-method combinations use five folds and complete available frozen test-gene rows.\n\n"
        "## Metric transformations\n"
        "- Used SSIM values from per-gene `SSIM` columns; no additional scaling convention was applied.\n"
        "- Transformed SPCC to `1 - SPCC` and SSIM to `1 - SSIM` for the displayed violin metrics.\n"
        "- RMSE and JS/JSD were displayed on their original per-gene scale.\n\n"
        "## Exclusions and optional outputs\n"
        "- Excluded internal ablation labels and stDiff from the figure.\n"
        "- No dataset-method combination in the three cross-platform panels was excluded for missing per-gene metrics.\n"
        f"- {all_six_note}\n\n"
        "## Run constraints\n"
        "- No model was rerun.\n"
        "- No prediction matrix was modified or recomputed.\n"
        "- No manuscript file was modified.\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    wide, long = build_per_gene_long()
    checks = validate_against_raw_source(wide)

    incomplete = checks[~checks["complete_five_folds"]]
    if not incomplete.empty:
        raise RuntimeError(
            "Incomplete fold coverage detected:\n"
            + incomplete[["dataset", "method", "raw_metric", "genes_by_fold"]].to_string(index=False)
        )
    if checks["abs_diff_vs_source"].max() > 1e-4:
        raise RuntimeError(
            "Per-gene summaries do not match Figure 5 raw source closely enough. "
            f"Max abs diff = {checks['abs_diff_vs_source'].max()}"
        )

    write_outputs(long, checks)
    plot_violins(long)
    print("Generated cross-platform per-gene violin Figure 5 candidate.")


if __name__ == "__main__":
    main()
