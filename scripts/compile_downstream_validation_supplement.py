#!/usr/bin/env python3
"""Compile standardized downstream validation supplement from existing audits.

This script only reads existing summary/audit CSV files and writes derived
tables, figures, captions and cautious wording notes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/workspace/GeneSPT")
OUT = ROOT / "final_output" / "downstream_validation_supplement"
OUT.mkdir(parents=True, exist_ok=True)

UP = ROOT / "final_output" / "downstream_upgraded_labels"
LEIDEN = ROOT / "final_output" / "downstream_leiden_sensitivity"
GSA = ROOT / "final_output" / "gsa_downstream_prototype"
RESCUE = ROOT / "final_output" / "downstream_rescue_audit"

IMPUTATION_METHODS = ["GeneSPT", "SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus"]
REFERENCE_METHODS = ["Observed-only", "Full-ST upper"]
METHOD_ORDER = ["Observed-only", "GeneSPT", "SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus", "Full-ST upper"]
COLORS = {
    "Observed-only": "#8a8a8a",
    "GeneSPT": "#c7352f",
    "SpaIM": "#3b8b8c",
    "Tangram": "#4f8f46",
    "TransPA": "#7a5ca8",
    "SpaGE": "#607d9a",
    "stPlus": "#c9852c",
    "Full-ST upper": "#222222",
}


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def fmt_rank(row: pd.Series, df: pd.DataFrame, methods: list[str]) -> str:
    parts = []
    for metric in ["ARI", "NMI", "AMI"]:
        col = f"{metric}_mean"
        sub = df[df["method"].isin(methods)].set_index("method")
        if row["method"] in sub.index:
            ranks = sub[col].rank(ascending=False, method="min")
            parts.append(f"{metric} {int(ranks.loc[row['method']])}/{len(sub)}")
        else:
            parts.append(f"{metric} NA")
    return "; ".join(parts)


def interpretation_seqfish(method: str, row: pd.Series) -> str:
    if method == "GeneSPT":
        return (
            "Not first for annotation-based Leiden clustering; strongest among imputation "
            "methods for group-mean MAE and marker-overlap recovery, with group-effect "
            "Spearman essentially tied with SpaGE."
        )
    if method == "Observed-only":
        return "Reference using measured train+validation genes only; not an imputation method."
    if method == "Full-ST upper":
        return "Upper-bound reference using true held-out test genes; not an imputation method."
    return "External imputation baseline evaluated with the same frozen splits and downstream pipeline."


def interpretation_mvc(label_source: str, method: str, row: pd.Series, rank_imp: str) -> str:
    if method == "Observed-only":
        return "Reference using measured train+validation genes only; not an imputation method."
    if method == "Full-ST upper":
        return "Upper-bound reference using true held-out test genes; not an imputation method."
    if method == "GeneSPT" and label_source == "weak full-ST Leiden reference clusters":
        return (
            "Ranks first among imputation methods for this weak topology-consistency label; "
            "this is not curated biological validation."
        )
    if method == "GeneSPT":
        return (
            "Does not rank first under partial author/community annotation; interpret as "
            "exploratory because labels cover only matched cells."
        )
    return "External imputation baseline evaluated under the same pipeline."


def build_seqfish_summary() -> pd.DataFrame:
    cluster = pd.read_csv(UP / "downstream_upgraded_leiden_default_summary.csv")
    cluster = cluster[cluster["task"].eq("seqFISH_cell_types")].copy()
    diff = pd.read_csv(RESCUE / "celltype_differential_signal_recovery.csv")
    diff = (
        diff[diff["dataset"].eq("seqFISH+ cortex/SVZ")]
        .groupby("method", as_index=False)
        .agg(
            group_effect_spearman=("group_effect_spearman", "mean"),
            group_mean_MAE=("group_mean_MAE", "mean"),
            top20_marker_overlap=("top20_marker_overlap", "mean"),
            top50_marker_overlap=("top50_marker_overlap", "mean"),
            top20_marker_jaccard=("top20_marker_jaccard", "mean"),
            top50_marker_jaccard=("top50_marker_jaccard", "mean"),
        )
    )
    rows = []
    for _, row in cluster.iterrows():
        method = row["method"]
        d = diff[diff["method"].eq(method)]
        drow = d.iloc[0].to_dict() if not d.empty else {}
        out = {
            "method": method,
            "ARI": row["ARI_mean"],
            "NMI": row["NMI_mean"],
            "AMI": row["AMI_mean"],
            "rank_among_imputation_methods": fmt_rank(row, cluster, IMPUTATION_METHODS),
            "rank_including_observed_upper": fmt_rank(row, cluster, METHOD_ORDER),
            "group_effect_spearman": drow.get("group_effect_spearman", np.nan),
            "group_mean_MAE": drow.get("group_mean_MAE", np.nan),
            "top20_marker_overlap": drow.get("top20_marker_overlap", np.nan),
            "top50_marker_overlap": drow.get("top50_marker_overlap", np.nan),
            "top20_marker_jaccard": drow.get("top20_marker_jaccard", np.nan),
            "top50_marker_jaccard": drow.get("top50_marker_jaccard", np.nan),
            "interpretation": interpretation_seqfish(method, row),
        }
        rows.append(out)
    df = pd.DataFrame(rows)
    return df.sort_values("method", key=lambda s: s.map({m: i for i, m in enumerate(METHOD_ORDER)}))


def weak_mvc_rows() -> pd.DataFrame:
    fair = pd.read_csv(GSA / "leiden_pipeline_fairness_summary.csv")
    sub = fair[(fair["dataset"].eq("MVC/STARmap")) & (fair["pipeline"].isin(["all_genes", "cluster_count_matched"])) & (fair["hvg_n"].isna())].copy()
    sub["label_source"] = "weak full-ST Leiden reference clusters"
    sub["label_quality"] = "weak_full_ST_reference_cluster"
    sub["n_labeled_cells"] = 1549
    sub["pipeline"] = sub["pipeline"].map(
        {
            "all_genes": "Leiden default, all genes",
            "cluster_count_matched": "Cluster-count matched Leiden",
        }
    )
    return sub


def partial_mvc_rows() -> pd.DataFrame:
    default = pd.read_csv(UP / "downstream_upgraded_leiden_default_summary.csv")
    default = default[default["task"].isin(["MVC_Annotation_partial", "MVC_broad_label_partial"])].copy()
    default["pipeline"] = "Leiden default, partial annotation"
    matched = pd.read_csv(UP / "downstream_upgraded_leiden_cluster_count_matched_summary.csv")
    matched = matched[matched["task"].isin(["MVC_Annotation_partial", "MVC_broad_label_partial"])].copy()
    matched["pipeline"] = "Cluster-count matched Leiden, partial annotation"
    combo = pd.concat([default, matched], ignore_index=True)
    combo["label_source"] = combo["task"].map(
        {
            "MVC_Annotation_partial": "partial STARmap Annotation",
            "MVC_broad_label_partial": "partial STARmap broad label",
        }
    )
    combo["n_labeled_cells"] = combo["n_units"].astype(int)
    return combo


def build_mvc_summary() -> pd.DataFrame:
    raw = pd.concat([weak_mvc_rows(), partial_mvc_rows()], ignore_index=True, sort=False)
    rows = []
    for (label_source, pipeline), sub in raw.groupby(["label_source", "pipeline"], sort=False):
        sub = sub.copy()
        for _, row in sub.iterrows():
            out = {
                "label_source": label_source,
                "label_quality": row["label_quality"],
                "n_labeled_cells": int(row["n_labeled_cells"]),
                "pipeline": pipeline,
                "method": row["method"],
                "ARI": row["ARI_mean"],
                "NMI": row["NMI_mean"],
                "AMI": row["AMI_mean"],
                "rank_among_imputation_methods": fmt_rank(row, sub, IMPUTATION_METHODS),
                "rank_including_observed_upper": fmt_rank(row, sub, METHOD_ORDER),
                "interpretation": interpretation_mvc(label_source, row["method"], row, ""),
            }
            rows.append(out)
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["label_source", "pipeline", "method"],
        key=lambda s: s.map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(s),
    )


def combined_table(seq: pd.DataFrame, mvc: pd.DataFrame) -> pd.DataFrame:
    seq2 = seq.copy()
    seq2.insert(0, "panel", "A")
    seq2.insert(1, "dataset", "seqFISH+ cortex/SVZ")
    seq2.insert(2, "label_source", "curated cell_types")
    seq2.insert(3, "label_quality", "curated_cell_type")
    seq2.insert(4, "n_labeled_cells", 913)
    seq2.insert(5, "pipeline", "Leiden default, curated annotation")
    mvc2 = mvc.copy()
    mvc2.insert(0, "panel", "B")
    mvc2.insert(1, "dataset", "MVC/STARmap")
    for col in ["group_effect_spearman", "group_mean_MAE", "top20_marker_overlap", "top50_marker_overlap", "top20_marker_jaccard", "top50_marker_jaccard"]:
        if col not in mvc2.columns:
            mvc2[col] = np.nan
    cols = [
        "panel",
        "dataset",
        "label_source",
        "label_quality",
        "n_labeled_cells",
        "pipeline",
        "method",
        "ARI",
        "NMI",
        "AMI",
        "rank_among_imputation_methods",
        "rank_including_observed_upper",
        "group_effect_spearman",
        "group_mean_MAE",
        "top20_marker_overlap",
        "top50_marker_overlap",
        "top20_marker_jaccard",
        "top50_marker_jaccard",
        "interpretation",
    ]
    return pd.concat([seq2[cols], mvc2[cols]], ignore_index=True)


def grouped_bars(ax, df: pd.DataFrame, title: str, methods: list[str] = METHOD_ORDER, metrics: list[str] = ["ARI", "NMI", "AMI"]):
    plot = df[df["method"].isin(methods)].copy()
    plot["method"] = pd.Categorical(plot["method"], categories=methods, ordered=True)
    plot = plot.sort_values("method")
    x = np.arange(len(metrics))
    width = 0.095 if len(methods) > 6 else 0.12
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    for i, method in enumerate(methods):
        sub = plot[plot["method"].eq(method)]
        if sub.empty:
            continue
        vals = [float(sub.iloc[0][m]) for m in metrics]
        ax.bar(x + offsets[i], vals, width=width, color=COLORS.get(method, "#777"), label=method, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)
    ax.grid(axis="x", visible=False)


def single_metric_bars(ax, df: pd.DataFrame, value_col: str, title: str, ylabel: str, lower_better: bool = False):
    methods = IMPUTATION_METHODS
    plot = df[df["method"].isin(methods)].copy()
    plot["method"] = pd.Categorical(plot["method"], categories=methods, ordered=True)
    plot = plot.sort_values("method")
    vals = plot[value_col].to_numpy(dtype=float)
    ax.bar(np.arange(len(plot)), vals, color=[COLORS[m] for m in plot["method"]], edgecolor="white", linewidth=0.5)
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(plot["method"], rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel + (" (lower is better)" if lower_better else ""))
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)
    ax.grid(axis="x", visible=False)


def make_figures(seq: pd.DataFrame, mvc: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    grouped_bars(axes[0, 0], seq, "A1. seqFISH+ curated labels")
    single_metric_bars(axes[0, 1], seq, "group_mean_MAE", "A2. seqFISH+ group mean MAE", "MAE", lower_better=True)
    single_metric_bars(axes[0, 2], seq, "top50_marker_overlap", "A3. seqFISH+ top-50 marker overlap", "overlap")
    weak = mvc[(mvc["label_source"].eq("weak full-ST Leiden reference clusters")) & (mvc["pipeline"].eq("Leiden default, all genes"))]
    grouped_bars(axes[1, 0], weak, "B1. MVC weak topology reference")
    ann = mvc[(mvc["label_source"].eq("partial STARmap Annotation")) & (mvc["pipeline"].eq("Leiden default, partial annotation"))]
    grouped_bars(axes[1, 1], ann, "B2. MVC partial Annotation")
    broad = mvc[(mvc["label_source"].eq("partial STARmap broad label")) & (mvc["pipeline"].eq("Leiden default, partial annotation"))]
    grouped_bars(axes[1, 2], broad, "B3. MVC partial broad label")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Exploratory downstream sensitivity analyses", y=1.065, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Observed-only and Full-ST upper are references and are not counted as imputation methods.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(OUT / "supp_figure_downstream_validation_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "supp_figure_downstream_validation_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    grouped_bars(axes[0], seq, "seqFISH+ clustering")
    single_metric_bars(axes[1], seq, "group_mean_MAE", "Group mean MAE", "MAE", lower_better=True)
    single_metric_bars(axes[2], seq, "top50_marker_overlap", "Top-50 marker overlap", "overlap")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "panelA_seqfish_downstream_annotation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "panelA_seqfish_downstream_annotation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    grouped_bars(axes[0], weak, "Weak topology reference")
    grouped_bars(axes[1], ann, "Partial Annotation")
    grouped_bars(axes[2], broad, "Partial broad label")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "panelB_mvc_topology_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "panelB_mvc_topology_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_docs(seq: pd.DataFrame, mvc: pd.DataFrame, supp: pd.DataFrame) -> None:
    (OUT / "downstream_pipeline_standardization.md").write_text(
        "\n".join(
            [
                "# Downstream Pipeline Standardization",
                "",
                "## Augmented Matrix Construction",
                "",
                "- Method-specific augmented matrix: train+validation true ST genes plus method-predicted held-out test genes.",
                "- Observed-only reference: train+validation true ST genes only.",
                "- Full-ST upper reference: train+validation+test true ST genes. This is an upper-bound reference, not an imputation method.",
                "",
                "## Preprocessing",
                "",
                "- Existing evaluator expression scale was used from saved matrices.",
                "- Genes/features were z-scored within each downstream matrix.",
                "- Zero-variance and non-finite features were removed.",
                "- PCA was applied with a fixed random seed before clustering.",
                "",
                "## Clustering",
                "",
                "- Default Leiden used `n_neighbors=15` and `resolution=1.0` where available.",
                "- Sensitivity analyses used a fixed Leiden resolution grid and selected cluster-count matched settings by closeness to the reference label class count, not by ARI/NMI/AMI.",
                "- KMeans summaries from earlier audits are retained as sensitivity outputs but are not used for a main-text claim.",
                "",
                "## Metrics and Ranking",
                "",
                "- Metrics: ARI, NMI, AMI, homogeneity and completeness.",
                "- Ranks are reported separately among imputation methods and including Observed-only / Full-ST upper references.",
                "- Observed-only and Full-ST upper are never counted as imputation methods.",
                "",
                "## UMAP",
                "",
                "- Any UMAP views are qualitative only, use a fixed seed and should follow a pre-specified representative-fold rule.",
                "- UMAP was not used for model selection or parameter tuning.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (OUT / "seqfish_downstream_annotation_summary.md").write_text(
        "# seqFISH+ Annotation-Based Downstream Summary\n\n"
        "Dataset: seqFISH+ cortex/SVZ. Label: curated `cell_types`, 913/913 matched cells.\n\n"
        + md_table(seq)
        + "\n\nInterpretation: GeneSPT is not first for Leiden clustering, but it shows the lowest group-mean MAE and strongest marker-overlap recovery among imputation methods in the differential-signal audit.\n",
        encoding="utf-8",
    )
    (OUT / "mvc_downstream_topology_sensitivity_summary.md").write_text(
        "# MVC/STARmap Downstream Topology Sensitivity Summary\n\n"
        "This table separates weak full-ST reference clustering from partial author/community annotation. The weak full-ST label is a topology-consistency audit, not curated biological validation.\n\n"
        + md_table(mvc)
        + "\n",
        encoding="utf-8",
    )
    (OUT / "supp_table_downstream_validation_sensitivity.md").write_text(
        "# Supplementary Table. Downstream Validation Sensitivity\n\n" + md_table(supp) + "\n",
        encoding="utf-8",
    )
    (OUT / "supp_figure_downstream_validation_sensitivity_caption.md").write_text(
        "Supplementary Figure Sx. Exploratory downstream sensitivity analyses. "
        "A, Annotation-based downstream evaluation in seqFISH+ cortex/SVZ using curated cell-type labels. "
        "B, Topology-consistency sensitivity analysis in MVC/STARmap using available reference clustering or partial author/community annotations. "
        "All methods were evaluated using the same augmented-matrix construction and Leiden clustering pipeline. "
        "Observed-only and Full-ST upper are shown as references but are not counted as imputation methods. "
        "These analyses are exploratory and are not used to claim universal downstream clustering improvement.\n",
        encoding="utf-8",
    )
    (OUT / "downstream_validation_wording_recommendation.md").write_text(
        "\n".join(
            [
                "# Downstream Validation Wording Recommendation",
                "",
                "## Safe wording",
                "",
                "\"Exploratory downstream sensitivity analyses using standardized Leiden pipelines and available annotations are provided in Supplementary Figure/Table Sx. These analyses suggest that GeneSPT can preserve selected cell-type differential signals in seqFISH+ cortex/SVZ and remains competitive in selected topology-consistency settings, but they do not support a universal downstream clustering claim.\"",
                "",
                "## Unsafe wording",
                "",
                "- GeneSPT improves downstream clustering.",
                "- GeneSPT improves all downstream analyses.",
                "- GeneSPT achieves universal downstream superiority.",
                "- GeneSPT validates biological discovery.",
                "",
                "## Placement",
                "",
                "Use this only as a supplementary sensitivity/audit statement. Do not add a main Figure 6.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "downstream_validation_supplement_changelog.md").write_text(
        "\n".join(
            [
                "# Downstream Validation Supplement Changelog",
                "",
                "- Compiled existing label provenance, upgraded-label clustering, Leiden sensitivity and differential-signal audit outputs.",
                "- No GeneSPT model was rerun.",
                "- No prediction matrices were modified.",
                "- No manuscript files were modified.",
                "- No new main Figure 6 was generated.",
                "- No SSIMx10 metric was used.",
                "- No test label tuning was performed.",
                "- No test-gene ground truth was used for selection.",
                "- Observed-only and Full-ST upper were handled as references.",
                "- Imputation-method ranks are reported separately from ranks that include Observed-only / Full-ST upper.",
                "",
                "## Input files",
                "",
                f"- {UP / 'downstream_upgraded_leiden_default_summary.csv'}",
                f"- {UP / 'downstream_upgraded_leiden_cluster_count_matched_summary.csv'}",
                f"- {UP / 'downstream_upgraded_kmeans_summary.csv'}",
                f"- {UP / 'downstream_upgraded_label_task_audit.csv'}",
                f"- {LEIDEN / 'leiden_default_summary.csv'}",
                f"- {GSA / 'leiden_pipeline_fairness_summary.csv'}",
                f"- {RESCUE / 'celltype_differential_signal_recovery.csv'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    seq = build_seqfish_summary()
    mvc = build_mvc_summary()
    supp = combined_table(seq, mvc)

    seq.to_csv(OUT / "seqfish_downstream_annotation_summary.csv", index=False)
    mvc.to_csv(OUT / "mvc_downstream_topology_sensitivity_summary.csv", index=False)
    supp.to_csv(OUT / "supp_table_downstream_validation_sensitivity.csv", index=False)

    make_figures(seq, mvc)
    write_docs(seq, mvc, supp)
    print(f"Wrote downstream validation supplement to {OUT}")


if __name__ == "__main__":
    main()
