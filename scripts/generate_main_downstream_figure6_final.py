#!/usr/bin/env python3
"""Generate the final manuscript-style Figure 6 from existing downstream sources.

This script only redraws the figure from existing audit/source files. It does
not rerun GeneSPT, does not modify prediction matrices, and does not tune any
downstream parameters.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score


ROOT = Path("/workspace/GeneSPT")
IN_REVISED = ROOT / "final_output" / "main_downstream_figure6_revised"
IN_ORIG = ROOT / "final_output" / "main_downstream_figure6"
IN_SUPP = ROOT / "final_output" / "downstream_validation_supplement"
OUT = ROOT / "final_output" / "main_downstream_figure6_final"
OUT.mkdir(parents=True, exist_ok=True)


IMPUTATION_METHODS = ["GeneSPT", "SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus"]
METHOD_ORDER = ["Observed-only", "GeneSPT", "SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus", "Full-ST upper"]
METHOD_LABELS = {
    "Observed-only": "Observed-only",
    "GeneSPT": "GeneSPT",
    "SpaIM": "SpaIM",
    "Tangram": "Tangram",
    "TransPA": "TransImp",
    "SpaGE": "SpaGE",
    "stPlus": "stPlus",
    "Full-ST upper": "Full-ST upper",
}
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
GRID = "#e8e8e8"


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.055, 1.04, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")


def draw_workflow(ax: plt.Axes) -> None:
    ax.axis("off")
    panel_label(ax, "A")
    ax.set_title("Downstream workflow", fontsize=9.2, fontweight="bold", pad=5)

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, fontsize: float = 7.0) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0.9,
            edgecolor="#b7b7b7",
            facecolor=fc,
            transform=ax.transAxes,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, transform=ax.transAxes, zorder=3)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#777777",
                transform=ax.transAxes,
                shrinkA=2,
                shrinkB=3,
                zorder=4,
            )
        )

    box(0.03, 0.70, 0.33, 0.12, "Measured\ntrain/val genes", "#f1f1f1")
    box(0.03, 0.46, 0.33, 0.12, "Predicted\nheld-out genes", "#ffe9e7")
    box(0.44, 0.58, 0.25, 0.13, "Augmented\nmatrix", "#fff7f6")
    box(0.735, 0.70, 0.235, 0.11, "PCA → kNN\n→ graph clustering", "#f8f8f8", fontsize=6.2)
    box(0.79, 0.50, 0.18, 0.10, "ARI/NMI\nAMI/Homogeneity", "#f8f8f8", fontsize=5.7)
    box(0.79, 0.28, 0.18, 0.11, "Differential\nsignal", "#f8f8f8")
    arrow(0.365, 0.76, 0.432, 0.665)
    arrow(0.365, 0.52, 0.432, 0.625)
    arrow(0.695, 0.655, 0.73, 0.755)
    arrow(0.855, 0.695, 0.855, 0.605)
    arrow(0.695, 0.61, 0.785, 0.34)


def draw_panel_b(fig: plt.Figure, spec, src: pd.DataFrame) -> None:
    outer = fig.add_subplot(spec)
    outer.axis("off")
    panel_label(outer, "B")
    inner = spec.subgridspec(3, 2, height_ratios=[0.16, 1.0, 1.0], wspace=0.46, hspace=0.64)
    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.45, "seqFISH+ differential signal", fontsize=9.2, fontweight="bold", ha="left", va="center")

    metric_specs = [
        ("group-effect Spearman", "group-effect\nSpearman ↑"),
        ("group-mean MAE", "group-mean\nMAE ↓"),
        ("top20 marker overlap", "top-20 marker-effect\noverlap count ↑"),
        ("top50 marker overlap", "top-50 marker-effect\noverlap count ↑"),
    ]
    y = np.arange(len(IMPUTATION_METHODS))
    for i, (metric, title) in enumerate(metric_specs):
        ax = fig.add_subplot(inner[1 + i // 2, i % 2])
        sub = src[src["metric"].eq(metric)]
        for yi, method in enumerate(IMPUTATION_METHODS):
            row = sub[sub["method"].eq(method)]
            if row.empty:
                continue
            val = float(row.iloc[0]["value"])
            if method == "GeneSPT":
                ax.scatter(
                    val,
                    yi,
                    s=46,
                    marker="o",
                    color=COLORS[method],
                    edgecolor="#111111",
                    linewidth=0.35,
                    zorder=5,
                )
            else:
                ax.scatter(val, yi, s=27, color=COLORS[method], zorder=3)
        if not sub.empty:
            direction = str(sub.iloc[0].get("metric_direction", "higher"))
            ranked = sub.dropna(subset=["value"]).copy()
            if direction == "lower":
                ranked = ranked.sort_values("value", ascending=True)
            else:
                ranked = ranked.sort_values("value", ascending=False)
            best = str(ranked.iloc[0]["method"]) if not ranked.empty else ""
            brow = sub[sub["method"].eq(best)]
            if not brow.empty:
                ax.scatter(
                    float(brow.iloc[0]["value"]),
                    IMPUTATION_METHODS.index(best),
                    s=78,
                    facecolors="none",
                    edgecolors="#111111",
                    linewidths=1.2,
                    zorder=4,
                )
        ax.set_title(title, fontsize=6.8, pad=2)
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABELS[m] for m in IMPUTATION_METHODS] if i % 2 == 0 else [])
        ax.invert_yaxis()
        ax.set_ylim(len(IMPUTATION_METHODS) - 0.45, -0.65)
        ax.grid(axis="x", color=GRID, linewidth=0.65)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="both", labelsize=6.2)
        clean_axes(ax)


def load_panel_c_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_path = IN_REVISED / "panelC_seqfish_fixed_fold0_leiden_source.csv"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Missing Panel C cluster source: {cluster_path}")
    clusters = pd.read_csv(cluster_path)

    metrics_path = ROOT / "final_output" / "downstream_leiden_sensitivity" / "leiden_clustering_metrics_long.csv"
    metrics_rows = []
    if metrics_path.exists():
        met = pd.read_csv(metrics_path)
        met = met[
            met["dataset"].eq("seqFISH+ cortex/SVZ")
            & met["fold"].eq(0)
            & met["method"].isin(["GeneSPT", "SpaGE"])
            & met["n_neighbors"].eq(15)
            & met["resolution"].eq(1.0)
            & met["seed"].eq(0)
        ]
        for _, row in met.iterrows():
            metrics_rows.append(
                {
                    "method": row["method"],
                    "ARI": float(row["ARI"]),
                    "NMI": float(row["NMI"]),
                    "AMI": float(row["AMI"]),
                    "metric_source": str(metrics_path),
                }
            )

    if len(metrics_rows) < 2:
        for method in ["GeneSPT", "SpaGE"]:
            sub = clusters[clusters["method"].eq(method)].copy()
            labels = sub["reference_label"].astype(str)
            pred = sub["leiden_cluster"].astype(int)
            metrics_rows.append(
                {
                    "method": method,
                    "ARI": adjusted_rand_score(labels, pred),
                    "NMI": normalized_mutual_info_score(labels, pred),
                    "AMI": adjusted_mutual_info_score(labels, pred),
                    "metric_source": "derived_from_panelC_seqfish_fixed_fold0_leiden_source.csv",
                }
            )
    metrics = pd.DataFrame(metrics_rows).drop_duplicates("method", keep="first")

    ref = clusters[clusters["method"].eq("Curated cell types")].copy()
    cell_order = ref["reference_label"].astype(str).value_counts().index.tolist()
    heat_rows = []
    for method in ["GeneSPT", "SpaGE"]:
        sub = clusters[clusters["method"].eq(method)].copy()
        sub["reference_label"] = pd.Categorical(sub["reference_label"].astype(str), categories=cell_order, ordered=True)
        sub["leiden_cluster"] = sub["leiden_cluster"].astype(int)
        counts = sub.groupby(["reference_label", "leiden_cluster"], observed=False).size().reset_index(name="n_cells")
        totals = counts.groupby("leiden_cluster", observed=False)["n_cells"].transform("sum")
        counts["cluster_total_cells"] = totals
        counts["fraction_cluster_composition"] = np.where(totals > 0, counts["n_cells"] / totals, 0.0)
        dominant = (
            counts.sort_values(["leiden_cluster", "fraction_cluster_composition"], ascending=[True, False])
            .groupby("leiden_cluster", observed=False)
            .first()
            .reset_index()[["leiden_cluster", "reference_label", "fraction_cluster_composition"]]
            .rename(columns={"reference_label": "dominant_cell_type", "fraction_cluster_composition": "dominant_fraction"})
        )
        dominant["_cell_order"] = dominant["dominant_cell_type"].astype(str).map({c: i for i, c in enumerate(cell_order)})
        dominant = dominant.sort_values(["_cell_order", "dominant_fraction", "leiden_cluster"], ascending=[True, False, True])
        cluster_order = dominant["leiden_cluster"].tolist()
        cluster_rank = {c: i for i, c in enumerate(cluster_order)}
        counts = counts.merge(dominant[["leiden_cluster", "dominant_cell_type", "dominant_fraction"]], on="leiden_cluster", how="left")
        counts["cluster_sort_index"] = counts["leiden_cluster"].map(cluster_rank)
        counts["method"] = method
        counts["dataset"] = "seqFISH+ cortex/SVZ"
        counts["fold"] = 0
        counts["normalization"] = "columns normalized to 1: fraction of each Leiden cluster belonging to each cell type"
        heat_rows.append(counts)
    heat = pd.concat(heat_rows, ignore_index=True)
    heat["reference_label"] = heat["reference_label"].astype(str)
    heat.to_csv(OUT / "panelC_seqfish_cluster_composition_source.csv", index=False)

    changelog = "\n".join(
        [
            "# Panel C Cluster Composition Changelog",
            "",
            "- Replaced the prior spatial cluster scatter view with a cluster-to-cell-type composition heatmap.",
            "- Used fixed fold0 Leiden clusters from the existing revised Panel C source.",
            "- Heatmap columns are Leiden clusters; rows are curated seqFISH+ cell types.",
            "- Values are column-normalized fractions of cells in each cluster belonging to each cell type.",
            "- Cluster columns are sorted by dominant curated cell type and dominant fraction.",
            "- No model rerun, no prediction matrix modification, and no test-label parameter tuning were performed.",
        ]
    )
    (OUT / "panelC_seqfish_cluster_composition_changelog.md").write_text(changelog + "\n", encoding="utf-8")
    return heat, metrics


def draw_panel_c(fig: plt.Figure, spec, heat: pd.DataFrame, metrics: pd.DataFrame) -> None:
    outer = fig.add_subplot(spec)
    outer.axis("off")
    panel_label(outer, "C")
    inner = spec.subgridspec(2, 3, height_ratios=[0.16, 1.0], width_ratios=[1.0, 1.0, 0.045], wspace=0.12, hspace=0.08)
    title_ax = fig.add_subplot(inner[0, 0:2])
    title_ax.axis("off")
    title_ax.text(0.0, 0.62, "seqFISH+ cluster composition", fontsize=9.2, fontweight="bold", ha="left", va="center")

    methods = [("GeneSPT", "GeneSPT Leiden clusters"), ("SpaGE", "SpaGE Leiden clusters")]
    cell_order = (
        heat[heat["method"].eq("GeneSPT")]
        .groupby("reference_label", observed=False)["n_cells"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    vmax = max(float(heat["fraction_cluster_composition"].max()), 0.01)
    images = []
    for i, (method, title) in enumerate(methods):
        ax = fig.add_subplot(inner[1, i])
        sub = heat[heat["method"].eq(method)].copy()
        cluster_order = (
            sub[["leiden_cluster", "cluster_sort_index"]]
            .drop_duplicates()
            .sort_values("cluster_sort_index")["leiden_cluster"]
            .tolist()
        )
        mat = (
            sub.pivot_table(index="reference_label", columns="leiden_cluster", values="fraction_cluster_composition", fill_value=0.0, observed=False)
            .reindex(index=cell_order, columns=cluster_order, fill_value=0.0)
        )
        im = ax.imshow(mat.values, aspect="auto", interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=vmax)
        images.append(im)
        mrow = metrics[metrics["method"].eq(method)]
        metric_text = ""
        if not mrow.empty:
            r = mrow.iloc[0]
            metric_text = f"ARI={r['ARI']:.2f}, NMI={r['NMI']:.2f}, AMI={r['AMI']:.2f}"
        ax.set_title(f"{title}\n{metric_text}", fontsize=7.4, pad=3)
        ax.set_xticks([])
        if i == 0:
            ax.set_yticks(np.arange(len(cell_order)))
            ax.set_yticklabels(cell_order, fontsize=6.1)
        else:
            ax.set_yticks(np.arange(len(cell_order)))
            ax.set_yticklabels([])
        ax.set_xlabel("Leiden clusters", fontsize=6.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_subplot(inner[1, 2])
    cbar = fig.colorbar(images[0], cax=cax)
    cbar.set_label("Cluster composition", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.8)


def draw_panel_d(fig: plt.Figure, spec, src: pd.DataFrame) -> None:
    outer = fig.add_subplot(spec)
    outer.axis("off")
    panel_label(outer, "D")
    inner = spec.subgridspec(2, 2, height_ratios=[0.16, 1.0], wspace=0.25, hspace=0.04)
    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.45, "Held-out cell-type effect recovery", fontsize=9.2, fontweight="bold", ha="left", va="center")
    plotted = src[src["plotted"].astype(bool)].copy()
    lim_vals = pd.concat([src["true_cell_type_effect"], src["predicted_cell_type_effect"]])
    lo, hi = np.nanpercentile(lim_vals, [1, 99])
    pad = 0.15 * (hi - lo)
    lo, hi = float(lo - pad), float(hi + pad)
    methods = ["GeneSPT"] + [m for m in src["method"].dropna().unique().tolist() if m != "GeneSPT"][:1]
    for i, method in enumerate(methods):
        ax = fig.add_subplot(inner[1, i])
        allm = src[src["method"].eq(method)]
        sub = plotted[plotted["method"].eq(method)]
        color = COLORS["GeneSPT"] if method == "GeneSPT" else COLORS.get(method, COLORS["SpaGE"])
        ax.scatter(sub["true_cell_type_effect"], sub["predicted_cell_type_effect"], s=4.4, alpha=0.15, color=color, linewidth=0)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#888888", linewidth=0.9)
        rho = spearmanr(allm["true_cell_type_effect"], allm["predicted_cell_type_effect"], nan_policy="omit").statistic
        mae = np.mean(np.abs(allm["true_cell_type_effect"] - allm["predicted_cell_type_effect"]))
        ax.set_title(method, fontsize=7.8, pad=3)
        ax.text(0.03, 0.95, f"rho={rho:.2f}\nMAE={mae:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color="#444444")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True effect")
        if i == 0:
            ax.set_ylabel("Predicted effect")
        else:
            ax.set_yticklabels([])
        ax.grid(color=GRID, linewidth=0.65)
        clean_axes(ax)


def draw_panel_e(ax: plt.Axes, src: pd.DataFrame) -> None:
    panel_label(ax, "E")
    metrics = [("ARI", "ARI"), ("AMI", "AMI"), ("homogeneity", "Homogeneity"), ("NMI", "NMI")]
    methods = [m for m in IMPUTATION_METHODS if m in set(src["method"].dropna())]
    dataset = "MVC/STARmap"
    if "dataset" in src.columns and not src["dataset"].dropna().empty:
        dataset = str(src["dataset"].dropna().iloc[0])
    title = "MVC topology-consistency" if "MVC" in dataset else "MHPR/MERFISH clustering"
    ax.set_title(title, fontsize=9.2, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if not methods:
        ax.text(0.5, 0.5, "No clustering source", ha="center", va="center", fontsize=7.0)
        return

    plot_ax = ax.inset_axes([0.08, 0.10, 0.90, 0.58])
    metric_cols = [m[0] for m in metrics]
    metric_labels = [m[1] for m in metrics]
    x = np.arange(len(metrics))
    row_by_method = {m: src[src["method"].eq(m)].iloc[0] for m in methods if not src[src["method"].eq(m)].empty}
    width = 0.12
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    all_vals: list[float] = []
    for i, method in enumerate(methods):
        vals = [float(row_by_method[method][metric]) for metric in metric_cols]
        all_vals.extend(vals)
        plot_ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.86,
            color=COLORS[method],
            edgecolor="#333333" if method == "GeneSPT" else "none",
            linewidth=0.45 if method == "GeneSPT" else 0.0,
            alpha=0.96,
            zorder=3,
        )

    ymin = 0.0
    ymax = max(0.65, max(all_vals) + 0.015)
    plot_ax.set_ylim(ymin, ymax)
    plot_ax.set_xlim(-0.55, len(metrics) - 0.45)
    plot_ax.set_xticks(x)
    plot_ax.set_xticklabels(metric_labels)
    plot_ax.set_ylabel("Clustering score")
    if "pipeline" in src.columns and not src["pipeline"].dropna().empty:
        pipeline_note = str(src["pipeline"].dropna().iloc[0]).replace(", ", "\n", 1)
        plot_ax.text(0.02, 0.92, pipeline_note, transform=plot_ax.transAxes, fontsize=6.0, color="#555555", va="top")
    plot_ax.grid(axis="y", color=GRID, linewidth=0.65)
    plot_ax.grid(axis="x", visible=False)
    clean_axes(plot_ax)


def write_text_outputs() -> None:
    caption = (
        "Figure 6. Representative downstream analysis using predicted genes. "
        "A, Standardized downstream workflow for constructing augmented expression matrices from measured train/validation genes and method-predicted held-out genes. "
        "B, Cell-type differential signal recovery in seqFISH+ cortex/SVZ using curated cell-type annotations; GeneSPT is highlighted, and the best method for each metric is circled. Marker-effect overlap is reported as overlap count. "
        "C, Cluster-to-cell-type composition of Leiden clusters in seqFISH+ cortex/SVZ for GeneSPT and SpaGE, the strongest external comparator shown for this panel. Heatmap columns represent Leiden clusters and rows represent curated cell types; values are column-normalized cluster compositions. "
        "D, Recovery of held-out cell-type expression effects in seqFISH+ cortex/SVZ, comparing predicted effects with true effects. "
        "E, MVC/STARmap topology-consistency analysis using an ST-derived reference. Observed-only and Full-ST upper are shown as references and are not counted as imputation methods. "
        "Together, these analyses evaluate representative downstream signals preserved by predicted genes; they are not used to claim universal downstream clustering improvement.\n"
    )
    (OUT / "figure6_downstream_analysis_using_predicted_genes_final_caption.md").write_text(caption, encoding="utf-8")

    cn = (
        "为进一步评估预测基因是否保留下游相关信号，我们进行了代表性下游分析（Figure 6）。"
        "在具有完整细胞类型注释的 seqFISH+ cortex/SVZ 数据集中，我们首先评估预测表达对细胞类型差异信号的恢复能力。"
        "该分析关注留出基因在不同细胞类型之间的表达效应是否能够被预测矩阵保留。"
        "GeneSPT 在 group-mean error 和 marker-effect overlap 等指标上表现具有竞争力，并在 held-out cell-type effect recovery 中表现出较高的秩相关性，提示其预测结果能够保留部分细胞类型相关表达差异。"
        "进一步地，我们使用统一的 PCA–kNN–Leiden 流程评估预测矩阵对空间表达拓扑结构的影响，并在 MVC/STARmap 上展示 topology-consistency 分析。"
        "Observed-only 和 Full-ST upper 分别作为原始观测参考和上界参考，不计入插补方法排名。"
        "总体而言，这些分析支持 GeneSPT 在代表性场景下保留下游相关信号，但也显示下游聚类表现受到 annotation 来源和分析流程影响，因此本文不将其解释为普适性的下游聚类提升结论。\n"
    )
    (OUT / "figure6_main_text_paragraph_cn.md").write_text(cn, encoding="utf-8")

    en = (
        "To further assess whether predicted genes preserve downstream-relevant signals, we performed representative downstream analyses (Figure 6). "
        "In the seqFISH+ cortex/SVZ dataset with complete curated cell-type annotations, we evaluated recovery of cell-type differential signals for held-out genes. "
        "This analysis asks whether expression effects across cell types are retained in the predicted expression matrix rather than treating clustering as the sole endpoint. "
        "GeneSPT was competitive for group-mean error and marker-effect overlap and showed a relatively high rank correlation in held-out cell-type effect recovery, suggesting that its predictions preserve selected cell-type-associated expression differences. "
        "We also used a standardized PCA–kNN–Leiden workflow to evaluate topology-consistency in MVC/STARmap. "
        "Observed-only and Full-ST upper were treated as references and were not included in imputation-method rankings. "
        "Overall, these analyses support preservation of representative downstream-relevant signals by GeneSPT, while also showing that downstream clustering performance depends on annotation source and analysis pipeline; we therefore do not interpret these results as a universal downstream clustering improvement claim.\n"
    )
    (OUT / "figure6_main_text_paragraph_en.md").write_text(en, encoding="utf-8")

    changelog = "\n".join(
        [
            "# Figure 6 Final Changelog",
            "",
            "- Redrew Figure 6 from existing downstream source files.",
            "- No model rerun was performed.",
            "- No prediction matrices were modified.",
            "- No manuscript files were modified.",
            "- No new experiments were added.",
            "- No new metrics beyond existing downstream audit/source values were introduced for method evaluation.",
            "- No test label tuning was performed.",
            "- Held-out test-gene ground truth was not used for parameter selection.",
            "- SSIMx10 was not used.",
            "- Panel C was changed from spatial cluster scatter to a cluster-to-cell-type composition heatmap using fixed fold0 Leiden clusters.",
            "- Observed-only and Full-ST upper are shown as references and are not counted as imputation methods.",
            "- MVC panel is topology-consistency analysis using an ST-derived reference, not curated biological validation.",
        ]
    )
    (OUT / "figure6_downstream_analysis_using_predicted_genes_final_changelog.md").write_text(changelog + "\n", encoding="utf-8")


def main() -> None:
    panel_b = pd.read_csv(IN_ORIG / "panelB_seqfish_differential_signal_source.csv")
    panel_d = pd.read_csv(IN_ORIG / "panelD_seqfish_effect_scatter_source.csv")
    panel_e = pd.read_csv(IN_ORIG / "panelE_mvc_topology_consistency_source.csv")
    panel_c_heat, panel_c_metrics = load_panel_c_source()
    panel_b.to_csv(OUT / "panelB_seqfish_differential_signal_source.csv", index=False)
    panel_d.to_csv(OUT / "panelD_effect_recovery_source.csv", index=False)
    panel_e.to_csv(OUT / "panelE_mvc_topology_source.csv", index=False)

    combined = pd.concat(
        [
            panel_b.assign(source_panel="B"),
            panel_c_heat.assign(source_panel="C"),
            panel_c_metrics.assign(source_panel="C_metrics"),
            panel_d.assign(source_panel="D"),
            panel_e.assign(source_panel="E"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(OUT / "figure6_downstream_analysis_using_predicted_genes_final_source.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(17.2, 10.6), facecolor="white", constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.95, 1.12],
        width_ratios=[1.05, 1.18, 1.42],
        hspace=0.46,
        wspace=0.34,
        left=0.052,
        right=0.982,
        top=0.84,
        bottom=0.105,
    )

    draw_workflow(fig.add_subplot(gs[0, 0]))
    draw_panel_b(fig, gs[0, 1], panel_b)
    draw_panel_d(fig, gs[0, 2], panel_d)
    draw_panel_c(fig, gs[1, 0:2], panel_c_heat, panel_c_metrics)
    draw_panel_e(fig.add_subplot(gs[1, 2]), panel_e)

    method_handles = [Patch(color=COLORS[m], label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    symbol_handles = [
        Line2D([0], [0], marker="o", color="#111111", markerfacecolor="none", markersize=6.5, label="Best external"),
    ]
    fig.legend(handles=method_handles + symbol_handles, loc="upper center", bbox_to_anchor=(0.52, 0.915), ncol=5, frameon=False)
    fig.suptitle("Representative downstream analysis using predicted genes", fontsize=12.2, fontweight="bold", y=0.972)
    fig.text(
        0.5,
        0.035,
        "Observed-only and Full-ST upper are references and are not counted as imputation methods. Downstream results are representative and pipeline-dependent.",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )

    fig.savefig(OUT / "figure6_downstream_analysis_using_predicted_genes_final.pdf")
    fig.savefig(OUT / "figure6_downstream_analysis_using_predicted_genes_final.png", dpi=300)
    plt.close(fig)
    write_text_outputs()
    print(f"Wrote final Figure 6 to {OUT}")


if __name__ == "__main__":
    main()
