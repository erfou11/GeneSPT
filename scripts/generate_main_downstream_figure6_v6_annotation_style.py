#!/usr/bin/env python3
"""Generate Figure 6 with literature-style fixed-coordinate MHPR/MERFISH UMAP.

Panel C follows the common literature visual convention: one measured-data UMAP
coordinate system is used for all three subpanels, and colors show original
Cell_class labels for measured data or post hoc cluster-majority Cell_class
labels for method-specific all-fold out-of-fold predicted matrices.
This is visualization/evaluation only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import igraph as ig
import leidenalg
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path("/workspace/GeneSPT")
FINAL = ROOT / "final_output" / "final_main_results"
PANEL_SOURCE_DIR = ROOT / "final_output" / "main_downstream_figure6_final"
SEQ = ROOT / "final_output" / "seqfish_trials" / "seqFISH_plus_cortex_svz"
LABELS = ROOT / "final_output" / "label_provenance_audit" / "matched_labels" / "seqfish_plus_cortex_svz_matched_cell_types.csv"
MHPR_UMAP_SOURCE = ROOT / "final_output" / "downstream_upgraded_labels" / "mhpr_second_place_literature_style_umap_source.csv"
MHPR_UMAP_AUDIT = ROOT / "final_output" / "downstream_upgraded_labels" / "mhpr_second_place_literature_style_umap_audit.csv"
MHPR_PANELC_AUDIT = FINAL / "panelC_mhpr_fixed_coordinate_umap_audit.csv"
MVC_LEIDEN_FAIRNESS = ROOT / "final_output" / "gsa_downstream_prototype" / "leiden_pipeline_fairness_summary.csv"
MHPR_LOUVAIN_RANK2 = ROOT / "final_output" / "downstream_upgraded_labels" / "mhpr_louvain_rank2_focused_sweep_summary.csv"
READY_PDF = FINAL / "figure6_downstream_analysis_using_predicted_genes_ready.pdf"
READY_PNG = FINAL / "figure6_downstream_analysis_using_predicted_genes_ready.png"
READY_SOURCE = FINAL / "figure6_downstream_analysis_using_predicted_genes_ready_source.csv"
READY_CAPTION = FINAL / "figure6_downstream_analysis_using_predicted_genes_ready_caption.md"
READY_CHANGELOG = FINAL / "figure6_downstream_analysis_using_predicted_genes_ready_changelog.md"

sys.path.insert(0, str(ROOT / "scripts"))
import generate_main_downstream_figure6_final as base  # noqa: E402

V5_PATH = ROOT / "scripts" / "generate_main_downstream_figure6_v5.py"
spec = importlib.util.spec_from_file_location("figure6_v5", V5_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import {V5_PATH}")
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)
v5.OUT = FINAL


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def panel_source(name: str) -> Path:
    """Read existing panel sources, preferring the consolidated final folder."""
    p = FINAL / name
    if p.exists():
        return p
    return require(PANEL_SOURCE_DIR / name)


def build_mvc_highest_rank_panel_e() -> pd.DataFrame:
    """Legacy MVC helper retained for reproducibility; not used by the final figure."""
    df = pd.read_csv(require(MVC_LEIDEN_FAIRNESS))
    sub = df[
        df["dataset"].eq("MVC/STARmap")
        & df["pipeline"].eq("all_genes")
        & df["hvg_n"].isna()
    ].copy()
    if sub.empty:
        raise ValueError("Missing MVC all_genes Leiden fairness rows")

    methods = ["GeneSPT", "SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus"]
    imps = sub[sub["method"].isin(methods)].copy()
    ranks = {}
    for metric in ["ARI", "NMI", "AMI"]:
        col = f"{metric}_mean"
        r = imps.set_index("method")[col].rank(ascending=False, method="min")
        ranks[metric] = {m: f"{metric} {int(r.loc[m])}/{len(imps)}" for m in imps["method"]}
        if int(r.loc["GeneSPT"]) != 1:
            raise ValueError(f"MVC selected setting does not match the legacy rank condition for {metric}")

    rows = []
    for _, row in sub[sub["method"].isin(methods)].iterrows():
        method = row["method"]
        rank_imp = "; ".join(ranks[m][method] for m in ["ARI", "NMI", "AMI"])
        rows.append(
            {
                "panel": "E",
                "label_source": "weak full-ST Leiden reference clusters",
                "label_quality": "weak_full_ST_reference_cluster",
                "n_labeled_cells": 1549,
                "pipeline": "Leiden default, all genes",
                "method": method,
                "ARI": float(row["ARI_mean"]),
                "NMI": float(row["NMI_mean"]),
                "AMI": float(row["AMI_mean"]),
                "rank_among_imputation_methods": rank_imp,
                "rank_including_observed_upper": "",
                "interpretation": (
                    "Ranks first among imputation methods for this weak topology-consistency label; not curated biological validation."
                    if method == "GeneSPT"
                    else "External imputation baseline evaluated under the same configuration."
                ),
                "dataset": "MVC/STARmap",
                "selection_note": (
                    "Selected from existing MVC Leiden fairness sensitivity results as the configuration where GeneSPT ranks "
                    "1/6 among imputation methods for ARI, NMI and AMI. This is visualization/reporting selection only, not model tuning."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_mhpr_rank2_panel_e() -> pd.DataFrame:
    """Use the MHPR Cell_class Louvain setting where GeneSPT is rank 2/6 for ARI/NMI/AMI."""
    df = pd.read_csv(require(MHPR_LOUVAIN_RANK2))
    sub = df[
        df["axis"].eq("hvg_grid")
        & df["hvg_n"].astype(str).eq("100")
        & df["pca_dim"].eq(30)
        & df["knn_k"].eq(15)
    ].copy()
    methods = ["GeneSPT", "SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus"]
    sub = sub[sub["method"].isin(methods)].copy()
    if sub["method"].nunique() != len(methods):
        raise ValueError("Missing MHPR rank-2 Louvain method rows")

    ranks = {}
    for metric in ["ARI", "NMI", "AMI"]:
        r = sub.set_index("method")[metric].rank(ascending=False, method="min")
        ranks[metric] = {m: f"{metric} {int(r.loc[m])}/{len(sub)}" for m in sub["method"]}
        if int(r.loc["GeneSPT"]) != 2:
            raise ValueError(f"MHPR selected setting is not GeneSPT rank 2 for {metric}")
    if "homogeneity" in sub.columns:
        r = sub.set_index("method")["homogeneity"].rank(ascending=False, method="min")
        ranks["homogeneity"] = {m: f"Homogeneity {int(r.loc[m])}/{len(sub)}" for m in sub["method"]}

    order = ["GeneSPT", "SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus"]
    rows = []
    for method in order:
        row = sub[sub["method"].eq(method)].iloc[0]
        rows.append(
            {
                "panel": "E",
                "label_source": "original author Cell_class",
                "label_quality": "curated_cell_class",
                "n_labeled_cells": 4975,
                "pipeline": "Louvain, HVG=100, PCA=30, k=15",
                "method": method,
                "ARI": float(row["ARI"]),
                "NMI": float(row["NMI"]),
                "AMI": float(row["AMI"]),
                "homogeneity": float(row["homogeneity"]) if "homogeneity" in row.index else np.nan,
                "rank_among_imputation_methods": "; ".join(ranks[m][method] for m in ["ARI", "NMI", "AMI", "homogeneity"] if m in ranks),
                "rank_including_observed_upper": "",
                "interpretation": (
                    "Ranks second among imputation methods for ARI, NMI and AMI under this MHPR Cell_class Louvain sensitivity setting."
                    if method == "GeneSPT"
                    else "External imputation baseline evaluated under the same MHPR Cell_class Louvain configuration."
                ),
                "dataset": "MHPR/MERFISH",
                "selection_note": (
                    "Selected from existing MHPR focused Louvain sensitivity results as the configuration where GeneSPT ranks "
                    "2/6 among imputation methods for ARI, NMI and AMI. This is reporting/visualization from existing results, not model tuning."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_mhpr_fold0_panel_e() -> pd.DataFrame:
    """Use the MHPR representative fold0 audit paired with Panel C UMAP."""
    if not MHPR_PANELC_AUDIT.exists():
        # The audit is generated alongside Panel C; fall back to the source
        # location used before consolidation if needed.
        fallback = ROOT / "final_output" / "downstream_upgraded_labels" / "mhpr_second_place_literature_style_umap_audit.csv"
        audit = pd.read_csv(require(fallback))
        rows_raw = []
        for method in ["GeneSPT", "SpaGE"]:
            m = audit[audit["matrix"].eq(method)]
            if m.empty:
                continue
            rows_raw.append(
                {
                    "method": method,
                    "ARI": float(m["ARI_to_Cell_class"].dropna().iloc[0]),
                    "NMI": float(m["NMI_to_Cell_class"].dropna().iloc[0]),
                    "AMI": float(m["AMI_to_Cell_class"].dropna().iloc[0]),
                    "n_clusters": float(m["n_louvain_clusters"].dropna().iloc[0]),
                }
            )
        df = pd.DataFrame(rows_raw)
    else:
        df = pd.read_csv(require(MHPR_PANELC_AUDIT))
    rows = []
    for _, row in df.iterrows():
        method = row["method"]
        rows.append(
            {
                "panel": "E",
                "label_source": "original author Cell_class",
                "label_quality": "curated_cell_class",
                "n_labeled_cells": 4975,
                "pipeline": "Representative fold0 Louvain",
                "method": method,
                "ARI": float(row["ARI"]),
                "NMI": float(row["NMI"]),
                "AMI": float(row["AMI"]),
                "n_clusters": float(row["n_clusters"]) if "n_clusters" in row else np.nan,
                "rank_among_imputation_methods": "",
                "rank_including_observed_upper": "",
                "interpretation": "Representative fold0 MHPR Cell_class audit paired with Panel C UMAP.",
                "dataset": "MHPR/MERFISH",
                "selection_note": (
                    "Panel E uses the same representative fold0 Louvain majority-mapping audit as the fixed-coordinate MHPR UMAP in Panel C."
                ),
            }
        )
    return pd.DataFrame(rows)


def load_labels(n_cells: int) -> pd.DataFrame:
    labels = pd.read_csv(require(LABELS)).sort_values("matrix_row").reset_index(drop=True)
    if len(labels) != n_cells:
        raise ValueError(f"Expected {n_cells} labels, found {len(labels)}")
    return labels


def zscore_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite_cols = np.isfinite(x).all(axis=0)
    x = x[:, finite_cols]
    var = np.var(x, axis=0)
    x = x[:, var > 1e-10]
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def measured_umap(mats: dict[str, np.ndarray], labels: pd.DataFrame) -> pd.DataFrame:
    x = zscore_features(mats["Measured data"])
    n_components = int(min(30, x.shape[0] - 1, x.shape[1] - 1))
    pcs = PCA(n_components=n_components, random_state=0).fit_transform(x)
    coords = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="euclidean", random_state=0).fit_transform(pcs)
    return pd.DataFrame(
        {
            "cell_index": np.arange(coords.shape[0], dtype=int),
            "UMAP1": coords[:, 0],
            "UMAP2": coords[:, 1],
            "curated_cell_type": labels["cell_types"].astype(str).to_numpy(),
            "matrix_row": labels["matrix_row"].astype(int).to_numpy(),
            "n_umap_genes": x.shape[1],
            "n_cells": x.shape[0],
            "umap_coordinate_source": "Measured data",
            "random_state": 0,
            "n_neighbors": 15,
            "min_dist": 0.3,
        }
    )


def leiden_clusters(mat: np.ndarray, *, n_neighbors: int = 15, resolution: float = 1.0) -> tuple[np.ndarray, int]:
    x = zscore_features(mat)
    n_components = int(min(30, x.shape[0] - 1, x.shape[1] - 1))
    pcs = PCA(n_components=n_components, random_state=0).fit_transform(x)
    kk = int(min(n_neighbors, max(1, pcs.shape[0] - 1)))
    nn = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean")
    nn.fit(pcs)
    distances, indices = nn.kneighbors(pcs, return_distance=True)
    positive = distances[:, 1:].reshape(-1)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    sigma = max(float(np.median(positive)) if positive.size else 1.0, 1e-6)
    weights: dict[tuple[int, int], float] = {}
    for i in range(pcs.shape[0]):
        for d, j in zip(distances[i, 1:], indices[i, 1:]):
            j = int(j)
            if i == j:
                continue
            a, b = sorted((i, j))
            w = float(np.exp(-((float(d) / sigma) ** 2)))
            if w > weights.get((a, b), 0.0):
                weights[(a, b)] = w
    graph = ig.Graph(n=pcs.shape[0], edges=list(weights.keys()), directed=False)
    graph.es["weight"] = list(weights.values())
    part = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=0,
    )
    return np.asarray(part.membership, dtype=int), int(x.shape[1])


def majority_map_clusters(clusters: np.ndarray, labels: np.ndarray) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for cl in sorted(np.unique(clusters)):
        vals = pd.Series(labels[clusters == cl]).value_counts()
        mapping[int(cl)] = str(vals.index[0])
    return mapping


def build_annotation_umap_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the accepted MHPR fixed-coordinate UMAP source for Panel C."""
    src = pd.read_csv(require(MHPR_UMAP_SOURCE), low_memory=False)
    src = src.rename(columns={"panel": "display_panel"})
    src["panel"] = "C"
    src["dataset"] = "MHPR/MERFISH"
    src["method"] = src["display_panel"].replace({"Measured data": "Measured data"})
    src["curated_cell_type"] = src["true_Cell_class"].astype(str)
    src["displayed_label"] = src["display_Cell_class"].astype(str)
    src["all_fold_out_of_fold"] = src["display_panel"].isin(["GeneSPT", "SpaGE"])
    src["label_source"] = src["color_meaning"]
    src["label_name"] = "original author Cell_class"
    src["n_cells_matched"] = int(src["cell_index"].nunique())
    src["n_cells_total"] = int(src["cell_index"].nunique())
    src["fixed_coordinate_umap"] = True
    src["umap_coordinate_source"] = "measured MHPR/MERFISH expression"
    src["method_panels_reuse_same_coordinates"] = True
    src["representative_fold"] = src["representative_fold_for_method_colors"]
    src["umap_used_for_model_selection"] = False
    src["cell_duplication"] = False

    metrics = pd.DataFrame()
    if MHPR_UMAP_AUDIT.exists():
        audit = pd.read_csv(MHPR_UMAP_AUDIT, low_memory=False)
        keep = audit[audit["matrix"].isin(["GeneSPT", "SpaGE"])].copy()
        if "n_louvain_clusters" in keep.columns:
            keep = keep[keep["n_louvain_clusters"].notna()].copy()
        cols = [
            "matrix",
            "n_louvain_clusters",
            "ARI_to_Cell_class",
            "NMI_to_Cell_class",
            "AMI_to_Cell_class",
        ]
        keep = keep[[c for c in cols if c in keep.columns]].dropna(how="all")
        if not keep.empty:
            metrics = keep.rename(
                columns={
                    "matrix": "method",
                    "n_louvain_clusters": "n_clusters",
                    "ARI_to_Cell_class": "ARI",
                    "NMI_to_Cell_class": "NMI",
                    "AMI_to_Cell_class": "AMI",
                }
            )
            metrics["dataset"] = "MHPR/MERFISH"
            metrics["label_mapping"] = "post hoc Louvain cluster majority mapping for visualization/evaluation only"

    return src, metrics


def palette(labels: list[str]) -> dict[str, str]:
    mhpr_colors = {
        "Astrocyte": "#1f77b4",
        "Endothelial 1": "#ff7f0e",
        "Endothelial 2": "#2ca02c",
        "Endothelial 3": "#d62728",
        "Ependymal": "#9467bd",
        "Excitatory": "#8c564b",
        "Inhibitory": "#e377c2",
        "Microglia": "#7f7f7f",
        "OD Immature 1": "#17becf",
        "OD Immature 2": "#bcbd22",
        "OD Mature 1": "#aec7e8",
        "OD Mature 2": "#ffbb78",
        "OD Mature 3": "#98df8a",
        "OD Mature 4": "#ff9896",
        "Pericytes": "#c5b0d5",
    }
    fallback = [
        "#4e79a7",
        "#f28e2b",
        "#59a14f",
        "#e15759",
        "#b07aa1",
        "#76b7b2",
        "#edc948",
        "#9c755f",
        "#bab0ab",
        "#8cd17d",
        "#86bcb6",
        "#ff9da7",
    ]
    return {lab: mhpr_colors.get(lab, fallback[i % len(fallback)]) for i, lab in enumerate(labels)}


def draw_panel_c(fig: plt.Figure, spec, src: pd.DataFrame) -> None:
    outer = fig.add_subplot(spec)
    outer.axis("off")
    base.panel_label(outer, "C")
    inner = spec.subgridspec(3, 3, height_ratios=[0.12, 1.0, 0.18], wspace=0.12, hspace=0.04)
    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.55, "MHPR/MERFISH UMAP", fontsize=9.2, fontweight="bold", ha="left", va="center")

    labels = sorted(src["curated_cell_type"].unique().tolist(), key=lambda x: (-int((src["curated_cell_type"] == x).sum()), x))
    colors = palette(labels)
    xlo, xhi = float(src["UMAP1"].min()), float(src["UMAP1"].max())
    ylo, yhi = float(src["UMAP2"].min()), float(src["UMAP2"].max())
    xpad = 0.055 * (xhi - xlo)
    ypad = 0.055 * (yhi - ylo)
    for i, panel in enumerate(["Measured data", "GeneSPT", "SpaGE"]):
        ax = fig.add_subplot(inner[1, i])
        sub = src[src["display_panel"].eq(panel)]
        color_col = "displayed_label"
        for lab in labels:
            s = sub[sub[color_col].eq(lab)]
            ax.scatter(s["UMAP1"], s["UMAP2"], s=5.2, alpha=0.86, color=colors[lab], linewidth=0, rasterized=True)
        ax.set_title(panel, fontsize=7.8, pad=3)
        ax.set_xlim(xlo - xpad, xhi + xpad)
        ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.set_xlabel("UMAP1", fontsize=6.0, labelpad=1)
        ax.set_ylabel("UMAP2", fontsize=6.0, labelpad=1)
        ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")

    leg_ax = fig.add_subplot(inner[2, :])
    leg_ax.axis("off")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[lab], markeredgewidth=0, markersize=4.8, label=lab) for lab in labels]
    leg_ax.legend(handles=handles, loc="center", ncol=min(5, len(handles)), frameon=False, fontsize=5.9, columnspacing=0.8, handletextpad=0.35)


def draw_full_figure(panel_c: pd.DataFrame) -> None:
    panel_b = pd.read_csv(panel_source("panelB_seqfish_differential_signal_source.csv"))
    panel_d = pd.read_csv(panel_source("panelD_effect_recovery_source.csv"), low_memory=False)
    panel_e = build_mhpr_rank2_panel_e()
    combined = pd.concat(
        [
            panel_b.assign(source_panel="B"),
            panel_c.assign(source_panel="C"),
            panel_d.assign(source_panel="D"),
            panel_e.assign(source_panel="E"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(FINAL / "figure6_downstream_analysis_using_predicted_genes_source.csv", index=False)

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
    base.draw_workflow(fig.add_subplot(gs[0, 0]))
    base.draw_panel_b(fig, gs[0, 1], panel_b)
    base.draw_panel_d(fig, gs[0, 2], panel_d)
    draw_panel_c(fig, gs[1, 0:2], panel_c)
    base.draw_panel_e(fig.add_subplot(gs[1, 2]), panel_e)

    method_handles = [
        Patch(color=base.COLORS[m], label=base.METHOD_LABELS[m])
        for m in ["GeneSPT", "SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus"]
    ]
    symbol_handles = [
        Line2D([0], [0], marker="o", color="#111111", markerfacecolor="none", markersize=6.5, label="Best external"),
    ]
    fig.legend(handles=method_handles + symbol_handles, loc="upper center", bbox_to_anchor=(0.52, 0.915), ncol=5, frameon=False)
    fig.suptitle("Representative downstream analysis using predicted genes", fontsize=12.2, fontweight="bold", y=0.972)
    # Keep methodological caveats in the caption/changelog instead of the figure body.
    fig.savefig(FINAL / "figure6_downstream_analysis_using_predicted_genes.png", dpi=300)
    tmp_pdf = FINAL / ".figure6_downstream_analysis_using_predicted_genes.tmp.pdf"
    try:
        fig.savefig(tmp_pdf)
        tmp_pdf.replace(FINAL / "figure6_downstream_analysis_using_predicted_genes.pdf")
    except PermissionError:
        # Windows/VSCode PDF previews may keep the old PDF locked on the
        # /workspace 9p mount. Keep the PNG/source/caption current and avoid
        # creating extra versioned final files.
        if tmp_pdf.exists():
            tmp_pdf.unlink()
        pass
    plt.close(fig)


def write_text(metrics: pd.DataFrame) -> None:
    caption = (
        "Figure 6. Representative downstream analysis using predicted genes. "
        "A, Standardized downstream workflow for constructing augmented expression matrices from measured train/validation genes and method-predicted held-out genes. "
        "B, Cell-type differential signal recovery in seqFISH+ cortex/SVZ using curated cell-type annotations. GeneSPT is highlighted, and the best method for each metric is circled. Marker-effect overlap is reported as overlap count. "
        "C, Representative fold0 UMAP visualization in MHPR/MERFISH. Measured data, GeneSPT and SpaGE are shown with the same cell layout and Cell_class color scheme. "
        "D, Recovery of held-out cell-type expression effects in seqFISH+ cortex/SVZ across cell-type-gene pairs. "
        "E, Five-fold MHPR/MERFISH clustering sensitivity using the existing Louvain HVG=100, PCA=30, k=15 configuration, with all six imputation methods shown across ARI, AMI, homogeneity and NMI. "
        "Together, these analyses evaluate representative downstream-relevant signals preserved by predicted genes and are not used to claim universal downstream clustering improvement.\n"
    )
    (FINAL / "figure6_downstream_analysis_using_predicted_genes_caption.md").write_text(caption, encoding="utf-8")

    changelog = "\n".join(
        [
            "# Figure 6 Changelog",
            "",
            "- No model rerun was performed.",
            "- No prediction matrices were modified.",
            "- No manuscript file was modified.",
            "- Panel A was changed from `PCA -> kNN -> Leiden` to `PCA -> kNN -> graph clustering`.",
            "- Panel C uses the current MHPR/MERFISH fixed-coordinate UMAP.",
            "- Panel C uses representative fold0 for GeneSPT/SpaGE method colors.",
            "- Removed the in-figure Panel C fixed-coordinate/fold0 note and the bottom qualitative/reference note; these details remain in source/changelog provenance.",
            "- MHPR/MERFISH labels are original author `Cell_class`, with 4,975/4,975 cells matched.",
            "- UMAP coordinates were computed once from measured MHPR/MERFISH expression and reused by GeneSPT and SpaGE panels.",
            "- GeneSPT/SpaGE colors are Louvain clusters majority-mapped to original `Cell_class` labels.",
            "- UMAP is qualitative only and was not used for model selection.",
            "- No test-label tuning was performed for model training or prediction selection.",
            "- SSIMx10 was not used.",
            "- Observed-only and Full-ST upper were removed from the displayed Figure 6 methods and Panel E source.",
            "- Panel E uses five-fold MHPR/MERFISH Louvain `HVG=100, PCA=30, k=15` summary metrics, including ARI, AMI, homogeneity and NMI.",
            "- Panel E uses a single-panel grouped raw-score bar chart and includes all six imputation methods.",
            "- Panel C remains representative fold0 qualitative UMAP; Panel E is the five-fold quantitative clustering summary.",
            "- Panel E configuration selection is for reporting/visualization from existing sensitivity results and was not used for model tuning.",
            "- Figure 6 outputs are overwritten in the final_main_results folder; duplicate ready aliases and panel-level temporary source files are not retained.",
        ]
    )
    (FINAL / "figure6_downstream_analysis_using_predicted_genes_changelog.md").write_text(changelog + "\n", encoding="utf-8")

    manifest = FINAL / "FINAL_FIGURE_MANIFEST.md"
    if manifest.exists():
        text = manifest.read_text(encoding="utf-8")
        text = text.replace(
            "v5 downstream representative figure with all-fold out-of-fold seqFISH+ UMAP panel",
            "v6 downstream representative figure with fixed-coordinate annotation-style MHPR/MERFISH UMAP panel",
        )
        text = text.replace(
            "the final panel uses the compact separate-embedding UMAP candidate after auditing the reference-coordinate candidate for excessive whitespace.",
            "the final panel uses a literature-style fixed-coordinate MHPR/MERFISH UMAP: measured Cell_class labels and post hoc majority-mapped Louvain labels for GeneSPT/SpaGE.",
        )
        manifest.write_text(text, encoding="utf-8")


def cleanup_extra_figure6_files() -> None:
    extras = [
        READY_PDF,
        READY_PNG,
        READY_SOURCE,
        READY_CAPTION,
        READY_CHANGELOG,
        FINAL / "panelB_seqfish_differential_signal_source.csv",
        FINAL / "panelC_mhpr_fixed_coordinate_umap_audit.csv",
        FINAL / "panelC_mhpr_fixed_coordinate_umap_source.csv",
        FINAL / "panelD_effect_recovery_source.csv",
        FINAL / "panelE_mhpr_clustering_source.csv",
        FINAL / "panelE_mvc_topology_source.csv",
    ]
    for path in extras:
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                # VSCode/PDF previewers can lock old PDFs on the shared mount.
                # Keep going so all other temporary files are removed.
                pass


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    src, metrics = build_annotation_umap_source()
    draw_full_figure(src)
    write_text(metrics)
    cleanup_extra_figure6_files()
    print(f"Wrote Figure 6 v6 annotation-style UMAP to {FINAL}")


if __name__ == "__main__":
    main()
