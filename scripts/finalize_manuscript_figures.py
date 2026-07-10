#!/usr/bin/env python3
"""Update draft figures and collect the latest manuscript figures in one folder."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/workspace/GeneSPT")
FINAL = ROOT / "final_output" / "final_main_results"
FIG4_SOURCE = ROOT / "final_output" / "submission_pack" / "figure4_hbc_representative_maps_source.csv"


RED = "#c62828"
TEXT = "#222222"
MUTED = "#666666"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def normalize_for_visualization(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(values, [2, 98])
    clipped = np.clip(values, lo, hi)
    denom = float(np.nanmax(clipped) - np.nanmin(clipped))
    if denom <= 1e-12:
        return np.zeros_like(clipped, dtype=float)
    return (clipped - float(np.nanmin(clipped))) / denom


def prediction_vector(npz_path: Path, gene_idx: int) -> np.ndarray:
    arr = np.load(require(npz_path), allow_pickle=True)
    test_idx = arr["test_idx"].astype(int)
    matches = np.where(test_idx == int(gene_idx))[0]
    if len(matches) != 1:
        raise ValueError(f"gene_idx {gene_idx} not found exactly once in {npz_path}")
    return arr["pred_test"][:, int(matches[0])].astype(float)


def draw_map(ax: plt.Axes, coords: pd.DataFrame, values: np.ndarray, title: str) -> mpl.collections.PathCollection:
    sc = ax.scatter(
        coords["x"],
        coords["y"],
        c=values,
        s=8.0,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=11.2, pad=5, color=TEXT)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return sc


def generate_figure4() -> None:
    manifest = pd.read_csv(require(FIG4_SOURCE))
    counts = pd.read_csv(require(Path(manifest["ground_truth_source"].iloc[0])), sep="\t", dtype=np.float32)
    coords = pd.read_csv(require(Path(manifest["coordinate_source"].iloc[0])), sep="\t")

    fig = plt.figure(figsize=(11.6, 6.5), constrained_layout=False)
    gs = fig.add_gridspec(
        len(manifest),
        5,
        width_ratios=[0.62, 1.0, 1.0, 1.0, 0.045],
        left=0.055,
        right=0.965,
        top=0.86,
        bottom=0.13,
        wspace=0.08,
        hspace=0.18,
    )
    fig.text(0.055, 0.955, "HBC representative held-out gene maps", fontsize=13.5, fontweight="bold", ha="left", va="top")

    for r, row in manifest.iterrows():
        label_ax = fig.add_subplot(gs[r, 0])
        label_ax.axis("off")
        gene = str(row["gene"])
        fold = int(row["fold"])
        best_method = str(row["best_external_method"])
        gene_idx = int(row["gene_idx"])

        label_ax.text(0.95, 0.58, gene, fontsize=13.5, fontweight="bold", ha="right", va="center", color=TEXT)
        label_ax.text(0.95, 0.43, f"fold{fold} test gene", fontsize=9.0, ha="right", va="center", color=MUTED)
        label_ax.text(0.95, 0.31, f"external: {best_method}", fontsize=9.0, ha="right", va="center", color=MUTED)

        gt = counts[gene].to_numpy(float)
        genespt = prediction_vector(Path(row["GeneSPT_prediction_source"]), gene_idx)
        external = prediction_vector(Path(row["best_external_prediction_source"]), gene_idx)
        values = [normalize_for_visualization(v) for v in [gt, genespt, external]]
        titles = [
            "Ground truth",
            f"GeneSPT\nSPCC={float(row['GeneSPT_SPCC']):.3f}",
            f"{best_method}\nSPCC={float(row['best_external_SPCC']):.3f}",
        ]

        axes = [fig.add_subplot(gs[r, c]) for c in [1, 2, 3]]
        cax = fig.add_subplot(gs[r, 4])
        sc = None
        for ax, arr, title in zip(axes, values, titles):
            sc = draw_map(ax, coords, arr, title)
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=7.8, length=2)
        cb.set_label("normalized expression", fontsize=8.0, labelpad=4)
        cax.yaxis.set_ticks_position("right")

    fig.text(
        0.055,
        0.055,
        "External comparator = strongest external baseline for each gene. Spatial maps are normalized for visualization only; SPCC values were computed on original prediction scale.",
        fontsize=8.7,
        color=MUTED,
        ha="left",
        va="bottom",
    )

    out_pdf = FINAL / "figure4_hbc_representative_maps.pdf"
    out_png = FINAL / "figure4_hbc_representative_maps.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    shutil.copy2(FIG4_SOURCE, FINAL / "figure4_hbc_representative_maps_source.csv")
    caption = (
        "Figure 4. Representative held-out gene predictions in the HBC dataset. "
        "Ground-truth spatial expression maps are compared with GeneSPT and the strongest external comparator for selected held-out genes. "
        "The external map title gives the selected method name and per-gene SPCC; the external comparator is the best-performing external method for that gene. "
        "Spatial maps were normalized within each map for visualization only, and all metric values were computed on the original prediction scale.\n"
    )
    (FINAL / "figure4_hbc_representative_maps_caption.md").write_text(caption, encoding="utf-8")


def copy_latest_figures() -> list[tuple[str, str, str]]:
    copied: list[tuple[str, str, str]] = []

    pairs = [
        (
            ROOT / "final_output" / "figure3_redesign" / "figure3_mechanism_ablation_redesign.pdf",
            FINAL / "figure3_mechanism_ablation.pdf",
            "Figure 3 PDF",
        ),
        (
            ROOT / "final_output" / "figure3_redesign" / "figure3_mechanism_ablation_redesign.png",
            FINAL / "figure3_mechanism_ablation.png",
            "Figure 3 PNG",
        ),
        (
            ROOT / "final_output" / "figure3_redesign" / "figure3_mechanism_ablation_redesign_source.csv",
            FINAL / "figure3_mechanism_ablation_source.csv",
            "Figure 3 source",
        ),
        (
            ROOT / "final_output" / "figure3_redesign" / "figure3_mechanism_ablation_redesign_caption.md",
            FINAL / "figure3_mechanism_ablation_caption.md",
            "Figure 3 caption",
        ),
        (
            FINAL / "pdf_exports" / "figure5_cross_platform_per_gene_performance_main.pdf",
            FINAL / "figure5_cross_platform_per_gene_performance_main.pdf",
            "Figure 5 PDF",
        ),
        (
            FINAL / "source_csv" / "figure5_cross_platform_per_gene_performance_main_source.csv",
            FINAL / "figure5_cross_platform_per_gene_performance_main_source.csv",
            "Figure 5 source",
        ),
        (
            FINAL / "captions" / "figure5_cross_platform_per_gene_performance_main_caption.md",
            FINAL / "figure5_cross_platform_per_gene_performance_main_caption.md",
            "Figure 5 caption",
        ),
        (
            FINAL / "figure6_downstream_analysis_using_predicted_genes.pdf",
            FINAL / "figure6_downstream_analysis_using_predicted_genes.pdf",
            "Figure 6 PDF",
        ),
        (
            FINAL / "figure6_downstream_analysis_using_predicted_genes.png",
            FINAL / "figure6_downstream_analysis_using_predicted_genes.png",
            "Figure 6 PNG",
        ),
        (
            FINAL / "figure6_downstream_analysis_using_predicted_genes_source.csv",
            FINAL / "figure6_downstream_analysis_using_predicted_genes_source.csv",
            "Figure 6 source",
        ),
        (
            FINAL / "figure6_downstream_analysis_using_predicted_genes_caption.md",
            FINAL / "figure6_downstream_analysis_using_predicted_genes_caption.md",
            "Figure 6 caption",
        ),
        (
            ROOT / "final_output" / "main_downstream_figure6_final" / "figure6_main_text_paragraph_cn.md",
            FINAL / "figure6_main_text_paragraph_cn.md",
            "Figure 6 CN paragraph",
        ),
        (
            ROOT / "final_output" / "main_downstream_figure6_final" / "figure6_main_text_paragraph_en.md",
            FINAL / "figure6_main_text_paragraph_en.md",
            "Figure 6 EN paragraph",
        ),
    ]

    for src, dst, label in pairs:
        if src.resolve() == dst.resolve():
            copied.append((label, str(dst), "already in final folder"))
            continue
        require(src)
        shutil.copy2(src, dst)
        copied.append((label, str(dst), "copied"))
    return copied


def write_manifests(copied: list[tuple[str, str, str]]) -> None:
    items = [
        ("Figure 1", "", "", "not found as a finalized image in final_output; workflow schematic still needs manual/source figure insertion if used"),
        (
            "Figure 2",
            "figure2_primary_benchmark_dotplot.pdf / .png",
            "figure2_primary_benchmark_dotplot_caption.md",
            "primary benchmark dot plot already in this final folder",
        ),
        (
            "Figure 3",
            "figure3_mechanism_ablation.pdf / .png",
            "figure3_mechanism_ablation_caption.md",
            "latest redesigned mechanism ablation copied here",
        ),
        (
            "Figure 4",
            "figure4_hbc_representative_maps.pdf / .png",
            "figure4_hbc_representative_maps_caption.md",
            "redrawn so the external map title uses the method name and SPCC only",
        ),
        (
            "Figure 5",
            "figure5_cross_platform_per_gene_performance_main.pdf / .png",
            "figure5_cross_platform_per_gene_performance_main_caption.md",
            "cross-platform per-gene violin main figure",
        ),
        (
            "Figure 6",
            "figure6_downstream_analysis_using_predicted_genes.pdf / .png",
            "figure6_downstream_analysis_using_predicted_genes_caption.md",
            "current downstream representative figure with fixed-coordinate annotation-style seqFISH+ UMAP panel",
        ),
    ]
    manifest_lines = [
        "# Final Manuscript Figure Manifest",
        "",
        f"Unified final folder: `{FINAL}`",
        "",
        "| item | figure files | caption | status / notes |",
        "|---|---|---|---|",
    ]
    for item, fig, cap, note in items:
        manifest_lines.append(f"| {item} | {fig or 'missing'} | {cap or 'missing'} | {note} |")
    manifest_lines.extend(
        [
            "",
            "## Packaging Notes",
            "",
            "- Future figure edits should overwrite files in this folder.",
            "- In Figure 4, in-panel external labels no longer say `Best external baseline`; the map title shows the external method name and SPCC.",
            "- In Figure 6 Panel C, the cluster-composition heatmap was replaced with a literature-style fixed-coordinate annotation UMAP: ground truth cell types and post hoc majority-mapped Leiden labels for GeneSPT/SpaGE.",
            "- Captions/legends carry the explanation of strongest external comparator where needed.",
        ]
    )
    (FINAL / "FINAL_FIGURE_MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    change_lines = [
        "# Final Figure Collection Changelog",
        "",
        "- Used existing folder `/workspace/GeneSPT/final_output/final_main_results` as the unified final manuscript figure folder; no new final folder was created.",
        "- Redrew Figure 4 from existing source CSV and saved prediction matrices, without rerunning models or changing prediction values.",
        "- Removed `Best external baseline` from Figure 4 map titles; third-column titles now show only the external method name and SPCC.",
        "- Updated Figure 6 source script so the cluster-composition panel uses the external method name in-panel rather than `Best external` wording.",
        "- Figure 6 is stored only under canonical filenames; older versioned Figure 6 files were removed.",
        "- Patched HBC case-study plotting scripts so future regenerations keep external-map titles as method name plus SPCC.",
        "- Copied the latest Figure 3, Figure 5, and Figure 6 files into the unified final folder.",
        "- Did not modify manuscript files or prediction matrices.",
        "- Did not use SSIMx10 or recompute model outputs.",
        "",
        "## Copied Files",
        "",
        "| item | destination | status |",
        "|---|---|---|",
    ]
    for label, dst, status in copied:
        change_lines.append(f"| {label} | `{dst}` | {status} |")
    (FINAL / "final_figure_collection_changelog.md").write_text("\n".join(change_lines) + "\n", encoding="utf-8")


def main() -> None:
    apply_style()
    FINAL.mkdir(parents=True, exist_ok=True)
    generate_figure4()
    copied = copy_latest_figures()
    write_manifests(copied)


if __name__ == "__main__":
    main()
