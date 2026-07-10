#!/usr/bin/env python3
"""Package submission figures/tables for use next to a Word draft."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path("/workspace/GeneSPT")
SRC = ROOT / "final_output" / "submission_pack"
FINAL = ROOT / "final_output" / "final_main_results"
OUT = ROOT / "final_output" / "submission_pack_for_word"
ZIP_PATH = ROOT / "final_output" / "submission_pack_for_word.zip"


def rel(path: Path) -> str:
    return path.relative_to(OUT).as_posix()


def ensure_dirs() -> None:
    for name in ["figures", "captions", "tables", "supplementary_figures", "supplementary_tables", "logs"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst_rel: str, records: list[dict], group: str, label: str, required: bool = False) -> Path | None:
    dst = OUT / dst_rel
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records.append({"group": group, "label": label, "status": "copied", "source": str(src), "target": dst_rel})
        return dst
    records.append({"group": group, "label": label, "status": "missing_required" if required else "missing_optional", "source": str(src), "target": dst_rel})
    return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")).replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_dirs()
    records: list[dict] = []

    # Main figures. Figure 1 image is optional; no current image was found in the project.
    figure1_candidates = [
        ROOT / "final_output" / "figure1_method_schematic.png",
        ROOT / "results" / "imformation" / "final_manuscript_figures" / "figure1_method_schematic.png",
        ROOT / "final_output" / "figure1_method_schematic.pdf",
    ]
    fig1_copied = None
    for cand in figure1_candidates:
        if cand.exists():
            suffix = cand.suffix.lower()
            fig1_copied = copy_file(cand, f"figures/figure1_method_schematic{suffix}", records, "main_figure", "Figure 1 image", required=False)
            break
    if fig1_copied is None:
        records.append({"group": "main_figure", "label": "Figure 1 image", "status": "missing_optional", "source": "searched known Figure 1 image candidates", "target": "figures/figure1_method_schematic.[png/pdf]"})
        copy_file(
            ROOT / "results" / "imformation" / "final_manuscript_figures" / "figure1_method_schematic_plan.md",
            "logs/figure1_method_schematic_plan.md",
            records,
            "log",
            "Figure 1 schematic plan",
            required=False,
        )

    main_fig_specs = [
        ("Figure 2", SRC / "figure2_primary_benchmark_dotplot_final.png", "figures/figure2_primary_benchmark_dotplot_final.png"),
        ("Figure 2 PDF", SRC / "figure2_primary_benchmark_dotplot_final.pdf", "figures/figure2_primary_benchmark_dotplot_final.pdf"),
        ("Figure 3", SRC / "figure3_mechanism_ablation.png", "figures/figure3_mechanism_ablation.png"),
        ("Figure 3 PDF", SRC / "figure3_mechanism_ablation.pdf", "figures/figure3_mechanism_ablation.pdf"),
        ("Figure 4", SRC / "figure4_hbc_representative_maps.png", "figures/figure4_hbc_representative_maps.png"),
        ("Figure 4 PDF", SRC / "figure4_hbc_representative_maps.pdf", "figures/figure4_hbc_representative_maps.pdf"),
        ("Figure 5", FINAL / "figure5_cross_platform_per_gene_performance_main.png", "figures/figure5_cross_platform_per_gene_performance_main.png"),
        ("Figure 5 PDF", FINAL / "pdf_exports" / "figure5_cross_platform_per_gene_performance_main.pdf", "figures/figure5_cross_platform_per_gene_performance_main.pdf"),
    ]
    for label, src, dst in main_fig_specs:
        copy_file(src, dst, records, "main_figure", label, required=not label.endswith("PDF"))

    # Captions.
    caption_specs = [
        ("Figure 2", SRC / "figure2_primary_benchmark_dotplot_final_caption.md", "captions/figure2_caption.md"),
        ("Figure 3", SRC / "figure3_mechanism_ablation_caption.md", "captions/figure3_caption.md"),
        ("Figure 4", SRC / "figure4_hbc_representative_maps_caption.md", "captions/figure4_caption.md"),
        ("Figure 5", FINAL / "captions" / "figure5_cross_platform_per_gene_performance_main_caption.md", "captions/figure5_caption.md"),
        ("Supplementary primary per-gene full metrics", FINAL / "captions" / "supp_figure_primary_per_gene_full_metrics_caption.md", "captions/supp_figure_primary_per_gene_full_metrics_caption.md"),
        ("Supplementary cross-platform per-gene full metrics", FINAL / "captions" / "supp_figure_cross_platform_per_gene_full_metrics_caption.md", "captions/supp_figure_cross_platform_per_gene_full_metrics_caption.md"),
    ]
    for label, src, dst in caption_specs:
        copy_file(src, dst, records, "caption", label, required=label.startswith("Figure"))

    # Main tables.
    table_specs = [
        ("Table 1 dataset/evaluation summary", ROOT / "final_output" / "table1_dataset_evaluation_summary.md", "tables/table1_dataset_evaluation_summary.md"),
        ("Table 1 dataset/evaluation summary CSV", ROOT / "final_output" / "table1_dataset_evaluation_summary.csv", "tables/table1_dataset_evaluation_summary.csv"),
        ("Table 2 GeneSPT final performance summary", FINAL / "final_table2_v2.md", "tables/table2_genespt_final_performance_summary.md"),
        ("Table 2 GeneSPT final performance summary CSV", FINAL / "source_csv" / "final_table2_v2.csv", "tables/table2_genespt_final_performance_summary.csv"),
    ]
    for label, src, dst in table_specs:
        copy_file(src, dst, records, "main_table", label, required=True)

    # Supplementary figures.
    supp_fig_specs = [
        ("Primary per-gene full metrics", SRC / "supp_figure_primary_per_gene_full_metrics.png", "supplementary_figures/supp_figure_primary_per_gene_full_metrics.png"),
        ("Primary per-gene full metrics PDF", SRC / "supp_figure_primary_per_gene_full_metrics.pdf", "supplementary_figures/supp_figure_primary_per_gene_full_metrics.pdf"),
        ("Cross-platform per-gene full metrics", SRC / "supp_figure_cross_platform_per_gene_full_metrics.png", "supplementary_figures/supp_figure_cross_platform_per_gene_full_metrics.png"),
        ("Cross-platform per-gene full metrics PDF", SRC / "supp_figure_cross_platform_per_gene_full_metrics.pdf", "supplementary_figures/supp_figure_cross_platform_per_gene_full_metrics.pdf"),
        ("All-dataset per-gene distributions", SRC / "supp_figure_all_dataset_per_gene_distributions.png", "supplementary_figures/supp_figure_all_dataset_per_gene_distributions.png"),
        ("All-dataset per-gene distributions PDF", SRC / "supp_figure_all_dataset_per_gene_distributions.pdf", "supplementary_figures/supp_figure_all_dataset_per_gene_distributions.pdf"),
        ("Ranked overview", SRC / "supp_figure_ranked_overview.png", "supplementary_figures/supp_figure_ranked_overview.png"),
        ("Ranked overview PDF", SRC / "supp_figure_ranked_overview.pdf", "supplementary_figures/supp_figure_ranked_overview.pdf"),
    ]
    for label, src, dst in supp_fig_specs:
        copy_file(src, dst, records, "supplementary_figure", label, required=False)

    # Supplementary tables.
    for idx, name in enumerate(
        [
            "dataset_provenance",
            "primary_full_metrics",
            "cross_platform_full_metrics",
            "descriptor_ablation",
            "psp_ablation",
            "method_availability",
        ],
        start=1,
    ):
        stem = f"supp_table_s{idx}_{name}"
        copy_file(SRC / f"{stem}.md", f"supplementary_tables/{stem}.md", records, "supplementary_table", stem, required=True)
        copy_file(SRC / f"{stem}.csv", f"supplementary_tables/{stem}.csv", records, "supplementary_table", stem + " CSV", required=True)

    # Logs and helper manifests.
    log_specs = [
        ("Original submission manifest", SRC / "manuscript_insertion_manifest.md", "logs/manuscript_insertion_manifest_original.md"),
        ("Supplementary figure manifest", SRC / "supplementary_figure_manifest.md", "logs/supplementary_figure_manifest.md"),
        ("Final consistency audit", SRC / "final_consistency_audit.md", "logs/final_consistency_audit.md"),
        ("Figure 2/Table 2 consistency check", SRC / "figure2_table2_consistency_check.md", "logs/figure2_table2_consistency_check.md"),
        ("Availability statement draft", SRC / "availability_statement_draft.md", "logs/availability_statement_draft.md"),
        ("Submission pack changelog", SRC / "submission_pack_changelog.md", "logs/submission_pack_changelog.md"),
    ]
    for label, src, dst in log_specs:
        copy_file(src, dst, records, "log", label, required=False)

    # Build Word-oriented manifest with relative paths only.
    captions = {
        "Figure 2": read_text(OUT / "captions" / "figure2_caption.md").strip(),
        "Figure 3": read_text(OUT / "captions" / "figure3_caption.md").strip(),
        "Figure 4": read_text(OUT / "captions" / "figure4_caption.md").strip(),
        "Figure 5": read_text(OUT / "captions" / "figure5_caption.md").strip(),
    }
    figure_rows = [
        {"figure": "Figure 1", "main file": "missing", "caption file": "missing", "word action": "Create/insert schematic manually; see logs/figure1_method_schematic_plan.md if present."},
        {"figure": "Figure 2", "main file": "figures/figure2_primary_benchmark_dotplot_final.png", "caption file": "captions/figure2_caption.md", "word action": "Insert into main manuscript as primary benchmark Figure 2."},
        {"figure": "Figure 3", "main file": "figures/figure3_mechanism_ablation.png", "caption file": "captions/figure3_caption.md", "word action": "Replace Figure 3 placeholders in sections 3.3 and 3.4."},
        {"figure": "Figure 4", "main file": "figures/figure4_hbc_representative_maps.png", "caption file": "captions/figure4_caption.md", "word action": "Insert after section 3.4 as new HBC representative-map section 3.5."},
        {"figure": "Figure 5", "main file": "figures/figure5_cross_platform_per_gene_performance_main.png", "caption file": "captions/figure5_caption.md", "word action": "Insert into cross-platform section; renumber that section to 3.6 if Figure 4 is inserted."},
    ]
    table_rows = [
        {"table": "Table 1", "main file": "tables/table1_dataset_evaluation_summary.md", "csv": "tables/table1_dataset_evaluation_summary.csv", "word action": "Insert into main manuscript dataset/evaluation summary table location."},
        {"table": "Table 2", "main file": "tables/table2_genespt_final_performance_summary.md", "csv": "tables/table2_genespt_final_performance_summary.csv", "word action": "Insert into main manuscript performance summary table location."},
    ]
    supp_table_rows = [
        {"supplement": f"Supplementary Table S{i}", "file": f"supplementary_tables/supp_table_s{i}_{name}.md", "csv": f"supplementary_tables/supp_table_s{i}_{name}.csv"}
        for i, name in enumerate(
            [
                "dataset_provenance",
                "primary_full_metrics",
                "cross_platform_full_metrics",
                "descriptor_ablation",
                "psp_ablation",
                "method_availability",
            ],
            start=1,
        )
    ]
    supp_fig_rows = [
        {"supplement": "Supplementary Figure", "file": "supplementary_figures/supp_figure_primary_per_gene_full_metrics.png", "note": "Primary per-gene full metrics."},
        {"supplement": "Supplementary Figure", "file": "supplementary_figures/supp_figure_cross_platform_per_gene_full_metrics.png", "note": "Cross-platform per-gene full metrics."},
        {"supplement": "Supplementary Figure", "file": "supplementary_figures/supp_figure_all_dataset_per_gene_distributions.png", "note": "All-dataset per-gene distributions."},
        {"supplement": "Supplementary Figure", "file": "supplementary_figures/supp_figure_ranked_overview.png", "note": "Ranked overview."},
    ]

    manifest = [
        "# Word Insertion Manifest",
        "",
        "All paths below are relative to this folder.",
        "",
        "## Main Figures",
        md_table(figure_rows),
        "",
        "## Main Figure Captions",
    ]
    for label in ["Figure 2", "Figure 3", "Figure 4", "Figure 5"]:
        manifest += [f"### {label}", captions[label] or "Caption missing.", ""]
    manifest += [
        "## Main Tables",
        md_table(table_rows),
        "",
        "## Supplementary Figures",
        md_table(supp_fig_rows),
        "",
        "## Supplementary Tables",
        md_table(supp_table_rows),
        "",
        "## Suggested Word Insertions",
        "- Replace Figure 3 placeholders in sections 3.3 and 3.4 with `figures/figure3_mechanism_ablation.png` and `captions/figure3_caption.md`.",
        "- Insert Figure 4 after section 3.4 as a new HBC representative-map section 3.5 using `figures/figure4_hbc_representative_maps.png` and `captions/figure4_caption.md`.",
        "- Renumber the current cross-platform section from 3.5 to 3.6 if Figure 4 is inserted.",
        "- Insert Figure 2 and Figure 5 at their existing main-figure placeholders or nearest matching result sections.",
        "- Insert Table 1 and Table 2 at the main manuscript table placeholders.",
        "- Put files under `supplementary_figures/` and `supplementary_tables/` in the supplement, not the main manuscript.",
        "",
        "## Missing Items",
    ]
    missing = [r for r in records if r["status"].startswith("missing")]
    manifest.append(md_table(missing) if missing else "- None.")
    (OUT / "WORD_INSERTION_MANIFEST.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    # Changelog.
    main_figs = [r for r in records if r["group"] == "main_figure" and r["status"] == "copied" and r["target"].endswith(".png")]
    supp_figs = [r for r in records if r["group"] == "supplementary_figure" and r["status"] == "copied" and r["target"].endswith(".png")]
    tables = [r for r in records if r["group"] in {"main_table", "supplementary_table"} and r["status"] == "copied"]
    changelog = [
        "# Packaging Changelog",
        "",
        "1. Main figures packaged:",
        *[f"- {r['label']}: `{r['target']}`" for r in main_figs],
        "2. Supplementary figures packaged:",
        *[f"- {r['label']}: `{r['target']}`" for r in supp_figs],
        "3. Tables packaged:",
        *[f"- {r['label']}: `{r['target']}`" for r in tables],
        "4. Missing files:",
        *( [f"- {r['label']}: {r['source']}" for r in missing] if missing else ["- None."] ),
        "5. No model was rerun.",
        "6. No prediction matrices were modified.",
        "7. No manuscript file was modified.",
    ]
    (OUT / "packaging_changelog.md").write_text("\n".join(changelog) + "\n", encoding="utf-8")

    # Machine-readable copy log.
    (OUT / "logs" / "packaged_file_inventory.csv").write_text(
        "group,label,status,source,target\n"
        + "\n".join(
            ",".join('"' + str(r[k]).replace('"', '""') + '"' for k in ["group", "label", "status", "source", "target"])
            for r in records
        )
        + "\n",
        encoding="utf-8",
    )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT.parent).as_posix())

    print(f"Packaged Word bundle: {OUT}")
    print(f"Zip: {ZIP_PATH}")


if __name__ == "__main__":
    main()
