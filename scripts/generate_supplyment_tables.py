from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/workspace/GeneSPT")
OUT = ROOT / "final_output" / "supplyment"
OUT.mkdir(parents=True, exist_ok=True)

NA = "NA"


FINAL_DATASETS = [
    ("Vis9A", "Vis9A_D7_spaim_effective4470", "Primary"),
    ("HBC", "HBC_shared16112", "Primary"),
    ("MHM", "MHM_shared14780", "Primary"),
    ("seqFISH+ cortex/SVZ", "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000", "Cross-platform"),
    ("MHPR/MERFISH", "MHPR_current_panel", "Cross-platform"),
    ("MVC/STARmap", "MVC_shared981", "Cross-platform"),
]

METHODS = ["GeneSPT", "SpaIM", "Tangram", "TransPA", "SpaGE", "stPlus"]

SOURCES: dict[str, Path] = {
    "s1_prior": ROOT / "final_output/submission_pack_final/supplementary_tables/supp_table_s1_dataset_provenance.csv",
    "s2_raw": ROOT / "final_output/final_main_results/source_csv/supp_table_all_methods_raw_metrics.csv",
    "figure3_source": ROOT / "final_output/figure3_redesign/figure3_mechanism_ablation_redesign_source.csv",
    "figure3_audit": ROOT / "final_output/figure3_redesign/figure3_redesign_consistency_audit.md",
    "figure6_source": ROOT / "final_output/final_main_results/figure6_downstream_analysis_using_predicted_genes_source.csv",
    "label_alignment": ROOT / "final_output/label_provenance_audit/label_alignment_audit.csv",
    "local_label_inventory": ROOT / "final_output/label_provenance_audit/local_label_file_inventory.csv",
    "label_download_manifest": ROOT / "final_output/label_provenance_audit/downloaded_label_files_manifest.csv",
    "downstream_seqfish": ROOT / "final_output/downstream_validation_supplement/seqfish_downstream_annotation_summary.csv",
    "downstream_mvc": ROOT / "final_output/downstream_validation_supplement/mvc_downstream_topology_sensitivity_summary.csv",
    "upgraded_kmeans": ROOT / "final_output/downstream_upgraded_labels/downstream_upgraded_kmeans_summary.csv",
    "upgraded_leiden_default": ROOT / "final_output/downstream_upgraded_labels/downstream_upgraded_leiden_default_summary.csv",
    "upgraded_leiden_matched": ROOT / "final_output/downstream_upgraded_labels/downstream_upgraded_leiden_cluster_count_matched_summary.csv",
    "mhpr_louvain_focus": ROOT / "final_output/downstream_upgraded_labels/mhpr_cluster_method_sensitivity_label_audit.csv",
    "primary_leiden": ROOT / "final_output/downstream_leiden_primary_sensitivity/primary_leiden_genespt_rank_summary.csv",
    "cross_leiden": ROOT / "final_output/downstream_leiden_sensitivity/leiden_genespt_rank_summary.csv",
}


MASK_DIRS = {
    "Vis9A_D7_spaim_effective4470": ROOT / "results/imformation/final_multidataset_masks/Vis9A_D7_spaim_effective4470",
    "HBC_shared16112": ROOT / "results/imformation/final_multidataset_masks/HBC_shared16112",
    "MHM_shared14780": ROOT / "results/imformation/final_multidataset_masks/MHM_shared14780",
    "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000": ROOT
    / "final_output/seqfish_trials/seqFISH_plus_cortex_svz/masks_zeisel_sccortex_ref_shared10000",
    "MHPR_current_panel": ROOT / "results/imformation/final_multidataset_masks/MHPR_current_panel",
    "MVC_shared981": ROOT / "results/imformation/final_multidataset_masks/MVC_shared981",
}

MANIFESTS = {
    "Vis9A_D7_spaim_effective4470": ROOT / "data/Vis9A_D7_spaim_effective4470/shared_gene_manifest.json",
    "HBC_shared16112": ROOT / "data/HBC_shared16112/shared_gene_manifest.json",
    "MHM_shared14780": ROOT / "data/MHM_shared14780/shared_gene_manifest.json",
    "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000": ROOT
    / "data/seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000/shared_gene_manifest.json",
    "MVC_shared981": ROOT / "data/MVC_shared981/shared_gene_manifest.json",
}


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def first_existing(paths: list[Path]) -> str:
    return "; ".join(rel(p) for p in paths if p.exists()) or NA


def split_counts(dataset_id: str) -> dict[str, Any]:
    mask = MASK_DIRS.get(dataset_id)
    out: dict[str, Any] = {
        "n_train_genes": NA,
        "n_validation_genes": NA,
        "n_test_genes": NA,
        "n_folds": 0,
        "split_source": NA,
    }
    if not mask or not mask.exists():
        return out
    split_files = sorted(mask.glob("fold*_split.json"))
    out["n_folds"] = len(split_files)
    out["split_source"] = rel(mask)
    if not split_files:
        return out
    with open(split_files[0]) as fh:
        j = json.load(fh)
    for key, dst in [
        ("train_gene_idx", "n_train_genes"),
        ("val_gene_idx", "n_validation_genes"),
        ("test_gene_idx", "n_test_genes"),
    ]:
        if key in j:
            out[dst] = len(j[key])
    return out


def manifest_info(dataset_id: str) -> dict[str, Any]:
    p = MANIFESTS.get(dataset_id)
    info = {"manifest_source": rel(p) if p and p.exists() else NA}
    if not p or not p.exists():
        return info
    try:
        j = json.load(open(p))
    except Exception:
        return info
    for k in [
        "shared_genes",
        "shared_genes_effective",
        "spatial_cells",
        "n_spots",
        "spatial_spots",
        "n_scrna_cells",
        "scrna_cells",
        "st_source",
        "scrna_source",
        "scrna_reference_interpretation",
    ]:
        if k in j:
            info[k] = j[k]
    return info


def build_s1() -> pd.DataFrame:
    prior = read_csv(SOURCES["s1_prior"])
    prior_map = {r["dataset"]: r for _, r in prior.iterrows()} if not prior.empty else {}
    align = read_csv(SOURCES["label_alignment"])

    dataset_notes = {
        "Vis9A": {
            "ST_platform": "Visium",
            "measurement_type": "sequencing-based ST",
            "spatial_unit_type": "spot",
            "ST_data_source": "GSE161318 Vis9A_D7 processed ST matrix",
            "ST_source_note": "SpaIM-compatible effective 4,470-gene panel used for complete method comparison.",
            "scRNA_reference": "matched D7 scRNA reference in processed dataset",
            "scRNA_reference_note": "Processed paired scRNA reference; exact source recorded in prior provenance table.",
            "preprocessing_summary": "Gene symbols deduplicated/intersected; SpaIM-compatible effective panel after min-cell filtering.",
            "exclusion_or_caveat": "Original GEO metadata row count differs from processed matrix; theta_* columns are proportions, not hard labels.",
        },
        "HBC": {
            "ST_platform": "Visium",
            "measurement_type": "sequencing-based ST",
            "spatial_unit_type": "spot",
            "ST_data_source": "HBC Visium section 1142243F",
            "ST_source_note": "Human breast cancer section under shared-gene strict gene5cv evaluation.",
            "scRNA_reference": "CID3586 scRNA",
            "scRNA_reference_note": "Human breast cancer scRNA reference.",
            "preprocessing_summary": "Shared 16,112-gene matrix with frozen strict whole-gene five-fold splits.",
            "exclusion_or_caveat": "Pathology Classification metadata exists, but current processed matrices do not retain strict barcode/coordinate alignment certification.",
        },
        "MHM": {
            "ST_platform": "Visium",
            "measurement_type": "sequencing-based ST",
            "spatial_unit_type": "spot",
            "ST_data_source": "processed MHM spatial matrix",
            "ST_source_note": "Mouse hypothalamus / brain sequencing-based ST benchmark.",
            "scRNA_reference": "matched mouse brain scRNA reference",
            "scRNA_reference_note": "Processed matched scRNA reference.",
            "preprocessing_summary": "Shared 14,780-gene matrix with frozen strict whole-gene five-fold splits.",
            "exclusion_or_caveat": "No curated MHM hard region/cell-type label found locally; evaluator-derived clusters remain weak/exploratory.",
        },
        "seqFISH+ cortex/SVZ": {
            "ST_platform": "seqFISH+",
            "measurement_type": "image-based spatial transcriptomics",
            "spatial_unit_type": "cell",
            "ST_data_source": "seqFISH+ mouse cortex/SVZ (Eng et al. 2019; GiottoData/drieslab seqfish_SS_cortex)",
            "ST_source_note": "Final short display name: seqFISH+ cortex/SVZ.",
            "scRNA_reference": "local Zeisel SScortex scRNA fallback",
            "scRNA_reference_note": "Local Zeisel SScortex fallback, not a dedicated broad cortex+SVZ reference from the original paper.",
            "preprocessing_summary": "Standardized 10,000 shared-gene panel with frozen strict whole-gene five-fold splits; GeneSPT uses LCR/validation-selected final readout.",
            "exclusion_or_caveat": "Reference pairing is practical fallback rather than original paired broad nervous-system reference.",
        },
        "MHPR/MERFISH": {
            "ST_platform": "MERFISH",
            "measurement_type": "image-based spatial transcriptomics",
            "spatial_unit_type": "cell",
            "ST_data_source": "Moffitt/Bambah-Mukku MERFISH hypothalamic preoptic region",
            "ST_source_note": "Current 154-gene panel used under frozen strict whole-gene five-fold masks.",
            "scRNA_reference": "matched hypothalamus scRNA matrix",
            "scRNA_reference_note": "Processed scRNA reference used for cross-platform prediction.",
            "preprocessing_summary": "MERFISH panel aligned with scRNA reference; frozen gene5cv splits use train/validation/test genes by whole gene.",
            "exclusion_or_caveat": "Original author Cell_class and Neuron_cluster_ID labels are matched exactly for downstream audit.",
        },
        "MVC/STARmap": {
            "ST_platform": "STARmap",
            "measurement_type": "image-based spatial transcriptomics",
            "spatial_unit_type": "cell",
            "ST_data_source": "STARmap mouse visual cortex",
            "ST_source_note": "Shared 981-gene panel.",
            "scRNA_reference": "visual cortex scRNA reference",
            "scRNA_reference_note": "Visual cortex scRNA reference matched to STARmap panel.",
            "preprocessing_summary": "Shared 981-gene matrix with frozen strict whole-gene five-fold splits.",
            "exclusion_or_caveat": "Downloaded STARmap annotation matches 1,207/1,549 cells; complete-dataset label use remains partial/exploratory.",
        },
    }

    rows = []
    for display, dataset_id, group in FINAL_DATASETS:
        base = prior_map.get(display, {})
        manifest = manifest_info(dataset_id)
        splits = split_counts(dataset_id)
        notes = dataset_notes[display]
        align_sub = align[align["dataset"].eq(display)] if not align.empty else pd.DataFrame()
        label_caveat = "; ".join(
            str(x) for x in align_sub.get("notes", pd.Series(dtype=str)).dropna().unique()[:3]
        )
        n_shared = (
            manifest.get("shared_genes_effective")
            or manifest.get("shared_genes")
            or base.get("shared genes", NA)
            or NA
        )
        n_units = (
            manifest.get("spatial_cells")
            or manifest.get("n_spots")
            or manifest.get("spatial_spots")
            or base.get("spots/cells", NA)
            or NA
        )
        rows.append(
            {
                "dataset_display_name": display,
                "dataset_id": dataset_id,
                "benchmark_group": group,
                "ST_platform": notes["ST_platform"],
                "measurement_type": notes["measurement_type"],
                "spatial_unit_type": notes["spatial_unit_type"],
                "n_units": n_units,
                "n_shared_genes_or_panel_genes": n_shared,
                "n_train_genes": splits["n_train_genes"],
                "n_validation_genes": splits["n_validation_genes"],
                "n_test_genes": splits["n_test_genes"],
                "n_folds": splits["n_folds"],
                "ST_data_source": notes["ST_data_source"],
                "ST_source_note": notes["ST_source_note"],
                "scRNA_reference": notes["scRNA_reference"],
                "scRNA_reference_note": notes["scRNA_reference_note"],
                "preprocessing_summary": notes["preprocessing_summary"],
                "gene_split_policy": "Strict whole-gene holdout; train/validation/test gene sets are frozen per fold.",
                "frozen_split_status": "frozen" if splits["n_folds"] == 5 else "not fully verified",
                "test_gene_use_policy": "Held-out test-gene ground truth used only for evaluation; not used for training/model selection.",
                "exclusion_or_caveat": notes["exclusion_or_caveat"],
                "main_text_display_name": display,
                "split_source": splits["split_source"],
                "manifest_source": manifest.get("manifest_source", NA),
                "label_alignment_caveat_source": first_existing([SOURCES["label_alignment"]]),
                "label_alignment_caveat": label_caveat or notes["exclusion_or_caveat"],
            }
        )
    return pd.DataFrame(rows)


def rank_with_direction(sub: pd.DataFrame, metric: str, higher: bool) -> pd.Series:
    return sub[metric].rank(ascending=not higher, method="min").astype("Int64")


def build_s2() -> pd.DataFrame:
    raw = read_csv(SOURCES["s2_raw"])
    if raw.empty:
        raise FileNotFoundError(SOURCES["s2_raw"])
    rows = []
    for dataset, sub in raw.groupby("dataset", sort=False):
        sub = sub[sub["method"].isin(METHODS)].copy()
        ranks = {
            "SPCC": rank_with_direction(sub, "SPCC_mean", True),
            "RMSE": rank_with_direction(sub, "RMSE_mean", False),
            "JS": rank_with_direction(sub, "JS_mean", False),
            "SSIM": rank_with_direction(sub, "SSIM_mean", True),
        }
        id_map = {d: i for d, i, _ in FINAL_DATASETS}
        group_map = {d: g for d, _, g in FINAL_DATASETS}
        for idx, r in sub.iterrows():
            method = r["method"]
            final_readout = "external baseline"
            if method == "GeneSPT":
                final_readout = "LCR / validation-selected final readout" if dataset == "seqFISH+ cortex/SVZ" else "final manuscript GeneSPT readout"
            rows.append(
                {
                    "dataset_display_name": dataset,
                    "dataset_id": id_map.get(dataset, NA),
                    "benchmark_group": group_map.get(dataset, r.get("role", NA)),
                    "method": method,
                    "method_type": "GeneSPT" if method == "GeneSPT" else "external baseline",
                    "availability_status": r.get("status", "complete"),
                    "folds_completed": r.get("folds", NA),
                    "SPCC_mean": r.get("SPCC_mean", np.nan),
                    "SPCC_std": np.nan,
                    "RMSE_mean": r.get("RMSE_mean", np.nan),
                    "RMSE_std": np.nan,
                    "JS_or_JSD_mean": r.get("JS_mean", np.nan),
                    "JS_or_JSD_std": np.nan,
                    "raw_SSIM_mean": r.get("SSIM_mean", np.nan),
                    "raw_SSIM_std": np.nan,
                    "rank_SPCC_if_available": int(ranks["SPCC"].loc[idx]),
                    "rank_RMSE_if_available": int(ranks["RMSE"].loc[idx]),
                    "rank_JS_if_available": int(ranks["JS"].loc[idx]),
                    "rank_raw_SSIM_if_available": int(ranks["SSIM"].loc[idx]),
                    "final_readout_or_calibration": final_readout,
                    "prediction_source": "Saved final prediction matrices / final source table; no rerun during supplyment assembly.",
                    "evaluator_source": rel(SOURCES["s2_raw"]),
                    "notes": "raw SSIM_mean used; ten-times-scaled SSIM excluded. Missing std is NA because final raw source stores five-fold means/status only.",
                }
            )
    return pd.DataFrame(rows)


def build_s3() -> pd.DataFrame:
    fig3 = read_csv(SOURCES["figure3_source"])
    if fig3.empty:
        raise FileNotFoundError(SOURCES["figure3_source"])
    rows = []
    panel_group = {
        "A": "descriptor_controls",
        "B": "GeneSPT-GC_vs_full_GeneSPT",
        "C": "PSP_controls",
    }
    for _, r in fig3.iterrows():
        panel = r.get("panel", NA)
        metric = r.get("metric", NA)
        if panel == "B":
            direction = "positive delta favors full GeneSPT"
        elif panel == "C":
            direction = "positive delta favors PSP variant over GeneSPT-GC"
        elif metric in ["SPCC", "SSIM", "raw SSIM"]:
            direction = "higher is better"
        elif metric in ["RMSE", "JS/JSD", "JS"]:
            direction = "lower is better"
        else:
            direction = NA
        rows.append(
            {
                "analysis_group": panel_group.get(panel, "mechanism_source"),
                "panel_or_source": f"Figure 3 Panel {panel}",
                "dataset": r.get("dataset", NA),
                "fold_or_summary": "five-fold mean",
                "model_or_control": r.get("setting", r.get("control", NA)),
                "metric": metric,
                "value": r.get("value", np.nan),
                "mean": r.get("value", np.nan),
                "std": r.get("std", np.nan),
                "baseline_value": r.get("baseline_value", np.nan),
                "delta": r.get("delta", np.nan),
                "delta_definition": r.get("delta_definition", NA),
                "direction": direction,
                "n_folds": r.get("n_folds", NA),
                "source_file": r.get("source_file", rel(SOURCES["figure3_source"])),
                "notes": r.get("note", ""),
            }
        )
    return pd.DataFrame(rows)


def figure6_panel_rows(fig6: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Panel B: already compact metric rows.
    b = fig6[fig6["panel"].eq("B")].copy()
    for _, r in b.iterrows():
        rows.append(
            {
                "analysis_type": "Figure 6 Panel B differential signal",
                "dataset": r.get("dataset", NA),
                "dataset_id": "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000",
                "label_column": r.get("label_name", r.get("label_source", NA)),
                "label_quality": r.get("label_quality", NA),
                "label_scope": r.get("label_source", NA),
                "n_units_total": r.get("n_cells_total", np.nan),
                "n_units_with_label": r.get("n_cells_matched", r.get("n_labeled_cells", np.nan)),
                "pipeline": r.get("pipeline", NA),
                "fold_or_summary": r.get("fold", r.get("representative_fold", NA)),
                "method": r.get("method", r.get("display_panel", NA)),
                "metric": r.get("metric", NA),
                "value": r.get("value", np.nan),
                "rank_among_imputation_methods": r.get("rank_among_imputation_methods", NA),
                "rank_including_references": r.get("rank_including_observed_upper", NA),
                "direction": r.get("metric_direction", NA),
                "source_file": rel(SOURCES["figure6_source"]),
                "interpretation": f"Best external for metric: {r.get('best_external_for_metric', NA)}",
                "caveat": "Representative downstream analysis; not used to claim universal downstream clustering improvement.",
            }
        )

    # Panel C: summarize the fixed-coordinate UMAP setup instead of writing one row per cell.
    c = fig6[fig6["panel"].eq("C")].copy()
    if not c.empty:
        summary = c.groupby(["dataset", "display_panel"], dropna=False).agg(
            n_units_total=("n_cells_total", "max"),
            n_units_with_label=("n_cells_matched", "max"),
            representative_fold=("representative_fold", "first"),
        ).reset_index()
        for _, r in summary.iterrows():
            rows.append(
                {
                    "analysis_type": "Figure 6 Panel C MHPR fixed-coordinate UMAP setup",
                    "dataset": r["dataset"],
                    "dataset_id": "MHPR_current_panel",
                    "label_column": "Cell_class",
                    "label_quality": "curated_cell_class",
                    "label_scope": "4975/4975 matched",
                    "n_units_total": r["n_units_total"],
                    "n_units_with_label": r["n_units_with_label"],
                    "pipeline": "fixed-coordinate UMAP from measured expression; fold0 Louvain colors majority-mapped to Cell_class",
                    "fold_or_summary": f"representative fold{r['representative_fold']} qualitative visualization",
                    "method": r["display_panel"],
                    "metric": "UMAP setup",
                    "value": np.nan,
                    "rank_among_imputation_methods": NA,
                    "rank_including_references": NA,
                    "direction": "qualitative only",
                    "source_file": rel(SOURCES["figure6_source"]),
                    "interpretation": "Measured data, GeneSPT and SpaGE panels use the same UMAP coordinate layout.",
                    "caveat": "UMAP was not used for model selection and should not be interpreted as quantitative proof of clustering improvement.",
                }
            )

    # Panel D: store rho/MAE summaries, not all scatter points.
    d = fig6[fig6["panel"].eq("D")].copy()
    d = d.dropna(subset=["true_cell_type_effect", "predicted_cell_type_effect", "method"])
    for method, sub in d.groupby("method"):
        rho = sub["true_cell_type_effect"].corr(sub["predicted_cell_type_effect"], method="spearman")
        mae = (sub["true_cell_type_effect"] - sub["predicted_cell_type_effect"]).abs().mean()
        for metric, value, direction in [
            ("held-out cell-type effect Spearman rho", rho, "higher is better"),
            ("held-out cell-type effect MAE", mae, "lower is better"),
        ]:
            rows.append(
                {
                    "analysis_type": "Figure 6 Panel D held-out cell-type effect recovery",
                    "dataset": "seqFISH+ cortex/SVZ",
                    "dataset_id": "seqFISH_plus_cortex_svz_zeisel_sccortex_ref_shared10000",
                    "label_column": "cell_types",
                    "label_quality": "curated_cell_type",
                    "label_scope": "913/913 matched",
                    "n_units_total": 913,
                    "n_units_with_label": 913,
                    "pipeline": "cell-type by held-out-gene one-vs-rest effect scatter summary",
                    "fold_or_summary": "all plotted cell-type-gene pairs",
                    "method": method,
                    "metric": metric,
                    "value": value,
                    "rank_among_imputation_methods": NA,
                    "rank_including_references": NA,
                    "direction": direction,
                    "source_file": rel(SOURCES["figure6_source"]),
                    "interpretation": "Summary of Panel D scatter source; full point-level source remains in Figure 6 source CSV.",
                    "caveat": "Differential-effect recovery is representative downstream analysis, not universal clustering improvement.",
                }
            )

    # Panel E: one row per method and metric.
    e = fig6[fig6["panel"].eq("E")].copy()
    for _, r in e.iterrows():
        for col, label in [("ARI", "ARI"), ("AMI", "AMI"), ("homogeneity", "Homogeneity"), ("NMI", "NMI")]:
            if col not in e.columns or pd.isna(r.get(col)):
                continue
            rows.append(
                {
                    "analysis_type": "Figure 6 Panel E MHPR Cell_class clustering",
                    "dataset": r.get("dataset", "MHPR/MERFISH"),
                    "dataset_id": "MHPR_current_panel",
                    "label_column": "Cell_class",
                    "label_quality": r.get("label_quality", "curated_cell_class"),
                    "label_scope": "4975/4975 matched",
                    "n_units_total": 4975,
                    "n_units_with_label": r.get("n_labeled_cells", 4975),
                    "pipeline": r.get("pipeline", NA),
                    "fold_or_summary": "five-fold mean",
                    "method": r.get("method", NA),
                    "metric": label,
                    "value": r.get(col),
                    "rank_among_imputation_methods": r.get("rank_among_imputation_methods", NA),
                    "rank_including_references": r.get("rank_including_observed_upper", NA),
                    "direction": "higher is better",
                    "source_file": rel(SOURCES["figure6_source"]),
                    "interpretation": r.get("interpretation", ""),
                    "caveat": "MHPR Cell_class clustering sensitivity; not used to claim universal downstream clustering improvement.",
                }
            )
    return pd.DataFrame(rows)


def label_provenance_rows() -> pd.DataFrame:
    align = read_csv(SOURCES["label_alignment"])
    rows = []
    if align.empty:
        return pd.DataFrame(rows)
    final_names = {d for d, _, _ in FINAL_DATASETS}
    for _, r in align[align["dataset"].isin(final_names)].iterrows():
        rows.append(
            {
                "analysis_type": "label provenance and alignment",
                "dataset": r.get("dataset", NA),
                "dataset_id": NA,
                "label_column": r.get("label_column", NA),
                "label_quality": r.get("label_quality", NA),
                "label_scope": r.get("usable_for_ARI_NMI", NA),
                "n_units_total": r.get("n_matrix_rows", np.nan),
                "n_units_with_label": r.get("n_matched", np.nan),
                "pipeline": r.get("alignment_method", NA),
                "fold_or_summary": "label alignment audit",
                "method": NA,
                "metric": "match_rate",
                "value": r.get("match_rate", np.nan),
                "rank_among_imputation_methods": NA,
                "rank_including_references": NA,
                "direction": "higher is better",
                "source_file": rel(SOURCES["label_alignment"]),
                "interpretation": f"Label classes: {r.get('n_label_classes', NA)}; largest class fraction: {r.get('largest_class_fraction', NA)}",
                "caveat": r.get("notes", NA),
            }
        )
    return pd.DataFrame(rows)


def downstream_summary_rows() -> pd.DataFrame:
    rows = []
    source_specs = [
        ("seqFISH+ annotation summary", SOURCES["downstream_seqfish"]),
        ("MVC downstream topology sensitivity", SOURCES["downstream_mvc"]),
        ("upgraded-label KMeans", SOURCES["upgraded_kmeans"]),
        ("upgraded-label Leiden default", SOURCES["upgraded_leiden_default"]),
        ("upgraded-label Leiden cluster-count matched", SOURCES["upgraded_leiden_matched"]),
    ]
    for analysis_name, path in source_specs:
        df = read_csv(path)
        if df.empty:
            continue
        for _, r in df.iterrows():
            dataset = r.get("dataset", "seqFISH+ cortex/SVZ" if "seqFISH" in analysis_name else NA)
            for metric_col in [
                "ARI",
                "NMI",
                "AMI",
                "homogeneity",
                "homogeneity_mean",
                "group_effect_spearman",
                "group_mean_MAE",
                "top20_marker_overlap",
                "top50_marker_overlap",
            ]:
                if metric_col not in df.columns or pd.isna(r.get(metric_col)):
                    continue
                metric_name = metric_col.replace("_mean", "")
                rows.append(
                    {
                        "analysis_type": analysis_name,
                        "dataset": dataset,
                        "dataset_id": NA,
                        "label_column": r.get("label_column", "cell_types" if "seqFISH" in analysis_name else r.get("label_source", NA)),
                        "label_quality": r.get("label_quality", "curated_cell_type" if "seqFISH" in analysis_name else NA),
                        "label_scope": r.get("label_scope", NA),
                        "n_units_total": r.get("n_units", np.nan),
                        "n_units_with_label": r.get("n_labeled_cells", np.nan),
                        "pipeline": r.get("pipeline", analysis_name),
                        "fold_or_summary": f"{r.get('n_folds', 'NA')}-fold mean" if "n_folds" in df.columns else "summary",
                        "method": r.get("method", NA),
                        "metric": "Homogeneity" if metric_name == "homogeneity" else metric_name,
                        "value": r.get(metric_col),
                        "rank_among_imputation_methods": r.get("rank_among_imputation_methods", NA),
                        "rank_including_references": r.get("rank_including_observed_upper", NA),
                        "direction": "lower is better" if metric_name == "group_mean_MAE" else "higher is better",
                        "source_file": rel(path),
                        "interpretation": r.get("interpretation", ""),
                        "caveat": "Observed-only and Full-ST upper, when present, are references and not imputation methods. Downstream results are exploratory/sensitivity analyses.",
                    }
                )
    return pd.DataFrame(rows)


def build_s4() -> pd.DataFrame:
    fig6 = read_csv(SOURCES["figure6_source"], low_memory=False)
    parts = [label_provenance_rows()]
    if not fig6.empty:
        parts.append(figure6_panel_rows(fig6))
    parts.append(downstream_summary_rows())
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    df = df.replace({np.nan: NA})
    df.to_csv(path, index=False)
    return path


def write_markdown_table(df: pd.DataFrame, path: Path, title: str, max_rows: int = 30) -> None:
    sample = df.head(max_rows).replace({np.nan: NA})
    more = "" if len(df) <= max_rows else f"\n\nShowing first {max_rows} of {len(df)} rows; full table is in CSV/XLSX.\n"
    path.write_text(f"# {title}\n\n{sample.to_markdown(index=False)}{more}\n", encoding="utf-8")


def write_workbook(tables: dict[str, pd.DataFrame]) -> Path:
    xlsx = OUT / "Supplementary_Tables_S1_to_S4.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            clean = df.replace({np.nan: NA})
            clean.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.book[sheet]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = 0
                letter = col[0].column_letter
                for cell in col[:200]:
                    max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
                ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 55)
    return xlsx


def main() -> None:
    for old in OUT.iterdir():
        if old.is_file():
            old.unlink()

    s1 = build_s1()
    s2 = build_s2()
    s3 = build_s3()
    s4 = build_s4()

    csv_paths = {
        "Supplementary_Table_S1_dataset_provenance_preprocessing_frozen_splits.csv": write_csv(
            s1, "Supplementary_Table_S1_dataset_provenance_preprocessing_frozen_splits.csv"
        ),
        "Supplementary_Table_S2_full_benchmark_metrics_method_availability.csv": write_csv(
            s2, "Supplementary_Table_S2_full_benchmark_metrics_method_availability.csv"
        ),
        "Supplementary_Table_S3_ablation_mechanism_controls.csv": write_csv(
            s3, "Supplementary_Table_S3_ablation_mechanism_controls.csv"
        ),
        "Supplementary_Table_S4_downstream_sensitivity_label_provenance.csv": write_csv(
            s4, "Supplementary_Table_S4_downstream_sensitivity_label_provenance.csv"
        ),
    }

    workbook = write_workbook(
        {
            "S1_dataset_provenance": s1,
            "S2_benchmark_metrics": s2,
            "S3_ablation_controls": s3,
            "S4_downstream_label": s4,
        }
    )

    readme = OUT / "README.md"
    readme.write_text(
        "# GeneSPT Supplementary Tables\n\n"
        "This supplyment folder contains four core supplementary tables for the GeneSPT manuscript.\n\n"
        "- Supplementary Table S1: Dataset provenance, preprocessing and frozen split summary.\n"
        "- Supplementary Table S2: Full benchmark metrics and method availability across all final datasets.\n"
        "- Supplementary Table S3: Full ablation and mechanism-control results.\n"
        "- Supplementary Table S4: Downstream sensitivity and label provenance audit.\n\n"
        "No supplementary figures are included in this folder. Metrics and annotations were collected from saved final outputs, central evaluator tables, source CSVs and audit files. Raw SSIM (`SSIM_mean`) is used; ten-times-scaled SSIM is not used. Observed-only and Full-ST upper are treated as reference settings where applicable, not as imputation methods. Downstream labels are categorized by provenance, including curated, author-matched, partial, topology-reference and unavailable/weak labels.\n\n"
        "No models were rerun, no external baselines were rerun, and no prediction matrices or manuscript files were modified during this supplyment assembly.\n",
        encoding="utf-8",
    )

    manifest_rows = []
    descriptions = {
        "Supplementary_Table_S1_dataset_provenance_preprocessing_frozen_splits.csv": "Dataset provenance, preprocessing and frozen split summary.",
        "Supplementary_Table_S2_full_benchmark_metrics_method_availability.csv": "Full raw benchmark metrics and method availability across final datasets.",
        "Supplementary_Table_S3_ablation_mechanism_controls.csv": "Descriptor, GeneSPT-GC/full GeneSPT and PSP mechanism-control source values.",
        "Supplementary_Table_S4_downstream_sensitivity_label_provenance.csv": "Downstream sensitivity, Figure 6 source summary and label provenance audit.",
        "Supplementary_Tables_S1_to_S4.xlsx": "Excel workbook containing S1-S4 as separate sheets.",
        "README.md": "Supplyment folder readme.",
        "supplyment_changelog.md": "Assembly changelog and source paths.",
        "manuscript_supplement_reference_update_note.md": "Recommended manuscript supplement reference renumbering note.",
        "quality_check_report.md": "Post-generation quality checks.",
    }
    source_inputs = {
        "Supplementary_Table_S1_dataset_provenance_preprocessing_frozen_splits.csv": first_existing(
            [SOURCES["s1_prior"], SOURCES["label_alignment"], *MASK_DIRS.values(), *MANIFESTS.values()]
        ),
        "Supplementary_Table_S2_full_benchmark_metrics_method_availability.csv": rel(SOURCES["s2_raw"]),
        "Supplementary_Table_S3_ablation_mechanism_controls.csv": first_existing(
            [SOURCES["figure3_source"], SOURCES["figure3_audit"]]
        ),
        "Supplementary_Table_S4_downstream_sensitivity_label_provenance.csv": first_existing(
            [
                SOURCES["figure6_source"],
                SOURCES["label_alignment"],
                SOURCES["downstream_seqfish"],
                SOURCES["downstream_mvc"],
                SOURCES["upgraded_kmeans"],
                SOURCES["upgraded_leiden_default"],
                SOURCES["upgraded_leiden_matched"],
            ]
        ),
    }
    for file_name, description in descriptions.items():
        manifest_rows.append(
            {
                "file_name": file_name,
                "description": description,
                "format": Path(file_name).suffix.lstrip(".") or "directory note",
                "source_inputs": source_inputs.get(file_name, "Generated from S1-S4 tables and local audit metadata."),
                "created_by": "scripts/generate_supplyment_tables.py",
                "notes": "Final supplyment package; no supplementary figures.",
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT / "supplyment_manifest.csv", index=False)

    update_note = OUT / "manuscript_supplement_reference_update_note.md"
    update_note.write_text(
        "# Manuscript Supplement Reference Update Note\n\n"
        "Use the following final supplementary material numbering:\n\n"
        "- Supplementary Table S1: Dataset provenance, preprocessing and frozen split summary.\n"
        "- Supplementary Table S2: Full benchmark metrics and method availability across all final datasets.\n"
        "- Supplementary Table S3: Full ablation and mechanism-control results.\n"
        "- Supplementary Table S4: Downstream sensitivity and label provenance audit.\n\n"
        "Remove references to Supplementary Figure S1-S3. The previous method-availability table is merged into Supplementary Table S2. The previous downstream sensitivity / label provenance material is consolidated into Supplementary Table S4. If the manuscript still refers to Supplementary Methods for dataset/evaluator details, update those references to Supplementary Table S1-S2 or to the code repository / data availability statement.\n",
        encoding="utf-8",
    )

    source_lines = "\n".join(f"- {k}: `{rel(v)}`" for k, v in SOURCES.items() if v.exists())
    changelog = OUT / "supplyment_changelog.md"
    changelog.write_text(
        f"# Supplyment Changelog\n\n"
        "- No GeneSPT model rerun was performed.\n"
        "- No external baseline rerun was performed.\n"
        "- No prediction matrices were modified.\n"
        "- No manuscript docx file was modified.\n"
        "- No supplementary figures were generated.\n"
        "- Supplementary materials were reduced to four core tables: S1-S4.\n"
        "- Raw SSIM (`SSIM_mean`) was used; ten-times-scaled SSIM was not used.\n"
        "- Method availability was included in Supplementary Table S2.\n"
        "- Descriptor and PSP ablation tables were merged into Supplementary Table S3 using the redesigned Figure 3 source.\n"
        "- Downstream sensitivity and label provenance were merged into Supplementary Table S4.\n"
        "- LohoffE1Z2 and stDiff were not included in the final benchmark supplement tables.\n\n"
        f"## Source Paths Used\n\n{source_lines}\n\n"
        "## Missing or Unresolved Items\n\n"
        "- Five-fold standard deviations are not available in the final all-method raw metrics source used for S2; std fields are retained as NA rather than inferred.\n"
        "- Some label sources remain weak/partial by provenance; this is recorded in S1 and S4 caveat fields.\n",
        encoding="utf-8",
    )

    # Quality checks
    checks = []
    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add_check("all_output_csv_exist", all(p.exists() for p in csv_paths.values()), str(csv_paths))
    add_check("csv_nonempty", all(pd.read_csv(p).shape[0] > 0 for p in csv_paths.values()), "All four CSV tables have at least one row.")
    add_check("excel_exists", workbook.exists(), rel(workbook))
    try:
        xl = pd.ExcelFile(workbook)
        add_check("excel_has_four_sheets", set(xl.sheet_names) == {"S1_dataset_provenance", "S2_benchmark_metrics", "S3_ablation_controls", "S4_downstream_label"}, str(xl.sheet_names))
    except Exception as e:
        add_check("excel_has_four_sheets", False, str(e))
    add_check("s1_has_six_final_datasets", set(s1["dataset_display_name"]) == {d for d, _, _ in FINAL_DATASETS}, str(sorted(s1["dataset_display_name"].unique())))
    add_check("s2_has_six_final_datasets", set(s2["dataset_display_name"]) == {d for d, _, _ in FINAL_DATASETS}, str(sorted(s2["dataset_display_name"].unique())))
    add_check("s2_has_final_methods", set(s2["method"]) == set(METHODS), str(sorted(s2["method"].unique())))
    add_check("s3_has_descriptor_controls", (s3["analysis_group"] == "descriptor_controls").any(), "Descriptor controls present.")
    add_check("s3_has_psp_controls", (s3["analysis_group"] == "PSP_controls").any(), "PSP controls present.")
    add_check("s4_has_label_provenance", (s4["analysis_type"] == "label provenance and alignment").any(), "Label provenance rows present.")
    add_check("s4_has_figure6_summary", s4["analysis_type"].astype(str).str.contains("Figure 6").any(), "Figure 6 rows present.")
    generated_files = [p.name for p in OUT.iterdir() if p.is_file()]
    add_check("no_supplementary_figures", not any(p.lower().endswith((".png", ".pdf")) and "figure" in p.lower() for p in generated_files), str(generated_files))
    add_check("no_old_s5_s6_final_files", not any("S5" in p or "S6" in p or "s5" in p or "s6" in p for p in generated_files), str(generated_files))
    text_join = "\n".join(
        p.read_text(errors="ignore") if p.suffix in {".md", ".csv"} else ""
        for p in OUT.iterdir()
        if p.is_file()
    )
    add_check("no_scaled_ssim_string", "SSIMx10_mean" not in text_join and "SSIM×10" not in text_join, "No forbidden scaled-SSIM strings in final text/CSV outputs.")

    qdf = pd.DataFrame(checks)
    header = "| check | passed | detail |\n|---|---|---|\n"
    body = "\n".join(
        f"| {row['check']} | {row['passed']} | {str(row['detail']).replace('|', '/')} |"
        for _, row in qdf.iterrows()
    )
    qtext = "# Quality Check Report\n\n" + header + body + "\n"
    if not qdf["passed"].all():
        qtext += "\nAt least one check failed; review details above.\n"
    (OUT / "quality_check_report.md").write_text(qtext, encoding="utf-8")

    print(f"Wrote supplyment package to {OUT}")
    print(qdf.to_string(index=False))


if __name__ == "__main__":
    main()
