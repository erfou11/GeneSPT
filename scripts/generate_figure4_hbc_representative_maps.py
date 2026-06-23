#!/usr/bin/env python3
"""Generate manuscript Figure 4 HBC held-out gene maps.

The figure is intentionally map-only: four representative HBC held-out genes
across ground truth, GeneSPT, and all compared baselines. Panel labels do not
show SPCC values.
Each map is independently normalized for visualization to preserve the accepted
visual appearance of the earlier CD74/TIMP1 figure; quantitative metrics remain
in the source CSV and centralized evaluator outputs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HBC_shared16112"
INFO = ROOT / "results" / "imformation"
OUT = ROOT.parent / "final_manuscript" / "figures"

COUNTS = DATA / "Spatial_count.txt"
COORDS = DATA / "Locations.txt"
GENE_LEVEL = INFO / "final_main_benchmark_recomputed_gene_level.csv"
SPAIM_METRICS = ROOT / "final_output" / "figure2_hbc_case_study_v4_source.csv"

GENESPT_ROOT = INFO / "final_readout_prediction_matrices" / "HBC_shared16112" / "TopoDiST-GC-PSP"
EXTERNAL_READOUT_ROOT = INFO / "final_readout_prediction_matrices" / "HBC_shared16112"
EXTERNAL_ROOTS = {
    "Tangram": EXTERNAL_READOUT_ROOT / "Tangram",
    "TransImp": EXTERNAL_READOUT_ROOT / "TransPA",
    "SpaGE": EXTERNAL_READOUT_ROOT / "SpaGE",
    "stPlus": EXTERNAL_READOUT_ROOT / "stPlus",
    "SpaIM": ROOT / "results" / "final_hbc_spaim_gene5cv_fullstrict",
}

MAIN_METHODS = ["Ground truth", "GeneSPT", "Tangram", "TransImp", "SpaIM", "SpaGE", "stPlus"]
SUPP_METHODS = MAIN_METHODS

FIXED_SELECTION = [
    {
        "gene": "CD74",
        "fold": 1,
        "pattern_type": "immune local enrichment",
        "reason": "immune-associated held-out gene with strong local enrichment and high spatial autocorrelation",
    },
    {
        "gene": "TIMP1",
        "fold": 4,
        "pattern_type": "stromal remodeling gradient",
        "reason": "matrix-remodeling/stromal-associated gene with broader spatial gradients",
    },
    {
        "gene": "COL1A1",
        "fold": 1,
        "pattern_type": "stromal ECM gradient",
        "reason": "stromal/ECM-associated held-out gene with visible broad-gradient structure and GeneSPT advantage over external baselines",
    },
    {
        "gene": "KRT14",
        "fold": 2,
        "pattern_type": "tumor epithelial enrichment",
        "reason": "basal/tumor-epithelial held-out gene with interpretable enrichment structure and non-top-ranked GeneSPT SPCC",
    },
]

PREFERRED_LOCAL = [
    {
        "gene": "SFRP2",
        "pattern_type": "CAF local enrichment",
        "reason": "stromal/CAF-associated held-out gene with a visible local-enrichment pattern and non-top-ranked GeneSPT SPCC",
    },
    {"gene": "CXCL14", "pattern_type": "local enrichment", "reason": "candidate hotspot-like held-out gene"},
    {"gene": "CCL21", "pattern_type": "immune local enrichment", "reason": "candidate immune hotspot-like held-out gene"},
    {"gene": "MS4A1", "pattern_type": "immune local enrichment", "reason": "candidate immune hotspot-like held-out gene"},
]

PREFERRED_BROAD = [
    {
        "gene": "KRT19",
        "pattern_type": "tumor epithelial gradient",
        "reason": "tumor/epithelial-associated held-out gene with broad expression and spatial-gradient structure; selected to avoid only top-ranked GeneSPT examples",
    },
    {"gene": "EPCAM", "pattern_type": "tumor epithelial / broad gradient", "reason": "candidate epithelial broad-gradient held-out gene"},
    {"gene": "KRT8", "pattern_type": "tumor epithelial / broad gradient", "reason": "candidate epithelial broad-gradient held-out gene"},
    {"gene": "COL1A1", "pattern_type": "stromal / ECM gradient", "reason": "candidate stromal/ECM broad-gradient held-out gene"},
    {"gene": "COL1A2", "pattern_type": "stromal / ECM gradient", "reason": "candidate stromal/ECM broad-gradient held-out gene"},
]


VIRIDIS = np.asarray(
    [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ],
    dtype=np.float32,
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_HEADER = font(25, True)
F_GENE = font(28, True)
F_ROLE = font(17, False)
F_SMALL = font(17, False)
F_PANEL = font(20, False)


def load_truth() -> tuple[np.ndarray, list[str], dict[str, int], pd.DataFrame]:
    counts = pd.read_csv(COUNTS, sep="\t", dtype=np.float32)
    genes = list(map(str, counts.columns))
    X = counts.to_numpy(np.float32)
    coords = pd.read_csv(COORDS, sep="\t")
    return X, genes, {g: i for i, g in enumerate(genes)}, coords


def load_spcc_table() -> pd.DataFrame:
    main = pd.read_csv(GENE_LEVEL, low_memory=False)
    main = main[main["dataset"].eq("HBC_shared16112")].copy()
    display = {
        "TopoDiST-GC-PSP": "GeneSPT",
        "Tangram": "Tangram",
        "TransPA": "TransImp",
        "SpaGE": "SpaGE",
        "stPlus": "stPlus",
    }
    main = main[main["method"].isin(display)].copy()
    main["method_display"] = main["method"].map(display)
    rows = main[["method_display", "fold", "gene_idx", "gene", "SPCC"]].copy()

    if SPAIM_METRICS.exists():
        spaim = pd.read_csv(SPAIM_METRICS, low_memory=False)
        spaim = spaim[
            spaim.get("panel", "").eq("D")
            & spaim.get("method_display", "").eq("SpaIM")
            & spaim.get("metric", "").eq("per-gene SPCC")
        ].copy()
        if not spaim.empty:
            spaim_rows = pd.DataFrame(
                {
                    "method_display": "SpaIM",
                    "fold": spaim["fold"].astype(int),
                    "gene_idx": spaim["gene_idx"].astype(int),
                    "gene": spaim["gene"].astype(str),
                    "SPCC": pd.to_numeric(spaim["value"], errors="coerce"),
                }
            )
            rows = pd.concat([rows, spaim_rows], ignore_index=True)
    return rows


def metric_value(metrics: pd.DataFrame, method: str, fold: int, gene_idx: int) -> float:
    sub = metrics[
        metrics["method_display"].eq(method)
        & metrics["fold"].astype(int).eq(int(fold))
        & metrics["gene_idx"].astype(int).eq(int(gene_idx))
    ]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub.iloc[0]["SPCC"], errors="coerce"))


def genespt_vector(fold: int, gene_idx: int) -> tuple[np.ndarray, str]:
    path = GENESPT_ROOT / f"fold{fold}" / "final_readout_prediction.npz"
    data = np.load(path)
    positions = {int(g): i for i, g in enumerate(data["test_idx"].astype(int))}
    pos = positions[int(gene_idx)]
    return data["pred_test"][:, pos].astype(np.float32), str(path)


def external_vector(method: str, fold: int, gene_idx: int) -> tuple[np.ndarray, str]:
    if method == "SpaIM":
        path = EXTERNAL_ROOTS[method] / f"fold{fold}" / "imputed_expression.npy"
        pred = np.load(path, mmap_mode="r")
        return np.asarray(pred[:, int(gene_idx)], dtype=np.float32), str(path)
    path = EXTERNAL_ROOTS[method] / f"fold{fold}" / "final_readout_prediction.npz"
    data = np.load(path)
    positions = {int(g): i for i, g in enumerate(data["test_idx"].astype(int))}
    pos = positions[int(gene_idx)]
    return data["pred_test"][:, pos].astype(np.float32), str(path)


def normalize_shared(arrays: list[np.ndarray]) -> tuple[list[np.ndarray], float, float]:
    pooled = np.concatenate([np.asarray(a, dtype=np.float32) for a in arrays])
    lo, hi = np.nanpercentile(pooled, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(pooled)), float(np.nanmax(pooled))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    norm = [np.clip((np.asarray(a, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32) for a in arrays]
    return norm, float(lo), float(hi)


def normalize_map(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = np.nanpercentile(arr, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return norm, float(lo), float(hi)


def build_metric_lookup(metrics: pd.DataFrame, method: str) -> pd.DataFrame:
    sub = metrics[metrics["method_display"].eq(method)].copy()
    return sub[["fold", "gene_idx", "gene", "SPCC"]].rename(columns={"SPCC": f"{method}_SPCC"})


def selection_candidates(X: np.ndarray, genes: list[str], metrics: pd.DataFrame) -> pd.DataFrame:
    genespt = build_metric_lookup(metrics, "GeneSPT")
    tangram = build_metric_lookup(metrics, "Tangram")
    merged = genespt.merge(tangram, on=["fold", "gene_idx", "gene"], how="inner")
    gene_idx = merged["gene_idx"].astype(int).to_numpy()
    values = X[:, gene_idx].astype(np.float32)
    merged["mean_expression"] = values.mean(axis=0)
    merged["detected_fraction"] = (values > 0).mean(axis=0)
    merged["spatial_variance"] = values.var(axis=0)
    merged["spatial_variance_rank"] = merged["spatial_variance"].rank(ascending=False, method="min").astype(int)
    merged["enrichment_ratio"] = np.nanpercentile(values, 99, axis=0) / np.clip(merged["mean_expression"].to_numpy(), 1e-6, None)
    merged["genespt_spcc_rank"] = merged["GeneSPT_SPCC"].rank(ascending=False, method="min").astype(int)
    return merged


def choose_preferred(
    candidates: pd.DataFrame,
    preferred: list[dict[str, str]],
    exclude: set[str],
    mode: str,
) -> dict[str, object]:
    usable = candidates[
        ~candidates["gene"].isin(exclude)
        & candidates["mean_expression"].gt(0.02)
        & candidates["detected_fraction"].gt(0.015)
        & candidates["spatial_variance_rank"].le(max(2500, int(len(candidates) * 0.35)))
    ].copy()
    if mode == "local":
        usable = usable[usable["enrichment_ratio"].gt(4.0)]
    else:
        usable = usable[usable["detected_fraction"].gt(0.08)]

    preferred_meta = {item["gene"]: item for item in preferred}
    pref = usable[usable["gene"].isin(preferred_meta)].copy()
    if not pref.empty:
        pref["preferred_order"] = pref["gene"].map({g: i for i, g in enumerate(preferred_meta)})
        row = pref.sort_values(["preferred_order", "genespt_spcc_rank"], ascending=[True, True]).iloc[0]
        meta = preferred_meta[str(row["gene"])]
    else:
        score = usable["enrichment_ratio"] if mode == "local" else usable["detected_fraction"] * usable["spatial_variance"]
        usable = usable.assign(selection_score=score)
        if usable.empty:
            raise RuntimeError(f"No usable {mode} candidate gene passed the selection filters")
        row = usable.sort_values(["selection_score", "GeneSPT_SPCC"], ascending=[False, False]).iloc[0]
        meta = {
            "gene": str(row["gene"]),
            "pattern_type": "local enrichment" if mode == "local" else "broad spatial gradient",
            "reason": f"automatically selected {mode} held-out gene passing expression, detection, and spatial-variance screens",
        }

    return {
        "gene": str(row["gene"]),
        "fold": int(row["fold"]),
        "pattern_type": meta["pattern_type"],
        "reason": meta["reason"],
    }


def select_genes(X: np.ndarray, genes: list[str], metrics: pd.DataFrame) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    candidates = selection_candidates(X, genes, metrics)
    selected = [dict(item) for item in FIXED_SELECTION]

    selected_rows = []
    for item in selected:
        row = candidates[candidates["gene"].eq(item["gene"])].iloc[0]
        selected_rows.append(
            {
                "gene": item["gene"],
                "fold_id": int(row["fold"]),
                "reason_for_selection": item["reason"],
                "pattern_type": item["pattern_type"],
                "GeneSPT_SPCC": float(row["GeneSPT_SPCC"]),
                "Tangram_SPCC": float(row["Tangram_SPCC"]),
                "mean_expression": float(row["mean_expression"]),
                "detected_fraction": float(row["detected_fraction"]),
                "spatial_variance_rank": int(row["spatial_variance_rank"]),
            }
        )
    selected_table = pd.DataFrame(selected_rows)

    top_candidates = candidates[
        candidates["mean_expression"].gt(0.02)
        & candidates["detected_fraction"].gt(0.015)
        & candidates["spatial_variance_rank"].le(max(2500, int(len(candidates) * 0.35)))
    ].sort_values(["spatial_variance_rank", "GeneSPT_SPCC"], ascending=[True, False]).head(10)

    return selected, selected_table, top_candidates


def viridis(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    x = v * (len(VIRIDIS) - 1)
    i0 = np.floor(x).astype(int)
    i1 = np.clip(i0 + 1, 0, len(VIRIDIS) - 1)
    t = (x - i0)[..., None]
    rgb = VIRIDIS[i0] * (1.0 - t) + VIRIDIS[i1] * t
    return rgb.astype(np.uint8)


def coordinate_pixels(coords: pd.DataFrame, size: int, pad: int = 18) -> tuple[np.ndarray, np.ndarray]:
    x = coords["x"].to_numpy(np.float32)
    y = coords["y"].to_numpy(np.float32)
    xpix = pad + (x - x.min()) / max(float(x.max() - x.min()), 1.0) * (size - 2 * pad)
    ypix = pad + (y - y.min()) / max(float(y.max() - y.min()), 1.0) * (size - 2 * pad)
    return xpix, ypix


def draw_map(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    size: int,
    coords_pix: tuple[np.ndarray, np.ndarray],
    values: np.ndarray,
    title: str,
    spcc: float | None,
    radius: int,
) -> None:
    xpix, ypix = coords_pix
    colors = viridis(values)
    order = np.argsort(values)
    for idx in order:
        cx = int(round(x0 + xpix[idx]))
        cy = int(round(y0 + ypix[idx]))
        c = tuple(int(v) for v in colors[idx])
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=c)
    draw.rectangle((x0, y0, x0 + size, y0 + size), outline=(228, 228, 228), width=1)
    if spcc is not None and math.isfinite(spcc):
        label = f"SPCC={spcc:.3f}"
        bbox = draw.textbbox((0, 0), label, font=F_PANEL)
        pad_x, pad_y = 8, 5
        tx = x0 + size - (bbox[2] - bbox[0]) - 2 * pad_x - 8
        ty = y0 + 8
        draw.rectangle(
            (tx, ty, tx + (bbox[2] - bbox[0]) + 2 * pad_x, ty + (bbox[3] - bbox[1]) + 2 * pad_y),
            fill=(255, 255, 255),
            outline=(230, 230, 230),
        )
        draw.text((tx + pad_x, ty + pad_y - 1), label, fill=(45, 45, 45), font=F_PANEL)


def build_rows(methods: list[str], selected: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    X, genes, gene_to_idx, coords = load_truth()
    metrics = load_spcc_table()
    rows: list[dict] = []
    source_records: list[dict] = []
    for item in selected:
        gene = item["gene"]
        fold = int(item["fold"])
        gene_idx = int(gene_to_idx[gene])
        arrays: dict[str, np.ndarray] = {"Ground truth": X[:, gene_idx].astype(np.float32)}
        sources: dict[str, str] = {"Ground truth": str(COUNTS)}
        spcc: dict[str, float | None] = {"Ground truth": None}
        for method in methods:
            if method == "Ground truth":
                continue
            if method == "GeneSPT":
                arrays[method], sources[method] = genespt_vector(fold, gene_idx)
            else:
                arrays[method], sources[method] = external_vector(method, fold, gene_idx)
            spcc[method] = metric_value(metrics, method, fold, gene_idx)
        norm_arrays: dict[str, np.ndarray] = {}
        norm_limits: dict[str, tuple[float, float]] = {}
        for method in methods:
            norm_arrays[method], lo, hi = normalize_map(arrays[method])
            norm_limits[method] = (lo, hi)
        rows.append(
            {
                **item,
                "gene_idx": gene_idx,
                "arrays": norm_arrays,
                "spcc": spcc,
                "visual_norm_limits": norm_limits,
            }
        )
        for method in methods:
            lo, hi = norm_limits[method]
            source_records.append(
                {
                    "figure": "Figure 4" if methods == MAIN_METHODS else "Supplementary Figure",
                    "gene": gene,
                    "fold": fold,
                    "gene_idx": gene_idx,
                    "role": item["pattern_type"],
                    "selection_reason": item["reason"],
                    "method": method,
                    "SPCC_original_scale": spcc.get(method),
                    "prediction_source": sources.get(method),
                    "truth_source": str(COUNTS),
                    "coordinate_source": str(COORDS),
                    "truth_transform": "raw count map for visualization, matching the earlier CD74/TIMP1 Figure 4 visual source",
                    "visualization_policy": "per-map 2nd-98th percentile clipping and min-max scaling to 0-1 for visualization only; SPCC values are from original-scale centralized evaluator outputs and are not printed in the figure",
                    "visual_norm_vmin": lo,
                    "visual_norm_vmax": hi,
                }
            )
    return rows, pd.DataFrame(source_records)


def render_grid(
    rows: list[dict],
    methods: list[str],
    out_png: Path,
    out_pdf: Path,
    title: str,
    map_size: int,
    label_w: int,
    col_gap: int,
    row_gap: int,
) -> None:
    top = 78
    left = 34
    right = 40
    bottom = 42
    width = left + label_w + len(methods) * map_size + (len(methods) - 1) * col_gap + right
    height = top + len(rows) * map_size + (len(rows) - 1) * row_gap + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    coords = pd.read_csv(COORDS, sep="\t")
    coords_pix = coordinate_pixels(coords, map_size)

    for c, method in enumerate(methods):
        x = left + label_w + c * (map_size + col_gap)
        bbox = draw.textbbox((0, 0), method, font=F_HEADER)
        draw.text((x + map_size / 2 - (bbox[2] - bbox[0]) / 2, 26), method, fill=(25, 25, 25), font=F_HEADER)

    for r, row in enumerate(rows):
        y = top + r * (map_size + row_gap)
        draw.text((left, y + map_size * 0.36), row["gene"], fill=(20, 20, 20), font=F_GENE)
        draw.text((left, y + map_size * 0.36 + 42), row["pattern_type"], fill=(88, 88, 88), font=F_ROLE)
        draw.text((left, y + map_size * 0.36 + 72), f"fold{row['fold']} test gene", fill=(105, 105, 105), font=F_SMALL)
        for c, method in enumerate(methods):
            x = left + label_w + c * (map_size + col_gap)
            draw_map(
                draw,
                x,
                y,
                map_size,
                coords_pix,
                row["arrays"][method],
                method,
                None,
                radius=max(2, int(round(map_size / 125))),
            )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, dpi=(300, 300))
    img.save(out_pdf, "PDF", resolution=300.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    X, genes, _, _ = load_truth()
    metrics = load_spcc_table()
    selected, selected_table, top_candidates = select_genes(X, genes, metrics)
    selected_table.to_csv(OUT / "figure4_selected_genes.csv", index=False)
    top_candidates.to_csv(OUT / "figure4_top10_candidate_genes.csv", index=False)

    main_rows, main_source = build_rows(MAIN_METHODS, selected)
    main_png = OUT / "figure4_hbc_representative_maps.png"
    main_pdf = OUT / "figure4_hbc_representative_maps.pdf"
    render_grid(
        main_rows,
        MAIN_METHODS,
        main_png,
        main_pdf,
        "HBC held-out gene maps across all methods",
        map_size=360,
        label_w=225,
        col_gap=22,
        row_gap=52,
    )
    main_source.to_csv(OUT / "figure4_hbc_representative_maps_source.csv", index=False)

    spcc_summary = main_source[main_source["method"].ne("Ground truth")].pivot_table(
        index="gene", columns="method", values="SPCC_original_scale", aggfunc="first"
    )
    spcc_summary.to_csv(OUT / "figure4_all_baseline_spcc_summary.csv")

    print(f"main_png={main_png}")
    print(f"main_pdf={main_pdf}")
    print(f"main_source={OUT / 'figure4_hbc_representative_maps_source.csv'}")
    print(f"selected_genes={OUT / 'figure4_selected_genes.csv'}")
    print(f"spcc_summary={OUT / 'figure4_all_baseline_spcc_summary.csv'}")


if __name__ == "__main__":
    main()
