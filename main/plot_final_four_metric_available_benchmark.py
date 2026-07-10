#!/usr/bin/env python3
"""Plot final available-dataset four-metric strict benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/workspace/GeneSPT")
INFO = ROOT / "results" / "imformation"
FIG_DIR = INFO / "final_manuscript_figures"
SUMMARY_PATH = INFO / "final_available_datasets_four_metric_summary.csv"
RANK_PATH = INFO / "final_available_datasets_four_metric_rank.csv"

METHOD_ORDER = [
    "GeneSPT-GC-PSP",
    "GC-MLP-PCA32-softplus",
    "SpaGE",
    "Tangram",
    "TransPA",
    "stPlus",
    "stDiff",
    "SpaIM",
]

METHOD_COLORS = {
    "GeneSPT-GC-PSP": "#0B6E69",
    "GC-MLP-PCA32-softplus": "#70A288",
    "SpaGE": "#7B8794",
    "Tangram": "#D8943C",
    "TransPA": "#7E62A3",
    "stPlus": "#4C78A8",
    "stDiff": "#B9412F",
    "SpaIM": "#C9A227",
}

DATASET_ORDER = ["Vis9A", "MHPR", "MVC", "MG"]
DATASET_TITLES = {
    "Vis9A": "Sequencing / 10X Visium",
    "MHPR": "Image-based / MERFISH",
    "MVC": "Image-based / STARmap",
    "MG": "Other public / seqFISH",
}


def clean_method(x: str) -> str:
    return {"GC-MLP-PCA32-softplus": "GC-MLP"}.get(x, x)


def plot_metric_grid(summary: pd.DataFrame) -> None:
    metrics = [
        ("SPCC_mean", "SPCC ↑", False),
        ("RMSE_mean", "RMSE ↓", True),
        ("JS_mean", "JS ↓", True),
        ("SSIM_mean", "SSIM ↑", False),
    ]
    fig, axes = plt.subplots(len(DATASET_ORDER), len(metrics), figsize=(16.4, 12.6), constrained_layout=True)
    for i, ds in enumerate(DATASET_ORDER):
        d0 = summary[summary["dataset_display"].eq(ds)].copy()
        d0["method"] = pd.Categorical(d0["method"], categories=METHOD_ORDER, ordered=True)
        d0 = d0.sort_values("method")
        for j, (metric, title, lower_better) in enumerate(metrics):
            ax = axes[i, j]
            d = d0[d0["status"].eq("complete")].dropna(subset=[metric]).copy()
            d = d.sort_values(metric, ascending=lower_better)
            labels = [clean_method(m) for m in d["method"].astype(str)]
            vals = d[metric].to_numpy(float)
            y = np.arange(len(d))
            ax.barh(
                y,
                vals,
                color=[METHOD_COLORS.get(m, "#999999") for m in d["method"].astype(str)],
                edgecolor="#222222",
                linewidth=0.45,
            )
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8.5)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_title(f"{ds} · {title}", fontsize=10.5, fontweight="bold")
            if j == 0:
                ax.set_ylabel(DATASET_TITLES.get(ds, ds), fontsize=10.5, fontweight="bold")
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            if vals.size:
                xmax = np.nanmax(vals)
                xmin = np.nanmin(vals)
                pad = (xmax - xmin) * 0.08 if xmax > xmin else abs(xmax) * 0.08 + 1e-3
                if lower_better:
                    ax.set_xlim(max(0.0, xmin - pad), xmax + pad)
                else:
                    ax.set_xlim(min(0.0, xmin - pad), xmax + pad)
                for yy, v in zip(y, vals):
                    ax.text(v, yy, f" {v:.3f}", va="center", fontsize=7.4)
    fig.suptitle(
        "Final strict whole-gene benchmark across available ST technologies",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "final_available_datasets_four_metric_benchmark.png", dpi=260)
    fig.savefig(FIG_DIR / "final_available_datasets_four_metric_benchmark.pdf")
    plt.close(fig)


def plot_rank_heatmap(summary: pd.DataFrame, ranks: pd.DataFrame) -> None:
    methods = METHOD_ORDER
    metric_order = ["SPCC", "RMSE", "JS", "SSIM"]
    rows: list[tuple[str, str]] = []
    for ds in DATASET_ORDER:
        available = summary[(summary["dataset_display"].eq(ds)) & (summary["status"].eq("complete"))]["method"].tolist()
        for method in methods:
            if method in available:
                rows.append((ds, method))
    arr = np.full((len(rows), len(metric_order)), np.nan)
    vals = np.full_like(arr, np.nan, dtype=float)
    for i, (ds, method) in enumerate(rows):
        for j, metric in enumerate(metric_order):
            sub = ranks[
                (ranks["dataset_display"].eq(ds))
                & (ranks["method"].eq(method))
                & (ranks["metric"].eq(metric))
            ]
            if sub.empty:
                continue
            arr[i, j] = float(sub["rank"].iloc[0])
            vals[i, j] = float(sub[f"{metric}_mean"].iloc[0])

    fig_h = max(7.0, 0.34 * len(rows) + 1.7)
    fig, ax = plt.subplots(figsize=(9.6, fig_h), constrained_layout=True)
    im = ax.imshow(arr, cmap="YlGn_r", vmin=1, vmax=8, aspect="auto")
    ax.set_xticks(range(len(metric_order)))
    ax.set_xticklabels(metric_order, fontsize=10.5, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{ds} · {clean_method(method)}" for ds, method in rows], fontsize=8.3)
    for i in range(len(rows)):
        for j in range(len(metric_order)):
            if np.isfinite(arr[i, j]):
                ax.text(
                    j,
                    i,
                    f"#{int(arr[i, j])}\n{vals[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.1,
                    fontweight="bold" if arr[i, j] == 1 else "normal",
                )
    ax.set_title("Rank heatmap, centrally recomputed four metrics", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("rank, lower is better")
    fig.savefig(FIG_DIR / "final_available_datasets_four_metric_rank_heatmap.png", dpi=260)
    fig.savefig(FIG_DIR / "final_available_datasets_four_metric_rank_heatmap.pdf")
    plt.close(fig)


def plot_method_availability(summary: pd.DataFrame) -> None:
    mat = np.zeros((len(DATASET_ORDER), len(METHOD_ORDER)))
    annot: list[list[str]] = []
    for i, ds in enumerate(DATASET_ORDER):
        row = []
        for j, method in enumerate(METHOD_ORDER):
            sub = summary[(summary["dataset_display"].eq(ds)) & (summary["method"].eq(method))]
            n = int(sub["n_done_folds"].iloc[0]) if not sub.empty else 0
            mat[i, j] = n / 5.0
            row.append(f"{n}/5")
        annot.append(row)
    fig, ax = plt.subplots(figsize=(12.8, 3.4), constrained_layout=True)
    im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([clean_method(m) for m in METHOD_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(DATASET_ORDER)))
    ax.set_yticklabels([f"{ds}\n{DATASET_TITLES[ds]}" for ds in DATASET_ORDER])
    for i in range(len(DATASET_ORDER)):
        for j in range(len(METHOD_ORDER)):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Prediction availability for final four-metric benchmark", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.023, pad=0.014, label="ready folds fraction")
    fig.savefig(FIG_DIR / "final_available_datasets_prediction_availability.png", dpi=260)
    fig.savefig(FIG_DIR / "final_available_datasets_prediction_availability.pdf")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_PATH)
    ranks = pd.read_csv(RANK_PATH)
    plot_metric_grid(summary)
    plot_rank_heatmap(summary, ranks)
    plot_method_availability(summary)
    print(FIG_DIR / "final_available_datasets_four_metric_benchmark.png")
    print(FIG_DIR / "final_available_datasets_four_metric_rank_heatmap.png")
    print(FIG_DIR / "final_available_datasets_prediction_availability.png")


if __name__ == "__main__":
    main()
