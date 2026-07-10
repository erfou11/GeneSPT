#!/usr/bin/env python3
"""Generate Figure 2 primary benchmark raw-value Cleveland dot plot.

Visualization-only script. It reads existing final summary tables and does not
rerun models, modify prediction matrices, or touch manuscript files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_output/final_main_results"
OUT.mkdir(parents=True, exist_ok=True)

TABLE1 = ROOT / "results/imformation/table1_primary_benchmark_final.csv"

PDF_OUT = OUT / "figure2_primary_benchmark_dotplot.pdf"
PNG_OUT = OUT / "figure2_primary_benchmark_dotplot.png"
SOURCE_OUT = OUT / "figure2_primary_benchmark_dotplot_source.csv"
CAPTION_OUT = OUT / "figure2_primary_benchmark_dotplot_caption.md"
CHANGELOG_OUT = OUT / "figure2_primary_benchmark_dotplot_changelog.md"

DATASET_ORDER = ["Vis9A", "HBC", "Cell2location"]
DATASET_LABELS = {
    "Vis9A": "Vis9A",
    "HBC": "HBC",
    "Cell2location": "Cell2location\nmouse brain",
}
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
METRICS = [
    ("SPCC", "SPCC ↑", "higher"),
    ("RMSE", "RMSE ↓", "lower"),
    ("JS", "JS/JSD ↓", "lower"),
    ("raw_SSIM", "SSIM ↑", "higher"),
]

FALLBACK_ROWS = [
    ("Vis9A", "GeneSPT", 0.1929, 1.3011, 0.4524, 0.0569),
    ("Vis9A", "SpaIM", 0.1898, 1.3153, 0.4714, 0.0276),
    ("Vis9A", "Tangram", 0.1785, 1.3042, 0.4691, 0.0387),
    ("Vis9A", "TransPA", 0.1296, 1.3380, 0.4754, 0.0543),
    ("Vis9A", "SpaGE", 0.1379, 1.3357, 0.4724, 0.0471),
    ("Vis9A", "stPlus", 0.1226, 1.3346, 0.4713, 0.0456),
    ("HBC", "GeneSPT", 0.1192, 1.3471, 0.4890, 0.0335),
    ("HBC", "SpaIM", 0.0978, 1.3766, 0.5362, 0.0103),
    ("HBC", "Tangram", 0.0964, 1.3727, 0.5328, 0.0118),
    ("HBC", "TransPA", 0.0878, 1.3633, 0.4962, 0.0293),
    ("HBC", "SpaGE", 0.0541, 1.3927, 0.5534, 0.0242),
    ("HBC", "stPlus", 0.0302, 1.3975, 0.5171, 0.0068),
    ("Cell2location", "GeneSPT", 0.1816, 1.2925, 0.3429, 0.0509),
    ("Cell2location", "SpaIM", 0.1577, 1.3476, 0.3595, 0.0190),
    ("Cell2location", "Tangram", 0.1706, 1.3160, 0.3600, 0.0299),
    ("Cell2location", "TransPA", 0.1177, 1.3522, 0.3722, 0.0217),
    ("Cell2location", "SpaGE", 0.1041, 1.3487, 0.3719, 0.0170),
    ("Cell2location", "stPlus", 0.0107, 1.3221, 0.3815, 0.0061),
]


def configure_fonts() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_method(method: str) -> str | None:
    if method in {"GeneSPT", "GeneSPT-GC-PSP", "current_descriptor_psp"}:
        return "GeneSPT"
    if method == "GeneSPT-GC":
        return None
    if method in METHOD_ORDER:
        return method
    return None


def load_table_source() -> tuple[pd.DataFrame, bool, str]:
    if TABLE1.exists():
        df = pd.read_csv(TABLE1)
        df["method_display"] = df["method"].map(normalize_method)
        df = df[df["dataset"].isin(DATASET_ORDER) & df["method_display"].isin(METHOD_ORDER)].copy()
        df = df.rename(
            columns={
                "SPCC_mean": "SPCC",
                "RMSE_mean": "RMSE",
                "JS_mean": "JS",
                "SSIM_mean": "raw_SSIM",
            }
        )
        df = df[["dataset", "method_display", "SPCC", "RMSE", "JS", "raw_SSIM", "status", "folds"]]
        df = df.rename(columns={"method_display": "method"})
        source_used = str(TABLE1)
        used_table = True
    else:
        df = pd.DataFrame(FALLBACK_ROWS, columns=["dataset", "method", "SPCC", "RMSE", "JS", "raw_SSIM"])
        df["status"] = "manual_fallback"
        df["folds"] = 5
        source_used = "manual fallback values supplied in task prompt"
        used_table = False

    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    df = df.sort_values(["dataset", "method"]).reset_index(drop=True)

    expected = {(dataset, method) for dataset in DATASET_ORDER for method in METHOD_ORDER}
    observed = {(str(row.dataset), str(row.method)) for row in df.itertuples(index=False)}
    if expected != observed:
        raise ValueError(f"Unexpected source coverage. Missing={sorted(expected - observed)}; extra={sorted(observed - expected)}")
    if (df["folds"].astype(int) != 5).any():
        raise ValueError("All Figure 2 rows must be five-fold summaries.")
    return df, used_table, source_used


def write_source(df: pd.DataFrame) -> pd.DataFrame:
    long_rows = []
    for row in df.itertuples(index=False):
        for metric, label, direction in METRICS:
            long_rows.append(
                {
                    "dataset": str(row.dataset),
                    "method": str(row.method),
                    "metric": metric,
                    "metric_label": label,
                    "raw_value": float(getattr(row, metric)),
                    "metric_direction": direction,
                    "folds": int(row.folds),
                    "status": str(row.status),
                }
            )
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(SOURCE_OUT, index=False)
    return long_df


def best_method(sub: pd.DataFrame, metric: str, direction: str) -> str:
    values = sub.set_index("method")[metric].astype(float)
    if direction == "higher":
        return str(values.idxmax())
    return str(values.idxmin())


def axis_limits(values: pd.Series) -> tuple[float, float]:
    vals = values.astype(float).to_numpy()
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = hi - lo
    if span == 0:
        span = max(abs(hi), 1.0) * 0.05
    pad = span * 0.22
    return lo - pad, hi + pad


def plot_dotplot(df: pd.DataFrame) -> None:
    configure_fonts()
    fig, axes = plt.subplots(len(DATASET_ORDER), len(METRICS), figsize=(14.8, 8.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.suptitle("Primary sequencing-based benchmark performance", fontsize=18, fontweight="bold", y=0.988)

    y_positions = np.arange(len(METHOD_ORDER))[::-1]
    y_lookup = {method: y for method, y in zip(METHOD_ORDER, y_positions)}

    for r, dataset in enumerate(DATASET_ORDER):
        dsub = df[df["dataset"].astype(str).eq(dataset)]
        for c, (metric, label, direction) in enumerate(METRICS):
            ax = axes[r, c]
            best = best_method(dsub, metric, direction)
            for method in METHOD_ORDER:
                value = float(dsub.loc[dsub["method"].astype(str).eq(method), metric].iloc[0])
                y = y_lookup[method]
                size = 95 if method == "GeneSPT" else 70
                ax.scatter(
                    value,
                    y,
                    s=size,
                    color=METHOD_COLORS[method],
                    edgecolor="#333333",
                    linewidth=0.65,
                    zorder=3,
                )
                if method == best:
                    ax.scatter(
                        value,
                        y,
                        s=size + 95,
                        facecolors="none",
                        edgecolors="#111111",
                        linewidth=1.25,
                        zorder=4,
                    )
            xmin, xmax = axis_limits(dsub[metric])
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-0.6, len(METHOD_ORDER) - 0.4)
            ax.set_title(label, fontsize=12.0, pad=7)
            ax.grid(axis="x", color="#dddddd", linewidth=0.65, alpha=0.7)
            ax.grid(axis="y", color="#eeeeee", linewidth=0.5, alpha=0.55)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#888888")
            ax.spines["bottom"].set_color("#888888")
            ax.tick_params(axis="x", labelsize=8.8)
            ax.tick_params(axis="y", length=0)
            ax.set_yticks(y_positions)
            if c == 0:
                ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=9.6)
                ax.set_ylabel(DATASET_LABELS[dataset], fontsize=12.0, fontweight="bold", labelpad=16)
            else:
                ax.set_yticklabels([])
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS[m],
            markeredgecolor="#333333",
            markersize=7.5 if m == "GeneSPT" else 6.8,
            label=METHOD_LABELS[m],
        )
        for m in METHOD_ORDER
    ]
    best_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="none",
        markeredgecolor="#111111",
        markeredgewidth=1.25,
        markersize=9.0,
        label="Best method",
    )
    fig.legend(
        handles=method_handles + [best_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.943),
        ncol=7,
        frameon=False,
        fontsize=10.2,
        columnspacing=1.45,
        handletextpad=0.55,
    )
    fig.text(
        0.5,
        0.025,
        "Dots show five-fold mean raw metric values. Axes are locally scaled within panels for readability.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#333333",
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.86, bottom=0.095, wspace=0.27, hspace=0.44)
    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_text_outputs(used_table: bool, source_used: str) -> None:
    CAPTION_OUT.write_text(
        "Figure 2. Primary sequencing-based benchmark performance under strict "
        "whole-gene evaluation. Dots show five-fold mean raw metric values for Vis9A, "
        "HBC and Cell2location mouse brain. GeneSPT is highlighted in red and compared with complete "
        "external baselines. The black open circle marks the best method within each "
        "dataset and metric. SPCC and SSIM are higher-is-better metrics, whereas "
        "RMSE and JS/JSD are lower-is-better metrics. Axes are locally scaled within "
        "panels for readability; full values are reported in Table 1.\n"
    )
    CHANGELOG_OUT.write_text(
        "# Figure 2 Primary Benchmark Dot Plot Changelog\n\n"
        f"1. Input source used: {source_used}.\n"
        f"2. Used table1_primary_benchmark_final.csv: {'yes' if used_table else 'no'}.\n"
        "3. Included only Vis9A, HBC and Cell2location mouse brain.\n"
        "4. Included only GeneSPT, SpaIM, Tangram, TransImp, SpaGE and stPlus for display.\n"
        "5. Internal GeneSPT-GC-PSP labels were displayed as GeneSPT; GeneSPT-GC was excluded.\n"
        "6. Used SSIM from SSIM_mean; SSIMx10_mean was not used.\n"
        "7. No model was rerun.\n"
        "8. No result values were modified.\n"
        "9. No rank-score transformation was used; points show raw metric values.\n"
        "10. The black open-circle marker denotes the best method within each dataset and metric.\n"
    )


def main() -> None:
    df, used_table, source_used = load_table_source()
    write_source(df)
    plot_dotplot(df)
    write_text_outputs(used_table, source_used)
    print(f"Wrote {PDF_OUT}")
    print(f"Wrote {PNG_OUT}")
    print(f"Wrote {SOURCE_OUT}")
    print(f"Wrote {CAPTION_OUT}")
    print(f"Wrote {CHANGELOG_OUT}")


if __name__ == "__main__":
    main()
