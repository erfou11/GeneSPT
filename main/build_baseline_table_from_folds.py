import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_KEYS = ["SPCC", "SSIM", "RMSE", "JS", "ARI", "NMI"]
GENESPT_SUMMARY_MAP = {
    "SPCC": "SPCC_gene_median_stdiff_style",
    "SSIM": "SSIM_gene_median_stdiff_style",
    "RMSE": "RMSE_gene_median_stdiff_style",
    "JS": "JS_gene_median_stdiff_style",
    "ARI": "ARI",
    "NMI": "NMI",
}


def _safe_float(value):
    if value is None:
        return np.nan
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if math.isfinite(value) else np.nan


def _read_genespt_latest(latest_root: Path):
    summary_path = latest_root / "summary" / "fold_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing GeneSPT fold summary: {summary_path}")
    fold_df = pd.read_csv(summary_path)
    row = {"method": "GeneSPT-condonlygate"}
    for out_key, in_key in GENESPT_SUMMARY_MAP.items():
        if in_key in fold_df.columns:
            row[out_key] = float(pd.to_numeric(fold_df[in_key], errors="coerce").mean())
        else:
            row[out_key] = np.nan
    if "ari_mode" in fold_df.columns and fold_df["ari_mode"].notna().any():
        row["ari_nmi_mode"] = str(fold_df["ari_mode"].dropna().iloc[0])
    else:
        row["ari_nmi_mode"] = ""
    return row


def _collect_fold_metric(root: Path, fold_idx: int):
    fold_dir = root / f"fold{fold_idx}"
    if not fold_dir.exists():
        return None

    result_csv = fold_dir / "final_result.csv"
    result_stdiff_style = fold_dir / "final_result_stdiff_style.csv"
    metrics_csv = fold_dir / "metrics.csv"

    row = {}
    if result_stdiff_style.exists():
        df = pd.read_csv(result_stdiff_style)
        if not df.empty:
            src = df.iloc[0].to_dict()
            row["SPCC"] = _safe_float(src.get("SPCC_gene_median_stdiff_style"))
            row["SSIM"] = _safe_float(src.get("SSIM_gene_median_stdiff_style"))
            row["RMSE"] = _safe_float(src.get("RMSE_gene_median_stdiff_style"))
            row["JS"] = _safe_float(src.get("JS_gene_median_stdiff_style"))
    elif result_csv.exists():
        df = pd.read_csv(result_csv)
        if not df.empty:
            src = df.iloc[0].to_dict()
            row["SPCC"] = _safe_float(src.get("SPCC"))
            row["SSIM"] = _safe_float(src.get("SSIM"))
            row["RMSE"] = _safe_float(src.get("RMSE"))
            row["JS"] = _safe_float(src.get("JS"))

    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        if not df.empty:
            src = df.iloc[0].to_dict()
            row["ARI"] = _safe_float(src.get("ARI"))
            row["NMI"] = _safe_float(src.get("NMI"))
            row["ari_nmi_mode"] = str(src.get("ARI_NMI_mode", ""))
    return row if row else None


def _aggregate_method(method_name: str, root: Path):
    rows = []
    for fold_idx in range(5):
        fold_row = _collect_fold_metric(root, fold_idx)
        if fold_row is not None:
            rows.append(fold_row)
    if not rows:
        raise FileNotFoundError(f"No fold outputs found under {root}")
    agg = {"method": method_name}
    df = pd.DataFrame(rows)
    for key in METRIC_KEYS:
        agg[key] = float(pd.to_numeric(df.get(key), errors="coerce").mean()) if key in df.columns else np.nan
    if "ari_nmi_mode" in df.columns and df["ari_nmi_mode"].astype(str).str.len().gt(0).any():
        agg["ari_nmi_mode"] = str(df["ari_nmi_mode"].astype(str).loc[df["ari_nmi_mode"].astype(str).str.len().gt(0)].iloc[0])
    else:
        agg["ari_nmi_mode"] = ""
    return agg


def _find_latest_genespt(root: Path):
    candidates = sorted(root.glob("mhpr_genespt_*"))
    if not candidates:
        raise FileNotFoundError(f"No GeneSPT runs found under {root}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--genespt-root", required=True)
    parser.add_argument("--stdiff-root", required=True)
    parser.add_argument("--spage-root", required=True)
    parser.add_argument("--tangram-root", required=True)
    parser.add_argument("--stplus-root", required=True)
    parser.add_argument("--transpa-root", required=True)
    parser.add_argument("--spaim-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    genespt_row = _read_genespt_latest(_find_latest_genespt(Path(args.genespt_root)))
    rows = [
        genespt_row,
        _aggregate_method("stDiff", Path(args.stdiff_root)),
        _aggregate_method("SpaGE", Path(args.spage_root)),
        _aggregate_method("Tangram", Path(args.tangram_root)),
        _aggregate_method("stPlus", Path(args.stplus_root)),
        _aggregate_method("tranSpa", Path(args.transpa_root)),
    ]
    if args.spaim_root:
        rows.append(_aggregate_method("SpaIM", Path(args.spaim_root)))

    header = ["method", "SPCC", "SSIM", "RMSE", "JS", "ARI", "NMI", "ari_nmi_mode"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"dataset:{args.dataset_label}\n")
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


if __name__ == "__main__":
    main()
