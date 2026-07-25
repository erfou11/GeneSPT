from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / "reproducibility" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(name: str) -> list[dict[str, str]]:
    path = ROOT / "manifests" / "protocol_a" / name
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_manifest_cardinality_and_method_set() -> None:
    predictions = read_rows("PREDICTION_MATRIX_MANIFEST.csv")
    truths = read_rows("PROTOCOL_A_TRUTH_MATRIX_MANIFEST.csv")
    baselines = read_rows("FORMAL_BASELINE_RUN_MANIFEST.csv")
    inputs = read_rows("INPUT_SPLIT_MANIFEST.csv")
    selections = read_rows("READOUT_SELECTION_MANIFEST.csv")
    mechanisms = read_rows("MECHANISM_ABLATION_MATRIX_MANIFEST.csv")
    assert len(predictions) == 180
    assert len(truths) == 30
    assert len(baselines) == 150
    assert len(inputs) == 30
    assert len(selections) == 60
    assert len(mechanisms) == 90
    assert {row["method"] for row in predictions} == {
        "GeneSPT",
        "Tangram",
        "TransImp",
        "SpaIM",
        "SpaGE",
        "stPlus",
    }
    assert all(row["fallback_used"] == "False" for row in baselines)
    assert all(row["candidate_count"] == "57" for row in selections)
    assert all(row["test_truth_accessed_before_lock"] == "False" for row in selections)
    assert {row["panel"] for row in mechanisms} == {"A", "B", "C"}
    assert all(row["readout"] == "identity" for row in mechanisms)
    assert all(row["posthoc_calibration"] == "none" for row in mechanisms)


def test_adapter_hashes_match_formal_manifest() -> None:
    paths = {
        "Tangram": ROOT / "baseline_adapters" / "tangram" / "run_tangram_mhpr_fold_from_split.py",
        "TransImp": ROOT / "baseline_adapters" / "transimp" / "run_transimp_mhpr_fold_from_split.py",
        "SpaIM": ROOT / "baseline_adapters" / "spaim" / "run_spaim_mhpr_fold_from_split.py",
        "SpaGE": ROOT / "baseline_adapters" / "spage" / "run_spage_mhpr_fold_from_split.py",
        "stPlus": ROOT / "baseline_adapters" / "stplus" / "run_stplus_mhpr_fold_from_split.py",
    }
    rows = read_rows("FORMAL_BASELINE_RUN_MANIFEST.csv")
    expected = {method: {row["adapter_sha256"] for row in rows if row["method"] == method} for method in paths}
    for method, path in paths.items():
        assert expected[method] == {sha256(path)}


def test_command_and_public_path_sanitization() -> None:
    exporter = load_script("export_protocol_a_release.py")
    value = exporter.sanitize_string(
        "/workspace/GeneSPT/results/protocol_a_full_rerun_20260711/baselines/x.npy"
    )
    assert value == "${FORMAL_RUN_ROOT}/baselines/x.npy"
    public_files = [
        ROOT / "manifests" / "protocol_a" / "FORMAL_BASELINE_RUN_MANIFEST.csv",
        ROOT / "manifests" / "protocol_a" / "PREDICTION_MATRIX_MANIFEST.csv",
        ROOT / "manifests" / "protocol_a" / "READOUT_SELECTION_MANIFEST.csv",
    ]
    banned = ("/workspace/", "D:\\TESTWORK001", "stDiff", "TransPA")
    for path in public_files:
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in banned)


def test_rank_summary_direction() -> None:
    evaluator = load_script("recompute_protocol_a_benchmark.py")
    rows = []
    for method, spcc, rmse in (("GeneSPT", 0.4, 1.0), ("Tangram", 0.3, 1.2)):
        rows.append(
            {
                "dataset": "tiny",
                "dataset_id": "tiny",
                "role": "Primary",
                "method": method,
                "result_layer": "identity",
                "fold_count": 5,
                "coverage": 1.0,
                "SPCC": spcc,
                "SPCC_std": 0.0,
                "RMSE": rmse,
                "RMSE_std": 0.0,
                "JSD": rmse / 10.0,
                "JSD_std": 0.0,
                "SSIM": spcc,
                "SSIM_std": 0.0,
            }
        )
    ranked = evaluator.rank_summary(rows)
    genespt = {(row["metric"], row["rank"]) for row in ranked if row["method"] == "GeneSPT"}
    assert genespt == {("SPCC", 1), ("RMSE", 1), ("JSD", 1), ("SSIM", 1)}
