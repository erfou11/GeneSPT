from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = "3376cc16cc6d8461edafc0aeb4519b92d18474b7"
BENCHMARK_METHODS = {
    "GeneSPT",
    "Tangram",
    "TransImp",
    "SpaIM",
    "SpaGE",
    "stPlus",
    "stAI",
}
FIGURE6_METHODS = BENCHMARK_METHODS


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def literal_assignment(relative: str, name: str):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a literal assignment in {relative}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stai_version_and_adapter_contract() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "protocol_a_baseline_versions.yaml").read_text(
            encoding="utf-8"
        )
    )
    stai = config["methods"]["stAI"]
    adapter = ROOT / stai["adapter"]
    assert adapter.is_file()
    assert stai["official_repository"] == "https://github.com/gszou99/stAI"
    assert stai["upstream_revision"] == OFFICIAL_COMMIT
    assert stai["article_doi"] == "10.1093/nar/gkaf158"
    assert stai["runtime_patch"] == "none"
    assert stai["parameters"]["epochs"] == 500
    assert stai["parameters"]["internal_models"] == 5
    assert stai["parameters"]["topk"] == 50
    assert stai["parameters"]["pseudo_labels"] == "disabled"

    source = adapter.read_text(encoding="utf-8")
    assert f'OFFICIAL_COMMIT = "{OFFICIAL_COMMIT}"' in source
    assert "test_gene_st_expression_in_package" in source
    assert '"test_st_truth_accessed": False' in source
    assert "D:\\TESTWORK001" not in source
    assert "/workspace/" not in source


def test_stai_formal_adoption_manifest_has_all_thirty_folds() -> None:
    path = (
        ROOT
        / "manifests"
        / "protocol_a"
        / "STAI_FORMAL_ADOPTION_MANIFEST.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["method"] == "stAI"
    assert manifest["official_stai_commit"] == OFFICIAL_COMMIT
    rows = manifest["folds"]
    assert len(rows) == 30
    assert len({(row["dataset_id"], int(row["fold"])) for row in rows}) == 30
    assert {int(row["fold"]) for row in rows} == set(range(5))
    assert all(len(row["prediction_sha256"]) == 64 for row in rows)
    assert all(len(row["test_gene_idx_sha256"]) == 64 for row in rows)


def test_formal_source_tables_cover_seven_methods() -> None:
    five = read_csv(
        "results/source_data/protocol_a/benchmark/formal_five_fold_metrics.csv"
    )
    folds = read_csv(
        "results/source_data/protocol_a/benchmark/formal_fold_metrics.csv"
    )
    bootstrap = read_csv(
        "results/source_data/protocol_a/benchmark/paired_gene_bootstrap.csv"
    )
    assert len(five) == 42
    assert len(folds) == 210
    assert len(bootstrap) == 144
    assert {row["method"] for row in five} == BENCHMARK_METHODS
    assert {row["method"] for row in folds} == BENCHMARK_METHODS
    assert {row["comparator"] for row in bootstrap} == BENCHMARK_METHODS - {
        "GeneSPT"
    }
    assert all("/workspace/" not in row["prediction_path"] for row in folds)
    assert all(
        row["prediction_path"].startswith("${FORMAL_RUN_ROOT}/") for row in folds
    )


def test_figure2_figure4_figure5_and_s2_cover_stai() -> None:
    figure2 = read_csv("results/source_data/protocol_a/figures/figure2/source.csv")
    figure4 = read_csv("results/source_data/protocol_a/figures/figure4/source.csv")
    s2 = read_csv(
        "results/source_data/protocol_a/supplementary/S2/"
        "supplementary_table_s2_formal_benchmark.csv"
    )
    figure5_manifest = json.loads(
        (
            ROOT
            / "results/source_data/protocol_a/figures/figure5/source_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert len(figure2) == 21
    assert {row["method"] for row in figure2} == BENCHMARK_METHODS
    assert sum(row["method"] == "stAI" for row in figure2) == 3

    assert len(figure4) == 32
    assert {row["method"] for row in figure4} == {
        "Ground truth",
        *BENCHMARK_METHODS,
    }
    assert sum(row["method"] == "stAI" for row in figure4) == 4
    assert all("D:\\TESTWORK001" not in json.dumps(row) for row in figure4)
    assert all("/workspace/" not in json.dumps(row) for row in figure4)

    assert len(s2) == 42
    assert {row["method"] for row in s2} == BENCHMARK_METHODS
    assert sum(row["method"] == "stAI" for row in s2) == 6

    assert figure5_manifest["rows"] == 311780
    assert set(figure5_manifest["methods"]) == BENCHMARK_METHODS
    assert len(figure5_manifest["sha256"]) == 64


def test_figure_generators_use_all_seven_formal_methods() -> None:
    figure2 = set(
        literal_assignment(
            "scripts/protocol_a_full/generate_protocol_a_figure2.py", "METHOD_ORDER"
        )
    )
    figure4 = set(
        literal_assignment(
            "scripts/protocol_a_full/generate_protocol_a_figure4.py", "METHOD_ORDER"
        )
    )
    figure5 = set(
        literal_assignment(
            "scripts/protocol_a_full/generate_protocol_a_figure5_s2.py",
            "FORMAL_METHODS",
        )
    )
    figure6 = set(
        literal_assignment(
            "scripts/protocol_a_full/generate_protocol_a_figure6.py", "METHODS"
        )
    )
    assert figure2 == BENCHMARK_METHODS
    assert figure4 == {"Ground truth", *BENCHMARK_METHODS}
    assert figure5 == BENCHMARK_METHODS
    assert figure6 == FIGURE6_METHODS


def test_lightweight_source_manifests_match_tracked_files() -> None:
    figure4_root = ROOT / "results/source_data/protocol_a/figures/figure4"
    figure4_manifest = json.loads(
        (figure4_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert figure4_manifest["sha256"] == sha256(figure4_root / "source.csv")

    s2_root = ROOT / "results/source_data/protocol_a/supplementary/S2"
    s2_manifest = json.loads((s2_root / "manifest.json").read_text(encoding="utf-8"))
    artifact = s2_manifest["artifacts"]["benchmark_table"]
    assert artifact["sha256"] == sha256(s2_root / artifact["file"])
