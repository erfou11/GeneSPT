from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_complete_set_metrics import run_self_test as run_metric_audit_self_test  # noqa: E402
from compare_cell2location_strict_psp import run_self_test as run_psp_self_test  # noqa: E402


def test_complete_set_audit_self_test() -> None:
    result = run_metric_audit_self_test()

    assert result == {"status": "PASS", "methods": 2, "genes": 3}


def test_strict_psp_comparison_self_test() -> None:
    result = run_psp_self_test()

    assert result["status"] == "PASS"
    assert result["folds"] == 2
    assert result["method_fold_rows"] == 4
    assert result["gene_level_rows"] == 8
