import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class MethodSpec:
    name: str
    script: str
    root_arg: str
    result_name: str
    extra_args: List[str]


METHOD_SPECS = {
    "spage": MethodSpec(
        name="SpaGE",
        script="/workspace/GeneSPT/baseline/SpaGE/run_spage_mhpr_fold_from_split.py",
        root_arg="spage_root",
        result_name="final_result_stdiff_style.csv",
        extra_args=[],
    ),
    "tangram": MethodSpec(
        name="Tangram",
        script="/workspace/GeneSPT/baseline/tangram/run_tangram_mhpr_fold_from_split.py",
        root_arg="tangram_root",
        result_name="final_result_stdiff_style.csv",
        extra_args=[],
    ),
    "stplus": MethodSpec(
        name="stPlus",
        script="/workspace/GeneSPT/baseline/stPlus/run_stplus_mhpr_fold_from_split.py",
        root_arg="stplus_root",
        result_name="final_result_stdiff_style.csv",
        extra_args=["--scrna-max-cells", "5000"],
    ),
    "transpa": MethodSpec(
        name="tranSpa",
        script="/workspace/GeneSPT/baseline/tranSpa-main/run_transpa_mhpr_fold_from_split.py",
        root_arg="transpa_root",
        result_name="final_result_stdiff_style.csv",
        extra_args=["--scrna-max-cells", "5000"],
    ),
    "spaim": MethodSpec(
        name="SpaIM",
        script="/workspace/GeneSPT/baseline/SpaIM-main/run_spaim_mhpr_fold_from_split.py",
        root_arg="spaim_root",
        result_name="final_result_stdiff_style.csv",
        extra_args=["--scrna-max-cells", "5000", "--epochs", "50", "--batch-size", "500"],
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--locations-path", required=True)
    parser.add_argument("--st-data", required=True)
    parser.add_argument("--sc-data", required=True)
    parser.add_argument("--gene-split-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_SPECS), required=True)
    parser.add_argument("--genespt-root")
    parser.add_argument("--stdiff-root")
    parser.add_argument("--spage-root")
    parser.add_argument("--tangram-root")
    parser.add_argument("--stplus-root")
    parser.add_argument("--transpa-root")
    parser.add_argument("--spaim-root")
    parser.add_argument("--table-output")
    return parser.parse_args()


def load_state(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"runs": {}, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_state(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_fold_state(state: Dict, method_key: str, fold_idx: int) -> Dict:
    method_state = state["runs"].setdefault(method_key, {})
    return method_state.setdefault(
        f"fold{fold_idx}",
        {"status": "pending", "attempts": 0, "started_at": None, "ended_at": None, "log_file": None},
    )


def build_command(args, spec: MethodSpec, fold_idx: int, output_root: Path) -> List[str]:
    return [
        sys.executable,
        spec.script,
        "--locations-path",
        args.locations_path,
        "--st-data",
        args.st_data,
        "--sc-data",
        args.sc_data,
        "--gene-split-json",
        str(Path(args.gene_split_dir) / f"fold{fold_idx}.json"),
        "--output-dir",
        str(output_root / f"fold{fold_idx}"),
        *spec.extra_args,
    ]


def expected_result_path(output_root: Path, spec: MethodSpec, fold_idx: int) -> Path:
    return output_root / f"fold{fold_idx}" / spec.result_name


def mark_done_if_existing(state: Dict, method_key: str, fold_idx: int, result_path: Path, log_file: Path) -> bool:
    if not result_path.exists():
        return False
    fold_state = ensure_fold_state(state, method_key, fold_idx)
    fold_state["status"] = "done"
    fold_state["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fold_state["log_file"] = str(log_file)
    fold_state["result_file"] = str(result_path)
    return True


def run_one_fold(args, state: Dict, method_key: str, spec: MethodSpec, fold_idx: int) -> None:
    output_root = Path(getattr(args, spec.root_arg))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"fold{fold_idx}").mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{method_key}_fold{fold_idx}.log"
    result_path = expected_result_path(output_root, spec, fold_idx)

    if mark_done_if_existing(state, method_key, fold_idx, result_path, log_file):
        return

    fold_state = ensure_fold_state(state, method_key, fold_idx)
    fold_state["status"] = "running"
    fold_state["attempts"] = int(fold_state.get("attempts", 0)) + 1
    fold_state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fold_state["log_file"] = str(log_file)

    cmd = build_command(args, spec, fold_idx, output_root)
    fold_state["command"] = shlex.join(cmd)
    save_state(Path(args.state_json), state)

    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] RUN {shlex.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd="/workspace")

    if proc.returncode == 0 and result_path.exists():
        fold_state["status"] = "done"
        fold_state["result_file"] = str(result_path)
    else:
        fold_state["status"] = "failed"
        fold_state["returncode"] = int(proc.returncode)
        fold_state["result_file"] = str(result_path)
    fold_state["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(Path(args.state_json), state)


def all_method_results_exist(args, method_keys: List[str]) -> bool:
    for method_key in method_keys:
        spec = METHOD_SPECS[method_key]
        root = Path(getattr(args, spec.root_arg))
        for fold_idx in range(5):
            if not expected_result_path(root, spec, fold_idx).exists():
                return False
    return True


def maybe_build_table(args) -> None:
    if not args.table_output:
        return
    if not args.genespt_root or not args.stdiff_root:
        return
    if not all_method_results_exist(args, args.methods):
        return
    cmd = [
        sys.executable,
        "/workspace/GeneSPT/main/build_baseline_table_from_folds.py",
        "--dataset-label",
        args.dataset_label,
        "--genespt-root",
        args.genespt_root,
        "--stdiff-root",
        args.stdiff_root,
        "--spage-root",
        args.spage_root,
        "--tangram-root",
        args.tangram_root,
        "--stplus-root",
        args.stplus_root,
        "--transpa-root",
        args.transpa_root,
        "--output",
        args.table_output,
    ]
    if args.spaim_root:
        cmd.extend(["--spaim-root", args.spaim_root])
    subprocess.run(cmd, check=True, cwd="/workspace")


def main():
    args = parse_args()
    state_path = Path(args.state_json)
    state = load_state(state_path)
    state["dataset_label"] = args.dataset_label
    state.pop("finished_at", None)
    state["last_resumed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    for method_key in args.methods:
        spec = METHOD_SPECS[method_key]
        for fold_idx in range(5):
            run_one_fold(args, state, method_key, spec, fold_idx)

    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state_path, state)
    maybe_build_table(args)


if __name__ == "__main__":
    main()
