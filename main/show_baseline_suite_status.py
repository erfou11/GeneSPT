import argparse
import json
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--table-output")
    return parser.parse_args()


def load_state(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing state file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def short_status(fold_state):
    status = fold_state.get("status", "unknown")
    attempts = fold_state.get("attempts", 0)
    return f"{status} (try={attempts})"


def running_processes():
    cmd = (
        "ps -eo pid,etime,%cpu,%mem,cmd | "
        "rg 'run_baseline_suite_resumable.py|run_spage_mhpr_fold_from_split.py|"
        "run_tangram_mhpr_fold_from_split.py|run_stplus_mhpr_fold_from_split.py|"
        "run_transpa_mhpr_fold_from_split.py|run_spaim_mhpr_fold_from_split.py' || true"
    )
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/workspace")
    text = out.stdout.strip()
    return text


def main():
    args = parse_args()
    state_path = Path(args.state_json)
    state = load_state(state_path)
    runs = state.get("runs", {})
    has_running = any(
        fold_state.get("status") == "running"
        for method_state in runs.values()
        for fold_state in method_state.values()
    )

    print(f"dataset: {state.get('dataset_label', '')}")
    print(f"state: {state_path}")
    if state.get("started_at"):
        print(f"started_at: {state['started_at']}")
    if state.get("last_resumed_at"):
        print(f"last_resumed_at: {state['last_resumed_at']}")
    if state.get("finished_at") and not has_running:
        print(f"finished_at: {state['finished_at']}")

    print()
    print("methods:")
    for method_name in sorted(runs):
        method_state = runs[method_name]
        done = []
        running = []
        failed = []
        pending = []
        for fold_name in sorted(method_state):
            fold_state = method_state[fold_name]
            status = fold_state.get("status")
            if status == "done":
                done.append(fold_name)
            elif status == "running":
                running.append(fold_name)
            elif status == "failed":
                failed.append(f"{fold_name}:{fold_state.get('returncode', '?')}")
            else:
                pending.append(fold_name)

        print(f"- {method_name}: done {len(done)}/5")
        if running:
            print(f"  running: {', '.join(running)}")
            for fold_name in running:
                fold_state = method_state[fold_name]
                print(f"  {fold_name}: {short_status(fold_state)}")
                if fold_state.get("log_file"):
                    print(f"  log: {fold_state['log_file']}")
        if failed:
            print(f"  failed: {', '.join(failed)}")
        if pending:
            print(f"  pending: {', '.join(pending)}")

    if args.table_output:
        table_path = Path(args.table_output)
        print()
        print(f"table: {'ready' if table_path.exists() else 'pending'}")
        print(f"path: {table_path}")

    procs = running_processes()
    print()
    print("live_processes:")
    if procs:
        print(procs)
    else:
        print("none")


if __name__ == "__main__":
    main()
