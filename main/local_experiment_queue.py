import argparse
import csv
import glob
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"
STATE_STALE = "stale"
TERMINAL_STATES = {STATE_DONE}
VALID_STATES = {STATE_PENDING, STATE_RUNNING, STATE_DONE, STATE_FAILED, STATE_STALE}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def append_event(status: Dict, event: str, extra: Optional[Dict] = None) -> None:
    history = status.setdefault("history", [])
    record = {"ts": utc_now(), "event": event}
    if extra:
        record.update(extra)
    history.append(record)
    if len(history) > 200:
        del history[:-200]


def shell_join(cmd) -> str:
    if isinstance(cmd, str):
        return cmd
    if isinstance(cmd, list):
        return shlex.join([str(x) for x in cmd])
    raise TypeError(f"Unsupported cmd type: {type(cmd)}")


def env_to_dict(env_obj) -> Dict[str, str]:
    if env_obj is None:
        return {}
    if isinstance(env_obj, dict):
        return {str(k): str(v) for k, v in env_obj.items()}
    raise TypeError(f"env must be a dict or null, got {type(env_obj)}")


def is_pid_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except Exception:
        return False
    return True


def read_manifest_jsonl(path: Path) -> List[Dict]:
    tasks = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            obj["_manifest_lineno"] = lineno
            tasks.append(obj)
    return tasks


def render_template(value, mapping: Dict[str, str]):
    if value is None:
        return None
    if isinstance(value, str):
        return value.format(**mapping)
    if isinstance(value, dict):
        return {str(k): render_template(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [render_template(v, mapping) for v in value]
    return value


def runner_busy_gpus() -> Dict[int, str]:
    gpu_map = {}
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        uuid_to_idx = {}
        for line in res.stdout.splitlines():
            if not line.strip():
                continue
            idx_str, uuid = [x.strip() for x in line.split(",", 1)]
            uuid_to_idx[uuid] = int(idx_str)

        proc_res = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc_res.returncode == 0:
            for line in proc_res.stdout.splitlines():
                if not line.strip():
                    continue
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 2:
                    continue
                uuid = parts[0]
                pid = parts[1]
                if uuid in uuid_to_idx:
                    gpu_map[uuid_to_idx[uuid]] = pid
    except Exception:
        return {}
    return gpu_map


@dataclass
class QueuePaths:
    manifest_path: Path
    runs_root: Path
    queue_root: Path
    claims_dir: Path
    runner_meta_path: Path
    summary_csv: Path


class LocalExperimentQueue:
    def __init__(
        self,
        manifest_path: Path,
        runs_root: Path,
        heartbeat_interval_sec: int,
        stale_timeout_sec: int,
        poll_interval_sec: int,
        max_concurrent_workers: int,
        respect_external_gpu_busy: bool,
        retry_failed: bool,
        auto_reclaim_stale: bool,
        exit_when_done: bool,
    ):
        self.paths = QueuePaths(
            manifest_path=manifest_path,
            runs_root=runs_root,
            queue_root=runs_root / "_queue",
            claims_dir=runs_root / "_queue" / "claims",
            runner_meta_path=runs_root / "_queue" / "runner.json",
            summary_csv=runs_root / "_queue" / "summary.csv",
        )
        self.heartbeat_interval_sec = int(heartbeat_interval_sec)
        self.stale_timeout_sec = int(stale_timeout_sec)
        self.poll_interval_sec = int(poll_interval_sec)
        self.max_concurrent_workers = int(max_concurrent_workers)
        self.respect_external_gpu_busy = bool(respect_external_gpu_busy)
        self.retry_failed = bool(retry_failed)
        self.auto_reclaim_stale = bool(auto_reclaim_stale)
        self.exit_when_done = bool(exit_when_done)
        self.runner_id = f"{os.uname().nodename}:{os.getpid()}:{int(time.time())}"
        self.active: Dict[str, Dict] = {}

        ensure_dir(self.paths.queue_root)
        ensure_dir(self.paths.claims_dir)
        ensure_dir(self.paths.runs_root)

    def task_run_dir(self, task: Dict) -> Path:
        custom = task.get("run_dir")
        if custom:
            return Path(custom)
        return self.paths.runs_root / str(task["id"])

    def task_paths(self, task: Dict) -> Dict[str, Path]:
        run_dir = self.task_run_dir(task)
        return {
            "run_dir": run_dir,
            "stdout": run_dir / "stdout.log",
            "stderr": run_dir / "stderr.log",
            "status": run_dir / "status.json",
            "heartbeat": run_dir / "heartbeat.json",
            "config": run_dir / "config_snapshot.json",
            "checkpoints": run_dir / "checkpoints",
            "claim": self.paths.claims_dir / f"{task['id']}.json",
        }

    def load_manifest(self) -> List[Dict]:
        tasks = read_manifest_jsonl(self.paths.manifest_path)
        seen = set()
        normalized = []
        for task in tasks:
            task_id = str(task["id"])
            if task_id in seen:
                raise ValueError(f"Duplicate task id in manifest: {task_id}")
            seen.add(task_id)
            task["id"] = task_id
            task["cmd"] = shell_join(task["cmd"])
            task["cwd"] = str(task.get("cwd") or "/workspace")
            task["env"] = env_to_dict(task.get("env"))
            task["gpu_id"] = None if task.get("gpu_id") in (None, "") else int(task["gpu_id"])
            task["resume_flag_or_resume_cmd"] = task.get("resume_flag_or_resume_cmd")
            task["max_retries"] = int(task.get("max_retries", 0))
            task["priority"] = int(task.get("priority", 0))
            task["checkpoint_glob"] = task.get("checkpoint_glob")
            task["heartbeat_timeout_sec"] = int(task.get("heartbeat_timeout_sec", self.stale_timeout_sec))
            normalized.append(task)
        return normalized

    def load_status(self, task: Dict) -> Dict:
        paths = self.task_paths(task)
        status = read_json(paths["status"], default=None)
        if status is None:
            status = {
                "id": task["id"],
                "state": STATE_PENDING,
                "attempts": 0,
                "pid": None,
                "gpu_id": task.get("gpu_id"),
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
        status.setdefault("id", task["id"])
        status.setdefault("state", STATE_PENDING)
        status.setdefault("attempts", 0)
        status.setdefault("gpu_id", task.get("gpu_id"))
        return status

    def save_status(self, task: Dict, status: Dict) -> None:
        status["updated_at"] = utc_now()
        write_json_atomic(self.task_paths(task)["status"], status)

    def save_heartbeat(self, task: Dict, payload: Dict) -> None:
        write_json_atomic(self.task_paths(task)["heartbeat"], payload)

    def load_heartbeat(self, task: Dict) -> Optional[Dict]:
        return read_json(self.task_paths(task)["heartbeat"], default=None)

    def claim_task(self, task: Dict, status: Dict) -> bool:
        claim_path = self.task_paths(task)["claim"]
        payload = {
            "id": task["id"],
            "runner_id": self.runner_id,
            "claimed_at": utc_now(),
            "pid": status.get("pid"),
        }
        try:
            fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return True

    def release_claim(self, task: Dict) -> None:
        claim_path = self.task_paths(task)["claim"]
        if claim_path.exists():
            claim_path.unlink()

    def try_mark_stale(self, task: Dict, status: Dict) -> Dict:
        if status.get("state") != STATE_RUNNING:
            return status
        heartbeat = self.load_heartbeat(task)
        stale_timeout = int(task.get("heartbeat_timeout_sec", self.stale_timeout_sec))
        if heartbeat is None:
            return status
        last_ts = heartbeat.get("ts_epoch")
        pid = heartbeat.get("pid") or status.get("pid")
        if last_ts is None:
            return status
        age = time.time() - float(last_ts)
        if age <= stale_timeout:
            return status
        if is_pid_alive(pid):
            return status
        status["state"] = STATE_STALE
        status["stale_at"] = utc_now()
        status["stale_reason"] = f"heartbeat_timeout>{stale_timeout}s"
        append_event(status, "stale_detected", {"age_sec": age})
        self.save_status(task, status)
        self.release_claim(task)
        return status

    def maybe_find_resume_checkpoint(self, task: Dict, run_dir: Path) -> Optional[Path]:
        pattern = task.get("checkpoint_glob")
        if not pattern:
            return None
        mapping = {
            "id": task["id"],
            "run_dir": str(run_dir),
            "gpu_id": "" if task.get("gpu_id") is None else str(task["gpu_id"]),
            "attempt": str(self.load_status(task).get("attempts", 0)),
            "queue_root": str(self.paths.runs_root),
        }
        rendered = render_template(pattern, mapping)
        matches = [Path(p) for p in glob.glob(rendered, recursive=True)]
        matches = [p for p in matches if p.exists()]
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def build_command(self, task: Dict, status: Dict, resume_checkpoint: Optional[Path]) -> Tuple[str, Dict[str, str], str]:
        run_dir = self.task_run_dir(task)
        mapping = {
            "id": task["id"],
            "run_dir": str(run_dir),
            "gpu_id": "" if task.get("gpu_id") is None else str(task["gpu_id"]),
            "attempt": str(status.get("attempts", 0)),
            "queue_root": str(self.paths.runs_root),
            "checkpoint": "" if resume_checkpoint is None else str(resume_checkpoint),
        }
        base_cmd = render_template(task["cmd"], mapping)
        resume_spec = task.get("resume_flag_or_resume_cmd")
        if resume_checkpoint is not None and resume_spec:
            rendered_resume = render_template(resume_spec, {**mapping, "cmd": base_cmd})
            if "{cmd}" in str(resume_spec):
                cmd = rendered_resume
            elif "{checkpoint}" in str(resume_spec):
                stripped = str(rendered_resume).strip()
                if stripped.startswith("python ") or stripped.startswith("/") or stripped.startswith("bash "):
                    cmd = stripped
                else:
                    cmd = f"{base_cmd} {stripped}"
            else:
                cmd = f"{base_cmd} {str(rendered_resume).strip()}"
        else:
            cmd = base_cmd

        env = os.environ.copy()
        env.update(render_template(task["env"], mapping) or {})
        if task.get("gpu_id") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(task["gpu_id"])
            env.setdefault("EXPERIMENT_GPU_ID", str(task["gpu_id"]))
        cwd = str(render_template(task["cwd"], mapping))
        return cmd, env, cwd

    def sync_checkpoints(self, task: Dict) -> None:
        run_dir = self.task_run_dir(task)
        ckpt_src = self.maybe_find_resume_checkpoint(task, run_dir)
        if ckpt_src is None:
            return
        ckpt_dir = self.task_paths(task)["checkpoints"]
        ensure_dir(ckpt_dir)
        dest = ckpt_dir / ckpt_src.name
        if dest.exists():
            return
        try:
            os.symlink(ckpt_src, dest)
        except FileExistsError:
            return
        except OSError:
            import shutil

            shutil.copy2(ckpt_src, dest)

    def launch_task(self, task: Dict, status: Dict) -> None:
        paths = self.task_paths(task)
        run_dir = paths["run_dir"]
        ensure_dir(run_dir)
        ensure_dir(paths["checkpoints"])
        resume_checkpoint = self.maybe_find_resume_checkpoint(task, run_dir)
        cmd, env, cwd = self.build_command(task, status, resume_checkpoint)

        status["attempts"] = int(status.get("attempts", 0)) + 1
        status["state"] = STATE_RUNNING
        status["started_at"] = utc_now()
        status["resume_checkpoint"] = None if resume_checkpoint is None else str(resume_checkpoint)
        append_event(status, "launch", {"attempt": status["attempts"], "resume_checkpoint": status["resume_checkpoint"]})
        self.save_status(task, status)

        snapshot = {
            "task": task,
            "resolved_command": cmd,
            "resolved_cwd": cwd,
            "resolved_env_overrides": {k: env[k] for k in task["env"].keys()},
            "gpu_id": task.get("gpu_id"),
            "attempt": status["attempts"],
            "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        }
        write_json_atomic(paths["config"], snapshot)

        stdout = paths["stdout"].open("a", encoding="utf-8")
        stderr = paths["stderr"].open("a", encoding="utf-8")
        stdout.write(f"\n[{utc_now()}] RUN attempt={status['attempts']} cmd={cmd}\n")
        stdout.flush()

        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        status["pid"] = int(proc.pid)
        self.save_status(task, status)
        self.save_heartbeat(task, {"id": task["id"], "pid": proc.pid, "ts": utc_now(), "ts_epoch": time.time(), "state": STATE_RUNNING})
        self.active[task["id"]] = {
            "proc": proc,
            "stdout": stdout,
            "stderr": stderr,
            "task": task,
            "status": status,
            "last_hb": 0.0,
        }

    def finalize_task(self, task: Dict, status: Dict, proc: subprocess.Popen) -> None:
        rc = proc.returncode
        if rc == 0:
            status["state"] = STATE_DONE
            status["ended_at"] = utc_now()
            append_event(status, "completed", {"returncode": rc})
        else:
            status["state"] = STATE_FAILED
            status["ended_at"] = utc_now()
            status["returncode"] = int(rc)
            append_event(status, "failed", {"returncode": rc})
        status["pid"] = None
        self.save_status(task, status)
        self.save_heartbeat(task, {"id": task["id"], "pid": None, "ts": utc_now(), "ts_epoch": time.time(), "state": status["state"]})
        self.release_claim(task)
        self.sync_checkpoints(task)

    def update_heartbeat(self, task: Dict, status: Dict) -> None:
        self.save_heartbeat(
            task,
            {
                "id": task["id"],
                "pid": status.get("pid"),
                "ts": utc_now(),
                "ts_epoch": time.time(),
                "state": status.get("state"),
                "attempts": status.get("attempts"),
            },
        )
        self.sync_checkpoints(task)

    def eligible_for_launch(self, task: Dict, status: Dict) -> bool:
        state = status.get("state", STATE_PENDING)
        if state == STATE_PENDING:
            return True
        if state == STATE_STALE and self.auto_reclaim_stale:
            return True
        if state == STATE_FAILED and self.retry_failed and int(status.get("attempts", 0)) < int(task.get("max_retries", 0)) + 1:
            return True
        return False

    def acquire_runner_lock(self) -> None:
        meta = read_json(self.paths.runner_meta_path, default=None)
        if meta and is_pid_alive(meta.get("pid")):
            raise RuntimeError(f"Runner already active: pid={meta['pid']} id={meta.get('runner_id')}")
        write_json_atomic(self.paths.runner_meta_path, {"runner_id": self.runner_id, "pid": os.getpid(), "started_at": utc_now()})

    def release_runner_lock(self) -> None:
        meta = read_json(self.paths.runner_meta_path, default=None)
        if meta and meta.get("pid") == os.getpid():
            self.paths.runner_meta_path.unlink(missing_ok=True)

    def available_gpu_ids(self, tasks: List[Dict]) -> Dict[Optional[int], bool]:
        used = {}
        for info in self.active.values():
            gpu_id = info["task"].get("gpu_id")
            if gpu_id is not None:
                used[gpu_id] = True
        if self.respect_external_gpu_busy:
            for gpu_id in runner_busy_gpus().keys():
                used[gpu_id] = True
        all_gpu_ids = {task.get("gpu_id") for task in tasks if task.get("gpu_id") is not None}
        return {gpu_id: (gpu_id not in used) for gpu_id in all_gpu_ids}

    def write_summary(self, tasks: List[Dict]) -> None:
        rows = []
        for task in tasks:
            status = self.load_status(task)
            hb = self.load_heartbeat(task) or {}
            rows.append(
                {
                    "id": task["id"],
                    "state": status.get("state"),
                    "attempts": status.get("attempts", 0),
                    "gpu_id": task.get("gpu_id"),
                    "priority": task.get("priority", 0),
                    "pid": status.get("pid"),
                    "started_at": status.get("started_at"),
                    "ended_at": status.get("ended_at"),
                    "last_heartbeat": hb.get("ts"),
                    "run_dir": str(self.task_run_dir(task)),
                }
            )
        ensure_dir(self.paths.summary_csv.parent)
        with self.paths.summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "state", "attempts", "gpu_id", "priority", "pid", "started_at", "ended_at", "last_heartbeat", "run_dir"],
            )
            writer.writeheader()
            writer.writerows(rows)

    def loop(self) -> None:
        self.acquire_runner_lock()
        try:
            while True:
                tasks = self.load_manifest()

                for task in tasks:
                    status = self.load_status(task)
                    self.try_mark_stale(task, status)

                finished_ids = []
                for task_id, info in list(self.active.items()):
                    proc = info["proc"]
                    task = info["task"]
                    status = self.load_status(task)
                    if proc.poll() is not None:
                        self.finalize_task(task, status, proc)
                        info["stdout"].close()
                        info["stderr"].close()
                        finished_ids.append(task_id)
                    else:
                        now = time.time()
                        if now - info["last_hb"] >= self.heartbeat_interval_sec:
                            self.update_heartbeat(task, status)
                            info["last_hb"] = now
                for task_id in finished_ids:
                    self.active.pop(task_id, None)

                self.write_summary(tasks)

                available_slots = max(0, self.max_concurrent_workers - len(self.active))
                gpu_available = self.available_gpu_ids(tasks)
                if available_slots > 0:
                    candidates = []
                    for task in tasks:
                        status = self.load_status(task)
                        if not self.eligible_for_launch(task, status):
                            continue
                        gpu_id = task.get("gpu_id")
                        if gpu_id is not None and not gpu_available.get(gpu_id, True):
                            continue
                        candidates.append((int(task.get("priority", 0)), task["id"], task, status))
                    candidates.sort(key=lambda x: (-x[0], x[1]))
                    for _, _, task, status in candidates[:available_slots]:
                        if not self.claim_task(task, status):
                            continue
                        self.launch_task(task, status)
                        gpu_id = task.get("gpu_id")
                        if gpu_id is not None:
                            gpu_available[gpu_id] = False

                has_eligible = any(self.eligible_for_launch(task, self.load_status(task)) for task in tasks)
                if self.exit_when_done and not self.active and not has_eligible:
                    break
                time.sleep(self.poll_interval_sec)
        finally:
            self.release_runner_lock()


def print_table(rows: List[Dict]) -> None:
    if not rows:
        print("No tasks.")
        return
    headers = ["id", "state", "attempts", "gpu_id", "pid", "started_at", "ended_at"]
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def load_summary_rows(runs_root: Path, manifest_path: Path) -> List[Dict]:
    summary_path = runs_root / "_queue" / "summary.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    q = LocalExperimentQueue(
        manifest_path=manifest_path,
        runs_root=runs_root,
        heartbeat_interval_sec=60,
        stale_timeout_sec=900,
        poll_interval_sec=10,
        max_concurrent_workers=1,
        respect_external_gpu_busy=False,
        retry_failed=False,
        auto_reclaim_stale=False,
        exit_when_done=True,
    )
    tasks = q.load_manifest()
    q.write_summary(tasks)
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def requeue_tasks(manifest_path: Path, runs_root: Path, task_ids: Optional[List[str]], states: List[str]) -> None:
    q = LocalExperimentQueue(
        manifest_path=manifest_path,
        runs_root=runs_root,
        heartbeat_interval_sec=60,
        stale_timeout_sec=900,
        poll_interval_sec=10,
        max_concurrent_workers=1,
        respect_external_gpu_busy=False,
        retry_failed=False,
        auto_reclaim_stale=False,
        exit_when_done=True,
    )
    tasks = q.load_manifest()
    selected = 0
    for task in tasks:
        if task_ids and task["id"] not in task_ids:
            continue
        status = q.load_status(task)
        if status.get("state") not in states:
            continue
        if is_pid_alive(status.get("pid")):
            continue
        q.release_claim(task)
        status["state"] = STATE_PENDING
        status["pid"] = None
        status.pop("ended_at", None)
        append_event(status, "manual_requeue", {"from_state": states})
        q.save_status(task, status)
        selected += 1
    print(f"Requeued {selected} task(s).")


def build_arg_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="subcommand", required=True)

    start = sub.add_parser("start")
    start.add_argument("--manifest", required=True)
    start.add_argument("--runs-root", required=True)
    start.add_argument("--heartbeat-interval-sec", type=int, default=60)
    start.add_argument("--stale-timeout-sec", type=int, default=900)
    start.add_argument("--poll-interval-sec", type=int, default=10)
    start.add_argument("--max-concurrent-workers", type=int, default=1)
    start.add_argument("--respect-external-gpu-busy", action="store_true")
    start.add_argument("--retry-failed", action="store_true")
    start.add_argument("--no-auto-reclaim-stale", dest="auto_reclaim_stale", action="store_false")
    start.add_argument("--keep-running", dest="exit_when_done", action="store_false")
    start.set_defaults(auto_reclaim_stale=True, exit_when_done=True)

    status = sub.add_parser("status")
    status.add_argument("--manifest", required=True)
    status.add_argument("--runs-root", required=True)

    failures = sub.add_parser("failures")
    failures.add_argument("--manifest", required=True)
    failures.add_argument("--runs-root", required=True)

    requeue = sub.add_parser("requeue")
    requeue.add_argument("--manifest", required=True)
    requeue.add_argument("--runs-root", required=True)
    requeue.add_argument("--ids", nargs="*")
    requeue.add_argument("--states", nargs="+", choices=[STATE_FAILED, STATE_STALE], default=[STATE_STALE, STATE_FAILED])

    return p


def main():
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest)
    runs_root = Path(args.runs_root)

    if args.subcommand == "start":
        q = LocalExperimentQueue(
            manifest_path=manifest_path,
            runs_root=runs_root,
            heartbeat_interval_sec=args.heartbeat_interval_sec,
            stale_timeout_sec=args.stale_timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
            max_concurrent_workers=args.max_concurrent_workers,
            respect_external_gpu_busy=args.respect_external_gpu_busy,
            retry_failed=args.retry_failed,
            auto_reclaim_stale=args.auto_reclaim_stale,
            exit_when_done=args.exit_when_done,
        )
        q.loop()
        return

    if args.subcommand == "status":
        rows = load_summary_rows(runs_root, manifest_path)
        print_table(rows)
        print(f"\nsummary_csv: {runs_root / '_queue' / 'summary.csv'}")
        return

    if args.subcommand == "failures":
        rows = [r for r in load_summary_rows(runs_root, manifest_path) if r.get("state") in {STATE_FAILED, STATE_STALE}]
        print_table(rows)
        return

    if args.subcommand == "requeue":
        requeue_tasks(manifest_path, runs_root, args.ids, args.states)
        return


if __name__ == "__main__":
    main()
