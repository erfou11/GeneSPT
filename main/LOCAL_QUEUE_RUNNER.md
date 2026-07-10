# Local Resumable Experiment Queue

## Files
- runner: `/workspace/GeneSPT/main/local_experiment_queue.py`
- example manifest: `/workspace/GeneSPT/results/manifests/example_local_queue.jsonl`
- current stDiff manifest: `/workspace/GeneSPT/results/manifests/stdiff_epoch_sweep_fold1.jsonl`

## Manifest format
One JSON object per line. Required or expected fields:
- `id`
- `cmd`
- `cwd`
- `env`
- `gpu_id`
- `resume_flag_or_resume_cmd`
- `max_retries`
- `priority` (optional)

Optional fields supported by the runner:
- `run_dir`
- `checkpoint_glob`
- `heartbeat_timeout_sec`

`cmd`, `cwd`, `env`, `resume_flag_or_resume_cmd`, and `checkpoint_glob` can use:
- `{id}`
- `{run_dir}`
- `{gpu_id}`
- `{attempt}`
- `{queue_root}`
- `{checkpoint}` (resume template only)

## Per-task output layout
Each task uses `runs_root/<id>/` unless `run_dir` is provided.

Inside each run dir the runner maintains:
- `stdout.log`
- `stderr.log`
- `status.json`
- `heartbeat.json`
- `config_snapshot.json`
- `checkpoints/`

The task command itself should write experiment-specific artifacts under the same run dir when possible, for example `--output-dir {run_dir}/output`.
For resumed training, prefer:
- `--output-dir {run_dir}/outputs`
- `--checkpoint-dir {run_dir}/checkpoints`
- `--metrics-file {run_dir}/outputs/metrics.csv`
- `--heartbeat-file {run_dir}/outputs/train_heartbeat.json`
- `--status-file {run_dir}/outputs/train_status.json`

## Commands
### Start queue
```bash
python /workspace/GeneSPT/main/local_experiment_queue.py start \
  --manifest /workspace/GeneSPT/results/manifests/example_local_queue.jsonl \
  --runs-root /workspace/GeneSPT/runs/example_queue \
  --max-concurrent-workers 1 \
  --respect-external-gpu-busy \
  --retry-failed
```

### Show status
```bash
python /workspace/GeneSPT/main/local_experiment_queue.py status \
  --manifest /workspace/GeneSPT/results/manifests/example_local_queue.jsonl \
  --runs-root /workspace/GeneSPT/runs/example_queue
```

### Show failed / stale tasks
```bash
python /workspace/GeneSPT/main/local_experiment_queue.py failures \
  --manifest /workspace/GeneSPT/results/manifests/example_local_queue.jsonl \
  --runs-root /workspace/GeneSPT/runs/example_queue
```

### Requeue failed / stale tasks
```bash
python /workspace/GeneSPT/main/local_experiment_queue.py requeue \
  --manifest /workspace/GeneSPT/results/manifests/example_local_queue.jsonl \
  --runs-root /workspace/GeneSPT/runs/example_queue \
  --states stale failed
```

## Stop / restart
- Recommended start mode for long queues:
```bash
nohup python /workspace/GeneSPT/main/local_experiment_queue.py start ... > queue.out 2>&1 &
```
- To stop the queue runner, kill the runner PID from `runs_root/_queue/runner.json`.
- If the runner dies, restart the same `start` command.
- On restart, tasks left in `running` with expired heartbeat and dead PID are marked `stale`.
- With default settings, `stale` tasks are automatically reclaimed and continued from queue order.

## Notes
- GPU scheduling is one-worker-per-gpu by manifest `gpu_id`.
- `--respect-external-gpu-busy` prevents the runner from starting a queued job while another non-runner process is already occupying the same GPU.
- `summary.csv` is generated under `runs_root/_queue/summary.csv`.
- If `checkpoint_glob` and `resume_flag_or_resume_cmd` are set, the runner resolves the newest checkpoint and appends `--resume-from {checkpoint}` or another manifest-defined resume fragment.

## Current demo queue
The current stDiff epoch sweep manifest is:
- `/workspace/GeneSPT/results/manifests/stdiff_epoch_sweep_fold1.jsonl`

Recommended launch:
```bash
nohup python /workspace/GeneSPT/main/local_experiment_queue.py start \
  --manifest /workspace/GeneSPT/results/manifests/stdiff_epoch_sweep_fold1.jsonl \
  --runs-root /workspace/GeneSPT/runs/stdiff_epoch_sweep_fold1 \
  --max-concurrent-workers 1 \
  --respect-external-gpu-busy \
  --retry-failed > /workspace/GeneSPT/results/stdiff_epoch_sweep_runner.out 2>&1 &
```
