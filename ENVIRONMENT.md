# GeneSPT Environment

This workspace should be run from a CUDA Docker container, not from Windows
Python and not from the bare WSL Python. The stable project path inside the
container is:

```text
/workspace/GeneSPT
```

## Recommended Entry Point

From `D:\TESTWORK001` on Windows:

```powershell
docker compose build genespt
docker compose up -d genespt
docker compose exec genespt bash -lc "cd /workspace/GeneSPT && python scripts/env_smoke_test.py"
```

For VS Code, use **Dev Containers: Reopen in Container**. The checked-in
`.devcontainer/devcontainer.json` opens the project at `/workspace/GeneSPT`
and uses `/opt/conda/bin/python`.

## Current CUDA Baseline

- Docker image: `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`
- Development image: `genespt/cuda-dev:pytorch2.1.2-cu118`
- Container Python: `3.10.13`
- PyTorch: `2.1.2`
- CUDA runtime in container: `11.8`
- Tested GPU visibility: one NVIDIA GPU is visible through Docker

The host driver can expose a newer CUDA version through `nvidia-smi`; that is
normal as long as the container CUDA runtime and PyTorch build stay compatible.

## Why This Setup

The previous VS Code workspace only selected a conda manager. That leaves three
possible runtimes competing with each other:

- Windows Python
- bare WSL Python at `/mnt/d/TESTWORK001`
- Docker Python at `/workspace/GeneSPT`

For this repository, Docker Python is the source of truth because it already has
the working CUDA/PyTorch stack and the spatial transcriptomics dependencies.

## Dependency Notes

`requirements.txt` records the direct non-PyTorch dependencies observed in the
working container. PyTorch is intentionally supplied by the Docker image. If the
image is changed, rerun:

```bash
python scripts/env_smoke_test.py
```

## Codex / Agent Rules

- Work from `/workspace/GeneSPT` inside the container for experiments.
- Keep `PYTHONPATH=/workspace/GeneSPT/main:/workspace/GeneSPT/scripts`.
- Do not run training from the Windows Python interpreter.
- Do not rewrite manuscript or experiment outputs as part of environment work.
- Keep long-running experiments behind an explicit queue or user request.
