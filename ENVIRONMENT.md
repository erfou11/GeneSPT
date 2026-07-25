# GeneSPT Environment

The public Docker environment is defined by the repository-root `Dockerfile`
and `compose.yaml`. The stable project path inside the container is:

```text
/workspace/GeneSPT
```

## Repository-root entry point

Run these commands from the cloned public repository root on Windows, Linux,
or macOS:

```bash
docker compose build genespt
docker compose run --rm genespt python scripts/env_smoke_test.py
```

Compose bind-mounts the repository root at `/workspace/GeneSPT`, so the checked
out scripts are available even though the image layer contains dependencies
only. The repository-root Docker and Compose files are the complete public
container contract.

## Current CUDA Baseline

- Docker image: `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`
- Compose image: `genespt/public:pytorch2.1.2-cu118`
- Container Python: `3.10.13`
- PyTorch: `2.1.2`
- CUDA runtime in container: `11.8`
- The no-data smoke command is valid without a GPU; CUDA visibility is reported
  rather than required

The host driver can expose a newer CUDA version through `nvidia-smi`; that is
normal as long as the container CUDA runtime and PyTorch build stay compatible.

## Runtime contract

Docker Python is the documented source of truth for the pinned manuscript
environment. The default Compose service deliberately does not reserve a GPU,
which keeps the no-data smoke check runnable on CPU-only reviewer machines.
Matrix-level archive verification is also CPU-only but requires the extracted
data archive. GPU-backed training requires a host with NVIDIA Container Toolkit and
an explicit local GPU reservation; it is outside the no-data smoke contract.

## Dependency Notes

`requirements.txt` records the direct non-PyTorch dependencies observed in the
working container. PyTorch is intentionally supplied by the Docker image. If the
image is changed, rerun:

```bash
python scripts/env_smoke_test.py
```

## Paths inside the container

- Repository root: `/workspace/GeneSPT`
- Public package: `/workspace/GeneSPT/src`
- Experiment modules: `/workspace/GeneSPT/main`
- Reproduction scripts: `/workspace/GeneSPT/scripts`

Compose sets all three source directories on `PYTHONPATH`. Generated outputs
belong under the repository-relative `results/` or `figures/` mounts.
