from __future__ import annotations

import importlib
import importlib.metadata
import sys
import warnings
from pathlib import Path


warnings.filterwarnings("ignore", category=FutureWarning)

REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("scanpy", "scanpy"),
    ("anndata", "anndata"),
    ("igraph", "igraph"),
    ("leidenalg", "leidenalg"),
    ("umap", "umap-learn"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("einops", "einops"),
    ("timm", "timm"),
    ("torch_geometric", "torch-geometric"),
    ("rdata", "rdata"),
    ("h5py", "h5py"),
    ("openpyxl", "openpyxl"),
    ("PIL", "Pillow"),
    ("statsmodels", "statsmodels"),
]


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"executable={sys.executable}")
    print(f"cwd={Path.cwd()}")

    try:
        import torch
    except Exception as exc:
        print(f"torch=MISSING:{exc!r}")
        return 1

    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"cuda_device_0={torch.cuda.get_device_name(0)}")

    missing = []
    for import_name, package_name in REQUIRED_MODULES:
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            missing.append(f"{package_name} ({import_name}): {exc!r}")
            continue
        print(f"{package_name}={package_version(package_name)}")

    if missing:
        print("missing_modules:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print("environment_smoke_test=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
