"""Environment smoke test for the paper-aligned GeneSPT package."""

from __future__ import annotations

import importlib


def main() -> None:
    for name in ["numpy", "pandas", "scipy", "sklearn", "torch", "yaml", "genespt"]:
        importlib.import_module(name)
        print(f"import_ok={name}")
    print("environment_smoke_test=ok")


if __name__ == "__main__":
    main()

