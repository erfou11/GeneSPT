#!/usr/bin/env python3
# Replot final manuscript figures from prepared source CSVs.
# Usage: python /workspace/GeneSPT/main/prepare_final_manuscript_package.py
# The preparation script is the canonical plotting entry point because it
# rebuilds source CSVs and figures together from the benchmark tables.
from pathlib import Path
import subprocess
subprocess.check_call(["python", str(Path(__file__).with_name("prepare_final_manuscript_package.py"))])
