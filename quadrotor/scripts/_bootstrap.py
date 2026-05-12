"""Repo and local script path setup for quadrotor/scripts entry points."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent

for path in (REPO_ROOT, SCRIPTS_DIR):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
