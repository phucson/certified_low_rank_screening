"""Run provenance for the replication archive.

Every experiment writes its artefact through `dump`, which records the library
version, the git commit, and the full run configuration alongside the results,
so a reported number can be traced to the code and seeds that produced it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REQUIRED_VERSION = "1.2.0"

__all__ = ["assert_version", "git_commit", "dump"]


def assert_version() -> str:
    """Fail loudly if the installed library is not the pinned version."""
    from . import __version__
    if __version__ != REQUIRED_VERSION:
        raise RuntimeError(
            f"certproj {REQUIRED_VERSION} is required to reproduce the "
            f"reported numbers; found {__version__}")
    return __version__


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        return "unknown"


def dump(path: str, rows, config: dict) -> None:
    """Write results plus provenance, creating the results directory."""
    from . import __version__
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "certproj_version": __version__,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "config": config,
        "results": rows,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
