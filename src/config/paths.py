"""
Single source of truth for all resource paths.
Walks up from this file until it finds a known root marker
(pyproject.toml, setup.py, or .git), so it works regardless of
CWD or how deep the import chain is.
"""
from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Climb the directory tree until we find a root marker."""
    markers = {"pyproject.toml", "setup.py", "setup.cfg", ".git"}
    current = Path(__file__).resolve()

    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            return parent

    # Fallback: three levels up from src/config/paths.py → project root
    return current.parent.parent.parent


PROJECT_ROOT = _find_project_root()
RESOURCES    = PROJECT_ROOT / "resources"
LANGUAGES    = RESOURCES / "languages"
AGENTS       = RESOURCES / "agents"