from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

from src.config.paths import LANGUAGES   # ← replaces the fragile relative path


ALIASES: dict[str, str] = {
    "python":     "python",
    "java":       "java",
    "kotlin":     "kotlin",
    "php":        "php",
    "javascript": "javascript",
    "typescript": "typescript",
    "go":         "go",
    "ruby":       "ruby",
    "c#":         "csharp",
    "csharp":     "csharp",
    "swift":      "swift",
    "rust":       "rust",
}


def _slug(language: str) -> str | None:
    return ALIASES.get(language.lower().strip())


@lru_cache(maxsize=None)
def _defaults() -> dict[str, Any]:
    path = LANGUAGES / "defaults.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=None)
def _load_hints(slug: str) -> dict[str, Any]:
    path = LANGUAGES / slug / "hints.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_hints(language: str) -> dict[str, Any]:
    slug = _slug(language)
    raw  = _load_hints(slug) if slug else {}
    return {**_defaults(), **raw}


def get_docker_image(language: str) -> str:
    return get_hints(language)["docker_image"]


def get_convention(language: str) -> str:
    return get_hints(language).get("convention", "")


def supported_languages() -> list[str]:
    return [
        entry.name
        for entry in LANGUAGES.iterdir()
        if entry.is_dir() and (entry / "hints.yaml").exists()
    ]