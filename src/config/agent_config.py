from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from src.config.paths import AGENTS   # ← replaces the fragile relative path


class AgentConfig:

    def __init__(self, agent_name: str) -> None:
        self._prompts_dir = AGENTS / agent_name / "prompts"
        self._cache: dict[str, dict] = {}

    def _load(self, filename: str) -> dict:
        if filename not in self._cache:
            path = self._prompts_dir / filename
            with open(path, encoding="utf-8") as fh:
                self._cache[filename] = yaml.safe_load(fh)
        return self._cache[filename]

    @staticmethod
    def _render(template: str, ctx: dict[str, Any]) -> str:
        class _Safe(dict):
            def __missing__(self, key: str) -> str:
                return f"{{{{key}}}}"
        return template.format_map(_Safe(ctx))

    def _r(self, filename: str, key: str, ctx: dict[str, Any]) -> str:
        return self._render(self._load(filename)[key], ctx)

    def system(self, verbose: bool, ctx: dict[str, Any]) -> str:
        return self._r("system.yaml", "verbose" if verbose else "compact", ctx)

    def human(self, verbose: bool, ctx: dict[str, Any]) -> str:
        return self._r("human.yaml", "verbose" if verbose else "compact", ctx)

    def nudge(self, verbose: bool, kind: str = "default") -> str:
        cfg = self._load("nudges.yaml")
        if kind == "tests_not_run":
            return cfg["tests_not_run"]
        return cfg["verbose"] if verbose else cfg["compact"]

    def test_output_suffix(self, test_output: str, max_len: int) -> str:
        cfg  = self._load("suffixes.yaml")
        tail = test_output[-max_len:]
        result = self._render(cfg["base"], {"test_output_tail": tail})
        result += cfg["sandbox_error"] if "[SANDBOX FAIL]" in test_output and "Docker" in test_output else cfg["fix_hint"]
        return result

    def lang_note(self, ctx: dict[str, Any]) -> str:
        if ctx.get("detected_language", "Unknown") == "Unknown":
            return ""
        cfg = self._load("lang_note.yaml")
        framework_suffix = (
            self._render(cfg["framework_suffix"], ctx)
            if ctx.get("detected_framework", "Unknown") != "Unknown"
            else ""
        )
        return self._render(cfg["template"], {**ctx, "framework_suffix": framework_suffix})