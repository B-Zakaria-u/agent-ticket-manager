"""
src/agents/testing_agent.py

TestingAgent — uses_tests=False (it writes tests but does not run them).
Updated to async build_context / get_tools / extra_state_updates for v2 BaseAgent.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

from src.agents.base_agent import BaseAgent
from src.config.paths import AGENTS
from src.state import AgentReport, GraphState
from src.tools.files import get_file_tools
from src.tools.folders import initiate_directory
from src.utils.language_detector import detect_language


@lru_cache(maxsize=None)
def _qa_tools() -> dict[str, dict]:
    path = AGENTS / "testing_agent" / "prompts" / "qa_tools.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=None)
def _language_runners() -> dict[str, dict]:
    path = AGENTS / "testing_agent" / "prompts" / "language_runners.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_catalogue(verbose: bool) -> str:
    tools = _qa_tools()
    if verbose:
        return "\n\n".join(
            f"[{key}]\n"
            f"  Concern   : {meta['concern']}\n"
            f"  Install   : {meta['install']}\n"
            f"  script.sh :\n"
            + "\n".join(f"    {line}" for line in meta["script_hint"].splitlines())
            for key, meta in tools.items()
        )
    return "\n".join(
        f"  • [{key}] {meta['concern']}" for key, meta in tools.items()
    )


def _build_catalogue_full() -> str:
    return "\n\n".join(
        f"[{key}]\n Concern: {meta['concern']}\n Install: {meta['install']}\n script.sh :\n"
        + "\n".join(f"    {line}" for line in meta["script_hint"].splitlines())
        for key, meta in _qa_tools().items()
    )


class TestingAgent(BaseAgent):

    agent_name = "testing_agent"
    uses_tests = False   # writes tests, does not execute them → no test nudge

    # ── Required: context dict ────────────────────────────────────────────────

    async def build_context(self, state: GraphState) -> dict[str, Any]:
        workspace_dir       = self._workspace_dir(state)
        language, framework = self._detect(state, workspace_dir)
        # Cache for reuse in get_tools / build_report / extra_state_updates
        self._cached_workspace = workspace_dir
        self._cached_lang      = language
        self._cached_fw        = framework

        profile             = state.get("model_profile", {})
        verbose             = bool(profile.get("system_verbose"))
        max_spec            = int(profile.get("max_spec",     1_500))
        max_ticket          = int(profile.get("max_test_out", 800))

        ticket = state.get("ticket_text", "")
        spec   = state.get("spec", "")

        return dict(
            detected_language   = language,
            detected_framework  = framework,
            tool_catalogue      = _build_catalogue(verbose),
            tool_catalogue_full = _build_catalogue_full(),
            file_list_str       = self._file_list(
                workspace_dir, int(profile.get("max_files", 30))
            ),
            ticket_text = (
                ticket if len(ticket) <= max_ticket
                else ticket[:max_ticket] + "\n...[ticket truncated]"
            ),
            spec_text = (
                spec if len(spec) <= max_spec
                else spec[:max_spec] + "\n...[spec truncated]"
            ),
        )

    # ── Required: native tools ────────────────────────────────────────────────

    async def get_tools(self, state: GraphState) -> list:
        workspace_dir = self._workspace_dir(state)
        initiate_directory(workspace_dir)
        return get_file_tools(workspace_dir)

    # ── Optional: enrich AgentReport ─────────────────────────────────────────

    async def build_report(
        self,
        *,
        status: str,
        summary: str,
        state: GraphState,
        tokens: int,
    ) -> AgentReport:
        base = await super().build_report(
            status=status, summary=summary, state=state, tokens=tokens,
        )
        base["tests_generated"] = self._files_written
        return base

    # ── Optional: extra state keys ────────────────────────────────────────────

    async def extra_state_updates(self, state: GraphState) -> dict[str, Any]:
        return {
            "tests_generated":    self._files_written,
            "detected_language":  self._cached_lang,
            "detected_framework": self._cached_fw,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _workspace_dir(state: GraphState) -> str:
        from src.config.paths import PROJECT_ROOT
        repo_url = state.get("repo_url", "")
        base     = PROJECT_ROOT / "workspace"
        return str(
            base / repo_url.split("/")[-1].replace(".git", "")
            if repo_url else base
        )

    @staticmethod
    def _detect(state: GraphState, workspace_dir: str) -> tuple[str, str]:
        language  = state.get("detected_language") or ""
        framework = state.get("detected_framework") or ""
        if not language or language == "Unknown":
            info      = detect_language(workspace_dir)
            language  = info.get("language", "Unknown")
            framework = info.get("framework", "Unknown")
        return language, framework

    @staticmethod
    def _file_list(workspace_dir: str, max_files: int) -> str:
        ignore = {".git", "__pycache__", "node_modules", ".venv"}
        files  = []
        if os.path.exists(workspace_dir):
            for root, dirs, fs in os.walk(workspace_dir):
                dirs[:] = [d for d in dirs if d not in ignore]
                for f in fs:
                    files.append(
                        os.path.relpath(os.path.join(root, f), workspace_dir)
                    )
        if len(files) > max_files:
            files = files[:max_files] + [
                f"... ({len(files) - max_files} more not shown)"
            ]
        return "\n".join(f"- {f}" for f in files)


# ── LangGraph node entrypoint ─────────────────────────────────────────────────

async def testing_agent_node(state: GraphState) -> dict:
    return await TestingAgent().run(state)