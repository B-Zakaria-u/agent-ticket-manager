"""
src/agents/spec_agent.py

SpecAgent — generates a technical specification from a ticket.

v3: build_context() now pre-loads the workspace snapshot locally via
    WorkspaceReader, so the LLM receives all relevant source files in
    the first prompt and does NOT need to call read_file iteratively.
    This eliminates N LLM round-trips (one per file) and cuts latency
    dramatically for typical repos.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent
from src.config.agent_config import AgentConfig
from src.state import AgentReport, GraphState
from src.tools.files import get_file_tools
from src.tools.search import get_search_tools
from src.utils.language_detector import detect_language
from src.utils.workspace_reader import read_workspace   # ← new local reader


class SpecAgent(BaseAgent):

    agent_name = "spec_agent"
    uses_tests = False   # produces a spec document, does not run tests

    def __init__(self) -> None:
        super().__init__()
        self._spec_text: str = ""

    # ── Required: context dict ────────────────────────────────────────────────

    async def build_context(self, state: GraphState) -> dict[str, Any]:
        workspace_dir         = self._workspace_dir(state)
        workspace_lang, ws_fw = self._detect(state, workspace_dir)
        self._cached_workspace = workspace_dir
        self._cached_lang      = workspace_lang
        self._cached_fw        = ws_fw

        profile    = state.get("model_profile", {})
        max_ticket = int(profile.get("max_test_out", 800))
        cfg        = AgentConfig(self.agent_name)
        ticket     = state.get("ticket_text", "")

        # ── Local workspace snapshot (pure Python, no LLM calls) ─────────────
        # budget_chars controls how much source code ends up in the prompt.
        # Tune this against your model's context window.
        snapshot = read_workspace(
            workspace_dir,
            ticket_text  = ticket,
            max_files    = int(profile.get("max_files", 60)),
            budget_chars = int(profile.get("workspace_budget_chars", 40_000)),
        )
        workspace_snapshot = snapshot.render()

        return dict(
            workspace_language  = workspace_lang,
            workspace_framework = ws_fw,
            lang_note           = cfg.lang_note({
                "detected_language":  workspace_lang,
                "detected_framework": ws_fw,
            }),
            # Legacy short file list kept for compact prompt fallback
            file_list_str = self._file_list(
                workspace_dir, int(profile.get("max_files", 30))
            ),
            ticket_text = (
                ticket if len(ticket) <= max_ticket
                else ticket[:max_ticket] + "\n...[ticket truncated]"
            ),
            # ── NEW: full file contents, pre-loaded locally ───────────────────
            workspace_snapshot  = workspace_snapshot,
            snapshot_file_count = len(snapshot.files),
            snapshot_budget_hit = snapshot.budget_reached,
        )

    # ── Required: native tools ────────────────────────────────────────────────

    async def get_tools(self, state: GraphState) -> list:
        workspace_dir = self._workspace_dir(state)
        agent = self   # capture for closure

        @tool
        def submit_spec(spec_text: str) -> str:
            """Submit the final technical specification. Call this exactly once
            with the full specification text when you are finished."""
            agent._spec_text = spec_text
            agent._files_written = True
            return "Specification submitted successfully."

        # read_file is still available as a fallback for files outside the
        # snapshot (e.g. very large files, or files added after snapshot was
        # taken), but the LLM should rarely need it now.
        return get_file_tools(workspace_dir) + get_search_tools() + [submit_spec]

    # ── Optional: enrich the AgentReport ─────────────────────────────────────

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
        base["metadata"] = {
            "language":  self._cached_lang,
            "framework": self._cached_fw,
            "workspace": self._cached_workspace,
            "spec_length": len(self._spec_text),
        }
        return base

    # ── Optional: extra state keys ────────────────────────────────────────────

    async def extra_state_updates(self, state: GraphState) -> dict[str, Any]:
        return {"spec": self._spec_text}

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
        ignore = {
            ".git", "__pycache__", "node_modules", ".venv",
            "venv", "env", "target", "build", "dist", ".gradle",
        }
        files = []
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

async def spec_agent_node(state: GraphState) -> dict:
    return await SpecAgent().run(state)