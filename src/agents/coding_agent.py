"""
src/agents/coding_agent.py

CodingAgent — unchanged domain logic, updated to use the v2 BaseAgent
(async build_context, async get_tools, async extra_state_updates).
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent
from src.config.agent_config import AgentConfig
from src.config.language_config import get_docker_image, get_hints
from src.state import AgentReport, GraphState
from src.tools.docker.sandbox import run_tests_in_sandbox
from src.tools.files import get_file_tools
from src.tools.search import get_search_tools
from src.utils.language_detector import detect_language


class CodingAgent(BaseAgent):

    agent_name = "coding_agent"
    uses_tests = True   # participates in the "run tests" nudge

    # ── Required: context dict ────────────────────────────────────────────────

    async def build_context(self, state: GraphState) -> dict[str, Any]:
        workspace_dir       = self._workspace_dir(state)
        language, framework = self._detect(state, workspace_dir)
        # Cache for reuse in get_tools / build_report / extra_state_updates
        self._cached_workspace = workspace_dir
        self._cached_lang      = language
        self._cached_fw        = framework

        hints               = get_hints(language)
        profile             = state.get("model_profile", {})
        max_spec            = int(profile.get("max_spec", 1_500))
        cfg                 = AgentConfig(self.agent_name)

        spec = state.get("spec", "")
        if not spec:
            raise ValueError(
                "CodingAgent requires a spec in state['spec']. "
                "Ensure SpecAgent has run successfully before CodingAgent."
            )
        spec_text = spec if len(spec) <= max_spec else spec[:max_spec] + "\n...[spec truncated]"

        return dict(
            detected_language  = language,
            detected_framework = framework,
            test_framework     = hints["framework"],
            file_pattern       = hints["file_pattern"],
            script_hint        = hints["script_hint"],
            lang_conventions   = hints["convention"],
            lang_note          = cfg.lang_note({
                "detected_language":  language,
                "detected_framework": framework,
                "lang_conventions":   hints["convention"],
            }),
            file_list_str      = self._file_list(
                workspace_dir, int(profile.get("max_files", 30))
            ),
            spec_text          = spec_text,
        )

    # ── Required: native tools ────────────────────────────────────────────────

    async def get_tools(self, state: GraphState) -> list:
        workspace_dir = self._cached_workspace
        language      = self._cached_lang
        docker_image  = get_docker_image(language)

        @tool
        def run_tests() -> str:
            """Run the unit test suite inside a Docker sandbox."""
            return run_tests_in_sandbox.invoke({
                "workspace_path": workspace_dir,
                "image_name":     docker_image,
            })

        return get_file_tools(workspace_dir) + get_search_tools() + [run_tests]

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
        }
        return base

    # ── Optional: extra state keys ────────────────────────────────────────────

    async def extra_state_updates(self, state: GraphState) -> dict[str, Any]:
        return {
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

        # Always re-detect if the spec is present — the spec may target
        # a different stack than the existing workspace files.
        spec = state.get("spec", "")
        if spec:
            info      = detect_language(workspace_dir, hint=spec)
            language  = info.get("language", language or "Unknown")
            framework = info.get("framework", framework or "Unknown")
        elif not language or language == "Unknown":
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

        infrastructure_files = {"script.sh", "Dockerfile", "docker-compose.yml"}

        files = []
        if os.path.exists(workspace_dir):
            for root, dirs, fs in os.walk(workspace_dir):
                dirs[:] = [d for d in dirs if d not in ignore]
                for f in fs:
                    rel = os.path.relpath(os.path.join(root, f), workspace_dir)
                    if f not in infrastructure_files:
                        files.append(rel)
                        
        if len(files) > max_files:
            files = files[:max_files] + [
                f"... ({len(files) - max_files} more not shown)"
            ]
        return "\n".join(f"- {f}" for f in files)


# ── LangGraph node entrypoint ─────────────────────────────────────────────────

async def coding_agent_node(state: GraphState) -> dict:
    return await CodingAgent().run(state)