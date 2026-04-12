"""
src/agents/coding_agent.py

CodingAgent — simplified to use only spec + system/human prompts.
"""
from __future__ import annotations

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
    uses_tests = True

    # ── Required: context dict ────────────────────────────────────────────────

    async def build_context(self, state: GraphState) -> dict[str, Any]:
        profile  = state.get("model_profile", {})
        max_spec = int(profile.get("max_spec", 1_500))

        spec = state.get("spec", "")
        if not spec:
            raise ValueError(
                "CodingAgent requires a spec in state['spec']. "
                "Ensure SpecAgent has run successfully before CodingAgent."
            )

        spec_text = spec if len(spec) <= max_spec else spec[:max_spec] + "\n...[spec truncated]"

        return dict(spec_text=spec_text)

    # ── Required: native tools ────────────────────────────────────────────────

    async def get_tools(self, state: GraphState) -> list:
        workspace_dir = self._workspace_dir(state)
        language, _   = self._detect(state, workspace_dir)
        docker_image  = get_docker_image(language)

        @tool
        def run_tests() -> str:
            """Run the unit test suite inside a Docker sandbox."""
            return run_tests_in_sandbox.invoke({
                "workspace_path": workspace_dir,
                "image_name":     docker_image,
            })

        return get_file_tools(workspace_dir) + get_search_tools() + [run_tests]

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


# ── LangGraph node entrypoint ─────────────────────────────────────────────────

async def coding_agent_node(state: GraphState) -> dict:
    return await CodingAgent().run(state)