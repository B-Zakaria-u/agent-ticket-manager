from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent
from src.config.agent_config import AgentConfig
from src.state import AgentReport, GraphState
from src.tools.git.git_tools import commit_and_push

# Provider-specific PR tools
REPOSITORY_TYPE = os.environ.get("REMOTE_REPOSITORY", "GITLAB").upper()

if REPOSITORY_TYPE == "GITLAB":
    from src.tools.gitlab.pr_tools import create_pull_request
else:
    from src.tools.github.pr_tools import create_pull_request



class PRAgent(BaseAgent):
    """Commits and pushes the fix, then opens a Pull Request."""


    agent_name = "pr_agent"
    uses_tests = False  # just creates the PR

    def __init__(self) -> None:
        super().__init__()
        self._pr_url: str = ""
        self._completed: bool = False

    # ── Required: context dict ────────────────────────────────────────────────

    async def build_context(self, state: GraphState) -> dict[str, Any]:
        return dict(
            ticket_text=state.get("ticket_text", ""),
            issue_number=state.get("issue_number", 0),
            branch_name=state.get("branch_name", "fix/automated"),
            repo_url=state.get("repo_url", ""),
        )

    # ── Required: native tools ────────────────────────────────────────────────

    async def get_tools(self, state: GraphState) -> list:
        agent = self

        @tool
        def submit_pr_results(pr_url: str) -> str:
            """Submit the final Pull Request URL. Call this exactly once after 
            successfully committing, pushing, and creating the pull request.
            """
            agent._pr_url = pr_url
            agent._completed = True
            agent._files_written = True
            return "PR URL saved successfully. Ready to finish."

        return [commit_and_push, create_pull_request, submit_pr_results]

    # ── Optional: extra state keys ────────────────────────────────────────────

    async def extra_state_updates(self, state: GraphState) -> dict[str, Any]:
        return {
            "pr_url": self._pr_url,
        }

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
        if self._completed:
            base["summary"] = f"Created Pull Request: {self._pr_url}"
            base["status"] = "success"
        else:
            base["summary"] = "Failed to create Pull Request."
            base["status"] = "failed"
            
        return base


# LangGraph node callable
async def pr_agent_node(state: GraphState) -> dict:
    return await PRAgent().run(state)

