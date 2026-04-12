"""Issue Scout Agent — SRP: GitHub/GitLab issue discovery, assignment, and repo setup."""
from __future__ import annotations

import logging
import os
import re
from typing import Any

# Import both clients
import gitlab
from github import Github
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from src.agents.base_agent import BaseAgent
from src.state import AgentReport, GraphState
# Generic git tools
from src.tools.git.git_tools import clone_or_pull_repo, create_branch

# Provider-specific issue tools
REPOSITORY_TYPE = os.environ.get("REMOTE_REPOSITORY", "GITLAB").upper()

if REPOSITORY_TYPE == "GITLAB":
    from src.tools.gitlab.issue_tools import assign_issue, list_open_issues
else:
    from src.tools.github.issue_tools import assign_issue, list_open_issues


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert a string to a URL-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len]


class IssueScoutAgent(BaseAgent):
    """Autonomous issue picker and repository bootstrapper for GitHub and GitLab."""

    agent_name = "issue_scout"
    uses_tests = False

    def __init__(self) -> None:
        super().__init__()
        self._issue_number: int = 0
        self._branch_name: str = ""
        self._repo_url: str = ""
        self._ticket_text: str = ""
        self._completed: bool = False

    async def build_context(self, state: GraphState) -> dict[str, Any]:
        return dict()

    async def get_tools(self, state: GraphState) -> list:
        agent = self
            
        @tool
        def submit_scout_results(issue_number: int) -> str:
            """Submit the scout run results after picking an issue, assigning it, 
            cloning the repo, and creating a new branch.
            """
            # At the top of submit_scout_results, before any API call:
            if issue_number <= 0:
                agent._completed = False
                return "Scout aborted: no unassigned issues available in the project."

            repo_type = REPOSITORY_TYPE
            
            if repo_type == "GITLAB":
                token = os.environ.get("GITLAB_TOKEN")
                repo_id = os.environ.get("GITLAB_REPOSITORY") # GitLab often uses ID or path
                url = os.environ.get("GITLAB_URL", "https://gitlab.com")  #Change to your gitlab url if not using gitlab.com
            else:
                token = os.environ.get("GITHUB_TOKEN")
                repo_id = os.environ.get("GITHUB_REPOSITORY")

            if not token or not repo_id:
                return f"{repo_type}_TOKEN or project identifier is missing."

            try:
                if repo_type == "GITLAB":
                    gl = gitlab.Gitlab(url, private_token=token)
                    gl.auth()
                    project = gl.projects.get(repo_id)
                    issue = project.issues.get(issue_number)
                    
                    # GitLab specific mapping
                    agent._ticket_text = f"#{issue.iid} — {issue.title}\n\n{issue.description or ''}"
                    agent._repo_url = project.http_url_to_repo
                else:
                    gh = Github(token)
                    repo = gh.get_repo(repo_id)
                    issue = repo.get_issue(issue_number)
                    
                    # GitHub specific mapping
                    agent._ticket_text = f"#{issue.number} — {issue.title}\n\n{issue.body or ''}"
                    agent._repo_url = repo.clone_url

                agent._issue_number = issue_number
                slug = _slugify(issue.title)
                agent._branch_name = f"fix/issue-{issue_number}-{slug}"
                agent._completed = True
                agent._files_written = True 
                
                
                return f"""
                Scout results successfully submitted for issue #{issue_number}.
                The current remote repository is {agent._repo_url}.
                The current branch is {agent._branch_name}.
                The current ticket text is {agent._ticket_text}.
                """

            except Exception as e:
                logger.error(f"Error fetching issue details: {e}", exc_info=True)
                return f"Error fetching issue details: {e}"

        return [
            list_open_issues,
            assign_issue,
            clone_or_pull_repo,
            create_branch,
            submit_scout_results,
        ]

    async def extra_state_updates(self, state: GraphState) -> dict[str, Any]:
        return {
            "ticket_text": self._ticket_text,
            "issue_number": self._issue_number,
            "branch_name": self._branch_name,
            "repo_url": self._repo_url,
        }

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
            base["summary"] = f"Successfully scouted issue #{self._issue_number}."
            base["status"] = "success"
        else:
            base["status"] = "failed"
        return base

async def issue_scout_node(state: GraphState) -> dict:
    return await IssueScoutAgent().run(state)