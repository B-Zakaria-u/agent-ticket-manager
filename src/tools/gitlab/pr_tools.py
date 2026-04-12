"""GitLab PR tools — SRP: merge request creation only."""
import os
import gitlab
from langchain_core.tools import tool


@tool
def create_pull_request(
    branch_name: str,
    title: str,
    body: str,
    base_branch: str = "main",
) -> str:
    """
    Open a Pull Request from *branch_name* into *base_branch*.

    Returns the URL of the created PR.

    Args:
        branch_name:  Head branch with the fix (e.g. ``fix/issue-42-add-guard``).
        title:        PR title (typically mirrors the issue title).
        body:         PR description — include a ``Closes #N`` reference so
                      Gitlab auto-closes the issue when the PR merges.
        base_branch:  Target branch (default ``main``).
    """
    try:
        token = os.environ.get("GITLAB_TOKEN")
        if not token:
            raise EnvironmentError("GITLAB_TOKEN environment variable is not set.")

        project_id = os.environ.get("GITLAB_REPOSITORY")
        if not project_id:
            raise EnvironmentError("GITLAB_REPOSITORY environment variable is not set.")

        url = os.environ.get("GITLAB_URL", "https://gitlab.com")
        gl = gitlab.Gitlab(url=url, private_token=token)
        gl.auth()
        
        project = gl.projects.get(project_id)
        mr = project.mergerequests.create({
            'source_branch': branch_name,
            'target_branch': base_branch,
            'title': title,
            'description': body,
        })
        return f"Merge Request opened: {mr.web_url}"

    except gitlab.exceptions.GitlabError as exc:
        msg = exc.response_body if hasattr(exc, "response_body") else str(exc)
        return f"GitLab API error: {msg}"
    except Exception as exc:
        return f"Error creating MR: {exc}"


def get_pr_tools() -> list:
    """Return Gitlab PR LangChain tools."""
    return [create_pull_request]
