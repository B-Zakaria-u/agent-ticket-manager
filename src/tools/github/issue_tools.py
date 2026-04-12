"""GitHub issue tools — SRP: issue discovery and self-assignment only."""
import os
from github import Github, GithubException
from langchain_core.tools import tool


def _get_client() -> Github:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError("GITHUB_TOKEN environment variable is not set.")
    return Github(token)


def _get_repo(client: Github):
    repo_name = os.environ.get("GITHUB_REPOSITORY")
    if not repo_name:
        raise EnvironmentError("GITHUB_REPOSITORY environment variable is not set.")
    return client.get_repo(repo_name)


@tool
def list_open_issues(max_results: int = 10) -> str:
    """
    Fetch open, unassigned issues from the configured GitHub repository.

    Returns a formatted list of issue numbers, titles, and body previews
    so the agent can pick the most relevant one to work on.

    Args:
        max_results: Maximum number of issues to return (default 10).
    """
    try:
        client = _get_client()
        app_user = client.get_user().login
        repo = _get_repo(client)
        
        issues = []
        for i in repo.get_issues(state="open"):
            if i.pull_request:
                continue
            # Include if unassigned OR if already assigned to our bot
            if not i.assignees or any(a.login == app_user for a in i.assignees):
                issues.append(i)
            if len(issues) >= max_results:
                break

        if not issues:
            return "No open unassigned issues found."

        lines = [f"Found {len(issues)} open unassigned issue(s):\n"]
        for issue in issues:
            body_preview = (issue.body or "")[:120].replace("\n", " ")
            lines.append(f"  #{issue.number} — {issue.title}\n    {body_preview}")
        return "\n".join(lines)

    except GithubException as exc:
        return f"GitHub API error: {exc.data}"
    except Exception as exc:
        return f"Error listing issues: {exc}"


@tool
def assign_issue(issue_number: int) -> str:
    """
    Self-assign a GitHub issue to the account that owns the GITHUB_TOKEN.

    The assignee is resolved automatically from the authenticated token,
    so no ``GITHUB_ASSIGNEE`` environment variable is required.

    Args:
        issue_number: The GitHub issue number to assign.
    """
    try:
        client = _get_client()
        # Resolve the app's own identity from the token
        app_user = client.get_user().login
        repo = _get_repo(client)
        issue = repo.get_issue(issue_number)
        if issue.assignees and any(a.login == app_user for a in issue.assignees):
            return f"Issue #{issue_number} is already assigned to you {app_user}."
        issue.add_to_assignees(app_user)
        # In assign_issue, update the return string:
        return (
            f"Issue #{issue_number} successfully assigned to @{client.user.username}. "
            f"Repository URL: {repo.http_url_to_repo}"
        )

    except GithubException as exc:
        return f"GitHub API error: {exc.data}"
    except Exception as exc:
        return f"Error assigning issue: {exc}"


def get_issue_tools() -> list:
    """Return GitHub issue LangChain tools."""
    return [list_open_issues, assign_issue]
