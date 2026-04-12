"""GitLab issue tools — SRP: issue discovery and self-assignment only."""
import os
import gitlab
from langchain_core.tools import tool

def _get_client() -> gitlab.Gitlab:
    token = os.environ.get("GITLAB_TOKEN")
    url = os.environ.get("GITLAB_URL", "https://gitlab.com")
    print(f"Client working on {url}")
    if not token:
        raise EnvironmentError("GITLAB_TOKEN environment variable is not set.")
    
    gl = gitlab.Gitlab(url, private_token=token)
    gl.auth()  # Validates the token and populates gl.user
    return gl

def _get_project(client: gitlab.Gitlab):
    # GitLab uses "Project Path" (e.g., 'username/repo' or numeric ID)
    project_id = os.environ.get("GITLAB_REPOSITORY")
    if not project_id:
        raise EnvironmentError("GITLAB_REPOSITORY environment variable is not set.")
    return client.projects.get(project_id)

@tool
def list_open_issues(max_results: int = 10) -> str:
    """
    Fetch open, unassigned issues from the configured GitLab project.

    Returns a formatted list of issue IIDs, titles, and descriptions.

    Args:
        max_results: Maximum number of issues to return (default 10).
    """
    try:
        client = _get_client()
        project = _get_project(client)
        
        # 1. Fetch ALL open issues first (without assignee filter) to see what's there
        # We increase per_page slightly to ensure we don't miss recent ones
        all_open = project.issues.list(
            state='opened',
            get_all=False,
            per_page=50
        )
        
        print(f"[DEBUG] GitLab API found {len(all_open)} open issues total (top 50).")
        
        # 2. Filter for unassigned issues locally for maximum reliability
        issues = [i for i in all_open if not i.assignee]
        
        print(f"[DEBUG] Found {len(issues)} unassigned issue(s) after local filtering.")
        
        if not issues:
            debug_info = f"No open unassigned issues found. (Total open: {len(all_open)})"
            if all_open:
                debug_info += "\nFound these open assigned issues:"
                for i in all_open:
                    assignee = i.assignee['username'] if i.assignee else "None"

                    if i.assignee and i.assignee['id'] == client.user.id:
                        debug_info += f"\n  - #{i.iid}: {i.title} (Assignee: {assignee})"
                        issues.append(i)
                    else:
                        debug_info += f"\n  - #{i.iid}: {i.title} (Assignee: {assignee})"
                        return debug_info

        # Limit to max_results after filtering
        issues = issues[:max_results]

        lines = [f"Found {len(issues)} open unassigned issue(s) in GitLab:\n"]
        for issue in issues:
            desc_preview = (issue.description or "")[:120].replace("\n", " ")
            lines.append(f"  #{issue.iid} — {issue.title}\n    {desc_preview}")
        return "\n".join(lines)

    except gitlab.exceptions.GitlabError as exc:
        return f"GitLab API error: {exc.response_body if hasattr(exc, 'response_body') else exc}"
    except Exception as exc:
        return f"Error listing issues: {exc}"

@tool
def assign_issue(issue_iid: int) -> str:
    """
    Self-assign a GitLab issue to the account that owns the GITLAB_TOKEN.

    Args:
        issue_iid: The GitLab issue IID (internal ID) to assign.
    """
    try:
        client = _get_client()
        current_user_id = client.user.id
        project = _get_project(client)
        
        # GitLab identifies issues by their IID within a project
        issue = project.issues.get(issue_iid)

        # Update the assignee
        issue.assignee_ids = [current_user_id]
        issue.save()
        
       # In assign_issue, make the instruction impossible to miss:
        return (
            f"Issue #{issue_iid} assigned to @{client.user.username}. "
            f"IMPORTANT: Use this exact URL for the next step — repo_url={project.http_url_to_repo}"
        )

    except gitlab.exceptions.GitlabError as exc:
        return f"GitLab API error: {exc.response_body if hasattr(exc, 'response_body') else exc}"
    except Exception as exc:
        return f"Error assigning issue: {exc}"

def get_issue_tools() -> list:
    """Return GitLab issue LangChain tools."""
    return [list_open_issues, assign_issue]