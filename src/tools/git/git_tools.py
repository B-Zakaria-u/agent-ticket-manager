"""Git tools — SRP: repository clone, branch management, and push only (Generic version)."""
import logging
import os
import re
import git
from langchain_core.tools import tool
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _get_auth_info(repo_url: str):
    """Detects provider and returns the appropriate token and host."""
    parsed = urlparse(repo_url)
    host = parsed.netloc
    
    # Map hosts to their respective environment variable tokens
    token_map = {
        "github.com": os.environ.get("GITHUB_TOKEN"),
        "gitlab.com": os.environ.get("GITLAB_TOKEN")
    }
    
    # Fallback for self-hosted or other instances
    token = token_map.get(host) or os.environ.get("GIT_TOKEN")
    return token, host

def _workspace_path(repo_url: str = None) -> str:
    base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "workspace")
    )
    if repo_url:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        return os.path.join(base, repo_name)
    return base

@tool
def clone_or_pull_repo(repo_url: str) -> str:
    """
    Clone the repository into ``workspace/`` or reset to match remote ``main``.
    Works with GitHub, GitLab, or other providers via env tokens.
    """
    token, _ = _get_auth_info(repo_url)
    if not token:
        logger.warning(f"No authentication token found for {repo_url}.")
        auth_url = repo_url
    else:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(repo_url)
        prefix = "oauth2:" if "gitlab" in parsed.netloc else ""
        auth_url = urlunparse(parsed._replace(netloc=f"{prefix}{token}@{parsed.netloc}"))

    workspace = _workspace_path(repo_url)
    try:
        if os.path.isdir(os.path.join(workspace, ".git")):
            repo = git.Repo(workspace)
            repo.git.stash("drop") if repo.git.stash("list") else None
            repo.remotes.origin.fetch()
            # Note: Hard reset to origin/main assumes 'main' as default. 
            repo.git.reset("--hard", "origin/main")
            repo.git.clean("-fd")
            return f"Reset {workspace} to match origin/main."
        else:
            git.Repo.clone_from(auth_url, workspace)
            return f"Cloned {repo_url} into {workspace}."
    except git.GitCommandError as exc:
        return f"Git error: {exc}"

@tool
def create_branch(branch_name: str, repo_url: str = None) -> str:
    """Create and checkout a new local branch."""
    workspace = _workspace_path(repo_url)
    
    # If no repo_url given, find the first valid git repo in workspace/
    if not repo_url:
        base = _workspace_path()
        subdirs = [os.path.join(base, d) for d in os.listdir(base)
                   if os.path.isdir(os.path.join(base, d, ".git"))]
        if not subdirs:
            return f"No git repository found in workspace: {base}"
        workspace = subdirs[0]  # pick the only/first repo

    try:
        repo = git.Repo(workspace)
        if any(h.name == branch_name for h in repo.heads):
            repo.heads[branch_name].checkout()
            return f"Checked out existing branch '{branch_name}'."
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()
        return f"Checked out new branch '{branch_name}'."
    except git.GitCommandError as exc:
        return f"Git error creating branch: {exc}"

@tool
def commit_and_push(
    commit_message: str,
    branch_name: str,
    repo_url: str = None,
    force: bool = False,
) -> str:
    """Stage, commit, and push changes using provider-specific authentication."""
    workspace = _workspace_path(repo_url)
    try:
        repo = git.Repo(workspace)
        repo.git.add(A=True)

        if not repo.is_dirty(index=True, working_tree=True, untracked_files=True):
            return "Nothing to commit — workspace is clean."

        repo.index.commit(commit_message)
        origin = repo.remotes.origin
        
        # Resolve auth dynamically for the push
        token, host = _get_auth_info(origin.url)
        
        if token:
            # Extract path (e.g., owner/repo) from current URL
            path = urlparse(origin.url).path.lstrip("/")
            # Use oauth2 for GitLab, standard token for GitHub
            prefix = "oauth2:" if "gitlab" in host else ""
            authenticated_url = f"https://{prefix}{token}@{host}/{path}"
            origin.set_url(authenticated_url)

        push_args = [f"{branch_name}:{branch_name}"]
        if force:
            push_args.insert(0, "--force")
            
        origin.push(refspec=push_args)
        return f"SUCCESS: Committed and pushed branch '{branch_name}' to {host}."
    except git.GitCommandError as exc:
        return f"FAILURE: Git error on commit/push: {exc}"

def get_git_tools() -> list:
    return [clone_or_pull_repo, create_branch, commit_and_push]