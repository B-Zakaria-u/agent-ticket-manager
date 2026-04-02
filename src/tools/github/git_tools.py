"""Git tools — SRP: repository clone, branch management, and push only."""
import os
import re
import git
from langchain_core.tools import tool


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
    Clone the repository into ``workspace/`` if it does not exist yet,
    or reset it to exactly match the remote ``main`` branch if it does.
    Any local changes (staged, unstaged, stashed, or untracked) are discarded.

    Uses the GITHUB_TOKEN for authenticated HTTPS access.

    Args:
        repo_url: HTTPS clone URL (e.g. https://github.com/org/repo.git).
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    auth_url = repo_url.replace("https://", f"https://{token}@")

    workspace = _workspace_path(repo_url)
    try:
        if os.path.isdir(os.path.join(workspace, ".git")):
            repo = git.Repo(workspace)

            # Drop any stashed changes so nothing is hiding
            try:
                repo.git.stash("drop")
            except git.GitCommandError:
                pass  # No stash exists — that's fine

            # Fetch all remote changes
            repo.remotes.origin.fetch()

            # Hard reset to match remote main exactly
            repo.git.reset("--hard", "origin/main")

            # Remove untracked files and directories
            repo.git.clean("-fd")

            print(f"Project reset to origin/main at: {workspace}")
            return f"Reset {workspace} to match origin/main."
        else:
            git.Repo.clone_from(auth_url, workspace)
            print(f"Project cloned to absolute path: {workspace}")
            return f"Cloned {repo_url} into {workspace}."
    except git.GitCommandError as exc:
        return f"Git error: {exc}"


@tool
def create_branch(branch_name: str, repo_url: str = None) -> str:
    """
    Create and checkout a new local branch in the workspace repository.

    Args:
        branch_name: Branch name to create (e.g. ``fix/issue-42-add-guard``).
        repo_url: Optional HTTPS clone URL to locate the correct directory.
    """
    workspace = _workspace_path(repo_url)
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
def commit_and_push(commit_message: str, branch_name: str, repo_url: str = None) -> str:
    """
    Stage all changes in the workspace, commit them with the given message,
    and push the branch to the ``origin`` remote.

    Args:
        commit_message: Commit message (should reference the issue number).
        branch_name:    Remote branch to push to.
        repo_url:       Optional HTTPS clone URL to locate the correct directory.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    workspace = _workspace_path(repo_url)
    try:
        repo = git.Repo(workspace)
        repo.git.add(A=True)

        if not repo.is_dirty(index=True, working_tree=True, untracked_files=True):
            return "Nothing to commit — workspace is clean."

        repo.index.commit(commit_message)

        # Set authenticated remote URL before pushing
        origin = repo.remotes.origin
        repo_name = os.environ.get("GITHUB_REPOSITORY", "")
        origin.set_url(f"https://{token}@github.com/{repo_name}.git")
        origin.push(refspec=f"{branch_name}:{branch_name}")
        return f"Committed and pushed branch '{branch_name}' to origin."
    except git.GitCommandError as exc:
        return f"Git error on commit/push: {exc}"


def get_git_tools() -> list:
    """Return git operation LangChain tools."""
    return [clone_or_pull_repo, create_branch, commit_and_push]
