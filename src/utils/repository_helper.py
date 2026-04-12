import os
from urllib.parse import urlparse

class RepositoryHelper:
    @staticmethod
    def _get_token_for_host(url: str) -> str:
        """
        Private helper to grab the right token based on the URL.
        The Agent's environment provides these, but the Tool retrieves them.
        """
        host = urlparse(url).netloc
        if "github.com" in host:
            return os.environ.get("GITHUB_TOKEN", "")
        elif "gitlab.com" in host:
            return os.environ.get("GITLAB_TOKEN", "")
        return os.environ.get("GIT_TOKEN", "") # Generic fallback

    @staticmethod
    def get_git_provider(url: str = None) -> str:
        """
        Detect the Git provider from the repository URL.
        """
        if url is None:
            url = os.environ.get("REMOTE_REPOSITORY", "gitlab")
        host = urlparse(url).netloc
        if "github.com" in host:
            return "github"
        elif "gitlab.com" in host:
            return "gitlab"
        return "generic"
