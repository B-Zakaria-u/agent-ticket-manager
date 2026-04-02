"""GitHub issue tools — SRP: issue discovery and self-assignment only."""
import os
from github import Github, GithubException
from langchain_core.tools import tool

