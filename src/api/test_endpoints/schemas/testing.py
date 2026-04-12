"""Pydantic request/response schemas for the testing API."""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AgentTestRequest(BaseModel):
    """Payload accepted by POST /test/{agent_name}."""

    state: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="A partial GraphState to initialize the agent with.",
    )
    mock_files: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="A dictionary mapping relative file paths to their contents, to be created in the workspace.",
    )
    cleanup: bool = Field(
        default=True,
        description="Whether to delete the mock workspace after the agent completes.",
    )


class AgentTestResponse(BaseModel):
    """Aggregate result from an individual agent test run."""

    status: str = Field(description="The final status reported by the agent (e.g., 'success', 'failed').")
    summary: str = Field(description="A brief summary of the agent's work.")
    state_updates: Dict[str, Any] = Field(description="The keys and values updated in the state by the agent.")
    artifacts: list[str] = Field(description="List of files written by the agent.")
    issues: list[str] = Field(description="List of issues or errors reported by the agent.")
    tokens: int = Field(description="Total tokens used during the test run.")
    workspace_path: str = Field(description="The path to the workspace used for the test.")
