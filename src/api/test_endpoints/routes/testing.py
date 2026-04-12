"""Testing routes — SRP: Testing individual agents in isolation."""
import os
import shutil
import uuid
from typing import Any, Dict, Type
from pathlib import Path

from fastapi import APIRouter, HTTPException
from src.api.test_endpoints.schemas.testing import AgentTestRequest, AgentTestResponse
from src.state import GraphState
from src.config.paths import PROJECT_ROOT
from src.utils.logger import log_request_start
from src.llm.base_config import (
    MODEL_PROFILE_STANDARD,
    MODEL_PROFILE_LOW,
    MODEL_PROFILE_HIGH,
    MODEL_PROFILE_CUSTOM,
)

# Agent Imports
from src.agents.coding_agent import CodingAgent
from src.agents.spec_agent import SpecAgent
from src.agents.testing_agent import TestingAgent
from src.agents.issue_scout.agent import IssueScoutAgent
from src.api.test_endpoints.data.defaults import (
    DEFAULT_SPEC_TICKET,
    DEFAULT_CODING_SPEC,
    DEFAULT_CODING_FILES,
    DEFAULT_TESTING_MOCK,
    DEFAULT_SCOUT_MOCK,
)

router = APIRouter(prefix="/test", tags=["Testing"])

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _initialize_state(overrides: Dict[str, Any]) -> GraphState:
    """Create a default GraphState and apply user overrides."""
    base_state: GraphState = {
        "log_file_path": "",
        "chat_log_file_path": "",
        "total_tokens": 0,
        "iteration_count": 0,
        "ticket_text": "",
        "spec": "",
        "issue_number": 0,
        "branch_name": "",
        "repo_url": "",
        "agent_reports": [],
        "orchestrator_inbox": {},
        "detected_language": "",
        "detected_framework": "",
        "test_output": "",
        "tests_passed": False,
        "tests_generated": False,
        "agent_outcome": "",
        "intent": "STANDARD_FLOW",
        "answer": "",
        "pipeline": [],
        "pipeline_step": 0,
        "next_node": "",
        "pr_url": "",
        "model_profile": MODEL_PROFILE_STANDARD,
        "max_tool_out": 2000,
        "mcp_servers": [],
        "messages": [],
    }
    # Deep merge or just update? For testing, simple update is usually fine.
    base_state.update(overrides)
    return base_state


def _setup_mock_workspace(mock_files: Dict[str, str]) -> Path:
    """Create a unique workspace directory and populate it with mock files."""
    test_id = str(uuid.uuid4())[:8]
    workspace_path = PROJECT_ROOT / "workspace" / f"test_{test_id}"
    workspace_path.mkdir(parents=True, exist_ok=True)

    for rel_path, content in mock_files.items():
        file_path = workspace_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    return workspace_path


def _cleanup_workspace(workspace_path: Path):
    """Delete the workspace directory."""
    if workspace_path.exists() and "test_" in workspace_path.name:
        shutil.rmtree(workspace_path)


async def _run_agent_test(
    agent_class: Type, request: AgentTestRequest, endpoint_name: str
) -> AgentTestResponse:
    """Generic logic for running an agent test."""
    workspace_path = _setup_mock_workspace(request.mock_files)
    
    # Initialize state
    state = _initialize_state(request.state)
    
    # Set workspace in state. CodingAgent/_workspace_dir expects repo_url 
    # but we can also handle it by setting repo_url to the dummy folder name
    # OR we can just point everything to the absolute path if we modify the agents slightly.
    # Actually, most agents use self._workspace_dir(state).
    # CodingAgent._workspace_dir uses PROJECT_ROOT / "workspace" / repo_url.split("/")[-1]
    
    # To make it work without modifying agents, we set repo_url to the folder name
    state["repo_url"] = workspace_path.name
    
    # Logging
    log_path, chat_log_path = log_request_start(
        endpoint=f"/test/{endpoint_name}",
        http_method="POST",
        initial_state=state,
        entry_agent=agent_class.agent_name,
    )
    state["log_file_path"] = log_path
    state["chat_log_file_path"] = chat_log_path

    try:
        agent = agent_class()
        result = await agent.run(state)
        
        report = result.get("orchestrator_inbox", {})
        
        # Build response
        response = AgentTestResponse(
            status=report.get("status", "unknown"),
            summary=report.get("summary", ""),
            state_updates={k: v for k, v in result.items() if k != "orchestrator_inbox" and k != "messages"},
            artifacts=report.get("artifacts", []),
            issues=report.get("issues", []),
            tokens=report.get("tokens", 0),
            workspace_path=str(workspace_path),
        )
        return response

    finally:
        if request.cleanup:
            _cleanup_workspace(workspace_path)


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #

@router.post("/spec", response_model=AgentTestResponse)
async def test_spec_agent(request: AgentTestRequest = AgentTestRequest()):
    """Test the Spec Agent in isolation. Uses default ticket if none provided."""
    state_dict = request.state or {}
    if not state_dict.get("ticket_text"):
        state_dict["ticket_text"] = DEFAULT_SPEC_TICKET
    
    request.state = state_dict
    return await _run_agent_test(SpecAgent, request, "spec")


@router.post("/coding", response_model=AgentTestResponse)
async def test_coding_agent(request: AgentTestRequest = AgentTestRequest()):
    """Test the Coding Agent in isolation. Uses default spec and files if none provided."""
    state_dict = request.state or {}
    mock_files = request.mock_files or {}

    if not state_dict.get("spec"):
        state_dict["spec"] = DEFAULT_CODING_SPEC
    
    if not mock_files:
        mock_files = DEFAULT_CODING_FILES

    request.state = state_dict
    request.mock_files = mock_files
    
    return await _run_agent_test(CodingAgent, request, "coding")


@router.post("/testing", response_model=AgentTestResponse)
async def test_testing_agent(request: AgentTestRequest = AgentTestRequest()):
    """Test the Testing Agent in isolation. Uses default mock files if none provided."""
    if not request.mock_files:
        request.mock_files = DEFAULT_TESTING_MOCK
    return await _run_agent_test(TestingAgent, request, "testing")


@router.post("/scout", response_model=AgentTestResponse)
async def test_scout_agent(request: AgentTestRequest = AgentTestRequest()):
    """Test the Issue Scout Agent in isolation. Uses default mock files if none provided."""
    if not request.mock_files:
        request.mock_files = DEFAULT_SCOUT_MOCK
    return await _run_agent_test(IssueScoutAgent, request, "scout")
