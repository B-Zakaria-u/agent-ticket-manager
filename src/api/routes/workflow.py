"""Workflow routes — SRP: HTTP request/response handling only."""
import asyncio
import os
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas.workflow import RunResponse, TicketRequest
from src.graph import build_graph
from src.utils.logger import log_request_start

router = APIRouter(prefix="/run", tags=["Workflow"])


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _base_state(ticket_text: str = "") -> dict:
    os.makedirs("workspace", exist_ok=True)
    return {
        "log_file_path": "",
        "chat_log_file_path": "",
        "total_tokens": 0,
        "ticket_text": ticket_text,
        "issue_number": 0,
        "branch_name": "",
        "repo_url": "",
        "detected_language": "",
        "detected_framework": "",
        "spec": "",
        "spec_feedback": "",
        "spec_iteration_count": 0,
        "test_output": "",
        "tests_passed": False,
        "tests_generated": False,
        "pr_url": "",
        "iteration_count": 0,
        "intent": "STANDARD_FLOW",
        "answer": "",
    }


def _extract_final(outputs: list[dict]) -> RunResponse:
    final: dict = {}
    for o in outputs:
        node_name = list(o.keys())[0]
        print(f"[pipeline] Node complete: {node_name}")
        for v in o.values():
            final.update(v)
    print("[pipeline] Run finished.")
    return RunResponse(
        spec=final.get("spec") or "",
        spec_feedback=final.get("spec_feedback") or "",
        test_output=final.get("test_output") or "",
        tests_passed=bool(final.get("tests_passed")),
        pr_url=final.get("pr_url") or "",
        iteration_count=final.get("iteration_count") or 0,
        intent=final.get("intent") or "STANDARD_FLOW",
        answer=final.get("answer") or "",
    )


async def _sse_stream(graph, initial_state: dict) -> AsyncGenerator[str, None]:
    for output in graph.stream(initial_state):
        node_name = list(output.keys())[0]
        print(f"[pipeline] Node stream: {node_name}")
        yield f"data: [node:{node_name}] {output[node_name]}\n\n"
        await asyncio.sleep(0)
    print("[pipeline] Streaming run finished.")


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #

@router.post("", response_model=RunResponse)
async def run_manual(request: TicketRequest) -> RunResponse:
    """
    Run the pipeline with a manually supplied ticket text.
    """
    if not request.ticket_text.strip():
        raise HTTPException(status_code=422, detail="ticket_text must not be empty.")

    initial_state = _base_state(request.ticket_text)
    log_path, chat_log_path = log_request_start(
        endpoint="/run",
        http_method="POST",
        initial_state=initial_state,
        entry_agent="Orchestrator Agent",
    )
    initial_state["log_file_path"] = log_path
    initial_state["chat_log_file_path"] = chat_log_path

    graph = build_graph()
    outputs = list(graph.stream(initial_state))
    return _extract_final(outputs)


@router.post("/auto", response_model=RunResponse)
async def run_auto() -> RunResponse:
    """
    Fully autonomous run — the Issue Scout picks an open GitHub issue,
    self-assigns it, clones the repo, and the pipeline fixes and pushes it.
    """
    initial_state = _base_state()
    log_path, chat_log_path = log_request_start(
        endpoint="/run/auto",
        http_method="POST",
        initial_state=initial_state,
        entry_agent="Orchestrator Agent",
    )
    initial_state["log_file_path"] = log_path
    initial_state["chat_log_file_path"] = chat_log_path

    graph = build_graph()
    outputs = list(graph.stream(initial_state))
    return _extract_final(outputs)


@router.post("/stream")
async def stream_manual(request: TicketRequest) -> StreamingResponse:
    """
    Stream a manual-ticket run as Server-Sent Events.
    One SSE event is emitted per completing agent node.
    """
    if not request.ticket_text.strip():
        raise HTTPException(status_code=422, detail="ticket_text must not be empty.")

    initial_state = _base_state(request.ticket_text)
    log_path, chat_log_path = log_request_start(
        endpoint="/run/stream",
        http_method="POST",
        initial_state=initial_state,
        entry_agent="Orchestrator Agent",
    )
    initial_state["log_file_path"] = log_path
    initial_state["chat_log_file_path"] = chat_log_path

    graph = build_graph()
    return StreamingResponse(
        _sse_stream(graph, initial_state),
        media_type="text/event-stream",
    )


@router.post("/auto/stream")
async def stream_auto() -> StreamingResponse:
    """Stream the fully autonomous GitHub issue-driven run as SSE."""
    initial_state = _base_state()
    log_path, chat_log_path = log_request_start(
        endpoint="/run/auto/stream",
        http_method="POST",
        initial_state=initial_state,
        entry_agent="Orchestrator Agent",
    )
    initial_state["log_file_path"] = log_path
    initial_state["chat_log_file_path"] = chat_log_path

    graph = build_graph()
    return StreamingResponse(
        _sse_stream(graph, initial_state),
        media_type="text/event-stream",
    )


@router.post("/merge")
async def merge_request():
    """Merge the pull request."""
    initial_state = _base_state()
    log_path, chat_log_path = log_request_start(
        endpoint="/run/merge",
        http_method="POST",
        initial_state=initial_state,
        entry_agent="Orchestrator Agent",
    )
    initial_state["log_file_path"] = log_path
    initial_state["chat_log_file_path"] = chat_log_path

    graph = build_graph()
    outputs = list(graph.stream(initial_state))
    return _extract_final(outputs)
