"""
orchestrator_agent.py — Dynamic Orchestrator (v2).

Responsibilities
----------------
1. First call  : parse raw ticket, set intent + initial pipeline.
2. Subsequent  : read the latest AgentReport from `orchestrator_inbox`,
                 decide what to do next, and write `next_node`.

`next_node` is the string name used by the LangGraph conditional edge
(see graph.py).  Valid values:

    "spec_agent"     → run SpecAgent
    "coding_agent"   → run CodingAgent
    "testing_agent"  → run TestingAgent
    "pr_agent"       → run PRAgent
    "retry"          → re-run the agent that just failed (increment attempts)
    "end"            → finish the graph
    "human_review"   → pause for human input  (interrupt_before in graph.py)

Decision logic
--------------
The orchestrator calls the LLM with:
  - the full list of previous AgentReports (summarised)
  - the latest report in detail
  - the original ticket / intent
  - the remaining pipeline
It asks for a JSON decision: {next_node, reason, updated_pipeline}.
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.base_config import (
    MODEL_PROFILE_CUSTOM,
    MODEL_PROFILE_HIGH,
    MODEL_PROFILE_LOW,
    MODEL_PROFILE_STANDARD,
    get_llm,
)
from src.state import AgentReport, GraphState
from src.utils.logger import log_chat_interaction, log_llm_interaction

# ── Constants ─────────────────────────────────────────────────────────────────

PROFILE_MAP = {
    "CUSTOM":   MODEL_PROFILE_CUSTOM,
    "LOW":      MODEL_PROFILE_LOW,
    "STANDARD": MODEL_PROFILE_STANDARD,
    "HIGH":     MODEL_PROFILE_HIGH,
}

VALID_NODES = {
    "issue_scout",
    "spec_agent",
    "coding_agent",
    "testing_agent",
    "pr_agent",
    "retry",
    "end",
    "human_review",
}

# Map user-facing pipeline names → internal node names
PIPELINE_NAME_MAP: dict[str, str] = {
    "Issue Scout": "issue_scout",
    "Spec Agent":    "spec_agent",
    "Coding Agent":  "coding_agent",
    "Testing Agent": "testing_agent",
    "PR Agent":      "pr_agent",
}

_MAX_REPORT_SUMMARY_LEN = 300   # chars per historical report in the prompt
_MAX_RETRY_ATTEMPTS     = 2     # how many times we allow re-running a failed agent


# ── Node function ─────────────────────────────────────────────────────────────

async def orchestrator_agent_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node.  Async so it can be used with async graph execution.
    Called:
      • once at the start (no orchestrator_inbox)
      • after every other agent completes (orchestrator_inbox is set)
    """
    inbox: AgentReport | None = state.get("orchestrator_inbox")

    if inbox is None:
        # ── First call: parse ticket and set up pipeline ───────────────────
        return await _initial_routing(state)
    else:
        # ── Subsequent calls: re-evaluate after an agent finished ──────────
        return await _dynamic_routing(state, inbox)


# ── First-call: intent detection + pipeline setup ─────────────────────────────

async def _initial_routing(state: GraphState) -> dict[str, Any]:
    ticket_text        = state.get("ticket_text", "").strip()
    log_file           = state.get("log_file_path", "")
    chat_log_file      = state.get("chat_log_file_path", "")
    total_tokens       = state.get("total_tokens", 0)

    if not ticket_text:
        print("[ Orchestrator ] No ticket text — waiting for Issue Scout.")
        return {
            "intent":       "STANDARD_FLOW",
            "pipeline":     ["Issue Scout", "Spec Agent", "Coding Agent", "Testing Agent", "PR Agent"],
            "pipeline_step": 0,
            "next_node":    "issue_scout",
            "agent_reports": [],
        }

    llm = get_llm()
    prompt = _build_initial_prompt(ticket_text)
    messages = [
        SystemMessage(
            content=(
                "You are an expert AI orchestrator. Route user requests to the right "
                "agent pipeline and respond ONLY with valid JSON."
            )
        ),
        HumanMessage(content=prompt),
    ]

    if chat_log_file:
        log_chat_interaction(chat_log_file, "Orchestrator (Initial)", messages)

    print(f"[ Orchestrator ] Detecting intent for: {ticket_text[:80]}...")
    response = await llm.ainvoke(messages)

    usage      = getattr(response, "usage_metadata", {}) or {}
    p_tok      = usage.get("input_tokens", 0)
    c_tok      = usage.get("output_tokens", 0)
    new_tokens = total_tokens + p_tok + c_tok

    if log_file:
        model = getattr(llm, "model", getattr(llm, "model_name", "unknown"))
        log_llm_interaction(log_file, "Orchestrator (Initial)", model, p_tok, c_tok)

    parsed   = _safe_parse(response.content)
    intent   = parsed.get("intent", "STANDARD_FLOW")
    pipeline = parsed.get("pipeline", ["Issue Scout", "Spec Agent", "Coding Agent", "Testing Agent", "PR Agent"])
    answer   = parsed.get("answer", "")
    spec     = ticket_text if intent == "CODING_WITH_SPEC" else ""
    next_node = _pipeline_to_node(pipeline, 0) if intent != "QUESTION" else "end"

    model_profile_name = os.getenv("MODEL_PROFILE", "CUSTOM").upper()
    model_profile      = PROFILE_MAP.get(model_profile_name, MODEL_PROFILE_CUSTOM)

    print(f"[ Orchestrator ] Intent={intent}  first_node={next_node}  pipeline={pipeline}")

    update: dict[str, Any] = {
        "intent":        intent,
        "answer":        answer,
        "pipeline":      pipeline,
        "pipeline_step": 0,
        "next_node":     next_node,
        "total_tokens":  new_tokens,
        "model_profile": model_profile,
        "agent_reports": [],
    }
    if spec:
        update["spec"] = spec
    return update


# ── Subsequent calls: dynamic re-routing based on AgentReport ─────────────────

async def _dynamic_routing(
    state: GraphState, inbox: AgentReport
) -> dict[str, Any]:
    llm           = get_llm()
    log_file      = state.get("log_file_path", "")
    chat_log_file = state.get("chat_log_file_path", "")
    total_tokens  = state.get("total_tokens", 0)
    pipeline      = state.get("pipeline", [])
    step          = state.get("pipeline_step", 0)
    reports       = state.get("agent_reports", [])
    intent        = state.get("intent", "STANDARD_FLOW")
    retry_counts  = state.get("retry_counts", {})

    agent_name = inbox.get("agent", "unknown")
    status     = inbox.get("status", "failed")

    print(
        f"[ Orchestrator ] Received report from '{agent_name}': "
        f"status={status}  issues={len(inbox.get('issues', []))}"
    )

    # ── Fast-path: success → advance pipeline ─────────────────────────────────
    if status == "success" and not inbox.get("issues"):
        next_step = step + 1
        next_node = _pipeline_to_node(pipeline, next_step)

        # Guard: coding_agent needs a spec
        if next_node == "coding_agent" and not state.get("spec"):
            print(
                "[ Orchestrator ] Fast-path blocked: "
                "coding_agent requires spec but state['spec'] is empty."
            )
            return {
                "pipeline_step": step,
                "next_node":     "spec_agent",   # re-run spec
                "total_tokens":  total_tokens,
            }

        print(f"[ Orchestrator ] Fast-path success → {next_node}")
        return {
            "pipeline_step": next_step,
            "next_node":     next_node,
            "total_tokens":  total_tokens,
        }

    # ── LLM-based decision for partial / failed / blocked ─────────────────────
    prompt = _build_routing_prompt(
        inbox=inbox,
        pipeline=pipeline,
        step=step,
        reports=reports,
        intent=intent,
        retry_counts=retry_counts,
        max_retries=_MAX_RETRY_ATTEMPTS,
    )
    messages = [
        SystemMessage(
            content=(
                "You are an expert AI orchestrator. Analyse agent reports and decide "
                "the next execution step. Respond ONLY with valid JSON."
            )
        ),
        HumanMessage(content=prompt),
    ]

    if chat_log_file:
        log_chat_interaction(chat_log_file, "Orchestrator (Dynamic)", messages)

    response = await llm.ainvoke(messages)

    usage      = getattr(response, "usage_metadata", {}) or {}
    p_tok      = usage.get("input_tokens", 0)
    c_tok      = usage.get("output_tokens", 0)
    new_tokens = total_tokens + p_tok + c_tok

    if log_file:
        model = getattr(llm, "model", getattr(llm, "model_name", "unknown"))
        log_llm_interaction(log_file, "Orchestrator (Dynamic)", model, p_tok, c_tok)

    parsed           = _safe_parse(response.content)
    next_node        = _validate_node(parsed.get("next_node", "end"))
    updated_pipeline = parsed.get("updated_pipeline", pipeline)
    reason           = parsed.get("reason", "")

    print(f"[ Orchestrator ] Decision: {next_node}  reason: {reason[:120]}")

    # Track retry attempts to avoid infinite loops
    new_retry_counts = dict(retry_counts)
    if next_node == "retry":
        new_retry_counts[agent_name] = new_retry_counts.get(agent_name, 0) + 1
        if new_retry_counts[agent_name] > _MAX_RETRY_ATTEMPTS:
            print(
                f"[ Orchestrator ] Max retries reached for '{agent_name}' "
                "— escalating to human_review."
            )
            next_node = "human_review"

    # Advance step if we're moving forward (not retrying)
    next_step = step if next_node in {"retry", "human_review"} else step + 1

    return {
        "pipeline":      updated_pipeline,
        "pipeline_step": next_step,
        "next_node":     next_node,
        "total_tokens":  new_tokens,
        "retry_counts":  new_retry_counts,
        "orchestrator_decision_reason": reason,
    }


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_initial_prompt(ticket_text: str) -> str:
    return f"""Analyze the following user request and categorize it.

Intents:
  QUESTION          — user asks a general question (no code changes needed).
  CODING_WITH_SPEC  — user wants code AND has already provided a full technical spec.
  STANDARD_FLOW     — high-level requirement or bug report (needs spec first).
  VALIDATE_MR       — user wants to validate a Merge Request / Pull Request.

Available pipeline agents (in typical order):
  ["Spec Agent", "Coding Agent", "Testing Agent", "PR Agent"]

User Request:
{ticket_text}

Respond ONLY with a JSON object:
{{
  "intent":   "<one of the four intents>",
  "reason":   "<brief explanation>",
  "pipeline": ["<agent1>", ...],
  "answer":   "<direct answer — ONLY for QUESTION intent, else empty string>"
}}"""


def _build_routing_prompt(
    *,
    inbox: AgentReport,
    pipeline: list[str],
    step: int,
    reports: list[AgentReport],
    intent: str,
    retry_counts: dict,
    max_retries: int,
) -> str:
    history_lines = []
    for r in reports[-5:]:    # last 5 reports to keep prompt compact
        snippet = r.get("summary", "")[:_MAX_REPORT_SUMMARY_LEN]
        history_lines.append(
            f"  • [{r.get('agent')}] status={r.get('status')} — {snippet}"
        )
    history_str = "\n".join(history_lines) or "  (none)"

    issues_str      = "\n".join(f"  - {i}" for i in inbox.get("issues", [])) or "  (none)"
    suggestions_str = "\n".join(f"  - {s}" for s in inbox.get("suggestions", [])) or "  (none)"
    retries_str     = json.dumps(retry_counts)

    remaining = pipeline[step + 1:]
    return f"""=== ORCHESTRATOR DECISION REQUIRED ===

Original intent : {intent}
Current pipeline: {pipeline}
Step completed  : {step}  ({pipeline[step] if step < len(pipeline) else 'N/A'})
Remaining steps : {remaining}

--- Latest Agent Report ---
Agent    : {inbox.get('agent')}
Status   : {inbox.get('status')}
Summary  : {inbox.get('summary', '')[:400]}
Artifacts: {inbox.get('artifacts', [])}
Issues   :
{issues_str}
Suggestions from agent:
{suggestions_str}

--- Historical Reports (last 5) ---
{history_str}

--- Retry Counts ---
{retries_str}   (max allowed per agent: {max_retries})

=== VALID NEXT NODES ===
  "spec_agent"    — run SpecAgent
  "coding_agent"  — run CodingAgent
  "testing_agent" — run TestingAgent
  "pr_agent"      — run PRAgent
  "retry"         — re-run the agent that just reported
  "human_review"  — pause for human input (use when blocked or max retries hit)
  "end"           — finish the pipeline

Respond ONLY with a JSON object:
{{
  "next_node":        "<node name from the list above>",
  "reason":           "<brief explanation of your decision>",
  "updated_pipeline": ["<agent1>", ...]
}}"""


# ── Utility helpers ───────────────────────────────────────────────────────────

def _safe_parse(raw: Any) -> dict:
    """Extract and parse JSON from LLM response (handles markdown fences)."""
    if isinstance(raw, list):
        raw = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
        )
    text = str(raw).strip()
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence)[1].split("```")[0].strip()
            break
    try:
        return json.loads(text)
    except Exception as exc:
        print(f"[ Orchestrator ] JSON parse error: {exc}")
        return {}


def _pipeline_to_node(pipeline: list[str], step: int) -> str:
    """Convert a pipeline step index to an internal node name."""
    if step >= len(pipeline):
        return "end"
    name = pipeline[step]
    return PIPELINE_NAME_MAP.get(name, name.lower().replace(" ", "_"))


def _validate_node(node: str) -> str:
    """Ensure the LLM returned a valid node name."""
    if node in VALID_NODES:
        return node
    print(f"[ Orchestrator ] Invalid next_node '{node}' — defaulting to 'end'.")
    return "end"