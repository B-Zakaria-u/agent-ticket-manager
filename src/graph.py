"""
src/graph.py — LangGraph graph with dynamic orchestrator routing.

Flow
----

                        ┌──────────────┐
    START ─────────────►│ orchestrator │◄─────────────────────────────┐
                        └──────┬───────┘                              │
                               │  next_node                           │
                    ┌──────────┼──────────┐─────────────┐             │
                    ▼          ▼          ▼             ▼             │
              spec_agent  coding_agent  testing_agent  pr_agent       │
                    │          │          │             │             │
                    └──────────┴──────────┴─────────────┘             │
                                          │                           │
                                          └──► orchestrator ──────────┘
                                               (reads inbox, sets next_node)
                                                         │
                                              next_node="end"
                                                         │
                                                        END

Special nodes
─────────────
  "retry"        → re-invoke the same agent (tracked via retry_counts in state)
  "human_review" → interrupt_before so a human can inspect / modify state
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.coding_agent import CodingAgent
from src.agents.orchestrator_agent import orchestrator_agent_node
from src.agents.testing_agent import TestingAgent
from src.agents.issue_scout.agent import IssueScoutAgent
from src.state import GraphState

# Import optional agents (add yours here)
try:
    from src.agents.spec_agent import SpecAgent
    _HAS_SPEC = True
except ImportError:
    _HAS_SPEC = False

try:
    from src.agents.pr.agent import PRAgent
    _HAS_PR = True
except ImportError:
    _HAS_PR = False


# ── Agent node wrappers ───────────────────────────────────────────────────────
async def _issue_scout_node(state: GraphState) -> dict:
    return await IssueScoutAgent().run(state)

async def _coding_node(state: GraphState) -> dict:
    return await CodingAgent().run(state)


async def _testing_node(state: GraphState) -> dict:
    return await TestingAgent().run(state)


async def _spec_node(state: GraphState) -> dict:
    if _HAS_SPEC:
        return await SpecAgent().run(state)
    # Fallback: pass-through if SpecAgent not implemented yet
    print("[ Graph ] SpecAgent not found — skipping to coding_agent.")
    return {
        "orchestrator_inbox": {
            "agent": "spec_agent",
            "status": "success",
            "summary": "Spec agent skipped (not implemented).",
            "artifacts": [],
            "issues": [],
            "suggestions": ["proceed_to_next_pipeline_step"],
            "tokens": 0,
        },
        "agent_reports": (state.get("agent_reports") or [])
        + [
            {
                "agent": "spec_agent",
                "status": "success",
                "summary": "Skipped.",
                "tokens": 0,
            }
        ],
    }


async def _pr_node(state: GraphState) -> dict:
    if _HAS_PR:
        return await PRAgent().run(state)
    print("[ Graph ] PRAgent not found — ending pipeline.")
    return {
        "orchestrator_inbox": {
            "agent": "pr_agent",
            "status": "success",
            "summary": "PR agent skipped (not implemented).",
            "artifacts": [],
            "issues": [],
            "suggestions": ["proceed_to_next_pipeline_step"],
            "tokens": 0,
        },
    }


# ── Retry wrapper ─────────────────────────────────────────────────────────────

_AGENT_NODE_MAP = {
    "issue_scout":   _issue_scout_node,
    "spec_agent":    _spec_node,
    "coding_agent":  _coding_node,
    "testing_agent": _testing_node,
    "pr_agent":      _pr_node,
}


async def _retry_node(state: GraphState) -> dict:
    """
    Re-run whichever agent last reported into orchestrator_inbox.
    The orchestrator already incremented retry_counts before routing here.
    """
    inbox      = state.get("orchestrator_inbox") or {}
    agent_name = inbox.get("agent", "")
    node_fn    = _AGENT_NODE_MAP.get(agent_name)

    if node_fn is None:
        print(f"[ Graph ] Retry requested for unknown agent '{agent_name}' — ending.")
        return {"next_node": "end"}

    print(f"[ Graph ] Retrying '{agent_name}'...")
    return await node_fn(state)


# ── Conditional edge: orchestrator → next node ────────────────────────────────

def _route(state: GraphState) -> str:
    """Read next_node from state and return the node name for LangGraph."""
    node = state.get("next_node") or "end"
    print(f"[ Graph ] Routing → {node}")
    return node


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    g.add_node("orchestrator",  orchestrator_agent_node)
    g.add_node("issue_scout",   _issue_scout_node)
    g.add_node("spec_agent",    _spec_node)
    g.add_node("coding_agent",  _coding_node)
    g.add_node("testing_agent", _testing_node)
    g.add_node("pr_agent",      _pr_node)
    g.add_node("retry",         _retry_node)
    g.add_node("human_review",  _human_review_node)   # interrupt_before set below

    # ── Edges: START → orchestrator ───────────────────────────────────────────
    g.add_edge(START, "orchestrator")

    # ── Conditional edge: orchestrator → wherever ─────────────────────────────
    g.add_conditional_edges(
        "orchestrator",
        _route,
        {
            "issue_scout":   "issue_scout",
            "spec_agent":    "spec_agent",
            "coding_agent":  "coding_agent",
            "testing_agent": "testing_agent",
            "pr_agent":      "pr_agent",
            "retry":         "retry",
            "human_review":  "human_review",
            "end":           END,
        },
    )

    # ── All agent nodes feed back into orchestrator ───────────────────────────
    for agent_node in ("issue_scout", "spec_agent", "coding_agent", "testing_agent", "pr_agent", "retry"):
        g.add_edge(agent_node, "orchestrator")

    # human_review can also return to orchestrator once a human has acted
    g.add_edge("human_review", "orchestrator")

    return g


async def _human_review_node(state: GraphState) -> dict:
    """
    Placeholder human-review node.
    In production, LangGraph's interrupt_before mechanism pauses here
    so an external system / human can inspect and modify state before
    the graph resumes toward orchestrator.
    """
    inbox = state.get("orchestrator_inbox", {})
    print(
        f"\n{'='*60}\n"
        f"[ HUMAN REVIEW REQUIRED ]\n"
        f"Agent : {inbox.get('agent')}\n"
        f"Status: {inbox.get('status')}\n"
        f"Issues:\n"
        + "\n".join(f"  - {i}" for i in inbox.get("issues", []))
        + f"\n{'='*60}\n"
    )
    # The human modifies state externally; we just pass through.
    # Set next_node to a safe default so the orchestrator re-evaluates.
    return {"next_node": "orchestrator", "orchestrator_inbox": {}}


# ── Compiled graph (with human-review interrupt) ──────────────────────────────

def compile_graph(checkpointer=None):
    """
    Returns a compiled LangGraph app.

    Pass a LangGraph checkpointer (e.g. MemorySaver) to enable persistence
    and human-in-the-loop interrupts.

    Example:
        from langgraph.checkpoint.memory import MemorySaver
        app = compile_graph(checkpointer=MemorySaver())
        result = await app.ainvoke(initial_state, config={"configurable": {"thread_id": "1"}})
    """
    graph = build_graph()
    kwargs: dict = {}
    if checkpointer:
        kwargs["checkpointer"]    = checkpointer
        kwargs["interrupt_before"] = ["human_review"]

    return graph.compile(**kwargs)