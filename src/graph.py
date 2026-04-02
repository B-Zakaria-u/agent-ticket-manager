from langgraph.graph import StateGraph, END

from src.state import GraphState
from src.agents.orchestrator_agent import orchestrator_agent_node
from src.agents.issue_scout.agent import issue_scout_node
from src.agents.spec_agent import spec_agent_node
from src.agents.testing_agent import testing_agent_node
from src.agents.coding_agent import coding_agent_node
from src.agents.pr.agent import pr_agent_node


def router_node(state: GraphState) -> dict:
    """
    Determines the next agent from the dynamic pipeline.
    """
    pipeline = state.get("pipeline", [])
    step = state.get("pipeline_step", 0)

    if not pipeline or step >= len(pipeline):
        print("[ Router ] Pipeline empty or complete. Transitioning to END.")
        return {"next_agent": END}

    next_agent = pipeline[step]
    print(f"[ Router ] Next agent in pipeline: {next_agent} (step {step + 1}/{len(pipeline)})")
    
    return {
        "next_agent": next_agent,
        "pipeline_step": step + 1
    }


def _route_dynamic(state: GraphState) -> str:
    """Route to the next agent specified by the router node."""
    return state.get("next_agent", END)


def _route_coding(state: GraphState) -> str:
    if state.get("tests_passed", False) or state.get("iteration_count", 0) >= 3:
        return "Router"
    return "Coding Agent"


def build_graph():
    """Compile and return the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    workflow.add_node("Orchestrator Agent", orchestrator_agent_node)
    workflow.add_node("Issue Scout",    issue_scout_node)
    workflow.add_node("Spec Agent",     spec_agent_node)
    workflow.add_node("Testing Agent",  testing_agent_node)
    workflow.add_node("Coding Agent",   coding_agent_node)
    workflow.add_node("PR Agent",       pr_agent_node)
    workflow.add_node("Router",          router_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("Orchestrator Agent")

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.add_edge("Orchestrator Agent", "Router")
    
    # After each agent, go back to Router to get the next one
    workflow.add_edge("Issue Scout", "Router")
    workflow.add_edge("Spec Agent", "Router")
    workflow.add_edge("Testing Agent", "Router")
    workflow.add_conditional_edges("Coding Agent", _route_coding)
    workflow.add_edge("PR Agent", "Router")
    
    # Router decides the next step
    workflow.add_conditional_edges("Router", _route_dynamic)

    return workflow.compile()
