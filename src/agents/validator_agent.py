from src.state import GraphState
from src.config.llm import get_llm
from src.tools.ast_analysis import get_ast_tools
from src.tools.graph_rag import get_graph_rag_tools
from src.utils.logger import log_llm_interaction, log_chat_interaction
from langchain_core.messages import SystemMessage, HumanMessage
import os


def validator_agent_node(state: GraphState) -> dict:
    """
    Validates the generated technical specification against completeness criteria
    and the existing codebase structure (via AST + GraphRAG).
    Flags naming collisions with existing symbols and missing implementation details.
    """
    llm = get_llm()
    spec = state.get("spec", "")
    iteration_count = state.get("spec_iteration_count", 1)
    log_file_path = state.get("log_file_path", "")
    chat_log_file_path = state.get("chat_log_file_path", "")
    total_tokens = state.get("total_tokens", 0)

    # Prevent infinite loops: if we have tried 3 times, force approval
    if iteration_count >= 3:
        print(f"[ Validator Agent ] Reached {iteration_count} iterations. Forcing VALID verdict to proceed.")
        return {"spec_feedback": "VALID"}

    # Build tools for codebase-awareness
    repo_url = state.get("repo_url", "")
    base_workspace = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    )
    if repo_url:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        workspace_dir = os.path.join(base_workspace, repo_name)
    else:
        workspace_dir = base_workspace
    ast_tools = get_ast_tools()
    graph_rag_tools = get_graph_rag_tools()
    all_tools = ast_tools + graph_rag_tools
    llm_with_tools = llm.bind_tools(all_tools)

    current_request_tokens = 0

    # ------------------------------------------------------------------ #
    # Pass 1: LLM inspects the workspace graph to gather codebase context #
    # ------------------------------------------------------------------ #
    inspect_messages = [
        SystemMessage(content=(
            "You are an expert technical reviewer with access to code-analysis tools.\n"
            "Before reviewing a specification, call `summarise_code_graph` on the workspace "
            "to understand what already exists, then call `query_code_graph` for key terms "
            "from the spec to check for naming collisions or missing dependencies."
        )),
        HumanMessage(content=(
            f"Workspace path: {workspace_dir}\n"
            f"Specification to review:\n{spec}\n\n"
            "Use your tools to inspect the workspace and gather context."
        ))
    ]

    if chat_log_file_path:
        log_chat_interaction(chat_log_file_path, "Validator Agent (inspection)", inspect_messages)

    inspection_response = llm_with_tools.invoke(inspect_messages)

    # Extract token usage for inspection pass
    usage = inspection_response.usage_metadata or {}
    p_tokens = usage.get("input_tokens", 0)
    c_tokens = usage.get("output_tokens", 0)
    current_request_tokens += (p_tokens + c_tokens)
    
    if log_file_path:
        model = getattr(llm, "model", getattr(llm, "model_name", "unknown-model"))
        log_llm_interaction(log_file_path, f"Validator Agent (inspection)", model, p_tokens, c_tokens)

    # Execute any tool calls the LLM requests during inspection
    tool_results: list[str] = []
    if hasattr(inspection_response, "tool_calls") and inspection_response.tool_calls:
        for tool_call in inspection_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            for t in all_tools:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    tool_results.append(f"[{tool_name} result]:\n{result}")
    else:
        tool_results = ["No workspace tools were called — proceeding with spec-only review."]

    # --------------------------------------------------------- #
    # Pass 2: Final VALID / feedback verdict                     #
    # --------------------------------------------------------- #
    verdict_prompt = (
        "Review the technical specification below against the following criteria:\n"
        "1. Does it clearly identify which files need to be modified or created?\n"
        "2. Does it provide a high-level description of the logic or changes required?\n"
        "3. Does it avoid clear naming collisions based on the workspace analysis?\n\n"
        "If the specification meets all 3 criteria, you MUST respond with EXACTLY 'VALID'.\n"
        "Do not be overly pedantic. If a developer has enough direction to write the code, approve it.\n"
        "If it fundamentally fails a criterion, provide specific actionable feedback — do NOT say VALID.\n\n"
        f"Specification to review:\n{spec}\n\n"
        f"Workspace analysis results:\n" + "\n".join(tool_results)
    )

    messages = [
        SystemMessage(content=(
            "You are a pragmatic technical reviewer. "
            "Your final answer must be either EXACTLY VALID, or specific actionable feedback. "
            "Err on the side of approval if the core architectural direction is clear."
        )),
        HumanMessage(content=verdict_prompt)
    ]

    if chat_log_file_path:
        log_chat_interaction(chat_log_file_path, "Validator Agent (verdict)", messages)

    print("[ Validator Agent ] Evaluating spec against codebase context for final verdict...")
    response = llm.invoke(messages)

    # Extract token usage for final verdict pass
    usage = response.usage_metadata or {}
    p_tokens = usage.get("input_tokens", 0)
    c_tokens = usage.get("output_tokens", 0)
    current_request_tokens += (p_tokens + c_tokens)
    
    if log_file_path:
        model = getattr(llm, "model", getattr(llm, "model_name", "unknown-model"))
        log_llm_interaction(log_file_path, "Validator Agent (verdict)", model, p_tokens, c_tokens)

    raw = response.content
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    content = str(raw).strip()

    # Handle the 'VALID' keyword precisely
    if content.upper().startswith("VALID"):
        feedback = "VALID"
    else:
        feedback = content

    return {
        "spec_feedback": feedback,
        "total_tokens": total_tokens + current_request_tokens
    }
