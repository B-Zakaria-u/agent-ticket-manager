from src.state import GraphState
from src.config.llm import get_llm
from src.utils.logger import log_llm_interaction, log_chat_interaction
from langchain_core.messages import SystemMessage, HumanMessage
import json
import os

def orchestrator_agent_node(state: GraphState) -> dict:
    """
    Analyzes the user input and determines the next step in the workflow.
    Intents:
    - QUESTION: Direct answer to a query.
    - CODING_WITH_SPEC: Skip spec agent, go to coding (spec detected in input).
    - STANDARD_FLOW: Go to spec agent.
    - VALIDATE_MR: Go to MR validation flow.
    """
    llm = get_llm()
    ticket_text = state.get("ticket_text", "")
    log_file_path = state.get("log_file_path", "")
    chat_log_file_path = state.get("chat_log_file_path", "")
    total_tokens = state.get("total_tokens", 0)

    if not ticket_text or not ticket_text.strip():
        print(f"[ Orchestrator Agent ] No ticket text provided. Routing to Issue Scout first...")
        return {
            "intent": "STANDARD_FLOW",
            "pipeline": ["Issue Scout", "Orchestrator Agent"],
            "pipeline_step": 0
        }

    prompt = (
        "Analyze the following user request and categorize it into one of these intents:\n"
        "1. QUESTION: The user is asking a general question about the project, code, or how things work.\n"
        "2. CODING_WITH_SPEC: The user wants to implement code and has ALREADY provided a detailed technical specification (file names, class names, logic, etc.).\n"
        "3. STANDARD_FLOW: The user has a high-level requirement or bug report that needs a technical specification generated first.\n"
        "4. VALIDATE_MR: The user wants to validate or check a Merge Request or Pull Request.\n\n"
        f"User Request: {ticket_text}\n\n"
        "Respond with a JSON object containing:\n"
        "- intent: one of the four categories above.\n"
        "- reason: a brief explanation for the category.\n"
        "- pipeline: a list of agent names to execute in order. Available agents: ['Spec Agent', 'Coding Agent', 'Testing Agent', 'PR Agent'].\n"
        "- answer: (ONLY for QUESTION intent) the direct answer to the user's question."
    )

    messages = [
        SystemMessage(content="You are an expert AI orchestrator. Your job is to route user requests to the appropriate processing agent."),
        HumanMessage(content=prompt)
    ]

    if chat_log_file_path:
        log_chat_interaction(chat_log_file_path, "Orchestrator Agent", messages)

    print(f"[ Orchestrator Agent ] Analyzing intent for: {ticket_text[:50]}...")
    response = llm.invoke(messages)

    # Extract token usage
    usage = response.usage_metadata or {}
    p_tokens = usage.get("input_tokens", 0)
    c_tokens = usage.get("output_tokens", 0)

    if log_file_path:
        model = getattr(llm, "model", getattr(llm, "model_name", "unknown-model"))
        log_llm_interaction(log_file_path, "Orchestrator Agent", model, p_tokens, c_tokens)

    raw = response.content
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    
    # Try to parse JSON from response
    try:
        # Simple JSON extraction in case there's markdown
        json_str = str(raw).strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        result = json.loads(json_str)
        intent = result.get("intent", "STANDARD_FLOW")
        answer = result.get("answer", "")
        pipeline = result.get("pipeline", [])
        
        # If the LLM detected CODING_WITH_SPEC, we should use the ticket_text as the spec
        spec = ""
        if intent == "CODING_WITH_SPEC":
            spec = ticket_text
            
    except Exception as e:
        print(f"[ Orchestrator Agent ] Error parsing response: {e}. Defaulting to STANDARD_FLOW.")
        intent = "STANDARD_FLOW"
        answer = ""
        spec = ""
        pipeline = ["Spec Agent", "Coding Agent", "Testing Agent", "PR Agent"]

    print(f"[ Orchestrator Agent ] Detected intent: {intent}")
    print(f"[ Orchestrator Agent ] Pipeline: {pipeline}")

    from src.config.llm import (
        MODEL_PROFILE_CUSTOM, 
        MODEL_PROFILE_LOW,
        MODEL_PROFILE_STANDARD,
        MODEL_PROFILE_HIGH
    )

    PROFILE_MAP = {
        "CUSTOM": MODEL_PROFILE_CUSTOM,
        "LOW": MODEL_PROFILE_LOW,
        "STANDARD": MODEL_PROFILE_STANDARD,
        "HIGH": MODEL_PROFILE_HIGH,
    }
    # Default to CUSTOM if not specified
    model_profile_name = os.getenv("MODEL_PROFILE","CUSTOM").upper()
    
    model_profile = PROFILE_MAP.get(model_profile_name, MODEL_PROFILE_CUSTOM)

    update = {
        "intent": intent,
        "answer": answer,
        "pipeline": pipeline,
        "pipeline_step": 0,
        "total_tokens": total_tokens + p_tokens + c_tokens,
        "model_profile": model_profile,
        "max_tool_out": 25,
    }
    if spec:
        update["spec"] = spec
        
    return update
