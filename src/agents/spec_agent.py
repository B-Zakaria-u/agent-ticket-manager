import os
from src.state import GraphState
from src.config.llm import get_llm
from src.utils.logger import log_llm_interaction, log_chat_interaction
from src.utils.language_detector import detect_language
from langchain_core.messages import SystemMessage, HumanMessage


def spec_agent_node(state: GraphState) -> dict:
    """
    Reads the development ticket text and generates a comprehensive technical specification.
    Also handles iterative feedback from the Validator Agent.
    Language-agnostic: deduces language and framework from workspace context.
    """
    llm = get_llm()
    ticket_text = state.get("ticket_text", "")
    log_file_path = state.get("log_file_path", "")
    chat_log_file_path = state.get("chat_log_file_path", "")
    total_tokens = state.get("total_tokens", 0)

    # ── Detect language from workspace ───────────────────────────────────────
    repo_url = state.get("repo_url", "")
    base_workspace = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    )
    if repo_url:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        workspace_dir = os.path.join(base_workspace, repo_name)
    else:
        workspace_dir = base_workspace
    lang_info = detect_language(workspace_dir)
    detected_language = lang_info.get("language", "Unknown")
    detected_framework = lang_info.get("framework", "Unknown")

    lang_hint = ""
    if detected_language != "Unknown":
        lang_hint = f"\nWorkspace language detected: **{detected_language}**"
        if detected_framework != "Unknown":
            lang_hint += f" / **{detected_framework}**"
        lang_hint += "\n"

    prompt = (
        f"Please write a robust technical specification for this ticket:\n\n{ticket_text}\n"
        f"{lang_hint}"
    )

    messages = [
        SystemMessage(content=(
            "You are an expert AI systems architect. Produce a clear, actionable technical specification. "
            "Examine the ticket and deduce the appropriate programming language and framework — "
            "use the detected workspace language/framework hint when provided. "
            "Ensure your specification matches the conventions and tech stack of the repository. "
            "Be specific about file names, class names, method names, and test frameworks appropriate "
            "to the detected language (e.g. JUnit for Java, pytest for Python, Jest for JavaScript, "
            "PHPUnit for PHP, Kotest for Kotlin)."
        )),
        HumanMessage(content=prompt)
    ]

    # Log full prompt/messages
    if chat_log_file_path:
        log_chat_interaction(chat_log_file_path, "Spec Agent", messages)

    print(f"[ Spec Agent ] Generating technical specification (language: {detected_language}, framework: {detected_framework})...")
    response = llm.invoke(messages)

    # Extract token usage
    usage = response.usage_metadata or {}
    p_tokens = usage.get("input_tokens", 0)
    c_tokens = usage.get("output_tokens", 0)

    if log_file_path:
        model = getattr(llm, "model", getattr(llm, "model_name", "unknown-model"))
        log_llm_interaction(log_file_path, "Spec Agent", model, p_tokens, c_tokens)

    raw = response.content
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)

    print("[ Spec Agent ] Specification generated successfully.")
    return {
        "spec": str(raw),
        "detected_language": detected_language,
        "detected_framework": detected_framework,
        "total_tokens": total_tokens + p_tokens + c_tokens
    }
