"""LLM factory — returns a ChatGoogleGenerativeAI, ChatGroq, or Local LLM client (LM Studio)."""
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def get_llm():
    """
    Return a chat model based on the LLM_PROVIDER environment variable.
    Supported providers: 'google' (default), 'groq', 'lmstudio'.
    """
    provider = os.getenv("LLM_PROVIDER", "google").lower()

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
            return ChatOpenRouter(
                api_key=os.getenv("OPEN_ROUTER_KEY"),
                model=os.getenv("OPEN_ROUTER_MODEL", "qwen/qwen3-coder:free"),
                temperature=0.2
            )
        except ImportError:
            raise ImportError(
                "Could not import langchain_openrouter. Please install it with: "
                "pip install langchain-openrouter"
            )

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.1,
            )
        except ImportError:
            raise ImportError(
                "Could not import langchain_groq. Please install it with: "
                "pip install langchain-groq"
            )

    if provider == "lmstudio":
        try:
            from langchain_openai import ChatOpenAI
            base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
            model_name = os.getenv("LMSTUDIO_MODEL", "local-model")
            return ChatOpenAI(
                base_url=base_url,
                model=model_name,
                openai_api_key="not-needed",
                temperature=0.1,
            )
        except ImportError:
            raise ImportError(
                "Could not import langchain_openai. Please install it with: "
                "pip install langchain-openai"
            )

    # Default to Google Gemini
    return ChatGoogleGenerativeAI(
        model=os.getenv("MODEL_NAME", "gemini-1.5-flash"),
        temperature=0.1,
    )


# =============================================================================
#                               MODEL PROFILE
# =============================================================================
# MODEL_PROFILE_LOW:        2B to 4B local model (~4K effective context)
# MODEL_PROFILE_STANDARD:   7B to 14B local model (~8-16K effective context)
# MODEL_PROFILE_HIGH:       27B to 30B local model (~32K effective context)
# MODEL_PROFILE_CUSTOM:     custom values

MODEL_PROFILE_LOW = {
    "max_context":    4_000,   # → 16_000
    "max_tool_out":   2_000,   # → 8_000
    "max_history":    6,       # → 20
    "max_files":      30,      # → 80
    "max_spec":       1_500,   # → 6_000
    "max_test_out":   800,     # → 3_000
    "system_verbose": False,   # → True  (re-enables full conventions/rules)
}

MODEL_PROFILE_STANDARD = {
    "max_context":    16_000,  # → 16_000
    "max_tool_out":   8_000,   # → 8_000
    "max_history":    20,      # → 20
    "max_files":      80,      # → 80
    "max_spec":       6_000,   # → 6_000
    "max_test_out":   3_000,   # → 3_000
    "system_verbose": True,    # → True  (re-enables full conventions/rules)
}

MODEL_PROFILE_HIGH = {
    "max_context":    32_000,  # → 32_000
    "max_tool_out":   16_000,  # → 16_000
    "max_history":    40,      # → 40
    "max_files":      160,     # → 160
    "max_spec":       12_000,  # → 12_000
    "max_test_out":   6_000,   # → 6_000
    "system_verbose": True,    # → True  (re-enables full conventions/rules)
}

MODEL_PROFILE_CUSTOM = {
    "max_context":    os.getenv("MAX_CONTEXT"),  # → 12_000
    "max_tool_out":   os.getenv("MAX_TOOL_OUT"),   # → 6_000
    "max_history":    os.getenv("MAX_HISTORY"),    # → 20
    "max_files":      os.getenv("MAX_FILES"),      # → 80
    "max_spec":       os.getenv("MAX_SPEC"),       # → 6_000
    "max_test_out":   os.getenv("MAX_TEST_OUT"),   # → 3_000
    "system_verbose": os.getenv("SYSTEM_VERBOSE"),    # → True  (re-enables full conventions/rules)
}