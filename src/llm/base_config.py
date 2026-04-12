"""
LLM factory — plug-and-play via environment variables.

Configuration (**.env**):
    LLM_MODEL       = provider/model-name        (required)
    LLM_TEMPERATURE = 0.1                         (optional, default 0.1)

Supported provider prefixes (handled by langchain `init_chat_model`):
    google_genai/   → ChatGoogleGenerativeAI   (pip install langchain-google-genai)
    openai/         → ChatOpenAI               (pip install langchain-openai)
    groq/           → ChatGroq                 (pip install langchain-groq)
    openrouter/     → ChatOpenRouter           (pip install langchain-openrouter)
    ollama/         → ChatOllama               (pip install langchain-ollama)

Examples:
    LLM_MODEL=google_genai/gemini-2.5-flash
    LLM_MODEL=groq/llama-3.3-70b-versatile
    LLM_MODEL=openai/gpt-4o
    LLM_MODEL=openrouter/qwen/qwen3-coder:free
    LLM_MODEL=ollama/qwen3:4b

Legacy compatibility:
    If LLM_MODEL is not set, falls back to LLM_PROVIDER + MODEL_NAME env vars
    from the old configuration format.
"""
from __future__ import annotations

import os
from functools import lru_cache

from langchain.chat_models import init_chat_model


# ── Provider mapping for legacy env vars ──────────────────────────────────────
_LEGACY_PROVIDER_PREFIX = {
    "google":      "google_genai",
    "groq":        "groq",
    "lmstudio":    "openai",
    "openrouter":  "openrouter",
    "openai":      "openai",
    "ollama":      "ollama",
}

_LEGACY_MODEL_ENV = {
    "google":      "MODEL_NAME",
    "groq":        "GROQ_MODEL",
    "lmstudio":    "LMSTUDIO_MODEL",
    "openrouter":  "OPEN_ROUTER_MODEL",
    "openai":      "MODEL_NAME",
    "ollama":      "MODEL_NAME",
}

_LEGACY_EXTRA_KWARGS = {
    "lmstudio": lambda: {"base_url": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                         "api_key": "not-needed"},
    "openrouter": lambda: {"api_key": os.getenv("OPEN_ROUTER_KEY", "")},
}


def _resolve_model_string() -> str:
    """Build a 'provider/model' string from env vars."""
    # ── New-style: single LLM_MODEL var ───────────────────────────────────
    model = os.getenv("LLM_MODEL", "").strip()
    if model:
        return model

    # ── Legacy fallback ───────────────────────────────────────────────────
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    prefix   = _LEGACY_PROVIDER_PREFIX.get(provider, provider)
    env_key  = _LEGACY_MODEL_ENV.get(provider, "MODEL_NAME")
    default  = "gemini-2.5-flash" if provider == "google" else "unknown-model"
    name     = os.getenv(env_key, default)
    return f"{prefix}/{name}"


@lru_cache(maxsize=1)
def get_llm():
    """
    Return a cached chat model instance configured from environment variables.

    The model is created once per process and reused for all subsequent calls.
    """
    model_string = _resolve_model_string()

    # Collect any extra kwargs from legacy config
    provider_key = os.getenv("LLM_PROVIDER", "openrouter").lower()
    extra = {}
    if not os.getenv("LLM_MODEL") and provider_key in _LEGACY_EXTRA_KWARGS:
        extra = _LEGACY_EXTRA_KWARGS[provider_key]()

    print(f"[ LLM ] Initializing: {model_string}")

    # Explicitly separate provider and model if possible
    # to avoid parsing errors with models containing slashes
    if "/" in model_string:
        provider, model_name = model_string.split("/", 1)
        return init_chat_model(
            model_name,
            model_provider=provider,
            **extra,
        )

    return init_chat_model(
        model_string,
        **extra,
    )


# =============================================================================
#                               MODEL PROFILE
# =============================================================================
# MODEL_PROFILE_LOW:        2B to 4B local model (~4K effective context)
# MODEL_PROFILE_STANDARD:   7B to 14B local model (~8-16K effective context)
# MODEL_PROFILE_HIGH:       27B to 30B local model (~32K effective context)
# MODEL_PROFILE_CUSTOM:     custom values

MODEL_PROFILE_LOW = {
    "max_context":    4_000,
    "max_tool_out":   2_000,
    "max_history":    6,
    "max_files":      30,
    "max_spec":       1_500,
    "max_test_out":   800,
    "system_verbose": False,
}

MODEL_PROFILE_STANDARD = {
    "max_context":    16_000,
    "max_tool_out":   8_000,
    "max_history":    20,
    "max_files":      80,
    "max_spec":       6_000,
    "max_test_out":   3_000,
    "system_verbose": True,
}

MODEL_PROFILE_HIGH = {
    "max_context":    32_000,
    "max_tool_out":   16_000,
    "max_history":    40,
    "max_files":      160,
    "max_spec":       12_000,
    "max_test_out":   6_000,
    "system_verbose": True,
}

MODEL_PROFILE_CUSTOM = {
    "max_context":    os.getenv("MAX_CONTEXT"),
    "max_tool_out":   os.getenv("MAX_TOOL_OUT"),
    "max_history":    os.getenv("MAX_HISTORY"),
    "max_files":      os.getenv("MAX_FILES"),
    "max_spec":       os.getenv("MAX_SPEC"),
    "max_test_out":   os.getenv("MAX_TEST_OUT"),
    "system_verbose": os.getenv("SYSTEM_VERBOSE"),
}