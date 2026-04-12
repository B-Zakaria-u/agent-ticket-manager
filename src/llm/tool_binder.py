from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from langchain_core.language_models import BaseChatModel

# tool_choice values each provider actually accepts
_FORCED_STRATEGIES = [
    {"tool_choice": "required"},       # OpenAI, Groq
    {"tool_choice": "any"},            # Anthropic
    {"tool_choice": {"type": "any"}},  # Anthropic (older SDK shape)
]


@dataclass
class ToolBinder:
    """
    Binds tools to an LLM once, probing for the right tool_choice
    convention so the caller never has to think about it.

    Usage:
        binder = ToolBinder(llm, tools)
        response = await binder.invoke(messages, forced=turn == 0)
    """
    llm:   BaseChatModel
    tools: list[Any]

    _forced: BaseChatModel = field(init=False, repr=False)
    _normal: BaseChatModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._normal = self.llm.bind_tools(self.tools)
        self._forced = self._probe_forced()

    # ── public ──────────────────────────────────────────────────────────────

    async def ainvoke(self, messages: list, *, forced: bool = False) -> Any:
        bound = self._forced if forced else self._normal
        return await bound.ainvoke(messages)

    def rebind(self, tools: list[Any]) -> "ToolBinder":
        """Return a new ToolBinder with a different tool set (e.g. after MCP load)."""
        return ToolBinder(llm=self.llm, tools=tools)

    # ── internal ─────────────────────────────────────────────────────────────

    def _probe_forced(self) -> BaseChatModel:
        """
        Try each known forced-tool-call convention in priority order.
        Falls back to normal binding if none work (some local models
        don't support tool_choice at all).
        """
        for kwargs in _FORCED_STRATEGIES:
            try:
                bound = self.llm.bind_tools(self.tools, **kwargs)
                # bind_tools is lazy — do a cheap attribute check to
                # catch providers that raise on construction
                _ = bound.bound if hasattr(bound, "bound") else bound
                return bound
            except (TypeError, ValueError, NotImplementedError):
                continue

        # graceful degradation: forced == normal for this provider
        return self._normal