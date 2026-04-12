#Tool used to repair broken calls from LLMs that doesnt respect the tool calling format
from __future__ import annotations

import json
import re
from typing import Any

# Matches both <tool_call>{...}</tool_call> and raw {...} blobs
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?})\s*</tool_call>"   # XML-wrapped
    r"|(\{[^{}]*\"name\"\s*:[^{}]*\"arguments\"\s*:[^{}]*\})",  # bare JSON
    re.DOTALL,
)


def repair_tool_calls(failed_generation: str) -> list[dict[str, Any]] | None:
    """
    Try to recover tool calls from a failed_generation string.

    Returns a list of {name, args} dicts on success, None if unrecoverable.
    """
    repaired = []

    for match in _TOOL_CALL_RE.finditer(failed_generation):
        raw = match.group(1) or match.group(2)
        if not raw:
            continue

        parsed = _try_parse(raw)
        if parsed is None:
            continue

        name = parsed.get("name")
        args = parsed.get("arguments") or parsed.get("args") or {}

        # arguments can arrive as a JSON string or already a dict
        if isinstance(args, str):
            args = _try_parse(args) or {}

        # Unescape common model mistakes in string values
        args = {k: _unescape_value(v) for k, v in args.items()}

        if name:
            repaired.append({"name": name, "args": args})

    return repaired if repaired else None


# ── helpers ───────────────────────────────────────────────────────────────────

def _try_parse(text: str) -> dict | None:
    """Attempt json.loads, then fall back to a tolerant regex extraction."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: extract key/value pairs with a lenient pattern
    try:
        # Fix unescaped inner quotes by stripping them (lossy but recoverable)
        cleaned = re.sub(r'(?<!\\)"(?![:,\{\}\[\]])', '\\"', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _unescape_value(v: Any) -> Any:
    """Fix double-escaped sequences the model sometimes emits."""
    if not isinstance(v, str):
        return v
    return (
        v
        .replace("\\\\n", "\n")
        .replace("\\\\t", "\t")
        .replace('\\\\"', '"')
        .replace("\\\\$", "$")
    )