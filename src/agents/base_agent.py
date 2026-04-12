"""
BaseAgent — shared agentic loop (v2).

Changes from v1
---------------
  1. MCP tools loaded automatically and merged with native tools each run.
  2. Agents emit an AgentReport via `emit_report()` at the end of every run —
     the orchestrator reads this to decide what happens next.
  3. All async hygiene fixed:
       - on_tool_result()       is now awaited
       - extra_state_updates()  is now awaited
       - get_tools()            is now consistently awaited
       - `llm_with_tools` undefined ref removed
       - double messages.append(response) removed
  4. `uses_tests` flag lets subclasses opt-out of the "tests not run" nudge.

To create a new agent, subclass this and implement:
  - agent_name      (str property)       → maps to resources/agents/<name>/
  - build_context() (→ dict)             → all {placeholders} for prompts
  - get_tools()     (→ list, async)      → native LangChain tools for this agent

Optionally override:
  - on_tool_result()                     → react to a specific tool's output
  - extra_state_updates()                → extra keys merged into returned dict
  - uses_tests   = False                 → suppress "tests not run" nudge
  - build_report()                       → customise the AgentReport content
"""
from __future__ import annotations

import abc
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.config.agent_config import AgentConfig
from src.llm.base_config import get_llm
from src.mcp.client import MCPClientManager
from src.state import AgentReport, GraphState
from src.utils.logger import log_chat_interaction, log_llm_interaction
from src.llm.tool_binder import ToolBinder
from src.llm.tool_call_repair import repair_tool_calls

class BaseAgent(abc.ABC):

    # Subclasses may set this to False to skip the "tests not run" nudge
    uses_tests: bool = True

    def __init__(self) -> None:
        self._files_written: bool = False
        self._issues:        list[str] = []
        self._artifacts:     list[str] = []

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        return await self.run(state)

    # ── Subclasses must implement ─────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def agent_name(self) -> str:
        """Matches the resources/agents/<agent_name>/ directory."""

    @abc.abstractmethod
    async def build_context(self, state: GraphState) -> dict[str, Any]:
        """Return the render context dict used by all prompt templates."""

    @abc.abstractmethod
    async def get_tools(self, state: GraphState) -> list:
        """Return the list of native LangChain tools for this agent."""

    # ── Optional hooks ────────────────────────────────────────────────────────

    async def on_tool_result(
        self, tool_name: str, result: str, state: GraphState
    ) -> None:
        """Called after every successful tool execution."""
        if tool_name == "write_file":
            self._files_written = True
            # Track written artifact paths when the result carries a path hint
            if "written" in result.lower() or "saved" in result.lower():
                self._artifacts.append(result.split()[-1])

    async def extra_state_updates(self, state: GraphState) -> dict[str, Any]:
        """Merged into the final returned dict. Override for agent-specific keys."""
        return {}

    async def build_report(
        self,
        *,
        status: str,
        summary: str,
        state: GraphState,
        tokens: int,
    ) -> AgentReport:
        """
        Override to customise the AgentReport emitted to the orchestrator.
        The base implementation covers the common fields; subclasses can call
        super() and extend the returned dict.
        """
        return {
            "agent":       self.agent_name,
            "status":      status,           # "success" | "failed" | "partial" | "blocked"
            "summary":     summary,
            "artifacts":   list(self._artifacts),
            "issues":      list(self._issues),
            "suggestions": self._default_suggestions(status, state),
            "tokens":      tokens,
        }

    # ── Core loop ─────────────────────────────────────────────────────────────

    async def run(self, state: GraphState) -> dict[str, Any]:
        cfg = AgentConfig(self.agent_name)
        llm = get_llm()

        # ── Profile ───────────────────────────────────────────────────────────
        profile      = state.get("model_profile", {})
        MAX_TOOL_OUT = int(profile.get("max_tool_out", 2_000))
        MAX_HISTORY  = int(profile.get("max_history",  6))
        verbose      = bool(profile.get("system_verbose"))
        max_test_out = int(profile.get("max_test_out", 800))

        # ── Logging ───────────────────────────────────────────────────────────
        log_file      = state.get("log_file_path", "")
        chat_log_file = state.get("chat_log_file_path", "")
        iteration     = state.get("iteration_count", 0)
        total_tokens  = state.get("total_tokens", 0)

        # ── Prompts ───────────────────────────────────────────────────────────
        ctx            = await self.build_context(state)
        system_content = cfg.system(verbose, ctx)
        human_content  = cfg.human(verbose, ctx)

        test_output = state.get("test_output", "")
        if test_output:
            human_content += cfg.test_output_suffix(test_output, max_test_out)

        messages: list = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

        # ── Tools: native + MCP ───────────────────────────────────────────────
        native_tools = await self.get_tools(state)
        mcp_tools    = await MCPClientManager.get_tools(state.get("mcp_servers"))
        tools        = native_tools + mcp_tools
        tool_map     = {t.name: t for t in tools}

        if mcp_tools:
            print(
                f"[ {self.agent_name} ] Loaded {len(mcp_tools)} MCP tool(s): "
                f"{[t.name for t in mcp_tools]}"
            )

        binder = ToolBinder(llm, tools)

        # ── Tracking ──────────────────────────────────────────────────────────
        request_tokens = 0
        tools_called   = 0
        last_test_out  = test_output
        tests_passed   = False
        MAX_TURNS      = 100

        for turn in range(MAX_TURNS):
            # 1. Select LLM variant and invoke
            try:
                response = await binder.ainvoke(messages, forced=(turn == 0))
            except Exception as exc:
                err_body = getattr(exc, "body", {}) or {}
                err_obj  = (err_body.get("error") or {}) if isinstance(err_body, dict) else {}

                if err_obj.get("code") == "tool_use_failed":
                    failed_gen = err_obj.get("failed_generation", "")
                    recovered  = repair_tool_calls(failed_gen)

                    if recovered:
                        print(
                            f"[ {self.agent_name} ] Turn {turn + 1}: "
                            f"repaired {len(recovered)} malformed tool call(s)."
                        )
                        from langchain_core.messages import AIMessage
                        response = AIMessage(
                            content="",
                            tool_calls=[
                                {"name": r["name"], "args": r["args"], "id": f"repaired_{i}"}
                                for i, r in enumerate(recovered)
                            ],
                        )
                        # skip token tracking — no usage metadata on a repaired call
                        messages.append(response)
                        goto_tool_exec = True
                    else:
                        print(
                            f"[ {self.agent_name} ] Turn {turn + 1}: "
                            "tool_use_failed unrecoverable — nudging simplification."
                        )
                        _append_nudge(
                            messages,
                            "Your last tool call had a formatting error. "
                            "Avoid escaped quotes or shell special characters "
                            "inside JSON arguments — pass raw content only.",
                        )
                        continue
                else:
                    err_msg = f"API Error on turn {turn + 1}: {exc}"
                    print(f"[ {self.agent_name} ] {err_msg}")
                    self._issues.append(err_msg)
                    report = await self._finalise_report(
                        status="failed", summary=err_msg, state=state,
                        tokens=request_tokens,
                    )
                    return {
                        "iteration_count":    iteration + 1,
                        "total_tokens":       total_tokens + request_tokens,
                        "test_output":        err_msg,
                        "tests_passed":       False,
                        "agent_outcome":      "failed",
                        "orchestrator_inbox": report,
                        "agent_reports":      _append_report(state, report),
                        **await self.extra_state_updates(state),
                    }
            else:
                goto_tool_exec = False

            # 2. Token tracking
            usage = {} if goto_tool_exec else (getattr(response, "usage_metadata", {}) or {})
            in_tok  = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            request_tokens += in_tok + out_tok
            
            messages.append(response)

            # 3. Logging
            if log_file:
                model = getattr(llm, "model", getattr(llm, "model_name", "unknown"))
                log_llm_interaction(log_file, self.agent_name, model, in_tok, out_tok)
            if chat_log_file and turn == 0:
                log_chat_interaction(
                    chat_log_file,
                    f"{self.agent_name} (Turn 1 Prompt)",
                    messages[:-1],  # log the prompt, not the response
                )
            if chat_log_file:
                log_chat_interaction(
                    chat_log_file,
                    f"{self.agent_name} (Turn {turn + 1} Response)",
                    response,
                )

            # ── No tool calls → check exit / nudge conditions ─────────────────
            if not response.tool_calls:
                if turn == 0 or tools_called == 0:
                    print(
                        f"[ {self.agent_name} ] Turn {turn + 1}: "
                        "no tool call on first attempt — nudging..."
                    )
                    messages.pop()
                    _append_nudge(messages, cfg.nudge(verbose))
                    continue

                if (
                    self.uses_tests
                    and not tests_passed
                    and self._files_written
                    and last_test_out == test_output
                ):
                    print(
                        f"[ {self.agent_name} ] Turn {turn + 1}: "
                        "files written but tests not run — nudging..."
                    )
                    messages.pop()
                    _append_nudge(messages, cfg.nudge(verbose, kind="tests_not_run"))
                    continue

                print(f"[ {self.agent_name} ] Turn {turn + 1}: done (no more tools).")
                break

            # ── Execute tool calls ────────────────────────────────────────────
            tools_called += len(response.tool_calls)
            print(
                f"[ {self.agent_name} ] Turn {turn + 1}: "
                f"{len(response.tool_calls)} tool call(s)..."
            )

            for call in response.tool_calls:
                name, args, call_id = call["name"], call["args"], call["id"]
                matched = tool_map.get(name)

                if matched is None:
                    result = (
                        f"Error: tool '{name}' not found. "
                        f"Available: {list(tool_map)}"
                    )
                    print(f"  -> '{name}' NOT FOUND")
                    self._issues.append(f"Unknown tool called: {name}")
                else:
                    arg_preview = ", ".join(
                        f"{k}={repr(v)[:60]}" for k, v in args.items()
                    )
                    print(f"  -> {name}({arg_preview})")
                    try:
                        raw_result = await matched.ainvoke(args)
                        result = str(raw_result)
                        preview = result[:200].replace("\n", "\\n")
                        print(
                            f"     {preview}"
                            f"{'...' if len(result) > 200 else ''}"
                        )

                        if len(result) > MAX_TOOL_OUT:
                            half   = MAX_TOOL_OUT // 2
                            result = (
                                result[:half]
                                + "\n...[OUTPUT TRUNCATED]...\n"
                                + result[-half:]
                            )

                        if name == "run_tests":
                            last_test_out = result
                            tests_passed  = result.startswith("[SANDBOX OK]")
                            if not tests_passed:
                                self._issues.append(
                                    f"Tests failed: {result[:300]}"
                                )

                        await self.on_tool_result(name, result, state)

                    except Exception as exc:
                        result = f"Error executing '{name}': {exc}"
                        print(f"     Tool error: {exc}")
                        self._issues.append(result)

                messages.append(
                    ToolMessage(content=result, tool_call_id=call_id)
                )

            # ── Prune conversation history ─────────────────────────────────────
            if len(messages) > 2 + MAX_HISTORY:
                recent = messages[2:][-MAX_HISTORY:]
                # Never start with an orphaned ToolMessage
                while recent and recent[0].type == "tool":
                    recent.pop(0)
                # Never start with an AIMessage that has unresolved tool_calls
                while recent and getattr(recent[0], "tool_calls", None):
                    recent.pop(0)
                messages = messages[:2] + recent

            if tests_passed:
                print(
                    f"[ {self.agent_name} ] Tests passed on turn {turn + 1}!"
                )
                break

        else:
            print(
                f"[ {self.agent_name} ] Reached max turns ({MAX_TURNS})."
            )
            self._issues.append(f"Reached max turns ({MAX_TURNS}) without completing.")

        # ── Build final report ────────────────────────────────────────────────
        if self.uses_tests:
            status  = "success" if tests_passed else "failed"
            summary = (
                "All tests passed."
                if tests_passed
                else f"Tests did not pass. Last output: {last_test_out[:300]}"
            )
        else:
            status  = "success" if self._files_written else "partial"
            summary = (
                f"Agent completed. Files written: {self._files_written}. "
                f"Issues: {len(self._issues)}."
            )

        report = await self._finalise_report(
            status=status, summary=summary, state=state, tokens=request_tokens,
        )

        return {
            "messages":        [messages[-1]],
            "iteration_count": iteration + 1,
            "total_tokens":    total_tokens + request_tokens,
            "test_output":     last_test_out,
            "tests_passed":    tests_passed,
            "agent_outcome":   status,
            # ── Orchestrator communication ─────────────────────────────────
            "orchestrator_inbox": report,
            "agent_reports":      _append_report(state, report),
            **await self.extra_state_updates(state),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _finalise_report(
        self,
        *,
        status: str,
        summary: str,
        state: GraphState,
        tokens: int,
    ) -> AgentReport:
        return await self.build_report(
            status=status, summary=summary, state=state, tokens=tokens,
        )

    def _default_suggestions(self, status: str, state: GraphState) -> list[str]:
        """Heuristic suggestions based on outcome — orchestrator uses these."""
        if status == "success":
            return ["proceed_to_next_pipeline_step"]
        if status == "failed":
            return ["retry_agent", "escalate_to_human"]
        if status == "partial":
            return ["continue_current_agent", "proceed_with_caution"]
        return []

# ── Module-level helpers ──────────────────────────────────────────────────────

def _append_nudge(messages: list, nudge: str) -> None:
    """Append a nudge to the last human message or add a new one."""
    if messages and messages[-1].type == "human":
        if nudge not in messages[-1].content[-200:]:
            messages[-1].content += f"\n\n{nudge}"
    else:
        messages.append(HumanMessage(content=nudge))


def _append_report(state: GraphState, report: AgentReport) -> list[AgentReport]:
    """Return a new list with the report appended (immutable-friendly)."""
    existing = list(state.get("agent_reports") or [])
    existing.append(report)
    return existing