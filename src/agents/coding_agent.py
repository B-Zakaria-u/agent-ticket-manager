import os
from src.state import GraphState
from src.config.llm import get_llm
from src.tools.files import get_file_tools
from src.tools.search import get_search_tools
from src.tools.docker.sandbox import run_tests_in_sandbox
from langchain_core.tools import tool
from src.utils.language_detector import detect_language
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.utils.logger import log_llm_interaction, log_chat_interaction
from typing import Any

# ── Per-language coding conventions ─────────────────────────────────────────
_LANG_CONVENTIONS: dict[str, str] = {
    "Python": (
        "Use Python idioms: type hints, dataclasses or Pydantic models, f-strings, "
        "and follow PEP 8. Use pytest for tests."
    ),
    "Java": (
        "Use Java idioms: proper access modifiers, JavaDoc comments, and follow "
        "standard Maven/Gradle project layout (src/main/java, src/test/java). "
        "Use JUnit 5 + Mockito for tests."
    ),
    "Kotlin": (
        "Use Kotlin idioms: data classes, extension functions, coroutines where appropriate. "
        "Follow standard Gradle project layout. Use Kotest or JUnit 5 for tests."
    ),
    "PHP": (
        "Use PHP idioms: PSR-12 coding standard, type declarations, namespaces. "
        "Use PHPUnit for tests. Follow Composer project conventions."
    ),
    "JavaScript": (
        "Use modern JavaScript (ES2020+): const/let, arrow functions, async/await, "
        "destructuring. Use Jest or Mocha for tests. Follow the project's existing "
        "package.json scripts."
    ),
    "TypeScript": (
        "Use TypeScript idioms: strict types, interfaces, generics. "
        "Use Jest or Vitest for tests. Follow the project's tsconfig.json."
    ),
    "Go": (
        "Use Go idioms: error wrapping, table-driven tests, interfaces. "
        "Use the standard testing package. Follow standard Go project layout."
    ),
    "Ruby": (
        "Use Ruby idioms: blocks, modules, descriptive method names. "
        "Use RSpec or Minitest for tests. Follow Bundler conventions."
    ),
    "C#": (
        "Use C# idioms: async/await, LINQ, proper exception handling. "
        "Use xUnit or NUnit for tests. Follow .NET project conventions."
    ),
    "Swift": (
        "Use Swift idioms: optionals, value types, protocols. "
        "Use XCTest for tests. Follow Swift Package Manager conventions."
    ),
    "Rust": (
        "Use Rust idioms: ownership, Result/Option types, iterators. "
        "Use the built-in test framework (#[test]). Follow Cargo conventions."
    ),
}

_TEST_FRAMEWORK_HINTS: dict[str, dict[str, str]] = {
    "Python": {
        "framework": "pytest",
        "docker_image": "python:3.13-slim",
        "file_pattern": "test_<module>.py in tests/",
        "script_hint": (
            "#!/bin/bash\n"
            "pip install -r /workspace/requirements.txt 2>/dev/null || true\n"
            "pip install pytest pytest-cov\n"
            "cd /workspace/{workspace_path} && pytest -v"
        ),
    },
    "Java": {
        "framework": "JUnit 5 + Mockito",
        "docker_image": "eclipse-temurin:21-jdk",
        "file_pattern": "<Module>Test.java in src/test/java/",
        "script_hint": (
            "#!/bin/bash\n"
            "cd /workspace/{workspace_path} && mvn test -B"
            "  # OR: ./gradlew test"
        ),
    },
    "Kotlin": {
        "framework": "Kotest or JUnit 5",
        "docker_image": "eclipse-temurin:21-jdk",
        "file_pattern": "<Module>Test.kt in src/test/kotlin/",
        "script_hint": (
            "#!/bin/bash\n"
            "cd /workspace/{workspace_path} && ./gradlew test"
        ),
    },
    "PHP": {
        "framework": "PHPUnit",
        "docker_image": "php:8.3-cli",
        "file_pattern": "<Module>Test.php in tests/",
        "script_hint": (
            "#!/bin/bash\n"
            "apt-get update -y && apt-get install -y php php-cli composer 2>/dev/null\n"
            "cd /workspace/{workspace_path} && composer install --no-interaction\n"
            "./vendor/bin/phpunit --testdox"
        ),
    },
    "JavaScript": {
        "framework": "Jest",
        "docker_image": "node:20-bookworm-slim",
        "file_pattern": "<module>.test.js in __tests__/ or alongside source",
        "script_hint": (
            "#!/bin/bash\n"
            "cd /workspace/{workspace_path} && npm install\n"
            "npm test"
        ),
    },
    "TypeScript": {
        "framework": "Jest + ts-jest or Vitest",
        "docker_image": "node:20-bookworm-slim",
        "file_pattern": "<module>.test.ts in __tests__/ or alongside source",
        "script_hint": (
            "#!/bin/bash\n"
            "cd /workspace/{workspace_path} && npm install\n"
            "npm test"
        ),
    },
    "Go": {
        "framework": "testing (standard library)",
        "docker_image": "golang:1.22",
        "file_pattern": "<module>_test.go alongside source files",
        "script_hint": (
            "#!/bin/bash\n"
            "cd /workspace/{workspace_path} && go test -v ./..."
        ),
    },
    "Ruby": {
        "framework": "RSpec or Minitest",
        "docker_image": "ruby:3.3-slim",
        "file_pattern": "<module>_spec.rb in spec/ or test_<module>.rb in test/",
        "script_hint": (
            "#!/bin/bash\n"
            "cd /workspace/{workspace_path} && bundle install\n"
            "bundle exec rspec"
        ),
    },
    "C#": {
        "framework": "xUnit or NUnit",
        "docker_image": "mcr.microsoft.com/dotnet/sdk:8.0",
        "file_pattern": "<Module>Tests.cs in <ProjectName>.Tests/",
        "script_hint": (
            "#!/bin/bash\n"
            "cd /workspace/{workspace_path} && dotnet test"
        ),
    },
    "Swift": {
        "framework": "XCTest",
        "docker_image": "swift:5.10",
        "file_pattern": "<Module>Tests.swift in Tests/",
        "script_hint": (
            "#!/bin/bash\n"
            "cd /workspace/{workspace_path} && swift test"
        ),
    },
    "Rust": {
        "framework": "built-in #[test]",
        "docker_image": "rust:1.78-slim",
        "file_pattern": "tests/ or inline #[cfg(test)] modules",
        "script_hint": (
            "#!/bin/bash\n"
            "cd /workspace/{workspace_path} && cargo test"
        ),
    },
}

_DEFAULT_DOCKER_IMAGE = "ubuntu:latest"


def _get_docker_image(language: str) -> str:
    return _TEST_FRAMEWORK_HINTS.get(language, {}).get(
        "docker_image", _DEFAULT_DOCKER_IMAGE
    )


def coding_agent_node(state: GraphState) -> dict:
    """
    TDD Senior Engineer: implements the specification with unit tests.
    Writes code + unit tests, runs them in a sandbox, and iterates until
    all tests pass (max 25 turns).
    Scope: unit tests only.
    """
    llm                = get_llm()
    spec               = state.get("spec", "")
    test_output        = state.get("test_output", "")
    iteration_count    = state.get("iteration_count", 0)
    log_file_path      = state.get("log_file_path", "")
    chat_log_file_path = state.get("chat_log_file_path", "")
    total_tokens       = state.get("total_tokens", 0)

    # ── Read model profile from state (injected by orchestrator) ─────────────
    profile      = state.get("model_profile", {})
    MAX_TOOL_OUT = int(profile.get("max_tool_out",   2_000))
    MAX_HISTORY  = int(profile.get("max_history",    6))
    MAX_FILES    = int(profile.get("max_files",      30))
    max_spec     = int(profile.get("max_spec",       1_500))
    max_test_out = int(profile.get("max_test_out",   800))
    verbose      = profile.get("system_verbose")

    # ── Workspace ────────────────────────────────────────────────────────────
    repo_url       = state.get("repo_url", "")
    base_workspace = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    )
    workspace_dir = (
        os.path.join(base_workspace, repo_url.split("/")[-1].replace(".git", ""))
        if repo_url else base_workspace
    )

    file_tools   = get_file_tools(workspace_dir)
    search_tools = get_search_tools()

    # ── Language detection ───────────────────────────────────────────────────
    detected_language  = state.get("detected_language") or ""
    detected_framework = state.get("detected_framework") or ""
    if not detected_language or detected_language == "Unknown":
        lang_info          = detect_language(workspace_dir)
        detected_language  = lang_info.get("language", "Unknown")
        detected_framework = lang_info.get("framework", "Unknown")

    docker_image = _get_docker_image(detected_language)

    @tool
    def run_tests() -> str:
        """
        Run the unit test suite inside a Docker sandbox.
        Call this after writing/modifying files to verify correctness.
        Returns the full test output including pass/fail details.
        """
        return run_tests_in_sandbox.invoke({
            "workspace_path": workspace_dir,
            "image_name":     docker_image,
        })

    all_tools             = file_tools + search_tools + [run_tests]
    #     llm_with_tools_forced = llm.bind_tools(all_tools, tool_choice="any")
    llm_with_tools_forced = llm.bind_tools(all_tools, tool_choice="auto")
    llm_with_tools        = llm.bind_tools(all_tools)

    # ── Workspace file listing — capped by profile ───────────────────────────
    ignore_dirs = {
        ".git", "__pycache__", "node_modules", ".venv",
        "venv", "env", "target", "build", "dist", ".gradle",
    }
    workspace_files = []
    if os.path.exists(workspace_dir):
        for root, dirs, files in os.walk(workspace_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in files:
                workspace_files.append(
                    os.path.relpath(os.path.join(root, f), workspace_dir)
                )
    if len(workspace_files) > MAX_FILES:
        workspace_files = workspace_files[:MAX_FILES] + [
            f"... ({len(workspace_files) - MAX_FILES} more not shown)"
        ]
    file_list_str = "\n".join(f"- {f}" for f in workspace_files)

    # ── Language hints ───────────────────────────────────────────────────────
    lang_conventions = _LANG_CONVENTIONS.get(detected_language, "")
    lang_hints       = _TEST_FRAMEWORK_HINTS.get(detected_language, {})
    test_framework   = lang_hints.get("framework", "the standard test framework for the detected language")
    file_pattern     = lang_hints.get("file_pattern", "tests/")
    script_hint      = lang_hints.get(
        "script_hint", "#!/bin/bash\n# Install dependencies and run unit tests"
    )

    # ── System prompt — verbose when profile allows ──────────────────────────
    if verbose:
        system_content = (
            "You are a senior Software Engineer practising strict TDD.\n"
            "Your deliverable is: working implementation code + create unit tests + passing unit tests.\n\n"
            "RULES:\n"
            "• Use 'read_file' to examine existing code.\n"
            "• Use 'write_file' to create or modify ANY file. "
            "  The tool OVERWRITES the entire file — always supply the complete content.\n"
            "• Both tools accept 'file_path' (preferred) or 'path' as the filename argument.\n"
            "• Never output partial code or placeholders like '# rest unchanged'.\n"
            "• You may call 'write_file' multiple times per turn.\n"
            "• All file paths must be relative to the workspace root.\n"
            f"• Use {detected_language} idioms and the {test_framework} test framework.\n"
            "• If you encounter an ImportError or missing symbol, search the codebase for where it should be defined.\n"
            "• DO NOT write functional, E2E, or performance tests — those are handled by the QA agent."
        )
    else:
        system_content = (
            f"You are a senior {detected_language} engineer doing strict TDD.\n"
            f"Test framework: {test_framework}\n"
            "Rules: read_file before writing, write_file sends full file content always, "
            "relative paths only, unit tests only — no E2E or functional tests."
        )

    # ── Human prompt — spec and test output capped by profile ────────────────
    spec_text = spec if len(spec) <= max_spec else spec[:max_spec] + "\n...[spec truncated]"

    if verbose:
        lang_note = ""
        if detected_language != "Unknown":
            lang_note = f"\nDetected language : {detected_language}"
            if detected_framework != "Unknown":
                lang_note += f" / framework: {detected_framework}"
            if lang_conventions:
                lang_note += f"\nConventions       : {lang_conventions}"
            lang_note += "\n"

        prompt = (
            f"ACTION: Implement the following specification.\n\n"
            f"{spec_text}\n\n"
            f"Workspace root is the current directory. "
            f"All file paths must be strictly relative to it.\n"
            f"Existing files in workspace:\n{file_list_str}\n"
            f"{lang_note}\n"
            f"UNIT TEST FRAMEWORK : {test_framework}\n"
            f"TEST FILE PATTERN   : {file_pattern}\n\n"
            f"script.sh reference template for {detected_language}:\n"
            f"```\n{script_hint}\n```\n\n"
            "WORKFLOW:\n"
            "1. Read any relevant existing files with 'read_file'.\n"
            "2. Write the implementation file(s) with 'write_file'.\n"
            "3. Write unit tests covering all logic branches with 'write_file'.\n"
            "4. All unit tests must be be generated and pass before proceeding.\n"
            "5. Run the tests with 'run_tests'.\n"
            "6. If any test fails, fix the code and repeat from step 4.\n"
            "NEGATIVE RULES (CRITICAL):\n"
            "- DO NOT write functional, E2E, or performance tests here.\n"
            "- DO NOT create two identical files with different names.\n"
            "- DO NOT duplicate code for the same functionality.\n"
        )
    else:
        prompt = (
            f"Implement this spec:\n{spec_text}\n\n"
            f"Workspace files:\n{file_list_str}\n\n"
            f"Test pattern: {file_pattern}\n"
            f"script.sh template:\n```\n{script_hint}\n```\n\n"
            "Steps: read relevant files → write implementation → write tests → run_tests → fix if failing."
        )

    if test_output:
        tail = test_output[-max_test_out:]
        prompt += f"\n\nLast test output:\n{tail}"
        if "[SANDBOX FAIL]" in test_output and "Docker" in test_output:
            prompt += "\nSandbox/environment error — fix script.sh dependencies."
        else:
            prompt += "\nFix failing tests. Read the implementation file first."

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=prompt),
    ]

    current_request_tokens = 0
    tools_called           = 0
    last_test_output       = test_output
    tests_passed           = False
    MAX_TOOL_TURNS         = 25
    tool_names_map         = {t.name: t for t in all_tools}

    for turn in range(MAX_TOOL_TURNS):
        if chat_log_file_path and turn == 0:
            log_chat_interaction(
                chat_log_file_path,
                f"Coding Agent (Turn {turn + 1} Prompt)",
                messages,
            )

        try:
            active_llm = llm_with_tools_forced if turn == 0 else llm_with_tools
            response   = active_llm.invoke(messages)
        except Exception as e:
            print(f"[ Coding Agent ] API Error on turn {turn + 1}: {e}")
            return {
                "iteration_count":    iteration_count + 1,
                "detected_language":  detected_language,
                "detected_framework": detected_framework,
                "total_tokens":       total_tokens + current_request_tokens,
                "test_output":        f"API Error: {e}",
                "tests_passed":       False,
            }

        usage    = getattr(response, "usage_metadata", {}) or {}
        p_tokens = usage.get("input_tokens", 0)
        c_tokens = usage.get("output_tokens", 0)
        current_request_tokens += p_tokens + c_tokens

        if log_file_path:
            model = getattr(llm, "model", getattr(llm, "model_name", "unknown-model"))
            log_llm_interaction(log_file_path, "Coding Agent", model, p_tokens, c_tokens)
        if chat_log_file_path:
            log_chat_interaction(
                chat_log_file_path,
                f"Coding Agent (Turn {turn + 1} Response)",
                response,
            )

        messages.append(response)

        # ── No tool calls ────────────────────────────────────────────────
        if not response.tool_calls:
            if turn == 0 or tools_called == 0:
                print(f"[ Coding Agent ] Turn {turn + 1}: nudging...")
                if messages and messages[-1].type == "ai":
                    messages.pop()
                nudge = (
                    "Use write_file to create code and tests, then run_tests to verify."
                    if not verbose else
                    "SYSTEM NUDGE: You MUST use tools to complete this task. "
                    "Call 'write_file' to create or update source code and test files, "
                    "and 'run_tests' to verify. Do NOT describe code in plain text. "
                    "Make sure all tests pass before finishing."
                )
                if messages and messages[-1].type == "human":
                    if "write_file" not in messages[-1].content[-100:]:
                        messages[-1].content += f"\n\n{nudge}"
                else:
                    messages.append(HumanMessage(content=nudge))
                continue
            else:
                if not tests_passed and last_test_output == test_output:
                    print(f"[ Coding Agent ] Turn {turn + 1}: files written but tests not run — nudging...")
                    if messages and messages[-1].type == "ai":
                        messages.pop()
                    nudge = "You wrote files but never called run_tests. Call run_tests now to verify."
                    if messages and messages[-1].type == "human":
                        messages[-1].content += f"\n\n{nudge}"
                    else:
                        messages.append(HumanMessage(content=nudge))
                    continue
                print(f"[ Coding Agent ] Turn {turn + 1}: done.")
                break

        tools_called += len(response.tool_calls)
        print(f"[ Coding Agent ] Turn {turn + 1}: {len(response.tool_calls)} tool call(s)...")

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id   = tool_call["id"]

            matched_tool = tool_names_map.get(tool_name)
            if matched_tool is None:
                result = f"Error: Tool '{tool_name}' not found. Available: {list(tool_names_map.keys())}"
                print(f"  -> Tool '{tool_name}' NOT FOUND")
            else:
                print(f"  -> {tool_name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in tool_args.items())})")
                try:
                    result = str(matched_tool.invoke(tool_args))
                    if tool_name == "run_tests":
                        last_test_output = result
                        tests_passed     = result.startswith("[SANDBOX OK]")
                    preview = result[:200].replace("\n", "\\n")
                    print(f"     {preview}{'...' if len(result) > 200 else ''}")
                    # ── Cap tool output by profile ───────────────────────
                    if len(result) > MAX_TOOL_OUT:
                        half   = MAX_TOOL_OUT // 2
                        result = result[:half] + "\n...[OUTPUT TRUNCATED]...\n" + result[-half:]
                except Exception as e:
                    result = f"Error executing tool '{tool_name}': {e}"
                    print(f"     Tool error: {e}")

            messages.append(ToolMessage(content=result, tool_call_id=tool_id))

        # ── Context pruning — capped by profile ──────────────────────────
        if len(messages) > 2 + MAX_HISTORY:
            core   = messages[:2]
            recent = messages[2:][-MAX_HISTORY:]
            while recent and recent[0].type == "tool":
                recent.pop(0)
            messages = core + recent

        if tests_passed:
            print(f"[ Coding Agent ] Tests passed on turn {turn + 1}!")
            break
    else:
        print(f"[ Coding Agent ] Reached max tool turns ({MAX_TOOL_TURNS}).")

    return {
        "iteration_count":    iteration_count + 1,
        "detected_language":  detected_language,
        "detected_framework": detected_framework,
        "total_tokens":       total_tokens + current_request_tokens,
        "test_output":        last_test_output,
        "tests_passed":       tests_passed,
    }