import os
from src.state import GraphState
from src.config.llm import get_llm
from src.utils.logger import log_llm_interaction, log_chat_interaction
from src.utils.language_detector import detect_language
from src.tools.folders import initiate_directory
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.tools.files import get_file_tools


# ── QA tool catalogue ──────────────────────────────────────────────────────────
_QA_TOOLS: dict[str, dict[str, str]] = {
    "selenium": {
        "concern": "Web UI interaction, button clicks, form submission, DOM validation, navigation",
        "install": "pip install selenium webdriver-manager pytest",
        "script_hint": (
            "#!/bin/bash\n"
            "pip install selenium webdriver-manager pytest\n"
            "cd /workspace/{workspace_path} && pytest -v tests/test_ui_*"
        ),
    },
    "playwright": {
        "concern": "Modern web UI, cross-browser E2E, screenshot/visual regression, JS-heavy SPAs",
        "install": "pip install playwright pytest-playwright && playwright install",
        "script_hint": (
            "#!/bin/bash\n"
            "pip install playwright pytest-playwright\n"
            "playwright install --with-deps chromium\n"
            "cd /workspace/{workspace_path} && pytest -v tests/test_e2e_*"
        ),
    },
    "cypress": {
        "concern": "JavaScript/TypeScript front-end E2E, real-time reload, component testing",
        "install": "npm install cypress --save-dev",
        "script_hint": (
            "#!/bin/bash\n"
            "apt-get update -y && apt-get install -y nodejs npm 2>/dev/null\n"
            "cd /workspace/{workspace_path} && npm install\n"
            "npx cypress run"
        ),
    },
    "appium": {
        "concern": "Native/hybrid mobile app testing (iOS & Android), gestures, device simulation",
        "install": "pip install Appium-Python-Client pytest",
        "script_hint": (
            "#!/bin/bash\n"
            "pip install Appium-Python-Client pytest\n"
            "# Appium server must be running: appium &\n"
            "cd /workspace/{workspace_path} && pytest -v tests/test_mobile_*"
        ),
    },
    "jmeter": {
        "concern": "Load testing, stress testing, throughput, latency, API performance at scale",
        "install": "apt-get install -y default-jre && wget apache-jmeter archive",
        "script_hint": (
            "#!/bin/bash\n"
            "apt-get update -y && apt-get install -y default-jre wget tar 2>/dev/null\n"
            "wget -q https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.6.3.tgz\n"
            "tar -xzf apache-jmeter-5.6.3.tgz\n"
            "cd /workspace/{workspace_path} && ../apache-jmeter-5.6.3/bin/jmeter -n -t tests/load_test.jmx -l results.jtl"
        ),
    },
    "locust": {
        "concern": "Python-native load/stress testing, concurrent users, ramp-up scenarios",
        "install": "pip install locust",
        "script_hint": (
            "#!/bin/bash\n"
            "pip install locust\n"
            "cd /workspace/{workspace_path} && locust -f tests/locustfile.py --headless -u 100 -r 10 --run-time 1m"
        ),
    },
    "requests_pytest": {
        "concern": "REST/GraphQL API contract testing, status codes, response schema, auth flows",
        "install": "pip install requests pytest jsonschema",
        "script_hint": (
            "#!/bin/bash\n"
            "pip install requests pytest jsonschema\n"
            "cd /workspace/{workspace_path} && pytest -v tests/test_api_*"
        ),
    },
    "postman_newman": {
        "concern": "Existing Postman collections, CI-friendly API regression runs",
        "install": "npm install -g newman",
        "script_hint": (
            "#!/bin/bash\n"
            "apt-get update -y && apt-get install -y nodejs npm 2>/dev/null\n"
            "npm install -g newman\n"
            "cd /workspace/{workspace_path} && newman run tests/collection.json -e tests/environment.json"
        ),
    },
    "owasp_zap": {
        "concern": "Security scanning, XSS, SQL injection, OWASP Top-10 vulnerability detection",
        "install": "docker pull ghcr.io/zaproxy/zaproxy:stable",
        "script_hint": (
            "#!/bin/bash\n"
            "apt-get update -y && apt-get install -y docker.io 2>/dev/null\n"
            "docker run -t ghcr.io/zaproxy/zaproxy:stable zap-baseline.py -t http://target-url"
        ),
    },
    "axe_playwright": {
        "concern": "WCAG accessibility compliance, screen-reader compatibility, colour contrast",
        "install": "pip install playwright pytest-playwright axe-playwright-python",
        "script_hint": (
            "#!/bin/bash\n"
            "pip install playwright pytest-playwright axe-playwright-python\n"
            "playwright install --with-deps chromium\n"
            "cd /workspace/{workspace_path} && pytest -v tests/test_accessibility_*"
        ),
    },
    "db_pytest": {
        "concern": "Data integrity, migrations, stored procedures, query correctness",
        "install": "pip install pytest sqlalchemy psycopg2-binary",
        "script_hint": (
            "#!/bin/bash\n"
            "pip install pytest sqlalchemy psycopg2-binary\n"
            "cd /workspace/{workspace_path} && pytest -v tests/test_db_*"
        ),
    },
}

_LANGUAGE_RUNNER: dict[str, dict[str, str]] = {
    "Python":     {"runner": "pytest",               "file_pattern": "tests/test_<concern>_*.py"},
    "Java":       {"runner": "mvn test",             "file_pattern": "src/test/java/**/*Test.java"},
    "JavaScript": {"runner": "npm test",             "file_pattern": "tests/*.test.js"},
    "TypeScript": {"runner": "npm test",             "file_pattern": "tests/*.test.ts"},
    "PHP":        {"runner": "./vendor/bin/phpunit", "file_pattern": "tests/*Test.php"},
    "Ruby":       {"runner": "bundle exec rspec",    "file_pattern": "spec/**/*_spec.rb"},
    "C#":         {"runner": "dotnet test",          "file_pattern": "**/*Tests.cs"},
    "Go":         {"runner": "go test ./...",        "file_pattern": "*_test.go"},
}

def _build_tool_catalogue(verbose: bool) -> str:
    """Short catalogue for LOW profile, full detail for STANDARD/HIGH."""
    if verbose:
        return "\n\n".join(
            f"[{key}]\n"
            f"  Concern   : {meta['concern']}\n"
            f"  Install   : {meta['install']}\n"
            f"  script.sh :\n"
            + "\n".join(f"    {line}" for line in meta["script_hint"].splitlines())
            for key, meta in _QA_TOOLS.items()
        )
    else:
        return "\n".join(
            f"  • [{key}] {meta['concern']}"
            for key, meta in _QA_TOOLS.items()
        )

def testing_agent_node(state: GraphState) -> dict:
    """
    Senior QA Engineer: analyses the ticket and spec to select the right QA
    tool(s) — Selenium, Playwright, Cypress, Appium, JMeter, Locust, OWASP ZAP,
    axe, or plain API/DB tests — then generates the test suite and a matching
    script.sh that sets up the environment inside a Docker container.
    """
    repo_url       = state.get("repo_url", "")
    base_workspace = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    )
    workspace_dir = (
        os.path.join(base_workspace, repo_url.split("/")[-1].replace(".git", ""))
        if repo_url else base_workspace
    )
    initiate_directory(workspace_dir)

    llm                = get_llm()
    ticket_text        = state.get("ticket_text", "")
    spec               = state.get("spec", "")
    log_file_path      = state.get("log_file_path", "")
    chat_log_file_path = state.get("chat_log_file_path", "")
    total_tokens       = state.get("total_tokens", 0)

    # ── Read model profile from state (injected by orchestrator) ─────────────
    profile      = state.get("model_profile", {})
    MAX_TOOL_OUT = int(profile.get("max_tool_out",   2_000))
    MAX_HISTORY  = int(profile.get("max_history",    6))
    MAX_FILES    = int(profile.get("max_files",      30))
    max_spec     = int(profile.get("max_spec",       1_500))
    max_ticket   = int(profile.get("max_test_out",   800))   # reuse for ticket truncation
    verbose      = profile.get("system_verbose", False)

    # ── Language detection ───────────────────────────────────────────────────
    detected_language  = state.get("detected_language") or ""
    detected_framework = state.get("detected_framework") or ""
    if not detected_language or detected_language == "Unknown":
        lang_info          = detect_language(workspace_dir)
        detected_language  = lang_info.get("language", "Unknown")
        detected_framework = lang_info.get("framework", "Unknown")

    lang_runner  = _LANGUAGE_RUNNER.get(detected_language, {})
    file_pattern = lang_runner.get("file_pattern", "tests/test_*.py")

    file_tools     = get_file_tools(workspace_dir)
    llm_with_tools = llm.bind_tools(file_tools)

    print(f"[ Testing Agent ] Analysing ticket to select QA tools "
          f"({detected_language}/{detected_framework})...")

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
    file_list_str  = "\n".join(f"- {f}" for f in workspace_files)
    tool_catalogue = _build_tool_catalogue(verbose)

    # ── Serialise full _QA_TOOLS so the agent has install + script_hint ──────
    qa_tools_detail = "\n\n".join(
        f"[{key}]\n"
        f"  Concern    : {meta['concern']}\n"
        f"  Install    : {meta['install']}\n"
        f"  script.sh  :\n"
        + "\n".join(f"    {line}" for line in meta["script_hint"].splitlines())
        for key, meta in _QA_TOOLS.items()
    )

    # Insert this after the profile variables are defined
    ticket_text_t = (
        ticket_text if len(ticket_text) <= max_ticket 
        else ticket_text[:max_ticket] + "\n...[ticket truncated]"
    )
    spec_t = (
        spec if len(spec) <= max_spec 
        else spec[:max_spec] + "\n...[spec truncated]"
    )

    if verbose:
        gen_prompt = f"""You are a senior QA engineer.
DETECTED LANGUAGE  : {detected_language}
DETECTED FRAMEWORK : {detected_framework}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — SELECT THE RIGHT QA TOOL(S)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the ticket and spec carefully, then choose the most appropriate QA
tool(s) from the catalogue below. You may combine tools when the ticket
covers multiple concerns (e.g. a broken UI button → Selenium; an API
that is slow → Locust).

Available QA tools:
{tool_catalogue}

Full tool details (install commands + script.sh templates):
{qa_tools_detail}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — GENERATE THE TEST FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Use 'read_file' to examine existing implementation code.
2. Use 'write_file' to create test files.
   • YOU MUST CALL THE 'write_file' TOOL! DO NOT write test scripts as plain text code blocks in your response.
   • DO NOT write unit tests — focus exclusively on QA-level tests
     (functional, E2E, performance, security, accessibility, etc.)
   • Name files following {detected_language} conventions: {file_pattern}
     Replace <concern> with the concern type (ui, api, load, mobile, etc.)
   • Write tests that directly validate the behaviour described in the spec
     and reproduce/prevent the issue described in the ticket.

NEGATIVE RULES (CRITICAL):
   • DO NOT modify source code or existing implementation files.
   • DO NOT implement the fix described in the ticket. That is the Coding Agent's job.
   • ONLY use 'write_file' for paths starting with 'tests/' or named 'script.sh'.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — CREATE script.sh
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Create a 'script.sh' at the workspace root (NOT inside tests/).
  • The script runs inside a Docker container — install every dependency
    needed for the chosen QA tool(s) before executing the tests.
  • Base it on the script_hint(s) from the selected tool(s) above,
    adapting paths and commands to match the actual workspace layout.

You may call 'write_file' multiple times in a single turn.
All file paths must be relative to the workspace root — never prepend
'workspace/' or an absolute path.

Existing files in workspace:
{file_list_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ISSUE TICKET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ticket_text_t}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECHNICAL SPECIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{spec_t}
"""
        system_content = (
            f"You are a senior QA Engineer specialising in quality assurance. "
            "Your job is to pick the right QA tool for the problem — not to default to any single tool. "
            "Read the ticket, select the best tool(s), write ONLY test files, and produce script.sh. "
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CRITICAL RULES:\n"
            "1. You are NOT a software engineer or developer. DO NOT implement code fixes.\n"
            "2. You MUST NOT modify source code or existing implementation files.\n"
            "3. You ONLY use 'write_file' for paths starting with 'tests/' or 'script.sh'.\n"
            "4. Your goal is to EXPOSE bugs and VALIDATE requirements, never to fix them.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Choices: Selenium (web UI), Playwright (modern E2E), Cypress (JS E2E), "
            "Appium (mobile), JMeter/Locust (performance/load), requests+pytest (API), "
            "OWASP ZAP (security), axe+Playwright (accessibility), SQLAlchemy+pytest (DB). "
            "IMPORTANT: You MUST use the 'write_file' tool to generate files. NEVER output raw test scripts in plain text. "
            "Read the ticket, write the tests, produce script.sh, and call tools. "
            "Note: 'read_file' and 'write_file' both accept 'file_path' (preferred) or 'path' as the filename argument. "
            "Context: All file paths must be relative to the workspace root."
        )
    else:
        gen_prompt = (
            f"Language: {detected_language} / {detected_framework}\n\n"
            f"QA tools available:\n{tool_catalogue}\n\n"
            f"Workspace files:\n{file_list_str}\n\n"
            f"Test file pattern: {file_pattern}\n\n"
            f"Ticket:\n{ticket_text_t}\n\n"
            f"Spec:\n{spec_t}\n\n"
            "Steps: read relevant files → pick QA tool(s) → write_file for tests/ and script.sh."
        )
        system_content = (
            "You are a senior QA engineer.\n"
            "Rules: pick the right QA tool, write test files only (tests/ or script.sh), "
            "never modify source code, never fix bugs.\n"
            "Use write_file — never output test scripts as plain text."
        )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=gen_prompt),
    ]

    current_request_tokens = 0
    tools_called           = 0
    MAX_TOOL_TURNS         = 20
    tool_names_map         = {t.name: t for t in file_tools}

    for turn in range(MAX_TOOL_TURNS):
        if chat_log_file_path and turn == 0:
            log_chat_interaction(
                chat_log_file_path,
                f"Testing Agent (Turn {turn + 1} Prompt)",
                messages,
            )

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            print(f"[ Testing Agent ] API Error on turn {turn + 1}: {e}")
            break

        usage    = response.usage_metadata or {}
        p_tokens = usage.get("input_tokens", 0)
        c_tokens = usage.get("output_tokens", 0)
        current_request_tokens += p_tokens + c_tokens

        if log_file_path:
            model = getattr(llm, "model", getattr(llm, "model_name", "unknown-model"))
            log_llm_interaction(log_file_path, "Testing Agent", model, p_tokens, c_tokens)
        if chat_log_file_path:
            log_chat_interaction(
                chat_log_file_path,
                f"Testing Agent (Turn {turn + 1} Response)",
                response,
            )

        messages.append(response)

        if not response.tool_calls:
            if current_request_tokens == 0 or turn == 0 or tools_called == 0:
                print(f"[ Testing Agent ] Turn {turn + 1}: No tool calls — nudging LLM to use tools...")
                if messages and messages[-1].type == "ai":
                    messages.pop()
                nudge_text = (
                    "SYSTEM NUDGE: You must use the 'write_file' tool to create the test files and script.sh. "
                    "Do NOT output test scripts as plain text code blocks in your response. Call the tool NOW."
                )
                if messages and messages[-1].type == "human":
                    if "SYSTEM NUDGE:" not in getattr(messages[-1], "content", ""):
                        messages[-1].content += f"\n\n{nudge_text}"
                else:
                    messages.append(HumanMessage(content=nudge_text))
                continue
            else:
                print(f"[ Testing Agent ] Turn {turn + 1}: No more tool calls — finishing.")
                break

        tools_called += len(response.tool_calls)
        print(f"[ Testing Agent ] Turn {turn + 1}: Executing {len(response.tool_calls)} tool call(s)...")

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id   = tool_call["id"]

            matched_tool = tool_names_map.get(tool_name)
            if matched_tool is None:
                result = f"Error: Tool '{tool_name}' not found. Available tools: {list(tool_names_map.keys())}"
                print(f"  -> Tool '{tool_name}' NOT FOUND")
            else:
                print(f"  -> Calling tool: {tool_name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in tool_args.items())})")
                try:
                    result = str(matched_tool.invoke(tool_args))
                    preview = result[:200].replace('\n', '\\n')
                    print(f"     Result: {preview}{'...' if len(result) > 200 else ''}")
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
            print(f"[ Testing Agent ] Tests passed on turn {turn + 1}!")
            break
    else:
        print(f"[ Testing Agent ] Reached max tool turns ({MAX_TOOL_TURNS}).")

    return {
        "tests_generated":    True,
        "detected_language":  detected_language,
        "detected_framework": detected_framework,
        "total_tokens":       total_tokens + current_request_tokens,
    }