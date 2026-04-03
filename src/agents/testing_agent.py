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
    "PHP":         {"runner": "./vendor/bin/phpunit", "file_pattern": "tests/*Test.php"},
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
    tool(s) — then generates the test suite and a matching script.sh.
    Iterates through tool calls until files are written or max turns reached.
    """
    # ── Configuration & State ────────────────────────────────────────────────
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

    # ── Read model profile from state ────────────────────────────────────────
    profile      = state.get("model_profile", {})
    MAX_TOOL_OUT = int(profile.get("max_tool_out",   2_000))
    MAX_HISTORY  = int(profile.get("max_history",    6))
    MAX_FILES    = int(profile.get("max_files",      30))
    max_spec     = int(profile.get("max_spec",       1_500))
    max_ticket   = int(profile.get("max_test_out",   800)) 
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
    tool_names_map = {t.name: t for t in file_tools}
    llm_with_tools = llm.bind_tools(file_tools)

    # ── Workspace file listing — capped by profile ───────────────────────────
    ignore_dirs = {".git", "__pycache__", "node_modules", ".venv"}
    workspace_files = []
    if os.path.exists(workspace_dir):
        for root, dirs, files in os.walk(workspace_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in files:
                workspace_files.append(
                    os.path.relpath(os.path.join(root, f), workspace_dir)
                )
    
    file_list_str = "\n".join(f"- {f}" for f in workspace_files[:MAX_FILES])
    if len(workspace_files) > MAX_FILES:
        file_list_str += f"\n... ({len(workspace_files) - MAX_FILES} more not shown)"

    tool_catalogue = _build_tool_catalogue(verbose)

    # Full tool details (for script_hint generation)
    qa_tools_detail = "\n\n".join(
        f"[{key}]\n Concern: {meta['concern']}\n Install: {meta['install']}\n script.sh :\n"
        + "\n".join(f"    {line}" for line in meta["script_hint"].splitlines())
        for key, meta in _QA_TOOLS.items()
    )

    ticket_text_t = (
        ticket_text if len(ticket_text) <= max_ticket 
        else ticket_text[:max_ticket] + "\n...[ticket truncated]"
    )
    spec_t = (
        spec if len(spec) <= max_spec 
        else spec[:max_spec] + "\n...[spec truncated]"
    )

    # ── Prompts ──────────────────────────────────────────────────────────────
    if verbose:
        gen_prompt = f"""You are a senior QA engineer.
DETECTED LANGUAGE  : {detected_language}
DETECTED FRAMEWORK : {detected_framework}

Available QA tools:
{tool_catalogue}

Full tool details:
{qa_tools_detail}

Existing files in workspace:
{file_list_str}

ISSUE TICKET:
{ticket_text_t}

TECHNICAL SPECIFICATION:
{spec_t}

WORKFLOW:
1. Read implementation with 'read_file'.
2. Write test files and script.sh with 'write_file'.
"""
        system_content = (
            "You are a senior QA Engineer. Pick the right tool(s), write test files only, and produce script.sh.\n"
            "CRITICAL RULES:\n"
            "1. DO NOT implement code fixes. DO NOT modify source code.\n"
            "2. ONLY use 'write_file' for paths starting with 'tests/' or 'script.sh'.\n"
            "3. You MUST use 'write_file'. Relative paths only."
        )
    else:
        gen_prompt = (
            f"Language: {detected_language}\n"
            f"QA tools: {tool_catalogue}\n"
            f"Files:\n{file_list_str}\n"
            f"Ticket: {ticket_text_t}\n"
            f"Spec: {spec_t}\n"
        )
        system_content = "Senior QA: pick tool, write tests/ or script.sh via write_file. No source code changes."

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=gen_prompt),
    ]

    # ── Iterative Loop ───────────────────────────────────────────────────────
    current_request_tokens = 0
    tools_called           = 0
    MAX_TOOL_TURNS         = 20
    files_written          = False

    print(f"[ Testing Agent ] Analysing ticket for {detected_language}...")

    for turn in range(MAX_TOOL_TURNS):
        if chat_log_file_path and turn == 0:
            log_chat_interaction(chat_log_file_path, f"Testing Agent (Turn {turn + 1} Prompt)", messages)

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            print(f"[ Testing Agent ] API Error: {e}")
            break

        usage = getattr(response, "usage_metadata", {}) or {}
        p_tokens, c_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        current_request_tokens += (p_tokens + c_tokens)

        if log_file_path:
            log_llm_interaction(log_file_path, "Testing Agent", getattr(llm, "model", "unknown"), p_tokens, c_tokens)
        if chat_log_file_path:
            log_chat_interaction(chat_log_file_path, f"Testing Agent (Turn {turn + 1} Response)", response)

        messages.append(response)

        # ── Tool calling logic & Nudges ──────────────────────────────────────
        if not response.tool_calls:
            if not files_written:
                nudge = "SYSTEM NUDGE: Use 'write_file' to create tests/ files and script.sh. Call the tool NOW."
                messages.append(HumanMessage(content=nudge))
                continue
            else:
                print(f"[ Testing Agent ] Finished.")
                break

        print(f"[ Testing Agent ] Turn {turn + 1}: Executing {len(response.tool_calls)} tool call(s)...")
        for tool_call in response.tool_calls:
            t_name, t_args, t_id = tool_call["name"], tool_call["args"], tool_call["id"]
            
            if t_name == "write_file":
                files_written = True

            matched_tool = tool_names_map.get(t_name)
            if matched_tool:
                try:
                    result = str(matched_tool.invoke(t_args))
                    if len(result) > MAX_TOOL_OUT:
                        result = result[:MAX_TOOL_OUT//2] + "\n...[TRUNCATED]...\n" + result[-MAX_TOOL_OUT//2:]
                except Exception as e:
                    result = f"Error executing tool '{t_name}': {e}"
            else:
                result = f"Error: Tool '{t_name}' not found."
            
            messages.append(ToolMessage(content=result, tool_call_id=t_id))

        # ── Context pruning ──────────────────────────────────────────────────
        if len(messages) > 2 + MAX_HISTORY:
            core = messages[:2]
            recent = messages[2:][-MAX_HISTORY:]
            while recent and recent[0].type == "tool":
                recent.pop(0)
            messages = core + recent

    return {
        "tests_generated":   files_written,
        "detected_language":  detected_language,
        "detected_framework": detected_framework,
        "total_tokens":       total_tokens + current_request_tokens,
    }