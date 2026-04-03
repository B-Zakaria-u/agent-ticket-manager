"""PR Agent — SRP: commit, push, and open a GitHub Pull Request.

Replaces the old GitLab-based PR agent with GitHub tooling.
"""
from src.agents.base import BaseAgentNode
from src.config.llm import get_llm
from src.state import GraphState
from src.tools.github.git_tools import commit_and_push
from src.tools.github.pr_tools import create_pull_request
from langchain_core.messages import SystemMessage, HumanMessage


from src.utils.logger import log_llm_interaction, log_chat_interaction


class PRAgent(BaseAgentNode):
    """Commits and pushes the fix, then opens a GitHub Pull Request."""

    def run(self, state: GraphState) -> dict:
        llm = get_llm()
        ticket_text = state.get("ticket_text", "")
        issue_number = state.get("issue_number", 0)
        branch_name = state.get("branch_name", "fix/automated")
        repo_url = state.get("repo_url", "")
        log_file_path = state.get("log_file_path", "")
        chat_log_file_path = state.get("chat_log_file_path", "")
        total_tokens = state.get("total_tokens", 0)

        all_tools = [commit_and_push, create_pull_request]
        llm_with_tools = llm.bind_tools(all_tools)

        # ── LLM drafts commit message and PR description ─────────────────────
        draft_messages = [
            SystemMessage(content=(
                "You are a CI/CD agent. Produce a concise git commit message and "
                "a short GitHub PR description for the fix described below. "
                "Format your response as:\n"
                "COMMIT: <one-line commit message>\n"
                "PR_BODY: <short markdown description, include 'Closes #<N>'>"
            )),
            HumanMessage(content=ticket_text),
        ]

        if chat_log_file_path:
            log_chat_interaction(chat_log_file_path, "PR Agent", draft_messages)
        
        print("[ PR Agent ] Drafting commit message and Pull Request body...")
        response = llm.invoke(draft_messages)

        # Extract token usage
        usage = response.usage_metadata or {}
        p_tokens = usage.get("input_tokens", 0)
        c_tokens = usage.get("output_tokens", 0)

        if log_file_path:
            model = getattr(llm, "model", getattr(llm, "model_name", "unknown-model"))
            log_llm_interaction(log_file_path, "PR Agent", model, p_tokens, c_tokens)

        raw = response.content
        if isinstance(raw, list):
            raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
        draft = str(raw).strip()

        # Parse COMMIT and PR_BODY from the LLM output
        commit_msg = "fix: automated patch via AI Dev System"
        pr_body = f"Automated fix.\n\nCloses #{issue_number}"

        for line in draft.splitlines():
            if line.startswith("COMMIT:"):
                commit_msg = line.replace("COMMIT:", "").strip()
            elif line.startswith("PR_BODY:"):
                pr_body = line.replace("PR_BODY:", "").strip()

        # ── Push the branch ───────────────────────────────────────────────────
        print(f"[ PR Agent ] Committing and pushing to branch: {branch_name} ...")
        push_result = commit_and_push.invoke({
            "commit_message": commit_msg,
            "branch_name": branch_name,
            "repo_url": repo_url,
            "force": True,  # Overwrite remote branch for this specific fix
        })

        if "FAILURE" in push_result:
            print(f"[ PR Agent ] Push failed: {push_result}")
            return {
                "pr_url": f"Push failed: {push_result}",
                "total_tokens": total_tokens + p_tokens + c_tokens
            }

        # ── Open the PR ───────────────────────────────────────────────────────
        pr_title = f"fix: {ticket_text.splitlines()[0][:72]}"
        print(f"[ PR Agent ] Opening GitHub Pull Request: '{pr_title}' ...")
        pr_result = create_pull_request.invoke({
            "branch_name": branch_name,
            "title": pr_title,
            "body": pr_body,
        })

        # Extract URL from result string
        pr_url = ""
        if "http" in pr_result:
            urls = [w for w in pr_result.split() if "http" in w]
            for url in urls:
                # Basic cleaning of URL in case it's in trailing punctuation/brackets
                clean_url = url.strip("()[]{},;\"'")
                if clean_url.startswith("http"):
                    pr_url = clean_url
                    break
        
        if not pr_url:
            # If no URL, return the error message from tools
            pr_url = pr_result

        return {
            "pr_url": pr_url,
            "total_tokens": total_tokens + p_tokens + c_tokens
        }


# LangGraph node callable
pr_agent_node = PRAgent()
