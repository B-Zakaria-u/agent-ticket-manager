"""Legacy top-level docker sandbox — delegates to the sub-package version.

This file exists for backward compatibility. The canonical implementation
lives in ``src.tools.docker.sandbox``.
"""
import docker
import os
from langchain_core.tools import tool

@tool
def run_tests_in_sandbox(workspace_path: str, image_name: str = "ubuntu:22.04") -> str:
    """
    Executes tests inside a sandboxed Docker container securely.
    Mounts the workspace directory and runs the repository's script.sh.
    Language-agnostic: script.sh handles all runtime installation and test commands.

    Args:
        workspace_path (str): The absolute path to the local git workspace to test.
        image_name (str): Docker image to use (default: ubuntu:22.04)
    """
    client = docker.from_env()
    abs_workspace = os.path.abspath(workspace_path)

    # ── Fix Line Endings ───────────────────────────────────────────────────
    script_path = os.path.join(abs_workspace, "script.sh")
    if os.path.exists(script_path):
        try:
            with open(script_path, "rb") as f:
                content = f.read()
            if b"\r\n" in content:
                with open(script_path, "wb") as f:
                    f.write(content.replace(b"\r\n", b"\n"))
        except Exception:
            pass

    try:
        container = client.containers.run(
            image=image_name,
            command='sh -c "chmod +x /workspace/script.sh && /workspace/script.sh"',
            volumes={abs_workspace: {'bind': '/workspace', 'mode': 'rw'}},
            working_dir="/workspace",
            detach=False,
            remove=True
        )
        return container.decode('utf-8')
    except docker.errors.ContainerError as e:
        stderr_out = e.stderr.decode('utf-8') if e.stderr else str(e)
        return f"Tests failed with error in image '{image_name}':\n{stderr_out}"
    except Exception as e:
        return f"Failed to execute sandbox run: {str(e)}"
