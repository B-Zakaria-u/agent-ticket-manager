"""Docker sandbox tool — ephemeral container execution for tests/scripts.

Security perimeter
------------------
* Only the ``workspace/`` directory is mounted (read-write).
* The container is always removed after execution (``remove=True``).
* A configurable timeout prevents indefinite hangs (default 120 s).

The tool pre-pulls the requested image, normalises Windows paths for
Docker Desktop, converts CRLF → LF in shell scripts, and returns
structured output with ``[SANDBOX OK]`` / ``[SANDBOX FAIL]`` markers
so the calling agent can reliably parse the result.
"""

import os
import platform

import docker
from docker.errors import ContainerError, ImageNotFound, APIError
from langchain_core.tools import tool

# ── Constants ────────────────────────────────────────────────────────────────
_DEFAULT_TIMEOUT = 120          # seconds
_MARKER_OK   = "[SANDBOX OK]"
_MARKER_FAIL = "[SANDBOX FAIL]"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalise_path_for_docker(path: str) -> str:
    """Convert a Windows absolute path to Docker-Desktop-compatible format.

    Docker Desktop on Windows accepts paths like ``C:/Users/...`` or
    ``//c/Users/...``.  Python's ``os.path.abspath`` returns backslash
    paths (``C:\\Users\\...``) which can cause mount failures.
    """
    if platform.system() == "Windows":
        # C:\\Users\\foo → C:/Users/foo
        path = path.replace("\\", "/")
    return path


def _fix_script_line_endings(workspace: str) -> None:
    """Convert CRLF → LF for every ``.sh`` file in *workspace* root."""
    for entry in os.listdir(workspace):
        if entry.endswith(".sh"):
            full = os.path.join(workspace, entry)
            try:
                raw = open(full, "rb").read()
                if b"\r\n" in raw:
                    print(f"  [Sandbox] Converting {entry} line endings to LF")
                    with open(full, "wb") as f:
                        f.write(raw.replace(b"\r\n", b"\n"))
            except OSError as exc:
                print(f"  [Sandbox] Warning: could not fix {entry}: {exc}")


def _ensure_image(client: docker.DockerClient, image: str) -> None:
    """Pull *image* if it is not already available locally."""
    try:
        client.images.get(image)
        print(f"  [Sandbox] Image '{image}' found locally")
    except ImageNotFound:
        print(f"  [Sandbox] Pulling image '{image}' (this may take a moment)…")
        try:
            client.images.pull(image)
            print(f"  [Sandbox] Image '{image}' pulled successfully")
        except APIError as exc:
            raise RuntimeError(
                f"Failed to pull Docker image '{image}': {exc}"
            ) from exc


# ── Main tool ────────────────────────────────────────────────────────────────

@tool
def run_tests_in_sandbox(
    workspace_path: str,
    image_name: str = "ubuntu:22.04",
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """Run the repository's ``script.sh`` inside an ephemeral Docker container.

    The *workspace_path* directory is mounted as ``/workspace`` inside the
    container.  Returns structured output prefixed with ``[SANDBOX OK]`` on
    success or ``[SANDBOX FAIL]`` on failure so the calling agent can parse
    the result deterministically.

    Args:
        workspace_path: Absolute path to the local workspace directory.
        image_name: Docker image to use (default: ubuntu:22.04).
        timeout: Max seconds the container may run (default: 120).
    """

    # ── 1. Resolve & validate workspace ──────────────────────────────────
    abs_workspace = os.path.abspath(workspace_path)
    script_path   = os.path.join(abs_workspace, "script.sh")

    if not os.path.isdir(abs_workspace):
        return (
            f"{_MARKER_FAIL}\n"
            f"Workspace directory does not exist: {abs_workspace}\n"
            "Hint: make sure the workspace path is correct."
        )

    if not os.path.isfile(script_path):
        return (
            f"{_MARKER_FAIL}\n"
            f"No script.sh found in workspace: {abs_workspace}\n"
            "Hint: create a script.sh at the workspace root before running tests."
        )

    # ── 2. Fix line endings ──────────────────────────────────────────────
    _fix_script_line_endings(abs_workspace)

    # ── 3. Connect to Docker ─────────────────────────────────────────────
    try:
        # Check connectivity with a short timeout
        docker.from_env(timeout=10).ping()
        # Create the actual client with default timeout for long-running tasks
        client = docker.from_env()
    except Exception as exc:
        return (
            f"{_MARKER_FAIL}\n"
            f"Cannot connect to Docker daemon: {exc}\n"
            "Hint: ensure Docker Desktop is running and the Docker socket is accessible."
        )

    # ── 4. Ensure the image is available ─────────────────────────────────
    try:
        _ensure_image(client, image_name)
    except RuntimeError as exc:
        return f"{_MARKER_FAIL}\n{exc}"

    # ── 5. Prepare volume mount (Windows-safe) ───────────────────────────
    docker_path = _normalise_path_for_docker(abs_workspace)
    print(f"  [Sandbox] Mounting {docker_path} → /workspace")

    # ── 6. Run the container ─────────────────────────────────────────────
    command = 'sh -c "chmod +x /workspace/script.sh && /workspace/script.sh"'

    try:
        container = client.containers.run(
            image=image_name,
            command=command,
            volumes={docker_path: {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            detach=True,
        )

        import requests
        try:
            result = container.wait(timeout=timeout)
            exit_code = result.get("StatusCode", -1)

            # Fetch output
            stdout_bytes = container.logs(stdout=True, stderr=False)
            stderr_bytes = container.logs(stdout=False, stderr=True)
            output_bytes = stdout_bytes + stderr_bytes
            decoded = output_bytes.decode("utf-8", errors="replace")

            if exit_code == 0:
                print(f"  [Sandbox] Container finished successfully")
                return f"{_MARKER_OK}\n{decoded}"
            else:
                print(f"  [Sandbox] Container exited with status {exit_code}")
                return (
                    f"{_MARKER_FAIL}\n"
                    f"Container exited with status {exit_code}.\n"
                    f"Output:\n{decoded}"
                )
        except requests.exceptions.ReadTimeout:
            print(f"  [Sandbox] Execution timed out after {timeout} seconds")
            return (
                f"{_MARKER_FAIL}\n"
                f"Docker execution error: Execution timed out after {timeout} seconds\n"
                "Hint: check if the script runs indefinitely or requires user input."
            )
        finally:
            try:
                container.stop(timeout=1)
                container.remove(v=True, force=True)
            except Exception:
                pass

    except Exception as exc:
        # Docker system-level failure (timeout, API error, etc.)
        error_msg = str(exc)
        print(f"  [Sandbox] Docker execution error: {error_msg}")
        return (
            f"{_MARKER_FAIL}\n"
            f"Docker execution error: {error_msg}\n"
            "Hint: check that Docker Desktop is running and has enough resources."
        )
