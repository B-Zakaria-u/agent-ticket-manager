"""File management tools — wraps LangChain FileManagementToolkit (SRP: filesystem I/O only)."""
import os
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import tool
from typing import Optional

def _clean_path(target: str) -> str:
    """Strip hallucinated '/workspace/', './', or absolute prefixes."""
    target = target.strip().replace("\\", "/")
    prefixes = ["/workspace/", "workspace/", "./", "C:/", "c:/"]
    for p in prefixes:
        if target.startswith(p):
            target = target[len(p):]
    if target.startswith("/"):
        target = target[1:]
    return target

def get_file_tools(workspace_dir: str) -> list:
    """
    Return LangChain file-management tools rooted at *workspace_dir*.
    Wraps read_file and write_file to be more resilient to LLM parameter "hallucinations".
    """
    os.makedirs(workspace_dir, exist_ok=True)
    toolkit = FileManagementToolkit(root_dir=workspace_dir)
    original_tools = toolkit.get_tools()
    
    modified_tools = []
    for t in original_tools:
        if t.name == "read_file":
            @tool
            def read_file(
                file_path: Optional[str] = None,
                path: Optional[str] = None,
                _t=t,  # ← capture current t immediately
            ) -> str:
                """Read a file from the workspace. Use either 'file_path' or 'path'."""
                target = file_path or path
                if not target:
                    return "Error: No file path provided. Use 'file_path'."
                target = _clean_path(target)
                return _t.invoke({"file_path": target})
            modified_tools.append(read_file)
            
        elif t.name == "write_file":
            @tool
            def write_file(
                text: str,
                file_path: Optional[str] = None,
                path: Optional[str] = None,
                _t=t,  # ← capture current t immediately
            ) -> str:
                """Write content to a file in the workspace. Overwrites existing content.
                Use 'file_path' or 'path' for the filename."""
                target = file_path or path
                if not target:
                    return "Error: No file path provided. Use 'file_path'."
                target = _clean_path(target)
                return _t.invoke({"file_path": target, "text": text})
            modified_tools.append(write_file)
            
        else:
            modified_tools.append(t)
            
    return modified_tools