"""
src/utils/workspace_reader.py

Reads and ranks workspace files locally — no LLM round-trips required.
Called once inside build_context() to produce a ready-to-embed snapshot
of the most relevant source files for the current ticket.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directories that are never useful
_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".env", "target", "build", "dist", ".gradle", ".idea", ".vscode",
    "coverage", ".mypy_cache", ".pytest_cache", ".tox", "htmlcov",
})

# Files we always want (entry-points, config, manifests)
_HIGH_PRIORITY_NAMES: frozenset[str] = frozenset({
    "main.py", "app.py", "server.py", "index.py", "manage.py",
    "main.ts", "index.ts", "app.ts", "server.ts",
    "main.js", "index.js", "app.js",
    "Main.java", "Application.java",
    "main.go",
    "Cargo.toml", "pyproject.toml", "package.json", "pom.xml",
    "build.gradle", "build.gradle.kts", "go.mod",
    "requirements.txt", "setup.py", "setup.cfg",
    "Makefile", "Dockerfile",
    "README.md",
})

# Source-code extensions worth reading
_SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".ts", ".js", ".tsx", ".jsx",
    ".java", ".kt", ".kts",
    ".go", ".rs",
    ".rb", ".php",
    ".c", ".cpp", ".h", ".hpp",
    ".cs",
    ".swift",
    ".toml", ".yaml", ".yml", ".json", ".xml",
    ".md", ".txt", ".sh", ".env.example",
})

# Extensions we skip even if they're source-adjacent
_SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".lock", ".sum", ".log", ".map",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".jar", ".war",
    ".pyc", ".class", ".o", ".so", ".dll", ".exe",
    ".db", ".sqlite", ".sqlite3",
    ".min.js", ".min.css",
})

# Max bytes to read from a single file (avoids giant generated files)
_MAX_FILE_BYTES: int = 12_000

# Total character budget for all file content injected into the prompt
_DEFAULT_BUDGET_CHARS: int = 40_000


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FileSnippet:
    rel_path: str
    content:  str
    priority: int          # lower = more important
    size:     int          # original byte size


@dataclass
class WorkspaceSnapshot:
    files:          list[FileSnippet] = field(default_factory=list)
    total_files:    int = 0
    skipped_files:  int = 0
    budget_reached: bool = False

    def render(self) -> str:
        """
        Produce the text block that goes straight into the prompt.
        Format is easy for an LLM to parse and reference.
        """
        lines: list[str] = []
        for f in self.files:
            lines.append(f"### {f.rel_path}")
            lines.append("```")
            lines.append(f.content)
            lines.append("```")
            lines.append("")

        footer_parts = [f"{len(self.files)}/{self.total_files} files included"]
        if self.budget_reached:
            footer_parts.append("token budget reached — lower-priority files omitted")
        if self.skipped_files:
            footer_parts.append(f"{self.skipped_files} binary/oversized files skipped")

        lines.append(f"<!-- {', '.join(footer_parts)} -->")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Keyword-based relevance scoring
# ---------------------------------------------------------------------------

def _keyword_score(rel_path: str, ticket_keywords: Sequence[str]) -> int:
    """
    Return a small bonus (negative priority delta) when the file path
    contains words that appear in the ticket.  Pure string ops — no LLM.
    """
    path_lower = rel_path.lower()
    return -sum(1 for kw in ticket_keywords if kw and kw.lower() in path_lower)


def _extract_keywords(ticket_text: str) -> list[str]:
    """
    Pull meaningful tokens from the ticket (min 4 chars, alpha only).
    Keeps the top-20 most frequent so the scoring stays O(files * 20).
    """
    words = re.findall(r"[a-zA-Z]{4,}", ticket_text)
    freq: dict[str, int] = {}
    for w in words:
        freq[w.lower()] = freq.get(w.lower(), 0) + 1
    # Sort by frequency, take top 20
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:20]]


# ---------------------------------------------------------------------------
# Priority assignment
# ---------------------------------------------------------------------------

def _priority(rel_path: str, keyword_bonus: int) -> int:
    """
    Lower number = read first.
      0  high-priority names (entry-points, manifests)
      1  test files
      2  other source files
      3  config / data files
    Then subtract keyword_bonus so ticket-relevant files float up.
    """
    name = Path(rel_path).name
    ext  = "".join(Path(rel_path).suffixes).lower()  # handles .min.js etc.

    if name in _HIGH_PRIORITY_NAMES:
        base = 0
    elif "test" in rel_path.lower() or "spec" in rel_path.lower():
        base = 1
    elif ext in {".py", ".ts", ".js", ".tsx", ".jsx",
                 ".java", ".kt", ".go", ".rs", ".rb", ".php",
                 ".c", ".cpp", ".cs", ".swift"}:
        base = 2
    else:
        base = 3

    return base + keyword_bonus   # keyword_bonus is negative → floats up


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def read_workspace(
    workspace_dir: str,
    *,
    ticket_text:   str  = "",
    max_files:     int  = 60,
    budget_chars:  int  = _DEFAULT_BUDGET_CHARS,
) -> WorkspaceSnapshot:
    """
    Walk *workspace_dir* on local disk, rank every source file by relevance,
    read the most important ones up to *budget_chars*, and return a
    WorkspaceSnapshot ready to be embedded in the LLM prompt.

    This runs entirely on local hardware — no network, no LLM calls.
    """
    root = Path(workspace_dir)
    if not root.exists():
        return WorkspaceSnapshot()

    keywords = _extract_keywords(ticket_text)

    # ── 1. Discover all candidate files ──────────────────────────────────────
    candidates: list[tuple[int, str]] = []   # (priority, rel_path)
    total = 0
    skipped = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored dirs in-place (affects os.walk recursion)
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]

        for fname in filenames:
            full_path = Path(dirpath) / fname
            rel_path  = str(full_path.relative_to(root))

            # Skip by extension
            suffix = full_path.suffix.lower()
            if suffix in _SKIP_EXTENSIONS:
                skipped += 1
                continue
            if suffix not in _SOURCE_EXTENSIONS and fname not in _HIGH_PRIORITY_NAMES:
                skipped += 1
                continue

            total += 1
            kw_bonus = _keyword_score(rel_path, keywords)
            prio     = _priority(rel_path, kw_bonus)
            candidates.append((prio, rel_path))

    # ── 2. Sort by priority (ascending = most important first) ────────────────
    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:max_files]          # hard cap on candidate count

    # ── 3. Read files within budget ───────────────────────────────────────────
    snapshot       = WorkspaceSnapshot(total_files=total, skipped_files=skipped)
    chars_used     = 0

    for prio, rel_path in candidates:
        full_path = root / rel_path
        try:
            raw = full_path.read_bytes()
        except OSError:
            skipped += 1
            continue

        # Skip binary-looking files (heuristic: >30 % non-printable bytes)
        text_bytes = raw[:_MAX_FILE_BYTES]
        non_print  = sum(1 for b in text_bytes if b < 9 or (13 < b < 32))
        if len(text_bytes) and non_print / len(text_bytes) > 0.30:
            skipped += 1
            continue

        try:
            content = text_bytes.decode("utf-8", errors="replace")
        except Exception:
            skipped += 1
            continue

        # Truncate if the file itself is huge
        truncated = False
        if len(raw) > _MAX_FILE_BYTES:
            content   = content + f"\n... [{len(raw) - _MAX_FILE_BYTES} bytes truncated]"
            truncated = True

        # Check global character budget
        if chars_used + len(content) > budget_chars:
            snapshot.budget_reached = True
            break

        chars_used += len(content)
        snapshot.files.append(
            FileSnippet(
                rel_path=rel_path,
                content=content,
                priority=prio,
                size=len(raw),
            )
        )

    return snapshot
