"""Language and framework detector — language-agnostic pipeline support.

Walks the workspace directory and infers the dominant programming language
and framework from file extensions + well-known config file markers.

Returns
-------
dict with:
  - ``language``  : str — e.g. "Python", "Java", "JavaScript", "PHP", "Kotlin"
  - ``framework`` : str — e.g. "Django", "Spring Boot", "Laravel", "Express", or "Unknown"
"""
import json
import os
from collections import Counter
from typing import Any


# ── Extension → Language mapping ────────────────────────────────────────────
_EXT_TO_LANG: dict[str, str] = {
    ".py":    "Python",
    ".java":  "Java",
    ".kt":    "Kotlin",
    ".kts":   "Kotlin",
    ".php":   "PHP",
    ".js":    "JavaScript",
    ".jsx":   "JavaScript",
    ".ts":    "TypeScript",
    ".tsx":   "TypeScript",
    ".cs":    "C#",
    ".go":    "Go",
    ".rb":    "Ruby",
    ".rs":    "Rust",
    ".cpp":   "C++",
    ".cc":    "C++",
    ".c":     "C",
    ".swift": "Swift",
    ".scala": "Scala",
}

# ── Directories / files to ignore while scanning ────────────────────────────
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "target", "build", "dist", ".gradle", ".idea", ".mvn",
}


def _count_extensions(workspace_path: str) -> Any:
    """Walk the workspace and count source-file extensions."""
    counts: Counter = Counter()
    for root, dirs, files in os.walk(workspace_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in _EXT_TO_LANG:
                counts[ext] += 1
    return counts


def _detect_python_framework(workspace_path: str) -> str:
    """Inspect requirements.txt / pyproject.toml to detect Python framework."""
    candidates = [
        os.path.join(workspace_path, "requirements.txt"),
        os.path.join(workspace_path, "pyproject.toml"),
        os.path.join(workspace_path, "setup.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                content = open(path, encoding="utf-8", errors="ignore").read().lower()
                if "fastapi" in content:
                    return "FastAPI"
                if "django" in content:
                    return "Django"
                if "flask" in content:
                    return "Flask"
                if "tornado" in content:
                    return "Tornado"
                if "aiohttp" in content:
                    return "aiohttp"
            except OSError:
                pass
    return "Unknown"


def _detect_java_framework(workspace_path: str) -> str:
    """Inspect pom.xml or build.gradle to detect Java/Kotlin framework."""
    for fname in ("pom.xml", "build.gradle", "build.gradle.kts"):
        path = os.path.join(workspace_path, fname)
        if os.path.exists(path):
            try:
                content = open(path, encoding="utf-8", errors="ignore").read().lower()
                if "spring-boot" in content:
                    return "Spring Boot"
                if "springframework" in content or "spring-context" in content:
                    return "Spring"
                if "jakarta" in content or "javax.ejb" in content or "wildfly" in content:
                    return "JEE"
                if "ktor" in content:
                    return "Ktor"
                if "quarkus" in content:
                    return "Quarkus"
                if "micronaut" in content:
                    return "Micronaut"
            except OSError:
                pass
    return "Unknown"


def _detect_php_framework(workspace_path: str) -> str:
    """Inspect composer.json to detect PHP framework."""
    path = os.path.join(workspace_path, "composer.json")
    if os.path.exists(path):
        try:
            data = json.loads(open(path, encoding="utf-8", errors="ignore").read())
            require = {**data.get("require", {}), **data.get("require-dev", {})}
            keys = " ".join(require.keys()).lower()
            if "laravel" in keys:
                return "Laravel"
            if "symfony" in keys:
                return "Symfony"
            if "codeigniter" in keys:
                return "CodeIgniter"
            if "slim" in keys:
                return "Slim"
        except (OSError, json.JSONDecodeError):
            pass
    return "Unknown"


def _detect_js_framework(workspace_path: str) -> str:
    """Inspect package.json to detect JS/TS framework."""
    path = os.path.join(workspace_path, "package.json")
    if os.path.exists(path):
        try:
            data = json.loads(open(path, encoding="utf-8", errors="ignore").read())
            deps_str = " ".join(
                list(data.get("dependencies", {}).keys())
                + list(data.get("devDependencies", {}).keys())
            ).lower()
            if "next" in deps_str:
                return "Next.js"
            if "react" in deps_str:
                return "React"
            if "vue" in deps_str:
                return "Vue.js"
            if "angular" in deps_str:
                return "Angular"
            if "express" in deps_str:
                return "Express"
            if "nestjs" in deps_str or "@nestjs" in deps_str:
                return "NestJS"
            if "nuxt" in deps_str:
                return "Nuxt.js"
        except (OSError, json.JSONDecodeError):
            pass
    return "Unknown"


def _detect_go_framework(workspace_path: str) -> str:
    """Inspect go.mod to detect Go framework."""
    path = os.path.join(workspace_path, "go.mod")
    if os.path.exists(path):
        try:
            content = open(path, encoding="utf-8", errors="ignore").read().lower()
            if "gin-gonic" in content:
                return "Gin"
            if "labstack/echo" in content:
                return "Echo"
            if "gofiber" in content:
                return "Fiber"
            if "beego" in content:
                return "Beego"
        except OSError:
            pass
    return "Unknown"


def _detect_ruby_framework(workspace_path: str) -> str:
    """Inspect Gemfile to detect Ruby framework."""
    path = os.path.join(workspace_path, "Gemfile")
    if os.path.exists(path):
        try:
            content = open(path, encoding="utf-8", errors="ignore").read().lower()
            if "rails" in content:
                return "Ruby on Rails"
            if "sinatra" in content:
                return "Sinatra"
            if "hanami" in content:
                return "Hanami"
        except OSError:
            pass
    return "Unknown"


# ── Framework detectors keyed by language ───────────────────────────────────
_FRAMEWORK_DETECTORS = {
    "Python":     _detect_python_framework,
    "Java":       _detect_java_framework,
    "Kotlin":     _detect_java_framework,    # shares same build files
    "PHP":        _detect_php_framework,
    "JavaScript": _detect_js_framework,
    "TypeScript": _detect_js_framework,
    "Go":         _detect_go_framework,
    "Ruby":       _detect_ruby_framework,
}


def detect_language(workspace_path: str, hint: str | None = None) -> dict[str, str]:
    """Detect the dominant programming language and framework in *workspace_path*.

    If *hint* is provided (e.g. the user's spec), the detector will also
    check whether the hint implies a different language/framework than what's
    on disk, and return the most likely combination.

    Args:
        workspace_path: Absolute or relative path to the workspace root.
        hint: Optional text (e.g. spec) to guide detection.

    Returns:
        A dict: ``{"language": "<Language>", "framework": "<Framework>"}``
        Both values fall back to ``"Unknown"`` if detection is inconclusive.
    """
    workspace_path = os.path.abspath(workspace_path)
    if not os.path.isdir(workspace_path):
        return {"language": "Unknown", "framework": "Unknown"}

    ext_counts = _count_extensions(workspace_path)

    if not ext_counts:
        return {"language": "Unknown", "framework": "Unknown"}

    # Pick the most frequent extension
    dominant_ext, _ = ext_counts.most_common(1)[0]
    language = _EXT_TO_LANG.get(dominant_ext, "Unknown")

    # Detect framework
    detector = _FRAMEWORK_DETECTORS.get(language)
    framework = detector(workspace_path) if detector else "Unknown"

    # If a hint is provided, check if it implies a different stack
    if hint:
        hint_lower = hint.lower()

        # Check for framework keywords in hint
        for lang, detector in _FRAMEWORK_DETECTORS.items():
            if lang.lower() in hint_lower:
                hint_framework = detector(workspace_path)
                if hint_framework != "Unknown":
                    # If hint clearly points to a different framework, prefer it
                    if framework == "Unknown" or hint_framework != framework:
                        language = lang
                        framework = hint_framework
                        break

    return {"language": language, "framework": framework}
