"""
Tool functions and OpenAI-style schemas for file and codebase operations.
Tools: list, read, search, write, edit source files, use_llm for research/delegation.
use_shell (dangerous mode only) for running shell commands.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from simplecoder.permissions import Permission, PermissionState
from simplecoder.safe import SafetyError, sanitize_output, validate_content, validate_shell_command

# Tool schemas

USE_SHELL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "use_shell",
        "description": "Run a shell command. Optional stdin for programs that need input. Use for ls, cat, python script.py, etc. For interactive programs: pass stdin with inputs separated by newlines; run, observe output, then run again with next inputs if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run (e.g. python script.py)"},
                "stdin": {"type": "string", "description": "Optional input to feed to the program via stdin. Use newlines to separate multiple inputs. E.g. 'yes\\nno' for two lines."},
            },
            "required": ["command"],
        },
    },
}


def get_tool_schemas(dangerous: bool = False) -> list[dict[str, Any]]:
    """Return tool schemas. Include use_shell only when dangerous=True."""
    schemas = list(TOOL_SCHEMAS)
    if dangerous:
        schemas = schemas + [USE_SHELL_SCHEMA]
    return schemas


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at the given path. Use to explore project structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list (default: current directory)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file. Use for source code or config files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for text or regex pattern in files under a directory. Optionally restrict by file pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text or regex pattern to search for"},
                    "path": {"type": "string", "description": "Root directory to search (default: current)"},
                    "glob": {"type": "string", "description": "File pattern e.g. '*.py' (optional)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content. Use for new files or full replacement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to write"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing a single occurrence of old_string with new_string. Preserves rest of file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to edit"},
                    "old_string": {"type": "string", "description": "Exact string to find (must match exactly)"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "use_llm",
            "description": "Use LLM to research, delegate subtasks, understand code/config, find documentation, answer questions. Pass a clear query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question or task for the LLM (e.g. explain this config, find docs for X, break down subtask)"},
                },
                "required": ["query"],
            },
        },
    },
]

# Helper functions


def _resolve_path(path: str | None, base: Path) -> Path:
    if not path or path == ".":
        return base
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def list_dir(path: str | None, base: Path, permissions: PermissionState) -> str:
    """List directory contents. Returns names and types."""
    root = _resolve_path(path, base)
    if not permissions.check(Permission.LIST, str(root), "list directory"):
        return f"[Permission denied] Cannot list: {root}"
    try:
        if not root.exists():
            return f"[Error] Path does not exist: {root}"
        if not root.is_dir():
            return f"[Error] Not a directory: {root}"
        entries = []
        for e in sorted(root.iterdir()):
            kind = "dir" if e.is_dir() else "file"
            entries.append(f"  {e.name} ({kind})")
        return "\n".join(entries) if entries else "(empty)"
    except OSError as err:
        return f"[Error] {err}"


def read_file(path: str, base: Path, permissions: PermissionState) -> str:
    """Read file contents."""
    root = _resolve_path(path, base)
    if not permissions.check(Permission.READ, str(root), "read file"):
        return f"[Permission denied] Cannot read: {root}"
    try:
        if not root.exists():
            return f"[Error] File not found: {root}"
        if not root.is_file():
            return f"[Error] Not a file: {root}"
        return root.read_text(encoding="utf-8", errors="replace")
    except OSError as err:
        return f"[Error] {err}"


def search_files(
    query: str,
    path: str | None,
    glob: str | None,
    base: Path,
    permissions: PermissionState,
) -> str:
    """Search for query (text or regex) in files under path."""
    root = _resolve_path(path, base)
    if not permissions.check(Permission.SEARCH, str(root), "search in directory"):
        return f"[Permission denied] Cannot search under: {root}"
    try:
        if not root.exists() or not root.is_dir():
            return f"[Error] Not a directory: {root}"
        try:
            pattern = re.compile(query)
        except re.error:
            pattern = re.compile(re.escape(query))
        glob_pattern = glob or "*"
        results: list[str] = []
        for f in root.rglob(glob_pattern):
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        rel = f.relative_to(root)
                        results.append(f"{rel}:{i}: {line.strip()}")
            except OSError:
                continue
        if not results:
            return f"No matches for '{query}' under {root}"
        return "\n".join(results[:100])
    except re.error as err:
        return f"[Error] Invalid regex: {err}"
    except OSError as err:
        return f"[Error] {err}"


def write_file(path: str, content: str, base: Path, permissions: PermissionState) -> str:
    """Write full file content (create or overwrite). Validates content for unsafe code."""
    try:
        validate_content(content)
    except SafetyError as e:
        return f"[Blocked] {e}"
    root = _resolve_path(path, base)
    if not permissions.check(Permission.WRITE, str(root), "write file"):
        return f"[Permission denied] Cannot write: {root}"
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.write_text(content, encoding="utf-8")
        return f"Wrote {root} ({len(content)} chars)"
    except OSError as err:
        return f"[Error] {err}"


def edit_file(path: str, old_string: str, new_string: str, base: Path, permissions: PermissionState) -> str:
    """Replace first occurrence of old_string with new_string in file."""
    root = _resolve_path(path, base)
    if not permissions.check(Permission.EDIT, str(root), "edit file"):
        return f"[Permission denied] Cannot edit: {root}"
    try:
        if not root.exists():
            return f"[Error] File not found: {root}"
        text = root.read_text(encoding="utf-8", errors="replace")
        if old_string not in text:
            return f"[Error] old_string not found in file. Ensure it matches exactly (including spaces/newlines)."
        new_text = text.replace(old_string, new_string, 1)
        root.write_text(new_text, encoding="utf-8")
        return f"Edited {root} (1 replacement)"
    except OSError as err:
        return f"[Error] {err}"


def use_llm(query: str, llm_fn: Callable[[str], str] | None) -> str:
    """Use LLM to answer query. Requires llm_fn to be passed from agent."""
    if not llm_fn:
        return "[Error] use_llm requires LLM; not available in this context."
    try:
        return llm_fn(query)
    except Exception as e:
        return f"[Error] {e}"


def use_shell(command: str, base: Path, permissions: PermissionState, stdin: str | None = None) -> str:
    """Run shell command. Optional stdin for programs that need input. Asks user first, validates, sanitizes output."""
    if not permissions.check(Permission.SHELL, command, f"run shell: {command}"):
        return "[Permission denied] User declined to run shell command."
    try:
        validate_shell_command(command)
    except SafetyError as e:
        return f"[Blocked] {e}"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=base,
            capture_output=True,
            text=True,
            timeout=60,
            input=stdin if stdin else None,
        )
        out = result.stdout or ""
        err = result.stderr or ""
        combined = (out + "\n" + err).strip() if err else out.strip()
        return sanitize_output(combined)[:8000] or f"(exit {result.returncode})"
    except subprocess.TimeoutExpired:
        return "[Error] Command timed out (60s)"
    except Exception as e:
        return f"[Error] {e}"


def run_tool(
    name: str,
    arguments: dict[str, Any],
    base: Path,
    permissions: PermissionState,
    llm_fn: Callable[[str], str] | None = None,
) -> str:
    """Dispatch to the right tool and return result string."""
    base = base.resolve()
    if name == "use_shell":
        return use_shell(
            arguments.get("command", ""),
            base,
            permissions,
            stdin=arguments.get("stdin"),
        )
    if name == "use_llm":
        return use_llm(arguments.get("query", ""), llm_fn)
    if name == "list_dir":
        return list_dir(arguments.get("path"), base, permissions)
    if name == "read_file":
        return read_file(arguments["path"], base, permissions)
    if name == "search_files":
        return search_files(
            arguments["query"],
            arguments.get("path"),
            arguments.get("glob"),
            base,
            permissions,
        )
    if name == "write_file":
        return write_file(arguments["path"], arguments["content"], base, permissions)
    if name == "edit_file":
        return edit_file(
            arguments["path"],
            arguments["old_string"],
            arguments["new_string"],
            base,
            permissions,
        )
    return f"[Error] Unknown tool: {name}"
