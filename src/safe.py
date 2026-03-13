"""
Safety validation: block malicious shell commands, code injection, and unsafe content.
Never process or output exec, eval, or dangerous shell patterns.
"""
from __future__ import annotations

import re


class SafetyError(Exception):
    """Raised when content fails safety validation."""

    pass


# Dangerous shell patterns (blocked)
_SHELL_BLOCKED = (
    r"\brm\s+-rf\s+/",  # rm -rf /
    r"\brm\s+-rf\s+\*",
    r"\bchmod\s+-R\s+777",
    r"\bchown\s+-R",
    r"\bmkfs\.",  # mkfs.ext4, etc.
    r"\bdd\s+if=.*of=/dev/",
    r"\b:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",  # fork bomb
    r"\bwget\s+.*\|\s*sh\b",
    r"\bcurl\s+.*\|\s*sh\b",
    r"\bcurl\s+.*\|\s*bash\b",
    r">\s*/dev/sd[a-z]",
    r">\s*/dev/nvme",
    r"\bformat\s+c:",
    r"\bdel\s+/[sf]\s+",
    r"\bformat\s+/",
    r"\$\(.*\)",  # command substitution (can hide malicious code)
    r"`[^`]+`",  # backtick command substitution
    r"\|\s*bash\s*$",
    r"\|\s*sh\s*$",
    r"\|\s*zsh\s*$",
    r"^\s*sudo\s+",
    r"^\s*su\s+",
    r"\bexec\s+",
    r"\beval\s+",
    r"\bpython\s+-c\s+.*exec\b",
    r"\bpython3\s+-c\s+.*exec\b",
    r"\bruby\s+-e\s+.*exec\b",
    r"\bperl\s+-e\s+.*exec\b",
    r"\bnode\s+-e\s+.*eval\b",
    r"\bbase64\s+-d\s*\|",
    r"\bopenssl\s+.*-d\s*\|",
    r";\s*rm\s+",
    r"&&\s*rm\s+-rf",
    r"\|\s*rm\s+",
    r"\bcat\s+.*\|\s*bash\b",
    r"\bcat\s+.*\|\s*sh\b",
    r"\bnc\s+-",
    r"\bncat\s+-",
    r"\bnetcat\s+-",
    r"/etc/passwd",
    r"/etc/shadow",
    r"\.\./\.\./",  # path traversal attempts
)
_SHELL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _SHELL_BLOCKED]

# Dangerous code patterns (for file content validation in write_file)
# Blocks code injection: exec, eval, __import__, compile, sensitive paths
_CODE_BLOCKED = (
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\b__import__\s*\(",
    r"\bcompile\s*\(",
    r"\bgetattr\s*\(\s*__builtins__",
    r"open\s*\(\s*[\"']/etc/(passwd|shadow)",
    r"open\s*\(\s*[\"']/dev/(sd|nvme)",
)
_CODE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _CODE_BLOCKED]

# Output sanitization: strip potential injection
_OUTPUT_DANGEROUS = (
    r"<\s*script",
    r"javascript:",
    r"on\w+\s*=",
    r"data:\s*text/html",
    r"\x00",  # null byte
)


def validate_shell_command(cmd: str) -> None:
    """Validate shell command. Raises SafetyError if dangerous."""
    if not cmd or not isinstance(cmd, str):
        raise SafetyError("Empty or invalid command")
    if len(cmd) > 4096:
        raise SafetyError("Command too long")
    normalized = " ".join(cmd.split())
    for pat in _SHELL_PATTERNS:
        if pat.search(normalized):
            raise SafetyError(f"Blocked: unsafe shell pattern")
    # Block newlines (multi-command injection)
    if "\n" in cmd or "\r" in cmd:
        raise SafetyError("Blocked: newlines in command")
    # Block semicolon chaining, but ignore semicolons inside quoted strings
    if ";" in cmd and not cmd.strip().startswith("cd "):
        in_single = False
        in_double = False
        unquoted_semi = False
        for ch in cmd:
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == ";" and not in_single and not in_double:
                unquoted_semi = True
                break
        if unquoted_semi:
            raise SafetyError("Blocked: command chaining")


def validate_content(content: str) -> None:
    """Validate file content. Raises SafetyError if contains dangerous code."""
    if not isinstance(content, str):
        raise SafetyError("Invalid content type")
    if len(content) > 2_000_000:
        raise SafetyError("Content too large")
    for pat in _CODE_PATTERNS:
        if pat.search(content):
            raise SafetyError("Blocked: unsafe code pattern")


def sanitize_output(text: str) -> str:
    """Sanitize output to prevent injection. Returns safe string."""
    if not isinstance(text, str):
        return str(text)
    result = text
    for pat in _OUTPUT_DANGEROUS:
        result = re.sub(pat, "[redacted]", result, flags=re.IGNORECASE)
    result = result.replace("\x00", "")
    return result
