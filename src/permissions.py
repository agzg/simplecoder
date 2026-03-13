"""
Task- and session-level permissions for file read/write and tool use.
User can grant/deny permissions per operation or set session defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    LIST = "list"
    SEARCH = "search"
    SHELL = "shell"


@dataclass
class PermissionState:
    """Session-level permission state with optional per-task overrides."""

    default: str = "ask"
    allowed_paths: set[str] = field(default_factory=set)
    denied_paths: set[str] = field(default_factory=set)
    task_grants: dict[tuple[Permission, str], bool] = field(default_factory=dict)
    ask_callback: Callable[[Permission, str, str], bool] | None = None

    def normalize_path(self, path: str) -> str:
        p = Path(path).resolve()
        try:
            return str(p)
        except OSError:
            return path

    def check(self, permission: Permission, path: str, detail: str = "") -> bool:
        """Check if permission is granted."""
        if permission == Permission.SHELL:
            norm = path  # command string, do not normalize
        else:
            norm = self.normalize_path(path)
        key = (permission, norm)
        if key in self.task_grants:
            return self.task_grants[key]
        # SHELL: always ask user (ignore default allow)
        if permission == Permission.SHELL:
            if self.ask_callback is not None:
                granted = self.ask_callback(permission, norm, detail)
                self.task_grants[key] = granted
                return granted
            return False
        for allowed in self.allowed_paths:
            if norm == allowed or norm.startswith(allowed.rstrip("/") + "/"):
                return True
        for denied in self.denied_paths:
            if norm == denied or norm.startswith(denied.rstrip("/") + "/"):
                return False
        if self.default == "allow":
            return True
        if self.default == "deny":
            return False
        if self.ask_callback is not None:
            granted = self.ask_callback(permission, norm, detail)
            self.task_grants[key] = granted
            return granted
        return False

    def grant(self, permission: Permission, path: str) -> None:
        norm = self.normalize_path(path)
        self.task_grants[(permission, norm)] = True

    def deny(self, permission: Permission, path: str) -> None:
        norm = self.normalize_path(path)
        self.task_grants[(permission, norm)] = False

    def set_default(self, policy: str) -> None:
        if policy not in ("allow", "ask", "deny"):
            raise ValueError("policy must be allow, ask, or deny")
        self.default = policy
