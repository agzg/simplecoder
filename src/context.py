"""
Context management and compacting: track token usage, loop history, reflections,
and summarize when exceeding a limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simplecoder.planner import Plan
from simplecoder.reflect import LoopRecord, Reflection

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class ContextManager:
    """
    Tracks messages, loop records, reflections, and total token estimate.
    When over max_tokens, summarizes older content and keeps the last k intact.
    """

    max_tokens: int = 32000
    keep_last_n: int = 10
    _current_tokens: int = 0
    _messages: list[dict[str, Any]] = field(default_factory=list)
    _loop_records: list[LoopRecord] = field(default_factory=list)
    _reflections: list[Reflection] = field(default_factory=list)

    def add_message(self, role: str, content: str | list[Any]) -> None:
        if isinstance(content, list):
            text = " ".join(
                (c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
            )
        else:
            text = content
        self._messages.append({"role": role, "content": content})
        self._current_tokens += estimate_tokens(text)

    def add_loop_record(self, record: LoopRecord) -> None:
        self._loop_records.append(record)
        self._current_tokens += estimate_tokens(record.to_compressed())

    def add_reflection(self, reflection: Reflection) -> None:
        self._reflections.append(reflection)
        self._current_tokens += estimate_tokens(reflection.to_context())

    def get_messages(self) -> list[dict[str, Any]]:
        return self._messages

    def get_loop_records(self) -> list[LoopRecord]:
        return self._loop_records

    def get_reflections(self) -> list[Reflection]:
        return self._reflections

    def reset(self) -> None:
        self._messages = []
        self._loop_records = []
        self._reflections = []
        self._current_tokens = 0

    def reset_messages(self) -> None:
        """Reset only messages and recount tokens from loop records and reflections."""
        self._messages = []
        self._current_tokens = 0
        for r in self._loop_records:
            self._current_tokens += estimate_tokens(r.to_compressed())
        for ref in self._reflections:
            self._current_tokens += estimate_tokens(ref.to_context())

    def token_estimate(self) -> int:
        return self._current_tokens

    def build_loop_reflection_context(
        self,
        max_loop_chars: int = 4000,
        max_reflection_chars: int = 2000,
        plan: Plan | None = None,
    ) -> str:
        """Build compressed context from loop records, reflections, and optional plan for the agent."""
        parts = []
        if plan and not plan.all_done():
            parts.append("[Subtask status - pick up from next pending]\n" + plan.summary())
        if self._reflections:
            reflection_text = "\n\n".join(r.to_context() for r in self._reflections[-5:])
            if len(reflection_text) > max_reflection_chars:
                reflection_text = reflection_text[:max_reflection_chars] + "\n... (truncated)"
            parts.append("[Previous reflections]\n" + reflection_text)
        if self._loop_records:
            loop_text = "\n\n".join(r.to_compressed() for r in self._loop_records[-8:])
            if len(loop_text) > max_loop_chars:
                loop_text = loop_text[:max_loop_chars] + "\n... (truncated)"
            parts.append("[Recent loop history]\n" + loop_text)
        return "\n\n".join(parts) if parts else ""
