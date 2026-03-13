"""
Task planning and decomposition: turn a task description into subtasks
and manage incremental completion toward the goal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class SubtaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class Subtask:
    title: str
    status: SubtaskStatus = SubtaskStatus.PENDING
    result: str | None = None

    def __str__(self) -> str:
        return f"[{self.status.value}] {self.title}"


@dataclass
class Plan:
    """A plan is a goal plus an ordered list of subtasks."""

    goal: str
    subtasks: list[Subtask] = field(default_factory=list)

    def next_pending(self) -> Subtask | None:
        for s in self.subtasks:
            if s.status == SubtaskStatus.PENDING:
                return s
        return None

    def mark_in_progress(self, subtask: Subtask) -> None:
        for s in self.subtasks:
            if s == subtask:
                s.status = SubtaskStatus.IN_PROGRESS
                return
        subtask.status = SubtaskStatus.IN_PROGRESS
        if subtask not in self.subtasks:
            self.subtasks.append(subtask)

    def mark_done(self, subtask: Subtask, result: str = "") -> None:
        for s in self.subtasks:
            if s is subtask or (s.title == subtask.title):
                s.status = SubtaskStatus.DONE
                s.result = result or s.result
                return
        subtask.status = SubtaskStatus.DONE
        subtask.result = result
        self.subtasks.append(subtask)

    def mark_skipped(self, subtask: Subtask, reason: str = "") -> None:
        for s in self.subtasks:
            if s is subtask or (s.title == subtask.title):
                s.status = SubtaskStatus.SKIPPED
                s.result = reason or s.result
                return
        subtask.status = SubtaskStatus.SKIPPED
        subtask.result = reason
        self.subtasks.append(subtask)

    def mark_failed(self, subtask: Subtask, reason: str = "") -> None:
        for s in self.subtasks:
            if s is subtask or (s.title == subtask.title):
                s.status = SubtaskStatus.FAILED
                s.result = reason or s.result
                return
        subtask.status = SubtaskStatus.FAILED
        subtask.result = reason
        self.subtasks.append(subtask)

    def all_done(self) -> bool:
        return all(
            s.status in (SubtaskStatus.DONE, SubtaskStatus.SKIPPED, SubtaskStatus.FAILED)
            for s in self.subtasks
        )

    def summary(self) -> str:
        lines = [f"Goal: {self.goal}", ""]
        for i, s in enumerate(self.subtasks, 1):
            lines.append(f"  {i}. {s}")
            if s.result:
                lines.append(f"      -> {s.result[:80]}{'...' if len(s.result) > 80 else ''}")
        return "\n".join(lines)


def make_plan(task: str, llm_subtasks_fn: Callable[[str], list[str]]) -> Plan:
    """Use LLM to decompose task into subtasks."""
    try:
        titles = llm_subtasks_fn(task)
    except Exception:
        titles = [task]
    if not titles:
        titles = [task]
    return Plan(goal=task, subtasks=[Subtask(title=t) for t in titles])
