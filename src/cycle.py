"""
Cycle detection: detect when the agent repeats the same or similar tool calls.
Interrupts and prompts the agent to try something else.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

MIN_SAME_REPEATS = 3  # same call repeated this many times
MIN_SEQUENCE_REPEATS = 2  # sequence repeated this many times
MAX_SEQUENCE_LEN = 4  # max sequence length to check
CYCLE_INTERRUPT_MSG = (
    "[CYCLE DETECTED] You are repeating the same tool calls. "
    "Try a different approach. Consider: a different tool, different arguments, "
    "or asking the user for clarification."
)


def _signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """Normalize tool call to a comparable signature."""
    args = arguments or {}
    # Sort keys for consistent comparison
    key_parts = []
    for k in sorted(args):
        v = args[k]
        if isinstance(v, str) and len(v) > 100:
            v = v[:100] + "..."
        key_parts.append(f"{k}={repr(v)}")
    return f"{tool_name}({','.join(key_parts)})"


def _signatures_from_acts(acts: list[dict[str, Any] | None]) -> list[str]:
    """Extract signatures from a list of act dicts (tool_name, arguments)."""
    out = []
    for a in acts:
        if not a:
            continue
        name = a.get("tool_name", "")
        args = a.get("arguments", {})
        out.append(_signature(name, args))
    return out


def detect_cycle(signatures: list[str]) -> bool:
    """
    Detect if the last entries form a cycle.
    Returns True if:
    - Same signature repeated MIN_SAME_REPEATS+ times, or
    - A sequence of 2..MAX_SEQUENCE_LEN signatures repeated MIN_SEQUENCE_REPEATS+ times.
    """
    if len(signatures) < MIN_SAME_REPEATS:
        return False

    # Check: same call repeated
    last = signatures[-1]
    count = 0
    for s in reversed(signatures):
        if s == last:
            count += 1
        else:
            break
    if count >= MIN_SAME_REPEATS:
        return True

    # Check: sequence repeated
    for period in range(2, min(MAX_SEQUENCE_LEN + 1, len(signatures) // MIN_SEQUENCE_REPEATS + 1)):
        needed = period * MIN_SEQUENCE_REPEATS
        if len(signatures) < needed:
            continue
        tail = signatures[-needed:]
        first_chunk = tail[:period]
        is_repeat = all(tail[i : i + period] == first_chunk for i in range(0, needed, period))
        if is_repeat:
            return True

    return False


def check_and_interrupt(
    acts: list[dict[str, Any] | None],
    current_act: dict[str, Any] | None,
) -> tuple[bool, str]:
    """
    Check if adding current_act to acts would form a cycle.
    Returns (should_interrupt, message).
    If should_interrupt, do not run the tool; inject the message instead.
    """
    if not current_act:
        return False, ""
    sigs = _signatures_from_acts(acts + [current_act])
    if detect_cycle(sigs):
        return True, CYCLE_INTERRUPT_MSG
    return False, ""
