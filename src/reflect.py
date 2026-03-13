"""
Reflection and importance scoring for the ReAct loop.
Assigns importance scores to loop iterations using LLM + heuristics.
When importance exceeds threshold, triggers reflection to adapt strategy.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

IMPORTANCE_THRESHOLD = 0.65
MAX_REFLECTION_CHARS = 1500


@dataclass
class LoopRecord:
    """Labelled record of one ReAct loop iteration."""

    input: str  # user prompt or subtask/objective
    reasoning: str  # agent's understanding
    act: dict[str, Any] | None  # tool_name, arguments (literal)
    observe: str  # tool output or observation
    update: str  # compressed state summary
    iteration: int = 0

    def to_compressed(self) -> str:
        """Compressed string: labels, literal args, filenames; no helping words."""
        parts = [
            f"[INPUT] {self.input[:200]}{'...' if len(self.input) > 200 else ''}",
            f"[REASONING] {self.reasoning[:300]}{'...' if len(self.reasoning) > 300 else ''}",
        ]
        if self.act:
            name = self.act.get("tool_name", "")
            args = self.act.get("arguments", {})
            args_str = json.dumps(args) if args else "{}"
            parts.append(f"[ACT] tool={name} args={args_str}")
        parts.append(f"[OBSERVE] {self.observe[:400]}{'...' if len(self.observe) > 400 else ''}")
        parts.append(f"[UPDATE] {self.update[:200]}{'...' if len(self.update) > 200 else ''}")
        return "\n".join(parts)


@dataclass
class Reflection:
    """A reflection on what happened and how to adapt."""

    trigger: str  # what triggered this reflection
    summary: str  # what happened
    adaptation: str  # how to change strategy
    loop_indices: list[int] = field(default_factory=list)  # which loops informed this

    def to_context(self) -> str:
        return f"[REFLECTION] trigger={self.trigger}\nsummary={self.summary}\nadaptation={self.adaptation}"


def _parse_summary_adaptation(
    raw: str,
    default_summary: str = "Observation noted.",
    default_adaptation: str = "Adjust approach as needed.",
) -> tuple[str, str]:
    """Parse SUMMARY/ADAPTATION from LLM response text. Returns (summary, adaptation)."""
    summary = ""
    adaptation = ""
    in_summary = True
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("ADAPTATION:"):
            in_summary = False
            adaptation = line[11:].strip()
        elif line.upper().startswith("SUMMARY:"):
            in_summary = True
            summary = line[8:].strip()
        elif in_summary and summary:
            summary += " " + line
        elif not in_summary and adaptation:
            adaptation += " " + line
        elif in_summary:
            summary = line
        else:
            adaptation = line
    if not summary:
        summary = default_summary
    if not adaptation:
        adaptation = default_adaptation
    if len(summary) > MAX_REFLECTION_CHARS:
        summary = summary[:MAX_REFLECTION_CHARS] + "..."
    if len(adaptation) > MAX_REFLECTION_CHARS:
        adaptation = adaptation[:MAX_REFLECTION_CHARS] + "..."
    return summary, adaptation


def compute_query_importance(query: str, llm_fn: Callable[[str], str]) -> float:
    """Use LLM to rate importance of user's query (0.0-1.0) for triggering reflection."""
    if not query or not query.strip():
        return 0.0
    prompt = f"""Rate how important this user message is for the agent to reflect and adapt (0.0-1.0).
High (0.7-1.0): user correcting the agent, giving feedback, requesting behavior change, pointing out errors, asking to fix/redo/undo. Also: requests that remain unclear despite conversation context, so the agent asks for clarification.
Low (0.0-0.4): routine task, first request, simple question, "continue", "yes".

User message: "{query[:500]}"

Reply with a single float between 0.0 and 1.0, nothing else."""
    try:
        raw = llm_fn(prompt).strip()
        for part in raw.replace(",", " ").split():
            try:
                score = float(part)
                return max(0.0, min(1.0, score))
            except ValueError:
                continue
    except Exception as e:
        logger.warning("Query importance scoring failed: %s", e)
    return 0.0


def generate_reflection_from_query(
    query: str,
    context: str,
    llm_fn: Callable[[str], str],
    prior_reflections: list[Reflection] | None = None,
) -> Reflection:
    """Use LLM to process context and user query to generate a reflection to save."""
    prior_ctx = ""
    if prior_reflections:
        prior_ctx = "\n\nPrior reflections:\n" + "\n".join(r.to_context() for r in prior_reflections[-3:])

    prompt = f"""The user said: "{query[:400]}"

Context (recent loop history, tool calls, observations):
{context[:3000] if context else "No context yet."}
{prior_ctx}

Based on the user's message and context, provide a reflection:
1. SUMMARY: What is the user conveying? What should the agent understand?
2. ADAPTATION: How should the agent change its approach or behavior?

Format:
SUMMARY: ...
ADAPTATION: ..."""
    try:
        raw = llm_fn(prompt).strip()
        summary, adaptation = _parse_summary_adaptation(
            raw,
            default_summary="User provided feedback or correction.",
            default_adaptation="Adapt to the user's request.",
        )
    except Exception as e:
        logger.warning("Reflection from query failed: %s", e)
        summary = "User feedback detected."
        adaptation = "Adjust behavior per user request."

    return Reflection(
        trigger=f"user query: {query[:100]}...",
        summary=summary,
        adaptation=adaptation,
        loop_indices=[],
    )


def _heuristic_importance(records: list[LoopRecord], idx: int) -> float:
    """Heuristic importance: repeated tool calls, errors, iteration depth."""
    score = 0.0
    r = records[idx]

    # Same tool+args repeated recently
    if r.act:
        tool_name = r.act.get("tool_name", "")
        args = r.act.get("arguments", {})
        path = args.get("path", args.get("query", ""))
        for i in range(max(0, idx - 3), idx):
            prev = records[i].act if i < len(records) else None
            if prev and prev.get("tool_name") == tool_name:
                prev_args = prev.get("arguments", {})
                prev_path = prev_args.get("path", prev_args.get("query", ""))
                if path and prev_path and path == prev_path:
                    score += 0.4  # likely do/undo or repeated attempt
                    break

    # Error in observe
    obs = (r.observe or "").lower()
    if "[error]" in obs or "not found" in obs:
        score += 0.3
    # Shell denial: force reflection (high importance)
    if "user declined to run shell command" in obs:
        score += 0.8
    elif "[permission denied]" in obs:
        score += 0.5

    # High iteration count
    if r.iteration >= 5:
        score += 0.15
    if r.iteration >= 8:
        score += 0.2

    return min(1.0, score)


def compute_importance(
    records: list[LoopRecord],
    idx: int,
    llm_fn: Callable[[str], str],
) -> float:
    """Combine heuristic and LLM-based importance score."""
    h_score = _heuristic_importance(records, idx)
    if idx >= len(records):
        return h_score

    r = records[idx]
    prompt = f"""Rate importance of this loop (0.0-1.0) for needing reflection.
High importance: user making corrections or giving feedback, repeated failures, circular behavior, stuck, wrong strategy.
Low importance: steady progress, first attempts.

{r.to_compressed()}

Reply with a single float between 0.0 and 1.0, nothing else."""
    try:
        raw = llm_fn(prompt).strip()
        for part in raw.replace(",", " ").split():
            try:
                llm_score = float(part)
                llm_score = max(0.0, min(1.0, llm_score))
                break
            except ValueError:
                continue
        else:
            llm_score = 0.5
    except Exception as e:
        logger.warning("LLM importance scoring failed: %s", e)
        llm_score = 0.5

    # Blend: heuristics catch obvious cases; LLM adds nuance
    return 0.5 * h_score + 0.5 * llm_score


def generate_reflection(
    records: list[LoopRecord],
    indices: list[int],
    trigger: str,
    llm_fn: Callable[[str], str],
    prior_reflections: list[Reflection] | None = None,
) -> Reflection:
    """Ask LLM to summarize what happened and how to adapt. Prior reflections can inform this one."""
    context_parts = []
    for i in indices:
        if i < len(records):
            context_parts.append(f"--- Loop {records[i].iteration} ---\n{records[i].to_compressed()}")
    context = "\n\n".join(context_parts)

    prior_ctx = ""
    if prior_reflections:
        prior_ctx = "\n\nPrior reflections (use to inform this one):\n" + "\n".join(
            r.to_context() for r in prior_reflections[-3:]
        )

    prompt = f"""You observed this pattern in the agent's behavior. Trigger: {trigger}
{prior_ctx}

{context}

Provide:
1. SUMMARY: What happened? (2-3 sentences, factual. Include tool names, filenames, literal actions.)
2. ADAPTATION: How should the agent change its strategy? (Concrete, actionable.)

Format:
SUMMARY: ...
ADAPTATION: ..."""
    try:
        raw = llm_fn(prompt).strip()
        summary, adaptation = _parse_summary_adaptation(
            raw,
            default_summary="Repeated or stuck behavior detected.",
            default_adaptation="Try a different approach; avoid repeating the same tool call.",
        )
    except Exception as e:
        logger.warning("Reflection generation failed: %s", e)
        summary = "Behavior pattern detected; strategy change recommended."
        adaptation = "Avoid repeating same actions; try alternative approach."

    return Reflection(
        trigger=trigger,
        summary=summary,
        adaptation=adaptation,
        loop_indices=indices,
    )


def generate_reflection_for_shell_denial(
    command: str,
    llm_fn: Callable[[str], str],
    prior_reflections: list[Reflection] | None = None,
) -> Reflection:
    """Generate reflection when user declined to run a shell command. Updates strategy for next tasks."""
    prior_ctx = ""
    if prior_reflections:
        prior_ctx = "\n\nPrior reflections:\n" + "\n".join(
            r.to_context() for r in prior_reflections[-3:]
        )

    prompt = f"""The user declined to run this shell command: "{command[:200]}"

This is a high-importance event. The agent must adapt how it handles tasks and subtasks.

{prior_ctx}

Provide:
1. SUMMARY: What happened? (User refused shell execution.)
2. ADAPTATION: How should the agent change its strategy? Be concrete:
   - Prefer file-based tools (read_file, write_file, edit_file, search_files) over use_shell when possible.
   - If a subtask requires shell (e.g. running tests), consider marking it as skipped or asking the user for permission first.
   - Do not repeatedly request shell commands the user has declined.
   - When blocked, mark subtasks as skipped/failed and move on or ask the user for guidance.

Format:
SUMMARY: ...
ADAPTATION: ..."""
    default_adaptation = (
        "Prefer file-based tools over shell. When blocked by shell refusal, "
        "mark subtasks as skipped and move on or ask the user for guidance."
    )
    try:
        raw = llm_fn(prompt).strip()
        summary, adaptation = _parse_summary_adaptation(
            raw,
            default_summary="User declined to run shell command.",
            default_adaptation=default_adaptation,
        )
    except Exception as e:
        logger.warning("Shell denial reflection failed: %s", e)
        summary = "User declined to run shell command."
        adaptation = (
            "Prefer file-based tools over shell. When blocked, mark subtasks as skipped "
            "and move on. Do not repeatedly request shell commands."
        )

    return Reflection(
        trigger="user declined shell command",
        summary=summary,
        adaptation=adaptation,
        loop_indices=[],
    )


# Error indicators in script/run output
_RUN_ERROR_INDICATORS = (
    "[error]",
    "traceback",
    "exception",
    "syntaxerror",
    "nameerror",
    "typeerror",
    "valueerror",
    "indexerror",
    "keyerror",
    "attributeerror",
    "modulenotfounderror",
    "filenotfounderror",
    "importerror",
    "runtimeerror",
    "assertionerror",
    "failed",
    "error:",
)


def is_run_error_output(observe_text: str) -> bool:
    """Detect if observe output indicates a script/run error (e.g. from use_shell)."""
    if not observe_text or len(observe_text) < 3:
        return False
    lower = observe_text.lower()
    return any(ind in lower for ind in _RUN_ERROR_INDICATORS)


def generate_reflection_for_run_error(
    observe_text: str,
    act_info: dict | None,
    llm_fn: Callable[[str], str],
    prior_reflections: list[Reflection] | None = None,
) -> Reflection:
    """Generate reflection when script/run output contains an error. Agent should fix before continuing."""
    prior_ctx = ""
    if prior_reflections:
        prior_ctx = "\n\nPrior reflections:\n" + "\n".join(
            r.to_context() for r in prior_reflections[-3:]
        )

    tool_name = (act_info or {}).get("tool_name", "")
    args = (act_info or {}).get("arguments", {})
    cmd = args.get("command", "") if isinstance(args, dict) else ""

    prompt = f"""The agent ran a command/script and the output indicates an error.

Tool: {tool_name}
Command: {cmd[:300] if cmd else "(N/A)"}

Output (excerpt):
{observe_text[:1200] if observe_text else "(empty)"}
{prior_ctx}

Provide:
1. SUMMARY: What error occurred? (Brief, factual.)
2. ADAPTATION: The agent must fix this error before continuing. Do not proceed to the next subtask until the error is resolved. Read the error message, identify the cause (syntax, missing file, wrong import, etc.), and fix the code. Then run again to verify.

Format:
SUMMARY: ...
ADAPTATION: ..."""
    default_adaptation = (
        "Fix the error before continuing. Read the error, identify the cause, "
        "edit the code, and run again to verify. Do not proceed until it passes."
    )
    try:
        raw = llm_fn(prompt).strip()
        summary, adaptation = _parse_summary_adaptation(
            raw,
            default_summary="Script/run produced an error.",
            default_adaptation=default_adaptation,
        )
    except Exception as e:
        logger.warning("Run error reflection failed: %s", e)
        summary = "Script/run produced an error."
        adaptation = (
            "Fix the error before continuing. Identify the cause from the output, "
            "edit the code, and run again. Do not proceed to next steps until it passes."
        )

    return Reflection(
        trigger="run/script error in output",
        summary=summary,
        adaptation=adaptation,
        loop_indices=[],
    )
