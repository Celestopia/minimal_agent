"""JSONL trace logging and human-readable formatting utilities.

Tracing is session-scoped: one conversation session appends to one JSONL file.
This keeps the full dialogue and all ReAct steps in a single chronological log.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


def trace_timestamp() -> str:
    """Return an ISO timestamp used both in logs and file names."""

    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class TraceLogger:
    """Append-only JSONL trace logger for one session."""

    trace_path: Path

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append one structured event to the current trace file."""

        event = {
            "timestamp": trace_timestamp(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    @classmethod
    def create(cls, trace_dir: Path, session_id: str) -> "TraceLogger":
        """Create or reopen the single JSONL trace file for one session."""

        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_name = f"{session_id}.jsonl"
        trace_path = trace_dir / trace_name
        return cls(trace_path=trace_path)


def render_trace(trace_path: str | Path) -> str:
    """Convert one JSONL trace into a readable plain-text report."""

    trace_path = Path(trace_path).expanduser().resolve()
    lines = [f"Trace Report: {trace_path}", ""]

    with trace_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            event = json.loads(raw_line)
            event_type = event["event_type"]
            timestamp = event["timestamp"]
            payload = event["payload"]

            lines.append(f"[{timestamp}] {event_type}")

            if event_type == "session_started":
                lines.append(f"  Session: {payload['session_id']}")
                if payload.get("system_prompt"):
                    lines.append("  System prompt:")
                    lines.extend(
                        f"    {line}" for line in payload["system_prompt"].splitlines()
                    )
                    lines.append("")
                    lines.append("-" * 120)
                    lines.append("")
            elif event_type == "turn_started":
                lines.append(f"  Turn: {payload['turn_number']}")
                lines.append(f"  User message: {payload['user_message']}")
                if payload.get("user_prompt"):
                    lines.append("  User prompt:")
                    lines.extend(
                        f"    {line}" for line in payload["user_prompt"].splitlines()
                    )
            elif event_type == "prompt_rendered":
                lines.append(f"  Step: {payload['step_index']}")
                if payload.get("system_prompt") is not None:
                    lines.append("  System prompt:")
                    lines.extend(
                        f"    {line}" for line in payload["system_prompt"].splitlines()
                    )
                lines.append("  User prompt:")
                lines.extend(f"    {line}" for line in payload["user_prompt"].splitlines())
            elif event_type == "llm_response":
                if payload.get("turn_number") is not None:
                    lines.append(f"  Turn: {payload['turn_number']}")
                lines.append(f"  Step: {payload['step_index']}")
                lines.append(f"  Model: {payload['model']}")
                if payload.get("usage"):
                    lines.append(f"  Usage: {payload['usage']}")
                lines.append("  Raw response:")
                lines.extend(f"    {line}" for line in payload["text"].splitlines())
            elif event_type == "decision_parsed":
                if payload.get("turn_number") is not None:
                    lines.append(f"  Turn: {payload['turn_number']}")
                lines.append(f"  Step: {payload['step_index']}")
                lines.append(f"  Malformed: {payload['malformed']}")
                if payload.get("status"):
                    lines.append(f"  Status: {payload['status']}")
                if payload.get("thought"):
                    lines.append(f"  Thought: {payload['thought']}")
                if payload.get("thought_summary"):
                    lines.append(f"  Thought summary: {payload['thought_summary']}")
                if payload.get("action"):
                    lines.append(f"  Action: {payload['action']}")
                if payload.get("action_input"):
                    lines.append(f"  Action input: {payload['action_input']}")
                if payload.get("final_answer"):
                    lines.append(f"  Final answer: {payload['final_answer']}")
                if payload.get("error_message"):
                    lines.append(f"  Parse error: {payload['error_message']}")
            elif event_type == "tool_result":
                if payload.get("turn_number") is not None:
                    lines.append(f"  Turn: {payload['turn_number']}")
                lines.append(f"  Step: {payload['step_index']}")
                lines.append(f"  Tool: {payload['tool_name']}")
                lines.append(f"  Success: {payload['success']}")
                lines.append("  Tool input:")
                tool_input = payload["tool_input"]
                if isinstance(tool_input, dict):
                    rendered_tool_input = json.dumps(
                        tool_input,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    lines.extend(f"    {line}" for line in rendered_tool_input.splitlines())
                else:
                    lines.extend(f"    {line}" for line in str(tool_input).splitlines())
                if payload.get("output_text"):
                    lines.append("  Tool output:")
                    lines.extend(f"    {line}" for line in payload["output_text"].splitlines())
                if payload.get("error_text"):
                    lines.append("  Tool error:")
                    lines.extend(f"    {line}" for line in payload["error_text"].splitlines())
                if payload.get("metadata"):
                    lines.append(f"  Metadata: {payload['metadata']}")
            elif event_type == "loop_observation":
                if payload.get("turn_number") is not None:
                    lines.append(f"  Turn: {payload['turn_number']}")
                lines.append(f"  Step: {payload['step_index']}")
                lines.append("  Observation appended to scratchpad:")
                lines.extend(f"    {line}" for line in payload["observation"].splitlines())
            elif event_type == "failure_counter_updated":
                if payload.get("turn_number") is not None:
                    lines.append(f"  Turn: {payload['turn_number']}")
                if payload.get("step_index") is not None:
                    lines.append(f"  Step: {payload['step_index']}")
                lines.append(
                    f"  Consecutive failures: {payload['consecutive_failures']}"
                )
                lines.append(f"  Reason: {payload['reason']}")
            elif event_type == "turn_finished":
                if payload.get("turn_number") is not None:
                    lines.append(f"  Turn: {payload['turn_number']}")
                lines.append(f"  Stop reason: {payload['stop_reason']}")
                lines.append(f"  Assistant message: {payload['assistant_message']}")
                lines.append("")
                lines.append("-" * 120)
                lines.append("")
            else:
                lines.append(f"  Payload: {payload}")

            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
