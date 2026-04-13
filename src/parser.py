"""Parser for the JSON-based ReAct protocol used by this project."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(slots=True)
class ParsedDecision:
    """Represents one parsed model decision inside the ReAct loop."""

    thought: str | None
    thought_summary: str | None
    status: str | None
    action: str | None
    action_input: dict[str, Any] | None
    final_answer: str | None
    raw_text: str
    malformed: bool
    error_message: str | None = None


def _validate_json_payload(payload: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate the assistant JSON content payload.
    
    Args:
        payload (dict): The assistant JSON content payload.

    Returns:
        is_valid, validation_error (tuple): A tuple containing a boolean indicating
            whether the payload is valid, and an error message if it is not.
    """

    required_keys = (
        "thought",
        "thought_summary",
        "status",
        "action",
        "action_input",
        "final_answer",
    )
    valid_statuses = (
        "tool_call",
        "final",
    )

    if set(payload.keys()) != set(required_keys):
        return (
            False,
            f"The JSON content must contain exactly these keys: {', '.join(required_keys)}.",
        )

    thought = payload["thought"]
    thought_summary = payload["thought_summary"]
    status = payload["status"]
    action = payload["action"]
    action_input = payload["action_input"]
    final_answer = payload["final_answer"]

    if not isinstance(thought, str) or not thought.strip():
        return False, "The 'thought' field must be a non-empty string."
    if not isinstance(thought_summary, str) or not thought_summary.strip():
        return False, "The 'thought_summary' field must be a non-empty string."
    if status not in valid_statuses:
        return False, f"The 'status' field must be one of {', '.join(valid_statuses)}."
    if action is not None and not isinstance(action, str):
        return False, "The 'action' field must be a string or null."
    if action_input is not None and not isinstance(action_input, dict):
        return False, "The 'action_input' field must be a dictionary or null."
    if final_answer is not None and not isinstance(final_answer, str):
        return False, "The 'final_answer' field must be a string or null."

    if status == "tool_call":
        if action is None or not action.strip():
            return False, "Tool-call steps must provide a non-empty 'action' string."
        if action_input is None:
            return False, "Tool-call steps must provide an 'action_input' JSON object."
        if final_answer is not None:
            return False, "Tool-call steps must set 'final_answer' to null."
    
    if status == "final":
        if final_answer is None or not final_answer.strip():
            return False, "Final steps must provide a non-empty 'final_answer' string."
        if action is not None:
            return False, "Final steps must set 'action' to null."
        if action_input is not None:
            return False, "Final steps must set 'action_input' to null."

    return True, None


def parse_react_output(text: str) -> ParsedDecision:
    """Parse the model output into either a tool call decision or a final answer."""

    text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParsedDecision(
            thought=None,
            thought_summary=None,
            status=None,
            action=None,
            action_input=None,
            final_answer=None,
            raw_text=text,
            malformed=True,
            error_message=f"Assistant content is not valid JSON: {exc}",
        )

    if not isinstance(payload, dict):
        return ParsedDecision(
            thought=None,
            thought_summary=None,
            status=None,
            action=None,
            action_input=None,
            final_answer=None,
            raw_text=text,
            malformed=True,
            error_message="Assistant content must be parsable to a dictionary.",
        )

    is_valid, validation_error = _validate_json_payload(payload)
    if not is_valid:
        return ParsedDecision(
            thought=payload.get("thought") if isinstance(payload.get("thought"), str) else None,
            thought_summary=payload.get("thought_summary")
                if isinstance(payload.get("thought_summary"), str)
                else None,
            status=payload.get("status") if isinstance(payload.get("status"), str) else None,
            action=payload.get("action") if isinstance(payload.get("action"), str) else None,
            action_input=payload.get("action_input")
                if isinstance(payload.get("action_input"), dict)
                else None,
            final_answer=payload.get("final_answer")
                if isinstance(payload.get("final_answer"), str)
                else None,
            raw_text=text,
            malformed=True,
            error_message=validation_error,
        )

    status = payload["status"].strip()
    thought = payload["thought"].strip()
    thought_summary = payload["thought_summary"].strip()
    action = payload["action"].strip() if isinstance(payload["action"], str) else None
    action_input = payload["action_input"]
    final_answer = payload["final_answer"].strip() if isinstance(payload["final_answer"], str) else None

    return ParsedDecision(
        thought=thought,
        thought_summary=thought_summary,
        status=status,
        action=action,
        action_input=action_input,
        final_answer=final_answer,
        raw_text=text,
        malformed=False,
    )
