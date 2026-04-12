"""Parser for the explicit ReAct text protocol used by this project."""

from __future__ import annotations

from dataclasses import dataclass
import re


FIELD_NAMES = ["Thought", "Thought Summary", "Action", "Action Input", "Final Answer"]


@dataclass(slots=True)
class ParsedDecision:
    """Represents one parsed model decision inside the ReAct loop."""

    thought: str | None
    thought_summary: str | None
    action: str | None
    action_input: str | None
    final_answer: str | None
    raw_text: str
    malformed: bool
    error_message: str | None = None


def _extract_field(text: str, field_name: str, next_fields: list[str]) -> str | None:
    """Extract one labeled field from the raw model output.

    The parser is intentionally strict enough to catch malformed responses while
    still allowing multiline tool inputs such as Python code.
    """

    escaped_name = re.escape(field_name)
    if next_fields:
        next_pattern = "|".join(re.escape(name) for name in next_fields)
        pattern = rf"{escaped_name}\s*:\s*(.*?)(?=\n(?:{next_pattern})\s*:|\Z)"
    else:
        pattern = rf"{escaped_name}\s*:\s*(.*)\Z"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def parse_react_output(text: str) -> ParsedDecision:
    """Parse the model output into either an action request or a final answer."""

    text = text.strip()
    thought = _extract_field(
        text, "Thought", ["Thought Summary", "Action", "Action Input", "Final Answer"]
    )
    thought_summary = _extract_field(
        text, "Thought Summary", ["Action", "Action Input", "Final Answer"]
    )
    action = _extract_field(text, "Action", ["Action Input", "Final Answer"])
    action_input = _extract_field(text, "Action Input", ["Final Answer"])
    final_answer = _extract_field(text, "Final Answer", [])

    if final_answer and action:
        return ParsedDecision(
            thought=thought,
            thought_summary=thought_summary,
            action=action,
            action_input=action_input,
            final_answer=final_answer,
            raw_text=text,
            malformed=True,
            error_message="The response contains both an Action and a Final Answer.",
        )

    if final_answer:
        return ParsedDecision(
            thought=thought,
            thought_summary=thought_summary,
            action=None,
            action_input=None,
            final_answer=final_answer,
            raw_text=text,
            malformed=False,
        )

    if action and action_input:
        return ParsedDecision(
            thought=thought,
            thought_summary=thought_summary,
            action=action,
            action_input=action_input,
            final_answer=None,
            raw_text=text,
            malformed=False,
        )

    return ParsedDecision(
        thought=thought,
        thought_summary=thought_summary,
        action=action,
        action_input=action_input,
        final_answer=None,
        raw_text=text,
        malformed=True,
        error_message=(
            "The response did not match the required explicit ReAct format. "
            "Expected either an Action block or a Final Answer block."
        ),
    )
