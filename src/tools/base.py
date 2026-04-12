"""Shared tool abstractions for the local ReAct agent.

The concrete tool implementations live in sibling modules such as
`calculator.py` and `python_tool.py`. This file keeps the common dataclasses in
one place so the rest of the runtime can work with a small uniform interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class ToolResult:
    """Normalized tool execution result."""

    tool_name: str
    tool_input: str
    success: bool
    output_text: str
    error_text: str = ""
    metadata: dict[str, Any] | None = None

    def to_observation(self) -> str:
        """Convert the result into the plain-text observation fed back to ReAct."""

        lines = [f"Tool: {self.tool_name}", f"Success: {self.success}"]
        if self.output_text:
            lines.append("Output:")
            lines.append(self.output_text)
        if self.error_text:
            lines.append("Error:")
            lines.append(self.error_text)
        if self.metadata:
            lines.append(f"Metadata: {self.metadata}")
        return "\n".join(lines)


@dataclass(slots=True)
class Tool:
    """Small callable wrapper that is easy to log and prompt."""

    name: str
    description: str
    runner: Callable[[str], ToolResult]

    def run(self, tool_input: str) -> ToolResult:
        """Execute the underlying tool."""

        return self.runner(tool_input)
