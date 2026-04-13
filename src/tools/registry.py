"""Tool registry assembly.

This module wires together the individual tool implementations into the
dictionary that the agent exposes to the model and uses at runtime.
"""

from __future__ import annotations

from config import ToolConfig
from tools.base import Tool
from tools.calculator import run_calculator
from tools.python_tool import DockerPythonExecutor


def build_tool_registry(config: ToolConfig) -> dict[str, Tool]:
    """Create the tool registry that is shown to the model and used at runtime."""

    tools: dict[str, Tool] = {}

    if config.calculator.enabled:
        tools["calculator"] = Tool(
            name="calculator",
            description=(
                "Evaluate one math expression. Supports arithmetic and common "
                "functions such as sqrt, sin, cos, tan, log, exp, abs, and round."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A single arithmetic or math expression.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            runner=run_calculator,
        )

    if config.python.enabled:
        executor = DockerPythonExecutor(
            docker_image=config.python.docker_image,
            timeout_seconds=config.python.timeout_seconds,
            memory_limit=config.python.memory_limit,
            cpu_limit=config.python.cpu_limit,
            pids_limit=config.python.pids_limit,
        )
        tools["python"] = Tool(
            name="python",
            description=(
                "Run a short Python snippet inside an isolated Docker container. "
                "No network and no file read/write access are allowed. "
                "Pass the full snippet in the 'code' tool argument."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "A short Python snippet to execute.",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            runner=executor.run,
        )

    return tools
