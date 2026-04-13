"""Centralized prompt definitions and rendering helpers.

This project now keeps its prompt texts in Python rather than in separate
template files. That makes the prompt protocol live alongside the rest of the
runtime logic and avoids an extra top-level `prompts/` directory.
"""

from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT_TEMPLATE = """You are a careful local coding agent that follows a classic explicit ReAct loop.

You are helping with small coding tasks.
You can remember prior turns only through the chat history that accompanies each request.

Available tools:
{tool_descriptions}

You must always return a JSON object in your assistant message content.

The assistant message content must be a JSON object with exactly these keys:
- "thought": string
- "thought_summary": string
- "action": string or null
- "action_input": object or null
- "status": one of ["tool_call", "final"]
- "final_answer": string or null

When you need a tool:
- Set "status" to "tool_call"
- Set "action" to one of: {tool_names}
- Set "action_input" to a JSON object containing the tool parameters
- Set "final_answer" to null
- After a tool call, the environment will send back a user message containing the structured tool observation JSON

When you reach a final answer:
- Set "status" to "final"
- Set "final_answer" to a explanatory string
- Set "action" and "action_input" to null

Rules:
1. Always include both "thought" and "thought_summary" in the JSON object.
2. Never invent tool results.
3. Prefer using calculator for direct math expressions.
4. Prefer using python for executable Python experiments or debugging checks.
5. The python tool runs in Docker with no network and no file read/write access.
6. When a prior step fails, use the observation to recover rather than repeating the same broken step blindly.
"""

USER_TURN_PROMPT_TEMPLATE = """Current User Question:
{user_input}
"""

OBSERVATION_PROMPT_TEMPLATE = """Observation from the previous step:
{observation}

Produce exactly one next ReAct step for the current question.
"""


@dataclass(slots=True)
class PromptRenderer:
    """Renders the small set of prompts used by the agent runtime."""

    def render_system_prompt(
        self,
        tool_descriptions: str,
        tool_names: str,
    ) -> str:
        """Render the system prompt used for every model call."""

        return SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
            tool_names=tool_names,
        )

    def render_user_prompt(self, user_input: str) -> str:
        """Render the current-turn user message."""

        return USER_TURN_PROMPT_TEMPLATE.format(user_input=user_input)

    def render_observation_prompt(self, observation: str) -> str:
        """Render the user-side observation message for the next ReAct step."""

        return OBSERVATION_PROMPT_TEMPLATE.format(observation=observation)
