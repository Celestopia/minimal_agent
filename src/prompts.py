"""Prompt template loading and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PromptRenderer:
    """Loads prompt templates from disk so they stay easy to inspect and edit."""

    prompt_dir: Path

    def _read_template(self, filename: str) -> str:
        path = self.prompt_dir / filename
        return path.read_text(encoding="utf-8")

    def render_system_prompt(
        self,
        tool_descriptions: str,
        tool_names: str,
    ) -> str:
        """Render the system prompt used for every model call."""

        template = self._read_template("system_prompt.txt")
        return template.format(
            tool_descriptions=tool_descriptions,
            tool_names=tool_names,
        )

    def render_user_prompt(self, user_input: str) -> str:
        """Render the current-turn user message."""

        template = self._read_template("user_turn_prompt.txt")
        return template.format(user_input=user_input)

    def render_observation_prompt(self, observation: str) -> str:
        """Render the user-side observation message for the next ReAct step."""

        template = self._read_template("observation_prompt.txt")
        return template.format(observation=observation)
