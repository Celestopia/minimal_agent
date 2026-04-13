"""Conversation memory and session persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    """Return an ISO-8601 timestamp in UTC."""

    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ConversationTurn:
    """One persisted conversational exchange."""

    user_message: str
    assistant_message: str
    trace_path: str
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class ConversationSession:
    """A durable conversation session that can be resumed later."""

    session_id: str
    system_prompt: str = ""
    created_at: str = field(default_factory=utc_now)
    turns: list[ConversationTurn] = field(default_factory=list)

    @property
    def next_turn_number(self) -> int:
        """Return the one-based turn index for the next user message."""

        return len(self.turns) + 1

    def append_turn(self, user_message: str, assistant_message: str, trace_path: str) -> None:
        """Record a completed turn in session history."""

        self.turns.append(
            ConversationTurn(
                user_message=user_message,
                assistant_message=assistant_message,
                trace_path=trace_path,
            )
        )

    def render_sliding_window(self, window_turns: int) -> str:
        """Render the most recent conversation turns for prompt injection."""

        recent_turns = self.turns[-window_turns:] if window_turns > 0 else []
        if not recent_turns:
            return "(no prior conversation)"

        chunks: list[str] = []
        for index, turn in enumerate(
            recent_turns, start=max(1, len(self.turns) - len(recent_turns) + 1)
        ):
            chunks.append(
                "\n".join(
                    [
                        f"Turn {index} User: {turn.user_message}",
                        f"Turn {index} Assistant: {turn.assistant_message}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    def to_message_history(self, window_turns: int) -> list[dict[str, str]]:
        """Return recent conversation turns as alternating chat messages."""

        recent_turns = self.turns[-window_turns:] if window_turns > 0 else []
        messages: list[dict[str, str]] = []
        for turn in recent_turns:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_message})
        return messages


class SessionStore:
    """Small helper that reads and writes session files from disk."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def session_path(self, session_id: str) -> Path:
        """Return the JSON file used for one session."""

        return self.session_dir / f"{session_id}.json"

    def load_or_create(self, session_id: str | None = None) -> ConversationSession:
        """Load an existing session or create a new one if it does not exist."""

        session_id = session_id or f"session-{uuid4().hex[:12]}"
        path = self.session_path(session_id)
        if not path.exists():
            session = ConversationSession(session_id=session_id)
            self.save(session)
            return session

        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        turns = [
            ConversationTurn(**turn_data)
            for turn_data in raw.get("turns", [])
        ]
        return ConversationSession(
            session_id=raw["session_id"],
            system_prompt=raw.get("system_prompt", ""),
            created_at=raw.get("created_at", utc_now()),
            turns=turns,
        )

    def save(self, session: ConversationSession) -> Path:
        """Persist one session atomically to JSON."""

        path = self.session_path(session.session_id)
        payload: dict[str, Any] = {
            "session_id": session.session_id,
            "system_prompt": session.system_prompt,
            "created_at": session.created_at,
            "turns": [asdict(turn) for turn in session.turns],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path
