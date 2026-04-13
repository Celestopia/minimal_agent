"""Command-line interface for the local ReAct agent repository."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from agent import ReActAgent
from config import load_agent_config
from tracing import render_trace


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser with the supported subcommands."""

    parser = argparse.ArgumentParser(
        description="Minimal local conversational ReAct agent."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Chat function
    chat_parser = subparsers.add_parser(
        "chat", help="Start an interactive multi-turn session."
    )
    chat_parser.add_argument(
        "--session-id",
        "-S",
        default=None,
        help="Resume or create a named session. If omitted, a fresh session is created.",
    )

    # Trace formatting function
    format_parser = subparsers.add_parser(
        "format-trace", help="Render one JSONL trace into plain text."
    )
    format_parser.add_argument("trace_path", help="Path to a JSONL trace file.")
    format_parser.add_argument(
        "--output",
        "-O",
        default=None,
        help="Optional output file path. If omitted, the report is printed to stdout.",
    )

    return parser


def run_chat(agent: ReActAgent, session_id: str | None) -> int:
    """Run the interactive chat loop."""

    session = agent.load_or_create_session(session_id)
    print(f"Session: {session.session_id}")
    print("Commands: /exit, /quit, /session, /history")

    while True:
        try:
            user_message = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            return 0

        if not user_message:
            continue
        if user_message in {"/exit", "/quit"}:
            print("Exiting chat.")
            return 0
        if user_message == "/session":
            print(f"Current session: {session.session_id}")
            continue
        if user_message == "/history":
            print("Chat history printed below.")
            print(session.render_sliding_window(agent.config.agent.sliding_window_turns))
            continue

        answer = agent.answer(session, user_message)
        print(f"\nAgent> {answer}")


def run_format_trace(trace_path: str, output_path: str | None) -> int:
    """Render one trace and either print it or write it to a file."""

    rendered = render_trace(trace_path)
    if output_path is None:
        print(rendered, end="")
        return 0

    target = Path(output_path).expanduser().resolve()
    target.write_text(rendered, encoding="utf-8")
    print(f"Wrote formatted trace to {target!r}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "format-trace":
        return run_format_trace(args.trace_path, args.output)

    config = load_agent_config(args.config)
    agent = ReActAgent(config)

    if args.command == "chat":
        return run_chat(agent, args.session_id)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
