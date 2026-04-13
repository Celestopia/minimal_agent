"""Focused unit tests for the local ReAct agent repository."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_agent_config
from memory import ConversationSession, SessionStore
from parser import parse_react_output
from tools import run_calculator
from tracing import TraceLogger, render_trace


class ParserTests(unittest.TestCase):
    """Verify the JSON-only ReAct parser."""

    def test_parse_tool_call_step(self) -> None:
        raw = (
            '{"thought":"I should verify the math exactly.",'
            '"thought_summary":"Use calculator.",'
            '"action":"calculator",'
            '"action_input":{"expression":"sqrt(81) + 2"},'
            '"status":"tool_call",'
            '"final_answer":null}'
        )
        decision = parse_react_output(raw)
        self.assertFalse(decision.malformed)
        self.assertEqual(decision.status, "tool_call")
        self.assertEqual(decision.action, "calculator")
        self.assertEqual(decision.action_input, {"expression": "sqrt(81) + 2"})

    def test_parse_final_answer(self) -> None:
        raw = (
            '{"thought":"I now know the result.",'
            '"thought_summary":"Answer directly.",'
            '"action":null,'
            '"action_input":null,'
            '"status":"final",'
            '"final_answer":"The result is 11."}'
        )
        decision = parse_react_output(raw)
        self.assertFalse(decision.malformed)
        self.assertEqual(decision.final_answer, "The result is 11.")

    def test_parse_tool_call_requires_action_input_object(self) -> None:
        raw = (
            '{"thought":"Need a tool.",'
            '"thought_summary":"Use tool.",'
            '"action":"calculator",'
            '"action_input":null,'
            '"status":"tool_call",'
            '"final_answer":null}'
        )
        decision = parse_react_output(raw)
        self.assertTrue(decision.malformed)
        self.assertIn("action_input", decision.error_message or "")


class CalculatorTests(unittest.TestCase):
    """Verify the safe calculator tool."""

    def test_calculator_supports_math_functions(self) -> None:
        result = run_calculator({"expression": "sqrt(81) + log(e)"})
        self.assertTrue(result.success)
        self.assertEqual(result.output_text, "10.0")
        self.assertEqual(result.tool_input, {"expression": "sqrt(81) + log(e)"})

    def test_calculator_rejects_unsafe_names(self) -> None:
        result = run_calculator({"expression": "__import__('os').system('echo bad idea')"})
        self.assertFalse(result.success)


class MemoryTests(unittest.TestCase):
    """Verify sliding-window rendering."""

    def test_render_sliding_window(self) -> None:
        session = ConversationSession(session_id="demo")
        session.append_turn("u1", "a1", "trace1")
        session.append_turn("u2", "a2", "trace2")
        session.append_turn("u3", "a3", "trace3")
        rendered = session.render_sliding_window(2)
        self.assertIn("Turn 2 User: u2", rendered)
        self.assertIn("Turn 3 Assistant: a3", rendered)
        self.assertNotIn("Turn 1 User: u1", rendered)

    def test_to_message_history(self) -> None:
        session = ConversationSession(session_id="demo")
        session.append_turn("u1", "a1", "trace1")
        session.append_turn("u2", "a2", "trace2")
        history = session.to_message_history(2)
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ],
        )

    def test_session_store_persists_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(Path(tmp_dir))
            session = ConversationSession(
                session_id="demo",
                system_prompt="You are the system prompt.",
            )
            store.save(session)
            loaded = store.load_or_create("demo")
            self.assertEqual(loaded.system_prompt, "You are the system prompt.")


class ConfigAndTraceTests(unittest.TestCase):
    """Verify config loading and trace formatting helpers."""

    def test_load_agent_config(self) -> None:
        config = load_agent_config(Path(__file__).resolve().parents[1] / "config.yaml")
        self.assertEqual(config.agent.max_steps_per_query, 10)
        self.assertEqual(config.tools.python.docker_image, "python:3.11-slim")

    def test_render_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "sample.jsonl"
            trace_path.write_text(
                "\n".join(
                    [
                        '{"timestamp":"2026-01-01T00:00:00+00:00","event_type":"session_started","payload":{"session_id":"demo","system_prompt":"You are the system prompt."}}',
                        '{"timestamp":"2026-01-01T00:00:00+00:00","event_type":"turn_started","payload":{"session_id":"demo","turn_number":1,"user_message":"hello"}}',
                        '{"timestamp":"2026-01-01T00:00:01+00:00","event_type":"turn_finished","payload":{"stop_reason":"final_answer","assistant_message":"hi"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rendered = render_trace(trace_path)
            self.assertIn("Trace Report", rendered)
            self.assertIn("System prompt:", rendered)
            self.assertIn("You are the system prompt.", rendered)
            self.assertIn("turn_started", rendered)
            self.assertIn("Assistant message: hi", rendered)

    def test_trace_logger_uses_one_file_per_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_dir = Path(tmp_dir)
            logger_one = TraceLogger.create(trace_dir, "session-demo")
            logger_two = TraceLogger.create(trace_dir, "session-demo")
            self.assertEqual(logger_one.trace_path, logger_two.trace_path)
            self.assertEqual(logger_one.trace_path.name, "session-demo.jsonl")


if __name__ == "__main__":
    unittest.main()
