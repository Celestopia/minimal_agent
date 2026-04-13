"""Core ReAct agent loop."""

from __future__ import annotations

import json
from pathlib import Path

from config import AgentConfig
from llm import DeepSeekChatClient
from memory import ConversationSession, SessionStore
from parser import parse_react_output
from prompts import PromptRenderer
from tools import ToolResult, build_tool_registry
from tracing import TraceLogger


class ReActAgent:
    """Local conversational ReAct agent with sliding-window memory."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.prompts = PromptRenderer()
        self.llm_client = DeepSeekChatClient(config.llm)
        self.session_store = SessionStore(config.paths.session_dir)
        self.tools = build_tool_registry(config.tools)

    def load_or_create_session(self, session_id: str | None = None) -> ConversationSession:
        """Load one persisted session or create a fresh session."""

        session = self.session_store.load_or_create(session_id)
        if not session.system_prompt:
            session.system_prompt = self._render_system_prompt()
            self.session_store.save(session)
        return session

    def _tool_descriptions(self) -> str:
        """Render the tool section injected into the system prompt."""

        rendered_lines: list[str] = []
        for tool in self.tools.values():
            rendered_lines.append(f"- {tool.name}: {tool.description}")
            properties = tool.parameters_schema.get("properties", {})
            if properties:
                rendered_lines.append("  action_input fields:")
                for field_name, field_schema in properties.items():
                    field_type = field_schema.get("type", "any")
                    field_description = field_schema.get("description", "").strip()
                    if field_description:
                        rendered_lines.append(
                            f"  - {field_name} ({field_type}): {field_description}"
                        )
                    else:
                        rendered_lines.append(f"  - {field_name} ({field_type})")
        return "\n".join(rendered_lines)

    def _render_system_prompt(self) -> str:
        """Render the invariant system prompt used for a session."""

        return self.prompts.render_system_prompt(
            tool_descriptions=self._tool_descriptions(),
            tool_names=", ".join(self.tools.keys()),
        )

    def _stop_message(self, reason: str) -> str:
        """Generate a user-facing fallback message for forced stops."""

        return (
            "The agent stopped before reaching a clean final answer. "
            f"Reason: {reason}"
        )

    def _build_react_messages(
        self,
        session: ConversationSession,
        user_prompt: str,
        react_messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Build the full chat message list sent to the model.

        The message list is structured as:
        1. One invariant system instruction message.
        2. Prior conversational turns as true user/assistant history.
        3. The current turn's user prompt.
        4. The ongoing ReAct trajectory as assistant messages, tool result
           messages, and occasional corrective user messages after failures.
        """

        messages: list[dict[str, object]] = [
            {"role": "system", "content": session.system_prompt}
        ]
        messages.extend(
            session.to_message_history(self.config.agent.sliding_window_turns)
        )
        messages.append({"role": "user", "content": user_prompt})
        messages.extend(react_messages)
        return messages

    def answer(
        self,
        session: ConversationSession,
        user_message: str,
    ) -> str:
        """Run one ReAct turn, update the session, and return the final answer."""

        turn_number = session.next_turn_number
        trace = TraceLogger.create(
            self.config.paths.trace_dir,
            session_id=session.session_id,
        )
        if not trace.trace_path.exists():
            trace.log(
                "session_started",
                {
                    "session_id": session.session_id,
                    "system_prompt": session.system_prompt,
                },
            )
        trace.log(
            "turn_started",
            {
                "session_id": session.session_id,
                "turn_number": turn_number,
                "user_message": user_message,
                "user_prompt": self.prompts.render_user_prompt(user_message),
                "config": self.config.public_snapshot(),
                "conversation_context": session.render_sliding_window(
                    self.config.agent.sliding_window_turns
                ),
            },
        )

        user_prompt = self.prompts.render_user_prompt(user_message)
        react_messages: list[dict[str, object]] = [] # Transient ReAct scratchpad for the current turn; will not be logged to session history.
        consecutive_failures = 0
        final_answer: str | None = None
        stop_reason = "max_steps_reached"
        
        for step_index in range(1, self.config.agent.max_steps_per_query + 1):
            # Each iteration asks the model for exactly one next ReAct step.
            # That step can end in one of three states:
            # 1. malformed output -> feed back a corrective observation and retry
            # 2. tool request -> execute the tool, append the observation, and continue
            # 3. final answer -> stop the loop
            messages = self._build_react_messages(session, user_prompt, react_messages)

            # Request the LLM for a response
            try:
                model_response = self.llm_client.generate(messages)
            except Exception as exc:
                consecutive_failures += 1
                observation = f"LLM call error: {type(exc).__name__}: {exc}"
                react_messages.append(
                    {
                        "role": "user",
                        "content": self.prompts.render_observation_prompt(observation),
                    }
                )
                trace.log(
                    "failure_counter_updated",
                    {
                        "turn_number": turn_number,
                        "step_index": step_index,
                        "consecutive_failures": consecutive_failures,
                        "reason": observation,
                    },
                )
                if consecutive_failures > self.config.agent.max_repeated_failures:
                    stop_reason = "too_many_repeated_failures"
                    break
                continue

            trace.log(
                "llm_response",
                {
                    "turn_number": turn_number,
                    "step_index": step_index,
                    "model": model_response.model,
                    "usage": model_response.usage,
                    "text": model_response.text,
                },
            )

            # Parse the model response to decide the next step
            decision = parse_react_output(model_response.text)
            trace.log(
                "decision_parsed",
                {
                    "turn_number": turn_number,
                    "step_index": step_index,
                    "thought": decision.thought,
                    "thought_summary": decision.thought_summary,
                    "status": decision.status,
                    "action": decision.action,
                    "action_input": decision.action_input,
                    "final_answer": decision.final_answer,
                    "malformed": decision.malformed,
                    "error_message": decision.error_message,
                },
            )

            if decision.malformed:
                consecutive_failures += 1
                observation = (
                    "Invalid response format. "
                    f"{decision.error_message or 'Please follow the protocol exactly.'}"
                )
                react_messages.append(
                    {
                        "role": "user",
                        "content": self.prompts.render_observation_prompt(observation),
                    }
                )
                trace.log(
                    "loop_observation",
                    {
                        "turn_number": turn_number,
                        "step_index": step_index,
                        "observation": observation,
                    },
                )
                trace.log(
                    "failure_counter_updated",
                    {
                        "turn_number": turn_number,
                        "step_index": step_index,
                        "consecutive_failures": consecutive_failures,
                        "reason": observation,
                    },
                )
                if consecutive_failures > self.config.agent.max_repeated_failures:
                    stop_reason = "too_many_repeated_failures"
                    break
                continue

            # If the final answer is reached, stop the loop.
            if decision.final_answer is not None:
                final_answer = decision.final_answer
                stop_reason = "final_answer"
                break

            # If a tool call is needed, execute the tool and append the observation.
            tool = self.tools.get(decision.action or "")
            if tool is None:
                result = ToolResult(
                    tool_name=decision.action or "(missing)",
                    tool_input=decision.action_input or {},
                    success=False,
                    output_text="",
                    error_text=(
                        f"Unknown tool '{decision.action}'. "
                        f"Available tools: {', '.join(self.tools.keys())}"
                    ),
                )
            else:
                result = tool.run(decision.action_input or {})

            trace.log(
                "tool_result",
                {
                    "turn_number": turn_number,
                    "step_index": step_index,
                    "tool_name": result.tool_name,
                    "tool_input": result.tool_input,
                    "success": result.success,
                    "output_text": result.output_text,
                    "error_text": result.error_text,
                    "metadata": result.metadata,
                },
            )

            observation_payload = result.to_observation_payload()
            observation = json.dumps(
                observation_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            react_messages.append({"role": "assistant", "content": model_response.text})
            react_messages.append(
                {
                    "role": "user",
                    "content": self.prompts.render_observation_prompt(observation),
                }
            )
            trace.log(
                "loop_observation",
                {
                    "turn_number": turn_number,
                    "step_index": step_index,
                    "observation": observation,
                },
            )

            # If the tool execution is successful, reset the failure counter and proceed;
            # If the tool execution fails, increment the failure counter and stop if it exceeds the threshold.
            if result.success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                trace.log(
                    "failure_counter_updated",
                    {
                        "turn_number": turn_number,
                        "step_index": step_index,
                        "consecutive_failures": consecutive_failures,
                        "reason": result.error_text or "Tool execution failed.",
                    },
                )
                if consecutive_failures > self.config.agent.max_repeated_failures:
                    stop_reason = "too_many_repeated_failures"
                    break

        # If the loop ends without a final answer, generate a fallback message.
        if final_answer is None:
            if stop_reason == "max_steps_reached":
                final_answer = self._stop_message(
                    f"it exceeded the configured {self.config.agent.max_steps_per_query} steps"
                )
            elif stop_reason == "too_many_repeated_failures":
                final_answer = self._stop_message(
                    "it exceeded the configured repeated failure threshold"
                )
            else:
                final_answer = self._stop_message(stop_reason)

        # Append the final answer of this turn to the session and save it.
        session.append_turn(
            user_message=user_message,
            assistant_message=final_answer,
            trace_path=str(trace.trace_path),
        )
        self.session_store.save(session)

        trace.log(
            "turn_finished",
            {
                "turn_number": turn_number,
                "stop_reason": stop_reason,
                "assistant_message": final_answer,
                "session_turn_count": len(session.turns),
            },
        )

        return final_answer
