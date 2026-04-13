# Project Design

## Goal

Build a simple but readable local AI agent repository for studying agent behavior on small Python coding and debugging tasks. The design intentionally favors clarity and inspectability over hidden abstractions.

## High-Level Architecture

The system is composed of six main layers:

1. `main.py` and `src/cli.py`
   `main.py` is the direct repository launcher for `python main.py ...`, and `src/cli.py` holds the interactive chat and trace-formatting command parsing and runtime dispatch. The CLI uses `argparse` subparsers so the repository has two command modes: `chat` and `format-trace`.

2. `memory.py`
   Persists conversations to `sessions/` and renders a sliding window of recent turns for prompt context.

3. `prompts.py`
   Stores the prompt definitions and renders them with runtime values.

4. `llm.py`
   Wraps the OpenAI Python SDK configured for DeepSeek's OpenAI-compatible endpoint.

5. `agent.py`
   Implements the explicit ReAct loop, retry and recovery behavior, stopping logic, trace logging, and session updates.

6. `src/tools/`
   Hosts the calculator tool, the Docker-backed Python execution tool, and the small registry/base modules that wire those tools into the agent.

## CLI Parsing Rules

The command-line interface is intentionally small and uses subcommands rather than one large flat argument surface.

- `python main.py chat` starts the interactive agent
- `python main.py chat --session-id demo` or `python main.py chat -S demo` resumes or creates a named session
- `python main.py format-trace traces/demo.jsonl` renders a trace report
- `python main.py format-trace traces/demo.jsonl --output report.txt` or `-O report.txt` writes the rendered report to a file

The parsing rules are:

- `--config` is a global optional argument on the top-level parser
- `chat` and `format-trace` are subcommands created with `argparse` subparsers
- `trace_path` is positional because it is the main required input of `format-trace`
- `--session-id` and `--output` are optional named arguments, each with a short alias

## ReAct Loop

For each user query inside a session, the runtime performs the following loop:

1. Render the invariant system prompt and the current-turn user prompt.
2. Build a full message list containing the system prompt, sliding-window conversation history, the current user turn, and any prior assistant/tool messages for the current question.
3. Call the model with JSON mode enabled.
4. Parse the assistant JSON content into `thought`, `thought_summary`, `status`, `action`, `action_input`, and `final_answer`.
5. If the model requested a tool, execute it from the parsed JSON fields, append the assistant JSON response to the running scratchpad, and then append a user observation message containing the structured tool result for the next step.
6. If the model returned a final answer, stop the loop and return that answer.
7. If the model output is malformed or a tool/model call fails, record the failure, append a corrective user message, and let the loop recover.

## Conversation State Management

The agent is conversational rather than single-shot. Session state is stored in `sessions/<session-id>.json` and contains:

- session id
- invariant system prompt
- creation timestamp
- ordered list of user/assistant turns
- trace path for each turn

Context management uses a simple sliding window, as requested. Only the last `agent.sliding_window_turns` exchanges are injected into the prompt. This keeps the prompt size bounded and easy to reason about.

## Prompt Strategy

Prompt definitions are stored in `src/prompts.py`.

This keeps the protocol close to the rest of the runtime logic, which is a good fit now that the prompt shape is part of the fixed agent design rather than an external template surface.

The system prompt requires a strict JSON-based ReAct schema:

- `thought`
- `thought_summary`
- `action`
- `action_input`
- `status`
- `final_answer`

Actions are parsed from the assistant JSON content itself. This keeps the whole ReAct step in one response channel and avoids relying on provider-specific native tool-call metadata.

The inclusion of both `thought` and `thought_summary` matches your requirement to log both detailed reasoning text and a shorter summary.

## Tools

### Calculator

The calculator uses a strict AST-based evaluator rather than Python `eval`. It supports:

- arithmetic operators
- unary operators
- constants such as `pi`, `e`, and `tau`
- math functions such as `sqrt`, `sin`, `cos`, `tan`, `log`, `exp`, `abs`, and `round`

### Python

The Python tool runs snippets in a Docker container with:

- no network
- read-only filesystem
- no host mounts
- reduced privileges
- configurable CPU, memory, process, and timeout limits

Inside the container, common file-opening paths are patched to raise `PermissionError` so the tool aligns with the requirement that it should not read or write files. This is a pragmatic local-control measure, not a perfect security guarantee.

## Trace Design

Each session produces one JSONL trace file in `traces/`, and every user turn appends to that file. Important events include:

- turn start
- session start with the invariant system prompt
- raw model response
- parsed decision
- tool execution result
- tool or corrective observation appended to the running turn state
- repeated failure counter updates
- turn finish and stop reason

This makes the trace suitable for:

- debugging agent behavior
- replaying runs
- studying prompt effects
- collecting examples for later evaluation

The repository also includes a formatter that converts a raw JSONL trace into a readable report.

## Retry And Stop Logic

The loop stops when one of the following occurs:

- the model emits `Final Answer`
- repeated recoverable failures exceed `agent.max_repeated_failures`
- steps exceed `agent.max_steps_per_query`

Recoverable failures include:

- malformed model output
- malformed assistant JSON content
- LLM call errors
- tool errors
- unknown tool names

The repeated-failure counter resets after a successful tool execution.

## Configuration

All adjustable hyperparameters are stored in `config.yaml`. This includes:

- model settings
- session/trace paths
- sliding-window length
- maximum steps
- repeated-failure threshold
- Docker tool limits

The DeepSeek API key is kept outside the main config in `keys.cfg`, which is loaded as a small YAML mapping.

## Why This Design Fits The Requirements

- It is local and CLI-only.
- It is explicitly conversational.
- It uses a classic ReAct loop.
- It exposes exactly the requested tools.
- It uses DeepSeek through an OpenAI-compatible wrapper.
- It records detailed traces in JSONL.
- It keeps prompts and hyperparameters editable on disk.
- It prioritizes code readability through docstrings and clear module boundaries.
