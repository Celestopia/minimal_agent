# Project Design

## Goal

Build a simple but readable local AI agent repository for studying agent behavior on small Python coding and debugging tasks. The design intentionally favors clarity and inspectability over hidden abstractions.

## High-Level Architecture

The system is composed of six main layers:

1. `main.py` and `src/cli.py`
   `main.py` is the direct repository launcher for `python main.py ...`, and `src/cli.py` holds the actual command parsing and runtime dispatch.

2. `memory.py`
   Persists conversations to `sessions/` and renders a sliding window of recent turns for prompt context.

3. `prompts.py`
   Loads prompt templates from the `prompts/` directory and injects runtime values.

4. `llm.py`
   Wraps the OpenAI Python SDK configured for DeepSeek's OpenAI-compatible endpoint.

5. `agent.py`
   Implements the explicit ReAct loop, retry and recovery behavior, stopping logic, trace logging, and session updates.

6. `src/tools/`
   Hosts the calculator tool, the Docker-backed Python execution tool, and the small registry/base modules that wire those tools into the agent.

## ReAct Loop

For each user query inside a session, the runtime performs the following loop:

1. Render the system prompt with tool descriptions and the recent conversation window.
2. Render the user prompt with the current question and scratchpad.
3. Call the model for exactly one next ReAct step.
4. Parse the model output into either:
   - an action request, or
   - a final answer
5. If the model requested a tool, execute it and append the resulting observation to the scratchpad.
6. If the model returned a final answer, stop the loop and return that answer.
7. If the model output is malformed or a tool/model call fails, record the failure, append a corrective observation, and let the loop recover.

## Conversation State Management

The agent is conversational rather than single-shot. Session state is stored in `sessions/<session-id>.json` and contains:

- session id
- creation timestamp
- ordered list of user/assistant turns
- trace path for each turn

Context management uses a simple sliding window, as requested. Only the last `agent.sliding_window_turns` exchanges are injected into the prompt. This keeps the prompt size bounded and easy to reason about.

## Prompt Strategy

Prompt templates are stored as standalone files:

- `prompts/system_prompt.txt`
- `prompts/user_turn_prompt.txt`

This keeps experimentation lightweight. You can change instructions, formatting rules, or tool guidance without editing the runtime.

The system prompt requires a strict explicit ReAct schema:

- `Thought`
- `Thought Summary`
- `Action`
- `Action Input`
- `Final Answer`

The inclusion of both `Thought` and `Thought Summary` matches your requirement to log both detailed reasoning text and a shorter summary.

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
- system prompt renders when the prompt changes
- per-step user prompt renders
- raw model response
- parsed decision
- tool execution result
- observation appended to scratchpad
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
- LLM call errors
- tool errors
- unknown tool names

The repeated-failure counter resets after a successful tool execution.

## Configuration

All adjustable hyperparameters are stored in `config.yaml`. This includes:

- model settings
- prompt/session/trace paths
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
