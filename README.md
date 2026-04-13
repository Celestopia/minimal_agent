# Minimal ReAct Agent

This repository is a local testing project for a conversational AI agent focused on small Python coding and debugging tasks. It uses a classic ReAct loop, DeepSeek through the OpenAI-compatible SDK, JSON-mode assistant outputs for both reasoning and tool decisions, a Docker-backed `python` tool, a safe `calculator` tool, sliding-window context management, and detailed JSONL trace logging.

## Features

- Conversational CLI agent with persistent session history
- Sliding-window context management across turns
- JSON-based ReAct protocol with full `thought`, `thought_summary`, `action`, `action_input`, and `final_answer`
- Docker-isolated Python execution with no network and no host file access
- Calculator tool with arithmetic and math functions such as `sqrt`, `sin`, and `log`
- Detailed per-session JSONL traces plus a formatter for human-readable replay
- YAML-based configuration for model settings and stopping thresholds
- Prompt definitions kept in `src/prompts.py`

## Repository Layout

- `config.yaml`: adjustable runtime configuration
- `keys.cfg`: DeepSeek API key file
- `src/`: implementation modules
- `tests/`: focused unit tests
- `traces/`: generated per-session JSONL traces
- `sessions/`: persisted conversation memory
- `PROJECT.md`: detailed design notes

## Setup

1. Create a virtual environment if you want one.
2. Install the runtime dependencies into your current Python environment:

```bash
pip install openai PyYAML
```

3. Make sure `keys.cfg` contains your DeepSeek key in YAML form:

```yaml
DEEPSEEK_API_KEY: your_key_here
```

4. Make sure Docker is installed and the daemon is running.
5. Optionally pre-pull the configured Python image:

```bash
docker pull python:3.11-slim
```

## Docker Requirements For The Python Tool

The Python tool depends on local Docker availability. The current implementation runs containers with:

- `--network none`
- `--read-only`
- `--cap-drop ALL`
- `--security-opt no-new-privileges`
- CPU, memory, and process limits from `config.yaml`
- no host volume mounts

This is a good local testing sandbox, but it is not a formally hardened security boundary. For higher-risk workloads, you would want a stronger isolation layer than plain Docker.

## Usage

The CLI uses `argparse` subcommands. The current command shape is:

```bash
python main.py [--config CONFIG] {chat,format-trace} ...
```

Command-specific argument rules:

- `chat` starts the interactive agent session
- `chat` accepts optional `--session-id` or `-S`
- `format-trace` requires one positional argument, `trace_path`
- `format-trace` accepts optional `--output` or `-O`

Start an interactive chat session:

```bash
python main.py chat
```

Resume or create a named session:

```bash
python main.py chat --session-id study-session
python main.py chat -S study-session
```

Render a JSONL trace into readable text:

```bash
python main.py format-trace traces/<trace-file>.jsonl
```

Write the formatted trace to a file:

```bash
python main.py format-trace traces/<trace-file>.jsonl --output trace_report.txt
python main.py format-trace traces/<trace-file>.jsonl -O trace_report.txt
```

## Configuration

All adjustable settings live in `config.yaml`. The most important ones are:

- `llm.model`
- `llm.temperature`
- `agent.max_steps_per_query`
- `agent.max_repeated_failures`
- `agent.sliding_window_turns`
- `tools.python.docker_image`
- `tools.python.timeout_seconds`
- `tools.python.memory_limit`
- `tools.python.cpu_limit`
- `tools.python.pids_limit`

## Testing

Run the local unit tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

The included tests focus on parsing, config loading, sliding-window memory, calculator behavior, and trace formatting. They do not require a live DeepSeek call or a running Docker daemon.

## Notes

- The final user answer is printed in human-readable form.
- Each session appends to one JSONL trace file in `traces/<session-id>.jsonl`.
- Session state is stored in `sessions/<session-id>.json`.
- The session record stores the invariant system prompt used for that session.
- Prompt definitions live in `src/prompts.py`.
- The main runtime entrypoint is `python main.py`.
