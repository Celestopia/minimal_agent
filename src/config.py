"""Configuration loading for the local ReAct agent project.

The repository keeps all adjustable runtime values in a YAML file so the
agent can be tuned without changing code. This module translates that YAML
document into typed dataclasses and resolves all repository-relative paths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised implicitly in this environment.
    yaml = None


@dataclass(slots=True)
class LLMConfig:
    """Settings for the OpenAI-compatible DeepSeek backend."""

    api_key_file: Path
    api_key_field: str
    base_url: str
    model: str
    temperature: float
    timeout_seconds: int


@dataclass(slots=True)
class AgentRuntimeConfig:
    """Controls the ReAct loop and memory strategy."""

    max_steps_per_query: int
    max_repeated_failures: int
    sliding_window_turns: int


@dataclass(slots=True)
class PathConfig:
    """Filesystem locations used by the repository."""

    root_dir: Path
    prompt_dir: Path
    trace_dir: Path
    session_dir: Path


@dataclass(slots=True)
class CalculatorToolConfig:
    """Configuration for the calculator tool."""

    enabled: bool


@dataclass(slots=True)
class PythonToolConfig:
    """Configuration for the Docker-backed Python tool."""

    enabled: bool
    docker_image: str
    timeout_seconds: int
    memory_limit: str
    cpu_limit: float
    pids_limit: int


@dataclass(slots=True)
class ToolConfig:
    """Combined tool configuration section."""

    calculator: CalculatorToolConfig
    python: PythonToolConfig


@dataclass(slots=True)
class AgentConfig:
    """Top-level project configuration."""

    llm: LLMConfig
    agent: AgentRuntimeConfig
    paths: PathConfig
    tools: ToolConfig

    def public_snapshot(self) -> dict[str, Any]:
        """Return a log-safe config snapshot without secrets."""

        snapshot = asdict(self)
        snapshot["llm"]["api_key_file"] = str(self.llm.api_key_file)
        snapshot["paths"]["root_dir"] = str(self.paths.root_dir)
        snapshot["paths"]["prompt_dir"] = str(self.paths.prompt_dir)
        snapshot["paths"]["trace_dir"] = str(self.paths.trace_dir)
        snapshot["paths"]["session_dir"] = str(self.paths.session_dir)
        return snapshot


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file and always return a mapping."""

    with path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()

    if yaml is not None:
        data = yaml.safe_load(raw_text) or {}
    else:
        data = _simple_yaml_load(raw_text)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return data


def _parse_scalar(value: str) -> Any:
    """Parse a scalar value for the lightweight YAML fallback loader."""

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def _simple_yaml_load(raw_text: str) -> dict[str, Any]:
    """Parse a tiny YAML subset used by this repository.

    This fallback intentionally supports only nested mappings with scalar values,
    which is enough for `config.yaml` and `keys.cfg`. If the project grows into
    more advanced YAML constructs, installing PyYAML remains the better option.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue

        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        content = line_without_comment.lstrip(" ")

        if ":" not in content:
            raise ValueError(
                f"Unsupported YAML syntax at {line_number}: {raw_line!r}"
            )

        key, raw_value = content.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(
                f"Could not determine YAML nesting at {line_number}: {raw_line!r}"
            )

        parent = stack[-1][1]
        if value == "":
            nested: dict[str, Any] = {}
            parent[key] = nested
            stack.append((indent, nested))
        else:
            parent[key] = _parse_scalar(value)

    return root


def load_agent_config(config_path: str | Path = "config.yaml") -> AgentConfig:
    """Load and normalize the repository configuration.

    Relative paths are resolved against the directory containing the YAML file,
    which keeps the repository portable even when the CLI is launched elsewhere.
    """

    config_path = Path(config_path).expanduser().resolve()
    raw = _read_yaml(config_path)
    root_dir = config_path.parent

    llm_raw = raw.get("llm", {})
    agent_raw = raw.get("agent", {})
    paths_raw = raw.get("paths", {})
    tools_raw = raw.get("tools", {})

    paths = PathConfig(
        root_dir=root_dir,
        prompt_dir=(root_dir / paths_raw.get("prompt_dir", "prompts")).resolve(),
        trace_dir=(root_dir / paths_raw.get("trace_dir", "traces")).resolve(),
        session_dir=(root_dir / paths_raw.get("session_dir", "sessions")).resolve(),
    )

    config = AgentConfig(
        llm=LLMConfig(
            api_key_file=(root_dir / llm_raw.get("api_key_file", "keys.cfg")).resolve(),
            api_key_field=llm_raw.get("api_key_field", "DEEPSEEK_API_KEY"),
            base_url=llm_raw.get("base_url", "https://api.deepseek.com"),
            model=llm_raw.get("model", "deepseek-chat"),
            temperature=float(llm_raw.get("temperature", 0.0)),
            timeout_seconds=int(llm_raw.get("timeout_seconds", 90)),
        ),
        agent=AgentRuntimeConfig(
            max_steps_per_query=int(agent_raw.get("max_steps_per_query", 10)),
            max_repeated_failures=int(agent_raw.get("max_repeated_failures", 5)),
            sliding_window_turns=int(agent_raw.get("sliding_window_turns", 6)),
        ),
        paths=paths,
        tools=ToolConfig(
            calculator=CalculatorToolConfig(
                enabled=bool(tools_raw.get("calculator", {}).get("enabled", True))
            ),
            python=PythonToolConfig(
                enabled=bool(tools_raw.get("python", {}).get("enabled", True)),
                docker_image=tools_raw.get("python", {}).get(
                    "docker_image", "python:3.11-slim"
                ),
                timeout_seconds=int(
                    tools_raw.get("python", {}).get("timeout_seconds", 15)
                ),
                memory_limit=str(
                    tools_raw.get("python", {}).get("memory_limit", "256m")
                ),
                cpu_limit=float(tools_raw.get("python", {}).get("cpu_limit", 1.0)),
                pids_limit=int(tools_raw.get("python", {}).get("pids_limit", 64)),
            ),
        ),
    )

    config.paths.trace_dir.mkdir(parents=True, exist_ok=True)
    config.paths.session_dir.mkdir(parents=True, exist_ok=True)

    return config


def load_api_key(llm_config: LLMConfig) -> str:
    """Load the DeepSeek API key from the configured key file."""

    key_data = _read_yaml(llm_config.api_key_file)
    api_key = key_data.get(llm_config.api_key_field)
    if not api_key or not isinstance(api_key, str):
        raise ValueError(
            f"Could not find '{llm_config.api_key_field}' in {llm_config.api_key_file}"
        )
    return api_key.strip()
