"""Docker-backed Python execution tool."""

from __future__ import annotations

import shutil
import subprocess

from tools.base import ToolResult


def _strip_code_fence(text: str) -> str:
    """Remove one surrounding Markdown code fence if present."""

    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


class DockerPythonExecutor:
    """Runs Python snippets in a locked-down Docker container.

    This is intentionally conservative:
    - no network
    - read-only filesystem
    - no host mounts
    - no file access inside the executed script
    - low CPU, memory, and process limits

    The design is suitable for local testing, but it is not a formally verified
    sandbox. The README and project notes make that boundary explicit.
    """

    def __init__(
        self,
        docker_image: str,
        timeout_seconds: int,
        memory_limit: str,
        cpu_limit: float,
        pids_limit: int,
    ) -> None:
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit

    def _command(self) -> list[str]:
        """Build the Docker command used for isolated execution."""

        wrapper = (
            "import builtins, os, pathlib, sys\n"
            "def _blocked(*args, **kwargs):\n"
            "    raise PermissionError('File access is disabled in the python tool.')\n"
            "builtins.open = _blocked\n"
            "os.open = _blocked\n"
            "pathlib.Path.open = lambda self, *args, **kwargs: _blocked()\n"
            "namespace = {'__name__': '__main__'}\n"
            "code = sys.stdin.read()\n"
            "exec(compile(code, '<agent_python>', 'exec'), namespace, namespace)\n"
        )

        return [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory_limit,
            "--cpus",
            str(self.cpu_limit),
            "--user",
            "65534:65534",
            "--workdir",
            "/",
            self.docker_image,
            "python",
            "-I",
            "-c",
            wrapper,
        ]

    def run(self, code: str) -> ToolResult:
        """Execute one Python snippet and capture stdout, stderr, and exit status."""

        if shutil.which("docker") is None:
            return ToolResult(
                tool_name="python",
                tool_input=code,
                success=False,
                output_text="",
                error_text="Docker is not installed or not available on PATH.",
            )

        normalized_code = _strip_code_fence(code)

        try:
            completed = subprocess.run(
                self._command(),
                input=normalized_code,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name="python",
                tool_input=normalized_code,
                success=False,
                output_text="",
                error_text=(
                    f"Execution timed out after {self.timeout_seconds} seconds."
                ),
            )
        except Exception as exc:
            return ToolResult(
                tool_name="python",
                tool_input=normalized_code,
                success=False,
                output_text="",
                error_text=f"{type(exc).__name__}: {exc}",
            )

        output_text = completed.stdout.strip()
        error_text = completed.stderr.strip()
        success = completed.returncode == 0

        if not output_text:
            output_text = "(no stdout)"
        if not error_text:
            error_text = ""

        return ToolResult(
            tool_name="python",
            tool_input=normalized_code,
            success=success,
            output_text=output_text,
            error_text=error_text,
            metadata={"exit_code": completed.returncode},
        )
