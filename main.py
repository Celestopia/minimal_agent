"""Repository entrypoint for direct local execution.

This launcher lets the project run with plain Python commands such as:

    python main.py chat
    python main.py ask --session-id demo "Write a Fibonacci function."

It avoids requiring `pip install -e .` by adding the local `src/` directory to
`sys.path` before importing the flat source modules.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_src_path() -> None:
    """Ensure the local `src/` directory is importable."""

    root_dir = Path(__file__).resolve().parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> int:
    """Run the project CLI through the local source tree."""

    _bootstrap_src_path()

    from cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
