"""Repository entrypoint for direct local execution."""

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
    sys.exit(main())
