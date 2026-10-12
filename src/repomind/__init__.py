"""RepoMind: versioned structural understanding of Python repositories."""

from .service import RepoMindService

__all__ = ["RepoMindService"]


def main() -> None:
    from .cli import main as cli_main

    raise SystemExit(cli_main())
