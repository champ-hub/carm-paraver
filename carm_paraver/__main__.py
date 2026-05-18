"""Command-line entry point for CARM Paraver."""

from __future__ import annotations


def main() -> None:
    # Importing runs the Dash app setup and CLI parsing.
    from .Paraver_CARM import run_server

    run_server()


if __name__ == "__main__":
    main()
