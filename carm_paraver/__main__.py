"""Command-line entry point for CARM Paraver."""

from __future__ import annotations


def main() -> None:
    # Importing runs the Dash app setup and CLI parsing.
    from . import Paraver_CARM  # noqa: F401


if __name__ == "__main__":
    main()
