#!/usr/bin/env python3
"""Compatibility wrapper for running the CARM app from a source checkout."""


def main() -> None:
    from carm_paraver.__main__ import main as _main

    _main()


if __name__ == "__main__":
    main()
