"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/cli.py

Top-level construct CLI entrypoint surface that forwards to the internal CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .src.cli import app


def main() -> None:
    app()


__all__ = ["app", "main"]
