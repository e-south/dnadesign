"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/cli.py

Top-level infer CLI entrypoint surface that forwards to the internal CLI module.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .src.cli import app


def main() -> None:
    app()


__all__ = ["app", "main"]
