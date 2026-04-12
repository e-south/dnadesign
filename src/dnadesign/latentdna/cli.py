"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/cli.py

Top-level latentdna CLI entrypoint surface that forwards to the internal CLI.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .src.cli import app


def main() -> None:
    app()


__all__ = ["app", "main"]
