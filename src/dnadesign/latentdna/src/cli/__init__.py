"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/cli/__init__.py

Internal CLI for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .app import app


def main() -> None:
    app()


__all__ = ["app", "main"]
