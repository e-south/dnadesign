"""
Public CLI entrypoint for latentdna.
"""

from .src.cli import app


def main() -> None:
    app()


__all__ = ["app", "main"]
