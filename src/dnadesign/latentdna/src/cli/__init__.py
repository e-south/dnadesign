"""
Internal CLI for latentdna.
"""

from .app import app


def main() -> None:
    app()


__all__ = ["app", "main"]
