from __future__ import annotations

from textwrap import dedent


def block(source: str) -> str:
    """Normalize generated marimo source fragments."""

    return dedent(source).strip("\n")


__all__ = ["block"]
