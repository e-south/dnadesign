"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/core/pydantic_validation.py

Safe formatting for Pydantic validation failures at BaseRender boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pydantic import ValidationError


def format_validation_error(error: ValidationError) -> str:
    """Describe validation locations and kinds without retaining raw input values."""

    descriptions: list[str] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in item["loc"]) or "contract"
        descriptions.append(f"{location}: validation failed ({item['type']})")
    return "; ".join(descriptions) or "contract validation failed"


__all__ = ["format_validation_error"]
