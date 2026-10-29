"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/foldcheck/models.py

Typed models for generic fold-check artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoldCheckIssue:
    """A generic fold-check validation issue."""

    check_id: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class FoldCheckSequenceRecord:
    """One sequence sent to a fold-check runtime."""

    sequence_id: str
    sequence: str
    sequence_hash: str
    source_kind: str
