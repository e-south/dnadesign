"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/metadata/join_keys.py

Shared metadata join-key contract for LatentDNA notebook overlays.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable

JOINABLE_KEY_PAIRS: tuple[tuple[str, str], ...] = (
    ("alias_id", "alias_id"),
    ("construct__anchor_id", "construct__anchor_id"),
    ("construct__anchor_id", "id"),
    # Anchor-only projection rows should join context summary tables by anchor id.
    ("id", "construct__anchor_id"),
    ("id", "id"),
    ("alignment_parent_sequence_id", "alignment_parent_sequence_id"),
    ("subject_id", "subject_id"),
    ("context_id", "context_id"),
)

JOINABLE_KEY_COLUMNS = frozenset({column for pair in JOINABLE_KEY_PAIRS for column in pair})


def candidate_join_key_pairs_for_columns(
    left_columns: Iterable[str],
    right_columns: Iterable[str],
) -> list[tuple[str, str]]:
    """Return supported join pairs present in both column collections."""

    left = {str(column) for column in left_columns}
    right = {str(column) for column in right_columns}
    return [
        (left_key, right_key) for left_key, right_key in JOINABLE_KEY_PAIRS if left_key in left and right_key in right
    ]


__all__ = [
    "JOINABLE_KEY_COLUMNS",
    "JOINABLE_KEY_PAIRS",
    "candidate_join_key_pairs_for_columns",
]
