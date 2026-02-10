"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/kinds.py

Resolve and validate optimizer kind identifiers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

GIBBS_ANNEAL_KIND = "gibbs_anneal"


def resolve_optimizer_kind(kind: object | None, *, context: str) -> str:
    """
    Return the canonical optimizer kind for user-facing sampling/analysis paths.

    Missing values default to gibbs_anneal. Any non-string or unsupported kind
    raises a ValueError with context for easier debugging.
    """
    if kind is None:
        return GIBBS_ANNEAL_KIND
    if not isinstance(kind, str):
        raise ValueError(f"{context}: optimizer kind must be a non-empty string.")
    resolved = kind.strip().lower()
    if not resolved:
        raise ValueError(f"{context}: optimizer kind must be a non-empty string.")
    if resolved != GIBBS_ANNEAL_KIND:
        raise ValueError(f"{context}: optimizer kind '{kind}' is unsupported; only '{GIBBS_ANNEAL_KIND}' is allowed.")
    return resolved
