"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/core/selection_contracts.py

Shared fail-fast parsers for selection contract fields used by runtime, CLI,
and analysis views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

_OBJECTIVE_MODES = frozenset({"maximize", "minimize"})
_TIE_HANDLING_MODES = frozenset({"competition_rank", "dense_rank", "ordinal"})
RESERVED_SELECTION_PARAM_KEYS = frozenset(
    {
        "top_k",
        "tie_handling",
        "objective_mode",
        "score_ref",
        "uncertainty_ref",
        "exclude_already_labeled",
    }
)


def extract_selection_plugin_params(selection_params: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in selection_params.items() if k not in RESERVED_SELECTION_PARAM_KEYS}


def resolve_selection_objective_mode(
    selection_params: Mapping[str, Any],
    *,
    error_cls: type[Exception] = ValueError,
    field_prefix: str = "selection.params",
) -> str:
    key = "objective_mode"
    if key not in selection_params:
        raise error_cls(f"{field_prefix}.{key} is required.")
    raw = selection_params.get(key)
    if raw is None:
        raise error_cls(f"{field_prefix}.{key} cannot be null.")
    mode = str(raw).strip().lower()
    if mode not in _OBJECTIVE_MODES:
        raise error_cls(f"{field_prefix}.{key} must be maximize|minimize.")
    return mode


def resolve_selection_tie_handling(
    selection_params: Mapping[str, Any],
    *,
    error_cls: type[Exception] = ValueError,
    field_prefix: str = "selection.params",
) -> str:
    key = "tie_handling"
    if key not in selection_params:
        raise error_cls(f"{field_prefix}.{key} is required.")
    raw = selection_params.get(key)
    if raw is None:
        raise error_cls(f"{field_prefix}.{key} cannot be null.")
    tie = str(raw).strip().lower()
    if tie not in _TIE_HANDLING_MODES:
        raise error_cls(f"{field_prefix}.{key} must be one of competition_rank|dense_rank|ordinal.")
    return tie
