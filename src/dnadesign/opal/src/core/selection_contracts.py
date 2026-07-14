"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/core/selection_contracts.py

Shared fail-fast parsers for selection contract fields used by runtime, CLI,.

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
        "require_exact_top_k",
    }
)


def extract_selection_plugin_params(selection_params: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in selection_params.items() if k not in RESERVED_SELECTION_PARAM_KEYS}


def require_exact_selection_count(
    selection_params: Mapping[str, Any],
    *,
    view_id: str,
    top_k: int,
    selected_count: int,
    tie_handling: str,
    error_cls: type[Exception] = ValueError,
) -> None:
    if not bool(selection_params.get("require_exact_top_k", False)) or selected_count == top_k:
        return
    plural = "s" if top_k != 1 else ""
    raise error_cls(
        f"Selection view {view_id!r} requires exactly {top_k} selected candidate{plural}, "
        f"but {tie_handling} selected {selected_count}. "
        "Resolve the boundary tie or choose an explicit tie policy before running the round."
    )


def resolve_selection_top_k(
    selection_params: Mapping[str, Any],
    *,
    view_id: str,
    override: int | None,
    error_cls: type[Exception] = ValueError,
) -> int:
    if override is None and "top_k" not in selection_params:
        raise error_cls(f"selection_views[{view_id}].selection.params.top_k is required.")
    top_k = int(override if override is not None else selection_params["top_k"])
    if top_k <= 0:
        raise error_cls(f"selection_views[{view_id}].selection.params.top_k must be > 0.")
    return top_k


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
