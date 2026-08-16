"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/layered_scatter_rounds.py

Resolve exact round manifests for layered-scatter selection overlays.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_layered_scatter_selection_rounds(
    choice: Mapping[str, Any],
    *,
    active_manifest: Mapping[str, Any],
) -> tuple[int, dict[int, Mapping[str, Any]]]:
    """Resolve one unambiguous manifest option for every exact campaign round."""

    exact_scopes: dict[str, int] = {}
    round_options: dict[int, Mapping[str, Any]] = {}
    options = choice.get("scope_options")
    scope_options = options if isinstance(options, (list, tuple)) else [choice]
    for option in scope_options:
        if not isinstance(option, Mapping):
            continue
        option_manifest = _mapping(option.get("manifest"))
        round_k = _single_selection_round(option_manifest.get("rounds"))
        if round_k is None:
            continue
        previous_option = round_options.setdefault(round_k, option)
        if previous_option is not option:
            previous_manifest = _mapping(previous_option.get("manifest"))
            if previous_manifest.get("run_id") != option_manifest.get("run_id"):
                raise ValueError(f"Layered-scatter selection round {round_k} has multiple run manifests.")
        run_id = str(option_manifest.get("run_id") or option.get("run_id") or "").strip()
        if run_id:
            previous = exact_scopes.setdefault(run_id, round_k)
            if previous != round_k:
                raise ValueError(f"Layered-scatter run {run_id!r} is bound to multiple rounds.")

    active = _single_selection_round(active_manifest.get("rounds"))
    if active is None:
        run_id = str(active_manifest.get("run_id") or "").strip()
        active = exact_scopes.get(run_id)
    if active is None:
        raise ValueError("Layered-scatter selection overlays require one exact manifest-backed round.")
    round_options.setdefault(active, choice)
    return active, round_options


def _single_selection_round(value: object) -> int | None:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        return None
    raw = value[0]
    if isinstance(raw, bool):
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and str(raw).strip() == str(parsed) else None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["resolve_layered_scatter_selection_rounds"]
