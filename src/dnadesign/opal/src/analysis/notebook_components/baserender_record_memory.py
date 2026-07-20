"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_record_memory.py

Scope and preference primitives for BaseRender notebook record controls.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .baserender_records import select_notebook_baserender_default_record_id


def build_notebook_baserender_record_memory_key(
    *,
    campaign_slug: Any,
    run_id: Any,
    round_value: Any,
    selection_view_id: Any,
    review_group_key: Any,
    deliverable_key: Any,
) -> str:
    """Build a stable key for one campaign/run/round/view/deliverable preference."""

    scope = {
        "campaign_slug": str(campaign_slug or "").strip(),
        "run_id": str(run_id or "").strip(),
        "round": _normalise_round(round_value),
        "selection_view_id": str(selection_view_id or "").strip(),
        "review_group_key": str(review_group_key or "").strip(),
        "deliverable_key": str(deliverable_key or "").strip(),
    }
    missing = [
        key
        for key in ("campaign_slug", "run_id", "selection_view_id", "review_group_key", "deliverable_key")
        if not scope[key]
    ]
    if missing:
        raise ValueError(f"BaseRender record memory scope is missing: {', '.join(missing)}.")
    return f"baserender_record_v1:{json.dumps(scope, sort_keys=True, separators=(',', ':'))}"


def resolve_notebook_baserender_preferred_record_id(
    record_options: Sequence[Any],
    annotation_counts: Mapping[str, int] | None,
    *,
    preferred_record_id: Any | None,
) -> str:
    """Use a remembered record only when it belongs to the current view."""

    options = [str(value) for value in record_options]
    preferred = str(preferred_record_id or "").strip()
    if preferred and preferred in options:
        return preferred
    return select_notebook_baserender_default_record_id(options, annotation_counts)


def _normalise_round(value: Any) -> int:
    try:
        round_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("BaseRender record memory scope requires an integer round.") from exc
    if isinstance(value, bool) or round_value < 0 or float(value) != float(round_value):
        raise ValueError(f"BaseRender record memory scope has invalid round {value!r}.")
    return round_value


__all__ = [
    "build_notebook_baserender_record_memory_key",
    "resolve_notebook_baserender_preferred_record_id",
]
