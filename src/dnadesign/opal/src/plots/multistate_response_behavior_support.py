"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/multistate_response_behavior_support.py

Shared presentation contracts for Multistate Response Behavior plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._mpl_utils import pretty_label
from .multistate_response_behavior_data import MultistateResponseBehaviorPlotData


def target_context(data: MultistateResponseBehaviorPlotData, params: Mapping[str, Any]) -> str:
    """Describe the exact target mask with presentation-only state labels."""

    labels = state_display_labels(data.state_ids, params.get("state_labels"))
    on_labels = [labels[state] for state, enabled in zip(data.state_ids, data.target_mask, strict=True) if enabled]
    off_labels = [labels[state] for state, enabled in zip(data.state_ids, data.target_mask, strict=True) if not enabled]
    target_name = str(params.get("target_name") or "").strip()
    prefix = f"{target_name} target" if target_name else "Target"
    return f"{prefix} ON: {', '.join(on_labels)} | OFF: {', '.join(off_labels)}"


def state_display_labels(state_ids: Sequence[str], configured: object) -> dict[str, str]:
    """Validate a complete display-label projection without changing state identity."""

    states = tuple(str(value) for value in state_ids)
    if configured is None:
        return {state: state for state in states}
    if not isinstance(configured, Mapping):
        raise ValueError("state_labels must be a mapping from exact state IDs to display labels.")
    if any(not isinstance(key, str) for key in configured):
        raise ValueError("state_labels keys must be exact string state IDs.")
    labels = {key: str(value).strip() for key, value in configured.items()}
    missing = sorted(set(states) - set(labels))
    extra = sorted(set(labels) - set(states))
    if missing or extra:
        raise ValueError(f"state_labels must match state_ids exactly; missing={missing}, extra={extra}.")
    if any(not labels[state] for state in states):
        raise ValueError("state_labels values must be non-empty.")
    if len(set(labels.values())) != len(labels):
        raise ValueError("state_labels values must be unique so plotted states remain distinguishable.")
    return labels


def selection_view_title(value: object, *, context: Any) -> str:
    """Add one concise active-view label without repeating the word ``view``."""

    title = nonempty(value, field="title")
    view_id = str(getattr(context, "selection_view_id", "") or "").strip()
    if not view_id:
        return title
    label = " ".join(
        token.upper() if len(token) == 1 and token.isalpha() else token for token in pretty_label(view_id).split()
    )
    return f"{title} · {label}"


def save_figure(context: Any, figure: Any) -> None:
    """Write one white-background publication artifact through PlotContext."""

    context.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        context.output_dir / context.filename,
        dpi=context.dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.1,
    )


def figsize(value: object) -> tuple[float, float]:
    """Parse a positive two-dimensional figure size."""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("figsize_in must contain exactly two values.")
    size = tuple(float(item) for item in value)
    if not all(np.isfinite(size)) or min(size) <= 0.0:
        raise ValueError("figsize_in values must be finite and positive.")
    return size


def positive_float(value: object, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return parsed


def unit_float(value: object, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return parsed


def nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    parsed = int(value)
    if float(value) != parsed or parsed < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return parsed


def nonempty(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty.")
    return text


__all__ = [
    "figsize",
    "nonempty",
    "nonnegative_int",
    "positive_float",
    "save_figure",
    "selection_view_title",
    "state_display_labels",
    "target_context",
    "unit_float",
]
