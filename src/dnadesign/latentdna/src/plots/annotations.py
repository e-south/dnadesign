"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/plots/annotations.py

Reference and label annotation contracts for static plot rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.errors import ContractViolationError
from ..contracts.plot import ResolvedPlotSpec
from ..presentation.visual_style import reference_annotation_label
from ..references.sets import resolve_reference_set_rows
from ..workspaces.loader import WorkspaceContext
from .tables import require_row_columns


@dataclass(frozen=True, slots=True)
class ResolvedAnnotationRows:
    """Rows and metadata needed to draw one static plot annotation overlay."""

    selected_rows: list[dict[str, object]]
    label_column: str | None
    expected_ids: list[str]
    state: dict[str, object]


def selected_label_rows(
    rows: list[dict[str, object]],
    *,
    label_column: str | None,
    label_values: list[str],
) -> list[dict[str, object]]:
    """Resolve explicit label overlays and fail fast on malformed row contracts."""

    if label_column is None or not label_values:
        return []
    require_row_columns(rows, [label_column], context="plot label annotations")
    selected = {str(value) for value in label_values}
    return [row for row in rows if str(row[label_column]) in selected]


def resolve_annotation_rows(
    context: WorkspaceContext,
    rows: list[dict[str, object]],
    *,
    spec: ResolvedPlotSpec,
) -> ResolvedAnnotationRows:
    """Resolve configured reference-set annotations for one static plot panel."""

    if spec.annotation is None:
        selected_rows = selected_label_rows(rows, label_column=spec.label_column, label_values=spec.label_values)
        return ResolvedAnnotationRows(
            selected_rows=selected_rows,
            label_column=spec.label_column,
            expected_ids=list(spec.label_values),
            state={
                "reference_set": None,
                "expected_ids": list(spec.label_values),
                "matched_ids": (
                    [str(row[spec.label_column]) for row in selected_rows] if spec.label_column is not None else []
                ),
                "complete": True,
            },
        )

    reference_set = context.config.reference_sets[spec.annotation.reference_set]
    match_column = reference_set.match_column
    label_column = reference_set.label_column or match_column
    resolution = resolve_reference_set_rows(reference_set, rows)
    expected_ids = resolution.expected_ids
    if resolution.missing_columns:
        return ResolvedAnnotationRows(
            selected_rows=[],
            label_column=None,
            expected_ids=expected_ids,
            state={
                "reference_set": spec.annotation.reference_set,
                "match_column": match_column,
                "label_column": label_column,
                "expected_ids": expected_ids,
                "matched_ids": [],
                "complete": False,
                "error": "missing_reference_columns",
                "missing_columns": resolution.missing_columns,
            },
        )
    missing_ids = [value for value in expected_ids if value not in resolution.matched_ids]
    if missing_ids and spec.annotation.missing_policy == "fail":
        raise ContractViolationError(
            f"reference_set {spec.annotation.reference_set!r} is missing required ids: {missing_ids}"
        )
    if not expected_ids and spec.annotation.missing_policy == "fail" and reference_set.require_non_empty:
        raise ContractViolationError(f"reference_set {spec.annotation.reference_set!r} matched no rows")
    complete = not missing_ids and (bool(expected_ids) or not reference_set.require_non_empty)
    if spec.annotation.missing_policy == "allow" and resolution.matched_ids:
        complete = True
    return ResolvedAnnotationRows(
        selected_rows=resolution.selected_rows,
        label_column=label_column,
        expected_ids=expected_ids,
        state={
            "reference_set": spec.annotation.reference_set,
            "match_column": match_column,
            "label_column": label_column,
            "expected_ids": expected_ids,
            "matched_ids": resolution.matched_ids,
            "missing_ids": missing_ids,
            "complete": complete,
        },
    )


def empty_annotation_state(
    context: WorkspaceContext,
    *,
    spec: ResolvedPlotSpec,
    error: str,
) -> dict[str, object]:
    """Return explicit annotation metadata when no rows can be annotated."""

    if spec.annotation is None:
        expected_ids = list(spec.label_values)
        state: dict[str, object] = {
            "reference_set": None,
            "expected_ids": expected_ids,
            "matched_ids": [],
            "complete": not expected_ids,
        }
        if expected_ids:
            state["error"] = error
        return state

    reference_set = context.config.reference_sets[spec.annotation.reference_set]
    expected_ids = [str(value) for value in getattr(reference_set, "ids", [])]
    return {
        "reference_set": spec.annotation.reference_set,
        "match_column": reference_set.match_column,
        "label_column": reference_set.label_column or reference_set.match_column,
        "expected_ids": expected_ids,
        "matched_ids": [],
        "missing_ids": expected_ids,
        "complete": False,
        "error": error,
    }


def annotation_label_text(
    context: WorkspaceContext,
    *,
    spec: ResolvedPlotSpec,
    row: dict[str, object],
    resolved_label_column: str,
) -> str:
    """Return the display label for an already-resolved annotation row."""

    if spec.annotation is None:
        return reference_annotation_label(str(row[resolved_label_column]))
    reference_set = context.config.reference_sets[spec.annotation.reference_set]
    display_labels = dict(getattr(reference_set, "display_labels", {}) or {})
    match_column = reference_set.match_column
    match_value = str(row.get(match_column, ""))
    return reference_annotation_label(str(display_labels.get(match_value, row[resolved_label_column])))
