"""
Shared row-column contracts for source-backed latentdna views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..contracts.errors import ContractViolationError
from ..reference_sets import reference_set_required_columns
from ..workspaces.loader import WorkspaceContext


def _unique_nonempty(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def reference_set_metadata_columns(context: WorkspaceContext) -> list[str]:
    columns: list[str] = []
    for reference_set in context.config.reference_sets.values():
        columns.extend(reference_set_required_columns(reference_set))
    return _unique_nonempty(columns)


def requested_source_metadata_columns(context: WorkspaceContext, *, source) -> list[str]:
    if source.metadata_include_mode == "replace":
        return _unique_nonempty(source.metadata_include or [])
    return _unique_nonempty([*(context.config.metadata.include or []), *(source.metadata_include or [])])


def derivation_dependency_columns(
    context: WorkspaceContext,
    *,
    columns: Iterable[str],
    available_columns: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    derivations = context.config.metadata.derivations or {}
    available = set(available_columns) if available_columns is not None else None
    dependencies: list[str] = []
    missing: list[str] = []
    for column in columns:
        derivation = derivations.get(column)
        if derivation is None:
            continue
        if derivation.kind in {"copy", "regex_capture", "map_values"}:
            dependencies.append(derivation.source)
            if available is not None and derivation.source not in available:
                missing.append(derivation.source)
            continue
        if derivation.kind == "coalesce":
            if available is None:
                dependencies.extend(derivation.sources)
                continue
            present_sources = [source for source in derivation.sources if source in available]
            if present_sources:
                dependencies.extend(present_sources)
                continue
            missing.extend(derivation.sources)
        if derivation.kind == "lookup":
            dependencies.append(derivation.left_key)
            if available is not None and derivation.left_key not in available:
                missing.append(derivation.left_key)
            continue
        if derivation.kind == "annotation":
            dependencies.extend(derivation.required_columns)
            if available is not None:
                missing.extend(column for column in derivation.required_columns if column not in available)
                found_any_group = False
                for group in derivation.any_required_column_groups:
                    if set(group).issubset(available):
                        dependencies.extend(group)
                        found_any_group = True
                if derivation.any_required_column_groups and not found_any_group:
                    for group in derivation.any_required_column_groups:
                        missing.extend(group)
                continue
            for group in derivation.any_required_column_groups:
                dependencies.extend(group)
    return _unique_nonempty(dependencies), _unique_nonempty(missing)


@dataclass(frozen=True, slots=True)
class SourceBackedViewRowContract:
    requested_metadata_columns: list[str]
    reference_set_metadata_columns: list[str]
    processing_row_columns: list[str]
    output_row_columns: list[str]
    derived_row_columns: list[str]

    @property
    def materialized_row_columns(self) -> list[str]:
        return [*self.output_row_columns, *self.derived_row_columns]


def source_backed_view_row_contract(
    context: WorkspaceContext,
    *,
    source_id: str,
    source,
    available_columns: Iterable[str],
) -> SourceBackedViewRowContract:
    available = set(available_columns)
    requested_metadata = requested_source_metadata_columns(context, source=source)
    reference_columns = reference_set_metadata_columns(context)
    configured_derivation_ids = set((context.config.metadata.derivations or {}).keys())
    metadata_dependency_columns, missing_dependency_columns = derivation_dependency_columns(
        context,
        columns=requested_metadata,
        available_columns=available,
    )
    if missing_dependency_columns:
        raise ContractViolationError(
            f"metadata derivation inputs are missing from source {source_id}: {missing_dependency_columns}"
        )

    processing_row_columns = [
        column
        for column in _unique_nonempty(
            [
                source.record_key,
                source.subject_key,
                *metadata_dependency_columns,
                *requested_metadata,
                *reference_columns,
                source.context_key,
            ]
        )
        if column in available
    ]
    output_row_columns = [
        column
        for column in _unique_nonempty(
            [
                source.record_key,
                source.subject_key,
                *requested_metadata,
                *reference_columns,
                source.context_key,
            ]
        )
        if column in available and column not in {"template_id", "construct__template_id"}
    ]

    derived_row_columns: list[str] = []
    for column in requested_metadata:
        if column in output_row_columns or column in derived_row_columns:
            continue
        if column in configured_derivation_ids:
            derived_row_columns.append(column)
            continue
        raise ContractViolationError(
            f"metadata column {column!r} cannot be materialized for source {source_id}: "
            "it is neither present in the source nor backed by a metadata derivation"
        )

    return SourceBackedViewRowContract(
        requested_metadata_columns=requested_metadata,
        reference_set_metadata_columns=reference_columns,
        processing_row_columns=processing_row_columns,
        output_row_columns=output_row_columns,
        derived_row_columns=derived_row_columns,
    )
