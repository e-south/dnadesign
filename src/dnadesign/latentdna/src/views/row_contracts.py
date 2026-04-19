"""
Shared row-column contracts for source-backed latentdna views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..contracts.errors import ContractViolationError
from ..contracts.workspace import PromoterMetadataCohortConfig
from ..workspaces.loader import WorkspaceContext

_PROMOTER_METADATA_INPUT_COLUMNS = (
    "densegen__plan",
    "densegen__required_regulators",
    "densegen__used_tfbs_detail",
    "usr_label__primary",
    "template_id",
    "construct__template_id",
)


def _unique_nonempty(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def reference_set_metadata_columns(context: WorkspaceContext) -> list[str]:
    columns: list[str] = []
    for reference_set in context.config.reference_sets.values():
        columns.append(reference_set.match_column)
        if reference_set.label_column:
            columns.append(reference_set.label_column)
    return _unique_nonempty(columns)


def requested_source_metadata_columns(context: WorkspaceContext, *, source) -> list[str]:
    return _unique_nonempty([*(context.config.metadata.include or []), *(source.metadata_include or [])])


def promoter_metadata_cohort_ids(context: WorkspaceContext, *, source_id: str | None = None) -> list[str]:
    return [
        cohort_id
        for cohort_id, cohort in context.config.cohorts.items()
        if isinstance(cohort, PromoterMetadataCohortConfig) and (source_id is None or cohort.source == source_id)
    ]


@dataclass(frozen=True, slots=True)
class SourceBackedViewRowContract:
    requested_metadata_columns: list[str]
    reference_set_metadata_columns: list[str]
    promoter_cohort_ids: list[str]
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
    promoter_cohort_ids = promoter_metadata_cohort_ids(context)
    auto_promoter_cohort_ids = promoter_metadata_cohort_ids(context, source_id=source_id)
    configured_derivation_ids = set((context.config.metadata.derivations or {}).keys())

    processing_row_columns = [
        column
        for column in _unique_nonempty(
            [
                source.record_key,
                source.subject_key,
                *requested_metadata,
                *reference_columns,
                *_PROMOTER_METADATA_INPUT_COLUMNS,
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

    derived_row_columns = [cohort_id for cohort_id in auto_promoter_cohort_ids if cohort_id not in output_row_columns]
    promoter_cohort_id_set = set(promoter_cohort_ids)
    for column in requested_metadata:
        if column in output_row_columns or column in derived_row_columns:
            continue
        if column == "construct_template_id" or column in configured_derivation_ids or column in promoter_cohort_id_set:
            derived_row_columns.append(column)
            continue
        raise ContractViolationError(
            f"metadata column {column!r} cannot be materialized for source {source_id}: "
            "it is neither present in the source nor backed by a derivation/cohort"
        )

    return SourceBackedViewRowContract(
        requested_metadata_columns=requested_metadata,
        reference_set_metadata_columns=reference_columns,
        promoter_cohort_ids=promoter_cohort_ids,
        processing_row_columns=processing_row_columns,
        output_row_columns=output_row_columns,
        derived_row_columns=derived_row_columns,
    )
