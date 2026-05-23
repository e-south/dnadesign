"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/products/sequence_views.py

Sequence-view builders for Construct emitted products.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.usr import SequenceViewRecord

from ..persistence.records import BuiltRecord


def build_normalize_sequence_view(
    *,
    record: BuiltRecord,
    output_dataset_id: str,
    parent_row: dict[str, object],
    source_start_0: int,
    source_end_0: int,
    anchor_start_0: int,
    anchor_end_0: int,
    recommended_pooling: str | None,
) -> SequenceViewRecord:
    if record.derived_metadata is None:
        raise ValueError("normalize sequence-view records require derived metadata.")
    return SequenceViewRecord(
        sequence_id=record.output_id,
        view_name=record.label_primary,
        aliases=list(record.label_aliases),
        product_kind=str(record.derived_metadata["derived__product_kind"]),
        context_kind="analysis_window",
        orientation="forward",
        analysis_only=bool(record.derived_metadata["derived__analysis_only"]),
        source_dataset_id=output_dataset_id,
        source_label=record.label_primary,
        parent_sequence_id=str(parent_row["id"]),
        parent_dataset_id=str(record.derived_metadata["derived__parent_dataset"]),
        derivation_id=f"{record.output_id}:{record.metadata['construct__spec_id']}",
        derivation_spec_id=str(record.derived_metadata["derived__spec_id"]),
        template_sequence_id=None,
        template_dataset_id=(
            str(record.derived_metadata["derived__template_dataset"])
            if record.derived_metadata["derived__template_dataset"]
            else None
        ),
        source_interval_start_0=source_start_0,
        source_interval_end_0=source_end_0,
        anchor_start_0=anchor_start_0,
        anchor_end_0=anchor_end_0,
        forward_anchor_start_0=anchor_start_0,
        forward_anchor_end_0=anchor_end_0,
        recommended_pooling=recommended_pooling,
        created_at=record.created_at,
        created_by="construct",
    )


def build_variant_sequence_view(
    *,
    record: BuiltRecord,
    output_dataset_id: str,
    recommended_pooling: str | None,
) -> SequenceViewRecord:
    orientation = str(record.metadata["construct__orientation"])
    return SequenceViewRecord(
        sequence_id=record.output_id,
        view_name=record.label_primary,
        aliases=list(record.label_aliases),
        product_kind="realized_context",
        context_kind="template_1kb",
        orientation="forward" if orientation == "forward" else "reverse_complement",
        analysis_only=False,
        source_dataset_id=output_dataset_id,
        source_label=record.label_primary,
        parent_sequence_id=str(record.metadata.get("construct__input_id") or ""),
        parent_dataset_id=str(record.metadata.get("construct__input_dataset") or ""),
        derivation_id=f"{record.output_id}:{record.metadata['construct__spec_id']}",
        derivation_spec_id=str(record.metadata["construct__spec_id"]),
        template_sequence_id=None,
        template_dataset_id=(
            str(record.metadata.get("construct__template_dataset") or "")
            if record.metadata.get("construct__template_dataset")
            else None
        ),
        source_interval_start_0=None,
        source_interval_end_0=None,
        anchor_start_0=int(record.metadata["construct__anchor_start"]),
        anchor_end_0=int(record.metadata["construct__anchor_end"]),
        forward_anchor_start_0=int(record.metadata["construct__forward_anchor_start"]),
        forward_anchor_end_0=int(record.metadata["construct__forward_anchor_end"]),
        recommended_pooling=recommended_pooling,
        created_at=record.created_at,
        created_by="construct",
    )


def append_variant_label_suffix(value: str | None, suffix: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(f"_{suffix}"):
        return text
    return f"{text}_{suffix}"
