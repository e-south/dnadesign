"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_products.py

Unit contracts for Construct emitted-product builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.construct.src.contracts.config import OutputVariantConfig
from dnadesign.construct.src.persistence.records import BuiltRecord
from dnadesign.construct.src.products.classic import build_variant_record


def _forward_record() -> BuiltRecord:
    return BuiltRecord(
        output_id="forward-sequence-id",
        sequence="AACCGG",
        alphabet="dna_4",
        metadata={
            "id": "forward-sequence-id",
            "construct__spec_id": "spec_a",
            "construct__input_id": "row_a",
            "construct__input_dataset": "input_refs",
            "construct__template_dataset": "template_refs",
            "construct__anchor_start": 1,
            "construct__anchor_end": 3,
            "construct__forward_anchor_start": 1,
            "construct__forward_anchor_end": 3,
            "construct__orientation": "forward",
            "construct__slots": [
                {
                    "slot_id": "anchor",
                    "role": "anchor",
                    "sequence_source": "input_field",
                    "sequence_field": "sequence",
                    "start": 1,
                    "end": 3,
                    "forward_start": 1,
                    "forward_end": 3,
                }
            ],
        },
        label_primary="anchor_a",
        label_aliases=["anchor_alias"],
        created_at="2026-05-22T00:00:00+00:00",
    )


def test_reverse_complement_variant_transforms_product_lineage_and_view_bounds() -> None:
    variant = OutputVariantConfig.model_validate(
        {
            "product_kind": "realized_context",
            "orientation": "reverse_complement",
            "recommended_pooling": "anchor_mean",
        }
    )

    record = build_variant_record(
        forward_record=_forward_record(),
        variant=variant,
        output_dataset_id="construct_outputs",
    )

    assert record.sequence == "CCGGTT"
    assert record.metadata["construct__anchor_start"] == 3
    assert record.metadata["construct__anchor_end"] == 5
    assert record.metadata["construct__parent_forward_construct_id"] == "forward-sequence-id"
    assert record.metadata["construct__slots"][0]["start"] == 3
    assert record.metadata["construct__slots"][0]["end"] == 5
    assert record.metadata["construct__slots"][0]["forward_start"] == 1
    assert record.metadata["construct__slots"][0]["forward_end"] == 3
    assert record.label_primary == "anchor_a_realized_context_reverse_complement"
    assert record.label_aliases == ["anchor_alias_realized_context_reverse_complement"]
    assert record.sequence_view is not None
    assert record.sequence_view.orientation == "reverse_complement"
    assert record.sequence_view.anchor_start_0 == 3
    assert record.sequence_view.anchor_end_0 == 5
    assert record.sequence_view.forward_anchor_start_0 == 1
    assert record.sequence_view.forward_anchor_end_0 == 3
