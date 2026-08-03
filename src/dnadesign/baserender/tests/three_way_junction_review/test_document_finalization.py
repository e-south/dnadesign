"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/three_way_junction_review/test_document_finalization.py

Complete-document validation for three-way-junction review evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import dnadesign.baserender as baserender
from dnadesign.junction.contracts import parse_request
from dnadesign.junction.design.planner import design_junction
from dnadesign.junction.presentation.review_contract import review_contracts
from dnadesign.junction.sequence import reverse_complement
from dnadesign.junction.tests.scenarios.factories import deterministic_dna, scale_request_mapping

from .fixtures import _payload, _payload_with_long_recovery_primers, _rename_target_geometry, _review_job


def _distinct_target_rows() -> tuple[dict[str, object], dict[str, object]]:
    first = _payload()
    second = _payload_with_long_recovery_primers()
    _rename_target_geometry(second, target_id="target-02")
    return first, second


def _real_universal_rows_with_unequal_target_lengths() -> list[dict[str, object]]:
    mapping = scale_request_mapping(
        target_count=2,
        target_length=240,
        topology="shared",
        nominal_fragment_oligo_length=96,
        search_range=2,
        barcode_generation_attempts=250_000,
    )
    prefix = "ACGTACGTACGTACGTACGT"
    suffix = "GATTACAAGATTACAAGATT"
    for index, (target, interior_length) in enumerate(zip(mapping["targets"], (200, 180), strict=True)):
        sequence = prefix + deterministic_dna(f"universal-review-target-{index}", interior_length) + suffix
        target["sequence"] = sequence
        target["recovery_primers"] = {
            "mode": "universal",
            "forward": {
                "binding_sequence": prefix,
                "five_prime_extension": "GGTCTCA",
            },
            "reverse": {
                "binding_sequence": reverse_complement(suffix),
                "five_prime_extension": "CGTCTCA",
            },
        }
    reviews = review_contracts(design_junction(parse_request(mapping)))
    return [review.model_dump(mode="json") for review in reviews]


def _target_specific_rows_with_exact_group_receipt() -> tuple[dict[str, object], dict[str, object]]:
    first, second = _distinct_target_rows()
    for row in (first, second):
        row["recovery"]["mode"] = "target_specific"  # type: ignore[index]
        row["search"]["locus_count"] = 2  # type: ignore[index]
        row["search"]["barcode_candidates_generated"] = 50  # type: ignore[index]
    return first, second


def _replace_row_barcode(row: dict[str, object], barcode: str) -> None:
    target = row["target"]["sequence_5to3"]  # type: ignore[index]
    junction = row["geometry"]["junctions"][0]  # type: ignore[index]
    toehold_end = junction["toehold_span"]["end"]
    junction["barcode"] = barcode
    junction["barcode_complement"] = reverse_complement(barcode)
    row["strands"][0]["barcode_bearing_sequence_5to3"] = target[:toehold_end] + barcode  # type: ignore[index]
    row["strands"][1]["barcode_bearing_sequence_5to3"] = (  # type: ignore[index]
        reverse_complement(barcode) + target[toehold_end:]
    )


def test_adapt_record_rejects_a_document_scoped_adapter() -> None:
    with pytest.raises(
        baserender.SchemaError,
        match="adapt_record cannot use document-scoped adapter 'three_way_junction_review_v1'; use adapt_records",
    ):
        baserender.adapt_record(_payload(), adapter_kind="three_way_junction_review_v1")


@pytest.mark.parametrize("loader_name", ["load_record_from_parquet", "load_records_from_parquet"])
def test_filtered_parquet_helpers_reject_a_document_scoped_adapter(
    loader_name: str,
    tmp_path: Path,
) -> None:
    loader = getattr(baserender, loader_name)
    kwargs: dict[str, object] = {
        "dataset_path": tmp_path / "source-does-not-need-to-exist.parquet",
        "adapter_kind": "three_way_junction_review_v1",
        "adapter_columns": {},
    }
    if loader_name == "load_record_from_parquet":
        kwargs["record_id"] = "target-01"
    else:
        kwargs["record_ids"] = ["target-01"]

    with pytest.raises(
        baserender.SchemaError,
        match="cannot use document-scoped adapter 'three_way_junction_review_v1'; use adapt_records",
    ):
        loader(**kwargs)


def test_adapter_rejects_a_group_junction_count_exceeding_locus_count() -> None:
    first, second = _distinct_target_rows()
    first["recovery"]["mode"] = "target_specific"  # type: ignore[index]
    second["recovery"]["mode"] = "target_specific"  # type: ignore[index]

    with pytest.raises(
        baserender.SchemaError,
        match="document junction total exceeds declared locus_count",
    ):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


def test_adapter_finalization_rejects_an_underfilled_group_locus_count() -> None:
    first, second = _distinct_target_rows()
    for row in (first, second):
        row["recovery"]["mode"] = "target_specific"  # type: ignore[index]
        row["search"]["locus_count"] = 3  # type: ignore[index]
        row["search"]["barcode_candidates_generated"] = 15  # type: ignore[index]

    with pytest.raises(
        baserender.SchemaError,
        match="document junction total does not match declared locus_count",
    ):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


def test_adapter_finalization_rejects_an_empty_document() -> None:
    with pytest.raises(
        baserender.SchemaError,
        match="document must contain at least one target",
    ):
        baserender.adapt_records([], adapter_kind="three_way_junction_review_v1")


def test_job_finalizes_before_publication(tmp_path: Path) -> None:
    first, second = _distinct_target_rows()
    first["recovery"]["mode"] = "target_specific"  # type: ignore[index]
    second["recovery"]["mode"] = "target_specific"  # type: ignore[index]
    source = tmp_path / "three_way_junction_review.v1.json"
    source.write_text(json.dumps([first, second]), encoding="utf-8")

    with pytest.raises(
        baserender.SchemaError,
        match="document junction total exceeds declared locus_count",
    ):
        baserender.run_job(
            _review_job(source, input_narrowing={"sample": {"mode": "first_n", "n": 1}}),
            caller_root=tmp_path,
        )

    assert not (tmp_path / "review-render").exists()


def test_adapter_rejects_duplicate_sequence_under_distinct_target_ids() -> None:
    first = _payload()
    second = _payload()
    second["target"]["target_id"] = "target-02"  # type: ignore[index]
    second["checks"][0]["subject"]["id"] = "target-02"  # type: ignore[index]

    with pytest.raises(
        baserender.SchemaError,
        match="document has a duplicate target sequence in one assembly group",
    ):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


def test_adapter_rejects_mixed_recovery_modes_within_one_assembly_group() -> None:
    first, second = _distinct_target_rows()
    second["recovery"]["mode"] = "target_specific"  # type: ignore[index]

    with pytest.raises(baserender.SchemaError, match="document mixes recovery modes within one assembly group"):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


def test_adapter_rejects_different_universal_primer_evidence_within_one_group() -> None:
    first, second = _distinct_target_rows()
    first["recovery"]["mode"] = "universal"  # type: ignore[index]
    second["recovery"]["mode"] = "universal"  # type: ignore[index]

    with pytest.raises(baserender.SchemaError, match="document has contradictory universal recovery primer evidence"):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


def test_adapter_rejects_target_specific_primers_that_resolve_another_group_target() -> None:
    rows = _real_universal_rows_with_unequal_target_lengths()
    for row in rows:
        row["recovery"]["mode"] = "target_specific"  # type: ignore[index]

    with pytest.raises(baserender.SchemaError, match="ambiguous target-specific recovery"):
        baserender.adapt_records(rows, adapter_kind="three_way_junction_review_v1")


def test_adapter_scopes_target_specific_recovery_to_one_assembly_group() -> None:
    first = _payload()
    second = _payload()
    _rename_target_geometry(second, target_id="target-02")
    for row in (first, second):
        row["recovery"]["mode"] = "target_specific"  # type: ignore[index]
    second["target"]["assembly_group_id"] = "assembly-02"  # type: ignore[index]
    second["search"]["assembly_group_id"] = "assembly-02"  # type: ignore[index]
    second["checks"][1]["subject"]["id"] = "assembly-02"  # type: ignore[index]

    records = baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")

    assert [record.id for record in records] == ["target-01", "target-02"]


def test_adapter_rejects_duplicate_plan_scoped_junction_identity() -> None:
    first, second = _target_specific_rows_with_exact_group_receipt()
    second["geometry"]["junctions"][0]["junction_id"] = "junction-01"  # type: ignore[index]
    second["strands"][0]["outgoing_junction_id"] = "junction-01"  # type: ignore[index]
    second["strands"][1]["incoming_junction_id"] = "junction-01"  # type: ignore[index]

    with pytest.raises(baserender.SchemaError, match="duplicate plan-scoped junction identity"):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


def test_adapter_rejects_duplicate_plan_scoped_fragment_identity() -> None:
    first, second = _target_specific_rows_with_exact_group_receipt()
    second["geometry"]["fragments"][0]["fragment_id"] = "fragment-01"  # type: ignore[index]
    second["geometry"]["junctions"][0]["left_fragment_id"] = "fragment-01"  # type: ignore[index]
    second["strands"][0]["fragment_id"] = "fragment-01"  # type: ignore[index]
    second["recovery"]["first_fragment_id"] = "fragment-01"  # type: ignore[index]

    with pytest.raises(baserender.SchemaError, match="duplicate plan-scoped fragment identity"):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


def test_adapter_rejects_reverse_complement_barcode_collision_within_group() -> None:
    first, second = _target_specific_rows_with_exact_group_receipt()
    _replace_row_barcode(first, "ACGTTGCA")
    first_barcode = first["geometry"]["junctions"][0]["barcode"]  # type: ignore[index]
    collision = reverse_complement(first_barcode)
    assert collision != first_barcode
    _replace_row_barcode(second, collision)

    with pytest.raises(
        baserender.SchemaError,
        match="duplicate barcode identity in one assembly group",
    ):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


def test_adapter_accepts_a_complete_coherent_multi_target_document() -> None:
    first, second = _target_specific_rows_with_exact_group_receipt()

    records = baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")

    assert [record.id for record in records] == ["target-01", "target-02"]


def test_adapter_accepts_a_complete_multi_target_document_from_the_real_producer() -> None:
    request = parse_request(
        scale_request_mapping(
            target_count=2,
            target_length=240,
            topology="shared",
            nominal_fragment_oligo_length=96,
            search_range=2,
            barcode_generation_attempts=250_000,
        )
    )
    reviews = review_contracts(design_junction(request))

    records = baserender.adapt_records(
        [review.model_dump(mode="json") for review in reviews],
        adapter_kind="three_way_junction_review_v1",
    )

    assert [record.id for record in records] == [review.target.target_id for review in reviews]


def test_adapter_accepts_real_universal_primers_across_unequal_target_lengths() -> None:
    records = baserender.adapt_records(
        _real_universal_rows_with_unequal_target_lengths(),
        adapter_kind="three_way_junction_review_v1",
    )

    assert [len(record.sequence) for record in records] == [240, 220]
