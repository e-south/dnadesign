"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/three_way_junction_review/test_document_coherence.py

Document-scoped evidence checks for three-way-junction review inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import dnadesign.baserender as baserender

from .fixtures import _payload, _payload_with_long_recovery_primers, _rename_target_geometry, _review_job


def _second_target() -> tuple[dict[str, object], dict[str, object]]:
    first = _payload()
    second = _payload_with_long_recovery_primers()
    first["recovery"]["mode"] = "target_specific"  # type: ignore[index]
    _rename_target_geometry(second, target_id="target-02")
    return first, second


@pytest.mark.parametrize(
    "input_narrowing",
    [None, {"limit": 1}, {"sample": {"mode": "first_n", "n": 1}}],
)
def test_job_validates_the_complete_review_document_before_narrowing(
    tmp_path: Path,
    input_narrowing: dict[str, object] | None,
) -> None:
    first, second = _second_target()
    second["search"]["toehold_seed"] = 99  # type: ignore[index]
    source = tmp_path / "three_way_junction_review.v1.json"
    source.write_text(json.dumps([first, second]), encoding="utf-8")

    with pytest.raises(
        baserender.SchemaError,
        match="contradictory assembly-group-wide search receipt at row 1",
    ):
        baserender.run_job(_review_job(source, input_narrowing=input_narrowing), caller_root=tmp_path)

    assert not (tmp_path / "review-render").exists()


@pytest.mark.parametrize(
    ("source_field", "replacement"),
    [
        ("algorithm", "different-algorithm"),
        ("request_sha256", f"sha256:{'d' * 64}"),
    ],
)
def test_adapter_rejects_inconsistent_metadata_for_one_plan_id(
    source_field: str,
    replacement: str,
) -> None:
    first, second = _second_target()
    second["source"][source_field] = replacement  # type: ignore[index]
    second["search"]["toehold_seed"] = 99  # type: ignore[index]

    with pytest.raises(baserender.SchemaError, match="contradictory source metadata at row 1"):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


@pytest.mark.parametrize("drift", ["changed_detail", "changed_status", "omitted"])
def test_adapter_rejects_inconsistent_assembly_group_check_evidence(drift: str) -> None:
    first, second = _second_target()
    if drift == "changed_detail":
        second["checks"][1]["detail"] = "different assembly-group evidence"  # type: ignore[index]
    else:
        additional_check = {
            "subject": {"kind": "assembly_group", "id": "assembly-01"},
            "check": "additional_group_check",
            "status": "passed",
            "detail": "present for every row or none",
        }
        first["checks"].append(additional_check)  # type: ignore[union-attr]
        if drift == "changed_status":
            second_check = {**additional_check, "status": "not_run"}
            second["checks"].append(second_check)  # type: ignore[union-attr]

    with pytest.raises(
        baserender.SchemaError,
        match="contradictory assembly-group-wide check evidence at row 1",
    ):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


def test_adapter_compares_assembly_group_check_evidence_without_using_row_order() -> None:
    first, second = _second_target()
    for row in (first, second):
        row["search"].update({"locus_count": 2, "barcode_candidates_generated": 50})  # type: ignore[union-attr]
    second["checks"].reverse()  # type: ignore[union-attr]

    records = baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")

    assert [record.id for record in records] == ["target-01", "target-02"]


def test_adapter_rejects_plan_wide_v1_complement_preparation_drift_across_groups() -> None:
    first, second = _second_target()
    for row in (first, second):
        row["source"]["algorithm"] = "dnadesign.junction.string.v1"  # type: ignore[index]
    second["target"]["assembly_group_id"] = "assembly-02"  # type: ignore[index]
    second["search"]["assembly_group_id"] = "assembly-02"  # type: ignore[index]
    second["checks"][1]["subject"]["id"] = "assembly-02"  # type: ignore[index]
    second["geometry"]["junctions"][0]["complement_end_preparation"] = (  # type: ignore[index]
        "downstream_phosphorylation"
    )

    with pytest.raises(
        baserender.SchemaError,
        match="contradictory plan-wide complement-end preparation at row 1",
    ):
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")


def test_adapter_scopes_v1_complement_preparation_to_one_plan() -> None:
    first, second = _second_target()
    for row in (first, second):
        row["source"]["algorithm"] = "dnadesign.junction.string.v1"  # type: ignore[index]
    second["source"]["plan_id"] = f"sha256:{'c' * 64}"  # type: ignore[index]
    second["geometry"]["junctions"][0]["complement_end_preparation"] = (  # type: ignore[index]
        "downstream_phosphorylation"
    )

    records = baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")

    assert [record.id for record in records] == ["target-01", "target-02"]


@pytest.mark.parametrize("contradictory_payload", [False, True])
def test_adapter_rejects_duplicate_target_identity_within_one_plan(
    contradictory_payload: bool,
) -> None:
    first = _payload()
    second = _payload_with_long_recovery_primers() if contradictory_payload else _payload()
    private_target_id = "PRIVATE-TARGET-SENTINEL"
    for payload in (first, second):
        payload["target"]["target_id"] = private_target_id  # type: ignore[index]
        payload["checks"][0]["subject"]["id"] = private_target_id  # type: ignore[index]

    with pytest.raises(
        baserender.SchemaError,
        match="duplicate target identity at row 1",
    ) as exc_info:
        baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")
    assert private_target_id not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_adapter_allows_target_id_reuse_across_distinct_plans() -> None:
    first = _payload()
    second = _payload()
    second["source"]["plan_id"] = f"sha256:{'c' * 64}"  # type: ignore[index]

    records = baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")

    assert [record.id for record in records] == ["target-01", "target-01"]


@pytest.mark.parametrize("distinct_group", ["plan", "assembly_group"])
def test_adapter_allows_distinct_plan_or_assembly_group_search_receipts(distinct_group: str) -> None:
    first, second = _second_target()
    second["search"]["toehold_seed"] = 99  # type: ignore[index]
    if distinct_group == "plan":
        second["source"]["plan_id"] = f"sha256:{'c' * 64}"  # type: ignore[index]
    else:
        second["target"]["assembly_group_id"] = "assembly-02"  # type: ignore[index]
        second["search"]["assembly_group_id"] = "assembly-02"  # type: ignore[index]
        second["checks"][1]["subject"]["id"] = "assembly-02"  # type: ignore[index]

    records = baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")

    assert [record.id for record in records] == ["target-01", "target-02"]
