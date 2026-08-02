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

from .fixtures import _payload, _review_job


def _second_target() -> tuple[dict[str, object], dict[str, object]]:
    first = _payload()
    second = _payload()
    second["target"]["target_id"] = "target-02"  # type: ignore[index]
    second["checks"][0]["subject"]["id"] = "target-02"  # type: ignore[index]
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

    with pytest.raises(baserender.SchemaError, match="contradictory pool-wide search receipt at row 1"):
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


@pytest.mark.parametrize("distinct_group", ["plan", "pool"])
def test_adapter_allows_distinct_plan_or_pool_search_receipts(distinct_group: str) -> None:
    first, second = _second_target()
    second["search"]["toehold_seed"] = 99  # type: ignore[index]
    if distinct_group == "plan":
        second["source"]["plan_id"] = f"sha256:{'c' * 64}"  # type: ignore[index]
    else:
        second["target"]["pool_id"] = "pool-02"  # type: ignore[index]
        second["search"]["pool_id"] = "pool-02"  # type: ignore[index]
        second["checks"][1]["subject"]["id"] = "pool-02"  # type: ignore[index]

    records = baserender.adapt_records([first, second], adapter_kind="three_way_junction_review_v1")

    assert [record.id for record in records] == ["target-01", "target-02"]
