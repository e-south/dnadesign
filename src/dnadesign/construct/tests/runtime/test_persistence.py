"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_persistence.py

Construct persistence contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.persistence import records as records_module
from dnadesign.construct.src.persistence.records import (
    BuiltRecord,
    output_records_for_overlay,
    records_to_write,
    require_output_conflict_policy,
    unique_records_by_output_id,
    validate_duplicate_output_aliases,
)
from dnadesign.construct.src.persistence.write_session import sequence_views_to_append
from dnadesign.usr import SequenceViewRecord

_CREATED_AT = "2026-01-01T00:00:00+00:00"


def _record(
    output_id: str,
    *,
    sequence: str = "AAAA",
    metadata: dict[str, object] | None = None,
    label_primary: str | None = None,
    label_aliases: list[str] | None = None,
    sequence_view: SequenceViewRecord | None = None,
) -> BuiltRecord:
    return BuiltRecord(
        output_id=output_id,
        sequence=sequence,
        alphabet="dna_4",
        metadata=metadata or {"id": output_id, "construct__spec_id": "spec"},
        label_primary=label_primary,
        label_aliases=label_aliases or [],
        created_at=_CREATED_AT,
        sequence_view=sequence_view,
    )


def _view(
    *,
    sequence_id: str = "seq_a",
    view_name: str = "anchor_a",
    derivation_spec_id: str = "spec_a",
    recommended_pooling: str = "anchor_mean",
    anchor_start_0: int = 1,
    anchor_end_0: int = 3,
) -> SequenceViewRecord:
    return SequenceViewRecord(
        sequence_id=sequence_id,
        view_name=view_name,
        aliases=["alias_a"],
        product_kind="realized_context",
        context_kind="template_custom",
        orientation="forward",
        source_dataset_id="constructs",
        parent_sequence_id="input_a",
        parent_dataset_id="inputs",
        derivation_id=f"{sequence_id}:{derivation_spec_id}",
        derivation_spec_id=derivation_spec_id,
        anchor_start_0=anchor_start_0,
        anchor_end_0=anchor_end_0,
        recommended_pooling=recommended_pooling,
        created_at=_CREATED_AT,
        created_by="construct",
    )


def _existing_view_payload(view: SequenceViewRecord) -> dict[str, object]:
    payload = view.model_dump(mode="python")
    payload.pop("created_at", None)
    payload.pop("created_by", None)
    return payload


def test_duplicate_output_ids_require_sequence_view_coverage() -> None:
    with pytest.raises(ValidationError, match="duplicate planned output id"):
        validate_duplicate_output_aliases([_record("dup"), _record("dup")])


def test_unique_records_by_output_id_fails_fast_on_sequence_drift() -> None:
    with pytest.raises(ValidationError, match="different sequence payload"):
        unique_records_by_output_id([_record("dup", sequence="AAAA"), _record("dup", sequence="CCCC")])


def test_output_records_for_overlay_excludes_ambiguous_duplicate_overlay_payloads() -> None:
    records = [
        _record("dup", metadata={"id": "dup", "construct__spec_id": "spec_a"}, label_primary="anchor_a"),
        _record("dup", metadata={"id": "dup", "construct__spec_id": "spec_b"}, label_primary="anchor_b"),
    ]

    assert output_records_for_overlay(records) == []


def test_require_output_conflict_policy_reports_unique_existing_collisions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records_module, "_existing_output_ids", lambda _root, _dataset: {"dup"})

    with pytest.raises(ValidationError, match="1 planned output id"):
        require_output_conflict_policy(
            [_record("dup"), _record("dup")],
            output_root=Path("/usr"),
            output_dataset="constructs",
            on_conflict="error",
        )


def test_records_to_write_honors_ignore_conflict_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records_module, "_existing_output_ids", lambda _root, _dataset: {"existing"})

    records = records_to_write(
        [_record("existing"), _record("new")],
        output_root=Path("/usr"),
        output_dataset="constructs",
        on_conflict="ignore",
    )

    assert [record.output_id for record in records] == ["new"]


def test_sequence_views_to_append_fails_fast_on_mutable_metadata_drift() -> None:
    view = _view(recommended_pooling="anchor_mean")
    drifted_existing = _existing_view_payload(view.model_copy(update={"recommended_pooling": "seq_mean"}))

    with pytest.raises(ValidationError, match="different metadata"):
        sequence_views_to_append([view], existing_by_id={str(view.view_id): drifted_existing})


def test_sequence_views_to_append_does_not_treat_same_sequence_as_same_view() -> None:
    view = _view(sequence_id="seq_a")

    assert sequence_views_to_append(
        [view],
        existing_by_id={"other_view": {"view_id": "other_view", "sequence_id": "seq_a"}},
    ) == [view]


def test_sequence_views_to_append_allows_distinct_slot_view_for_existing_sequence() -> None:
    existing = _view(sequence_id="seq_a", anchor_start_0=1, anchor_end_0=3)
    planned = _view(sequence_id="seq_a", derivation_spec_id="spec_b", anchor_start_0=7, anchor_end_0=11)

    assert sequence_views_to_append(
        [planned], existing_by_id={str(existing.view_id): _existing_view_payload(existing)}
    ) == [planned]


def test_sequence_views_to_append_allows_distinct_named_view_with_same_bounds() -> None:
    existing = _view(sequence_id="seq_a", view_name="anchor_a", anchor_start_0=1, anchor_end_0=3)
    planned = _view(
        sequence_id="seq_a",
        view_name="anchor_b",
        derivation_spec_id="spec_b",
        anchor_start_0=1,
        anchor_end_0=3,
    )

    assert sequence_views_to_append(
        [planned], existing_by_id={str(existing.view_id): _existing_view_payload(existing)}
    ) == [planned]
