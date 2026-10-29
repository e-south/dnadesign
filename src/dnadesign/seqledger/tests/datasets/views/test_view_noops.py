"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/datasets/views/test_view_noops.py

Repeated view writes do not rewrite sidecars or emit misleading change events.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

import pytest

from dnadesign.seqledger import (
    Dataset,
    SequenceViewRecord,
    ViewSemanticsRecord,
    ensure_sequence_contract_namespaces,
    sequence_views_path,
    view_semantics_path,
    write_sequence_views,
    write_view_semantics,
)


def _fixture(tmp_path):
    ensure_sequence_contract_namespaces(tmp_path)
    dataset = Dataset(tmp_path, "example")
    dataset.init(source="fixture")
    ids = dataset.add_sequences(["ACGT", "GGGG"], bio_type="dna", alphabet="dna_4").ids
    views = [
        SequenceViewRecord(
            sequence_id=row_id,
            product_kind="source_record",
            source_dataset_id=dataset.name,
            orientation="forward",
            created_at="2026-01-01T00:00:00Z",
        )
        for row_id in ids
    ]
    return dataset, views


@pytest.mark.parametrize(
    "kind,policy",
    [
        ("view", "idempotent"),
        ("view", "replace"),
        ("view", "append_alias"),
        ("semantics", "idempotent"),
        ("semantics", "replace"),
    ],
)
def test_exact_repeat_preserves_file_and_events(tmp_path, kind, policy):
    dataset, views = _fixture(tmp_path)
    writer, path, rows = write_sequence_views, sequence_views_path(dataset), views
    if kind == "semantics":
        write_sequence_views(dataset, views)
        writer, path = write_view_semantics, view_semantics_path(dataset)
        rows = [
            ViewSemanticsRecord(view_id=view.view_id, sequence_id=view.sequence_id, created_at=view.created_at)
            for view in views
        ]
    assert writer(dataset, rows, actor={"tool": "fixture", "run_id": "test"}) == 2
    before = path.stat()
    contents = path.read_bytes()
    events = (dataset.dir / ".events.log").read_bytes()
    assert writer(dataset, rows, conflict_policy=policy, actor={"tool": "fixture", "run_id": "test"}) == 0
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert path.read_bytes() == contents
    assert (dataset.dir / ".events.log").read_bytes() == events


def test_mixed_batch_counts_only_changed_views(tmp_path):
    dataset, views = _fixture(tmp_path)
    write_sequence_views(dataset, views[:1])
    assert (
        write_sequence_views(dataset, views, conflict_policy="idempotent", actor={"tool": "fixture", "run_id": "test"})
        == 1
    )
    event = json.loads((dataset.dir / ".events.log").read_text().splitlines()[-1])
    assert event["metrics"]["views_written"] == 1
