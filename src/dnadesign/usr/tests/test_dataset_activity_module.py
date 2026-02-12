"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_activity_module.py

Module tests for dataset activity helpers that handle meta-note and event IO.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr.src.dataset_activity import append_meta_note, record_dataset_activity_event


def test_append_meta_note_creates_header_and_code_block(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "demo"
    meta_path = dataset_dir / "meta.md"
    timestamp = "2026-02-12T10:00:00+00:00"

    append_meta_note(
        dataset_dir=dataset_dir,
        dataset_name="demo",
        meta_path=meta_path,
        title="ran smoke test",
        code_block="echo hello",
        timestamp_utc=timestamp,
    )

    text = meta_path.read_text(encoding="utf-8")
    assert "name: demo" in text
    assert "- 2026-02-12T10:00:00+00:00: ran smoke test" in text
    assert "```bash" in text
    assert "echo hello" in text


def test_record_dataset_activity_event_writes_jsonl(tmp_path: Path) -> None:
    dataset_root = tmp_path
    records_path = tmp_path / "demo" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"id": ["a"], "sequence": ["ATCG"], "alphabet": ["DNA"], "bio_type": ["dna"]}),
        records_path,
    )

    events_path = tmp_path / "demo" / ".events.log"
    record_dataset_activity_event(
        events_path=events_path,
        action="materialize",
        dataset_name="demo",
        dataset_root=dataset_root,
        records_path=records_path,
        args={"step": "finish"},
    )

    lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["action"] == "materialize"
    assert payload["dataset"]["name"] == "demo"
    assert payload["args"]["step"] == "finish"
