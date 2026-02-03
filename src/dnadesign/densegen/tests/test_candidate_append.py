from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_candidate_store import write_candidate_records


def _write_candidates(path: Path, records: list[dict]) -> None:
    write_candidate_records(
        records,
        debug_output_dir=path,
        debug_label="demo",
        motif_id="m1",
        motif_hash="hash",
    )


def test_candidate_append_requires_candidate_id(tmp_path: Path) -> None:
    out_dir = tmp_path / "candidates"
    _write_candidates(
        out_dir,
        [
            {
                "candidate_id": "c1",
                "input_name": "demo",
                "sequence": "AAAA",
            }
        ],
    )

    with pytest.raises(ValueError, match="candidate_id"):
        _write_candidates(
            out_dir,
            [
                {
                    "input_name": "demo",
                    "sequence": "CCCC",
                }
            ],
        )


def test_candidate_append_requires_schema_match(tmp_path: Path) -> None:
    out_dir = tmp_path / "candidates"
    _write_candidates(
        out_dir,
        [
            {
                "candidate_id": "c1",
                "input_name": "demo",
                "sequence": "AAAA",
            }
        ],
    )

    with pytest.raises(ValueError, match="schema"):
        _write_candidates(
            out_dir,
            [
                {
                    "candidate_id": "c2",
                    "input_name": "demo",
                    "sequence": "CCCC",
                    "score": 1.0,
                }
            ],
        )
