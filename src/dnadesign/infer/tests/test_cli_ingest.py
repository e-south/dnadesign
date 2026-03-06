"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_cli_ingest.py

Characterization tests for infer CLI ingest request builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.infer.cli_ingest import build_extract_ingest, build_generate_ingest
from dnadesign.infer.errors import ConfigError


def test_build_extract_ingest_prefers_usr_source(tmp_path: Path) -> None:
    request = build_extract_ingest(
        seq=[],
        seq_file=None,
        usr="demo_ds",
        field="sequence",
        ids="id1,id2",
        usr_root=tmp_path,
        pt=None,
        records_jsonl=None,
        i_know_this_is_pickle=False,
        guard_pickle=lambda _flag: None,
    )

    assert request.ingest.source == "usr"
    assert request.ingest.dataset == "demo_ds"
    assert request.ingest.root == tmp_path.as_posix()
    assert request.ingest.ids == ["id1", "id2"]
    assert request.inputs is None


def test_build_extract_ingest_pt_invokes_pickle_guard(tmp_path: Path) -> None:
    seen = {"called_with": None}

    def _guard(flag: bool) -> None:
        seen["called_with"] = flag

    request = build_extract_ingest(
        seq=[],
        seq_file=None,
        usr=None,
        field="sequence",
        ids=None,
        usr_root=None,
        pt=tmp_path / "x.pt",
        records_jsonl=None,
        i_know_this_is_pickle=True,
        guard_pickle=_guard,
    )

    assert seen["called_with"] is True
    assert request.ingest.source == "pt_file"
    assert request.inputs == (tmp_path / "x.pt").as_posix()


def test_build_extract_ingest_requires_source() -> None:
    with pytest.raises(ConfigError, match="Provide one of --seq/--seq-file/--usr/--pt/--records-jsonl"):
        build_extract_ingest(
            seq=[],
            seq_file=None,
            usr=None,
            field="sequence",
            ids=None,
            usr_root=None,
            pt=None,
            records_jsonl=None,
            i_know_this_is_pickle=False,
            guard_pickle=lambda _flag: None,
        )


def test_build_generate_ingest_usr_sets_dataset_contract(tmp_path: Path) -> None:
    request = build_generate_ingest(
        prompt=[],
        prompt_file=None,
        usr="prompt_ds",
        field="sequence",
        ids="a,b",
        usr_root=tmp_path,
    )

    assert request.ingest.source == "usr"
    assert request.ingest.dataset == "prompt_ds"
    assert request.ingest.ids == ["a", "b"]
    assert request.inputs is None
