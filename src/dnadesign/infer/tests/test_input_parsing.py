"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_input_parsing.py

Unit tests for shared infer input parsing helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.infer.input_parsing import load_nonempty_lines, read_ids_arg


def test_read_ids_arg_none() -> None:
    assert read_ids_arg(None) is None


def test_read_ids_arg_csv_text() -> None:
    assert read_ids_arg("id1,id2, id3") == ["id1", "id2", "id3"]


def test_read_ids_arg_file_newlines(tmp_path: Path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("id1\n\nid2\n id3 \n", encoding="utf-8")
    assert read_ids_arg(ids_file.as_posix()) == ["id1", "id2", "id3"]


def test_read_ids_arg_file_csv(tmp_path: Path) -> None:
    ids_file = tmp_path / "ids.csv"
    ids_file.write_text("id1, id2 ,id3", encoding="utf-8")
    assert read_ids_arg(ids_file.as_posix()) == ["id1", "id2", "id3"]


def test_load_nonempty_lines_strips_empty_lines(tmp_path: Path) -> None:
    text_file = tmp_path / "lines.txt"
    text_file.write_text("AA\n\nBB\n CC \n", encoding="utf-8")
    assert load_nonempty_lines(text_file) == ["AA", "BB", "CC"]
