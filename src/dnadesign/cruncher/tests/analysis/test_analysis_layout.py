"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_analysis_layout.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

from dnadesign.cruncher.analysis.layout import list_analysis_entries, list_analysis_entries_verbose


def test_analysis_entries_verbose_marks_unindexed_when_summary_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)

    entries = list_analysis_entries_verbose(run_dir)
    assert entries
    entry = entries[0]
    assert entry["kind"] == "unindexed"
    assert entry["label"] == "analysis (unindexed)"
    assert entry["warnings"]


def test_list_analysis_entries_skips_archive_when_summary_missing_analysis_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    archive_dir = run_dir / "analysis" / "_archive" / "broken-entry"
    archive_dir.mkdir(parents=True)
    (archive_dir / "summary.json").write_text("{}\n")

    entries = list_analysis_entries(run_dir)
    assert entries == []


def test_list_analysis_entries_verbose_skips_archive_when_summary_missing_analysis_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    archive_dir = run_dir / "analysis" / "_archive" / "broken-entry"
    archive_dir.mkdir(parents=True)
    (archive_dir / "summary.json").write_text("{}\n")

    entries = list_analysis_entries_verbose(run_dir)
    assert entries
    assert all(entry.get("kind") != "archive" for entry in entries)
