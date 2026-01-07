"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_analysis_layout.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

from dnadesign.cruncher.utils.analysis_layout import list_analysis_entries_verbose


def test_analysis_entries_verbose_marks_unindexed_when_summary_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    (analysis_dir / "plots").mkdir(parents=True)

    entries = list_analysis_entries_verbose(run_dir)
    assert entries
    entry = entries[0]
    assert entry["kind"] == "unindexed"
    assert entry["label"] == "analysis (unindexed)"
    assert entry["warnings"]
