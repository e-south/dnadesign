"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_missing_manifest_hint.py

Validate manifest-missing hints for interrupted runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.cruncher.artifacts.manifest import load_manifest


def test_missing_run_manifest_hint(tmp_path: Path) -> None:
    run_dir = tmp_path / "sample_run"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "dummy.json").write_text("{}")

    with pytest.raises(FileNotFoundError) as exc:
        load_manifest(run_dir)
    msg = str(exc.value).lower()
    assert "interrupted" in msg
    assert "sample.optimizer.name" in msg
    assert "cruncher sample" in msg
    assert "dummy.json" in msg
